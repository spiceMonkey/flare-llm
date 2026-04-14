
import math
from dataclasses import dataclass
from ..specs.model_spec import LlmModelSpec
from ..specs.system_spec import SystemSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.tuner_spec import TuningSpec
from ..utils import GB_TO_BYTES, TB_TO_FLOPS, US_TO_SECONDS


# ────────────────────────────────────────────────────────────
# Result dataclasses
# ────────────────────────────────────────────────────────────

@dataclass
class PrefillFlopsResults:
    F_proj_prefill: float           # (4H² + 4HH_kv) * S per layer
    F_attn_kv_prefill: float        # 4 * S² * H per layer
    F_ffn_prefill: float            # 6 * H * I_eff * S per layer
    F_layer_prefill: float          # sum per layer (before sharding)
    F_prefill_device: float         # per-device total across L/PP layers


@dataclass
class PrefillTrafficResults:
    T_theta_device: float           # weight read traffic (same as decode)
    T_kv_write_device: float        # KV cache write traffic for S_input tokens
    T_prefill_device: float         # total per-device traffic


@dataclass
class PrefillCommResults:
    t_TP_prefill: float
    t_EP_prefill: float
    t_SP_prefill: float
    t_PP_prefill: float
    t_prefill_comm: float           # total per-stage communication


@dataclass
class PrefillLatencyResults:
    t_prefill_compute: float        # F_prefill_device / R_GPU
    t_prefill_mem: float            # T_prefill_device / B_eff_mem
    t_prefill_local: float          # max(compute, mem)
    t_prefill_comm: float
    t_pipeline_warmup: float        # (PP-1) * t_stage
    t_prefill: float                # full hardware prefill latency

    # Batched prefill
    B_prefill: int
    t_prefill_batched: float        # t_prefill with B_prefill scaling

    # Chunked prefill
    chunk_size: int                 # C (0 = no chunking)
    n_chunks: int
    t_prefill_chunked: float        # sum of k-dependent chunk latencies


# ────────────────────────────────────────────────────────────
# Prefill FLOPs (documentation/modeling/prefill.md §1)
# ────────────────────────────────────────────────────────────

def compute_prefill_flops(
    model: LlmModelSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
) -> PrefillFlopsResults:
    """Per-device prefill FLOPs. Doc: documentation/modeling/prefill.md §1.5"""

    L = model.L
    H = model.H
    H_kv = model.H_kv()
    S = tuner.S_input
    PP = partition.PP
    TP = partition.TP
    EP = max(1, partition.EP)
    SP = partition.SP

    # Effective FFN dim (MoE: k_active * I_moe; dense: I_dense)
    if model.moe is not None:
        L_moe = model.moe.n_moe_layers if model.moe.n_moe_layers else L
        L_dense = L - L_moe
        EP = min(EP, max(1, model.moe.n_experts))
        k = model.moe.k_active
        I_moe = model.moe.I_moe
    else:
        L_moe = 0
        L_dense = L
        k = 0
        I_moe = 0

    I_dense = model.I_dense

    # Per-layer unsharded FLOPs
    F_proj = (4 * H**2 + 4 * H * H_kv) * S
    F_attn = 4 * S**2 * H
    F_ffn_dense = 6 * H * I_dense * S
    F_ffn_moe = 6 * H * k * I_moe * S

    F_layer_dense = F_proj + F_attn + F_ffn_dense
    F_layer_moe = F_proj + F_attn + F_ffn_moe

    # Per-device with parallelism sharding
    F_device_dense = (L_dense / PP) * (
        F_proj / TP + F_attn / (TP * SP) + F_ffn_dense / TP
    )
    F_device_moe = (L_moe / PP) * (
        F_proj / TP + F_attn / (TP * SP) + F_ffn_moe / (TP * EP)
    )

    F_prefill_device = F_device_dense + F_device_moe

    F_layer_prefill = F_layer_dense  # representative (dense) for diagnostics

    return PrefillFlopsResults(
        F_proj_prefill=F_proj,
        F_attn_kv_prefill=F_attn,
        F_ffn_prefill=F_ffn_dense,
        F_layer_prefill=F_layer_prefill,
        F_prefill_device=F_prefill_device,
    )


# ────────────────────────────────────────────────────────────
# Prefill Traffic (documentation/modeling/prefill.md §3.1)
# ────────────────────────────────────────────────────────────

def compute_prefill_traffic(
    model: LlmModelSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
) -> PrefillTrafficResults:
    """Per-device HBM traffic for prefill pass."""

    L = model.L
    H = model.H
    H_kv = model.H_kv()
    S = tuner.S_input
    b = model.bytes_per_param
    PP = partition.PP
    TP = partition.TP
    EP = max(1, partition.EP)
    SP = partition.SP

    if model.moe is not None:
        L_moe = model.moe.n_moe_layers if model.moe.n_moe_layers else L
        L_dense = L - L_moe
        N_exp = max(1, model.moe.n_experts)
        EP = min(EP, N_exp)
        I_moe = model.moe.I_moe
    else:
        L_moe = 0
        L_dense = L
        N_exp = 1
        I_moe = 0

    I_dense = model.I_dense

    # Weight read traffic (same as decode — weights loaded once)
    T_theta_dense = (L_dense / PP) * (
        (2 * H**2 + 2 * H * H_kv) / TP + (3 * H * I_dense) / TP
    ) * b
    T_theta_moe = (L_moe / PP) * (
        (2 * H**2 + 2 * H * H_kv) / TP + (3 * H * I_moe * N_exp) / (TP * EP)
    ) * b
    T_theta_device = T_theta_dense + T_theta_moe

    # KV cache write traffic: writing S_input KV entries
    T_kv_write_device = (L / PP) * (2 * S * H_kv * b) / (TP * SP)

    T_prefill_device = T_theta_device + T_kv_write_device

    return PrefillTrafficResults(
        T_theta_device=T_theta_device,
        T_kv_write_device=T_kv_write_device,
        T_prefill_device=T_prefill_device,
    )


# ────────────────────────────────────────────────────────────
# Prefill Communication (documentation/modeling/prefill.md §3.2)
# ────────────────────────────────────────────────────────────

def compute_prefill_comm(
    model: LlmModelSpec,
    system: SystemSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
) -> PrefillCommResults:
    """Per-stage communication time for prefill (S_input-scaled messages)."""

    H = model.H
    H_kv = model.H_kv()
    L = model.L
    S = tuner.S_input
    b = model.bytes_per_param
    PP = partition.PP
    TP = partition.TP
    EP = max(1, partition.EP)
    SP = partition.SP

    n_TP = tuner.n_TP_collectives
    n_EP = tuner.n_EP_collectives
    n_SP = tuner.n_SP_collectives

    dom_PP = system.get_domain("PP")
    dom_TP = system.get_domain("TP")
    dom_EP = system.get_domain("EP")
    dom_SP = system.get_domain("SP")

    a_PP = dom_PP.alpha_us * US_TO_SECONDS
    a_TP = dom_TP.alpha_us * US_TO_SECONDS
    a_EP = dom_EP.alpha_us * US_TO_SECONDS
    a_SP = dom_SP.alpha_us * US_TO_SECONDS

    B_PP = dom_PP.bandwidth_GBps * GB_TO_BYTES
    B_TP = dom_TP.bandwidth_GBps * GB_TO_BYTES
    B_EP = dom_EP.bandwidth_GBps * GB_TO_BYTES
    B_SP = dom_SP.bandwidth_GBps * GB_TO_BYTES

    tp_algorithm = getattr(tuner, "tp_algorithm", "ring").lower()
    ep_algorithm = getattr(tuner, "ep_algorithm", "ring").lower()

    # PP: S_input-scaled activation hop
    msg_PP = (S * H / TP) * b
    t_PP = a_PP + msg_PP / B_PP if PP > 1 else 0.0

    # TP: S_input-scaled all-reduce
    if TP > 1:
        msg_TP = S * H * b
        if tp_algorithm == "ring":
            t_TP = 2 * (TP - 1) * a_TP + 2 * ((TP - 1) / TP) * (msg_TP / B_TP)
        else:
            t_TP = 2 * math.ceil(math.log2(TP)) * a_TP + 2 * (msg_TP / B_TP)
    else:
        t_TP = 0.0

    # EP
    if model.moe is not None:
        N_exp = max(1, model.moe.n_experts)
        EP = min(EP, N_exp)
        k_active = model.moe.k_active
    else:
        EP = 1
        k_active = 1
    if EP > 1:
        msg_EP = k_active * S * H * b
        if ep_algorithm == "ring":
            t_EP = 2 * (EP - 1) * a_EP + 2 * (EP - 1) * (msg_EP / (EP * B_EP))
        else:
            t_EP = 2 * math.ceil(math.log2(EP)) * a_EP + 2 * (msg_EP / B_EP)
    else:
        t_EP = 0.0

    # SP: KV all-gather (S_input-scaled)
    if SP > 1:
        msg_SP = (S / SP) * (2 * H_kv / TP) * b
        t_SP = (SP - 1) * a_SP + (SP - 1) * (msg_SP / B_SP)
    else:
        t_SP = 0.0

    # MoE layer count
    L_moe = model.moe.n_moe_layers if (model.moe and model.moe.n_moe_layers) else (L if model.moe else 0)

    t_prefill_comm = (
        (L / PP) * (n_TP * t_TP + n_SP * t_SP)
        + (L_moe / PP) * (n_EP * t_EP)
        + t_PP
    )

    return PrefillCommResults(
        t_TP_prefill=t_TP,
        t_EP_prefill=t_EP,
        t_SP_prefill=t_SP,
        t_PP_prefill=t_PP,
        t_prefill_comm=t_prefill_comm,
    )


# ────────────────────────────────────────────────────────────
# Prefill Latency (documentation/modeling/prefill.md §3-5)
# ────────────────────────────────────────────────────────────

def compute_prefill_latency(
    system: SystemSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
    model: LlmModelSpec,
    flops: PrefillFlopsResults,
    traffic: PrefillTrafficResults,
    comm: PrefillCommResults,
) -> PrefillLatencyResults:
    """Hardware prefill latency: single-request, batched, and chunked."""

    R_gpu = system.device.peak_flops_TF * TB_TO_FLOPS
    B_eff_mem = system.device.hbm_bandwidth_GBps * GB_TO_BYTES

    PP = partition.PP
    TP = partition.TP
    EP = max(1, partition.EP)
    SP = partition.SP
    rho = tuner.overlap_factor

    S = tuner.S_input
    B_pf = tuner.B_prefill
    C = tuner.chunk_size

    H = model.H
    H_kv = model.H_kv()
    L = model.L

    # ── Single-request prefill (§3) ──────────────────────

    t_prefill_compute = flops.F_prefill_device / R_gpu
    t_prefill_mem = traffic.T_prefill_device / B_eff_mem
    t_prefill_local = max(t_prefill_compute, t_prefill_mem)
    t_prefill_comm = comm.t_prefill_comm

    # Pipeline warmup: (PP-1) stages must fill before first token emerges
    # Each stage takes approximately t_prefill_local (prefill is typically compute-bound)
    t_pipeline_warmup = (PP - 1) * t_prefill_local if PP > 1 else 0.0

    t_prefill = (
        t_prefill_local
        + max(0.0, t_prefill_comm - rho * t_prefill_local)
        + t_pipeline_warmup
    )

    # ── Batched prefill (§4) ─────────────────────────────

    if B_pf > 1:
        # FLOPs scale linearly with B_prefill
        t_batched_compute = B_pf * flops.F_prefill_device / R_gpu
        # Traffic: weights loaded once + B_pf * KV writes
        T_batched = traffic.T_theta_device + B_pf * traffic.T_kv_write_device
        t_batched_mem = T_batched / B_eff_mem
        t_batched_local = max(t_batched_compute, t_batched_mem)
        t_prefill_batched = (
            t_batched_local
            + max(0.0, t_prefill_comm - rho * t_batched_local)
            + t_pipeline_warmup
        )
    else:
        t_prefill_batched = t_prefill

    # ── Chunked prefill (§5) ─────────────────────────────

    if C > 0 and S > 0:
        n_chunks = math.ceil(S / C)

        # Effective FFN dim for linear FLOPs
        if model.moe is not None:
            I_eff = model.moe.k_active * model.moe.I_moe
        else:
            I_eff = model.I_dense

        # Linear FLOPs per chunk (constant across chunks)
        F_linear_per_chunk = (L / PP) * (
            (4 * H**2 + 4 * H * H_kv) * C / TP
            + 6 * H * I_eff * C / (TP * EP)
        )

        # Weight traffic per chunk (same each chunk — full weight read)
        T_theta_chunk = traffic.T_theta_device
        # KV write per chunk: C new entries
        T_kv_write_chunk = (L / PP) * (2 * C * H_kv * model.bytes_per_param) / (TP * SP)

        # Chunk-level comm (approximately constant, S_input → C)
        # Simplified: scale comm proportionally to C/S
        t_chunk_comm = comm.t_prefill_comm * (C / S) if S > 0 else 0.0

        total_chunked = 0.0
        for k in range(1, n_chunks + 1):
            # Attention FLOPs for chunk k: attends to kC accumulated KV positions
            F_attn_chunk_k = (L / PP) * (4 * C * k * C * H) / (TP * SP)
            F_chunk_k = F_linear_per_chunk + F_attn_chunk_k

            t_chunk_compute_k = F_chunk_k / R_gpu

            # Memory: weights + KV write + KV read (kC entries for attention)
            T_kv_read_k = (L / PP) * (2 * k * C * H_kv * model.bytes_per_param) / (TP * SP)
            T_chunk_k = T_theta_chunk + T_kv_write_chunk + T_kv_read_k
            t_chunk_mem_k = T_chunk_k / B_eff_mem

            t_chunk_local_k = max(t_chunk_compute_k, t_chunk_mem_k)
            t_chunk_k = t_chunk_local_k + max(0.0, t_chunk_comm - rho * t_chunk_local_k)
            total_chunked += t_chunk_k

        t_prefill_chunked = total_chunked
    else:
        n_chunks = 0
        t_prefill_chunked = t_prefill  # no chunking → same as unchunked

    return PrefillLatencyResults(
        t_prefill_compute=t_prefill_compute,
        t_prefill_mem=t_prefill_mem,
        t_prefill_local=t_prefill_local,
        t_prefill_comm=t_prefill_comm,
        t_pipeline_warmup=t_pipeline_warmup,
        t_prefill=t_prefill,
        B_prefill=B_pf,
        t_prefill_batched=t_prefill_batched,
        chunk_size=C,
        n_chunks=n_chunks,
        t_prefill_chunked=t_prefill_chunked,
    )
