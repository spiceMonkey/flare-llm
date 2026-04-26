
import math
from dataclasses import dataclass
from ..specs.model_spec import LlmModelSpec
from ..specs.system_spec import SystemSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.tuner_spec import TuningSpec
from ..utils import GB_TO_BYTES, TB_TO_FLOPS
from .primitives import (
    dense_weight_bytes,
    moe_weight_bytes,
    kv_bytes_per_seq,
    linear_flops_per_token,
    aggregate_per_stage,
    cost_collective,
)


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
    """Per-device prefill FLOPs. Doc: documentation/modeling/prefill.md §1.5

    Linear contribution (proj + FFN + router) comes from the shared
    `linear_flops_per_token` primitive scaled by S_input. Attention is
    phase-specific (S² scaling) and stays inline.
    """

    L = model.L
    H = model.H
    H_kv = model.H_kv()
    S = tuner.S_input
    PP = partition.PP
    TP = partition.TP
    SP = partition.SP
    I_dense = model.I_dense

    # Linear FLOPs for the full pass: per-token contribution × S_input tokens
    F_linear_device = linear_flops_per_token(model, partition) * S

    # Attention FLOPs: 4·S²·H per layer, sharded by TP·SP
    F_attn_device = (L / PP) * (4 * S**2 * H) / (TP * SP)

    F_prefill_device = F_linear_device + F_attn_device

    # Unsharded diagnostic values (per layer, representative dense)
    F_proj = (4 * H**2 + 4 * H * H_kv) * S
    F_attn = 4 * S**2 * H
    F_ffn_dense = 6 * H * I_dense * S
    F_layer_prefill = F_proj + F_attn + F_ffn_dense

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

    S = tuner.S_input

    # Weight read traffic (same as decode — weights loaded once per pass)
    T_theta_device = (
        dense_weight_bytes(model, partition) + moe_weight_bytes(model, partition)
    )

    # KV cache write traffic: writing S_input KV entries for one sequence
    T_kv_write_device = kv_bytes_per_seq(model, partition, S)

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
    *,
    tokens_per_step: int | None = None,
) -> PrefillCommResults:
    """Per-stage communication time for prefill (S_input-scaled messages).

    `tokens_per_step` lets callers evaluate the collectives at a token
    count other than `tuner.S_input`. Defaults to single-request behavior
    (tokens = S_input). The latency path passes explicit values for
    batched (B_prefill · S_input) and chunked-per-chunk (C) cases.
    """

    H = model.H
    H_kv = model.H_kv()
    L = model.L
    S = tuner.S_input
    tokens = S if tokens_per_step is None else max(0, int(tokens_per_step))
    b = model.bytes_per_param
    PP = partition.PP
    TP = partition.TP
    EP = max(1, partition.EP)
    SP = partition.SP

    n_TP = tuner.n_TP_collectives
    n_EP = tuner.n_EP_collectives
    n_SP = tuner.n_SP_collectives

    # Resolve EP group size up front so the dispatcher sees the correct radix.
    if model.moe is not None:
        N_exp = max(1, model.moe.n_experts)
        EP = min(EP, N_exp)
        k_active = model.moe.k_active
    else:
        EP = 1
        k_active = 1

    # Prefill reads the per-phase fields; falls back to legacy single-knob.
    tp_algorithm = getattr(tuner, "tp_algorithm_prefill",
                           getattr(tuner, "tp_algorithm", "ring")).lower()
    ep_algorithm = getattr(tuner, "ep_algorithm_prefill",
                           getattr(tuner, "ep_algorithm", "ring")).lower()
    torus_alg = getattr(tuner, "torus_algorithm", "ring").lower()
    inc_enabled = bool(getattr(tuner, "inc_enabled", True))

    if tp_algorithm == "auto" or ep_algorithm == "auto":
        raise ValueError(
            "TuningSpec has algorithm='auto' for prefill; resolve via "
            "core.collective_algo_opt.optimize_collective_algorithms(...) "
            "before InferenceCalculator.run()."
        )

    def _cost(coll: str, op: str, M: float, G: int, alg: str = "ring") -> float:
        return cost_collective(
            system.get_tier_chain(coll), op, M, G,
            algorithm=alg, torus_algorithm=torus_alg,
            inc_enabled=inc_enabled,
        )

    # PP: token-scaled activation hop
    if PP > 1:
        msg_PP = (tokens * H / TP) * b
        t_PP = _cost("PP", "p2p", msg_PP, 2)
    else:
        t_PP = 0.0

    # TP: token-scaled all-reduce
    if TP > 1:
        msg_TP = tokens * H * b
        t_TP = _cost("TP", "all_reduce", msg_TP, TP, alg=tp_algorithm)
    else:
        t_TP = 0.0

    # EP: MoE all-to-all
    if EP > 1:
        msg_EP = k_active * tokens * H * b
        t_EP = _cost("EP", "moe_a2a", msg_EP, EP, alg=ep_algorithm)
    else:
        t_EP = 0.0

    # SP: KV all-gather (token-scaled)
    if SP > 1:
        msg_SP = (tokens / SP) * (2 * H_kv / TP) * b
        t_SP = _cost("SP", "all_gather", msg_SP, SP)
    else:
        t_SP = 0.0

    # MoE layer count
    L_moe = model.moe.n_moe_layers if (model.moe and model.moe.n_moe_layers) else (L if model.moe else 0)

    t_prefill_comm = aggregate_per_stage(
        L=L, L_moe=L_moe, PP=PP,
        n_TP=n_TP, t_TP=t_TP,
        n_SP=n_SP, t_SP=t_SP,
        n_EP=n_EP, t_EP=t_EP,
        t_PP=t_PP,
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
        # Comm scales with the batched token count (B_pf · S), not S alone:
        # collective messages carry per-step activations whose payload grows
        # with tokens per step. α is unchanged; β (payload/BW) grows with B_pf.
        comm_batched = compute_prefill_comm(
            model, system, partition, tuner, tokens_per_step=B_pf * S,
        )
        t_prefill_batched = (
            t_batched_local
            + max(0.0, comm_batched.t_prefill_comm - rho * t_batched_local)
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

        # Chunk-level comm: evaluate collectives at C tokens per step.
        # α (latency term) is unchanged per chunk; β (payload) scales with C,
        # not with C/S. Linear C/S scaling underestimates α-dominated small-C.
        comm_chunk = compute_prefill_comm(
            model, system, partition, tuner, tokens_per_step=C,
        )
        t_chunk_comm = comm_chunk.t_prefill_comm

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
