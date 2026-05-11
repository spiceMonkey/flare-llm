
import math
from dataclasses import dataclass
from typing import Optional
from ..specs.framework_spec import FrameworkSpec
from ..specs.model_spec import LlmModelSpec
from ..specs.system_spec import SystemSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.tuner_spec import TuningSpec
from ..utils import GB_TO_BYTES, TB_TO_FLOPS
from .decode_model import _eta_TC_at_mb, _eta_beta_at_B, effective_peak_flops_TF
from .memory_placement import resolve_placement, t_mem_from_placement
from .primitives import (
    dense_weight_bytes,
    moe_weight_bytes,
    kv_bytes_per_seq,
    linear_flops_per_token,
    aggregate_per_stage,
    cost_collective,
    p2p_hop,
    assign_tier_per_axis,
    tier_at,
    D_attn,
    D_exp,
    D_kv,
    D_emb,
    G_TP,
    G_EP,
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
    t_prefill_compute: float        # raw F_prefill_device / R_GPU
    t_prefill_compute_eff: float    # Tensor-Core-derated: t_compute / η_TC
    eta_TC: float                   # Tensor Core efficiency at this prefill payload
    t_prefill_mem: float            # multi-tier sum, sram.md §2.1
    t_prefill_local: float          # max(compute_eff, mem)
    t_prefill_comm: float
    t_SW_per_stage: float           # per-stage CPU dispatch budget = (L/PP) · k · τ_launch
    t_pipeline_warmup: float        # (PP-1) * t_stage
    t_LM_prefill: float             # LM head one-shot on stage PP-1 (prefill.md §3.4)
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
    framework: FrameworkSpec,
) -> PrefillFlopsResults:
    """Per-device prefill FLOPs. Doc: documentation/modeling/prefill.md §1.5

    Linear contribution (proj + FFN + router) comes from the shared
    `linear_flops_per_token` primitive scaled by S_input. Attention is
    phase-specific (S² scaling) and stays inline.
    """

    fw = framework

    L = model.L
    H = model.H
    S = tuner.S_input
    PP = partition.PP
    SP = partition.SP
    I_dense = model.I_dense

    # Linear FLOPs for the full pass: per-token contribution × S_input tokens
    F_linear_device = linear_flops_per_token(model, partition, fw) * S

    # Attention FLOPs (score + value, per pass: per-token form scaled by S
    # query positions). GQA / MHA: 4·S²·H per layer / (D_kv·SP). MLA:
    # variant-specific score / value per token per layer (mode-dependent)
    # × S query positions / SP.
    if model.mla is not None:
        from .primitives.mla_flops import mla_score_value_flops_per_layer_per_device
        F_sv_per_layer_per_device_per_token = (
            mla_score_value_flops_per_layer_per_device(model, partition, fw, S, fw.mla_mode)
        )
        F_attn_device = (L / PP) * F_sv_per_layer_per_device_per_token * S / SP
    else:
        F_attn_device = (L / PP) * (4 * S**2 * H) / (D_kv(partition, fw) * SP)

    F_prefill_device = F_linear_device + F_attn_device

    # Unsharded diagnostic values (per layer, representative dense). For MLA
    # models the projection cost is variant-specific; we report the
    # mode-dependent MLA proj-per-token × S for the F_proj diagnostic, and
    # the MLA score / value cost × S for the F_attn diagnostic. For GQA / MHA
    # these collapse to the historical (4H²+4HH_kv)·S and 4S²H forms.
    if model.mla is not None:
        from .primitives.mla_flops import (
            mla_proj_flops_per_layer_per_device,
            mla_score_value_flops_per_layer_per_device,
        )
        # Diagnostic forms: cluster-total per-layer (no D_attn / D_kv divisor)
        # — synthesize by computing the per-device value at TP=1 / DP-attn,
        # which gives the unsharded form. Keep it simple: use per-device with
        # current partition (not strictly "unsharded" but matches the post-
        # MLA convention of "what the partition produces").
        F_proj_per_token = mla_proj_flops_per_layer_per_device(model, partition, fw, fw.mla_mode)
        F_proj = F_proj_per_token * S
        F_attn = mla_score_value_flops_per_layer_per_device(
            model, partition, fw, S, fw.mla_mode
        ) * S
    else:
        H_kv = model.H_kv()
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
    framework: FrameworkSpec,
) -> PrefillTrafficResults:
    """Per-device HBM traffic for prefill pass."""

    S = tuner.S_input

    # Weight read traffic (same as decode — weights loaded once per pass)
    T_theta_device = (
        dense_weight_bytes(model, partition, framework)
        + moe_weight_bytes(model, partition, framework)
    )

    # KV cache write traffic: writing S_input KV entries for one sequence
    T_kv_write_device = kv_bytes_per_seq(model, partition, framework, S)

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
    framework: FrameworkSpec,
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
    SP = partition.SP

    # Collective group sizes (notation.md §1; equal to TP and EP across all
    # three production-relevant configurations, threaded via helpers for
    # consistency with the abstract divisor symbols D_kv / D_emb).
    g_TP = G_TP(partition)
    g_EP = G_EP(partition)

    fw = framework
    n_TP = fw.n_TP_collectives
    n_EP = fw.n_EP_collectives
    n_SP = fw.n_SP_collectives

    # Resolve EP group size up front so the dispatcher sees the correct radix.
    if model.moe is not None:
        N_exp = max(1, model.moe.n_experts)
        g_EP = min(g_EP, N_exp)
        k_active = model.moe.k_active
    else:
        g_EP = 1
        k_active = 1

    # Algorithm selection lives on FrameworkSpec (Phase E). Prefill reads
    # the per-phase fields directly; "auto" must have been resolved by
    # `optimize_collective_algorithms` upstream.
    tp_algorithm = fw.tp_algorithm_prefill.lower()
    ep_algorithm = fw.ep_algorithm_prefill.lower()
    torus_alg = fw.torus_algorithm.lower()
    inc_enabled = fw.inc_enabled

    if tp_algorithm == "auto" or ep_algorithm == "auto":
        raise ValueError(
            "FrameworkSpec has algorithm='auto' for prefill; resolve via "
            "core.collective_algo_opt.optimize_collective_algorithms(...) "
            "before InferenceCalculator.run()."
        )

    def _cost(coll: str, op: str, M: float, G: int, alg: str = "ring") -> float:
        return cost_collective(
            system.get_tier_chain(coll), op, M, G,
            algorithm=alg, torus_algorithm=torus_alg,
            inc_enabled=inc_enabled,
        )

    # PP: token-scaled activation hop. Per-rank payload uses H/D_kv (matches
    # decode.md §5.1 and prefill.md §3.2). Cost at the *correct* fabric tier
    # under nested-layout rule (see decode_model.compute_comm for the full
    # rationale and partition_layout.py for the helper).
    d_kv = D_kv(partition, fw)
    if PP > 1:
        msg_PP = (tokens * H / d_kv) * b
        pp_tier_idx = assign_tier_per_axis(partition, fw, system, role="PP")["PP"]
        pp_tier = tier_at(system, "PP", pp_tier_idx)
        t_PP = p2p_hop(msg_PP, pp_tier.alpha_us * 1e-6, pp_tier.bw_per_port_GBps * 1e9)
    else:
        t_PP = 0.0

    # TP: token-scaled all-reduce (or all-gather under DP-attention for the
    # attention block — see DP-attn swap below)
    if g_TP > 1:
        msg_TP = tokens * H * b
        t_TP = _cost("TP", "all_reduce", msg_TP, g_TP, alg=tp_algorithm)
    else:
        t_TP = 0.0
        msg_TP = 0.0

    # DP-attention swap (decode.md §5.3 + prefill.md §3.2): under
    # framework.attention_mode="dp" the per-layer attention TP all-reduce is
    # replaced by a single TP all-gather at the attention → FFN transition.
    # The FFN's TP all-reduce remains. Per-stage adjustment after
    # aggregate_per_stage.
    if g_TP > 1 and fw.attention_mode == "dp":
        t_TP_AG = _cost("TP", "all_gather", msg_TP, g_TP)
    else:
        t_TP_AG = 0.0

    # EP: MoE all-to-all. Per-rank dispatch payload depends on the MoE A2A
    # data-flow pattern (prefill.md §3.2; same two patterns as decode.md §5.2).
    #   "gather" (default) — full per-step tokens per rank
    #   "scatter" + DP-attn — tokens / G_TP per rank (~G_TP× reduction)
    scatter_direct = (
        fw.moe_a2a_pattern == "scatter"
        and fw.attention_mode == "dp"
        and g_TP > 1
    )
    if g_EP > 1:
        if scatter_direct:
            msg_EP = k_active * (tokens / g_TP) * H * b
        else:
            msg_EP = k_active * tokens * H * b
        t_EP = _cost("EP", "moe_a2a", msg_EP, g_EP, alg=ep_algorithm)
    else:
        t_EP = 0.0

    # SP: KV all-gather (token-scaled). Per-rank gathered output convention
    # (collective_cost.py §6: M = G·shard). Uses D_kv for the head/seq divisor
    # (matches prefill.md §3.2 / decode.md §5.4).
    if SP > 1:
        msg_SP = tokens * (2 * H_kv / d_kv) * b
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

    # DP-attention adjustment: replace one of the n_TP per-layer all-reduces
    # (the attention output AR) with the cheaper AG.
    if t_TP_AG > 0.0 and t_TP > 0.0:
        t_prefill_comm += (L / PP) * (t_TP_AG - t_TP)

    # Scatter-direct (prefill.md §3.2): MoE layers fire neither the pre-MoE
    # TP all-gather nor the post-MoE TP all-reduce under DP-attn + scatter.
    if scatter_direct and L_moe > 0:
        per_moe_layer_tp = (n_TP - 1) * t_TP + t_TP_AG
        t_prefill_comm -= (L_moe / PP) * per_moe_layer_tp

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
    framework: FrameworkSpec,
) -> PrefillLatencyResults:
    """Hardware prefill latency: single-request, batched, and chunked."""

    fw = framework

    # Precision-aware compute peak (see decode_model.effective_peak_flops_TF):
    # peak_flops_TF in system spec is FP16 dense per chip; the working
    # precision peak scales linearly with bytes_per_param.
    R_gpu = effective_peak_flops_TF(system, model.bytes_per_param) * TB_TO_FLOPS
    tiers = system.device.get_tiers()

    PP = partition.PP
    SP = partition.SP
    rho = fw.comm_overlap_factor
    rho_SW = fw.sw_overlap_factor

    # Collective group sizes (notation.md §1) for kernel-launch SW count
    # axis-presence guards. Equal to TP and EP across all three production
    # configurations; threaded via helpers for consistency with the abstract
    # divisor symbols D_kv / D_emb.
    g_TP_pf = G_TP(partition)
    g_EP_pf = G_EP(partition)

    # Abstract sharding factors (notation.md §1) for the LM head and chunked
    # prefill formulas. d_kv composes with SP for KV-attention divisors.
    d_attn_pf = D_attn(partition, fw)
    d_kv_pf = D_kv(partition, fw)
    d_emb_pf = D_emb(partition)
    if model.moe is not None:
        d_exp_pf_moe = D_exp(partition, fw, layer_kind="moe", n_exp_cap=max(1, model.moe.n_experts))
    else:
        d_exp_pf_moe = D_exp(partition, fw, layer_kind="dense")
    d_exp_pf_dense = D_exp(partition, fw, layer_kind="dense")

    S = tuner.S_input
    B_pf = tuner.B_prefill
    C = tuner.chunk_size

    H = model.H
    H_kv = model.H_kv()
    L = model.L

    # SW dispatch budget per stage. Prefill is one forward pass per request
    # (or per chunk) — there is no microbatch round structure as in decode,
    # so the per-stage formula is L/PP layers worth of launches plus one
    # PP boundary's worth of P2P sends/recvs (inert when PP = 1):
    #     t_SW_per_stage = (L/PP) · k · τ_launch  +  k_pp_hop · τ_launch
    # where k decomposes into compute + collective kernel counts and the
    # collective counts are zero on axes where the parallelism is 1.
    # The PP-hop term uses the middle-stage 2× factor (recv + send) by
    # default; edge stages do only one direction (off by one k_pp_hop·τ,
    # negligible at PP >> 1).
    n_TP_calls = fw.n_TP_collectives if g_TP_pf > 1 else 0
    # n_EP_collectives counts NCCL API calls directly (dispatch + combine
    # = 2 per MoE layer); see decode_model._t_SW_per_microbatch. EP launches
    # only fire on MoE layers — split the layer term into dense + MoE
    # contributions, matching the L_moe/PP factor in §5.5's t_comm formula.
    n_EP_calls = fw.n_EP_collectives if g_EP_pf > 1 else 0
    n_SP_calls = fw.n_SP_collectives if SP > 1 else 0
    if model.moe is not None:
        L_moe_total = model.moe.n_moe_layers if model.moe.n_moe_layers else L
    else:
        L_moe_total = 0
    k_dense = fw.kernels_per_layer_compute + fw.kernels_per_collective_call * (n_TP_calls + n_SP_calls)
    k_moe_extra = fw.kernels_per_collective_call * n_EP_calls
    layers_per_stage = L / PP if PP > 0 else L
    moe_layers_per_stage = L_moe_total / PP if PP > 0 else L_moe_total
    k_pp_hop = fw.kernels_per_pp_hop if PP > 1 else 0
    t_SW_per_stage = (
        layers_per_stage * k_dense * fw.kernel_launch_us * 1e-6
        + moe_layers_per_stage * k_moe_extra * fw.kernel_launch_us * 1e-6
        + k_pp_hop * fw.kernel_launch_us * 1e-6
    )

    def _compose_SW(t_local_gpu: float) -> float:
        """Compose per-stage GPU work with SW dispatch via ρ_SW.

        Base + unhidden-overflow form (same pattern as compute/comm overlap
        in decode.md §6.2). GPU work is the base; SW dispatch overlaps for
        ρ_SW · t_local_gpu; any remainder serializes after.

        ρ_SW = 1 → t_local_gpu + max(0, t_SW - t_local_gpu) = max(t_local_gpu, t_SW)
        ρ_SW = 0 → t_local_gpu + t_SW_per_stage (no overlap)
        """
        return t_local_gpu + max(0.0, t_SW_per_stage - rho_SW * t_local_gpu)

    # Per-tier memory time helper. T_kv_write_device is per-request; the
    # placement layer treats it the same as decode's T_KV (sram.md §1.3
    # "T_KV,i is per-request bytes"). For chunked / batched prefill the same
    # helper is reused with B = batch count and the chunk's KV term.
    def _t_mem(T_theta_total: float, T_kv_per_req: float, B: int) -> float:
        plc = resolve_placement(
            T_theta_device=T_theta_total,
            T_kv_per_request_device=T_kv_per_req,
            B=max(1, B),
            tiers=tiers,
            placement=tuner.placement,
        )
        eta_beta_B = _eta_beta_at_B(system.device.bw_efficiency, max(1, B))
        return t_mem_from_placement(
            plc, B=max(1, B), tiers=tiers,
            eta_beta_curve_factor=eta_beta_B,
        )

    # ── Single-request prefill (§3) ──────────────────────

    # LM head one-shot on stage PP-1 (prefill.md §1.5 / §3.4):
    #   F_LM = 2·B_pf·H·V/TP — only the last position per request
    #   T_LM = HVb/TP (TP-sharded weights) + B_pf·V·b (logit rows)
    # Added outside warmup since the LM head fires once at the end of the
    # prefill traversal (after the pipeline is filled), not per stage.
    # For chunked prefill it fires once after the last chunk (same one-shot).
    BW_top = tiers[0].bandwidth_GBps * tiers[0].eta_beta * GB_TO_BYTES
    V = model.vocab_size
    b = model.bytes_per_param
    B_pf_eff = max(1, B_pf)

    def _t_LM(B_eff: int) -> float:
        F_lm = 2.0 * B_eff * H * V / max(1, d_emb_pf)
        T_lm = (H * V * b) / max(1, d_emb_pf) + B_eff * V * b
        t_c = F_lm / R_gpu if R_gpu > 0 else 0.0
        t_m = T_lm / BW_top if BW_top > 0 else 0.0
        return max(t_c, t_m)

    t_LM_prefill = _t_LM(B_pf_eff)

    t_prefill_compute = flops.F_prefill_device / R_gpu
    # η_TC at prefill payload (mb_eff = B_prefill · S / PP). With typical
    # S ≥ 256 this saturates to 1.0 for any reasonable curve (the wgmma
    # M-tile floor is 64), so prefill is essentially unaffected by the
    # derate. Applied for consistency with decode and to capture small-S
    # corner cases.
    mb_prefill = max(1, B_pf) * max(1, S) / max(1, PP)
    eta_TC = _eta_TC_at_mb(system.device.tensor_core_efficiency, mb_prefill)
    t_prefill_compute_eff = t_prefill_compute / eta_TC if eta_TC > 0 else float("inf")
    t_prefill_mem = _t_mem(traffic.T_theta_device, traffic.T_kv_write_device, B=1)
    t_prefill_local_gpu = max(t_prefill_compute_eff, t_prefill_mem)
    t_prefill_local = _compose_SW(t_prefill_local_gpu)
    t_prefill_comm = comm.t_prefill_comm

    # Pipeline warmup: (PP-1) stages must fill before first token emerges
    # Each stage takes approximately t_prefill_local (prefill is typically compute-bound)
    t_pipeline_warmup = (PP - 1) * t_prefill_local if PP > 1 else 0.0

    t_prefill = (
        t_prefill_local
        + max(0.0, t_prefill_comm - rho * t_prefill_local)
        + t_pipeline_warmup
        + t_LM_prefill
    )

    # ── Batched prefill (§4) ─────────────────────────────

    if B_pf > 1:
        # FLOPs scale linearly with B_prefill
        t_batched_compute = B_pf * flops.F_prefill_device / R_gpu
        # η_TC at the batched payload (already computed above using B_pf · S).
        t_batched_compute_eff = t_batched_compute / eta_TC if eta_TC > 0 else float("inf")
        # Traffic: weights loaded once + B_pf * KV writes per request
        # (multi-tier sum; placement re-resolved at this B).
        t_batched_mem = _t_mem(
            traffic.T_theta_device, traffic.T_kv_write_device, B=B_pf,
        )
        t_batched_local_gpu = max(t_batched_compute_eff, t_batched_mem)
        t_batched_local = _compose_SW(t_batched_local_gpu)
        # Comm scales with the batched token count (B_pf · S), not S alone:
        # collective messages carry per-step activations whose payload grows
        # with tokens per step. α is unchanged; β (payload/BW) grows with B_pf.
        comm_batched = compute_prefill_comm(
            model, system, partition, tuner, fw, tokens_per_step=B_pf * S,
        )
        t_prefill_batched = (
            t_batched_local
            + max(0.0, comm_batched.t_prefill_comm - rho * t_batched_local)
            + t_pipeline_warmup
            + t_LM_prefill
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

        # Linear FLOPs per chunk (constant across chunks). Use D_attn for
        # projections, D_exp (MoE-aware) for FFN. For pure dense models the
        # MoE divisor collapses to TP via D_exp(layer_kind="dense").
        d_exp_for_chunk = d_exp_pf_moe if model.moe is not None else d_exp_pf_dense
        F_linear_per_chunk = (L / PP) * (
            (4 * H**2 + 4 * H * H_kv) * C / d_attn_pf
            + 6 * H * I_eff * C / d_exp_for_chunk
        )

        # Weight traffic per chunk (same each chunk — full weight read)
        T_theta_chunk = traffic.T_theta_device
        # KV write per chunk: C new entries (D_kv * SP divisor matches §1.4 / §3.1)
        T_kv_write_chunk = (L / PP) * (2 * C * H_kv * model.bytes_per_param) / (d_kv_pf * SP)

        # Chunk-level comm: evaluate collectives at C tokens per step.
        # α (latency term) is unchanged per chunk; β (payload) scales with C,
        # not with C/S. Linear C/S scaling underestimates α-dominated small-C.
        comm_chunk = compute_prefill_comm(
            model, system, partition, tuner, fw, tokens_per_step=C,
        )
        t_chunk_comm = comm_chunk.t_prefill_comm

        total_chunked = 0.0
        for k in range(1, n_chunks + 1):
            # Attention FLOPs for chunk k: attends to kC accumulated KV positions.
            # Sharded by D_kv·SP per prefill.md §1.5 / §5.1.
            F_attn_chunk_k = (L / PP) * (4 * C * k * C * H) / (d_kv_pf * SP)
            F_chunk_k = F_linear_per_chunk + F_attn_chunk_k

            t_chunk_compute_k = F_chunk_k / R_gpu

            # Memory: weights + KV write + KV read (kC entries for attention).
            # Per-tier sum (sram.md §2.1): KV-write + KV-read folded into a
            # single per-request KV term for placement purposes. Uses D_kv·SP.
            T_kv_read_k = (L / PP) * (2 * k * C * H_kv * model.bytes_per_param) / (d_kv_pf * SP)
            T_kv_chunk_k = T_kv_write_chunk + T_kv_read_k
            t_chunk_mem_k = _t_mem(T_theta_chunk, T_kv_chunk_k, B=1)

            # η_TC at chunk payload (mb_chunk = B_pf · C / PP). For typical
            # C the saturated curve gives 1.0 — same caveat as unchunked.
            mb_chunk = max(1, B_pf) * max(1, C) / max(1, PP)
            eta_TC_chunk = _eta_TC_at_mb(system.device.tensor_core_efficiency, mb_chunk)
            t_chunk_compute_eff_k = t_chunk_compute_k / eta_TC_chunk if eta_TC_chunk > 0 else float("inf")
            t_chunk_local_gpu_k = max(t_chunk_compute_eff_k, t_chunk_mem_k)
            t_chunk_local_k = _compose_SW(t_chunk_local_gpu_k)
            t_chunk_k = t_chunk_local_k + max(0.0, t_chunk_comm - rho * t_chunk_local_k)
            total_chunked += t_chunk_k

        # LM head fires once after the last chunk (one projection per request,
        # not per chunk), mirroring prefill.md §3.4.
        t_prefill_chunked = total_chunked + t_LM_prefill
    else:
        n_chunks = 0
        t_prefill_chunked = t_prefill  # no chunking → same as unchunked

    return PrefillLatencyResults(
        t_prefill_compute=t_prefill_compute,
        t_prefill_compute_eff=t_prefill_compute_eff,
        eta_TC=eta_TC,
        t_prefill_mem=t_prefill_mem,
        t_prefill_local=t_prefill_local,
        t_prefill_comm=t_prefill_comm,
        t_SW_per_stage=t_SW_per_stage,
        t_pipeline_warmup=t_pipeline_warmup,
        t_LM_prefill=t_LM_prefill,
        t_prefill=t_prefill,
        B_prefill=B_pf,
        t_prefill_batched=t_prefill_batched,
        chunk_size=C,
        n_chunks=n_chunks,
        t_prefill_chunked=t_prefill_chunked,
    )
