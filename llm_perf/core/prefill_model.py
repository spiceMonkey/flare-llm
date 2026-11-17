
import math
from dataclasses import dataclass, replace
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
    attn_proj_flops_per_token,
    ffn_flops_per_token,
    dense_weight_bytes,
    moe_weight_bytes,
    kv_bytes_per_seq,
    linear_flops_per_token,
    aggregate_per_stage,
    cost_collective,
    p2p_hop,
    assign_tier_per_axis,
    axis_stride,
    tier_at,
    D_batch,
    D_kv,
    D_emb,
    G_TP,
    G_EP,
    group_batch,
    moe_expert_flops_per_token,
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
    T_theta_device: float           # weight read traffic (same as decode); B_pf-independent
    T_kv_write_per_request: float   # per-device, per-prefill-request KV write traffic (S_input tokens worth). Mirrors decode's T_kv_token: the per-unit building block, NOT the batched-prefill aggregate. For B_pf concurrent requests, the device-aggregate is B_pf * T_kv_write_per_request.
    T_prefill_device: float         # single-request per-device total traffic: T_theta_device + T_kv_write_per_request


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
    t_kernel_per_stage: float           # per-stage CPU dispatch budget = (L/PP) · k · τ_launch
    t_pipeline_warmup: float        # (PP-1) * t_stage, where t_stage includes the stage's unhidden comm
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
# Prefill FLOPs (prefill.md §1)
# ────────────────────────────────────────────────────────────

def compute_prefill_flops(
    model: LlmModelSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
    framework: FrameworkSpec,
) -> PrefillFlopsResults:
    """Per-device prefill FLOPs. Doc: prefill.md §1.5

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
# Prefill Traffic (prefill.md §3.1)
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
    T_kv_write_per_request = kv_bytes_per_seq(model, partition, framework, S)

    T_prefill_device = T_theta_device + T_kv_write_per_request

    return PrefillTrafficResults(
        T_theta_device=T_theta_device,
        T_kv_write_per_request=T_kv_write_per_request,
        T_prefill_device=T_prefill_device,
    )


# ────────────────────────────────────────────────────────────
# Prefill Communication (prefill.md §3.2)
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
    b = model.act_bytes()       # collective payloads are activations
    b_kv = model.kv_bytes()     # the pass-KV gather moves cache entries
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
    align_policy = fw.torus_align_policy.lower()
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
            inc_enabled=inc_enabled, align_policy=align_policy,
            stride=axis_stride(partition, fw, coll),
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
        if model.mla is not None:
            # A latent-attention cache is one latent per token per layer, not
            # per-head keys and values; under TP-attention it is not split by
            # head either (attention.md §3.6).
            w_lat = model.mla.kv_bytes_per_token_per_layer(b_kv)
            msg_SP = tokens * (w_lat if fw.attention_mode == "tp" else w_lat / d_kv)
        else:
            msg_SP = tokens * (2 * H_kv / d_kv) * b_kv
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
# Prefill Latency (prefill.md §3-5)
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
    rho_kernel = fw.kernel_overlap_factor

    # Collective group sizes (notation.md §1) for kernel-launch SW count
    # axis-presence guards. Equal to TP and EP across all three production
    # configurations; threaded via helpers for consistency with the abstract
    # divisor symbols D_kv / D_emb.
    g_TP_pf = G_TP(partition)
    g_EP_pf = G_EP(partition)

    # Abstract sharding factors (notation.md §1) for the LM head and chunked
    # prefill formulas. d_kv composes with SP for KV-attention divisors.
    d_kv_pf = D_kv(partition, fw)
    d_emb_pf = D_emb(partition)

    S = tuner.S_input
    B_pf = tuner.B_prefill
    C = tuner.chunk_size

    H = model.H
    L = model.L

    # SW dispatch budget per stage. Prefill is one forward pass per request
    # (or per chunk) — there is no microbatch round structure as in decode,
    # so the per-stage formula is L/PP layers worth of launches plus one
    # PP boundary's worth of P2P sends/recvs (inert when PP = 1):
    #     t_kernel_per_stage = (L/PP) · k · τ_launch  +  k_pp_hop · τ_launch
    # where k decomposes into compute + collective kernel counts and the
    # collective counts are zero on axes where the parallelism is 1.
    # The PP-hop term uses the middle-stage 2× factor (recv + send) by
    # default; edge stages do only one direction (off by one k_pp_hop·τ,
    # negligible at PP >> 1).
    n_TP_calls = fw.n_TP_collectives if g_TP_pf > 1 else 0
    # n_EP_collectives counts NCCL API calls directly (dispatch + combine
    # = 2 per MoE layer); see decode_model._t_kernel_per_microbatch. EP launches
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
    t_kernel_per_stage = (
        layers_per_stage * k_dense * fw.kernel_launch_us * 1e-6
        + moe_layers_per_stage * k_moe_extra * fw.kernel_launch_us * 1e-6
        + k_pp_hop * fw.kernel_launch_us * 1e-6
    )

    def _compose_SW(t_local_gpu: float) -> float:
        """Compose per-stage GPU work with SW dispatch via ρ_kernel.

        Base + unhidden-overflow form (same pattern as compute/comm overlap
        in decode.md §6.2). GPU work is the base; SW dispatch overlaps for
        ρ_kernel · t_local_gpu; any remainder serializes after.

        ρ_kernel = 1 → t_local_gpu + max(0, t_kernel - t_local_gpu) = max(t_local_gpu, t_kernel)
        ρ_kernel = 0 → t_local_gpu + t_kernel_per_stage (no overlap)
        """
        return t_local_gpu + max(0.0, t_kernel_per_stage - rho_kernel * t_local_gpu)

    # Per-tier memory time helper. T_kv_write_per_request is per-request; the
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
    # LM head rows per device group (the batch divided across EP groups
    # under the orthogonal layout; == B_prefill otherwise).
    B_pf_eff = max(1, group_batch(B_pf, partition, fw))

    def _t_LM(B_eff: int) -> float:
        F_lm = 2.0 * B_eff * H * V / max(1, d_emb_pf)
        T_lm = (H * V * model.bytes_per_param) / max(1, d_emb_pf) + B_eff * V * model.act_bytes()
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
    t_prefill_mem = _t_mem(traffic.T_theta_device, traffic.T_kv_write_per_request, B=1)
    t_prefill_local_gpu = max(t_prefill_compute_eff, t_prefill_mem)
    t_prefill_local = _compose_SW(t_prefill_local_gpu)
    t_prefill_comm = comm.t_prefill_comm

    # The prompt crosses every pipeline stage before the first token can be
    # sampled, and each stage pays its own collectives and PP hop:
    #     t_prefill = PP · t_stage + t_LM
    # `t_pipeline_warmup` is the (PP − 1) stages after the first.
    t_prefill_stage = t_prefill_local + max(0.0, t_prefill_comm - rho * t_prefill_local)
    t_pipeline_warmup = (PP - 1) * t_prefill_stage if PP > 1 else 0.0

    t_prefill = t_prefill_stage + t_pipeline_warmup + t_LM_prefill

    # ── Batched prefill (§4) ─────────────────────────────

    if B_pf > 1:
        # FLOPs scale with B_prefill — but under DP-attention the batch is
        # sequence-sharded across the TP group, so per-rank attention work
        # scales with B_pf/G_TP, not B_pf. The FFN/MoE block always scales
        # with full B_pf (each rank does its D_exp-sharded share of FFN for
        # all batch). Mirrors the decode-side compute_flops correction;
        # the score/value piece already absorbs the batch shard via D_kv=G_TP
        # under DP-attn, so only the projection term needs the explicit
        # divisor here.
        F_attn_proj_pp = attn_proj_flops_per_token(model, partition, fw) * S
        F_ffn_pp = ffn_flops_per_token(model, partition, fw) * S
        F_attn_sv_pp = flops.F_prefill_device - F_attn_proj_pp - F_ffn_pp
        #
        # Orthogonal TP×EP additionally divides the requests across the EP
        # device groups (`D_batch`): a group prefills its own B_pf/EP
        # requests' attention, dense FFN and router, while the expert FFN
        # sees every request's tokens routed to its shard.
        Bg_pf = group_batch(B_pf, partition, fw)
        if fw.attention_mode == "dp" and partition.TP > 1:
            B_attn = Bg_pf / partition.TP
        else:
            B_attn = Bg_pf
        if D_batch(partition, fw) == 1:
            F_batched_total = (
                B_attn * F_attn_proj_pp
                + B_pf * F_attn_sv_pp
                + B_pf * F_ffn_pp
            )
        else:
            F_moe_expert_pp = moe_expert_flops_per_token(model, partition, fw) * S
            F_batched_total = (
                B_attn * F_attn_proj_pp
                + Bg_pf * F_attn_sv_pp
                + Bg_pf * (F_ffn_pp - F_moe_expert_pp)
                + B_pf * F_moe_expert_pp
            )
        t_batched_compute = F_batched_total / R_gpu
        # η_TC at the batched payload (already computed above using B_pf · S).
        t_batched_compute_eff = t_batched_compute / eta_TC if eta_TC > 0 else float("inf")
        # Traffic: weights loaded once + B_pf * KV writes per request
        # (multi-tier sum; placement re-resolved at this B).
        t_batched_mem = _t_mem(
            traffic.T_theta_device, traffic.T_kv_write_per_request, B=Bg_pf,
        )
        t_batched_local_gpu = max(t_batched_compute_eff, t_batched_mem)
        t_batched_local = _compose_SW(t_batched_local_gpu)
        # Comm scales with the batched token count (B_pf · S), not S alone:
        # collective messages carry per-step activations whose payload grows
        # with tokens per step. α is unchanged; β (payload/BW) grows with B_pf.
        comm_batched = compute_prefill_comm(
            model, system, partition, tuner, fw, tokens_per_step=Bg_pf * S,
        )
        # Same traversal as the single-request case, at the batched stage time.
        t_batched_stage = (
            t_batched_local
            + max(0.0, comm_batched.t_prefill_comm - rho * t_batched_local)
        )
        t_prefill_batched = max(1, PP) * t_batched_stage + t_LM_prefill
    else:
        t_prefill_batched = t_prefill

    # ── Chunked prefill (§5) ─────────────────────────────

    if C > 0 and S > 0:
        n_chunks = math.ceil(S / C)

        # Linear FLOPs per chunk (constant across chunks): the same shared
        # per-token primitive the unchunked pass uses (projections, dense and
        # MoE FFN, router; attention-variant and layout aware), × C tokens.
        # The last chunk holds what is left of the prompt (S − (n−1)·C tokens),
        # which is fewer than C unless C divides S; a chunk size above the
        # prompt length is one chunk of S tokens.
        F_linear_per_token = linear_flops_per_token(model, partition, fw)
        F_linear_per_chunk = F_linear_per_token * C

        # Weight traffic per chunk (same each chunk — full weight read)
        T_theta_chunk = traffic.T_theta_device
        # KV write per chunk: C new entries (variant-aware, same primitive
        # as the unchunked pass).
        T_kv_write_chunk = kv_bytes_per_seq(model, partition, fw, C)

        # Chunk-level comm: evaluate collectives at C tokens per step.
        # α (latency term) is unchanged per chunk; β (payload) scales with C,
        # not with C/S. Linear C/S scaling underestimates α-dominated small-C.
        comm_chunk = compute_prefill_comm(
            model, system, partition, tuner, fw, tokens_per_step=C,
        )
        t_chunk_comm = comm_chunk.t_prefill_comm

        if model.mla is not None:
            from .primitives.mla_flops import mla_score_value_flops_per_layer_per_device

        total_chunked = 0.0
        t_chunk_k = 0.0
        t_chunk_max = 0.0
        for k in range(1, n_chunks + 1):
            # Attention FLOPs for chunk k. The unchunked pass charges every
            # query the full S positions (the 4·S²·H convention of prefill.md
            # §1.2: implementations compute whole tiles and mask), which is
            # twice the causal work. Chunk k's C queries causally see
            # (k−1)·C earlier positions plus half the chunk on average, so
            # the same convention charges them (2k−1)·C positions: the chunks
            # then sum to exactly the unchunked 4·S²·H, and chunking by
            # itself neither adds nor removes attention work. Sharded by
            # D_kv·SP per prefill.md §1.5 / §5.1; MLA models use the
            # variant-specific score/value cost at that context length.
            C_k = min(C, S - (k - 1) * C)          # tokens in chunk k
            full = C_k == C
            S_attn_k = 2 * (k - 1) * C + C_k        # = (2k−1)·C for a full chunk
            if model.mla is not None:
                F_sv_per_tok = mla_score_value_flops_per_layer_per_device(
                    model, partition, fw, S_attn_k, fw.mla_mode
                )
                F_attn_chunk_k = (L / PP) * F_sv_per_tok * C_k / SP
            else:
                F_attn_chunk_k = (L / PP) * (4 * C_k * S_attn_k * H) / (d_kv_pf * SP)
            F_chunk_k = (F_linear_per_chunk if full else F_linear_per_token * C_k) + F_attn_chunk_k

            t_chunk_compute_k = F_chunk_k / R_gpu

            # Memory: weights + KV write (this chunk) + KV read of the
            # (k−1)·C positions earlier chunks already stored. Per-tier sum
            # (sram.md §2.1): write + read folded into a single per-request
            # KV term for placement purposes.
            T_kv_read_k = kv_bytes_per_seq(model, partition, fw, (k - 1) * C)
            T_kv_write_k = T_kv_write_chunk if full else kv_bytes_per_seq(model, partition, fw, C_k)
            T_kv_chunk_k = T_kv_write_k + T_kv_read_k
            t_chunk_mem_k = _t_mem(T_theta_chunk, T_kv_chunk_k, B=1)

            # η_TC at chunk payload (mb_chunk = B_pf · C / PP). For typical
            # C the saturated curve gives 1.0 — same caveat as unchunked.
            mb_chunk = max(1, B_pf) * max(1, C_k) / max(1, PP)
            eta_TC_chunk = _eta_TC_at_mb(system.device.tensor_core_efficiency, mb_chunk)
            t_chunk_compute_eff_k = t_chunk_compute_k / eta_TC_chunk if eta_TC_chunk > 0 else float("inf")
            t_chunk_local_gpu_k = max(t_chunk_compute_eff_k, t_chunk_mem_k)
            t_chunk_local_k = _compose_SW(t_chunk_local_gpu_k)
            t_comm_k = t_chunk_comm if full else compute_prefill_comm(
                model, system, partition, tuner, fw, tokens_per_step=C_k,
            ).t_prefill_comm
            t_chunk_k = t_chunk_local_k + max(0.0, t_comm_k - rho * t_chunk_local_k)
            total_chunked += t_chunk_k
            t_chunk_max = max(t_chunk_max, t_chunk_k)

        # Chunks follow each other through the pipeline one stage apart, so
        # the first stage sees them back to back (the sum above) and the
        # remaining PP − 1 stages add the longest chunk's stage time each:
        # every chunk takes the same time on every stage, and for such a
        # flow the makespan is the sum plus (PP − 1) times the longest job.
        # With whole chunks the longest is the last (attention grows with
        # k); a short final chunk does not shorten the tail. The LM head
        # fires once after the last chunk (one projection per request, not
        # per chunk), mirroring prefill.md §3.4.
        t_chunk_tail = (PP - 1) * t_chunk_max if PP > 1 else 0.0
        t_prefill_chunked = total_chunked + t_chunk_tail + t_LM_prefill
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
        t_kernel_per_stage=t_kernel_per_stage,
        t_pipeline_warmup=t_pipeline_warmup,
        t_LM_prefill=t_LM_prefill,
        t_prefill=t_prefill,
        B_prefill=B_pf,
        t_prefill_batched=t_prefill_batched,
        chunk_size=C,
        n_chunks=n_chunks,
        t_prefill_chunked=t_prefill_chunked,
    )


# ────────────────────────────────────────────────────────────
# Suffix prefill on a cached prefix (prefix reuse)
# ────────────────────────────────────────────────────────────

@dataclass
class SuffixPrefillResults:
    S_prefix: int                   # tokens whose cache is already present
    S_suffix: int                   # new tokens prefilled: S_input - S_prefix
    F_suffix_device: float          # per-device FLOPs across L/PP layers
    t_compute_eff: float            # Tensor-Core-derated compute time
    t_mem: float                    # weights + KV write (suffix) + KV read (prefix)
    t_local: float                  # max(compute_eff, mem) composed with the SW budget
    t_comm: float                   # per-stage collectives at S_suffix tokens
    t_stage: float                  # t_local + unhidden comm
    t_LM: float                     # LM head on the last position, once
    t_suffix: float                 # PP * t_stage + t_LM


def compute_suffix_prefill(
    model: LlmModelSpec,
    system: SystemSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
    framework: FrameworkSpec,
    S_prefix: int,
) -> SuffixPrefillResults:
    """Prefill of the last `S_input - S_prefix` tokens on top of a cached prefix.

    The new tokens attend to every position of the prompt. Under the
    convention of the chunked model above (each query is charged the full
    tile, so chunk k of size C is charged (2k-1)*C positions) a suffix of C
    tokens after a prefix of P is charged 2P + C positions per query: its
    attention work is exactly that of the whole prompt less that of the
    prefix, 4*H*((P+C)^2 - P^2) per layer. It is priced as one chunk of that
    model, for a prefix of ANY length:

      compute  linear FLOPs of C tokens + that attention work
      memory   weights once + KV write of C entries + KV read of P entries
      comm     the stage's collectives at C tokens per step
      total    PP stages, then the LM head once

    With P a multiple of C this is the chunk the chunked model prices for
    the same tokens; with P = 0 it is one chunk covering the whole prompt.
    Single request, like the chunked model (`tuner.B_prefill` enters only
    the Tensor-Core payload). `tuner.S_input` is the whole prompt.
    """
    fw = framework
    S = tuner.S_input
    if int(S_prefix) != S_prefix or S_prefix < 0:
        raise ValueError(f"S_prefix must be a non-negative integer, got {S_prefix!r}")
    if S_prefix >= S:
        raise ValueError(f"S_prefix ({S_prefix}) must be below the prompt length S_input ({S}): at least one token is prefilled")
    P = int(S_prefix)
    C = S - P

    # The whole-prompt pass supplies what does not depend on the split: the
    # per-stage SW dispatch budget, the LM head and the weight traffic.
    whole = replace(tuner, chunk_size=0)
    traffic = compute_prefill_traffic(model, partition, whole, fw)
    lat = compute_prefill_latency(
        system, partition, whole, model,
        compute_prefill_flops(model, partition, whole, fw), traffic,
        compute_prefill_comm(model, system, partition, whole, fw), fw,
    )

    R_gpu = effective_peak_flops_TF(system, model.bytes_per_param) * TB_TO_FLOPS
    tiers = system.device.get_tiers()
    PP, SP, L, H = partition.PP, partition.SP, model.L, model.H
    d_kv = D_kv(partition, fw)

    F_linear = linear_flops_per_token(model, partition, fw) * C
    S_attn = 2 * P + C
    if model.mla is not None:
        from .primitives.mla_flops import mla_score_value_flops_per_layer_per_device
        F_attn = (L / PP) * mla_score_value_flops_per_layer_per_device(model, partition, fw, S_attn, fw.mla_mode) * C / SP
    else:
        F_attn = (L / PP) * (4 * C * S_attn * H) / (d_kv * SP)
    F_suffix = F_linear + F_attn
    t_compute = F_suffix / R_gpu

    T_kv = kv_bytes_per_seq(model, partition, fw, C) + kv_bytes_per_seq(model, partition, fw, P)
    plc = resolve_placement(
        T_theta_device=traffic.T_theta_device,
        T_kv_per_request_device=T_kv,
        B=1,
        tiers=tiers,
        placement=tuner.placement,
    )
    t_mem = t_mem_from_placement(
        plc, B=1, tiers=tiers,
        eta_beta_curve_factor=_eta_beta_at_B(system.device.bw_efficiency, 1),
    )

    mb = max(1, tuner.B_prefill) * max(1, C) / max(1, PP)
    eta_TC = _eta_TC_at_mb(system.device.tensor_core_efficiency, mb)
    t_compute_eff = t_compute / eta_TC if eta_TC > 0 else float("inf")
    t_local_gpu = max(t_compute_eff, t_mem)
    t_local = t_local_gpu + max(0.0, lat.t_kernel_per_stage - fw.kernel_overlap_factor * t_local_gpu)
    t_comm = compute_prefill_comm(model, system, partition, tuner, fw, tokens_per_step=C).t_prefill_comm
    t_stage = t_local + max(0.0, t_comm - fw.comm_overlap_factor * t_local)
    return SuffixPrefillResults(
        S_prefix=P,
        S_suffix=C,
        F_suffix_device=F_suffix,
        t_compute_eff=t_compute_eff,
        t_mem=t_mem,
        t_local=t_local,
        t_comm=t_comm,
        t_stage=t_stage,
        t_LM=lat.t_LM_prefill,
        t_suffix=max(1, PP) * t_stage + lat.t_LM_prefill,
    )
