"""Decode-phase performance model — consolidates flops + traffic + comm + latency.

Mirrors the shape of `prefill_model.py`: four pure functions on typed
dataclasses, each returning a small result dataclass. Internally the
phase-agnostic physics (weight/KV footprint, linear FLOPs, collective
cost) lives in `core/primitives/`; this module wires those primitives
together with decode-specific pieces (attention that scales with S, not
S²; message sizes that are B·H·b, not S·H·b).

Documentation: `decode.md`.

Public surface (preserved from the pre-refactor four-file split):
  - compute_flops(model, partition, tuner) → FlopsResults
  - compute_traffic(model, partition, tuner) → TrafficResults
  - compute_comm(model, system, partition, tuner) → CommResults
  - compute_latency(model, system, partition, tuner, flops, traffic, comm) → LatencyResults
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

from ..specs.framework_spec import FrameworkSpec
from ..specs.model_spec import LlmModelSpec
from ..specs.system_spec import SystemSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.tuner_spec import TuningSpec
from ..utils import GB_TO_BYTES, TB_TO_FLOPS
from .memory_placement import (
    PlacementResult,
    resolve_placement,
    spill_seconds_per_byte,
    t_mem_from_placement,
    weights_seconds_per_byte,
)
from .primitives import (
    attn_proj_flops_per_token,
    ffn_flops_per_token,
    moe_expert_flops_per_token,
    dense_weight_bytes,
    moe_weight_bytes,
    embedding_bytes,
    moe_weight_traffic_bytes,
    kv_bytes_per_seq,
    linear_flops_per_token,
    aggregate_per_stage,
    cost_collective,
    p2p_hop,
    assign_tier_per_axis,
    axis_stride,
    replicas_in_system,
    tier_at,
    D_kv,
    D_emb,
    G_TP,
    G_EP,
    N_replica,
    decode_microbatch,
    D_batch,
    group_batch,
    attn_rank_batch,
    kv_batch,
)


# ────────────────────────────────────────────────────────────
# Effective compute peak under linear byte-ratio scaling
# ────────────────────────────────────────────────────────────

# Reference precision: FP16 (2 bytes per parameter). System specs store
# peak_flops_TF as FP16 dense per chip; the framework derives precision-
# specific peaks via linear byte scaling.
_FP16_BYTES = 2.0


def effective_peak_flops_TF(system: SystemSpec, bytes_per_param: float) -> float:
    """Precision-aware compute peak (TFLOPS) per device.

    Uniform convention across all system specs: ``peak_flops_TF`` stores
    the **FP16 dense per-chip peak**. Lower precisions get a linear byte-
    ratio boost: ``peak(p) = peak_FP16 * (2 / bytes_per_param)``. Phase F:
    multiplied by static `peak_flops_eta` (per-device sustained / nameplate
    deflator, sibling to MemoryTierSpec.eta_beta on the memory side; ideal
    1.0 = chip sustains nameplate FP16 dense peak).

    Examples on GB200 NVL72 (peak_FP16 = 2250 TF/GPU, peak_flops_eta = 1.0):
      - FP16 (b=2.0): 2250 TF
      - FP8  (b=1.0): 4500 TF
      - FP4  (b=0.5): 9000 TF

    A device whose published peaks do not follow that rule carries them in
    `DeviceSpec.peak_flops_TF_by_precision` (bytes per parameter → TFLOPs),
    which then replaces the byte-ratio rule. d-Matrix Corsair is the case
    in the database: its vendor figures quadruple per halving of precision
    (75 / 300 / 1200 TFLOPs per chiplet at 16 / 8 / 4 bits), so the linear
    rule would under-state its 4-bit peak by 2× and over-state its 16-bit
    peak by 2×. An exact key is used as is; a precision between two keys is
    interpolated geometrically; one outside the table is byte-ratio scaled
    from the nearest key.
    """
    bpp = max(1e-9, bytes_per_param)
    eta = system.device.peak_flops_eta
    table = system.device.peak_flops_TF_by_precision
    if table:
        keys = sorted(table)
        if bpp <= keys[0]:
            return table[keys[0]] * (keys[0] / bpp) * eta
        if bpp >= keys[-1]:
            return table[keys[-1]] * (keys[-1] / bpp) * eta
        for lo, hi in zip(keys, keys[1:]):
            if lo <= bpp <= hi:
                if bpp == lo:
                    return table[lo] * eta
                if bpp == hi:
                    return table[hi] * eta
                w = math.log(bpp / lo) / math.log(hi / lo)
                return table[lo] * (table[hi] / table[lo]) ** w * eta
    return system.device.peak_flops_TF * eta * (_FP16_BYTES / bpp)


# ────────────────────────────────────────────────────────────
# SW-overhead helpers (decode.md §7.1)
# ────────────────────────────────────────────────────────────

def _piecewise_linear_lookup(curve: Optional[Dict[int, float]], x: float) -> float:
    """Piecewise-linear interpolation of `curve` at point `x`.

    None ⇒ 1.0 always (legacy: no derate).
    `x` clamps to the curve's [min_key, max_key] range.
    """
    if curve is None or not curve:
        return 1.0
    keys = sorted(curve.keys())
    if x <= keys[0]:
        return float(curve[keys[0]])
    if x >= keys[-1]:
        return float(curve[keys[-1]])
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= x <= hi:
            t = (x - lo) / (hi - lo)
            return float(curve[lo]) + t * (float(curve[hi]) - float(curve[lo]))
    return 1.0  # unreachable; defensive


def _eta_TC_at_mb(curve: Optional[Dict[int, float]], mb: float) -> float:
    """Tensor Core efficiency at microbatch `mb` (compute roofline derate)."""
    return _piecewise_linear_lookup(curve, mb)


def _eta_beta_at_B(curve: Optional[Dict[int, float]], B: float) -> float:
    """B-dependent sustained HBM bandwidth at active-sequence count `B`.

    Multiplicative derate on top of constant `bw_eta` and per-tier `eta_beta`.
    None ⇒ 1.0 always (legacy: no B-dependent derate). See decode.md §6.2.
    """
    return _piecewise_linear_lookup(curve, B)


def _t_kernel_per_microbatch(
    L: int,
    L_moe: int,
    tuner: TuningSpec,
    framework: FrameworkSpec,
    partition: PartitionSpec,
) -> float:
    """Per-microbatch dispatch budget on a single PP stage.

        t_kernel = (L/PP)     · (k_compute + k_coll · (n_TP + n_SP)) · τ_launch
             + (L_moe/PP) · k_coll · n_EP · τ_launch
             + k_pp_hop   · τ_launch    (P2P send/recv per microbatch transit)

    Same units as `t_stage` (per microbatch on this stage), so the two
    compose directly via the SW overlap factor in the t_step,user formula
    without a unit mismatch.

    Layer breakdown:
      - All L/PP layers on this stage fire compute + TP + SP launches.
      - Only the L_moe/PP MoE layers on this stage fire EP launches
        (dense layers don't trigger MoE A2A). Mirrors the per-layer
        accounting in decode.md §5.5's t_comm formula.
      - Under scatter-direct dispatch with DP-attention a MoE layer runs
        no TP collective, so its TP launches are taken back out.

    PP-hop term: one microbatch transit triggers k_pp_hop P2P kernels
    on a middle stage (default 2: 1 recv + 1 send). Edge stages do only
    one direction; the formula uses the middle-stage value (off by
    half a launch on edges, negligible at PP > 1). Inert when PP=1
    (no inter-stage hops).

    `n_collectives_per_layer` counts only collectives that actually
    fire for the current shape (zero when the corresponding parallelism
    axis is 1).

    Returns 0 when kernel_launch_us is 0 (legacy behavior).
    """
    tau_us = framework.kernel_launch_us
    if tau_us <= 0.0:
        return 0.0
    k_c = framework.kernels_per_layer_compute
    k_coll = framework.kernels_per_collective_call
    n_TP_calls = framework.n_TP_collectives if G_TP(partition) > 1 else 0
    # n_EP_collectives counts NCCL API calls directly (dispatch + combine
    # = 2 per MoE layer); each costs one single-direction A2A in dispatch.py.
    n_EP_calls = framework.n_EP_collectives if G_EP(partition) > 1 else 0
    n_SP_calls = framework.n_SP_collectives if partition.SP > 1 else 0
    PP = max(1, partition.PP)
    layers_per_stage = L / PP
    moe_layers_per_stage = L_moe / PP
    k_dense = k_c + k_coll * (n_TP_calls + n_SP_calls)
    k_moe_extra = k_coll * n_EP_calls
    # Scatter-direct dispatch under DP-attention fires no TP collective on a
    # MoE layer (compute_comm prices none), so those layers launch none.
    if framework.moe_a2a_pattern == "scatter" and framework.attention_mode == "dp" and G_TP(partition) > 1:
        k_moe_extra -= k_coll * n_TP_calls
    t_layer = layers_per_stage * k_dense * tau_us * 1e-6
    t_moe = moe_layers_per_stage * k_moe_extra * tau_us * 1e-6
    k_pp_hop = framework.kernels_per_pp_hop if partition.PP > 1 else 0
    t_pp = k_pp_hop * tau_us * 1e-6
    return t_layer + t_moe + t_pp


# ────────────────────────────────────────────────────────────
# Result dataclasses (names preserved from flops/traffic/comm/latency_model)
# ────────────────────────────────────────────────────────────

@dataclass
class FlopsResults:
    F_token_device: float
    F_layer_per_device: float
    F_step_device: float


@dataclass
class TrafficResults:
    T_theta: float          # per-device weight traffic per stage pass (batch-independent for dense; microbatch-aware for MoE)
    T_kv_token: float       # per-device, per-token KV traffic (one active sequence's KV-history read, all stage layers); matches decode.md §2.3 T_{KV,token}
    T_kv_device: float      # per-device, per-stage-pass KV traffic aggregated over the microbatch: mb * T_kv_token (mb = B at PP = 1); matches decode.md §2.3 T_{KV,device}
    T_token_eff: float      # effective per-token (B=1) traffic
    T_step_eff: float       # effective per-step (batched) traffic = T_theta + T_kv_device


@dataclass
class CommResults:
    t_PP: float
    t_TP: float
    t_EP: float
    t_SP: float
    t_comm_stage: float
    msg_PP_bytes: float
    msg_TP_bytes: float
    msg_EP_bytes: float
    msg_SP_bytes: float


@dataclass
class LatencyResults:
    t_compute: float          # raw roofline compute time for one stage pass = F_step / R_gpu
    t_compute_eff: float      # Tensor-Core-derated compute time (= t_compute / η_TC)
    eta_TC: float             # Tensor Core efficiency factor at this mb (1.0 = peak)
    t_mem: float
    t_local: float            # max(t_compute_eff, t_mem) — memory-or-compute-bound roofline
    t_comm: float
    t_stage: float            # GPU-only step time (compute + comm + overlap)
    t_kernel: float               # per-round CPU dispatch budget = L · k · τ_launch
    t_LM: float               # LM head one-shot latency on stage PP-1 (decode.md §6.2)
    t_step_user: float
    pp_bubble_factor: float   # diagnostic: stages per in-flight microbatch, max(1, PP/B); 1.0 = pipeline full
    TPS_single: float
    TTPS: float
    B: int
    TPOT: float
    B_star: float
    # Speculative-decoding extension (decode.md §8). When tuner.n_tok_draft = 0
    # (vanilla decode, default), N_tok_per_step = 1.0 and TPOT_spec == TPOT.
    # When speculation is enabled, t_step_user_verify is the verify-step
    # latency (compute + comm scaled by n_tok_verify; t_mem ≈ unchanged) and
    # TPOT_spec = t_step_user_verify / N_tok_per_step.
    N_tok_per_step: float = 1.0
    t_step_user_verify: float = 0.0
    TPOT_spec: float = 0.0
    # Per-sequence serving runtime overhead (decode.md §7.3). GROSS per-step
    # host cost: c_seq_us * B * 1e-6. Parallel to t_kernel which also
    # stores the gross dispatch budget — both are diagnostic surfaces. The
    # actual contribution to t_step_user is the OVERLAP-GATED form
    # max(0, t_step_seq - seq_overlap_factor * t_step_base), applied
    # internally; readers comparing the t_step_seq curve against t_step,user
    # in plots see the overlap gate's hidden portion as the gap.
    t_step_seq: float = 0.0
    # Pipeline composition (decode.md §7.2). `microbatch` is the sequences
    # per stage pass (B/PP when the pipeline is full, 1 below that);
    # `pp_traversal_factor` is the stage passes a token pays (= PP).
    # t_step_base = pp_traversal_factor · t_stage_with_kernel(microbatch) + t_LM.
    pp_traversal_factor: float = 1.0
    microbatch: float = 1.0
    # Sequences one device group handles per stage pass: microbatch / D_batch
    # (orthogonal TP×EP divides the batch across the EP groups).
    group_batch: float = 1.0


# ────────────────────────────────────────────────────────────
# Decode FLOPs (decode.md §3)
# ────────────────────────────────────────────────────────────

def compute_flops(
    model: LlmModelSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
    framework: FrameworkSpec,
) -> FlopsResults:
    """Per-device decode FLOPs per token (and per step)."""
    L = model.L
    H = model.H
    PP = partition.PP
    SP = partition.SP
    S = tuner.S_decode
    # Per-pass batch: the microbatch one stage processes (== B at PP = 1).
    B = decode_microbatch(tuner.B_decode, PP)

    # Per-token FLOPs decomposed by which per-rank batch divisor applies.
    # - Attention projection (Q/K/V/O) and attention score/value: under
    #   DP-attention, each rank processes only B/G_TP tokens through the
    #   attention block (batch is sequence-sharded across the TP group),
    #   so per-step attention compute scales with B/G_TP, not B. Under
    #   TP-attention, each rank sees all B tokens but does 1/D_attn of
    #   the per-token work (head-sharded), so per-step = B × per-token.
    # - FFN (dense FFN + MoE FFN + MoE router): unaffected by the
    #   attention mode — each rank processes all B tokens through its
    #   D_exp-sharded FFN / expert set, so per-step = B × per-token.
    F_attn_proj_per_token = attn_proj_flops_per_token(model, partition, framework)
    F_ffn_per_token = ffn_flops_per_token(model, partition, framework)
    if model.mla is not None:
        from .primitives.mla_flops import mla_score_value_flops_per_layer_per_device
        F_sv_per_layer = mla_score_value_flops_per_layer_per_device(
            model, partition, framework, model.attn_tokens(S), framework.mla_mode
        )
        F_attn_sv_per_token = (L / PP) * F_sv_per_layer / SP
    else:
        # A query scores the positions its layer sees: all S, or the window on a window layer.
        F_attn_sv_per_token = (L / PP) * (4 * model.attn_tokens(S) * model.H_q()) / (D_kv(partition, framework) * SP)

    F_attn_per_token = F_attn_proj_per_token + F_attn_sv_per_token
    F_token_device = F_attn_per_token + F_ffn_per_token
    F_layer_per_device = F_token_device / (L / PP) if L > 0 else 0.0

    # Per-rank batch divisor for the attention block.
    # Note: the score/value piece (F_attn_sv) already contains /D_kv, which
    # under DP-attn happens to equal G_TP — so multiplying by full B still
    # gives the correct per-rank score/value cost. The over-counting bug
    # was specifically in the projection (D_attn = 1 under DP-attn means
    # the per-token projection is full-cost; multiplying by B then over-
    # counts by G_TP). We apply the B/G_TP divisor to the *projection*
    # contribution only; the score/value piece is left at × B.
    #
    # Orthogonal TP×EP additionally divides the batch across the EP device
    # groups (`D_batch`): each group runs attention, dense FFN and the
    # router for its own B/EP sequences, while the expert FFN sees tokens
    # from the whole batch routed to its shard.
    Bg = group_batch(B, partition, framework)
    B_attn_proj = attn_rank_batch(Bg, partition, framework)
    B_sv = kv_batch(Bg, partition, framework)
    if D_batch(partition, framework) == 1:
        F_step_device = (
            B_attn_proj * F_attn_proj_per_token
            + B_sv * F_attn_sv_per_token
            + B * F_ffn_per_token
        )
    else:
        F_moe_expert_per_token = moe_expert_flops_per_token(model, partition, framework)
        F_home_ffn_per_token = F_ffn_per_token - F_moe_expert_per_token
        F_step_device = (
            B_attn_proj * F_attn_proj_per_token
            + B_sv * F_attn_sv_per_token
            + Bg * F_home_ffn_per_token
            + B * F_moe_expert_per_token
        )

    return FlopsResults(
        F_token_device=F_token_device,
        F_layer_per_device=F_layer_per_device,
        F_step_device=F_step_device,
    )


# ────────────────────────────────────────────────────────────
# Decode Traffic (decode.md §4)
# ────────────────────────────────────────────────────────────

def compute_traffic(
    model: LlmModelSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
    framework: FrameworkSpec,
) -> TrafficResults:
    """Per-stage-pass HBM traffic per device (weights + KV cache read).

    "Per pass" means one microbatch crossing one pipeline stage; a token
    pays this once on each of the PP stages (see `decode_microbatch`).

    Dense weights (attention, dense-FFN) are read every step → traffic
    equals footprint. MoE expert weights are read only for the experts
    the current batch actually touches → traffic uses an expectation-
    of-touched-experts correction that converges to the footprint at
    large B. See `core/primitives/weight_quantities.py` for the formula
    and the uniform-routing assumption.
    """
    S = tuner.S_decode
    # Per-pass batch: the microbatch one stage processes (== B at PP = 1).
    B = decode_microbatch(tuner.B_decode, partition.PP)

    # Parameter traffic (embedding is outside the forward path by convention).
    # Dense weights: read every step. MoE: uses B-aware expectation of touched
    # experts (only the routed-to experts contribute to per-step HBM traffic).
    T_theta = (
        dense_weight_bytes(model, partition, framework)
        + moe_weight_traffic_bytes(model, partition, framework, B)
    )
    # KV read traffic for one sequence of S context tokens (per-token / per-sequence,
    # per-device — decode.md §2.3 T_{KV,token}).
    T_kv_token = kv_bytes_per_seq(model, partition, framework, S)
    # Device-aggregate per pass: each sequence the device's group serves
    # reads its own KV history (decode.md §2.3 T_{KV,device} = B_g · T_{KV,token},
    # with B_g = B / D_batch the group's share of the batch; B_g = B unless
    # the orthogonal layout spreads the batch over EP groups).
    T_kv_device = kv_batch(group_batch(B, partition, framework), partition, framework) * T_kv_token

    T_token_eff = T_theta + T_kv_token
    # Batched step: weights loaded once, KV read per sequence in the batch.
    T_step_eff = T_theta + T_kv_device

    return TrafficResults(
        T_theta=T_theta,
        T_kv_token=T_kv_token,
        T_kv_device=T_kv_device,
        T_token_eff=T_token_eff,
        T_step_eff=T_step_eff,
    )


# ────────────────────────────────────────────────────────────
# Decode Communication (decode.md §5)
# ────────────────────────────────────────────────────────────

def compute_comm(
    model: LlmModelSpec,
    system: SystemSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
    framework: FrameworkSpec,
) -> CommResults:
    """Per-stage decode communication time (seconds)."""
    H = model.H
    H_kv = model.H_kv()
    L = model.L
    PP = partition.PP
    SP = partition.SP
    S = tuner.S_decode
    # Collective payloads carry one microbatch (== B at PP = 1).
    # ... of the sequences this device group serves (orthogonal TP×EP
    # divides the batch across EP groups; see `D_batch`).
    B = max(1.0, group_batch(decode_microbatch(tuner.B_decode, PP), partition, framework))
    b = model.act_bytes()       # collective payloads are activations
    b_kv = model.kv_bytes()     # except the pass-KV gather, which moves cache entries

    fw = framework

    # Collective group sizes (notation.md §1; equal to TP and EP across all
    # three production-relevant configurations, but threaded via helpers for
    # consistency with the abstract divisor symbols D_kv / D_emb).
    g_TP = G_TP(partition)
    g_EP = G_EP(partition)

    n_TP = fw.n_TP_collectives
    n_EP = fw.n_EP_collectives
    n_SP = fw.n_SP_collectives

    if model.moe is not None:
        N_exp = max(1, model.moe.n_experts)
        g_EP = min(g_EP, N_exp)
        k = model.moe.k_active
    else:
        g_EP = 1
        k = 1

    # Algorithm selection lives on FrameworkSpec (Phase E). Decode reads
    # the per-phase fields directly; "auto" must have been resolved by
    # `optimize_collective_algorithms` upstream.
    tp_algorithm = fw.tp_algorithm_decode.lower()
    ep_algorithm = fw.ep_algorithm_decode.lower()
    torus_alg = fw.torus_algorithm.lower()
    align_policy = fw.torus_align_policy.lower()
    inc_enabled = fw.inc_enabled

    if tp_algorithm == "auto" or ep_algorithm == "auto":
        raise ValueError(
            "FrameworkSpec has algorithm='auto' for decode; resolve via "
            "core.collective_algo_opt.optimize_collective_algorithms(...) "
            "before InferenceCalculator.run()."
        )

    def _cost(coll: str, op: str, M: float, G: int, alg: str = "ring") -> float:
        return cost_collective(
            system.get_tier_chain(coll), op, M, G,
            algorithm=alg, torus_algorithm=torus_alg,
            inc_enabled=inc_enabled, align_policy=align_policy,
            # Nested layout: an outer axis's ranks sit a whole inner block
            # apart, which decides the fabric tier the collective runs on.
            stride=axis_stride(partition, framework, coll),
        )

    # PP: shard-preserving hop of one microbatch × (H/D_kv) activation bytes.
    # A single-stage pipeline has no inter-stage forward.
    #
    # Cost the hop at the *correct* fabric tier under nested-layout rule
    # (DP→PP→EP→TP→SP, fast axes inner). The legacy `_cost("PP", "p2p", _, 2)`
    # call always priced PP at tier 0 because G=2 picks the innermost tier;
    # under nested layout, PP boundaries can cross outer tiers (PCIe / IB)
    # when PP × inner-axes (TP, EP, SP) exceeds an inner tier's reach.
    # `assign_tier_per_axis` resolves the right tier per partition.
    d_kv = D_kv(partition, framework)
    if PP > 1:
        msg_PP = B * (H / d_kv) * b
        pp_tier_idx = assign_tier_per_axis(partition, framework, system, role="PP")["PP"]
        pp_tier = tier_at(system, "PP", pp_tier_idx)
        bw_Bps = pp_tier.bw_per_port_GBps * 1e9
        alpha_s = pp_tier.alpha_us * 1e-6
        t_PP = p2p_hop(msg_PP, alpha_s, bw_Bps)
    else:
        msg_PP = 0.0
        t_PP = 0.0

    # EP: 2-pass all-to-all (Dispatch + Combine) over k·H activation bytes.
    # Per-rank dispatch payload depends on the MoE A2A data-flow pattern
    # (decode.md §5.2):
    #   "gather" (default) — full B per rank; payload = B·k·H·b
    #   "scatter" + DP-attn — B/G_TP per rank; payload = (B/G_TP)·k·H·b
    # Scatter is a no-op outside DP-attn (the structural prerequisite is
    # that attention has DP-sharded the batch across the TP group).
    scatter_direct = (
        fw.moe_a2a_pattern == "scatter"
        and fw.attention_mode == "dp"
        and g_TP > 1
    )
    if g_EP > 1:
        if scatter_direct:
            # A rank holds whole sequences, so the one that dispatches sends at
            # least one sequence's tokens (the floor binds only below G_TP).
            msg_EP = (max(1.0, B / g_TP) if B > 0 else 0.0) * k * H * b
        else:
            msg_EP = B * k * H * b
        t_EP = _cost("EP", "moe_a2a", msg_EP, g_EP, alg=ep_algorithm)
    else:
        t_EP = 0.0
        msg_EP = 0.0

    # TP: 2-pass all-reduce of B·H activation bytes
    if g_TP > 1:
        msg_TP = B * H * b
        t_TP = _cost("TP", "all_reduce", msg_TP, g_TP, alg=tp_algorithm)
    else:
        t_TP = 0.0
        msg_TP = 0.0

    # DP-attention swap (decode.md §5.3 + notation.md §1 lookup): under
    # framework.attention_mode="dp", the per-layer attention TP all-reduce is
    # replaced by a single TP all-gather at the attention → FFN transition.
    # The FFN's TP all-reduce remains. Here we precompute the AG cost; the
    # per-stage adjustment happens after aggregate_per_stage below.
    if g_TP > 1 and fw.attention_mode == "dp":
        t_TP_AG = _cost("TP", "all_gather", msg_TP, g_TP)
    else:
        t_TP_AG = 0.0

    # SP (ring attention over a sequence-sharded KV cache). Two exact
    # variants exist (Yang et al., "Context Parallelism for Scalable
    # Million-Token Inference", arXiv:2411.01783):
    #   pass-Q  — the new token's query visits every rank, each rank attends
    #             over its own KV shard, and the partial outputs are merged
    #             back. The payload is one query's worth of activations, so
    #             the round trip is costed as an all-reduce of mb·(H/D_kv).
    #             This is the variant that paper uses for decode.
    #   pass-KV — every rank gathers the whole KV cache: an all-gather of
    #             mb·S·(2·H_kv/D_kv) cache bytes per layer. Right for a full
    #             prefill, far too expensive for a one-token query.
    if SP > 1:
        if fw.sp_decode_mode == "pass_kv":
            if model.mla is not None:
                # One latent per token per layer, not head-split under
                # TP-attention (attention.md §3.6).
                w_lat = model.mla.kv_bytes_per_token_per_layer(b_kv)
                msg_SP = B * model.attn_tokens(S) * (w_lat if fw.attention_mode == "tp" else w_lat / d_kv)
            else:
                msg_SP = B * model.attn_tokens(S) * (2 * H_kv / d_kv) * b_kv
            t_SP = _cost("SP", "all_gather", msg_SP, SP)
        else:
            msg_SP = B * (H / d_kv) * b
            t_SP = _cost("SP", "all_reduce", msg_SP, SP)
    else:
        t_SP = 0.0
        msg_SP = 0.0

    if model.moe is not None:
        L_moe = model.moe.n_moe_layers if model.moe.n_moe_layers else L
    else:
        L_moe = 0

    t_comm_stage = aggregate_per_stage(
        L=L, L_moe=L_moe, PP=PP,
        n_TP=n_TP, t_TP=t_TP,
        n_SP=n_SP, t_SP=t_SP,
        n_EP=n_EP, t_EP=t_EP,
        t_PP=t_PP,
    )

    # DP-attention adjustment (decode.md §5.3 + §5.5): one of the n_TP per-layer
    # all-reduces is the attention output AR — under attention_mode="dp",
    # replace it with the (cheaper) AG. Saving per layer = (t_TP_AR − t_TP_AG);
    # applied across L/PP layers per stage.
    if t_TP_AG > 0.0 and t_TP > 0.0:
        t_comm_stage += (L / PP) * (t_TP_AG - t_TP)

    # Scatter-direct (decode.md §5.2): under DP-attn + scatter-direct, MoE
    # layers fire neither the pre-MoE TP all-gather nor the post-MoE TP
    # all-reduce — the dispatch operates on per-rank sharded tokens and the
    # combine returns expert outputs to the same per-rank shards. Subtract
    # the MoE-layer TP contribution that aggregate_per_stage and the AG-AR
    # adjustment above counted uniformly.
    if scatter_direct and L_moe > 0:
        per_moe_layer_tp = (n_TP - 1) * t_TP + t_TP_AG  # 1 AR + 1 AG under DP-attn
        t_comm_stage -= (L_moe / PP) * per_moe_layer_tp

    return CommResults(
        msg_PP_bytes=msg_PP,
        msg_TP_bytes=msg_TP,
        msg_EP_bytes=msg_EP,
        msg_SP_bytes=msg_SP,
        t_PP=t_PP,
        t_TP=t_TP,
        t_EP=t_EP,
        t_SP=t_SP,
        t_comm_stage=t_comm_stage,
    )


# ────────────────────────────────────────────────────────────
# Decode Latency (decode.md §6)
# ────────────────────────────────────────────────────────────

def compute_latency(
    model: LlmModelSpec,
    system: SystemSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
    framework: FrameworkSpec,
    flops: FlopsResults,
    traffic: TrafficResults,
    comm: CommResults,
) -> LatencyResults:
    """Per-token latency and throughput (seconds, tokens/s).

    The per-stage roofline gives the cost of one pipeline stage processing
    one microbatch of `mb = decode_microbatch(B, PP)` sequences. A token is
    not produced until its microbatch has crossed all PP stages and the LM
    head, so the inter-token time is the traversal sum, not one stage:
        t_stage_with_kernel = t_stage_GPU(mb) + max(0, t_kernel - ρ_kernel · t_stage_GPU(mb))
        t_step_base = PP · t_stage_with_kernel + t_LM(mb)
        t_step_user = t_step_base + max(0, t_step_seq
                                          - ρ_seq · t_step_base)
    Throughput is B tokens per t_step_user: the PP stages work on
    different microbatches concurrently, which is what PP buys. It does
    not shorten any one sequence's path through the L layers.

    `t_stage_GPU` is the GPU-side compute + comm time (with optional Tensor
    Core efficiency derate at small microbatch). `t_kernel` is the per-round
    CPU dispatch budget (decode.md §7.1). The two are
    composed via `kernel_overlap_factor` ρ_kernel, the fraction of the GPU stage
    time behind which launch time can hide: ρ_kernel=1 gives `max(t_stage_GPU,
    t_kernel)`, ρ_kernel=0 strict serialization.
    `t_step_seq` is the per-sequence serving runtime overhead (decode.md §7.3).
    Gross host work is `t_step_seq = (c_orch_us + c_seq_us · B) · 1e-6`, a
    per-step floor plus a per-sequence term;
    composition via `seq_overlap_factor` ρ_seq uses the same physics
    as ρ_kernel — under CUDA-Graph replay (ρ_seq = 1, default) the CPU runs
    ahead and host work hides behind GPU compute until it exceeds `t_step_base`.
    """
    B = tuner.B_decode
    PP = partition.PP

    # Precision-aware compute peak: peak_flops_TF in the system spec is
    # FP16 dense per chip; linear byte scaling lifts to the model's
    # working precision (FP8 / FP4 / INT8 / INT4). See
    # effective_peak_flops_TF docstring for the convention.
    R_gpu = effective_peak_flops_TF(system, model.bytes_per_param) * TB_TO_FLOPS

    # Per-stage-pass roofline: mb tokens computed, the stage's weights
    # loaded once. (`flops`, `traffic` and `comm` are already per pass.)
    # `t_mem` opens the legacy single-bandwidth term over device memory tiers
    # per sram.md §2.1: t_mem(B) = Σ_i (T_θ,i + B·T_KV,i) / BW_eff,i.
    # Resolves placement (greedy "auto" or operator-pinned) before summing.
    # On a single-tier device with eta_beta=1.0 (PR1 legacy shim), this
    # collapses to T_step_eff / BW_mem — bitwise identical to pre-PR2.
    #
    # Where bytes live is decided by what must be *resident* — the full
    # weight footprint (every expert, plus the embedding) and the KV cache
    # of all B in-flight sequences — not by what one step happens to read.
    # The per-step weight traffic is then charged to each tier in
    # proportion to the weights that tier holds. On a single-tier device
    # the proportion is exactly 1 and this reduces to T_theta / BW.
    tiers = system.device.get_tiers()
    # Only the vocabulary matrix a step reads (the LM head) enters the
    # placement: an untied input embedding is held but never read in bulk, so
    # it is taken to sit in whatever capacity is left and must not displace
    # layer weights from a fast tier. The capacity check (memory_model) does
    # count it.
    M_theta_resident = (
        dense_weight_bytes(model, partition, framework)
        + moe_weight_bytes(model, partition, framework)
        + embedding_bytes(model, partition)
    )
    residency = resolve_placement(
        T_theta_device=M_theta_resident,
        T_kv_per_request_device=traffic.T_kv_token,
        B=kv_batch(group_batch(max(1, B), partition, framework), partition, framework),
        tiers=tiers,
        placement=tuner.placement,
    )
    placement = PlacementResult(
        weights_per_tier=[
            traffic.T_theta * (w_i / M_theta_resident) if M_theta_resident > 0 else 0.0
            for w_i in residency.weights_per_tier
        ],
        kv_per_request_per_tier=list(residency.kv_per_request_per_tier),
    )
    t_compute = flops.F_step_device / R_gpu

    # Tensor Core efficiency derate at small microbatch.
    # η_TC ramps from ~0 at mb=1 (FP8 below the wgmma M=64 floor) to ~1
    # at mb ≥ 4·tile (compute-bound peak). curve=None ⇒ η_TC=1 (legacy).
    mb = max(1.0, decode_microbatch(B, PP))
    # Sequences one device group handles per pass (== mb unless the
    # orthogonal layout divides the batch across EP groups).
    mb_dev = group_batch(mb, partition, framework)
    eta_TC = _eta_TC_at_mb(system.device.tensor_core_efficiency, mb_dev)
    t_compute_eff = t_compute / eta_TC if eta_TC > 0 else float("inf")

    # B-dependent HBM sustained bandwidth derate (decode.md §6.2; Phase F:
    # curve lives on DeviceSpec. Opt-in via system.device.bw_efficiency,
    # otherwise η_β(B) = 1 and t_mem is unchanged).
    eta_beta_B = _eta_beta_at_B(system.device.bw_efficiency, mb_dev)
    kv_mb = kv_batch(mb_dev, partition, framework)
    t_mem = t_mem_from_placement(
        placement, B=kv_mb, tiers=tiers,
        eta_beta_curve_factor=eta_beta_B,
    )
    t_local = max(t_compute_eff, t_mem)

    t_comm = comm.t_comm_stage
    rho = framework.comm_overlap_factor
    t_stage = t_local + max(0.0, t_comm - rho * t_local)

    # Per-microbatch per-stage CPU dispatch budget (decode.md §7.1).
    # Composed with t_stage via ρ_kernel: full overlap (default) ⇒ max(...);
    # zero overlap ⇒ t_stage + t_kernel. EP launches only fire on MoE layers
    # (mirrors the L_moe/PP factor in §5.5's t_comm formula).
    if model.moe is not None:
        L_moe_total = model.moe.n_moe_layers if model.moe.n_moe_layers else model.L
    else:
        L_moe_total = 0
    t_kernel = _t_kernel_per_microbatch(model.L, L_moe_total, tuner, framework, partition)
    rho_kernel = framework.kernel_overlap_factor
    # Base + unhidden-overflow form (same pattern as compute/comm overlap in
    # decode.md §6.2). GPU work is the base; SW dispatch overlaps for
    # ρ_kernel · t_stage; any remainder serializes after.
    #   ρ_kernel = 1 → t_stage + max(0, t_kernel - t_stage) = max(t_stage, t_kernel)
    #             (SW fully hidden when t_stage >= t_kernel; SW-bound floor otherwise)
    #   ρ_kernel = 0 → t_stage + t_kernel (no overlap, costs add)
    t_stage_with_kernel = t_stage + max(0.0, t_kernel - rho_kernel * t_stage)

    # A token crosses every stage once: PP stage passes per token.
    pp_traversal_factor = float(max(1, PP))
    # Diagnostic only — stages per in-flight microbatch (1.0 = pipeline
    # full, > 1 = some stages idle). It no longer multiplies the step time.
    pp_bubble_factor = max(1.0, PP / max(1, B))

    # Top-tier memory bandwidth (also used by B* below). Multi-tier devices use
    # tier 0's effective bandwidth as a fast-tier proxy.
    BW_top = tiers[0].bandwidth_GBps * tiers[0].eta_beta * GB_TO_BYTES

    # LM head one-shot on stage PP-1 (decode.md §6.2 / §7.2):
    #   F_LM = 2·mb·H·V / D_emb
    #   T_LM = HVb/D_emb (weights, sharded by D_emb) + mb·V·b (logits, replicated)
    #   t_LM = max(F_LM/R_gpu, T_LM/BW_top)
    # Added outside the PP traversal because a token passes the LM head once,
    # after the last stage. D_emb = TP across all three
    # tp_ep_layout/attention_mode configurations (notation.md §1).
    V = model.vocab_size
    d_emb = D_emb(partition)
    B_eff = mb_dev
    F_lm = 2.0 * B_eff * model.H * V / d_emb
    # Projection weights at the weight precision; logit rows are activations.
    T_lm_weights = (model.H * V * model.bytes_per_param) / d_emb
    T_lm = T_lm_weights + B_eff * V * model.act_bytes()
    t_lm_compute = F_lm / R_gpu if R_gpu > 0 else 0.0
    if len(tiers) == 1:
        t_lm_mem = T_lm / BW_top if BW_top > 0 else 0.0
    else:
        # Multi-tier device: the projection matrix is read from wherever the
        # weights live (mostly the slow tier when they overflow the fast
        # one); the logit rows are written to the fast tier.
        t_lm_mem = (
            T_lm_weights * weights_seconds_per_byte(residency, tiers)
            + (B_eff * V * model.act_bytes()) / BW_top
        )
    t_LM = max(t_lm_compute, t_lm_mem)

    # Multi-tier device, "auto" weight placement: the pooled placement above
    # lets the vocabulary matrix (input embedding / LM head) claim fast-tier
    # capacity in proportion to its size, on every pipeline stage. But only
    # the last stage reads it, once per token, while layer weights are read
    # on every stage pass (and the first stage's embedding lookup is a
    # gather of a few rows). So also price the placement that gives the
    # matrix the lowest claim — layer weights and KV first, the matrix in
    # whatever capacity is left — and keep whichever gives the shorter step.
    # In a full pipeline the slowest stage sets the pace, so both variants
    # charge every stage alike. Capacity is unaffected: the end stages hold
    # the matrix either way (memory_model). Single-tier devices never reach
    # this branch, where residency does not change the time.
    M_vocab = embedding_bytes(model, partition)
    if len(tiers) > 1 and tuner.placement.weights_tier == "auto" and M_vocab > 0:
        M_layers = M_theta_resident - M_vocab
        B_res = kv_batch(group_batch(max(1, B), partition, framework), partition, framework)
        residency_alt = resolve_placement(
            T_theta_device=M_layers,
            T_kv_per_request_device=traffic.T_kv_token,
            B=B_res,
            tiers=tiers,
            placement=tuner.placement,
        )
        placement_alt = PlacementResult(
            weights_per_tier=[
                traffic.T_theta * (w_i / M_layers) if M_layers > 0 else 0.0
                for w_i in residency_alt.weights_per_tier
            ],
            kv_per_request_per_tier=list(residency_alt.kv_per_request_per_tier),
        )
        t_mem_alt = t_mem_from_placement(
            placement_alt, B=kv_mb, tiers=tiers,
            eta_beta_curve_factor=eta_beta_B,
        )
        t_local_alt = max(t_compute_eff, t_mem_alt)
        t_stage_alt = t_local_alt + max(0.0, t_comm - rho * t_local_alt)
        t_stage_with_kernel_alt = t_stage_alt + max(0.0, t_kernel - rho_kernel * t_stage_alt)
        t_lm_mem_alt = (
            T_lm_weights * spill_seconds_per_byte(M_vocab, residency_alt, B_res, tiers)
            + (B_eff * V * model.act_bytes()) / BW_top
        )
        t_LM_alt = max(t_lm_compute, t_lm_mem_alt)
        if (
            t_stage_with_kernel_alt * pp_traversal_factor + t_LM_alt
            < t_stage_with_kernel * pp_traversal_factor + t_LM
        ):
            t_mem, t_local, t_stage = t_mem_alt, t_local_alt, t_stage_alt
            t_stage_with_kernel = t_stage_with_kernel_alt
            t_lm_mem, t_LM = t_lm_mem_alt, t_LM_alt

    # Per-sequence serving runtime overhead (decode.md §7.3): host-side work
    # that scales linearly with active-sequence count B (block-table gather,
    # sampling, scheduler). **Composed with GPU work via seq_overlap_factor
    # ρ_seq** — under CUDA-Graph replay the CPU runs ahead and host work
    # hides behind GPU compute until it exceeds the per-step GPU time. Only the
    # *excess* over the per-step GPU window contributes to t_step_user.
    #   ρ_seq = 1 → max(0, t_step_seq - t_step_base)  (full overlap;
    #                   host work hides while CPU has GPU compute to amortize against)
    #   ρ_seq = 0 → t_step_seq  (eager-mode; host work always blocks)
    # Per-step gross host work: B-independent orchestration floor
    # (c_orch · 1) plus per-sequence inner loops (c_seq · B). Both pay
    # the same overlap composition against the per-step hardware window.
    t_step_seq = (framework.c_orch_us + framework.c_seq_us * max(1, B)) * 1e-6
    rho_seq = framework.seq_overlap_factor
    t_step_base = t_stage_with_kernel * pp_traversal_factor + t_LM
    t_step_user = t_step_base + max(0.0, t_step_seq - rho_seq * t_step_base)
    TPOT = t_step_user
    # t_step_seq is the GROSS per-step host work (= c_orch + c_seq · B).
    # Stored as-is in LatencyResults for plotting; the gated contribution
    # to t_step_user is the inline max(...) above — readers see the gap
    # between t_step_seq and t_step_user as the amount the overlap factor
    # hid behind GPU work.

    TPS_single = B / t_step_user if t_step_user > 0 else 0.0

    # Replica size depends on tp_ep_layout (notation.md §1): orthogonal uses
    # the full product PP·TP·EP·SP; co-located uses PP·max(TP,EP)·SP since TP
    # and EP share the same physical GPU set.
    replica_size = N_replica(partition, framework)
    # Stage blocks tile fabric islands, not the flat device count (multi-rack).
    DP = replicas_in_system(system, replica_size, stages=max(1, PP))
    TTPS = DP * TPS_single

    # B* crossover: batch size where the system transitions from
    # memory-bound to compute-bound. For multi-tier devices, sram.md §2.2
    # gives the exact two-tier form when weights and KV live on separate
    # tiers; the single-tier formula here matches that special case W=K=tier-0.
    # B* uses the per-token KV term (decode.md §4 B^star formula uses T_{KV,token}).
    # The roofline crossover is in microbatch terms; B = PP · mb when the
    # pipeline is full, so the replica-level crossover is PP times that.
    # Per in-flight sequence, a device's compute and KV read grow at the
    # group's share of the batch (1/D_batch) for attention, dense FFN and
    # KV, and at the full rate for the expert FFN.
    d_batch = D_batch(partition, framework)
    # Under DP-attention a rank runs the (unsharded) projections for 1/G_TP of
    # its group's sequences, so their slope per in-flight sequence is F_proj/G_TP.
    F_token_slope = flops.F_token_device
    if framework.attention_mode == "dp" and partition.TP > 1:
        F_token_slope -= attn_proj_flops_per_token(model, partition, framework) * (1.0 - 1.0 / partition.TP)
    if d_batch == 1:
        F_slope = F_token_slope
        T_kv_slope = traffic.T_kv_token
    else:
        F_moe_expert = moe_expert_flops_per_token(model, partition, framework)
        F_slope = (F_token_slope - F_moe_expert) / d_batch + F_moe_expert
        T_kv_slope = traffic.T_kv_token / d_batch
    denom = F_slope * BW_top - T_kv_slope * R_gpu
    B_star = (pp_traversal_factor * traffic.T_theta * R_gpu / denom) if denom > 0 else float("inf")

    # Speculative-decoding extension (decode.md §8). Compose verify-step
    # quantities by scaling compute and comm payloads by n_tok_verify;
    # weight traffic and KV traffic stay invariant (decode.md §8.3). When
    # n_tok_draft = 0, the verify quantities reduce to the vanilla ones.
    n_tok_draft = max(0, tuner.n_tok_draft)
    p_accept = max(0.0, min(1.0, tuner.p_accept))
    n_tok_verify = n_tok_draft + 1
    if n_tok_draft > 0 and p_accept > 0.0:
        # Truncated geometric expected accepted draft tokens (Leviathan eq. 5):
        #   E[N_accept] = Σ_{d=1..n_draft} p^d = p(1 - p^n_draft) / (1 - p)
        # plus 1 for the always-accepted target prediction.
        if p_accept < 1.0:
            E_accept = p_accept * (1.0 - p_accept ** n_tok_draft) / (1.0 - p_accept)
        else:
            E_accept = float(n_tok_draft)
        N_tok_per_step = 1.0 + E_accept
    else:
        N_tok_per_step = 1.0

    if n_tok_verify > 1:
        # Verify-step roofline: compute scales by n_tok_verify, memory
        # invariant (FlashAttention batched-Q assumption; decode.md §8.3).
        t_compute_verify_eff = t_compute_eff * n_tok_verify
        t_local_verify = max(t_compute_verify_eff, t_mem)
        # Communication payload scales linearly with n_tok_verify; α-side
        # startup terms are unchanged. The §5.5 cost is dominated by the
        # β-side at production batch sizes, so this scaling captures the
        # leading-order effect.
        t_comm_verify = t_comm * n_tok_verify
        t_stage_verify = t_local_verify + max(0.0, t_comm_verify - rho * t_local_verify)
        t_stage_with_kernel_verify = t_stage_verify + max(0.0, t_kernel - rho_kernel * t_stage_verify)
        # LM head: compute scales by n_tok_verify (one logits projection
        # per query token), memory invariant. Per decode.md §8.3 this is
        # max(...) of the two — at typical FP4 vocab sizes the LM head
        # often stays memory-bound through small B even under speculation.
        t_lm_compute_verify = t_lm_compute * n_tok_verify
        t_LM_verify = max(t_lm_compute_verify, t_lm_mem)
        # t_step_seq fires once per verify step too; apply the same overlap
        # gate against the verify-step hardware window.
        t_step_base_verify = t_stage_with_kernel_verify * pp_traversal_factor + t_LM_verify
        t_step_user_verify = t_step_base_verify + max(0.0, t_step_seq - rho_seq * t_step_base_verify)
        TPOT_spec = t_step_user_verify / N_tok_per_step
    else:
        t_step_user_verify = t_step_user
        TPOT_spec = TPOT

    return LatencyResults(
        t_compute=t_compute,
        t_compute_eff=t_compute_eff,
        eta_TC=eta_TC,
        t_mem=t_mem,
        t_local=t_local,
        t_comm=t_comm,
        t_stage=t_stage,
        t_kernel=t_kernel,
        t_LM=t_LM,
        t_step_user=t_step_user,
        pp_bubble_factor=pp_bubble_factor,
        pp_traversal_factor=pp_traversal_factor,
        microbatch=mb,
        group_batch=mb_dev,
        TPS_single=TPS_single,
        TTPS=TTPS,
        B=B,
        TPOT=TPOT,
        B_star=B_star,
        N_tok_per_step=N_tok_per_step,
        t_step_user_verify=t_step_user_verify,
        TPOT_spec=TPOT_spec,
        t_step_seq=t_step_seq,
    )
