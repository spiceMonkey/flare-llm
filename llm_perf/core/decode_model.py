"""Decode-phase performance model — consolidates flops + traffic + comm + latency.

Mirrors the shape of `prefill_model.py`: four pure functions on typed
dataclasses, each returning a small result dataclass. Internally the
phase-agnostic physics (weight/KV footprint, linear FLOPs, collective
cost) lives in `core/primitives/`; this module wires those primitives
together with decode-specific pieces (attention that scales with S, not
S²; message sizes that are B·H·b, not S·H·b).

Documentation: `documentation/modeling/decode.md`.

Public surface (preserved from the pre-refactor four-file split):
  - compute_flops(model, partition, tuner) → FlopsResults
  - compute_traffic(model, partition, tuner) → TrafficResults
  - compute_comm(model, system, partition, tuner) → CommResults
  - compute_latency(model, system, partition, tuner, flops, traffic, comm) → LatencyResults
"""

from dataclasses import dataclass
from typing import Dict, Optional

from ..specs.framework_spec import FrameworkSpec
from ..specs.model_spec import LlmModelSpec
from ..specs.system_spec import SystemSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.tuner_spec import TuningSpec
from ..utils import GB_TO_BYTES, TB_TO_FLOPS
from .memory_placement import resolve_placement, t_mem_from_placement
from .primitives import (
    attn_proj_flops_per_token,
    ffn_flops_per_token,
    dense_weight_bytes,
    moe_weight_bytes,
    moe_weight_traffic_bytes,
    kv_bytes_per_seq,
    linear_flops_per_token,
    aggregate_per_stage,
    cost_collective,
    p2p_hop,
    assign_tier_per_axis,
    tier_at,
    D_kv,
    D_emb,
    G_TP,
    G_EP,
    N_replica,
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

    **Known limitation**: d-Matrix MXINT4 throughput is 4× MXINT8 rather
    than the 2× linear byte scaling predicts (block-sparse acceleration in
    the INT4 path). With FP16 baseline = 150 TF/chiplet, the framework
    computes 600 TF/chiplet for MXINT4, but the published peak is
    1200 TF/chiplet — a 2× under-estimate on d-Matrix INT4 / FP4 only.
    Linear byte scaling holds for every other modeled architecture
    (NVIDIA Hopper / Blackwell, TPU v5p / v6e, Groq LPU).
    """
    bpp = max(1e-9, bytes_per_param)
    return system.device.peak_flops_TF * system.device.peak_flops_eta * (_FP16_BYTES / bpp)


# ────────────────────────────────────────────────────────────
# SW-overhead helpers (kernel_launch_overhead.md §5)
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
    T_theta: float          # per-device weight traffic per step (B-independent for dense; B-aware for MoE)
    T_kv_token: float       # per-device, per-token KV traffic (one active sequence's KV-history read, all stage layers); matches decode.md §2.3 T_{KV,token}
    T_kv_device: float      # per-device, per-step KV traffic aggregated over B sequences: B * T_kv_token; matches decode.md §2.3 T_{KV,device}(B)
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
    t_compute: float          # raw roofline compute time = F_step / R_gpu
    t_compute_eff: float      # Tensor-Core-derated compute time (= t_compute / η_TC)
    eta_TC: float             # Tensor Core efficiency factor at this mb (1.0 = peak)
    t_mem: float
    t_local: float            # max(t_compute_eff, t_mem) — memory-or-compute-bound roofline
    t_comm: float
    t_stage: float            # GPU-only step time (compute + comm + overlap)
    t_kernel: float               # per-round CPU dispatch budget = L · k · τ_launch
    t_LM: float               # LM head one-shot latency on stage PP-1 (decode.md §6.2)
    t_step_user: float
    pp_bubble_factor: float
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
    # Per-sequence serving runtime overhead (decode.md §7.2). GROSS per-step
    # host cost: c_serving_per_seq_us * B * 1e-6. Parallel to t_kernel which also
    # stores the gross dispatch budget — both are diagnostic surfaces. The
    # actual contribution to t_step_user is the OVERLAP-GATED form
    # max(0, t_serving - serving_overlap_factor * t_GPU_step), applied
    # internally; readers comparing the t_serving curve against t_step,user
    # in plots see the overlap gate's hidden portion as the gap.
    t_serving: float = 0.0


# ────────────────────────────────────────────────────────────
# Decode FLOPs (documentation/modeling/decode.md §3)
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
    B = tuner.B_decode

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
            model, partition, framework, S, framework.mla_mode
        )
        F_attn_sv_per_token = (L / PP) * F_sv_per_layer / SP
    else:
        F_attn_sv_per_token = (L / PP) * (4 * S * H) / (D_kv(partition, framework) * SP)

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
    if framework.attention_mode == "dp" and partition.TP > 1:
        B_attn_proj = B / partition.TP
    else:
        B_attn_proj = B
    F_step_device = (
        B_attn_proj * F_attn_proj_per_token
        + B * F_attn_sv_per_token
        + B * F_ffn_per_token
    )

    return FlopsResults(
        F_token_device=F_token_device,
        F_layer_per_device=F_layer_per_device,
        F_step_device=F_step_device,
    )


# ────────────────────────────────────────────────────────────
# Decode Traffic (documentation/modeling/decode.md §4)
# ────────────────────────────────────────────────────────────

def compute_traffic(
    model: LlmModelSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
    framework: FrameworkSpec,
) -> TrafficResults:
    """Per-step HBM traffic per device (weights + KV cache read).

    Dense weights (attention, dense-FFN) are read every step → traffic
    equals footprint. MoE expert weights are read only for the experts
    the current batch actually touches → traffic uses an expectation-
    of-touched-experts correction that converges to the footprint at
    large B. See `core/primitives/weight_quantities.py` for the formula
    and the uniform-routing assumption.
    """
    S = tuner.S_decode
    B = tuner.B_decode

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
    # Device-aggregate per step: B sequences each read their own KV history
    # (decode.md §2.3 T_{KV,device}(B) = B · T_{KV,token}).
    T_kv_device = B * T_kv_token

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
# Decode Communication (documentation/modeling/decode.md §5)
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
    B = max(1, tuner.B_decode)
    b = model.bytes_per_param

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
            inc_enabled=inc_enabled,
        )

    # PP: shard-preserving hop of B tokens × (H/D_kv) activation bytes.
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
            msg_EP = (B / g_TP) * k * H * b
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

    # SP: 1-pass ring all-gather over the full KV (per-rank gathered
    # output convention — collective_cost.py §6 calls this "M = G·shard").
    # KV head-or-seq divisor uses D_kv from notation.md §1.
    if SP > 1:
        msg_SP = B * S * (2 * H_kv / d_kv) * b
        t_SP = _cost("SP", "all_gather", msg_SP, SP)
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
# Decode Latency (documentation/modeling/decode.md §6)
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
    the current batch. For a user observing inter-token latency we apply a
    pipeline-bubble correction when B < PP:
        t_GPU_step = max(t_stage_GPU, t_kernel) · max(1, PP / B) + t_LM
        t_step_user = t_GPU_step + max(0, t_serving_gross
                                          - ρ_serving · t_GPU_step)

    `t_stage_GPU` is the GPU-side compute + comm time (with optional Tensor
    Core efficiency derate at small microbatch). `t_kernel` is the per-round
    CPU dispatch budget (kernel_launch_overhead.md §5). The two are
    composed via `kernel_overlap_factor` ρ_kernel: ρ_kernel=1 means SW is fully hidden
    by GPU work (just `max`), ρ_kernel=0 means strict serialization.
    `t_serving` is the per-sequence serving runtime overhead (decode.md §7.2).
    Gross host work is `t_serving_gross = c_serving_per_seq_us · B · 1e-6`;
    composition via `serving_overlap_factor` ρ_serving uses the same physics
    as ρ_kernel — under CUDA-Graph replay (ρ_serving = 1, default) the CPU runs
    ahead and host work hides behind GPU compute until it exceeds `t_GPU_step`.
    """
    B = tuner.B_decode
    PP = partition.PP

    # Precision-aware compute peak: peak_flops_TF in the system spec is
    # FP16 dense per chip; linear byte scaling lifts to the model's
    # working precision (FP8 / FP4 / INT8 / INT4). See
    # effective_peak_flops_TF docstring for the convention.
    R_gpu = effective_peak_flops_TF(system, model.bytes_per_param) * TB_TO_FLOPS

    # Step-level roofline: B tokens computed, weights loaded once.
    # `t_mem` opens the legacy single-bandwidth term over device memory tiers
    # per sram.md §2.1: t_mem(B) = Σ_i (T_θ,i + B·T_KV,i) / BW_eff,i.
    # Resolves placement (greedy "auto" or operator-pinned) before summing.
    # On a single-tier device with eta_beta=1.0 (PR1 legacy shim), this
    # collapses to T_step_eff / BW_mem — bitwise identical to pre-PR2.
    tiers = system.device.get_tiers()
    placement = resolve_placement(
        T_theta_device=traffic.T_theta,
        T_kv_per_request_device=traffic.T_kv_token,
        B=max(1, B),
        tiers=tiers,
        placement=tuner.placement,
    )
    t_compute = flops.F_step_device / R_gpu

    # Tensor Core efficiency derate at small microbatch.
    # mb = B / PP (microbatch size in steady-state inflight pipeline).
    # η_TC ramps from ~0 at mb=1 (FP8 below the wgmma M=64 floor) to ~1
    # at mb ≥ 4·tile (compute-bound peak). curve=None ⇒ η_TC=1 (legacy).
    mb = max(1, B) / max(1, PP)
    eta_TC = _eta_TC_at_mb(system.device.tensor_core_efficiency, mb)
    t_compute_eff = t_compute / eta_TC if eta_TC > 0 else float("inf")

    # B-dependent HBM sustained bandwidth derate (decode.md §6.2; Phase F:
    # curve lives on DeviceSpec. Opt-in via system.device.bw_efficiency,
    # otherwise η_β(B) = 1 and t_mem is unchanged).
    eta_beta_B = _eta_beta_at_B(system.device.bw_efficiency, max(1, B))
    t_mem = t_mem_from_placement(
        placement, B=max(1, B), tiers=tiers,
        eta_beta_curve_factor=eta_beta_B,
    )
    t_local = max(t_compute_eff, t_mem)

    t_comm = comm.t_comm_stage
    rho = framework.comm_overlap_factor
    t_stage = t_local + max(0.0, t_comm - rho * t_local)

    # Per-microbatch per-stage CPU dispatch budget (kernel_launch_overhead.md §5).
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

    pp_bubble_factor = max(1.0, PP / max(1, B))

    # Top-tier memory bandwidth (also used by B* below). Multi-tier devices use
    # tier 0's effective bandwidth as a fast-tier proxy.
    BW_top = tiers[0].bandwidth_GBps * tiers[0].eta_beta * GB_TO_BYTES

    # LM head one-shot on stage PP-1 (decode.md §6.2 / §7.2):
    #   F_LM,step = 2·B·H·V / D_emb
    #   T_LM,step = HVb/D_emb (weights, sharded by D_emb) + B·V·b (logits, replicated)
    #   t_LM = max(F_LM/R_gpu, T_LM/BW_top)
    # Added outside γ_pp because the LM head fires once per step regardless of
    # bubble depth (it is not pipelined across PP stages). D_emb = TP across all
    # three tp_ep_layout/attention_mode configurations (notation.md §1).
    V = model.vocab_size
    d_emb = D_emb(partition)
    b = model.bytes_per_param
    B_eff = max(1, B)
    F_lm = 2.0 * B_eff * model.H * V / d_emb
    T_lm = (model.H * V * b) / d_emb + B_eff * V * b
    t_lm_compute = F_lm / R_gpu if R_gpu > 0 else 0.0
    t_lm_mem = T_lm / BW_top if BW_top > 0 else 0.0
    t_LM = max(t_lm_compute, t_lm_mem)

    # Per-sequence serving runtime overhead (decode.md §7.2): host-side work
    # that scales linearly with active-sequence count B (block-table gather,
    # sampling, scheduler). **Composed with GPU work via serving_overlap_factor
    # ρ_serving** — under CUDA-Graph replay the CPU runs ahead and host work
    # hides behind GPU compute until it exceeds the per-step GPU time. Only the
    # *excess* over the per-step GPU window contributes to t_step_user.
    #   ρ_serving = 1 → max(0, t_serving_gross - t_GPU_step)  (full overlap;
    #                   host work hides while CPU has GPU compute to amortize against)
    #   ρ_serving = 0 → t_serving_gross  (eager-mode; host work always blocks)
    t_serving_gross = framework.c_serving_per_seq_us * max(1, B) * 1e-6
    rho_serving = framework.serving_overlap_factor
    t_GPU_step = t_stage_with_kernel * pp_bubble_factor + t_LM
    t_serving_excess = max(0.0, t_serving_gross - rho_serving * t_GPU_step)

    t_step_user = t_GPU_step + t_serving_excess
    TPOT = t_step_user
    # Diagnostic surface (parallel to t_kernel): expose the GROSS per-step host
    # cost in LatencyResults so plots and breakdown tables show the per-
    # sequence work as it scales with B, even when the overlap gate clips
    # its contribution to t_step_user to zero. The gated contribution lives
    # in t_step_user; readers see the gap between the t_serving curve and
    # t_step,user as the amount the overlap gate has hidden.
    t_serving = t_serving_gross

    TPS_single = B / t_step_user if t_step_user > 0 else 0.0

    # Replica size depends on tp_ep_layout (notation.md §1): orthogonal uses
    # the full product PP·TP·EP·SP; co-located uses PP·max(TP,EP)·SP since TP
    # and EP share the same physical GPU set.
    replica_size = N_replica(partition, framework)
    DP = system.num_devices // replica_size
    TTPS = DP * TPS_single

    # B* crossover: batch size where the system transitions from
    # memory-bound to compute-bound. For multi-tier devices, sram.md §2.2
    # gives the exact two-tier form when weights and KV live on separate
    # tiers; the single-tier formula here matches that special case W=K=tier-0.
    # B* uses the per-token KV term (decode.md §4 B^star formula uses T_{KV,token}).
    denom = flops.F_token_device * BW_top - traffic.T_kv_token * R_gpu
    B_star = (traffic.T_theta * R_gpu / denom) if denom > 0 else float("inf")

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
        # t_serving fires once per verify step too; apply the same overlap
        # gate against the verify-step GPU window.
        t_GPU_step_verify = t_stage_with_kernel_verify * pp_bubble_factor + t_LM_verify
        t_serving_verify = max(0.0, t_serving_gross - rho_serving * t_GPU_step_verify)
        t_step_user_verify = t_GPU_step_verify + t_serving_verify
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
        TPS_single=TPS_single,
        TTPS=TTPS,
        B=B,
        TPOT=TPOT,
        B_star=B_star,
        N_tok_per_step=N_tok_per_step,
        t_step_user_verify=t_step_user_verify,
        TPOT_spec=TPOT_spec,
        t_serving=t_serving,
    )
