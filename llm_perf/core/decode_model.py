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

from ..specs.model_spec import LlmModelSpec
from ..specs.system_spec import SystemSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.tuner_spec import TuningSpec
from ..utils import GB_TO_BYTES, TB_TO_FLOPS
from .memory_placement import resolve_placement, t_mem_from_placement
from .primitives import (
    dense_weight_bytes,
    moe_weight_bytes,
    kv_bytes_per_seq,
    linear_flops_per_token,
    aggregate_per_stage,
    cost_collective,
)


# ────────────────────────────────────────────────────────────
# SW-overhead helpers (kernel_launch_overhead.md §5)
# ────────────────────────────────────────────────────────────

def _eta_TC_at_mb(curve: Optional[Dict[int, float]], mb: float) -> float:
    """Piecewise-linear lookup of Tensor Core efficiency at microbatch `mb`.

    None ⇒ 1.0 always (legacy: no compute derate).
    `mb` clamps to the curve's [min_key, max_key] range.
    """
    if curve is None or not curve:
        return 1.0
    keys = sorted(curve.keys())
    if mb <= keys[0]:
        return float(curve[keys[0]])
    if mb >= keys[-1]:
        return float(curve[keys[-1]])
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= mb <= hi:
            t = (mb - lo) / (hi - lo)
            return float(curve[lo]) + t * (float(curve[hi]) - float(curve[lo]))
    return 1.0  # unreachable; defensive


def _t_SW_per_round(
    L: int,
    tuner: TuningSpec,
    partition: PartitionSpec,
) -> float:
    """Per-round CPU dispatch budget on each device.

        t_SW = L · (k_compute + k_per_collective · n_collectives_per_layer) · τ_launch
             + 2 · PP · k_pp_hop · τ_launch    (PP send/recv launches)

    `n_collectives_per_layer` counts only collectives that actually fire for
    the current shape (zero when the corresponding parallelism axis is 1).

    The PP-hop term comes from steady-state inflight batching: each stage
    handles ≈ 2 P2P kernels per microbatch (1 recv + 1 send) × PP
    microbatches per round = 2·PP launches per stage per round. Inert when
    PP=1 (no inter-stage hops). Edge stages do only one direction; the
    formula uses the middle-stage 2× factor for simplicity (off by one
    PP·τ on edge stages, negligible at PP >> 1).

    Returns 0 when kernel_launch_us is 0 (legacy behavior).
    """
    tau_us = tuner.kernel_launch_us
    if tau_us <= 0.0:
        return 0.0
    k_c = tuner.kernels_per_layer_compute
    k_coll = tuner.kernels_per_collective_call
    n_TP = tuner.n_TP_collectives if partition.TP > 1 else 0
    n_EP = tuner.n_EP_collectives if max(1, partition.EP) > 1 else 0
    n_SP = tuner.n_SP_collectives if partition.SP > 1 else 0
    k = k_c + k_coll * (n_TP + n_EP + n_SP)
    t_layer = L * k * tau_us * 1e-6
    n_pp_per_round = 2 * partition.PP if partition.PP > 1 else 0
    t_pp = n_pp_per_round * tuner.kernels_per_pp_hop * tau_us * 1e-6
    return t_layer + t_pp


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
    T_theta: float
    T_kv: float
    T_token_eff: float
    T_step_eff: float


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
    t_SW: float               # per-round CPU dispatch budget = L · k · τ_launch
    t_step_user: float
    pp_bubble_factor: float
    TPS_single: float
    TTPS: float
    B: int
    TPOT: float
    B_star: float


# ────────────────────────────────────────────────────────────
# Decode FLOPs (documentation/modeling/decode.md §3)
# ────────────────────────────────────────────────────────────

def compute_flops(
    model: LlmModelSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
) -> FlopsResults:
    """Per-device decode FLOPs per token (and per step)."""
    L = model.L
    H = model.H
    PP = partition.PP
    TP = partition.TP
    SP = partition.SP
    S = tuner.S_decode
    B = tuner.B_decode

    # Linear FLOPs per token (proj + FFN + MoE router)
    F_linear_per_token = linear_flops_per_token(model, partition)
    # Decode attention: 4·S·H/(TP·SP) per token per layer
    F_attn_per_token = (L / PP) * (4 * S * H) / (TP * SP)

    F_token_device = F_linear_per_token + F_attn_per_token
    F_layer_per_device = F_token_device / (L / PP) if L > 0 else 0.0
    F_step_device = B * F_token_device

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
) -> TrafficResults:
    """Per-token HBM traffic per device (weights + KV cache read)."""
    S = tuner.S_decode
    B = tuner.B_decode

    # Parameter traffic (embedding is outside the forward path by convention).
    T_theta = dense_weight_bytes(model, partition) + moe_weight_bytes(model, partition)
    # KV read traffic for one sequence of S context tokens.
    T_kv = kv_bytes_per_seq(model, partition, S)

    T_token_eff = T_theta + T_kv
    # Batched step: weights loaded once, KV read per sequence in the batch.
    T_step_eff = T_theta + B * T_kv

    return TrafficResults(
        T_theta=T_theta,
        T_kv=T_kv,
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
) -> CommResults:
    """Per-stage decode communication time (seconds)."""
    H = model.H
    H_kv = model.H_kv()
    L = model.L
    PP = partition.PP
    TP = partition.TP
    EP = max(1, partition.EP)
    SP = partition.SP
    S = tuner.S_decode
    B = max(1, tuner.B_decode)
    b = model.bytes_per_param

    n_TP = tuner.n_TP_collectives
    n_EP = tuner.n_EP_collectives
    n_SP = tuner.n_SP_collectives

    if model.moe is not None:
        N_exp = max(1, model.moe.n_experts)
        EP = min(EP, N_exp)
        k = model.moe.k_active
    else:
        EP = 1
        k = 1

    # Decode reads the per-phase fields; falls back to legacy single-knob
    # if the per-phase field is at default and the legacy field was overridden.
    tp_algorithm = getattr(tuner, "tp_algorithm_decode",
                           getattr(tuner, "tp_algorithm", "ring")).lower()
    ep_algorithm = getattr(tuner, "ep_algorithm_decode",
                           getattr(tuner, "ep_algorithm", "ring")).lower()
    torus_alg = getattr(tuner, "torus_algorithm", "ring").lower()
    inc_enabled = bool(getattr(tuner, "inc_enabled", True))

    if tp_algorithm == "auto" or ep_algorithm == "auto":
        raise ValueError(
            "TuningSpec has algorithm='auto' for decode; resolve via "
            "core.collective_algo_opt.optimize_collective_algorithms(...) "
            "before InferenceCalculator.run()."
        )

    def _cost(coll: str, op: str, M: float, G: int, alg: str = "ring") -> float:
        return cost_collective(
            system.get_tier_chain(coll), op, M, G,
            algorithm=alg, torus_algorithm=torus_alg,
            inc_enabled=inc_enabled,
        )

    # PP: shard-preserving hop of B tokens × (H/TP) activation bytes.
    # A single-stage pipeline has no inter-stage forward.
    if PP > 1:
        msg_PP = B * (H / TP) * b
        t_PP = _cost("PP", "p2p", msg_PP, 2)
    else:
        msg_PP = 0.0
        t_PP = 0.0

    # EP: 2-pass all-to-all (Dispatch + Combine) over k·H activation bytes
    if EP > 1:
        msg_EP = B * k * H * b
        t_EP = _cost("EP", "moe_a2a", msg_EP, EP, alg=ep_algorithm)
    else:
        t_EP = 0.0
        msg_EP = 0.0

    # TP: 2-pass all-reduce of B·H activation bytes
    if TP > 1:
        msg_TP = B * H * b
        t_TP = _cost("TP", "all_reduce", msg_TP, TP, alg=tp_algorithm)
    else:
        t_TP = 0.0
        msg_TP = 0.0

    # SP: 1-pass ring all-gather over the full KV (per-rank gathered
    # output convention — collective_cost.py §6 calls this "M = G·shard").
    if SP > 1:
        msg_SP = B * S * (2 * H_kv / TP) * b
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
    flops: FlopsResults,
    traffic: TrafficResults,
    comm: CommResults,
) -> LatencyResults:
    """Per-token latency and throughput (seconds, tokens/s).

    The per-stage roofline gives the cost of one pipeline stage processing
    the current batch. For a user observing inter-token latency we apply a
    pipeline-bubble correction when B < PP:
        t_step_user = max(t_stage_GPU, t_SW) · max(1, PP / B).

    `t_stage_GPU` is the GPU-side compute + comm time (with optional Tensor
    Core efficiency derate at small microbatch). `t_SW` is the per-round
    CPU dispatch budget (kernel_launch_overhead.md §5). The two are
    composed via `sw_overlap_factor` ρ_SW: ρ_SW=1 means SW is fully hidden
    by GPU work (just `max`), ρ_SW=0 means strict serialization.
    """
    B = tuner.B_decode
    PP = partition.PP

    R_gpu = system.device.peak_flops_TF * TB_TO_FLOPS

    # Step-level roofline: B tokens computed, weights loaded once.
    # `t_mem` opens the legacy single-bandwidth term over device memory tiers
    # per sram.md §2.1: t_mem(B) = Σ_i (T_θ,i + B·T_KV,i) / BW_eff,i.
    # Resolves placement (greedy "auto" or operator-pinned) before summing.
    # On a single-tier device with eta_beta=1.0 (PR1 legacy shim), this
    # collapses to T_step_eff / BW_mem — bitwise identical to pre-PR2.
    tiers = system.device.get_tiers()
    placement = resolve_placement(
        T_theta_device=traffic.T_theta,
        T_kv_per_request_device=traffic.T_kv,
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
    eta_TC = _eta_TC_at_mb(tuner.tensor_core_efficiency, mb)
    t_compute_eff = t_compute / eta_TC if eta_TC > 0 else float("inf")

    t_mem = t_mem_from_placement(placement, B=max(1, B), tiers=tiers)
    t_local = max(t_compute_eff, t_mem)

    t_comm = comm.t_comm_stage
    rho = tuner.overlap_factor
    t_stage = t_local + max(0.0, t_comm - rho * t_local)

    # Per-round CPU dispatch budget (kernel_launch_overhead.md §5).
    # Composed with t_stage via ρ_SW: full overlap (default) ⇒ max(...);
    # zero overlap ⇒ t_stage + t_SW.
    t_SW = _t_SW_per_round(model.L, tuner, partition)
    rho_SW = tuner.sw_overlap_factor
    t_stage_with_SW = max(t_stage, rho_SW * t_stage + (1.0 - rho_SW) * (t_stage + t_SW))
    # Simpler reading at the two boundary points:
    #   ρ_SW = 1 → t_stage_with_SW = t_stage  (SW fully hidden by GPU work)
    #   ρ_SW = 0 → t_stage_with_SW = max(t_stage, t_stage + t_SW) = t_stage + t_SW
    # For the strict "max" interpretation when SW is the bottleneck, callers
    # who want max(t_stage, t_SW) instead of additive should set ρ_SW such
    # that the formula above matches; the additive form is the safer upper
    # bound and is what we ship as default behavior away from full overlap.
    t_stage_with_SW = max(t_stage_with_SW, t_SW)

    pp_bubble_factor = max(1.0, PP / max(1, B))
    t_step_user = t_stage_with_SW * pp_bubble_factor
    TPOT = t_step_user

    TPS_single = B / t_step_user if t_step_user > 0 else 0.0

    replica_size = partition.PP * partition.TP * max(1, partition.EP) * partition.SP
    DP = system.num_devices // replica_size
    TTPS = DP * TPS_single

    # B* crossover: batch size where the system transitions from
    # memory-bound to compute-bound. For multi-tier devices this uses tier 0's
    # effective bandwidth as a representative fast-tier proxy (sram.md §2.2
    # gives the exact two-tier form when weights and KV live on separate
    # tiers; the single-tier formula here matches that special case W=K=tier-0).
    BW_top = tiers[0].bandwidth_GBps * tiers[0].eta_beta * GB_TO_BYTES
    denom = flops.F_token_device * BW_top - traffic.T_kv * R_gpu
    B_star = (traffic.T_theta * R_gpu / denom) if denom > 0 else float("inf")

    return LatencyResults(
        t_compute=t_compute,
        t_compute_eff=t_compute_eff,
        eta_TC=eta_TC,
        t_mem=t_mem,
        t_local=t_local,
        t_comm=t_comm,
        t_stage=t_stage,
        t_SW=t_SW,
        t_step_user=t_step_user,
        pp_bubble_factor=pp_bubble_factor,
        TPS_single=TPS_single,
        TTPS=TTPS,
        B=B,
        TPOT=TPOT,
        B_star=B_star,
    )
