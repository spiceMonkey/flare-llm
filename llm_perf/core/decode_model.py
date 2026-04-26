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
  - compute_latency(system, partition, tuner, flops, traffic, comm) → LatencyResults
"""

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
    t_compute: float
    t_mem: float
    t_local: float
    t_comm: float
    t_stage: float
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

    # SP: 1-pass ring all-gather over KV shard
    if SP > 1:
        msg_SP = B * (S / SP) * (2 * H_kv / TP) * b
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
        t_step_user = t_stage · max(1, PP / B).
    """
    B = tuner.B_decode
    PP = partition.PP

    R_gpu = system.device.peak_flops_TF * TB_TO_FLOPS
    B_eff_mem = system.device.hbm_bandwidth_GBps * GB_TO_BYTES

    # Step-level roofline: B tokens computed, weights loaded once.
    t_compute = flops.F_step_device / R_gpu
    t_mem = traffic.T_step_eff / B_eff_mem
    t_local = max(t_compute, t_mem)

    t_comm = comm.t_comm_stage
    rho = tuner.overlap_factor
    t_stage = t_local + max(0.0, t_comm - rho * t_local)

    pp_bubble_factor = max(1.0, PP / max(1, B))
    t_step_user = t_stage * pp_bubble_factor
    TPOT = t_step_user

    TPS_single = B / t_step_user if t_step_user > 0 else 0.0

    replica_size = partition.PP * partition.TP * max(1, partition.EP) * partition.SP
    DP = system.num_devices // replica_size
    TTPS = DP * TPS_single

    # B* crossover: batch size where the system transitions from
    # memory-bound to compute-bound.
    denom = flops.F_token_device * B_eff_mem - traffic.T_kv * R_gpu
    B_star = (traffic.T_theta * R_gpu / denom) if denom > 0 else float("inf")

    return LatencyResults(
        t_compute=t_compute,
        t_mem=t_mem,
        t_local=t_local,
        t_comm=t_comm,
        t_stage=t_stage,
        t_step_user=t_step_user,
        pp_bubble_factor=pp_bubble_factor,
        TPS_single=TPS_single,
        TTPS=TTPS,
        B=B,
        TPOT=TPOT,
        B_star=B_star,
    )
