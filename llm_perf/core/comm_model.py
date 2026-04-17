
import math
from dataclasses import dataclass
from ..specs.model_spec import LlmModelSpec
from ..specs.system_spec import SystemSpec, span_tiers
from ..specs.partition_spec import PartitionSpec
from ..specs.tuner_spec import TuningSpec
from ..utils import GB_TO_BYTES, US_TO_SECONDS


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


def _fabric_cost(system: SystemSpec, collective: str, group_size: int) -> tuple[float, float]:
    """Return (alpha_s, bw_bytes_per_s) for `collective` over `group_size` ranks.

    Walks the concatenated tier chain of the collective's fabric list,
    summing α and flooring bandwidth to the narrowest tier actually crossed.
    """
    tiers = system.get_tier_chain(collective)
    alpha_us, bw_GBps, _ = span_tiers(tiers, max(1, int(group_size)))
    return alpha_us * US_TO_SECONDS, bw_GBps * GB_TO_BYTES


def compute_comm(
    model: LlmModelSpec,
    system: SystemSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
) -> CommResults:
    """Compute per-token communication times (seconds)."""

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

    # Expert-parallel bookkeeping up front (EP group size feeds _fabric_cost)
    if model.moe is not None:
        N_exp = max(1, model.moe.n_experts)
        EP = min(EP, N_exp)
        k = model.moe.k_active
    else:
        N_exp = 1
        EP = 1
        k = 1

    # Per-collective α (s) and BW (bytes/s) resolved against the relevant group size.
    # PP is a point-to-point hop between adjacent stages → group size 2.
    a_PP, B_PP = _fabric_cost(system, "PP", 2)
    a_TP, B_TP = _fabric_cost(system, "TP", TP)
    a_EP, B_EP = _fabric_cost(system, "EP", EP)
    a_SP, B_SP = _fabric_cost(system, "SP", SP)

    # PP: shard-preserving hop of B tokens × (H/TP) activation bytes
    msg_PP = B * (H / TP) * b
    t_PP = a_PP + msg_PP / B_PP if B_PP > 0 else 0.0

    # Algorithm choices (default to "ring" if fields are missing)
    tp_algorithm = getattr(tuner, "tp_algorithm", "ring").lower()
    ep_algorithm = getattr(tuner, "ep_algorithm", "ring").lower()

    # EP: 2-pass all-to-all of size kH; default k=1 if no MoE
    if EP > 1:
        # B tokens × k active experts × H activation bytes per expert
        msg_EP = B * k * H * b  # bytes per device
        if ep_algorithm == "ring":
            # 2-pass ring all-to-all (Dispatch + Combine)
            t_EP = 2 * (EP - 1) * a_EP + 2 * (EP - 1) * (msg_EP / (EP * B_EP))
        elif ep_algorithm == "tree":
            # Approx: 2-pass log2 tree all-to-all
            t_EP = 2 * math.ceil(math.log2(EP)) * a_EP + 2 * (msg_EP / B_EP)
        else:
            raise ValueError(f"Unsupported ep_algorithm: {ep_algorithm!r}")
    else:
        t_EP = 0.0
        msg_EP = 0.0

    # TP: 2-pass all-reduce of B tokens × H activation bytes
    if TP > 1:
        msg_TP = B * H * b  # bytes
        if tp_algorithm == "ring":
            # 2-pass ring all-reduce
            t_TP = 2 * (TP - 1) * a_TP + 2 * ((TP - 1) / TP) * (msg_TP / B_TP)
        elif tp_algorithm == "tree":
            # 2-pass tree all-reduce (reduce + broadcast)
            t_TP = 2 * math.ceil(math.log2(TP)) * a_TP + 2 * (msg_TP / B_TP)
        else:
            raise ValueError(f"Unsupported tp_algorithm: {tp_algorithm!r}")
    else:
        t_TP = 0.0
        msg_TP = 0.0

    # SP: 1-pass ring for KV shard (All-Gather) over B sequences
    if SP > 1:
        msg_SP = B * (S / SP) * (2 * H_kv / TP) * b
        t_SP = (SP - 1) * a_SP + (SP - 1) * (msg_SP / B_SP)
    else:
        t_SP = 0.0
        msg_SP = 0.0

    # Determine layer split for EP communication (MoE layers only)
    if model.moe is not None:
        L_moe = model.moe.n_moe_layers if model.moe.n_moe_layers else L
    else:
        L_moe = 0

    # Per-stage comm: TP and SP apply to all layers, EP only to MoE layers
    t_comm_stage = (
        (L / PP) * (n_TP * t_TP + n_SP * t_SP)  # All layers
        + (L_moe / PP) * (n_EP * t_EP)           # MoE layers only
        + t_PP
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
