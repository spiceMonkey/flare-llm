
import math
from dataclasses import dataclass
from ..specs.model_spec import LlmModelSpec
from ..specs.system_spec import SystemSpec
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
    b = model.bytes_per_param

    n_TP = tuner.n_TP_collectives
    n_EP = tuner.n_EP_collectives

    # Network domains
    dom_PP = system.get_domain("PP")
    dom_TP = system.get_domain("TP")
    dom_EP = system.get_domain("EP")
    dom_SP = system.get_domain("SP")

    # Convert alpha from us to s
    a_PP = dom_PP.alpha_us * US_TO_SECONDS
    a_TP = dom_TP.alpha_us * US_TO_SECONDS
    a_EP = dom_EP.alpha_us * US_TO_SECONDS
    a_SP = dom_SP.alpha_us * US_TO_SECONDS

    # Convert GB/s → bytes/s (decimal)
    B_PP = dom_PP.bandwidth_GBps * GB_TO_BYTES
    B_TP = dom_TP.bandwidth_GBps * GB_TO_BYTES
    B_EP = dom_EP.bandwidth_GBps * GB_TO_BYTES
    B_SP = dom_SP.bandwidth_GBps * GB_TO_BYTES
    
    # PP: shard-preserving hop of H/TP
    msg_PP = (H / TP) * b
    t_PP = a_PP + msg_PP / B_PP

    # Algorithm choices (default to "ring" if fields are missing)
    tp_algorithm = getattr(tuner, "tp_algorithm", "ring").lower()
    ep_algorithm = getattr(tuner, "ep_algorithm", "ring").lower()

    # EP: 1-pass all-to-all of size kH; default k=1 if no MoE
    if model.moe is not None:
        N_exp = max(1, model.moe.n_experts)
        EP = min(EP, N_exp)
        k = model.moe.k_active
    else:
        N_exp = 1
        EP = 1
        k = 1
    if EP > 1:
        msg_EP = k * H * b  # bytes per device
        if ep_algorithm == "ring":
            # 1-pass ring all-to-all (simple alpha-beta)
            t_EP = (EP - 1) * a_EP + (EP - 1) * (msg_EP / (EP * B_EP))
        elif ep_algorithm == "tree":
            # Approx: log2 tree all-to-all
            t_EP = math.ceil(math.log2(EP)) * a_EP + (msg_EP / B_EP)
        else:
            raise ValueError(f"Unsupported ep_algorithm: {ep_algorithm!r}")
    else:
        t_EP = 0.0
        msg_EP = 0.0

    # TP: 2-pass all-reduce of size H/TP
    if TP > 1:
        msg_TP = (H / TP) * b  # bytes
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

    # SP: 2-pass ring for KV shard
    if SP > 1:
        msg_SP = (S / SP) * (2 * H_kv / TP) * b
        t_SP = 2 * (SP - 1) * a_SP + 2 * ((SP - 1) / SP) * (msg_SP / B_SP)
    else:
        t_SP = 0.0
        msg_SP = 0.0

    # Per-stage comm (same for all PP stages in this simple model)
    t_comm_stage = (L / PP) * (n_TP * t_TP + n_EP * t_EP + t_SP) + t_PP

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
