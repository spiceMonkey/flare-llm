
from dataclasses import dataclass
from ..specs.system_spec import SystemSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.tuner_spec import TuningSpec
from ..utils import GB_TO_BYTES, TB_TO_FLOPS
from .flops_model import FlopsResults
from .traffic_model import TrafficResults
from .comm_model import CommResults


@dataclass
class LatencyResults:
    t_compute: float
    t_mem: float
    t_local: float
    t_comm: float
    t_stage: float          # per-pipeline-stage step time (pre-bubble)
    t_step_user: float      # user-observed step latency (t_stage * bubble)
    pp_bubble_factor: float # max(1, PP / B): >1 when the pipeline is underfilled
    TPS_single: float
    TTPS: float
    B: int
    TPOT: float             # user-observed time per output token = t_step_user
    B_star: float


def compute_latency(
    system: SystemSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
    flops: FlopsResults,
    traffic: TrafficResults,
    comm: CommResults,
) -> LatencyResults:
    """Compute per-token latency and throughput (seconds, tokens/s).

    The per-stage roofline gives the cost of one pipeline stage processing
    the current batch. For a user observing inter-token latency, we must
    account for the pipeline bubble when B < PP: the single batch (one
    microbatch for decode) traverses all PP stages, so
        t_step_user = t_stage * max(1, PP / B).
    When B >= PP the pipeline can be kept full and each user sees t_stage
    between tokens.
    """

    B = tuner.B_decode
    PP = partition.PP

    # Device throughput in FLOPs/s
    R_gpu = system.device.peak_flops_TF * TB_TO_FLOPS

    # Effective memory bandwidth in bytes/s (GB/s → bytes/s, decimal)
    B_eff_mem = system.device.hbm_bandwidth_GBps * GB_TO_BYTES

    # Step-level roofline: B tokens computed, weights loaded once
    t_compute = flops.F_step_device / R_gpu
    t_mem = traffic.T_step_eff / B_eff_mem
    t_local = max(t_compute, t_mem)

    t_comm = comm.t_comm_stage
    rho = tuner.overlap_factor

    # Per-stage step time (overlap-aware).
    t_stage = t_local + max(0.0, t_comm - rho * t_local)

    # Pipeline-bubble correction.
    pp_bubble_factor = max(1.0, PP / max(1, B))
    t_step_user = t_stage * pp_bubble_factor

    # TPOT: user-observed time between tokens = step latency.
    TPOT = t_step_user

    # Single-replica throughput: B tokens output per step, one step per
    # t_step_user. Matches TTPS = DP * B / t_step_user.
    TPS_single = B / t_step_user if t_step_user > 0 else 0.0

    replica_size = partition.PP * partition.TP * max(1, partition.EP) * partition.SP
    DP = system.num_devices // replica_size
    TTPS = DP * TPS_single

    # B* crossover: batch size where the system transitions from
    # memory-bound to compute-bound.
    denom = flops.F_token_device * B_eff_mem - traffic.T_kv * R_gpu
    if denom > 0:
        B_star = traffic.T_theta * R_gpu / denom
    else:
        B_star = float("inf")

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
