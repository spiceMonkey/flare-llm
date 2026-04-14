
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
    t_token: float
    TPS_single: float
    TTPS: float
    B: int
    TPOT: float
    B_star: float


def compute_latency(
    system: SystemSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
    flops: FlopsResults,
    traffic: TrafficResults,
    comm: CommResults,
) -> LatencyResults:
    """Compute per-token latency and throughput (seconds, tokens/s)."""

    B = tuner.B_decode

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

    # Unified Overlap Model:
    # t_token = t_local + unhidden_comm
    # unhidden_comm = max(0, t_comm - rho * t_local)
    t_token = t_local + max(0.0, t_comm - rho * t_local)

    # TPOT: time per output token = step time / batch size
    TPOT = t_token / B if B > 0 else 0.0

    t_stage = t_token
    TPS_single = B / t_stage if t_stage > 0 else 0.0

    # Infer DP from system size and partition dimensions
    replica_size = partition.PP * partition.TP * max(1, partition.EP) * partition.SP
    DP = system.num_devices // replica_size
    TTPS = DP * TPS_single

    # B* crossover: batch size where system transitions from memory-bound to compute-bound
    # B* = T_theta * R_gpu / (F_token * B_eff_mem - T_kv * R_gpu)
    denom = flops.F_token_device * B_eff_mem - traffic.T_kv * R_gpu
    if denom > 0:
        B_star = traffic.T_theta * R_gpu / denom
    else:
        B_star = float('inf')  # always memory-bound

    return LatencyResults(
        t_compute=t_compute,
        t_mem=t_mem,
        t_local=t_local,
        t_comm=t_comm,
        t_token=t_token,
        TPS_single=TPS_single,
        TTPS=TTPS,
        B=B,
        TPOT=TPOT,
        B_star=B_star,
    )
