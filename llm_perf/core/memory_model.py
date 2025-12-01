
from dataclasses import dataclass
from ..specs.model_spec import LlmModelSpec
from ..specs.system_spec import SystemSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.tuner_spec import TuningSpec
from ..utils import GB_TO_BYTES


@dataclass
class MemoryResults:
    M_theta_device: float
    M_act_device: float
    M_kv_device: float
    M_total_device: float
    fits_in_HBM: bool


def compute_memory(
    model: LlmModelSpec,
    system: SystemSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
) -> MemoryResults:
    """Compute per-device static memory footprint (bytes)."""

    L = model.L
    H = model.H
    H_kv = model.H_kv()
    V = model.vocab_size
    b = model.bytes_per_param

    PP = partition.PP
    TP = partition.TP
    EP = max(1, partition.EP)
    SP = partition.SP
    S = tuner.S_decode

    # MoE parameters
    if model.moe is not None:
        N_exp = max(1, model.moe.n_experts)
        EP = min(EP, N_exp)
        I = model.moe.I_moe
    else:
        N_exp = 1
        EP = 1
        I = model.I_dense

    # Parameter memory M_theta_device
    moe_term = (2 * H * I * N_exp) / (TP * EP)

    M_theta_device = (
        (L / PP)
        * (
            (H**2 + 3 * H * H_kv) / TP
            + moe_term
        )
        * b
        + (V * H / TP) * b
    )

    # Activation memory M_act_device
    M_act_device = (L / PP) * (4 * H + 2 * H_kv) * b

    # KV memory M_kv_device
    M_kv_device = (L / PP) * (2 * S * H_kv * b) / (TP * SP)

    M_total = M_theta_device + M_act_device + M_kv_device

    # HBM in bytes
    HBM_bytes = system.device.hbm_capacity_GB * GB_TO_BYTES
    fits = M_total <= HBM_bytes

    return MemoryResults(
        M_theta_device=M_theta_device,
        M_act_device=M_act_device,
        M_kv_device=M_kv_device,
        M_total_device=M_total,
        fits_in_HBM=fits,
    )
