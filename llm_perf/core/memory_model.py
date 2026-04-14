
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
    B = tuner.B_decode

    # Determine layer split (MoE vs dense)
    if model.moe is not None:
        L_moe = model.moe.n_moe_layers if model.moe.n_moe_layers else L
        L_dense = L - L_moe
        N_exp = max(1, model.moe.n_experts)
        EP = min(EP, N_exp)
        I_moe = model.moe.I_moe
    else:
        L_moe = 0
        L_dense = L
        N_exp = 1
        EP = 1
        I_moe = 0

    I_dense = model.I_dense

    # Parameter memory M_theta_device (split by layer type)
    # Dense layers: EP=1, N_exp=1
    M_theta_dense = (L_dense / PP) * (
        (2 * H**2 + 2 * H * H_kv) / TP + (3 * H * I_dense) / TP
    ) * b

    # MoE layers: use EP, N_exp, I_moe
    M_theta_moe = (L_moe / PP) * (
        (2 * H**2 + 2 * H * H_kv) / TP + (3 * H * I_moe * N_exp) / (TP * EP)
    ) * b

    M_theta_device = M_theta_dense + M_theta_moe + (V * H / TP) * b

    # Activation memory M_act_device (one-layer working set, scales with B)
    M_act_device = B * (4 * H + 2 * H_kv) * b

    # KV memory M_kv_device (B sequences, each with context S)
    M_kv_device = B * (L / PP) * (2 * S * H_kv * b) / (TP * SP)

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
