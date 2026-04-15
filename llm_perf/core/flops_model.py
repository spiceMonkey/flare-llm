
from dataclasses import dataclass
from ..specs.model_spec import LlmModelSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.tuner_spec import TuningSpec

@dataclass
class FlopsResults:
    F_token_device: float   # FLOPs per token per device
    F_layer_per_device: float
    F_step_device: float    # FLOPs per decode step (B tokens) per device


def compute_flops(
    model: LlmModelSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
) -> FlopsResults:
    """Compute per-device FLOPs per token on a PP stage."""

    L = model.L
    H = model.H
    H_kv = model.H_kv()
    PP = partition.PP
    TP = partition.TP
    EP = max(1, partition.EP)
    SP = partition.SP
    S = tuner.S_decode

    # Determine layer split (MoE vs dense)
    if model.moe is not None:
        L_moe = model.moe.n_moe_layers if model.moe.n_moe_layers else L
        L_dense = L - L_moe
        N_exp = max(1, model.moe.n_experts)
        EP = min(EP, N_exp)
        k = model.moe.k_active
        I_moe = model.moe.I_moe
    else:
        L_moe = 0
        L_dense = L
        N_exp = 0
        EP = 1
        k = 0
        I_moe = 0

    I_dense = model.I_dense

    # Dense layer FLOPs per device (per layer)
    F_layer_dense = (
        (4 * H**2 + 4 * H * H_kv) / TP
        + (6 * H * I_dense) / TP  # EP=1 for dense
        + (4 * S * H) / (TP * SP)
        # No router FLOPs for dense layers
    )

    # MoE layer FLOPs per device (per layer)
    F_layer_moe = (
        (4 * H**2 + 4 * H * H_kv) / TP
        + (6 * H * k * I_moe) / (TP * EP)
        + (4 * S * H) / (TP * SP)
        + 2 * H * N_exp  # Router FLOPs (unsharded)
    )

    B = tuner.B_decode

    F_token_device = (L_dense / PP) * F_layer_dense + (L_moe / PP) * F_layer_moe

    # Compute average F_layer_per_device for backward compatibility
    F_layer_per_device = F_token_device / (L / PP) if L > 0 else 0.0

    # Step FLOPs: B tokens processed per decode step
    F_step_device = B * F_token_device

    return FlopsResults(
        F_token_device=F_token_device,
        F_layer_per_device=F_layer_per_device,
        F_step_device=F_step_device,
    )
