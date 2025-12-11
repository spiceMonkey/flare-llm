
from dataclasses import dataclass
from ..specs.model_spec import LlmModelSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.tuner_spec import TuningSpec

@dataclass
class FlopsResults:
    F_token_device: float   # FLOPs per token per device
    F_layer_per_device: float


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

    # MoE effective params
    if model.moe is not None:
        N_exp = max(1, model.moe.n_experts)
        EP = min(EP, N_exp)
        I_eff = model.moe.k_active * model.moe.I_moe
        N_eff = N_exp
    else:
        N_exp = 1
        EP = 1
        I_eff = model.I_dense
        N_eff = 0

    # Per-layer FLOPs per device
    F_layer_per_device = (
        (2 * H**2 + 6 * H * H_kv) / TP
        + (4 * H * I_eff) / (TP * EP)
        + (4 * S * H_kv) / (TP * SP)
        + 2 * H * N_eff
    )

    F_token_device = (L / PP) * F_layer_per_device

    return FlopsResults(
        F_token_device=F_token_device,
        F_layer_per_device=F_layer_per_device,
    )
