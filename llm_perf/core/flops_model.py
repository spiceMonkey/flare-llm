
from dataclasses import dataclass
from ..specs.model_spec import LlmModelSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.tuner_spec import TuningSpec
from .primitives import linear_flops_per_token


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
    """Compute per-device FLOPs per token on a PP stage.

    Linear contribution (proj + FFN + router) comes from the shared
    `linear_flops_per_token` primitive. Attention (4·S·H/(TP·SP) per
    token per layer) is decode-specific and stays inline.
    """

    L = model.L
    H = model.H
    PP = partition.PP
    TP = partition.TP
    SP = partition.SP
    S = tuner.S_decode
    B = tuner.B_decode

    # Linear FLOPs per token (proj + FFN + MoE router, router unsharded)
    F_linear_per_token = linear_flops_per_token(model, partition)

    # Decode attention FLOPs per token: 4·S·H/(TP·SP), summed across L layers
    F_attn_per_token = (L / PP) * (4 * S * H) / (TP * SP)

    F_token_device = F_linear_per_token + F_attn_per_token

    # Average per-layer FLOPs (diagnostic)
    F_layer_per_device = F_token_device / (L / PP) if L > 0 else 0.0

    # Step FLOPs: B tokens processed per decode step
    F_step_device = B * F_token_device

    return FlopsResults(
        F_token_device=F_token_device,
        F_layer_per_device=F_layer_per_device,
        F_step_device=F_step_device,
    )
