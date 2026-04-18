
from dataclasses import dataclass
from ..specs.model_spec import LlmModelSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.tuner_spec import TuningSpec
from .primitives import dense_weight_bytes, moe_weight_bytes, kv_bytes_per_seq


@dataclass
class TrafficResults:
    T_theta: float
    T_kv: float          # KV traffic per token (single sequence)
    T_token_eff: float   # effective per-token traffic at B=1
    T_step_eff: float    # effective per-step traffic at batch B


def compute_traffic(
    model: LlmModelSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
) -> TrafficResults:
    """Compute per-token memory traffic per device (bytes).

    Weight traffic matches M_theta (dense + MoE layers; embedding is
    outside the forward path and excluded by convention). KV traffic
    scales with the decode context length S and the batch B.
    """

    S = tuner.S_decode
    B = tuner.B_decode

    # Parameter traffic = dense + MoE weight bytes (no embedding in fwd path).
    T_theta = dense_weight_bytes(model, partition) + moe_weight_bytes(model, partition)

    # KV traffic for one sequence of S tokens.
    T_kv = kv_bytes_per_seq(model, partition, S)

    T_token_eff = T_theta + T_kv

    # Batched step traffic: weights loaded once, KV read per sequence per step.
    T_step_eff = T_theta + B * T_kv

    return TrafficResults(
        T_theta=T_theta,
        T_kv=T_kv,
        T_token_eff=T_token_eff,
        T_step_eff=T_step_eff,
    )
