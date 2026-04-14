
from dataclasses import dataclass
from ..specs.model_spec import LlmModelSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.tuner_spec import TuningSpec


@dataclass
class TrafficResults:
    T_theta: float
    T_act: float
    T_kv: float          # KV traffic per token (single sequence)
    T_token_eff: float   # effective per-token traffic at B=1
    T_step_eff: float    # effective per-step traffic at batch B


def compute_traffic(
    model: LlmModelSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
) -> TrafficResults:
    """Compute per-token memory traffic per device (bytes)."""

    L = model.L
    H = model.H
    H_kv = model.H_kv()
    PP = partition.PP
    TP = partition.TP
    EP = max(1, partition.EP)
    SP = partition.SP
    S = tuner.S_decode
    b = model.bytes_per_param

    c_act = tuner.c_act  # heuristic constant for activation traffic

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

    # Parameter traffic - dense layers
    T_theta_dense = (L_dense / PP) * (
        (2 * H**2 + 2 * H * H_kv) / TP + (3 * H * I_dense) / TP
    ) * b

    # Parameter traffic - MoE layers
    T_theta_moe = (L_moe / PP) * (
        (2 * H**2 + 2 * H * H_kv) / TP + (3 * H * I_moe * N_exp) / (TP * EP)
    ) * b

    T_theta = T_theta_dense + T_theta_moe

    # Activation traffic
    T_act = (L / PP) * c_act * H * b

    # KV traffic
    T_kv = (L / PP) * (2 * S * H_kv * b) / (TP * SP)

    B = tuner.B_decode

    T_token_eff = T_theta + T_act + T_kv

    # Batched step traffic: weights loaded once, KV read per sequence
    T_step_eff = T_theta + T_act + B * T_kv

    return TrafficResults(
        T_theta=T_theta,
        T_act=T_act,
        T_kv=T_kv,
        T_token_eff=T_token_eff,
        T_step_eff=T_step_eff,
    )
