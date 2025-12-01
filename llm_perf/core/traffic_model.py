
from dataclasses import dataclass
from ..specs.model_spec import LlmModelSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.tuner_spec import TuningSpec


@dataclass
class TrafficResults:
    T_theta: float
    T_act: float
    T_kv: float
    T_token_eff: float


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

    gamma_FA = tuner.flash_attn_gain
    gamma_FMLP = tuner.flash_mlp_gain
    c_act = tuner.c_act  # heuristic constant for activation traffic

    # MoE params
    if model.moe is not None:
        N_exp = max(1, model.moe.n_experts)
        EP = min(EP, N_exp)
        I = model.moe.I_moe
    else:
        N_exp = 1
        EP = 1
        I = model.I_dense

    moe_term = (3 * H * I * N_exp) / (TP * EP * gamma_FMLP)

    # Parameter traffic
    T_theta = (
        (L / PP)
        * (
            (H**2 + 3 * H * H_kv) / (TP * gamma_FA)
            + moe_term
        )
        * b
    )

    # Activation traffic
    T_act = (L / PP) * c_act * H * b

    # KV traffic
    T_kv = (L / PP) * (2 * S * H_kv * b) / (TP * SP)

    T_token_eff = T_theta + T_act + T_kv

    return TrafficResults(
        T_theta=T_theta,
        T_act=T_act,
        T_kv=T_kv,
        T_token_eff=T_token_eff,
    )
