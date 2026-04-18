"""Weight (θ) footprint primitives — per-device bytes for transformer weights.

Splits the weight footprint along the dense/MoE axis so callers can compose
exactly what they need. Dense and MoE layers share the same attention
projections (Q,K,V,O) but differ in the FFN: dense has a single I_dense-wide
FFN replicated across TP ranks, while MoE has N_exp expert FFNs of width
I_moe distributed across EP ranks.

All three functions return bytes on one device (TP·EP·PP aware). Embedding
bytes are returned separately because they're outside the layer loop.
"""

from ...specs.model_spec import LlmModelSpec
from ...specs.partition_spec import PartitionSpec


def _split_layers(model: LlmModelSpec) -> tuple[int, int, int, int, int]:
    """Return (L_dense, L_moe, N_exp, EP_cap, I_moe) with the same clamp
    logic that core/memory_model.py and core/traffic_model.py apply."""
    L = model.L
    if model.moe is not None:
        L_moe = model.moe.n_moe_layers if model.moe.n_moe_layers else L
        L_dense = L - L_moe
        N_exp = max(1, model.moe.n_experts)
        I_moe = model.moe.I_moe
    else:
        L_moe = 0
        L_dense = L
        N_exp = 1
        I_moe = 0
    return L_dense, L_moe, N_exp, I_moe


def dense_weight_bytes(model: LlmModelSpec, partition: PartitionSpec) -> float:
    """Per-device weight bytes for the dense-layer slice of the model.

        M_theta_dense = (L_dense / PP) · ((2H² + 2HH_kv)/TP + 3HI_dense/TP) · b
    """
    L = model.L
    if model.moe is not None:
        L_moe = model.moe.n_moe_layers if model.moe.n_moe_layers else L
        L_dense = L - L_moe
    else:
        L_dense = L

    H = model.H
    H_kv = model.H_kv()
    I_dense = model.I_dense
    b = model.bytes_per_param
    PP = partition.PP
    TP = partition.TP

    return (L_dense / PP) * (
        (2 * H**2 + 2 * H * H_kv) / TP + (3 * H * I_dense) / TP
    ) * b


def moe_weight_bytes(model: LlmModelSpec, partition: PartitionSpec) -> float:
    """Per-device weight bytes for the MoE-layer slice of the model.

        M_theta_moe = (L_moe / PP) · ((2H² + 2HH_kv)/TP + 3HI_moe·N_exp/(TP·EP)) · b

    Returns 0.0 for dense-only models. EP is clamped by N_exp to match
    the existing convention (EP > N_exp collapses to EP = N_exp).
    """
    if model.moe is None:
        return 0.0

    L = model.L
    L_moe = model.moe.n_moe_layers if model.moe.n_moe_layers else L
    N_exp = max(1, model.moe.n_experts)
    I_moe = model.moe.I_moe

    H = model.H
    H_kv = model.H_kv()
    b = model.bytes_per_param
    PP = partition.PP
    TP = partition.TP
    EP = min(max(1, partition.EP), N_exp)

    return (L_moe / PP) * (
        (2 * H**2 + 2 * H * H_kv) / TP + (3 * H * I_moe * N_exp) / (TP * EP)
    ) * b


def embedding_bytes(model: LlmModelSpec, partition: PartitionSpec) -> float:
    """Per-device embedding (and LM head) bytes.

        M_embed = V · H / TP · b

    The embedding is sharded across TP ranks and replicated across PP
    stages in the current convention — matches memory_model.py.
    """
    return (model.vocab_size * model.H / partition.TP) * model.bytes_per_param
