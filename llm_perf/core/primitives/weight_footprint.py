"""Weight (θ) footprint primitives — per-device bytes for transformer weights.

Splits the weight footprint along the dense/MoE axis so callers can compose
exactly what they need. Dense and MoE layers share the same attention
projections (Q,K,V,O) but differ in the FFN: dense always TP-shards the FFN;
MoE shards experts by D_exp (= TP*EP under orthogonal layout, EP under
co-located).

All three functions return bytes on one device (D-factor and PP aware). The
abstract divisors D_attn / D_exp / D_emb are resolved from the
PartitionSpec via `sharding_factors`; see `notation.md §1` and `decode.md
§1.4`. Embedding bytes are returned separately because they're outside the
layer loop.
"""

from ...specs.model_spec import LlmModelSpec
from ...specs.partition_spec import PartitionSpec
from .sharding_factors import D_attn, D_emb, D_exp


def _split_layers(model: LlmModelSpec) -> tuple[int, int, int, int]:
    """Return (L_dense, L_moe, N_exp, I_moe) with the same clamp
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

        M_theta_dense = (L_dense / PP) · ((2H² + 2HH_kv)/D_attn + 3HI_dense/TP) · b

    Dense FFN always uses /TP regardless of layout (no EP axis to overlap;
    co-location does not apply to dense layers). Attention follows D_attn.
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
    d_attn = D_attn(partition)
    d_exp_dense = D_exp(partition, layer_kind="dense")

    return (L_dense / PP) * (
        (2 * H**2 + 2 * H * H_kv) / d_attn + (3 * H * I_dense) / d_exp_dense
    ) * b


def moe_weight_bytes(model: LlmModelSpec, partition: PartitionSpec) -> float:
    """Per-device weight bytes for the MoE-layer slice of the model.

        M_theta_moe = (L_moe / PP) · ((2H² + 2HH_kv)/D_attn + 3HI_moe·N_exp/D_exp) · b

    Returns 0.0 for dense-only models. EP is clamped by N_exp internally
    by `D_exp(layer_kind="moe", n_exp_cap=N_exp)`: under the orthogonal
    layout the divisor is TP * min(EP, N_exp); under co-located it is
    min(EP, N_exp).
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
    d_attn = D_attn(partition)
    d_exp_moe = D_exp(partition, layer_kind="moe", n_exp_cap=N_exp)

    return (L_moe / PP) * (
        (2 * H**2 + 2 * H * H_kv) / d_attn + (3 * H * I_moe * N_exp) / d_exp_moe
    ) * b


def embedding_bytes(model: LlmModelSpec, partition: PartitionSpec) -> float:
    """Per-device embedding (and LM head) bytes.

        M_embed = V · H / D_emb · b

    The embedding is sharded along the vocab dimension by D_emb (= TP under
    all configurations) and replicated across PP stages in the current
    convention — matches memory_model.py.
    """
    return (model.vocab_size * model.H / D_emb(partition)) * model.bytes_per_param
