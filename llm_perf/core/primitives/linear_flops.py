"""Per-token linear FLOPs primitive — projections + FFN + MoE router.

Returns the per-device FLOPs attributable to the *linear* (attention-free)
portion of one decoded token summed across all layers:

    F_linear_per_token =
        (L_dense/PP) · [(4H² + 4HH_kv)/TP + 6HI_dense/TP]
      + (L_moe  /PP) · [(4H² + 4HH_kv)/TP + 6HkI_moe/(TP·EP) + 2HN_exp]

Why these terms:
- (4H² + 4HH_kv)/TP — Q/K/V/O projections, sharded across TP.
- 6HI_dense/TP — dense FFN (gate, up, down), I_dense wide, TP-sharded.
- 6HkI_moe/(TP·EP) — MoE FFN: k active experts per token, sharded across TP·EP.
- 2HN_exp — MoE router gate GEMM (H → N_exp), unsharded across TP
  (see documentation/modeling/decode.md §3.4).

Attention FLOPs are NOT in this primitive because the shape differs by
phase: decode sees 4·S·H/(TP·SP) per token (fixed S), prefill sees
4·S²·H/(TP·SP) per *pass* (not per token). Callers add phase-specific
attention inline.

For prefill, `linear_flops_per_token * S_input` gives the full linear
contribution across the prefill pass.
"""

from ...specs.model_spec import LlmModelSpec
from ...specs.partition_spec import PartitionSpec


def linear_flops_per_token(
    model: LlmModelSpec,
    partition: PartitionSpec,
) -> float:
    """Per-device, per-token linear FLOPs summed across all layers."""
    L = model.L
    H = model.H
    H_kv = model.H_kv()
    PP = partition.PP
    TP = partition.TP

    if model.moe is not None:
        L_moe = model.moe.n_moe_layers if model.moe.n_moe_layers else L
        L_dense = L - L_moe
        N_exp = max(1, model.moe.n_experts)
        EP = min(max(1, partition.EP), N_exp)
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

    # Dense contribution per layer: Q/K/V/O + FFN
    F_layer_dense = (4 * H**2 + 4 * H * H_kv) / TP + (6 * H * I_dense) / TP

    # MoE contribution per layer: Q/K/V/O + routed FFN + router gate (unsharded)
    F_layer_moe = (
        (4 * H**2 + 4 * H * H_kv) / TP
        + (6 * H * k * I_moe) / (TP * EP)
        + 2 * H * N_exp
    )

    return (L_dense / PP) * F_layer_dense + (L_moe / PP) * F_layer_moe
