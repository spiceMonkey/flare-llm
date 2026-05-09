"""Per-token linear FLOPs primitive — projections + FFN + MoE router.

Returns the per-device FLOPs attributable to the *linear* (attention-free)
portion of one decoded token summed across all layers:

    F_linear_per_token =
        (L_dense/PP) · [(4H² + 4HH_kv)/D_attn + 6HI_dense/TP]
      + (L_moe  /PP) · [(4H² + 4HH_kv)/D_attn + 6HkI_moe/D_exp + 2HN_exp]

Why these terms:
- (4H² + 4HH_kv)/D_attn — Q/K/V/O projection FLOPs (TP under TP-attn,
  unsharded under DP-attn). D_attn from notation.md §1.
- 6HI_dense/TP — dense FFN (gate, up, down), I_dense wide, always
  TP-sharded (no EP axis to overlap on dense).
- 6HkI_moe/D_exp — MoE FFN: k active experts per token, sharded by D_exp
  (= TP*EP under orthogonal layout, EP under co-located).
- 2HN_exp — MoE router gate GEMM (H → N_exp), unsharded
  (see documentation/modeling/decode.md §3.4).

Attention FLOPs are NOT in this primitive because the shape differs by
phase: decode sees 4·S·H/(D_kv·SP) per token (fixed S), prefill sees
4·S²·H/(D_kv·SP) per *pass* (not per token). Callers add phase-specific
attention inline.

For prefill, `linear_flops_per_token * S_input` gives the full linear
contribution across the prefill pass.
"""

from ...specs.model_spec import LlmModelSpec
from ...specs.partition_spec import PartitionSpec
from .sharding_factors import D_attn, D_exp


def linear_flops_per_token(
    model: LlmModelSpec,
    partition: PartitionSpec,
) -> float:
    """Per-device, per-token linear FLOPs summed across all layers."""
    L = model.L
    H = model.H
    H_kv = model.H_kv()
    PP = partition.PP
    d_attn = D_attn(partition)
    d_exp_dense = D_exp(partition, layer_kind="dense")

    if model.moe is not None:
        L_moe = model.moe.n_moe_layers if model.moe.n_moe_layers else L
        L_dense = L - L_moe
        N_exp = max(1, model.moe.n_experts)
        d_exp_moe = D_exp(partition, layer_kind="moe", n_exp_cap=N_exp)
        k = model.moe.k_active
        I_moe = model.moe.I_moe
    else:
        L_moe = 0
        L_dense = L
        N_exp = 0
        d_exp_moe = d_exp_dense  # unused but defined
        k = 0
        I_moe = 0

    I_dense = model.I_dense

    # Dense contribution per layer: Q/K/V/O + FFN
    F_layer_dense = (4 * H**2 + 4 * H * H_kv) / d_attn + (6 * H * I_dense) / d_exp_dense

    # MoE contribution per layer: Q/K/V/O + routed FFN + router gate (unsharded)
    F_layer_moe = (
        (4 * H**2 + 4 * H * H_kv) / d_attn
        + (6 * H * k * I_moe) / d_exp_moe
        + 2 * H * N_exp
    )

    return (L_dense / PP) * F_layer_dense + (L_moe / PP) * F_layer_moe
