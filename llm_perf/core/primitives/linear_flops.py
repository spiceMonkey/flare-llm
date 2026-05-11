"""Per-token linear FLOPs primitive — projections + FFN + MoE router.

Returns the per-device FLOPs attributable to the *linear* (attention-free)
portion of one decoded token summed across all layers:

    F_linear_per_token =
        (L_dense/PP) · [F_attn_proj + 6HI_dense/TP]
      + (L_moe  /PP) · [F_attn_proj + 6HkI_moe/D_exp + 2HN_exp]

where `F_attn_proj` is the attention projection FLOPs per layer per
device, branching on attention variant:
- GQA / MHA: `(4H² + 4HH_kv) / D_attn` — TP under TP-attn,
  unsharded under DP-attn (D_attn from notation.md §1).
- MLA: variant-specific projection FLOPs from
  `mla_flops.mla_proj_flops_per_layer_per_device`, which handles the
  shardable (W_UQ / W_UK / W_UV / W_O) vs replicated (W_DQ / W_DKV)
  split under TP-attn and the materialized vs absorbed mode dispatch.

Other terms unchanged:
- 6HI_dense/TP — dense FFN (gate, up, down), I_dense wide, always
  TP-sharded (no EP axis to overlap on dense).
- 6HkI_moe/D_exp — MoE FFN: k active experts per token, sharded by D_exp
  (= TP*EP under orthogonal layout, EP under co-located).
- 2HN_exp — MoE router gate GEMM (H → N_exp), unsharded
  (see documentation/modeling/decode.md §3.4).

Attention score / value FLOPs are NOT in this primitive because the
shape differs by phase: decode sees 4·S·H/(D_kv·SP) per token (fixed S),
prefill sees 4·S²·H/(D_kv·SP) per *pass*. Callers add phase-specific
attention inline (using `mla_score_value_flops_per_layer_per_device` for
MLA models).

For prefill, `linear_flops_per_token * S_input` gives the full linear
contribution across the prefill pass.
"""

from ...specs.framework_spec import FrameworkSpec
from ...specs.model_spec import LlmModelSpec
from ...specs.partition_spec import PartitionSpec
from .mla_flops import mla_proj_flops_per_layer_per_device
from .sharding_factors import D_attn, D_exp


def linear_flops_per_token(
    model: LlmModelSpec,
    partition: PartitionSpec,
    framework: FrameworkSpec,
) -> float:
    """Per-device, per-token linear FLOPs summed across all layers.

    `framework` selects MLA mode (`mla_mode`) and the attention dispatch
    pattern (`attention_mode`, `layout`) consumed by the sharding-factor
    helpers. `attention_mode` / `layout` are ignored for GQA / MHA models;
    `mla_mode` is ignored for non-MLA models.
    """
    L = model.L
    H = model.H
    PP = partition.PP
    d_attn = D_attn(partition, framework)
    d_exp_dense = D_exp(partition, framework, layer_kind="dense")

    if model.moe is not None:
        L_moe = model.moe.n_moe_layers if model.moe.n_moe_layers else L
        L_dense = L - L_moe
        N_exp = max(1, model.moe.n_experts)
        d_exp_moe = D_exp(partition, framework, layer_kind="moe", n_exp_cap=N_exp)
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

    # Per-layer per-device attention projection FLOPs — branch on variant.
    if model.mla is not None:
        F_attn_proj = mla_proj_flops_per_layer_per_device(
            model, partition, framework, framework.mla_mode
        )
    else:
        H_kv = model.H_kv()
        F_attn_proj = (4 * H**2 + 4 * H * H_kv) / d_attn

    # Dense contribution per layer: attention projection + FFN
    F_layer_dense = F_attn_proj + (6 * H * I_dense) / d_exp_dense

    # MoE contribution per layer: attention projection + routed FFN + router gate
    F_layer_moe = (
        F_attn_proj
        + (6 * H * k * I_moe) / d_exp_moe
        + 2 * H * N_exp
    )

    return (L_dense / PP) * F_layer_dense + (L_moe / PP) * F_layer_moe
