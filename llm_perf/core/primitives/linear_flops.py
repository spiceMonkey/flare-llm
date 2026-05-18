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
  (= TP*EP under orthogonal tp_ep_layout, EP under co-located).
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


def attn_proj_flops_per_token(
    model: LlmModelSpec,
    partition: PartitionSpec,
    framework: FrameworkSpec,
) -> float:
    """Per-device per-token attention-projection FLOPs (Q/K/V/O), all layers.

    Returns the per-token cost *as seen by one rank*: under TP-attn the
    head-sharded share (per-layer divided by D_attn = TP), under DP-attn
    the full per-token projection cost (D_attn = 1, weights replicated).
    Per-step composition in `decode_model.compute_flops` applies the
    correct per-rank batch divisor — TP-attn ranks see all B tokens (so
    F_step = B · F_attn_proj_per_token), DP-attn ranks see only B/G_TP
    tokens for the attention block (so F_step = (B/G_TP) · F_attn_proj_per_token).
    """
    L = model.L
    H = model.H
    PP = partition.PP
    d_attn = D_attn(partition, framework)

    if model.mla is not None:
        F_attn_proj = mla_proj_flops_per_layer_per_device(
            model, partition, framework, framework.mla_mode
        )
    else:
        H_kv = model.H_kv()
        F_attn_proj = (4 * H**2 + 4 * H * H_kv) / d_attn

    return (L / PP) * F_attn_proj


def ffn_flops_per_token(
    model: LlmModelSpec,
    partition: PartitionSpec,
    framework: FrameworkSpec,
) -> float:
    """Per-device per-token FFN + MoE-router FLOPs, all layers.

    Returns dense-FFN + MoE-routed-FFN + MoE-router-gate compute, summed
    across L layers and divided by per-component sharding factors
    (D_exp_dense = TP, D_exp_moe per the §3.2 layout). Independent of
    `attention_mode` — FFN per-token cost scales the same way under
    TP-attn and DP-attn at the per-step level (each rank processes
    all B tokens through its expert / FFN shard).
    """
    L = model.L
    H = model.H
    PP = partition.PP
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
        d_exp_moe = d_exp_dense  # unused
        k = 0
        I_moe = 0

    I_dense = model.I_dense

    F_layer_dense_ffn = (6 * H * I_dense) / d_exp_dense
    F_layer_moe_ffn = (6 * H * k * I_moe) / d_exp_moe + 2 * H * N_exp

    return (L_dense / PP) * F_layer_dense_ffn + (L_moe / PP) * F_layer_moe_ffn


def linear_flops_per_token(
    model: LlmModelSpec,
    partition: PartitionSpec,
    framework: FrameworkSpec,
) -> float:
    """Per-device, per-token linear FLOPs summed across all layers.

    Diagnostic surface: sum of attention-projection + FFN per-token costs
    as seen by one rank. For per-step cost under DP-attention use
    `decode_model.compute_flops` which applies the correct per-rank batch
    divisor (B / G_TP) to the attention portion (the per-token formula
    here treats DP-attn as "full per-token cost replicated on all ranks"
    via D_attn = 1; the per-step composition fixes the batch sharding).

    `framework` selects MLA mode (`mla_mode`) and the attention dispatch
    pattern (`attention_mode`, `tp_ep_layout`) consumed by the sharding-
    factor helpers. `attention_mode` / `tp_ep_layout` are ignored for
    GQA / MHA models; `mla_mode` is ignored for non-MLA models.
    """
    return (
        attn_proj_flops_per_token(model, partition, framework)
        + ffn_flops_per_token(model, partition, framework)
    )
