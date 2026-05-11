"""Per-component effective sharding factors (D_*, G_*) and replica size.

Implements the unified deployment-knob abstraction documented in
`notation.md §1` and used throughout `decode.md §1.4 / §2.1 / §2.3 / §3.5 /
§5 / §6.1` and the symmetric prefill formulas.

The framework supports three production-relevant `(layout, attention_mode)`
configurations. The lookup table:

    layout       attention_mode  D_attn  D_exp(MoE)   D_kv          D_emb  G_TP    G_EP  N_replica
    orthogonal   tp              TP      TP*EP        TP   (head)   TP     TP(AR)  EP    PP*TP*EP*SP
    orthogonal   dp              1       TP*EP        TP   (seq)    TP     TP(AG)  EP    PP*TP*EP*SP
    co_located   dp              1       EP           max(TP,EP)    TP     TP(AG)  EP    PP*max(TP,EP)*SP
                                                       (seq)

Dense FFN always uses D_exp = TP regardless of layout (no EP axis to overlap).
A theoretical fourth combination (co_located + tp) is rejected by the
combined `(partition, framework)` invariant check (see compose_check below)
because no separate TP group exists for head-sharding to land on under
co-location.

Phase H: `attention_mode` and `layout` live on FrameworkSpec, not on
PartitionSpec. Every helper here takes both `partition` (PP/TP/EP/SP) and
`framework` (attention_mode/layout). At the default (orthogonal + tp) every
helper collapses to the legacy raw TP / EP / SP form.
"""

from ...specs.framework_spec import FrameworkSpec
from ...specs.partition_spec import PartitionSpec


def D_attn(partition: PartitionSpec, framework: FrameworkSpec) -> int:
    """Effective per-device divisor for attention weights (Q/K/V/O).

    Returns TP under TP-attention (head-shard); 1 under DP-attention
    (weights replicated on every TP rank).
    """
    if framework.attention_mode == "dp":
        return 1
    return partition.TP


def D_exp(
    partition: PartitionSpec,
    framework: FrameworkSpec,
    *,
    layer_kind: str = "moe",
    n_exp_cap: int | None = None,
) -> int:
    """Effective per-device divisor for FFN / expert weights.

    `layer_kind` selects the divisor convention for the layer in question:
      - "dense" → always TP (no EP axis to overlap; co-location does not apply
        to dense layers).
      - "moe" → TP*EP under orthogonal layout, EP under co-located layout.

    `n_exp_cap` (MoE only) clamps the effective EP by the number of experts
    in the model. Required when EP > N_exp would otherwise inflate the
    divisor beyond what is physically meaningful: with EP=8 but only 2
    experts, only 2 ranks can hold a unique expert, so the divisor is 2 not
    8. Pass `model.moe.n_experts` from the call site.

    For a model with mixed dense + MoE layers, callers compute each layer's
    contribution with the matching `layer_kind`.
    """
    if layer_kind == "dense":
        return partition.TP
    if layer_kind != "moe":
        raise ValueError(f"D_exp: layer_kind must be 'dense' or 'moe', got {layer_kind!r}")
    EP = max(1, partition.EP)
    if n_exp_cap is not None:
        EP = min(EP, max(1, n_exp_cap))
    if framework.layout == "co_located":
        return EP
    return partition.TP * EP


def D_kv(partition: PartitionSpec, framework: FrameworkSpec) -> int:
    """Effective per-device divisor for KV cache (head- or sequence-shard).

    Excludes the SP axis. Per-device KV memory and traffic always carry an
    additional /SP factor on top of D_kv.

    Returns:
      - TP under orthogonal layout (head-shard under TP-attn or sequence-shard
        across the TP-as-DP-attn group under DP-attn — same byte count).
      - max(TP, EP) under co-located layout (sequence-shard across the entire
        replica's GPU set).
    """
    if framework.layout == "co_located":
        return max(partition.TP, max(1, partition.EP))
    return partition.TP


def D_emb(partition: PartitionSpec) -> int:
    """Effective per-device divisor for embedding / LM head weights.

    Always TP — embeddings and LM head are TP-sharded along the vocab
    dimension regardless of layout or attention mode.
    """
    return partition.TP


def G_TP(partition: PartitionSpec) -> int:
    """Collective group size for TP collectives (AR or AG)."""
    return partition.TP


def G_EP(partition: PartitionSpec) -> int:
    """Collective group size for EP all-to-all."""
    return max(1, partition.EP)


def N_replica(partition: PartitionSpec, framework: FrameworkSpec) -> int:
    """Devices per model replica.

    Orthogonal layout: PP * TP * EP * SP.
    Co-located layout: PP * max(TP, EP) * SP (TP and EP overlay on the same
    physical GPU set).
    """
    EP = max(1, partition.EP)
    if framework.layout == "co_located":
        return partition.PP * max(partition.TP, EP) * partition.SP
    return partition.PP * partition.TP * EP * partition.SP


def compose_check(partition: PartitionSpec, framework: FrameworkSpec) -> None:
    """Validate the combined `(partition, framework)` invariants.

    Two compose-time invariants — formerly enforced inside
    PartitionSpec.__post_init__ when `attention_mode` and `layout` lived
    there — now apply at the (partition, framework) join point. Calculator
    paths that consume both specs should call this once at the start of
    pipeline assembly.

    Raises ValueError on:
      - co-located layout with TP != EP (production deployments overlay
        TP and EP on the same physical GPUs and use TP == EP, e.g. DSv3
        TP=EP=8); asymmetric (TP, EP) on a co-located layout is not
        modeled.
      - co-located layout with attention_mode != "dp" (no separate TP
        group exists for head-sharding to land on under co-location).
    """
    if framework.layout == "co_located":
        if framework.attention_mode != "dp":
            raise ValueError(
                "compose_check: framework.layout='co_located' forces "
                "framework.attention_mode='dp' (no separate TP group exists "
                "for head-sharding to land on under co-location); got "
                f"attention_mode={framework.attention_mode!r}"
            )
        if partition.TP != partition.EP:
            raise ValueError(
                "compose_check: framework.layout='co_located' requires "
                "partition.TP == partition.EP (TP and EP share the same "
                f"physical GPU set in production deployments); got TP="
                f"{partition.TP}, EP={partition.EP}"
            )
