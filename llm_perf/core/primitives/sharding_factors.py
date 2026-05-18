"""Per-component effective sharding factors (D_*, G_*) and replica size.

Implements the unified deployment-knob abstraction documented in
`notation.md §1` and used throughout `decode.md §1.4 / §2.1 / §2.3 / §3.5 /
§5 / §6.1` and the symmetric prefill formulas.

The framework supports four production-relevant `(tp_ep_layout, attention_mode)`
configurations. The lookup table:

    tp_ep_layout  attention_mode  D_attn  D_exp(MoE)   D_kv          D_emb  G_TP    G_EP  N_replica
    orthogonal    tp              TP      TP*EP        TP   (head)   TP     TP(AR)  EP    PP*TP*EP*SP
    co_located    tp              TP      EP           TP   (head)   TP     TP(AR)  EP    PP*max(TP,EP)*SP
    orthogonal    dp              1       TP*EP        TP   (seq)    TP     TP(AG)  EP    PP*TP*EP*SP
    co_located    dp              1       EP           max(TP,EP)    TP     TP(AG)  EP    PP*max(TP,EP)*SP
                                                        (seq)

Dense FFN always uses D_exp = TP regardless of tp_ep_layout (no EP axis to
overlap). The fourth combination (co_located + tp) models the production
pattern where TP=EP overlay on the same physical GPU set (e.g., the DSr1 /
NVL72 panel-(b) deployment with TP=EP=8 on 8 GPUs) and attention is
head-sharded across the same group. The TP all-reduce and the MoE all-to-all
both run on that shared 8-GPU set; the structural invariant `TP == EP`
under co-location still holds.

Phase H: `attention_mode` and `tp_ep_layout` live on FrameworkSpec, not on
PartitionSpec. Every helper here takes both `partition` (PP/TP/EP/SP) and
`framework` (attention_mode/tp_ep_layout). At the default
(orthogonal + tp) every helper collapses to the legacy raw TP / EP / SP form.
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
      - "moe" → TP*EP under orthogonal tp_ep_layout, EP under co-located.

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
    if framework.tp_ep_layout == "co_located":
        return EP
    return partition.TP * EP


def D_kv(partition: PartitionSpec, framework: FrameworkSpec) -> int:
    """Effective per-device divisor for KV cache (head- or sequence-shard).

    Excludes the SP axis. Per-device KV memory and traffic always carry an
    additional /SP factor on top of D_kv.

    Returns:
      - TP under orthogonal tp_ep_layout (head-shard under TP-attn or
        sequence-shard across the TP-as-DP-attn group under DP-attn —
        same byte count).
      - max(TP, EP) under co-located tp_ep_layout. Under DP-attention this
        is the sequence-shard across the entire replica's GPU set; under
        TP-attention (TP == EP forced under co-location) this is the head-
        shard across the TP group, which is the same set of ranks as the
        EP group — numerically equal to max(TP, EP) since TP == EP.
    """
    if framework.tp_ep_layout == "co_located":
        return max(partition.TP, max(1, partition.EP))
    return partition.TP


def D_emb(partition: PartitionSpec) -> int:
    """Effective per-device divisor for embedding / LM head weights.

    Always TP — embeddings and LM head are TP-sharded along the vocab
    dimension regardless of tp_ep_layout or attention_mode.
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

    Orthogonal tp_ep_layout: PP * TP * EP * SP.
    Co-located tp_ep_layout: PP * max(TP, EP) * SP (TP and EP overlay on the
    same physical GPU set).
    """
    EP = max(1, partition.EP)
    if framework.tp_ep_layout == "co_located":
        return partition.PP * max(partition.TP, EP) * partition.SP
    return partition.PP * partition.TP * EP * partition.SP


def compose_check(partition: PartitionSpec, framework: FrameworkSpec) -> None:
    """Validate the combined `(partition, framework)` invariants.

    One structural invariant applies at the (partition, framework) join
    point. Calculator paths that consume both specs should call this once
    at the start of pipeline assembly.

    Raises ValueError on:
      - tp_ep_layout='co_located' with TP != EP. Co-located layouts overlay
        TP and EP on the same physical GPUs (e.g., DSr1 NVL72 TP=EP=8 on
        8 GPUs); asymmetric (TP, EP) on a co-located layout is not modeled.

    Both attention_mode values ('tp' and 'dp') are allowed under either
    layout. The co_located + tp_attention combination models the production
    pattern where TP heads and EP experts both shard across the same
    physical GPU set — see the module docstring's four-row lookup table
    for the resulting D_attn / D_exp / D_kv / N_replica semantics.
    """
    if framework.tp_ep_layout == "co_located":
        if partition.TP != partition.EP:
            raise ValueError(
                "compose_check: framework.tp_ep_layout='co_located' requires "
                "partition.TP == partition.EP (TP and EP share the same "
                f"physical GPU set in production deployments); got TP="
                f"{partition.TP}, EP={partition.EP}"
            )
