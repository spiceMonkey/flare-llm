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

The co_located rows apply only when both axes are in use (TP > 1 and EP > 1,
see `overlaid`); with TP = 1 or EP = 1 the orthogonal rows apply.

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


def overlaid(partition: PartitionSpec, framework: FrameworkSpec) -> bool:
    """True when TP and EP are overlaid on the same devices: the co-located
    layout with both axes in use (TP > 1 and EP > 1).

    With TP = 1 or EP = 1 there is nothing to overlay, and a replica under
    the co-located layout IS the orthogonal replica: EP groups of TP devices,
    the batch divided across the groups, experts sharded TP * EP ways. Every
    divisor below takes the layout through this, so the two layouts agree in
    that case instead of pricing whole-batch attention on every device
    (TP = 1) or unsharded experts (EP = 1).
    """
    return framework.tp_ep_layout == "co_located" and partition.TP > 1 and max(1, partition.EP) > 1


def scatter_direct(partition: PartitionSpec, framework: FrameworkSpec) -> bool:
    """True when scatter-direct MoE dispatch is in effect (decode.md §5.2).

    Three conditions: the stack offers it (`moe_a2a_pattern == "scatter"`),
    attention is DP-sharded, and every device holds whole experts, which is
    the overlaid layout (co-located, TP = EP > 1). Under the orthogonal
    layout an EP group's experts are TP-sharded, and a TP-sharded FFN needs
    its input on every rank of the group and a reduce after it, so the
    gather pattern applies whatever the flag says; with EP = 1 there is no
    dispatch at all. A flag that cannot take effect is not an error (stacks
    ship it as a capability), it just prices the gather pattern.
    """
    return (framework.moe_a2a_pattern == "scatter" and framework.attention_mode == "dp"
            and overlaid(partition, framework))


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
    if overlaid(partition, framework):
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
    if overlaid(partition, framework):
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
      - tp_ep_layout='co_located' with TP != EP when both exceed 1. Co-located layouts overlay
        TP and EP on the same physical GPUs (e.g., DSr1 NVL72 TP=EP=8 on
        8 GPUs); asymmetric (TP, EP) on a co-located layout is not modeled.

    Both attention_mode values ('tp' and 'dp') are allowed under either
    layout. The co_located + tp_attention combination models the production
    pattern where TP heads and EP experts both shard across the same
    physical GPU set — see the module docstring's four-row lookup table
    for the resulting D_attn / D_exp / D_kv / N_replica semantics.
    """
    # With TP = 1 or EP = 1 there is nothing to overlay and the two layouts
    # are the same replica (a dense model run under a stack whose default
    # layout is co-located is the common case), so the rule applies only
    # when both axes are in use.
    if framework.tp_ep_layout == "co_located" and partition.TP > 1 and partition.EP > 1:
        if partition.TP != partition.EP:
            raise ValueError(
                "compose_check: framework.tp_ep_layout='co_located' requires "
                "partition.TP == partition.EP (TP and EP share the same "
                f"physical GPU set in production deployments); got TP="
                f"{partition.TP}, EP={partition.EP}"
            )


def run_check(model, system, partition: PartitionSpec, framework: FrameworkSpec, *, context_tokens: int) -> None:
    """Refuse a run the model cannot describe. Called by both calculators on entry.

    Raises ValueError when:
      - the (partition, framework) pair breaks `compose_check`;
      - EP exceeds the model's number of experts (1 for a dense model): the
        expert sharding was clamped while the batch was still divided by EP;
      - one replica needs more devices than the system has (the result used
        to be a throughput of zero with no error);
    and warns (UserWarning) when the context is longer than the model's
    stated maximum (`LlmModelSpec.max_seq_len`), which the run used to accept
    silently.
    """
    compose_check(partition, framework)
    n_exp = model.moe.n_experts if model.moe is not None else 1
    if partition.EP > n_exp:
        raise ValueError(
            f"EP={partition.EP} exceeds the {n_exp} expert(s) of {model.name}: expert parallelism places whole "
            "experts on separate devices, so EP cannot exceed their number (and is 1 for a dense model)"
        )
    n_rep = N_replica(partition, framework)
    if n_rep > system.num_devices:
        raise ValueError(
            f"one replica needs {n_rep} devices (PP={partition.PP}, TP={partition.TP}, EP={partition.EP}, "
            f"SP={partition.SP}, layout {framework.tp_ep_layout!r}) but the system has {system.num_devices}"
        )
    if context_tokens > model.max_seq_len:
        # A warning, not an error: the cost model is well defined past the
        # limit and what-if studies use it, but the run describes a context
        # the model as specified cannot take.
        import warnings
        warnings.warn(
            f"context of {context_tokens} tokens exceeds the max_seq_len of {model.max_seq_len} stated for "
            f"{model.name}: the result is for a context this model cannot accept",
            UserWarning, stacklevel=3,
        )


def decode_microbatch(B: int, PP: int) -> float:
    """Sequences one pipeline stage processes per pass (decode.md §7.2).

    `B` is the number of sequences decoding concurrently on one replica.
    Autoregressive decode cannot start a sequence's next token until its
    previous token has left the last stage, so the `B` sequences circulate
    through the `PP` stages as `min(B, PP)` microbatches, one per stage at
    a time:

        mb = B / PP   if B ≥ PP   (every stage busy, PP microbatches in flight)
        mb = 1        if B < PP   (B single-sequence microbatches, PP − B stages idle)

    Every per-pass quantity — FLOPs, weight and KV traffic, collective
    payloads, the LM head — is evaluated at `mb`, not `B`. At PP = 1 this
    is `B` exactly.
    """
    PP = max(1, PP)
    if B >= PP:
        return B / PP
    return float(min(B, 1))


def D_batch(partition: PartitionSpec, framework: FrameworkSpec) -> int:
    """Number of device groups the replica's batch is divided across.

    Under the orthogonal layout a replica is `EP` groups of `TP` devices.
    Each group carries its own copy of the attention and dense-FFN weights
    and its own KV cache, so the groups serve *different* sequences: the
    batch is divided `EP` ways for everything except the expert FFN, whose
    tokens are routed across all `TP · EP` expert shards by the all-to-all.
    Without this division every group would recompute the whole batch's
    attention and hold the whole batch's KV cache — `EP` copies of the same
    work.

    Under the co-located layout TP and EP are the same devices, so there is
    one group and the divisor is 1 (DP-attention's own `B / G_TP` batch
    shard is applied separately, where it always was).
    """
    if overlaid(partition, framework):
        return 1
    return max(1, partition.EP)


def group_batch(B: float, partition: PartitionSpec, framework: FrameworkSpec) -> float:
    """Sequences one device group handles: `B / D_batch`, floored at one.

    Same shape as `decode_microbatch`: with fewer sequences than groups,
    each busy group holds one sequence and the rest are idle, so the
    latency of a step is that of a group batch of 1, not of a fraction.
    """
    d = D_batch(partition, framework)
    if B >= d:
        return B / d
    return float(min(B, 1))


def attn_rank_batch(Bg: float, partition: PartitionSpec, framework: FrameworkSpec) -> float:
    """Sequences the busiest rank runs through attention: Bg / G_TP floored at one
    under DP-attention (a rank holds whole sequences); Bg under TP-attention."""
    if framework.attention_mode == "dp" and partition.TP > 1 and Bg > 0:
        return max(1.0, Bg / partition.TP)
    return Bg


def kv_batch(Bg: float, partition: PartitionSpec, framework: FrameworkSpec) -> float:
    """Multiplier for per-sequence quantities that already carry /D_kv
    (kv_bytes_per_seq, score/value FLOPs). Bg, except under DP-attention, where the
    busiest rank holds max(1, Bg/G_TP) WHOLE sequences = max(Bg, G_TP) shards."""
    if framework.attention_mode == "dp" and partition.TP > 1 and Bg > 0:
        return max(Bg, float(partition.TP))
    return Bg
