from dataclasses import dataclass


@dataclass
class PartitionSpec:
    """Parallel partitioning of the model across devices.

    Purely describes how we shard: PP, TP, EP, SP. DP is inferred from the
    total number of devices available.

    Phase H: `attention_mode` and `tp_ep_layout` (formerly `layout`) were
    moved to FrameworkSpec — they describe how the stack dispatches the
    attention block and how TP / EP map to physical GPUs, both stack-axis
    decisions, not sharding-factor decisions. PartitionSpec now carries
    only the four parallelism factors. The compose-time invariants formerly
    enforced here (co-located tp_ep_layout requires TP == EP and
    attention_mode == "dp") are now enforced by
    `core.primitives.sharding_factors.compose_check(partition, framework)`.

    See `notation.md §1` for the unified deployment-knob abstraction (the
    per-component effective sharding factors D_attn, D_exp, D_kv, D_emb that
    combine PartitionSpec with FrameworkSpec to encode all three production-
    relevant configurations in one lookup table) and `decode.md §1.4 / §5.3`
    for the per-device formulas. The framework helpers in
    `core/primitives/sharding_factors.py` resolve those abstract factors from
    the (PartitionSpec, FrameworkSpec) join.
    """

    PP: int
    TP: int
    EP: int
    SP: int
