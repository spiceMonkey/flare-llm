from dataclasses import dataclass

@dataclass
class PartitionSpec:
    """
    Parallel partitioning of the model across devices.
    Purely describes how we shard: PP, TP, EP, SP.
    DP is inferred from the total number of devices available.

    `attention_mode` selects between two parallelism patterns *inside the
    attention block only*:
      - "tp" (default): attention weights are TP-sharded by head; KV cache is
        head-sharded across TP ranks. Original derivation (§1–§7 of
        documentation/modeling/decode.md).
      - "dp": attention weights are replicated on every TP rank; KV cache is
        sequence-sharded across TP ranks; the per-layer attention all-reduce
        is replaced by a TP all-gather at the attention → FFN transition.
        Per-device KV bytes and attention FLOPs are invariant under the swap.
        See decode.md §8 for the full derivation.
    Dense FFN and MoE FFN remain TP-sharded / EP-sharded under both modes.

    **Topology limitation.** This spec keeps the orthogonal-axis invariant
    `replica_size = PP * TP * EP * SP`, i.e. TP and EP map to *different*
    GPUs within a replica. The dp_attention mode above is mathematically
    correct for that orthogonal layout. It does **not** represent DSv3 /
    SGLang production deployments where TP and EP share the same physical
    GPUs (TP=8 EP=8 on 8 GPUs, world size 8 not 64) — that pattern requires
    relaxing the orthogonality invariant and is deferred to a future
    extension (see scratch/model_specific_extensions.md).
    """

    PP: int
    TP: int
    EP: int
    SP: int
    attention_mode: str = "tp"