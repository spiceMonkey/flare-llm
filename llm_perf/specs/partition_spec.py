from dataclasses import dataclass


_VALID_LAYOUTS = ("orthogonal", "co_located")
_VALID_ATTENTION_MODES = ("tp", "dp")


@dataclass
class PartitionSpec:
    """
    Parallel partitioning of the model across devices.
    Purely describes how we shard: PP, TP, EP, SP.
    DP is inferred from the total number of devices available.

    `layout` selects between two physical mappings of TP and EP to GPUs:
      - "orthogonal" (default): TP and EP map to *different* GPU sets.
        Replica size = PP * TP * EP * SP.
      - "co_located": TP and EP overlay on the *same* GPU set (DSv3 / SGLang
        / NVIDIA Dynamo production decode pattern). Replica size =
        PP * max(TP, EP) * SP. Co-location forces `attention_mode = "dp"`
        because no separate TP-only group exists for head-sharding to land
        on; the constructor enforces this invariant.

    `attention_mode` selects between two parallelism patterns *inside the
    attention block only*:
      - "tp" (default): attention weights are TP-sharded by head; KV cache is
        head-sharded across TP ranks.
      - "dp": attention weights are replicated on every TP rank; KV cache is
        sequence-sharded across the DP-attn group; the per-layer attention
        all-reduce is replaced by a TP all-gather at the attention → FFN
        transition. Per-device KV bytes and attention FLOPs are invariant
        under the orthogonal TP-attn ↔ DP-attn swap.

    Dense FFN remains TP-sharded under both layouts. MoE expert weights are
    sharded by TP*EP under orthogonal and by EP only under co-located (no
    further TP shard within an expert).

    See `notation.md §1` for the unified deployment-knob abstraction (the
    per-component effective sharding factors D_attn, D_exp, D_kv, D_emb that
    encode all three production-relevant configurations in one lookup table)
    and `decode.md §1.4 / §5.3` for the per-device formulas. The framework
    helpers in `core/primitives/sharding_factors.py` resolve those abstract
    factors from a PartitionSpec instance.
    """

    PP: int
    TP: int
    EP: int
    SP: int
    attention_mode: str = "tp"
    layout: str = "orthogonal"

    def __post_init__(self):
        if self.layout not in _VALID_LAYOUTS:
            raise ValueError(
                f"PartitionSpec.layout must be one of {_VALID_LAYOUTS}, got {self.layout!r}"
            )
        if self.attention_mode not in _VALID_ATTENTION_MODES:
            raise ValueError(
                f"PartitionSpec.attention_mode must be one of {_VALID_ATTENTION_MODES}, "
                f"got {self.attention_mode!r}"
            )
        if self.layout == "co_located":
            if self.attention_mode != "dp":
                raise ValueError(
                    "PartitionSpec layout='co_located' forces attention_mode='dp' "
                    "(no separate TP group exists for head-sharding to land on under "
                    "co-location); got attention_mode=" + repr(self.attention_mode)
                )
            # Co-location has TP and EP overlaid on the same physical GPUs.
            # Production deployments use TP == EP (e.g. DSv3 with TP=EP=8);
            # asymmetric (TP, EP) on a co-located layout is not modeled here.
            if self.TP != self.EP:
                raise ValueError(
                    f"PartitionSpec layout='co_located' requires TP == EP (TP and EP "
                    f"share the same physical GPU set in production deployments); "
                    f"got TP={self.TP}, EP={self.EP}"
                )