"""Shared analytical primitives used by both decode_model and prefill_model.

These functions encode the phase-agnostic physics of a transformer inference
pipeline: weight footprint, KV cache footprint, per-token linear FLOPs,
and collective communication cost. Each primitive is a pure function of
typed spec dataclasses — no side effects, no global state. See
`documentation/modeling/decode.md` and `.../prefill.md` for the symbol
conventions and derivations.
"""

from .collective_cost import (
    p2p_hop,
    ring_all_reduce,
    tree_all_reduce,
    ring_moe_all_to_all,
    tree_moe_all_to_all,
    ring_all_gather,
    aggregate_per_stage,
)
from .weight_footprint import (
    dense_weight_bytes,
    moe_weight_bytes,
    embedding_bytes,
)
from .kv_footprint import kv_bytes_per_seq
from .linear_flops import linear_flops_per_token

__all__ = [
    # collective_cost
    "p2p_hop",
    "ring_all_reduce",
    "tree_all_reduce",
    "ring_moe_all_to_all",
    "tree_moe_all_to_all",
    "ring_all_gather",
    "aggregate_per_stage",
    # footprints
    "dense_weight_bytes",
    "moe_weight_bytes",
    "embedding_bytes",
    "kv_bytes_per_seq",
    "linear_flops_per_token",
]
