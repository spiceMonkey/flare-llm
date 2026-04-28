"""Shared analytical primitives used by both decode_model and prefill_model.

These functions encode the phase-agnostic physics of a transformer inference
pipeline: weight footprint, KV cache footprint, per-token linear FLOPs,
and collective communication cost. Each primitive is a pure function of
typed spec dataclasses — no side effects, no global state. See
`documentation/modeling/decode.md` and `.../prefill.md` for the symbol
conventions and derivations.

The collective_cost primitives are auto-synced from
``spiceMonkey/collective-comm`` (see the AUTO-SYNCED banner at the top of
``collective_cost.py`` and ``.github/workflows/sync-collectives.yml``).
This module re-exports the full upstream surface grouped by category
(BC / Reduce / AR / AG / RS / A2A / hierarchical / contention) for ease
of discovery.
"""

from .collective_cost import (
    # Point-to-point
    p2p_hop,
    # Broadcast
    ring_broadcast,
    tree_broadcast,
    inc_broadcast,
    torus_broadcast,
    # Reduce
    ring_reduce,
    tree_reduce,
    inc_reduce,
    torus_reduce,
    # All-reduce
    ring_all_reduce,
    tree_all_reduce,
    rabenseifner_all_reduce,
    inc_all_reduce,
    torus_all_reduce,
    # All-gather
    ring_all_gather,
    recursive_doubling_all_gather,
    pat_all_gather,
    inc_all_gather,
    torus_all_gather,
    # Reduce-scatter
    ring_reduce_scatter,
    recursive_halving_reduce_scatter,
    pat_reduce_scatter,
    inc_reduce_scatter,
    torus_reduce_scatter,
    # All-to-all
    pairwise_a2a,
    ring_relay_a2a,
    bruck_a2a,
    inc_a2a,
    torus_a2a,
    # Hierarchical composition
    hierarchical_all_reduce,
    hierarchical_all_reduce_ring_ring,
    hierarchical_all_gather,
    hierarchical_reduce_scatter,
    # Contention coefficient helpers
    apply_eta,
    realistic_cost,
)
from .stage_aggregator import aggregate_per_stage
from .weight_footprint import (
    dense_weight_bytes,
    moe_weight_bytes,
    embedding_bytes,
)
from .kv_footprint import kv_bytes_per_seq
from .linear_flops import linear_flops_per_token
from .dispatch import cost_collective, enumerate_options
from .partition_layout import (
    DEFAULT_ORDER as NESTED_LAYOUT_ORDER,
    assign_tier_per_axis,
    tier_at,
)

__all__ = [
    # P2P
    "p2p_hop",
    # Broadcast
    "ring_broadcast",
    "tree_broadcast",
    "inc_broadcast",
    "torus_broadcast",
    # Reduce
    "ring_reduce",
    "tree_reduce",
    "inc_reduce",
    "torus_reduce",
    # All-reduce
    "ring_all_reduce",
    "tree_all_reduce",
    "rabenseifner_all_reduce",
    "inc_all_reduce",
    "torus_all_reduce",
    # All-gather
    "ring_all_gather",
    "recursive_doubling_all_gather",
    "pat_all_gather",
    "inc_all_gather",
    "torus_all_gather",
    # Reduce-scatter
    "ring_reduce_scatter",
    "recursive_halving_reduce_scatter",
    "pat_reduce_scatter",
    "inc_reduce_scatter",
    "torus_reduce_scatter",
    # All-to-all
    "pairwise_a2a",
    "ring_relay_a2a",
    "bruck_a2a",
    "inc_a2a",
    "torus_a2a",
    # Hierarchical composition
    "hierarchical_all_reduce",
    "hierarchical_all_reduce_ring_ring",
    "hierarchical_all_gather",
    "hierarchical_reduce_scatter",
    # Contention coefficient helpers
    "apply_eta",
    "realistic_cost",
    # llm_perf-local stage aggregator
    "aggregate_per_stage",
    # footprints
    "dense_weight_bytes",
    "moe_weight_bytes",
    "embedding_bytes",
    "kv_bytes_per_seq",
    "linear_flops_per_token",
    # dispatcher
    "cost_collective",
    "enumerate_options",
    # nested-layout helper
    "NESTED_LAYOUT_ORDER",
    "assign_tier_per_axis",
    "tier_at",
]
