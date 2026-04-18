"""Topology-aware collective cost dispatcher — switching.md §10.

Replaces the old `span_tiers(...) -> (α_sum, BW_min)` flatten-then-apply
pattern with a tier walk that picks each tier's contribution based on its
`.topology` discriminator. The walker's semantics match `span_tiers`
exactly: tiers are traversed innermost-first, and the collective is said
to "cross" a tier once the cumulative reach $\\prod \\mathrm{ports}$ is
below the group size. `G` is never clamped — primitives consume the
caller's `G` regardless of cumulative reach (preserves span_tiers
behavior for small out-of-bounds cases).

**Bit-identity guarantee (Phase C, see scratch/switching_upgrade.md §10.4):**
For a tier chain where every crossed tier is a `CrossbarTier`, this
function returns *exactly* the same float as the pre-refactor pattern
`ring_or_tree(M, G, *span_tiers(tiers, G)[:2])`. The crossbar branch
below reproduces that flatten step verbatim — no rounding, no
intermediate arithmetic reshuffling.

Torus and dragonfly tiers route into their respective primitives.
Pure-dragonfly tier sets (a single DragonflyTier, matching the canonical
balanced dragonfly parameterization) dispatch to `dragonfly_*`; multi-
dragonfly chains and genuinely mixed crossbar/torus/dragonfly chains
still fall back to the crossbar-flatten bound with a `UserWarning`
(Phase E expands coverage to explicit hybrid compositions).
"""

from __future__ import annotations

import warnings
from typing import List, Tuple

from ...specs.system_spec import TierSpec, span_tiers
from ...utils import GB_TO_BYTES, US_TO_SECONDS
from .collective_cost import (
    dragonfly_all_gather,
    dragonfly_all_reduce,
    dragonfly_moe_all_to_all,
    p2p_hop,
    ring_all_gather,
    ring_all_reduce,
    ring_moe_all_to_all,
    torus_all_gather,
    torus_all_reduce,
    torus_moe_all_to_all,
    tree_all_reduce,
    tree_moe_all_to_all,
)

# Accepted `op` values.
_OPS = ("all_reduce", "all_gather", "moe_a2a", "p2p")


def cost_collective(
    tiers: List[TierSpec],
    op: str,
    M: float,
    G: int,
    algorithm: str = "ring",
    torus_algorithm: str = "ring",
    worst_case: bool = False,
) -> float:
    """Cost a collective of group size `G` bytes `M` over `tiers`, in seconds.

    Args:
      tiers: innermost-first tier list (typically `system.get_tier_chain(coll)`).
      op: one of "all_reduce", "all_gather", "moe_a2a", "p2p".
      M: per-rank message size in bytes.
      G: collective group size (number of ranks).
      algorithm: "ring" or "tree" — crossbar-path choice for AR / MoE-A2A.
        Ignored for `all_gather` (always ring) and `p2p` (single hop), and
        on torus tiers (see `torus_algorithm`).
      torus_algorithm: "ring" (dim-by-dim) or "swing". "swing" raises
        `NotImplementedError` — reserved for switching.md §8.7 follow-up.
      worst_case: dragonfly Valiant-routing flag (Phase D); currently
        ignored since dragonfly tiers fall back to crossbar-flatten.

    Returns:
      Collective time in seconds. Returns 0.0 for G <= 1 or empty tier list.
    """
    if op not in _OPS:
        raise ValueError(f"Unknown op: {op!r}; allowed: {_OPS}")
    if G <= 1 or not tiers:
        return 0.0

    # Walk tiers innermost-first, collecting those the collective crosses.
    # Matches `span_tiers` semantics: stop once cumulative reach ≥ G.
    crossed: List[TierSpec] = []
    reach = 1
    for tier in tiers:
        reach *= max(1, tier.ports)
        crossed.append(tier)
        if reach >= G:
            break

    topologies = {t.topology for t in crossed}

    # Pure crossbar: delegate to span_tiers so the α-sum / BW-min
    # accumulation order is byte-identical to the pre-refactor path
    # (span_tiers uses `+=` / iterative `min`; a Python `sum()` / `min()`
    # call here drifts by 1-2 ULP on randomized float inputs).
    if topologies == {"crossbar"}:
        alpha_us, bw_GBps, _ = span_tiers(tiers, G)
        return _crossbar_cost(
            op, M, G, alpha_us * US_TO_SECONDS, bw_GBps * GB_TO_BYTES, algorithm
        )

    # Pure torus: dispatch to torus primitives (Phase B).
    if topologies == {"torus"}:
        if torus_algorithm == "swing":
            raise NotImplementedError(
                "torus_algorithm='swing' not yet implemented; "
                "see switching.md §8.7."
            )
        if torus_algorithm != "ring":
            raise ValueError(
                f"Unsupported torus_algorithm: {torus_algorithm!r}"
            )
        return _torus_cost(op, M, G, crossed)

    # Pure dragonfly, single tier: dispatch to dragonfly primitives (Phase D).
    # Multi-dragonfly chains fall through to the mixed-topology fallback —
    # their hierarchical composition isn't in §3.4 and is reserved for a
    # future pass.
    if topologies == {"dragonfly"} and len(crossed) == 1:
        return _dragonfly_cost(op, M, G, crossed[0], algorithm, worst_case)

    # Anything else (multi-dragonfly, or genuinely mixed): Phase D does not
    # cost these precisely. Fall back to crossbar-flatten so the interface is
    # live and numerically bounded; emit a warning so callers notice.
    warnings.warn(
        f"cost_collective: topology mix {sorted(topologies)} not yet "
        f"modeled; falling back to crossbar-flatten bound. "
        f"See switching.md §10 (Phase D adds dragonfly primitives).",
        UserWarning,
        stacklevel=2,
    )
    alpha_us, bw_GBps, _ = span_tiers(tiers, G)
    return _crossbar_cost(
        op, M, G, alpha_us * US_TO_SECONDS, bw_GBps * GB_TO_BYTES, algorithm
    )


# ────────────────────────────────────────────────────────────
# Crossbar path (bit-identical to pre-refactor span_tiers flatten)
# ────────────────────────────────────────────────────────────

def _crossbar_cost(
    op: str, M: float, G: int, alpha_s: float, bw_Bps: float, algorithm: str
) -> float:
    if op == "p2p":
        return p2p_hop(M, alpha_s, bw_Bps)
    if op == "all_gather":
        return ring_all_gather(M, G, alpha_s, bw_Bps)
    if op == "all_reduce":
        if algorithm == "ring":
            return ring_all_reduce(M, G, alpha_s, bw_Bps)
        if algorithm == "tree":
            return tree_all_reduce(M, G, alpha_s, bw_Bps)
        raise ValueError(f"Unsupported algorithm for all_reduce: {algorithm!r}")
    if op == "moe_a2a":
        if algorithm == "ring":
            return ring_moe_all_to_all(M, G, alpha_s, bw_Bps)
        if algorithm == "tree":
            return tree_moe_all_to_all(M, G, alpha_s, bw_Bps)
        raise ValueError(f"Unsupported algorithm for moe_a2a: {algorithm!r}")
    raise AssertionError(f"unreachable op={op!r}")  # caller validated


# ────────────────────────────────────────────────────────────
# Torus path (Phase B primitives)
# ────────────────────────────────────────────────────────────

def _torus_cost(op: str, M: float, G: int, crossed: List[TierSpec]) -> float:
    """Cost a collective on an all-torus tier chain.

    Multi-tier torus chains concatenate their `dims` tuples into the effective
    k-D structure the dim-by-dim primitives consume. Dim-alignment is checked
    against prefix-products of the concatenated dims; misalignment falls back
    to a flat-ring conservative bound with a `UserWarning`.
    """
    full_dims: Tuple[int, ...] = tuple(
        d for t in crossed for d in t.dims  # type: ignore[attr-defined]
    )
    # α / BW aggregation for multi-tier torus: sum α (every tier-boundary
    # adds a hop), take min BW (narrowest tier binds). Delegate to
    # span_tiers for identical accumulation order to the crossbar path.
    alpha_us, bw_GBps, _ = span_tiers(crossed, G)
    alpha_s = alpha_us * US_TO_SECONDS
    bw_Bps = bw_GBps * GB_TO_BYTES

    subdims, aligned = _align_to_dims(G, full_dims)
    if not aligned:
        warnings.warn(
            f"Torus collective G={G} misaligned with dims={full_dims}; "
            f"using conservative flat-ring bound. "
            f"See switching.md §8.3 for dim-aligned layouts.",
            UserWarning,
            stacklevel=3,
        )
        # Conservative bound: flat-ring on G ranks, ignoring dim structure.
        return _crossbar_cost(op, M, G, alpha_s, bw_Bps, algorithm="ring")

    if op == "p2p":
        # Torus p2p is a single-hop forward (PP stage handoff). Use α+M/BW.
        return p2p_hop(M, alpha_s, bw_Bps)
    if op == "all_reduce":
        return torus_all_reduce(M, subdims, alpha_s, bw_Bps)
    if op == "all_gather":
        return torus_all_gather(M, subdims, alpha_s, bw_Bps)
    if op == "moe_a2a":
        return torus_moe_all_to_all(M, subdims, alpha_s, bw_Bps)
    raise AssertionError(f"unreachable op={op!r}")  # caller validated


# ────────────────────────────────────────────────────────────
# Dragonfly path (Phase D primitives)
# ────────────────────────────────────────────────────────────

def _dragonfly_cost(
    op: str, M: float, G: int, tier, algorithm: str, worst_case: bool
) -> float:
    """Cost a collective on a single DragonflyTier using the three-tier
    hierarchical decomposition (switching.md §9.4).

    The full tier reach is `p·a·g`; for `G < reach` we truncate the tier
    triple to the innermost levels that cover `G`. A group of size `G ≤ p`
    stays inside one router; `p < G ≤ p·a` fills one group; larger `G` uses
    the full (p, a, g_eff) triple. This keeps dispatcher semantics aligned
    with span_tiers: `G` is the authoritative group size, and the cost
    formula only includes tiers the collective actually crosses.
    """
    p = tier.p_endpoints
    a = tier.a_routers
    g = tier.g_groups
    if G <= p:
        p_eff, a_eff, g_eff = G, 1, 1
    elif G <= p * a:
        # G covers at most one group — L1 rings the subset of routers needed.
        # Integer alignment (G multiple of p) gives exact; misaligned G is
        # conservatively rounded up (one extra router participates).
        p_eff = p
        a_eff = (G + p - 1) // p
        g_eff = 1
    else:
        p_eff = p
        a_eff = a
        g_eff = (G + p * a - 1) // (p * a)
        if g_eff > g:
            g_eff = g

    alpha_r_s = tier.alpha_us * US_TO_SECONDS
    bw_r_Bps = tier.bw_per_port_GBps * GB_TO_BYTES
    alpha_l_s = tier.alpha_local_us * US_TO_SECONDS
    bw_l_Bps = tier.bw_local_GBps * GB_TO_BYTES
    alpha_g_s = tier.alpha_global_us * US_TO_SECONDS
    bw_g_Bps = tier.bw_global_GBps * GB_TO_BYTES

    if op == "p2p":
        # Conservative single-hop: use the slowest tier crossed.
        if g_eff > 1:
            return p2p_hop(M, alpha_g_s, bw_g_Bps)
        if a_eff > 1:
            return p2p_hop(M, alpha_l_s, bw_l_Bps)
        return p2p_hop(M, alpha_r_s, bw_r_Bps)
    if op == "all_reduce":
        if algorithm not in ("ring", "tree"):
            raise ValueError(f"Unsupported algorithm for all_reduce: {algorithm!r}")
        # Tree variant is not modeled per-tier on dragonfly in this pass;
        # fall back to the hierarchical ring (conservative vs tree on large g).
        return dragonfly_all_reduce(
            M, p_eff, a_eff, g_eff,
            alpha_r_s, bw_r_Bps,
            alpha_l_s, bw_l_Bps,
            alpha_g_s, bw_g_Bps,
            worst_case=worst_case,
        )
    if op == "all_gather":
        return dragonfly_all_gather(
            M, p_eff, a_eff, g_eff,
            alpha_r_s, bw_r_Bps,
            alpha_l_s, bw_l_Bps,
            alpha_g_s, bw_g_Bps,
            worst_case=worst_case,
        )
    if op == "moe_a2a":
        if algorithm not in ("ring", "tree"):
            raise ValueError(f"Unsupported algorithm for moe_a2a: {algorithm!r}")
        return dragonfly_moe_all_to_all(
            M, p_eff, a_eff, g_eff,
            alpha_r_s, bw_r_Bps,
            alpha_l_s, bw_l_Bps,
            alpha_g_s, bw_g_Bps,
            worst_case=worst_case,
        )
    raise AssertionError(f"unreachable op={op!r}")  # caller validated


def _align_to_dims(G: int, dims: Tuple[int, ...]) -> Tuple[Tuple[int, ...], bool]:
    """Return (subdims, aligned) — the prefix-aligned sub-torus for `G` ranks.

    Dim-aligned means `G == prod(dims[:k])` for some k ≥ 1. That lets the
    dim-by-dim ring AR walk exactly `k` ring dims. If no prefix matches,
    returns the full dims with aligned=False so the caller can fall back.
    """
    prefix = 1
    acc: List[int] = []
    for d in dims:
        acc.append(d)
        prefix *= d
        if prefix == G:
            return tuple(acc), True
        if prefix > G:
            return dims, False
    # G exceeds prod(dims) — use full dims and flag as misaligned.
    return dims, prefix == G
