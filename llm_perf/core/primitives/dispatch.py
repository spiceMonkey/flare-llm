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

Torus tiers route into their torus primitives. Genuinely mixed
crossbar/torus chains fall back to the crossbar-flatten bound with a
`UserWarning` (explicit hybrid compositions are a follow-up).
"""

from __future__ import annotations

import warnings
from typing import List, Tuple

from ...specs.system_spec import TierSpec, span_tiers
from ...utils import GB_TO_BYTES, US_TO_SECONDS
from .collective_cost import (
    inc_a2a,
    inc_all_gather,
    inc_all_reduce,
    p2p_hop,
    pairwise_a2a,
    ring_all_gather,
    ring_all_reduce,
    torus_all_gather,
    torus_all_reduce,
    torus_moe_all_to_all,
    tree_all_reduce,
    tree_moe_all_to_all,
)

# Accepted `op` values.
_OPS = ("all_reduce", "all_gather", "moe_a2a", "p2p")


def _span_tiers_scaled(
    tiers: List[TierSpec], G: int
) -> Tuple[float, float, int]:
    """Like `span_tiers` but applies per-tier contention coefficients.

    Each tier contributes `alpha_us * tier.eta_alpha` to the α-sum and
    `bw_per_port_GBps * tier.eta_beta` to the BW-min. With the defaults
    `eta_alpha = eta_beta = 1.0` this is bit-identical to `span_tiers`
    (IEEE-754 `x * 1.0 == x` for all finite x). See
    documentation/modeling/contention.md §4.
    """
    if G <= 1 or not tiers:
        return 0.0, 0.0, 0
    alpha_total = 0.0
    bw_min = float("inf")
    reach = 1
    crossed = 0
    for tier in tiers:
        reach *= max(1, tier.ports)
        alpha_total += tier.alpha_us * tier.eta_alpha
        bw_min = min(bw_min, tier.bw_per_port_GBps * tier.eta_beta)
        crossed += 1
        if reach >= G:
            return alpha_total, bw_min, crossed
    return alpha_total, bw_min, crossed


def cost_collective(
    tiers: List[TierSpec],
    op: str,
    M: float,
    G: int,
    algorithm: str = "ring",
    torus_algorithm: str = "ring",
    inc_enabled: bool = True,
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
      inc_enabled: when True, route AR/AG over a crossbar chain whose every
        crossed tier declares `inc != "none"` to the INC primitives
        (n_α collapse + BW-eff doubling for AR). When False, force software
        ring/tree on the same chain. Inert on torus and on mixed chains.

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

    # Pure crossbar: delegate to _span_tiers_scaled so the α-sum / BW-min
    # accumulation order is byte-identical to the pre-refactor path
    # (span_tiers uses `+=` / iterative `min`; a Python `sum()` / `min()`
    # call here drifts by 1-2 ULP on randomized float inputs). With the
    # default η = 1.0 for every crossed tier, the scaled helper is
    # bit-identical to span_tiers (IEEE-754 `x * 1.0 == x`).
    if topologies == {"crossbar"}:
        # INC eligibility: every crossed tier must declare inc != "none"
        # AND the caller must not have opted out. AR / AG / RS route via the
        # SHARP-class switch ALU + multicast crossbar (collectives.md §3.4,
        # §4.4) — needs `inc != "none"` on every tier. A2A routes via HW
        # crossbar scatter-gather (§5.4) — needs `inc == "hw_a2a"` on every
        # tier (NOT covered by sharp_class). p2p is a single hop, no INC.
        if inc_enabled and all(_is_inc(t) for t in crossed):
            if op in ("all_reduce", "all_gather"):
                return _inc_crossbar_cost(op, M, G, crossed)
            if op == "moe_a2a" and all(_is_hw_a2a(t) for t in crossed):
                return _inc_crossbar_cost(op, M, G, crossed)
        alpha_us, bw_GBps, _ = _span_tiers_scaled(tiers, G)
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

    # Genuinely mixed crossbar/torus: fall back to crossbar-flatten so the
    # interface stays live and numerically bounded; emit a warning so callers
    # notice. Explicit hybrid compositions are a follow-up.
    warnings.warn(
        f"cost_collective: topology mix {sorted(topologies)} not yet "
        f"modeled; falling back to crossbar-flatten bound.",
        UserWarning,
        stacklevel=2,
    )
    alpha_us, bw_GBps, _ = _span_tiers_scaled(tiers, G)
    return _crossbar_cost(
        op, M, G, alpha_us * US_TO_SECONDS, bw_GBps * GB_TO_BYTES, algorithm
    )


# ────────────────────────────────────────────────────────────
# Crossbar path (bit-identical to pre-refactor span_tiers flatten)
# ────────────────────────────────────────────────────────────

def _is_inc(tier: TierSpec) -> bool:
    """Tier supports any INC capability (sharp_class or hw_a2a)."""
    return getattr(tier, "inc", "none") != "none"


def _is_hw_a2a(tier: TierSpec) -> bool:
    """Tier supports HW A2A specifically (Tomahawk Ultra / Rubin)."""
    return getattr(tier, "inc", "none") == "hw_a2a"


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
            return pairwise_a2a(M, G, alpha_s, bw_Bps)
        if algorithm == "tree":
            return tree_moe_all_to_all(M, G, alpha_s, bw_Bps)
        raise ValueError(f"Unsupported algorithm for moe_a2a: {algorithm!r}")
    raise AssertionError(f"unreachable op={op!r}")  # caller validated


# ────────────────────────────────────────────────────────────
# INC path — crossbar tiers with switch-hosted reduction
# ────────────────────────────────────────────────────────────

def _inc_crossbar_cost(
    op: str, M: float, G: int, crossed: List[TierSpec]
) -> float:
    """Route AR/AG/A2A on an INC-enabled crossbar chain through the INC primitives.

    α aggregation: sum `inc_alpha_us` (or `alpha_us` when the override is 0)
    across every crossed tier, scaled by each tier's `eta_alpha`. This maps
    to the k-tier scale-out depth: one α contribution per tier level, with
    the round-trip factor applied inside the primitive.

    BW: narrowest link among crossed tiers, scaled by each tier's `eta_beta`.
    Matches the crossbar α/BW reduction and lets contention modeling
    (collectives.md §7) flow through identically.

    Routing per op (collectives.md §3.4 / §4.4 / §5.4):
      - AR: switch ALU + multicast crossbar; both n_α collapse AND BW-eff
        doubling. Routes through `inc_all_reduce`.
      - AG / RS: multicast or ALU+scatter; α-only collapse, BW unchanged
        (software ring already hits BW_eff = BW). Routes through
        `inc_all_gather` / `inc_reduce_scatter`.
      - A2A: HW crossbar scatter-gather; α-only collapse, BW stays at the
        bisection bound. Routes through `inc_a2a`. Caller (eligibility
        check in cost_collective) is responsible for verifying every tier
        is `hw_a2a` — this branch only fires when the gating allows it.
    """
    alpha_us_total = 0.0
    bw_GBps_min = float("inf")
    for t in crossed:
        per_tier_alpha_us = t.inc_alpha_us if t.inc_alpha_us > 0.0 else t.alpha_us
        alpha_us_total += per_tier_alpha_us * t.eta_alpha
        bw_GBps_min = min(bw_GBps_min, t.bw_per_port_GBps * t.eta_beta)

    alpha_s = alpha_us_total * US_TO_SECONDS
    bw_Bps = bw_GBps_min * GB_TO_BYTES

    if op == "all_reduce":
        return inc_all_reduce(M, alpha_s, bw_Bps)
    if op == "all_gather":
        return inc_all_gather(M, G, alpha_s, bw_Bps)
    if op == "moe_a2a":
        return inc_a2a(M, G, alpha_s, bw_Bps)
    raise AssertionError(f"INC path reached with unsupported op={op!r}")


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
    # adds a hop), take min BW (narrowest tier binds). Delegate to the
    # scaled helper for identical accumulation order to the crossbar path
    # and to pick up any per-tier contention coefficients (η=1 default
    # preserves bit-identity with the pre-contention path).
    alpha_us, bw_GBps, _ = _span_tiers_scaled(crossed, G)
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
