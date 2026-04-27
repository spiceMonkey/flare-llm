"""Topology-aware collective cost dispatcher — collectives.md §3–§6.

Walks a fabric chain innermost-first, picking each tier's contribution
based on its `.topology` discriminator. Crossbar tiers flatten to a single
$(\\alpha_\\mathrm{sum}, \\mathrm{BW}_\\min)$ pair consumed by the star
primitives in `collective_cost.py`. Torus tiers route into the dim-decomposed
primitives. Tiers are traversed innermost-first; a collective is said to
"cross" a tier once cumulative reach $\\prod \\mathrm{ports}$ is below the
group size. `G` is never clamped — primitives consume the caller's `G`
regardless of cumulative reach.

**Bit-identity guarantee.** For a tier chain where every crossed tier is a
`CrossbarTier`, this function returns *exactly* the same float as the
pattern `ring_or_tree(M, G, *span_tiers(tiers, G)[:2])`. The crossbar
branch below reproduces that flatten step verbatim — no rounding, no
intermediate arithmetic reshuffling.

Genuinely mixed crossbar/torus chains fall back to the crossbar-flatten
bound with a `UserWarning` (explicit hybrid compositions are a follow-up).
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
    ring_reduce_scatter,
    torus_a2a,
    torus_all_gather,
    torus_all_reduce,
    tree_all_reduce,
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
        `NotImplementedError` — reserved for a future Swing-AR primitive.
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

    classes = {_routing_class(t) for t in crossed}

    # Pure crossbar (CrossbarTier or full mesh): delegate to _span_tiers_scaled
    # so the α-sum / BW-min accumulation order is byte-identical to the
    # pre-refactor path. Full mesh routes through the same primitives because
    # cost formulas are identical (single hop, full bisection).
    if classes == {"crossbar"}:
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
        # Multi-tier crossbar AR / AG: route through the hierarchical RS →
        # sub-AR → AG composition (collectives.md §3.3 / §4.3). Inner = first
        # crossed tier; outer = remaining crossed tier(s) flattened. Single-
        # tier still uses the flat path. The `algorithm` knob maps to the
        # outer phase's choice (inner is always ring — only shipped).
        if len(crossed) > 1 and op in ("all_reduce", "all_gather"):
            return _hierarchical_crossbar_cost(op, M, G, crossed, algorithm)
        # Multi-tier crossbar MoE A2A: per-destination-class accounting
        # (collectives.md §5.3). A2A doesn't telescope — cost itemizes by
        # destination class (intra-pod / outer). The `algorithm` knob is
        # ignored (only pairwise direct-send is shipped; tree is deprecated).
        if len(crossed) > 1 and op == "moe_a2a":
            return _hierarchical_a2a_cost(M, G, crossed)
        alpha_us, bw_GBps, _ = _span_tiers_scaled(tiers, G)
        return _crossbar_cost(
            op, M, G, alpha_us * US_TO_SECONDS, bw_GBps * GB_TO_BYTES, algorithm
        )

    # Pure torus (TorusTier or k-D mesh): dispatch to torus primitives.
    # k-D mesh tiers use the same dim-decomposed primitives but pass
    # `wraparound=False` to torus_moe_all_to_all (halved bisection → /4).
    if classes == {"torus"}:
        if torus_algorithm == "swing":
            raise NotImplementedError(
                "torus_algorithm='swing' not yet implemented."
            )
        if torus_algorithm != "ring":
            raise ValueError(
                f"Unsupported torus_algorithm: {torus_algorithm!r}"
            )
        return _torus_cost(op, M, G, crossed)

    # Genuinely mixed crossbar/torus (or mesh-mixed): fall back to crossbar-
    # flatten so the interface stays live and numerically bounded; emit a
    # warning so callers notice. Explicit hybrid compositions are a follow-up.
    topologies = {t.topology for t in crossed}
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


def enumerate_options(
    tiers: List[TierSpec],
    op: str,
    M: float,
    G: int,
    inc_enabled: bool = True,
) -> List[Tuple[str, float]]:
    """Enumerate admissible (algorithm_name, cost_seconds) pairs for a collective.

    The post-partition optimizer (`core/collective_algo_opt.py`) consumes this
    surface to pick the optimal SW algorithm per (phase × collective). For each
    (op, tier-chain, M, G, inc_enabled) tuple, returns every algorithm the
    dispatcher supports for that cell, paired with its α-β cost in seconds.

    Algorithm name space:
      - Star (single-tier or multi-tier crossbar):
          AR:        "ring", "tree", "tree_pipelined", optionally "inc"
          AG:        "ring", optionally "inc"
          MoE A2A:   "ring" (= pairwise direct-send), optionally "inc"
          P2P:       "p2p" (single hop, no algorithm choice)
      - Torus:
          all ops:   "torus-dim-ring"  (single option; INC structurally absent
                                        on torus per `system_spec.TorusTier`)
      - Mixed crossbar/torus: same as the dispatcher's flat-fallback path
                              (warns at cost_collective time, not here).

    "tree" maps to the literal P=1 binomial tree (BW coefficient = log_G);
    "tree_pipelined" maps to the asymptotic P→P* form (BW coefficient = 1)
    that NCCL ships at bulk M.

    INC eligibility (collectives.md §3.4 / §4.4 / §5.4):
      - AR / AG via "inc" requires every crossed tier to declare `inc != "none"`.
      - MoE A2A via "inc" requires every crossed tier to declare `inc == "hw_a2a"`
        (sharp_class does not accelerate A2A).
      - When `inc_enabled=False`, no "inc" entries are returned.

    Returns:
      Ordered list of (name, cost_seconds) pairs. Empty for G <= 1 or empty
      tier list (no work to enumerate). The optimizer picks `min(.., key=cost)`.
    """
    if op not in _OPS:
        raise ValueError(f"Unknown op: {op!r}; allowed: {_OPS}")
    if G <= 1 or not tiers:
        return []

    # Walk tiers innermost-first, collecting those the collective crosses.
    # Same semantics as cost_collective.
    crossed: List[TierSpec] = []
    reach = 1
    for tier in tiers:
        reach *= max(1, tier.ports)
        crossed.append(tier)
        if reach >= G:
            break

    topologies = {t.topology for t in crossed}
    classes = {_routing_class(t) for t in crossed}

    # Pure torus / k-D mesh: dim-decomposed-ring is the only shipped algorithm;
    # INC is structurally absent (TorusTier and MeshTier have no inc field).
    # p2p is a single hop on torus too.
    if classes == {"torus"}:
        cost = _torus_cost(op, M, G, crossed)
        name = "p2p" if op == "p2p" else "torus-dim-ring"
        return [(name, cost)]

    # Pure crossbar (or mixed — falls into crossbar-flatten path with a warn at
    # cost_collective time; here we silently mirror the dispatcher's behavior).
    options: List[Tuple[str, float]] = []
    alpha_us, bw_GBps, _ = _span_tiers_scaled(tiers, G)
    alpha_s = alpha_us * US_TO_SECONDS
    bw_Bps = bw_GBps * GB_TO_BYTES

    if op == "p2p":
        options.append(("p2p", _crossbar_cost(op, M, G, alpha_s, bw_Bps, "ring")))
        return options

    # AR / AG: multi-tier crossbar routes through hierarchical (collectives.md
    # §3.3 / §4.3); single-tier crossbar (and the mixed-chain fallback) use
    # the flat path. The `ring`/`tree` names map to the outer-phase choice
    # on multi-tier; inner is always ring.
    use_hierarchical = (
        len(crossed) > 1
        and classes == {"crossbar"}
        and op in ("all_reduce", "all_gather")
    )

    if op == "all_gather":
        if use_hierarchical:
            options.append(
                ("ring", _hierarchical_crossbar_cost(op, M, G, crossed, "ring"))
            )
        else:
            options.append(("ring", _crossbar_cost(op, M, G, alpha_s, bw_Bps, "ring")))
    elif op == "all_reduce":
        if use_hierarchical:
            options.append(
                ("ring", _hierarchical_crossbar_cost(op, M, G, crossed, "ring"))
            )
            options.append(
                ("tree", _hierarchical_crossbar_cost(op, M, G, crossed, "tree"))
            )
            options.append(
                ("tree_pipelined", _hierarchical_crossbar_cost(op, M, G, crossed, "tree_pipelined"))
            )
        else:
            options.append(("ring", _crossbar_cost(op, M, G, alpha_s, bw_Bps, "ring")))
            options.append(("tree", _crossbar_cost(op, M, G, alpha_s, bw_Bps, "tree")))
            options.append(("tree_pipelined", _crossbar_cost(op, M, G, alpha_s, bw_Bps, "tree_pipelined")))
    elif op == "moe_a2a":
        # MoE A2A: multi-tier crossbar uses per-destination-class accounting
        # (collectives.md §5.3); single-tier uses flat pairwise.
        if len(crossed) > 1 and classes == {"crossbar"}:
            options.append(("ring", _hierarchical_a2a_cost(M, G, crossed)))
        else:
            options.append(("ring", _crossbar_cost(op, M, G, alpha_s, bw_Bps, "ring")))

    # INC variants — only on pure crossbar (CrossbarTier; full mesh has no
    # `inc` field so it's automatically excluded by `_is_inc`).
    if (
        inc_enabled
        and classes == {"crossbar"}
        and all(_is_inc(t) for t in crossed)
    ):
        if op in ("all_reduce", "all_gather"):
            options.append(("inc", _inc_crossbar_cost(op, M, G, crossed)))
        elif op == "moe_a2a" and all(_is_hw_a2a(t) for t in crossed):
            options.append(("inc", _inc_crossbar_cost(op, M, G, crossed)))

    return options


# ────────────────────────────────────────────────────────────
# Crossbar path (bit-identical to pre-refactor span_tiers flatten)
# ────────────────────────────────────────────────────────────

def _is_inc(tier: TierSpec) -> bool:
    """Tier supports any INC capability (sharp_class or hw_a2a)."""
    return getattr(tier, "inc", "none") != "none"


def _is_hw_a2a(tier: TierSpec) -> bool:
    """Tier supports HW A2A specifically (Tomahawk Ultra / Rubin)."""
    return getattr(tier, "inc", "none") == "hw_a2a"


def _routing_class(tier: TierSpec) -> str:
    """Return the dispatcher's cost-routing class for a tier.

    "crossbar"    — single-tier-equivalent: CrossbarTier OR full mesh
                     (cost = single hop, full bisection).
    "torus"       — dim-decomposed: TorusTier OR k-D mesh
                     (cost = open-line bucket brigade; A2A bisection-bound).
    """
    topo = getattr(tier, "topology", "crossbar")
    if topo == "crossbar":
        return "crossbar"
    if topo == "torus":
        return "torus"
    if topo == "mesh":
        # Full mesh routes through crossbar primitives; k-D mesh through torus.
        return "crossbar" if getattr(tier, "full", False) else "torus"
    raise ValueError(f"unknown tier topology: {topo!r}")


def _all_wraparound(crossed: List[TierSpec]) -> bool:
    """True iff every torus-routed tier has wraparound edges.

    TorusTier always has wraparound. MeshTier with `full=False` is a k-D
    mesh and does NOT have wraparound. Determines the bisection coefficient
    on `torus_moe_all_to_all` (D_max/8 vs D_max/4).
    """
    for t in crossed:
        if getattr(t, "topology", None) == "mesh" and not getattr(t, "full", False):
            return False
    return True


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
        if algorithm == "tree_pipelined":
            return tree_all_reduce(M, G, alpha_s, bw_Bps, pipelined=True)
        raise ValueError(f"Unsupported algorithm for all_reduce: {algorithm!r}")
    if op == "moe_a2a":
        # MoE Dispatch+Combine round-trip: caller's M is the one-way per-rank
        # payload, so we double the underlying single-direction A2A cost. The
        # 2× factor lives here (not inside pairwise_a2a) because it is a
        # property of the dispatcher's MoE contract, not of the primitive.
        if algorithm == "ring":
            return 2 * pairwise_a2a(M, G, alpha_s, bw_Bps)
        raise ValueError(f"Unsupported algorithm for moe_a2a: {algorithm!r}")
    raise AssertionError(f"unreachable op={op!r}")  # caller validated


# ────────────────────────────────────────────────────────────
# Hierarchical crossbar path — RS → sub-AR → AG composition for AR;
# inner-then-outer cascade for AG / RS. Per collectives.md §3.3 / §4.3.
# ────────────────────────────────────────────────────────────

def _hierarchical_crossbar_cost(
    op: str, M: float, G: int, crossed: List[TierSpec], outer_alg: str
) -> float:
    """Hierarchical crossbar AR / AG / RS for multi-tier chains.

    Inner = first crossed tier; outer = remaining crossed tiers flattened to
    a single $(\\alpha_\\mathrm{sum}, \\mathrm{BW}_\\min)$ pair (recursive
    decomposition for k > 2 crossed tiers is a follow-up). Inner uses ring
    (the only shipped variant for inner RS / AG); outer uses `outer_alg ∈
    {ring, tree}` for AR.

    AR (collectives.md §3.3): inner RS + outer sub-AR(M·L/N) + inner AG.
    AG (collectives.md §4.3): inner AG + outer AG(G_inner·M).
    RS: outer RS(M/L) + inner RS(M/N) — time-reverse of AG.

    All three preserve the flat-ring β coefficient (telescoping); the α-side
    compresses from $2(N-1)\\alpha$ to $2((G_\\mathrm{inner}{-}1)\\alpha_\\mathrm{inner}
    + (L{-}1)\\alpha_\\mathrm{outer})$ for AR.
    """
    # Inner / outer split. Inner is the first crossed tier; outer is the rest.
    inner_tier = crossed[0]
    outer_tiers = crossed[1:]
    if not outer_tiers:
        # Single tier — caller should have routed to flat path. Defensive.
        raise AssertionError(
            "_hierarchical_crossbar_cost called with single-tier chain"
        )

    G_inner = max(1, inner_tier.ports)
    L = max(1, G // G_inner)  # outer-tier rank count

    alpha_inner_s = (
        inner_tier.alpha_us * inner_tier.eta_alpha * US_TO_SECONDS
    )
    bw_inner_Bps = (
        inner_tier.bw_per_port_GBps * inner_tier.eta_beta * GB_TO_BYTES
    )

    # Outer (alpha, BW): flatten across remaining crossed tiers. For 2-tier
    # chains this is a single tier; for 3+ tiers it's a flat approximation
    # (recursive hierarchical is a follow-up).
    alpha_outer_us = sum(t.alpha_us * t.eta_alpha for t in outer_tiers)
    bw_outer_GBps = min(
        t.bw_per_port_GBps * t.eta_beta for t in outer_tiers
    )
    alpha_outer_s = alpha_outer_us * US_TO_SECONDS
    bw_outer_Bps = bw_outer_GBps * GB_TO_BYTES

    if op == "all_reduce":
        # Inner RS + outer sub-AR + inner AG.
        # M = per-rank full vector (AR convention, unchanged upstream).
        # New AG/RS convention: M = per-rank pre-scatter input / gathered
        # output, so inner RS and inner AG both pass the full M.
        t_inner_rs = ring_reduce_scatter(
            M, G_inner, alpha_inner_s, bw_inner_Bps
        )
        # Outer sub-AR: payload per rank after inner RS = M / G_inner.
        M_outer = M / G_inner
        if outer_alg == "ring":
            t_outer = ring_all_reduce(
                M_outer, L, alpha_outer_s, bw_outer_Bps
            )
        elif outer_alg == "tree":
            t_outer = tree_all_reduce(
                M_outer, L, alpha_outer_s, bw_outer_Bps
            )
        elif outer_alg == "tree_pipelined":
            t_outer = tree_all_reduce(
                M_outer, L, alpha_outer_s, bw_outer_Bps, pipelined=True
            )
        else:
            raise ValueError(
                f"Unsupported outer_alg for hierarchical AR: {outer_alg!r}"
            )
        t_inner_ag = ring_all_gather(
            M, G_inner, alpha_inner_s, bw_inner_Bps
        )
        return t_inner_rs + t_outer + t_inner_ag

    if op == "all_gather":
        # Inner AG → outer AG cascade.
        # New AG convention: M = per-rank gathered output. After inner AG
        # each rank holds G_inner shards = M / L bytes; after outer AG
        # each rank holds the full M.
        t_inner = ring_all_gather(
            M / L, G_inner, alpha_inner_s, bw_inner_Bps
        )
        t_outer = ring_all_gather(
            M, L, alpha_outer_s, bw_outer_Bps
        )
        return t_inner + t_outer

    if op == "reduce_scatter":
        # Outer RS → inner RS cascade (time-reverse of AG).
        # New RS convention: M = per-rank pre-scatter input. Outer RS
        # consumes M and produces M/L; inner RS consumes M/L and produces
        # M/(L·G_inner) = M/N.
        t_outer = ring_reduce_scatter(
            M, L, alpha_outer_s, bw_outer_Bps
        )
        t_inner = ring_reduce_scatter(
            M / L, G_inner, alpha_inner_s, bw_inner_Bps
        )
        return t_outer + t_inner

    raise AssertionError(
        f"_hierarchical_crossbar_cost: unsupported op={op!r}; AR/AG/RS only"
    )


def _hierarchical_a2a_cost(
    M: float, G: int, crossed: List[TierSpec]
) -> float:
    """Hierarchical A2A per-destination-class accounting (collectives.md §5.3).

    A2A doesn't decompose hierarchically — every source-destination pair carries
    a distinct payload, so cross-tier permutation traffic equals the full
    $(G-1)/G \\cdot M$ bytes per rank regardless of schedule. The implemented
    pattern is pairwise direct-send with each destination paying its own path
    cost; the total is a destination-weighted sum:

        t = 2 · [ (G_inner − 1) · (α_inner + (M/G)/BW_inner)
              +   (G − G_inner) · (α_outer + (M/G)/BW_outer) ]

    Pod boundary = first crossed tier's reach (G_inner). Same-leaf and cross-
    leaf cross-pod destinations are collapsed into a single "outer" class —
    explicit pod_size override is a future enhancement when 3-tier hierarchies
    (intra-pod / same-leaf / cross-leaf) need to be modeled. The factor of 2
    bakes in the Dispatch + Combine round-trip (matches `pairwise_a2a` and
    `inc_a2a` conventions).
    """
    inner_tier = crossed[0]
    outer_tiers = crossed[1:]
    if not outer_tiers:
        raise AssertionError(
            "_hierarchical_a2a_cost called with single-tier chain"
        )

    G_inner = max(1, inner_tier.ports)

    alpha_inner_s = (
        inner_tier.alpha_us * inner_tier.eta_alpha * US_TO_SECONDS
    )
    bw_inner_Bps = (
        inner_tier.bw_per_port_GBps * inner_tier.eta_beta * GB_TO_BYTES
    )
    alpha_outer_us = sum(t.alpha_us * t.eta_alpha for t in outer_tiers)
    bw_outer_GBps = min(
        t.bw_per_port_GBps * t.eta_beta for t in outer_tiers
    )
    alpha_outer_s = alpha_outer_us * US_TO_SECONDS
    bw_outer_Bps = bw_outer_GBps * GB_TO_BYTES

    chunk = M / G if G > 0 else 0.0  # one-way per-destination payload
    n_intra = max(0, G_inner - 1)    # destinations inside the pod
    n_outer = max(0, G - G_inner)    # destinations outside the pod

    if bw_inner_Bps <= 0 or (n_outer > 0 and bw_outer_Bps <= 0):
        return 0.0

    t_intra = n_intra * (alpha_inner_s + chunk / bw_inner_Bps)
    t_outer = (
        n_outer * (alpha_outer_s + chunk / bw_outer_Bps)
        if n_outer > 0
        else 0.0
    )
    return 2 * (t_intra + t_outer)  # Dispatch + Combine round-trip


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
        # MoE Dispatch+Combine: 2× wrap, same as the SW crossbar / torus
        # paths. Keeps the dispatcher's MoE contract topology-uniform.
        return 2 * inc_a2a(M, G, alpha_s, bw_Bps)
    raise AssertionError(f"INC path reached with unsupported op={op!r}")


# ────────────────────────────────────────────────────────────
# Torus path (Phase B primitives)
# ────────────────────────────────────────────────────────────

def _torus_cost(op: str, M: float, G: int, crossed: List[TierSpec]) -> float:
    """Cost a collective on a torus-routed tier chain.

    Accepts both `TorusTier` and k-D-mesh `MeshTier` (full=False); they share
    the same dim-decomposed primitives. Multi-tier chains concatenate their
    `dims` tuples into the effective k-D structure. Dim-alignment is checked
    against prefix-products of the concatenated dims; misalignment falls back
    to a flat-ring conservative bound with a `UserWarning`. If any tier is a
    k-D mesh (no wraparound), `torus_a2a` is called with
    `wraparound=False` (halved bisection → $D_\\mathrm{max}/4$ on A2A).
    """
    full_dims: Tuple[int, ...] = tuple(
        d for t in crossed for d in t.dims  # type: ignore[attr-defined]
    )
    wraparound = _all_wraparound(crossed)
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
            f"See collectives.md §3.2 for dim-aligned layouts.",
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
        # MoE Dispatch+Combine: same 2× wrap as the crossbar path so the
        # dispatcher's MoE contract is topology-uniform (caller passes the
        # one-way per-rank payload regardless of fabric type).
        return 2 * torus_a2a(M, subdims, alpha_s, bw_Bps,
                             wraparound=wraparound)
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
