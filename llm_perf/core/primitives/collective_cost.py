# ─────────────────────────────────────────────────────────────────────
# AUTO-SYNCED FROM UPSTREAM — DO NOT EDIT LOCALLY
# Source of truth: https://github.com/spiceMonkey/collective-comm
# Path:            code/core/collective_cost.py
# Sync workflow:   .github/workflows/sync-collectives.yml
#
# Edits made directly to this file will be overwritten on the next sync.
# To change behaviour, open a PR against the upstream repo, then re-run
# the sync (or wait for the Monday cron). Local-only adapters belong in
# sibling modules (see e.g. stage_aggregator.py).
#
# Doc-path note: docstrings here reference upstream
# `documentation/modeling/`; that subtree lands locally at
# `documentation/modeling/collectives/`.
# ─────────────────────────────────────────────────────────────────────

"""Collective algorithm cost primitives — α-β cost model.

Source of truth for every formula here is the modeling series under
``documentation/modeling/``:

- ``01_collective_algorithms.md`` — α-β model and per-primitive derivations.
- ``02_topology_mapping.md`` — single-tier specializations (star, torus, mesh).
- ``03_hierarchical_topologies.md`` — multi-tier Clos / fat-tree composition.
- ``04_in_network_collectives.md`` — switch-ALU / multicast / scatter-gather INC.
- ``05_contention_and_congestion.md`` — η contention coefficients.

Conventions
-----------
``M`` is always the per-rank payload measured in bytes. The exact meaning per
primitive matches the doc convention:

- BC / Reduce: full broadcast / reduce payload.
- AR: full vector each rank starts and ends with.
- AG: total gathered output per rank (``N · shard``).
- RS: total pre-scatter input per rank (``N · shard``).
- A2A: total per-rank A2A payload (``N`` chunks of size ``M/N`` each).
- P2P: single-message payload.

``G`` is the group rank count, ``alpha_s`` is per-hop latency in seconds, and
``bw_Bps`` is per-link bandwidth in bytes per second. Each function returns
seconds. Functions return ``0.0`` on no-op inputs (``G ≤ 1`` or non-positive
bandwidth) so callers can skip shape-level guards.

Pipelined variants
------------------
For schedules that benefit from chunking (BC, Reduce, DBT AR), the
``pipelined`` flag selects between:

- ``pipelined=False`` (default): the literal P=1 schedule with full ``M`` on
  every active link per step. Matches the closed-form derivations in
  ``01_collective_algorithms.md`` §3-§5.
- ``pipelined=True``: the asymptotic P→P* limit where the BW coefficient
  collapses to 1 (Appendix C). Hides the ``O(√M)`` correction.

Ring AR / AG / RS / A2A are *intrinsically* pipelined — their closed forms
already are the asymptote, so no flag is exposed.
"""

import math
import warnings
from typing import Sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prod(xs: Sequence[int]) -> int:
    n = 1
    for x in xs:
        n *= x
    return n


def apply_eta(
    alpha_s: float,
    bw_Bps: float,
    eta_alpha: float = 1.0,
    eta_beta: float = 1.0,
) -> tuple[float, float]:
    """Pre-discount (α, BW) by contention coefficients — ``05_contention_and_congestion.md`` §4.

        α_eff  = α · η_α       (η_α ≥ 1 inflates latency)
        BW_eff = BW · η_β      (η_β ∈ (0, 1] deflates bandwidth)

    The realistic primitive cost is then evaluated by passing
    ``(alpha_eff, bw_eff)`` into any of the cost functions below — no
    algorithm-level plumbing needed (per the per-tier η framing in §4.2).

    Calibration profile (``05_contention_and_congestion.md`` §4.1):

    - Crossbar (NVLink + NVSwitch, no SHARP):    η_α = 1.00, η_β = 0.80
    - NVLS (NVLink SHARP, in-network reduction): η_α = 1.00, η_β = 0.52
    - Torus (off-prefix + concurrent groups):    η_α = 1.20, η_β = 0.60
    - Fat-tree / Clos at oversubscription s:     η_β = min(η_β_hw, 1/s)
    """
    return alpha_s * eta_alpha, bw_Bps * eta_beta


# ---------------------------------------------------------------------------
# Point-to-point
# ---------------------------------------------------------------------------


def p2p_hop(M: float, alpha_s: float, bw_Bps: float) -> float:
    """Single send / recv — ``01_collective_algorithms.md`` §8.

        t = α + M / BW

    The degenerate N=2 case of any collective. Used per inter-stage hop in
    pipeline parallelism.
    """
    if bw_Bps <= 0:
        return 0.0
    return alpha_s + M / bw_Bps


# ---------------------------------------------------------------------------
# Broadcast (BC)
# ---------------------------------------------------------------------------


def ring_broadcast(
    M: float, G: int, alpha_s: float, bw_Bps: float, pipelined: bool = False
) -> float:
    """Ring (chain) broadcast — ``01_collective_algorithms.md`` §3.1.

        non-pipelined (P=1):       t = (G-1)·α + (G-1) · M / BW
        pipelined (P→P*):          t ≈ (G-1)·α + M / BW

    The pipelined form hides an ``O(√M)`` correction
    ``2·sqrt((G-2)·α·M/BW)`` (Appendix C); it vanishes relative to ``M/BW``
    as ``M → ∞``.
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    bw_coeff = 1.0 if pipelined else (G - 1)
    return (G - 1) * alpha_s + bw_coeff * (M / bw_Bps)


def tree_broadcast(
    M: float, G: int, alpha_s: float, bw_Bps: float, pipelined: bool = False
) -> float:
    """Binomial tree broadcast — ``01_collective_algorithms.md`` §3.2.

        non-pipelined (P=1):    t = ⌈log₂ G⌉·α + ⌈log₂ G⌉ · M / BW
        pipelined (P→P*):       t ≈ ⌈log₂ G⌉·α + M / BW

    NCCL's default BC path runs the pipelined tree at bulk M.
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    log_G = math.ceil(math.log2(G))
    bw_coeff = 1.0 if pipelined else log_G
    return log_G * alpha_s + bw_coeff * (M / bw_Bps)


def inc_broadcast(M: float, alpha_s: float, bw_Bps: float) -> float:
    """In-network multicast broadcast — ``01_collective_algorithms.md`` §3.3,
    ``04_in_network_collectives.md`` §1.2.

        t = 2·α_switch + M / BW

    Source rank pushes M upstream to the switch (1 hop); switch crossbar
    multicasts to all destination ports concurrently (1 hop). Two hops
    independent of N. ``alpha_s`` is the summed per-tier switch cut-through α
    on one direction (single-switch star: ``α_switch``; k-tier scale-out
    aggregation tree: ``Σ α_tier``).
    """
    if bw_Bps <= 0:
        return 0.0
    return 2 * alpha_s + M / bw_Bps


# ---------------------------------------------------------------------------
# Reduce
# ---------------------------------------------------------------------------


def ring_reduce(
    M: float, G: int, alpha_s: float, bw_Bps: float, pipelined: bool = False
) -> float:
    """Ring (chain) reduce — ``01_collective_algorithms.md`` §4.1.

        non-pipelined (P=1):    t = (G-1)·α + (G-1) · M / BW
        pipelined (P→P*):       t ≈ (G-1)·α + M / BW

    Time-reverse of ring BC with on-chip add at every hop.
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    bw_coeff = 1.0 if pipelined else (G - 1)
    return (G - 1) * alpha_s + bw_coeff * (M / bw_Bps)


def tree_reduce(
    M: float, G: int, alpha_s: float, bw_Bps: float, pipelined: bool = False
) -> float:
    """Binomial tree reduce — ``01_collective_algorithms.md`` §4.2.

        non-pipelined (P=1):    t = ⌈log₂ G⌉·α + ⌈log₂ G⌉ · M / BW
        pipelined (P→P*):       t ≈ ⌈log₂ G⌉·α + M / BW

    Tree strictly dominates ring for standalone Reduce (same BW floor at
    log-depth α); ring Reduce is rarely selected by tuners.
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    log_G = math.ceil(math.log2(G))
    bw_coeff = 1.0 if pipelined else log_G
    return log_G * alpha_s + bw_coeff * (M / bw_Bps)


def inc_reduce(M: float, alpha_s: float, bw_Bps: float) -> float:
    """In-network reduce (switch ALU) — ``01_collective_algorithms.md`` §4.3,
    ``04_in_network_collectives.md`` §1.1.

        t = 2·α_switch + M / BW

    Each rank pushes V_i upstream (1 hop); switch ALU sums on-chip and
    forwards the single reduced result to the root (1 hop).
    """
    if bw_Bps <= 0:
        return 0.0
    return 2 * alpha_s + M / bw_Bps


# ---------------------------------------------------------------------------
# All-reduce (AR)
# ---------------------------------------------------------------------------


def ring_all_reduce(M: float, G: int, alpha_s: float, bw_Bps: float) -> float:
    """Ring (Patarasuk-Yuan) all-reduce — ``01_collective_algorithms.md`` §5.1.

        t = 2(G-1)·α + 2·(G-1)/G · M / BW

    Already the pipelined asymptote (P=N) by construction — the M/N chunked
    payload bakes pipelining into the derivation, so no ``pipelined`` flag is
    needed. BW coefficient hits the AR floor of 2 as G → ∞.
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    return 2 * (G - 1) * alpha_s + 2 * ((G - 1) / G) * (M / bw_Bps)


def tree_all_reduce(
    M: float, G: int, alpha_s: float, bw_Bps: float, pipelined: bool = False
) -> float:
    """Double binary tree (DBT) all-reduce — ``01_collective_algorithms.md`` §5.2.

        non-pipelined (P=1):    t = 2⌈log₂ G⌉·α + ⌈log₂ G⌉ · M / BW
        pipelined (P→P*):       t ≈ 2⌈log₂ G⌉·α + M / BW

    NCCL's non-ring multi-node default. The pipelined floor matches INC AR's
    BW floor at the algorithmic level, but is a *lower bound* in practice —
    real-world DBT carries ``c_real ≥ 1`` above ``M/BW`` due to finite
    pipeline depth and per-step overhead, which is what lets ring re-take the
    crown at large M (§5.3 practice caveat).

    Note: ``04_in_network_collectives.md`` §3.1's N=512 anchor uses an
    ``n_β = 2`` "dual-touch" form (BW term = 2·M/BW) rather than either form
    here — it reflects the realized cost where every byte crosses each
    endpoint link twice. To reproduce the 45 μs anchor, use that form
    directly: ``2·log_G·α + 2·M/BW``.
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    log_G = math.ceil(math.log2(G))
    bw_coeff = 1.0 if pipelined else log_G
    return 2 * log_G * alpha_s + bw_coeff * (M / bw_Bps)


def rabenseifner_all_reduce(
    M: float, G: int, alpha_s: float, bw_Bps: float
) -> float:
    """Rabenseifner halving-doubling AR — ``01_collective_algorithms.md`` App. B.2.

        t = 2⌈log₂ G⌉·α + 2·(G-1)/G · M / BW

    Two complementary hypercube passes (recursive halving for RS, recursive
    doubling for AG) with chunk-exponential payloads. Matches ring's BW floor
    with log-depth α — but **not shipped by NCCL** for AR; reference only.
    Power-of-2 G is required for clean partner pairings.
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    log_G = math.ceil(math.log2(G))
    return 2 * log_G * alpha_s + 2 * ((G - 1) / G) * (M / bw_Bps)


def inc_all_reduce(M: float, alpha_s: float, bw_Bps: float) -> float:
    """In-network all-reduce (switch ALU + multicast) —
    ``04_in_network_collectives.md`` §1.4 / §2.

        t = 2·α_switch + M / BW

    The unique INC primitive that lifts BW_eff from BW/2 (software dual-touch)
    to BW: switch ALU reduces the N inputs in-fabric and the multicast xbar
    replicates the single result to all N output ports, on opposite directions
    of each full-duplex endpoint link. ``alpha_s`` is the summed per-tier
    switch α on one direction (single switch: ``α_switch``; k-tier
    aggregation tree: ``Σ α_tier``); the leading factor of 2 covers the
    reduce-up + multicast-down round trip.
    """
    if bw_Bps <= 0:
        return 0.0
    return 2 * alpha_s + M / bw_Bps


# ---------------------------------------------------------------------------
# All-gather / Reduce-scatter (AG / RS)
# ---------------------------------------------------------------------------


def ring_all_gather(M: float, G: int, alpha_s: float, bw_Bps: float) -> float:
    """Ring all-gather — ``01_collective_algorithms.md`` §6.

        t = (G-1)·α + (G-1)/G · M / BW

    M is the per-rank gathered output (= G · per-rank shard). Already
    pipelined (P=N) like ring AR. AG is exactly Phase 2 of ring AR with no
    reduction; RS is Phase 1 (same formula).
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    return (G - 1) * alpha_s + ((G - 1) / G) * (M / bw_Bps)


def ring_reduce_scatter(M: float, G: int, alpha_s: float, bw_Bps: float) -> float:
    """Ring reduce-scatter — ``01_collective_algorithms.md`` §6.

        t = (G-1)·α + (G-1)/G · M / BW

    M is the per-rank pre-scatter input (= G · per-rank output shard). RS is
    the time-reverse of AG; same wiring, accumulating partial sum on every
    forward instead of overwriting.
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    return (G - 1) * alpha_s + ((G - 1) / G) * (M / bw_Bps)


def recursive_doubling_all_gather(
    M: float, G: int, alpha_s: float, bw_Bps: float
) -> float:
    """Recursive-doubling AG — ``01_collective_algorithms.md`` App. B.4.

        t = ⌈log₂ G⌉·α + (G-1)/G · M / BW

    Hypercube butterfly with doubling chunk size. MPI menus ship this
    alongside ring; NCCL does not. Power-of-2 G required for clean pairings.
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    log_G = math.ceil(math.log2(G))
    return log_G * alpha_s + ((G - 1) / G) * (M / bw_Bps)


def recursive_halving_reduce_scatter(
    M: float, G: int, alpha_s: float, bw_Bps: float
) -> float:
    """Recursive-halving RS — ``01_collective_algorithms.md`` App. B.4.

        t = ⌈log₂ G⌉·α + (G-1)/G · M / BW

    Time-reverse of recursive-doubling AG: hypercube butterfly with halving
    chunk size, accumulating on receive. Power-of-2 G required.
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    log_G = math.ceil(math.log2(G))
    return log_G * alpha_s + ((G - 1) / G) * (M / bw_Bps)


def pat_all_gather(M: float, G: int, alpha_s: float, bw_Bps: float) -> float:
    """Parallel Aggregated Trees AG — ``01_collective_algorithms.md`` App. A,
    referenced as the NCCL 2.23+ scale-out path (1 rank per node).

        t = ⌈log₂ G⌉·α + (G-1)/G · M / BW

    Reversed-Bruck schedule with bounded staging buffer; ships only inter-node
    1-rank-per-node. Same α-β cost as recursive-doubling AG but pipelineable
    against the inter-node fabric's staging constraints.
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    log_G = math.ceil(math.log2(G))
    return log_G * alpha_s + ((G - 1) / G) * (M / bw_Bps)


def pat_reduce_scatter(M: float, G: int, alpha_s: float, bw_Bps: float) -> float:
    """PAT reduce-scatter — ``01_collective_algorithms.md`` App. A.

        t = ⌈log₂ G⌉·α + (G-1)/G · M / BW

    Time-reverse of PAT AG; same scale-out 1-rank-per-node restriction.
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    log_G = math.ceil(math.log2(G))
    return log_G * alpha_s + ((G - 1) / G) * (M / bw_Bps)


def inc_all_gather(M: float, G: int, alpha_s: float, bw_Bps: float) -> float:
    """In-network all-gather — ``04_in_network_collectives.md`` §1.4.

        t = 2·α_switch + (G-1)/G · M / BW

    α-only collapse: software ring AG already runs at BW_eff = BW on a
    full-duplex star, so the switch multicast lifts only the α side from
    ``(G-1)·α`` to ``2·α_switch``. ``alpha_s`` is the summed per-tier switch α
    (one direction).
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    return 2 * alpha_s + ((G - 1) / G) * (M / bw_Bps)


def inc_reduce_scatter(M: float, G: int, alpha_s: float, bw_Bps: float) -> float:
    """In-network reduce-scatter — ``04_in_network_collectives.md`` §1.4.

        t = 2·α_switch + (G-1)/G · M / BW

    Mirror of inc_all_gather: α-only collapse, BW unchanged from software
    ring RS on a full-duplex crossbar.
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    return 2 * alpha_s + ((G - 1) / G) * (M / bw_Bps)


# ---------------------------------------------------------------------------
# All-to-all (A2A)
# ---------------------------------------------------------------------------


def pairwise_a2a(M: float, G: int, alpha_s: float, bw_Bps: float) -> float:
    """Pairwise direct-send A2A — ``01_collective_algorithms.md`` §7.2.

        t = (G-1)·α + (G-1)/G · M / BW

    NCCL default on full-bisection fabrics (fat-tree, Clos, NVSwitch). M is
    the per-rank total A2A payload (G chunks of size M/G each). Workloads
    that run A2A back-to-back (MoE Dispatch + Combine) double both terms
    externally — see ``01_collective_algorithms.md`` §7.2 final paragraph.
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    return (G - 1) * alpha_s + ((G - 1) / G) * (M / bw_Bps)


def ring_relay_a2a(M: float, G: int, alpha_s: float, bw_Bps: float) -> float:
    """Ring relay A2A — ``01_collective_algorithms.md`` §7.1.

        t = (G-1)·α + (G-1)/G · M / BW

    α-β-equivalent to pairwise direct-send. Used on bisection-limited fabrics
    where shortest-arc forwarding through intermediate ranks is the only
    reachable schedule (pure ring or open-line, no full bisection).
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    return (G - 1) * alpha_s + ((G - 1) / G) * (M / bw_Bps)


def bruck_a2a(M: float, G: int, alpha_s: float, bw_Bps: float) -> float:
    """Bruck log-round A2A — ``01_collective_algorithms.md`` App. B.5.

        t = ⌈log₂ G⌉·α + ⌈log₂ G⌉/2 · M / BW

    Log-depth latency at the cost of an ``⌈log₂ G⌉/2`` BW coefficient — every
    byte crosses ``log₂ G / 2`` link-hops on average (chunks bounce around
    the hypercube before reaching their final destination). Reference only;
    NCCL / RCCL do not ship Bruck.
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    log_G = math.ceil(math.log2(G))
    return log_G * alpha_s + (log_G / 2.0) * (M / bw_Bps)


def inc_a2a(M: float, G: int, alpha_s: float, bw_Bps: float) -> float:
    """Hardware A2A (crossbar scatter-gather) — ``04_in_network_collectives.md`` §1.3.

        t = α_switch + (G-1)/G · M / BW

    Tomahawk Ultra (Ethernet, shipping 2025) and planned Rubin-gen NVSwitches
    collapse software A2A's (G-1) endpoint-driven scheduling rounds into one
    switch-driven transaction. Per-rank wire-side bytes are unchanged from
    software pairwise (the switch routes verbatim — no aggregation, no
    replication), so the BW term stays at the bisection bound.

    ``alpha_s`` is ``α_switch`` (single transaction); not ``2·α_switch`` like
    AR/AG/RS, because A2A is a single descriptor + single routing pass with
    no reduce-up + multicast-down round trip.

    Not on shipping NVSwitch Gen4 (NVL72) or Quantum-X800; software pairwise
    is the only A2A path on those fabrics.
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    return alpha_s + ((G - 1) / G) * (M / bw_Bps)


# ---------------------------------------------------------------------------
# Torus / k-D mesh primitives (single-tier, dim-decomposed)
# ---------------------------------------------------------------------------


def torus_all_reduce(
    M: float, dims: Sequence[int], alpha_s: float, bw_Bps: float
) -> float:
    """k-D torus dim-decomposed ring AR — ``02_topology_mapping.md`` §3.4.

        t = 2·Σ(d_i - 1)·α + 2·(N-1)/N · M / BW,    N = ∏ d_i

    Latency telescopes from flat ring's ``2(N-1)·α`` to ``2·Σ(d_i-1)·α``;
    BW coefficient stays at ring's ``2(N-1)/N`` floor (per-dim chunks
    telescope cleanly across phases). Reduces to ring AR for ``dims=(N,)``.

    Costs the ideal dim-aligned schedule. Worst-case group misalignment is a
    dispatch-layer concern (see ``02_topology_mapping.md`` §5.2.2).
    """
    N = _prod(dims) if dims else 0
    if N <= 1 or bw_Bps <= 0:
        return 0.0
    hops = sum(d - 1 for d in dims)
    return 2 * hops * alpha_s + 2 * ((N - 1) / N) * (M / bw_Bps)


def torus_all_gather(
    M: float, dims: Sequence[int], alpha_s: float, bw_Bps: float
) -> float:
    """k-D torus dim-decomposed ring AG — ``02_topology_mapping.md`` §3.5.

        t = Σ(d_i - 1)·α + (N - 1)/N · M / BW,    N = ∏ d_i

    Half of torus AR. M is the per-rank gathered output.
    """
    N = _prod(dims) if dims else 0
    if N <= 1 or bw_Bps <= 0:
        return 0.0
    hops = sum(d - 1 for d in dims)
    return hops * alpha_s + ((N - 1) / N) * (M / bw_Bps)


def torus_reduce_scatter(
    M: float, dims: Sequence[int], alpha_s: float, bw_Bps: float
) -> float:
    """k-D torus dim-decomposed ring RS — ``02_topology_mapping.md`` §3.5.

        t = Σ(d_i - 1)·α + (N - 1)/N · M / BW,    N = ∏ d_i

    Time-reverse of torus AG. M is the per-rank pre-scatter input.
    """
    N = _prod(dims) if dims else 0
    if N <= 1 or bw_Bps <= 0:
        return 0.0
    hops = sum(d - 1 for d in dims)
    return hops * alpha_s + ((N - 1) / N) * (M / bw_Bps)


def torus_a2a(
    M: float,
    dims: Sequence[int],
    alpha_s: float,
    bw_Bps: float,
    wraparound: bool = True,
) -> float:
    """Bisection-bound A2A on k-D torus / k-D mesh —
    ``02_topology_mapping.md`` §3.6 (torus) / §4.2 (mesh).

    Wraparound torus (``wraparound=True``):

        diam = Σ ⌊d_i / 2⌋
        t ≈ diam·α + (d_max / 8) · M / BW

    The min-bisection severs ``2N/d_max`` links; cross-cut traffic is
    ``N·M/4`` per direction, so each cut link carries ``d_max·M/8`` one way.

    k-D mesh (``wraparound=False``):

        diam = Σ (d_i - 1)
        t ≈ diam·α + (d_max / 4) · M / BW

    Wraparound edges missing → bisection cut halves → cut links carry 2×
    the bytes. Diameter also changes from ``Σ⌊d_i/2⌋`` (torus) to
    ``Σ(d_i-1)`` (open-line mesh).

    Only ``d_max`` (not N) scales the BW term: asymmetric layouts pay
    disproportionately. At cubic ``d_i = N^(1/k)`` with ``wraparound=True``,
    the BW term equals ``M / BW`` — torus matches star pairwise.

    M is the per-rank total A2A payload (G chunks of size M/G).
    """
    N = _prod(dims) if dims else 0
    if N <= 1 or bw_Bps <= 0:
        return 0.0
    if wraparound:
        diam = sum(d // 2 for d in dims)
        denom = 8.0
    else:
        diam = sum(d - 1 for d in dims)
        denom = 4.0
    d_max = max(dims)
    return diam * alpha_s + d_max * M / (denom * bw_Bps)


def torus_broadcast(
    M: float,
    dims: Sequence[int],
    alpha_s: float,
    bw_Bps: float,
    wraparound: bool = True,
    pipelined: bool = False,
) -> float:
    """k-D torus / k-D mesh broadcast — ``02_topology_mapping.md`` §3.2 / §4.2.

    Wraparound torus (bidirectional ring per dim):

        non-pipelined:  t = Σ⌊d_i/2⌋·α + Σ⌊d_i/2⌋ · M / BW
        pipelined:      t ≈ Σ⌊d_i/2⌋·α + M / BW

    k-D mesh (open-line per dim, unidirectional from end):

        non-pipelined:  t = Σ(d_i-1)·α + Σ(d_i-1) · M / BW
        pipelined:      t ≈ Σ(d_i-1)·α + M / BW

    Pipelined form hits the ``M/BW`` ceiling (``02_topology_mapping.md`` §5.1
    BC row).
    """
    N = _prod(dims) if dims else 0
    if N <= 1 or bw_Bps <= 0:
        return 0.0
    diam = sum(d // 2 for d in dims) if wraparound else sum(d - 1 for d in dims)
    bw_coeff = 1.0 if pipelined else diam
    return diam * alpha_s + bw_coeff * (M / bw_Bps)


def torus_reduce(
    M: float,
    dims: Sequence[int],
    alpha_s: float,
    bw_Bps: float,
    wraparound: bool = True,
    pipelined: bool = False,
) -> float:
    """k-D torus / k-D mesh reduce — time-reverse of torus_broadcast,
    ``02_topology_mapping.md`` §3.3 / §4.2.

    Same α-β cost as torus_broadcast (sink instead of source).
    """
    return torus_broadcast(M, dims, alpha_s, bw_Bps, wraparound, pipelined)


# ---------------------------------------------------------------------------
# Hierarchical composition (multi-tier Clos / fat-tree)
# ---------------------------------------------------------------------------


def hierarchical_all_reduce(
    t_rs_inner: float,
    t_ar_outer: float,
    t_ag_inner: float,
) -> float:
    """Hierarchical AR composition — ``03_hierarchical_topologies.md`` §2.1.

        AR ≡ inner RS → outer sub-AR → inner AG

    The outer sub-AR ships the *telescoped* payload ``M·L/N`` (M = per-rank
    full vector, L = number of outer groups, N = total ranks). Caller
    computes each phase cost using the per-tier (α, BW) and an algorithm of
    choice (ring, DBT, INC, torus dim-decomp, …).

    For convenience helpers, see ``hierarchical_all_reduce_ring_ring``.
    """
    return t_rs_inner + t_ar_outer + t_ag_inner


def hierarchical_all_reduce_ring_ring(
    M: float,
    N: int,
    L: int,
    alpha_inner: float,
    bw_inner: float,
    alpha_outer: float,
    bw_outer: float,
) -> float:
    """Hierarchical AR with ring at both tiers —
    ``03_hierarchical_topologies.md`` §2.1 (NVL72+IB SuperPOD case study).

    L outer groups × (N/L) inner ranks each. M is the per-rank full vector.
    Outer phase ships the telescoped payload ``M·L/N`` per the §2.1
    derivation. Reduces to flat ring AR when ``L = 1`` or ``N = L``.
    """
    if N <= 1:
        return 0.0
    if L <= 1:
        return ring_all_reduce(M, N, alpha_inner, bw_inner)
    if L >= N:
        return ring_all_reduce(M, N, alpha_outer, bw_outer)
    n_inner = N // L
    payload_outer = M * L / N
    rs = ring_reduce_scatter(M, n_inner, alpha_inner, bw_inner)
    ar = ring_all_reduce(payload_outer, L, alpha_outer, bw_outer)
    ag = ring_all_gather(M, n_inner, alpha_inner, bw_inner)
    return rs + ar + ag


def hierarchical_all_gather(
    t_ag_inner: float,
    t_ag_outer: float,
) -> float:
    """Hierarchical AG composition — ``03_hierarchical_topologies.md`` §2.1
    summary table.

        AG ≡ inner AG → outer AG

    α: ``(N/L - 1)·α_inner + (L - 1)·α_outer``;
    BW: ``(N - 1)/N · M / BW`` (whichever tier dominates).

    No payload telescoping — both phases carry their respective per-rank
    output volumes.
    """
    return t_ag_inner + t_ag_outer


def hierarchical_reduce_scatter(
    t_rs_outer: float,
    t_rs_inner: float,
) -> float:
    """Hierarchical RS composition — ``03_hierarchical_topologies.md`` §2.1.

        RS ≡ outer RS → inner RS

    Time-reverse of hierarchical AG.
    """
    return t_rs_outer + t_rs_inner


# ---------------------------------------------------------------------------
# Realistic-cost helpers
# ---------------------------------------------------------------------------


def realistic_cost(
    cost_fn,
    *args,
    eta_alpha: float = 1.0,
    eta_beta: float = 1.0,
    alpha_pos: int = -2,
    bw_pos: int = -1,
    **kwargs,
) -> float:
    """Apply η contention coefficients to any primitive in this module —
    ``05_contention_and_congestion.md`` §4.

    Looks up ``alpha_s`` and ``bw_Bps`` in ``args`` by position
    (default: last two, matching every primitive's signature in this file)
    and applies ``apply_eta`` before calling. Use only when the primitive's
    signature places (α, BW) at the conventional positions; otherwise pass
    pre-discounted values directly via ``apply_eta``.

    Example:

        from collective_cost import ring_all_reduce, realistic_cost
        # NVLink+NVSwitch crossbar profile (η_α=1.0, η_β=0.80)
        t = realistic_cost(ring_all_reduce, M, G, alpha, bw,
                           eta_alpha=1.0, eta_beta=0.80)
    """
    args = list(args)
    args[alpha_pos], args[bw_pos] = apply_eta(
        args[alpha_pos], args[bw_pos], eta_alpha, eta_beta
    )
    return cost_fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Sanity check against the canonical regression anchors
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """Reproduce the canonical N=512 anchors from CLAUDE.md.

    Targets (M=16 MB, α=0.5 μs, BW=900 GB/s):
      04 §3.1 ideal:
        - star + ring AR              ≈ 546 μs
        - star + DBT (dual-touch)     ≈ 45 μs    (uses 2·M/BW BW term)
        - hyp. 512-port star + NVLS   ≈ 18.8 μs
        - torus 8³ dim-decomp ring AR ≈ 57 μs
      05 §6.1 realistic η:
        - INC (η_β=0.52)              ≈ 35 μs
        - DBT (η_β=0.80)              ≈ 53 μs
        - torus (η_α=1.20, η_β=0.60)  ≈ 84 μs
    """
    M = 16e6
    G = 512
    alpha = 0.5e-6
    bw = 900e9

    star_ring = ring_all_reduce(M, G, alpha, bw) * 1e6
    # DBT dual-touch ceiling — explicit n_β=2 form per 04 §3.1 anchor:
    log_G = math.ceil(math.log2(G))
    star_dbt_dt = (2 * log_G * alpha + 2.0 * (M / bw)) * 1e6
    star_inc = inc_all_reduce(M, alpha, bw) * 1e6
    torus = torus_all_reduce(M, (8, 8, 8), alpha, bw) * 1e6

    print(f"Ideal at N=512, M=16MB, α=0.5μs, BW=900GB/s:")
    print(f"  star + ring AR:               {star_ring:7.2f} μs   (target ~546)")
    print(f"  star + DBT (dual-touch):      {star_dbt_dt:7.2f} μs   (target ~45)")
    print(f"  hyp. 512-port star + NVLS:    {star_inc:7.2f} μs   (target ~18.8)")
    print(f"  torus 8³ dim-decomp ring AR:  {torus:7.2f} μs   (target ~57)")

    # Realistic η — 05 §6.1
    a_x, b_x = apply_eta(alpha, bw, eta_alpha=1.00, eta_beta=0.52)  # NVLS
    inc_real = (2 * a_x + M / b_x) * 1e6
    a_x, b_x = apply_eta(alpha, bw, eta_alpha=1.00, eta_beta=0.80)  # crossbar
    dbt_real = (2 * log_G * a_x + 2.0 * (M / b_x)) * 1e6
    a_x, b_x = apply_eta(alpha, bw, eta_alpha=1.20, eta_beta=0.60)  # torus
    torus_real = torus_all_reduce(M, (8, 8, 8), a_x, b_x) * 1e6

    print(f"\nRealistic η at the same anchor:")
    print(f"  star + NVLS (η_β=0.52):       {inc_real:7.2f} μs   (target ~35)")
    print(f"  star + DBT  (η_β=0.80):       {dbt_real:7.2f} μs   (target ~53)")
    print(f"  torus (η_α=1.20, η_β=0.60):   {torus_real:7.2f} μs   (target ~84)")


if __name__ == "__main__":
    _self_test()
