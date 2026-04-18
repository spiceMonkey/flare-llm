"""Collective algorithm cost primitives — α-β latency model.

Single source of truth for the ring/tree collective formulas in
`documentation/modeling/decode.md §5`. Both `core/decode_model.py`
(after consolidation) and `core/prefill_model.py` dispatch through
these functions; only the per-device message size differs by phase.

All functions return seconds. Inputs are:
  - M:          per-device message size in bytes
  - G:          collective group size (number of ranks)
  - alpha_s:    per-hop latency in seconds
  - bw_Bps:     effective per-link bandwidth in bytes/s

Functions return 0.0 when the collective is a no-op (G ≤ 1) or when the
link bandwidth is non-positive, so callers can skip shape-level guards.
"""

import math
from typing import Sequence


def p2p_hop(M: float, alpha_s: float, bw_Bps: float) -> float:
    """Point-to-point hop — decode.md §5.1.

        t = α + M / BW

    Used for the PP stage-to-stage forward. The caller is responsible
    for gating this on "is there more than one PP stage."
    """
    if bw_Bps <= 0:
        return 0.0
    return alpha_s + M / bw_Bps


def ring_all_reduce(M: float, G: int, alpha_s: float, bw_Bps: float) -> float:
    """Ring all-reduce — decode.md §5.3.1.

        t = 2(G-1)·α + 2·(G-1)/G · M/BW
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    return 2 * (G - 1) * alpha_s + 2 * ((G - 1) / G) * (M / bw_Bps)


def tree_all_reduce(M: float, G: int, alpha_s: float, bw_Bps: float) -> float:
    """Tree (recursive-halving/doubling) all-reduce — decode.md §5.3.2.

        t ≈ 2·⌈log₂ G⌉·α + 2·M/BW
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    return 2 * math.ceil(math.log2(G)) * alpha_s + 2 * (M / bw_Bps)


def ring_moe_all_to_all(M: float, G: int, alpha_s: float, bw_Bps: float) -> float:
    """Ring MoE all-to-all with Dispatch+Combine round-trip — decode.md §5.2.1.

        t = 2(G-1)·α + 2(G-1) · M/(G·BW)

    The factor of 2 bakes in the bidirectional nature of MoE routing
    (each token traverses the fabric once on Dispatch and once on
    Combine). M is the one-way per-device payload (typically k·H·b per
    token, summed across the step's tokens).
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    return 2 * (G - 1) * alpha_s + 2 * (G - 1) * (M / (G * bw_Bps))


def tree_moe_all_to_all(M: float, G: int, alpha_s: float, bw_Bps: float) -> float:
    """Tree MoE all-to-all with Dispatch+Combine round-trip — decode.md §5.2.2.

        t ≈ 2·⌈log₂ G⌉·α + 2·M/BW
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    return 2 * math.ceil(math.log2(G)) * alpha_s + 2 * (M / bw_Bps)


def ring_all_gather(M: float, G: int, alpha_s: float, bw_Bps: float) -> float:
    """Ring all-gather — decode.md §5.4.

        t = (G-1)·α + (G-1) · M/BW

    Used for the SP KV all-gather. M is the per-rank KV shard size.
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    return (G - 1) * alpha_s + (G - 1) * (M / bw_Bps)


def _prod(xs: Sequence[int]) -> int:
    n = 1
    for x in xs:
        n *= x
    return n


def torus_all_reduce(
    M: float, dims: Sequence[int], alpha_s: float, bw_Bps: float
) -> float:
    """k-D torus dim-by-dim ring all-reduce — switching.md §8.3.

        t = 2·Σ(D_i - 1)·α + 2·(N-1)/N · M/BW,    N = ∏ D_i

    Latency term compresses from 2(N-1)α (flat ring) to 2·Σ(D_i-1)α by
    running RS along each dim then AG in reverse (Patarasuk-Yuan bandwidth
    term is preserved — bound holds on any tree-connected fabric).
    Reduces to ring_all_reduce for dims=(N,).

    Best case: dim-aligned group layout under structured admissible traffic.
    Worst-case group misalignment is surfaced by the dispatch layer, not
    this primitive — here we cost the ideal dim-by-dim schedule.
    """
    N = _prod(dims) if dims else 0
    if N <= 1 or bw_Bps <= 0:
        return 0.0
    hops = sum(d - 1 for d in dims)
    return 2 * hops * alpha_s + 2 * ((N - 1) / N) * (M / bw_Bps)


def torus_all_gather(
    M: float, dims: Sequence[int], alpha_s: float, bw_Bps: float
) -> float:
    """k-D torus dim-by-dim ring all-gather — switching.md §8.4.

        t = Σ(D_i - 1)·α + (N - 1) · M/BW,    N = ∏ D_i

    M is the per-rank shard; the gathered volume at each rank is N·M.
    Reduces to ring_all_gather for dims=(N,).
    """
    N = _prod(dims) if dims else 0
    if N <= 1 or bw_Bps <= 0:
        return 0.0
    hops = sum(d - 1 for d in dims)
    return hops * alpha_s + (N - 1) * (M / bw_Bps)


def torus_moe_all_to_all(
    M: float, dims: Sequence[int], alpha_s: float, bw_Bps: float
) -> float:
    """Bisection-limited all-to-all on a k-D wraparound torus —
    switching.md §8.5.

        t ≈ diam·α + D_max·M / (4·BW_link)
        diam = Σ ⌊D_i / 2⌋,    D_max = max(dims)

    Aggregate traffic N·M/2 crosses the min-bisection; wraparound torus
    bisection is BW_bisect^min = 2·N·BW_link / D_max, so only D_max — not
    N — scales the bandwidth term. Asymmetric layouts pay disproportionately.

    M is the per-rank total A2A payload (Dispatch+Combine baked in by the
    caller, matching ring_moe_all_to_all's convention).
    """
    N = _prod(dims) if dims else 0
    if N <= 1 or bw_Bps <= 0:
        return 0.0
    diam = sum(d // 2 for d in dims)
    d_max = max(dims)
    return diam * alpha_s + d_max * M / (4 * bw_Bps)


def dragonfly_all_reduce(
    M: float,
    p: int, a: int, g: int,
    alpha_r_s: float, bw_r_Bps: float,
    alpha_l_s: float, bw_l_Bps: float,
    alpha_g_s: float, bw_g_Bps: float,
    worst_case: bool = False,
) -> float:
    """Hierarchical 3-level ring AR on a balanced dragonfly — switching.md §9.4.

        t_AR = 2(p-1)α_r + 2(p-1)/p · M/BW_r           (L0 intra-router)
             + 2(a-1)α_l + 2(a-1)/a · (M/p)/BW_l       (L1 intra-group)
             + c·[2(g-1)α_g + 2(g-1)/g · (M/(p·a))/BW_g]   (L2 inter-group)

    where `c = 2` under `worst_case=True` (Valiant routing: L2 α-hops and
    effective BW both double) and `c = 1` otherwise (adaptive minimal
    routing under structured admissible traffic, [KDSA08 §3-4, JAIN22 §4]).

    Payload shrinks down the tiers as L0 RS → L1 RS: by factor p after L0,
    then by factor a after L1, so L2 sees M/(p·a) per rank. Trivial tiers
    (p=1, a=1, or g=1) contribute zero — i.e. `dragonfly_all_reduce` with
    a=g=1 reduces to `ring_all_reduce(M, p, α_r, BW_r)`.
    """
    if p * a * g <= 1:
        return 0.0
    t = 0.0
    if p > 1 and bw_r_Bps > 0:
        t += 2 * (p - 1) * alpha_r_s + 2 * ((p - 1) / p) * (M / bw_r_Bps)
    if a > 1 and bw_l_Bps > 0:
        t += 2 * (a - 1) * alpha_l_s + 2 * ((a - 1) / a) * ((M / p) / bw_l_Bps)
    if g > 1 and bw_g_Bps > 0:
        l2 = 2 * (g - 1) * alpha_g_s + 2 * ((g - 1) / g) * ((M / (p * a)) / bw_g_Bps)
        t += 2 * l2 if worst_case else l2
    return t


def dragonfly_all_gather(
    M: float,
    p: int, a: int, g: int,
    alpha_r_s: float, bw_r_Bps: float,
    alpha_l_s: float, bw_l_Bps: float,
    alpha_g_s: float, bw_g_Bps: float,
    worst_case: bool = False,
) -> float:
    """Hierarchical 3-level ring AG on a balanced dragonfly — switching.md §9.4.

        t_AG = (p-1)α_r + (p-1) · M/BW_r                 (L0 intra-router)
             + (a-1)α_l + (a-1) · (p·M)/BW_l             (L1 intra-group)
             + c·[(g-1)α_g + (g-1) · (p·a·M)/BW_g]       (L2 inter-group)

    Per-rank shard grows by factor p after L0 AG and by factor p·a after L1,
    so L2 sees (p·a·M) per rank. `c = 2` under worst_case (Valiant L2 BW
    halving), else `c = 1`. Reduces to `ring_all_gather(M, p, α_r, BW_r)`
    when a=g=1.
    """
    if p * a * g <= 1:
        return 0.0
    t = 0.0
    if p > 1 and bw_r_Bps > 0:
        t += (p - 1) * alpha_r_s + (p - 1) * (M / bw_r_Bps)
    if a > 1 and bw_l_Bps > 0:
        t += (a - 1) * alpha_l_s + (a - 1) * ((p * M) / bw_l_Bps)
    if g > 1 and bw_g_Bps > 0:
        l2 = (g - 1) * alpha_g_s + (g - 1) * ((p * a * M) / bw_g_Bps)
        t += 2 * l2 if worst_case else l2
    return t


def dragonfly_moe_all_to_all(
    M: float,
    p: int, a: int, g: int,
    alpha_r_s: float, bw_r_Bps: float,
    alpha_l_s: float, bw_l_Bps: float,
    alpha_g_s: float, bw_g_Bps: float,
    worst_case: bool = False,
) -> float:
    """Hierarchical 3-level MoE A2A on a balanced dragonfly — switching.md §9.4.

    Same three-tier structure as `dragonfly_all_reduce`; each tier runs
    Dispatch+Combine on the payload arriving at that tier:

        t_A2A = 2(p-1)α_r + 2(p-1)/p · M/BW_r            (L0)
              + 2(a-1)α_l + 2(a-1)/a · (M/p)/BW_l        (L1)
              + c·[2(g-1)α_g + 2(g-1)/g · (M/(p·a))/BW_g] (L2)

    L2 typically dominates for MoE A2A because expert routing is often
    adversarial relative to the dragonfly's uniform-admissible assumption;
    `worst_case=True` (Valiant doubling) is more frequently justified here
    than for AR. Mirrors the A2A ≡ AR formula identity already used in
    `ring_moe_all_to_all` / `ring_all_reduce`.
    """
    return dragonfly_all_reduce(
        M, p, a, g,
        alpha_r_s, bw_r_Bps,
        alpha_l_s, bw_l_Bps,
        alpha_g_s, bw_g_Bps,
        worst_case=worst_case,
    )


def aggregate_per_stage(
    L: int,
    L_moe: int,
    PP: int,
    n_TP: int,
    t_TP: float,
    n_SP: int,
    t_SP: float,
    n_EP: int,
    t_EP: float,
    t_PP: float,
) -> float:
    """Per-pipeline-stage communication time — decode.md §5.5.

        t = (L/PP)·(n_TP·t_TP + n_SP·t_SP)
          + (L_moe/PP)·(n_EP·t_EP)
          + t_PP

    TP and SP collectives apply to every layer on the stage.
    EP collectives apply only to MoE layers (L_moe/PP per stage).
    The PP hop is a single inter-stage forward per step.

    Prefill cross-references this aggregation — see
    `documentation/modeling/prefill.md §3.2` ("Following the same
    structure as decode.md §5.5").
    """
    return (L / PP) * (n_TP * t_TP + n_SP * t_SP) + (L_moe / PP) * (n_EP * t_EP) + t_PP
