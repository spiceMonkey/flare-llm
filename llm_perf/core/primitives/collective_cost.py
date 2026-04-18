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
