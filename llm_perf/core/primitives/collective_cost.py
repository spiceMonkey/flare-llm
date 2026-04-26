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
    """Double binary tree (pipelined BW-optimal) all-reduce — collectives.md §4.1.

        t = 2·⌈log₂ G⌉·α + 2(G-1)/G · M/BW

    Pipelined DBT matches ring's bandwidth term (per [SST09]) while cutting the
    α-term from 2(G-1) to 2·⌈log₂ G⌉. Empirically NCCL crosses over ring→DBT at
    small M [DEMYST-NCCL].
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    return 2 * math.ceil(math.log2(G)) * alpha_s + 2 * ((G - 1) / G) * (M / bw_Bps)


def pairwise_a2a(M: float, G: int, alpha_s: float, bw_Bps: float) -> float:
    """Pairwise direct-send all-to-all with Dispatch+Combine round-trip
    — collectives.md §5.1 / decode.md §5.2.1.

        t = 2(G-1)·α + 2(G-1) · M/(G·BW)

    The factor of 2 bakes in the bidirectional nature of MoE routing
    (each token traverses the fabric once on Dispatch and once on
    Combine). M is the one-way per-device payload (typically k·H·b per
    token, summed across the step's tokens). Per-rank wire-side bytes
    are (G-1)/G · M (each rank ships its (G-1)/G fraction); the doubled
    α and BW terms reflect the round-trip.

    Underlying NCCL primitive is pairwise direct-send (every rank exchanges
    with every other in parallel through the switch). Bruck / log-hop A2A
    is reference-only and not shipped on any production stack covered here
    — see `documentation/explaining/collectives/01_collective_algorithms.md`
    Appendix B.5.
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    return 2 * (G - 1) * alpha_s + 2 * (G - 1) * (M / (G * bw_Bps))


# Back-compat alias: existing imports `from ...collective_cost import
# ring_moe_all_to_all` keep working. The "ring" name is misleading per the
# new collectives.md §5.1 (the underlying primitive is pairwise direct-send,
# not a ring). New code should prefer `pairwise_a2a`.
ring_moe_all_to_all = pairwise_a2a


def tree_moe_all_to_all(M: float, G: int, alpha_s: float, bw_Bps: float) -> float:
    """**Deprecated.** Double binary tree MoE A2A is reference-only and not
    shipped on any production stack covered here — see collectives.md §5.1
    and `01_collective_algorithms.md` App. B.5. Use `pairwise_a2a` for the
    star A2A cost; `inc_a2a` for HW A2A on Tomahawk Ultra / Rubin.

        t = 2·⌈log₂ G⌉·α + 2(G-1)/G · M/BW

    Emits a `DeprecationWarning` on call. Slated for removal once all
    `ep_algorithm='tree'` call sites have been migrated.
    """
    import warnings
    warnings.warn(
        "tree_moe_all_to_all is deprecated: tree A2A is reference-only and "
        "not shipped on any production stack (collectives.md §5.1). Use "
        "pairwise_a2a or inc_a2a instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    return 2 * math.ceil(math.log2(G)) * alpha_s + 2 * ((G - 1) / G) * (M / bw_Bps)


def ring_all_gather(M: float, G: int, alpha_s: float, bw_Bps: float) -> float:
    """Ring all-gather — decode.md §5.4 / collectives.md §4.1.

        t = (G-1)·α + (G-1) · M/BW

    Used for the SP KV all-gather. M is the per-rank KV shard size.
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    return (G - 1) * alpha_s + (G - 1) * (M / bw_Bps)


def ring_reduce_scatter(M: float, G: int, alpha_s: float, bw_Bps: float) -> float:
    """Ring reduce-scatter — collectives.md §4.1 (RS is the time-reverse of AG).

        t = (G-1)·α + (G-1) · M/BW

    M is the per-rank shard. Used as the inner phase of hierarchical AR
    (§3.3 RS → sub-AR → AG composition); not called directly from the
    decode/prefill pipelines today.
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    return (G - 1) * alpha_s + (G - 1) * (M / bw_Bps)


def inc_all_reduce(M: float, alpha_s: float, bw_Bps: float) -> float:
    """In-network all-reduce — collectives/03_in_network_collectives.md §1.2-1.3.

        t = 2·α_switch + M / BW

    The switch ALU reduces N inputs in-fabric and multicasts back. Endpoint
    pushes M once upstream, receives M once downstream — no peer forwarding —
    so the BW term matches the raw link rate (BW_eff = BW, halving ring's
    2·M/BW). `alpha_s` is the **summed** per-tier switch cut-through α on
    the INC aggregation tree (one direction); the factor of 2 covers the
    reduce-up + multicast-down round trip. For a single-switch star pass
    alpha_s = α_switch; for a k-tier scale-out tree pass alpha_s = Σ α_tier.
    """
    if bw_Bps <= 0:
        return 0.0
    return 2 * alpha_s + M / bw_Bps


def inc_all_gather(M: float, G: int, alpha_s: float, bw_Bps: float) -> float:
    """In-network all-gather — collectives.md §4.4.

        t = 2·α_switch + (G-1) · M / BW

    INC collapses only the α-term for AG: software ring AG already runs at
    BW_eff = BW on a full-duplex star, so the BW term is unchanged from
    ring. The α-term collapses from (G-1)·α_endpoint to 2·α_switch.
    `alpha_s` is the summed per-tier switch α on the INC path.
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    return 2 * alpha_s + (G - 1) * (M / bw_Bps)


def inc_reduce_scatter(M: float, G: int, alpha_s: float, bw_Bps: float) -> float:
    """In-network reduce-scatter — collectives.md §4.4 (mirrors inc_all_gather).

        t = 2·α_switch + (G-1) · M / BW

    α-only collapse. Like AG, RS already runs at BW_eff = BW under software
    ring on a full-duplex star, so the switch ALU's contribution is purely
    on the α-term. Used as the inner phase of hierarchical AR when the
    inner tier is INC-enabled.
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    return 2 * alpha_s + (G - 1) * (M / bw_Bps)


def inc_a2a(M: float, G: int, alpha_s: float, bw_Bps: float) -> float:
    """Hardware all-to-all (crossbar scatter-gather) — collectives.md §5.4.

        t = α_switch + (G-1)/G · M / BW

    Tomahawk Ultra (Ethernet, shipped 2025) and planned Rubin-generation
    NVSwitches collapse software A2A's (G-1) endpoint-driven scheduling
    rounds into one switch-driven transaction (per-chunk descriptor parsing
    + parallel routing within the crossbar). Per-rank wire-side bytes are
    unchanged from software pairwise — the switch routes verbatim, no
    aggregation or replication — so the BW term stays at the bisection
    bound. The α-only efficiency win comes from descriptor batching at
    the rank, not from algorithmic-tree collapse.

    `alpha_s` is α_switch (single switch transaction); not 2·α_switch as
    AR/AG/RS use, because A2A is a single descriptor + single routing pass
    (no reduce-up + multicast-down round trip).

    Not on shipping NVSwitch Gen4 (NVL72) or Quantum-X800; only routes here
    when every crossed tier declares inc == "hw_a2a".
    """
    if G <= 1 or bw_Bps <= 0:
        return 0.0
    return alpha_s + ((G - 1) / G) * (M / bw_Bps)


def _prod(xs: Sequence[int]) -> int:
    n = 1
    for x in xs:
        n *= x
    return n


def torus_all_reduce(
    M: float, dims: Sequence[int], alpha_s: float, bw_Bps: float
) -> float:
    """k-D torus dim-by-dim ring all-reduce — collectives.md §3.2.

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
    """k-D torus dim-by-dim ring all-gather — collectives.md §4.2.

        t = Σ(D_i - 1)·α + (N - 1) · M/BW,    N = ∏ D_i

    M is the per-rank shard; the gathered volume at each rank is N·M.
    Reduces to ring_all_gather for dims=(N,).
    """
    N = _prod(dims) if dims else 0
    if N <= 1 or bw_Bps <= 0:
        return 0.0
    hops = sum(d - 1 for d in dims)
    return hops * alpha_s + (N - 1) * (M / bw_Bps)


def torus_reduce_scatter(
    M: float, dims: Sequence[int], alpha_s: float, bw_Bps: float
) -> float:
    """k-D torus dim-by-dim ring reduce-scatter — collectives.md §4.2 (RS = AG time-reverse).

        t = Σ(D_i - 1)·α + (N - 1) · M/BW,    N = ∏ D_i

    M is the per-rank shard. Used as the inner phase of hierarchical AR
    when the inner tier is a torus; not called directly from the
    decode/prefill pipelines today.
    """
    N = _prod(dims) if dims else 0
    if N <= 1 or bw_Bps <= 0:
        return 0.0
    hops = sum(d - 1 for d in dims)
    return hops * alpha_s + (N - 1) * (M / bw_Bps)


def torus_moe_all_to_all(
    M: float, dims: Sequence[int], alpha_s: float, bw_Bps: float,
    wraparound: bool = True,
) -> float:
    """Bisection-limited all-to-all on a k-D torus or k-D mesh — collectives.md §5.2.

    Wraparound torus (default, `wraparound=True`):

        t ≈ diam·α + D_max·M / (8·BW_link),    diam = Σ ⌊D_i / 2⌋

    Per-link bytes in one direction: L→R cross-cut traffic is
    (N/2)·(N/2)·(M/N) = N·M/4. The min-bisection severs 2·N/D_max links
    on a wraparound torus, so each cut link carries (N·M/4)/(2·N/D_max) =
    D_max·M/8 bytes in one direction.

    k-D mesh (`wraparound=False`): same diameter formula but the bisection
    cut is **halved** (no wraparound edges severed), so each cut link carries
    twice the bytes:

        t ≈ diam·α + D_max·M / (4·BW_link)

    A2A on k-D mesh is exactly 2× worse than same-shape torus on the BW term;
    AR / AG / RS are unaffected (open-line bucket brigade still telescopes
    BW-optimally — see `torus_all_reduce` / `torus_all_gather`).

    Only D_max — not N — scales the bandwidth term; asymmetric layouts pay
    disproportionately. At cubic D_i = N^(1/k) and `wraparound=True`, the BW
    term equals M/BW_link — torus matches star pairwise.

    M is the per-rank total A2A payload (Dispatch+Combine baked in by the
    caller, matching pairwise_a2a's convention).
    """
    N = _prod(dims) if dims else 0
    if N <= 1 or bw_Bps <= 0:
        return 0.0
    diam = sum(d // 2 for d in dims)
    d_max = max(dims)
    denom = 8.0 if wraparound else 4.0
    return diam * alpha_s + d_max * M / (denom * bw_Bps)


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
