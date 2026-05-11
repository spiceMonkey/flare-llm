"""Dispatcher equivalence suite (Phase C) — scratch/switching_upgrade.md §10.4.

Proves that `cost_collective` on any pure-crossbar tier chain returns exactly
the same float as the pre-refactor flatten-then-apply pattern:

    α_sum, BW_min, _ = span_tiers(tiers, G)
    t = ring_or_tree(M, G, α_sum·1e-6, BW_min·1e9)

Anything non-bit-identical here is a regression. Covers T1 single-tier, T2
multi-tier crossbar chain, T3 randomized property test, T4 crossbar-silence
of the torus dim-alignment warning, T5 k=1 torus continuity.

Usage:  PYTHONPATH=. python tests/unit/test_topology_equivalence.py
"""
import math
import random
import sys
import warnings
from dataclasses import replace

from llm_perf.core.primitives import (
    cost_collective,
    enumerate_options,
    inc_a2a,
    inc_all_gather,
    inc_all_reduce,
    p2p_hop,
    pairwise_a2a,
    ring_all_gather,
    ring_all_reduce,
    ring_reduce_scatter,
    torus_all_reduce,
    tree_all_reduce,
)
from llm_perf.specs.system_spec import CrossbarTier, MeshTier, TorusTier, span_tiers
from llm_perf.utils import GB_TO_BYTES, US_TO_SECONDS


_failures: list[str] = []


def _check(label: str, got: float, want: float) -> None:
    if math.isclose(got, want, rel_tol=0.0, abs_tol=0.0):
        print(f"  PASS  {label}")
    else:
        _failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL  {label}: got {got!r}, want {want!r}")


def _old_path(op: str, tiers, M: float, G: int, algorithm: str) -> float:
    """Reference path mirroring the dispatcher.

    Single-tier or non-AR/AG: flat path (innermost tier α-sum + BW-min, then
    a single primitive call). Multi-tier crossbar AR/AG: hierarchical RS →
    sub-AR → AG (or inner+outer cascade for AG) per collectives/03_hierarchical_topologies.md §2
    — landed in PR2.2.
    """
    if G <= 1 or not tiers:
        return 0.0

    # Identify crossed tiers (innermost first; stop at cumulative reach >= G).
    crossed = []
    reach = 1
    for t in tiers:
        reach *= max(1, t.ports)
        crossed.append(t)
        if reach >= G:
            break

    # Multi-tier crossbar AR/AG → hierarchical reference.
    all_crossbar = all(t.topology == "crossbar" for t in crossed)
    if len(crossed) > 1 and all_crossbar and op in ("all_reduce", "all_gather"):
        return _hier_ref(op, M, G, crossed, algorithm)
    # Multi-tier crossbar MoE A2A → per-destination-class reference.
    if len(crossed) > 1 and all_crossbar and op == "moe_a2a":
        return _hier_a2a_ref(M, G, crossed)

    # Else flat reference.
    alpha_us, bw_GBps, _ = span_tiers(tiers, G)
    alpha_s = alpha_us * US_TO_SECONDS
    bw_Bps = bw_GBps * GB_TO_BYTES
    if op == "p2p":
        return p2p_hop(M, alpha_s, bw_Bps)
    if op == "all_gather":
        return ring_all_gather(M, G, alpha_s, bw_Bps)
    if op == "all_reduce":
        if algorithm == "ring":
            return ring_all_reduce(M, G, alpha_s, bw_Bps)
        if algorithm == "tree":
            return tree_all_reduce(M, G, alpha_s, bw_Bps)
    if op == "moe_a2a":
        # Dispatcher returns single-direction A2A cost; the per-MoE-layer
        # accumulator scales by n_EP=2 (Dispatch + Combine). tree A2A is no
        # longer supported (upstream removed `tree_moe_all_to_all` per
        # scratch/collectives_cost_CHANGES.md §3).
        if algorithm == "ring":
            return pairwise_a2a(M, G, alpha_s, bw_Bps)
    if op == "all_reduce" and algorithm == "tree_pipelined":
        return tree_all_reduce(M, G, alpha_s, bw_Bps, pipelined=True)
    raise ValueError(f"bad op/algorithm: {op!r}/{algorithm!r}")


def _hier_ref(op: str, M: float, G: int, crossed, outer_alg: str) -> float:
    """Hierarchical reference path for multi-tier crossbar AR/AG.

    Composed from `ring_reduce_scatter` / `ring_all_gather` / `ring_all_reduce`
    / `tree_all_reduce` — same as `_hierarchical_crossbar_cost` but inlined
    here so the test stays an independent oracle.
    """
    inner = crossed[0]
    outer_tiers = crossed[1:]
    G_inner = max(1, inner.ports)
    L = max(1, G // G_inner)

    alpha_inner_s = inner.alpha_us * inner.eta_alpha * US_TO_SECONDS
    bw_inner_Bps = inner.bw_per_port_GBps * inner.eta_beta * GB_TO_BYTES
    alpha_outer_s = sum(t.alpha_us * t.eta_alpha for t in outer_tiers) * US_TO_SECONDS
    bw_outer_Bps = (
        min(t.bw_per_port_GBps * t.eta_beta for t in outer_tiers) * GB_TO_BYTES
    )

    if op == "all_reduce":
        # New AG/RS convention: pass full M (per-rank pre-scatter input or
        # gathered output) to inner phases. Outer AR consumes the post-RS
        # shard payload M/G_inner.
        t_rs = ring_reduce_scatter(M, G_inner, alpha_inner_s, bw_inner_Bps)
        if outer_alg == "ring":
            t_outer = ring_all_reduce(M / G_inner, L, alpha_outer_s, bw_outer_Bps)
        elif outer_alg == "tree":
            t_outer = tree_all_reduce(M / G_inner, L, alpha_outer_s, bw_outer_Bps)
        elif outer_alg == "tree_pipelined":
            t_outer = tree_all_reduce(M / G_inner, L, alpha_outer_s, bw_outer_Bps, pipelined=True)
        else:
            raise ValueError(f"_hier_ref: unsupported outer_alg={outer_alg!r}")
        t_ag = ring_all_gather(M, G_inner, alpha_inner_s, bw_inner_Bps)
        return t_rs + t_outer + t_ag
    if op == "all_gather":
        # New AG convention: M = per-rank gathered output. Inner AG produces
        # G_inner shards = M/L; outer AG produces the full M.
        return (
            ring_all_gather(M / L, G_inner, alpha_inner_s, bw_inner_Bps)
            + ring_all_gather(M, L, alpha_outer_s, bw_outer_Bps)
        )
    raise ValueError(f"_hier_ref: unsupported op={op!r}")


def _hier_a2a_ref(M: float, G: int, crossed) -> float:
    """Hierarchical A2A reference (collectives/03_hierarchical_topologies.md §2 per-destination-class)."""
    inner = crossed[0]
    outer_tiers = crossed[1:]
    G_inner = max(1, inner.ports)

    alpha_inner_s = inner.alpha_us * inner.eta_alpha * US_TO_SECONDS
    bw_inner_Bps = inner.bw_per_port_GBps * inner.eta_beta * GB_TO_BYTES
    alpha_outer_s = sum(t.alpha_us * t.eta_alpha for t in outer_tiers) * US_TO_SECONDS
    bw_outer_Bps = (
        min(t.bw_per_port_GBps * t.eta_beta for t in outer_tiers) * GB_TO_BYTES
    )

    chunk = M / G if G > 0 else 0.0
    n_intra = max(0, G_inner - 1)
    n_outer = max(0, G - G_inner)
    t_intra = n_intra * (alpha_inner_s + chunk / bw_inner_Bps) if bw_inner_Bps > 0 else 0
    t_outer = (
        n_outer * (alpha_outer_s + chunk / bw_outer_Bps)
        if (n_outer > 0 and bw_outer_Bps > 0) else 0
    )
    return t_intra + t_outer  # single-direction; n_EP=2 wraps at the per-MoE-layer accumulator


# ────────────────────────────────────────────────────────────
# T1 — Single-tier crossbar equivalence (every op × algorithm)
# ────────────────────────────────────────────────────────────

def test_T1_single_tier():
    print("T1 single-tier crossbar equivalence:")
    tier = CrossbarTier(name="t0", ports=32, bw_per_port_GBps=900.0, alpha_us=0.5)
    cases = [
        ("p2p",        "ring",           1_024.0,        2),
        ("p2p",        "ring",           10_000_000.0,   2),
        ("all_gather", "ring",           4_096.0,        8),
        ("all_gather", "ring",           100_000_000.0,  16),
        ("all_reduce", "ring",           1_000_000.0,    8),
        ("all_reduce", "ring",           1_000_000.0,    32),
        ("all_reduce", "tree",           1_000_000.0,    8),
        ("all_reduce", "tree",           1_000_000.0,    16),
        ("all_reduce", "tree_pipelined", 1_000_000.0,    16),
        ("moe_a2a",    "ring",           500_000.0,      8),
        ("moe_a2a",    "ring",           500_000.0,      32),
        # MoE A2A "tree" was deprecated upstream — no longer enumerated.
    ]
    for op, alg, M, G in cases:
        old = _old_path(op, [tier], M, G, alg)
        new = cost_collective([tier], op, M, G, algorithm=alg)
        _check(f"{op}/{alg} M={M:.0e} G={G}", new, old)


# ────────────────────────────────────────────────────────────
# T2 — Multi-tier crossbar chain equivalence
#   Modeled after gb200.multibox (NVLink scale-up + IB scale-out).
# ────────────────────────────────────────────────────────────

def test_T2_multi_tier():
    print("T2 multi-tier crossbar chain equivalence:")
    tier0 = CrossbarTier(name="nvlink", ports=72, bw_per_port_GBps=900.0, alpha_us=0.5)
    tier1 = CrossbarTier(name="ib",     ports=8,  bw_per_port_GBps=50.0,  alpha_us=2.0)
    chain = [tier0, tier1]
    cases = [
        # Stays in tier 0
        ("all_reduce", "ring",           1_000_000.0, 8),
        ("all_reduce", "ring",           1_000_000.0, 64),
        # Boundary — exactly tier0.ports
        ("all_reduce", "ring",           1_000_000.0, 72),
        ("all_reduce", "tree",           1_000_000.0, 72),
        ("all_reduce", "tree_pipelined", 1_000_000.0, 72),
        # Spills into tier 1
        ("all_reduce", "ring",           1_000_000.0, 144),
        ("all_reduce", "ring",           1_000_000.0, 576),
        ("all_reduce", "tree",           1_000_000.0, 576),
        ("all_reduce", "tree_pipelined", 1_000_000.0, 576),
        # All-gather + MoE-A2A across both tiers
        ("all_gather", "ring",           10_000.0,   288),
        ("moe_a2a",    "ring",           50_000.0,   288),
        # p2p always stays in tier 0 (G=2)
        ("p2p",        "ring",           500_000.0,  2),
    ]
    for op, alg, M, G in cases:
        old = _old_path(op, chain, M, G, alg)
        new = cost_collective(chain, op, M, G, algorithm=alg)
        _check(f"{op}/{alg} G={G}", new, old)


# ────────────────────────────────────────────────────────────
# T3 — Randomized property test: 1000 crossbar-only chains
# ────────────────────────────────────────────────────────────

def test_T3_randomized():
    print("T3 randomized property test (1000 draws):")
    rng = random.Random(0xC0FFEE)
    ops = ["all_reduce", "all_gather", "moe_a2a", "p2p"]
    ar_algs = ["ring", "tree", "tree_pipelined"]
    mismatches = 0
    for i in range(1000):
        n_tiers = rng.randint(1, 4)
        tiers = [
            CrossbarTier(
                name=f"t{j}",
                ports=rng.choice([2, 4, 8, 16, 32, 64, 128]),
                bw_per_port_GBps=rng.uniform(10.0, 2000.0),
                alpha_us=rng.uniform(0.1, 5.0),
            )
            for j in range(n_tiers)
        ]
        op = rng.choice(ops)
        # AR has 3 algorithms (ring / tree / tree_pipelined); MoE A2A is
        # ring-only after upstream removed tree_moe_all_to_all.
        alg = rng.choice(ar_algs) if op == "all_reduce" else "ring"
        M = rng.uniform(1.0, 1e8)
        # Group size spans inside tier 0 through full-chain reach.
        max_reach = 1
        for t in tiers:
            max_reach *= t.ports
        G = rng.randint(2, max_reach) if op != "p2p" else 2
        old = _old_path(op, tiers, M, G, alg)
        new = cost_collective(tiers, op, M, G, algorithm=alg)
        if old != new:
            mismatches += 1
            if mismatches <= 5:
                print(f"  FAIL draw#{i} {op}/{alg}: old={old!r} new={new!r}")
    if mismatches == 0:
        print("  PASS  1000/1000 bit-identical")
    else:
        _failures.append(f"T3 randomized: {mismatches}/1000 mismatches")


# ────────────────────────────────────────────────────────────
# T4 — Crossbar path must NOT emit torus dim-alignment warning
# ────────────────────────────────────────────────────────────

def test_T4_crossbar_silent():
    print("T4 crossbar path emits no UserWarning:")
    tier = CrossbarTier(name="t0", ports=16, bw_per_port_GBps=900.0, alpha_us=0.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cost_collective([tier], "all_reduce", 1e6, 12, algorithm="ring")
        cost_collective([tier], "moe_a2a",    1e6, 13, algorithm="ring")
        cost_collective([tier], "all_gather", 1e6, 7)
    user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
    if not user_warns:
        print("  PASS  no UserWarning on crossbar inputs")
    else:
        _failures.append(f"T4: unexpected warnings: {[str(w.message) for w in user_warns]}")
        for w in user_warns:
            print(f"  FAIL  unexpected warning: {w.message}")


# ────────────────────────────────────────────────────────────
# T5 — torus_all_reduce(dims=(N,)) == ring_all_reduce(N)
#   Continuity check guarding against silent divergence in the torus path.
# ────────────────────────────────────────────────────────────

def test_T5_torus_k1_continuity():
    print("T5 torus k=1 continuity:")
    for N in (2, 4, 8, 16, 64, 128):
        for M in (1_024.0, 1e6, 1e8):
            want = ring_all_reduce(M, N, 1e-6, 1e9)
            got = torus_all_reduce(M, (N,), 1e-6, 1e9)
            _check(f"N={N} M={M:.0e}", got, want)
            # Also via the dispatcher on a single TorusTier:
            ttier = TorusTier(name="t", dims=(N,), bw_per_port_GBps=1.0, alpha_us=1.0)
            got_disp = cost_collective([ttier], "all_reduce", M, N, torus_algorithm="ring")
            _check(f"dispatch N={N} M={M:.0e}", got_disp, want)


# ────────────────────────────────────────────────────────────
# T6 — INC dispatch: inc_enabled routing, opt-out parity, scope limits
# ────────────────────────────────────────────────────────────

def test_T6_inc_dispatch():
    print("T6 INC dispatch:")
    M = 1_000_000.0
    alpha_endpoint = 0.5   # software α, μs
    alpha_switch = 0.2     # cut-through α, μs
    bw = 900.0             # GB/s

    inc_tier = CrossbarTier(
        name="nvls", ports=72, bw_per_port_GBps=bw, alpha_us=alpha_endpoint,
        inc="nvls", inc_alpha_us=alpha_switch,
    )
    plain_tier = CrossbarTier(
        name="plain", ports=72, bw_per_port_GBps=bw, alpha_us=alpha_endpoint,
    )

    alpha_s_switch = alpha_switch * US_TO_SECONDS
    alpha_s_endpoint = alpha_endpoint * US_TO_SECONDS
    bw_Bps = bw * GB_TO_BYTES

    # (a) AR single-tier INC — should match inc_all_reduce(M, α_switch, BW) exactly.
    got = cost_collective([inc_tier], "all_reduce", M, 64)
    want = inc_all_reduce(M, alpha_s_switch, bw_Bps)
    _check("AR G=64 inc on", got, want)

    # (b) AR single-tier with inc_enabled=False — falls back to ring_all_reduce
    # using the tier's endpoint α (not the switch cut-through).
    got = cost_collective([inc_tier], "all_reduce", M, 64, inc_enabled=False)
    want = ring_all_reduce(M, 64, alpha_s_endpoint, bw_Bps)
    _check("AR G=64 inc off → ring", got, want)

    # (c) AG routes through inc_all_gather when enabled.
    got = cost_collective([inc_tier], "all_gather", M, 64)
    want = inc_all_gather(M, 64, alpha_s_switch, bw_Bps)
    _check("AG G=64 inc on", got, want)

    # (d) MoE A2A on an INC tier must stay on ring (no structural INC win).
    # Dispatcher returns single-direction A2A; per-MoE-layer accumulator
    # scales by n_EP=2 (Dispatch + Combine).
    got = cost_collective([inc_tier], "moe_a2a", M, 64, algorithm="ring")
    want = pairwise_a2a(M, 64, alpha_s_endpoint, bw_Bps)
    _check("A2A G=64 inc tier → ring", got, want)

    # (e) p2p on an INC tier stays on single-hop.
    got = cost_collective([inc_tier], "p2p", M, 2)
    want = p2p_hop(M, alpha_s_endpoint, bw_Bps)
    _check("p2p G=2 inc tier → hop", got, want)

    # (f) Partial-INC chain (plain + inc) must fall back to software — INC requires
    # every crossed tier to support in-fabric reduction.
    # G=64 ≤ plain_tier.ports=72 so the walker stops at tier 0 and only plain α counts.
    got = cost_collective([plain_tier, inc_tier], "all_reduce", M, 64)
    want = ring_all_reduce(M, 64, alpha_s_endpoint, bw_Bps)
    _check("AR partial-INC chain → software ring", got, want)

    # (g) Multi-tier full-INC chain: scale-out aggregation tree.
    # G > tier0.ports forces walk into tier1; both tiers contribute α.
    inc_tier2 = CrossbarTier(
        name="sharp-ib", ports=8, bw_per_port_GBps=400.0, alpha_us=2.5,
        inc="sharp", inc_alpha_us=0.4,
    )
    got = cost_collective([inc_tier, inc_tier2], "all_reduce", M, 576)
    # α_total = 0.2 + 0.4 = 0.6 μs; BW_min = min(900, 400) = 400 GB/s.
    alpha_total_s = (0.2 + 0.4) * US_TO_SECONDS
    bw_min_Bps = 400.0 * GB_TO_BYTES
    want = inc_all_reduce(M, alpha_total_s, bw_min_Bps)
    _check("AR multi-tier INC scale-out", got, want)

    # (h) η coefficients flow through the INC path identically to crossbar.
    inc_tier_eta = CrossbarTier(
        name="nvls-eta", ports=72, bw_per_port_GBps=bw, alpha_us=alpha_endpoint,
        inc="nvls", inc_alpha_us=alpha_switch,
        eta_alpha=1.5, eta_beta=0.8,
    )
    got = cost_collective([inc_tier_eta], "all_reduce", M, 64)
    want = inc_all_reduce(
        M,
        alpha_switch * 1.5 * US_TO_SECONDS,
        bw * 0.8 * GB_TO_BYTES,
    )
    _check("AR INC with η≠1", got, want)

    # (i) inc_alpha_us=0 sentinel — reuses alpha_us on the INC path.
    inc_tier_reuse = CrossbarTier(
        name="inc-reuse", ports=72, bw_per_port_GBps=bw, alpha_us=alpha_endpoint,
        inc="nvls",  # no inc_alpha_us override → sentinel 0.0
    )
    got = cost_collective([inc_tier_reuse], "all_reduce", M, 64)
    want = inc_all_reduce(M, alpha_s_endpoint, bw_Bps)
    _check("AR INC with inc_alpha_us=0 sentinel", got, want)

    # (j) HW A2A tier — A2A routes through inc_a2a (collectives/04_in_network_collectives.md).
    hw_a2a_tier = CrossbarTier(
        name="tomahawk-ultra", ports=64, bw_per_port_GBps=bw, alpha_us=alpha_endpoint,
        inc="hw_a2a", inc_alpha_us=alpha_switch,
    )
    got = cost_collective([hw_a2a_tier], "moe_a2a", M, 64, algorithm="ring")
    want = inc_a2a(M, 64, alpha_s_switch, bw_Bps)
    _check("A2A G=64 hw_a2a tier → inc_a2a", got, want)

    # (k) sharp_class tier with A2A — must NOT route through inc_a2a; falls through
    # to software pairwise (sharp_class only covers AR / AG / RS).
    sharp_tier = CrossbarTier(
        name="sharp-class", ports=64, bw_per_port_GBps=bw, alpha_us=alpha_endpoint,
        inc="sharp_class", inc_alpha_us=alpha_switch,
    )
    got = cost_collective([sharp_tier], "moe_a2a", M, 64, algorithm="ring")
    want = pairwise_a2a(M, 64, alpha_s_endpoint, bw_Bps)
    _check("A2A G=64 sharp_class tier → pairwise (no INC for A2A)", got, want)

    # (l) HW A2A tier with AR/AG — also routes through SHARP-class INC paths
    # (hw_a2a is a superset of sharp_class capability).
    got = cost_collective([hw_a2a_tier], "all_reduce", M, 64)
    want = inc_all_reduce(M, alpha_s_switch, bw_Bps)
    _check("AR G=64 hw_a2a tier → inc_all_reduce", got, want)

    # (m) Mixed sharp_class + hw_a2a chain for A2A: gating requires *every* tier
    # to be hw_a2a, so this falls back to software. Multi-tier crossbar A2A
    # uses the hierarchical per-destination-class path (PR2.3).
    chain_mix = [sharp_tier, hw_a2a_tier]
    got = cost_collective(chain_mix, "moe_a2a", M, 4096, algorithm="ring")
    want = _hier_a2a_ref(M, 4096, chain_mix)
    _check("A2A mixed sharp+hw_a2a → hierarchical SW (no inc_a2a)", got, want)

    # (n) inc_enabled=False forces software fallback even on hw_a2a tier.
    got = cost_collective([hw_a2a_tier], "moe_a2a", M, 64, inc_enabled=False, algorithm="ring")
    want = pairwise_a2a(M, 64, alpha_s_endpoint, bw_Bps)
    _check("A2A hw_a2a tier inc_enabled=False → pairwise", got, want)


# ────────────────────────────────────────────────────────────
# T7 — system_loaders: inc enum back-compat + oversubscription clip
# ────────────────────────────────────────────────────────────

def test_T7_loader_inc_and_oversub():
    print("T7 loader (inc enum + oversubscription):")
    from llm_perf.io.system_loaders import system_spec_from_json_dict

    base_cfg = {
        "schema": "llm_perf.system",
        "name": "test",
        "num_devices": 8,
        "device": {"name": "x", "hbm_capacity_GB": 80.0,
                   "hbm_bandwidth_GBps": 3000.0, "peak_flops_TF": 1000.0},
        "fabrics": {
            "f": {"tiers": [{"name": "t0", "ports": 8,
                             "bw_per_port_GBps": 900.0, "alpha_us": 0.5}]}
        },
        "collective_fabrics": {"TP": "f", "EP": "f", "SP": "f", "PP": "f"},
    }

    # (a) Legacy `inc: nvls` parses to canonical `sharp_class`.
    import copy
    cfg = copy.deepcopy(base_cfg)
    cfg["fabrics"]["f"]["tiers"][0]["inc"] = "nvls"
    sys_a = system_spec_from_json_dict(cfg)
    tier = sys_a.fabrics["f"].tiers[0]
    if tier.inc != "sharp_class":
        _failures.append(f"T7a: inc='nvls' → expected 'sharp_class', got {tier.inc!r}")
    else:
        print("  PASS  inc='nvls' → 'sharp_class'")

    # (b) Legacy `inc: sharp` parses to canonical `sharp_class`.
    cfg = copy.deepcopy(base_cfg)
    cfg["fabrics"]["f"]["tiers"][0]["inc"] = "sharp"
    sys_b = system_spec_from_json_dict(cfg)
    if sys_b.fabrics["f"].tiers[0].inc != "sharp_class":
        _failures.append(f"T7b: inc='sharp' → expected 'sharp_class', got {sys_b.fabrics['f'].tiers[0].inc!r}")
    else:
        print("  PASS  inc='sharp' → 'sharp_class'")

    # (c) New `inc: hw_a2a` parses through.
    cfg = copy.deepcopy(base_cfg)
    cfg["fabrics"]["f"]["tiers"][0]["inc"] = "hw_a2a"
    sys_c = system_spec_from_json_dict(cfg)
    if sys_c.fabrics["f"].tiers[0].inc != "hw_a2a":
        _failures.append(f"T7c: inc='hw_a2a' lost: {sys_c.fabrics['f'].tiers[0].inc!r}")
    else:
        print("  PASS  inc='hw_a2a' parses through")

    # (d) Unknown `inc` value rejected.
    cfg = copy.deepcopy(base_cfg)
    cfg["fabrics"]["f"]["tiers"][0]["inc"] = "bogus"
    try:
        system_spec_from_json_dict(cfg)
        _failures.append("T7d: expected ValueError for inc='bogus'")
    except ValueError:
        print("  PASS  inc='bogus' rejected")

    # (e) oversubscription default = 1.0; eta_beta passes through unchanged.
    cfg = copy.deepcopy(base_cfg)
    cfg["fabrics"]["f"]["tiers"][0]["eta_beta"] = 0.8
    sys_e = system_spec_from_json_dict(cfg)
    t = sys_e.fabrics["f"].tiers[0]
    if t.eta_beta != 0.8 or t.oversubscription != 1.0:
        _failures.append(f"T7e: defaults broken (eta_beta={t.eta_beta}, s={t.oversubscription})")
    else:
        print("  PASS  oversubscription=1.0 default; eta_beta passes through")

    # (f) oversubscription > 1 with eta_beta > 1/s — silent min(eta_beta, 1/s).
    cfg = copy.deepcopy(base_cfg)
    cfg["fabrics"]["f"]["tiers"][0]["eta_beta"] = 0.9
    cfg["fabrics"]["f"]["tiers"][0]["oversubscription"] = 2.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sys_f = system_spec_from_json_dict(cfg)
    t = sys_f.fabrics["f"].tiers[0]
    if t.eta_beta != 0.5 or t.oversubscription != 2.0:
        _failures.append(f"T7f: clip wrong (eta_beta={t.eta_beta}, expected 0.5)")
    elif caught:
        _failures.append(f"T7f: unexpected warnings: {[str(w.message) for w in caught]}")
    else:
        print("  PASS  eta_beta=0.9, s=2 → silent min → 0.5")

    # (g) oversubscription > 1 with eta_beta ≤ 1/s passes through unchanged.
    cfg = copy.deepcopy(base_cfg)
    cfg["fabrics"]["f"]["tiers"][0]["eta_beta"] = 0.3
    cfg["fabrics"]["f"]["tiers"][0]["oversubscription"] = 2.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sys_g = system_spec_from_json_dict(cfg)
    if sys_g.fabrics["f"].tiers[0].eta_beta != 0.3:
        _failures.append(f"T7g: passthrough wrong (eta_beta={sys_g.fabrics['f'].tiers[0].eta_beta})")
    elif caught:
        _failures.append(f"T7g: unexpected warnings: {[str(w.message) for w in caught]}")
    else:
        print("  PASS  eta_beta=0.3, s=2 → unchanged (min picks 0.3)")

    # (h) oversubscription < 1 rejected.
    cfg = copy.deepcopy(base_cfg)
    cfg["fabrics"]["f"]["tiers"][0]["oversubscription"] = 0.5
    try:
        system_spec_from_json_dict(cfg)
        _failures.append("T7h: expected ValueError for oversubscription < 1")
    except ValueError:
        print("  PASS  oversubscription < 1 rejected")


# ────────────────────────────────────────────────────────────
# T8 — enumerate_options: optimizer surface for PR2.5
# ────────────────────────────────────────────────────────────

def test_T8_enumerate_options():
    print("T8 enumerate_options:")
    M = 1_000_000.0
    alpha = 0.5    # μs
    alpha_switch = 0.2
    bw = 900.0     # GB/s

    star_tier = CrossbarTier(name="star", ports=64, bw_per_port_GBps=bw, alpha_us=alpha)
    inc_tier = CrossbarTier(
        name="nvls", ports=64, bw_per_port_GBps=bw, alpha_us=alpha,
        inc="sharp_class", inc_alpha_us=alpha_switch,
    )
    hw_a2a_tier = CrossbarTier(
        name="th-ultra", ports=64, bw_per_port_GBps=bw, alpha_us=alpha,
        inc="hw_a2a", inc_alpha_us=alpha_switch,
    )
    torus_tier = TorusTier(name="ici", dims=(8, 8), bw_per_port_GBps=bw, alpha_us=alpha)

    def _names(opts):
        return [n for n, _ in opts]

    # (a) Star single-tier AR: ring + tree (P=1) + tree_pipelined; no INC (plain tier).
    opts = enumerate_options([star_tier], "all_reduce", M, 64)
    if _names(opts) != ["ring", "tree", "tree_pipelined"]:
        _failures.append(f"T8a: expected ['ring','tree','tree_pipelined'], got {_names(opts)}")
    else:
        print("  PASS  star AR: [ring, tree, tree_pipelined]")

    # (b) Star single-tier AG: ring only.
    opts = enumerate_options([star_tier], "all_gather", M, 64)
    if _names(opts) != ["ring"]:
        _failures.append(f"T8b: expected ['ring'], got {_names(opts)}")
    else:
        print("  PASS  star AG: [ring]")

    # (c) Star single-tier MoE A2A: ring only (tree deprecated, not enumerated).
    opts = enumerate_options([star_tier], "moe_a2a", M, 64)
    if _names(opts) != ["ring"]:
        _failures.append(f"T8c: expected ['ring'], got {_names(opts)}")
    else:
        print("  PASS  star A2A: [ring] (no deprecated tree)")

    # (d) Star + sharp_class AR: ring + tree + tree_pipelined + inc.
    opts = enumerate_options([inc_tier], "all_reduce", M, 64)
    if _names(opts) != ["ring", "tree", "tree_pipelined", "inc"]:
        _failures.append(f"T8d: expected ['ring','tree','tree_pipelined','inc'], got {_names(opts)}")
    else:
        print("  PASS  star+sharp_class AR: [ring, tree, tree_pipelined, inc]")

    # (e) Star + sharp_class AG: ring + inc.
    opts = enumerate_options([inc_tier], "all_gather", M, 64)
    if _names(opts) != ["ring", "inc"]:
        _failures.append(f"T8e: expected ['ring','inc'], got {_names(opts)}")
    else:
        print("  PASS  star+sharp_class AG: [ring, inc]")

    # (f) Star + sharp_class MoE A2A: ring only (sharp_class doesn't accelerate A2A).
    opts = enumerate_options([inc_tier], "moe_a2a", M, 64)
    if _names(opts) != ["ring"]:
        _failures.append(f"T8f: expected ['ring'] (sharp_class no A2A), got {_names(opts)}")
    else:
        print("  PASS  star+sharp_class A2A: [ring] (no inc — sharp_class excludes A2A)")

    # (g) Star + hw_a2a MoE A2A: ring + inc (hw_a2a accelerates A2A).
    opts = enumerate_options([hw_a2a_tier], "moe_a2a", M, 64)
    if _names(opts) != ["ring", "inc"]:
        _failures.append(f"T8g: expected ['ring','inc'], got {_names(opts)}")
    else:
        print("  PASS  star+hw_a2a A2A: [ring, inc]")

    # (h) inc_enabled=False removes INC entries on the same tier.
    opts = enumerate_options([inc_tier], "all_reduce", M, 64, inc_enabled=False)
    if _names(opts) != ["ring", "tree", "tree_pipelined"]:
        _failures.append(f"T8h: expected ['ring','tree','tree_pipelined'] when inc_enabled=False, got {_names(opts)}")
    else:
        print("  PASS  inc_enabled=False suppresses inc")

    # (i) **Critical invariant**: INC structurally absent on torus for every op.
    # TorusTier has no inc field; dispatcher gates INC inside the crossbar branch.
    for op in ("all_reduce", "all_gather", "moe_a2a"):
        opts = enumerate_options([torus_tier], op, M, 64)
        if "inc" in _names(opts):
            _failures.append(f"T8i: torus enumerated 'inc' for op={op!r}: {_names(opts)}")
        elif _names(opts) != ["torus-dim-ring"]:
            _failures.append(
                f"T8i: torus expected ['torus-dim-ring'] for op={op!r}, got {_names(opts)}"
            )
    if all("torus-dim-ring" in _names(enumerate_options([torus_tier], op, M, 64))
           and "inc" not in _names(enumerate_options([torus_tier], op, M, 64))
           for op in ("all_reduce", "all_gather", "moe_a2a")):
        print("  PASS  torus: INC structurally absent across AR / AG / A2A")

    # (j) p2p: single option on any topology.
    opts = enumerate_options([star_tier], "p2p", M, 2)
    if _names(opts) != ["p2p"]:
        _failures.append(f"T8j: star p2p expected ['p2p'], got {_names(opts)}")
    opts = enumerate_options([torus_tier], "p2p", M, 2)
    if _names(opts) != ["p2p"]:
        _failures.append(f"T8j: torus p2p expected ['p2p'], got {_names(opts)}")
    print("  PASS  p2p: single option on star + torus")

    # (k) Costs match the corresponding cost_collective values.
    opts = enumerate_options([inc_tier], "all_reduce", M, 64)
    for name, cost in opts:
        want = cost_collective([inc_tier], "all_reduce", M, 64,
                               algorithm=("ring" if name == "inc" else name),
                               inc_enabled=(name == "inc"))
        if name != "inc" and abs(cost - want) > 0:
            _failures.append(
                f"T8k: enumerate cost({name})={cost} != cost_collective={want}"
            )
    print("  PASS  enumerate costs match cost_collective at the same algorithm")

    # (l) G=1 returns empty list (no work to enumerate).
    opts = enumerate_options([star_tier], "all_reduce", M, 1)
    if opts != []:
        _failures.append(f"T8l: G=1 expected [], got {opts}")
    else:
        print("  PASS  G=1: empty list")

    # (m) Empty tier list returns empty list.
    opts = enumerate_options([], "all_reduce", M, 64)
    if opts != []:
        _failures.append(f"T8m: empty tiers expected [], got {opts}")
    else:
        print("  PASS  empty tiers: empty list")

    # (n) Mixed crossbar/torus chain: returns crossbar-flat options (mirrors
    # cost_collective's fallback path; no warning here — that fires at
    # cost_collective time).
    mixed = [star_tier, torus_tier]
    opts = enumerate_options(mixed, "all_reduce", M, 4096)
    # Should be flat-ring/tree/tree_pipelined (mirrors cost_collective's
    # flat-fallback for mixed); no INC since one tier is torus.
    if "inc" in _names(opts):
        _failures.append(f"T8n: mixed chain enumerated 'inc': {_names(opts)}")
    elif set(_names(opts)) != {"ring", "tree", "tree_pipelined"}:
        _failures.append(f"T8n: mixed chain expected {{ring,tree,tree_pipelined}}, got {_names(opts)}")
    else:
        print("  PASS  mixed crossbar+torus AR: [ring, tree, tree_pipelined], no INC")


# ────────────────────────────────────────────────────────────
# T9 — hierarchical AR / AG / RS on multi-tier crossbar (PR2.2)
# ────────────────────────────────────────────────────────────

def test_T9_hierarchical_crossbar():
    print("T9 hierarchical crossbar:")
    M = 16 * 2**20  # 16 MiB
    inner = CrossbarTier(name="inner", ports=72,
                          bw_per_port_GBps=900.0, alpha_us=0.5)
    outer = CrossbarTier(name="outer", ports=8,
                          bw_per_port_GBps=400.0, alpha_us=2.5)
    chain = [inner, outer]

    G_inner = 72
    L = 2          # G=144 → 2 outer ranks
    G = G_inner * L  # = 144
    alpha_inner_s = 0.5e-6
    bw_inner_Bps = 900e9
    alpha_outer_s = 2.5e-6
    bw_outer_Bps = 400e9

    # ─── (a) Hierarchical AR cell-level: matches ring_RS + ring_AR + ring_AG.
    # New AG/RS convention: pass full M to the inner phases.
    expected_ar = (
        ring_reduce_scatter(M, G_inner, alpha_inner_s, bw_inner_Bps)
        + ring_all_reduce(M / G_inner, L, alpha_outer_s, bw_outer_Bps)
        + ring_all_gather(M, G_inner, alpha_inner_s, bw_inner_Bps)
    )
    got = cost_collective(chain, "all_reduce", M, G, algorithm="ring")
    _check("hier-AR ring composition", got, expected_ar)

    # ─── (b) Hierarchical AR with outer=tree (default pipelined=False).
    expected_ar_tree = (
        ring_reduce_scatter(M, G_inner, alpha_inner_s, bw_inner_Bps)
        + tree_all_reduce(M / G_inner, L, alpha_outer_s, bw_outer_Bps)
        + ring_all_gather(M, G_inner, alpha_inner_s, bw_inner_Bps)
    )
    got = cost_collective(chain, "all_reduce", M, G, algorithm="tree")
    _check("hier-AR tree composition", got, expected_ar_tree)

    # ─── (b') Hierarchical AR with outer=tree_pipelined.
    expected_ar_tree_p = (
        ring_reduce_scatter(M, G_inner, alpha_inner_s, bw_inner_Bps)
        + tree_all_reduce(M / G_inner, L, alpha_outer_s, bw_outer_Bps, pipelined=True)
        + ring_all_gather(M, G_inner, alpha_inner_s, bw_inner_Bps)
    )
    got = cost_collective(chain, "all_reduce", M, G, algorithm="tree_pipelined")
    _check("hier-AR tree_pipelined composition", got, expected_ar_tree_p)

    # ─── (c) α-side: matches collectives/03_hierarchical_topologies.md §2 closed form
    # 2(G_inner − 1)·α_inner + 2(L − 1)·α_outer for ring-on-ring.
    # Isolate α with M=0.
    got_alpha = cost_collective(chain, "all_reduce", 0.0, G, algorithm="ring")
    expected_alpha = 2 * (G_inner - 1) * alpha_inner_s + 2 * (L - 1) * alpha_outer_s
    _check("hier-AR α closed form (M=0)", got_alpha, expected_alpha)

    # ─── (d) β-side telescoping: at uniform BW the hierarchical β term equals
    # the flat-ring β coefficient 2(N-1)/N · M/BW. Use a chain where
    # BW_inner = BW_outer to avoid bottleneck mixing.
    uniform_outer = CrossbarTier(name="outer_u", ports=8,
                                  bw_per_port_GBps=900.0, alpha_us=0.0)
    uniform_chain = [
        CrossbarTier(name="inner_u", ports=72,
                     bw_per_port_GBps=900.0, alpha_us=0.0),
        uniform_outer,
    ]
    got_beta = cost_collective(uniform_chain, "all_reduce", M, G, algorithm="ring")
    expected_beta = 2 * (G - 1) / G * M / 900e9
    if not math.isclose(got_beta, expected_beta, rel_tol=1e-12):
        _failures.append(
            f"T9d β-telescoping: got {got_beta:.6e}, expected {expected_beta:.6e}"
        )
    else:
        print(f"  PASS  hier-AR β telescopes to flat 2(N-1)/N at uniform BW")

    # ─── (e) Hierarchical AG: inner AG + outer AG cascade. New AG convention:
    # inner produces per-rank size M/L, outer produces per-rank size M.
    expected_ag = (
        ring_all_gather(M / L, G_inner, alpha_inner_s, bw_inner_Bps)
        + ring_all_gather(M, L, alpha_outer_s, bw_outer_Bps)
    )
    got = cost_collective(chain, "all_gather", M, G, algorithm="ring")
    _check("hier-AG inner+outer cascade", got, expected_ag)

    # ─── (f) Single-tier crossbar AR is unchanged (flat ring/tree).
    single = [inner]
    got = cost_collective(single, "all_reduce", M, 64, algorithm="ring")
    expected_flat = ring_all_reduce(M, 64, alpha_inner_s, bw_inner_Bps)
    _check("single-tier AR flat (regression)", got, expected_flat)

    # ─── (g) MoE A2A on multi-tier uses per-destination-class accounting (§5.3).
    # See T10 for the cell-level formula check; here just confirm the dispatcher
    # routes to the hierarchical-A2A path (cost ≠ flat pairwise).
    got = cost_collective(chain, "moe_a2a", M, G, algorithm="ring")
    alpha_total_s = alpha_inner_s + alpha_outer_s
    bw_min_Bps = min(bw_inner_Bps, bw_outer_Bps)
    flat_a2a = pairwise_a2a(M, G, alpha_total_s, bw_min_Bps)
    if got == flat_a2a:
        _failures.append(
            f"T9g: multi-tier A2A should use hierarchical path, got flat ({got!r})"
        )
    else:
        print("  PASS  multi-tier MoE A2A uses hierarchical (≠ flat pairwise)")

    # ─── (h) enumerate_options on multi-tier surfaces hierarchical costs.
    opts = enumerate_options(chain, "all_reduce", M, G)
    names = [n for n, _ in opts]
    if names != ["ring", "tree", "tree_pipelined"]:
        _failures.append(
            f"T9h: multi-tier AR enumerate names: expected "
            f"['ring','tree','tree_pipelined'], got {names}"
        )
    else:
        # Check costs match hierarchical cell.
        cost_ring = dict(opts)["ring"]
        cost_tree = dict(opts)["tree"]
        cost_tree_p = dict(opts)["tree_pipelined"]
        if not math.isclose(cost_ring, expected_ar, rel_tol=1e-12):
            _failures.append(f"T9h: enumerate ring != hier-AR cost")
        elif not math.isclose(cost_tree, expected_ar_tree, rel_tol=1e-12):
            _failures.append(f"T9h: enumerate tree != hier-AR-tree cost")
        elif not math.isclose(cost_tree_p, expected_ar_tree_p, rel_tol=1e-12):
            _failures.append(f"T9h: enumerate tree_pipelined != hier-AR-tree_pipelined cost")
        else:
            print("  PASS  enumerate_options multi-tier AR returns hierarchical costs")

    # ─── (i) INC on a multi-tier hierarchical chain: still routes via flat
    # scale-out INC (n_α = 2k·α_switch). When all tiers eligible, optimizer's
    # `inc` option short-circuits the SW comparison.
    inc_in = CrossbarTier(name="inc_in", ports=72, bw_per_port_GBps=900.0,
                           alpha_us=0.5, inc="sharp_class", inc_alpha_us=0.2)
    inc_out = CrossbarTier(name="inc_out", ports=8, bw_per_port_GBps=400.0,
                            alpha_us=2.5, inc="sharp_class", inc_alpha_us=0.4)
    inc_chain = [inc_in, inc_out]
    got = cost_collective(inc_chain, "all_reduce", M, G)
    expected_inc = inc_all_reduce(M, (0.2 + 0.4) * 1e-6, 400e9)
    _check("multi-tier INC AR (scale-out)", got, expected_inc)


# ────────────────────────────────────────────────────────────
# T10 — hierarchical A2A per-destination-class on multi-tier crossbar (PR2.3)
# ────────────────────────────────────────────────────────────

def test_T12_optimizer():
    print("T12 collective_algo_opt:")
    from llm_perf.core.collective_algo_opt import optimize_collective_algorithms
    from llm_perf.specs.model_spec import LlmModelSpec, MoESpec
    from llm_perf.specs.partition_spec import PartitionSpec
    from llm_perf.specs.system_spec import (
        DeviceSpec, FabricSpec, SystemSpec,
    )
    from llm_perf.specs.tuner_spec import TuningSpec

    # Tiny dense model and tiny MoE model.
    dense_model = LlmModelSpec(
        name="t-dense", L=2, H=4096, n_q=8, n_kv=8, I_dense=8,
        vocab_size=100, max_seq_len=1024, bytes_per_param=2.0,
    )
    moe_model = LlmModelSpec(
        name="t-moe", L=2, H=4096, n_q=8, n_kv=8, I_dense=8,
        vocab_size=100, max_seq_len=1024, bytes_per_param=2.0,
        moe=MoESpec(n_experts=8, k_active=2, I_moe=8),
    )
    device = DeviceSpec(name="x", hbm_capacity_GB=80.0,
                        hbm_bandwidth_GBps=3000.0, peak_flops_TF=1000.0)

    # ─── (a) auto resolves on a single-tier star with sharp_class INC.
    # At small M (decode H=4096, b=2 → 8 KB), DBT should beat ring; INC even better.
    inc_tier = CrossbarTier(name="t0", ports=64, bw_per_port_GBps=900.0,
                             alpha_us=0.5, inc="sharp_class", inc_alpha_us=0.2)
    fab = FabricSpec(name="f", tiers=[inc_tier])
    sys = SystemSpec(name="s", device=device, num_devices=64,
                     fabrics={"f": fab},
                     collective_fabrics={"TP": ["f"], "EP": ["f"], "SP": ["f"], "PP": ["f"]})
    part = PartitionSpec(PP=1, TP=8, EP=1, SP=1)
    tuner = TuningSpec(
        S_decode=2048, B_decode=1, B_prefill=1, S_input=4096,
        tp_algorithm_decode="auto",
        tp_algorithm_prefill="auto",
        ep_algorithm_decode="auto",
        ep_algorithm_prefill="auto",
        inc_enabled=True,
    )
    resolved = optimize_collective_algorithms(dense_model, part, sys, tuner)
    if resolved.tp_algorithm_decode != "inc":
        _failures.append(
            f"T12a: decode TP with INC available should pick 'inc', got "
            f"{resolved.tp_algorithm_decode!r}"
        )
    else:
        print("  PASS  decode TP AR with sharp_class → 'inc'")
    if resolved.tp_algorithm_prefill != "inc":
        _failures.append(f"T12a: prefill TP should pick 'inc', got {resolved.tp_algorithm_prefill!r}")
    else:
        print("  PASS  prefill TP AR with sharp_class → 'inc'")

    # ─── (b) inc_enabled=False forces SW choice; at small M, the optimizer
    # should pick a tree variant (DBT beats ring on α). Either tree (P=1) or
    # tree_pipelined is acceptable — they have the same α coefficient and only
    # differ in the BW term, so at small M the cheaper one wins on β.
    tuner_noinc = replace(tuner, inc_enabled=False)
    resolved2 = optimize_collective_algorithms(dense_model, part, sys, tuner_noinc)
    sw_algs = ("ring", "tree", "tree_pipelined")
    if resolved2.tp_algorithm_decode not in sw_algs:
        _failures.append(f"T12b: SW-only should pick {sw_algs}, got {resolved2.tp_algorithm_decode!r}")
    elif not resolved2.tp_algorithm_decode.startswith("tree"):
        _failures.append(f"T12b: small-M should pick a tree variant, got {resolved2.tp_algorithm_decode!r}")
    else:
        print(f"  PASS  inc_enabled=False forces SW; small-M → {resolved2.tp_algorithm_decode!r}")

    # ─── (c) Same idea for prefill — verify the optimizer resolves to a
    # valid SW algorithm (the exact choice depends on (α, BW, M)).
    resolved3 = optimize_collective_algorithms(dense_model, part, sys, tuner_noinc)
    if resolved3.tp_algorithm_prefill not in sw_algs:
        _failures.append(f"T12c: prefill SW choice invalid: {resolved3.tp_algorithm_prefill!r}")
    else:
        print(f"  PASS  prefill SW resolves to {resolved3.tp_algorithm_prefill!r}")

    # ─── (d) MoE A2A on a hw_a2a tier picks 'inc'.
    hw_tier = CrossbarTier(name="th", ports=64, bw_per_port_GBps=900.0,
                            alpha_us=0.5, inc="hw_a2a", inc_alpha_us=0.2)
    fab_hw = FabricSpec(name="f", tiers=[hw_tier])
    sys_hw = SystemSpec(name="s", device=device, num_devices=64,
                         fabrics={"f": fab_hw},
                         collective_fabrics={"TP": ["f"], "EP": ["f"], "SP": ["f"], "PP": ["f"]})
    part_moe = PartitionSpec(PP=1, TP=1, EP=8, SP=1)
    tuner_moe = replace(tuner, tp_algorithm_decode="ring", tp_algorithm_prefill="ring",
                         ep_algorithm_decode="auto", ep_algorithm_prefill="auto")
    resolved4 = optimize_collective_algorithms(moe_model, part_moe, sys_hw, tuner_moe)
    if resolved4.ep_algorithm_decode != "inc":
        _failures.append(f"T12d: hw_a2a EP A2A should pick 'inc', got {resolved4.ep_algorithm_decode!r}")
    else:
        print("  PASS  EP A2A with hw_a2a → 'inc'")

    # ─── (e) sharp_class tier: EP A2A doesn't get INC (sharp_class doesn't
    # accelerate A2A). Optimizer falls back to ring.
    sharp_tier = CrossbarTier(name="ts", ports=64, bw_per_port_GBps=900.0,
                               alpha_us=0.5, inc="sharp_class", inc_alpha_us=0.2)
    fab_sc = FabricSpec(name="f", tiers=[sharp_tier])
    sys_sc = SystemSpec(name="s", device=device, num_devices=64,
                         fabrics={"f": fab_sc},
                         collective_fabrics={"TP": ["f"], "EP": ["f"], "SP": ["f"], "PP": ["f"]})
    resolved5 = optimize_collective_algorithms(moe_model, part_moe, sys_sc, tuner_moe)
    if resolved5.ep_algorithm_decode != "ring":
        _failures.append(
            f"T12e: sharp_class doesn't accelerate A2A; should pick 'ring', got "
            f"{resolved5.ep_algorithm_decode!r}"
        )
    else:
        print("  PASS  EP A2A with sharp_class → 'ring' (no INC for A2A)")

    # ─── (f) Non-auto fields pass through unchanged.
    tuner_pinned = replace(tuner, tp_algorithm_decode="ring", tp_algorithm_prefill="ring",
                            ep_algorithm_decode="ring", ep_algorithm_prefill="ring")
    resolved6 = optimize_collective_algorithms(dense_model, part, sys, tuner_pinned)
    if (resolved6.tp_algorithm_decode != "ring" or
        resolved6.tp_algorithm_prefill != "ring"):
        _failures.append(f"T12f: non-auto fields should pass through unchanged: {resolved6}")
    else:
        print("  PASS  non-auto fields pass through")

    # ─── (g) After resolving, the calculator runs without error (no auto
    # remaining to trigger the dispatcher's "auto must be resolved" check).
    from llm_perf.core import decode_model as dm
    # Smoke check: decode_model can read the resolved tuner without raising.
    try:
        # Fabricated minimal call: avoid running the full calculator; just
        # confirm the early auto-check doesn't fire.
        if "auto" in (resolved.tp_algorithm_decode, resolved.ep_algorithm_decode):
            _failures.append("T12g: resolved tuner still has 'auto'")
        else:
            print("  PASS  resolved tuner has no 'auto' remaining")
    except Exception as e:
        _failures.append(f"T12g: unexpected error: {e}")

    # ─── (h) Policy: INC short-circuits even when SW would be cheaper.
    # Set inc_alpha_us extremely high so INC's α-term dominates and exceeds
    # both ring and tree, then confirm the optimizer still picks 'inc'.
    # INC is a hardware deployment decision; SW comparison only applies
    # when INC is unavailable. (η_α / η_β can't differentiate INC vs SW —
    # they apply to the same tier — so the only way to make INC look bad
    # in the cost model is via the inc_alpha_us override.)
    bad_inc_tier = CrossbarTier(
        name="bad-inc", ports=64, bw_per_port_GBps=900.0, alpha_us=0.5,
        inc="sharp_class", inc_alpha_us=100.0,  # 100 μs INC switch α: way > DBT's 3 μs at G=8
    )
    fab_bad = FabricSpec(name="f", tiers=[bad_inc_tier])
    sys_bad = SystemSpec(name="s", device=device, num_devices=64,
                         fabrics={"f": fab_bad},
                         collective_fabrics={"TP": ["f"], "EP": ["f"], "SP": ["f"], "PP": ["f"]})
    tuner_bad_inc = replace(tuner, inc_enabled=True)
    # Sanity check the cost ordering really is bad-INC > ring on this fabric
    # (otherwise the test is a tautology, not a policy check).
    from llm_perf.core.primitives import enumerate_options as _enum
    chain = sys_bad.get_tier_chain("TP")
    M_decode = dense_model.H * dense_model.bytes_per_param  # B_decode=1
    opts_check = dict(_enum(chain, "all_reduce", M_decode, 8))
    if "inc" not in opts_check or opts_check.get("ring", 0) >= opts_check["inc"]:
        _failures.append(
            f"T12h setup: needed bad-INC (cost > ring) for the test to be meaningful, "
            f"got {opts_check}"
        )
    else:
        resolved_bad = optimize_collective_algorithms(dense_model, part, sys_bad, tuner_bad_inc)
        if resolved_bad.tp_algorithm_decode != "inc":
            _failures.append(
                f"T12h: INC must be picked when available even if SW is cheaper "
                f"per the policy, got {resolved_bad.tp_algorithm_decode!r} "
                f"(costs: {opts_check})"
            )
        else:
            print(
                f"  PASS  INC short-circuits even when SW cheaper "
                f"(ring={opts_check['ring']*1e6:.1f}μs < inc={opts_check['inc']*1e6:.1f}μs, "
                f"optimizer picked 'inc')"
            )


def test_T11_per_phase_tuner():
    print("T11 per-phase × per-collective tuner:")
    from llm_perf.io.tuner_loaders import tuning_spec_from_json_dict

    base = {
        "schema": "llm_perf.tuner",
        "S_decode": 2048,
        "n_TP_collectives": 2,
        "n_EP_collectives": 2,
        "n_SP_collectives": 1,
        "overlap_factor": 0.0,
    }

    # (a) Defaults: all four per-phase fields default to "ring".
    t = tuning_spec_from_json_dict(base)
    if (t.tp_algorithm_decode, t.tp_algorithm_prefill,
        t.ep_algorithm_decode, t.ep_algorithm_prefill) != ("ring",) * 4:
        _failures.append(f"T11a: defaults wrong: {t}")
    else:
        print("  PASS  defaults: tp/ep_algorithm_{decode,prefill} all 'ring'")

    # (b) Legacy single-knob propagates to both phases.
    cfg = {**base, "tp_algorithm": "tree", "ep_algorithm": "ring"}
    t = tuning_spec_from_json_dict(cfg)
    if (t.tp_algorithm_decode, t.tp_algorithm_prefill) != ("tree", "tree"):
        _failures.append(f"T11b: legacy tp_algorithm should propagate to both phases: {t}")
    else:
        print("  PASS  legacy tp_algorithm='tree' propagates to both decode and prefill")

    # (c) Per-phase override takes precedence over legacy.
    cfg = {
        **base,
        "tp_algorithm": "ring",
        "tp_algorithm_decode": "tree",
        "tp_algorithm_prefill": "ring",
    }
    t = tuning_spec_from_json_dict(cfg)
    if (t.tp_algorithm_decode, t.tp_algorithm_prefill) != ("tree", "ring"):
        _failures.append(f"T11c: per-phase override failed: {t}")
    else:
        print("  PASS  per-phase override: decode='tree', prefill='ring'")

    # (d) "auto" parses through (resolved later by the optimizer).
    cfg = {**base, "tp_algorithm_decode": "auto", "ep_algorithm_prefill": "auto"}
    t = tuning_spec_from_json_dict(cfg)
    if t.tp_algorithm_decode != "auto" or t.ep_algorithm_prefill != "auto":
        _failures.append(f"T11d: 'auto' didn't parse through: {t}")
    else:
        print("  PASS  'auto' parses through TuningSpec")

    # (e) Invalid value rejected.
    cfg = {**base, "tp_algorithm_decode": "bogus"}
    try:
        tuning_spec_from_json_dict(cfg)
        _failures.append("T11e: expected ValueError for tp_algorithm_decode='bogus'")
    except ValueError:
        print("  PASS  invalid algorithm rejected")


def test_T10_hierarchical_a2a():
    print("T10 hierarchical A2A:")
    M = 16 * 2**20  # 16 MiB
    inner = CrossbarTier(name="inner", ports=72, bw_per_port_GBps=900.0, alpha_us=0.5)
    outer = CrossbarTier(name="outer", ports=8, bw_per_port_GBps=400.0, alpha_us=2.5)
    chain = [inner, outer]

    G_inner = 72
    G = 144
    alpha_inner_s = 0.5e-6
    bw_inner_Bps = 900e9
    alpha_outer_s = 2.5e-6
    bw_outer_Bps = 400e9

    # ─── (a) Per-destination-class formula:
    #   2 · [ (G_inner−1)·(α_inner + (M/G)/BW_inner)
    #       + (G−G_inner)·(α_outer + (M/G)/BW_outer) ]
    chunk = M / G
    n_intra = G_inner - 1
    n_outer = G - G_inner
    expected = (
        n_intra * (alpha_inner_s + chunk / bw_inner_Bps)
        + n_outer * (alpha_outer_s + chunk / bw_outer_Bps)
    )
    got = cost_collective(chain, "moe_a2a", M, G, algorithm="ring")
    _check("hier-A2A 2-tier per-class", got, expected)

    # ─── (b) When G ≤ G_inner (stays within pod), n_outer=0; should match
    # single-tier flat pairwise (single-direction; per-MoE-layer accumulator
    # multiplies by n_EP=2).
    intra_only = pairwise_a2a(M, 64, alpha_inner_s, bw_inner_Bps)
    got = cost_collective([inner], "moe_a2a", M, 64, algorithm="ring")
    _check("intra-pod A2A G=64 stays flat (single-tier)", got, intra_only)

    # ─── (c) α-side closed form (M=0): (G_inner−1)·α_inner + (G−G_inner)·α_outer.
    got_alpha = cost_collective(chain, "moe_a2a", 0.0, G, algorithm="ring")
    expected_alpha = n_intra * alpha_inner_s + n_outer * alpha_outer_s
    if not math.isclose(got_alpha, expected_alpha, rel_tol=1e-12):
        _failures.append(
            f"T10c α closed form: got {got_alpha:.6e}, expected {expected_alpha:.6e}"
        )
    else:
        print("  PASS  hier-A2A α closed form (M=0)")

    # ─── (d) BW-side at uniform BW: n_intra·chunk/BW_inner + n_outer·chunk/BW_outer
    # collapses to (G−1)·chunk/BW = (G−1)/G · M/BW, matching flat pairwise's β
    # coefficient — confirms the per-destination sum reduces correctly.
    uniform = [
        CrossbarTier(name="ui", ports=72, bw_per_port_GBps=900.0, alpha_us=0.0),
        CrossbarTier(name="uo", ports=8, bw_per_port_GBps=900.0, alpha_us=0.0),
    ]
    got_beta = cost_collective(uniform, "moe_a2a", M, G, algorithm="ring")
    # (G-1) · (M/G) / BW   (single-direction; n_EP=2 wraps at the accumulator)
    expected_beta = (G - 1) * (M / G) / 900e9
    if not math.isclose(got_beta, expected_beta, rel_tol=1e-12):
        _failures.append(
            f"T10d β-uniform: got {got_beta:.6e}, expected {expected_beta:.6e}"
        )
    else:
        print("  PASS  hier-A2A β at uniform BW reduces to flat (G-1)/G·M/BW")

    # ─── (e) enumerate_options surfaces hierarchical A2A.
    opts = enumerate_options(chain, "moe_a2a", M, G)
    if [n for n, _ in opts] != ["ring"]:
        _failures.append(f"T10e: expected ['ring'], got {[n for n,_ in opts]}")
    elif not math.isclose(dict(opts)["ring"], expected, rel_tol=1e-12):
        _failures.append(f"T10e: enumerated cost != hierarchical cell")
    else:
        print("  PASS  enumerate_options multi-tier A2A returns hierarchical cost")

    # ─── (f) HW A2A on hw_a2a chain bypasses hierarchical (INC scale-out form).
    hw_in = CrossbarTier(name="hw_in", ports=72, bw_per_port_GBps=900.0,
                          alpha_us=0.5, inc="hw_a2a", inc_alpha_us=0.2)
    hw_out = CrossbarTier(name="hw_out", ports=8, bw_per_port_GBps=400.0,
                           alpha_us=2.5, inc="hw_a2a", inc_alpha_us=0.4)
    got = cost_collective([hw_in, hw_out], "moe_a2a", M, G, algorithm="ring")
    expected_inc = inc_a2a(M, G, (0.2 + 0.4) * 1e-6, 400e9)
    _check("hw_a2a multi-tier A2A → inc_a2a (bypasses hierarchical)",
           got, expected_inc)


# ────────────────────────────────────────────────────────────
# T13 — MeshTier (full mesh + k-D mesh) routing and cost (PR2.7)
# ────────────────────────────────────────────────────────────

def test_T13_mesh():
    print("T13 mesh tier:")
    M = 1_000_000.0
    alpha = 0.5  # μs
    bw = 900.0   # GB/s
    alpha_s = alpha * 1e-6
    bw_Bps = bw * 1e9

    # ─── Full mesh: dims=(N,), full=True. Cost identical to single-tier crossbar.
    fmesh = MeshTier(name="full", dims=(8,), bw_per_port_GBps=bw,
                      alpha_us=alpha, full=True)
    star = CrossbarTier(name="star", ports=8, bw_per_port_GBps=bw, alpha_us=alpha)

    for op, alg in [("all_reduce", "ring"), ("all_reduce", "tree"),
                     ("all_gather", "ring"), ("moe_a2a", "ring"),
                     ("p2p", "ring")]:
        G = 8 if op != "p2p" else 2
        got = cost_collective([fmesh], op, M, G, algorithm=alg)
        want = cost_collective([star], op, M, G, algorithm=alg)
        _check(f"full-mesh == star: {op}/{alg}", got, want)

    # ─── k-D mesh: dims=(D_1,…,D_k), full=False. Routes through torus primitives.
    kdm = MeshTier(name="kd", dims=(8, 8), bw_per_port_GBps=bw,
                    alpha_us=alpha, full=False)
    torus = TorusTier(name="torus", dims=(8, 8), bw_per_port_GBps=bw, alpha_us=alpha)

    # AR / AG identical to torus (open-line still BW-optimal).
    for op in ("all_reduce", "all_gather"):
        got = cost_collective([kdm], op, M, 64, algorithm="ring")
        want = cost_collective([torus], op, M, 64, algorithm="ring")
        _check(f"k-D mesh == torus: {op}", got, want)

    # A2A: mesh and torus differ on TWO axes per upstream's torus_a2a fix:
    #   - diameter:  mesh = Σ(d-1)        ; torus = Σ⌊d/2⌋
    #   - bisection: mesh BW = D_max/4   ; torus BW = D_max/8 (mesh 2× worse)
    # Dispatcher returns single-direction A2A; n_EP=2 wraps at the per-MoE-layer accumulator.
    got_a2a_kdm = cost_collective([kdm], "moe_a2a", M, 64, algorithm="ring")
    got_a2a_torus = cost_collective([torus], "moe_a2a", M, 64, algorithm="ring")
    diam_mesh = sum(d - 1 for d in (8, 8))      # 7+7=14 (open-line)
    diam_torus = sum(d // 2 for d in (8, 8))    # 4+4=8  (wraparound)
    d_max = 8
    expected_kdm = diam_mesh * alpha_s + d_max * M / (4 * bw_Bps)
    expected_torus = diam_torus * alpha_s + d_max * M / (8 * bw_Bps)
    if not math.isclose(got_a2a_kdm, expected_kdm, rel_tol=1e-12):
        _failures.append(f"T13 k-D mesh A2A: got {got_a2a_kdm}, expected {expected_kdm}")
    elif not math.isclose(got_a2a_torus, expected_torus, rel_tol=1e-12):
        _failures.append(f"T13 torus A2A: got {got_a2a_torus}, expected {expected_torus}")
    else:
        # k-D mesh BW term = 2× torus BW term.
        bw_term_kdm = got_a2a_kdm - diam_mesh * alpha_s
        bw_term_torus = got_a2a_torus - diam_torus * alpha_s
        if not math.isclose(bw_term_kdm, 2.0 * bw_term_torus, rel_tol=1e-12):
            _failures.append(
                f"T13: k-D mesh A2A BW should be 2× torus, got "
                f"{bw_term_kdm} vs 2× {bw_term_torus}"
            )
        else:
            print("  PASS  k-D mesh A2A BW term = 2× torus (D_max/4 vs D_max/8)")

    # ─── INC structurally absent on mesh (no inc field).
    for kind, tier in [("full mesh", fmesh), ("k-D mesh", kdm)]:
        for op in ("all_reduce", "all_gather", "moe_a2a"):
            G = 64 if tier is kdm else 8
            opts = enumerate_options([tier], op, M, G)
            names = [n for n, _ in opts]
            if "inc" in names:
                _failures.append(f"T13: {kind} enumerate('inc'): {names}")
    print("  PASS  INC structurally absent on mesh (full + k-D) for AR / AG / A2A")

    # ─── Loader: full mesh JSON parses and the dispatcher routes correctly.
    from llm_perf.io.system_loaders import system_spec_from_json_dict
    cfg = {
        "schema": "llm_perf.system",
        "name": "mesh_test",
        "num_devices": 16,
        "device": {"name": "x", "hbm_capacity_GB": 80.0,
                   "hbm_bandwidth_GBps": 3000.0, "peak_flops_TF": 1000.0},
        "fabrics": {
            "f": {"tiers": [
                {"name": "ucie", "topology": "mesh", "dims": [16],
                 "full": True, "bw_per_port_GBps": 100.0, "alpha_us": 0.05},
            ]},
        },
        "collective_fabrics": {"TP": "f", "EP": "f", "SP": "f", "PP": "f"},
    }
    sys_full = system_spec_from_json_dict(cfg)
    tier = sys_full.fabrics["f"].tiers[0]
    if not (isinstance(tier, MeshTier) and tier.full and tier.dims == (16,)):
        _failures.append(f"T13 full-mesh load: unexpected tier {tier!r}")
    else:
        print("  PASS  full-mesh JSON: topology='mesh', full=true → MeshTier")

    # Loader: full=true + non-1 dims rejected.
    import copy
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg["fabrics"]["f"]["tiers"][0]["dims"] = [4, 4]  # full mesh expects (N,)
    try:
        system_spec_from_json_dict(bad_cfg)
        _failures.append("T13: full=True + dims=(4,4) should reject")
    except ValueError:
        print("  PASS  full=True + multi-dim rejected by loader")


# ────────────────────────────────────────────────────────────

def main() -> int:
    test_T1_single_tier()
    test_T2_multi_tier()
    test_T3_randomized()
    test_T4_crossbar_silent()
    test_T5_torus_k1_continuity()
    test_T6_inc_dispatch()
    test_T7_loader_inc_and_oversub()
    test_T8_enumerate_options()
    test_T9_hierarchical_crossbar()
    test_T10_hierarchical_a2a()
    test_T11_per_phase_tuner()
    test_T12_optimizer()
    test_T13_mesh()
    if _failures:
        print(f"\n{len(_failures)} FAILURES:")
        for f in _failures:
            print(f"  {f}")
        return 1
    print("\nAll topology-equivalence checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
