"""Unit tests for the nested-layout helper and tier-aware PP cost.

Covers:
  (a) assign_tier_per_axis on a 3-tier system (d-Matrix squadrack):
      - PP fits in tier 0 → PP=0
      - PP × TP exceeds tier 0 but fits tier 1 → PP=1
      - PP × TP × EP exceeds tier 1 → PP=2
  (b) Single-tier system (NVL72): all axes always at tier 0
  (c) Trivial axis sizes: PP=1 → tier 0; EP=1 / SP=1 → tier 0
  (d) Out-of-bounds group: clamps to outermost tier (does not raise)
  (e) End-to-end: t_PP via compute_comm matches the tier-assigned p2p_hop cost

Usage:  PYTHONPATH=. python tests/unit/test_partition_layout.py
"""
import math
import sys

from llm_perf.calculators.inference_calculator import InferenceCalculator
from llm_perf.core.primitives import (
    NESTED_LAYOUT_ORDER,
    assign_tier_per_axis,
    p2p_hop,
    tier_at,
)
from llm_perf.io import load_model_spec, load_system_spec, load_tuning_spec
from llm_perf.specs.partition_spec import PartitionSpec


_failures: list[str] = []


def _eq(label: str, got, want) -> None:
    if got == want:
        print(f"  PASS  {label}: {got!r}")
    else:
        _failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL  {label}: got {got!r}, want {want!r}")


def _close(label: str, got: float, want: float, rel_tol: float = 1e-12) -> None:
    if math.isclose(got, want, rel_tol=rel_tol, abs_tol=0.0):
        print(f"  PASS  {label}: {got!r} ≈ {want!r}")
    else:
        _failures.append(f"{label}: got {got!r}, want ~{want!r}")
        print(f"  FAIL  {label}: got {got!r}, want ~{want!r}")


# ────────────────────────────────────────────────────────────
# (a) 3-tier system tier mapping (d-Matrix squadrack)
# ────────────────────────────────────────────────────────────

def test_squadrack_pp_tier_assignment() -> None:
    print("\ntest_squadrack_pp_tier_assignment")
    s = load_system_spec("llm_perf/database/system/dmatrix.squadrack.json")
    # 5-tier chain: package(4) → DMX(16) → PCIe-A(32) → PCIe-B(64) → Ethernet(512).

    # PP=2 × TP=2 = 4 → fits tier 0 (package)
    a = assign_tier_per_axis(PartitionSpec(PP=2, TP=2, EP=1, SP=1), s)
    _eq("PP=2 TP=2: PP→tier 0", a["PP"], 0)

    # PP=4 × TP=4 = 16 → fits tier 1 cumulative (DMX Bridge)
    a = assign_tier_per_axis(PartitionSpec(PP=4, TP=4, EP=1, SP=1), s)
    _eq("PP=4 TP=4: PP→tier 1", a["PP"], 1)

    # PP=8 × TP=8 = 64 → fits tier 3 cumulative (PCIe-B inter-hemisphere)
    a = assign_tier_per_axis(PartitionSpec(PP=8, TP=8, EP=1, SP=1), s)
    _eq("PP=8 TP=8: PP→tier 3", a["PP"], 3)

    # PP=16 × TP=8 = 128 → needs tier 4 (Ethernet, cumulative 512)
    a = assign_tier_per_axis(PartitionSpec(PP=16, TP=8, EP=1, SP=1), s)
    _eq("PP=16 TP=8: PP→tier 4", a["PP"], 4)

    # PP=8 × TP=16 × EP=4 = 512: TP=16 needs tier 1 (cumul 16); EP=4 needs
    # tier 3 (cumul 16·4=64 ≤ 64); PP=8 needs tier 4 (cumul 64·8=512 ≤ 512).
    a = assign_tier_per_axis(PartitionSpec(PP=8, TP=16, EP=4, SP=1), s)
    _eq("TP=16 EP=4 PP=8: TP→1 EP→3 PP→4",
        (a["TP"], a["EP"], a["PP"]),
        (1, 3, 4))


# ────────────────────────────────────────────────────────────
# (b) Single-tier system collapses everything to tier 0
# ────────────────────────────────────────────────────────────

def test_nvl72_single_tier() -> None:
    print("\ntest_nvl72_single_tier")
    s = load_system_spec("llm_perf/database/system/gb200.72gpu.json")
    # Single tier with 72 ports. Every axis maps to tier 0.

    for shape in [
        PartitionSpec(PP=1, TP=1, EP=1, SP=1),
        PartitionSpec(PP=8, TP=8, EP=1, SP=1),
        PartitionSpec(PP=32, TP=2, EP=1, SP=1),
        PartitionSpec(PP=4, TP=16, EP=1, SP=1),
    ]:
        a = assign_tier_per_axis(shape, s)
        for ax, idx in a.items():
            if ax == "TP" or ax == "PP":
                _eq(f"NVL72 {ax} → tier 0  (shape PP={shape.PP} TP={shape.TP})", idx, 0)


# ────────────────────────────────────────────────────────────
# (c) Trivial axis sizes
# ────────────────────────────────────────────────────────────

def test_trivial_axes() -> None:
    print("\ntest_trivial_axes")
    s = load_system_spec("llm_perf/database/system/dmatrix.squadrack.json")

    # PP=1 → no comm; tier 0 placeholder
    a = assign_tier_per_axis(PartitionSpec(PP=1, TP=8, EP=1, SP=1), s)
    _eq("PP=1: PP→tier 0", a["PP"], 0)
    _eq("PP=1: SP=1 EP=1 → tier 0", (a["SP"], a["EP"]), (0, 0))

    # All ones → all tier 0
    a = assign_tier_per_axis(PartitionSpec(PP=1, TP=1, EP=1, SP=1), s)
    _eq("All ones → all tier 0", set(a.values()), {0})


# ────────────────────────────────────────────────────────────
# (d) Out-of-bounds: clamp to outermost
# ────────────────────────────────────────────────────────────

def test_out_of_range_clamps() -> None:
    print("\ntest_out_of_range_clamps")
    s = load_system_spec("llm_perf/database/system/dmatrix.squadrack.json")
    # Squadrack max cumulative reach = 4·4·2·2·8 = 512. Force a partition
    # whose group exceeds 512 by setting absurd values via override.
    p = PartitionSpec(PP=64, TP=16, EP=4, SP=1)  # 64*16*4 = 4096 > 512
    a = assign_tier_per_axis(p, s)
    last = len(s.get_tier_chain("PP")) - 1
    _eq("Oversize PP → clamps to outermost tier", a["PP"], last)


# ────────────────────────────────────────────────────────────
# (e) End-to-end: compute_comm uses the tier-assigned p2p_hop cost
# ────────────────────────────────────────────────────────────

def test_decode_pp_uses_assigned_tier() -> None:
    print("\ntest_decode_pp_uses_assigned_tier")
    m = load_model_spec("llm_perf/database/model/llama3.1_70b.json")
    m.bytes_per_param = 0.5
    s = load_system_spec("llm_perf/database/system/dmatrix.squadrack.json")
    t = load_tuning_spec("llm_perf/database/tuner/example.tuner.json")
    t.S_decode = 8192
    t.B_decode = 32

    # Shapes spanning multiple tiers of the 5-tier squadrack chain
    # (package 4 → DMX 16 → PCIe-A 32 → PCIe-B 64 → Ethernet 512).
    cases = [
        (PartitionSpec(PP=2, TP=2, EP=1, SP=1),  0),  # tier 0 (cumul 4)
        (PartitionSpec(PP=4, TP=4, EP=1, SP=1),  1),  # tier 1 (cumul 16)
        (PartitionSpec(PP=8, TP=8, EP=1, SP=1),  3),  # tier 3 (cumul 64)
        (PartitionSpec(PP=32, TP=8, EP=1, SP=1), 4),  # tier 4 (cumul 256, fits 512)
    ]
    H = m.H
    b = m.bytes_per_param
    for p, expected_tier in cases:
        r = InferenceCalculator(m, s, p, t).run()
        # Hand-derive expected t_PP from the tier values.
        msg_PP = t.B_decode * (H / p.TP) * b
        tier = tier_at(s, "PP", expected_tier)
        expected_t_PP = p2p_hop(msg_PP, tier.alpha_us * 1e-6, tier.bw_per_port_GBps * 1e9)
        _close(
            f"t_PP at PP={p.PP} TP={p.TP} = p2p_hop tier {expected_tier} (α={tier.alpha_us}μs, BW={tier.bw_per_port_GBps}GB/s)",
            r.comm.t_PP, expected_t_PP, rel_tol=1e-12,
        )


# ────────────────────────────────────────────────────────────


def main() -> int:
    test_squadrack_pp_tier_assignment()
    test_nvl72_single_tier()
    test_trivial_axes()
    test_out_of_range_clamps()
    test_decode_pp_uses_assigned_tier()
    if _failures:
        print(f"\n{len(_failures)} FAILURES:")
        for f in _failures:
            print(f"  {f}")
        return 1
    print("\nAll partition-layout unit tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
