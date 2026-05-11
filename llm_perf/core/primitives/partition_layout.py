"""Nested-partition layout: which fabric tier holds which parallelism axis.

The nested-partitioning rule used across the framework — and standard in
production deployments — is

    DP (outermost / slowest) → PP → EP → TP → SP (innermost / fastest)

with collective-heavy axes (TP, EP, SP) pinned to the innermost (highest-
bandwidth) fabric tier and bandwidth-tolerant axes (PP, DP) spilling to
outer tiers as needed. ``assign_tier_per_axis`` resolves a `PartitionSpec`
+ `SystemSpec` into a per-axis tier index under this rule.

The output is consumed by the comm-cost path so that, e.g., PP send/recv
between two adjacent stages is priced at the *correct* tier — the
framework would otherwise price every PP hop at tier 0 because
`cost_collective` is called with `G=2` (single hop), which always
resolves to the innermost tier even when PP physically spans an outer
fabric tier (e.g., d-Matrix squadrack PP=32 with TP=8 spanning the
ethernet rack-tier, not just the package D2D mesh).

Under TP+EP co-location (`FrameworkSpec.layout == "co_located"`), TP and EP
are overlaid on the same physical GPU set, so the cumulative-reach
calculation must count them once (as `max(TP, EP)`) rather than as their
product. The helper `_axis_size_for_reach` encodes this.
"""

from typing import Dict, Tuple

from ...specs.framework_spec import FrameworkSpec
from ...specs.partition_spec import PartitionSpec
from ...specs.system_spec import SystemSpec, TierSpec


# Inner-to-outer order used by the nested-layout rule.
DEFAULT_ORDER: Tuple[str, ...] = ("SP", "TP", "EP", "PP")


def _axis_size_for_reach(partition: PartitionSpec, framework: FrameworkSpec, axis: str) -> int:
    """Effective per-axis multiplier for cumulative-reach accumulation.

    Returns the raw ``getattr(partition, axis)`` for orthogonal layouts
    (the canonical product `SP*TP*EP*PP`). Under co-located layouts, TP and
    EP overlay on the same physical GPUs, so:

      - axis "TP" returns ``max(TP, EP)`` (the merged axis size)
      - axis "EP" returns 1 (already counted by the merged TP-axis above)

    Inert for non-overlapping axes (PP, SP) and orthogonal layouts.
    """
    if axis not in ("TP", "EP"):
        return max(1, getattr(partition, axis, 1))
    if framework.layout != "co_located":
        return max(1, getattr(partition, axis, 1))
    # Co-located: TP and EP share physical GPUs.
    if axis == "TP":
        return max(partition.TP, max(1, partition.EP))
    return 1  # axis == "EP" — absorbed into the merged TP axis above


def assign_tier_per_axis(
    partition: PartitionSpec,
    framework: FrameworkSpec,
    system: SystemSpec,
    role: str = "TP",
    order: Tuple[str, ...] = DEFAULT_ORDER,
) -> Dict[str, int]:
    """Map each parallelism axis to a fabric-tier index under nested layout.

    Walks ``order`` from innermost to outermost. For each axis, the
    cumulative product of inner-axis sizes is multiplied by this axis's
    size; the resulting "group reach" is matched against the cumulative
    reach of the fabric tiers (``prod(tier.ports)``), and the smallest
    tier whose reach covers the group is assigned.

    Trivial axes (size <= 1) get ``tier_index = 0`` (no comm; no cost).
    Groups that exceed the fabric's outermost reach also clamp to the
    outermost tier (the cost will be the worst case for that fabric).

    Co-located layouts merge TP and EP via ``_axis_size_for_reach`` so the
    cumulative product matches the actual replica size
    ``PP * max(TP, EP) * SP`` rather than the orthogonal `PP*TP*EP*SP`.

    Returns ``{axis_name: tier_index}`` for every axis in ``order``.
    The dict ordering matches ``order`` so callers can iterate predictably.
    """
    chain = system.get_tier_chain(role)
    if not chain:
        return {ax: 0 for ax in order}

    cumulative_reach = []
    reach = 1
    for tier in chain:
        reach *= max(1, tier.ports)
        cumulative_reach.append(reach)
    last_tier = len(chain) - 1

    assignment: Dict[str, int] = {}
    cumulative_group = 1
    for ax in order:
        # Raw axis size still drives the "is this axis trivial?" check —
        # if the user set EP=1, the EP collective never fires (orthogonal
        # or co-located). The cumulative-reach multiplier is layout-aware
        # via _axis_size_for_reach.
        raw_n = max(1, getattr(partition, ax, 1))
        if raw_n <= 1:
            assignment[ax] = 0
            continue
        cumulative_group *= _axis_size_for_reach(partition, framework, ax)
        assigned = last_tier  # default: outermost
        for i, r in enumerate(cumulative_reach):
            if r >= cumulative_group:
                assigned = i
                break
        assignment[ax] = assigned
    return assignment


def tier_at(system: SystemSpec, role: str, tier_index: int) -> TierSpec:
    """Return the tier at ``tier_index`` of the role's fabric chain.

    Out-of-range indices clamp to the outermost tier rather than raising,
    matching the behavior of ``assign_tier_per_axis``.
    """
    chain = system.get_tier_chain(role)
    if not chain:
        raise ValueError(f"No tier chain configured for collective role {role!r}")
    return chain[max(0, min(tier_index, len(chain) - 1))]
