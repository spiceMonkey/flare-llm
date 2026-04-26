
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union


@dataclass
class DeviceSpec:
    """Single device (GPU/NPU) specs."""

    name: str
    hbm_capacity_GB: float        # capacity per device (GB, decimal)
    hbm_bandwidth_GBps: float      # effective memory bandwidth (GB/s)
    peak_flops_TF: float           # peak compute throughput (TFLOPs)


@dataclass
class CrossbarTier:
    """One crossbar switching tier (single monolithic chip abstraction).

    Default topology for all existing system JSONs. `ports` is the radix —
    the number of ranks reachable within this tier from any single rank.
    Cumulative reach at tier k is the product of ports over tiers 0..k.
    See documentation/modeling/collectives.md §3.1 / §3.4 for the cost forms consumed by each tier kind.

    Contention coefficients `eta_alpha` (≥ 1, inflates α) and `eta_beta`
    (∈ (0, 1], deflates BW) coarsen dynamic contention into the α–β model
    per documentation/modeling/contention.md. Defaults are 1.0 (ideal).

    `inc` flags in-network collective support on this tier's switch ASIC:
    "none" (software collectives only), "sharp_class" (NVLink SHARP / NVLS,
    Quantum SHARP, Spectrum-X SHARP — switch ALU + multicast crossbar;
    accelerates AR / AG / RS), or "hw_a2a" (Tomahawk Ultra / Rubin-class
    crossbar scatter-gather; accelerates AR / AG / RS *and* A2A). The
    legacy values "nvls" and "sharp" are accepted by the loader and mapped
    to "sharp_class" for backwards compatibility. When set, `inc_alpha_us`
    optionally overrides the switch-cut-through α used on the INC path —
    0.0 means "reuse alpha_us" (i.e., model assumes the tier's α already
    captures cut-through latency).

    See documentation/modeling/collectives.md §3.4, §4.4, §5.4 and
    documentation/explaining/collectives/04_in_network_collectives.md.
    """

    name: str
    ports: int                     # radix at this tier (reachable ranks)
    bw_per_port_GBps: float        # effective per-port bandwidth (GB/s)
    alpha_us: float                # per-traversal latency (microseconds)
    topology: str = "crossbar"
    eta_alpha: float = 1.0         # contention α-inflator (≥ 1; ideal = 1)
    eta_beta: float = 1.0          # contention BW-deflator (∈ (0, 1]; ideal = 1)
    inc: str = "none"              # "none" | "sharp_class" | "hw_a2a"
    inc_alpha_us: float = 0.0      # switch-cut-through α on INC path; 0 = reuse alpha_us
    oversubscription: float = 1.0  # tier oversubscription ratio s ≥ 1; caps eta_beta at 1/s


@dataclass
class TorusTier:
    """One k-D torus tier.

    `dims` is (D_1, ..., D_k); reach is prod(dims). Each node has 2k neighbor
    links; `bw_per_port_GBps` is the per-link single-direction bandwidth.
    Diameter = sum(floor(D_i/2)); bisection cut binds the largest dim.
    See documentation/modeling/collectives.md §3.2 / §4.2 / §5.2 for the dim-decomposed primitive cost forms.

    Contention coefficients `eta_alpha`, `eta_beta` as in `CrossbarTier`;
    see documentation/modeling/contention.md.
    """

    name: str
    dims: Tuple[int, ...]          # (D_1, ..., D_k)
    bw_per_port_GBps: float        # per-link single-direction BW (GB/s)
    alpha_us: float                # per-hop latency (μs)
    topology: str = "torus"
    eta_alpha: float = 1.0         # contention α-inflator
    eta_beta: float = 1.0          # contention BW-deflator

    @property
    def ports(self) -> int:
        """Reach = prod(dims). Exposed for compatibility with span_tiers."""
        n = 1
        for d in self.dims:
            n *= d
        return n


# Back-compat alias: existing imports `from ...system_spec import SwitchTierSpec`
# keep working. SwitchTierSpec IS CrossbarTier at runtime, so isinstance checks
# and positional construction `SwitchTierSpec(name, ports, bw, alpha)` are
# preserved. New code should prefer CrossbarTier.
SwitchTierSpec = CrossbarTier

# Discriminated-union type alias for static checkers. At runtime, use
# `tier.topology` to branch.
TierSpec = Union[CrossbarTier, TorusTier]


@dataclass
class FabricSpec:
    """One physical network technology, described as an ordered tier list.

    A fabric represents a single underlying network (e.g. NVLink5, InfiniBand,
    Ethernet). Collectives map onto an ordered chain of fabrics in
    `SystemSpec.collective_fabrics`; tiers from every fabric in the chain are
    walked innermost-first to cost a collective via `span_tiers`.
    """

    name: str
    tiers: List[TierSpec] = field(default_factory=list)


def span_tiers(
    tiers: List[TierSpec], group_size: int
) -> Tuple[float, float, int]:
    """Return (alpha_total_us, bw_min_GBps, n_tiers_crossed) for a collective
    over `group_size` ranks walking `tiers` innermost-first.

    Accumulates α across every tier touched and floors bandwidth to the
    narrowest tier actually crossed. Returns (0, 0, 0) for trivial collectives.
    If `group_size` exceeds total reach, returns the outermost configuration.

    Crossbar-only helper: the α-sum / BW-min flatten pattern is exact for
    crossbar tiers but approximate for torus. Topology-aware dispatch lives
    in `core/primitives/dispatch.py`.
    """
    if group_size <= 1 or not tiers:
        return 0.0, 0.0, 0

    alpha_total = 0.0
    bw_min = float("inf")
    reach = 1
    crossed = 0
    for tier in tiers:
        reach *= max(1, tier.ports)
        alpha_total += tier.alpha_us
        bw_min = min(bw_min, tier.bw_per_port_GBps)
        crossed += 1
        if reach >= group_size:
            return alpha_total, bw_min, crossed

    return alpha_total, bw_min, crossed


COLLECTIVES: Tuple[str, ...] = ("TP", "EP", "SP", "PP")


@dataclass
class SystemSpec:
    """System / cluster description.

    Networks are declared once as named fabrics; each collective (TP/EP/SP/PP)
    maps to an ordered list of fabric names it escalates through. Collectives
    spanning more ranks than a fabric's innermost tier can reach continue into
    the next fabric in the chain — this lets scale-up (NVLink) and scale-out
    (IB/Ethernet) be modeled as distinct physical networks.
    """

    name: str
    device: DeviceSpec
    num_devices: int
    fabrics: Dict[str, FabricSpec]
    collective_fabrics: Dict[str, List[str]]

    def get_fabric_chain(self, collective: str) -> List[FabricSpec]:
        """Return the ordered FabricSpec list for a collective (TP/EP/SP/PP)."""
        chain_names = self.collective_fabrics[collective]
        return [self.fabrics[name] for name in chain_names]

    def get_tier_chain(self, collective: str) -> List[TierSpec]:
        """Return the concatenated tier list across the collective's fabric chain."""
        chain: List[TierSpec] = []
        for fabric in self.get_fabric_chain(collective):
            chain.extend(fabric.tiers)
        return chain
