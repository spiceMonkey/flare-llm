
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class DeviceSpec:
    """Single device (GPU/NPU) specs."""

    name: str
    hbm_capacity_GB: float        # capacity per device (GB, decimal)
    hbm_bandwidth_GBps: float      # effective memory bandwidth (GB/s)
    peak_flops_TF: float           # peak compute throughput (TFLOPs)


@dataclass
class SwitchTierSpec:
    """One switching tier within a fabric.

    A tier models a homogeneous switching layer (e.g. one rail of NVSwitches,
    or one inter-rack aggregation layer). `ports` is the radix — the number
    of ranks reachable within this tier from any single rank. Cumulative
    reach at tier k is the product of ports over tiers 0..k.
    """

    name: str
    ports: int                     # radix at this tier (reachable ranks)
    bw_per_port_GBps: float        # effective per-port bandwidth (GB/s)
    alpha_us: float                # per-traversal latency (microseconds)


@dataclass
class FabricSpec:
    """One physical network technology, described as an ordered tier list.

    A fabric represents a single underlying network (e.g. NVLink5, InfiniBand,
    Ethernet). Collectives map onto an ordered chain of fabrics in
    `SystemSpec.collective_fabrics`; tiers from every fabric in the chain are
    walked innermost-first to cost a collective via `span_tiers`.
    """

    name: str
    tiers: List[SwitchTierSpec] = field(default_factory=list)


def span_tiers(
    tiers: List[SwitchTierSpec], group_size: int
) -> Tuple[float, float, int]:
    """Return (alpha_total_us, bw_min_GBps, n_tiers_crossed) for a collective
    over `group_size` ranks walking `tiers` innermost-first.

    Accumulates α across every tier touched and floors bandwidth to the
    narrowest tier actually crossed. Returns (0, 0, 0) for trivial collectives.
    If `group_size` exceeds total reach, returns the outermost configuration.
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

    def get_tier_chain(self, collective: str) -> List[SwitchTierSpec]:
        """Return the concatenated tier list across the collective's fabric chain."""
        chain: List[SwitchTierSpec] = []
        for fabric in self.get_fabric_chain(collective):
            chain.extend(fabric.tiers)
        return chain
