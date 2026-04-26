
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class MemoryTierSpec:
    """One memory tier exposed by a device.

    Per `documentation/modeling/sram.md §1.1`, devices expose an ordered list
    of memory tiers, fastest first. Each tier carries a capacity, an effective
    peak read bandwidth, an optional first-byte latency, and an optional
    sustained-bandwidth deflator. The `eta_beta` deflator follows the same
    convention as collective contention (`collectives.md §7`): 1.0 = peak,
    < 1 = sustained-throughput losses (HBM refresh + bank conflicts ≈ 0.92,
    LPDDR5 ≈ 0.85, SRAM ≈ 1.0; sram.md §1.2). The `alpha_us` first-byte cost
    is structurally negligible for steady-state decode (see sram.md §2.1) and
    is dropped by the device-level roofline; it is kept on the spec for
    small-read regimes (paged-attention block fetch, flash-style spill).
    """

    name: str                      # e.g. "hbm", "sram", "lpddr5"
    capacity_GB: float             # per-device capacity (GB, decimal)
    bandwidth_GBps: float          # peak read bandwidth (GB/s)
    alpha_us: float = 0.0          # first-byte latency (μs); decode roofline drops it
    eta_beta: float = 1.0          # sustained-BW deflator ∈ (0, 1]; 1.0 = peak


@dataclass
class DeviceSpec:
    """Single device (GPU/NPU) specs.

    The legacy fields `hbm_capacity_GB` / `hbm_bandwidth_GBps` model the
    device's main DRAM-like tier and remain required for back-compat. Two
    additive paths to multi-tier:

      1. **Top-level `sram_capacity_MB` / `sram_bandwidth_TBps`** (PR3): a
         fast on-die SRAM cache layered on top of the main DRAM tier.
         Shipped naming convention for SRAM-augmented devices like d-Matrix
         Corsair (SRAM + LPDDR5 represented as SRAM + "hbm" slot). Units
         chosen to match natural magnitudes: MB and TB/s.

      2. **Explicit `tiers: List[MemoryTierSpec]`** (PR1): an ordered list
         for arbitrary multi-tier topologies. Use this when the field-name
         conventions of (1) don't fit (e.g., a 3+ tier device, or non-HBM
         main memory where you want an accurate `name` like "lpddr5").

    Downstream code should call `get_tiers()` instead of reading any of these
    fields directly. The shim materializes:

      - `tiers` if non-empty (path 2)
      - `[SRAM tier, HBM tier]` if `sram_*` is set (path 1, sram.md §1.1)
      - `[HBM tier]` otherwise (legacy single-tier; preserves regression)

    Auto-materialized tiers use `eta_beta = 1.0` to keep `t_mem` numerically
    identical to the legacy `T_step / BW_mem` formula. New devices wanting
    the sram.md §1.2 defaults (HBM=0.92, LPDDR5=0.85) should use path 2.
    """

    name: str
    hbm_capacity_GB: float        # capacity per device (GB, decimal)
    hbm_bandwidth_GBps: float      # effective memory bandwidth (GB/s)
    peak_flops_TF: float           # peak compute throughput (TFLOPs)
    sram_capacity_MB: Optional[float] = None   # optional fast SRAM tier (MB)
    sram_bandwidth_TBps: Optional[float] = None  # optional fast SRAM BW (TB/s)
    tiers: List["MemoryTierSpec"] = field(default_factory=list)

    def get_tiers(self) -> List["MemoryTierSpec"]:
        """Return the device's memory tier list, materializing a shim from
        the legacy / sram_* fields when `tiers` is empty.

        Materialization order (sram.md §1.1, fastest first):
          - explicit `tiers` (if non-empty)
          - `[SRAM tier, HBM tier]` if `sram_*` set (PR3 convention)
          - `[HBM tier]` (legacy single-tier path)
        """
        if self.tiers:
            return self.tiers
        materialized: List["MemoryTierSpec"] = []
        if self.sram_capacity_MB is not None and self.sram_bandwidth_TBps is not None:
            materialized.append(
                MemoryTierSpec(
                    name="sram",
                    capacity_GB=self.sram_capacity_MB / 1000.0,  # MB → GB (decimal)
                    bandwidth_GBps=self.sram_bandwidth_TBps * 1000.0,  # TB/s → GB/s
                    alpha_us=0.0,
                    eta_beta=1.0,
                )
            )
        materialized.append(
            MemoryTierSpec(
                name="hbm",
                capacity_GB=self.hbm_capacity_GB,
                bandwidth_GBps=self.hbm_bandwidth_GBps,
                alpha_us=0.0,
                eta_beta=1.0,
            )
        )
        return materialized


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


@dataclass
class MeshTier:
    """One mesh tier — handles both full mesh and k-D mesh via the `full` flag.

    `full=True`: full mesh — every node connects directly to every other node
    (single hop, full bisection). $\\binom{N}{2}$ edges. Examples: chiplet UCIe
    interposer, DGX-1/DGX-2 hybrid cube-mesh (legacy). Cost formulas match a
    single-tier crossbar exactly (single hop, full bisection); the dispatcher
    routes a full-mesh tier through the crossbar primitives. `dims` is
    expected to be a 1-tuple `(N,)` here.

    `full=False`: k-D mesh — torus without wraparound edges. Open-line bucket
    brigade along each axis. AR/AG/RS cost formulas match torus exactly
    (open-line still telescopes BW-optimally); A2A pays a 2× BW penalty
    versus torus because the bisection cut is halved (missing wraparound
    edges) — $D_\\mathrm{max}/4$ instead of $D_\\mathrm{max}/8$. The
    dispatcher routes a k-D-mesh tier through the torus primitives with
    `wraparound=False`.

    No `inc` field — mesh has no switch ASIC; INC is structurally absent.

    See documentation/modeling/collectives.md §1 (mesh notes) and the
    explainer `02_topology_mapping.md §4` for full vs k-D mesh derivations.
    """

    name: str
    dims: Tuple[int, ...]          # full=True: (N,); full=False: per-axis extents
    bw_per_port_GBps: float        # per-link single-direction BW (GB/s)
    alpha_us: float                # per-link / per-hop latency (μs)
    full: bool = False             # True for full mesh, False for k-D mesh
    topology: str = "mesh"
    eta_alpha: float = 1.0         # contention α-inflator
    eta_beta: float = 1.0          # contention BW-deflator

    @property
    def ports(self) -> int:
        """Reach = prod(dims). For full mesh dims=(N,) so ports=N; for k-D mesh
        ports = ∏ dim_i = N (total node count). Exposed for compatibility with
        span_tiers and the dispatcher's tier walk."""
        n = 1
        for d in self.dims:
            n *= d
        return n


# Discriminated-union type alias for static checkers. At runtime, use
# `tier.topology` (and for mesh, `tier.full`) to branch.
TierSpec = Union[CrossbarTier, TorusTier, MeshTier]


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
