
from dataclasses import dataclass
from typing import Dict


@dataclass
class DeviceSpec:
    """Single device (GPU/NPU) specs."""

    name: str
    hbm_capacity_GB: float        # capacity per device (GB, decimal)
    hbm_bandwidth_GBps: float      # effective memory bandwidth (GB/s)
    peak_flops_TF: float           # peak compute throughput (TFLOPs)


@dataclass
class NetworkDomainSpec:
    """Network domain parameters for a specific role (TP, EP, SP, PP)."""

    name: str
    alpha_us: float                # latency per message (microseconds)
    bandwidth_GBps: float          # effective bandwidth (GB/s)


@dataclass
class SystemSpec:
    """System / cluster description."""

    name: str
    device: DeviceSpec
    num_devices: int
    network_domains: Dict[str, NetworkDomainSpec]

    def get_domain(self, role: str) -> NetworkDomainSpec:
        """Return the network domain spec for a given role (TP/EP/SP/PP)."""
        return self.network_domains[role]
