
"""DRAM3D bandwidth utility (documentation/modeling/dram3d.md).

Derives HBM bandwidth from physical interface parameters and can update
existing database/system/*.json files with the computed values.
"""

import json
from pathlib import Path
from typing import Optional


def compute_hbm_bandwidth(
    die_area_mm2: float,
    pitch_um: float,
    data_pin_fraction: float,
    data_rate_gbps: float,
    n_dies: int = 8,
) -> dict:
    """Derive HBM bandwidth from physical parameters.

    Args:
        die_area_mm2: DRAM die area in mm².
        pitch_um: Hybrid bonding / µbump pitch in µm.
        data_pin_fraction: Fraction of total pins used for data (eta_data, typically 0.3-0.5).
        data_rate_gbps: Data rate per pin in Gbps.
        n_dies: Number of stacked DRAM dies.

    Returns:
        Dict with derived quantities:
          - n_pins_total: total pad count on die
          - n_pins_data: data pad count
          - bw_per_die_GBps: bandwidth per die interface (GB/s)
          - bw_total_GBps: total bandwidth (n_dies × bw_per_die) (GB/s)
    """
    die_area_um2 = die_area_mm2 * 1e6  # mm² → µm²
    n_pins_total = int(die_area_um2 / (pitch_um ** 2))
    n_pins_data = int(n_pins_total * data_pin_fraction)

    # Each data pin transfers data_rate_gbps Gbit/s = data_rate_gbps/8 GB/s
    bw_per_die_GBps = n_pins_data * data_rate_gbps / 8.0
    bw_total_GBps = n_dies * bw_per_die_GBps

    return {
        "n_pins_total": n_pins_total,
        "n_pins_data": n_pins_data,
        "bw_per_die_GBps": bw_per_die_GBps,
        "bw_total_GBps": bw_total_GBps,
    }


def update_system_json(
    system_json_path: str,
    hbm_bandwidth_GBps: float,
    hbm_capacity_GB: Optional[float] = None,
    output_path: Optional[str] = None,
) -> None:
    """Read a database/system/*.json and update the device memory parameters.

    Args:
        system_json_path: Path to the system JSON file.
        hbm_bandwidth_GBps: New HBM bandwidth value (GB/s).
        hbm_capacity_GB: New HBM capacity (GB), or None to leave unchanged.
        output_path: Output path. If None, overwrites the input file.
    """
    path = Path(system_json_path)
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg["device"]["hbm_bandwidth_GBps"] = hbm_bandwidth_GBps
    if hbm_capacity_GB is not None:
        cfg["device"]["hbm_capacity_GB"] = hbm_capacity_GB

    out_path = Path(output_path) if output_path else path
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
