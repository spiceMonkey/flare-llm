# llm_perf/io/system_loaders.py

import json
from pathlib import Path
from typing import Any, Dict

from ..specs.system_spec import DeviceSpec, NetworkDomainSpec, SystemSpec
from ..utils import (
    validate_positive_int_fields,
    validate_nonnegative_float_fields,
    validate_positive_float_fields,
)


def _load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def system_spec_from_json_dict(cfg: Dict[str, Any]) -> SystemSpec:
    """
    Build SystemSpec from a config dict.

    Expected format:

        {
          "schema": "llm_perf.system",
          "name": "...",
          "num_devices": 64,
          "device": {
            "name": "...",
            "hbm_capacity_GB": 80.0,
            "hbm_bandwidth_GBps": 3350.0,
            "peak_flops_TF": 1000.0
          },
          "network_domains": {
            "TP": { "name": "...", "alpha_us": 1.0, "bandwidth_GBps": 400.0 },
            ...
          }
        }
    """
    schema = cfg.get("schema", "llm_perf.system")
    if not schema.startswith("llm_perf.system"):
        raise ValueError(f"Unsupported system schema: {schema}")

    # num_devices must be integer >= 1
    validate_positive_int_fields(
        cfg,
        ["num_devices"],
        prefix="system configuration",
    )

    dev_cfg = cfg["device"]
    # Device numeric fields must be floats > 0.0
    validate_positive_float_fields(
        dev_cfg,
        ["hbm_capacity_GB", "hbm_bandwidth_GBps", "peak_flops_TF"],
        prefix="device configuration",
    )
    device = DeviceSpec(
        name=str(dev_cfg["name"]),
        hbm_capacity_GB=float(dev_cfg["hbm_capacity_GB"]),
        hbm_bandwidth_GBps=float(dev_cfg["hbm_bandwidth_GBps"]),
        peak_flops_TF=float(dev_cfg["peak_flops_TF"]),
    )

    net_cfg = cfg["network_domains"]
    network_domains: Dict[str, NetworkDomainSpec] = {}
    for role, nd in net_cfg.items():
        # alpha_us must be >= 0.0; bandwidth_GBps must be > 0.0
        validate_nonnegative_float_fields(
            nd,
            ["alpha_us"],
            prefix=f"network domain '{role}'",
        )
        validate_positive_float_fields(
            nd,
            ["bandwidth_GBps"],
            prefix=f"network domain '{role}'",
        )

        network_domains[role] = NetworkDomainSpec(
            name=str(nd["name"]),
            alpha_us=float(nd["alpha_us"]),
            bandwidth_GBps=float(nd["bandwidth_GBps"]),
        )

    return SystemSpec(
        name=str(cfg.get("name", "unnamed_system")),
        device=device,
        num_devices=int(cfg["num_devices"]),
        network_domains=network_domains,
    )


def load_system_spec(path: str | Path) -> SystemSpec:
    """
    Load SystemSpec from a JSON file.

    Example:
        system = load_system_spec("system.json")
    """
    cfg = _load_json(path)
    return system_spec_from_json_dict(cfg)
