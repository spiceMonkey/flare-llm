# llm_perf/io/system_loaders.py

import json
from pathlib import Path
from typing import Any, Dict, List

from ..specs.system_spec import (
    COLLECTIVES,
    CrossbarTier,
    DeviceSpec,
    DragonflyTier,
    FabricSpec,
    SwitchTierSpec,
    SystemSpec,
    TierSpec,
    TorusTier,
)
from ..utils import (
    validate_positive_int_fields,
    validate_nonnegative_float_fields,
    validate_positive_float_fields,
)


def _load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_crossbar_tier(prefix: str, name: str, tc: Dict[str, Any]) -> CrossbarTier:
    validate_positive_int_fields(tc, ["ports"], prefix=prefix)
    validate_nonnegative_float_fields(tc, ["alpha_us"], prefix=prefix)
    validate_positive_float_fields(tc, ["bw_per_port_GBps"], prefix=prefix)
    return CrossbarTier(
        name=name,
        ports=int(tc["ports"]),
        bw_per_port_GBps=float(tc["bw_per_port_GBps"]),
        alpha_us=float(tc["alpha_us"]),
    )


def _parse_torus_tier(prefix: str, name: str, tc: Dict[str, Any]) -> TorusTier:
    if "dims" not in tc:
        raise ValueError(f"{prefix}: torus tier missing 'dims' tuple")
    dims_cfg = tc["dims"]
    if not isinstance(dims_cfg, list) or not dims_cfg:
        raise ValueError(f"{prefix}: 'dims' must be a non-empty list of positive ints")
    dims: List[int] = []
    for j, d in enumerate(dims_cfg):
        if not isinstance(d, int) or d <= 0:
            raise ValueError(f"{prefix}: dims[{j}] must be a positive int, got {d!r}")
        dims.append(int(d))
    validate_nonnegative_float_fields(tc, ["alpha_us"], prefix=prefix)
    validate_positive_float_fields(tc, ["bw_per_port_GBps"], prefix=prefix)
    return TorusTier(
        name=name,
        dims=tuple(dims),
        bw_per_port_GBps=float(tc["bw_per_port_GBps"]),
        alpha_us=float(tc["alpha_us"]),
    )


def _parse_dragonfly_tier(prefix: str, name: str, tc: Dict[str, Any]) -> DragonflyTier:
    validate_positive_int_fields(
        tc, ["p_endpoints", "a_routers", "h_global", "g_groups"], prefix=prefix
    )
    validate_nonnegative_float_fields(
        tc, ["alpha_us", "alpha_local_us", "alpha_global_us"], prefix=prefix
    )
    validate_positive_float_fields(
        tc, ["bw_per_port_GBps", "bw_local_GBps", "bw_global_GBps"], prefix=prefix
    )
    return DragonflyTier(
        name=name,
        p_endpoints=int(tc["p_endpoints"]),
        a_routers=int(tc["a_routers"]),
        h_global=int(tc["h_global"]),
        g_groups=int(tc["g_groups"]),
        bw_per_port_GBps=float(tc["bw_per_port_GBps"]),
        alpha_us=float(tc["alpha_us"]),
        alpha_local_us=float(tc["alpha_local_us"]),
        alpha_global_us=float(tc["alpha_global_us"]),
        bw_local_GBps=float(tc["bw_local_GBps"]),
        bw_global_GBps=float(tc["bw_global_GBps"]),
    )


_TOPOLOGY_PARSERS = {
    "crossbar": _parse_crossbar_tier,
    "torus": _parse_torus_tier,
    "dragonfly": _parse_dragonfly_tier,
}


def _parse_tier(prefix: str, name: str, tc: Dict[str, Any]) -> TierSpec:
    topology = str(tc.get("topology", "crossbar"))
    parser = _TOPOLOGY_PARSERS.get(topology)
    if parser is None:
        raise ValueError(
            f"{prefix}: unknown topology {topology!r}; "
            f"supported: {sorted(_TOPOLOGY_PARSERS)}"
        )
    return parser(prefix, name, tc)


def _parse_fabric(fabric_name: str, fc: Dict[str, Any]) -> FabricSpec:
    """Parse one fabric entry. Requires a non-empty 'tiers' list.

    Each tier may declare `"topology": "crossbar" | "torus" | "dragonfly"`;
    the field defaults to `"crossbar"` so existing system JSONs parse unchanged.
    """
    if "tiers" not in fc:
        raise ValueError(f"fabric '{fabric_name}': missing 'tiers' list")
    tiers_cfg = fc["tiers"]
    if not isinstance(tiers_cfg, list) or not tiers_cfg:
        raise ValueError(f"fabric '{fabric_name}': 'tiers' must be a non-empty list")

    tiers: List[TierSpec] = []
    for i, tc in enumerate(tiers_cfg):
        prefix = f"fabric '{fabric_name}' tier[{i}]"
        tier_name = str(tc.get("name", f"tier{i}"))
        tiers.append(_parse_tier(prefix, tier_name, tc))
    return FabricSpec(name=fabric_name, tiers=tiers)


def _normalize_chain(collective: str, value: Any) -> List[str]:
    """Normalize a collective_fabrics value to a list of fabric names.

    String sugar: `"nvlink5"` → `["nvlink5"]`.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        if not value:
            raise ValueError(
                f"collective '{collective}': fabric chain must be non-empty"
            )
        return list(value)
    raise ValueError(
        f"collective '{collective}': must map to a fabric name or list of fabric names, "
        f"got {type(value).__name__}"
    )


def system_spec_from_json_dict(cfg: Dict[str, Any]) -> SystemSpec:
    """Build SystemSpec from a config dict.

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
          "fabrics": {
            "nvlink5": {
              "tiers": [
                {"name": "intra-rack-nvswitch", "ports": 72,
                 "bw_per_port_GBps": 900.0, "alpha_us": 0.5}
              ]
            },
            "ib": {
              "tiers": [
                {"name": "inter-rack-quantum-ib", "ports": 8,
                 "bw_per_port_GBps": 400.0, "alpha_us": 2.5}
              ]
            }
          },
          "collective_fabrics": {
            "TP": ["nvlink5", "ib"],
            "EP": "nvlink5",
            "SP": "nvlink5",
            "PP": ["nvlink5", "ib"]
          }
        }

    A collective's value may be a single string (single-fabric shorthand) or
    an ordered list of fabric names (escalation chain, innermost first).
    """
    schema = cfg.get("schema", "llm_perf.system")
    if not schema.startswith("llm_perf.system"):
        raise ValueError(f"Unsupported system schema: {schema}")

    validate_positive_int_fields(
        cfg,
        ["num_devices"],
        prefix="system configuration",
    )
    num_devices = int(cfg["num_devices"])

    dev_cfg = cfg["device"]
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

    if "fabrics" not in cfg:
        raise ValueError("system configuration: missing 'fabrics' block")
    if "collective_fabrics" not in cfg:
        raise ValueError("system configuration: missing 'collective_fabrics' block")

    fab_cfg = cfg["fabrics"]
    if not isinstance(fab_cfg, dict) or not fab_cfg:
        raise ValueError("'fabrics' must be a non-empty object")
    fabrics: Dict[str, FabricSpec] = {}
    for fabric_name, fc in fab_cfg.items():
        fabrics[str(fabric_name)] = _parse_fabric(str(fabric_name), fc)

    coll_cfg = cfg["collective_fabrics"]
    if not isinstance(coll_cfg, dict):
        raise ValueError("'collective_fabrics' must be an object")

    # Validation: unknown collectives, missing collectives, dangling fabric refs.
    unknown = set(coll_cfg.keys()) - set(COLLECTIVES)
    if unknown:
        raise ValueError(
            f"collective_fabrics: unknown collective(s) {sorted(unknown)}; "
            f"allowed: {list(COLLECTIVES)}"
        )
    missing = set(COLLECTIVES) - set(coll_cfg.keys())
    if missing:
        raise ValueError(
            f"collective_fabrics: missing required collective(s) {sorted(missing)}; "
            f"all of {list(COLLECTIVES)} must be mapped"
        )

    collective_fabrics: Dict[str, List[str]] = {}
    for collective in COLLECTIVES:
        chain = _normalize_chain(collective, coll_cfg[collective])
        for fabric_name in chain:
            if fabric_name not in fabrics:
                raise ValueError(
                    f"collective_fabrics['{collective}']: fabric '{fabric_name}' "
                    f"not defined in 'fabrics'; known: {sorted(fabrics.keys())}"
                )
        collective_fabrics[collective] = chain

    return SystemSpec(
        name=str(cfg.get("name", "unnamed_system")),
        device=device,
        num_devices=num_devices,
        fabrics=fabrics,
        collective_fabrics=collective_fabrics,
    )


def load_system_spec(path: str | Path) -> SystemSpec:
    """Load SystemSpec from a JSON file."""
    cfg = _load_json(path)
    return system_spec_from_json_dict(cfg)
