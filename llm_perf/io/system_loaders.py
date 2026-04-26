# llm_perf/io/system_loaders.py

import json
from pathlib import Path
from typing import Any, Dict, List

from ..specs.system_spec import (
    COLLECTIVES,
    CrossbarTier,
    DeviceSpec,
    FabricSpec,
    MemoryTierSpec,
    MeshTier,
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


def _eta_alpha(prefix: str, tc: Dict[str, Any], key: str = "eta_alpha") -> float:
    """Parse an optional eta_alpha field (contention α-inflator, ≥ 1, default 1.0)."""
    if key not in tc:
        return 1.0
    val = tc[key]
    if not isinstance(val, (int, float)) or val < 1.0:
        raise ValueError(
            f"{prefix}: '{key}' must be a float ≥ 1.0 (α-inflator), got {val!r}"
        )
    return float(val)


def _eta_beta(prefix: str, tc: Dict[str, Any], key: str = "eta_beta") -> float:
    """Parse an optional eta_beta field (contention BW-deflator, ∈ (0, 1], default 1.0)."""
    if key not in tc:
        return 1.0
    val = tc[key]
    if not isinstance(val, (int, float)) or val <= 0.0 or val > 1.0:
        raise ValueError(
            f"{prefix}: '{key}' must be a float in (0, 1] (BW-deflator), got {val!r}"
        )
    return float(val)


_INC_MODES = ("none", "sharp_class", "hw_a2a")
# Legacy values mapped to the new enum on load. `nvls` (NVLink SHARP) and
# `sharp` (Quantum / Spectrum-X SHARP) both map to `sharp_class` because they
# share the same modeling form (switch ALU + multicast crossbar; AR / AG / RS
# routing). HW A2A is a separate, newer capability with no legacy alias.
_INC_LEGACY_ALIASES = {"nvls": "sharp_class", "sharp": "sharp_class"}


# Default eta_beta per tier name (sram.md §1.2). When a JSON memory-tier
# entry omits eta_beta, the loader fills it in based on the tier's name (case
# insensitive). Names not in this table fall back to 1.0 (calibrated peak).
_MEMORY_TIER_DEFAULT_ETA_BETA = {
    "sram": 1.0,
    "hbm": 0.92,
    "lpddr5": 0.85,
    "lpddr": 0.85,
}


def _parse_memory_tier(prefix: str, tc: Dict[str, Any]) -> MemoryTierSpec:
    """Parse one entry of `device.tiers[]`.

    Required: name (str), capacity_GB (positive float), bandwidth_GBps
    (positive float). Optional: alpha_us (≥ 0, default 0.0), eta_beta
    (∈ (0, 1], default looked up from `_MEMORY_TIER_DEFAULT_ETA_BETA` by
    tier name or 1.0 if unknown). See sram.md §1.1 / §1.2.
    """
    if not isinstance(tc, dict):
        raise ValueError(f"{prefix}: memory tier entry must be an object, got {tc!r}")
    if "name" not in tc or not isinstance(tc["name"], str):
        raise ValueError(f"{prefix}: memory tier entry missing string 'name'")
    name = str(tc["name"])
    inner_prefix = f"{prefix} tier '{name}'"
    validate_positive_float_fields(
        tc, ["capacity_GB", "bandwidth_GBps"], prefix=inner_prefix
    )
    if "alpha_us" in tc:
        validate_nonnegative_float_fields(tc, ["alpha_us"], prefix=inner_prefix)
        alpha_us = float(tc["alpha_us"])
    else:
        alpha_us = 0.0
    if "eta_beta" in tc:
        eta_beta = _eta_beta(inner_prefix, tc, key="eta_beta")
    else:
        eta_beta = _MEMORY_TIER_DEFAULT_ETA_BETA.get(name.lower(), 1.0)
    return MemoryTierSpec(
        name=name,
        capacity_GB=float(tc["capacity_GB"]),
        bandwidth_GBps=float(tc["bandwidth_GBps"]),
        alpha_us=alpha_us,
        eta_beta=eta_beta,
    )


def _parse_inc(prefix: str, tc: Dict[str, Any]) -> str:
    """Parse the optional `inc` field (in-network collective flavor).

    Accepts both the new `{none, sharp_class, hw_a2a}` enum and the legacy
    `{none, nvls, sharp}` values (mapped to `sharp_class` for back-compat
    with existing system JSON files).
    """
    if "inc" not in tc:
        return "none"
    val = tc["inc"]
    if not isinstance(val, str):
        raise ValueError(
            f"{prefix}: 'inc' must be a string, got {val!r}"
        )
    canonical = _INC_LEGACY_ALIASES.get(val.lower(), val.lower())
    if canonical not in _INC_MODES:
        raise ValueError(
            f"{prefix}: 'inc' must be one of {list(_INC_MODES)} "
            f"(or legacy {sorted(_INC_LEGACY_ALIASES)}), got {val!r}"
        )
    return canonical


def _parse_inc_alpha_us(prefix: str, tc: Dict[str, Any]) -> float:
    """Parse the optional `inc_alpha_us` override. 0.0 sentinel = reuse alpha_us."""
    if "inc_alpha_us" not in tc:
        return 0.0
    val = tc["inc_alpha_us"]
    if not isinstance(val, (int, float)) or val < 0.0:
        raise ValueError(
            f"{prefix}: 'inc_alpha_us' must be a nonneg float, got {val!r}"
        )
    return float(val)


def _parse_oversubscription(prefix: str, tc: Dict[str, Any]) -> float:
    """Parse the optional `oversubscription` field (ratio s ≥ 1; default 1.0)."""
    if "oversubscription" not in tc:
        return 1.0
    val = tc["oversubscription"]
    if not isinstance(val, (int, float)) or val < 1.0:
        raise ValueError(
            f"{prefix}: 'oversubscription' must be a float ≥ 1.0, got {val!r}"
        )
    return float(val)


def _apply_oversubscription_cap(
    eta_beta: float, oversubscription: float
) -> float:
    """Apply the §7.2 oversubscription cap: realized eta_beta = min(provided, 1/s).

    Per collectives.md §7.2: a tier oversubscribed at ratio s ≥ 1 has aggregate
    upper-tier BW = (1/s) of aggregate downlink demand. The realized eta_beta
    is the minimum of the operator-supplied hardware-floor heuristic and the
    structural 1/s cap. Both inputs are model parameters, not error states —
    the operator supplies the hardware floor, the loader applies the structural
    cap on top, and the effective value is just min(...).
    """
    return min(eta_beta, 1.0 / oversubscription)


def _parse_crossbar_tier(prefix: str, name: str, tc: Dict[str, Any]) -> CrossbarTier:
    validate_positive_int_fields(tc, ["ports"], prefix=prefix)
    validate_nonnegative_float_fields(tc, ["alpha_us"], prefix=prefix)
    validate_positive_float_fields(tc, ["bw_per_port_GBps"], prefix=prefix)
    eta_beta_raw = _eta_beta(prefix, tc)
    oversub = _parse_oversubscription(prefix, tc)
    eta_beta = _apply_oversubscription_cap(eta_beta_raw, oversub)
    return CrossbarTier(
        name=name,
        ports=int(tc["ports"]),
        bw_per_port_GBps=float(tc["bw_per_port_GBps"]),
        alpha_us=float(tc["alpha_us"]),
        eta_alpha=_eta_alpha(prefix, tc),
        eta_beta=eta_beta,
        inc=_parse_inc(prefix, tc),
        inc_alpha_us=_parse_inc_alpha_us(prefix, tc),
        oversubscription=oversub,
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
        eta_alpha=_eta_alpha(prefix, tc),
        eta_beta=_eta_beta(prefix, tc),
    )


def _parse_mesh_tier(prefix: str, name: str, tc: Dict[str, Any]) -> MeshTier:
    if "dims" not in tc:
        raise ValueError(f"{prefix}: mesh tier missing 'dims' tuple")
    dims_cfg = tc["dims"]
    if not isinstance(dims_cfg, list) or not dims_cfg:
        raise ValueError(f"{prefix}: 'dims' must be a non-empty list of positive ints")
    dims: List[int] = []
    for j, d in enumerate(dims_cfg):
        if not isinstance(d, int) or d <= 0:
            raise ValueError(f"{prefix}: dims[{j}] must be a positive int, got {d!r}")
        dims.append(int(d))
    full = bool(tc.get("full", False))
    if full and len(dims) != 1:
        raise ValueError(
            f"{prefix}: full mesh expects dims to be a 1-tuple (N,), got {dims!r}"
        )
    validate_nonnegative_float_fields(tc, ["alpha_us"], prefix=prefix)
    validate_positive_float_fields(tc, ["bw_per_port_GBps"], prefix=prefix)
    return MeshTier(
        name=name,
        dims=tuple(dims),
        bw_per_port_GBps=float(tc["bw_per_port_GBps"]),
        alpha_us=float(tc["alpha_us"]),
        full=full,
        eta_alpha=_eta_alpha(prefix, tc),
        eta_beta=_eta_beta(prefix, tc),
    )


_TOPOLOGY_PARSERS = {
    "crossbar": _parse_crossbar_tier,
    "torus": _parse_torus_tier,
    "mesh": _parse_mesh_tier,
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

    Each tier may declare `"topology": "crossbar" | "torus"`; the field
    defaults to `"crossbar"` so existing system JSONs parse unchanged.
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
    tiers: List[MemoryTierSpec] = []
    if "tiers" in dev_cfg:
        raw_tiers = dev_cfg["tiers"]
        if not isinstance(raw_tiers, list) or not raw_tiers:
            raise ValueError(
                "device configuration: 'tiers' must be a non-empty list of memory tiers"
            )
        tiers = [
            _parse_memory_tier(f"device configuration tier[{idx}]", tc)
            for idx, tc in enumerate(raw_tiers)
        ]
    sram_capacity_MB = None
    sram_bandwidth_TBps = None
    has_sram_cap = "sram_capacity_MB" in dev_cfg
    has_sram_bw = "sram_bandwidth_TBps" in dev_cfg
    if has_sram_cap != has_sram_bw:
        raise ValueError(
            "device configuration: 'sram_capacity_MB' and 'sram_bandwidth_TBps' "
            "must be set together (or both omitted)"
        )
    if has_sram_cap:
        validate_positive_float_fields(
            dev_cfg, ["sram_capacity_MB", "sram_bandwidth_TBps"],
            prefix="device configuration",
        )
        sram_capacity_MB = float(dev_cfg["sram_capacity_MB"])
        sram_bandwidth_TBps = float(dev_cfg["sram_bandwidth_TBps"])
    device = DeviceSpec(
        name=str(dev_cfg["name"]),
        hbm_capacity_GB=float(dev_cfg["hbm_capacity_GB"]),
        hbm_bandwidth_GBps=float(dev_cfg["hbm_bandwidth_GBps"]),
        peak_flops_TF=float(dev_cfg["peak_flops_TF"]),
        sram_capacity_MB=sram_capacity_MB,
        sram_bandwidth_TBps=sram_bandwidth_TBps,
        tiers=tiers,
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
