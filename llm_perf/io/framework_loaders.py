"""FrameworkSpec JSON loaders.

Companion to `framework_spec.FrameworkSpec`. Mirrors the pattern of
`tuner_loaders.py` / `model_loaders.py`: a `framework_spec_from_json_dict`
that builds a FrameworkSpec from a parsed JSON dict (validating the
mode-string fields against their whitelists), plus a `load_framework_spec`
that reads a file path. Database-stem lookup (`load_framework_from_db`)
lives in `database_loaders.py` alongside the other spec families.
"""

import json
from pathlib import Path
from typing import Any, Dict

from ..specs.framework_spec import FrameworkSpec


def _load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def framework_spec_from_json_dict(cfg: Dict[str, Any]) -> FrameworkSpec:
    """
    Build FrameworkSpec from a config dict.

    Expected format:

        {
          "schema": "llm_perf.framework",
          "name": "dynamo-trt",
          "c_serving_per_seq_us": 0.0,
          "kernel_launch_us": 7.0,
          "kernels_per_layer_compute": 10,
          "kernels_per_collective_call": 2,
          "kernels_per_pp_hop": 2,
          "sw_overlap_factor": 1.0,
          "moe_a2a_pattern": "scatter",
          "mla_mode": "absorbed",
          "inc_enabled": true
        }

    All fields except `name` fall through to FrameworkSpec dataclass
    defaults when absent.
    """
    schema = cfg.get("schema", "llm_perf.framework")
    if not schema.startswith("llm_perf.framework"):
        raise ValueError(f"Unsupported framework schema: {schema}")

    moe_a2a_pattern = cfg.get("moe_a2a_pattern", "gather")
    if moe_a2a_pattern not in ("gather", "scatter"):
        raise ValueError(
            f"framework configuration: 'moe_a2a_pattern' must be 'gather' or "
            f"'scatter', got {moe_a2a_pattern!r}"
        )

    mla_mode = cfg.get("mla_mode", "absorbed")
    if mla_mode not in ("absorbed", "materialized"):
        raise ValueError(
            f"framework configuration: 'mla_mode' must be 'absorbed' or "
            f"'materialized', got {mla_mode!r}"
        )

    _defaults = FrameworkSpec(name="_defaults")
    return FrameworkSpec(
        name=str(cfg.get("name", "unnamed_framework")),
        c_serving_per_seq_us=float(cfg.get("c_serving_per_seq_us", _defaults.c_serving_per_seq_us)),
        kernel_launch_us=float(cfg.get("kernel_launch_us", _defaults.kernel_launch_us)),
        kernels_per_layer_compute=int(cfg.get("kernels_per_layer_compute", _defaults.kernels_per_layer_compute)),
        kernels_per_collective_call=int(cfg.get("kernels_per_collective_call", _defaults.kernels_per_collective_call)),
        kernels_per_pp_hop=int(cfg.get("kernels_per_pp_hop", _defaults.kernels_per_pp_hop)),
        sw_overlap_factor=float(cfg.get("sw_overlap_factor", _defaults.sw_overlap_factor)),
        moe_a2a_pattern=moe_a2a_pattern,
        mla_mode=mla_mode,
        inc_enabled=bool(cfg.get("inc_enabled", _defaults.inc_enabled)),
    )


def load_framework_spec(path: str | Path) -> FrameworkSpec:
    """Load FrameworkSpec from a JSON file."""
    cfg = _load_json(path)
    return framework_spec_from_json_dict(cfg)
