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


# Whitelists for collective algorithm fields. Sourced from
# `core/primitives/dispatch.enumerate_options()` per-op return values
# plus the "auto" sentinel that triggers `optimize_collective_algorithms`.
TP_ALGORITHM_VALUES = ("ring", "tree", "tree_pipelined", "inc", "auto")
EP_ALGORITHM_VALUES = ("ring", "inc", "auto")
TORUS_ALGORITHM_VALUES = ("ring", "swing", "auto")

# Whitelists for attention dispatch + TP/EP physical overlay.
ATTENTION_MODE_VALUES = ("tp", "dp")
TP_EP_LAYOUT_VALUES = ("orthogonal", "co_located")


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
          "inc_enabled": true,

          "tp_algorithm_decode": "auto",
          "tp_algorithm_prefill": "auto",
          "ep_algorithm_decode": "auto",
          "ep_algorithm_prefill": "auto",
          "torus_algorithm": "auto",
          "n_TP_collectives": 2,
          "n_EP_collectives": 2,
          "n_SP_collectives": 1,
          "comm_overlap_factor": 0.0
        }

    All fields except `name` fall through to FrameworkSpec dataclass
    defaults when absent. Algorithm fields accept "auto" to trigger
    cost-model resolution via `optimize_collective_algorithms`.
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

    def _algo(field: str, default: str, allowed: tuple) -> str:
        v = str(cfg.get(field, default)).lower()
        if v not in allowed:
            raise ValueError(
                f"framework configuration: '{field}' must be one of {list(allowed)}, "
                f"got {v!r}"
            )
        return v

    tp_decode = _algo("tp_algorithm_decode", _defaults.tp_algorithm_decode, TP_ALGORITHM_VALUES)
    tp_prefill = _algo("tp_algorithm_prefill", _defaults.tp_algorithm_prefill, TP_ALGORITHM_VALUES)
    ep_decode = _algo("ep_algorithm_decode", _defaults.ep_algorithm_decode, EP_ALGORITHM_VALUES)
    ep_prefill = _algo("ep_algorithm_prefill", _defaults.ep_algorithm_prefill, EP_ALGORITHM_VALUES)
    torus_alg = _algo("torus_algorithm", _defaults.torus_algorithm, TORUS_ALGORITHM_VALUES)

    attention_mode = _algo("attention_mode", _defaults.attention_mode, ATTENTION_MODE_VALUES)
    if "layout" in cfg and "tp_ep_layout" not in cfg:
        raise ValueError(
            "framework configuration: 'layout' was renamed to 'tp_ep_layout' "
            "to make the TP/EP-overlay scope explicit. Update your JSON."
        )
    tp_ep_layout = _algo("tp_ep_layout", _defaults.tp_ep_layout, TP_EP_LAYOUT_VALUES)

    overlap = float(cfg.get("comm_overlap_factor", _defaults.comm_overlap_factor))
    if not (0.0 <= overlap <= 1.0):
        raise ValueError(
            f"framework configuration: 'comm_overlap_factor' must be in [0, 1], got {overlap}"
        )
    sw_overlap = float(cfg.get("sw_overlap_factor", _defaults.sw_overlap_factor))
    if not (0.0 <= sw_overlap <= 1.0):
        raise ValueError(
            f"framework configuration: 'sw_overlap_factor' must be in [0, 1], got {sw_overlap}"
        )

    return FrameworkSpec(
        name=str(cfg.get("name", "unnamed_framework")),
        c_serving_per_seq_us=float(cfg.get("c_serving_per_seq_us", _defaults.c_serving_per_seq_us)),
        kernel_launch_us=float(cfg.get("kernel_launch_us", _defaults.kernel_launch_us)),
        kernels_per_layer_compute=int(cfg.get("kernels_per_layer_compute", _defaults.kernels_per_layer_compute)),
        kernels_per_collective_call=int(cfg.get("kernels_per_collective_call", _defaults.kernels_per_collective_call)),
        kernels_per_pp_hop=int(cfg.get("kernels_per_pp_hop", _defaults.kernels_per_pp_hop)),
        sw_overlap_factor=sw_overlap,
        moe_a2a_pattern=moe_a2a_pattern,
        mla_mode=mla_mode,
        inc_enabled=bool(cfg.get("inc_enabled", _defaults.inc_enabled)),
        tp_algorithm_decode=tp_decode,
        tp_algorithm_prefill=tp_prefill,
        ep_algorithm_decode=ep_decode,
        ep_algorithm_prefill=ep_prefill,
        torus_algorithm=torus_alg,
        n_TP_collectives=int(cfg.get("n_TP_collectives", _defaults.n_TP_collectives)),
        n_EP_collectives=int(cfg.get("n_EP_collectives", _defaults.n_EP_collectives)),
        n_SP_collectives=int(cfg.get("n_SP_collectives", _defaults.n_SP_collectives)),
        attention_mode=attention_mode,
        tp_ep_layout=tp_ep_layout,
        comm_overlap_factor=overlap,
    )


def load_framework_spec(path: str | Path) -> FrameworkSpec:
    """Load FrameworkSpec from a JSON file."""
    cfg = _load_json(path)
    return framework_spec_from_json_dict(cfg)
