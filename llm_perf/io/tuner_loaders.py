import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..specs.tuner_spec import MemoryPlacementSpec, TuningSpec
from ..utils import (
    validate_positive_int_fields,
    validate_nonnegative_int_fields,
    validate_nonnegative_float_fields,
    validate_positive_float_fields,
    TP_ALGORITHMS,
    EP_ALGORITHMS,
    TORUS_ALGORITHMS,
)


def _load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def tuning_spec_from_json_dict(cfg: Dict[str, Any]) -> TuningSpec:
    """
    Build TuningSpec from a config dict.

    tuner.json format:

        {
          "schema": "llm_perf.tuner",

          "S_decode": 4096,

          "tp_algorithm": "ring",
          "ep_algorithm": "tree"

          "n_TP_collectives": 2,
          "n_EP_collectives": 2,
          "n_SP_collectives": 1,

          "overlap_factor": 0.3,

        }
    """
    schema = cfg.get("schema", "llm_perf.tuner")
    if not schema.startswith("llm_perf.tuner"):
        raise ValueError(f"Unsupported tuner schema: {schema}")

    # Legacy single-knob fields (deprecated; preserved for back-compat).
    tp_algorithm = str(cfg.get("tp_algorithm", "ring")).lower()
    ep_algorithm = str(cfg.get("ep_algorithm", "ring")).lower()
    torus_algorithm = str(cfg.get("torus_algorithm", "ring")).lower()

    # Per-phase × per-collective fields (new in PR2.4). When omitted, fall
    # back to the legacy single-knob value.
    tp_algorithm_decode = str(cfg.get("tp_algorithm_decode", tp_algorithm)).lower()
    tp_algorithm_prefill = str(cfg.get("tp_algorithm_prefill", tp_algorithm)).lower()
    ep_algorithm_decode = str(cfg.get("ep_algorithm_decode", ep_algorithm)).lower()
    ep_algorithm_prefill = str(cfg.get("ep_algorithm_prefill", ep_algorithm)).lower()

    for name, val in [
        ("tp_algorithm", tp_algorithm),
        ("tp_algorithm_decode", tp_algorithm_decode),
        ("tp_algorithm_prefill", tp_algorithm_prefill),
    ]:
        if val not in TP_ALGORITHMS:
            raise ValueError(f"Unsupported {name}: {val!r}; allowed: {list(TP_ALGORITHMS)}")
    for name, val in [
        ("ep_algorithm", ep_algorithm),
        ("ep_algorithm_decode", ep_algorithm_decode),
        ("ep_algorithm_prefill", ep_algorithm_prefill),
    ]:
        if val not in EP_ALGORITHMS:
            raise ValueError(f"Unsupported {name}: {val!r}; allowed: {list(EP_ALGORITHMS)}")
    if torus_algorithm not in TORUS_ALGORITHMS:
        raise ValueError(
            f"Unsupported torus_algorithm: {torus_algorithm!r}; "
            f"allowed: {list(TORUS_ALGORITHMS)}"
        )

    # Positive integer checks
    validate_positive_int_fields(
        cfg,
        ["S_decode"],
        prefix="tuning configuration",
    )

    # Nonnegative integer checks
    validate_nonnegative_int_fields(
        cfg,
        [
            "n_TP_collectives",
            "n_EP_collectives",
            "n_SP_collectives",
        ],
        prefix="tuning configuration",
    )

    # Nonnegative floats: overlap_factor
    validate_nonnegative_float_fields(
        cfg,
        ["overlap_factor"],
        prefix="tuning configuration",
    )

    # Additional check for overlap_factor <= 1.0
    if float(cfg.get("overlap_factor", 0.0)) > 1.0:
        raise ValueError(f"overlap_factor must be <= 1.0, got {cfg['overlap_factor']}")

    # SW-overhead fields (kernel_launch_overhead.md §5). All optional;
    # defaults disable the term so legacy tuner JSONs keep their meaning.
    sw_int_fields = [f for f in ("kernels_per_layer_compute", "kernels_per_collective_call", "kernels_per_pp_hop") if f in cfg]
    if sw_int_fields:
        validate_nonnegative_int_fields(cfg, sw_int_fields, prefix="tuning configuration")
    sw_float_fields = [f for f in ("kernel_launch_us", "sw_overlap_factor") if f in cfg]
    if sw_float_fields:
        validate_nonnegative_float_fields(cfg, sw_float_fields, prefix="tuning configuration")
    if float(cfg.get("sw_overlap_factor", 1.0)) > 1.0:
        raise ValueError(
            f"sw_overlap_factor must be <= 1.0, got {cfg['sw_overlap_factor']}"
        )

    eta_TC_cfg = cfg.get("tensor_core_efficiency", None)
    if eta_TC_cfg is None:
        eta_TC: Optional[Dict[int, float]] = None
    else:
        if not isinstance(eta_TC_cfg, dict):
            raise ValueError(
                "tuning configuration: 'tensor_core_efficiency' must be an "
                f"object mapping mb→efficiency, got {eta_TC_cfg!r}"
            )
        eta_TC = {}
        for k, v in eta_TC_cfg.items():
            mb = int(k)
            eta = float(v)
            if mb < 1:
                raise ValueError(
                    f"tensor_core_efficiency: mb keys must be >= 1, got {mb}"
                )
            if not (0.0 <= eta <= 1.0):
                raise ValueError(
                    f"tensor_core_efficiency[{mb}]: efficiency must be in [0, 1], got {eta}"
                )
            eta_TC[mb] = eta

    # MemoryPlacementSpec block (sram.md §1.3 Operator-Specified policy).
    # JSON shape:  "placement": {"weights_tier": "sram", "kv_tier": "auto"}
    # Both fields default to "auto" → greedy fastest-first.
    placement_cfg = cfg.get("placement", {})
    if not isinstance(placement_cfg, dict):
        raise ValueError(
            f"tuning configuration: 'placement' must be an object, got {placement_cfg!r}"
        )
    placement = MemoryPlacementSpec(
        weights_tier=str(placement_cfg.get("weights_tier", "auto")),
        kv_tier=str(placement_cfg.get("kv_tier", "auto")),
        auto_priority=str(placement_cfg.get("auto_priority", "weights")),
    )

    # SW-overhead fields fall back to TuningSpec dataclass defaults when
    # absent from JSON — keeps a single source of truth for the production
    # baseline.
    _defaults = TuningSpec()
    return TuningSpec(
        n_TP_collectives=int(cfg.get("n_TP_collectives", 2)),
        n_EP_collectives=int(cfg.get("n_EP_collectives", 2)),
        n_SP_collectives=int(cfg.get("n_SP_collectives", 1)),
        overlap_factor=float(cfg.get("overlap_factor", 0.0)),
        S_decode=int(cfg.get("S_decode", 2048)),
        tp_algorithm=tp_algorithm,
        ep_algorithm=ep_algorithm,
        tp_algorithm_decode=tp_algorithm_decode,
        tp_algorithm_prefill=tp_algorithm_prefill,
        ep_algorithm_decode=ep_algorithm_decode,
        ep_algorithm_prefill=ep_algorithm_prefill,
        B_decode=int(cfg.get("B_decode", 1)),
        S_input=int(cfg.get("S_input", 0)),
        B_prefill=int(cfg.get("B_prefill", 1)),
        chunk_size=int(cfg.get("chunk_size", 0)),
        torus_algorithm=torus_algorithm,
        inc_enabled=bool(cfg.get("inc_enabled", True)),
        placement=placement,
        kernels_per_layer_compute=int(cfg.get("kernels_per_layer_compute", _defaults.kernels_per_layer_compute)),
        kernels_per_collective_call=int(cfg.get("kernels_per_collective_call", _defaults.kernels_per_collective_call)),
        kernels_per_pp_hop=int(cfg.get("kernels_per_pp_hop", _defaults.kernels_per_pp_hop)),
        kernel_launch_us=float(cfg.get("kernel_launch_us", _defaults.kernel_launch_us)),
        sw_overlap_factor=float(cfg.get("sw_overlap_factor", _defaults.sw_overlap_factor)),
        tensor_core_efficiency=eta_TC if eta_TC is not None else _defaults.tensor_core_efficiency,
    )


def load_tuning_spec(path: str | Path) -> TuningSpec:
    cfg = _load_json(path)
    return tuning_spec_from_json_dict(cfg)
