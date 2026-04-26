import json
from pathlib import Path
from typing import Any, Dict

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
          "n_EP_collectives": 1,
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
    )

    return TuningSpec(
        n_TP_collectives=int(cfg.get("n_TP_collectives", 2)),
        n_EP_collectives=int(cfg.get("n_EP_collectives", 1)),
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
    )


def load_tuning_spec(path: str | Path) -> TuningSpec:
    cfg = _load_json(path)
    return tuning_spec_from_json_dict(cfg)
