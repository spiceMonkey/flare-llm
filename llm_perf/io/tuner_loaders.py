import json
from pathlib import Path
from typing import Any, Dict

from ..specs.tuner_spec import TuningSpec
from ..utils import (
    validate_positive_int_fields,
    validate_nonnegative_int_fields,
    validate_nonnegative_float_fields,
    validate_positive_float_fields,
    TP_ALGORITHMS,
    EP_ALGORITHMS,
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

          "c_act": 5.0,
        
          "overlap_factor": 0.3,

        }
    """
    schema = cfg.get("schema", "llm_perf.tuner")
    if not schema.startswith("llm_perf.tuner"):
        raise ValueError(f"Unsupported tuner schema: {schema}")

    tp_algorithm = str(cfg.get("tp_algorithm", "ring")).lower()
    ep_algorithm = str(cfg.get("ep_algorithm", "ring")).lower()

    if tp_algorithm not in TP_ALGORITHMS:
        raise ValueError(f"Unsupported tp_algorithm: {tp_algorithm!r}")
    if ep_algorithm not in EP_ALGORITHMS:
        raise ValueError(f"Unsupported ep_algorithm: {ep_algorithm!r}")

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

    # Nonnegative floats: c_act and overlap_factor
    validate_nonnegative_float_fields(
        cfg,
        ["c_act", "overlap_factor"],
        prefix="tuning configuration",
    )

    # Additional check for overlap_factor <= 1.0
    if float(cfg.get("overlap_factor", 0.0)) > 1.0:
        raise ValueError(f"overlap_factor must be <= 1.0, got {cfg['overlap_factor']}")

    return TuningSpec(
        n_TP_collectives=int(cfg.get("n_TP_collectives", 2)),
        n_EP_collectives=int(cfg.get("n_EP_collectives", 1)),
        n_SP_collectives=int(cfg.get("n_SP_collectives", 1)),
        overlap_factor=float(cfg.get("overlap_factor", 0.0)),
        S_decode=int(cfg.get("S_decode", 2048)),
        tp_algorithm=tp_algorithm,
        ep_algorithm=ep_algorithm,
        c_act=float(cfg.get("c_act", 5.0)),
    )


def load_tuning_spec(path: str | Path) -> TuningSpec:
    cfg = _load_json(path)
    return tuning_spec_from_json_dict(cfg)
