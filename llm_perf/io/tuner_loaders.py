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

    # Framework-axis fields (kernels_per_*, kernel_launch_us,
    # sw_overlap_factor, moe_a2a_pattern, mla_mode, inc_enabled,
    # t_serving_per_seq_us) moved to FrameworkSpec — see
    # `framework_spec.py`. Tuner JSONs that still set these fields will
    # be rejected with a clear error so users know to migrate to a
    # framework JSON.
    _moved_to_framework = (
        "kernels_per_layer_compute", "kernels_per_collective_call",
        "kernels_per_pp_hop", "kernel_launch_us", "sw_overlap_factor",
        "moe_a2a_pattern", "mla_mode", "inc_enabled",
        "t_serving_per_seq_us",
    )
    leaked = [f for f in _moved_to_framework if f in cfg]
    if leaked:
        raise ValueError(
            f"tuning configuration: fields {leaked} were moved to "
            f"FrameworkSpec — load a framework JSON via "
            f"`load_framework_from_db('<stack>')` instead. See "
            f"`llm_perf/database/framework/` for available stacks."
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

    eta_BW_cfg = cfg.get("bw_efficiency", None)
    if eta_BW_cfg is None:
        eta_BW: Optional[Dict[int, float]] = None
    else:
        if not isinstance(eta_BW_cfg, dict):
            raise ValueError(
                "tuning configuration: 'bw_efficiency' must be an "
                f"object mapping B→efficiency, got {eta_BW_cfg!r}"
            )
        eta_BW = {}
        for k, v in eta_BW_cfg.items():
            B = int(k)
            eta = float(v)
            if B < 1:
                raise ValueError(
                    f"bw_efficiency: B keys must be >= 1, got {B}"
                )
            if not (0.0 < eta <= 1.0):
                raise ValueError(
                    f"bw_efficiency[{B}]: efficiency must be in (0, 1], got {eta}"
                )
            eta_BW[B] = eta

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
        placement=placement,
        tensor_core_efficiency=eta_TC if eta_TC is not None else _defaults.tensor_core_efficiency,
        bw_efficiency=eta_BW if eta_BW is not None else _defaults.bw_efficiency,
        n_tok_draft=int(cfg.get("n_tok_draft", _defaults.n_tok_draft)),
        p_accept=float(cfg.get("p_accept", _defaults.p_accept)),
    )


def load_tuning_spec(path: str | Path) -> TuningSpec:
    cfg = _load_json(path)
    return tuning_spec_from_json_dict(cfg)
