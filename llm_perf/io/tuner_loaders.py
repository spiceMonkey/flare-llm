import json
from pathlib import Path
from typing import Any, Dict

from ..specs.tuner_spec import MemoryPlacementSpec, TuningSpec
from ..utils import (
    validate_positive_int_fields,
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

        }
    """
    schema = cfg.get("schema", "llm_perf.tuner")
    if not schema.startswith("llm_perf.tuner"):
        raise ValueError(f"Unsupported tuner schema: {schema}")

    # Positive integer checks
    validate_positive_int_fields(
        cfg,
        ["S_decode"],
        prefix="tuning configuration",
    )

    # Framework-axis fields moved to FrameworkSpec across two phases.
    # Tuner JSONs that still set them are rejected with a clear migration
    # hint pointing at `load_framework_from_db('<stack>')`.
    _moved_to_framework = (
        # Phase B (host overhead + execution mode):
        "kernels_per_layer_compute", "kernels_per_collective_call",
        "kernels_per_pp_hop", "kernel_launch_us", "kernel_overlap_factor",
        "moe_a2a_pattern", "mla_mode", "inc_enabled",
        "t_serving_per_seq_us",
        # Phase E (collective dispatch + comm/compute overlap):
        "tp_algorithm_decode", "tp_algorithm_prefill",
        "ep_algorithm_decode", "ep_algorithm_prefill",
        "tp_algorithm", "ep_algorithm",  # deprecated single-knob aliases
        "torus_algorithm",
        "n_TP_collectives", "n_EP_collectives", "n_SP_collectives",
        "overlap_factor",      # legacy name; new name is `comm_overlap_factor`
        "comm_overlap_factor", # framework-axis
    )
    _moved_to_device = (
        # Phase F (chip-side calibration curves):
        "tensor_core_efficiency",  # → DeviceSpec.tensor_core_efficiency
        "bw_efficiency",           # → DeviceSpec.bw_efficiency
    )
    leaked_fw = [f for f in _moved_to_framework if f in cfg]
    if leaked_fw:
        raise ValueError(
            f"tuning configuration: fields {leaked_fw} were moved to "
            f"FrameworkSpec — load a framework JSON via "
            f"`load_framework_from_db('<stack>')` instead. See "
            f"`llm_perf/database/framework/` for available stacks."
        )
    leaked_dev = [f for f in _moved_to_device if f in cfg]
    if leaked_dev:
        raise ValueError(
            f"tuning configuration: fields {leaked_dev} were moved to "
            f"DeviceSpec (sibling of peak_flops_TF / peak_flops_eta and "
            f"per-tier eta_beta). Add them to the device block of the "
            f"system JSON instead. See llm_perf/database/system/."
        )

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
        S_decode=int(cfg.get("S_decode", 2048)),
        B_decode=int(cfg.get("B_decode", 1)),
        S_input=int(cfg.get("S_input", 0)),
        B_prefill=int(cfg.get("B_prefill", 1)),
        chunk_size=int(cfg.get("chunk_size", 0)),
        placement=placement,
        n_tok_draft=int(cfg.get("n_tok_draft", _defaults.n_tok_draft)),
        p_accept=float(cfg.get("p_accept", _defaults.p_accept)),
    )


def load_tuning_spec(path: str | Path) -> TuningSpec:
    cfg = _load_json(path)
    return tuning_spec_from_json_dict(cfg)
