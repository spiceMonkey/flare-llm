
import json
from pathlib import Path
from typing import Any, Dict

from ..specs.disagg_spec import DisaggSpec


def disagg_spec_from_json_dict(cfg: Dict[str, Any]) -> DisaggSpec:
    """Build DisaggSpec from a config dict.

    JSON schema: llm_perf.disagg
    """
    schema = cfg.get("schema", "llm_perf.disagg")
    if not schema.startswith("llm_perf.disagg"):
        raise ValueError(f"Unsupported disagg schema: {schema}")

    return DisaggSpec(
        disaggregated=bool(cfg.get("disaggregated", False)),
        colo_alpha_us=float(cfg.get("colo_alpha_us", 0.0)),
        colo_repack_GBps=float(cfg.get("colo_repack_GBps", 0.0)),
        colo_repack_eta=float(cfg.get("colo_repack_eta", 1.0)),
        inter_alpha_us=float(cfg.get("inter_alpha_us", 0.0)),
        inter_bandwidth_GBps=float(cfg.get("inter_bandwidth_GBps", 0.0)),
        N_WR=int(cfg.get("N_WR", 0)),
        tau_WR_us=float(cfg.get("tau_WR_us", 0.0)),
        overlap_rho_KV=float(cfg.get("overlap_rho_KV", 0.0)),
        repack_GBps=float(cfg.get("repack_GBps", 0.0)),
        repack_eta=float(cfg.get("repack_eta", 1.0)),
    )


def load_disagg_spec(path: str | Path) -> DisaggSpec:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return disagg_spec_from_json_dict(cfg)
