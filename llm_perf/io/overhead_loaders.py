
import json
from pathlib import Path
from typing import Any, Dict

from ..specs.overhead_spec import OverheadSpec


def overhead_spec_from_json_dict(cfg: Dict[str, Any]) -> OverheadSpec:
    """Build OverheadSpec from a config dict.

    JSON schema: llm_perf.overhead
    """
    schema = cfg.get("schema", "llm_perf.overhead")
    if not schema.startswith("llm_perf.overhead"):
        raise ValueError(f"Unsupported overhead schema: {schema}")

    return OverheadSpec(
        t_sched_us=float(cfg.get("t_sched_us", 0.0)),
        t_tok_us=float(cfg.get("t_tok_us", 0.0)),
        t_graph_us=float(cfg.get("t_graph_us", 0.0)),
        t_sample_us=float(cfg.get("t_sample_us", 0.0)),
        t_detok_us=float(cfg.get("t_detok_us", 0.0)),
    )


def load_overhead_spec(path: str | Path) -> OverheadSpec:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return overhead_spec_from_json_dict(cfg)
