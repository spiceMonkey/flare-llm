import json
from pathlib import Path
from typing import Any, Dict

from ..specs.partition_spec import PartitionSpec
from ..utils import validate_positive_int_fields


def _load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def partition_spec_from_json_dict(cfg: Dict[str, Any]) -> PartitionSpec:
    """
    Build PartitionSpec from a config dict.

    partition.json format:

        {
          "schema": "llm_perf.partition",
          "PP": 4,
          "TP": 4,
          "EP": 8,
          "SP": 2
        }
    """
    schema = cfg.get("schema", "llm_perf.partition")
    if not schema.startswith("llm_perf.partition"):
        raise ValueError(f"Unsupported partition schema: {schema}")

    # Ensure all parallelism dimensions are integers ≥ 1
    validate_positive_int_fields(
        cfg,
        ["PP", "TP", "EP", "SP"],
        prefix="partition configuration",
    )

    return PartitionSpec(
        PP=int(cfg["PP"]),
        TP=int(cfg["TP"]),
        EP=int(cfg["EP"]),
        SP=int(cfg["SP"]),
    )


def load_partition_spec(path: str | Path) -> PartitionSpec:
    cfg = _load_json(path)
    return partition_spec_from_json_dict(cfg)
