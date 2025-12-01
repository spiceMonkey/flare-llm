# llm_perf/io/model_loaders.py

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..specs.model_spec import LlmModelSpec, MoESpec
from ..utils import validate_positive_int_fields


def _load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_moe(moe_cfg: Optional[Dict[str, Any]]) -> Optional[MoESpec]:
    if moe_cfg is None:
        return None

    # Basic sanity checks for MoE fields: all must be integers > 1
    validate_positive_int_fields(
        moe_cfg,
        ["n_experts", "k_active", "I_moe"],
        prefix="MoE configuration",
    )

    if "n_moe_layers" in moe_cfg and moe_cfg["n_moe_layers"] is not None:
        validate_positive_int_fields(
            moe_cfg,
            ["n_moe_layers"],
            prefix="MoE configuration",
        )

    return MoESpec(
        n_experts=int(moe_cfg["n_experts"]),
        k_active=int(moe_cfg["k_active"]),
        I_moe=int(moe_cfg["I_moe"]),
        n_moe_layers=int(moe_cfg["n_moe_layers"])
        if "n_moe_layers" in moe_cfg and moe_cfg["n_moe_layers"] is not None
        else None,
    )


def model_spec_from_json_dict(cfg: Dict[str, Any]) -> LlmModelSpec:
    """
    Build LlmModelSpec from a config dict.

    Expected format:

        {
          "schema": "llm_perf.model",
          "name": "...",
          "L": 32,
          "H": 4096,
          "n_q": 32,
          "n_kv": 8,
          "I_dense": 14336,
          "vocab_size": 128256,
          "max_seq_len": 8192,
          "bytes_per_param": 2,
          "moe": { ... } | null
        }

    If 'schema' is missing, we still try to parse using the same keys.
    """
    schema = cfg.get("schema", "llm_perf.model")
    if not schema.startswith("llm_perf.model"):
        # Future: support HF configs or other schemas
        raise ValueError(f"Unsupported model schema: {schema}")

    validate_positive_int_fields(
        cfg,
        [
            "L",
            "H",
            "n_q",
            "n_kv",
            "I_dense",
            "vocab_size",
            "max_seq_len",
            "bytes_per_param",
        ],
        # bytes_per_param is often given as a float in JSON
        allow_float_for_int=True,
        prefix="model configuration",
    )

    moe = _parse_moe(cfg.get("moe"))

    return LlmModelSpec(
        name=str(cfg.get("name", "unnamed_model")),
        L=int(cfg["L"]),
        H=int(cfg["H"]),
        n_q=int(cfg["n_q"]),
        n_kv=int(cfg["n_kv"]),
        I_dense=int(cfg["I_dense"]),
        vocab_size=int(cfg["vocab_size"]),
        max_seq_len=int(cfg["max_seq_len"]),
        bytes_per_param=float(cfg["bytes_per_param"]),
        moe=moe,
    )


def load_model_spec(path: str | Path) -> LlmModelSpec:
    """
    Load LlmModelSpec from a JSON file.

    Example:
        model = load_model_spec("model.json")
    """
    cfg = _load_json(path)
    return model_spec_from_json_dict(cfg)
