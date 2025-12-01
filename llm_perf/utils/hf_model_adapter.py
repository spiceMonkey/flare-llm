# llm_perf/utils/hf_model_adapter.py

"""
HuggingFace config -> llm_perf model JSON adapter.

This module converts a HuggingFace-style config.json into the canonical
llm_perf.model.v1 JSON format used by llm_perf.io.model_loaders.

Typical usage:

    from pathlib import Path
    from llm_perf.utils import convert_hf_config_to_model_json
    from llm_perf.io import load_model_spec

    hf_config = Path("external/Qwen/Qwen3-VL-235B/config.json")
    out_json  = Path("llm_perf/database/llm.model/qwen3_vl_235b.json")

    convert_hf_config_to_model_json(
        hf_config_path=hf_config,
        out_path=out_json,
        name_override="Qwen3-VL-235B",
        bytes_per_param_override=None,  # let adapter infer (e.g. FP8)
        L_override=None,               # or set manually if needed
        overwrite=True,
    )

    spec = load_model_spec(out_json)

The adapter is designed to handle:
  * Plain LLaMA/Qwen-style LLM configs (flat)
  * Qwen3-VL-style multimodal configs where the text LLM lives under
    `text_config`
  * FP8 / quantization hints in `quantization_config`
  * MoE hints (num_experts, num_experts_per_tok, moe_intermediate_size)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Basic JSON loading with helpful error messages
# ---------------------------------------------------------------------------

def _load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"JSON file is empty: {path}")

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"File is not valid JSON: {path}") from e


# ---------------------------------------------------------------------------
# Helpers: “text_config” vs flat configs
# ---------------------------------------------------------------------------

def _text_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    For multimodal configs (e.g. Qwen3-VL), the language model config
    is nested under 'text_config'. For plain LLMs, everything is top-level.
    """
    tc = cfg.get("text_config")
    if isinstance(tc, dict):
        return tc
    return cfg


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------

def _extract_num_layers(cfg: Dict[str, Any]) -> int:
    """
    Infer the number of transformer layers (L) from common keys.

    For multimodal models, we look in text_config first, then top-level.
    """
    tc = _text_cfg(cfg)

    candidate_keys = [
        "num_hidden_layers",
        "n_layer",
        "num_layers",
        "n_layers",
        # some enc/dec or text-specific variants
        "num_hidden_layers_text",
        "num_layers_text",
    ]

    for key in candidate_keys:
        if key in tc:
            try:
                return int(tc[key])
            except (TypeError, ValueError):
                pass

    # As a fallback, try the same keys on the top-level cfg
    for key in candidate_keys:
        if key in cfg:
            try:
                return int(cfg[key])
            except (TypeError, ValueError):
                pass

    raise KeyError(
        "HF config is missing a recognizable layer-count key. "
        "Checked (in text_config/top-level): "
        + ", ".join(candidate_keys)
    )


def _extract_intermediate_dim(cfg: Dict[str, Any]) -> int:
    """
    Get dense FFN intermediate dimension for the text model.

    Common keys:
      * intermediate_size  (LLaMA/Qwen/Qwen3-VL)
      * ffn_hidden_size
    """
    tc = _text_cfg(cfg)
    if "intermediate_size" in tc:
        return int(tc["intermediate_size"])
    if "ffn_hidden_size" in tc:
        return int(tc["ffn_hidden_size"])

    raise KeyError(
        "HF config missing 'intermediate_size' / 'ffn_hidden_size' "
        "in text_config or top-level."
    )


def _extract_heads(cfg: Dict[str, Any]) -> tuple[int, int]:
    """
    Return (n_q, n_kv) for the text model.

    Qwen3-VL and many others use:
      * num_attention_heads
      * num_key_value_heads (for GQA/MQA; if missing, assume n_kv = n_q)
    """
    tc = _text_cfg(cfg)

    if "num_attention_heads" not in tc:
        raise KeyError("HF config missing 'num_attention_heads' in text_config/top-level.")

    n_q = int(tc["num_attention_heads"])
    n_kv = int(tc.get("num_key_value_heads", n_q))
    return n_q, n_kv


def _extract_vocab_size(cfg: Dict[str, Any]) -> int:
    """
    Try to get vocab size from text_config or top-level.
    """
    tc = _text_cfg(cfg)
    for key in ["vocab_size", "pad_vocab_size_to"]:
        if key in tc:
            try:
                return int(tc[key])
            except (TypeError, ValueError):
                pass

    for key in ["vocab_size", "pad_vocab_size_to"]:
        if key in cfg:
            try:
                return int(cfg[key])
            except (TypeError, ValueError):
                pass

    raise KeyError("Could not find vocab size in HF config (expected 'vocab_size').")


def _extract_max_seq_len(cfg: Dict[str, Any]) -> int:
    """
    Try to get maximum sequence length, preferring the text sub-config.
    """
    tc = _text_cfg(cfg)

    candidate_keys = [
        "max_position_embeddings",
        "max_sequence_length",
        "seq_length",
        "n_positions",
        "max_seq_len",
    ]

    for key in candidate_keys:
        if key in tc:
            try:
                return int(tc[key])
            except (TypeError, ValueError):
                pass

    for key in candidate_keys:
        if key in cfg:
            try:
                return int(cfg[key])
            except (TypeError, ValueError):
                pass

    # Fallback default if nothing obvious is available
    return 2048


def _infer_bytes_per_param(cfg: Dict[str, Any]) -> int:
    """
    Try to infer bytes-per-param from HF config.

    Preference order:
      1) quantization_config.quant_method == "fp8" -> 1 byte
      2) dtype hints from text_config / top-level:
           'bfloat16' / 'bf16' -> 2
           'float16' / 'fp16'  -> 2
           'float32' / 'fp32'  -> 4
           'float8' / 'fp8'    -> 1
      3) default: 2 (bf16-like)
    """
    # Quantization hints (e.g. fp8)
    qcfg = cfg.get("quantization_config")
    if isinstance(qcfg, dict):
        qmethod = str(qcfg.get("quant_method", "")).lower()
        if "fp8" in qmethod or "float8" in qmethod:
            return 1

    # Dtype hints
    tc = _text_cfg(cfg)
    dtype = cfg.get("torch_dtype") or cfg.get("dtype") or tc.get("dtype")

    if isinstance(dtype, str):
        d = dtype.lower()
        if "bfloat16" in d or "bf16" in d:
            return 2
        if "float16" in d or "fp16" in d or "half" in d:
            return 2
        if "float32" in d or "fp32" in d:
            return 4
        if "float8" in d or "fp8" in d:
            return 1

    # Fallback
    return 2


def _extract_hidden_size(cfg: Dict[str, Any]) -> int:
    """
    Extract hidden size H for the text model.

    Common keys:
      * hidden_size
      * d_model
    """
    tc = _text_cfg(cfg)
    if "hidden_size" in tc:
        return int(tc["hidden_size"])
    if "d_model" in tc:
        return int(tc["d_model"])

    # Try top-level as a fallback
    if "hidden_size" in cfg:
        return int(cfg["hidden_size"])
    if "d_model" in cfg:
        return int(cfg["d_model"])

    raise KeyError("HF config missing 'hidden_size' / 'd_model' for text model.")


def _extract_moe_dict(cfg: Dict[str, Any], L: int) -> Optional[Dict[str, Any]]:
    """
    If the text model is MoE, build the llm_perf 'moe' dict:

        {
          "n_experts": ...,
          "k_active": ...,
          "I_moe": ...,
          "n_moe_layers": L
        }

    Qwen3-VL text_config typically has:
      * num_experts
      * num_experts_per_tok
      * moe_intermediate_size

    For now, we assume all L layers are MoE (n_moe_layers = L).
    This can be refined later if per-layer patterns are known.
    """
    tc = _text_cfg(cfg)

    required_keys = ("num_experts", "num_experts_per_tok", "moe_intermediate_size")
    if not all(k in tc for k in required_keys):
        return None

    try:
        n_experts = int(tc["num_experts"])
        k_active = int(tc["num_experts_per_tok"])
        I_moe = int(tc["moe_intermediate_size"])
    except (TypeError, ValueError):
        return None

    return {
        "n_experts": n_experts,
        "k_active": k_active,
        "I_moe": I_moe,
        "n_moe_layers": L,
    }


# ---------------------------------------------------------------------------
# Public conversion functions
# ---------------------------------------------------------------------------

def hf_config_to_llm_perf_model_dict(
    cfg: Dict[str, Any],
    name_override: Optional[str] = None,
    bytes_per_param_override: Optional[float] = None,
    L_override: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convert a HuggingFace config dict into the canonical llm_perf.model.v1 JSON dict.

    Output dict shape:

        {
          "schema": "llm_perf.model",
          "name": "...",
          "L": ...,
          "H": ...,
          "n_q": ...,
          "n_kv": ...,
          "I_dense": ...,
          "vocab_size": ...,
          "max_seq_len": ...,
          "bytes_per_param": ...,
          "moe": {...} or null
        }
    """
    if L_override is not None:
        L = int(L_override)
    else:
        L = _extract_num_layers(cfg)

    H = _extract_hidden_size(cfg)
    n_q, n_kv = _extract_heads(cfg)
    I_dense = _extract_intermediate_dim(cfg)
    vocab_size = _extract_vocab_size(cfg)
    max_seq_len = _extract_max_seq_len(cfg)

    if bytes_per_param_override is not None:
        b = float(bytes_per_param_override)
    else:
        b = _infer_bytes_per_param(cfg)

    # Derive a reasonable name if not overridden
    tc = _text_cfg(cfg)
    name = (
        name_override
        or tc.get("model_type")
        or cfg.get("model_type")
        or (cfg.get("architectures", ["hf_model"])[0] if cfg.get("architectures") else "hf_model")
    )

    moe = _extract_moe_dict(cfg, L)

    return {
        "schema": "llm_perf.model",
        "name": str(name),
        "L": L,
        "H": H,
        "n_q": n_q,
        "n_kv": n_kv,
        "I_dense": I_dense,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "bytes_per_param": b,
        "moe": moe,
    }


def convert_hf_config_to_model_json(
    hf_config_path: str | Path,
    out_path: str | Path,
    name_override: Optional[str] = None,
    bytes_per_param_override: Optional[float] = None,
    L_override: Optional[int] = None,
    overwrite: bool = False,
) -> Path:
    """
    Load a HuggingFace config.json, convert it to llm_perf.model.v1 JSON, and write it.

    Args:
        hf_config_path:
            Path to the HuggingFace config.json.
        out_path:
            Path where the llm_perf model JSON should be written
            (e.g. llm_perf/database/llm.model/qwen3_vl_235b.json).
        name_override:
            If not None, override the 'name' field.
        bytes_per_param_override:
            If not None, override bytes_per_param (e.g., 2.0, 4.0, 1.0).
        L_override:
            If not None, override the number of layers L.
        overwrite:
            If False and out_path exists, raises FileExistsError.

    Returns:
        Path to the written JSON file.
    """
    hf_cfg = _load_json(hf_config_path)
    model_dict = hf_config_to_llm_perf_model_dict(
        hf_cfg,
        name_override=name_override,
        bytes_per_param_override=bytes_per_param_override,
        L_override=L_override,
    )

    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(model_dict, f, indent=2, sort_keys=True)

    return out_path
