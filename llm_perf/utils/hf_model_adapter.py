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


_MOE_COUNT_KEYS = ("num_experts", "n_routed_experts", "num_local_experts")
_MOE_ACTIVE_KEYS = ("num_experts_per_tok", "experts_per_token")
_MLA_KEYS = ("kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim")


def _first(tc: Dict[str, Any], keys) -> Optional[int]:
    for k in keys:
        if tc.get(k) is not None:
            return int(tc[k])
    return None


def _extract_moe_dict(cfg: Dict[str, Any], L: int) -> Optional[Dict[str, Any]]:
    """
    If the text model is a mixture of experts, build the llm_perf 'moe' dict:

        {"n_experts": ..., "k_active": ..., "I_moe": ..., "n_moe_layers": ...}

    Config styles covered:
      * Qwen3:        num_experts, num_experts_per_tok, moe_intermediate_size
      * DeepSeek-V3:  n_routed_experts, num_experts_per_tok, moe_intermediate_size,
                      n_shared_experts (always active, same size as a routed
                      expert: added to k_active), first_k_dense_replace (the
                      first k layers are dense: n_moe_layers = L - k)
      * Mixtral, gpt-oss: num_local_experts, num_experts_per_tok or
                      experts_per_token; the experts use `intermediate_size`

    A config with no expert count is dense (returns None). A config that has
    an expert count but lacks the active count or an expert width is an
    error, not a dense model.
    """
    tc = _text_cfg(cfg)
    n_experts = _first(tc, _MOE_COUNT_KEYS)
    if n_experts is None or n_experts <= 1:
        return None
    k_active = _first(tc, _MOE_ACTIVE_KEYS)
    I_moe = _first(tc, ("moe_intermediate_size", "intermediate_size"))
    if k_active is None or I_moe is None:
        raise ValueError(
            "HF config describes a mixture of experts "
            f"({[k for k in _MOE_COUNT_KEYS if k in tc]}) but lacks "
            f"{'the experts-per-token count' if k_active is None else 'an expert width'}; "
            "refusing to convert it as a dense model"
        )
    k_active += int(tc.get("n_shared_experts") or 0)
    n_dense = int(tc.get("first_k_dense_replace") or 0)
    return {
        "n_experts": n_experts,
        "k_active": k_active,
        "I_moe": I_moe,
        "n_moe_layers": max(0, L - n_dense),
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

    out = {
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
    # Separate input-embedding and LM-head matrices: memory holds both.
    tie = tc.get("tie_word_embeddings", cfg.get("tie_word_embeddings"))
    if tie is not None:
        out["tied_embeddings"] = bool(tie)
    # Heads wider than H / n_q (Qwen3-VL): the KV cache follows head_dim.
    head_dim = tc.get("head_dim")
    if head_dim is not None and int(head_dim) * n_q != H:
        out["head_dim"] = int(head_dim)
    # Sliding-window layers (gpt-oss): only when the config says which layers
    # use the window; a bare `sliding_window` value is often unused.
    layer_types = tc.get("layer_types") or []
    n_win = sum(1 for t in layer_types if "sliding" in str(t))
    if n_win and tc.get("sliding_window"):
        out["attn_window"] = int(tc["sliding_window"])
        out["n_window_layers"] = n_win
    # Latent attention (DeepSeek-V3 style) is not converted: the result
    # would price a per-head key/value cache the model does not have.
    if any(tc.get(k) is not None for k in _MLA_KEYS):
        import warnings
        warnings.warn(
            "HF config uses latent attention (kv_lora_rank / qk_*_head_dim); the adapter does not "
            "convert it, so add an 'mla' block to the model JSON by hand before using it",
            stacklevel=2,
        )
    return out


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
