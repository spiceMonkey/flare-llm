"""Programmatic helper to adapt HuggingFace config files into llm_perf model cards."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_perf.io import load_model_spec
from llm_perf.utils import convert_hf_config_to_model_json
from llm_perf.specs.model_spec import LlmModelSpec

HF_FILE_PATH = Path("archives/hf")
HF_FILE_NAME = "qwen3_vl_235b_a22b_thinking_fp8.json"
HF_SAMPLE = HF_FILE_PATH / HF_FILE_NAME

HF_ADAPTED_PATH = Path("llm_perf/database/model")
HF_ADAPTED_NAME = "qwen3_vl_235b_fp8.json"
HF_ADAPTED_OUTPUT = HF_ADAPTED_PATH / HF_ADAPTED_NAME

FFN_PROJECTION_FACTOR = 3  # gate + up + down projections in SiLU/GLU blocks


def estimate_total_model_size(model: LlmModelSpec) -> Tuple[int, float]:
    """Return (parameter_count, total_bytes) for the given model spec."""

    L = model.L
    H = model.H
    H_kv = model.H_kv()
    V = model.vocab_size
    b = model.bytes_per_param

    # Attention + projection weights across all layers
    attn_params = L * (H**2 + 3 * H * H_kv)

    router_params = 0
    if model.moe is not None:
        n_exp = max(1, model.moe.n_experts)
        n_moe_layers = model.moe.n_moe_layers or L
        n_moe_layers = max(0, min(L, n_moe_layers))
        n_dense_layers = max(0, L - n_moe_layers)
        ffn_moe = (
            FFN_PROJECTION_FACTOR
            * H
            * model.moe.I_moe
            * n_exp
            * n_moe_layers
        )
        ffn_dense = FFN_PROJECTION_FACTOR * H * model.I_dense * n_dense_layers
        router_params = H * n_exp * n_moe_layers
    else:
        ffn_moe = 0
        ffn_dense = FFN_PROJECTION_FACTOR * H * model.I_dense * L

    embed_params = V * H

    total_params = attn_params + ffn_dense + ffn_moe + router_params + embed_params
    total_bytes = total_params * b
    return int(total_params), float(total_bytes)


def format_bytes(num_bytes: float) -> str:
    """Pretty-print bytes using decimal units."""

    units = ["B", "KB", "MB", "GB", "TB", "PB", "EB"]
    value = num_bytes
    idx = 0
    while value >= 1000 and idx < len(units) - 1:
        value /= 1000
        idx += 1
    return f"{value:.2f} {units[idx]}"

def main():

    converted_spec = None
    if HF_SAMPLE.exists():
        print(f"Adapting HuggingFace config → llm_perf JSON: {HF_SAMPLE}")
        converted_path = convert_hf_config_to_model_json(
            hf_config_path=HF_SAMPLE,
            out_path=HF_ADAPTED_OUTPUT,
            overwrite=True,
        )
        print(f"Wrote adapted model card to {converted_path}")
        # Load the converted JSON using the normal llm_perf loader
        converted_spec = load_model_spec(converted_path)
        print("[OK] Loaded LlmModelSpec via llm_perf.io.load_model_spec")
        print(f"Converted spec name: {converted_spec.name}")
        # Print selected fields for sanity check
        print("\n=== LlmModelSpec Summary ===")
        print(f"  name:            {converted_spec.name}")
        print(f"  L (layers):      {converted_spec.L}")
        print(f"  H (hidden):      {converted_spec.H}")
        print(f"  n_q (heads):     {converted_spec.n_q}")
        print(f"  n_kv (kv heads): {converted_spec.n_kv}")
        print(f"  I_dense:         {converted_spec.I_dense}")
        print(f"  vocab_size:      {converted_spec.vocab_size}")
        print(f"  max_seq_len:     {converted_spec.max_seq_len}")
        print(f"  bytes_per_param: {converted_spec.bytes_per_param}")
        print(f"  d_head:          {converted_spec.d_head():.2f}")
        print(f"  H_kv:            {converted_spec.H_kv():.2f}")
        print(f"  MoE spec:        {converted_spec.moe}")

        total_params, total_bytes = estimate_total_model_size(converted_spec)
        print("\n=== Model Footprint ===")
        print(f"  total params:    {total_params:,}")
        print(f"  model size:      {format_bytes(total_bytes)} ({total_bytes/1e9:.2f} GB)")

    else:
        print("No HF config found at", HF_SAMPLE)

    print("\n✓ Test complete. Compare these values with your original HF config.json.")


if __name__ == "__main__":
    main()


