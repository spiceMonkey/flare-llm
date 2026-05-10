#!/usr/bin/env python3
"""Qwen-3.5-397B-A17B across all single-island stacks — framework vs InferenceX.

Thin wrapper around `coverage_sweep.py --model "Qwen-3.5-397B-A17B"` for
discoverability. Sweeps all (hardware, framework) cells where the
Qwen-3.5-397B-A17B model spec applies and the partition fits on one
NVLink island. Generates per-cell plots by default.

Architecture caveat (from the model spec): HYBRID linear/full attention
model. layer_types alternates 3:1 linear-attn to full-attn (15 full +
45 linear of 60 total). The framework treats EVERY layer as full
attention (O(S²) compute, full KV cache), which OVER-estimates
attention FLOPs by ~4× and KV cache by ~4× at long context. Hybrid
attention + linear-attention stacks are a documented gap; predicted
decode latency is an upper bound on the real workload.

Usage:
    python benchmark/validate/qwen35_397b_a17b_all.py
    python benchmark/validate/qwen35_397b_a17b_all.py --hardware b300
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from coverage_sweep import main as sweep_main


def main() -> int:
    args = list(sys.argv[1:])
    if not any(a == "--model" or a.startswith("--model=") for a in args):
        args.extend(["--model", "Qwen-3.5-397B-A17B"])
    if "--no-plot" in args:
        args.remove("--no-plot")
    elif "--plot" not in args:
        args.append("--plot")
    sys.argv = [sys.argv[0]] + args
    return sweep_main()


if __name__ == "__main__":
    sys.exit(main())
