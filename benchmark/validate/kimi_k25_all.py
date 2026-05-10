#!/usr/bin/env python3
"""Kimi-K2.5 across all single-island stacks — framework vs InferenceX.

Thin wrapper around `coverage_sweep.py --model "Kimi-K2.5"` for
discoverability. Sweeps all (hardware, framework) cells where the
Kimi-K2.5 model spec applies and the partition fits on one NVLink
island. Generates per-cell plots by default.

Architecture caveats (from the model spec): DeepSeek-V3 architecture
(MLA + MoE), modeled via n_kv=5 approximation; vision tower (vt_*)
parameters not modeled — language-only path.

Usage:
    python benchmark/validate/kimi_k25_all.py
    python benchmark/validate/kimi_k25_all.py --framework dynamo-trt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from coverage_sweep import main as sweep_main


def main() -> int:
    args = list(sys.argv[1:])
    if not any(a == "--model" or a.startswith("--model=") for a in args):
        args.extend(["--model", "Kimi-K2.5"])
    if "--no-plot" in args:
        args.remove("--no-plot")
    elif "--plot" not in args:
        args.append("--plot")
    sys.argv = [sys.argv[0]] + args
    return sweep_main()


if __name__ == "__main__":
    sys.exit(main())
