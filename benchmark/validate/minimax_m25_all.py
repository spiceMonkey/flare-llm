#!/usr/bin/env python3
"""MiniMax-M2.5 across all single-island stacks — framework vs InferenceX.

Thin wrapper around `coverage_sweep.py --model "MiniMax-M2.5"` for
discoverability. Sweeps all (hardware, framework) cells where the
MiniMax-M2.5 model spec applies and the partition fits on one NVLink
island. Generates per-cell plots by default.

Architecture notes (from the model spec): standard GQA attention
(n_kv=8), uniform full attention every layer, no shared expert
(shared_intermediate_size=0). MTP modules (num_mtp_modules=3) not
included in T_theta — minor under-count on per-device parameter
footprint.

Usage:
    python benchmark/validate/minimax_m25_all.py
    python benchmark/validate/minimax_m25_all.py --hardware b200
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from coverage_sweep import main as sweep_main


def main() -> int:
    args = list(sys.argv[1:])
    if not any(a == "--model" or a.startswith("--model=") for a in args):
        args.extend(["--model", "MiniMax-M2.5"])
    if "--no-plot" in args:
        args.remove("--no-plot")
    elif "--plot" not in args:
        args.append("--plot")
    sys.argv = [sys.argv[0]] + args
    return sweep_main()


if __name__ == "__main__":
    sys.exit(main())
