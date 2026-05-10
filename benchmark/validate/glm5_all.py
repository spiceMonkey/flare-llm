#!/usr/bin/env python3
"""GLM-5 across all single-island stacks — framework vs InferenceX.

Thin wrapper around `coverage_sweep.py --model GLM-5` for
discoverability. Sweeps all (hardware, framework) cells where the GLM-5
model spec applies and the partition fits on one NVLink island.
Generates per-cell plots by default.

Architecture caveats (from the model spec): MLA approximation via
n_kv=5; DeepSeek Sparse Attention (DSA, index_topk=2048) not modeled
— attention treated as full O(S²); over-estimates attention FLOPs at
long context.

Usage:
    python benchmark/validate/glm5_all.py
    python benchmark/validate/glm5_all.py --hardware h200
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from coverage_sweep import main as sweep_main


def main() -> int:
    args = list(sys.argv[1:])
    if not any(a == "--model" or a.startswith("--model=") for a in args):
        args.extend(["--model", "GLM-5"])
    if "--no-plot" in args:
        args.remove("--no-plot")
    elif "--plot" not in args:
        args.append("--plot")
    sys.argv = [sys.argv[0]] + args
    return sweep_main()


if __name__ == "__main__":
    sys.exit(main())
