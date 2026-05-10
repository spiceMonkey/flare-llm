#!/usr/bin/env python3
"""DeepSeek-V4-Pro across all single-island stacks — framework vs InferenceX.

Thin wrapper around `coverage_sweep.py --model "DeepSeek-V4-Pro"` for
discoverability. Sweeps all (hardware, framework) cells where the
DeepSeek-V4-Pro model spec applies and the partition fits on one
NVLink island. Generates per-cell plots by default.

Architecture caveats (from the model spec): MLA approximation via
n_kv=5; CSA / HCA / sliding-window / MTP head extensions not modeled
— predicted decode latency is an upper bound on the real workload.

Usage:
    python benchmark/validate/deepseek_v4_pro_all.py
    python benchmark/validate/deepseek_v4_pro_all.py --hardware b300
    python benchmark/validate/deepseek_v4_pro_all.py --framework vllm --no-plot
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from coverage_sweep import main as sweep_main


def main() -> int:
    # Inject --model and --plot defaults if user didn't override
    args = list(sys.argv[1:])
    if not any(a == "--model" or a.startswith("--model=") for a in args):
        args.extend(["--model", "DeepSeek-V4-Pro"])
    if "--no-plot" in args:
        args.remove("--no-plot")
    elif "--plot" not in args:
        args.append("--plot")
    sys.argv = [sys.argv[0]] + args
    return sweep_main()


if __name__ == "__main__":
    sys.exit(main())
