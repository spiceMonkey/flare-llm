#!/usr/bin/env python3
"""Llama-3.3-70B-Instruct-FP8 on H200 / TRT-LLM — framework vs InferenceX.

Hopper baseline for Llama-3.3-70B-FP8 — companion to
`llama3_70b_b200_trt.py`. Same model, same framework, different hardware
generation. Useful for sanity-checking that the framework's predictions
scale sensibly across HBM/Tensor-Core generations.

InferenceX has TP shapes {1, 2, 4, 8} × ISL/OSL ∈ {1k/1k, 1k/8k, 8k/1k}
on h200/trt. We model the canonical ISL=OSL=1024 cut at all four TP
shapes.

Usage:
    python benchmark/validate/llama3_70b_h200_trt.py
    python benchmark/validate/llama3_70b_h200_trt.py --bw-eta 0.55 --c-serving-us 75
"""
import argparse
import sys

from common import (
    add_common_cli, error_table, eta_filename_tag, eta_subtitle,
    load_measured, log_spaced_B, plot_tpot_vs_B, predict_at, run_framework,
)


MODEL = "Llama-3.3-70B-Instruct-FP8"
SYSTEM = "h200.8gpu"
PRECISION = "fp8"
ISL, OSL = 1024, 1024
TP_SHAPES = (1, 2, 4, 8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--tp", type=int, choices=TP_SHAPES, default=None,
                    help="Run only this TP shape (default: all four)")
    add_common_cli(ap)
    args = ap.parse_args()

    targets = (args.tp,) if args.tp else TP_SHAPES
    all_rows: list[tuple] = []
    for tp in targets:
        measured = load_measured(
            MODEL, isl=ISL, osl=OSL, precision=PRECISION,
            decode_tp=tp, decode_ep=1, num_decode_gpu=tp,
            framework={"trt", "trt-llm", "trtllm"},
            hardware="h200",
        )
        print(f"\n[TP={tp}] dec={tp} ISL={ISL} OSL={OSL} {PRECISION} | {len(measured)} measured points")
        if not measured:
            continue

        framework = run_framework(
            model="llama3.1_70b", system_id=SYSTEM,
            PP=1, TP=tp, EP=1, SP=1,
            attention_mode="tp", layout="orthogonal",
            num_devices=tp, S_decode=ISL + OSL // 2,
            B_sweep=log_spaced_B(512),
            flops_eta=args.flops_eta, bw_eta=args.bw_eta,
            c_serving_us=args.c_serving_us,
            bytes_per_param=1,  # FP8
        )
        for m in measured:
            pred = predict_at(
                model="llama3.1_70b", system_id=SYSTEM,
                PP=1, TP=tp, EP=1, SP=1,
                attention_mode="tp", layout="orthogonal",
                num_devices=tp, S_decode=ISL + OSL // 2, B=m.B,
                flops_eta=args.flops_eta, bw_eta=args.bw_eta,
                c_serving_us=args.c_serving_us, bytes_per_param=1,
            )
            all_rows.append((f"TP={tp}", m.B, m.tpot_ms, pred))

        out = args.out_dir / f"llama3_70b_h200_trt_tp{tp}{eta_filename_tag(args.flops_eta, args.bw_eta, args.c_serving_us)}.png"
        plot_tpot_vs_B(
            framework=framework, measured=measured,
            title=f"Llama-3.3-70B-FP8 / H200 / TRT-LLM — TP={tp} on {tp}-GPU server",
            subtitle=f"PP=1 TP={tp} EP=1 attention_mode=tp | ISL={ISL} OSL={OSL} FP8 | "
                     f"{eta_subtitle(args.flops_eta, args.bw_eta, args.c_serving_us)}",
            out_path=out,
        )
        print(f"  saved: {out.relative_to(args.out_dir.parent.parent)}")

    if all_rows:
        print()
        print(error_table(all_rows, title="Llama-3.3-70B-FP8 / H200 / TRT-LLM — framework vs InferenceX"))
        if args.check is not None:
            import numpy as np
            mae = np.mean(np.abs([(p - m) / m * 100 for _, _, m, p in all_rows]))
            if mae > args.check:
                print(f"\nFAIL: MAE {mae:.1f}% > threshold {args.check}%")
                return 1
            print(f"\nPASS: MAE {mae:.1f}% ≤ threshold {args.check}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
