#!/usr/bin/env python3
"""Llama-3.3-70B-Instruct-FP8 on B200 / TRT-LLM — framework vs InferenceX.

InferenceX has 144 measured rows on `hardware=b200, framework=trt`
covering TP ∈ {1, 2, 4, 8} × ISL/OSL ∈ {1k/1k, 1k/8k, 8k/1k}. We model
the canonical ISL=OSL=1024 cut at all four TP shapes — clean dense GQA,
no MoE / MLA approximations.

Note on system spec: uses `b200.8gpu` (single 8-GPU NVSwitch5 island);
identical predictions to `gb200.72gpu` for TP ≤ 8. Llama-3.3-70B has the
same architecture as Llama-3.1-70B, so the framework loads `llama3.1_70b`
and overrides bytes_per_param=1 (FP8).

Usage:
    python benchmark/validate/llama3_70b_b200_trt.py
    python benchmark/validate/llama3_70b_b200_trt.py --bw-eta 0.55 --c-serving-us 75
"""
import argparse
import sys

from common import (
    add_common_cli, error_table, eta_filename_tag, eta_subtitle,
    load_measured, log_spaced_B, plot_tpot_vs_B, predict_at, run_framework,
)


MODEL = "Llama-3.3-70B-Instruct-FP8"
SYSTEM = "b200.8gpu"
PRECISION = "fp8"
ISL, OSL = 1024, 1024
TP_SHAPES = (1, 2, 4, 8)

# Per-stack calibration. Raw TRT-LLM on dense Llama hits a much harder BW
# derate than MoE models on the same stack (dense GEMMs sustain less of
# peak HBM than MoE expert hopping — counter-intuitive, but matches B200
# production reports). Best-fit (bw_eta=0.4, c_serving=50 µs/seq) gives
# ~7% MAE on TP=4. Note c_serving lower than dsr1_b200_trt or
# gpt_oss_120b_h200_trt — at small B (typical for Llama on B200/TRT) the
# host loop is in the under-amortized regime where per-sequence cost is
# closer to floor.
DEFAULT_BW_ETA = 0.4
DEFAULT_C_SERVING_US = 50.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--tp", type=int, choices=TP_SHAPES, default=None,
                    help="Run only this TP shape (default: all four)")
    add_common_cli(ap, default_bw_eta=DEFAULT_BW_ETA, default_c_serving_us=DEFAULT_C_SERVING_US)
    args = ap.parse_args()

    targets = (args.tp,) if args.tp else TP_SHAPES
    all_rows: list[tuple] = []
    for tp in targets:
        measured = load_measured(
            MODEL, isl=ISL, osl=OSL, precision=PRECISION,
            decode_tp=tp, decode_ep=1, num_decode_gpu=tp,
            framework={"trt", "trt-llm", "trtllm"},
            hardware="b200",
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
        rows = []
        for m in measured:
            pred = predict_at(
                model="llama3.1_70b", system_id=SYSTEM,
                PP=1, TP=tp, EP=1, SP=1,
                attention_mode="tp", layout="orthogonal",
                num_devices=tp, S_decode=ISL + OSL // 2, B=m.B,
                flops_eta=args.flops_eta, bw_eta=args.bw_eta,
                c_serving_us=args.c_serving_us, bytes_per_param=1,
            )
            rows.append((f"TP={tp}", m.B, m.tpot_ms, pred))
            all_rows.append((f"TP={tp}", m.B, m.tpot_ms, pred))

        out = args.out_dir / f"llama3_70b_b200_trt_tp{tp}{eta_filename_tag(args.flops_eta, args.bw_eta, args.c_serving_us)}.png"
        plot_tpot_vs_B(
            framework=framework, measured=measured,
            title=f"Llama-3.3-70B-FP8 / B200 / TRT-LLM — TP={tp} on {tp}-GPU server",
            subtitle=f"PP=1 TP={tp} EP=1 attention_mode=tp | ISL={ISL} OSL={OSL} FP8 | "
                     f"{eta_subtitle(args.flops_eta, args.bw_eta, args.c_serving_us)}",
            out_path=out,
        )
        print(f"  saved: {out.relative_to(args.out_dir.parent.parent)}")

    if all_rows:
        print()
        print(error_table(all_rows, title="Llama-3.3-70B-FP8 / B200 / TRT-LLM — framework vs InferenceX"))
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
