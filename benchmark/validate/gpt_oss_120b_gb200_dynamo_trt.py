#!/usr/bin/env python3
"""gpt-oss-120b on GB200 / Dynamo-TRT — framework vs InferenceX.

Single cut: TP=4 EP=1 on 4 GPUs (the densest "exact-modelable" bucket in
the dataset, no co-located complications). Calibration anchor for the
Dynamo-TRT-LLM serving stack — best-fit `c_serving ≈ 22 µs/seq` per
`decode.md §7.2` calibration table.

Usage:
    python benchmark/validate/gpt_oss_120b_gb200_dynamo_trt.py
    python benchmark/validate/gpt_oss_120b_gb200_dynamo_trt.py --c-serving-us 22
"""
import argparse
import sys

from common import (
    add_common_cli, error_table, eta_filename_tag, eta_subtitle,
    load_measured, log_spaced_B, plot_tpot_vs_B, predict_at, run_framework,
)


MODEL = "gpt-oss-120b"
SYSTEM = "gb200.72gpu"
PRECISION = "fp4"
ISL, OSL = 1024, 1024
TP, EP, NUM = 4, 1, 4

# Per-stack calibration. Anchor case for Dynamo+TRT — fits to ~9% MAE at
# (bw_eta=1.0, c_serving=22 µs/seq), exactly matching the §7.2 calibration
# range for "production CUDA-Graph stacks". Confirms the 22 µs anchor is
# Dynamo-specific (raw TRT-LLM needs c_serving 4–5× higher; see
# dsr1_b200_trt and llama3_70b_*).
DEFAULT_BW_ETA = 1.0
DEFAULT_C_SERVING_US = 22.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    add_common_cli(ap, default_bw_eta=DEFAULT_BW_ETA, default_c_serving_us=DEFAULT_C_SERVING_US)
    args = ap.parse_args()

    measured = load_measured(
        MODEL, isl=ISL, osl=OSL, precision=PRECISION,
        decode_tp=TP, decode_ep=EP, num_decode_gpu=NUM,
        framework={"dynamo-trt", "trt", "trt-llm", "trtllm", "dynamo-trt-llm"},
        hardware="gb200",
    )
    print(f"[gpt-oss-120b] TP={TP} EP={EP} dec={NUM} ISL={ISL} OSL={OSL} {PRECISION} | "
          f"{len(measured)} measured points")
    if not measured:
        print("  no measured rows for this cut", file=sys.stderr)
        return 1

    framework = run_framework(
        model="gpt_oss_120b", system_id=SYSTEM,
        PP=1, TP=TP, EP=EP, SP=1,
        attention_mode="tp", layout="orthogonal",
        num_devices=NUM, S_decode=ISL + OSL // 2,
        B_sweep=log_spaced_B(512),
        flops_eta=args.flops_eta, bw_eta=args.bw_eta,
        c_serving_us=args.c_serving_us,
    )
    rows = []
    for m in measured:
        pred = predict_at(
            model="gpt_oss_120b", system_id=SYSTEM,
            PP=1, TP=TP, EP=EP, SP=1,
            attention_mode="tp", layout="orthogonal",
            num_devices=NUM, S_decode=ISL + OSL // 2, B=m.B,
            flops_eta=args.flops_eta, bw_eta=args.bw_eta,
            c_serving_us=args.c_serving_us,
        )
        rows.append((f"TP={TP} EP={EP}", m.B, m.tpot_ms, pred))

    out = args.out_dir / f"gpt_oss_120b_dynamo_trt_tp{TP}_ep{EP}_dec{NUM}{eta_filename_tag(args.flops_eta, args.bw_eta, args.c_serving_us)}.png"
    plot_tpot_vs_B(
        framework=framework, measured=measured,
        title=f"gpt-oss-120b / GB200 / Dynamo-TRT — TP={TP} EP={EP} dec={NUM}",
        subtitle=f"PP=1 TP={TP} EP={EP} attention_mode=tp | ISL={ISL} OSL={OSL} FP4 | "
                 f"{eta_subtitle(args.flops_eta, args.bw_eta, args.c_serving_us)}",
        out_path=out,
    )
    print(f"  saved: {out.relative_to(args.out_dir.parent.parent)}\n")
    print(error_table(rows, title=f"gpt-oss-120b / GB200 / Dynamo-TRT — framework vs InferenceX"))

    if args.check is not None:
        import numpy as np
        mae = np.mean(np.abs([(p - m) / m * 100 for _, _, m, p in rows]))
        if mae > args.check:
            print(f"\nFAIL: MAE {mae:.1f}% > threshold {args.check}%")
            return 1
        print(f"\nPASS: MAE {mae:.1f}% ≤ threshold {args.check}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
