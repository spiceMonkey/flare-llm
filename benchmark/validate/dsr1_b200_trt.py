#!/usr/bin/env python3
"""DeepSeek-R1-0528 on B200 / TRT-LLM (raw) — framework vs InferenceX.

Companion to `dsr1_gb200_dynamo_trt.py` — same model, different stack:
single 8-GPU DGX/HGX B200 server (not the rack-scale GB200 NVL72) running
raw TensorRT-LLM (not the Dynamo serving orchestrator). Tests whether the
framework predictions hold across:

  - smaller hardware island (8 GPUs vs 72)
  - simpler serving stack (raw TRT vs Dynamo+TRT)

InferenceX has data on TP ∈ {4, 8} EP=1 (orthogonal+TP-attn) and one
TP=EP=8 dec=64 cut (orthogonal — separate TP=EP groups, NOT co-located).
This driver covers the densest orthogonal cut: TP=8 EP=1 dec=8.

Usage:
    python benchmark/validate/dsr1_b200_trt.py
    python benchmark/validate/dsr1_b200_trt.py --bw-eta 0.7 --c-serving-us 25
"""
import argparse
import sys

from common import (
    add_common_cli, error_table, eta_filename_tag, eta_subtitle,
    load_measured, log_spaced_B, plot_tpot_vs_B, predict_at, run_framework,
)


MODEL = "DeepSeek-R1-0528"
SYSTEM = "b200.8gpu"
PRECISION = "fp4"
ISL, OSL = 1024, 1024
TP, EP, NUM = 8, 1, 8

# Per-stack calibration. Raw TRT-LLM has substantially higher per-sequence
# host overhead than Dynamo+TRT (no Dynamo Python orchestrator absorbing
# the bookkeeping into a single CUDA-Graph launch); fits to ~17% MAE at
# (bw_eta=0.7, c_serving=100 µs/seq). The 100 µs is at the high end of
# §7.2's 30–60 µs Python-heavy stack range — note raw TRT-LLM's per-step
# loop is C++ but exposes more individual kernel launches than Dynamo.
DEFAULT_BW_ETA = 0.7
DEFAULT_C_SERVING_US = 100.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    add_common_cli(ap, default_bw_eta=DEFAULT_BW_ETA, default_c_serving_us=DEFAULT_C_SERVING_US)
    args = ap.parse_args()

    measured = load_measured(
        MODEL, isl=ISL, osl=OSL, precision=PRECISION,
        decode_tp=TP, decode_ep=EP, num_decode_gpu=NUM,
        framework={"trt", "trt-llm", "trtllm"},
        hardware="b200",
    )
    print(f"[DSR1 / B200 / TRT] TP={TP} EP={EP} dec={NUM} ISL={ISL} OSL={OSL} {PRECISION} | "
          f"{len(measured)} measured points")
    if not measured:
        print("  no measured rows for this cut", file=sys.stderr)
        return 1

    framework = run_framework(
        model="deepseek_r1_0528", system_id=SYSTEM,
        PP=1, TP=TP, EP=EP, SP=1,
        attention_mode="tp", layout="orthogonal",
        num_devices=NUM, S_decode=ISL + OSL // 2,
        B_sweep=log_spaced_B(8192),
        flops_eta=args.flops_eta, bw_eta=args.bw_eta,
        c_serving_us=args.c_serving_us,
        bytes_per_param=0.5,  # FP4
    )
    rows = []
    for m in measured:
        pred = predict_at(
            model="deepseek_r1_0528", system_id=SYSTEM,
            PP=1, TP=TP, EP=EP, SP=1,
            attention_mode="tp", layout="orthogonal",
            num_devices=NUM, S_decode=ISL + OSL // 2, B=m.B,
            flops_eta=args.flops_eta, bw_eta=args.bw_eta,
            c_serving_us=args.c_serving_us, bytes_per_param=0.5,
        )
        rows.append((f"TP={TP} EP={EP}", m.B, m.tpot_ms, pred))

    out = args.out_dir / f"dsr1_b200_trt_tp{TP}_ep{EP}_dec{NUM}{eta_filename_tag(args.flops_eta, args.bw_eta, args.c_serving_us)}.png"
    plot_tpot_vs_B(
        framework=framework, measured=measured,
        title=f"DSR1 / B200 / TRT-LLM — TP={TP} EP={EP} dec={NUM}",
        subtitle=f"PP=1 TP={TP} EP={EP} attention_mode=tp | ISL={ISL} OSL={OSL} FP4 | "
                 f"{eta_subtitle(args.flops_eta, args.bw_eta, args.c_serving_us)}",
        out_path=out,
    )
    print(f"  saved: {out.relative_to(args.out_dir.parent.parent)}\n")
    print(error_table(rows, title=f"DSR1 / B200 / TRT-LLM — framework vs InferenceX"))

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
