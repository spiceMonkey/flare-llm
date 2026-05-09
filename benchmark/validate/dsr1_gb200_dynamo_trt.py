#!/usr/bin/env python3
"""DeepSeek-R1-0528 on GB200 / Dynamo-TRT — framework vs InferenceX.

Two cuts in the InferenceX dataset for this (model, hardware, framework)
triple:

  1. EXACT (orthogonal + TP-attention, dec_tp=N dec_ep=1): natively
     modeled, no approximation. Densest bucket is TP=36 EP=1 dec=36.

  2. CO-LOCATED (DSv3 production shape, dec_tp=dec_ep=N on N GPUs,
     dec_dp_attention=True): natively modeled by
     PartitionSpec(layout="co_located", attention_mode="dp"). InferenceX
     publishes three replica sizes — 8, 16, 32 GPU.

This script runs both cuts at default knobs unless --flops-eta / --bw-eta
/ --c-serving-us are set on the CLI.

Usage:
    python benchmark/validate/dsr1_gb200_dynamo_trt.py
    python benchmark/validate/dsr1_gb200_dynamo_trt.py --cut exact
    python benchmark/validate/dsr1_gb200_dynamo_trt.py --cut colocated --bw-eta 0.7 --c-serving-us 5
"""
import argparse
import sys

from common import (
    add_common_cli, error_table, eta_filename_tag, eta_subtitle,
    load_measured, log_spaced_B, plot_tpot_vs_B, predict_at, run_framework,
)


MODEL = "DeepSeek-R1-0528"
SYSTEM = "gb200.72gpu"
PRECISION = "fp4"
ISL, OSL = 1024, 1024

# Per-stack calibration baked in as the driver default — overrideable via CLI.
# Calibrated on the colocated TP=EP=8 cut (the canonical DSv3 production
# shape on this stack); fits to ~11% MAE. Dynamo+TRT-LLM shows
# c_serving close to the §7.2 anchor (Dynamo CUDA-Graph stack ≈ 22 µs/seq).
DEFAULT_BW_ETA = 1.0
DEFAULT_C_SERVING_US = 5.0


def run_exact(args) -> tuple[list[tuple], int]:
    """Cut 1: TP-only orthogonal config. Densest is TP=36 EP=1 dec=36."""
    bucket = (36, 1, 36)
    measured = load_measured(
        MODEL, isl=ISL, osl=OSL, precision=PRECISION,
        decode_tp=bucket[0], decode_ep=bucket[1], num_decode_gpu=bucket[2],
        framework={"dynamo-trt", "trt", "trt-llm", "trtllm", "dynamo-trt-llm"},
        hardware="gb200",
    )
    print(f"\n[EXACT] TP={bucket[0]} EP=1 dec={bucket[2]}, ISL={ISL} OSL={OSL} {PRECISION} | "
          f"{len(measured)} measured points")
    if not measured:
        return [], 0

    framework = run_framework(
        model="deepseek_r1_0528", system_id=SYSTEM,
        PP=1, TP=bucket[0], EP=1, SP=1,
        attention_mode="tp", layout="orthogonal",
        num_devices=bucket[2], S_decode=ISL + OSL // 2,
        B_sweep=log_spaced_B(2048),
        flops_eta=args.flops_eta, bw_eta=args.bw_eta,
        c_serving_us=args.c_serving_us,
        bytes_per_param=0.5,
    )

    rows = []
    for m in measured:
        pred = predict_at(
            model="deepseek_r1_0528", system_id=SYSTEM,
            PP=1, TP=bucket[0], EP=1, SP=1,
            attention_mode="tp", layout="orthogonal",
            num_devices=bucket[2], S_decode=ISL + OSL // 2, B=m.B,
            flops_eta=args.flops_eta, bw_eta=args.bw_eta,
            c_serving_us=args.c_serving_us, bytes_per_param=0.5,
        )
        rows.append((f"TP={bucket[0]} EP=1", m.B, m.tpot_ms, pred))

    out = args.out_dir / f"dsr1_dynamo_trt_exact_tp{bucket[0]}_ep1_dec{bucket[2]}{eta_filename_tag(args.flops_eta, args.bw_eta, args.c_serving_us)}.png"
    plot_tpot_vs_B(
        framework=framework, measured=measured,
        title=f"DSR1 / GB200 / Dynamo-TRT — EXACT bucket: TP={bucket[0]} EP=1 dec={bucket[2]}",
        subtitle=f"PP=1 TP={bucket[0]} EP=1 attention_mode=tp | ISL={ISL} OSL={OSL} FP4 | "
                 f"{eta_subtitle(args.flops_eta, args.bw_eta, args.c_serving_us)}",
        out_path=out,
    )
    print(f"  saved: {out.relative_to(args.out_dir.parent.parent)}")
    return rows, len(measured)


def run_colocated(args) -> tuple[list[tuple], int]:
    """Cut 2: TP+EP co-located DSv3 production shape (TP=EP=N on N GPUs)."""
    rows = []
    n = 0
    for tp_ep in (8, 16, 32):
        measured = load_measured(
            MODEL, isl=ISL, osl=OSL, precision=PRECISION,
            decode_tp=tp_ep, decode_ep=tp_ep, num_decode_gpu=tp_ep,
            dp_attention=True,
            framework={"dynamo-trt", "trt", "trt-llm", "trtllm", "dynamo-trt-llm"},
            hardware="gb200",
        )
        print(f"\n[CO-LOCATED] TP=EP={tp_ep} on {tp_ep}-GPU replica, "
              f"ISL={ISL} OSL={OSL} {PRECISION} | {len(measured)} measured points")
        if not measured:
            continue
        n += len(measured)

        framework = run_framework(
            model="deepseek_r1_0528", system_id=SYSTEM,
            PP=1, TP=tp_ep, EP=tp_ep, SP=1,
            attention_mode="dp", layout="co_located",
            num_devices=tp_ep, S_decode=ISL + OSL // 2,
            B_sweep=log_spaced_B(8192),
            flops_eta=args.flops_eta, bw_eta=args.bw_eta,
            c_serving_us=args.c_serving_us,
            bytes_per_param=0.5,
        )
        for m in measured:
            pred = predict_at(
                model="deepseek_r1_0528", system_id=SYSTEM,
                PP=1, TP=tp_ep, EP=tp_ep, SP=1,
                attention_mode="dp", layout="co_located",
                num_devices=tp_ep, S_decode=ISL + OSL // 2, B=m.B,
                flops_eta=args.flops_eta, bw_eta=args.bw_eta,
                c_serving_us=args.c_serving_us, bytes_per_param=0.5,
            )
            rows.append((f"TP=EP={tp_ep}", m.B, m.tpot_ms, pred))

        out = args.out_dir / f"dsr1_dynamo_trt_colocated_tp{tp_ep}ep{tp_ep}{eta_filename_tag(args.flops_eta, args.bw_eta, args.c_serving_us)}.png"
        plot_tpot_vs_B(
            framework=framework, measured=measured,
            title=f"DSR1 / GB200 / Dynamo-TRT — CO-LOCATED TP=EP={tp_ep} on {tp_ep}-GPU replica",
            subtitle=f"layout=co_located attention_mode=dp PP=1 TP={tp_ep} EP={tp_ep} SP=1 | "
                     f"ISL={ISL} OSL={OSL} FP4 | {eta_subtitle(args.flops_eta, args.bw_eta, args.c_serving_us)}",
            out_path=out,
        )
        print(f"  saved: {out.relative_to(args.out_dir.parent.parent)}")

    return rows, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--cut", choices=["exact", "colocated", "all"], default="all")
    add_common_cli(ap, default_bw_eta=DEFAULT_BW_ETA, default_c_serving_us=DEFAULT_C_SERVING_US)
    args = ap.parse_args()

    all_rows: list[tuple] = []
    if args.cut in ("exact", "all"):
        rows, _ = run_exact(args)
        all_rows.extend(rows)
    if args.cut in ("colocated", "all"):
        rows, _ = run_colocated(args)
        all_rows.extend(rows)

    if all_rows:
        print()
        print(error_table(all_rows, title="DSR1 / GB200 / Dynamo-TRT — framework vs InferenceX"))
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
