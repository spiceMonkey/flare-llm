#!/usr/bin/env python3
"""DeepSeek-R1-0528 on GB300 / Dynamo-TRT — framework vs InferenceX.

Companion to dsr1_gb300_dynamo_sglang.py (same model, same hardware,
different software stack) and dsr1_gb200_dynamo_trt.py (same model,
same stack, different hardware generation). Lets us isolate the
framework axis (Dynamo+TRT vs Dynamo+SGLang on the same gb300) cleanly
from the hardware axis (Blackwell Ultra vs Blackwell on the same
Dynamo+TRT stack).

Two cuts in the InferenceX dataset for this (model, hardware,
framework) triple:

  1. ORTHO (TP=8 EP=8 dec=32, dp_attn=False — TP-attention): the
     framework runs 4 replicas of 8 GPUs each (N_replica = 32 / 8).
     Five measured B values from 5 to 192. Scatter-direct is inert
     here (TP-attention disables it).

  2. CO-LOCATED (TP=EP=N dec=N, dp_attn=True — DSv3 production
     shape, single replica): three N values — 8, 16, 32. Each has
     a small number of measured points at large B. Scatter-direct
     applies here (DP-attention).

Usage:
    python benchmark/validate/dsr1_gb300_dynamo_trt.py
    python benchmark/validate/dsr1_gb300_dynamo_trt.py --cut ortho
    python benchmark/validate/dsr1_gb300_dynamo_trt.py --cut colocated
"""
import argparse
import sys

from common import (
    add_common_cli, error_table, eta_filename_tag, eta_subtitle, topology_tag,
    load_measured, log_spaced_B, plot_tpot_vs_B, predict_at, run_framework,
)


MODEL = "DeepSeek-R1-0528"
SYSTEM = "gb300.72gpu"
PRECISION = "fp4"
ISL, OSL = 1024, 1024

# Per-stack calibration. Production DSv3 on Dynamo+TRT-LLM uses
# scatter-direct MoE A2A (decode.md §5.2) under co-located DP-attention;
# the ORTHO cut uses TP-attention so scatter is inert, but the
# kernel_launch / c_serving / bw_eta knobs still apply. Fits to ~18% MAE
# overall (ortho=20%, colo=15%). Notable: bw_eta = 1.0 fits best —
# Blackwell Ultra HBM3e under TRT-LLM appears to sustain very close to
# nameplate peak, in contrast to gb200/dynamo-trt where bw_eta ≈ 0.7
# fit better (likely a controller / kernel-fusion improvement).
DEFAULT_BW_ETA = 1.0
DEFAULT_C_SERVING_US = 5.0
DEFAULT_KERNEL_LAUNCH_US = 7.0
DEFAULT_MOE_A2A_PATTERN = "scatter"


def run_ortho(args) -> tuple[list[tuple], int]:
    """Cut 1: TP=8 EP=8 dec=32 orthogonal (4 replicas of 8 GPUs each, TP-attn)."""
    rows: list[tuple] = []
    measured = load_measured(
        MODEL, isl=ISL, osl=OSL, precision=PRECISION,
        decode_tp=8, decode_ep=8, num_decode_gpu=32,
        dp_attention=False,
        framework={"dynamo-trt", "trt", "trt-llm", "trtllm", "dynamo-trt-llm"},
        hardware="gb300",
    )
    print(f"\n[ORTHO] TP=8 EP=8 dec=32 (4 replicas × 8 GPUs, TP-attn) | "
          f"{len(measured)} measured points")
    if not measured:
        return [], 0

    framework = run_framework(
        model="deepseek_r1_0528", system_id=SYSTEM,
        PP=1, TP=8, EP=8, SP=1,
        attention_mode="tp", layout="orthogonal",
        num_devices=32, S_decode=ISL + OSL // 2,
        B_sweep=log_spaced_B(2048),
        flops_eta=args.flops_eta, bw_eta=args.bw_eta,
        c_serving_us=args.c_serving_us,
        moe_a2a_pattern=DEFAULT_MOE_A2A_PATTERN,
        kernel_launch_us=DEFAULT_KERNEL_LAUNCH_US,
        bytes_per_param=0.5,
    )
    for m in measured:
        pred = predict_at(
            model="deepseek_r1_0528", system_id=SYSTEM,
            PP=1, TP=8, EP=8, SP=1,
            attention_mode="tp", layout="orthogonal",
            num_devices=32, S_decode=ISL + OSL // 2, B=m.B,
            flops_eta=args.flops_eta, bw_eta=args.bw_eta,
            c_serving_us=args.c_serving_us,
            moe_a2a_pattern=DEFAULT_MOE_A2A_PATTERN,
            kernel_launch_us=DEFAULT_KERNEL_LAUNCH_US,
            bytes_per_param=0.5,
        )
        rows.append(("TP=8 EP=8 dec=32 ortho", m.B, m.tpot_ms, pred))

    out = args.out_dir / f"dsr1_gb300_dynamo_trt_ortho_tp8ep8_dec32{eta_filename_tag(args.flops_eta, args.bw_eta, args.c_serving_us)}.png"
    plot_tpot_vs_B(
        framework=framework, measured=measured,
        title="DSR1 / GB300 / Dynamo-TRT — ORTHO TP=8 EP=8 dec=32 (4 replicas, TP-attn)",
        subtitle=f"PP=1 TP=8 EP=8 attention_mode=tp layout=orthogonal | "
                 f"ISL={ISL} OSL={OSL} FP4 | "
                 f"sys={SYSTEM} | {topology_tag(SYSTEM)} | {eta_subtitle(args.flops_eta, args.bw_eta, args.c_serving_us)}",
        out_path=out,
    )
    print(f"  saved: {out.relative_to(args.out_dir.parent.parent)}")
    return rows, len(measured)


def run_colocated(args) -> tuple[list[tuple], int]:
    """Cut 2: TP+EP co-located DSv3 production shape — TP=EP={8,16,32}."""
    rows: list[tuple] = []
    n = 0
    for tp_ep in (8, 16, 32):
        measured = load_measured(
            MODEL, isl=ISL, osl=OSL, precision=PRECISION,
            decode_tp=tp_ep, decode_ep=tp_ep, num_decode_gpu=tp_ep,
            dp_attention=True,
            framework={"dynamo-trt", "trt", "trt-llm", "trtllm", "dynamo-trt-llm"},
            hardware="gb300",
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
            moe_a2a_pattern=DEFAULT_MOE_A2A_PATTERN,
            kernel_launch_us=DEFAULT_KERNEL_LAUNCH_US,
            bytes_per_param=0.5,
        )
        for m in measured:
            pred = predict_at(
                model="deepseek_r1_0528", system_id=SYSTEM,
                PP=1, TP=tp_ep, EP=tp_ep, SP=1,
                attention_mode="dp", layout="co_located",
                num_devices=tp_ep, S_decode=ISL + OSL // 2, B=m.B,
                flops_eta=args.flops_eta, bw_eta=args.bw_eta,
                c_serving_us=args.c_serving_us,
                moe_a2a_pattern=DEFAULT_MOE_A2A_PATTERN,
                kernel_launch_us=DEFAULT_KERNEL_LAUNCH_US,
                bytes_per_param=0.5,
            )
            rows.append((f"TP=EP={tp_ep}", m.B, m.tpot_ms, pred))

        out = args.out_dir / f"dsr1_gb300_dynamo_trt_colocated_tp{tp_ep}ep{tp_ep}{eta_filename_tag(args.flops_eta, args.bw_eta, args.c_serving_us)}.png"
        plot_tpot_vs_B(
            framework=framework, measured=measured,
            title=f"DSR1 / GB300 / Dynamo-TRT — CO-LOCATED TP=EP={tp_ep} on {tp_ep}-GPU replica",
            subtitle=f"layout=co_located attention_mode=dp PP=1 TP={tp_ep} EP={tp_ep} SP=1 | "
                     f"ISL={ISL} OSL={OSL} FP4 | sys={SYSTEM} | {topology_tag(SYSTEM)} | {eta_subtitle(args.flops_eta, args.bw_eta, args.c_serving_us)}",
            out_path=out,
        )
        print(f"  saved: {out.relative_to(args.out_dir.parent.parent)}")
    return rows, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--cut", choices=["ortho", "colocated", "all"], default="all")
    add_common_cli(ap, default_bw_eta=DEFAULT_BW_ETA, default_c_serving_us=DEFAULT_C_SERVING_US)
    args = ap.parse_args()

    all_rows: list[tuple] = []
    if args.cut in ("ortho", "all"):
        rows, _ = run_ortho(args)
        all_rows.extend(rows)
    if args.cut in ("colocated", "all"):
        rows, _ = run_colocated(args)
        all_rows.extend(rows)

    if all_rows:
        print()
        print(error_table(all_rows, title="DSR1 / GB300 / Dynamo-TRT — framework vs InferenceX"))
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
