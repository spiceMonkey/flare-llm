#!/usr/bin/env python3
"""gpt-oss-120b on GB200 / Dynamo-TRT — framework vs InferenceX.

One deployment: TP=4 EP=1 on 4 GPUs, one decode worker. The vendored data
holds two cuts of it, which differ only in prompt length:

  --isl 1024   (default) ISL/OSL 1024/1024, 7 points. The constants below are
               fitted on this cut.
  --isl 8192   ISL/OSL 8192/1024, 8 points. Not used in the fit: running it
               with the same constants is the held-out check for this setup.

Usage:
    python benchmark/validate/gpt_oss_120b_gb200_dynamo_trt.py
    python benchmark/validate/gpt_oss_120b_gb200_dynamo_trt.py --isl 8192
"""
import argparse
import sys

from common import (
    add_common_cli, error_table, eta_filename_tag, eta_subtitle, topology_tag,
    load_measured, log_spaced_B, plot_tpot_vs_B, predict_at, run_framework,
)


MODEL = "gpt-oss-120b"
SYSTEM = "gb200.72gpu"
PRECISION = "fp4"
OSL = 1024
TP, EP, NUM = 4, 1, 4

# Calibration is per setup (model, hardware, serving stack). Four constants,
# fitted on 2026-11-16 by least mean absolute error on the seven points of the
# ISL 1024 cut:
#   bw_eta                 multiplies the memory bandwidth, on top of the 0.7
#                          sustained fraction the GB200 system spec carries;
#   c_seq                  host work per sequence per step (microseconds);
#   c_orch                 host work per step that does not depend on batch;
#   kernel_overlap_factor  the fraction of the GPU stage time behind which
#                          kernel-launch time can hide:
#                          t = t_stage + max(0, t_kernel - rho * t_stage).
#                          Below one it turns the hard corner of
#                          max(t_stage, t_kernel) into the gradual bend the
#                          measured points show.
# Two things are properties of the Dynamo + TRT-LLM stack and shared with the
# DeepSeek-R1 validator: the per-kernel launch time (4 microseconds, between
# the graph-replay optimum near 1.5 and the eager-mode floor near 4.5) and
# host work that adds to the GPU step instead of hiding behind it
# (seq_overlap_factor = 0).
DEFAULT_BW_ETA = 0.69
DEFAULT_C_SEQ_US = 5.5
DEFAULT_KERNEL_LAUNCH_US = 4.0
DEFAULT_C_ORCH_US = 0.0
DEFAULT_SEQ_OVERLAP = 0.0
DEFAULT_KERNEL_OVERLAP = 0.55


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    add_common_cli(ap, default_bw_eta=DEFAULT_BW_ETA, default_c_seq_us=DEFAULT_C_SEQ_US,
                   default_c_orch_us=DEFAULT_C_ORCH_US)
    ap.add_argument("--isl", type=int, default=1024, choices=(1024, 8192),
                    help="prompt length of the cut: 1024 (fitted) or 8192 (held out of the fit)")
    args = ap.parse_args()
    ISL = args.isl

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
        attention_mode="tp", tp_ep_layout="orthogonal",
        num_devices=NUM, S_decode=ISL + OSL // 2,
        B_sweep=log_spaced_B(512),
        flops_eta=args.flops_eta, bw_eta=args.bw_eta,
        c_seq_us=args.c_seq_us, c_orch_us=args.c_orch_us,
        seq_overlap_factor=DEFAULT_SEQ_OVERLAP,
        kernel_launch_us=DEFAULT_KERNEL_LAUNCH_US, kernel_overlap_factor=DEFAULT_KERNEL_OVERLAP,
    )
    rows = []
    for m in measured:
        pred = predict_at(
            model="gpt_oss_120b", system_id=SYSTEM,
            PP=1, TP=TP, EP=EP, SP=1,
            attention_mode="tp", tp_ep_layout="orthogonal",
            num_devices=NUM, S_decode=ISL + OSL // 2, B=m.B,
            flops_eta=args.flops_eta, bw_eta=args.bw_eta,
            c_seq_us=args.c_seq_us, c_orch_us=args.c_orch_us,
            seq_overlap_factor=DEFAULT_SEQ_OVERLAP,
            kernel_launch_us=DEFAULT_KERNEL_LAUNCH_US, kernel_overlap_factor=DEFAULT_KERNEL_OVERLAP,
        )
        rows.append((f"TP={TP} EP={EP}", m.B, m.tpot_ms, pred))

    cut = "" if ISL == 1024 else f"_isl{ISL}"
    out = args.out_dir / f"gpt_oss_120b_dynamo_trt_tp{TP}_ep{EP}_dec{NUM}{cut}{eta_filename_tag(args.flops_eta, args.bw_eta, args.c_seq_us)}.png"
    plot_tpot_vs_B(
        framework=framework, measured=measured,
        title=f"gpt-oss-120b / GB200 / Dynamo-TRT — TP={TP} EP={EP} dec={NUM}",
        subtitle=f"PP=1 TP={TP} EP={EP} attention_mode=tp | ISL={ISL} OSL={OSL} FP4 | "
                 f"sys={SYSTEM} | {topology_tag(SYSTEM)} | {eta_subtitle(args.flops_eta, args.bw_eta, args.c_seq_us, args.c_orch_us)}",
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
