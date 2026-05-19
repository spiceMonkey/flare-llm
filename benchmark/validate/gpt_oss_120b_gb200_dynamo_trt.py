#!/usr/bin/env python3
"""gpt-oss-120b on GB200 / Dynamo-TRT — framework vs InferenceX.

Single cut: TP=4 EP=1 on 4 GPUs (the densest "exact-modelable" bucket in
the dataset, no co-located complications). Calibration anchor for the
Dynamo-TRT-LLM serving stack — best-fit `c_seq ≈ 22 µs/seq` per
`decode.md §7.3` calibration table.

Usage:
    python benchmark/validate/gpt_oss_120b_gb200_dynamo_trt.py
    python benchmark/validate/gpt_oss_120b_gb200_dynamo_trt.py --c-seq-us 22
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
ISL, OSL = 1024, 1024
TP, EP, NUM = 4, 1, 4

# Per-stack calibration. Dynamo+TRT anchor case. The previous super-nameplate
# bw_eta=1.4286 (physically impossible) was a tuning hack compensating for the
# pre-`moe_weight_traffic_bytes` over-count of T_theta (full N_exp footprint
# read every step regardless of B). With T_theta(B) correctly modeled as the
# expectation of touched experts, bw_eta resets to a realistic 0.7 (matches
# the DSr1 GB200 calibration on the same hardware). The residual ~38% MAE
# reflects a remaining slope/offset gap: at B=1 the model under-predicts
# (~0.8 ms vs measured 1.9 ms — suggests a B-independent per-step overhead
# not captured), and at large B it over-predicts (predicted curve rises
# faster than measured). Resolving this likely requires a fixed per-step
# overhead term or precision-aware kernel-launch budget; documented as a
# §5 Limitation.
DEFAULT_BW_ETA = 0.7
# 5 µs/seq — Dynamo+TRT stack default (matches dynamo_trt.json and the
# DSr1 Dynamo+TRT validator). At panel-(a) max B=128, c_seq·B = 0.64 ms
# vs t_step_base ~5-10 ms → fully hidden by the overlap gate (ρ_seq=1),
# so the value does not affect MAE. Previous value 22 µs was a tuning hack
# under the old additive (no-overlap) model; with the overlap gate the
# stack-wide 5 µs default is the consistent choice.
DEFAULT_C_SEQ_US = 5.0
# 4.0 µs — Dynamo-orchestrator calibrated effective per-kernel cost (same
# as the DSr1 Dynamo+TRT validator). Sits between CUDA-Graph optimum
# (1.5 µs) and TaxBreak's eager-mode floor (4.5-4.7 µs); the spread
# absorbs the per-step scheduler-tick path and MoE kernel-count
# under-count. Lifts panel-(a) small-B prediction (B=1..4) from
# ~0.8 ms to ~1.8 ms, matching measured ~1.9 ms.
DEFAULT_KERNEL_LAUNCH_US = 4.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    add_common_cli(ap, default_bw_eta=DEFAULT_BW_ETA, default_c_seq_us=DEFAULT_C_SEQ_US)
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
        attention_mode="tp", tp_ep_layout="orthogonal",
        num_devices=NUM, S_decode=ISL + OSL // 2,
        B_sweep=log_spaced_B(512),
        flops_eta=args.flops_eta, bw_eta=args.bw_eta,
        c_seq_us=args.c_seq_us,
        kernel_launch_us=DEFAULT_KERNEL_LAUNCH_US,
    )
    rows = []
    for m in measured:
        pred = predict_at(
            model="gpt_oss_120b", system_id=SYSTEM,
            PP=1, TP=TP, EP=EP, SP=1,
            attention_mode="tp", tp_ep_layout="orthogonal",
            num_devices=NUM, S_decode=ISL + OSL // 2, B=m.B,
            flops_eta=args.flops_eta, bw_eta=args.bw_eta,
            c_seq_us=args.c_seq_us,
            kernel_launch_us=DEFAULT_KERNEL_LAUNCH_US,
        )
        rows.append((f"TP={TP} EP={EP}", m.B, m.tpot_ms, pred))

    out = args.out_dir / f"gpt_oss_120b_dynamo_trt_tp{TP}_ep{EP}_dec{NUM}{eta_filename_tag(args.flops_eta, args.bw_eta, args.c_seq_us)}.png"
    plot_tpot_vs_B(
        framework=framework, measured=measured,
        title=f"gpt-oss-120b / GB200 / Dynamo-TRT — TP={TP} EP={EP} dec={NUM}",
        subtitle=f"PP=1 TP={TP} EP={EP} attention_mode=tp | ISL={ISL} OSL={OSL} FP4 | "
                 f"sys={SYSTEM} | {topology_tag(SYSTEM)} | {eta_subtitle(args.flops_eta, args.bw_eta, args.c_seq_us)}",
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
