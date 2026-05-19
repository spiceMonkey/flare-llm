#!/usr/bin/env python3
"""gpt-oss-120b on H200 / TRT-LLM — framework vs InferenceX.

Hopper baseline for the gpt-oss-120b model — same model as the GB200
Dynamo-TRT validator, different generation hardware (Hopper vs Blackwell),
different framework (raw TRT-LLM vs Dynamo+TRT). InferenceX has TP shapes
{1, 2, 4, 8} on h200/trt; we cover all four at ISL=OSL=1024.

Usage:
    python benchmark/validate/gpt_oss_120b_h200_trt.py
    python benchmark/validate/gpt_oss_120b_h200_trt.py --tp 4
"""
import argparse
import sys

from common import (
    add_common_cli, error_table, eta_filename_tag, eta_subtitle, topology_tag,
    load_measured, log_spaced_B, plot_tpot_vs_B, predict_at, run_framework,
)


MODEL = "gpt-oss-120b"
SYSTEM = "h200.8gpu"
PRECISION = "fp4"
ISL, OSL = 1024, 1024
TP_SHAPES = (1, 2, 4, 8)

# Per-stack calibration. Raw TRT-LLM on Hopper fits to ~9% MAE at
# (bw_eta=0.7, c_seq=100 µs/seq) — matches the dsr1_b200_trt
# calibration, suggesting the c_seq for raw TRT-LLM is consistent
# across HW generations (Hopper / Blackwell). Per-sequence host overhead
# is primarily a property of the framework, not the GPU.
DEFAULT_BW_ETA = 1.0
DEFAULT_C_SEQ_US = 100.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--tp", type=int, choices=TP_SHAPES, default=None,
                    help="Run only this TP shape (default: all four)")
    add_common_cli(ap, default_bw_eta=DEFAULT_BW_ETA, default_c_seq_us=DEFAULT_C_SEQ_US)
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
            model="gpt_oss_120b", system_id=SYSTEM,
            PP=1, TP=tp, EP=1, SP=1,
            attention_mode="tp", tp_ep_layout="orthogonal",
            num_devices=tp, S_decode=ISL + OSL // 2,
            B_sweep=log_spaced_B(512),
            flops_eta=args.flops_eta, bw_eta=args.bw_eta,
            c_seq_us=args.c_seq_us,
        )
        for m in measured:
            pred = predict_at(
                model="gpt_oss_120b", system_id=SYSTEM,
                PP=1, TP=tp, EP=1, SP=1,
                attention_mode="tp", tp_ep_layout="orthogonal",
                num_devices=tp, S_decode=ISL + OSL // 2, B=m.B,
                flops_eta=args.flops_eta, bw_eta=args.bw_eta,
                c_seq_us=args.c_seq_us,
            )
            all_rows.append((f"TP={tp}", m.B, m.tpot_ms, pred))

        out = args.out_dir / f"gpt_oss_120b_h200_trt_tp{tp}{eta_filename_tag(args.flops_eta, args.bw_eta, args.c_seq_us)}.png"
        plot_tpot_vs_B(
            framework=framework, measured=measured,
            title=f"gpt-oss-120b / H200 / TRT-LLM — TP={tp} on {tp}-GPU server",
            subtitle=f"PP=1 TP={tp} EP=1 attention_mode=tp | ISL={ISL} OSL={OSL} FP4 | "
                     f"sys={SYSTEM} | {topology_tag(SYSTEM)} | {eta_subtitle(args.flops_eta, args.bw_eta, args.c_seq_us)}",
            out_path=out,
        )
        print(f"  saved: {out.relative_to(args.out_dir.parent.parent)}")

    if all_rows:
        print()
        print(error_table(all_rows, title="gpt-oss-120b / H200 / TRT-LLM — framework vs InferenceX"))
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
