#!/usr/bin/env python3
"""DeepSeek-R1-0528 on GB300 / Dynamo+SGLang — framework vs InferenceX.

Three cuts in the InferenceX dataset for this (model, hardware, framework)
triple, all `disagg=True` (separate prefill cluster):

  1. EXACT (orthogonal + TP-attention, dec_tp=4 dec_ep=1 dec=8 or 16):
     framework-modelable directly.
  2. CO-LOCATED (DSv3 production shape, dec_tp=dec_ep=N on N GPUs,
     dec_dp_attention=True): natively modeled by
     PartitionSpec(layout="co_located", attention_mode="dp"). InferenceX
     publishes three replica sizes — 8, 32, 48 GPU.

Cross-check vs `dsr1_gb200_dynamo_trt.py`: same model, different rack
generation (gb300 = Blackwell Ultra) + different framework class
(Dynamo+SGLang vs Dynamo+TRT).

Usage:
    python benchmark/validate/dsr1_gb300_dynamo_sglang.py
    python benchmark/validate/dsr1_gb300_dynamo_sglang.py --cut colocated
"""
import argparse
import sys

from common import (
    add_common_cli, error_table, eta_filename_tag, eta_subtitle,
    load_measured, log_spaced_B, plot_tpot_vs_B, predict_at, run_framework,
)


MODEL = "DeepSeek-R1-0528"
SYSTEM = "gb300.72gpu"
PRECISION = "fp4"
ISL, OSL = 1024, 1024

# Per-stack calibration. Dynamo+SGLang on GB300 is the worst-fitting cut
# in the validator suite — even the best-fit knobs leave ~55% MAE. The
# signed-error pattern (under-predict at small B, over-predict at huge
# B>4000) is consistent with a missing B-saturation correction
# (`bw-eta-vs-batch` TODO in `scratch/model_specific_extensions.md`).
#
# We DEFAULT to (bw_eta=0.4, c_serving=0): the empirically best-fit tuple
# from `_calibrate.py`. A non-zero c_serving multiplies the error at huge
# B (a c_serving=25 µs/seq makes t_serving=200 ms at B=8192, which is
# larger than the entire measured TPOT of ~23 ms — the linear-in-B
# t_serving model breaks down outside the 1≤B≤1024 range per `decode.md
# §7.2`). Users who care about this stack should sweep the knobs and
# accept that the framework can't fit this case well without the
# B-saturation extension.
DEFAULT_BW_ETA = 0.4
DEFAULT_C_SERVING_US = 0.0


def run_exact(args) -> tuple[list[tuple], int]:
    """Cut 1: TP-only orthogonal, TP=4 EP=1 across 8 or 16 decode GPUs."""
    rows: list[tuple] = []
    n = 0
    for dec in (8, 16):
        measured = load_measured(
            MODEL, isl=ISL, osl=OSL, precision=PRECISION,
            decode_tp=4, decode_ep=1, num_decode_gpu=dec,
            framework={"dynamo-sglang", "sglang"},
            hardware="gb300",
        )
        print(f"\n[EXACT] TP=4 EP=1 dec={dec}, ISL={ISL} OSL={OSL} {PRECISION} | "
              f"{len(measured)} measured points")
        if not measured:
            continue
        n += len(measured)

        framework = run_framework(
            model="deepseek_r1_0528", system_id=SYSTEM,
            PP=1, TP=4, EP=1, SP=1,
            attention_mode="tp", layout="orthogonal",
            num_devices=dec, S_decode=ISL + OSL // 2,
            B_sweep=log_spaced_B(2048),
            flops_eta=args.flops_eta, bw_eta=args.bw_eta,
            c_serving_us=args.c_serving_us,
            bytes_per_param=0.5,
        )
        for m in measured:
            pred = predict_at(
                model="deepseek_r1_0528", system_id=SYSTEM,
                PP=1, TP=4, EP=1, SP=1,
                attention_mode="tp", layout="orthogonal",
                num_devices=dec, S_decode=ISL + OSL // 2, B=m.B,
                flops_eta=args.flops_eta, bw_eta=args.bw_eta,
                c_serving_us=args.c_serving_us, bytes_per_param=0.5,
            )
            rows.append((f"TP=4 EP=1 dec={dec}", m.B, m.tpot_ms, pred))

        out = args.out_dir / f"dsr1_gb300_dynamo_sglang_exact_tp4_ep1_dec{dec}{eta_filename_tag(args.flops_eta, args.bw_eta, args.c_serving_us)}.png"
        plot_tpot_vs_B(
            framework=framework, measured=measured,
            title=f"DSR1 / GB300 / Dynamo+SGLang — EXACT bucket: TP=4 EP=1 dec={dec}",
            subtitle=f"PP=1 TP=4 EP=1 attention_mode=tp | ISL={ISL} OSL={OSL} FP4 | "
                     f"{eta_subtitle(args.flops_eta, args.bw_eta, args.c_serving_us)}",
            out_path=out,
        )
        print(f"  saved: {out.relative_to(args.out_dir.parent.parent)}")
    return rows, n


def run_colocated(args) -> tuple[list[tuple], int]:
    """Cut 2: TP+EP co-located DSv3 production shape — TP=EP={8,32,48}."""
    rows: list[tuple] = []
    n = 0
    for tp_ep in (8, 32, 48):
        measured = load_measured(
            MODEL, isl=ISL, osl=OSL, precision=PRECISION,
            decode_tp=tp_ep, decode_ep=tp_ep, num_decode_gpu=tp_ep,
            dp_attention=True,
            framework={"dynamo-sglang", "sglang"},
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

        out = args.out_dir / f"dsr1_gb300_dynamo_sglang_colocated_tp{tp_ep}ep{tp_ep}{eta_filename_tag(args.flops_eta, args.bw_eta, args.c_serving_us)}.png"
        plot_tpot_vs_B(
            framework=framework, measured=measured,
            title=f"DSR1 / GB300 / Dynamo+SGLang — CO-LOCATED TP=EP={tp_ep} on {tp_ep}-GPU replica",
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
        print(error_table(all_rows, title="DSR1 / GB300 / Dynamo+SGLang — framework vs InferenceX"))
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
