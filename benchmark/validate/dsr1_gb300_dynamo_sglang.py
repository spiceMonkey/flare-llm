#!/usr/bin/env python3
"""DeepSeek-R1-0528 on GB300 / Dynamo+SGLang — framework vs InferenceX.

Three cuts in the InferenceX dataset for this (model, hardware, framework)
triple, all `disagg=True` (separate prefill cluster):

  1. EXACT (orthogonal + TP-attention, dec_tp=4 dec_ep=1 dec=8 or 16):
     framework-modelable directly.
  2. CO-LOCATED (DSv3 production shape, dec_tp=dec_ep=N on N GPUs,
     dec_dp_attention=True): natively modeled by
     PartitionSpec(tp_ep_layout="co_located", attention_mode="dp"). InferenceX
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
    add_common_cli, error_table, eta_filename_tag, eta_subtitle, topology_tag,
    load_measured, log_spaced_B, plot_tpot_vs_B, predict_at, run_framework,
)


MODEL = "DeepSeek-R1-0528"
SYSTEM = "gb300.72gpu"
PRECISION = "fp4"
ISL, OSL = 1024, 1024

# Per-stack calibration. Dynamo+SGLang on GB300 with --enable-deepep-moe
# uses the scatter-direct MoE A2A pattern (decode.md §5.2): dispatch
# operates on per-rank attention-sharded tokens of size B/G_TP rather
# than gathering full B to every rank. The framework models this via
# moe_a2a_pattern="scatter" on the tuner. Combined with realistic per-
# step host overhead (c_serving ≈ 1 µs/seq, kernel_launch ≈ 12 µs) and
# bw_eta ≈ 0.9 for HBM3e on Blackwell Ultra, the colocated cut fits to
# ~9% MAE across TP=EP={32,48} and B ∈ {512..8192}. The EXACT cut
# (TP=4 EP=1, EP-inert) is harder to fit at small B (kernel-launch floor
# dominates 4-token decode); ~38% MAE there is structural and not
# improvable with current knobs. Overall ~17% MAE across both cuts.
#
# Recalibrated post-MLA-migration (mla(stage 1-3): real MLASpec on
# deepseek_r1_0528). Previous (0.9, 1.0) gave 70.4% overall MAE; new
# (0.9, 0.0) gives 57.7% / max 242.5%. Same root cause as the other
# Dynamo drivers: c_serving 1→0 since Dynamo+SGLang absorbs per-seq
# work into the orchestrator. Note this driver still has high residual
# MAE — the colocated TP=EP=32, 48 cells at B≥4000 are structurally
# over-predicted (likely a shared-MoE / DeepEP hot-path the framework
# doesn't yet model precisely); not addressable via the per-stack knobs.
DEFAULT_BW_ETA = 1.0
# 5 µs/seq — stack-realistic floor below the panel break-even at the
# measured operating points. At B=8192 (largest panel-(d) measured)
# c_serving·B = 41 ms ≈ t_GPU_step ~30-40 ms, so the overlap gate may
# contribute a few ms but doesn't dominate. Setting >0 makes the
# t_serving curve visible in the cost-component plot. Production
# Dynamo+SGLang under CUDA-Graph replay amortizes host work better
# than the docstring's 25-50 µs Python-heavy range would predict,
# hence the calibrated low value here (the JSON sglang.json keeps
# the 40 µs Python-heavy default for general-purpose users).
DEFAULT_C_SERVING_US = 5.0
# 12 µs — matches dynamo_sglang.json's canonical anchor. The Dynamo
# orchestrator absorbs SGLang's per-sequence host work (c_serving, ρ_serving)
# but does NOT collapse SGLang's per-kernel Python wrapping — many SGLang-
# side kernels (custom scatter-direct A2A, MLA-specific kernels) fall
# outside the Dynamo CUDA Graph and retain Python interpreter overhead.
# Empirically validated on this DSr1 / GB300 cut: at the lower dynamo_trt
# anchor (4 µs) the predicted small-B floor under-shoots measured ~13 ms
# by ~3×; the 12 µs value (the pure-SGLang Python-paths anchor) lifts
# the predicted floor to ~13 ms matching observation.
DEFAULT_KERNEL_LAUNCH_US = 12.0
DEFAULT_MOE_A2A_PATTERN = "scatter"


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
            attention_mode="tp", tp_ep_layout="orthogonal",
            num_devices=dec, S_decode=ISL + OSL // 2,
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
                PP=1, TP=4, EP=1, SP=1,
                attention_mode="tp", tp_ep_layout="orthogonal",
                num_devices=dec, S_decode=ISL + OSL // 2, B=m.B,
                flops_eta=args.flops_eta, bw_eta=args.bw_eta,
                c_serving_us=args.c_serving_us,
                moe_a2a_pattern=DEFAULT_MOE_A2A_PATTERN,
                kernel_launch_us=DEFAULT_KERNEL_LAUNCH_US,
                bytes_per_param=0.5,
            )
            rows.append((f"TP=4 EP=1 dec={dec}", m.B, m.tpot_ms, pred))

        out = args.out_dir / f"dsr1_gb300_dynamo_sglang_exact_tp4_ep1_dec{dec}{eta_filename_tag(args.flops_eta, args.bw_eta, args.c_serving_us)}.png"
        plot_tpot_vs_B(
            framework=framework, measured=measured,
            title=f"DSR1 / GB300 / Dynamo+SGLang — EXACT bucket: TP=4 EP=1 dec={dec}",
            subtitle=f"PP=1 TP=4 EP=1 attention_mode=tp | ISL={ISL} OSL={OSL} FP4 | "
                     f"sys={SYSTEM} | {topology_tag(SYSTEM)} | {eta_subtitle(args.flops_eta, args.bw_eta, args.c_serving_us)}",
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
            attention_mode="dp", tp_ep_layout="co_located",
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
                attention_mode="dp", tp_ep_layout="co_located",
                num_devices=tp_ep, S_decode=ISL + OSL // 2, B=m.B,
                flops_eta=args.flops_eta, bw_eta=args.bw_eta,
                c_serving_us=args.c_serving_us,
                moe_a2a_pattern=DEFAULT_MOE_A2A_PATTERN,
                kernel_launch_us=DEFAULT_KERNEL_LAUNCH_US,
                bytes_per_param=0.5,
            )
            rows.append((f"TP=EP={tp_ep}", m.B, m.tpot_ms, pred))

        out = args.out_dir / f"dsr1_gb300_dynamo_sglang_colocated_tp{tp_ep}ep{tp_ep}{eta_filename_tag(args.flops_eta, args.bw_eta, args.c_serving_us)}.png"
        plot_tpot_vs_B(
            framework=framework, measured=measured,
            title=f"DSR1 / GB300 / Dynamo+SGLang — CO-LOCATED TP=EP={tp_ep} on {tp_ep}-GPU replica",
            subtitle=f"tp_ep_layout=co_located attention_mode=dp PP=1 TP={tp_ep} EP={tp_ep} SP=1 | "
                     f"ISL={ISL} OSL={OSL} FP4 | sys={SYSTEM} | {topology_tag(SYSTEM)} | {eta_subtitle(args.flops_eta, args.bw_eta, args.c_serving_us)}",
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
