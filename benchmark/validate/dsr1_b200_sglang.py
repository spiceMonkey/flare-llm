#!/usr/bin/env python3
"""DeepSeek-R1-0528 on B200 multi-box / SGLang — framework vs InferenceX.

Multi-tier deployment validator. 16 B200 GPUs span 2 DGX boxes joined by
ConnectX-8 XDR InfiniBand:

  - TP=4 EP=4 orthogonal layout, dp_attention=False (TP-attn)
  - TP groups stay intra-box on NVLink5  (α=0.5 µs, 900 GB/s)
  - EP groups cross IB between boxes     (α=2.5 µs, 100 GB/s)
  - MoE A2A dispatch hits the IB tier each layer

Sweeps three workloads on the same shape — short symmetric (1K/1K),
KV-deep decode (8K/1K), long-generation decode (1K/8K). 6 geometric B
values per workload × 3 = 18 measured rows, exercising the compute-bound
→ memory-bound transition under a real cross-island fabric chain.

Usage:
    python benchmark/validate/dsr1_b200_sglang.py
    python benchmark/validate/dsr1_b200_sglang.py --bw-eta 0.7
"""
import argparse
import sys

from common import (
    add_common_cli, error_table, eta_filename_tag, eta_subtitle, topology_tag,
    load_measured, log_spaced_B, plot_tpot_vs_B, predict_at, run_framework,
)


MODEL = "DeepSeek-R1-0528"
SYSTEM = "b200.multibox"
PRECISION = "fp4"
TP, EP, NUM = 4, 4, 16
# Long-generation workload (1K input, 8K output) — KV grows during decode
# from 1K to 9K, mean S_decode = 5K. Tests the model's KV-traffic scaling
# under the multi-tier IB fabric chain. Per-row S_decode = ISL + OSL//2
# (linear-average KV depth across the decode window).
WORKLOADS = [(1024, 8192)]

# Per-stack calibration. Pure SGLang (no Dynamo orchestrator) is Python-
# heavy: c_serving=40 µs/seq with serving_overlap_factor=0.0 (decode.md
# §7.2 — Python interpreter wrapping per-seq sampling breaks the CPU-
# runs-ahead invariant, so host work serializes after GPU compute).
# Kernel launch ≈ 10 µs/kernel with kernel_overlap_factor=0.3 (eager-mode
# floor + interpreter stalls limit dispatch overlap). These knobs live
# in `database/framework/sglang.json`; the validator passes them through
# explicitly so this file documents what's calibrated.
DEFAULT_BW_ETA = 0.7
DEFAULT_C_SERVING_US = 40.0
DEFAULT_KERNEL_LAUNCH_US = 10.0
DEFAULT_MOE_A2A_PATTERN = "scatter"
# Python-heavy SGLang stack: serving cost fully serializes (ρ_serving=0),
# kernel-dispatch overlap limited to 0.3 (eager-mode + interpreter stalls),
# no async comm overlap (ρ_comm=0). All three match database/framework/sglang.json.
DEFAULT_SERVING_OVERLAP = 0.0
DEFAULT_KERNEL_OVERLAP = 0.3
DEFAULT_COMM_OVERLAP = 0.0


def _run_workload(args, isl: int, osl: int):
    """Run framework sweep + collect per-row predictions for one workload."""
    measured = load_measured(
        MODEL, isl=isl, osl=osl, precision=PRECISION,
        decode_tp=TP, decode_ep=EP, num_decode_gpu=NUM,
        dp_attention=False,
        framework={"sglang"},
        hardware="b200",
    )
    S_decode = isl + osl // 2
    framework = run_framework(
        model="deepseek_r1_0528", system_id=SYSTEM,
        PP=1, TP=TP, EP=EP, SP=1,
        attention_mode="tp", tp_ep_layout="orthogonal",
        num_devices=NUM, S_decode=S_decode,
        B_sweep=log_spaced_B(512),
        flops_eta=args.flops_eta, bw_eta=args.bw_eta,
        c_serving_us=args.c_serving_us,
        moe_a2a_pattern=DEFAULT_MOE_A2A_PATTERN,
        kernel_launch_us=DEFAULT_KERNEL_LAUNCH_US,
        bytes_per_param=0.5,
        serving_overlap_factor=DEFAULT_SERVING_OVERLAP,
        kernel_overlap_factor=DEFAULT_KERNEL_OVERLAP,
        comm_overlap_factor=DEFAULT_COMM_OVERLAP,
    )
    rows = []
    for m in measured:
        pred = predict_at(
            model="deepseek_r1_0528", system_id=SYSTEM,
            PP=1, TP=TP, EP=EP, SP=1,
            attention_mode="tp", tp_ep_layout="orthogonal",
            num_devices=NUM, S_decode=S_decode, B=m.B,
            flops_eta=args.flops_eta, bw_eta=args.bw_eta,
            c_serving_us=args.c_serving_us,
            moe_a2a_pattern=DEFAULT_MOE_A2A_PATTERN,
            kernel_launch_us=DEFAULT_KERNEL_LAUNCH_US,
            bytes_per_param=0.5,
            serving_overlap_factor=DEFAULT_SERVING_OVERLAP,
            kernel_overlap_factor=DEFAULT_KERNEL_OVERLAP,
            comm_overlap_factor=DEFAULT_COMM_OVERLAP,
        )
        rows.append((f"TP={TP} EP={EP} {isl}/{osl}", m.B, m.tpot_ms, pred))
    return framework, measured, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    add_common_cli(ap, default_bw_eta=DEFAULT_BW_ETA,
                   default_c_serving_us=DEFAULT_C_SERVING_US)
    args = ap.parse_args()

    cuts = [_run_workload(args, isl, osl) for (isl, osl) in WORKLOADS]
    all_rows = []
    for (isl, osl), (_, meas, rows) in zip(WORKLOADS, cuts):
        print(f"[B200/SGLang multi-tier] TP={TP} EP={EP} g={NUM} | "
              f"ISL={isl} OSL={osl} | {len(meas)} measured points")
        all_rows.extend(rows)

    fw_prim, meas_prim, _ = cuts[0]
    prim_lbl = f"ISL={WORKLOADS[0][0]} OSL={WORKLOADS[0][1]}"

    out = (
        args.out_dir
        / f"dsr1_b200_sglang_tp{TP}_ep{EP}_g{NUM}{eta_filename_tag(args.flops_eta, args.bw_eta, args.c_serving_us)}.png"
    )
    plot_tpot_vs_B(
        framework=fw_prim, measured=meas_prim,
        title=f"DSR1 / B200 multi-box / SGLang — TP={TP} EP={EP} on {NUM} GPUs (2 boxes via IB)",
        subtitle=f"PP=1 TP={TP} EP={EP} attention_mode=tp tp_ep_layout=orthogonal | "
                 f"ISL={WORKLOADS[0][0]} OSL={WORKLOADS[0][1]} long-gen | FP4 | "
                 f"sys={SYSTEM} | {topology_tag(SYSTEM)} | "
                 f"{eta_subtitle(args.flops_eta, args.bw_eta, args.c_serving_us)}",
        out_path=out,
        primary_label=prim_lbl,
    )
    print(f"  saved: {out.relative_to(args.out_dir.parent.parent)}\n")
    print(error_table(all_rows, title="DSR1 / B200 multi-box / SGLang — framework vs InferenceX"))

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
