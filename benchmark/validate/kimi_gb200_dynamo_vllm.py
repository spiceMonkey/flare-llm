#!/usr/bin/env python3
"""Kimi-K2.5 on GB200 NVL72 / Dynamo+vLLM — framework vs InferenceX.

Large-batch DP-attention validator. 16 GB200 GPUs allocated from one NVL72
rack (single-tier NVLink5 NVSwitch5 domain). TP=EP=16 co-located, DP-attn —
each rank holds a 1/16 attention shard and 1/16 expert shard overlaid in
the same NVLink group. Single replica (DP=1).

Stack: Dynamo orchestrator wrapping vLLM runtime. Per-stack calibration
lives in `database/framework/dynamo_vllm.json`; this validator passes the
JSON values through explicitly so the file is self-documenting:
  - c_seq=5 µs/seq, seq_overlap=1.0 (CUDA-Graph absorption)
  - kernel_launch=8 µs (between dynamo_trt's 4 µs and dynamo_sglang's 12 µs)
  - kernel_overlap=1.0, moe_a2a=scatter

Two workloads (1K/1K, 8K/1K), 6 + 4 = 10 measured rows. B range 256-4096
exercises the large-batch regime — KV traffic and comm dominance,
complementary to the low-B compute-bound panel (c). Plot zooms in to
B ≥ 100 to focus on the high-B story.

Usage:
    python benchmark/validate/kimi_gb200_dynamo_vllm.py
    python benchmark/validate/kimi_gb200_dynamo_vllm.py --bw-eta 0.8
"""
import argparse
import sys

from common import (
    add_common_cli, error_table, eta_filename_tag, eta_subtitle, topology_tag,
    load_measured, log_spaced_B, plot_tpot_vs_B, predict_at, run_framework,
)


MODEL = "Kimi-K2.5"
SYSTEM = "gb200.72gpu"
PRECISION = "fp4"
TP, EP, NUM = 16, 16, 16
# Short symmetric workload (1K input, 1K output) — KV depth ~1.5K mean.
# At TP=EP=16 / 16 GPUs / DP-attn, exercises compute-bound and per-step
# host-floor regimes at large B. Flat measured TPOT across B=256-4096
# (19 → 24 ms, only 24% growth over 16× B) is the apparent per-step host
# floor — the c_step TODO motivator. Per-row S_decode = ISL + OSL//2.
WORKLOADS = [(1024, 1024)]

# Per-stack calibration — mirrors database/framework/dynamo_vllm.json.
DEFAULT_BW_ETA = 0.7
DEFAULT_C_SEQ_US = 5.0
DEFAULT_KERNEL_LAUNCH_US = 8.0
DEFAULT_SEQ_OVERLAP = 1.0
DEFAULT_KERNEL_OVERLAP = 1.0
DEFAULT_COMM_OVERLAP = 0.0
DEFAULT_MOE_A2A_PATTERN = "scatter"

# Plot range: start at B=10 to show the full model behavior (t_mem floor
# at small B, growth through the compute-bound knee, large-B comm regime)
# even though measured points only land at B≥256. The gap between the
# growing model curve and the nearly-flat measured at B=256-4096 makes the
# per-step host floor visually unambiguous.
B_PLOT_MIN = 10
B_PLOT_MAX = 10000


def _run_workload(args, isl: int, osl: int):
    """Run framework sweep + collect per-row predictions for one workload."""
    measured = load_measured(
        MODEL, isl=isl, osl=osl, precision=PRECISION,
        decode_tp=TP, decode_ep=EP, num_decode_gpu=NUM,
        dp_attention=True,
        framework={"dynamo-vllm"},
        hardware="gb200",
    )
    S_decode = isl + osl // 2
    framework = run_framework(
        model="kimi_k25", system_id=SYSTEM,
        PP=1, TP=TP, EP=EP, SP=1,
        attention_mode="dp", tp_ep_layout="co_located",
        num_devices=NUM, S_decode=S_decode,
        B_sweep=log_spaced_B(B_PLOT_MAX, B_min=B_PLOT_MIN),
        flops_eta=args.flops_eta, bw_eta=args.bw_eta,
        c_seq_us=args.c_seq_us,
        moe_a2a_pattern=DEFAULT_MOE_A2A_PATTERN,
        kernel_launch_us=DEFAULT_KERNEL_LAUNCH_US,
        bytes_per_param=0.5,
        seq_overlap_factor=DEFAULT_SEQ_OVERLAP,
        kernel_overlap_factor=DEFAULT_KERNEL_OVERLAP,
        comm_overlap_factor=DEFAULT_COMM_OVERLAP,
    )
    rows = []
    for m in measured:
        pred = predict_at(
            model="kimi_k25", system_id=SYSTEM,
            PP=1, TP=TP, EP=EP, SP=1,
            attention_mode="dp", tp_ep_layout="co_located",
            num_devices=NUM, S_decode=S_decode, B=m.B,
            flops_eta=args.flops_eta, bw_eta=args.bw_eta,
            c_seq_us=args.c_seq_us,
            moe_a2a_pattern=DEFAULT_MOE_A2A_PATTERN,
            kernel_launch_us=DEFAULT_KERNEL_LAUNCH_US,
            bytes_per_param=0.5,
            seq_overlap_factor=DEFAULT_SEQ_OVERLAP,
            kernel_overlap_factor=DEFAULT_KERNEL_OVERLAP,
            comm_overlap_factor=DEFAULT_COMM_OVERLAP,
        )
        rows.append((f"TP={TP} EP={EP} {isl}/{osl}", m.B, m.tpot_ms, pred))
    return framework, measured, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    add_common_cli(ap, default_bw_eta=DEFAULT_BW_ETA,
                   default_c_seq_us=DEFAULT_C_SEQ_US)
    args = ap.parse_args()

    cuts = [_run_workload(args, isl, osl) for (isl, osl) in WORKLOADS]
    all_rows = []
    for (isl, osl), (_, meas, rows) in zip(WORKLOADS, cuts):
        print(f"[Kimi/GB200/Dynamo+vLLM] TP={TP} EP={EP} g={NUM} | "
              f"ISL={isl} OSL={osl} | {len(meas)} measured points")
        all_rows.extend(rows)

    fw_prim, meas_prim, _ = cuts[0]
    prim_lbl = f"ISL={WORKLOADS[0][0]} OSL={WORKLOADS[0][1]}"

    out = (
        args.out_dir
        / f"kimi_gb200_dynamo_vllm_tp{TP}_ep{EP}_g{NUM}{eta_filename_tag(args.flops_eta, args.bw_eta, args.c_seq_us)}.png"
    )
    plot_tpot_vs_B(
        framework=fw_prim, measured=meas_prim,
        title=f"Kimi-K2.5 / GB200 NVL72 / Dynamo+vLLM — TP=EP={TP} on {NUM} GPUs (DP-attn)",
        subtitle=f"PP=1 TP={TP} EP={EP} attention_mode=dp tp_ep_layout=co_located | "
                 f"ISL={WORKLOADS[0][0]} OSL={WORKLOADS[0][1]} short-ctx | FP4 | "
                 f"sys={SYSTEM} | {topology_tag(SYSTEM)} | "
                 f"{eta_subtitle(args.flops_eta, args.bw_eta, args.c_seq_us)}",
        out_path=out,
        primary_label=prim_lbl,
        xlim=(B_PLOT_MIN, B_PLOT_MAX),
    )
    print(f"  saved: {out.relative_to(args.out_dir.parent.parent)}\n")
    print(error_table(all_rows, title="Kimi-K2.5 / GB200 / Dynamo+vLLM — framework vs InferenceX"))

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
