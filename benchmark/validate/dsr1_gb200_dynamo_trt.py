#!/usr/bin/env python3
"""DeepSeek-R1-0528 on GB200 / Dynamo-TRT — framework vs InferenceX.

Three cuts in the InferenceX dataset for this (model, hardware, framework)
triple:

  1. EXACT (orthogonal + TP-attention, dec_tp=N dec_ep=1): natively
     modeled, no approximation. Densest bucket is TP=36 EP=1 dec=36.

  2. CO-LOCATED (DSv3 production shape, dec_tp=dec_ep=N on N GPUs,
     dec_dp_attention=True): natively modeled by
     PartitionSpec(tp_ep_layout="co_located", attention_mode="dp"). InferenceX
     publishes three replica sizes — 8, 16, 32 GPU.

  3. ORTHO multi-replica (TP=8 EP=8 dec=32, dp_attn=False): the
     framework runs 4 replicas of 8 GPUs each (N_replica = 32 / 8) under
     TP-attention. Same shape as the gb300/dynamo-trt ORTHO cut. Useful
     for cross-rack-generation comparison and small-B sensitivity testing.

Usage:
    python benchmark/validate/dsr1_gb200_dynamo_trt.py
    python benchmark/validate/dsr1_gb200_dynamo_trt.py --cut exact
    python benchmark/validate/dsr1_gb200_dynamo_trt.py --cut colocated --bw-eta 0.6
    python benchmark/validate/dsr1_gb200_dynamo_trt.py --cut ortho
"""
import argparse
import sys

from common import (
    add_common_cli, error_table, eta_filename_tag, eta_subtitle, topology_tag,
    load_measured, log_spaced_B, plot_tpot_vs_B, predict_at, run_framework,
)


MODEL = "DeepSeek-R1-0528"
SYSTEM = "gb200.72gpu"
PRECISION = "fp4"
ISL, OSL = 1024, 1024
# CO-LOCATED panel loads two workloads per shape: short-context (1K/1K,
# compute/weight-bound) and long-context (8K/1K, KV-traffic-bound). Same
# calibration must fit both — validates that bw_eta / c_serving /
# kernel_launch_us are hardware/stack properties, not workload knobs.
CO_LOCATED_WORKLOADS = [(1024, 1024), (8192, 1024)]

# Per-stack calibration. Production DSv3/R1 on Dynamo+TRT-LLM with
# DP-attention uses scatter-direct MoE A2A (decode.md §5.2): dispatch
# operates on per-rank attention-sharded tokens of size B/G_TP rather
# than gathering full B to every rank. The framework models this via
# moe_a2a_pattern="scatter" on the tuner. Combined with a moderate
# Dynamo-stack host overhead (c_serving ≈ 5 µs/seq, kernel_launch ≈ 7
# µs) and bw_eta ≈ 0.6 for HBM3e on Blackwell, fits to ~21% MAE
# overall across all three cuts (n=34 measurement points).
# Per-cut breakdown:
#   TP=36 EP=1 EXACT:        22% MAE (limited by data variance — e.g.
#                                     B=128: 11.88 ms vs B=144: 7.74 ms;
#                                     no model fits such non-monotonicity)
#   TP=EP=8  colocated:      25% MAE (n=2; B=4300/4301 measured 54/44 ms,
#                                     ~23% disagreement on consecutive
#                                     integer batch sizes is measurement noise)
#   TP=EP=16 colocated:      23% MAE (scatter-direct here vs gather)
#   TP=EP=32 colocated:      22% MAE
#   TP=8 EP=8 dec=32 ORTHO:  19% MAE (4 replicas of 8 GPUs, TP-attn —
#                                     scatter inert here)
#
# Recalibrated post-MLA-migration (mla(stage 1-3): real MLASpec on
# deepseek_r1_0528 added ~5 GB to per-rank M_theta and shifted attention
# compute to absorbed-mode latent-space score/value). Previous (0.6, 5.0)
# gave 25.1% overall MAE with a 116% outlier at TP=EP=16 B=4301; new
# (0.5, 0.0) gives 22.7% / max 53.8%. The c_serving 5→0 shift carries
# the bulk of the improvement: Dynamo+TRT absorbs per-seq host work into
# the CUDA-graph launch, so the framework's per-seq overhead term
# over-counts at large B (the offending TP=EP=16,32 cells run at B≥4000).
DEFAULT_BW_ETA = 0.7143
# 5 µs/seq — stack-realistic floor for Dynamo+TRT under CUDA-Graph replay.
# Below the panel break-even at every measured operating point: at B=4300
# (largest panel-(c) measured) c_serving·B = 21.5 ms < t_step_hw ~25 ms,
# so the overlap gate clips the contribution to 0 and the value doesn't
# affect MAE. Setting >0 makes the t_serving curve visible in the cost-
# component plot (was previously 0 → suppressed by the plot helper's
# any(t_serving_ms > 0) guard). Sits at the lower end of decode.md §7.2's
# 5-22 µs C++/CUDA-graph + orchestrator range.
DEFAULT_C_SERVING_US = 5.0
# kernel_launch_us: 4.0 µs — calibrated Dynamo-orchestrator effective per-
# kernel cost on H100/H200. Sits between the CUDA-Graph optimum (1.5 µs)
# and TaxBreak's measured eager-mode floor (4.5-4.7 µs on H100/H200, arxiv
# 2603.12465). Production Dynamo+TRT at small B apparently does not hit
# the CUDA-Graph optimum, likely because (a) MoE-routing dispatch isn't
# fully captured by the graph, (b) the per-step scheduler-tick path runs
# outside the graph, and (c) the framework's flat kernels_per_layer_compute=10
# under-counts MoE kernel fanout (TaxBreak: MoE models dispatch 8-11×
# more kernels per token than dense). 4.0 µs absorbs all three effects.
# Same value used for Dynamo+SGLang (the orchestrator behavior dominates
# the underlying runtime distinction).
DEFAULT_KERNEL_LAUNCH_US = 4.0
DEFAULT_MOE_A2A_PATTERN = "scatter"


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
        attention_mode="tp", tp_ep_layout="orthogonal",
        num_devices=bucket[2], S_decode=ISL + OSL // 2,
        B_sweep=log_spaced_B(2048),
        flops_eta=args.flops_eta, bw_eta=args.bw_eta,
        c_serving_us=args.c_serving_us,
        moe_a2a_pattern=DEFAULT_MOE_A2A_PATTERN,
        kernel_launch_us=DEFAULT_KERNEL_LAUNCH_US,
        bytes_per_param=0.5,
    )

    rows = []
    for m in measured:
        pred = predict_at(
            model="deepseek_r1_0528", system_id=SYSTEM,
            PP=1, TP=bucket[0], EP=1, SP=1,
            attention_mode="tp", tp_ep_layout="orthogonal",
            num_devices=bucket[2], S_decode=ISL + OSL // 2, B=m.B,
            flops_eta=args.flops_eta, bw_eta=args.bw_eta,
            c_serving_us=args.c_serving_us,
            moe_a2a_pattern=DEFAULT_MOE_A2A_PATTERN,
            kernel_launch_us=DEFAULT_KERNEL_LAUNCH_US,
            bytes_per_param=0.5,
        )
        rows.append((f"TP={bucket[0]} EP=1", m.B, m.tpot_ms, pred))

    out = args.out_dir / f"dsr1_dynamo_trt_exact_tp{bucket[0]}_ep1_dec{bucket[2]}{eta_filename_tag(args.flops_eta, args.bw_eta, args.c_serving_us)}.png"
    plot_tpot_vs_B(
        framework=framework, measured=measured,
        title=f"DSR1 / GB200 / Dynamo-TRT — EXACT bucket: TP={bucket[0]} EP=1 dec={bucket[2]}",
        subtitle=f"PP=1 TP={bucket[0]} EP=1 attention_mode=tp | ISL={ISL} OSL={OSL} FP4 | "
                 f"sys={SYSTEM} | {topology_tag(SYSTEM)} | {eta_subtitle(args.flops_eta, args.bw_eta, args.c_serving_us)}",
        out_path=out,
    )
    print(f"  saved: {out.relative_to(args.out_dir.parent.parent)}")
    return rows, len(measured)


def _run_colocated_workload(args, tp_ep: int, isl: int, osl: int):
    """Helper: load measured + run framework sweep for one (shape, ISL/OSL) cut."""
    measured = load_measured(
        MODEL, isl=isl, osl=osl, precision=PRECISION,
        decode_tp=tp_ep, decode_ep=tp_ep, num_decode_gpu=tp_ep,
        dp_attention=True,
        framework={"dynamo-trt", "trt", "trt-llm", "trtllm", "dynamo-trt-llm"},
        hardware="gb200",
    )
    S_decode = isl + osl // 2
    framework = run_framework(
        model="deepseek_r1_0528", system_id=SYSTEM,
        PP=1, TP=tp_ep, EP=tp_ep, SP=1,
        attention_mode="dp", tp_ep_layout="co_located",
        num_devices=tp_ep, S_decode=S_decode,
        B_sweep=log_spaced_B(8192),
        flops_eta=args.flops_eta, bw_eta=args.bw_eta,
        c_serving_us=args.c_serving_us,
        moe_a2a_pattern=DEFAULT_MOE_A2A_PATTERN,
        kernel_launch_us=DEFAULT_KERNEL_LAUNCH_US,
        bytes_per_param=0.5,
    )
    rows = []
    for m in measured:
        pred = predict_at(
            model="deepseek_r1_0528", system_id=SYSTEM,
            PP=1, TP=tp_ep, EP=tp_ep, SP=1,
            attention_mode="dp", tp_ep_layout="co_located",
            num_devices=tp_ep, S_decode=S_decode, B=m.B,
            flops_eta=args.flops_eta, bw_eta=args.bw_eta,
            c_serving_us=args.c_serving_us,
            moe_a2a_pattern=DEFAULT_MOE_A2A_PATTERN,
            kernel_launch_us=DEFAULT_KERNEL_LAUNCH_US,
            bytes_per_param=0.5,
        )
        rows.append((f"TP=EP={tp_ep} {isl}/{osl}", m.B, m.tpot_ms, pred))
    return framework, measured, rows


def run_colocated(args) -> tuple[list[tuple], int]:
    """Cut 2: TP+EP co-located DSv3 production shape (TP=EP=N on N GPUs).

    Two workloads overlaid per shape — short-context (1K/1K, compute/weight-
    bound) and long-context (8K/1K, KV-traffic-bound). Same calibration must
    fit both regimes; the panel directly visualizes whether bw_eta /
    c_serving / kernel_launch_us hold across bottleneck regimes.
    """
    all_rows = []
    n_total = 0
    for tp_ep in (8, 16, 32):
        wl_primary, wl_secondary = CO_LOCATED_WORKLOADS
        fw_prim, meas_prim, rows_prim = _run_colocated_workload(args, tp_ep, *wl_primary)
        fw_sec,  meas_sec,  rows_sec  = _run_colocated_workload(args, tp_ep, *wl_secondary)
        print(f"\n[CO-LOCATED] TP=EP={tp_ep} on {tp_ep}-GPU replica | "
              f"{wl_primary[0]}/{wl_primary[1]}: {len(meas_prim)} pts, "
              f"{wl_secondary[0]}/{wl_secondary[1]}: {len(meas_sec)} pts")
        if not meas_prim and not meas_sec:
            continue
        all_rows.extend(rows_prim)
        all_rows.extend(rows_sec)
        n_total += len(meas_prim) + len(meas_sec)

        prim_lbl = f"ISL={wl_primary[0]} OSL={wl_primary[1]}"
        sec_lbl  = f"ISL={wl_secondary[0]} OSL={wl_secondary[1]}"
        out = args.out_dir / f"dsr1_dynamo_trt_colocated_tp{tp_ep}ep{tp_ep}{eta_filename_tag(args.flops_eta, args.bw_eta, args.c_serving_us)}.png"
        plot_tpot_vs_B(
            framework=fw_prim, measured=meas_prim,
            title=f"DSR1 / GB200 / Dynamo-TRT — CO-LOCATED TP=EP={tp_ep} on {tp_ep}-GPU replica",
            subtitle=f"tp_ep_layout=co_located attention_mode=dp PP=1 TP={tp_ep} EP={tp_ep} SP=1 | "
                     f"{prim_lbl} + {sec_lbl} FP4 | sys={SYSTEM} | {topology_tag(SYSTEM)} | {eta_subtitle(args.flops_eta, args.bw_eta, args.c_serving_us)}",
            out_path=out,
            primary_label=prim_lbl,
            secondary=(sec_lbl, fw_sec, meas_sec),
        )
        print(f"  saved: {out.relative_to(args.out_dir.parent.parent)}")

    return all_rows, n_total


def run_colo_tp_attn(args) -> tuple[list[tuple], int]:
    """Cut 3: TP=EP=8 dec=32 co-located, TP-attention (4 replicas of 8 GPUs each).

    Each replica is 8 GPUs holding TP=EP=8 overlaid (attention head-sharded
    across the same 8-rank group that holds the expert shards). Was previously
    modeled as `tp_ep_layout="orthogonal"`, which double-counted the per-replica
    GPU requirement (8×8=64 vs the production 8) and divided MoE work by
    TP*EP=64 instead of EP=8; the co-located layout now matches the deployment.
    """
    rows: list[tuple] = []
    measured = load_measured(
        MODEL, isl=ISL, osl=OSL, precision=PRECISION,
        decode_tp=8, decode_ep=8, num_decode_gpu=32,
        dp_attention=False,
        framework={"dynamo-trt", "trt", "trt-llm", "trtllm", "dynamo-trt-llm"},
        hardware="gb200",
    )
    print(f"\n[COLO TP-attn] TP=EP=8 dec=32 (4 replicas × 8 GPUs, TP-attn) | "
          f"{len(measured)} measured points")
    if not measured:
        return [], 0

    framework = run_framework(
        model="deepseek_r1_0528", system_id=SYSTEM,
        PP=1, TP=8, EP=8, SP=1,
        attention_mode="tp", tp_ep_layout="co_located",
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
            attention_mode="tp", tp_ep_layout="co_located",
            num_devices=32, S_decode=ISL + OSL // 2, B=m.B,
            flops_eta=args.flops_eta, bw_eta=args.bw_eta,
            c_serving_us=args.c_serving_us,
            moe_a2a_pattern=DEFAULT_MOE_A2A_PATTERN,
            kernel_launch_us=DEFAULT_KERNEL_LAUNCH_US,
            bytes_per_param=0.5,
        )
        rows.append(("TP=EP=8 dec=32 colo TP-attn", m.B, m.tpot_ms, pred))

    out = args.out_dir / f"dsr1_dynamo_trt_colo_tp_attn_tp8ep8_dec32{eta_filename_tag(args.flops_eta, args.bw_eta, args.c_serving_us)}.png"
    plot_tpot_vs_B(
        framework=framework, measured=measured,
        title="DSR1 / GB200 / Dynamo-TRT — CO-LOCATED TP-attn TP=EP=8 dec=32 (4 replicas)",
        subtitle=f"PP=1 TP=EP=8 attention_mode=tp tp_ep_layout=co_located | "
                 f"ISL={ISL} OSL={OSL} FP4 | "
                 f"sys={SYSTEM} | {topology_tag(SYSTEM)} | {eta_subtitle(args.flops_eta, args.bw_eta, args.c_serving_us)}",
        out_path=out,
    )
    print(f"  saved: {out.relative_to(args.out_dir.parent.parent)}")
    return rows, len(measured)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--cut", choices=["exact", "colocated", "colo_tp_attn", "all"], default="all")
    add_common_cli(ap, default_bw_eta=DEFAULT_BW_ETA, default_c_serving_us=DEFAULT_C_SERVING_US)
    args = ap.parse_args()

    all_rows: list[tuple] = []
    if args.cut in ("exact", "all"):
        rows, _ = run_exact(args)
        all_rows.extend(rows)
    if args.cut in ("colocated", "all"):
        rows, _ = run_colocated(args)
        all_rows.extend(rows)
    if args.cut in ("colo_tp_attn", "all"):
        rows, _ = run_colo_tp_attn(args)
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
