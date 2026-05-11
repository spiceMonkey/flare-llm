#!/usr/bin/env python3
"""Comprehensive single-island coverage sweep across InferenceX.

Sweeps every (model, hardware, framework, partition) combination in the
InferenceX dataset that fits on one NVLink island (i.e., dec_gpu ≤
island_size for the chip family) and that the framework currently
supports a model spec for. Uses a per-(hardware, framework)
calibration table for the host-overhead knobs; cells without a
calibrated entry fall back to a stack-class default (Dynamo-style,
raw-Python-heavy, raw-C++).

This is COVERAGE breadth, not depth. For per-cut detailed analysis use
the per-stack drivers (`dsr1_gb200_dynamo_trt.py` etc.). The MAE
numbers here are a first look; per-cell calibration would tighten them.

Usage:
    python benchmark/validate/coverage_sweep.py
    python benchmark/validate/coverage_sweep.py --model DeepSeek-R1-0528
    python benchmark/validate/coverage_sweep.py --hardware gb300
    python benchmark/validate/coverage_sweep.py --framework dynamo-trt
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import dataclasses  # noqa: E402

from common import (  # noqa: E402
    load_measured, predict_at, system_with_eta,
    log_spaced_B, plot_tpot_vs_B, run_framework, topology_tag,
)
from llm_perf import InferenceCalculator  # noqa: E402
from llm_perf.io import load_model_from_db, load_system_from_db  # noqa: E402
from llm_perf.specs.framework_spec import FrameworkSpec  # noqa: E402
from llm_perf.specs.partition_spec import PartitionSpec  # noqa: E402
from llm_perf.specs.tuner_spec import TuningSpec  # noqa: E402


# ────────────────────────────────────────────────────────────────────────────
# Maps and tables
# ────────────────────────────────────────────────────────────────────────────

INFERENCEX_TO_FRAMEWORK_MODEL = {
    "DeepSeek-R1-0528": "deepseek_r1_0528",
    "DeepSeek-V4-Pro": "deepseek_v4_pro",
    "GLM-5": "glm5",
    "Kimi-K2.5": "kimi_k25",
    "Llama-3.3-70B-Instruct-FP8": "llama3.1_70b",
    "MiniMax-M2.5": "minimax_m25",
    "Qwen-3.5-397B-A17B": "qwen35_397b_a17b",
    "gpt-oss-120b": "gpt_oss_120b",
}

# bytes-per-param overrides per InferenceX model (most ship FP8 / FP4)
BYTES_PER_PARAM = {
    "DeepSeek-R1-0528": 0.5,            # FP4 in production
    "DeepSeek-V4-Pro": 0.5,             # FP4 mixed
    "GLM-5": 1.0,                       # FP8/BF16
    "Kimi-K2.5": 1.0,
    "Llama-3.3-70B-Instruct-FP8": 1.0,  # FP8
    "MiniMax-M2.5": 1.0,                # FP8
    "Qwen-3.5-397B-A17B": 1.0,
    "gpt-oss-120b": 0.5,                # FP4 typical
}

# NVLink island sizes per chip family — collectives ≤ this stay on the
# fastest fabric tier; collectives > this escalate to InfiniBand on
# multi-box specs (b200/b300 only as of Phase 3).
ISLAND_SIZE = {"b200": 8, "b300": 8, "gb200": 72, "gb300": 72, "h100": 8, "h200": 8}

# Single-island system specs (NVLink-only fabric).
SYSTEM_ID = {
    "b200": "b200.8gpu", "b300": "b300.8gpu",
    "gb200": "gb200.72gpu", "gb300": "gb300.72gpu",
    "h100": "h100.8gpu", "h200": "h200.8gpu",
}

# Multi-box specs (NVLink intra-box + IB inter-box). Phase 3 covers
# all NVIDIA chip families with InferenceX multi-box rows. Maps
# (chip, dec_gpu) → spec id.
MULTIBOX_SYSTEM_ID = {
    # Blackwell (B-series): NVLink5 + ConnectX-8 XDR
    ("b200", 16): "b200.16gpu", ("b200", 24): "b200.24gpu",
    ("b200", 40): "b200.40gpu", ("b200", 48): "b200.48gpu",
    ("b200", 64): "b200.64gpu",
    ("b300", 16): "b300.16gpu", ("b300", 20): "b300.20gpu",
    ("b300", 24): "b300.24gpu", ("b300", 32): "b300.32gpu",
    ("b300", 40): "b300.40gpu", ("b300", 64): "b300.64gpu",
    # Hopper (H-series): NVLink4 + ConnectX-7 NDR
    ("h100", 16): "h100.16gpu", ("h100", 48): "h100.48gpu",
    ("h200", 16): "h200.16gpu", ("h200", 48): "h200.48gpu",
    ("h200", 56): "h200.56gpu", ("h200", 64): "h200.64gpu",
    ("h200", 72): "h200.72gpu",
}


def system_for(hw: str, dec_gpu: int) -> str | None:
    """Return the best system spec id for a (chip, dec_gpu) tuple, or None
    if no spec exists (multi-box H100/H200/AMD)."""
    if hw not in ISLAND_SIZE:
        return None
    if dec_gpu <= ISLAND_SIZE[hw]:
        return SYSTEM_ID[hw]
    # > island — try multi-box specs
    return MULTIBOX_SYSTEM_ID.get((hw, dec_gpu))


# ────────────────────────────────────────────────────────────────────────────
# Per-(hw, fw) calibration table
# ────────────────────────────────────────────────────────────────────────────
#
# Calibrated entries come from the per-stack drivers; uncalibrated cells
# fall back by stack class.

CALIBRATED = {
    # Recalibrated by dsr1_gb300_dynamo_sglang.py post-MLA-migration (~58% MAE)
    ("gb300", "dynamo-sglang"): dict(bw_eta=1.0, c_serving=0.0, kernel_launch=12.0, pattern="scatter"),
    # Recalibrated by dsr1_gb300_dynamo_trt.py post-MLA-migration (~20% MAE)
    ("gb300", "dynamo-trt"):    dict(bw_eta=1.1111, c_serving=0.0, kernel_launch=7.0,  pattern="scatter"),
    # Recalibrated by dsr1_gb200_dynamo_trt.py post-MLA-migration (~23% MAE across 4 cuts)
    ("gb200", "dynamo-trt"):    dict(bw_eta=0.7143, c_serving=0.0, kernel_launch=7.0,  pattern="scatter"),
    # Calibrated by gpt_oss_120b_gb200_dynamo_trt.py (~9% MAE)
    ("gb200", "dynamo-trt-oss"): dict(bw_eta=1.4286, c_serving=22.0, kernel_launch=1.5, pattern="gather"),
    # Calibrated by llama3_70b_b200_trt.py (~27% MAE)
    ("b200", "trt"):            dict(bw_eta=0.5714, c_serving=0.0,  kernel_launch=1.5, pattern="gather"),
    # Calibrated by llama3_70b_h200_trt.py (~12% MAE)
    ("h200", "trt"):            dict(bw_eta=0.7857, c_serving=75.0, kernel_launch=1.5, pattern="gather"),
}

# Stack-class fallbacks used when (hw, fw) isn't in CALIBRATED. Roughly
# matches `decode.md §7.2` per-stack c_serving table; bw_eta picks per
# chip class from `decode.md §6.2`. Updated post-MLA-migration: the
# Dynamo-orchestrator stacks absorb per-seq host work into the CUDA-graph
# launch, so c_serving for the dynamo-* family is set to 0 (the prior
# 2-5 µs values over-counted at large B once MLA's correct higher
# M_theta and KV-on-TP-attn pushed predicted t_step up).
STACK_CLASS = {
    # Aggressively-fused C++/CUDA-Graph runtimes wrapped by an orchestrator.
    # c_serving = 0 — the orchestrator absorbs per-step bookkeeping
    # entirely into one CUDA-Graph launch; per-sequence overhead is
    # negligible vs the per-step kernel-launch budget.
    "dynamo-trt":     dict(c_serving=0.0,   kernel_launch=7.0,  pattern="scatter"),
    "dynamo-sglang":  dict(c_serving=0.0,   kernel_launch=12.0, pattern="scatter"),
    # dynamo-vllm kept at 2.0 — DSv4-Pro on gb200/dynamo-vllm regresses
    # at 0.0 (29.8% → 33.2% MAE) so the prior 2.0 default is a better
    # fit for that stack. Less evidence than dynamo-trt/sglang since the
    # DSv3-individual drivers don't cover dynamo-vllm cuts.
    "dynamo-vllm":    dict(c_serving=2.0,   kernel_launch=12.0, pattern="scatter"),
    # Raw C++ runtimes (no orchestrator).
    "trt":            dict(c_serving=50.0,  kernel_launch=1.5,  pattern="gather"),
    "trt-llm":        dict(c_serving=50.0,  kernel_launch=1.5,  pattern="gather"),
    "trtllm":         dict(c_serving=50.0,  kernel_launch=1.5,  pattern="gather"),
    # Python-heavy stacks (eager-mode interpreters dominate per-sequence work).
    "vllm":           dict(c_serving=20.0,  kernel_launch=10.0, pattern="scatter"),
    "sglang":         dict(c_serving=20.0,  kernel_launch=10.0, pattern="scatter"),
}

# Per-chip bw_eta default (HBM3e on Blackwell sustains higher than HBM3 on Hopper).
BW_ETA_BY_CHIP = {
    # Per-chip bw_eta default — Phase F: chip baselines now live on
    # DeviceSpec (hbm_eta_beta in system JSONs). This table reduces to
    # the identity (1.0 everywhere); kept for callers that still index
    # by chip name. Stack-specific ratios live in CALIBRATED entries
    # above (per-(hw, fw)) or in per-driver DEFAULT_BW_ETA constants.
    "b200": 1.0, "b300": 1.0,
    "gb200": 1.0, "gb300": 1.0,
    "h100": 1.0, "h200": 1.0,
}


def get_knobs(hw: str, fw: str) -> dict:
    """Lookup calibration knobs for (hardware, framework). Calibrated
    entries win over stack-class fallbacks."""
    if (hw, fw) in CALIBRATED:
        return CALIBRATED[(hw, fw)]
    cls = STACK_CLASS.get(fw, dict(c_serving=30.0, kernel_launch=5.0, pattern="gather"))
    return dict(
        bw_eta=BW_ETA_BY_CHIP.get(hw, 0.7),
        c_serving=cls["c_serving"],
        kernel_launch=cls["kernel_launch"],
        pattern=cls["pattern"],
    )


# ────────────────────────────────────────────────────────────────────────────
# Sweep
# ────────────────────────────────────────────────────────────────────────────


def infer_partition(m) -> tuple[int, int, int, int, str, str]:
    """Pick (PP, TP, EP, SP, attention_mode, tp_ep_layout) from a measured row."""
    PP, SP = 1, 1
    TP, EP = m.decode_tp, max(1, m.decode_ep)
    dec = m.num_decode_gpu
    n_replica_orth = dec // (PP * TP * EP * SP)
    n_replica_colo = dec // (PP * max(TP, EP) * SP)
    if m.decode_dp_attention:
        # DP-attn requires either orthogonal+TP=EP*K or co-located. Heuristic:
        # if max(TP,EP) divides dec evenly under co-located, use co-located.
        if dec % max(TP, EP) == 0 and TP == EP:
            return PP, TP, EP, SP, "dp", "co_located"
        return PP, TP, EP, SP, "dp", "orthogonal"
    return PP, TP, EP, SP, "tp", "orthogonal"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--model", help="Filter to one InferenceX model name")
    ap.add_argument("--hardware", help="Filter to one hardware (e.g., gb300)")
    ap.add_argument("--framework", help="Filter to one framework (e.g., dynamo-trt)")
    ap.add_argument("--isl", type=int, default=1024)
    ap.add_argument("--osl", type=int, default=1024)
    ap.add_argument("--check", type=float, default=None,
                    help="Exit non-zero if overall MAE %% > threshold")
    ap.add_argument("--verbose", action="store_true",
                    help="Print every row, not just per-cell summaries")
    ap.add_argument("--plot", action="store_true",
                    help="Generate one TPOT-vs-B plot per (model, hw, fw, partition shape) cell.")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "benchmark" / "results",
                    help="Directory for plots when --plot is set")
    args = ap.parse_args()
    args.out_dir = Path(args.out_dir)

    # When --plot is set, we also need the framework sweep per partition
    # shape (not just point predictions). Track unique (model, hw, fw,
    # partition shape) tuples seen and plot one per tuple.
    plot_keys_seen: set[tuple] = set()
    measured_by_plot: dict[tuple, list] = defaultdict(list)
    plot_meta: dict[tuple, dict] = {}

    # Aggregator: per (model, hw, fw, label) errors; per-cell aggregates derived
    cell_rows: dict[tuple, list[tuple]] = defaultdict(list)
    skipped_oom = 0
    skipped_other = 0

    inferencex_models = sorted(INFERENCEX_TO_FRAMEWORK_MODEL)
    if args.model:
        inferencex_models = [args.model]

    for inf_model in inferencex_models:
        framework_model = INFERENCEX_TO_FRAMEWORK_MODEL[inf_model]
        bpp = BYTES_PER_PARAM[inf_model]
        ms = load_measured(inf_model, isl=args.isl, osl=args.osl)
        for m in ms:
            if args.hardware and m.hardware != args.hardware:
                continue
            if args.framework and m.framework != args.framework:
                continue
            sys_id = system_for(m.hardware, m.num_decode_gpu)
            if sys_id is None:
                continue  # no spec (multi-box H100/H200 or AMD — Phase 3/4)

            knobs = get_knobs(m.hardware, m.framework)
            PP, TP, EP, SP, attn_mode, tp_ep_layout = infer_partition(m)

            # Pre-flight: skip rows where the model can't structurally fit on
            # the labeled (PP*TP*EP*SP) GPUs. The InferenceX dataset has rows
            # that use FSDP/ZeRO-3 (DP across all GPUs with parameter
            # sharding) which the framework's TP/EP/PP labeling can't
            # represent; predicting them as "model replicated on each rank"
            # blows up by orders of magnitude.
            try:
                spec = load_model_from_db(framework_model)
                spec = dataclasses.replace(spec, bytes_per_param=bpp)
                sys_spec = load_system_from_db(sys_id)
                sys_spec = dataclasses.replace(sys_spec, num_devices=m.num_decode_gpu)
                p_spec = PartitionSpec(PP=PP, TP=TP, EP=EP, SP=SP)
                t_spec = TuningSpec(S_decode=m.isl + m.osl // 2, B_decode=m.B)
                fw_spec = FrameworkSpec(
                    name="precheck",
                    attention_mode=attn_mode,
                    tp_ep_layout=tp_ep_layout,
                    moe_a2a_pattern=knobs["pattern"],
                    kernel_launch_us=knobs["kernel_launch"],
                    c_serving_per_seq_us=knobs["c_serving"],
                )
                r_check = InferenceCalculator(spec, sys_spec, p_spec, t_spec, fw_spec).run()
                if not r_check.memory.fits_in_HBM:
                    skipped_oom += 1
                    if args.verbose:
                        print(f"  SKIP-OOM {inf_model} {m.hardware}/{m.framework} TP={TP} EP={EP} dec={m.num_decode_gpu} dp_attn={m.decode_dp_attention} B={m.B}", file=sys.stderr)
                    continue
            except Exception as e:
                skipped_other += 1
                if args.verbose:
                    print(f"  SKIP-PRECHECK {inf_model} {m.hardware}/{m.framework}: {e}", file=sys.stderr)
                continue

            try:
                pred = predict_at(
                    model=framework_model, system_id=sys_id,
                    PP=PP, TP=TP, EP=EP, SP=SP,
                    attention_mode=attn_mode, tp_ep_layout=tp_ep_layout,
                    num_devices=m.num_decode_gpu, S_decode=m.isl + m.osl // 2, B=m.B,
                    flops_eta=1.0, bw_eta=knobs["bw_eta"],
                    c_serving_us=knobs["c_serving"],
                    moe_a2a_pattern=knobs["pattern"],
                    kernel_launch_us=knobs["kernel_launch"],
                    bytes_per_param=bpp,
                )
            except Exception as e:
                if args.verbose:
                    print(f"  ERROR {inf_model} {m.hardware}/{m.framework} TP={TP} EP={EP} dec={m.num_decode_gpu} dp_attn={m.decode_dp_attention} B={m.B}: {e}", file=sys.stderr)
                continue

            label = f"TP={TP} EP={EP} dec={m.num_decode_gpu} dp_attn={m.decode_dp_attention}"
            cell_rows[(inf_model, m.hardware, m.framework)].append(
                (label, m.B, m.tpot_ms, pred)
            )

            if args.plot:
                # Group by (model, hw, fw, partition signature) so each
                # unique partition shape gets one plot with all measured
                # points for that shape overlaid on the framework sweep.
                pkey = (inf_model, m.hardware, m.framework,
                        m.num_decode_gpu, TP, EP, attn_mode, tp_ep_layout)
                measured_by_plot[pkey].append(m)
                if pkey not in plot_meta:
                    plot_meta[pkey] = dict(
                        framework_model=framework_model,
                        bpp=bpp,
                        knobs=knobs,
                        sys_id=sys_id,
                        S_decode=m.isl + m.osl // 2,
                        PP=PP, TP=TP, EP=EP, SP=SP,
                        attn_mode=attn_mode, tp_ep_layout=tp_ep_layout,
                        num_devices=m.num_decode_gpu,
                    )

    # Render plots (one per (model, hw, fw, partition shape) cell)
    if args.plot and measured_by_plot:
        plotted = 0
        for pkey, measured_pts in sorted(measured_by_plot.items()):
            inf_model, hw, fw, dec, TP, EP, attn_mode, tp_ep_layout = pkey
            meta = plot_meta[pkey]
            B_max = max(2 * max(m.B for m in measured_pts), 256)
            try:
                framework = run_framework(
                    model=meta["framework_model"], system_id=meta["sys_id"],
                    PP=meta["PP"], TP=TP, EP=EP, SP=meta["SP"],
                    attention_mode=attn_mode, tp_ep_layout=tp_ep_layout,
                    num_devices=dec, S_decode=meta["S_decode"],
                    B_sweep=log_spaced_B(B_max),
                    flops_eta=1.0, bw_eta=meta["knobs"]["bw_eta"],
                    c_serving_us=meta["knobs"]["c_serving"],
                    moe_a2a_pattern=meta["knobs"]["pattern"],
                    kernel_launch_us=meta["knobs"]["kernel_launch"],
                    bytes_per_param=meta["bpp"],
                )
            except Exception as e:
                if args.verbose:
                    print(f"  PLOT-SKIP {inf_model} {hw}/{fw} TP={TP} EP={EP} dec={dec}: {e}", file=sys.stderr)
                continue

            slug = f"{inf_model.replace('/', '_').replace(' ', '_')}__{hw}_{fw}__TP{TP}_EP{EP}_dec{dec}_{attn_mode}_{tp_ep_layout}"
            out = args.out_dir / f"sweep__{slug}.png"
            topo = topology_tag(meta["sys_id"])
            plot_tpot_vs_B(
                framework=framework, measured=measured_pts,
                title=f"{inf_model} / {hw} / {fw} — TP={TP} EP={EP} dec={dec} ({attn_mode}, {tp_ep_layout})",
                subtitle=(f"PP={meta['PP']} TP={TP} EP={EP} SP={meta['SP']} | ISL={meta['S_decode']*2//3} OSL={meta['S_decode']*2//3} | "
                          f"sys={meta['sys_id']} | {topo} | "
                          f"bw_eta={meta['knobs']['bw_eta']:.2f} c_serving={meta['knobs']['c_serving']:.0f}us "
                          f"kl={meta['knobs']['kernel_launch']:.0f}us pattern={meta['knobs']['pattern']}"),
                out_path=out,
            )
            plotted += 1
        print(f"[--plot] wrote {plotted} plots to {args.out_dir.relative_to(REPO_ROOT)}/sweep__*.png\n")

    # Print summary
    overall_errs = []
    print(f"{'model':<35} {'hw':>6} {'frame':<14} {'n':>4} {'MAE%':>7} {'p50%':>7} {'p90%':>7}")
    print("-" * 100)
    for (inf_model, hw, fw), rows in sorted(cell_rows.items()):
        errs = [abs((p - mm) / mm) * 100 for _, _, mm, p in rows]
        mae = float(np.mean(errs))
        p50 = float(np.percentile(errs, 50))
        p90 = float(np.percentile(errs, 90))
        print(f"{inf_model:<35} {hw:>6} {fw:<14} {len(rows):>4} {mae:>6.1f}% {p50:>6.1f}% {p90:>6.1f}%")
        overall_errs.extend(errs)
        if args.verbose:
            for label, B, mm, p in rows:
                err = (p - mm) / mm * 100
                print(f"    {label:<48} B={B:>6}  meas={mm:>7.2f}ms  pred={p:>7.2f}ms  err={err:>+7.1f}%")
    print("-" * 100)
    if overall_errs:
        overall = float(np.mean(overall_errs))
        print(f"{'OVERALL':<63} {len(overall_errs):>4} {overall:>6.1f}%  "
              f"p50={np.percentile(overall_errs, 50):.1f}% "
              f"p90={np.percentile(overall_errs, 90):.1f}%")
    if skipped_oom or skipped_other:
        print(f"\nSkipped: {skipped_oom} rows model-doesn't-fit-in-HBM (likely FSDP/ZeRO-3 in dataset, "
              f"not representable in framework's TP/EP/PP labeling), "
              f"{skipped_other} other pre-check failures")
        if args.check is not None:
            if overall > args.check:
                print(f"\nFAIL: overall MAE {overall:.1f}% > threshold {args.check}%")
                return 1
            print(f"\nPASS: overall MAE {overall:.1f}% ≤ threshold {args.check}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
