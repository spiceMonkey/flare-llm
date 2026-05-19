"""Shared helpers for InferenceX cross-validation drivers.

Each driver script in `benchmark/validate/` consumes the vendored
InferenceX dataset under `benchmark/inferenceX/data/flat/<Model>.csv` and
runs the llm_perf framework against the same deployment shape, then prints
a per-(TP, B) error table and writes a TPOT-vs-B plot. This module
factors out the loaders, derate wrappers, error formatting, and plot style
so each driver stays small and focused on its specific configuration.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# Make `from llm_perf import ...` work when running drivers as standalone
# scripts (without `pip install -e .`). Drivers import this module first;
# this side effect threads the repo root onto sys.path before any llm_perf
# import resolves.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm_perf import InferenceCalculator  # noqa: E402
from llm_perf.io import (  # noqa: E402
    load_model_from_db,
    load_system_from_db,
    load_tuner_from_db,
)
from llm_perf.specs.partition_spec import PartitionSpec  # noqa: E402
from llm_perf.specs.system_spec import SystemSpec  # noqa: E402
from llm_perf.specs.tuner_spec import TuningSpec  # noqa: E402

# ────────────────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────────────────

INFERENCEX_FLAT = _REPO_ROOT / "benchmark" / "inferenceX" / "data" / "flat"
DEFAULT_RESULTS_DIR = _REPO_ROOT / "benchmark" / "results"


# ────────────────────────────────────────────────────────────────────────────
# Measured-data loader
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class MeasuredPoint:
    """One InferenceX measurement row (subset of fields used by drivers)."""

    B: int                  # `conc` — steady-state in-flight requests (≈ per-step batch)
    tpot_ms: float          # `mean_tpot` × 1000 — primary y-axis quantity
    isl: int
    osl: int
    decode_tp: int
    decode_ep: int
    num_decode_gpu: int
    decode_dp_attention: bool
    spec_method: str
    framework: str
    hardware: str


def _to_bool(v: str) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes"}


def load_measured(
    model: str,
    *,
    isl: int | None = None,
    osl: int | None = None,
    decode_tp: int | None = None,
    decode_ep: int | None = None,
    num_decode_gpu: int | None = None,
    dp_attention: bool | None = None,
    spec_method: str = "none",
    framework: str | Iterable[str] | None = None,
    hardware: str | Iterable[str] | None = None,
    precision: str | None = None,
) -> list[MeasuredPoint]:
    """Load a filtered slice of `benchmark/inferenceX/data/flat/<model>.csv`.

    All filter args are optional — None means "match anything". String-set
    filters (`framework`, `hardware`) accept either a single value or any
    iterable of allowed values. Returns rows sorted by B (ascending).

    Drivers should pass enough filters to land on a single (model,
    hardware, framework, deployment-shape) cut; the resulting MeasuredPoint
    list is the y-axis for the comparison plot.
    """
    path = INFERENCEX_FLAT / f"{model}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"InferenceX data for {model!r} not found at {path}. "
            f"Run `python benchmark/inferenceX/fetch.py` to refresh."
        )

    fw_set = {framework} if isinstance(framework, str) else (set(framework) if framework else None)
    hw_set = {hardware} if isinstance(hardware, str) else (set(hardware) if hardware else None)

    out: list[MeasuredPoint] = []
    with path.open() as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in reader:
            try:
                if isl is not None and int(row["isl"]) != isl:
                    continue
                if osl is not None and int(row["osl"]) != osl:
                    continue
                if decode_tp is not None and int(row["decode_tp"]) != decode_tp:
                    continue
                if decode_ep is not None and int(row["decode_ep"]) != decode_ep:
                    continue
                if num_decode_gpu is not None and int(row["num_decode_gpu"]) != num_decode_gpu:
                    continue
                if dp_attention is not None and _to_bool(row["decode_dp_attention"]) != dp_attention:
                    continue
                if (row.get("spec_method") or "none").strip() != spec_method:
                    continue
                if fw_set and row["framework"] not in fw_set:
                    continue
                if hw_set and row["hardware"] not in hw_set:
                    continue
                if precision is not None and row.get("precision") != precision:
                    continue
                if not row.get("mean_tpot"):
                    continue  # drop incomplete runs
                out.append(MeasuredPoint(
                    B=int(row["conc"]),
                    tpot_ms=float(row["mean_tpot"]) * 1000.0,
                    isl=int(row["isl"]),
                    osl=int(row["osl"]),
                    decode_tp=int(row["decode_tp"]),
                    decode_ep=int(row["decode_ep"]),
                    num_decode_gpu=int(row["num_decode_gpu"]),
                    decode_dp_attention=_to_bool(row["decode_dp_attention"]),
                    spec_method=(row.get("spec_method") or "none").strip(),
                    framework=row["framework"],
                    hardware=row["hardware"],
                ))
            except (KeyError, ValueError):
                continue

    out.sort(key=lambda m: m.B)
    return out


# ────────────────────────────────────────────────────────────────────────────
# System derate wrapper
# ────────────────────────────────────────────────────────────────────────────


def _resize_outer_fabric_tier(system: SystemSpec, num_devices: int) -> SystemSpec:
    """For multi-tier systems, set the outer fabric tier's `ports` to
    `ceil(num_devices / inner_tier.ports)` so the IB tier scales with the
    declared cluster size. Inert for single-tier systems.

    Used to parameterize multibox templates (b200.multibox, b300.multibox,
    h100.multibox, h200.multibox) by `num_devices` rather than carrying one
    pre-shaped JSON per supported GPU count.
    """
    tp_chain = system.collective_fabrics.get("TP")
    if not isinstance(tp_chain, list) or len(tp_chain) < 2:
        return system
    inner_name = tp_chain[0]
    outer_name = tp_chain[-1]
    if inner_name == outer_name:
        return system
    inner_tier = system.fabrics[inner_name].tiers[0]
    n_boxes = (num_devices + inner_tier.ports - 1) // inner_tier.ports
    outer_fab = system.fabrics[outer_name]
    new_outer_tiers = [dataclasses.replace(outer_fab.tiers[0], ports=n_boxes), *outer_fab.tiers[1:]]
    new_outer = dataclasses.replace(outer_fab, tiers=new_outer_tiers)
    new_fabrics = {**system.fabrics, outer_name: new_outer}
    return dataclasses.replace(system, fabrics=new_fabrics)


def system_with_eta(
    system: SystemSpec,
    *,
    num_devices: int | None = None,
    flops_eta: float = 1.0,
    bw_eta: float = 1.0,
) -> SystemSpec:
    """Apply nameplate-to-sustained discount factors and override device count.

    `flops_eta ∈ (0, 1]` scales `device.peak_flops_TF` (effective compute peak).
    `bw_eta ∈ (0, 1]` scales every memory tier's `bandwidth_GBps` (and the
    legacy `hbm_bandwidth_GBps` field on single-tier devices).
    `num_devices` overrides the cluster size if provided. For multi-tier
    systems it also auto-resizes the outer fabric tier's `ports` to match
    (so `b200.multibox` template loads correctly at any GPU count).
    Returns a new SystemSpec; the original is not mutated.
    """
    if num_devices is None:
        s = system
    else:
        s = dataclasses.replace(system, num_devices=num_devices)
        s = _resize_outer_fabric_tier(s, num_devices)
    if flops_eta == 1.0 and bw_eta == 1.0:
        return s

    d = s.device
    new_d = dataclasses.replace(d, peak_flops_TF=d.peak_flops_TF * flops_eta)
    if bw_eta != 1.0:
        if d.tiers:
            new_tiers = [
                dataclasses.replace(t, bandwidth_GBps=t.bandwidth_GBps * bw_eta)
                for t in d.tiers
            ]
            new_d = dataclasses.replace(new_d, tiers=new_tiers)
        else:
            new_d = dataclasses.replace(new_d, hbm_bandwidth_GBps=d.hbm_bandwidth_GBps * bw_eta)
    return dataclasses.replace(s, device=new_d)


def topology_tag(sys_id: str) -> str:
    """Short subtitle tag describing the fabric topology of a system spec.

    Returns 'single-tier (<fabric>)' for one-fabric specs (e.g. 8gpu /
    72gpu specs) or 'multi-tier: N <inner>-islands via <outer>' for
    hierarchical specs (e.g. b200.multibox / gb200.multibox).
    Empty string if the spec can't be loaded.

    Used by `coverage_sweep.py` and per-stack drivers to make the
    plot subtitle visually distinguish single-box from multi-box
    deployments at a glance.
    """
    try:
        sys_spec = load_system_from_db(sys_id)
    except Exception:
        return ""
    tp_chain = sys_spec.collective_fabrics.get("TP")
    if isinstance(tp_chain, list) and len(tp_chain) > 1:
        inner_name, outer_name = tp_chain[0], tp_chain[-1]
        inner_ports = sys_spec.fabrics[inner_name].tiers[0].ports
        boxes = (sys_spec.num_devices + inner_ports - 1) // inner_ports
        return f"multi-tier: {boxes} {inner_name}-islands via {outer_name}"
    fabric_name = tp_chain if isinstance(tp_chain, str) else (tp_chain[0] if tp_chain else "?")
    return f"single-tier ({fabric_name})"


# ────────────────────────────────────────────────────────────────────────────
# Framework runner
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class FrameworkPoint:
    B: int
    TPOT_ms: float
    fits_in_HBM: bool
    t_compute_ms: float
    t_mem_ms: float
    t_local_ms: float
    t_comm_ms: float
    t_stage_ms: float
    t_kernel_ms: float
    t_LM_ms: float
    t_step_seq_ms: float


def run_framework(
    *,
    model: str,
    system_id: str,
    PP: int,
    TP: int,
    EP: int,
    SP: int,
    attention_mode: str = "tp",
    tp_ep_layout: str = "orthogonal",
    num_devices: int,
    S_decode: int,
    B_sweep: Sequence[int],
    flops_eta: float = 1.0,
    bw_eta: float = 1.0,
    c_seq_us: float = 0.0,
    moe_a2a_pattern: str = "gather",
    kernel_launch_us: float | None = None,
    bytes_per_param: float | None = None,
    seq_overlap_factor: float | None = None,
    kernel_overlap_factor: float | None = None,
    comm_overlap_factor: float | None = None,
) -> list[FrameworkPoint]:
    """Run InferenceCalculator across a B sweep, return per-B latency breakdown.

    `bytes_per_param` overrides the model spec's stored precision when
    provided (e.g. 0.5 for FP4, 1 for FP8). Pass None to use the spec's
    default. Phase F: the bw_efficiency η_β(B) and tensor_core_efficiency
    η_TC(mb) curves now live on DeviceSpec — set them in the system JSON
    rather than passing through the wrapper.
    """
    from llm_perf.specs import FrameworkSpec  # local import to avoid cycle in some configs

    m = load_model_from_db(model)
    if bytes_per_param is not None:
        m = dataclasses.replace(m, bytes_per_param=bytes_per_param)
    s = system_with_eta(load_system_from_db(system_id), num_devices=num_devices,
                        flops_eta=flops_eta, bw_eta=bw_eta)
    p = PartitionSpec(PP=PP, TP=TP, EP=EP, SP=SP)

    # Build a FrameworkSpec from the per-driver knobs. Phase H: attention_mode
    # and layout flow here from the driver since they're stack-axis decisions
    # (not sharding-factor decisions). Other framework fields fall to
    # FrameworkSpec defaults — kernel_overlap_factor=1, mla_mode='absorbed',
    # inc_enabled=True, kernels_per_*=defaults.
    fw_kwargs = dict(
        name="benchmark-driver",
        attention_mode=attention_mode,
        tp_ep_layout=tp_ep_layout,
        c_seq_us=c_seq_us,
        moe_a2a_pattern=moe_a2a_pattern,
        # InferenceX measurements were not captured with SHARP-class INC
        # engaged; opt out explicitly so the cost model picks SW even when
        # the fabric advertises inc != "none" (e.g. NVL72 NVSwitch5 is NVLS-
        # capable hardware but production Dynamo+TRT/SGLang runs did not
        # enable it during the measurement window). Notebooks that *do* study
        # INC effects (pareto_collective_algorithms, pareto_vs_kernel_launch,
        # pareto_vs_mem_alpha*) construct their own FrameworkSpec with
        # `inc_enabled=True` and explicit `tp_algorithm_decode="auto"`.
        inc_enabled=False,
    )
    if kernel_launch_us is not None:
        fw_kwargs["kernel_launch_us"] = kernel_launch_us
    if seq_overlap_factor is not None:
        fw_kwargs["seq_overlap_factor"] = seq_overlap_factor
    if kernel_overlap_factor is not None:
        fw_kwargs["kernel_overlap_factor"] = kernel_overlap_factor
    if comm_overlap_factor is not None:
        fw_kwargs["comm_overlap_factor"] = comm_overlap_factor
    framework = FrameworkSpec(**fw_kwargs)

    out: list[FrameworkPoint] = []
    for B in B_sweep:
        t = TuningSpec(S_decode=S_decode, B_decode=B)
        try:
            r = InferenceCalculator(m, s, p, t, framework).run()
        except Exception as e:  # don't kill the sweep on one bad B
            print(f"  ERROR at B={B}: {e}", file=sys.stderr)
            continue
        out.append(FrameworkPoint(
            B=B,
            TPOT_ms=r.latency.TPOT * 1000,
            fits_in_HBM=r.memory.fits_in_HBM,
            t_compute_ms=r.latency.t_compute * 1000,
            t_mem_ms=r.latency.t_mem * 1000,
            t_local_ms=r.latency.t_local * 1000,
            t_comm_ms=r.latency.t_comm * 1000,
            t_stage_ms=r.latency.t_stage * 1000,
            t_kernel_ms=r.latency.t_kernel * 1000,
            t_LM_ms=r.latency.t_LM * 1000,
            t_step_seq_ms=r.latency.t_step_seq * 1000,
        ))
    return out


def log_spaced_B(B_max: int, *, B_min: int = 1) -> list[int]:
    """Powers-of-two ∪ log-spaced points up to B_max — same shape as decode notebooks."""
    if B_max <= B_min:
        return [B_min]
    import math
    n = max(int(math.log10(B_max) * 18), 30)
    raw = {max(B_min, int(round(10 ** (i * math.log10(B_max) / n)))) for i in range(n + 1)}
    p = 1
    while p <= B_max:
        raw.add(p)
        p *= 2
    raw.add(B_min)
    raw.add(B_max)
    return sorted(raw)


def predict_at(
    *,
    model: str,
    system_id: str,
    PP: int, TP: int, EP: int, SP: int,
    attention_mode: str = "tp",
    tp_ep_layout: str = "orthogonal",
    num_devices: int,
    S_decode: int,
    B: int,
    flops_eta: float = 1.0,
    bw_eta: float = 1.0,
    c_seq_us: float = 0.0,
    moe_a2a_pattern: str = "gather",
    kernel_launch_us: float | None = None,
    bytes_per_param: float | None = None,
    seq_overlap_factor: float | None = None,
    kernel_overlap_factor: float | None = None,
    comm_overlap_factor: float | None = None,
) -> float:
    """Predict TPOT (ms) at a single B — used to align with measured points."""
    pts = run_framework(
        model=model, system_id=system_id,
        PP=PP, TP=TP, EP=EP, SP=SP,
        attention_mode=attention_mode, tp_ep_layout=tp_ep_layout,
        num_devices=num_devices, S_decode=S_decode,
        B_sweep=[B],
        flops_eta=flops_eta, bw_eta=bw_eta, c_seq_us=c_seq_us,
        moe_a2a_pattern=moe_a2a_pattern,
        kernel_launch_us=kernel_launch_us,
        bytes_per_param=bytes_per_param,
        seq_overlap_factor=seq_overlap_factor,
        kernel_overlap_factor=kernel_overlap_factor,
        comm_overlap_factor=comm_overlap_factor,
    )
    if not pts:
        return float("nan")
    return pts[0].TPOT_ms


# ────────────────────────────────────────────────────────────────────────────
# Error table
# ────────────────────────────────────────────────────────────────────────────


def error_table(
    rows: Sequence[tuple],  # iterable of (label, B, measured_ms, predicted_ms)
    *,
    title: str | None = None,
) -> str:
    """Pretty-print a (label, B, measured, predicted, err%) table + summary.

    `label` is a free-form string per row (e.g. "TP=8" or "config A"); rows
    sharing a label are subtotaled. The final line is the overall MAE.
    """
    if not rows:
        return "(no rows)"

    lines = []
    if title:
        lines.append(title)
        lines.append("-" * len(title))
    lines.append(f"{'label':<20} {'B':>6} {'meas_ms':>10} {'pred_ms':>10} {'err%':>9}")
    lines.append("-" * 60)

    by_label: dict[str, list[float]] = {}
    overall: list[float] = []
    last_label = None
    for label, B, meas, pred in rows:
        if last_label is not None and label != last_label:
            errs = by_label.get(last_label, [])
            if errs:
                lines.append(f"  {last_label} MAE={np.mean(np.abs(errs)):.1f}% max={np.max(np.abs(errs)):.1f}% (n={len(errs)})")
                lines.append("")
        last_label = label
        err = (pred - meas) / meas * 100 if meas else 0.0
        by_label.setdefault(label, []).append(err)
        overall.append(err)
        lines.append(f"{label:<20} {B:>6} {meas:>10.3f} {pred:>10.3f} {err:>+8.1f}%")

    if last_label is not None:
        errs = by_label.get(last_label, [])
        if errs:
            lines.append(f"  {last_label} MAE={np.mean(np.abs(errs)):.1f}% max={np.max(np.abs(errs)):.1f}% (n={len(errs)})")

    lines.append("-" * 60)
    lines.append(
        f"OVERALL MAE={np.mean(np.abs(overall)):.1f}% "
        f"max={np.max(np.abs(overall)):.1f}% n={len(overall)}"
    )
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# Plot helper
# ────────────────────────────────────────────────────────────────────────────


def plot_tpot_vs_B(
    *,
    framework: Sequence[FrameworkPoint],
    measured: Sequence[MeasuredPoint],
    title: str,
    subtitle: str,
    out_path: Path,
    primary_label: str | None = None,
    secondary: (
        tuple[str, Sequence[FrameworkPoint], Sequence[MeasuredPoint]]
        | list[tuple[str, Sequence[FrameworkPoint], Sequence[MeasuredPoint]]]
        | None
    ) = None,
    xlim: tuple[float, float] | None = None,
) -> None:
    """Single-panel TPOT-vs-B plot with framework breakdown + measured points.

    Plots compute / mem / comm / LM / serving / total components (matching
    the existing sandbox style). Measured points overlay as markers labeled
    with their concurrency.

    `secondary`: optional overlay(s) of additional workloads' TPOT model curve
    and measured scatter (no breakdown). Accepts either a single
    `(label, framework, measured)` tuple or a list of them. Use when a single
    calibration is validated against multiple workload regimes on the same
    panel (e.g. short- vs long-context decode).
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # When LLM_PERF_PAPER_STYLE=1 is set (by the paper figure pipeline),
    # bump fonts/lines so panels read well when tiled into the §5 composite.
    _paper = os.environ.get("LLM_PERF_PAPER_STYLE") == "1"
    f_scale = 2.4 if _paper else 1.0
    l_scale = 2.0 if _paper else 1.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(13, 7))

    fx = [p for p in framework if p.fits_in_HBM] or list(framework)
    bs = [p.B for p in fx]
    ax.plot(bs, [p.t_compute_ms for p in fx], "-",  c="crimson",     lw=1.3*l_scale, alpha=0.85, label="t_compute")
    ax.plot(bs, [p.t_mem_ms     for p in fx], "-",  c="steelblue",   lw=1.3*l_scale, alpha=0.85, label="t_mem")
    ax.plot(bs, [p.t_local_ms   for p in fx], "-.", c="darkviolet",  lw=1.6*l_scale, alpha=0.95, label="t_local = max(t_compute, t_mem)")
    ax.plot(bs, [p.t_comm_ms    for p in fx], "--", c="forestgreen", lw=1.3*l_scale, alpha=0.85, label="t_comm")
    ax.plot(bs, [p.t_LM_ms      for p in fx], "--", c="darkorange",  lw=1.0*l_scale, alpha=0.7,  label="t_LM (one-shot)")
    ax.plot(bs, [p.t_kernel_ms      for p in fx], "-.", c="goldenrod",   lw=1.6*l_scale, alpha=0.95, label="t_kernel (launch dispatch)")
    if any(p.t_step_seq_ms > 0 for p in fx):
        ax.plot(bs, [p.t_step_seq_ms for p in fx], "--", c="mediumvioletred", lw=1.3*l_scale, alpha=0.85, label="t_step_seq (per-seq)")
    ax.plot(bs, [p.TPOT_ms      for p in fx], "-",  c="black",       lw=2.5*l_scale,             label="TPOT (composed)")

    if measured:
        prim_label = (f"InferenceX measured {primary_label} (n={len(measured)})"
                      if primary_label else f"InferenceX measured (n={len(measured)})")
        ax.scatter([m.B for m in measured], [m.tpot_ms for m in measured],
                   c="navy", marker="D", s=70*(l_scale**2), edgecolors="black", linewidths=0.5*l_scale,
                   zorder=10, label=prim_label)
        for m in measured:
            ax.annotate(f"c={m.B}", (m.B, m.tpot_ms),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=6*f_scale, color="navy", alpha=0.7)

    if secondary is not None:
        # Normalize to list. A bare 3-tuple is a single overlay (legacy form).
        sec_list = (
            [secondary]
            if (isinstance(secondary, tuple) and len(secondary) == 3
                and isinstance(secondary[0], str))
            else list(secondary)
        )
        sec_styles = [
            ("teal",          "^",  ( 6, -10)),
            ("darkmagenta",   "s",  ( 6,  10)),
            ("saddlebrown",   "v",  (-12, -10)),
        ]
        for i, (sec_label, sec_fw, sec_meas) in enumerate(sec_list):
            color, marker, dxy = sec_styles[i % len(sec_styles)]
            sec_fx = [p for p in sec_fw if p.fits_in_HBM] or list(sec_fw)
            sec_bs = [p.B for p in sec_fx]
            ax.plot(sec_bs, [p.TPOT_ms for p in sec_fx], "-",
                    c=color, lw=2.5*l_scale,
                    label=f"TPOT ({sec_label})")
            if sec_meas:
                ax.scatter([m.B for m in sec_meas], [m.tpot_ms for m in sec_meas],
                           c=color, marker=marker, s=70*(l_scale**2), edgecolors="black",
                           linewidths=0.5*l_scale, zorder=10,
                           label=f"InferenceX measured {sec_label} (n={len(sec_meas)})")
                for m in sec_meas:
                    ax.annotate(f"c={m.B}", (m.B, m.tpot_ms),
                                textcoords="offset points", xytext=dxy,
                                fontsize=6*f_scale, color=color, alpha=0.7)

    ax.set_xlabel("Concurrency (B)", fontsize=12*f_scale)
    ax.set_ylabel("Time per decode step (ms)", fontsize=12*f_scale)
    ax.tick_params(axis="both", labelsize=10*f_scale)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.3, which="both")

    if _paper:
        # Strip suptitle / axis title / legend — the composite owns those.
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
    else:
        ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=True, framealpha=0.95)
        ax.set_title("TPOT and component breakdown vs Concurrency", fontsize=11)
        fig.suptitle(f"{title}\n{subtitle}", fontsize=11)
        fig.tight_layout(rect=[0, 0, 0.82, 0.94])
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Standard CLI args
# ────────────────────────────────────────────────────────────────────────────


def add_common_cli(
    ap: argparse.ArgumentParser,
    *,
    default_flops_eta: float = 1.0,
    default_bw_eta: float = 1.0,
    default_c_seq_us: float = 0.0,
) -> None:
    """Register the standard CLI args every driver should support.

    Drivers may override the *defaults* (not the surface) to bake in
    per-stack calibration. The CLI arg names stay uniform so users can sweep
    by overriding on the command line (e.g. `--bw-eta 1.0` to disable a
    baked-in derate). The defaults reflect the per-(model, hardware,
    framework) calibration that lands within reasonable MAE; running with
    `--flops-eta 1.0 --bw-eta 1.0 --c-seq-us 0` reverts to the peak
    roofline.

    The c_seq knob is primarily framework-bound (serving stack: Python
    interpreter weight, CUDA-Graph replay, fused vs Python sampling); see
    `validate/README.md` for the framework × HW knob structure.
    """
    ap.add_argument("--flops-eta", type=float, default=default_flops_eta,
                    help=f"Discount factor on device.peak_flops_TF (sustained / nameplate). "
                         f"Driver default: {default_flops_eta}.")
    ap.add_argument("--bw-eta", type=float, default=default_bw_eta,
                    help=f"Discount factor on every memory tier's bandwidth_GBps. "
                         f"Driver default: {default_bw_eta}.")
    ap.add_argument("--c-seq-us", type=float, default=default_c_seq_us,
                    help=f"Per-sequence serving runtime overhead c_seq (µs/seq). "
                         f"Driver default: {default_c_seq_us}.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                    help="Where to write plots (default: benchmark/results/).")
    ap.add_argument("--check", type=float, default=None, metavar="MAE_PCT",
                    help="Exit non-zero if overall MAE exceeds this percentage. "
                         "Use for CI smoke tests.")


def eta_subtitle(flops_eta: float, bw_eta: float, c_seq_us: float) -> str:
    bits = []
    bits.append("peak FLOPs+BW" if (flops_eta == 1.0 and bw_eta == 1.0)
                else f"flops_eta={flops_eta:.2f}, bw_eta={bw_eta:.2f}")
    bits.append(f"c_seq={c_seq_us:.0f} µs/seq" if c_seq_us > 0
                else "c_seq=0 (off)")
    return " | ".join(bits)


def eta_filename_tag(flops_eta: float, bw_eta: float, c_seq_us: float) -> str:
    parts = []
    if flops_eta != 1.0 or bw_eta != 1.0:
        parts.append(f"flops{flops_eta:.2f}_bw{bw_eta:.2f}")
    if c_seq_us > 0:
        parts.append(f"serv{c_seq_us:.0f}us")
    return ("_" + "_".join(parts).replace(".", "p")) if parts else ""
