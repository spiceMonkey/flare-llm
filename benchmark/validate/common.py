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
    load_partition_from_db,
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
    tpot_ms: float          # `median_tpot` × 1000 — primary y-axis quantity
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
                if not row.get("median_tpot"):
                    continue  # drop incomplete runs
                out.append(MeasuredPoint(
                    B=int(row["conc"]),
                    tpot_ms=float(row["median_tpot"]) * 1000.0,
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
    `num_devices` overrides the cluster size if provided.
    Returns a new SystemSpec; the original is not mutated.
    """
    s = system if num_devices is None else dataclasses.replace(system, num_devices=num_devices)
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
    t_SW_ms: float
    t_LM_ms: float
    t_serving_ms: float


def run_framework(
    *,
    model: str,
    system_id: str,
    PP: int,
    TP: int,
    EP: int,
    SP: int,
    attention_mode: str = "tp",
    layout: str = "orthogonal",
    num_devices: int,
    S_decode: int,
    B_sweep: Sequence[int],
    flops_eta: float = 1.0,
    bw_eta: float = 1.0,
    c_serving_us: float = 0.0,
    bytes_per_param: float | None = None,
) -> list[FrameworkPoint]:
    """Run InferenceCalculator across a B sweep, return per-B latency breakdown.

    `bytes_per_param` overrides the model spec's stored precision when
    provided (e.g. 0.5 for FP4, 1 for FP8). Pass None to use the spec's
    default.
    """
    m = load_model_from_db(model)
    if bytes_per_param is not None:
        m = dataclasses.replace(m, bytes_per_param=bytes_per_param)
    s = system_with_eta(load_system_from_db(system_id), num_devices=num_devices,
                        flops_eta=flops_eta, bw_eta=bw_eta)
    p = PartitionSpec(PP=PP, TP=TP, EP=EP, SP=SP,
                      attention_mode=attention_mode, layout=layout)

    out: list[FrameworkPoint] = []
    for B in B_sweep:
        t = TuningSpec(S_decode=S_decode, B_decode=B, t_serving_per_seq_us=c_serving_us)
        try:
            r = InferenceCalculator(m, s, p, t).run()
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
            t_SW_ms=r.latency.t_SW * 1000,
            t_LM_ms=r.latency.t_LM * 1000,
            t_serving_ms=r.latency.t_serving * 1000,
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
    layout: str = "orthogonal",
    num_devices: int,
    S_decode: int,
    B: int,
    flops_eta: float = 1.0,
    bw_eta: float = 1.0,
    c_serving_us: float = 0.0,
    bytes_per_param: float | None = None,
) -> float:
    """Predict TPOT (ms) at a single B — used to align with measured points."""
    pts = run_framework(
        model=model, system_id=system_id,
        PP=PP, TP=TP, EP=EP, SP=SP,
        attention_mode=attention_mode, layout=layout,
        num_devices=num_devices, S_decode=S_decode,
        B_sweep=[B],
        flops_eta=flops_eta, bw_eta=bw_eta, c_serving_us=c_serving_us,
        bytes_per_param=bytes_per_param,
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
) -> None:
    """Single-panel TPOT-vs-B plot with framework breakdown + measured points.

    Plots compute / mem / comm / LM / serving / total components (matching
    the existing sandbox style). Measured points overlay as markers labeled
    with their concurrency.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(13, 7))

    fx = [p for p in framework if p.fits_in_HBM] or list(framework)
    bs = [p.B for p in fx]
    ax.plot(bs, [p.t_compute_ms for p in fx], "-",  c="crimson",     lw=1.3, alpha=0.85, label="t_compute")
    ax.plot(bs, [p.t_mem_ms     for p in fx], "-",  c="steelblue",   lw=1.3, alpha=0.85, label="t_mem")
    ax.plot(bs, [p.t_local_ms   for p in fx], ":",  c="dimgray",     lw=1.0, alpha=0.7,  label="t_local = max(t_compute, t_mem)")
    ax.plot(bs, [p.t_comm_ms    for p in fx], "--", c="forestgreen", lw=1.3, alpha=0.85, label="t_comm")
    ax.plot(bs, [p.t_LM_ms      for p in fx], "--", c="darkorange",  lw=1.0, alpha=0.7,  label="t_LM (one-shot)")
    ax.plot(bs, [p.t_SW_ms      for p in fx], ":",  c="purple",      lw=1.0, alpha=0.6,  label="t_SW (kernel dispatch)")
    if any(p.t_serving_ms > 0 for p in fx):
        ax.plot(bs, [p.t_serving_ms for p in fx], "--", c="teal",    lw=1.3, alpha=0.85, label="t_serving (per-seq)")
    ax.plot(bs, [p.TPOT_ms      for p in fx], "-",  c="black",       lw=2.5,             label="TPOT (composed)")

    if measured:
        ax.scatter([m.B for m in measured], [m.tpot_ms for m in measured],
                   c="navy", marker="D", s=70, edgecolors="black", linewidths=0.5,
                   zorder=10, label=f"InferenceX measured (n={len(measured)})")
        for m in measured:
            ax.annotate(f"c={m.B}", (m.B, m.tpot_ms),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=6, color="navy", alpha=0.7)

    ax.set_xlabel("Concurrency (B)", fontsize=12)
    ax.set_ylabel("Time per decode step (ms)", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
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


def add_common_cli(ap: argparse.ArgumentParser) -> None:
    """Register the four CLI args every driver should support."""
    ap.add_argument("--flops-eta", type=float, default=1.0,
                    help="Discount factor on device.peak_flops_TF (sustained / nameplate). Default 1.0 (no derate).")
    ap.add_argument("--bw-eta", type=float, default=1.0,
                    help="Discount factor on every memory tier's bandwidth_GBps. Default 1.0 (no derate).")
    ap.add_argument("--c-serving-us", type=float, default=0.0,
                    help="Per-sequence serving runtime overhead c_serving (µs/seq). Default 0 (off).")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                    help="Where to write plots (default: benchmark/results/).")
    ap.add_argument("--check", type=float, default=None, metavar="MAE_PCT",
                    help="Exit non-zero if overall MAE exceeds this percentage. "
                         "Use for CI smoke tests.")


def eta_subtitle(flops_eta: float, bw_eta: float, c_serving_us: float) -> str:
    bits = []
    bits.append("peak FLOPs+BW" if (flops_eta == 1.0 and bw_eta == 1.0)
                else f"flops_eta={flops_eta:.2f}, bw_eta={bw_eta:.2f}")
    bits.append(f"c_serving={c_serving_us:.0f} µs/seq" if c_serving_us > 0
                else "c_serving=0 (off)")
    return " | ".join(bits)


def eta_filename_tag(flops_eta: float, bw_eta: float, c_serving_us: float) -> str:
    parts = []
    if flops_eta != 1.0 or bw_eta != 1.0:
        parts.append(f"flops{flops_eta:.2f}_bw{bw_eta:.2f}")
    if c_serving_us > 0:
        parts.append(f"serv{c_serving_us:.0f}us")
    return ("_" + "_".join(parts).replace(".", "p")) if parts else ""
