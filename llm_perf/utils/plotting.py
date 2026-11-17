"""Lightweight plotting helpers for llm_perf experiments.

matplotlib is imported inside the plotting functions, not at module level, so
that ``import llm_perf`` and the calculators work without it installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def _pyplot():
    """Import matplotlib.pyplot on first use."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "llm_perf plotting helpers need matplotlib (pip install matplotlib); "
            "the rest of llm_perf does not."
        ) from exc
    return plt


def save_config_tps_scatter(
    config_labels: Sequence[str],
    tps_values: Sequence[float],
    output_path: str | Path,
    *,
    title: str | None = None,
    ylabel: str = "TTPS (tokens/s)",
) -> Path:
    """Render a scatter plot of configurations vs. throughput."""

    if len(config_labels) != len(tps_values):
        raise ValueError("config_labels and tps_values must have the same length")
    if not config_labels:
        raise ValueError("At least one data point is required to build a plot")

    plt = _pyplot()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xs = range(len(config_labels))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(xs, tps_values, c="tab:blue", edgecolors="black")
    ax.set_xlabel("Partition configuration index")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_xticks(list(xs), config_labels, rotation=45, ha="right")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
