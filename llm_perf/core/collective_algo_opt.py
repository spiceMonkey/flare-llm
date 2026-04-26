"""Post-partition collective-algorithm optimizer.

Standalone pure function that resolves `auto` placeholders in `TuningSpec` to
concrete algorithm names per (phase × collective). Runs once after the
partition is fixed; iterates over `enumerate_options(...)` for each cell and
picks `min(cost)`.

INC selection is honored via `tuner.inc_enabled` — when True, INC is one of the
enumerated options and naturally wins on cost when available; when False, only
SW options (ring / tree) are enumerated. Per the user's design intent, INC
selection is orthogonal to SW algorithm choice; the optimizer doesn't override
the user's `inc_enabled` flag.

Resolution scope:
  - tp_algorithm_decode  / tp_algorithm_prefill   (TP all-reduce)
  - ep_algorithm_decode  / ep_algorithm_prefill   (EP MoE all-to-all)
  - SP is always ring AG (no knob — only shipped variant per `collectives.md
    §4.5`); the SP-related fields don't exist on TuningSpec.

Non-`auto` fields pass through unchanged. If the partition makes a collective
trivial (e.g. TP=1 → no AR work), the field resolves to `"ring"` as a stable
sentinel (the dispatcher returns 0.0 either way).
"""
from __future__ import annotations

from dataclasses import replace
from typing import List, Tuple

from ..specs.model_spec import LlmModelSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.system_spec import SystemSpec, TierSpec
from ..specs.tuner_spec import TuningSpec
from .primitives import enumerate_options


def optimize_collective_algorithms(
    model: LlmModelSpec,
    partition: PartitionSpec,
    system: SystemSpec,
    tuner: TuningSpec,
) -> TuningSpec:
    """Resolve `auto` algorithm fields by per-cell `min(enumerate_options(...))`.

    Args:
      model: model architecture (provides H, bytes_per_param, MoE k_active).
      partition: parallelism factors (TP, EP, SP, PP).
      system: system spec (provides tier chains for TP / EP).
      tuner: tuning knobs; `auto` fields will be resolved.

    Returns:
      A new TuningSpec with all `auto` fields replaced by concrete names.
      Non-`auto` fields are preserved as-is. INC selection respects
      `tuner.inc_enabled`.
    """
    k_active = model.moe.k_active if model.moe is not None else 1
    H = model.H
    b = model.bytes_per_param
    inc_enabled = tuner.inc_enabled

    new_fields = {}

    # ─── Decode ─────────────────────────────────────────────────────────
    # Decode TP AR: M = B_decode · H · b, G = TP.
    if tuner.tp_algorithm_decode == "auto":
        new_fields["tp_algorithm_decode"] = _resolve(
            tier_chain=system.get_tier_chain("TP"),
            op="all_reduce",
            M=tuner.B_decode * H * b,
            G=partition.TP,
            inc_enabled=inc_enabled,
        )
    # Decode EP A2A: M = B_decode · k · H · b, G = EP.
    if tuner.ep_algorithm_decode == "auto":
        new_fields["ep_algorithm_decode"] = _resolve(
            tier_chain=system.get_tier_chain("EP"),
            op="moe_a2a",
            M=tuner.B_decode * k_active * H * b,
            G=partition.EP,
            inc_enabled=inc_enabled,
        )

    # ─── Prefill ────────────────────────────────────────────────────────
    # Prefill TP AR: M = tokens · H · b, where tokens = B_prefill · S_input.
    # When S_input = 0 (decode-only run), prefill cost paths are inert; pick
    # "ring" as a stable default.
    tokens_prefill = tuner.B_prefill * tuner.S_input
    if tuner.tp_algorithm_prefill == "auto":
        new_fields["tp_algorithm_prefill"] = _resolve(
            tier_chain=system.get_tier_chain("TP"),
            op="all_reduce",
            M=tokens_prefill * H * b,
            G=partition.TP,
            inc_enabled=inc_enabled,
        )
    # Prefill EP A2A: M = tokens · k · H · b.
    if tuner.ep_algorithm_prefill == "auto":
        new_fields["ep_algorithm_prefill"] = _resolve(
            tier_chain=system.get_tier_chain("EP"),
            op="moe_a2a",
            M=tokens_prefill * k_active * H * b,
            G=partition.EP,
            inc_enabled=inc_enabled,
        )

    if not new_fields:
        return tuner
    return replace(tuner, **new_fields)


def _resolve(
    tier_chain: List[TierSpec],
    op: str,
    M: float,
    G: int,
    inc_enabled: bool,
) -> str:
    """Pick the min-cost algorithm name from `enumerate_options`.

    Empty option set (G ≤ 1 or empty chain) → "ring" sentinel (no work).
    """
    options = enumerate_options(tier_chain, op, M, G, inc_enabled=inc_enabled)
    if not options:
        return "ring"
    name, _ = min(options, key=lambda no_pair: no_pair[1])
    return name
