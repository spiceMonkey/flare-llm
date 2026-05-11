"""Post-partition collective-algorithm optimizer.

Standalone pure function that resolves `"auto"` placeholders in
`FrameworkSpec` to concrete algorithm names per (phase × collective).
Runs once after the partition is fixed. Returns a NEW FrameworkSpec
with the auto fields resolved; non-`auto` fields pass through unchanged.

(Phase E: the algorithm fields moved from TuningSpec to FrameworkSpec.
Pre-Phase-E this resolved into TuningSpec; the cost-model logic is
unchanged, only the input/output type.)

Resolution policy:
  - If `framework.inc_enabled` is True AND INC is structurally available
    for this op on this tier chain → pick `"inc"` directly (hardware
    deployment priority; SW costs not compared, since a deployment
    decision doesn't flip on a tuning-grade cost difference).
  - Otherwise (INC unavailable for this op, or `inc_enabled=False`):
    enumerate the SW alternatives and pick `argmin(cost)`.

Resolution scope:
  - tp_algorithm_decode  / tp_algorithm_prefill   (TP all-reduce)
  - ep_algorithm_decode  / ep_algorithm_prefill   (EP MoE all-to-all)
  - SP is always ring AG (only shipped variant per
    `collectives/01_collective_algorithms.md §6`).

If the partition makes a collective trivial (e.g. TP=1 → no AR work),
the field resolves to `"ring"` as a stable sentinel (the dispatcher
returns 0.0 either way).
"""
from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Tuple

from ..specs.framework_spec import FrameworkSpec
from ..specs.model_spec import LlmModelSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.system_spec import SystemSpec, TierSpec
from ..specs.tuner_spec import TuningSpec
from .primitives import G_EP, G_TP, enumerate_options


def optimize_collective_algorithms(
    model: LlmModelSpec,
    partition: PartitionSpec,
    system: SystemSpec,
    tuner: TuningSpec,
    framework: FrameworkSpec,
) -> FrameworkSpec:
    """Resolve `"auto"` algorithm fields on FrameworkSpec by cost-model
    selection. Returns a NEW FrameworkSpec with the auto fields resolved.

    Args:
      model: model architecture (provides H, bytes_per_param, MoE k_active).
      partition: parallelism factors (TP, EP, SP, PP).
      system: system spec (provides tier chains for TP / EP).
      tuner: workload knobs (provides B_decode, B_prefill, S_input — used
             to compute message size M for the cost model).
      framework: SW-stack spec — algorithm fields read from here, INC
             selection respects `framework.inc_enabled`.

    Returns:
      A new FrameworkSpec with all `auto` fields replaced by concrete
      names. Non-`auto` fields pass through unchanged.
    """
    k_active = model.moe.k_active if model.moe is not None else 1
    H = model.H
    b = model.bytes_per_param
    inc_enabled = framework.inc_enabled

    new_fields = {}

    # ─── Decode ─────────────────────────────────────────────────────────
    # Decode TP AR: M = B_decode · H · b, G = TP.
    if framework.tp_algorithm_decode == "auto":
        new_fields["tp_algorithm_decode"] = _resolve(
            tier_chain=system.get_tier_chain("TP"),
            op="all_reduce",
            M=tuner.B_decode * H * b,
            G=G_TP(partition),
            inc_enabled=inc_enabled,
        )
    # Decode EP A2A: M = B_decode · k · H · b, G = EP.
    if framework.ep_algorithm_decode == "auto":
        new_fields["ep_algorithm_decode"] = _resolve(
            tier_chain=system.get_tier_chain("EP"),
            op="moe_a2a",
            M=tuner.B_decode * k_active * H * b,
            G=G_EP(partition),
            inc_enabled=inc_enabled,
        )

    # ─── Prefill ────────────────────────────────────────────────────────
    # Prefill TP AR: M = tokens · H · b, where tokens = B_prefill · S_input.
    # When S_input = 0 (decode-only run), prefill cost paths are inert; pick
    # "ring" as a stable default.
    tokens_prefill = tuner.B_prefill * tuner.S_input
    if framework.tp_algorithm_prefill == "auto":
        new_fields["tp_algorithm_prefill"] = _resolve(
            tier_chain=system.get_tier_chain("TP"),
            op="all_reduce",
            M=tokens_prefill * H * b,
            G=G_TP(partition),
            inc_enabled=inc_enabled,
        )
    # Prefill EP A2A: M = tokens · k · H · b.
    if framework.ep_algorithm_prefill == "auto":
        new_fields["ep_algorithm_prefill"] = _resolve(
            tier_chain=system.get_tier_chain("EP"),
            op="moe_a2a",
            M=tokens_prefill * k_active * H * b,
            G=G_EP(partition),
            inc_enabled=inc_enabled,
        )

    if not new_fields:
        return framework
    return replace(framework, **new_fields)


def _resolve(
    tier_chain: List[TierSpec],
    op: str,
    M: float,
    G: int,
    inc_enabled: bool,
) -> str:
    """Pick the algorithm per the policy in this module's docstring.

    1. If "inc" is among the enumerated options → return "inc" directly
       (hardware-deployment priority; SW costs not compared).
    2. Else if SW options exist → return `min(cost)` among them.
    3. Else (empty option set, e.g. G ≤ 1 or empty chain) → "ring" sentinel.
    """
    options = enumerate_options(tier_chain, op, M, G, inc_enabled=inc_enabled)
    if not options:
        return "ring"
    if any(name == "inc" for name, _ in options):
        return "inc"
    name, _ = min(options, key=lambda no_pair: no_pair[1])
    return name
