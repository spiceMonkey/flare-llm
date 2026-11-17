"""Post-partition collective-algorithm optimizer.

Standalone pure function that resolves `"auto"` placeholders in
`FrameworkSpec` to concrete algorithm names per (phase × collective).
Runs once after the partition is fixed. Returns a NEW FrameworkSpec
with the auto fields resolved; non-`auto` fields pass through unchanged.

(Phase E: the algorithm fields moved from TuningSpec to FrameworkSpec.
Pre-Phase-E this resolved into TuningSpec; the cost-model logic is
unchanged, only the input/output type.)

Resolution policy: enumerate every schedule the fabric admits for the op
and pick `argmin(cost)`. In-network collectives (`"inc"`) are one of the
candidates when `framework.inc_enabled` is True and a crossed tier offers
them; they win when they are cheaper, which is nearly always inside one
switch domain, and lose when a software schedule is faster (a runtime
picks its algorithm per call from predicted time and is not forced onto
the switch path). An exact tie goes to `"inc"`.

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
from .primitives import G_EP, G_TP, axis_stride, decode_microbatch, enumerate_options, group_batch
from .primitives.sharding_factors import scatter_direct


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
    b = model.act_bytes()   # collective payloads are activations
    inc_enabled = framework.inc_enabled
    align_policy = framework.torus_align_policy.lower()

    new_fields = {}

    # ─── Decode ─────────────────────────────────────────────────────────
    # Decode collectives carry one pipeline microbatch (== B_decode at PP = 1).
    mb_decode = max(1.0, group_batch(decode_microbatch(tuner.B_decode, partition.PP), partition, framework))
    # Decode TP AR: M = mb · H · b, G = TP.
    if framework.tp_algorithm_decode == "auto":
        new_fields["tp_algorithm_decode"] = _resolve(
            tier_chain=system.get_tier_chain("TP"),
            op="all_reduce",
            M=mb_decode * H * b,
            G=G_TP(partition),
            inc_enabled=inc_enabled,
            align_policy=align_policy,
            stride=axis_stride(partition, framework, "TP"),
        )
    # Decode EP A2A: M = mb · k · H · b, G = EP.
    if framework.ep_algorithm_decode == "auto":
        new_fields["ep_algorithm_decode"] = _resolve(
            tier_chain=system.get_tier_chain("EP"),
            op="moe_a2a",
            # Scatter-direct dispatch under DP-attention sends each rank's own
            # share of the batch, as compute_comm prices it.
            M=(max(1.0, mb_decode / G_TP(partition)) if scatter_direct(partition, framework) else mb_decode) * k_active * H * b,
            G=G_EP(partition),
            inc_enabled=inc_enabled,
            align_policy=align_policy,
            stride=axis_stride(partition, framework, "EP"),
        )

    # ─── Prefill ────────────────────────────────────────────────────────
    # Prefill TP AR: M = tokens · H · b, where tokens = B_prefill · S_input.
    # When S_input = 0 (decode-only run), prefill cost paths are inert; pick
    # "ring" as a stable default.
    # Batched prefill divides its requests across the EP device groups under
    # the orthogonal layout, like decode (see `D_batch`).
    tokens_prefill = group_batch(tuner.B_prefill, partition, framework) * tuner.S_input
    if framework.tp_algorithm_prefill == "auto":
        new_fields["tp_algorithm_prefill"] = _resolve(
            tier_chain=system.get_tier_chain("TP"),
            op="all_reduce",
            M=tokens_prefill * H * b,
            G=G_TP(partition),
            inc_enabled=inc_enabled,
            align_policy=align_policy,
            stride=axis_stride(partition, framework, "TP"),
        )
    # Prefill EP A2A: M = tokens · k · H · b.
    if framework.ep_algorithm_prefill == "auto":
        new_fields["ep_algorithm_prefill"] = _resolve(
            tier_chain=system.get_tier_chain("EP"),
            op="moe_a2a",
            # Scatter-direct: a rank's own share, at least one whole request.
            M=(max(tokens_prefill / G_TP(partition), tuner.S_input) if scatter_direct(partition, framework)
               else tokens_prefill) * k_active * H * b,
            G=G_EP(partition),
            inc_enabled=inc_enabled,
            align_policy=align_policy,
            stride=axis_stride(partition, framework, "EP"),
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
    align_policy: str = "prefix",
    stride: int = 1,
) -> str:
    """Pick the algorithm per the policy in this module's docstring.

    1. Options exist → return `min(cost)` among them, "inc" included; an
       exact tie goes to "inc".
    2. Empty option set (e.g. G ≤ 1 or empty chain) → "ring" sentinel.
    """
    options = enumerate_options(
        tier_chain, op, M, G, inc_enabled=inc_enabled, align_policy=align_policy,
        stride=stride,
    )
    if not options:
        return "ring"
    name, _ = min(options, key=lambda no_pair: (no_pair[1], no_pair[0] != "inc"))
    return name
