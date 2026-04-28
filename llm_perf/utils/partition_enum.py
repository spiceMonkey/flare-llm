"""Partition enumeration helper for sweep notebooks.

Centralises the practical (PP, TP, EP, SP) constraints used across the
`notebooks/pareto_*` and `notebooks/ttft_*` sweeps so callers do not
re-derive them inline.

Constraints applied (all must hold):
  - pp * tp * ep * sp <= num_devices
  - pp <= pp_max  (default 16; matches production deployment limits where
    deeper PP creates intractable bubble / TTFT trade-offs)
  - tp <= n_kv  (each TP rank must hold ≥ 1 KV head)
  - tp <= n_experts and ep <= n_experts  (MoE only — sharding cannot
    exceed expert count)
  - tp * ep <= scale_up_domain  (avoid crossing the rack-local NVLink /
    NVL boundary for collective traffic)

For dense models (no MoE), `ep_max = 1` and the n_experts caps on TP/EP
are inert — the n_kv cap on TP still applies.
"""

from typing import List, Optional

from ..specs.model_spec import LlmModelSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.system_spec import SystemSpec


_DEFAULT_PP_LADDER = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60]
_DEFAULT_SP_LADDER = [1, 2, 4, 8, 16, 32, 64]


def _power_of_2_ladder(max_val: int) -> List[int]:
    """Powers-of-2 ladder up to and including max_val (≥ 1)."""
    if max_val < 1:
        return [1]
    out = [1]
    n = 2
    while n <= max_val:
        out.append(n)
        n *= 2
    return out


def scale_up_domain_size(
    system: SystemSpec,
    role: str = "TP",
    *,
    scale_up_tier_index: int = 0,
) -> int:
    """Cumulative number of devices reachable through tier `scale_up_tier_index` (inclusive).

    `scale_up_tier_index = 0` returns the ports of tier 0 alone — the
    innermost domain (e.g., 72 for NVL72; 16 for the d-Matrix pair-of-cards
    mesh). `scale_up_tier_index = i` returns the cumulative reach
    `prod(tier[k].ports for k in 0..i)` — useful when a deployment treats
    a multi-tier fabric as a single scale-up unit (e.g., d-Matrix tier-1
    PCIe extends the domain to a full server = 16 × 4 = 64 chiplets).

    Out-of-range indices clamp to the last available tier rather than
    raising — so a single-tier system like NVL72 with
    `scale_up_tier_index=1` still returns its tier-0 reach (72) instead of
    erroring.
    """
    chain = system.get_tier_chain(role)
    if not chain:
        raise ValueError(f"No tier chain configured for collective role {role!r}")
    idx = max(0, min(scale_up_tier_index, len(chain) - 1))
    reach = 1
    for t in chain[: idx + 1]:
        reach *= t.ports
    return reach


def enumerate_partitions(
    model: LlmModelSpec,
    system: SystemSpec,
    *,
    num_devices: Optional[int] = None,
    pp_max: int = 16,
    pp_choices: Optional[List[int]] = None,
    sp_choices: Optional[List[int]] = None,
    tp_max_override: Optional[int] = None,
    ep_max_override: Optional[int] = None,
    scale_up_domain_override: Optional[int] = None,
    scale_up_tier_index: int = 0,
) -> List[PartitionSpec]:
    """Enumerate valid (PP, TP, EP, SP) partitions for `model` on `system`.

    See module docstring for the full list of constraints.

    Parameters
    ----------
    num_devices : int, optional
        Total devices available. Defaults to `system.num_devices`. Override
        when sweeping cluster size at fixed system spec.
    pp_max : int
        Maximum PP. Default 16 (production limit; deeper PP rarely deploys
        because bubble / TTFT costs grow faster than t_stage shrinks).
    pp_choices, sp_choices : list of int, optional
        Override the enumeration ladders. Defaults are the standard ladders
        used across the notebook suite.
    tp_max_override, ep_max_override : int, optional
        Override the auto-derived TP / EP caps. Useful for sensitivity
        sweeps that intentionally cross the natural cap.
    scale_up_domain_override : int, optional
        Override the auto-derived scale-up domain size with an absolute
        value. Takes precedence over `scale_up_tier_index`. Useful when
        the sweep replaces the system fabric to test a hypothetical
        topology.
    scale_up_tier_index : int, default 0
        Index into the fabric chain (`system.get_tier_chain(role)`) up to
        which the scale-up domain extends. 0 = innermost tier only
        (default — every collective stays within tier 0 of the fabric);
        1 = cumulative reach through tier 1 (e.g., d-Matrix server-wide
        via PCIe = 16 × 4 = 64); etc. Indices beyond the chain length
        clamp to the last tier. Inert when `scale_up_domain_override` is
        also set.
    """
    if num_devices is None:
        num_devices = system.num_devices

    n_kv = model.n_kv
    n_experts = model.moe.n_experts if model.moe is not None else None

    if tp_max_override is not None:
        tp_max = tp_max_override
    else:
        tp_max = min(n_kv, n_experts) if n_experts is not None else n_kv

    if ep_max_override is not None:
        ep_max = ep_max_override
    else:
        ep_max = n_experts if n_experts is not None else 1

    if scale_up_domain_override is not None:
        scale_up = scale_up_domain_override
    else:
        scale_up = scale_up_domain_size(system, scale_up_tier_index=scale_up_tier_index)

    if pp_choices is None:
        pp_choices = [pp for pp in _DEFAULT_PP_LADDER if pp <= pp_max]
    if sp_choices is None:
        sp_choices = list(_DEFAULT_SP_LADDER)

    tp_choices = _power_of_2_ladder(tp_max)
    ep_choices = _power_of_2_ladder(ep_max)

    out: List[PartitionSpec] = []
    for pp in pp_choices:
        for tp in tp_choices:
            for ep in ep_choices:
                if tp * ep > scale_up:
                    continue
                for sp in sp_choices:
                    if pp * tp * ep * sp > num_devices:
                        continue
                    out.append(PartitionSpec(PP=pp, TP=tp, EP=ep, SP=sp))
    return out


def describe_constraints(
    model: LlmModelSpec,
    system: SystemSpec,
    pp_max: int = 16,
    *,
    scale_up_tier_index: int = 0,
) -> str:
    """One-line summary of the active constraints, for notebook printouts."""
    n_kv = model.n_kv
    n_experts = model.moe.n_experts if model.moe is not None else None
    scale_up = scale_up_domain_size(system, scale_up_tier_index=scale_up_tier_index)
    if n_experts is not None:
        tp_max = min(n_kv, n_experts)
        return (
            f"PP ≤ {pp_max}; TP ≤ min(n_kv={n_kv}, n_experts={n_experts}) = {tp_max}; "
            f"EP ≤ n_experts={n_experts}; TP·EP ≤ scale_up={scale_up}"
            f" (tier_idx={scale_up_tier_index})"
        )
    return (
        f"PP ≤ {pp_max}; TP ≤ n_kv={n_kv}; EP = 1 (dense); "
        f"TP·EP ≤ scale_up={scale_up} (tier_idx={scale_up_tier_index})"
    )
