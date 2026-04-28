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


def scale_up_domain_size(system: SystemSpec, role: str = "TP") -> int:
    """Number of devices reachable in the innermost scale-up tier.

    Returns the `ports` field of tier 0 of the fabric chain mapped to
    `role`. For NVL72 this is 72; for d-Matrix pair_mesh tier-0 it is 4
    (server-local).
    """
    chain = system.get_tier_chain(role)
    if not chain:
        raise ValueError(f"No tier chain configured for collective role {role!r}")
    return chain[0].ports


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
        Override the auto-derived scale-up domain size. Useful when the
        sweep replaces the system fabric to test a hypothetical topology.
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
        scale_up = scale_up_domain_size(system)

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


def describe_constraints(model: LlmModelSpec, system: SystemSpec, pp_max: int = 16) -> str:
    """One-line summary of the active constraints, for notebook printouts."""
    n_kv = model.n_kv
    n_experts = model.moe.n_experts if model.moe is not None else None
    scale_up = scale_up_domain_size(system)
    if n_experts is not None:
        tp_max = min(n_kv, n_experts)
        return (
            f"PP ≤ {pp_max}; TP ≤ min(n_kv={n_kv}, n_experts={n_experts}) = {tp_max}; "
            f"EP ≤ n_experts={n_experts}; TP·EP ≤ scale_up={scale_up}"
        )
    return (
        f"PP ≤ {pp_max}; TP ≤ n_kv={n_kv}; EP = 1 (dense); "
        f"TP·EP ≤ scale_up={scale_up}"
    )
