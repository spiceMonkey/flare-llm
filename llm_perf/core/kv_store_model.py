"""Warm memory: the off-GPU tier list of a host, and what it does for sessions.

Hot memory is GPU memory and is read on every decode step; it is modelled by
the decode and prefill calculators and is not touched here. Warm memory is
host memory (DRAM, CXL) that holds caches which are NOT being decoded: a
reusable prefix, or the cache of an idle session. It is read only when a
cache moves, so it never enters the decode step; it enters the end-to-end
layer through

  * the fetch of a prefix on a hit      (`prefix_fetch_from` -> core.kv_reuse),
  * the capacity for idle sessions      (`warm_sessions`),
  * the store and resume of a session   (`tier_store`, `tier_fetch`).

Every move is priced by `core.kv_transfer.transfer_time` over the tier's
path: the tier's own rate first (read rate for a fetch, write rate for a
store), then the stages of the host path the tier names.

Session capacity. A replica holds N_active sequences in GPU memory
(`KVPagingResults.max_sequences`). A tier holds a whole number of idle caches:

  N_parked   = floor(n_hosts * C_tier / M_KV(S))
  N_sessions = min(N_active / (1 - f_idle), N_active + N_parked)

With a fraction f_idle of sessions idle at any moment, N_active busy ones
correspond to N_active / (1 - f_idle) sessions in all, as long as there is
room for the idle ones. An idle cache may stay in GPU memory while a slot is
free there; only the overflow is parked, so the room for all caches together
is N_active + N_parked. (Requiring every idle cache to leave the GPUs would
give the stricter N_parked / f_idle.)

The tier list is ordered and each tier names its own path, so a flash tier
or a remote store is one more entry; nothing here is specific to DRAM.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from ..specs.kv_offload_spec import KVOffloadSpec
from ..specs.system_spec import HostSpec, KVTierSpec, PathStageSpec
from ..utils import GB_TO_BYTES, MB_TO_BYTES
from .kv_transfer import TransferResult, transfer_time


@dataclass
class WarmSessionResults:
    tier: str                       # the tier that holds idle sessions
    M_KV_session: float             # bytes: cluster-aggregate cache of one session at the context length
    capacity_bytes: float           # bytes: the tier's capacity over the hosts the replica spans
    N_active: int                   # sequences the replica holds in GPU memory
    N_parked_capacity: int          # idle caches the tier can hold
    N_sessions: float               # sessions the replica serves, active and idle together
    N_parked: float                 # idle sessions actually parked: N_sessions - N_active
    sessions_gain: float            # N_sessions / N_active (1.0 when N_active = 0)
    limited_by: str                 # "idle_fraction" | "tier_capacity" | "none"
    t_store: float                  # seconds: one session's cache from the GPUs to the tier
    t_resume_fetch: float           # seconds: back from the tier into the GPUs
    store_rate_GBps: float
    fetch_rate_GBps: float
    fetch_bottleneck: str


def tier_stages(host: HostSpec, tier: KVTierSpec, *, write: bool = False) -> List[PathStageSpec]:
    """The path between a tier and the GPUs: the tier itself, then its host path."""
    rate = tier.write_GBps if write and tier.write_GBps is not None else tier.read_GBps
    first = PathStageSpec(name=tier.name, bw_per_port_GBps=rate, ports=tier.ports, per="path", alpha_us=tier.alpha_us)
    return [first] + (list(host.get_path(tier.path)) if tier.path else [])


def _move(M_bytes: float, host: HostSpec, tier: KVTierSpec, n_gpus: int, write: bool) -> TransferResult:
    return transfer_time(
        M_bytes, tier_stages(host, tier, write=write),
        n_hosts=host.hosts_for(n_gpus), n_gpus=n_gpus, eta_busy=host.eta_busy,
        block_bytes=host.block_MB * MB_TO_BYTES, tau_block_us=host.tau_block_us, pipelined=host.pipelined,
    )


def tier_fetch(M_bytes: float, host: HostSpec, tier_name: str, n_gpus: int) -> TransferResult:
    """Time to bring `M_bytes` of cache from a tier into `n_gpus` GPUs."""
    return _move(M_bytes, host, host.get_tier(tier_name), n_gpus, write=False)


def tier_store(M_bytes: float, host: HostSpec, tier_name: str, n_gpus: int) -> TransferResult:
    """Time to move `M_bytes` of cache from `n_gpus` GPUs to a tier."""
    return _move(M_bytes, host, host.get_tier(tier_name), n_gpus, write=True)


def prefix_fetch_from(host: Optional[HostSpec], offload: Optional[KVOffloadSpec]):
    """The `core.kv_reuse.PrefixFetch` the strategy implies, or None when it names no prefix tier."""
    if host is None or offload is None or not offload.prefix_tier:
        return None
    from .kv_reuse import PrefixFetch   # kv_reuse imports the prefill calculator; keep this module light
    return PrefixFetch(tier_stages(host, host.get_tier(offload.prefix_tier)), offload.fetch_rule)


def warm_sessions(
    *,
    host: HostSpec,
    tier_name: str,
    M_KV_session: float,
    N_active: int,
    idle_fraction: float,
    n_gpus: int,
) -> WarmSessionResults:
    """Sessions one replica serves when idle caches are parked in a warm tier.

    `M_KV_session` is the whole cache of one session (all of its shards);
    `N_active` the sequences the replica holds in GPU memory; `n_gpus` the
    replica's devices, which fixes how many hosts (and so how much of the
    tier) it has.
    """
    if not 0.0 <= idle_fraction < 1.0:
        raise ValueError(f"idle_fraction must be in [0, 1), got {idle_fraction!r}")
    if M_KV_session <= 0.0:
        raise ValueError(f"M_KV_session must be > 0, got {M_KV_session!r}")
    if N_active < 0:
        raise ValueError(f"N_active must be >= 0, got {N_active!r}")
    tier = host.get_tier(tier_name)
    capacity = host.hosts_for(n_gpus) * tier.capacity_GB * GB_TO_BYTES
    n_cap = int(math.floor(capacity / M_KV_session))

    wanted = N_active / (1.0 - idle_fraction)           # sessions the idle fraction calls for
    held = float(N_active + n_cap)                      # sessions the tier allows
    n_sessions = min(wanted, held)
    if idle_fraction == 0.0 or N_active == 0:
        limited = "none"
    else:
        limited = "idle_fraction" if wanted <= held else "tier_capacity"

    store = tier_store(M_KV_session, host, tier_name, n_gpus)
    fetch = tier_fetch(M_KV_session, host, tier_name, n_gpus)
    return WarmSessionResults(
        tier=tier.name,
        M_KV_session=M_KV_session,
        capacity_bytes=capacity,
        N_active=N_active,
        N_parked_capacity=n_cap,
        N_sessions=n_sessions,
        N_parked=n_sessions - N_active,
        sessions_gain=n_sessions / N_active if N_active > 0 else 1.0,
        limited_by=limited,
        t_store=store.t_total,
        t_resume_fetch=fetch.t_total,
        store_rate_GBps=store.rate_GBps,
        fetch_rate_GBps=fetch.rate_GBps,
        fetch_bottleneck=fetch.bottleneck,
    )
