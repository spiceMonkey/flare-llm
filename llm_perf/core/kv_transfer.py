"""One transfer function for every move of a KV cache.

A cache of M bytes moves over a PATH: an ordered chain of stages, written in
the same format as a network fabric's tiers (ports in parallel, a rate per
port, a latency). The stage latencies add, and the bytes move at the slowest
stage:

  t_move(M) = sum_i alpha_i + n_blocks * tau_block + M / min_i(rate_i)

  rate_i = ports_i * BW_i            for a hardware stage
         = eta_busy * ports_i * BW_i for a host-side stage (per = "host")

This is the reduction the network model applies to a chain of switch tiers
(alpha_total, BW_min). The same function prices

  * the prefill -> decode handoff (first stage = the link between pools),
  * a fetch or a resume from an off-GPU tier (first stage = the tier's read rate),
  * a store to an off-GPU tier (first stage = the tier's write rate).

`ports` of a stage counts per host, per GPU or per path (`PathStageSpec.per`),
so one chain describes a host with any number of GPUs and a replica that
spans any number of hosts.

A copy that is staged whole at each stage (store-and-forward) pays every
stage's time in turn; `pipelined=False` selects that.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..specs.system_spec import PathStageSpec
from ..utils import GB_TO_BYTES

US_TO_S = 1e-6


@dataclass
class TransferResult:
    t_total: float          # seconds: t_alpha + t_blocks + t_bytes
    t_alpha: float          # seconds: sum of the stage latencies
    t_blocks: float         # seconds: per-block CPU cost
    t_bytes: float          # seconds: bandwidth term
    rate_GBps: float        # delivered rate of the bandwidth term (the slowest stage when pipelined)
    bottleneck: str         # name of the slowest stage ("" for an empty path)
    n_blocks: int           # blocks the per-block cost was charged for


def stage_rate(stage: PathStageSpec, *, n_hosts: int = 1, n_gpus: int = 1, eta_busy: float = 1.0) -> float:
    """Bytes per second one stage carries for a transfer into `n_gpus` GPUs on `n_hosts` hosts."""
    if stage.per == "host":
        units = stage.ports * n_hosts
    elif stage.per == "gpu":
        units = stage.ports * n_gpus
    else:
        units = stage.ports
    rate = stage.bw_per_port_GBps * GB_TO_BYTES * units
    # The busy factor is the extra loss while the CPU is busy serving, so it
    # applies to the CPU-side stages only.
    return rate * eta_busy if stage.per == "host" else rate


def transfer_time(
    M_bytes: float,
    stages: Sequence[PathStageSpec],
    *,
    n_hosts: int = 1,
    n_gpus: int = 1,
    eta_busy: float = 1.0,
    block_bytes: float = 0.0,
    tau_block_us: float = 0.0,
    pipelined: bool = True,
) -> TransferResult:
    """Time to move `M_bytes` over `stages` (see the module docstring)."""
    if M_bytes < 0.0:
        raise ValueError(f"transfer_time needs M_bytes >= 0, got {M_bytes!r}")
    if n_hosts < 1 or n_gpus < 1:
        raise ValueError(f"transfer_time needs n_hosts >= 1 and n_gpus >= 1, got {n_hosts}, {n_gpus}")
    if not 0.0 < eta_busy <= 1.0:
        raise ValueError(f"transfer_time needs eta_busy in (0, 1], got {eta_busy!r}")
    if block_bytes < 0.0 or tau_block_us < 0.0:
        raise ValueError(f"transfer_time needs block_bytes >= 0 and tau_block_us >= 0, got {block_bytes!r}, {tau_block_us!r}")

    t_alpha = sum(s.alpha_us for s in stages) * US_TO_S if stages else 0.0

    n_blocks = int(math.ceil(M_bytes / block_bytes)) if block_bytes > 0.0 and tau_block_us > 0.0 else 0
    t_blocks = n_blocks * tau_block_us * US_TO_S

    rates = [stage_rate(s, n_hosts=n_hosts, n_gpus=n_gpus, eta_busy=eta_busy) for s in stages]
    for s, r in zip(stages, rates):
        if not r > 0.0:
            raise ValueError(f"transfer path stage {s.name!r} has no bandwidth ({r!r} bytes/s)")
    if not stages:
        t_bytes, rate, slowest = 0.0, math.inf, ""
    else:
        i_min = min(range(len(rates)), key=rates.__getitem__)
        slowest = stages[i_min].name
        if pipelined:
            rate = rates[i_min]
            t_bytes = M_bytes / rate
        else:
            t_bytes = sum(M_bytes / r for r in rates)
            rate = M_bytes / t_bytes if t_bytes > 0.0 else rates[i_min]

    return TransferResult(
        t_total=t_alpha + t_blocks + t_bytes,
        t_alpha=t_alpha,
        t_blocks=t_blocks,
        t_bytes=t_bytes,
        rate_GBps=rate / GB_TO_BYTES,
        bottleneck=slowest,
        n_blocks=n_blocks,
    )
