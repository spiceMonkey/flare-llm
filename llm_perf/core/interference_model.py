"""Prefill interference in a shared (co-located) replica, and the comparison
of a shared deployment with a split one.

A shared replica runs prefill and decode on the same devices and is modelled
as time-sliced: while a prefill pass runs, decode steps wait. Steady state,
closed form, no queueing.

  d          = lambda_rep * t_prefill / B_pf        fraction of wall clock spent in prefill
  TPOT_colo  = TPOT(B) / (1 - d)                    mean time per output token
  gap_max    = TPOT(B) + t_block                    worst gap between two tokens
  lambda_max = 1 / (t_prefill / B_pf + S_out * TPOT(B) / B)

`t_block` is what one decode step can be made to wait: a whole prefill pass,
or one chunk when prefill is chunked.

`lambda_max` is the request rate at which the B decode slots are all busy.
It gives the shared replica S_out * lambda_max / g tokens per second per GPU,
which is exactly what a balanced split of the same replica shape gives
(`core.pool_model`): with equal shapes and a free handoff, sharing and
splitting have the same mean throughput per GPU. The split wins on the worst
gap, on a different shape or hardware per phase, and on the batch a latency
target allows. `compare_shared_split` reports the last of these: the best
throughput per GPU each deployment reaches inside a TPOT target and a
worst-gap target.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass
class SharedReplicaResults:
    duty: float                     # d: fraction of wall clock in prefill at the given rate
    TPOT: float                     # mean time per output token, inflated by 1 / (1 - d); inf when d >= 1
    gap_max: float                  # worst gap between two tokens: one step + t_block
    t_block: float                  # longest wait prefill imposes on a decode step
    rate_max: float                 # lambda_max: requests/s at which the decode batch is full
    throughput_per_gpu_max: float   # output tokens/s per GPU at rate_max
    utilisation: float              # rate / rate_max
    feasible: bool                  # rate <= rate_max (the decode batch can hold the requests in progress)


@dataclass
class DeploymentPoint:
    """The best operating point of one deployment inside the latency targets."""
    feasible: bool
    B_decode: int = 0               # decode batch chosen
    throughput_per_gpu: float = 0.0 # output tokens/s per GPU, over every GPU of the deployment
    requests_per_gpu: float = 0.0
    TPOT: float = 0.0               # mean time per output token at that point
    gap_max: float = 0.0            # worst gap between two tokens at that point


@dataclass
class SharedSplitComparison:
    shared: DeploymentPoint
    split: DeploymentPoint
    # split throughput per GPU over shared; inf when only the split is
    # feasible, 0 when only the shared one is, nan when neither is.
    gain: float


def _check_common(t_prefill: float, B_prefill: int, TPOT: float, B_decode: int, S_out: int, g: int) -> None:
    if t_prefill < 0.0:
        raise ValueError(f"t_prefill must be >= 0, got {t_prefill!r}")
    if TPOT <= 0.0:
        raise ValueError(f"TPOT must be > 0, got {TPOT!r}")
    if B_prefill < 1 or B_decode < 1 or g < 1:
        raise ValueError(f"batches and replica size must be >= 1, got B_prefill={B_prefill}, B_decode={B_decode}, g={g}")
    if int(S_out) != S_out or S_out < 1:
        raise ValueError(f"output tokens must be a positive integer, got {S_out!r}")


def shared_replica(
    *,
    rate: float,
    t_prefill: float,
    B_prefill: int,
    TPOT: float,
    B_decode: int,
    S_out: int,
    g: int,
    t_block: Optional[float] = None,
) -> SharedReplicaResults:
    """One shared replica of `g` devices serving `rate` requests per second.

    `t_prefill` is the time of one prefill pass over `B_prefill` prompts on
    this replica (the chunked figure when prefill is chunked), `TPOT` the
    time per token of an uninterrupted decode step at batch `B_decode`.
    `t_block` defaults to the whole prefill pass.
    """
    _check_common(t_prefill, B_prefill, TPOT, B_decode, S_out, g)
    if not rate >= 0.0:
        raise ValueError(f"rate must be >= 0, got {rate!r}")
    if t_block is None:
        t_block = t_prefill
    if t_block < 0.0:
        raise ValueError(f"t_block must be >= 0, got {t_block!r}")

    t_pf_req = t_prefill / B_prefill                 # replica-seconds of prefill per request
    t_dec_req = S_out * TPOT / B_decode              # replica-seconds of decode per request
    duty = rate * t_pf_req
    rate_max = 1.0 / (t_pf_req + t_dec_req)
    return SharedReplicaResults(
        duty=duty,
        TPOT=TPOT / (1.0 - duty) if duty < 1.0 else math.inf,
        gap_max=TPOT + t_block,
        t_block=t_block,
        rate_max=rate_max,
        throughput_per_gpu_max=S_out * rate_max / g,
        utilisation=rate / rate_max,
        feasible=rate <= rate_max * (1.0 + 1e-12),
    )


def compare_shared_split(
    *,
    tpot_shared: Mapping[int, float],
    tpot_decode_pool: Mapping[int, float],
    t_prefill_shared: float,
    t_prefill_pool: float,
    B_prefill: int,
    S_out: int,
    g_shared: int,
    g_prefill: int,
    g_decode: int,
    tpot_target: float = math.inf,
    gap_target: float = math.inf,
    t_block_shared: Optional[float] = None,
) -> SharedSplitComparison:
    """Best throughput per GPU of a shared and of a split deployment inside two latency targets.

    `tpot_shared` and `tpot_decode_pool` map a decode batch to the
    uninterrupted time per token on the shared replica and on the decode
    pool's replica; each deployment picks the batch that serves it best.

    Shared: at batch B the request rate is limited by the full decode batch
    (`rate_max`) and by the TPOT target, which caps the prefill duty at
    1 - TPOT(B) / target. Its worst gap is one step plus `t_block_shared`.

    Split: the decode pool is never interrupted, so its TPOT and its worst
    gap are both TPOT(B); throughput per GPU counts both pools at the
    balanced ratio. The handoff is taken as costing no GPU time.
    """
    if not tpot_target > 0.0 or not gap_target > 0.0:
        raise ValueError(f"targets must be > 0, got tpot_target={tpot_target!r}, gap_target={gap_target!r}")
    if not tpot_shared or not tpot_decode_pool:
        raise ValueError("compare_shared_split needs at least one decode batch for each deployment")
    if t_block_shared is None:
        t_block_shared = t_prefill_shared

    shared = DeploymentPoint(feasible=False)
    for B, tpot in sorted(tpot_shared.items()):
        _check_common(t_prefill_shared, B_prefill, tpot, B, S_out, g_shared)
        gap = tpot + t_block_shared
        if tpot > tpot_target or gap > gap_target:
            continue
        t_pf_req = t_prefill_shared / B_prefill
        rate = 1.0 / (t_pf_req + S_out * tpot / B)                  # full decode batch
        if t_pf_req > 0.0 and math.isfinite(tpot_target):
            rate = min(rate, (1.0 - tpot / tpot_target) / t_pf_req)  # TPOT target caps the duty
        tps = S_out * rate / g_shared
        if tps > shared.throughput_per_gpu:
            duty = rate * t_pf_req
            shared = DeploymentPoint(True, B, tps, rate / g_shared, tpot / (1.0 - duty) if duty < 1.0 else math.inf, gap)

    split = DeploymentPoint(feasible=False)
    for B, tpot in sorted(tpot_decode_pool.items()):
        _check_common(t_prefill_pool, B_prefill, tpot, B, S_out, g_decode)
        if g_prefill < 1:
            raise ValueError(f"g_prefill must be >= 1, got {g_prefill}")
        if tpot > tpot_target or tpot > gap_target:
            continue
        gpu_s = g_prefill * t_prefill_pool / B_prefill + g_decode * S_out * tpot / B
        tps = S_out / gpu_s
        if tps > split.throughput_per_gpu:
            split = DeploymentPoint(True, B, tps, 1.0 / gpu_s, tpot, tpot)

    if shared.feasible and split.feasible:
        gain = split.throughput_per_gpu / shared.throughput_per_gpu if shared.throughput_per_gpu > 0.0 else math.inf
    elif split.feasible:
        gain = math.inf
    elif shared.feasible:
        gain = 0.0
    else:
        gain = math.nan
    return SharedSplitComparison(shared=shared, split=split, gain=gain)
