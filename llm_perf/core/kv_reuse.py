"""Fetch-versus-recompute break-even for a cached prompt (KV-cache reuse).

A prompt whose key-value (KV) cache is already stored somewhere — host
memory, a CXL pool, NVMe, another node — can be served either by
recomputing it (a prefill pass) or by fetching the stored cache. Fetching
wins on latency when moving the bytes is faster than running prefill over
the same tokens:

    M_KV / BW_fetch  <  t_prefill      <=>      BW_fetch  >  M_KV / t_prefill

The right-hand side is the break-even bandwidth: how fast a cache tier must
deliver into the serving devices before it is worth having, for a given
model, hardware, partition and prompt length. Both of its terms are outputs
of the framework, so no new spec is needed.

Scope and limits:
  - Whole-prompt reuse: the cached entry covers all `S_input` tokens. A
    shared prefix followed by a fresh suffix is not modeled here.
  - Latency only. Fetching also frees the prefill devices for other
    requests; that throughput gain needs two-pool accounting and is not
    credited, so on fast hardware this comparison understates the value of
    reuse.
  - `fetch_bandwidth_GBps` is the delivered rate into the devices over the
    whole path (media, link, host-to-device hop), per stream. Several
    devices of one host loading at once share the host's uplink; pass the
    per-stream rate they actually get.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..calculators.prefill_calculator import PrefillCalculator
from ..specs.framework_spec import FrameworkSpec
from ..specs.model_spec import LlmModelSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.system_spec import SystemSpec
from ..specs.tuner_spec import TuningSpec
from ..utils import GB_TO_BYTES
from .primitives import kv_bytes_cluster


@dataclass
class KVReuseBreakEven:
    S_input: int                          # prompt tokens covered by the cached entry
    M_KV: float                           # bytes: cluster-aggregate KV cache of the prompt
    t_recompute: float                    # seconds: prefill time over the prompt
    BW_break_even_GBps: float             # M_KV / t_recompute; inf when prefill takes no time
    # Filled only when a fetch bandwidth is given:
    fetch_bandwidth_GBps: Optional[float] = None   # delivered rate per stream
    fetch_streams: int = 1
    t_fetch: Optional[float] = None       # seconds: alpha + M_KV / (streams * BW_fetch)
    saving: Optional[float] = None        # seconds: t_recompute - t_fetch (positive = fetching wins)
    fetch_wins: Optional[bool] = None


def kv_fetch_break_even(
    model: LlmModelSpec,
    system: SystemSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
    framework: Optional[FrameworkSpec] = None,
    *,
    fetch_bandwidth_GBps: Optional[float] = None,
    fetch_streams: int = 1,
    fetch_alpha_us: float = 0.0,
) -> KVReuseBreakEven:
    """Break-even between fetching a cached prompt and recomputing it.

    Args:
      model, system, partition, tuner, framework: the usual five-spec
        composition of the prefill deployment. `tuner.S_input` is the
        prompt length; `tuner.B_prefill` and `tuner.chunk_size` are used as
        in `PrefillCalculator` (the unchunked prefill time is compared).
      fetch_bandwidth_GBps: delivered fetch rate per stream. When given,
        the result also carries the fetch time and the latency saving.
      fetch_streams: device pairs (or drives, or links) fetching shards of
        the cache in parallel.
      fetch_alpha_us: fixed cost per fetch (first-byte latency, lookup).

    Returns:
      KVReuseBreakEven. `BW_break_even_GBps` is the total delivered
      bandwidth (all streams together) a tier needs for fetching to match
      recomputation.
    """
    if fetch_streams < 1:
        raise ValueError(f"fetch_streams must be >= 1, got {fetch_streams}")
    S = tuner.S_input
    M_KV = kv_bytes_cluster(model, S)
    t_recompute = PrefillCalculator(model, system, partition, tuner, framework).run().latency.t_prefill
    bw_break_even = (M_KV / t_recompute) / GB_TO_BYTES if t_recompute > 0 else float("inf")
    out = KVReuseBreakEven(
        S_input=S, M_KV=M_KV, t_recompute=t_recompute, BW_break_even_GBps=bw_break_even,
    )
    if fetch_bandwidth_GBps is not None:
        if fetch_bandwidth_GBps <= 0:
            raise ValueError(f"fetch_bandwidth_GBps must be > 0, got {fetch_bandwidth_GBps}")
        t_fetch = fetch_alpha_us * 1e-6 + M_KV / (fetch_streams * fetch_bandwidth_GBps * GB_TO_BYTES)
        out.fetch_bandwidth_GBps = fetch_bandwidth_GBps
        out.fetch_streams = fetch_streams
        out.t_fetch = t_fetch
        out.saving = t_recompute - t_fetch
        out.fetch_wins = t_fetch < t_recompute
    return out
