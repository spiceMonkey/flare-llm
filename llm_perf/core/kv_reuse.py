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

`kv_prefix_reuse` generalises this to a cached PREFIX followed by a fresh
suffix, prices the fetch with the shared transfer function over a path, and
adds a hit rate: expected time to first token and expected GPU prefill time
per request. The break-even becomes

    BW_break-even = M_KV(S_prefix) / (t_prefill(S_input) - t_suffix(S_prefix, S_input))

and reduces to the whole-prompt figure when the prefix is the whole prompt.

Scope and limits of `kv_fetch_break_even`:
  - Whole-prompt reuse: the cached entry covers all `S_input` tokens.
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
from typing import Optional, Sequence

from ..calculators.prefill_calculator import PrefillCalculator
from ..specs.framework_spec import FrameworkSpec
from ..specs.model_spec import LlmModelSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.system_spec import PathStageSpec, SystemSpec
from ..specs.tuner_spec import TuningSpec
from ..utils import GB_TO_BYTES, MB_TO_BYTES
from .kv_transfer import transfer_time
from .prefill_model import compute_suffix_prefill
from .primitives import kv_bytes_cluster

FETCH_RULES = ("auto", "always")


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



# ---------------------------------------------------------------------------
# Prefix reuse with a hit rate
# ---------------------------------------------------------------------------

@dataclass
class PrefixFetch:
    """Where a reusable prefix is fetched from, and when the fetch is taken.

    `stages` is the path from the tier into the GPUs: the tier's read rate
    first, then the host stages (`core.kv_transfer`). `rule`:
      "auto"   - a latency rule: on a hit the fetch is taken only when
                 fetch + suffix prefill is faster than prefilling the whole
                 prompt; a skipped fetch costs a full prefill of GPU work;
      "always" - every hit is fetched, whatever it does to latency, which
                 keeps the whole saving in GPU work.
    """

    stages: Sequence[PathStageSpec]
    rule: str = "auto"

    def __post_init__(self) -> None:
        if self.rule not in FETCH_RULES:
            raise ValueError(f"PrefixFetch.rule must be one of {FETCH_RULES}, got {self.rule!r}")
        if not self.stages:
            raise ValueError("PrefixFetch.stages is empty; a fetch needs a path")


@dataclass
class PrefixReuseResults:
    S_input: int
    S_prefix: int
    hit_rate: float
    rule: str
    M_KV_prefix: float              # bytes: cluster-aggregate cache of the prefix
    t_prefill: float                # seconds: prefill of the whole prompt (a miss)
    t_suffix: float                 # seconds: prefill of the suffix on the cached prefix
    t_fetch: float                  # seconds: the prefix over the path
    fetch_rate_GBps: float          # delivered rate of that transfer
    fetch_bottleneck: str           # stage that set it
    t_hit: float                    # t_fetch + t_suffix
    fetch_taken: bool               # is a hit served from the cache (always True under "always")
    t_first_hit: float              # prefill-phase time of a hit as served
    E_t_first: float                # expected prefill-phase time of a request
    E_t_work: float                 # expected GPU prefill time of a request
    work_saved: float               # 1 - E_t_work / t_prefill
    BW_break_even_GBps: float       # M_KV(S_prefix) / (t_prefill - t_suffix)


def kv_prefix_reuse(
    model: LlmModelSpec,
    system: SystemSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
    framework: Optional[FrameworkSpec] = None,
    *,
    S_prefix: int,
    hit_rate: float,
    fetch: PrefixFetch,
    t_prefill: Optional[float] = None,
) -> PrefixReuseResults:
    """Latency and GPU work of a request when a prefix of its prompt may be cached.

    `system` / `partition` are the deployment that prefills (the prefill pool
    when there are two). `tuner.S_input` is the whole prompt. `t_prefill` is
    the whole-prompt prefill time when the caller already has it; otherwise
    it is computed. The fetch is one transfer of the prefix's cache into the
    replica's GPUs; `system.host`, when present, supplies the hosts the
    replica spans and the host-side calibration terms.

    A prefix equal to the whole prompt is priced like `kv_fetch_break_even`:
    nothing is prefilled on a hit.
    """
    if framework is None:
        framework = FrameworkSpec.default()
    S = tuner.S_input
    if not 0.0 <= hit_rate <= 1.0:
        raise ValueError(f"hit_rate must be in [0, 1], got {hit_rate!r}")
    if int(S_prefix) != S_prefix or S_prefix < 1 or S_prefix > S:
        raise ValueError(f"S_prefix must be an integer in [1, S_input={S}], got {S_prefix!r}")
    if t_prefill is None:
        t_prefill = PrefillCalculator(model, system, partition, tuner, framework).run().latency.t_prefill

    t_suffix = (compute_suffix_prefill(model, system, partition, tuner, framework, S_prefix).t_suffix
                if S_prefix < S else 0.0)

    M = kv_bytes_cluster(model, S_prefix)
    n_gpus = partition.PP * partition.TP * partition.EP * partition.SP
    host = system.host
    if host is not None:
        tr = transfer_time(M, fetch.stages, n_hosts=host.hosts_for(n_gpus), n_gpus=n_gpus, eta_busy=host.eta_busy,
                           block_bytes=host.block_MB * MB_TO_BYTES, tau_block_us=host.tau_block_us,
                           pipelined=host.pipelined)
    else:
        tr = transfer_time(M, fetch.stages, n_gpus=n_gpus)

    t_hit = tr.t_total + t_suffix
    taken = True if fetch.rule == "always" else t_hit < t_prefill
    t_first_hit = t_hit if taken else t_prefill
    t_work_hit = t_suffix if taken else t_prefill
    E_t_first = hit_rate * t_first_hit + (1.0 - hit_rate) * t_prefill
    E_t_work = hit_rate * t_work_hit + (1.0 - hit_rate) * t_prefill
    gap = t_prefill - t_suffix
    return PrefixReuseResults(
        S_input=S,
        S_prefix=int(S_prefix),
        hit_rate=hit_rate,
        rule=fetch.rule,
        M_KV_prefix=M,
        t_prefill=t_prefill,
        t_suffix=t_suffix,
        t_fetch=tr.t_total,
        fetch_rate_GBps=tr.rate_GBps,
        fetch_bottleneck=tr.bottleneck,
        t_hit=t_hit,
        fetch_taken=taken,
        t_first_hit=t_first_hit,
        E_t_first=E_t_first,
        E_t_work=E_t_work,
        work_saved=1.0 - E_t_work / t_prefill if t_prefill > 0 else 0.0,
        BW_break_even_GBps=(M / gap) / GB_TO_BYTES if gap > 0 else float("inf"),
    )
