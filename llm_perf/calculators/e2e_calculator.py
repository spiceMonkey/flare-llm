
from dataclasses import dataclass
from typing import Any, Optional

from ..specs.model_spec import LlmModelSpec
from ..specs.system_spec import PathStageSpec, SystemSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.tuner_spec import TuningSpec
from ..specs.overhead_spec import OverheadSpec
from ..specs.disagg_spec import DisaggSpec
from ..specs.framework_spec import FrameworkSpec
from ..specs.workload_spec import WorkloadSpec
from ..specs.kv_offload_spec import KVOffloadSpec
from ..utils import GB_TO_BYTES, MB_TO_BYTES
from ..core.primitives import kv_bytes_cluster
from ..core.pool_model import PoolResults, compute_pools, kv_layout, replica_devices
from ..core.kv_transfer import TransferResult, transfer_time
from ..core.kv_paging_model import KVPagingConfig, compute_kv_paging
from ..core.kv_store_model import WarmSessionResults, prefix_fetch_from, warm_sessions
from .inference_calculator import InferenceResults
from .prefill_calculator import PrefillResults


US_TO_S = 1e-6


@dataclass
class E2EResults:
    # TTFT (seconds) — unified across co-lo and disagg branches
    TTFT: float                 # t_sched + t_prefill (+ t_handoff; + one decode step when a disaggregated decode worker emits token 1)
    TTFT_chunked: float         # TTFT with chunked prefill

    # TPOT (seconds)
    TPOT: float                 # user-observed inter-token time: t_step_user(B), or the
                                # speculative-decoding value t_step_user_verify / N_tok_per_step

    # Throughput
    throughput_per_gpu: float   # tokens/s per GPU
    interactivity: float        # 1 / TPOT (tokens/s perceived by user)
    TTPS: float                 # total cluster throughput

    # Gap between the first and the second output token (seconds). One decode
    # step, plus the KV handoff when a disaggregated prefill worker emits
    # token 1 and the transfer therefore delays token 2 instead of TTFT.
    t_second_token: float

    # KV handoff diagnostics
    t_handoff: float            # seconds (co-lo or disagg, depending on DisaggSpec.disaggregated)
    t_repack: float             # seconds (scale-up repack component; 0 if no repack)
    M_KV_total: float           # bytes (cluster-aggregate KV cache)

    # Framework overhead breakdown
    t_sched: float              # seconds
    t_framework_per_step: float # per-step overhead (seconds)

    # Parallel streams the disaggregated transfer was priced with
    # (`DisaggSpec.transfer_streams`, or the derived count under
    # `auto_streams`). 0 when no disaggregated transfer was priced.
    handoff_streams: int = 0
    # Delivered rate of the disaggregated transfer's bandwidth term and the
    # stage that set it ("link" = the link between pools; otherwise a stage
    # of the decode host's handoff path, when the transfer is staged).
    handoff_rate_GBps: float = 0.0
    handoff_bottleneck: str = ""

    # Caches in flight to one decode replica, in sequences: request rate per
    # decode replica x (t_handoff + rho_KV x t_prefill), the time from the
    # transfer's first byte to the start of decode. Each holds GPU memory;
    # pass this to `compute_kv_paging(..., inflight_sequences=...)`. 0 unless
    # the deployment is disaggregated and pools were sized.
    inflight_sequences: float = 0.0
    M_KV_inflight: float = 0.0      # bytes, replica-aggregate: inflight_sequences x M_KV_total

    # Prefix reuse (core/kv_reuse.py: PrefixReuseResults). None unless the
    # workload carries a hit rate and a prefix length and a fetch path was
    # given. TTFT above stays the time of a request that prefills its whole
    # prompt; TTFT_hit is a request served from the cache, TTFT_expected the
    # mean over the hit rate (equal to TTFT without reuse). With reuse,
    # `pools` is sized on the expected GPU prefill time per request.
    reuse: Optional[Any] = None
    TTFT_hit: float = 0.0
    TTFT_expected: float = 0.0

    # Warm memory for idle sessions (core/kv_store_model.py). None unless a
    # KVOffloadSpec names an idle tier on the decode system's host and a
    # workload was given. `N_active` there is the paging model's
    # max_sequences for this replica, caches in flight included.
    warm: Optional[WarmSessionResults] = None

    # Two-pool sizing (core/pool_model.py). None unless a WorkloadSpec and
    # prefill results were given. `throughput_per_gpu` above counts the
    # decode replica's GPUs only; `pools.throughput_per_gpu` counts both pools.
    pools: Optional[PoolResults] = None


class E2ECalculator:
    """Assemble end-to-end metrics from decode + prefill + overhead + disagg (prefill.md §6, e2e.md)."""

    def __init__(
        self,
        decode_results: InferenceResults,
        prefill_results: Optional[PrefillResults],
        overhead: OverheadSpec,
        disagg: DisaggSpec,
        model: LlmModelSpec,
        system: SystemSpec,
        partition: PartitionSpec,
        tuner: TuningSpec,
        *,
        workload: Optional[WorkloadSpec] = None,
        prefill_system: Optional[SystemSpec] = None,
        prefill_partition: Optional[PartitionSpec] = None,
        framework: Optional[FrameworkSpec] = None,
        prefill_framework: Optional[FrameworkSpec] = None,
        prefix_fetch: Optional[Any] = None,
        offload: Optional[KVOffloadSpec] = None,
        paging: Optional[KVPagingConfig] = None,
    ) -> None:
        """`system` / `partition` describe the decode replica. The keyword
        arguments are all optional and leave every result unchanged when
        omitted:

        workload           tokens generated per request and, optionally, the
                           request rate: turns on two-pool sizing.
        prefill_system,    the prefill pool, when it is not the decode
        prefill_partition  replica's own shape. `prefill_results` must then
                           come from a PrefillCalculator run on them.
        framework,         the stacks of the two pools; only consulted to
        prefill_framework  compare KV layouts under `DisaggSpec.auto_repack`,
                           and to price the suffix prefill under reuse.
        prefix_fetch       a `core.kv_reuse.PrefixFetch`: the path a cached
                           prefix is fetched over and the fetch rule. Turns
                           on prefix reuse when the workload has a hit rate
                           and a prefix length.
        offload            the warm-memory strategy. Its prefix tier (on the
                           prefill system's host) supplies the fetch path
                           when `prefix_fetch` is not given; its idle tier
                           (on the decode system's host) parks idle sessions.
        paging             KV paging parameters used to count the sequences
                           the decode replica holds in GPU memory; defaults.
        """
        self.decode = decode_results
        self.prefill = prefill_results
        self.overhead = overhead
        self.disagg = disagg
        self.model = model
        self.system = system
        self.partition = partition
        self.tuner = tuner
        self.workload = workload
        self.prefill_system = prefill_system if prefill_system is not None else system
        self.prefill_partition = prefill_partition if prefill_partition is not None else partition
        self.framework = framework if framework is not None else FrameworkSpec.default()
        self.prefill_framework = prefill_framework if prefill_framework is not None else self.framework
        self.offload = offload
        self.paging = paging
        if prefix_fetch is None:
            prefix_fetch = prefix_fetch_from(self.prefill_system.host, offload)
        self.prefix_fetch = prefix_fetch

    def run(self) -> E2EResults:
        oh = self.overhead
        dg = self.disagg
        dec = self.decode.latency

        # Framework overhead.
        # `oh.t_graph_us` is the legacy flat per-step CUDA-graph constant;
        # it is superseded when the decode latency model exposes a derived
        # per-round SW budget (`LatencyResults.t_kernel > 0`, which is the
        # production default since the kernel-launch refactor). Including
        # both would double-count, so we only fold `t_graph_us` in when the
        # derived term is zero (legacy path / SW disabled).
        t_sched = (oh.t_sched_us + oh.t_tok_us) * US_TO_S
        t_graph_legacy_us = oh.t_graph_us if dec.t_kernel <= 0.0 else 0.0
        t_framework_per_step = (t_graph_legacy_us + oh.t_detok_us) * US_TO_S

        # Prefill latency (0 if no prefill results)
        if self.prefill is not None:
            t_prefill = self.prefill.latency.t_prefill
            t_prefill_chunked = self.prefill.latency.t_prefill_chunked
        else:
            t_prefill = 0.0
            t_prefill_chunked = 0.0

        # KV cache volume (cluster-aggregate, prefill.md §6.2). Latent-attention
        # models store one shared latent per token per layer, not per-head K/V.
        M_KV_total = kv_bytes_cluster(self.model, self.tuner.S_input)

        # KV handoff: branch on DisaggSpec.disaggregated (prefill.md §6.3 / §6.4)
        self._transfer: Optional[TransferResult] = None
        t_handoff, t_repack = self._compute_handoff(M_KV_total, t_prefill)
        handoff_streams = self._streams() if dg.disaggregated and dg.inter_bandwidth_GBps > 0 else 0

        # TTFT assembly (prefill.md §6.5; e2e.md §2). t_prefill ends with the
        # LM head on the last prompt position, so the prefill pass itself
        # yields token 1. What follows depends on who streams it:
        #   - co-located: the same engine emits it; the repack into the
        #     decode layout (t_handoff, usually zero) is the only extra.
        #   - disaggregated, first_token="decode": the decode worker waits
        #     for the KV cache, re-runs the last prompt position (one decode
        #     step) and emits the token.
        #   - disaggregated, first_token="prefill": the prefill worker emits
        #     it at once; the transfer delays the second token instead and
        #     is reported in t_handoff only.
        if not dg.disaggregated:
            t_after_prefill = t_handoff
        elif dg.first_token == "prefill":
            t_after_prefill = 0.0
        elif dg.first_token == "decode":
            t_after_prefill = t_handoff + dec.t_step_user
        else:
            raise ValueError(
                f"DisaggSpec.first_token must be 'decode' or 'prefill', got {dg.first_token!r}"
            )
        TTFT = t_sched + t_prefill + t_after_prefill
        TTFT_chunked = t_sched + t_prefill_chunked + t_after_prefill

        # Token 1 -> token 2. Whatever part of the handoff was not on the
        # TTFT path lands here: all of it when the prefill worker emits
        # token 1, none otherwise.
        t_second_token = dec.t_step_user + (t_handoff - t_after_prefill if dg.disaggregated and dg.first_token == "prefill" else 0.0)

        # TPOT (e2e.md §3). With speculative decoding enabled the decode model
        # reports the per-accepted-token time separately; use it, and scale
        # throughput by the same ratio (N_tok_per_step tokens per verify step).
        # The draft model's own cost is not modelled, so this is an upper
        # bound on the speculative gain.
        speculative = dec.N_tok_per_step > 1.0 and dec.TPOT_spec > 0.0
        TPOT = dec.TPOT_spec if speculative else dec.TPOT

        # Throughput (e2e.md §5)
        TTPS = dec.TTPS * (dec.TPOT / dec.TPOT_spec) if speculative else dec.TTPS
        throughput_per_gpu = TTPS / self.system.num_devices if self.system.num_devices > 0 else 0.0
        interactivity = 1.0 / TPOT if TPOT > 0 else 0.0

        # Prefix reuse: a fraction of requests fetch a cached prefix and
        # prefill only the suffix. TTFT itself keeps meaning a request that
        # prefills its whole prompt.
        reuse = None
        wl = self.workload
        if (wl is not None and self.prefill is not None and self.prefix_fetch is not None
                and wl.prefix_hit_rate > 0.0 and wl.prefix_tokens > 0):
            if self.tuner.B_prefill != 1:
                raise ValueError(
                    "prefix reuse is priced for unbatched prefill (TuningSpec.B_prefill = 1), "
                    f"got B_prefill={self.tuner.B_prefill}"
                )
            # Imported here: core.kv_reuse imports the prefill calculator.
            from ..core.kv_reuse import kv_prefix_reuse
            reuse = kv_prefix_reuse(
                self.model, self.prefill_system, self.prefill_partition, self.tuner, self.prefill_framework,
                S_prefix=wl.prefix_tokens, hit_rate=wl.prefix_hit_rate, fetch=self.prefix_fetch,
                t_prefill=t_prefill,
            )
        TTFT_hit = t_sched + reuse.t_first_hit + t_after_prefill if reuse is not None else 0.0
        TTFT_expected = t_sched + reuse.E_t_first + t_after_prefill if reuse is not None else TTFT

        # Two-pool sizing: GPU time per request in each pool, the balanced
        # pool ratio and throughput per GPU over both pools. Under reuse the
        # prefill pool's work per request is the expected one.
        pools = None
        if self.workload is not None and self.prefill is not None:
            pools = compute_pools(
                t_prefill=reuse.E_t_work if reuse is not None else t_prefill,
                B_prefill=self.tuner.B_prefill,
                TPOT=TPOT,
                B_decode=self.tuner.B_decode,
                g_prefill=replica_devices(self.prefill_partition),
                g_decode=replica_devices(self.partition),
                workload=self.workload,
            )

        # Caches in flight to one decode replica (Little's law: arrival rate
        # x time a cache holds decode memory before its sequence is decoded).
        # That time runs from the first byte's arrival: the part of the
        # transfer that overlaps prefill (rho_KV x t_prefill) plus the
        # handoff after prefill ends. Memory is taken as reserved when the
        # transfer starts, which is the upper bound; time spent waiting for a
        # decode slot is not included. With a request rate the replica sees
        # its share of it; without one it is taken at the utilisation cap.
        inflight = 0.0
        if pools is not None and dg.disaggregated:
            if self.workload.request_rate is None:
                rate_rep = self.workload.utilisation_cap * pools.mu_decode
            elif pools.n_decode_replicas:
                rate_rep = self.workload.request_rate / pools.n_decode_replicas
            else:
                rate_rep = 0.0
            inflight = rate_rep * (t_handoff + dg.overlap_rho_KV * t_prefill)

        # Warm memory: idle sessions parked in a tier of the decode host.
        warm = None
        host = self.system.host
        if (self.offload is not None and self.offload.idle_tier and host is not None
                and self.workload is not None and self.tuner.S_decode > 0):
            paged = compute_kv_paging(
                self.model, self.system, self.partition, self.tuner, self.framework, self.decode.memory,
                self.paging if self.paging is not None else KVPagingConfig(), inflight_sequences=inflight,
            )
            warm = warm_sessions(
                host=host,
                tier_name=self.offload.idle_tier,
                M_KV_session=kv_bytes_cluster(self.model, self.tuner.S_decode),
                N_active=paged.max_sequences,
                idle_fraction=self.workload.idle_fraction,
                n_gpus=replica_devices(self.partition),
                overflow_tiers=self.offload.overflow_tiers,
            )

        return E2EResults(
            TTFT=TTFT,
            TTFT_chunked=TTFT_chunked,
            t_second_token=t_second_token,
            TPOT=TPOT,
            throughput_per_gpu=throughput_per_gpu,
            interactivity=interactivity,
            TTPS=TTPS,
            t_handoff=t_handoff,
            t_repack=t_repack,
            M_KV_total=M_KV_total,
            t_sched=t_sched,
            t_framework_per_step=t_framework_per_step,
            handoff_streams=handoff_streams,
            handoff_rate_GBps=self._transfer.rate_GBps if self._transfer is not None else 0.0,
            handoff_bottleneck=self._transfer.bottleneck if self._transfer is not None else "",
            inflight_sequences=inflight,
            M_KV_inflight=inflight * M_KV_total,
            pools=pools,
            reuse=reuse,
            TTFT_hit=TTFT_hit,
            TTFT_expected=TTFT_expected,
            warm=warm,
        )

    def _streams(self) -> int:
        """Device pairs that carry the disaggregated transfer in parallel."""
        dg = self.disagg
        if dg.auto_streams:
            return min(replica_devices(self.prefill_partition), replica_devices(self.partition))
        return max(1, int(dg.transfer_streams))

    def _needs_repack(self) -> bool:
        """False only under `auto_repack` with equal KV layouts in the two pools."""
        if not self.disagg.auto_repack:
            return True
        return (kv_layout(self.model, self.prefill_partition, self.prefill_framework)
                != kv_layout(self.model, self.partition, self.framework))

    def _compute_handoff(self, M_KV_total: float, t_prefill: float) -> tuple[float, float]:
        """Compute t_handoff and t_repack from DisaggSpec (prefill.md §6.3/§6.4)."""
        dg = self.disagg

        if dg.disaggregated:
            # §6.4 disaggregated transfer
            if dg.inter_bandwidth_GBps <= 0:
                return 0.0, 0.0
            # The path: the link between pools (device pairs transferring in
            # parallel each carry their own shard), then, for a staged
            # transfer, the decode host's path from host memory into the GPUs.
            stages = [PathStageSpec(name="link", bw_per_port_GBps=dg.inter_bandwidth_GBps,
                                    ports=self._streams(), per="path",
                                    alpha_us=dg.inter_alpha_us + dg.N_WR * dg.tau_WR_us)]
            host = self.system.host
            kw = {}
            if dg.staged and host is not None and host.handoff_path:
                g_d = replica_devices(self.partition)
                stages += host.get_path(host.handoff_path)
                kw = dict(n_hosts=host.hosts_for(g_d), n_gpus=g_d, eta_busy=host.eta_busy,
                          block_bytes=host.block_MB * MB_TO_BYTES, tau_block_us=host.tau_block_us,
                          pipelined=host.pipelined)
            tr = self._transfer = transfer_time(M_KV_total, stages, **kw)
            t_repack = (
                M_KV_total / (dg.repack_GBps * GB_TO_BYTES) * dg.repack_eta
                if dg.repack_GBps > 0 and self._needs_repack() else 0.0
            )
            # Overlap with prefill shortens the handoff down to a residual
            # that cannot be overlapped; the residual is never longer than
            # the un-overlapped transfer.
            t_serial = tr.t_total + t_repack
            t_floor = min(dg.overlap_resid_us * US_TO_S, t_serial)
            t_handoff = max(t_floor, t_serial - dg.overlap_rho_KV * t_prefill)
            return t_handoff, t_repack

        # §6.3 co-located handoff
        t_repack = (
            M_KV_total / (dg.colo_repack_GBps * GB_TO_BYTES) * dg.colo_repack_eta
            if dg.colo_repack_GBps > 0 and self._needs_repack() else 0.0
        )
        t_handoff = dg.colo_alpha_us * US_TO_S + t_repack
        return t_handoff, t_repack
