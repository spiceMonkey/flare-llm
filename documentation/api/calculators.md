# API Reference — Calculators

**Author:** Yue Lu  
**Date:** November 2026  

End-to-end entry points. Each composes the specs, runs the five-stage pipeline, and returns a results dataclass.

---

## `llm_perf.calculators.inference_calculator`

### class `InferenceResults`

**Fields**

| Field | Type | Default |
|---|---|---|
| `memory` | `MemoryResults` | — |
| `flops` | `FlopsResults` | — |
| `traffic` | `TrafficResults` | — |
| `comm` | `CommResults` | — |
| `latency` | `LatencyResults` | — |

### class `InferenceCalculator`

High-level façade for inference performance modeling.

Five-spec composition (`model × system × partition × tuner × framework`). `framework` is optional; when omitted, defaults to `FrameworkSpec.default()` (neutral / no-overhead — pure roofline). Pass an explicit framework to model production stack behavior; load via `load_framework_from_db("dynamo_trt")` or construct ad-hoc with `FrameworkSpec(...)`.

#### `__init__(self, model: LlmModelSpec, system: SystemSpec, partition: PartitionSpec, tuner: TuningSpec, framework: Optional[FrameworkSpec]=None) -> None`

#### `run(self) -> InferenceResults`

---

## `llm_perf.calculators.prefill_calculator`

### class `PrefillResults`

**Fields**

| Field | Type | Default |
|---|---|---|
| `flops` | `PrefillFlopsResults` | — |
| `traffic` | `PrefillTrafficResults` | — |
| `comm` | `PrefillCommResults` | — |
| `latency` | `PrefillLatencyResults` | — |

### class `PrefillCalculator`

Prefill performance calculator (documentation/modeling/prefill.md).

Five-spec composition (`model × system × partition × tuner × framework`), matching `InferenceCalculator`. `framework` is optional; when omitted it defaults to `FrameworkSpec.default()` (neutral roofline).

#### `__init__(self, model: LlmModelSpec, system: SystemSpec, partition: PartitionSpec, tuner: TuningSpec, framework: Optional[FrameworkSpec]=None) -> None`

#### `run(self) -> PrefillResults`

---

## `llm_perf.calculators.e2e_calculator`

### class `E2EResults`

**Fields**

| Field | Type | Default |
|---|---|---|
| `TTFT` | `float` | — |
| `TTFT_chunked` | `float` | — |
| `TPOT` | `float` | — |
| `throughput_per_gpu` | `float` | — |
| `interactivity` | `float` | — |
| `TTPS` | `float` | — |
| `t_second_token` | `float` | — |
| `t_handoff` | `float` | — |
| `t_repack` | `float` | — |
| `M_KV_total` | `float` | — |
| `t_sched` | `float` | — |
| `t_framework_per_step` | `float` | — |
| `handoff_streams` | `int` | `0` |
| `handoff_rate_GBps` | `float` | `0.0` |
| `handoff_bottleneck` | `str` | `''` |
| `inflight_sequences` | `float` | `0.0` |
| `M_KV_inflight` | `float` | `0.0` |
| `reuse` | `Optional[Any]` | `None` |
| `TTFT_hit` | `float` | `0.0` |
| `TTFT_expected` | `float` | `0.0` |
| `warm` | `Optional[WarmSessionResults]` | `None` |
| `pools` | `Optional[PoolResults]` | `None` |

### class `E2ECalculator`

Assemble end-to-end metrics from decode + prefill + overhead + disagg (prefill.md §6, e2e.md).

#### `__init__(self, decode_results: InferenceResults, prefill_results: Optional[PrefillResults], overhead: OverheadSpec, disagg: DisaggSpec, model: LlmModelSpec, system: SystemSpec, partition: PartitionSpec, tuner: TuningSpec, *, workload: Optional[WorkloadSpec]=None, prefill_system: Optional[SystemSpec]=None, prefill_partition: Optional[PartitionSpec]=None, framework: Optional[FrameworkSpec]=None, prefill_framework: Optional[FrameworkSpec]=None, prefix_fetch: Optional[Any]=None, offload: Optional[KVOffloadSpec]=None, paging: Optional[KVPagingConfig]=None) -> None`

`system` / `partition` describe the decode replica. The keyword arguments are all optional and leave every result unchanged when
omitted:

workload           tokens generated per request and, optionally, the
                   request rate: turns on two-pool sizing.
prefill\_system,    the prefill pool, when it is not the decode prefill\_partition  replica's own shape. `prefill_results` must then
                   come from a PrefillCalculator run on them.
framework,         the stacks of the two pools; only consulted to prefill\_framework  compare KV layouts under `DisaggSpec.auto_repack`,
                   and to price the suffix prefill under reuse.
prefix\_fetch       a `core.kv_reuse.PrefixFetch`: the path a cached
                   prefix is fetched over and the fetch rule. Turns
                   on prefix reuse when the workload has a hit rate
                   and a prefix length.
offload            the warm-memory strategy. Its prefix tier (on the
                   prefill system's host) supplies the fetch path
                   when `prefix_fetch` is not given; its idle tier
                   (on the decode system's host) parks idle sessions.
paging             KV paging parameters used to count the sequences
                   the decode replica holds in GPU memory; defaults.

#### `run(self) -> E2EResults`

---
