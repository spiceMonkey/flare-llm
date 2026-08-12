# API Reference — Calculators

**Author:** Yue Lu  
**Date:** August 2026  

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
| `t_handoff` | `float` | — |
| `t_repack` | `float` | — |
| `M_KV_total` | `float` | — |
| `t_sched` | `float` | — |
| `t_framework_per_step` | `float` | — |

### class `E2ECalculator`

Assemble end-to-end metrics from decode + prefill + overhead + disagg (prefill.md §6, e2e.md).

#### `__init__(self, decode_results: InferenceResults, prefill_results: Optional[PrefillResults], overhead: OverheadSpec, disagg: DisaggSpec, model: LlmModelSpec, system: SystemSpec, partition: PartitionSpec, tuner: TuningSpec) -> None`

#### `run(self) -> E2EResults`

---
