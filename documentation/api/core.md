# API Reference — Core

**Author:** Yue Lu  
**Date:** August 2026  

Pure analytical model functions. Each takes (model, system, partition, tuner, framework) and returns a results dataclass; no global state.

---

## `llm_perf.core.memory_model`

### class `MemoryResults`

**Fields**

| Field | Type | Default |
|---|---|---|
| `M_theta_device` | `float` | — |
| `M_act_device` | `float` | — |
| `M_kv_token` | `float` | — |
| `M_kv_device` | `float` | — |
| `M_total_device` | `float` | — |
| `fits_in_HBM` | `bool` | — |
| `M_resident_per_tier` | `List[float]` | `field(default_factory=list)` |

### `compute_memory(model: LlmModelSpec, system: SystemSpec, partition: PartitionSpec, tuner: TuningSpec, framework: FrameworkSpec) -> MemoryResults`

Compute per-device static memory footprint (bytes).

Capacity check is per-tier (sram.md §1.3): the placement layer is invoked with the current B\_decode; CapacityError → fits\_in\_HBM=False (legacy field name retained). On a single-tier device the per-tier breakdown collapses to one entry equal to M\_total — matches pre-PR2 `M_total <= HBM_bytes` exactly.

---

## `llm_perf.core.decode_model`

Decode-phase performance model — consolidates flops + traffic + comm + latency.

Mirrors the shape of `prefill_model.py`: four pure functions on typed dataclasses, each returning a small result dataclass. Internally the phase-agnostic physics (weight/KV footprint, linear FLOPs, collective cost) lives in `core/primitives/`; this module wires those primitives together with decode-specific pieces (attention that scales with S, not S²; message sizes that are B·H·b, not S·H·b).

Documentation: `documentation/modeling/decode.md`.

Public surface (preserved from the pre-refactor four-file split):
- compute\_flops(model, partition, tuner) → FlopsResults
- compute\_traffic(model, partition, tuner) → TrafficResults
- compute\_comm(model, system, partition, tuner) → CommResults
- compute\_latency(model, system, partition, tuner, flops, traffic, comm) → LatencyResults

### class `FlopsResults`

**Fields**

| Field | Type | Default |
|---|---|---|
| `F_token_device` | `float` | — |
| `F_layer_per_device` | `float` | — |
| `F_step_device` | `float` | — |

### class `TrafficResults`

**Fields**

| Field | Type | Default |
|---|---|---|
| `T_theta` | `float` | — |
| `T_kv_token` | `float` | — |
| `T_kv_device` | `float` | — |
| `T_token_eff` | `float` | — |
| `T_step_eff` | `float` | — |

### class `CommResults`

**Fields**

| Field | Type | Default |
|---|---|---|
| `t_PP` | `float` | — |
| `t_TP` | `float` | — |
| `t_EP` | `float` | — |
| `t_SP` | `float` | — |
| `t_comm_stage` | `float` | — |
| `msg_PP_bytes` | `float` | — |
| `msg_TP_bytes` | `float` | — |
| `msg_EP_bytes` | `float` | — |
| `msg_SP_bytes` | `float` | — |

### class `LatencyResults`

**Fields**

| Field | Type | Default |
|---|---|---|
| `t_compute` | `float` | — |
| `t_compute_eff` | `float` | — |
| `eta_TC` | `float` | — |
| `t_mem` | `float` | — |
| `t_local` | `float` | — |
| `t_comm` | `float` | — |
| `t_stage` | `float` | — |
| `t_kernel` | `float` | — |
| `t_LM` | `float` | — |
| `t_step_user` | `float` | — |
| `pp_bubble_factor` | `float` | — |
| `TPS_single` | `float` | — |
| `TTPS` | `float` | — |
| `B` | `int` | — |
| `TPOT` | `float` | — |
| `B_star` | `float` | — |
| `N_tok_per_step` | `float` | `1.0` |
| `t_step_user_verify` | `float` | `0.0` |
| `TPOT_spec` | `float` | `0.0` |
| `t_step_seq` | `float` | `0.0` |

### `effective_peak_flops_TF(system: SystemSpec, bytes_per_param: float) -> float`

Precision-aware compute peak (TFLOPS) per device.

Uniform convention across all system specs: ``peak\_flops\_TF`` stores the **FP16 dense per-chip peak**. Lower precisions get a linear byte-
ratio boost: ``peak(p) = peak\_FP16 * (2 / bytes\_per\_param)``. Phase F:
multiplied by static `peak_flops_eta` (per-device sustained / nameplate deflator, sibling to MemoryTierSpec.eta\_beta on the memory side; ideal
1.0 = chip sustains nameplate FP16 dense peak).

Examples on GB200 NVL72 (peak\_FP16 = 2250 TF/GPU, peak\_flops\_eta = 1.0):
- FP16 (b=2.0): 2250 TF
- FP8  (b=1.0): 4500 TF
- FP4  (b=0.5): 9000 TF

**Known limitation**: d-Matrix MXINT4 throughput is 4× MXINT8 rather
than the 2× linear byte scaling predicts (block-sparse acceleration in the INT4 path). With FP16 baseline = 150 TF/chiplet, the framework computes 600 TF/chiplet for MXINT4, but the published peak is 1200 TF/chiplet — a 2× under-estimate on d-Matrix INT4 / FP4 only. Linear byte scaling holds for every other modeled architecture (NVIDIA Hopper / Blackwell, TPU v5p / v6e, Groq LPU).

### `compute_flops(model: LlmModelSpec, partition: PartitionSpec, tuner: TuningSpec, framework: FrameworkSpec) -> FlopsResults`

Per-device decode FLOPs per token (and per step).

### `compute_traffic(model: LlmModelSpec, partition: PartitionSpec, tuner: TuningSpec, framework: FrameworkSpec) -> TrafficResults`

Per-step HBM traffic per device (weights + KV cache read).

Dense weights (attention, dense-FFN) are read every step → traffic equals footprint. MoE expert weights are read only for the experts the current batch actually touches → traffic uses an expectation- of-touched-experts correction that converges to the footprint at large B. See `core/primitives/weight_quantities.py` for the formula and the uniform-routing assumption.

### `compute_comm(model: LlmModelSpec, system: SystemSpec, partition: PartitionSpec, tuner: TuningSpec, framework: FrameworkSpec) -> CommResults`

Per-stage decode communication time (seconds).

### `compute_latency(model: LlmModelSpec, system: SystemSpec, partition: PartitionSpec, tuner: TuningSpec, framework: FrameworkSpec, flops: FlopsResults, traffic: TrafficResults, comm: CommResults) -> LatencyResults`

Per-token latency and throughput (seconds, tokens/s).

The per-stage roofline gives the cost of one pipeline stage processing the current batch. For a user observing inter-token latency we apply a
pipeline-bubble correction when B < PP:
    t_step_base = max(t_stage_GPU, t_kernel) · max(1, PP / B) + t_LM
    t_step_user = t_step_base + max(0, t_step_seq
                                      - ρ_seq · t_step_base)

`t_stage_GPU` is the GPU-side compute + comm time (with optional Tensor Core efficiency derate at small microbatch). `t_kernel` is the per-round CPU dispatch budget (kernel\_launch\_overhead.md §5). The two are composed via `kernel_overlap_factor` ρ\_kernel: ρ\_kernel=1 means SW is fully hidden by GPU work (just `max`), ρ\_kernel=0 means strict serialization. `t_step_seq` is the per-sequence serving runtime overhead (decode.md §7.3). Gross host work is `t_step_seq = c_seq_us · B · 1e-6`; composition via `seq_overlap_factor` ρ\_seq uses the same physics as ρ\_kernel — under CUDA-Graph replay (ρ\_seq = 1, default) the CPU runs ahead and host work hides behind GPU compute until it exceeds `t_step_base`.

---

## `llm_perf.core.prefill_model`

### class `PrefillFlopsResults`

**Fields**

| Field | Type | Default |
|---|---|---|
| `F_proj_prefill` | `float` | — |
| `F_attn_kv_prefill` | `float` | — |
| `F_ffn_prefill` | `float` | — |
| `F_layer_prefill` | `float` | — |
| `F_prefill_device` | `float` | — |

### class `PrefillTrafficResults`

**Fields**

| Field | Type | Default |
|---|---|---|
| `T_theta_device` | `float` | — |
| `T_kv_write_per_request` | `float` | — |
| `T_prefill_device` | `float` | — |

### class `PrefillCommResults`

**Fields**

| Field | Type | Default |
|---|---|---|
| `t_TP_prefill` | `float` | — |
| `t_EP_prefill` | `float` | — |
| `t_SP_prefill` | `float` | — |
| `t_PP_prefill` | `float` | — |
| `t_prefill_comm` | `float` | — |

### class `PrefillLatencyResults`

**Fields**

| Field | Type | Default |
|---|---|---|
| `t_prefill_compute` | `float` | — |
| `t_prefill_compute_eff` | `float` | — |
| `eta_TC` | `float` | — |
| `t_prefill_mem` | `float` | — |
| `t_prefill_local` | `float` | — |
| `t_prefill_comm` | `float` | — |
| `t_kernel_per_stage` | `float` | — |
| `t_pipeline_warmup` | `float` | — |
| `t_LM_prefill` | `float` | — |
| `t_prefill` | `float` | — |
| `B_prefill` | `int` | — |
| `t_prefill_batched` | `float` | — |
| `chunk_size` | `int` | — |
| `n_chunks` | `int` | — |
| `t_prefill_chunked` | `float` | — |

### `compute_prefill_flops(model: LlmModelSpec, partition: PartitionSpec, tuner: TuningSpec, framework: FrameworkSpec) -> PrefillFlopsResults`

Per-device prefill FLOPs. Doc: documentation/modeling/prefill.md §1.5

Linear contribution (proj + FFN + router) comes from the shared `linear_flops_per_token` primitive scaled by S\_input. Attention is phase-specific (S² scaling) and stays inline.

### `compute_prefill_traffic(model: LlmModelSpec, partition: PartitionSpec, tuner: TuningSpec, framework: FrameworkSpec) -> PrefillTrafficResults`

Per-device HBM traffic for prefill pass.

### `compute_prefill_comm(model: LlmModelSpec, system: SystemSpec, partition: PartitionSpec, tuner: TuningSpec, framework: FrameworkSpec, *, tokens_per_step: int | None=None) -> PrefillCommResults`

Per-stage communication time for prefill (S\_input-scaled messages).

`tokens_per_step` lets callers evaluate the collectives at a token count other than `tuner.S_input`. Defaults to single-request behavior (tokens = S\_input). The latency path passes explicit values for batched (B\_prefill · S\_input) and chunked-per-chunk (C) cases.

### `compute_prefill_latency(system: SystemSpec, partition: PartitionSpec, tuner: TuningSpec, model: LlmModelSpec, flops: PrefillFlopsResults, traffic: PrefillTrafficResults, comm: PrefillCommResults, framework: FrameworkSpec) -> PrefillLatencyResults`

Hardware prefill latency: single-request, batched, and chunked.

---

## `llm_perf.core.kv_paging_model`

### class `KVPagingConfig`

KV paging parameters (documentation/modeling/kv.md §2).

**Fields**

| Field | Type | Default |
|---|---|---|
| `block_size` | `int` | `16` |
| `beam_width` | `int` | `1` |
| `system_overhead_GB` | `float` | `1.5` |

### class `KVPagingResults`

**Fields**

| Field | Type | Default |
|---|---|---|
| `M_block` | `float` | — |
| `N_blocks_per_seq` | `int` | — |
| `phi_avg` | `float` | — |
| `M_HBM_KV_avail` | `float` | — |
| `max_sequences` | `int` | — |
| `S_max` | `float` | — |

### `compute_kv_paging(model: LlmModelSpec, system: SystemSpec, partition: PartitionSpec, tuner: TuningSpec, framework: FrameworkSpec, memory: MemoryResults, paging: KVPagingConfig) -> KVPagingResults`

KV paging analysis. Doc: documentation/modeling/kv.md §2-6

---

## `llm_perf.core.memory_placement`

Per-data-class memory tier placement (sram.md §1.3, §2.1).

Standalone pure functions that:
1. `resolve_placement(...)` — split per-device weight bytes (T\_θ) and
     per-request KV bytes (T_KV) across the device's memory tier list,
     respecting the per-tier capacity constraint
       T_θ,i + B · T_KV,i ≤ C_i
     under the policy declared by `MemoryPlacementSpec` ("auto" greedy
     fastest-first, or operator-pinned to a named tier).
2. `t_mem_from_placement(...)` — assemble the multi-tier roofline memory
     time per sram.md §2.1 (full α–β form):
       t_mem(B) = Σ_i [ α_i · 𝟙(bytes_i > 0) + (T_θ,i + B · T_KV,i) / BW_eff,i ]

The α term is one transaction per tier per step (the simplest no-paging assumption — a single fetch covers all bytes on that tier). For the on-die / co-packaged tiers in scope here (SRAM / HBM / LPDDR5) it is structurally negligible (sram.md §2.1) and the JSON specs leave α\_i = 0; carrying it explicitly keeps the formula in the standard α–β form and lets small-read regimes reinstate it without a structural change. A future refinement (txn-granularity, see TODO at the bottom) would replace the 1-transaction count with ⌈bytes\_i / G\_i⌉ for those paged / small-read workloads.

Single-tier reduction: when the device exposes one tier (the legacy shim path from PR1) with α\_0 = 0, greedy "auto"/"auto" puts everything on tier 0 and `t_mem_from_placement` collapses to the legacy `T_step / BW_mem` expression — bitwise identical to pre-PR2 behavior.

Activations (T\_act) are not modeled here — `decode.md §2.2` drops them per the FlashAttention-applied assumption (sram.md §1.1 closing paragraph).

### class `CapacityError`

*Bases: `ValueError`*

Raised when a placement does not fit in the available device tiers.

Carries the data class that overflowed ("weights" or "kv") and the bytes still pending after walking all tiers, so callers can produce a diagnostic ("decode at B=N exceeds device capacity by X GB").

#### `__init__(self, data_class: str, overflow_bytes: float, message: str)`

### class `PlacementResult`

Output of `resolve_placement`.

Each list is indexed by tier (parallel to `device.get_tiers()`).
- weights\_per\_tier[i] = T\_θ,i (bytes resident on tier i)
- kv\_per\_request\_per\_tier[i] = T\_KV,i (per-request bytes on tier i)
Conservation: sum(weights\_per\_tier) = T\_θ,device,
              sum(kv_per_request_per_tier) = T_KV,device.

**Fields**

| Field | Type | Default |
|---|---|---|
| `weights_per_tier` | `List[float]` | — |
| `kv_per_request_per_tier` | `List[float]` | — |

### `resolve_placement(T_theta_device: float, T_kv_per_request_device: float, B: int, tiers: List[MemoryTierSpec], placement: MemoryPlacementSpec) -> PlacementResult`

Split T\_θ and T\_KV across the device's memory tiers per `placement`.

Args:
T\_theta\_device: per-device weight bytes (sum across all tiers). T\_kv\_per\_request\_device: per-device KV bytes for one request. B: number of in-flight requests sharing the device tiers. tiers: device's ordered tier list (fastest first). placement: per-data-class policy ("auto" greedy or "<name>" pin).

Returns:
PlacementResult with per-tier byte breakdown.

Raises:
CapacityError: if the placement does not fit in the available tiers. ValueError: if a pin name does not match any tier.

### `placement_fits(placement: PlacementResult, B: int, tiers: List[MemoryTierSpec]) -> bool`

True iff every tier's resident bytes (T\_θ,i + B · T\_KV,i) fit in C\_i.

Companion to `resolve_placement`. The "auto" greedy placement is permissive on overflow (bytes accumulate on the last tier); use this predicate to detect the unfit state without aborting latency math.

### `t_mem_from_placement(placement: PlacementResult, B: int, tiers: List[MemoryTierSpec], eta_beta_curve_factor: float=1.0) -> float`

Multi-tier decode roofline memory time per sram.md §2.1 (full α–β form):

    t_mem(B) = Σ_i [ α_i · 𝟙(bytes_i > 0) + (T_θ,i + B · T_KV,i) / BW_eff,i ]

where BW\_eff,i = BW\_i · η\_β,i (sram.md §1.2) and bytes\_i = T\_θ,i + B · T\_KV,i. The α\_i term counts one first-byte fetch per tier per step — the simplest no-paging assumption (one transaction covers all bytes resident on that tier). It is gated on bytes\_i > 0 so an unused tier pays no startup cost.

For the on-die / co-packaged tiers in scope here (SRAM / HBM / LPDDR5) α\_i is structurally negligible and the JSON specs leave it at 0 (sram.md §2.1 magnitude argument); the term is carried in the formula so the device roofline stays in the standard α–β form and so small-read regimes can reinstate it without restructuring.

TODO(txn-granularity): replace the 1-transaction count with N\_txn,i = ⌈bytes\_i / G\_i⌉ when small-read regimes (paged-attention KV blocks, flash-style weight chunks) come online. Current form is the upper-bound (most amortized) case.

Single-tier reduction: when len(tiers) = 1 with η\_β = 1.0 and α\_0 = 0 (PR1 legacy shim), this returns (T\_θ + B · T\_KV) / BW exactly — matches pre-PR2 `t_mem = T_step / BW_mem` to floating-point equality.

`eta_beta_curve_factor` multiplies the effective bandwidth of every tier (decode.md §6.2 / notation.md §20). It is the lookup result of `TuningSpec.bw_efficiency` at the current active-sequence count B —
1.0 when the curve is unset, preserving legacy behavior bitwise.

---

## `llm_perf.core.collective_algo_opt`

Post-partition collective-algorithm optimizer.

Standalone pure function that resolves `"auto"` placeholders in `FrameworkSpec` to concrete algorithm names per (phase × collective). Runs once after the partition is fixed. Returns a NEW FrameworkSpec with the auto fields resolved; non-`auto` fields pass through unchanged.

(Phase E: the algorithm fields moved from TuningSpec to FrameworkSpec. Pre-Phase-E this resolved into TuningSpec; the cost-model logic is unchanged, only the input/output type.)

Resolution policy:
- If `framework.inc_enabled` is True AND INC is structurally available
    for this op on this tier chain → pick `"inc"` directly (hardware
    deployment priority; SW costs not compared, since a deployment
    decision doesn't flip on a tuning-grade cost difference).
- Otherwise (INC unavailable for this op, or `inc_enabled=False`):
    enumerate the SW alternatives and pick `argmin(cost)`.

Resolution scope:
- tp\_algorithm\_decode  / tp\_algorithm\_prefill   (TP all-reduce)
- ep\_algorithm\_decode  / ep\_algorithm\_prefill   (EP MoE all-to-all)
- SP is always ring AG (only shipped variant per
    `collectives/01_collective_algorithms.md §6`).

If the partition makes a collective trivial (e.g. TP=1 → no AR work), the field resolves to `"ring"` as a stable sentinel (the dispatcher returns 0.0 either way).

### `optimize_collective_algorithms(model: LlmModelSpec, partition: PartitionSpec, system: SystemSpec, tuner: TuningSpec, framework: FrameworkSpec) -> FrameworkSpec`

Resolve `"auto"` algorithm fields on FrameworkSpec by cost-model selection. Returns a NEW FrameworkSpec with the auto fields resolved.

Args:
model: model architecture (provides H, bytes\_per\_param, MoE k\_active). partition: parallelism factors (TP, EP, SP, PP). system: system spec (provides tier chains for TP / EP). tuner: workload knobs (provides B\_decode, B\_prefill, S\_input — used
         to compute message size M for the cost model).
framework: SW-stack spec — algorithm fields read from here, INC
         selection respects `framework.inc_enabled`.

Returns:
A new FrameworkSpec with all `auto` fields replaced by concrete names. Non-`auto` fields pass through unchanged.

---
