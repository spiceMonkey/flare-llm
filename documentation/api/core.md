# API Reference — Core

**Author:** Yue Lu  
**Date:** November 2026  

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
| `pp_traversal_factor` | `float` | `1.0` |
| `microbatch` | `float` | `1.0` |
| `group_batch` | `float` | `1.0` |

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

A device whose published peaks do not follow that rule carries them in `DeviceSpec.peak_flops_TF_by_precision` (bytes per parameter → TFLOPs), which then replaces the byte-ratio rule. d-Matrix Corsair is the case in the database: its vendor figures quadruple per halving of precision (75 / 300 / 1200 TFLOPs per chiplet at 16 / 8 / 4 bits), so the linear rule would under-state its 4-bit peak by 2× and over-state its 16-bit peak by 2×. An exact key is used as is; a precision between two keys is interpolated geometrically; one outside the table is byte-ratio scaled from the nearest key.

### `compute_flops(model: LlmModelSpec, partition: PartitionSpec, tuner: TuningSpec, framework: FrameworkSpec) -> FlopsResults`

Per-device decode FLOPs per token (and per step).

### `compute_traffic(model: LlmModelSpec, partition: PartitionSpec, tuner: TuningSpec, framework: FrameworkSpec) -> TrafficResults`

Per-stage-pass HBM traffic per device (weights + KV cache read).

"Per pass" means one microbatch crossing one pipeline stage; a token pays this once on each of the PP stages (see `decode_microbatch`).

Dense weights (attention, dense-FFN) are read every step → traffic equals footprint. MoE expert weights are read only for the experts the current batch actually touches → traffic uses an expectation- of-touched-experts correction that converges to the footprint at large B. See `core/primitives/weight_quantities.py` for the formula and the uniform-routing assumption.

### `compute_comm(model: LlmModelSpec, system: SystemSpec, partition: PartitionSpec, tuner: TuningSpec, framework: FrameworkSpec) -> CommResults`

Per-stage decode communication time (seconds).

### `compute_latency(model: LlmModelSpec, system: SystemSpec, partition: PartitionSpec, tuner: TuningSpec, framework: FrameworkSpec, flops: FlopsResults, traffic: TrafficResults, comm: CommResults) -> LatencyResults`

Per-token latency and throughput (seconds, tokens/s).

The per-stage roofline gives the cost of one pipeline stage processing one microbatch of `mb = decode_microbatch(B, PP)` sequences. A token is not produced until its microbatch has crossed all PP stages and the LM
head, so the inter-token time is the traversal sum, not one stage:
    t_step_base = PP · max(t_stage_GPU(mb), t_kernel) + t_LM(mb)
    t_step_user = t_step_base + max(0, t_step_seq
                                      - ρ_seq · t_step_base)
Throughput is B tokens per t\_step\_user: the PP stages work on different microbatches concurrently, which is what PP buys. It does not shorten any one sequence's path through the L layers.

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

### class `SuffixPrefillResults`

**Fields**

| Field | Type | Default |
|---|---|---|
| `S_prefix` | `int` | — |
| `S_suffix` | `int` | — |
| `F_suffix_device` | `float` | — |
| `t_compute_eff` | `float` | — |
| `t_mem` | `float` | — |
| `t_local` | `float` | — |
| `t_comm` | `float` | — |
| `t_stage` | `float` | — |
| `t_LM` | `float` | — |
| `t_suffix` | `float` | — |

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

### `compute_suffix_prefill(model: LlmModelSpec, system: SystemSpec, partition: PartitionSpec, tuner: TuningSpec, framework: FrameworkSpec, S_prefix: int) -> SuffixPrefillResults`

Prefill of the last `S_input - S_prefix` tokens on top of a cached prefix.

The new tokens attend to every position of the prompt. Under the convention of the chunked model above (each query is charged the full tile, so chunk k of size C is charged (2k-1)*C positions) a suffix of C tokens after a prefix of P is charged 2P + C positions per query: its attention work is exactly that of the whole prompt less that of the prefix, 4*H*((P+C)^2 - P^2) per layer. It is priced as one chunk of that
model, for a prefix of ANY length:

compute  linear FLOPs of C tokens + that attention work memory   weights once + KV write of C entries + KV read of P entries comm     the stage's collectives at C tokens per step total    PP stages, then the LM head once

With P a multiple of C this is the chunk the chunked model prices for the same tokens; with P = 0 it is one chunk covering the whole prompt. Single request, like the chunked model (`tuner.B_prefill` enters only the Tensor-Core payload). `tuner.S_input` is the whole prompt.

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
| `M_act_per_seq` | `float` | `0.0` |
| `inflight_sequences` | `float` | `0.0` |

### `compute_kv_paging(model: LlmModelSpec, system: SystemSpec, partition: PartitionSpec, tuner: TuningSpec, framework: FrameworkSpec, memory: MemoryResults, paging: KVPagingConfig, *, inflight_sequences: float=0.0) -> KVPagingResults`

KV paging analysis. Doc: documentation/modeling/kv.md §2-6

`inflight_sequences` is the number of caches in flight to this replica during a disaggregated handoff (`E2EResults.inflight_sequences`): each already holds its cache in GPU memory but is not decoded yet, so it takes the room of a resident sequence's cache without its activations.

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

### `weights_seconds_per_byte(placement: PlacementResult, tiers: List[MemoryTierSpec]) -> float`

Seconds to read one byte of weights, averaged over where weights live.

`Σ_i (w_i / W) / BW_eff,i`: a read of weight bytes that are spread over the tiers like the resident weights are. Used for weight reads that are not part of the per-layer traffic (the LM head projection), so that a device whose weights sit mostly on a slow tier does not get them at the fast tier's bandwidth. On a single-tier device this is `1 / BW_eff`.

### `spill_seconds_per_byte(nbytes: float, placement: PlacementResult, B: int, tiers: List[MemoryTierSpec]) -> float`

Seconds to read one byte of a block that is placed *after* `placement`.

The block of `nbytes` takes whatever capacity `placement` (weights plus the KV cache of `B` requests) left free, fastest tier first; bytes that fit nowhere land on the last tier, as in the permissive "auto" rule of `resolve_placement`. Returns `Σ_i (x_i / nbytes) / BW_eff,i` for the resulting split `x_i`.

Used for the vocabulary matrix (input embedding / LM head) when it is given the lowest claim on fast memory: only the last pipeline stage reads it, once per token, while layer weights are read on every stage pass (sram.md §2).

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

Resolution policy: enumerate every schedule the fabric admits for the op and pick `argmin(cost)`. In-network collectives (`"inc"`) are one of the candidates when `framework.inc_enabled` is True and a crossed tier offers them; they win when they are cheaper, which is nearly always inside one switch domain, and lose when a software schedule is faster (a runtime picks its algorithm per call from predicted time and is not forced onto the switch path). An exact tie goes to `"inc"`.

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

## `llm_perf.core.interference_model`

Prefill interference in a shared (co-located) replica, and the comparison of a shared deployment with a split one.

A shared replica runs prefill and decode on the same devices and is modelled as time-sliced: while a prefill pass runs, decode steps wait. Steady state, closed form, no queueing.

d          = lambda\_rep * t\_prefill / B\_pf        fraction of wall clock spent in prefill TPOT\_colo  = TPOT(B) / (1 - d)                    mean time per output token gap\_max    = TPOT(B) + t\_block                    worst gap between two tokens lambda\_max = 1 / (t\_prefill / B\_pf + S\_out * TPOT(B) / B)

`t_block` is what one decode step can be made to wait: a whole prefill pass, or one chunk when prefill is chunked.

`lambda_max` is the request rate at which the B decode slots are all busy. It gives the shared replica S\_out * lambda\_max / g tokens per second per GPU, which is exactly what a balanced split of the same replica shape gives (`core.pool_model`): with equal shapes and a free handoff, sharing and splitting have the same mean throughput per GPU. The split wins on the worst gap, on a different shape or hardware per phase, and on the batch a latency target allows. `compare_shared_split` reports the last of these: the best throughput per GPU each deployment reaches inside a TPOT target and a worst-gap target.

### class `SharedReplicaResults`

**Fields**

| Field | Type | Default |
|---|---|---|
| `duty` | `float` | — |
| `TPOT` | `float` | — |
| `gap_max` | `float` | — |
| `t_block` | `float` | — |
| `rate_max` | `float` | — |
| `throughput_per_gpu_max` | `float` | — |
| `utilisation` | `float` | — |
| `feasible` | `bool` | — |

### class `DeploymentPoint`

The best operating point of one deployment inside the latency targets.

**Fields**

| Field | Type | Default |
|---|---|---|
| `feasible` | `bool` | — |
| `B_decode` | `int` | `0` |
| `throughput_per_gpu` | `float` | `0.0` |
| `requests_per_gpu` | `float` | `0.0` |
| `TPOT` | `float` | `0.0` |
| `gap_max` | `float` | `0.0` |

### class `SharedSplitComparison`

**Fields**

| Field | Type | Default |
|---|---|---|
| `shared` | `DeploymentPoint` | — |
| `split` | `DeploymentPoint` | — |
| `gain` | `float` | — |

### `shared_replica(*, rate: float, t_prefill: float, B_prefill: int, TPOT: float, B_decode: int, S_out: int, g: int, t_block: Optional[float]=None) -> SharedReplicaResults`

One shared replica of `g` devices serving `rate` requests per second.

`t_prefill` is the time of one prefill pass over `B_prefill` prompts on this replica (the chunked figure when prefill is chunked), `TPOT` the time per token of an uninterrupted decode step at batch `B_decode`. `t_block` defaults to the whole prefill pass.

### `compare_shared_split(*, tpot_shared: Mapping[int, float], tpot_decode_pool: Mapping[int, float], t_prefill_shared: float, t_prefill_pool: float, B_prefill: int, S_out: int, g_shared: int, g_prefill: int, g_decode: int, tpot_target: float=math.inf, gap_target: float=math.inf, t_block_shared: Optional[float]=None) -> SharedSplitComparison`

Best throughput per GPU of a shared and of a split deployment inside two latency targets.

`tpot_shared` and `tpot_decode_pool` map a decode batch to the uninterrupted time per token on the shared replica and on the decode pool's replica; each deployment picks the batch that serves it best.

Shared: at batch B the request rate is limited by the full decode batch (`rate_max`) and by the TPOT target, which caps the prefill duty at 1 - TPOT(B) / target. Its worst gap is one step plus `t_block_shared`.

Split: the decode pool is never interrupted, so its TPOT and its worst gap are both TPOT(B); throughput per GPU counts both pools at the balanced ratio. The handoff is taken as costing no GPU time.

---

## `llm_perf.core.kv_reuse`

Fetch-versus-recompute break-even for a cached prompt (KV-cache reuse).

A prompt whose key-value (KV) cache is already stored somewhere — host memory, a CXL pool, NVMe, another node — can be served either by recomputing it (a prefill pass) or by fetching the stored cache. Fetching wins on latency when moving the bytes is faster than running prefill over
the same tokens:

    M_KV / BW_fetch  <  t_prefill      <=>      BW_fetch  >  M_KV / t_prefill

The right-hand side is the break-even bandwidth: how fast a cache tier must deliver into the serving devices before it is worth having, for a given model, hardware, partition and prompt length. Both of its terms are outputs of the framework, so no new spec is needed.

`kv_prefix_reuse` generalises this to a cached PREFIX followed by a fresh suffix, prices the fetch with the shared transfer function over a path, and adds a hit rate: expected time to first token and expected GPU prefill time per request. The break-even becomes

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

### class `KVReuseBreakEven`

**Fields**

| Field | Type | Default |
|---|---|---|
| `S_input` | `int` | — |
| `M_KV` | `float` | — |
| `t_recompute` | `float` | — |
| `BW_break_even_GBps` | `float` | — |
| `fetch_bandwidth_GBps` | `Optional[float]` | `None` |
| `fetch_streams` | `int` | `1` |
| `t_fetch` | `Optional[float]` | `None` |
| `saving` | `Optional[float]` | `None` |
| `fetch_wins` | `Optional[bool]` | `None` |

### class `PrefixFetch`

Where a reusable prefix is fetched from, and when the fetch is taken.

`stages` is the path from the tier into the GPUs: the tier's read rate
first, then the host stages (`core.kv_transfer`). `rule`:
"auto"   - a latency rule: on a hit the fetch is taken only when
             fetch + suffix prefill is faster than prefilling the whole
             prompt; a skipped fetch costs a full prefill of GPU work;
"always" - every hit is fetched, whatever it does to latency, which
             keeps the whole saving in GPU work.

**Fields**

| Field | Type | Default |
|---|---|---|
| `stages` | `Sequence[PathStageSpec]` | — |
| `rule` | `str` | `'auto'` |

### class `PrefixReuseResults`

**Fields**

| Field | Type | Default |
|---|---|---|
| `S_input` | `int` | — |
| `S_prefix` | `int` | — |
| `hit_rate` | `float` | — |
| `rule` | `str` | — |
| `M_KV_prefix` | `float` | — |
| `t_prefill` | `float` | — |
| `t_suffix` | `float` | — |
| `t_fetch` | `float` | — |
| `fetch_rate_GBps` | `float` | — |
| `fetch_bottleneck` | `str` | — |
| `t_hit` | `float` | — |
| `fetch_taken` | `bool` | — |
| `t_first_hit` | `float` | — |
| `E_t_first` | `float` | — |
| `E_t_work` | `float` | — |
| `work_saved` | `float` | — |
| `BW_break_even_GBps` | `float` | — |

### `kv_fetch_break_even(model: LlmModelSpec, system: SystemSpec, partition: PartitionSpec, tuner: TuningSpec, framework: Optional[FrameworkSpec]=None, *, fetch_bandwidth_GBps: Optional[float]=None, fetch_streams: int=1, fetch_alpha_us: float=0.0) -> KVReuseBreakEven`

Break-even between fetching a cached prompt and recomputing it.

Args:
model, system, partition, tuner, framework: the usual five-spec
    composition of the prefill deployment. `tuner.S_input` is the
    prompt length; `tuner.B_prefill` and `tuner.chunk_size` are used as
    in `PrefillCalculator` (the unchunked prefill time is compared).
fetch\_bandwidth\_GBps: delivered fetch rate per stream. When given,
    the result also carries the fetch time and the latency saving.
fetch\_streams: device pairs (or drives, or links) fetching shards of
    the cache in parallel.
fetch\_alpha\_us: fixed cost per fetch (first-byte latency, lookup).

Returns:
KVReuseBreakEven. `BW_break_even_GBps` is the total delivered bandwidth (all streams together) a tier needs for fetching to match recomputation.

### `kv_prefix_reuse(model: LlmModelSpec, system: SystemSpec, partition: PartitionSpec, tuner: TuningSpec, framework: Optional[FrameworkSpec]=None, *, S_prefix: int, hit_rate: float, fetch: PrefixFetch, t_prefill: Optional[float]=None) -> PrefixReuseResults`

Latency and GPU work of a request when a prefix of its prompt may be cached.

`system` / `partition` are the deployment that prefills (the prefill pool when there are two). `tuner.S_input` is the whole prompt. `t_prefill` is the whole-prompt prefill time when the caller already has it; otherwise it is computed. The fetch is one transfer of the prefix's cache into the replica's GPUs; `system.host`, when present, supplies the hosts the replica spans and the host-side calibration terms.

A prefix equal to the whole prompt is priced like `kv_fetch_break_even`:
nothing is prefilled on a hit.

---

## `llm_perf.core.kv_store_model`

Warm memory: the off-GPU tier list of a host, and what it does for sessions.

Hot memory is GPU memory and is read on every decode step; it is modelled by the decode and prefill calculators and is not touched here. Warm memory is host memory (DRAM, CXL) that holds caches which are NOT being decoded: a reusable prefix, or the cache of an idle session. It is read only when a cache moves, so it never enters the decode step; it enters the end-to-end layer through

* the fetch of a prefix on a hit      (`prefix_fetch_from` -> core.kv\_reuse),
* the capacity for idle sessions      (`warm_sessions`),
* the store and resume of a session   (`tier_store`, `tier_fetch`).

Every move is priced by `core.kv_transfer.transfer_time` over the tier's path: the tier's own rate first (read rate for a fetch, write rate for a store), then the stages of the host path the tier names.

Session capacity. A replica holds N\_active sequences in GPU memory
(`KVPagingResults.max_sequences`). A tier holds a whole number of idle caches:

N\_parked   = floor(n\_hosts * C\_tier / M\_KV(S)) N\_sessions = min(N\_active / (1 - f\_idle), N\_active + N\_parked)

With a fraction f\_idle of sessions idle at any moment, N\_active busy ones correspond to N\_active / (1 - f\_idle) sessions in all, as long as there is room for the idle ones. An idle cache may stay in GPU memory while a slot is free there; only the overflow is parked, so the room for all caches together is N\_active + N\_parked. (Requiring every idle cache to leave the GPUs would give the stricter N\_parked / f\_idle.)

The tier list is ordered and each tier names its own path, so a flash tier or a remote store is one more entry; nothing here is specific to DRAM.

### class `WarmSessionResults`

**Fields**

| Field | Type | Default |
|---|---|---|
| `tier` | `str` | — |
| `M_KV_session` | `float` | — |
| `capacity_bytes` | `float` | — |
| `N_active` | `int` | — |
| `N_parked_capacity` | `int` | — |
| `N_sessions` | `float` | — |
| `N_parked` | `float` | — |
| `sessions_gain` | `float` | — |
| `limited_by` | `str` | — |
| `t_store` | `float` | — |
| `t_resume_fetch` | `float` | — |
| `store_rate_GBps` | `float` | — |
| `fetch_rate_GBps` | `float` | — |
| `fetch_bottleneck` | `str` | — |

### `tier_stages(host: HostSpec, tier: KVTierSpec, *, write: bool=False) -> List[PathStageSpec]`

The path between a tier and the GPUs: the tier itself, then its host path.

### `tier_fetch(M_bytes: float, host: HostSpec, tier_name: str, n_gpus: int) -> TransferResult`

Time to bring `M_bytes` of cache from a tier into `n_gpus` GPUs.

### `tier_store(M_bytes: float, host: HostSpec, tier_name: str, n_gpus: int) -> TransferResult`

Time to move `M_bytes` of cache from `n_gpus` GPUs to a tier.

### `prefix_fetch_from(host: Optional[HostSpec], offload: Optional[KVOffloadSpec])`

The `core.kv_reuse.PrefixFetch` the strategy implies, or None when it names no prefix tier.

### `warm_sessions(*, host: HostSpec, tier_name: str, M_KV_session: float, N_active: int, idle_fraction: float, n_gpus: int) -> WarmSessionResults`

Sessions one replica serves when idle caches are parked in a warm tier.

`M_KV_session` is the whole cache of one session (all of its shards); `N_active` the sequences the replica holds in GPU memory; `n_gpus` the replica's devices, which fixes how many hosts (and so how much of the tier) it has.

---

## `llm_perf.core.kv_transfer`

One transfer function for every move of a KV cache.

A cache of M bytes moves over a PATH: an ordered chain of stages, written in the same format as a network fabric's tiers (ports in parallel, a rate per port, a latency). The stage latencies add, and the bytes move at the slowest
stage:

t\_move(M) = sum\_i alpha\_i + n\_blocks * tau\_block + M / min\_i(rate\_i)

rate\_i = ports\_i * BW\_i            for a hardware stage
         = eta_busy * ports_i * BW_i for a host-side stage (per = "host")

This is the reduction the network model applies to a chain of switch tiers (alpha\_total, BW\_min). The same function prices

* the prefill -> decode handoff (first stage = the link between pools),
* a fetch or a resume from an off-GPU tier (first stage = the tier's read rate),
* a store to an off-GPU tier (first stage = the tier's write rate).

`ports` of a stage counts per host, per GPU or per path (`PathStageSpec.per`), so one chain describes a host with any number of GPUs and a replica that spans any number of hosts.

A copy that is staged whole at each stage (store-and-forward) pays every stage's time in turn; `pipelined=False` selects that.

### class `TransferResult`

**Fields**

| Field | Type | Default |
|---|---|---|
| `t_total` | `float` | — |
| `t_alpha` | `float` | — |
| `t_blocks` | `float` | — |
| `t_bytes` | `float` | — |
| `rate_GBps` | `float` | — |
| `bottleneck` | `str` | — |
| `n_blocks` | `int` | — |

### `stage_rate(stage: PathStageSpec, *, n_hosts: int=1, n_gpus: int=1, eta_busy: float=1.0) -> float`

Bytes per second one stage carries for a transfer into `n_gpus` GPUs on `n_hosts` hosts.

### `transfer_time(M_bytes: float, stages: Sequence[PathStageSpec], *, n_hosts: int=1, n_gpus: int=1, eta_busy: float=1.0, block_bytes: float=0.0, tau_block_us: float=0.0, pipelined: bool=True) -> TransferResult`

Time to move `M_bytes` over `stages` (see the module docstring).

---

## `llm_perf.core.pool_model`

Two-pool sizing for phase disaggregation (prefill pool + decode pool).

Pure functions over quantities the decode and prefill calculators already produce. Steady state, closed form, no queueing: a pool is a set of identical replicas, each of which completes requests at a fixed service rate.

mu\_p = B\_pf / t\_prefill(S\_in, B\_pf)        requests/s of one prefill replica mu\_d = B / (S\_out * TPOT(B))               requests/s of one decode replica

With g devices per replica, one request costs g\_p / mu\_p GPU-seconds of prefill and g\_d / mu\_d GPU-seconds of decode, so

G\_p / G\_d      = (g\_p / mu\_p) / (g\_d / mu\_d)          balanced device ratio tokens/s / GPU = S\_out / (g\_p / mu\_p + g\_d / mu\_d)    over both pools

Neither depends on the request rate. Given a rate and a utilisation cap, the whole-replica counts follow by rounding up.

`kv_layout` says how a sequence's cache is split across the devices of a replica; a handoff needs a repack only when the two pools' layouts differ.

### class `PoolResults`

**Fields**

| Field | Type | Default |
|---|---|---|
| `g_prefill` | `int` | — |
| `g_decode` | `int` | — |
| `mu_prefill` | `float` | — |
| `mu_decode` | `float` | — |
| `gpu_s_prefill` | `float` | — |
| `gpu_s_decode` | `float` | — |
| `pool_ratio` | `float` | — |
| `throughput_per_gpu` | `float` | — |
| `requests_per_gpu` | `float` | — |
| `n_prefill_replicas` | `Optional[int]` | `None` |
| `n_decode_replicas` | `Optional[int]` | `None` |
| `n_prefill_gpus` | `Optional[int]` | `None` |
| `n_decode_gpus` | `Optional[int]` | `None` |
| `n_gpus` | `Optional[int]` | `None` |
| `utilisation_prefill` | `Optional[float]` | `None` |
| `utilisation_decode` | `Optional[float]` | `None` |

### `replica_devices(partition: PartitionSpec) -> int`

Devices in one replica: PP * TP * EP * SP.

### `kv_layout(model: LlmModelSpec, partition: PartitionSpec, framework: FrameworkSpec) -> Tuple[int, int, int]`

How one sequence's KV cache is split across a replica: (PP, D\_kv\_eff, SP).

PP splits it by layer, SP by sequence position, and the KV divisor by head (or by sequence under DP-attention). A latent-attention cache is not head-structured, so under TP-attention TP does not split it (attention.md §3.6) and the divisor is 1. EP does not enter: an EP group holds whole sequences, so it changes where a cache lives and not its shape.

### `compute_pools(*, t_prefill: float, B_prefill: int, TPOT: float, B_decode: int, g_prefill: int, g_decode: int, workload: WorkloadSpec) -> PoolResults`

Size a prefill pool and a decode pool for one prompt / output shape.

`t_prefill` is the time one prefill replica takes for a batch of `B_prefill` prompts; `TPOT` is the user-observed time per output token of a decode replica running `B_decode` sequences.

---
