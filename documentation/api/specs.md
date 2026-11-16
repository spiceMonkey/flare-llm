# API Reference — Specs

**Author:** Yue Lu  
**Date:** November 2026  

Typed dataclass inputs. The composition is model x system x partition x tuner x framework, plus optional overhead/disagg specs for end-to-end runs.

---

## `llm_perf.specs.model_spec`

### class `MoESpec`

MoE configuration, if the model uses experts.

**Fields**

| Field | Type | Default |
|---|---|---|
| `n_experts` | `int` | — |
| `k_active` | `int` | — |
| `I_moe` | `int` | — |
| `n_moe_layers` | `Optional[int]` | `None` |

### class `MLASpec`

Multi-head Latent Attention configuration (DSv3-style).

When present on `LlmModelSpec.mla`, replaces the GQA per-head K/V accounting with a head-shared latent representation. The per-token KV cache stores only the joint latent `c_KV` of dimension `d_c + d_qk_rope`, not per-head K and V. Per-layer attention parameters are the sum of six matrices (W\_DQ, W\_UQ, W\_DKV, W\_UK, W\_UV, W\_O); see `documentation/modeling/attention.md §3.3`.

When `mla` is set, the GQA-derived `n_kv` field on `LlmModelSpec` is ignored for KV-cache and attention-parameter accounting.

**Fields**

| Field | Type | Default |
|---|---|---|
| `d_c` | `int` | — |
| `d_q_c` | `int` | — |
| `d_qk_nope` | `int` | — |
| `d_qk_rope` | `int` | — |
| `d_v` | `int` | — |

#### `kv_bytes_per_token_per_layer(self, bytes_per_param: float) -> float`

Per-token-per-layer KV cache base for MLA: `(d_c + d_qk_rope) * b`.

See `attention.md §3.4`. Note no factor of 2: MLA caches a single joint latent, not separate K and V.

#### `per_layer_attn_params(self, H: int, n_q: int) -> int`

Per-layer attention parameter count: sum of six matrices.

See `attention.md §3.3` for the derivation.

#### `per_layer_attn_params_replicated(self, H: int, n_q: int) -> int`

Down-projection params: W\_DQ + W\_DKV.

Not head-structured — under TP-attention these are replicated on every rank (`attention.md §3.6`). `n_q` is unused but retained for API symmetry with `per_layer_attn_params_shardable`.

#### `per_layer_attn_params_shardable(self, H: int, n_q: int) -> int`

Up-projection + output params: W\_UQ + W\_UK + W\_UV + W\_O.

Head-structured — under TP-attention these divide by G\_TP across ranks (`attention.md §3.6`). Independent of execution mode (the materialized vs absorbed distinction in §3.5 affects only when W\_UK / W\_UV multiply by data, not whether they exist).

### class `LlmModelSpec`

Transformer / LLM architecture spec.

**Fields**

| Field | Type | Default |
|---|---|---|
| `name` | `str` | — |
| `L` | `int` | — |
| `H` | `int` | — |
| `n_q` | `int` | — |
| `n_kv` | `int` | — |
| `I_dense` | `int` | — |
| `vocab_size` | `int` | — |
| `max_seq_len` | `int` | — |
| `bytes_per_param` | `float` | — |
| `moe` | `Optional[MoESpec]` | `None` |
| `mla` | `Optional[MLASpec]` | `None` |
| `kv_bytes_per_elem` | `Optional[float]` | `None` |
| `act_bytes_per_elem` | `Optional[float]` | `None` |

#### `kv_bytes(self) -> float`

Bytes per KV-cache element.

#### `act_bytes(self) -> float`

Bytes per activation / logit / collective-payload element.

#### `d_head(self) -> float`

Head dimension d\_head = H / n\_q.

#### `H_kv(self) -> float`

KV projection dimension H\_kv = n\_kv * d\_head (GQA / MHA path).

---

## `llm_perf.specs.system_spec`

### class `MemoryTierSpec`

One memory tier exposed by a device.

Per `documentation/modeling/sram.md §1.1`, devices expose an ordered list of memory tiers, fastest first. Each tier carries a capacity, an effective peak read bandwidth, a first-byte latency, and a sustained-bandwidth deflator. The `eta_beta` deflator follows the same convention as
collective contention (`collectives/05_contention_and_congestion.md`):
1.0 = peak, < 1 = sustained-throughput losses (HBM refresh + bank
conflicts ≈ 0.92, LPDDR5 ≈ 0.85, SRAM ≈ 1.0; sram.md §1.2).

The `alpha_us` first-byte cost enters the device roofline as one transaction per tier per step (sram.md §2.1, full α–β form). For the on-die / co-packaged tiers in scope here it is structurally negligible (SRAM ~1 ns, HBM ~10 ns, LPDDR5 ~200 ns) and the JSON specs leave it at 0; it is kept on the spec so the formula stays in the standard α–β form and so small-read regimes (paged-attention block fetch, flash-style spill — sram.md §2.1) can reinstate it.

**Fields**

| Field | Type | Default |
|---|---|---|
| `name` | `str` | — |
| `capacity_GB` | `float` | — |
| `bandwidth_GBps` | `float` | — |
| `alpha_us` | `float` | `0.0` |
| `eta_beta` | `float` | `1.0` |

### class `DeviceSpec`

Single device (GPU/NPU) specs.

The legacy fields `hbm_capacity_GB` / `hbm_bandwidth_GBps` model the device's main DRAM-like tier and remain required for back-compat. Two
additive paths to multi-tier:

1. **Top-level `sram_capacity_MB` / `sram_bandwidth_TBps`** (PR3): a
     fast on-die SRAM cache layered on top of the main DRAM tier.
     Shipped naming convention for SRAM-augmented devices like d-Matrix
     Corsair (SRAM + LPDDR5 represented as SRAM + "hbm" slot). Units
     chosen to match natural magnitudes: MB and TB/s.

2. **Explicit `tiers: List[MemoryTierSpec]`** (PR1): an ordered list
     for arbitrary multi-tier topologies. Use this when the field-name
     conventions of (1) don't fit (e.g., a 3+ tier device, or non-HBM
     main memory where you want an accurate `name` like "lpddr5").

Downstream code should call `get_tiers()` instead of reading any of these
fields directly. The shim materializes:

- `tiers` if non-empty (path 2)
- `[SRAM tier, HBM tier]` if `sram_*` is set (path 1, sram.md §1.1)
- `[HBM tier]` otherwise (legacy single-tier; preserves regression)

Auto-materialized tiers use `eta_beta = 1.0` to keep `t_mem` numerically identical to the legacy `T_step / BW_mem` formula. New devices wanting the sram.md §1.2 defaults (HBM=0.92, LPDDR5=0.85) should use path 2.

**Fields**

| Field | Type | Default |
|---|---|---|
| `name` | `str` | — |
| `hbm_capacity_GB` | `float` | — |
| `hbm_bandwidth_GBps` | `float` | — |
| `peak_flops_TF` | `float` | — |
| `sram_capacity_MB` | `Optional[float]` | `None` |
| `sram_bandwidth_TBps` | `Optional[float]` | `None` |
| `tiers` | `List['MemoryTierSpec']` | `field(default_factory=list)` |
| `peak_flops_eta` | `float` | `1.0` |
| `peak_flops_TF_by_precision` | `Optional[Dict[float, float]]` | `None` |
| `hbm_eta_beta` | `float` | `1.0` |
| `tensor_core_efficiency` | `Optional[Dict[int, float]]` | `None` |
| `bw_efficiency` | `Optional[Dict[int, float]]` | `None` |

#### `get_tiers(self) -> List['MemoryTierSpec']`

Return the device's memory tier list, materializing a shim from the legacy / sram\_* fields when `tiers` is empty.

Materialization order (sram.md §1.1, fastest first):
- explicit `tiers` (if non-empty)
- `[SRAM tier, HBM tier]` if `sram_*` set (PR3 convention)
- `[HBM tier]` (legacy single-tier path)

### class `CrossbarTier`

One crossbar switching tier (single monolithic chip abstraction).

Default topology for all existing system JSONs. `ports` is the radix — the number of ranks reachable within this tier from any single rank. Cumulative reach at tier k is the product of ports over tiers 0..k. See documentation/modeling/collectives/02\_topology\_mapping.md §2 (star) / collectives/04\_in\_network\_collectives.md (INC) for the cost forms consumed by each tier kind.

Contention coefficients `eta_alpha` (≥ 1, inflates α) and `eta_beta` (∈ (0, 1], deflates BW) coarsen dynamic contention into the α–β model per documentation/modeling/contention.md. Defaults are 1.0 (ideal).

`inc` flags in-network collective support on this tier's switch ASIC:
"none" (software collectives only), "sharp\_class" (NVLink SHARP / NVLS, Quantum SHARP, Spectrum-X SHARP — switch ALU + multicast crossbar; accelerates AR / AG / RS), or "hw\_a2a" (Tomahawk Ultra / Rubin-class crossbar scatter-gather; accelerates AR / AG / RS *and* A2A). The legacy values "nvls" and "sharp" are accepted by the loader and mapped to "sharp\_class" for backwards compatibility. When set, `inc_alpha_us` optionally overrides the switch-cut-through α used on the INC path —
0.0 means "reuse alpha\_us" (i.e., model assumes the tier's α already
captures cut-through latency).

`schedule_island_ports` (default 0 = off) lets software on this single-tier fabric also group its ranks into islands of that many ports and run the hierarchical schedule (reduce-scatter in each island, a sub-all-reduce across islands, all-gather in each island) that a multi-tier chain of the same geometry is given; the dispatcher then uses the cheaper of the flat and the grouped schedule. A switch that reaches every rank in one hop can run any grouping, so this is a schedule choice, not a hardware property. Set it on an idealized flat reference fabric that is compared against a tiered one (to the tiered fabric's inner radix): without it the flat fabric is only offered one ring or tree over the whole group and can price above the fabric it is meant to bound. Use the tiered fabric's inner radix and nothing smaller: a smaller island would also regroup collectives that fit inside one island of the tiered fabric, and the two fabrics would then differ for reasons other than the tier boundary. Left at 0 on real systems, where a pinned algorithm keeps its literal single-level meaning.

See documentation/modeling/collectives/04\_in\_network\_collectives.md and documentation/modeling/collectives/04\_in\_network\_collectives.md.

**Fields**

| Field | Type | Default |
|---|---|---|
| `name` | `str` | — |
| `ports` | `int` | — |
| `bw_per_port_GBps` | `float` | — |
| `alpha_us` | `float` | — |
| `topology` | `str` | `'crossbar'` |
| `eta_alpha` | `float` | `1.0` |
| `eta_beta` | `float` | `1.0` |
| `inc` | `str` | `'none'` |
| `inc_alpha_us` | `float` | `0.0` |
| `oversubscription` | `float` | `1.0` |
| `schedule_island_ports` | `int` | `0` |

### class `TorusTier`

One k-D torus tier.

`dims` is (D\_1, ..., D\_k); reach is prod(dims). Each node has 2k neighbor links; `bw_per_port_GBps` is the per-link single-direction bandwidth. Diameter = sum(floor(D\_i/2)); bisection cut binds the largest dim. See documentation/modeling/collectives/02\_topology\_mapping.md §3 for the dim-decomposed primitive cost forms.

Contention coefficients `eta_alpha`, `eta_beta` as in `CrossbarTier`; see documentation/modeling/contention.md.

**Fields**

| Field | Type | Default |
|---|---|---|
| `name` | `str` | — |
| `dims` | `Tuple[int, ...]` | — |
| `bw_per_port_GBps` | `float` | — |
| `alpha_us` | `float` | — |
| `topology` | `str` | `'torus'` |
| `eta_alpha` | `float` | `1.0` |
| `eta_beta` | `float` | `1.0` |

#### `ports(self) -> int`

Reach = prod(dims). Exposed for compatibility with span\_tiers.

### class `MeshTier`

One mesh tier — handles both full mesh and k-D mesh via the `full` flag.

`full=True`: full mesh — every node connects directly to every other node (single hop, full bisection). $\binom{N}{2}$ edges. Examples: chiplet UCIe interposer, DGX-1/DGX-2 hybrid cube-mesh (legacy). Cost formulas match a single-tier crossbar exactly (single hop, full bisection); the dispatcher routes a full-mesh tier through the crossbar primitives. `dims` is expected to be a 1-tuple `(N,)` here.

`full=False`: k-D mesh — torus without wraparound edges. Open-line bucket brigade along each axis. AR/AG/RS cost formulas match torus exactly (open-line still telescopes BW-optimally); A2A pays a 2× BW penalty versus torus because the bisection cut is halved (missing wraparound edges) — $D_\mathrm{max}/4$ instead of $D_\mathrm{max}/8$. The dispatcher routes a k-D-mesh tier through the torus primitives with `wraparound=False`.

No `inc` field — mesh has no switch ASIC; INC is structurally absent.

See documentation/modeling/collectives/02\_topology\_mapping.md §4 (mesh) and the explainer `02_topology_mapping.md §4` for full vs k-D mesh derivations.

**Fields**

| Field | Type | Default |
|---|---|---|
| `name` | `str` | — |
| `dims` | `Tuple[int, ...]` | — |
| `bw_per_port_GBps` | `float` | — |
| `alpha_us` | `float` | — |
| `full` | `bool` | `False` |
| `topology` | `str` | `'mesh'` |
| `eta_alpha` | `float` | `1.0` |
| `eta_beta` | `float` | `1.0` |

#### `ports(self) -> int`

Reach = prod(dims). For full mesh dims=(N,) so ports=N; for k-D mesh ports = ∏ dim\_i = N (total node count). Exposed for compatibility with span\_tiers and the dispatcher's tier walk.

### class `FabricSpec`

One physical network technology, described as an ordered tier list.

A fabric represents a single underlying network (e.g. NVLink5, InfiniBand, Ethernet). Collectives map onto an ordered chain of fabrics in `SystemSpec.collective_fabrics`; tiers from every fabric in the chain are walked innermost-first to cost a collective via `span_tiers`.

**Fields**

| Field | Type | Default |
|---|---|---|
| `name` | `str` | — |
| `tiers` | `List[TierSpec]` | `field(default_factory=list)` |

### class `PathStageSpec`

One stage of a path that a KV cache takes between memory and the GPUs.

Same format as a network fabric tier (`ports` in parallel, a rate per port, a latency), so a root complex, a PCIe link, a CXL device or a CXL switch is each one entry and a path is an ordered list of them. The stage latencies add and the bytes move at the slowest stage (`core.kv_transfer.transfer_time`).

`per` says what `ports` counts:
"host" - per host; the stage is CPU-side (a root complex, the copy
           through pinned memory) and takes `HostSpec.eta_busy`;
"gpu"  - per GPU that receives the transfer (a PCIe link into a GPU); "path" - as written (the links between two pools, the CXL devices or
           drives behind one path).

**Fields**

| Field | Type | Default |
|---|---|---|
| `name` | `str` | — |
| `bw_per_port_GBps` | `float` | — |
| `ports` | `int` | `1` |
| `per` | `str` | `'path'` |
| `alpha_us` | `float` | `0.0` |

### class `KVTierSpec`

One off-GPU tier of a host that can hold KV caches (warm memory).

Not a device memory tier (`MemoryTierSpec`): the decode step never reads it. It is read only when a cache moves, and it reaches the GPUs over a path of the host (`HostSpec.paths`), which `path` names.

Host DRAM and CXL memory are described this way today. A flash tier is the same entry with a write rate below its read rate; a remote store is the same entry whose path contains a network stage.

**Fields**

| Field | Type | Default |
|---|---|---|
| `name` | `str` | — |
| `capacity_GB` | `float` | — |
| `read_GBps` | `float` | — |
| `write_GBps` | `Optional[float]` | `None` |
| `ports` | `int` | `1` |
| `alpha_us` | `float` | `0.0` |
| `path` | `str` | `''` |

### class `HostSpec`

The host side of a system: how many GPUs share a host, and the paths a KV cache takes between the host and its GPUs.

Optional on `SystemSpec`. The decode and prefill calculators never read it; only the end-to-end layer does. Without it no host limit is applied to any transfer.

`paths` maps a name to an ordered chain of stages, from the memory side to the GPU. The rate of a CPU-side stage (`per="host"`) is a measured, per-platform number, not a derate of the PCIe lane rate: on a current 8-GPU server the lanes add up to several hundred GB/s while the measured rate into the GPUs through the root complex is tens of GB/s.

Calibration terms, all neutral by default:
eta\_busy      extra loss on the CPU-side stages while the host is busy
                serving, in (0, 1];
tau\_block\_us  CPU cost per cache block moved, charged
                ceil(bytes / block_MB) times when both are set;
pipelined     True: the stages stream, the slowest sets the rate;
                False: each stage stores the whole cache before the next
                starts, so the stage times add.

**Fields**

| Field | Type | Default |
|---|---|---|
| `devices_per_host` | `int` | — |
| `paths` | `Dict[str, List[PathStageSpec]]` | `field(default_factory=dict)` |
| `handoff_path` | `str` | `''` |
| `eta_busy` | `float` | `1.0` |
| `tau_block_us` | `float` | `0.0` |
| `block_MB` | `float` | `0.0` |
| `pipelined` | `bool` | `True` |
| `tiers` | `List[KVTierSpec]` | `field(default_factory=list)` |

#### `get_tier(self, name: str) -> KVTierSpec`

#### `get_path(self, name: str) -> List[PathStageSpec]`

#### `hosts_for(self, n_devices: int) -> int`

Hosts that a replica of `n_devices` GPUs spans.

### class `SystemSpec`

System / cluster description.

Networks are declared once as named fabrics; each collective (TP/EP/SP/PP) maps to an ordered list of fabric names it escalates through. Collectives spanning more ranks than a fabric's innermost tier can reach continue into the next fabric in the chain — this lets scale-up (NVLink) and scale-out (IB/Ethernet) be modeled as distinct physical networks.

**Fields**

| Field | Type | Default |
|---|---|---|
| `name` | `str` | — |
| `device` | `DeviceSpec` | — |
| `num_devices` | `int` | — |
| `fabrics` | `Dict[str, FabricSpec]` | — |
| `collective_fabrics` | `Dict[str, List[str]]` | — |
| `host` | `Optional[HostSpec]` | `None` |

#### `get_fabric_chain(self, collective: str) -> List[FabricSpec]`

Return the ordered FabricSpec list for a collective (TP/EP/SP/PP).

#### `get_tier_chain(self, collective: str) -> List[TierSpec]`

Return the concatenated tier list across the collective's fabric chain.

### `span_tiers(tiers: List[TierSpec], group_size: int) -> Tuple[float, float, int]`

Return (alpha\_total\_us, bw\_min\_GBps, n\_tiers\_crossed) for a collective over `group_size` ranks walking `tiers` innermost-first.

Accumulates α across every tier touched and floors bandwidth to the narrowest tier actually crossed. Returns (0, 0, 0) for trivial collectives. If `group_size` exceeds total reach, returns the outermost configuration.

Crossbar-only helper: the α-sum / BW-min flatten pattern is exact for crossbar tiers but approximate for torus. Topology-aware dispatch lives in `core/primitives/dispatch.py`.

---

## `llm_perf.specs.partition_spec`

### class `PartitionSpec`

Parallel partitioning of the model across devices.

Purely describes how we shard: PP, TP, EP, SP. DP is inferred from the total number of devices available.

Phase H: `attention_mode` and `tp_ep_layout` (formerly `layout`) were moved to FrameworkSpec — they describe how the stack dispatches the attention block and how TP / EP map to physical GPUs, both stack-axis decisions, not sharding-factor decisions. PartitionSpec now carries only the four parallelism factors. The compose-time invariants formerly enforced here (co-located tp\_ep\_layout requires TP == EP and attention\_mode == "dp") are now enforced by `core.primitives.sharding_factors.compose_check(partition, framework)`.

See `notation.md §1` for the unified deployment-knob abstraction (the per-component effective sharding factors D\_attn, D\_exp, D\_kv, D\_emb that combine PartitionSpec with FrameworkSpec to encode all three production- relevant configurations in one lookup table) and `decode.md §1.4 / §5.3` for the per-device formulas. The framework helpers in `core/primitives/sharding_factors.py` resolve those abstract factors from the (PartitionSpec, FrameworkSpec) join.

**Fields**

| Field | Type | Default |
|---|---|---|
| `PP` | `int` | — |
| `TP` | `int` | — |
| `EP` | `int` | — |
| `SP` | `int` | — |

---

## `llm_perf.specs.tuner_spec`

### class `MemoryPlacementSpec`

Per-data-class memory tier placement (sram.md §1.3).

Each field selects the tier that holds the corresponding data class:
- "auto": greedy fastest-first — fill faster tiers first, spill to
    slower tiers when capacity is exhausted (sram.md §1.3 first policy).
- "<tier\_name>": pin this data class to the named tier (must match a
    `MemoryTierSpec.name` on the device); CapacityError if it doesn't fit
    (sram.md §1.3 second policy — d-Matrix Aviator-style mode toggle).

`auto_priority` controls the greedy tiebreaker when **both** fields are "auto": which class claims the fastest tier first. Default "weights" matches the convention that weights are a stable size for a given deployment and should pin to the fast tier. Set "kv" to flip the order when KV-bound workloads (long context, large batch) want SRAM-resident KV at the cost of spilling weights. Inert when either class is explicitly pinned.

Defaults are "auto" / "auto" / "weights", which on a single-tier device collapses to the legacy "everything on HBM" behavior — bitwise identical to pre-PR2 `t_mem = T_step / BW_mem`.

**Fields**

| Field | Type | Default |
|---|---|---|
| `weights_tier` | `str` | `'auto'` |
| `kv_tier` | `str` | `'auto'` |
| `auto_priority` | `str` | `'weights'` |

### class `TuningSpec`

Workload knobs and chip-side calibration curves. Orthogonal to:
- ModelSpec (architecture)
- SystemSpec (hardware + per-tier eta)
- PartitionSpec (sharding)
- FrameworkSpec (SW-stack runtime: host overhead, collective algos,
                   MLA mode, MoE A2A pattern, etc.)

**Fields**

| Field | Type | Default |
|---|---|---|
| `S_decode` | `int` | `2048` |
| `B_decode` | `int` | `1` |
| `S_input` | `int` | `0` |
| `B_prefill` | `int` | `1` |
| `chunk_size` | `int` | `0` |
| `placement` | `MemoryPlacementSpec` | `field(default_factory=MemoryPlacementSpec)` |
| `n_tok_draft` | `int` | `0` |
| `p_accept` | `float` | `0.0` |

---

## `llm_perf.specs.framework_spec`

FrameworkSpec — SW-stack-specific runtime behavior.

Captures the host-side overhead model, execution-mode choices, and collective-dispatch knobs that depend on the production serving stack rather than the workload, model, or hardware. Five orthogonal spec
axes:

    ModelSpec    = architecture (LlmModelSpec, MoESpec, MLASpec)
    SystemSpec   = hardware (DeviceSpec, FabricSpec, MemoryTierSpec)
    PartitionSpec = sharding (PP, TP, EP, SP)
    TuningSpec   = workload (S, B, chunk_size, placement, speculation,
                   chip-side derate curves: tensor_core_efficiency,
                   bw_efficiency)
    FrameworkSpec = stack runtime (host overhead, kernel launch budget,
                   collective algorithms + counts + overlap, MLA mode,
                   MoE A2A pattern, INC opt-in)

Any (model, system, partition, tuner, framework) tuple is a runnable deployment configuration. Pre-canned framework JSONs live in `database/framework/` and correspond 1:1 to the production stack identifiers in InferenceX measurement metadata. See `documentation/modeling/decode.md §7.1, §7.2` (host overhead), `§5` (collective algorithms), `§6` (overlap composition).

### class `FrameworkSpec`

SW-stack-specific runtime behavior model.

Sourced from `decode.md §7.1, §7.2` (host overhead) and `attention.md §3.5` (MLA execution mode). Per-stack typical values live in the `database/framework/` JSONs.

**Fields**

| Field | Type | Default |
|---|---|---|
| `name` | `str` | — |
| `c_seq_us` | `float` | `0.0` |
| `c_orch_us` | `float` | `0.0` |
| `seq_overlap_factor` | `float` | `1.0` |
| `kernel_launch_us` | `float` | `1.5` |
| `kernels_per_layer_compute` | `int` | `10` |
| `kernels_per_collective_call` | `int` | `2` |
| `kernels_per_pp_hop` | `int` | `2` |
| `kernel_overlap_factor` | `float` | `1.0` |
| `moe_a2a_pattern` | `str` | `'gather'` |
| `mla_mode` | `str` | `'absorbed'` |
| `inc_enabled` | `bool` | `True` |
| `tp_algorithm_decode` | `str` | `'ring'` |
| `tp_algorithm_prefill` | `str` | `'ring'` |
| `ep_algorithm_decode` | `str` | `'ring'` |
| `ep_algorithm_prefill` | `str` | `'ring'` |
| `torus_algorithm` | `str` | `'ring'` |
| `torus_align_policy` | `str` | `'prefix'` |
| `sp_decode_mode` | `str` | `'pass_q'` |
| `n_TP_collectives` | `int` | `2` |
| `n_EP_collectives` | `int` | `2` |
| `n_SP_collectives` | `int` | `1` |
| `attention_mode` | `str` | `'tp'` |
| `tp_ep_layout` | `str` | `'orthogonal'` |
| `comm_overlap_factor` | `float` | `0.0` |

#### `default(cls) -> 'FrameworkSpec'`

Neutral / no-overhead defaults — matches the legacy TuningSpec defaults before the framework split. Algorithms are concrete ("ring" everywhere) so the calculator runs out-of-box without requiring an `optimize_collective_algorithms` pass — pure roofline. Use a per-stack JSON (e.g. `load_framework_from_db("dynamo_trt")`) when modeling a production stack; those JSONs default to algorithm="auto" so the optimizer selects per cell.

---

## `llm_perf.specs.overhead_spec`

### class `OverheadSpec`

Framework / CPU-stack overhead parameters (documentation/modeling/framework.md).

Scope: CPU and software-stack overheads only. Network-fabric overheads (disaggregated KV transfer, co-located repack) live in `DisaggSpec`.

All values default to 0.0 (zero-overhead baseline).

**`t_graph_us` is a legacy fallback.** With the kernel-launch refactor
(kernel\_launch\_overhead.md §5), the per-round CUDA-graph dispatch budget is derived from TuningSpec (`kernels_per_layer_compute`, `kernels_per_collective_call`, `kernel_launch_us`) and surfaced as `LatencyResults.t_kernel`. The E2E calculator uses `t_graph_us` only when `LatencyResults.t_kernel == 0` (SW modeling disabled by setting `kernel_launch_us = 0` in the tuner). Setting both is harmless — the derived term takes precedence — but `t_graph_us` is now redundant for most users.

**Fields**

| Field | Type | Default |
|---|---|---|
| `t_sched_us` | `float` | `0.0` |
| `t_tok_us` | `float` | `0.0` |
| `t_graph_us` | `float` | `0.0` |
| `t_detok_us` | `float` | `0.0` |

---

## `llm_perf.specs.disagg_spec`

### class `DisaggSpec`

KV-handoff configuration (prefill.md §6).

The `disaggregated` flag selects which branch applies; the other branch's fields are ignored. Defaults give zero handoff cost (co-located, partition-matched).

**Fields**

| Field | Type | Default |
|---|---|---|
| `disaggregated` | `bool` | `False` |
| `first_token` | `str` | `'decode'` |
| `colo_alpha_us` | `float` | `0.0` |
| `colo_repack_GBps` | `float` | `0.0` |
| `colo_repack_eta` | `float` | `1.0` |
| `inter_alpha_us` | `float` | `0.0` |
| `inter_bandwidth_GBps` | `float` | `0.0` |
| `transfer_streams` | `int` | `1` |
| `auto_streams` | `bool` | `False` |
| `N_WR` | `int` | `0` |
| `tau_WR_us` | `float` | `0.0` |
| `overlap_rho_KV` | `float` | `0.0` |
| `overlap_resid_us` | `float` | `0.0` |
| `staged` | `bool` | `False` |
| `repack_GBps` | `float` | `0.0` |
| `repack_eta` | `float` | `1.0` |
| `auto_repack` | `bool` | `False` |

---

## `llm_perf.specs.kv_offload_spec`

### class `KVOffloadSpec`

Warm-memory strategy: what is kept off the GPUs, and where.

Read only by the end-to-end layer. The hardware it refers to is the off-GPU tier list of the host (`HostSpec.tiers`); this spec names tiers of that list and says when a cached prefix is fetched. The workload side (how often a prefix is hit, what fraction of sessions is idle) is on `WorkloadSpec`.

Hot memory is untouched by it: weights and the cache of a sequence that is being decoded always stay in GPU memory.

prefix\_tier  the tier that holds reusable prefixes; "" = none, so no
               prefix reuse is priced from a tier;
idle\_tier    the tier that holds the caches of idle sessions; "" =
               none, so no session is parked;
overflow\_tiers  further tiers that take idle sessions once `idle_tier`
               is full, filled in the order given (for example host DRAM
               first, then flash). Empty = one tier only;
fetch\_rule   "auto":   a hit is fetched only when fetch + suffix
                         prefill beats prefilling the whole prompt (a
                         latency rule; a skipped fetch costs a full
                         prefill of GPU work);
               "always": every hit is fetched.

**Fields**

| Field | Type | Default |
|---|---|---|
| `prefix_tier` | `str` | `''` |
| `idle_tier` | `str` | `''` |
| `fetch_rule` | `str` | `'auto'` |
| `overflow_tiers` | `Tuple[str, ...]` | `()` |

---

## `llm_perf.specs.workload_spec`

### class `WorkloadSpec`

Request-level workload, read only by the end-to-end layer.

The decode and prefill calculators never see this spec: they price one step and one prefill pass. The end-to-end layer needs to know how many tokens a request generates, and optionally how many requests arrive, to turn those two prices into GPU time per request and a pool size.

Prompt length and the two batch sizes stay on `TuningSpec` (`S_input`, `B_prefill`, `B_decode`).

**Fields**

| Field | Type | Default |
|---|---|---|
| `output_tokens` | `int` | — |
| `request_rate` | `Optional[float]` | `None` |
| `utilisation_cap` | `float` | `1.0` |
| `prefix_hit_rate` | `float` | `0.0` |
| `prefix_tokens` | `int` | `0` |
| `idle_fraction` | `float` | `0.0` |

---
