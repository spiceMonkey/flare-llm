# API Reference — Core primitives

**Author:** Yue Lu  
**Date:** November 2026  

Building blocks the core models compose: sharding factors and partition layout, weight / KV / FLOP quantities, collective cost formulas, topology-aware dispatch, and the per-stage communication aggregator.

---

## `llm_perf.core.primitives.sharding_factors`

Per-component effective sharding factors (D\_*, G\_*) and replica size.

Implements the unified deployment-knob abstraction documented in `notation.md §1` and used throughout `decode.md §1.4 / §2.1 / §2.3 / §3.5 / §5 / §6.1` and the symmetric prefill formulas.

The framework supports four production-relevant `(tp_ep_layout, attention_mode)`
configurations. The lookup table:

    tp_ep_layout  attention_mode  D_attn  D_exp(MoE)   D_kv          D_emb  G_TP    G_EP  N_replica
    orthogonal    tp              TP      TP*EP        TP   (head)   TP     TP(AR)  EP    PP*TP*EP*SP
    co_located    tp              TP      EP           TP   (head)   TP     TP(AR)  EP    PP*max(TP,EP)*SP
    orthogonal    dp              1       TP*EP        TP   (seq)    TP     TP(AG)  EP    PP*TP*EP*SP
    co_located    dp              1       EP           max(TP,EP)    TP     TP(AG)  EP    PP*max(TP,EP)*SP
                                                        (seq)

Dense FFN always uses D\_exp = TP regardless of tp\_ep\_layout (no EP axis to overlap). The fourth combination (co\_located + tp) models the production pattern where TP=EP overlay on the same physical GPU set (e.g., the DSr1 / NVL72 panel-(b) deployment with TP=EP=8 on 8 GPUs) and attention is head-sharded across the same group. The TP all-reduce and the MoE all-to-all both run on that shared 8-GPU set; the structural invariant `TP == EP` under co-location still holds.

Phase H: `attention_mode` and `tp_ep_layout` live on FrameworkSpec, not on PartitionSpec. Every helper here takes both `partition` (PP/TP/EP/SP) and `framework` (attention\_mode/tp\_ep\_layout). At the default (orthogonal + tp) every helper collapses to the legacy raw TP / EP / SP form.

### `D_attn(partition: PartitionSpec, framework: FrameworkSpec) -> int`

Effective per-device divisor for attention weights (Q/K/V/O).

Returns TP under TP-attention (head-shard); 1 under DP-attention (weights replicated on every TP rank).

### `D_exp(partition: PartitionSpec, framework: FrameworkSpec, *, layer_kind: str='moe', n_exp_cap: int | None=None) -> int`

Effective per-device divisor for FFN / expert weights.

`layer_kind` selects the divisor convention for the layer in question:
- "dense" → always TP (no EP axis to overlap; co-location does not apply
    to dense layers).
- "moe" → TP*EP under orthogonal tp\_ep\_layout, EP under co-located.

`n_exp_cap` (MoE only) clamps the effective EP by the number of experts in the model. Required when EP > N\_exp would otherwise inflate the divisor beyond what is physically meaningful: with EP=8 but only 2 experts, only 2 ranks can hold a unique expert, so the divisor is 2 not
8. Pass `model.moe.n_experts` from the call site.

For a model with mixed dense + MoE layers, callers compute each layer's contribution with the matching `layer_kind`.

### `D_kv(partition: PartitionSpec, framework: FrameworkSpec) -> int`

Effective per-device divisor for KV cache (head- or sequence-shard).

Excludes the SP axis. Per-device KV memory and traffic always carry an additional /SP factor on top of D\_kv.

Returns:
- TP under orthogonal tp\_ep\_layout (head-shard under TP-attn or
    sequence-shard across the TP-as-DP-attn group under DP-attn —
    same byte count).
- max(TP, EP) under co-located tp\_ep\_layout. Under DP-attention this
    is the sequence-shard across the entire replica's GPU set; under
    TP-attention (TP == EP forced under co-location) this is the head-
    shard across the TP group, which is the same set of ranks as the
    EP group — numerically equal to max(TP, EP) since TP == EP.

### `D_emb(partition: PartitionSpec) -> int`

Effective per-device divisor for embedding / LM head weights.

Always TP — embeddings and LM head are TP-sharded along the vocab dimension regardless of tp\_ep\_layout or attention\_mode.

### `G_TP(partition: PartitionSpec) -> int`

Collective group size for TP collectives (AR or AG).

### `G_EP(partition: PartitionSpec) -> int`

Collective group size for EP all-to-all.

### `N_replica(partition: PartitionSpec, framework: FrameworkSpec) -> int`

Devices per model replica.

Orthogonal tp\_ep\_layout: PP * TP * EP * SP. Co-located tp\_ep\_layout: PP * max(TP, EP) * SP (TP and EP overlay on the same physical GPU set).

### `compose_check(partition: PartitionSpec, framework: FrameworkSpec) -> None`

Validate the combined `(partition, framework)` invariants.

One structural invariant applies at the (partition, framework) join point. Calculator paths that consume both specs should call this once at the start of pipeline assembly.

Raises ValueError on:
- tp\_ep\_layout='co\_located' with TP != EP when both exceed 1. Co-located layouts overlay
    TP and EP on the same physical GPUs (e.g., DSr1 NVL72 TP=EP=8 on
    8 GPUs); asymmetric (TP, EP) on a co-located layout is not modeled.

Both attention\_mode values ('tp' and 'dp') are allowed under either layout. The co\_located + tp\_attention combination models the production pattern where TP heads and EP experts both shard across the same physical GPU set — see the module docstring's four-row lookup table for the resulting D\_attn / D\_exp / D\_kv / N\_replica semantics.

### `decode_microbatch(B: int, PP: int) -> float`

Sequences one pipeline stage processes per pass (decode.md §7.2).

`B` is the number of sequences decoding concurrently on one replica. Autoregressive decode cannot start a sequence's next token until its previous token has left the last stage, so the `B` sequences circulate through the `PP` stages as `min(B, PP)` microbatches, one per stage at
a time:

    mb = B / PP   if B ≥ PP   (every stage busy, PP microbatches in flight)
    mb = 1        if B < PP   (B single-sequence microbatches, PP − B stages idle)

Every per-pass quantity — FLOPs, weight and KV traffic, collective payloads, the LM head — is evaluated at `mb`, not `B`. At PP = 1 this is `B` exactly.

### `D_batch(partition: PartitionSpec, framework: FrameworkSpec) -> int`

Number of device groups the replica's batch is divided across.

Under the orthogonal layout a replica is `EP` groups of `TP` devices. Each group carries its own copy of the attention and dense-FFN weights and its own KV cache, so the groups serve *different* sequences: the batch is divided `EP` ways for everything except the expert FFN, whose tokens are routed across all `TP · EP` expert shards by the all-to-all. Without this division every group would recompute the whole batch's attention and hold the whole batch's KV cache — `EP` copies of the same work.

Under the co-located layout TP and EP are the same devices, so there is one group and the divisor is 1 (DP-attention's own `B / G_TP` batch shard is applied separately, where it always was).

### `group_batch(B: float, partition: PartitionSpec, framework: FrameworkSpec) -> float`

Sequences one device group handles: `B / D_batch`, floored at one.

Same shape as `decode_microbatch`: with fewer sequences than groups, each busy group holds one sequence and the rest are idle, so the latency of a step is that of a group batch of 1, not of a fraction.

---

## `llm_perf.core.primitives.partition_layout`

Nested-partition layout: which fabric tier holds which parallelism axis.

The nested-partitioning rule used across the framework — and standard in production deployments — is

    DP (outermost / slowest) → PP → EP → TP → SP (innermost / fastest)

with collective-heavy axes (TP, EP, SP) pinned to the innermost (highest- bandwidth) fabric tier and bandwidth-tolerant axes (PP, DP) spilling to outer tiers as needed. ``assign\_tier\_per\_axis`` resolves a `PartitionSpec` + `SystemSpec` into a per-axis tier index under this rule.

The output is consumed by the comm-cost path so that, e.g., PP send/recv between two adjacent stages is priced at the *correct* tier — the framework would otherwise price every PP hop at tier 0 because `cost_collective` is called with `G=2` (single hop), which always resolves to the innermost tier even when PP physically spans an outer fabric tier (e.g., d-Matrix squadrack PP=32 with TP=8 spanning the ethernet rack-tier, not just the package D2D mesh).

Under TP+EP co-location (`FrameworkSpec.tp_ep_layout == "co_located"`), TP and EP are overlaid on the same physical GPU set, so the cumulative-reach calculation must count them once (as `max(TP, EP)`) rather than as their product. The helper `_axis_size_for_reach` encodes this.

### `axis_stride(partition: PartitionSpec, framework: FrameworkSpec, axis: str) -> int`

Device slots each rank of `axis`'s collective group occupies.

Under the nested layout an axis's group is built out of whole blocks of the axes inside it: a tensor-parallel rank is one sequence-parallel group, an expert-parallel rank is one TP×SP block, a pipeline stage is one EP×TP×SP block. The stride is the size of that block, so a group of `G` ranks physically spans `G · stride` devices. `cost_collective` uses it to decide which fabric tiers the collective crosses and how many of the group's ranks share an inner island.

Co-located layouts overlay EP on the TP device set, so the EP stride is the TP stride (`SP`), not `SP · TP`.

### `assign_tier_per_axis(partition: PartitionSpec, framework: FrameworkSpec, system: SystemSpec, role: str='TP', order: Tuple[str, ...]=DEFAULT_ORDER) -> Dict[str, int]`

Map each parallelism axis to a fabric-tier index under nested layout.

Walks ``order`` from innermost to outermost. For each axis, the cumulative product of inner-axis sizes is multiplied by this axis's size; the resulting "group reach" is matched against the cumulative reach of the fabric tiers (``prod(tier.ports)``), and the smallest tier whose reach covers the group is assigned.

Trivial axes (size <= 1) get ``tier\_index = 0`` (no comm; no cost). Groups that exceed the fabric's outermost reach also clamp to the outermost tier (the cost will be the worst case for that fabric).

Co-located layouts merge TP and EP via ``\_axis\_size\_for\_reach`` so the cumulative product matches the actual replica size ``PP * max(TP, EP) * SP`` rather than the orthogonal `PP*TP*EP*SP`.

Returns ``{axis\_name: tier\_index}`` for every axis in ``order``. The dict ordering matches ``order`` so callers can iterate predictably.

### `replicas_in_system(system: SystemSpec, replica_size: int, stages: int=1, role: str='PP') -> int`

How many whole replicas the system holds, respecting fabric islands.

A replica is `stages` pipeline stages, each a block of `replica_size // stages` devices (the TP x EP x SP group). The block is the unit whose collectives are priced on the inner fabric tier, so it is the unit that must sit whole inside an island; consecutive stages may be in different islands, and their hop is priced on the tier that joins them. Blocks therefore tile islands, and a replica needs `stages` of
them:

    blocks   = islands x floor(island / block)  (+ whatever fits in a partial island)
    replicas = blocks // stages

Eight 72-GPU racks hold eight 64-GPU blocks: eight single-stage replicas, four two-stage ones, or two three-stage ones — not the nine, four and three that dividing 576 by the replica size gives. Devices left over inside an island idle.

On a single-island fabric this is `num_devices // replica_size`. A block larger than the outermost island reach also falls back to the flat count; its collectives are already priced on the outer tiers.

### `tier_at(system: SystemSpec, role: str, tier_index: int) -> TierSpec`

Return the tier at ``tier\_index`` of the role's fabric chain.

Out-of-range indices clamp to the outermost tier rather than raising, matching the behavior of ``assign\_tier\_per\_axis``.

---

## `llm_perf.core.primitives.weight_quantities`

Weight (θ) footprint and per-step traffic primitives — per-device bytes.

This module hosts two complementary quantities for the transformer-weights (θ) bookkeeping. They share most of the same algebra but differ in one crucial place for MoE — see `moe_weight_traffic_bytes` below.

────────────────────────────────────────────────────────────────────────── Footprint vs. traffic ──────────────────────────────────────────────────────────────────────────

* **Footprint (`M_θ`)** — maximum resident set in HBM. Must hold *every*
expert weight even if a given step touches only some of them, because the *next* step might select any expert. Used by `compute_memory`.

* **Per-step traffic (`T_θ`)** — bytes actually loaded into the compute
pipeline this step. For dense weights (attention, dense-FFN) traffic equals footprint (every weight is read every step). For MoE expert weights traffic depends on which experts the current batch actually selects: at small B, only ~B·k\_active expert-touch events fire and most expert weights sit idle in HBM that step; at large enough B, the touched fraction → 1 and traffic converges to the full footprint, which it never exceeds. Used by `compute_traffic`.

The decode roofline gates per-step compute against `T_θ / BW_HBM`, so using the footprint as the traffic surrogate (the historical default, exact for dense models) over-counts t\_mem for MoE at small B.

────────────────────────────────────────────────────────────────────────── Splits ──────────────────────────────────────────────────────────────────────────

Dense and MoE layers share the same attention projections (Q,K,V,O) but differ in the FFN: dense always TP-shards the FFN; MoE shards experts by D\_exp (= TP*EP under orthogonal tp\_ep\_layout, EP under co-located).

All functions return bytes on one device (D-factor and PP aware). The abstract divisors D\_attn / D\_exp / D\_emb are resolved from the PartitionSpec via `sharding_factors`; see `notation.md §1` and `decode.md §1.4`. Embedding bytes are returned separately because they're outside the layer loop.

### `dense_weight_bytes(model: LlmModelSpec, partition: PartitionSpec, framework: FrameworkSpec) -> float`

Per-device weight bytes for the dense-layer slice of the model.

    M_theta_dense = (L_dense / PP) · (attn_per_device + 3HI_dense/TP) · b

where `attn_per_device` is the per-device per-layer attention parameter count from `_per_layer_attn_params_per_device` (handles GQA / MHA's P/D\_attn and MLA's TP-attn-aware shardable / replicated split). Dense FFN always uses /TP regardless of tp\_ep\_layout.

### `moe_weight_bytes(model: LlmModelSpec, partition: PartitionSpec, framework: FrameworkSpec) -> float`

Per-device weight footprint (M\_θ) for the MoE-layer slice.

    M_theta_moe = (L_moe / PP) · (attn_per_device + 3HI_moe·N_exp/D_exp) · b

This is the *maximum resident set* required in HBM — every expert weight must be held even if a given step touches only some of them (the next step may select any expert). For per-step **traffic** (bytes loaded into the compute pipeline this step), use `moe_weight_traffic_bytes(...)` which applies the expectation-of- touched-experts correction.

Returns 0.0 for dense-only models.

### `moe_weight_traffic_bytes(model: LlmModelSpec, partition: PartitionSpec, framework: FrameworkSpec, B: float) -> float`

Per-device per-stage-pass **traffic** (T\_θ) for the MoE-layer slice.

Differs from `moe_weight_bytes` (the footprint) in the expert-FFN term: only the experts the current batch actually touches contribute to per-step HBM traffic. Attention projections and the router gate are read every step (no expert-selection masking) — those terms are identical to the footprint formula.

    T_theta_moe(B) =
        (L_moe / PP) · (attn_per_device + 3HI_moe · (N_exp/D_exp) · f_touched(B)) · b

`N_exp/D_exp` is the expert population a rank holds (the footprint term; it may be fractional when experts are TP-sliced), and `f_touched` is the fraction of experts the batch reaches. Under the
*uniform routing* assumption each token picks `k_active` distinct
experts out of `N_exp`, so a given expert is missed by one token with probability `1 − k_active/N_exp` and by all `B` tokens with that to
the power `B`:

    f_touched(B) = 1 − (1 − k_active / N_exp) ** B

Every rank holding a slice of a touched expert reads its slice, so the same fraction applies to every rank whatever the TP/EP sharding — the traffic can never exceed the footprint. At small B (B·k\_active ≪ N\_exp) f\_touched ≈ B·k\_active/N\_exp and traffic scales linearly with B; at large B it converges to the footprint formula. `B` is the batch one stage pass processes (the decode microbatch), and may be fractional.

The uniform-routing assumption is a known source of model inaccuracy:
real production routers can exhibit hot-spotting (some experts touched more often than others) and load-balancing-loss anti-correlation; modeling that requires per-deployment routing statistics not generally available.

Returns 0.0 for dense-only models.

### `embedding_bytes(model: LlmModelSpec, partition: PartitionSpec) -> float`

Per-device embedding (and LM head) bytes.

    M_embed = V · H / D_emb · b

The embedding is sharded along the vocab dimension by D\_emb (= TP under all configurations) and replicated across PP stages in the current convention — matches memory\_model.py.

---

## `llm_perf.core.primitives.kv_quantities`

KV cache footprint and per-step traffic — per-device bytes per sequence.

Single primitive that serves *both* quantities simultaneously: the KV memory footprint and the KV per-step traffic share the same formula because, in decode, the full KV cache of every in-flight sequence is read at every step (each sequence's score / value attention traverses its entire context). Unlike the weight-θ split in `weight_quantities.py`, no expert-style masking applies to KV — what's resident is what's read.

    M_kv_per_seq = T_kv_per_seq = (L / PP) · 2 · n_tokens · H_kv · b / (D_kv · SP)

- Factor of 2 covers keys and values.
- (L/PP) is the layer slice owned by this pipeline stage.
- D\_kv (notation.md §1) is the per-device head- or sequence-shard divisor
(excluding SP). Under orthogonal layouts D\_kv = TP for both attention modes (head-shard under TP-attn, sequence-shard across the TP-as-DP-attn group under DP-attn — same byte count). Under co-located layouts D\_kv = max(TP, EP) (sequence-shard across the entire replica's GPUs).
- SP further shards the sequence dimension on top of D\_kv.

Callers multiply by B sequences / B\_prefill as needed: memory uses B × kv\_bytes\_per\_seq for residency; traffic uses B × kv\_bytes\_per\_seq for the per-step batched read.

### `kv_bytes_cluster(model: LlmModelSpec, n_tokens: int) -> float`

Cluster-aggregate KV-cache bytes of one sequence of `n_tokens` (prefill.md §6.2): the whole cache, before any parallelism shards it.

- GQA / MHA: keys and values, `2 · L · n_tokens · H_kv · b`.
- MLA: one head-shared latent per token per layer.

This is the volume a prefill-to-decode handoff moves and the volume a cache tier stores for a reusable prompt.

### `kv_bytes_per_seq(model: LlmModelSpec, partition: PartitionSpec, framework: FrameworkSpec, n_tokens: int) -> float`

Per-device KV bytes for a single sequence of length n\_tokens.

Branches on attention variant and mode (`attention.md §3.4 / §3.6`):
- GQA / MHA: `2 · H_kv · b` per token per layer, divided by D\_kv
(head-shard under TP-attn, sequence-shard under DP-attn — same byte count).
- MLA + DP-attn: head-shared latent `(d_c + d_qk_rope) · b` per token
per layer, divided by D\_kv (sequence-shard across the DP-attn group — same B/G\_TP semantics as GQA).
- MLA + TP-attn: latent is replicated on every rank (not
head-structured). No D\_kv divisor — only SP still applies.

---

## `llm_perf.core.primitives.linear_flops`

Per-token linear FLOPs primitive — projections + FFN + MoE router.

Returns the per-device FLOPs attributable to the *linear* (attention-free)
portion of one decoded token summed across all layers:

    F_linear_per_token =
        (L_dense/PP) · [F_attn_proj + 6HI_dense/TP]
      + (L_moe  /PP) · [F_attn_proj + 6HkI_moe/D_exp + 2HN_exp]

where `F_attn_proj` is the attention projection FLOPs per layer per
device, branching on attention variant:
- GQA / MHA: `(4H² + 4HH_kv) / D_attn` — TP under TP-attn,
unsharded under DP-attn (D\_attn from notation.md §1).
- MLA: variant-specific projection FLOPs from
`mla_flops.mla_proj_flops_per_layer_per_device`, which handles the shardable (W\_UQ / W\_UK / W\_UV / W\_O) vs replicated (W\_DQ / W\_DKV) split under TP-attn and the materialized vs absorbed mode dispatch.

Other terms unchanged:
- 6HI\_dense/TP — dense FFN (gate, up, down), I\_dense wide, always
TP-sharded (no EP axis to overlap on dense).
- 6HkI\_moe/D\_exp — MoE FFN: k active experts per token, sharded by D\_exp
(= TP*EP under orthogonal tp\_ep\_layout, EP under co-located).
- 2HN\_exp — MoE router gate GEMM (H → N\_exp), unsharded
(see decode.md §3.4).

Attention score / value FLOPs are NOT in this primitive because the shape differs by phase: decode sees 4·S·H/(D\_kv·SP) per token (fixed S), prefill sees 4·S²·H/(D\_kv·SP) per *pass*. Callers add phase-specific attention inline (using `mla_score_value_flops_per_layer_per_device` for MLA models).

For prefill, `linear_flops_per_token * S_input` gives the full linear contribution across the prefill pass.

### `attn_proj_flops_per_token(model: LlmModelSpec, partition: PartitionSpec, framework: FrameworkSpec) -> float`

Per-device per-token attention-projection FLOPs (Q/K/V/O), all layers.

Returns the per-token cost *as seen by one rank*: under TP-attn the head-sharded share (per-layer divided by D\_attn = TP), under DP-attn the full per-token projection cost (D\_attn = 1, weights replicated). Per-step composition in `decode_model.compute_flops` applies the correct per-rank batch divisor — TP-attn ranks see all B tokens (so F\_step = B · F\_attn\_proj\_per\_token), DP-attn ranks see only B/G\_TP tokens for the attention block (so F\_step = (B/G\_TP) · F\_attn\_proj\_per\_token).

### `ffn_flops_per_token(model: LlmModelSpec, partition: PartitionSpec, framework: FrameworkSpec) -> float`

Per-device per-token FFN + MoE-router FLOPs, all layers.

Returns dense-FFN + MoE-routed-FFN + MoE-router-gate compute, summed across L layers and divided by per-component sharding factors (D\_exp\_dense = TP, D\_exp\_moe per the §3.2 layout). Independent of `attention_mode` — FFN per-token cost scales the same way under TP-attn and DP-attn at the per-step level (each rank processes all B tokens through its expert / FFN shard).

### `moe_expert_flops_per_token(model: LlmModelSpec, partition: PartitionSpec, framework: FrameworkSpec) -> float`

Per-device per-token expert-FFN FLOPs alone, all MoE layers.

The slice of `ffn_flops_per_token` that scales with the *whole* replica batch: tokens from every device group are routed across all expert shards, so each shard sees `k_active / D_exp` of every token. The remainder of `ffn_flops_per_token` (dense FFN and the router gate) is computed on a token's home group and scales with the group batch. Returns 0.0 for dense-only models.

### `linear_flops_per_token(model: LlmModelSpec, partition: PartitionSpec, framework: FrameworkSpec) -> float`

Per-device, per-token linear FLOPs summed across all layers.

Diagnostic surface: sum of attention-projection + FFN per-token costs as seen by one rank. For per-step cost under DP-attention use `decode_model.compute_flops` which applies the correct per-rank batch divisor (B / G\_TP) to the attention portion (the per-token formula here treats DP-attn as "full per-token cost replicated on all ranks" via D\_attn = 1; the per-step composition fixes the batch sharding).

`framework` selects MLA mode (`mla_mode`) and the attention dispatch pattern (`attention_mode`, `tp_ep_layout`) consumed by the sharding- factor helpers. `attention_mode` / `tp_ep_layout` are ignored for GQA / MHA models; `mla_mode` is ignored for non-MLA models.

---

## `llm_perf.core.primitives.mla_flops`

Multi-head Latent Attention (MLA) FLOP primitives.

Splits MLA's per-layer attention compute into two phase-independent pieces
matching the breakdown in `attention.md §3.7`:

    F_attn,MLA(S) = F_proj_MLA + F_score_value_MLA(S)

where `F_proj_MLA` is the S-independent projection cost (down-projections, up-projections, output projection) and `F_score_value_MLA(S)` is the S-scaling score / value compute. Both branch on the execution mode set
on the tuner (`tuner.mla_mode`):

    "materialized" — reconstruct per-head K, V from the latent each step,
                     run standard MHA over (n_q heads, d_qk_nope / d_v).
    "absorbed"     — fold W_UK / W_UV into Q / O at compile time, run
                     attention entirely in the d_c-dimensional latent space.

The materialized form pays a fixed per-step W\_UK / W\_UV reconstruction cost on the new token; the absorbed form skips that but pays a larger per-past-token score / value cost (in d\_c space rather than d\_qk\_nope / d\_v). Production frameworks default to absorbed at S ≳ 1K.

Sharding follows the same convention as GQA / MHA in
`primitives/linear_flops.py` and `decode_model.compute_flops`:
- Projection FLOPs are sharded by `D_attn` (TP under TP-attn,
1 under DP-attn — replicated weights, but the per-token cost is matched by the corresponding `B / TP` user split downstream).
- Score / value FLOPs are sharded by `D_kv` (TP under both modes,
representing head-shard and user-shard respectively — same byte count).

Callers compose `mla_proj_flops_per_layer_per_device` into the linear-flops primitive and `mla_score_value_flops_per_layer_per_device` into the attention-FLOPs term of `decode_model.compute_flops` / `prefill_model.compute_prefill_flops`.

### `mla_proj_flops_per_layer_per_device(model: LlmModelSpec, partition: PartitionSpec, framework: FrameworkSpec, mla_mode: str) -> float`

Per-device per-layer MLA projection FLOPs (S-independent).

Returns the projection FLOPs that replace the GQA / MHA `(4H² + 4HH_kv) / D_attn` term in `linear_flops_per_token`. Includes W\_DQ, W\_UQ, W\_DKV, the output projection (W\_O materialized or absorbed-W\_O), and — in materialized mode only — the per-step W\_UK / W\_UV reconstruction on the new token.

### `mla_score_value_flops_per_layer_per_device(model: LlmModelSpec, partition: PartitionSpec, framework: FrameworkSpec, S: float, mla_mode: str) -> float`

Per-device per-layer MLA score / value FLOPs (S-scaling).

Returns the score + value compute that replaces the GQA / MHA `4 · S · H / D_kv` term in `decode_model.compute_flops` (and the `4 · S² · H / D_kv` analog in `prefill_model.compute_prefill_flops`, which the caller obtains by passing `S = S_input` and multiplying by `S_input` for the per-pass form).

---

## `llm_perf.core.primitives.collective_cost`

Collective algorithm cost primitives — α-β cost model.

Source of truth for every formula here is the modeling series under
``documentation/modeling/``:

- ``01\_collective\_algorithms.md`` — α-β model and per-primitive derivations.
- ``02\_topology\_mapping.md`` — single-tier specializations (star, torus, mesh).
- ``03\_hierarchical\_topologies.md`` — multi-tier Clos / fat-tree composition.
- ``04\_in\_network\_collectives.md`` — switch-ALU / multicast / scatter-gather INC.
- ``05\_contention\_and\_congestion.md`` — η contention coefficients.

Conventions
-----------
``M`` is always the per-rank payload measured in bytes. The exact meaning per
primitive matches the doc convention:

- BC / Reduce: full broadcast / reduce payload.
- AR: full vector each rank starts and ends with.
- AG: total gathered output per rank (``N · shard``).
- RS: total pre-scatter input per rank (``N · shard``).
- A2A: total per-rank A2A payload (``N`` chunks of size ``M/N`` each).
- P2P: single-message payload.

``G`` is the group rank count, ``alpha\_s`` is per-hop latency in seconds, and ``bw\_Bps`` is the per-rank port bandwidth (one direction) in bytes per second — on a star each rank has one such port; on a torus / mesh it is the per-link bandwidth. Each function returns seconds. Functions return ``0.0`` on no-op group sizes (``G ≤ 1``) so callers can skip shape-level guards, and raise ``ValueError`` on a non-positive bandwidth (a dead link is not a free one).

Pipelined variants
------------------
For schedules that benefit from chunking (BC, Reduce, DBT AR), the
``pipelined`` flag selects between:

- ``pipelined=False`` (default): the literal P=1 schedule with full ``M`` on
every active link per step. Matches the closed-form derivations in ``01\_collective\_algorithms.md`` §3-§5.
- ``pipelined=True``: the asymptotic P→P* limit where the ``⌈log₂ G⌉`` (or
``G-1``) BW multiplier collapses (Appendix C). Hides the ``O(√M)`` correction. The floor is ``M/BW`` for BC / Reduce and ``2·M/BW`` for DBT AR on a single-port fabric (see ``tree\_all\_reduce``).

Ring AR / AG / RS are *intrinsically* pipelined — their closed forms already are the asymptote, so no flag is exposed.

### `apply_eta(alpha_s: float, bw_Bps: float, eta_alpha: float=1.0, eta_beta: float=1.0) -> tuple[float, float]`

Pre-discount (α, BW) by contention coefficients — ``05\_contention\_and\_congestion.md`` §4.

    α_eff  = α · η_α       (η_α ≥ 1 inflates latency)
    BW_eff = BW · η_β      (η_β ∈ (0, 1] deflates bandwidth)

The realistic primitive cost is then evaluated by passing ``(alpha\_eff, bw\_eff)`` into any of the cost functions below — no algorithm-level plumbing needed (per the per-tier η framing in §4.2).

Calibration profile (``05\_contention\_and\_congestion.md`` §4.1):

- Crossbar (NVLink + NVSwitch, no SHARP):    η\_α = 1.00, η\_β = 0.80
- NVLS (NVLink SHARP, in-network reduction): η\_α = 1.00, η\_β = 0.52
- Torus (off-prefix + concurrent groups):    η\_α = 1.20, η\_β = 0.60
- Fat-tree / Clos at oversubscription s:     η\_β = min(η\_β\_hw, 1/s)   (§4.2)

### `p2p_hop(M: float, alpha_s: float, bw_Bps: float) -> float`

Single send / recv — ``01\_collective\_algorithms.md`` §8.

    t = α + M / BW

The degenerate N=2 case of any collective. Used per inter-stage hop in pipeline parallelism.

### `ring_broadcast(M: float, G: int, alpha_s: float, bw_Bps: float, pipelined: bool=False) -> float`

Ring (chain) broadcast — ``01\_collective\_algorithms.md`` §3.1.

    non-pipelined (P=1):       t = (G-1)·α + (G-1) · M / BW
    pipelined (P→P*):          t ≈ (G-1)·α + M / BW

The pipelined form hides an ``O(√M)`` correction ``2·sqrt((G-2)·α·M/BW)`` (Appendix C); it vanishes relative to ``M/BW`` as ``M → ∞``.

### `tree_broadcast(M: float, G: int, alpha_s: float, bw_Bps: float, pipelined: bool=False) -> float`

Binomial tree broadcast — ``01\_collective\_algorithms.md`` §3.2.

    non-pipelined (P=1):    t = ⌈log₂ G⌉·α + ⌈log₂ G⌉ · M / BW
    pipelined (P→P*):       t ≈ ⌈log₂ G⌉·α + M / BW

Binomial-tree BC is the MPI-library default (MPICH / Open MPI). NCCL restricts ``ncclBroadcast`` to its ring algorithm. On a single-port fabric the pipelined ``M/BW`` floor is that of the two-tree [SST09] schedule (each tree carries ``M/2``, so a rank's port moves ``M`` per direction); a pure binomial root would push ``⌈log₂ G⌉·M`` through its port and a lone binary tree ``2M``.

### `inc_broadcast(M: float, alpha_s: float, bw_Bps: float) -> float`

In-network multicast broadcast — ``01\_collective\_algorithms.md`` §3.3, ``04\_in\_network\_collectives.md`` §1.2.

    t = 2·α_switch + M / BW

Source rank pushes M upstream to the switch (1 hop); switch crossbar multicasts to all destination ports concurrently (1 hop). Two hops independent of N. ``alpha\_s`` is the summed per-tier switch cut-through α on one direction (single-switch star: ``α\_switch``; k-tier scale-out aggregation tree: ``Σ α\_tier``).

### `ring_reduce(M: float, G: int, alpha_s: float, bw_Bps: float, pipelined: bool=False) -> float`

Ring (chain) reduce — ``01\_collective\_algorithms.md`` §4.1.

    non-pipelined (P=1):    t = (G-1)·α + (G-1) · M / BW
    pipelined (P→P*):       t ≈ (G-1)·α + M / BW

Time-reverse of ring BC with on-chip add at every hop.

### `tree_reduce(M: float, G: int, alpha_s: float, bw_Bps: float, pipelined: bool=False) -> float`

Binomial tree reduce — ``01\_collective\_algorithms.md`` §4.2.

    non-pipelined (P=1):    t = ⌈log₂ G⌉·α + ⌈log₂ G⌉ · M / BW
    pipelined (P→P*):       t ≈ ⌈log₂ G⌉·α + M / BW

Tree dominates ring for standalone Reduce on the pure α-β model (same BW floor at log-depth α). MPI libraries ship the tree; NCCL restricts ``ncclReduce`` to its ring algorithm.

### `inc_reduce(M: float, alpha_s: float, bw_Bps: float) -> float`

In-network reduce (switch ALU) — ``01\_collective\_algorithms.md`` §4.3, ``04\_in\_network\_collectives.md`` §1.1.

    t = 2·α_switch + M / BW

Each rank pushes V\_i upstream (1 hop); switch ALU sums on-chip and forwards the single reduced result to the root (1 hop).

### `ring_all_reduce(M: float, G: int, alpha_s: float, bw_Bps: float) -> float`

Ring (Patarasuk-Yuan) all-reduce — ``01\_collective\_algorithms.md`` §5.1.

    t = 2(G-1)·α + 2·(G-1)/G · M / BW

Already the pipelined asymptote (P=N) by construction — the M/N chunked payload bakes pipelining into the derivation, so no ``pipelined`` flag is needed. BW coefficient hits the AR floor of 2 as G → ∞.

### `tree_all_reduce(M: float, G: int, alpha_s: float, bw_Bps: float, pipelined: bool=False, multi_port: bool=False) -> float`

Double binary tree (DBT) all-reduce — ``01\_collective\_algorithms.md`` §5.2.

Single-port fabric (default — ``bw\_Bps`` is the per-rank port bandwidth, as on a star; this is the series convention and the form behind the
N=512 anchors in ``04\_in\_network\_collectives.md`` §3.1):

    non-pipelined (P=1):    t = 2⌈log₂ G⌉·α + 2⌈log₂ G⌉ · M / BW
    pipelined (P→P*):       t ≈ 2⌈log₂ G⌉·α + 2 · M / BW

Each rank is interior in one tree and a leaf in the other, so it receives ``M/2`` from each of its two children during reduce and ``M/2`` from each of its two parents during broadcast: ``2M`` in (and ``2M`` out) through one port. Pipelining removes waiting, not bytes, so the BW coefficient floors at 2 — the same dual-touch ``BW\_eff = BW/2`` that ring AR pays, and the factor INC AR removes.

Multi-port fabric (``multi\_port=True`` — every tree edge is its own full-BW link, up to 4 per rank: parent and two children in one tree, parent in the other): both BW terms halve,

    non-pipelined (P=1):    t = 2⌈log₂ G⌉·α + ⌈log₂ G⌉ · M / BW
    pipelined (P→P*):       t ≈ 2⌈log₂ G⌉·α + M / BW

which is the per-link accounting of the §5.2 walk-through (reduce and broadcast passes run back-to-back).

NCCL's non-ring AR path. The pipelined forms are *lower bounds* in practice — real DBT carries ``c\_real ≥ 2`` (single-port) above ``M/BW`` because of finite pipeline depth and per-step overhead, which is what lets ring re-take the crown at large M (§5.3 practice caveat).

The byte count assumes a rank with two children, i.e. ``G ≥ 4``; for ``G = 2`` or ``3`` the closed form overcounts (two ranks need only one exchange, ``α + M/BW``).

### `rabenseifner_all_reduce(M: float, G: int, alpha_s: float, bw_Bps: float) -> float`

Rabenseifner halving-doubling AR — ``01\_collective\_algorithms.md`` App. B.2.

    t = 2⌈log₂ G⌉·α + 2·(G-1)/G · M / BW

Two complementary hypercube passes (recursive halving for RS, recursive doubling for AG) with chunk-exponential payloads. Matches ring's BW floor with log-depth α — but **not shipped by NCCL** for AR; reference only. Power-of-2 G is required for clean partner pairings.

### `inc_all_reduce(M: float, alpha_s: float, bw_Bps: float) -> float`

In-network all-reduce (switch ALU + multicast) — ``04\_in\_network\_collectives.md`` §1.4 / §2.

    t = 2·α_switch + M / BW

The unique INC primitive that lifts BW\_eff from BW/2 (software dual-touch — ring's ``2(G-1)/G`` and single-port DBT's ``2``) to BW: switch ALU reduces the N inputs in-fabric and the multicast xbar replicates the single result to all N output ports, on opposite directions of each full-duplex endpoint link. ``alpha\_s`` is the summed per-tier switch α on one direction (single switch: ``α\_switch``; k-tier aggregation tree: ``Σ α\_tier``); the leading factor of 2 covers the reduce-up + multicast-down round trip.

### `ring_all_gather(M: float, G: int, alpha_s: float, bw_Bps: float) -> float`

Ring all-gather — ``01\_collective\_algorithms.md`` §6.

    t = (G-1)·α + (G-1)/G · M / BW

M is the per-rank gathered output (= G · per-rank shard). Already pipelined (P=N) like ring AR. AG is exactly Phase 2 of ring AR with no reduction; RS is Phase 1 (same formula).

### `ring_reduce_scatter(M: float, G: int, alpha_s: float, bw_Bps: float) -> float`

Ring reduce-scatter — ``01\_collective\_algorithms.md`` §6.

    t = (G-1)·α + (G-1)/G · M / BW

M is the per-rank pre-scatter input (= G · per-rank output shard). RS is the time-reverse of AG; same wiring, accumulating partial sum on every forward instead of overwriting.

### `recursive_doubling_all_gather(M: float, G: int, alpha_s: float, bw_Bps: float) -> float`

Recursive-doubling AG — ``01\_collective\_algorithms.md`` App. B.4.

    t = ⌈log₂ G⌉·α + (G-1)/G · M / BW

Hypercube butterfly with doubling chunk size. MPI menus ship this alongside ring; NCCL does not. Power-of-2 G required for clean pairings.

### `recursive_halving_reduce_scatter(M: float, G: int, alpha_s: float, bw_Bps: float) -> float`

Recursive-halving RS — ``01\_collective\_algorithms.md`` App. B.4.

    t = ⌈log₂ G⌉·α + (G-1)/G · M / BW

Time-reverse of recursive-doubling AG: hypercube butterfly with halving chunk size, accumulating on receive. Power-of-2 G required.

### `pat_all_gather(M: float, G: int, alpha_s: float, bw_Bps: float) -> float`

Parallel Aggregated Trees AG — ``01\_collective\_algorithms.md`` App. A, referenced as the NCCL 2.23+ scale-out path (1 rank per node).

    t = ⌈log₂ G⌉·α + (G-1)/G · M / BW

Small-operation regime of PAT (Jeaugey, arXiv:2506.20252): a reversed-dimension Bruck schedule — far dimension first, payload aggregating per transfer — so the α-β cost equals recursive-doubling AG. PAT needs only a logarithmic amount of internal buffering; once the per-rank size exceeds the buffer, the schedule degrades toward a linear number of full-buffer transfers (the α count grows; not modelled here). NCCL 2.23 ships it inter-node at 1 rank per node.

### `pat_reduce_scatter(M: float, G: int, alpha_s: float, bw_Bps: float) -> float`

PAT reduce-scatter — ``01\_collective\_algorithms.md`` App. A.

    t = ⌈log₂ G⌉·α + (G-1)/G · M / BW

Time-reverse of PAT AG (near dimension first); same small-operation regime and the same 1-rank-per-node shipping restriction.

### `inc_all_gather(M: float, G: int, alpha_s: float, bw_Bps: float) -> float`

In-network all-gather — ``04\_in\_network\_collectives.md`` §1.4.

    t = 2·α_switch + (G-1)/G · M / BW

α-only collapse: software ring AG already runs at BW\_eff = BW on a full-duplex star, so the switch multicast lifts only the α side from ``(G-1)·α`` to ``2·α\_switch``. ``alpha\_s`` is the summed per-tier switch α (one direction).

### `inc_reduce_scatter(M: float, G: int, alpha_s: float, bw_Bps: float) -> float`

In-network reduce-scatter — ``04\_in\_network\_collectives.md`` §1.4.

    t = 2·α_switch + (G-1)/G · M / BW

Mirror of inc\_all\_gather: α-only collapse, BW unchanged from software ring RS on a full-duplex crossbar.

### `pairwise_a2a(M: float, G: int, alpha_s: float, bw_Bps: float) -> float`

Pairwise direct-send A2A — ``01\_collective\_algorithms.md`` §7.2.

    t = (G-1)·α + (G-1)/G · M / BW

NCCL default on full-bisection fabrics (fat-tree, Clos, NVSwitch). M is the per-rank total A2A payload (G chunks of size M/G each). Workloads that run A2A back-to-back (MoE Dispatch + Combine) double both terms externally — see ``01\_collective\_algorithms.md`` §7.2 final paragraph.

### `ring_relay_a2a(M: float, G: int, alpha_s: float, bw_Bps: float) -> float`

Ring relay A2A on a bidirectional ring — ``01\_collective\_algorithms.md`` §7.1 (the ``k = 1`` case of ``02\_topology\_mapping.md`` §3.6).

    t ≈ ⌊G/2⌋·α + (G/8) · M / BW

Shortest-arc forwarding through intermediate ranks. ``bw\_Bps`` is the bandwidth of each ring link (a rank has two). The ring's bisection is two links per direction and ``G·M/4`` bytes cross it each way, so the busiest link carries ``G·M/8`` — the BW term grows with ``G`` and exceeds pairwise direct-send's single-port ``(G-1)/G`` from ``G = 8`` on. The α term is the longest path (``⌊G/2⌋`` hops) with chunks streamed back-to-back.

``G/8`` assumes antipodal chunks are split across both directions (even ``G``); a fixed rightward tie-break gives ``(G+2)/8`` — the lockstep ``3α + (3/4)·M/BW`` of the §7.1 trace at ``G = 4``. Evaluated as ``torus\_a2a(M, (G,), ...)``, so ``G = 2`` (one link, coefficient 1/2) follows the size-2 rule documented there.

### `bruck_a2a(M: float, G: int, alpha_s: float, bw_Bps: float) -> float`

Bruck log-round A2A — ``01\_collective\_algorithms.md`` App. B.5.

    t = ⌈log₂ G⌉·α + ⌈log₂ G⌉/2 · M / BW

Log-depth latency at the cost of an ``⌈log₂ G⌉/2`` BW coefficient — every byte crosses ``log₂ G / 2`` link-hops on average (chunks bounce around the hypercube before reaching their final destination). Reference only; NCCL / RCCL do not ship Bruck.

### `inc_a2a(M: float, G: int, alpha_s: float, bw_Bps: float) -> float`

Hardware A2A (crossbar scatter-gather) — ``04\_in\_network\_collectives.md`` §1.3.

    t = 2·α_switch + (G-1)/G · M / BW

A what-if primitive: no vendor documents a hardware all-to-all collective (Tomahawk Ultra names AllReduce, Broadcast and AllGather; Quantum-X800's SHARPv4 lists an undefined "ScatterGather"). It would collapse software A2A's (G-1) endpoint-driven scheduling rounds into one switch-driven transaction. Per-rank wire-side bytes are unchanged from software pairwise (the switch routes verbatim — no aggregation, no replication), so the BW term stays at the per-port bound.

``alpha\_s`` is ``α\_switch``; the factor of 2 is the same one-hop-in, one-hop-out traversal every INC primitive in this module pays.

No hardware A2A is documented on NVSwitch (NVL72) or Quantum-X800; software pairwise is the only documented A2A path on those fabrics.

### `torus_all_reduce(M: float, dims: Sequence[int], alpha_s: float, bw_Bps: float) -> float`

k-D torus dim-decomposed ring AR — ``02\_topology\_mapping.md`` §3.4.

    t = 2·Σ(d_i - 1)·α + 2·(N-1)/N · M / BW,    N = ∏ d_i

Latency telescopes from flat ring's ``2(N-1)·α`` to ``2·Σ(d\_i-1)·α``; BW coefficient stays at ring's ``2(N-1)/N`` floor (per-dim chunks telescope cleanly across phases). Reduces to ring AR for ``dims=(N,)``.

Costs the ideal dim-aligned schedule. Worst-case group misalignment is a dispatch-layer concern (see ``02\_topology\_mapping.md`` §5.2, item 2).

### `torus_all_gather(M: float, dims: Sequence[int], alpha_s: float, bw_Bps: float) -> float`

k-D torus dim-decomposed ring AG — ``02\_topology\_mapping.md`` §3.5.

    t = Σ(d_i - 1)·α + (N - 1)/N · M / BW,    N = ∏ d_i

Half of torus AR. M is the per-rank gathered output.

### `torus_reduce_scatter(M: float, dims: Sequence[int], alpha_s: float, bw_Bps: float) -> float`

k-D torus dim-decomposed ring RS — ``02\_topology\_mapping.md`` §3.5.

    t = Σ(d_i - 1)·α + (N - 1)/N · M / BW,    N = ∏ d_i

Time-reverse of torus AG. M is the per-rank pre-scatter input.

### `torus_a2a(M: float, dims: Sequence[int], alpha_s: float, bw_Bps: float, wraparound: bool=True) -> float`

Bisection-bound A2A on k-D torus / k-D mesh — ``02\_topology\_mapping.md`` §3.6 (torus) / §4.2 (mesh).

Wraparound torus (``wraparound=True``):

    diam = Σ ⌊d_i / 2⌋
    t ≈ diam·α + (d_max / 8) · M / BW

The min-bisection severs ``2N/d\_max`` links; cross-cut traffic is ``N·M/4`` per direction, so each cut link carries ``d\_max·M/8`` one way.

k-D mesh (``wraparound=False``):

    diam = Σ (d_i - 1)
    t ≈ diam·α + (d_max / 4) · M / BW

Wraparound edges missing → bisection cut halves → cut links carry 2× the bytes. Diameter also changes from ``Σ⌊d\_i/2⌋`` (torus) to ``Σ(d\_i-1)`` (open-line mesh).

Only ``d\_max`` (not N) scales the BW term: asymmetric layouts pay disproportionately. With ``wraparound=True`` the coefficient is ``d\_max/8`` — ``1`` at ``8×8×8`` (matching star pairwise's ``≈ M/BW``), ``2`` at ``16³``, ``0.5`` at ``4³``.

A dim of size 2 has no distinct wraparound link (it coincides with the direct link), so cutting across it severs only ``N/2`` links and that dim contributes ``d\_i/4 = 1/2`` instead of ``d\_i/8``. The BW coefficient is therefore the worst cut over dims, ``max\_i c\_i`` with ``c\_i = d\_i/8`` for ``d\_i ≥ 3`` and ``1/2`` for ``d\_i = 2`` — which equals ``d\_max/8`` whenever ``d\_max ≥ 4``. For odd ``d`` the exact per-dim value is ``(d²-1)/(8d)`` (``(d²-1)/(4d)`` on the mesh); the ``≈`` covers it.

M is the per-rank total A2A payload (G chunks of size M/G).

### `torus_broadcast(M: float, dims: Sequence[int], alpha_s: float, bw_Bps: float, wraparound: bool=True, pipelined: bool=False) -> float`

k-D torus / k-D mesh broadcast — ``02\_topology\_mapping.md`` §3.2 / §4.2.

Wraparound torus (bidirectional ring per dim):

    non-pipelined:  t = Σ⌊d_i/2⌋·α + Σ⌊d_i/2⌋ · M / BW
    pipelined:      t ≈ Σ⌊d_i/2⌋·α + M / BW

k-D mesh (open-line per dim, unidirectional from end):

    non-pipelined:  t = Σ(d_i-1)·α + Σ(d_i-1) · M / BW
    pipelined:      t ≈ Σ(d_i-1)·α + M / BW

Pipelined form hits the ``M/BW`` ceiling (``02\_topology\_mapping.md`` §5.1 BC row).

### `torus_reduce(M: float, dims: Sequence[int], alpha_s: float, bw_Bps: float, wraparound: bool=True, pipelined: bool=False) -> float`

k-D torus / k-D mesh reduce — time-reverse of torus\_broadcast, ``02\_topology\_mapping.md`` §3.3 / §4.2.

Same α-β cost as torus\_broadcast (sink instead of source).

### `hierarchical_all_reduce(t_rs_inner: float, t_ar_outer: float, t_ag_inner: float) -> float`

Hierarchical AR composition — ``03\_hierarchical\_topologies.md`` §2.1.

    AR ≡ inner RS → outer sub-AR → inner AG

The outer sub-AR ships the *telescoped* payload ``M·L/N`` (M = per-rank full vector, L = number of outer groups, N = total ranks). Caller computes each phase cost using the per-tier (α, BW) and an algorithm of choice (ring, DBT, INC, torus dim-decomp, …).

For convenience helpers, see ``hierarchical\_all\_reduce\_ring\_ring``.

### `hierarchical_all_reduce_ring_ring(M: float, N: int, L: int, alpha_inner: float, bw_inner: float, alpha_outer: float, bw_outer: float) -> float`

Hierarchical AR with ring at both tiers — ``03\_hierarchical\_topologies.md`` §2.1 (NVL72+IB SuperPOD case study).

L outer groups × (N/L) inner ranks each. M is the per-rank full vector. Outer phase ships the telescoped payload ``M·L/N`` per the §2.1 derivation. Reduces to flat ring AR when ``L = 1`` or ``N = L``.

    α:  2(N/L - 1)·α_inner + 2(L - 1)·α_outer
    BW: 2(N - L)/N · M/BW_inner + 2(L - 1)/N · M/BW_outer

``L`` must divide ``N``.

### `hierarchical_all_gather(t_ag_outer: float, t_ag_inner: float) -> float`

Hierarchical AG composition — ``03\_hierarchical\_topologies.md`` §2.1 summary table.

    AG ≡ outer AG → inner AG

The AG half of hierarchical AR. The outer phase gathers across the ``L`` groups first, while each rank still holds only its ``M/N`` shard, so the slow outer tier carries the telescoped payload ``M·L/N``; the inner phase
then gathers the full ``M`` on the fast tier:

    t = t_AG,outer(L, M·L/N) + t_AG,inner(N/L, M)

(second argument = per-rank gathered output of that phase). Running the phases the other way round is still correct but ships ``(L-1)/L · M`` over the outer tier.

### `hierarchical_reduce_scatter(t_rs_inner: float, t_rs_outer: float) -> float`

Hierarchical RS composition — ``03\_hierarchical\_topologies.md`` §2.1.

    RS ≡ inner RS → outer RS

The RS half of hierarchical AR (time-reverse of hierarchical AG): the inner phase shrinks each rank's holding from ``M`` to ``M·L/N`` on the
fast tier before anything crosses the outer tier:

    t = t_RS,inner(N/L, M) + t_RS,outer(L, M·L/N)

### `hierarchical_all_gather_ring_ring(M: float, N: int, L: int, alpha_inner: float, bw_inner: float, alpha_outer: float, bw_outer: float) -> float`

Hierarchical AG with ring at both tiers — ``03\_hierarchical\_topologies.md`` §2.1.

    α:  (N/L - 1)·α_inner + (L - 1)·α_outer
    BW: (N - L)/N · M/BW_inner + (L - 1)/N · M/BW_outer

M is the per-rank gathered output. Exactly half of ``hierarchical\_all\_reduce\_ring\_ring``; reduces to flat ring AG when ``L = 1`` or ``N = L``. ``L`` must divide ``N``.

### `hierarchical_reduce_scatter_ring_ring(M: float, N: int, L: int, alpha_inner: float, bw_inner: float, alpha_outer: float, bw_outer: float) -> float`

Hierarchical RS with ring at both tiers — ``03\_hierarchical\_topologies.md`` §2.1.

Time-reverse of ``hierarchical\_all\_gather\_ring\_ring``; same cost. M is the per-rank pre-scatter input. ``L`` must divide ``N``.

### `realistic_cost(cost_fn, *args, eta_alpha: float=1.0, eta_beta: float=1.0, **kwargs) -> float`

Apply η contention coefficients to a single-tier primitive in this module — ``05\_contention\_and\_congestion.md`` §4.

Binds the call to ``cost\_fn``'s signature, scales its ``alpha\_s`` and ``bw\_Bps`` arguments through ``apply\_eta`` and calls it. Flags such as ``pipelined`` or ``wraparound`` may be passed positionally or by keyword. Functions without ``alpha\_s`` / ``bw\_Bps`` parameters (the hierarchical helpers, which take per-tier values) raise ``TypeError`` — pre-discount each tier with ``apply\_eta`` instead (§4.2).

Example:

    from core.collective_cost import ring_all_reduce, realistic_cost
    # NVLink+NVSwitch crossbar profile (η_α=1.0, η_β=0.80)
    t = realistic_cost(ring_all_reduce, M, G, alpha, bw,
                       eta_alpha=1.0, eta_beta=0.80)

### `canonical_anchors() -> list[tuple[str, float, float]]`

Evaluate the canonical N=512 anchors; returns (label, value μs, target μs).

---

## `llm_perf.core.primitives.dispatch`

Topology-aware collective cost dispatcher — collectives/00\_summary.md §4–§7.

Walks a fabric chain innermost-first, picking each tier's contribution based on its `.topology` discriminator. Crossbar tiers flatten to a single $(\alpha_\mathrm{sum}, \mathrm{BW}_\min)$ pair consumed by the star primitives in `collective_cost.py`. Torus tiers route into the dim-decomposed primitives. Tiers are traversed innermost-first; a collective is said to "cross" a tier once cumulative reach $\prod \mathrm{ports}$ is below the group size. `G` is never clamped — primitives consume the caller's `G` regardless of cumulative reach.

**Bit-identity guarantee.** For a tier chain where every crossed tier is a
`CrossbarTier`, this function returns *exactly* the same float as the pattern `ring_or_tree(M, G, *span_tiers(tiers, G)[:2])`. The crossbar branch below reproduces that flatten step verbatim — no rounding, no intermediate arithmetic reshuffling.

Genuinely mixed crossbar/torus chains fall back to the crossbar-flatten bound with a `UserWarning` (explicit hybrid compositions are a follow-up).

### `cost_collective(tiers: List[TierSpec], op: str, M: float, G: int, algorithm: str='ring', torus_algorithm: str='ring', inc_enabled: bool=True, align_policy: str='prefix', stride: int=1) -> float`

Cost a collective of group size `G` bytes `M` over `tiers`, in seconds.

Args:
tiers: innermost-first tier list (typically `system.get_tier_chain(coll)`). op: one of "all\_reduce", "all\_gather", "moe\_a2a", "p2p". M: per-rank message size in bytes. G: collective group size (number of ranks). algorithm: "ring", "tree" or "tree\_pipelined" — software schedule on
    the crossbar path for AR ("ring" only for MoE-A2A). Ignored for
    `all_gather` (always ring) and `p2p` (single hop), and on torus
    tiers (see `torus_algorithm`). "inc" asks for the in-network path
    outright, with the cheapest software schedule on any phase whose
    tier has no in-network support.
torus\_algorithm: "ring" (dim-by-dim) or "swing". "swing" raises
    `NotImplementedError` — reserved for a future Swing-AR primitive.
inc\_enabled: when True, in-network collectives (INC) are a candidate
    on every crossbar tier that declares `inc != "none"` (`"hw_a2a"`
    for MoE-A2A), and the cheaper of the INC path and the software
    `algorithm` is returned — a runtime is never forced onto the
    switch path when its software schedule is faster. On a multi-tier
    chain INC is applied per phase of the hierarchical schedule
    (collectives/03_hierarchical_topologies.md §3.1), so a tier
    without it keeps its software phase. When False, software only.
    Inert on torus and on mixed chains.
align\_policy: "prefix" (default) or "greedy" — which sub-torus layouts
    a torus chain admits for a group of `G` ranks (see `_align_to_dims`).
    Group sizes that fail the policy fall back to a flat-ring bound with
    a `UserWarning`. Inert on crossbar chains.
stride: device slots each rank of the group occupies under the nested
    layout (see `partition_layout.axis_stride`). The group physically
    spans `G · stride` devices, so that product — not `G` — decides
    which tiers the collective crosses, and an inner island of `ports`
    devices holds only `ports // stride` of the group's ranks. Example:
    expert parallelism over tensor-parallel groups of 8 on 8-GPU boxes
    has stride 8, so every EP peer sits in another box. Default 1
    (ranks are adjacent devices) reproduces the unstrided cost exactly.
    Inert on torus chains, where an outer axis takes its own dimension.

Returns:
Collective time in seconds. Returns 0.0 for G <= 1 or empty tier list.

### `enumerate_options(tiers: List[TierSpec], op: str, M: float, G: int, inc_enabled: bool=True, align_policy: str='prefix', stride: int=1) -> List[Tuple[str, float]]`

Enumerate admissible (algorithm\_name, cost\_seconds) pairs for a collective.

The post-partition optimizer (`core/collective_algo_opt.py`) consumes this surface to pick the optimal SW algorithm per (phase × collective). For each (op, tier-chain, M, G, inc\_enabled) tuple, returns every algorithm the dispatcher supports for that cell, paired with its α-β cost in seconds.

Algorithm name space:
- Star (single-tier or multi-tier crossbar):
      AR:        "ring", "tree", "tree_pipelined", optionally "inc"
      AG:        "ring", optionally "inc"
      MoE A2A:   "ring" (= pairwise direct-send), optionally "inc"
      P2P:       "p2p" (single hop, no algorithm choice)
- Torus:
      all ops:   "torus-dim-ring"  (single option; INC structurally absent
                                    on torus per `system_spec.TorusTier`)
- Mixed crossbar/torus: same as the dispatcher's flat-fallback path
                          (warns at cost_collective time, not here).

"tree" maps to the literal P=1 binomial tree (BW coefficient = log\_G); "tree\_pipelined" maps to the asymptotic P→P* form (BW coefficient = 1) that NCCL ships at bulk M.

INC eligibility (collectives/04\_in\_network\_collectives.md):
- AR / AG via "inc" needs a crossed tier that declares `inc != "none"`.
    On a multi-tier chain INC is applied per phase of the hierarchical
    schedule (collectives/03_hierarchical_topologies.md §3.1): a tier
    without it keeps its (cheapest) software phase.
- MoE A2A via "inc" needs `inc == "hw_a2a"` (sharp\_class does not
    accelerate A2A).
- When `inc_enabled=False`, no "inc" entries are returned.
"inc" is one more candidate: it is not assumed to be the cheapest.

`align_policy` is forwarded to the torus path so the optimizer prices the same sub-torus layouts the dispatcher will later use. Inert on crossbar. `stride` has the same meaning as in `cost_collective` (device slots per rank of the group); inert on torus.

Returns:
Ordered list of (name, cost\_seconds) pairs. Empty for G <= 1 or empty tier list (no work to enumerate). The optimizer picks `min(.., key=cost)`.

---

## `llm_perf.core.primitives.stage_aggregator`

Per-pipeline-stage communication aggregator.

Local helper extracted from the upstream-synced ``collective\_cost.py``; the upstream module no longer carries llm\_perf-specific glue (every function there is a pure α-β collective primitive). This file owns the TP/SP/EP/PP combination logic — see ``decode.md`` §5.5 for the derivation and ``prefill.md`` §3.2 for the prefill cross-reference.

### `aggregate_per_stage(L: int, L_moe: int, PP: int, n_TP: int, t_TP: float, n_SP: int, t_SP: float, n_EP: int, t_EP: float, t_PP: float) -> float`

Per-pipeline-stage communication time — decode.md §5.5.

    t = (L/PP)·(n_TP·t_TP + n_SP·t_SP)
      + (L_moe/PP)·(n_EP·t_EP)
      + t_PP

TP and SP collectives apply to every layer on the stage. EP collectives apply only to MoE layers (L\_moe/PP per stage). The PP hop is a single inter-stage forward per step.

Prefill cross-references this aggregation — see ``prefill.md`` §3.2 ("Following the same structure as decode.md §5.5").

---
