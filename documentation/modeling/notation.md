# Notation Reference

Shared symbol definitions and architectural conventions for all documents under
`documentation/modeling/`. Each section notes which document first uses or extends
the symbols. Tagged citations (e.g. `[MEGATRON]`, `[FA2]`) resolve to entries in
`references.md`.

---

## 1. Parallelism Architecture
_(→ decode.md)_

All documents in this suite assume a fixed nesting order for parallelism dimensions:

$$
\text{DP} \;\rightarrow\; \text{PP} \;\rightarrow\; \text{EP} \;\rightarrow\; \text{TP} \;\rightarrow\; \text{SP}
$$

This order reflects how model state is partitioned and reused during inference. Each level depends on all outer levels having already determined weight placement, token routing, or tensor partitioning.

| Level | What it partitions | Why this ordering is required |
|-------|--------------------|------------------------------|
| **DP** | Entire model replica | Must wrap all state; inner groups cannot cross DP boundaries. |
| **PP** | Layers | Layer ownership must be decided before experts/tensor shards are assigned. |
| **EP** | Experts | Expert placement must be fixed before tensor sharding splits expert matrices. |
| **TP** | Weight matrices | TP defines weight shards used identically across all SP ranks. |
| **SP** | KV cache sequence dimension | KV is activation state only; must be sharded after all weight placement. |

### DP — outermost (replicated model weights)

DP creates fully independent model replicas for throughput scaling. No weight partitioning happens inside DP groups; all inner dimensions (PP, EP, TP, SP) apply **within** each DP replica.

### PP — inside DP (layers assigned before experts/tensor sharding)

PP determines **which layers live on which devices**. Only after PP is fixed can experts (EP), tensor dimensions (TP), and KV partitions (SP) be assigned. PP stages own their local weights and KV cache.

### EP — inside PP (expert groups belong to specific layers)

EP distributes MoE experts within the layers assigned by PP. Expert weights must be placed (EP) before tensor-parallel shards apply (TP). The expert-parallel degree must satisfy:
$$EP \le N_{\text{exp}}$$
In practice EP usually divides $N_{\text{exp}}$, but the only hard constraint is $EP \le N_{\text{exp}}$.

### TP — inside EP (tensor sharding within a defined expert/layer partition)

TP splits matrices within each expert or dense block. After TP, each rank holds a fraction of $H$ or $H_{kv}$. SP requires all SP ranks to share identical TP-sharded weights, so TP must precede SP.

### SP — innermost (KV sharding after all weights are fixed)

SP shards the **KV cache** (activation state), not model parameters. Only after DP/PP/EP/TP are fixed can the KV sequence dimension be partitioned.

### Production deviations from the strict orthogonal layout

The strict orthogonal nesting above is the canonical mental model and the default convention for §1–§14. Two production-driven deviations modify how individual components are sharded across devices:

1. **Attention parallelism mode swap (TP-attn ↔ DP-attn) [DSV3, SGLANG-DPATTN].** A per-block selector on the attention block only: under DP-attention the attention weights are replicated on every TP rank and tokens are partitioned along the sequence dimension instead of head-sharding. The per-layer attention all-reduce (AR) is replaced by a TP all-gather (AG) under the conventional pattern; under the alternative scatter-direct MoE A2A pattern (`decode.md §5.2`) the AG is skipped entirely and the dispatch operates on per-rank sharded tokens, trading per-rank A2A payload for stricter backend requirements. Dense and MoE blocks remain TP- and EP-sharded in both patterns.

2. **TP+EP co-location.** A layout-level choice: TP and EP map to the *same* physical GPUs of a replica rather than disjoint GPU sets (DeepSeek-V3 / R1 production decode on SGLang and NVIDIA Dynamo). With both axes overlapped on the same `max(TP, EP)` GPUs, no separate TP group exists for head-sharding to land on, so attention falls back to DP-attn by construction. Expert weights are no longer further TP-sharded within an expert-owning rank — each owned expert sits whole on its host GPU.

To carry these deviations through every per-device formula in `decode.md` without conditional branching, we abstract each shardable component's effective per-device divisor into a named symbol:

- $D_{\text{attn}}$ — Effective shard factor for attention weights (Q/K/V/O projections).
- $D_{\text{exp}}$ — Effective shard factor for FFN / expert weights. **Per-layer-type:** dense FFN always uses $D_{\text{exp}} = TP$ (no EP axis to overlap; co-location does not apply); MoE FFN follows the table below.
- $D_{\text{kv}}$ — Effective shard factor for KV cache by *head or sequence* (excludes the SP axis). Per-device KV memory and traffic always carry an additional $/SP$ factor on top of $D_{\text{kv}}$, reflecting sequence-parallelism sharding when SP is enabled.
- $D_{\text{emb}}$ — Effective shard factor for embedding / LM head weights.
- $G_{TP}$, $G_{EP}$ — Collective group sizes for TP and EP comms (group of devices participating in the per-layer collective). Annotated with the collective primitive (AR = all-reduce, AG = all-gather) where the swap depends on the attention mode.
- $N_{\text{replica}}$ — Devices per model replica.

The mapping from `(layout, attention_mode)` to factor values (MoE FFN row for $D_{\text{exp}}$; dense FFN always uses $TP$):

| layout | attention_mode | $D_{\text{attn}}$ | $D_{\text{exp}}$ (MoE) | $D_{\text{kv}}$ | $D_{\text{emb}}$ | $G_{TP}$ | $G_{EP}$ | $N_{\text{replica}}$ |
|---|---|---|---|---|---|---|---|---|
| orthogonal | TP-attn *(default)* | $TP$ | $TP \cdot EP$ | $TP$ (head) | $TP$ | $TP$ (AR) | $EP$ | $PP \cdot TP \cdot EP \cdot SP$ |
| orthogonal | DP-attn | $1$ | $TP \cdot EP$ | $TP$ (seq) | $TP$ | $TP$ (AG) | $EP$ | $PP \cdot TP \cdot EP \cdot SP$ |
| co-located | DP-attn *(production)* | $1$ | $EP$ | $\max(TP, EP)$ (seq) | $TP$ | $TP$ (AG) | $EP$ | $PP \cdot \max(TP, EP) \cdot SP$ |

KV cache footprint is **invariant** under the orthogonal TP-attn ↔ DP-attn swap ($D_{\text{kv}} = TP$ either way) — head-sharded by $TP$ vs sequence-sharded by $TP$ ranks gives the same per-device byte count; only the meaning of the divisor changes (`(head)` vs `(seq)` annotation).

A theoretical fourth combination — co-located + TP-attn — is not used in production: with TP and EP mapped to the same physical GPUs, there is no separate TP group for head-sharding to land on, so attention falls back to DP-attn by construction.

**Enabling conditions for co-location.** Per-device HBM must hold $1/EP$ (rather than $1/(TP \cdot EP)$) of the expert weights, OR the attention block is compact enough (Multi-head Latent Attention (MLA) or aggressive grouped-query attention (GQA)) that DP-attn replication costs little. DSv3 satisfies both via MLA + FP4 quantization [DSV3].

All `decode.md` derivations from §1 onward use the abstract factors $D_{\text{attn}}$, $D_{\text{exp}}$, $D_{\text{kv}}$, $D_{\text{emb}}$, $G_{TP}$, $G_{EP}$, $N_{\text{replica}}$. Each downstream section in `decode.md` opens with a small summary table mapping the abstract factors used in that section to each of the three configurations; this `notation.md §1` table is the canonical source. Per-device formulas live in `decode.md §1.4` (memory), `§2.1`, `§2.3` (traffic), `§3.5` (FLOPs), `§5.3`, `§5.5` (communication); operational guidance on when each mode pays lives in `§6.3` (partition strategy).

---

## 2. Parallelism Dimensions
_(→ decode.md)_

The five parallelism dimensions ($DP$, $PP$, $TP$, $EP$, $SP$) are **partition counts** — how many ways each kind of state is sharded. Their physical mapping to GPUs is governed by the layout choice (§1): under the orthogonal layout each $(PP, TP, EP, SP)$ tuple picks a unique GPU within a replica; under TP+EP co-location the $TP$ and $EP$ axes are *overlaid* on the same physical GPU set. The number of GPUs per replica is given by $N_{\text{replica}}$ (§1 lookup table), which itself depends on the layout.

- $DP$ — Data Parallelism. Number of full model replicas; each handles disjoint input batches.
  $$DP = \left\lfloor \frac{N_{\text{GPUs}}}{N_{\text{replica}}} \right\rfloor$$
  where $N_{\text{replica}} = PP \cdot TP \cdot EP \cdot SP$ under the orthogonal layout (default) and $N_{\text{replica}} = PP \cdot \max(TP, EP) \cdot SP$ under TP+EP co-location.
- $PP$ — Pipeline Parallelism. Layers split into stages; each stage holds $L_{\text{stage}} = L / PP$ layers. Independent of layout.
- $TP$ — Tensor Parallelism. Sharding factor for matrix multiplies (Megatron-LM column/row parallel). Under orthogonal layout, TP picks one GPU per attention/FFN shard; under co-location, TP and EP overlap on the same physical GPUs and the attention block falls back to DP-attn (the actual per-device divisors are encoded in $D_{\text{attn}}$, $D_{\text{exp}}$ from §1).
- $EP$ — Expert Parallelism. Sharding factor for MoE experts (each EP rank owns a subset of experts; tokens routed via all-to-all). Under orthogonal layout, EP picks one GPU per expert shard; under co-location, EP and TP share physical GPUs (per $D_{\text{exp}}$ in §1).
- $SP$ — Sequence Parallelism. Ring-attention-style KV sharding for inference; sequence dimension partitioned across devices for KV storage. Independent of layout (composes additively with $D_{\text{kv}}$).

---

## 3. Model Dimensions
_(→ decode.md)_

- $L$ — Number of transformer layers.
- $L_{\text{moe}}$ — Number of MoE layers (defaults to $L$ if all layers are MoE, or $0$ for dense).
- $L_{\text{dense}} = L - L_{\text{moe}}$ — Number of dense layers.
- $V$ — Vocabulary size.
- $H$ — Hidden size (model dimension); applies to embeddings, LM head, FFN, and attention projections.
- $n_q$ — Number of query heads.
- $d_{\text{head}} = H / n_q$ — Head dimension.
- $n_{kv}$ — Number of KV heads (in GQA, $n_{kv} < n_q$).
- $H_{kv} = n_{kv} \cdot d_{\text{head}}$ — Total KV projection dimension.
- $I_{\text{dense}}$ — FFN intermediate dimension for dense layers.
- $I_{\text{moe}}$ — FFN intermediate dimension per MoE expert layer.
- $I_{\text{eff}}$ — Unified FFN intermediate dimension for FLOPs:
  $I_{\text{eff}} = I_{\text{dense}}$ (dense) or $k \cdot I_{\text{moe}}$ (MoE).
- $N_{\text{exp}}$ — Number of experts per MoE layer.
- $N_{\text{eff}}$ — Unified expert count for FLOPs: $0$ (dense) or $N_{\text{exp}}$ (MoE).
- $k$ — Number of experts selected per token (top-$k$ routing).

**MLA (Multi-head Latent Attention) dimensions** (DeepSeek-V3 / R1, DeepSeek-V4-Pro, GLM-5, Kimi-K2.5; full coverage in `attention.md §3`). When a model uses MLA, the per-head Q / K / V symbols above are replaced by a compressed-latent decomposition:

- $d_c$ — KV latent dimension (head-shared); KV cache stores $d_c$ per token per layer instead of $2 H_{kv}$.
- $d_{q,c}$ — Query latent dimension.
- $d_{qk,\mathrm{nope}}$ — Non-positional Q / K head dimension.
- $d_{qk,\mathrm{rope}}$ — Rotary-position-embedded (RoPE) Q / K head dimension (head-shared on the K side).
- $d_v$ — Value head dimension.

For non-MLA models these symbols are unused; the standard $H$, $n_q$, $n_{kv}$, $H_{kv}$ apply per `decode.md` and `prefill.md`.

---

## 4. Sequence, Batch, and Precision
_(→ decode.md §4 for batch scaling; → prefill.md for prefill batch)_

- $S$ — Decode context length (tokens in KV cache during decoding).
- $S_{\text{input}}$ — Input sequence length for prefill.
- $B$ — Decode batch size: number of **independent user requests** decoded in the same step, each carrying its own KV cache. Weights are loaded once and shared across all $B$ requests; KV reads scale linearly with $B$. $B=1$ for single-request decode.
- $B_{\text{eff}}$ — **Per-step** realized decode batch size under continuous batching: the number of user requests contributing a decode token in a single iteration. Unlike the static $B$ (configured or peak admissible), $B_{\text{eff}}$ fluctuates step to step as requests finish (EOS) or new ones are admitted, and may be smaller than $B$ if prefill slots displace decode slots in the same step (e.g. chunked-prefill). The steady-state mean is $\overline{B_{\text{eff}}}$ (§14).
- $B_{\text{prefill}}$ — Number of independent user requests batched together in a single prefill pass.
- $b$ — Bytes per parameter/activation element (e.g., bf16 → $b=2$, fp8 → $b=1$).

---

## 5. Memory
_(→ decode.md; → kv.md for paging extensions)_

Parameter sizes:
- $P_{\text{attn}}$ — Attention parameter count.
- $P_{\text{FFN}}$ — Unified FFN/MoE parameter count.
- $P_{\text{emb}}$ — Embedding parameter count.
- $P_{\text{lm}}$ — LM head parameter count (0 if weight-tied).

Memory capacity (bytes **stored** in HBM):
- $M_{\theta,\text{device}}$ — Parameter memory on this device.
- $M_{\text{KV,device}}$ — KV cache storage (keys + values).
- $M_{\text{act,device}}$ — Activation working memory per token during decoding.
- $M_{\text{HBM}}$ — Available HBM capacity per device.

Memory traffic (bytes **moved** between HBM and compute per token):
- $T_{\theta,\text{device}}$ — Parameter traffic (weights read per token).
- $T_{\text{KV,device}}$ — KV traffic (read + write per new token).
- $T_{\text{act,device}}$ — Activation traffic (intermediate reads/writes).
- $T_{\text{token,device}}$ — Total per-token traffic on this device.
- $T_{\text{token,device}}^{\text{eff}}$ — Effective traffic after FlashAttention-style optimizations.

---

## 6. Device Compute and Bandwidth
_(→ decode.md; → dram3d.md for 3D DRAM extensions)_

- $N_{\text{GPUs}}$ — Total devices in the cluster.
- $R_{\text{GPU}}$ — Precision-aware compute throughput (FLOPs/s). System spec stores `peak_flops_TF` as the **FP16 dense per-chip peak** (uniform reference across all systems); the framework derives the working-precision peak by linear byte scaling: $R_{\text{GPU}}(b) = \mathrm{peak\_flops\_TF} \cdot (2 / b)$ for `bytes_per_param = b`. See `decode.md §3.1` for the full convention and the d-Matrix INT4 caveat (block-sparse acceleration not captured by the linear rule).
- $BW_{\text{mem}}$ — Effective HBM bandwidth (bytes/s).

---

## 7. Networking
_(→ decode.md; → `collectives/00_summary.md` §4–§7 for the shipped-primitive cost table consumed by decode.md / prefill.md; → `collectives/05_contention_and_congestion.md` §4 for dynamic η_α / η_β)_

The system model names physical networks as **fabrics**; each fabric is an ordered list of switching tiers, innermost first. A collective (TP / EP / SP / PP) declares an ordered **fabric chain** — the named sequence of fabrics that collective traverses from inner-most to outer-most. Walking that chain innermost-first yields a single flattened tier list; a collective of group size $G$ spans tiers $0..k$ where $k$ is the smallest index with $\prod_{i=0}^{k} P_i \ge G$.

- $P_{role,i}$ — Reach at tier $i$ along the $role$ collective's fabric chain (ranks reachable within that tier from any single rank). Topology-dependent: crossbar $P_i$ = switch radix; torus $P_i = \prod_j D_j$.
- $\alpha_{role,i}$ — Per-traversal startup latency of tier $i$ along the $role$ collective's fabric chain (μs).
- $BW_{role,i}$ — Effective per-port single-direction bandwidth of tier $i$ along the $role$ collective's fabric chain (GB/s, post-$\eta$).
- $\alpha_{role}(G)$ — **Span latency** for a collective of group size $G$: $\alpha_{role}(G) = \sum_{i \le k} \alpha_{role,i}$ with $k$ as above. Crossbar-only shorthand — torus tiers use their dim-decomposed primitives (`collectives/02_topology_mapping.md §3`) rather than a flat α-sum.
- $BW_{role}(G)$ — **Span bandwidth** for a collective of group size $G$: $BW_{role}(G) = \min_{i \le k} BW_{role,i}$ (narrowest crossed tier dominates, across fabric boundaries as well as within a fabric). Crossbar-only shorthand, same caveat as above.
- $n_{TP}$ — Number of TP collective iterations per layer per token.
- $n_{EP}$ — Number of EP collective iterations per layer per token.
- $n_{SP}$ — Number of SP collective iterations per layer per token.

**Torus tier symbols** (`collectives/02_topology_mapping.md §1, §3`):
- $k$ — Torus dimensionality (number of axes).
- $(D_1, \ldots, D_k)$ — Per-dim extents; reach $N = \prod_i D_i$.
- $D_\mathrm{max}$ — $\max_i D_i$; sets the A2A bisection floor.
- $\mathrm{diam}$ — Wraparound diameter: $\sum_i \lfloor D_i / 2 \rfloor$.
- $BW_\mathrm{link}$ — Per-link single-direction bandwidth (equal to tier's $BW_i$).
- $BW_\mathrm{bisect}^\mathrm{min}$ — Minimum bisection capacity: $2 N BW_\mathrm{link} / D_\mathrm{max}$.

**Contention coefficients** (`collectives/05_contention_and_congestion.md`):
- $\eta_\alpha$ — Dynamic α-inflator for a switching tier ($\geq 1$; ideal = 1). Captures serialization penalties under concurrent collectives and off-prefix layouts that steady-state microbenchmarks miss.
- $\eta_\beta$ — Dynamic BW-deflator for a switching tier ($\in (0, 1]$; ideal = 1). Captures runtime bandwidth loss beyond the calibrated peak. Hierarchical fabrics cap upper-tier $\eta_\beta$ at $\min(\eta_\beta^\mathrm{hw}, 1/s)$ where $s$ is the oversubscription ratio (`collectives/05_contention_and_congestion.md §4`).

**Single-tier shorthand.** A chain with one crossbar fabric and one crossbar tier collapses to the flat pair $\alpha_{role} \equiv \alpha_{role,0}$, $BW_{role} \equiv BW_{role,0}$, independent of $G$. The decode/prefill equations in decode.md and prefill.md are written against this flat pair; multi-tier and torus analyses substitute the appropriate span quantity (or torus-native formula) with $G$ set by the role-specific group size (e.g. $G = \text{TP}$ for TP collectives, $G = \text{EP}$ for EP). For PP point-to-point hops, the tier is selected via the **nested-layout convention** below — *not* by `G = 2`, which would always pick tier 0. See `collectives/00_summary.md §4–§7` for the shipped-primitive cost table (ring / DBT AR on star, dim-decomposed ring on torus, hierarchical RS → sub-AR → AG, in-network reduction via NVLS / Quantum SHARP / Tomahawk Ultra) consumed by decode.md §5 and prefill.md §3.2; `collectives/05_contention_and_congestion.md` for $\eta_\alpha / \eta_\beta$ application.

**Nested-layout convention.** Per-axis tier assignment under the production-standard layout `DP → PP → EP → TP → SP` (innermost = highest-bandwidth). Walk the fabric chain inner→outer; for each axis (in inner-to-outer order: SP, TP, EP, PP), assign the smallest tier whose cumulative reach $\prod_{i \le t} P_i$ holds the cumulative product of inner axes × this axis. Example on d-Matrix squadrack (3-tier chain $P = (16, 4, 8)$, cumulative $(16, 64, 512)$): $TP=8, EP=1, PP=2$ → TP at tier 0 ($8 \le 16$), PP at tier 0 ($16 \le 16$); $TP=8, PP=8$ → PP at tier 1 ($64 \le 64$); $TP=8, PP=32$ → PP at tier 2 ($256 \le 512$). Single-tier systems (e.g., NVL72) collapse all axes to tier 0.

**Collective-primitive coefficients** (`collectives/00_summary.md §4–§7`):
- $n_\alpha$ — Coefficient on $\alpha$ in a shipped-primitive cost formula (number of startup traversals). Per-primitive values in `collectives/00_summary.md §4` (or full derivations in `collectives/01_collective_algorithms.md`).
- $n_\beta$ — Coefficient on $M / \mathrm{BW}$. Per-primitive values in `collectives/00_summary.md §4` (full derivations in `collectives/01_collective_algorithms.md`).
- $\alpha_\mathrm{switch}$ — Switch cut-through latency (200–400 ns) consumed by in-network collective formulas (`collectives/04_in_network_collectives.md`).
- $\mathrm{BW_{eff}} = \mathrm{BW} / n_\beta$ — Effective per-rank bandwidth seen by a collective. AR alone has $\mathrm{BW_{eff}} = \mathrm{BW}/2$ in software and $\mathrm{BW_{eff}} = \mathrm{BW}$ under INC (switch ALU + multicast crossbar fuses the two halves; `collectives/04_in_network_collectives.md`).
- $\mathrm{ar\_algorithm}$ — Tuning-knob symbol selecting star AR algorithm: admissible values $\{\mathrm{ring}, \mathrm{DBT}\}$, default $\mathrm{ring}$. Does not apply to torus AR (only dim-decomposed ring is shipped). See `collectives/02_topology_mapping.md §2`.

---

## 8. FLOPs
_(→ decode.md)_

Attention:
- $F_Q, F_K, F_V, F_O$ — FLOPs for Q, K, V, output projections.
- $F_{\text{proj}}$ — Combined Q/K/V/O projection FLOPs.
- $F_{\text{score}}$ — Attention score FLOPs ($QK^\top$).
- $F_{\text{value}}$ — Value application FLOPs (Attn·V).
- $F_{\text{attn,KV}}$ — Score + value FLOPs combined.
- $F_{\text{attn}}$ — Total attention FLOPs per layer.

FFN and MoE:
- $F_{\text{ffn,dense}}$ — Dense FFN FLOPs per layer.
- $F_{\text{router}}$ — Router FLOPs per token (MoE).
- $F_{\text{expert}}$ — FLOPs per expert MLP per token.
- $F_{\text{ffn,moe}}$ — MoE FFN FLOPs per layer.
- $F_{\text{ffn}}$ — Unified FFN FLOPs (dense or MoE).

Layer and token:
- $F_{\text{layer}}$ — Total FLOPs per layer (attention + FFN; norm dropped as negligible).
- $F_{\text{layer,device}}$ — FLOPs per layer per device after sharding.
- $F_{\text{token,device}}$ — Total FLOPs per generated token on this device (decode).

---

## 9. Decode Timing and Throughput
_(→ decode.md)_

- $t_{\text{compute}}$ — Per-token compute time at peak Tensor Core throughput: $F_{\text{token,device}} / R_{\text{GPU}}$.
- $\eta_{\mathrm{TC}}(\mathrm{mb})$ — Tensor Core efficiency factor at microbatch $\mathrm{mb} = B/PP$. Piecewise-linear from a user-supplied curve; defaults to 1 (no derate). Captures the wgmma / mma.sync M-tile floor (kernel_launch_overhead.md §2, practical_pp_choice.md §3.3).
- $t_{\text{compute}}^{\mathrm{eff}}$ — Tensor-Core-derated compute time: $t_{\text{compute}} / \eta_{\mathrm{TC}}(\mathrm{mb})$.
- $t_{\text{mem}}$ — Per-token memory time: $T_{\text{token,device}}^{\text{eff}} / BW_{\text{mem}}$.
- $t_{\text{local}}$ — Roofline local time: $\max(t_{\text{compute}}^{\mathrm{eff}}, t_{\text{mem}})$.
- $t_{TP}, t_{EP}, t_{SP}, t_{PP}$ — Communication time per step per parallelism type (message sizes scale with $B$; see decode.md §5).
- $t_{\text{comm}}$ — Combined communication time per decode step per PP stage.
- $t_{\text{stage}}$ — Per-PP-stage GPU-side step time (overlap-aware, pre-bubble):
  $$t_{\text{stage}} = t_{\text{local}} + \max(0,\; t_{\text{comm}} - \rho \cdot t_{\text{local}})$$
- $\tau_{\mathrm{launch}}$ — Per-kernel CPU dispatch latency (typical: ~1.5 μs with CUDA Graphs, ~7 μs without).
- $k$ — Kernel launches per layer per microbatch: $k = k_{\mathrm{compute}} + k_{\mathrm{collective}} \cdot (n_{\mathrm{TP}}^{\mathrm{calls}} + n_{\mathrm{EP}}^{\mathrm{calls}} + n_{\mathrm{SP}}^{\mathrm{calls}})$. Per-axis NCCL API call counts (zero when that parallelism axis is 1): $n_{\mathrm{TP}}^{\mathrm{calls}} = n_{\mathrm{TP\_collectives}}$; $n_{\mathrm{SP}}^{\mathrm{calls}} = n_{\mathrm{SP\_collectives}}$; **$n_{\mathrm{EP}}^{\mathrm{calls}} = 2 \cdot n_{\mathrm{EP\_collectives}}$** because the cost-model treats one MoE A2A as a single round-trip (with the 2× wrapped inside `_cost("moe_a2a")`), but the launch counter must expand to 2 actual NCCL calls (dispatch + combine).
- $k_{\mathrm{pp\_hop}}$ — Kernels per PP boundary per microbatch on each device: typically 2 (1 recv + 1 send); 1 if `ncclSendRecv` or a custom kernel fuses the pair.
- $t_{\mathrm{SW}}$ — Per-round CPU dispatch budget on each device: $t_{\mathrm{SW}} = L \cdot k \cdot \tau_{\mathrm{launch}} + PP \cdot k_{\mathrm{pp\_hop}} \cdot \tau_{\mathrm{launch}}$. The second term counts inter-stage P2P launches ($k_{\mathrm{pp\_hop}}$ per microbatch × $PP$ microbatches per round); inert when $PP = 1$ (kernel_launch_overhead.md §5).
- $\rho_{\mathrm{SW}}$ — SW-overlap factor $\in [0, 1]$: fraction of $t_{\mathrm{stage}}$ that hides $t_{\mathrm{SW}}$ via async kernel dispatch. Default 1 (full overlap — upper-end case, accurate for CUDA-Graphs-replayed production stacks where the CPU has 1000× slack). Empirical production typically measures ~0.85–0.95 with CUDA Graphs; eager-mode Python serving sits at ~0.3–0.6. The 1.0 default matches the roofline philosophy; see `decode.md §7.1` for the canonical definition and operational caveats.
- $\gamma_{\text{pp}}$ — Pipeline-bubble multiplier:
  $$\gamma_{\text{pp}} = \max\left(1,\; \frac{PP}{B}\right)$$
  Equal to 1 when the pipeline is kept full ($B \ge PP$); greater than 1 when a single microbatch must traverse all PP stages sequentially ($B < PP$).
- $t_{\text{step,user}}$ — User-observed per-step decode time:
  $$t_{\text{step,user}} = \max\!\bigl(t_{\text{stage}},\ \rho_{\mathrm{SW}} \cdot t_{\text{stage}} + (1 - \rho_{\mathrm{SW}}) \cdot (t_{\text{stage}} + t_{\mathrm{SW}}),\ t_{\mathrm{SW}}\bigr) \cdot \gamma_{\text{pp}}$$
  Reduces to $t_{\text{stage}} \cdot \gamma_{\text{pp}}$ when $t_{\mathrm{SW}} = 0$ (SW disabled).
- $\rho$ — Compute-comm overlap factor $\in [0,1]$: fraction of $t_{\text{local}}$ that hides $t_{\text{comm}}$.
- $TPS_{\text{single}}$ — Per-DP-replica decode throughput: $B / t_{\text{step,user}}$ (tokens/s).
- $TTPS$ — Global decode throughput across all DP replicas: $DP \cdot B / t_{\text{step,user}}$ (tokens/s).

---

## 10. Batch Scaling
_(→ decode.md §4)_

- $OI(B)$ — Operational intensity as a function of batch size $B$:
  $$OI(B) = \frac{B \times F_{\text{token,device}}}{T_{\theta,\text{device}} + B \times T_{\text{KV,device}}}$$
- $B^*$ — Crossover batch size where the roofline transitions from memory-bound to compute-bound:
  $$B^* = \frac{T_{\theta,\text{device}} \times R_{\text{GPU}}}{F_{\text{token,device}} \times BW_{\text{mem}} - T_{\text{KV,device}} \times R_{\text{GPU}}}$$
  **Existence:** finite and positive iff $F_{\text{token,device}} / T_{\text{KV,device}} > R_{\text{ridge}}$ (asymptotic OI ceiling exceeds the ridge point). When violated — e.g., very long contexts on small models — decode stays memory-bound at every $B$ and $B^{\star} \to \infty$ (decode.md §4).
- $\text{TPOT}(B)$ — Batched Time Per Output Token (user-observed): $t_{\text{step,user}}(B)$.
  Memory-bound ($B \ll B^*$): $\approx T_{\theta,\text{device}} / BW_{\text{mem}}$ (flat in $B$).
  Compute-bound ($B \gg B^*$): $\approx B \cdot F_{\text{token,device}} / R_{\text{GPU}}$ (linear in $B$).

---

## 11. Prefill and TTFT
_(→ prefill.md)_

FLOPs:
- $F_{\text{proj,prefill}}$ — Q/K/V/O projection FLOPs for prefill: $(4H^2 + 4HH_{kv}) S_{\text{input}}$.
- $F_{\text{score,prefill}}$ — Attention score FLOPs: $2 S_{\text{input}}^2 H$.
- $F_{\text{value,prefill}}$ — Value application FLOPs: $2 S_{\text{input}}^2 H$.
- $F_{\text{ffn,prefill}}$ — FFN FLOPs for prefill: $6 H I_{\text{eff}} S_{\text{input}}$.
- $F_{\text{router,prefill}}$ — Router gate FLOPs per MoE layer for prefill: $2 H N_{\text{exp}} S_{\text{input}}$ (unsharded across TP; zero for dense layers).
- $F_{\text{layer,prefill}}$ — Per-layer prefill FLOPs (projections + $S^2$ attention + FFN; MoE layers additionally include the router term).
- $F_{\text{prefill,device}}$ — Total prefill FLOPs per device across all layers on this PP stage.

Timing:
- $t_{\text{prefill,compute}}$ — Prefill compute time: $F_{\text{prefill,device}} / R_{\text{GPU}}$.
- $t_{\text{prefill,mem}}$ — Prefill memory time: $T_{\text{prefill,device}} / BW_{\text{mem}}$.
- $t_{\text{prefill,local}}$ — Prefill roofline local time: $\max(t_{\text{prefill,compute}}, t_{\text{prefill,mem}})$.
- $t_{\text{prefill,comm}}$ — Total prefill communication time (TP/EP/SP/PP collectives).
- $t_{\text{chunk}}$ — Latency of one chunked-prefill iteration.
- $t_{\text{pipeline,warmup}}$ — Pipeline fill time: $(PP-1) \times t_{\text{stage}}$.
- $t_{\text{prefill}}$ — Hardware prefill latency for one request: $\max(t_{\text{prefill,local}},\; t_{\text{prefill,comm}}) + t_{\text{pipeline,warmup}}$ (derived in prefill.md §3).
- $t_{\text{prefill,total}}$ — End-to-end hardware prefill latency including handoff: $t_{\text{prefill}} + t_{\text{handoff}} + t_{\text{pipeline,warmup,dec}}$ (prefill.md §6.5).
- $TTFT$ — Time To First Token (assembled from the terms above plus framework overheads in §13):
  $$TTFT = t_{\text{sched}} + t_{\text{tok}} + t_{\text{prefill}} + t_{\text{handoff}} + t_{\text{step,user}}$$
  where $t_{\text{sched}}, t_{\text{tok}}$ are defined in §13 and $t_{\text{handoff}}$ is below (0 only when prefill and decode share an identical partition).
- $TTFT_{\text{disagg}}$ — TTFT for disaggregated prefill architecture (uses the refined $t_{\text{KV-transfer}}^{\text{eff}}$ for $t_{\text{handoff}}$).

Chunked prefill:
- $C$ — Chunk size in tokens.
- $N_{\text{chunks}}$ — Number of chunks: $\lceil S_{\text{input}} / C \rceil$.

Crossover:
- $S_{\text{input}}^{\star}$ — Prefill compute-bound crossover: $S_{\text{input}}^{\star} = (b/2) \times R_{\text{ridge}}$.
- $R_{\text{ridge}}$ — Device ridge point: $R_{\text{GPU}} / BW_{\text{mem}}$ (FLOPs/byte).

KV handoff — volumes (cluster-aggregate and per-device views):
- $M_{\text{KV,total}}$ — Total KV bytes from one prefill pass (cluster-aggregate): $2 L S_{\text{input}} H_{kv} b$.
- $M_{\text{KV,shard,p}}$ — Per-prefill-device KV shard (all layers that device holds): $M_{\text{KV,total}} / (TP_p \cdot SP_p)$ for one PP stage; used by prefill.md §6.
- $M_{\text{KV-transfer}}$ — Per-device KV bytes to transfer (sharded across $TP \cdot SP$, one PP stage's worth of layers): $2 S_{\text{input}} H_{kv} b \times L / (PP \times TP \times SP)$. Used by e2e.md in the simple α–β model.

KV handoff — latency model (prefill.md §6):
- $t_{\text{handoff}}$ — KV handoff time from prefill to decode; co-located or disaggregated branch:
  $$t_{\text{handoff}} = \begin{cases} t_{\text{handoff,colo}} & \text{(co-located, §6.3)}\\ t_{\text{KV-transfer}}^{\text{eff}} & \text{(disaggregated, §6.4)} \end{cases}$$
- $t_{\text{handoff,colo}}$ — Co-located KV layout-transition latency (scale-up collective):
  $$t_{\text{handoff,colo}} = \alpha_{\text{intra}} + \frac{M_{\text{KV,total}}}{BW_{\text{intra}}} \cdot \eta_{\text{repack}}$$
  Equals 0 only when prefill and decode partitions match exactly.
- $t_{\text{KV-transfer}}^{\text{bulk}}$ — Textbook α–β disaggregated transfer (no overlap, no overheads): $\alpha_{\text{inter}} + M_{\text{KV,total}} / BW_{\text{inter}}$.
- $t_{\text{KV-transfer}}^{\text{eff}}$ — Refined disaggregated transfer (overheads + layer-wise streaming):
  $$t_{\text{KV-transfer}}^{\text{eff}} = \max\!\left(0,\; \alpha_{\text{inter}}^{\text{eff}} + \frac{M_{\text{KV,total}}}{BW_{\text{inter}}} + t_{\text{repack}} - \rho_{KV}\cdot t_{\text{prefill}} \right)$$
- $t_{\text{KV-transfer}}$ — Generic KV transfer latency in the simple α–β model (used by e2e.md): $\alpha_{\text{inter}} + M_{\text{KV-transfer}} / BW_{\text{inter}}$ (0 for co-located prefill+decode in the simple model).
- $t_{\text{repack}}$ — Layout repack on decode side (scale-up all-gather): $M_{\text{KV,total}} / BW_{\text{intra,d}} \cdot \eta_{\text{repack}}$.

KV handoff — bandwidth and startup parameters:
- $BW_{\text{inter}}$ — *Effective, delivered* end-to-end per-GPU inter-cluster bandwidth (calibration knob). Absorbs PCIe egress, NIC sharing, and HBM-write inefficiencies; is **not** the NIC catalog line rate. See prefill.md §6.4.
- $BW_{\text{intra}}$, $BW_{\text{intra,d}}$ — Scale-up fabric bandwidth (NVLink / NVSwitch); decode-side variant for repack cost.
- $\alpha_{\text{inter}}$ — Inter-cluster link startup latency (single round-trip).
- $\alpha_{\text{inter}}^{\text{eff}}$ — Effective startup including RDMA WR posting: $\alpha_{\text{inter}} + N_{\text{WR}} \cdot \tau_{\text{WR}}$.
- $\alpha_{\text{intra}}$ — Scale-up collective startup (≈1–5 µs over NVLink/NVSwitch).
- $\eta_{\text{repack}}$ — Layout-repack inefficiency factor ($\in [1, 2]$); covers non-contiguous gather + paged-block writes.
- $\rho_{KV}$ — Layer-wise streaming overlap factor for disaggregated KV transfer ($\in [0, 1]$); fraction of $t_{\text{prefill}}$ that hides KV transfer (MoonCake / NVIDIA Dynamo pattern).
- $N_{\text{WR}}$ — Number of RDMA work requests posted in one handoff: $\approx L \cdot TP_p \cdot SP_p$.
- $\tau_{\text{WR}}$ — Per-RDMA-WR posting latency (≈1 µs).
- $t_{\text{pipeline,warmup,dec}}$ — Pipeline warmup on decode cluster after handoff: $(PP_{\text{dec}} - 1) \cdot t_{\text{stage,dec}}$.

---

## 12. KV Cache Management
_(→ kv.md)_

- $\text{BLK}_{KV}$ — KV block size in tokens (PagedAttention page size; typical: 16 or 32).
- $N_{\text{blocks}}(S)$ — Blocks allocated for a sequence of length $S$: $\lceil S / \text{BLK}_{KV} \rceil$.
- $\varphi(S)$ — Fragmentation factor: ratio of allocated KV memory to ideally occupied KV memory.
  $$\varphi(S) = \frac{\lceil S / \text{BLK}_{KV} \rceil \times \text{BLK}_{KV}}{S}, \qquad \varphi_{\text{avg}} \approx 1 + \frac{\text{BLK}_{KV}}{2S}$$
- $M_{\text{HBM,KV,avail}}$ — HBM capacity available for KV storage after weights and activations.
- $S_{\max}$ — Maximum supportable context length given $M_{\text{HBM,KV,avail}}$ and $\varphi$:
  $$S_{\max} \approx \frac{M_{\text{HBM,KV,avail}} \times D_{\text{kv}} \times SP}{\varphi_{\text{avg}} \times 2 H_{kv} b \times L/PP}$$
  $D_{\text{kv}}$ values: $TP$ (orthogonal + TP-attn or DP-attn) / $\max(TP, EP)$ (co-located + DP-attn).

---

## 13. Framework Overhead
_(→ framework.md)_

Scope: CPU / software-stack overhead only. Network-fabric overheads (e.g., disaggregated KV transfer) are handled in §11 / prefill.md §6.

Per-request (once):
- $t_{\text{tok}}$ — Tokenization latency (CPU BPE/SP processing).
- $t_{\text{sched}}$ — Request scheduling / batch assembly latency.

Per-step (each decode iteration):
- $t_{\text{detok}}$ — Response streaming / detokenization latency per output token.

_(Kernel-launch / CUDA-Graph dispatch overheads $t_{\text{stage,sw}}$ are HW-side and live in decode.md §7.1, not in framework.md. The LM head GEMM and post-LM-head sampling kernel are absorbed into $t_{\text{LM,hw}}$ on the GPU side; see decode.md §6.2 / §7.2.)_

Request scope:
- $T_{\text{out}}$ — Number of output tokens per request.

_(Total framework overhead $t_{\text{framework}}$ is assembled in framework.md §3.)_

---

## 14. End-to-End Metrics
_(→ e2e.md)_

Core metrics:
- $TTFT$ — Time To First Token (defined in §11 above; assembled in e2e.md §2).
- $\text{TPOT}(B)$ — Time Per Output Token (user-observed): $t_{\text{step,user}}(B)$ (defined in §10 above).
- $\text{Tput/GPU}$ — Output tokens per second per GPU: $TTPS / N_{\text{GPUs}}$.
- $\text{Interactivity}$ — Per-user output rate: $1 / \text{TPOT} = 1 / t_{\text{step,user}}(B)$ (tokens/s/request).
- $\text{Goodput}$ — Maximum request arrival rate $\lambda$ the cluster can sustain while keeping both TTFT and TPOT below operator-set SLOs at percentile $p$ (e2e.md §1.5):
  $$\text{Goodput} = \max\,\lambda \;\;\text{s.t.}\;\; P_{p}[TTFT(\lambda)] \le TTFT_{\text{SLO}} \;\text{and}\; P_{p}[\text{TPOT}(\lambda)] \le \text{TPOT}_{\text{SLO}}$$
  **Scope note:** speculative decoding, preemption-driven recompute, and cancellation effects are real goodput drains but not modeled in this suite.
- $TTFT_{\text{SLO}}$ — Operator-set upper bound on TTFT (seconds) used in the goodput definition.
- $\text{TPOT}_{\text{SLO}}$ — Operator-set upper bound on TPOT (seconds) used in the goodput definition.
- $p$ — SLO compliance percentile (typically 90 or 99).

SLO-derived partition feasibility (slo.md):
- $B_{\max}$ — Largest batch size $B$ satisfying the TPOT SLO at the candidate partition shape; in Zone 3, $B_{\max} \approx R_{\text{GPU}} \cdot \text{TPOT}_{\text{SLO}} / F_{\text{token,device}}$ (slo.md §2.2).
- $B_{\text{HBM}}$ — Largest batch size $B$ permitted by HBM capacity at the candidate partition shape: $B_{\text{HBM}} \approx HBM_{\text{free}} / (T_{\text{KV,device}} \cdot S)$ (slo.md §2.2).
- $B_{\text{op}}$ — SLO-feasible operating batch size: $\min(B_{\max}, B_{\text{HBM}})$ (slo.md §5.1).
- $PP_{\max}$ — Largest pipeline-parallel depth satisfying the TTFT SLO at the candidate input length: $PP_{\max} \approx 1 + (\text{TTFT}_{\text{SLO}} - t_{\text{sched}} - t_{\text{prefill,local}} - t_{\text{step,user}}) / t_{\text{stage,max}}$ (slo.md §3.1).
- $\lambda^*$ — Goodput rate; the maximum $\lambda$ over the SLO-feasible region; the optimization target for partition selection (slo.md §5.1).
- $\mathcal{F}_{\text{SLO}}$ — Joint feasibility region in $(PP, TP, EP, SP, B)$ space where the TPOT floor, TPOT bound, TTFT bound, and HBM capacity all hold (slo.md §4.1).
- $\Delta_{\text{dyn}}$ — Dynamic-stability cushion on $\overline{B}$ to absorb p99 batch-size variance: $\Delta_{\text{dyn}} \approx z_p \sqrt{\overline{B}}$ for Poisson arrivals (slo.md §4.2).
- $z_p$ — Standard-normal $p$-quantile: 1.28 for p90, 1.96 for p95, 2.33 for p99 (slo.md §4.2).

Continuous batching:
- $\lambda$ — Request arrival rate (requests/second).
- $N_{\text{out}}$ — Number of output tokens in a single response.
- $N_{\text{out}}^{\star}$ — Crossover output length where TTFT equals the decode contribution: $N_{\text{out}}^{\star} \approx TTFT / \text{TPOT} + 1$ (e2e.md §4).
- $\overline{B_{\text{eff}}}$ — Mean effective batch size in steady-state continuous batching.
- $\overline{\text{TPOT}}$ — Average TPOT over a request's decode lifetime.
- $N_{\text{GPUs,per-replica}}$ — GPUs per DP replica; equal to $N_{\text{replica}}$ (§1 lookup table). Resolves to $PP \cdot TP \cdot EP \cdot SP$ under the orthogonal layout (default) and $PP \cdot \max(TP, EP) \cdot SP$ under TP+EP co-location.

Pareto relationship (per-replica, parameterized by $B$):
$$\text{Tput/GPU} \times \text{TPOT} = \frac{B}{N_{\text{GPUs,per-replica}}}$$
The ceiling $1/N_{\text{GPUs,per-replica}}$ applies at $B=1$; for $B>1$, a single replica can sit on a higher hyperbola.

---

## 15. 3D DRAM
_(→ dram3d.md)_

Physical parameters:
- $A_{\text{die}}$ — DRAM die area (mm²).
- $p_{HB}$ — Hybrid bonding pitch: center-to-center pad spacing (µm).
- $\eta_{\text{data}}$ — Data pin fraction: proportion of pads carrying data signals.
- $f_{\text{data}}$ — Data rate per pin (Gbps).
- $N_{\text{dies}}$ — Number of DRAM dies stacked on the logic die.

Derived:
- $N_{\text{pins,total}}$ — Total pad count (area-limited): $\lfloor A_{\text{die}} / p_{HB}^2 \rfloor$.
- $N_{\text{pins,data}}$ — Data pad count: $N_{\text{pins,total}} \times \eta_{\text{data}}$.
- $BW_{\text{die}}$ — Raw bandwidth per die interface: $N_{\text{pins,data}} \times f_{\text{data}} / 8$ (GB/s).
- $BW_{\text{conservative}}$ — Lower bound: single logic-facing interface ($= BW_{\text{die}}$).
- $BW_{\text{optimistic}}$ — Upper bound: independent per-die interfaces ($= N_{\text{dies}} \times BW_{\text{die}}$).

Latency:
- $k_{\text{interconnect}}$ — Latency reduction factor vs. standard HBM bump interconnect.
- $\ell_{3D}$ — Estimated 3D DRAM read latency: $\ell_{\text{HBM}} / k_{\text{interconnect}}$ (ns).

---

## 16. SRAM-Centric Memory Hierarchy
_(→ sram.md)_

Tier list (ordered, fastest first):
- $n$ — Number of memory tiers exposed by a device.
- $i$ — Tier index, $0 \le i < n$, ordered fastest first.

Per-tier physical parameters:
- $C_i$ — Tier $i$ capacity per device (bytes).
- $BW_i$ — Tier $i$ peak read bandwidth from compute (bytes/s).
- $\alpha_i$ — Tier $i$ first-byte latency floor (seconds); does not enter steady-state decode timing.
- $\eta_{\beta,i}$ — Tier $i$ sustained-bandwidth deflator ($\in (0, 1]$). Defaults: SRAM $\approx 1.0$, HBM $\approx 0.92$, LPDDR5 $\approx 0.85$ (sram.md §1.2).

Derived:
- $BW_{\text{eff},i} = BW_i \cdot \eta_{\beta,i}$ — Effective tier $i$ bandwidth (bytes/s).

Placement:
- $\pi$ — Placement assigning each data class to one or more tiers.
- $T_{\theta,i}$ — Weight bytes residing on tier $i$; $\sum_i T_{\theta,i} = T_{\theta,\text{device}}$.
- $T_{\text{KV},i}$ — Per-request KV bytes residing on tier $i$; $\sum_i T_{\text{KV},i} = T_{\text{KV,device}}$.
- Capacity constraint per tier: $T_{\theta,i} + B \cdot T_{\text{KV},i} \le C_i$ (sram.md §1.3; uses the per-device, per-request $T_{\text{KV,device}}$ from `decode.md §2.3`, which already bakes in $D_{\text{kv}} \cdot SP$ and the context length $S$).
- **Greedy priority** — Tiebreaker when neither weights nor KV is explicitly pinned to a tier: weights-first (default) fills weights into the fastest tier and gives KV the remainder; KV-first flips the order. Inert when either class is explicitly pinned.

Multi-tier roofline (sram.md §2.1) — full $\alpha$–$\beta$ form:
$$t_{\text{mem}}(B) = \sum_i \left[\, \alpha_i + \frac{T_{\theta,i} + B \cdot T_{\text{KV},i}}{BW_{\text{eff},i}} \,\right]$$
- Dropped-$\alpha$ form (used for device-level decode roofline; sram.md §2.1 justifies on magnitude grounds, $\alpha < 0.1\%$ of $t_{\text{mem}}$):
$$t_{\text{mem}}(B) = \sum_i \frac{T_{\theta,i} + B \cdot T_{\text{KV},i}}{BW_{\text{eff},i}}$$
- Single-tier reduction ($n=1$) recovers `decode.md §4.3` exactly with $BW_{\text{mem}} \equiv BW_{\text{eff},0}$.

Two-tier crossover (sram.md §2.2; weights pinned to tier $W$, KV to tier $K$):
$$B^*_{W,K} = \frac{R_{\text{GPU}} \cdot T_{\theta,\text{device}} / BW_{\text{eff},W}}{F_{\text{token,device}} - R_{\text{GPU}} \cdot T_{\text{KV,device}} / BW_{\text{eff},K}}$$
- Reduces to the single-tier $B^\star$ of §10 when $W = K$.

---

## 17. Attention Parallelism Modes and Layout Deviations
_(→ §1 above; `decode.md §1.4` / §2 / §3.5 / §5)_

This section's prior content (per-DP-attn symbol register) has been **subsumed by the unified deployment-knob abstraction in §1**. The attention parallelism mode (TP-attn ↔ DP-attn) and the layout choice (orthogonal ↔ TP+EP co-located) are now expressed through the per-component effective sharding factors $D_{\text{attn}}$, $D_{\text{exp}}$, $D_{\text{kv}}$, $D_{\text{emb}}$ and collective group sizes $G_{TP}$, $G_{EP}$ defined in §1. Every per-device formula in `decode.md §1.4` (memory), `§2.1` / `§2.3` (traffic), `§3.5` (FLOPs), and `§5.3` / `§5.5` (communication) is written directly in terms of those abstract factors and resolves automatically under any of the three production-relevant `(layout, attention_mode)` configurations via the §1 lookup table.

KV cache footprint and per-device KV traffic are **invariant** under the TP-attn ↔ DP-attn swap (head-sharded by $TP$ vs sequence-sharded by $SP$ give the same per-device byte count when $TP = SP$, and the abstract $D_{\text{kv}}$ encodes whichever divisor applies; see `decode.md §1.4` for the accounting). The TP collective primitive swaps from all-reduce (AR) under TP-attn to all-gather (AG) under DP-attn — encoded by the AR/AG annotation on $G_{TP}$ in the §1 lookup table.

For operational guidance on **when each mode pays** (the trade-off between attention memory footprint, replica-size shrinkage, and per-collective $\alpha$-cost), see `decode.md §6` (partition strategy).

---

## 18. Speculative Decoding (MTP / EAGLE / Medusa)
_(→ decode.md §8)_

Speculative decoding breaks the 1:1 token-per-step mapping of vanilla decode by running a cheap draft head to propose multiple candidate tokens, then verifying all of them in a single target-model pass. Per-step output count exceeds 1 on average; per-step cost grows by the verify multiplier. Symbols introduced in `decode.md §8`.

- $n_{\text{tok,draft}}$ — Number of draft tokens proposed per verify step (typical 3–5; 1 disables speculation).
- $n_{\text{tok,verify}} = n_{\text{tok,draft}} + 1$ — Tokens evaluated per verify step per sequence (the proposed tokens plus the target's always-accepted next prediction).
- $p_{\text{accept}}$ — Per-token draft acceptance probability ∈ [0, 1] (calibrate against the deployment).
- $\mathbb{E}[N_{\text{accept,draft}}]$ — Expected accepted draft tokens per step under independent acceptance (truncated geometric, `decode.md §8.2`):
  $$\mathbb{E}[N_{\text{accept,draft}}] = \sum_{d=1}^{n_{\text{tok,draft}}} p_{\text{accept}}^d = \frac{p_{\text{accept}} \, (1 - p_{\text{accept}}^{n_{\text{tok,draft}}})}{1 - p_{\text{accept}}}$$
- $N_{\text{tok/step}} = 1 + \mathbb{E}[N_{\text{accept,draft}}]$ — Total expected accepted tokens per verify step. Bound: $1 \le N_{\text{tok/step}} \le n_{\text{tok,verify}}$.
- $t_{\text{compute}}^{\text{verify}}(B) = n_{\text{tok,verify}} \cdot t_{\text{compute}}(B)$ — Verify-step compute time (FLOPs scale linearly with $n_{\text{tok,verify}}$; `decode.md §8.3`).
- $t_{\text{mem}}^{\text{verify}}(B) \approx t_{\text{mem}}(B)$ — Verify-step memory time (weights amortize once per step; KV reads piggyback under FlashAttention-style batched-Q kernels; `decode.md §8.3`).
- $t_{\text{local}}^{\text{verify}}(B) = \max\bigl( n_{\text{tok,verify}} \cdot t_{\text{compute}}(B),\; t_{\text{mem}}(B) \bigr)$ — Verify-step roofline local time.
- $t_{\text{comm}}^{\text{verify}}(B)$ — Verify-step communication budget; §5.5 expression with all per-layer message sizes scaled by $n_{\text{tok,verify}}$ (α-side terms unchanged).
- $t_{\text{step,user}}^{\text{verify}}(B)$ — Verify-step user-observed step time; §7.2 composition with the verify-step quantities substituted in.
- $\mathrm{TPOT_{spec}}(B) = t_{\text{step,user}}^{\text{verify}}(B) / N_{\text{tok/step}}$ — Effective TPOT under speculative decoding (`decode.md §8.4`).
- $B^\star_{\text{spec}}$ — Verify-step compute-bound crossover batch (`decode.md §8.4`):
  $$B^\star_{\text{spec}} \approx \frac{T_{\theta,\text{device}} \cdot R_{\text{GPU}}}{n_{\text{tok,verify}} \cdot F_{\text{token,device}} \cdot BW_{\text{mem}} - T_{\text{KV,device}} \cdot R_{\text{GPU}}}$$
  Falls below the vanilla $B^\star$ of §10 by approximately $n_{\text{tok,verify}}$.

---

## 19. Per-Sequence Serving Runtime Overhead
_(→ decode.md §7.2)_

Captures host-side per-step work that scales with the active-sequence count $B$ — PagedAttention block-table gather, continuous-batching scheduler decisions, per-sequence sampling glue, token-append bookkeeping. Distinct from the kernel-launch dispatch budget $t_{\mathrm{stage,sw}}$ (per microbatch, near-constant in $B$; `decode.md §7.1`) and from the per-request scheduler latency $t_{\mathrm{sched}}$ of §13 above (once per request, not once per step). Symbols introduced in `decode.md §7.2`.

- $c_{\mathrm{serving}}$ — Per-sequence per-step CPU serving runtime constant (seconds/seq/step). Stack-dependent calibration knob: ≈10 µs/seq for aggressively fused C++ stacks, ≈20–30 µs/seq for production CUDA-Graph stacks (TensorRT-LLM, NVIDIA Dynamo), ≈30–60 µs/seq for Python-heavy stacks (vLLM, SGLang in eager mode) (`decode.md §7.2`).
- $t_{\mathrm{serving}}(B) = c_{\mathrm{serving}} \cdot B$ — Per-step serving runtime overhead. Additive to $t_{\mathrm{step,user}}(B)$, not overlapped with GPU work; sits outside the pipeline-bubble multiplier $\gamma_{\mathrm{pp}}$ since it fires once per step regardless of bubble depth (`decode.md §7.2`, used in `decode.md §7.3`).

When $c_{\mathrm{serving}} = 0$ (the default), $t_{\mathrm{serving}}(B) = 0$ and the §7.3 user-observed step time formula is recovered exactly — the term is opt-in.

---

## 20. B-Dependent Sustained Memory Bandwidth
_(→ decode.md §6.2)_

Captures the loss of HBM sustained / nameplate ratio as the active-sequence count $B$ grows: bank-conflict rate rises with concurrent KV address streams, memory-controller queues saturate, and paged-attention block-table updates crowd in over PCIe. Distinct from the per-tier hardware deflator $\eta_{\beta,i}$ of §16 (a static device property) and from the per-fabric switching-tier $\eta_\beta$ of §7 (a network-side runtime BW loss); the symbol below is the **memory-side** runtime BW curve as a function of the active-sequence count.

- $\eta_\beta(B)$ — Per-step effective HBM sustained / nameplate ratio as a function of the active-sequence count $B$ ($\in (0, 1]$). Piecewise-linear interpolation between anchor batch sizes; clamps to the boundary value below the smallest anchor and above the largest. Representative HBM3e anchor set on Blackwell-class production stacks: $\{1 \to 0.92,\; 64 \to 0.85,\; 512 \to 0.75,\; 4096 \to 0.55\}$ (`decode.md §6.2`).
- $BW_{\mathrm{eff}}(B) = BW_{\mathrm{mem,nameplate}} \cdot \eta_{\beta,\mathrm{tier}} \cdot \eta_\beta(B)$ — Composition rule. $\eta_\beta(B)$ multiplies on top of any per-tier $\eta_{\beta,i}$ from §16. In practice the analyst selects one of these (constant per-tier $\eta_{\beta,i}$ or B-dependent $\eta_\beta(B)$) to carry the sustained-vs-peak gap; composing both is supported but rarely necessary (`decode.md §6.2`).

When $\eta_\beta(B) \equiv 1$ the constant-bandwidth $t_{\mathrm{mem}}$ formula of `decode.md §4.3` is recovered exactly.

---

## 21. Attention Variants
_(→ attention.md)_

The `decode.md` and `prefill.md` cost formulas assume standard multi-head attention (MHA) or grouped-query attention (GQA). Models that depart from this baseline use variant-specific substitutions for the attention block's parameter count, KV cache footprint, traffic, and per-token compute. Per-variant symbols and equations are documented in `attention.md`; this section is the symbol-register pointer:

- **Multi-Head Attention (MHA)** (`attention.md §1`) — original transformer formulation; LLaMA-1, GPT-3. Standard symbols ($H$, $n_q$, $d_{\mathrm{head}}$) from §3 above; no extensions.
- **Grouped-Query Attention (GQA)** (`attention.md §2`) — LLaMA-3, Mistral, Qwen-2/3, most modern dense LLMs. Adds $n_{kv}$ (already listed in §3 above).
- **Multi-head Latent Attention (MLA)** (`attention.md §3`) — DeepSeek-V3 / R1, DeepSeek-V4-Pro, GLM-5, Kimi-K2.5. Symbols: $d_c$, $d_{q,c}$, $d_{qk,\mathrm{nope}}$, $d_{qk,\mathrm{rope}}$, $d_v$ (already listed in §3 above).
- **Sliding-window attention** (`attention.md §4`, placeholder) — Mistral, GPT-OSS, Gemma. Symbol: $W$ (per-token attention window).
- **DeepSeek Sparse Attention (DSA)** (`attention.md §5`, placeholder) — DeepSeek-V4-Pro, GLM-5. Symbol: $k_{\mathrm{attn}}$ (top-$k$ tokens attended).
- **Hybrid linear / full attention** (`attention.md §6`, placeholder) — Qwen-3.5, Jamba, Hymba. Symbol: per-layer `layer_type` selector.

When a model uses a non-MHA variant, the `decode.md` and `prefill.md` formulas for $P_{\mathrm{attn}}$, $M_{\mathrm{KV}}$, $T_{\mathrm{KV}}$, and $F_{\mathrm{attn}}$ carry inline references to the matching `attention.md` subsection.
