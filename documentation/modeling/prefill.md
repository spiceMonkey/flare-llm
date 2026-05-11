# Prefill and Time-To-First-Token (TTFT) Performance Model

**Modeling Compute-Bound Prefill, Batched Prefill, Chunked Prefill, and Disaggregated Architectures**

<br/>

**Author:** Yue Lu  
**Date:** April 2026

**Keywords:**  
LLM inference, prefill, TTFT, time-to-first-token, GEMM, compute-bound, memory-bound, roofline,  chunked prefill, batched prefill, disaggregated prefill, KV transfer, FlashAttention, pipeline warmup

---

Prefill processes the entire input sequence in a **single forward pass**, producing the KV cache entries for all input tokens and computing the first contextual representation. Unlike autoregressive decoding — where one token is processed per step and weight-reads dominate (GEMV, memory-bound) — prefill operates on $S_{\text{input}}$ tokens simultaneously, turning every linear layer into a matrix-matrix multiplication (GEMM) that scales with $S_{\text{input}}$. The resulting arithmetic intensity grows with sequence length, making prefill **compute-bound at typical input lengths** while decode remains perpetually memory-bound at batch size $B = 1$. Time To First Token (TTFT) encompasses three sequential phases: (1) the prefill pass itself, (2) optional KV-cache transfer from a prefill cluster to a decode cluster in disaggregated architectures, and (3) pipeline warmup latency as the first token traverses all PP stages. This document derives the full TTFT model from first principles, covering single-request, batched, chunked, and disaggregated scenarios.

---

<div style="page-break-after: always;"></div>

# Table of Contents

- [1. Prefill FLOPs](#1-prefill-flops)
  - [1.1 Projection FLOPs (Q/K/V/O)](#11-projection-flops-qkvo)
  - [1.2 Attention Score and Value FLOPs](#12-attention-score-and-value-flops)
  - [1.3 FFN FLOPs](#13-ffn-flops)
  - [1.4 Per-Layer Total and FlashAttention Note](#14-per-layer-total-and-flashattention-note)
  - [1.5 Per-Device Total FLOPs Under Parallelism](#15-per-device-total-flops-under-parallelism)

- [2. Compute vs. Memory Bound: Prefill vs. Decode](#2-compute-vs-memory-bound-prefill-vs-decode)
  - [2.1 Arithmetic Intensity of Prefill](#21-arithmetic-intensity-of-prefill)
  - [2.2 Arithmetic Intensity of Decode](#22-arithmetic-intensity-of-decode)
  - [2.3 Ridge Point and Regime Crossover](#23-ridge-point-and-regime-crossover)

- [3. Hardware Prefill Latency](#3-hardware-prefill-latency)
  - [3.1 Prefill Local Time (Roofline)](#31-prefill-local-time-roofline)
  - [3.2 Communication During Prefill](#32-communication-during-prefill)
  - [3.3 Pipeline Warmup Latency](#33-pipeline-warmup-latency)
  - [3.4 Hardware Prefill Latency Formula](#34-hardware-prefill-latency-formula)

- [4. Optimal Batch Size for GPU Utilization](#4-optimal-batch-size-for-gpu-utilization)

- [5. Chunked Prefill](#5-chunked-prefill)
  - [5.1 Per-Chunk Latency](#51-per-chunk-latency)
  - [5.2 Total Prefill Time Under Chunking](#52-total-prefill-time-under-chunking)
  - [5.3 Throughput Benefit and Head-of-Line Blocking Reduction](#53-throughput-benefit-and-head-of-line-blocking-reduction)

- [6. Prefill→Decode KV Handoff](#6-prefill-decode-kv-handoff)
  - [6.1 Architecture Variants](#61-architecture-variants)
  - [6.2 KV Cache Volume (shared)](#62-kv-cache-volume-shared)
  - [6.3 Co-located Handoff: KV Layout Transition](#63-co-located-handoff-kv-layout-transition)
  - [6.4 Disaggregated Handoff: Inter-Cluster KV Transfer](#64-disaggregated-handoff-inter-cluster-kv-transfer)
  - [6.5 Hardware Prefill Latency (unified)](#65-hardware-prefill-latency-unified)

---

<div style="page-break-before: always;"></div>

# 1. Prefill FLOPs

During decoding a single token is processed per step; all weight multiplications are **matrix–vector** (GEMV) operations. During prefill all $S_{\text{input}}$ tokens are processed in one forward pass; every linear layer becomes a **matrix–matrix** (GEMM) operation whose FLOP count scales with $S_{\text{input}}$.

**Notation used in this section** (see `notation.md §§3–4, 8, 11` for the full table):

| Symbol | Meaning |
|--------|---------|
| $S_{\text{input}}$ | Input sequence length (number of tokens in the prefill batch) |
| $H$ | Hidden size (model dimension) |
| $H_{kv} = n_{kv} \cdot d_{\text{head}}$ | Total KV projection dimension (GQA; [GQA]) |
| $I_{\text{eff}}$ | Effective FFN intermediate size ($I_{\text{dense}}$ for dense; $k \cdot I_{\text{moe}}$ for MoE) |
| $L$ | Total number of transformer layers |
| $PP, TP, EP, SP$ | Pipeline, tensor, expert, sequence parallelism degrees |

All counts follow the **multiply-accumulate = 2 FLOPs** convention (one multiply + one add per MAC).

---

## 1.1 Projection FLOPs (Q/K/V/O)

For decoding a single token, each projection matrix multiplies a vector of length $H$ (or $H_{kv}$) against the weight matrix.
For prefill, the operation is a GEMM of shape $[S_{\text{input}} \times H] \times [H \times H_{\text{out}}]$, contributing $2 \cdot S_{\text{input}} \cdot H \cdot H_{\text{out}}$ FLOPs.

The four projection matrices and their output dimensions:

| Projection | Weight shape | Output dim | FLOPs |
|------------|-------------|------------|-------|
| $W_Q$ | $H \times H$ | $H$ | $2 H^2 S_{\text{input}}$ |
| $W_K$ | $H \times H_{kv}$ | $H_{kv}$ | $2 H H_{kv} S_{\text{input}}$ |
| $W_V$ | $H \times H_{kv}$ | $H_{kv}$ | $2 H H_{kv} S_{\text{input}}$ |
| $W_O$ | $H \times H$ | $H$ | $2 H^2 S_{\text{input}}$ |

Summing:

$$
F_{\text{proj,prefill}} =
(4H^2 + 4 H H_{kv}) \cdot S_{\text{input}}
$$

This matches the decode projection formula multiplied by $S_{\text{input}}$, reflecting the transition from GEMV to GEMM.

> **Variant note.** Multi-head Latent Attention (MLA) replaces the projection FLOPs with a six-matrix decomposition over compressed Q and KV latents. Per-token compute under both materialized and absorbed execution modes is given in `attention.md §3.7`; KV cache write per prefill token is $(d_c + d_{qk,\mathrm{rope}}) \cdot b$ instead of $2 H_{kv} \cdot b$.

---

## 1.2 Attention Score and Value FLOPs

### Attention scores ($QK^\top$)

During prefill, the full causal attention matrix is computed. Each of the $n_q$ query heads independently computes dot products with its corresponding (broadcast) KV head across all $S_{\text{input}}$ positions. Per query head: $2 d_{\text{head}} S_{\text{input}}^2$ FLOPs. Summed over all $n_q$ query heads:

$$
F_{\text{score,prefill}} =
2 \cdot S_{\text{input}}^2 \cdot H
$$

> **Note:** The factor of 2 accounts for both the multiply and accumulate in each dot product. The causal mask halves the *average* work, but practical implementations (FlashAttention tile loops) often process the full upper-right with a mask, and published FLOPs budgets consistently use the full $S^2$ form [FA1, FA2]. We adopt the same convention.

> **GQA note:** In GQA ($n_{kv} < n_q$), each KV head is shared by $n_q / n_{kv}$ query heads [GQA]. The KV cache **memory** scales with $H_{kv}$ (only $n_{kv}$ unique heads are stored), but the attention **FLOPs** scale with $H = n_q d_{\text{head}}$ because every query head independently computes attention.

### Value application (Attn · V)

After softmax, each of the $n_q$ query heads computes a weighted sum over $S_{\text{input}}$ cached values of its corresponding value head. Per head: $2 d_{\text{head}} S_{\text{input}}^2$ FLOPs. Total:

$$
F_{\text{value,prefill}} =
2 \cdot S_{\text{input}}^2 \cdot H
$$

### Combined attention KV FLOPs

$$
F_{\text{attn,KV,prefill}} =
F_{\text{score,prefill}} + F_{\text{value,prefill}} =
4 \cdot S_{\text{input}}^2 \cdot H
$$

---

## 1.3 FFN FLOPs

For a gated MLP (SwiGLU/GeGLU style) with three weight matrices (gate, up, down) and effective intermediate dimension $I_{\text{eff}}$:

$$
F_{\text{ffn,prefill}} =
6 \cdot H \cdot I_{\text{eff}} \cdot S_{\text{input}}
$$

> **Convention note:** A gated MLP has three full GEMMs — gate ($H \to I$), up ($H \to I$), and down ($I \to H$) — each contributing $2HIS_{\text{input}}$ FLOPs, for a total of $6HI_{\text{eff}}S_{\text{input}}$. The elementwise activation (SiLU/GeLU) and gate multiply are $O(I)$ and genuinely negligible. The training scaling-law literature ([KAPLAN-SCALING], [CHINCHILLA]) uses $4HI$, corresponding to a non-gated 2-matrix FFN. Since this model targets **inference** of modern gated-MLP architectures, we use the exact $6HI$ throughout. This ensures accurate hardware prefill latency predictions in the compute-bound regime and yields the exact OI result $\text{OI} = 2S_{\text{input}}/b$ without approximation (since FLOPs $= 2 \times$ params for every weight matrix). See also `decode.md §3.3`.

---

## 1.3.5 MoE Router Gate FLOPs

For MoE layers, each token first passes through a lightweight **routing gate** — a small $H \to N_{\text{exp}}$ GEMM that produces one logit per expert. The top-$k$ experts are then selected by argmax over this gate vector, and only their FFN weights run on each token. The gate itself contributes FLOPs independent of $k$:

$$
F_{\text{router,prefill}} =
2 \cdot H \cdot N_{\text{exp}} \cdot S_{\text{input}}
$$

The router is **replicated, not sharded** across TP ranks: the gate matrix $H \times N_{\text{exp}}$ is tiny (e.g., $20480 \times 16 \approx 330\text{k}$ weights on GPT-1.8T) and sharding it would cost more in synchronization than it saves in FLOPs. Each TP rank redundantly computes the full gate. This term is **zero for dense layers** and contributes a small fraction of total FLOPs in practice (0.1–0.5% on production MoE models), but must be included for consistency with decode (see `decode.md §3.3`, "MoE FFN FLOPs") and for correct arithmetic intensity accounting.

---

## 1.4 Per-Layer Total and FlashAttention Note

Combining projections, attention KV, and FFN per transformer layer. Dense and MoE layers differ in their FFN term and in whether they include router FLOPs:

$$
F_{\text{layer,prefill}}^{\text{dense}} =
F_{\text{proj,prefill}} + F_{\text{attn,KV,prefill}} + F_{\text{ffn,prefill}}^{\text{dense}} =
(4H^2 + 4 H H_{kv} + 6 H I_{\text{dense}}) S_{\text{input}}
\;+\;
4 S_{\text{input}}^2 H
$$

$$
F_{\text{layer,prefill}}^{\text{MoE}} =
F_{\text{proj,prefill}} + F_{\text{attn,KV,prefill}} + F_{\text{ffn,prefill}}^{\text{MoE}} + F_{\text{router,prefill}} =
(4H^2 + 4 H H_{kv} + 6 H k I_{\text{moe}} + 2 H N_{\text{exp}}) S_{\text{input}}
\;+\;
4 S_{\text{input}}^2 H
$$

The first term grows as $O(S_{\text{input}})$ and corresponds to GEMM operations on the weight matrices (including the MoE router). The second term grows as $O(S_{\text{input}}^2)$ and comes from the full pairwise attention computation.

### Regime crossover

The $S_{\text{input}}^2$ term dominates when:

$$
4 S_{\text{input}}^2 H >
(4H^2 + 4 H H_{kv} + 6 H I_{\text{eff}}) S_{\text{input}}
$$

Simplifying:

$$
S_{\text{input}} >
\frac{4H^2 + 4 H H_{kv} + 6 H I_{\text{eff}}}{4 H} =
H + H_{kv} + \tfrac{3}{2} I_{\text{eff}}
$$

For a model with $H = 8192$, $H_{kv} = 1024$, and $I_{\text{eff}} = 28672$ (e.g., LLaMA-3 70B with GQA $n_{kv} = 8$, $d_{\text{head}} = 128$), the crossover is at $S_{\text{input}} \approx 52\text{k}$ tokens. For smaller models (e.g., $H = 4096$, $H_{kv} = 512$, $I_{\text{eff}} = 14336$), crossover occurs at $\sim 26\text{k}$ tokens. At common prefill lengths ($\le 8\text{k}$ tokens), the linear projection term typically dominates.

### FlashAttention does NOT reduce prefill FLOPs

FlashAttention [FA1, FA2] tiles the $S \times S$ attention matrix into SRAM-resident blocks, reducing the number of HBM round-trips and thus **memory traffic**. However, the number of floating-point operations remains identical — FlashAttention is an IO-aware reorganization of the same computation, not a FLOP reduction. The $4 S_{\text{input}}^2 H$ term applies with or without FlashAttention; the difference appears only in the memory traffic model (Section 3).

---

## 1.5 Per-Device Total FLOPs Under Parallelism

Parallelism distributes prefill FLOPs across devices following the same canonical nesting as decode (`DP → PP → EP → TP → SP` under the orthogonal layout) [MEGATRON, MEGATRON3]. The same per-component effective sharding factors $D_{\text{attn}}$, $D_{\text{exp}}$, $D_{\text{kv}}$, $D_{\text{emb}}$ from `notation.md §1` apply here so a single expression covers all three production-relevant configurations. Resolving the abstract factors:

| configuration | $D_{\text{attn}}$ | $D_{\text{exp}}$ (MoE) | $D_{\text{kv}}$ | $D_{\text{emb}}$ |
|---|---|---|---|---|
| orthogonal + TP-attn | $TP$ | $TP \cdot EP$ | $TP$ (head) | $TP$ |
| orthogonal + DP-attn | $1$ | $TP \cdot EP$ | $TP$ (seq) | $TP$ |
| co-located + DP-attn | $1$ | $EP$ | $\max(TP, EP)$ (seq) | $TP$ |

Dense FFN always uses $D_{\text{exp}} = TP$. The KV-attention compute term carries an additional $/SP$ factor on top of $D_{\text{kv}}$ when sequence parallelism is enabled. Prefill per-device FLOPs are **invariant** under the TP-attn ↔ DP-attn swap (the per-rank work is the same whether split by head or by sequence — both reduce to a $1/D_{\text{attn}}$ or $1/D_{\text{kv}}$ share); attention weight footprint and the per-layer attention TP collective primitive (AR vs AG) change as derived in `decode.md §1.4` / `§5.3`. Formulas below use the abstract factors directly without further repetition of the table.

### PP sharding

Pipeline parallelism assigns $L/PP$ layers to each stage. Since the prefill pass traverses all layers, each PP stage handles exactly its local layers:

$$
F_{\text{layer,prefill,device}} = \frac{L}{PP} \cdot F_{\text{layer,prefill}}^{\text{sharded by D-factors and SP}}
$$

### Tensor sharding (projections, FFN GEMMs)

The Q/K/V/O projection FLOPs follow the attention divisor; the FFN GEMMs follow the expert divisor:

- Projections: $(4H^2 + 4 H H_{kv}) S_{\text{input}} / D_{\text{attn}}$
- FFN: $6 H I_{\text{eff}} S_{\text{input}} / D_{\text{exp}}$

### KV-attention sharding

The per-token attention score / value FLOPs scale as $4 S_{\text{input}}^2 H$ per layer. They are sharded by the same divisor that shards the KV cache itself ($D_{\text{kv}}$ — head shard under TP-attn or sequence shard under DP-attn). Sequence parallelism (SP) further partitions the $S_{\text{input}}$ query positions across SP ranks (each SP rank computes attention over $S_{\text{input}}/SP$ queries via ring KV passing):

$$
F_{\text{attn,KV,prefill,device}} = \frac{4 S_{\text{input}}^2 H}{D_{\text{kv}} \cdot SP}
$$

### Expert sharding (MoE only)

For MoE layers, expert weights are sharded by $D_{\text{exp}}$ (the abstract divisor encodes the orthogonal $TP \cdot EP$ shard or the co-located $EP$-only shard):

$$
F_{\text{ffn,prefill,device}}^{\text{MoE}} = \frac{6 H k I_{\text{moe}} S_{\text{input}}}{D_{\text{exp}}}
$$

### Full per-device expression

Collecting all terms, separated by layer type. Let $L_{\text{dense}}$ and $L_{\text{moe}}$ be the number of dense and MoE layers (with $L_{\text{dense}} + L_{\text{moe}} = L$):

$$
F_{\text{prefill,device}} =
\frac{L_{\text{dense}}}{PP}
\underbrace{\left[
\frac{(4H^2 + 4 H H_{kv}) S_{\text{input}}}{D_{\text{attn}}} +
\frac{6 H I_{\text{dense}} S_{\text{input}}}{TP} +
\frac{4 S_{\text{input}}^2 H}{D_{\text{kv}} \cdot SP}
\right]}_{\text{dense layer}}
\;+\;
\frac{L_{\text{moe}}}{PP}
\underbrace{\left[
\frac{(4H^2 + 4 H H_{kv}) S_{\text{input}}}{D_{\text{attn}}} +
\frac{6 H k I_{\text{moe}} S_{\text{input}}}{D_{\text{exp}}} +
\frac{4 S_{\text{input}}^2 H}{D_{\text{kv}} \cdot SP} +
2 H N_{\text{exp}} S_{\text{input}}
\right]}_{\text{MoE layer (router unsharded)}}
$$

For a **pure dense model** ($L_{\text{moe}} = 0$, $EP = 1$, so $D_{\text{exp}} = TP$):

$$
F_{\text{prefill,device}}^{\text{dense}} =
\frac{L}{PP}
\left[
\frac{(4H^2 + 4 H H_{kv}) S_{\text{input}}}{D_{\text{attn}}} +
\frac{6 H I_{\text{dense}} S_{\text{input}}}{TP} +
\frac{4 S_{\text{input}}^2 H}{D_{\text{kv}} \cdot SP}
\right]
$$

For a **pure MoE model** ($L_{\text{dense}} = 0$), the router term $2 H N_{\text{exp}} S_{\text{input}}$ stays unsharded — it is replicated on every TP rank (§1.3.5).

### Per-step (batched) FLOPs

A batched prefill pass concurrently processes $B_{\text{prefill}}$ requests, each carrying $S_{\text{input}}$ tokens. Because every request runs the same forward pass independently — no cross-request attention — FLOPs scale linearly with $B_{\text{prefill}}$:

$$
F_{\text{prefill,step,device}}(B_{\text{prefill}}) \;=\; B_{\text{prefill}} \cdot F_{\text{prefill,device}}
$$

Note the attention term scales as $B_{\text{prefill}} \cdot S_{\text{input}}^2$ (each request's $S^2$ independently), **not** $(B_{\text{prefill}} \cdot S_{\text{input}})^2$ — there is no quadratic cross-request term.

This is the per-step, per-device FLOP count consumed in the §3 roofline. All downstream HW latency formulas (§3, §4) carry the $(B_{\text{prefill}})$ argument explicitly.

### LM head FLOPs (prefill)

To emit the first generated token of each prefilled request, the model runs the LM head ($H \to V$ projection) on the **last position's hidden state only** — the intermediate positions' logits are not needed by the sampler. The LM head is sharded by $D_{\text{emb}}$ along the vocab dimension and **lives only on the last PP stage** (stage $PP{-}1$); it is not divided by $L$, $PP$, $EP$, or $SP$, mirroring the decode treatment in `decode.md §3`.

$$
F_{\text{LM,prefill,step,device}}(B_{\text{prefill}}) = \frac{2 \, B_{\text{prefill}} \, H \, V}{D_{\text{emb}}} \quad \text{(stage } PP{-}1 \text{ only)}
$$

**Scaling note.** Unlike the prefill body — which scales with $B_{\text{prefill}} \cdot S_{\text{input}}$ — the LM head scales with **only $B_{\text{prefill}}$**, since one $H \to V$ projection per request suffices. For typical $S_{\text{input}} \gtrsim 100$, the LM head is well under 1% of body FLOPs and is included for symmetry with decode rather than because it materially shifts TTFT.

LM head weight + output traffic on stage $PP{-}1$ (mirroring decode's $T_{\text{LM},\theta,\text{device}}$):

$$
T_{\text{LM,prefill,step,device}}(B_{\text{prefill}}) = \frac{H V \, b}{D_{\text{emb}}} + B_{\text{prefill}} \, V \, b \quad \text{(stage } PP{-}1 \text{ only)}
$$

The first term is the $D_{\text{emb}}$-sharded weight read; the second is the $B_{\text{prefill}}$ logit rows written to HBM. Both terms feed the LM head roofline introduced in §3.4.

---

<div style="page-break-before: always;"></div>

# 2. Compute vs. Memory Bound: Prefill vs. Decode

A fundamental insight into LLM inference performance is that **prefill and decode lie in different hardware regimes** — prefill is (typically) compute-bound, decode is (always) memory-bound at $B = 1$. This section derives the arithmetic intensities for both phases and identifies the crossover condition, following the roofline framework [ROOFLINE].

The **ridge point** of a device is:

$$
R_{\text{ridge}} =
\frac{R_{\text{GPU}}}{BW_{\text{mem}}}
\quad (\text{FLOPs per byte})
$$

A kernel is compute-bound if $\text{OI} > R_{\text{ridge}}$, and memory-bound otherwise.

---

## 2.1 Arithmetic Intensity of Prefill

Arithmetic intensity (OI) is the ratio of FLOPs to bytes moved from HBM during the computation.

### Weight-dominated regime (linear projections)

For the projection and FFN GEMMs, the weight matrices are read once per prefill pass. The weight traffic per layer is $T_{\theta,\text{layer}} = (P_{\text{attn}} / D_{\text{attn}} + P_{\text{FFN}} / D_{\text{exp}}) \cdot b$ bytes. The projection and FFN FLOPs are $\bigl((4H^2 + 4 H H_{kv})/D_{\text{attn}} + 6 H I_{\text{eff}} / D_{\text{exp}}\bigr) \cdot S_{\text{input}}$.

Since every weight matrix contributes FLOPs $= 2 \times$ params (the $6HI$ convention matches the $3HI$ parameter count exactly), the per-component divisors cancel and the OI is layout-invariant:

$$
\text{OI}_{\text{prefill,proj}} =
\frac{\bigl((4H^2 + 4 H H_{kv})/D_{\text{attn}} + 6 H I_{\text{eff}} / D_{\text{exp}}\bigr) S_{\text{input}}}{\bigl((2H^2 + 2 H H_{kv})/D_{\text{attn}} + 3 H I_{\text{eff}} N_{\text{exp}}/D_{\text{exp}}\bigr)\, b} =
\frac{2 S_{\text{input}}}{b}
$$

The arithmetic intensity of the linear projection stage **scales linearly with $S_{\text{input}}$** — each weight byte is reused across all $S_{\text{input}}$ token vectors in the GEMM.

### Attention computation regime

For the attention score and value computation, the "weights" are the $Q$, $K$, $V$ activations themselves (all loaded from HBM or kept in SRAM with FlashAttention). Without FlashAttention, the traffic is $O(S_{\text{input}}^2)$ (storing the full attention matrix); with FlashAttention [FA1, FA2], the $S \times S$ matrix is tiled and never fully materialized, reducing traffic to $O(S_{\text{input}})$. In both cases the FLOPs are $4 S_{\text{input}}^2 H / (D_{\text{kv}} \cdot SP)$.

For performance modeling, we treat FlashAttention as the default and note that the attention computation is generally compute-bound in the prefill regime (the tiled SRAM reuse pattern achieves high OI within each tile).

### Summary

For the projection- and FFN-dominated components of prefill (the $O(S_{\text{input}})$ FLOP terms), at the per-request baseline ($B_{\text{prefill}} = 1$, weights dominate the traffic denominator):

$$
\text{OI}_{\text{prefill}}(B_{\text{prefill}} = 1)
\approx
\frac{2 S_{\text{input}}}{b}
$$

For batched prefill in the weight-dominated regime, OI scales linearly with $B_{\text{prefill}}$ (each weight byte serves $B_{\text{prefill}} \cdot S_{\text{input}}$ tokens):

$$
\text{OI}_{\text{prefill}}(B_{\text{prefill}})
\approx
\frac{2 B_{\text{prefill}} \cdot S_{\text{input}}}{b}
\qquad (\text{weight-dominated})
$$

In the KV-write-dominated regime (very long $S$), OI saturates at the per-token attention ratio independent of $B_{\text{prefill}}$.

---

## 2.2 Arithmetic Intensity of Decode

During decode at batch size $B = 1$, each linear layer is a GEMV: one vector of length $H$ (or $H_{kv}$) is multiplied against a weight matrix. The weight matrix is read in full ($\sim H^2 b$ bytes per layer), yielding $\sim 2 H^2$ FLOPs. The OI of a weight-matrix GEMV is:

$$
\text{OI}_{\text{decode,GEMV}}
\approx
\frac{2 H^2}{H^2\, b} =
\frac{2}{b}
$$

At batch size $B > 1$ (batched decode), each weight column is reused across $B$ output tokens:

$$
\text{OI}_{\text{decode}}
\approx
\frac{2 B}{b}
$$

For $B = 1$ and $b = 2$ (bf16), $\text{OI}_{\text{decode}} = 1$ FLOP/byte — far below any modern GPU's ridge point.

$$
\text{OI}_{\text{decode}} \approx \frac{2B}{b}
\quad\Longrightarrow\quad
\textbf{always memory-bound at }B = 1
$$

---

## 2.3 Ridge Point and Regime Crossover

For an H100 SXM5 [H100-SPEC]:

- $R_{\text{GPU}} \approx 989$ TFlops/s (bf16 with Tensor Cores)
- $BW_{\text{mem}} \approx 3.35$ TB/s (HBM3)
- $R_{\text{ridge}} = R_{\text{GPU}} / BW_{\text{mem}} \approx 295$ FLOPs/byte

Comparing with OI expressions (bf16, $b = 2$):

| Phase | OI expression | Regime |
|-------|--------------|--------|
| Decode ($B=1$) | $2/b = 1$ | Memory-bound ($1 \ll 295$) |
| Decode ($B$) | $2B/b$ | Compute-bound when $B > 295$ |
| Prefill ($B_{\text{prefill}} = 1$) | $2 S_{\text{input}}/b$ | Compute-bound when $S_{\text{input}} > 295$ |
| Prefill ($B_{\text{prefill}}$) | $2 B_{\text{prefill}} \cdot S_{\text{input}}/b$ | Compute-bound when $B_{\text{prefill}} \cdot S_{\text{input}} > 295$ |

**Prefill becomes compute-bound for $B_{\text{prefill}} \cdot S_{\text{input}} \gtrsim 295$ tokens** on H100 with bf16. At single-request prefill, this happens around $S_{\text{input}} = 295$; batching short prompts can push much shorter sequences into the compute-bound zone (see §4).

This is the crossover condition:

$$
S_{\text{input}}^{\star} =
\frac{b}{2} \cdot R_{\text{ridge}} =
\frac{b}{2} \cdot \frac{R_{\text{GPU}}}{BW_{\text{mem}}}
$$

In practice, prefill sequences are almost always longer than $S_{\text{input}}^{\star}$, so the prefill pass is **virtually always compute-bound** at single-request inference on modern accelerators. The exact threshold varies with precision and HBM bandwidth; on A100 (HBM2E, $R_{\text{ridge}} \approx 156$ FLOPs/byte), $S_{\text{input}}^{\star} \approx 156$ tokens with bf16.

---

<div style="page-break-before: always;"></div>

# 3. Hardware Prefill Latency

We now derive the hardware prefill latency $t_{\text{prefill}}(B_{\text{prefill}})$ for a batched prefill pass on a co-located prefill+decode cluster (no disaggregation). The single-request case is recovered at $B_{\text{prefill}} = 1$.

> **Scope note:** This section computes the hardware-only prefill latency $t_{\text{prefill}}(B_{\text{prefill}})$, which is one component of the full Time-To-First-Token (TTFT). The complete TTFT additionally includes scheduling overhead $t_{\text{sched}}$, tokenization $t_{\text{tok}}$, and the first decode step $t_{\text{step,user}}$. See `e2e.md` §2.1 for full TTFT assembly.

$t_{\text{prefill}}(B_{\text{prefill}})$ decomposes into three sequential phases:

$$
t_{\text{prefill}}(B_{\text{prefill}}) = t_{\text{prefill,local}}(B_{\text{prefill}}) + \max\!\bigl(0,\; t_{\text{prefill,comm}}(B_{\text{prefill}}) - \rho\, t_{\text{prefill,local}}(B_{\text{prefill}})\bigr) + t_{\text{pipeline,warmup}}(B_{\text{prefill}})
$$

Each term is derived below. The structure mirrors decode's per-step roofline; the differences are (i) FLOPs scale with $B_{\text{prefill}} \cdot S_{\text{input}}$ instead of just $B$, (ii) the KV cache is *written* (not read) during prefill, so the per-step memory term has a $B_{\text{prefill}} \cdot T_{\text{KV,write,device}}$ component; (iii) the pipeline must fill from empty, contributing a $(PP-1)$ stage-time warmup.

---

## 3.1 Prefill Local Time (Roofline)

The per-step local prefill time on a single PP stage is the roofline of compute and memory time, both functions of $B_{\text{prefill}}$.

### Compute time

Given $F_{\text{prefill,step,device}}(B_{\text{prefill}}) = B_{\text{prefill}} \cdot F_{\text{prefill,device}}$ from Section 1.5:

$$
t_{\text{prefill,compute}}(B_{\text{prefill}}) =
\frac{F_{\text{prefill,step,device}}(B_{\text{prefill}})}{R_{\text{GPU}}(b)} =
\frac{B_{\text{prefill}} \cdot F_{\text{prefill,device}}}{R_{\text{GPU}}(b)}
$$

Compute scales linearly with $B_{\text{prefill}}$ — every batched request runs the same forward pass independently. $R_{\text{GPU}}(b)$ is the **precision-aware compute peak**: the framework convention is to store FP16 dense per chip in `peak_flops_TF` and scale linearly with `bytes_per_param`: $R_{\text{GPU}}(b) = \text{peak\_flops\_TF} \cdot (2 / b)$. See `decode.md §4` for the full convention and the d-Matrix INT4 caveat.

### Memory time

During prefill, weights are read from HBM (same as decode, amortized once per pass regardless of $B_{\text{prefill}}$). However, the **KV cache is being written** (not read) for the input tokens — it is not yet present and therefore contributes **write traffic**, not read traffic, scaling per-request with $B_{\text{prefill}}$:

$$
T_{\text{prefill,step,device}}(B_{\text{prefill}}) =
T_{\theta,\text{device}} +
B_{\text{prefill}} \cdot T_{\text{KV,write,device}}
$$

where the two components are:

**Weight read traffic** $T_{\theta,\text{device}}$ — same as decode (see `decode.md §2.1`), $B_{\text{prefill}}$-independent:

$$
T_{\theta,\text{device}}
\approx
\frac{L}{PP}
\left(
\frac{(2H^2 + 2 H H_{kv})b}{D_{\text{attn}}} +
\frac{3 H I_{\text{eff}} N_{\text{exp}} b}{D_{\text{exp}}}
\right)
$$

**Per-request KV cache write traffic** $T_{\text{KV,write,device}}$ — for the $S_{\text{input}}$ tokens being prefilled by one request:

$$
T_{\text{KV,write,device}} =
\frac{L}{PP}
\cdot
\frac{2 S_{\text{input}} H_{kv}\, b}{D_{\text{kv}} \cdot SP}
\quad \text{(per request)}
$$

This is the cost of writing the keys and values for all $S_{\text{input}}$ tokens to HBM for one request; with $B_{\text{prefill}}$ batched requests, total per-step write traffic is $B_{\text{prefill}}$ times this. Because HBM write bandwidth equals read bandwidth, we use $BW_{\text{mem}}$ for both.

### FlashAttention reduces attention read traffic

Without FlashAttention, the $S_{\text{input}} \times S_{\text{input}}$ attention score matrix would need to be written to and read from HBM, contributing $O(S_{\text{input}}^2)$ traffic per request. With FlashAttention [FA1, FA2], the attention computation is fused into SRAM-tiled blocks, and the $S \times S$ matrix is never materialized in HBM. This eliminates the dominant attention traffic term.

The effective per-step memory time is:

$$
t_{\text{prefill,mem}}(B_{\text{prefill}}) =
\frac{T_{\text{prefill,step,device}}(B_{\text{prefill}})}{BW_{\text{mem}}} =
\frac{T_{\theta,\text{device}} + B_{\text{prefill}} \cdot T_{\text{KV,write,device}}}{BW_{\text{mem}}}
$$

### Roofline local time

$$
t_{\text{prefill,local}}(B_{\text{prefill}}) =
\max\bigl(
t_{\text{prefill,compute}}(B_{\text{prefill}}),\;
t_{\text{prefill,mem}}(B_{\text{prefill}})
\bigr)
$$

At typical prefill lengths and modest $B_{\text{prefill}}$ (compute-bound regime, Section 2.3): $t_{\text{prefill,compute}} \gg t_{\text{prefill,mem}}$, so $t_{\text{prefill,local}}(B_{\text{prefill}}) \approx t_{\text{prefill,compute}}(B_{\text{prefill}}) = B_{\text{prefill}} \cdot t_{\text{prefill,compute}}(1)$ — linear in $B_{\text{prefill}}$.

---

## 3.2 Communication During Prefill

The communication collectives required during prefill are structurally the same as during decode (same TP/EP/SP/PP operations per layer), but **message sizes scale with the per-step token count** $B_{\text{prefill}} \cdot S_{\text{input}}$ — each batched request contributes $S_{\text{input}}$ tokens to every collective payload. Each per-collective $t_*(B_{\text{prefill}})$ below carries the $(B_{\text{prefill}})$ argument explicitly: α-side stays constant, β-side scales linearly with $B_{\text{prefill}} \cdot S_{\text{input}}$.

All collective latencies follow the $\alpha$–$\beta$ model [ALPHA-BETA]:

$$
t = \alpha + \frac{\text{message size}}{B_{\text{eff}}}
$$

**Delegation to the `collectives/` explainer subseries.** Shipped-primitive cost formulas (ring AR, DBT AR, ring AG / RS, pairwise A2A on star; dim-decomposed ring and bisection-bound A2A on torus; hierarchical RS → sub-AR → AG; in-network reduction via NVLS / Quantum SHARP / Tomahawk Ultra) are documented in `collectives/01_collective_algorithms.md` (per-algorithm derivations) and `collectives/02_topology_mapping.md` (star / torus / mesh specializations), with hierarchical composition in `collectives/03_hierarchical_topologies.md`, in-network primitives in `collectives/04_in_network_collectives.md`, and contention coefficients $(\eta_\alpha, \eta_\beta)$ in `collectives/05_contention_and_congestion.md`. The cheatsheet at `collectives/00_summary.md` indexes all of these. This section substitutes the prefill per-rank message sizes (scaled by $B_{\text{prefill}} \cdot S_{\text{input}}$) into those primitives; the $\alpha_{XP}$ and $BW_{XP}$ values are fabric-chain span quantities per `notation.md §7`.

### TP collectives (prefill)

The per-layer TP collective synchronizes the partial hidden-state outputs after Row-Parallel matrix multiplications (the **all-reduce (AR)** under TP-attention or under any FFN block) or redistributes per-rank sequence-shards to the TP-sharded form expected by the downstream FFN (the **all-gather (AG)** at the attention → FFN boundary under DP-attention). During prefill, the post-projection output has shape $[B_{\text{prefill}} \cdot S_{\text{input}} \times H]$, so the per-call logical message size is $M_{TP}^{\text{prefill}}(B_{\text{prefill}}) = B_{\text{prefill}} \cdot S_{\text{input}} \cdot H \cdot b$ (vs. $H \cdot b$ during single-token decode). The collective group size is $G_{TP}$ from `notation.md §1` (equal to $TP$ across all three configurations). NCCL ships ring (large-$M$) and DBT (small-$M$) variants for both AR and AG on a star fabric, selected via `tuner.ar_algorithm` (`collectives/02_topology_mapping.md §2`). The per-call costs:

$$
t_{TP,\text{AR}}^{\text{prefill,ring}}(B_{\text{prefill}}) \;=\; 2(G_{TP}-1)\,\alpha_{TP} \;+\; 2 \cdot \frac{G_{TP}-1}{G_{TP}} \cdot \frac{B_{\text{prefill}} \cdot S_{\text{input}} \cdot H \cdot b}{BW_{\text{TP}}}
$$

$$
t_{TP,\text{AG}}^{\text{prefill,ring}}(B_{\text{prefill}}) \;=\; (G_{TP}-1)\,\alpha_{TP} \;+\; \frac{G_{TP}-1}{G_{TP}} \cdot \frac{B_{\text{prefill}} \cdot S_{\text{input}} \cdot H \cdot b}{BW_{\text{TP}}}
$$

The AG cost is half the AR cost (one ring pass instead of the AR's two: reduce-scatter then all-gather). The DBT variants substitute $2\lceil \log_2 G_{TP}\rceil \cdot \alpha_{TP}$ (AR) or $\lceil \log_2 G_{TP}\rceil \cdot \alpha_{TP}$ (AG) for the $\alpha$-side coefficient; the $\beta$-side terms are unchanged. Torus TP fabrics (dim-decomposed ring, TPU / Trainium) use the torus AR / AG forms of `collectives/02_topology_mapping.md §3` with the same $M_{TP}^{\text{prefill}}(B_{\text{prefill}})$. The $B_{\text{prefill}} \cdot S_{\text{input}}$ factor in the bandwidth term means TP communication during prefill is substantially larger than during decode — for $S_{\text{input}} = 4096$ and $B_{\text{prefill}} = 4$, the per-collective payload is $\sim 16{,}000\times$ that of single-token decode.

The per-layer TP cost depends on attention mode: under TP-attention the attention output fires an AR; under DP-attention it fires an AG. The FFN / MoE expert output projection always fires an AR (TP-sharded under both modes). The per-layer aggregate is assembled in the per-stage budget below (§3.2 "Total per-stage communication").

> **Implementation note — tiled prefill and $\alpha_{TP}$ accumulation:** In practice, large-$S_{\text{input}}$ prefill is often processed in $k$ sub-sequence tiles (e.g., to fit within SRAM or network buffer limits). Each tile launches an independent all-reduce, accumulating the $\alpha_{TP}$ startup latency $k$ times. The total un-hidden $\alpha$ overhead is $k \times \max(0,\, \alpha_{TP} - \rho \cdot t_{\text{tile,compute}})$, where $t_{\text{tile,compute}}$ is the compute time for a single tile. For fine-grained tiling with small tiles, each tile's compute-to-communication ratio mirrors the full-sequence overlap structure, so the $\rho$ factor still absorbs the hiding benefit. However, for very large $S_{\text{input}}$ and small tile sizes, the accumulated $k \cdot \alpha_{TP}$ term can become non-negligible even when each individual $\alpha_{TP}$ is fully hidden. The formulas above model a single collective per layer; the tiling multiplier $k$ can be incorporated when tile size is a known design parameter.

### EP All-to-All (prefill, MoE)

For MoE layers, the Dispatch + Combine payload per direction is $B_{\text{prefill}} \cdot S_{\text{input}} \cdot k \cdot H \cdot b$ (vs. $k \cdot H \cdot b$ during single-token decode). The collective group size is $G_{EP}$ from `notation.md §1` (equal to $EP$ across all three configurations). The shipped A2A primitive is pairwise direct-send (NCCL on star; bisection-bound pairwise on torus) — see `collectives/01_collective_algorithms.md §7`. Substituting $M_{EP}^{\text{prefill}}(B_{\text{prefill}}) = B_{\text{prefill}} \cdot S_{\text{input}} \cdot k H \cdot b$ into the star pairwise form (the $\times 2$ Dispatch + Combine factor is recovered via $n_{EP} = 2$ in the per-layer accumulator below):

$$
t_{EP}^{\text{prefill}}(B_{\text{prefill}}) \;=\; (G_{EP}-1)\,\alpha_{EP} \;+\; \frac{G_{EP}-1}{G_{EP}} \cdot \frac{B_{\text{prefill}} \cdot S_{\text{input}} \cdot k H \cdot b}{BW_{\text{EP}}}
$$

For torus EP fabrics, use the torus A2A form of `collectives/02_topology_mapping.md §3` with $M = B_{\text{prefill}} \cdot S_{\text{input}} \cdot k H \cdot b$.

The same two A2A data-flow patterns described for decode in `decode.md §5.2` apply identically to prefill under DP-attention. The formula above models the **gather-then-dispatch** ceiling — full $B_{\text{prefill}} \cdot S_{\text{input}}$ tokens are present on every rank before MoE dispatch, paired with the per-MoE-layer TP all-gather counted in §3.2's TP collective. The **scatter-direct** alternative (DeepEP-style [DEEPEP], operating on per-rank sequence-sharded attention outputs of size $B_{\text{prefill}} \cdot S_{\text{input}} / G_{TP}$) replaces the formula above by substituting $B_{\text{prefill}} \cdot S_{\text{input}} \to B_{\text{prefill}} \cdot S_{\text{input}} / G_{TP}$ and dropping the per-MoE-layer TP all-gather contribution from §3.2's per-stage accumulator. The prefill regime is compute-bound (long collective-hiding windows behind general matrix multiply (GEMM) tiles) and the per-rank reduction is structurally smaller than in decode (prefill payloads scale with $S_{\text{input}}$, so the $G_{TP}$-fold cut directly buys back wall-clock when MoE comm enters the critical path); both patterns therefore see meaningful use in production [SGLANG-DPATTN].

### PP Hop (prefill)

The PP hop forwards the hidden-state shard to the next stage. With cross-section rank alignment, each device forwards its local share of shape $[B_{\text{prefill}} \cdot S_{\text{input}} \times H/D_{\text{kv}}]$; this is a single point-to-point transfer (see `decode.md §5.1` for the p2p rationale and the unified $D_{\text{kv}}$-based per-rank payload):

$$
t_{PP}^{\text{prefill}}(B_{\text{prefill}}) \;=\; \alpha_{PP} \;+\; \frac{B_{\text{prefill}} \cdot S_{\text{input}} \cdot (H / D_{\text{kv}}) \cdot b}{BW_{\text{PP}}}
$$

$\alpha_{PP}$ and $BW_{PP}$ are tier-aware: under the orthogonal layout (`DP → PP → EP → TP → SP`, fast axes inner), the PP boundary uses the fabric tier whose cumulative reach holds $PP \cdot \text{(inner-axes product)}$. See `decode.md §5.1` for the full convention; the same tier-allocation logic carries over to co-located layouts with $\max(TP, EP)$ substituted for $TP \cdot EP$ in the cumulative reach.

### SP All-Gather (prefill)

During prefill with SP, each SP rank holds $S_{\text{input}}/SP$ of each request's input sequence. Ring Attention circulates KV shards so each device's query block can attend to the full input; the shipped primitive is ring AG per `collectives/01_collective_algorithms.md §6`. Each batched request streams its own KV shard, so the per-rank payload is $M_{SP}^{\text{prefill}}(B_{\text{prefill}}) = B_{\text{prefill}} \cdot (S_{\text{input}} / SP) \cdot (2 H_{kv} / D_{\text{kv}}) \cdot b$ (the KV shard composes the $D_{\text{kv}}$ partition with the SP sequence partition):

$$
t_{SP}^{\text{prefill}}(B_{\text{prefill}}) \;=\; (SP-1)\,\alpha_{SP} \;+\; (SP-1) \cdot \frac{B_{\text{prefill}} \cdot (S_{\text{input}} / SP) \cdot (2 H_{kv} / D_{\text{kv}}) \cdot b}{BW_{\text{SP}}}
$$

For torus SP fabrics, use the torus AG form of `collectives/02_topology_mapping.md §3` with the same $M_{SP}^{\text{prefill}}(B_{\text{prefill}})$.

### Total per-stage communication time (prefill)

Following the same structure as `decode.md §5.5`, collectives within each layer are sequential. The per-layer TP cost is the sum of one attention TP collective (AR under TP-attention or AG under DP-attention) and one FFN TP all-reduce (always AR):

$$
t_{TP}^{\text{prefill,layer}}(B_{\text{prefill}}) \;=\; t_{TP}^{\text{prefill,attn}}(B_{\text{prefill}}) \;+\; t_{TP,\text{AR}}^{\text{prefill}}(B_{\text{prefill}}), \qquad
t_{TP}^{\text{prefill,attn}}(B_{\text{prefill}}) \;=\; \begin{cases} t_{TP,\text{AR}}^{\text{prefill}}(B_{\text{prefill}}) & \text{TP-attention} \\ t_{TP,\text{AG}}^{\text{prefill}}(B_{\text{prefill}}) & \text{DP-attention} \end{cases}
$$

The per-axis NCCL API call counts $n_{TP} = 2$, $n_{EP} = 2$, $n_{SP} = 1$ are the same as decode (`decode.md §5.5`). The total per-stage prefill comm is:

$$
t_{\text{prefill,comm}}(B_{\text{prefill}}) =
\frac{L}{PP}
\bigl(
t_{TP}^{\text{prefill,layer}}(B_{\text{prefill}}) +
n_{SP}\, t_{SP}^{\text{prefill}}(B_{\text{prefill}})
\bigr) +
\frac{L_{\text{moe}}}{PP}
\bigl(
n_{EP}\, t_{EP}^{\text{prefill}}(B_{\text{prefill}})
\bigr) +
t_{PP}^{\text{prefill}}(B_{\text{prefill}})
$$

where $n_{EP} = 2$ for MoE layers (Dispatch + Combine, each costing one single-direction A2A) and $0$ for dense; $n_{SP} = 1$ when SP is enabled, $0$ otherwise.

---

## 3.3 Pipeline Warmup Latency

With $PP$ pipeline stages, the first token's hidden state must traverse all stages **sequentially** before the first decoded token can be emitted. Since the pipeline is empty at the start of prefill, there is no parallel filling:

$$
t_{\text{pipeline,warmup}}(B_{\text{prefill}}) =
(PP - 1) \cdot t_{\text{stage}}(B_{\text{prefill}})
$$

where $t_{\text{stage}}(B_{\text{prefill}})$ is the latency to process one prefill batch through a single PP stage (approximately $t_{\text{prefill,local}}(B_{\text{prefill}})$ for the bottleneck stage, plus its inter-stage PP hop $t_{PP}^{\text{prefill}}(B_{\text{prefill}})$). For uniform PP stages:

$$
t_{\text{pipeline,warmup}}(B_{\text{prefill}})
\approx
(PP - 1) \cdot
\bigl(t_{\text{prefill,local}}(B_{\text{prefill}}) + t_{PP}^{\text{prefill}}(B_{\text{prefill}})\bigr)
$$

For $PP = 1$ (no pipeline parallelism), $t_{\text{pipeline,warmup}}(B_{\text{prefill}}) = 0$.

---

## 3.4 Hardware Prefill Latency Formula

Combining all three phases, with overlap factor $\rho \in [0, 1]$ capturing the fraction of prefill communication that can be hidden behind compute, an SW dispatch budget $t_{\text{stage,sw}}$ for per-stage CPU kernel-launch overhead, and the once-per-pass LM head roofline $t_{\text{LM,prefill,hw}}$ on stage $PP{-}1$:

$$
t_{\text{prefill}}(B_{\text{prefill}}) = t_{\text{prefill,local}}(B_{\text{prefill}}) + \max\!\bigl(0,\; t_{\text{prefill,comm}}(B_{\text{prefill}}) - \rho\, t_{\text{prefill,local}}(B_{\text{prefill}})\bigr) + t_{\text{pipeline,warmup}}(B_{\text{prefill}}) + t_{\text{LM,prefill,hw}}(B_{\text{prefill}})
$$

**Interpretation of each term:**

- **$t_{\text{prefill,local}}(B_{\text{prefill}})$** is the roofline local time with three corrections added to the legacy $\max(t_{\text{compute}}, t_{\text{mem}})$ form: a Tensor Core efficiency derate at small effective microbatch, an HBM sustained-bandwidth derate against $\eta_\beta(B_{\text{prefill}})$, and an SW composition for per-stage CPU kernel-launch overhead. All three are detailed below.
- **$\max(0,\, t_{\text{prefill,comm}}(B_{\text{prefill}}) - \rho\, t_{\text{prefill,local}}(B_{\text{prefill}}))$** is residual communication after compute–communication overlap. In the compute-bound prefill regime, $t_{\text{prefill,local}}$ is large, so significant communication hiding ($\rho \approx 0.8$–$1.0$) is achievable.
- **$t_{\text{pipeline,warmup}}(B_{\text{prefill}})$** is the pipeline fill penalty; grows with $PP$ and with $B_{\text{prefill}} \cdot S_{\text{input}}$ since both $t_{PP}^{\text{prefill}}$ and $t_{\text{prefill,local}}$ scale with the per-step token count.
- **$t_{\text{LM,prefill,hw}}(B_{\text{prefill}})$** is the once-per-pass LM head roofline on stage $PP{-}1$, defined below. It is added outside the warmup because it fires once at the end of the prefill traversal (after the pipeline is filled), not per stage.

### LM head latency (stage PP-1 only)

Mirroring `decode.md §6.2`, the LM head $H \to V$ projection is a per-pass one-shot kernel on stage $PP{-}1$ that uses the FLOPs and traffic from §1.5:

$$
t_{\text{LM,prefill,hw}}(B_{\text{prefill}}) =
\max\!\left(
  \frac{2 B_{\text{prefill}} H V / D_{\text{emb}}}{R_{\text{GPU}}},\;
  \frac{H V b / D_{\text{emb}} + B_{\text{prefill}} V b}{BW_{\text{mem}}}
\right)
$$

For chunked prefill (§5), the LM head only fires on the **last chunk** — once per request, after the final position's hidden state is available — so $t_{\text{LM,prefill,hw}}$ adds the same one-shot term to the chunked total rather than once per chunk. In typical prefill regimes ($S_{\text{input}} \gtrsim 100$) this term is well under 1% of the body cost.

### Tensor Core efficiency derate

The compute term is corrected for small effective microbatch (notation.md §9):

$$
t_{\text{compute}}^{\text{eff}}(B_{\text{prefill}}) = \frac{t_{\text{compute}}(B_{\text{prefill}})}{\eta_{\text{TC}}(\text{mb}_{\text{prefill}})}, \qquad \text{mb}_{\text{prefill}} = \frac{B_{\text{prefill}} \cdot S_{\text{input}}}{PP}
$$

For prefill, $\text{mb}_{\text{prefill}}$ is large enough at typical $S$ that $\eta_{\text{TC}} \approx 1$; the term is included for consistency with decode and to capture small-$S$ corner cases.

### Memory bandwidth derate

By symmetry with the compute term, the memory term may be corrected for the $B$-dependent HBM sustained-bandwidth curve $\eta_\beta(B)$ from `decode.md §6.2` and `notation.md §20`:

$$
t_{\text{mem}}^{\text{eff}}(B_{\text{prefill}}) = \frac{t_{\text{mem}}(B_{\text{prefill}})}{\eta_\beta(B_{\text{prefill}})}
$$

For prefill the curve is typically inert: prefill is compute-bound at production $S_{\text{input}}$, so $t_{\text{mem}}$ is not on the critical path; and the dominant mechanism behind the curve (concurrent KV reads from many decode sequences) does not arise during prefill (KV is being **written**, not read). The term is included for symmetry with the decode formulation and to handle disaggregated chunked-prefill regimes where prefill chunks share a memory subsystem with concurrent decode traffic.

### SW composition

The per-stage CPU dispatch budget $t_{\text{stage,sw}}$ is applied via the SW-overlap factor $\rho_{\text{SW}}$ in the same base + unhidden-overflow form as decode (`decode.md §7.1` and `§7.3`). Let $t_{\text{prefill,GPU}}(B_{\text{prefill}}) = \max\bigl(t_{\text{compute}}^{\text{eff}}(B_{\text{prefill}}),\, t_{\text{mem}}^{\text{eff}}(B_{\text{prefill}})\bigr)$ denote the GPU-side roofline:

$$
t_{\text{prefill,local}}(B_{\text{prefill}}) = t_{\text{prefill,GPU}}(B_{\text{prefill}}) + \max\!\bigl(0,\; t_{\text{stage,sw}} - \rho_{\text{SW}} \cdot t_{\text{prefill,GPU}}(B_{\text{prefill}})\bigr)
$$

$t_{\text{stage,sw}}$ is the same per-stage launch budget as decode (`decode.md §7.1`):

$$
t_{\text{stage,sw}} = \tau_{\text{launch}} \cdot \left[ \frac{L}{PP} \bigl( k_{\text{compute}} + k_{\text{collective}}(n_{TP} + n_{SP}) \bigr) + \frac{L_{\text{moe}}}{PP} \cdot k_{\text{collective}} \cdot n_{EP} + k_{\text{pp\_hop}} \right]
$$

with the unified $n_{EP} = 2$ for MoE layers (Dispatch + Combine). $t_{\text{stage,sw}}$ is $B_{\text{prefill}}$-independent — kernel launches pay $\tau_{\text{launch}}$ once per launch event regardless of payload. Each prefill forward pass is a single end-to-end sweep with no microbatch round structure (cf. decode's pipelined steady state); the per-stage formula applies once per stage per prefill pass. With `kernel_launch_us = 0` the term vanishes (legacy roofline).

> **Overlap note:** The overlap factor $\rho$ is an original parameterization (this work); see `references.md`. In the compute-bound prefill regime, compute and communication can be overlapped aggressively by pipelining GEMM tiles with collective operations (e.g., using NCCL + CUDA stream concurrency). Practical $\rho$ values are system-dependent but commonly 0.5–0.9. The independent $\rho_{\text{SW}}$ governs CPU-GPU dispatch overlap; default $\rho_{\text{SW}} = 1$ assumes async dispatch keeps the GPU command queue full.

---

<div style="page-break-before: always;"></div>

# 4. Optimal Batch Size for GPU Utilization

§3 derived $t_{\text{prefill}}(B_{\text{prefill}})$ with $B_{\text{prefill}}$ threaded through the compute, memory, communication, and warmup terms. This section translates that scaling into a serving-side design rule: pick the batch size that keeps the GPU compute-bound while respecting the prefill latency SLO.

In continuous batching, requests may have different lengths; we model the uniform-length case ($B_{\text{prefill}}$ requests of $S_{\text{input}}$ tokens each) for clarity. The analysis generalizes by replacing $S_{\text{input}}$ with the average padded length.

### Minimum batch size for compute saturation

The GPU is compute-bound when:

$$
\text{OI}(B_{\text{prefill}}, S_{\text{input}})
\ge R_{\text{ridge}}
$$

Using $\text{OI}_{\text{prefill}} \approx 2 S_{\text{input}} / b$ (§2.1), single-request prefill is already compute-bound for $S_{\text{input}} \gtrsim S_{\text{input}}^{\star}$ (§2.3). Batching primarily helps when $S_{\text{input}} < S_{\text{input}}^{\star}$, in which case the minimum batch size to enter the compute-bound regime is:

$$
B_{\text{prefill}}^{\min} =
\left\lceil \frac{R_{\text{ridge}} \cdot b}{2 \cdot S_{\text{input}}} \right\rceil =
\left\lceil \frac{R_{\text{GPU}}}{BW_{\text{mem}} \cdot S_{\text{input}}} \right\rceil
$$

For H100 and $S_{\text{input}} = 64$ tokens (a very short prompt): $B_{\text{prefill}}^{\min} \approx \lceil 295/64 \rceil \approx 5$ requests needed before compute-saturation.

### Maximum batch size from prefill latency SLO

The prefill latency SLO constrains the maximum allowable batch size. In the compute-bound regime, $t_{\text{prefill}}(B_{\text{prefill}}) \approx B_{\text{prefill}} \cdot t_{\text{prefill}}(1)$ — every batched request waits for the full pass before getting its first token — so:

$$
B_{\text{prefill}}^{\max} =
\left\lfloor \frac{t_{\text{prefill,SLO}}}{t_{\text{prefill}}(1)} \right\rfloor
$$

### Optimal operating point

$$
B_{\text{prefill}}^{\text{opt}} =
\min\!\bigl(B_{\text{prefill}}^{\max},\; B_{\text{avail}}\bigr)
$$

where $B_{\text{avail}}$ is the number of requests available in the queue. Batching beyond $B_{\text{prefill}}^{\max}$ violates the prefill latency SLO; batching below $B_{\text{prefill}}^{\min}$ leaves the GPU under-utilized.

---

<div style="page-break-before: always;"></div>

# 5. Chunked Prefill

For very long prompts ($S_{\text{input}} \gg 1$), prefilling the entire sequence in one pass can monopolize the GPU for a long time and stall ongoing decode iterations for other requests in the system. **Chunked prefill** [SARATHI] addresses this by splitting the input sequence into $\lceil S_{\text{input}} / C \rceil$ chunks of $C$ tokens each and interleaving prefill chunks with decode iterations.

---

## 5.1 Per-Chunk Latency

Each chunk processes $C$ tokens. The **linear terms** (projections + FFN) are independent of position and identical for every chunk. However, the **attention term** is chunk-index-dependent: chunk $k$ (processing tokens $[(k{-}1)C,\; kC)$) has its $C$ queries attend to the **full accumulated KV cache** of $kC$ positions [SARATHI], not just the $C$ tokens within the chunk itself. The per-chunk FLOPs for chunk $k = 1, 2, \ldots, N_{\text{chunks}}$:

$$
F_{\text{chunk,device}}^{(k)} = \frac{L}{PP} \left[ \frac{(4H^2 + 4 H H_{kv}) C}{D_{\text{attn}}} + \frac{6 H I_{\text{eff}} C}{D_{\text{exp}}} + \frac{4 \cdot C \cdot kC \cdot H}{D_{\text{kv}} \cdot SP} \right]
$$

The first two terms inside the brackets are **linear** (projections + FFN, constant across chunks). The third term is the **attention** component, which grows with chunk index $k$ since chunk $k$ attends to $kC$ accumulated KV positions.

The first chunk ($k = 1$) attends to $C$ positions; the last chunk ($k = N_{\text{chunks}}$) attends to approximately $S_{\text{input}}$ positions.

The per-chunk local time (roofline):

$$
t_{\text{chunk,compute}}^{(k)} =
\frac{F_{\text{chunk,device}}^{(k)}}{R_{\text{GPU}}}
$$

$$
t_{\text{chunk,local}}^{(k)} =
\max\left(
t_{\text{chunk,compute}}^{(k)},\;
t_{\text{chunk,mem}}^{(k)}
\right)
$$

where $t_{\text{chunk,mem}}^{(k)}$ accounts for weight reads (same per chunk), KV cache writes ($C$ new entries per chunk), and KV cache reads ($kC$ entries for the attention pass). In practice, weight reads dominate the memory time, so the $k$-dependence is a small correction.

The per-chunk communication time is approximately constant (dominated by TP allreduce of activations, which scales with $C$ not $kC$):

$$
t_{\text{chunk,comm}}
\approx
t_{\text{prefill,comm}}|_{S_{\text{input}} = C}
$$

Per-chunk overlap-adjusted latency:

$$
t_{\text{chunk}}^{(k)} = t_{\text{chunk,local}}^{(k)} + \max\!\left(0,\; t_{\text{chunk,comm}} - \rho\, t_{\text{chunk,local}}^{(k)}\right)
$$

---

## 5.2 Total Prefill Time Under Chunking

The number of chunks is $N_{\text{chunks}} = \lceil S_{\text{input}} / C \rceil$. Because per-chunk FLOPs vary with $k$, the total time is a sum over chunks:

$$
t_{\text{prefill,chunked}} =
\sum_{k=1}^{N_{\text{chunks}}} t_{\text{chunk}}^{(k)}
$$

### Total FLOPs decomposition

The total FLOPs across all chunks separate into constant (linear) and growing (attention) contributions:

- **Linear terms:** $N_{\text{chunks}} \times C = S_{\text{input}}$ tokens processed in total — identical to the unchunked FLOPs from §1.5.
- **Attention terms:**

$$
\sum_{k=1}^{N} \frac{4 k C^2 H}{D_{\text{kv}} \cdot SP} =
\frac{4 C^2 H}{D_{\text{kv}} \cdot SP} \cdot \frac{N(N{+}1)}{2}
\;\approx\;
\frac{2 S_{\text{input}}^2 H}{D_{\text{kv}} \cdot SP}
$$

The chunked attention total ($\approx 2S^2H$) is half the unchunked convention ($4S^2H$ from §1.2). This is **not** a FLOP reduction from chunking — it reflects the **causal structure**: each chunk naturally computes attention only to its accumulated context (positions $0$ through $kC$), faithfully capturing the lower-triangular causal mask. The unchunked formula uses the full-matrix $4S^2H$ convention (see §1.2 note: practical FlashAttention implementations process the full tile and mask, so published FLOPs budgets use $S^2$ rather than $S^2/2$). The actual compute work is the same regardless of chunking.

> **Practical equivalence:** At typical prefill lengths where linear terms ($O(S)$) dominate the $S^2$ attention terms, the total compute time under chunking is approximately the same as unchunked:
>
> $$
> t_{\text{prefill,chunked}} \approx t_{\text{prefill,unchunked}} \quad (\text{compute-bound, linear terms dominant})
> $$
>
> The per-chunk variation in $t_{\text{chunk}}^{(k)}$ is small when attention is a minor fraction of total FLOPs (i.e., $S_{\text{input}} \ll H + H_{kv} + \frac{3}{2}I_{\text{eff}}$, the regime crossover from §1.4).

Hardware prefill latency for a chunked-prefill request:

$$
t_{\text{prefill,chunked}}^{\text{total}} \approx t_{\text{prefill,chunked}} + t_{\text{pipeline,warmup}}
$$

Because the prefill is spread over $N_{\text{chunks}}$ scheduler iterations, the observed prefill latency from the requesting client is the time until all chunks complete and the first token is generated — longer than unchunked prefill for the same request in isolation (due to inter-chunk scheduling overhead and interleaved decode iterations).

---

## 5.3 Throughput Benefit and Head-of-Line Blocking Reduction

The core benefit of chunked prefill is **eliminating head-of-line blocking** in a continuous-batching serving system.

### Without chunked prefill

A long prefill with $S_{\text{input}} = 16384$ tokens on a single GPU takes $t_{\text{prefill}} \approx F_{\text{prefill}} / R_{\text{GPU}}$ — potentially tens to hundreds of milliseconds. All decode-phase requests that arrive during this time must wait, increasing their TPOT.

### With chunked prefill

Each chunk of $C$ tokens occupies approximately $t_{\text{chunk}} = (C / S_{\text{input}}) \cdot t_{\text{prefill}}$ compute time. Decode iterations for other requests execute between chunks. The maximum TPOT penalty for concurrent decode requests is bounded by $t_{\text{chunk}}$ rather than the full $t_{\text{prefill}}$.

**Trade-off:**

| Metric | Unchunked prefill | Chunked prefill ($C$ tokens) |
|--------|-------------------|------------------------------|
| Per-request prefill latency | Low (one pass) | Higher (spread over iterations) |
| Concurrent decode TPOT | High spikes during long prefill | Bounded by $t_{\text{chunk}}$ |
| System throughput | Lower (head-of-line blocking) | Higher (decode and prefill interleaved) |
| Chunk size selection | N/A | Tune $C$ to balance prefill latency and TPOT SLOs |

See [SARATHI] for empirical validation of chunked prefill scheduling and its effect on P99 prefill latency and decode latency under mixed workloads.

---

<div style="page-break-before: always;"></div>

# 6. Prefill→Decode KV Handoff

After prefill produces the full KV cache for $S_{\text{input}}$ tokens, that cache must be **handed off** to the decode pipeline before the first decode step can run. The handoff is rarely free, even when prefill and decode share the same cluster: any difference between the prefill-side partition $(TP_p, PP_p, SP_p)$ and the decode-side partition $(TP_d, PP_d, EP_d, SP_d)$ forces the KV state to be reshaped, and any inter-cluster boundary forces it to traverse a slower fabric than the on-device HBM that wrote it.

This section models the handoff cost in two architectures:

- **Co-located** (§6.3) — prefill and decode share the same cluster but typically differ in partition; handoff cost is an internal scale-up collective.
- **Disaggregated** (§6.4) — prefill and decode run on separate clusters connected by a slower inter-cluster fabric [DISAGG-PREFILL]; handoff cost includes inter-cluster bytes plus several device-level overheads commonly hidden inside an effective $BW_{\text{inter}}$.

§6.5 then assembles the unified hardware prefill latency for both cases.

---

## 6.1 Architecture Variants

In a **co-located** system, a single GPU class must satisfy two competing requirements: high HBM bandwidth for decode and high compute throughput for prefill. These can conflict — a chip with wide HBM but lower FLOPs/byte favors decode; a chip with narrow HBM but high FLOPs/byte favors prefill.

A **disaggregated** architecture [DISAGG-PREFILL] runs prefill and decode on separate clusters, each optimized for its dominant workload:

- **Prefill cluster** — high peak FLOPs, compute-optimal partition (lower HBM bandwidth per FLOP is acceptable since prefill is GEMM-bound).
- **Decode cluster** — high HBM bandwidth, memory-optimal partition (large HBM capacity for KV cache, fast HBM for GEMV throughput).

Disaggregation also lets the two clusters scale independently: long-prompt / short-output workloads benefit from more prefill nodes; chatbot-style short-prompt / long-output workloads benefit from more decode nodes. See [DISAGG-PREFILL] (DistServe) for a full system design and goodput analysis.

In **both** architectures the prefill-side partition is rarely identical to the decode-side partition, so the KV handoff is non-trivial in both cases — it is just that the bottleneck differs (scale-up NVLink for co-located vs. scale-out inter-cluster fabric for disaggregated).

---

## 6.2 KV Cache Volume (shared)

The total KV cache produced by a complete prefill pass (all $L$ layers, both keys and values, all $S_{\text{input}}$ tokens) is identical in both architectures:

$$
M_{\text{KV,total}}=
2 \cdot L \cdot S_{\text{input}} \cdot H_{kv} \cdot b
\quad \text{(bytes)}
$$

This follows directly from the per-layer KV cache expression in `decode.md §1.3`, summed over all $L$ layers. With GQA [GQA], $H_{kv} = n_{kv} \cdot d_{\text{head}} \ll H$, substantially reducing the transfer volume compared to MHA ($H_{kv} = H$).

**Example (LLaMA-3 70B):** $L = 80$, $n_{kv} = 8$, $d_{\text{head}} = 128$, $H_{kv} = 1024$, $S_{\text{input}} = 4096$, $b = 2$ (bf16):

$$
M_{\text{KV,total}}=
2 \times 80 \times 4096 \times 1024 \times 2
\approx 1.34 \text{ GB}
$$

If the prefill cluster shards the KV cache across $D_{\text{kv,p}} \cdot SP_p$ ranks (where $D_{\text{kv,p}}$ is the prefill-side $D_{\text{kv}}$ from `notation.md §1`; equal to $TP_p$ for the typical orthogonal + TP-attn prefill, or $\max(TP_p, EP_p)$ for a co-located DP-attn prefill), each prefill device holds

$$
M_{\text{KV,shard,p}} = \frac{M_{\text{KV,total}}}{D_{\text{kv,p}} \cdot SP_p}
$$

bytes locally. The handoff models below operate on this per-device shard volume, then account for any reshaping required to land the cache in the decode-side layout.

---

## 6.3 Co-located Handoff: KV Layout Transition

When prefill and decode run on the same cluster but on different partitions, the KV cache produced by prefill in layout $(TP_p, PP_p, SP_p)$ must be re-sharded into the decode layout $(TP_d, PP_d, EP_d, SP_d)$ before the first decode step. This transition is an internal scale-up collective over the cluster's NVLink-class fabric — **not** zero, even though no inter-cluster transfer is involved.

### When the handoff is non-trivial

| Partition difference | Required reshaping |
|----------------------|--------------------|
| $TP_p = TP_d$, $PP_p = PP_d$, $SP_p = SP_d$ | None (KV is already in place) |
| $TP_p \neq TP_d$ | All-gather over the head-dimension across the new TP group |
| $SP_p \neq SP_d$ | All-gather over the sequence-dimension across the new SP group |
| $PP_p \neq PP_d$ | Layer-shard relocation across PP ranks (point-to-point transfers along the new pipeline) |

In practice all three differ simultaneously: prefill clusters favor low TP and low SP to keep GEMMs large, while decode clusters favor high TP and high SP to keep HBM traffic per device low.

### Handoff latency model

Treat the layout transition as a single **all-gather-equivalent** collective over the decode-cluster scale-up fabric, with effective per-device bandwidth $BW_{\text{intra}}$ (NVLink, ≈900 GB/s on H100/B200; ≈1.8 TB/s on GB200 NVL72):

$$
t_{\text{handoff,colo}} = \alpha_{\text{intra}} + \frac{M_{\text{KV,total}}}{BW_{\text{intra}}} \cdot \eta_{\text{repack}}
$$

where:

- $\alpha_{\text{intra}}$ — scale-up collective startup latency (≈1–5 µs over NVLink/NVSwitch).
- $\eta_{\text{repack}} \in [1, 2]$ — repack inefficiency factor capturing non-contiguous gather patterns and the HBM write back into PagedAttention blocks (see §6.4 item 3).
- The total bytes moved across the fabric is $M_{\text{KV,total}}$ (each rank's local shard is sent to wherever the new decode partition needs it).

When the partitions match exactly, $t_{\text{handoff,colo}} = 0$.

---

## 6.4 Disaggregated Handoff: Inter-Cluster KV Transfer

In a disaggregated architecture, the prefill cluster's KV cache must be moved across an **inter-cluster fabric** (RoCEv2, InfiniBand HDR/NDR, or NVLink-C2C in special cases) to the decode cluster. This subsection refines the standard $\alpha$–$\beta$ bound to expose four device-level overheads that must be absorbed into the **effective delivered bandwidth** $BW_{\text{inter}}$ (not the NIC catalog line rate), and adds the layer-wise streaming optimization used by production systems.

> **Convention.** Throughout this document, $BW_{\text{inter}}$ denotes the *effective, delivered, end-to-end* per-GPU inter-cluster bandwidth after PCIe egress, NIC sharing, and HBM-write inefficiencies (overheads 1, 3 below). It is a calibration knob, not a catalog spec.

### α–β baseline

The textbook bound [ALPHA-BETA] for one bulk transfer of the full KV cache:

$$
t_{\text{KV-transfer}}^{\text{bulk}} = \alpha_{\text{inter}} + \frac{M_{\text{KV,total}}}{BW_{\text{inter}}}
$$

Reference NIC line rates: InfiniBand HDR (200 Gb/s per port) → ≈20 GB/s unidirectional; ConnectX-7 (400 Gb/s) → ≈50 GB/s; NVLink-C2C (Grace–Hopper) → ≈900 GB/s. The *delivered* $BW_{\text{inter}}$ is typically a fraction of these (see below).

For the LLaMA-3 70B example assuming $BW_{\text{inter}} \approx 20$ GB/s: $1.34\text{ GB} / 20\text{ GB/s} \approx 67$ ms — comparable to or larger than the prefill compute itself.

This $\alpha$–$\beta$ form is the worst case: serial bulk transfer with no compute overlap. The next two subsections refine both the bandwidth (absorbing device-level overheads) and the latency (accounting for layer-wise streaming).

### Device-level overheads absorbed into $BW_{\text{inter}}$

Four effects degrade the *delivered* end-to-end bandwidth $BW_{\text{inter}}$ below the NIC line rate:

1. **PCIe egress on the prefill side.** PCIe Gen5 ×16 = 64 GB/s per NIC, often shared across 2–4 GPUs, giving 16–32 GB/s of effective egress per GPU. For modern NICs (ConnectX-7 at 50 GB/s line rate) this PCIe ceiling — not the NIC — is the real bottleneck. $BW_{\text{inter}}$ must be parameterized as the *delivered, end-to-end* per-GPU bandwidth after NIC and PCIe sharing, not as the catalog NIC rate.

2. **Layout repack on the decode side.** As in §6.3, mismatched $(TP, SP)$ between prefill and decode forces an all-gather-equivalent over the **decode-cluster** scale-up fabric. This consumes scale-up bandwidth ($BW_{\text{intra,d}}$) on the destination, not $BW_{\text{inter}}$, and runs after the bytes arrive. Add it as an additive term:

   $$
   t_{\text{repack}} = \frac{M_{\text{KV,total}}}{BW_{\text{intra,d}}} \cdot \eta_{\text{repack}}
   $$

3. **HBM write into PagedAttention blocks.** The decode side writes the arriving KV cache into PagedAttention blocks (`kv.md §2`), each of size $\text{BLK}_{KV}$. The block-stride write pattern hits 50–80% of peak HBM bandwidth, not 100%. For a typical 1–2 GB cache this adds 10–50 µs and can be folded into $\eta_{\text{repack}}$ above.

4. **RDMA work-request posting.** Each (layer, shard) tuple is typically a separate RDMA work request; total $N_{\text{WR}} \approx L \cdot D_{\text{kv,p}} \cdot SP_p$ requests, at ≈1 µs each (mostly async, but the head-of-queue latency is not). This inflates the effective startup beyond a single $\alpha_{\text{inter}}$:

   $$
   \alpha_{\text{inter}}^{\text{eff}} = \alpha_{\text{inter}} + N_{\text{WR}} \cdot \tau_{\text{WR}}
   $$

   For $L = 80$, $D_{\text{kv,p}} = TP_p = 8$, $SP_p = 1$, $\tau_{\text{WR}} = 1$ µs → $N_{\text{WR}} \cdot \tau_{\text{WR}} = 640$ µs, often larger than $\alpha_{\text{inter}}$ itself.

### Layer-wise streaming overlap ($\rho_{KV}$)

Production disaggregated systems (MoonCake [MOONCAKE], NVIDIA Dynamo [DYNAMO], DistServe [DISAGG-PREFILL]) start streaming the KV for layer $k$ the **moment** prefill of layer $k$ completes, rather than waiting for the full prefill to finish. With $L$ layers, the first $(L-1)/L$ of the KV transfer can pipeline behind the remaining $(L-1)/L$ of prefill, hiding the bulk of $t_{\text{KV-transfer}}$ when prefill itself is long.

Define a layer-wise streaming overlap factor $\rho_{KV} \in [0, 1]$:

- $\rho_{KV} = 0$ — serial bulk transfer (the §6.4 baseline).
- $\rho_{KV} = 1$ — fully hidden behind prefill (only feasible when $t_{\text{prefill}} \ge M_{\text{KV,total}} / BW_{\text{inter}}$).

Calibration: real systems report $\rho_{KV} \in [0.8, 0.95]$ for long-$S_{\text{input}}$ workloads on well-tuned MoonCake/Dynamo deployments [MOONCAKE]; $\rho_{KV} \approx 0$ for short-prompt workloads where prefill finishes before the first KV chunk lands.

### Refined transfer latency

Combining the four overheads with the streaming overlap:

$$
t_{\text{KV-transfer}}^{\text{eff}}=
\max\!\left(0,\;
\alpha_{\text{inter}}^{\text{eff}} +
\frac{M_{\text{KV,total}}}{BW_{\text{inter}}} +
t_{\text{repack}} -
\rho_{KV}\cdot t_{\text{prefill}}
\right)
$$

The three correction terms relative to the textbook $\alpha$–$\beta$ form:

- $\alpha_{\text{inter}}^{\text{eff}}$ replaces $\alpha_{\text{inter}}$, absorbing RDMA WR posting (overhead 4).
- $BW_{\text{inter}}$ is interpreted as the *effective, delivered* per-GPU bandwidth, absorbing PCIe-egress and HBM-write inefficiencies (overheads 1, 3) rather than the NIC line rate.
- $t_{\text{repack}}$ is added explicitly to expose the scale-up cost on the decode side (overhead 2), since it consumes a different fabric than the inter-cluster transfer.
- $-\rho_{KV} \cdot t_{\text{prefill}}$ subtracts the portion hidden behind layer-wise streaming.

For modeling purposes, $BW_{\text{inter}}$ and $\rho_{KV}$ are calibration knobs: measure them on the target system rather than computing them from line-rate specs.

---

## 6.5 Hardware Prefill Latency (unified)

Let the prefill cluster have hardware parameters $R_{\text{GPU,pre}}$ and $BW_{\text{mem,pre}}$, and the decode cluster have $R_{\text{GPU,dec}}$ and $BW_{\text{mem,dec}}$. The three phases of hardware prefill latency are:

### Phase 1: Prefill on the prefill cluster

$$
t_{\text{prefill}} = t_{\text{prefill,local,pre}} + \max\!\left(0,\; t_{\text{prefill,comm,pre}} - \rho_{\text{pre}}\, t_{\text{prefill,local,pre}}\right) + t_{\text{pipeline,warmup,pre}}
$$

All "pre" subscripts refer to the prefill cluster's hardware and partition. In a co-located deployment, the prefill cluster *is* the same physical cluster as the decode cluster, but typically running on a different worker pool with its own partition.

### Phase 2: KV handoff

$$
t_{\text{handoff}} =
\begin{cases}
t_{\text{handoff,colo}} & \text{(co-located, §6.3)}\\[4pt]
t_{\text{KV-transfer}}^{\text{eff}} & \text{(disaggregated, §6.4)}
\end{cases}
$$

For co-located deployments with identical prefill/decode partitions, $t_{\text{handoff}} = 0$.

### Phase 3: Pipeline warmup on the decode cluster

When the first decoded token must traverse the decode cluster's pipeline from stage 1 to stage $PP_{\text{dec}}$:

$$
t_{\text{pipeline,warmup,dec}} =
(PP_{\text{dec}} - 1) \cdot t_{\text{stage,dec}}
$$

### Total hardware prefill latency

$$
t_{\text{prefill,total}} = t_{\text{prefill}} + t_{\text{handoff}} + t_{\text{pipeline,warmup,dec}}
$$

System knobs for each term:

| Term | Reduced by |
|------|-----------|
| $t_{\text{prefill}}$ | More/faster prefill nodes; higher TP for larger-batch GEMM; FP8 quantization |
| $t_{\text{handoff,colo}}$ | Match prefill/decode partitions where possible; faster scale-up fabric ($BW_{\text{intra}}$) |
| $t_{\text{KV-transfer}}^{\text{eff}}$ | Higher *delivered* $BW_{\text{inter}}^{\text{eff}}$ (NVLink-C2C, NDR); KV quantization; GQA [GQA]; layer-wise streaming ($\rho_{KV} \to 1$) |
| $t_{\text{pipeline,warmup,dec}}$ | Smaller $PP_{\text{dec}}$; prefetch KV before decode pipeline clears |

### Comparison: co-located vs. disaggregated

| Architecture | $t_{\text{handoff}}$ when partitions match | $t_{\text{handoff}}$ when partitions differ |
|--------------|---------------------------------------------|---------------------------------------------|
| **Co-located** | $0$ | $\alpha_{\text{intra}} + M_{\text{KV,total}} / BW_{\text{intra}} \cdot \eta_{\text{repack}}$ |
| **Disaggregated** | (rare in practice) | $\max(0,\; \alpha_{\text{inter}}^{\text{eff}} + M_{\text{KV,total}} / BW_{\text{inter}}^{\text{eff}} + t_{\text{repack}} - \rho_{KV} \cdot t_{\text{prefill}})$ |

Co-located prefill latency is lower for individual requests when the partitions match. Disaggregated systems achieve better **system goodput** under mixed workloads because prefill no longer interferes with ongoing decode iterations [DISAGG-PREFILL], and modern layer-wise streaming ($\rho_{KV} \to 1$) erases most of the apparent transfer-latency penalty.

---

<div style="page-break-before: always;"></div>

# Symbol Index (new in this document)

The following symbols are introduced in this document and extend `notation.md §11`.

| Symbol | Definition |
|--------|-----------|
| $S_{\text{input}}$ | Input sequence length (number of tokens in the prefill pass) |
| $B_{\text{prefill}}$ | Number of requests batched together in a single prefill pass |
| $C$ | Chunk size: number of tokens per chunk in chunked prefill |
| $N_{\text{chunks}}$ | Number of chunks: $\lceil S_{\text{input}} / C \rceil$ |
| $F_{\text{prefill,device}}$ | Total prefill FLOPs per device (all layers on this PP stage) |
| $F_{\text{layer,prefill}}$ | Per-layer prefill FLOPs (full-sequence GEMM + $S^2$ attention) |
| $F_{\text{proj,prefill}}$ | Q/K/V/O projection FLOPs for prefill: $(4H^2 + 4HH_{kv}) S_{\text{input}}$ |
| $F_{\text{score,prefill}}$ | Attention score FLOPs for prefill: $2 S_{\text{input}}^2 H$ |
| $F_{\text{value,prefill}}$ | Value application FLOPs for prefill: $2 S_{\text{input}}^2 H$ |
| $F_{\text{ffn,prefill}}$ | FFN FLOPs for prefill: $6 H I_{\text{eff}} S_{\text{input}}$ |
| $T_{\text{KV,write,device}}$ | HBM write traffic for KV cache during prefill |
| $t_{\text{prefill,compute}}$ | Prefill compute time: $F_{\text{prefill,device}} / R_{\text{GPU}}$ |
| $t_{\text{prefill,mem}}$ | Prefill memory time: $T_{\text{prefill,device}} / BW_{\text{mem}}$ |
| $t_{\text{prefill,local}}$ | Prefill roofline local time: $\max(t_{\text{prefill,compute}}, t_{\text{prefill,mem}})$ |
| $t_{\text{prefill,comm}}$ | Total prefill communication time (TP/EP/SP/PP collectives) |
| $F_{\text{chunk,device}}^{(k)}$ | Per-device FLOPs for chunk $k$; attention term is $k$-dependent (§5.1) |
| $t_{\text{chunk}}^{(k)}$ | Overlap-adjusted latency of chunk $k$ (varies with $k$ due to attention) |
| $t_{\text{pipeline,warmup}}$ | Pipeline fill time: $(PP-1) \times t_{\text{stage}}$ |
| $t_{\text{prefill}}$ | Hardware prefill latency on the prefill cluster (Phase 1) |
| $t_{\text{prefill,total}}$ | Total hardware prefill latency including handoff and decode warmup |
| $M_{\text{KV,total}}$ | Total KV cache bytes produced by one prefill pass: $2 L S_{\text{input}} H_{kv} b$ |
| $M_{\text{KV,shard,p}}$ | Per-prefill-device KV shard: $M_{\text{KV,total}} / (TP_p \cdot SP_p)$ |
| $t_{\text{handoff}}$ | KV handoff time from prefill to decode (co-located or disaggregated) |
| $t_{\text{handoff,colo}}$ | Co-located KV layout-transition latency (scale-up collective) |
| $t_{\text{KV-transfer}}^{\text{bulk}}$ | Textbook $\alpha$–$\beta$ disaggregated transfer (no overlap, no overheads) |
| $t_{\text{KV-transfer}}^{\text{eff}}$ | Disaggregated transfer including device overheads and $\rho_{KV}$ overlap |
| $t_{\text{repack}}$ | Layout repack time on decode side: $M_{\text{KV,total}} / BW_{\text{intra,d}} \cdot \eta_{\text{repack}}$ |
| $BW_{\text{inter}}$ | Inter-cluster interconnect line-rate bandwidth (bytes/s) |
| $BW_{\text{inter}}^{\text{eff}}$ | *Delivered* end-to-end inter-cluster bandwidth (after PCIe/HBM/NIC sharing) |
| $BW_{\text{intra}}$, $BW_{\text{intra,d}}$ | Scale-up fabric bandwidth (NVLink); decode-side variant |
| $\alpha_{\text{inter}}$ | Inter-cluster link startup latency |
| $\alpha_{\text{inter}}^{\text{eff}}$ | Effective startup including RDMA WR posting: $\alpha_{\text{inter}} + N_{\text{WR}} \tau_{\text{WR}}$ |
| $\alpha_{\text{intra}}$ | Scale-up collective startup (≈1–5 µs over NVLink/NVSwitch) |
| $\eta_{\text{repack}}$ | Layout-repack inefficiency factor ($\in [1, 2]$); covers non-contiguous gather + paged-block writes |
| $\rho_{KV}$ | Layer-wise streaming overlap factor for disagg KV transfer ($\in [0, 1]$) |
| $N_{\text{WR}}$ | Number of RDMA work requests posted ($\approx L \cdot TP_p \cdot SP_p$) |
| $\tau_{\text{WR}}$ | Per-RDMA-WR posting latency (≈1 µs) |
| $S_{\text{input}}^{\star}$ | Compute-bound crossover: prefill is compute-bound for $S_{\text{input}} > S_{\text{input}}^{\star}$ |
| $R_{\text{ridge}}$ | Device ridge point: $R_{\text{GPU}} / BW_{\text{mem}}$ (FLOPs/byte) [ROOFLINE] |

---

# References

This document cites the following references from `references.md`:

- **[FA1]** Dao et al. (2022) — FlashAttention: tiled attention for HBM traffic reduction during prefill.
- **[FA2]** Dao (2023) — FlashAttention-2: improved parallelism across heads and sequence dimensions.
- **[GQA]** Ainslie et al. (2023) — Grouped-query attention; $H_{kv} = n_{kv} \cdot d_{\text{head}}$; KV volume reduction.
- **[MEGATRON]** Shoeybi et al. (2019) — TP column/row-parallel linear layers; prefill GEMM sharding.
- **[MEGATRON3]** Narayanan et al. (2021) — PP schedules; SP; pipeline warmup and 1F1B.
- **[SARATHI]** Agrawal et al. (2023) — Chunked prefill scheduling; head-of-line blocking reduction.
- **[DISAGG-PREFILL]** Zhong et al. (2024) — DistServe: disaggregated prefill–decode; KV transfer latency.
- **[MOONCAKE]** Qin et al. / Moonshot AI (2024) — MoonCake: KV-centric disaggregated serving; layer-wise KV streaming.
- **[DYNAMO]** NVIDIA (2024–2025) — Dynamo: production disaggregated inference with layer-wise KV transfer.
- **[ROOFLINE]** Williams et al. (2009) — Roofline model; OI; ridge point; compute vs. memory bound.
- **[ALPHA-BETA]** Hockney (1994) — $\alpha$–$\beta$ collective latency model.
- **[H100-SPEC]** NVIDIA (2022) — H100 SXM5 hardware specs: $R_{\text{GPU}}$, $BW_{\text{mem}}$, ridge point values.
