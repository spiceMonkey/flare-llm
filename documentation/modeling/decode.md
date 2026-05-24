# Decode and Time-Per-Output-Token (TPOT) Performance Model

**Author:** Yue Lu  
**Date:** November 2025  

**Keywords:**  
LLM inference, Transformer, parallelism, tensor parallelism, expert parallelism, sequence parallelism, pipeline parallelism, distributed systems, KV cache, collective communication, latency, throughput, cluster topology, performance modeling

---

<div style="page-break-after: always;"></div>

# Table of Contents

- [1. Memory Footprint](#1-memory-footprint)
  - [1.0 Parameter Definitions (P vs W)](#10-parameter-definitions-p-vs-w)
  - [1.1 Model Parameter Memory](#11-model-parameter-memory)
  - [1.2 Activation Memory (Per-token Working Memory)](#12-activation-memory-per-token-working-memory)
  - [1.3 KV Cache Memory](#13-kv-cache-memory)
  - [1.4 Per-device Memory Footprint After Parallelism Sharding](#14-per-device-memory-footprint-after-parallelism-sharding)

- [2. Memory Traffic During Decoding](#2-memory-traffic-during-decoding)
  - [2.1 Model Parameter Traffic $T_{\theta,\text{device}}$](#21-model-parameter-traffic-t_theta_textdevice)
  - [2.2 Activation Traffic $T_{\text{act,device}}$](#22-activation-traffic-t_textactdevice)
  - [2.3 KV Cache Traffic $T_{\text{KV,device}}(B)$](#23-kv-cache-traffic-t_textkvdeviceb)
  - [2.4 Total and Effective Traffic](#24-total-and-effective-traffic)
  - [2.5 Static Memory Footprint vs. Memory Traffic](#25-static-memory-footprint-vs-memory-traffic-important-distinction)

- [3. Compute (FLOPs) per Token](#3-compute-flops-per-token)
  - [3.1 Q/K/V and Output Projections](#31-qkv-and-output-projections)
  - [3.2 Attention Scores and Value Application](#32-attention-scores-and-value-application)
  - [3.3 FFN FLOPs (Unified Dense + MoE)](#33-ffn-flops-unified-dense--moe)
  - [3.4 LayerNorm and Elementwise FLOPs](#34-layernorm-and-elementwise-flops)
  - [3.5 Per-Device FLOPs per Layer Under TP, SP, EP, and PP](#35-per-device-flops-per-layer-under-tp-sp-ep-and-pp)

- [4. Compute vs. Memory Bound (Roofline Model)](#4-compute-vs-memory-bound-roofline-model)

- [5. Communication Time During Decoding](#5-communication-time-during-decoding)
  - [5.1 Pipeline Parallel (PP) Hop](#51-pipeline-parallel-pp-hop)
  - [5.2 Expert Parallel (EP) All-to-All (MoE Dispatch and Combine)](#52-expert-parallel-ep-all-to-all-moe-dispatch-and-combine)
  - [5.3 Tensor Parallel (TP) Communication](#53-tensor-parallel-tp-communication)
  - [5.4 Sequence Parallel (SP) Communication](#54-sequence-parallel-sp-communication)
  - [5.5 Total Communication Time Per Step on a PP Stage](#55-total-communication-time-per-step-on-a-pp-stage)

- [6. Partition Strategy and Hardware Latency](#6-partition-strategy-and-hardware-latency)
  - [6.1 Model Partition Strategy from HBM Constraints](#61-model-partition-strategy-from-hbm-constraints)
  - [6.2 Local and Networking Per-Step Latency](#62-local-and-networking-per-step-latency)
  - [6.3 When Each Layout / Attention-Mode Pays](#63-when-each-layout--attention-mode-pays)

- [7. Host-Side Overheads and Throughput](#7-host-side-overheads-and-throughput)
  - [7.1 Kernel-launch overhead](#71-kernel-launch-overhead)
  - [7.2 Per-step hardware window](#72-per-step-hardware-window)
  - [7.3 Per-sequence serving runtime overhead and throughput](#73-per-sequence-serving-runtime-overhead-and-throughput)

- [8. Speculative Decoding (MTP / EAGLE / Medusa)](#8-speculative-decoding-mtp--eagle--medusa)
  - [8.1 Verify-step setup](#81-verify-step-setup)
  - [8.2 Acceptance model and expected accepted tokens](#82-acceptance-model-and-expected-accepted-tokens)
  - [8.3 Verify-step roofline](#83-verify-step-roofline)
  - [8.4 Effective TPOT under speculative decoding](#84-effective-tpot-under-speculative-decoding)
  - [8.5 Where speculative decoding wins and loses](#85-where-speculative-decoding-wins-and-loses)

---

<div style="page-break-before: always;"></div>

# 1. Memory Footprint

This section defines parameter sizes and memory footprint for a given set of model parameters. The memory footprint include those from model weights, per-token activation/working memories, and KV cache. We avoid model-wide parameter aggregation here and instead focus on **per-layer** quantities, because pipeline-parallel stages own disjoint sets of layers. All parameter definitions assume stored precision of $b$ bytes per element (e.g., bf16 = 2 bytes).

**Layout and mode parameterization.** Per-device formulas in §1.4 and below (and analogously in §2.1 / §2.3 / §3.5 / §5.3 / §5.5) are written in terms of the per-component effective sharding factors $D_{\text{attn}}$, $D_{\text{exp}}$, $D_{\text{kv}}$, $D_{\text{emb}}$ and the collective group sizes $G_{TP}$, $G_{EP}$ from `notation.md §1` rather than raw $TP$ / $EP$ divisors. This lets a single expression cover all four production-relevant `(layout, attention_mode)` configurations — orthogonal + TP-attention, orthogonal + DP-attention (DSv3 / SGLang on disjoint TP and EP groups), co-located + DP-attention (DSv3 / SGLang on a single NVLink island, $\max(TP, EP)$ GPUs per replica), and co-located + TP-attention (DSr1 / NVL72 panel-(b) shape: TP=EP overlaid on the same GPU set with attention head-sharded across that group) — without conditional branching. Each downstream section opens with a summary table mapping the abstract factors used in that section to each of the four configurations; the `notation.md §1` lookup table is the canonical source.

---

## 1.0 Parameter Definitions (P vs W)

For any weight matrix $W$, the parameter count is

$$
P(W) = \text{number of elements in } W
$$

Total stored parameter memory is

$$
M_\theta = P \cdot b
$$

---

## 1.1 Model Parameter Memory

### Embedding and LM Head Parameters

Modern LLM architectures (GPT-3/4, LLaMA families, PaLM, Qwen, DeepSeek, etc.) typically use embedding dimension $E = H$, so all internal projections operate on vectors in $\mathbb{R}^H$.

For the token embedding:

$$
W_{\text{emb}} \in \mathbb{R}^{V \times H},
\quad
P_{\text{emb}} = V H
$$

If the LM head is tied with embeddings, i.e. $W_{lm}=W_{emb}^T$ so that $W_{emb}$ can be re-used:

$$
P_{\text{lm}} = 0
$$

If untied:

$$
W_{\text{lm}} \in \mathbb{R}^{H \times V},
\quad
P_{\text{lm}} = V H
$$

### Attention Parameters

For hidden size $H$, head dimension $d_{\text{head}}$, and KV dimension $H_{kv} = n_{kv} \, d_{\text{head}}$ (supporting grouped-query attention where $n_{kv} \le n_q$ [GQA] and multi-query attention where $n_{kv} = 1$ [MQA]):

- $W_Q \in \mathbb{R}^{H \times H}$  
- $W_K \in \mathbb{R}^{H \times H_{kv}}$  
- $W_V \in \mathbb{R}^{H \times H_{kv}}$  
- $W_O \in \mathbb{R}^{H \times H}$  

In GQA, each of the $n_q$ query heads produces a $d_{\text{head}}$-dimensional output; the concatenated result across all query heads is $n_q \times d_{\text{head}} = H$-dimensional, regardless of how many KV heads are used. The output projection therefore always maps $\mathbb{R}^H \to \mathbb{R}^H$.

Parameter counts:

$$
P_Q = H^2, \qquad
P_K = H H_{kv}, \qquad
P_V = H H_{kv}, \qquad
P_O = H^2
$$

We define attention parameters:

$$
P_{\text{attn}} = P_Q + P_K + P_V + P_O
$$

> **Variant note.** Models with non-MHA attention (Multi-head Latent Attention (MLA), sliding-window attention, DeepSeek Sparse Attention (DSA), hybrid linear / full attention) substitute a different $P_{\text{attn}}$ formula here. Per-variant decompositions and worked examples are in `attention.md` (e.g. `attention.md §3.3` for MLA's six-matrix per-layer parameter sum). The `decode.md` outer composition (per-device sharding, multi-layer aggregation, traffic, FLOPs roll-up) is unchanged — only this term is replaced.

### Unified FFN Parameters (Dense or MoE)

Each transformer layer contains an FFN module. In modern LLM architectures, this FFN is almost
always implemented as a gated MLP (GeLU/SiLU/GLU/SwiGLU–style) with **three** linear projections:

- a gate projection,
- an “up” projection (expansion),
- a “down” projection (contraction).

> **Convention:** This document assumes **gated FFN (SwiGLU/GeGLU style) throughout**, giving three weight matrices per FFN block. Standard (non-gated) FFN uses two matrices (up + down), yielding $2HI$ parameters. The gated form is used by LLaMA, Qwen, DeepSeek, and most modern LLMs.

For a hidden size $H$ and FFN intermediate dimension $I$, this yields an FFN parameter count

$$
P_{\text{FFN}} = 3 H I N_{\text{exp}}.
$$

Here $N_{\text{exp}}$ denotes the number of experts **per layer**:

- **For a dense MLP model:** $I = I_{\text{dense}}, \; N_{\text{exp}} = 1$.
- **For a MoE model:** $I = I_{\text{moe}}$, with $N_{\text{exp}} > 1$.

These two model cases are mutually exclusive **per layer**.

### LayerNorm parameters

LayerNorm or RMSNorm contain $\mathcal{O}(H)$ parameters (scale and optional bias). These are negligible compared to attention and FFN weights and are omitted in scaling formulas.

---

## 1.2 Activation Memory (Per-token Working Memory)

During autoregressive decoding, the model processes **one token at a time**. As a result, the only activations that need to be stored in memory are the **temporary, layer-local working buffers** used in the forward pass of the *current token*.  

These activations are **not reused** across layers or across tokens and therefore are:

- **not dependent on sequence length** $S$,
- **proportional to batch size** $B$ (each sequence in the batch needs its own working buffers),
- **not EP- or TP-sharded**,  
- and **extremely small** relative to model parameter memory (Section 1.1) and KV cache memory (Section 1.3).

Below we account for all activation tensors in a model layer that must be alive **concurrently** for one decoding token.

### Q, K, V projections

For the current hidden state $h \in \mathbb{R}^{H}$, the layer computes:

- $Q \in \mathbb{R}^{H}$
- $K \in \mathbb{R}^{H_{kv}}$
- $V \in \mathbb{R}^{H_{kv}}$

This contributes:

$$
H + 2H_{kv}
$$

### Attention score accumulation buffer (FlashAttention-like kernels)

Attention score computation normally requires a temporary buffer. FlashAttention-style fused kernels avoid storing full $S$-length score vectors and instead use a **single internal workspace** of size $H$ during streaming softmax.

This adds:

$$ + H
$$

### Attention output buffer

After applying attention weights to $V$ and combining across heads, we form the attention output $O_{\text{attn}} \in \mathbb{R}^{H}$.

This output must exist before the output projection is applied, contributing:

$$+ H$$

### FFN working buffer

Following attention and normalization, the FFN block needs at least one temporary buffer of size $H$ to hold either the FFN input or the FFN output before residual addition. Even with kernel fusion, this buffer cannot always overlap with the attention intermediates.

This adds:

$$+ H$$

Summing all simultaneously required buffers per sequence:

$$
P_{\text{act}} = 4H + 2H_{kv}
$$

In bytes, for a batch of $B$ sequences:
$$
M_{\text{act,layer}} = B \cdot (4H + 2H_{kv}) \cdot b
$$

This footprint is **small** compared to parameter memory and KV cache, even at large batch sizes. For example, at $B=128$, $H=8192$, $H_{kv}=1024$, $b=2$: $M_{\text{act,layer}} \approx 9$ MB per layer — negligible against hundreds of GB of parameter memory.

---

## 1.3 KV Cache Memory

Section 1.1 described the memory footprint of model parameters (static), and Section 1.2 covered the
activation memory required during decoding (per-token, dynamic). This section describes the **KV cache**, which is a *runtime* structure generated during the **pre-fill phase**, when the model processes the entire input sequence of length $S$.

During pre-fill, each attention layer produces:

- one key vector of dimension $H_{kv}$,
- one value vector of dimension $H_{kv}$,

for each input token. Because decoding adds only one new token at a time, the vast majority of KV memory comes from **pre-fill**, not decoding.

For a single attention layer, the KV cache consists of:

- Keys: $K \in \mathbb{R}^{S \times H_{kv}}$  
- Values: $V \in \mathbb{R}^{S \times H_{kv}}$

The KV cache size scales with $H_{kv} = n_{kv} d_{\text{head}}$; using grouped-query attention ($n_{kv} < n_q$) [GQA] or multi-query attention ($n_{kv} = 1$) [MQA] directly reduces this footprint.

> **Variant note.** Multi-head Latent Attention (MLA) further compresses the per-token KV cache by storing a head-shared latent of dimension $d_c + d_{qk,\mathrm{rope}}$ instead of $2 H_{kv}$. For DSv3-class numbers this is roughly 25× smaller than the dense MHA equivalent. See `attention.md §3.4` for the MLA per-token-per-layer formula and `attention.md §3.6` for sharding behavior under TP-attn / DP-attn.

Thus, the total KV elements for one layer are:

$$
P_{KV, layer} = S \cdot (2H_{kv}) = 2 S H_{kv}
$$

In bytes (per sequence):

$$
M_{\text{KV,layer}} =
2 S H_{kv} \cdot b \quad \text{(per sequence, per layer)}
$$

This is **static** once pre-fill is complete; decoding contributes only an additional $2H_{kv} b$ per
generated token, which is negligible relative to the full cache.

---

## 1.4 Per-device Memory Footprint After Parallelism Sharding

So far we've completed the all the memory footprint estimation for a model layer. When we introduce different parallelism schemes, some of these memories would be sharded by one or more of these parallelism dimensions, resulting in a somewhat complicated memory aggregateion per device. We now describe how these parameters are distributed across devices under PP/EP/TP/SP, and then derive simple modeling approximations.

This section uses the per-component effective sharding factors $D_{\text{attn}}$, $D_{\text{exp}}$, $D_{\text{kv}}$, $D_{\text{emb}}$ from `notation.md §1` rather than raw $TP$ / $EP$ divisors, so a single expression covers all four production-relevant configurations. Resolving the abstract factors:

| configuration | $D_{\text{attn}}$ | $D_{\text{exp}}$ (MoE) | $D_{\text{kv}}$ | $D_{\text{emb}}$ |
|---|---|---|---|---|
| orthogonal + TP-attn | $TP$ | $TP \cdot EP$ | $TP$ (head) | $TP$ |
| co-located + TP-attn | $TP$ | $EP$ | $TP$ (head) | $TP$ |
| orthogonal + DP-attn | $1$ | $TP \cdot EP$ | $TP$ (seq) | $TP$ |
| co-located + DP-attn | $1$ | $EP$ | $\max(TP, EP)$ (seq) | $TP$ |

Dense FFN always uses $D_{\text{exp}} = TP$ (no EP axis to overlap; co-location does not apply). KV memory and traffic carry an additional $/SP$ factor on top of $D_{\text{kv}}$ when sequence parallelism is enabled. Under co-location the structural constraint $TP = EP$ holds, so $\max(TP, EP)$ and $TP$ collapse to the same value in the co-located rows. Formulas below use these symbols directly without further repetition of the table.

### Per-device Parameter Memory

Each transformer layer has two parameter groups:

- **Attention parameters** $P_{\text{attn}}$, sharded by the effective attention divisor $D_{\text{attn}}$.
- **FFN parameters** $P_{\text{FFN}}$, sharded by the effective expert/FFN divisor $D_{\text{exp}}$. Dense FFN layers always use $D_{\text{exp}} = TP$ (no EP axis exists to overlap with), so the same formula applies to both dense and MoE FFNs given the layer-type-aware $D_{\text{exp}}$.

For a *single layer* on a device, the stored parameter memory is

$$
M_{\theta,\text{layer}} =
\frac{P_{\text{attn}}\, b}{D_{\text{attn}}}
\;+\;
\frac{P_{\text{FFN}}\, b}{D_{\text{exp}}}
$$

Pipeline parallelism (PP) assigns **disjoint sets of layers** to different stages. Let $L_s$ be the set of layers that live on PP stage $s$, and let $M_{\theta,\text{layer},\ell}$ be the per-layer memory from the expression above.

Excluding embeddings and LM head, the parameter memory per device on PP stage $s$ is

$$
M_{\theta,\text{layers}}^{(s)} =
\sum_{\ell \in L_s}
M_{\theta,\text{layer},\ell}
$$

Embeddings and LM head appear only on two stages:

- **Intermediate PP stages** (no embedding / LM head):
  $$
  M_{\theta,\text{device}}^{(\text{mid})} =
  M_{\theta,\text{layers}}^{(\text{mid})}
  $$

- **First PP stage** (with token embedding):
  $$
  M_{\theta,\text{device}}^{(1)} =
  M_{\theta,\text{layers}}^{(\text{mid})}
  \;+\;
  \frac{P_{\text{emb}}\, b}{D_{\text{emb}}}
  $$

- **Final PP stage** (with LM head):
  $$
  M_{\theta,\text{device}}^{(\text{PP})} =
  M_{\theta,\text{layers}}^{(\text{mid})}
  \;+\;
  \frac{P_{\text{lm}}\, b}{D_{\text{emb}}}
  $$

If each intermediate PP stage holds approximately $L/PP$ layers of similar size, and we use representative per-layer values $P_{\text{attn}}$ and $P_{\text{FFN}}$, then

$$
M_{\theta,\text{device}}^{(\text{mid})} =
\frac{L}{PP} M_{\theta,\text{layer}} =
\frac{L}{PP}\;
\left(
\frac{2H^2 + 2 H H_{kv}}{D_{\text{attn}}}
+
\frac{3 H I N_{\text{exp}}}{D_{\text{exp}}}
\right) b
$$

For capacity planning we use a worst-case PP stage budget, adding one $\frac{VH}{D_{\text{emb}}}b$ term to account for embedding/LM weights residing on boundary stages. Intermediate stages are slightly smaller. Therefore:

$$
M_{\theta,\text{device}} =
\frac{L}{PP}\;
\left(
\frac{2H^2 + 2 H H_{kv}}{D_{\text{attn}}}
\;+\;
\frac{3 H I N_{\text{exp}}}{D_{\text{exp}}}
\right) b
+\frac{VH}{D_{\text{emb}}} b
$$

For a **dense MLP model**: $I = I_{\text{dense}}$, and $N_{\text{exp}} = EP = 1$ (so $D_{\text{exp}} = TP$ by definition).

For a **MoE model**: $I = I_{\text{moe}}$, with $N_{\text{exp}} > 1$ and $EP \ge 1$; consult the `notation.md §1` lookup table to resolve $D_{\text{exp}}$ under co-location.

### Mixed MoE/Dense Architectures

Many modern architectures use a **mixed** design where only some layers are MoE (e.g., alternating dense and MoE layers, or MoE only in deeper layers). For such models, the parameter memory must be computed separately for dense and MoE layers, with the layer-type-aware $D_{\text{exp}}$ resolving to $TP$ on dense layers and to the table value on MoE layers:

$$
M_{\theta,\text{device}} =
M_{\theta,\text{dense}} + M_{\theta,\text{moe}} + \frac{VH}{D_{\text{emb}}} b
$$

where:

$$
M_{\theta,\text{dense}} =
\frac{L_{\text{dense}}}{PP}\;
\left(
\frac{2H^2 + 2 H H_{kv}}{D_{\text{attn}}}
\;+\;
\frac{3 H I_{\text{dense}}}{TP}
\right) b
$$

$$
M_{\theta,\text{moe}} =
\frac{L_{\text{moe}}}{PP}\;
\left(
\frac{2H^2 + 2 H H_{kv}}{D_{\text{attn}}}
\;+\;
\frac{3 H I_{\text{moe}} N_{\text{exp}}}{D_{\text{exp}}}
\right) b
$$

Dense layers use $EP = 1$ and $N_{\text{exp}} = 1$ implicitly (FFN always TP-sharded only); MoE layers use the specified $EP$ and $N_{\text{exp}}$ values with $D_{\text{exp}}$ resolved from the layout table.

### Per-device Activation Memory

The per-layer working activation footprint for a batch of $B$ sequences is $B \cdot (4H + 2H_{kv})$.

In standard sequential layer execution, only one layer's activation buffers are live at any time —
each layer's output overwrites the previous layer's buffer before the next layer begins. The $L/PP$ multiplier therefore does *not* appear: earlier layers' activations are not retained while later layers execute. A $2\times$ factor could apply when double-buffering for PP communication overlap, but this is negligible in practice and omitted.

The per-device activation memory during decoding is therefore one layer's worth:

$$
M_{\text{act,device}}(B) =
B \cdot (4H + 2H_{kv}) \, b
$$

### Per-device KV Cache Memory

Only attention layers produce KV cache. KV is sharded by the per-device head-or-sequence factor $D_{\text{kv}}$ (notation.md §1) and additionally by SP when sequence parallelism is enabled. EP and DP do not modify KV layout; PP only affects how many layers are assigned to a stage.

The semantic of $D_{\text{kv}}$ depends on the attention mode:

- Under **TP-attn** (default), $D_{\text{kv}} = TP$ and the divisor is the head-shard across the $TP$ ranks.
- Under **DP-attn** (DSv3 / SGLang), $D_{\text{kv}} = TP$ for orthogonal layout (sequence-shard across the $TP$ ranks now acting as DP-attn ranks; per-device byte count unchanged from TP-attn) or $D_{\text{kv}} = \max(TP, EP)$ for co-located layout (sequence-shard across the entire replica's GPU set).

For a batch of $B$ resident sequences, each carries its own KV history. We factor the per-device KV memory into a **per-sequence** building block $M_{\text{KV,token}}$ that captures the per-device byte cost of one sequence's KV history (the symbol uses "token" to parallel the per-token compute and traffic terms in §2.3, §3.5 — for decode each active sequence equals one new token per step):

$$
M_{\text{KV,token}}
\;=\;
\frac{L}{PP}
\cdot
\frac{(2 S H_{kv}) b}{D_{\text{kv}} \cdot SP}
$$

and the **per-device** KV memory footprint as the linear-in-$B$ scaling:

$$
M_{\text{KV,device}}(B) \;=\; B \cdot M_{\text{KV,token}}
\;=\;
B \cdot \frac{L}{PP}
\cdot
\frac{(2 S H_{kv}) b}{D_{\text{kv}} \cdot SP}
$$

which:

- Scales linearly with $B$ — every active sequence in the batch holds its own KV cache.
- For long-context inference (e.g., $S \in [16\text{k}, 128\text{k}]$), $B \cdot S(2H_{kv})b$ is large enough that KV can exceed parameter memory unless aggressively reduced through $D_{\text{kv}}$ and $SP$.
- Each decoded token adds only $2H_{kv} b$ per sequence, which is negligible compared to the pre-fill KV footprint.

### Total Per-device Static Memory Footprint

Summing all the memory footprint we derive from section 1.1 - 1.4 together, we can therefore get the "minimum required" memory size for the device to host the model under a particular PP/EP/TP/SP partition.

$$
M_{\text{device}}^{\text{total}}(B) = M_{\theta,\text{device}} + M_{\text{act,device}}(B) + M_{\text{KV,device}}(B)
$$

Note: $M_{\theta,\text{device}}$ is $B$-independent (weights are loaded once and shared across all sequences in the batch); $M_{\text{act,device}}(B)$ and $M_{\text{KV,device}}(B)$ both scale linearly with $B$.

For **uniform architectures** (all dense or all MoE):

$$
M_{\text{device}}^{\text{total}} =
\frac{L}{PP}\;
\left[
\frac{2H^2 + 2 H H_{kv}}{D_{\text{attn}}}
+
\frac{3 H I N_{\text{exp}}}{D_{\text{exp}}}
\right] b
+
B \cdot \frac{L}{PP} \cdot \frac{2 S H_{kv}}{D_{\text{kv}} \cdot SP} \cdot b
+
B(4H + 2H_{kv}) b
+\frac{VH}{D_{\text{emb}}} b
$$

For a dense model: $I = I_{\text{dense}}, \text{ } N_{\text{exp}}=EP=1$ (with $D_{\text{exp}} = TP$).

And for a MoE model: $I = I_{\text{moe}}$.

For **mixed MoE/dense architectures** (where $L_{\text{moe}} < L$):

$$
\begin{aligned}
M_{\text{device}}^{\text{total}} = \;&
\frac{L_{\text{dense}}}{PP}\;
\left[
\frac{2H^2 + 2 H H_{kv}}{D_{\text{attn}}}
+
\frac{3 H I_{\text{dense}}}{TP}
\right] b \\
+\;&
\frac{L_{\text{moe}}}{PP}\;
\left[
\frac{2H^2 + 2 H H_{kv}}{D_{\text{attn}}}
+
\frac{3 H I_{\text{moe}} N_{\text{exp}}}{D_{\text{exp}}}
\right] b \\
+\;&
B \cdot \frac{L}{PP} \cdot
\frac{2 S H_{kv}}{D_{\text{kv}} \cdot SP}
\cdot b \\
+\;&
B(4H + 2H_{kv}) b
+\frac{VH}{D_{\text{emb}}} b
\end{aligned}
$$

*Dense FFN keeps $/TP$ regardless of layout (no EP axis to overlap); MoE FFN, attention weights, KV, and embeddings all use the abstract divisors with the layout table.*

Note: Activation memory ($B(4H+2H_{kv})b$, one layer at a time) and KV cache apply to all $L$ layers. KV cache uses the total layer count $L/PP$ since all layers' KV tensors are concurrently resident.

---

<div style="page-break-before: always;"></div>

# 2. Memory Traffic During Decoding

Section 1 quantified the *static* memory footprint of the model — how many bytes of parameters, KV cache, and activations must **fit** in device HBM.

This section instead focuses on **memory traffic per decode step**, i.e., the bytes that must flow between HBM and compute cores during decoding. This traffic directly determines the memory-bound component of decoding performance (Section 4's roofline model).

**Crucial Distinction for Decoding:**
In autoregressive decoding, each step generates one new token per active sequence (B tokens per step for a batch of B). Unlike prefill — where weights are loaded once and reused across $S_{\text{input}}$ tokens within a single pass — decoding reloads the **entire model weight matrix** from HBM **every step**, regardless of $B$. KV cache, by contrast, is read per-sequence: each of the $B$ active sequences streams its own KV history. So per step:

- **Weight traffic** ($T_{\theta,\text{device}}$): independent of $B$ (loaded once, shared across all $B$ tokens — equivalently, per-token weight traffic = $T_{\theta,\text{device}}/B$ shrinks as $B$ grows).
- **KV traffic**: scales linearly with $B$ (each sequence reads its own history).

Throughout this section, the per-step traffic quantities below are consistent with this asymmetry. Optimizations like FlashAttention or Fused-MLP do **not** reduce weight traffic; they only reduce the traffic of intermediate activations.

This section uses the per-component effective sharding factors $D_{\text{attn}}$, $D_{\text{exp}}$, $D_{\text{kv}}$, $D_{\text{emb}}$ from `notation.md §1`. The four-row lookup mapping `(layout, attention_mode)` to factor values is given once in §1.4 above (and canonically in `notation.md §1`); we do not repeat it here. Reminder of the SP-specific adjustment: KV traffic carries an additional $/SP$ factor on top of $D_{\text{kv}}$ when sequence parallelism is enabled.

---

## 2.1 Model Parameter Traffic $T_{\theta,\text{device}}$

Following Section 1, we use:

- $P_{\text{attn}}$: Q/K/V/O projection parameters  
- $P_{\text{FFN}}$: dense FFN (or non-expert MoE core) parameters  

$P_{\text{emb}}$ is small (one row read per token) and absorbed into the embedding lookup overhead — we drop it from the steady-state traffic model. $P_{\text{lm}}$ is **kept** as a stage-PP-1-only term (see "LM head parameter traffic" below) — it can rival ~10% of the body for large $V$ and warrants explicit accounting.

### Attention parameter traffic

Because $P_{\text{attn}}$ is defined **per layer**, a PP stage with $L_{\text{stage}} = L/PP$ layers has $L_{\text{stage}} P_{\text{attn}}$ attention parameters. These are sharded by $D_{\text{attn}}$.

Since every weight must be read per token:

$$
T_{\theta,\text{attn}} =
\frac{L}{PP}
\cdot
\frac{P_{\text{attn}} \, b}{D_{\text{attn}}}
$$

> **Variant note.** For Multi-head Latent Attention (MLA) models, $P_{\text{attn}}$ above is the per-layer MLA parameter sum from `attention.md §3.3`; the per-rank sharding split also differs ($W_{DQ}$, $W_{DKV}$ are not head-shared) and is given by `attention.md §3.6`.

### FFN parameter traffic

Similarly, the FFN parameters $P_{\text{FFN}}$ are sharded by $D_{\text{exp}}$. Although fused kernels (e.g., FlashMLP) avoid writing intermediate activations (like the gate tensor) to HBM, they still require reading the gate, up, and down projection weights fully.

$$
T_{\theta,\text{FFN}} =
\frac{L}{PP}\;
\frac{P_{\text{FFN}}}{D_{\text{exp}}}\; b
$$

### Final parameter-traffic expression (dense layers)

For dense layers (every weight is read every step), combining these terms:

$$
T_{\theta,\text{device}}^{\text{dense}}
=
\frac{L}{PP}
\left(
  \frac{P_{\text{attn}}}{D_{\text{attn}}}
  +
  \frac{P_{\text{FFN}}}{D_{\text{exp}}}
\right) b =
\frac{L}{PP}\;
\left(
\frac{2H^2 + 2 H H_{kv}}{D_{\text{attn}}}
\;+\;
\frac{3 H I_{\text{dense}}}{TP}
\right) b
$$

(Dense FFN keeps $D_{\text{exp}} = TP$ regardless of layout, no EP axis to overlap.) For MoE layers the expert-FFN term is *not* the full-footprint expression; see the mixed-architecture form below.

### Mixed MoE/Dense Architectures

For mixed architectures, parameter traffic is computed separately for dense and MoE layers (with dense FFN keeping $D_{\text{exp}} = TP$ regardless of layout):

$$
T_{\theta,\text{device}} =
T_{\theta,\text{dense}} + T_{\theta,\text{moe}}
$$

where:

$$
T_{\theta,\text{dense}} =
\frac{L_{\text{dense}}}{PP}\;
\left(
\frac{2H^2 + 2 H H_{kv}}{D_{\text{attn}}}
\;+\;
\frac{3 H I_{\text{dense}}}{TP}
\right) b
$$

For MoE layers, *per-step traffic* differs from the *static footprint* (cf. §1.4): expert weights enter HBM traffic only for the experts the current batch actually selects, while the static footprint must hold every expert because the next step may select any of them. The conventional "all weights read each step" simplification is exact for dense layers but over-counts for MoE at small $B$. Writing $\mathbb{E}[N_{\text{exp,touched}}^{\text{rank}}]$ for the expected number of unique experts touched on a single rank per step:

$$
T_{\theta,\text{moe}} =
\frac{L_{\text{moe}}}{PP}\;
\left(
\frac{2H^2 + 2 H H_{kv}}{D_{\text{attn}}}
\;+\;
3 H I_{\text{moe}} \cdot \mathbb{E}[N_{\text{exp,touched}}^{\text{rank}}]
\right) b
$$

Under the uniform-routing assumption (each token's $k$ active experts drawn independently and uniformly from the global $N_{\text{exp}}$ pool, then routed to whichever rank holds the selected expert), with $N_{\text{exp/rank}} = N_{\text{exp}} / D_{\text{exp}}$ experts held on each rank and $t_{\text{tokens/rank}} = B k_{\text{active}} / D_{\text{exp}}$ expert-touch events per rank per step:

$$
\mathbb{E}[N_{\text{exp,touched}}^{\text{rank}}]
= N_{\text{exp/rank}}\!\left(1 - \!\left(1 - \frac{1}{N_{\text{exp/rank}}}\right)^{\!t_{\text{tokens/rank}}}\right)
$$

Asymptotics:

- $B \to 0$: $\mathbb{E}[N_{\text{exp,touched}}^{\text{rank}}] \to t_{\text{tokens/rank}}$ (linear in $B$); traffic dominated by attention + a small expert slice.
- $B k_{\text{active}} \gg N_{\text{exp}}$: $\mathbb{E}[N_{\text{exp,touched}}^{\text{rank}}] \to N_{\text{exp/rank}}$; expression recovers the full-footprint form $3 H I_{\text{moe}} N_{\text{exp}} / D_{\text{exp}}$.

**Routing-uniformity caveat.** Real production routers can deviate from uniform routing: load-balancing-loss anti-correlation, expert hot-spotting, or capacity-factor saturation can make some experts touched more often than others, raising $\mathbb{E}[N_{\text{exp,touched}}^{\text{rank}}]$ above the uniform-routing value at fixed $B$. Modeling this requires per-deployment routing statistics not generally available; the uniform-routing assumption above is documented as a known source of model inaccuracy at small $B$.

### LM head parameter traffic

The LM head ($H \to V$ projection) is sharded by $D_{\text{emb}}$ across the vocab dimension and **lives only on the last PP stage** (stage $PP{-}1$). It is not divided by $PP$, $EP$, or $SP$. The per-step weight read on stage $PP{-}1$ is:

$$
T_{\text{LM},\theta,\text{device}} = \frac{V H \, b}{D_{\text{emb}}} \quad \text{(stage } PP{-}1 \text{ only)}
$$

Because the LM head fires once per step (not per layer), it is bookkept as a separate additive term rather than folded into the per-layer $T_{\theta,\text{device}}$. The roofline composition in §6 / §7 adds $t_{\text{LM}}$ on top of the per-stage body cost; on stages $0..PP{-}2$ this term is zero.

Section 1 showed that the per-layer activation footprint for a single decoding token is small. However, without optimization, the traffic to read/write these activations—especially the $S \times S$ attention scores—would be massive ($O(S^2)$).

### The Role of FlashAttention

**FlashAttention** [FA1, FA2] avoids materializing the $S \times S$ score matrix in HBM by streaming the tiled attention computation through on-chip SRAM. More precisely: Q, K, V reads remain $O(SH)$; the $O(S^2 d^2 / M)$ score matrix IO (per [FA1] Theorem 2, where $d = d_{\text{head}}$ and $M$ is SRAM size) is reduced to $O(S^2 d / \sqrt{M})$ via tiling, compared to $O(S^2 d)$ for standard attention. For large $S$ and modern GPU SRAM sizes, this makes the $O(S^2)$ term negligible and leaves KV reads as the dominant activation traffic.

Because FlashAttention drastically reduces the score matrix traffic, the residual activation traffic (hidden-state loads/stores, FFN buffers) is $O(H)$ per layer — negligible compared to the weight and KV cache terms for large models. We drop $T_{\text{act,device}}$ from the traffic model here. Residual kernel-level activation overhead is treated as an empirical correction in `framework.md`.

---

## 2.3 KV Cache Traffic $T_{\text{KV,device}}(B)$

KV cache must be **fully read** for each new token to compute attention against the history.
For large $S$, the write term (appending the new token) is negligible compared to reading the history.

The per-sequence per-layer KV access is approximately:

$$
T_{\text{KV,layer}}
\approx
2 S H_{kv} \, b \quad \text{(per sequence, per layer)}
$$

> **Variant note.** Multi-head Latent Attention (MLA) replaces $2 H_{kv}$ above with the head-shared latent dimension $(d_c + d_{qk,\mathrm{rope}})$, typically 25× smaller than the equivalent dense MHA. See `attention.md §3.4`.

KV is sharded by $D_{\text{kv}}$ (head or sequence depending on layout/mode; see §1.4) and by $SP$ (sequence parallelism, when enabled). Each device sees a $\frac{1}{D_{\text{kv}} \cdot SP}$ shard of the per-layer traffic.

We factor the per-device KV traffic into a **per-token** building block $T_{\text{KV,token}}$ (per-device traffic generated by one active sequence per step, summed across the $L/PP$ layers on the stage) and a linear-in-$B$ aggregator. The naming parallels the per-token compute terms $F_{\text{attn,token,device}}$ and $F_{\text{ffn,token,device}}$ in §3.5 — for decode each active sequence produces one new token per step, so per-token and per-sequence coincide:

$$
T_{\text{KV,token}}
\;\approx\;
\frac{L}{PP}
\cdot
\frac{2 S H_{kv} \, b}{D_{\text{kv}} \cdot SP}
$$

For a PP stage with $B$ active sequences in the batch — each streaming its own KV history — the **per-step per-device KV traffic** is:

$$
T_{\text{KV,device}}(B) \;=\; B \cdot T_{\text{KV,token}}
\;\approx\;
B \cdot \frac{L}{PP}
\cdot
\frac{2 S H_{kv} \, b}{D_{\text{kv}} \cdot SP}
$$

The linear $B$ scaling reflects that each sequence in the batch reads its own KV cache independently. FlashAttention does **not** reduce this term: keys and values from history must always be loaded to compute each sequence's current-token attention, regardless of tiling strategy.

---

## 2.4 Total and Effective Traffic

Combining the expressions derived in Sections 2.1–2.3 (with activation traffic dropped as negligible), the **effective** total per-step traffic is:

$$
T_{\text{step,device}}^{\text{eff}}(B)
\approx
T_{\theta,\text{device}}
+
T_{\text{KV,device}}(B)
$$

Weight traffic is $B$-independent (one load per step); KV traffic scales linearly with $B$. Substituting yields the **final fully expanded expression**:

$$
T_{\text{step,device}}^{\text{eff}}(B)
\;\approx\;
\underbrace{\frac{L}{PP}
\left(
\frac{2H^2 + 2 H H_{kv}}{D_{\text{attn}}}
+
\frac{3 H I N_{\text{exp}}}{D_{\text{exp}}}
\right) b}_{T_{\theta,\text{device}}\ \text{(weights, once per step)}}
\;+\;
\underbrace{B \cdot \frac{L}{PP} \cdot \frac{2 S H_{kv}}{D_{\text{kv}} \cdot SP} b}_{T_{\text{KV,device}}(B)\ \text{(KV, per sequence)}}
$$

The first group is **weight traffic** (loaded once per step regardless of $B$), and the second is **KV cache traffic** (each of the $B$ active sequences reads its own history).

---

## 2.5 Static Memory Footprint vs. Memory Traffic (Important Distinction)

Sections 1 and 2 play different roles in the overall performance model:

**Static Memory Footprint (Section 1)** Determines whether a $(DP, PP, EP, TP, SP)$ configuration can *fit* on a device (Capacity Constraint)

$$
M_{\text{device}}^{\text{total}} \le M_{\text{HBM}}
$$

**Memory Traffic (Section 2)** Determines the *bandwidth-limited latency* per decode step (Bandwidth Constraint)

$$
t_{\text{mem}}(B) =
\frac{T_{\text{step,device}}^{\text{eff}}(B)}
     {BW_{\text{mem}}}
$$

This distinction is critical: Section 1 tells us **which parallelism configurations are viable**, while Section 2 tells us **how fast decoding can proceed** for those viable configurations.

---

# 3. Compute (FLOPs) per Token

During inference, the FLOPs required to generate a token depend on the transformer layer structure. We distinguish between:

- **Prefill FLOPs (GEMM-dominant):** $O(S^2)$ across the full input sequence  
- **Decoding FLOPs (GEMV-dominant):** $O(S)$ for one additional token  

This section focuses on **decoding FLOPs**, which determine TPS throughput. All FLOPs below represent **per-token**, **per-layer**, **decoding** FLOPs.

---

## 3.1 Q/K/V and Output Projections

Q, K, and V projections are vector–matrix multiplications of shapes:

- Q: $[1 \times H] \cdot [H \times H]$  
- K: $[1 \times H] \cdot [H \times H_{kv}]$  
- V: $[1 \times H] \cdot [H \times H_{kv}]$  
- Output: $[1 \times H] \cdot [H \times H]$

### Projection FLOPs

$$
F_Q = 2H^2, \qquad
F_K = 2H H_{kv}, \qquad
F_V = 2H H_{kv}, \qquad
F_O = 2H^2.
$$

Where the factor 2 accounts for each multiply-accumulate pair in the standard GEMV convention.

### Total

$$
F_{\text{proj}} = 4H^2 + 4H H_{kv}
$$

If $H_{kv} = H$ (MHA), this reduces to $8H^2$.

> **Variant note.** Multi-head Latent Attention (MLA) replaces the projection FLOPs above with the down / up-projection cascade detailed in `attention.md §3.7` ($W_{DQ}$, $W_{UQ}$, $W_{DKV}$, $W_O$ plus optional $W_{UK}$, $W_{UV}$ depending on execution mode). The per-step total differs structurally between materialized and absorbed modes; both are given in `attention.md §3.5` (execution modes) and `§3.7` (per-mode FLOP breakdown).

## 3.2 Attention Scores and Value Application

During decoding, the newly generated token attends to all $S$ cached tokens in the KV cache for this layer. Conceptually, for each layer we can treat the cached keys and values as:

- $K_{\text{cache}} \in \mathbb{R}^{S \times H_{kv}}$  
- $V_{\text{cache}} \in \mathbb{R}^{S \times H_{kv}}$,

where $H_{kv} = n_{kv} d_{\text{head}}$ is the total KV projection dimension.

### Scores (Q · Kᵀ)

Each of the $n_q$ query heads independently computes a dot product against its corresponding (broadcast) KV head over $S$ cached positions. Per query head: $2 d_{\text{head}} S$ FLOPs. Summed over all $n_q$ query heads:

$$
F_{\text{score}} = 2 \, n_q \, d_{\text{head}} \, S = 2 S H
$$

> **GQA note:** In GQA ($n_{kv} < n_q$), each KV head is shared by $n_q / n_{kv}$ query heads [GQA]. The KV cache **memory** scales with $H_{kv}$ (only $n_{kv}$ unique heads are stored), but the attention **FLOPs** scale with $H$ because every query head independently computes attention scores and value-weighted sums.

### Value application (Attn · V)

After applying softmax to the scores, each of the $n_q$ query heads computes a weighted sum over the $S$ cached values of its corresponding value head. Per head: $2 d_{\text{head}} S$ FLOPs. Total:

$$
F_{\text{value}} = 2 S H
$$

### Total KV attention FLOPs

Combining the two:

$$
F_{\text{attn,KV}} = F_{\text{score}} + F_{\text{value}} = 4 S H
$$

This term captures the **sequence-length-dependent** attention cost during decoding. For MHA ($n_{kv} = n_q$, $H_{kv} = H$), this is numerically identical to the older $4SH_{kv}$ formulation; the distinction matters for GQA/MQA models where $H_{kv} < H$.

## 3.3 FFN FLOPs (Unified Dense + MoE)

To match the parameter definitions in Section 1.1, we express FFN FLOPs using a **unified formulation** that works for both dense FFN layers and MoE layers.

### Dense FFN FLOPs

For a gated FFN (the convention assumed throughout; see §1.1), a dense FFN consists of three GEMVs:

- a gate projection: $H \rightarrow I$,
- an up projection: $H \rightarrow I$, and
- a down (contraction) projection: $I \rightarrow H$.

Each GEMV costs $2HD$ FLOPs, giving:

$$
F_{\text{ffn,dense}} = 6 H I_{dense}
$$

> **Convention note — gated MLPs:** §1.1 counts **three** weight matrices for a gated FFN (gate, up, and down projections), giving $P_{\text{FFN}} = 3HI$ parameters. Each GEMV contributes $2HI$ FLOPs, so the exact total is $6HI$ — which this document uses throughout. The training scaling-law literature ([KAPLAN-SCALING], [CHINCHILLA]) uses $4HI$, which corresponds to a non-gated 2-matrix FFN with expansion ratio $4H$. Since this model targets **inference** of modern gated-MLP architectures (LLaMA, Qwen, DeepSeek, Mistral), we use the exact $6HI$ to ensure accurate compute-bound predictions (prefill TTFT, batched decode at $B > B^*$). With $6HI$ FLOPs and $3HI$ parameters, the FLOP-to-parameter ratio is exactly $2$ for every weight matrix, yielding the clean OI result $\text{OI} = 2/b$ without approximation.

Dense FFN layers always have $EP = 1$.

### MoE FFN FLOPs

For MoE layers, each token is routed to $k$ active experts (top-$k$ gating) [DEEPSPEED-MOE]; in practice $k=1$ [SWITCH] or $k=2$ [MIXTRAL]:

- the **router** is applied to the full hidden vector:
  $$
  F_{\text{router}} = 2 H N_{\text{exp}}
  $$
  where $N_{\text{exp}}$ is the total number of experts.

- for each of the $k$ selected experts for this token, the FFN computation is:
  $$
  F_{\text{expert}} = 6 H I_{\text{moe}}
  $$

Thus the MoE FFN FLOPs for one token in one layer are:

$$
F_{\text{ffn,moe}} =
F_{\text{router}}
+
k \cdot F_{\text{expert}} =
2 H N_{\text{exp}}
+
k (6 H I_{\text{moe}})
$$

### Unified FFN FLOP Term

We now define **effective FFN FLOP parameters**:

- Effective FFN “matrix” dimension:
  $$
  I_{\text{eff}} =
  \begin{cases}
  I_{\text{dense}}, & \text{dense layer}, \\
  k I_{\text{moe}}, & \text{MoE layer},
  \end{cases}
  $$

- Effective router multiplicity:
  $$
  N_{\text{eff}} =
  \begin{cases}
  0, & \text{dense layer}, \\
  N_{\text{exp}}, & \text{MoE layer}.
  \end{cases}
  $$

With these definitions, both dense and MoE FFN FLOPs can be written in a **single unified form**:

$$
F_{\text{ffn}} = 6H I_{\text{eff}} + 2H N_{\text{eff}}
$$

This matches:

- Dense layer:
  $F_{\text{ffn}} = 6H I_{\text{dense}} + 2H \cdot 0 = 6H I_{\text{dense}}$
- MoE layer:
  $F_{\text{ffn}} = 6H (k I_{\text{moe}}) + 2H N_{\text{exp}} = 6H I_{\text{moe}} k + 2H N_{\text{exp}}$

> **Note on MoE compute vs. traffic.** The fixed factor $k$ in $F_{\text{ffn,moe}}$ stays exact even at small $B$, in contrast to MoE *weight traffic* (§2.1 *MoE weight traffic*), where bytes-read-per-step depends on the expected number of distinct experts touched on each rank and is genuinely sub-linear in $B$. The asymmetry is the per-(token, expert) coupling: **compute is independent** — two tokens hitting the same expert do two separate gated MLP evaluations, so $B \cdot k$ correctly counts total expert invocations and the per-rank batched FLOPs are exactly $B \cdot k \cdot 6 H I_{\text{moe}} / D_{\text{exp}}$ under uniform routing (no expectation needed). **Traffic is shared** — one HBM read of an expert's weights serves all tokens hitting it that step — so it saturates sub-linearly in $B$ and requires the touched-experts expectation formula. The same per-(token, expert) independence applies to the router term $2 H N_{\text{exp}}$, which fires once per token regardless of routing collisions.

---

## 3.4 LayerNorm and Elementwise FLOPs

LayerNorm, RMSNorm, residual additions, and elementwise ops scale linearly with $H$ and are ~4 orders of magnitude smaller than the dominant FFN FLOPs for large models. We drop $F_{\text{norm}}$ from all per-device expressions. Norm overhead is an empirical correction handled in `framework.md`.

---

## 3.5 Per-Device FLOPs per Layer Under TP, SP, EP, and PP

For a single decoding token, the FLOPs for one transformer layer are:

$$
F_{\text{layer}}
\approx
F_{\text{proj}} + F_{\text{attn,KV}} + F_{\text{ffn}},
$$

where:

- $F_{\text{proj}} = 4H^2 + 4H H_{kv}$ (Section 3.1),
- $F_{\text{attn,KV}} = 4 S H$ (Section 3.2),
- $F_{\text{ffn}}$ is dense or MoE (Section 3.3).

$F_{\text{norm}}$ is dropped per Section 3.4.

This subsection uses the per-component effective sharding factors from `notation.md §1`; the four-row lookup mapping `(layout, attention_mode)` to factor values is in §1.4 above (canonical source: `notation.md §1`). FLOPs-specific reminder: the KV-attention compute term $F_{\text{attn,KV}}$ carries an additional $/SP$ factor on top of $D_{\text{kv}}$ when sequence parallelism is enabled.

To find **per-device FLOPs**, we apply sharding from each parallelism dimension and then multiply by the number of layers on the PP stage.

---

### Tensor sharding (projections, FFN GEMMs)

The Q/K/V/O projection FLOPs follow the attention divisor:

$$
F_{\text{proj}}^{\text{device}} = \frac{F_{\text{proj}}}{D_{\text{attn}}}
$$

The FFN GEMMs follow the expert divisor:

$$
F_{\text{ffn}}^{\text{device}} = \frac{F_{\text{ffn}}}{D_{\text{exp}}}
$$

Both are standard column- and row-parallel matmul shardings [MEGATRON]; the abstract divisors collapse to the familiar `/TP` (TP-attn projections), `/(TP·EP)` (orthogonal MoE FFN), or `/EP` (co-located MoE FFN) per the table above.

---

### KV-attention sharding

The per-token attention score / value FLOPs $F_{\text{attn,KV}} = 4SH$ are sharded by the same divisor that shards the KV cache itself. Under TP-attention each rank computes attention for its own heads ($D_{\text{kv}} = TP$); under DP-attention each rank computes attention for its own sequences ($D_{\text{kv}} = TP$ orthogonal, $\max(TP, EP)$ co-located). Sequence parallelism (SP) further shards the sequence dimension within the rank [MEGATRON3]:

$$
F_{\text{attn,KV}}^{\text{device}} =
\frac{4 S H}{D_{\text{kv}} \cdot SP}
$$

SP does **not** further reduce $F_{\text{proj}}$, $F_{\text{ffn}}$, or router FLOPs.

---

### Expert sharding (MoE only)

EP applies only to MoE layers. Router FLOPs operate on the full hidden state and are **not sharded** (every token routes through the full router on every device that holds the router weights). Expert FFN GEMMs are sharded by $D_{\text{exp}}$:

$$
F_{\text{expert}}^{\text{device}} =
\frac{k \, (6 H I_{\text{moe}})}{D_{\text{exp}}}
$$

Dense layers always have $EP = 1$ and $D_{\text{exp}} = TP$.

---

### PP (Pipeline Parallelism)

PP assigns whole layers to stages.  
Each device in a PP stage owns:

$$
\frac{L}{PP} \text{ layers}.
$$

Thus:

$$
F_{\text{token, device}}
\approx
\frac{L}{PP}
\left(
F_{\text{proj}}^{\text{device}} + F_{\text{attn,KV}}^{\text{device}} + F_{\text{ffn}}^{\text{device}}
\right)
$$

### Total Per-device FLOPs

Dropping the negligible $F_{\text{norm}}$ and also substituting everything yields the **final fully expanded expression** per-device FLOPs for a single decoded token:

$$
F_{\text{token,device}}
\;\approx\;
\frac{L}{PP}
\left(
\frac{4H^{2} + 4H H_{kv}}{D_{\text{attn}}}
\;+\;
\frac{6H I_{\text{eff}}}{D_{\text{exp}}}
\;+\;
\frac{4 S H}{D_{\text{kv}} \cdot SP}
\;+\;
2H N_{\text{eff}}
\right)
$$

For a **dense MLP model**: $I_{\text{eff}} = I_{\text{dense}},\quad N_{\text{eff}} = 0,\quad EP = 1$ (and $D_{\text{exp}} = TP$ by definition).

For a **MoE model**: $I_{\text{eff}} = k I_{\text{moe}},\quad N_{\text{eff}} = N_{\text{exp}},\quad EP \ge 1$.

### Mixed MoE/Dense Architectures

For mixed architectures, FLOPs are computed separately for dense and MoE layers (with dense FFN keeping $D_{\text{exp}} = TP$ regardless of layout):

$$
F_{\text{token,device}} =
F_{\text{dense,device}} + F_{\text{moe,device}}
$$

**Dense layer FLOPs** (per device, for all dense layers on this PP stage):

$$
F_{\text{dense,device}} =
\frac{L_{\text{dense}}}{PP}
\left(
\frac{4H^{2} + 4H H_{kv}}{D_{\text{attn}}}
\;+\;
\frac{6H I_{\text{dense}}}{TP}
\;+\;
\frac{4 S H}{D_{\text{kv}} \cdot SP}
\right)
$$

Note: Dense layers have no router FLOPs and use $EP = 1$.

**MoE layer FLOPs** (per device, for all MoE layers on this PP stage):

$$
F_{\text{moe,device}} =
\frac{L_{\text{moe}}}{PP}
\left(
\frac{4H^{2} + 4H H_{kv}}{D_{\text{attn}}}
\;+\;
\frac{6H k I_{\text{moe}}}{D_{\text{exp}}}
\;+\;
\frac{4 S H}{D_{\text{kv}} \cdot SP}
\;+\;
2H N_{\text{exp}}
\right)
$$

Note: The router term $2H N_{\text{exp}}$ is unsharded (applied to the full hidden state before expert selection). Prefill FLOPs (the $O(S^2)$ regime) are covered in [prefill.md](prefill.md).

### LM head FLOPs

Mirroring the LM head traffic term in §2.1, the $H \to V$ projection is sharded by $D_{\text{emb}}$ along the vocab dimension and **lives only on the last PP stage** (stage $PP{-}1$). It is not divided by $L$, $PP$, $EP$, or $SP$. The per-step compute on stage $PP{-}1$ is:

$$
F_{\text{LM,step,device}}(B) = \frac{2 \, B \, H \, V}{D_{\text{emb}}} \quad \text{(stage } PP{-}1 \text{ only)}
$$

This scales linearly with $B$ (one $H \to V$ projection per sequence) and fires once per step, not per layer. It is bookkept as a separate additive term and combined with $T_{\text{LM},\theta,\text{device}}$ into a one-shot LM-head roofline $t_{\text{LM}}$ in §6 / §7; on stages $0..PP{-}2$ this term is zero.

### Per-step (batched) FLOPs

A decode step processes $B$ sequences concurrently and produces 1 new token per sequence. Per-rank per-step compute scales with the number of tokens that rank actually processes — which depends on the attention mode because **DP-attention sequence-shards the batch across the TP group**, while TP-attention replicates the batch and head-shards the attention block. Per-rank per-step compute is therefore composed from two pieces:

- **Attention block** (Q/K/V/O projection + score / value compute). Under TP-attention each rank sees all $B$ tokens but does $1/D_{\text{attn}}$ of the per-token work (head-sharded); per-step attention compute scales as $B \cdot F_{\text{attn,token,device}}$. Under DP-attention each rank holds the full per-token attention weight set ($D_{\text{attn}} = 1$) but only processes $B / G_{TP}$ tokens (its sequence shard); per-step attention compute scales as $(B / G_{TP}) \cdot F_{\text{attn,token,device}}$. The two pictures collapse to the same per-rank flux at $D_{\text{attn}} = G_{TP} = TP$ for TP-attention, but diverge by a factor of $G_{TP}$ under DP-attention if the batch divisor is omitted.
- **FFN / MoE block** (dense FFN, MoE routed FFN, router gate). Per-token FFN work is independent of attention mode: each rank processes all $B$ tokens through its $D_{\text{exp}}$-sharded FFN / expert set, so per-step FFN compute scales as $B \cdot F_{\text{ffn,token,device}}$ regardless of TP-attn vs DP-attn.

Writing the per-rank per-step compute explicitly:

$$
F_{\text{step,device}}(B) \;=\; B_{\text{attn}}^{\text{rank}} \cdot F_{\text{attn,token,device}} \;+\; B \cdot F_{\text{ffn,token,device}}
$$

with the per-rank batch divisor for the attention block:

$$
B_{\text{attn}}^{\text{rank}} \;=\;
\begin{cases}
B / G_{TP} & \text{if attention\_mode} = \text{DP-attn} \\
B & \text{if attention\_mode} = \text{TP-attn}
\end{cases}
$$

**Score / value carries its own batch divisor.** The per-token score / value term $F_{\text{attn,KV}}^{\text{device}} = 4SH / (D_{\text{kv}} \cdot SP)$ already contains $D_{\text{kv}}$. Under DP-attn $D_{\text{kv}} = G_{TP}$ holds for all production layouts (orthogonal: $D_{\text{kv}} = TP = G_{TP}$; co-located: $D_{\text{kv}} = \max(TP, EP) = TP = G_{TP}$ under the structural $TP = EP$ co-located constraint), so the $B \cdot F_{\text{attn,KV}}^{\text{device}}$ form gives the correct per-rank result even without explicit batch-divisor bookkeeping — the $D_{\text{kv}}$ divisor already absorbs the batch shard. The bug surface is the *projection* term ($Q/K/V/O$), whose per-token formula divides by $D_{\text{attn}}$ (= 1 under DP-attn) rather than $G_{TP}$, and therefore needs the explicit $B / G_{TP}$ scaling above.

This is the per-step, per-device FLOP count consumed in the roofline (§4). All downstream HW latency formulas (§4–§6) carry the $(B)$ argument explicitly.

---

<div style="page-break-before: always;"></div>

# 4. Compute vs. Memory Bound (Roofline Model)

Sections 2 and 3 derived the **per-step memory traffic** $T_{\text{step,device}}^{\text{eff}}(B) = T_{\theta,\text{device}} + B \cdot T_{\text{KV,token}}$ and the **per-step FLOPs per device** $F_{\text{step,device}}(B) = B_{\text{attn}}^{\text{rank}} \cdot F_{\text{attn,token,device}} + B \cdot F_{\text{ffn,token,device}}$ (with $B_{\text{attn}}^{\text{rank}} = B / G_{TP}$ under DP-attention and $B$ under TP-attention; see §3.5 *Per-step (batched) FLOPs*). The compute roofline divides FLOPs by sustained device throughput $R_{\text{GPU}}$ (FLOPs/s):

$$
t_{\text{compute}}(B) = \frac{F_{\text{step,device}}(B)}{R_{\text{GPU}}} = \frac{B \cdot F_{\text{token,device}}}{R_{\text{GPU}}}
$$

The memory roofline opens up across the device's memory hierarchy. Modern accelerators expose an ordered list of memory tiers $i = 0, 1, \ldots, n-1$ (fastest first): single-tier HBM on H100 / B200, two-tier SRAM + LPDDR5 on d-Matrix Corsair, single-tier on-die SRAM on Groq LPU. A **placement policy** $\pi$ assigns weight and KV bytes to specific tiers, splitting $T_{\theta,\text{device}}$ into per-tier shares $T_{\theta,0}, \ldots, T_{\theta,n-1}$ and $T_{\text{KV},\text{device}}$ into $T_{\text{KV},0}, \ldots, T_{\text{KV},n-1}$. Each tier carries its own effective bandwidth $BW_{\text{eff},i} = BW_i \cdot \eta_{\beta,i}$ (peak rate deflated by a sustained-throughput contention factor $\eta_{\beta,i} \in (0, 1]$, see `sram.md §1.2`) and a first-byte latency $\alpha_i$ paid once per non-empty placement on that tier (the standard $\alpha$–$\beta$ form, treating each tier as a single transaction per step). The per-step memory time across all tiers is:

$$
t_{\text{mem}}(B) = \sum_{i=0}^{n-1} \left[\, \alpha_i \cdot \mathbb{1}\!\bigl(\mathrm{bytes}_i > 0\bigr) \;+\; \frac{T_{\theta,i} + B \cdot T_{\text{KV},i}}{BW_{\text{eff},i}} \,\right]
$$

The $\alpha_i$ floor is kept here so the device roofline matches the standard $\alpha$–$\beta$ form rather than a $\beta$-only shorthand. For the on-die / co-packaged tiers in scope here — SRAM, HBM, LPDDR5 — $\alpha_i$ falls in the 1 ns – 200 ns range and contributes well under 0.1% of $t_{\text{mem}}$ at decode-step granularity, so dropping it for simplicity is a safe approximation in subsequent derivations. The full $\alpha$–$\beta$ form is also what gets reinstated in small-read regimes (paged-attention block fetch, flash-style spill — see `sram.md §2.1`). Tier definitions, placement policies (greedy fastest-first, operator-pinned), and worked numerical examples for d-Matrix Corsair / B200 / Groq LPU live in `sram.md`; this document treats $T_{\theta,i}$ and $T_{\text{KV},i}$ as given inputs.

**Single-tier reduction.** When $n = 1$ — single-tier devices like H100 / B200 / Groq LPU, or any system modeled before opening up the tier list — the sum collapses to one term:

$$
t_{\text{mem}}(B) = \alpha_0 \;+\; \frac{T_{\theta,\text{device}} + B \cdot T_{\text{KV,token}}}{BW_{\text{eff},0}}
\qquad (n = 1)
$$

with $BW_{\text{eff},0} \equiv BW_{\text{mem}}$. The Operational Intensity and $B^\star$ analyses below are written against this single-tier shorthand and additionally drop $\alpha_0$ (negligible for on-die tiers) for compactness; the multi-tier crossover (weights and KV pinned to different tiers — d-Matrix Capacity Mode being the canonical example) is derived in `sram.md §2.2`.

$t_{\text{compute}}(B)$ is the time assuming unlimited memory bandwidth — linear in $B$ since every sequence contributes its own per-token FLOPs. $t_{\text{mem}}(B)$ is the time assuming compute is free — within each tier, weights amortize once per step (the $T_{\theta,i}$ terms are $B$-independent) while KV reads scale with $B$ (each sequence reads its own history). For long-context LLMs the $B \cdot T_{\text{KV},i}$ contribution summed across tiers is often dominant.

The per-step **local latency** on this PP stage is the roofline of the two:

$$
t_{\text{local}}(B) =
\max\bigl( t_{\text{compute}}(B),\; t_{\text{mem}}(B) \bigr)
$$

The asymmetric $B$-scaling of the two arms (compute linear in $B$; memory weights flat per tier, KV linear per tier) means the regime can flip with $B$ — characterized by the operational-intensity analysis below.

---

### Operational Intensity (Ops:Byte)

The **operational intensity** for decoding on this device is [ROOFLINE]:

$$
\text{OI}(B) =
\frac{F_{\text{step,device}}(B)}{T_{\text{step,device}}^{\text{eff}}(B)} =
\frac{B \cdot F_{\text{token,device}}}{T_{\theta,\text{device}} + B \cdot T_{\text{KV,token}}}
\quad \text{(FLOPs per byte)}
$$

For the **single-token, single-sequence** baseline ($B = 1$), the formula collapses to $\text{OI}(1) = F_{\text{token,device}} / (T_{\theta,\text{device}} + T_{\text{KV,token}})$, which establishes the fundamental memory-bound character of decode. As $B$ grows, $\text{OI}(B)$ rises (weight reads amortize across more sequences); the crossover-batch analysis is in the dedicated subsection below.

High-level interpretation:

- **High OI** → more FLOPs per byte → *compute-bound*
- **Low OI** → fewer FLOPs per byte → *memory-bound*

This rating is compared to the device’s memory-to-compute ratio (the ridge point):

$$
\frac{R_{\text{GPU}}}{BW_{\text{mem}}}
$$

If $\text{OI}(B) > R_{\text{GPU}} / BW_{\text{mem}}$ the step is **compute-bound** (and $t_{\text{local}}(B) = t_{\text{compute}}(B)$); otherwise it is **memory-bound** (and $t_{\text{local}}(B) = t_{\text{mem}}(B)$).

---

### Dominant-Term Approximation

In practice, the OI is often approximated using only the largest FLOP and traffic terms. With $B$ sequences per step and the per-component sharding factors from `notation.md §1`:

- Per-step FLOPs dominated by:
  $$
  B \cdot \max\!\left( \frac{2H^2}{D_{\text{attn}}},\; \frac{6H I_{\text{eff}}}{D_{\text{exp}}},\; \frac{4 S H}{D_{\text{kv}} \cdot SP} \right)
  $$

- Per-step memory traffic dominated by the KV term (each sequence reads its own KV history):
  $$
  B \cdot \frac{2 S H_{kv}}{D_{\text{kv}} \cdot SP}\, b
  $$

Thus for long-context decoding (attention FLOPs dominate the FLOP max):

$$
\text{OI}(B) \approx
\frac{B \cdot 4 S H / (D_{\text{kv}} \cdot SP)}{B \cdot 2 S H_{kv} / (D_{\text{kv}} \cdot SP)\, b}
= \frac{2H}{H_{kv}\, b}
$$

Both $B$ and $(D_{\text{kv}} \cdot SP)$ cancel — the OI is shape- and batch-independent in this regime. (The cancellation makes intuitive sense: every sequence reads its own KV and computes its own attention, so the per-sequence FLOP-to-byte ratio is what matters; sharding splits both terms identically.)

For MHA ($H_{kv} = H$), this reduces to $2/b$. For GQA models, $H/H_{kv} = n_q/n_{kv}$ amplifies the OI — e.g., for $n_q/n_{kv} = 8$ (LLaMA-3 70B) the OI is $16/b$. Even so, this is far below typical ridge points (~300 FLOPs/byte on H100), so long-context decode remains **memory-bound** in practice.

The short-context limit collapses to the same $2/b$ for a different reason: when weights dominate, FFN FLOPs $6HI_\text{eff}$ over FFN traffic $3HI_\text{eff} \cdot b$ also gives $2/b$. So the path-independent answer is **OI ≈ 2/b at $B=1$**, with $b$ being the only free knob.

---

### Compute-Bound Crossover ($B^\star$)

The full $\text{OI}(B) = B \cdot F_{\text{token,device}} / (T_{\theta,\text{device}} + B \cdot T_{\text{KV,token}})$ has two limiting regimes:

**Memory-bound limit** ($B$ small, weight traffic dominates the denominator):

$$
\lim_{B \to 0} \text{OI}(B) =
\frac{B \cdot F_{\text{token,device}}}{T_{\theta,\text{device}}}
\;\to\; 0
$$

At small $B$, the model is **weight-traffic-limited**. The $B = 1$ case recovers the single-token OI established above.

**Compute-bound limit** ($B$ large, KV traffic dominates the denominator):

$$
\lim_{B \to \infty} \text{OI}(B) =
\frac{F_{\text{token,device}}}{T_{\text{KV,token}}}
$$

At large $B$, KV cache reads dominate and the intensity saturates at this asymptotic ceiling.

**Ridge-point crossover.** The **crossover batch size** $B^\star$ is the point at which the roofline transitions from memory-bound to compute-bound. Setting $\text{OI}(B^\star) = R_{\text{GPU}} / BW_{\text{mem}}$ (the ridge point per [ROOFLINE]) and solving:

$$
B^\star =
\frac{T_{\theta,\text{device}} \cdot R_{\text{GPU}}}
     {F_{\text{token,device}} \cdot BW_{\text{mem}} - T_{\text{KV,token}} \cdot R_{\text{GPU}}}
$$

**Existence condition.** $B^\star$ is finite and positive iff the denominator is positive. Rearranging into ridge-point form:

$$
B^\star < \infty
\quad\Longleftrightarrow\quad
\frac{F_{\text{token,device}}}{T_{\text{KV,token}}} \;>\; \frac{R_{\text{GPU}}}{BW_{\text{mem}}} = R_{\text{ridge}}
$$

i.e., the **arithmetic intensity of KV traffic alone** (per-token FLOPs over per-sequence KV bytes) must exceed the device ridge point. Intuition: $\text{OI}(B)$ asymptotes to $F_{\text{token,device}} / T_{\text{KV,token}}$ as $B \to \infty$; if this asymptotic ceiling itself sits below $R_{\text{ridge}}$, the roofline is never crossed regardless of batch size.

**No-crossover regime.** When the inequality is violated — typical of very long contexts on small models, where $T_{\text{KV,token}}$ grows linearly in $S$ while $F_{\text{token,device}}$ stays fixed — decode remains memory-bound at every $B$, and $B^\star \to \infty$. In this regime batching still amortizes *weight* traffic ($T_{\theta,\text{device}} / B$ per token), which continues to reduce per-sequence TPOT, but it **cannot** push the step into the compute-bound zone; adding more sequences only adds linear KV bandwidth pressure until HBM saturates.

**Weight-dominated approximation.** When $T_{\text{KV,token}}$ is small relative to $T_{\theta,\text{device}} / B^\star$ (short-context decode where weight traffic dominates), $B^\star$ simplifies to:

$$
B^\star \;\approx\;
\frac{T_{\theta,\text{device}}}{F_{\text{token,device}}}
\cdot
\frac{R_{\text{GPU}}}{BW_{\text{mem}}}
\qquad (\text{weight-dominated regime})
$$

Intuitive interpretation: $T_{\theta,\text{device}} / F_{\text{token,device}}$ is the inverse single-token OI (bytes per FLOP), and $R_{\text{GPU}} / BW_{\text{mem}}$ is the ridge point (FLOPs per byte). Their product is the batch size at which weight reuse tips the balance from memory-bound to compute-bound.

---

<div style="page-break-before: always;"></div>

# 5. Communication Time During Decoding

Section 4 defined the **local per-step latency** on each device as the roofline of compute and the multi-tier memory sum:

$$
t_{\text{local}}(B) = \max\!\bigl(t_{\text{compute}}(B),\; t_{\text{mem}}(B)\bigr),
\qquad
t_{\text{mem}}(B) = \sum_{i=0}^{n-1} \frac{T_{\theta,i} + B \cdot T_{\text{KV},i}}{BW_{\text{eff},i}}
$$

(Single-tier devices collapse the sum to one term, recovering $t_{\text{mem}}(B) = (T_{\theta,\text{device}} + B \cdot T_{\text{KV,token}}) / BW_{\text{mem}}$ — see §4.) We now incorporate the **inter-device communication time** that arises during decoding under distributed parallelism. The four within-replica parallelism dimensions $PP$, $EP$, $TP$, $SP$ each contribute their own per-step collective; $DP$ has none (replicas are independent). Their *physical mapping* to GPUs follows the layout choice (`notation.md §1`): the orthogonal layout maps each axis to a disjoint GPU set within a replica (the canonical Megatron-LM nesting `DP → PP → EP → TP → SP`), while TP+EP co-location overlays $TP$ and $EP$ on the same physical GPU set (DSv3 / SGLang / NVIDIA Dynamo production decode). Collective group sizes $G_{TP}$, $G_{EP}$ from `notation.md §1` carry through identically across both layouts; the per-rank message sizes use the per-component sharding factors $D_{\text{kv}}$, $D_{\text{attn}}$ to absorb layout-dependent shape changes.

All communication costs follow the standard $\alpha$–$\beta$ latency model [ALPHA-BETA]:

$$
t_{\text{comm}} = \alpha + \frac{\text{message size}}{B_{\text{eff}}}
$$

where $\alpha$ is the collective or hop latency, and $B_{\text{eff}}$ is the sustained bandwidth of the communication path.

The parameters $\alpha$ and $B_{\text{eff}}$ in this model are not abstract: they are **topology-dependent physical properties** of the underlying interconnect. Different parallelism domains—TP, EP, SP, and PP—may be mapped to **different network fabrics** or different portions of the same physical topology (e.g., NVSwitch star within a node, 2D/3D torus across nodes, or hybrid switch-plus-fabric designs). Consequently, each collective type sees its own communication characteristics, with potentially different latency constants and effective bandwidths. To keep the analysis general, we denote these as $\alpha_{XP}$ and $B_{\text{eff},XP}$ for TP, EP, SP, and PP respectively. Their actual numerical values depend on the system’s physical layout, routing scheme, and bisection bandwidth properties (e.g., constant-hop NVSwitch vs. hop-scaling torus fabrics). The following sections therefore use $\alpha_{XP}$ and $B_{\text{eff},XP}$ as **collective-specific, topology-aware** parameters, to be instantiated according to the actual deployment mapping.

**Delegation to the `collectives/` explainer subseries.** The shipped collective primitives (ring AR, double binary tree AR, ring AG / RS, pairwise A2A on star; dim-decomposed ring and bisection-bound A2A on torus; hierarchical RS → sub-AR → AG; in-network reduction via NVLS / Quantum SHARP / Tomahawk Ultra) are cost-modeled in `collectives/01_collective_algorithms.md` (per-algorithm) and `collectives/02_topology_mapping.md` (star / torus / mesh), with hierarchical composition in `collectives/03_hierarchical_topologies.md`, in-network primitives in `collectives/04_in_network_collectives.md`, and contention coefficients $(\eta_\alpha, \eta_\beta)$ in `collectives/05_contention_and_congestion.md`; the cheatsheet at `collectives/00_summary.md` indexes all of these. This section instantiates those primitives with the decode-scale per-rank message sizes defined below; derivations and the $(\alpha_\text{sum}, BW_\text{min})$ tier-chain accumulation live there. The $\alpha_{XP}$ and $BW_{XP}$ values used below are the fabric-chain span quantities from `notation.md §7`.

### Message sizes and their shard structure

To remain consistent with the compute and memory models, we strictly define the payload size for each collective type. Note the distinction between *storage size* (sharded) and *communication payload* (often full-width). Each shape below is given per step for batch size $B$; the per-token shape is the per-step shape divided by $B$. KV-gather (SP) scales with the number of sequences whose KV must be gathered, i.e., also $\propto B$.

- **PP (Pipeline Parallel):**
  Per-rank payload $B \cdot H / D_{\text{kv}}$ per step.
  *Rationale:* High-performance PP (e.g., Megatron-LM) preserves cross-section rank alignment, so each device only forwards its local share of the activation to the corresponding rank in the next stage. Under TP-attention this is the head-shard ($H/TP$ per token); under DP-attention it is the sequence shard (1/$D_{\text{kv}}$ of the $B$ sequences, each carrying full $H$). Both reduce to a per-rank PP payload of $B \cdot H / D_{\text{kv}}$ per step.

- **EP (Expert Parallel):**
  Per-rank logical payload $B \cdot k \cdot H$ per step (with the standard $(G_{EP} - 1)/G_{EP}$ factor in the all-to-all cost form distributing this across peers).
  *Rationale:* MoE routing sends token activations to experts. While the traffic is bidirectional (Dispatch + Combine), we model this by applying a factor of 2 to the *collective steps* in Section 5.2 rather than doubling the base message size here.

- **TP (Tensor Parallel):**
  Per-rank logical payload $B \cdot H$ per step (under both AR and AG primitives — AG carries the same total bytes, the per-rank shard $B \cdot H / G_{TP}$ is what each rank initially holds before the AG fans it out).
  *Rationale:* Row-Parallel matrix multiplication produces a vector of **partial sums** of full width $H$ (TP-attn AR), or each rank already holds the full attention output for its sequence subset and the AG redistributes it to the TP-sharded form for the next FFN block (DP-attn AG).

- **SP (Sequence Parallel):**
  Per-rank payload $B \cdot \frac{S}{SP} \cdot \frac{2H_{kv}}{D_{\text{kv}}}$ per step (equivalently $B \cdot S \cdot 2H_{kv} / (D_{\text{kv}} \cdot SP)$).
  *Rationale:* Ring Attention streams the distributed KV blocks around the ring. The per-rank KV shard composes the $D_{\text{kv}}$ partition (head shard under TP-attn or sequence shard under DP-attn) with the SP sequence partition. With $B$ concurrent sequences per step, each sequence's KV shard is streamed independently.

---

## 5.1 Pipeline Parallel (PP) Hop

Pipeline Parallelism (PP) forwards activations from one pipeline stage to the next [MEGATRON3]. Because TP is nested inside PP, high-performance implementations (e.g., Megatron-LM PP, DeepSpeed PP, NVIDIA NeMo, and FasterTransformer) preserve the **TP rank alignment** across all PP stages. That is, TP rank $i$ in stage $s$ corresponds directly to TP rank $i$ in stage $s{+}1$.  

This alignment has an important consequence:  
**each device only needs to forward its local share of the hidden state to the rank-aligned device on the next stage**, not a full $H$-dimensional vector. The full activation is conceptually transferred across the PP boundary, but it is split naturally across $D_{\text{kv}}$ separate device-to-device links (where $D_{\text{kv}}$ is the head- or sequence-shard divisor of `notation.md §1`; under TP-attention this is the head shard, under DP-attention the sequence shard).

Thus the PP hop behaves as a **single, shard-sized point-to-point transfer** with per-rank message size $\approx H / D_{\text{kv}}$ per token. For a decode step processing $B$ sequences concurrently, the per-step payload scales linearly:

$$
t_{PP}(B) =
\alpha_{PP}
+
\frac{B \cdot (H / D_{\text{kv}})\, b}{BW_{\text{PP}}}
$$

The α-term (single point-to-point latency) is paid once per hop regardless of payload; only the β-term scales with $B$. This shard-preserving PP design avoids the extra TP collectives that would be required if stages exchanged full activations and then re-sharded them. Maintaining cross-section rank consistency across stages therefore yields a significantly faster pipeline, and is the standard strategy in modern LLM training and inference systems.

**Tier-aware PP cost (nested-layout convention).** $\alpha_{PP}$ and $BW_{PP}$ above are *not* uniformly tier-0 fabric values. They are the latency and bandwidth of the **specific tier** the PP boundary physically crosses, which depends on where PP sits in the nested layout `DP → PP → EP → TP → SP` (innermost = highest-bandwidth tier). The per-axis tier assignment walks the fabric chain inner→outer, allocating each axis to the smallest tier whose cumulative reach holds the cumulative product of inner axes × this axis. For example, on d-Matrix squadrack (3-tier chain: 16 × 4 × 8):

- $PP=2, TP=8$: cumulative $8 \cdot 2 = 16 \le$ tier-0 cap → **PP at tier 0** (pair-of-cards mesh, $\alpha=0.115$ μs, BW=64 GB/s).
- $PP=8, TP=8$: cumulative $8 \cdot 8 = 64 \le$ tier-1 cap → **PP at tier 1** (PCIe, $\alpha=0.65$ μs).
- $PP=32, TP=8$: cumulative $8 \cdot 32 = 256 > 64$ → **PP at tier 2** (Ethernet, $\alpha=2.0$ μs, BW=50 GB/s).

On single-tier systems (e.g., NVL72), every axis collapses to tier 0 and the legacy tier-0 PP pricing is recovered exactly. This is a worst-case-tier model — within a single PP cost call the *outermost* tier the boundary could possibly cross is used. A finer per-hop blend (some boundaries within tier 0, some across tier 1) is left as a future refinement; the worst-case form matches the conservative engineering view that "PP runs across servers" for sweeps where it does.

---

## 5.2 Expert Parallel (EP) All-to-All (MoE Dispatch and Combine)

MoE layers require exchanging token activations across the expert-parallel (EP) dimension via all-to-all routing [DEEPSPEED-MOE]. EP communication follows a **bidirectional dispatch-and-combine pattern**: token activations are routed from the source rank to the rank holding the selected expert (top-$k$), and the expert's output is then sent back to the source rank to be added to the residual stream. We model these as **two distinct A2A collectives per MoE layer** ($n_{EP} = 2$ in §5.5), each costing one single-direction A2A — this aligns the cost-model bookkeeping with the kernel-launch counter (§7.1), since dispatch and combine are also two separate NCCL API calls.

Let $k$ denote the number of active experts per token. Each direction carries a $kHb$ byte per-rank per-token payload; for a decode step of $B$ sequences the per-step payload is $B \cdot kHb$. The collective group size is $G_{EP}$ from `notation.md §1` (equal to $EP$ across all four configurations). The shipped A2A primitive is pairwise direct-send (NCCL on star; bisection-bound pairwise on torus). Bruck / log-hop A2A does **not** ship and does not appear in the cost — see `collectives/01_collective_algorithms.md §7` for the primitive derivations. On a star topology, the per-direction A2A cost is:

$$
t_{EP}(B) \;=\; (G_{EP} - 1)\,\alpha_{EP} \;+\; \frac{G_{EP} - 1}{G_{EP}} \cdot \frac{B \cdot k H \, b}{BW_{\text{EP}}}
$$

§5.5's per-layer accumulator multiplies this by $n_{EP} = 2$ to recover the full Dispatch + Combine cost. For torus EP fabrics, substitute the bisection-bound form of `collectives/02_topology_mapping.md §3` with $M = B \cdot kHb$. For dense models ($EP = G_{EP} = 1$), $t_{EP}(B) = 0$.

> **Note on expectation under uniform routing.** Unlike MoE weight traffic (§2.1), where bytes-read-per-step is the expected number of *touched experts* per rank and is genuinely sub-linear in $B$ at small $B$, the per-rank A2A send / receive volume above is the deterministic $\frac{G_{EP}-1}{G_{EP}} \cdot B k H b$. Each token-expert assignment generates its own dispatch payload regardless of routing collisions — two tokens hitting the same expert send two separate hidden vectors — so the β-side scales exactly linearly in $B$ with no expectation correction. The same independence applies to TP, SP, and PP collectives in §5.3–§5.5: their payloads are deterministic by construction (every rank participates in every collective every step), and the expectation question does not arise. The only place expectation would tighten the EP A2A model is the *number of active destination links* at extremely small $B \cdot k < G_{EP}$ — when some destination ranks receive zero tokens that step. We omit this correction: the regime is rare in production decode (running EP this wide at $B$ this small has no throughput justification), the α-cost dominates the budget at small $B$ anyway, and the resulting overestimate is bounded by $(G_{EP} - \mathbb{E}[\text{active dests}]) \cdot \alpha_{EP}$ — typically well under 1% of $t_{\text{step}}$.

> **Co-located note.** Under the co-located layout the $G_{EP}$ ranks share physical GPUs with the $G_{TP}$ ranks. The all-to-all wire payload per call is unchanged (still $B \cdot kHb$ per direction across $G_{EP}$ ranks), but the EP A2A and any TP collective on the same physical GPUs cannot run concurrently — they share the NVLink bandwidth and serialize at the fabric layer. This shows up in the §5.5 per-stage accumulator naturally because TP and EP collectives are already accumulated sequentially.

### Two A2A data-flow patterns under DP-attention

Under DP-attention, the activations entering the MoE block come from $G_{TP}$ separately-batched attention replicas (each rank holds $B / G_{TP}$ tokens of attention output, since the batch is split across the $G_{TP}$ DP-attention ranks while attention parameters are replicated on each rank). Two production patterns exist for how these per-rank tokens are routed through the MoE all-to-all (A2A); they differ by a factor of $G_{TP}$ in the per-rank dispatch payload, which dominates EP cost at large $B$:

- **Gather-then-dispatch.** Insert a TP all-gather before the MoE block to bring the full $B$ tokens onto every rank, then dispatch from full $B$. Per-rank dispatch payload $M_{\text{disp}} = B \cdot k H b$. Per-rank combine payload symmetrically $B \cdot k H b$. Adds one TP all-gather per MoE layer at $B \cdot Hb$ payload (the post-attention AG already counted in §5.3 — under this pattern the AG serves both the FFN and the MoE input). This is the conservative pattern shipped by general-purpose MoE backends that assume the residual stream is replicated across TP ranks before any sparse-block compute. Total per-rank wire bytes per MoE layer ≈ $B \cdot Hb \cdot (1 + 2k)$.

- **Scatter-direct.** Skip the AG; the dispatch A2A operates directly on the per-rank $B / G_{TP}$ attention-sharded tokens, routing each token to its $k$ destination experts wherever they live across the $G_{EP}$ ranks. Per-rank dispatch payload $M_{\text{disp}} = (B / G_{TP}) \cdot k H b$ — a factor of $G_{TP}$ smaller. The combine returns the expert outputs back to the originating attention rank, payload symmetrically $(B / G_{TP}) \cdot k H b$. No AG fires before MoE; the residual stays sharded across attention ranks throughout the layer. This is the pattern shipped by DeepEP [DEEPEP] and the SGLang DeepEP integration [SGLANG-DPATTN], which expose dispatch APIs parameterized by per-rank token counts rather than a globally-gathered batch. Total per-rank wire bytes per MoE layer ≈ $2 (B / G_{TP}) \cdot k H b$ — i.e. a $G_{TP} \cdot (1 + 2k)/(2k)$ reduction over gather-then-dispatch (~$G_{TP}$× for typical $k$).

The cost-model formulas in this section are written for the **gather-then-dispatch** pattern (the conservative ceiling) so that $M_{\text{disp}} = B \cdot k H b$. To model scatter-direct, substitute $B \to B / G_{TP}$ in the EP A2A formulas of this section and drop the per-MoE-layer TP all-gather contribution from §5.5.

**Trade-offs.** Scatter-direct is unambiguously smaller in bytes-on-wire and is the production-favored pattern for very-large-$B$ DSv3-class deployments where MoE comm would otherwise dominate. The cost is in ergonomics and fabric assumptions: it requires the MoE backend to be written with sharded-token awareness (variable per-rank token counts after routing, no implicit "I have the full residual" assumption), pairs naturally with hook-based dispatch / combine kernels that overlap the A2A with expert compute on a separate stream rather than blocking, and the high-throughput kernels assume bisection-bandwidth-class fabrics (intra-NVLink-island today; cross-node paths use Remote Direct Memory Access (RDMA) transports with their own latency floor). Gather-then-dispatch remains the right model when these assumptions don't hold — small $G_{TP}$ where the AG is cheap, MoE backends without sharded-token support, or fabrics where the AG ride-along to the FFN amortizes the cost.

---

## 5.3 Tensor Parallel (TP) Communication

TP groups compute each layer in parallel across $G_{TP}$ devices (group size from `notation.md §1`; $G_{TP} = TP$ across all four configurations) using column- and row-parallel linear layers [MEGATRON]. Each layer fires two TP collectives: one in the **attention** block (after the output projection) and one in the **MLP / MoE expert** block (after the down projection). The collective in the attention block depends on the attention parallelism mode:

- **TP-attention (default):** all-reduce (AR) — sums partial outputs across the $G_{TP}$ ranks that each hold a shard of the heads.
- **DP-attention:** all-gather (AG) — each rank already holds the full attention output for its assigned sequence subset; the AG gathers all sequences' hidden states so the downstream TP-sharded FFN can run [DSV3, SGLANG-DPATTN].

The MLP / MoE block always fires an AR (it is TP-sharded under both attention modes).

**Critical Note on Message Size:** unlike PP (which sends a shard), the TP collectives operate on the **full hidden state vector** ($H$). Each device owns only a shard of the weights, but the post-row-parallel output (or pre-FFN gathered hidden state) is a vector of size $H$ that must be moved globally; the payload is $Hb$ bytes per token, not $(H/G_{TP})b$. For a decode step of $B$ sequences the per-step payload is $B \cdot Hb$.

NCCL ships two algorithms on a star fabric for both AR and AG — ring (large-$M$) and double binary tree (DBT, small-$M$). Selection is a manual tuner knob (`tuner.ar_algorithm`, default `"ring"`; see `collectives/02_topology_mapping.md §2`). Both are pipelined and bandwidth-optimal; only the $n_\alpha$ coefficient differs. For the decode payload $M = B \cdot Hb$, the per-call costs are:

$$
t_{TP,\text{AR}}^{\text{ring}}(B) \;=\; 2(G_{TP} - 1)\,\alpha_{TP} \;+\; 2 \cdot \frac{G_{TP} - 1}{G_{TP}} \cdot \frac{B \cdot H b}{BW_{\text{TP}}}
$$

$$
t_{TP,\text{AG}}^{\text{ring}}(B) \;=\; (G_{TP} - 1)\,\alpha_{TP} \;+\; \frac{G_{TP} - 1}{G_{TP}} \cdot \frac{B \cdot H b}{BW_{\text{TP}}}
$$

The AG cost is exactly half the AR cost (one pass through the ring instead of the AR's two: reduce-scatter then all-gather). The DBT variants substitute $2\lceil \log_2 G_{TP}\rceil \cdot \alpha_{TP}$ (AR) or $\lceil \log_2 G_{TP}\rceil \cdot \alpha_{TP}$ (AG) for the $\alpha$-side coefficient; the $\beta$-side terms are unchanged. For torus TP fabrics (dim-decomposed ring, shipped on TPU / Trainium), substitute the torus AR / AG forms of `collectives/02_topology_mapping.md §3` with $M = B \cdot Hb$. Derivation and the ring-vs-DBT empirical crossover behavior are in `collectives/02_topology_mapping.md §2` (cost) and explainer `02 §2`.

The per-layer TP cost depends on attention mode and is assembled into the per-stage budget in §5.5.

---

## 5.4 Sequence Parallel (SP) Communication

Sequence Parallelism (SP) in inference typically refers to **Ring Attention** [RING-ATTN]. The KV cache is partitioned along the sequence dimension $S$; to compute attention for a new token the Query ($Q$) stays local and KV blocks rotate around the ring so that the local $Q$ attends to the full history. This is a **pass-KV** ring variant — the standard choice for KV-cache-dominated inference where KV is large relative to $Q$. (A pass-Q variant exists for training, where $Q$ is full-sequence; see [HUANG-CP-2024].)

The ring operation is effectively an **All-Gather** (streaming the distributed KV cache to every rank), not an All-Reduce. DeepSpeed-Ulysses [DEEPSPEED-ULYSSES] is an alternative SP approach using all-to-all instead of ring; unlike ring, it is bounded by the number of attention heads rather than the number of devices. Tree-based SP variants are theoretically possible but no production implementation ships them — KV shards are large and must be processed in sequence order. For modeling purposes, we assume **ring-style, pass-KV SP communication**, costed via the ring AG primitive of `collectives/01_collective_algorithms.md §6`.

### SP Ring Communication Latency

Each active sequence streams its own KV shard around the ring. With $B$ concurrent sequences per step, the per-rank payload is $M_\text{SP}(B) = B \cdot (S / SP) \cdot (2 H_{kv} / D_{\text{kv}}) \cdot b$ — the KV shard composes the $D_{\text{kv}}$ partition (head shard under TP-attention, sequence shard under DP-attention) with the SP sequence partition. Substituting into the star ring AG cost of `collectives/01_collective_algorithms.md §6`:

$$
t_{SP}(B) \;=\; (SP - 1)\,\alpha_{SP} \;+\; (SP - 1) \cdot \frac{B \cdot (S / SP) \cdot (2 H_{kv} / D_{\text{kv}}) \cdot b}{BW_{\text{SP}}}
$$

The per-step message size composes the $D_{\text{kv}}$ partition and the SP sequence partition, with $B$ multiplying because each sequence in the batch independently gathers its own KV shard. For torus SP fabrics, use the torus AG form of `collectives/02_topology_mapping.md §3` with the same $M_\text{SP}(B)$.

**Decode overlap note:** in single-token decode, per-token compute time is small, so communication overlap with compute ($\rho_{\text{comm}}$) is unlikely to be significant for SP. The unified $\rho_{\text{comm}}$ in §6.2 applies to all collective traffic; if SP dominates the comm budget on a given config, calibrate $\rho_{\text{comm}}$ down accordingly rather than zeroing it per-axis (the cost model has no per-axis $\rho_{\text{comm}}$ knob).

---

## 5.5 Total Communication Time Per Step on a PP Stage

Sections 5.1–5.4 provide **per-step, per-layer** communication costs for each parallelism axis (TP, EP, SP), and a **per-step, per-hop** cost for PP. Each carries the $(B)$ argument explicitly: α-side stays constant in $B$ (one collective per layer per stage regardless of payload), β-side scales linearly. We now combine them into the total per-step communication time on a given pipeline-parallel (PP) stage.

### Per-layer vs. per-stage normalization

A Transformer layer contains exactly one Attention block and one MLP/MoE block. Each block triggers a fixed number of communication collectives, and within each layer, TP, EP, and SP collectives are strictly ordered:

- **Attention block**
  - 1 TP collective at the attention → FFN transition. The primitive depends on attention mode: **all-reduce (AR)** under TP-attention (default), **all-gather (AG)** under DP-attention. Per-call cost is $t_{TP,\text{AR}}(B)$ or $t_{TP,\text{AG}}(B)$ from §5.3.
  - 1 SP collective (if SP is enabled).

- **MLP / MoE block**
  - 1 TP all-reduce at the FFN / expert output projection (always AR; not affected by attention mode).
  - 2 EP all-to-all calls (Dispatch + Combine; MoE layers only).

This gives the per-layer **NCCL API call counts**: $n_{TP} = 2$ (one in attention, one in FFN/expert), $n_{EP} = 2$ (Dispatch + Combine on MoE layers, $0$ on dense layers), $n_{SP} = 1$ (when SP is enabled, $0$ otherwise). These counts are layout- and mode-independent — they describe how many distinct collective calls fire per layer, not how expensive each call is. The per-call cost is what splits by mode (AR vs AG inside $t_{TP}^{\text{attn}}$). Kernel-launch accounting in §7.1 reuses these $n_*$ counts.

These collectives must complete before the token can advance to the next layer. Since a PP stage contains $L/PP$ *sequential* layers and TP, EP, and SP operations within each layer depend on one another (e.g., TP → SP in attention and EP → TP in MoE), they are **strictly sequential** and do not overlap. The per-layer TP cost is the sum of one attention TP collective and one FFN TP all-reduce:

$$
t_{TP}^{\text{layer}}(B) \;=\; t_{TP}^{\text{attn}}(B) \;+\; t_{TP,\text{AR}}(B), \qquad
t_{TP}^{\text{attn}}(B) \;=\; \begin{cases} t_{TP,\text{AR}}(B) & \text{TP-attention} \\ t_{TP,\text{AG}}(B) & \text{DP-attention} \end{cases}
$$

Switching from TP-attn to DP-attn replaces the attention AR with an AG, saving $t_{TP,\text{AR}}(B) - t_{TP,\text{AG}}(B) = (G_{TP} - 1)\alpha_{TP} + (G_{TP} - 1)/G_{TP} \cdot B \cdot Hb / BW_{TP}$ per layer (one AG's worth of $\alpha$ + bandwidth, since AG is half the cost of AR).

### Adding PP hop cost

The PP hop is different: it is a **per-step, per-hop** cost rather than a per-layer cost. A step's microbatch is forwarded once from PP stage $s$ to stage $s{+}1$, with latency $t_{PP}(B)$ as defined in Section 5.1.

The total per-step communication time on this stage is:

$$
t_{\text{comm}}(B) =
\frac{L}{PP}
\bigl(
t_{TP}^{\text{layer}}(B)
+
n_{SP}\, t_{SP}(B)
\bigr)
+
\frac{L_{\text{moe}}}{PP}
\bigl(
n_{EP}\, t_{EP}(B)
\bigr)
+
t_{PP}(B)
$$

where $n_{EP} = 2$ (dispatch + combine), $n_{SP} = 1$ (when SP is enabled, 0 otherwise), and $t_{TP}^{\text{layer}}(B)$ is mode-dependent as above.

### Interpretation

- The first term accumulates **TP and SP collectives** required by all $L/PP$ layers on this PP stage (both dense and MoE layers have attention blocks requiring these collectives). The TP collective's primitive split (AR + AR under TP-attn, AG + AR under DP-attn) is encoded inside $t_{TP}^{\text{layer}}(B)$.
- The second term accumulates **EP collectives** required only by the $L_{\text{moe}}/PP$ MoE layers on this PP stage. Dense layers do not require EP communication.
- The third term accounts for the **one PP hop** that forwards the step's microbatch to the next stage.
- This combined expression represents the **total communication work** per step for the stage. Whether this communication becomes the latency bottleneck or is hidden by overlap is addressed in Section 4's roofline-style model and Section 6's end-to-end pipeline analysis.

### Mixed MoE/Dense Architectures

For architectures where only some layers are MoE (e.g., $L_{\text{moe}} < L$), the EP communication cost is proportionally reduced. This is particularly important for models like DeepSeek-V2 or Mixtral variants that alternate between dense and MoE layers.

For a **pure dense model**: $L_{\text{moe}} = 0$, so the EP term vanishes entirely.

For a **pure MoE model**: $L_{\text{moe}} = L$, recovering the original formula.

### Summary of Collective Types and Message Sizes

| Parallelism | Occurs in | Collective Type | Calls/layer | Message Size (per device, per step) | Layer Types |
|-------------|-----------|------------------|-------------|----------------------------|-------------|
| **PP** | between layers | point-to-point | 1 | $B\cdot(H/D_{\text{kv}})\,b$ | All |
| **TP (attn)** | attention output | AR (TP-attn) or AG (DP-attn) | 1 | $B\cdot H\,b$ | All |
| **TP (FFN)** | MLP / expert output | AR | 1 | $B\cdot H\,b$ | All |
| **EP** | MoE FFN | all-to-all | 2 | $B\cdot kH\,b$ | MoE only |
| **SP** | attention | all-gather (ring) | 1 | $B\cdot(S/SP)\cdot (2H_{kv}/D_{\text{kv}})\, b$ | All |

At $B=1$ these reduce to the classical single-token payloads. The B-factor reflects that a decode step processes $B$ activations concurrently, so each collective carries $B \times$ the per-token activation vector.

### Practical Guidance: Shipped Algorithm Selection

Each collective in this section uses the algorithm that is actually shipped on the target fabric; other algorithms (Bruck A2A, recursive-doubling AR, PAT AG) are reference-only and live in `modeling/collectives/01 App. B`. Selection rules:

- **TP All-Reduce:** NCCL ships both ring and double binary tree (DBT) on a star fabric; the choice is a manual tuner knob `tuner.ar_algorithm` (`collectives/02_topology_mapping.md §2`), default `"ring"`. On torus fabrics (TPU / Trainium), only dim-decomposed ring ships — the knob is ignored. Empirical crossover: DBT wins at small $M$, ring wins at large $M$ ([DEMYST-NCCL]).

- **EP All-to-All:** NCCL ships pairwise direct-send on a star; TPU / Trainium ships the bisection-bound pairwise form on a torus (`collectives/01_collective_algorithms.md §7`). Log-hop (Bruck) A2A is **not** shipped and does not appear in this section's formulas.

- **SP All-Gather:** Ring AG is the only shipped form in production inference stacks — KV shards are large and must be processed in sequence order, so tree variants are impractical. This applies to both star (`collectives/01_collective_algorithms.md §6`) and torus (`collectives/02_topology_mapping.md §3`).

See `collectives/00_summary.md §4–§7` for the full shipped-primitive inventory and per-topology cost formulas (including hierarchical RS → sub-AR → AG and in-network reduction); `collectives/05_contention_and_congestion.md` for the contention coefficients $(\eta_\alpha, \eta_\beta)$.

---

<div style="page-break-before: always;"></div>

# 6. Partition Strategy and Hardware Latency

This chapter brings memory, compute, and communication together at the **per-stage** level. It produces $t_{\text{stage,hw}}$, the hardware-intrinsic GPU-side step time, and the HBM-feasibility constraint that gates which partitions are even runnable. Two axes are covered:

1. Feasible model partitioning via **HBM limits** (§6.1)
2. Local per-token latency via the **compute–memory roofline** plus collective overlap (§6.2)

§7 layers scheduling and software costs on top of $t_{\text{stage,hw}}(B)$ to produce the user-observed metrics: it derives the pipeline bubble and kernel-launch overhead that turn $t_{\text{stage,hw}}(B)$ into $t_{\text{step,user}}(B)$ (and the throughput metrics TPS / TTPS). TTFT is owned by `prefill.md`.

---

## 6.1 Model Partition Strategy from HBM Constraints

A parallel configuration $(DP, PP, EP, TP, SP)$ together with the layout / attention-mode choice (notation.md §1) is **feasible** only if each device can store:

- its **parameter shard**,
- its **KV-cache shard**, and
- the **activation workspace** needed for a single decoding token,

within the available HBM capacity $M_{\text{HBM}}$.

This subsection uses the per-component effective sharding factors $D_{\text{attn}}$, $D_{\text{exp}}$, $D_{\text{kv}}$, $D_{\text{emb}}$ from §1.4 above (canonical source: `notation.md §1`). Replica counting introduces one additional per-layout quantity not used elsewhere in decode.md, $N_{\text{replica}}$ (the number of independent model copies a given partition produces), which we list here for the partition-strategy derivations below:

| layout × attention_mode | $N_{\text{replica}}$ |
|---|---|
| orthogonal + TP-attn | $PP \cdot TP \cdot EP \cdot SP$ |
| co-located + TP-attn | $PP \cdot \max(TP, EP) \cdot SP$ |
| orthogonal + DP-attn | $PP \cdot TP \cdot EP \cdot SP$ |
| co-located + DP-attn | $PP \cdot \max(TP, EP) \cdot SP$ |

The two co-located rows collapse to the same $N_{\text{replica}}$ formula because $TP = EP$ structurally under co-location.

We define the total per-device static footprint as

$$
M_{\text{device}}^{\text{total}}(B) =
M_{\theta,\text{device}}
+
M_{\text{act,device}}(B)
+
M_{\text{KV,device}}(B)
\;\le\;
M_{\text{HBM}}
$$

Using the per-device memory expressions derived in Section 1, the **fully expanded** form for uniform architectures is:

$$
\frac{L}{PP}\;
\left[
\frac{2H^2 + 2 H H_{kv}}{D_{\text{attn}}}
\;+\;
\frac{3 H I N_{\text{exp}}}{D_{\text{exp}}}
+
\frac{2 S H_{kv}}{D_{\text{kv}} \cdot SP}
\right] b
+\;
B(4H + 2H_{kv}) b
+
\frac{V H}{D_{\text{emb}}} b
\le\;
M_{\text{HBM}}
$$

where:

- the bracketed term is the **intermediate PP-stage** footprint (parameters, activations, KV cache).
- and the final $\frac{V H}{D_{\text{emb}}} b$ term models the **worst-case embedding / LM-head overhead** on boundary PP stages.
- for a **dense MLP model**: $I = I_{\text{dense}}$, and $N_{\text{exp}} = EP = 1$ (so $D_{\text{exp}} = TP$).
- for a **MoE model**: $I = I_{\text{moe}}$, with $N_{\text{exp}} > 1$ and $EP \ge 1$.

For **mixed MoE/dense architectures**, the memory constraint uses the split formula from Section 1.4, where dense and MoE layer contributions are computed separately.

### Calculating DP for a Fixed Total HBM Capacity

The total Data Parallelism degree ($DP$) is constrained by both the total cluster size ($N_{\text{GPUs}}$) and the memory headroom available on each device.

1. **Memory Headroom Requirement:** $DP$ scaling is only possible if $M_{\text{device}}^{\text{total}} \le M_{\text{HBM}}$ for the chosen inner sharding degrees ($PP, EP, TP, SP$) and layout / attention-mode choice.
2. **Replication Logic:** Each model replica requires a dedicated group of $N_{\text{replica}}$ devices (notation.md §1; layout-dependent).

Let $N_{\text{GPUs}}$ be the total number of devices in the cluster. The maximum achievable $DP$ count is:

$$
DP = \left\lfloor \frac{N_{\text{GPUs}}}{N_{\text{replica}}} \right\rfloor
$$

with $N_{\text{replica}} = PP \cdot TP \cdot EP \cdot SP$ under the orthogonal layout and $N_{\text{replica}} = PP \cdot \max(TP, EP) \cdot SP$ under co-location.

**Physical Interpretation:**

- **Scaling Limit:** To increase $DP$ for higher throughput ($TTPS$), one must either add more total GPUs to the cluster or increase inner sharding (e.g., higher $PP$ or $SP$) to reduce $M_{\text{device}}^{\text{total}}$, though the latter consumes more devices per replica.
- **Footprint vs. Replica Count:** There is a direct trade-off: higher sharding degrees "thin out" the memory footprint per device to fit large context $S$ or large models, but they simultaneously reduce the number of independent replicas that can fit in a fixed cluster.
- **Co-location's replica-shrink effect:** Switching to TP+EP co-location drops $N_{\text{replica}}$ from $PP \cdot TP \cdot EP \cdot SP$ to $PP \cdot \max(TP, EP) \cdot SP$ — at $TP = EP = 8$ this is an 8× reduction (from 64 GPUs to 8 GPUs per replica), correspondingly multiplying $DP$ for the same cluster size. The cost is that each device must now hold $1/EP$ of the experts (rather than $1/(TP \cdot EP)$); see §6.3 for when this trade pays.

---

## 6.2 Local and Networking Per-Step Latency

All quantities below are **per step, per stage**, with $B$ sequences batched together in one decode iteration. They are restatements of the building blocks from §3–§5 with $B$ threaded explicitly.

### Compute-bound latency

$$
t_{\text{compute}}(B) =
\frac{B \cdot F_{\text{token,device}}}{R_{\text{GPU}}}
$$

### Memory-bandwidth-bound latency

Weights amortize once per step; KV reads scale with $B$ (§4):

$$
t_{\text{mem}}(B) =
\frac{T_{\theta,\text{device}} + B \cdot T_{\text{KV,token}}}{BW_{\text{mem}}}
$$

This is the single-tier, dropped-$\alpha$ shorthand of the multi-tier $\alpha$–$\beta$ form in §4 — kept here for compactness on the assumption that on-die tier $\alpha$ contributes well under 0.1% of $t_{\text{mem}}$. Reinstate $\alpha_i$ per §4 when modeling small-read regimes.

**Sustained vs nameplate $BW_{\mathrm{mem}}$.** The system spec stores HBM bandwidth at the nameplate / datasheet value (e.g., 8 TB/s for HBM3e on B200, 4.8 TB/s for HBM3 on H200). Real production stacks sustain only a fraction of this peak: the dominant losses are bank conflicts under concurrent address streams (KV reads from many sequences hitting the same row), memory-controller queue depth, and competition between weight reads and KV-page-table gathers. Reported anchors across HBM generations and decode workloads:

| Hardware (HBM gen) | Sustained / nameplate $BW_{\mathrm{mem}}$ | Notes |
|---|---|---|
| H100 / H200 (HBM3) | ~0.55 | Older memory controllers; sustained around 2.6 TB/s on H200's nameplate 4.8 TB/s. |
| B200 / GB200 (HBM3e) | 0.7–1.0 | Better controllers + Blackwell command processor; the upper end (1.0) holds for software stacks that fuse aggressively on MoE workloads where access patterns are friendly; the lower end (0.7) for less-fused stacks on dense models. |
| B300 / GB300 (HBM3e) | similar to B200 | Same memory subsystem class. |
| Dense models on minimally-fused runtimes | as low as 0.4 | Worst-case observed on dense 70B-class models. Dense general matrix multiply (GEMM) sustains *less* of peak than MoE expert hopping on Blackwell — counter-intuitive but consistent with B200 production reports. |

The right derate depends jointly on the chip, the model class, and the software stack — not on the chip alone — so it belongs in the calibration layer (a per-tier deflator on the device spec, or a single-tier override at analysis time) rather than baked into the device spec as a fixed property.

#### B-dependent sustained bandwidth $\eta_\beta(B)$

A constant $\eta_\beta$ captures the average sustained / nameplate ratio for a given (chip, software stack) but understates a real effect: HBM sustained bandwidth degrades with the active-sequence count $B$. Three mechanisms drive this:

- **Bank conflicts grow with concurrent address streams.** At small $B$ the weight-load pattern is well-coalesced — every weight byte loaded once, in long contiguous bursts. As $B$ grows, each step also reads $B$ separate KV histories whose addresses are scattered across HBM banks (paged-attention block-table indirection); collisions on the same DRAM row reduce the row-buffer hit rate.
- **Memory-controller queue depth saturates.** Each HBM stack has a finite queue of in-flight requests. At small $B$ the queue is mostly weight reads; at large $B$ the queue fills with KV reads competing for the same controllers, and per-request latency grows even as throughput plateaus.
- **PCIe metadata bandwidth crowds in.** Paged-attention block-table updates and per-step scheduler decisions push small writes to GPU memory across PCIe between forward passes. At very large $B$ this "noise" memory traffic eats into the HBM channels' usable bandwidth.

Reported HBM3e numbers on production-style decode workloads with paged attention: ~92% of peak at low concurrency ($B \le 16$), ~85% at moderate $B$ ($\sim 64$), ~75% at production $B$ ($\sim 512$), ~55% at very large $B$ ($\sim 4000$). The decline is monotone but non-linear — fastest in the $B \in [16, 512]$ range, flattening above $B \approx 4000$.

Promoting $\eta_\beta$ to a function of $B$ extends the per-step memory-bandwidth-bound latency to:

$$
t_{\mathrm{mem}}(B) = \frac{T_{\theta,\mathrm{device}} + B \cdot T_{\mathrm{KV,device}}}{BW_{\mathrm{mem,nameplate}} \cdot \eta_\beta(B)}
$$

When $\eta_\beta(B) \equiv 1$ the formula reduces to the constant-bandwidth form above. A practical specification is a piecewise-linear curve through a small set of anchor batch sizes; a representative anchor set for HBM3e on Blackwell-class production stacks:

$$
\eta_\beta(B) \in \{(1, 0.92),\; (64, 0.85),\; (512, 0.75),\; (4096, 0.55)\}
$$

with linear interpolation between adjacent anchors and clamping at the boundaries.

**Composition.** $\eta_\beta(B)$ composes multiplicatively with any constant per-tier deflator $\eta_{\beta,\mathrm{tier}}$ (sram.md §1.2) and with the static sustained / nameplate calibration described above: $BW_{\mathrm{eff}}(B) = BW_{\mathrm{nameplate}} \cdot \eta_{\beta,\mathrm{tier}} \cdot \eta_\beta(B)$. In practice the analyst selects one of these to carry the sustained-vs-peak gap rather than composing all of them — a single calibrated quantity is easier to reason about.

**When the curve matters.** For decode workloads with $B \le 1024$ a constant $\eta_\beta$ is usually sufficient — one calibration point captures the dominant sustained-vs-peak gap. The B-dependent curve becomes load-bearing only when (a) decode runs in a genuinely $t_{\mathrm{mem}}$-bound regime — dense models on small-replica server-class deployments, very long contexts, MoE deployments with cheap all-to-all (small expert parallelism (EP), small per-expert FLOPs) — **and** (b) $B$ extends across more than ~1.5 decades so the BW-saturation curvature shows up against measured time per output token (TPOT). Workloads dominated by collective communication (large EP, MoE all-to-all (A2A) on multi-node fabrics) sit on a different bottleneck and are unaffected by $\eta_\beta(B)$.

### Roofline local latency

$$
t_{\text{local}}(B) =
\max\bigl(t_{\text{compute}}(B),\; t_{\text{mem}}(B)\bigr)
$$

### Collective communication latency

From §5.5, with each per-axis $t_*(B)$ carrying its own $B$ scaling on the β-side and the per-layer TP cost $t_{TP}^{\text{layer}}(B)$ encoding the attention-mode-dependent AR / AG split (§5.3):

$$
t_{\text{comm}}(B)
\approx
\frac{L}{PP}
\bigl(
t_{TP}^{\text{layer}}(B) +
n_{SP}\, t_{SP}(B)
\bigr) +
\frac{L_{\text{moe}}}{PP}
\bigl(
n_{EP}\, t_{EP}(B)
\bigr) +
t_{PP}(B)
$$

Note: EP collectives only apply to MoE layers ($L_{\text{moe}}$), while TP and SP collectives apply to all layers.

### Unified Overlap Model

We introduce an overlap factor $\rho_{\text{comm}} \in [0, 1]$ representing the fraction of local compute/memory time that is successfully utilized to hide communication.

The effective per-stage HW time is the local time plus any **unhidden** communication:

$$
t_{\text{stage,hw}}(B) =
t_{\text{local}}(B)
+
\max\bigl(0,\; t_{\text{comm}}(B) - \rho_{\text{comm}} \cdot t_{\text{local}}(B)\bigr)
$$

This is the GPU-side wall-clock cost of one pipeline stage processing one decode step's worth of $B$ sequences — purely **hardware-intrinsic** (compute + memory + interconnect, after collective overlap), no scheduling or host-side overhead. §7 introduces the per-stage kernel-launch dispatch budget $t_{\text{stage,kernel}}$ (§7.1), assembles the per-step hardware window with pipeline-bubble correction (§7.2), and adds the per-sequence serving overhead plus throughput (§7.3) to produce the user-observed step time $t_{\text{step,user}}$.

**Regimes:**

- **$\rho_{\text{comm}} = 0$ (No Overlap):**
  $$t_{\text{stage,hw}}(B) = t_{\text{local}}(B) + t_{\text{comm}}(B)$$
  Typical for naive implementations or strictly sequential dependencies.

- **$\rho_{\text{comm}} = 1$ (Perfect Overlap Opportunity):**
  $$t_{\text{stage,hw}}(B) = t_{\text{local}}(B) + \max(0, t_{\text{comm}}(B) - t_{\text{local}}(B)) = \max(t_{\text{local}}(B), t_{\text{comm}}(B))$$
  Achieved by highly optimized kernels (e.g., Ring Attention) where independent work exists.

- **$0 < \rho_{\text{comm}} < 1$ (Partial Overlap):**
  Models real-world overheads (synchronization barriers, partial dependency chains) that prevent utilizing the full local duration for hiding comms. Kernel-launch overhead is *not* part of $\rho_{\text{comm}}$ here — it is modeled separately as $t_{\text{stage,kernel}}$ with its own overlap factor $\rho_{\text{kernel}}$ in §7.1.

**Realistic $\rho_{\text{comm}}$ for decode is small.** A single decode step traverses every layer once with a tight data-dependency chain — attention → TP all-reduce → MoE dispatch (A2A) → expert FFN → MoE combine (A2A) → norm → next layer — and each step processes one batch through the full stack. Unlike training (and unlike prefill, which is compute-bound and has long collective windows to hide behind GEMM tiles), decode has no microbatch rotation to pipeline communication behind compute when $PP$ is small (the typical production case). The within-layer overlap opportunities are narrow: stream-concurrent expert dispatch within an MoE block (dispatch to expert $i+1$ overlaps with FFN of expert $i$), TP all-reduce overlapped with the post-projection layernorm, and next-step host-side preparation (sampling, KV append) overlapped with the current step's tail. Aggressively fused MoE dispatch implementations claim to hide roughly half the A2A behind compute under favorable conditions; vanilla decode stacks see far less. The realistic decode range is $\rho_{\text{comm}} \approx 0$–$0.3$, with $\rho_{\text{comm}} = 0$ a sensible default for any deployment without explicit dispatch fusion or stream-concurrent expert pipelining. The $\rho_{\text{comm}} \approx 0.7$–$1.0$ values familiar from training and prefill literature are not directly applicable here.

### LM head latency (stage PP-1 only)

The LM head $H \to V$ projection is a per-step one-shot kernel that fires only on the last PP stage. Its FLOPs ($2BHV/D_{\text{emb}}$, §3) and weight + output traffic ($T_{\text{LM},\theta,\text{device}} = HVb/D_{\text{emb}}$ from §2.1, plus output activations $B \cdot V \cdot b$) define a small standalone roofline:

$$
t_{\text{LM,hw}}(B) =
\max\!\left(
  \frac{2 B H V / D_{\text{emb}}}{R_{\text{GPU}}},\;
  \frac{H V b / D_{\text{emb}} + B V b}{BW_{\text{mem}}}
\right)
$$

Because the LM head lives on stage $PP{-}1$ only (not divided by $PP$, $EP$, or $SP$), it is bookkept as a separate additive term rather than folded into the per-layer $t_{\text{local}}(B)$. §7.2 composes it on top of the per-stage body cost as a stage-$PP{-}1$ surcharge — under the uniform-stage assumption this makes stage $PP{-}1$ the throughput bottleneck.

## 6.3 When Each Layout / Attention-Mode Pays

§6.1 and §6.2 cover *what* each $(layout, attention\_mode)$ configuration costs in HBM and per-step latency. This subsection covers *which* configuration to pick for a given deployment — when each of the four production-relevant configurations from `notation.md §1` actually pays off relative to the orthogonal + TP-attention default.

### Orthogonal + TP-attention (the default)

The legacy Megatron-LM partition: every linear layer is TP-sharded by head, KV cache is head-sharded by $TP$, the per-layer attention block fires an all-reduce. Use this when:

- **Multi-head attention (MHA) or grouped-query attention (GQA) with non-trivial $H_{kv}$**, where attention parameters are 8–15% of model size and replicating them on every TP rank would meaningfully inflate the per-device memory budget.
- **Small $G_{TP}$ on a single fast fabric tier** (e.g., $TP \le 8$ on NVLink), where the per-attention all-reduce is small enough that the AR → AG savings in DP-attention are second-order.
- **Mixed dense + MoE architectures** where the attention block's TP shard is meaningful relative to the dense FFN's TP shard.

### Orthogonal + DP-attention

Replicates attention weights on every TP rank; replaces the per-layer attention all-reduce with an all-gather (half the cost). The trade is one-attention-AR-per-layer saved against $TP - 1$ extra copies of the attention weights in HBM and one extra all-reduce-equivalent of attention weight memory traffic per step. Favorable when:

1. **Attention parameters are a small fraction of model size.** Multi-head Latent Attention (MLA, DSv3) shrinks $H_{kv}$ from $\sim H$ in MHA down to $\sim H / 14$, dropping attention parameters from $\sim$15% of the model to $\sim$3%. Replicating 3% of the weights costs little; saving an all-reduce per layer is worth a lot when there are 60+ layers and the all-reduce sits on the critical path. [DSV3, SGLANG-DPATTN]
2. **$G_{TP}$ is large enough that $G_{TP} - 1$ all-reduces of $B \cdot Hb$ payload per stage dominate the latency budget.** At $G_{TP} = 8$ on NVLink the savings are second-order; at $G_{TP} = 32$ on a multi-node setup they become first-order.
3. **Memory-bound regime** ($B$ small relative to crossover $B^\star$ from §4). Here the extra attention weight traffic adds at most $\frac{G_{TP} - 1}{G_{TP}} \cdot \frac{M_{\text{attn, TP-attn}}}{M_{\theta,\text{device, TP-attn}}}$ to the per-step weight traffic — for an MLA-class model with attention parameters at $\sim$3% of total weights and $G_{TP} = 8$, that is a ~2.6% increase in $t_{\text{mem}}$. The per-layer AR savings typically dwarf this term, so the trade is favorable; in compute-bound regime the trade is closer to neutral because attention compute is invariant under the swap.

For models with full MHA or GQA where attention parameters are 8–15% of total weight, the calculus is closer; for models with MLA the trade essentially always favors DP-attention.

### Co-located + DP-attention

TP and EP overlap on the same physical GPUs of a replica, dropping $N_{\text{replica}}$ from $PP \cdot TP \cdot EP \cdot SP$ to $PP \cdot \max(TP, EP) \cdot SP$. With both axes inside one NVLink island, both the TP collective and the EP all-to-all stay on the fastest fabric tier. The catch: each device now holds $1/EP$ of the experts (no further TP shard within an expert), which inflates per-device expert memory by a factor of $TP$ relative to the orthogonal layout. Favorable when:

1. **Per-device HBM can absorb the larger expert footprint.** Either the expert weights are small enough (low $N_{\text{exp}}$ or compact $I_{\text{moe}}$) or aggressive quantization (FP4 / INT4) keeps $1/EP$ of the experts under HBM. DSv3 satisfies this via FP4 + a sparsity factor of $k = 8$ active experts out of $N_{\text{exp}} = 256$. [DSV3]
2. **All collectives benefit from staying inside one fast fabric tier.** Orthogonal $TP = 8, EP = 8$ on 64 GPUs spans multiple NVLink islands — the EP all-to-all crosses a slower tier on every layer, paying the slow tier's $\alpha$ on every collective. Co-located $TP = EP = 8$ on 8 GPUs keeps both collectives on NVLink. The latency win is concentrated in the small-$M$ decode regime where $\alpha$ dominates.
3. **Attention mode choice under co-location.** Co-location supports both attention modes on the overlaid GPU set: TP-attn head-shards across the same group that holds the experts (no replication cost; per-layer TP all-reduce fires on the same ranks as the MoE A2A), and DP-attn replicates attention and batch-shards (replication cost equal to orthogonal + DP-attn — fine for MLA / aggressive GQA models, costly for full MHA). Pick TP-attn under co-location when the attention block is head-structured enough to benefit from sharding ($n_q$ divides $TP$, GQA $n_{kv} \geq TP$, or full MHA); pick DP-attn when MLA flatness or aggressive-GQA $n_{kv}$ saturation makes attention replication free anyway.

In short: co-location is the right choice when the workload is MoE-dominated, attention is small (MLA-class), and keeping the world inside one NVLink island unlocks meaningful $\alpha$-cost savings on every collective.

### Quick decision table

| Symptom | Recommended configuration |
|---|---|
| Dense or small-MoE model, MHA/GQA attention, $G_{TP} \le 8$ | orthogonal + TP-attn (the default) |
| MLA-class attention, single fabric tier, $G_{TP} \le 8$ | orthogonal + DP-attn |
| MLA-class attention, large $G_{TP}$ on multi-tier fabric | orthogonal + DP-attn |
| MLA-class attention, MoE-dominant, $TP = EP$ targets a single NVLink island | co-located + DP-attn |
| MoE-dominant, head-structured attention (MHA / GQA with $n_{kv} \ge TP$), $TP = EP$ overlaid on the same NVLink island | co-located + TP-attn (DSr1 / NVL72 panel-(b) pattern) |
| Multi-tier fabric where orthogonal would put EP A2A on slower tier | co-located + DP-attn (when memory permits) |

---

<div style="page-break-before: always;"></div>

# 7. Host-Side Overheads and Throughput

§6.2 produced $t_{\text{stage,hw}}(B)$ (per-stage body) and $t_{\text{LM,hw}}(B)$ (one-shot LM head on stage $PP{-}1$) — the **hardware-intrinsic** per-step time pieces at batch size $B$, the lower bound set by tensor-core compute, HBM bandwidth, and on-/off-chip interconnect. Two derived quantities follow directly from them:

- **Per-replica throughput bottleneck** $\max_j t_{\text{stage,hw},j}(B) + t_{\text{LM,hw}}(B)$ — sets the steady-state output rate (tokens/s per replica): once the pipeline is full, the bottleneck stage (always $PP{-}1$ under uniform body) finishes a step every $t_{\text{stage,hw}}(B) + t_{\text{LM,hw}}(B)$ seconds, so the user-observed token rate is gated by the slowest stage. This is the quantity that drives **TPS** in §7.3.
- **Pipeline traversal time** $\sum_{j=1}^{PP} t_{\text{stage,hw},j}(B) + t_{\text{LM,hw}}(B)$ — the cost a token pays to walk all $PP$ stages once and then through the LM head. The *first* token of any sequence always pays this (the pipeline starts empty before prefill), so it is the primary HW-side contribution to **TTFT** (covered in prefill.md); subsequent decode steps pay only the bottleneck-stage cost once the pipeline is filled.

A real serving step does not run at this lower bound. Three additional costs accrue on every token generation step on top of $t_{\text{stage,hw}}(B)$ — the first two are **host-side overheads** (CPU work and command-processor dispatch sitting outside the GPU pipeline), the third is a scheduling inefficiency:

1. **Kernel-launch overhead** (per microbatch, ~constant in $B$) — the host-side dispatch budget for the layer + collective + PP-hop kernels that fire each microbatch. Each kernel launch pays a fixed $\tau_{\text{launch}}$ regardless of payload size, so the term scales with kernel count, not with $B$. Whether it is hidden by GPU work or surfaces as a hard floor depends on async dispatch overlap. §7.1 derives this as $t_{\text{stage,kernel}}$ in the same per-step per-stage units as $t_{\text{stage,hw}}$.
2. **Pipeline bubble** when the pipeline is underfilled — a scheduling-driven inefficiency that scales with $PP/B$ when $B < PP$. The bubble factor $\gamma_{\text{pp}}$ is a **global multiplier** on whatever per-stage body cost dominates the round, so it sits *outside* the per-axis composition rather than as a per-axis correction.
3. **Per-sequence serving runtime overhead** (per sequence, linear in $B$) — host-side bookkeeping the runtime must perform once per active sequence per step: PagedAttention block-table assembly, continuous-batching scheduler decisions, sampling glue, token-append. The term scales as $O(B)$ and sits on the critical path of every step (it cannot overlap with GPU work in eager mode; under CUDA-Graph replay it hides behind the per-step GPU window until it overflows). §7.3 derives this as $t_{\text{step,seq}}(B)$ and assembles the full $t_{\text{step,user}}(B)$ on top of §7.2's per-step hardware window.

The two host-side overheads are distinct in three ways: scaling with $B$ (constant vs linear), where the cost is paid (per microbatch dispatch vs once-per-step head-node bookkeeping), and whether they can overlap with GPU work (yes, partially, for kernel-launch via CUDA Graphs / async pipelining; partially under CUDA-Graph replay for serving runtime, zero overlap in eager-mode stacks). §7.2 assembles $t_{\text{stage,hw}}(B)$, $t_{\text{stage,kernel}}$, $\gamma_{\text{pp}}$, and $t_{\text{LM,hw}}(B)$ into the per-step hardware window $t_{\text{step,base}}(B)$; §7.3 adds $t_{\text{step,seq}}(B)$ on top to recover the user-observed step time $t_{\text{step,user}}(B)$ and derives the throughput metrics TPS / TTPS.

## 7.1 Kernel-launch overhead

The user-observed step time also includes a dispatch budget $t_{\text{stage,kernel}}$ — the cost of getting one microbatch's worth of kernels onto the GPU on this stage. Where that cost is paid depends on the execution mode: in **eager** mode it is literal CPU `cudaLaunchKernel` latency on the host side ($\tau_{\text{launch}} \approx 7\,\mu s$); under **CUDA Graphs / DAG replay** the host issues a single `cudaGraphLaunch` per microbatch and the per-node work shifts to the GPU's command processor walking the captured DAG ($\tau_{\text{launch}} \approx 1.5\,\mu s$, dominated by command-buffer overhead, not CPU API time). The framework collapses both regimes into a single $\tau_{\text{launch}}$ knob — the term name "kernel-launch overhead" applies to both paths; earlier drafts called this "SW overhead" but that label is now reserved for the umbrella category covering this term plus per-sequence serving runtime (§7.3) and any future per-step host floor.

$t_{\text{stage,kernel}}$ is **per microbatch, per stage** — same units as $t_{\text{stage,hw}}$ (§6.2), so the two compose directly in §7.2 without a unit mismatch. EP launches only fire on the $L_{\text{moe}}/PP$ MoE layers this stage owns (mirroring the $L_{\text{moe}}/PP$ factor in §5.5's $t_{\text{comm}}$ formula); for a pure dense model $L_{\text{moe}} = 0$ and the EP term vanishes.

$$
t_{\text{stage,kernel}} = \tau_{\text{launch}} \cdot \left[ \frac{L}{PP} \bigl( k_{\text{compute}} + k_{\text{collective}}(n_{TP} + n_{SP}) \bigr) + \frac{L_{\text{moe}}}{PP} \cdot k_{\text{collective}} \cdot n_{EP} + k_{\text{pp\_hop}} \right]
$$

**What $k_{\text{compute}}$ counts.** $k_{\text{compute}}$ is the number of non-NCCL compute kernels fired per layer — GEMMs, attention, layernorms, activations, residuals. With a fused stack (FlashAttention + Megatron-style fused MLP) a decode layer fires roughly: (1) fused QKV projection, (2) FlashAttention forward, (3) output projection, (4) fused residual + post-attention norm, (5) fused gate + up + activation, (6) down projection, (7) residual. ~8–12 launches in production; default is 10. Aggressive ahead-of-time compilation (e.g. TensorRT engine) drops this to 3–5; eager-mode PyTorch without fusion can push it to 30+. Tighter compilation reduces $k_{\text{compute}}$; CUDA Graphs reduce $\tau_{\text{launch}}$ — these are independent levers and production stacks generally use both.

**Per-layer accounting — two nested layers for collectives.** $n_{TP}, n_{EP}, n_{SP}$ from §5.5 count *NCCL API calls* per layer (logical operations: e.g. $n_{TP} = 2$ means "two `ncclAllReduce` calls per layer — one in attn, one in MLP"). The $t_{\text{comm}}$ formula in §5.5 prices each call once via $n_* \cdot t_*$. The launch counter goes one level deeper: each NCCL API call internally fires $k_{\text{collective}}$ separate `cudaLaunchKernel` events (default 2 — typically a setup/coordination kernel plus the reduction kernel), so the *total kernel launches per layer per axis* is $n_* \cdot k_{\text{collective}}$. That product is what $\tau_{\text{launch}}$ is paid for. Custom collectives that fuse the call into a single kernel set $k_{\text{collective}} = 1$; multi-kernel implementations may set it higher. Each $n_*$ is zeroed when the corresponding axis size is 1 (the collective never fires).

Three additive contributions in the bracket: (1) compute + TP + SP launches on every layer this stage owns ($L/PP$ layers); (2) EP launches on MoE layers only ($L_{\text{moe}}/PP$ layers); (3) one P2P transit (default $k_{\text{pp\_hop}} = 2$: 1 recv + 1 send on a middle stage; edge stages do only one direction, off by half a launch — negligible at $PP > 1$). The PP-hop term is inert when $PP = 1$.

$t_{\text{stage,kernel}}$ is composed with the GPU-side $t_{\text{stage,hw}}$ via the kernel-launch overlap factor $\rho_{\text{kernel}} \in [0, 1]$ — at $\rho_{\text{kernel}} = 1$ (the production default for CUDA-Graphs-replayed serving), async dispatch fully overlaps with GPU work and $t_{\text{stage,kernel}}$ acts as a *floor* that only kicks in when the per-microbatch launch budget exceeds $t_{\text{stage,hw}}$; at $\rho_{\text{kernel}} = 0$, dispatch is synchronous and the costs add linearly. Setting `kernel_launch_us = 0` in the tuner zeros $t_{\text{stage,kernel}}$ entirely (legacy roofline behavior).

A separate Tensor Core efficiency term $\eta_{\text{TC}}(\text{mb})$ derates the compute roofline at small microbatch; with the default $\eta_{\text{TC}} = 1$ (no curve set in the tuner) this term is a no-op.

## 7.2 Per-step hardware window

Building on the kernel-launch dispatch budget $t_{\text{stage,kernel}}$ from §7.1 and the per-stage hardware body $t_{\text{stage,hw}}(B)$ from §6.2, we assemble the **per-step hardware window** $t_{\text{step,base}}(B)$ — the full hardware-side budget for one decode step on the bottleneck stage, including pipeline-bubble inflation and the LM head surcharge on stage $PP{-}1$. When the serving runtime adds no host-side per-sequence work (the host-overhead-free roofline), the user-observed step time equals $t_{\text{step,base}}(B)$. §7.3 then adds the per-sequence serving overhead on top of $t_{\text{step,base}}(B)$ to recover the full $t_{\text{step,user}}(B)$ and derives throughput from it.

### Saturated step time (no bubble)

When the pipeline is **fully saturated** ($B \ge PP$), every stage holds a different microbatch on every step and all $PP$ stages run in parallel. The per-step hardware window collapses to the slowest per-stage cost — the assembled HW + kernel-launch cost on the bottleneck stage, plus the LM head surcharge on stage $PP{-}1$, with no bubble penalty.

Composing $t_{\text{stage,hw}}(B)$ (§6.2) and $t_{\text{stage,kernel}}$ (§7.1) follows the same overlap pattern as the compute/comm overlap in §6.2: hardware work runs as the base, host dispatch overlaps for a fraction $\rho_{\text{kernel}}$ of it, and any unhidden remainder serializes after. The LM head term $t_{\text{LM,hw}}(B)$ from §6.2 then adds on top (it fires once per step on the bottleneck stage, outside the per-stage composition):

$$
t_{\text{step,base}}^{\text{sat}}(B) \;=\; t_{\text{stage,hw}}(B) + \max(0,\; t_{\text{stage,kernel}} - \rho_{\text{kernel}} \cdot t_{\text{stage,hw}}(B)) + t_{\text{LM,hw}}(B)
$$

Under the uniform-stage assumption (all $PP$ stages have identical body cost), the bottleneck stage is $PP{-}1$ and the additive $t_{\text{LM,hw}}(B)$ exactly captures its extra workload. For $PP = 1$, this collapses to "single-stage body + LM head".

**Regimes (assuming $\rho_{\text{kernel}} = 1$, the production CUDA-Graph default):**

- $t_{\text{stage,hw}}(B) \ge t_{\text{stage,kernel}}$: the overflow term is 0, $t_{\text{step,base}}^{\text{sat}}(B) = t_{\text{stage,hw}}(B) + t_{\text{LM,hw}}(B)$ (HW-bound on the body — async dispatch fully hidden by hardware work). This is the typical regime for moderate-to-large $B$ on production stacks.
- $t_{\text{stage,hw}}(B) < t_{\text{stage,kernel}}$: the kernel-launch budget surfaces as a floor on $t_{\text{step,base}}$ (kernel-launch-bound on the body — common at small $B$ on dense decode).
- $\rho_{\text{kernel}} = 0$: no overlap — kernel-launch dispatch serializes additively after GPU compute. Eager-mode stacks land here.

When $t_{\text{stage,kernel}} = 0$ (kernel-launch modeling disabled) the formula collapses to $t_{\text{stage,hw}}(B) + t_{\text{LM,hw}}(B)$ — the pure GPU-side roofline.

### Pipeline bubble correction

The saturated form above assumes the pipeline is full. When it is not, the bubble factor inflates the step time. **How the PP pipeline works:** pipeline parallelism splits the model's $L$ layers into $PP$ contiguous stages, each owned by a different device (or device group). A token cannot skip stages — it is transformed by stage 0, then forwarded to stage 1, and so on through stage $PP{-}1$ before its logits are produced. To keep all stages busy in parallel, the batch of $B$ active sequences is split into microbatches that flow through the pipeline back-to-back: while stage 0 processes microbatch $i{+}1$, stage 1 is processing microbatch $i$, stage 2 is processing microbatch $i{-}1$, etc. This is the standard PP execution pattern from [MEGATRON3].

Two regimes follow from this picture, depending on whether there are enough microbatches in flight to keep every stage busy. The same scheduling logic applies to both HW and kernel-launch per-stage costs — the bubble doesn't care whether a stage is GPU- or dispatch-bound:

- **Pipeline full ($B \ge PP$).** Every stage runs a different microbatch in parallel; the saturated formula above holds. The full traversal cost $\sum_j t_{\text{stage,hw},j}(B)$ is paid only by the *first* token (TTFT), not by each subsequent decode step.
- **Pipeline underfilled ($B < PP$).** With fewer microbatches than stages, parallelism collapses: a single microbatch must walk all $PP$ stages in series before the user sees the next token, so both HW and kernel-launch step costs grow toward their traversal sums (with uniform stages and $B = 1$, these become $PP \cdot t_{\text{stage,hw}}(1)$ and $PP \cdot t_{\text{stage,kernel}}$).

A first-order bubble correction captures both regimes with a single multiplier:

$$
\gamma_{\text{pp}} = \max\left(1,\; \frac{PP}{B}\right)
$$

At $B \ge PP$, $\gamma_{\text{pp}} = 1$ and the bubble vanishes. At $B = 1, PP > 1$ it equals $PP$. At $B = 1, PP = 1$ it also reduces to unity, recovering the non-pipelined decode model. Because $\gamma_{\text{pp}}$ scales the per-stage body costs (HW and kernel-launch) identically, it multiplies the body composition only; the LM head fires once per step on stage $PP{-}1$ regardless of bubble depth and is added outside $\gamma_{\text{pp}}$:

$$
t_{\text{step,base}}(B) \;=\; \gamma_{\text{pp}} \cdot \bigl[t_{\text{stage,hw}}(B) + \max(0,\; t_{\text{stage,kernel}} - \rho_{\text{kernel}} \cdot t_{\text{stage,hw}}(B))\bigr] \;+\; t_{\text{LM,hw}}(B)
$$

This is the full **per-step hardware window** — the time the host can use to hide per-sequence serving work behind GPU compute. At $B = 1, PP = 1$ with $t_{\text{stage,kernel}} = 0$, it reduces to $t_{\text{stage,hw}}(1) + t_{\text{LM,hw}}(1)$ — single-stage body plus LM head, the non-pipelined decode model with a vocab projection at the end.

When the serving runtime contributes no per-sequence host work ($c_{\mathrm{seq}} = 0$, see §7.3), the user-observed step time equals the hardware window: $t_{\text{step,user}}(B) = t_{\text{step,base}}(B)$. The general case adds the per-sequence serving overhead on top — derived in §7.3.

## 7.3 Per-sequence serving runtime overhead and throughput

§7.1 captured the per-microbatch kernel-launch dispatch budget, and §7.2 assembled the per-step hardware window $t_{\text{step,base}}(B)$ from it. A separate class of host-side overhead is **per active sequence** rather than per microbatch: work the runtime must perform once for each of the $B$ in-flight sequences on every decode step, regardless of payload size. Production inference serving stacks (vLLM, TensorRT-LLM, SGLang, NVIDIA Dynamo) all incur this overhead [VLLM, ORCA, DYNAMO]; four components dominate it:

- **Block-table gather and metadata marshaling.** PagedAttention stores each sequence's key-value (KV) cache as a list of fixed-size blocks; on every step the runtime assembles the per-sequence block table (one indirection per active sequence) and ships it to the attention kernel as a metadata buffer. The per-sequence cost is small but unavoidable, and the total scales as $O(B)$.
- **Continuous-batching scheduler decisions.** Iteration-level scheduling (the ORCA / vLLM design [ORCA]) re-evaluates the active set on every step: admit any newly arrived request that fits, evict any sequence that just finished, decide whether prefills can piggyback this step. The decision logic is per-sequence and runs on the CPU between two consecutive forward passes.
- **Per-sequence sampling glue.** Greedy / top-$k$ / top-$p$ / multinomial sampling runs one decision per sequence per step. The logits-softmax-then-sample kernel itself is GPU-side and is absorbed into $t_{\text{LM,hw}}(B)$, but the host wraps each sampling decision in per-sequence Python / C++ glue (penalty application, stop-token checks, output-token append, end-of-sequence (EOS) / stop-string detection) that is genuinely $O(B)$ on the CPU.
- **Token-append and KV-write bookkeeping.** Append the newly sampled token to the per-sequence output buffer, advance the per-sequence position counter, allocate a new block on overflow, write the block-table update — one bookkeeping pass per active sequence per step.

Alongside these four per-sequence components, every decode step incurs a **B-independent orchestration floor** that fires once per step regardless of how many sequences are active:

- **Per-step orchestration**. CUDA-Graph launch decision logic, scheduler tick (active-set advance, time-budget accounting), output framing (response-stream multiplexing for streaming responses), KV block-table refresh metadata, and any Python sampler / scheduler glue that runs *between* graph replays rather than per-sequence. These costs survive CUDA-Graph absorption — the graph captures the GPU kernel sequence but not the host-side decisions that fire between graph launches. Empirically the dominant source of the residual under-prediction in DP-attention large-B regimes (§5 validation panel d).

This per-step host work is distinct from two other overheads tracked elsewhere:

- The kernel-launch dispatch budget $t_{\text{stage,kernel}}$ from §7.1 is **per microbatch per stage** and is essentially independent of $B$. Each NCCL or `cudaLaunchKernel` event fires once per layer regardless of the per-call payload; the per-event cost models GPU-command-processor latency or eager-mode CPU API time, not host-side per-step orchestration.
- The per-request scheduler latency $t_{\text{sched}}$ tracked in framework.md §1 fires **once per request** when the request enters the system, not on every decode step. The bookkeeping captured here is the recurring delta on top of $t_{\text{sched}}$ that fires every step the request remains active.

The aggregate per-step gross host cost combines the B-independent orchestration floor $c_{\mathrm{orch}}$ with the per-sequence linear term $c_{\mathrm{seq}} \cdot B$:

$$
t_{\text{step,seq}}(B) = c_{\mathrm{orch}} + c_{\mathrm{seq}} \cdot B
$$

with units of seconds per step (the framework parameterizes both knobs in microseconds for ergonomics). $c_{\mathrm{orch}}$ captures the once-per-step orchestration work; $c_{\mathrm{seq}}$ captures the per-sequence inner loops. Default $c_{\mathrm{orch}} = c_{\mathrm{seq}} = 0$ gives the legacy host-overhead-free roofline; both terms are opt-in calibration knobs.

**Composition with the hardware window.** Like $t_{\text{stage,kernel}}$ (§7.1), the per-sequence serving work admits a CUDA-Graph-replay overlap: under graph replay the CPU launches the per-step graph in $\sim$1.5 µs and is then free for the duration of the hardware step window $t_{\text{step,base}}(B)$ (§7.2), which it uses for exactly this per-sequence work (block-table assembly for the *next* step, sampling glue for the *previous* step's outputs, scheduler bookkeeping). The host work hides behind the hardware window until $t_{\text{step,seq}}(B)$ exceeds $t_{\text{step,base}}(B)$, at which point only the excess blocks. The composition is parameterized by an overlap factor $\rho_{\mathrm{seq}} \in [0, 1]$, applied to the per-step hardware window from §7.2:

$$
t_{\text{step,user}}(B) \;=\; t_{\text{step,base}}(B) + \max\!\bigl(0,\; t_{\text{step,seq}}(B) - \rho_{\mathrm{seq}} \cdot t_{\text{step,base}}(B)\bigr)
$$

with $\rho_{\mathrm{seq}} = 1$ for CUDA-Graph-replay stacks (full overlap; default), $\rho_{\mathrm{seq}} = 0$ for eager-mode stacks where Python interpreter stalls between graph launches break the CPU-runs-ahead invariant (host work always blocks). The two regimes are:

- **CUDA-Graph regime** ($\rho_{\mathrm{seq}} = 1$). When $t_{\text{step,seq}}(B) \le t_{\text{step,base}}(B)$, the overflow is 0 and $t_{\text{step,user}}(B) = t_{\text{step,base}}(B)$ — host work is fully hidden. When $t_{\text{step,seq}}(B) > t_{\text{step,base}}(B)$, only the excess blocks the next step's hardware window.
- **Eager regime** ($\rho_{\mathrm{seq}} = 0$). The overflow becomes the full $t_{\text{step,seq}}(B)$ — the Python interpreter cannot use the GPU compute window for the next step's setup, so host work serializes after GPU compute.

The serving term sits **outside** the pipeline-bubble multiplier $\gamma_{\text{pp}}$ since it fires once per step on the head node regardless of how the body is pipelined across stages — the bubble factor is already absorbed inside $t_{\text{step,base}}(B)$ from §7.2. This composition is identical in form to $t_{\text{stage,kernel}}$'s composition via $\rho_{\mathrm{kernel}}$ (§7.1); the two terms model different host-side work (per-microbatch dispatch vs per-sequence runtime) but admit the same overlap physics.

When $c_{\mathrm{seq}} = 0$, $t_{\text{step,user}}(B) = t_{\text{step,base}}(B)$ — the user-observed step time collapses to the hardware-only roofline derived in §7.2. When $t_{\text{stage,kernel}} = 0$ as well (host-overhead-free), the formula reduces all the way to $t_{\text{stage,hw}}(B) + t_{\text{LM,hw}}(B)$ at $\gamma_{\text{pp}} = 1$.

**Stack-dependent calibration.** $c_{\mathrm{seq}}$ captures the constant factor in the per-sequence inner loop, which differs across serving frameworks (Python-heavy vs C++/CUDA-Graph-heavy), runtime configurations (eager mode vs CUDA Graphs, Python sampling vs fused-CUDA sampling), and paged-attention block sizes. Reported empirical ranges:

| Stack | $c_{\mathrm{seq}}$ range | Notes |
|---|---|---|
| **C++/CUDA-Graph runtime under a serving orchestrator** (e.g. NVIDIA Dynamo wrapping TensorRT-LLM) | 5–22 µs/seq | The orchestrator absorbs most per-step bookkeeping into a single CUDA-Graph launch, leaving only the irreducible per-sequence sampling and block-table work. |
| **Mixed orchestrator + Python-internals runtime** (e.g. NVIDIA Dynamo wrapping SGLang) | 25–50 µs/seq | The orchestrator helps but the underlying runtime's Python-heavy paths still dominate. |
| **Raw C++ runtime, no orchestrator** (e.g. raw TensorRT-LLM) | 50–100 µs/seq | More individual kernel launches per step. Consistent across HW generations (Hopper / Blackwell) — $c_{\mathrm{seq}}$ is primarily a property of the software stack, not the GPU. |
| **Aggressively fused C++ stacks** (with fused sampling kernel) | ~10 µs/seq lower bound | Achievable when penalty application + multinomial draw + token-append fuse into a single CUDA invocation per batch. |
| **Python-heavy stacks** (eager-mode interpreters, e.g. vLLM, SGLang in eager mode) | 30–60 µs/seq | Dominated by the Python interpreter wrapping the per-sequence sampling decision; CUDA-Graph replay does not help because the CPU work is **between** graph launches. |

The software-stack axis dominates the chip axis: the same chip + same model under different stacks differs by 4–5× in $c_{\mathrm{seq}}$, while the same stack on different chips differs by less than 2×. The host CPU class and PCIe topology contribute weakly through how much of the bookkeeping the GPU command processor (vs the host CPU) absorbs.

> **Speculative-decoding extension.** The TPOT identity $\text{TPOT}(B) = t_{\text{step,user}}(B)$ holds for vanilla decode (one accepted token per sequence per step). Under speculative decoding (MTP / EAGLE / Medusa) each step costs more (the verify pass evaluates $n_{\text{tok,verify}}$ tokens per sequence) but emits more output ($N_{\text{tok/step}}$ accepted tokens per sequence on average). The effective TPOT is $t_{\text{step,user}}^{\text{verify}}(B) / N_{\text{tok/step}}$ — derivation and regime analysis in §8.

### Throughput (TPS, TTPS)

A single DP replica emits one token per sequence per step, so it outputs $B$ tokens every $t_{\text{step,user}}(B)$:

$$
TPS_{\text{single}}(B) = \frac{B}{t_{\text{step,user}}(B)}
$$

Across $DP$ fully independent replicas (no cross-replica coupling), the total cluster throughput scales linearly:

$$
TTPS(B) = DP \cdot TPS_{\text{single}}(B) = \frac{DP \cdot B}{t_{\text{step,user}}(B)}
$$

When $B \ge PP$ the bubble factor is unity and $TPS_{\text{single}}(B) = B / t_{\text{step,user}}(B)$ collapses to throughput gated by the slowest stage — the bottleneck stage is $PP{-}1$ and its cost is the kernel-launch-composed body plus the once-per-step LM head $t_{\text{LM,hw}}(B)$, augmented by any unhidden serving overflow from this section.

---

# 8. Speculative Decoding (MTP / EAGLE / Medusa)

Sections 1–7 model the **vanilla decode** step: every active sequence advances by exactly one new token per step. **Speculative decoding** breaks this 1:1 mapping by having a cheap draft model propose multiple candidate tokens per sequence per step, then having the target model verify all of them in a single parallel pass — accepting a prefix of the draft based on per-token logit comparison and falling back to standard sampling for the first rejected position [LEVIATHAN]. The accepted tokens are emitted as the step's output. Because verification runs the target model **once per step regardless of how many tokens are accepted**, the per-step latency is bounded above by the target model's verify cost while the per-step output count can exceed one — yielding a TPOT speedup whenever the verification cost stays comparable to vanilla decode and acceptance is non-trivial.

DeepSeek-V3 ships a built-in Multi-Token Prediction (MTP) head trained jointly with the base model that drafts $n_{\text{tok,draft}}$ additional tokens per step from shared base activations [DSV3]. EAGLE [EAGLE] and Medusa [MEDUSA] are alternative draft architectures (feature-level autoregressive draft and parallel decoding heads, respectively); from the cost model's perspective they differ only in the per-token acceptance probability $p_{\text{accept}}$ and the draft chain length $n_{\text{tok,draft}}$. The two parameters that fully determine the speculative-decoding TPOT model are:

- $n_{\text{tok,draft}}$ — number of draft tokens proposed per verify step (typical 3–5; 1 disables speculation).
- $p_{\text{accept}}$ — per-token draft acceptance probability ∈ [0, 1] (typical 0.6–0.85; calibrated against the deployment).

A separate per-token verify-cost calibration constant is **not** introduced — the verify-step cost is derived from the roofline below rather than fitted.

## 8.1 Verify-step setup

A verify step processes $n_{\text{tok,verify}} = n_{\text{tok,draft}} + 1$ tokens per active sequence (the $n_{\text{tok,draft}}$ proposed tokens plus the target model's own next-token prediction, which is always accepted). For $B$ active sequences the verify step's effective token volume is $B \cdot n_{\text{tok,verify}}$ per stage. The verify step:

1. **Runs the target model forward** on $B \cdot n_{\text{tok,verify}}$ query tokens, attending to each sequence's existing $S$-token KV cache plus the draft tokens themselves (the FlashAttention kernel batches the draft queries into a single attention pass per sequence per layer).
2. **Compares draft logits to target logits** at each draft position; accepts the longest matching prefix; samples a corrective token from the distribution-difference at the first mismatch.
3. **Emits between 1 and $n_{\text{tok,verify}}$ accepted tokens per sequence** and appends their KV entries to the cache.

The draft model itself runs as a forward pass on the same accelerator; for MTP the draft is the target model's own MTP head and the draft cost is a small fixed surcharge (an extra projection per draft position from shared base activations), absorbed into the verify-step compute below. For EAGLE / Medusa with a separate draft model, the draft pass adds its own roofline cost — see §8.5.

## 8.2 Acceptance model and expected accepted tokens

Under independent per-token acceptance with success probability $p_{\text{accept}}$, the number of accepted draft tokens follows a truncated geometric distribution: the chain accepts position 1 with probability $p_{\text{accept}}$, positions 1–2 with probability $p_{\text{accept}}^2$, ..., positions 1–$n_{\text{tok,draft}}$ with probability $p_{\text{accept}}^{n_{\text{tok,draft}}}$. The expected number of accepted draft tokens per step is the closed-form sum [LEVIATHAN, eq. 5]:

$$
\mathbb{E}[N_{\text{accept,draft}}] \;=\; \sum_{d=1}^{n_{\text{tok,draft}}} p_{\text{accept}}^d \;=\; \frac{p_{\text{accept}} \, (1 - p_{\text{accept}}^{n_{\text{tok,draft}}})}{1 - p_{\text{accept}}}
$$

The always-accepted target prediction adds one more token per step, giving the **total expected accepted tokens per verify step**:

$$
N_{\text{tok/step}} \;=\; 1 + \mathbb{E}[N_{\text{accept,draft}}] \;=\; 1 + \frac{p_{\text{accept}} \, (1 - p_{\text{accept}}^{n_{\text{tok,draft}}})}{1 - p_{\text{accept}}}
$$

For $(n_{\text{tok,draft}}, p_{\text{accept}}) = (4, 0.7)$: $N_{\text{tok/step}} \approx 1 + 0.7 + 0.49 + 0.343 + 0.240 \approx 2.78$.

The bound $1 \le N_{\text{tok/step}} \le n_{\text{tok,verify}}$ is tight: $p_{\text{accept}} = 0$ yields 1 (vanilla decode equivalent) and $p_{\text{accept}} = 1$ yields $n_{\text{tok,verify}}$ (every draft accepted). The independence assumption is empirically reasonable for MTP and EAGLE at moderate draft depths but breaks down at $n_{\text{tok,draft}} > 5$ as accepted-prefix correlations grow [EAGLE §4]; calibrate $p_{\text{accept}}$ against the deployment rather than treating it as an architectural constant.

## 8.3 Verify-step roofline

The verify step runs the same model as vanilla decode but with $n_{\text{tok,verify}}$ tokens per sequence instead of 1. Three of the per-step cost components scale with this multiplier; one does not:

- **Compute scales linearly with $n_{\text{tok,verify}}$.** Every Q/K/V/O projection, every attention score, every FFN GEMM does $n_{\text{tok,verify}}$ times more work per sequence: $F_{\text{token,device}}^{\text{verify}} = n_{\text{tok,verify}} \cdot F_{\text{token,device}}$.
- **Communication payload scales linearly with $n_{\text{tok,verify}}$.** Each per-layer TP collective (all-reduce under TP-attention, all-gather under DP-attention; see §5.3) carries $n_{\text{tok,verify}}$ times the activations: per-rank message size is $B \cdot n_{\text{tok,verify}} \cdot H b$ instead of $B \cdot Hb$. The MoE Dispatch + Combine all-to-alls scale identically.
- **Weight traffic is unchanged.** Weights load once per verify step, exactly as in vanilla decode: $T_{\theta,\text{device}}^{\text{verify}} = T_{\theta,\text{device}}$.
- **KV-cache read traffic is approximately unchanged.** Each sequence's existing $S$-token KV is read once per layer per step regardless of how many query tokens piggyback on the read (FlashAttention's standard batched-Q kernel pattern). The draft tokens add at most $n_{\text{tok,draft}}$ positions to the per-sequence KV scan, which is negligible at $S \gg n_{\text{tok,draft}}$ — typical decode contexts have $S$ in the thousands and $n_{\text{tok,draft}}$ ≤ 5: $T_{\text{KV,token}}^{\text{verify}} \approx T_{\text{KV,token}}$.

Threading these through the §6.2 / §7.3 composition, the verify-step compute time, memory time, and roofline are:

$$
t_{\text{compute}}^{\text{verify}}(B) \;=\; n_{\text{tok,verify}} \cdot t_{\text{compute}}(B), \qquad
t_{\text{mem}}^{\text{verify}}(B) \;\approx\; t_{\text{mem}}(B)
$$

$$
t_{\text{local}}^{\text{verify}}(B) \;=\; \max\bigl( n_{\text{tok,verify}} \cdot t_{\text{compute}}(B),\; t_{\text{mem}}(B) \bigr)
$$

The communication budget per stage scales by the same multiplier on every payload-bearing term:

$$
t_{\text{comm}}^{\text{verify}}(B) \;=\; \text{(\S5.5 expression with all per-layer message sizes scaled by } n_{\text{tok,verify}}\text{)}
$$

The α-side terms (per-collective startup latencies $(G_{EP} - 1)\alpha_{EP}$, $2(G_{TP} - 1)\alpha_{TP}$, etc.) are unchanged — one collective per layer fires regardless of payload size.

> **Composition with DP-attention.** When DP-attention mode (notation.md §1) and speculative decoding are both enabled — the DSv3 production configuration — the two extensions compose orthogonally: the §5.3 attention AR → AG swap fires on every verify step, and the per-rank AG message size scales by $n_{\text{tok,verify}}$ (the §5.3 AG expression with $M = B \cdot n_{\text{tok,verify}} \cdot Hb$). All other DP-attn deltas (attention weight footprint in §1.4, KV invariance, attention FLOPs invariance in §3.5) carry through unchanged because they are independent of the per-step token count.

The verify-step user-observed step time is then the §7.3 composition with the verify-step quantities substituted in. The per-sequence serving overhead $t_{\text{step,seq}}(B)$ from §7.2 fires once per verify step (not once per accepted token) and is added before the $1 / N_{\text{tok/step}}$ amortization derived below:

$$
t_{\text{step,user}}^{\text{verify}}(B) \;=\; \gamma_{\text{pp}} \cdot \bigl[ t_{\text{stage,hw}}^{\text{verify}}(B) + \max(0, t_{\text{stage,kernel}} - \rho_{\text{kernel}} \cdot t_{\text{stage,hw}}^{\text{verify}}(B)) \bigr] + t_{\text{LM,hw}}^{\text{verify}}(B) + t_{\text{step,seq}}(B)
$$

where the verify-step LM head follows the same roofline-of-compute-and-memory shape as elsewhere in the model — its compute scales linearly with $n_{\text{tok,verify}}$ (each verify-step query needs its own logits to compare against the draft) but the $V \times H$ projection matrix loads once per step regardless of query count:

$$
t_{\text{LM,hw}}^{\text{verify}}(B) \;=\; \max\!\bigl( n_{\text{tok,verify}} \cdot t_{\text{LM,compute}}(B),\; t_{\text{LM,mem}}(B) \bigr)
$$

For DSv3-class vocab and FP4 quantization the LM head sits at the memory-bound floor across the typical decode-batch range and $t_{\text{LM,hw}}^{\text{verify}}(B) \approx t_{\text{LM,hw}}(B)$; only at large $B$ does the compute term cross over and the linear $n_{\text{tok,verify}}$ scaling kick in.

## 8.4 Effective TPOT under speculative decoding

Each verify step costs $t_{\text{step,user}}^{\text{verify}}(B)$ wall-clock time and emits $N_{\text{tok/step}}$ accepted tokens per sequence on average. The effective per-output-token latency is:

$$
\mathrm{TPOT_{spec}}(B) \;=\; \frac{t_{\text{step,user}}^{\text{verify}}(B)}{N_{\text{tok/step}}}
$$

In the **memory-bound regime** ($n_{\text{tok,verify}} \cdot t_{\text{compute}}(B) < t_{\text{mem}}(B)$, common at small-to-moderate $B$ for MoE / MLA decode), the verify step is roofline-equivalent to the vanilla decode step ($t_{\text{local}}^{\text{verify}} \approx t_{\text{local}}$) and TPOT collapses to a clean speedup:

$$
\mathrm{TPOT_{spec}}(B) \;\approx\; \frac{\text{TPOT}(B)}{N_{\text{tok/step}}} \qquad \text{(memory-bound regime)}
$$

For DSv3 / R1 on GB200 NVL72 at typical decode batch sizes, this is the operative regime — measurements show MTP rows at 0.4–0.6× the baseline TPOT, consistent with $N_{\text{tok/step}} \approx 1.7$–$2.5$.

In the **compute-bound regime** ($n_{\text{tok,verify}} \cdot t_{\text{compute}}(B) > t_{\text{mem}}(B)$, large $B$), the verify step grows by the full $n_{\text{tok,verify}}$ multiplier and the speedup ratio becomes:

$$
\frac{\mathrm{TPOT_{spec}}(B)}{\text{TPOT}(B)} \;\approx\; \frac{n_{\text{tok,verify}}}{N_{\text{tok/step}}} \qquad \text{(compute-bound regime)}
$$

This ratio is $\ge 1$ because $N_{\text{tok/step}} \le n_{\text{tok,verify}}$ — speculative decoding **does not help** in the strict compute-bound regime and ties at $p_{\text{accept}} = 1$ (every draft accepted, no wasted compute).

The compute-bound crossover under speculation shifts down by $\approx n_{\text{tok,verify}}$ relative to vanilla decode (§4): the verify step hits compute-bound at:

$$
B^\star_{\text{spec}} \;\approx\; \frac{T_{\theta,\text{device}} \cdot R_{\text{GPU}}}{n_{\text{tok,verify}} \cdot F_{\text{token,device}} \cdot BW_{\text{mem}} - T_{\text{KV,token}} \cdot R_{\text{GPU}}}
$$

For typical MoE decode where $B^\star$ from §4 is in the hundreds-to-thousands range, $B^\star_{\text{spec}}$ at $n_{\text{tok,verify}} = 5$ falls to the tens-to-hundreds range — speculation buys back compute headroom that the vanilla decode roofline left on the table.

## 8.5 Where speculative decoding wins and loses

The TPOT-vs-batch curve under speculation has three operating regimes:

1. **Memory-bound win** ($B \ll B^\star_{\text{spec}}$): $\mathrm{TPOT_{spec}} \approx \text{TPOT} / N_{\text{tok/step}}$. Speedup ≈ $N_{\text{tok/step}}$ (typically 1.7–3×). This is the regime where MTP / EAGLE / Medusa pay off and where production deployments operate.
2. **Crossover band** ($B \approx B^\star_{\text{spec}}$): partial speedup; the verify step is starting to be compute-limited but $N_{\text{tok/step}}$ amortization still wins. Calibrate $p_{\text{accept}}$ to the deployment to predict the exact crossover.
3. **Compute-bound break-even or loss** ($B \gg B^\star_{\text{spec}}$): $\mathrm{TPOT_{spec}} \approx (n_{\text{tok,verify}} / N_{\text{tok/step}}) \cdot \text{TPOT}$. The ratio is $\ge 1$ — at best a tie ($p_{\text{accept}} \to 1$), at worst a slowdown of up to $n_{\text{tok,verify}}$ at $p_{\text{accept}} \to 0$. The framework's speculation-on default should be disabled or $n_{\text{tok,draft}}$ reduced when sweeping into this regime.

**Separate draft model (EAGLE / Medusa, not MTP).** The above derivation folds the draft cost into the verify pass under the assumption that drafts come from the target model itself (MTP). For a separate draft model whose forward cost is a non-trivial fraction of the verify step, add the draft latency as a serial term: $t_{\text{step}}^{\text{spec,sep}}(B) = t_{\text{draft}}(B) + t_{\text{step,user}}^{\text{verify}}(B)$. EAGLE's draft is $\approx$10% of the target verify cost; Medusa's parallel heads add a small constant per head. We omit a detailed draft-model roofline here — the cost is workload-specific and best calibrated rather than derived.

**Acceptance-rate calibration.** $p_{\text{accept}}$ is **not** a per-architecture constant — it depends on the model, the draft method, the temperature, the corpus, and the position in the sequence. Reported empirical values at temperature 0: MTP on DSv3 ≈ 0.85–0.90 per-token at draft depth 1 [DSV3]; EAGLE on Llama-3 70B ≈ 0.75–0.85 at depth 4 [EAGLE]; Medusa-2 on Vicuna-13B ≈ 0.6–0.7 across heads [MEDUSA]. The framework treats $p_{\text{accept}}$ as a tuning knob to be measured on the target deployment.
