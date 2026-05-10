# Attention Variants

**Author:** Yue Lu  
**Date:** May 2026  

This document covers attention architectures that depart from the standard multi-head attention (MHA) / grouped-query attention (GQA) baseline assumed in `decode.md` and `prefill.md`. When a model uses one of the variants below, the framework's per-device parameter, key-value (KV) cache, traffic, and floating-point operation (FLOP) formulas need a variant-aware substitution for the attention block; the rest of the layer (feed-forward network (FFN), Mixture-of-Experts (MoE), collective communication) is unchanged.

**Scope.** Each variant section gives (a) an architectural overview, (b) a symbol register specific to that variant, (c) per-layer parameter count, (d) KV cache footprint per token per layer, (e) per-token compute, and (f) sharding behavior under tensor parallelism (TP) attention and data parallelism (DP) attention. The decode and prefill cost formulas in `decode.md` / `prefill.md` carry inline cross-references to the matching subsection here for each affected term.

**Variants in this document:**

- §1 Multi-head Latent Attention (MLA) — DeepSeek-V3 / R1, DeepSeek-V4-Pro, GLM-5, Kimi-K2.5
- §2 Sliding-window attention (placeholder) — Mistral, GPT-OSS, Gemma
- §3 DeepSeek Sparse Attention (DSA) (placeholder) — DeepSeek-V4-Pro, GLM-5
- §4 Hybrid linear / full attention (placeholder) — Qwen-3.5, Jamba, Hymba

---

## 1. Multi-head Latent Attention (MLA)

### 1.1 Architectural overview

MLA replaces the standard per-head Q / K / V projections with two compressed paths: a query latent and a jointly-compressed KV latent [DSV3]. Per token:

- **Query path.** The hidden state $h \in \mathbb{R}^H$ is first down-projected to a query latent $c_Q \in \mathbb{R}^{d_{q,c}}$ via $W_{DQ}$, then up-projected to per-head queries via $W_{UQ}$. Each per-head query has two parts: a non-positional component $q_i^{\mathrm{nope}} \in \mathbb{R}^{d_{qk,\mathrm{nope}}}$ and a rotary-position-embedded (RoPE) component $q_i^{\mathrm{rope}} \in \mathbb{R}^{d_{qk,\mathrm{rope}}}$.
- **KV path.** The hidden state is jointly down-projected to a KV latent $c_{KV} \in \mathbb{R}^{d_c + d_{qk,\mathrm{rope}}}$ via $W_{DKV}$. The first $d_c$ dimensions form a head-shared K / V latent; the trailing $d_{qk,\mathrm{rope}}$ dimensions form the RoPE-positional K, also shared across all heads.
- **Per-head reconstruction.** Non-positional keys $k_i^{\mathrm{nope}}$ are reconstructed on demand as $W_{UK,i} \cdot c_{KV}[:d_c]$; per-head values $v_i$ as $W_{UV,i} \cdot c_{KV}[:d_c]$. The RoPE part is shared, not per-head.
- **Attention.** Computed per head against $(k_i^{\mathrm{nope}}, c_{KV}[d_c:])$ and $v_i$, then per-head outputs are projected back to the hidden dimension via $W_O$.

The compression has two benefits relative to standard MHA. First, the KV cache stores only the latent $c_{KV}$ — $(d_c + d_{qk,\mathrm{rope}})$ per token per layer — instead of the full per-head K and V at $2 \cdot n_h \cdot d_{qk}$ bytes per token per layer. Second, the per-step latent space is compact enough that attention can run entirely in the $d_c$-dimensional space without materializing the full per-head K and V each step (see §1.5 absorbed mode).

### 1.2 Symbol register

| Symbol | Description | Typical DSv3 value |
|--------|-------------|-------|
| $d_c$ | KV latent dimension (head-shared) | 512 |
| $d_{q,c}$ | Query latent dimension | 1536 |
| $d_{qk,\mathrm{nope}}$ | Non-positional Q / K head dimension | 128 |
| $d_{qk,\mathrm{rope}}$ | RoPE-positional Q / K head dimension (head-shared on K side) | 64 |
| $d_v$ | Value head dimension | 128 |
| $n_h$ | Number of attention heads (same as `notation.md §3`) | 128 |

Composite shorthand: $d_{qk} = d_{qk,\mathrm{nope}} + d_{qk,\mathrm{rope}}$ — total per-head Q / K dimension on the query side.

The MHA / GQA symbols $H$, $H_{kv} = n_{kv} \cdot d_{\mathrm{head}}$, $d_{\mathrm{head}} = H / n_h$ from `decode.md` and `notation.md §3` are not used in MLA accounting; the relationship $H = n_h \cdot d_{\mathrm{head}}$ does not constrain MLA dimensions, which are independent design choices.

### 1.3 Per-layer attention parameter count

The MHA / GQA per-layer attention parameter count from `decode.md §1.1`:

$$P_{\mathrm{attn,MHA}} = 2 H^2 + 2 H \cdot H_{kv}$$

The MLA per-layer attention parameter count is the sum of the six weight matrices:

$$P_{\mathrm{attn,MLA}} = \underbrace{H \cdot d_{q,c}}_{W_{DQ}} + \underbrace{d_{q,c} \cdot n_h \cdot d_{qk}}_{W_{UQ}} + \underbrace{H \cdot (d_c + d_{qk,\mathrm{rope}})}_{W_{DKV}} + \underbrace{n_h \cdot d_c \cdot d_{qk,\mathrm{nope}}}_{W_{UK}} + \underbrace{n_h \cdot d_c \cdot d_v}_{W_{UV}} + \underbrace{n_h \cdot d_v \cdot H}_{W_O}$$

In bytes: multiply by $b$ (bytes per parameter, `notation.md §4`).

The two large terms for typical deployments are $W_{UQ}$ (scales with $n_h \cdot d_{qk}$ on the inside dimension) and $W_O$ (scales with $n_h \cdot d_v \cdot H$). The four smaller terms ($W_{DQ}$, $W_{DKV}$, $W_{UK}$, $W_{UV}$) are roughly an order of magnitude smaller in DSv3-class configurations; see the worked example in §1.8.

### 1.4 KV cache footprint

Per token per layer, the MHA / GQA form from `decode.md §1.3` is $M_{\mathrm{KV,MHA}} = 2 \cdot H_{kv} \cdot b$.

The MLA per-token-per-layer KV cache stores only the joint latent:

$$M_{\mathrm{KV,MLA}} = (d_c + d_{qk,\mathrm{rope}}) \cdot b$$

This is much smaller than MHA / GQA on equivalent models. A worked DSv3 comparison: MLA stores $(512 + 64) \cdot 0.5 = 288$ bytes per token per layer at FP4 (`notation.md §4`); equivalent dense MHA at $H_{kv} = H = 7168$ would store $2 \cdot 7168 \cdot 0.5 = 7168$ bytes per token per layer — about 25× larger. The framework's prior $n_{kv}=5$ approximation in some MLA model specs (giving $H_{kv} = 280$) under-counts MLA's real KV by ~3% but mis-attributes the ~75% under-count of attention parameters into the $H_{kv}$ field; §1.8 quantifies the residual error this causes for total per-device weight footprint.

### 1.5 Two execution modes

MLA admits two equivalent ways of computing attention per step. Both consume the same KV cache content $(c_{KV}[:d_c], c_{KV}[d_c:])$ and produce the same output — they differ in where the $W_{UK}$ and $W_{UV}$ multiplications happen.

#### Materialized mode

For each step, decompress the latent into per-head K and V, then run standard attention:

- Read $c_{KV}$ from the KV cache for all $S$ past tokens (KV cache traffic per token per layer = $(d_c + d_{qk,\mathrm{rope}}) \cdot b$).
- Compute $K_i = W_{UK,i} \cdot c_{KV}[:d_c]$ for each head $i$ and each past token (one-time per token added to the cache; cached in fast on-die memory if possible).
- Compute $V_i = W_{UV,i} \cdot c_{KV}[:d_c]$ similarly.
- Run standard multi-head attention on $(K_i, c_{KV}[d_c:])$ and $V_i$.

The materialization step costs $2 \cdot n_h \cdot d_c \cdot (d_{qk,\mathrm{nope}} + d_v)$ FLOPs per token-added-to-cache per layer (for MLA-class decode this is once per step per layer, since one new token enters the cache per step). It also requires a transient per-head K / V buffer of size $n_h \cdot S \cdot (d_{qk,\mathrm{nope}} + d_v) \cdot b$ in fast memory.

#### Absorbed mode (production default)

Fold the up-projections into Q and O at compile time so attention runs entirely in the $d_c$-dimensional latent space:

- Precompute $W_{UQ} \otimes W_{UK}$ so the query is projected directly to a form that can dot-product against $c_{KV}[:d_c]$ — no per-step K reconstruction.
- Precompute $W_{UV} \otimes W_O$ so the output projection consumes a weighted sum of $c_{KV}[:d_c]$ entries — no per-step V reconstruction.

Effectively the per-step attention is:

$$\mathrm{score}_i = q'_i \cdot c_{KV}[:d_c]^T + q_i^{\mathrm{rope}} \cdot c_{KV}[d_c:]^T$$

$$\mathrm{output}_i = \mathrm{softmax}(\mathrm{score}_i) \cdot c_{KV}[:d_c]$$

followed by the absorbed output projection. The compute is dominated by two per-past-token multiplications in $d_c$ space (score and value sum), each $O(n_h \cdot d_c)$ per past token per layer, instead of the materialized form's $O(n_h \cdot d_{qk,\mathrm{nope}} + n_h \cdot d_v)$ per past token. For DSv3 with $d_c = 512$ and $d_{qk,\mathrm{nope}} = d_v = 128$, the absorbed score / value cost per past token is $\sim 4 \times$ the materialized form's score / value cost ($2 \cdot 128 \cdot 512 = 131{,}072$ vs $128 \cdot (128 + 128) = 32{,}768$ FLOPs per past token per layer) — but the absorbed form skips the per-token K and V materialization entirely and avoids the transient per-head K / V buffer. The DSv3 paper itself derives the canonical decode-time forward pass without explicitly naming the two modes; the materialized / absorbed distinction is a production-framework implementation choice. Production frameworks (NVIDIA TensorRT-LLM, SGLang's DeepSeek-V3 path) use absorbed mode by default for decode; materialized mode appears in some reference and CPU-fallback implementations.

The exact per-mode FLOP breakdown is given in §1.6.

### 1.6 Per-token compute

Per-layer attention FLOPs per decode step (one new token, $S$ past tokens in cache):

$$F_{\mathrm{attn,MLA,materialized}}(S) = \underbrace{2 H \cdot d_{q,c}}_{W_{DQ}} + \underbrace{2 \cdot d_{q,c} \cdot n_h \cdot d_{qk}}_{W_{UQ}} + \underbrace{2 H \cdot (d_c + d_{qk,\mathrm{rope}})}_{W_{DKV}} + \underbrace{2 \cdot n_h \cdot d_c \cdot d_{qk,\mathrm{nope}}}_{W_{UK} \text{ on new token}} + \underbrace{2 \cdot n_h \cdot d_c \cdot d_v}_{W_{UV} \text{ on new token}} + \underbrace{2 S \cdot n_h \cdot d_{qk}}_{Q \cdot K^T} + \underbrace{2 S \cdot n_h \cdot d_v}_{\mathrm{softmax} \cdot V} + \underbrace{2 \cdot n_h \cdot d_v \cdot H}_{W_O}$$

$$F_{\mathrm{attn,MLA,absorbed}}(S) = \underbrace{2 H \cdot d_{q,c}}_{W_{DQ}} + \underbrace{2 \cdot d_{q,c} \cdot n_h \cdot d_{qk}}_{W_{UQ}} + \underbrace{2 H \cdot (d_c + d_{qk,\mathrm{rope}})}_{W_{DKV}} + \underbrace{2 S \cdot n_h \cdot (d_c + d_{qk,\mathrm{rope}})}_{\mathrm{score in latent}} + \underbrace{2 S \cdot n_h \cdot d_c}_{\mathrm{value in latent}} + \underbrace{2 \cdot n_h \cdot d_c \cdot H}_{\text{absorbed } W_O}$$

The structure of the difference: materialized pays a fixed (S-independent) cost per step to reconstruct K and V for the new token, then standard attention scales with $S$ in the smaller $d_{qk,\mathrm{nope}}$ and $d_v$ dimensions. Absorbed pays no per-step reconstruction but the per-past-token attention scales with the larger $d_c$ dimension. Crossover happens at moderate $S$; production deployments at $S \gtrsim 1\mathrm{K}$ generally prefer absorbed because the $S$-scaling savings on the per-token reconstruction exceed the $d_c$-vs-$d_{qk,\mathrm{nope}}$ overhead on the score / value compute.

For prefill, where the input has $S_{\mathrm{input}}$ tokens entering the cache simultaneously, the materialization cost amortizes over the full input: the per-token cost is the same $(W_{UK}, W_{UV})$ work but consolidated into a single GEMM. Per-token compute differences between modes are smaller in prefill than in decode.

### 1.7 Sharding under TP-attention / DP-attention

Under TP-attention (`attention_mode="tp"`, head-sharded), the MLA weights split as follows across the $G_{TP}$ tensor-parallel ranks:

- $W_{UQ}$, $W_{UK}$, $W_{UV}$, $W_O$ are head-sharded — each rank gets $n_h / G_{TP}$ heads' worth of these weights.
- $W_{DQ}$, $W_{DKV}$ are not sharded — they project from / to the hidden dimension $H$ and are replicated on every rank.
- KV cache footprint per rank: the latent $c_{KV}$ is computed once per rank via $W_{DKV}$ and stored full (no head-sharding — the latent is shared across all heads by construction).

The per-rank attention parameter footprint is therefore:

$$P_{\mathrm{attn,device}} = \frac{1}{G_{TP}} \cdot (W_{UQ} + W_{UK} + W_{UV} + W_O) + W_{DQ} + W_{DKV}$$

Replacing the per-device attention parameter term in `decode.md §1.4` for MLA models.

Under DP-attention (`attention_mode="dp"`, batch-sharded), all MLA weights are replicated on every rank ($D_{\mathrm{attn}} = 1$ per `notation.md §1`), and the batch is split across the $G_{TP}$ attention DP groups. Per-rank KV cache footprint scales with the per-rank token count $B / G_{TP}$ instead of the full batch:

$$M_{\mathrm{KV,device}} = \frac{B}{G_{TP}} \cdot S \cdot (d_c + d_{qk,\mathrm{rope}}) \cdot b \cdot \frac{L}{PP}$$

(per-stage form; the $L / PP$ factor and the rest of the outer composition are unchanged from `decode.md §1.4` and §2.3).

### 1.8 Worked example — DSv3 / DSR1

DeepSeek-V3 / R1 architecture: $H = 7168$, $n_h = 128$, $d_c = 512$, $d_{q,c} = 1536$, $d_{qk,\mathrm{nope}} = 128$, $d_{qk,\mathrm{rope}} = 64$, $d_v = 128$, $L = 61$ (60 MoE + 1 dense), bytes per parameter $b$ depends on quantization (FP4 → 0.5, FP8 → 1, BF16 → 2).

Per-layer attention parameter count:

| Matrix | Term | Value |
|--------|------|------:|
| $W_{DQ}$ | $H \cdot d_{q,c}$ | 11.0 M |
| $W_{UQ}$ | $d_{q,c} \cdot n_h \cdot (d_{qk,\mathrm{nope}} + d_{qk,\mathrm{rope}})$ | 37.7 M |
| $W_{DKV}$ | $H \cdot (d_c + d_{qk,\mathrm{rope}})$ | 4.1 M |
| $W_{UK}$ | $n_h \cdot d_c \cdot d_{qk,\mathrm{nope}}$ | 8.4 M |
| $W_{UV}$ | $n_h \cdot d_c \cdot d_v$ | 8.4 M |
| $W_O$ | $n_h \cdot d_v \cdot H$ | 117.4 M |
| **Total** | | **187.0 M** |

The MHA-equivalent approximation $P_{\mathrm{attn,MHA}} = 2 H^2 + 2 H \cdot H_{kv}$ with $H_{kv} = 5 \cdot 56 = 280$ (the $n_{kv} = 5$ approximation used in some legacy model specs) gives 102.8 M + 4.0 M = 106.8 M per layer — under-counting the real MLA attention parameter count by 80.2 M per layer (~43% relative under-count on the attention block alone).

Across all 61 layers: under-count is $\sim 4.9$ B parameters out of DSv3's 671 B total — about 0.7% of total weight footprint. The KV cache footprint, by contrast, is approximately right under the $n_{kv} = 5$ approximation (288 bytes per token per layer real MLA vs 280 bytes approximation = ~3% under). For decode workloads where KV cache traffic dominates, the approximation is acceptable; for prefill / TTFT predictions and per-device static memory accounting, the real MLA accounting is meaningfully more accurate.

---

## 2. Sliding-window attention (placeholder)

When the attention block is capped at a per-token sliding window of $W$ past tokens (Mistral 7B, GPT-OSS, Gemma), the KV cache caps at $W$ tokens per layer per sequence (with rolling eviction) and the attention compute scales with $W$ instead of the full context length $S$. To be filled in when the framework lands sliding-window support.

---

## 3. DeepSeek Sparse Attention (DSA) (placeholder)

DSA introduces a top-k token selector before attention so each query attends to only the top $k$ most-relevant past tokens (DeepSeek-V4-Pro, GLM-5). The KV cache is unchanged but the per-step attention compute scales with $k$ instead of $S$ for the score / value stage. To be filled in when the framework lands DSA support.

---

## 4. Hybrid linear / full attention (placeholder)

Some model classes interleave linear-attention layers (Mamba / RWKV / RetNet style) with full-attention layers (Qwen-3.5, Jamba, Hymba). Linear-attention layers replace softmax attention with a cumulative state update that does not store a per-token KV cache; per-layer cost is $O(d \cdot d_{\mathrm{state}})$ regardless of $S$. To be filled in when the framework lands a per-layer layer-type field.
