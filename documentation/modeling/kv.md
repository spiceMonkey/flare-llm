# KV Cache Management: Paging, Fragmentation, and Traffic Overhead

**Author:** Yue Lu  
**Date:** April 2026

**Keywords:**  
KV cache, PagedAttention, block allocation, fragmentation, HBM capacity, memory management, LLM inference, paged memory, beam search, copy-on-write, sequence parallelism

---

## Abstract

The baseline KV cache model in `decode.md` §1.3 and §2.3 assumes contiguous allocation: keys and values for all $S$ token positions are stored and accessed as a single dense tensor per layer. Real production serving systems, however, use **paged KV allocation** (PagedAttention [VLLM]) to enable dynamic memory sharing, efficient preemption, and fine-grained capacity management. Paging introduces **internal fragmentation** — the last block of each sequence is typically partially filled — and additional traffic overhead for block-table lookups, copy-on-write branching, and potential preemption events. This document quantifies those overheads analytically, derives a fragmentation-corrected HBM capacity model, and shows how paging interacts with TP, SP, and PP sharding.

---

## Table of Contents

- [1. Baseline KV Model (from decode.md)](#1-baseline-kv-model-from-decodemd)
- [2. PagedAttention Block Structure](#2-pagedattention-block-structure)
- [3. Fragmentation Factor](#3-fragmentation-factor)
  - [3.1 Internal Fragmentation](#31-internal-fragmentation)
  - [3.2 External Fragmentation](#32-external-fragmentation)
- [4. Block Allocation Traffic Overhead](#4-block-allocation-traffic-overhead)
  - [4.1 Block Table Lookup](#41-block-table-lookup)
  - [4.2 Copy-on-Write for Beam Search and Parallel Sampling](#42-copy-on-write-for-beam-search-and-parallel-sampling)
  - [4.3 Preemption and Recompute](#43-preemption-and-recompute)
- [5. Interaction with TP and SP Sharding](#5-interaction-with-tp-and-sp-sharding)
- [6. Effective HBM Capacity Model](#6-effective-hbm-capacity-model)

---

> **Scope:** This document models standard GQA/MQA KV cache with $H_{kv} = n_{kv} \cdot d_{\text{head}}$ per token per layer. It does **not** model Multi-head Latent Attention (MLA) as used in DeepSeek-V2/V3, where a low-rank latent vector of dimension $d_c \ll H_{kv}$ is stored per token instead. MLA's KV cache volume is computed differently and would require a separate model section.

## 1. Baseline KV Model (from decode.md)

This section briefly restates the contiguous KV model established in `decode.md` §1.3 and §2.3 as the reference point. The derivations are not repeated here; refer to those sections for the full treatment.

### Per-layer KV memory (§1.3)

For a single attention layer with a context of $S$ tokens and KV projection dimension $H_{kv}$ (with $H_{kv} = n_{kv} \cdot d_{\text{head}}$, per [GQA]):

$$
M_{\text{KV,layer}} = 2 \, S \, H_{kv} \, b
$$

where the factor of 2 accounts for both keys and values, and $b$ is bytes per element.

### Per-device KV memory after sharding (§1.4)

Under pipeline parallelism (PP), tensor parallelism (TP), and sequence parallelism (SP), each device owns $L/PP$ layers, $1/TP$ of the head-channel dimension, and $1/SP$ of the sequence dimension:

$$
M_{\text{KV,device}} = \frac{L}{PP} \cdot \frac{2 \, S \, H_{kv} \, b}{TP \cdot SP}
$$

### Per-device KV traffic per token (§2.3)

Each new generated token requires reading the full KV history. Per device:

$$
T_{\text{KV,device}} \approx \frac{L}{PP} \cdot \frac{2 \, S \, H_{kv} \, b}{TP \cdot SP}
$$

FlashAttention does not reduce this term: history keys and values must always be loaded regardless of tiling strategy.

**Assumption in the baseline:** both expressions above treat KV storage as a contiguous dense tensor. The remainder of this document models the overhead introduced when that assumption is relaxed by paged allocation.

---

## 2. PagedAttention Block Structure

PagedAttention [VLLM] divides the KV cache into fixed-size **blocks**, each storing KV entries for $\text{BLK}_{KV}$ consecutive token positions. Blocks are allocated from a shared pool and may be physically non-contiguous; a per-sequence **block table** maps logical block indices to physical addresses.

### Block size and per-block storage

Each block stores keys and values for $\text{BLK}_{KV}$ token positions across all $L/PP$ layers on the device, with the head-channel dimension sharded by TP and SP. The per-block KV footprint is:

$$
M_{\text{block}} = \text{BLK}_{KV} \cdot \frac{2 \, H_{kv} \, b \cdot (L/PP)}{TP \cdot SP}
$$

This is the granularity at which memory is allocated and freed.

> **Convention note:** This document defines a KV block at the device level, spanning all $L/PP$ layers on the device. vLLM's implementation allocates blocks per-layer (each block stores $\text{BLK}_{KV}$ tokens for one layer). The two conventions yield identical total memory; the device-level block simplifies the fragmentation and capacity analysis below.

### Blocks required per sequence

A logical sequence of $S$ tokens requires:

$$
N_{\text{blocks}}(S) = \left\lceil \frac{S}{\text{BLK}_{KV}} \right\rceil
$$

blocks. The **last block** is filled with $S \bmod \text{BLK}_{KV}$ tokens; if this is non-zero, the remainder of the block is **reserved but unused** — the source of internal fragmentation.

### Symbol definitions

| Symbol | Definition |
|--------|-----------|
| $\text{BLK}_{KV}$ | KV block size in tokens (typical: 16 or 32) |
| $N_{\text{blocks}}(S)$ | Blocks allocated for a sequence of length $S$: $\lceil S / \text{BLK}_{KV} \rceil$ |
| $\varphi$ | Fragmentation factor: ratio of allocated KV memory to ideally occupied KV memory |
| $M_{\text{HBM,KV,avail}}$ | HBM capacity available for KV storage after weights and activations |
| $S_{\max}$ | Maximum supportable context length given $M_{\text{HBM,KV,avail}}$ and $\varphi$ |

---

## 3. Fragmentation Factor

### 3.1 Internal Fragmentation

The last block of each sequence is on average half-full for a uniform distribution of sequence lengths relative to block size. More precisely, the expected number of wasted token slots per sequence is:

$$
\mathbb{E}[\text{wasted tokens}] = \frac{\text{BLK}_{KV} - 1}{2} \approx \frac{\text{BLK}_{KV}}{2}
$$

We define the **fragmentation factor** $\varphi$ as the ratio of allocated capacity to logically occupied capacity:

$$
\varphi(S) = \frac{N_{\text{blocks}}(S) \cdot \text{BLK}_{KV}}{S} = \frac{\lceil S / \text{BLK}_{KV} \rceil \cdot \text{BLK}_{KV}}{S}
$$

**Boundary behavior:**

- $S \gg \text{BLK}_{KV}$: $\varphi \to 1$ — fragmentation is negligible for long contexts.
- $S = \text{BLK}_{KV}$: $\varphi = 1$ — exactly one full block, no waste.
- $S = 1$: $\varphi = \text{BLK}_{KV}$ — worst case, one entire block reserved for a single token.

For **uniform** sequence-length distribution over $[1, S_{\max}]$, the expectation of $\varphi$ simplifies to a first-order approximation:

$$
\varphi_{\text{avg}} \approx 1 + \frac{\text{BLK}_{KV}}{2 S}
$$

This approximation holds when $S \gg \text{BLK}_{KV} / 2$, which is satisfied for all realistic serving workloads (e.g., $S \ge 512$, $\text{BLK}_{KV} \le 32$).

**Practical implication:** for $\text{BLK}_{KV} = 16$ and $S = 4096$, $\varphi_{\text{avg}} \approx 1 + 16/(2 \cdot 4096) \approx 1.002$, i.e., less than 0.2% overhead. For $S = 64$, $\varphi_{\text{avg}} \approx 1.125$, i.e., 12.5% overhead — a meaningful penalty at short contexts.

### 3.2 External Fragmentation

**External fragmentation** arises when freed blocks cannot be coalesced into large enough contiguous regions for new allocations. Because PagedAttention allocates and frees at fixed block granularity (blocks are interchangeable), **external fragmentation is structurally zero** for LLM serving: any free block can satisfy any allocation request. This is a deliberate design property of [VLLM].

The one exception is systems that require physically contiguous multi-block regions (e.g., for hardware DMA or non-paged attention kernels); this case is outside the scope of the analytical model here.

---

## 4. Block Allocation Traffic Overhead

The baseline KV traffic $T_{\text{KV,device}}$ from `decode.md` §2.3 accounts for reading and writing KV tensor values. Paged allocation introduces additional memory operations beyond this: block-table lookups, copy-on-write for branching, and preemption.

### 4.1 Block Table Lookup

Each attention kernel invocation must resolve the physical address of each block via the block table before accessing KV values. For a sequence of length $S$:

- Block table has $N_{\text{blocks}}(S) = \lceil S / \text{BLK}_{KV} \rceil$ entries, each a pointer (typically 4 or 8 bytes).
- Total block table traffic per sequence per layer: $\lceil S / \text{BLK}_{KV} \rceil \cdot p_{\text{ptr}}$ bytes, where $p_{\text{ptr}} \in \{4, 8\}$.

For $\text{BLK}_{KV} = 16$, $S = 4096$, and $p_{\text{ptr}} = 8$: block table traffic is $256 \times 8 = 2048$ bytes per layer, versus $2 \times 4096 \times 128 \times 2 \approx 2$ MB of KV values. The ratio is approximately $0.05\%$ — **negligible** for all practical configurations.

**Modeling decision:** block table lookup traffic is omitted from the analytical traffic model.

### 4.2 Copy-on-Write for Beam Search and Parallel Sampling

When a sequence **branches** (e.g., beam search with width $W$, or parallel sampling), PagedAttention defers physical copying using a copy-on-write (CoW) mechanism [VLLM]: shared blocks are reference-counted and copied only when a branch writes a new token that would modify a shared block.

For beam search with beam width $W$, the expected extra traffic per decode step from CoW copies is:

$$
T_{\text{CoW}} = (W - 1) \cdot M_{\text{block}}
$$

This follows because all $W$ beams share a common prefix; each branch eventually needs to write into its own physical block, triggering at most $W-1$ block copies per divergence point. PagedAttention [VLLM] performs CoW at block granularity — only the one block being written to at the divergence point needs copying, not the shared prefix history. $M_{\text{block}}$ is defined in §2:

$$
M_{\text{block}} = \text{BLK}_{KV} \cdot \frac{2 \, H_{kv} \, b \cdot (L/PP)}{TP \cdot SP}
$$

For **greedy decode** ($W = 1$) or **independent parallel sampling** (no shared prefix):

$$
T_{\text{CoW}} = 0
$$

**Analytical model:** for single-sequence greedy decode, CoW traffic is zero and $T_{\text{KV,device}}$ from the baseline model is exact. For beam search with width $W$, augment $T_{\text{KV,device}}$ by the block-level copy cost at each branch point:

$$
T_{\text{KV,device}}^{\text{paged}}
\approx
T_{\text{KV,device}} + (W - 1) \cdot M_{\text{block}}
$$

where $T_{\text{KV,device}}$ is the baseline contiguous-allocation traffic and $M_{\text{block}}$ is the per-device size of a single KV block (the unit of physical copy). This formula represents the actual block-level copy cost per divergence event; using the full per-device KV capacity $M_{\text{KV,device}}$ instead would overstate the cost by a factor of $S / \text{BLK}_{KV}$ (e.g., $256\times$ for $S=4096$, $\text{BLK}_{KV}=16$).

### 4.3 Preemption and Recompute

When the block pool is exhausted (e.g., under bursty load or a request with unexpectedly long context), PagedAttention can **preempt** a running sequence by swapping its KV blocks to CPU DRAM or by discarding them entirely and recomputing on re-schedule [VLLM]. Both strategies introduce substantial overhead:

- **Swap:** PCIe transfer at ~32–64 GB/s (vs. HBM at ~3 TB/s on H100), typically $50\times$ slower. The traffic is $M_{\text{KV,device}}$ in each direction.
- **Recompute:** equivalent to re-running the prefill phase for the preempted sequence, contributing $O(S^2)$ FLOPs.

These overheads are **workload-dependent** and not amenable to closed-form analytical treatment. They are treated as empirical correction terms in `framework.md`. The design implication is that $\varphi$ and $M_{\text{HBM,KV,avail}}$ should be sized conservatively to keep the block pool occupancy below ~85–90% in steady state, avoiding preemption entirely under expected workloads.

---

## 5. Interaction with TP and SP Sharding

### Tensor Parallelism (TP)

Each TP rank holds $H_{kv}/TP$ of the head-channel dimension. Block allocation is **TP-local**: each rank independently manages its own shard of each block's KV values. There is no cross-TP coordination required for paging; the block table on each TP rank contains physical addresses for that rank's local KV shard.

Block size $\text{BLK}_{KV}$ is defined in terms of token positions (not channel elements), so it is invariant to TP degree. The fragmentation factor $\varphi$ is likewise independent of TP.

> **Mode note (DP-attention).** Under DP-attention mode (`decode.md §8`) the head-sharded KV layout above is replaced by a sequence-sharded layout: each TP rank holds full-head KV for $1/TP$ of the active sequences instead of $1/TP$ heads of all sequences. Per-device KV bytes and per-device KV traffic are invariant under this swap (`decode.md §8.2`). The PagedAttention block accounting in this section applies to either layout — what changes is which sequences (vs which head-channels) live on a given device.

### Sequence Parallelism (SP)

Each SP rank holds $S/SP$ contiguous token positions from the KV sequence. Block allocation is **per-SP-rank**: each rank manages an independent block pool covering its $S/SP$ token positions, with its own block table.

**Alignment constraint:** to avoid cross-shard block waste, $\text{BLK}_{KV}$ should divide $S/SP$ evenly:

$$
\text{BLK}_{KV} \mid \frac{S}{SP}
$$

If this condition is not met, each SP rank independently incurs last-block fragmentation on the boundary of its $S/SP$ window, effectively multiplying the fragmentation overhead by SP for misaligned configurations.

**Recommended practice:** choose $\text{BLK}_{KV}$ such that $\text{BLK}_{KV} \le S / (2 \cdot SP)$ to keep the per-rank fragmentation factor within 2:

$$
\varphi_{\text{avg,SP-rank}} \approx 1 + \frac{\text{BLK}_{KV}}{2 \cdot (S/SP)} = 1 + \frac{\text{BLK}_{KV} \cdot SP}{2 S}
$$

For $SP > 1$, the effective fragmentation is therefore amplified by a factor of $SP$ relative to the single-device case with the same $\text{BLK}_{KV}$ and $S$.

### Pipeline Parallelism (PP)

Each PP stage owns $L/PP$ disjoint layers and manages KV blocks for those layers independently. There is no cross-stage KV sharing or block table coordination. Fragmentation and capacity constraints apply independently within each PP stage; all stages have equal block pool size (assuming uniform layer assignment).

---

## 6. Effective HBM Capacity Model

### Available HBM for KV storage

After subtracting parameter memory and activation workspace, the HBM available for KV blocks is:

$$
M_{\text{HBM,KV,avail}} = M_{\text{HBM}} - M_{\theta,\text{device}} - M_{\text{act,device}} - M_{\text{sys}}
$$

where $M_{\text{sys}}$ is a system-level overhead reserve (CUDA context, kernel workspace, OS pages; typically 1–2 GB per GPU). $M_{\theta,\text{device}}$ and $M_{\text{act,device}}$ are defined in `decode.md` §1.4.

### KV capacity after fragmentation

The block pool can store at most $M_{\text{HBM,KV,avail}} / M_{\text{block}}$ physical blocks, but due to internal fragmentation only a fraction $1/\varphi_{\text{avg}}$ of that capacity holds useful token data. The effective KV storage capacity (in token-bytes at the device level) is:

$$
M_{\text{KV,capacity}} = \frac{M_{\text{HBM,KV,avail}}}{\varphi_{\text{avg}}}
$$

### Maximum supportable context length

Combining with the per-device KV memory formula from §1, the maximum context length $S_{\max}$ satisfying the HBM constraint is:

$$
\frac{L}{PP} \cdot \frac{2 \, S_{\max} \, H_{kv} \, b}{TP \cdot SP} \le M_{\text{KV,capacity}}
$$

Solving for $S_{\max}$:

$$
S_{\max} = \frac{M_{\text{KV,capacity}} \cdot TP \cdot SP}{2 \, H_{kv} \, b \cdot (L/PP)}
= \frac{\left(M_{\text{HBM}} - M_{\theta,\text{device}} - M_{\text{act,device}} - M_{\text{sys}}\right) \cdot TP \cdot SP}{\varphi_{\text{avg}} \cdot 2 \, H_{kv} \, b \cdot (L/PP)}
$$

Substituting $\varphi_{\text{avg}} \approx 1 + \text{BLK}_{KV} / (2 S)$ and noting that this gives an implicit equation in $S_{\max}$, the leading-order closed-form solution (valid when $\text{BLK}_{KV} \ll S$) is obtained by evaluating $\varphi_{\text{avg}} \approx 1$ on the right-hand side for a first estimate $\hat{S}$, then correcting:

$$
S_{\max} \approx \frac{\hat{S}}{1 + \text{BLK}_{KV} / (2\hat{S})}
\approx \hat{S} \left(1 - \frac{\text{BLK}_{KV}}{2\hat{S}}\right)
= \hat{S} - \frac{\text{BLK}_{KV}}{2}
$$

where $\hat{S}$ is the contiguous-allocation estimate from `decode.md` §6.1:

$$
\hat{S} = \frac{\left(M_{\text{HBM}} - M_{\theta,\text{device}} - M_{\text{act,device}}\right) \cdot TP \cdot SP}{2 \, H_{kv} \, b \cdot (L/PP)}
$$

The paging correction reduces $S_{\max}$ by exactly $\text{BLK}_{KV} / 2$ tokens — one half-block per sequence — which is negligible for large contexts. The fragmentation penalty matters most in the regime $S \lesssim \text{BLK}_{KV}$, where it can reduce usable capacity substantially.

### Summary: fragmentation-corrected HBM constraint

The HBM feasibility condition from `decode.md` §6.1 is refined by replacing the nominal KV capacity with the fragmentation-penalized version:

$$
\frac{L}{PP}
\left[
\frac{2H^2 + 2 H H_{kv}}{TP} + \frac{3 H I \, N_{\text{exp}}}{TP \cdot EP} + \varphi_{\text{avg}} \cdot \frac{2 S H_{kv}}{TP \cdot SP}
\right] b + B(4H + 2H_{kv}) b + \frac{V H}{TP} b
\le M_{\text{HBM}} - M_{\text{sys}}
$$

For $\varphi_{\text{avg}} = 1$ (contiguous allocation), this reduces exactly to the expression in §6.1 of `decode.md`.

---

## References

**[VLLM]**  
Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C.H., Gonzalez, J.E., Zhang, H., & Stoica, I. (2023).  
*Efficient Memory Management for Large Language Model Serving with PagedAttention.*  
SOSP 2023. arXiv:2309.06180.  
→ PagedAttention; block-based KV paging; fragmentation analysis.

**[GQA]**  
Ainslie, J., Lee-Thorp, J., de Jong, M., Zelaski, T., Sanghai, S., & Xu, Y. (2023).  
*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.*  
EMNLP 2023. arXiv:2305.13245.  
→ Grouped-query attention; $n_{kv} < n_q$; KV cache size reduction via $H_{kv} = n_{kv} \cdot d_{\text{head}}$.

**[decode.md §1.3]** — Baseline per-layer and per-device KV memory derivation.

**[decode.md §2.3]** — Baseline per-device KV traffic per token.

**[decode.md §6.1]** — HBM feasibility constraint under contiguous KV allocation.

See `documentation/references.md` for full bibliography.
