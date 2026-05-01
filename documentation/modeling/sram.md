# SRAM-Centric Memory Hierarchy Model

**Multi-Tier Effective Bandwidth, Per-Data-Class Placement, and Mode Tradeoffs for SRAM-Resident Inference**

<br/>

**Author:** Yue Lu  
**Date:** April 2026

**Keywords:**  
SRAM, on-die SRAM, LPDDR, HBM, memory hierarchy, effective bandwidth, roofline, KV cache placement, batch amortization, Groq LPU, d-Matrix Corsair, in-memory compute

---

<div style="page-break-after: always;"></div>

# Table of Contents

- [1. Memory Hierarchy Spec](#1-memory-hierarchy-spec)
  - [1.1 Tier Parameters and Ordering](#11-tier-parameters-and-ordering)
  - [1.2 Effective Bandwidth and Contention](#12-effective-bandwidth-and-contention)
  - [1.3 Per-Data-Class Placement](#13-per-data-class-placement)
- [2. Multi-Tier Decode Roofline](#2-multi-tier-decode-roofline)
  - [2.1 Per-Tier Roofline](#21-per-tier-roofline)
  - [2.2 Crossover and Two Operating Modes](#22-crossover-and-two-operating-modes)
- [3. Numerical Examples](#3-numerical-examples)
  - [3.1 GPU Baseline — B200 HBM3e (Single Tier)](#31-gpu-baseline--b200-hbm3e-single-tier)
  - [3.2 d-Matrix Corsair, Performance Mode (Llama-3-8B)](#32-d-matrix-corsair-performance-mode-llama-3-8b)
  - [3.3 d-Matrix Corsair, Capacity Mode (Llama-3-70B)](#33-d-matrix-corsair-capacity-mode-llama-3-70b)
  - [3.4 Groq LPU, Single SRAM Tier (Llama-2-70B)](#34-groq-lpu-single-sram-tier-llama-2-70b)
- [Symbol Summary](#symbol-summary)
- [References](#references)

---

<div style="page-break-before: always;"></div>

## Introduction

The decode roofline in `decode.md §4` assumes a single memory tier — typically HBM — and collapses per-token memory time to $t_{\mathrm{mem}} = T_{\mathrm{token,device}}^{\mathrm{eff}} / BW_{\mathrm{mem}}$. SRAM-centric inference accelerators violate that single-tier assumption. Groq's Language Processing Unit (LPU) places all weights and the KV cache directly in distributed on-die SRAM at 80 TB/s per chip with no DRAM in the device path [GROQ-LPU, GROQ-DEEPDIVE]. d-Matrix's Corsair card splits memory across two physical tiers per device — 2 GB SRAM at 150 TB/s for the active working set, and 256 GB Low-Power DDR5 (LPDDR5) at 400 GB/s for the capacity tail — and exposes a runtime mode selector that picks where weights live [DMATRIX-HC25, CHIPSANDCHEESE-DM]. Conventional GPUs sit at the other extreme: HBM at 1.5–8 TB/s with on-die L1/L2 SRAM used only as a per-tile scratchpad inside FlashAttention [FA1, FA2].

The contrast worth drawing up front is between two layers of the optimization stack. FlashAttention-style kernel tiling is a **software-level** optimization that targets the *transient working data* of the attention step — the $S \times S$ score matrix that would otherwise round-trip through HBM ([FA1] Theorem 2; `decode.md §2.2`). It does not, and structurally cannot, reduce the per-token loads of model weights or the KV cache, which are the **fundamental** bandwidth costs of decode. SRAM-centric architectures address that fundamental cost at the **hardware** level by placing weights — and, when capacity permits, the KV cache itself — on a memory tier with one to two orders of magnitude more bandwidth than HBM. The two optimizations are complementary but operate on different cost components: FA collapses an attention-internal traffic term that is already a secondary contributor on long contexts; SRAM-resident placement collapses the **dominant** per-token traffic terms ($T_\theta$ and $T_{\mathrm{KV}}$ in the roofline of `decode.md §4`). This document is about the second lever, on the assumption that the first is already in effect (per `decode.md §2.2`, $T_{\mathrm{act}}$ is dropped from the model — the FA-applied regime is the default).

This document extends the decode roofline to a generic ordered list of memory tiers, derives per-tier effective bandwidth, places each per-token data class — weights, KV cache, activations — onto tiers under a capacity constraint, and reassembles the roofline memory term as a sum across tiers. The new framework recovers the legacy single-tier form exactly when a device exposes only HBM, so existing `SystemSpec` consumers continue to work unchanged. Section 6 walks the model through three reference architectures: a B200 HBM3e baseline, d-Matrix Corsair in both Performance and Capacity Modes, and the Groq LPU's single SRAM tier.

The model is purely analytical and consumes the same `SystemSpec` / `PartitionSpec` / `TuningSpec` inputs already used by `decode.md` and `prefill.md`. Energy-per-bit modeling is out of scope here — the bandwidth-resident-weight pitch is justified on bandwidth (and therefore latency) alone; the bytes-moved decomposition in §2.1 makes a per-tier joules-per-bit accounting trivial to add later, but no joule numbers appear in this document.

---

# 1. Memory Hierarchy Spec

This section defines the static specification a multi-tier device exposes: the per-tier physical parameters (§1.1), the effective bandwidth that enters the roofline after contention (§1.2), and the placement that decides which data class lives on which tier under the per-tier capacity budget (§1.3).

## 1.1 Tier Parameters and Ordering

A device exposes an ordered list of $n$ memory tiers, fastest first, indexed $i = 0, 1, \ldots, n - 1$. Each tier is parameterized by four quantities, all observable from device datasheets or chip-level talks.

| Symbol | Description | Units |
|--------|-------------|-------|
| $C_i$ | Tier capacity per device | bytes |
| $BW_i$ | Tier peak read bandwidth from compute | bytes/s |
| $\alpha_i$ | First-byte latency floor | seconds |
| $\eta_{\beta,i}$ | Sustained-bandwidth deflator (post-refresh, post-conflict) | dimensionless, $\in (0, 1]$ |

The legacy single-tier model in `decode.md` corresponds to a one-element tier list (HBM only): $C_0 = M_{\mathrm{HBM}}$, $BW_0 = BW_{\mathrm{mem}}$, and $\eta_{\beta,0} \approx 0.92$ (HBM refresh + bank conflicts cost roughly 8% of theoretical peak [VIKS-SRAM]). The new tier list collapses to that one-tier form by construction.

The tier order is an **ordering constraint, not a hardware-managed cache hierarchy**: data resident in tier $i$ is read directly from tier $i$, not promoted to a faster tier on access. This matches the on-die SRAM model used by Groq and d-Matrix, where weight tiles are placed at compile time and the runtime never migrates them to a smaller faster tier [GROQ-DEEPDIVE, DMATRIX-HC25]. CPU-style demand-paging caches are out of scope; on a GPU the HBM → L2 → SMEM hierarchy is collapsed into the single $BW_{\mathrm{mem}}$ tier, with FlashAttention's tiling savings already captured in `decode.md §2.2` by dropping the score-matrix traffic from the model entirely (KV traffic is unaffected).

## 1.2 Effective Bandwidth and Contention

The per-tier bandwidth that enters the roofline is the peak bandwidth deflated by sustained-throughput losses:

$$BW_{\mathrm{eff},i} = BW_i \cdot \eta_{\beta,i}$$

The dynamic-deflator framework already used for collective bandwidth in `collectives/05_contention_and_congestion.md §4` applies here unchanged — $\eta_{\beta,i} = 1$ is the calibrated peak ceiling, and any value below 1 captures real-world losses. SRAM tiers run very close to peak ($\eta_{\beta,\mathrm{SRAM}} \approx 1.0$) because they have no refresh, no row activation, and no bank conflicts at the granularity that matters for tile-aligned tensor loads [VIKS-SRAM]. HBM and LPDDR5 sit lower: $\eta_{\beta,\mathrm{HBM}} \approx 0.92$ (refresh + bank conflicts ≈ 8%) and $\eta_{\beta,\mathrm{LPDDR5}} \approx 0.85$ (refresh plus a lower interface efficiency at burst boundaries) are the defaults used in §4.

The first-byte latency $\alpha_i$ is the per-transaction startup cost on tier $i$ (Hockney-style $\alpha$–$\beta$ model, as in `collectives/01_collective_algorithms.md §1`). It enters the full multi-tier roofline in §2.1 but is dropped for device-level decode timing on magnitude grounds — see §2.1 for the bookkeeping and the small-read regimes (paged-attention block fetch, LLM-in-a-Flash spill [LLM-FLASH]) where reinstating it matters.

## 1.3 Per-Data-Class Placement

Three classes of data move from memory to compute during decode (per `decode.md §2`):

- $T_{\theta,\mathrm{device}}$ — weights (model parameters), read once per decode step regardless of batch size $B$.
- $T_{\mathrm{KV,device}}$ — KV cache, read once **per request** per step (so the per-step KV traffic scales linearly with $B$).
- $T_{\mathrm{act,device}}$ — activations (intermediate working memory), small and resident in registers / shared memory; absorbed into $t_{\mathrm{compute}}$ in `decode.md §4` and not re-modeled here.

A **placement** $\pi$ assigns each data class to one tier or splits it across tiers:

$$\pi(\theta) = (T_{\theta,0}, T_{\theta,1}, \ldots), \qquad \pi(\mathrm{KV}) = (T_{\mathrm{KV},0}, T_{\mathrm{KV},1}, \ldots)$$

with conservation $\sum_i T_{\theta,i} = T_{\theta,\mathrm{device}}$ and $\sum_i T_{\mathrm{KV},i} = T_{\mathrm{KV,device}}$. Per `decode.md §2.3`, $T_{\mathrm{KV,device}}$ is the **per-device, per-request** KV bytes (the $\mathrm{TP} \cdot \mathrm{SP}$ sharding and the current context length $S$ are already baked in), so the per-device per-tier capacity constraint at decode time is simply:

$$T_{\theta,i} + B \cdot T_{\mathrm{KV},i} \;\le\; C_i$$

The left-hand side is the resident footprint on tier $i$ of one device — weights (read once per step, so footprint = traffic) plus KV cache across all $B$ active requests assigned to that tier.

Two placement policies cover the practical cases:

**Greedy fastest-first.** Fill tier 0 with the priority class until it spills; place the remainder on tier 1; place the second class on whichever tier has remaining capacity, prioritizing the fastest tier that fits. The priority class is **weights by default** (the `auto_priority="weights"` field on `MemoryPlacementSpec`) on the rationale that weights are a stable size for a given deployment while KV grows with context length and batch — pinning the stable class to the fast tier is more robust. Set `auto_priority="kv"` to flip the order when the KV term dominates and SRAM-resident KV matters more than SRAM-resident weights. This matches the Groq deployment model (a single fast tier, no spill option) and d-Matrix Performance Mode (weights pinned to SRAM, KV on SRAM if it fits else LPDDR5) [DMATRIX-HC25].

**Operator-specified.** Each data class is pinned to a chosen tier by a `MemoryPlacementSpec` field on the tuner; pin overflow raises `CapacityError`. This matches d-Matrix Capacity Mode, where the operator pins weights to LPDDR5 to free SRAM for a larger batch or longer context [DMATRIX-HC25]. The two-mode toggle exposed by d-Matrix's Aviator runtime is exactly this placement-spec switch. `auto_priority` is inert for any class that is explicitly pinned.

---

# 2. Multi-Tier Decode Roofline

This section opens the per-token roofline term by tier (§2.1) and then derives the batch-size crossover and the two well-known operating modes that the formula admits (§2.2).

## 2.1 Per-Tier Roofline

For a fixed placement $\pi$ and batch size $B$, the per-step memory time on tier $i$ follows the standard $\alpha$–$\beta$ form: a per-transaction startup latency plus a bytes-moved term at the tier's effective bandwidth. Treating each tier as a single transaction per step (the simplest no-paging assumption — one fetch covers all bytes resident on the tier), the per-step memory time across all tiers is:

$$t_{\mathrm{mem}}(B) \;=\; \sum_{i=0}^{n-1} \left[\, \alpha_i \cdot \mathbb{1}\!\bigl(\mathrm{bytes}_i > 0\bigr) \;+\; \frac{T_{\theta,i} + B \cdot T_{\mathrm{KV},i}}{BW_{\mathrm{eff},i}} \,\right]$$

The indicator on $\alpha_i$ gates the startup cost on a non-empty placement: a tier holding no weights and no KV pays no first-byte fee. The $\alpha_i$ floor is retained for completeness so the device roofline is the full $\alpha$–$\beta$ form rather than a $\beta$-only shorthand. For the on-die / co-packaged tiers in scope here — SRAM, HBM, LPDDR5 — $\alpha_i$ sits in the 1 ns (SRAM) to ~200 ns (LPDDR5) range, while the per-step bytes term lands in the 10 µs to 10 ms range across the §3 examples; $\alpha_i$ contributes well under 0.1% of $t_{\mathrm{mem}}$ in every case. Keeping the $\alpha_i$ term in the formula is therefore harmless on accuracy and useful as the structural anchor for *small-read* regimes (paged-attention block fetches, flash-style chunked weight loads) where it does start to bind; a future refinement will replace the 1-transaction count with $N_{\mathrm{txn},i} = \lceil \mathrm{bytes}_i / G_i \rceil$ for those workloads.

The form here is the structural generalization of the legacy `decode.md §4.3` formula and matches the per-token decode roofline of LIMINAL [LIMINAL, Eq. 1] with the aggregate-bandwidth term opened up into per-tier components and the $\alpha$ floor reinstated. Two consequences fall out of the bytes-moved part:

1. **Weights amortize over batch within their tier.** The $T_{\theta,i}$ term carries no $B$ factor — doubling $B$ halves the per-token weight cost on every tier where weights live.
2. **KV cache does not amortize.** The $B \cdot T_{\mathrm{KV},i}$ term scales linearly with $B$ on every tier. Increasing batch trades weight-bound time for KV-bound time but never eliminates the KV component. This matches the empirical "Mind the Memory Gap" observation that large-batch GPU inference remains memory-bound up to and past the throughput plateau, because attention reads scale linearly with $B$ while attention's arithmetic intensity stays nearly constant [MIND-GAP].

The full local roofline becomes

$$t_{\mathrm{local}}(B) = \max\!\bigl(t_{\mathrm{compute}}(B),\; t_{\mathrm{mem}}(B)\bigr)$$

with $t_{\mathrm{compute}}(B) = B \cdot F_{\mathrm{token,device}} / R_{\mathrm{GPU}}$ as in `decode.md §4.2`. The pipeline-stage assembly and bubble correction in `decode.md §6.2` (overlap factor $\rho$, bubble multiplier $\gamma_{\mathrm{pp}}$) apply unchanged on top of this $t_{\mathrm{local}}(B)$.

The single-tier reduction is exact: when $n = 1$ the sum has one term and

$$t_{\mathrm{mem}}(B) = \alpha_0 \;+\; \frac{T_{\theta,\mathrm{device}} + B \cdot T_{\mathrm{KV,device}}}{BW_{\mathrm{eff},0}}$$

recovering `decode.md §4.3` with $BW_{\mathrm{mem}} \equiv BW_{\mathrm{eff},0}$ when $\alpha_0$ is dropped (the negligible-$\alpha$ regime for on-die / co-packaged tiers).

## 2.2 Crossover and Two Operating Modes

The single-tier crossover batch size $B^*$ from `notation.md §10` generalizes naturally. When weights live entirely on one tier (call it $W$) and KV lives entirely on another (call it $K$), setting $t_{\mathrm{compute}}(B) = t_{\mathrm{mem}}(B)$ and solving for $B$ yields:

$$B^*_{W,K} = \frac{R_{\mathrm{GPU}} \cdot T_{\theta,\mathrm{device}} / BW_{\mathrm{eff},W}}{F_{\mathrm{token,device}} - R_{\mathrm{GPU}} \cdot T_{\mathrm{KV,device}} / BW_{\mathrm{eff},K}}$$

This reduces to the single-tier `notation.md §10` form when $W = K$. The structural insight is the asymmetric way the two tiers enter:

- A **large $BW_{\mathrm{eff},W}$** (weights on a fast tier) shrinks the numerator and pulls $B^*$ toward 1. SRAM-resident weights drive $B^*$ into the 1–4 range, which is what produces d-Matrix's reported low-latency batched throughput at very small batch [DMATRIX-HC25].
- A **small $BW_{\mathrm{eff},K}$** (KV on a slow tier) shrinks the denominator and pushes $B^* \to \infty$. Once $R_{\mathrm{GPU}} \cdot T_{\mathrm{KV,device}} / BW_{\mathrm{eff},K} \ge F_{\mathrm{token,device}}$ the system stays memory-bound at any batch — the KV-bound long-context regime of `notation.md §10` re-stated in tier terms.

These two behaviors recover two well-known operating modes for two-tier devices:

**Performance Mode (weights and KV both resident in fastest tier).** Both terms in $t_{\mathrm{mem}}$ use $BW_{\mathrm{eff},0}$; the absolute memory time is minimized; $B^*$ is small because the high $BW_W$ shrinks the numerator. This is d-Matrix's recipe for sub-2 ms TPOT on Llama-3-70B-class workloads when the model fits in aggregate SRAM [DMATRIX-HC25] and the implicit Groq mode (the LPU has only one tier) [GROQ-LPU].

**Capacity Mode (weights pinned to slower tier).** Weights pay $BW_{\mathrm{eff},1}$ instead of $BW_{\mathrm{eff},0}$; per-token weight time grows by the bandwidth ratio $BW_{\mathrm{eff},0} / BW_{\mathrm{eff},1}$. For d-Matrix Corsair this ratio is roughly $150 \mathrm{\,TB/s} / 0.34 \mathrm{\,TB/s} \approx 440\times$, so per-token weight cost grows by the same factor. The freed SRAM capacity admits a larger batch, longer context, or a larger model, raising aggregate throughput at the expense of per-user TPOT — exactly the Aviator-runtime tradeoff described in [DMATRIX-HC25, CHIPSANDCHEESE-DM].

---

# 3. Numerical Examples

Three reference architectures, evaluated against decode-only workloads at $B = 16$, $S = 8192$ context, with default $\eta_\beta$ values from §1.2 ($\eta_{\beta,\mathrm{HBM}} = 0.92$, $\eta_{\beta,\mathrm{LPDDR5}} = 0.85$, $\eta_{\beta,\mathrm{SRAM}} = 1.0$). Per-device traffic terms follow `decode.md §2`:

$$T_{\theta,\mathrm{device}} \;=\; \frac{\text{model bytes}}{\mathrm{TP} \cdot \mathrm{EP}}, \qquad T_{\mathrm{KV,device}} \;=\; \frac{2 S L H_{kv} b}{\mathrm{TP} \cdot \mathrm{SP}}$$

## 3.1 GPU Baseline — B200 HBM3e (Single Tier)

Tier list: HBM only. $C_0 = 192 \mathrm{\,GB}$, $BW_0 = 8 \mathrm{\,TB/s}$, $\eta_{\beta,0} = 0.92$ → $BW_{\mathrm{eff},0} \approx 7.36 \mathrm{\,TB/s}$ [H100-SPEC, HBM-SPEC].

Workload: Llama-3-70B at FP8 (model bytes ≈ 70 GB), $L = 80$, $H = 8192$, $n_{kv} = 8$, $\mathrm{TP} = 8$, single replica. Then $T_{\theta,\mathrm{device}} \approx 8.75 \mathrm{\,GB}$ and $T_{\mathrm{KV,device}} \approx 1.05 \mathrm{\,GB}$ at $B = 16$:

$$t_{\mathrm{mem}} \;=\; \frac{8.75 + 16 \cdot 1.05}{7{,}360} \mathrm{\,s} \;=\; \frac{25.5 \mathrm{\,GB}}{7.36 \mathrm{\,TB/s}} \;\approx\; 3.5 \mathrm{\,ms}$$

The KV term contributes 2.3 ms and weights contribute 1.2 ms — KV-dominated at $B = 16$, consistent with the GPU large-batch regime described in [MIND-GAP].

## 3.2 d-Matrix Corsair, Performance Mode (Llama-3-8B)

Tier list per card: SRAM (tier 0) + LPDDR5 (tier 1). $C_{\mathrm{SRAM}} = 2 \mathrm{\,GB}$, $BW_{\mathrm{SRAM}} = 150 \mathrm{\,TB/s}$, $\eta_{\beta} = 1.0$; $C_{\mathrm{LPDDR5}} = 256 \mathrm{\,GB}$, $BW_{\mathrm{LPDDR5}} = 0.4 \mathrm{\,TB/s}$, $\eta_\beta = 0.85$ → $BW_{\mathrm{eff,LPDDR5}} \approx 0.34 \mathrm{\,TB/s}$ [DMATRIX-HC25, CHIPSANDCHEESE-DM].

Workload: Llama-3-8B at MXINT8 ≈ 8 GB ($L = 32$, $H = 4096$, $n_{kv} = 8$, $d_{\text{head}} = 128$, so $H_{kv} = 1024$). d-Matrix reports an 8-card single-server deployment for this model in SRAM mode [DMATRIX-HC25]; treating each card as a $\mathrm{TP}$ unit, per-card $T_{\theta} = 1 \mathrm{\,GB}$ pinned to SRAM. Per-request KV after the 8-way shard is $2 \cdot 8192 \cdot 32 \cdot 1024 / 8 \approx 67 \mathrm{\,MB}$, so at $B = 16$ the aggregate KV is ≈ 1.07 GB per card — fits in the remaining SRAM with margin. Both data classes on tier 0:

$$t_{\mathrm{mem}} \;=\; \frac{1 \mathrm{\,GB} + 16 \cdot 0.067 \mathrm{\,GB}}{150 \mathrm{\,TB/s}} \;\approx\; 14 \mathrm{\,\mu s}$$

Memory time is essentially free. The reported ~1 ms TPOT for Llama-3-8B [DMATRIX-HC25] is dominated by compute and inter-chiplet synchronization, not memory traffic — exactly the regime SRAM-resident weights are designed to produce.

## 3.3 d-Matrix Corsair, Capacity Mode (Llama-3-70B)

Same tier list as §3.2 but with weights pinned to LPDDR5 because Llama-3-70B does not fit in aggregate SRAM (32 GB across 16 cards vs 70 GB at MXINT8). d-Matrix reports a 16-card / 128-chiplet rack deployment for this configuration [DMATRIX-HC25]; per-card $T_{\theta} \approx 4.4 \mathrm{\,GB}$ (70 GB / 16 cards), pinned to tier 1. Per-card per-request KV at $L = 80$, $H_{kv} = 1024$, $S = 8192$ after the 16-way shard is $2 \cdot 8192 \cdot 80 \cdot 1024 / 16 \approx 84 \mathrm{\,MB}$, so at $B = 16$ aggregate KV ≈ 1.34 GB per card — small enough to remain on SRAM (tier 0).

$$t_{\mathrm{mem}} \;=\; \underbrace{\frac{16 \cdot 0.084 \mathrm{\,GB}}{150 \mathrm{\,TB/s}}}_{\mathrm{KV\;on\;SRAM\;(tier\;0)}} \;+\; \underbrace{\frac{4.4 \mathrm{\,GB}}{0.34 \mathrm{\,TB/s}}}_{\mathrm{weights\;on\;LPDDR5\;(tier\;1)}} \;\approx\; 9 \mathrm{\,\mu s} + 13 \mathrm{\,ms} \;\approx\; 13 \mathrm{\,ms}$$

The LPDDR5 weight load swamps everything else and the system reverts to a $\sim 0.34 \mathrm{\,TB/s}$ effective bandwidth for the dominant term — the same $\sim 440\times$ slowdown predicted by the bandwidth ratio in §2.2. The reported 2 ms TPOT [DMATRIX-HC25] is achieved at a substantially larger aggregate batch / pipeline that further amortizes weights, illustrating why Capacity Mode is paired with high-throughput rather than low-latency operating points.

## 3.4 Groq LPU, Single SRAM Tier (Llama-2-70B)

Tier list per chip: SRAM only. $C_0 = 230 \mathrm{\,MB}$, $BW_0 = 80 \mathrm{\,TB/s}$, $\eta_{\beta,0} = 1.0$ [GROQ-LPU].

Workload: Llama-2-70B at INT8 = 70 GB. The single-tier capacity constraint forces $\mathrm{TP} \ge \lceil 70 \mathrm{\,GB} / 0.23 \mathrm{\,GB} \rceil = 305$ chips just for weights; layering KV on the same SRAM grows the deployment to a reported 576+ chips [GROQ-DEEPDIVE]. Per-chip $T_{\theta} = 230 \mathrm{\,MB}$ (full SRAM), $T_{\mathrm{KV}}$ negligible per-chip after the 305-way shard.

$$t_{\mathrm{mem}} \;\approx\; \frac{230 \mathrm{\,MB}}{80 \mathrm{\,TB/s}} \;\approx\; 2.9 \mathrm{\,\mu s}$$

The local memory tier is essentially free; the published TPOT of ~3 ms for Llama-2-70B inference on Groq [GROQ-LPU] is driven by per-step inter-chip communication across the 305+ chip topology, not local memory access. The bottleneck migrates entirely to the fabric, which is the price the architecture pays for spanning a 70 GB model across 230 MB tiers.

## Summary

| Architecture | Tiers | Decode-bound regime |
|--------------|-------|---------------------|
| B200 HBM3e (Llama-3-70B, $B=16$) | 1 (HBM) | KV-dominated; $t_{\mathrm{mem}} \approx 3.5$ ms |
| d-Matrix Performance (Llama-3-8B) | 2; weights+KV in SRAM | Compute-bound; $t_{\mathrm{mem}} \approx 14$ µs |
| d-Matrix Capacity (Llama-3-70B) | 2; weights in LPDDR5 | Weight-load-bound; $t_{\mathrm{mem}} \approx 13$ ms |
| Groq LPU (Llama-2-70B) | 1 (SRAM) | Comm-bound; $t_{\mathrm{mem}} \approx 3$ µs |

---

# Symbol Summary

Symbols introduced in this document; consolidated into `notation.md §16`.

| Symbol | Description | Units |
|--------|-------------|-------|
| $n$ | Number of memory tiers exposed by a device | integer |
| $i$ | Tier index, $0 \le i < n$, ordered fastest first | integer |
| $C_i$ | Tier $i$ capacity per device | bytes |
| $BW_i$ | Tier $i$ peak read bandwidth | bytes/s |
| $\alpha_i$ | Tier $i$ first-byte latency floor | seconds |
| $\eta_{\beta,i}$ | Tier $i$ sustained-bandwidth deflator | dimensionless |
| $BW_{\mathrm{eff},i}$ | Effective tier $i$ bandwidth: $BW_i \cdot \eta_{\beta,i}$ | bytes/s |
| $\pi$ | Placement assigning each data class to tiers | — |
| $T_{\theta,i}$ | Weight bytes residing on tier $i$ | bytes |
| $T_{\mathrm{KV},i}$ | Per-request KV bytes residing on tier $i$ | bytes |
| $B^*_{W,K}$ | Two-tier crossover with weights on $W$, KV on $K$ | — |

---

# References

- **[GROQ-LPU]** — Groq, *LPU Architecture Overview.* https://groq.com/lpu-architecture. SRAM-only inference accelerator: 230 MB on-die SRAM at 80 TB/s per chip; deterministic compiler-scheduled execution. Anchor for the §1.1 single-tier-SRAM motivation and the §3.4 example.
- **[GROQ-DEEPDIVE]** — He, K. (2024), *Inside the Groq LPU.* https://01.me/en/2024/02/groq/. Llama-2-70B chip-count derivation: 305+ chips for INT8 weights, 576+ with KV cache. Used for §3.4 deployment math.
- **[DMATRIX-HC25]** — Patrick Kennedy / ServeTheHome (2025), coverage of d-Matrix Corsair at Hot Chips 2025. https://www.servethehome.com/d-matrix-corsair-in-memory-computing-for-ai-inference-at-hot-chips-2025/. Per-card spec: 2 GB SRAM at 150 TB/s + 256 GB LPDDR5 at 400 GB/s; 8 chiplets per card with 64×64 INT8 (or 64×128 INT4) matmul tiles; 60k tokens/s @ 1 ms/token on Llama-3-8B (8-card server) and 30k tokens/s @ 2 ms/token on Llama-3-70B (16-card / 128-chiplet rack); two operating modes via Aviator runtime. Anchor for §1.1, §1.3, §2.2, §3.2–§3.3.
- **[CHIPSANDCHEESE-DM]** — Wong, C. (2024), *d-Matrix Corsair: 256GB of LPDDR for AI Models.* Chips and Cheese. https://chipsandcheese.com/p/d-matrix-corsair-256gb-of-lpddr-for. Independent confirmation of the per-package SRAM/LPDDR split and Performance/Capacity mode toggle. Used in §1.1 and §3.2–§3.3.
- **[VIKS-SRAM]** — Bhardwaj, V. (2024), *A Close Look at SRAM for Inference in the Age of HBM Supremacy.* Vik's Newsletter. https://www.viksnewsletter.com/p/a-close-look-at-sram-for-inference. Anchor for the $\eta_{\beta,\mathrm{HBM}} \approx 0.92$ vs $\eta_{\beta,\mathrm{SRAM}} \approx 1.0$ comparison: HBM loses roughly 8% to refresh + bank conflicts that SRAM avoids.
- **[LIMINAL]** — Diamantopoulos, D., Pothineni, N., et al. (2025), *Efficient LLM Inference: Bandwidth, Compute, Synchronization, and Capacity are all you need.* arXiv:2507.14397. Per-token decode roofline $T_{\mathrm{Mem}} = (\mathrm{Batch\_KV\_Bytes} + \mathrm{Model\_Bytes}) / \mathrm{System\_Aggregate\_Bandwidth}$; mean absolute error of 7.6% against measured hardware. The §2.1 multi-tier form is a per-tier opening of LIMINAL Eq. 1 and matches the single-tier reduction exactly.
- **[MIND-GAP]** — *Mind the Memory Gap: Unveiling GPU Bottlenecks in Large-Batch LLM Inference* (2025). arXiv:2503.08311. Empirical demonstration that large-batch inference remains memory-bound: attention arithmetic intensity stays nearly constant across batch sizes because KV traffic scales linearly with $B$. Used in §2.1 to justify the "weights amortize, KV does not" claim.
- **[LLM-FLASH]** — Alizadeh, K., Mirzadeh, I., et al. (2024), *LLM in a flash: Efficient Large Language Model Inference with Limited Memory.* arXiv:2312.11514. Three-tier flash → DRAM → compute hierarchy; chunk-size amortization of first-byte latency; sliding-window weight cache. Used in §1.2 to justify retaining $\alpha_i$ in the tier spec for small-read regimes.
- **[FA1]**, **[FA2]**, **[ROOFLINE]**, **[VLLM]**, **[H100-SPEC]**, **[HBM-SPEC]** — see `references.md`. Cited from §1.1, §1.2, §3.1 as cross-references to the established decode model and to anchor the GPU baseline numbers.

_Full bibliographic entries for all tags are in `references.md`. Entries new to this document — `[GROQ-LPU]`, `[GROQ-DEEPDIVE]`, `[DMATRIX-HC25]`, `[CHIPSANDCHEESE-DM]`, `[VIKS-SRAM]`, `[LIMINAL]`, `[MIND-GAP]`, `[LLM-FLASH]` — should be appended to `references.md` in the same change._
