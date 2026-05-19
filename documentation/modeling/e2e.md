# End-to-End LLM Inference Metrics

**Assembling TTFT, TPOT, Throughput/GPU, Interactivity, and the Throughput–Latency Pareto Frontier**

<br/>

**Author:** Yue Lu  
**Date:** April 2026

**Keywords:**  
LLM inference, TTFT, time-to-first-token, TPOT, time-per-output-token, throughput per GPU, interactivity, continuous batching, chunked prefill, Pareto frontier, roofline model, throughput–latency tradeoff

---

<div style="page-break-after: always;"></div>

# Table of Contents

- [1. Metric Definitions](#1-metric-definitions)
  - [1.1 Time To First Token (TTFT)](#11-time-to-first-token-ttft)
  - [1.2 Time Per Output Token (TPOT)](#12-time-per-output-token-tpot)
  - [1.3 Throughput per GPU](#13-throughput-per-gpu)
  - [1.4 Interactivity](#14-interactivity)
  - [1.5 Goodput](#15-goodput)

- [2. TTFT Assembly](#2-ttft-assembly)
  - [2.1 Single-Request TTFT](#21-single-request-ttft)
  - [2.2 Batched-Prefill TTFT](#22-batched-prefill-ttft)
  - [2.3 Chunked-Prefill TTFT](#23-chunked-prefill-ttft)

- [3. TPOT Assembly](#3-tpot-assembly)
  - [3.1 Static Batching TPOT](#31-static-batching-tpot)
  - [3.2 Continuous Batching TPOT](#32-continuous-batching-tpot)

- [4. When Prefill Dominates Decode](#4-when-prefill-dominates-decode)

- [5. Throughput/GPU and Interactivity](#5-throughputgpu-and-interactivity)
  - [5.1 Throughput/GPU Derivation](#51-throughputgpu-derivation)
  - [5.2 Interactivity (per-request streaming rate)](#52-interactivity-per-request-streaming-rate)

- [6. Throughput–Latency Pareto Frontier](#6-throughputlatency-pareto-frontier)
  - [6.1 Levers Shaping the Pareto Curve](#61-levers-shaping-the-pareto-curve)
  - [6.2 Three Zones of the Pareto Frontier](#62-three-zones-of-the-pareto-frontier)
  - [6.3 Throughput–Latency Identity](#63-throughputlatency-identity)
  - [6.4 Throughput–Interactivity Axis Mapping](#64-throughputinteractivity-axis-mapping)

- [Symbol Summary](#symbol-summary)

---

<div style="page-break-before: always;"></div>

# 1. Metric Definitions

This section defines all six end-to-end metrics precisely. They are the outputs users and system operators care about; all prior modeling documents (memory, FLOPs, traffic, communication, latency) contribute to computing them. Two of these six — **Throughput/GPU** (system efficiency) and **Interactivity** (user-perceived quality) — are the standard axes used by production LLM inference benchmarks. The others provide essential context and diagnostic signal.

Symbols used throughout this document are defined in `notation.md`; see especially §§4, 9, 11, and 14. New symbols introduced here are summarized in the [Symbol Summary](#symbol-summary) at the end.

---

## 1.1 Time To First Token (TTFT)

**Definition.** $TTFT$ is the wall-clock elapsed time from the moment a request is received by the serving system to the moment the **first output token** is returned to the caller. It encompasses all latency incurred before the first token can be streamed: request scheduling, tokenization, the prefill forward pass, optional KV cache transfer (disaggregated architectures), and the first decode step that produces token 1.

$$
TTFT = t_{\text{sched}} + t_{\text{tok}} + t_{\text{prefill}} + t_{\text{KV-transfer}} + t_{\text{step,user}}
$$

where $t_{\text{KV-transfer}} = 0$ for co-located prefill+decode. The prefill latency $t_{\text{prefill}}$ is derived in full in `prefill.md §3`; framework overhead terms $t_{\text{sched}}$ and $t_{\text{tok}}$ are defined in `framework.md §2`.

> **Kernel-launch dispatch overhead.** Both $t_{\text{prefill}}$ and $t_{\text{step,user}}$ already incorporate per-stage CPU/host dispatch budget through their respective roofline compositions — see `decode.md §7.1` for the canonical definition of $t_{\text{stage,kernel}}$ and `prefill.md §3.4` for how prefill consumes it. The TTFT formula above does not add a separate kernel-launch term to avoid double counting; the legacy per-step `OverheadSpec.t_graph_us` / `t_launch_us` constants are only used when `kernel_launch_us = 0` in the tuner (kernel-launch modeling explicitly disabled).

> **LM head.** Both $t_{\text{prefill}}$ and $t_{\text{step,user}}$ also subsume the LM head $H \to V$ projection — a once-per-pass GPU-side roofline term on the last PP stage (`prefill.md §3.4` / `decode.md §6.2 / §7.2`). It is not a separate add-on at the TTFT level. The post-LM-head sampling kernel is folded into the same $t_{\text{LM,hw}}$ surcharge, not into $t_{\text{framework}}$.

**Key property.** TTFT is a *latency-to-first-byte* (TTFB) metric. For interactive applications, a high TTFT produces a blank screen during prefill — the most perceptible user experience degradation for long prompts.

---

## 1.2 Time Per Output Token (TPOT)

**Definition.** $\text{TPOT}$ is the **user-observed inter-token latency** for tokens 2 through $N_{\text{out}}$ of a single response (the decode phase). Each decode step produces exactly one new token per active sequence, so a user's TPOT equals the full step time $t_{\text{step,user}}$ — **not** amortized across the $B$ parallel sequences. The full form is assembled in two stages: `decode.md §7.2` builds the per-step hardware window from the per-stage HW roofline, the kernel-launch dispatch budget, the pipeline bubble multiplier, and the once-per-step LM head surcharge; `decode.md §7.3` then composes the per-sequence serving overhead on top:

$$
\text{TPOT} = t_{\text{step,user}}(B) = t_{\text{step,hw}}(B) + \max\!\bigl(0,\; t_{\text{serving,gross}}(B) - \rho_{\mathrm{serving}} \cdot t_{\text{step,hw}}(B)\bigr)
$$

with
$$
t_{\text{step,hw}}(B) = \gamma_{\text{pp}} \cdot \bigl[ t_{\text{stage,hw}}(B) + \max\!\bigl(0,\; t_{\text{stage,kernel}} - \rho_{\text{kernel}} \cdot t_{\text{stage,hw}}(B)\bigr) \bigr] + t_{\text{LM,hw}}(B)
$$

where $B$ is the number of sequences decoded concurrently, $t_{\text{stage,hw}}(B)$ is the overlap-aware per-stage HW step time (`decode.md §6.2`), $t_{\text{stage,kernel}}$ is the per-stage CPU/host kernel-launch dispatch budget composed via $\rho_{\text{kernel}}$ (`decode.md §7.1`), $\gamma_{\text{pp}} = \max(1, PP/B)$ is the pipeline bubble correction (`decode.md §7.2`) — at $B \ge PP$ the pipeline is kept full and the factor is 1; at $B < PP$ the single microbatch pays full pipeline depth per token — $t_{\text{LM,hw}}(B)$ is the LM head $H \to V$ projection roofline on stage $PP{-}1$ (added outside $\gamma_{\text{pp}}$ since it fires once per step, not per stage), and $t_{\text{serving,gross}}(B) = c_{\mathrm{serving}} \cdot B$ is the per-sequence serving runtime overhead composed via $\rho_{\mathrm{serving}}$ (`decode.md §7.3`).

**Key property.** TPOT is the *streaming rate* perceived by the user. A TPOT of 50 ms means one new token appears every 50 ms — a rate of 20 tokens/s. Human reading comprehension speed is approximately 5–15 tokens/s; TPOT below 100 ms (>10 tokens/s) is a common production SLA threshold.

---

## 1.3 Throughput per GPU

**Definition.** The rate of output token generation per physical GPU, expressed in output tokens/second/GPU:

$$
\text{Tput/GPU} = \frac{TTPS}{N_{\text{GPUs}}}
$$

where $TTPS$ is the global cluster token throughput (tokens/s, all sequences) defined in `decode.md §7.3`, and $N_{\text{GPUs}}$ is the total number of GPUs in the cluster. This is the standard X-axis of throughput–latency benchmark plots.

---

## 1.4 Interactivity

**Definition.** The rate at which a single user receives output tokens, expressed in output tokens/second per request:

$$
\text{Interactivity} = \frac{1}{\text{TPOT}}
$$

This is the standard Y-axis of throughput–latency benchmark plots. Higher interactivity means faster streaming to the individual user. The reciprocal relationship makes clear that Interactivity and TPOT encode identical information in different units.

---

## 1.5 Goodput

**Definition.** The maximum request arrival rate $\lambda$ (requests/second) the cluster can sustain while keeping both TTFT and TPOT below operator-set service-level objectives [DISAGG-PREFILL]:

$$
\text{Goodput} = \max\,\lambda \quad \text{s.t.} \quad
P_{p}\!\left[\,TTFT(\lambda)\,\right] \le TTFT_{\text{SLO}}
\quad\text{and}\quad
P_{p}\!\left[\,\text{TPOT}(\lambda)\,\right] \le \text{TPOT}_{\text{SLO}}
$$

where $P_{p}[\cdot]$ is the $p$-th percentile of the per-request distribution (typically $p \in \{90, 99\}$).

**Tie to model parameters.** Both constraints are explicit functions of quantities defined elsewhere in this suite:

- $TTFT(\lambda)$ assembles from $t_{\text{sched}}$, $t_{\text{prefill}}$, $t_{\text{handoff}}$, and $t_{\text{step,user}}$ (§2; `prefill.md §6`). Under load it inflates as the prefill queue lengthens.
- $\text{TPOT}(\lambda) = t_{\text{step,user}}(\overline{B_{\text{eff}}}(\lambda))$ via continuous batching (§3.2), with $\overline{B_{\text{eff}}}(\lambda) \approx \lambda \cdot (TTFT + N_{\text{out}} \cdot \overline{\text{TPOT}})$ from Little's law (§3.2).

The two SLOs jointly bound $\lambda$: $TTFT_{\text{SLO}}$ caps how long the prefill queue can grow; $\text{TPOT}_{\text{SLO}}$ caps how large $\overline{B_{\text{eff}}}$ can become before per-step time crosses into Zone 2/3 of the roofline (§6.2). Goodput is the **tighter** of the two — whichever SLO binds first sets $\lambda$.

**Out of scope.** Speculative-decoding rejections, preemption-driven recompute (`kv.md §4.3`), and request-cancellation effects are real goodput drains but are not modeled in this suite.

**Partition-feasibility derivation.** Inverting the goodput definition into hard bounds on the partition shape $(PP, TP, EP, SP)$ and the operating batch size $B$ — the $B_{\max}$ from $\mathrm{TPOT_{SLO}}$, the $PP_{\max}$ from $\mathrm{TTFT_{SLO}}$, the joint feasibility region $\mathcal{F}_{\text{SLO}}$, and the goodput-optimal partition-selection sweep — is the subject of `slo.md`. This section establishes the goodput definition; `slo.md` builds the partition-feasibility analysis on top of it.

---

<div style="page-break-before: always;"></div>

# 2. TTFT Assembly

TTFT varies substantially depending on the serving architecture (co-located vs. disaggregated), the prefill batching policy (single request vs. batched vs. chunked), and whether the system is under load. This section assembles the full TTFT formula for each scenario, building on the per-phase models derived in `prefill.md` and `framework.md`.

---

## 2.1 Single-Request TTFT

### Phase decomposition

For a single request on a **co-located** prefill+decode cluster (no disaggregation), TTFT decomposes into four sequential phases:

1. **$t_{\text{sched}}$** — Request scheduling and batch assembly latency: the serving scheduler assigns KV memory pages, builds the batch metadata tensor, and triggers the first CUDA kernel (or graph replay). From `framework.md §2.4`, $t_{\text{sched}}$ is empirical and typically 10–200 µs depending on batch size and scheduler implementation.

2. **$t_{\text{tok}}$** — Tokenization latency: raw text is converted to token IDs on CPU. From `framework.md §2.1`, $t_{\text{tok}} \sim 0.1\text{–}2$ ms and is often pipelined with the previous request's decode tail, making it negligible in steady-state batch serving. We retain it for single-request analysis.

3. **$t_{\text{prefill}}$** — Prefill forward pass latency: the model processes all $S_{\text{input}}$ tokens in a single GEMM-dominated pass. The full derivation is in `prefill.md §3.4`; the boxed result is:

   $$
   t_{\text{prefill}} = t_{\text{prefill,local}} + \max\left(0,\; t_{\text{prefill,comm}} - \rho\, t_{\text{prefill,local}}\right) + t_{\text{pipeline,warmup}} + t_{\text{LM,prefill,hw}}
   $$

   where $t_{\text{prefill,local}}$ is the per-stage roofline time including SW composition (`prefill.md §3.4`), $t_{\text{prefill,comm}}$ is the collective communication time during prefill (same TP/EP/SP structure as decode, scaled by $S_{\text{input}}$), $\rho$ is the overlap factor (same as in decode, `decode.md §6.2`), $t_{\text{pipeline,warmup}} = (PP - 1) \times t_{\text{stage,max}}$ is the time for the prefill pass to fill the pipeline (`prefill.md §3.3`), and $t_{\text{LM,prefill,hw}}$ is the once-per-pass LM head $H \to V$ projection on stage $PP{-}1$ (`prefill.md §3.4`) — added outside the warmup since it fires once at the end of the traversal, not per stage.

4. **$t_{\text{step,user}}$** — First decode step: one forward pass of the decode kernel, generating token 1. From `decode.md §7.3`:

   $$
   t_{\text{step,user}} = \gamma_{\text{pp}} \cdot \bigl[ t_{\text{stage,hw}} + \max\!\bigl(0,\; t_{\text{stage,kernel}} - \rho_{\text{kernel}} \cdot t_{\text{stage,hw}}\bigr) \bigr] + t_{\text{LM,hw}}
   $$

   This is the same formula as the steady-state TPOT (§1.2) — the first decode step is just the first invocation of the per-step decode kernel.

### Boxed result (co-located)

Combining all four phases, and absorbing $t_{\text{tok}}$ into $t_{\text{sched}}$ as a lumped scheduling overhead:

$$
TTFT_{\text{single}} = t_{\text{sched}} + t_{\text{prefill}} + t_{\text{step,user}}
$$

> **Simplification note:** $t_{\text{tok}}$ (tokenization) from the general TTFT definition in §1.1 is absorbed into $t_{\text{sched}}$ here as a lumped scheduling overhead — both are CPU-side pre-compute latencies in the 0.1–2 ms range. The general form $TTFT = t_{\text{sched}} + t_{\text{tok}} + t_{\text{prefill}} + t_{\text{KV-transfer}} + t_{\text{step,user}}$ (§1.1) is exact; the boxed formula above is a simplified co-located single-request variant.

### With disaggregated prefill

When prefill and decode run on **separate** GPU clusters [DISAGG-PREFILL], the KV cache generated by the prefill cluster must be transferred over the inter-cluster interconnect before decode can begin. Using the α–β latency model (`framework.md §3`):

$$
t_{\text{KV-transfer}} = \alpha_{\text{inter}} + \frac{M_{\text{KV-transfer}}}{BW_{\text{inter}}}
$$

where $M_{\text{KV-transfer}} = \frac{2 \cdot S_{\text{input}} \cdot H_{kv} \cdot b}{TP \cdot SP} \cdot \frac{L}{PP}$ is the per-device KV transfer volume. TTFT becomes:

$$
TTFT_{\text{disagg}} = t_{\text{sched}} + t_{\text{prefill}} + t_{\text{KV-transfer}} + t_{\text{step,user}}
$$

The motivation for disaggregation is that the prefill and decode phases have fundamentally different computational characteristics (compute-bound vs. memory-bound) and therefore benefit from different hardware configurations. The cost is the added $t_{\text{KV-transfer}}$ latency. For large KV caches and slow inter-cluster links, this can be the dominant TTFT term.

---

## 2.2 Batched-Prefill TTFT

When $B_{\text{prefill}} > 1$ requests are prefilled together in a single forward pass, the computational structure changes in an asymmetric way: **FLOPs scale with $B_{\text{prefill}}$, but weight traffic does not.**

### FLOPs under batched prefill

Each request contributes $S_{\text{input}}$ tokens to the joint prefill batch of size $B_{\text{prefill}} \times S_{\text{input}}$ (assuming equal-length prompts for simplicity). The projection and FFN GEMMs have shape $[B_{\text{prefill}} \cdot S_{\text{input}} \times H] \times [H \times H_{\text{out}}]$, so FLOPs scale linearly:

$$
F_{\text{prefill,batch}} = B_{\text{prefill}} \times F_{\text{prefill,single}}
$$

where $F_{\text{prefill,single}}$ is the per-request FLOPs from `prefill.md §1`.

### Weight traffic under batched prefill

The weight matrices are read **once** from HBM regardless of $B_{\text{prefill}}$, because all $B_{\text{prefill}}$ sequences share the same weights in the GEMM. Weight traffic is therefore invariant in $B_{\text{prefill}}$:

$$
T_{\theta,\text{device}}(B_{\text{prefill}}) = T_{\theta,\text{device}} \qquad (\text{independent of } B_{\text{prefill}})
$$

This is precisely the batching benefit: more FLOPs are executed per byte of weight loaded, improving arithmetic intensity and GPU utilization.

### Batched prefill local time

Generalizing `prefill.md §3.1` to batch size $B_{\text{prefill}}$:

$$
t_{\text{prefill,local}}(B_{\text{prefill}}) =
\max\left(
\frac{B_{\text{prefill}} \times F_{\text{prefill,device}}}{R_{\text{GPU}}},\;
\frac{T_{\theta,\text{device}} + B_{\text{prefill}} \times T_{\text{KV,write,device}}}{BW_{\text{mem}}}
\right)
$$

The KV write traffic $T_{\text{KV,write,device}}$ scales with $B_{\text{prefill}}$ because each request in the batch writes its own KV entries. For large $B_{\text{prefill}}$, the compute term dominates and the latency grows linearly with $B_{\text{prefill}}$; at small $B_{\text{prefill}}$ in the compute-bound prefill regime ($S_{\text{input}} \gg S_{\text{input}}^{\star}$, see `prefill.md §2.3`), compute is already the bottleneck even at $B_{\text{prefill}} = 1$, so the latency likewise scales linearly.

### TTFT for the last request in the batch

From a user's perspective, the worst-case TTFT applies to the **last request** admitted to the prefill batch — that request waits for the entire joint prefill to complete before its first token is produced:

$$
TTFT_{\text{batched}} = t_{\text{sched}} + t_{\text{prefill,local}}(B_{\text{prefill}}) + \max\left(0,\; t_{\text{prefill,comm}} - \rho\, t_{\text{prefill,local}}\right) + t_{\text{pipeline,warmup}} + t_{\text{LM,prefill,hw}}(B_{\text{prefill}}) + t_{\text{step,user}}
$$

where $t_{\text{LM,prefill,hw}}(B_{\text{prefill}})$ scales with **only $B_{\text{prefill}}$** (one $H \to V$ projection per request, last position only — see `prefill.md §1.5`), not with $B_{\text{prefill}} \cdot S_{\text{input}}$.

Batching prefill requests improves GPU utilization (higher arithmetic intensity → better hardware efficiency) at the cost of increased TTFT for late arrivals in the batch. The optimal $B_{\text{prefill}}$ balances throughput and tail TTFT latency; see `prefill.md §4` for the batch-size optimization.

---

## 2.3 Chunked-Prefill TTFT

Chunked prefill [SARATHI] splits the prefill of a single request into $N_{\text{chunks}}$ smaller chunks of $C$ tokens each, interleaving each chunk with one decode iteration. This technique reduces **head-of-line blocking**: long prefill passes in standard serving can stall the decode pipeline for tens to hundreds of milliseconds, inflating TPOT for all existing decode requests. Chunking limits the maximum stall duration per decode iteration to $t_{\text{chunk}}$, the time to process one $C$-token chunk.

### Chunk count

For a prompt of length $S_{\text{input}}$ with chunk size $C$:

$$
N_{\text{chunks}} = \left\lceil \frac{S_{\text{input}}}{C} \right\rceil
$$

### Per-chunk latency

Each chunk is a mini-prefill of $C$ tokens. The per-chunk local time follows the same roofline formula as a full prefill (`prefill.md §3.1`) with $S_{\text{input}}$ replaced by $C$:

$$
t_{\text{chunk}} = \max\left(\frac{F_{\text{prefill,device}}(C)}{R_{\text{GPU}}},\; \frac{T_{\theta,\text{device}} + T_{\text{KV,write,device}}(C)}{BW_{\text{mem}}}\right)
$$

For small $C$ (e.g., $C = 256$), the chunk is likely memory-bound (weight traffic dominates KV write traffic), meaning $t_{\text{chunk}} \approx T_{\theta,\text{device}} / BW_{\text{mem}}$ — identical to one decode step latency. This is the **chunked-prefill design point**: each chunk costs approximately the same as one decode step, which is the minimum possible disruption to the decode pipeline.

### TTFT under chunking

The new request's TTFT is the time to process all $N_{\text{chunks}}$ chunks sequentially (each occupying one decode slot), plus scheduling overhead:

$$
TTFT_{\text{chunked}} = t_{\text{sched}} + N_{\text{chunks}} \times t_{\text{chunk}} + t_{\text{LM,prefill,hw}} + t_{\text{step,user}}
\approx t_{\text{sched}} + \left\lceil \frac{S_{\text{input}}}{C} \right\rceil \times t_{\text{chunk}} + t_{\text{LM,prefill,hw}} + t_{\text{step,user}}
$$

The pipeline warmup $t_{\text{pipeline,warmup}}$ applies once across the entire prefill sequence rather than per chunk (the pipeline is kept warm by the ongoing decode traffic), so it does not multiply by $N_{\text{chunks}}$. Likewise, the LM head $t_{\text{LM,prefill,hw}}$ fires **once** after the last chunk produces the final position's hidden state — not once per chunk (`prefill.md §3.4`).

### Tradeoff

| Smaller $C$ | Larger $C$ |
|-------------|------------|
| Less stall per decode iteration → better TPOT for existing requests | More stall per iteration → worse TPOT for existing requests |
| More chunks required → worse TTFT for new request | Fewer chunks → better TTFT for new request |
| Approaches standard decode behavior | Approaches standard (non-chunked) prefill behavior |

The optimal $C$ depends on the SLA balance between new-request TTFT and existing-request TPOT. A common production heuristic is $C \approx B \cdot S_{\text{decode}}$, matching the chunk FLOPs to the concurrent decode work so that the GPU is equally utilized during chunk and decode steps [SARATHI].

---

<div style="page-break-before: always;"></div>

# 3. TPOT Assembly

TPOT is the per-sequence inter-token latency during the decode phase. Its derivation from the roofline model was developed in detail in `decode.md §7.3`; this section assembles the result for both static and continuous batching, and explains the steady-state behavior under each scheduling policy.

---

## 3.1 Static Batching TPOT

In **static batching**, all $B$ requests in the batch start together, are padded to the same output length, and finish together. The batch composition is fixed for the entire decode session.

### Per-step wall-clock time

At each decode step, the model processes $B$ tokens simultaneously (one per sequence). The overlap-aware per-stage HW step time from `decode.md §6.2` is:

$$
t_{\text{stage,hw}}(B) = t_{\text{local}}(B) + \max\left(0,\; t_{\text{comm}}(B) - \rho \cdot t_{\text{local}}(B)\right)
$$

where the batched local time is (`decode.md §7.3`):

$$
t_{\text{local}}(B) =
\max\left(
\frac{B \times F_{\text{token,device}}}{R_{\text{GPU}}},\;
\frac{T_{\theta,\text{device}} + B \times T_{\text{KV,device}}}{BW_{\text{mem}}}
\right)
$$

Composing with the per-stage kernel-launch dispatch budget $t_{\text{stage,kernel}}$ (`decode.md §7.1`), applying the pipeline bubble correction $\gamma_{\text{pp}} = \max(1, PP/B)$, and adding the once-per-step LM head $t_{\text{LM,hw}}(B)$ on stage $PP{-}1$ (`decode.md §6.2 / §7.2`), then composing per-sequence serving overhead (`decode.md §7.3`) gives the user-observed step time:

$$
t_{\text{step,user}}(B) = \gamma_{\text{pp}} \cdot \bigl[ t_{\text{stage,hw}}(B) + \max\!\bigl(0,\; t_{\text{stage,kernel}} - \rho_{\text{kernel}} \cdot t_{\text{stage,hw}}(B)\bigr) \bigr] + t_{\text{LM,hw}}(B)
$$

### TPOT definition

Each decode step emits exactly **one token per active sequence** and takes $t_{\text{step,user}}(B)$ wall-clock seconds from the user's perspective. A given sequence's new token appears once per step, so TPOT equals the full step time — **not** $t_{\text{stage}}/B$ (which would be the amortized per-token cost across all sequences, i.e., the inverse of throughput-per-step, a throughput metric rather than a latency metric):

$$
\text{TPOT}_{\text{static}}(B) = t_{\text{step,user}}(B)
$$

Consistency with §1.2 and `decode.md §7.3`: the user observes one token per decode step per sequence, and the step time is set by the slowest pipeline stage (plus any bubble for $B < PP$). The throughput metric $B/t_{\text{step,user}}$ (tokens/s across the replica) is what is divided by $N_{\text{GPUs}}$ to get Throughput/GPU.

### Regime behavior

From the batch-size analysis in `decode.md §7.3` (assuming $B \ge PP$ so bubble factor is 1):

**Memory-bound regime** ($B \ll B^*$, weight traffic dominates):
$$
t_{\text{stage}}(B) \approx \frac{T_{\theta,\text{device}}}{BW_{\text{mem}}} \quad\Rightarrow\quad \text{TPOT}_{\text{static}}(B) \approx \frac{T_{\theta,\text{device}}}{BW_{\text{mem}}}
$$

TPOT is approximately **flat in $B$**: weights stream once per step regardless of $B$, and step time stays pinned at the weight-streaming cost. Cluster throughput grows with $B$ (more tokens per step) while user-observed latency stays constant.

**Compute-bound regime** ($B \gg B^*$, compute dominates):
$$
t_{\text{stage}}(B) \approx \frac{B \times F_{\text{token,device}}}{R_{\text{GPU}}} \quad\Rightarrow\quad \text{TPOT}_{\text{static}}(B) \approx \frac{B \times F_{\text{token,device}}}{R_{\text{GPU}}}
$$

TPOT grows **linearly with $B$**: once compute saturates, every added sequence extends the step time and every user sees the extension.

**Summary:**

| Regime | Condition | $\text{TPOT}_{\text{static}}(B)$ |
|--------|-----------|----------------------------------|
| Memory-bound | $B \ll B^*$ | $\approx T_{\theta,\text{device}} / BW_{\text{mem}}$ (flat in $B$) |
| Crossover | $B = B^*$ | knee of the throughput–latency Pareto curve |
| Compute-bound | $B \gg B^*$ | $\approx B \cdot F_{\text{token,device}} / R_{\text{GPU}}$ (linear in $B$) |

---

## 3.2 Continuous Batching TPOT

In **continuous batching** [VLLM], requests arrive and depart asynchronously. At each decode iteration, the scheduler assembles a new batch from active requests; requests that complete (reach their stop token or maximum length) are immediately removed and new requests admitted. The effective batch size $B_{\text{eff}}$ therefore varies from iteration to iteration.

### Per-request average TPOT

A request that requires $N_{\text{out}}$ decode steps experiences a different $B_{\text{eff},i}$ at each step $i$. Using $\text{TPOT}(B) = t_{\text{step,user}}(B)$ (§1.2, `decode.md §7.3`), the average TPOT over the full response is the mean step time:

$$
\overline{\text{TPOT}} = \frac{1}{N_{\text{out}}} \sum_{i=1}^{N_{\text{out}}} t_{\text{step,user}}(B_{\text{eff},i})
$$

$t_{\text{step,user}}(B)$ is flat in the memory-bound zone and grows linearly with $B$ in the compute-bound zone (see §3.1). Therefore:

- Under loads where $\overline{B_{\text{eff}}} \ll B^*$: $\overline{\text{TPOT}} \approx t_{\text{step,user}}(\overline{B_{\text{eff}}})$ (stable in load).
- Under loads where $\overline{B_{\text{eff}}} \gtrsim B^*$: $\overline{\text{TPOT}}$ tracks the mean compute load and grows with $\overline{B_{\text{eff}}}$.

### Steady-state effective batch size (Little's Law)

In steady state under Poisson request arrivals at rate $\lambda$ (requests/second), **Little's Law** relates the mean number of active requests $\overline{B_{\text{eff}}}$ to the mean sojourn time:

$$
\overline{B_{\text{eff}}} = \lambda \times \mathbb{E}[\text{sojourn time}]
$$

The sojourn time for a single request is the total wall-clock time it occupies a decode slot: $TTFT + \sum_{i=1}^{N_{\text{out}}} t_{\text{step,user}}(B_{\text{eff},i})$. Since the user-observed step time is the TPOT, the mean sojourn time is:

$$
\overline{B_{\text{eff}}} \approx \lambda \times \left(TTFT + \mathbb{E}[N_{\text{out}}] \times \overline{\text{TPOT}}\right)
\qquad \text{where } \overline{\text{TPOT}} = \mathbb{E}[t_{\text{step,user}}(B_{\text{eff}})]
$$

This is self-referential ($\overline{\text{TPOT}}$ depends on $\overline{B_{\text{eff}}}$) and resolved numerically in practice:

- At **low load** ($\lambda$ small): $\overline{B_{\text{eff}}}$ is small → each step is memory-bound → TPOT $\approx T_\theta/BW_{\text{mem}}$ (flat; weight-streaming cost regardless of $B$). Throughput is low because $B$ is small, but per-user latency is already at the memory-bound floor.
- At **higher load** ($\lambda$ growing): $\overline{B_{\text{eff}}}$ grows → throughput grows nearly linearly while TPOT stays flat, until the batch crosses $B^*$.
- At **saturation** ($B_{\text{eff}} > B^*$): throughput plateaus at $R_{\text{GPU}}/F_{\text{token,device}}$; TPOT grows linearly with $B_{\text{eff}}$ from that point on.

### The efficiency–load relationship

With TPOT defined as the user-observed step time ($t_{\text{step,user}}$), **higher load improves throughput at constant TPOT** through Zone 1, then trades TPOT for throughput beyond $B^*$. The throughput axis saturates at a hard compute ceiling; the TPOT axis has no ceiling and grows linearly past $B^*$. The throughput–latency identity $\text{Tput/GPU} \times \text{TPOT} = B / N_{\text{GPUs,per-replica}}$ (§6.3) is linear in $B$: in Zone 1, $B$ rises while $\text{TPOT}$ stays at the memory-bound floor (so $\text{Tput/GPU}$ rises proportionally); in Zone 3, $\text{Tput/GPU}$ pins at the compute ceiling while $\text{TPOT}$ grows with $B$. This tradeoff is developed in full in §6.

---

<div style="page-break-before: always;"></div>

# 4. When Prefill Dominates Decode

For a response of $N_{\text{out}}$ output tokens, the total decode contribution to wall-clock latency is $(N_{\text{out}} - 1) \times \text{TPOT}$ — token 1 is already produced by the end of TTFT, and tokens $2, \ldots, N_{\text{out}}$ each cost one TPOT. The relative weight of prefill (captured in TTFT) versus decode (captured in the TPOT term) depends on $N_{\text{out}}$.

**Short responses** ($N_{\text{out}}$ small): the decode contribution $(N_{\text{out}} - 1) \times \text{TPOT}$ is small relative to TTFT. Latency is **prefill-dominated**; reducing $S_{\text{input}}$ or using chunked prefill has the largest impact.

**Long responses** ($N_{\text{out}}$ large): the prefill contribution is amortized over many decode steps; TTFT contributes negligibly. Latency is **decode-dominated**; increasing GPU count (DP replicas) or batch efficiency has the largest impact.

**Crossover point.** The two contributions are equal when:

$$
(N_{\text{out}} - 1) \times \text{TPOT} = TTFT
\quad\Longrightarrow\quad
N_{\text{out}}^{\star} \approx \frac{TTFT}{\text{TPOT}} + 1
$$

For $N_{\text{out}} \ll N_{\text{out}}^{\star}$, the response is prefill-dominated; for $N_{\text{out}} \gg N_{\text{out}}^{\star}$, decode-dominated.

**Numerical example (H100, 70B-class dense, $S_{\text{input}} = 2048$, $B = 1$):**

- $t_{\text{prefill}} \approx 100$ ms (compute-bound prefill at $S_{\text{input}} = 2048$; from `prefill.md §3`)
- $\text{TPOT} \approx 20$ ms (memory-bound decode at $B = 1$; from $T_{\theta} / BW_{\text{mem}}$ with 70B weights at bf16 and 3.35 TB/s HBM)
- $t_{\text{sched}} \approx 0.1$ ms; $t_{\text{step,user}} \approx 20$ ms (first decode step ≈ TPOT at $B = 1$)

$TTFT \approx 100 + 20 = 120$ ms gives $N_{\text{out}}^{\star} = 120/20 + 1 = 7$ tokens. For conversational responses ($N_{\text{out}} = 50$–$250$), TTFT contributes 2–11% of the total streaming time and decode dominates; at $N_{\text{out}} = 1000$, TTFT is well under 1%. The takeaway: **TTFT optimization matters most for very short, latency-sensitive responses** (under ~$N_{\text{out}}^{\star}$ tokens), while **TPOT optimization dominates everywhere else**.

---

<div style="page-break-before: always;"></div>

# 5. Throughput/GPU and Interactivity

## 5.1 Throughput/GPU Derivation

From `decode.md §7.3`, the global decode throughput across all DP replicas is:

$$
TTPS(B) = DP \cdot TPS_{\text{single}}(B) = DP \cdot \frac{B}{t_{\text{step,user}}(B)}
$$

where $t_{\text{step,user}}(B)$ is the full per-step decode time including SW composition, pipeline bubble, and LM head (§1.2 / `decode.md §7.3`). Each step emits $B$ tokens (one per active sequence), so the per-replica rate is $B / t_{\text{step,user}}$ tokens/s.

The total GPU count in the cluster is:

$$
N_{\text{GPUs}} = DP \cdot PP \cdot TP \cdot EP \cdot SP
$$

Substituting into the Tput/GPU definition:

$$
\text{Tput/GPU}
= \frac{TTPS(B)}{N_{\text{GPUs}}}
= \frac{DP \cdot TPS_{\text{single}}(B)}{DP \cdot PP \cdot TP \cdot EP \cdot SP}
$$

$$
\text{Tput/GPU} = \frac{TPS_{\text{single}}(B)}{PP \cdot TP \cdot EP \cdot SP} = \frac{B}{t_{\text{step,user}}(B) \cdot PP \cdot TP \cdot EP \cdot SP}
$$

### Insight: DP is the only "free" dimension

Data Parallelism ($DP$) cancels from both numerator and denominator: adding DP replicas multiplies $TTPS$ and $N_{\text{GPUs}}$ equally, leaving Tput/GPU unchanged. The DP dimension scales aggregate throughput without affecting per-GPU efficiency.

Pipeline ($PP$), Tensor ($TP$), Expert ($EP$), and Sequence ($SP$) parallelism all appear in the denominator only. They expand the denominator because each added GPU reduces per-device FLOPs and memory traffic but introduces communication overhead — the per-stage $t_{\text{stage},j}$ decreases with more GPUs, but communication overhead partially offsets this gain, and more total GPU-seconds are consumed per token generated. Therefore:

- **DP** is the preferred dimension for scaling cluster throughput without TPOT impact.
- **TP/PP/EP/SP** are used to fit large models into available HBM, but each carries a Tput/GPU cost (more GPUs per token, with communication overhead).

---

## 5.2 Interactivity (per-request streaming rate)

From §1.4, Interactivity = $1/\text{TPOT}$. From §3.1, $\text{TPOT}(B) = t_{\text{step,user}}(B)$ (no division by $B$ — every step emits one token per user, so a single user waits one full step for their next token regardless of how many other users share the batch):

$$
\text{Interactivity}(B) = \frac{1}{t_{\text{step,user}}(B)}
$$

The factor of $B$ does **not** appear in the numerator: each decode step produces $B$ tokens — one for each of the $B$ active users — so $1/t_{\text{step,user}}(B)$ is already the per-user streaming rate. The aggregate per-replica rate $B / t_{\text{step,user}}(B)$ is $TPS_{\text{single}}$ (§5.1), a throughput quantity, not interactivity.

**At the memory-bound operating point** ($B \ll B^*$, $t_{\text{step,user}} \approx T_{\theta,\text{device}}/BW_{\text{mem}}$):
$$
\text{Interactivity}(B) \approx \frac{BW_{\text{mem}}}{T_{\theta,\text{device}}} \qquad (\text{flat in } B)
$$

Interactivity is **constant in $B$** in the memory-bound regime: weights stream once per step regardless of $B$, so per-user latency stays pinned at the weight-streaming floor. Adding more sequences grows Tput/GPU at no cost to per-user latency, but each user's streaming rate does **not** improve.

**At the compute-bound operating point** ($B \gg B^*$, $t_{\text{step,user}} \approx B \times F_{\text{token,device}} / R_{\text{GPU}}$):
$$
\text{Interactivity}(B) \approx \frac{R_{\text{GPU}}}{B \times F_{\text{token,device}}} \quad\propto\quad \frac{1}{B}
$$

Interactivity **degrades as $1/B$** in the compute-bound regime: each added sequence extends the step time, slowing every user proportionally. Tput/GPU saturates at the compute ceiling while per-user latency continues to worsen with $B$.

**Human reading threshold.** For streaming LLM output to feel "live" to a human reader, interactivity above approximately 5–15 tokens/s (TPOT below 67–200 ms) is required. This maps to a specific constraint on $t_{\text{step,user}}$ — a maximum allowable step time — which, through the roofline model, constrains the maximum batch size $B$ that can be served while meeting the SLA.

---

<div style="page-break-before: always;"></div>

# 6. Throughput–Latency Pareto Frontier

The Pareto frontier is the set of (Tput/GPU, Interactivity) pairs that are achievable without waste: no point on the frontier can improve both metrics simultaneously. This section assembles the frontier from the roofline model, identifies the three operating zones, derives the throughput–latency identity, and maps the result onto the standard throughput–interactivity axes used by production LLM inference benchmarks.

> **Terminology note:** In this model, Tput/GPU rises with $B$ throughout the memory-bound regime while per-user Interactivity stays pinned at the weight-streaming floor; in the compute-bound regime, Tput/GPU saturates while Interactivity falls as $1/B$. The frontier traced by sweeping $B$ is therefore an L-shape (flat-top, vertical-right-edge) rather than a hyperbola — the classical multi-objective tradeoff appears only past $B^*$. We use "Pareto frontier" in the ML systems sense — the set of efficient operating points.

---

## 6.1 Levers Shaping the Pareto Curve

Three principal levers shift the operating point along the Pareto frontier, or move the frontier itself:

1. **Batch size $B$ (or $B_{\text{eff}}$ under continuous batching).** Moving along the frontier. Increasing $B$ in the memory-bound regime ($B < B^*$) raises Tput/GPU at no cost to Interactivity (which sits at the weight-streaming floor). Increasing $B$ past $B^*$ saturates Tput/GPU and starts degrading Interactivity as $1/B$. Decreasing $B$ in the compute-bound regime improves Interactivity at the cost of lower Tput/GPU; decreasing $B$ within Zone 1 lowers Tput/GPU without buying any per-user latency improvement.

2. **Parallelism configuration (TP, PP, EP, SP; fixed DP).** Shifting the frontier. TP/PP/EP/SP change $t_{\text{step,user}}(B)$ by altering per-device FLOPs, traffic, and communication overhead. They also change $B^*$ (the crossover batch size from `decode.md §4`). Better parallelism configurations can push the frontier outward (higher Tput/GPU for the same Interactivity).

3. **Context length $S$.** Shifting the frontier inward. Longer decode contexts $S$ increase KV cache traffic $T_{\text{KV,device}}$, which lowers $B^*$ (the system becomes compute-bound at smaller batches) and increases $t_{\text{step,user}}$ for all $B$, shrinking both Tput/GPU and Interactivity simultaneously.

---

## 6.2 Three Zones of the Pareto Frontier

The Pareto curve sweeps $B$ from $1$ to $\infty$. We derive its shape here in the (Tput/GPU, Interactivity) coordinate system used by throughput–latency benchmark plots, building on the per-step roofline from `decode.md §4`.

Define Tput/GPU and Interactivity as explicit functions of $B$:

$$
\text{Tput/GPU}(B) = \frac{B}{t_{\text{step,user}}(B) \times N_{\text{GPUs,per-replica}}}
$$

$$
\text{Interactivity}(B) = \frac{1}{t_{\text{step,user}}(B)}
$$

where $N_{\text{GPUs,per-replica}} = PP \cdot TP \cdot EP \cdot SP$ is the number of GPUs per DP replica. (Tput/GPU has a $B$ in the numerator because it counts all $B$ tokens emitted per step across the replica; Interactivity does not, because each user receives just one of those tokens per step — see §5.2.)

### Zone 1 — Memory-bound ($B < B^*$)

Weight traffic dominates the per-step memory term:

$$
t_{\text{step,user}}(B) \approx \frac{T_{\theta,\text{device}}}{BW_{\text{mem}}}
\qquad (B \ll B^*)
$$

The step time is approximately **constant** in $B$. Therefore:

$$
\text{Tput/GPU}(B) \approx \frac{B \cdot BW_{\text{mem}}}{T_{\theta,\text{device}} \cdot N_{\text{GPUs,per-replica}}} \quad\propto\quad B
$$

$$
\text{Interactivity}(B) = \frac{1}{t_{\text{step,user}}(B)} \approx \frac{BW_{\text{mem}}}{T_{\theta,\text{device}}} \qquad (\text{flat in } B)
$$

**Zone 1 behavior:** Tput/GPU grows linearly with $B$ while Interactivity stays at the weight-streaming floor. This is the **free-lunch zone for throughput**: weight loads amortize over more sequences, so aggregate efficiency improves at no cost to per-user latency. Per-user streaming speed does *not* improve here — every user already waits one full weight-streaming step per token, and adding sequences cannot make that wait shorter.

### Zone 2 — Crossover ($B \approx B^*$)

The system operates near the ridge point. Tput/GPU is approaching its compute ceiling; Interactivity is leaving the memory-bound floor. This is the **knee** of the curve — the highest throughput attainable before per-user latency starts degrading, and the natural SLA operating point for production deployments balancing GPU utilization with latency SLAs.

### Zone 3 — Compute-bound ($B > B^*$)

KV cache traffic (or equivalently, compute) dominates the per-step roofline:

$$
t_{\text{step,user}}(B) \approx \frac{B \times F_{\text{token,device}}}{R_{\text{GPU}}}
\qquad (B \gg B^*)
$$

The step time grows linearly with $B$. Therefore:

$$
\text{Tput/GPU}(B) \approx \frac{R_{\text{GPU}}}{F_{\text{token,device}} \cdot N_{\text{GPUs,per-replica}}} \qquad (\text{constant in } B)
$$

$$
\text{Interactivity}(B) = \frac{1}{t_{\text{step,user}}(B)} \approx \frac{R_{\text{GPU}}}{B \times F_{\text{token,device}}} \quad\propto\quad \frac{1}{B}
$$

**Zone 3 behavior:** Throughput plateaus at the compute ceiling, but Interactivity now **degrades as $1/B$**: every added sequence extends the step time and slows every user proportionally. The Pareto curve turns down — past $B^*$, adding more $B$ buys nothing on the throughput axis and actively hurts per-user latency.

### Summary table

| Zone | Condition | $\text{Tput/GPU}(B)$ | $\text{Interactivity}(B)$ |
|------|-----------|----------------------|---------------------------|
| 1 — Memory-bound | $B \ll B^*$ | $\propto B$ (growing) | $\approx BW_{\text{mem}} / T_{\theta,\text{device}}$ (flat) |
| 2 — Crossover | $B \approx B^*$ | near compute ceiling | leaving memory-bound floor |
| 3 — Compute-bound | $B \gg B^*$ | $\approx R_{\text{GPU}} / (F \cdot N)$ (flat) | $\approx R_{\text{GPU}} / (B \cdot F)$ ($\propto 1/B$, degrading) |

---

## 6.3 Throughput–Latency Identity

A useful identity emerges when we compute the product $\text{Tput/GPU} \times \text{TPOT}$ at a given operating point. Using $\text{Tput/GPU}(B) = B / (t_{\text{step,user}}(B) \cdot N_{\text{GPUs,per-replica}})$ and $\text{TPOT}(B) = t_{\text{step,user}}(B)$ (§1.2 / §3.1, no division by $B$):

$$
\text{Tput/GPU}(B) \times \text{TPOT}(B)
= \frac{B}{t_{\text{step,user}}(B) \cdot N_{\text{GPUs,per-replica}}} \times t_{\text{step,user}}(B)
= \frac{B}{N_{\text{GPUs,per-replica}}}
$$

The $t_{\text{step,user}}$ terms cancel, leaving:

$$
\text{Tput/GPU} \times \text{TPOT} = \frac{B}{N_{\text{GPUs,per-replica}}}
$$

Unlike a true ridge-style ceiling, this product is **not constant** — it scales linearly with $B$. The intuition: at a fixed parallelism configuration, every additional sequence adds one token per step to the per-replica output (raising $\text{Tput/GPU}$) while also extending the per-step time in Zone 3 (raising $\text{TPOT}$); both effects compound into the linear-in-$B$ product.

**What this identity does and does not say.**

- *Does say* — given any two of $\{\text{Tput/GPU},\, \text{TPOT},\, B\}$ and the parallelism configuration $N_{\text{GPUs,per-replica}}$, the third is fixed. This is a powerful operational consistency check on benchmark numbers.
- *Does not say* — that the curve is a hyperbola, or that $B$ alone trades $\text{Tput/GPU}$ for $\text{Interactivity}$. The actual Pareto curve has the L-shape described in §6.2: a **flat top** (Zone 1, Tput/GPU rises while Interactivity sits at the memory-bound floor) and a **vertical right edge** (Zone 3, Tput/GPU saturates while Interactivity falls as $1/B$).

**Hardware-model co-design implication.** Reducing $N_{\text{GPUs,per-replica}}$ — i.e. fitting the model with less inner sharding (lower $PP \cdot TP \cdot EP \cdot SP$) — lifts $\text{Tput/GPU}$ at every $\text{TPOT}$ along the Pareto curve. The objective stays the same as before: **minimize $N_{\text{GPUs,per-replica}}$ subject to fitting the model in HBM**.

---

## 6.4 Throughput–Interactivity Axis Mapping

Production LLM inference benchmarks typically organize performance on a two-axis scatter plot:

- **X-axis: Throughput/GPU** — output tokens per second per GPU
- **Y-axis: Interactivity** — output tokens per second per request ($= 1/\text{TPOT}$)

A system at a single operating point (fixed hardware, model, parallelism, and batch size) appears as one point on this plot. As batch size $B$ varies:

- **Zone 1** (increasing $B$ from 1 toward $B^*$): Tput/GPU grows linearly while Interactivity stays flat at the memory-bound floor — the operating point moves **rightward** along a horizontal line. This is the free-lunch zone for *throughput*: weight traffic amortizes across more sequences, raising aggregate efficiency at no cost to per-user latency.
- **Zone 2** (near $B^*$): Tput/GPU approaches its compute ceiling; Interactivity begins to leave the memory-bound floor — the **knee** of the curve.
- **Zone 3** (beyond $B^*$): Tput/GPU plateaus while Interactivity drops as $1/B$ — the operating point moves **straight down** along a vertical line. Adding $B$ no longer helps throughput and actively hurts per-user latency.

The Pareto frontier traced by sweeping $B$ from 1 to $\infty$ therefore has an **L-shape** (rotated):

1. **Zone 1 segment**: horizontal — Tput/GPU rises rightward, Interactivity sits at the memory-bound floor.
2. **Zone 2 knee**: the corner of the L, where Tput/GPU saturates and Interactivity starts falling.
3. **Zone 3 segment**: vertical — Tput/GPU stays at its ceiling, Interactivity falls as $1/B$.

**The ideal operating point** is at or near $B^*$: the corner of the L, where Tput/GPU has just reached its ceiling and Interactivity has not yet started degrading.

Different hardware configurations (H100 vs. A100), model sizes (7B vs. 70B vs. 405B), and parallelism choices shift the L outward or inward. A 3D-stacked accelerator with higher HBM bandwidth [ACCELSTACK] increases $BW_{\text{mem}}$, which raises the Zone 1 floor (per-user interactivity floor) and pushes $B^*$ higher — extending the horizontal Zone 1 segment to the right before Tput/GPU saturates.

The §6.3 identity — $\text{Tput/GPU} \times \text{TPOT} = B / N_{\text{GPUs,per-replica}}$ — constrains every operating point in $B$: given any two of $\{\text{Tput/GPU},\, \text{TPOT},\, B\}$, the third is determined by the parallelism configuration. Systems with smaller $N_{\text{GPUs,per-replica}}$ (more aggressive model sharding) achieve higher Tput/GPU at every $(\text{TPOT}, B)$ point — a better hardware-efficiency Pareto curve.

---

<div style="page-break-before: always;"></div>

# Symbol Summary

The following symbols are introduced in this document and are **not** defined in `notation.md §14` (which refers to this document for their definitions). All other symbols used here are defined in `notation.md` and cross-referenced to their home sections.

| Symbol | Definition | First used |
|--------|-----------|-----------|
| $N_{\text{out}}$ | Number of output tokens in a single response (decode length) | §1.2 |
| $N_{\text{out}}^{\star}$ | Crossover output length at which TTFT = decode contribution ($(N-1)\cdot\text{TPOT}$) | §4 |
| $C$ | Chunked-prefill chunk size (tokens per chunk) | §2.3 |
| $N_{\text{chunks}}$ | Number of prefill chunks: $\lceil S_{\text{input}} / C \rceil$ | §2.3 |
| $t_{\text{chunk}}$ | Per-chunk prefill latency (roofline at $C$ tokens) | §2.3 |
| $\lambda$ | Request arrival rate (requests/second) for continuous batching analysis | §3.2 |
| $N_{\text{GPUs,per-replica}}$ | GPUs per DP replica: $PP \cdot TP \cdot EP \cdot SP$ | §5.1 |
| $\overline{B_{\text{eff}}}$ | Mean effective batch size in steady-state continuous batching | §3.2 |
| $\overline{\text{TPOT}}$ | Average TPOT over a request's decode lifetime under continuous batching | §3.2 |
| $\text{Tput/GPU}$ | System throughput per GPU: $TTPS / N_{\text{GPUs}}$ (tokens/s/GPU) | §1.3 |
| $\text{Interactivity}$ | Per-user output rate: $1/\text{TPOT}$ (tokens/s/request) | §1.4 |
| $\text{Goodput}$ | Maximum $\lambda$ such that both TTFT and TPOT SLOs hold at percentile $p$ | §1.5 |
| $TTFT_{\text{SLO}}$ | Operator-set upper bound on TTFT (seconds) used in the goodput definition | §1.5 |
| $\text{TPOT}_{\text{SLO}}$ | Operator-set upper bound on TPOT (seconds) used in the goodput definition | §1.5 |
| $p$ | SLO compliance percentile (typically 90 or 99) | §1.5 |

The following existing symbols from `notation.md` are used extensively; they are listed here for reading convenience:

| Symbol | Defined in | Meaning |
|--------|-----------|---------|
| $TTFT$ | `notation.md §11` | Time To First Token |
| $t_{\text{prefill}}$ | `notation.md §11` | Full prefill latency (roofline + comm + pipeline warmup + LM head) |
| $t_{\text{prefill,local}}$ | `notation.md §11` | Per-stage prefill local roofline time (incl. SW composition) |
| $t_{\text{prefill,comm}}$ | `notation.md §11` | Prefill communication time |
| $t_{\text{LM,prefill,hw}}$ | `prefill.md §3.4` | LM head one-shot roofline on stage $PP{-}1$ during prefill |
| $t_{\text{step,user}}$ | `notation.md §9` | User-observed per-step decode time (HW + SW + bubble + LM head) |
| $t_{\text{stage,hw}}$ | `decode.md §6.2` | Per-stage HW step time (compute + comm + overlap) |
| $t_{\text{stage,kernel}}$ | `decode.md §7.1` | Per-stage CPU/host kernel-launch dispatch budget |
| $t_{\text{LM,hw}}$ | `decode.md §6.2` | LM head one-shot roofline on stage $PP{-}1$ during decode |
| $\gamma_{\text{pp}}$ | `decode.md §7.2` | Pipeline bubble factor $\max(1, PP/B)$ |
| $\rho_{\text{kernel}}$ | `decode.md §7.1` | Kernel-launch (CPU/GPU dispatch) overlap factor |
| $TPS_{\text{single}}$ | `notation.md §9` | Single-replica decode throughput (tokens/s) |
| $TTPS$ | `notation.md §9` | Global decode throughput (tokens/s) |
| $t_{\text{sched}}$ | `notation.md §13` | Request scheduling / batch assembly latency |
| $t_{\text{KV-transfer}}$ | `notation.md §13` | Disaggregated KV cache transfer latency |
| $B_{\text{eff}}$ | `notation.md §4` | Effective batch size under continuous batching |
| $B^*$ | `decode.md §4` | Crossover batch size (memory-bound → compute-bound) |
| $\rho$ | `notation.md §9` | Compute–communication overlap factor |

---

## References

- [VLLM] Kwon et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023. arXiv:2309.06180 — Continuous batching; PagedAttention.
- [SARATHI] Agrawal et al. (2023). *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills.* arXiv:2308.16369 — Chunked prefill scheduling; head-of-line blocking reduction.
- [DISAGG-PREFILL] Zhong et al. (2024). *DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving.* OSDI 2024. arXiv:2401.09670 — Disaggregated prefill; KV transfer latency; goodput framework.
- [ROOFLINE] Williams, Waterman & Patterson (2009). *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* CACM 52(4) — Operational intensity; ridge point.
- [H100-SPEC] NVIDIA Corporation (2022). *NVIDIA H100 Tensor Core GPU Architecture.* WP-10792-001 — H100 SXM5 specs used in numerical examples.
- [ACCELSTACK] Bai et al. (2025). *AccelStack: A Co-Design Framework for 3D-Stacked Accelerators.* HKUST FACT Lab — 3D DRAM bandwidth modeling.
- Sibling documents: [decode.md](decode.md), [prefill.md](prefill.md), [kv.md](kv.md), [framework.md](framework.md), [notation.md](notation.md), [references.md](references.md).
