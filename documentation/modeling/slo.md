# Service Level Objectives and the Partition Feasibility Boundary

**Deriving $B$ and $(PP, TP, EP, SP)$ Bounds from TTFT and TPOT SLOs**

<br/>

**Author:** Yue Lu  
**Date:** April 2026

**Keywords:**  
LLM inference, service level objective (SLO), goodput, TTFT, TPOT, partition feasibility, batch size cap, pipeline parallelism cap, disaggregated prefill, continuous batching, Pareto frontier, roofline model

---

<div style="page-break-after: always;"></div>

# Table of Contents

- [1. SLOs as Operational Constraints](#1-slos-as-operational-constraints)
  - [1.1 The Two Canonical SLOs](#11-the-two-canonical-slos)
  - [1.2 The Compliance Percentile $p$](#12-the-compliance-percentile-p)
  - [1.3 Goodput as the Optimization Target](#13-goodput-as-the-optimization-target)

- [2. The TPOT-SLO to $B_{\max}$ Mapping](#2-the-tpot-slo-to-b_max-mapping)
  - [2.1 Inverting the Decode Roofline](#21-inverting-the-decode-roofline)
  - [2.2 Three Roofline Zones Reread as SLO Regimes](#22-three-roofline-zones-reread-as-slo-regimes)
  - [2.3 Worked Example — GPT-1.8T MoE on NVL72](#23-worked-example--gpt-18t-moe-on-nvl72)

- [3. The TTFT-SLO to $PP_{\max}$ Mapping](#3-the-ttft-slo-to-pp_max-mapping)
  - [3.1 TTFT Linearity in PP](#31-ttft-linearity-in-pp)
  - [3.2 Chunked Prefill Softens the Bound](#32-chunked-prefill-softens-the-bound)
  - [3.3 Disaggregation When the Two SLOs Disagree on PP](#33-disaggregation-when-the-two-slos-disagree-on-pp)

- [4. The Joint Feasibility Region](#4-the-joint-feasibility-region)
  - [4.1 Both SLOs Must Hold](#41-both-slos-must-hold)
  - [4.2 Static vs. Dynamic Feasibility — the $B \ge PP$ Cushion](#42-static-vs-dynamic-feasibility--the-b-ge-pp-cushion)
  - [4.3 Why $p99$ Narrows the Region](#43-why-p99-narrows-the-region)

- [5. Goodput-Optimal Partition](#5-goodput-optimal-partition)
  - [5.1 Goodput as Objective, SLOs as Constraints](#51-goodput-as-objective-slos-as-constraints)
  - [5.2 Sweep Recipe](#52-sweep-recipe)
  - [5.3 Out of Scope](#53-out-of-scope)

- [6. Workload-Specific SLO Targets](#6-workload-specific-slo-targets)
  - [6.1 Conversational Chat](#61-conversational-chat)
  - [6.2 Agentic / Tool-Use](#62-agentic--tool-use)
  - [6.3 Batch / Offline Inference](#63-batch--offline-inference)

- [References](#references)
- [Symbol Summary](#symbol-summary)

---

<div style="page-break-before: always;"></div>

# 1. SLOs as Operational Constraints

A service-level objective (SLO) is an operator-set upper bound on a user-observed latency metric, paired with a compliance percentile (typically the 90th, 95th, or 99th). The serving system is expected to keep the metric below the bound for at least the specified percentile of requests. SLOs are the primary mechanism by which production deployments translate user-experience requirements ("first token within 500 ms," "5 tokens/s streaming or better") into operational constraints on the serving stack — and through the roofline composed in `decode.md §6.2` and `prefill.md §3.4`, into hard bounds on the partition shape $(PP, TP, EP, SP)$ and the operating batch size $B$.

This document derives those bounds. Section 2 inverts the decode roofline to produce $B_{\max}$ as a function of $\mathrm{TPOT_{SLO}}$ at a given partition shape; Section 3 inverts the prefill roofline to produce $PP_{\max}$ as a function of $\mathrm{TTFT_{SLO}}$ at a given input-prompt length; Section 4 intersects the two bounds (plus a dynamic stability cushion) to define the feasibility region; Section 5 frames partition choice as goodput maximization over that region. Section 6 maps common workload classes onto representative SLO targets.

The goodput definition itself is established in `e2e.md §1.5`; this document treats that section as the canonical statement and builds the partition-feasibility derivation on top of it.

---

## 1.1 The Two Canonical SLOs

Production LLM inference frameworks ([VLLM], [DYNAMO], [TENSORRT-LLM]) almost universally expose two latency-side SLOs:

- **$\mathrm{TTFT_{SLO}}$ — time-to-first-token (TTFT) bound.** Operator-set upper bound on the time from request admission to the first output token reaching the user. Typical targets: 200 ms (interactive chat), 500 ms (multi-turn assistant), 2 s (batch document tasks).
- **$\mathrm{TPOT_{SLO}}$ — time-per-output-token (TPOT) bound.** Operator-set upper bound on the inter-token latency during the streaming decode phase. Typical targets: 50 ms (15 tokens/s, "comfortable reading speed"), 100 ms (10 tokens/s, "live streaming"), 200 ms (5 tokens/s, "minimum viable streaming"). Below 5 tokens/s the user perceives the response as stalled rather than streaming.

A request *meets* the joint SLO if **both** its TTFT and its full-response mean TPOT fall below the respective bounds. Both are user-observable: TTFT determines the "feels responsive?" judgment a user makes within the first second; TPOT determines whether the streaming output is comfortable to read or feels sluggish. The two metrics are produced by structurally different serving phases (compute-bound prefill versus memory-bound decode) and therefore have different sensitivities to the partition shape — a misalignment that drives much of the production architecture choice analyzed in §3.3.

Throughput-side SLOs (requests/second per replica, tokens/second per cluster) exist but are typically **derived** quantities. For a fixed latency-SLO pair, the maximum sustainable arrival rate $\lambda$ is uniquely determined — that quantity is the goodput defined in §1.3, and it serves as the optimization target rather than an independent constraint.

Other latency-side SLOs occasionally surface in production (end-to-end response-completion deadline, cancellation-tolerant best-effort tail) but reduce in our framework to one of the two canonical SLOs above plus a fixed output-length distribution; we focus on the canonical pair throughout.

---

## 1.2 The Compliance Percentile $p$

The SLO is met "at percentile $p$" when at least $p\%$ of requests fall below the bound. The choice of $p$ has non-trivial consequences for partition choice because the per-request distribution of TTFT and TPOT is non-Gaussian and right-skewed under realistic traffic patterns:

- **Mean-driven SLOs** ($p = 50$, equivalently a target on $\mathbb{E}[\mathrm{TPOT}]$). Easy to satisfy and easy to model; the steady-state roofline composition in `decode.md §6.2` directly produces a mean-TPOT estimate. Mean-driven SLOs are typical in offline batch evaluation, internal service-level agreements without strict user-experience claims, and synthetic benchmarks (most published throughput–latency curves implicitly use $p \le 50$).
- **Tail-driven SLOs** ($p \in \{90, 95, 99\}$). The standard for user-facing production deployments. Tail-driven SLOs penalize variance: a serving stack whose mean-TPOT is well within the bound but whose p99 spikes during pipeline-bubble recovery, KV-cache eviction, or admission-control queuing fails the SLO and is operationally non-viable.

For analytical modeling we work with the mean throughout the derivations of §2 and §3, then add a **tail-cushion margin** in §4.3 to translate mean-driven feasibility into tail-driven feasibility. This factorization mirrors the [DISAGG-PREFILL] approach: derive the mean-TPOT and mean-TTFT bounds from a roofline, then verify the percentile bound through simulation or empirical workload replay.

The cushion is non-trivial because tail amplification is partition-shape-dependent. Deeper PP narrows the dynamic-stability margin (§4.2), so tail-TPOT under bursty traffic can be much larger than mean-TPOT for the same shape — even when mean-TPOT looks comfortable. This is one of the structural reasons production frameworks default to shallower PP than the steady-state roofline would prefer.

---

## 1.3 Goodput as the Optimization Target

A serving cluster does not operate at a single $(B, \lambda)$ point: it operates over a *range* of arrival rates, with $B$ varying iteration-to-iteration as the continuous-batching scheduler admits and retires requests (`e2e.md §3.2`). The natural optimization target is therefore not "maximize throughput" but **maximize the arrival rate $\lambda$ that the cluster can sustain while keeping both SLOs satisfied at percentile $p$.** This quantity is the goodput, defined in `e2e.md §1.5`:

$$
\mathrm{Goodput} \;=\; \max\,\lambda
\quad \text{s.t.} \quad
P_p\!\left[\mathrm{TTFT}(\lambda)\right] \le \mathrm{TTFT_{SLO}}
\;\;\text{and}\;\;
P_p\!\left[\mathrm{TPOT}(\lambda)\right] \le \mathrm{TPOT_{SLO}}
$$

Goodput differs from raw throughput in three operationally important ways. **(1)** Throughput counts every token emitted; goodput counts only tokens emitted by requests that met the SLO. A cluster running at 100% utilization but failing the p99 SLO has high throughput and zero goodput from a contractual standpoint. **(2)** Goodput is the binding metric for production capacity planning: the number of replicas required to serve a given request rate is $\lceil \lambda_{\text{offered}} / \mathrm{Goodput} \rceil$, not $\lceil \lambda_{\text{offered}} / \mathrm{Throughput}_{\max} \rceil$. **(3)** Goodput is the right comparison axis when evaluating partition shapes or hardware platforms — a hardware configuration that achieves higher peak throughput but lower goodput is a worse production deployment, full stop.

The goodput framework was formalized by [ALPASERVE] (statistical multiplexing across SLO-bound replicas) and adopted as the optimization target by [DISAGG-PREFILL] (DistServe), [SPLITWISE] (phase-split SLO co-design), and [MOONCAKE]. The conceptual move — from throughput to goodput as the figure of merit — is now standard in serving-systems research and informs the production defaults of [VLLM], [TENSORRT-LLM], and [DYNAMO]. We adopt it as the framework's optimization target; the rest of this document derives the constraints that define the feasibility region over which goodput is maximized.

---

<div style="page-break-before: always;"></div>

# 2. The TPOT-SLO to $B_{\max}$ Mapping

The TPOT-SLO bounds the per-step decode latency $t_{\mathrm{step,user}}(B)$, which in turn bounds the operating batch size $B$. The bound is non-trivial because $t_{\mathrm{step,user}}(B)$ is non-monotone-flat in $B$ — it sits at a memory-bound floor up to the crossover batch $B^\star$ from `decode.md §3.3`, then grows linearly past it. The TPOT-SLO either falls in the flat region (in which case the bound is loose and not binding), the linear region (in which case the bound is tight and gives a closed-form $B_{\max}$), or below the floor (in which case no $B$ satisfies the SLO at the current partition shape, and the partition is infeasible).

> **Speculative-decoding extension.** When the deployment runs with speculative decoding enabled (Multi-Token Prediction (MTP) / EAGLE / Medusa, `decode.md §8`), the user-observed TPOT is $\mathrm{TPOT_{spec}}(B) = t_{\mathrm{step,user}}^{\text{verify}}(B) / N_{\text{tok/step}}$ rather than $t_{\mathrm{step,user}}(B)$. The SLO then bounds $\mathrm{TPOT_{spec}}$, and both the verify-step floor and the verify-step crossover $B^\star_{\text{spec}}$ from `decode.md §8.4` substitute for their vanilla counterparts in the analysis below. In the memory-bound regime — the dominant operating regime for MoE / MLA decode at production batch sizes — the substitution is a clean $1/N_{\text{tok/step}}$ scaling on TPOT and the §2.2 Zone 1 / Zone 2 / Zone 3 partition-feasibility argument carries through unchanged.

---

## 2.1 Inverting the Decode Roofline

The full decode-step time from `decode.md §7.2` is:

$$
t_{\mathrm{step,user}}(B)
\;=\; \gamma_{\mathrm{pp}} \cdot \big[t_{\mathrm{stage,hw}}(B) + \max(0, t_{\mathrm{stage,kernel}} - \rho_{\mathrm{kernel}} \, t_{\mathrm{stage,hw}}(B))\big]
\;+\; t_{\mathrm{LM,hw}}(B)
$$

where $\gamma_{\mathrm{pp}} = \max(1, PP/B)$ is the pipeline bubble factor, $t_{\mathrm{stage,hw}}(B)$ is the overlap-aware per-stage hardware roofline (`decode.md §6.2`), $t_{\mathrm{stage,kernel}}$ is the per-stage CPU/host kernel-launch dispatch budget (`decode.md §7.1`), $\rho_{\mathrm{kernel}}$ is the dispatch-overlap factor, and $t_{\mathrm{LM,hw}}(B)$ is the once-per-step LM head $H \to V$ projection on stage $PP{-}1$.

For batch sizes $B \ge PP$ (the pipeline-full regime, see §4.2 for why this is the operational target), $\gamma_{\mathrm{pp}} = 1$ and the formula simplifies. The hardware stage time itself is a max-of-rooflines:

$$
t_{\mathrm{stage,hw}}(B)
\;=\; t_{\mathrm{stage,local}}(B) + \max(0, t_{\mathrm{stage,comm}}(B) - \rho \, t_{\mathrm{stage,local}}(B))
$$

with the local term given by the compute/memory roofline:

$$
t_{\mathrm{stage,local}}(B)
\;=\; \max\!\left(\frac{B \cdot F_{\mathrm{token,device}}}{R_{\mathrm{GPU}}},\; \frac{T_{\theta,\mathrm{device}} + B \cdot T_{\mathrm{KV,device}}}{BW_{\mathrm{mem}}}\right)
$$

The dependence on $B$ is qualitatively simple: $t_{\mathrm{stage,local}}$ is approximately constant in $B$ for $B \ll B^\star$ (weight-stream-bound, the $T_{\theta,\mathrm{device}} / BW_{\mathrm{mem}}$ term dominates) and grows linearly in $B$ for $B \gg B^\star$ (compute-bound, the $B \cdot F_{\mathrm{token,device}} / R_{\mathrm{GPU}}$ term dominates). Setting $t_{\mathrm{step,user}}(B) = \mathrm{TPOT_{SLO}}$ and solving for $B$ gives $B_{\max}$. We carry the algebra zone-by-zone in §2.2 because the closed-form expression and the operational interpretation differ across zones.

---

## 2.2 Three Roofline Zones Reread as SLO Regimes

The decode roofline has three zones in $B$ (see `e2e.md §6.2` for the full development); the TPOT-SLO bound looks structurally different in each.

### Zone 1 — Memory-bound floor ($B \ll B^\star$)

The step time sits at the weight-streaming floor:

$$
t_{\mathrm{step,user}}(B) \approx \frac{T_{\theta,\mathrm{device}}}{BW_{\mathrm{mem}}}
\qquad (B \ll B^\star)
$$

If $\mathrm{TPOT_{SLO}} > T_{\theta,\mathrm{device}} / BW_{\mathrm{mem}}$, the SLO is satisfied at every $B$ in the memory-bound regime; $B_{\max}$ is then determined by the *Zone-3 ceiling* below, not by Zone 1. If $\mathrm{TPOT_{SLO}} < T_{\theta,\mathrm{device}} / BW_{\mathrm{mem}}$, **no $B$ satisfies the SLO at this partition shape** — the per-device weight footprint is too large, and a more aggressive sharding (larger $TP$, $PP$, $EP$, or $SP$) is required. This is the cleanest, most operationally consequential SLO bound: it tells the partition optimizer that a candidate shape is infeasible *before* any batch-size sweep begins.

### Zone 2 — Crossover knee ($B \approx B^\star$)

The system operates at the ridge point; both rooflines contribute comparably to $t_{\mathrm{step,user}}$. For SLO purposes, Zone 2 is the **natural operating point**: throughput-per-GPU has saturated near the compute ceiling, and per-user latency has not yet started degrading. A well-tuned production deployment targets $B \approx B^\star$ at peak load.

### Zone 3 — Compute-bound ceiling ($B \gg B^\star$)

The step time grows linearly with $B$:

$$
t_{\mathrm{step,user}}(B) \approx \frac{B \cdot F_{\mathrm{token,device}}}{R_{\mathrm{GPU}}}
\qquad (B \gg B^\star)
$$

Setting equal to $\mathrm{TPOT_{SLO}}$ and solving yields the SLO-bound batch size:

$$
\boxed{\,B_{\max} \;\approx\; \frac{R_{\mathrm{GPU}} \cdot \mathrm{TPOT_{SLO}}}{F_{\mathrm{token,device}}}\,}
\qquad (\text{Zone 3})
$$

This is the operationally tight bound for any deployment that has crossed $B^\star$. It is independent of $T_{\theta,\mathrm{device}}$ — once compute-bound, the per-device weight footprint no longer affects per-step time — and depends only on the compute capacity per device $R_{\mathrm{GPU}}$ (FLOPS), the per-token FLOPs $F_{\mathrm{token,device}}$ (model architecture and partition shape), and the SLO target itself.

### Combined feasibility condition

A partition shape $(PP, TP, EP, SP)$ at sequence length $S$ is **TPOT-feasible** if and only if the Zone 1 floor is below the SLO:

$$
\frac{T_{\theta,\mathrm{device}}(PP, TP, EP, SP)}{BW_{\mathrm{mem}}} \;\le\; \mathrm{TPOT_{SLO}}
$$

For partition shapes that pass this check, $B_{\max}$ is given by the Zone-3 expression above, capped by the HBM capacity bound (`decode.md §1.4`) — at most $B$ KV-caches must fit in HBM alongside the weights and activation buffers. The full feasibility condition for the partition-batch tuple is:

$$
B \;\le\; \min\!\left(
\underbrace{\frac{R_{\mathrm{GPU}} \cdot \mathrm{TPOT_{SLO}}}{F_{\mathrm{token,device}}}}_{\text{TPOT-SLO bound}},\;
\underbrace{\frac{HBM_{\mathrm{free}}}{T_{\mathrm{KV,device}} \cdot S}}_{\text{HBM bound}}
\right)
\qquad
\text{subject to}\;\;
\frac{T_{\theta,\mathrm{device}}}{BW_{\mathrm{mem}}} \;\le\; \mathrm{TPOT_{SLO}}
$$

where $HBM_{\mathrm{free}}$ is the HBM remaining after weights and activations are accounted for. The TPOT-SLO bound is decode-roofline-derived; the HBM bound is capacity-derived. Whichever binds first sets the operating $B_{\max}$.

---

## 2.3 Worked Example — GPT-1.8T MoE on NVL72

Concrete numbers make the bound tangible. Take the GPT-1.8T mixture-of-experts (MoE) model used elsewhere in the framework's partition sweeps, served on a 72-GPU NVL72 rack at FP4 quantization (bytes per parameter $b = 0.5$). Baseline parameters: $L = 120$ layers, $H = 18432$ hidden, $n_q = 128$ query heads, $n_{kv} = 16$ KV heads, dense FFN intermediate $I_{\mathrm{dense}} = H$, MoE intermediate $I_{\mathrm{moe}} = H/2$, $n_{\mathrm{experts}} = 16$, $k_{\mathrm{active}} = 2$. NVL72 device specs: $R_{\mathrm{GPU}} = 2.5\,\mathrm{PFLOPS}$ at FP4, $BW_{\mathrm{mem}} = 8\,\mathrm{TB/s}$.

Consider a candidate partition shape $PP=8$, $TP=8$, $EP=1$, $SP=1$ ($N_{\mathrm{GPUs,per-replica}} = 64$, $DP = 1$). Per-device weight footprint at FP4 is approximately $T_{\theta,\mathrm{device}} \approx 14\,\mathrm{GB}$ (full model 1.8 TB, sharded across 64 devices, with the MoE expert weights paged across the EP axis if $EP > 1$). The Zone-1 floor is then:

$$
t_{\mathrm{step,user}}^{\,\mathrm{floor}} \;\approx\; \frac{14\,\mathrm{GB}}{8\,\mathrm{TB/s}} \;=\; 1.75\,\mathrm{ms}
$$

Three SLO scenarios:

| $\mathrm{TPOT_{SLO}}$ | Workload class | Floor check | $B_{\max}$ from Zone-3 bound |
|---:|:---|:---|:---|
| 100 ms | live streaming, 10 tok/s | $1.75 \le 100$ ✓ | $\approx \frac{2.5 \cdot 10^{15} \cdot 0.1}{F_{\mathrm{token,device}}} \approx 100\text{–}200$ |
| 50 ms | comfortable reading, 15 tok/s | $1.75 \le 50$ ✓ | $\approx 50\text{–}100$ |
| 30 ms | tight conversational, 30 tok/s | $1.75 \le 30$ ✓ | $\approx 30\text{–}60$ |

The exact Zone-3 numbers depend on $F_{\mathrm{token,device}}$ which scales as $L \cdot H^2 / N_{\mathrm{GPUs,per-replica}}$ for the dense and active-expert paths (`decode.md §3`); the order of magnitude is what matters here. At the conversational target ($\mathrm{TPOT_{SLO}} = 30\,\mathrm{ms}$) and $B_{\max} \approx 50$, the operating point is well inside the memory-bound regime (where $B^\star$ for this shape is typically 64–256 at decode, see `decode.md §3.3` for the formula). The TPOT-SLO does not bind in Zone 3; the **HBM-capacity bound** binds instead, since 50 sequences with $S = 8192$ KV cache exceed the per-device HBM headroom.

Two takeaways. **(i)** For tight-conversational SLOs on dense-attention workloads, the binding constraint is rarely the TPOT-SLO itself — it is the HBM-capacity bound or (equivalently) the memory-bound floor. The role of the TPOT-SLO is to *eliminate partition shapes whose floor is too high*, not to set a tight $B_{\max}$. **(ii)** Pushing into Zone 3 to maximize per-GPU throughput requires either a very loose TPOT-SLO (batch / async workloads, see §6.3) or a partition shape whose $F_{\mathrm{token,device}}$ has been driven low enough by $TP$ / $EP$ sharding to keep $B \cdot F / R$ under the SLO. The latter is exactly the partition-vs-SLO co-design loop that §5 formalizes.

---

<div style="page-break-before: always;"></div>

# 3. The TTFT-SLO to $PP_{\max}$ Mapping

The TTFT-SLO bounds the time from request admission to first output token, which composes the scheduling overhead, the prefill forward pass, the optional inter-cluster KV transfer (under disaggregation), and the first decode step (`e2e.md §2.1`). The component most sensitive to partition shape is the prefill forward pass, which in a co-located pipeline-parallel deployment includes a **pipeline-warmup** term that scales linearly with PP. The TTFT-SLO therefore translates almost directly into a $PP_{\max}$ at fixed input length $S_{\mathrm{input}}$.

---

## 3.1 TTFT Linearity in PP

From `prefill.md §3.3 / §3.4`, the prefill forward pass for a single request traverses all $PP$ pipeline stages sequentially before the first token can be sampled. Unlike decode, prefill does *not* have inflight microbatching across stages on a single request — the prompt is one indivisible work unit, so the pipeline cannot be filled, and the warmup phase cannot be amortized:

$$
t_{\mathrm{prefill}}
\;=\; t_{\mathrm{prefill,local}}(S_{\mathrm{input}}) + \max(0, t_{\mathrm{prefill,comm}} - \rho \, t_{\mathrm{prefill,local}})
\;+\; \underbrace{(PP - 1) \cdot t_{\mathrm{stage,max}}}_{t_{\mathrm{pipeline,warmup}}}
\;+\; t_{\mathrm{LM,prefill,hw}}
$$

The warmup term $(PP - 1) \cdot t_{\mathrm{stage,max}}$ is the time for the first prompt to traverse the PP-stage pipeline before its first token is sampled on stage $PP{-}1$. Doubling $PP$ approximately doubles the warmup, with no compensating reduction in $t_{\mathrm{prefill,local}}$ (which scales as $L/PP$ in the per-stage compute but already accounts for that scaling — the warmup is the *pipeline-fill* surcharge on top).

Composing into the full single-request co-located TTFT (`e2e.md §2.1`):

$$
\mathrm{TTFT_{single}}
\;=\; t_{\mathrm{sched}} + t_{\mathrm{prefill}} + t_{\mathrm{step,user}}
\;\approx\; t_{\mathrm{sched}} + t_{\mathrm{prefill,local}}(S_{\mathrm{input}}) + (PP - 1) \cdot t_{\mathrm{stage,max}} + t_{\mathrm{step,user}}
$$

Setting equal to $\mathrm{TTFT_{SLO}}$ and solving for $PP$:

$$
\boxed{\,
PP_{\max}
\;\approx\;
1 + \frac{\mathrm{TTFT_{SLO}} - t_{\mathrm{sched}} - t_{\mathrm{prefill,local}}(S_{\mathrm{input}}) - t_{\mathrm{step,user}}}{t_{\mathrm{stage,max}}}
\,}
$$

For long input prompts (compute-bound prefill, $S_{\mathrm{input}} \gg S_{\mathrm{input}}^{\star}$ from `prefill.md §2.3`), $t_{\mathrm{prefill,local}}$ already consumes the bulk of $\mathrm{TTFT_{SLO}}$ and the warmup budget is small; $PP_{\max}$ is consequently small (often 1 or 2). For short input prompts (memory-bound prefill, weights stream once but cannot be amortized across a small token count), $t_{\mathrm{prefill,local}}$ is small and $PP_{\max}$ is more permissive — but short-prompt deployments are also exactly the deployments where *decode* dominates the user experience, so the TPOT-SLO bound from §2 typically dominates partition choice anyway.

The partition consequence is unambiguous: **TTFT-SLO caps PP from above**; TPOT-SLO caps the per-device weight footprint from below (which favors *more* sharding, i.e. larger $PP \cdot TP \cdot EP \cdot SP$). The two SLOs are aligned on the need for some sharding but disagree on which axis should carry it. PP carries the TTFT cost; TP, EP, and SP do not (or carry it only through their per-stage communication overhead). This asymmetry is the structural reason production frameworks default to TP-first intra-node and reach for PP only when forced by HBM-capacity (`prefill.md §6` covers the same observation from the prefill side).

---

## 3.2 Chunked Prefill Softens the Bound

[SARATHI] introduces **chunked prefill**: split the prefill of a single request into $N_{\mathrm{chunks}} = \lceil S_{\mathrm{input}} / C \rceil$ smaller chunks of $C$ tokens each, and interleave each chunk with a decode iteration. The motivation in [SARATHI] is to reduce **head-of-line blocking** — long prefill passes can stall the decode pipeline for tens to hundreds of milliseconds, inflating decode-phase TPOT for all in-flight requests. From the TTFT-SLO perspective, chunking has a different but compatible benefit: each chunk's prefill traverses the PP pipeline independently, and the warmup cost is paid *per chunk*. The prefill latency under chunking from `e2e.md §2.3`:

$$
\mathrm{TTFT_{chunked}}
\;\approx\; t_{\mathrm{sched}} + N_{\mathrm{chunks}} \cdot t_{\mathrm{chunk}} + t_{\mathrm{step,user}}
$$

where $t_{\mathrm{chunk}}$ is the per-chunk roofline at $C$ tokens (compute-bound for typical $C \in [256, 2048]$). The pipeline-warmup term reappears inside each $t_{\mathrm{chunk}}$, but the per-chunk warmup is paid against a $C$-token chunk rather than the full $S_{\mathrm{input}}$-token prompt — so the *relative* cost of pipeline depth scales as $PP / C$ rather than $PP / S_{\mathrm{input}}$, and the absolute TTFT inflation from PP is largely independent of $S_{\mathrm{input}}$. Chunking does not eliminate the $PP_{\max}$ bound, but it makes the bound less sensitive to prompt length.

This is operationally important. Without chunking, a deployment that meets $\mathrm{TTFT_{SLO}}$ on $S_{\mathrm{input}} = 1024$ prompts at $PP = 4$ may fail it on $S_{\mathrm{input}} = 8192$ prompts at the same $PP$. With chunking and a fixed $C$, the same $PP$ produces approximately the same per-chunk TTFT contribution, and the TTFT inflation from long prompts comes from the additional chunks rather than from pipeline depth. Production frameworks that target long-context workloads ([VLLM] continuous batching with chunked prefill enabled, [DYNAMO]) rely on this softening to keep PP feasible at long $S_{\mathrm{input}}$.

The cost of chunking is a per-chunk surcharge in goodput: each chunk pays its own scheduling overhead, its own pipeline warmup, and its own all-reduce kernels. For very small $C$ this surcharge dominates; for very large $C$ chunking degenerates into single-shot prefill. Typical production tunings sit at $C \in [512, 2048]$, balancing TTFT softening against per-chunk overhead.

---

## 3.3 Disaggregation When the Two SLOs Disagree on PP

When the TTFT-SLO and the TPOT-SLO produce structurally incompatible partition shapes — TTFT prefers shallow PP (§3.1), TPOT wants enough sharding to keep the per-device weight footprint below the floor (§2.2) — co-located prefill+decode is forced into a compromise that satisfies neither well. **Disaggregated prefill** [DISAGG-PREFILL] resolves the conflict by running the two phases on physically separate clusters with independent partition shapes, connected by a KV-cache transfer over an inter-cluster fabric.

The structural argument. Prefill and decode have qualitatively different roofline characters: prefill is compute-bound at typical $S_{\mathrm{input}}$ and benefits from high-FLOPS hardware with shallow PP; decode is memory-bound and benefits from high-bandwidth HBM with whatever sharding is required to fit the per-device weight footprint under the TPOT-SLO floor. [DISAGG-PREFILL] (DistServe), [SPLITWISE] (phase-split co-design), and [MOONCAKE] (KV-centric architecture) all motivate disaggregation through this asymmetry, and the production deployment in [DYNAMO] adopts it. The cost of disaggregation is the inter-cluster KV-cache transfer:

$$
t_{\mathrm{KV\text{-}transfer}}
\;=\; \alpha_{\mathrm{inter}} + \frac{M_{\mathrm{KV\text{-}transfer}}}{BW_{\mathrm{inter}}}
$$

where $M_{\mathrm{KV\text{-}transfer}} = \frac{2 \cdot S_{\mathrm{input}} \cdot H_{kv} \cdot b}{TP \cdot SP} \cdot \frac{L}{PP}$ is the per-device KV-cache transfer volume (`e2e.md §2.1`). Whether disaggregation improves TTFT or worsens it depends on the algebraic comparison:

$$
\underbrace{(PP_{\mathrm{co}} - 1) \cdot t_{\mathrm{stage,max,co}}}_{\text{co-located warmup penalty}}
\;\;\text{vs.}\;\;
\underbrace{t_{\mathrm{KV\text{-}transfer}} + (PP_{\mathrm{prefill}} - 1) \cdot t_{\mathrm{stage,max,prefill}}}_{\text{disagg transfer + prefill-cluster warmup}}
$$

Disaggregation wins when $PP_{\mathrm{prefill}}$ is much smaller than $PP_{\mathrm{co}}$ (the prefill cluster doesn't need the deep PP that the decode cluster needs to fit the model) and when $BW_{\mathrm{inter}}$ is high enough that $t_{\mathrm{KV\text{-}transfer}}$ stays below the warmup it eliminates. Modern layer-wise KV streaming (covered in [MOONCAKE] and adopted by [DYNAMO]) overlaps the transfer with the prefill itself, driving the effective transfer-latency penalty toward zero — at which point disaggregation is essentially free TTFT for any workload where the SLOs disagree on PP.

The goodput improvement from disaggregation comes from a second source: **independent scaling**. A co-located cluster sized for the larger of the two SLO footprints (typically decode) underutilizes the over-provisioned dimension; a disaggregated cluster sizes prefill and decode independently and runs each at its own goodput optimum. [DISAGG-PREFILL] reports goodput improvements of 2–4× on workloads where prefill and decode preferences diverge (long prompts with short responses, or short prompts with long responses). The tradeoff is operational complexity: two clusters, two scheduling decisions, KV-transfer protocol, and request-routing logic.

For our framework's purposes, the disaggregation analysis collapses into a simple decision rule. Compute the goodput-optimal partition under the two-cluster assumption (separate $PP_{\mathrm{prefill}}$ and $PP_{\mathrm{decode}}$); compute it under the one-cluster constraint ($PP_{\mathrm{prefill}} = PP_{\mathrm{decode}}$); the larger of the two goodputs identifies whether disaggregation is beneficial at the offered workload mix. We return to this in §5.

---

<div style="page-break-before: always;"></div>

# 4. The Joint Feasibility Region

The bounds from §2 ($B_{\max}$ from $\mathrm{TPOT_{SLO}}$) and §3 ($PP_{\max}$ from $\mathrm{TTFT_{SLO}}$) are independent constraints — both must hold for a deployment to meet the joint SLO. Together with HBM capacity, dynamic-stability, and tail-cushion margins, they define the *feasibility region*: the set of $(PP, TP, EP, SP, B)$ tuples over which goodput maximization (§5) is performed.

---

## 4.1 Both SLOs Must Hold

The joint feasibility region is the intersection of the per-SLO regions:

$$
\mathcal{F}_{\mathrm{SLO}}
\;=\;
\Big\{ (PP, TP, EP, SP, B) \;\Big|\;
\underbrace{\frac{T_{\theta,\mathrm{device}}}{BW_{\mathrm{mem}}} \le \mathrm{TPOT_{SLO}}}_{\text{TPOT floor}}
\;\wedge\;
\underbrace{B \le B_{\max}}_{\text{TPOT bound}}
\;\wedge\;
\underbrace{PP \le PP_{\max}}_{\text{TTFT bound}}
\;\wedge\;
\underbrace{\mathrm{HBM}_{\mathrm{used}}(B) \le \mathrm{HBM}_{\mathrm{capacity}}}_{\text{capacity}}
\Big\}
$$

A partition shape that fails the TPOT floor is *structurally* infeasible: no batch size $B$ rescues it. A partition shape that passes the TPOT floor but has $PP > PP_{\max}$ is *prefill-infeasible*: it can sustain decode but cannot deliver first tokens within the TTFT budget. A shape that satisfies both SLO bounds but exceeds HBM capacity at the SLO-bound $B$ is *capacity-infeasible*.

The first prune (TPOT floor) is the cheapest to apply and eliminates the largest fraction of the partition space; we run it first in §5.2's sweep recipe. Among shapes that pass the floor, the joint constraint $B \le \min(B_{\max}, B_{\mathrm{HBM}})$ produces the tight per-shape operating bound. The TTFT bound acts as an outer constraint on $PP$ alone and can often be evaluated independently of $B$.

---

## 4.2 Static vs. Dynamic Feasibility — the $B \ge PP$ Cushion

The bounds in §2 and §3 are *static* — they assume a steady-state operating point. Production traffic is not steady-state: arrivals are bursty (Poisson at best, often heavier-tailed), individual requests complete asynchronously as their stop tokens fire, and sequence lengths vary by orders of magnitude across requests. The active-batch count $B(t)$ at any instant is a stochastic process that fluctuates around its mean, and **the pipeline-bubble factor $\gamma_{\mathrm{pp}} = \max(1, PP/B(t))$ is not flat under fluctuation** — it stays at 1 as long as $B(t) \ge PP$, then jumps to $PP/B(t) > 1$ the instant the active batch dips below $PP$.

The consequence for SLO feasibility. A deployment whose mean-$B$ satisfies $\overline{B} \ge PP$ but whose instantaneous $B(t)$ occasionally dips below $PP$ pays a *multiplicative* TPOT penalty during those dips. For a tail-driven SLO at $p99$, a 1% probability of $B(t) < PP$ is enough to lift the tail TPOT by a factor of $PP / B_{\mathrm{dip}}$ — a 60-stage pipeline that dips to $B = 30$ during a momentary lull pays a $2\times$ tail-TPOT penalty on those iterations. Under the joint SLO, this tail penalty must be absorbed by the cushion between mean-TPOT and $\mathrm{TPOT_{SLO}}$.

The dynamic-feasibility constraint is therefore stricter than the static one. We require the *operating mean batch* $\overline{B}$ to satisfy:

$$
\overline{B} \;\ge\; \underbrace{PP}_{\text{static minimum}} \;+\; \underbrace{\Delta_{\mathrm{dyn}}(p, \sigma_B)}_{\text{dynamic cushion}}
$$

where $\sigma_B$ is the standard deviation of $B(t)$ under the offered load and $\Delta_{\mathrm{dyn}}$ scales with the tail percentile $p$ and the load variance. Concretely, $\Delta_{\mathrm{dyn}} \approx z_p \cdot \sigma_B$ where $z_p$ is the standard-normal $p$-quantile (1.28 for p90, 2.33 for p99). For Poisson arrivals at request rate $\lambda$ with mean sojourn time $\bar{\tau}$, $\overline{B} = \lambda \bar{\tau}$ and $\sigma_B = \sqrt{\lambda \bar{\tau}}$, giving $\Delta_{\mathrm{dyn}} \approx z_p \sqrt{\overline{B}}$ — a square-root scaling. For p99 at $\overline{B} = 100$, the cushion is $\approx 23$, so the *operating* $\overline{B}$ must be 23 above the static minimum $PP$ to keep the p99 dip-rate below 1%.

Two operational implications. **(i)** Deeper PP requires a larger operating batch, both to satisfy the static minimum and to absorb the dynamic cushion. A deployment sized for $PP = 60$ needs $\overline{B} \ge 80$ at p99; a deployment sized for $PP = 8$ needs $\overline{B} \ge 12$. The required arrival rate to sustain the larger $\overline{B}$ scales accordingly (Little's Law, `e2e.md §3.2`), and clusters with insufficient offered load see deep-PP shapes fail the dynamic-feasibility check even when they pass the static one. **(ii)** The B≥PP cushion is the underlying mechanism behind production's preference for TP-first over PP-first: TP carries no analogous dynamic-stability requirement, since its per-layer all-reduce is independent of the active-batch count. PP and EP both require maintaining a minimum active-batch — but PP's bubble penalty under shortfall is multiplicative across all active sequences, while EP's is per-expert and partial.

The bound on $\Delta_{\mathrm{dyn}}$ depends on the workload model (Poisson, Gamma-distributed sojourn, heavier-tailed bursts) and the scheduler's admission-control policy; rigorous derivation is deferred to the simulation framework. For the analytical feasibility region we use the conservative rule $\overline{B} \ge 1.5 \cdot PP$ at p99, which absorbs the $z_p \sqrt{\overline{B}}$ cushion for realistic operating points.

---

## 4.3 Why $p99$ Narrows the Region

The mean-driven feasibility region of §4.1 ignores variance; the tail-driven version of §4.2 absorbs *batch* variance into the cushion. A third source of variance is sequence-length variance: requests with longer prompts inflate prefill warmup and longer decode tails inflate per-step compute. For a tail-driven $\mathrm{TTFT_{SLO}}$ at p99, the binding scenario is the longest-prompt request — and the partition shape must satisfy the TTFT bound at $S_{\mathrm{input,max}}$ (or the p99 of the prompt-length distribution), not at $\overline{S_{\mathrm{input}}}$. This typically tightens $PP_{\max}$ by a factor of 1.5–3× depending on the prompt-length distribution.

For TPOT, the analogous tail effect is sequence-length-induced KV-cache growth: a long-decode request accumulates KV traffic linearly in $S$, eventually pushing $T_{\mathrm{KV,device}}$ above the threshold where $B^\star$ collapses (`decode.md §3.3` no-crossover regime). The p99 TPOT for the long-tailed sequence-length distribution can be substantially higher than the mean — and the tail-feasibility bound on $B_{\max}$ is correspondingly tighter than the mean bound from §2.2.

These tail effects are workload-distribution-dependent and not captured analytically in our roofline framework; they require empirical workload profiling or distribution-aware simulation. For partition-feasibility analysis we apply the cushion factors (typically 1.5× on $PP_{\max}$ and 0.7× on $B_{\max}$ for p99 vs. mean) and verify the result against profiling data. The framework's published partition sweeps use mean-driven bounds to keep the sweep computationally tractable; the tail cushions are applied at deployment time.

---

<div style="page-break-before: always;"></div>

# 5. Goodput-Optimal Partition

The feasibility region of §4 defines what is *allowed*; the goodput function of §1.3 defines what is *optimal*. Goodput-optimal partition choice is the maximization of the goodput over the feasibility region — a constrained optimization problem with a tractable structure, which we develop as a sweep recipe.

---

## 5.1 Goodput as Objective, SLOs as Constraints

Restated formally:

$$
\max_{(PP, TP, EP, SP) \in \mathcal{F}_{\mathrm{SLO}}}
\;\;\;\mathrm{Goodput}(PP, TP, EP, SP)
\;=\;
\frac{B_{\mathrm{op}}(PP, TP, EP, SP)}{\overline{TTFT} + N_{\mathrm{out}} \cdot \overline{\mathrm{TPOT}}}
$$

via Little's Law (`e2e.md §3.2`), where $B_{\mathrm{op}}$ is the SLO-feasible operating batch (the smaller of $B_{\max}$ and the HBM bound), and $\overline{TTFT}$, $\overline{\mathrm{TPOT}}$ are the mean-driven values at $B_{\mathrm{op}}$. The goodput is dimensioned as requests/second/replica; multiplying by $DP$ gives the cluster goodput.

The structure of the maximum. Because $B_{\mathrm{op}}$ enters the numerator and $\overline{\mathrm{TPOT}}$ enters the denominator, and because both are functions of the partition shape with opposite-sign sensitivities, the goodput function has interior maxima — there is generally a "sweet spot" partition rather than a corner solution. The TPOT-SLO floor (§2.2) eliminates the corner of *too little sharding* (per-device weight footprint above the floor); the TTFT-SLO bound (§3.1) eliminates the corner of *too much PP*; the HBM bound eliminates the corner of *too much B*. Within those constraints, the goodput-optimal point typically sits at moderate sharding with $B \approx B^\star$ — exactly the Zone-2 knee identified in `e2e.md §6.2` as the natural production operating point.

---

## 5.2 Sweep Recipe

The feasibility region is discrete (factor combinations of $PP$, $TP$, $EP$, $SP$ that divide the device count) and small (typically a few hundred candidate shapes for a 64–256 device cluster). Brute-force enumeration is computationally trivial. The sweep:

1. **Enumerate partition shapes.** For each integer factorization $N = PP \cdot TP \cdot EP \cdot SP \cdot DP$ of the device count, generate the candidate $(PP, TP, EP, SP)$ tuple.

2. **Apply the TPOT floor.** Compute $T_{\theta,\mathrm{device}}(PP, TP, EP, SP)$ from `decode.md §1.4`; reject shapes where $T_{\theta,\mathrm{device}} / BW_{\mathrm{mem}} > \mathrm{TPOT_{SLO}}$.

3. **Apply the TTFT bound.** Compute $t_{\mathrm{prefill,local}}(S_{\mathrm{input}})$ from `prefill.md §3` and $t_{\mathrm{stage,max}}$; reject shapes where the resulting $PP_{\max}$ from §3.1 is less than the candidate $PP$.

4. **Solve for $B_{\mathrm{op}}$.** For each surviving shape, compute $B_{\max}$ from the Zone-3 expression in §2.2 and $B_{\mathrm{HBM}}$ from the capacity bound; set $B_{\mathrm{op}} = \min(B_{\max}, B_{\mathrm{HBM}})$. If $B_{\mathrm{op}} < 1$ the shape is infeasible; reject.

5. **Apply the dynamic-stability cushion.** For surviving shapes, verify $B_{\mathrm{op}} \ge 1.5 \cdot PP$ (§4.2); reject shapes where the operating batch falls below the cushion.

6. **Compute goodput.** Evaluate the goodput function from §5.1 at $(PP, TP, EP, SP, B_{\mathrm{op}})$. Rank surviving shapes by goodput.

7. **Co-located vs. disaggregated comparison.** Repeat steps 1–6 under the disaggregated assumption with independent $PP_{\mathrm{prefill}}$ and $PP_{\mathrm{decode}}$; if the disaggregated maximum exceeds the co-located maximum by more than a workload-specific threshold (typically 10–20% to compensate for operational complexity), recommend disaggregation [DISAGG-PREFILL].

The framework's `notebooks/` partition sweeps implement steps 1, 2, 4, and 6 directly; the TTFT bound (step 3) and the dynamic-stability cushion (step 5) are applied as post-filters when SLO-aware analysis is required. The disaggregated comparison (step 7) is currently a manual analysis driven by workload-mix awareness; automating it would require an integrated prefill+decode optimizer, which is a natural extension of the analytical framework but beyond the current scope.

---

## 5.3 Out of Scope

Several real goodput drains are not modeled in this framework:

- **SLO-aware scheduling** — admission control, request prioritization by deadline, preemption of low-priority requests. Production stacks ([SARATHI], [DYNAMO]) implement these to recover goodput when the offered load exceeds the static feasibility region. Modeling them requires queue-theoretic or simulation-based analysis.
- **Speculative decoding rejection** — speculative-decoding pipelines emit tokens at higher peak rate but pay a rejection probability that affects effective TPOT. The roofline model captures the per-step time of speculative kernels but not the rejection-rate-driven goodput penalty.
- **Preemption-driven recompute** — when the scheduler evicts a long-running request to admit a higher-priority one, the evicted request's KV cache must be regenerated on resumption (`kv.md §4.3`). This is a real goodput cost and is not in our analytical model.
- **Cancellation effects** — users abandoning long-running responses produce wasted compute that does not count toward goodput. The fraction is workload-specific and not modeled.

These omissions are intentional: each is a load-dependent or scheduler-dependent effect that lies outside the scope of a static roofline-based feasibility analysis. The bounds derived in §2–§4 are correct *upper bounds* on goodput under ideal scheduling; production deployments achieve some fraction of those bounds depending on scheduler quality and workload characteristics. A reasonable rule of thumb (consistent with [DISAGG-PREFILL] and [ALPASERVE] reported numbers): a well-tuned production stack achieves 60–80% of the analytical goodput bound; a poorly tuned one can achieve well under 30%.

---

<div style="page-break-before: always;"></div>

# 6. Workload-Specific SLO Targets

The bounds of §2–§5 are parametric in the SLO targets. This section maps representative workload classes to their typical SLO-target ranges and identifies which bound binds in each class. The targets below are operational defaults observed across published serving benchmarks ([DISAGG-PREFILL], [VLLM], [DYNAMO]) and production deployment guides; specific deployments tune these to their own user-experience requirements.

---

## 6.1 Conversational Chat

**Profile.** Interactive multi-turn dialogue; user reads streaming output in real time; response lengths typically 50–300 tokens; prompt lengths 50–2000 tokens.

**Typical SLOs.** $\mathrm{TTFT_{SLO}} \in [200, 500]\,\mathrm{ms}$ at p95; $\mathrm{TPOT_{SLO}} \in [50, 100]\,\mathrm{ms}$ at p95 (10–20 tokens/s streaming).

**Binding constraint.** Both SLOs are typically loose enough that the binding constraint is *neither* — the deployment runs at the Zone-2 knee, and HBM capacity (KV-cache footprint at moderate $B$) is the binding constraint instead. The SLOs serve to *eliminate* shapes whose floor is above the TPOT bound (the §2.2 floor check) and whose PP-warmup exceeds the TTFT budget (the §3.1 bound), but within the surviving shapes the goodput optimum is primarily HBM-driven. Production frameworks default to $TP \in [4, 8]$ and $PP \in [1, 4]$ for this class, with disaggregation rarely worthwhile since both phases have moderate latency budgets.

---

## 6.2 Agentic / Tool-Use

**Profile.** LLM-driven agents calling external tools, generating structured output (JSON, code), looping with intermediate verification; each LLM invocation is part of a larger pipeline whose user-perceived latency is the *sum* across many invocations. Response lengths often short (10–100 tokens, structured); prompt lengths often very long (5000–32000 tokens, including tool-output history).

**Typical SLOs.** $\mathrm{TTFT_{SLO}} \in [100, 300]\,\mathrm{ms}$ at p95 (tighter than chat — every invocation pays it); $\mathrm{TPOT_{SLO}} \in [30, 50]\,\mathrm{ms}$ at p95 (tighter than chat — short responses amplify per-token overhead). Long prompts make the TTFT side particularly tight.

**Binding constraint.** $\mathrm{TTFT_{SLO}}$ is the dominant pressure. The combination of long prompts and tight TTFT targets pushes $PP_{\max}$ down hard (§3.1); chunked prefill (§3.2) is essentially mandatory. Disaggregation [DISAGG-PREFILL] is often worthwhile because the long-prompt-short-response asymmetry drives different optimal partitions for the two phases. The Zone-1 floor check (§2.2) typically rules out under-sharded shapes since long prompts inflate the per-token KV traffic, lowering $B^\star$ and the achievable $B$.

---

## 6.3 Batch / Offline Inference

**Profile.** Background processing — corpus summarization, document classification, large-scale evaluation, synthetic-data generation. No interactive user; the user-observable metric is end-to-end completion time of the entire job, not per-request latency.

**Typical SLOs.** $\mathrm{TTFT_{SLO}} \to \infty$ (no per-request constraint); $\mathrm{TPOT_{SLO}} \to \infty$. Practically: very loose targets at p50 to keep the worst-case throughput bounded ($\mathrm{TPOT_{SLO}} \in [500, 2000]\,\mathrm{ms}$), but no tail constraint.

**Binding constraint.** Goodput maximization without SLO bounds reduces to throughput maximization. The optimal operating point is well into Zone 3 of the roofline (`e2e.md §6.2`), with $B$ limited only by HBM capacity. Deep PP and aggressive sharding are both feasible since neither SLO bound binds. The framework's partition sweep without SLO filters (and the `notebooks/pareto_basic.ipynb` results) directly target this regime; the bounds derived in §2–§4 collapse to vacuous when both SLOs are infinite, recovering the throughput-only optimum.

---

<div style="page-break-before: always;"></div>

# References

The following references are cited in this document; full bibliographic entries are in `references.md`.

- **[ALPASERVE]** Li, Z., Zheng, L., Zhong, Y., et al. (2023). *AlpaServe: Statistical multiplexing with model parallelism for deep learning serving.* OSDI 2023. — Original goodput framing for SLO-bound serving; statistical multiplexing across replicas.
- **[DISAGG-PREFILL]** Zhong, Y., Liu, S., Chen, J., et al. (2024). *DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving.* OSDI 2024. arXiv:2401.09670. — Prefill–decode disaggregation; goodput as optimization target; KV-transfer cost analysis.
- **[SPLITWISE]** Patel, P., Choukse, E., et al. (2024). *Splitwise: Efficient Generative LLM Inference Using Phase Splitting.* ISCA 2024. — Phase-split SLO co-design; prefill/decode hardware asymmetry.
- **[SARATHI]** Agrawal, A., Panwar, A., Mohan, J., et al. (2023). *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills.* arXiv:2308.16369. — Chunked prefill; head-of-line blocking reduction; SLO-aware scheduling.
- **[MOONCAKE]** Qin, R., Li, Z., He, W., et al. (2024). *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving.* arXiv:2407.00079. — KV-centric disaggregated serving; layer-wise KV streaming; production deployment.
- **[DYNAMO]** NVIDIA Corporation. (2024–2025). *NVIDIA Dynamo: Distributed Inference Serving Framework.* — Production disaggregated inference; multi-cluster orchestration; chunked prefill at scale.
- **[VLLM]** Kwon, W., Li, Z., Zhuang, S., et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023. arXiv:2309.06180. — Continuous batching; PagedAttention; production-grade scheduling.
- **[ORCA]** Yu, G.-I., Jeong, J. S., Kim, G.-W., Kim, S., & Chun, B.-G. (2022). *Orca: A Distributed Serving System for Transformer-Based Generative Models.* OSDI 2022. — Iteration-level batching; the scheduling foundation for continuous-batch serving.
- **[TENSORRT-LLM]** NVIDIA Corporation. (2023–2024). *TensorRT-LLM Documentation.* — Production deployment guidance; partition recommendations across model classes.
- **[ROOFLINE]** Williams, S., Waterman, A., & Patterson, D. (2009). *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* CACM 52(4). — Source of the compute-vs-memory ridge-point construction used in Zone analysis.

---

<div style="page-break-before: always;"></div>

# Symbol Summary

The following symbols are introduced in this document and are not defined in `notation.md §14`. All other symbols used here ($\mathrm{TTFT}$, $\mathrm{TPOT}$, $\gamma_{\mathrm{pp}}$, $B^\star$, $T_{\theta,\mathrm{device}}$, $F_{\mathrm{token,device}}$, $R_{\mathrm{GPU}}$, $BW_{\mathrm{mem}}$, $\mathrm{Goodput}$, $\mathrm{TTFT_{SLO}}$, $\mathrm{TPOT_{SLO}}$, $p$, $\lambda$) are defined in `notation.md` and cross-referenced to their home modeling document.

| Symbol | Definition | First used |
|--------|------------|-----------|
| $B_{\max}$ | Largest batch size $B$ satisfying $\mathrm{TPOT_{SLO}}$ at the candidate partition shape. Closed form in Zone 3: $B_{\max} \approx R_{\mathrm{GPU}} \cdot \mathrm{TPOT_{SLO}} / F_{\mathrm{token,device}}$ | §2.2 |
| $B_{\mathrm{HBM}}$ | Largest batch size $B$ permitted by HBM capacity at the candidate partition shape: $B_{\mathrm{HBM}} \approx HBM_{\mathrm{free}} / (T_{\mathrm{KV,device}} \cdot S)$ | §2.2 |
| $B_{\mathrm{op}}$ | SLO-feasible operating batch size: $\min(B_{\max}, B_{\mathrm{HBM}})$ | §5.1 |
| $PP_{\max}$ | Largest pipeline-parallel depth satisfying $\mathrm{TTFT_{SLO}}$ at the candidate input length: $PP_{\max} \approx 1 + (\mathrm{TTFT_{SLO}} - t_{\mathrm{sched}} - t_{\mathrm{prefill,local}} - t_{\mathrm{step,user}}) / t_{\mathrm{stage,max}}$ | §3.1 |
| $\lambda^*$ | Goodput rate: maximum $\lambda$ over the SLO-feasible region; the optimization target | §5.1 |
| $\mathcal{F}_{\mathrm{SLO}}$ | Joint feasibility region in $(PP, TP, EP, SP, B)$ space (TPOT floor, TPOT bound, TTFT bound, HBM capacity all hold) | §4.1 |
| $\Delta_{\mathrm{dyn}}$ | Dynamic-stability cushion on $\overline{B}$ to absorb p99 batch-size variance: $\Delta_{\mathrm{dyn}} \approx z_p \sqrt{\overline{B}}$ for Poisson arrivals | §4.2 |
| $z_p$ | Standard-normal $p$-quantile: 1.28 for p90, 1.96 for p95, 2.33 for p99 | §4.2 |
| $C$ | Chunked-prefill chunk size (tokens per chunk) | §3.2 |
| $N_{\mathrm{chunks}}$ | Number of prefill chunks: $\lceil S_{\mathrm{input}} / C \rceil$ | §3.2 |
| $t_{\mathrm{chunk}}$ | Per-chunk prefill latency (roofline at $C$ tokens) | §3.2 |
| $t_{\mathrm{KV\text{-}transfer}}$ | Inter-cluster KV-cache transfer latency under disaggregation | §3.3 |
| $BW_{\mathrm{inter}}$ | Inter-cluster fabric bandwidth (disaggregation) | §3.3 |
| $M_{\mathrm{KV\text{-}transfer}}$ | Per-device KV-cache transfer volume under disaggregation: $\frac{2 \cdot S_{\mathrm{input}} \cdot H_{kv} \cdot b}{TP \cdot SP} \cdot \frac{L}{PP}$ | §3.3 |

