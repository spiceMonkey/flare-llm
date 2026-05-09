# Interpreting ISL / OSL in Benchmark Data

**Author:** Yue Lu  
**Date:** May 2026

**Keywords:**  
LLM inference, benchmarking, ISL, OSL, decode context length, KV cache growth, roofline calibration, InferenceX, time per output token (TPOT)

---

# Table of Contents

- [1. Definitions](#1-definitions)
- [2. The Mismatch — Single $S$ vs Growing KV](#2-the-mismatch--single-s-vs-growing-kv)
- [3. Three Choices of Representative $S$](#3-three-choices-of-representative-s)
- [4. When the Choice Matters](#4-when-the-choice-matters)
- [5. Practical Recipe](#5-practical-recipe)
- [6. Worked Example](#6-worked-example)
- [7. What to Do When the Benchmark Reports a Range](#7-what-to-do-when-the-benchmark-reports-a-range)
- [8. Key Takeaways](#8-key-takeaways)

---

# 1. Definitions

Public inference benchmarks (InferenceX / InferenceMAX, MLPerf-Inference, vLLM benchmarks, NVIDIA GenAI-Perf) report request-level latency and throughput numbers parameterized by two integers: **ISL** (input sequence length) and **OSL** (output sequence length). The benchmark sends a prompt of **ISL** tokens, then requests exactly **OSL** generated tokens, and reports per-step decode latency (Time Per Output Token, TPOT) along with end-to-end metrics. Common presets are "1k/1k", "8k/1k", "1k/8k" — meaning ISL/OSL pairs of (1024, 1024), (8192, 1024), (1024, 8192).

The analytical decode roofline derived in `documentation/modeling/decode.md` is parameterized by a single context length $S$ — the number of tokens whose key/value entries the per-step attention scan must read from cache. The ISL / OSL pair from a benchmark and the single $S$ in the roofline don't line up trivially: real decode walks through a *range* of context lengths, while the roofline produces *one* number per call. This document explains the mismatch, names the three reasonable substitutions, and provides a recipe for cross-checking analytical predictions against benchmark TPOT.

---

# 2. The Mismatch — Single $S$ vs Growing KV

A benchmark request executes in two phases. **Prefill** runs once and processes all ISL prompt tokens in parallel — this contributes to time-to-first-token (TTFT), not TPOT, and is out of scope for this document. **Decode** then runs OSL times, generating one new token per step. Each decode step appends one token to the key/value (KV) cache, so the cache grows monotonically:

| Decode step $i$ (1-indexed) | KV cache length at start of step | Tokens attended to by this step |
|---|---|---|
| 1 (first decoded token) | ISL | ISL |
| 2 | ISL + 1 | ISL + 1 |
| ... | ... | ... |
| OSL (last decoded token) | ISL + OSL − 1 | ISL + OSL − 1 |

Per-step decode cost grows with the cache length at each step (`context_length_impact.md §2` — KV traffic and attention FLOPs both scale linearly with $S$). The benchmark's reported TPOT is a single number summarizing all OSL of these per-step costs, typically reported as the median or the mean over the OSL window. The analytical roofline, by contrast, is evaluated at one $S$ per call — it cannot natively model the $i$-by-$i$ growth.

So a faithful comparison requires choosing one representative $S$ to feed into the roofline so the resulting TPOT matches what the benchmark averages over.

---

# 3. Three Choices of Representative $S$

There are three defensible ways to pick a single $S$ to feed into the roofline:

| Choice | $S$ | When it's the right choice |
|---|---|---|
| **First-step** | ISL | Comparing against the benchmark's *first* decode TPOT (occasionally reported separately) |
| **Mid-window** | ISL + OSL/2 | Comparing against the benchmark's *median* or *mean* TPOT — the standard reporting convention |
| **Last-step** | ISL + OSL − 1 | Worst-case prediction; comparing against the benchmark's p99 TPOT or the late-decode regime |

The mid-window choice is the right default for cross-checking against a typical benchmark TPOT column. It corresponds to the median position in a uniform sweep from ISL to ISL + OSL − 1, which lines up with how benchmarks report median TPOT when the per-step cost is roughly linear in $S$ (the decode roofline's KV-bound regime).

The first-step choice gives the most optimistic prediction. The last-step choice gives the most pessimistic. These bracket the realistic range and are useful when sensitivity-checking how much the analytical curve can move just from the $S$-ambiguity.

---

# 4. When the Choice Matters

The TPOT spread across the three choices depends on which decode regime the deployment is in (`context_length_impact.md §2.2 / §2.3`):

- **KV-bound regime** ($B$ large enough that KV traffic dominates the per-step cost). Per-step time grows linearly with $S$, so the spread between first-step and last-step is $(S_{\text{last}} - S_{\text{first}}) / S_{\text{first}}$, which equals OSL / ISL when ISL ≫ 1. For ISL = 1024, OSL = 1024 this is a 100% spread. For ISL = 8192, OSL = 1024 it shrinks to ~12%. **The choice matters most when OSL is comparable to or larger than ISL.**

- **Weight-bound regime** ($B$ small, weights dominate per-step cost). Per-step time is nearly $S$-invariant. The three choices give nearly identical TPOT predictions, and the choice doesn't matter — pick whichever is convenient.

- **Attention-FLOP-bound regime** ($B$ very large or compute-bound for attention). Per-step compute also grows linearly with $S$ in the attention term — same spread shape as KV-bound. This regime is uncommon in production decode but appears at very long context.

Quick test: compute the predicted TPOT at $S = $ ISL and at $S = $ ISL + OSL. If the two are within ~5% of each other, the deployment is weight-bound and the $S$-choice doesn't matter for this comparison. If the spread is larger, document which choice was used and report the bracketed range.

---

# 5. Practical Recipe

When cross-checking an analytical TPOT prediction against a benchmark's median TPOT for a given (ISL, OSL) pair:

1. **Default** to $S = $ ISL + OSL/2. Compute the analytical TPOT at this $S$ and compare against the benchmark's median.
2. **Bracket** by also computing TPOT at $S = $ ISL and at $S = $ ISL + OSL. Report the analytical prediction as a range when the spread exceeds 10%.
3. **State the choice in plots and tables.** Annotate "S = ISL + OSL/2" (or whichever) on every overlay so the reader knows the analytical curve is a single point sample, not a sweep.
4. **For p99 comparisons,** use $S = $ ISL + OSL − 1 (the last decode step is also typically the slowest one; p99 TPOT lives near the worst-case).
5. **For TTFT,** $S$ is irrelevant — TTFT is a prefill quantity (`documentation/modeling/prefill.md`).

---

# 6. Worked Example

Take the InferenceX preset (ISL = 1024, OSL = 1024) on DeepSeek-R1 / GB200 NVL72 / FP4 / TP=32 EP=32. Benchmark reports median TPOT around 15 ms per the spec=none rows.

Analytical predictions at the three $S$ choices, using the §6.2 roofline of `decode.md`:

| Choice | $S$ | Predicted TPOT (rough, depends on $B$) |
|---|---|---|
| First-step | 1024 | ~4.5 ms |
| Mid-window | 1536 | ~4.9 ms |
| Last-step | 2047 | ~5.3 ms |

The 4.5–5.3 ms range from $S$-choice alone is small (~18%) compared to the 3× gap between the analytical roofline and the measured 15 ms — the dominant gap drivers in this comparison are sustained HBM bandwidth, all-to-all efficiency, and serving-runtime overhead, not the $S$-choice ambiguity.

Take a different preset with larger OSL, ISL = 1024 OSL = 8192. The same model:

| Choice | $S$ | Predicted TPOT |
|---|---|---|
| First-step | 1024 | ~4.5 ms |
| Mid-window | 5120 | ~7.6 ms |
| Last-step | 9215 | ~11.0 ms |

Now the $S$-choice spread is ~140% — the analytical TPOT range across the OSL window is wider than the analytical-vs-measured gap. Reporting a bracketed range becomes essential, and ideally the comparison plot should show the analytical curve at all three $S$ choices as a shaded band rather than a single line.

---

# 7. What to Do When the Benchmark Reports a Range

Some benchmarks (vLLM's bench scripts, NVIDIA GenAI-Perf with full distribution output) report the per-step TPOT distribution rather than a single summary. In that case:

1. **Plot the measured TPOT distribution** (median, p50–p99 box) along the OSL axis if available. Most benchmarks don't expose per-step measurements, so this often reduces to (1) a median value plus (2) a p99 value at the request level.
2. **Plot the analytical TPOT sweep** $S = $ ISL → ISL + OSL − 1 as a curve — this captures the per-step growth that the benchmark's distribution implicitly contains.
3. **Match summary statistics.** The benchmark median should sit near the analytical curve evaluated at $S = $ ISL + OSL/2; the benchmark p99 should sit near the analytical curve at $S = $ ISL + OSL − 1.

When the benchmark only reports a single TPOT number (the typical case for InferenceX and MLPerf-Inference), step 1 is unavailable and the §5 recipe is the best you can do.

---

# 8. Key Takeaways

- ISL is the prompt length; OSL is the generated output length. Both are integers reported per benchmark preset. ISL drives prefill and TTFT; OSL drives the *number* of decode steps.
- Each decode step attends to a KV cache of length ISL through ISL + OSL − 1. The benchmark's TPOT is a summary statistic over this range.
- The analytical roofline takes one $S$ per call. Default to $S = $ ISL + OSL/2 for matching median TPOT; bracket with $S = $ ISL and $S = $ ISL + OSL when reporting.
- The choice matters most when OSL is comparable to or larger than ISL, and when the deployment is in the KV-bound or attention-FLOP-bound regime. In weight-bound regime the choice is immaterial.
- Always annotate the $S$ choice on overlay plots and comparison tables. Treat the analytical-vs-measured gap and the $S$-choice ambiguity as separate sources of uncertainty.

---

## References

`documentation/modeling/decode.md` — full decode roofline derivation; `S` enters via the KV traffic term (`§2.3`) and the attention FLOPs term (`§3.2`).

`documentation/modeling/prefill.md` — TTFT model; ISL drives this phase, not OSL.

`documentation/explaining/context_length_impact.md` — how decode cost scales with `S` and where the regime boundaries sit.
