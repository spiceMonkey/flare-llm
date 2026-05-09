# Framework Overhead Model: Per-Phase Latency Constants

**Author:** Yue Lu  
**Date:** April 2026

---

The hardware roofline model in `decode.md` establishes a lower bound on per-token decode latency, governed by compute throughput and HBM bandwidth. Real serving systems, however, introduce additional latency above this bound that lives entirely on the host CPU and serving-software path: tokenization at request entry, request scheduling and batch assembly, token sampling, and response streaming. These are empirical constants that require profiling on the target system — they cannot be derived from hardware specifications alone. This document catalogs the CPU / software-stack overhead terms, explains where each comes from, and defines the per-request framework overhead $t_{\text{framework}}$ that enters end-to-end cost accounting in `e2e.md`. Kernel-launch and CUDA-Graph dispatch overheads are folded into the per-stage HW model (`decode.md §7.1`), not catalogued here. Hardware-level overheads tied to the network fabric (e.g., disaggregated KV transfer) are covered in `prefill.md §6`; memory-traffic calibration constants live in `decode.md`.

---

## Table of Contents

- [1. Overhead Classification](#1-overhead-classification)
- [2. CPU / Software-Stack Overhead Terms](#2-cpu--software-stack-overhead-terms)
  - [2.1 Tokenization Latency ($t_{\text{tok}}$)](#21-tokenization-latency-t_texttok)
  - [2.2 Request Scheduling / Batch Assembly ($t_{\text{sched}}$)](#22-request-scheduling--batch-assembly-t_textsched)
  - [2.3 Response Streaming / Detokenization ($t_{\text{detok}}$)](#23-response-streaming--detokenization-t_textdetok)
- [3. Total Framework Overhead Per Request](#3-total-framework-overhead-per-request)
- [4. Symbol Summary](#4-symbol-summary)

---

<div style="page-break-before: always;"></div>

## 1. Overhead Classification

The table below catalogs the CPU / software-stack overhead terms, organized by the serving phase in which each occurs, the symbol used throughout this suite, and representative ranges from published sources and hardware measurements. All terms are empirical — they depend on the serving framework, OS, and CPU/tokenizer library path, and must be profiled on the target deployment.

| Overhead | Phase | Symbol | Typical Range |
|----------|-------|--------|---------------|
| Tokenization | Prefill entry | $t_{\text{tok}}$ | ~0.1–2 ms |
| Request scheduling / batch assembly | Serving | $t_{\text{sched}}$ | ~10–200 µs |
| Response streaming / detokenization | Decode per-token | $t_{\text{detok}}$ | ~1–10 µs/token |

> **Out of scope for this document.** Kernel-launch / CUDA-Graph dispatch overhead is folded into the per-stage HW model and lives in `decode.md §7.1` (the $t_{\text{stage,sw}}$ term, composed inside $t_{\text{step,user}}$). The LM head $H \to V$ projection plus the post-LM-head sampling kernel (softmax + optional top-$k$/top-$p$ + multinomial draw) are similarly GPU-side per-step work and are modeled inside the per-step HW roofline as the once-per-step $t_{\text{LM,hw}}$ term on the last PP stage (`decode.md §2.1 / §3 / §6.2 / §7.2`). Disaggregated KV transfer latency (an α–β network-fabric term) is derived in `prefill.md §6.4`. Memory-traffic calibration constants used by the decode traffic model live in `decode.md`.

---

## 2. CPU / Software-Stack Overhead Terms

Each subsection below describes what the term measures, why it exists at the systems level, how to calibrate it on a target deployment, and typical values from published sources where available.

---

### §2.1 Tokenization Latency ($t_{\text{tok}}$)

**What it measures.** $t_{\text{tok}}$ is the wall-clock time from receiving a raw text request to the point where token IDs are ready and the first CUDA kernel for the prefill pass is launched. It includes byte-pair encoding (BPE) or SentencePiece lookup, Unicode normalization, and any pre-processing applied by the serving framework.

**Why it exists.** Tokenization runs on CPU in most serving stacks (vLLM, TensorRT-LLM, TGI). While GPU-side tokenization libraries exist, they are not universally deployed. The CPU→GPU handoff itself adds a small scheduling gap.

**Scaling behavior.** $t_{\text{tok}}$ scales weakly with input length: roughly $\mathcal{O}(S_{\text{input}})$ character-level operations, but constant factors are small (BPE lookup is table-driven). For $S_{\text{input}} \leq 4096$ tokens the cost is typically 0.1–2 ms and is often pipelined with the previous request's decode tail.

**Calibration.** Measure the latency from HTTP request receipt to the timestamp of the first `cudaLaunchKernel` call in a profiler trace (e.g., Nsight Systems). Vary $S_{\text{input}}$ to isolate the length-dependent component.

**Typical values.** 0.1–0.5 ms for short prompts ($S_{\text{input}} \leq 512$); up to 2 ms for very long prompts ($S_{\text{input}} \approx 8192$) on a single CPU thread. Parallelizing tokenization across CPU threads or pipelining it with KV allocation reduces the effective contribution to near zero for batch serving.

---

### §2.2 Request Scheduling / Batch Assembly ($t_{\text{sched}}$)

**What it measures.** $t_{\text{sched}}$ is the time the serving scheduler spends at the start of each decode iteration selecting which requests to include in the next batch, allocating or reclaiming KV cache pages, and preparing the batch metadata tensors (token IDs, position IDs, block tables).

**Why it exists.** Continuous batching (the standard production serving mode) requires the scheduler to run between every decode step. The scheduler must:

1. Check the priority queue of waiting requests.
2. Estimate whether a new request's KV cache pages fit in available HBM.
3. Run the PagedAttention block allocator [VLLM] to assign physical KV blocks.
4. Optionally, perform chunked-prefill piggybacking [SARATHI] for waiting requests.
5. Construct the attention metadata (block tables, sequence lengths) needed by the kernel.

All of this runs on CPU and must complete before the GPU decode kernel can launch.

**Calibration.** Instrument the scheduler loop with Python-level timestamps (`time.perf_counter_ns()`). Vary the number of concurrently active requests and the scheduler policy (FCFS, preemptive, chunked prefill) to characterize the dependence on batch size.

**Typical values.** 10–50 µs for small batches (B ≤ 8) with simple FCFS scheduling; 50–200 µs for large batches (B ≥ 64) with full PagedAttention block allocation and chunked-prefill scheduling.

---

### §2.3 Response Streaming / Detokenization ($t_{\text{detok}}$)

**What it measures.** $t_{\text{detok}}$ is the per-token cost of converting the sampled token ID back to a UTF-8 string fragment and transmitting it to the client (via HTTP chunked transfer or server-sent events).

**Why it exists.** Detokenization is the inverse of tokenization: a table lookup from token ID to byte sequence. It runs on CPU and is followed by a streaming write to the network socket. For most vocabulary sizes, the lookup itself is negligible; the dominant cost is the system call for the socket write plus any Python overhead in the serving framework.

**Scaling behavior.** $t_{\text{detok}}$ applies once per generated token and is effectively constant in token output length (it does not grow with $T_{\text{out}}$, only accumulates). The cumulative detokenization cost is $T_{\text{out}} \times t_{\text{detok}}$, which is small relative to the total decode time for any non-trivially sized model.

**Calibration.** Measure round-trip latency from GPU token ID to client receipt at the application layer, minus network round-trip time (RTT). The residual is approximately $t_{\text{detok}}$ (the post-LM-head sampling kernel is folded into $t_{\text{LM,hw}}$ on the GPU side, not into this term).

**Typical values.** ~1–10 µs per token for the detokenization lookup itself. Socket write latency varies with network configuration and is separate from this constant.

---

## 3. Total Framework Overhead Per Request

Combining all CPU / software-stack overhead terms, the total framework latency for a complete request consisting of a prefill pass followed by $T_{\text{out}}$ decode steps is:

$$
t_{\text{framework}} = t_{\text{tok}} + t_{\text{sched}} + T_{\text{out}} \cdot t_{\text{detok}}
$$

where the three groups are: tokenization (once, at start), scheduling (once per batch), and per-decode-step overhead (repeated $T_{\text{out}}$ times). Kernel-launch / dispatch overhead is *not* in this sum — it is folded into $t_{\text{step,user}}$ at the HW level via `decode.md §7.1`. The LM head GEMM and the sampling kernel are likewise GPU-side and absorbed into $t_{\text{LM,hw}}$ inside $t_{\text{step,user}}$ (`decode.md §7.2`).

**Notes on this decomposition:**

1. $t_{\text{sched}}$ is charged once per batch assembly event, not once per request. In continuous batching, a new batch is assembled every decode step, so $t_{\text{sched}}$ is effectively per-step for the active batch, not per-request. At the per-request level, it integrates to $T_{\text{out}} \times t_{\text{sched}}$ amortized over all requests in the batch.

2. Disaggregated KV-transfer latency is **not** included here; it is a network-fabric term handled in `prefill.md §6.4` as part of the prefill→decode handoff.

**Relationship to hardware model.** The full per-request latency is:

$$
t_{\text{total}} = t_{\text{TTFT}} + T_{\text{out}} \cdot t_{\text{step,user}} + t_{\text{framework}}
$$

where $t_{\text{TTFT}}$ and $t_{\text{step,user}}$ are defined in `prefill.md` and `decode.md` respectively. End-to-end assembly is in `e2e.md`.

**Order-of-magnitude comparison.** For a representative decode-heavy workload ($T_{\text{out}} = 512$, greedy sampling, co-located):

| Term | Per-step | Cumulative ($T_{\text{out}}=512$) |
|------|----------|----------------------------------|
| $t_{\text{step,user}}$ (hardware, includes kernel-launch budget per `decode.md §7.1` and $t_{\text{LM,hw}}$ per `decode.md §7.2`) | ~2–20 ms | 1–10 s |
| $t_{\text{detok}}$ | ~5 µs | ~2.5 ms |
| $t_{\text{tok}}$ (once) | — | ~0.5 ms |
| $t_{\text{sched}}$ (once, amortized) | — | ~0.1 ms |

Framework overhead is typically 1–5% of total request latency for well-optimized serving stacks on large models.

---

## 4. Symbol Summary

All symbols introduced in this document; these are consolidated into §13 of `notation.md`.

| Symbol | Definition | Units |
|--------|-----------|-------|
| $t_{\text{tok}}$ | Tokenization latency (CPU BPE/SP processing) | ms |
| $t_{\text{sched}}$ | Request scheduling / batch assembly latency | µs |
| $t_{\text{detok}}$ | Response streaming / detokenization latency per token | µs/token |
| $t_{\text{framework}}$ | Total CPU / software-stack overhead per request (§3) | ms |
| $T_{\text{out}}$ | Number of output tokens generated per request | tokens |

---

## References

**[VLLM]**  
Kwon et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023. arXiv:2309.06180.  
→ Scheduling loop latency; PagedAttention block allocation; batch assembly cost.

**[SARATHI]**  
Agrawal et al. (2023). *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills.* arXiv:2308.16369.  
→ Chunked prefill scheduling overhead; $t_{\text{sched}}$ characterization under mixed prefill–decode batching.
