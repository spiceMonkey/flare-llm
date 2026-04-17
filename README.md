# llm_perf

`llm_perf` is a lightweight, first-principles analytical framework for large-language-model inference performance modeling. It predicts latency, throughput, and memory footprint of LLM inference on a given cluster *before* you build or rent it — from a JSON description of the model, the hardware, the parallelism layout, and a handful of tuning knobs.

The core is a five-stage pipeline (memory → FLOPs → traffic → comm → latency) extended with prefill, end-to-end metric assembly, KV paging, chunked prefill, and disaggregated prefill/decode. Everything is composable pure functions over typed dataclasses — no global state, no training-specific baggage.

---

## Modeled Architecture

![LLM inference cluster architecture](assets/cluster_architecture.png)

The diagram above shows the components of an LLM inference cluster that `llm_perf` models analytically. The system is organized as a disaggregated prefill/decode pipeline with a shared distributed KV cache underneath.

**Serving Framework** sits at the top of the stack — continuous batching, request scheduling, tokenization, and KV-aware routing. The framework contributes per-request and per-step CPU overhead (scheduler dispatch, CUDA graph replay, token sampling, detokenization) modeled via `OverheadSpec` and folded into E2E latency. See [`modeling/framework.md`](documentation/modeling/framework.md).

**Prefill Cluster** is the compute-heavy phase that processes the full input prompt in one (or multiple chunked) forward passes. Each device runs the same transformer layers but at sequence-length `S` rather than single-token decode. Prefill FLOPs scale quadratically with `S` in the attention block and linearly in the FFN/projection layers. See [`modeling/prefill.md`](documentation/modeling/prefill.md) and [`core/prefill_model.py`](llm_perf/core/prefill_model.py).

**Decode Cluster** is the memory-bandwidth-bound autoregressive phase. Devices are connected via a scale-up switch that carries TP, EP, and SP collectives; pipeline-parallel (PP) stages communicate via point-to-point sends. Data parallelism (DP) replicates the full pipeline to increase throughput without affecting per-request latency. The roofline model inside each device balances compute time against HBM read time, and the overlap-aware latency model hides communication behind whichever is the bottleneck. See [`modeling/tpot.md`](documentation/modeling/tpot.md) and [`core/latency_model.py`](llm_perf/core/latency_model.py).

**KV Transfer** interconnect bridges the two clusters in a disaggregated deployment. When prefill and decode run on separate device groups, the KV cache produced during prefill must be shipped to the decode cluster before autoregressive generation can begin. The transfer cost (startup latency α + bulk BW) is modeled in [`modeling/e2e.md`](documentation/modeling/e2e.md) and can dominate TTFT for short prompts or low-bandwidth fabrics.

**Distributed KV Cache** spans HBM, host DRAM, and SSD tiers. `llm_perf` models PagedAttention-style block accounting — block size, per-sequence block count, internal fragmentation, and effective HBM capacity after subtracting weights and activations — to determine the maximum concurrent-sequence batch a given partition can serve. See [`modeling/kv.md`](documentation/modeling/kv.md) and [`core/kv_paging_model.py`](llm_perf/core/kv_paging_model.py).

The **scale-up switch** within each cluster carries collective traffic for tensor parallelism (TP), expert parallelism (EP), and sequence parallelism (SP). The switching model accounts for effective per-port bandwidth under aggregate capacity constraints, latency (α), and the algorithm choice (ring vs. tree). See [`modeling/switching.md`](documentation/modeling/switching.md) and [`core/comm_model.py`](llm_perf/core/comm_model.py).

---

## Repository Layout

```
.
├── README.md                         — this file
├── quickstart.ipynb                  — tutorial: load specs, run the full stack
├── pareto_basic.ipynb                — full (partition, B) exploration space  (case study)
├── pareto_vs_cluster_size.ipynb      — decode Pareto × cluster size (N)        (case study)
├── pareto_vs_io.ipynb                — decode Pareto × scale-up I/O sweep      (case study)
├── pareto_vs_mem.ipynb               — decode Pareto × HBM-BW sweep            (case study)
├── pareto_vs_flops.ipynb             — decode Pareto × peak-FLOPS sweep        (case study)
├── pareto_vs_overhead.ipynb          — decode Pareto × framework overhead      (case study)
├── ttft_vs_io.ipynb                  — TTFT × mismatched-partition disagg I/O  (case study)
├── ttft_vs_chunk.ipynb               — TTFT × chunk-size sweep (co-lo)         (case study)
├── documentation/
│   ├── modeling/                     — methodology derivations (notation, tpot, prefill, e2e, kv, framework, dram3d)
│   └── explaining/                   — design-intent walkthroughs
├── scripts/convert_hf_model.py       — HF→llm_perf model converter
└── llm_perf/
    ├── calculators/
    │   ├── inference_calculator.py   — decode-phase orchestration
    │   ├── prefill_calculator.py     — prefill-phase orchestration
    │   └── e2e_calculator.py         — TTFT/TPOT/throughput assembly
    ├── core/
    │   ├── memory_model.py           — M_θ, M_act, M_kv, fits_in_HBM
    │   ├── flops_model.py            — F_token, F_prefill
    │   ├── traffic_model.py          — T_θ, T_act, T_kv
    │   ├── comm_model.py             — TP/EP/SP/PP collective times
    │   ├── latency_model.py          — roofline + overlap-aware t_token, TPOT, B*
    │   ├── prefill_model.py          — prefill stack incl. chunked prefill
    │   └── kv_paging_model.py        — paged-attention block accounting
    ├── database/                     — model / system / partition / tuner JSONs
    ├── specs/                        — LlmModelSpec, SystemSpec, PartitionSpec, TuningSpec, OverheadSpec, DisaggSpec
    ├── io/                           — JSON loaders + list helpers per schema
    └── utils/                        — constants, equations, HF adapter, DRAM3D helper, plotting
```

---

## Quickstart

```bash
python -m venv .llm_perf
source .llm_perf/bin/activate
pip install jupyter matplotlib numpy
jupyter notebook quickstart.ipynb
```

The quickstart walks through discovery, loading, running `InferenceCalculator`, and inspecting the memory/FLOPs/traffic/comm/latency breakdown.

### Programmatic usage

```python
from llm_perf import InferenceCalculator
from llm_perf.calculators.prefill_calculator import PrefillCalculator
from llm_perf.calculators.e2e_calculator import E2ECalculator
from llm_perf.io import load_model_spec, load_system_spec, load_tuning_spec
from llm_perf.specs.partition_spec import PartitionSpec
from llm_perf.specs.overhead_spec import OverheadSpec
from llm_perf.specs.disagg_spec import DisaggSpec

model     = load_model_spec("llm_perf/database/model/gpt_1_8t_moe.json")
system    = load_system_spec("llm_perf/database/system/gb200.72gpu.json")
tuner     = load_tuning_spec("llm_perf/database/tuner/gpt_1_8t_moe.tuner.json")
partition = PartitionSpec(PP=8, TP=8, EP=1, SP=1)
tuner.S_input, tuner.S_decode, tuner.B_decode = 8192, 8192, 1

decode   = InferenceCalculator(model, system, partition, tuner).run()
prefill  = PrefillCalculator(model, system, partition, tuner).run()
e2e      = E2ECalculator(
    decode, prefill,
    overhead=OverheadSpec(t_graph_us=100.0),   # CUDA graph overhead
    disagg=DisaggSpec(),                        # co-lo, matched partition
    model=model, system=system, partition=partition, tuner=tuner,
).run()

print(f"TTFT       = {e2e.TTFT*1e3:.1f} ms")
print(f"TPOT       = {e2e.TPOT*1e3:.2f} ms")
print(f"tok/s/GPU  = {e2e.throughput_per_gpu:.1f}")
```

---

## Case Studies

Each notebook is a self-contained design question with a plot and a short takeaway. They're meant as reading material — a reader can step through the cells to understand how a specific decision (partition, I/O BW, HBM BW, overhead, chunk size, disagg) shapes the end-to-end metric that matters.

All seven case studies use **GPT-1.8T MoE @ FP4** on **GB200-class devices** (NVL72 baseline; cluster-size study extends past 72 GPUs hypothetically).

### `pareto_basic.ipynb` — the full exploration space behind the frontier

![full (partition, B) exploration cloud vs. extracted frontier](assets/pareto_basic.png)

*Question: where does the Pareto frontier come from? What does the underlying point cloud look like?*

Enumerates every valid `(PP, TP, EP, SP)` partition, sweeps `B` from 1 to the KV-paging max per partition, then extracts the upper-right envelope in (interactivity, throughput/GPU) space. Left panel shows the full cloud with the frontier overlaid; right panel colors the same cloud by pipeline parallelism (PP) so the regime segmentation is visible.

**Headline:** at baseline GB200 NVL72, **91 valid partitions → 2,247 `(partition, B)` evaluations → 38 frontier points (~1.7% of the cloud)**. Of those 38, `PP=8 TP=8 EP=1` claims 34 and `PP=6 TP=4 EP=1` claims the remaining 4. PP dominates regime selection: shallow PP sits in the high-interactivity corner (low per-GPU throughput, small B), deep PP in the high-throughput corner (large B amortizes warmup). The later notebooks (`pareto_vs_io`, `pareto_vs_mem`, `pareto_vs_overhead`) re-run this exact enumeration once per hardware/overhead anchor and plot only the frontier — this notebook is what's underneath.

### `pareto_vs_cluster_size.ipynb` — decode Pareto under cluster-size scaling

![decode Pareto vs. cluster size](assets/pareto_vs_cluster_size.png)

*Question: as the cluster grows from 64 to 1024 GPUs (per-device hardware held fixed), how does the frontier shift — and what happens when larger fabrics lose effective bandwidth?*

Enumerates every valid `(PP, TP, EP, SP)` partition with `PP·TP·EP·SP ≤ N` for each `N ∈ {64, 128, 256, 512, 1024}`, sweeps `B` per partition, extracts the frontier per `N`. Solid curves assume ideal fabric (η=1.0); dashed curves apply a diminishing fabric efficiency η for large clusters (η=0.90 at N=256, 0.80 at N=512, 0.70 at N=1024) to model head-of-line blocking, buffering contention, and protocol overhead at scale.

**Headline:** winning `PP` climbs to `L=120` (one layer per rank) by `N=128`, then further devices go to TP. Winning TP steps **1 → 1 → 2 → 4 → 8** as N doubles from 64 → 1024. EP stays at 1 throughout — MoE routing overhead outweighs expert parallelism at this scale and batch size. With diminishing η, the frontier at large N pulls inward — the gap between ideal and η-discounted grows with cluster size, quantifying the cost of fabric inefficiency on achievable throughput and interactivity.

| N    | dominant winner (× points) | replica | DP | util |
|------|----------------------------|---------|----|------|
| 64   | `PP=60  TP=1` (×25)       | 60  | 1  | 93.8% |
| 128  | `PP=120 TP=1` (×26)       | 120 | 1  | 93.8% |
| 256  | `PP=120 TP=2` (×30)       | 240 | 1  | 93.8% |
| 512  | `PP=120 TP=4` (×26)       | 480 | 1  | 93.8% |
| 1024 | `PP=120 TP=8` (×22)       | 960 | 1  | 93.8% |

Device utilization is the silent cost — `DP = N // replica` is floored, so partitions whose replica doesn't divide `N` waste devices. All listed `N` land on ≥93.8% util shapes, but "in-between" sizes (e.g. N=768) force partial-utilization choices and produce diverse but less efficient frontiers. **Practical rule:** pick cluster sizes that are multiples of `L` (or its divisors like 60) to land on high-utilization frontiers.

### `pareto_vs_io.ipynb` — decode Pareto under scale-up I/O provisioning

![decode Pareto vs. scale-up I/O](assets/pareto_vs_io.png)

*Question: how does the partition-optimal Pareto frontier move as you vary scale-up NVLink bandwidth and α?*

Sweeps BW (1× → ~2.67× GB200 baseline) and α (1.0× → 0.25× baseline) as two panels. Enumerates all valid (PP, TP, EP, SP) partitions, finds the upper-right envelope in (interactivity, throughput/GPU) space, annotates winners.

**Headline:** the frontier shifts smoothly with I/O provisioning but winning partitions re-order at corners — low-BW regimes favor shallower PP and more TP locality; high-BW regimes favor deeper PP that exploits cheap cross-stage comm.

### `pareto_vs_mem.ipynb` — decode Pareto under HBM-BW scaling

![decode Pareto vs. HBM bandwidth](assets/pareto_vs_mem.png)

*Question: as HBM bandwidth grows (DRAM-3D / stacked-memory trajectory), which partition wins?*

Sweeps HBM BW from 1× (8 TB/s, baseline) to 4× (32 TB/s) at fixed scale-up I/O and FLOPS.

**Headline:** optimal TP shrinks from 8 → 1 as HBM BW grows.

| HBM BW | Dominant winner (×points on frontier) |
|---|---|
| 1× (8 TB/s)  | `PP=8 TP=8 EP=1` × 34 |
| 2× (16 TB/s) | TP begins shifting down; diversity grows |
| 4× (32 TB/s) | `PP=8 TP=4` variants claim 17 of the frontier |
| ideal (∞)   | `PP=6–8 TP=1` — TP collective is pure overhead |

Scarce memory bandwidth favors wide TP to parallelize weight reads; abundant memory bandwidth makes the TP collective pure overhead.

### `pareto_vs_flops.ipynb` — decode Pareto under peak-FLOPS scaling

![decode Pareto vs. peak FLOPS at two context lengths](assets/pareto_vs_flops_short_vs_long.png)

*Question: if GPU peak FLOPS grew without any change to HBM or scale-up I/O, how much further would the decode Pareto frontier push?*

Sweeps peak FLOPS from 0.5× (H100-class ~4.5 PF) to 4× (~36 PF) at fixed HBM (8 TB/s) and scale-up I/O, plus an `ideal compute` (FLOPS → ∞) reference. Runs the same sweep at two context lengths to expose the regime boundary.

**Headline (split by regime):**
- **At `S`=8192 (left): all five curves overlap exactly** — peak FLOPS is a non-knob. `t_mem / t_compute` is 6×–1400× across every `B`, so the memory-bound asymptote pins the frontier. More FLOPS only widens the machine balance point further from the workload's per-`B` arithmetic intensity.
- **At `S`=1024 (right): the right corner fans out** — shorter context shrinks `T_kv`'s slope in `B` ~8×, pushing per-`B` AI above GB200's 1125 FLOPs/byte balance. At `B`=8192 on 1× FLOPS the step flips compute-bound; 4× FLOPS then lifts high-`B` throughput/GPU from ~7850 → ~9550, tracing the ideal-compute ceiling.

**Complementary read with `pareto_vs_mem`:** at long `S` this workload is memory-bound end-to-end, so HBM BW moves both corners and FLOPS moves neither. At short `S`, the corners split — memory BW still drives the high-interactivity corner, FLOPS drives the high-throughput corner.

**Prefill / training are the opposite regime** — their attention compute grows as `S²` while KV traffic grows as `S`, putting arithmetic intensity orders of magnitude above any hardware balance point. FLOPS is the primary knob for TTFT and for training throughput. See `documentation/explaining/why_flops_doesnt_help_at_long_context.md` for the full AI / slope derivation and the decode-vs-prefill-vs-training contrast, and `documentation/explaining/frontier_convergence_at_high_b.md` for the $B^*$ story.

### `pareto_vs_overhead.ipynb` — decode Pareto under framework overhead

![decode Pareto vs. framework overhead](assets/pareto_vs_overhead.png)

*Question: how does per-step framework overhead (Python scheduler, CUDA graph replay, sampling, detokenization) bend the frontier? Does it change which partition wins?*

Applies framework overhead post-hoc — runs the hardware sweep once per partition, then re-prices per overhead value. Six anchors map to real serving stacks:

| `t_oh` | Framework regime |
|---|---|
| 0 μs | Ideal / theoretical lower bound |
| 100 μs | TensorRT-LLM / SGLang / vLLM v1 (CUDA graphs + persistent + async scheduler) |
| 500 μs | Production vLLM / well-tuned TGI (CUDA graphs + continuous batching) |
| 1 ms | vLLM v0 / default TGI (partial graph capture) |
| 2 ms | Eager-mode serving or heavy Python scheduler |
| 5 ms | Unoptimized HF `generate()` loop |

**Headline:** overhead is an **asymmetric tax** — it crushes the high-interactivity (small B) corner but barely moves the high-throughput corner. Despite that, the winning partition at every corner is **stable** across all six overhead values — overhead shifts you along the frontier but does not re-order partition choice.

### `ttft_vs_io.ipynb` — mismatched-partition disaggregation: does it pay off?

![TTFT vs. inter-cluster BW/α under mismatched-partition disagg](assets/ttft_vs_io.png)

*Question: if prefill and decode clusters use different partitions (prefill shape optimized for its compute profile, decode shape optimized for its own), when does the KV handoff cost get paid back by faster prefill?*

Fixes decode partition at `PP=8 TP=8 EP=1 SP=1` (the decode-Pareto winner) and compares two mismatched prefill shapes vs. co-lo reference across the **2–32k commercial prompt band** (anchored to ShareGPT / Splitwise / DistServe / Mooncake traces):

- **Mismatch A** — wide-TP prefill (`PP=1 TP=16 EP=4`): PP/TP/EP **all differ** from decode.
- **Mismatch B** — MoE-EP prefill (`PP=2 TP=8 EP=4`): PP+EP differ; TP matches.

Sweeps inter-cluster BW (50 GB/s → 3.6 TB/s) and α (0.5 μs → 5 ms) as two panels.

**Headline:** in the 2–32k commercial band, **mismatched-partition disagg doesn't pay off** — at any BW, at any α. Compute savings from the mismatched prefill partitions are 0.4–8 ms; the KV handoff tax exceeds those savings everywhere. The win case is long-context (64k+) workloads (Mooncake P99 territory) where wide-TP prefill's attention-FLOP savings materialize. For the bulk of commercial traffic, matched-partition co-lo — or disagg-with-matched-partition for scheduling benefits — is the simpler choice.

### `ttft_vs_chunk.ipynb` — chunked prefill sweet-spot

![TTFT vs. chunk size across 2k–32k prompts](assets/ttft_vs_chunk.png)

*Question: what chunk size minimizes TTFT for chunked prefill, and how does the sweet spot shift with prompt length?*

Fixes partition at `PP=8 TP=8 EP=1 SP=1` (co-lo, matched). Sweeps chunk size log-spaced from 128 tokens up to `S_input` across the same 2–32k band.

**Headline:** a universal sweet spot of `C* ≈ 2048` tokens across the entire commercial band.

| S_input | Workload | C\* | n_chunks | TTFT\* | vs. single-pass |
|---|---|---|---|---|---|
| 2k | chat+hist     | 2048 | 1 | 3.8 ms   | 5.4× |
| 4k | code/RAG      | 2048 | 2 | 7.5 ms   | 5.6× |
| 8k | prod+RAG      | 2048 | 4 | 15.1 ms  | 5.9× |
| 16k | long-doc     | 2048 | 8 | 31.1 ms  | 6.3× |
| 32k | reason/agent | 2048 | 16 | 66.5 ms | 7.0× |

The U-shape is genuine — small C (128) pays `n_chunks × T_θ` weight re-reads; large C pays quadratic attention; optimum sits near 1–2k tokens. A single hard-coded `C = 1-2k` is within 5–10% of optimal across the full band, so production engines don't need per-request tuning. Most of the headline speedup is from avoiding the PP warmup tax that single-pass prefill pays fully; within the chunked regime itself the win is a more modest ~1–3%.

---

## Utilities

**HuggingFace Adapter** — converts any HuggingFace `config.json` (including MoE and GQA variants) into the `llm_perf.model` schema so it can be used directly with the calculators. Available as a library call (`convert_hf_config_to_model_json()`) or as a CLI script (`python scripts/convert_hf_model.py`). See [`utils/hf_model_adapter.py`](llm_perf/utils/hf_model_adapter.py).

**DRAM3D Bandwidth Calculator** — derives HBM bandwidth from physical die-interface parameters (die area, bump pitch, data-pin fraction, data rate, number of dies) to evaluate future memory classes (HBM3E, HBM4, HBM4E) before silicon is available. Can also update an existing `database/system/*.json` file in place with the computed bandwidth. See [`utils/dram3d.py`](llm_perf/utils/dram3d.py) and [`modeling/dram3d.md`](documentation/modeling/dram3d.md).

---

## Contributing

- Open issues or PRs for new spec types, adapters, or analytical improvements.
- Keep JSON schemas backward compatible when possible.
- Run the quickstart notebook after large changes to confirm the pipeline still loads and runs.

---

## License

MIT — see [LICENSE](LICENSE).
