# InferenceX benchmark dataset

A snapshot of the public InferenceX™ benchmark dataset, used by the llm_perf
modeling suite for cross-validation against measured production-stack
performance.

**Provenance.** All data here comes from the InferenceX public API, one request per model: `https://inferencex.semianalysis.com/api/v1/benchmarks?model=<model>` (for example `?model=gpt-oss-120b`; the endpoint answers HTTP 400 without the `model` parameter). InferenceX is maintained by SemiAnalysis LLC and licensed under Apache 2.0 (<https://github.com/SemiAnalysisAI/InferenceX>); its license names the copyright holders as "Copyright 2025 SemiAnalysis LLC, Advanced Micro Devices, NVIDIA CORPORATION". See `LICENSE` and `NOTICE` in this directory for the upstream license text and attribution requirement.

**Two snapshots.** `data/` is the snapshot of 9 May 2026, and every validation number this repository publishes is computed from it. `data_2026-11/` is a second snapshot fetched in November 2026 and kept beside the first instead of replacing it, because the upstream dataset is not append-only: between the two fetches it gained rows for new hardware and new cuts, and it also dropped or replaced rows the validators use (for example, the DeepSeek-R1 cuts on GB200 with Dynamo and TensorRT-LLM that the DeepSeek-R1 driver reads went from 43 points to 14). Keeping the first snapshot keeps the published numbers reproducible; the second serves new cuts. `python benchmark/inferenceX/fetch.py --snapshot <name>` writes a further one without touching `data/`, and the loader in `benchmark/validate/common.py` takes `snapshot="2026-11"`.

## What's here

```
inferenceX/
├── LICENSE                Apache 2.0 (verbatim from upstream)
├── NOTICE                 Attribution + provenance requirements
├── README.md              This file
├── SCHEMA.md              Per-field documentation of the BenchmarkRow schema
├── fetch.py               Refresh script — re-runs the API pull
├── summary.json           Snapshot metadata (fetch date, per-model row counts)
└── data/
    ├── raw/               One JSON array per model — verbatim API response
    │   ├── DeepSeek-R1-0528.json
    │   ├── DeepSeek-V4-Pro.json
    │   ├── GLM-5.json
    │   ├── Kimi-K2.5.json
    │   ├── Llama-3.3-70B-Instruct-FP8.json
    │   ├── MiniMax-M2.5.json
    │   ├── Qwen-3.5-397B-A17B.json
    │   └── gpt-oss-120b.json
    └── flat/              One CSV per model — metrics flattened to top-level
        └── (same eight files as .csv)
├── summary_2026-11.json   Metadata of the second snapshot
└── data_2026-11/
    └── flat/              The same eight models, fetched in November 2026 (CSV only)
```

The raw JSON preserves the upstream schema verbatim (with the metric
quantiles nested under `metrics`); the flat CSV hoists those quantiles to
top-level columns for easy slicing with pandas / awk / spreadsheets. Pick
whichever is convenient — they contain the same information.

## Models in the snapshot

Counts are from the most recent `fetch.py` run; check `summary.json` for the
exact timestamp and per-model breakdowns.

| Model | Architecture | Where to look |
|---|---|---|
| **DeepSeek-R1-0528** | MoE + MLA, 671B / 37B active | `data/{raw,flat}/DeepSeek-R1-0528.{json,csv}` |
| **DeepSeek-V4-Pro** | MoE + MLA (DSv4) | `data/{raw,flat}/DeepSeek-V4-Pro.{json,csv}` |
| **GLM-5** | MoE | `data/{raw,flat}/GLM-5.{json,csv}` |
| **Kimi-K2.5** | MoE | `data/{raw,flat}/Kimi-K2.5.{json,csv}` |
| **Llama-3.3-70B-Instruct-FP8** | Dense GQA, 70B | `data/{raw,flat}/Llama-3.3-70B-Instruct-FP8.{json,csv}` |
| **MiniMax-M2.5** | MoE | `data/{raw,flat}/MiniMax-M2.5.{json,csv}` |
| **Qwen-3.5-397B-A17B** | MoE, 397B / 17B active | `data/{raw,flat}/Qwen-3.5-397B-A17B.{json,csv}` |
| **gpt-oss-120b** | MoE, 120B | `data/{raw,flat}/gpt-oss-120b.{json,csv}` |

The set of models served by the API changes over time. Run
`python fetch.py --discover` to re-scan the dashboard JS for new
identifiers; add anything new to the `MODELS` list at the top of `fetch.py`.

## Hardware coverage in the snapshot

The hardware/framework mix varies per model. Common keys observed across
the dataset:

| Hardware | Where it shows up |
|---|---|
| `b200` | NVIDIA HGX/DGX B200 (Blackwell, 8 GPU per server) |
| `b300` | NVIDIA HGX/DGX B300 (Blackwell Ultra) |
| `gb200` | NVIDIA GB200 NVL72 (rack-scale Blackwell + Grace) |
| `gb300` | NVIDIA GB300 NVL72 |
| `h100`, `h200` | NVIDIA HGX/DGX Hopper |
| `mi300x`, `mi325x`, `mi355x` | AMD Instinct |
| `tpu-v5p` | Google Cloud TPU v5p |

| Framework | What it is |
|---|---|
| `trt`, `trt-llm`, `trtllm` | NVIDIA TensorRT-LLM (raw, no Dynamo) |
| `dynamo-trt`, `dynamo-trt-llm` | NVIDIA Dynamo serving stack on top of TRT-LLM |
| `vllm` | vLLM |
| `sglang` | SGLang |
| `dynamo-sglang` | NVIDIA Dynamo on SGLang |
| `mori-sglang` | AMD Mori-tuned SGLang |
| `atom` | AMD-optimized stack |

## How to refresh

```bash
# Refresh every model in the MODELS list:
python benchmark/inferenceX/fetch.py

# Refresh a single model:
python benchmark/inferenceX/fetch.py --model gpt-oss-120b

# Re-scan the dashboard for new model identifiers:
python benchmark/inferenceX/fetch.py --discover
```

Refresh overwrites the on-disk files; the upstream API periodically
backfills and corrects measurements, so running `fetch.py` periodically is
the intended pattern. `summary.json` records the `fetched_at_utc` timestamp.

## Schema

See `SCHEMA.md` for per-field documentation of the BenchmarkRow.

## License and attribution

Apache License 2.0 — see `LICENSE`. Per the upstream's attribution policy
(see `NOTICE`), any work derived from this dataset must cite InferenceX as
the source and label third-party reproductions as "Unofficial". The
`benchmark/inferenceX/` subtree is a vendored snapshot, not a fork of the
upstream code; the upstream
<https://github.com/SemiAnalysisAI/InferenceX> remains canonical.

## Relationship to llm_perf

This dataset is consumed by the validation drivers under [`benchmark/validate/`](../validate/), which compare the analytical model with measured production-stack performance. The model spec → InferenceX model-identifier mapping lives in those drivers, not here — keeping the benchmark folder strictly upstream-derived.
