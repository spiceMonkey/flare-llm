# InferenceX BenchmarkRow schema

Each row in `data/raw/<Model>.json` (or one CSV row in `data/flat/<Model>.csv`
after flattening) is a single benchmark measurement. The fields below are
the union observed across the current snapshot.

## Configuration fields (deployment shape)

| Field | Type | Meaning |
|---|---|---|
| `model` | string | Model identifier (matches `MODELS` in `fetch.py`) |
| `hardware` | string | GPU / accelerator class — `b200`, `gb200`, `h200`, `mi355x`, etc. |
| `framework` | string | Serving stack — `trt`, `dynamo-trt`, `sglang`, `vllm`, etc. |
| `precision` | string | Weight precision — `fp4`, `fp8`, `bf16`, `int8`, etc. |
| `isl` | int | Input sequence length (prefill tokens per request) |
| `osl` | int | Output sequence length (decoded tokens per request) |
| `conc` | int | Concurrency — number of in-flight requests at steady state |
| `disagg` | bool | Disaggregated prefill / decode (separate clusters) vs co-located |
| `is_multinode` | bool | Spans multiple physical nodes |
| `spec_method` | string | `none`, `mtp`, `eagle`, `medusa` (speculative decoding) |
| `date` | string | Run date (ISO `YYYY-MM-DD`) |

## Partition fields

Prefill and decode side each have their own partition shape; equal when
not disaggregated.

| Field | Meaning |
|---|---|
| `prefill_tp`, `decode_tp` | Tensor-parallel degree |
| `prefill_ep`, `decode_ep` | Expert-parallel degree |
| `prefill_dp_attention`, `decode_dp_attention` | DP-attention enabled (boolean) |
| `prefill_num_workers`, `decode_num_workers` | Worker / replica count |
| `num_prefill_gpu`, `num_decode_gpu` | Total GPU count on each side |

## Metric fields (nested under `metrics` in raw, hoisted in flat)

For each of `tpot`, `ttft`, `e2el`, `itl`, and `intvty` (interactivity in
tok/s/user) the API reports four quantiles — `mean_*`, `median_*`,
`p99_*`, `std_*`. Throughput is `tput_per_gpu`, plus the prefill/decode
breakdowns `input_tput_per_gpu` / `output_tput_per_gpu` (tok/s/GPU).

| Quantile | Field naming | Notes |
|---|---|---|
| Median | `median_<metric>` | Most stable single-row summary |
| Mean | `mean_<metric>` | Sensitive to long tails |
| P99 | `p99_<metric>` | Tail behavior |
| Std | `std_<metric>` | Run-to-run variance |

| Metric | Unit | Meaning |
|---|---|---|
| `*_tpot` | seconds | Time per output token (decode step time, user-observed) |
| `*_ttft` | seconds | Time to first token (prefill + queue + first decode step) |
| `*_e2el` | seconds | End-to-end latency for the full request |
| `*_itl` | seconds | Inter-token latency (similar to TPOT, slightly different aggregation) |
| `*_intvty` | tok/s/user | Interactivity = `1 / TPOT` per request |
| `tput_per_gpu` | tok/s/GPU | Aggregate output rate divided by `num_decode_gpu` |
| `input_tput_per_gpu` | tok/s/GPU | Prefill-side throughput |
| `output_tput_per_gpu` | tok/s/GPU | Decode-side throughput |

## Provenance

| Field | Meaning |
|---|---|
| `image` | Container image identifier (often null) |
| `run_url` | Link to the GitHub Actions run that produced this row |

## Quick gotchas observed during the comparison work

- `metrics` is sometimes `null` in the raw schema for runs that didn't
  complete; the flat CSV omits those rows. Filter on `median_tpot is not
  None` (raw) or non-empty `median_tpot` column (flat).
- `decode_dp_attention` is the **flag whether DP-attention was enabled**,
  not the DP-attention group size. Pair with `decode_tp`/`decode_ep` and
  `num_decode_gpu` to infer the layout shape.
- `conc` is the steady-state in-flight request count, equal to the
  per-decode-step batch size `B` for a single decode replica under
  continuous batching. Disaggregated runs split work across the prefill
  and decode clusters; `conc` here still refers to the decode-side
  in-flight count.
- `precision` describes the **weight precision**; activations are usually
  bf16 or fp16 regardless. Set `bytes_per_param` in the model spec to
  match the listed precision.
- For models in the snapshot that have `disagg=True` rows with
  `decode_dp_attention=True` and `decode_tp == decode_ep == num_decode_gpu`,
  the deployment is the co-located TP+EP shape natively modeled by
  `FrameworkSpec(tp_ep_layout="co_located", attention_mode="dp")` — see
  `notation.md §1` and `decode.md §6.3`.
