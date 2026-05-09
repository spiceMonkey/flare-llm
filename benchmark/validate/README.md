# Framework validators (vs InferenceX)

Cross-validate the llm_perf analytical model against measured production-stack
performance from the vendored InferenceX dataset under
`benchmark/inferenceX/`.

Each driver in this folder targets one `(model, hardware, framework)` cut.
It loads the matching measured rows from `benchmark/inferenceX/data/flat/`,
runs the framework on the same deployment shape, prints a per-(TP, B) error
table, and writes a TPOT-vs-B plot with the framework's component breakdown
(compute / mem / comm / LM / serving / total) overlaid against the InferenceX
scatter.

## Drivers

| Driver | Model spec | System | Framework | Cuts |
|---|---|---|---|---|
| `dsr1_gb200_dynamo_trt.py` | `deepseek_r1_0528` | `gb200.72gpu` | dynamo-trt | EXACT (TP=36 EP=1), CO-LOCATED (TP=EP={8,16,32}) |
| `gpt_oss_120b_gb200_dynamo_trt.py` | `gpt_oss_120b` | `gb200.72gpu` | dynamo-trt | TP=4 EP=1 dec=4 |
| `llama3_70b_b200_trt.py` | `llama3.1_70b` (override FP8) | `b200.8gpu` | trt | TP ∈ {1, 2, 4, 8} |

## Running

```bash
# Each driver runs at default knobs (no derate, no serving overhead):
python benchmark/validate/dsr1_gb200_dynamo_trt.py
python benchmark/validate/gpt_oss_120b_gb200_dynamo_trt.py
python benchmark/validate/llama3_70b_b200_trt.py

# Apply tuning knobs (uniform across drivers via add_common_cli):
python benchmark/validate/llama3_70b_b200_trt.py --bw-eta 0.55 --c-serving-us 75

# Pick a single sub-cut:
python benchmark/validate/dsr1_gb200_dynamo_trt.py --cut colocated
python benchmark/validate/llama3_70b_b200_trt.py --tp 8

# CI smoke test mode — exit non-zero if MAE exceeds threshold:
python benchmark/validate/dsr1_gb200_dynamo_trt.py --check 35

# Override output directory (default: benchmark/results/):
python benchmark/validate/llama3_70b_b200_trt.py --out-dir /tmp/plots
```

## Common CLI args (registered by `common.add_common_cli`)

| Arg | Default | What it does |
|---|---|---|
| `--flops-eta` | 1.0 | Multiply `device.peak_flops_TF` by this factor (sustained / nameplate) |
| `--bw-eta` | 1.0 | Multiply every memory tier's `bandwidth_GBps` by this factor |
| `--c-serving-us` | 0.0 | Per-sequence serving runtime overhead, µs/seq (decode.md §7.2) |
| `--out-dir` | `benchmark/results/` | Where to write plots |
| `--check MAE_PCT` | None | If set, exit non-zero when overall MAE exceeds this |

Each driver may also expose its own filters (`--cut`, `--tp`, etc.) — see
`--help`.

## Adding a new validator

1. Confirm the model is in the InferenceX dataset:
   `python benchmark/inferenceX/fetch.py --discover`
   (run `fetch.py` again if anything new shows up to refresh the local snapshot).

2. Confirm there's a model spec in `llm_perf/database/model/<id>.json` that
   matches the InferenceX `model` identifier — close-enough architectures
   are fine if you override `bytes_per_param` to match the measured precision
   (Llama-3.3-70B uses the Llama-3.1 spec + FP8 override).

3. Confirm there's a system spec in `llm_perf/database/system/<id>.json` that
   matches the InferenceX `hardware`. Add one if not — e.g. `b200.8gpu` for
   single-server B200.

4. Copy one of the existing drivers as a template, replace the constants at
   the top (`MODEL`, `SYSTEM`, `PRECISION`, `ISL`, `OSL`, the partition shape),
   and adjust the `load_measured(...)` filters to land on your specific cut.

5. Smoke-test:
   `python benchmark/validate/<your_driver>.py --check 50`

## What goes in `common.py`

Anything used by ≥2 drivers. Currently:

- `load_measured()` — InferenceX CSV loader with a uniform filter API
- `system_with_eta()` — derate wrapper (flops / BW / num_devices override)
- `run_framework()` — sweep B, return per-B latency breakdown
- `predict_at()` — single-B framework prediction (used to align with measured points)
- `error_table()` — per-row + per-label MAE summary
- `plot_tpot_vs_B()` — single shared plot style
- `add_common_cli()` / `eta_subtitle()` / `eta_filename_tag()` — uniform CLI

If you find yourself copy-pasting more than ~20 lines between drivers, hoist
into `common.py`.

## Output

Plots land in `benchmark/results/` (gitignored). Filenames encode the cut and
any tuning knobs, e.g.

```
benchmark/results/
  dsr1_dynamo_trt_exact_tp36_ep1_dec36.png
  dsr1_dynamo_trt_colocated_tp8ep8.png
  llama3_70b_b200_trt_tp8_flops1p00_bw0p55_serv75us.png
```
