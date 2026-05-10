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
| `dsr1_gb200_dynamo_trt.py` | `deepseek_r1_0528` | `gb200.72gpu` | dynamo-trt | EXACT (TP=36 EP=1), CO-LOCATED (TP=EP={8,16,32}), ORTHO (TP=8 EP=8 dec=32) |
| `dsr1_gb300_dynamo_trt.py` | `deepseek_r1_0528` | `gb300.72gpu` | dynamo-trt | ORTHO (TP=8 EP=8 dec=32), CO-LOCATED (TP=EP={8,16,32}) |
| `dsr1_gb300_dynamo_sglang.py` | `deepseek_r1_0528` | `gb300.72gpu` | dynamo-sglang | EXACT (TP=4 EP=1), CO-LOCATED (TP=EP={8,32,48}) |
| `dsr1_b200_trt.py` | `deepseek_r1_0528` | `b200.8gpu` | trt | TP=8 EP=1 dec=8 |
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
| `--flops-eta` | per-driver | Multiply `device.peak_flops_TF` by this factor (sustained / nameplate) |
| `--bw-eta` | per-driver | Multiply every memory tier's `bandwidth_GBps` by this factor |
| `--c-serving-us` | per-driver | Per-sequence serving runtime overhead, µs/seq (decode.md §7.2) |
| `--out-dir` | `benchmark/results/` | Where to write plots |
| `--check MAE_PCT` | None | If set, exit non-zero when overall MAE exceeds this |

The three calibration knobs (`--flops-eta`, `--bw-eta`, `--c-serving-us`)
default to **per-driver** values — each script ships with `DEFAULT_BW_ETA`
/ `DEFAULT_C_SERVING_US` constants tuned to its specific (model, hardware,
framework) cut so out-of-box runs land within reasonable MAE without users
having to know the magic numbers. Override on the command line for sweeps;
`--bw-eta 1.0 --c-serving-us 0` reverts to the peak roofline.

Each driver may also expose its own filters (`--cut`, `--tp`, etc.) — see
`--help`.

## How the per-driver defaults are structured

The two calibration knobs sit on different axes:

| Knob | Primary axis | Why |
|---|---|---|
| `c_serving` | **framework** (Dynamo+TRT, raw TRT, dynamo+sglang, vllm, …) | Host-side per-sequence work — block-table gather, sampling glue, scheduler — runs on the CPU between forward passes. The dominant variable is the serving stack (Python heaviness, CUDA-Graph use, fused vs Python sampling), with weak HW dependence through CPU class and PCIe. |
| `bw_eta` | **HW × framework** | Sustained HBM BW depends on chip generation (HBM3 vs HBM3e) and on access pattern (dense GEMM vs paged-KV vs MoE expert hopping). Different frameworks are kinder or harder on the controllers. |
| `flops_eta` | HW (sparse vs dense ratios), model | Less commonly needed — most miscalibration shows up on `bw_eta`. |

Empirical anchors observed across the validator suite:

| Framework | Typical c_serving | Notes |
|---|---|---|
| `dynamo-trt` | 5–22 µs/seq | Aggressive C++/CUDA-Graph. The 22 µs anchor in `decode.md §7.2` matches gpt-oss/Dynamo+TRT exactly. |
| `dynamo-sglang` | 25–50 µs/seq | Dynamo wrapper over Python-heavier SGLang internals. |
| `trt`, `trt-llm` (raw) | 50–100 µs/seq | No Dynamo orchestrator; more individual kernel launches per step. |
| `sglang`, `vllm` (raw) | 30–60 µs/seq (estimated) | Python interpreter dominates. (Not yet validated; add a driver to confirm.) |

`bw_eta` ranges by chip:

| Chip | `bw_eta` range observed |
|---|---|
| HBM3e on Blackwell (B200/GB200) | 0.7–1.0 (better with Dynamo wrapping) |
| HBM3e on Blackwell Ultra (B300/GB300) | similar; less data |
| HBM3 on Hopper (H100/H200) | 0.55–0.7 |
| Dense models on raw TRT | as low as 0.4 (`llama3_70b_b200_trt`) |

### Re-tuning the defaults

`_calibrate.py` runs a small grid sweep across all drivers and prints the
best-fit `(bw_eta, c_serving)` per cut. Re-run after `fetch.py` pulls new
InferenceX rows; copy the values into the per-driver `DEFAULT_*` constants
and update the docstring comment with the resulting MAE.

```bash
python benchmark/validate/_calibrate.py
```

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
