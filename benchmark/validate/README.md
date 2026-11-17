# Framework validators (vs InferenceX)

Cross-validate the llm_perf analytical model against measured production-stack performance from the vendored InferenceX dataset under `benchmark/inferenceX/`.

Each driver in this folder targets one `(model, hardware, framework)` cut. It loads the matching measured rows from `benchmark/inferenceX/data/flat/`, runs the framework on the same deployment shape, prints a per-(TP, B) table of time-per-output-token (TPOT) error, and writes a TPOT-vs-B plot with the framework's component breakdown (compute / mem / comm / LM / serving / total) overlaid against the InferenceX scatter.

The published set is the two per-cut drivers the root README walks through.

## Drivers

| Driver | Model spec | System | Framework | Cuts |
|---|---|---|---|---|
| `gpt_oss_120b_gb200_dynamo_trt.py` | `gpt_oss_120b` | `gb200.72gpu` | dynamo-trt | TP=4 EP=1 dec=4 — mean absolute error (MAE) 5.6% over the 7 points of the fitted cut (ISL 1024); 12.4% over the 8 points of the held-out cut (`--isl 8192`) |
| `dsr1_gb200_dynamo_trt.py` | `deepseek_r1_0528` | `gb200.72gpu` | dynamo-trt | EXACT (TP=36 EP=1), CO-LOCATED with DP-attention (TP=EP={8,16,32}), CO-LOCATED with TP-attention (TP=EP=8, four replicas, `--cut colo_tp_attn`) — 4.7% MAE over 14 points on the TP-attention cut; 13.6% over all 43 points; 15.1% on a cut held out of the fit |

Both are calibrated per setup: `bw_eta`, `c_seq`, the per-step host floor `c_orch` and the kernel-launch overlap `kernel_overlap_factor` are fitted per (model, hardware, framework) cell on that cell's InferenceX rows (least mean absolute error, 2026-11-16), while the per-kernel launch time and the additive host composition (`seq_overlap_factor = 0`) are shared by every cut on that serving stack. Constants fitted on one model do not carry over to another model on the same hardware and stack (mean errors of 16% to 44% when tried), so a fitted error is a statement about calibration quality; the error on a cut held out of the fit is the closer thing to a prediction test. The measured `conc` is the client concurrency of the whole endpoint, so `load_measured` divides it by `decode_num_workers` to get the per-replica batch the model works in. Everything else is first-principles. `bw_eta` multiplies on top of the `hbm_eta_beta` the system spec already carries (0.7 on GB200), so the shipped 0.69 and 0.60 mean 0.48× and 0.42× nameplate HBM bandwidth; it is a fitted residual, not a measured hardware constant. Accuracy on an *uncalibrated* cell is materially worse: the constants of another setup, or the unfitted values of a framework JSON, give errors of 20% and more on these same points.

## Running

```bash
# Each driver runs at its calibrated defaults:
python benchmark/validate/gpt_oss_120b_gb200_dynamo_trt.py
python benchmark/validate/dsr1_gb200_dynamo_trt.py

# Apply tuning knobs (uniform across drivers via add_common_cli):
python benchmark/validate/gpt_oss_120b_gb200_dynamo_trt.py --bw-eta 0.55 --c-seq-us 75

# Pick a single sub-cut:
python benchmark/validate/dsr1_gb200_dynamo_trt.py --cut colo_tp_attn

# CI smoke test mode — exit non-zero if MAE exceeds threshold:
python benchmark/validate/dsr1_gb200_dynamo_trt.py --check 35

# Override output directory (default: benchmark/results/):
python benchmark/validate/dsr1_gb200_dynamo_trt.py --out-dir /tmp/plots
```

## Common CLI args (registered by `common.add_common_cli`)

| Arg | Default | What it does |
|---|---|---|
| `--flops-eta` | per-driver | Multiply `device.peak_flops_TF` by this factor (sustained / nameplate) |
| `--bw-eta` | per-driver | Multiply every memory tier's `bandwidth_GBps` by this factor, on top of the tier's own `eta_beta` from the system spec |
| `--c-seq-us` | per-driver | Per-sequence serving runtime overhead, µs/seq (decode.md §7.2) |
| `--c-orch-us` | per-driver | Per-step host floor, µs/step, independent of batch |
| `--out-dir` | `benchmark/results/` | Where to write plots |
| `--check MAE_PCT` | None | If set, exit non-zero when overall MAE exceeds this |

The four calibration knobs (`--flops-eta`, `--bw-eta`, `--c-seq-us`, `--c-orch-us`) default to **per-driver** values — each script ships with `DEFAULT_BW_ETA` / `DEFAULT_C_SEQ_US` / `DEFAULT_C_ORCH_US` constants tuned to its specific (model, hardware, framework) cut so out-of-box runs land within reasonable MAE without users having to know the magic numbers. Override on the command line for sweeps; `--bw-eta 1.0 --c-seq-us 0 --c-orch-us 0` removes the fitted memory and host terms (the system spec's `hbm_eta_beta`, the 4 µs kernel launch and its overlap factor still apply, so this is not a bare roofline).

Each driver may also expose its own filters (`--cut`, `--tp`, etc.) — see `--help`.

## How the per-driver defaults are structured

The calibration knobs sit on different axes:

| Knob | Primary axis | Why |
|---|---|---|
| `c_seq` | **framework** (Dynamo+TRT, raw TRT, dynamo+sglang, vllm, …) | Host-side per-sequence work — block-table gather, sampling glue, scheduler — runs on the CPU between forward passes. The dominant variable is the serving stack (Python heaviness, graph-replay use, fused vs Python sampling), with weak HW dependence through CPU class and PCIe. |
| `bw_eta` | **HW × framework** | Sustained HBM BW depends on chip generation (HBM3 vs HBM3e) and on access pattern (dense GEMM vs paged-KV vs MoE expert hopping). Different frameworks are kinder or harder on the controllers. |
| `c_orch` | **cut** (model × framework) | Per-step host floor that does not depend on batch, fitted alongside `bw_eta`. Without it, and with host work assumed to hide fully behind the GPU step, `bw_eta` absorbs the host cost and its best-fit value falls as TP grows. |
| `flops_eta` | HW (sparse vs dense ratios), model | Less commonly needed — most miscalibration shows up on `bw_eta`. |

Empirical anchors observed across the validator suite:

| Framework | Typical c_seq | Notes |
|---|---|---|
| `dynamo-trt` | about 5 µs/seq | Aggressive C++ with graph replay. The two published drivers fit 5.5 (gpt-oss-120b) and 4.9 (DeepSeek-R1). |
| `dynamo-sglang` | not measured | Expected above `dynamo-trt` from its Python share; the framework JSON ships 5 µs/seq as a starting point. |
| `trt`, `trt-llm` (raw) | 50–100 µs/seq | No Dynamo orchestrator; more individual kernel launches per step. |
| `sglang`, `vllm` (raw) | 30–60 µs/seq (estimated) | Python interpreter dominates. |

`bw_eta` ranges by chip (the driver multiplier; multiply by the system spec's `hbm_eta_beta` to get the fraction of nameplate):

| Chip | `bw_eta` range observed |
|---|---|
| HBM3e on Blackwell (B200/GB200) | 0.60–1.07 once the host floor is fitted separately |
| HBM3e on Blackwell Ultra (B300/GB300) | similar; less data |
| HBM3 on Hopper (H100/H200) | 0.55–1.05; the low end is from fits that had no separate host floor |
| Dense models on raw TRT | as low as 0.4 |

### Re-tuning the defaults

Re-tune after `fetch.py` pulls new InferenceX rows: grid-sweep `(bw_eta, c_seq)` for the cut, copy the best fit into the driver's `DEFAULT_*` constants, and update the docstring comment with the resulting MAE.

## Adding a new validator

1. Confirm the model is in the InferenceX dataset: `python benchmark/inferenceX/fetch.py --discover` (run `fetch.py` again if anything new shows up to refresh the local snapshot).

2. Confirm there's a model spec in `llm_perf/database/model/<id>.json` that matches the InferenceX `model` identifier — close-enough architectures are fine if you override `bytes_per_param` to match the measured precision.

3. Confirm there's a system spec in `llm_perf/database/system/<id>.json` that matches the InferenceX `hardware`. Add one if not — e.g. `b200.8gpu` for single-server B200.

4. Copy one of the existing drivers as a template, replace the constants at the top (`MODEL`, `SYSTEM`, `PRECISION`, `ISL`, `OSL`, the partition shape), and adjust the `load_measured(...)` filters to land on your specific cut.

5. Smoke-test: `python benchmark/validate/<your_driver>.py --check 50`

## What goes in `common.py`

Anything used by ≥2 drivers. Currently:

- `load_measured()` — InferenceX CSV loader with a uniform filter API
- `system_with_eta()` — derate wrapper (flops / BW / num_devices override)
- `run_framework()` — sweep B, return per-B latency breakdown
- `predict_at()` — single-B framework prediction (used to align with measured points)
- `error_table()` — per-row + per-label MAE summary
- `plot_tpot_vs_B()` — single shared plot style
- `add_common_cli()` / `eta_subtitle()` / `eta_filename_tag()` — uniform CLI

If you find yourself copy-pasting more than ~20 lines between drivers, hoist into `common.py`.

## Output

Plots land in `benchmark/results/` (ignored by git). Filenames encode the cut and any tuning knobs, e.g.

```
benchmark/results/
  dsr1_dynamo_trt_exact_tp36_ep1_dec36.png
  dsr1_dynamo_trt_colocated_tp8ep8.png
  gpt_oss_120b_dynamo_trt_tp4_ep1_dec4_flops1p00_bw0p69_serv6us.png
```
