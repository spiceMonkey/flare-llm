# llm_perf

`llm_perf` is a lightweight toolkit for large language model (LLM) inference performance modeling. It ships a structured JSON database for models/systems/partitions/tuners, analytical cost models, and a tutorial notebook that walks through the entire workflow.

## Features

- **Typed specs** for models, hardware systems, partition plans, and tuning knobs
- **Core analytical models** (memory, FLOPs, traffic, communication, latency)
- **InferenceCalculator** that runs the full stack and returns structured results
- **HuggingFace adapter** that converts HF `config.json` files into the `llm_perf` schema
- **Quickstart notebook** (`quickstart.ipynb`) demonstrating database discovery, optional HF conversion, and diagnostic output
- **Ready-to-run scripts** for HuggingFace config conversion and partition sweeps (with separate latency/memory vs. comm/traffic tables plus Matplotlib scatter plots)

## Repository Layout

```
.
├── quickstart.ipynb
├── documentation/
│   ├── codebase.structure.md
│   ├── equations.cheetsheet.{md,pdf}
│   └── modeling.methodology.{md,pdf}
├── scripts/
│   ├── convert_hf_model.py
│   └── partition_sweep.py
└── llm_perf/
    ├── calculators/
    │   └── inference_calculator.py
    ├── core/
    │   ├── comm_model.py
    │   ├── flops_model.py
    │   ├── latency_model.py
    │   ├── memory_model.py
    │   └── traffic_model.py
    ├── database/
    │   ├── model/
    │   ├── system/
    │   ├── partition/
    │   └── tuner/
    ├── io/
    │   ├── database_loaders.py
    │   ├── model_loaders.py
    │   ├── partition_loaders.py
    │   ├── system_loaders.py
    │   └── tuner_loaders.py
    ├── specs/
    │   ├── model_spec.py
    │   ├── partition_spec.py
    │   ├── system_spec.py
    │   └── tuner_spec.py
    └── utils/
        ├── constants.py
        ├── data_check.py
        ├── equations.py
        ├── hf_model_adapter.py
        └── plotting.py
```

## Quickstart

1. **Install dependencies** (Python 3.10+ recommended). Create a virtual environment and install Jupyter plus any optional plotting stacks you need.
2. **Open `quickstart.ipynb`** and run the cells sequentially. The notebook demonstrates:
   - Listing available IDs in the on-disk database
   - Loading specs via `llm_perf.io`
   - Converting a HuggingFace config using `convert_hf_config_to_model_json`
   - Inspecting partitions with the nested DP→PP→EP→TP→SP hierarchy
   - Running `InferenceCalculator` to obtain memory/FLOPs/traffic/communication/latency diagnostics
3. **Customize specs** by editing JSON files under `llm_perf/database/**` or generating new ones via the HF adapter.

## Utility Scripts

Two small helpers live under `scripts/` for repeatable experiments outside the notebook:

1. **`convert_hf_model.py`** — converts a HuggingFace `config.json`, writes the adapted JSON into `llm_perf/database/model/`, reloads it via `load_model_spec`, and prints a summary that now includes estimated total parameters and byte footprint.

    ```powershell
    .\.llm_perf\Scripts\python.exe scripts\convert_hf_model.py
    ```

    Adjust `HF_FILE_NAME`/paths inside the script as needed. Because it bootstraps `sys.path`, you can run it from the repo root without extra setup.

2. **`partition_sweep.py`** — iterates every `(PP, TP, EP, SP)` factorization for a given cluster budget (DP fixed, default 1), runs `InferenceCalculator` per config, prints *two* aligned tables (first: TPS/TPS plus latency + memory footprint, second: TPS/TPS plus collective message sizes and parameter/activation/KV traffic), and saves a scatter plot via `llm_perf.utils.save_config_tps_scatter`.

    ```powershell
    .\.llm_perf\Scripts\python.exe scripts\partition_sweep.py --cluster-size 32 --system llm_perf/database/system/h100.32gpu.json
    ```

    The plot filename automatically incorporates the hardware system name (e.g., `artifacts/partition_sweep_h100_32gpu_cluster.png`). Install Matplotlib in your environment to enable plotting (already bundled in `.llm_perf`).

## Programmatic Usage

```python
from llm_perf import InferenceCalculator
from llm_perf.io import (
    load_model_from_db,
    load_system_from_db,
    load_partition_from_db,
    load_tuner_from_db,
)

model = load_model_from_db("example.model.dense")
system = load_system_from_db("example.sys")
partition = load_partition_from_db("example.partition")
tuner = load_tuner_from_db("example.tuner")

calc = InferenceCalculator(model, system, partition, tuner)
results = calc.run()
print(results.latency.TTPS)
```

## HuggingFace Adapter

To bring an HF `config.json` into the local database:

```python
from pathlib import Path
from llm_perf.utils import convert_hf_config_to_model_json

hf_config = Path("llm_perf/database/model/external.model/hf/qwen3_vl_235b_a22b_thinking_fp8.json")
out_path = Path("llm_perf/database/model/qwen3_vl_235b_fp8.json")

convert_hf_config_to_model_json(
    hf_config_path=hf_config,
    out_path=out_path,
    name_override="qwen3_vl_235b_fp8",
    overwrite=True,
)
```

The new JSON will appear alongside other model cards and can be loaded with `load_model_from_db` immediately.

## Contributing

- Open issues or PRs for new spec types, adapters, or analytical improvements.
- Keep JSON schemas backward compatible when possible.
- Run the quickstart notebook after large changes to ensure the workflow still succeeds.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
