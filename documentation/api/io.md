# API Reference — IO

**Author:** Yue Lu  
**Date:** August 2026  

Spec loaders. `load_*_from_db` reads the bundled JSON database by id; `load_*_spec` / `*_spec_from_json_dict` read an arbitrary path or dict.

---

## `llm_perf.io.model_loaders`

### `model_spec_from_json_dict(cfg: Dict[str, Any]) -> LlmModelSpec`

Build LlmModelSpec from a config dict.

Expected format:

    {
      "schema": "llm_perf.model",
      "name": "...",
      "L": 32,
      "H": 4096,
      "n_q": 32,
      "n_kv": 8,
      "I_dense": 14336,
      "vocab_size": 128256,
      "max_seq_len": 8192,
      "bytes_per_param": 2,
      "moe": { ... } | null
    }

If 'schema' is missing, we still try to parse using the same keys.

### `load_model_spec(path: str | Path) -> LlmModelSpec`

Load LlmModelSpec from a JSON file.

Example:
    model = load_model_spec("model.json")

---

## `llm_perf.io.system_loaders`

### `system_spec_from_json_dict(cfg: Dict[str, Any]) -> SystemSpec`

Build SystemSpec from a config dict.

Expected format:

    {
      "schema": "llm_perf.system",
      "name": "...",
      "num_devices": 64,
      "device": {
        "name": "...",
        "hbm_capacity_GB": 80.0,
        "hbm_bandwidth_GBps": 3350.0,
        "peak_flops_TF": 1000.0
      },
      "fabrics": {
        "nvlink5": {
          "tiers": [
            {"name": "intra-rack-nvswitch", "ports": 72,
             "bw_per_port_GBps": 900.0, "alpha_us": 0.5}
          ]
        },
        "ib": {
          "tiers": [
            {"name": "inter-rack-quantum-ib", "ports": 8,
             "bw_per_port_GBps": 400.0, "alpha_us": 2.5}
          ]
        }
      },
      "collective_fabrics": {
        "TP": ["nvlink5", "ib"],
        "EP": "nvlink5",
        "SP": "nvlink5",
        "PP": ["nvlink5", "ib"]
      }
    }

A collective's value may be a single string (single-fabric shorthand) or an ordered list of fabric names (escalation chain, innermost first).

### `load_system_spec(path: str | Path) -> SystemSpec`

Load SystemSpec from a JSON file.

---

## `llm_perf.io.partition_loaders`

### `partition_spec_from_json_dict(cfg: Dict[str, Any]) -> PartitionSpec`

Build PartitionSpec from a config dict.

partition.json format:

    {
      "schema": "llm_perf.partition",
      "PP": 4,
      "TP": 4,
      "EP": 8,
      "SP": 2
    }

Phase H: `attention_mode` and `tp_ep_layout` (formerly `layout`) are no longer accepted here — they live on FrameworkSpec. Files that still carry them will raise ValueError so that stale configs are caught explicitly rather than silently dropping the field.

### `load_partition_spec(path: str | Path) -> PartitionSpec`

---

## `llm_perf.io.tuner_loaders`

### `tuning_spec_from_json_dict(cfg: Dict[str, Any]) -> TuningSpec`

Build TuningSpec from a config dict.

tuner.json format:

    {
      "schema": "llm_perf.tuner",

      "S_decode": 4096,

      "tp_algorithm": "ring",
      "ep_algorithm": "tree"

      "n_TP_collectives": 2,
      "n_EP_collectives": 2,
      "n_SP_collectives": 1,

    }

### `load_tuning_spec(path: str | Path) -> TuningSpec`

---

## `llm_perf.io.framework_loaders`

FrameworkSpec JSON loaders.

Companion to `framework_spec.FrameworkSpec`. Mirrors the pattern of `tuner_loaders.py` / `model_loaders.py`: a `framework_spec_from_json_dict` that builds a FrameworkSpec from a parsed JSON dict (validating the mode-string fields against their whitelists), plus a `load_framework_spec` that reads a file path. Database-stem lookup (`load_framework_from_db`) lives in `database_loaders.py` alongside the other spec families.

### `framework_spec_from_json_dict(cfg: Dict[str, Any]) -> FrameworkSpec`

Build FrameworkSpec from a config dict.

Expected format:

    {
      "schema": "llm_perf.framework",
      "name": "dynamo-trt",

      "c_orch_us": 0.0,
      "c_seq_us": 0.0,
      "kernel_launch_us": 7.0,
      "kernels_per_layer_compute": 10,
      "kernels_per_collective_call": 2,
      "kernels_per_pp_hop": 2,
      "kernel_overlap_factor": 1.0,

      "moe_a2a_pattern": "scatter",
      "mla_mode": "absorbed",
      "inc_enabled": true,

      "tp_algorithm_decode": "auto",
      "tp_algorithm_prefill": "auto",
      "ep_algorithm_decode": "auto",
      "ep_algorithm_prefill": "auto",
      "torus_algorithm": "auto",
      "torus_align_policy": "prefix",
      "n_TP_collectives": 2,
      "n_EP_collectives": 2,
      "n_SP_collectives": 1,
      "comm_overlap_factor": 0.0
    }

All fields except `name` fall through to FrameworkSpec dataclass defaults when absent. Algorithm fields accept "auto" to trigger cost-model resolution via `optimize_collective_algorithms`.

### `load_framework_spec(path: str | Path) -> FrameworkSpec`

Load FrameworkSpec from a JSON file.

---

## `llm_perf.io.overhead_loaders`

### `overhead_spec_from_json_dict(cfg: Dict[str, Any]) -> OverheadSpec`

Build OverheadSpec from a config dict.

JSON schema: llm\_perf.overhead

### `load_overhead_spec(path: str | Path) -> OverheadSpec`

---

## `llm_perf.io.disagg_loaders`

### `disagg_spec_from_json_dict(cfg: Dict[str, Any]) -> DisaggSpec`

Build DisaggSpec from a config dict.

JSON schema: llm\_perf.disagg

### `load_disagg_spec(path: str | Path) -> DisaggSpec`

---

## `llm_perf.io.database_loaders`

### `list_hw_system_ids() -> List[str]`

List available hardware system IDs from llm\_perf/database/system.

Returns filename stems, e.g. ["h100\_node", "h100\_cluster\_64"].

### `load_system_from_db(system_id: str)`

Load a SystemSpec from llm\_perf/database/system/{system\_id}.json using the standard system loader.

### `list_model_ids() -> List[str]`

List available LLM model IDs from llm\_perf/database/model.

### `load_model_from_db(model_id: str)`

Load a LlmModelSpec from llm\_perf/database/model/{model\_id}.json using the standard model loader.

### `list_tuner_ids() -> List[str]`

List available tuner IDs from llm\_perf/database/tuner.

### `load_tuner_from_db(tuner_id: str)`

Load a TuningSpec from llm\_perf/database/tuner/{tuner\_id}.json.

### `list_framework_ids() -> List[str]`

List available framework IDs from llm\_perf/database/framework.

### `load_framework_from_db(framework_id: str)`

Load a FrameworkSpec from llm\_perf/database/framework/{framework\_id}.json.

---
