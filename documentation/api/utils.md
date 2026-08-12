# API Reference — Utils

**Author:** Yue Lu  
**Date:** August 2026  

Helpers: HuggingFace model adapter, DRAM bandwidth calculator, constants, validation, and plotting.

---

## `llm_perf.utils.hf_model_adapter`

HuggingFace config -> llm\_perf model JSON adapter.

This module converts a HuggingFace-style config.json into the canonical llm\_perf.model.v1 JSON format used by llm\_perf.io.model\_loaders.

Typical usage:

    from pathlib import Path
    from llm_perf.utils import convert_hf_config_to_model_json
    from llm_perf.io import load_model_spec

    hf_config = Path("external/Qwen/Qwen3-VL-235B/config.json")
    out_json  = Path("llm_perf/database/llm.model/qwen3_vl_235b.json")

    convert_hf_config_to_model_json(
        hf_config_path=hf_config,
        out_path=out_json,
        name_override="Qwen3-VL-235B",
        bytes_per_param_override=None,  # let adapter infer (e.g. FP8)
        L_override=None,               # or set manually if needed
        overwrite=True,
    )

    spec = load_model_spec(out_json)

The adapter is designed to handle:
* Plain LLaMA/Qwen-style LLM configs (flat)
* Qwen3-VL-style multimodal configs where the text LLM lives under
    `text_config`
* FP8 / quantization hints in `quantization_config`
* MoE hints (num\_experts, num\_experts\_per\_tok, moe\_intermediate\_size)

### `hf_config_to_llm_perf_model_dict(cfg: Dict[str, Any], name_override: Optional[str]=None, bytes_per_param_override: Optional[float]=None, L_override: Optional[int]=None) -> Dict[str, Any]`

Convert a HuggingFace config dict into the canonical llm\_perf.model.v1 JSON dict.

Output dict shape:

    {
      "schema": "llm_perf.model",
      "name": "...",
      "L": ...,
      "H": ...,
      "n_q": ...,
      "n_kv": ...,
      "I_dense": ...,
      "vocab_size": ...,
      "max_seq_len": ...,
      "bytes_per_param": ...,
      "moe": {...} or null
    }

### `convert_hf_config_to_model_json(hf_config_path: str | Path, out_path: str | Path, name_override: Optional[str]=None, bytes_per_param_override: Optional[float]=None, L_override: Optional[int]=None, overwrite: bool=False) -> Path`

Load a HuggingFace config.json, convert it to llm\_perf.model.v1 JSON, and write it.

Args:
    hf_config_path:
        Path to the HuggingFace config.json.
    out_path:
        Path where the llm_perf model JSON should be written
        (e.g. llm_perf/database/llm.model/qwen3_vl_235b.json).
    name_override:
        If not None, override the 'name' field.
    bytes_per_param_override:
        If not None, override bytes_per_param (e.g., 2.0, 4.0, 1.0).
    L_override:
        If not None, override the number of layers L.
    overwrite:
        If False and out_path exists, raises FileExistsError.

Returns:
    Path to the written JSON file.

---

## `llm_perf.utils.dram3d`

DRAM3D bandwidth utility (documentation/modeling/dram3d.md).

Derives HBM bandwidth from physical interface parameters and can update existing database/system/*.json files with the computed values.

### `compute_hbm_bandwidth(die_area_mm2: float, pitch_um: float, data_pin_fraction: float, data_rate_gbps: float, n_dies: int=8) -> dict`

Derive HBM bandwidth from physical parameters.

Args:
    die_area_mm2: DRAM die area in mm².
    pitch_um: Hybrid bonding / µbump pitch in µm.
    data_pin_fraction: Fraction of total pins used for data (eta_data, typically 0.3-0.5).
    data_rate_gbps: Data rate per pin in Gbps.
    n_dies: Number of stacked DRAM dies.

Returns:
    Dict with derived quantities:
      - n_pins_total: total pad count on die
      - n_pins_data: data pad count
      - bw_per_die_GBps: bandwidth per die interface (GB/s)
      - bw_total_GBps: total bandwidth (n_dies × bw_per_die) (GB/s)

### `update_system_json(system_json_path: str, hbm_bandwidth_GBps: float, hbm_capacity_GB: Optional[float]=None, output_path: Optional[str]=None) -> None`

Read a database/system/*.json and update the device memory parameters.

Args:
    system_json_path: Path to the system JSON file.
    hbm_bandwidth_GBps: New HBM bandwidth value (GB/s).
    hbm_capacity_GB: New HBM capacity (GB), or None to leave unchanged.
    output_path: Output path. If None, overwrites the input file.

---

## `llm_perf.utils.partition_enum`

Partition enumeration helper for sweep notebooks.

Centralises the practical (PP, TP, EP, SP) constraints used across the `notebooks/pareto_*` and `notebooks/ttft_*` sweeps so callers do not re-derive them inline.

Constraints applied (all must hold):
- pp * tp * ep * sp <= num\_devices
- pp <= pp\_max  (default 32; the typical production cap where deeper
    PP starts running into bubble / TTFT / microbatch-floor headwinds)
- tp <= n\_kv  (each TP rank must hold ≥ 1 KV head)
- tp <= n\_experts and ep <= n\_experts  (MoE only — sharding cannot
    exceed expert count)
- tp * ep <= scale\_up\_domain  (avoid crossing the rack-local NVLink /
    NVL boundary for collective traffic)

For dense models (no MoE), `ep_max = 1` and the n\_experts caps on TP/EP are inert — the n\_kv cap on TP still applies.

### `scale_up_domain_size(system: SystemSpec, role: str='TP', *, scale_up_tier_index: int=0) -> int`

Cumulative number of devices reachable through tier `scale_up_tier_index` (inclusive).

`scale_up_tier_index = 0` returns the ports of tier 0 alone — the innermost domain (e.g., 72 for NVL72; 16 for the d-Matrix pair-of-cards mesh). `scale_up_tier_index = i` returns the cumulative reach `prod(tier[k].ports for k in 0..i)` — useful when a deployment treats a multi-tier fabric as a single scale-up unit (e.g., d-Matrix tier-1 PCIe extends the domain to a full server = 16 × 4 = 64 chiplets).

Out-of-range indices clamp to the last available tier rather than raising — so a single-tier system like NVL72 with `scale_up_tier_index=1` still returns its tier-0 reach (72) instead of erroring.

### `enumerate_partitions(model: LlmModelSpec, system: SystemSpec, *, num_devices: Optional[int]=None, pp_max: int=32, pp_choices: Optional[List[int]]=None, sp_choices: Optional[List[int]]=None, tp_max_override: Optional[int]=None, ep_max_override: Optional[int]=None, scale_up_domain_override: Optional[int]=None, scale_up_tier_index: int=0) -> List[PartitionSpec]`

Enumerate valid (PP, TP, EP, SP) partitions for `model` on `system`.

See module docstring for the full list of constraints.

Parameters
----------
num\_devices : int, optional
    Total devices available. Defaults to `system.num_devices`. Override
    when sweeping cluster size at fixed system spec.
pp\_max : int
    Maximum PP. Default 32 (typical production cap — deeper PP starts
    running into bubble / TTFT / microbatch-floor headwinds even with
    inflight batching, but 32 leaves room to explore the cliff).
pp\_choices, sp\_choices : list of int, optional
    Override the enumeration ladders. Defaults are the standard ladders
    used across the notebook suite.
tp\_max\_override, ep\_max\_override : int, optional
    Override the auto-derived TP / EP caps. Useful for sensitivity
    sweeps that intentionally cross the natural cap.
scale\_up\_domain\_override : int, optional
    Override the auto-derived scale-up domain size with an absolute
    value. Takes precedence over `scale_up_tier_index`. Useful when
    the sweep replaces the system fabric to test a hypothetical
    topology.
scale\_up\_tier\_index : int, default 0
    Index into the fabric chain (`system.get_tier_chain(role)`) up to
    which the scale-up domain extends. 0 = innermost tier only
    (default — every collective stays within tier 0 of the fabric);
    1 = cumulative reach through tier 1 (e.g., d-Matrix server-wide
    via PCIe = 16 × 4 = 64); etc. Indices beyond the chain length
    clamp to the last tier. Inert when `scale_up_domain_override` is
    also set.

### `describe_constraints(model: LlmModelSpec, system: SystemSpec, pp_max: int=32, *, scale_up_tier_index: int=0) -> str`

One-line summary of the active constraints, for notebook printouts.

---

## `llm_perf.utils.data_check`

### `validate_int_fields(cfg: Dict[str, Any], fields: Iterable[str], *, min_value: Optional[int]=None, allow_float_for_int: bool=False, prefix: str='configuration') -> None`

Generic validator for integer-like fields in a dict.

- Ensures each field exists in ``cfg``.
- Ensures each value can be converted to int (optionally from float).
- Enforces an optional lower bound ``min\_value`` on the integer value.

Raises ``ValueError`` with a message that includes a snapshot of the requested fields and a list of per-field errors if any constraint is violated.

### `validate_positive_int_fields(cfg: Dict[str, Any], fields: Iterable[str], *, allow_float_for_int: bool=False, prefix: str='configuration') -> None`

Specialization of ``validate\_int\_fields`` enforcing values >= 1.

### `validate_nonnegative_int_fields(cfg: Dict[str, Any], fields: Iterable[str], *, allow_float_for_int: bool=False, prefix: str='configuration') -> None`

Specialization of ``validate\_int\_fields`` enforcing values >= 0.

### `validate_float_fields(cfg: Dict[str, Any], fields: Iterable[str], *, min_value: Optional[float]=None, prefix: str='configuration') -> None`

Generic validator for float-like fields in a dict.

- Ensures each field exists in ``cfg``.
- Ensures each value can be converted to float.
- Enforces an optional lower bound ``min\_value`` on the float value.

### `validate_nonnegative_float_fields(cfg: Dict[str, Any], fields: Iterable[str], *, prefix: str='configuration') -> None`

Specialization enforcing float values >= 0.0.

### `validate_positive_float_fields(cfg: Dict[str, Any], fields: Iterable[str], *, prefix: str='configuration') -> None`

Specialization enforcing float values > 0.0.

---

## `llm_perf.utils.equations`

Static registry of canonical equations (LaTeX + Python literal strings).

### class `LlmPerfEquations`

#### `list_ids(cls)`

#### `get(cls, eq_id: str)`

#### `latex(cls, eq_id: str)`

#### `expr(cls, eq_id: str)`

---

## `llm_perf.utils.plotting`

Lightweight plotting helpers for llm\_perf experiments.

### `save_config_tps_scatter(config_labels: Sequence[str], tps_values: Sequence[float], output_path: str | Path, *, title: str | None=None, ylabel: str='TTPS (tokens/s)') -> Path`

Render a scatter plot of configurations vs. throughput.

---
