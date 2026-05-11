"""Capture a golden regression fixture for the phase-model refactor.

Sweeps a representative grid of (model, system, partition, tuner)
configurations and serializes every numeric field of every dataclass
that `InferenceCalculator.run()` and `PrefillCalculator.run()` return.

The output file `golden.json` is the reference that `verify.py` replays
post-refactor to catch any numerical drift. This script is run **once**
at the tip of Phase 0a (router fix applied) so the golden is bit-identical
across both phases.

Usage:  python tests/regression/capture_golden.py
"""
import dataclasses
import json
import math
import sys
import warnings
from pathlib import Path

# Silence expected dim-alignment / mixed-chain UserWarnings emitted by the
# dispatcher for torus and hybrid systems. The golden captures numerics,
# not warning streams, so muting keeps the capture log legible.
warnings.filterwarnings("ignore", category=UserWarning)

from llm_perf.calculators.inference_calculator import InferenceCalculator
from llm_perf.calculators.prefill_calculator import PrefillCalculator
from llm_perf.io import (
    load_model_from_db,
    load_partition_from_db,
    load_system_from_db,
    load_tuner_from_db,
)
from llm_perf.specs.partition_spec import PartitionSpec
from llm_perf.specs.tuner_spec import TuningSpec


# Representative grid. The aim is broad coverage across model class (dense
# vs MoE), system topology (flat vs hierarchical fabric chain), and every
# parallelism axis — without exploding the fixture size.

MODELS = [
    "example.model.dense",
    "example.model.moe",
    "gpt_1_8t_moe",
    "deepseek_r1_0528",
]

SYSTEMS = [
    "example.sys",
    # Multi-tier NVL576 baseline. Loaded with framework.inc_enabled=False
    # below to capture the no-INC SW path (matches the prior
    # gb200.nvl576.hierarchical.json semantics, which was deleted in favor
    # of the multibox template + framework toggle pattern).
    "gb200.multibox",
    "tpu.v5p.pod",
    # NOTE: hypothetical / topology-variant systems used by the pareto_*
    # notebooks (flat 576-port "ideal", rail-optimized "rail.inc",
    # dmatrix.server.fast) are constructed inline in those notebooks via
    # deepcopy + small mutation. The recapture workstream that fixes this
    # script (post-Phase-E, also needs FrameworkSpec migration + drop of
    # the removed load_partition_from_db import) should reproduce them
    # the same way if golden coverage of those scenarios is wanted.
]

# (PP, TP, EP, SP). Each row exercises a distinct sharding regime.
PARTITION_SHAPES = [
    (1, 1, 1, 1),   # unsharded baseline
    (1, 4, 1, 1),   # TP only
    (1, 8, 1, 1),   # TP deep
    (2, 4, 1, 1),   # TP + PP
    (4, 8, 1, 1),   # TP + PP deep
    (1, 4, 4, 1),   # TP + EP (MoE)
    (1, 8, 8, 1),   # TP + EP deep (MoE)
    (2, 4, 2, 1),   # TP + PP + EP
    (1, 2, 1, 2),   # TP + SP
    (2, 4, 2, 2),   # all four axes
    (1, 144, 1, 1), # cross-rack TP (exercises hierarchical AR on multi-tier)
]

# Tuner shapes, hand-picked to cover the decode-only, prefill, batched,
# chunked, and long-context paths without Cartesian blowup.
# (S_decode, B_decode, S_input, B_prefill, chunk_size)
TUNER_SHAPES = [
    (1024, 1, 0, 1, 0),         # decode-only, single, short context
    (1024, 16, 0, 1, 0),         # decode-only, batched, short context
    (8192, 1, 0, 1, 0),         # decode-only, single, long context
    (8192, 64, 0, 1, 0),         # decode-only, large batch, long context
    (2048, 1, 4096, 1, 0),       # prefill, single request
    (2048, 1, 4096, 16, 0),      # prefill, batched
    (2048, 1, 16384, 1, 2048),   # prefill, chunked long context
]


def _replace_tuner(base: TuningSpec, **kwargs) -> TuningSpec:
    return dataclasses.replace(base, **kwargs)


def _fits_partition(model, system, pp, tp, ep, sp) -> bool:
    """Reject partitions that don't fit the system or model topology."""
    total_shards = pp * tp * ep * sp
    if total_shards > system.num_devices:
        return False
    # DP must be at least 1.
    if system.num_devices % total_shards != 0:
        return False
    # EP > 1 only makes sense for MoE models and must not exceed n_experts.
    if ep > 1:
        if model.moe is None:
            return False
        if ep > model.moe.n_experts:
            return False
    return True


def _to_jsonable(obj):
    """Recursively convert dataclasses / dicts / lists to JSON-safe types."""
    if dataclasses.is_dataclass(obj):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, float):
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        if math.isnan(obj):
            return "nan"
        return obj
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    # Fallback: repr (shouldn't hit numeric fields)
    return repr(obj)


def _run_case(model, system, partition, tuner):
    """Run both calculators and flatten the result trees for serialization."""
    out = {}
    ir = InferenceCalculator(model, system, partition, tuner).run()
    out["decode"] = {
        "memory": _to_jsonable(ir.memory),
        "flops": _to_jsonable(ir.flops),
        "traffic": _to_jsonable(ir.traffic),
        "comm": _to_jsonable(ir.comm),
        "latency": _to_jsonable(ir.latency),
    }
    # Prefill path only activates when S_input > 0. Still capture the result
    # for completeness — with S_input=0, prefill quantities collapse to
    # trivial values (zero FLOPs / zero KV writes / non-zero overhead floor).
    pr = PrefillCalculator(model, system, partition, tuner).run()
    out["prefill"] = {
        "flops": _to_jsonable(pr.flops),
        "traffic": _to_jsonable(pr.traffic),
        "comm": _to_jsonable(pr.comm),
        "latency": _to_jsonable(pr.latency),
    }
    return out


def main() -> int:
    out_path = Path(__file__).parent / "golden.json"
    fixture = {"schema": "llm_perf.regression.golden.v1", "cases": []}

    n_kept = 0
    n_skipped = 0
    n_failed = 0

    base_partition = load_partition_from_db("example.partition")
    base_tuner = load_tuner_from_db("example.tuner")

    for model_id in MODELS:
        model = load_model_from_db(model_id)
        for system_id in SYSTEMS:
            system = load_system_from_db(system_id)
            for pp, tp, ep, sp in PARTITION_SHAPES:
                if not _fits_partition(model, system, pp, tp, ep, sp):
                    n_skipped += 1
                    continue
                partition = PartitionSpec(PP=pp, TP=tp, EP=ep, SP=sp)
                for s_dec, b_dec, s_in, b_pf, chunk in TUNER_SHAPES:
                    tuner = _replace_tuner(
                        base_tuner,
                        S_decode=s_dec,
                        B_decode=b_dec,
                        S_input=s_in,
                        B_prefill=b_pf,
                        chunk_size=chunk,
                    )
                    case_id = (
                        f"{model_id}|{system_id}|PP={pp},TP={tp},EP={ep},SP={sp}"
                        f"|S_dec={s_dec},B_dec={b_dec},S_in={s_in},B_pf={b_pf},C={chunk}"
                    )
                    try:
                        results = _run_case(model, system, partition, tuner)
                    except Exception as exc:  # noqa: BLE001
                        fixture["cases"].append({
                            "id": case_id, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                        })
                        n_failed += 1
                        continue
                    fixture["cases"].append({
                        "id": case_id,
                        "status": "ok",
                        "inputs": {
                            "model": model_id,
                            "system": system_id,
                            "partition": {"PP": pp, "TP": tp, "EP": ep, "SP": sp},
                            "tuner": {
                                "S_decode": s_dec, "B_decode": b_dec,
                                "S_input": s_in, "B_prefill": b_pf,
                                "chunk_size": chunk,
                            },
                        },
                        "results": results,
                    })
                    n_kept += 1

    with out_path.open("w") as f:
        json.dump(fixture, f, indent=2, sort_keys=True)

    print(f"Golden fixture written: {out_path}")
    print(f"  kept={n_kept}  skipped={n_skipped}  failed={n_failed}")
    print(f"  total cases in fixture: {len(fixture['cases'])}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
