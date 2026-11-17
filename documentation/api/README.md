# API Reference

**Author:** Yue Lu  
**Date:** November 2026  

Auto-generated reference for the public `llm_perf` API: signatures, dataclass fields and docstrings are extracted from the package source by a generator script. Do not edit these pages by hand; they are overwritten whenever the reference is regenerated, so change the docstring or signature in the source instead.

Shorthand used throughout: tensor parallelism (TP), expert parallelism (EP), pipeline parallelism (PP), sequence parallelism (SP), data parallelism (DP), Mixture-of-Experts (MoE), key/value cache (KV cache), time-to-first-token (TTFT), time-per-output-token (TPOT), high-bandwidth memory (HBM), in-network collective (INC).

## Modules

| Page | Scope | Public items |
|---|---|---|
| [Calculators](calculators.md) | End-to-end entry points. Each composes the specs, runs the five-stage pipeline, and returns a results dataclass. | 6 |
| [Specs](specs.md) | Typed dataclass inputs. The composition is model x system x partition x tuner x framework, plus optional overhead/disagg specs for end-to-end runs. | 22 |
| [Core](core.md) | Pure analytical model functions. Each takes (model, system, partition, tuner, framework) and returns a results dataclass; no global state. | 55 |
| [Core primitives](core_primitives.md) | Building blocks the core models compose: sharding factors and partition layout, weight / KV / FLOP quantities, collective cost formulas, topology-aware dispatch, and the per-stage communication aggregator. | 71 |
| [IO](io.md) | Spec loaders. `load_*_from_db` reads the bundled JSON database by id; `load_*_spec` / `*_spec_from_json_dict` read an arbitrary path or dict. | 22 |
| [Utils](utils.md) | Helpers: HuggingFace model adapter, DRAM bandwidth calculator, constants, validation, and plotting. | 15 |

## Minimal usage

```python
from llm_perf import InferenceCalculator, PartitionSpec
from llm_perf.core.collective_algo_opt import optimize_collective_algorithms
from llm_perf.io import (
    load_model_from_db, load_system_from_db, load_tuner_from_db, load_framework_from_db,
)

model     = load_model_from_db("example.model.dense")
system    = load_system_from_db("example.sys")
partition = PartitionSpec(PP=1, TP=4, EP=1, SP=1)
tuner     = load_tuner_from_db("example.tuner")

# A per-stack framework file leaves its collective algorithms on "auto":
# resolve them for this partition and batch before running the calculator.
# (Leave the framework argument out to run the neutral default, which needs
# no such pass. The co-located stacks, "dynamo_*", also require TP == EP when both exceed 1.)
framework = load_framework_from_db("trt")
framework = optimize_collective_algorithms(model, partition, system, tuner, framework)

results = InferenceCalculator(model, system, partition, tuner, framework).run()
print(results.latency.TTPS, results.memory.fits_in_HBM)
```

For the methodology behind these models, see the [Decode Modeling book](https://spicemonkey.github.io/flare-llm/).
