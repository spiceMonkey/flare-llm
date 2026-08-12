# API Reference

**Author:** Yue Lu  
**Date:** August 2026  

Auto-generated reference for the public `llm_perf` API, extracted from the source by `scripts/gen_api_docs.py`. Regenerate after changing any public signature: `python scripts/gen_api_docs.py`.

Shorthand used throughout: tensor parallelism (TP), expert parallelism (EP), pipeline parallelism (PP), sequence parallelism (SP), data parallelism (DP), Mixture-of-Experts (MoE), key/value cache (KV cache), time-to-first-token (TTFT), time-per-output-token (TPOT), high-bandwidth memory (HBM), in-network collective (INC).

## Modules

| Page | Scope | Public items |
|---|---|---|
| [Calculators](calculators.md) | End-to-end entry points. Each composes the specs, runs the five-stage pipeline, and returns a results dataclass. | 6 |
| [Specs](specs.md) | Typed dataclass inputs. The composition is model x system x partition x tuner x framework, plus optional overhead/disagg specs for end-to-end runs. | 17 |
| [Core](core.md) | Pure analytical model functions. Each takes (model, system, partition, tuner, framework) and returns a results dataclass; no global state. | 28 |
| [IO](io.md) | Spec loaders. `load_*_from_db` reads the bundled JSON database by id; `load_*_spec` / `*_spec_from_json_dict` read an arbitrary path or dict. | 22 |
| [Utils](utils.md) | Helpers: HuggingFace model adapter, DRAM bandwidth calculator, constants, validation, and plotting. | 15 |

## Minimal usage

```python
from llm_perf import InferenceCalculator, PartitionSpec
from llm_perf.io import (
    load_model_from_db, load_system_from_db, load_tuner_from_db, load_framework_from_db,
)

model     = load_model_from_db("example.model.dense")
system    = load_system_from_db("example.sys")
partition = PartitionSpec(PP=1, TP=4, EP=1, SP=1)
tuner     = load_tuner_from_db("example.tuner")
framework = load_framework_from_db("dynamo_trt")

results = InferenceCalculator(model, system, partition, tuner, framework).run()
print(results.latency.TTPS, results.memory.fits_in_HBM)
```

For the methodology behind these models, see the [Decode Modeling book](https://spicemonkey.github.io/flare-llm/).
