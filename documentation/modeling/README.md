# Decode Modeling

**Author:** Yue Lu

First-principles analytical derivation of decode-phase latency, throughput, and memory bandwidth for LLM inference on a cluster. The chapters are intentionally **implementation-independent** — they describe the cost model, the assumptions, and the symbol conventions; the FLARE code in [`spiceMonkey/flare-llm`](https://github.com/spiceMonkey/flare-llm) is one possible realization.

**Where to start.** [Decode and TPOT](decode.md) is the centerpiece. The other two chapters cover the supporting physics it references:

- [Attention variants](attention.md) — MHA / GQA / MQA / MLA, what each costs per token-per-layer
- [Memory hierarchy](sram.md) — multi-tier roofline that generalizes the legacy single-HBM model to SRAM-augmented and 3D-stacked devices

For the collective-communication cost model that decode references throughout (α–β cost form, ring / tree / hierarchical algorithms, topology composition, in-network reduction), see the dedicated tutorial at [spicemonkey.github.io/collective-comm](https://spicemonkey.github.io/collective-comm/) — linked from the [Collective Communication](collective_comm.md) chapter.

**Scope.** This book is intentionally focused. Prefill, end-to-end metrics, SLO feasibility, KV paging, and serving-framework overhead each have their own derivation docs in the [`documentation/modeling/`](https://github.com/spiceMonkey/flare-llm/tree/main/documentation/modeling) tree on GitHub, but they're not in this book yet.
