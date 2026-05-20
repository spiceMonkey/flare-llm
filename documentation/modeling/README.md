# FLARE Modeling

**Author:** Yue Lu

First-principles analytical derivations for LLM inference performance modeling on a cluster. The chapters in this book are intentionally **implementation-independent** — they describe the cost model, the assumptions, and the symbol conventions; the FLARE code in [`spiceMonkey/flare-llm`](https://github.com/spiceMonkey/flare-llm) is one possible realization.

**Where to start.** [Decode and TPOT](decode.md) is the centerpiece — every other chapter is either a prerequisite (notation, attention variants, memory hierarchy) or a downstream composition (prefill, end-to-end, SLO). If you only have time for one chapter, read that one.

**Reading paths.**

- **Decode roofline (the main flow):** Notation → Decode → Attention variants → Memory hierarchy → Serving framework overhead.
- **Prefill and end-to-end:** Decode → Prefill → E2E latency metrics → SLO feasibility.
- **Collective communication (workload-agnostic):** the [Collective Communication](collectives/00_summary.md) subseries mirrors [`spiceMonkey/collective-comm`](https://github.com/spiceMonkey/collective-comm) and can be read independently.
