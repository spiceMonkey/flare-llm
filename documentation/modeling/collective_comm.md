# Collective Communication

The collective-communication cost model, algorithm library, and topology composition that this book references throughout (in [Decode and TPOT](decode.md), [Prefill and TTFT](prefill.md), [E2E latency metrics](e2e.md), and elsewhere) live in a dedicated tutorial site:

## → [spicemonkey.github.io/collective-comm](https://spicemonkey.github.io/collective-comm/)

That site covers:

- The Hockney α–β cost model
- Algorithm catalog (ring, tree / DBT, Rabenseifner halving-doubling, PAT, hierarchical, torus, in-network)
- Topology mapping (crossbar, torus, mesh)
- Hierarchical composition (RS → sub-AR → AG telescoping)
- In-network collectives (NVLS, Quantum SHARP, hw_a2a)
- Contention and congestion models

The source repository is [`spiceMonkey/collective-comm`](https://github.com/spiceMonkey/collective-comm), and an auto-synced mirror of those docs lives under [`documentation/modeling/collectives/`](https://github.com/spiceMonkey/flare-llm/tree/main/documentation/modeling/collectives) in this repo for offline reference.
