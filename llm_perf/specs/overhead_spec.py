
from dataclasses import dataclass


@dataclass
class OverheadSpec:
    """Framework / CPU-stack overhead parameters (documentation/modeling/framework.md).

    Scope: CPU and software-stack overheads only. Network-fabric overheads
    (disaggregated KV transfer, co-located repack) live in `DisaggSpec`.

    All values default to 0.0 (zero-overhead baseline).
    """

    # Per-request overhead (microseconds)
    t_sched_us: float = 0.0        # scheduling + batch assembly
    t_tok_us: float = 0.0          # tokenization

    # Per-step overhead (microseconds)
    t_graph_us: float = 0.0        # CUDA graph replay
    t_sample_us: float = 0.0       # token sampling
    t_detok_us: float = 0.0        # detokenization
