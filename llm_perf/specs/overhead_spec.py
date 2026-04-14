
from dataclasses import dataclass


@dataclass
class OverheadSpec:
    """Framework overhead parameters (documentation/modeling/framework.md).

    All values default to 0.0 (zero-overhead co-located baseline).
    """

    # Per-request overhead (microseconds)
    t_sched_us: float = 0.0        # scheduling + batch assembly
    t_tok_us: float = 0.0          # tokenization

    # Per-step overhead (microseconds)
    t_graph_us: float = 0.0        # CUDA graph replay
    t_sample_us: float = 0.0       # token sampling
    t_detok_us: float = 0.0        # detokenization

    # Disaggregated KV transfer (0 = co-located)
    disagg_alpha_us: float = 0.0   # inter-cluster startup latency
    disagg_bandwidth_GBps: float = 0.0  # inter-cluster bandwidth (GB/s)
