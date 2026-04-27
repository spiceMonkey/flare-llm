
from dataclasses import dataclass


@dataclass
class OverheadSpec:
    """Framework / CPU-stack overhead parameters (documentation/modeling/framework.md).

    Scope: CPU and software-stack overheads only. Network-fabric overheads
    (disaggregated KV transfer, co-located repack) live in `DisaggSpec`.

    All values default to 0.0 (zero-overhead baseline).

    **`t_graph_us` is a legacy fallback.** With the kernel-launch refactor
    (kernel_launch_overhead.md §5), the per-round CUDA-graph dispatch budget
    is derived from TuningSpec (`kernels_per_layer_compute`,
    `kernels_per_collective_call`, `kernel_launch_us`) and surfaced as
    `LatencyResults.t_SW`. The E2E calculator uses `t_graph_us` only when
    `LatencyResults.t_SW == 0` (SW modeling disabled by setting
    `kernel_launch_us = 0` in the tuner). Setting both is harmless — the
    derived term takes precedence — but `t_graph_us` is now redundant for
    most users.
    """

    # Per-request overhead (microseconds)
    t_sched_us: float = 0.0        # scheduling + batch assembly
    t_tok_us: float = 0.0          # tokenization

    # Per-step overhead (microseconds)
    t_graph_us: float = 0.0        # legacy CUDA graph replay constant; superseded by LatencyResults.t_SW
    t_sample_us: float = 0.0       # token sampling
    t_detok_us: float = 0.0        # detokenization
