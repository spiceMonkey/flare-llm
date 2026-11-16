from dataclasses import dataclass
from typing import Optional


@dataclass
class WorkloadSpec:
    """Request-level workload, read only by the end-to-end layer.

    The decode and prefill calculators never see this spec: they price one
    step and one prefill pass. The end-to-end layer needs to know how many
    tokens a request generates, and optionally how many requests arrive, to
    turn those two prices into GPU time per request and a pool size.

    Prompt length and the two batch sizes stay on `TuningSpec`
    (`S_input`, `B_prefill`, `B_decode`).
    """

    # S_out: tokens generated per request.
    output_tokens: int

    # lambda: requests per second offered to the deployment. None sizes
    # nothing: only per-request and per-GPU quantities are reported.
    request_rate: Optional[float] = None

    # Highest utilisation a pool is sized to, in (0, 1]. Queueing is not
    # modelled; this is the headroom an operator leaves for it. 1.0 sizes
    # each pool to exactly carry the offered rate.
    utilisation_cap: float = 1.0

    # Prefix reuse. A fraction `prefix_hit_rate` of requests find the first
    # `prefix_tokens` tokens of their prompt already cached off-GPU; they
    # fetch that cache and prefill only the rest. The hit rate is an input:
    # predicting it is out of scope. Both default to "no reuse".
    prefix_hit_rate: float = 0.0
    prefix_tokens: int = 0

    # Fraction of sessions that are idle at any moment, in [0, 1): waiting on
    # the user or on a tool, with a cache that is not being decoded. Idle
    # caches can be parked in warm memory (`KVOffloadSpec.idle_tier`), which
    # frees GPU memory for active ones. An input, like the hit rate.
    idle_fraction: float = 0.0

    def __post_init__(self) -> None:
        if int(self.output_tokens) != self.output_tokens or self.output_tokens <= 0:
            raise ValueError(f"WorkloadSpec.output_tokens must be a positive integer, got {self.output_tokens!r}")
        if self.request_rate is not None and not self.request_rate >= 0.0:
            raise ValueError(f"WorkloadSpec.request_rate must be >= 0 or None, got {self.request_rate!r}")
        if not 0.0 < self.utilisation_cap <= 1.0:
            raise ValueError(f"WorkloadSpec.utilisation_cap must be in (0, 1], got {self.utilisation_cap!r}")
        if not 0.0 <= self.prefix_hit_rate <= 1.0:
            raise ValueError(f"WorkloadSpec.prefix_hit_rate must be in [0, 1], got {self.prefix_hit_rate!r}")
        if int(self.prefix_tokens) != self.prefix_tokens or self.prefix_tokens < 0:
            raise ValueError(f"WorkloadSpec.prefix_tokens must be a non-negative integer, got {self.prefix_tokens!r}")
        if not 0.0 <= self.idle_fraction < 1.0:
            raise ValueError(f"WorkloadSpec.idle_fraction must be in [0, 1), got {self.idle_fraction!r}")
