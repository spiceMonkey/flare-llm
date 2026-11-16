from dataclasses import dataclass
from typing import Tuple

FETCH_RULES = ("auto", "always")


@dataclass
class KVOffloadSpec:
    """Warm-memory strategy: what is kept off the GPUs, and where.

    Read only by the end-to-end layer. The hardware it refers to is the
    off-GPU tier list of the host (`HostSpec.tiers`); this spec names tiers
    of that list and says when a cached prefix is fetched. The workload side
    (how often a prefix is hit, what fraction of sessions is idle) is on
    `WorkloadSpec`.

    Hot memory is untouched by it: weights and the cache of a sequence that
    is being decoded always stay in GPU memory.

      prefix_tier  the tier that holds reusable prefixes; "" = none, so no
                   prefix reuse is priced from a tier;
      idle_tier    the tier that holds the caches of idle sessions; "" =
                   none, so no session is parked;
      overflow_tiers  further tiers that take idle sessions once `idle_tier`
                   is full, filled in the order given (for example host DRAM
                   first, then flash). Empty = one tier only;
      fetch_rule   "auto":   a hit is fetched only when fetch + suffix
                             prefill beats prefilling the whole prompt (a
                             latency rule; a skipped fetch costs a full
                             prefill of GPU work);
                   "always": every hit is fetched.
    """

    prefix_tier: str = ""
    idle_tier: str = ""
    fetch_rule: str = "auto"
    overflow_tiers: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.fetch_rule not in FETCH_RULES:
            raise ValueError(f"KVOffloadSpec.fetch_rule must be one of {FETCH_RULES}, got {self.fetch_rule!r}")
        self.overflow_tiers = tuple(self.overflow_tiers)
        if self.overflow_tiers and not self.idle_tier:
            raise ValueError("KVOffloadSpec.overflow_tiers needs an idle_tier to overflow from")
        names = (self.idle_tier,) + self.overflow_tiers
        if self.idle_tier and len(set(names)) != len(names):
            raise ValueError(f"KVOffloadSpec: idle_tier and overflow_tiers must be distinct, got {names}")
