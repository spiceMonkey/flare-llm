from dataclasses import dataclass

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
      fetch_rule   "auto":   a hit is fetched only when fetch + suffix
                             prefill beats prefilling the whole prompt (a
                             latency rule; a skipped fetch costs a full
                             prefill of GPU work);
                   "always": every hit is fetched.
    """

    prefix_tier: str = ""
    idle_tier: str = ""
    fetch_rule: str = "auto"

    def __post_init__(self) -> None:
        if self.fetch_rule not in FETCH_RULES:
            raise ValueError(f"KVOffloadSpec.fetch_rule must be one of {FETCH_RULES}, got {self.fetch_rule!r}")
