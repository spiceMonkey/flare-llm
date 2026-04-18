from dataclasses import dataclass

@dataclass
class TuningSpec:
    """
    Execution / approximation knobs that are independent of the partition layout.
    """
    # Scenario sequence length
    S_decode: int = 2048

    # Collective algorithms: "ring" or "tree"
    tp_algorithm: str = "ring"
    ep_algorithm: str = "ring"

    # Collectives per layer
    n_TP_collectives: int = 2
    n_EP_collectives: int = 2
    n_SP_collectives: int = 1

    # Overlap factor ρ in [0, 1]: Fraction of local time utilized to hide comms.
    # t_stage = t_local + max(0, t_comm - ρ * t_local)
    overlap_factor: float = 0.0

    # Batch size for decode phase (B=1 is single-request decode)
    B_decode: int = 1

    # Prefill parameters (used by PrefillCalculator)
    S_input: int = 0            # prefill sequence length (0 = decode only)
    B_prefill: int = 1          # number of requests batched in prefill
    chunk_size: int = 0         # chunked prefill C (0 = no chunking)

    # Topology-specific collective algorithms. Inert on crossbar fabrics;
    # consumed by core/primitives/dispatch.cost_collective.
    #   torus_algorithm="swing" is reserved; raises NotImplementedError for now.
    #   collective_worst_case toggles dragonfly Valiant routing in Phase D.
    torus_algorithm: str = "ring"
    collective_worst_case: bool = False

