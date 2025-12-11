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

    # Heuristic constant for activation traffic scaling
    c_act: float = 5.0

    # Overlap factor ρ in [0, 1]: Fraction of local time utilized to hide comms.
    # t_token = t_local + max(0, t_comm - ρ * t_local)
    overlap_factor: float = 0.0

