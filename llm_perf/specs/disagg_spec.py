
from dataclasses import dataclass


@dataclass
class DisaggSpec:
    """KV-handoff configuration (prefill.md §6).

    The `disaggregated` flag selects which branch applies; the other branch's
    fields are ignored. Defaults give zero handoff cost (co-located,
    partition-matched).
    """

    # Mode flag — False = co-located integrated, True = disaggregated prefill/decode
    disaggregated: bool = False

    # Which worker emits the first output token under disaggregation. Inert
    # when `disaggregated` is False (the one engine samples token 1 straight
    # from its prefill pass).
    #   "decode"  — the prefill worker runs with a one-token budget and hands
    #               over the KV cache; the decode worker re-runs the last
    #               prompt position and streams token 1. The transfer and one
    #               decode step both land on TTFT. vLLM's disaggregated
    #               prefill and NVIDIA Dynamo work this way.
    #   "prefill" — the prefill worker streams the token it sampled and the
    #               KV transfer delays the *second* token instead. DistServe
    #               and Splitwise work this way.
    first_token: str = "decode"

    # Co-located handoff (§6.3) — used when disaggregated=False
    colo_alpha_us: float = 0.0             # α_intra (scale-up startup)
    colo_repack_GBps: float = 0.0          # BW_intra; 0 → no repack modeled
    colo_repack_eta: float = 1.0           # η_repack ∈ [1, 2]

    # Disaggregated transfer (§6.4) — used when disaggregated=True
    # BW_inter is the *effective, delivered* per-GPU bandwidth (calibration knob),
    # not the NIC catalog line rate.
    inter_alpha_us: float = 0.0            # α_inter (single round-trip)
    inter_bandwidth_GBps: float = 0.0      # BW_inter (effective, delivered)
    N_WR: int = 0                          # RDMA WR count per handoff
    tau_WR_us: float = 0.0                 # per-WR posting latency
    overlap_rho_KV: float = 0.0            # ρ_KV ∈ [0, 1]
    repack_GBps: float = 0.0               # BW_intra,d (decode-side scale-up); 0 → no repack
    repack_eta: float = 1.0                # η_repack ∈ [1, 2]
