
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
    inter_bandwidth_GBps: float = 0.0      # BW_inter (effective, delivered), per stream
    # Number of device pairs that transfer in parallel. A replica's KV cache
    # is spread over its devices, and each device with its own NIC can ship
    # its shard at the same time, so the cache moves at
    # transfer_streams x BW_inter. Set it to min(prefill replica devices,
    # decode replica devices) when every device has a NIC; 1 (default) models
    # a single link carrying the whole cache. Measured multi-NIC transfer
    # rates scale this way (Mooncake reports 87 GB/s over 4 x 200 Gb/s and
    # 190 GB/s over 8 x 400 Gb/s).
    transfer_streams: int = 1
    # True derives the stream count from the two replicas instead of using
    # `transfer_streams`: min(prefill replica devices, decode replica
    # devices), every device with its own link. With no separate prefill
    # pool given to the calculator both replicas are the decode replica.
    auto_streams: bool = False
    N_WR: int = 0                          # RDMA WR count per handoff
    tau_WR_us: float = 0.0                 # per-WR posting latency
    overlap_rho_KV: float = 0.0            # ρ_KV ∈ [0, 1]
    # Part of the transfer that overlap with prefill cannot remove (layer-wise
    # streaming still ships the last layers after prefill ends). The handoff
    # is never shorter than this, nor is the floor ever longer than the
    # un-overlapped transfer itself. Splitwise measures about 8 ms on A100 and
    # about 5 ms on H100. 0 (default) keeps the plain max(0, ...) clamp.
    overlap_resid_us: float = 0.0
    # False (default): the transfer goes straight into GPU memory (RDMA with
    # GPU-direct), so only the link between pools limits it. True: it lands
    # in host memory first and is then copied into the GPUs over the decode
    # system's `HostSpec.handoff_path`, whose stages join the path. Inert
    # when the decode system has no HostSpec.
    staged: bool = False
    repack_GBps: float = 0.0               # BW_intra,d (decode-side scale-up); 0 → no repack
    repack_eta: float = 1.0                # η_repack ∈ [1, 2]

    # True charges a repack (either branch) only when the prefill and the
    # decode replica split a sequence's cache differently
    # (`core.pool_model.kv_layout`); with equal layouts the repack is zero
    # whatever the repack rate. False (default) charges it whenever a repack
    # rate is set, as before.
    auto_repack: bool = False
