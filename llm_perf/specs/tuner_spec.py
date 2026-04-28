from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class MemoryPlacementSpec:
    """Per-data-class memory tier placement (sram.md §1.3).

    Each field selects the tier that holds the corresponding data class:
      - "auto": greedy fastest-first — fill faster tiers first, spill to
        slower tiers when capacity is exhausted (sram.md §1.3 first policy).
      - "<tier_name>": pin this data class to the named tier (must match a
        `MemoryTierSpec.name` on the device); CapacityError if it doesn't fit
        (sram.md §1.3 second policy — d-Matrix Aviator-style mode toggle).

    `auto_priority` controls the greedy tiebreaker when **both** fields are
    "auto": which class claims the fastest tier first. Default "weights"
    matches the convention that weights are a stable size for a given
    deployment and should pin to the fast tier. Set "kv" to flip the order
    when KV-bound workloads (long context, large batch) want SRAM-resident
    KV at the cost of spilling weights. Inert when either class is
    explicitly pinned.

    Defaults are "auto" / "auto" / "weights", which on a single-tier device
    collapses to the legacy "everything on HBM" behavior — bitwise identical
    to pre-PR2 `t_mem = T_step / BW_mem`.
    """

    weights_tier: str = "auto"   # tier name or "auto"
    kv_tier: str = "auto"        # tier name or "auto"
    auto_priority: str = "weights"  # "weights" or "kv"


@dataclass
class TuningSpec:
    """
    Execution / approximation knobs that are independent of the partition layout.
    """
    # Scenario sequence length
    S_decode: int = 2048

    # Per-phase × per-collective algorithm choice.
    #   Admissible values: "ring", "tree", "auto".
    #   "auto" is a placeholder — must be resolved by
    #   `core/collective_algo_opt.optimize_collective_algorithms(...)` before
    #   passing the tuner to `InferenceCalculator.run()`. Reaching the
    #   dispatcher with "auto" raises ValueError.
    #   SP is always ring AG (no knob — only shipped option per
    #   collectives/01_collective_algorithms.md §6).
    #   The legacy fields `tp_algorithm` / `ep_algorithm` are deprecated
    #   single-knob aliases; the loader copies them into both _decode and
    #   _prefill when the per-phase fields are unspecified.
    tp_algorithm_decode: str = "ring"
    tp_algorithm_prefill: str = "ring"
    ep_algorithm_decode: str = "ring"
    ep_algorithm_prefill: str = "ring"

    # Legacy single-knob fields (deprecated; loader-only fallbacks). New code
    # should set the per-phase fields directly.
    tp_algorithm: str = "ring"
    ep_algorithm: str = "ring"

    # NCCL API call counts per layer. These match both the cost-model
    # accumulator (decode.md §5.5) and the SW launch counter (decode.md §6.3.2)
    # so a single field describes both.
    # n_TP_collectives: TP all-reduces per layer (post-attn + post-FFN = 2).
    # n_EP_collectives: MoE A2A calls per MoE layer (dispatch + combine = 2);
    #     each call costs one single-direction A2A — see dispatch.py's
    #     `_cost("moe_a2a", ...)`.
    # n_SP_collectives: SP all-gathers per layer (1 with ring SP).
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
    torus_algorithm: str = "ring"

    # In-network collectives opt-out. When True (default), dispatcher routes
    # AR/AG over any crossbar tier chain whose every tier declares inc != "none"
    # to the INC primitives (n_α collapse + BW-eff doubling for AR).
    # Set False to force software ring/tree fallback — useful for A/B
    # measurements and for hardware where SHARP is disabled at runtime.
    # Inert on tier chains where any crossed tier has inc == "none".
    inc_enabled: bool = True

    # Per-data-class memory tier placement (sram.md §1.3). Defaults are
    # "auto"/"auto" — greedy fastest-first, which collapses to legacy
    # behavior on single-tier devices. New multi-tier devices may pin
    # weights or KV to a named tier (e.g. d-Matrix Capacity Mode pins
    # weights to "lpddr5" to free SRAM for larger batch / context).
    placement: MemoryPlacementSpec = field(default_factory=MemoryPlacementSpec)

    # ── SW overhead modeling (kernel_launch_overhead.md §5) ─────────────
    # Per-microbatch dispatch budget on each PP stage (same units as t_stage):
    #     t_SW = τ_launch · [
    #              (L / PP)     · (k_compute + k_collective · (n_TP + n_SP))
    #            + (L_moe / PP) · k_collective · n_EP
    #            + k_pp_hop
    #          ]
    # where the n_* terms are the per-layer collective call counts above
    # (zeroed for axes where the parallelism is 1, i.e. that collective
    # never fires). EP launches only fire on the L_moe/PP MoE layers this
    # stage owns (mirrors the L_moe/PP factor in §5.5's t_comm formula).
    # The PP-hop term contributes k_pp_hop launches per microbatch transit
    # and is inert when PP = 1.
    #
    # Production-realistic defaults: CUDA Graphs on (τ_launch ≈ 1.5 μs),
    # ~10 kernels per layer (after typical fusion), ~2 kernels per NCCL
    # collective call, full async overlap (ρ_SW = 1) so SW acts as a
    # *floor* on t_step_user — kicks in only when t_SW > t_stage. To
    # disable the SW term entirely (legacy roofline), set
    # `kernel_launch_us = 0.0` in tuner JSON.
    kernels_per_layer_compute: int = 10
    kernels_per_collective_call: int = 2
    # Kernels per PP boundary on each device, per microbatch: 1 recv from
    # upstream + 1 send to downstream = 2 (middle stages). Edge stages do
    # only one direction; the formula below treats every stage as middle
    # for simplicity (off by one PP × τ on edge stages, negligible at PP>>1).
    # Set to 1 if `ncclSendRecv` (or a custom kernel) fuses the pair.
    kernels_per_pp_hop: int = 2
    kernel_launch_us: float = 1.5       # 0 disables; ~1.5 μs with CUDA Graphs, ~7 μs without
    # ρ_SW ∈ [0, 1]; 1 = full async overlap (SW hidden by GPU work).
    # **Caveat:** 1.0 is the upper-end case — accurate for CUDA-Graphs-replayed
    # steady-state on TensorRT-LLM / vLLM / SGLang where the CPU is one
    # `cudaGraphLaunch` per microbatch and has 1000× slack. Empirically these
    # stacks measure ~0.85-0.95. Eager-mode PyTorch / Python serving sees
    # ρ_SW ~0.3-0.6 because Python interpreter overhead breaks the
    # CPU-runs-ahead invariant. The 1.0 default matches the framework's
    # roofline philosophy (upper bound; dial down to model imperfections).
    # Note that t_SW is still a *hard floor* when t_SW > t_stage regardless
    # of ρ_SW, so the optimistic default does not hide the dispatch tax in
    # the SW-bound regime.
    sw_overlap_factor: float = 1.0

    # Tensor Core efficiency curve η_TC(mb) for compute roofline.
    # Maps microbatch size mb (= B / PP) to a derate factor in [0, 1].
    # `compute_latency` uses piecewise-linear interpolation between
    # adjacent keys; mb values below the minimum key clamp to that key's
    # efficiency, mb values above the maximum clamp to that key's value.
    # When None, η_TC = 1.0 always (legacy behavior — no compute derate).
    # Representative FP8 ramp on Hopper / Blackwell:
    #     {1: 0.05, 16: 0.4, 64: 0.8, 256: 1.0}
    # See documentation/explaining/practical_pp_choice.md §3.3 for the
    # tile-floor argument that motivates this curve.
    tensor_core_efficiency: Optional[Dict[int, float]] = None

