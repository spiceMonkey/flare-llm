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
    Workload knobs and chip-side calibration curves. Orthogonal to:
      - ModelSpec (architecture)
      - SystemSpec (hardware + per-tier eta)
      - PartitionSpec (sharding)
      - FrameworkSpec (SW-stack runtime: host overhead, collective algos,
                       MLA mode, MoE A2A pattern, etc.)
    """
    # Scenario sequence length
    S_decode: int = 2048

    # Batch size for decode phase (B=1 is single-request decode)
    B_decode: int = 1

    # Prefill parameters (used by PrefillCalculator)
    S_input: int = 0            # prefill sequence length (0 = decode only)
    B_prefill: int = 1          # number of requests batched in prefill
    chunk_size: int = 0         # chunked prefill C (0 = no chunking)

    # ── Framework-axis fields moved to FrameworkSpec ───────────────────
    # The following fields used to live here; they are now on FrameworkSpec:
    #   Phase B (mla(stage 5/framework-spec phase B)):
    #     - c_serving_per_seq_us (renamed from t_serving_per_seq_us)
    #     - kernel_launch_us, kernels_per_layer_compute,
    #       kernels_per_collective_call, kernels_per_pp_hop
    #     - sw_overlap_factor
    #     - moe_a2a_pattern, mla_mode, inc_enabled
    #   Phase E:
    #     - tp_algorithm_decode / tp_algorithm_prefill
    #     - ep_algorithm_decode / ep_algorithm_prefill
    #     - tp_algorithm / ep_algorithm (legacy single-knob aliases — DROPPED)
    #     - torus_algorithm
    #     - n_TP_collectives / n_EP_collectives / n_SP_collectives
    #     - overlap_factor → renamed comm_overlap_factor (ρ_comm; mirrors
    #       sw_overlap_factor on the SW-vs-GPU side)
    # See FrameworkSpec docstring + database/framework/ for stack JSONs.
    # Loader rejects legacy tuner JSONs that still set these fields with
    # a clear migration hint pointing at load_framework_from_db().

    # Per-data-class memory tier placement (sram.md §1.3). Defaults are
    # "auto"/"auto" — greedy fastest-first, which collapses to legacy
    # behavior on single-tier devices. New multi-tier devices may pin
    # weights or KV to a named tier (e.g. d-Matrix Capacity Mode pins
    # weights to "lpddr5" to free SRAM for larger batch / context).
    placement: MemoryPlacementSpec = field(default_factory=MemoryPlacementSpec)

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

    # B-dependent sustained HBM bandwidth curve η_β(B) for memory roofline.
    # Maps active-sequence count B to a derate factor in (0, 1] applied
    # multiplicatively on top of the per-tier `eta_beta` and the constant
    # `bw_eta` calibration knob. Piecewise-linear interpolation between
    # adjacent keys; B values below the minimum key clamp to that key's
    # efficiency, B values above the maximum clamp to that key's value.
    # When None, η_β(B) = 1.0 always (legacy behavior — no B-dependent
    # derate; the constant `bw_eta` and per-tier `eta_beta` continue to
    # apply unchanged).
    # Representative HBM3e ramp on Blackwell production stacks:
    #     {1: 0.92, 64: 0.85, 512: 0.75, 4096: 0.55}
    # Mirrors the `tensor_core_efficiency` shape exactly. See
    # documentation/modeling/decode.md §6.2 for the derivation and
    # documentation/modeling/notation.md §20 for the symbol register.
    bw_efficiency: Optional[Dict[int, float]] = None

    # ── Speculative decoding (decode.md §8) ────────────────────────────
    # n_tok_draft = 0 disables speculation (vanilla decode, default).
    # n_tok_draft > 0 enables a Multi-Token Prediction (MTP) / EAGLE / Medusa
    # style verify pass that processes n_tok_verify = n_tok_draft + 1 tokens
    # per sequence per step and emits N_tok_per_step accepted tokens on
    # average. p_accept is the per-token draft acceptance probability ∈ [0,1].
    # Both must be set non-trivially for speculation to take effect; the
    # latency model derives N_tok_per_step from the truncated geometric
    # acceptance distribution (decode.md §8.2) and TPOT_spec from the
    # verify-step roofline (decode.md §8.3, §9.4).
    n_tok_draft: int = 0
    p_accept: float = 0.0

