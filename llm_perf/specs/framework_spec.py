"""FrameworkSpec — SW-stack-specific runtime behavior.

Captures the host-side overhead model, execution-mode choices, and
collective-dispatch knobs that depend on the production serving stack
rather than the workload, model, or hardware. Five orthogonal spec
axes:

    ModelSpec    = architecture (LlmModelSpec, MoESpec, MLASpec)
    SystemSpec   = hardware (DeviceSpec, FabricSpec, MemoryTierSpec)
    PartitionSpec = sharding (PP, TP, EP, SP)
    TuningSpec   = workload (S, B, chunk_size, placement, speculation,
                   chip-side derate curves: tensor_core_efficiency,
                   bw_efficiency)
    FrameworkSpec = stack runtime (host overhead, kernel launch budget,
                   collective algorithms + counts + overlap, MLA mode,
                   MoE A2A pattern, INC opt-in)

Any (model, system, partition, tuner, framework) tuple is a runnable
deployment configuration. Pre-canned framework JSONs live in
`database/framework/` and correspond 1:1 to the production stack
identifiers in InferenceX measurement metadata. See
`documentation/modeling/decode.md §7.1, §7.2` (host overhead),
`§5` (collective algorithms), `§6` (overlap composition).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class FrameworkSpec:
    """SW-stack-specific runtime behavior model.

    Sourced from `decode.md §7.1, §7.2` (host overhead) and
    `attention.md §3.5` (MLA execution mode). Per-stack typical values
    live in the `database/framework/` JSONs.
    """

    name: str

    # ── Host-side overhead model (decode.md §7.1, §7.2) ────────────────
    # Per-sequence per-step host work (decode.md §7.3):
    # PagedAttention block-table gather, continuous-batching scheduler,
    # per-sequence sampling glue, token-append + KV bookkeeping. Linear
    # in B. **Composed with GPU work via `seq_overlap_factor`** —
    # under CUDA-Graph replay the CPU runs ahead and host work hides
    # behind GPU compute until it exceeds the per-step GPU time.
    #
    # Gross per-step host work:
    #   t_step_seq = c_seq_us * B * 1e-6     [seconds]
    #
    # Net contribution to t_step_user (after overlap with the per-step
    # hardware window t_step_base = γ_pp · t_stage,with_kernel + t_LM):
    #   t_step_seq = max(0, t_step_seq
    #                     - seq_overlap_factor * t_step_base)
    #
    # Stack-dependent ranges for c_seq_us (decode.md §7.3):
    # - C++/CUDA-graph + orchestrator (Dynamo+TRT): 5-22 µs/seq
    # - Mixed orchestrator + Python (Dynamo+SGLang): 25-50 µs/seq
    # - Raw C++ runtime (raw TRT-LLM): 50-100 µs/seq
    # - Aggressively fused C++: ~10 µs/seq lower bound
    # - Python-heavy (vLLM, SGLang eager): 30-60 µs/seq
    c_seq_us: float = 0.0

    # Fraction of t_step_seq hidden behind GPU compute. Same physics
    # as kernel_overlap_factor (below) but applied to host-side per-sequence
    # work rather than per-kernel dispatch. 1.0 (default) = full CUDA-
    # Graph-replay overlap — CPU runs ahead, host work hides until it
    # exceeds the per-step GPU time. 0.0 = eager-mode serialization,
    # host work always blocks. Caveat: only modulates the *hideable*
    # portion; t_step_seq remains a hard floor when it exceeds
    # t_step_base regardless of this knob.
    seq_overlap_factor: float = 1.0

    # Per-kernel dispatch budget (decode.md §7.1).
    #   t_stage_kernel = tau_launch * (k_compute + k_collective + k_pp_hop)
    # Production-realistic anchors:
    # - CUDA Graphs replay (Dynamo / TRT-LLM): ~1.5 us
    # - Eager-mode PyTorch / Python serving: ~7 us
    # - SGLang Python paths: ~12 us
    kernel_launch_us: float = 1.5

    # Kernel-fanout assumptions per layer (decode.md §6.3.2). After
    # typical fusion: ~10 compute kernels per layer, 2 per NCCL
    # collective call, 2 per PP-hop (1 send + 1 recv unless fused).
    kernels_per_layer_compute: int = 10
    kernels_per_collective_call: int = 2
    kernels_per_pp_hop: int = 2

    # Fraction of t_stage_kernel hidden behind GPU compute (decode.md §7.1).
    # 1.0 = full async overlap (CUDA-Graph replay steady-state on
    # TensorRT-LLM / vLLM / SGLang where the CPU runs ahead). Eager-mode
    # PyTorch sees ~0.3-0.6 because the Python interpreter breaks the
    # CPU-runs-ahead invariant. Caveat: kernel_overlap_factor only modulates
    # the *hideable* portion; t_stage_kernel remains a hard floor when it
    # exceeds t_stage_GPU regardless of this knob. The "kernel" in the
    # name distinguishes this from the other host-side overlap factors
    # (seq_overlap_factor for per-sequence host work, and a future
    # per-step host floor); together they form the "SW overhead" umbrella.
    kernel_overlap_factor: float = 1.0

    # ── Framework execution-mode choices ───────────────────────────────
    # MoE All-to-All data-flow pattern under DP-attention (decode.md §5.2).
    #   "gather"  — gather-then-dispatch (default); dispatch on full B.
    #               Per-rank dispatch payload = B*k*H*b. Conservative
    #               ceiling shipped by general-purpose MoE backends.
    #   "scatter" — scatter-direct; no AG before MoE. Dispatch operates
    #               on per-rank attention-sharded tokens of size B/G_TP.
    #               ~G_TP× payload reduction. Production DSv3/R1 pattern
    #               when DeepEP-style backends are in use. Requires
    #               attention_mode="dp" (no-op otherwise).
    moe_a2a_pattern: str = "gather"

    # MLA execution mode (attention.md §3.5). Inert when model has no
    # MLA extension (`model.mla is None`).
    #   "absorbed"     — production default. Folds W_UK / W_UV into Q / O
    #                    at compile time so attention runs in d_c-dim
    #                    latent space. Used by NVIDIA TensorRT-LLM and
    #                    SGLang's DSv3 path.
    #   "materialized" — reference / CPU-fallback mode. Reconstructs
    #                    per-head K, V from the latent each step.
    mla_mode: str = "absorbed"

    # In-network collectives opt-out. When True (default), dispatcher
    # routes AR/AG over any crossbar tier chain whose every tier
    # declares inc != "none" to the INC primitives (n_alpha collapse +
    # BW-eff doubling for AR). Set False to force software ring/tree
    # fallback — useful for A/B measurements.
    inc_enabled: bool = True

    # ── Collective algorithm selection (decode.md §5.5; §5.7) ──────────
    # Per-collective-class algorithm name. Values are strings drawn from
    # `core/primitives/dispatch.enumerate_options()`'s output for each op.
    # The framework's `optimize_collective_algorithms` resolves the
    # special value "auto" by enumerating cost-model options and picking
    # `min(cost)` (with INC priority when `inc_enabled` AND structurally
    # available — see `core/collective_algo_opt.py` policy).
    #
    # Available options per op (from primitives/dispatch.py):
    #   TP all-reduce (tp_algorithm_*):
    #     "ring"  — bandwidth-optimal at large M; default fallback.
    #     "tree"  — alpha-optimal at small M; literal P=1 binomial tree.
    #     "tree_pipelined" — pipelined variant of tree (lower alpha at large M).
    #     "inc"   — in-network reduction when fabric supports it (NVLS,
    #               IB SHARP, etc.). Selected automatically by the optimizer
    #               when `inc_enabled` and any crossed tier has inc != "none".
    #     "auto"  — resolve via cost model (requires
    #               `optimize_collective_algorithms` before calculator.run()).
    #   EP MoE A2A (ep_algorithm_*):
    #     "ring"  — pairwise direct-send (= "pairwise" in some frameworks).
    #     "inc"   — hardware A2A acceleration (only on tiers with inc=="hw_a2a").
    #     "auto"  — resolve via cost model.
    #   Torus (torus_algorithm):
    #     "ring"  — dim-by-dim ring on torus fabrics.
    #     "swing" — RESERVED (raises NotImplementedError).
    #     "auto"  — resolve via cost model (only "ring" available today).
    #
    # Per-stack JSONs in `database/framework/` typically default to "auto",
    # forcing the optimizer to pick the right algorithm for the
    # (workload × system × group size) cell. `FrameworkSpec.default()`
    # uses "ring" so the calculator runs out-of-box without an optimizer
    # pass (pure roofline).
    tp_algorithm_decode: str = "ring"
    tp_algorithm_prefill: str = "ring"
    ep_algorithm_decode: str = "ring"
    ep_algorithm_prefill: str = "ring"
    torus_algorithm: str = "ring"

    # ── Per-layer collective call counts (decode.md §5.5) ──────────────
    # How many NCCL API calls fire per transformer layer per microbatch.
    # Defaults match the canonical TP-attention transformer:
    #   - 2 TP all-reduces per layer (post-attn + post-FFN). Custom-fused
    #     stacks may reduce this to 1 via sequence-parallel rewrites.
    #   - 2 EP MoE A2A calls per MoE layer (dispatch + combine).
    #   - 1 SP all-gather per layer (with ring SP — only shipped variant
    #     per `collectives/01_collective_algorithms.md §6`).
    # Counts are zeroed dynamically in `_t_kernel_per_microbatch` when the
    # corresponding parallelism axis is 1 (no collective fires).
    n_TP_collectives: int = 2
    n_EP_collectives: int = 2
    n_SP_collectives: int = 1

    # ── Attention dispatch + TP/EP physical overlay (Phase H) ──────────
    # Per-stack defaults for two cross-cutting choices that affect every
    # downstream sharding factor (D_attn, D_kv, D_exp, N_replica). Both
    # are stack-axis decisions captured here so PartitionSpec can stay
    # purely numeric (PP/TP/EP/SP). The cross-spec invariant
    # (tp_ep_layout='co_located' forces attention_mode='dp' AND TP == EP)
    # is enforced by sharding_factors.compose_check(partition, framework).
    #
    # attention_mode (dispatch policy for the attention block):
    #   "tp" — head-shard the K, V matrices across the TP group; per-rank
    #          KV scales as 1/G_TP. Default for non-MLA models.
    #   "dp" — replicate Q / K / V projection weights and shard the batch
    #          across the TP-as-DP-attn group; per-rank KV scales as
    #          1/G_TP via user split. Production-default for MLA models
    #          on Dynamo-orchestrator stacks (TP-attn buys no KV
    #          reduction for MLA — attention.md §3.6).
    #
    # tp_ep_layout (whether TP and EP groups overlay on the same physical GPUs):
    #   "orthogonal" — TP and EP are independent axes. Replica spans
    #                  PP*TP*EP*SP devices. Default for raw-TRT and other
    #                  stacks that don't co-locate.
    #   "co_located" — TP and EP overlay on the same GPU set (TP == EP).
    #                  Replica spans PP*max(TP,EP)*SP devices. Required by
    #                  DeepEP-style scatter-direct dispatch. Production-
    #                  default for Dynamo+TRT/SGLang on MoE models.
    attention_mode: str = "tp"
    tp_ep_layout: str = "orthogonal"

    # ── Comm/compute overlap (decode.md §6.2) ──────────────────────────
    # Fraction ρ_comm of GPU compute time used to hide collective comm
    # (distinct from `kernel_overlap_factor` which is the kernel-launch-vs-GPU overlap):
    #
    #   t_stage = t_local + max(0, t_comm - ρ_comm * t_local)
    #
    # ρ_comm = 0 → strict serialization (conservative roofline default).
    # ρ_comm = 1 → full async overlap (NCCL streams hide comm completely).
    # Production CUDA-Graph stacks (TRT-LLM / SGLang) typically achieve
    # 0.4–0.7 in steady-state decode; eager-mode stacks see ~0.0–0.3.
    comm_overlap_factor: float = 0.0

    @classmethod
    def default(cls) -> "FrameworkSpec":
        """Neutral / no-overhead defaults — matches the legacy TuningSpec
        defaults before the framework split. Algorithms are concrete
        ("ring" everywhere) so the calculator runs out-of-box without
        requiring an `optimize_collective_algorithms` pass — pure roofline.
        Use a per-stack JSON (e.g. `load_framework_from_db("dynamo_trt")`)
        when modeling a production stack; those JSONs default to
        algorithm="auto" so the optimizer selects per cell.
        """
        return cls(name="default")
