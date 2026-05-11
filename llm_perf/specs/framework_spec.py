"""FrameworkSpec — SW-stack-specific runtime behavior.

Captures the host-side overhead model and execution-mode choices that
depend on the production serving stack rather than the workload, model,
or hardware. Splits cleanly out of TuningSpec along the natural axis:

    TuningSpec  = workload knobs (S, B, chunk_size, collective algos,
                  placement, speculation, n_*_collectives, overlap_factor)
    FrameworkSpec = stack knobs (c_serving, kernel_launch_us, kernels_per_*,
                  sw_overlap_factor, moe_a2a_pattern, mla_mode, inc_enabled)
    PartitionSpec = sharding (PP, TP, EP, SP, attention_mode, layout)
    SystemSpec  = hardware (DeviceSpec, FabricSpec, MemoryTierSpec)
    ModelSpec   = architecture (LlmModelSpec, MoESpec, MLASpec)

The five specs are orthogonal axes of an inference deployment. The
calculator composes them: any (model, system, partition, tuner, framework)
tuple is a runnable deployment configuration.

Pre-canned framework JSONs live in `database/framework/` and correspond
1:1 to the production stack identifiers used in InferenceX measurement
metadata: `dynamo-trt`, `dynamo-sglang`, `dynamo-vllm`, `trt`, `vllm`,
`sglang`. See `documentation/modeling/decode.md §7.1, §7.2` for the
underlying analytical model and per-stack `c_serving` ranges.
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
    # Per-sequence per-step host work (decode.md §7.2):
    # PagedAttention block-table gather, continuous-batching scheduler,
    # per-sequence sampling glue, token-append + KV bookkeeping. Linear
    # in B, on the critical path (no overlap), outside the PP bubble.
    #
    #   t_serving = c_serving_per_seq_us * B * 1e-6     [seconds]
    #
    # Stack-dependent ranges (decode.md §7.2):
    # - C++/CUDA-graph + orchestrator (Dynamo+TRT): 5-22 µs/seq
    # - Mixed orchestrator + Python (Dynamo+SGLang): 25-50 µs/seq
    # - Raw C++ runtime (raw TRT-LLM): 50-100 µs/seq
    # - Aggressively fused C++: ~10 µs/seq lower bound
    # - Python-heavy (vLLM, SGLang eager): 30-60 µs/seq
    c_serving_per_seq_us: float = 0.0

    # Per-kernel dispatch budget (decode.md §7.1).
    #   t_stage_sw = tau_launch * (k_compute + k_collective + k_pp_hop)
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

    # Fraction of t_stage_sw hidden behind GPU compute (decode.md §6.3).
    # 1.0 = full async overlap (CUDA-Graph replay steady-state on
    # TensorRT-LLM / vLLM / SGLang where the CPU runs ahead). Eager-mode
    # PyTorch sees ~0.3-0.6 because the Python interpreter breaks the
    # CPU-runs-ahead invariant. Caveat: sw_overlap_factor only modulates
    # the *hideable* portion; t_stage_sw remains a hard floor when it
    # exceeds t_stage_GPU regardless of this knob.
    sw_overlap_factor: float = 1.0

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

    @classmethod
    def default(cls) -> "FrameworkSpec":
        """Neutral / no-overhead defaults — matches the legacy TuningSpec
        defaults before the framework split. Use this as a fallback when
        no production stack is being modeled (pure roofline).
        """
        return cls(name="default")
