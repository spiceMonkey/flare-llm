"""Multi-head Latent Attention (MLA) FLOP primitives.

Splits MLA's per-layer attention compute into two phase-independent pieces
matching the breakdown in `attention.md §3.7`:

    F_attn,MLA(S) = F_proj_MLA + F_score_value_MLA(S)

where `F_proj_MLA` is the S-independent projection cost (down-projections,
up-projections, output projection) and `F_score_value_MLA(S)` is the
S-scaling score / value compute. Both branch on the execution mode set
on the tuner (`tuner.mla_mode`):

    "materialized" — reconstruct per-head K, V from the latent each step,
                     run standard MHA over (n_q heads, d_qk_nope / d_v).
    "absorbed"     — fold W_UK / W_UV into Q / O at compile time, run
                     attention entirely in the d_c-dimensional latent space.

The materialized form pays a fixed per-step W_UK / W_UV reconstruction
cost on the new token; the absorbed form skips that but pays a larger
per-past-token score / value cost (in d_c space rather than d_qk_nope /
d_v). Production frameworks default to absorbed at S ≳ 1K.

Sharding follows the same convention as GQA / MHA in
`primitives/linear_flops.py` and `decode_model.compute_flops`:
- Projection FLOPs are sharded by `D_attn` (TP under TP-attn,
  1 under DP-attn — replicated weights, but the per-token cost is
  matched by the corresponding `B / TP` user split downstream).
- Score / value FLOPs are sharded by `D_kv` (TP under both modes,
  representing head-shard and user-shard respectively — same byte count).

Callers compose `mla_proj_flops_per_layer_per_device` into the linear-flops
primitive and `mla_score_value_flops_per_layer_per_device` into the
attention-FLOPs term of `decode_model.compute_flops` /
`prefill_model.compute_prefill_flops`.
"""

from ...specs.framework_spec import FrameworkSpec
from ...specs.model_spec import LlmModelSpec, MLASpec
from ...specs.partition_spec import PartitionSpec
from .sharding_factors import D_attn, D_kv


def mla_proj_flops_per_layer_per_device(
    model: LlmModelSpec,
    partition: PartitionSpec,
    framework: FrameworkSpec,
    mla_mode: str,
) -> float:
    """Per-device per-layer MLA projection FLOPs (S-independent).

    Returns the projection FLOPs that replace the GQA / MHA
    `(4H² + 4HH_kv) / D_attn` term in `linear_flops_per_token`. Includes
    W_DQ, W_UQ, W_DKV, the output projection (W_O materialized or
    absorbed-W_O), and — in materialized mode only — the per-step
    W_UK / W_UV reconstruction on the new token.
    """
    assert model.mla is not None, "mla_proj_flops_per_layer_per_device requires model.mla"
    mla: MLASpec = model.mla
    H = model.H
    n_q = model.n_q
    d_qk = mla.d_qk_nope + mla.d_qk_rope
    d_attn = D_attn(partition, framework)

    # Down-projections are replicated under TP-attn (no D_attn divisor),
    # but for symbol consistency with the GQA path we match its convention:
    # divide everything by D_attn (= TP under TP-attn). See note below.
    F_W_DQ = 2 * H * mla.d_q_c
    F_W_DKV = 2 * H * (mla.d_c + mla.d_qk_rope)

    # Up-projections + W_O (mode-dependent)
    F_W_UQ = 2 * mla.d_q_c * n_q * d_qk
    if mla_mode == "materialized":
        F_W_UK_new = 2 * n_q * mla.d_c * mla.d_qk_nope
        F_W_UV_new = 2 * n_q * mla.d_c * mla.d_v
        F_W_O = 2 * n_q * mla.d_v * H
        F_shardable = F_W_UQ + F_W_UK_new + F_W_UV_new + F_W_O
    elif mla_mode == "absorbed":
        # W_UK and W_UV folded into Q / O at compile time; the absorbed
        # output projection runs in d_c space.
        F_W_O_absorbed = 2 * n_q * mla.d_c * H
        F_shardable = F_W_UQ + F_W_O_absorbed
    else:
        raise ValueError(f"mla_mode must be 'materialized' or 'absorbed', got {mla_mode!r}")

    # MLA + TP-attn: down-projections replicated (no D_attn divisor),
    # up-projections + W_O head-sharded by G_TP. MLA + DP-attn: D_attn = 1
    # so all-divided-by-1 = full per-token (matches GQA convention).
    if framework.attention_mode == "tp":
        return (F_W_DQ + F_W_DKV) + F_shardable / partition.TP
    return (F_W_DQ + F_W_DKV + F_shardable) / d_attn


def mla_score_value_flops_per_layer_per_device(
    model: LlmModelSpec,
    partition: PartitionSpec,
    framework: FrameworkSpec,
    S: float,
    mla_mode: str,
) -> float:
    """Per-device per-layer MLA score / value FLOPs (S-scaling).

    Returns the score + value compute that replaces the GQA / MHA
    `4 · S · H / D_kv` term in `decode_model.compute_flops` (and the
    `4 · S² · H / D_kv` analog in `prefill_model.compute_prefill_flops`,
    which the caller obtains by passing `S = S_input` and multiplying by
    `S_input` for the per-pass form).
    """
    assert model.mla is not None, "mla_score_value_flops_per_layer_per_device requires model.mla"
    mla: MLASpec = model.mla
    n_q = model.n_q
    d_qk = mla.d_qk_nope + mla.d_qk_rope
    d_kv = D_kv(partition, framework)

    if mla_mode == "materialized":
        # Score: Q · K^T over (n_q, d_qk), Value: softmax · V over (n_q, d_v)
        F_full = 2 * S * n_q * d_qk + 2 * S * n_q * mla.d_v
    elif mla_mode == "absorbed":
        # Score in latent: Q' · c_KV over (n_q, d_c + d_qk_rope)
        # Value in latent: softmax · c_KV over (n_q, d_c)
        F_full = 2 * S * n_q * (mla.d_c + mla.d_qk_rope) + 2 * S * n_q * mla.d_c
    else:
        raise ValueError(f"mla_mode must be 'materialized' or 'absorbed', got {mla_mode!r}")

    return F_full / d_kv
