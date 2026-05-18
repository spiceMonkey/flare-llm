"""Weight (θ) footprint and per-step traffic primitives — per-device bytes.

This module hosts two complementary quantities for the transformer-weights
(θ) bookkeeping. They share most of the same algebra but differ in one
crucial place for MoE — see `moe_weight_traffic_bytes` below.

──────────────────────────────────────────────────────────────────────────
Footprint vs. traffic
──────────────────────────────────────────────────────────────────────────

* **Footprint (`M_θ`)** — maximum resident set in HBM. Must hold *every*
  expert weight even if a given step touches only some of them, because
  the *next* step might select any expert. Used by `compute_memory`.

* **Per-step traffic (`T_θ`)** — bytes actually loaded into the compute
  pipeline this step. For dense weights (attention, dense-FFN) traffic
  equals footprint (every weight is read every step). For MoE expert
  weights traffic depends on which experts the current batch actually
  selects: at small B, only ~B·k_active expert-touch events fire and
  most expert weights sit idle in HBM that step; at large enough B,
  expectation-of-touched-experts → N_exp and traffic converges to the
  full footprint. Used by `compute_traffic`.

The decode roofline gates per-step compute against `T_θ / BW_HBM`, so
using the footprint as the traffic surrogate (the historical default,
exact for dense models) over-counts t_mem for MoE at small B.

──────────────────────────────────────────────────────────────────────────
Splits
──────────────────────────────────────────────────────────────────────────

Dense and MoE layers share the same attention projections (Q,K,V,O) but
differ in the FFN: dense always TP-shards the FFN; MoE shards experts by
D_exp (= TP*EP under orthogonal tp_ep_layout, EP under co-located).

All functions return bytes on one device (D-factor and PP aware). The
abstract divisors D_attn / D_exp / D_emb are resolved from the
PartitionSpec via `sharding_factors`; see `notation.md §1` and `decode.md
§1.4`. Embedding bytes are returned separately because they're outside
the layer loop.
"""

import math

from ...specs.framework_spec import FrameworkSpec
from ...specs.model_spec import LlmModelSpec
from ...specs.partition_spec import PartitionSpec
from .sharding_factors import D_attn, D_emb, D_exp


def _split_layers(model: LlmModelSpec) -> tuple[int, int, int, int]:
    """Return (L_dense, L_moe, N_exp, I_moe) with the same clamp
    logic that core/memory_model.py and core/traffic_model.py apply."""
    L = model.L
    if model.moe is not None:
        L_moe = model.moe.n_moe_layers if model.moe.n_moe_layers else L
        L_dense = L - L_moe
        N_exp = max(1, model.moe.n_experts)
        I_moe = model.moe.I_moe
    else:
        L_moe = 0
        L_dense = L
        N_exp = 1
        I_moe = 0
    return L_dense, L_moe, N_exp, I_moe


def _per_layer_attn_params(model: LlmModelSpec) -> float:
    """Per-layer attention parameter count (cluster-total, before sharding).

    GQA / MHA: `2H² + 2HH_kv` (the §2.3 form, reducing to `4H²` at MHA
    limit). MLA: sum of six matrices from `attention.md §3.3`, computed
    via `MLASpec.per_layer_attn_params`.
    """
    if model.mla is not None:
        return float(model.mla.per_layer_attn_params(model.H, model.n_q))
    H = model.H
    H_kv = model.H_kv()
    return 2 * H**2 + 2 * H * H_kv


def _per_layer_attn_params_per_device(
    model: LlmModelSpec, partition: PartitionSpec, framework: FrameworkSpec
) -> float:
    """Per-device per-layer attention parameter count (after sharding).

    Branches on attention variant and mode:
    - GQA / MHA all modes, MLA + DP-attn:  `_per_layer_attn_params(model) / D_attn`
      (existing convention — `D_attn = TP` under TP-attn, `1` under DP-attn).
    - MLA + TP-attn: down-projections (W_DQ, W_DKV) are not head-structured
      and stay replicated on every rank, while up-projections (W_UQ, W_UK,
      W_UV, W_O) are head-sharded by G_TP. Returns
      `replicated + shardable / G_TP`. See `attention.md §3.6`.
    """
    if model.mla is not None and framework.attention_mode == "tp":
        H, n_q = model.H, model.n_q
        P_repl = model.mla.per_layer_attn_params_replicated(H, n_q)
        P_shrd = model.mla.per_layer_attn_params_shardable(H, n_q)
        return P_repl + P_shrd / partition.TP
    return _per_layer_attn_params(model) / D_attn(partition, framework)


def dense_weight_bytes(
    model: LlmModelSpec, partition: PartitionSpec, framework: FrameworkSpec
) -> float:
    """Per-device weight bytes for the dense-layer slice of the model.

        M_theta_dense = (L_dense / PP) · (attn_per_device + 3HI_dense/TP) · b

    where `attn_per_device` is the per-device per-layer attention parameter
    count from `_per_layer_attn_params_per_device` (handles GQA / MHA's
    P/D_attn and MLA's TP-attn-aware shardable / replicated split).
    Dense FFN always uses /TP regardless of tp_ep_layout.
    """
    L = model.L
    if model.moe is not None:
        L_moe = model.moe.n_moe_layers if model.moe.n_moe_layers else L
        L_dense = L - L_moe
    else:
        L_dense = L

    H = model.H
    I_dense = model.I_dense
    b = model.bytes_per_param
    PP = partition.PP
    d_exp_dense = D_exp(partition, framework, layer_kind="dense")

    P_attn_dev = _per_layer_attn_params_per_device(model, partition, framework)

    return (L_dense / PP) * (
        P_attn_dev + (3 * H * I_dense) / d_exp_dense
    ) * b


def moe_weight_bytes(
    model: LlmModelSpec, partition: PartitionSpec, framework: FrameworkSpec
) -> float:
    """Per-device weight footprint (M_θ) for the MoE-layer slice.

        M_theta_moe = (L_moe / PP) · (attn_per_device + 3HI_moe·N_exp/D_exp) · b

    This is the *maximum resident set* required in HBM — every expert
    weight must be held even if a given step touches only some of them
    (the next step may select any expert). For per-step **traffic**
    (bytes loaded into the compute pipeline this step), use
    `moe_weight_traffic_bytes(...)` which applies the expectation-of-
    touched-experts correction.

    Returns 0.0 for dense-only models.
    """
    if model.moe is None:
        return 0.0

    L = model.L
    L_moe = model.moe.n_moe_layers if model.moe.n_moe_layers else L
    N_exp = max(1, model.moe.n_experts)
    I_moe = model.moe.I_moe

    H = model.H
    b = model.bytes_per_param
    PP = partition.PP
    d_exp_moe = D_exp(partition, framework, layer_kind="moe", n_exp_cap=N_exp)

    P_attn_dev = _per_layer_attn_params_per_device(model, partition, framework)

    return (L_moe / PP) * (
        P_attn_dev + (3 * H * I_moe * N_exp) / d_exp_moe
    ) * b


def moe_weight_traffic_bytes(
    model: LlmModelSpec,
    partition: PartitionSpec,
    framework: FrameworkSpec,
    B: int,
) -> float:
    """Per-device per-step **traffic** (T_θ) for the MoE-layer slice.

    Differs from `moe_weight_bytes` (the footprint) in the expert-FFN
    term: only the experts the current batch actually touches contribute
    to per-step HBM traffic. Attention projections and the router gate
    are read every step (no expert-selection masking) — those terms are
    identical to the footprint formula.

        T_theta_moe(B) =
            (L_moe / PP) · (attn_per_device + 3HI_moe · E[touched/rank]) · b

    where the expected number of unique experts touched on a single rank
    per step, under the *uniform routing* assumption, is:

        E[touched/rank] = N_per · (1 - (1 - 1/N_per) ** t_per)

      with  N_per = N_exp / D_exp_moe      (experts held on this rank)
            t_per = B · k_active / D_exp_moe   (tokens routed to this rank)

    At small B (B·k_active ≪ N_exp) → E[touched/rank] ≈ t_per and
    traffic scales linearly with B. At large B → E[touched/rank] → N_per
    and traffic converges to the footprint formula. The uniform-routing
    assumption is a known source of model inaccuracy: real production
    routers can exhibit hot-spotting (some experts touched more often
    than others) and load-balancing-loss anti-correlation; modeling that
    requires per-deployment routing statistics not generally available.

    Returns 0.0 for dense-only models.
    """
    if model.moe is None:
        return 0.0

    L = model.L
    L_moe = model.moe.n_moe_layers if model.moe.n_moe_layers else L
    N_exp = max(1, model.moe.n_experts)
    I_moe = model.moe.I_moe
    k_active = max(1, model.moe.k_active)

    H = model.H
    b = model.bytes_per_param
    PP = partition.PP
    d_exp_moe = D_exp(partition, framework, layer_kind="moe", n_exp_cap=N_exp)
    B = max(1, int(B))

    P_attn_dev = _per_layer_attn_params_per_device(model, partition, framework)

    # Per-rank expert population and per-rank expert-touch event count.
    N_per_rank = max(1.0, N_exp / d_exp_moe)
    t_per_rank = B * k_active / d_exp_moe

    # Expected unique experts touched on this rank, under uniform routing.
    # Closed form: N · (1 - (1 - 1/N)^t). Use the log-domain stable form
    # for the small-1/N regime; otherwise direct evaluation.
    if N_per_rank > 1.0:
        # log((1 - 1/N)) is well-defined and negative for N > 1.
        prob_miss = math.exp(t_per_rank * math.log(1.0 - 1.0 / N_per_rank))
        E_touched = N_per_rank * (1.0 - prob_miss)
    else:
        # Single expert per rank — always touched if any token routes here.
        E_touched = 1.0 if t_per_rank > 0.0 else 0.0

    # Expert-FFN traffic uses E[touched/rank] in place of N_exp/D_exp.
    return (L_moe / PP) * (
        P_attn_dev + (3 * H * I_moe) * E_touched
    ) * b


def embedding_bytes(model: LlmModelSpec, partition: PartitionSpec) -> float:
    """Per-device embedding (and LM head) bytes.

        M_embed = V · H / D_emb · b

    The embedding is sharded along the vocab dimension by D_emb (= TP under
    all configurations) and replicated across PP stages in the current
    convention — matches memory_model.py.
    """
    return (model.vocab_size * model.H / D_emb(partition)) * model.bytes_per_param
