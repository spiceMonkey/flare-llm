"""Two-pool sizing for phase disaggregation (prefill pool + decode pool).

Pure functions over quantities the decode and prefill calculators already
produce. Steady state, closed form, no queueing: a pool is a set of identical
replicas, each of which completes requests at a fixed service rate.

  mu_p = B_pf / t_prefill(S_in, B_pf)        requests/s of one prefill replica
  mu_d = B / (S_out * TPOT(B))               requests/s of one decode replica

With g devices per replica, one request costs g_p / mu_p GPU-seconds of
prefill and g_d / mu_d GPU-seconds of decode, so

  G_p / G_d      = (g_p / mu_p) / (g_d / mu_d)          balanced device ratio
  tokens/s / GPU = S_out / (g_p / mu_p + g_d / mu_d)    over both pools

Neither depends on the request rate. Given a rate and a utilisation cap, the
whole-replica counts follow by rounding up.

`kv_layout` says how a sequence's cache is split across the devices of a
replica; a handoff needs a repack only when the two pools' layouts differ.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from ..specs.framework_spec import FrameworkSpec
from ..specs.model_spec import LlmModelSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.workload_spec import WorkloadSpec
from .primitives import D_kv


@dataclass
class PoolResults:
    # Replica shapes
    g_prefill: int                  # devices per prefill replica
    g_decode: int                   # devices per decode replica

    # Service rates (requests/s per replica)
    mu_prefill: float               # B_pf / t_prefill; inf when prefill takes no time
    mu_decode: float                # B / (S_out * TPOT)

    # GPU time one request costs in each pool (GPU-seconds)
    gpu_s_prefill: float            # g_p / mu_p
    gpu_s_decode: float             # g_d / mu_d

    # Balanced sizing, independent of the request rate
    pool_ratio: float               # G_p / G_d
    throughput_per_gpu: float       # output tokens/s per GPU, counted over BOTH pools
    requests_per_gpu: float         # requests/s per GPU, counted over both pools

    # Whole-replica sizing; None unless the workload carries a request rate
    n_prefill_replicas: Optional[int] = None
    n_decode_replicas: Optional[int] = None
    n_prefill_gpus: Optional[int] = None
    n_decode_gpus: Optional[int] = None
    n_gpus: Optional[int] = None
    utilisation_prefill: Optional[float] = None     # offered rate / pool capacity
    utilisation_decode: Optional[float] = None


def replica_devices(partition: PartitionSpec) -> int:
    """Devices in one replica: PP * TP * EP * SP."""
    return partition.PP * partition.TP * partition.EP * partition.SP


def kv_layout(model: LlmModelSpec, partition: PartitionSpec, framework: FrameworkSpec) -> Tuple[int, int, int]:
    """How one sequence's KV cache is split across a replica: (PP, D_kv_eff, SP).

    PP splits it by layer, SP by sequence position, and the KV divisor by
    head (or by sequence under DP-attention). A latent-attention cache is
    not head-structured, so under TP-attention TP does not split it
    (attention.md §3.6) and the divisor is 1. EP does not enter: an EP group
    holds whole sequences, so it changes where a cache lives and not its shape.
    """
    if model.mla is not None and framework.attention_mode == "tp":
        d_kv_eff = 1
    else:
        d_kv_eff = D_kv(partition, framework)
    return (partition.PP, d_kv_eff, partition.SP)


def _replicas_for(rate: float, mu: float, cap: float) -> int:
    """Smallest whole number of replicas that carries `rate` at utilisation <= cap."""
    if rate <= 0.0 or math.isinf(mu):
        return 0
    need = rate / (mu * cap)
    # Guard the ceiling against a quotient that lands an ulp above an integer.
    return int(math.ceil(need * (1.0 - 1e-12)))


def compute_pools(
    *,
    t_prefill: float,
    B_prefill: int,
    TPOT: float,
    B_decode: int,
    g_prefill: int,
    g_decode: int,
    workload: WorkloadSpec,
) -> PoolResults:
    """Size a prefill pool and a decode pool for one prompt / output shape.

    `t_prefill` is the time one prefill replica takes for a batch of
    `B_prefill` prompts; `TPOT` is the user-observed time per output token of
    a decode replica running `B_decode` sequences.
    """
    if TPOT <= 0.0:
        raise ValueError(f"compute_pools needs TPOT > 0, got {TPOT!r}")
    if t_prefill < 0.0:
        raise ValueError(f"compute_pools needs t_prefill >= 0, got {t_prefill!r}")
    if B_prefill < 1 or B_decode < 1 or g_prefill < 1 or g_decode < 1:
        raise ValueError(
            "compute_pools needs batches and replica sizes >= 1, got "
            f"B_prefill={B_prefill}, B_decode={B_decode}, g_prefill={g_prefill}, g_decode={g_decode}"
        )
    S_out = workload.output_tokens

    mu_p = B_prefill / t_prefill if t_prefill > 0.0 else math.inf
    mu_d = B_decode / (S_out * TPOT)
    gpu_s_p = g_prefill / mu_p          # 0 when prefill takes no time
    gpu_s_d = g_decode / mu_d
    gpu_s = gpu_s_p + gpu_s_d

    res = PoolResults(
        g_prefill=g_prefill,
        g_decode=g_decode,
        mu_prefill=mu_p,
        mu_decode=mu_d,
        gpu_s_prefill=gpu_s_p,
        gpu_s_decode=gpu_s_d,
        pool_ratio=gpu_s_p / gpu_s_d,
        throughput_per_gpu=S_out / gpu_s,
        requests_per_gpu=1.0 / gpu_s,
    )

    rate = workload.request_rate
    if rate is not None:
        cap = workload.utilisation_cap
        n_p = _replicas_for(rate, mu_p, cap)
        n_d = _replicas_for(rate, mu_d, cap)
        res.n_prefill_replicas = n_p
        res.n_decode_replicas = n_d
        res.n_prefill_gpus = n_p * g_prefill
        res.n_decode_gpus = n_d * g_decode
        res.n_gpus = res.n_prefill_gpus + res.n_decode_gpus
        res.utilisation_prefill = rate / (n_p * mu_p) if n_p > 0 else 0.0
        res.utilisation_decode = rate / (n_d * mu_d) if n_d > 0 else 0.0
    return res
