"""Per-data-class memory tier placement (sram.md §1.3, §2.1).

Standalone pure functions that:
  1. `resolve_placement(...)` — split per-device weight bytes (T_θ) and
     per-request KV bytes (T_KV) across the device's memory tier list,
     respecting the per-tier capacity constraint
       T_θ,i + B · T_KV,i ≤ C_i
     under the policy declared by `MemoryPlacementSpec` ("auto" greedy
     fastest-first, or operator-pinned to a named tier).
  2. `t_mem_from_placement(...)` — assemble the multi-tier roofline memory
     time per sram.md §2.1 (dropped-α form):
       t_mem(B) = Σ_i (T_θ,i + B · T_KV,i) / BW_eff,i

Single-tier reduction: when the device exposes one tier (the legacy shim
path from PR1), greedy "auto"/"auto" puts everything on tier 0 and
`t_mem_from_placement` collapses to the legacy `T_step / BW_mem`
expression — bitwise identical to pre-PR2 behavior.

Activations (T_act) are not modeled here — `decode.md §2.2` drops them per
the FlashAttention-applied assumption (sram.md §1.1 closing paragraph).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..specs.system_spec import MemoryTierSpec
from ..specs.tuner_spec import MemoryPlacementSpec
from ..utils import GB_TO_BYTES


class CapacityError(ValueError):
    """Raised when a placement does not fit in the available device tiers.

    Carries the data class that overflowed ("weights" or "kv") and the
    bytes still pending after walking all tiers, so callers can produce a
    diagnostic ("decode at B=N exceeds device capacity by X GB").
    """

    def __init__(self, data_class: str, overflow_bytes: float, message: str):
        super().__init__(message)
        self.data_class = data_class
        self.overflow_bytes = overflow_bytes


@dataclass
class PlacementResult:
    """Output of `resolve_placement`.

    Each list is indexed by tier (parallel to `device.get_tiers()`).
      - weights_per_tier[i] = T_θ,i (bytes resident on tier i)
      - kv_per_request_per_tier[i] = T_KV,i (per-request bytes on tier i)
    Conservation: sum(weights_per_tier) = T_θ,device,
                  sum(kv_per_request_per_tier) = T_KV,device.
    """

    weights_per_tier: List[float]
    kv_per_request_per_tier: List[float]


def _find_tier_index(tiers: List[MemoryTierSpec], name: str) -> int:
    for idx, t in enumerate(tiers):
        if t.name == name:
            return idx
    available = [t.name for t in tiers]
    raise ValueError(
        f"MemoryPlacementSpec pins to tier {name!r} but device exposes "
        f"tiers {available!r}"
    )


def resolve_placement(
    T_theta_device: float,
    T_kv_per_request_device: float,
    B: int,
    tiers: List[MemoryTierSpec],
    placement: MemoryPlacementSpec,
) -> PlacementResult:
    """Split T_θ and T_KV across the device's memory tiers per `placement`.

    Args:
      T_theta_device: per-device weight bytes (sum across all tiers).
      T_kv_per_request_device: per-device KV bytes for one request.
      B: number of in-flight requests sharing the device tiers.
      tiers: device's ordered tier list (fastest first).
      placement: per-data-class policy ("auto" greedy or "<name>" pin).

    Returns:
      PlacementResult with per-tier byte breakdown.

    Raises:
      CapacityError: if the placement does not fit in the available tiers.
      ValueError: if a pin name does not match any tier.
    """
    if not tiers:
        raise ValueError("resolve_placement: empty tier list")
    if placement.auto_priority not in ("weights", "kv"):
        raise ValueError(
            f"MemoryPlacementSpec.auto_priority must be 'weights' or 'kv', "
            f"got {placement.auto_priority!r}"
        )
    n = len(tiers)
    weights_per_tier: List[float] = [0.0] * n
    kv_per_tier: List[float] = [0.0] * n
    capacities = [t.capacity_GB * GB_TO_BYTES for t in tiers]
    remaining = list(capacities)

    # ── Pin step: explicit named tiers reserve their full byte budget first ──
    weights_left = T_theta_device
    kv_per_req_left = T_kv_per_request_device

    if placement.weights_tier != "auto":
        idx = _find_tier_index(tiers, placement.weights_tier)
        if T_theta_device > remaining[idx]:
            raise CapacityError(
                "weights",
                T_theta_device - remaining[idx],
                f"weights pinned to tier {placement.weights_tier!r} ({remaining[idx] / GB_TO_BYTES:.3f} GB) "
                f"do not fit ({T_theta_device / GB_TO_BYTES:.3f} GB required)",
            )
        weights_per_tier[idx] = T_theta_device
        remaining[idx] -= T_theta_device
        weights_left = 0.0

    if placement.kv_tier != "auto":
        idx = _find_tier_index(tiers, placement.kv_tier)
        kv_total = B * T_kv_per_request_device
        if kv_total > remaining[idx]:
            raise CapacityError(
                "kv",
                kv_total - remaining[idx],
                f"KV cache (B={B}) pinned to tier {placement.kv_tier!r} "
                f"({remaining[idx] / GB_TO_BYTES:.3f} GB) does not fit "
                f"({kv_total / GB_TO_BYTES:.3f} GB required)",
            )
        kv_per_tier[idx] = T_kv_per_request_device
        remaining[idx] -= kv_total
        kv_per_req_left = 0.0

    # ── Greedy step: "auto" data classes flow into remaining capacity ──
    # Order is set by `placement.auto_priority`: "weights" (default) fills
    # weights into the fastest tier first, then KV gets remaining capacity;
    # "kv" flips the order, useful for KV-bound workloads where SRAM-resident
    # KV matters more than SRAM-resident weights. On overflow, "auto"
    # placement is permissive — leftover bytes accumulate on the last
    # (slowest) tier without raising. This preserves the legacy behavior
    # that t_mem was always computable (the unfit state was surfaced via
    # memory_model.fits_in_HBM, not by aborting the latency path).
    # `placement_fits()` below is the canonical fit predicate.
    def _fill_weights():
        nonlocal weights_left
        for i in range(n):
            if weights_left <= 0:
                break
            take = min(remaining[i], weights_left)
            weights_per_tier[i] += take
            remaining[i] -= take
            weights_left -= take
        if weights_left > 0:
            weights_per_tier[-1] += weights_left
            remaining[-1] -= weights_left
            weights_left = 0.0

    def _fill_kv():
        nonlocal kv_per_req_left
        if B <= 0:
            return
        kv_total_left = B * kv_per_req_left
        for i in range(n):
            if kv_total_left <= 0:
                break
            take = min(remaining[i], kv_total_left)
            kv_per_tier[i] += take / B
            remaining[i] -= take
            kv_total_left -= take
        if kv_total_left > 0:
            kv_per_tier[-1] += kv_total_left / B
            remaining[-1] -= kv_total_left
        kv_per_req_left = 0.0

    if placement.auto_priority == "weights":
        if weights_left > 0:
            _fill_weights()
        if kv_per_req_left > 0:
            _fill_kv()
    else:  # "kv" — already validated above
        if kv_per_req_left > 0:
            _fill_kv()
        if weights_left > 0:
            _fill_weights()

    return PlacementResult(
        weights_per_tier=weights_per_tier,
        kv_per_request_per_tier=kv_per_tier,
    )


def placement_fits(
    placement: PlacementResult,
    B: int,
    tiers: List[MemoryTierSpec],
) -> bool:
    """True iff every tier's resident bytes (T_θ,i + B · T_KV,i) fit in C_i.

    Companion to `resolve_placement`. The "auto" greedy placement is
    permissive on overflow (bytes accumulate on the last tier); use this
    predicate to detect the unfit state without aborting latency math.
    """
    for w_i, kv_i, tier in zip(
        placement.weights_per_tier,
        placement.kv_per_request_per_tier,
        tiers,
    ):
        if w_i + max(1, B) * kv_i > tier.capacity_GB * GB_TO_BYTES + 1e-9:
            return False
    return True


def t_mem_from_placement(
    placement: PlacementResult,
    B: int,
    tiers: List[MemoryTierSpec],
) -> float:
    """Multi-tier decode roofline memory time per sram.md §2.1 (dropped-α form):

        t_mem(B) = Σ_i (T_θ,i + B · T_KV,i) / BW_eff,i

    where BW_eff,i = BW_i · η_β,i (sram.md §1.2). The α_i first-byte term
    is dropped at the device level (sram.md §2.1 magnitude argument).

    Single-tier reduction: when len(tiers) = 1 with η_β = 1.0 (PR1 legacy
    shim), this returns (T_θ + B · T_KV) / BW exactly — matches pre-PR2
    `t_mem = T_step / BW_mem` to floating-point equality.
    """
    total = 0.0
    for w_i, kv_i, tier in zip(
        placement.weights_per_tier,
        placement.kv_per_request_per_tier,
        tiers,
    ):
        bw_eff = tier.bandwidth_GBps * tier.eta_beta * GB_TO_BYTES
        if bw_eff <= 0:
            continue
        total += (w_i + B * kv_i) / bw_eff
    return total
