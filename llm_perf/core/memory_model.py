
from dataclasses import dataclass, field
from typing import List

from ..specs.model_spec import LlmModelSpec
from ..specs.system_spec import SystemSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.tuner_spec import TuningSpec
from ..utils import GB_TO_BYTES
from .memory_placement import CapacityError, placement_fits, resolve_placement
from .primitives import (
    dense_weight_bytes,
    moe_weight_bytes,
    embedding_bytes,
    kv_bytes_per_seq,
)


@dataclass
class MemoryResults:
    M_theta_device: float
    M_act_device: float
    M_kv_device: float
    M_total_device: float
    fits_in_HBM: bool                       # legacy name; True iff every tier fits
    M_resident_per_tier: List[float] = field(default_factory=list)  # bytes per tier


def compute_memory(
    model: LlmModelSpec,
    system: SystemSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
) -> MemoryResults:
    """Compute per-device static memory footprint (bytes).

    Capacity check is per-tier (sram.md §1.3): the placement layer is
    invoked with the current B_decode; CapacityError → fits_in_HBM=False
    (legacy field name retained). On a single-tier device the per-tier
    breakdown collapses to one entry equal to M_total — matches pre-PR2
    `M_total <= HBM_bytes` exactly.
    """

    H = model.H
    H_kv = model.H_kv()
    b = model.bytes_per_param
    S = tuner.S_decode
    B = tuner.B_decode

    # Parameter memory M_theta_device (dense weights + MoE weights + embedding)
    M_theta_device = (
        dense_weight_bytes(model, partition)
        + moe_weight_bytes(model, partition)
        + embedding_bytes(model, partition)
    )

    # Activation memory M_act_device (one-layer working set, scales with B).
    # Activations stay in registers / shared memory and are not modeled by
    # the placement layer (sram.md §1.3); they count against the device's
    # fastest tier's capacity for the fit check below.
    M_act_device = B * (4 * H + 2 * H_kv) * b

    # KV memory M_kv_device (B sequences, each with context S)
    M_kv_device = B * kv_bytes_per_seq(model, partition, S)

    M_total = M_theta_device + M_act_device + M_kv_device

    # Per-tier capacity check via the placement layer (sram.md §1.3).
    # "auto" placement is permissive on overflow; pinned placement raises
    # CapacityError because the user explicitly chose an impossible mapping.
    tiers = system.device.get_tiers()
    M_resident_per_tier: List[float] = [0.0] * len(tiers)
    fits = True
    try:
        placement = resolve_placement(
            T_theta_device=M_theta_device,
            T_kv_per_request_device=kv_bytes_per_seq(model, partition, S),
            B=max(1, B),
            tiers=tiers,
            placement=tuner.placement,
        )
        for i, (w_i, kv_i) in enumerate(
            zip(placement.weights_per_tier, placement.kv_per_request_per_tier)
        ):
            M_resident_per_tier[i] = w_i + max(1, B) * kv_i
        # Activations land on tier 0 (fastest); add to its residency before
        # the per-tier capacity check.
        M_resident_per_tier[0] += M_act_device
        fits = placement_fits(placement, B=max(1, B), tiers=tiers)
        # Activations might still overflow tier 0 even when weights+KV fit.
        if M_resident_per_tier[0] > tiers[0].capacity_GB * GB_TO_BYTES + 1e-9:
            fits = False
    except CapacityError:
        fits = False

    return MemoryResults(
        M_theta_device=M_theta_device,
        M_act_device=M_act_device,
        M_kv_device=M_kv_device,
        M_total_device=M_total,
        fits_in_HBM=fits,
        M_resident_per_tier=M_resident_per_tier,
    )
