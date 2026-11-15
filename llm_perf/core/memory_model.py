
from dataclasses import dataclass, field, replace
from typing import List

from ..specs.framework_spec import FrameworkSpec
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
    group_batch,
)


@dataclass
class MemoryResults:
    M_theta_device: float
    M_act_device: float
    M_kv_token: float       # per-device, per-sequence KV memory (one resident sequence's KV history, all stage layers); matches decode.md §1.3 M_{KV,token}
    M_kv_device: float      # per-device KV memory aggregated over the sequences its group holds: (B / D_batch) * M_kv_token; matches decode.md §1.3 M_{KV,device}
    M_total_device: float
    fits_in_HBM: bool                       # legacy name; True iff every tier fits
    M_resident_per_tier: List[float] = field(default_factory=list)  # bytes per tier


def compute_memory(
    model: LlmModelSpec,
    system: SystemSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
    framework: FrameworkSpec,
) -> MemoryResults:
    """Compute per-device static memory footprint (bytes).

    Capacity check is per-tier (sram.md §1.3): the placement layer is
    invoked with the current B_decode; CapacityError → fits_in_HBM=False
    (legacy field name retained). On a single-tier device the per-tier
    breakdown collapses to one entry equal to M_total — matches pre-PR2
    `M_total <= HBM_bytes` exactly.
    """

    H = model.H
    b = model.act_bytes()   # only the activation working set uses `b` below
    S = tuner.S_decode
    # Sequences resident on one device: the replica's batch divided across
    # the EP device groups under the orthogonal layout (== B_decode otherwise).
    B = group_batch(tuner.B_decode, partition, framework)

    # Parameter memory M_theta_device (dense weights + MoE weights + embedding)
    M_theta_device = (
        dense_weight_bytes(model, partition, framework)
        + moe_weight_bytes(model, partition, framework)
        + embedding_bytes(model, partition)
    )

    # Activation memory M_act_device (one-layer working set, scales with B).
    # Activations stay in registers / shared memory and are not modeled by
    # the placement layer (sram.md §1.3); they count against the device's
    # fastest tier's capacity for the fit check below.
    #
    # Branches on attention variant. Both forms are rough proxies for the
    # actual one-layer attention working set; the dominant downstream
    # terms (M_theta_device, M_kv_device) are accurate, and activation
    # memory is typically small in absolute terms (MB-scale at B=1).
    # - GQA / MHA: 4H residual / FFN intermediate + 2 H_kv for the K, V
    #   per-head buffers (matches the historical convention).
    # - MLA: 4H residual / FFN intermediate
    #        + n_q · d_qk for the materialized per-head Q (always built
    #          at runtime via W_UQ)
    #        + n_q · d_v for the per-head attention output (fed to W_O)
    #        + (d_c + d_qk_rope) for the on-die copy of c_KV used during
    #          attention (read-only working buffer; not the cache itself).
    if model.mla is not None:
        mla = model.mla
        n_q = model.n_q
        d_qk = mla.d_qk_nope + mla.d_qk_rope
        M_act_device = B * (
            4 * H
            + n_q * d_qk
            + n_q * mla.d_v
            + (mla.d_c + mla.d_qk_rope)
        ) * b
    else:
        H_kv = model.H_kv()
        M_act_device = B * (4 * H + 2 * H_kv) * b

    # KV memory: per-sequence building block (decode.md §1.3 M_{KV,token})
    # and the device-aggregate for B resident sequences.
    M_kv_token = kv_bytes_per_seq(model, partition, framework, S)
    M_kv_device = B * M_kv_token

    M_total = M_theta_device + M_act_device + M_kv_device

    # Per-tier capacity check via the placement layer (sram.md §1.3).
    # "auto" placement is permissive on overflow; pinned placement raises
    # CapacityError because the user explicitly chose an impossible mapping.
    tiers = system.device.get_tiers()
    M_resident_per_tier: List[float] = [0.0] * len(tiers)
    fits = True
    # Activations live on the fastest tier and are not movable, so that
    # tier's capacity is reduced by them before weights and KV are placed.
    # Placing weights and KV first and adding activations afterwards would
    # report a multi-tier device as not fitting whenever its fast tier is
    # full, even though the placement could simply have left room.
    act_GB = M_act_device / GB_TO_BYTES
    tiers_for_placement = [
        replace(tiers[0], capacity_GB=max(0.0, tiers[0].capacity_GB - act_GB)),
        *tiers[1:],
    ]
    try:
        placement = resolve_placement(
            T_theta_device=M_theta_device,
            T_kv_per_request_device=M_kv_token,
            B=max(1, B),
            tiers=tiers_for_placement,
            placement=tuner.placement,
        )
        for i, (w_i, kv_i) in enumerate(
            zip(placement.weights_per_tier, placement.kv_per_request_per_tier)
        ):
            M_resident_per_tier[i] = w_i + max(1, B) * kv_i
        # Activations land on tier 0 (fastest); add to its residency before
        # the per-tier capacity check.
        M_resident_per_tier[0] += M_act_device
        fits = placement_fits(placement, B=max(1, B), tiers=tiers_for_placement)
        # Activations might still overflow tier 0 even when weights+KV fit.
        if M_resident_per_tier[0] > tiers[0].capacity_GB * GB_TO_BYTES + 1e-9:
            fits = False
    except CapacityError:
        fits = False

    return MemoryResults(
        M_theta_device=M_theta_device,
        M_act_device=M_act_device,
        M_kv_token=M_kv_token,
        M_kv_device=M_kv_device,
        M_total_device=M_total,
        fits_in_HBM=fits,
        M_resident_per_tier=M_resident_per_tier,
    )
