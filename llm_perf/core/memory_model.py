
from dataclasses import dataclass
from ..specs.model_spec import LlmModelSpec
from ..specs.system_spec import SystemSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.tuner_spec import TuningSpec
from ..utils import GB_TO_BYTES
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
    fits_in_HBM: bool


def compute_memory(
    model: LlmModelSpec,
    system: SystemSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
) -> MemoryResults:
    """Compute per-device static memory footprint (bytes)."""

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

    # Activation memory M_act_device (one-layer working set, scales with B)
    M_act_device = B * (4 * H + 2 * H_kv) * b

    # KV memory M_kv_device (B sequences, each with context S)
    M_kv_device = B * kv_bytes_per_seq(model, partition, S)

    M_total = M_theta_device + M_act_device + M_kv_device

    # HBM in bytes
    HBM_bytes = system.device.hbm_capacity_GB * GB_TO_BYTES
    fits = M_total <= HBM_bytes

    return MemoryResults(
        M_theta_device=M_theta_device,
        M_act_device=M_act_device,
        M_kv_device=M_kv_device,
        M_total_device=M_total,
        fits_in_HBM=fits,
    )
