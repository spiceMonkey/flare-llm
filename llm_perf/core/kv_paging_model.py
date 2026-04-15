
import math
from dataclasses import dataclass
from ..specs.model_spec import LlmModelSpec
from ..specs.system_spec import SystemSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.tuner_spec import TuningSpec
from ..utils import GB_TO_BYTES
from .memory_model import MemoryResults


@dataclass
class KVPagingConfig:
    """KV paging parameters (documentation/modeling/kv.md §2)."""
    block_size: int = 16                # tokens per block (B_block)
    beam_width: int = 1                 # W (1 = greedy, no CoW)
    system_overhead_GB: float = 1.5     # CUDA context + kernel workspace


@dataclass
class KVPagingResults:
    M_block: float              # per-block KV footprint (bytes)
    N_blocks_per_seq: int       # blocks needed per sequence
    phi_avg: float              # average internal fragmentation factor
    M_HBM_KV_avail: float      # HBM available for KV after weights+act+sys (bytes)
    max_sequences: int          # max concurrent sequences at given S
    S_max: float                # max supportable context length for 1 sequence


def compute_kv_paging(
    model: LlmModelSpec,
    system: SystemSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
    memory: MemoryResults,
    paging: KVPagingConfig,
) -> KVPagingResults:
    """KV paging analysis. Doc: documentation/modeling/kv.md §2-6"""

    L = model.L
    H_kv = model.H_kv()
    b = model.bytes_per_param
    S = tuner.S_decode
    PP = partition.PP
    TP = partition.TP
    SP = partition.SP

    block_size = paging.block_size

    # Per-block KV footprint: block_size tokens × 2 (K+V) × H_kv × b × (L/PP) / (TP*SP)
    M_block = block_size * 2 * H_kv * b * (L / PP) / (TP * SP)

    # Blocks per sequence
    N_blocks_per_seq = math.ceil(S / block_size)

    # Average internal fragmentation: last block is half-full on average
    phi_avg = 1.0 + block_size / (2.0 * S) if S > 0 else 1.0

    # Available HBM for KV cache
    HBM_bytes = system.device.hbm_capacity_GB * GB_TO_BYTES
    sys_overhead = paging.system_overhead_GB * GB_TO_BYTES
    M_HBM_KV_avail = HBM_bytes - memory.M_theta_device - memory.M_act_device - sys_overhead
    M_HBM_KV_avail = max(0.0, M_HBM_KV_avail)

    # Max concurrent sequences
    M_per_seq = N_blocks_per_seq * M_block * phi_avg
    max_sequences = int(M_HBM_KV_avail / M_per_seq) if M_per_seq > 0 else 0

    # Max context length for a single sequence (with paging fragmentation)
    M_per_token_kv = 2 * H_kv * b * (L / PP) / (TP * SP)
    S_max = M_HBM_KV_avail / (M_per_token_kv * phi_avg) if M_per_token_kv > 0 else 0.0

    return KVPagingResults(
        M_block=M_block,
        N_blocks_per_seq=N_blocks_per_seq,
        phi_avg=phi_avg,
        M_HBM_KV_avail=M_HBM_KV_avail,
        max_sequences=max_sequences,
        S_max=S_max,
    )
