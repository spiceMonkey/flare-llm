
import math
from dataclasses import dataclass
from ..specs.framework_spec import FrameworkSpec
from ..specs.model_spec import LlmModelSpec
from ..specs.system_spec import SystemSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.tuner_spec import TuningSpec
from ..utils import GB_TO_BYTES
from .memory_model import MemoryResults
from .primitives import D_batch, D_kv, group_batch, kv_batch


@dataclass
class KVPagingConfig:
    """KV paging parameters (kv.md §2)."""
    block_size: int = 16                # tokens per block (B_block)
    beam_width: int = 1                 # W: NOT modelled; any value prices as 1 (a warning says so)
    system_overhead_GB: float = 1.5     # CUDA context + kernel workspace

    def __post_init__(self) -> None:
        if self.beam_width != 1:
            import warnings
            warnings.warn(
                "KVPagingConfig.beam_width is not modelled: copy-on-write sharing between beams has no term, "
                f"so beam_width={self.beam_width} gives the same result as 1",
                UserWarning, stacklevel=2,
            )


@dataclass
class KVPagingResults:
    M_block: float              # per-block KV footprint (bytes)
    N_blocks_per_seq: int       # blocks needed per sequence
    phi_avg: float              # average internal fragmentation factor
    M_HBM_KV_avail: float      # HBM available for KV after weights+act+sys (bytes)
    max_sequences: int          # max concurrent sequences at given S
    S_max: float                # max supportable context length for 1 sequence
    M_act_per_seq: float = 0.0  # activation working set one resident sequence adds on a device (bytes)
    inflight_sequences: float = 0.0   # caches in flight to the replica that were charged (sequences)


def compute_kv_paging(
    model: LlmModelSpec,
    system: SystemSpec,
    partition: PartitionSpec,
    tuner: TuningSpec,
    framework: FrameworkSpec,
    memory: MemoryResults,
    paging: KVPagingConfig,
    *,
    inflight_sequences: float = 0.0,
) -> KVPagingResults:
    """KV paging analysis. Doc: kv.md §2-6

    `inflight_sequences` is the number of caches in flight to this replica
    during a disaggregated handoff (`E2EResults.inflight_sequences`): each
    already holds its cache in GPU memory but is not decoded yet, so it takes
    the room of a resident sequence's cache without its activations.
    """
    if inflight_sequences < 0.0:
        raise ValueError(f"compute_kv_paging needs inflight_sequences >= 0, got {inflight_sequences!r}")

    L = model.L
    b = model.kv_bytes()
    S = tuner.S_decode
    PP = partition.PP
    SP = partition.SP

    # KV head/seq divisor (notation.md §1) — TP under orthogonal tp_ep_layout,
    # max(TP, EP) under co-located. Composes with SP for per-device KV bytes.
    d_kv = D_kv(partition, framework)

    block_size = paging.block_size

    # Per-token-per-layer KV cache base. GQA / MHA: 2·H_kv·b (separate K
    # and V). MLA: head-shared latent (d_c + d_qk_rope)·b — see
    # attention.md §3.4. Under MLA + TP-attn the latent is not
    # head-structured, so D_kv does not apply (attention.md §3.6).
    if model.mla is not None:
        kv_base_per_tok_per_layer = model.mla.kv_bytes_per_token_per_layer(b)
        d_kv_eff = 1 if framework.attention_mode == "tp" else d_kv
    else:
        kv_base_per_tok_per_layer = 2 * model.H_kv() * b
        d_kv_eff = 1 if (not model.kv_head_split and framework.attention_mode == "tp") else d_kv

    # Under DP-attention a rank holds whole sequences: a block, and a whole
    # sequence, sits on one rank and not as 1/G_TP shares on all of them.
    # `whole` is G_TP there and 1 under TP-attention.
    whole = kv_batch(1.0, partition, framework)

    # Per-block KV footprint: block_size tokens × base × (L/PP) / (d_kv_eff*SP)
    M_block = block_size * kv_base_per_tok_per_layer * (L / PP) / (d_kv_eff * SP) * whole

    # Blocks per sequence
    N_blocks_per_seq = math.ceil(S / block_size)

    # Average internal fragmentation: last block is half-full on average
    phi_avg = 1.0 + block_size / (2.0 * S) if S > 0 else 1.0

    # Available HBM for KV cache
    HBM_bytes = system.device.hbm_capacity_GB * GB_TO_BYTES
    sys_overhead = paging.system_overhead_GB * GB_TO_BYTES
    M_HBM_KV_avail = HBM_bytes - memory.M_theta_device - memory.M_act_device - sys_overhead
    M_HBM_KV_avail = max(0.0, M_HBM_KV_avail)

    # Max concurrent sequences. One sequence's cache on a device is its S
    # tokens times the average fragmentation factor (kv.md §3.1, §6): phi_avg
    # already carries the partly filled last block, so the token count is S
    # itself and not the whole-block count N_blocks * block_size. (Rounding up
    # to whole blocks AND applying phi_avg counted that block twice whenever S
    # was not a multiple of the block size; at a multiple the two agree.)
    M_per_token_kv = kv_base_per_tok_per_layer * (L / PP) / (d_kv_eff * SP)
    M_per_seq = S * M_per_token_kv * phi_avg
    # Every resident sequence brings its cache AND its activation working
    # set. `memory.M_act_device` is the working set of the tuner's own batch
    # on one device, so one sequence's share is that over the sequences the
    # device holds. (M_HBM_KV_avail above keeps its meaning: what is left for
    # the cache at the tuner's batch.)
    n_groups = D_batch(partition, framework)
    B_dev = group_batch(tuner.B_decode, partition, framework)
    M_act_per_seq = memory.M_act_device / B_dev if B_dev > 0 else 0.0
    M_room = max(0.0, HBM_bytes - memory.M_theta_device - sys_overhead)
    # Caches in flight are spread over the device groups like the batch is.
    M_inflight_dev = (inflight_sequences / n_groups) * M_per_seq
    # Sequences one device can hold, times the device groups the batch is
    # divided across (EP under the orthogonal layout, else 1) — the replica's
    # concurrent-sequence capacity, in the same units as B_decode.
    n_dev = max(0, int((M_room - M_inflight_dev) / (M_per_seq + M_act_per_seq))) if M_per_seq > 0 else 0
    if whole != 1.0:
        # DP-attention: every rank of the group holds the same whole number
        # of sequences, so the group's count is a multiple of G_TP.
        n_dev = (n_dev // int(whole)) * int(whole)
    max_sequences = n_dev * n_groups

    # Max context length for a single sequence (with paging fragmentation)
    S_max = M_HBM_KV_avail / (M_per_token_kv * whole * phi_avg) if M_per_token_kv > 0 else 0.0

    return KVPagingResults(
        M_block=M_block,
        N_blocks_per_seq=N_blocks_per_seq,
        phi_avg=phi_avg,
        M_HBM_KV_avail=M_HBM_KV_avail,
        max_sequences=max_sequences,
        S_max=S_max,
        M_act_per_seq=M_act_per_seq,
        inflight_sequences=inflight_sequences,
    )
