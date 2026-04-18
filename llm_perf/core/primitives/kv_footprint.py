"""KV cache footprint primitive — per-device bytes for a single sequence.

Given a sequence of `n_tokens` KV entries, returns the per-device bytes
needed to store that sequence's keys and values across all layers:

    M_kv = (L / PP) · 2 · n_tokens · H_kv · b / (TP · SP)

- Factor of 2 covers keys and values.
- (L/PP) is the layer slice owned by this pipeline stage.
- TP and SP both shard the KV dimension (head-parallel + seq-parallel).

This is the single source of truth for KV traffic (decode per-step read,
prefill per-pass write) and KV memory (batched decode residency, paged
block sizing). Callers multiply by B sequences / B_prefill as needed.
"""

from ...specs.model_spec import LlmModelSpec
from ...specs.partition_spec import PartitionSpec


def kv_bytes_per_seq(
    model: LlmModelSpec,
    partition: PartitionSpec,
    n_tokens: int,
) -> float:
    """Per-device KV bytes for a single sequence of length n_tokens."""
    L = model.L
    H_kv = model.H_kv()
    b = model.bytes_per_param
    PP = partition.PP
    TP = partition.TP
    SP = partition.SP
    return (L / PP) * (2 * n_tokens * H_kv * b) / (TP * SP)
