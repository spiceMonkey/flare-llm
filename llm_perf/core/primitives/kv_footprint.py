"""KV cache footprint primitive — per-device bytes for a single sequence.

Given a sequence of `n_tokens` KV entries, returns the per-device bytes
needed to store that sequence's keys and values across all layers:

    M_kv = (L / PP) · 2 · n_tokens · H_kv · b / (D_kv · SP)

- Factor of 2 covers keys and values.
- (L/PP) is the layer slice owned by this pipeline stage.
- D_kv (notation.md §1) is the per-device head- or sequence-shard divisor
  (excluding SP). Under orthogonal layouts D_kv = TP for both attention
  modes (head-shard under TP-attn, sequence-shard across the TP-as-DP-attn
  group under DP-attn — same byte count). Under co-located layouts
  D_kv = max(TP, EP) (sequence-shard across the entire replica's GPUs).
- SP further shards the sequence dimension on top of D_kv.

This is the single source of truth for KV traffic (decode per-step read,
prefill per-pass write) and KV memory (batched decode residency, paged
block sizing). Callers multiply by B sequences / B_prefill as needed.
"""

from ...specs.model_spec import LlmModelSpec
from ...specs.partition_spec import PartitionSpec
from .sharding_factors import D_kv


def kv_bytes_per_seq(
    model: LlmModelSpec,
    partition: PartitionSpec,
    n_tokens: int,
) -> float:
    """Per-device KV bytes for a single sequence of length n_tokens.

    For MLA models (`model.mla` set), the per-token-per-layer base is the
    head-shared latent `(d_c + d_qk_rope) * b` rather than the GQA form
    `2 * H_kv * b`. The latent is not head-structured, so the head-shard
    divisor `D_kv` does not apply under TP-attention; for Phase 1 we
    retain the divisor for sharding-symbol consistency, accepting a
    per-rank under-count under TP-attention until §3.7 sharding lands
    proper MLA-aware D_kv handling. Cluster-total KV is correct.
    """
    L = model.L
    b = model.bytes_per_param
    PP = partition.PP
    SP = partition.SP

    if model.mla is not None:
        per_tok_per_layer = model.mla.kv_bytes_per_token_per_layer(b)
        return (L / PP) * (n_tokens * per_tok_per_layer) / (D_kv(partition) * SP)

    H_kv = model.H_kv()
    return (L / PP) * (2 * n_tokens * H_kv * b) / (D_kv(partition) * SP)
