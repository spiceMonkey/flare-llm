"""KV cache footprint and per-step traffic — per-device bytes per sequence.

Single primitive that serves *both* quantities simultaneously: the KV
memory footprint and the KV per-step traffic share the same formula
because, in decode, the full KV cache of every in-flight sequence is
read at every step (each sequence's score / value attention traverses
its entire context). Unlike the weight-θ split in `weight_quantities.py`,
no expert-style masking applies to KV — what's resident is what's read.

    M_kv_per_seq = T_kv_per_seq = (L / PP) · 2 · n_tokens · H_kv · b / (D_kv · SP)

- Factor of 2 covers keys and values.
- (L/PP) is the layer slice owned by this pipeline stage.
- D_kv (notation.md §1) is the per-device head- or sequence-shard divisor
  (excluding SP). Under orthogonal layouts D_kv = TP for both attention
  modes (head-shard under TP-attn, sequence-shard across the TP-as-DP-attn
  group under DP-attn — same byte count). Under co-located layouts
  D_kv = max(TP, EP) (sequence-shard across the entire replica's GPUs).
- SP further shards the sequence dimension on top of D_kv.

Callers multiply by B sequences / B_prefill as needed: memory uses
B × kv_bytes_per_seq for residency; traffic uses B × kv_bytes_per_seq
for the per-step batched read.
"""

from ...specs.framework_spec import FrameworkSpec
from ...specs.model_spec import LlmModelSpec
from ...specs.partition_spec import PartitionSpec
from .sharding_factors import D_kv


def kv_bytes_per_seq(
    model: LlmModelSpec,
    partition: PartitionSpec,
    framework: FrameworkSpec,
    n_tokens: int,
) -> float:
    """Per-device KV bytes for a single sequence of length n_tokens.

    Branches on attention variant and mode (`attention.md §3.4 / §3.6`):
    - GQA / MHA: `2 · H_kv · b` per token per layer, divided by D_kv
      (head-shard under TP-attn, sequence-shard under DP-attn — same
      byte count).
    - MLA + DP-attn: head-shared latent `(d_c + d_qk_rope) · b` per token
      per layer, divided by D_kv (sequence-shard across the DP-attn group
      — same B/G_TP semantics as GQA).
    - MLA + TP-attn: latent is replicated on every rank (not
      head-structured). No D_kv divisor — only SP still applies.
    """
    L = model.L
    b = model.bytes_per_param
    PP = partition.PP
    SP = partition.SP

    if model.mla is not None:
        per_tok_per_layer = model.mla.kv_bytes_per_token_per_layer(b)
        if framework.attention_mode == "tp":
            return (L / PP) * (n_tokens * per_tok_per_layer) / SP
        return (L / PP) * (n_tokens * per_tok_per_layer) / (D_kv(partition, framework) * SP)

    H_kv = model.H_kv()
    return (L / PP) * (2 * n_tokens * H_kv * b) / (D_kv(partition, framework) * SP)
