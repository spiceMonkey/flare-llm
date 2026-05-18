# Layout × Attention-Mode: The Four Decode Configurations

**Author:** Yue Lu  
**Date:** May 2026  

**Keywords:**  
LLM inference, decode, tensor parallelism, expert parallelism, attention parallelism, TP-attention, DP-attention, co-located layout, orthogonal layout, MoE sharding, KV cache, DeepSeek-R1, NVL72

---

This doc walks through the four valid `(tp_ep_layout, attention_mode)` combinations for transformer decode deployments, what each does to the workload data (batch, KV) and model state (attention weights, expert weights), and what production deployment each maps to. Each combination is one corner of a 2×2 grid:

```
                              attention_mode
                       TP-attn          DP-attn
                  ┌──────────────┬──────────────┐
                  │              │              │
   orthogonal     │  Corner A    │  Corner C    │
                  │   (default)  │   (rare)     │
   tp_ep_layout   ├──────────────┼──────────────┤
                  │              │              │
   co-located     │  Corner B    │  Corner D    │
                  │  (DSr1 NVL72)│  (DSv3/R1)   │
                  └──────────────┴──────────────┘
```

## The two axes

### Layout: where do TP and EP groups land physically?

The **layout** axis chooses how the tensor-parallel (TP) and expert-parallel (EP) groups map onto physical GPUs within a model replica.

- **Orthogonal layout.** TP and EP groups span *disjoint* GPU sets. Within a replica, every (TP-rank, EP-rank) pair picks a unique GPU. A replica needs `PP × TP × EP × SP` GPUs. This is the canonical Megatron-LM nesting (`DP → PP → EP → TP → SP`, fast axes inner).

- **Co-located layout.** TP and EP groups overlay on the *same* GPU set within a replica. Structural constraint: `TP = EP` (both groups span the same physical ranks, so the count must match). A replica needs only `PP × max(TP, EP) × SP = PP × TP × SP` GPUs — smaller than orthogonal by a factor of `min(TP, EP)`. Each device holds one whole expert (no further TP-sharding of expert weights), so per-device weight footprint is larger by a factor of `TP`.

### Attention mode: how does attention compute shard?

The **attention mode** axis chooses how the attention block uses the TP group.

- **TP-attention.** Each TP rank owns `n_q / TP` query heads. Per-token attention computes the score / value reduction on just its heads, then a TP all-reduce (AR) on the post-output projection re-merges across ranks. KV cache is head-sharded across TP ranks.

- **DP-attention.** Attention weights are *replicated* on every TP rank. The batch is sharded along the sequence / user dimension instead: each rank serves `B / G_TP` users with the full `n_q` heads. KV cache is sequence-sharded (each rank holds the full KV for its share of users). The per-layer attention AR is replaced by a TP all-gather (AG) at the attention → FFN boundary (one AG instead of one AR per layer).

## Walking through each corner

Throughout this section, the running example is a hypothetical model with `n_q = 4` attention heads and 4 experts per layer, deployed at `PP = 1, TP = 4, EP = 4, SP = 1`. Layouts shown as flat 2D GPU grids; arrows indicate per-step collectives.

### Corner A: orthogonal + TP-attention *(the default)*

```
Replica = TP × EP = 4 × 4 = 16 GPUs

                  EP rank →
               0     1     2     3
            ┌─────┬─────┬─────┬─────┐
TP   0  →   │ G0  │ G1  │ G2  │ G3  │   ← TP group (head-shards attention)
rank        ├─────┼─────┼─────┼─────┤
     1  →   │ G4  │ G5  │ G6  │ G7  │
↓           ├─────┼─────┼─────┼─────┤
     2  →   │ G8  │ G9  │ G10 │ G11 │
            ├─────┼─────┼─────┼─────┤
     3  →   │ G12 │ G13 │ G14 │ G15 │
            └─────┴─────┴─────┴─────┘
               ↑     ↑     ↑     ↑
               EP groups (each column = one EP group)
```

- **Batch B (the users / sequences in flight)**: **replicated** across each TP group (the row). Every rank in the row sees all `B` sequences. This is forced by TP-attention: since attention is head-sharded, each rank computes its head's contribution to *every* user's attention output, so every rank needs every user's input.
- **KV cache**: head-sharded along with the heads. Each rank stores K,V for its heads × all `B` users × `S` context tokens. Per-rank `M_kv ∝ B · S · H_kv / TP`.
- **Attention weights**: head-sharded by `TP = 4`. Each rank holds `1/TP` of the attention weights (1 of 4 heads × Q/K/V/O projections).
- **Expert weights**: each expert lives on one EP rank (one column of the grid) but is *further TP-sharded* within that EP rank. So expert 0 lives across `G0, G4, G8, G12` (the EP-rank-0 column), each holding `1/TP` of expert 0's weight matrix. `D_exp = TP · EP = 16` — each device holds `1/16` of any one expert.
- **Per-step collectives**: TP all-reduce per layer (group size 4, across the row) and EP all-to-all per MoE layer (group size 4, across the column). They run on disjoint GPU sets, so they don't share NVLink bandwidth — can overlap.
- **Per-replica GPU count**: `PP · TP · EP · SP = 1 · 4 · 4 · 1 = 16`.
- **Per-device weight footprint**: `(P_attn + 3·H·I_moe·N_exp) / 16` for MoE layers — split across all 16 GPUs.

This is the default that Megatron-LM and most non-DeepSeek deployments use. The 16-GPU per-replica cost is a real penalty if your model has bigger TP and EP figures (DSr1 with TP=EP=8 would need 64 GPUs per replica orthogonal — which is exactly why Corner B exists).

### Corner B: co-located + TP-attention *(DSr1 / NVL72 panel-(b))*

```
Replica = max(TP, EP) = 4 GPUs (TP = EP = 4 forced)

         GPU0           GPU1           GPU2           GPU3
       ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
       │ TP0    │    │ TP1    │    │ TP2    │    │ TP3    │
       │   =    │    │   =    │    │   =    │    │   =    │
       │ EP0    │    │ EP1    │    │ EP2    │    │ EP3    │
       │        │    │        │    │        │    │        │
       │ head 0 │    │ head 1 │    │ head 2 │    │ head 3 │
       │ exp 0  │    │ exp 1  │    │ exp 2  │    │ exp 3  │
       └────────┘    └────────┘    └────────┘    └────────┘
       └─────────── one TP group (= same EP group) ──────────┘
       (attention TP AR + MoE EP A2A both fire on these 4 GPUs)
```

- **Batch B**: **replicated** across the 4-GPU group. Every rank sees all `B` sequences. Same reason as Corner A: TP-attention head-shards attention, so every rank needs every user's input to compute its head.
- **KV cache**: head-sharded. Each rank stores K,V for its head × all `B` users × `S` context tokens. Same per-rank `M_kv` as Corner A.
- **Attention weights**: head-sharded by `TP = 4`. Each rank holds `1/TP` of the attention weights (1 of 4 heads).
- **Expert weights**: each rank holds **one whole expert**. `D_exp = EP = 4` (not `TP·EP` — no further TP-shard since TP and EP are the same physical group). Per-rank expert weight is `TP×` larger than Corner A.
- **Per-step collectives**: TP all-reduce per layer (group size 4) and EP all-to-all per MoE layer (group size 4). Both fire on the **same** 4-GPU group, so they share NVLink bandwidth and **serialize** at the fabric layer (they cannot overlap).
- **Per-replica GPU count**: `PP · max(TP, EP) · SP = 1 · 4 · 1 = 4`. **4× smaller** than Corner A.
- **Per-device weight footprint**: `(P_attn / TP + 3·H·I_moe·N_exp / EP)` for MoE layers — attention is still TP-sharded so weights divide by TP, but experts only divide by EP. So per-device memory is `~TP×` higher than Corner A for the expert term.

**Why is this feasible?** The TP group physically *is* the EP group, but they are conceptually distinct: TP describes "which heads / hidden-dim chunk does this rank own", EP describes "which expert does this rank own". The same 4 GPUs play both roles simultaneously. Attention head-sharding lands on the same ranks that hold the experts; the per-layer TP all-reduce and the per-MoE-layer EP all-to-all both run on the 4-GPU NVLink island.

**This is exactly the DSr1 / NVL72 production deployment for the §5 panel-(b) cut.** TP=EP=8 on a single 8-GPU NVLink island, attention head-sharded across those 8 ranks, each rank holding one whole expert. The InferenceX row reports `decode_tp=8, decode_ep=8, num_decode_gpu=32, decode_dp_attention=False` — 4 replicas of 8 GPUs each.

### Corner C: orthogonal + DP-attention *(rare)*

```
Replica = TP × EP = 16 GPUs (same as Corner A)

           EP rank →
           0    1    2    3
TP   0  ┌────┬────┬────┬────┐
rank    │ G0 │ G1 │ G2 │ G3 │ ← attention REPLICATED on these 4
     1  ├────┼────┼────┼────┤   (each serves B/4 users, full n_q heads)
↓       │ G4 │ G5 │ G6 │ G7 │
     2  ├────┼────┼────┼────┤
        │ G8 │ G9 │G10 │G11 │
     3  ├────┼────┼────┼────┤
        │G12 │G13 │G14 │G15 │
        └────┴────┴────┴────┘
        ↑    ↑    ↑    ↑
        EP groups (unchanged from Corner A)
```

- **Batch B**: **sharded** across each TP group (the row). Each rank holds `B / TP = B/4` sequences for the *full* `n_q = 4` heads. This is the DP-attention flip: instead of replicating B and splitting heads (Corner A), DP-attn replicates heads and splits users.
- **KV cache**: sequence-sharded (one user's K,V sits whole on one rank, instead of being split across heads). Each rank stores K,V for its `B/4` users × full heads × `S` tokens. Per-rank `M_kv` byte count is the same as Corner A (`B · S · H_kv / TP`), but the *what* differs: TP-attn slices each user's KV across ranks, DP-attn keeps each user's KV whole on one rank (simpler paged-attention bookkeeping).
- **Attention weights**: **replicated** across the TP group. Each rank holds the full set of `n_q = 4` heads' Q/K/V/O weights — no `/TP` divisor. Per-rank attention memory is `TP×` larger than Corner A.
- **Expert weights**: same as Corner A (TP × EP sharded, `D_exp = 16`). Each rank holds `1/16` of one expert.
- **Per-step collectives**: TP all-gather per layer at the attn → FFN boundary (each rank produces a `B/4`-shaped activation that needs to be re-merged into a `B`-shaped activation for the TP-sharded FFN). The TP-AR cost from Corner A is replaced by a (cheaper) TP-AG. EP all-to-all per MoE layer unchanged. TP and EP collectives still run on disjoint GPU sets — can overlap.
- **Per-replica GPU count**: `PP · TP · EP · SP = 16`. Same as Corner A.
- **Per-device weight footprint**: attention weights replicated (no `/TP`), so per-device attention memory is `TP×` higher than Corner A. Expert weights identical to Corner A.

**Why feasible?** Same physical layout as Corner A but a different decision about how to use the TP group on the attention block. Replicating attention costs `TP×` more attention-weight memory but saves the per-layer TP AR (replaced by a cheaper AG). Tends to pay off when attention is *small* relative to FFN (MLA-class models, aggressive GQA) — the replication cost is then negligible.

Rare in production because Corner D usually dominates: if you're going to replicate attention, you might as well co-locate TP and EP to also shrink the per-replica GPU count.

### Corner D: co-located + DP-attention *(DSv3 / DSr1 wide-deployment)*

```
Replica = max(TP, EP) = 4 GPUs (TP = EP = 4 forced)

         GPU0           GPU1           GPU2           GPU3
       ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
       │ TP0    │    │ TP1    │    │ TP2    │    │ TP3    │
       │   =    │    │   =    │    │   =    │    │   =    │
       │ EP0    │    │ EP1    │    │ EP2    │    │ EP3    │
       │        │    │        │    │        │    │        │
       │ attn:  │    │ attn:  │    │ attn:  │    │ attn:  │
       │ full   │    │ full   │    │ full   │    │ full   │
       │ heads, │    │ heads, │    │ heads, │    │ heads, │
       │ B/4    │    │ B/4    │    │ B/4    │    │ B/4    │
       │ users  │    │ users  │    │ users  │    │ users  │
       │        │    │        │    │        │    │        │
       │ exp 0  │    │ exp 1  │    │ exp 2  │    │ exp 3  │
       └────────┘    └────────┘    └────────┘    └────────┘
       └─────────── one TP group (= same EP group) ──────────┘
       (attention AG at attn→FFN boundary + MoE EP A2A both on these 4)
```

- **Batch B**: **sharded** across the 4-GPU group. Each rank holds `B / TP = B/4` sequences for the full `n_q = 4` heads. Same DP-attention partitioning as Corner C, just on a smaller (`max(TP, EP) = 4`) physical group.
- **KV cache**: sequence-sharded. Each rank holds the whole KV for its `B/4` users — one user's KV sits entirely on one rank.
- **Attention weights**: **replicated** across the 4-GPU group. Each rank holds the full set of `n_q = 4` heads' weights — `TP×` larger per-rank attention memory than Corner B.
- **Expert weights**: each rank holds one whole expert. `D_exp = EP = 4`. Same as Corner B.
- **Per-step collectives**: TP all-gather per layer at attn → FFN boundary (group size 4) + EP all-to-all per MoE layer (group size 4). Both on the same physical 4-GPU group; serialize.
- **Per-replica GPU count**: `4`. Same as Corner B.
- **Per-device weight footprint**: attention weights replicated (`TP×` more attention memory than Corner B) + one whole expert per rank.

**Why feasible?** Co-location's TP=EP overlay combined with DP-attention's batch-sharding. Attention runs locally on each rank with no per-layer TP AR (just a single AG at attn→FFN); each rank serves its share of users end-to-end. The MoE A2A still fires across the same 4 GPUs on every MoE layer.

This is the canonical DSv3 / R1 production deployment for `TP=EP=16` or `TP=EP=32`. MLA's "flat" KV (latent vector, not head-structured) makes attention replication essentially free.

## Cross-corner comparison

| Property | A: ortho + TP-attn | B: colo + TP-attn | C: ortho + DP-attn | D: colo + DP-attn |
|---|---|---|---|---|
| **Batch B per TP rank** | `B` (replicated) | `B` (replicated) | `B/TP` (sharded) | `B/TP` (sharded) |
| **Heads per rank** | `n_q/TP` (head-shard) | `n_q/TP` (head-shard) | `n_q` (full) | `n_q` (full) |
| `D_attn` | `TP` | `TP` | `1` | `1` |
| `D_exp` (MoE) | `TP · EP` | `EP` | `TP · EP` | `EP` |
| `D_kv` | `TP` (head) | `TP` (head) | `TP` (seq) | `max(TP,EP)` (seq) |
| Per-replica GPUs | `PP·TP·EP·SP` | `PP·max(TP,EP)·SP` | `PP·TP·EP·SP` | `PP·max(TP,EP)·SP` |
| Per-device attn weight | `1/TP` | `1/TP` | `1` (replicated) | `1` (replicated) |
| Per-device expert weight | `1/(TP·EP)` | `1/EP` | `1/(TP·EP)` | `1/EP` |
| Attn collective | TP AR per layer | TP AR per layer | TP AG per layer | TP AG per layer |
| MoE collective | EP A2A per MoE layer | EP A2A per MoE layer | EP A2A per MoE layer | EP A2A per MoE layer |
| TP-EP collective overlap | parallel (disjoint GPUs) | serialize (shared GPUs) | parallel | serialize |

Three things to note:

1. **Layout axis controls per-replica GPU count, not collective shape.** Corners A↔B and C↔D differ only in whether TP and EP overlay on the same physical ranks. The collective cost per call is identical; what changes is which fabric tier each rank's collective lands on (orthogonal can span tiers; co-located stays on one tier by construction), and whether TP and EP collectives can run in parallel.

2. **Attention mode controls how batch B and heads are split across the TP group.** Corners A↔C and B↔D differ in one binary choice on the attention block: TP-attn replicates B across the TP group and head-shards (each rank computes its head for every user), while DP-attn shards B across the TP group and replicates heads (each rank computes every head for its sub-batch of `B/TP` users). The downstream consequences — per-layer collective primitive (AR vs AG), attention weight footprint (sharded vs replicated), KV cache layout (per-head slice of every user vs whole KV of some users) — all follow from this one choice. The MoE side is untouched by this axis.

3. **Per-device expert weight is the load-bearing memory difference.** Orthogonal corners (A, C) divide expert weights by `TP·EP`; co-located corners (B, D) divide by `EP` only. For DSr1 at `TP=EP=8`, the per-device expert weight is `8×` larger under co-location. The trade is per-replica GPU count: co-location uses `8×` fewer GPUs per replica. So co-location is a memory-for-GPU-count trade.

## When to pick which

This is the §6.3-style operational guidance, distilled:

- **Corner A (orthogonal + TP-attn) — the default.** Pick when the per-device memory of `(P_attn + 3·H·I_moe·N_exp) / (TP·EP)` fits in HBM and the per-replica GPU count `PP·TP·EP·SP` is acceptable. Dense or small-MoE models with MHA/GQA attention at small `G_TP` land here naturally.

- **Corner B (co-located + TP-attn) — DSr1 / NVL72 panel-(b).** Pick when (i) the model is MoE-dominated and per-device expert memory under `D_exp = EP` fits in HBM, (ii) attention is head-structured enough to benefit from sharding (`n_q ≥ TP`, GQA `n_kv ≥ TP`, or MHA), and (iii) keeping the world inside one NVLink island matters (small-`G_TP` regime where collective `α` cost dominates). DSr1 at `TP=EP=8` on a single 8-GPU NVLink island is the canonical fit.

- **Corner C (orthogonal + DP-attn) — rare.** Pick when MLA-class attention makes head-sharding pointless (latent KV is not head-structured) but you don't want to co-locate (e.g., the orthogonal layout's TP and EP groups are needed for some other reason — multi-tenant scheduling, partitioning constraints). The framework supports it; production rarely lands here because Corner D dominates whenever you've already decided to replicate attention.

- **Corner D (co-located + DP-attn) — DSv3 / DSr1 wide.** Pick when (i) MLA-class attention makes replication cheap, (ii) MoE-dominant compute benefits from co-location's smaller per-replica GPU count, and (iii) you want all collectives on a single NVLink island for `α`-cost savings. DSv3 / R1 production decode at `TP=EP=16` or `TP=EP=32` is the canonical fit.

## References

- `notation.md §1` — canonical (layout, attention_mode) lookup table.
- `decode.md §1.4 / §2 / §3.5 / §6.1 / §6.3` — per-section parallel tables + operational guidance on when each corner pays.
- `prefill.md §3.x` — same lookup applied to prefill compute sharding.
- `core/primitives/sharding_factors.py` — `compose_check` invariant and the per-component effective sharding factors (`D_attn`, `D_exp`, `D_kv`, `D_emb`, `G_TP`, `G_EP`, `N_replica`).
- `benchmark/validate/dsr1_gb200_dynamo_trt.py:run_colo_tp_attn` — the production-validator cut for Corner B (DSr1 / GB200 panel-(b)).
