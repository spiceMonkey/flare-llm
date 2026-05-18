# Collective Algorithms: A Network Topology-Free Introduction

**Author:** Yue Lu  
**Date:** April 2026  

Collective operations are the vocabulary of distributed GPU workloads — broadcast, reduce, all-reduce, all-gather, reduce-scatter, all-to-all, point-to-point. They show up in LLM training, LLM inference, HPC simulation, and any other multi-rank compute that needs to synchronize or redistribute state. This note defines each primitive, walks through the dominant algorithms on small concrete examples — ring and binomial tree for broadcast and reduce, ring and double binary tree (DBT) for all-reduce, ring for all-gather / reduce-scatter, pairwise direct-send for all-to-all — and maps every primitive to the parallelism pattern — tensor parallelism (TP), expert parallelism (EP), sequence parallelism (SP), pipeline parallelism (PP) — that uses it. A generic pipelining derivation (Appendix C) covers the mechanism by which all log-depth schedules collapse their bandwidth terms from $L \cdot M/\mathrm{BW}$ to $M/\mathrm{BW}$. No physical topology yet — just algorithms as they'd run on an abstract fully-connected fabric. Topology enters in the companion note `02_topology_mapping.md`.

# Table of Contents

1. [The α-β cost model](#1-the-α-β-cost-model)
2. [The seven primitives](#2-the-seven-primitives)
3. [Broadcast](#3-broadcast)
   - 3.1 [Ring BC](#31-ring-bc)
   - 3.2 [Binomial tree BC](#32-binomial-tree-bc)
   - 3.3 [Comparison and practical adoption](#33-comparison-and-practical-adoption)
4. [Reduce](#4-reduce)
   - 4.1 [Ring Reduce](#41-ring-reduce)
   - 4.2 [Binomial tree Reduce](#42-binomial-tree-reduce)
   - 4.3 [Comparison and practical adoption](#43-comparison-and-practical-adoption)
5. [All-reduce](#5-all-reduce)
   - 5.1 [Ring AR](#51-ring-ar)
   - 5.2 [Double binary tree (DBT) AR](#52-double-binary-tree-dbt-ar)
   - 5.3 [Comparison and practical adoption](#53-comparison-and-practical-adoption)
6. [All-gather / reduce-scatter](#6-all-gather--reduce-scatter)
7. [All-to-all](#7-all-to-all)
   - 7.1 [Ring A2A](#71-ring-a2a)
   - 7.2 [Pairwise direct-send A2A](#72-pairwise-direct-send-a2a)
   - 7.3 [Comparison and practical adoption](#73-comparison-and-practical-adoption)
8. [Point-to-point hop](#8-point-to-point-hop)
9. [Mapping primitives to DP, TP, EP, SP, PP](#9-mapping-primitives-to-dp-tp-ep-sp-pp)
10. [Appendix A: Parallel Aggregated Trees (PAT) — scale-out AG / RS](#appendix-a-parallel-aggregated-trees-pat--scale-out-ag--rs)
11. [Appendix B: Non-mainline AR / AG / RS / A2A variants](#appendix-b-non-mainline-ar--ag--rs--a2a-variants)
    - B.1 [Simple recursive-doubling AR](#b1-simple-recursive-doubling-ar)
    - B.2 [Rabenseifner halving-doubling AR](#b2-rabenseifner-halving-doubling-ar)
    - B.3 [Why neither AR variant is shipped](#b3-why-neither-ar-variant-is-shipped)
    - B.4 [Recursive-doubling AG / recursive-halving RS](#b4-recursive-doubling-ag--recursive-halving-rs)
    - B.5 [Bruck A2A](#b5-bruck-a2a)
12. [Appendix C: Asymptotic form of linear schedules (via pipelining)](#appendix-c-asymptotic-form-of-linear-schedules-via-pipelining)
13. [Appendix D: algbw vs busbw — the NCCL-tests vocabulary](#appendix-d-algbw-vs-busbw--the-nccl-tests-vocabulary)
14. [Further reading](#further-reading)

---

## 1. The α-β cost model

Every collective algorithm on every topology costs time in the same two-part shape. The time to move a message of $M$ bytes between two ranks on a link of bandwidth $\mathrm{BW}$ is modeled as

$$t = \alpha + \frac{M}{\mathrm{BW}}$$

- $\alpha$ (**per-hop latency**): fixed setup + switch traversal + propagation, measured in seconds. Does not depend on message size.
- $M / \mathrm{BW}$ (**bandwidth term**): time spent pushing bytes through the wire.

A collective-level schedule summarizes its $k$ sequential hops and its per-rank total bytes moved into two dimensionless counts:

$$t = n_\alpha \cdot \alpha + n_\beta \cdot \frac{M}{\mathrm{BW}}$$

- $n_\alpha$ (**latency count**) — the number of sequential synchronization hops. Ring and tree schedules on $N$ ranks chain $O(N)$ and $O(\log_2 N)$ hops respectively; in-network collective (INC) primitives that host the reduction inside a switch collapse $n_\alpha$ to $2$ regardless of $N$.
- $n_\beta$ (**bandwidth count**) — the multiplicative factor on the per-rank payload $M/\mathrm{BW}$, aggregating how many link-traversal-equivalents of the full $M$-byte message each rank's critical path moves. Good algorithms keep $n_\beta$ close to the lower bound set by the primitive: $(N-1)/N \to 1$ for one-way redistribution (shipping one rank's share across the cut is unavoidable) and $2(N-1)/N \to 2$ for all-reduce (every byte of the result must visit an endpoint link twice — once in, once out). Switch-hosted INC further collapses the all-reduce $n_\beta$ to 1 by moving the reduction into the fabric (`04_in_network_collectives.md §1.4`).

The primitives themselves (all-reduce, all-gather, reduce-scatter, all-to-all, and the others) are defined in §2; INC is covered in `04_in_network_collectives.md`. The per-primitive and per-algorithm values of $(n_\alpha, n_\beta)$ are worked out in §§3–8.

Writing both costs in $(n_\alpha, n_\beta)$ form makes latency-term and bandwidth-term efficiency directly comparable across topologies and across in-network vs software schedules. "Cost" means wall-clock time under this model; contention and congestion (which inflate $\alpha$ and deflate $\mathrm{BW}$) are covered in `05_contention_and_congestion.md`.

---

## 2. The seven primitives

Seven collectives cover essentially all multi-rank communication in distributed GPU workloads (LLM training and inference, HPC simulation, distributed gradient descent, etc.). The first six are defined on a group of $N$ ranks each holding data of size $M$; the last is point-to-point. In the table below, $V$ denotes a payload of size $M$ bytes (treated as a vector for reduction purposes — e.g., a gradient tensor, an activation shard, a key-value (KV) cache chunk); $V_i$ is the copy held by rank $i$, and $\sum_i V_i$ is the element-wise sum across ranks. For the primitives whose name contains "reduce", the reduction operator is commutative-associative (sum, max, min, bit-op); the pure-movement primitives (broadcast, all-gather, all-to-all) just redistribute data and don't combine it.

| Primitive | What each rank starts with | What each rank ends with |
|---|---|---|
| **Broadcast** | rank 0 holds $V$; others hold nothing | all ranks hold $V$ |
| **Reduce** | rank $i$ holds $V_i$ | rank 0 holds $\sum_i V_i$ (others unchanged) |
| **All-reduce (AR)** | rank $i$ holds $V_i$ | all ranks hold $\sum_i V_i$ |
| **Reduce-scatter (RS)** | rank $i$ holds length-$M$ vector $V_i$ | rank $i$ holds one chunk of size $M/N$ equal to $\sum_j V_j[\mathrm{chunk}_i]$ |
| **All-gather (AG)** | rank $i$ holds one chunk of size $M/N$ | all ranks hold the concatenation of all $N$ chunks |
| **All-to-all (A2A)** | rank $i$ holds $N$ chunks of size $M/N$; chunk $j$ is destined for rank $j$ | rank $i$ holds $N$ chunks of size $M/N$; chunk $j$ was sent by rank $j$ (specifically, rank $j$'s original "chunk $i$") |
| **P2P (send/recv)** | rank $a$ holds $V$ | rank $b$ holds $V$ |

*A2A indexing convention:* chunks in the **send** buffer are indexed by **destination** (rank $i$'s chunk $j$ is what rank $i$ wants to give to rank $j$); chunks in the **receive** buffer are indexed by **sender** (rank $i$'s received chunk $j$ is what rank $j$ gave it, which was rank $j$'s original chunk $i$). This matches the MPI `MPI_Alltoall` layout.

Three useful identities:

- **AR ≡ RS + AG** (*logically*). The end state of AR is identical to running RS then AG: RS leaves each rank with one fully-reduced $M/N$-chunk, and AG then redistributes those chunks so everyone has all $N$ of them. This is a semantic equivalence, not a prescription — some AR algorithms (bandwidth-optimal **ring** AR [Patarasuk-Yuan 2009], dim-decomposed ring AR on torus/mesh) do execute as a literal RS phase followed by an AG phase, but others (**recursive halving-doubling** / Rabenseifner, **recursive doubling**, tree AR, in-network AR) never materialize the intermediate RS end-state as a distinct phase. The identity is still useful for reasoning about cost lower bounds (§5) regardless of implementation.
- **AR ≡ Reduce + Broadcast** (*also logically, but strictly worse in cost*). Reduce aggregates every $V_i$ into rank 0's $\sum_i V_i$; broadcast then replicates it to all ranks. The end state matches AR's, but composing the two primitives ships the full reduced $M$-byte result across the fabric *twice* — each phase carries an $M/\mathrm{BW}$-floor bandwidth term — whereas tree-based and ring AR fuse the two directions so each byte traverses an endpoint link only twice total (matching the $2(N-1)/N$ lower bound). This identity is useful as a conceptual baseline; no production AR implementation uses it. §4.3 works out the cost gap.
- **A2A is a permutation, not a reduction**. No summation happens; every rank ends up holding data that belongs elsewhere. Mixture-of-Experts (MoE) expert dispatch is exactly this pattern (plus a reverse A2A for the combine).

Sections 3–7 walk through the six collective primitives in the table order: §3 Broadcast, §4 Reduce, §5 AR, §6 AG / RS, §7 A2A. Broadcast and Reduce lead because their tree-based schedules are the structural building blocks for tree AR in §5, so putting them up front lets the DBT construction cite primitives already in scope. AR, AG / RS, and A2A are the reduction / redistribution primitives that dominate time in any collective-heavy workload (distributed data parallelism (DDP) / fully-sharded data parallelism (FSDP) training, LLM inference, MoE routing, HPC all-reduce). Each of §§5–7 follows the same template: a ring-based derivation (the NCCL large-$M$ default for all three primitives), the tree-based variant that NCCL actually ships alongside ring when one exists (DBT for AR), and a comparison that explains the NCCL selection rule between them. Variants that appear in the literature and in other runtimes — simple recursive-doubling AR, Rabenseifner halving-doubling AR, recursive-doubling AG / recursive-halving RS, Parallel Aggregated Trees (PAT), Bruck A2A — are not in the NCCL shipping menu but remain useful reference points; their full step-by-step derivations live in Appendix B, with Appendix A giving the same treatment to the one shipping non-ring AG / RS algorithm — Parallel Aggregated Trees (PAT, NCCL 2.23+, scale-out). Cross-references to each appendix subsection appear inline where the main-text comparisons call for them. §8 covers point-to-point, §9 maps primitives to parallelism axes, Appendix C derives the generic pipelining mechanism that the §3 / §4 / §5 collectives invoke to collapse their bandwidth terms, and Appendix D defines the algbw / busbw reporting convention used in NCCL-tests.

---

## 3. Broadcast

Broadcast (BC) is the simplest one-to-many primitive: rank 0 starts holding $V$ (size $M$ bytes), and all $N$ ranks must end holding a copy of $V$. It shows up in parameter loading from rank 0 at initialization, in metadata and embedding distribution, in the down-multicast half of tree AR (§5.2), and in any dataflow that needs to replicate a single reference payload across a group. Two algorithms cover the practical cost landscape — ring (§3.1, bandwidth-optimal via pipelining, linear-$\alpha$) and binomial tree (§3.2, latency-optimal log-$\alpha$; with chunked pipelining the BW term collapses to the same $M/\mathrm{BW}$ floor as ring — see [Appendix C](#appendix-c-asymptotic-form-of-linear-schedules-via-pipelining) for the generic derivation). §3.3 explains which one NCCL `ncclBroadcast` dispatches for a given $(N, M)$ regime and points to `04_in_network_collectives.md §1.1` for the switch-multicast primitive that collapses $n_\alpha$ to 2 regardless of $N$.

### 3.1 Ring BC

Arrange ranks in a ring $R_0 \to R_1 \to \ldots \to R_{N-1} \to R_0$ with $R_0$ as the source. BC has a single source, so data flows one way from $R_0$ down the ring and the wrap-around edge $R_{N-1} \to R_0$ is idle — the schedule uses the ring's $N-1$ forward edges as a chain. The same ring topology (same physical wiring, same per-rank in/out neighbor set) is reused by ring AR in §5.1 and ring AG / RS in §6, where every rank sends to every other and the wrap-around carries real traffic.

**Ring for $N = 4$** (arrows are the forward direction used by BC):

```
      R0 ───→ R1 ───→ R2 ───→ R3
```

Chunk $V$ into $P$ equal pieces of size $M/P$ and stream them down the ring: once a rank has received chunk $s$ from its left neighbor, it forwards chunk $s$ to its right neighbor while concurrently receiving chunk $s+1$ from its left. Every forward link carries one chunk per step in steady state — a software analogue of physical wormhole forwarding. We walk through $N = 4$ with $P = N$ (one chunk per rank, matching ring AR's convention in §5.1 and giving the cleanest symmetric fill/drain), so $V = [V_0, V_1, V_2, V_3]$ with each $V_s$ of size $M/N$ initially held by $R_0$; the other three ranks start empty. The $P = N$ choice is illustrative — the asymptotic cost reached for larger $P$ per Appendix C is discussed after the walkthrough.

**Initial state.**

```
R0: [V0  V1  V2  V3]
R1: [ —              ]
R2: [ —              ]
R3: [ —              ]
```

**After step 1** — $R_0$ ships $V_0$ to $R_1$. Pipeline is filling; only one link is active.

```
Step 1: R0 → R1 (V0)

R0: [V0  V1  V2  V3]
R1: [V0              ]   ← received V0
R2: [ —              ]
R3: [ —              ]
```

**After step 2** — two links active concurrently: $R_0 \to R_1$ streams $V_1$ while $R_1 \to R_2$ forwards $V_0$.

```
Step 2: R0 → R1 (V1); R1 → R2 (V0) concurrently

R0: [V0  V1  V2  V3]
R1: [V0  V1          ]
R2: [V0              ]
R3: [ —              ]
```

**After step 3** — pipeline fully populated: three links carry $V_2, V_1, V_0$ in the same step. $V_0$ reaches the tail rank $R_3$.

```
Step 3: R0 → R1 (V2); R1 → R2 (V1); R2 → R3 (V0) concurrently

R0: [V0  V1  V2  V3]
R1: [V0  V1  V2      ]
R2: [V0  V1          ]
R3: [V0              ]   ← first chunk arrives at tail
```

**After step 4** — last chunk enters the pipeline; $R_1$ is now complete.

```
Step 4: R0 → R1 (V3); R1 → R2 (V2); R2 → R3 (V1) concurrently

R0: [V0  V1  V2  V3]
R1: [V0  V1  V2  V3]   ← complete
R2: [V0  V1  V2      ]
R3: [V0  V1          ]
```

**After step 5** — $R_0$ is idle; two links still active for drain.

```
Step 5: R1 → R2 (V3); R2 → R3 (V2) concurrently

R0: [V0  V1  V2  V3]
R1: [V0  V1  V2  V3]
R2: [V0  V1  V2  V3]   ← complete
R3: [V0  V1  V2      ]
```

**After step 6** — final drain.

```
Step 6: R2 → R3 (V3)

R0: [V0  V1  V2  V3]
R1: [V0  V1  V2  V3]
R2: [V0  V1  V2  V3]
R3: [V0  V1  V2  V3]   ← complete — BC done
```

Total: $N - 1 + P - 1 = 6$ steps (fill $N-1$, then drain $P-1$ after the last chunk enters at $R_0$). Each step ships $M/P$ bytes over its active link at cost $\alpha + M/(P\\,\mathrm{BW})$:

$$t_{\mathrm{ring\\,BC}} = (N + P - 2)\left(\alpha + \frac{M}{P\\,\mathrm{BW}}\right)$$

At optimal $P^* = \sqrt{(N{-}2)M/(\alpha \mathrm{BW})}$, in the large-$M$ limit, the cost approaches the **asymptotic form**:

$$t_{\mathrm{ring\\,BC}} \approx (N - 1)\\,\alpha + \frac{M}{\mathrm{BW}}$$

The "$\approx$" hides an $O(\sqrt{M})$ correction of order $2\sqrt{(N{-}2)\alpha M/\mathrm{BW}}$ between the two floors — $(N{-}1)\alpha$ is the minimum the latency term can hit ($P \to 1$), $M/\mathrm{BW}$ is the minimum the BW term can hit ($P \to \infty$), and neither limit is physical; $P^*$ lives between them. The correction is $O(\sqrt{M})$ while the dominant $M/\mathrm{BW}$ term is $O(M)$, so it vanishes *relative to* the asymptotic floor as $M \to \infty$ but is nonzero at any finite $M$. See [Appendix C](#appendix-c-asymptotic-form-of-linear-schedules-via-pipelining) for the full derivation.

$n_\alpha = N - 1$, $n_\beta = 1$ — bandwidth-optimal (matches the asymptotic tree floor in §3.2) but linear $\alpha$ in $N$. Ring BC loses to pipelined tree on the $\alpha$ side but keeps the same BW floor, so it wins on simple or low-radix fabrics where tree-schedule bookkeeping is expensive relative to the $\alpha$ advantage — or on ring-wired physical topologies where the logical ring maps directly onto the copper (same logic that keeps ring AR in NCCL's shipping menu for large-$M$; see §5.3).

### 3.2 Binomial tree BC

The binomial tree BC **doubles the set of ranks holding $V$** at every step. At step $k$, every rank already holding $V$ pairs with one rank that doesn't and ships $V$ over the link; concurrent pairs run in parallel. After $\lceil \log_2 N \rceil$ steps every rank holds $V$. We walk through $N = 4$ with $V = [v_0, v_1, v_2, v_3]$ (size $M$) initially held by $R_0$; the other three ranks start empty.

**Tree structure for $N = 4$.** The send pattern below traces out the binomial tree $B_2$ rooted at $R_0$:

```
      R0            (depth 0, source)
      │  │
      ▼  ▼
    R1   R2         (depth 1)
    │
    ▼
    R3              (depth 2)
```

$R_0$'s two children are the roots of $B_1$ (on $\\{R_1, R_3\\}$) and $B_0$ (on $\\{R_2\\}$) — the recursive construction of $B_k$ hangs a $B_{k-1}, B_{k-2}, \ldots, B_0$ off the root in descending size. Level-counts $(1, 2, 1)$ are row $2$ of Pascal's triangle — $\binom{2}{0}, \binom{2}{1}, \binom{2}{2}$ — which is what "binomial" refers to: the number of nodes at depth $d$ in $B_k$ is $\binom{k}{d}$. Depth is $\lceil \log_2 N \rceil = 2$, setting $n_\alpha$.

**Initial state.**

```
R0: [v0  v1  v2  v3]
R1: [ —             ]
R2: [ —             ]
R3: [ —             ]

holding-set: {R0}
```

**After step 1** — $R_0 \to R_1$. The holding set doubles from $\\{R_0\\}$ to $\\{R_0, R_1\\}$.

```
Step 1: R0 sends V to R1

R0: [v0  v1  v2  v3]
R1: [v0  v1  v2  v3]   ← received from R0
R2: [ —             ]
R3: [ —             ]

holding-set: {R0, R1}
```

**After step 2** — $R_0 \to R_2$ and $R_1 \to R_3$, concurrently (two disjoint pairs, both sides of the holding set each sending to one new partner). The holding set doubles to all 4 ranks.

```
Step 2: R0 → R2 and R1 → R3 concurrently

R0: [v0  v1  v2  v3]
R1: [v0  v1  v2  v3]
R2: [v0  v1  v2  v3]   ← received from R0
R3: [v0  v1  v2  v3]   ← received from R1

holding-set: {R0, R1, R2, R3} — BC complete
```

BC completes in $\lceil \log_2 4 \rceil = 2$ sequential steps; at general $N$ in $\lceil \log_2 N \rceil$ steps. Each step moves the full $M$-byte payload over one link per active pair, so the per-step cost is $\alpha + M/\mathrm{BW}$ and the sequential steps sum:

$$t_{\mathrm{bin\\,BC}} = \lceil \log_2 N \rceil \cdot \alpha + \lceil \log_2 N \rceil \cdot \frac{M}{\mathrm{BW}}$$

$n_\alpha = \lceil \log_2 N \rceil$ (latency-optimal — a lower bound for BC over a binary-combinable fabric), $n_\beta = \lceil \log_2 N \rceil$ (bandwidth-suboptimal — same $\log N$ coefficient weakness as simple recursive-doubling AR in [App. B.1](#b1-simple-recursive-doubling-ar)).

**Asymptotic form (bandwidth-bound regime; pipelined implementation).** Chunk $V$ into $P$ sub-segments of size $M/P$ and stream them through the tree — each internal node forwards chunk $s$ to its children while receiving chunk $s+1$ from its parent — turning every tree edge into a conveyor. Substituting $L = \lceil \log_2 N \rceil$ (tree depth) into the Appendix C master formula and optimizing at $P^* = \sqrt{(\lceil \log_2 N \rceil - 1)M/(\alpha \mathrm{BW})}$ in the large-$M$ limit gives the **asymptotic floor**:

$$t_{\mathrm{bin\\,BC}} \approx \lceil \log_2 N \rceil \cdot \alpha + \frac{M}{\mathrm{BW}}$$

The "$\approx$" hides an $O(\sqrt{M})$ interference correction $2\sqrt{(\lceil \log_2 N \rceil - 1)\alpha M/\mathrm{BW}}$ between the two floors — the $\alpha$-count picks up a $(P^*{-}1)\alpha$ surcharge, and the BW term carries a fill/drain residue of the same magnitude — both vanishing *relative to $M/\mathrm{BW}$* as $M \to \infty$ but nonzero at any finite $M$. This is the form NCCL's tree-path `ncclBroadcast` runs for bulk $M$ — simultaneously latency- and bandwidth-optimal for BC on unbounded-port fabrics. Full derivation of the $(L+P-1)$ slot count, $P^*$, the correction, and the port-budget caveat that licenses the collapse on real fabrics are all in [Appendix C](#appendix-c-asymptotic-form-of-linear-schedules-via-pipelining); the binomial tree satisfies the port budget (busiest rank holds ≤ 3 concurrent tree-edge partners) so the collapse applies.

### 3.3 Comparison and practical adoption

Three software implementations (binomial tree $P=1$, binomial tree $P^*$, ring $P^*$) plus a hardware-multicast path sit on the table. Before presenting the combined view, the hardware option needs a brief introduction — the two software algorithms were fully specified in §§3.1–3.2, but the multicast primitive has not yet appeared in this chapter.

**Hardware multicast primitive.** Switched fabrics expose a switch-multicast primitive (InfiniBand Quantum Scalable Hierarchical Aggregation and Reduction Protocol (SHARP), NVSwitch Gen3+ / NVLink SHARP (NVLS), many PCIe switches[^pcie-mcast]) that replicates a single $M$-byte payload across all destination ports in one switch-local operation. The source rank pushes $V$ upstream to the switch (1 hop); the switch crossbar then drives $V$ out all $N-1$ destination ports concurrently in one multicast fan-out (1 more hop). Two switch-crossing hops total, **independent of $N$**:

$$t_{\mathrm{INC,\\,BC}} \\;\approx\\; 2\\,\alpha_\mathrm{switch} + \frac{M}{\mathrm{BW}}$$

$\alpha_\mathrm{switch}$ is the switch cut-through latency — $\sim 100$–$200$ ns for NVSwitch Gen3/4, $\sim 250$ ns for Broadcom Tomahawk Ultra — typically well below the $\alpha \approx 0.5$–$1\\,\mu$s that software BC accumulates across $\lceil \log_2 N \rceil$ or $N-1$ endpoint-driven hops.

**All four options side by side.** The two software algorithms each have a finite-$P$ and an asymptotic form; which form wins is governed by the $M$-regime, since the choice of $P$ (non-pipelined $P = 1$ vs asymptotic $P = P^*$) is dictated by whether the α term or the BW term dominates. Switch multicast is an orthogonal hardware-dependent alternative and applies across regimes where the fabric exposes it:

| Algorithm | α term | BW term | Regime / adoption |
|---|---|---|---|
| Binomial tree, non-pipelined ($P = 1$) | $\lceil \log_2 N \rceil \cdot \alpha$ | $\lceil \log_2 N \rceil \cdot M/\mathrm{BW}$ | Small $M$ (α-bound, $M/\mathrm{BW} \ll \alpha$); NCCL/MPI tree path |
| Binomial tree, asymptotic ($P^*$) | $\lceil \log_2 N \rceil \cdot \alpha$ | $M/\mathrm{BW}$ | Medium $M$; NCCL/MPI tree path |
| Ring, asymptotic ($P^*$) | $(N-1) \cdot \alpha$ | $M/\mathrm{BW}$ | Large $M$ (BW-bound, $M/\mathrm{BW} \gg \alpha$); NCCL/MPI ring path |
| Switch multicast (INC) | $2 \cdot \alpha_\mathrm{switch}$ | $M/\mathrm{BW}$ | All $M$ where fabric exposes multicast; NVLS (intra-NVLink), SHARP (IB), Tomahawk Ultra (Ethernet) |

**Reading the rows as a progression.** At small $M$ the BW term is negligible no matter how many times $L$ multiplies it, so software picks whichever algorithm has the smallest $L$ on the $\alpha$ side — binomial tree, with $L = \lceil \log_2 N \rceil$ vs ring's $N-1$. At medium $M$ the BW term matters, so the tree starts pipelining to collapse its $\lceil\log_2 N\rceil \cdot M/\mathrm{BW}$ coefficient down to $M/\mathrm{BW}$; the α term stays at log depth, so tree remains favored. At large $M$ the BW term dominates and both tree-asymptotic and ring-asymptotic sit at the same $M/\mathrm{BW}$ floor; the pure α-β model says tree still wins (smaller α term), but in practice implementation overhead (finite pipeline depth $P \ll P^*$, per-step kernel cost) keeps tree's BW coefficient above the asymptotic 1, while ring's intrinsically-pipelined $P = N$ schedule hits the floor cleanly — the same ring-vs-tree crossover logic that §5.3 derives in full for AR.

INC sits alongside this progression as a hardware alternative rather than a fourth regime. It collapses the α term to an $N$-independent constant (the α-side win) but — unlike AR — does **not** lift the BW ceiling, since pipelined tree and ring BC both already reach $M/\mathrm{BW}$, which is the single-rank-port floor that switch multicast also sits at. INC BC's entire win is therefore on the α side: dramatic at small $M$ (where software BC pays the full $\lceil \log_2 N \rceil \cdot \alpha$ or $(N-1)\alpha$) and converging toward software as $M$ grows and the $M/\mathrm{BW}$ term dominates. See `04_in_network_collectives.md §1.1–§1.2` for the mechanism and per-primitive scope (multicast helps BC, AG, and the broadcast-half of AR).

[^pcie-mcast]: PCIe switches (Broadcom PEX, Microchip Switchtec, etc.) have supported hardware multicast since the PCIe Gen2 spec added it as an optional feature, and Gen4/5 AI-oriented switches expose it reliably. They are therefore useful for BC and AG — and the BC-half of AR where the target fabric is a PCIe tree — but they have **no in-switch ALU**: the multicast primitive replicates payloads only, it does not combine them. PCIe switches are consequently absent from the INC rows of the Reduce and RS tables (§4.3, §6), and do not contribute to the BW-side effective-BW lift that SHARP / NVLS give AR (see `04_in_network_collectives.md §1.4`).

---

## 4. Reduce

Reduce is the time-reverse dual of Broadcast: every rank $i$ starts with its own $V_i$ (size $M$), and rank 0 ends holding $\sum_i V_i$. It shows up in gradient accumulation onto a parameter server, loss aggregation for logging, score collection in some serving paths, and the up-reduce half of tree AR (§5.2). The two schedules mirror BC with time reversed: ring Reduce is a chain accumulating toward the sink (time-reverse of the §3.1 ring BC chain spreading from the source), and binomial-tree Reduce is the same tree as §3.2 with every edge arrow flipped — data flows leaves → root with a summation at every internal node instead of root → leaves. §4.1 covers the ring variant and §4.2 the binomial-tree variant, both with pipelining applied to collapse $n_\beta$ to 1 via the generic mechanism in [Appendix C](#appendix-c-asymptotic-form-of-linear-schedules-via-pipelining). §4.3 relates Reduce back to AR (showing why naïvely composing Reduce + BC is strictly worse than the fused DBT AR in §5.2) and points to the in-network Reduce primitive in `04_in_network_collectives.md §1.1`.

### 4.1 Ring Reduce

Arrange the ranks in a chain $R_{N-1} \to R_{N-2} \to \ldots \to R_1 \to R_0$ with $R_0$ as the root. Reduce has a single sink — the mirror image of BC's single source — so data flows one way along the chain toward $R_0$ and the wrap-around edge $R_0 \to R_{N-1}$ is idle. Each forwarding rank **adds its own $V_i$ into the incoming partial sum** before passing it downstream; the partial sum arriving at $R_0$ has already been summed across $R_{N-1}$ down to $R_1$, and $R_0$ folds in its own $V_0$ to complete. The same ring topology (same physical wiring, same in/out neighbor set) is reused by ring AR in §5.1 and ring AG / RS in §6.1, where every rank contributes on every link and the wrap-around carries real traffic.

**Ring for $N = 4$** (arrows are the forward direction used by Reduce — data flows toward the root):

```
      R3 ───→ R2 ───→ R1 ───→ R0
```

Chunk each rank's $V_i$ into $P$ equal pieces of size $M/P$ and stream them down the chain: once a rank has received partial-sum chunk $s$ from its upstream neighbor, it adds its own chunk $s$ into the arriving vector and forwards the updated partial sum downstream, while concurrently receiving chunk $s+1$ from upstream. Every forward link carries one chunk per step in steady state — the same conveyor mechanic as ring BC (§3.1), with one on-chip add per hop. We walk through $N = 4$ with $P = N$ (one chunk per rank position, matching ring AR's convention in §5.1 and giving the cleanest symmetric fill/drain). Each $V_i$ splits into four chunks $V_{i,s}$ of size $M/4$; write $S_s = V_{0,s} + V_{1,s} + V_{2,s} + V_{3,s}$ for the target full-sum chunk that $R_0$ must hold at the end.

**Initial state.**

```
R3: [V30  V31  V32  V33]   ← source end of chain
R2: [V20  V21  V22  V23]
R1: [V10  V11  V12  V13]
R0: [V00  V01  V02  V03]   ← root (final destination)
```

**After step 1** — $R_3$ ships $V_{3,0}$ to $R_2$; $R_2$ sums it into its own slot 0. Pipeline is filling; only one link is active.

```
Step 1: R3 → R2 (V30); R2 accumulates

R3: [V30         V31  V32  V33]
R2: [V30+V20     V21  V22  V23]   ← slot 0 now 2-way partial over {R3, R2}
R1: [V10         V11  V12  V13]
R0: [V00         V01  V02  V03]
```

**After step 2** — two links active concurrently: $R_3 \to R_2$ streams $V_{3,1}$ while $R_2 \to R_1$ forwards the 2-way partial for slot 0; $R_1$ folds in $V_{1,0}$.

```
Step 2: R3 → R2 (V31); R2 → R1 (V30+V20) concurrently

R3: [V30         V31         V32  V33]
R2: [V30+V20     V31+V21     V22  V23]
R1: [V30+V20+V10 V11         V12  V13]   ← slot 0 now 3-way partial over {R3, R2, R1}
R0: [V00         V01         V02  V03]
```

**After step 3** — pipeline fully populated: three links carry chunks at three stages of accumulation, and $R_0$ receives the first complete slot.

```
Step 3: R3 → R2 (V32); R2 → R1 (V31+V21); R1 → R0 (V30+V20+V10) concurrently

R3: [V30         V31         V32         V33]
R2: [V30+V20     V31+V21     V32+V22     V23]
R1: [V30+V20+V10 V31+V21+V11 V12         V13]
R0: [S0          V01         V02         V03]   ← slot 0 fully reduced at root
```

**After step 4** — last chunk enters the pipeline; slot 1 completes at $R_0$.

```
Step 4: R3 → R2 (V33); R2 → R1 (V32+V22); R1 → R0 (V31+V21+V11) concurrently

R3: [V30         V31         V32         V33]
R2: [V30+V20     V31+V21     V32+V22     V33+V23]
R1: [V30+V20+V10 V31+V21+V11 V32+V22+V12 V13]
R0: [S0          S1          V02         V03]
```

**After step 5** — $R_3$ is idle (its last chunk has already entered the pipeline); two links still active for drain.

```
Step 5: R2 → R1 (V33+V23); R1 → R0 (V32+V22+V12) concurrently

R3: [V30         V31         V32         V33]
R2: [V30+V20     V31+V21     V32+V22     V33+V23]
R1: [V30+V20+V10 V31+V21+V11 V32+V22+V12 V33+V23+V13]
R0: [S0          S1          S2          V03]
```

**After step 6** — final drain; $R_0$ receives the last partial and folds in $V_{0,3}$.

```
Step 6: R1 → R0 (V33+V23+V13)

R0: [S0  S1  S2  S3]   ← Reduce complete at root
```

Total: $N - 1 + P - 1 = 6$ steps (fill $N-1$ for the head chunk to traverse the chain, then drain $P-1$ after the last chunk enters at $R_{N-1}$). Each step ships $M/P$ bytes over its active link at cost $\alpha + M/(P\\,\mathrm{BW})$:

$$t_{\mathrm{ring\\,reduce}} = (N + P - 2)\left(\alpha + \frac{M}{P\\,\mathrm{BW}}\right)$$

At optimal $P^* = \sqrt{(N{-}2)M/(\alpha \mathrm{BW})}$, in the large-$M$ limit, the cost approaches the **asymptotic form**:

$$t_{\mathrm{ring\\,reduce}} \approx (N - 1)\\,\alpha + \frac{M}{\mathrm{BW}}$$

The "$\approx$" hides an $O(\sqrt{M})$ correction of order $2\sqrt{(N{-}2)\alpha M/\mathrm{BW}}$ between the two floors — same structure as ring BC in §3.1, same [Appendix C](#appendix-c-asymptotic-form-of-linear-schedules-via-pipelining) derivation, modified only by the extra on-chip add at each hop. The add is a reduce-op FLOP local to each rank's accelerator; under the standard α-β accounting it is assumed non-bottlenecking (modern accelerators run reduce-ops at a throughput well above network BW) and contributes to neither $\alpha$ nor the BW term.

$n_\alpha = N - 1$, $n_\beta = 1$ — bandwidth-optimal (matches the asymptotic tree floor in §4.2) but linear $\alpha$ in $N$. Unlike the ring-vs-tree AR crossover at large $M$ (§5.3), for standalone Reduce the pipelined tree (§4.2) matches the same $M/\mathrm{BW}$ floor with $\lceil \log_2 N \rceil$ latency instead of $N-1$, so tree strictly dominates on port-adequate fabrics. Ring Reduce does ship in the production algorithm menus — NCCL reuses the shared ring infrastructure that Ring AllReduce runs on and exposes it via `NCCL_ALGO=Ring`, and MPICH / OpenMPI carry ring Reduce in their algorithm tables — but the auto-tuner rarely selects it for standalone Reduce because there is no ring-vs-tree crossover incentive at the single-sink level. The schedule remains load-bearing in two places even so: (i) the RS phase of ring AR / ring RS (§§5.1, 6.1) executes exactly this chain-accumulate mechanic on each chunk, composed with a ring wrap-around so every rank ends up owning one fully-reduced chunk instead of funneling everything to a single root; and (ii) in parameter-server-style deployments where only $R_0$ needs the result and the topology is a literal chain (early MapReduce-era setups, some inference-serving paths aggregating scores), ring Reduce is the schedule the software falls back to.

### 4.2 Binomial tree Reduce

At step $k$, every rank whose $k$-th bit is 1 sends its current vector to its paired "bit-0" partner, which **adds the received vector into its local copy**. The active-contributor set halves each step: $\\{R_0, R_1, R_2, R_3\\} \to \\{R_0, R_2\\} \to \\{R_0\\}$ at $N = 4$. After $\lceil \log_2 N \rceil$ steps only rank 0 remains, holding the full $\sum_i V_i$. We trace $N = 4$ with the same initial state as §5.1 (each $R_i$ holds $V_i = [v_{i,0}, v_{i,1}, v_{i,2}, v_{i,3}]$).

**Tree structure for $N = 4$.** The same binomial tree $B_2$ as §3.2, with every edge's arrow reversed — data flows leaves → root with summation at every internal node:

```
      R0            (depth 0, root — collects sum)
     ▲  ▲
     │  │
    R1   R2         (depth 1)
    ▲
    │
    R3              (depth 2)
```

Read side-by-side with §3.2: identical edges, opposite direction; BC and Reduce are the same tree schedule read in opposite time directions. Depth is $\lceil \log_2 N \rceil = 2$, setting $n_\alpha$ just as for BC.

**Initial state.**

```
R0: [v00  v01  v02  v03]
R1: [v10  v11  v12  v13]
R2: [v20  v21  v22  v23]
R3: [v30  v31  v32  v33]

active-set: {R0, R1, R2, R3}
```

**After step 1** — $R_1 \to R_0$ and $R_3 \to R_2$, concurrently (two disjoint pairs; the "bit-1 senders" fold into their "bit-0 receivers"). $R_0$ and $R_2$ each sum the incoming vector into their local copy; $R_1$ and $R_3$ drop out.

```
Step 1: R1 → R0 (R0 sums in); R3 → R2 (R2 sums in)

R0: [v00+v10   v01+v11   v02+v12   v03+v13]   ← 2-way sum over {R0, R1}
R1: stale                                              ← sent up
R2: [v20+v30   v21+v31   v22+v32   v23+v33]   ← 2-way sum over {R2, R3}
R3: stale                                              ← sent up

active-set: {R0, R2}
```

**After step 2** — $R_2 \to R_0$. The active set halves to $\\{R_0\\}$; $R_0$ now holds the full 4-way sum.

```
Step 2: R2 → R0 (R0 sums in)

R0: [S0  S1  S2  S3]   ← full 4-way sum at root
R1: stale
R2: stale
R3: stale

where S_k = v_{0,k} + v_{1,k} + v_{2,k} + v_{3,k}.
active-set: {R0} — Reduce complete
```

Reduce is done in $\lceil \log_2 4 \rceil = 2$ sequential steps; at general $N$ in $\lceil \log_2 N \rceil$ steps. Each step moves the full $M$-byte vector on the active link:

$$t_{\mathrm{bin\\,reduce}} = \lceil \log_2 N \rceil \cdot \alpha + \lceil \log_2 N \rceil \cdot \frac{M}{\mathrm{BW}}$$

Same $(n_\alpha, n_\beta)$ as binomial tree BC (§3.2), and for the same reason: the tree depth sets both the sequential step count and the per-step full-$M$ transfer. Read side-by-side with the BC matrix in §3.2, the relationship is literal: flip the arrow direction on every edge and replace "send $V$ and overwrite" with "send current vector and sum in" — BC and Reduce are the same tree schedule read in opposite time directions.

**Asymptotic form (bandwidth-bound regime; pipelined implementation).** Chunks of size $M/P$ flow **up** the tree with internal nodes reducing on the fly: once a parent has received chunk $s$ from both of its children, it forwards the summed chunk upward while reducing chunk $s+1$ in parallel. Substituting $L = \lceil \log_2 N \rceil$ (tree depth) into the Appendix C master formula and optimizing at $P^* = \sqrt{(\lceil \log_2 N \rceil - 1)M/(\alpha \mathrm{BW})}$ in the large-$M$ limit, the BW coefficient drops from $\lceil \log_2 N \rceil$ to 1 up to an $O(\sqrt{M})$ correction $2\sqrt{(\lceil \log_2 N \rceil - 1)\alpha M/\mathrm{BW}}$:

$$t_{\mathrm{bin\\,reduce}} \approx \lceil \log_2 N \rceil \cdot \alpha + \frac{M}{\mathrm{BW}}$$

This is what NCCL's tree-path `ncclReduce` runs for bulk $M$ — simultaneously latency- and bandwidth-optimal for Reduce on port-adequate fabrics, mirroring pipelined BC. The generic derivation and the port-budget caveat are in [Appendix C](#appendix-c-asymptotic-form-of-linear-schedules-via-pipelining); the binomial tree satisfies the port budget so the collapse applies. Ring Reduce (§4.1) reaches the same $M/\mathrm{BW}$ floor but with $N-1$ latency hops instead of $\lceil \log_2 N \rceil$, so at the standalone-Reduce level there is no ring-vs-tree crossover — tree dominates.

### 4.3 Comparison and practical adoption

Three software implementations (binomial tree $P=1$, binomial tree $P^*$, ring $P^*$) plus a hardware in-network path sit on the table. Before presenting the combined view, the hardware option needs a brief introduction — the two software algorithms were fully specified in §§4.1–4.2, but the switch-ALU reduce primitive has not yet appeared in this chapter.

**Hardware in-network Reduce primitive.** Switched fabrics expose a switch-hosted ALU (InfiniBand Quantum SHARP, NVSwitch Gen3+ / NVLS) that sums $N$ incoming $M$-byte flits into one $M$-byte output in one switch-local operation. Each rank pushes $V_i$ upstream to the switch (1 hop); the switch crossbar reduces the $N$ payloads on-chip and forwards the single reduced result to the root rank (1 more hop). Two switch-crossing hops total, **independent of $N$**:

$$t_{\mathrm{INC,\\,Reduce}} \\;\approx\\; 2\\,\alpha_\mathrm{switch} + \frac{M}{\mathrm{BW}}$$

Only the switch-ALU half of the INC primitive is used here, symmetric to BC in §3.3 which used only the multicast half — no multicast is needed because the output goes to a single destination. $\alpha_\mathrm{switch}$ is the switch cut-through plus ALU latency — $\sim 100$–$200$ ns for NVSwitch Gen3/4 with NVLS and comparable on IB Quantum-2 with SHARPv2 — typically well below the $\alpha \approx 0.5$–$1\\,\mu$s that software Reduce accumulates across $\lceil \log_2 N \rceil$ or $N-1$ endpoint-driven hops. PCIe switches appear in §3.3's BC list but are absent here because they have no ALU (see footnote at §3.3); Ethernet in-network reduction (RoCE-based aggregation on newer AI-focused switches) is an evolving direction, omitted until its ALU-inclusive latency is well-characterized.

**All four options side by side.** The two software algorithms each have a finite-$P$ and an asymptotic form; which form wins is governed by the $M$-regime, since the choice of $P$ (non-pipelined $P = 1$ vs asymptotic $P = P^*$) is dictated by whether the α term or the BW term dominates. Switch-ALU Reduce is an orthogonal hardware-dependent alternative and applies across regimes where the fabric exposes it:

| Algorithm | α term | BW term | Regime / adoption |
|---|---|---|---|
| Binomial tree, non-pipelined ($P = 1$) | $\lceil \log_2 N \rceil \cdot \alpha$ | $\lceil \log_2 N \rceil \cdot M/\mathrm{BW}$ | Small $M$ (α-bound, $M/\mathrm{BW} \ll \alpha$); NCCL/MPI tree path |
| Binomial tree, asymptotic ($P^*$) | $\lceil \log_2 N \rceil \cdot \alpha$ | $M/\mathrm{BW}$ | Medium-to-large $M$; NCCL/MPI tree path (default for bulk $M$) |
| Ring, asymptotic ($P^*$) | $(N-1) \cdot \alpha$ | $M/\mathrm{BW}$ | Available in NCCL (`NCCL_ALGO=Ring`, shared ring-AR infra) and MPI menus, but rarely selected for standalone Reduce — tree dominates at the same BW floor with smaller α. Same mechanic composes into the RS phase of ring AR / RS (§§5.1, 6) |
| Switch-ALU Reduce (INC) | $2 \cdot \alpha_\mathrm{switch}$ | $M/\mathrm{BW}$ | All $M$ where fabric exposes it; IB SHARP, NVSwitch NVLS |

**Reading the rows as a progression.** At small $M$ the BW term is negligible no matter what coefficient multiplies it, so software picks whichever algorithm has the smallest $L$ on the α side — binomial tree, with $L = \lceil \log_2 N \rceil$ vs ring's $N-1$. At medium $M$ the BW term matters, so the tree pipelines to collapse its $\lceil \log_2 N \rceil \cdot M/\mathrm{BW}$ coefficient down to $M/\mathrm{BW}$; the α term stays at log depth, so tree remains favored. At large $M$ — unlike BC (§3.3) and AR (§5.3) where ring wins on the BW side via its intrinsically-pipelined $P = N$ schedule — Reduce has **no ring-vs-tree crossover**: ring's BW floor matches tree's $M/\mathrm{BW}$ but its α term remains $(N-1)\alpha$ vs tree's $\lceil\log_2 N\rceil \cdot \alpha$, so tree strictly dominates across the full $M$ range. This is why `ncclReduce`'s tuner reaches for the tree path across small and large $M$ alike — the same $(N, M)$ crossover exists between $P=1$ and $P^*$ within the tree family, but not between tree and ring.

INC sits alongside this progression as a hardware alternative rather than a fifth regime. It collapses the α term to an $N$-independent constant (the α-side win) but — unlike AR — does **not** lift the BW ceiling, since pipelined tree and ring both already reach $M/\mathrm{BW}$, which is the single-destination-port floor that switch-ALU Reduce also sits at. INC Reduce's entire win is therefore on the α side: dramatic at small $M$ (where software Reduce pays the full $\lceil \log_2 N \rceil \cdot \alpha$) and converging toward software tree as $M$ grows. See `04_in_network_collectives.md §1.1–§1.2` for the switch-ALU mechanism and per-primitive scope; the BW-side ceiling lift that AR uniquely receives (from fusing the reduce and broadcast halves so each byte crosses the fabric once instead of twice) is laid out in `04_in_network_collectives.md §1.4`.

**Why Reduce stays standalone and does not build AR.** Reduce + BC is semantically equivalent to AR but strictly worse in cost. Composing two pipelined-tree primitives costs $2\lceil \log_2 N \rceil \cdot \alpha + 2 M/\mathrm{BW}$ (pipelined tree in each direction). DBT AR's $2\lceil \log_2 N \rceil \cdot \alpha + M/\mathrm{BW}$ (§5.2) has the **same α count but half the BW term**, because DBT's two complementary trees each carry half the message concurrently on the opposite direction of the same full-duplex link instead of shipping the full reduced result a second time. Modern AR schedules (§5) exploit this fusion and avoid the decomposition. Reduce therefore appears as a standalone primitive only when the output is single-rank by design — gradient accumulation onto a parameter server, loss aggregation for logging, score collection in some serving paths — not as a building block for AR.

---

## 5. All-reduce

AR is the workhorse reduction primitive: every rank starts with its own length-$M$ vector $V_i$ and ends holding the elementwise sum $\sum_i V_i$. Two families of algorithms dominate in practice: **ring-based** (§5.1), which chains $2(N-1)$ sequential exchanges along a logical ring and hits the Patarasuk-Yuan bandwidth lower bound, and **tree-based** (§5.2), which collapses the step count to $O(\log N)$ via hypercube / binary-tree pairings. NCCL ships ring and double binary tree (DBT) — the two algorithms covered in the main text below. Two tree variants from the MPI literature and older HPC work — simple recursive doubling and Rabenseifner halving-doubling — remain instructive as reference points even though they are not in NCCL's shipping menu; their derivations live in [Appendix B](#appendix-b-non-mainline-ar--ag--rs--a2a-variants), and [Appendix B.3](#b3-why-neither-ar-variant-is-shipped) consolidates the port-budget / partner-cycling argument against them. §5.3 summarizes the comparison between the shipped algorithms — ring, DBT, and the hardware in-network AR primitive — and explains the NCCL selection rule between ring and DBT.

### 5.1 Ring AR

Ring AR on $N$ ranks runs RS in $N-1$ steps, then AG in $N-1$ steps — the Patarasuk-Yuan bandwidth-optimal construction. We walk through $N=4$, each rank holding a length-4 vector, chunked into 4 pieces.

**Setup.** Four ranks $R_0, R_1, R_2, R_3$ wired into a ring — each rank concurrently sends to its right neighbor and receives from its left neighbor every step:

```
      R0 ───→ R1 ───→ R2 ───→ R3
       ▲                       │
       └───────────────────────┘
```

Each rank $R_i$ holds a vector $V_i = [v_{i,0}, v_{i,1}, v_{i,2}, v_{i,3}]$. Target: every rank ends up holding $[S_0, S_1, S_2, S_3]$ where $S_k = \sum_i v_{i,k}$.

**Phase 1: reduce-scatter (3 steps).**

At step $t \in \\{1, 2, \ldots, N-1\\}$, rank $i$ **sends** chunk $(i - t + 1) \bmod N$ to its right neighbor and **receives** chunk $(i - t) \bmod N$ from its left neighbor, adding the incoming value into its local copy at that slot. That slot — $(i - t) \bmod N$ — is the rank's **accumulator chunk** for step $t$. Sanity-check at $t=1$: $R_i$ sends its own-index chunk $i$ rightward and accumulates into slot $(i-1) \bmod N$.

The accumulator slot index shifts by $-1 \bmod N$ every step. After $N-1$ steps, each rank's accumulator sits at slot $(i - (N-1)) \bmod N = (i+1) \bmod N$ and holds the fully reduced $N$-way sum for that slot.

```
Initial (before any step):

R0: [v00 v01 v02 v03]
R1: [v10 v11 v12 v13]
R2: [v20 v21 v22 v23]
R3: [v30 v31 v32 v33]
```

**After step 1** — each rank sends one chunk right and receives one chunk from its left neighbor, adding it into its local copy. Only the accumulator chunk changes; every other slot is untouched.

```
Step 1: each rank adds one incoming chunk into its accumulator slot

R0: [v00           v01           v02           v03+v33     ]   ← accumulator: chunk 3
R1: [v10+v00       v11           v12           v13         ]   ← accumulator: chunk 0
R2: [v20           v21+v11       v22           v23         ]   ← accumulator: chunk 1
R3: [v30           v31           v32+v22       v33         ]   ← accumulator: chunk 2
```

Every accumulator now holds a **2-term partial sum**. The slots not marked as accumulators still hold the rank's original values — they'll stay that way until AG phase overwrites them.

**After step 2** — each rank forwards the chunk it just updated to its right neighbor, so the partial sum grows by one more term. The accumulator slot index shifts by $-1 \bmod 4$ each step.

```
Step 2: each rank forwards its step-1 accumulator; receiver adds it in

R0: [v00           v01           v02+v22+v32   v03+v33     ]   ← accumulator: chunk 2 (3 terms)
R1: [v10+v00       v11           v12           v13+v03+v33 ]   ← accumulator: chunk 3 (3 terms)
R2: [v20+v10+v00   v21+v11       v22           v23         ]   ← accumulator: chunk 0 (3 terms)
R3: [v30           v31+v21+v11   v32+v22       v33         ]   ← accumulator: chunk 1 (3 terms)
```

Trace of what flowed into each accumulator:

- R0 chunk 2 picked up $v_{22}+v_{32}$ from R3 (R3's chunk 2 after step 1).
- R1 chunk 3 picked up $v_{03}+v_{33}$ from R0.
- R2 chunk 0 picked up $v_{10}+v_{00}$ from R1.
- R3 chunk 1 picked up $v_{21}+v_{11}$ from R2.

**After step 3** — one more forward, and each accumulator absorbs the final term. The accumulator is now a **full 4-way sum** — that's $S_k$, the fully-reduced chunk for slot $k$.

```
Step 3: each rank forwards its step-2 accumulator; receiver adds the last summand

R0: [v00           S1            v02+v22+v32   v03+v33     ]   ← fully reduced: chunk 1 = S1
R1: [v10+v00       v11           S2            v13+v03+v33 ]   ← fully reduced: chunk 2 = S2
R2: [v20+v10+v00   v21+v11       v22           S3          ]   ← fully reduced: chunk 3 = S3
R3: [S0            v31+v21+v11   v32+v22       v33         ]   ← fully reduced: chunk 0 = S0

where S_k = v_{0,k} + v_{1,k} + v_{2,k} + v_{3,k}.
```

This is exactly the RS end-state: each rank holds **one** fully-reduced chunk (on a different slot), and the other three slots still contain junk partial sums. Those will be overwritten in the next phase.

**Cost accounting.** Each of the 3 steps costs $\alpha + M/(4\\,\mathrm{BW})$ (one handshake plus one $M/N$-sized chunk over the link). The 3 steps run **sequentially** (step $t+1$ forwards the chunk that step $t$ just accumulated), so the costs add: the whole RS phase takes

$$t_{\mathrm{RS}} = 3\alpha + 3 \cdot \frac{M}{4\\,\mathrm{BW}} \qquad \text{(totals across all 3 steps, not per step).}$$

Generalizing to $N$ ranks: $(N-1)\alpha + (N-1)\\,M/(N\\,\mathrm{BW})$.

**Phase 2: all-gather (3 steps).**

After RS, each rank owns exactly one fully-reduced chunk — $R_i$ owns slot $(i+1) \bmod N$ — and needs the other three. The plan: same ring, same direction (send right, receive from left), **no reduction** — each rank forwards what it just received, overwriting the junk in the receiver's corresponding slot.

At step $t \in \\{1, 2, \ldots, N-1\\}$, rank $i$ **sends** chunk $(i - t + 2) \bmod N$ to its right neighbor and **overwrites** its local slot $(i - t + 1) \bmod N$ with the chunk received from its left neighbor. Sanity-check at $t=1$: $R_i$ forwards its freshly-reduced slot $(i+1) \bmod N$ rightward and overwrites its own-index slot $i$ with an incoming fully-reduced chunk.

These are exactly the RS formulas **shifted by $+1$ in slot index**. Same ring, same direction, same $-1 \bmod N$ slot-index shift per step — just a $+1$ offset in where the action starts. The reason for the offset: RS leaves $R_i$ owning slot $(i+1) \bmod N$ (not slot $i$), so AG begins by forwarding that $+1$-offset slot.

Starting state (using `?` for the stale partial sums from RS — they're unreadable garbage, about to be overwritten):

```
Before AG step 1 (only the fully-reduced chunk per rank is known-good):

R0: [ ?     S1    ?     ?  ]
R1: [ ?     ?    S2     ?  ]
R2: [ ?     ?     ?    S3  ]
R3: [S0     ?     ?     ?  ]
```

**After AG step 1** — each rank sends its one fully-reduced chunk right. The receiver overwrites its corresponding slot.

```
AG step 1: R_i sends its one known chunk to R_{i+1}

R0: [S0    S1    ?     ?  ]   (received S0 from R3 → chunk 0 overwritten)
R1: [ ?    S1   S2     ?  ]   (received S1 from R0 → chunk 1 overwritten)
R2: [ ?    ?    S2    S3  ]   (received S2 from R1 → chunk 2 overwritten)
R3: [S0    ?     ?    S3  ]   (received S3 from R2 → chunk 3 overwritten)
```

Each rank now holds **two** fully-reduced chunks.

**After AG step 2** — each rank forwards the chunk it *just* received (not its original one) to the right. Same overwrite pattern.

```
AG step 2: forward what you just received

R0: [S0    S1    ?    S3  ]   (received S3 from R3 → chunk 3 overwritten)
R1: [S0    S1   S2     ?  ]   (received S0 from R0 → chunk 0 overwritten)
R2: [ ?    S1   S2    S3  ]   (received S1 from R1 → chunk 1 overwritten)
R3: [S0    ?    S2    S3  ]   (received S2 from R2 → chunk 2 overwritten)
```

Three of the four slots are now full-reduced on every rank.

**After AG step 3** — one last forward completes the ring.

```
AG step 3: forward what you just received

R0: [S0    S1   S2    S3  ]
R1: [S0    S1   S2    S3  ]
R2: [S0    S1   S2    S3  ]
R3: [S0    S1   S2    S3  ]
```

Every rank now holds the complete reduced vector $[S_0, S_1, S_2, S_3]$. AR is done.

**Cost accounting (AG).** Same structure as RS — 3 sequential steps, each $\alpha + M/(4\\,\mathrm{BW})$. Total for the AG phase:

$$t_{\mathrm{AG}} = 3\alpha + 3 \cdot \frac{M}{4\\,\mathrm{BW}} \qquad \text{(totals across all 3 steps).}$$

Generalizing to $N$ ranks: $(N-1)\alpha + (N-1)\\,M/(N\\,\mathrm{BW})$.

**Total cost.** Combining RS + AG for $N$ ranks:

$$t_{\mathrm{ring\\,AR}} = 2(N-1)\\,\alpha + 2 \cdot \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}$$

For $N=4$: $6\alpha + 1.5 \cdot M/\mathrm{BW}$. In the $N \to \infty$ limit the BW coefficient $2(N{-}1)/N$ approaches the **asymptotic form** $2 M/\mathrm{BW}$ — every byte must leave its originating rank once and arrive at each of the other $N{-}1$ ranks once, giving a floor of $2 M/\mathrm{BW}$ that ring hits nearly perfectly.

**Asymptotic form (bandwidth-bound regime; intrinsically pipelined implementation).** Unlike ring BC (§3.1) and ring Reduce (§4.1) — which trace a single message through $N{-}1$ sequential hops and reach their asymptote only at optimal pipeline depth $P^{*} = \sqrt{(N{-}2)M/(\alpha\mathrm{BW})}$ — ring AR's closed form above **is already the pipelined asymptote** with $P = N$. The Patarasuk-Yuan construction bakes the pipelining into the derivation via the $M/N$ chunked payload; the $2(N{-}1)$-step schedule is not the $P = 1$ baseline that gets pipelined later, it is the pipelined schedule. Three structural reasons no further collapse is possible:

- **No fill or drain.** Every rank starts with its own full vector, so at step 1 every rank is already concurrently sending one chunk on its right-neighbor link and receiving a different chunk from its left — steady state from step 1 to step $2(N{-}1)$. Unlike ring BC / Reduce whose pipeline fills over $N{-}1$ inactive steps while the first message traverses the chain, ring AR runs $N$ parallel chunk-pipelines in lockstep (each originating rank drives its own $N$-hop journey), so there is no fill phase to amortize.
- **$P > N$ cannot improve the BW floor.** The bottleneck neighbor-link already carries $2(N{-}1) \cdot M/N$ total bytes across the collective — matching the per-rank BW lower bound for AR (each of $M$ bytes must leave its origin rank once and arrive at each of the other $N{-}1$ ranks once). Finer segmentation shrinks per-step payload but not total bytes per link; the BW coefficient $2(N{-}1)/N \to 2\\,M/\mathrm{BW}$ is a true lower bound, not an asymptote approached in a limit.
- **No $O(\sqrt{M})$ correction.** The [Appendix C](#appendix-c-asymptotic-form-of-linear-schedules-via-pipelining) master formula's $2\sqrt{(L{-}1)\alpha M/\mathrm{BW}}$ correction originates in the $L{-}1$ idle steps while a single-source pipeline fills. Ring AR has no single source and no fill, so that correction is absent — the closed form is exact under the conflict-free ring assumption, not an $\approx$-approximation:

$$t_{\mathrm{ring\\,AR}} \\;=\\; 2(N{-}1)\\,\alpha + 2 \cdot \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}$$

This property — bandwidth-optimal without needing the pipelining trick — is the structural reason ring stays competitive at large $M$ despite its $O(N) \cdot \alpha$ latency, and the reason ring re-takes the crown from DBT at bulk $M$ in NCCL's tuner. Revisited in §5.3 where ring's exact asymptote sits beside DBT's pipelined $M/\mathrm{BW}$ floor (whose real-world coefficient $c_{\mathrm{real}} \geq 1$ inflates above that floor under implementation overhead) and the INC primitive's $M/\mathrm{BW}$ floor.

### 5.2 Double binary tree (DBT) AR

Tree-based AR collapses ring's $N-1$ sequential steps into $\lceil \log_2 N \rceil$ by using a **binary-hypercube pairing pattern** instead of a ring. At each step, every rank exchanges data with exactly one partner, and the "reachable set" doubles in size each step — so the whole $N$-rank group is covered in $\log_2 N$ steps.

Three tree-flavored AR variants appear in the literature, each reading AR through one of its two natural decompositions (RS + AG, or reduce + broadcast):

- **Simple recursive doubling — does not decompose.** Single-pass butterfly / hypercube sweep, equivalent to recursive-doubling AG with addition substituted for concatenation (adding two $M$-sized vectors keeps the result $M$-sized, so no half grows or shrinks). Minimum latency, but the price of single-pass simplicity is sending the full $M$ at every step, so bandwidth scales as $\log_2 N \cdot M$. **Not shipped by NCCL.** Full derivation: [Appendix B.1](#b1-simple-recursive-doubling-ar).
- **Rabenseifner halving-doubling ≡ RS + AG.** Two complementary hypercube passes with chunk-exponential payloads. Recognizes that step $k$ only needs the step-$k$-relevant chunks (size $2^{k-1} \cdot M/N$ or its complement), which shrinks the BW coefficient from $\log_2 N$ to the optimal $2(N{-}1)/N$ at the cost of $2\times$ more steps. **Not shipped by NCCL.** Full derivation: [Appendix B.2](#b2-rabenseifner-halving-doubling-ar).
- **Double binary tree (DBT) [SST09] ≡ reduce + broadcast (× 2 trees).** Two complementary tree passes, full-vector payloads split into pipeline segments. The second tree $T_2$ runs concurrently with complementary interior/leaf roles so both link directions stay busy, halving the per-link message at every step. **Shipped by NCCL** as the non-ring multi-node default; derived in full below.

**Why NCCL ships DBT and not the other two is the subject of [Appendix B.3](#b3-why-neither-ar-variant-is-shipped)**; the short version is that DBT is the only tree-flavored variant whose per-rank concurrency requirement fits the port budget of real fabrics, which in turn lets pipelining collapse its $\log_2 N$ BW coefficient down to near-ring. The other two serialize on bounded-port fabrics.

**Single binary tree at $N=4$.** Tree structure:

```
         R0 (root, depth 0)
        /  \
       R1   R2  (depth 1)
      /
     R3  (depth 2)
```

$R_0$ is root. $R_1$ and $R_2$ are $R_0$'s children. $R_3$ is $R_1$'s child. Depth = 2 = $\lceil \log_2 4 \rceil$.

Initial state (same $v_{i,k}$ as before):

```
R0: [v00 v01 v02 v03]
R1: [v10 v11 v12 v13]
R2: [v20 v21 v22 v23]
R3: [v30 v31 v32 v33]
```

**Phase 1 — Reduce (depth $\log_2 N$ steps, leaves → root).**

**Step 1 (depth 2 → depth 1).** $R_3$ sends its full vector up to $R_1$; $R_1$ sums. $R_2$ is idle (no children).

```
R0: [v00         v01         v02         v03       ]   ← unchanged (root waiting)
R1: [v10+v30     v11+v31     v12+v32     v13+v33   ]   ← 2-way sum over {R1, R3}
R2: [v20         v21         v22         v23       ]   ← unchanged (leaf, waiting)
R3: stale                                                ← sent up
```

**Step 2 (depth 1 → depth 0).** $R_1$ and $R_2$ concurrently send their vectors up to $R_0$. $R_0$ sums both incoming into its local copy.

```
R0: [S0          S1          S2          S3        ]   ← 4-way full sum
R1: stale                                                ← sent up
R2: stale                                                ← sent up
R3: stale
```

**Phase 2 — Broadcast (depth $\log_2 N$ steps, root → leaves).**

**Step 3 (depth 0 → depth 1).** $R_0$ concurrently sends $S$ down to $R_1$ and $R_2$.

```
R0: [S0 S1 S2 S3]
R1: [S0 S1 S2 S3]   ← received
R2: [S0 S1 S2 S3]   ← received
R3: stale
```

**Step 4 (depth 1 → depth 2).** $R_1$ sends $S$ down to $R_3$. $R_2$ is idle (no children).

```
R0: [S0 S1 S2 S3]
R1: [S0 S1 S2 S3]
R2: [S0 S1 S2 S3]
R3: [S0 S1 S2 S3]   ← received
```

All 4 ranks now hold $S$. Single-tree AR done in $2 \lceil \log_2 N \rceil = 4$ sequential steps.

**Cost of single binary tree.** Each step moves the full $M$-byte vector on the active link: $\alpha + M/\mathrm{BW}$. 4 sequential steps → total $4\alpha + 4M/\mathrm{BW}$. Generalizing:

$$t_{\mathrm{single\\,tree\\,AR}} = 2\lceil \log_2 N \rceil \\, \alpha + 2\lceil \log_2 N \rceil \cdot \frac{M}{\mathrm{BW}}$$

Latency hits the log-depth lower bound for a binary-combinable reduction, but the BW coefficient is $\log_2 N$ — a factor of $\log_2 N / 2$ above the ring-optimal $2(N{-}1)/N \to 2$ floor. The inefficiency: at steps 1 and 4, only one link is active; at steps 2 and 3, multiple links are active but each one is driving only one direction. Half the full-duplex link capacity is idle across the algorithm.

**The double-tree optimization.** Construct a second tree $T_2$ whose role assignments are the complement of $T_1$: every rank that is interior in $T_1$ is a leaf in $T_2$, and vice versa. For $N=4$:

```
T1:         R0                T2:          R3
           /  \                            /  \
          R1   R2                         R0   R2
         /                                      \
        R3                                       R1
```

Role check:

| Rank | $T_1$ | $T_2$ |
|---|---|---|
| $R_0$ | root (internal) | leaf |
| $R_1$ | internal | leaf |
| $R_2$ | leaf | internal |
| $R_3$ | leaf | root (internal) |

Each rank is interior in *exactly one* tree. Now run AR on $T_1$ and $T_2$ **concurrently**, each on a different half of the message: chunks $\\{0, 1\\}$ (left half, $M/2$ bytes) flow through $T_1$ — reduced at $R_0$, broadcast from $R_0$. Chunks $\\{2, 3\\}$ (right half, $M/2$ bytes) flow through $T_2$ — reduced at $R_3$, broadcast from $R_3$. Because the two trees have complementary roles, a rank's $T_1$ traffic travels on one direction of the full-duplex link while its $T_2$ traffic travels on the other — both halves make progress every step.

Initial state (`|` separates the left half carried by $T_1$ from the right half carried by $T_2$):

```
               left half (T1)        |       right half (T2)
R0: [        v00         v01         |       v02         v03        ]
R1: [        v10         v11         |       v12         v13        ]
R2: [        v20         v21         |       v22         v23        ]
R3: [        v30         v31         |       v32         v33        ]
```

**Step 1 — reduce, depth 2 → 1 in both trees.** $T_1$: $R_3 \to R_1$ (left half). $T_2$: $R_1 \to R_2$ (right half). Concurrent.

```
R0: [        v00         v01         |       v02         v03        ]   ← idle both trees
R1: [       v10+v30    v11+v31       |      stale       stale       ]   T1: +R3 left; T2: right sent up
R2: [        v20         v21         |     v22+v12    v23+v13       ]   T2: +R1 right
R3: [       stale       stale        |       v32         v33        ]   T1: left sent up
```

**Step 2 — reduce, depth 1 → 0 in both trees.** $T_1$: $R_1, R_2 \to R_0$ (left halves, two concurrent sends into $R_0$). $T_2$: $R_0, R_2 \to R_3$ (right halves, two concurrent sends into $R_3$). Concurrent.

```
R0: [         S0          S1         |      stale       stale       ]   T1: 4-way left; T2: right sent up
R1: [       stale       stale        |      stale       stale       ]
R2: [       stale       stale        |      stale       stale       ]
R3: [       stale       stale        |        S2          S3        ]   T2: 4-way right

where S_k = v_{0,k} + v_{1,k} + v_{2,k} + v_{3,k}.
```

After the two reduce steps, $R_0$ holds the fully-reduced left half and $R_3$ holds the fully-reduced right half — mirror-image completions of each tree's reduce phase.

**Step 3 — broadcast, depth 0 → 1 in both trees.** $T_1$: $R_0 \to R_1, R_2$ (left half). $T_2$: $R_3 \to R_0, R_2$ (right half). Concurrent.

```
R0: [         S0          S1         |        S2          S3        ]   T2: received right from R3
R1: [         S0          S1         |      stale       stale       ]   T1: received left from R0
R2: [         S0          S1         |        S2          S3        ]   both trees: received
R3: [       stale       stale        |        S2          S3        ]
```

**Step 4 — broadcast, depth 1 → 2 in both trees.** $T_1$: $R_1 \to R_3$ (left half). $T_2$: $R_2 \to R_1$ (right half). Concurrent.

```
R0: [         S0          S1         |        S2          S3        ]
R1: [         S0          S1         |        S2          S3        ]   T2: received right from R2
R2: [         S0          S1         |        S2          S3        ]
R3: [         S0          S1         |        S2          S3        ]   T1: received left from R1
```

All 4 ranks now hold $[S_0, S_1, S_2, S_3]$ in $2\lceil \log_2 N \rceil = 4$ sequential steps — same step count as single-tree, but each step moves only $M/2$ bytes per link (the other half is on the sibling tree, which occupies the opposite direction of the same full-duplex link).

**Cost.** $2\lceil \log_2 N \rceil$ sequential steps, each costing $\alpha + (M/2)/\mathrm{BW}$ (handshake + half-message on the active link):

$$t_{\mathrm{double\\,tree\\,AR}} = 2\lceil \log_2 N \rceil \\, \alpha + \lceil \log_2 N \rceil \cdot \frac{M}{\mathrm{BW}}$$

For $N=4$: $4\alpha + 2\\,M/\mathrm{BW}$ — $2\times$ better than single-tree's $4\\,M/\mathrm{BW}$, and at this size coincidentally matching ring's $2\\,M/\mathrm{BW}$. The bandwidth term improves on single-tree's $2\log_2 N \cdot M/\mathrm{BW}$ by a factor of $2$, thanks to the half-message-per-link property the second tree enables.

**Asymptotic form (bandwidth-bound regime; pipelined implementation).** The closed form above is the non-pipelined schedule with $P = 1$ — each tree carries one full $M/2$-byte segment per step. Cut each tree's half-message into $P$ equal segments and stream them through the $L = 2\lceil \log_2 N \rceil$-step schedule: segments at different tree depths use disjoint physical edges (a segment-$s$ forward on a depth-$k$ edge and a segment-$(s{+}1)$ forward on a depth-$(k{-}1)$ edge don't conflict), and the two trees occupy opposite directions of each full-duplex link (so $T_1$ and $T_2$ segments don't conflict either). Per-rank concurrency peaks at $\sim 3$ partners — the busiest role is a tree's root at the final reduce step, concurrently receiving from its up-to-2 children while participating in the sibling tree's broadcast as a leaf — well within the port budget of any modern fabric tier (3+ NVLink channels on-node, 3+ NICs off-node). Substituting $L = 2\lceil \log_2 N \rceil$ and per-segment payload $M/(2P)$ into the [Appendix C](#appendix-c-asymptotic-form-of-linear-schedules-via-pipelining) master formula and optimizing:

$$t_{\mathrm{DBT,\\,pipe}}(P^{*}) \\;\approx\\; 2\lceil \log_2 N \rceil \cdot \alpha + \frac{M}{\mathrm{BW}}$$

The "$\approx$" hides an $O(\sqrt{M})$ correction of order $2\sqrt{(2\lceil \log_2 N \rceil - 1)\\,\alpha M/\mathrm{BW}}$ — same [Appendix C](#appendix-c-asymptotic-form-of-linear-schedules-via-pipelining) master-formula structure as ring BC (§3.1) and ring Reduce (§4.1), now with $L = 2\lceil \log_2 N \rceil$; the correction vanishes relative to the $M/\mathrm{BW}$ floor as $M \to \infty$. **The pure-model $\lceil \log_2 N \rceil$ factor in the BW term is gone** — replaced by the App-C floor of 1 — while the latency term stays at $2\lceil \log_2 N \rceil \alpha$. That is the "log-depth latency AND near-ring bandwidth" combination that makes DBT the shipping choice for small-to-medium $M$ on NCCL. This $M/\mathrm{BW}$ floor is a *lower bound* only: implementation reality (finite pipeline depth $P \ll P^*$, per-step kernel launch overhead, imperfect edge-by-edge overlap) pushes the real-world DBT BW coefficient to some $c_{\mathrm{real}} \geq 1$ above this floor — §5.3's practice caveat explains how that inflation determines where NCCL's tuner flips from DBT to ring at bulk $M$.

### 5.3 Comparison and practical adoption

Three software forms plus a hardware in-network path sit on the table: ring (§5.1, intrinsically pipelined with $P = N$), DBT non-pipelined ($P = 1$, §5.2), DBT asymptotic ($P^*$, §5.2), and switch-ALU AR (INC). Before presenting the combined view, the hardware option needs a brief introduction — the three software forms were fully specified in §§5.1–5.2, but INC AR has not yet appeared in this chapter. Two MPI-era variants — simple recursive doubling and Rabenseifner — are omitted from the table because NCCL / RCCL do not ship them; their derivations and the port-budget / partner-cycling argument that rules them out live in [Appendix B.3](#b3-why-neither-ar-variant-is-shipped).

**Hardware in-network AR primitive.** Switched fabrics expose a switch-hosted ALU (InfiniBand Quantum SHARP / SHARPv2, NVSwitch Gen3+ / NVLS) that fuses AR's two halves into a single on-chip operation: the switch crossbar reduces the $N$ incoming $M$-byte flits into one output, then multicasts that output back down to all $N$ ranks. Each endpoint link carries $M$ bytes up (one flit into the switch) and $M$ bytes down (the multicast copy) on opposite directions of the full-duplex link. Because the up and down halves use opposite directions they overlap — unlike software AR, whose two halves are serialized within the step schedule — so the BW term drops from software AR's $2\\,M/\mathrm{BW}$ floor to $M/\mathrm{BW}$. Two switch-crossing hops total, **independent of $N$**:

$$t_{\mathrm{INC,\\,AR}} \\;\approx\\; 2\\,\alpha_\mathrm{switch} + \frac{M}{\mathrm{BW}}$$

Both halves of the INC primitive are used here — the switch-ALU half (compare §4.3 INC Reduce, which used only the ALU half) and the switch-multicast half (compare §3.3 INC BC, which used only the multicast half). $\alpha_\mathrm{switch}$ is the switch cut-through plus ALU latency — $\sim 100$–$200$ ns for NVSwitch Gen3/4 with NVLS and comparable on IB Quantum-2 with SHARPv2 — typically well below the $\alpha \approx 0.5$–$1\\,\mu$s that software AR accumulates across $2\lceil \log_2 N \rceil$ or $2(N-1)$ endpoint-driven hops. PCIe switches are absent here for the same ALU-less reason they are absent from §4.3 (see footnote at §3.3). The BW-side lift — unique to AR among the primitives in this chapter — is derived in `04_in_network_collectives.md §1.4`.

**All four options side by side.** The two DBT rows differ by the choice of $P$ (non-pipelined $P = 1$ vs asymptotic $P = P^*$), and which wins is dictated by the $M$-regime since that choice is governed by whether the α term or the BW term dominates. Switch-ALU AR is an orthogonal hardware-dependent alternative and applies across regimes where the fabric exposes it:

| Algorithm | α term | BW term | Regime / adoption |
|---|---|---|---|
| Ring, intrinsically pipelined ($P = N$, §5.1) | $2(N{-}1)\\,\alpha$ | $\dfrac{2(N{-}1)}{N} \cdot \dfrac{M}{\mathrm{BW}}$ | Large-$M$ AR in NCCL (ring re-takes the crown — see practice caveat) |
| DBT, non-pipelined ($P = 1$, §5.2) | $2\lceil \log_2 N \rceil \\, \alpha$ | $\lceil \log_2 N \rceil \cdot \dfrac{M}{\mathrm{BW}}$ | Small-$M$ AR (α-bound regime); NCCL DBT path before pipelining kicks in |
| DBT, asymptotic ($P^*$, §5.2) | $2\lceil \log_2 N \rceil \\, \alpha$ | $\dfrac{M}{\mathrm{BW}}$ | Small-to-medium $M$ AR in NCCL (default DBT path at bulk $M$) |
| Switch-ALU AR (INC) | $2 \cdot \alpha_\mathrm{switch}$ | $M/\mathrm{BW}$ | All $M$ where fabric exposes it; IB SHARP/SHARPv2, NVSwitch NVLS |

**Reading the rows as a progression.** At small $M$ the BW term is negligible no matter what multiplies it, so software picks the algorithm with the smallest $\alpha$ count — DBT's $2\lceil \log_2 N \rceil$ beats ring's $2(N{-}1)$, and NCCL runs DBT. At medium $M$ the BW term matters, and DBT pipelines to collapse its $\lceil \log_2 N \rceil \cdot M/\mathrm{BW}$ coefficient down to the $M/\mathrm{BW}$ α-β floor; α stays at log depth, so DBT remains favored. Under the pure α-β model, DBT strictly dominates ring at every $M > 0$ — both are at log-depth or linear-depth α and DBT's BW coefficient is half of ring's. At large $M$, however — unlike Reduce (§4.3) where tree strictly dominates ring — AR exhibits a genuine ring-vs-DBT crossover *in practice*: ring's $2(N{-}1)/N \to 2$ floor is attained intrinsically (no pipelining overhead to pay), while DBT's $M/\mathrm{BW}$ is only a *lower bound* and the real-world coefficient inflates above it (see practice caveat below), so ring re-takes the crown once $M$ is large enough that its $O(N) \cdot \alpha$ latency amortizes into the pipeline.

INC sits alongside this progression as a hardware alternative rather than a fourth regime. Unlike BC (§3.3) and Reduce (§4.3) where INC's entire win is on the α side, **AR uniquely receives a BW-side lift**: the switch-fused reduce+multicast makes each byte cross each endpoint link once in each direction concurrently rather than sequentially as software schedules do, pushing the BW coefficient from ring's $2(N{-}1)/N \to 2\\,M/\mathrm{BW}$ floor down to $M/\mathrm{BW}$. INC matches DBT's α-β asymptote on the BW side — but hits it as a true hardware floor without the practice inflation ($c_{\mathrm{real}} \geq 1$) that DBT pays in software. See `04_in_network_collectives.md §1.4` for the effective-BW derivation. The α-side win is the same $N$-independent $2\\,\alpha_\mathrm{switch}$ — dramatic relative to software's $O(\log N)$ or $O(N)$ α hops.

**No α-β crossover exists between ring and DBT.** On the pure α-β model with DBT at its pipelined floor $M/\mathrm{BW}$, DBT strictly dominates ring at every $M > 0$: its latency term $2\lceil \log_2 N \rceil \alpha$ is strictly below ring's $2(N{-}1)\alpha$ for all $N \geq 4$, and its BW coefficient $1$ is half of ring's $2(N{-}1)/N \to 2$. The observed large-$M$ inversion where ring re-takes the crown is entirely an implementation-practice effect — see practice caveat below. Crossovers against the non-shipped rec-doub and Rabenseifner variants are in [Appendix B.3](#b3-why-neither-ar-variant-is-shipped).

**Practice caveat — where ring re-takes the crown.** NCCL's tuner picks DBT for small-$M$ AR and ring for large-$M$ AR, and published benchmarks [DEMYST-NCCL] confirm this inversion. The $M/\mathrm{BW}$ floor from §5.2 is a *lower bound* only: tree-specific implementation overhead (finite pipeline depth $P \ll P^{*}$, per-step kernel complexity, CUDA launch and synchronization granularity, imperfect edge-by-edge overlap) pushes the real-world DBT BW coefficient to some $c_{\mathrm{real}} \geq 1$ above the floor, while ring's $2(N{-}1)/N \to 2$ is attained intrinsically with no pipelining overhead to pay. Crossover in $M$ from equating the two:

$$M_*^{\\,\mathrm{practice}} \\;=\\; \frac{2\\,(N - 1 - \lceil \log_2 N \rceil)\\,\alpha \cdot \mathrm{BW}}{c_{\mathrm{real}} - 2(N{-}1)/N}.$$

For $c_{\mathrm{real}} = 2$ (a representative setting where DBT's per-byte cost roughly doubles the floor) at scale-up fabric parameters ($\alpha = 0.5\\,\mu$s, $\mathrm{BW} = 900\\,\mathrm{GB/s}$ — NVLink-5 / NVSwitch class): $N = 72 \Rightarrow M_*^{\\,\mathrm{practice}} \approx 2\\,\mathrm{GB}$; $N = 512 \Rightarrow M_*^{\\,\mathrm{practice}} \approx 116\\,\mathrm{GB}$. For $c_{\mathrm{real}} \to 2(N{-}1)/N$ the crossover goes to infinity (DBT never loses); for $c_{\mathrm{real}} < 2(N{-}1)/N$ there is no real crossover and DBT dominates ring everywhere, matching the α-β result. The α-β-only reasoning remains the right intuition for small-to-medium $M$ where DBT's lower $\alpha$ dominates; ring wins when per-step BW overhead is small enough that its $O(N) \cdot \alpha$ latency amortizes into the pipeline.

---

## 6. All-gather / reduce-scatter

AG and RS are the two halves of AR and also useful primitives in their own right. **AG** starts with each rank holding one $M/N$-byte chunk and ends with all ranks holding the concatenation of all $N$ chunks. **RS** is the dual: each rank starts with the full $M$-byte vector and ends holding one $M/N$ chunk reduced across all ranks. Both appear pervasively in sharded-parameter training (gradient RS into sharded optimizer state, weight AG before each forward pass), sequence-sharded activations, and any scheme that splits a tensor across ranks and needs to rematerialize or reduce it.

Like AR, both primitives have ring-based and tree-flavored implementations. **Unlike AR, however, no tree-flavored AG / RS variant beats ring on α-β for scale-up.** The two trees in the literature — recursive-doubling AG / recursive-halving RS (MPI menu, [App. B.4](#b4-recursive-doubling-ag--recursive-halving-rs)) and Parallel Aggregated Trees (PAT, NCCL 2.23+, [App. A](#appendix-a-parallel-aggregated-trees-pat--scale-out-ag--rs)) — both match ring's $(N{-}1)/N$ BW coefficient without beating it, so their only edge is a $\lceil \log_2 N \rceil$ vs $N{-}1$ α term. That α edge only pays off when $N \cdot \alpha$ is non-negligible — the scale-out regime where $N$ = node count and inter-node α is in the μs range. On scale-up ($N \lesssim 72$, NVLink/NVSwitch α ≈ 0.5 μs), $(N{-}1)\alpha$ stays within tens of μs and ring wins on concrete practice grounds that the α-β model misses. So **NCCL ships ring as the sole scale-up AG / RS path**, and dispatches to PAT only for inter-node AG / RS at 1 rank per node. This section derives ring, explains why it wins the scale-up comparison despite its higher α, and calls out the scale-out PAT exception at the end. Rec-doubling / rec-halving stays in App. B.4 as MPI reference; PAT mechanics are in App. A.

**Ring wiring.** Ring-based AG is exactly Phase 2 of ring AR from §5.1; ring-based RS is exactly Phase 1. Same wiring, same forward direction — each rank concurrently sends one chunk right and receives one chunk from its left neighbor every step:

```
      R0 ───→ R1 ───→ R2 ───→ R3
       ▲                       │
       └───────────────────────┘
```

AG forwards received chunks (overwrite on arrival). RS forwards an accumulating partial sum (add on arrival). As standalone collectives they each take $N-1$ steps:

$$t_{\mathrm{ring\\,RS}} = t_{\mathrm{ring\\,AG}} = (N-1)\\,\alpha + \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}$$

where $M$ is the **per-rank final total volume** — for AG, the size each rank ends with; for RS, the size each rank starts with before the reduce. The bandwidth term is exactly half of ring AR's because you only do one of the two phases.

**Asymptotic form (bandwidth-bound regime; intrinsically pipelined implementation).** Ring AG / RS inherits ring AR's property (§5.1): the closed form above **is already the pipelined asymptote** with $P = N$. The $M/N$ chunked payload bakes pipelining into the derivation; the $N{-}1$-step schedule is not a $P = 1$ baseline that gets pipelined later, it is the pipelined schedule. Three structural reasons no further collapse is possible, paralleling §5.1:

- **No fill or drain.** $N$ parallel chunk-pipelines run in lockstep — for AG each of the $N$ original chunks drives its own $N{-}1$-hop journey from its origin rank; for RS each output chunk accumulates along a symmetric $N{-}1$-hop path. Every link is busy from step 1 to step $N{-}1$; no fill phase to amortize.
- **$P > N$ cannot improve the BW floor.** The bottleneck neighbor-link carries $(N{-}1) \cdot M/N$ total bytes over the collective — matching the per-rank BW lower bound (each byte crosses each endpoint link $(N{-}1)/N$ times on average). Finer segmentation shrinks per-step payload but not total bytes per link; the BW coefficient $(N{-}1)/N$ is a true lower bound, not an asymptote approached in a limit.
- **No $O(\sqrt{M})$ correction.** The [Appendix C](#appendix-c-asymptotic-form-of-linear-schedules-via-pipelining) master formula's $\sqrt{M}$ correction originates in the $L{-}1$ idle steps while a single-source pipeline fills. Ring AG / RS has no single source and no fill, so that correction is absent — the closed form is exact under the conflict-free ring assumption, not an $\approx$-approximation:

$$t_{\mathrm{ring\\,RS}} \\;=\\; t_{\mathrm{ring\\,AG}} \\;=\\; (N{-}1)\\,\alpha + \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}$$

Exactly half of ring AR's BW coefficient — consistent with AG / RS being one of AR's two phases. This is the structural reason ring is BW-optimal for AG / RS without needing pipelining gymnastics, and the reason the tree-flavored alternatives (App. B.4, App. A) can only *match* this floor, not beat it — discussed next.

**Why ring is the scale-up default.** On α-β arithmetic the tree variants strictly dominate ring: both App. B.4 and App. A match ring's $(N{-}1)/N$ BW coefficient with a $\lceil \log_2 N \rceil$ α vs ring's $N{-}1$ α. Yet NCCL ships ring for every scale-up AG / RS configuration. The inversion is an implementation-practice effect — four concrete advantages the α-β model misses add up to more than the trees' log-depth α edge at scales where $\alpha \cdot N$ is small:

- **Pipeline feasibility.** Ring's per-round chunk is a constant $M/N$, so chunks stream through the ring at wire speed — the α term amortizes across the pipeline and vanishes at large $M$, exactly as derived above. Rec-doubling AG ships chunks of geometrically growing size $M/N, 2M/N, 4M/N, \ldots$; a pipelined schedule would need custom per-round chunk geometry, or drain the pipeline at every round boundary. Rec-halving RS has the mirror problem (chunks halving each round). PAT's reversed-Bruck schedule is pipelineable but engineered around the inter-node fabric's staging constraints, not the scale-up fabric's.
- **Non-power-of-2 support.** Rec-doub / rec-halv strictly need $N = 2^k$ for clean bit-pairings. Ring handles any $N$, including the odd counts that fall out of pipeline-parallel stage assignments or irregular job configurations — NCCL routinely runs on $N = 6$, $N = 72$ (NVL72), and other non-power-of-2 counts.
- **Kernel simplicity.** Ring is a single communication loop with fixed chunk size; rec-doubling is $\lceil \log_2 N \rceil$ distinct rounds, each with a different partner offset and chunk size. Each additional kernel-launch boundary costs CUDA setup time that the α-β model treats as free.
- **Compute overlap.** The steady-chunk pattern of ring AG overlaps naturally with per-chunk compute (e.g., FSDP / ZeRO gather-then-compute). Rec-doubling's round boundaries are global sync points that break overlap.

Also relevant: unlike AR, **no hardware in-network primitive exists for AG / RS**. SHARP / NVLS fuse the AR-specific simultaneous reduce+multicast in the switch ALU; neither pure concatenation (AG) nor pure per-chunk reduction (RS) receives that two-halves-at-once BW-side lift (see `04_in_network_collectives.md §2`), so the scale-up choice is purely between software variants and ring wins.

MPI (MPICH, OpenMPI) does ship rec-doubling AG and rec-halving RS as algorithm options, typically dispatched by message-size threshold at runtime; NCCL's single-algorithm-per-primitive default prioritizes the pipelineable ring case on scale-up.

**Scale-out exception: PAT at 1 rank per node.** PAT [NCCL-PAT] addresses the one regime where ring's $(N{-}1)\alpha$ term becomes prohibitive — one rank per node communicating over the NIC / inter-node path, so $N$ is the node count rather than the intra-node GPU count. On scale-up ($\alpha \approx 0.5\\,\mu$s, $N \lesssim 72$) $(N{-}1)\alpha$ stays in the tens of μs; on scale-out with inter-node α in the μs range and $N$ reaching the hundreds, the same term blows up to hundreds of μs. PAT reverses the Bruck offset schedule (largest hops first) and ships a **bounded intermediate buffer** so it composes with the inter-node path's finite staging memory. Two constraints limit where PAT ships:

1. **Inter-node only, 1 rank per node.** The NCCL 2.23 implementation restricts PAT to one rank per node because only the inter-node phase is implemented; intra-node AG / RS still runs ring. This matches the scale-out design intent — PAT's log-depth structure pays off against the scale-out fabric's α cost, not against on-node NVLink latency.
2. **Scale-up doesn't benefit.** On a scale-up NVLink / NVSwitch domain, ring's α cost is already small and its BW-optimal pipelining keeps it at the BW floor. Replacing ring with PAT would trade pipeline-friendliness for a log-depth α schedule that doesn't fit the scale-up port budget any better than rec-doubling does — the same partner-cycling objection from AR ([Appendix B.3](#b3-why-neither-ar-variant-is-shipped)) applies.

The scale-out motivation and the specific "reversed Bruck + bounded buffer" mechanism are worked step-by-step at $N = 8$ in [Appendix A](#appendix-a-parallel-aggregated-trees-pat--scale-out-ag--rs).

---

## 7. All-to-all

A2A permutes data: rank $i$ holds $N$ chunks where chunk $j$ is destined for rank $j$; after the collective, rank $i$ holds $N$ chunks where chunk $j$ was contributed by rank $j$. The transpose pattern is a pure permutation — no summation, no reduction — so aggregate cross-fabric traffic is exactly $(N{-}1)M$ bytes (each of the $N$ ranks ships $(N{-}1)/N$ of its per-rank payload to other ranks). This makes A2A the **most bandwidth-hungry** of the primitives in this note: unlike AR, whose RS+AG decomposition compresses the BW term to $2(N{-}1)/N \cdot M/\mathrm{BW}$, A2A has no such decomposition to hide behind.

**Worked example at $N=4$.** Each rank $R_i$ starts with $N$ chunks $v_{i,0}, v_{i,1}, v_{i,2}, v_{i,3}$, where $v_{i,j}$ is the chunk that $R_i$ contributes *to* $R_j$. The transpose pattern arises in any permutation-shaped redistribution — MoE expert dispatch, distributed FFTs, matrix transposes, shuffle-in-parallel sorting — and the derivation below is agnostic to which.

```
Initial (each rank holds 4 chunks, one per destination):

R0: [v00 → R0   v01 → R1   v02 → R2   v03 → R3]
R1: [v10 → R0   v11 → R1   v12 → R2   v13 → R3]
R2: [v20 → R0   v21 → R1   v22 → R2   v23 → R3]
R3: [v30 → R0   v31 → R1   v32 → R2   v33 → R3]
```

After A2A, each rank holds the column of chunks destined for it:

```
Final state:

R0: [v00  v10  v20  v30]    ← all chunks whose destination was R0
R1: [v01  v11  v21  v31]
R2: [v02  v12  v22  v32]
R3: [v03  v13  v23  v33]
```

The (source, destination) matrix is transposed along the diagonal. Two α-β-equivalent software schedules implement this transpose — §7.1 derives the **ring relay** (the natural fit on bisection-limited fabrics such as torus / hypercube / on-chip rings), §7.2 derives the **pairwise direct-send** schedule that NCCL ships on switched fabrics (fat-tree / Clos / NVSwitch), and §7.3 compares both against Bruck's $O(\log N)$-latency alternative (derivation in [Appendix B.5](#b5-bruck-a2a)) and explains the NCCL selection rule.

### 7.1 Ring A2A

The ring relay runs A2A on the same wiring as ring AR's Phase 1 / 2 from §5.1 — but with both directions of the full-duplex ring in use at every step. Shortest-arc routing sends each chunk the shorter way around the ring (right if the forward distance $d(i, j) = (j - i) \bmod N \le N/2$, left otherwise); intermediate ranks forward hop-by-hop when $d > 1$. We walk through $N = 4$.

**Setup.** Four ranks $R_0, R_1, R_2, R_3$ wired into a bi-directional ring — each rank concurrently drives both its right-edge and left-edge links every step:

```
      ┌──────────────────┐
      ▼                  │
      R0 ⇄ R1 ⇄ R2 ⇄ R3
      │                  ▲
      └──────────────────┘
```

Each rank $R_i$ starts with 4 chunks $\\{v_{i,0}, v_{i,1}, v_{i,2}, v_{i,3}\\}$ (same initial layout as the §7 intro); target is to deliver each $v_{i,j}$ to $R_j$.

**Per-rank schedule.** Shortest-arc routing assigns each of $R_i$'s three foreign chunks to one direction: $v_{i, i+1}$ goes 1 hop right, $v_{i, i+2}$ goes 2 hops right (antipode tie, broken rightward), $v_{i, i-1}$ goes 1 hop left. Each rank ships 2 chunks rightward (one direct, one via a single relay hop) and 1 leftward over the $N{-}1 = 3$ steps:

| Step | Right-edge send | Left-edge send |
|------|-----------------|----------------|
| 1 | inject $v_{i, i+1}$ (1 hop; arrives at $R_{i+1}$) | inject $v_{i, i-1}$ (1 hop; arrives at $R_{i-1}$) |
| 2 | inject $v_{i, i+2}$ (hop 1 of 2; parks at $R_{i+1}$) | idle |
| 3 | forward $v_{i-1, i+1}$ (received at step 2 from $R_{i-1}$; arrives at $R_{i+1}$) | idle |

Starting state (same as §7 intro):

```
R0: {v00, v01, v02, v03}
R1: {v10, v11, v12, v13}
R2: {v20, v21, v22, v23}
R3: {v30, v31, v32, v33}
```

**After step 1** — every 1-hop chunk arrives. Each rank receives two chunks, one from each neighbor:

```
Right sends:  R0→R1: v01    R1→R2: v12    R2→R3: v23    R3→R0: v30
Left sends:   R0→R3: v03    R1→R0: v10    R2→R1: v21    R3→R2: v32

R0 holds: {v00, v10,      v30}      ← missing v20 (antipode, 2 hops away)
R1 holds: {v01, v11, v21     }      ← missing v31
R2 holds: {     v12, v22, v32}      ← missing v02
R3 holds: {v03,      v23, v33}      ← missing v13
```

**After step 2** — every rank injects its antipode (2-hop) chunk onto the right channel. The chunks park at the intermediate rank with one more hop to go. Left channel is idle — all leftward chunks already delivered at step 1.

```
Right sends:  R0→R1: v02    R1→R2: v13    R2→R3: v20    R3→R0: v31
(left channel idle)

Mid-relay state — chunks parked at intermediate ranks:
  v02 at R1 (destined R2)      v13 at R2 (destined R3)
  v20 at R3 (destined R0)      v31 at R0 (destined R1)

(No ranks' "missing" set changes at step 2 — every received chunk is mid-relay.)
```

**After step 3** — each parked chunk makes its final hop by being forwarded rightward:

```
Forwards:     R0→R1: v31    R1→R2: v02    R2→R3: v13    R3→R0: v20
(left channel idle)

R0: {v00, v10, v20, v30}    ✓
R1: {v01, v11, v21, v31}    ✓
R2: {v02, v12, v22, v32}    ✓
R3: {v03, v13, v23, v33}    ✓
```

A2A complete. Each right edge carried 3 chunks of size $M/N$ each (one per step); each left edge carried 1 chunk at step 1 and was idle afterward. At larger $N$ the right/left split becomes more balanced (more chunks fall on the left half of the shortest-arc partition), but the busier direction always sets the BW bound.

**Cost accounting.** The bottleneck (right-edge) link carries $N{-}1$ chunks of size $M/N$ across the $N{-}1$ sequential steps; each step costs $\alpha + (M/N)/\mathrm{BW}$. Summing:

$$t_{\mathrm{ring\\,A2A}} \\;=\\; (N{-}1)\\,\alpha \\;+\\; \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}$$

**Asymptotic form (bandwidth-bound regime; intrinsically pipelined implementation).** Like ring AR (§5.1) and ring AG / RS (§6), ring A2A's closed form above **is already the pipelined asymptote** with $P = N$. The $M/N$ chunked payload bakes pipelining into the derivation; the $N{-}1$-step schedule is not a $P = 1$ baseline that gets pipelined later, it is the pipelined schedule. Three structural reasons no further collapse is possible:

- **No fill or drain.** Every rank starts with all $N$ of its chunks, so at step 1 every rank is already concurrently driving both its right-edge and left-edge links — steady state from step 1 to step $N{-}1$. No fill phase to amortize.
- **$P > N$ cannot improve the BW floor.** The bottleneck right-edge link carries $(N{-}1) \cdot M/N$ total bytes across the collective — matching the per-rank BW lower bound for A2A (each rank must ship $(N{-}1)/N$ of its $M$ bytes to other ranks). Finer segmentation shrinks per-step payload but not total bytes per link; the BW coefficient $(N{-}1)/N$ is a true lower bound, not an asymptote approached in a limit.
- **No $O(\sqrt{M})$ correction.** The [Appendix C](#appendix-c-asymptotic-form-of-linear-schedules-via-pipelining) master formula's $\sqrt{M}$ correction originates in the $L{-}1$ idle steps while a single-source pipeline fills. Ring A2A has no single source and no fill, so the closed form is exact under the conflict-free ring assumption, not an $\approx$-approximation.

The schedule lands naturally on **bisection-limited fabrics** — torus, hypercube, pure-ring / NVLink-ring interconnects — where not every rank pair has a direct fabric path, because each rank drives only its two adjacent links (well within any fabric's port budget). On switched fabrics with full bisection, the alternative pairwise schedule (§7.2) matches this α-β cost without the intermediate-hop relay.

### 7.2 Pairwise direct-send A2A

Pairwise direct-send runs A2A as $N{-}1$ rounds of simultaneous send/receive between rank pairs — step $t$ uses partner offset $+t$, so each chunk crosses exactly one fabric hop (no intermediate relay). This requires a fabric where every rank pair has a direct path: fat-tree / Clos / NVSwitch.

**Setup.** Four ranks $R_0, R_1, R_2, R_3$ on a full-bisection fabric. At step $t \in \\{1, 2, 3\\}$ every rank concurrently sends to partner $(i+t) \bmod N$ and receives from partner $(i-t) \bmod N$ over the full-duplex link. Step 1 uses offset $+1$, step 2 offset $+2$, step 3 offset $+3$:

```
step 1 (offset +1)     step 2 (offset +2)     step 3 (offset +3)

  R0 ──→ R1              R0 ──→ R2              R0 ──→ R3
  R1 ──→ R2              R1 ──→ R3              R1 ──→ R0
  R2 ──→ R3              R2 ──→ R0              R2 ──→ R1
  R3 ──→ R0              R3 ──→ R1              R3 ──→ R2

(each step: one send + one receive per rank, full-duplex; every
 concurrent send goes to a distinct switch port — no contention)
```

Each rank $R_i$ starts with 4 chunks $\\{v_{i,0}, v_{i,1}, v_{i,2}, v_{i,3}\\}$ (same initial layout as the §7 intro); target is to deliver each $v_{i,j}$ to $R_j$.

**After step 1** (offset $+1$) — each rank ships its "right-neighbor" chunk and receives from its "left-neighbor".

```
Sends:  R0→R1: v01    R1→R2: v12    R2→R3: v23    R3→R0: v30

R0: {v00,      v02, v03,  v30}    ← received v30 from R3
R1: {v01, v10, v11,      v13}     ← received v01 from R0
R2: {     v12, v20, v21, v22}     ← received v12 from R1
R3: {          v23, v31, v32, v33}    ← received v23 from R2
```

Each rank now holds 2 of its 4 final chunks (its own self-chunk plus one received).

**After step 2** (offset $+2$) — each rank ships its "2-away" chunk directly to the antipode.

```
Sends:  R0→R2: v02    R1→R3: v13    R2→R0: v20    R3→R1: v31

R0: {v00,           v03,  v20, v30}    ← received v20 from R2
R1: {v01, v10, v11,             v31}   ← received v31 from R3
R2: {     v02, v12,      v21, v22}     ← received v02 from R0
R3: {          v13, v23,      v32, v33}    ← received v13 from R1
```

**After step 3** (offset $+3$) — each rank ships its last remaining foreign chunk.

```
Sends:  R0→R3: v03    R1→R0: v10    R2→R1: v21    R3→R2: v32

R0: {v00, v10, v20, v30}    ← received v10 from R1
R1: {v01, v11, v21, v31}    ← received v21 from R2
R2: {v02, v12, v22, v32}    ← received v32 from R3
R3: {v03, v13, v23, v33}    ← received v03 from R0
```

Every rank now holds its column of the transpose — A2A complete in $N{-}1 = 3$ steps.

**Cost accounting.** Each of the $N{-}1$ steps costs $\alpha + (M/N)/\mathrm{BW}$ (one handshake plus one $M/N$-sized chunk over the full-duplex link). The steps run sequentially (step $t+1$ cannot overlap step $t$'s send/receive on the same endpoint):

$$t_{\mathrm{pairwise\\,A2A}} \\;=\\; (N{-}1)\\,\alpha \\;+\\; \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}$$

Identical α-β form to the ring relay (§7.1) — both schedules hit the $(N{-}1)/N \cdot M/\mathrm{BW}$ BW lower bound. The practical distinction is routing: pairwise requires full bisection (no chunk traverses more than one fabric hop), ring does not.

**Asymptotic form (bandwidth-bound regime; already exact).** The closed form above is exact under the full-bisection assumption — there is nothing to pipeline away. The three structural reasons parallel ring A2A (§7.1):

- **No fill or drain.** Every rank drives one send and one receive on a full-duplex link from step 1; the $N{-}1$ permutation rounds are steady state throughout, with no single-source pipeline to fill.
- **Segmenting $M$ does not improve the BW floor.** Each rank ships $(N{-}1)/N \cdot M$ bytes total across $N{-}1$ distinct partner rounds — the per-rank BW lower bound for A2A. Splitting each round's $M/N$ chunk into finer sub-chunks replaces one handshake with several (more α, same bytes), and so cannot beat the floor.
- **No $O(\sqrt{M})$ correction.** No single source, no fill — the [Appendix C](#appendix-c-asymptotic-form-of-linear-schedules-via-pipelining) pipelining correction is absent; the formula is exact under the conflict-free bisection assumption.

**NCCL implementation.** NCCL ships pairwise direct-send via its staggered P2P scheduler (`scheduleP2pTasksToPlan`), which offsets the per-rank partner order by rank index so step 1 routes $R_0 \to R_1, R_1 \to R_2, \ldots$ (adjacent offsets), step 2 routes offset $+2$, and so on — spreading concurrent sends across distinct switch-port pairs and avoiding per-step head-of-line blocking on any single port. The pairwise schedule is the NCCL default on scale-up NVSwitch fabrics and on switched scale-out fabrics (fat-tree / Clos); the ring relay surfaces only on bisection-limited topologies whose fabric physically lacks a direct path between every rank pair.

Workloads that run A2A back-to-back (e.g., MoE dispatch followed by a reverse A2A combine) double the total, giving $2(N{-}1)\alpha + 2(N{-}1)M/(N\\,\mathrm{BW})$ for the round trip under either schedule.

### 7.3 Comparison and practical adoption

Two software forms sit on the table: ring relay (§7.1) and pairwise direct-send (§7.2). Unlike AR (§5.3) there is no hardware in-network option — switch primitives (SHARP / NVLS / Tomahawk Ultra INC) cannot accelerate A2A's BW side, for structural reasons summarized at the end of this section. One MPI-era variant — Bruck — is omitted from the table because NCCL / RCCL do not ship it; its $O(\log N)$-latency / $O(\log N)$-BW-coefficient derivation and the single-algorithm-default reasoning that rules it out of GPU collectives live in [Appendix B.5](#b5-bruck-a2a).

| Algorithm | α term | BW term | Regime / adoption |
|---|---|---|---|
| Ring relay (§7.1) | $(N{-}1)\\,\alpha$ | $\dfrac{N-1}{N} \cdot \dfrac{M}{\mathrm{BW}}$ | Bisection-limited fabrics (torus, hypercube, NVLink-ring); $N$-hop relay lands naturally on fabrics without full bisection |
| Pairwise direct-send (§7.2) | $(N{-}1)\\,\alpha$ | $\dfrac{N-1}{N} \cdot \dfrac{M}{\mathrm{BW}}$ | Full-bisection fabrics (fat-tree / Clos / NVSwitch). **NCCL default** |

**Reading the rows as a progression.** Ring relay and pairwise direct-send are α-β-equivalent — same $(N{-}1)\alpha$ latency, same BW-optimal $(N{-}1)/N$ coefficient — so there is no α-β crossover between them; the choice is dictated by the fabric's bisection, not the message size. Pairwise needs every rank pair to have a direct path (no intermediate relay), so it runs on switched / full-bisection fabrics; ring ships the same α-β cost on bisection-limited topologies by using both directions of the ring and forwarding through intermediate ranks. The topology dimension matters more here than in §5.3 or §6 for one reason specific to A2A:

- **A2A is the collective most likely to be bandwidth-bound on a bisection-constrained fabric.** The aggregate $(N{-}1)M$ cross-fabric traffic has to pass through the bisection of whatever physical topology the $N$ ranks sit on. On topologies with limited bisection (torus), this bound turns A2A into the rate-limiting step for permutation-heavy workloads (MoE, distributed FFT, distributed sort) — and drives physical-layout decisions about which ranks sit inside which high-BW island. The topology discussion in `02_topology_mapping.md` makes this quantitative.

**Why NCCL ships pairwise (and not ring relay) on scale-up.** Scale-up NVLink / NVSwitch fabrics offer full bisection, so both schedules are feasible — and NCCL picks pairwise. Three practical reasons the α-β model misses:

- **Zero relay overhead.** Ring relay requires intermediate ranks to forward chunks hop-by-hop (step 3 at $N=4$ forwards step-2 arrivals); every relay is an extra kernel entry and SM-side copy. Pairwise is zero-copy endpoint-to-endpoint — each chunk crosses one fabric hop via a single DMA.
- **Switch port-spreading.** Pairwise's staggered partner offsets (`scheduleP2pTasksToPlan`) route concurrent sends across distinct NVSwitch port pairs, so no switch port sees more than one outgoing chunk per step. Ring relay concentrates all traffic on each rank's two ring neighbors; on NVSwitch that's two fixed ports, and head-of-line blocking between concurrent flows on those ports slows the whole schedule.
- **Any $N$.** Ring relay's shortest-arc routing has a tie-break asymmetry at antipode (rightward for even $N$), which makes the BW split between left and right edges unequal at small $N$ (left-channel idle after step 1 at $N=4$); pairwise is symmetric at every $N$.

The ring relay remains the schedule of choice where the fabric forces it — pure-ring or torus topologies without full bisection — which is exactly its role in `02_topology_mapping.md §3.3`.

**Switch primitives don't help A2A on the BW side.** Unlike BC / AR / AG, A2A benefits from neither the switch-ALU reduction primitive nor switch-multicast replication: the $N(N{-}1)$ per-destination payloads are all distinct (no shared payload for the switch to multicast) and there is no reduction (no aggregation for the switch ALU to compute). The BW term is structurally fixed at $(N{-}1)/N \cdot M/\mathrm{BW}$ per rank regardless of switch capability. Hardware A2A primitives (Tomahawk Ultra INC, Rubin-generation NVSwitches) collapse the α side only — one switch-crossbar operation instead of $(N{-}1)$ endpoint-scheduled ones — not the BW side. See `04_in_network_collectives.md §1.1–§1.3` for the structural argument and per-primitive treatment.

---

## 8. Point-to-point hop

A single send/recv between two ranks — the degenerate $N=2$ case of any collective, shown for completeness in the same style as §§3–7.

**Worked example.** Sender $R_A$ holds payload $v$ (size $M$ bytes); receiver $R_B$ is empty. One step, one link, one $M$-byte transfer:

```
Initial:
R_A: {v}
R_B: { }

Step 1 — R_A sends v to R_B over the link:

R_A: {v}        ← kept (send is non-destructive)
R_B: {v}        ← received
```

**Cost.** One handshake plus $M$ bytes on a single link:

$$t_{\mathrm{P2P}} = \alpha + \frac{M}{\mathrm{BW}}$$

Trivial, but it shows up once per pipeline step: pipeline parallelism (PP) passes activations stage-by-stage as a chain of P2P hops. If the PP stage count is $P$, the per-step PP cost is $\alpha + M_{\mathrm{PP}} / \mathrm{BW}$ per hop, contributing to the stage-level comm time.

---

## 9. Mapping primitives to DP, TP, EP, SP, PP

Large-model distributed execution uses up to five orthogonal parallelism axes, each mapped to a collective primitive. Four of them (TP/SP/EP/PP) contribute cost **inside every forward / decode step** and dominate inference latency; DP (data parallel) adds a **once-per-training-step** gradient AR that is absent from pure inference but often the largest single collective in training.

| Parallelism | Regime | Collective | Algorithm | α term | BW term |
|---|---|---|---|---|---|
| **DP** | training only | **AR** | Ring (§5.1) | $2(N{-}1)\\,\alpha$ | $2 \cdot (N{-}1)/N \cdot M/\mathrm{BW}$ |
| **DP** | training only | **AR** | DBT (§5.2) | $2\\,\lceil \log_2 N \rceil \\, \alpha$ | $\lceil \log_2 N \rceil \cdot M/\mathrm{BW}$ |
| **DP** | training only | **AR** | hw INC — SHARP / NVLS (§5.3) | $2\\,\alpha_\mathrm{switch}$ | $M/\mathrm{BW}$ |
| **TP** | train + infer | **AR** | Ring (§5.1) | $2(N{-}1)\\,\alpha$ | $2 \cdot (N{-}1)/N \cdot M/\mathrm{BW}$ |
| **TP** | train + infer | **AR** | DBT (§5.2) | $2\\,\lceil \log_2 N \rceil \\, \alpha$ | $\lceil \log_2 N \rceil \cdot M/\mathrm{BW}$ |
| **TP** | train + infer | **AR** | hw INC — SHARP / NVLS (§5.3) | $2\\,\alpha_\mathrm{switch}$ | $M/\mathrm{BW}$ |
| **SP** | train + infer | **AG** | Ring (§6) | $(N{-}1)\\,\alpha$ | $(N{-}1)/N \cdot M/\mathrm{BW}$ |
| **EP** | train + infer | **A2A** | Ring relay (§7.1, bisection-limited fabrics) | $(N{-}1)\\,\alpha$ | $(N{-}1)/N \cdot M/\mathrm{BW}$ |
| **EP** | train + infer | **A2A** | Pairwise direct-send (§7.2, full-bisection fabrics) | $(N{-}1)\\,\alpha$ | $(N{-}1)/N \cdot M/\mathrm{BW}$ |
| **PP** | train + infer | **P2P** | Single send/recv (§8) | $\alpha$ | $M/\mathrm{BW}$ |

DP is the only axis whose collective is **training-exclusive**: pure inference has no gradient to reduce, so no DP collective fires per decode step. TP / SP / EP / PP all contribute on every forward / decode step and therefore show up in inference latency as well as training iteration time. A single decoding step on a model sharded across TP/SP/EP/PP issues one AR per TP layer, one AG per SP layer, one A2A per MoE layer (dispatch + combine = two A2A), and one P2P per pipeline hop. Training adds a backward-pass AG / RS per layer plus a once-per-step DP gradient AR on the full gradient tensor (or once per gradient-accumulation window) — that DP AR runs the same ring / DBT / hw-INC path as TP but is typically the single largest collective in a training run because it ships every parameter's gradient.

### Vocabulary aside: why these mappings and not others?

- **DP → AR** (not A2A or AG): DP replicas each compute gradients on different mini-batches and must **sum** them to get the true batch gradient — a once-per-training-step AR that's absent from pure inference. Same logic as TP's partial-output reduction, just at a different granularity (whole-gradient vs per-layer partial).
- **TP → AR** (not AG): TP splits weight matrices across ranks; each rank computes a **partial** output of row-sharded attention / MLP projections that must be **summed** to produce the true output. RS+AG is the bandwidth-optimal way to do that sum.
- **SP → AG** (not AR): SP splits tokens across ranks; each rank holds **distinct** partial KV cache shards that other ranks need — no reduction, just gather.
- **EP → A2A** (not AR): MoE routing is a **permutation** — token $t$ belongs to whichever expert the router picked; it needs to physically move there. Dispatch and combine are both A2A.
- **PP → P2P** (not a collective): PP is a chain — the activation at stage $s$ only goes to stage $s+1$. No group operation, just one send/recv per hop.

---

## Appendix A: Parallel Aggregated Trees (PAT) — scale-out AG / RS

PAT [NCCL-PAT] ships in NCCL starting with the 2.23 release and is the **first NCCL-shipped collective algorithm whose designed-for operating point is the scale-out fabric** (NICs, InfiniBand / RoCE) — as distinguished from the scale-up NVLink / NVSwitch island that ring and DBT already cover well. It implements AG and RS with $O(\log N)$ latency in the node count $N$ while matching ring's bandwidth-optimal $(N{-}1)/N$ coefficient — strictly dominating ring on α-β at the scale-out regime where $N$ runs into the hundreds and per-node α is in the microseconds. The rest of this appendix motivates PAT against the §6 ring baseline, lays out the parallel-trees construction and why its specific partner order fits scale-out NIC pipelines, works through the schedule at $N = 8$, derives the cost, and closes with the NCCL dispatch rules. PAT gets its own appendix (rather than sitting alongside the non-mainline variants in [Appendix B](#appendix-b-non-mainline-ar--ag--rs--a2a-variants)) because it is the one log-depth AG / RS variant NCCL has chosen to ship, and its design is worth the same "practical adoption" treatment the main text gives ring, DBT, and switch-ALU AR.

**Why scale-out AG / RS needs a different algorithm.** At scale-out the communicator $N$ is the **node count**, not the intra-island GPU count. Large training and inference jobs run with $N$ in the hundreds to thousands of nodes. Two properties flip relative to the intra-NVLink regime §6's ring was designed for:

- **$\alpha$ is much larger.** Scale-out NIC $\alpha$ is in the microseconds (InfiniBand EDR/HDR ≈ 1–2 μs, plus a kernel-launch and proxy-thread floor on NCCL's scale-out path of a few additional μs). Ring AG at $N = 512$ nodes is $(N{-}1)\alpha \approx 511 \times 2\\,\mu$s $\approx 1$ ms of pure α — swamping most realistic per-rank $M/N$ payloads in bandwidth cost.
- **Port budget is 1–2.** Scale-out endpoints present 1–2 active NIC ports per rank (GPUDirect RDMA from one GPU typically drives one NIC, though 2–8 NICs per node exist). The partner-cycling schedule that kills rec-doubling on-node — needs $\lceil \log_2 N \rceil$ concurrent partners at steady state to pipeline, see [Appendix B.3](#b3-why-neither-ar-variant-is-shipped) — kills it even harder at scale-out.

Ring's $(N{-}1)\alpha$ dominates the collective; partner-cycling schedules serialize on the 1–2 NIC budget; the algorithm PAT ships needs to combine log-depth latency with a bounded-port, bounded-buffer pipeline. This is the niche PAT was designed for.

**The parallel-trees concept.** PAT structures an $N$-node AG as **$N$ independent binary-tree broadcasts, one per source chunk, advancing in lockstep**. Chunk $c_i$ originates at rank $R_i$ and reaches all other ranks along its own tree of depth $\lceil \log_2 N \rceil$; each tree has the same shape, just rooted at a different rank. At round $r$, every tree advances by exactly one edge, so the whole set of $N$ trees covers all $N$ ranks in $\lceil \log_2 N \rceil$ rounds.

```
PAT at N=8 — the tree rooted at R0 (broadcasting c0).
Offsets are taken in reverse Bruck order: farthest first (4), then 2, then 1.

Round 1 (offset 4):    R0 ──────────► R4
                       
Round 2 (offset 2):    R0 ──► R2           R4 ──► R6
                       
Round 3 (offset 1):    R0 → R1   R2 → R3   R4 → R5   R6 → R7

After 3 rounds: R0, R1, … R7 all hold c0.

All 8 trees (one rooted at each Ri, same shape) run concurrently — each link carries
exactly one chunk per round, which is the bounded-buffer property PAT needs.
```

Two properties fall out of the construction:

1. **Per-link payload per round is constant at $M/N$.** Unlike rec-doubling AG (payload $M/N, 2M/N, 4M/N, \ldots$), PAT does not let the exchange buffer grow. The intermediate buffer required per rank is therefore bounded by $O(M/N)$, which matters because scale-out proxy threads stage data through finite registered-memory buffers and don't have room for the full $M$.
2. **Partner order "farthest first" gets long-RTT transfers in flight early.** PAT uses the Bruck-family offset sequence ($i \pm 2^k$ for $k = 0, \ldots, \lceil \log_2 N \rceil - 1$) **in reverse order** — at $N = 8$ the partner offset sequence is $4, 2, 1$ rather than Bruck's $1, 2, 4$. On scale-out, the longest-logical-distance exchange (offset $N/2$) also corresponds to the physically most distant node pair (worst-case cross-fabric RTT). Scheduling it first means the long transfer overlaps with the later, shorter-offset transfers on the NIC's DMA engines. Bruck's "nearest first" would leave the long transfer on the critical path at the end, after the NIC has already drained its pipeline.

**Schedule at $N = 8$.** Each rank starts holding its own chunk $c_i$ of size $M/N = M/8$.

```
Round 1 (offset 4, partner i XOR 4):
  Pairs: (R0,R4), (R1,R5), (R2,R6), (R3,R7)
  Each pair exchanges its one chunk.
  After: R0 has {c0,c4}; R1 has {c1,c5}; … ; R7 has {c3,c7}.  [2 chunks each]

Round 2 (offset 2, partner i XOR 2):
  Pairs: (R0,R2), (R1,R3), (R4,R6), (R5,R7)
  Each pair exchanges one selected chunk of the "current owned" set
  (the one the other side still needs from this pair's perspective).
  After: each rank holds 4 chunks of the 8. [4 chunks each]

Round 3 (offset 1, partner i XOR 1):
  Pairs: (R0,R1), (R2,R3), (R4,R5), (R6,R7)
  Each pair exchanges one selected chunk.
  After: every rank holds all 8 chunks. [8 chunks — complete]
```

The "one selected chunk" per round per pair is determined by the chunk-to-tree assignment: round $r$ advances, for every source $s$, exactly the one edge of $s$'s broadcast tree at depth $r$. With $N$ trees each of depth $\log_2 N$ running concurrently, the round advances all trees one level at once. The bounded-buffer property follows because a rank participates in exactly one tree-edge per round per direction.

**Cost.** $\lceil \log_2 N \rceil$ sequential rounds, each shipping $M/N$ bytes per link, yielding a per-round cost of $\alpha + (M/N)/\mathrm{BW}$. Total per-rank on-wire volume is $(N{-}1)\cdot M/N$ — the AG BW lower bound — accumulated across the $\lceil \log_2 N \rceil$ rounds. Crucially the bandwidth term is **not** $\lceil \log_2 N \rceil \cdot M/N \cdot \mathrm{BW}^{-1}$ (which would be log-depth × per-round payload summed naïvely) because on a full-duplex link each rank both sends and receives each round, and across the $\lceil \log_2 N \rceil$ rounds the total data each rank actually ships out is exactly $(N{-}1) \cdot M/N$ bytes — one per chunk it doesn't originate, forwarded once on its outbound direction:

$$t_{\mathrm{PAT}} = \lceil \log_2 N \rceil \\, \alpha + \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}$$

RS is symmetric (reverse the schedule and replace forward-concat with sum-in). Compared to ring's $(N{-}1)\alpha + (N{-}1)/N \cdot M/\mathrm{BW}$: same BW coefficient, $O(\log N)$ α vs ring's $O(N)$ α — exactly the α reduction that makes sense of PAT's shipping niche.

**Practical adoption.** The NCCL tuner dispatches PAT only when the scale-out operating conditions line up; everywhere else stays on ring. Two shipping constraints define the scope:

1. **Inter-node only, 1 rank per node.** The NCCL 2.23 implementation restricts PAT to one rank per node because only the inter-node phase is implemented; intra-node AG / RS still runs ring. With $G > 1$ ranks per node, the intra-node ranks share a NIC (the scale-out port), and the tree-edge-per-round property degenerates — multiple tree-edges at the same round land on the same NIC and serialize, collapsing the log-depth advantage. The 2.23 release notes document this restriction directly; a future "multi-rank-per-node PAT" would need a hierarchical composition (intra-node ring / tree at the leaves, PAT between node leaders) that NCCL has not yet shipped.
2. **Scale-up doesn't benefit.** On a scale-up NVLink / NVSwitch domain ($N \leq 72$, $\alpha \approx 0.5\\,\mu$s), ring AG's $(N{-}1)\alpha$ is already in the tens of μs and ring's pipelining keeps BW at the floor. Replacing ring with PAT would trade pipeline-friendliness for a log-depth α schedule without any α budget to recover on the NVLink side, and it would inherit a port-budget obstruction similar to rec-doubling's (see [Appendix B.3](#b3-why-neither-ar-variant-is-shipped)) because tree-edge-per-round still requires more than the 2 NVLink ring directions at intermediate tree levels. NCCL 2.23 explicitly routes intra-node traffic through ring.

The scope is therefore: **PAT is the inter-node scale-out AG / RS algorithm at 1 rank per node**; ring covers every other AG / RS configuration. Unlike the non-mainline variants in [Appendix B](#appendix-b-non-mainline-ar--ag--rs--a2a-variants) — which never escape the MPI-menu status — PAT is the single non-ring AG / RS algorithm NCCL has chosen to ship, and the scale-out α budget is what made the engineering investment worth it.

---

## Appendix B: Non-mainline AR / AG / RS / A2A variants

Four algorithms from the MPI literature and older HPC work appear prominently in MPICH / OpenMPI's algorithm menus but are absent from the NCCL / RCCL shipping menu: simple recursive-doubling AR ([B.1](#b1-simple-recursive-doubling-ar)), Rabenseifner halving-doubling AR ([B.2](#b2-rabenseifner-halving-doubling-ar)), recursive-doubling AG / recursive-halving RS ([B.4](#b4-recursive-doubling-ag--recursive-halving-rs)), and Bruck A2A ([B.5](#b5-bruck-a2a)). Each has a clean pure α-β cost that looks competitive on paper — some strictly dominate the shipped algorithm — yet the port-budget, pipeline-compatibility, and single-default arguments rule them out on real GPU fabrics. [B.1](#b1-simple-recursive-doubling-ar) and [B.2](#b2-rabenseifner-halving-doubling-ar) derive the two AR variants at $N = 4$; [B.3](#b3-why-neither-ar-variant-is-shipped) consolidates the AR-specific "why neither ships" analysis — numerical crossovers against DBT, partner-cycling as pipeline-kill, power-of-2 and topology-adversarial secondary factors — that §5.3 only gestures at. [B.4](#b4-recursive-doubling-ag--recursive-halving-rs) specializes the same schedule family to standalone AG / RS and carries the scale-up-ring-wins argument of §6 through explicitly; [B.5](#b5-bruck-a2a) does the analogous job for A2A and explains why NCCL's single-default picks pairwise rather than Bruck. Each subsection gives the full step-by-step trace and the cost formula cited by the main-text comparison.

### B.1 Simple recursive-doubling AR

Simple recursive-doubling AR is the one-phase butterfly / hypercube sweep: at step $k \in \\{1, \ldots, \lceil \log_2 N \rceil\\}$, every rank $i$ exchanges its **full current vector** with partner $i \oplus 2^{k-1}$ and sums the received vector into its local copy. After $\lceil \log_2 N \rceil$ steps, every rank holds the $N$-way sum. We walk through $N=4$; same initial state as §5.1 (each $R_i$ holds $V_i = [v_{i,0}, v_{i,1}, v_{i,2}, v_{i,3}]$).

**Step 1 (partner $i \oplus 1$, offset $2^0 = 1$).** Pairs: $(R_0, R_1)$ and $(R_2, R_3)$. Each pair exchanges full vectors and sums.

```
After step 1 (2-way partial sums):

R0: [v00+v10   v01+v11   v02+v12   v03+v13]
R1: [v00+v10   v01+v11   v02+v12   v03+v13]   ← identical to R0
R2: [v20+v30   v21+v31   v22+v32   v23+v33]
R3: [v20+v30   v21+v31   v22+v32   v23+v33]   ← identical to R2
```

**Step 2 (partner $i \oplus 2$, offset $2^1 = 2$).** Pairs: $(R_0, R_2)$ and $(R_1, R_3)$. Each pair again exchanges full vectors and sums — but now the "full vector" is already a 2-way partial sum from step 1.

```
After step 2 (4-way full sums):

R0: [S0   S1   S2   S3]
R1: [S0   S1   S2   S3]
R2: [S0   S1   S2   S3]
R3: [S0   S1   S2   S3]

where S_k = v_{0,k} + v_{1,k} + v_{2,k} + v_{3,k}.
```

All four ranks hold the complete reduced vector after $\lceil \log_2 4 \rceil = 2$ sequential steps.

**Cost.** Each step moves the **full** $M$-byte vector across the active link (full-duplex, so both partners send concurrently over opposite directions; per-step cost is $\alpha + M/\mathrm{BW}$). With $\lceil \log_2 N \rceil$ sequential steps:

$$t_{\mathrm{rec\\,doubling\\,AR}} = \lceil \log_2 N \rceil \\, \alpha + \lceil \log_2 N \rceil \cdot \frac{M}{\mathrm{BW}}$$

**Strengths.** Minimum latency term $\lceil \log_2 N \rceil \alpha$ — a lower bound for any reduction over a binary-combinable tree. Single-phase schedule (no separate RS / AG split), so the runtime is simpler than Rabenseifner's.

**Weakness.** The BW coefficient grows as $\log_2 N$ rather than saturating at the ring-optimal $2(N-1)/N \to 2$. At $N = 512$, that's $9 \cdot M/\mathrm{BW}$ versus ring's $\approx 2 \cdot M/\mathrm{BW}$ — a $4.5\times$ BW penalty that kills it on large-$M$ AR. The [§A.3](#a3-why-neither-variant-is-shipped) pipelining analysis further explains why this BW coefficient **cannot be rescued by segmentation** on a bounded-port fabric: each step targets a different partner, so a pipelined schedule would need $\lceil \log_2 N \rceil$ concurrent physical ports per rank, which no real scale-up or scale-out fabric provides.

### B.2 Rabenseifner halving-doubling AR

Rabenseifner's halving-doubling AR (RHD) [TRG05] recognizes the AR ≡ RS + AG identity from §2 and builds each phase from a chunk-exponential hypercube schedule: the RS phase halves the per-step payload, the AG phase doubles it, and together the two phases ship only $2(N-1)/N \cdot M$ bytes per rank — matching ring's BW-optimal lower bound at only $2\lceil \log_2 N \rceil$ steps. We trace $N=4$ with the same $V_i$ as §5.1, chunked into 4 pieces.

**Phase 1 — Reduce-scatter (recursive halving, $\lceil \log_2 N \rceil$ steps).**

At step $k \in \\{1, \ldots, \lceil \log_2 N \rceil\\}$, rank $i$ pairs with partner $i \oplus 2^{k-1}$, sends the half of its chunks the partner will own at the end of this step, and receives (summing in) the complementary half. The per-step payload **halves** each round because the "chunks I still own" set halves.

**Step 1 ($k=1$, partner $i \oplus 1$).** Split the 4-chunk vector into lower-half $\\{0, 1\\}$ and upper-half $\\{2, 3\\}$. Pairs: $(R_0, R_1)$ → $R_0$ keeps lower, $R_1$ keeps upper. $(R_2, R_3)$ → $R_2$ keeps lower, $R_3$ keeps upper. Each rank sends $M/2$ bytes.

```
After RS step 1:

R0: [v00+v10   v01+v11    ?         ?      ]   (owns {0,1}; sent {2,3} to R1)
R1: [   ?         ?     v02+v12  v03+v13  ]   (owns {2,3})
R2: [v20+v30   v21+v31    ?         ?      ]   (owns {0,1})
R3: [   ?         ?     v22+v32  v23+v33  ]   (owns {2,3})
```

**Step 2 ($k=2$, partner $i \oplus 2$).** Each pair subdivides its half again. $(R_0, R_2)$ both own $\\{0, 1\\}$; $R_0$ keeps chunk $0$, $R_2$ keeps chunk $1$. $(R_1, R_3)$ both own $\\{2, 3\\}$; $R_1$ keeps chunk $3$, $R_3$ keeps chunk $2$. Each rank sends $M/4$ bytes.

```
After RS step 2 (full 4-way sums in each rank's one owned chunk):

R0: [S0     ?     ?     ?  ]   ← chunk 0 fully reduced
R1: [ ?     ?     ?    S3  ]   ← chunk 3 fully reduced
R2: [ ?    S1     ?     ?  ]   ← chunk 1 fully reduced
R3: [ ?     ?    S2     ?  ]   ← chunk 2 fully reduced
```

This is the RS end-state. Each rank owns exactly one fully-reduced $M/N$ chunk.

**Phase 2 — All-gather (recursive doubling, $\lceil \log_2 N \rceil$ steps).** The same partner schedule in **reverse order**: step 1 uses partner $i \oplus 2$ (last RS partner), step 2 uses partner $i \oplus 1$ (first RS partner). Each step **doubles** the chunks-owned set; payload grows from $M/N$ to $2M/N$, $\ldots$, to $M/2$ by the last step.

**Step 1 ($k=1$ of AG, partner $i \oplus 2$).** $R_0$ sends its $S_0$ to $R_2$; $R_2$ sends its $S_1$ to $R_0$. Pair $(R_1, R_3)$ similarly exchanges $\\{S_3, S_2\\}$. Each rank sends $M/4$.

```
After AG step 1:

R0: [S0   S1    ?    ?]
R1: [ ?    ?   S2   S3]
R2: [S0   S1    ?    ?]
R3: [ ?    ?   S2   S3]
```

**Step 2 ($k=2$ of AG, partner $i \oplus 1$).** $R_0$ sends its $\\{S_0, S_1\\}$ half to $R_1$; $R_1$ sends its $\\{S_2, S_3\\}$ half to $R_0$. Symmetric for $(R_2, R_3)$. Each rank sends $M/2$.

```
After AG step 2:

R0: [S0   S1   S2   S3]
R1: [S0   S1   S2   S3]
R2: [S0   S1   S2   S3]
R3: [S0   S1   S2   S3]
```

AR done in $2\lceil \log_2 4 \rceil = 4$ sequential steps.

**Cost accounting.** Per-step payloads follow a geometric schedule:

| Phase | Step | Payload | Per-step cost |
|---|---|---|---|
| RS | 1 | $M/2$ | $\alpha + (M/2)/\mathrm{BW}$ |
| RS | 2 | $M/4$ | $\alpha + (M/4)/\mathrm{BW}$ |
| AG | 1 | $M/4$ | $\alpha + (M/4)/\mathrm{BW}$ |
| AG | 2 | $M/2$ | $\alpha + (M/2)/\mathrm{BW}$ |

Total at $N=4$: $4\alpha + (M/2 + M/4 + M/4 + M/2)/\mathrm{BW} = 4\alpha + (3/2)\\,M/\mathrm{BW}$. The bandwidth series is $M \cdot \sum_{k=1}^{\log_2 N} 2^{-k} = M \cdot (N-1)/N$ for one phase, doubled for the two-phase RS+AG. Generalizing:

$$t_{\mathrm{Rabenseifner\\,AR}} = 2\lceil \log_2 N \rceil \\, \alpha + 2 \cdot \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}$$

At $N=4$ this evaluates to $4\alpha + 1.5\\,M/\mathrm{BW}$, matching the trace above.

**Strengths.** Bandwidth-optimal (same $2(N-1)/N$ BW floor as ring) at $O(\log N)$ latency instead of ring's $O(N)$. Strictly dominates both ring (same BW, fewer α) and simple recursive doubling (same α order, better BW) on α-β arithmetic. This is the "best on paper" AR algorithm for power-of-2 $N$.

**Weakness.** Same partner-cycling failure mode as simple recursive doubling — each RS step and each AG step exchanges with a different partner, so pipelining requires $\lceil \log_2 N \rceil$ concurrent ports per rank. On bounded-port fabrics (2 ring directions on NVLink, 2–8 NICs off-node, small-digit scale-up switch uplinks), the pipeline serializes and the non-pipelined BW term of $2(N-1)/N \cdot M/\mathrm{BW}$ is the best it attains. DBT (§5.2), which starts from a worse $\log_2 N \cdot M/\mathrm{BW}$ non-pipelined BW but **pipelines successfully** on a 3-port-budget fabric, matches or beats Rabenseifner's asymptote — which is why NCCL ships DBT and not Rabenseifner despite Rabenseifner's more attractive paper cost. Also strictly needs $N = 2^k$; non-power-of-2 requires a reduction-first embedding step that breaks the clean halving geometry. [§B.3](#b3-why-neither-ar-variant-is-shipped) consolidates the partner-cycling argument and the numerical crossovers against the shipped algorithms.

### B.3 Why neither AR variant is shipped

The per-subsection "Weakness" paragraphs of [B.1](#b1-simple-recursive-doubling-ar) and [B.2](#b2-rabenseifner-halving-doubling-ar) each sketch why their variant doesn't ship. This subsection consolidates those arguments — introducing the shared port-budget / partner-cycling mechanism and the numerical crossovers against shipped DBT / ring that §5.3 only gestures at — so they can be cited as one unit from the main text and from the AG / RS / A2A subsections that follow.

Simple recursive-doubling AR (B.1) and Rabenseifner halving-doubling AR (B.2) arrive at the §5.3 comparison with pure α-β costs that look competitive or better than DBT:

- **Rec-doubling:** $\lceil \log_2 N \rceil \\, \alpha + \lceil \log_2 N \rceil \cdot M/\mathrm{BW}$ — minimum possible latency term (half of DBT's) at the price of a $\log_2 N$ BW coefficient.
- **Rabenseifner:** $2\lceil \log_2 N \rceil \\, \alpha + 2(N{-}1)/N \cdot M/\mathrm{BW}$ — matches ring's BW-optimal floor at log-depth latency; strictly dominates ring on α-β arithmetic at every $(N, M)$.

Yet NCCL / RCCL ship neither. The reason is a structural mismatch between their partner schedules and the port budget of every realistic fabric tier — the same obstruction [Appendix C §C.4](#c4-port-budget-caveat) describes for linear schedules in general, specialized here to the partner-cycling family.

**The port-budget / partner-cycling failure mode.** Both variants exchange with a *different* partner at every step: the hypercube schedule $i \oplus 2^{k-1}$ cycles through $\lceil \log_2 N \rceil$ distinct partners across the algorithm. Pipelining ([Appendix C](#appendix-c-asymptotic-form-of-linear-schedules-via-pipelining)), which is the mechanism that collapses a log-depth schedule's BW coefficient from $L$ down to $\sim 1$, requires the segments-in-flight at steady state to use *physically disjoint links simultaneously* — segment $s$ heading to partner $2^{k-1}$ on one port must not queue behind segment $s{+}1$ heading to partner $2^{k}$ on the same port. That needs $\lceil \log_2 N \rceil$ concurrent physical ports per rank. Real fabrics expose ~2 NVLink ring directions on-node, 2–8 NICs off-node, and small-single-digit switch uplinks at the scale-up tier. For $N = 512$ that is $\lceil \log_2 N \rceil = 9$ concurrent partners needed — past every budget. The pipeline serializes, and each variant is stuck at its non-pipelined BW coefficient.

DBT escapes this trap because its schedule targets *at most ~3 concurrent partners per rank* at peak — a tree node's ≤ 2 children during reduce plus the sibling-tree leaf edge on the opposite direction — well inside every realistic fabric port budget. That is why DBT's non-pipelined $\lceil \log_2 N \rceil \cdot M/\mathrm{BW}$ collapses under the [Appendix C](#appendix-c-asymptotic-form-of-linear-schedules-via-pipelining) master formula to the $M/\mathrm{BW}$ α-β asymptote (§5.2), while rec-doubling and Rabenseifner cannot collapse theirs.

**Numerical crossover: rec-doubling vs DBT.** Rec-doubling's non-pipelined cost vs DBT's pipelined α-β asymptote:

$$\lceil \log_2 N \rceil \\, \alpha + \lceil \log_2 N \rceil \cdot \frac{M}{\mathrm{BW}} \\;\\;\text{vs}\\;\\; 2\lceil \log_2 N \rceil \\, \alpha + \frac{M}{\mathrm{BW}}$$

Rec-doubling wins when $(\lceil \log_2 N \rceil - 1) \cdot M/\mathrm{BW} < \lceil \log_2 N \rceil \\, \alpha$, i.e., at

$$M_* \\;=\\; \frac{\lceil \log_2 N \rceil}{\lceil \log_2 N \rceil - 1} \cdot \alpha \cdot \mathrm{BW} \\;\\;\longrightarrow\\;\\; M_* \to \alpha \cdot \mathrm{BW} \\;\text{as}\\; N \to \infty.$$

At $N = 512$, $\alpha = 0.5\\,\mu$s, $\mathrm{BW} = 900\\,\mathrm{GB/s}$: $M_* = 9/8 \cdot 0.5\\,\mu\mathrm{s} \cdot 900\\,\mathrm{GB/s} \approx 506\\,\mathrm{KB}$. Rec-doubling's α edge only buys anything on sub-MB messages — a regime NCCL already serves via the LL / LL128 fast-path on ring, without adding a separate algorithm. Past the crossover the $\log_2 N$ BW coefficient means a $9\times$ BW penalty at $N = 512$ versus DBT's near-1. (Practice inflation $c_{\mathrm{real}} \geq 1$ of DBT's coefficient — see §5.3 practice caveat — only shifts $M_*$ slightly; the sub-MB conclusion is robust.)

**Numerical crossover: Rabenseifner vs DBT.** Same α term ($2\lceil \log_2 N \rceil \\, \alpha$ on both sides), so the comparison is purely on the BW coefficient: Rabenseifner's $2(N{-}1)/N \to 2$ vs DBT pipelined's $1$. DBT wins at every $M > 0$ on α-β — the margin is a factor of $\sim 2$ at large $N$. Practice inflation $c_{\mathrm{real}} \geq 1$ of DBT's coefficient (see §5.3 practice caveat) closes this margin somewhat but, for $c_{\mathrm{real}} < 2(N{-}1)/N$, keeps DBT ahead. Rabenseifner's paper dominance over ring and over non-pipelined DBT never materializes for the shipping choice because the algorithm DBT is compared against is the *pipelined* DBT, which is what NCCL actually runs. Had Rabenseifner been able to pipeline, it would drop below $2(N{-}1)/N$ toward its own $M/\mathrm{BW}$ floor and the comparison would tighten further — but it cannot, for the port-budget reason above.

**Compared to ring.** Ring (§5.1) is intrinsically pipelined with $P = N$ and achieves its $2(N{-}1)/N \to 2$ BW floor for free, without any of the port-budget gymnastics. Rabenseifner matches ring's BW floor with a log-depth α (an improvement), but without pipelining it only reaches that floor, not below. Rec-doubling's $\log_2 N$ BW coefficient is worse than ring's $\sim 2$ at every $N \geq 8$, so rec-doubling is ring-dominated at large $M$ and DBT-dominated at small $M$ — no regime where it is the right choice.

**Two secondary factors** reinforce the choice:

1. **Power-of-2 restriction.** Both variants strictly require $N = 2^k$; non-power-of-2 rank counts need a reduction-first embedding step (pair off $2\lfloor N/2 \rfloor$ ranks into virtual pairs, run on $2^{\lfloor \log_2 N \rfloor}$, re-distribute) that breaks the clean halving geometry and adds 2 extra α hops plus an asymmetric BW term. NCCL routinely runs on $N = 6$, $N = 72$ (NVL72), and other non-power-of-2 counts.
2. **Topology-adversarial partner sequence.** The $i \oplus 2^{k-1}$ partner at the largest $k$ crosses the communicator bisection on every fabric, whereas tree and ring schedules have clean dim-decomposed embeddings on torus / fat-tree that keep most traffic local (see `02_topology_mapping.md`). Partner-cycling repeatedly pays the worst-case link latency that dim-decomposed schedules amortize.

Together: partner-cycling kills the pipeline, DBT's pipelined asymptote dominates Rabenseifner on BW and rec-doubling on BW for all but sub-MB messages, the α edge rec-doubling carries on those sub-MB messages is already served by the LL fast-path on ring, and the non-power-of-2 / topology issues add further friction. NCCL / RCCL ship ring and DBT; rec-doubling and Rabenseifner stay in the MPI menu as reference points.

The port-budget / partner-cycling argument above generalizes beyond AR. The next two subsections specialize it: [B.4](#b4-recursive-doubling-ag--recursive-halving-rs) applies it to standalone AG / RS (where Rabenseifner's two phases become two independent log-depth algorithms), and [B.5](#b5-bruck-a2a) applies the single-default reasoning to the one A2A variant with log-depth α — Bruck's — where pipelining is not even the central objection because no BW-optimal log-depth A2A exists for pipelining to rescue.

### B.4 Recursive-doubling AG / recursive-halving RS

Recursive-doubling AG is exactly Phase 2 of Rabenseifner AR in [B.2](#b2-rabenseifner-halving-doubling-ar), run standalone: each rank starts with one $M/N$-byte chunk and ends holding the full concatenation. Recursive-halving RS is the dual (Phase 1 of Rabenseifner run standalone): each rank starts with the full $M$-byte vector and ends holding one $M/N$-byte reduced chunk. Both use the same hypercube partner schedule ($i \oplus 2^{k-1}$ at step $k$) with the payload doubling in AG and halving in RS across $\lceil \log_2 N \rceil$ steps.

**AG trace at $N = 4$.** Start: $R_i$ holds only its own chunk $c_i$ of size $M/4$.

**Step 1 ($k = 1$, partner $i \oplus 1$).** Pairs $(R_0, R_1)$ and $(R_2, R_3)$ each exchange their one chunk. After: each rank holds 2 chunks ($M/2$ total).

```
R0: [c0   c1    ?    ?]
R1: [c0   c1    ?    ?]
R2: [ ?    ?   c2   c3]
R3: [ ?    ?   c2   c3]
```

**Step 2 ($k = 2$, partner $i \oplus 2$).** Pairs $(R_0, R_2)$ and $(R_1, R_3)$ each exchange their 2-chunk block. After: every rank holds all 4 chunks.

```
R0: [c0   c1   c2   c3]
R1: [c0   c1   c2   c3]
R2: [c0   c1   c2   c3]
R3: [c0   c1   c2   c3]
```

AG done in $\lceil \log_2 4 \rceil = 2$ steps. RS is the mirror image: start from a full vector, run the same partner schedule in reverse order (step 1 partner $i \oplus 2$, step 2 partner $i \oplus 1$), and the payload **halves** each step so each rank ends owning one fully-reduced $M/N$ chunk.

**Cost.** The per-step payload follows a geometric schedule. For AG with $\lceil \log_2 N \rceil$ steps, step $k$ ships $2^{k-1} \cdot M/N$ bytes per rank per direction; total per-rank ship = $M/N \cdot \sum_{k=1}^{\log_2 N} 2^{k-1} = (N-1) \cdot M/N$ bytes — matching ring's BW lower bound. RS is symmetric. Per-step cost $\alpha + 2^{k-1} M / (N \\, \mathrm{BW})$; summing:

$$t_{\mathrm{rec\\,doub\\,AG}} = t_{\mathrm{rec\\,halv\\,RS}} = \lceil \log_2 N \rceil \\, \alpha + \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}$$

where $M$ is the per-rank final total volume (for AG) or initial total volume (for RS).

**Strengths.** BW-optimal (matches ring) at $O(\log N)$ latency instead of ring's $O(N)$. On paper, strictly dominates ring AG / RS on α-β arithmetic for all $(N, M)$.

**Weakness.** The same partner-cycling issue from Rabenseifner AR applies: each step targets a different partner, so pipelining fails on bounded-port fabrics and the non-pipelined log-depth BW term of $(N-1)/N \cdot M/\mathrm{BW}$ is the best this algorithm attains. On a 2-port-budget fabric (NVLink pair, 2-NIC scale-out), pipelining ring ($P = N$ segments on the 2 neighbor links) already saturates the asymptotic floor; switching to rec-doubling gains the $\log_2 N \to N-1$ α reduction but at the cost of a pipeline-unfriendly schedule that can't amortize that α into the BW path. Also, the geometric payload makes per-round chunk sizes irregular, which complicates overlap with compute. Needs $N = 2^k$. For all these reasons NCCL does not ship rec-doubling AG / rec-halving RS; MPI (MPICH, OpenMPI) ships it and dispatches by message-size threshold.

The one AG / RS variant NCCL *does* ship — Parallel Aggregated Trees (PAT) — solves this problem by keeping the per-round payload constant at $M/N$ (bounded buffer, pipeline-friendly) at the cost of a more involved chunk-to-tree assignment; see [Appendix A](#appendix-a-parallel-aggregated-trees-pat--scale-out-ag--rs) for the full derivation and shipping rules. PAT and rec-doubling AG use the same partner-offset family but in opposite orders: rec-doubling takes offsets $1, 2, 4, \ldots$ with exponentially growing per-round buffers; PAT takes $N/2, \ldots, 2, 1$ with a constant $M/N$ per-round buffer. The difference is precisely what moves PAT into NCCL's shipping menu and keeps rec-doubling out.

### B.5 Bruck A2A

For A2A, the log-depth partner-cycling variant is Bruck's algorithm [BHKUW97]. The partner-cycling pipeline-kill argument from [B.3](#b3-why-neither-ar-variant-is-shipped) applies here too, but it is not the central objection: A2A has no natural BW-optimal log-depth algorithm for pipelining to rescue — the inflated BW coefficient is intrinsic to the bit-butterfly schedule, not a pipelining artifact. What rules Bruck out of NCCL is the single-algorithm-per-primitive default, which picks pairwise (§7.2) for its large-$M$ BW optimality and accepts the larger α term at small $M$ rather than shipping a second algorithm with a runtime dispatch threshold.

Bruck's A2A achieves $\lceil \log_2 N \rceil$ rounds — vs pairwise's $N-1$ — by running a butterfly / hypercube pattern on the bits of the destination index, combined with a pre-rotation and a post-rotation of the local buffer that together turn "send chunk to rank $j$" into "send the slots of my buffer whose current index has bit $k$ set to partner $i + 2^k$". The cost is a $\log_2 N / 2 \cdot M/\mathrm{BW}$ BW term — inflated relative to pairwise's $(N-1)/N \cdot M/\mathrm{BW}$ — plus $O(M)$ local rotation work that's not on the wire. We trace the schedule at $N = 4$ and give the cost.

**Pre-rotation.** Before any on-wire communication, rank $i$ cyclically rotates its send buffer **left by $i$ positions** (i.e., the chunk originally at slot $j$ moves to slot $(j - i) \bmod N$). The purpose is to align the "bit-$k$-of-destination-offset" pattern across ranks so that a single partner offset $+2^k$ per round suffices.

```
Initial (slots indexed by destination):       After pre-rotation (rank i shifts left by i):

R0: [v00  v01  v02  v03]                      R0: [v00  v01  v02  v03]   (i=0: no shift)
R1: [v10  v11  v12  v13]                      R1: [v11  v12  v13  v10]   (i=1: shift left by 1)
R2: [v20  v21  v22  v23]                      R2: [v22  v23  v20  v21]   (i=2: shift left by 2)
R3: [v30  v31  v32  v33]                      R3: [v33  v30  v31  v32]   (i=3: shift left by 3)
```

**Communication rounds ($\lceil \log_2 N \rceil = 2$ at $N=4$).** At round $k \in \\{0, 1, \ldots, \lceil \log_2 N \rceil - 1\\}$, rank $i$ sends to partner $(i + 2^k) \bmod N$ and receives from $(i - 2^k) \bmod N$. The chunks sent at round $k$ are the slots of the current buffer whose index has bit $k$ set ($N/2$ slots, total $M/2$ bytes per round); the chunks at the other $N/2$ slots stay put. The partner's corresponding slots overwrite the local ones on receive.

**Round 0 (partner $i+1 \bmod 4$, send slots with bit 0 set — i.e., slots 1 and 3, $M/2$ bytes).**

```
Sends (slots 1 and 3 of current buffer):
  R0 → R1: [v01, v03]          R1 → R2: [v12, v10]
  R2 → R3: [v23, v21]          R3 → R0: [v30, v32]

After round 0 (slots 1 and 3 overwritten by incoming):

R0: [v00  v30  v02  v32]        (slots 1,3 from R3)
R1: [v11  v01  v13  v03]        (slots 1,3 from R0)
R2: [v22  v12  v20  v10]        (slots 1,3 from R1)
R3: [v33  v23  v31  v21]        (slots 1,3 from R2)
```

**Round 1 (partner $i+2 \bmod 4$, send slots with bit 1 set — i.e., slots 2 and 3, $M/2$ bytes).**

```
Sends (slots 2 and 3 of current buffer):
  R0 → R2: [v02, v32]          R1 → R3: [v13, v03]
  R2 → R0: [v20, v10]          R3 → R1: [v31, v21]

After round 1 (slots 2 and 3 overwritten by incoming):

R0: [v00  v30  v20  v10]        (slots 2,3 from R2)
R1: [v11  v01  v31  v21]        (slots 2,3 from R3)
R2: [v22  v12  v02  v32]        (slots 2,3 from R0)
R3: [v33  v23  v13  v03]        (slots 2,3 from R1)
```

**Post-rotation.** Each rank $i$ cyclically rotates its buffer **right by $i$ positions** and optionally reverses the order (implementation detail; the net effect is to permute the received chunks into MPI / NCCL A2A output layout). After post-rotation:

```
Final (MPI/NCCL A2A output layout: slot j holds the chunk sent from R_j):

R0: [v00  v10  v20  v30]
R1: [v01  v11  v21  v31]
R2: [v02  v12  v22  v32]
R3: [v03  v13  v23  v33]
```

Matches the expected A2A end-state in §7.

**Cost.** The pre- and post-rotations are **local memory copies** (not on-wire; they do consume memory bandwidth and an intermediate buffer of size $M$, but in the α-β model we count on-wire time only). The $\lceil \log_2 N \rceil$ rounds each ship $M/2$ bytes per rank per direction, giving

$$t_{\mathrm{Bruck\\,A2A}} = \lceil \log_2 N \rceil \\, \alpha + \lceil \log_2 N \rceil \cdot \frac{M}{2\\,\mathrm{BW}}$$

At $N = 4$: $2\alpha + M/\mathrm{BW}$, compared to pairwise's $3\alpha + (3/4)M/\mathrm{BW}$ at the same $N$. Bruck wins on α; pairwise wins on BW — the crossover is at $M^* \sim \alpha \cdot \mathrm{BW}$, computed in §7.3.

**Strengths.** Minimum-latency A2A (log-depth). The BW term is $\log_2 N / 2 \cdot M/\mathrm{BW}$ rather than pairwise's $(N-1)/N \cdot M/\mathrm{BW}$, so Bruck wins at small $M$ where the α term dominates.

**Weakness.** BW term scales with $\log_2 N$ rather than saturating at the lower bound; above the $M^*$ crossover, pairwise is strictly faster. The pre/post rotations require an intermediate buffer of $O(M)$ per rank, plus $O(M)$ local-memory copies that are free in the α-β model but non-trivial in practice. The bit-test routing is cleanest at $N = 2^k$; non-power-of-2 needs padding or a modified final round. Partner cycling across the $\log_2 N$ rounds is the same bounded-port objection that killed rec-doubling AR, though A2A is less bothered by it because there is no natural bandwidth-optimal log-depth A2A for pipelining to rescue — the inflated BW coefficient is fundamental, not a pipelining artifact. NCCL's single-algorithm policy picks pairwise for its large-$M$ optimality; MPI ships both Bruck and pairwise and dispatches by message-size threshold.

---

## Appendix C: Asymptotic form of linear schedules (via pipelining)

The collectives in §3 (binomial tree BC), §4 (binomial tree Reduce), and §5 (ring AR and double binary tree AR) all cite an "asymptotic form" $L\alpha + M/\mathrm{BW}$ that collapses the otherwise-$L$-scaled bandwidth term to the single-link floor. This appendix separates two things that the main text glues together:

- **Pipelining** is the *implementation mechanism*: chunk the message into $P$ sub-segments and stream them through the $L$-step schedule like parts on an assembly line. The finite-$P$ cost is the master formula $t_\mathrm{pipe}(P) = (L + P - 1)(\alpha + M/(P\\,\mathrm{BW}))$ — exact, not asymptotic.
- **The asymptotic form** is the *cost floor* the implementation reaches in the bandwidth-bound limit $M \to \infty$ at optimal segmentation $P^*$. The two terms $L\alpha$ and $M/\mathrm{BW}$ are separate floors — each the minimum its coefficient can attain — and the sum is approached up to an $O(\sqrt{M})$ correction that vanishes relative to $M/\mathrm{BW}$.

C.1–C.4 derive the finite-$P$ master formula, its optimum $P^*$, the $\sqrt{M}$ correction between exact and asymptotic, and the port-budget caveat that licenses the collapse on real fabrics. C.5 lists how each primitive instantiates $L(N)$. Everything here applies identically to any schedule with a linear dependency chain — so future variants (larger $k$-ary trees, alternate butterfly orderings) inherit the same formulas.

### C.1 Why pipeline?

Start with the non-pipelined schedule: $L$ sequential steps, each shipping the full $M$ bytes. Total cost $L\alpha + L\\,M/\mathrm{BW}$. The ugly part is the second term — the bandwidth coefficient scales linearly with $L$. The inefficiency is easy to see: while step $k$ is busy pushing bytes, steps $1, \ldots, k{-}1$ sit idle (they finished their part of the transfer several slots ago, but there's nothing queued for them to do). If we could keep every step busy at once, the whole algorithm would finish in roughly the time of *one* step's work regardless of $L$. Pipelining is the trick that makes that happen.

### C.2 The conveyor idea and space-time diagram

Cut the message $M$ into $P$ equal sub-segments of size $M/P$ and feed them through the schedule one after another, like parts on an assembly line. Segment 1 enters step 1 at time slot 1, moves to step 2 at slot 2, then to step 3 at slot 3, and so on. Segment 2 enters step 1 at slot 2 (one slot behind segment 1) and trails it through. By the time segment $P$ has entered the pipeline, segments $1, \ldots, P{-}1$ are already distributed across the later steps — and each slot now advances *every* segment by one step simultaneously.

Rows = steps in the schedule, columns = time slots. Each slot costs exactly $\alpha + (M/P)/\mathrm{BW}$ (one handshake plus one sub-segment's worth of transfer). Cell $(k, t)$ = "segment at step $k$ during slot $t$". Small example with $L=3$ steps and $P=3$ segments:

```
              slot 1    slot 2    slot 3    slot 4    slot 5
    step 1:   seg1      seg2      seg3      ·         ·
    step 2:   ·         seg1      seg2      seg3      ·
    step 3:   ·         ·         seg1      seg2      seg3

              └──── fill ────┘    steady    └──── drain ────┘
                 (P-1 = 2)       (1 slot)        (L-1 = 2)
```

Three phases:

- **Fill** (slots 1 to $P{-}1$) — the pipeline is warming up. Step 1 has work from slot 1; step 2 doesn't start until slot 2; step $L$ doesn't start until slot $L$.
- **Steady state** (slots where all $L$ steps are busy simultaneously). In the picture: slot 3 has $\text{seg3}$ at step 1, $\text{seg2}$ at step 2, $\text{seg1}$ at step 3 — every step advancing one segment per slot. This is the regime where the pipeline pays for itself.
- **Drain** (slots $P+1$ to $L+P{-}1$) — segments that entered late are still finishing. Step 1 has nothing left to do.

**Total slot count: $L + P - 1$.** The assembly-line identity: the *last* segment (segment $P$) enters step 1 at slot $P$ and needs $L - 1$ additional slots to advance through the remaining steps, exiting at slot $P + (L-1) = L + P - 1$. In the diagram: $3 + 3 - 1 = 5$ slots. Matches.

### C.3 Master formula and optimal segmentation

**Parameters at a glance.** $L$ is the schedule's sequential-step depth (fixed by the algorithm once $N$ is chosen); $P$ is the segment count (a free knob the scheduler picks, then optimizes below). $L$ and $P$ are independent — the mapping $L(N)$ is schedule-specific:

| Schedule | $L$ | source |
|---|---|---|
| Ring BC (§3.1) | $N - 1$ | hops from source to tail rank |
| Ring Reduce (§4.1) | $N - 1$ | hops from source rank to root along chain |
| Binomial tree BC / Reduce (§3.2, §4.2) | $\lceil \log_2 N \rceil$ | tree depth |
| Ring AR (§5.1) | $2(N - 1)$ | RS pass + AG pass |
| DBT AR (§5.2) | $2 \lceil \log_2 N \rceil$ | reduce pass + broadcast pass |

The derivation below treats $L$ as given; each primitive substitutes its own $L(N)$ at the end. The full instantiation table (including per-rank port requirements) is in [C.5](#c5-instantiation-across-primitives).

Each slot costs $\alpha + (M/P)/\mathrm{BW}$; there are $L + P - 1$ of them:

$$t_{\mathrm{pipe}}(P) = (L + P - 1)\left(\alpha + \frac{M/P}{\mathrm{BW}}\right)$$

To see *where the $L$ factor goes*, expand the bandwidth piece. Using $(L{+}P{-}1)/P = 1 + (L{-}1)/P$:

$$t_{\mathrm{pipe}}(P) \\;=\\; \underbrace{(L + P - 1)\\,\alpha}_{\substack{\text{hop count}\\\\ L\alpha\\,+\\,(P-1)\alpha\\\\ \text{extra handshakes for fill/drain}}} \\;+\\; \underbrace{\frac{M}{\mathrm{BW}}}_{\substack{\text{steady-state BW}\\\\ \text{(saturates at 1}\cdot M/\mathrm{BW}\text{)}}} \\;+\\; \underbrace{\frac{(L - 1)\\,M}{P\\,\mathrm{BW}}}_{\substack{\text{fill+drain overhead}\\\\ \text{(shrinks like }1/P\text{)}}}$$

Three things to notice:

1. **The $L$-dependence has *left* the dominant BW term.** Non-pipelined BW was $L \cdot M/\mathrm{BW}$; here the $L$ survives only inside the fill/drain piece, which vanishes as $P \to \infty$.
2. **The saturated BW floor is $M/\mathrm{BW}$** — the volume a single rank-port must push regardless of $P$ or $L$. You can't go below that without using more ports.
3. **$P$ can't grow arbitrarily large**, because the hop-count term picks up $(P{-}1)\alpha$ extra handshakes. More segments = more slot boundaries = more $\alpha$.

**Sanity check with numbers.** Set $L = 3$.

- **Non-pipelined** ($P = 1$): cost $= 3\alpha + 3M/\mathrm{BW}$. BW coefficient 3.
- **Pipelined with $P = 3$**: $5\alpha + M/\mathrm{BW} + (2/3)M/\mathrm{BW} = 5\alpha + 1.67\\,M/\mathrm{BW}$. BW coefficient dropped from 3 to 1.67 at the cost of 2 extra handshakes.
- **Pipelined with $P = 10$**: $12\alpha + M/\mathrm{BW} + 0.2\\,M/\mathrm{BW} = 12\alpha + 1.2\\,M/\mathrm{BW}$. Within 20 % of the asymptotic floor, at the cost of 9 extra handshakes.
- **$P \to \infty$**: cost $\to \infty \cdot \alpha + M/\mathrm{BW}$. Asymptotic BW is achieved, but latency has exploded — clearly past the optimum.

**Optimal segmentation.** Minimize by differentiating with respect to $P$ (continuous approximation):

$$\frac{dt_{\mathrm{pipe}}}{dP} \\;=\\; \alpha \\;-\\; \frac{(L - 1)\\,M}{P^2\\,\mathrm{BW}} \\;=\\; 0 \quad\Longrightarrow\quad P^{*} \\;=\\; \sqrt{\frac{(L - 1)\\,M}{\alpha\\,\mathrm{BW}}}$$

Substituting back:

$$t_{\mathrm{pipe}}(P^{*}) \\;=\\; L\\,\alpha \\;+\\; \frac{M}{\mathrm{BW}} \\;+\\; 2\sqrt{\frac{(L - 1)\\,\alpha\\,M}{\mathrm{BW}}}$$

The square-root correction grows as $\sqrt{M}$ — much slower than the original $LM/\mathrm{BW}$ or the floor $M/\mathrm{BW}$. The punch line:

> **In the bandwidth-bound regime, pipelining collapses the BW coefficient from $L$ down to (essentially) 1, at a modest $O(\sqrt{M})$ latency-term surcharge. The base $L\alpha$ hop-count cost is untouched.**

### C.4 Port-budget caveat

The diagram above is a logical picture — "row $k$" is "step $k$ of the schedule", and each cell shows which segment is being *logically processed* there. To turn this into real wall-clock time, every cell in the steady-state column must correspond to a *physical transfer on a distinct link*. If multiple cells in the same column try to drive the **same physical port** on the same rank, only one of them actually runs; the others queue, and the pipeline collapses partway back to serial.

Counting concurrency at a single rank $R$: during steady state, $R$ is simultaneously participating in some number of steps — at most $L$, and typically the *distinct partners across the schedule's steps*. Each concurrent step corresponds to one edge out of $R$, so $R$ needs that many **concurrent physical ports** to sustain the pipeline fully. "Port" here is deliberately generic: whichever level of the fabric the collective traverses, the budget is set by the number of distinct physical links out of $R$ at *that* level. Concretely, the tight budget can be any of:

- **Scale-up on-package or on-node**: NVLink lanes / ring-direction channels out of a GPU (typically 2 ring directions or 4–18 NVLink channels per chip), on-package chiplet-to-chiplet interconnect ports, or PCIe lanes to a host switch — the scale-up fabric that stays inside a high-bandwidth island.
- **Scale-up switch ports**: the upstream ports a single endpoint presents into an NVSwitch / UALink / proprietary scale-up switch tier. A GPU typically connects with a handful of links into that switch, and the switch itself has a fixed radix.
- **Scale-out off-node**: NIC count per GPU or per node (typically 2–8 NICs on modern training nodes), which sets the per-rank port budget for inter-node collectives.
- **Mesh / torus fabrics**: on a $k$-dimensional torus each rank has exactly $2k$ neighbor ports (2 for a 1-D ring, 4 for 2-D, 6 for 3-D, etc.), fixed by the *topology* rather than by the endpoint. This is the tightest and least flexible port budget of any fabric family: an algorithm needing $\log_2 N$ concurrent partners has no hope on a 3-D torus, and even ring-family algorithms have to be carefully dimension-decomposed to stay inside the budget.

Whatever level applies, the number is $O(1)$-to-single-digit, not $O(\log N)$. Algorithms whose steady-state concurrency exceeds that number serialize on the oversubscribed port regardless of whether the oversubscription is inside a scale-up switch, on a torus neighbor link, on the scale-out NIC, or across on-package links. Torus-specific consequences — dimension-decomposed ring AR on torus, and the full mapping from the algorithms here to each topology — are worked through in `02_topology_mapping.md`.

```
Case A — schedule's per-rank port requirement fits the budget:

  slot 3 (steady state at rank R):
    step 1 edge ─────▶  port A     carrying seg3
    step 2 edge ─────▶  port B     carrying seg2
    step 3 edge ─────▶  port C     carrying seg1
                  └── 3 distinct links, 3 segments in parallel ✓
                      conveyor runs as drawn


Case B — schedule's per-rank port requirement exceeds the budget:

  slot 3 (steady state at rank R, suppose only port A exists):
    step 1 edge ───┐
    step 2 edge ───┼───▶  port A   ← only one gets served per slot
    step 3 edge ───┘                  other two queue behind it
                  └── pipeline collapses toward serial L·(α + M/BW)
```

The guiding question for any schedule is: **does the per-rank steady-state concurrency fit the fabric's port budget at every tier the collective traverses?** If yes, the pipelined formula from C.3 holds; if no, the schedule serializes back to (or toward) its non-pipelined cost.

### C.5 Instantiation across primitives

The algorithms elsewhere in this note invoke the above construction with the following parameters. "Steady-state concurrency" is the distinct-port count required per rank when the pipeline is full; "fits typical budget?" is the answer under a 2–8 port fabric tier (either 2 NVLink ring directions, 2–8 NICs, or the 3+ switch uplinks of a modern NVSwitch).

| Primitive | $L$ | Per-rank steady-state concurrency | Fits typical budget? | Pipelined BW term |
|---|---|---|---|---|
| Ring BC (§3.1) | $N-1$ | 2 neighbor links (in + out) | yes | $M/\mathrm{BW}$ |
| Ring Reduce (§4.1) | $N-1$ | 2 neighbor links (in + out) | yes | $M/\mathrm{BW}$ |
| Binomial tree BC (§3.2) | $\lceil\log_2 N\rceil$ | tree parent + up to $\lceil\log_2 N\rceil$ children at root; $\leq 2$ at interior | yes at interior; root fan-out needs $\lceil\log_2 N\rceil$ ports for full pipeline | $M/\mathrm{BW}$ interior-limited; root bound by port count |
| Binomial tree Reduce (§4.2) | $\lceil\log_2 N\rceil$ | same as BC (time-reverse dual) | same | $M/\mathrm{BW}$ |
| Ring AR (§5.1) | $2(N-1)$ | 2 neighbor links throughout | yes — intrinsically pipelined with $P = N$ | $2(N-1)/N \cdot M/\mathrm{BW}$ |
| Double binary tree AR (§5.2) | $2\lceil\log_2 N\rceil$ | $\leq 3$ (interior in one tree + leaf in sibling) | yes on any 3+ port tier | near $2\\,M/\mathrm{BW}$ in practice |
| Plain recursive doubling AR (App. B.1) | $\lceil\log_2 N\rceil$ | $\lceil\log_2 N\rceil$ distinct partners | **no** at $N = 512$ (9 partners vs 2–8 ports) | does not improve — collapses to $\log_2 N \cdot M/\mathrm{BW}$ |
| Rabenseifner AR (App. B.2) | $2\lceil\log_2 N\rceil$ | $\lceil\log_2 N\rceil$ distinct partners per phase | **no** at typical $N$ | does not improve — collapses to $2(N-1)/N \cdot M/\mathrm{BW}$ |

Two observations:

1. **Tree BC/Reduce** (§3.2 / §4.2) technically need $\lceil\log_2 N\rceil$ concurrent ports at the root to fully pipeline the initial fan-out (or final fan-in). In practice NCCL's pipelined tree implementation sustains a modest $P$ that keeps the bandwidth close to $M/\mathrm{BW}$ at mid-to-large $M$ without saturating the root's port count; the asymptotic floor is $M/\mathrm{BW}$ and the root is the tightest bottleneck.
2. **AR on partner-cycling schedules** (plain recursive doubling and Rabenseifner) is where the port-budget caveat bites hardest. Their table-form BW terms assume a pipeline that can't actually overlap on real fabrics, which is the asymmetry [Appendix B.3](#b3-why-neither-ar-variant-is-shipped) expands on and the reason NCCL ships ring + double binary tree instead.

---

## Appendix D: algbw vs busbw — the NCCL-tests vocabulary

NCCL-tests [NCCL-TESTS] and most collective-benchmark workflows report two bandwidth metrics: **algbw** (algorithm bandwidth) and **busbw** (bus bandwidth). Both derive from the same measured wall-clock time $t$ for a collective of size $M$, but they normalize differently — and picking the right one for a given question is what makes NCCL-tests output interpretable.

**algbw — application-facing throughput.** $M$ bytes delivered in time $t$, treating the collective as an opaque data transfer:

$$\mathrm{algbw} = \frac{M}{t}$$

This is the direct input to end-to-end latency budgets: a user issuing "reduce 1 GB of gradients in 10 ms" sees algbw = 100 GB/s, and that's exactly what a step-time model should plug into its communication-wall-clock term.

**busbw — link-level throughput.** The $M$-byte-equivalent traffic that flowed on any single endpoint's link during the collective. It multiplies algbw by a primitive-dependent correction factor that accounts for how many bytes each endpoint actually moved per unit of delivered work:

$$\mathrm{busbw} = \mathrm{algbw} \cdot C_{\mathrm{primitive}}(N)$$

where $C_{\mathrm{primitive}}(N)$ is the algorithm's $n_\beta$ coefficient (§1) — the per-rank byte-move count normalized by $M$. Per-primitive factors per the NCCL-tests convention [NCCL-TESTS]:

| Primitive | $C(N)$ | Rationale |
|---|---|---|
| AR | $2(N-1)/N$ | RS + AG both move $(N-1)/N \cdot M$ per rank |
| AG | $(N-1)/N$ | each rank forwards the other $N-1$ ranks' slices |
| RS | $(N-1)/N$ | each rank sends its non-own slices |
| BC | $(N-1)/N$ | root fans out, leaves receive once (asymptotic) |
| A2A | $(N-1)/N$ | each rank's one $(N-1)/N$ of its payload goes off-rank |
| Reduce | 1 | only the root's final accumulator link matters |

Example: on an $N = 100$ AR measurement with algbw $= 450\\,\mathrm{GB/s}$, busbw $= 450 \cdot 2 \cdot 99/100 \approx 891\\,\mathrm{GB/s}$ — which is the number to compare directly against the fabric's peak link BW to judge algorithmic efficiency.

**Why both metrics coexist — and when to use which.** algbw is the right number for performance modeling: step time is (compute + communication), and the communication term is $M / \mathrm{algbw}$. busbw is the right number for diagnosing *whether the fabric ceiling is reached*: it normalizes out the algorithm's structural $n_\beta$ factor (e.g., AR's 2× from the RS + AG decomposition) so the resulting number is directly comparable across primitives to the fabric's peak link BW. A collective whose busbw equals peak link BW has hit the fabric ceiling; one whose busbw is well below peak has headroom that a better algorithm or better placement might recover.

This is the measurement convention that `05_contention_and_congestion.md §4.1` uses to back out $\eta_\beta$: realized busbw divided by peak link BW gives the contention-coefficient ratio directly.

---

## Further reading

- **`02_topology_mapping.md`** — how ring / tree / log algorithms map to star (crossbar) and torus fabrics, with latency + BW derivations per topology, the torus dim-decomp AR mechanism worked out on a 2×2 example, and a side-by-side comparison at fixed $G$.
- **`04_in_network_collectives.md`** — how switch-resident reduction engines (SHARP, NVLS) replace $O(N)$ endpoint hops with $O(1)$ switch hops on star topologies.
- **`05_contention_and_congestion.md`** — how dynamic contention coefficients $\eta_\alpha \ge 1$ and $\eta_\beta \in (0, 1]$ modify the idealized costs above under realistic traffic.
- **`03_hierarchical_topologies.md`** — how the primitives above compose across tiers (RS → sub-AR → AG for AR, and AG / RS / A2A generalizations) when a cluster stacks star / torus / mesh / Clos fabrics.
- **`references.md`** — primary-source bibliography (Hockney's α-β model, Patarasuk-Yuan ring AR, Thakur-Rabenseifner-Gropp RHD, Sanders-Speck-Träff DBT, Bruck log A2A, PAT / NCCL 2.23 scale-out AG/RS, Jeaugey et al. "Demystifying NCCL" 2025).
- **Patarasuk & Yuan (2009)**, "Bandwidth optimal all-reduce algorithms for clusters of workstations" — the foundational proof that ring AR achieves the $2(N-1)/N \cdot M/\mathrm{BW}$ bandwidth lower bound.
