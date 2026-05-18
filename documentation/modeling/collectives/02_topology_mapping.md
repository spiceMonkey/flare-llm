# Mapping Collective Algorithms to Physical Topologies

**Author:** Yue Lu  
**Date:** April 2026  

The previous note (`01_collective_algorithms.md`) costs collective primitives on an abstract fully-connected fabric. Real clusters don't have that. Inside a scale-up domain they have a star topology of GPUs around an NVSwitch, a 2D / 3D torus for TPU pods, or a mesh for chiplet interposers or short-reach PCIe fabrics. This note focuses on those **single-tier, scale-up** topologies — walking through each with a 2×2-style diagram, showing how ring / tree / dim-decomposed ring AR maps onto each, and deriving the per-primitive cost formulas summarized by topology in §5. The canonical scale-out / multi-tier fabric — fat-tree / Clos — is inherently hierarchical and lives in `03_hierarchical_topologies.md` §2 alongside the composition rules that apply once topologies layer.

No network contention modeled here — every physical link is assumed to carry the active collective alone at peak bandwidth, with no concurrent flows from other collectives, other groups, or background traffic sharing the link. Realistic link sharing, bisection oversubscription, and concurrent-group conflicts are scored in `05_contention_and_congestion.md`.

# Table of Contents

1. [Topology primer](#1-topology-primer)
2. [Star topology](#2-star-topology)
   - 2.1 [Broadcast](#21-broadcast)
   - 2.2 [Reduce](#22-reduce)
   - 2.3 [All-reduce](#23-all-reduce)
   - 2.4 [All-gather / reduce-scatter](#24-all-gather--reduce-scatter)
   - 2.5 [All-to-all](#25-all-to-all)
3. [Torus topology](#3-torus-topology)
   - 3.1 [Dim-decomposition](#31-dim-decomposition)
   - 3.2 [Broadcast](#32-broadcast)
   - 3.3 [Reduce](#33-reduce)
   - 3.4 [All-reduce](#34-all-reduce)
   - 3.5 [All-gather / reduce-scatter](#35-all-gather--reduce-scatter)
   - 3.6 [All-to-all](#36-all-to-all)
4. [Mesh topology](#4-mesh-topology)
   - 4.1 [Full mesh](#41-full-mesh)
   - 4.2 [k-D mesh](#42-k-d-mesh)
5. [Summary and limitations](#5-summary-and-limitations)
   - 5.1 [Cost summary by topology](#51-cost-summary-by-topology)
   - 5.2 [Limitations](#52-limitations)
6. [Appendix A: Dim-decomposed Rabenseifner halving-doubling](#appendix-a-dim-decomposed-rabenseifner-halving-doubling)
7. [Appendix B: N, D, d — rank count, diameter, and dim sizes](#appendix-b-n-d-d--rank-count-diameter-and-dim-sizes)
8. [Appendix C: Per-chunk-per-step analysis vs bottleneck analysis](#appendix-c-per-chunk-per-step-analysis-vs-bottleneck-analysis)
9. [Further reading](#further-reading)

---

## 1. Topology primer

This note is a three-topology catalog of **single-tier, scale-up** fabrics: **star** (endpoints around a central high-radix switch — e.g., NVSwitch, §2), **torus** (a $k$-dimensional lattice with wraparound — e.g., TPU pods, §3), and **mesh** (direct all-to-all or $k$-D lattice without wraparound — e.g., chiplet interposer, PCIe point-to-point, §4). Star and torus cover scale-up domains at pod level; mesh covers short-reach direct fabrics. The canonical scale-out / multi-tier fabric — **fat-tree / Clos** — is inherently hierarchical and covered in `03_hierarchical_topologies.md` §2. Four properties describe any topology:

- **Diameter** $D$: hop count between the farthest rank pair.
- **Bisection bandwidth**: total BW across the narrowest cut that splits the ranks into two equal halves.
- **Radix**: ports per switch (for switched fabrics) or ports per router (for direct fabrics).
- **Scalability**: how wiring scales with $N$.

### 1.1 Star (single high-radix switch)

All $N$ endpoints attach to one central switch. $D = 2$ link hops (endpoint → switch → endpoint). This is NVLink 5 + NVSwitch Gen4 within a single NVL72 domain.

```
        ┌──────────────────────────────────────┐
        │              NVSwitch                │
        │         (N ports, full crossbar)     │
        └──┬─────┬─────┬─────┬─────┬─────┬────┘
           │     │     │     │     │     │
          R0    R1    R2    R3    ...   RN-1
```

- **Pros:** any-to-any in 2 hops. Collective algorithm is unconstrained by wiring since star can emulate any logical topology (ring, tree).
- **Cons:** scale limit is switch radix. A single NVSwitch Gen4 domain caps at ~72 GPUs; beyond that you need a multi-tier fabric.

### 1.2 Torus (2D or 3D lattice with wraparound)

Ranks are laid out on a $k$-dimensional lattice with dim sizes $(d_1, \ldots, d_k)$; each rank connects to its +1 and −1 neighbors per dim, with wraparound closing each axis into a ring. The neighbor count per rank is $\sum_i \min(d_i - 1, 2)$ — **at most $2k$**, reached when every $d_i \geq 3$. At a dim with $d_i = 2$ the +1 and −1 neighbors coincide, so that dim contributes 1 neighbor instead of 2. Asymmetric shapes may or may not trigger this — it depends per-dim: e.g., an $A \times A \times B$ torus with $A \geq 3$ and $B = 2$ gives $2 + 2 + 1 = 5$ neighbors per rank, but the same shape with $B = 4$ (or any $B \geq 3$) saturates back to $2 + 2 + 2 = 6$ because the $\min(d_i - 1, 2)$ term caps at 2 once $d_i$ clears 3. Google TPU pods use a 3D torus (typically symmetric — TPU v5p's 16×16×16 — so all dims are non-degenerate).

```
Torus structure: d = 2 per axis (degenerate, 1 link/dim) vs d ≥ 3 per axis (non-degenerate, 2 links/dim). Neighbors per rank = Σᵢ min(dᵢ − 1, 2).

──────────── Degenerate d = 2 per axis (direct ≡ wraparound, k links/rank) ────────────

   2D: 2×2 = 4 ranks                         3D: 2×2×2 = 8 ranks
   (k = 2, 2 links per rank)                 (k = 3, 3 links per rank)

                                                    (0,1,1)━━━━━(1,1,1)
                                                      ╱┃           ╱┃
                                                     ╱ ┃          ╱ ┃
                                               (0,1,0)━━━━━━(1,1,0) ┃
      (0,1)━━(1,1)                                ┃   ┃         ┃   ┃
        ┃      ┃                                  ┃ (0,0,1)━━━━━┃━(1,0,1)
        ┃      ┃                                  ┃  ╱          ┃  ╱
      (0,0)━━(1,0)                                ┃ ╱           ┃ ╱
                                               (0,0,0)━━━━━━(1,0,0)

   every drawn edge is both direct-neighbor and wraparound-neighbor: (0,y) ↔ (1,y)
   is the single X-edge (similarly Y, Z). At d = 2 the 2k rule collapses to k.

────────── Non-degenerate d = 3 per axis (direct ≠ wraparound, 2k links/rank) ──────────

   2D: 3×3 = 9 ranks                         2k = 4 links per rank
   (flat view, wraparound marked with ↔ ↕)

         ↕     ↕     ↕                       ↔  X-wraparound: (0,y) ━ (2,y) per y
    ↔  (0,2)━(1,2)━(2,2)  ↔                  ↕  Y-wraparound: (x,0) ━ (x,2) per x
          ┃     ┃     ┃
    ↔  (0,1)━(1,1)━(2,1)  ↔                  example: (0,0)'s 4 neighbors are
          ┃     ┃     ┃                        (1,0), (2,0)  [X]
    ↔  (0,0)━(1,0)━(2,0)  ↔                    (0,1), (0,2)  [Y]
         ↕     ↕     ↕

   3D: 3×3×3 = 27 ranks                      2k = 6 links per rank
   (stacked Z-slices, each slice a 3×3 2D torus as above)

    Z = 0 slice                 Z = 1 slice                 Z = 2 slice

       ↕      ↕      ↕             ↕      ↕      ↕             ↕      ↕      ↕
   ↔(0,2,0)━(1,2,0)━(2,2,0)↔   ↔(0,2,1)━(1,2,1)━(2,2,1)↔   ↔(0,2,2)━(1,2,2)━(2,2,2)↔
       ┃      ┃      ┃             ┃      ┃      ┃             ┃      ┃      ┃
   ↔(0,1,0)━(1,1,0)━(2,1,0)↔   ↔(0,1,1)━(1,1,1)━(2,1,1)↔   ↔(0,1,2)━(1,1,2)━(2,1,2)↔
       ┃      ┃      ┃             ┃      ┃      ┃             ┃      ┃      ┃
   ↔(0,0,0)━(1,0,0)━(2,0,0)↔   ↔(0,0,1)━(1,0,1)━(2,0,1)↔   ↔(0,0,2)━(1,0,2)━(2,0,2)↔
       ↕      ↕      ↕             ↕      ↕      ↕             ↕      ↕      ↕
                            ━━Z━━→                      ━━Z━━→
     ┌───────────────── Z-wraparound (Z = 2 slice → Z = 0 slice) ───────────────┐
     ▼                                                                           │
  Z = 0 slice                                                             Z = 2 slice

     ↔  X-wraparound within each slice (row closes into a 3-ring, 3 rows per slice)
     ↕  Y-wraparound within each slice (column closes into a 3-ring, 3 columns per slice)
     Z  direct Z-links between adjacent slices: (x,y,0) ━ (x,y,1), (x,y,1) ━ (x,y,2);
        plus Z-wraparound (x,y,2) ━ (x,y,0) closes each Z-chain into a 3-ring (9 total)

     examples — neighbors grouped by axis, with the slice of each one spelled out:

       (1,1,1) body-center (all 6 direct, no wraparound):
         X:  (0,1,1), (2,1,1)   — both in Z = 1 slice (left and right of (1,1,1))
         Y:  (1,0,1), (1,2,1)   — both in Z = 1 slice (below and above (1,1,1))
         Z:  (1,1,0), (1,1,2)   — (1,1,0) sits in Z = 0 slice, (1,1,2) in Z = 2 slice

       (0,0,0) corner (3 direct + 3 via wraparound, marked *):
         X:  (1,0,0), (2,0,0)*  — both in Z = 0 slice; * wraparound from X = 0 to X = 2
         Y:  (0,1,0), (0,2,0)*  — both in Z = 0 slice; * wraparound from Y = 0 to Y = 2
         Z:  (0,0,1), (0,0,2)*  — (0,0,1) in Z = 1 slice; (0,0,2)* in Z = 2 slice (wrap)

       (0,1,2) edge-midpoint (4 direct + 2 via wraparound, marked *):
         X:  (1,1,2), (2,1,2)*  — both in Z = 2 slice; * wraparound from X = 0 to X = 2
         Y:  (0,0,2), (0,2,2)   — both in Z = 2 slice; Y = 1 is interior so both direct
         Z:  (0,1,1), (0,1,0)*  — (0,1,1) in Z = 1 slice; (0,1,0)* in Z = 0 slice (wrap Z = 2 → 0)

     production TPU v5p pods: 16×16×16 = 4096 ranks (same structure, each axis a 16-ring)

──────────── Asymmetric shapes: degeneration is per-dim, not per-shape ────────────

   Asymmetry alone does NOT collapse anything — each dim is judged independently by its
   own dᵢ. Only a dim with dᵢ = 2 contributes 1 neighbor instead of 2.

   per-dim rule: dᵢ = 1 → 0 neighbors on axis i (axis is trivial, single rank)
                 dᵢ = 2 → 1 neighbor on axis i (direct ≡ wraparound, collapsed)
                 dᵢ ≥ 3 → 2 neighbors on axis i (direct ≠ wraparound, full)

   total neighbors per rank = Σᵢ min(dᵢ − 1, 2), bounded above by 2k.

   worked examples (k = 3):
     3 × 3 × 3     (symmetric, all ≥ 3)     → 2 + 2 + 2 = 6   (full 2k, non-degenerate)
     3 × 3 × 4     (asymmetric, all ≥ 3)    → 2 + 2 + 2 = 6   (still full 2k — asymmetry alone doesn't degenerate)
     16 × 16 × 4   (asymmetric, all ≥ 3)    → 2 + 2 + 2 = 6   (same; common TPU slice shape)
     4 × 4 × 2     (asymmetric, one = 2)    → 2 + 2 + 1 = 5   (Z-axis collapsed to a 2-ring)
     2 × 2 × 2     (symmetric, all = 2)     → 1 + 1 + 1 = 3   (all axes collapsed — the k case)
     8 × 1 × 1     (one axis trivial)       → 2 + 0 + 0 = 2   (effectively a 1D 8-ring)

   key distinction: "asymmetric" means the dᵢ differ; "degenerate on axis i" means dᵢ = 2.
   They are orthogonal — 3 × 3 × 4 is asymmetric but fully non-degenerate.
```

- **Pros:** no central switch — each rank has at most $2k$ links (fewer when any $d_i = 2$; general formula $\sum_i \min(d_i - 1, 2)$ per §1.2 intro). Linear wire cost in $N$. Dim structure enables dim-by-dim ring collectives that compress latency from $O(N)$ to $O(N^{1/k})$.
- **Cons:** diameter $D = \sum_i \lfloor d_i / 2 \rfloor$ grows as $N^{1/k}$; bisection scales as $N^{(k-1)/k}$ — sub-linear. All-to-all is bisection-constrained (more below). Realistic efficiency $\eta$ is typically lower than star's (longer path length accumulates per-hop overhead — details in `05_contention_and_congestion.md` §4). Fault tolerance is the weakest among the topologies here: a single rank failure breaks ring continuity in each of its $k$ dims, disrupting dim-decomposition; remapping either requires OCS-style slice reshaping (TPU v4+) or marking the whole slice unavailable (simpler torus fabrics).

### 1.3 Mesh (direct all-to-all or $k$-D lattice without wraparound)

Mesh is the direct-wiring counterpart to *both* star and torus — no central switch (the star-side simplification) and no wraparound (the torus-side simplification). Two regimes bracket the range, each corresponding to one of those parents: **full mesh** (the star analog), where every rank has a direct link to every other ($N(N-1)/2$ edges) — same any-to-any connectivity as star, but with the switch replaced by a dedicated per-pair link; and **$k$-D mesh** (the torus analog), where ranks sit on a $k$-D lattice with neighbor links only and **no wraparound** — same product-structured wiring as torus, but with the "seams" cut. The torus neighbor formula $\sum_i \min(d_i - 1, 2)$ applies unchanged to interior ranks; boundary ranks (at position 0 or $d_i - 1$ on any axis with $d_i \geq 3$) lose 1 neighbor per boundary axis — the wraparound that would have supplied the missing direction isn't there. §4 develops each precisely.

```
Full mesh (4 ranks):        2D mesh example: 4×4 = 16 ranks (no wraparound)

   R0 ━━━━━ R1             (0,3)━(1,3)━(2,3)━(3,3)
    ┃╲     ╱┃                ┃    ┃    ┃    ┃
    ┃ ╲   ╱ ┃              (0,2)━(1,2)━(2,2)━(3,2)
    ┃  ╲ ╱  ┃                ┃    ┃    ┃    ┃
    ┃   ╳   ┃              (0,1)━(1,1)━(2,1)━(3,1)
    ┃  ╱ ╲  ┃                ┃    ┃    ┃    ┃
    ┃ ╱   ╲ ┃              (0,0)━(1,0)━(2,0)━(3,0)
    ┃╱     ╲┃
   R3 ━━━━━ R2              (no edges wrap around between (3,·) and (0,·))
```

- **Pros:** full mesh removes the switch entirely — endpoint-to-endpoint α drops the cut-through hop — and every pair owns its own dedicated link so A2A is free of bisection or port-sharing pressure. $k$-D mesh retains torus's product-structured dim-decomposition — same $O(N^{1/k})$ α scaling for AR / AG / RS, just with each dim-phase running an open-line RS/AG instead of a closed-ring RS/AG (roughly 2× the α-coefficient per dim because the line loses the ring's bidirectional parallelism, but the $O(N^{1/k})$ order is preserved) — while using fewer physical links (no wraparound edges).
- **Cons:** full mesh has $O(N^2)$ wire count and $O(N)$ ports-per-rank, practically capping at ~8–16 endpoints (legacy DGX-1/DGX-2 hybrid cube-mesh, chiplet interposer fabrics). $k$-D mesh has half the bisection of a same-shape torus because the wraparound links are missing — A2A pays a 2× BW penalty relative to torus. $k$-D mesh inherits the same realistic-η and fault-tolerance weaknesses as torus (lattice continuity breaks on rank failure; details in `05_contention_and_congestion.md` §4). Full mesh is more forgiving on both axes — single-hop paths avoid the per-hop η accumulation, and any-to-any wiring means a rank failure only removes that endpoint without breaking any continuity structure.

---

## 2. Star topology

**Star is the most flexible substrate for mapping collective algorithms to hardware.** Any-to-any in 2 hops means every logical "communication edge" in a ring, DBT, or pairwise schedule lands on a single switch cut-through — the algorithm's logical topology *is* the physical topology, with no wiring-induced translation. Any of the seven primitives from `01_collective_algorithms.md` runs on a star at its *pure* α-β cost — no topology correction, no dim-decomposition, no hierarchical sub-tiers. **The algorithm is a software choice, not a wiring constraint.** This direct-mapping property is unique to star among the topologies in this series: torus forces dim-decomposition (§3); multi-tier fat-tree / Clos fabrics force tier-aware schedules that sub-divide the collective across per-tier sub-groups. There are no star-specific "special" schedules; this section only records the α calibration that makes the cost formulas concrete on a representative scale-up star, and defers algorithm selection to the prior note. In-network collectives (SHARP, covered in `04_in_network_collectives.md`) build on this by collapsing even the tree's $\log_2 N$ endpoint hops to $O(1)$ switch-hop latency — an amplification of star's algorithmic freedom, not a separate algorithm class.

**α and BW calibration on a scale-up star.** On NVLink-5 / NVSwitch Gen4 class hardware:

- $\alpha \approx 0.5\\,\mu$s — switch cut-through ($\sim$100 ns) plus endpoint software overhead ($\sim$400 ns). Each algorithm "hop" in the star primitive costs this $\alpha$, which already includes the physical 2-link endpoint→switch→endpoint traversal as a single cut-through event.
- $\mathrm{BW} \approx 900\\,\mathrm{GB/s}$ per direction per GPU — a single NVLink-5 port's unidirectional bandwidth. All cost formulas from the prior note use this as the per-port BW.

### 2.1 Broadcast

**Relation to `01_collective_algorithms.md` §3.** All three options from the prior note — ring BC (§3.1), binomial-tree BC (§3.2), and hardware switch multicast (§3.3) — run on a star at their pure costs. Star contributes only the α and BW calibration above.

**Commercial shipment.** NCCL ships binomial-tree BC on star as the default software path. NVSwitch Gen3+ / NVLS exposes the switch-multicast primitive that collapses α to the $2\alpha_{\mathrm{switch}}$ $N$-independent floor (see `04_in_network_collectives.md` §1.2).

### 2.2 Reduce

**Relation to `01_collective_algorithms.md` §4.** Reduce is the time-reverse dual of BC: ring Reduce (§4.1), binomial-tree Reduce (§4.2), and switch-ALU Reduce / SHARP / NVLS (§4.3) all run on a star at their pure costs. Per `01_collective_algorithms.md` §4.3, tree strictly dominates ring across the full $M$-range for standalone Reduce (same BW floor, smaller α), so star's effective menu reduces to tree + INC.

**Commercial shipment.** NCCL ships tree Reduce on star. NVSwitch Gen3+ / NVLS and InfiniBand SHARP provide the switch-ALU path that collapses the α count to an $N$-independent constant (see `04_in_network_collectives.md` §1.1).

### 2.3 All-reduce

**Relation to `01_collective_algorithms.md` §5.** All four AR algorithms from the prior note (ring in §5.1, DBT in §5.2, plus simple rec-doubling in App. B.1 and Rabenseifner in App. B.2 for reference) run on a star at their pure costs — no topology correction, no dim-decomposition. Star contributes only the α and BW calibration above.

**Commercial shipment.** NCCL ships both ring and DBT on star. The $(N, M)$ selection behavior between them is a runtime-scheduler concern, not a topology concern — see `01_collective_algorithms.md` §5.3 for the algorithm-selection discussion.

### 2.4 All-gather / reduce-scatter

**Relation to `01_collective_algorithms.md` §6.** Star runs both ring AG/RS (§6) and rec-doubling AG / rec-halving RS (App. B.4) at their pure costs from the prior note. No star-specific correction to the cost formulas: `BW` = per-port BW, α = cut-through + software, plug in and read off.

**Commercial shipment.** NCCL ships ring AG/RS on star. The ring-vs-log-depth selection rationale is a runtime-scheduler concern — see `01_collective_algorithms.md` §6.

### 2.5 All-to-all

**Relation to `01_collective_algorithms.md` §7.** A2A on a star is bounded by per-rank port BW, not by the switch fabric. The switch has aggregate BW $N \cdot \mathrm{BW}$ — far more than any collective needs — but each endpoint has only one duplex link, so the information-theoretic lower bound on cost is $\frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}} \approx M/\mathrm{BW}$ (each rank must ship $(N-1)M/N$ bytes through its one port). **Star's win over torus for A2A is that it *achieves* this BW lower bound with no bisection bottleneck and no dim-decomposition to engineer** — just peak per-port BW applied uniformly across all $N(N-1)$ pairs.

**Commercial shipment.** NCCL ships pairwise direct-send (`01_collective_algorithms.md` §7.2) on star. Full algorithm-selection rationale — including the bisection-limited vs full-bisection fabric rule that picks pairwise on star but ring-relay on torus — lives in `01_collective_algorithms.md` §7.3.

---

## 3. Torus topology

A torus trades star's algorithmic flexibility for linear wire cost at scale. The return on that trade is a single powerful optimization that distinguishes torus from star / fat-tree / Clos: **dim-decomposition** — rewriting each primitive as $k$ serial phases, one per physical dim, each running an independent $d_i$-rank ring on its own disjoint wiring. The payoff is structural: the flat-ring $O(N)$ α term collapses to $O(N^{1/k})$ while the bandwidth term stays at the flat-ring bandwidth optimum. **This is the sole reason a torus can match or beat a star for AR / AG / RS despite its longer average path length.** Take dim-decomposition away and a torus is strictly worse than a star for every primitive.

§3.1 develops the general dim-decomposition mechanism with a 2×2×2 visual; §3.2 and §3.3 cover BC and Reduce, and §3.4–§3.6 specialize dim-decomp to AR / AG / RS / A2A. Realistic-$\eta$ re-scoring of both wins and losses lives in `05_contention_and_congestion.md`.

### 3.1 Dim-decomposition

Dim-decomposition is the topology-aware rewrite that makes torus competitive on collective cost. This subsection defines the rewrite, visualizes the per-dim phases on a 2×2×2 torus, states the correctness / BW-optimality / latency-compression properties, and catalogs where the decomposition applies and where it breaks. The primitive-specific subsections (§3.2 BC, §3.3 Reduce, §3.4 AR, §3.5 AG/RS, §3.6 A2A) all specialize this framework — chunk-size bookkeeping, AG-vs-RS pairing, cost formulas.

**The rewrite.** A flat collective over $N$ ranks is decomposed into $k$ serial phases, one per physical dim. Within phase $i$, $(N/d_i)$ independent rings of size $d_i$ run *concurrently* on the $(N/d_i)$ dim-lines perpendicular to axis $i$; each ring uses only dim-$i$ links, and different rings share no wiring.

**Visual — three phases on a 2×2×2 torus.** The 2×2×2 has 8 ranks and 12 edges split into three axis families: 4 X-edges, 4 Y-edges, 4 Z-edges. Dim-decomposition runs three sequential phases; each phase uses only one axis's edges while the other two are idle.

```
Torus 2×2×2 — 8 ranks labeled a–h (link families labeled; X horizontal, Y into-page, Z vertical):

              g━━━━━━h           ━━━  horizontal edges = X-axis links (4 total)
             ╱┃     ╱┃            ╱   diagonal edges   = Y-axis links (4 total)
            ╱ ┃    ╱ ┃             ┃  vertical edges   = Z-axis links (4 total)
           c━━━━━━d  ┃
           ┃  ┃   ┃  ┃
           ┃  e━━━┃━━f
           ┃ ╱    ┃ ╱
           ┃╱     ┃╱
           a━━━━━━b

Phase X  (X-links carry all traffic; Y and Z links idle):
  4 concurrent 2-rank rings along horizontal edges —
    { a ↔ b },  { c ↔ d },  { e ↔ f },  { g ↔ h }.

Phase Y  (Y-links carry all traffic; X and Z links idle):
  4 concurrent 2-rank rings along diagonal (into-page) edges —
    { a ↔ e },  { b ↔ f },  { c ↔ g },  { d ↔ h }.

Phase Z  (Z-links carry all traffic; X and Y links idle):
  4 concurrent 2-rank rings along vertical edges —
    { a ↔ c },  { b ↔ d },  { e ↔ g },  { f ↔ h }.
```

At $d_i = 2$ per axis (shown here) each "ring" collapses to a pair-exchange (direct ≡ wraparound); at $d_i \geq 3$ each phase runs genuine $d_i$-rings, but the structure "each phase uses only its axis's links, rings run concurrently on disjoint wiring" is unchanged. A TPU v5p $16 \times 16 \times 16$ torus runs 256 concurrent 16-rings per dim-phase.

**Concrete savings vs flat ring on the same 8 ranks.** To see what dim-decomposition buys, compare against running a flat 8-ring schedule on the same 2×2×2 torus (ignoring the product structure — the linearized rank order still runs correctly, but each logical hop crosses many physical torus links on average):

```
Flat-ring schedule on 2×2×2 torus (8 ranks linearized into a single ring):

   a ━ b ━ e ━ f ━ c ━ d ━ g ━ h ━┐
   ↑                              │
   └──────────────────────────────┘
        wraparound (flat-ring closure)

   N − 1 = 7 sequential hops per ring traversal; every rank sends / receives
   one neighbor exchange at a time. No inter-rank parallelism.


Dim-decomposed schedule on 2×2×2 torus:

   Phase X            →   Phase Y            →   Phase Z
   ─────────────────       ─────────────────       ─────────────────
   4 pair-exchanges        4 pair-exchanges        4 pair-exchanges
   concurrent              concurrent              concurrent
   on X-links              on Y-links              on Z-links

   Σ (dᵢ − 1) = 3 sequential hops per ring traversal — less than half of 7.
   Within each phase, 4 pair-rings run on physically disjoint wiring simultaneously.
```

**7 vs 3 sequential hops → ~2.3× α-compression at $N = 8$, for free** — same total bytes moved, same BW term, just fewer serial waits. The ratio grows as $(N - 1) / \sum_i (d_i - 1)$: at $8 \times 8 \times 8$ that is $511 / 21 = 24\times$; at $16 \times 16 \times 16$ it is $4095 / 45 = 91\times$. Any specific primitive (AR = two traversals, AG or RS = one, A2A = different schedule entirely) multiplies through the same ratio — the compression is a pure schedule rewrite discovering the $k$ parallel dim-rings the torus's wiring was already providing.

**Why it works.** Three properties follow from the torus's $k$-D Cartesian-product wiring:

- **Correctness** — reduction (and concatenation, for AG) is associative, so the collective factors across dims:
  $\sum_{x, y, z} V_{x,y,z} = \sum_z \big( \sum_y ( \sum_x V_{x,y,z} ) \big).$
  Each inner sum is an independent problem handled by one dim-phase.
- **Bandwidth-optimality** — the $(N/d_i)$ independent rings in phase $i$ use physically disjoint link sets (different rings of the same dim live on different dim-lines). No inter-ring contention wastes BW.
- **Latency compression** — $k$ serial phases, each $O(d_i) = O(N^{1/k})$ long, total $O(k \cdot N^{1/k}) = O(N^{1/k})$ for fixed $k$.

**Where dim-decomposition breaks.** The compression is not a blanket win on torus:

- You *can* still embed a flat $N$-ring on a torus (each logical hop then crosses many physical links), paying the full flat-ring $O(N)$ cost — dim-decomposition is the optimization that skips this, not the only correct schedule.
- Compression requires **dim-aligned group layouts** (§3.4). Off-prefix groups collapse the decomposition back toward flat-ring.
- **A2A is bisection-bound, not per-port** (§3.6) — dim-decomposition doesn't help the BW side because A2A's aggregate cross-cut traffic is fixed by semantics, not schedule. Cost scales with $d_{\max}$, not $N$.
- **Asymmetric dim sizes pay disproportionately** — a pathological shape like $256 \times 2 \times 2$ costs far more than cubic $8 \times 8 \times 8$ at the same $N$.

### 3.2 Broadcast

BC on a torus ships as **dim-decomposed ring BC** — the §3.1 framework with ring BC (`01_collective_algorithms.md` §3.1) as the inner per-dim primitive. Unlike AR, BC has only $k$ phases (no reverse half): the source fans out along dim 1, every dim-1 recipient then fans out along dim 2 concurrently, and so on until all $N$ ranks hold the payload. Unidirectional ring BC per dim costs $(d_i - 1)\alpha + M/\mathrm{BW}$; a **bidirectional** variant using both ring directions concurrently reaches the ring-diameter lower bound of $\lfloor d_i / 2 \rfloor \alpha$ per dim (≈2× α compression for $d_i \geq 3$).

> **Aside — why tree BC doesn't help on torus.** On fully-connected substrates (star, §2.1), binomial tree BC compresses α to $\lceil \log_2 N \rceil$ because every logical tree edge crosses one physical hop. On a $d$-ring, a tree's logical "distance-$d/2$ step" covers $d/2$ physical hops, and the subsequent halvings sum back to the same $d - 1$ total physical α as ring BC. The log-depth α advantage requires any-to-any 1-hop reachability, which torus per-dim wiring doesn't provide. Bidirectional ring BC's $\lfloor d_i / 2 \rfloor \alpha$ is the best α achievable on ring wiring.

**Algorithm.** Run per-dim ring BC one dim at a time. Phase order on a 3D torus: X-BC → Y-BC → Z-BC — $k$ phases total. After phase $i$, the set of ranks holding the payload has grown by a factor of $d_i$.

**Why it works — 2×2×2 worked example.** Eight ranks labeled a–h, same slice convention as §3.4 (a at origin (0,0,0), z=0 slice holds a/b/e/f, z=1 slice holds c/d/g/h). Source is rank a, holding an $M$-byte payload $V$. All other ranks start empty (shown as $\cdot$).

```
Initial state — only a holds V:

Z = 0 slice:                 Z = 1 slice:
     ┌─────┬─────┐                ┌─────┬─────┐
y=1  │  ·  │  ·  │           y=1  │  ·  │  ·  │
     ├─────┼─────┤                ├─────┼─────┤
y=0  │  V  │  ·  │           y=0  │  ·  │  ·  │
     └─────┴─────┘                └─────┴─────┘
       x=0   x=1                    x=0   x=1
```

*Phase 1 (X-BC).* One active 2-ring along X — the ring containing the source: {a, b}. a sends $V$ to b along the X-link. The other three X-rings ({e,f}, {c,d}, {g,h}) are dormant: no source holds $V$ yet.

```
After Phase 1 (X-BC) — a, b hold V:

Z = 0 slice:                 Z = 1 slice:
     ┌─────┬─────┐                ┌─────┬─────┐
y=1  │  ·  │  ·  │           y=1  │  ·  │  ·  │
     ├─────┼─────┤                ├─────┼─────┤
y=0  │  V  │  V  │           y=0  │  ·  │  ·  │
     └─────┴─────┘                └─────┴─────┘
       x=0   x=1                    x=0   x=1
```

*Phase 2 (Y-BC).* Two active 2-rings along Y, one per (x, z=0): {a, e}, {b, f}. a → e and b → f concurrently. The other two Y-rings ({c,g}, {d,h}) are still dormant.

```
After Phase 2 (Y-BC) — entire Z=0 slice holds V:

Z = 0 slice:                 Z = 1 slice:
     ┌─────┬─────┐                ┌─────┬─────┐
y=1  │  V  │  V  │           y=1  │  ·  │  ·  │
     ├─────┼─────┤                ├─────┼─────┤
y=0  │  V  │  V  │           y=0  │  ·  │  ·  │
     └─────┴─────┘                └─────┴─────┘
       x=0   x=1                    x=0   x=1
```

*Phase 3 (Z-BC).* Four concurrent 2-rings along Z, one per (x, y): {a, c}, {b, d}, {e, g}, {f, h}. Every Z-ring now has a source (the entire Z=0 slice holds $V$), so all four rings broadcast concurrently. After Phase 3, all 8 ranks hold $V$.

```
After Phase 3 (Z-BC) — all 8 ranks hold V. BC complete:

Z = 0 slice:                 Z = 1 slice:
     ┌─────┬─────┐                ┌─────┬─────┐
y=1  │  V  │  V  │           y=1  │  V  │  V  │
     ├─────┼─────┤                ├─────┼─────┤
y=0  │  V  │  V  │           y=0  │  V  │  V  │
     └─────┴─────┘                └─────┴─────┘
       x=0   x=1                    x=0   x=1
```

BC complete: every rank holds $V$. Total phase count: $k = 3$; total sequential α-hops: $\sum_i (d_i - 1) = 3$ — half of AR's $2\sum_i (d_i - 1) = 6$, since BC is one-way (source → all, no reverse half).

**Bidirectional variant — 3×3 torus example.** For $d_i \geq 3$, using both ring directions concurrently cuts α roughly in half. On a 3×3 torus (9 ranks, source = a at (0, 0)):

```
3×3 torus grid (wraparound on every row and column):

     ┌───┬───┬───┐
y=2  │ g │ h │ i │          X-wraps (per row):   a↔c, d↔f, g↔i
     ├───┼───┼───┤          Y-wraps (per col):   a↔g, b↔h, c↔i
y=1  │ d │ e │ f │
     ├───┼───┼───┤
y=0  │ a │ b │ c │
     └───┴───┴───┘
       x=0 x=1 x=2

Initial — only a holds V:         Phase 1 (X-BC, bidirectional):
                                  a sends V → b (forward)
     ┌───┬───┬───┐                       and V → c (via X-wrap),
y=2  │ · │ · │ · │                       concurrently on row y=0.
     ├───┼───┼───┤                Only row y=0 has a source; rows
y=1  │ · │ · │ · │                y=1, y=2 stay dormant this phase.
     ├───┼───┼───┤
y=0  │ V │ · │ · │
     └───┴───┴───┘
       x=0 x=1 x=2

After Phase 1 (X-BC) — row y=0 holds V:

     ┌───┬───┬───┐
y=2  │ · │ · │ · │
     ├───┼───┼───┤
y=1  │ · │ · │ · │
     ├───┼───┼───┤                        1 α elapsed
y=0  │ V │ V │ V │
     └───┴───┴───┘
       x=0 x=1 x=2

Phase 2 (Y-BC, bidirectional on every column in parallel):
Each column's y=0 rank sends V → y=1 (forward) AND V → y=2 (via Y-wrap)
concurrently. All 3 columns fire in parallel because every column now
has a V-source in row y=0.

After Phase 2 (Y-BC) — all 9 ranks hold V:

     ┌───┬───┬───┐
y=2  │ V │ V │ V │
     ├───┼───┼───┤
y=1  │ V │ V │ V │                        1 α elapsed
     ├───┼───┼───┤
y=0  │ V │ V │ V │
     └───┴───┴───┘
       x=0 x=1 x=2

Total: 2 α + M/BW  (vs unidirectional ring's (3-1) + (3-1) = 4 α — 2× α win.)
```

On a 3-ring, bidirectional completes in 1 α per dim because each rank's two ring-neighbors cover the rest of the ring. For larger $d_i$ the pattern generalizes to a $\lfloor d_i/2 \rfloor$-step wave radiating outward from the source (e.g., 4-ring = 2 α, 8-ring = 4 α). The 2×2×2 worked example below uses $d_i = 2$ where bidirectional and unidirectional coincide trivially (1 α per dim).

**Cost formula.** Each dim-$i$ phase is a $d_i$-rank pipelined ring BC at cost $(d_i - 1)\alpha + M/\mathrm{BW}$ unidirectional or $\lfloor d_i/2 \rfloor \alpha + M/\mathrm{BW}$ bidirectional (`01_collective_algorithms.md` §3.1). With pipelining across the $k$ dim phases, the BW term telescopes into a single $M/\mathrm{BW}$ (each phase re-uses the payload pipelined in from the previous dim, so only the source-side dispatch bottleneck — one $M$-worth of bytes through one outbound port — lands on the wall clock):

$$t_{\mathrm{torus,BC,ring}} \\;\approx\\; \sum_i (d_i - 1)\\,\alpha \\;+\\; \frac{M}{\mathrm{BW}} \quad\text{(unidirectional)}$$

$$t_{\mathrm{torus,BC,bidir}} \\;\approx\\; \sum_i \lfloor d_i / 2 \rfloor\\,\alpha \\;+\\; \frac{M}{\mathrm{BW}} \quad\text{(bidirectional)}$$

**Switch-hosted multicast (NVLS / SHARP) is structurally unavailable on torus** — there is no central switch to fan out from — so the α floor is purely software-driven, unlike star (§2.1) where INC collapses α to $2\alpha_{\mathrm{switch}}$.

**Commercial shipment.** TPU/Trainium expose BC as a dim-decomposed pipelined primitive; whether XLA / NeuronX run unidirectional or bidirectional per-dim ring is a tuner detail not publicly documented. Cross-pod BC on composite fabrics (scale-up torus + scale-out Clos) reverts to per-tier schedules — see `03_hierarchical_topologies.md`.

### 3.3 Reduce

Reduce on a torus ships as **dim-decomposed ring Reduce** — the time-reverse dual of §3.2's BC. Every rank contributes its own $V_i$; contributions accumulate toward a single root via the inverse dim-decomposition. Phase order reverses BC's: collapse the last dim first, so that the set of ranks holding live partial sums shrinks by a factor of $d_i$ each phase until only the root remains. As with BC, the **bidirectional ring variant** reaches $\lfloor d_i/2 \rfloor \alpha$ per dim by reducing from both halves of the ring toward the root concurrently, vs unidirectional's $d_i - 1$; the tree-BC caveat from §3.2 applies symmetrically (log-depth α needs any-to-any reachability, which ring wiring lacks).

**Algorithm.** Run per-dim ring Reduce one dim at a time. Phase order on a 3D torus with root a at (0, 0, 0): Z-Reduce → Y-Reduce → X-Reduce — $k$ phases total. Each phase accumulates partial sums toward the root's coordinate on that axis.

**Why it works — 2×2×2 worked example.** Eight ranks labeled a–h, same slice convention as §3.4. Root is rank a. Each rank initially holds its own $M$-byte vector $V_i$.

```
Initial state — each rank holds its own V_i:

Z = 0 slice:                    Z = 1 slice:
     ┌─────┬─────┐                   ┌─────┬─────┐
y=1  │ V_e │ V_f │              y=1  │ V_g │ V_h │
     ├─────┼─────┤                   ├─────┼─────┤
y=0  │ V_a │ V_b │              y=0  │ V_c │ V_d │
     └─────┴─────┘                   └─────┴─────┘
```

*Phase 1 (Z-Reduce).* Four concurrent 2-rings along Z, one per (x, y): {a, c}, {b, d}, {e, g}, {f, h}. Each pair reduces toward its z=0 endpoint — c → a, d → b, g → e, h → f, with z=0 ranks summing into their own buffer. After Phase 1, the Z=0 slice holds 2-rank partial sums; the Z=1 slice is dormant (its state has been absorbed).

```
After Phase 1 (Z-Reduce) — Z=0 slice holds pairwise sums; Z=1 dormant:

Z = 0 slice:                           Z = 1 slice:
     ┌───────────┬───────────┐              ┌──────────┬──────────┐
y=1  │ V_e + V_g │ V_f + V_h │         y=1  │ dormant  │ dormant  │
     ├───────────┼───────────┤              ├──────────┼──────────┤
y=0  │ V_a + V_c │ V_b + V_d │         y=0  │ dormant  │ dormant  │
     └───────────┴───────────┘              └──────────┴──────────┘
```

*Phase 2 (Y-Reduce).* Two active 2-rings along Y in the Z=0 slice: {a, e}, {b, f}. Each pair reduces toward y=0: e → a, f → b. Row y=1 becomes dormant.

```
After Phase 2 (Y-Reduce) — row y=0 of Z=0 holds 4-rank partial sums:

Z = 0 slice:                                         Z = 1 slice: dormant
     ┌───────────────────────┬───────────────────────┐
y=1  │        dormant        │        dormant        │
     ├───────────────────────┼───────────────────────┤
y=0  │ V_a+V_c+V_e+V_g       │ V_b+V_d+V_f+V_h       │
     └───────────────────────┴───────────────────────┘
```

*Phase 3 (X-Reduce).* One active 2-ring along X in row (y=0, z=0): {a, b}. b reduces toward x=0: b → a. After Phase 3, only rank a holds live state, and it holds the full reduction.

```
After Phase 3 (X-Reduce) — a holds the full sum. Reduce complete:

Z = 0 slice:
     ┌───────────────────────────────────┬─────────┐
y=1  │              dormant              │ dormant │
     ├───────────────────────────────────┼─────────┤
y=0  │ V_a+V_b+V_c+V_d+V_e+V_f+V_g+V_h   │ dormant │
     └───────────────────────────────────┴─────────┘

Z = 1 slice: dormant

  Rank a holds Σ = V_a + V_b + V_c + V_d + V_e + V_f + V_g + V_h.
```

Reduce complete: root a holds $\Sigma = \sum_i V_i$. Total phase count: $k = 3$; total sequential α-hops: $\sum_i (d_i - 1) = 3$ — same as BC (time-reverse dual), half of AR.

**Bidirectional variant — 3×3 torus example.** Time-reverse of §3.2's bidirectional BC. On a 3×3 torus (9 ranks, root = a at (0, 0)), let $S_x$ denote the sum of column $x$: $S_0 = V_a + V_d + V_g$, $S_1 = V_b + V_e + V_h$, $S_2 = V_c + V_f + V_i$. Final answer: $\Sigma = S_0 + S_1 + S_2 = \sum_i V_i$.

```
3×3 torus grid (same layout as §3.2; root = a at (0,0)):

     ┌───┬───┬───┐
y=2  │ g │ h │ i │          X-wraps: a↔c, d↔f, g↔i
     ├───┼───┼───┤          Y-wraps: a↔g, b↔h, c↔i
y=1  │ d │ e │ f │
     ├───┼───┼───┤
y=0  │ a │ b │ c │
     └───┴───┴───┘
       x=0 x=1 x=2

Initial — each rank holds its own V_i:

     ┌─────┬─────┬─────┐
y=2  │ V_g │ V_h │ V_i │
     ├─────┼─────┼─────┤
y=1  │ V_d │ V_e │ V_f │
     ├─────┼─────┼─────┤
y=0  │ V_a │ V_b │ V_c │
     └─────┴─────┴─────┘
       x=0   x=1   x=2

Phase 1 (Y-Reduce, bidirectional on every column in parallel):
For each column, y=1 sends its V forward to y=0 AND y=2 sends its V via Y-wrap
to y=0, concurrently. Root-of-column (y=0 rank) sums both into its own V.
All 3 columns fire in parallel (3 concurrent 3-rings).

After Phase 1 (Y-Reduce) — row y=0 holds column-sums S_0, S_1, S_2:

     ┌─────┬─────┬─────┐
y=2  │  —  │  —  │  —  │   (dormant)
     ├─────┼─────┼─────┤
y=1  │  —  │  —  │  —  │   (dormant)
     ├─────┼─────┼─────┤
y=0  │ S_0 │ S_1 │ S_2 │         1 α elapsed
     └─────┴─────┴─────┘
       x=0   x=1   x=2

Phase 2 (X-Reduce, bidirectional on row y=0):
b sends S_1 forward to a AND c sends S_2 via X-wrap to a, concurrently.
a sums both into its own S_0.

After Phase 2 (X-Reduce) — a holds the full sum Σ:

     ┌─────┬─────┬─────┐
y=2  │  —  │  —  │  —  │
     ├─────┼─────┼─────┤
y=1  │  —  │  —  │  —  │
     ├─────┼─────┼─────┤
y=0  │  Σ  │  —  │  —  │         1 α elapsed
     └─────┴─────┴─────┘
       x=0   x=1   x=2

Total: 2 α + M/BW  (vs unidirectional ring's (3-1) + (3-1) = 4 α — 2× α win.)
```

The 2×2×2 worked example above uses $d_i = 2$ where bidirectional and unidirectional coincide (1 α per dim either way); the 3×3 illustration makes the bidirectional wave-toward-root mechanic visible.

**Cost formula.** Each dim-$i$ phase is a $d_i$-rank pipelined ring Reduce at cost $(d_i - 1)\alpha + M/\mathrm{BW}$ unidirectional or $\lfloor d_i/2 \rfloor \alpha + M/\mathrm{BW}$ bidirectional (`01_collective_algorithms.md` §4.1). Symmetric to BC (§3.2), the BW term telescopes to $M/\mathrm{BW}$ overall under pipelining:

$$t_{\mathrm{torus,Reduce,ring}} \\;\approx\\; \sum_i (d_i - 1)\\,\alpha \\;+\\; \frac{M}{\mathrm{BW}} \quad\text{(unidirectional)}$$

$$t_{\mathrm{torus,Reduce,bidir}} \\;\approx\\; \sum_i \lfloor d_i / 2 \rfloor\\,\alpha \\;+\\; \frac{M}{\mathrm{BW}} \quad\text{(bidirectional)}$$

As with BC, the torus has no in-network ALU (ICI is link-only) — so the α floor is software-driven; the RS phase of the full torus AR (§3.4) stays the preferred path when every rank needs the result anyway, avoiding the Reduce+BC decomposition (see `01_collective_algorithms.md` §4.3).

### 3.4 All-reduce

AR on a torus ships as **dim-decomposed ring** — the §3.1 framework with ring RS + AG as the inner per-dim primitive (Patarasuk-Yuan, `01_collective_algorithms.md` §5.1, just on a $d_i$-rank ring). TPU (XLA / JAX) and AWS Trainium (NeuronX CCL) both default to it. A tree-flavored alternative — **dim-decomposed Rabenseifner halving-doubling** — swaps the inner primitive from ring to halving-doubling for additional α compression; it is not shipping on any production torus stack (Appendix A has the full derivation) but serves as a reference point that sharpens the production-ring analysis.

**Algorithm.** Run ring RS+AG one dim at a time. Phase order on a 3D torus: X-RS, Y-RS, Z-RS, then Z-AG, Y-AG, X-AG — $2k$ phases total. The AG dims traverse in reverse of the RS dims so that chunk sizes grow back symmetrically to how they shrank during RS (required for clean BW telescoping below; not for correctness — any AG dim order finishes AR correctly).

**Why it works — 2×2×2 worked example.** Eight ranks labeled a–h; each holds a length-8 input vector of size $M$ bytes (8 chunks of size $M/8$). Rank a's input is $[a_0, a_1, \ldots, a_7]$, rank b's is $[b_0, \ldots, b_7]$, and so on through h. The target AR output at every rank is $[\Sigma_0, \Sigma_1, \ldots, \Sigma_7]$ where $\Sigma_i = a_i + b_i + c_i + d_i + e_i + f_i + g_i + h_i$.

We visualize state as two 2×2 slices (Z = 0 front, Z = 1 back) — the stacked-slice form from §1.2 — where the letter in each cell names the rank at that position. The schedule runs **3 RS phases** (X-RS → Y-RS → Z-RS) followed by **3 AG phases** (Z-AG → Y-AG → X-AG); each phase uses only its axis's links (§3.1).

```
Initial state — each rank holds its own 8-chunk input vector:

Z = 0 slice:
     ┌───────────────────────────┬───────────────────────────┐
y=1  │ [e₀,e₁,e₂,e₃,e₄,e₅,e₆,e₇] │ [f₀,f₁,f₂,f₃,f₄,f₅,f₆,f₇] │
     ├───────────────────────────┼───────────────────────────┤
y=0  │ [a₀,a₁,a₂,a₃,a₄,a₅,a₆,a₇] │ [b₀,b₁,b₂,b₃,b₄,b₅,b₆,b₇] │
     └───────────────────────────┴───────────────────────────┘
       x=0                         x=1

Z = 1 slice:
     ┌───────────────────────────┬───────────────────────────┐
y=1  │ [g₀,g₁,g₂,g₃,g₄,g₅,g₆,g₇] │ [h₀,h₁,h₂,h₃,h₄,h₅,h₆,h₇] │
     ├───────────────────────────┼───────────────────────────┤
y=0  │ [c₀,c₁,c₂,c₃,c₄,c₅,c₆,c₇] │ [d₀,d₁,d₂,d₃,d₄,d₅,d₆,d₇] │
     └───────────────────────────┴───────────────────────────┘
       x=0                         x=1
```

*Phase 1 (X-RS).* 4 concurrent 2-rings along X, one per (y, z) pair: {a,b}, {c,d}, {e,f}, {g,h}. Representative ring {a, b}: a keeps chunks 0–3, sends 4–7 to b; b keeps 4–7, sends 0–3 to a; each adds received into kept. Other rings do the same on their own pairs.

```
After Phase 1 (X-RS) — chunk size M → M/2; each rank holds 4 chunks:

Z = 0 slice:
     ┌───────────────────────────┬───────────────────────────┐
y=1  │ [e₀+f₀,e₁+f₁,e₂+f₂,e₃+f₃] │ [e₄+f₄,e₅+f₅,e₆+f₆,e₇+f₇] │
     ├───────────────────────────┼───────────────────────────┤
y=0  │ [a₀+b₀,a₁+b₁,a₂+b₂,a₃+b₃] │ [a₄+b₄,a₅+b₅,a₆+b₆,a₇+b₇] │
     └───────────────────────────┴───────────────────────────┘
       x=0                         x=1

Z = 1 slice:
     ┌───────────────────────────┬───────────────────────────┐
y=1  │ [g₀+h₀,g₁+h₁,g₂+h₂,g₃+h₃] │ [g₄+h₄,g₅+h₅,g₆+h₆,g₇+h₇] │
     ├───────────────────────────┼───────────────────────────┤
y=0  │ [c₀+d₀,c₁+d₁,c₂+d₂,c₃+d₃] │ [c₄+d₄,c₅+d₅,c₆+d₆,c₇+d₇] │
     └───────────────────────────┴───────────────────────────┘
       x=0                         x=1
```

All four X-rings ran concurrently on physically disjoint X-links — row-0 of the Z=0 slice doesn't wait on row-1, nor on anything in the Z=1 slice.

*Phase 2 (Y-RS).* 4 concurrent 2-rings along Y, one per (x, z) pair: {a,e}, {b,f}, {c,g}, {d,h}. Representative ring {a, e}: each now holds 4 chunks; a keeps chunks 0–1, sends 2–3 to e; e keeps 2–3, sends 0–1 to a; each adds.

```
After Phase 2 (Y-RS) — chunk size M/2 → M/4; each rank holds 2 chunks:

Z = 0 slice:
     ┌───────────────────────────┬───────────────────────────┐
y=1  │ [a₂+b₂+e₂+f₂,a₃+b₃+e₃+f₃] │ [a₆+b₆+e₆+f₆,a₇+b₇+e₇+f₇] │
     ├───────────────────────────┼───────────────────────────┤
y=0  │ [a₀+b₀+e₀+f₀,a₁+b₁+e₁+f₁] │ [a₄+b₄+e₄+f₄,a₅+b₅+e₅+f₅] │
     └───────────────────────────┴───────────────────────────┘
       x=0                         x=1

Z = 1 slice:
     ┌───────────────────────────┬───────────────────────────┐
y=1  │ [c₂+d₂+g₂+h₂,c₃+d₃+g₃+h₃] │ [c₆+d₆+g₆+h₆,c₇+d₇+g₇+h₇] │
     ├───────────────────────────┼───────────────────────────┤
y=0  │ [c₀+d₀+g₀+h₀,c₁+d₁+g₁+h₁] │ [c₄+d₄+g₄+h₄,c₅+d₅+g₅+h₅] │
     └───────────────────────────┴───────────────────────────┘
       x=0                         x=1
```

*Phase 3 (Z-RS).* 4 concurrent 2-rings along Z, one per (x, y) pair — this is the first phase that crosses slices: {a,c}, {b,d}, {e,g}, {f,h}. Representative ring {a, c}: each holds 2 chunks; a keeps chunk 0, sends 1 to c; c keeps 1, sends 0 to a; each adds.

```
After Phase 3 (Z-RS) — chunk size M/4 → M/8 = M/N; each rank holds 1 fully-reduced chunk:

Z = 0 slice:
     ┌───────────────────────────┬───────────────────────────┐
y=1  │ [a₂+b₂+c₂+d₂+e₂+f₂+g₂+h₂] │ [a₆+b₆+c₆+d₆+e₆+f₆+g₆+h₆] │
     ├───────────────────────────┼───────────────────────────┤
y=0  │ [a₀+b₀+c₀+d₀+e₀+f₀+g₀+h₀] │ [a₄+b₄+c₄+d₄+e₄+f₄+g₄+h₄] │
     └───────────────────────────┴───────────────────────────┘
       x=0                         x=1

Z = 1 slice:
     ┌───────────────────────────┬───────────────────────────┐
y=1  │ [a₃+b₃+c₃+d₃+e₃+f₃+g₃+h₃] │ [a₇+b₇+c₇+d₇+e₇+f₇+g₇+h₇] │
     ├───────────────────────────┼───────────────────────────┤
y=0  │ [a₁+b₁+c₁+d₁+e₁+f₁+g₁+h₁] │ [a₅+b₅+c₅+d₅+e₅+f₅+g₅+h₅] │
     └───────────────────────────┴───────────────────────────┘
       x=0                         x=1

  RS complete. 8 fully-reduced chunks (Σ₀..Σ₇ where Σᵢ = aᵢ+bᵢ+cᵢ+dᵢ+eᵢ+fᵢ+gᵢ+hᵢ)
  scattered across 8 ranks with no duplication.
```

*Phases 4–6 (Z-AG → Y-AG → X-AG).* Reverse the RS dim order. Each AG phase is a 2-ring pair-exchange with no reduction, doubling per-rank chunk count. Below, $\Sigma_i = a_i+b_i+c_i+d_i+e_i+f_i+g_i+h_i$ as in Phase 3.

```
Starting state for AG (Phase-3 state re-expressed using Σ notation):

Z = 0 slice:
     ┌──────────┬──────────┐
y=1  │   [Σ₂]   │   [Σ₆]   │
     ├──────────┼──────────┤
y=0  │   [Σ₀]   │   [Σ₄]   │
     └──────────┴──────────┘
       x=0        x=1

Z = 1 slice:
     ┌──────────┬──────────┐
y=1  │   [Σ₃]   │   [Σ₇]   │
     ├──────────┼──────────┤
y=0  │   [Σ₁]   │   [Σ₅]   │
     └──────────┴──────────┘
       x=0        x=1
```

*Phase 4 (Z-AG).* 4 concurrent 2-rings: {a,c}, {b,d}, {e,g}, {f,h}. Each Z-pair exchanges its single chunk — a sends Σ₀ to c, c sends Σ₁ to a; similarly for the other three pairs.

```
After Phase 4 (Z-AG) — chunk count 1 → 2 per rank:

Z = 0 slice:
     ┌──────────┬──────────┐
y=1  │ [Σ₂, Σ₃] │ [Σ₆, Σ₇] │
     ├──────────┼──────────┤
y=0  │ [Σ₀, Σ₁] │ [Σ₄, Σ₅] │
     └──────────┴──────────┘
       x=0        x=1

Z = 1 slice:
     ┌──────────┬──────────┐
y=1  │ [Σ₂, Σ₃] │ [Σ₆, Σ₇] │
     ├──────────┼──────────┤
y=0  │ [Σ₀, Σ₁] │ [Σ₄, Σ₅] │
     └──────────┴──────────┘
       x=0        x=1

  Z=0 and Z=1 slices now hold identical chunk pairings — Z-AG copied each
  single chunk to its Z-partner, so ranks sharing an (x, y) position match.
```

*Phase 5 (Y-AG).* 4 concurrent 2-rings: {a,e}, {b,f}, {c,g}, {d,h}. Each Y-pair exchanges its 2 chunks — a sends [Σ₀, Σ₁] to e, e sends [Σ₂, Σ₃] to a; similarly for the other three pairs.

```
After Phase 5 (Y-AG) — chunk count 2 → 4 per rank:

Z = 0 slice:
     ┌─────────────────────┬─────────────────────┐
y=1  │ [Σ₀, Σ₁, Σ₂, Σ₃]    │ [Σ₄, Σ₅, Σ₆, Σ₇]    │
     ├─────────────────────┼─────────────────────┤
y=0  │ [Σ₀, Σ₁, Σ₂, Σ₃]    │ [Σ₄, Σ₅, Σ₆, Σ₇]    │
     └─────────────────────┴─────────────────────┘
            x=0                   x=1

Z = 1 slice:
     ┌─────────────────────┬─────────────────────┐
y=1  │ [Σ₀, Σ₁, Σ₂, Σ₃]    │ [Σ₄, Σ₅, Σ₆, Σ₇]    │
     ├─────────────────────┼─────────────────────┤
y=0  │ [Σ₀, Σ₁, Σ₂, Σ₃]    │ [Σ₄, Σ₅, Σ₆, Σ₇]    │
     └─────────────────────┴─────────────────────┘
            x=0                   x=1

  Both slices identical, and within each slice every x-column is uniform —
  Y-AG merged the (x, z)-pairs, so the 4-chunk set depends only on x.
```

*Phase 6 (X-AG).* 4 concurrent 2-rings: {a,b}, {c,d}, {e,f}, {g,h}. Each X-pair exchanges its 4 chunks — a sends [Σ₀, Σ₁, Σ₂, Σ₃] to b, b sends [Σ₄, Σ₅, Σ₆, Σ₇] to a; similarly for the other three pairs.

```
After Phase 6 (X-AG) — chunk count 4 → 8 per rank. AR complete:

Z = 0 slice:
     ┌───────────────────────────────────┬───────────────────────────────────┐
y=1  │ [Σ₀, Σ₁, Σ₂, Σ₃, Σ₄, Σ₅, Σ₆, Σ₇]  │ [Σ₀, Σ₁, Σ₂, Σ₃, Σ₄, Σ₅, Σ₆, Σ₇]  │
     ├───────────────────────────────────┼───────────────────────────────────┤
y=0  │ [Σ₀, Σ₁, Σ₂, Σ₃, Σ₄, Σ₅, Σ₆, Σ₇]  │ [Σ₀, Σ₁, Σ₂, Σ₃, Σ₄, Σ₅, Σ₆, Σ₇]  │
     └───────────────────────────────────┴───────────────────────────────────┘
                   x=0                                 x=1

Z = 1 slice:
     ┌───────────────────────────────────┬───────────────────────────────────┐
y=1  │ [Σ₀, Σ₁, Σ₂, Σ₃, Σ₄, Σ₅, Σ₆, Σ₇]  │ [Σ₀, Σ₁, Σ₂, Σ₃, Σ₄, Σ₅, Σ₆, Σ₇]  │
     ├───────────────────────────────────┼───────────────────────────────────┤
y=0  │ [Σ₀, Σ₁, Σ₂, Σ₃, Σ₄, Σ₅, Σ₆, Σ₇]  │ [Σ₀, Σ₁, Σ₂, Σ₃, Σ₄, Σ₅, Σ₆, Σ₇]  │
     └───────────────────────────────────┴───────────────────────────────────┘
                   x=0                                 x=1

  Every rank a through h now holds the full reduced vector
  [Σ₀, Σ₁, Σ₂, Σ₃, Σ₄, Σ₅, Σ₆, Σ₇].
```

AR complete: every rank holds $[\Sigma_0, \Sigma_1, \ldots, \Sigma_7]$. Total phase count: $2k = 6$; total sequential α-hops: $2 \sum_i (d_i - 1) = 6$, matching the §3.1 savings comparison.

**Why AG reverses the RS dim order.** The pairing is what makes the cost derivation telescope cleanly. Phase 1 (X-RS) shrinks chunks by $d_x$; the matching Phase 6 (X-AG) grows them back by $d_x$. Phase 2 (Y-RS) shrinks by $d_y$; the matching Phase 5 (Y-AG) grows by $d_y$. Phase 3 (Z-RS) / Phase 4 (Z-AG) pair similarly. Each RS–AG pair moves the same total bytes per rank, and the per-phase BW term depends only on the chunk size at that phase. Running AG in a different dim order would still produce correct output — each dim's AG stage is a self-contained copy — but the chunk-size bookkeeping would not match up against the RS phases and the telescoping derivation below would be messier.

**Cost formula — α-term derivation.** Each dim-$i$ phase runs a $d_i$-rank ring RS (or AG), whose ring-primitive α-cost from `01_collective_algorithms.md` §5.1 is $(d_i - 1)\\,\alpha$: a $d_i$-rank ring takes $d_i - 1$ sequential neighbor-exchange steps to complete one full RS (or AG) traversal. The schedule runs $k$ RS phases followed by $k$ AG phases, each on its own dim's ring, so the total sequential α-hops is the sum over *all* $2k$ phases:

$$n_\alpha = \underbrace{\sum_{i=1}^{k} (d_i - 1)}_{\text{k RS phases}} \\;+\\; \underbrace{\sum_{i=1}^{k} (d_i - 1)}_{\text{k AG phases}} \\;=\\; 2 \sum_i (d_i - 1)$$

**Concrete counts** — the index $i$ ranges over the $k$ dims (for a 3D torus, $i \in \\{1, 2, 3\\}$ with $d_1, d_2, d_3$ the per-axis sizes):

- **2×2×2 torus** ($k = 3$, all $d_i = 2$):
  $$n_\alpha = \sum_{i=1}^{3}(2 - 1) \\;+\\; \sum_{i=1}^{3}(2 - 1) \\;=\\; 3 + 3 \\;=\\; 6$$
  (3 RS hops + 3 AG hops; matches the §3.1 savings comparison.)

- **4×4×4 torus — TPU v4 / v5p single-rack cube** ($N = 64$, $k = 3$, all $d_i = 4$):
  $$n_\alpha = \sum_{i=1}^{3}(4 - 1) \\;+\\; \sum_{i=1}^{3}(4 - 1) \\;=\\; 9 + 9 \\;=\\; 18$$

- **8×8×8 torus — TPU v4 / v5p multi-rack pod slice** ($N = 512$, used as the canonical anchor in §5 and `05_contention_and_congestion.md` §6):
  $$n_\alpha = \sum_{i=1}^{3}(8 - 1) \\;+\\; \sum_{i=1}^{3}(8 - 1) \\;=\\; 21 + 21 \\;=\\; 42$$

- **16×16×16 — TPU v4 / v5p full pod** ($N = 4096$, $k = 3$, all $d_i = 16$):
  $$n_\alpha = \sum_{i=1}^{3}(16 - 1) \\;+\\; \sum_{i=1}^{3}(16 - 1) \\;=\\; 45 + 45 \\;=\\; 90$$

- **Asymmetric 16×16×4** ($k = 3$, $d_1 = d_2 = 16$, $d_3 = 4$):
  $$n_\alpha = \big[(16-1) + (16-1) + (4-1)\big] + \big[(16-1) + (16-1) + (4-1)\big] \\;=\\; 33 + 33 \\;=\\; 66$$

Note that phases are **sequential across dims** (X-RS waits for Y-RS to start; cannot overlap because the next phase's input is the current phase's output). The parallelism is **within** a phase — the $(N/d_i)$ independent dim-rings run concurrently on disjoint wiring (§3.1), which is what keeps the BW term bandwidth-optimal rather than inflating the α term.

**Cost formula — BW-term derivation.** The BW cost for a single dim-$i$ ring RS on a $d_i$-rank ring (from `01_collective_algorithms.md` §5.1) is $\frac{d_i - 1}{d_i} \cdot \frac{M'}{\mathrm{BW}}$, where $M'$ is the **per-rank input size at phase entry**. Each rank sends $(d_i - 1)/d_i$ fraction of its data through its one outgoing link during that ring.

The subtlety: $M'$ shrinks from phase to phase because each RS phase reduces the chunk size by the factor $d_i$ of that phase's ring. On a 3D torus running X-RS → Y-RS → Z-RS, chunk size evolves $M \to M/d_x \to M/(d_x d_y) \to M/(d_x d_y d_z) = M/N$. Per-RS-phase BW contribution:

| RS phase | Ring size | Per-rank input at entry | BW term |
|---|---|---|---|
| 1 (X-RS) | $d_x$ | $M$ | $\frac{d_x - 1}{d_x} \cdot \frac{M}{\mathrm{BW}}$ |
| 2 (Y-RS) | $d_y$ | $M / d_x$ | $\frac{d_y - 1}{d_y} \cdot \frac{M}{d_x \mathrm{BW}} = \frac{d_y - 1}{d_x d_y} \cdot \frac{M}{\mathrm{BW}}$ |
| 3 (Z-RS) | $d_z$ | $M / (d_x d_y)$ | $\frac{d_z - 1}{d_z} \cdot \frac{M}{d_x d_y \mathrm{BW}} = \frac{d_z - 1}{d_x d_y d_z} \cdot \frac{M}{\mathrm{BW}}$ |

Summing the three RS phases:

$$\text{RS BW term} \\;=\\; \frac{M}{\mathrm{BW}} \cdot \left( \frac{d_x - 1}{d_x} \\;+\\; \frac{d_y - 1}{d_x d_y} \\;+\\; \frac{d_z - 1}{d_x d_y d_z} \right)$$

**Telescoping trick.** Rewrite each summand using $(d - 1)/d = 1 - 1/d$:

$$\underbrace{\left(1 - \tfrac{1}{d_x}\right)}_{\text{phase 1}} \\;+\\; \underbrace{\left(\tfrac{1}{d_x} - \tfrac{1}{d_x d_y}\right)}_{\text{phase 2}} \\;+\\; \underbrace{\left(\tfrac{1}{d_x d_y} - \tfrac{1}{d_x d_y d_z}\right)}_{\text{phase 3}} \\;=\\; 1 - \tfrac{1}{d_x d_y d_z} \\;=\\; \frac{N - 1}{N}$$

Inner fractions cancel pairwise — $1/d_x$ from phase 1 cancels $1/d_x$ from phase 2's positive side; $1/(d_x d_y)$ cancels between phase 2 and phase 3 likewise. What survives is $1 - 1/N$. So:

$$\text{RS BW term} \\;=\\; \frac{N - 1}{N} \cdot \frac{M}{\mathrm{BW}}$$

**AG contribution — same derivation in reverse.** AG phases run in reverse dim order (Z-AG → Y-AG → X-AG), each *growing* the chunk size by the paired dim's factor. Ring AG on a $d_i$-rank ring has BW cost $\frac{d_i - 1}{d_i} \cdot \frac{M'}{\mathrm{BW}}$, identical to RS but with $M'$ now the **per-rank output size at phase exit**. Chunk size evolves $M/N \to M/(d_x d_y) \to M/d_x \to M$ across the three AG phases. Per-AG-phase BW contribution:

| AG phase | Ring size | Per-rank output at exit | BW term |
|---|---|---|---|
| 4 (Z-AG) | $d_z$ | $M / (d_x d_y)$ | $\frac{d_z - 1}{d_z} \cdot \frac{M}{d_x d_y \mathrm{BW}} = \frac{d_z - 1}{d_x d_y d_z} \cdot \frac{M}{\mathrm{BW}}$ |
| 5 (Y-AG) | $d_y$ | $M / d_x$ | $\frac{d_y - 1}{d_y} \cdot \frac{M}{d_x \mathrm{BW}} = \frac{d_y - 1}{d_x d_y} \cdot \frac{M}{\mathrm{BW}}$ |
| 6 (X-AG) | $d_x$ | $M$ | $\frac{d_x - 1}{d_x} \cdot \frac{M}{\mathrm{BW}}$ |

Summing the three AG phases:

$$\text{AG BW term} \\;=\\; \frac{M}{\mathrm{BW}} \cdot \left( \frac{d_z - 1}{d_x d_y d_z} \\;+\\; \frac{d_y - 1}{d_x d_y} \\;+\\; \frac{d_x - 1}{d_x} \right)$$

These are the **same three summands as the RS sum** (just indexed in reverse), so the same telescoping collapse applies. Using $(d - 1)/d = 1 - 1/d$:

$$\underbrace{\left(\tfrac{1}{d_x d_y} - \tfrac{1}{d_x d_y d_z}\right)}_{\text{phase 4}} \\;+\\; \underbrace{\left(\tfrac{1}{d_x} - \tfrac{1}{d_x d_y}\right)}_{\text{phase 5}} \\;+\\; \underbrace{\left(1 - \tfrac{1}{d_x}\right)}_{\text{phase 6}} \\;=\\; 1 - \tfrac{1}{N} \\;=\\; \frac{N - 1}{N}$$

Same pairwise cancellation as RS — $1/d_x$ between phases 5 and 6, $1/(d_x d_y)$ between phases 4 and 5, with $1$ and $-1/N$ surviving. So:

$$\text{AG BW term} \\;=\\; \frac{N - 1}{N} \cdot \frac{M}{\mathrm{BW}}$$

**RS and AG are BW-symmetric by construction**: each AG phase moves the same total bytes per rank as its paired RS phase (phase 4 mirrors phase 3, phase 5 mirrors phase 2, phase 6 mirrors phase 1). The byte-count match is why the two sums evaluate to the same $(N-1)/N \cdot M / \mathrm{BW}$. Combined:

$$\text{Total BW term} \\;=\\; \underbrace{\frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}}_{\text{RS}} \\;+\\; \underbrace{\frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}}_{\text{AG}} \\;=\\; 2 \cdot \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}$$

This is the flat-ring bandwidth-optimal floor, unchanged by dim-decomposition.

**Final cost formula**:

$$t_{\mathrm{torus,AR}} \\;=\\; 2 \sum_i (d_i - 1)\\,\alpha \\;+\\; 2 \cdot \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}, \quad N = \prod_i d_i$$

**Concrete BW counts** (AR BW term $= 2 \cdot (N-1)/N \cdot M / \mathrm{BW}$):

- **2×2×2 torus** ($N = 8$): BW term $= 2 \cdot (7/8) \cdot M/\mathrm{BW} = (7/4) \cdot M/\mathrm{BW} \approx 1.75 \cdot M/\mathrm{BW}$.
- **8×8×8 torus** ($N = 512$): BW term $= 2 \cdot (511/512) \cdot M/\mathrm{BW} \approx 1.996 \cdot M/\mathrm{BW}$.
- **16×16×16 TPU v5p** ($N = 4096$): BW term $= 2 \cdot (4095/4096) \cdot M/\mathrm{BW} \approx 2 \cdot M/\mathrm{BW}$.

At large $N$ the factor $(N-1)/N \to 1$, so AR asymptotically costs $2M/\mathrm{BW}$ on BW — same as flat-ring AR's BW bound.

**You pay less α, the same BW as flat ring** — that's the entire arbitrage of dim-decomposition.

**Commercial adoption.** Dim-decomposed ring AR is the default AR primitive on production torus fabrics:

- **Google TPU v4 / v5p** — 3D torus up to $16 \times 16 \times 16$ per slice; XLA / JAX emit dim-decomposed ring RS+AG for every sharded reduction [TPU-V4].
- **AWS Trainium2** — each Trn2 instance is a 2D NeuronLink torus of 16 chips; Trn2 UltraServer composes four instances into a 3D 64-chip torus via a Z-dim NeuronLink [TRN2-ARCH]. NeuronX CCL ships dim-decomposed ring AR as the default kernel [NEURON-CC].

Message-size-specialized variants and runtime-scheduler tuning are outside this note's topology-mapping scope.

**Floating-point limitation.** Dim-decomposed AR reorders the floating-point sum — a rank adds partial sums coming from different dim-phases — which is not bit-identical to a flat-ring ordering. This is the one numerical caveat specific to AR; §3.5's standalone AG drops it (no reduction, bit-copying is exact) while standalone RS inherits it.

The caveat is scoped to **floating-point** only. Integer reductions — quantized-gradient AR, histogram aggregation, counting workloads — are unaffected because two's-complement integer addition is associative modulo $2^n$: any AR schedule (flat ring, dim-decomposed, tree, hierarchical) produces bit-identical output for integer inputs, with overflow (if any) occurring deterministically.

### 3.5 All-gather / reduce-scatter

**Relation to §3.4.** AG and RS are the two halves of dim-decomposed AR, so everything from §3.4 carries over directly — the X-phase / Y-phase decomposition, the dim-decomposed ring primitive (with the Rabenseifner-per-dim variant in Appendix A as a reference point), the disjoint-wiring parallelism argument (per §3.1), and the chunk-size telescoping.

**Cost formula.** Cost halves vs §3.4 because only one half (RS or AG) runs:

$$t_{\mathrm{torus,AG\\,or\\,RS,ring}} = \sum_i (d_i - 1)\\,\alpha + \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}$$

**Commercial shipment.** Dim-decomposed ring AG and RS ship on the same production torus stacks as the AR variant in §3.4: TPU via XLA / JAX, Trn2 via NeuronX CCL. The Rabenseifner-per-dim variant (Appendix A) is not shipped as an AG / RS kernel for the reasons discussed there.

**Limitations.** The §3.4 limitations (plus the Appendix A caveats if the Rabenseifner path is ever chosen) all apply verbatim with one relaxation: standalone AG has no reduction, so the floating-point non-associativity concern (§3.4's "FP limitation" paragraph) drops out entirely — any AG schedule on any layout produces bit-identical output because bit-copying is exact. Standalone RS still reduces and inherits the AR floating-point behavior unchanged.

### 3.6 All-to-all

**Relation to `01_collective_algorithms.md` §7.** The shipped algorithm is **ring relay** (§7.1) — §7.3's selection rule pegs ring relay as the natural fit on bisection-limited fabrics precisely because not every rank pair has a direct fabric path, so shortest-arc forwarding through intermediate ranks is the only reachable schedule; pairwise direct-send (§7.2) requires the full bisection that torus lacks. The α-β cost $(N-1)\alpha + (N-1)/N \cdot M/\mathrm{BW}$ and the partner-selection / chunk-routing / pre- and post-rotation bookkeeping port over verbatim from §7.1. What changes on a torus is the *delivered* bandwidth. A2A has no RS+AG-style decomposition to hide behind (§7 opener makes this explicit: aggregate cross-fabric traffic is $(N-1) M$ bytes regardless of schedule), so unlike AR in §3.4 — where per-dim ring kept torus BW at the star level while only paying more α hops — A2A pays a hard bisection penalty on torus that no scheduling trick can recover on the BW side, while still enjoying the latency compression from dim-decomposition on the α side. **Whether torus helps or hurts vs star therefore depends entirely on which term dominates at the target $(N, M, \alpha, \mathrm{BW})$.**

**How ring-relay A2A moves packets on a torus.** Each rank has $N$ chunks of size $M/N$, one destined for each rank (including itself). Under ring-relay, each chunk travels from source to destination along a **shortest-arc path** through the torus lattice, hop by hop. Intermediate ranks forward chunks they don't own without combining, reducing, or holding — pure point-to-point forwarding. Multiple chunks move concurrently: every link can carry an independent in-flight chunk on each direction of its full-duplex channel.

To make this concrete, label the 16 ranks of the 4×4 torus with letters a–p and write chunk $x{\to}Y$ for the chunk held by rank $x$ destined for rank $Y$:

```
4×4 torus — ranks labeled a–p (wraparound on every row and column):

             ↕    ↕    ↕    ↕           ↕ Y-wraparound per column:
          ┌──┴──┬──┴──┬──┴──┬──┴──┐         a ↔ m, b ↔ n, c ↔ o, d ↔ p
  ↔ y=3   │  m  │  n  │  o  │  p  │ ↔
          ├─────┼─────┼─────┼─────┤     ↔ X-wraparound per row:
  ↔ y=2   │  i  │  j  │  k  │  l  │ ↔       a ↔ d, e ↔ h, i ↔ l, m ↔ p
          ├─────┼─────┼─────┼─────┤
  ↔ y=1   │  e  │  f  │  g  │  h  │ ↔     (these wraparound edges are the
          ├─────┼─────┼─────┼─────┤       "seams" that distinguish torus
  ↔ y=0   │  a  │  b  │  c  │  d  │ ↔       from the same-shape k-D mesh
          └──┬──┴──┬──┴──┬──┴──┬──┘         in §4.2, which has them cut)
             ↕    ↕    ↕    ↕
             x=0  x=1  x=2  x=3

  initial state: each rank x holds 16 chunks [x→a, x→b, x→c, ..., x→p],
                 one destined for each rank (including itself).
  final state:   each rank y holds 16 chunks [a→y, b→y, c→y, ..., p→y],
                 one received from each source.
```

**Sample chunk journeys from rank a.** Rank a has 15 chunks to dispatch (keeping $a{\to}a$ for itself). Four representative routes, showing the range of path lengths and how wraparound can serve as an equally-short alternative:

```
a→b   direct east, 1 hop:              a ──→ b                        (1 hop)

a→d   via X-wraparound, 1 hop:         a ═══ d                        (1 hop)
                                        └─ X-wrap (a↔d), not a→b→c→d (3 hops)

a→c   east, relay through b:           a ──→ b ──→ c                  (2 hops)
                                        step 1 step 2

a→k   diameter, X then Y (direct):     a ──→ b ──→ c ──→ g ──→ k     (4 hops)
                                         X     X     Y     Y

a→k   alternative via X-wraparound:    a ══─► d ──→ c ──→ g ──→ k     (4 hops)
                                        X-wrap  X     Y     Y
      (also: a→e→f→g→k Y-first, a→b→f→j→k X-Y-X-Y, a→m→n→o→k via
       Y-wrap + X hops, etc. — multiple equally-short paths exist.)
```

At step 1, rank a injects 4 chunks — one onto each of its 4 outbound links (east to b, west via wraparound to d, north to e, south via wraparound to m). At step 2, rank b has received $a{\to}b$ (keep it — destination reached) and also forwards $a{\to}c$ east, while rank a injects the next-hop chunks. The schedule continues; by step $\mathrm{diam} = 4$, every chunk has arrived. All 15 other ranks run the same protocol in parallel with their own 15 chunks each, so 240 chunks are in flight concurrently across the fabric.

From this mechanism the two cost terms split cleanly:

- **α-term**: every chunk takes at most $\mathrm{diam} = \sum_i \lfloor d_i / 2 \rfloor$ hops. All ranks forward concurrently, so the wall-clock α-cost is set by the longest single-chunk journey — *not* the sum of all chunk hops. For the 4×4: diameter = 4, so α-term = $4\alpha$. (Contrast §3.4's ring RS, where α-cost was $(d - 1)\alpha$ per phase because chunks moved in lockstep one-hop-per-step; A2A ring-relay doesn't step-sync.)
- **BW-term**: ring RS's step-by-step summing (§3.4) relies on uniform per-step link loading, which A2A's pipelined, asymmetric traffic doesn't provide. Appendix D develops the replacement framework — **bottleneck analysis** (`re-use count × per-use time`) — and works this case end-to-end. The structural fact for torus A2A is that cross-half traffic can only use the **bisection cut** links, so the cut's aggregate capacity upper-bounds cross-half throughput, and since every cross-cut chunk uses exactly one cut edge the total byte-count is fixed by A2A's semantics (not by scheduling).

**Bisection cut for the 4×4 torus.** Drawing a vertical line between columns $x = 1$ and $x = 2$ partitions the ranks into two equal halves, severing the X-axis links that connect left to right:

```
Bisection cut (│ marks severed X-links; ranks labeled as in the grid above):

                    │         │
  y=3       m ─── n │ o ─── p │ (X-wrap m ↔ p crosses the cut)
  y=2       i ─── j │ k ─── l │ (X-wrap i ↔ l crosses the cut)
  y=1       e ─── f │ g ─── h │ (X-wrap e ↔ h crosses the cut)
  y=0       a ─── b │ c ─── d │ (X-wrap a ↔ d crosses the cut)
                    │         │
            x=0   x=1   x=2   x=3

  Per row: 1 direct cut (between x=1 and x=2) + 1 wraparound cut = 2 severed links.
  4 rows × 2 = 8 severed full-duplex links total; each at per-link BW.
```

Each left-half rank owns exactly one outbound cut link — 4 direct links at $x{=}1 \to x{=}2$ and 4 X-wraparound links at $x{=}0 \to x{=}3$ — so the cut has 8 severed edges total.

**Deriving the BW term.** Using a **single-link view**: pick any one cut link, count how many chunks re-use it over the phase, and multiply by the per-chunk transit time. Assuming ideal pipelining — each re-use back-to-back, with the link at 100% utilization — the phase ends when the link's last occupant finishes its hop, so the phase time equals that link's total busy-time. (By symmetry on a regular torus every cut link carries the same load, so any one link works.)

Focus on `b→c` (direct cut at $y=0$, L→R direction). Under uniform shortest-path routing, the 64 cross-cut chunks distribute across the 8 cut links → 8 chunks per cut link. The 8 chunks passing through `b→c` under X-first routing from row $y=0$ to the $x=2$ column:

| # | Chunk | Route | Total hops |
|---|---|---|---|
| 1 | $a \to c$ | $a \to b \to c$ | 2 |
| 2 | $a \to g$ | $a \to b \to c \to g$ | 3 |
| 3 | $a \to k$ | $a \to b \to c \to g \to k$ | 4 |
| 4 | $a \to o$ | $a \to b \to c \to o$ *(Y-wrap on last hop)* | 3 |
| 5 | $b \to c$ | $b \to c$ | 1 |
| 6 | $b \to g$ | $b \to c \to g$ | 2 |
| 7 | $b \to k$ | $b \to c \to g \to k$ | 3 |
| 8 | $b \to o$ | $b \to c \to o$ *(Y-wrap on last hop)* | 2 |

Each chunk uses `b→c` once for $M/(16\\,\mathrm{BW})$. With 8 back-to-back re-uses:

$$t_{\mathrm{BW}} \\;=\\; 8 \cdot \frac{M}{16\\,\mathrm{BW}} \\;=\\; \frac{M}{2\\,\mathrm{BW}}$$

Appendix D.3 derives the same result via the equivalent aggregate whole-cut view (4M cross-cut bytes / $8\\,\mathrm{BW}$) and contrasts both against per-chunk-per-step reasoning.

**Generalizing to arbitrary torus shape.** Extending the whole-cut view to arbitrary $d_i$, the BW term scales to $d_{\max}/8 \cdot M/\mathrm{BW}$. Appendix D.3 tracks how each of the three whole-cut quantities (severed-link count, cross-cut traffic, cut throughput) scales with $N$ and $d_{\max}$, and derives this formula.

**General cost formula.** Combining the BW term with the α-term (one full ring-relay traversal across the diameter):

$$t_{\mathrm{torus,A2A}} \\;\approx\\; \mathrm{diam} \cdot \alpha \\;+\\; \frac{d_{\max}}{8} \cdot \frac{M}{\mathrm{BW}}, \qquad \mathrm{diam} = \sum_i \lfloor d_i / 2 \rfloor$$

The bandwidth term scales with $d_{\max}$ — *not* $N$. On star (the pairwise-direct baseline from `01_collective_algorithms.md` §7.2) the equivalent BW term is $(N-1)/N \cdot M/\mathrm{BW} \approx M/\mathrm{BW}$; the torus-vs-star penalty is therefore $d_{\max}/8$.

> **Aside — "bisection bandwidth" in other literature.** Network-design texts often introduce **bisection bandwidth** as a named quantity — the total one-direction throughput across the worst-case equal-halves cut — and write the A2A BW term as $(NM/4) / \mathrm{BW_{bisect}}$. For a wraparound torus, $\mathrm{BW_{bisect}} = (2N/d_{\max}) \cdot \mathrm{BW}$, which is exactly what "severed-link count × per-link BW" evaluates to in the Appendix D.3 derivation. This note keeps the per-link BW as the only symbol and computes the cut throughput inline; readers encountering $\mathrm{BW_{bisect}}$ elsewhere should mentally map it to $(2N/d_{\max}) \cdot \mathrm{BW}$ on a torus, or $s \cdot N \cdot \mathrm{BW}$ with $s$ the oversubscription ratio on a fat-tree (`03_hierarchical_topologies.md` §1).

**Dim-decomposed A2A — alternative schedule.** Running $k$ successive per-dim A2As (§3.1-style decomposition, with bidirectional ring-relay A2A as the per-dim primitive — each phase runs A2A on every $d_i$-ring in parallel, using dim-$i$ wiring alone) gives the approximate cost

$$t_{\mathrm{torus,A2A,decomp}} \\;\approx\\; \mathrm{diam} \cdot \alpha \\;+\\; \frac{\sum_i d_i}{8} \cdot \frac{M}{\mathrm{BW}}$$

— per-rank holding stays $M$ across phases (data redistributes without changing total per-rank volume), so each phase costs $\lfloor d_i/2 \rfloor \alpha + d_i/8 \cdot M/\mathrm{BW}$ and the $k$ phases sum sequentially. The α-term matches flat's $\mathrm{diam} \cdot \alpha$; the BW-term is worse than flat's $d_{\max}/8$ by factor $(\sum_i d_i) / d_{\max}$ — ${\sim}k\times$ on a cubic $k$-D torus — because dim-decomposed idles $k{-}1$ dims' wiring per phase while flat uses all dims concurrently via shortest-arc routing.

**Commercial shipment.** Both flat ring-relay A2A (`01_collective_algorithms.md` §7.1, shortest-arc forwarding on a linearized rank order) and the dim-decomposed schedule above ship on TPU and Trainium as the MoE expert-dispatch / expert-combine primitive. The bisection penalty is the dominant design pressure: TPU v4's optical circuit switches [TPU-V4] exist partly so that operators can *reshape* the torus slice into the most cubic layout that matches the MoE's expert count, minimizing $d_{\max}$. Trainium's Trn2 topology [TRN2-ARCH] is fixed at 2D per instance + Z-dim across instances — roughly cubic for 64-chip UltraServers — by the same reasoning. MoE workloads that outgrow a single square slice and must span multiple pods see sharp A2A-cost cliffs, which is why production MoE training/inference pipelines pin expert count to match the slice's natural dimensions whenever possible.

**Limitations (new vs §3.4 / §3.5).**

1. **No decomposition escape for the BW term.** The dim-decomposed AR from §3.4 kept torus BW at star-level by routing each per-dim phase over disjoint wiring. For A2A, dim-decomposition is strictly *worse* than flat (${\sim}k\times$ on BW, as derived above) — splitting into per-dim A2As idles most of the fabric each phase, and the total cross-bisection aggregate is fixed by the collective's semantics regardless of schedule. The $d_{\max}/8$ penalty of flat A2A is architectural, not algorithmic.
2. **Layout sensitivity is severe.** Unlike AR (§3.4 note), A2A cost scales with $d_{\max}$ rather than $\sum (d_i - 1)$. A $16 \times 8 \times 4$ layout is 2× worse than $8 \times 8 \times 8$ for the same $N = 512$; a $32 \times 4 \times 4$ layout is 4× worse. Operators tuning MoE on torus spend real effort finding the most cubic viable slice.

**What mitigates the bisection penalty.** Two mitigations actually move the A2A cost on torus:

- **Shape choice — minimize $d_{\max}$.** Pick the most cubic layout at the given rank count (cubic > skinny; higher-D > lower-D). For $N = 512$, moving from $32 \times 16$ (2D, $d_{\max} = 32$) to $8 \times 8 \times 8$ (3D, $d_{\max} = 8$) is a 4× BW improvement for free. On TPU v4 / v5p this is a job-launch decision because OCS [TPU-V4] can reconfigure the physical slice shape (and dimensionality) between jobs; on fixed-topology systems like Trainium [TRN2-ARCH] it's baked into the hardware generation.
- **Per-link ICI bandwidth bumps.** Each TPU / Trainium generation raises per-link $\mathrm{BW}$, shrinking the absolute BW term even though the $d_{\max}/8$ coefficient is unchanged.

---

## 4. Mesh topology

Mesh is the direct-wiring complement to star and the wraparound-free complement to torus. Two regimes use the word "mesh" interchangeably in the literature: **full mesh** (§4.1), where every rank-pair has a dedicated link, and **$k$-D mesh** (§4.2), a torus with the wraparound edges removed. Cost behavior can be summarized as "star minus the switch α" for full mesh and "torus minus the wraparound bisection" for $k$-D mesh — this section makes each precise.

### 4.1 Full mesh

Full mesh replaces the central switch with $N(N-1)/2$ direct edges, one per rank pair. Every collective primitive runs at its star cost with one adjustment: the per-hop α drops the switch cut-through term and becomes pure endpoint-to-endpoint link latency. On chiplet-scale copper this is ~100–300 ns per hop (vs NVSwitch's ~500 ns cut-through); at board-scale NVLink it is comparable to star's α because the PHY cost dominates. **No switch ASIC means no in-network reduction path** — SHARP / NVLS are structurally unavailable — and **no switch multicast**, so AG and BC cannot benefit from switch-resident replication.

**Algorithm coverage.** Full mesh has any-to-any direct links between every rank pair, so every logical edge in any schedule from `01_collective_algorithms.md` lands on exactly one physical link. All software algorithms, such as ring, binomial tree, DBT, rec-doubling / rec-halving, Rabenseifner, pairwise direct-send, Bruck, and PAT, run at their pure α-β costs, with the single substitution $\alpha \to \alpha_{\mathrm{link}}$. The lone exception is the INC paths (SHARP / NVLS / switch multicast): full mesh has no switch ASIC, so the in-network formulas from `01_collective_algorithms.md` §3.3 / §4.3 / §5.3 do not apply.

**Example — ring AR.** The `01_collective_algorithms.md` §5.1 ring-AR cost

$$t_{\mathrm{ring\\,AR}} = 2(N-1)\\,\alpha + 2\\,\frac{N-1}{N}\cdot\frac{M}{\mathrm{BW}}$$

maps onto full mesh as

$$t_{\mathrm{mesh,\\,ring\\,AR}} = 2(N-1)\\,\alpha_{\mathrm{link}} + 2\\,\frac{N-1}{N}\cdot\frac{M}{\mathrm{BW}}$$

Same formula; only α changes (switch cut-through term drops, leaving pure PHY-to-PHY link latency). Every other algorithm substitutes the same way — no topology correction beyond this α calibration.

In the pure α-β model, A2A gains essentially nothing over star: NVSwitch-class star already hits the per-port BW bound via pairwise direct-send, and removing the switch at most recovers the cut-through time. Note the $(N-1)/N$ coefficient is serial over each rank's single outbound α-β port-slot per step — the $N-1$ dedicated wires don't dissolve the per-rank port budget in the pure α-β model, so the formula has the same serialized shape as star's. The structural advantage of full mesh is not cost-theoretic; it is that **dedicated per-pair links carry no inter-pair contention** — which matters under concurrent collective groups (scored in `05_contention_and_congestion.md`).

**Commercial shipment — where full mesh actually appears.** Three regimes dominate:

1. **Chiplet interposer (UCIe, EMIB, HBM base-die):** small $N$ (2–8 chiplets), direct PHY-to-PHY links on silicon / organic substrate; no switch in the package.
2. **Legacy pre-NVSwitch NVLink (DGX-1 / DGX-2 hybrid cube-mesh):** 8-way or 16-way NVLink-1/2 used hybrid cube-mesh wiring where each GPU linked directly to a subset of peers. NVSwitch (NVLink-3+) replaced this because full mesh's $O(N^2)$ wire count stops scaling past ~16 endpoints.
3. **PCIe peer-to-peer inside a root complex:** GPUs DMA to each other over per-pair PCIe lanes; an informal full-mesh analog, though traffic in practice still traverses a root-complex switch.

**Limitations.**

1. **Scalability wall at $O(N^2)$ wires.** Each added rank needs $N$ new links; total wire count grows quadratically. Practical limit is 8–16 ranks on any substrate; beyond that, switches win on wire-count economics.
2. **No switch means no in-network primitive.** The $n_\alpha \to 2$ collapse via NVLS / SHARP is unavailable; switch-multicast-accelerated AG is unavailable. Full mesh is the "purest" α-β fabric in that sense — every cost reduces to the analytical formula with no hardware-side shortcut.

### 4.2 $k$-D mesh

A $k$-D mesh is a torus **without** the wraparound edges. The torus neighbor formula $\sum_i \min(d_i - 1, 2)$ (§1.2) becomes the **interior-rank ceiling** that only ranks not on any boundary reach; a rank at position 0 or $d_i - 1$ on any axis with $d_i \geq 3$ has 1 neighbor on that axis instead of 2 (wraparound missing). Axes with $d_i \leq 2$ behave the same in mesh and torus (wraparound was already degenerate at $d_i = 2$ or absent at $d_i = 1$). Concretely: for a 3×3×3 mesh, the body-center rank (1, 1, 1) still has 6 neighbors, a face-center rank like (1, 1, 0) has 5, an edge-midpoint like (0, 1, 0) has 4, and a corner like (0, 0, 0) has only 3 — one per non-boundary direction. The dim-decomposition argument from §3.1 carries over verbatim — each dim's RS/AG runs on a $d_i$-rank **open line** rather than a $d_i$-rank closed ring.

**Algorithm coverage — what transfers from torus (§3).** The dim-decomposition framework (§3.1) and every per-primitive derivation in §3.2–§3.6 port over directly with "open $d_i$-line" substituted for "$d_i$-ring" as the per-dim inner primitive. Bucket-brigade RS/AG on an open line has the same α and BW as ring RS/AG (chunk-size telescoping is identical — each rank still sends $(d_i - 1)/d_i$ of its chunk per half); pipelined per-dim BC/Reduce inherit the same $\sum_i (d_i - 1)\alpha + M/\mathrm{BW}$ form. So the unidirectional torus formulas transfer verbatim:

$$t_{\mathrm{mesh,AR}} = 2\sum_i (d_i - 1)\\,\alpha + 2\\,\frac{N-1}{N}\cdot\frac{M}{\mathrm{BW}}, \qquad t_{\mathrm{mesh,BC}} = t_{\mathrm{mesh,Reduce}} = \sum_i (d_i - 1)\\,\alpha + \frac{M}{\mathrm{BW}}$$

(AG / RS are half of AR.) Like torus, mesh is link-only — no switch ASIC, so the INC paths from `01` §3.3 / §4.3 / §5.3 are unavailable.

**What does *not* transfer.** Two effects distinguish mesh from torus:

1. **Bisection halves → A2A BW doubles.** The bisection cut along the longest dim severs $N/d_{\max}$ links — half of the torus's $2N/d_{\max}$ (wraparound edges are missing). The flat A2A BW term rises from torus's $d_{\max}/8 \cdot M/\mathrm{BW}$ to $d_{\max}/4 \cdot M/\mathrm{BW}$:

   $$t_{\mathrm{mesh,A2A}} \approx \mathrm{diam}\cdot\alpha + \frac{d_{\max}}{4}\cdot\frac{M}{\mathrm{BW}}, \qquad \mathrm{diam} = \sum_i (d_i - 1)$$

   **Mesh A2A is consistently 2× worse than same-shape torus A2A.** At cubic $8 \times 8 \times 8$ the penalty is $d_{\max}/4 = 2\times$ vs star's $\approx M/\mathrm{BW}$; a 2D $32 \times 16$ mesh pays $8\times$, vs 2D torus's $4\times$. The dim-decomposed A2A alternative (§3.6-style) inherits the same 2× cut penalty per phase: $t_{\mathrm{mesh,A2A,decomp}} \approx \mathrm{diam}\cdot\alpha + (\sum_i d_i)/4 \cdot M/\mathrm{BW}$.

2. **Bidirectional BC/Reduce α-halving doesn't apply cleanly.** Torus's bidirectional ring BC/Reduce (§3.2/§3.3) cuts per-dim α to $\lfloor d_i/2 \rfloor$ by radiating through both ring directions — including the wrap. On an open line the wrap is absent: a source at one end (typical for mesh BC) reaches the far end only via unidirectional forward, so α stays at $(d_i - 1)$ per dim. A source at the middle *can* use both directions concurrently to achieve $\lceil (d_i-1)/2 \rceil$ α, but that's a source-placement optimization, not a substrate property. The formulas above use the worst-case unidirectional α.

**Scheduling caveat on realized BW.** The α-β formulas assume bidirectional pipelining that keeps every link fully utilized — matching the ring case's uniform per-link load. A naïve unidirectional schedule on an open line can realize up to 2× worse BW because middle links bottleneck while endpoint links idle; well-tuned implementations hit the formula, but worst-case-bounded cost models should apply a factor up to 2 on the BW term when scheduling discipline is not guaranteed.

**Commercial shipment.** $k$-D mesh appears primarily as an implementation detail rather than a top-level AI-fabric choice: Intel's Xeon Phi KNL used a 2D mesh-on-die for its CHA tile interconnect; some pre-production Trainium / TPU prototypes used meshes before upgrading to torus; chiplet-scale HBM base-die interconnects often use small 2×2 or 4×4 meshes because on-silicon wraparound wiring is expensive and dim sizes are tiny. In production large-scale AI fabrics, torus dominates — the wraparound cost is small (~$N$ extra links) and the 2× bisection payoff is large.

**Limitations.**

1. **2× bisection penalty vs same-shape torus.** Applies to A2A directly and, under concurrent traffic, to any primitive that saturates bisection. For pure AR/AG/RS the α-β formula matches torus, but the real link-load skew (middle-heavy on open lines) erases part of that at high utilization.
2. **Dim-decomposition still applies unchanged.** Ring-per-dim becomes line-per-dim. The correctness proof is identical; only the per-dim endpoint behavior changes.

---

## 5. Summary and limitations

### 5.1 Cost summary by topology

Cost table for each (topology, primitive) pair at its dominant shipped algorithm under the ideal α-β model (no contention, uniform per-link BW, uniform per-hop $\alpha$; torus / k-D mesh rows assume contiguous dim-aligned group placement). All BW terms assume asymptotic pipelining (the $P \to P^*$ limit from `01_collective_algorithms.md` Appendix C).

| Topology | Primitive | Algorithm | Latency term | BW term | Shipped by |
|---|---|---|---|---|---|
| **Star** (§2) | BC | Binomial tree (pipelined) | $\lceil \log_2 N \rceil\\,\alpha$ | $M/\mathrm{BW}$ | NCCL (default BC path) |
|  | Reduce | Binomial tree (pipelined) | $\lceil \log_2 N \rceil\\,\alpha$ | $M/\mathrm{BW}$ | NCCL (default Reduce path) |
|  | AR | Double binary tree (pipelined) | $2\\,\lceil \log_2 N \rceil\\,\alpha$ | $M/\mathrm{BW}$ | NCCL (small–mid $M$) |
|  | AR | Ring | $2(N{-}1)\\,\alpha$ | $2(N{-}1)/N \cdot M/\mathrm{BW}$ | NCCL (bulk $M$) |
|  | AG / RS | Ring | $(N{-}1)\\,\alpha$ | $(N{-}1)/N \cdot M/\mathrm{BW}$ | NCCL |
|  | A2A | Pairwise direct-send | $(N{-}1)\\,\alpha$ | $(N{-}1)/N \cdot M/\mathrm{BW}$ | NCCL |
| **Torus** (§3) | BC | Dim-decomp ring (bidirectional) | $\sum_i \lfloor d_i/2 \rfloor\\,\alpha$ | $M/\mathrm{BW}$ | TPU, Trainium |
|  | Reduce | Dim-decomp ring (bidirectional) | $\sum_i \lfloor d_i/2 \rfloor\\,\alpha$ | $M/\mathrm{BW}$ | TPU, Trainium |
|  | AR | Dim-decomp ring | $2 \sum_i (d_i{-}1)\\,\alpha$ | $2(N{-}1)/N \cdot M/\mathrm{BW}$ | TPU (XLA / JAX), Trainium (NeuronX CCL) |
|  | AG / RS | Dim-decomp ring | $\sum_i (d_i{-}1)\\,\alpha$ | $(N{-}1)/N \cdot M/\mathrm{BW}$ | TPU, Trainium |
|  | A2A | Ring relay (bisection-bound) | $\mathrm{diam}\cdot\alpha$, $\mathrm{diam} = \sum_i \lfloor d_i/2 \rfloor$ | $(d_{\max}/8) \cdot M/\mathrm{BW}$ | TPU, Trainium |
| **Full mesh** (§4.1) | BC | Binomial tree (pipelined) | $\lceil \log_2 N \rceil\\,\alpha_{\mathrm{link}}$ | $M/\mathrm{BW}$ | Chiplet interposer (UCIe), DGX-1/2 hybrid cube-mesh (legacy), PCIe P2P |
|  | Reduce | Binomial tree (pipelined) | $\lceil \log_2 N \rceil\\,\alpha_{\mathrm{link}}$ | $M/\mathrm{BW}$ | same |
|  | AR | Double binary tree (pipelined) | $2\\,\lceil \log_2 N \rceil\\,\alpha_{\mathrm{link}}$ | $M/\mathrm{BW}$ | same |
|  | AR | Ring | $2(N{-}1)\\,\alpha_{\mathrm{link}}$ | $2(N{-}1)/N \cdot M/\mathrm{BW}$ | same |
|  | AG / RS | Ring | $(N{-}1)\\,\alpha_{\mathrm{link}}$ | $(N{-}1)/N \cdot M/\mathrm{BW}$ | same |
|  | A2A | Pairwise direct-send | $(N{-}1)\\,\alpha_{\mathrm{link}}$ | $(N{-}1)/N \cdot M/\mathrm{BW}$ | same |
| **k-D mesh** (§4.2) | BC | Dim-decomp open-line | $\sum_i (d_i{-}1)\\,\alpha$ | $M/\mathrm{BW}$ | Chiplet HBM base-die, Xeon Phi KNL tile mesh |
|  | Reduce | Dim-decomp open-line | $\sum_i (d_i{-}1)\\,\alpha$ | $M/\mathrm{BW}$ | same |
|  | AR | Dim-decomp open-line | $2 \sum_i (d_i{-}1)\\,\alpha$ | $2(N{-}1)/N \cdot M/\mathrm{BW}$ | same |
|  | AG / RS | Dim-decomp open-line | $\sum_i (d_i{-}1)\\,\alpha$ | $(N{-}1)/N \cdot M/\mathrm{BW}$ | same |
|  | A2A | Ring relay (bisection halved) | $\mathrm{diam}\cdot\alpha$, $\mathrm{diam} = \sum_i (d_i{-}1)$ | $(d_{\max}/4) \cdot M/\mathrm{BW}$ | same |

Six observations that the rest of this series builds on:

1. **Torus preserves BW-optimality for AR / AG / RS but pays in α.** The torus ring entries keep the same $(N{-}1)/N$ (or $2(N{-}1)/N$) BW coefficient as their star counterparts — dim-decomposition routes each phase over disjoint per-dim copper, so no BW is wasted. The torus-vs-star gap is entirely in the α term: $\sum (d_i{-}1)$ vs star's $\log_2 N$ (DBT) or $N{-}1$ (ring). Production uses ring on both sides because the α gap is small once $M$ is large.
2. **BC and Reduce hit the $M/\mathrm{BW}$ BW ceiling under pipelining, on every topology.** The entire topology-vs-topology gap for BC / Reduce lives in the α term: $\log_2 N$ (star tree), $\sum_i \lfloor d_i/2 \rfloor$ (bidirectional torus), $\sum_i (d_i{-}1)$ (open-line k-D mesh or unidirectional torus). The wraparound halves torus's α relative to same-shape k-D mesh; star's log-depth α beats both at any $N$ the switch can accommodate.
3. **Torus pays a hard BW penalty on A2A under asymmetric shapes.** The $d_{\max}/8 \cdot M/\mathrm{BW}$ BW term has no algorithmic escape — it's set by the bisection cut, not by the schedule. At cubic $N^{1/k}$ per dim the penalty is $1\times$ (torus matches star's BW term exactly), but it scales linearly with $d_{\max}$ and reaches $32\times$ at $256 \times 2 \times 2$ — which is why torus pods aggressively reshape slices toward cubic layouts for MoE workloads.
4. **k-D mesh is torus with a 2× A2A penalty.** For AR / AG / RS / BC / Reduce, k-D mesh's open-line substitutes for torus's ring with no change to the unidirectional α form ($\sum (d_i{-}1)\alpha$ either way) and no change to BW ($(N{-}1)/N \cdot M/\mathrm{BW}$ telescoping still holds on bucket brigade). For A2A, missing wraparound edges halve the bisection cut, so the BW term is $d_{\max}/4$ vs torus's $d_{\max}/8$ — exactly $2\times$ worse.
5. **Tree-flavored algorithms ship on star but not on torus / mesh.** DBT is NCCL's default on switched fabrics and is selected by the tuner for small-$M$ AR; the structurally analogous dim-decomposed Rabenseifner variant on torus (Appendix A) is absent from both TPU and Trainium runtime stacks because the 1.5–4× α compression at production dim sizes $d_i \in \\{4, 8, 16\\}$ rarely beats the simpler ring kernel in practice. This is a fabric-economics decision, not a correctness one.
6. **Star has an additional α-compression escape hatch via in-network collectives — direct-wired topologies (torus, full mesh, k-D mesh) do not.** Within a switched fabric, the reduction operation can move into the switch ASIC itself: switch-resident ALUs reduce flits on the fly and multicast the result back, collapsing $n_\alpha$ from $\log_2 N$ (DBT) all the way to $2$, independent of $N$. NVLS (NVSwitch), Quantum SHARP (InfiniBand fat-tree / Clos), and Tomahawk Ultra INC (Ethernet) all exploit this. Direct-wired topologies have no analogous path — the $N$-dependent $\alpha$ term comes from neighbor-router hops, which cannot be collapsed by moving the reduce into a central switch that does not exist. See `04_in_network_collectives.md` for the full mechanism and cost model.

### 5.2 Limitations

Every cost formula above assumes the collective runs **alone** with perfect link-level scheduling. Real deployments break this in four ways; `05_contention_and_congestion.md` covers each with coefficient models.

1. **Concurrent collective groups.** DP replicas simultaneously issue TP all-reduces. On a star, non-overlapping port subsets handle them trivially. On torus, if all replicas share a common dim, they contend for the same physical links. Best-case torus assumes this doesn't happen.
2. **Off-prefix group layouts.** Torus dim-decomp only hits $2 \sum(d_i - 1)$ hops when the group ranks are contiguous along dim prefixes. A scatter-pattern allocator (typical in shared clusters) produces groups whose ranks are spread arbitrarily — the physical hop count can be 2–4× the ideal, and the formula falls back to flat-ring bound.
3. **Skewed A2A traffic.** Real MoE routing shows 3–10× skew between hot and cold experts. The uniform-bisection bound underestimates tail latency. Star shrugs this off (every port has the same BW); torus pays — hot-expert destinations concentrated along one dim saturate that dim's bisection well before the uniform bound predicts.
4. **Mixed traffic classes.** Gradient AR, activation AG, optimizer RS, KV cache streaming, and control messages compete for the same fabric — the specifics depend on workload (training vs inference, dense vs MoE), but the structural point is the same: wormhole / cut-through routing keeps the BW per message stable at peak utilization but adds queuing delay at high saturation.

Also worth noting: everything above assumes the software picks the right algorithm for the fabric. Any dispatch layer (NCCL, MPI, a custom tuner) that maps a fabric to its matching algorithm delivers these costs; one that forces flat-ring AR on a torus will see the flat-ring cost, not the dim-decomp cost — the formula follows the algorithm, not the wiring.

---

## Appendix A: Dim-decomposed Rabenseifner halving-doubling

This appendix preserves the full derivation of the tree-flavored torus AR variant for reference. **It is not shipping** on any production torus stack — neither XLA / JAX on TPU nor NeuronX CCL on Trainium chooses it — and the §3.4 main-text discussion selects dim-decomposed ring instead. The material here exists so the comparison remains complete: it sharpens *why* ring-per-dim wins on torus by showing exactly what the Rabenseifner-per-dim alternative gives up.

**Relation to §3.1.** The compositional framework is identical: run $k$ serial dim-phases for RS, then $k$ more for AG, with $(N/d_i)$ concurrent rings per phase on disjoint copper. The only swap is the inner per-dim primitive — replace each dim's $d_i$-rank ring RS (or AG) with the $d_i$-rank recursive halving-doubling schedule from `01_collective_algorithms.md` Appendix B.2 (AR) / Appendix B.4 (standalone RS or AG) (power-of-2 $d_i$ required; for non-power-of-2 $d_i$ the schedule reduces to ring or to a hybrid form). The dim-decomposition argument itself — associativity of reduction, chunk-size telescoping across phases, concurrent ring execution on disjoint copper — is unchanged.

**Why swap the inner primitive?** Recursive halving-doubling takes $\lceil \log_2 d_i \rceil$ α hops per dim-phase instead of $(d_i - 1)$ for ring, while keeping the same $(d_i - 1)/d_i \cdot \mathrm{chunk}/\mathrm{BW}$ bandwidth term per phase (because each rank still sends $(d_i - 1)/d_i$ of its current chunk, just in $\log_2 d_i$ bursts of doubling size rather than $d_i - 1$ bursts of fixed size). The per-dim α compression is modest — $d_i = 4 \to 2$ hops vs 3 for ring (1.5×); $d_i = 8 \to 3$ vs 7 (2.3×); $d_i = 16 \to 4$ vs 15 (3.75×) — but compounds across dims.

**Worked example — 4×4 torus with Rabenseifner per dim.** Same 2D grid as §3.4, but each row/column now runs halving-doubling instead of ring. On $d_x = 4$ the X-phase RS runs $\lceil \log_2 4 \rceil = 2$ halving-doubling steps per row (step 1: exchange halves with partner 2 away; step 2: exchange quarters with partner 1 away), giving $n_\alpha = 2$ per phase versus ring's $n_\alpha = 3$. The bandwidth telescoping matches ring's exactly — after the X-phase RS each rank holds $M/4$ bytes, after Y-phase RS $M/16$ — because halving-doubling's BW coefficient $(D-1)/D$ is identical to ring's for the same chunk math. Total for 2D AR: $n_\alpha = 2 \cdot 2 \cdot 2 = 8$ versus ring's $2 \cdot 2 \cdot 3 = 12$, with identical BW term.

**Cost formula.** Summing $\lceil \log_2 d_i \rceil$ α hops per dim per half (RS or AG) and applying the same BW telescoping as §3.4:

$$t_{\mathrm{torus,AR,Rab}} \\;=\\; 2 \sum_i \lceil \log_2 d_i \rceil\\,\alpha + 2 \cdot \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}, \quad N = \prod_i d_i$$

For the AG / RS halves:

$$t_{\mathrm{torus,AG\\,or\\,RS,Rab}} = \sum_i \lceil \log_2 d_i \rceil\\,\alpha + \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}$$

**Latency compression at $N = 512$.** For the $8 \times 8 \times 8$ layout: $\sum \lceil \log_2 d_i \rceil = 9$ hops, so $n_\alpha = 18$ for AR versus dim-decomp ring's 42 (2.3× compression) and flat ring's 1022 (57× compression). The marginal step from ring-per-dim (42) to Rabenseifner-per-dim (18) saves 24 α hops; at $\alpha = 0.5\\,\mu$s that is 12 μs — meaningful at very small $M$, negligible once the BW term crosses a few tens of μs.

**Why it does not ship on production torus fabrics.** Four constraints together:

1. **Power-of-2 dim sizes required.** The halving-doubling schedule assumes $d_i$ is a power of 2 along each active dim. TPU v4 / v5p pods configured via OCS can choose $d_i \in \\{2, 4, 8, 16\\}$ per dim — compatible — but Trainium's fixed 2D-plus-Z 64-chip shape bakes in $d_i = \\{4, 4, 4\\}$. Any dim mismatch forces fallback to ring-per-dim for that dim, eroding the compression.
2. **Small per-dim $d_i$ limits the α savings.** At $d_i = 4$ the α compression is 1.5× (3 hops → 2); at $d_i = 8$ it is 2.3×. The $\log_2 d_i$ advantage only widens for large $d_i$, but production dim sizes top out around 16 because larger dims degrade A2A (§3.6 bisection-penalty scales as $d_{\max}$).
3. **Kernel complexity dominates the α savings at bulk $M$.** Each halving-doubling step has butterfly-pattern neighbor exchange with doubling chunk size; mapping this onto the torus's fixed $2k$-neighbor wiring requires per-step chunk recomputation on the chip's reduce-engine. The ring kernel uses a single static schedule. For the $M \geq$ few hundred KB bulk regime where production workloads live, the BW term dominates and the extra hops the Rabenseifner variant saves do not pay for the kernel-complexity overhead.
4. **α term is not on the critical path once BW dominates.** At $N = 512$, $\alpha = 0.5\\,\mu$s, $\mathrm{BW} = 900\\,\mathrm{GB/s}$, and $M = 16\\,\mathrm{MB}$, the ring-per-dim AR cost is 21 μs α + 35.5 μs BW = 57 μs total; Rabenseifner-per-dim trims this to 9 μs α + 35.5 μs BW = 45 μs — a 21% saving on paper that is routinely erased by the kernel-complexity overhead in point 3.

This is a fabric-economics and software-simplicity decision, not a correctness one: dim-decomposed Rabenseifner-per-dim produces the same numerical result (up to floating-point ordering) as dim-decomposed ring, and either could be implemented on the same hardware. Production stacks ship the simpler ring-per-dim kernel because its α disadvantage is small at production dim sizes and its BW coefficient is exactly the same.

**Limitations.** All of §3.4's floating-point-associativity, dim-aligned-group-layout, and wraparound-wiring caveats apply verbatim. The per-dim halving-doubling schedule adds the power-of-2 $d_i$ constraint on top.

---

## Appendix B: N, D, d — rank count, diameter, and dim sizes

Three symbols recur throughout the torus / mesh cost formulas in §3–§4: **N** (total rank count), **D** (diameter), and **$d_i$** (per-axis dim size). They are related but distinct — N and D are both derived from the set of $d_i$ values, but each plays a different role in the cost formulas. This appendix defines each, shows how they connect, and works through examples.

### B.1 Definitions

- **$d_i$ — dim size along axis $i$.** The number of ranks sitting on axis $i$ of a k-D lattice. For a k-D torus or mesh, the collection $(d_1, d_2, \ldots, d_k)$ is the **shape**. A 4×3×2 lattice has $d_1 = 4$, $d_2 = 3$, $d_3 = 2$.
  - **Symmetric**: all $d_i$ equal — write $d^k$ or "$d \times d \times d$ ...". TPU v5p's $16 \times 16 \times 16$ has $d = 16$.
  - **Asymmetric**: $d_i$ differ — e.g., $16 \times 16 \times 4$.
  - $d_{\max} = \max_i d_i$: the largest dim size, which sets the A2A bisection penalty (§3.6).

- **$N$ — total rank count.** For a k-D lattice, $N$ is the product of dim sizes:

  $$N = \prod_i d_i = d_1 \cdot d_2 \cdots d_k$$

  Pick the shape, multiply the $d_i$s, get $N$.

- **$D$ — diameter.** The hop count between the farthest rank pair (worst-case shortest path). A single scalar describing the fabric, derived from the $d_i$ values via topology-specific formulas:

  | Topology | Diameter formula |
  |---|---|
  | Torus (wraparound, §3) | $D = \sum_i \lfloor d_i / 2 \rfloor$ |
  | k-D mesh (no wraparound, §4.2) | $D = \sum_i (d_i - 1)$ |
  | Star (single switch, §2) | $D = 2$ |
  | Full mesh (§4.1) | $D = 1$ |
  | 2-tier leaf-spine fat-tree (`03_hierarchical_topologies.md` §1) | $D = 4$ |
  | 3-tier Clos (`03_hierarchical_topologies.md` §1) | $D = 6$ |

### B.2 Connecting the three

Pick the shape first; both $N$ and $D$ fall out:

```
     shape = (d₁, d₂, ..., dₖ)
                  │
                  ├──── product ────────→   N  (total ranks; grows as dᵏ in k-D)
                  │
                  └──── sum over axes ──→   D  (diameter; grows linearly in k)
                        per-axis worst-case hops:
                          torus: ⌊dᵢ / 2⌋
                          mesh:  dᵢ − 1
```

**Per-axis worst case**:

- **Torus axis** (ring of $d$ ranks): walking to the opposite rank takes $\lfloor d / 2 \rfloor$ hops in either direction (wraparound gives the shortcut).
- **Mesh axis** (open line of $d$ ranks): walking end-to-end takes $d - 1$ hops; no shortcut.

**Sum over axes**: routing factors across axes (dim-independent routing), so the worst-case pair's hop count is the sum of per-axis worst cases.

### B.3 Worked example — 3×3 torus

Concrete small case: $d = 3$ on both axes, $k = 2$.

- **Shape**: (3, 3).
- **$N$** = 3 · 3 = **9 ranks**.
- **$D$ (torus)** = $\lfloor 3/2 \rfloor + \lfloor 3/2 \rfloor$ = 1 + 1 = **2 hops**.
- **$D$ (mesh, same shape)** = (3−1) + (3−1) = **4 hops**.

Trace worst-case from rank (0, 0) in the torus:

```
3 × 3 torus (from §1.2, d = 3 per axis):

   ↕     ↕     ↕
↔ (0,2)━(1,2)━(2,2) ↔
     ┃     ┃     ┃
↔ (0,1)━(1,1)━(2,1) ↔         Hop count from (0, 0) to each other rank:
     ┃     ┃     ┃
↔ (0,0)━(1,0)━(2,0) ↔           (1, 0) = 1     (2, 0) = 1 (X-wrap shortcut)
   ↕     ↕     ↕                (0, 1) = 1     (0, 2) = 1 (Y-wrap shortcut)
                                 (1, 1) = 2     (2, 1) = 2
                                 (1, 2) = 2     (2, 2) = 2

                                 Worst case: 2 hops → D = 2 ✓
```

On a 3 × 3 mesh (wraparound removed), (0, 0) ↔ (2, 2) takes 2 + 2 = 4 hops — no X-wrap or Y-wrap shortcut.

### B.4 Same $N$, different shapes

One rank count $N$ admits many shapes, each with its own diameter:

| Shape | $(d_1, \ldots, d_k)$ | k | $N$ | $D$ (torus) | $D$ (mesh) |
|---|---|---|---|---|---|
| 1D 512-ring | (512) | 1 | 512 | $\lfloor 512/2 \rfloor$ = 256 | 511 |
| 2D 32 × 16 | (32, 16) | 2 | 512 | 16 + 8 = 24 | 31 + 15 = 46 |
| 3D 8 × 8 × 8 | (8, 8, 8) | 3 | 512 | 4 + 4 + 4 = 12 | 7·3 = 21 |
| 4D 8 × 4 × 4 × 4 | (8, 4, 4, 4) | 4 | 512 | 4 + 2·3 = 10 | 7 + 3·3 = 16 |

Two patterns:

1. **Higher $k$ → smaller $D$ at fixed $N$.** From 1D (D = 256) to 4D (D = 10), diameter compresses ~25×. This is the $O(N^{1/k})$ scaling that §3 attributes to dim-decomposition — factoring the collective across $k$ dims lets each dim-ring be short.
2. **Wraparound halves per-axis $D$.** 3D 8³ torus has D = 12; same-shape mesh has D = 21 (~1.75× larger). At larger $d$ the ratio approaches 2× — the structural reason torus outperforms mesh on α-term (§4.2 pros).

### B.5 Role in cost formulas

The three quantities map to different parts of the α-β cost model:

| Quantity | Appears in | Example formula |
|---|---|---|
| $N$ | BW term | $(N-1)/N \cdot M / \mathrm{BW}$ (flat-ring AR bandwidth coefficient) |
| $d_i$ | BW-telescoping during dim-decomp | after phase-$i$ RS, chunk size becomes $M / \prod_{j \leq i} d_j$ |
| $D$ | α-term (traversal latency) | flat-ring AR pays $2(N-1)\alpha$; dim-decomposed AR pays $2 \sum_i (d_i - 1) \alpha$ |
| $d_{\max}$ | A2A bisection penalty | torus A2A BW term = $d_{\max}/8 \cdot M/\mathrm{BW}$ |

Practical rule of thumb:

- **Whole-fabric size** → $N$.
- **Per-axis width** → $d_i$.
- **Worst-case hop count** → $D$.
- **Largest axis** → $d_{\max}$.

Conflating any two of these (e.g., writing $D$ for both the per-axis width and the diameter) makes formulas like $D = \sum_i \lfloor d_i / 2 \rfloor$ read as $D = \sum_i \lfloor D_i / 2 \rfloor$ — same letter on both sides, requiring the reader to mentally disambiguate inner-$D$ vs outer-$D$. The uppercase-$D$ / lowercase-$d$ split avoids this entirely.

---

## Appendix C: Per-chunk-per-step analysis vs bottleneck analysis

The BW-term for different primitives in §3 and §4 is computed two different ways:

- **Per-chunk-per-step analysis** — used for ring RS / AG and dim-decomposed AR (§3.4, §3.5).
- **Bottleneck analysis** — used for A2A (§3.6, §4.2) and fat-tree A2A (`03_hierarchical_topologies.md` §2.2).

This appendix explains when each applies, how they differ, and why bottleneck analysis is the more general formulation. §3.6 references the results here.

### C.1 Per-chunk-per-step analysis (lockstep, symmetric traffic)

For ring RS on a $d$-rank ring, every rank sends one chunk per step through its one outbound link. By symmetry:

- Every rank performs the same operation at every step.
- Every link carries the same bytes per step.

Total BW cost is computed **step by step**:

$$t_{\mathrm{BW, ring\\,RS}} = \sum_{\text{steps}} \frac{\text{chunk size at this step}}{\mathrm{BW}} \\;=\\; (d - 1) \cdot \frac{M/d}{\mathrm{BW}} \\;=\\; \frac{d - 1}{d} \cdot \frac{M}{\mathrm{BW}}$$

This works because the traffic is **lockstep symmetric** — no link is more loaded than any other, and summing per-step-per-rank costs captures the full BW cost. For dim-decomposed AR (§3.4), the same technique is applied per dim-phase with telescoping chunk sizes, summing to $(N-1)/N \cdot M/\mathrm{BW}$ across all phases.

### C.2 Bottleneck analysis (pipelined, possibly asymmetric traffic)

For A2A ring-relay, chunks flow independently along shortest-arc paths across the fabric. At any given moment, different links carry different chunks; there is no synchronous per-step pattern where every rank does the same thing. Per-step-per-rank summing doesn't apply directly.

Instead, identify the link (or group of links) carrying the most bytes over the whole phase and compute time to drain:

$$t_{\mathrm{BW}} \\;=\\; \frac{\text{bytes through the bottleneck}}{\text{bottleneck throughput}}$$

**Why this works.** The phase ends only when every byte has reached its destination. The most-loaded link is the last to drain — lighter-loaded links finish earlier but don't reduce the phase time. Any scheduling trick that doesn't reduce the bottleneck's load doesn't reduce this time.

For A2A on torus, the natural bottleneck is the **bisection cut**: all cross-half traffic must pass through it (intra-half links can't carry it), so the cut's capacity is the structural upper bound on cross-half throughput.

### C.3 Concrete example — 4×4 torus A2A

Take the 4×4 torus A2A worked example (§3.6). We compute the BW-term three ways: (1) per-chunk-per-step reasoning applied to a single chunk's journey, (2) bottleneck analysis on the whole cut (aggregate bytes / aggregate throughput), and (3) bottleneck analysis zoomed to one specific cut link. (2) and (3) are the same bottleneck analysis at different scopes and give the same answer; (1) gives a different (smaller) number, and that gap is the lesson.

**(1) Per-chunk-per-step reasoning on a single chunk** — what $a{\to}k$'s own bytes pay for its journey. Chunk $a \to k$ takes a 4-hop route:

$$a \to b \to c \to g \to k$$

Each hop moves a chunk of size $M/N = M/16$ at per-link $\mathrm{BW}$, so each hop's transit time is $M/(16 \cdot \mathrm{BW})$. Summed across all 4 hops:

$$t_{a{\to}k\\,\text{personal}} \\;=\\; 4 \cdot \frac{M}{16 \\, \mathrm{BW}} \\;=\\; \frac{M}{4 \\, \mathrm{BW}}$$

This is **not** the phase time — after $a{\to}k$ arrives, the cut link that carried its cross-half hop is still busy serving other chunks.

**(2) Bottleneck analysis — aggregate whole-cut view** (total cross-cut bytes drained by aggregate cut throughput).

*Total cross-cut bytes, one direction.* Every left-half rank (8 of them) sends one chunk of size $M/N = M/16$ to every right-half rank (8 of them):

$$\text{cross-cut bytes, L}\to\text{R} \\;=\\; \tfrac{N}{2} \cdot \tfrac{N}{2} \cdot \tfrac{M}{N} \\;=\\; 8 \cdot 8 \cdot \tfrac{M}{16} \\;=\\; 4M$$

*Byte-count is fixed by A2A semantics (not by routing).* A shortest-arc route from a left-half source to a right-half destination transitions halves exactly once — it uses exactly one cut edge, regardless of which shortest path is chosen. For $a{\to}k$, three equally-short routes exist; each uses exactly one cut hop:

```
  X-first direct:   a ─I─→ b ─C─→ c ─I─→ g ─I─→ k        (cut at hop 2)
  Y-first:          a ─I─→ e ─I─→ f ─C─→ g ─I─→ k        (cut at hop 3)
  X-wraparound:     a ═C═→ d ─I─→ c ─I─→ g ─I─→ k        (cut at hop 1)

  I = intra-half hop, C = cut hop.
```

So all 4M cross-cut bytes pass through the 8 cut links exactly once, total — no scheduling trick reduces this.

*Aggregate cut throughput, one direction.* 8 cut links × per-link $\mathrm{BW}$ per direction = $8\\,\mathrm{BW}$. (Per-link BW is per-direction on a full-duplex link, as calibrated in §2; the symmetric R→L flow runs on the opposite channels of the same 8 links and drains concurrently.)

*Divide:*

$$t_{\mathrm{BW}} \\;=\\; \frac{4M}{8\\,\mathrm{BW}} \\;=\\; \frac{M}{2\\,\mathrm{BW}}$$

**(3) Bottleneck analysis — single-link view** (zoom in on one cut link's busy-time). Focus on `b→c` (direct cut at $y=0$, L→R direction). Under uniform shortest-path routing, the 64 cross-cut chunks distribute across the 8 cut links → 8 chunks per cut link. The 8 chunks passing through `b→c` under X-first routing from row $y=0$ to the $x=2$ column:

| # | Chunk | Route | Total hops |
|---|---|---|---|
| 1 | $a \to c$ | $a \to b \to c$ | 2 |
| 2 | $a \to g$ | $a \to b \to c \to g$ | 3 |
| 3 | $a \to k$ | $a \to b \to c \to g \to k$ | 4 |
| 4 | $a \to o$ | $a \to b \to c \to o$ *(Y-wrap on last hop)* | 3 |
| 5 | $b \to c$ | $b \to c$ | 1 |
| 6 | $b \to g$ | $b \to c \to g$ | 2 |
| 7 | $b \to k$ | $b \to c \to g \to k$ | 3 |
| 8 | $b \to o$ | $b \to c \to o$ *(Y-wrap on last hop)* | 2 |

Each chunk uses `b→c` exactly **once** somewhere in its journey (position doesn't matter for the BW arithmetic — only total occupancy does). Each use costs $M/(16 \cdot \mathrm{BW})$. Total occupancy:

$$t_{b{\to}c\\,\text{occupancy}} \\;=\\; 8 \cdot \frac{M}{16 \\, \mathrm{BW}} \\;=\\; \frac{M}{2 \\, \mathrm{BW}}$$

(2) and (3) agree, as they must for a symmetric torus where every cut link carries the same load — cut-wide average = any-one-link occupancy.

**Why the phase time is the bottleneck occupancy, not the per-chunk journey.** $a{\to}k$ personally uses `b→c` for only $M/(16 \cdot \mathrm{BW})$ — one of the 8 transits that link carries. After $a{\to}k$ finishes its own hop through `b→c`, the link still has 7 more chunks to serve. The phase doesn't end until **all chunks have arrived**, including the 7 others. So phase time = link's total occupancy = $M/(2 \cdot \mathrm{BW})$, not $a{\to}k$'s personal $M/(4 \cdot \mathrm{BW})$.

**Generalizing to arbitrary torus shape.** The whole-cut view (2) extends to arbitrary $d_i$ by tracking how each of its quantities scales with $N$ and $d_{\max}$:

| Quantity | 4×4 value | General formula |
|---|---|---|
| Severed-link count | 8 | $2N / d_{\max}$ (2 links per dim-line × $N/d_{\max}$ dim-lines across the cut) |
| Cross-cut chunks (one direction) | 64 | $(N/2) \cdot (N/2) = N^2/4$ |
| Cross-cut bytes (one direction) | $4M$ | $(N^2/4) \cdot (M/N) = NM/4$ |
| Cut throughput (one direction) | $8\\,\mathrm{BW}$ | $(2N / d_{\max}) \cdot \mathrm{BW}$ |
| Chunks per cut link | 8 | $(N^2/4) / (2N/d_{\max}) = N \cdot d_{\max}/8$ |

Two quantities of interest fall out of this table. **The BW term**, dividing one-direction traffic by one-direction throughput (full-duplex → the symmetric opposite direction finishes concurrently, so this is the total BW phase time):

$$t_{\mathrm{BW}} \\;=\\; \frac{NM/4}{(2N/d_{\max}) \cdot \mathrm{BW}} \\;=\\; \frac{d_{\max}}{8} \cdot \frac{M}{\mathrm{BW}}$$

4×4 check: $d_{\max}/8 \cdot M/\mathrm{BW} = 4/8 \cdot M/\mathrm{BW} = M/(2\\,\mathrm{BW})$ ✓. This $d_{\max}/8$ factor is what §3.6's general cost formula uses.

**The bottleneck-vs-personal ratio**, quantifying how much more loaded the bottleneck link is than one chunk's private path (chunks sharing the bottleneck, divided by hops in one chunk's journey):

$$\frac{t_{\mathrm{bottleneck}}}{t_{\mathrm{longest chunk personal}}} \\;=\\; \frac{\text{chunks per cut link}}{\text{longest chunk hop count}} \\;=\\; \frac{N \cdot d_{\max} / 8}{\mathrm{diam}}$$

4×4 check: $(16 \cdot 4 / 8) / 4 = 8/4 = 2$ ✓ — the bottleneck serves 8 chunks while a single chunk's journey has only 4 hops, so the link is "shared more broadly" than any one chunk's path is "long."

Evaluated across a few shapes:

| Shape | $N$ | $d_{\max}$ | $\mathrm{diam}$ | Chunks per cut link | Longest journey (hops) | Ratio |
|---|---|---|---|---|---|---|
| 4×4 (our case) | 16 | 4 | 4 | 8 | 4 | 2 |
| 8×8 | 64 | 8 | 8 | 64 | 8 | 8 |
| 8×8×8 | 512 | 8 | 12 | 512 | 12 | ~43 |
| 16×16×16 (TPU v5p) | 4096 | 16 | 24 | 8192 | 24 | ~341 |

The ratio grows roughly as $N / \mathrm{diam}$: large tori accumulate many chunks on each cut link while the longest journey grows only slowly (linearly in dim-size), so the bottleneck term increasingly dominates any individual chunk's personal BW cost. The 4×4 case is pedagogically convenient because the ratio is a clean small number; at production scale the bottleneck is hundreds of times larger than a single chunk's BW cost.

**Re-use interpretation (bridge analogy).** What the bottleneck formula is actually computing: **how many times is the bottleneck link re-used over the phase, and how long does each re-use take?**

$$t_{\mathrm{BW, bottleneck}} \\;=\\; \underbrace{(\text{\\# re-uses of bottleneck link})}_{\text{= \\# chunks through it}} \\;\times\\; \underbrace{(\text{per-use time})}_{\text{= chunk size / BW}}$$

At b→c: 8 re-uses × M/(16·BW) per re-use = M/(2·BW). The ideal formula assumes **perfect pipelining** — each re-use happens back-to-back, with the link at 100% utilization during its busy period. (Real deployments fall short due to switch queueing, flow-control stalls, and scheduler bubbles; that gap is priced by $\eta_\beta$ in `05_contention_and_congestion.md` §4.)

Picture 8 people each needing to cross a 1-minute bridge. Each person's own crossing takes 1 minute — short. But the bridge is re-used 8 times, back-to-back, so it's occupied for 8 minutes total. The party finishes when the last person exits, at minute 8, even though no single person's crossing took longer than 1. Phase time tracks the **bridge's total busy-time** (= re-use count × per-use time), not any one person's crossing time.

The 4×4 A2A case maps exactly:

- "People" = chunks (8 of them passing through b→c)
- "1-minute crossing" = M/(16·BW) per-chunk transit
- "Party finishes" = last chunk exits b→c → M/(2·BW)

### C.4 When the two analyses coincide

On regular torus with uniform shortest-path routing, graph symmetry forces every link to carry identical load ($M/2$ bytes per direction in the 4×4 A2A case). The bottleneck load equals the average load, so bottleneck-based and per-link-uniform calculations give the same answer. Per-chunk-per-step analysis — if applied carefully to account for pipelining — reaches the same number.

For ring RS, traffic is balanced by construction. Per-step-per-rank is the easiest formulation and gives the right answer directly; bottleneck analysis would give the same result but is unnecessary overhead.

**So when traffic is symmetric**, pick whichever formulation is easier to compute. For ring, it's per-step-per-rank. For A2A on regular torus, it's the bisection bottleneck.

### C.5 When bottleneck analysis is necessary

Bottleneck analysis is strictly more general; it gives the tight bound even when per-link symmetry breaks:

- **Oversubscribed fat-tree** (`03_hierarchical_topologies.md` §1): upper-tier links loaded $s \times$ more than lower-tier. Per-link-uniform doesn't apply.
- **Off-prefix torus group layouts** (§7): some dim-lines loaded more than others when the collective's ranks don't tile the torus cleanly.
- **Hotspot traffic** (e.g., skewed MoE expert routing): some destinations receive disproportionately more traffic, concentrating load on specific links.
- **Non-uniform routing policies**: deterministic routing may not evenly distribute chunks across alternate paths.

In each of these cases, per-link loads differ from the average and only bottleneck analysis gives the tight time bound. "Average-based" reasoning ($t = \text{total bytes} / \text{total fabric BW}$) underestimates because it assumes perfect load balancing that the fabric doesn't deliver.

### C.6 Relationship to $\eta_\beta$ in `05_contention_and_congestion.md`

**The bottleneck formula derived above is the ideal lower bound — it does NOT include any $\eta_\beta$ discount.** Every $t_{\mathrm{BW}}$ computed in this note (§3.6's $M/(2\\,\mathrm{BW})$ for 4×4 torus A2A, `03_hierarchical_topologies.md` §2.2's fat-tree A2A term, etc.) assumes:

- The bottleneck link runs at 100% utilization during its busy period.
- Every re-use of the bottleneck happens back-to-back with no gaps between chunks.
- No switch queueing delay, no head-of-line blocking, no flow-control stalls.
- No cross-traffic interference, no imperfect pipelining bubbles.

Under those assumptions, the formula gives the **fastest possible phase time** on the bottleneck — the ideal target that a perfectly-scheduled deployment would hit. Real deployments sit above this lower bound because those assumptions fail in practice: switches do queue, flow-control does stall, chunks don't arrive at relay points perfectly aligned, and other traffic shares the same ports.

The coefficient $\eta_\beta \in (0, 1]$ introduced in `05_contention_and_congestion.md §4` is a **separate multiplicative discount applied on top of this note's ideal formulas**, quantifying the real-vs-ideal gap:

$$t_{\mathrm{BW, realized}} \\;=\\; \frac{\text{bottleneck bytes}}{\eta_\beta \cdot \mathrm{BW}} \\;=\\; \frac{t_{\mathrm{BW, ideal}}}{\eta_\beta}$$

**Bottom line.** Throughout `02_topology_mapping.md`, wherever you see a cost formula with plain $\mathrm{BW}$ (no $\eta_\beta$), it is the **$\eta_\beta = 1$ idealization**. The $\eta_\beta$ factor is not hidden inside any of the derivations here; it is strictly a `05_contention_and_congestion.md`-layer concept, applied externally when scoring realistic performance. When reading a `02_topology_mapping.md` BW-term, assume perfect pipelining and zero contention.

---

## Further reading

- **`01_collective_algorithms.md`** — topology-free derivations of the ring, tree, RS, AG, and A2A primitives. Prerequisite for the per-topology cost formulas above.
- **`04_in_network_collectives.md`** — how SHARP / NVLS collapses the $O(N)$ endpoint-hop cost on star topologies to $O(1)$ switch-hop cost.
- **`05_contention_and_congestion.md`** — extending the ideal formulas above with $\eta_\alpha$, $\eta_\beta$ coefficients to score concurrent-group and off-prefix effects.
- **`03_hierarchical_topologies.md`** — composing the single-tier topologies above into multi-tier hierarchies (NVL72 star inside a Clos, chiplet mesh inside a star inside a Clos), with the associated hierarchical-collective cost rules and device-domain allocation problem.
- **`references.md`** — primary-source citations for the cost formulas in this note (Patarasuk-Yuan bandwidth-optimal AR, Chan et al. dim-decomposed AR).
