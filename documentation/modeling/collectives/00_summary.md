# Notation and Key Equations Cheatsheet

**Author:** Yue Lu  
**Date:** April 2026  

A one-page reference for the symbols, primitives, and canonical α-β cost formulas used throughout `01_collective_algorithms.md` – `05_contention_and_congestion.md`. Each row points to the section that derives it; equations here are stated, not justified.

# Table of Contents

1. [Symbols](#1-symbols)
2. [The seven primitives](#2-the-seven-primitives)
3. [Cost-model skeleton](#3-cost-model-skeleton)
4. [Per-algorithm costs (topology-free)](#4-per-algorithm-costs-topology-free)
5. [Per-topology specializations](#5-per-topology-specializations)
6. [Hierarchical composition](#6-hierarchical-composition)
7. [In-network collectives (INC)](#7-in-network-collectives-inc)
8. [Realistic cost: η coefficients](#8-realistic-cost-η-coefficients)

---

## 1. Symbols

| Symbol | Meaning | Defined in |
|---|---|---|
| $\alpha$ | per-hop latency (setup + switch traversal + propagation) | `01_collective_algorithms.md` §1 |
| $M$ | message size in bytes | `01_collective_algorithms.md` §1 |
| $\mathrm{BW}$ | per-link bandwidth (one direction) | `01_collective_algorithms.md` §1 |
| $N$ | rank count in the collective group | `01_collective_algorithms.md` §2 |
| $V_i$ | payload held by rank $i$ (size $M$) | `01_collective_algorithms.md` §2 |
| $n_\alpha$ | latency-term count (sequential synchronization hops) | `01_collective_algorithms.md` §1 |
| $n_\beta$ | bandwidth-term count (per-rank bytes moved, in units of $M$) | `01_collective_algorithms.md` §1 |
| $\mathrm{BW_{eff}}$ | effective per-rank bandwidth seen by the collective ($\mathrm{BW}/n_\beta$, accounting for dual-touch on AR) | `04_in_network_collectives.md` §1.4 |
| $D$ | network diameter (worst-case hop count between rank pair) | `02_topology_mapping.md` §1, App. B |
| $d_i$ | dim-$i$ rank count on a torus / mesh ($\prod_i d_i = N$) | `02_topology_mapping.md` §1.2, App. B |
| $d_{\max}$ | $\max_i d_i$ — controls torus A2A bisection penalty | `02_topology_mapping.md` §3.6 |
| $k$ | torus / mesh dimensionality (number of dims) | `02_topology_mapping.md` §3 |
| $\alpha_{\mathrm{switch}}$ | switch cut-through latency (~100–400 ns; $\ll \alpha$) | `04_in_network_collectives.md` §1.1 |
| $s$ | fat-tree / Clos oversubscription ratio at a tier boundary ($s \geq 1$) | `03_hierarchical_topologies.md` §1 |
| $L$ | outer-tier rank count in a 2-tier hierarchy ($N = L \cdot N_{\mathrm{inner}}$) | `03_hierarchical_topologies.md` §2.1 |
| $\eta_\alpha \geq 1$ | realistic α-side contention coefficient (inflation factor) | `05_contention_and_congestion.md` §4 |
| $\eta_\beta \in (0, 1]$ | realistic BW-side contention coefficient (utilization factor) | `05_contention_and_congestion.md` §4 |
| algbw | algorithm bandwidth: $M / t$ (per-call effective rate) | `01_collective_algorithms.md` App. D |
| busbw | bus bandwidth: $C_{\mathrm{primitive}}(N) \cdot M / t$ (link-level rate) | `01_collective_algorithms.md` App. D |

---

## 2. The seven primitives

| Primitive | Group state in → out |
|---|---|
| **Broadcast (BC)** | rank 0 holds $V$ → all hold $V$ |
| **Reduce** | rank $i$ holds $V_i$ → rank 0 holds $\sum_i V_i$ |
| **All-reduce (AR)** | rank $i$ holds $V_i$ → all hold $\sum_i V_i$ |
| **Reduce-scatter (RS)** | rank $i$ holds $V_i$ (size $M$) → rank $i$ holds chunk $\sum_j V_j[\mathrm{chunk}_i]$ (size $M/N$) |
| **All-gather (AG)** | rank $i$ holds chunk $i$ (size $M/N$) → all hold concatenation (size $M$) |
| **All-to-all (A2A)** | rank $i$ holds $N$ chunks indexed by destination → rank $i$ holds $N$ chunks indexed by sender |
| **Point-to-point (P2P)** | rank $a$ holds $V$ → rank $b$ holds $V$ |

**Identity:** $\mathrm{AR} \equiv \mathrm{RS} + \mathrm{AG}$ (both halves over the same group). Used by ring AR, Rabenseifner halving-doubling, hierarchical RS → sub-AR → AG.

---

## 3. Cost-model skeleton

Every collective cost in this series factors as

$$t \\;=\\; n_\alpha \cdot \alpha \\;+\\; n_\beta \cdot \frac{M}{\mathrm{BW}}$$

- $n_\alpha$ counts sequential hops on the algorithm's critical path.
- $n_\beta$ counts per-rank byte movement in units of $M$ (not the total fabric traffic).

A primitive's BW-side lower bound is set by the cut-through structure:

| Primitive | $n_\beta$ lower bound | Why |
|---|---|---|
| BC, Reduce, AG, RS | $(N-1)/N \to 1$ | one rank's share crosses the cut once |
| AR | $2(N-1)/N \to 2$ | every byte is touched twice (RS in, AG out) — *unless* INC fuses the two halves |
| A2A | $(N-1)/N$ per rank | $N(N-1)$ pairwise distinct payloads bisection-bound |

---

## 4. Per-algorithm costs (topology-free)

Costs assume an abstract fully-connected fabric (every rank can reach every other in 1 hop). Topology corrections come from §5.

| Algorithm | Primitive | $n_\alpha$ | $n_\beta$ | Source |
|---|---|---|---|---|
| Ring (chain) | BC | $(N-1)$ raw / $\to 1$ pipelined | $1$ | `01_collective_algorithms.md` §3.1 |
| Binomial tree | BC | $\lceil \log_2 N \rceil$ raw / pipelined to $\to 1$ | $1$ | `01_collective_algorithms.md` §3.2 |
| Ring | Reduce | $(N-1)$ raw / pipelined to $\to 1$ | $1$ | `01_collective_algorithms.md` §4.1 |
| Binomial tree | Reduce | $\lceil \log_2 N \rceil$ raw / pipelined to $\to 1$ | $1$ | `01_collective_algorithms.md` §4.2 |
| Ring | AR | $2(N-1)$ | $2(N-1)/N \to 2$ | `01_collective_algorithms.md` §5.1 |
| Double binary tree (DBT) | AR | $2 \lceil \log_2 N \rceil$ | $2$ (BW-eff = BW/2) | `01_collective_algorithms.md` §5.2 |
| Recursive-halving + doubling (Rabenseifner) | AR | $2 \lceil \log_2 N \rceil$ | $2(N-1)/N \to 2$ | `01_collective_algorithms.md` App. B.2 |
| Simple recursive-doubling | AR | $\lceil \log_2 N \rceil$ | $\lceil \log_2 N \rceil$ | `01_collective_algorithms.md` App. B.1 |
| Ring | AG / RS | $(N-1)$ | $(N-1)/N \to 1$ | `01_collective_algorithms.md` §6 |
| PAT (parallel aggregated trees) | AG / RS | $\lceil \log_2 N \rceil$ | $(N-1)/N \to 1$ | `01_collective_algorithms.md` App. A |
| Recursive-doubling AG / -halving RS | AG / RS | $\lceil \log_2 N \rceil$ | $(N-1)/N \to 1$ | `01_collective_algorithms.md` App. B.4 |
| Pairwise direct-send | A2A | $(N-1)$ | $(N-1)/N$ | `01_collective_algorithms.md` §7.2 |
| Ring-relay | A2A | $(N-1)$ | varies (bisection-bound on torus) | `01_collective_algorithms.md` §7.1 |
| Bruck (log-round) | A2A | $\lceil \log_2 N \rceil$ | $\lceil \log_2 N \rceil / 2$ | `01_collective_algorithms.md` App. B.5 |

**Headline AR formulas:**

$$t_{\mathrm{ring\\,AR}} \\;=\\; 2(N-1)\\,\alpha \\;+\\; \frac{2(N-1)}{N} \cdot \frac{M}{\mathrm{BW}}$$

$$t_{\mathrm{tree\\,AR}} \\;=\\; 2\lceil \log_2 N \rceil \\,\alpha \\;+\\; 2\\,\frac{M}{\mathrm{BW}}$$

(DBT hits the BW floor as a hardware ceiling under asymptotic pipelining; the practice gap is documented in `01_collective_algorithms.md` §5.3.)

---

## 5. Per-topology specializations

Single-tier scale-up fabrics. AR formula entries; AG / RS halve both terms; BC / Reduce hit $M/\mathrm{BW}$ under pipelining and inherit the topology's α term.

| Topology | $n_\alpha$ (AR) | $n_\beta$ (AR) | A2A BW term | Source |
|---|---|---|---|---|
| Star (single switch) | $2(N-1)$ ring / $2 \lceil \log_2 N \rceil$ DBT | $2(N-1)/N \to 2$ | $(N-1)/N \cdot M/\mathrm{BW}$ pairwise | `02_topology_mapping.md` §2 |
| Torus ($k$-D, $\prod d_i = N$, dim-decomposed ring) | $2 \sum_i (d_i - 1)$ | $2(N-1)/N \to 2$ | $(d_{\max}/8) \cdot M/\mathrm{BW}$ at cubic shapes | `02_topology_mapping.md` §3.4, §3.6 |
| Torus dim-decomposed Rabenseifner (not shipped) | $2 \sum_i \lceil \log_2 d_i \rceil$ | $2(N-1)/N \to 2$ | — | `02_topology_mapping.md` App. A |
| Full mesh | identical to star at $\alpha \to \alpha_{\mathrm{link}}$ | $2(N-1)/N \to 2$ | $(N-1)/N \cdot M/\mathrm{BW}$ | `02_topology_mapping.md` §4.1 |
| $k$-D mesh (no wraparound) | $2 \sum_i (d_i - 1)$ | $2(N-1)/N \to 2$ | $(d_{\max}/4) \cdot M/\mathrm{BW}$ — 2× worse than torus | `02_topology_mapping.md` §4.2 |

Bisection: torus $\mathrm{BW}_{\mathrm{bisect}} = (2N/d_{\max}) \cdot \mathrm{BW}$; fat-tree $s \cdot N \cdot \mathrm{BW}$ (oversubscription).

---

## 6. Hierarchical composition

Multi-tier Clos / fat-tree. Two-tier AR with $L$ outer ranks each containing $N/L$ inner ranks decomposes as **inner RS → outer sub-AR → inner AG**:

$$t_{\mathrm{AR}}^{\mathrm{hier}} \\;=\\; t_{\mathrm{RS,inner}}(M) \\;+\\; t_{\mathrm{AR,outer}}\\!\left(\tfrac{M\\,L}{N}\right) \\;+\\; t_{\mathrm{AG,inner}}(M)$$

The middle phase ships the *telescoped* payload $ML/N$ — which is why hierarchical AR scales: the costliest tier (typically outer) carries less data.

**Oversubscription at any tier boundary** multiplies that tier's BW term by $s$ (equivalent to $\eta_\beta = 1/s$ on cross-tier traffic):

$$t_{\mathrm{BW,outer}} \\;=\\; s \cdot \frac{M\\,L/N}{\mathrm{BW}_{\mathrm{outer}}}$$

A2A is the outlier: it cannot telescope, so per-tier ranks see $(N-1)/N \cdot M/\mathrm{BW}$ at the outer tier's bisection rate. See `03_hierarchical_topologies.md` §2.2.

---

## 7. In-network collectives (INC)

INC moves the reduction or replication into the switch ASIC (NVLS / Quantum SHARP / Spectrum-X SHARP / Tomahawk Ultra). On a single-switch star, $n_\alpha$ collapses from $O(N)$ or $O(\log N)$ down to $\sim 2$, and AR uniquely also gets a BW-eff doubling because the switch fuses the reduce and broadcast halves.

| Primitive | Required HW | $n_\alpha$ (INC) | $\mathrm{BW_{eff}}$ (INC) | Speedup vs software |
|---|---|---|---|---|
| AR | switch ALU + multicast xbar | $\sim 2\\,\alpha_{\mathrm{switch}}$ | $\mathrm{BW}$ (vs SW $\mathrm{BW}/2$) | α: $\sim (N-1)$ or $\log_2 N$ ×; BW: **2×** |
| Reduce | switch ALU | $\sim \alpha_{\mathrm{switch}}$ | $\mathrm{BW}$ | α: $\sim \log_2 N$ ×; BW: 1× |
| RS | switch ALU | $\sim 2\\,\alpha_{\mathrm{switch}}$ | $\mathrm{BW}$ | α: $\sim (N-1)/2$ ×; BW: 1× |
| AG | multicast xbar | $\sim 2\\,\alpha_{\mathrm{switch}}$ | $\mathrm{BW}$ | α: $\sim (N-1)/2$ ×; BW: 1× |
| BC | multicast xbar | $\sim 2\\,\alpha_{\mathrm{switch}}$ | $\mathrm{BW}$ | α: $\sim \log_2 N$ ×; BW: 1× |
| A2A | crossbar scatter-gather (HW A2A; emerging) | $\sim \alpha_{\mathrm{switch}}$ | $(N-1)/N \cdot \mathrm{BW}$ | α: $\sim N$ ×; BW: 1× |

**Scope.** SHARP-class INC is single-switch-domain (NVLS) or SHARP-enabled aggregation tree (Quantum SHARP, Spectrum-X SHARP). HW A2A ships today on Tomahawk Ultra and is on Rubin's roadmap. Cross-domain traffic falls back to software DBT / ring.

---

## 8. Realistic cost: η coefficients

Ideal-formula upper bounds inflate to realized cost via two scalar coefficients per (fabric, primitive):

$$t_{\mathrm{realized}} \\;=\\; \eta_\alpha \cdot n_\alpha \cdot \alpha \\;+\\; n_\beta \cdot \frac{M}{\eta_\beta \cdot \mathrm{BW}}$$

with $\eta_\alpha \geq 1$ and $\eta_\beta \in (0, 1]$.

| Fabric / regime | $\eta_\alpha$ | $\eta_\beta$ | Calibration source |
|---|---|---|---|
| Crossbar (NVLink + NVSwitch, no SHARP) | 1.00 | 0.80 | NCCL-tests H100 AR busbw 360 / 450 GB/s |
| NVLS (NVLink SHARP, in-network reduction) | 1.00 | 0.52 | NVLS busbw 470 / DBT 360 GB/s, 1.3× lift / framing |
| Torus (off-prefix + concurrent groups) | 1.20 | 0.60 | TPU v4 twisted-vs-untwisted 1.63× upper bound |
| Fat-tree / Clos upper tier at oversubscription $s$ | 1.00 | $\min(\eta_\beta^{\mathrm{hw}}, 1/s)$ | structural |

**Per-tier η in a hierarchy.** Each phase of the §6 hierarchical decomposition uses its own tier's $(\eta_\alpha^{\mathrm{tier}}, \eta_\beta^{\mathrm{tier}})$.

See `05_contention_and_congestion.md` §4 for the full calibration profile and §5 for the realistic re-run of the $N = 512$ ladder.

---

## Further reading

- `01_collective_algorithms.md` — α-β model and per-primitive algorithm derivations; algbw / busbw appendix.
- `02_topology_mapping.md` — single-tier specializations (star, torus, mesh) and the $N = 512$ ideal comparison.
- `03_hierarchical_topologies.md` — multi-tier Clos / fat-tree, composition rules, NVL72 SuperPOD case study, INC at inner / outer tier.
- `04_in_network_collectives.md` — SHARP-class INC mechanics, AR-only BW-eff doubling, per-primitive ceilings, scale-up vs scale-out.
- `05_contention_and_congestion.md` — η calibration, per-tier η on hierarchies, realistic re-run of the $N = 512$ ladder.
- `references.md` — primary sources for each formula and empirical value.
