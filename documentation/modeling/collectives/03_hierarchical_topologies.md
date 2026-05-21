# Hierarchical Topologies: Composition, Partitioning, and Optimization

**Author:** Yue Lu  
**Date:** April 2026  

Single-tier fabrics hit hard scaling ceilings — NVSwitch radix, torus dimension wiring cost, chiplet pin count — and once a deployment needs more ranks than any one tier can support, hierarchical composition is inevitable. Real clusters layer tiers accordingly: a GB200 NVL72 pod is a **star** of 72 GPUs; pods connect through an **InfiniBand Clos**; chiplets inside each GPU package form a **mesh** on the base die. This note covers the canonical multi-tier fabric (§1 Clos architecture), what changes when topologies compose across tiers (§2 composition rules), and how in-network collectives and contention coefficients interact with hierarchical schedules (§3).

# Table of Contents

1. [Multi-tier Clos architecture](#1-multi-tier-clos-architecture)
   - 1.1 [Case study: NVIDIA NVL72 SuperPOD](#11-case-study-nvidia-nvl72-superpod)
   - 1.2 [Cost-model symbols anchored on this SuperPOD](#12-cost-model-symbols-anchored-on-this-superpod)
2. [Composition rules for hierarchical collectives](#2-composition-rules-for-hierarchical-collectives)
   - 2.1 [The RS → sub-AR → AG pattern](#21-the-rs--sub-ar--ag-pattern)
   - 2.2 [A2A on hierarchies — the outlier](#22-a2a-on-hierarchies--the-outlier)
   - 2.3 [When hierarchical helps, when it hurts](#23-when-hierarchical-helps-when-it-hurts)
3. [INC and contention in hierarchies](#3-inc-and-contention-in-hierarchies)
   - 3.1 [SHARP composes at any tier of the hierarchy](#31-sharp-composes-at-any-tier-of-the-hierarchy)
   - 3.2 [Per-tier η in a hierarchical schedule](#32-per-tier-η-in-a-hierarchical-schedule)
   - 3.3 [Pareto implications](#33-pareto-implications)
4. [Appendix A: Worked rail-optimized NVL72 SuperPOD topology (L = 32)](#appendix-a-worked-rail-optimized-nvl72-superpod-topology-l--32)
5. [Appendix B: The k-ary fat-tree](#appendix-b-the-k-ary-fat-tree)
6. [Further reading](#further-reading)

---

## 1. Multi-tier Clos architecture

Each single-layer topology from `02_topology_mapping.md` has a hard scaling ceiling:

- **Star** caps at switch radix (NVSwitch Gen4: 72 ports).
- **Torus** caps at dimensional-wiring cost (TPU v5p: $16^3 = 4096$).
- **Full mesh** caps at $O(N^2)$ wire count (~8–16 endpoints).

Beyond each ceiling, the only scaling path is to **compose topologies across tiers** — one topology inner, a different one outer. Modern production stacks are always multi-tier: NVL72 (star) inside InfiniBand (Clos); chiplets (mesh) inside a GPU inside NVLink (star) inside InfiniBand (Clos); TPU chips (torus) inside an ICI slice (torus) inside an inter-slice link. The canonical multi-tier switched construction is the **Clos**: endpoints attach to **leaf switches**, leaves uplink to **spine switches**, and optionally a **super-spine** tier closes a 3-tier Clos. Its cost behavior is controlled by per-tier α-β pairs and an oversubscription ratio that caps upper-tier bandwidth.

**Per-tier characterization.** Each tier is described by four parameters: domain size $N_i$, per-hop latency $\alpha_i$, per-link BW $\mathrm{BW}_i$, and (for Clos tiers) oversubscription $s_i \geq 1$. The cost of any collective over the full $N = \prod_i N_i$ ranks is a composition of per-tier primitive costs — §2 derives the composition rules; this section anchors them on the Clos topology itself.

**Leaf-spine structure.** $N$ endpoints split across $L$ leaf switches ($N/L$ per leaf). Each leaf switch has two port populations:

- **Downlink ports** face endpoints below — $p_{\mathrm{down}} = N/L$ ports per leaf, each at port-BW.
- **Uplink ports** face the spine tier above — $p_{\mathrm{up}}$ ports per leaf, each at the same port-BW (normally all ports on a switch are identical; "downlink" vs "uplink" is a role assignment, not hardware).

A spine tier of $S$ switches carries all inter-leaf traffic; multiple spines provide multi-path routing and aggregate cross-leaf bandwidth. A small concrete instance:

```
Two-tier Clos (N = 16 endpoints; L = 4 leaves × S = 2 spines; s = 2):

                         ┌──────┐           ┌──────┐
   spine tier:           │  S0  │           │  S1  │              each spine: 4 downlinks
                         └──────┘           └──────┘                (one per leaf)

                          full bipartite — 8 leaf-spine links total
                            (every leaf connects to every spine)

                      ┌────┐    ┌────┐    ┌────┐    ┌────┐
   leaf tier:         │ L0 │    │ L1 │    │ L2 │    │ L3 │         each leaf:
                      └────┘    └────┘    └────┘    └────┘           2 uplinks (→ S0, S1)
                       ▼▼▼▼      ▼▼▼▼      ▼▼▼▼      ▼▼▼▼            4 downlinks
                       R0-3      R4-7      R8-11     R12-15

   Wiring:
     every L_i has 2 uplinks — one to S0, one to S1.
     every S_j has 4 downlinks — one to each of L0 … L3.

   Port-budget: 4 down-port BW per leaf (to endpoints) vs. 2 up-port BW per leaf (to spines)
                → oversubscription s = 4 / 2 = 2 (defined below).
```

**Cost anatomy.** Intra-leaf traffic traverses 2 link hops (endpoint → leaf → endpoint), at α ≈ a single-switch star (~0.5 μs on InfiniBand NDR class). Cross-leaf traffic traverses 4 hops (endpoint → leaf → spine → leaf → endpoint), at roughly 2× the α. Bandwidth is capped by the minimum of endpoint BW, leaf-uplink aggregate BW, and spine bisection.

**Oversubscription.** The oversubscription ratio at the leaf-to-spine boundary (same notion at any switch tier) is

$$s \\;=\\; \frac{p_{\mathrm{down}}\cdot\mathrm{BW}_{\mathrm{down}}}{p_{\mathrm{up}}\cdot\mathrm{BW}_{\mathrm{up}}} \\;\geq\\; 1$$

The special case $s = 1$ ($p_{\mathrm{up}}\cdot\mathrm{BW}_{\mathrm{up}} = p_{\mathrm{down}}\cdot\mathrm{BW}_{\mathrm{down}}$) is **full-bisection / non-blocking** — every endpoint can drive its full BW to any other endpoint through the spine without contention. NVIDIA Quantum-class InfiniBand pods are engineered this way by default; Ethernet leaf-spine pods are often full-bisection at the pod level. When full-bisection is too expensive, upper tiers are oversubscribed at $s = 2$, $s = 4$, etc. At $s = 4$ a leaf can receive $4\times$ more traffic from its endpoints than it can push to the spine: under uniform cross-leaf traffic each flow gets only $\mathrm{BW}/s$ northbound. The uplinks are the bottleneck, not the endpoint ports or the spines. The contention-coefficient model in `05_contention_and_congestion.md` captures this directly as $\eta_\beta \approx 1/s$ on upper-tier-crossing traffic — a 4:1 oversubscribed super-spine inflates cross-pod AR's BW term by 4×. **The α term is not inflated by $s$** — oversubscription is a BW knob, not a latency knob. §3.2 below reuses this $\eta_\beta$ hook when scoring hierarchical AR with realistic contention.

**Scaling to three tiers.** When a single spine tier's radix runs out, pods of leaf-spine hierarchies hang under a **super-spine** tier that routes cross-pod traffic. Diameter grows to 6 link hops (endpoint → leaf → spine → super-spine → spine → leaf → endpoint), the per-tier α-β story extends one layer, and each new boundary carries its own $s$:

```
3-tier Clos (schematic):

   [Super-spine]  cross-pod traffic only
        ↑
   [Spine (per pod)]  cross-leaf within pod, plus up to super-spine
        ↑
   [Leaf (per pod)]   endpoints attached
        ↑
   [Endpoints]
```

Production examples: large InfiniBand SuperPOD deployments (thousands of GPUs), Meta's research clusters, and most hyperscaler Ethernet scale-out fabrics. Further layering (4-tier or beyond) is rarely built — once a pod reaches super-spine scale, it's usually cheaper to add pods laterally than another vertical tier.

**Naming: Clos vs fat-tree.** Historically distinct: **Clos** (Charles Clos, 1953) is the general multi-tier switched family, allowing any $s \geq 1$; **Leiserson fat-tree** [LEIS85] is a tree with link bandwidth growing toward the root. The modern datacenter realization — the **k-ary fat-tree** of [AL-FARES08] — is built from commodity $k$-port switches with $k/2$ up / $k/2$ down at every switch, yielding a Clos with $s = 1$ at every tier; Appendix B has the port-by-port build. Production AI fabrics (NVIDIA DGX SuperPOD [DGX-SUPERPOD], Meta's Research SuperCluster [META-RSC], hyperscaler Ethernet training pods) follow this structural pattern — tiered Clos with full bipartite at every tier boundary — but don't rigidly hit Al-Fares's exact port accounting: vendors mix switch radices across tiers, and the common design runs $s = 1$ at the intra-pod leaf-spine tier (for tight training locality within a Scalable Unit) while oversubscribing the super-spine between pods where cost dominates over bandwidth. In the rest of this note, **Clos** means the general multi-tier fabric at any $s$, and **fat-tree** is shorthand for the $s = 1$ k-ary variant when the non-blocking property matters — the usage aligns with production InfiniBand / Ethernet vendor literature.

### 1.1 Case study: NVIDIA NVL72 SuperPOD

The NVL72 SuperPOD grounds the abstract Clos above in real hardware and surfaces a common source of reader confusion: **every GPU is attached to *two* physically separate fabrics, so "leaf" and "innermost tier" are both valid terms in this setup but refer to different switches.**

**Two independent fabrics per GPU.** Every GPU in an NVL72 pod has two distinct egress paths:

- **NVLink → NVSwitch (scale-up, intra-pod).** 72 GPUs per NVL72 pod on NVLink Gen5 at 1800 GB/s per GPU. Traffic stays within the pod — NVLink ports do not reach a NIC or ToR.
- **PCIe → ConnectX NIC → IB ToR (scale-out, inter-pod).** One ConnectX-7 (NDR) or ConnectX-8 (XDR) NIC per GPU via PCIe Gen5 — 72 NICs per NVL72. NDR: 50 GB/s per NIC; XDR / Quantum-X800: 100 GB/s. NICs uplink to IB Quantum-2 / X800 ToR switches, which uplink to IB spines.

NCCL routes intra-pod peers over NVLink and cross-pod peers over IB; the two fabrics never share a data path.

```
Two-fabric structure (one NVL72 pod, 72 GPUs):

      ┌──────────────── compute pod ────────────────┐
      │                                             │
      │   ┌───────┐   NVLink   ┌─────────────┐      │
      │   │ GPU_0 │────────────│             │      │    ← scale-up (NVSwitch)
      │   └───┬───┘            │  NVSwitch   │      │      NVLink Gen5: 1800 GB/s
      │       │ PCIe           │   fabric    │      │      (72 GPUs per pod)
      │   ┌───┴───┐            └─────────────┘      │
      │   │ NIC_0 │   (1 NIC per GPU — 72 total)    │
      │   └───┬───┘                                 │
      └───────┼─────────────────────────────────────┘
              │   PCIe Gen5  →  ConnectX NIC  →  IB
              ▼
      ┌─────────────┐           ┌─────────────┐
      │  IB ToR     │ ◄───────► │  IB Spine   │          ← scale-out Clos
      │ (Quantum-2  │           │ (Quantum-2) │            (leaf ↔ spine per §1)
      │  "leaf")    │           └─────────────┘            4-hop cross-pod path:
      └─────────────┘                                      NIC → ToR → Spine → ToR → NIC
```

**Mapping to the abstract hierarchy.**

| Abstract tier (used in §2) | NVL72 SuperPOD realization |
|---|---|
| **Innermost** (scale-up) | NVLink Gen5 within the pod, 72 GPUs, 1800 GB/s per GPU |
| Outer leaf | IB Quantum-2 / Quantum-X800 ToR |
| Outer spine | IB Quantum-2 / Quantum-X800 spine |
| Outer super-spine | Multi-pod deployments (>72 GPUs across multiple NVL72s) |

**The common confusion — "leaf" ≠ "innermost tier."** In Clos literature, **leaf** is a position in the switched fabric — the *first* switch an endpoint hits. For NVL72 SuperPODs, that first switch is the IB ToR. But the *innermost tier of the overall hierarchy* is the NVSwitch/NVLink fabric, which sits **below** the NIC in the topology — intra-pod traffic never passes through the NIC or ToR at all. The IB ToR is the entry point of the *outer* tier, not the inside. The gap between the two tiers is large — roughly ~36× in per-GPU BW (1800 GB/s NVLink vs ~50 GB/s NDR) and several × in α (NVSwitch cut-through vs cross-pod IB path). §2 picks this up and shows how to compose collectives across tiers so that the fast inner fabric carries most of the work and the slow outer fabric is crossed as little as possible.

**How are 72 NICs actually wired to IB ToRs?** The two-fabric diagram above draws the IB ToR as a single box, which is a deliberate abstraction — the previous paragraphs care only about *which fabric* a packet uses (NVLink vs IB), not about how many ToRs the scale-out fabric has. But in real hardware, an NVL72's 72 NICs fan out to actual switches, and the choice of fan-out pattern determines port-count requirements, failure blast radius, and whether NCCL can exploit rail locality. Two layouts appear in production deployments.

A natural first guess: attach all 72 NICs to one ToR. For full bisection ($s = 1$), that ToR would need 72 downlinks to NICs + 72 uplinks to the spine — exactly **144 ports**. Quantum-X800 Q3400's radix is 144 [QX800], so this math works by construction. Whether this "pod-local" layout is what NVIDIA actually ships is a separate question; it turns out the SuperPOD reference architecture uses a different pattern (rail-optimized) that also lands on 144-port ToRs but for a different reason. Both are worth walking through.

**Layout 1 — Pod-local ToR.** All 72 NICs from one NVL72 attach to a single ToR; 72 downlinks to NICs + 72 uplinks to the spine = exactly 144 ports at $s = 1$ full-bisection, matching Quantum-X800 Q3400's radix [QX800] one-for-one.

```
   Pod-local layout (1 ToR per NVL72; all 72 NICs into one X800):

      ┌─────────────── one NVL72 (72 GPUs / 72 NICs) ──────────────┐
      │                                                            │
      │     tray  0:    [N]  [N]  [N]  [N]                         │
      │     tray  1:    [N]  [N]  [N]  [N]                         │
      │       ...                                                  │
      │     tray 17:    [N]  [N]  [N]  [N]                         │
      │                                                            │
      └─────────────────────────────┬──────────────────────────────┘
                                    │  all 72 NICs (one bundle)
                                    ▼
                        ┌─────────────────────────┐
                        │  Quantum-X800 Q3400     │   ← 144 ports:
                        │  (one ToR per NVL72)    │     72 down = 72 NICs
                        └────────────┬────────────┘     72 up  → IB spine
                                     │
                                     ▼
                              ═══ IB spine fabric ═══
```

Simple, but the entire NVL72 shares one failure domain, and one ToR per pod adds up fast at SU scale.

**Layout 2 — Rail-optimized (NVIDIA SuperPOD reference [DGX-SUPERPOD]).** NICs at the same *position within a compute tray* across *different* NVL72 pods share one ToR. Each GB200 NVL72 compute tray holds 4 GPUs / 4 NICs, so each NIC goes to one of 4 rails. Every NVL72 is striped across 4 rail-ToRs, so any one ToR sees only 18 NICs per pod (one per tray), letting a 144-port X800 aggregate multiple pods on the same rail within one SU.

```
   Rail-optimized fan-out (NVL72 striped across 4 rail-ToRs):

      ┌─────────────── one NVL72 (72 GPUs / 72 NICs) ──────────────┐
      │                                                            │
      │     tray  0:   [r0]  [r1]  [r2]  [r3]                      │
      │     tray  1:   [r0]  [r1]  [r2]  [r3]                      │
      │       ...                                                  │
      │     tray 17:   [r0]  [r1]  [r2]  [r3]                      │
      │                 │     │     │     │                        │
      └─────────────────┼─────┼─────┼─────┼────────────────────────┘
                        │     │     │     │
                        ▼     ▼     ▼     ▼
                     ┌─────┬─────┬─────┬─────┐
                     │ToR_0│ToR_1│ToR_2│ToR_3│   ← one Quantum-X800 ToR per rail
                     │144p │144p │144p │144p │     (18 NICs from this NVL72 each)
                     └──┬──┴──┬──┴──┬──┴──┬──┘
                        │     │     │     │
                        ▼     ▼     ▼     ▼
                          ═══ IB spine fabric ═══

      Each ToR's 144 ports = 72 NIC downlinks (= 18 × up to 4 NVL72s on this rail
                                                  at s = 1 full-bisection)
                            + 72 spine uplinks.
```

**The inverse view — what one rail-i ToR sees.** The diagram above shows one NVL72 fanning out to 4 rail-ToRs. The flip side is what each rail-ToR collects: it aggregates the rail-i NICs from up to 4 NVL72 pods. Critically, **each pod contributes only 18 NICs to this ToR** (its rail-i slice — one NIC per tray); the pod's other 54 NICs land on the other three rails' ToRs. So one ToR is *not* "consumed" by a single pod — four pods can share it.

```
   Inverse view: one rail-i ToR_Z aggregates rail-i NICs from up to 4 NVL72s

      ┌─NVL72_A─┐  ┌─NVL72_B─┐  ┌─NVL72_C─┐  ┌─NVL72_D─┐
      │ rail-i  │  │ rail-i  │  │ rail-i  │  │ rail-i  │
      │ 18 NICs │  │ 18 NICs │  │ 18 NICs │  │ 18 NICs │
      └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
           │            │            │            │
           ▼            ▼            ▼            ▼
      ┌────┬────────────┬────────────┬────────────┬────┐
      │           rail-i ToR_Z (Quantum-X800)          │   ← 144 ports:
      │                                                │     72 down = 4 × 18 NICs
      └───────────────────────┬────────────────────────┘     72 up  → IB spine
                              │
                              ▼
                       ═══ IB spine fabric ═══
```

This is what makes "Same ToR" cross-pod traffic feasible: ToR_Z physically sees 4 pods' rail-i NICs and can bridge any pair of them in 2 hops without involving the spine. Beyond 4 NVL72s on a rail, a second rail-i ToR is needed and cross-ToR-on-rail-i traffic must use the spine (4-hop path).

**Traffic behavior under rail-optimized.** First, the NVLink rule: **intra-pod traffic (GPU_A ↔ GPU_B inside the same NVL72) always uses NVLink** — IB never enters the picture, regardless of rail assignment. The cases below are all *cross-pod*; they must use IB because NVLink does not extend off-pod, and the rail-optimized topology determines how many IB hops are needed.

- **Cross-pod, same rail, same ToR** — NVL72_X and NVL72_Y both attach their rail-i NICs to the same physical ToR_Z: **2 hops** (NIC → ToR_Z → NIC). The IB spine is not needed because one ToR bridges both pods. This applies when up to 4 NVL72s share a rail's ToR (144-port X800 at $s = 1$: 72 down ÷ 18 NICs/pod = 4 pods).
- **Cross-pod, same rail, different ToRs** — once the rail spans more than 4 NVL72s, rail-i NICs spread across rail-i ToR_A and rail-i ToR_B: **4 hops** (NIC → ToR_A → IB spine → ToR_B → NIC). The spine is the only physical path between two same-rail ToRs.
- **Cross-pod, cross-rail** — GPU_i on rail i in pod A ↔ GPU_j on rail j ≠ i in pod B: **NVLink rail bridge — 1 NVLink hop + 4 IB hops on rail j.** The 4 rail fabrics are physically disjoint at every tier (no IB cable connects rail-i switches to rail-j switches), so cross-rail does *not* go through any IB spine. Instead, the source first sends over NVLink intra-pod-A to a sibling GPU whose NIC lands on rail j, then that sibling uplinks normally on rail j (NIC → ToR_j → spine_j → ToR_j → NIC). Cross-rail pays one extra ~0.5 μs intra-pod NVLink hop on top of the destination rail's 4-hop IB cost.

NCCL auto-detects rails and prefers same-rail peering for α-sensitive TP traffic [DGX-SUPERPOD] — a 2-hop same-rail path beats the 4-hop cross-rail one. This is why rail-optimized is the production norm despite using 4× more ToRs than pod-local.

**Bottom line on the 144-port ToR usage.** Both layouts use 144-port X800, but sized for different jobs:

- **Pod-local**: 144 = 72 NICs + 72 spine uplinks, one NVL72 exactly ([QX800]).
- **Rail-optimized**: 144 ports aggregates a rail across up to 4 NVL72s — ~72 NICs from multiple pods + 72 spine uplinks ([DGX-SUPERPOD]).

The X800 radix matches both interpretations comfortably, which is why Q3400 is the reference ToR for GB200 SuperPODs.

### 1.2 Cost-model symbols anchored on this SuperPOD

§2's cost formulas and case studies use a small set of symbols. We anchor their meaning and concrete values once here so downstream subsections can plug in numbers without re-introducing notation. The values come directly from the NVL72 + IB hardware described in §1.1 above.

*Latency (α).* Three distinct switch paths surface in production, matching the three traffic classes from §1.1's "Traffic behavior under rail-optimized":

| Symbol | Path | NVL72 + IB value |
|---|---|---|
| $\alpha_{\mathrm{inner}}$ | Intra-pod GPU↔GPU on scale-up (NVSwitch cut-through) | $\approx 0.5\\,\mu$s |
| $\alpha_{\mathrm{leaf}}$ | Cross-pod 2-hop IB path (NIC → shared rail-ToR → NIC) | $\approx 2\\,\mu$s |
| $\alpha_{\mathrm{spine}}$ | Cross-pod 4-hop IB path (NIC → ToR → spine → ToR → NIC), end-to-end including NIC serialization | $\approx 8\\,\mu$s |

$\alpha_{\mathrm{leaf}}$ applies when pods share a rail-ToR — possible only at small scale, since a 144-port X800 at $s = 1$ holds at most 4 NVL72s per rail. $\alpha_{\mathrm{spine}}$ is the general cross-pod cost once the rail spans more than 4 NVL72s.

§2.1's hierarchical-AR formulas use a *derived* symbol $\alpha_{\mathrm{outer}}$ for the cross-pod α — the **effective** cross-pod latency for the deployment, equal to:

- $\alpha_{\mathrm{outer}} = \alpha_{\mathrm{leaf}}$ when all participating pods share a rail-ToR (small scale).
- $\alpha_{\mathrm{outer}} = \alpha_{\mathrm{spine}}$ when pods span multiple rail-ToRs (production scale).
- A weighted average of the two when the schedule has both same-ToR and spine-traversing hops (e.g., a ring AR across 8 pods split 4-and-4 across two rail-ToRs sees 6 same-ToR hops at $\alpha_{\mathrm{leaf}}$ and 2 cross-ToR hops at $\alpha_{\mathrm{spine}}$).

Case studies plug in concrete values: §2.1's $L = 2$ walk-through uses $\alpha_{\mathrm{outer}} = \alpha_{\mathrm{leaf}}$ (small-scale, pods share a rail-ToR); a production-scale $L = 32$ deployment would use $\alpha_{\mathrm{outer}} = \alpha_{\mathrm{spine}}$ instead. §2.2's A2A formula itemizes by destination, so it doesn't use $\alpha_{\mathrm{outer}}$ — all three primitive α values ($\alpha_{\mathrm{inner}}$, $\alpha_{\mathrm{leaf}}$, $\alpha_{\mathrm{spine}}$) appear directly.

*Bandwidth (BW), per GPU per direction.*

| Symbol | Source | NVL72 + IB value |
|---|---|---|
| $\mathrm{BW}_{\mathrm{inner}}$ | NVLink Gen5 (per direction; 1800 GB/s bidirectional aggregate) | $\approx 900\\,\mathrm{GB/s}$ |
| $\mathrm{BW}_{\mathrm{outer}}$ | ConnectX-7 NDR (ConnectX-8 XDR doubles to 100 GB/s) | $\approx 50\\,\mathrm{GB/s}$ |

In multi-tier formulas, $\mathrm{BW}_{\mathrm{bottleneck}} \approx \mathrm{BW}_{\mathrm{outer}}$ once any cross-pod traffic is involved.

*Shape parameters.* $N$ = total ranks; $L$ = number of pods; $N_{\mathrm{inner}} = N/L$ = GPUs per pod (= 72 on NVL72); $p$ = pods sharing a rail-ToR ($\leq 4$ at $s = 1$).

These symbols carry through §2 unchanged.

---

## 2. Composition rules for hierarchical collectives

**The core idea.** In single-tier, a collective like AR is one algorithm running over one fabric — pick a ring, a DBT, or Rabenseifner, evaluate with that fabric's $(\alpha, \mathrm{BW})$, done. In multi-tier, the same user-level collective often gets re-expressed as a **combination of different primitives running on different tiers**. A hierarchical AR isn't "run AR on tier 1, then AR on tier 2" — it's a sequence like **RS on the inner tier, AR on the outer tier, AG on the inner tier** that composes to the same result. The per-tier primitives need not be the same primitive the user asked for; they're chosen so each tier does the work it's best at, and so the outer tier sees as little data as possible.

**Why this re-expression is possible (and necessary).** Two reasons.

1. **Tiers are heterogeneous.** An intra-pod hop costs ~0.5 μs at ~900 GB/s on NVSwitch while a cross-pod hop costs 2–8 μs at ~50 GB/s on InfiniBand. A flat schedule that treats the Clos as one big star pays $\log N \cdot \alpha_{\mathrm{spine}}$ on every algorithmic hop and $2(N-1)/N \cdot M/\mathrm{BW}_{\mathrm{outer}}$ on every byte — wasting the fast intra-pod fabric entirely. Decomposing across tiers lets each tier pay only its own α on its own phase, and lets the outer tier carry a shrunk payload rather than the full $M$.
2. **The collective's algebraic structure allows it.** Reduction is associative — reducing locally first then globally gives the same answer as reducing in one flat pass. Broadcast is replication — copying locally then globally is equivalent to one flat copy. These are the structural hooks that let the single-tier collective be factored into a multi-phase schedule. Each of AR, AG, RS, BC, Reduce has such a factoring; the specific phase pattern differs (§2.1 tabulates all five).

**Two design levers in every hierarchical schedule.** Once you accept that a single-tier collective can be re-expressed as a cross-tier combination, the schedule has two knobs: *which primitive runs on which tier*, and *how much payload crosses each tier boundary*. The canonical AR pattern (RS → sub-AR → AG) uses both: it puts the bulk of the α-hops on the fast inner tier and shrinks the payload from $M$ to $ML/N$ before any byte crosses the slow outer tier. §2.1 develops this pattern for AR and shows how AG, RS, BC, Reduce fall out as halves or trivial cascades of it.

**The exception: A2A.** Not every collective has the algebraic structure to decompose. All-to-all has no associativity or replication hook — every source-destination pair carries a distinct payload, so the cross-tier permutation traffic equals the full $(N-1)/N \cdot M$ bytes regardless of how the schedule is phased. §2.2 works through why A2A doesn't decompose and how its cost formula becomes a destination-weighted sum over distance classes rather than a phased pipeline.

The cost formulas below use the symbols defined in §1.2 — $\alpha_{\mathrm{inner}}$, $\alpha_{\mathrm{leaf}}$, $\alpha_{\mathrm{spine}}$, $\mathrm{BW}_{\mathrm{inner}}$, $\mathrm{BW}_{\mathrm{outer}}$, plus the shape parameters $N$, $L$, $N_{\mathrm{inner}}$, $p$. Refer back to that table for concrete NVL72 + IB values when plugging numbers in.

### 2.1 The RS → sub-AR → AG pattern

Hierarchical AR composes the primitives from `01_collective_algorithms.md` across tiers by exploiting reduction's associativity. For a 2-tier hierarchy with $L$ outer groups of $N/L$ inner ranks each, the canonical schedule is:

```
Phase 1 (intra-group RS):   each of L groups runs independent RS across its N/L ranks
                            → each rank now holds (L/N)·M bytes of partially-reduced data
                            → per-group RS runs on the inner tier's fabric at (α_inner, BW_inner)

Phase 2 (cross-group AR):   L ranks-per-rank-position run AR across the outer tier
                            → the M·L/N byte payload now sees the outer tier's (α_outer, BW_outer)
                              (α_outer = α_leaf, α_spine, or a mix — see §1.2)
                            → this AR can itself be another hierarchical AR if k > 2 tiers

Phase 3 (intra-group AG):   each of L groups runs independent AG across its N/L ranks
                            → each rank now holds the fully-reduced M bytes
                            → reverse of phase 1 on the inner tier
```

**Why this composes correctly.** Reduction is associative, so the inner-then-outer order produces the same result as a flat AR across all $N$ ranks. The per-rank chunk size telescopes: after inner RS, each rank holds $M / (N/L) = ML/N$ bytes (not $M$), so the outer AR moves less data. This is the same bandwidth-telescoping argument as torus dim-decomposition (`02_topology_mapping.md §3.1`), generalized across heterogeneous tiers.

**Worked example — N = 4, L = 2.** Four ranks {A, B, C, D}; leaf 0 holds {A, B}, leaf 1 holds {C, D}. Each rank starts with a 4-chunk vector; target is every rank ending with $\Sigma = [\Sigma_0, \Sigma_1, \Sigma_2, \Sigma_3]$ where $\Sigma_k = a_k + b_k + c_k + d_k$ sums all four ranks' chunk-$k$ values.

```
  Initial state              Phase 1: intra-leaf RS       Phase 2: cross-leaf sub-AR
  (each rank full M)         (N/L = 2 ranks per leaf)     (L = 2 ranks per chunk pair)
                             [runs on NVSwitch fabric]    [runs on IB ToR + spine]
                             [intra-pod, fast]            [cross-pod, slow]

  Leaf 0                     Leaf 0                       Leaf 0
    A: [a0,a1,a2,a3]           A: [a0+b0, a1+b1,  · ,  · ]  A: [ Σ0 ,  Σ1 ,  · ,  · ]
    B: [b0,b1,b2,b3]   ──►     B: [ · ,  · , a2+b2,a3+b3]   B: [ · ,  · ,  Σ2 ,  Σ3 ]

  Leaf 1                     Leaf 1                       Leaf 1
    C: [c0,c1,c2,c3]           C: [c0+d0, c1+d1,  · ,  · ]  C: [ Σ0 ,  Σ1 ,  · ,  · ]
    D: [d0,d1,d2,d3]   ──►     D: [ · ,  · , c2+d2,c3+d3]   D: [ · ,  · ,  Σ2 ,  Σ3 ]

  Payload per rank: M        Payload per rank: M·L/N       Sub-AR pairs (one AR per
                             = M/2 (leaf-partial sums      chunk-position group):
                              on 2 chunks)                   {A, C} sum chunks 0, 1
                                                             {B, D} sum chunks 2, 3

  Phase 3: intra-leaf AG (exchange chunks within each leaf)
  [runs on NVSwitch fabric — mirrors Phase 1]

  Leaf 0: A ↔ B exchange         A: [Σ0, Σ1, Σ2, Σ3]
                                 B: [Σ0, Σ1, Σ2, Σ3]
  Leaf 1: C ↔ D exchange         C: [Σ0, Σ1, Σ2, Σ3]
                                 D: [Σ0, Σ1, Σ2, Σ3]

  AR complete: every rank holds the full Σ.
```

**Decoding "sub-AR".** The phase-2 AR runs on a sub-group of $L$ ranks (one per leaf), not on all $N$. There are $N/L$ such sub-groups — one per chunk-position — and they run concurrently. Each sub-AR completes the reduction for its chunk positions across leaves: if A holds the leaf-0 partial of chunks 0–1 and C holds the leaf-1 partial of the same chunks, AR on {A, C} produces the full $N$-way reduction for chunks 0–1. The trick is that RS in phase 1 shrinks the "which chunks each rank owns" set to $L/N$ of the full vector, so the phase-2 AR only needs to operate on that narrow slice — paying $L$-ranks α-cost on an $ML/N$-byte payload instead of $N$-ranks α-cost on $M$ bytes. Phase 3 AG then broadcasts the fully-reduced slices back within each leaf so every rank ends up with all $M$ bytes.

**Cost formula.** Summing the three phases:

$$t_{\mathrm{AR}} \\;=\\; t_{\mathrm{RS,inner}}\\!\left(\tfrac{N}{L}, M\right) \\;+\\; t_{\mathrm{AR,outer}}\\!\left(L, \tfrac{ML}{N}\right) \\;+\\; t_{\mathrm{AG,inner}}\\!\left(\tfrac{N}{L}, M\right)$$

Each term instantiates the primitive cost formulas from `01_collective_algorithms.md` / `02_topology_mapping.md` with the matching tier's $(\alpha, \mathrm{BW})$. For $k > 2$ tiers, the outer AR term is itself hierarchical — recursion down the tree until the innermost tier is reached.

**Inner-tier cost in composed deployments.** When the Clos leaf port attaches a whole scale-up pod (e.g., an NVL72 with 72 GPUs behind one NVSwitch) rather than a single GPU, the "inner tier" in the hierarchical schedule *is* that scale-up fabric: $t_{\mathrm{RS,inner}}$ and $t_{\mathrm{AG,inner}}$ instantiate the star (or torus, or mesh) cost from `02_topology_mapping.md §5.1` at the pod's $(\alpha_{\mathrm{inner}}, \mathrm{BW}_{\mathrm{inner}})$; the outer AR phase instantiates the Clos AR at $(\alpha_{\mathrm{outer}}, \mathrm{BW}_{\mathrm{outer}})$, where $\alpha_{\mathrm{outer}}$ resolves to $\alpha_{\mathrm{leaf}}$, $\alpha_{\mathrm{spine}}$, or a mix per the §1.2 catalog. **Intra-pod α-β cost is part of the hierarchical total by construction — not an overhead layered on top.** The same holds for AG, RS, BC, Reduce: each row of the summary table below evaluates its `inner` term at the scale-up fabric's cost.

**The $\alpha$ and BW structure.** For 2-tier with ring-on-ring:

$$t_{\mathrm{AR}}^{\mathrm{2\text{-}tier}} \approx 2\\!\left(\tfrac{N}{L} - 1\right)\alpha_{\mathrm{inner}} + 2(L-1)\alpha_{\mathrm{outer}} + \frac{2(N-1)}{N}\cdot\frac{M}{\mathrm{BW}_{\mathrm{bottleneck}}}$$

where $\mathrm{BW}_{\mathrm{bottleneck}} = \min(\mathrm{BW}_{\mathrm{inner}}, \mathrm{BW}_{\mathrm{outer}} \cdot N/(L \cdot \text{cross-tier share}))$ accounts for the lower-throughput tier dominating BW. In practice $\mathrm{BW}_{\mathrm{outer}} \ll \mathrm{BW}_{\mathrm{inner}}$ (IB at 50 GB/s vs NVLink at 900 GB/s), so $\mathrm{BW}_{\mathrm{bottleneck}} \approx \mathrm{BW}_{\mathrm{outer}}$ once the outer tier participates.

**AG, RS, BC, Reduce drop out.** The three-phase pattern above already contains AG and RS as its two halves: standalone **AG** runs inner AG → outer AG (phase 3 logic only, no reduction); standalone **RS** runs outer RS → inner RS (phase 1 logic, time-reverse). **BC** and **Reduce** cascade trivially across tiers — no reduction to amortize. The full summary:

| Primitive | 2-tier composition | α term | BW term (at $s=1$, uniform BW) |
|---|---|---|---|
| **AR** | inner RS → outer AR → inner AG | $2(N/L - 1)\alpha_{\mathrm{inner}} + 2(L - 1)\alpha_{\mathrm{outer}}$ | $2(N-1)/N \cdot M/\mathrm{BW}$ |
| **AG** | inner AG → outer AG | $(N/L - 1)\alpha_{\mathrm{inner}} + (L - 1)\alpha_{\mathrm{outer}}$ | $(N-1)/N \cdot M/\mathrm{BW}$ |
| **RS** | outer RS → inner RS | $(L - 1)\alpha_{\mathrm{outer}} + (N/L - 1)\alpha_{\mathrm{inner}}$ | $(N-1)/N \cdot M/\mathrm{BW}$ |
| **BC** | outer BC → inner BC | $\lceil\log_2 L\rceil\\,\alpha_{\mathrm{outer}} + \lceil\log_2(N/L)\rceil\\,\alpha_{\mathrm{inner}}$ | $M/\mathrm{BW}$ (pipelined) |
| **Reduce** | inner Reduce → outer Reduce | $\lceil\log_2(N/L)\rceil\\,\alpha_{\mathrm{inner}} + \lceil\log_2 L\rceil\\,\alpha_{\mathrm{outer}}$ | $M/\mathrm{BW}$ (pipelined) |

(Plug in $\alpha_{\mathrm{outer}} = \alpha_{\mathrm{leaf}}$ for small deployments where pods share a rail-ToR, $\alpha_{\mathrm{outer}} = \alpha_{\mathrm{spine}}$ for production deployments spanning multiple rail-ToRs, or a weighted mix when the schedule has both same-ToR and spine-traversing hops; see §1.2.)

For oversubscription $s > 1$ at any tier boundary, multiply the BW term by $s$ on traffic crossing that boundary — equivalent to $\eta_\beta \approx 1/s$ in `05_contention_and_congestion.md` §4.2. AG and RS cost **half of AR** on both α (one pass, not two) and BW (one RS-style telescoping, not the doubled AR round-trip); BC and Reduce stay at the $M/\mathrm{BW}$ pipelined ceiling regardless of tier count, with α scaling log-depth on each tier.

**Case study: NVL72 + IB SuperPOD (AR).** Using the shared terminology values, take a 2-pod deployment ($L = 2$, $N = 144$, $M = 16\\,\mathrm{MB}$, ring-on-ring schedule). At this scale the two pods share a rail-ToR ($p = 2$, well within the $p \leq 4$ budget), so $\alpha_{\mathrm{outer}} = \alpha_{\mathrm{leaf}} \approx 2\\,\mu$s for the outer phase:

- **Inner RS** (ring on 72 GPUs over NVSwitch): $(N_{\mathrm{inner}}-1)\\,\alpha_{\mathrm{inner}} = 71 \cdot 0.5 = \mathbf{35.5\\,\mu s}$ α; $\frac{71}{72}\cdot\frac{M}{\mathrm{BW}_{\mathrm{inner}}} \approx \mathbf{17.5\\,\mu s}$ BW.
- **Outer sub-AR** (ring on 2 pods through the shared rail-ToR): payload has telescoped to $ML/N = 0.22\\,\mathrm{MB}$; $2(L-1)\\,\alpha_{\mathrm{outer}} = 2 \cdot 1 \cdot 2 = \mathbf{4\\,\mu s}$ α + $\frac{2(L-1)}{L}\cdot\frac{ML/N}{\mathrm{BW}_{\mathrm{outer}}} \approx \mathbf{4.4\\,\mu s}$ BW.
- **Inner AG**: mirrors inner RS — **35.5 μs α + 17.5 μs BW**.

**Hierarchical total: ≈ 75 μs α + 39 μs BW ≈ 114 μs.**

How does this compare to two hypothetical flat-ring baselines at the same $N = 144$?

| Schedule (all at $N = 144$, $M = 16\\,\mathrm{MB}$) | α (μs) | BW (μs) | **Total** |
|---|---|---|---|
| Flat ring on a hypothetical 144-GPU NVSwitch domain (NVLink doesn't actually span pods, but assume it could): $2(N-1)\\,\alpha_{\mathrm{inner}} + 2(N-1)/N \cdot M/\mathrm{BW}_{\mathrm{inner}}$ | 143 | 35 | **~178** |
| **Hierarchical NVLink + IB** (this case study) | **75** | **39** | **~114** |
| Flat ring forced over IB only (ignore NVLink): $2(N-1)\\,\alpha_{\mathrm{spine}} + 2(N-1)/N \cdot M/\mathrm{BW}_{\mathrm{outer}}$ | 2,288 | 636 | **~2,920** |

Two takeaways:

1. **Hierarchical beats flat-NVSwitch (114 vs 178 μs) even on the same fast fabric.** The α saving comes from running $N/L$ parallel inner rings of length $N/L - 1$ instead of one ring of length $N - 1$. The α formula goes from $2(N-1)\\,\alpha$ to $2(N/L - 1)\\,\alpha + 2(L - 1)\\,\alpha_{\mathrm{outer}}$ — at $L = 2$ this is $286\alpha \to 144\alpha$ on a uniform-α basis, a ~50% reduction *before* the outer tier even gets involved. The "shorter parallel rings" effect; optimal $L \approx \sqrt{N}$ gives roughly $\sqrt{N}$ speedup on the α term.
2. **Hierarchical beats flat-IB by ~26×, almost entirely on α**, because 71 of the 75 μs of α work runs on NVSwitch at $\alpha_{\mathrm{inner}} = 0.5\\,\mu$s instead of on IB at $\alpha_{\mathrm{spine}} = 8\\,\mu$s, *and* payload telescoping cuts what crosses the outer tier from $M$ down to $M/72$.

So hierarchical wins on two distinct axes: shorter parallel inner rings (works on any fabric, beats flat-NVSwitch) plus payload telescoping when the outer tier is slower (turns the IB-side BW from 636 μs into 4.4 μs). At production scale ($L = 32$, pods spanning multiple rail-ToRs so $\alpha_{\mathrm{outer}} = \alpha_{\mathrm{spine}}$), the gap to flat baselines widens further, and SHARP on the spine (§3.1) collapses the outer α to $\sim 2\alpha_{\mathrm{switch}}$.

### 2.2 A2A on hierarchies — the outlier

A2A is the one primitive where hierarchical composition doesn't help. No reduction or replication semantics to exploit across tiers — every source-destination pair carries a distinct payload, and the cross-tier permutation traffic equals the full $(N-1) \cdot (1 - 1/N) \cdot M$ bytes regardless of schedule. The BW bound is the **outermost tier's bisection**; no hierarchical schedule can beat it. If the outer tier is oversubscribed at $s$ or has per-leaf BW below per-endpoint BW, A2A realized BW drops by the full factor. Unlike §2.1, A2A's cost itemizes per destination — each pairwise send pays its destination's *actual* path latency rather than an average — so all three α values from the §2 shared terminology ($\alpha_{\mathrm{inner}}$, $\alpha_{\mathrm{leaf}}$, $\alpha_{\mathrm{spine}}$) appear simultaneously in its cost formula.

**Cost formula.** Implementations ship as pairwise direct-send: each rank serializes $N{-}1$ outgoing messages of size $M/N$ through its outbound port. Each send pays its destination's actual path cost ($\alpha + (M/N)/\mathrm{BW}$); the total is a destination-weighted sum. In a real NVL72 + IB SuperPOD, peers split across **three** distance classes (counts shown for $L$ pods total, $p$ pods sharing a rail-ToR, $N_{\mathrm{inner}}$ GPUs per pod):

| Peer count | Destination class | Path α | Path BW |
|---|---|---|---|
| $N_{\mathrm{inner}} - 1$ | Intra-pod (NVSwitch scale-up) | $\alpha_{\mathrm{inner}}$ | $\mathrm{BW}_{\mathrm{inner}}$ |
| $(p - 1)\\,N_{\mathrm{inner}}$ | Same-leaf cross-pod (through shared rail-ToR) | $\alpha_{\mathrm{leaf}}$ | $\mathrm{BW}_{\mathrm{outer}}$ |
| $(L - p)\\,N_{\mathrm{inner}}$ | Cross-leaf cross-pod (through IB spine) | $\alpha_{\mathrm{spine}}$ | $\mathrm{BW}_{\mathrm{outer}}$ |

The α and BW terms separate out as:

$$t_\alpha^{\mathrm{A2A}} \\;=\\; (N_{\mathrm{inner}} - 1)\\,\alpha_{\mathrm{inner}} \\;+\\; (p - 1)\\,N_{\mathrm{inner}}\\,\alpha_{\mathrm{leaf}} \\;+\\; (L - p)\\,N_{\mathrm{inner}}\\,\alpha_{\mathrm{spine}}$$

$$t_{\mathrm{BW}}^{\mathrm{A2A}} \\;=\\; (N_{\mathrm{inner}} - 1)\\,\frac{M/N}{\mathrm{BW}_{\mathrm{inner}}} \\;+\\; (N - N_{\mathrm{inner}})\\,\frac{M/N}{\mathrm{BW}_{\mathrm{outer}}} \cdot s$$

Intra-pod α contributes as a summand over same-pod peers — **not a literal "intra-pod A2A added on top of cross-pod A2A,"** but a single $N{-}1$-send schedule split by destination class. The inner tier's high BW only helps on the $N_{\mathrm{inner}} - 1$ intra-pod sends; the rest pays $\mathrm{BW}_{\mathrm{outer}}$, slowed further by any outer-tier oversubscription $s$.

**Worst case** (all peers cross-leaf, e.g., MoE EP spread across the full SU): $t_\alpha \to (N{-}1)\\,\alpha_{\mathrm{spine}}$ — the dominant scenario at production scale. **Best case** (all peers same-pod): $t_\alpha \to (N{-}1)\\,\alpha_{\mathrm{inner}}$ — only feasible when $N \leq N_{\mathrm{inner}} = 72$.

**Single-GPU endpoint Clos** (the simpler abstract case where each leaf-ToR port holds one GPU, no scale-up pods). The intra-pod term drops out and the formula reduces to $t_\alpha = (N/L - 1)\\,\alpha_{\mathrm{leaf}} + (N - N/L)\\,\alpha_{\mathrm{spine}}$ where $L$ counts leaf-ToRs and $N/L$ counts GPUs per leaf-ToR. BW = $(N{-}1)/N \cdot M/\mathrm{BW}_{\mathrm{outer}} \cdot s$, uniform across all sends.

**Case study: NVL72 + IB SuperPOD (A2A).** Using the shared terminology values, take the same $L = 2$ pods sharing rail-ToRs as the §2.1 case study ($p = 2$, $N = 144$, $M = 16\\,\mathrm{MB}$, so $M/N \approx 0.111\\,\mathrm{MB}$). A2A's per-destination accounting splits the $N-1 = 143$ peers into two distance classes here (no $\alpha_{\mathrm{spine}}$ peers because $p = L$). Each rank serializes 143 pairwise sends:

- **71 intra-pod sends** over NVLink: $71 \cdot \\!\bigl(\alpha_{\mathrm{inner}} + (M/N)/\mathrm{BW}_{\mathrm{inner}}\bigr) = 71 \cdot (0.5 + 0.12)\\,\mu$s ≈ **44 μs**.
- **72 same-leaf cross-pod sends** over IB through shared ToR: $72 \cdot \\!\bigl(\alpha_{\mathrm{leaf}} + (M/N)/\mathrm{BW}_{\mathrm{outer}}\bigr) = 72 \cdot (2 + 2.2)\\,\mu$s ≈ **304 μs**.

**Hierarchical A2A total ≈ 348 μs.**

How does this compare to flat-A2A baselines at the same $N = 144$?

| Schedule ($N = 144$, $M = 16\\,\mathrm{MB}$) | α (μs) | BW (μs) | **Total** |
|---|---|---|---|
| Flat A2A on hypothetical 144-GPU NVSwitch (NVLink can't actually span pods): $(N{-}1)\\,\alpha_{\mathrm{inner}} + (N{-}1)\\,(M/N)/\mathrm{BW}_{\mathrm{inner}}$ | 71.5 | 17.6 | **~89** |
| **Hierarchical A2A on NVL72 + IB at $L=2$ same rail-ToR** (this case study) | 179 | 169 | **~348** |
| Hierarchical A2A on NVL72 + IB at $L=2$ *different* rail-ToRs (spine traversal): $\alpha_{\mathrm{leaf}}$ replaced by $\alpha_{\mathrm{spine}}$ for cross-pod sends | 611 | 169 | **~780** |
| Flat A2A on IB only (no NVLink, every peer via spine): $(N{-}1)\\,\alpha_{\mathrm{spine}} + (N{-}1)\\,(M/N)/\mathrm{BW}_{\mathrm{outer}}$ | 1,144 | 318 | **~1,460** |

**Lesson: don't pull A2A out of the inner domain.** Unlike AR, A2A loses every time it crosses a tier boundary. AR's hierarchical decomposition saves on both α (parallel inner rings) and BW (payload telescoping); A2A has *neither* lever — every send carries a distinct payload (no reduction to amortize, no telescoping), and serial pairwise sends pay full α per cross-pod hop (no shorter-rings shortcut). On the same $N = 144$, flat-NVSwitch A2A (89 μs) is **~4× faster** than the same-ToR hierarchical version (348 μs) and **~9× faster** than the cross-ToR variant (780 μs).

This is why MoE expert parallelism placement on the inner-most fabric (NVL72 NVSwitch within a single pod) is the production rule whenever it fits. NCCL's same-rail / same-ToR preference for A2A-heavy traffic mitigates the penalty when EP must span pods, but no scheduling trick recovers the full intra-pod cost — every GPU pair pulled out of NVLink roughly quadruples its A2A contribution.

### 2.3 When hierarchical helps, when it hurts

**Hierarchical composition helps when:**

- **BW is tier-mismatched.** If the inner tier is 10–20× faster than the outer (NVLink vs InfiniBand), doing as much of the reduction as possible inside the inner tier shrinks the outer tier's payload from $M$ to $ML/N$. For a 2-tier system with $L = 16$ inner groups and $N = 1024$, the outer tier carries only $M/64$ of the original payload — a 64× reduction in cross-tier BW demand.
- **Latency is tier-mismatched.** Inner-tier α is ~0.5 μs (NVSwitch), outer-tier α is ~2–5 μs (IB), so doing most algorithmic hops on the inner tier wins both on the α term and on the BW term.

**Hierarchical composition hurts when:**

- **Collective is A2A** (see §2.2 — no decomposition gain).
- **Outer tier is fast enough to do a flat AR.** A deployment entirely inside one NVL72 pod has no outer-tier penalty, and flat star AR (DBT / INC) beats any hierarchical scheme.
- **Group shape doesn't factor nicely.** If $N = 513$ doesn't divide into a clean (inner, outer) factorization, hierarchical phases pay padding overhead.
- **Outer tier oversubscription $s$ is large.** See §3 — high $s$ compresses the hierarchical advantage.

---

## 3. INC and contention in hierarchies

In a multi-tier hierarchy, every reduction-or-replication primitive accumulates α at *both* tiers (§2.1's summary table). At production scale ($N_{\mathrm{inner}} = 72$, $L = 32$, $\alpha_{\mathrm{inner}} = 0.5\\,\mu$s, $\alpha_{\mathrm{outer}} = \alpha_{\mathrm{spine}} = 8\\,\mu$s, software DBT), AR pays ~7 μs of inner α + ~80 μs of outer α; AG/RS/BC/Reduce pay half each. The outer term is the biggest single contributor in absolute terms, but the inner term is not negligible — and once the outer term is compressed, the inner term becomes the new bottleneck.

The single biggest lever to compress these costs is **in-network collectives (INC)**: pushing reduction or multicast logic into the switch ASIC itself, so the affected phase pays $\sim 2\\,\alpha_{\mathrm{switch}}$ (a few hundred ns of switch cut-through) instead of $O(\log L)$ endpoint-driven rounds. INC applies at *both* tiers of the hierarchy, and **both are valuable**:

- **Outer-tier INC** (Quantum SHARP on IB Quantum-2 / X800, Spectrum-X SHARP on Ethernet, Tomahawk Ultra-based fabrics) sees the **biggest absolute α saving** — collapses ~80 μs of outer DBT down to ~1 μs at $L = 32$. This is what gives hierarchical AR at $L = 32$ pods its production-relevance.
- **Inner-tier INC** (NVLink SHARP / NVLS on NVSwitch Gen4 within an NVL72 pod) is **equally important** for two reasons: (i) it compresses the inner α from ~7 μs to ~0.4 μs, which becomes the dominant residual once outer INC is deployed; (ii) it lifts intra-pod AR's $\mathrm{BW_{eff}}$ from $\mathrm{BW}/2$ to $\mathrm{BW}$ — a measured ~1.3× BW win (470+ GB/s vs ~360 GB/s on H100, [NVLINK-SHARP]) that the outer tier can't replicate because the dual-touch pattern only manifests on endpoint-driven trees.

The two compose: a deployment with INC at both tiers gets the outer α collapse plus the inner BW lift simultaneously. INC is hardware-gated; see `04_in_network_collectives.md` §2 for per-fabric mechanics. This section focuses on how INC composes with the hierarchical schedules of §2.

Two cross-cuts:

- **§3.1**: SHARP at either or both tiers collapses the corresponding phase of every reduction-or-replication primitive — uniform structural saving across AR, AG, RS, BC, Reduce. A2A is the lone exception (no fit on traditional INC hardware; dedicated HW A2A on Rubin / Tomahawk Ultra is a separate primitive).
- **§3.2**: per-tier η coefficients (NVSwitch crossbar contention, IB Clos oversubscription) feed back into §2.1's hierarchical formulas, modifying realized cost without changing the algorithm shape.

### 3.1 SHARP composes at any tier of the hierarchy

The SHARP family collapses $n_\alpha$ to ~2 on its host fabric (`04_in_network_collectives.md` §1). In a hierarchical schedule (§2.1), SHARP can be installed independently at the inner tier, the outer tier, or both — each replacing its tier's endpoint-driven phase with an INC operation at switch cut-through latency $\alpha_{\mathrm{switch}}$. Tiers without INC continue to use software DBT or ring.

**Per-tier SHARP impact on the α term**, at production scale ($N_{\mathrm{inner}} = 72$, $L = 32$, $\alpha_{\mathrm{inner}} = 0.5\\,\mu$s, $\alpha_{\mathrm{outer}} = \alpha_{\mathrm{spine}} = 8\\,\mu$s; switch latencies $\alpha_{\mathrm{sw}}^{\mathrm{inner}} \approx 0.2\\,\mu$s for NVLS, $\alpha_{\mathrm{sw}}^{\mathrm{outer}} \approx 0.5\\,\mu$s for Quantum SHARP):

| Primitive | Inner α (software DBT) | Inner α (NVLS) | Outer α (software DBT) | Outer α (Quantum SHARP) |
|---|---|---|---|---|
| **AR** | $2\lceil\log_2 N_{\mathrm{inner}}\rceil\\,\alpha_{\mathrm{inner}} \approx 7\\,\mu$s | $\sim 2\\,\alpha_{\mathrm{sw}}^{\mathrm{inner}} \approx 0.4\\,\mu$s | $2\lceil\log_2 L\rceil\\,\alpha_{\mathrm{outer}} \approx 80\\,\mu$s | $\sim 2\\,\alpha_{\mathrm{sw}}^{\mathrm{outer}} \approx 1\\,\mu$s |
| **AG** | $\lceil\log_2 N_{\mathrm{inner}}\rceil\\,\alpha_{\mathrm{inner}} \approx 3.5\\,\mu$s | $\sim \alpha_{\mathrm{sw}}^{\mathrm{inner}} \approx 0.2\\,\mu$s | $\lceil\log_2 L\rceil\\,\alpha_{\mathrm{outer}} \approx 40\\,\mu$s | $\sim \alpha_{\mathrm{sw}}^{\mathrm{outer}} \approx 0.5\\,\mu$s |
| **RS** | $\lceil\log_2 N_{\mathrm{inner}}\rceil\\,\alpha_{\mathrm{inner}} \approx 3.5\\,\mu$s | $\sim \alpha_{\mathrm{sw}}^{\mathrm{inner}} \approx 0.2\\,\mu$s | $\lceil\log_2 L\rceil\\,\alpha_{\mathrm{outer}} \approx 40\\,\mu$s | $\sim \alpha_{\mathrm{sw}}^{\mathrm{outer}} \approx 0.5\\,\mu$s |
| **BC** | $\lceil\log_2 N_{\mathrm{inner}}\rceil\\,\alpha_{\mathrm{inner}} \approx 3.5\\,\mu$s | $\sim \alpha_{\mathrm{sw}}^{\mathrm{inner}} \approx 0.2\\,\mu$s | $\lceil\log_2 L\rceil\\,\alpha_{\mathrm{outer}} \approx 40\\,\mu$s | $\sim \alpha_{\mathrm{sw}}^{\mathrm{outer}} \approx 0.5\\,\mu$s |
| **Reduce** | $\lceil\log_2 N_{\mathrm{inner}}\rceil\\,\alpha_{\mathrm{inner}} \approx 3.5\\,\mu$s | $\sim \alpha_{\mathrm{sw}}^{\mathrm{inner}} \approx 0.2\\,\mu$s | $\lceil\log_2 L\rceil\\,\alpha_{\mathrm{outer}} \approx 40\\,\mu$s | $\sim \alpha_{\mathrm{sw}}^{\mathrm{outer}} \approx 0.5\\,\mu$s |
| **A2A** | $(N_{\mathrm{inner}}{-}1)\\,\alpha_{\mathrm{inner}} \approx 35.5\\,\mu$s | No NVLS fit on Gen4<sup>*</sup> | — (no decomposition; §2.2) | No fit on traditional SHARP |

<sup>*</sup>NVSwitch Gen4 NVLS supports AR / AG / BC but not A2A; Rubin-generation NVSwitches are expected to extend to A2A, and Tomahawk Ultra adds HW A2A on Ethernet today (`04_in_network_collectives.md` §1.2).

**Three structural points:**

1. **Both tiers benefit from INC, in different ways.** Outer-tier INC dominates *absolute* α savings (80 → 1 μs is the biggest single move at $L = 32$). But once outer INC is in place, inner-tier α becomes the new dominant cost (7 of 8 μs total residual). Inner-tier NVLS compresses that residual to ~0.4 μs — the *only* lever that brings hierarchical AR within striking distance of single-pod cost.
2. **AR uniquely compounds α and BW wins, and the BW win is inner-tier-specific.** Endpoint-driven AR pays the $\mathrm{BW_{eff}} = \mathrm{BW}/2$ dual-touch penalty; SHARP reduces in the switch ALU and eliminates the pattern. NVLS-measured intra-pod AR busbw rises from ~360 GB/s to 470+ GB/s (~1.3×) on H100 [NVLINK-SHARP]. The outer tier inherits the same BW lift on its phase, but the inner tier's lift is what matters at large $M$ because the outer payload is already telescoped to $ML/N$. AG / RS / BC / Reduce never had the dual-touch penalty, so SHARP gives them only the α collapse at either tier.
3. **A2A has no INC path on traditional SHARP at either tier.** Rubin-gen NVSwitches and Tomahawk Ultra add dedicated HW A2A as a separate primitive (efficiency win, not the same structural collapse), but on current-shipping NVL72 + Quantum-X800, A2A stays software-scheduled.

**Composing both tiers.** A deployment with NVLS + Quantum SHARP at $L = 32$ pays roughly $0.4 + 1 = 1.4\\,\mu$s of α total + ~9 μs BW (telescoped outer + NVLS-lifted inner) ≈ **~10 μs** end-to-end — essentially as fast as a single-pod NVLS AR despite spanning 32 pods. Outer-only SHARP (without NVLS) gets to ~$7 + 1 = 8\\,\mu$s α + ~13 μs BW ≈ ~21 μs; the NVLS contribution is the difference between "10 μs" and "21 μs" — half the remaining cost. That's why NVLS + Quantum SHARP is the standard combo on production GB200 SuperPODs.

### 3.2 Per-tier η in a hierarchical schedule

The per-tier $\eta$ profile from `05_contention_and_congestion.md` §4.2 feeds directly into §2.1's hierarchical formulas: each phase uses its own tier's $(\eta_\alpha, \eta_\beta)$ when computing realistic cost. For a 2-tier AR with oversubscription $s > 1$ at the outer tier:

$$t_{\mathrm{AR,realistic}}^{\mathrm{2\text{-}tier}} \\;=\\; t_{\mathrm{RS,inner}}^{(\eta_\alpha^\mathrm{inner},\\, \eta_\beta^\mathrm{inner})} \\;+\\; t_{\mathrm{AR,outer}}^{(\eta_\alpha^\mathrm{outer},\\, \min(\eta_\beta^\mathrm{hw},\\, 1/s))} \\;+\\; t_{\mathrm{AG,inner}}^{(\eta_\alpha^\mathrm{inner},\\, \eta_\beta^\mathrm{inner})}$$

The "inner" and "outer" labels here match the tier labels from §2.1's hierarchical schedule (NVSwitch and IB Clos respectively) — *not* the path-specific α subscripts (α_leaf, α_spine) introduced in §1.2 for A2A's per-destination accounting.

**Typical per-tier η values for NVL72 + IB SuperPOD** (from `05_contention_and_congestion.md` §4.2):

| Tier | Role | $\eta_\alpha$ | $\eta_\beta$ at $s = 1$ | $\eta_\beta$ at $s = 2$ | $\eta_\beta$ at $s = 4$ |
|---|---|---|---|---|---|
| **Inner** (NVSwitch within pod) | Intra-pod scale-up | ≈ 1.00 | ≈ 0.80 | n/a (NVSwitch typically not oversubscribed) | n/a |
| **Outer** (IB Quantum-2 / X800 Clos) | Cross-pod scale-out | ≈ 1.10–1.30 | ≈ 0.80 | $\min(0.80, 0.50) = 0.50$ | $\min(0.80, 0.25) = 0.25$ |

The outer-tier $\eta_\beta$ is bounded by $\min(\eta_\beta^{\mathrm{hw}}, 1/s)$: oversubscription pins it at $1/s$ once $s > 1.25$. The outer-tier $\eta_\alpha$ inflates additively from cross-tier queueing under load (1.10–1.30 typical, climbs to 2–3× past 80% utilization). The inner-tier coefficients are deployment-stable because NVSwitch is engineered for full bisection.

**Implication.** Oversubscription at the outer tier makes that tier relatively more expensive for BW-heavy primitives — the outer phase's BW term inflates by $s$ regardless of which collective is running. Any tier-assignment optimizer (e.g., a partition-strategy search over hierarchical cost) must feed $\eta_\beta^{\mathrm{outer}}(s)$ into the formula, not just the ideal $(\alpha, \mathrm{BW})$, or it will miss-rank deployments where oversubscription matters.

**Interaction with SHARP.** SHARP-on-outer (§3.1) collapses the outer-α term but does **not** sidestep the tier-BW cap from $s$ — oversubscription still binds the outer BW side regardless of INC, because the switch ALU can only forward at the tier's aggregate bandwidth. This is why production SHARP-enabled deployments often run $s = 1$ at the SHARP tier even when other tiers are oversubscribed — to keep the BW-side benefit intact.

### 3.3 Pareto implications

Three patterns emerge from combining hierarchical cost with INC and η:

1. **SHARP compresses the "extra tier" penalty almost completely** for AR-heavy workloads. A 3-tier Clos with SHARP at the super-spine runs AR at a cost close to 2-tier without SHARP.
2. **A2A has no INC path on shipping hardware** (`04_in_network_collectives.md §1.1` — no reduction or replication semantics). EP A2A is the primitive whose cost is hardest to compress by going hierarchical — which is why MoE inference workloads are so sensitive to tier topology choice.
3. **Contention η compresses Pareto margins, not rankings.** Under realistic η, the winning hierarchical schedule typically doesn't change — the gap to the runner-up just narrows. This matches the observation in `05_contention_and_congestion.md §5.3` that realistic η tightens every margin without reshuffling winners.

---

## Appendix A: Worked rail-optimized NVL72 SuperPOD topology (L = 32)

§1.1's case study introduced the rail-optimized layout (one rail-ToR per NIC position in a compute tray; each NVL72 striped across 4 rails). This appendix works through the actual switch counts and Clos topology for a production-scale $L = 32$ NVL72 + Quantum-X800 deployment — the canonical "GB200 SuperPOD" anchor referenced from `04_in_network_collectives.md §3`.

The build-up walks bottom-up. **A.1** the smallest physical layers — compute tray and the NVL72 rack (pod). **A.2** the pod ↔ leaf-ToR connection (fan-out and inverse view, rail-optimized). **A.3** the leaf ↔ spine connection — what "full bipartite" actually means in cables, and why the leaf-spine pair count is $8 \times 4$ but the cable count is $8 \times 4 \times 18$. **A.4** puts everything together at the cluster scale (per-rail topology + switch counts + 4-rail independence).

**Cluster parameters.**

- $L = 32$ NVL72 pods, $N_{\mathrm{inner}} = 72$ GPUs per pod → $N = 2304$ GPUs total.
- 4 rails (one per NIC position in a GB200 compute tray; 18 trays per NVL72, so 18 rail-$i$ NICs per pod).
- Quantum-X800 Q3400 leaf and spine switches: 144 ports at 800 Gbps XDR each.
- Full bisection ($s = 1$) within each rail's fat-tree.

### A.1 The compute tray and the NVL72 rack (pod)

This subsection grounds the smallest two physical layers — **(1) compute tray** and **(2) NVL72 pod = 1 rack** — before the rest of the appendix walks up to the cluster scale.

**(1) Compute tray.** The smallest replicable unit is a 1U **GB200 compute tray** with **4 GPUs and 4 NICs** (one NIC per GPU, mounted next to it on the tray). Each GPU has two physically distinct egress paths off the tray:

```
GB200 compute tray (1 of 18 per NVL72 rack — 4 GPUs / 4 NICs):

   to NVSwitch trays in this rack (NVLink, rack-internal — every GPU has its own)
            ▲            ▲            ▲            ▲
            ║            ║            ║            ║
   ┌────────╫────────────╫────────────╫────────────╫─────┐
   │        ║            ║            ║            ║     │
   │      GPU_0        GPU_1        GPU_2        GPU_3   │
   │        │            │            │            │     │   PCIe Gen5
   │      NIC_0        NIC_1        NIC_2        NIC_3   │   (1 NIC/GPU)
   │        │            │            │            │     │
   └────────┼────────────┼────────────┼────────────┼─────┘
            ▼            ▼            ▼            ▼
         (rail 0)     (rail 1)     (rail 2)     (rail 3)
       4 IB cables out of the tray, one per rail, up to the IB ToR layer
```

**Rail rule (cluster-wide).** GPU_$i$'s NIC always lands on rail $i$, and **this rule is identical across every tray (0–17) of every pod in the cluster** (NVL72_0, NVL72_1, …). So slot $i$ across every tray in every pod forms a single **rail-$i$ communicator of $L$ GPUs** (32 GPUs at $L = 32$) — one GPU per pod, all reachable through rail-$i$'s IB fabric without ever leaving the rail. A GPU's rail is set by its tray-slot position, not by which tray or which pod it lives in. (Cross-rail traffic — GPU_$i$ ↔ GPU_$j$ for $i \neq j$ across pods — uses NVLink as a rail bridge; see §1.1's "Cross-pod, cross-rail" bullet.)

**(2) NVL72 pod = 1 rack.** The full NVL72 is one physical rack containing **18 compute trays + 9 NVSwitch trays** plus power/management hardware. The 9 NVSwitch trays form the **NVLink fabric** that connects all 72 GPUs intra-pod — every GPU's NVLink ports cable to the NVSwitch trays, forming a rack-internal switched fabric. **No NVLink cable leaves the rack.** The 72 NIC outputs (4 NICs/tray × 18 trays) all exit upward toward the IB ToR layer:

```
NVL72 pod = 1 rack (72 GPUs / 9 NVSwitch trays / 72 NICs out):

   ╔══════════════════ NVL72 rack ══════════════════╗
   ║                                                ║
   ║   compute tray  0:  [GPU_0|GPU_1|GPU_2|GPU_3]  ║
   ║   compute tray  1:  [GPU_0|GPU_1|GPU_2|GPU_3]  ║
   ║      ...     (more compute trays)              ║
   ║                                                ║
   ║   ──────── 9 NVSwitch trays ────────           ║
   ║       (NVLink fabric for all 72 GPUs;          ║
   ║        rack-internal, no cable exits)          ║
   ║   ──────────────────────────────               ║
   ║                                                ║
   ║      ...     (more compute trays)              ║
   ║   compute tray 17:  [GPU_0|GPU_1|GPU_2|GPU_3]  ║
   ║                                                ║
   ╚═════════════════════ ║║║║ ═════════════════════╝
                          ▼▼▼▼  72 IB-bound NIC cables exit upward
                                (4 NICs/tray × 18 trays)
                                → IB ToR(s) on top of rack(s)
```

The two physical fabrics (NVLink scale-up + IB scale-out) are the two egress paths from §1.1's "Two independent fabrics per GPU" — what changes here is the *granularity*: the rail bullet labels in (1) tell you which IB rail each GPU's NIC lands on. **Every NVL72 rack contributes 18 NICs (one per tray's slot $i$) to each of the 4 rails.**

A.2 walks the next layer up — how each rack's 72 NICs reach IB ToRs (pod → leaf). A.3 then covers leaf ↔ spine. A.4 puts everything together at full $L = 32$ cluster scale.

### A.2 Pod ↔ leaf-ToR connection (rail-optimized)

Two complementary views capture how each NVL72's 72 NICs reach IB leaf-ToRs: **fan-out** (one pod → 4 rail-ToRs) and **aggregate / inverse** (one rail-ToR ← up to 4 pods). Both are introduced in §1.1's Layout 2; this subsection rephrases them in the appendix's $L = 32$ context.

**View 1 — Fan-out (one pod → 4 rail-ToRs).** NICs at slot $i$ across every tray (one per tray × 18 trays = 18 NICs) bundle together and exit the rack to the rail-$i$ leaf-ToR; the same pattern at slots 1, 2, 3 fills the other 3 rail-ToRs. **Each pod is striped across 4 rail-ToRs, contributing 18 NICs to each.**

```
Pod-side fan-out (one NVL72 → 4 rail-ToRs):

      ┌────────────── NVL72_0 (72 GPUs / 72 NICs) ───────────────┐
      │                                                          │
      │     tray  0:   [r0]     [r1]     [r2]     [r3]           │
      │     tray  1:   [r0]     [r1]     [r2]     [r3]           │
      │       ...                                                │
      │     tray 17:   [r0]     [r1]     [r2]     [r3]           │
      │                 │        │        │        │             │
      └─────────────────┼────────┼────────┼────────┼─────────────┘
                        │        │        │        │
                        ▼        ▼        ▼        ▼
                     ┌──┬───┐ ┌──┬───┐ ┌──┬───┐ ┌──┬───┐
                     │rail-0│ │rail-1│ │rail-2│ │rail-3│   ← 4 distinct (separate) leaf-ToRs
                     │Leaf 0│ │Leaf 0│ │Leaf 0│ │Leaf 0│     that this pod connects to (rail-i
                     │ 144p │ │ 144p │ │ 144p │ │ 144p │     Leaf 0 for i = 0, 1, 2, 3). At L
                     └──────┘ └──────┘ └──────┘ └──────┘     = 32 each rail has 8 such leaves
                                                             (A.4); this pod hits 1 per rail
                                                             — its leaf-group's rail-i Leaf 0.
```

**View 2 — Aggregate / inverse (one rail-ToR ← up to 4 pods).** A 144-port X800 Q3400 leaf-ToR at $s = 1$ has 72 down-ports for NICs. Since each pod contributes only 18 NICs to a given rail, **one leaf-ToR can absorb the rail-$i$ slice of up to 4 NVL72 pods** ($72 / 18 = 4$). At $L = 32$, this rule replicates **8 times per rail**: pods are partitioned into 8 leaf-groups of 4 pods each, and each leaf-group attaches to one rail-leaf-ToR per rail (pods 0–3 → rail-$i$ Leaf 0; pods 4–7 → rail-$i$ Leaf 1; …; pods 28–31 → rail-$i$ Leaf 7).

```
ToR-side aggregate (one rail-i leaf-ToR ← 4 NVL72 pods — repeated 8× per rail):

      ┌─NVL72_0─┐  ┌─NVL72_1─┐  ┌─NVL72_2─┐  ┌─NVL72_3─┐
      │ rail-i  │  │ rail-i  │  │ rail-i  │  │ rail-i  │     ← only the rail-i slice
      │ 18 NICs │  │ 18 NICs │  │ 18 NICs │  │ 18 NICs │       of each pod is shown;
      └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       the other 54 NICs/pod
           │            │            │            │             land on rails j ≠ i
           ▼            ▼            ▼            ▼            (4 × 18 = 72 NICs total
      ┌────┬────────────┬────────────┬────────────┬────┐       into one rail-i leaf)
      │       rail-i Leaf 0 (X800 Q3400, 144 ports)    │
      │  72 down (= 4 × 18 NICs) + 72 up (→ 4 spines)  │
      └────────────────────────┬───────────────────────┘
                               │
                               ▼  72 up-ports — leaf-to-spine fan-out covered in A.3
```

The two views are reciprocal: the fan-out shows where one pod's 4 NIC bundles land; the aggregate shows what one leaf-ToR collects. The 18 NICs each pod contributes to a rail-$i$ leaf-ToR are precisely that pod's **slot-$i$ NICs** (NIC_$i$ across trays 0–17) — A.1's rail rule, applied 4 times. So `rail-i Leaf 0` is aggregating *same-slot peers across 4 different pods*: 18 GPU_$i$ from pod 0, 18 from pod 1, 18 from pod 2, 18 from pod 3 — 72 same-slot GPUs total. Across the cluster:

- **8 leaf-groups per rail** × **4 rails** = 32 leaf-groups cluster-wide; each leaf-group is one (rail, 4-pod) tuple wired to one leaf-ToR. Scaling beyond $L = 32$ adds more leaf-groups per rail (and more spines accordingly — see A.4).
- **One pod participates in 4 leaf-groups simultaneously**, one per rail. Pod 0's rail-0 NICs land on rail-0 Leaf 0; its rail-1 NICs land on rail-1 Leaf 0; etc. The pod's failure domain spans 4 leaves rather than 1 — a small intentional blast-radius decoupling vs the §1.1 Layout 1 ("pod-local ToR") alternative.

### A.3 Up-side micro-architecture: rail-leaf-ToRs → rail-spines

Each leaf-ToR's 72 up-ports reach the spine layer. With **4 spines per rail** (derived in A.4) and full bisection ($s = 1$), the 72 up-ports split into 4 groups of $72 / 4 = 18$. The key fact, easy to miss: **every leaf-spine pair is connected by 18 parallel cables, not one.** This is the "fattening" mechanism of the k-ary fat-tree (Appendix B): aggregate cross-tier BW grows toward the root through path multiplicity, not through link width.

**Aside — is there a rule like A.2's tray-slot rule for the 18-cable groups?** No. A.2's pod-to-leaf wiring has a structural rule rooted in hardware: the 18 NICs from a pod that go to rail-$i$ are exactly *slot $i$ across all 18 trays* (GPU_$i$ → rail-$i$). At the leaf-spine layer, the 4 spines are *equivalent* destinations within the same rail's fabric — nothing about a packet picks one over another, and ECMP (5-tuple hashing) distributes flows across all 4 at runtime. Any partition of the leaf's 72 up-ports into 4 groups of 18 satisfies the k-ary fat-tree property; production cabling typically uses contiguous port blocks (ports 0–17 → spine 0, 18–35 → spine 1, etc.), often aligned with switch ASIC dies for lower same-die latency, but this is install convention, not a workload constraint. (A 3-tier fat-tree adds a real rule: each spine indexed $j$ connects only to "plane $j$" of super-spines per the Al-Fares construction — see Appendix B.2 — but that's beyond the 2-tier per-rail scope here.)

**Per-leaf fan-out** (one leaf, 72 up-ports to 4 spines):

```
            ┌──────── rail-i Leaf 0 (72 up-ports) ────────┐
            └────┬───────────┬───────────┬───────────┬────┘
                 │           │           │           │
              18 │        18 │        18 │        18 │  (parallel cables each)
                 │           │           │           │
                 ▼           ▼           ▼           ▼
            ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
            │ Spine 0 │ │ Spine 1 │ │ Spine 2 │ │ Spine 3 │
            └─────────┘ └─────────┘ └─────────┘ └─────────┘

  Same fan-out shape repeats for every one of the rail's 8 leaves.
```

**Inverse view — what one spine collects.** A rail's spine has 144 down-ports. With 8 leaves **per rail** each contributing 18 parallel cables, the spine's down-ports are exactly filled $8 \times 18 = 144$:

```
       Leaf 0    Leaf 1    Leaf 2    Leaf 3    Leaf 4    Leaf 5    Leaf 6    Leaf 7
       ┌────┐    ┌────┐    ┌────┐    ┌────┐    ┌────┐    ┌────┐    ┌────┐    ┌────┐
       │ 18 │    │ 18 │    │ 18 │    │ 18 │    │ 18 │    │ 18 │    │ 18 │    │ 18 │
       └─┬──┘    └─┬──┘    └─┬──┘    └─┬──┘    └─┬──┘    └─┬──┘    └─┬──┘    └─┬──┘
         │         │         │         │         │         │         │         │
         ▼         ▼         ▼         ▼         ▼         ▼         ▼         ▼
       ┌─────────────────────────────────────────────────────────────────────────┐
       │  Spine j (X800 Q3400, 144 ports — all 144 used as down-ports)           │
       │  No up-ports: in a 2-tier Clos, the spine is the top of the fabric.     │
       └─────────────────────────────────────────────────────────────────────────┘
```

**All 144 spine ports are downlinks — that's what makes the 4 fat-trees "independent."** Each spine's full radix is consumed serving its 8 rail-local leaves ($8 \times 18 = 144$); **no port is left over for an inter-rail uplink.** With no spine on rail $i$ wired to anything on rail $j \neq i$, the 4 rails are 4 physically disjoint IB fabrics by *port allocation*, not just by cabling convention — the spine's port budget itself forecloses any cross-rail connection. (A 3-tier deployment would split each spine's 144 ports as $72$ down + $72$ up to a *per-rail* super-spine — still rail-separated, since cross-rail traffic continues through the NVLink rail bridge described in §1.1's "Cross-pod, cross-rail" bullet.)

### A.4 Putting it together: cluster overview and switch counts

Combining A.1 (tray + rack), A.2 (pod ↔ leaf), and A.3 (leaf ↔ spine) gives the full $L = 32$ cluster: **4 independent rail fat-trees side by side**, with the same 32 pods wired into all 4 rails via the rail-stripe fan-out (A.2 View 1). The macro view first, then a per-rail detail.

```
Cluster macro view — 4 rails + 32 shared pods:

      ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐
      │  rail 0 │      │  rail 1 │      │  rail 2 │      │  rail 3 │
      │ 4 spines│      │ 4 spines│      │ 4 spines│      │ 4 spines│
      │ 8 leaves│      │ 8 leaves│      │ 8 leaves│      │ 8 leaves│
      └────┬────┘      └────┬────┘      └────┬────┘      └────┬────┘
           │                │                │                │
           ▼                ▼                ▼                ▼
      ┌──────────────────── 32 NVL72 pods ─────────────────────────┐
      │  Each pod's 72 NICs split 18+18+18+18 across the 4 rails.  │
      │  Same 32 pods are shared by all 4 rails (no replication).  │
      └────────────────────────────────────────────────────────────┘

   Switches in different rails are physically distinct: e.g., the cluster has
   4 separate X800s named "Spine 0" — one per rail; same for "Leaf 0..7".
   No IB cable crosses rails (A.3). Cluster totals: 16 spines + 32 leaves.
```

**Full cluster topology** — each Spine/Leaf box below stands for **4 distinct X800 switches** (one per rail), shown via the `×4` row inside each box. Pods at the bottom are shared across all 4 rails.

```
NVL72 SuperPOD full cluster topology at L = 32 — 2304 GPUs across 4 rails

                  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
    Spine layer:  │Spine │  │Spine │  │Spine │  │Spine │   4 spines × 4 rails =
                  │  0   │  │  1   │  │  2   │  │  3   │   16 X800 spines total
                  │dn:144│  │dn:144│  │dn:144│  │dn:144│   (144 ports each, all
                  │up:  0│  │up:  0│  │up:  0│  │up:  0│    consumed as downlinks
                  │  ×4  │  │  ×4  │  │  ×4  │  │  ×4  │    — no super-spine in
                  └─┬────┘  └─┬────┘  └─┬────┘  └─┬────┘     a 2-tier per rail)
                    │         │         │         │
                    full bipartite within each rail: every leaf ↔ every
                    spine, 18 parallel cables per pair at s = 1.
                    No IB cable crosses rails (A.3 callout above).
                    │         │         │         │
                  ┌─┴────┐  ┌─┴────┐  ┌─┴────┐  ┌─┴────┐         ┌─┴────┐
    Leaf layer:   │Leaf 0│  │Leaf 1│  │Leaf 2│  │Leaf 3│   ...   │Leaf 7│   8 leaves × 4 rails
                  │ ToR  │  │ ToR  │  │ ToR  │  │ ToR  │         │ ToR  │   = 32 X800 leaves total
                  │dn: 72│  │dn: 72│  │dn: 72│  │dn: 72│         │dn: 72│   (dn = 4 pods × 18 NICs;
                  │up: 72│  │up: 72│  │up: 72│  │up: 72│         │up: 72│    up = 4 spines × 18
                  │  ×4  │  │  ×4  │  │  ×4  │  │  ×4  │         │  ×4  │    parallel cables)
                  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘         └──┬───┘
                     │         │         │         │                │
                     │ each leaf collects 72 NICs (= 18 from each of 4 pods)
                     ▼         ▼         ▼         ▼                ▼
                  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐         ┌──────┐
                  │ pods │  │ pods │  │ pods │  │ pods │   ...   │ pods │   pods SHARED across
                  │ 0-3  │  │ 4-7  │  │ 8-11 │  │12-15 │         │28-31 │   all 4 rails (no ×4):
                  └──────┘  └──────┘  └──────┘  └──────┘         └──────┘   each pod's 72 NICs
                                                                              already split 18+18+18+18
                                                                              across the 4 rails
```

**Switch-count derivation.** **Per rail** at $L = 32$:

- NICs to absorb **per rail**: $32\\,\text{pods} \times 18\\,\text{NICs/pod} = 576\\,\text{NICs}$.
- Each X800 leaf at $s = 1$: $72$ down-ports (NICs) + $72$ up-ports (spines) = $144$ → **leaves per rail = $576 / 72 = 8$** (matches A.2's "8 leaf-groups per rail").
- Each leaf has $72$ up-ports, distributed across $S$ spines as $72/S$ parallel links per leaf-spine pair. For non-blocking ($s = 1$) at the spine tier, $S$ must absorb all $8 \times 72 = 576$ leaf-up-port total; with 144-port spines: $S = 576 / 144 = 4$.
- **Spines per rail = 4**, each with $8\\,\text{leaves} \times 18 = 144$ down-ports fully utilized (matches A.3's column-sum).

**Cluster totals.** Switches multiply by 4 across rails; pods do not.

| | **Per rail** | Cluster (× 4 rails) |
|---|---|---|
| Leaf switches | 8 | 32 |
| Spine switches | 4 | 16 |
| Total X800 switches | 12 | **48** |
| **Pods** (shared across rails) | — | **32** (not multiplied) |

This is the topology referenced as the "production GB200 SuperPOD" anchor in `03_hierarchical_topologies.md §1.1`'s case study, `03_hierarchical_topologies.md §1.2`'s symbol catalog (where $\alpha_{\mathrm{outer}} = \alpha_{\mathrm{spine}} = 8\\,\mu$s applies because pods span 8 leaves per rail), and in `04_in_network_collectives.md §3`'s discussion of why the imaginary 512-port single-switch is hypothetical.

---

## Appendix B: The k-ary fat-tree

§1 treats "fat-tree" and "Clos" as the general multi-tier switched family, parameterized by the oversubscription ratio $s$. In practice, virtually every AI datacenter deploys a specific realization: the **k-ary fat-tree** of [AL-FARES08]. This appendix gives an explicit diagram and port-by-port accounting for a small $k = 4$ instance, contrasts it with the original Leiserson (1985) fat-tree concept, and explains why the k-ary variant dominates deployment.

### B.1 Traditional (Leiserson) fat-tree

Leiserson [LEIS85] proposed the fat-tree as a universal routing network for hardware-efficient supercomputing. The defining property: **link bandwidth grows toward the root**. Unlike a plain binary tree (where every link has the same BW and the root becomes a bottleneck), a fat-tree has "fat" branches at the top with BW scaled to match the aggregate of everything below.

```
Leiserson fat-tree (conceptual, BW grows toward root):

                              ┌─────┐
                              │  R  │            root switch
                              └──┬──┘
                                 │  ═══ 4× link bandwidth
                       ┌─────────┴─────────┐
                       │                   │
                    ┌──┴──┐             ┌──┴──┐
                    │ I1  │             │ I2  │   intermediate tier
                    └─┬─┬─┘             └─┬─┬─┘
                      │ │  ═══ 2× link    │ │
                  ┌───┘ └───┐         ┌───┘ └───┐
                  │         │         │         │
               ┌──┴─┐    ┌──┴─┐    ┌──┴─┐    ┌──┴─┐
               │ E1 │    │ E2 │    │ E3 │    │ E4 │  edge tier
               └──┬─┘    └──┬─┘    └──┬─┘    └──┬─┘
                  │  1×     │         │         │
                  ●         ●         ●         ●    endpoints

   BW scaling rule: link BW doubles at each level going up.
   Aggregate BW at every tier stays constant; the root never bottlenecks.
```

**Properties:**

- Switches at different tiers can have **different port counts and link speeds**.
- BW scaling rule is **abstract** — factor-of-2 per level, factor-of-k, or any monotone increase.
- Mostly an academic / theoretical construction — not practical to build from commodity parts.

**Practical limitation:** requires custom (non-uniform) switch hardware. You can't go to a vendor catalog and buy a "root" switch with 4× the BW of an "intermediate" switch — that's not how the commodity-switch market works.

### B.2 k-ary fat-tree (Al-Fares construction)

Al-Fares et al. [AL-FARES08] realized fat-tree properties using **only commodity $k$-port switches** — every switch in the entire fabric is identical. BW growth toward the root is achieved not by fattening individual links but by **multiplying parallel paths** at upper tiers.

**Parameter**: a single even integer $k$ (the switch radix) determines everything.

| Quantity | Formula | $k = 4$ | $k = 48$ | $k = 64$ |
|---|---|---|---|---|
| Switch radix (ports/switch) | $k$ | 4 | 48 | 64 |
| Number of pods | $k$ | 4 | 48 | 64 |
| Edge switches per pod | $k/2$ | 2 | 24 | 32 |
| Aggregation switches per pod | $k/2$ | 2 | 24 | 32 |
| Core switches | $(k/2)^2$ | 4 | 576 | 1,024 |
| Endpoints per edge switch | $k/2$ | 2 | 24 | 32 |
| Endpoints per pod | $(k/2)^2$ | 4 | 576 | 1,024 |
| **Total endpoints** | $k^3/4$ | **16** | **27,648** | **65,536** |
| Total switches | $5k^2/4$ | 20 | 2,880 | 5,120 |

**Uniform port split per switch** (every switch — edge, aggregation, core — has $k$ ports split evenly):

```
  ┌───────────────┐       k/2 "up" ports     (toward higher tier)
  │               │  ←──────────────────────
  │    SWITCH     │
  │   (k ports)   │  ←──────────────────────
  └───────────────┘       k/2 "down" ports   (toward lower tier)

  This uniform k/2-up / k/2-down split enforces s = 1 at every tier:
  the switch cannot receive more traffic downward than it can push upward.
```

**Full diagram: $k = 4$ fat-tree (20 switches, 16 endpoints)**

```
  CORE tier: (k/2)² = 4 switches; each has k = 4 ports (all downward, one per pod)

                 ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
                 │ C(1,1) │ │ C(1,2) │ │ C(2,1) │ │ C(2,2) │      4 core switches
                 └──┬┬┬┬──┘ └──┬┬┬┬──┘ └──┬┬┬┬──┘ └──┬┬┬┬──┘      each with 4 ports
                    ││││       ││││       ││││       ││││         (one per pod)
                    ││││  16 core↔aggregation links total (each core has one link to
                    ││││  each pod's aggregation tier — specifically, core C(i,j)
                    ││││  connects to aggregation switch indexed j in each pod)
                    ▼▼▼▼       ▼▼▼▼       ▼▼▼▼       ▼▼▼▼

   POD 0                 POD 1                 POD 2                 POD 3

  AGG tier (k/2 = 2 per pod; k/2 up-ports to core, k/2 down-ports to edge)

   ┌────┐ ┌────┐         ┌────┐ ┌────┐         ┌────┐ ┌────┐         ┌────┐ ┌────┐
   │ A₀₀│ │ A₀₁│         │ A₁₀│ │ A₁₁│         │ A₂₀│ │ A₂₁│         │ A₃₀│ │ A₃₁│
   └─┬┬─┘ └─┬┬─┘         └─┬┬─┘ └─┬┬─┘         └─┬┬─┘ └─┬┬─┘         └─┬┬─┘ └─┬┬─┘
     ⇅──────⇅               ⇅─────⇅               ⇅─────⇅               ⇅─────⇅
     (within-pod full bipartite: each agg ↔ each edge, 4 links per pod, 16 total)

  EDGE tier (k/2 = 2 per pod; k/2 up-ports to agg, k/2 down-ports to endpoints)

   ┌────┐ ┌────┐         ┌────┐ ┌────┐         ┌────┐ ┌────┐         ┌────┐ ┌────┐
   │ E₀₀│ │ E₀₁│         │ E₁₀│ │ E₁₁│         │ E₂₀│ │ E₂₁│         │ E₃₀│ │ E₃₁│
   └─┬┬─┘ └─┬┬─┘         └─┬┬─┘ └─┬┬─┘         └─┬┬─┘ └─┬┬─┘         └─┬┬─┘ └─┬┬─┘
     ││     ││             ││     ││             ││     ││             ││     ││

  ENDPOINTS: 2 per edge switch × 2 edges per pod × 4 pods = 16 total

     R₀R₁  R₂R₃          R₄R₅   R₆R₇          R₈R₉   R₁₀R₁₁       R₁₂R₁₃  R₁₄R₁₅
```

**Per-switch port accounting** (every switch has $k = 4$ ports; split 2 up, 2 down):

| Switch type | 2 "up" ports connect to | 2 "down" ports connect to |
|---|---|---|
| Edge $E_{p,i}$ (pod $p$, edge $i$) | Both aggregation switches in pod $p$ | 2 endpoints |
| Aggregation $A_{p,j}$ (pod $p$, agg $j$) | 2 of the 4 core switches — specifically cores $C(i, j)$ for $i \in \\{1, 2\\}$ | Both edge switches in pod $p$ |
| Core $C(i, j)$ | N/A (top tier) | 1 aggregation switch per pod — $k$ = 4 ports, one per pod, connecting to agg $j$ of each pod |

**Wiring rule (where the magic happens)**: Core switches are indexed by a pair $(i, j)$ with $i, j \in \\{1, \ldots, k/2\\}$. Core $C(i, j)$'s $p$-th port connects to aggregation switch $j$ in pod $p$. This partitions the $(k/2)^2$ cores into $k/2$ parallel **planes** — each plane consists of $k/2$ cores all talking to the same aggregation index across every pod. Different planes (different $j$) carry independent cross-pod traffic, enabling ECMP load balancing.

### B.3 Comparison: Leiserson vs Al-Fares

| Property | Leiserson (1985) | k-ary (Al-Fares 2008) |
|---|---|---|
| Switch uniformity | Not required — switches can differ per tier | Required — every switch has exactly $k$ ports |
| How "fat" is achieved | Wider / higher-BW links at upper tiers | More parallel paths at upper tiers (same link BW everywhere) |
| Parameterization | Abstract BW scaling rule, per tier | Single integer $k$ |
| Construction | Theoretical / conceptual | Concrete recipe buildable from catalog parts |
| Oversubscription | Not specified | $s = 1$ at every tier by construction |
| Commodity hardware | No | Yes — identical $k$-port switches |
| Is it a Clos? | Not necessarily | Yes, rearrangeably non-blocking |
| Where you find it | Academic / custom supercomputers | Ubiquitous in AI datacenters |

The two share the "fat-tree" name but achieve the fattening through different mechanisms: Leiserson by link width, Al-Fares by parallel path count. The Al-Fares version is what "fat-tree" means in datacenter practice.

### B.4 Why k-ary dominates AI datacenters

- **Commodity economics**: single switch SKU throughout the fabric → single vendor negotiation, single firmware stream, interchangeable spares. One 48-port switch model can build fabrics from 48 endpoints (small pilot) to 27,648 endpoints (full $k^3/4$ deployment).
- **Cabling predictability**: every switch has the same $k/2$-up / $k/2$-down port assignment, so cabling templates repeat verbatim per pod.
- **Full bisection by construction**: $s = 1$ at every tier means no cross-tier BW bottleneck under permutation traffic — any pairing of endpoints can run at full link rate.
- **ECMP-native**: $(k/2)^2$ core switches give $(k/2)^2$ equivalent paths between any two pods; standard ECMP hashing spreads flows evenly without fabric-specific logic.
- **Single-knob scaling**: need more endpoints? Pick a larger $k$ (different commodity switch), rebuild. Don't need to redesign the topology itself — the recipe is the same at any $k$.

**Production examples:**

| Fabric | Switch radix $k$ | Total endpoint capacity | Deployments |
|---|---|---|---|
| NVIDIA SuperPOD (Quantum-2 IB NDR) | 40 (HCA) / 64 (switch) | ~16–32K GPUs per pod | DGX SuperPOD, Meta RSC |
| NVIDIA Quantum-X800 | 144 | >500K endpoints per fabric | Blackwell-class GB200 deployments |
| Broadcom Tomahawk 5 (Ethernet) | 256–512 | Millions via 3-tier | Hyperscaler AI clusters |
| Arista 7060X / 7800 (Ethernet) | 32–128 | Varies | Meta, Microsoft, Google |

---

## Further reading

- **`01_collective_algorithms.md`** — per-primitive cost formulas (ring, DBT, RHD, Rabenseifner, pairwise); the building blocks composed here.
- **`02_topology_mapping.md`** — single-tier scale-up costs for star, torus, mesh; §5.1 has the per-topology cost-formula summary that §2.2 here builds on.
- **`04_in_network_collectives.md`** — SHARP / NVLS / Quantum SHARP / Spectrum-X SHARP mechanics; §3 above relies on the scale-out INC cost model from §2.2 there.
- **`05_contention_and_congestion.md`** — per-tier $\eta$ calibration (§5.1); feeds §3.2 here.
- **`references.md`** — primary sources on Megatron-LM parallelism (Shoeybi et al.), hierarchical collective scheduling (Thakur & Gropp, MagPie), and the Leiserson (1985) fat-tree foundational paper.
