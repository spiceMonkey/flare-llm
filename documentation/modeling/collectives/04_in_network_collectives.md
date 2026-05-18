# In-Network Collectives: SHARP, NVLS, and the $N_{\mathrm{hops}}$ Collapse

**Author:** Yue Lu  
**Date:** April 2026  

On switched fabrics — single-switch star (e.g., NVSwitch) or multi-tier star-of-stars (fat-tree / Clos) — conventional collectives pay an endpoint-driven latency term that grows with group size: $2(N-1)\alpha$ for ring AR, $2\lceil \log_2 N \rceil\alpha$ for the best software tree (DBT / RHD). Every algorithmic step is an endpoint round trip through the switch, and the $\alpha$ floor ($\sim 1\\,\mu$s) is set by endpoint software — scheduling, kernel launch, NIC engine setup — paid once per step. For latency-sensitive traffic (small $M$, interactive collectives, fine-grained reductions), this $n_\alpha \cdot \alpha$ term dominates total cost, and no software schedule can eliminate it because the $\alpha$ floor is outside the algorithm's control.

**In-network collectives (INC)** attack this term at its source by moving the reduction into the switch ASIC itself. Switch-resident ALUs reduce flits as they arrive and multicast the result back; endpoints see the reduced output in a single logical round trip, and the endpoint software overhead is paid **once** rather than $O(N)$ or $O(\log N)$ times. On a single-switch star, $n_\alpha$ collapses from $2(N-1)$ or $2\lceil \log_2 N \rceil$ down to $2$, independent of $N$. On a multi-tier switched fabric, the aggregation tree is built from switches rather than endpoints, so $n_\alpha = O(\log_r N)$ but at switch cut-through latency ($\sim$ 200-400 ns) rather than endpoint RTT ($\sim 1-3\\,\mu$s) — a 5-10× per-hop saving on top of the structural collapse.

The speedup is **scoped to switched fabrics**: star, fat-tree / Clos. Torus-native collectives don't benefit — their $N$-dependent $\alpha$ term comes from neighbor-router hops rather than endpoint-driven switch round trips, and there is no switch-hosted ALU in the reduce path. This note focuses on the switched-fabric case: what the hardware does, how the $O(N) \to O(1)$ latency collapse manifests at $N = 512$ on a hypothetical single-switch star (chosen for consistency with the running example in `02_topology_mapping.md §5.1` and `05_contention_and_congestion.md §5`), and where the mechanism breaks down (reducible types, precision, cross-domain fallback). Three shipping implementations anchor the discussion — NVLS (NVLink SHARP on NVSwitch, single-switch star), Quantum SHARP (InfiniBand, multi-tier), and Tomahawk Ultra INC (Ethernet, emerging).

# Table of Contents

1. [How INC speeds up collectives](#1-how-inc-speeds-up-collectives)
   - 1.1 [In-network reduction (switch ALU)](#11-in-network-reduction-switch-alu)
   - 1.2 [Switch multicast (crossbar fanout)](#12-switch-multicast-crossbar-fanout)
   - 1.3 [Crossbar scatter-gather — HW A2A (emerging)](#13-crossbar-scatter-gather--hw-a2a-emerging)
   - 1.4 [Cost-model savings — consolidated summary](#14-cost-model-savings--consolidated-summary)
2. [Scale-up vs scale-out](#2-scale-up-vs-scale-out)
   - 2.1 [Scale-up: single-switch star](#21-scale-up-single-switch-star)
   - 2.2 [Scale-out: multi-tier aggregation tree](#22-scale-out-multi-tier-aggregation-tree)
3. [Worked example at $N = 512$](#3-worked-example-at-n--512)
   - 3.1 [The full $N = 512$ ladder (AR)](#31-the-full-n--512-ladder-ar)
   - 3.2 [Regime sensitivity (AR)](#32-regime-sensitivity-ar)
   - 3.3 [Cross-primitive comparison (non-AR primitives)](#33-cross-primitive-comparison-non-ar-primitives)
4. [INC-star vs torus](#4-inc-star-vs-torus)
5. [Further reading](#further-reading)

---

## 1. How INC speeds up collectives

Conventional software collectives treat the switch as a passive forwarder: each algorithmic step is an endpoint-to-endpoint message routed through the switch's crossbar without modification. Every step pays one endpoint-software $\alpha$ (NCCL scheduling + CUDA launch + NIC engine setup, $\sim 1\\,\mu$s) plus switch cut-through ($\sim 200$ ns) plus negligible wire propagation. For AR on $N$ endpoints: ring needs $2(N-1)$ such steps, DBT / RHD needs $2\lceil \log_2 N \rceil$ (§5 of `01_collective_algorithms.md`). The $\alpha$ floor is software-set and cannot be reduced by any choice of algorithm.

```
Software ring AR on a star — every arrow is an endpoint → switch → endpoint round trip,
with one α worth of endpoint software overhead paid at each step

  R0 → switch → R1 → switch → R2 → switch → ... → R(N-1) → switch → R0
  [N-1 RS steps]                                 then [N-1 AG steps]
```

INC replaces the passive-forwarder assumption with an active-switch one. **Three structurally distinct hardware capabilities** have shipped or are emerging on production switch ASICs, each accelerating a different set of collectives:

- **§1.1 — In-network reduction (switch ALU).** *Many in, one out* semantics. Matches **Reduce**, **RS**, and the reduce-phase of **AR**.
- **§1.2 — Switch multicast (crossbar fanout).** *One in, many out* semantics. Matches **BC**, **AG**, and the multicast-phase of **AR**.
- **§1.3 — Crossbar scatter-gather (HW A2A, emerging).** *Many in, many out via per-destination routing* semantics. Matches **A2A** only.

§1.1 and §1.2 are the **SHARP-class** INC family (NVLink SHARP / NVLS, Quantum SHARP, Spectrum-X SHARP, Tomahawk Ultra) — they collapse $O(N) \to O(1)$ on $\alpha$ and (for AR alone) lift the BW ceiling. §1.3 is a structurally different primitive (Broadcom Tomahawk Ultra today, Rubin-generation NVSwitches next), giving an efficiency win on A2A's $\alpha$ but no structural data collapse. **§1.4 consolidates** the per-primitive α-term and BW-term savings into a single summary table.

### 1.1 In-network reduction (switch ALU)

**Mechanism.** The switch ASIC has ALUs in its crossbar fabric alongside the port-routing logic. They consume $N$ incoming $M$-byte flits from participating ports, compute sum / max / min / bit-op elementwise at line rate, and emit **one** $M$-byte reduced output. The semantics — *many in, one out* — match **Reduce**, **RS**, and the reduce-phase of **AR**. They do *not* match AG, BC, or A2A (no reduction in their semantics).

**How it accelerates each affected primitive.**

```
Reduce — one root R* receives the sum of all N values
─────────────────────────────────────────────────────────

  Software tree-Reduce:
    Level 0: pairs of endpoints reduce (N/2 pairs in parallel; one α per level)
    Level 1: pairs of pair-results reduce (N/4 pairs)
    ...
    Total: ⌈log₂ N⌉ levels of endpoint round trips

  INC Reduce:
    R0     ─push M─►┐
    R1     ─push M─►├─► [Switch ALU: elementwise sum] ─push M─► root R*
    R2     ─push M─►│
     ...            │
    R(N-1) ─push M─►┘
    Total: ~1 logical round trip — one push per rank, one receive at root
```

```
RS — every rank receives a distinct M/N-byte slice of the reduced result
────────────────────────────────────────────────────────────────────────

  Software ring RS:
    N-1 sequential ring steps, each forwarding (N-1)/N · M bytes

  INC RS:                                          
    R0     ─push M─►┐                          ┌─slice 0 (M/N) ─► R0
    R1     ─push M─►├─► [Switch ALU: sum] ─►   ├─slice 1 (M/N) ─► R1
    R2     ─push M─►│   then scatter slices    ├─slice 2 (M/N) ─► R2
        ...         │                          |         ...  
    R(N-1) ─push M─►┘                          └─slice N-1 (M/N) ─► R(N-1)
    Total: ~2 logical round trips — push, in-fabric ALU + scatter, receive
```

**AR's reduce phase** uses the same switch-ALU mechanism as Reduce; the M-byte reduced result feeds straight into the multicast engine (§1.2) for AR's broadcast phase, instead of being delivered to a single root.

**Improvement vs software.** The α and BW effects on these primitives:

- **α-term**: from $O(N)$ endpoint round trips (ring) or $O(\log N)$ levels (tree) down to $\sim 2$ logical round trips through the switch, regardless of $N$. Speedup $\sim (N-1)/2$ vs ring, $\sim \log_2 N$ vs tree.
- **BW-term**: stays at the same per-rank ceiling as software ($M/\mathrm{BW}$ for Reduce; $(N-1)/N \cdot M/\mathrm{BW}$ for RS) — software ring already saturates the link via full-duplex forwarding. INC reduction is α-only on these two primitives.
- **AR is special**: composes the switch ALU + multicast xbar and additionally lifts $\mathrm{BW_{eff}}$ from $\mathrm{BW}/2$ to $\mathrm{BW}$ — see §1.4.

**Commercial shipment** (switch-ALU support):

- **NVLS** (NVLink SHARP) — NVSwitch Gen3 (H100 era) and Gen4 (B200 / GB200 / NVL72). $\alpha_{\mathrm{switch}} \approx 100$–$200$ ns. Capped at NVSwitch radix (72 GPUs in NVL72).
- **Quantum SHARP** — IB Quantum-2 / Quantum-X800 switches.
- **Spectrum-X SHARP** — NVIDIA Spectrum-X Ethernet switches.
- **Tomahawk Ultra** (Broadcom, shipped 2025) — first commodity-Ethernet INC ASIC, $\alpha_{\mathrm{switch}} \approx 250$ ns.

### 1.2 Switch multicast (crossbar fanout)

**Mechanism.** The switch has a multicast routing table. A single $M$-byte write to a multicast group ID is replicated by the crossbar to every destination port in the group in **one switch-local operation**, without traversing the aggregate ingress bandwidth $N$ times. The semantics — *one in, many out* — match **BC**, **AG** (which decomposes as $N$ concurrent broadcasts, one per rank's slice), and the broadcast-phase of **AR**. They do *not* match RS, Reduce, or A2A.

**How it accelerates each affected primitive.**

```
BC — root R0 sends M bytes to all other ranks
─────────────────────────────────────────────

  Software pipelined-tree BC:
    Level 0: R0 sends M to one peer (one α)
    Level 1: both forward M to one peer each (one α, parallel)
    ...
    Total: ⌈log₂ N⌉ α-rounds

  INC BC:
                              ┌─► R1
    R0 ─push M─► [Switch     ─┤─► R2
                  multicast:  ├─► ...
                  fanout]     └─► R(N-1)
    Total: ~1 logical round trip — single multicast write, parallel fanout
```

```
AG — every rank pushes its M/N slice; every rank receives all N slices
──────────────────────────────────────────────────────────────────────

  Software ring AG:
    N-1 sequential ring steps; each rank forwards its current slice + receives next

  INC AG  (= N concurrent broadcasts, one per source slice):
    R0     ─push slice_0─►     ┌─► R1, R2, ..., R(N-1)  receive slice_0
    R1     ─push slice_1─►    ─┤─► R0, R2, ..., R(N-1)  receive slice_1
    R2     ─push slice_2─►  switch ─► R0, R1, ..., R(N-1) receive slice_2
     ...                    multicast    ...
    R(N-1) ─push slice_(N-1)─► └─► R0, R1, ..., R(N-2)  receive slice_(N-1)

    All N multicasts pipeline through the switch's multicast engine concurrently;
    every rank ends with N slices = M bytes total.
    Total: ~2 logical round trips — N concurrent pushes + N concurrent receives
```

**AR's multicast phase** uses the same switch-multicast mechanism as BC; the multicast source is the switch ALU's reduced output (§1.1) rather than a designated root rank.

**Improvement vs software.**

- **α-term**: from $O(N)$ (ring AG) or $O(\log N)$ (pipelined tree BC) down to $\sim 2$ logical round trips. Speedup $\sim (N-1)/2$ vs ring AG, $\sim \log_2 N$ vs tree BC.
- **BW-term**: stays at $M/\mathrm{BW}$ per rank — software ring AG / pipelined tree BC already saturate the link in full-duplex via concurrent forward + receive. Multicast's gain is α-only here too.
- **AR is special** — see §1.4.

**Commercial shipment.**

- **SHARP-class INC** — switch multicast support is bundled with switch-ALU support; the same generations from §1.1 apply. NVLS supports BC / AG / AR within an NVSwitch domain; Quantum SHARP and Spectrum-X SHARP support multicast across multi-tier IB / Ethernet fat-trees; Tomahawk Ultra supports it on Ethernet scale-up.
- **PCIe switch multicast** — Broadcom PEX series (formerly PLX), Microchip Switchtec, and Astera Labs PCIe switches implement the PCIe-spec **MCAST** capability (introduced in PCIe 2.1, ~2010): a single posted write to a multicast-group ID is replicated by the crossbar to all destination ports in one switch operation. Structurally the same as the SHARP-class multicast above, but scoped to a PCIe domain (covers BC / AG along the CPU↔GPU and GPU↔NIC paths within a host). For modern NVLink-based AI clusters this path is less central than NVLS, since intra-node GPU↔GPU traffic stays on NVLink — but PCIe multicast is a valid INC primitive for non-NVLink accelerator topologies (some AMD MI-series configurations, custom inference cards, accelerators interconnected over PCIe fabric) and was the dominant scale-up multicast option in pre-NVLink HPC clusters.

### 1.3 Crossbar scatter-gather — HW A2A (emerging)

**Mechanism.** A2A doesn't fit either §1.1 or §1.2's SHARP-class capabilities — there's no reduction (every send carries a distinct payload, nothing combines) and no replication (every destination receives different data, nothing is multicast). So none of the structural $O(N) \to O(1)$ machinery from those subsections applies. But there is a **third, separate** INC primitive that *does* accelerate A2A: **hardware-driven crossbar scatter-gather**.

Software A2A has each rank serialize $N{-}1$ outbound sends through its single port; each send pays $\alpha$ (endpoint scheduling + kernel launch + NIC engine setup). HW A2A flips this:

1. Each rank concatenates its $N{-}1$ destination chunks into **one combined buffer** of size $(N{-}1)\\,M/N$, with per-chunk destination metadata.
2. The rank submits the buffer to the switch in a single transaction (one $\alpha$).
3. The switch crossbar reads the per-chunk routing tags and **forwards each chunk to its destination port in parallel within the switch**.
4. Each destination receives its $M/N$ chunk in one switch-driven transaction.

The $N{-}1$ endpoint-driven scheduling rounds collapse to ~1 endpoint submit + 1 switch routing pass + 1 endpoint receive. Per-rank α drops from $(N{-}1)\\,\alpha$ to $\sim\alpha_{\mathrm{switch}}$.

```
A2A — every source-destination pair carries a distinct payload
──────────────────────────────────────────────────────────────

  R0's send_buffer in local memory (same data layout in software and HW A2A):

      [chunk_0 (→R0)][chunk_1 (→R1)][chunk_2 (→R2)]...[chunk_(N-1) (→R(N-1))]
       ←─ M/N ─►      ←─ M/N ─►      ←─ M/N ─►         ←─ M/N ─►
                       └──────── (N-1)·M/N bytes leave the rank ────────┘


  Software pairwise A2A (N-1 separate sends per rank):

    R0 ─send(chunk_1, R1)──────► switch ─► R1     [α + (M/N)/BW]
    R0 ─send(chunk_2, R2)──────► switch ─► R2     [α + (M/N)/BW]
    ...                                                              ← N-1 endpoint-scheduled rounds
    R0 ─send(chunk_(N-1),R(N-1))► switch ─► R(N-1) [α + (M/N)/BW]

    Each send pulls one chunk from send_buffer; each pays a separate α
    (NCCL scheduling + kernel launch + NIC engine setup).


  Hardware A2A (one bulk transaction per rank):

    R0 ──(submit base ptr + descriptor)──► [Switch crossbar:
            (descriptor = per-chunk          parses descriptor;
             dest tags for chunks 1..N-1)    routes each chunk in
                                             parallel to its dest port]
                                                        ─► R1   receives chunk_1
                                                        ─► R2   receives chunk_2
                                                              ...
                                                        ─► R(N-1) receives chunk_(N-1)

    Same send_buffer bytes are physically transferred; only the submission
    descriptor differs (1 descriptor + 1 endpoint α, vs N-1 descriptors and α's).
    Per-rank cost: ~1 submit + 1 switch routing pass + 1 receive ≈ ~α_switch
```

**Crossbar routing structure — multicast xbar (§1.2) vs scatter-gather engine (§1.3).** Both bypass the switch ALU; they differ in how the crossbar dispatches data:

```
Multicast (one input → N replicas; data fanout)
─────────────────────────────────────────────────

                              ┌─► R1     [same chunk M]
                              │
  R0 ─push chunk M─► switch ──┼─► R2     [same chunk M]
                              │
                               ...
                              │
                              └─► R(N-1) [same chunk M]

  Crossbar primitive: replicate one input to many outputs.
  Bytes: 1 input chunk → N identical output copies.
  HW blocks: multicast group table + crossbar fanout duplication logic.


Scatter-gather (one bulk input → N distinct chunks; data permutation)
─────────────────────────────────────────────────────────────────────

                                                  ┌─► R1     [chunk_1]
                                                  │
  R0 ─push [chunk_1│chunk_2│...│chunk_(N-1)]──────┼─► R2     [chunk_2]
                                                  │
                            + descriptor (dest tags) ...
                                                  │
                                                  └─► R(N-1) [chunk_(N-1)]

  Crossbar primitive: parse descriptor, route each chunk to its tagged port.
  Bytes: 1 bulk input → N distinct outputs (no replication).
  HW blocks: descriptor parser + per-chunk address routing + concurrent
             multi-source ingress arbitration (N ranks may submit at once).
```

The two diagrams above show the contrasting data flows. The table below summarizes the feature-level differences side-by-side — same crossbar fabric, but each capability has its own ingress format, routing rule, output pattern, and required HW blocks:

| Aspect | Multicast xbar (§1.2) | Scatter-gather engine (§1.3) |
|---|---|---|
| Semantics | one in, many out (replication) | many in, many out (permutation) |
| Ingress | 1 buffer of M bytes at 1 input port | 1 bulk buffer of (N−1)·M/N bytes + descriptor table at 1 input port |
| Routing rule | replicate every byte to every port in the multicast group | parse per-chunk destination tags; route each chunk to its specific destination port |
| Per-byte fanout | 1 source → N destinations (with byte replication) | 1 source → 1 destination per chunk (no replication) |
| Egress | same bytes appear on every group port | different bytes appear on different ports |
| Required HW blocks | multicast group table + crossbar fanout duplication logic | descriptor parser + per-chunk address routing + concurrent multi-source ingress arbitration |
| Uses switch ALU? | no | no |
| Collectives accelerated | BC, AG, AR-multicast-phase | A2A only |

The two operations are silicon-distinct: multicast needs replication logic on the data path; scatter-gather needs descriptor parsing and per-chunk dispatch. Existing SHARP-class switches (NVSwitch Gen4, Quantum-X800) ship multicast but not the scatter-gather descriptor handler — that's why HW A2A requires a generation jump (Tomahawk Ultra today, Rubin NVSwitches next), even though both capabilities live in the crossbar fabric and neither needs the ALU.

**HW A2A is an "efficiency" win — same per-rank bytes, fewer endpoint α's.** This puts it in the same category as multicast for AG/BC and ALU-reduce for RS/Reduce: in all of these, the switch does parallel work the endpoints would have done sequentially, but the per-rank byte count is unchanged. The unique *structural* INC win remains **AR alone**, where the composed switch-ALU + multicast-xbar pair halves per-rank bytes from $\sim 2M$ to $M$ ($\mathrm{BW_{eff}}$: $\mathrm{BW}/2 \to \mathrm{BW}$); §1.4 makes this explicit in the summary table.

What distinguishes HW A2A from the SHARP-class efficiency wins is **how** the switch parallelizes:

1. **SHARP-class multicast / ALU reduce** — the switch parallelizes a *tree of endpoint operations* in its crossbar fabric (multicast group fanout, or ALU aggregation). Software does $\log_2 N$ or $N{-}1$ sequential rounds of pairwise endpoint forwarding; the switch does the equivalent work in 1 hardware operation. The α saving comes from collapsing the algorithmic tree depth.
2. **HW A2A scatter-gather** — the switch parallelizes the *endpoint submission* via descriptor batching. Per-chunk routing inside the switch is essentially normal point-to-point switching; what's new is that one bulk transaction at the ingress port subsumes $N{-}1$ separate endpoint scheduling events. The α saving comes from descriptor batching at the rank, not from algorithmic-tree collapse.

In both cases the per-rank wire-side byte count is identical to software ($(N{-}1)/N \cdot M$ for A2A, AG, RS; $M$ for BC, Reduce; etc.) — so the BW term is unchanged and the win is α-only. AR is the lone structural exception.

**Cost comparison** (intra-pod A2A at $N = 72$, $M = 16\\,\mathrm{MB}$, $\alpha_{\mathrm{inner}} = 0.5\\,\mu$s, $\alpha_{\mathrm{switch}} \approx 0.2\\,\mu$s, $\mathrm{BW}_{\mathrm{inner}} = 900\\,\mathrm{GB/s}$):

| | α term | BW term | Total |
|---|---|---|---|
| Software pairwise A2A | $(N{-}1)\\,\alpha = 35.5\\,\mu$s | $(N{-}1)/N \cdot M/\mathrm{BW} \approx 17.5\\,\mu$s | **~53 μs** |
| Hardware A2A (Rubin / Tomahawk Ultra) | $\sim 2\\,\alpha_{\mathrm{switch}} \approx 0.4\\,\mu$s | unchanged ≈ 17.5 μs | **~18 μs** |

About a **3× total speedup** at $M = 16\\,\mathrm{MB}$, dominated by the α collapse; at small $M$ where α dominates, the speedup approaches $(N{-}1)/2 \approx 36\times$. At large $M$ where BW dominates, both schedules converge toward the bisection bound ($\sim M/\mathrm{BW}$).

**Commercial shipment:**

- **NVSwitch Gen4** (current GB200 NVL72): no HW A2A — A2A runs software-scheduled through the crossbar.
- **Rubin-generation NVSwitches** (next gen): planned HW A2A support, extending NVLS beyond AR / BC / AG.
- **Broadcom Tomahawk Ultra** (Ethernet, shipped 2025): yes — first commodity-Ethernet HW A2A primitive [TH-ULTRA].
- **Quantum-X800** (current IB): no HW A2A on shipping silicon.

### 1.4 Cost-model savings — consolidated summary

§1.1, §1.2, and §1.3 each accelerate a different subset of collectives. The net α-term and BW-term effects across all primitives, evaluated on a single-switch star with $N$ ranks, $M$-byte payload, switch cut-through $\alpha_{\mathrm{switch}}$:

| Primitive | Required INC HW | α (best software) | α (INC) | α speedup | $\mathrm{BW_{eff}}$ (software) | $\mathrm{BW_{eff}}$ (INC) | BW lift |
|---|---|---|---|---|---|---|---|
| **AR** | Switch ALU + Multicast xbar | $2(N{-}1)\alpha$ (ring) or $2\lceil\log_2 N\rceil\alpha$ (DBT) | $\sim 2\\,\alpha_{\mathrm{switch}}$ | $\sim (N{-}1)$ or $\sim \log_2 N$ × | $\mathrm{BW}/2$ | $\mathrm{BW}$ | **2×** |
| **Reduce** | Switch ALU | $\lceil\log_2 N\rceil\alpha$ | $\sim \alpha_{\mathrm{switch}}$ | $\sim \log_2 N$ × | $\mathrm{BW}$ (saturated) | $\mathrm{BW}$ | $1\times$ |
| **RS** | Switch ALU | $(N{-}1)\alpha$ | $\sim 2\\,\alpha_{\mathrm{switch}}$ | $\sim (N{-}1)/2$ × | $\mathrm{BW}$ | $\mathrm{BW}$ | $1\times$ |
| **AG** | Multicast xbar | $(N{-}1)\alpha$ | $\sim 2\\,\alpha_{\mathrm{switch}}$ | $\sim (N{-}1)/2$ × | $\mathrm{BW}$ | $\mathrm{BW}$ | $1\times$ |
| **BC** | Multicast xbar | $\lceil\log_2 N\rceil\alpha$ (pipelined tree) | $\sim \alpha_{\mathrm{switch}}$ | $\sim \log_2 N$ × | $\mathrm{BW}$ | $\mathrm{BW}$ | $1\times$ |
| **A2A** | Scatter-gather engine | $(N{-}1)\alpha$ (pairwise) | $\sim \alpha_{\mathrm{switch}}$ | $\sim (N{-}1)$ × | $\mathrm{BW}$ (bisection) | $\mathrm{BW}$ (bisection) | $1\times$ |

Three structural observations:

**1. AR is the unique primitive that uses both the switch ALU and the multicast xbar.** Endpoints push their contributions into the switch ALU (reduce phase from §1.1); the ALU emits the single $M$-byte result into the multicast xbar; the crossbar replicates it back to all ports (multicast phase from §1.2). Together these complete the entire AR in one logical round trip, regardless of $N$ — endpoint software overhead is paid once for the whole collective.

This composition is why AR alone gets *both* a structural α collapse AND a BW-eff doubling. Every byte of the reduced result must visit an endpoint link **twice** in software AR (once into the reduction, once back out for redistribution); ring makes this explicit (RS phase + AG phase), and DBT pays the same dual-touch in fewer steps. NCCL's DBT in fact moves the **full** $2M$ per rank, hitting $\mathrm{BW_{eff}} = \mathrm{BW}/2$ exactly. INC's switch ALU + multicast pair does both phases inside the fabric, so each endpoint moves only $M$ bytes total — one upstream, one downstream, on opposite directions of the full-duplex link. The $2\times$ ratio is the algorithmic ceiling; realized lift is smaller (~$1.3\times$ measured on NVLS) and is treated quantitatively in `05_contention_and_congestion.md`.

**2. AG / BC / RS / Reduce get α-only wins because software already saturates link BW.** A full-duplex star runs ring AG / pipelined tree BC at $\mathrm{BW_{eff}} = \mathrm{BW}$ already — each step, a rank forwards $M/N$ outbound while receiving $M/N$ inbound concurrently, so per-rank wall-clock BW is $(N-1)/N \cdot M/\mathrm{BW} \approx M/\mathrm{BW}$. The two-touch pattern that costs AR a factor of two never applied. INC's BW-side numbers for these primitives match software (not lift it), so their speedups are dramatic at small $M$ (α-bound) and converge to $1\times$ at large $M$ (BW-bound).

**3. A2A's win is α-only and "efficiency" rather than "structural".** Total cross-sectional bytes ($(N{-}1)\\,M$) are unchanged because the switch routes them verbatim — no aggregation collapse, no replication. The α reduction from $(N{-}1)\alpha$ to $\sim\alpha_{\mathrm{switch}}$ (§1.3) comes from batching $N{-}1$ endpoint-scheduling rounds into one switch transaction, not from an algorithmic-tree collapse. Where HW A2A doesn't ship (current NVSwitch Gen4 / Quantum-X800), A2A pays the full software $(N{-}1)\alpha$ on its only path — software pairwise direct-send. SHARP-class INC at the same fabrics gives A2A nothing because the switch ALU and multicast xbar need aggregation or replication semantics, which A2A lacks.

**Per-hop α substitution at multi-tier scale.** Even for primitives where INC's $n_\alpha$ collapse stays $O(\log N)$ on a multi-tier fabric (the aggregation-tree depth $k = \lceil\log_r N\rceil$), each switch-level α is $\sim 200$–$400$ ns (cut-through) instead of the $\sim 1$–$3\\,\mu$s endpoint RTT — a 5–10× per-hop saving stacked on top of the structural collapse, because the endpoint software overhead is paid **once** at collective launch rather than once per level. §2.2 picks up the multi-tier story.

---

## 2. Scale-up vs scale-out

The INC mechanism — switch ASIC reduces in-fabric, endpoints see one round trip — applies at two very different deployment scales. Within one switch domain (**scale-up**), $n_\alpha = 2$ regardless of $N$. Across a multi-tier switched fabric (**scale-out**), $n_\alpha = 2k$ where $k = \lceil\log_r N\rceil$ is the aggregation-tree depth, but each hop is switch cut-through rather than endpoint RTT. Each scale category has two shipping implementations — one NVLink / InfiniBand (NVIDIA) and one Ethernet (Broadcom / NVIDIA).

### 2.1 Scale-up: single-switch star

The entire AR runs inside one switch domain, so $n_\alpha = 2$ end-to-end, independent of $N$ up to the switch radix cap:

$$t_{\mathrm{INC, AR}}^{\mathrm{scale-up}} \\;\approx\\; 2\alpha_{\mathrm{switch}} + \frac{M}{\mathrm{BW}}$$

per the §1.4 derivation ($\mathrm{BW_{eff}} = \mathrm{BW}$, the full algorithmic ceiling for AR). Two hardware features compose: **hardware multicast** (a single write replicated by the switch to all destination ports in one switch-local operation) and **hardware all-reduce** (switch ALU combines contributions across the SHARP group and multicasts the reduced result back).

Two shipping implementations on different fabrics:

- **NVLS — NVLink / NVSwitch.** NVIDIA's NVLink SHARP, introduced with NVSwitch Gen3 (H100 era) and extended in NVSwitch Gen4 (B200 / GB200 / NVL72). Domain size is capped by NVSwitch radix — 72 GPUs per NVL72 pod. $\alpha_{\mathrm{switch}} \approx 100$–$200$ ns. Current gen supports AR / BC / AG; cross-pod traffic falls back to software (NCCL picks DBT or ring per message size). Rubin-generation NVSwitches are expected to extend NVLS to A2A.
- **Tomahawk Ultra INC — Ethernet.** Broadcom Tomahawk Ultra (shipped 2025) is the first Ethernet switch ASIC with in-network collectives [TH-ULTRA]. 51.2 Tbps full-duplex, $\alpha_{\mathrm{switch}} \approx 250$ ns. Structurally equivalent to NVLS — single-switch star with in-fabric reduction — but on commodity Ethernet, breaking the NVLink-only monopoly on scale-up INC and opening the door to INC-enabled Ethernet fabrics at HPC / AI scale-up cost points.

### 2.2 Scale-out: multi-tier aggregation tree

Across thousands of endpoints spread over many switch levels, the INC primitive is an **aggregation tree** whose internal nodes are switch ASICs. Each level reduces its incoming flits and forwards the $M$-byte partial upward; the root completes the reduction and multicasts back down:

$$t_{\mathrm{INC, AR}}^{\mathrm{scale-out}} \\;\approx\\; 2k \cdot \alpha_{\mathrm{switch}} + \frac{M}{\mathrm{BW}}$$

where $k$ is the **number of switch tiers** an $M$-byte flit traverses from an endpoint to the root of the aggregation tree. For a fat-tree with per-switch radix $r$ (fan-in per level), covering $N$ endpoints requires $k = \lceil\log_r N\rceil$ tiers — e.g., $r = 64$ ports per switch covers $N = 4096$ endpoints in $k = 2$ tiers (leaf + spine), or $N = 262{,}144$ in $k = 3$ (leaf + spine + super-spine). The factor of $2k$ on the $\alpha$ term accounts for the round trip: $k$ switch hops up to the root during reduce, then $k$ back down during multicast. $\alpha_{\mathrm{switch}} \approx 200$–$400$ ns is the switch cut-through, not endpoint software RTT. The BW term stays at $M/\mathrm{BW}$ — the same algorithmic ceiling as scale-up ($\mathrm{BW_{eff}} = \mathrm{BW}$ from §1.4), because each endpoint still only pushes $M$ up and receives $M$ down, and cut-through pipelining at each tier keeps the endpoint's outbound and inbound directions overlapping once the pipeline is filled ($\sim 2k\alpha_\mathrm{switch}$).

Concretely, for $N = 4096$ across a 3-tier aggregation tree, AR latency is $\sim 2$–$3\\,\mu$s — even though software ring AR at the same $N$ would need 8190 sequential endpoint $\alpha$s ($\sim 8$ ms at endpoint $\alpha = 1\\,\mu$s). The $\sim 3000\times$ speedup is dominated by the endpoint-to-switch $\alpha$ substitution combined with the structural $O(N) \to O(\log N)$ collapse.

Two shipping implementations on different fabrics:

- **Quantum SHARP — InfiniBand.** NVIDIA Mellanox Quantum-2 and Quantum-X800 switches on IB fat-tree / Clos topologies. The established scale-out INC path for large training clusters.
- **Spectrum-X SHARP — Ethernet.** NVIDIA's Ethernet-side analog, running the SHARP protocol over RoCE on Spectrum-X switches. Brings the scale-out INC story to commodity-Ethernet clusters — complementing Tomahawk Ultra on the scale-up side.

---

## 3. Worked example at $N = 512$

To track the same $N = 512$ anchor used in `05_contention_and_congestion.md §5`, apply scale-up INC to a **hypothetical single-switch star with 512 ports**. Real scale-up INC (NVL72) caps at $N = 72$ — a 512-port single-switch INC ASIC does not exist today, so a production $N \geq 512$ deployment would use **scale-out INC over a multi-tier Clos** (`03_hierarchical_topologies.md` Appendix A works through the rail-optimized GB200 SuperPOD topology at $L = 32$ NVL72s / $N = 2304$ GPUs, showing the actual 8-leaves + 4-spines per-rail fat-tree that current high-count clusters deploy). We pick the single-switch abstraction for the worked example below because it applies the §1.1 / §1.2 / §1.4 algorithmic ceilings directly ($n_\alpha = 2$, $\mathrm{BW_{eff}} = \mathrm{BW}$); scale-out INC at $k = 2$, $\alpha_\mathrm{switch} = 0.5\\,\mu$s gives essentially the same total ($\sim 20\\,\mu$s vs $\sim 19\\,\mu$s for scale-up) because the $M / \mathrm{BW}$ term dominates at $M = 16\\,\mathrm{MB}$.

Per-link $\alpha = 0.5\\,\mu$s (switch cut-through + endpoint software), $\mathrm{BW} = 900\\,\mathrm{GB/s}$, $M = 16\\,\mathrm{MB}$.

### 3.1 The full $N = 512$ ladder (AR)

Evaluating the three software / torus rows from the formulas in `02_topology_mapping.md §5.1` and adding one **NVLS-style INC** row on the hypothetical single-switch star:

| Topology | Algorithm | $n_\alpha$ | $\alpha$ term | BW term | **Total** |
|---|---|---|---|---|---|
| Star (crossbar) | Ring AR | 1022 | 511 μs | 35.5 μs | **546 μs** |
| Star (crossbar) | DBT / RHD | 18 | 9 μs | 35.5 μs | **45 μs** |
| Hypothetical 512-port star | **NVLS-style INC** | 2 | 1 μs | 17.8 μs | **18.8 μs** |
| Torus $8 \times 8 \times 8$ | Dim-decomp ring | 42 | 21 μs | 35.5 μs | **57 μs** |

INC closes two gaps at once:

- **$\alpha$ side.** $n_\alpha$ collapses from 18 (DBT) or 1022 (ring) to 2. The α term shrinks from 9 μs (DBT) to 1 μs (INC) — small in absolute terms at this $M$, but consequential at smaller $M$ (see §3.2).
- **BW side.** $\mathrm{BW_{eff}}$ doubles from $\mathrm{BW}/2$ (any software schedule) to $\mathrm{BW}$ (INC), halving the BW term from 35.5 μs to 17.8 μs.

Net: star + INC at **18.8 μs** beats the best software pairing (star + DBT at 45 μs) by $\sim 2.4\times$ and the torus at 57 μs by $\sim 3\times$. Against the pathological star+ring row (546 μs) the speedup is $\sim 29\times$, but that row is a cautionary baseline, not a design we'd ship. The $\sim 2\times$ ceiling from §1.4 is the asymptotic BW-side INC speedup for AR; the full $\sim 2.4\times$ vs DBT at $M = 16\\,\mathrm{MB}$ reflects the residual $\alpha$-side contribution.

### 3.2 Regime sensitivity (AR)

The $N = 512$ speedup of INC over the best software pairing (DBT) varies sharply with $M$. Sweeping $M$ from $\alpha$-bound to BW-bound:

| $M$ | Star + DBT | Star + INC | Speedup (DBT → INC) |
|---|---|---|---|
| 10 KB ($\alpha$-bound) | 9.02 μs | 1.01 μs | **~9×** |
| 1 MB | 11.2 μs | 2.1 μs | **~5×** |
| 16 MB (anchor) | 45 μs | 18.8 μs | **~2.4×** |
| 1 GB (BW-bound) | 2.23 ms | 1.11 ms | **~2×** |

At small $M$ the $\alpha$-term collapse dominates: DBT still needs 18 synchronizations at 0.5 μs each (= 9 μs); INC needs 2 (= 1 μs). At large $M$ the BW-term ratio takes over and the speedup converges to the $2\times$ payload-count ceiling. The transition is governed entirely by the $\alpha$-vs-BW crossover for DBT: $M^\star \approx n_\alpha^{\mathrm{DBT}} \cdot \alpha \cdot \mathrm{BW} / 2 = 18 \cdot 0.5\\,\mu\mathrm{s} \cdot 900\\,\mathrm{GB/s} / 2 \approx 4\\,\mathrm{MB}$. Below $M^\star$ INC's $\alpha$ savings dominate; above, its BW savings do.

The ceilings in this section assume frictionless cut-through, no ALU or multicast contention, and no scheduler overhead. `05_contention_and_congestion.md` §5 re-runs the same $N = 512$ ladder under realistic $\eta$ and quantifies how much of each ceiling survives in deployment.

### 3.3 Cross-primitive comparison (non-AR primitives)

§3.1–§3.2 focused on AR. The $\alpha$-side INC collapse applies to AG, RS, BC, and Reduce as well; the BW-side collapse is AR-exclusive (§1.4 — the other reduction-or-replication primitives already hit $\mathrm{BW_{eff}} = \mathrm{BW}$ in software via full-duplex operation). A2A gets no SHARP-class INC lift on any hardware; HW A2A (§1.3) is the separate path. The table runs the same $N = 512$, $M = 16\\,\mathrm{MB}$, $\alpha = 0.5\\,\mu$s, $\mathrm{BW} = 900\\,\mathrm{GB/s}$ anchor across all five non-AR primitives:

| Primitive | Topology / Algorithm | $n_\alpha$ | $\alpha$ term | BW term | **Total** |
|---|---|---|---|---|---|
| **AG / RS** | Star ring | 511 | 255.5 μs | 17.7 μs | **273 μs** |
|  | Star rec-doub AG / rec-halv RS (`01_collective_algorithms.md` App. B.4) | 9 | 4.5 μs | 17.7 μs | **22.2 μs** |
|  | Hypothetical 512-port star + INC | 2 | 1.0 μs | 17.7 μs | **18.7 μs** |
|  | Torus $8 \times 8 \times 8$ dim-decomp ring | 21 | 10.5 μs | 17.7 μs | **28.2 μs** |
| **BC / Reduce** | Star ring | 511 | 255.5 μs | 17.8 μs | **273 μs** |
|  | Star pipelined tree (DBT) | 9 | 4.5 μs | 17.8 μs | **22.3 μs** |
|  | Hypothetical 512-port star + INC | 1 | 0.5 μs | 17.8 μs | **18.3 μs** |
|  | Torus $8 \times 8 \times 8$ dim-decomp bidirectional | 12 | 6.0 μs | 17.8 μs | **23.8 μs** |
| **A2A** | Star pairwise (NCCL) | 511 | 255.5 μs | 17.7 μs | **273 μs** |
|  | Hypothetical 512-port star + SHARP-class INC | — | — | — | **N/A** (no aggregation/replication semantics; §1.1) |
|  | Hypothetical 512-port star + HW A2A (Tomahawk Ultra / Rubin) | $\sim 2$ | $\sim 1\\,\mu$s | 17.7 μs | **~19 μs** (α-only collapse; §1.3) |
|  | Torus $8 \times 8 \times 8$ bisection-bound (TPU / Trainium) | 12 | 6 μs | 17.8 μs | **23.8 μs** |

Three observations:

1. **AG / RS / BC / Reduce INC closes the α-side gap but not the BW-side gap.** Star + INC (~18 μs) beats the best software (~22 μs) by only $\sim 1.2\times$ at this anchor — a much tighter margin than AR's $2.4\times$ at the same $M$. The entire gap is the $\alpha$-term collapse (e.g., AG / RS rec-doub: $4.5 \to 1.0\\,\mu$s; BC / Reduce DBT: $4.5 \to 0.5\\,\mu$s); BW terms match because software rec-doub / DBT already hits $\mathrm{BW_{eff}} = \mathrm{BW}$ on a full-duplex star. At small $M$ the ratio widens toward the $\alpha$-only ceiling ($\sim 4.5\times$ for AG/RS, $\sim 9\times$ for BC/Reduce); at large $M$ all four collapse to $1\times$.
2. **A2A's INC story is split across two capabilities, neither giving the AR-style structural collapse.** SHARP-class INC (switch ALU + multicast) gives A2A no win at all on any hardware — the primitive's per-destination payloads have no aggregation or replication structure to exploit. The separate HW A2A primitive (§1.3) ships today on Tomahawk Ultra and is planned for Rubin-generation NVSwitches; it collapses the α-side per-destination scheduling to $\sim 1\alpha$ but leaves the BW term at the bisection bound (which a switch ALU cannot reduce by forwarding-verbatim semantics). Current shipping NVSwitch Gen4 (NVL72) and Quantum-X800 include neither, so A2A on those platforms falls back to software-scheduled pairwise sends and the topology-side win path: torus bisection-bound reduces $(N{-}1)\alpha$ to $\mathrm{diam}\cdot\alpha$ at the cost of a $D_{\max}/8$ BW penalty (which vanishes at cubic shapes).
3. **Torus stays competitive when INC is unavailable.** At $N = 512$, $M = 16\\,\mathrm{MB}$, torus 8³ dim-decomp is $\sim 1.3$–$1.5\times$ slower than hypothetical INC across AG / RS (28.2 μs vs 18.7) and BC / Reduce (23.8 vs 18.3), but $\sim 10\times$ faster than star ring — the dim-decomposition's $\sum(D_i - 1)$ or $\sum\lfloor D_i/2 \rfloor$ $\alpha$ collapse carries most of the win. Torus is the best A2A option by a wide margin when INC is unavailable (23.8 μs vs star pairwise's 273 μs — $\sim 11.5\times$), because the dim-decomposed $\mathrm{diam}\cdot\alpha$ latency term cuts deeper while the $D_{\max}/8$ BW penalty is $1\times$ at the cubic 8³ shape.

The pattern across all six primitives: **AR is the only primitive where INC pays back at every $M$** (both $\alpha$ and BW wins); AG / RS / BC / Reduce wins concentrate at small $M$ (α-only); A2A's only INC path is the separate HW A2A primitive (§1.3), which gives an α-side efficiency win on Tomahawk Ultra today (and on Rubin NVSwitches in the next generation) but no BW lift. `05_contention_and_congestion.md` §5 layers realistic $\eta$ on each of these rows and quantifies which speedups survive contention.

---

## 4. INC-star vs torus

§1.4 summarizes the per-primitive INC vs software-on-star comparison. This section completes the architectural picture by adding **torus** as the third design point — the natural choice when INC is unavailable — and asking, for each primitive, how INC-on-star compares to a same-$N$ torus at their algorithmic ceilings. The table echoes `02_topology_mapping.md §5.1` (per-primitive cost on each topology) and adds INC rows under **Star**:

| Topology | Primitive | Algorithm | α term | BW term | INC-on-star speedup (small M / large M) |
|---|---|---|---|---|---|
| **Star** | AR | Ring | $2(N-1)\\,\alpha$ | $2(N-1)/N \cdot M/\mathrm{BW}$ | $\sim (N-1)\times$ / $\sim 2\times$ |
|  | AR | DBT | $2\lceil\log_2 N\rceil\\,\alpha$ | $2(N-1)/N \cdot M/\mathrm{BW}$ | $\sim \log_2 N\times$ / $\sim 2\times$ |
|  | AR | **INC** (ALU + multicast) | $2\\,\alpha_{\mathrm{switch}}$ | $M/\mathrm{BW}$ | — (reference) |
|  | AG / RS | Ring | $(N-1)\\,\alpha$ | $(N-1)/N \cdot M/\mathrm{BW}$ | $\sim (N-1)/2\\,\times$ / $1\times$ |
|  | AG / RS | **INC** (multicast or ALU+scatter) | $2\\,\alpha_{\mathrm{switch}}$ | $(N-1)/N \cdot M/\mathrm{BW}$ | — (reference) |
|  | BC / Reduce | Pipelined tree (DBT) | $\lceil\log_2 N\rceil\\,\alpha$ | $M/\mathrm{BW}$ | $\sim \log_2 N\times$ / $1\times$ |
|  | BC / Reduce | **INC** (multicast or ALU) | $\alpha_{\mathrm{switch}}$ | $M/\mathrm{BW}$ | — (reference) |
|  | A2A | Pairwise | $(N-1)\\,\alpha$ | $(N-1)/N \cdot M/\mathrm{BW}$ | $\sim (N-1)\times$ / $1\times$ (vs HW A2A); no speedup with SHARP-class only |
|  | A2A | **INC** (HW A2A scatter-gather; §1.3) | $\alpha_{\mathrm{switch}}$ | $(N-1)/N \cdot M/\mathrm{BW}$ | — (reference; ships on Tomahawk Ultra / Rubin) |
| **Torus** | AR | Dim-decomp ring | $2\sum_i (D_i-1)\\,\alpha$ | $2(N-1)/N \cdot M/\mathrm{BW}$ | $\sim \sum_i (D_i-1)\times$ / $\sim 2\times$ |
|  | AG / RS | Dim-decomp ring | $\sum_i (D_i-1)\\,\alpha$ | $(N-1)/N \cdot M/\mathrm{BW}$ | $\sim \sum_i (D_i-1)/2\\,\times$ / $1\times$ |
|  | BC / Reduce | Dim-decomp bidirectional | $\sum_i \lfloor D_i/2 \rfloor\\,\alpha$ | $M/\mathrm{BW}$ | $\sim \sum_i \lfloor D_i/2 \rfloor\\,\times$ / $1\times$ |
|  | A2A | Pairwise — bisection-bound | $\mathrm{diam}\cdot\alpha$ | $D_{\max}/8 \cdot M/\mathrm{BW}$ | $\sim \mathrm{diam}\times$ / $\sim D_{\max}/8\\,\times$ ($1\times$ at $8^3$; $\sim 2\times$ at $16^3$ — typical TPU pod slice shape) |

Star-row α and BW-eff values match §1.4's per-primitive summary; the torus rows are the dim-decomposed costs from `02_topology_mapping.md §3` plugged in for the architectural-swap comparison. Three observations on the INC-star vs torus comparison:

1. **AR is the only primitive where INC-on-star structurally beats torus on BOTH axes.** α-side: $\sim \sum_i(D_i-1)$ hops on torus vs $\sim 2$ on INC-star — a deep-vs-shallow tree gap. BW-side: torus AR pays the same $2\times$ dual-touch BW penalty as software-on-star ($\mathrm{BW_{eff}} = \mathrm{BW}/2$), while INC-on-star halves per-rank traffic via the composed switch ALU + multicast xbar pair → $\mathrm{BW_{eff}} = \mathrm{BW}$. **Torus has no analog of this composition.** So INC-star beats torus on AR at every $M$: ~2× from BW alone at large $M$, plus the α-side gap at smaller $M$.
2. **For α-only primitives (AG / RS / BC / Reduce), INC-star and torus are α-side competitors; both saturate link BW.** Torus dim-decomp on a full-duplex ring achieves $\mathrm{BW_{eff}} = \mathrm{BW}$, just like ring AG / pipelined-tree BC on a star. The remaining gap is α-only — INC-star's $\sim \alpha_{\mathrm{switch}}$ or $2\alpha_{\mathrm{switch}}$ vs torus's $\sum_i(D_i-1)\\,\alpha$ or $\sum_i \lfloor D_i/2\rfloor\\,\alpha$. At production sizes ($N = 512$ as 8³ torus: $\sum(D_i-1) = 21$, $\sum\lfloor D_i/2\rfloor = 12$), INC-star's α-side advantage is order of magnitude in the latency-bound regime; at large $M$ both architectures converge to the same $M/\mathrm{BW}$ ceiling. **The choice between INC-star and torus on these four is rarely a deal-breaker** — deployment-side considerations (rack power, switch radix, cabling) often dominate.
3. **A2A's topology choice flips with hardware availability.** SHARP-class INC gives A2A no win on any hardware. With HW A2A (Tomahawk Ultra today, Rubin NVSwitches next), star + HW A2A collapses α to $\sim \alpha_{\mathrm{switch}}$ but the BW term stays bisection-bound — same as torus. So both architectures end up bisection-bound; INC-star edges out torus only on α. **Without HW A2A (current GB200 NVL72 / Quantum-X800), A2A on a star pays the full software $(N-1)\\,\alpha$, and torus's $\mathrm{diam}\cdot\alpha$ saves an order of magnitude** — making torus the natural fallback for A2A-heavy workloads (TPU / Trainium clusters lean on this; star-only deployments without HW A2A struggle on MoE EP traffic). The right topology decision flips depending on whether the deployment has HW A2A — the most fabric-architecture-sensitive primitive in this comparison.

**Limitations** — every cost formula above is an algorithmic ceiling. Six restrictions apply in practice:

1. **Scope: single switch domain (or SHARP-enabled hierarchy).** NVLS works only within one NVSwitch domain. Quantum SHARP works across a SHARP-enabled IB fabric, but only if all switches in the aggregation tree support the protocol. Cross-domain traffic falls back to software.
2. **Reducible types: limited.** Production switch ALUs support sum, max, min, bit-and/or/xor, and a few related ops over BF16/FP16/FP32 (FP64 varies by switch generation). **Compound reductions like softmax (which requires exp() + sum + per-element division) and top-k are not supported in shipping SHARP-class hardware** — they fall back to either in-fabric approximations on programmable-pipeline switches (Tofino-class research prototypes; published academic work but not productized) or full software execution. Roadmap extensions (Rubin-generation NVSwitches and successor Quantum SHARP versions) may broaden the op set; verify against current vendor documentation when reduction semantics matter beyond the standard set.
3. **Precision: hardware-fixed.** Switch ALUs typically operate in BF16 with FP32 accumulators; some SHARP-enabled IB switches support FP32 directly. The accumulation order is also fixed by the switch tree shape, which can produce slightly different numerical results than ring AR — usually within numerical tolerance but worth flagging for bitwise-reproducibility use cases.
4. **Message alignment.** INC operations require specific alignment (typically 128 B or higher); very small messages can fall through to software — NCCL's heuristic handles this, but a sweep across $M$ may show a sudden floor where the runtime switches paths.
5. **BW-regime convergence.** INC's dominant win is in the latency-term regime. AR's $\sim$70× speedup at small $M$ shrinks to $\sim 2\times$ at large $M$; AG / RS / BC / Reduce shrink to $1\times$; A2A's HW A2A gain similarly converges to $1\times$ at the bisection bound.
6. **Ceilings vs realized speedups.** The ratios above are **algorithmic ceilings** — they assume frictionless cut-through pipelining, no multicast contention, and no NCCL / switch-ALU scheduling overhead. Realized speedups on production hardware are smaller (e.g., NVLink NVLS measures $\sim 1.3\times$ BW-regime lift vs the $2\times$ ceiling). The full contention-coefficient treatment — calibrating $\eta_\alpha, \eta_\beta$ against published busbw measurements and re-running the comparison under realistic factors — is in `05_contention_and_congestion.md`.

---

## Further reading

- **`01_collective_algorithms.md`** — baseline ring and tree AR costs that INC compresses against; the $\alpha$-$\beta$ model and the Rabenseifner / DBT derivations referenced throughout.
- **`02_topology_mapping.md` §2** — $\alpha$ / BW calibration on a scale-up star and star-specific observations (e.g., DBT's port-uniform utilization on the crossbar). §5.1 observation 6 flags INC as star's $\alpha$-compression escape hatch.
- **`05_contention_and_congestion.md`** — contention coefficients that modify software collectives; INC paths are less sensitive to contention because they use only one switch operation.
- **`03_hierarchical_topologies.md` §3** — how INC at both tiers of the hierarchy composes: outer-tier INC (Quantum SHARP / Spectrum-X SHARP / Tomahawk Ultra at the Clos spine) for the biggest absolute α saving, and inner-tier INC (NVLS within an NVL72 pod) for the residual α plus the BW-eff lift on intra-pod AR. NVLS + Quantum SHARP combined brings hierarchical AR at L = 32 pods to roughly the cost of a single-pod NVLS AR.
- **`references.md`** — primary sources for SHARP (Graham et al. 2016), NVLink SHARP / NVLS (NVIDIA 2023 whitepapers, NCCL 2.27 release notes), and Tomahawk Ultra INC (Broadcom 2025).
- **Graham et al. (2016)**, "Scalable Hierarchical Aggregation Protocol (SHARP)" — the architectural paper.
- **NVIDIA (2023)**, NVLink SHARP and NVSwitch Multicast/Reduce Whitepapers.
