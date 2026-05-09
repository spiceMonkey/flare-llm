# 3D-Stacked DRAM Bandwidth Model

**Deriving Effective Memory Bandwidth from Hybrid Bonding Interface Parameters**

<br/>

**Author:** Yue Lu  
**Date:** April 2026

**Keywords:**  
3D-stacked DRAM, hybrid bonding, HBM4E, memory bandwidth, pin density, die area, roofline model, LLM inference, AccelStack

---

<div style="page-break-after: always;"></div>

# Table of Contents

- [1. Physical Interface Parameters](#1-physical-interface-parameters)
- [2. Pin Count and Bandwidth Derivation](#2-pin-count-and-bandwidth-derivation)
  - [2.1 Step 1 — Total Pins from Die Area and Pitch](#21-step-1--total-pins-from-die-area-and-pitch)
  - [2.2 Step 2 — Data Pins After Overhead Allocation](#22-step-2--data-pins-after-overhead-allocation)
  - [2.3 Step 3 — Bandwidth per Die](#23-step-3--bandwidth-per-die)
  - [2.4 Step 4 — Two Stacking Models](#24-step-4--two-stacking-models)
  - [2.5 Effective Bandwidth Is Always Discounted from Peak](#25-effective-bandwidth-is-always-discounted-from-peak)
- [3. Numerical Examples](#3-numerical-examples)
  - [3.1 Scenario 1 — HBM3E Baseline (Calibration)](#31-scenario-1--hbm3e-baseline-calibration)
  - [3.2 Scenario 2 — Near-Term Hybrid Bonding ($p_{HB}$ = 2 µm)](#32-scenario-2--near-term-hybrid-bonding-p_hb--2-µm)
  - [3.3 Scenario 3 — Aggressive Hybrid Bonding ($p_{HB}$ = 0.9 µm)](#33-scenario-3--aggressive-hybrid-bonding-p_hb--09-µm)
- [4. Latency Model](#4-latency-model)

---

<div style="page-break-before: always;"></div>

## Introduction

Conventional HBM attaches DRAM dies to a logic die via through-silicon vias (TSVs) and solder-bump interconnects (copper pillars / microbumps), with pitches on the order of **~50 µm** [LAU-PKG][SEMIANALYSIS-HB]; at these pitches the number of interface pins is limited by package-level I/O layout rather than die area. 3D stacking with **hybrid bonding** collapses this interconnect to direct copper-to-copper bonds through a dielectric interlayer, reaching **~10 µm near-term and sub-micron in next-generation designs** [LAU-PKG][SOIC-UHD] — unlocking bandwidth that scales with die area rather than package perimeter. AccelStack captures this qualitatively as "hybrid bonding techniques could potentially offer 5–30× smaller than that of microbumps" [ACCELSTACK, §I]. This document derives the effective memory bandwidth $BW_{\text{mem}}$ from first principles, following the methodology of AccelStack §III-C2 [ACCELSTACK]; §III-C2 gives the qualitative framework but contains no dedicated equation, so the derivation here is an original first-principles extension. The resulting $BW_{\text{mem}}$ is the quantity that enters the roofline model (see `decode.md` §4).

---

# 1. Physical Interface Parameters

The model is parameterized by five physical quantities. All are observable from die specifications or packaging datasheets.

| Symbol | Description | Units | Example |
|--------|-------------|-------|---------|
| $A_{\text{die}}$ | DRAM die area (the face of one DRAM die bonded to the logic die) | mm² | 100 mm² |
| $p_{HB}$ | Hybrid bonding pitch: center-to-center spacing between adjacent pads | µm | 0.9 µm |
| $\eta_{\text{data}}$ | Data pin fraction: proportion of all pads used for data signals | dimensionless | 0.4 |
| $f_{\text{data}}$ | Data rate per pin | Gbps | 8 Gbps |
| $N_{\text{dies}}$ | Number of DRAM dies stacked on the logic die | integer | 4 |

**On pitch ranges.** The interconnect landscape spans roughly two orders of magnitude in pitch. Representative commercial and demonstrated datapoints:

| Technology | Pitch | Pad density | Anchor |
|-----------|-------|-------------|--------|
| C4 flip-chip | ~100–150 µm | ~100 bumps/mm² | Nvidia A100 [SEMIANALYSIS-HB] |
| Conventional microbump / Cu pillar | **~50 µm** | **~400 bumps/mm²** | Intel Foveros (2020) [LAU-PKG]; A100 copper pillars [SEMIANALYSIS-HB] |
| Advanced microbump | ~20–25 µm | ~1,600–2,500 bumps/mm² | Industry trend |
| Hybrid bonding — near-term | **~10 µm** | **~10,000 pads/mm²** | Intel Foveros Direct (2020) [LAU-PKG] |
| Hybrid bonding — current production | ~1–2.5 µm | ~160 K–1 M bonds/mm² | Sony CIS; TCB/W2W (Sakuma ECTC 2024, in [ACCELSTACK]) |
| Hybrid bonding — next-generation | **sub-µm (≥0.5)** | **≥1.2 M bonds/mm²** | TSMC SoIC UHD (2020) [SOIC-UHD] |

At 1 µm pitch, a 100 mm² die supports on the order of $10^8$ pads — four orders of magnitude more than a conventional flip-chip package at the same footprint. The qualitative framing used by AccelStack — hybrid bonding pitch is "5–30× smaller than that of microbumps" [ACCELSTACK, §I] — is consistent with the table above (50 µm ÷ 10 µm ≈ 5×; 50 µm ÷ 1.7 µm ≈ 30×).

**On $\eta_{\text{data}}$.** Not all pads carry data. Power delivery, ground return, clock distribution, ECC check bits, and control/command signals collectively consume roughly 50–70 % of total pads in real DRAM designs. The data pin fraction $\eta_{\text{data}} \approx 0.3$–$0.5$ captures this allocation; a value of 0.4 is used as the default throughout this document.

---

# 2. Pin Count and Bandwidth Derivation

## 2.1 Step 1 — Total Pins from Die Area and Pitch

Pads are arranged in a 2D grid with spacing $p_{HB}$ in both the $x$ and $y$ directions. The number of grid points that fit within the die area $A_{\text{die}}$ is:

$$
N_{\text{pins,total}} = \left\lfloor \frac{A_{\text{die}}}{p_{HB}^2} \right\rfloor
$$

where $A_{\text{die}}$ and $p_{HB}$ must be expressed in consistent units (both in µm², or $A_{\text{die}}$ in µm² = $A_{\text{die,mm}^2} \times 10^6$). This is the theoretical maximum pad count, assuming full-area coverage.

## 2.2 Step 2 — Data Pins After Overhead Allocation

Of the $N_{\text{pins,total}}$ pads, only a fraction $\eta_{\text{data}}$ carries data:

$$
N_{\text{pins,data}} = \left\lfloor N_{\text{pins,total}} \cdot \eta_{\text{data}} \right\rfloor
$$

## 2.3 Step 3 — Bandwidth per Die

Each data pin runs at $f_{\text{data}}$ Gbps (gigabits per second). Converting to GB/s (gigabytes per second):

$$
BW_{\text{die}} = N_{\text{pins,data}} \times \frac{f_{\text{data}}}{8} \quad [\text{GB/s}]
$$

The factor of 8 converts from bits to bytes. Substituting Steps 1 and 2:

$$
BW_{\text{die}} = \left\lfloor \left\lfloor \frac{A_{\text{die}}}{p_{HB}^2} \right\rfloor \cdot \eta_{\text{data}} \right\rfloor \times \frac{f_{\text{data}}}{8} \quad [\text{GB/s}]
$$

This is the bandwidth presented by a single DRAM die's interface to whatever it is bonded to — either the logic die (in Model A/B below) or the die below it in the stack.

## 2.4 Step 4 — Two Stacking Models

Whether additional dies in the stack contribute independent bandwidth depends on the physical integration topology. Two bounding models are presented.

### Model A — Conservative: Single Logic-Facing Interface

In this model, $N_{\text{dies}}$ DRAM dies are stacked vertically, but only the **bottom die** bonds directly to the logic die. Upper dies route their data through the bottom die's TSVs, and are therefore bottlenecked by the single bottom-die-to-logic-die interface. The memory controller sees one logical port regardless of stack height:

$$
BW_{\text{conservative}} = BW_{\text{die}}
$$

Stack height increases **capacity** ($N_{\text{dies}} \times$ capacity per die) but does not increase bandwidth in this model. This matches the HBM architecture, where each stack of 4 or 8 dies exposes a single 1024-bit wide interface to the package.

### Model B — Optimistic: Independent Per-Die Interfaces

In this model, each DRAM die has its own direct hybrid-bonded interface to the logic die (true 3D integration, where each die is individually bonded to a redistribution layer on the logic die's surface). Each die contributes independently:

$$
BW_{\text{optimistic}} = N_{\text{dies}} \times BW_{\text{die}}
$$

This is the theoretical upper bound for fully independent die-to-die connections. It requires that the logic die's surface have sufficient area to accommodate $N_{\text{dies}}$ separate DRAM footprints simultaneously.

### Bandwidth Bounds

$$
BW_{\text{conservative}} \;\le\; BW_{\text{mem}} \;\le\; BW_{\text{optimistic}}
$$

**Which model to use.** The AccelStack paper (§III-C2) does not resolve this question definitively [ACCELSTACK]. Practical 3D DRAM designs, including the Samsung HBM4 roadmap and near-term HBM4E architectures, indicate that the conservative model is more realistic: the bottom die becomes the bandwidth bottleneck and upper dies primarily add capacity. The conservative model is therefore the **recommended default**; the optimistic model serves as a research upper bound for future fully-disaggregated 3D integration.

## 2.5 Effective Bandwidth Is Always Discounted from Peak

The $BW_{\text{mem}}$ derived in §2.1–2.4 is a **peak ceiling** — the product of pad count and per-pin data rate. Real kernels never see all of it: coalescing quality, tile geometry, and PHY layout collectively impose an efficiency discount. Datasheet HBM numbers are similarly peak; the sustained bandwidth that enters the roofline (see `notation.md §6`) is always lower. No new symbol is introduced for this discount — it is absorbed into $BW_{\text{mem}}$ as used elsewhere in the suite.

AccelStack's GEMM latency model [ACCELSTACK, Eq. 2] makes the tile dependence of this discount explicit. For a matmul $(m, k) \times (k, n)$ on a $p \times p$ systolic array with tile $(t_m, t_n, t_k)$:

$$
t_{\text{comp}} = \left\lceil \frac{m}{t_m} \right\rceil \left\lceil \frac{n}{t_n} \right\rceil \left\lceil \frac{k}{t_k} \right\rceil \cdot 2 \left\lceil \frac{t_m}{p} \right\rceil \left\lceil \frac{t_n}{p} \right\rceil t_k
$$

Tile size $(t_m, t_n, t_k)$ sets operand reuse and therefore the BW demand per compute cycle: smaller tiles demand more BW per cycle, which only a fast and non-contended memory interface can sustain.

**Why 3D hybrid bonding lifts effective BW closer to peak.** AccelStack §III-C (Fig. 6) places "distributed controllers and PHYs, aligned with PEs" on the logic die — a structural advantage that HBM cannot match:

1. **Per-die PHYs, not per-stack.** A hybrid-bonded stack exposes many independent PHY regions per die, whereas HBM funnels 4–8 dies through a single 1,024-bit package edge. Small, fine-grained tile accesses do not serialize.
2. **PE-aligned placement.** Operand fetch traverses one vertical hop rather than crossing an on-die NoC — intra-die routing contention vanishes.
3. **Latency headroom.** With $\ell_{3D} \approx 20$–$50$ ns vs. ~$300$ ns for HBM access [ACCELSTACK §III-C], the smallest tile that keeps the pipeline full shrinks by the same factor.

The practical consequence: AccelStack's tiled-GEMM model combined with 3D DRAM's PE-aligned PHYs brings the **effective bandwidth closer to the peak** derived in §2.1–2.4 — **another advantage of 3D DRAM beyond the raw pin × data-rate lift**. The scenario numbers in §3 remain peak ceilings; HBM deployments sit further below them than 3D-stacked deployments do.

---

# 3. Numerical Examples

## 3.1 Scenario 1 — HBM3E Baseline (Calibration)

This scenario calibrates the model against a known, shipping product (SK Hynix HBM3E) to validate the methodology and reveal where the assumptions break down.

**Parameters:**

| Parameter | Value |
|-----------|-------|
| $A_{\text{die}}$ | 82 mm² (SK Hynix HBM3E die) |
| $p_{HB}$ | 55 µm (microbump / TSV, not hybrid bonding) |
| $\eta_{\text{data}}$ | 0.5 |
| $f_{\text{data}}$ | 6.4 Gbps per pin |
| $N_{\text{dies}}$ | 4 |

**Computation:**

$$
N_{\text{pins,total}} = \left\lfloor \frac{82 \times 10^6\;\mu\text{m}^2}{(55\;\mu\text{m})^2} \right\rfloor = \left\lfloor \frac{82{,}000{,}000}{3{,}025} \right\rfloor = \left\lfloor 27{,}107 \right\rfloor = 27{,}107
$$

$$
N_{\text{pins,data}} = \lfloor 27{,}107 \times 0.5 \rfloor = 13{,}553
$$

$$
BW_{\text{die}} = 13{,}553 \times \frac{6.4}{8} = 13{,}553 \times 0.8 \approx 10{,}842 \;\text{GB/s}
$$

**Reconciliation against datasheet.** Actual HBM3E delivers approximately 1.2 TB/s per 4-die stack — that is, ~300 GB/s per die — using a physical interface of 1,024 data pins per die at 6.4 Gbps per pin (per [HBM-SPEC]). The formula above predicts 13,553 data pins, whereas reality is 1,024. The discrepancy is a factor of ~13×, and the explanation is straightforward: **conventional HBM is pad-layout-limited, not area-limited.** At 55 µm microbump pitch, only 27,107 bumps fit in the die footprint in principle, but the actual HBM I/O design uses far fewer, fixed-position pads constrained by the PHY circuit layout and package routing — not by bump density. The area-based formula only becomes accurate when hybrid bonding enables true area-limited pad density. The key insight is: the 3D hybrid bonding model is meaningful precisely because it removes the pad-layout bottleneck.

## 3.2 Scenario 2 — Near-Term Hybrid Bonding ($p_{HB}$ = 2 µm)

**Parameters:**

| Parameter | Value |
|-----------|-------|
| $A_{\text{die}}$ | 100 mm² |
| $p_{HB}$ | 2 µm |
| $\eta_{\text{data}}$ | 0.4 |
| $f_{\text{data}}$ | 8 Gbps per pin |
| $N_{\text{dies}}$ | 4 |

**Computation:**

$$
N_{\text{pins,total}} = \left\lfloor \frac{100 \times 10^6}{(2)^2} \right\rfloor = \left\lfloor \frac{100{,}000{,}000}{4} \right\rfloor = 25{,}000{,}000
$$

$$
N_{\text{pins,data}} = \lfloor 25{,}000{,}000 \times 0.4 \rfloor = 10{,}000{,}000
$$

$$
BW_{\text{die}} = 10{,}000{,}000 \times \frac{8}{8} = 10{,}000{,}000 \;\text{GB/s} \cdot 10^{-3} = 10{,}000 \;\text{GB/s} = 10 \;\text{TB/s}
$$

**Results:**

$$
BW_{\text{conservative}} = 10 \;\text{TB/s} \quad (1\text{ interface, regardless of }N_{\text{dies}})
$$

$$
BW_{\text{optimistic}} = 4 \times 10 = 40 \;\text{TB/s} \quad (4\text{ independent interfaces})
$$

## 3.3 Scenario 3 — Aggressive Hybrid Bonding ($p_{HB}$ = 0.9 µm)

**Parameters:**

| Parameter | Value |
|-----------|-------|
| $A_{\text{die}}$ | 100 mm² |
| $p_{HB}$ | 0.9 µm |
| $\eta_{\text{data}}$ | 0.4 |
| $f_{\text{data}}$ | 8 Gbps per pin |
| $N_{\text{dies}}$ | 4 |

**Computation:**

$$
N_{\text{pins,total}} = \left\lfloor \frac{100 \times 10^6}{(0.9)^2} \right\rfloor = \left\lfloor \frac{100{,}000{,}000}{0.81} \right\rfloor = \left\lfloor 123{,}456{,}790 \right\rfloor = 123{,}456{,}790
$$

$$
N_{\text{pins,data}} = \lfloor 123{,}456{,}790 \times 0.4 \rfloor = 49{,}382{,}716
$$

$$
BW_{\text{die}} = 49{,}382{,}716 \times \frac{8}{8} \approx 49{,}383 \;\text{GB/s} \approx 49 \;\text{TB/s}
$$

**Results:**

$$
BW_{\text{conservative}} \approx 49 \;\text{TB/s} \qquad BW_{\text{optimistic}} \approx 4 \times 49 = 197 \;\text{TB/s}
$$

**Caveats at aggressive pitch.** At 0.9 µm pitch the pin count approaches $1.2 \times 10^8$ per die, and the aggregate I/O power draw ($\sim N_{\text{pins,data}} \times f_{\text{data}} \times E_{\text{bit}}$) may exceed the power budget of the die interface. In practice, $f_{\text{data}}$ may need to drop to 2–4 Gbps per pin to remain within acceptable power density, reducing bandwidth proportionally. Signal integrity and reference voltage distribution at sub-µm pitch are additional practical constraints not captured by this first-principles model.

---

# 4. Latency Model

Shortening the vertical interconnect path in a 3D stack reduces access latency relative to HBM. Following AccelStack §III-C2 [ACCELSTACK]:

$$
\ell_{3D} \approx \frac{\ell_{\text{HBM}}}{k_{\text{interconnect}}}
$$

where $k_{\text{interconnect}} > 1$ is the latency reduction factor attributable to the shorter signal path (no interposer, no package trace, direct vertical bond). Typical HBM read latency is approximately 100 ns at the row-activation level [HBM-SPEC]. With hybrid bonding reducing the vertical path from ~100–200 µm (HBM microbump + interposer) to ~1–5 µm (direct bond), an estimated:

$$
k_{\text{interconnect}} = 2 \text{–} 5 \quad \Longrightarrow \quad \ell_{3D} \approx 20 \text{–} 50 \;\text{ns}
$$

**Impact on LLM decode.** For steady-state decode with large weight tensors and sequential KV cache access, the system is strongly memory-**bandwidth**-bound, not latency-bound. The improved latency $\ell_{3D}$ does not affect $t_{\text{step,user}}$ in the roofline regime (see `decode.md` §4). Latency reduction does matter for workloads with high spatial locality and frequent cache-miss patterns (e.g., sparse retrieval, tree-attention with irregular access), but those are outside the scope of the steady-state decode model.

---

# Symbol Summary

Symbols introduced in this document; consolidated into `notation.md` §15.

| Symbol | Description | Units |
|--------|-------------|-------|
| $A_{\text{die}}$ | DRAM die area | mm² |
| $p_{HB}$ | Hybrid bonding pitch (center-to-center pad spacing) | µm |
| $\eta_{\text{data}}$ | Data pin fraction (fraction of pads carrying data) | dimensionless |
| $f_{\text{data}}$ | Data rate per pin | Gbps |
| $N_{\text{dies}}$ | Number of stacked DRAM dies | integer |
| $N_{\text{pins,total}}$ | Total pad count (area-limited) | integer |
| $N_{\text{pins,data}}$ | Data pad count after overhead allocation | integer |
| $BW_{\text{die}}$ | Raw bandwidth per die interface | GB/s |
| $BW_{\text{conservative}}$ | Lower bound: single logic-facing interface | GB/s |
| $BW_{\text{optimistic}}$ | Upper bound: independent per-die interfaces | GB/s |
| $t_m, t_n, t_k$ | GEMM tile dimensions in AccelStack Eq. 2 (§2.5) | integer |
| $p$ | Systolic array edge (PEs per side) in AccelStack Eq. 2 | integer |
| $k_{\text{interconnect}}$ | Latency reduction factor vs. HBM (§4) | dimensionless |
| $\ell_{3D}$ | Estimated 3D DRAM read latency | ns |

---

# References

- **[ACCELSTACK]** — Primary source for the 3D DRAM interface methodology (§III-C2: 3D DRAM bandwidth from hybrid bonding pitch and pin count). The bandwidth derivation in `dram3d.md` is a first-principles extension; §III-C2 provides the qualitative framework without a dedicated equation. Also source of the "5–30× smaller than microbumps" qualitative ratio used in §1.
- **[LAU-PKG]** — Lau 2021, *J. Microelectronics and Electronic Packaging*. Primary quantitative anchor for the §1 pitch table: Intel Foveros Direct (2020) "bumpless pad pitch reduces from 50 µm (for microbumps) to 10 µm; density rises from 400 bumps/mm² to 10,000 pads/mm²".
- **[SOIC-UHD]** — Chen et al., ECTC 2020. Sub-micron pitch hybrid bonding at ≥1.2 M bonds/mm²; upper-bound datapoint for §1 and Scenario 3 (§3.3).
- **[SEMIANALYSIS-HB]** — SemiAnalysis cross-section analysis of Nvidia A100: ~130 µm C4, ~50 µm copper pillars. Industry-press reference for the §1 pitch table.
- **[HBM-SPEC]** — JEDEC JESD235D: HBM2E / HBM3 / HBM3E pin bandwidth and capacity specs. Used for calibration in Scenario 1 (§3.1) and latency baseline in §4.
- **[H100-SPEC]** — NVIDIA H100 whitepaper: 3.35 TB/s HBM3 bandwidth per GPU. Used as the per-device reconciliation reference point in §5.

_Full bibliographic entries for all tags are in `references.md`._
