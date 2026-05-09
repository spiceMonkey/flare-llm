# References

Shared bibliography for all `modeling.*.md` documents.  
Each entry carries a short tag (e.g. `[MEGATRON]`) used for inline citations across the suite.

---

## Parallelism and Distributed Training

**[MEGATRON]**  
Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., & Catanzaro, B. (2019).  
*Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.*  
arXiv:1909.08053.  
→ Column/row-parallel linear layers (TP); pipeline stage scheduling.

**[MEGATRON3]**  
Narayanan, D., Shoeybi, M., Casper, J., LeGresley, P., Patwary, M., Korthikanti, V., Vainbrand, D., Kashinkunti, P., Bernauer, J., Catanzaro, B., Phanishayee, A., & Zaharia, M. (2021).  
*Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.*  
SC '21. arXiv:2104.04473.  
→ Interleaved pipeline schedules; 1F1B; sequence parallelism (SP).

**[DEEPSPEED-MOE]**  
Rajbhandari, S., Li, C., Yao, Z., Zhang, M., Aminabadi, R.Y., Awan, A.A., Rasley, J., & He, Y. (2022).  
*DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale.*  
ICML 2022. arXiv:2201.05596.  
→ Expert parallelism (EP); all-to-all routing; top-k gating FLOPs.

---

## Attention and Memory Efficiency

**[FA1]**  
Dao, T., Fu, D.Y., Ermon, S., Rudra, A., & Ré, C. (2022).  
*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.*  
NeurIPS 2022. arXiv:2205.14135.  
→ Tiled attention; HBM traffic reduction; activation I/O lower bound.

**[FA2]**  
Dao, T. (2023).  
*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.*  
ICLR 2024. arXiv:2307.08691.  
→ Improved parallelism across heads and sequence; GPU utilization analysis.

**[GQA]**  
Ainslie, J., Lee-Thorp, J., de Jong, M., Zelaski, T., Sanghai, S., & Xu, Y. (2023).  
*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.*  
EMNLP 2023. arXiv:2305.13245.  
→ Grouped-query attention; $n_{kv} < n_q$; KV cache size reduction.

**[MQA]**  
Shazeer, N. (2019).  
*Fast Transformer Decoding: One Write-Head is All You Need.*  
arXiv:1911.02150.  
→ Multi-query attention origin; $n_{kv}=1$.

---

## Mixture of Experts

**[SWITCH]**  
Fedus, W., Zoph, B., & Shazeer, N. (2022).  
*Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.*  
JMLR 2022. arXiv:2101.03961.  
→ Top-1 routing; capacity factor; load balancing loss.

**[MIXTRAL]**  
Jiang, A.Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D.S., de las Casas, D., Hanna, E.B., Bressand, F., Lengyel, G., Bour, G., Lample, G., Lavaud, L.R., Saulnier, L., Lachaux, M-A., Stock, P., Subramanian, P., Yang, J., Antoniak, S., Le Scao, T., Gervet, T., Lavril, T., Wang, T., Lacroix, T., & El Sayed, W. (2024).  
*Mixtral of Experts.*  
arXiv:2401.04088.  
→ Sliding-window attention + MoE; top-2 routing in practice.

---

## Roofline and Performance Modeling

**[ROOFLINE]**  
Williams, S., Waterman, A., & Patterson, D. (2009).  
*Roofline: An Insightful Visual Performance Model for Multicore Architectures.*  
CACM 52(4). doi:10.1145/1498765.1498785.  
→ Operational intensity; memory-bound vs compute-bound regimes; ridge point.

**[ALPHA-BETA]**  
Hockney, R. (1994).  
*The Communication Challenge for MPP: Intel Paragon and Meiko CS-2.*  
Parallel Computing 20(3).  
→ α–β latency model for collective communication: $t = \alpha + n/\beta$.

---

## Serving Systems and KV Cache

**[VLLM]**  
Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C.H., Gonzalez, J.E., Zhang, H., & Stoica, I. (2023).  
*Efficient Memory Management for Large Language Model Serving with PagedAttention.*  
SOSP 2023. arXiv:2309.06180.  
→ PagedAttention; block-based KV paging; fragmentation analysis.

**[ORCA]**  
Yu, G.I., Jeong, J.S., Kim, G.W., Kim, S., & Chun, B.G. (2022).  
*Orca: A Distributed Serving System for Transformer-Based Generative Models.*  
OSDI 2022.  
→ Iteration-level continuous batching; per-step admission/eviction; the scheduling pattern that motivates the per-sequence serving runtime overhead in `decode.md §7.2`.

**[ALPASERVE]**  
Li, Z., Zheng, L., Zhong, Y., Liu, V., Sheng, Y., Jin, X., Huang, Y., Chen, Z., Zhang, H., Gonzalez, J.E., & Stoica, I. (2023).  
*AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving.*  
OSDI 2023. arXiv:2302.11665.  
→ Original goodput framing for SLO-bound deep learning serving; statistical multiplexing across replicas; partition selection under latency SLOs.

**[SPLITWISE]**  
Patel, P., Choukse, E., Zhang, C., Shah, A., Goiri, Í., Maleki, S., & Bianchini, R. (2024).  
*Splitwise: Efficient Generative LLM Inference Using Phase Splitting.*  
ISCA 2024. arXiv:2311.18677.  
→ Prefill–decode phase split with hardware-asymmetric clusters; SLO-driven co-design of prefill and decode hardware.

**[SARATHI]**  
Agrawal, A., Panwar, A., Mohan, J., Kwatra, N., Gulavani, B.S., & Ramjee, R. (2023).  
*SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills.*  
arXiv:2308.16369.  
→ Chunked prefill scheduling; head-of-line blocking reduction.

**[DISAGG-PREFILL]**  
Zhong, Y., Liu, S., Chen, J., Hu, J., Zhu, Y., Liu, X., Jin, X., & Zhang, H. (2024).  
*DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving.*  
OSDI 2024. arXiv:2401.09670.  
→ Prefill–decode disaggregation; KV transfer latency between clusters.

**[MOONCAKE]**  
Qin, R., Li, Z., He, W., Zhang, M., Wu, Y., Zheng, W., & Xu, X. / Moonshot AI. (2024).  
*Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving.*  
arXiv:2407.00079.  
→ KV-centric disaggregated serving; layer-wise KV streaming over RDMA; production deployment.

**[DYNAMO]**  
NVIDIA Corporation. (2024–2025).  
*NVIDIA Dynamo: Distributed Inference Serving Framework.*  
NVIDIA technical documentation, https://github.com/ai-dynamo/dynamo.  
→ Production disaggregated inference; layer-wise KV transfer; multi-cluster orchestration.

---

## Hardware Specifications

**[H100-SPEC]**  
NVIDIA Corporation. (2022).  
*NVIDIA H100 Tensor Core GPU Architecture.*  
NVIDIA Whitepaper WP-10792-001.  
→ H100 SXM5: 989 TF/s (bf16 TC), 3.35 TB/s HBM3, 80 GB HBM per GPU; NVLink 4.0 900 GB/s bisection. (HBM3E appears in H200/B200; H100 SXM5 ships with HBM3.)

**[HBM-SPEC]**  
JEDEC Solid State Technology Association. (2023).  
*High Bandwidth Memory (HBM) DRAM Standard JESD235D.*  
→ HBM2E / HBM3 / HBM3E pin bandwidth and capacity specs.

---

## 3D Stacking and Advanced Packaging

**[ACCELSTACK]**  
Bai, Y., et al. (2025).  
*AccelStack: A Co-Design Framework for 3D-Stacked Accelerators.*  
HKUST FACT Lab. arXiv (preprint).  
→ §III-C2: 3D DRAM bandwidth from hybrid bonding pitch and pin count. Eq. 2: two-level tiled-GEMM compute latency — tile size $(t_m, t_n, t_k)$ sets operand reuse and therefore BW demand per compute cycle. §III-C (Fig. 6, p. 4) places "distributed controllers and PHYs, aligned with PEs" on the logic die, and reports 3D hybrid-bonded DRAM access latency "over 2×" better than HBM's ~300 ns. Introduction (p. 2) gives the qualitative ratio: hybrid bonding pitch is "5–30× smaller than that of microbumps" (citing TSMC SoIC [Chen 2019 ECTC]).  
_Note: §III-C2 contains no dedicated BW equation — `dram3d.md` §2 extends it from first principles. `dram3d.md` §2.5 uses Eq. 2 and the PE-aligned-PHY structure of §III-C to argue that 3D stacking brings effective BW closer to the peak ceiling (another 3D DRAM advantage beyond the raw pin × data-rate lift), without introducing a separate formal symbol._

**[LAU-PKG]**  
Lau, J.H. (2021).  
*State-of-the-Art and Outlooks of Chiplets Heterogeneous Integration and Hybrid Bonding.*  
Journal of Microelectronics and Electronic Packaging, 18(4):145–160.  
→ Comprehensive industry review of advanced packaging pitch regimes. Primary datapoint for Intel Foveros Direct (2020): "bumpless pad pitch reduces from **50 µm (for microbumps) to 10 µm**, density increases from 400 bumps/mm² to 10,000 pads/mm²" (p. 149). Covers AMD 3D V-Cache (2021), Xilinx CoWoS, Ponte Vecchio.

**[SOIC-UHD]**  
Chen, M.F., Yang, C.L., et al. (2020).  
*Ultra High Density SoIC with Sub-micron Bond Pitch.*  
IEEE ECTC 2020, pp. 576–581.  
→ Sub-micron pitch hybrid bonding demonstrated at ≥1.2 M bonds/mm². Follows the 2019 foundational SoIC paper (Chen et al., ECTC 2019, pp. 594–599). Used as the near-term-ceiling datapoint for §3 Scenario 3 (aggressive hybrid bonding).

**[SEMIANALYSIS-HB]**  
Patel, D., Nishball, D. (2024).  
*Hybrid Bonding Process Flow — Advanced Packaging Part 5.*  
SemiAnalysis. https://newsletter.semianalysis.com/p/hybrid-bonding-process-flow-advanced  
→ Cross-section analysis of Nvidia A100: **~130 µm C4 flip-chip**, **~50 µm copper pillars (microbumps)**. Characterizes the Sony CIS production regime (~1 µm hybrid bonding). Industry-press reference for the commercial pitch landscape.

---

## Network Topologies and Collective Algorithms

**[PY09]**  
Patarasuk, P., & Yuan, X. (2009).  
*Bandwidth Optimal All-Reduce Algorithms for Clusters of Workstations.*  
Journal of Parallel and Distributed Computing, 69(2):117–124.  
→ Ring all-reduce achieves $2(N-1)/N \cdot M/BW$ bandwidth-optimal bound on any tree-connected fabric. Used for the shipped-primitive cost tables in collectives/02_topology_mapping.md (star ring + torus dim-decomposed ring).

**[CHPV07]**  
Chan, E., Heimlich, M., Purkayastha, A., & van de Geijn, R. (2007).  
*Collective Communication: Theory, Practice, and Experience.*  
Concurrency and Computation: Practice and Experience, 19(13):1749–1783.  
→ Dimension-decomposed all-reduce framework; telescoping derivation of multi-dim ring costs. Used in collectives/02_topology_mapping.md §3 for the dim-decomposed ring AR / AG / RS derivation.

**[SST09]**  
Sanders, P., Speck, J., & Träff, J.L. (2009).  
*Two-Tree Algorithms for Full Bandwidth Broadcast, Reduction and Scan.*  
ICPP 2009, pp. 1–10.  
→ Double binary tree structure with complementary-role pipelining achieving $2(N-1)/N \cdot M/\mathrm{BW}$ bandwidth (matching ring). Canonical citation for the shipped NCCL DBT AR form in collectives/02_topology_mapping.md §2.

**[DEMYST-NCCL]**  
Jeaugey, S., et al. (2025).  
*Demystifying NCCL: An In-Depth Analysis of GPU Communication Protocols and Algorithms.*  
arXiv:2507.04786.  
→ NCCL tuner internals; empirical ring-vs-DBT AR crossover (DBT wins small-$M$, ring wins large-$M$) contrary to the pure-α-β prediction. Supports the `tuner.ar_algorithm` manual-knob design in collectives/02_topology_mapping.md §2.

**[TRN2-ARCH]**  
AWS. (2024).  
*AWS Trainium2 Architecture Overview.*  
AWS Neuron documentation and re:Invent 2024 sessions.  
→ 2D-per-instance + Z-dim NeuronLink torus; 64-chip UltraServer topology. Motivating hardware for the torus primitives in collectives/02_topology_mapping.md §3.

**[NEURON-CC]**  
AWS. (2024–2025).  
*AWS Neuron Collective Communication Library.*  
AWS Neuron documentation, https://awsdocs-neuron.readthedocs-hosted.com/.  
→ Dim-decomposed ring AR / AG / RS as the default kernel on Trainium torus fabrics. Shipped-algorithm citation for collectives/02_topology_mapping.md §3.

**[KDSA08]**  
Kim, J., Dally, W.J., Scott, S., & Abts, D. (2008).  
*Technology-Driven, Highly-Scalable Dragonfly Topology.*  
ISCA 2008, pp. 77–88.  
→ Dragonfly $(p, a, h, g)$ parameterization; minimal adaptive routing (diameter 3) vs. Valiant routing (diameter 5); canonical balanced construction $g = a h + 1$. Foundational dragonfly reference (dragonfly support is currently out of scope in this codebase).

**[JAIN22]**  
Jain, P., et al. (2022).  
*Optimized MPI Collective Algorithms for Dragonfly Topology.*  
ICS 2022.  
→ Hierarchical three-tier AR decomposition for dragonfly; confirms full global-link utilization under uniform admissible routing.

**[SLINGSHOT]**  
De Sensi, D., Di Girolamo, S., McMahon, K., Roweth, D., & Hoefler, T. (2020).  
*An In-Depth Analysis of the Slingshot Interconnect.*  
SC 2020.  
→ Rosetta 64-port dragonfly; measured α-calibration across router/group/global tiers; adaptive-routing effectiveness under real workloads.

**[SWING]**  
Cascagrande, M., De Sensi, D., et al. (2024).  
*Swing: Short-cutting Rings for Higher-Bandwidth Allreduce.*  
arXiv:2401.09356.  
→ Alternative torus AR algorithm that short-cuts non-adjacent rank pairs; not implemented (`torus_algorithm="swing"` raises `NotImplementedError`).

**[HAMMESH]**  
Hoefler, T., Bonato, S., De Sensi, D., Di Girolamo, S., Li, S., Heddes, M., Belk, J., Goel, D., Castro, M., & Scott, S. (2022).  
*HammingMesh: A Network Topology for Large-Scale Deep Learning.*  
SC 2022.  
→ Irregular torus-family topology with improved bisection at fixed link count; alternative tier type, not modeled.

**[TPU-V4]**  
Jouppi, N.P., Kurian, G., Li, S., Ma, P., Nagarajan, R., Nai, L., Patil, N., Subramanian, S., Swing, A., Towles, B., Young, C., Zhou, X., Zhou, Z., & Patterson, D. (2023).  
*TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings.*  
ISCA 2023.  
→ 3D torus with optical circuit switching for slice reconfiguration; twisted-torus 1.63× A2A gain on asymmetric layouts (§V). Motivates the bisection-bound A2A formula and $D_\mathrm{max}$ layout-sensitivity analysis in collectives/02_topology_mapping.md §3.

**[PAARD]**  
*Proximity-Aware All-Reduce on Dragonfly.*  
ISPA 2021.  
→ Topology-aware AR scheduling on dragonfly under multi-job interference.

**[VALIANT81]**  
Valiant, L.G. (1981).  
*A Scheme for Fast Parallel Communication.*  
SIAM Journal on Computing, 11(2):350–361.  
→ Randomized two-hop routing via intermediate node for adversarial-traffic balancing; classical dragonfly worst-case routing fallback.

**[NCCL-TESTS]**  
NVIDIA Corporation. (2020–2025).  
*NCCL Performance Tests.*  
https://github.com/NVIDIA/nccl-tests ; PERFORMANCE.md.  
→ Canonical busbw/algbw measurement methodology for NCCL collectives; published H100/A100 intra-node AR busbw ≈ 360 GB/s against 450 GB/s peak NVLink4 unidirectional. Calibration source for the crossbar $\eta_\beta \approx 0.80$ entry in collectives/05_contention_and_congestion.md §4.

**[NVLINK-SHARP]**  
NVIDIA Corporation. (2023–2024).  
*NVLink SHARP (NVLS) — In-Network All-Reduce Acceleration.*  
GTC talk S62129; NVIDIA NCCL release notes.  
→ NVSwitch-based in-network reduction; AR busbw rises from ~360 to 470–480 GB/s (can exceed raw link rate because data volume halves). Used in collectives/04_in_network_collectives.md for the INC AR cost form and §7.3 for the back-solved $\eta_\beta^\mathrm{INC} \approx 0.52$.

**[SHARP-IB]**  
Graham, R.L., Bureddy, D., Lui, P., Rosenstock, H., Shainer, G., Bloch, G., Goldenerg, D., Dubman, M., Kotchubievsky, S., Koushnir, V., Levi, L., Margolin, A., Ronen, T., Shpiner, A., Wertheim, O., & Zahavi, E. (2016).  
*Scalable Hierarchical Aggregation Protocol (SHArP): A Hardware Architecture for Efficient Data Reduction.*  
COMHPC Workshop at SC16.  
→ Original SHARP specification for InfiniBand; hardware-offloaded reduction trees; reports ~95% of network bandwidth utilization and 2–5× speedup over host-based reduction. Referenced in collectives/04_in_network_collectives.md for the multi-tier scale-out INC AR cost form ($n_\alpha = 2k$, $n_\beta = 1$).

**[FRONTIER-ARCH]**  
Holmen, J. (2024).  
*Frontier Exascale Architecture: AMD MI250X and HPE Slingshot.*  
ATPESC 2024, Argonne National Laboratory Training Track 2 Talk 2.  
https://extremecomputingtraining.anl.gov/wp-content/uploads/sites/96/2024/08/ATPESC-2024-Track-2-Talk-2-Holmen-Frontier-Exascale-Architecture-AMD-MI250x-and-HPE-Slingshot.pdf  
→ Documents Frontier's Slingshot-11 dragonfly structural parameters: 80 groups (74 compute + 5 I/O + 1 management), three-hop minimal routing, and the global-to-injection bandwidth ratio of 57% (total global 270+270 TB/s). Structural calibration reference for the per-tier $\eta_\beta$ cap pattern in `collectives/05_contention_and_congestion.md §4`.

---

## LLM Scaling and Architecture Surveys

**[KAPLAN-SCALING]**  
Kaplan, J., McCandlish, S., Henighan, T., Brown, T.B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020).  
*Scaling Laws for Neural Language Models.*  
arXiv:2001.08361.  
→ FLOPs ≈ 6N per training token; parameter count estimates.

**[CHINCHILLA]**  
Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., Casas, D.D.L., Hendricks, L.A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., Rae, J.W., Vinyals, O., & Sifre, L. (2022).  
*Training Compute-Optimal Large Language Models.*  
NeurIPS 2022. arXiv:2203.15556.  
→ Compute-optimal token/parameter ratio; inference FLOPs context.

---

## Sequence Parallelism

**[RING-ATTN]**  
Liu, H., Zaharia, M., & Abbeel, P. (2023).  
*Ring Attention with Blockwise Transformers for Near-Infinite Context.*  
ICLR 2024. arXiv:2310.01889.  
→ Ring-based KV sharding; pass-KV variant; (SP-1) rotation steps; single-pass streaming (no scatter phase). Basis for SP collective latency equation in §5.4.

**[HUANG-CP-2024]**  
Huang, Y., et al. (2024).  
*Context Parallelism for Scalable Million-Token Inference.*  
arXiv:2411.01783.  
→ Extends Ring Attention with pass-KV and pass-Q variants; latency analysis for decode vs. prefill; used to confirm TP × SP interaction and per-step message size in §5.4.

**[DEEPSPEED-ULYSSES]**  
Rajbhandari, S., et al. (2023).  
*DeepSpeed-Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models.*  
arXiv:2309.14509.  
→ Alternative SP approach using all-to-all (not ring). Bounded by number of attention heads (cannot exceed $n_{kv}$). Contrasted with ring SP in §5.4 to justify ring assumption for large-context inference.

---

## Empirical Constants and Profiling

**[TENSORRT-LLM]**  
NVIDIA Corporation. (2023–2025).  
*TensorRT-LLM: Open-Source Library for Optimized LLM Inference.*  
https://github.com/NVIDIA/TensorRT-LLM  
→ Fused kernel implementations for attention, LayerNorm/RMSNorm, FFN; optimized sampling kernels; CUDA graph integration.

---

## SRAM-Centric Memory Hierarchy

**[GROQ-LPU]**  
Groq, Inc. (2024).  
*LPU Architecture Overview.*  
https://groq.com/lpu-architecture  
→ SRAM-only inference accelerator: **230 MB on-die SRAM at 80 TB/s per chip**, deterministic compiler-scheduled execution. Anchor for the §1.1 motivation and §3.4 numerical example in `sram.md`.

**[GROQ-DEEPDIVE]**  
He, K. (2024).  
*Inside the Groq LPU.*  
https://01.me/en/2024/02/groq/  
→ Llama-2-70B chip-count derivation: 305+ chips for INT8 weights, 576+ once KV cache is included. Used for `sram.md §3.4` deployment math.

**[DMATRIX-HC25]**  
Kennedy, P. / ServeTheHome (2025).  
*d-Matrix Corsair In-Memory Computing For AI Inference at Hot Chips 2025.*  
https://www.servethehome.com/d-matrix-corsair-in-memory-computing-for-ai-inference-at-hot-chips-2025/  
→ Per-card spec: **2 GB SRAM at 150 TB/s + 256 GB LPDDR5 at 400 GB/s**; 8 chiplets per card with 64×64 INT8 (or 64×128 INT4) DIMC matmul tiles; 60k tokens/s @ 1 ms/token on Llama-3-8B (8-card server) and 30k tokens/s @ 2 ms/token on Llama-3-70B (16-card / 128-chiplet rack); two operating modes ("Performance" / "Capacity") via Aviator runtime. Anchor for `sram.md` §1.1, §1.3, §2.2, §3.2–§3.3.

**[CHIPSANDCHEESE-DM]**  
Wong, C. (2024).  
*d-Matrix Corsair: 256GB of LPDDR for AI Models.*  
Chips and Cheese. https://chipsandcheese.com/p/d-matrix-corsair-256gb-of-lpddr-for  
→ Independent walk-through of the per-package SRAM/LPDDR split and Performance/Capacity mode toggle. Cross-confirms the per-card numbers used in `sram.md §3.2–§3.3`.

**[VIKS-SRAM]**  
Bhardwaj, V. (2024).  
*A Close Look at SRAM for Inference in the Age of HBM Supremacy.*  
Vik's Newsletter. https://www.viksnewsletter.com/p/a-close-look-at-sram-for-inference  
→ HBM refresh + bank-conflict overhead (~8% of theoretical peak) versus SRAM at near-peak. Source for the default $\eta_{\beta,\text{HBM}} \approx 0.92$ and $\eta_{\beta,\text{SRAM}} \approx 1.0$ in `sram.md §1.2`.

**[LIMINAL]**  
Diamantopoulos, D., Pothineni, N., et al. (2025).  
*Efficient LLM Inference: Bandwidth, Compute, Synchronization, and Capacity are all you need.*  
arXiv:2507.14397.  
→ Per-token decode roofline $T_{\text{Mem}} = (\text{Batch\_KV\_Bytes} + \text{Model\_Bytes}) / \text{System\_Aggregate\_Bandwidth}$, mean absolute error 7.6% against measured hardware. The multi-tier $t_{\text{mem}}(B)$ in `sram.md §2.1` is a per-tier opening of LIMINAL Eq. 1; single-tier reduction is exact.

**[MIND-GAP]**  
*Mind the Memory Gap: Unveiling GPU Bottlenecks in Large-Batch LLM Inference* (2025).  
arXiv:2503.08311.  
→ Empirical demonstration that large-batch inference remains memory-bound: attention arithmetic intensity stays nearly constant across batch sizes because KV traffic scales linearly with $B$. Used in `sram.md §2.1` to justify the "weights amortize, KV does not" claim.

**[LLM-FLASH]**  
Alizadeh, K., Mirzadeh, I., et al. (2024).  
*LLM in a flash: Efficient Large Language Model Inference with Limited Memory.*  
arXiv:2312.11514.  
→ Three-tier flash → DRAM → compute hierarchy; first-byte latency amortization across larger chunks; sliding-window weight cache. Justifies retaining $\alpha_i$ in the tier spec for small-read regimes (`sram.md §1.2`).

---

## Speculative Decoding and DP-Attention

**[LEVIATHAN]**  
Leviathan, Y., Kalman, M., & Matias, Y. (2023).  
*Fast Inference from Transformers via Speculative Decoding.*  
ICML 2023. arXiv:2211.17192.  
→ Original speculative-decoding paper; truncated geometric expected-acceptance-length derivation (their Eq. 5) consumed by `decode.md §9.2`.

**[EAGLE]**  
Li, Y., Wei, F., Zhang, C., & Zhang, H. (2024).  
*EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.*  
ICML 2024. arXiv:2401.15077.  
→ Feature-level autoregressive draft head; per-token acceptance ≈ 0.75–0.85 at draft depth 4. Source for the calibration ranges in `decode.md §9.5`.

**[MEDUSA]**  
Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J.D., Chen, D., & Dao, T. (2024).  
*Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.*  
arXiv:2401.10774.  
→ Parallel decoding heads (no separate draft model); empirical per-head acceptance ≈ 0.6–0.7 on Vicuna-13B. Source for the cross-architecture acceptance comparison in `decode.md §9.5`.

**[DSV3]**  
DeepSeek-AI, et al. (2024).  
*DeepSeek-V3 Technical Report.*  
arXiv:2412.19437.  
→ Multi-Token Prediction (MTP) head architecture and training; reports per-token acceptance ≈ 0.85–0.90 at draft depth 1 (their §3.2). Also describes the DP-attention + EP production decode deployment (their §4); used in `decode.md §8` and §9.

**[SGLANG-DPATTN]**  
LMSYS (2024). *Achieving Faster Open-Source LLM Inference (DP-Attention Blog Post).*  
https://lmsys.org/blog/  
→ SGLang's `--enable-dp-attention` flag and the AR → AG swap pattern; empirical comm savings on DSv3 / R1 deployments. Cross-confirms the per-layer collective accounting in `decode.md §8.3`.

---

## Original Modeling Contributions

The following constants and models are **original to this document suite** and are not derived from a published paper. They should be marked as such in inline citations:

- **$\rho$ (overlap factor)** — The parameterization $t_{\text{step,user}} = t_{\text{local}} + \max(0, t_{\text{comm}} - \rho \cdot t_{\text{local}})$ is an original formulation introduced in `decode.md`. The concept of compute–communication overlap is discussed in [MEGATRON3] and [DEEPSPEED-MOE] but without this exact model. Use "this work" as the citation.

---

_To cite in a doc: use the tag in brackets, e.g. "per [ROOFLINE], the ridge point is $R_{\text{GPU}} / BW_{\text{mem}}$". Original-to-this-work items use "this work". New references should be added here and tagged consistently._
