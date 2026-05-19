# MoE Weight Traffic — The Expected-Touched-Experts Formula

**Author:** Yue Lu
**Date:** May 2026

A first-principles walkthrough of why MoE per-step weight traffic is the **expected number of distinct experts touched**, derived from the routing physics, illustrated with diagrams, with worked numerical examples and comparison to the naive "always load everything" baseline.

---

## 1. The physical question

In a Mixture-of-Experts (MoE) transformer layer, the FFN is replaced by:

```
                     ┌─────────────────────────────────────┐
                     │  Router (a small linear: H → N_exp) │
                     │  picks top-k experts for each token │
                     └──────────────────┬──────────────────┘
                                        │
                                        ▼
        Token → routes to k of N_exp experts → sums their outputs

           Expert pool (on this rank, after EP-sharding):
           ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐
           │ E_1  │ E_2  │ E_3  │ E_4  │ ...  │      │ E_N  │
           └──────┴──────┴──────┴──────┴──────┴──────┴──────┘
                     each holds 3·H·I_moe weights (b bytes each)
```

The per-step **weight traffic** is the number of bytes the device must load from HBM to feed the kernels that run this step. The question is:

> **Out of the $N$ expert weight blocks resident on this rank, how many actually need to be loaded from HBM in this step?**

This number can be much smaller than $N$ when the batch $B$ is small (few tokens, few experts touched), and approaches $N$ when $B$ is large (everyone gets touched).

---

## 2. How MoE routing works (the mechanism)

Before we can answer the traffic question, we need to be precise about how MoE decides which experts a token visits. A common point of confusion is whether MoE sends a token to **all** experts and post-selects the top-k results, or whether the **router decides top-k first** and only those experts ever compute.

The answer is the latter: **the router selects top-k BEFORE the experts run, and only those k experts compute.** This is the whole point of MoE — sparse activation. If we ran all $N$ experts and picked top-k results, MoE would be strictly worse than dense (same compute, more parameters).

### 2.1 The forward pass — one token's trip through an MoE layer

```
   Input hidden state h ∈ ℝ^H    (post-attention output for this token)
       │
       ▼
   ┌────────────────────────────────────────────────────────────┐
   │  STEP 1: ROUTER                                            │
   │  A small linear layer:  scores = h · W_g                   │
   │  W_g ∈ ℝ^(H × N_exp) → outputs N_exp scalar scores         │
   │                                                            │
   │  Cost: 2·H·N_exp FLOPs   ← cheap, much smaller than        │
   │                            one expert FFN (6·H·I_moe)      │
   └─────────────────────────┬──────────────────────────────────┘
                             │
                             ▼  scores = [s_1, s_2, ..., s_N]
   ┌────────────────────────────────────────────────────────────┐
   │  STEP 2: TOP-K SELECTION                                   │
   │  Pick the k indices with largest scores (e.g., {17, 94})   │
   │  Softmax over those k → mixing weights (w_17, w_94)        │
   │                                                            │
   │  Cost: O(N_exp) — tiny                                     │
   └─────────────────────────┬──────────────────────────────────┘
                             │
                             ▼  "this token goes to E_17 and E_94 only"
   ┌────────────────────────────────────────────────────────────┐
   │  STEP 3: DISPATCH (all-to-all if EP > 1)                   │
   │  Send the token's activation to the ranks holding E_17     │
   │  and E_94. Other N - k experts NEVER see this token.       │
   └─────────────────────────┬──────────────────────────────────┘
                             │
                  ┌──────────┴───────────┐
                  ▼                      ▼
   ┌────────────────────────┐  ┌────────────────────────┐
   │ STEP 4a: E_17          │  │ STEP 4b: E_94          │
   │ Expert FFN runs        │  │ Expert FFN runs        │
   │ (3 GEMVs, H × I_moe)   │  │ (3 GEMVs, H × I_moe)   │
   │                        │  │                        │
   │ HBM load: E_17 weights │  │ HBM load: E_94 weights │
   │           = 3·H·I_moe·b│  │           = 3·H·I_moe·b│
   └────────────┬───────────┘  └────────────┬───────────┘
                │                           │
                │  output_17                │  output_94
                ▼                           ▼
   ┌────────────────────────────────────────────────────────────┐
   │  STEP 5: COMBINE (weighted sum)                            │
   │  h_out = w_17 · output_17 + w_94 · output_94               │
   │  Mixing weights are the softmaxed top-k scores from step 2 │
   └────────────────────────────────────────────────────────────┘
                             │
                             ▼
                          Output h_out (back into the residual stream)

   Experts 1, 2, 3, ..., 16, 18, 19, ..., 93, 95, ..., 128
                  ← NEVER ran for this token. Their weights stayed
                    cold in HBM (assuming no other token wanted them).
```

This is the **conditional computation** pattern: a tiny gating network (the router) decides which big experts to invoke. The savings vs. dense:

| Quantity | Dense FFN | MoE top-k (k=2, N=128) |
|---|---|---|
| FFN parameters resident | $3HI_{\text{dense}}$ | $128 \cdot 3HI_{\text{moe}}$ (more params, supports sparse activation) |
| FFN compute per token | $6HI_{\text{dense}}$ | $k \cdot 6HI_{\text{moe}} = 12HI_{\text{moe}}$ (≈constant despite 128× more params) |
| Expert weights touched per token | All of them | $k = 2$ only |

### 2.2 How does the router decide?

The router is a **learned linear layer** ($W_g \in \mathbb{R}^{H \times N_{\text{exp}}}$). Its decision for token $h$ comes from:

$$s = h \cdot W_g \qquad \mathcal{I}_k = \text{topk}(s, k) \qquad w_i = \frac{\exp(s_i)}{\sum_{j \in \mathcal{I}_k} \exp(s_j)}$$

The score $s_i = h \cdot W_g[:, i]$ is the dot product between the token's hidden state and the $i$-th column of $W_g$. Each column acts as a learned **"expert preference vector"** — the dot product is large when $h$ matches that vector's direction.

There's no built-in semantic meaning — the router doesn't know "this is a science question, send to expert 17". The router weight $W_g$ is **trained jointly with the experts**: gradients flow back through the softmax weights $w_i$ → into the scores $s_i$ → into $W_g$. Over training, this drives two emergent behaviors:

- **Specialization**: each expert learns to be good at a subset of token patterns; the router learns to send those patterns to that expert.
- **Differentiation**: if two experts initially behave the same, gradient pressure spreads them apart so the router can usefully distinguish them.

Crucially, the **top-k operation is non-differentiable** — sorting has no gradient. So gradients only flow through the chosen $k$ experts and their softmax weights; the other $N - k$ experts get no gradient for this token. This makes training delicate: an expert that's never picked never gets updated and stays at its initialization, leading to the **dead-expert problem** discussed next.

### 2.3 The dead-expert problem and the load-balancing fix

Without intervention, MoE training collapses: a few "winner" experts get picked for most tokens, get trained more, become better, and dominate routing further. The remaining experts atrophy. You end up with effectively a dense model that wastes parameter capacity.

The standard fix is an **auxiliary load-balancing loss** [SWITCH, GSHARD]:

$$\mathcal{L_{\text{aux}}} = N_{\text{exp}} \cdot \sum_{i=1}^{N_{\text{exp}}} f_i \cdot P_i$$

where $f_i$ is the fraction of tokens routed to expert $i$ in this batch (hard count after top-k) and $P_i$ is the mean routing probability for expert $i$ (soft, averaged over batch). This loss is minimized when both $f_i$ and $P_i$ are uniform across experts (each gets $1/N$ share). Adding it to the main loss pushes the router to spread tokens across experts during training.

Variations seen in production:

- **Switch Transformer** [SWITCH]: top-1 routing + aux loss, plus a "capacity factor" hard cap (each expert can take at most `capacity_factor · B / N` tokens; overflow is dropped).
- **Mixtral** [MIXTRAL] / **DeepSeek-V3** [DSV3]: top-k (k=2 or 8) with aux loss; DeepSeek-V3 adds bias-based load balancing where per-expert biases are updated per step to encourage uniformity without affecting gradients.
- **Expert-choice** [EXPERTCHOICE]: inverts the matching — experts pick the top-c tokens they want, instead of tokens picking experts. Guarantees perfect load balance by construction.

### 2.4 What trained routers actually do (empirical findings)

For a language model with $\sim$128 experts trained on diverse data, what does the router learn? Empirical analyses [MIXTRAL-ANALYSIS, SWITCHANAL]:

- Routes are **not** cleanly aligned with high-level topics (no "math expert" or "code expert" pattern in interpretable units).
- Routes correlate weakly with **syntactic features** (POS tags, punctuation, position in sequence).
- Different layers route differently — early layers tend toward more uniform routing, later layers more specialized.
- Routing decisions are partially **token-identity-driven**: certain rare tokens or special characters consistently go to particular experts.

So the experts don't end up as crisply interpretable units. They're more like a learned soft clustering of input patterns that, combined with the load-balancing pressure, fills the capacity of all $N$ experts.

### 2.5 Why this matters for the traffic formula

The router's decisions are what determine **which experts get touched per step**. The traffic formula in §5-§6 assumes those decisions are **uniformly random across experts** (each token picks each expert with probability $1/N$). The above context determines how good that assumption is:

| Routing regime | Effect on the expectation formula |
|---|---|
| **Without load balancing** | Routing concentrates on winner experts → real touched count is **smaller** than the formula predicts (the same few experts hit every step). Formula over-estimates traffic. The bigger problem is throughput: winner experts become bottlenecks. |
| **With load balancing (typical production)** | Routing is close to uniform → formula is close to accurate. Slight deviation: real routing tends to be **tighter** than uniform (load balancing pushes assignment more evenly than pure random), so the formula very slightly over-estimates touched count and traffic. Conservative direction. |
| **Expert-choice (hard balanced)** | Perfect uniformity → formula's saturated regime kicks in exactly at $t = N$ (the knee is sharp, not soft — see §11). |

For most production systems with load-balancing aux loss, **the uniform-routing formula is a reasonable first-order model**. It's the analytically tractable middle ground between "no balancing" and "perfect balancing".

---

## 3. Why the naive "always load everything" answer is wrong at small B

The simplest model is:

$$T_\theta^{\text{moe, naive}} = M_\theta^{\text{moe}} = \frac{L_{\text{moe}}}{PP} \cdot \frac{N_{\text{exp}}}{D_{\text{exp}}} \cdot 3HI_{\text{moe}} \cdot b$$

i.e., load **all** experts (the full per-rank footprint) every step. This assumes every expert is read into the compute units every step, regardless of how many tokens actually need it.

**Why is this wrong at small B?** Consider $B = 1$ (one active sequence) with $k = 2$ (top-2 routing) and $N = 128$ experts:

```
            B=1 token, k=2 active experts:

            Token → picks E_17 and E_94

            Expert pool:
            ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
            │  │  │  │  │  │  │  │  │  │  │  │   <- 128 experts
            └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
                          ▲              ▲
                          │              │
                          │      ════════╣ E_94 loaded
                  ════════╣ E_17 loaded

            Number of experts loaded: 2 (out of 128)
            Number of experts UNTOUCHED: 126 — their weights stay cold in HBM
```

Reading 126 idle experts every step would be wasteful, and the hardware doesn't do it — production MoE kernels only load the experts the routing decision selected. The naive "always load $M_\theta^{\text{moe}}$" model over-counts traffic by ~64× in this scenario, predicting an unnecessarily slow TPOT.

At large B, every expert ends up touched and the naive answer becomes correct (you really do load all 128). The interesting modeling question is the **transition** between the two regimes.

---

## 4. Per-token vs total-batch expert loads (resolving a common confusion)

A common point of confusion: "if each token always uses $k$ experts, doesn't that mean we always load $k$ experts even at B=1?" Yes — and the formula handles this correctly. The subtlety is separating two scaling effects that are easy to conflate:

| Effect | Driver | How it scales |
|---|---|---|
| Per-token compute work | $k$ experts per token (fixed by router config) | Constant in B (each token always does $k$ expert GEMVs) |
| **Number of distinct experts loaded** from HBM | Union of touched experts across the batch | Grows with $t = B \cdot k$, saturating at $N$ |

The first effect is **per-(token, expert) compute** — already covered in `decode.md §3.3`. Compute is **independent**: two tokens hitting the same expert do two GEMVs, no sharing. So compute scales as $B \cdot k$ exactly.

The second effect is **traffic**, the topic of this document. Traffic is **shared**: two tokens hitting the same expert pay for that expert's weight load **once**. The set of weights HBM has to deliver is the **union** of all touched experts — and the union grows sub-linearly in B because of collisions.

### Worked example: B=4 tokens, k=2, N=128

```
        The k=2 routing decisions (each token picks top-2 experts):

   Token 1 → picks E_17 and E_94
   Token 2 → picks E_94 and E_3      ←━ E_94 already requested by Token 1
   Token 3 → picks E_50 and E_22
   Token 4 → picks E_22 and E_60     ←━ E_22 already requested by Token 3

   Total token-expert ASSIGNMENTS:  4 tokens × 2 experts = 8 assignments  (= t = B·k)
   Total distinct EXPERTS to load:  {3, 17, 22, 50, 60, 94}  =  6 distinct
                                    (2 collisions: E_94 hit ×2, E_22 hit ×2)
```

So the **"ball" in the balls-and-bins model is one (token, expert) assignment, NOT one token**. For B tokens with k experts each, we throw $t = B \cdot k$ balls. The formula then asks: how many distinct bins (experts) catch at least one ball?

### How the touched set grows with B (visual)

```
   B=1 token  (t = 2 assignments):    each token always picks k=2 experts,
                                       so even at B=1 we load 2 experts
                                       (not 1, not 128 — exactly k)
   bins:  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐...
          │  │  │██│  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │██│ ← ×128
          └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
          2 distinct loaded.  126 cold.  Traffic = 2/128 = 1.6% of M_θ^moe

   B=4 tokens  (t = 8 assignments):   some collisions; touched grows
                                       almost linearly
   bins:  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
          │  │  │██│  │██│  │  │  │██│  │██│  │  │  │  │  │██│██│██│
          └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
          ~7 distinct loaded.  Formula gives E[T] ≈ 7.8 (some collisions).

   B=64 tokens  (t = 128 assignments):  many collisions; saturation knee
                                         (t/N = 1 → ~63% touched)
   bins:  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
          │██│██│██│  │██│██│██│██│██│██│██│  │██│██│██│██│██│██│██│
          └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
          ~81/128 distinct.  Average ~1.6 assignments per touched bin.
          Some bins hit 3-4 times, some still empty.

   B=256 tokens  (t = 512 assignments):  saturated; nearly all experts hit
   bins:  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
          │██│██│██│██│██│██│██│██│██│██│██│██│██│██│██│██│██│██│██│
          └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
          ~126/128 distinct.  Traffic = 98% of M_θ^moe — near saturation.
```

### The crucial physics

When two tokens both want expert $E_{94}$, the GPU kernel **loads $E_{94}$'s weights from HBM exactly once**. Both tokens then compute against the in-cache weights — no second HBM read. The traffic cost is paid per **distinct touched expert**, not per token-expert assignment.

That's why "savings" exist even though each token uses $k$ experts:
- Small B: few token-expert assignments → mostly distinct → load count ≈ $t = B \cdot k$ → traffic linear in B
- Large B: many token-expert assignments → heavy collisions → load count ≈ $N$ (saturated) → traffic constant

### Quick sanity checks against the formula

For $N = 128$ experts, $k = 2$:

| B | $t = B \cdot k$ | $\mathbb{E}[T]$ formula | Interpretation |
|---|---|---|---|
| 1 | 2 | $128 \cdot (1 - (127/128)^2) \approx 2.0$ | Always load exactly **k=2** experts at B=1 (no collisions possible with only 2 throws) ✓ |
| 4 | 8 | $\approx 7.8$ | 8 throws, ~0.2 expected collisions → 7.8 distinct |
| 64 | 128 | $\approx 81$ | At the knee: $t/N = 1$, hit 63% of experts ($1 - 1/e$) |
| 256 | 512 | $\approx 126$ | Saturated — 4× more throws than bins, only ~2/128 untouched |

The formula correctly accounts for the per-token $k$ factor (it's already in $t = B \cdot k$). What it adds is the **collision math** — counting *unique* loads instead of total assignments.

---

## 5. Setup: balls and bins

Map the routing process to a classical probability setup:

| Routing concept | Balls-and-bins concept |
|---|---|
| $N_{\text{per rank}}$ experts on this rank | $N$ bins |
| One token-expert assignment (1 token picks 1 expert) | One ball thrown into a bin |
| $t = B \cdot k / D_{\text{exp}}$ total assignments to this rank per step | $t$ balls thrown |
| Number of distinct experts that received ≥1 assignment | Number of distinct non-empty bins |

**Uniform routing assumption**: under load-balancing-aware training, each token independently picks each expert with equal probability $1/N$. Each ball lands in any bin with equal probability $1/N$, independently of other balls.

This is an approximation — real production routers have some non-uniformity (load-balancing-loss anti-correlation, expert affinity for certain token types). But it's tractable analytically and a reasonable first-order model.

```
                The setup:

                N bins (= experts on this rank)
        ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
        │  │  │  │  │  │  │  │  │  │  │  │
        └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

        t balls (= token-expert assignments) thrown
        uniformly at random into the bins, INDEPENDENTLY

           ⚫            ⚫
              ⚫           ⚫⚫
                   ⚫⚫       ⚫
        ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
        │⚫│  │  │⚫│  │⚫⚫│⚫│  │⚫│⚫│  │
        └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
         ▲           ▲     ▲▲▲   ▲     ▲  ▲
         │           │     │     │     │  │
         touched    untouched   "touched" but multiple
                                ━━━━ ONE weight load serves all

        Want: E[number of distinct bins that received ≥1 ball]
```

---

## 6. Derivation

Use **linearity of expectation** — a beautifully simple trick that turns a hard combinatorial question into a small sum of per-bin probabilities.

Define indicator random variables:

$$X_i = \begin{cases} 1 & \text{if bin } i \text{ received at least one ball} \\ 0 & \text{otherwise} \end{cases}$$

Then the total number of distinct touched bins is:

$$T = \sum_{i=1}^{N} X_i$$

And by linearity of expectation (which works regardless of whether the $X_i$ are independent):

$$\mathbb{E}[T] = \sum_{i=1}^{N} \mathbb{E}[X_i] = N \cdot \mathbb{E}[X_1]$$

(by symmetry — all bins are interchangeable, so $\mathbb{E}[X_i]$ is the same for every $i$).

Now compute $\mathbb{E}[X_1] = P(X_1 = 1) = P(\text{bin } 1 \text{ is touched})$.

It's easier to compute the **complementary** probability — the probability that bin 1 is **NOT** touched (received zero balls). Each ball independently lands in bin 1 with probability $1/N$, and misses bin 1 with probability $1 - 1/N$. With $t$ independent throws:

$$P(\text{bin 1 NOT touched}) = \left(1 - \frac{1}{N}\right)^t$$

```
                The complementary calculation:

        Throw 1: misses bin 1 with prob (N-1)/N
        Throw 2: misses bin 1 with prob (N-1)/N
        Throw 3: misses bin 1 with prob (N-1)/N
                ...
        Throw t: misses bin 1 with prob (N-1)/N

        All t throws miss bin 1 (independently):
        prob = ((N-1)/N)^t = (1 - 1/N)^t

        So: P(bin 1 IS touched) = 1 - (1 - 1/N)^t
```

Putting it together:

$$\boxed{\mathbb{E}[T] = N \cdot \left(1 - \left(1 - \frac{1}{N}\right)^t\right)}$$

This is the **expected number of distinct touched experts** on a rank that holds $N$ experts and receives $t$ token-expert assignments per step.

In the framework code (`weight_quantities.py:moe_weight_traffic_bytes`):

```python
N_per_rank = N_exp / D_exp_moe
t_per_rank = B * k_active / D_exp_moe
E_touched = N_per_rank * (1 - (1 - 1/N_per_rank) ** t_per_rank)
T_theta_moe = (L_moe / PP) * (attn_per_device + 3*H*I_moe * E_touched) * b
```

---

## 7. Two limiting regimes

The formula has two characteristic behaviors depending on the ratio $t/N$ — i.e., the number of assignments per expert.

### 7.1 Sparse regime ($t \ll N$): linear in B

When you have far fewer balls than bins, almost every ball lands in an empty bin. Taylor-expand $(1 - 1/N)^t$ around small $t/N$:

$$\left(1 - \frac{1}{N}\right)^t \approx 1 - \frac{t}{N} + O\!\left(\frac{t^2}{N^2}\right)$$

So:

$$\mathbb{E}[T] \approx N \cdot \frac{t}{N} = t$$

In other words: **at small $t$, you touch approximately $t$ experts** (each token-expert pair lands in a different expert with high probability). Traffic grows **linearly in B** (since $t \propto B$).

```
        Sparse regime (t = 4, N = 16):
              ⚫        ⚫            ⚫
                                ⚫
        ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
        │⚫│  │  │  │  │⚫│  │  │⚫│⚫│  │  │  │  │  │  │
        └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
         ▲           ▲           ▲   ▲
         All 4 balls in distinct bins  → 4 experts touched, ~= t
         12 experts untouched          → their weights stay cold
```

Why physically? When you batch only a few tokens, the routing decisions almost certainly pick distinct experts. The system reads only those few experts' weights from HBM, leaving the rest cold.

### 7.2 Saturated regime ($t \gg N$): asymptotes to N

When you have far more balls than bins, every bin gets multiple balls. The probability that any specific bin gets missed shrinks exponentially:

$$\left(1 - \frac{1}{N}\right)^t \to 0 \quad \text{as } t \to \infty$$

So:

$$\mathbb{E}[T] \to N$$

**At large $t$, you touch essentially all $N$ experts.** Traffic **saturates at the full per-rank MoE footprint** = $N \cdot 3HI_{\text{moe}} \cdot b$ bytes.

```
        Saturated regime (t = 64, N = 16):
        ⚫⚫⚫⚫ ⚫⚫⚫  ⚫⚫⚫⚫⚫ ⚫⚫⚫⚫⚫⚫ ⚫⚫⚫⚫⚫⚫⚫⚫⚫
         ⚫⚫⚫⚫    ⚫⚫⚫⚫⚫    ⚫⚫⚫     ⚫⚫⚫⚫⚫⚫⚫⚫
        ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
        │⚫⚫│⚫⚫│⚫⚫│⚫⚫│⚫⚫│⚫⚫│⚫⚫│⚫⚫│⚫⚫│⚫⚫│⚫⚫│⚫⚫│⚫⚫│⚫⚫│⚫⚫│⚫⚫│
        └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
         Every bin touched at least once → 16 experts touched ≈ N
         Extra balls don't add new expert loads → traffic saturated
```

Why physically? When you batch many tokens, the law of large numbers takes over. With high probability every expert gets at least one token, so every expert's weights must be loaded. Beyond saturation, adding more tokens doesn't add new weight loads — they reuse the experts already loaded. Total per-step weight traffic is **constant in B** past this point.

### 7.3 The crossover

The transition between the two regimes happens around $t \sim N$. More precisely, the curve "knees" around $t \approx N$ where the expected loading per bin equals 1. Below that, linear; above, saturating.

For the per-rank quantities under uniform routing:
- $N_{\text{per rank}} = N_{\text{exp}} / D_{\text{exp}}$
- $t_{\text{per rank}} = B \cdot k / D_{\text{exp}}$

Setting $t_{\text{per rank}} = N_{\text{per rank}}$ gives the per-rank crossover at:

$$B^{\text{moe-knee}} = \frac{N_{\text{exp}}}{k}$$

(independent of $D_{\text{exp}}$! both numerator and denominator scale the same way with EP — the per-rank knee always sits at $B \cdot k = N_{\text{exp}}$ globally).

---

## 8. The expected-touched curve

Plotting $\mathbb{E}[T] / N$ vs $t/N$ shows the universal shape:

```
          1.0 ┤ saturation asymptote: E[T] → N as t → ∞
              │              ◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦
              │       ◦◦◦◦◦◦◦
              │   ◦◦◦◦                ← KNEE around t/N ≈ 1
   E[T]/N    │ ◦◦
              │◦
          0.5 ┤◦         "soft" transition: smooth curve, no
              ◦          sharp kink — pure exponential decay
              ◦          of the miss probability
              ◦
              │◦         sparse-regime tangent: y = t/N (linear)
          0.0 │◦
              └─────────────────────────────────────────────────
              0          1            2            3            4
                                    t / N

      Specific values:
        t/N = 0.1   →  E[T]/N ≈ 0.095   (very close to t/N: linear)
        t/N = 0.5   →  E[T]/N ≈ 0.393
        t/N = 1.0   →  E[T]/N ≈ 0.632   (the knee — 1 - 1/e)
        t/N = 2.0   →  E[T]/N ≈ 0.865
        t/N = 3.0   →  E[T]/N ≈ 0.950
        t/N = 5.0   →  E[T]/N ≈ 0.993
```

This is the standard "coupon collector" curve. At $t/N = 1$, you've touched $1 - 1/e \approx 63\%$ of bins. To touch 95% of bins, you need $t/N \approx 3$. To touch 99.3%, $t/N \approx 5$.

---

## 9. Worked example: GPT-1.8T MoE on a single rank

Take a realistic MoE config — **$N_{\text{exp}} = 128$**, **$k = 2$** (top-2 routing), **EP = 1** (no expert parallelism, so $N_{\text{per rank}} = 128$ and $t_{\text{per rank}} = 2B$).

The per-rank touched-expert count and the resulting weight traffic (relative to the full $M_\theta^{\text{moe}}$ footprint):

| Batch B | $t = 2B$ | $\mathbb{E}[T] = 128 \cdot (1 - (127/128)^{2B})$ | $\mathbb{E}[T]/128$ | Traffic / $M_\theta^{\text{moe}}$ |
|---:|---:|---:|---:|---:|
| 1 | 2 | 2.0 | 1.6% | **1.6%** ← only 2/128 |
| 2 | 4 | 4.0 | 3.1% | 3.1% |
| 4 | 8 | 7.8 | 6.1% | 6.1% |
| 16 | 32 | 28.6 | 22% | 22% |
| 32 | 64 | 50.6 | 40% | 40% |
| 64 | 128 | 81.0 | 63% | **63%** ← the knee |
| 128 | 256 | 110.6 | 86% | 86% |
| 256 | 512 | 125.7 | 98% | 98% |
| 512 | 1024 | 127.96 | 99.97% | **≈100%** ← saturated |
| 1024 | 2048 | 127.99996 | ≈100% | 100% |

Key observations:
- At **B=1**, only 2 out of 128 experts are touched — traffic is **64× less** than the naive "always load $M_\theta^{\text{moe}}$" answer.
- At **B=64**, we're at the knee — 63% of experts touched. Traffic is roughly linear-to-sub-linear in B up to here.
- By **B=256**, we're 98% saturated. Past this, MoE weight traffic is essentially constant in B.

**This single curve explains the Pareto-frontier kink visible in the `pareto_basic` notebook:**
- The high-interactivity region (small B) benefits dramatically from the expectation formula — TPOT predictions drop because most weights aren't actually loaded.
- The compute-bound region (large B) is identical to the naive answer — every expert is touched, both formulas agree.
- The **transition between the two regimes is the visible kink**, and its location is the MoE saturation knee at $B \cdot k \approx N_{\text{exp}}$ ≈ B=64 for this config.

---

## 10. Expectation curve vs naive footprint, side-by-side

```
   T_θ^moe                                   M_θ^moe (full footprint)
       │                                       ━━━━━━━━━━━━━━━━━━━━━━━━━
       │  Naive:           ━━━━━━━━━━━━━━━━━━━━ "always load everything"
       │  always = M_θ     ↑                    (correct only at saturation)
       │                   │
       │                   │
       │                   │
       │  Expectation      │
       │  formula          │      ◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦  ← saturates
       │  N·(1−(1−1/N)^t): │ ◦◦◦◦◦                            here at M_θ
       │                   ◦◦                ↑
       │              ◦◦◦◦   ◦              the knee
       │         ◦◦◦◦  ◦                    around B·k ≈ N_exp
       │     ◦◦◦◦
       │ ◦◦◦◦                ← linear-ish at small B (touched ≈ t)
       │◦      ← naive over-counts by N/touched ≈ 64× at B=1 in this config
       │
       └─────────────────────────────────────────────────────────────
       0                                              B (active sequences)
       └────────────┬────────────┴───────────────────┘
              sparse / linear      saturated / constant
```

The visible Pareto kink is exactly this regime change. Under the expectation formula, traffic ramps up from $\sim 0$ to $M_\theta^{\text{moe}}$ across $B \in [1, \sim N_{\text{exp}}/k]$ — that ramp creates a visible curvature in TPOT (and hence in the Pareto frontier). Under the naive footprint model, traffic is flat at $M_\theta^{\text{moe}}$ for every B and TPOT has no MoE-driven knee — a smoother but physically incorrect curve.

---

## 11. Comparison to the load-balanced extreme

The formula assumes **independent uniform routing**. The other extreme is **perfectly load-balanced routing** — every expert receives exactly $\lceil t / N \rceil$ tokens (or as close as possible). This is what an idealized router would do if it could see all $t$ assignments and partition them optimally.

Load-balanced touched count:
- If $t < N$: touched = $t$ (each token to a distinct expert, no collisions)
- If $t \geq N$: touched = $N$ (all experts loaded, hard saturation)

Comparing the two curves:

```
   E[touched] / N
       │
       │     load-balanced:  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1.0 ┤              ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       │           ◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦◦
       │       ◦◦◦◦       uniform (this formula): smooth, asymptotic
       │     ◦◦              load-balanced: piecewise linear,
   0.5 ┤    ◦                                 hard knee at t/N = 1
       │   ◦
       │  ◦  ← both curves identical in the sparse regime
       │ ◦      (t small ⇒ collision-free, hits every distinct bin)
       │◦
   0.0 │◦
       └─────────────────────────────────────────────────────────────
       0           1           2           3           4           5
                                     t / N
```

**The two curves match in the sparse regime** (no collisions either way) and **converge at large t** (saturation). They differ only in the transition zone — uniform has a smooth knee around $t/N = 1$; load-balanced has a sharp corner.

Real production routers sit between these two curves, biased toward load-balanced by the auxiliary loss but not perfectly so. The uniform formula is therefore a **slight over-estimate of touched experts** in the transition zone (and hence a slight over-estimate of traffic), which is the conservative direction for the framework — overpredicting TPOT, not underpredicting.

If you wanted to model the load-balanced case explicitly, replace the expectation with $\min(t, N)$. Most practical scenarios live closer to uniform because batched routing rarely achieves perfect load balancing on every step.

---

## 12. Summary cheat-sheet

| Question | Answer |
|---|---|
| What does the formula compute? | The expected number of distinct experts touched per rank per step under uniform routing |
| Formula? | $\mathbb{E}[T] = N \cdot (1 - (1 - 1/N)^t)$ where $N = N_{\text{exp}} / D_{\text{exp}}$, $t = B \cdot k / D_{\text{exp}}$ |
| Sparse limit ($t \ll N$)? | $\mathbb{E}[T] \approx t$ — traffic linear in B |
| Saturated limit ($t \gg N$)? | $\mathbb{E}[T] \to N$ — traffic constant at $M_\theta^{\text{moe}}$ |
| Crossover (knee) location? | $t = N$, i.e., $B \cdot k \approx N_{\text{exp}}$ globally |
| Touched at $t = N$? | $N \cdot (1 - 1/e) \approx 0.632 N$ |
| Why does compute NOT need this? | Compute per token is **independent** (each token-expert pair does its own GEMV); traffic is **shared** (one HBM read per expert serves all tokens hitting it). See `decode.md §3.3` note. |
| Real-world deviation from formula? | Load-balancing loss makes real routing tighter than uniform → uniform formula is a slight over-estimate of touched, hence slight over-estimate of traffic (conservative direction). |
| Where in the code? | `llm_perf/core/primitives/weight_quantities.py:moe_weight_traffic_bytes` |
| Where in the doc? | `decode.md §2.1 MoE weight traffic` |
| Where in validation? | `pareto_basic.ipynb` (visible kink at high-interactivity end of GPT-1.8T MoE Pareto) |
