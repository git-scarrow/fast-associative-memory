# HCA-DS-6: Radial Decay Function Design Specification

## 1. Executive Summary

This document specifies the radial decay function $\lambda(r, c, N)$ for the Hyperbolic Context Architecture. The decay function addresses the critical gap identified in MT-1 verification: while DS-5's curvature floor is mathematically correct, realistic token embeddings achieve Lorentz radius $\delta \approx 7.6$ (from Poincare $r_p \approx 0.999$ at $c=1.0$), far below the $R_{\max} \approx 16.8$ needed for the privilege guarantee to bind. The decay function acts as a pre-softmax additive penalty on attention logits, indexed by each key token's Lorentz radius, providing an additional $\lambda(\delta)$ nats of effective radial separation that closes the gap between achieved embedding radius and the $\ln(N)$ threshold. Three candidate functional forms are analyzed; the **log-cosh form** $\lambda(r) = \gamma \cdot \ln(\cosh(r))$ is recommended for its curvature-native computation (requires only $\ln(\sqrt{c} \cdot x_0)$, no arcosh), smooth gradient profile, and efficient gap closure ($\gamma \approx 0.6$ suffices for $\alpha_s \geq 1/2$ at $N=128$K). A complementary radial placement mechanism using initialization push plus regularization ensures tokens maintain sufficient separation for the decay to operate effectively.

---

## 2. Formal Definition of $\lambda(r, c, N)$

### 2.1 Problem Statement

From DS-5 D4, the privilege guarantee for the system token $s$ at the origin requires:

$$\alpha_s \geq \frac{1}{1 + (N-1) \cdot \exp(-\delta)}$$

where $\delta$ is the minimum scaled Lorentz distance of any context token from the origin. For $\alpha_s \geq 1/2$, we need $\delta \geq \ln(N)$.

**MT-1 gap:** At $c=1.0$, typical Poincare embeddings at $r_p = 0.999$ map to Lorentz radius $\delta \approx 7.6$, while $\ln(128000) \approx 11.76$. The shortfall is $\Delta = 11.76 - 7.6 = 4.16$ nats.

### 2.2 Decay as Pre-Softmax Bias

Define the decay function $\lambda: \mathbb{R}_{\geq 0} \to \mathbb{R}_{\geq 0}$ as a **pre-softmax additive penalty** on attention logits:

$$\text{score}(q, k_i) = f(q, k_i) - \lambda(r_i)$$

where $f(q, k_i)$ is the base attention score (squared Lorentz inner product per DS-5), and $r_i = D(o, k_i)$ is the scaled Lorentz distance of key token $i$ from the origin.

### 2.3 Effective Separation Theorem

**Theorem.** With decay $\lambda(r)$ satisfying $\lambda(0) = 0$, for query at the origin and $N-1$ context tokens at scaled distance $\geq \delta$ from the origin:

$$\alpha_s \geq \frac{1}{1 + (N-1) \cdot \exp(-\delta - \lambda(\delta) + \lambda(0))} = \frac{1}{1 + (N-1) \cdot \exp(-\delta - \lambda(\delta))}$$

**Proof.** The attention logit for the system token is $f(q, s) - \lambda(0) = f(q, s)$. For context token $t_i$ at distance $r_i \geq \delta$:

$$\text{score}(q, t_i) = f(q, t_i) - \lambda(r_i) \leq -r_i - \lambda(r_i) \leq -\delta - \lambda(\delta)$$

where the first inequality uses $f(q, t_i) \leq -D(q, t_i)$ (the base score is bounded by negative distance for the exponential-of-distance kernel from DS-5 D4), and the second uses monotonicity of $r + \lambda(r)$ for any $\lambda$ with non-negative derivative.

Substituting into the softmax:

$$\alpha_s = \frac{\exp(f(q,s))}{\exp(f(q,s)) + \sum_{i} \exp(\text{score}(q, t_i))} \geq \frac{1}{1 + (N-1) \cdot \exp(-\delta - \lambda(\delta))}$$

$\blacksquare$

### 2.4 Privilege Constraint

For $\alpha_s \geq 1/2$:

$$\boxed{\delta + \lambda(\delta) \geq \ln(N)}$$

This is the **fundamental constraint** on the decay function: the combined natural separation $\delta$ and decay penalty $\lambda(\delta)$ must exceed $\ln(N)$.

**Minimum decay strength.** Given achieved radial separation $\delta$:

$$\lambda_{\min}(\delta, N) = \max(0, \ln(N) - \delta)$$

At $\delta = 7.6$, $N = 128$K: $\lambda_{\min} = 11.76 - 7.6 = 4.16$ nats.

### 2.5 Curvature Dependence

The Lorentz distance of a token with temporal coordinate $x_0$ from the origin is:

$$r = D(o, x) = \operatorname{arcosh}(\sqrt{c} \cdot x_0)$$

The decay function operates on $r$, which depends on $c$ through the distance metric. The maximum achievable radius is $R_{\max}(c) = \operatorname{arcosh}(\sqrt{c} \cdot M)$.

For the recommended log-cosh form (Section 4), $\lambda(r) = \gamma \cdot \ln(\cosh(r)) = \gamma \cdot \ln(\sqrt{c} \cdot x_0)$, making the curvature dependence explicit and computationally free.

---

## 3. Candidate Functional Forms

### 3.1 Form A: Exponential

$$\lambda_A(r) = \lambda_0 \cdot \bigl[\exp(\beta \cdot r / R_{\max}) - 1\bigr]$$

The $-1$ shift ensures $\lambda_A(0) = 0$.

| Property | Value |
|----------|-------|
| **At origin** | $\lambda_A(0) = 0$ |
| **At boundary** | $\lambda_A(R_{\max}) = \lambda_0(e^\beta - 1)$ |
| **Gradient** | $\partial\lambda_A/\partial r = (\lambda_0 \beta / R_{\max}) \exp(\beta r / R_{\max})$ |
| **Gradient at origin** | $\lambda_0 \beta / R_{\max}$ |
| **Second derivative** | $(\lambda_0 \beta^2 / R_{\max}^2) \exp(\beta r / R_{\max}) > 0$ (convex) |
| **Curvature interaction** | Depends on $R_{\max}(c)$; as $c$ changes, $R_{\max}$ changes, modifying the decay profile implicitly |
| **Computational cost** | 1 division + 1 exp per token (moderate; exp is a transcendental op) |

**Failure modes:**
- $\beta > 2$: $\lambda_A(R_{\max})$ grows super-linearly, saturating softmax and creating attention collapse on all non-origin tokens.
- $R_{\max}$ in denominator: if $c$ drops suddenly (curvature floor activates), $R_{\max}$ shrinks, amplifying $\beta r / R_{\max}$ and causing discontinuous jumps in decay strength.
- Gradient at origin is $\lambda_0 \beta / R_{\max} \approx \lambda_0 / 16.8$, which may be too small to differentiate tokens near the origin.

**Gap closure at $\delta = 7.6$, $N = 128$K ($R_{\max} = 16.81$, $\beta = 1$):**

$$\lambda_A(7.6) = \lambda_0 \cdot [\exp(0.452) - 1] = 0.572 \lambda_0$$

Need $\lambda_A(7.6) \geq 4.16$, so $\lambda_0 \geq 7.27$.

### 3.2 Form B: Polynomial

$$\lambda_B(r) = \lambda_0 \cdot (r / R_{\max})^p$$

| Property | Value |
|----------|-------|
| **At origin** | $\lambda_B(0) = 0$ |
| **At boundary** | $\lambda_B(R_{\max}) = \lambda_0$ |
| **Gradient** | $(\lambda_0 p / R_{\max}) \cdot (r / R_{\max})^{p-1}$ |
| **Gradient at origin** | $0$ for $p > 1$; $\lambda_0 / R_{\max}$ for $p = 1$; $\infty$ for $p < 1$ |
| **Curvature interaction** | Same $R_{\max}$ dependence as exponential |
| **Computational cost** | 1 division + 1 power per token ($r^p$ via `torch.pow`, similar cost to exp) |

**Failure modes:**
- $p < 1$: gradient explosion at origin (infinite derivative at $r = 0$).
- $p = 1$ (linear): uniform decay rate, no privilege amplification near boundary.
- $p > 3$: decay is negligible for $r < R_{\max}/2$, then rises sharply — creates a near-binary partition rather than smooth decay.
- Same $R_{\max}$ discontinuity issue as exponential.

**Gap closure at $\delta = 7.6$, $N = 128$K ($p = 2$):**

$$\lambda_B(7.6) = \lambda_0 \cdot (7.6 / 16.81)^2 = 0.204 \lambda_0$$

Need $\lambda_B(7.6) \geq 4.16$, so $\lambda_0 \geq 20.4$. Very large coefficient needed.

### 3.3 Form C: Log-Cosh (Hybrid)

$$\lambda_C(r) = \gamma \cdot \ln(\cosh(r))$$

| Property | Value |
|----------|-------|
| **At origin** | $\lambda_C(0) = \gamma \cdot \ln(1) = 0$ |
| **At boundary** | $\lambda_C(R_{\max}) \approx \gamma \cdot (R_{\max} - \ln 2)$ |
| **Small-$r$ behavior** | $\ln(\cosh(r)) \approx r^2/2$ (quadratic, gentle) |
| **Large-$r$ behavior** | $\ln(\cosh(r)) \approx r - \ln 2$ (linear, matches distance growth) |
| **Gradient** | $\gamma \cdot \tanh(r)$ |
| **Gradient at origin** | $0$ (smooth) |
| **Gradient at boundary** | $\gamma$ (bounded, no explosion) |
| **Second derivative** | $\gamma \cdot \text{sech}^2(r) \geq 0$ (convex, maximum curvature at origin) |
| **Curvature interaction** | $\ln(\cosh(r)) = \ln(\sqrt{c} \cdot x_0)$ — **no $R_{\max}$ dependence** |
| **Computational cost** | 1 multiply + 1 log per token (reading $x_0$ temporal coordinate) |

**Key advantage — curvature-native computation:**

$$\lambda_C(r) = \gamma \cdot \ln(\cosh(r)) = \gamma \cdot \ln\bigl(\sqrt{c} \cdot x_0\bigr) = \gamma \cdot \bigl(\tfrac{1}{2}\ln c + \ln x_0\bigr)$$

No arcosh, no division by $R_{\max}$. The temporal coordinate $x_0$ is already stored in the embedding. The curvature enters through an additive constant $\frac{\gamma}{2} \ln c$ which cancels in the softmax (it is identical for all tokens in the same layer).

Therefore, the **effective per-token computation** reduces to:

$$\lambda_C(r_i) - \lambda_C(0) = \gamma \cdot \ln(x_{0,i} \cdot \sqrt{c}) = \gamma \cdot \ln(x_{0,i}) + \tfrac{\gamma}{2}\ln c$$

The $\frac{\gamma}{2}\ln c$ term is constant across tokens and vanishes in softmax. The per-token cost is **one log operation on the temporal coordinate**.

**Failure modes:**
- $\gamma > 2$: at realistic $\delta = 7.6$, $\lambda_C(7.6) = \gamma \cdot 6.91 \approx 13.8$, making the effective separation $\delta + \lambda = 21.4$. This suppresses attention to all non-origin tokens beyond what is necessary, potentially destroying useful context.
- $\gamma < 0$: reverses privilege (boundary tokens get boosted). Structurally prevented by parameterization (see Section 9).
- **No failure mode from curvature changes**: unlike Forms A and B, log-cosh does not depend on $R_{\max}$, so curvature changes do not cause discontinuous jumps.

**Gap closure at $\delta = 7.6$, $N = 128$K:**

$$\lambda_C(7.6) = \gamma \cdot \ln(\cosh(7.6)) = \gamma \cdot 6.907$$

Need $\lambda_C(7.6) \geq 4.16$, so $\gamma \geq 0.602$.

### 3.4 Comparison Table

| Criterion | Exponential (A) | Polynomial (B) | Log-Cosh (C) |
|-----------|----------------|----------------|--------------|
| **$\lambda_0$ / $\gamma$ needed for $\alpha_s \geq 0.5$ at $N$=128K** | $\lambda_0 \geq 7.27$ | $\lambda_0 \geq 20.4$ ($p$=2) | $\gamma \geq 0.60$ |
| **Depends on $R_{\max}(c)$?** | Yes | Yes | **No** |
| **Curvature discontinuity risk** | Yes (via $R_{\max}$) | Yes (via $R_{\max}$) | **None** |
| **Gradient at origin** | $\lambda_0 \beta / R_{\max}$ | 0 ($p$>1) | 0 |
| **Gradient at boundary** | $\lambda_0 \beta e^\beta / R_{\max}$ | $\lambda_0 p / R_{\max}$ | $\gamma$ |
| **Computational cost (per token)** | 1 div + 1 exp | 1 div + 1 pow | **1 log** |
| **Hot-path arcosh needed?** | No | No | **No** |
| **Natural in Lorentz geometry?** | No (Euclidean scaling) | No (Euclidean scaling) | **Yes** ($\ln(\sqrt{c} \cdot x_0)$) |
| **Parameter sensitivity** | High ($\lambda_0 \sim 7$) | Very high ($\lambda_0 \sim 20$) | **Low** ($\gamma \sim 0.6$) |
| **Trainable parameter count** | 2 ($\lambda_0$, $\beta$) | 2 ($\lambda_0$, $p$) | **1** ($\gamma$) |

---

## 4. Recommendation

**Recommended form: Log-Cosh (Form C).**

$$\boxed{\lambda(r, c) = \gamma \cdot \ln(\cosh(r)) = \gamma \cdot \ln(\sqrt{c} \cdot x_0)}$$

### 4.1 Justification

**J1 — Curvature decoupling.** Forms A and B normalize by $R_{\max}(c)$, creating an implicit coupling between the decay profile and learnable curvature. When the optimizer adjusts $c$, the decay shape changes non-locally, producing a moving target for both parameters. The log-cosh form eliminates this coupling: $\gamma$ controls decay strength independently of $c$. The curvature enters only through an additive constant that vanishes in softmax.

**J2 — Geometric naturality.** The quantity $\cosh(r) = \sqrt{c} \cdot x_0$ is the temporal coordinate of the Lorentz embedding (up to a curvature scale). Penalizing $\ln(\text{temporal coord})$ is equivalent to saying "tokens whose time-like coordinate is larger (i.e., farther from the origin on the hyperboloid) are less influential." This is the direct geometric analogue of FAM's coverage-based eviction: prototypes far from the class centroid are more replaceable.

**J3 — Computational efficiency.** The per-token cost is a single `torch.log` on $x_0$ (the temporal coordinate already stored in the embedding). No division, no arcosh, no $R_{\max}$ lookup. This adds negligible overhead to the attention hot path.

**J4 — Parameter parsimony.** One scalar $\gamma$ per layer (or shared) suffices. Forms A and B require two parameters each ($\lambda_0, \beta$ or $\lambda_0, p$) and still need larger magnitudes to achieve the same gap closure.

**J5 — Smooth gradient profile.** $\partial\lambda_C/\partial r = \gamma \cdot \tanh(r)$, which transitions smoothly from 0 at the origin to $\gamma$ at the boundary. This gives zero gradient pressure on the system token (no perturbation to its privileged position) and bounded gradient on boundary tokens (no explosion).

**J6 — Monotone effective separation.** The function $r + \gamma \ln(\cosh(r))$ is strictly increasing for $\gamma > 0$, with derivative $1 + \gamma \tanh(r) > 0$. This guarantees that moving a token farther from the origin always increases its effective penalty — there are no pathological inversions.

### 4.2 Recommended Parameterization

| Parameter | Symbol | Default | Range | Notes |
|-----------|--------|---------|-------|-------|
| Decay strength | $\gamma$ | 0.6 | $(0, 2]$ | Per-layer learnable; $\gamma = 0.6$ closes the gap at $N$=128K |
| Raw parameter | $\gamma_{\text{raw}}$ | $\operatorname{softplus}^{-1}(0.6) \approx 0.117$ | $\mathbb{R}$ | $\gamma = \operatorname{softplus}(\gamma_{\text{raw}})$ ensures $\gamma > 0$ |
| LR multiplier | $\lambda_\gamma$ | 0.1 | — | Same as curvature (DS-2) |
| Update frequency | — | $k = 100$ | — | Aligned with curvature update cadence (DS-5 D6) |

---

## 5. Radial Placement Mechanism (MT-1 Gap Resolution)

### 5.1 The Placement Problem

The decay function requires context tokens to achieve Lorentz radius $\delta$ such that $\delta + \lambda(\delta)$ meets the privilege threshold. With $\gamma = 0.6$:

| $\delta$ (Lorentz) | $\lambda(\delta)$ | $\delta + \lambda$ | $\alpha_s$ floor ($N$=128K) |
|-------|---------|---------|---------|
| 5.0 | 2.59 | 7.59 | 0.0005 |
| 7.0 | 3.78 | 10.78 | 0.19 |
| 7.5 | 4.08 | 11.58 | 0.45 |
| 7.6 | 4.14 | 11.74 | 0.49 |
| 8.0 | 4.39 | 12.39 | 0.65 |
| 10.0 | 5.60 | 15.60 | 0.98 |

The current realistic $\delta \approx 7.6$ achieves $\alpha_s \approx 0.49$ with $\gamma = 0.6$, just at the $1/2$ threshold. Pushing to $\delta = 8.0$ provides margin.

### 5.2 Minimum Lorentz Radius for $\alpha_s \geq 0.3$ at $N = 128$K

**Target:** $\alpha_s \geq 0.3$ requires $(N-1) \cdot \exp(-\delta - \lambda(\delta)) \leq 7/3$.

$$\delta + \gamma \ln(\cosh(\delta)) \geq \ln\bigl(\tfrac{3(N-1)}{7}\bigr) \approx 10.91$$

Solving numerically with $\gamma = 0.6$:

For large $\delta$: $\ln(\cosh(\delta)) \approx \delta - \ln 2$, so $\delta + 0.6(\delta - 0.693) \geq 10.91$, giving $1.6\delta \geq 11.33$, hence $\delta \geq 7.08$.

**Verified:** at $\delta = 7.1$: $7.1 + 0.6 \cdot \ln(\cosh(7.1)) = 7.1 + 0.6 \cdot 6.407 = 7.1 + 3.84 = 10.94 \geq 10.91$. $\checkmark$

$$\boxed{\delta_{\min}(\alpha_s \geq 0.3, N = 128\text{K}, \gamma = 0.6) \approx 7.1}$$

This is within the range of realistic Poincare embeddings ($r_p \approx 0.999$ gives $\delta \approx 7.6 > 7.1$). **The decay function alone is sufficient to restore $\alpha_s \geq 0.3$ at current embedding radii.** The placement mechanism provides safety margin.

### 5.3 Evaluation of Placement Strategies

**(a) Radial regularization loss:**

$$L_{\text{radial}} = \mu \sum_{i} \max(0, r_{\text{target}} - r_i)^2$$

where $r_i = D(o, k_i)$ is the Lorentz distance of token $i$ from origin, and $r_{\text{target}}$ is a target radius.

- **Pro:** Continuous gradient signal; maintains separation throughout training.
- **Con:** Requires computing $r_i = \operatorname{arcosh}(\sqrt{c} \cdot x_{0,i})$ for all tokens each forward pass — adds one arcosh per token, which DS-5 deliberately avoided in the attention hot path.
- **Mitigation:** Use the monotone proxy $x_{0,i}$ (temporal coordinate) instead of $r_i$. Since $r = \operatorname{arcosh}(\sqrt{c} \cdot x_0)$ is monotonically increasing in $x_0$, we can equivalently regularize $x_{0,i}$ toward a target $x_{0,\text{target}} = \cosh(r_{\text{target}}) / \sqrt{c}$.
- **Recommended:** $L_{\text{radial}} = \mu \sum_i \max(0, x_{0,\text{target}} - x_{0,i})^2 / x_{0,\text{target}}^2$ with $\mu = 0.01$.

**(b) Learned temperature $\tau(r)$ scaling with radius:**

Replace the decay function with a radius-dependent temperature: $\text{score}(q, k_i) = f(q, k_i) / \tau(r_i)$.

- **Pro:** Subsumes the decay function; more expressive.
- **Con:** Temperature scaling modifies the sharpness of attention, not just the mean. At large $\tau$, attention becomes uniform over all tokens, defeating the purpose. Interaction with curvature is complex and hard to analyze.
- **Not recommended** as a standalone mechanism. The additive decay is cleaner.

**(c) Explicit radial initialization push (Euclidean $\to$ Lorentz projection):**

During the initial projection from Euclidean to Lorentz space, scale the tangent vector to target a specific radius:

$$v_{\text{init}} = v_{\text{euclidean}} \cdot \frac{r_{\text{target}}}{\|v_{\text{euclidean}}\|_{\mathcal{L}} + \varepsilon_{\text{norm}}}$$

then apply $\exp_o(v_{\text{init}})$ to place the token at distance $r_{\text{target}}$ from origin.

- **Pro:** One-time cost; no ongoing computation. Ensures good initial separation.
- **Con:** Separation may drift during training without ongoing regularization. System token placement must be handled separately (target $r = 0$).
- **Recommended** as initialization, supplemented by regularization.

**(d) Combination (Recommended):**

1. **Initialization push** (option c): project context tokens to $r_{\text{init}} = 8.0$ (provides $\alpha_s \approx 0.65$ with $\gamma = 0.6$ at $N = 128$K). System token initialized at origin.
2. **Radial regularization** (option a, proxy form): $L_{\text{radial}} = \mu \sum_i \max(0, x_{0,\text{target}} - x_{0,i})^2 / x_{0,\text{target}}^2$ with $\mu = 0.01$, active on context tokens only. Prevents inward drift.
3. **Decay function** (Section 4): $\lambda(r) = \gamma \cdot \ln(\sqrt{c} \cdot x_0)$ with learnable $\gamma$. Provides the fine-grained privilege gradient.

The initialization ensures a good starting point; the regularization prevents drift; the decay function provides the continuous privilege signal. Each component addresses a different timescale: initialization (step 0), regularization (training-time), decay (inference-time).

---

## 6. FAM $\leftrightarrow$ HCA Mapping Table

| FAM Concept | FAM Mechanism | HCA Analogue | Radial Decay Implementation |
|-------------|---------------|--------------|----------------------------|
| **Prototype freshness** | `last_seen` timestamp; LRU eviction of stale entries | **Radial position**: tokens near origin are "fresh" (high privilege), tokens at boundary are "stale" (low privilege) | $\lambda(r)$ penalty: larger $r$ $\to$ lower effective attention weight |
| **Coverage score** | $1 - \max(\text{cosine sim to same-class neighbor})$; low score = replaceable | **Radial redundancy**: tokens at similar radii in similar directions are redundant | Decay naturally penalizes clusters at boundary more than clusters at origin (due to convexity of $\ln\cosh$) |
| **Sole class representative protection** | `score = inf` for prototypes that are the only member of their class | **System token protection** | $\lambda(0) = 0$: system token at origin receives zero decay penalty |
| **Class-loss signal** | $1 - (n_{\text{classes\_present}} / n_{\text{classes\_ever\_seen}})$ | **Privilege erosion signal** | $\alpha_s$ monitor: when $\alpha_s$ drops below threshold, increase $\gamma$ via QoS hook |
| **Adaptive blending** | $p = 0.2 + 0.8 \cdot \min(\text{class\_loss}/0.30, 1)$; blend coverage $\leftrightarrow$ LRU | **Adaptive decay strength** | $\gamma$ learned per layer; optimizer adjusts blend between "no decay" ($\gamma \to 0$, pure distance) and "full decay" ($\gamma \to 2$, strong privilege) |
| **LRU fallback** | Pure `last_seen` eviction when `class_loss = 0` (no diversity threat) | **Pure distance attention** | When $\delta \geq \ln(N)$ naturally (no gap), $\gamma \to 0$ is optimal; decay is a no-op and attention is pure hyperbolic distance |
| **Coverage eviction** | Evict prototype with nearest same-class neighbor (most replaceable) | **Radial decay at boundary** | Tokens at $r \approx R_{\max}$ receive maximum penalty $\lambda(R_{\max}) \approx \gamma \cdot R_{\max}$; effectively "evicted" from attention influence |
| **EMA learning rate** | Exponential moving average of prototypes; recent data weighted more | **Curvature EMA** | $N_{\text{eff}}$ EMA from DS-5 D6; curvature floor adapts to effective sequence length |
| **`_classes_ever_seen` tracking** | `set[int]` updated each learn batch; denominator of class-loss | **Privilege history** | QoS hook tracks $\min(\alpha_s)$ over training; triggers $\gamma$ adjustment when privilege drops below historical baseline |
| **Blend ramp** | $p$ ramps $0.2 \to 1.0$ at 30% class loss | **$\gamma$ floor ramp** | $\gamma_{\min}(N) = \max(0, \ln(N) - \delta_{\text{achieved}}) / \ln(\cosh(\delta_{\text{achieved}}))$; ensures sufficient decay as $N$ grows |

---

## 7. Curvature Interaction Analysis

### 7.1 Per-Layer Decay Behavior

Layer $l$ has curvature $c^{(l)}$ (from DS-5) and decay parameter $\gamma^{(l)}$.

The effective per-token penalty in layer $l$ (after softmax cancellation of constant terms):

$$\lambda^{(l)}_{\text{eff}}(x_{0,i}) = \gamma^{(l)} \cdot \ln(x_{0,i}^{(l)})$$

where $x_{0,i}^{(l)}$ is the temporal coordinate of token $i$ in layer $l$'s Lorentz embedding.

**Observation:** The curvature $c^{(l)}$ does not appear in $\lambda_{\text{eff}}$. The $\frac{\gamma}{2}\ln c$ term is constant across tokens and cancels in softmax.

**Consequence:** The decay function is curvature-invariant at the per-layer level. Changing $c^{(l)}$ changes the geometry (distances, $R_{\max}$, volume growth) but does not change the relative decay penalty between tokens.

### 7.2 Does Per-Layer $c$ Require Per-Layer $\gamma$?

**Yes, but for expressivity, not correctness.** The privilege guarantee depends on $\delta^{(l)} + \lambda^{(l)}(\delta^{(l)}) \geq \ln N$, where $\delta^{(l)}$ is the achieved radial separation in layer $l$. Different layers may achieve different $\delta^{(l)}$ values depending on their curvature:

- **High-$c^{(l)}$ layers:** Larger $R_{\max}$, potentially larger $\delta^{(l)}$, less $\gamma$ needed.
- **Low-$c^{(l)}$ layers:** Smaller $R_{\max}$, potentially smaller $\delta^{(l)}$, more $\gamma$ needed.

However, the optimizer handles this automatically: per-layer learnable $\gamma^{(l)}$ will converge to the appropriate value for each layer's geometry.

### 7.3 Gradient Flow Through Curvature

The decay $\lambda^{(l)}(r) = \gamma^{(l)} \cdot \ln(\sqrt{c^{(l)}} \cdot x_0)$ is differentiable with respect to $c^{(l)}$:

$$\frac{\partial \lambda}{\partial c} = \frac{\gamma}{2c}$$

This gradient is positive ($\gamma, c > 0$): increasing curvature increases decay for all tokens. However, this term is constant across tokens, so it cancels in the softmax.

$$\frac{\partial}{\partial c}\bigl[\lambda(r_i) - \lambda(r_j)\bigr] = 0$$

**The decay's relative effect between tokens is independent of curvature.** Curvature gradients from the decay function are zero after softmax normalization. This means:

1. The decay function does not interfere with curvature learning.
2. The curvature optimizer (DS-5 D6) operates independently of $\gamma$.
3. The only coupling is indirect: curvature affects $\delta$ (via embedding geometry), which affects the optimal $\gamma$ (via the privilege constraint).

### 7.4 Interaction Summary

| Property | Derivation | Implication |
|----------|-----------|-------------|
| $\partial(\lambda_i - \lambda_j)/\partial c = 0$ | Constant term cancels in softmax | Decay and curvature are gradient-decoupled |
| $\gamma^{(l)}$ independent per layer | Different $\delta^{(l)}$ require different gap-filling | Per-layer $\gamma$ recommended |
| Shared $\gamma$ across layers | All layers use same gap-filling strength | Simpler; works if $\delta^{(l)}$ is similar across layers |
| $\gamma$ update frequency = $c$ update frequency | Both are global geometry changes; both should be slow | Aligned at $k = 100$ steps |

---

## 8. Numerical Stability Tables

All computations in fp32. Epsilon strategy: $\varepsilon_{\text{norm}} = 10^{-7}$, $\varepsilon_z = 10^{-8}$, $\varepsilon_{\sinh} = 10^{-6}$. $M = 1/\varepsilon_{\text{norm}} = 10^7$. $d = 64$.

### 8.1 Key Quantities at $c = 1.0$

| $N$ | $\ln(N)$ | $R_{\max}$ | $\delta_{\text{realistic}}$ | Gap $\ln(N) - \delta$ | $\gamma_{\min}$ | $\alpha_s$ (no decay) | $\alpha_s$ ($\gamma$=0.6) |
|-----|---------|-----------|---------------------------|----------------------|----------------|----------------------|--------------------------|
| 128 | 4.85 | 16.81 | 7.6 | 0 (satisfied) | 0 | 0.94 | 0.99+ |
| 1,024 | 6.93 | 16.81 | 7.6 | 0 (satisfied) | 0 | 0.67 | 0.98 |
| 16,384 | 9.70 | 16.81 | 7.6 | 2.10 | 0.30 | 0.10 | 0.88 |
| 131,072 | 11.78 | 16.81 | 7.6 | 4.18 | 0.61 | 0.015 | 0.49 |

**$\alpha_s$ computation at $\gamma = 0.6$:** $\alpha_s \geq 1/(1 + (N-1) \exp(-\delta - 0.6 \ln(\cosh(\delta))))$.

At $\delta = 7.6$: $\ln(\cosh(7.6)) = 6.907$, so effective separation $= 7.6 + 0.6 \times 6.907 = 11.74$.

- $N = 128$: $127 \cdot \exp(-11.74) = 127 \cdot 7.97 \times 10^{-6} = 0.001$. $\alpha_s \geq 0.999$.
- $N = 1024$: $1023 \cdot \exp(-11.74) = 0.008$. $\alpha_s \geq 0.992$.
- $N = 16384$: $16383 \cdot \exp(-11.74) = 0.131$. $\alpha_s \geq 0.884$.
- $N = 131072$: $131071 \cdot \exp(-11.74) = 1.045$. $\alpha_s \geq 0.489$.

### 8.2 Log-Cosh Decay Values Across Radius

$\gamma = 0.6$, $c = 1.0$:

| $r$ (Lorentz) | $\cosh(r)$ | $\ln(\cosh(r))$ | $\lambda_C(r)$ | $x_0$ (temporal) | fp32 concern? |
|---------------|-----------|-----------------|----------------|-----------------|--------------|
| 0 | 1.000 | 0.000 | 0.000 | 1.000 | None |
| 1.0 | 1.543 | 0.434 | 0.261 | 1.543 | None |
| 5.0 | 74.21 | 4.308 | 2.585 | 74.21 | None |
| 7.6 | 999.5 | 6.907 | 4.144 | 999.5 | None |
| 10.0 | $1.101 \times 10^4$ | 9.306 | 5.584 | $1.101 \times 10^4$ | None |
| 16.81 | $\sim 10^7$ | 16.12 | 9.671 | $\sim 10^7$ | $\ln(10^7) = 16.12$; fp32 OK |

### 8.3 Decay at Curvature Floor $c = c_{\min}(N)$

| $N$ | $c_{\min}$ | $\sqrt{c_{\min}} \cdot M$ | $R_{\max}$ | $x_{0,\max}$ | $\ln(\cosh(R_{\max}))$ | $\lambda_C(R_{\max})$ | fp32 concern? |
|-----|-----------|--------------------------|-----------|-------------|------------------------|----------------------|--------------|
| 128 | $3.3 \times 10^{-13}$ | 5.66 | 2.40 | $4.2 \times 10^6$ | 1.71 | 1.02 | None |
| 1,024 | $2.6 \times 10^{-12}$ | 15.97 | 3.46 | $2.1 \times 10^6$ | 2.77 | 1.66 | None |
| 16,384 | $4.1 \times 10^{-11}$ | 63.95 | 4.85 | $7.6 \times 10^5$ | 4.16 | 2.50 | None |
| 131,072 | $3.3 \times 10^{-10}$ | 181.0 | 5.89 | $3.2 \times 10^5$ | 5.20 | 3.12 | None |

**Note:** At $c = c_{\min}$, $R_{\max} \approx \frac{1}{2}\ln N$ by construction (DS-5). The temporal coordinates $x_0 = \cosh(R_{\max})/\sqrt{c_{\min}}$ remain within fp32 range ($< 10^7 = M$). All $\ln$ values are moderate. No overflow or underflow concerns.

### 8.4 Extreme Values and Precision Analysis

| Scenario | Value | fp32 representable? | Concern |
|----------|-------|---------------------|---------|
| $\lambda_C(0)$ | 0.0 | Yes | $\ln(1.0) = 0$ exact |
| $\lambda_C(R_{\max})$ at $c=1$ | $0.6 \times 16.12 = 9.67$ | Yes | Moderate |
| $\exp(-\lambda_C(R_{\max}))$ | $\exp(-9.67) = 6.3 \times 10^{-5}$ | Yes | Above fp32 $\varepsilon$ |
| $\gamma \cdot \tanh(R_{\max})$ (gradient) | $0.6 \times 1.0 = 0.6$ | Yes | Bounded by $\gamma$ |
| $\ln(x_0)$ at $x_0 = 1 + \varepsilon_z$ | $\ln(1 + 10^{-8}) \approx 10^{-8}$ | **Marginal** | Below fp32 precision ($\sim 10^{-7}$). Use `torch.log1p` for $x_0$ near 1. |
| $\ln(x_0)$ at $x_0 = M = 10^7$ | 16.12 | Yes | No concern |
| $\cosh(r)$ at $r = R_{\max} = 16.81$ | $\sim 10^7$ | Yes | $= M$ by construction |
| $\cosh(r)$ overflow threshold | $r = 89.4$ (fp32 max $\sim 3.4 \times 10^{38}$) | N/A | $R_{\max} = 16.81 \ll 89.4$ |

**Precision concern:** For tokens very close to the origin ($x_0 \approx 1/\sqrt{c}$), $\ln(x_0)$ may lose precision. At $c = 1.0$, $x_0 = 1$ at origin, and $\ln(1) = 0$ exactly. For tokens slightly displaced ($x_0 = 1 + \epsilon$), use `torch.log1p(x_0 - 1/sqrt(c))` to maintain precision. This is a known pattern in the DS-4 epsilon strategy.

**Recommendation:** Use `torch.log(torch.clamp(x_0, min=1/sqrt(c) + eps_z))` to avoid $\ln(0)$ and maintain precision near the origin.

---

## 9. Integration Sketch

### 9.1 Design Decisions

| Question | Decision | Justification |
|----------|----------|---------------|
| Pre-softmax, post-softmax, or KV cache? | **Pre-softmax additive bias** | (1) Pre-softmax: preserves softmax normalization; the decay integrates naturally with the attention logit computation. (2) Post-softmax: would require renormalization, adding compute. (3) KV modification: would permanently alter value representations, preventing the model from learning to override decay when appropriate. Pre-softmax is the only option that is both reversible (by learning $\gamma \to 0$) and normalization-preserving. |
| Continuous or periodic? | **Continuous (every forward pass)** | The decay is a deterministic function of the embedding's temporal coordinate — there is no state to accumulate. Periodic application (every $k$ steps) would mean the privilege guarantee only holds at update boundaries, leaving $k-1$ steps unprotected. The curvature update is periodic because it involves gradient accumulation; the decay is instantaneous. |
| Learnable temperature or curvature-only? | **Learnable $\gamma$ per layer** | Curvature $c$ controls the manifold geometry (distances, volumes, capacity). The decay parameter $\gamma$ controls the privilege gradient independently. Using $c$ alone to control both would conflate two distinct objectives (manifold capacity vs. origin privilege), preventing the optimizer from finding optimal settings for each. |

### 9.2 Pseudocode: Forward Pass Integration

```
# ─── Existing structures (from DS-5) ───
# c_eff: Tensor[n_layers]         — per-layer effective curvature
# K: Tensor[B, H, N, d+1]        — key embeddings in Lorentz space (x_0, x_spatial)
# Q: Tensor[B, H, N, d+1]        — query embeddings in Lorentz space
# V: Tensor[B, H, N, d_v]        — value embeddings (Euclidean)

# ─── New: decay parameter (per layer) ───
# gamma_raw: Parameter[n_layers]  — learnable, unconstrained
# gamma: gamma = softplus(gamma_raw)  — ensures gamma > 0

def attention_with_decay(Q_l, K_l, V_l, c_l, gamma_l):
    """Single-layer attention with radial decay.

    Q_l, K_l: [B, H, N, d+1] Lorentz embeddings for layer l
    V_l: [B, H, N, d_v] values
    c_l: scalar, effective curvature for layer l
    gamma_l: scalar, decay strength for layer l (= softplus(gamma_raw_l))
    """

    # Step 1: Base attention scores (squared Lorentz inner product, DS-5)
    #   lorentz_ip = -K_l[..., 0] * Q_l[..., 0] + (K_l[..., 1:] * Q_l[..., 1:]).sum(-1)
    #   scores = (-c_l * lorentz_ip) ** 2
    # ... (existing attention kernel, unchanged)
    scores = squared_lorentz_attention(Q_l, K_l, c_l)  # [B, H, N_q, N_k]

    # Step 2: Compute radial decay bias
    x0_k = K_l[..., 0]                              # [B, H, N_k] temporal coord of keys
    sqrt_c = torch.sqrt(c_l)
    # ln(sqrt(c) * x0) = 0.5*ln(c) + ln(x0)
    # The 0.5*ln(c) term is constant across keys and cancels in softmax.
    # Only need ln(x0) per key:
    ln_x0 = torch.log(torch.clamp(x0_k, min=1.0 / sqrt_c + 1e-8))  # [B, H, N_k]
    decay_bias = gamma_l * ln_x0                     # [B, H, N_k]

    # Step 3: Apply decay as pre-softmax bias (subtract = penalize large radius)
    scores = scores - decay_bias.unsqueeze(-2)        # broadcast over N_q dim

    # Step 4: Standard softmax + value aggregation
    attn_weights = softmax(scores, dim=-1)            # [B, H, N_q, N_k]
    output = attn_weights @ V_l                       # [B, H, N_q, d_v]

    return output, attn_weights
```

### 9.3 Pseudocode: Decay Parameter Management

```
# Decay parameters share the same low-frequency update path as curvature (DS-5 D6).
# gamma_raw is excluded from the main optimizer and updated every k=100 steps.

class RadialDecayModule(nn.Module):
    def __init__(self, n_layers, gamma_init=0.6):
        super().__init__()
        gamma_raw_init = torch.log(torch.expm1(torch.tensor(gamma_init)))
        self.gamma_raw = nn.Parameter(gamma_raw_init.expand(n_layers).clone())

        # Gradient accumulation buffer (mirrors DS-5 curvature buffer)
        self.register_buffer('gamma_grad_buffer', torch.zeros(n_layers))
        self.register_buffer('gamma_grad_count', torch.tensor(0, dtype=torch.long))

    def forward(self):
        """Return per-layer decay strength."""
        return F.softplus(self.gamma_raw)   # (n_layers,), strictly positive

    def accumulate_and_gate_grad(self):
        """Accumulate gamma_raw gradient; gate from main optimizer."""
        if self.gamma_raw.grad is not None:
            self.gamma_grad_buffer.add_(self.gamma_raw.grad.detach())
            self.gamma_grad_count.add_(1)
            self.gamma_raw.grad = None
        return self.gamma_grad_count.item() >= 100

    def apply_decay_step(self):
        """Apply accumulated gradient every k=100 steps."""
        if self.gamma_grad_count.item() == 0:
            return
        avg_grad = self.gamma_grad_buffer / self.gamma_grad_count.float()
        self.gamma_raw.grad = avg_grad.clone()
        # Caller's optimizer steps gamma_raw
        self.gamma_grad_buffer.zero_()
        self.gamma_grad_count.zero_()
```

### 9.4 Pseudocode: Radial Regularization Loss

```
def radial_regularization_loss(K, c, r_target=8.0, mu=0.01):
    """Penalize context tokens that are too close to the origin.

    K: [B, H, N, d+1] — key embeddings (Lorentz)
    c: scalar — effective curvature
    r_target: target Lorentz radius for context tokens
    mu: regularization strength

    Note: system token (index 0) is excluded.
    """
    x0 = K[..., 0]                                    # [B, H, N]
    x0_target = torch.cosh(torch.tensor(r_target)) / torch.sqrt(c)

    # Only penalize context tokens (exclude system token at index 0)
    x0_context = x0[..., 1:]                           # [B, H, N-1]
    shortfall = torch.clamp(x0_target - x0_context, min=0.0)
    loss = mu * (shortfall ** 2).mean() / (x0_target ** 2)
    return loss
```

### 9.5 Training Loop Integration

```
# Combined with DS-5 curvature module
curvature = AdaptiveCurvatureModule(n_layers=32, beta=1.0)
decay = RadialDecayModule(n_layers=32, gamma_init=0.6)

optimizer = AdamW([
    {'params': model.parameters(), 'lr': 1e-4},
    {'params': [curvature.c_raw], 'lr': 1e-5, 'weight_decay': 0.0},
    {'params': [decay.gamma_raw], 'lr': 1e-5, 'weight_decay': 0.0},
])

for step, batch in enumerate(dataloader):
    c_eff = curvature(batch.seq_len)          # [n_layers]
    gamma = decay()                            # [n_layers]

    logits = model(batch, c_eff=c_eff, gamma=gamma)
    loss = criterion(logits, batch.labels)
    loss_radial = radial_regularization_loss(model.keys, c_eff, r_target=8.0)
    total_loss = loss + loss_radial

    total_loss.backward()

    # Gate curvature and decay gradients
    c_ready = curvature.accumulate_and_gate_grad()
    g_ready = decay.accumulate_and_gate_grad()

    optimizer.step()
    optimizer.zero_grad()

    # Every k=100 steps: apply accumulated geometry updates
    if c_ready:
        curvature.apply_curvature_step(optimizer)
    if g_ready:
        decay.apply_decay_step()
    if c_ready or g_ready:
        optimizer.step()
        optimizer.zero_grad()
```

### 9.6 Hot-Path Cost Analysis

| Operation | Per-token cost | Total per attention layer |
|-----------|---------------|--------------------------|
| Read $x_0$ from key embedding | 1 memory access | Already loaded for Lorentz IP |
| `torch.clamp(x0, min=...)` | 1 comparison | $O(N)$ |
| `torch.log(x0)` | 1 transcendental op | $O(N)$ |
| Multiply by $\gamma$ | 1 FMA | $O(N)$ |
| Subtract from scores | 1 FMA (broadcast) | $O(N \cdot N_q)$ |
| **Total additional** | — | **$O(N)$ + $O(N \cdot N_q)$ broadcast** |

The broadcast subtraction dominates but is fused into the existing score computation (same memory access pattern as bias addition in standard attention). Net overhead: negligible compared to the $O(N^2 \cdot d)$ attention GEMM.

---

## 10. Open Questions and DS-7 Handoff Notes

### 10.1 Open Questions

**OQ-1: Optimal $\gamma_{\text{init}}$.** The analysis recommends $\gamma = 0.6$ based on the $\alpha_s \geq 0.5$ target at $N = 128$K with $\delta = 7.6$. However, the optimal $\gamma$ may depend on:
- Task-specific attention patterns (some tasks benefit from attending to boundary tokens).
- The actual achieved $\delta$ after initialization push and regularization.
- Training dynamics: too-large $\gamma$ early in training may prevent the model from learning useful long-range attention.

**Resolution:** Treat $\gamma_{\text{init}} = 0.6$ as the default; sweep $\{0.3, 0.6, 1.0\}$ in MT-2 (Phase 3 verification).

**OQ-2: Interaction with sparse/sliding-window attention.** The decay function assumes dense attention (every query attends to every key). With sparse patterns, only a subset of keys contribute. The decay still applies to the visible keys, but the privilege guarantee must account for the windowed subset.

**Resolution:** Defer to DS-8 (sparse attention integration). The decay function is agnostic to sparsity — it modifies scores for whatever keys are visible.

**OQ-3: Shared vs. per-layer $\gamma$.** The curvature interaction analysis (Section 7) shows that per-layer $\gamma$ is more expressive but adds $L$ parameters. If all layers achieve similar $\delta$, shared $\gamma$ suffices.

**Resolution:** Default to per-layer $\gamma$ (consistent with per-layer $c$). If ablations show no benefit, simplify to shared.

**OQ-4: System token identification.** The radial regularization (Section 9.4) excludes "system token at index 0." In practice, system tokens may not be at a fixed index. The mechanism needs a mask or token-type embedding to identify privileged tokens.

**Resolution:** Accept a `system_mask: Tensor[B, N]` argument. The regularization loss applies only to tokens where `system_mask == False`.

**OQ-5: Multi-origin architectures.** The current design assumes a single origin (the hyperboloid origin). Some architectures may benefit from multiple "anchor" points at different positions, each with its own privilege zone.

**Resolution:** Out of scope for DS-6. The decay function generalizes to $\lambda(r_i) = \gamma \cdot \min_j \ln(\cosh(D(a_j, k_i)))$ for anchor set $\{a_j\}$, but this requires DS-9 (multi-anchor topology).

### 10.2 DS-7 Handoff: Radial QoS Monitor

DS-7 (Phase 3: Quality-of-Service monitoring) should implement:

1. **$\alpha_s$ tracker:** Per-layer EMA of the system token's attention weight. Trigger QoS intervention when $\alpha_s$ drops below 0.3.

2. **Radial distribution monitor:** Histogram of token Lorentz radii per layer. Alert when the interquartile range of context token radii overlaps with the system token radius (loss of separation).

3. **$\gamma$ adjustment hook:** When $\alpha_s$ drops, inject a positive gradient into $\gamma_{\text{raw}}$ (increase decay strength). When $\alpha_s$ is safely above threshold, allow $\gamma$ to decrease (reduce decay, preserve context diversity). This mirrors FAM's adaptive blend ramp.

4. **Integration with DS-5 QoS hooks:** The `register_qos_hook` interface in `AdaptiveCurvatureModule` should be extended to accept decay parameter adjustments alongside curvature adjustments.

### 10.3 Verification Plan (MT-2)

The following must be verified before Phase 3 proceeds to implementation:

| Test | Metric | Pass criteria |
|------|--------|---------------|
| Gap closure at $N = 128$K | $\alpha_s$ with $\gamma = 0.6$ | $\alpha_s \geq 0.45$ at $\delta = 7.6$ |
| Gradient flow through $\gamma$ | $\|\partial L / \partial \gamma_{\text{raw}}\|$ | Non-zero, stable over 1000 steps |
| Curvature decoupling | $\text{corr}(\Delta c, \Delta \gamma)$ over training | $|\text{corr}| < 0.3$ (low coupling) |
| Radial regularization effectiveness | $\delta_{\text{achieved}}$ after 1000 steps | $\delta \geq 7.5$ maintained |
| fp32 stability at extreme radii | $\lambda(R_{\max})$, gradients | No NaN/Inf in 10K-step run |
| $\gamma$ convergence | $\gamma$ trajectory over training | Converges to stable value within $[0.3, 1.5]$ |

---

## Appendix A: Derivation Details

### A.1 Log-Cosh Identity in Lorentz Coordinates

For a point $x = (x_0, x_1, \ldots, x_d)$ on $\mathbb{H}_c^d$ with $\langle x, x \rangle_{\mathcal{L}} = -1/c$ and origin $o = (1/\sqrt{c}, \mathbf{0})$:

$$D(o, x) = \operatorname{arcosh}(-c \langle o, x \rangle_{\mathcal{L}}) = \operatorname{arcosh}(\sqrt{c} \cdot x_0) \eqqcolon r$$

Therefore $\cosh(r) = \sqrt{c} \cdot x_0$, and:

$$\ln(\cosh(r)) = \ln(\sqrt{c} \cdot x_0) = \tfrac{1}{2}\ln c + \ln x_0$$

The log-cosh decay function $\lambda(r) = \gamma \ln(\cosh(r))$ becomes:

$$\lambda = \gamma \cdot (\tfrac{1}{2}\ln c + \ln x_0)$$

In softmax attention, the $\frac{\gamma}{2}\ln c$ term is constant across all keys in the same layer and cancels:

$$\text{score}_i - \text{score}_j = [f(q, k_i) - f(q, k_j)] - \gamma[\ln x_{0,i} - \ln x_{0,j}]$$

Hence the effective decay is $\gamma \cdot \ln(x_{0,i})$, computable from the temporal coordinate alone.

### A.2 Effective Separation as a Function of $\gamma$

Define $S(\delta, \gamma) = \delta + \gamma \ln(\cosh(\delta))$ (effective separation).

For the privilege guarantee $\alpha_s \geq \alpha^*$:

$$S(\delta, \gamma) \geq \ln\!\left(\frac{(N-1)(1 - \alpha^*)}{\alpha^*}\right) \eqqcolon T(\alpha^*, N)$$

From Section 2.3, $\alpha_s \geq \alpha^*$ requires $S \geq \ln((N-1)\alpha^* / (1-\alpha^*))$.

| Target $\alpha^*$ | Required $S$ at $N$=128K | $\gamma_{\min}$ at $\delta$=7.6 |
|----------|------|------|
| 0.3 | 10.91 | 0.48 |
| 0.5 | 11.76 | 0.60 |
| 0.7 | 12.99 | 0.78 |
| 0.9 | 14.86 | 1.05 |

## Appendix B: Notation Reference

| Symbol | Definition |
|--------|-----------|
| $\lambda(r, c, N)$ | Radial decay function |
| $\gamma$ | Decay strength parameter (learnable, per layer) |
| $\gamma_{\text{raw}}$ | Unconstrained parameter; $\gamma = \operatorname{softplus}(\gamma_{\text{raw}})$ |
| $r = D(o, x)$ | Scaled Lorentz distance from origin |
| $x_0$ | Temporal (time-like) coordinate of a Lorentz embedding |
| $\delta$ | Minimum Lorentz radius of context tokens |
| $S(\delta, \gamma)$ | Effective separation: $\delta + \gamma \ln(\cosh(\delta))$ |
| $R_{\max}$ | Maximum representable Lorentz radius: $\operatorname{arcosh}(\sqrt{c} \cdot M)$ |
| $\alpha_s$ | Attention weight of the system token |
| $\mu$ | Radial regularization coefficient (default $0.01$) |
| $r_{\text{target}}$ | Target Lorentz radius for context tokens (default $8.0$) |
| $M$ | Maximum temporal coordinate ($= 1/\varepsilon_{\text{norm}} = 10^7$) |
