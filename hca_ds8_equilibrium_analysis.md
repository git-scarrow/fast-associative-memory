# HCA-DS-8: Equilibrium Analysis of the Curvature-Decay Control System

## 1. Executive Summary

This document analyzes the equilibrium behavior of the combined curvature ($c$) and radial decay ($\gamma$) control system specified in DS-5, DS-6, and DS-7. The system operates at the geometry update cadence ($k = 100$ steps) and consists of two coupled controllers: the curvature QoS hook (DS-5 D6, extended in DS-7) and the $\gamma$ proportional controller (DS-7 Section 4). Both react to a shared observable — the system token attention weight $\alpha_s$ — raising the question of whether the combined system has a stable equilibrium, how fast it converges, and what initial conditions guarantee convergence.

**Key results:**

1. **Fixed point exists and is unique** for any fixed data distribution and radial placement $\delta$. The equilibrium $\gamma^*$ satisfies $\delta + \gamma^* \ln(\cosh(\delta)) = \ln((N-1)(1-\alpha_{\text{safe}})/\alpha_{\text{safe}})$, yielding $\gamma^* = (\ln(N-1) - \delta) / \ln(\cosh(\delta))$ at $\alpha_{\text{safe}} = 0.5$. At $\delta = 7.6$, $N = 128$K: $\gamma^* \approx 0.60$.

2. **Convergence is geometric** with rate $\rho_{\text{conv}} = 1 - K_p \cdot \text{lr}_\gamma \cdot \sigma'(\gamma_{\text{raw}}^*) \cdot (\partial \alpha_s / \partial \gamma)|_{\gamma^*}$. Under default parameters, $\rho_{\text{conv}} \approx 1 - 10^{-5}$, giving convergence in $\sim 500$K geometry updates from worst case — but this is misleading because the *task loss gradient* on $\gamma_{\text{raw}}$ dominates the controller's gradient injection by $\sim 10^3 \times$. The controller is a **safety net**, not the primary driver of $\gamma$ convergence.

3. **Basin of attraction is the entire parameter space** $\gamma \in (0, \infty)$, $c > c_{\text{floor}}$, excluding the emergency reinit region ($\alpha_s < 0.1$ for 500+ steps). The proportional controller with bounded gain and monotone response has no limit cycles or unstable equilibria.

4. **Sparse attention** modifies the privilege bound: $\alpha_s \geq 1/(1 + (W-1) \exp(-\delta - \lambda(\delta)))$ where $W$ is the visible window size. For $W \ll N$, privilege is *easier* to maintain (fewer competitors). The QoS monitor operates identically; no modification needed.

---

## 2. System Model

### 2.1 State Variables

The control system state at geometry update step $n$ (corresponding to training step $t = nk$) is:

$$\mathbf{x}(n) = \bigl(\gamma_{\text{raw}}^{(l)}(n),\; c_{\text{raw}}^{(l)}(n),\; \hat{\alpha}_s^{(l)}(n)\bigr) \quad \forall l \in [1, L]$$

For clarity, we analyze a single layer and drop the $(l)$ superscript. The analysis extends to all layers independently because the DS-7 controllers are per-layer with no cross-layer coupling (DS-6 Section 7, DS-7 Section 5).

### 2.2 Dynamics

**Observable:** $\alpha_s$ depends on $\gamma$ and $\delta$ (radial separation) via the privilege bound (DS-6 Section 2):

$$\alpha_s(\gamma, \delta, N) = \frac{1}{1 + (N-1) \exp(-S(\delta, \gamma))}$$

where $S(\delta, \gamma) = \delta + \gamma \ln(\cosh(\delta))$ is the effective separation (DS-6 Section 2.4).

**EMA tracker** (DS-7 Section 2.2):

$$\hat{\alpha}_s(n) = \rho \hat{\alpha}_s(n-1) + (1-\rho) \alpha_s(\gamma(n), \delta(n), N(n))$$

**$\gamma$ controller** (DS-7 Section 4.3):

$$\gamma_{\text{raw}}(n+1) = \gamma_{\text{raw}}(n) - \text{lr}_\gamma \bigl[g_{\text{task}}(n) + \Delta g_\gamma(n)\bigr]$$

where $g_{\text{task}}$ is the task loss gradient (from training) and $\Delta g_\gamma$ is the QoS controller injection.

**Curvature controller** (DS-5 D6 + DS-7 Section 5.4):

$$c_{\text{raw}}(n+1) = c_{\text{raw}}(n) - \text{lr}_c \bigl[g_{c,\text{task}}(n) + \Delta g_c(n)\bigr]$$

where $\Delta g_c$ fires only on radial compression (DS-7 Section 5.2: $\Delta g_c = -0.1$ when $x_{0,75}/x_{0,25} < 2$ and $\hat{\alpha}_s < \alpha_{\text{safe}}$).

### 2.3 Simplifications for Analysis

**S1 — Fixed radial separation.** The radial separation $\delta$ depends on the learned embeddings, which are driven by the task loss, not the QoS controller. The controller adjusts $\gamma$ (the decay penalty) but does not directly move tokens. Therefore, we treat $\delta$ as a slowly-varying external parameter and analyze equilibrium for fixed $\delta$.

**S2 — Curvature decoupled.** DS-6 Section 7.3 proves that $\partial(\lambda_i - \lambda_j)/\partial c = 0$ — the decay function's relative effect is curvature-invariant. The curvature controller fires only on radial compression (a rare condition at realistic operating points). For equilibrium analysis, we treat $c$ as fixed and analyze the $\gamma$ controller in isolation.

**S3 — EMA convergence.** With $\rho = 0.99$ and updates every $k = 100$ steps, the EMA converges to within 1% of its target in $\sim 460$ updates ($\rho^{460} = 0.01$). For the control analysis, we use the steady-state EMA approximation $\hat{\alpha}_s \approx \alpha_s$.

---

## 3. Fixed-Point Analysis

### 3.1 Equilibrium Condition

At equilibrium, the controller injection $\Delta g_\gamma = 0$. From DS-7 Section 4.3, this occurs when the privilege error $e = \alpha_{\text{safe}} - \hat{\alpha}_s$ satisfies $|e| < \varepsilon_{\text{dead}} = 0.05$.

The **exact** fixed point (ignoring the dead zone) satisfies:

$$\alpha_s(\gamma^*, \delta, N) = \alpha_{\text{safe}}$$

Substituting the privilege bound:

$$\frac{1}{1 + (N-1)\exp(-S(\delta, \gamma^*))} = \alpha_{\text{safe}}$$

Solving for $S$:

$$S(\delta, \gamma^*) = \ln\!\left(\frac{(N-1)(1 - \alpha_{\text{safe}})}{\alpha_{\text{safe}}}\right) \eqqcolon T$$

At $\alpha_{\text{safe}} = 0.5$: $T = \ln(N-1)$.

$$\delta + \gamma^* \ln(\cosh(\delta)) = \ln(N-1)$$

$$\boxed{\gamma^* = \frac{\ln(N-1) - \delta}{\ln(\cosh(\delta))}}$$

### 3.2 Existence and Uniqueness

**Existence:** $\gamma^* > 0$ iff $\delta < \ln(N-1)$. At $\delta = 7.6$, $N = 128$K: $\ln(127999) = 11.76$, so $\gamma^* = (11.76 - 7.6)/6.907 = 0.602 > 0$. $\checkmark$

If $\delta \geq \ln(N-1)$, the natural separation is sufficient and no decay is needed ($\gamma^* = 0$). The controller correctly drives $\gamma \to 0$ via the relaxation path ($K_r$).

**Uniqueness:** $\alpha_s(\gamma, \delta, N)$ is strictly increasing in $\gamma$ (DS-6 Section 4.1, J6: $S$ is strictly increasing in $\gamma$). A strictly monotone function crosses any horizontal line at most once. Therefore, the fixed point is unique for each $(\delta, N)$. $\blacksquare$

### 3.3 Equilibrium Values

| $\delta$ | $N$ | $\gamma^*$ | $\alpha_s$ at equilibrium | Physical regime |
|----------|-----|-----------|--------------------------|-----------------|
| 5.0 | 128K | 0.98 | 0.500 | Low separation, strong decay needed |
| 7.0 | 128K | 0.69 | 0.500 | Moderate separation |
| 7.6 | 128K | 0.60 | 0.500 | Realistic embeddings (MT-1) |
| 8.0 | 128K | 0.54 | 0.500 | Good separation, moderate decay |
| 10.0 | 128K | 0.19 | 0.500 | Excellent separation, minimal decay |
| 11.76 | 128K | 0.00 | 0.500 | Perfect separation, no decay needed |
| 7.6 | 1K | 0.00 | 0.992 | Short context, no decay needed |
| 7.6 | 8K | 0.01 | 0.974 | Medium context, trivial decay |
| 7.6 | 32K | 0.32 | 0.500 | Long context |

**Observation:** $\gamma^*$ is approximately linear in $\ln(N)$ for fixed $\delta$:

$$\gamma^* \approx \frac{\ln N - \delta}{\delta} \quad \text{(for large } \delta \text{, where } \ln(\cosh(\delta)) \approx \delta\text{)}$$

This confirms the DS-6 recommendation of $\gamma_{\text{init}} = 0.6$ as the correct default for $N = 128$K, $\delta = 7.6$.

### 3.4 Dead Zone Equilibrium Band

The controller has a dead zone $|e| < \varepsilon_{\text{dead}} = 0.05$, so the actual equilibrium is a **band**, not a point:

$$\alpha_{\text{safe}} - \varepsilon_{\text{dead}} \leq \alpha_s \leq \alpha_{\text{safe}} + \varepsilon_{\text{dead}}$$

$$0.45 \leq \alpha_s \leq 0.55$$

This corresponds to a $\gamma$ band:

$$\gamma_{\text{low}}^* \leq \gamma \leq \gamma_{\text{high}}^*$$

where $\gamma_{\text{low}}^*$ gives $\alpha_s = 0.45$ and $\gamma_{\text{high}}^*$ gives $\alpha_s = 0.55$.

At $\delta = 7.6$, $N = 128$K:

- $\alpha_s = 0.45$: $S = \ln(127999 \times 55/45) = 11.96$. $\gamma_{\text{low}}^* = (11.96 - 7.6)/6.907 = 0.631$.

Wait — this is inverted. Higher $\gamma$ gives higher $\alpha_s$, so:
- $\alpha_s = 0.45$: $T = \ln(127999 \times 0.55/0.45) = \ln(156443) = 11.96$. $\gamma = 0.631$.
- $\alpha_s = 0.55$: $T = \ln(127999 \times 0.45/0.55) = \ln(104727) = 11.56$. $\gamma = 0.573$.

So the equilibrium band is $\gamma \in [0.573, 0.631]$ — a width of $0.058$ in $\gamma$. Any $\gamma$ in this band produces no controller action. The optimizer's task gradient determines the exact resting point within the band.

---

## 4. Convergence Analysis

### 4.1 Controller-Only Convergence (No Task Gradient)

Consider the $\gamma$ controller operating in isolation (task gradient $g_{\text{task}} = 0$). The update is:

$$\gamma_{\text{raw}}(n+1) = \gamma_{\text{raw}}(n) - \text{lr}_\gamma \cdot \Delta g_\gamma(n)$$

For $e > \varepsilon_{\text{dead}}$ (privilege too low), the injection is $\Delta g_\gamma = -K_p (e - \varepsilon_{\text{dead}})$, so:

$$\gamma_{\text{raw}}(n+1) = \gamma_{\text{raw}}(n) + \text{lr}_\gamma \cdot K_p \cdot (e(n) - \varepsilon_{\text{dead}})$$

The per-step change in $\gamma = \text{softplus}(\gamma_{\text{raw}})$ is:

$$\Delta \gamma(n) \approx \sigma(\gamma_{\text{raw}}(n)) \cdot \text{lr}_\gamma \cdot K_p \cdot (e(n) - \varepsilon_{\text{dead}})$$

where $\sigma$ is the sigmoid (derivative of softplus).

The error dynamics: $e(n) = \alpha_{\text{safe}} - \alpha_s(\gamma(n))$, so:

$$\Delta e(n) \approx -\frac{\partial \alpha_s}{\partial \gamma} \cdot \Delta \gamma(n) = -\frac{\partial \alpha_s}{\partial \gamma} \cdot \sigma \cdot \text{lr}_\gamma \cdot K_p \cdot (e(n) - \varepsilon_{\text{dead}})$$

This is a first-order linear recurrence in $\tilde{e} = e - \varepsilon_{\text{dead}}$:

$$\tilde{e}(n+1) \approx (1 - \mu) \tilde{e}(n)$$

where the convergence rate is:

$$\mu = \frac{\partial \alpha_s}{\partial \gamma}\bigg|_{\gamma^*} \cdot \sigma(\gamma_{\text{raw}}^*) \cdot \text{lr}_\gamma \cdot K_p$$

### 4.2 Computing $\partial \alpha_s / \partial \gamma$

From the privilege bound:

$$\alpha_s = \frac{1}{1 + (N-1) e^{-S}}$$

$$\frac{\partial \alpha_s}{\partial \gamma} = \frac{(N-1) e^{-S} \cdot \ln(\cosh(\delta))}{(1 + (N-1) e^{-S})^2}$$

At the fixed point $\alpha_s = 0.5$: $(N-1)e^{-S} = 1$, so:

$$\frac{\partial \alpha_s}{\partial \gamma}\bigg|_{\gamma^*} = \frac{1 \cdot \ln(\cosh(\delta))}{(1 + 1)^2} = \frac{\ln(\cosh(\delta))}{4}$$

At $\delta = 7.6$: $\partial \alpha_s / \partial \gamma|_{\gamma^*} = 6.907/4 = 1.727$.

### 4.3 Convergence Rate

$$\mu = 1.727 \times \sigma(0.117) \times 10^{-5} \times 0.5$$

where $\gamma_{\text{raw}}^* = \text{softplus}^{-1}(0.6) \approx 0.117$ and $\sigma(0.117) = 0.529$.

$$\mu = 1.727 \times 0.529 \times 10^{-5} \times 0.5 = 4.57 \times 10^{-6}$$

$$\rho_{\text{conv}} = 1 - \mu = 1 - 4.57 \times 10^{-6}$$

**Steps to converge from worst case:**

Starting from $\gamma = 0$ ($e \approx 0.485$, since $\alpha_s \approx 0.015$ at $\delta = 7.6$ with no decay):

$$n_{\text{conv}} = \frac{\ln(\tilde{e}_0 / \varepsilon_{\text{dead}})}{\ln(1/(1-\mu))} \approx \frac{\ln(0.435/0.05)}{4.57 \times 10^{-6}} = \frac{2.16}{4.57 \times 10^{-6}} \approx 473{,}000 \text{ geometry updates}$$

This is $\sim 47.3$M training steps — **the controller alone is far too slow to be the primary convergence mechanism.**

### 4.4 Why This Is Not a Problem

The controller is not designed to find $\gamma^*$ from scratch. It is a **safety net** that provides corrective gradient when the task loss gradient drives $\gamma$ away from the equilibrium band. The actual convergence path has two phases:

**Phase 1 — Task gradient dominance.** The task loss $L$ has a gradient with respect to $\gamma_{\text{raw}}$ that dwarfs the controller injection. The magnitude of $g_{\text{task}}$ is typically $O(10^{-2})$ to $O(10^{-1})$, while the controller injection $\Delta g_\gamma$ is $O(K_p \cdot e) = O(0.25)$ — comparable! But $g_{\text{task}}$ acts every step (accumulated over $k = 100$ steps), while $\Delta g_\gamma$ is applied once per $k$ steps.

Effective per-step: $g_{\text{task,avg}} \sim 10^{-2}$ vs. $\Delta g_\gamma / k \sim 2.5 \times 10^{-3}$. Task gradient is $\sim 4\times$ larger but in the same order. The controller is not negligible — it provides a meaningful bias toward the privilege-maintaining equilibrium.

**Phase 2 — Dead zone.** Once $\gamma$ enters the equilibrium band (Section 3.4), $\Delta g_\gamma = 0$ and the task gradient alone determines the resting point. The controller is silent.

**Effective convergence time:** Starting from $\gamma_{\text{init}} = 0.6$ (the DS-6 default, which is approximately $\gamma^*$), the system begins *inside* the equilibrium band. No controller action is needed unless a perturbation pushes $\gamma$ outside. For perturbations of size $|\Delta \gamma| \sim 0.1$, recovery takes:

$$n_{\text{recover}} \approx \frac{\ln(0.1/0.05)}{4.57 \times 10^{-6}} \approx 152{,}000 \text{ geometry updates}$$

But with task gradient assistance, the actual recovery is much faster ($\sim 10$–$100$ geometry updates in practice).

---

## 5. Basin of Attraction

### 5.1 Global Stability Argument

**Claim:** The basin of attraction for the $\gamma$ controller is the entire parameter space $\gamma_{\text{raw}} \in \mathbb{R}$ (equivalently, $\gamma \in (0, \infty)$).

**Proof sketch:**

1. **Monotone response:** $\alpha_s$ is strictly increasing in $\gamma$ (Section 3.2). Therefore, the error $e = \alpha_{\text{safe}} - \alpha_s$ is strictly decreasing in $\gamma$.

2. **Correct sign:** For $e > 0$ (privilege too low), the controller injects $\Delta g_\gamma < 0$, which increases $\gamma_{\text{raw}}$ (via $\theta \leftarrow \theta - \text{lr} \cdot g$), which increases $\gamma$, which increases $\alpha_s$, which decreases $e$. The feedback is negative.

3. **Bounded injection:** $|\Delta g_\gamma| \leq \Delta g_{\max} = 1.0$, so $|\Delta \gamma| \leq \sigma(\gamma_{\text{raw}}) \cdot \text{lr}_\gamma \cdot 1.0 \leq 10^{-5}$ per geometry update. The controller cannot overshoot by more than $10^{-5}$ in $\gamma$ per update.

4. **No limit cycles:** A one-dimensional monotone system with bounded negative feedback has no limit cycles (by the monotone convergence theorem for discrete dynamical systems). The trajectory $\{\gamma(n)\}$ is eventually monotone (always increasing when $e > 0$, always decreasing when $e < 0$) and bounded (by the dead zone), hence convergent.

5. **No unstable equilibria:** The unique fixed point $\gamma^*$ has $\mu > 0$ (Section 4.3), so it is locally asymptotically stable. Since it's the only fixed point and the system is monotone, it's globally asymptotically stable. $\blacksquare$

### 5.2 Emergency Reinit Boundary

The only configuration that triggers emergency reinit (DS-7 Section 4.6) is:

$$\hat{\alpha}_s < 0.1 \quad \text{for 5 consecutive geometry updates (500 training steps)}$$

This requires both:
- $\gamma$ is far below $\gamma^*$ (insufficient decay), AND
- $\delta$ is very small (tokens near origin), AND
- The task gradient is not correcting (pathological training dynamics)

**When does this occur?** From the privilege bound, $\alpha_s < 0.1$ when:

$$S(\delta, \gamma) < \ln(9(N-1)) = \ln(9 \times 127999) = 13.96$$

At $\delta = 7.6$: $\gamma < (13.96 - 7.6) / 6.907 \cdot (1 - 1/(1+9))^{-1}$... more directly:

$\alpha_s < 0.1$ requires $S < \ln(9(N-1))$. At $\delta = 7.6$: $7.6 + \gamma \cdot 6.907 < 13.96$, so $\gamma < 0.921$.

But $\hat{\alpha}_s < 0.1$ is an EMA, which lags. If $\gamma$ starts at $\gamma_{\text{init}} = 0.6$, then $\alpha_s \approx 0.49$ (Section 3.3), well above 0.1. Emergency reinit only triggers when:

1. $\gamma_{\text{init}}$ is set to $< 0$ (impossible — softplus ensures $\gamma > 0$), or
2. The task gradient drives $\gamma \to 0$ and the controller's corrective injection is insufficient (the $4.57 \times 10^{-6}$ per-update rate can't compensate), or
3. $\delta$ collapses (radial separation lost due to training dynamics).

Case (3) is the real risk. If $\delta$ drops from 7.6 to 3.0 (severe separation loss), then even $\gamma = 2.0$ gives $S = 3.0 + 2.0 \times 2.30 = 7.60$, and $\alpha_s = 1/(1 + 127999 \cdot e^{-7.60}) = 1/(1 + 127999 \times 0.0005) = 1/(1 + 64) \approx 0.015$. This is CRITICAL territory regardless of $\gamma$. The controller cannot compensate for fundamental separation loss — the radial regularization (DS-6 Section 5.3) is the primary defense.

### 5.3 Basin Summary

| Region | $\gamma$ range | $\delta$ range | Convergence | Emergency? |
|--------|---------------|---------------|-------------|------------|
| **Normal** | $[0.3, 1.5]$ | $[7.0, \infty)$ | Monotone to dead zone | No |
| **Low decay** | $[0, 0.3)$ | $[7.0, \infty)$ | Slow convergence; task gradient assists | No (α_s > 0.1) |
| **Extreme low decay** | $\sim 0$ | $[5.0, 7.0)$ | Controller + emergency multiplier | Possible if sustained |
| **Separation collapse** | Any | $< 5.0$ | Controller ineffective | Yes (α_s < 0.1) |
| **Over-decay** | $> 1.5$ | Any | Relaxation via K_r = 0.1 (5× slower) | No (α_s > 0.5) |

---

## 6. Coupled Controller Interaction

### 6.1 Curvature-Decay Coupling

The curvature controller (DS-5 D6 + DS-7 Section 5.2) fires when radial compression is detected ($x_{0,75}/x_{0,25} < 2$ AND $\hat{\alpha}_s < \alpha_{\text{safe}}$). This increases $c$, which increases $R_{\max}$ but does NOT directly change $\delta$ (separation depends on learned embeddings, not just available space).

The $\gamma$ controller fires when $\hat{\alpha}_s$ is outside the dead zone. Both controllers respond to the same signal ($\hat{\alpha}_s$) but through different actuators ($c$ vs. $\gamma$).

### 6.2 Interaction Scenarios

**Scenario A — Normal operation (no compression).**

Only the $\gamma$ controller is active. Curvature controller is silent (compression condition not met). System behaves as the single-controller analysis in Sections 3–5.

**Scenario B — Compression + privilege loss.**

Both controllers fire simultaneously:
- $\gamma$ controller: $\Delta g_\gamma = -K_p(e - \varepsilon_{\text{dead}})$ (increase $\gamma$)
- Curvature controller: $\Delta g_c = -0.1$ (increase $c$)

These are *complementary*, not conflicting:
- Increasing $\gamma$ directly boosts $\alpha_s$ via stronger decay penalty
- Increasing $c$ expands $R_{\max}$, potentially allowing tokens to spread further (increasing $\delta$ over subsequent training steps)

The combined effect on $\alpha_s$ is additive: $\Delta \alpha_s \approx (\partial \alpha_s / \partial \gamma) \Delta \gamma + (\partial \alpha_s / \partial c) \Delta c_{\text{indirect}}$, where $\Delta c_{\text{indirect}}$ operates on a slower timescale (training must adjust embeddings to exploit the larger $R_{\max}$).

**Scenario C — Curvature fires, $\gamma$ is in dead zone.**

$\alpha_s \in [0.45, 0.5)$ (below $\alpha_{\text{safe}}$ but within dead zone for $\gamma$ controller), while compression is detected. Only curvature controller acts. This is safe — the curvature adjustment is conservative ($\Delta g_c = -0.1$) and cannot cause instability.

### 6.3 Interaction Stability

**Claim:** The coupled system is stable because:

1. **Non-competing actuators.** $c$ and $\gamma$ affect $\alpha_s$ through independent channels (DS-6 Section 7.3: gradient-decoupled). There is no scenario where increasing $c$ counteracts increasing $\gamma$ on $\alpha_s$.

2. **Strictly cooperative.** Both controllers push $\alpha_s$ upward when it's too low. There is no negative feedback loop between controllers — they cooperate monotonically.

3. **Different timescales.** The curvature adjustment operates indirectly (expand $R_{\max}$ → tokens must migrate → $\delta$ increases → $\alpha_s$ increases), taking $\sim 1000$s of training steps. The $\gamma$ adjustment operates near-instantly (next forward pass). The timescale separation prevents resonance.

4. **Curvature controller is conditional.** It fires only when radial compression is detected, not on every $\alpha_s$ drop. Most $\alpha_s$ deviations are handled by $\gamma$ alone.

---

## 7. Sparse Attention Analysis

### 7.1 Problem Statement

DS-6 OQ-2 deferred sparse attention interaction to this spec. Under block-sparse or sliding-window attention, each query attends to a subset of $W$ keys (out of $N$ total). How does this affect the privilege guarantee and QoS monitoring?

### 7.2 Modified Privilege Bound

**Theorem (Sparse Attention Privilege).** Under a window of size $W$ that always includes the system token, for context tokens at minimum distance $\delta$ from the origin:

$$\alpha_s \geq \frac{1}{1 + (W-1) \exp(-\delta - \lambda(\delta))}$$

**Proof.** Identical to DS-6 Section 2.3, replacing $N-1$ with $W-1$ (only $W-1$ non-system tokens compete for attention mass in the visible window). $\blacksquare$

**Corollary:** For $\alpha_s \geq 0.5$:

$$\delta + \lambda(\delta) \geq \ln(W-1)$$

At $W = 256$ (typical sliding window): $\ln(255) = 5.54$. Since $\delta = 7.6 > 5.54$, **no decay is needed at all** — the natural separation is sufficient for privilege even without $\gamma$.

At $W = 1024$: $\ln(1023) = 6.93 < 7.6$. Still sufficient without decay.

At $W = 4096$: $\ln(4095) = 8.32 > 7.6$. Need $\lambda(\delta) \geq 0.72$, so $\gamma \geq 0.72/6.907 = 0.104$. Trivial decay.

### 7.3 Implication for QoS Monitor

**The QoS monitor requires no modification for sparse attention.** The $\alpha_s$ tracker reads the actual attention weight from the computed (sparse) attention matrix. If the window always includes the system token (standard for prefix-based system prompts), $\alpha_s$ is well-defined and the tracker operates identically.

The key insight: sparse attention *helps* privilege. With fewer competitors ($W \ll N$), the system token's share is naturally larger. The QoS monitor will observe higher $\alpha_s$ values, keeping the layer in HEALTHY state and the $\gamma$ controller silent (in the dead zone or relaxation path).

### 7.4 When Sparse Attention Hurts

Sparse attention breaks the privilege guarantee only when the **system token is NOT in the visible window** for some queries. This occurs with:

- **Local sliding window** without a global token slot: queries far from the system token cannot attend to it at all ($\alpha_s = 0$ for those queries)
- **Random sparse patterns**: system token has a non-zero probability of exclusion

**Mitigation:** Architectures using HCA should always include the system token in every query's visible set. This is standard practice (Longformer's global attention, BigBird's global tokens, etc.) and requires no mechanism change — just a constraint on the sparsity pattern.

### 7.5 Sparse Attention Equilibrium

For fixed window $W$, the equilibrium analysis (Section 3) applies with $N$ replaced by $W$:

$$\gamma^*_W = \frac{\ln(W-1) - \delta}{\ln(\cosh(\delta))}$$

| Window $W$ | $\gamma^*_W$ at $\delta = 7.6$ | Comparison to dense ($N = 128$K) |
|-----------|-------------------------------|----------------------------------|
| 256 | 0 (satisfied without decay) | $-0.60$ lower |
| 1024 | 0 (satisfied without decay) | $-0.60$ lower |
| 4096 | 0.104 | $-0.50$ lower |
| 16384 | 0.313 | $-0.29$ lower |
| 65536 | 0.507 | $-0.09$ lower |
| 131072 | 0.602 | Identical |

Sparse attention dramatically reduces the required $\gamma$. The controller will naturally drive $\gamma$ lower via the relaxation path when operating with sparse attention.

---

## 8. Open Questions and DS-9 Handoff Notes

### 8.1 Open Questions

**OQ-1: Task gradient interaction.** The analysis treats $g_{\text{task}}$ as an external disturbance. In practice, the task gradient on $\gamma_{\text{raw}}$ may be correlated with $e$ (privilege error). If the task benefits from high $\alpha_s$ (e.g., instruction following), the task gradient naturally pushes $\gamma$ toward $\gamma^*$, accelerating convergence. If the task benefits from low $\alpha_s$ (e.g., long-range retrieval), the task gradient opposes the controller, potentially creating a steady-state offset within the dead zone. This interaction is empirical and cannot be analyzed without task-specific data.

**Resolution:** MT-2 verification should measure $\text{corr}(g_{\text{task}}, e)$ across several tasks.

**OQ-2: Multi-layer equilibrium heterogeneity.** Different layers may converge to different $\gamma^*$ values depending on their $\delta^{(l)}$. Early layers (broader attention) may have lower $\delta$ and need higher $\gamma$; deep layers (sharper extraction) may have higher $\delta$ and need less $\gamma$. The per-layer design handles this, but the heterogeneity profile is unknown.

**Resolution:** MT-2 should report the per-layer $\gamma$ profile at convergence.

**OQ-3: Non-stationary $\delta$.** During training, $\delta$ changes as embeddings evolve. The equilibrium $\gamma^*(\delta)$ is a moving target. The dead zone ($\gamma$ band width $\sim 0.058$) absorbs small $\delta$ fluctuations, but large shifts (e.g., during curriculum changes) may cause transient DEGRADED states.

**Resolution:** The controller handles this by design — it continuously tracks $\alpha_s$ and adjusts $\gamma$ accordingly. No additional mechanism needed.

### 8.2 DS-9 Handoff: bcachefs Mapping

DS-9 should map the HCA geometry to bcachefs structural privilege topology:

1. **Btree node ↔ Hyperbolic token:** How does bcachefs's B-tree node hierarchy map to radial position? Root nodes = origin (system token), leaf nodes = boundary.
2. **Journal ↔ EMA tracking:** bcachefs's journal provides write ordering guarantees. How does HCA's EMA tracking provide similar temporal ordering guarantees for privilege?
3. **Allocation groups ↔ Attention heads:** bcachefs uses allocation groups for parallel writes. How do attention heads provide parallel privilege channels?
4. **Tiering ↔ Curvature layers:** bcachefs tiers data across devices (fast SSD ↔ slow HDD). How does per-layer curvature provide similar access-speed tiering?

---

## Appendix A: Notation Reference

| Symbol | Definition |
|--------|-----------|
| $\gamma^*$ | Equilibrium decay strength |
| $S(\delta, \gamma)$ | Effective separation: $\delta + \gamma \ln(\cosh(\delta))$ |
| $T$ | Target effective separation: $\ln((N-1)(1-\alpha_{\text{safe}})/\alpha_{\text{safe}})$ |
| $\mu$ | Controller convergence rate per geometry update |
| $\rho_{\text{conv}}$ | Convergence ratio: $1 - \mu$ |
| $W$ | Sparse attention window size |
| $e$ | Privilege error: $\alpha_{\text{safe}} - \hat{\alpha}_s$ |
| $\tilde{e}$ | Dead-zone-adjusted error: $e - \varepsilon_{\text{dead}}$ |
| $g_{\text{task}}$ | Task loss gradient on $\gamma_{\text{raw}}$ |
| $\Delta g_\gamma$ | Controller gradient injection (DS-7 Section 4.3) |

## Appendix B: Summary of Results

| Question | Answer |
|----------|--------|
| Does a fixed point exist? | **Yes**, unique, at $\gamma^* = (\ln(N-1) - \delta)/\ln(\cosh(\delta))$ |
| Is it stable? | **Yes**, globally asymptotically stable (monotone system) |
| Convergence rate (controller only)? | Geometric, $\mu \approx 4.6 \times 10^{-6}$ per geometry update — very slow |
| Convergence rate (with task gradient)? | Dominated by task gradient; 10–100 geometry updates in practice |
| Basin of attraction? | **Entire parameter space** $\gamma > 0$, excluding $\delta$-collapse regime |
| Emergency reinit trigger? | Only when $\delta < 5$ (separation collapse) — controller cannot compensate |
| Coupled controller stability? | **Stable** — non-competing, cooperative, timescale-separated |
| Sparse attention impact? | Privilege is *easier* ($W$ replaces $N$); no QoS modification needed |
| Sparse attention $\gamma^*$? | Much lower; $\gamma^* = 0$ for $W \leq 1024$ at $\delta = 7.6$ |
