# HCA-DS-5: Adaptive-Curvature Kernel Design Specification

## Executive Summary

This document specifies the adaptive-curvature kernel module that enforces the scaling constraint $R_{\max} \geq \frac{1}{2}\ln N$ at runtime, unifying DS-2's learnable per-layer curvature parameterization with DS-3's scaling requirement. The core mechanism is a dynamic curvature floor derived from the actual sequence length, applied as a hard `max()` over the DS-2 softplus output. A static floor based on $N_{\max}$ provides a permanent safety net; a lightweight EMA tracker of effective sequence length tightens the floor to actual operating conditions. The entire module is ~80 LOC, GPU-native, adds zero ops to the attention hot path, and integrates with the existing ~140 LOC custom optimizer stack from FA-3.

**Key results:**
- Closed-form mapping: $c_{\min}(N) = \cosh^2\!\bigl(\tfrac{1}{2}\ln N\bigr) / M^2 = (N + 2 + N^{-1}) / (4M^2)$
- Attention guarantee: $\alpha_s \geq 1/(1 + \sqrt{N})$ under $\frac{1}{2}\ln N$ scaling; strengthening to $\ln N$ yields an N-independent $\alpha_s \geq 1/2$
- Curvature does NOT explode: $c_{\min}(128\text{K}) \approx 3.2 \times 10^{-10}$, far below $c_{\text{init}} = 1.0$
- Per-layer enforcement with shared floor: maximum flexibility, guaranteed global bound

---

## D3 — Curvature ↔ R_max Closed-Form Mapping

All derivations are in the Lorentz (hyperboloid) model per DS-1.

### Setup

The hyperboloid of curvature $c > 0$ in $\mathbb{R}^{n+1}$:
$$\mathbb{H}_c^n = \{x \in \mathbb{R}^{n+1} : \langle x, x \rangle_{\mathcal{L}} = -1/c,\; x_0 > 0\}$$

where $\langle x, y \rangle_{\mathcal{L}} = -x_0 y_0 + \sum_{i=1}^{n} x_i y_i$.

The origin of the hyperboloid is $o = (1/\sqrt{c},\, 0, \ldots, 0)$.

For a point $x = (x_0, x_{\text{spatial}})$ on $\mathbb{H}_c^n$, the temporal component satisfies:
$$x_0 = \sqrt{1/c + \|x_{\text{spatial}}\|^2}$$

The hyperbolic distance between two points uses the **scaled distance** convention from DS-3:
$$D(x, y) = \operatorname{arcosh}(-c \langle x, y \rangle_{\mathcal{L}})$$

This equals $\sqrt{c} \cdot d_c(x,y)$ where $d_c$ is the intrinsic metric. All prior results (DS-2, DS-3) use this convention.

### Forward mapping: $(c, M) \to R_{\max}$

The maximum temporal coordinate is bounded by numerical precision: $x_{0,\max} = M$ where $M = 1/\varepsilon_{\text{norm}} = 10^7$ (FA-1 epsilon strategy).

The maximum scaled distance from the origin to any representable point:

$$R_{\max}(c, M) = \operatorname{arcosh}(\sqrt{c} \cdot M)$$

**Derivation:** For $x = (M, x_{\text{spatial}})$ and origin $o = (1/\sqrt{c}, \mathbf{0})$:
$$-c \langle o, x \rangle_{\mathcal{L}} = -c \cdot \bigl(-(1/\sqrt{c}) \cdot M\bigr) = \sqrt{c} \cdot M$$
$$D(o, x) = \operatorname{arcosh}(\sqrt{c} \cdot M)$$

For large argument ($\sqrt{c} \cdot M \gg 1$), $\operatorname{arcosh}(z) \approx \ln(2z)$, giving $R_{\max} \approx \ln(2\sqrt{c}\,M)$.

### Inverse mapping: target $R_{\max} \to$ minimum $c$

**Given:** $R_{\max} \geq R_{\text{target}}$. **Find:** minimum $c$.

$$\operatorname{arcosh}(\sqrt{c} \cdot M) \geq R_{\text{target}}$$
$$\sqrt{c} \cdot M \geq \cosh(R_{\text{target}})$$
$$c \geq \frac{\cosh^2(R_{\text{target}})}{M^2}$$

### Application: $R_{\text{target}} = \frac{1}{2} \ln N$

$$c_{\min}(N, M) = \frac{\cosh^2\!\bigl(\tfrac{1}{2}\ln N\bigr)}{M^2}$$

Expanding $\cosh\!\bigl(\tfrac{1}{2}\ln N\bigr)$:
$$\cosh\!\bigl(\tfrac{1}{2}\ln N\bigr) = \frac{e^{\frac{1}{2}\ln N} + e^{-\frac{1}{2}\ln N}}{2} = \frac{\sqrt{N} + 1/\sqrt{N}}{2}$$

$$\cosh^2\!\bigl(\tfrac{1}{2}\ln N\bigr) = \frac{(\sqrt{N} + 1/\sqrt{N})^2}{4} = \frac{N + 2 + 1/N}{4}$$

$$\boxed{c_{\min}(N, M) = \frac{N + 2 + 1/N}{4M^2}}$$

For large $N$: $c_{\min} \approx N/(4M^2)$, confirming DS-2's result.

### Numerical evaluation

| $N$ | $c_{\min}(N)$ | $R_{\max}$ at $c = 1.0$ | Floor active? |
|-----|---------------|--------------------------|---------------|
| 128 | $3.25 \times 10^{-13}$ | 16.12 | No ($c_{\text{init}} \gg c_{\min}$) |
| 1K | $2.51 \times 10^{-12}$ | 16.12 | No |
| 8K | $2.00 \times 10^{-11}$ | 16.12 | No |
| 32K | $8.00 \times 10^{-11}$ | 16.12 | No |
| 128K | $3.20 \times 10^{-10}$ | 16.12 | No |

At $c_{\text{init}} = 1.0$: $R_{\max} = \operatorname{arcosh}(10^7) \approx 16.12$, which satisfies $R_{\max} \geq \frac{1}{2}\ln(128000) \approx 5.88$ with enormous margin. The floor only becomes relevant if the optimizer drives $c$ down by $\sim$9 orders of magnitude.

### Parametric generalization

For $R_{\max} \geq \beta \ln N$ with tunable $\beta > 0$:
$$c_{\min}^{(\beta)}(N, M) = \frac{\cosh^2(\beta \ln N)}{M^2} \approx \frac{N^{2\beta}}{4M^2}$$

| $\beta$ | $c_{\min}(128\text{K})$ | Guarantee |
|---------|------------------------|-----------|
| 0.5 | $3.2 \times 10^{-10}$ | $\alpha_s \geq 1/\sqrt{N}$ |
| 0.75 | $1.8 \times 10^{-7}$ | $\alpha_s \geq N^{-1/4}$ |
| 1.0 | $4.1 \times 10^{-5}$ | $\alpha_s \geq 1/2$ (N-independent) |

All values remain far below $c_{\text{init}} = 1.0$, so the stronger $\beta = 1$ scaling is feasible at negligible curvature cost. See D4 for the full attention analysis.

---

## D1 — Enforcement Mechanism

### Mode A: Conservative Fixed Schedule

Pre-compute $c_{\text{floor}} = c_{\min}(N_{\max})$ before training. Apply as a hard lower bound:
$$c_{\text{eff}}^{(l)} = \max\!\bigl(c_{\text{floor}},\; c_{\min,\text{global}} + \operatorname{softplus}(c_{\text{raw}}^{(l)})\bigr)$$

where $c_{\min,\text{global}}$ is the DS-2 static minimum and $c_{\text{floor}}$ is the scaling-derived floor.

**Gradient flow:** $\partial c_{\text{eff}} / \partial c_{\text{raw}} = \sigma(c_{\text{raw}}) \cdot \mathbf{1}[c_{\text{learned}} > c_{\text{floor}}]$ where $\sigma$ is the sigmoid. When the floor is active, the gradient is exactly zero — the constraint binds and the optimizer cannot reduce $c$ further.

**Optimizer interaction:** With Adam, the second-moment accumulator for $c_{\text{raw}}$ decays toward zero during prolonged floor-active periods. When the floor eventually releases (e.g., shorter sequences), the first gradient step uses a stale second moment, potentially causing a large step. Mitigation: the $\lambda = 0.1$ LR multiplier from DS-2 bounds the maximum step size.

**Failure modes:** None beyond DS-2's catalog. The floor is a pure lower bound and cannot cause curvature explosion or gradient vanishing.

### Mode B: Fully Dynamic (Runtime-Adaptive)

Track the effective sequence length via an exponential moving average:
$$N_{\text{eff}}^{(t)} = \alpha \cdot N_{\text{eff}}^{(t-1)} + (1 - \alpha) \cdot N_{\text{batch}}^{(t)}$$

where $N_{\text{batch}}$ is the maximum sequence length in the current batch and $\alpha = 0.99$ (time constant ~100 steps).

Compute a dynamic floor:
$$c_{\text{floor}}^{(t)} = \frac{\cosh^2\!\bigl(\beta \ln N_{\text{eff}}^{(t)}\bigr)}{M^2}$$

#### Formulation comparison: softplus-input modulation vs. output floor

**Option 1 — Softplus-input modulation:**
$$c_{\text{eff}} = c_{\min} + \operatorname{softplus}\!\bigl(c_{\text{raw}} + \Delta(N_{\text{eff}})\bigr)$$

where $\Delta(N_{\text{eff}}) = \operatorname{softplus}^{-1}(c_{\text{floor}} - c_{\min}) - \operatorname{softplus}^{-1}(c_{\text{init}} - c_{\min})$ shifts the softplus input to match the floor.

- **Pro:** Smooth; gradient always flows through $c_{\text{raw}}$.
- **Con:** Changes the parameterization semantics — $c_{\text{raw}} = 0$ maps to different $c_{\text{eff}}$ at different $N_{\text{eff}}$. Creates a time-varying loss landscape that confuses Adam's momentum. The modulation $\Delta$ must be recomputed and backpropagated through, coupling curvature gradients to the EMA state.

**Option 2 — Output floor (hard max):**
$$c_{\text{learned}} = c_{\min} + \operatorname{softplus}(c_{\text{raw}})$$
$$c_{\text{eff}} = \max(c_{\text{learned}},\; c_{\text{floor}}(N_{\text{eff}}))$$

- **Pro:** DS-2 parameterization completely unchanged. Floor is a pure constraint — gradients flow normally when the learned value exceeds the floor, and are zero when the floor binds (correct behavior). No coupling between $c_{\text{raw}}$ and $N_{\text{eff}}$.
- **Con:** Zero gradient when floor is active. (This is a feature, not a bug — see gradient analysis above.)

**Recommendation: Option 2 (output floor).** The DS-2 parameterization is a prior design decision; extending it should be additive, not modifying. The `max()` operation is a single GPU instruction with no overhead.

### Hybrid Mode (Recommended Default)

Combine Mode A (static floor from $N_{\max}$) with Mode B (dynamic floor from $N_{\text{eff}}$):

$$c_{\text{floor,static}} = c_{\min}(N_{\max}) \quad \text{(computed once before training)}$$
$$c_{\text{floor,dynamic}}^{(t)} = c_{\min}(N_{\text{eff}}^{(t)}) \quad \text{(updated each forward pass)}$$
$$c_{\text{eff}}^{(l)} = \max\!\bigl(c_{\text{floor,static}},\; c_{\text{floor,dynamic}},\; c_{\min,\text{global}} + \operatorname{softplus}(c_{\text{raw}}^{(l)})\bigr)$$

The static floor guarantees the invariant even if the EMA lags or is reset. The dynamic floor tracks actual $N_{\text{eff}}$ for tighter constraint tracking during variable-length training.

In practice, $c_{\text{floor,dynamic}} \geq c_{\text{floor,static}}$ whenever $N_{\text{eff}} \geq N_{\max}$ (sequences longer than expected). Otherwise, $c_{\text{floor,static}}$ is the binding constraint and the dynamic path is a no-op.

### Comparison Table

| Property | Mode A (Static) | Mode B (Dynamic) | Hybrid (Default) |
|----------|-----------------|-------------------|-------------------|
| **Runtime overhead** | Zero (pre-computed scalar) | 1 EMA update + 1 `cosh²` per step | Same as Mode B |
| **Guarantee strength** | Exact for $N \leq N_{\max}$ | Exact if EMA tracks $N$ faithfully; lags on sudden jumps | Exact always (static floor catches EMA lag) |
| **Gradient flow** | Zero when floor active | Zero when floor active | Zero when floor active |
| **Variable-length support** | Conservative (uses worst-case $N_{\max}$) | Tracks actual $N_{\text{eff}}$ | Both: tight tracking + worst-case safety |
| **Phase 3 QoS extensibility** | None (static) | QoS can modulate $N_{\text{eff}}$ or inject a curvature bias | QoS hooks on the dynamic path; static floor remains as safety net |
| **Recommended for** | Fixed-length deployments | Research / variable-length training | Production default |

---

## D2 — Per-Layer vs. Global Curvature

### Analysis

The constraint $R_{\max} \geq \frac{1}{2}\ln N$ depends on $N$ (context length), which is identical across layers. The curvature floor $c_{\text{floor}}(N)$ is therefore a single scalar shared by all layers.

**Option 1 — Independent per-layer enforcement:**
$$c_{\text{eff}}^{(l)} = \max\!\bigl(c_{\text{floor}},\; c_{\min} + \operatorname{softplus}(c_{\text{raw}}^{(l)})\bigr)$$

Each layer $l$ has its own learnable $c_{\text{raw}}^{(l)}$. The shared floor $c_{\text{floor}}$ applies element-wise.

| Pro | Con |
|-----|-----|
| Maximum expressivity: early layers can use lower $c$ (broader context), deep layers higher $c$ (strict extraction) — exactly the DS-2 design intent | More parameters ($L$ scalars vs. 1) — negligible overhead |
| Floor is structurally enforced by `max()` — no layer can violate the bound regardless of optimizer behavior | Per-layer curvature monitoring requires $L$ tracked scalars |
| Identical to DS-2 parameterization with one additional `max()` | |

**Option 2 — Shared floor + per-layer offset:**
$$c_{\text{base}} = \max\!\bigl(c_{\text{floor}},\; c_{\min} + \operatorname{softplus}(c_{\text{raw,global}})\bigr)$$
$$c_{\text{eff}}^{(l)} = c_{\text{base}} + \operatorname{softplus}(\delta_{\text{raw}}^{(l)})$$

A single global $c_{\text{raw,global}}$ sets the base; per-layer offsets $\delta_{\text{raw}}^{(l)} \geq 0$ add curvature.

| Pro | Con |
|-----|-----|
| Global guarantee trivially satisfied: $c_{\text{eff}}^{(l)} \geq c_{\text{base}} \geq c_{\text{floor}}$ | Layers can only ADD curvature above the base, not reduce it — violates DS-2's "early layers broader context" principle |
| Fewer failure modes (offsets are always positive) | Two parameter types (global + offset) instead of one per-layer |
| | Reparameterization changes DS-2's structure |

### Interaction analysis

For layer $l$ with curvature $c^{(l)}$, the per-layer $R_{\max}$ is:
$$R_{\max}^{(l)} = \operatorname{arcosh}\!\bigl(\sqrt{c^{(l)}} \cdot M\bigr)$$

The constraint $R_{\max}^{(l)} \geq \frac{1}{2}\ln N$ is monotonically increasing in $c^{(l)}$, so $c^{(l)} \geq c_{\text{floor}}$ is necessary and sufficient for each layer independently. There is no cross-layer coupling in the constraint — each layer's guarantee is self-contained.

### Recommendation: Option 1 (Independent per-layer with shared floor)

**Justification:**
1. **Minimal extension of DS-2**: adds one `max()` operation per layer, no reparameterization.
2. **Structural guarantee**: the `max()` is in the forward pass, not a training-time penalty — the bound holds by construction at every step, including initialization.
3. **Maximum expressivity**: layers can independently negotiate curvature above the floor.
4. **No cross-layer coupling**: the constraint is per-layer separable, so independent enforcement is both correct and simpler.

---

## D4 — Attention Weight Guarantee

### Setup

System token $s$ at the origin $o = (1/\sqrt{c}, \mathbf{0})$. Query $q$ at scaled distance $\rho = D(q, o)$ from the origin. Context tokens $t_1, \ldots, t_{N-1}$ at scaled distances $\rho_i = D(t_i, o)$ from origin, with $\rho_i \leq R_{\max}$.

The attention weight for $s$ (using exponential-of-distance scoring from the softmax-arcosh cancellation, per the GPU cost model):
$$\alpha_s = \frac{\exp(-D(q,s))}{\exp(-D(q,s)) + \sum_{i=1}^{N-1} \exp(-D(q,t_i))}$$

where $D(x,y) = \operatorname{arcosh}(-c\langle x,y\rangle_{\mathcal{L}})$ is the scaled distance.

### Theorem (Privilege Floor under R_max Scaling)

**Under the enforced scaling $R_{\max} \geq \beta \ln N$ with $\beta > 0$, for the system token $s$ at the origin and all context tokens at scaled distance $\geq \delta$ from the origin:**

$$\alpha_s \geq \frac{1}{1 + (N-1) \cdot \exp(\rho - \delta)}$$

**In particular, for a query at the origin ($\rho = 0$) and context tokens at the boundary ($\delta = R_{\max} \geq \beta \ln N$):**

$$\alpha_s \geq \frac{1}{1 + (N-1) \cdot N^{-\beta}}$$

### Proof

By the reverse triangle inequality in the metric space $(\mathbb{H}_c^n, D)$:
$$D(q, t_i) \geq |D(o, t_i) - D(q, o)| = |\rho_i - \rho| \geq \delta - \rho$$

(the last inequality holds when $\delta \leq \rho_i$ and $\rho \leq \delta$, i.e., the query is closer to the origin than the context tokens).

Therefore:
$$\exp(-D(q, t_i)) \leq \exp(-(\delta - \rho)) = \exp(\rho - \delta)$$

Summing over all $N - 1$ context tokens:
$$\sum_{i=1}^{N-1} \exp(-D(q,t_i)) \leq (N-1) \cdot \exp(\rho - \delta)$$

The numerator satisfies $\exp(-D(q,s)) = \exp(-\rho)$. Since $f(x,y) = x/(x+y)$ is decreasing in $y$:
$$\alpha_s \geq \frac{\exp(-\rho)}{\exp(-\rho) + (N-1)\exp(\rho - \delta)} = \frac{1}{1 + (N-1)\exp(2\rho - \delta)}$$

Setting $\rho = 0$ (query at origin) and $\delta = R_{\max} \geq \beta \ln N$:
$$\alpha_s \geq \frac{1}{1 + (N-1) \cdot \exp(-\beta \ln N)} = \frac{1}{1 + (N-1) \cdot N^{-\beta}}$$

$\blacksquare$

### Corollaries

**Corollary 1** ($\beta = 1/2$, DS-3 scaling): $\alpha_s \geq \frac{1}{1 + (N-1)/\sqrt{N}} = \frac{\sqrt{N}}{N + \sqrt{N} - 1} \approx \frac{1}{\sqrt{N}}$

This is the **viability threshold** from DS-2: origin-token weight decays as $1/\sqrt{N}$, slower than standard attention's $1/N$.

**Corollary 2** ($\beta = 1$, strengthened scaling): $\alpha_s \geq \frac{1}{1 + (N-1)/N} \to \frac{1}{2}$

This is a **truly N-independent** lower bound. The curvature cost is $c_{\min} \approx N^2/(4M^2) = 4.1 \times 10^{-5}$ at $N = 128$K, still $\sim$4 orders of magnitude below $c_{\text{init}} = 1.0$.

**Corollary 3** (general query at distance $\rho$ from origin, $\beta = 1/2$):
$$\alpha_s \geq \frac{1}{1 + (N-1) \cdot \exp(2\rho)/\sqrt{N}}$$

This is $\Omega(1)$ (N-independent) when $\rho < \frac{1}{4}\ln N$, i.e., the query is in the **inner half** of the hyperbolic ball (by log-radius).

### Numerical evaluation (query at origin, context at boundary)

| $N$ | $\beta = 0.5$ bound | $\beta = 0.75$ bound | $\beta = 1.0$ bound | $c_{\min}(\beta\!=\!1)$ |
|-----|---------------------|----------------------|---------------------|-------------------------|
| 128 | 0.0811 | 0.190 | 0.502 | $4.1 \times 10^{-11}$ |
| 1K | 0.0307 | 0.151 | 0.501 | $2.5 \times 10^{-9}$ |
| 8K | 0.0111 | 0.108 | 0.500 | $1.6 \times 10^{-7}$ |
| 32K | 0.00558 | 0.0797 | 0.500 | $2.6 \times 10^{-6}$ |
| 128K | 0.00279 | 0.0581 | 0.500 | $4.1 \times 10^{-5}$ |

### Design recommendation

The deliverable requests "$\alpha_s \geq f(c, r_s)$ independent of $N$." This requires $\beta \geq 1$. Since $c_{\min}(\beta\!=\!1, N\!=\!128\text{K}) = 4.1 \times 10^{-5} \ll c_{\text{init}} = 1.0$, the curvature cost is negligible. **Recommend $\beta = 1$ as the default**, with $\beta$ exposed as a hyperparameter.

The enforcement mechanism (D1) is parameterized by $\beta$ throughout; setting $\beta = 1/2$ recovers the original DS-3 constraint.

---

## D5 — Failure Mode Catalog

### Scaling-specific failure modes

The four failure modes from DS-2 (collapse to origin, Euclidean recovery, gradient vanishing, learnable collapse) remain unchanged. The adaptive-curvature module introduces five additional failure surfaces.

### Failure mode × mitigation matrix

| # | Failure Mode | Trigger | Observable Symptom | Mitigation A | Mitigation B | Recommendation |
|---|-------------|---------|-------------------|-------------|-------------|----------------|
| S1 | **Curvature explosion as N grows** | $c_{\min}(N) = N^{2\beta}/(4M^2)$ grows polynomially with $N$ | $c_{\text{eff}}$ increases, $\cosh/\sinh$ in exp/log maps overflow | Not a practical risk: at $\beta = 1$, $c_{\min}(128\text{K}) = 4.1 \times 10^{-5}$, which is $\sim$4 OOM below $c_{\text{init}} = 1.0$. Even at $N = 10^9$, $c_{\min} \approx 2.5$. | Add a hard ceiling $c_{\max} = 100$ to prevent $c_{\text{eff}}$ from entering overflow territory for $\cosh(\sqrt{c} \cdot \|v\|)$ | Monitor only; add ceiling if training ever approaches $N > 10^6$ |
| S2 | **Softplus saturation at extreme N** | $c_{\text{floor}}$ exceeds $\operatorname{softplus}(c_{\text{raw}})$ for all layers | All layers lock to the floor; $c_{\text{raw}}$ gradients are zero across the board | Raise $c_{\text{init}}$ to $c_{\text{floor}} + 1.0$ when $c_{\text{floor}} > 0.5$ | Add soft penalty $L_{\text{curv}} = \lambda_c \max(0, c_{\text{floor}} - c_{\text{learned}})^2$ to provide gradient signal even when floor binds | Mitigation B: soft penalty as early warning, with hard floor as guarantee |
| S3 | **Gradient dead zone (clamp)** | Floor active for extended period → $c_{\text{raw}}$ gradient identically zero | $c_{\text{raw}}$ parameter drifts via weight decay while receiving no gradient; when floor releases, $c_{\text{eff}}$ jumps discontinuously | Disable weight decay on $c_{\text{raw}}$ when floor is active | Add soft penalty (same as S2-B) to maintain gradient signal below the floor | Mitigation A (disable weight decay on $c_{\text{raw}}$ always — it's a scalar, weight decay is meaningless) |
| S4 | **Batch N variability** | Variable-length sequences: some are 128 tokens, some are 32K, same batch | $c_{\text{floor}}$ computed from $\max(N_i)$ is overly conservative for short sequences; EMA lags sudden jumps to long sequences | Use $\max(N_i)$ within the batch for $c_{\text{floor}}$ computation (conservative, correct) | Maintain separate per-bucket EMA for length ranges | Mitigation A: use $\max(N_i)$ per batch. The EMA smooths across batches; within a batch, the longest sequence determines the constraint. |
| S5 | **Numerical instability at $c \to c_{\min}$** | Curvature approaches the floor ($c \approx 10^{-10}$ for $\beta = 0.5$) | exp/log maps operate on a nearly flat manifold; potential precision loss in $\cosh(\sqrt{c}\|v\|) \approx 1 + c\|v\|^2/2$ | **(a) Hybrid Euclidean fallback:** when $c < c_{\text{threshold}}$, bypass exp/log maps and use flat-space linear operations directly | **(b) Clamping + monitoring** inside the two custom `torch.autograd.Function` implementations (exp\_map, log\_map per DS-4): use Taylor expansion for $\sqrt{c}\|v\| < \varepsilon$ | See analysis below |

### S5 detailed analysis: low-curvature numerical stability

When $c$ is very small ($c \ll 1$), the Lorentz exp-map:
$$\exp_x(v) = \cosh(\sqrt{c}\,\|v\|_{\mathcal{L}})\, x + \frac{\sinh(\sqrt{c}\,\|v\|_{\mathcal{L}})}{\sqrt{c}\,\|v\|_{\mathcal{L}}}\, v$$

approaches the Euclidean limit smoothly:
- $\cosh(\sqrt{c}\,\|v\|) \to 1 + c\|v\|^2/2$
- $\sinh(\sqrt{c}\,\|v\|)/(\sqrt{c}\,\|v\|) \to 1 + c\|v\|^2/6$

This is the **benign** direction — small curvature is numerically easier, not harder. The `sinch` function ($\sinh(z)/z$) is well-conditioned near $z = 0$ and already has a Taylor fallback in the DS-4 exp\_map implementation.

The dangerous direction is $c \to \infty$ (handled by DS-2's gradient clipping + weight decay), not $c \to 0$.

**Evaluation of mitigations:**

| Mitigation | Overhead | Correctness | Complexity |
|-----------|----------|-------------|------------|
| (a) Euclidean fallback | Branch in forward pass; two code paths to maintain | Correct but discontinuous at the threshold — curvature crossing $c_{\text{threshold}}$ causes a discrete jump in the computational graph | ~40 LOC additional |
| (b) Clamping + Taylor | Zero overhead (already required by DS-4 epsilon strategy) | Continuous; Taylor expansion IS the low-$c$ limit of the exact formula | ~5 LOC (Taylor branch exists) |

**Recommendation: Mitigation (b) — clamping + monitoring.** The low-$c$ regime is numerically benign. The existing Taylor fallback in exp/log maps (DS-4) handles it correctly. Add a runtime counter that logs when $c < 10^{-6}$ for diagnostic purposes, but take no corrective action.

---

## D6 — Curvature Update Path

### Design

The curvature parameter $c_{\text{raw}}$ receives its own low-frequency update path, decoupled from the attention hot path.

### Update frequency

**Every $k = 100$ optimizer steps.**

Justification:
1. **Curvature changes are global geometry changes.** Adjusting $c$ reshapes the entire manifold, affecting all embeddings simultaneously. High-frequency changes cause the embedding space to "breathe," destabilizing attention patterns before they converge.
2. **Curvature gradients are noisy.** The gradient $\partial L / \partial c_{\text{raw}}$ flows through the entire attention computation and aggregates over all tokens. At any single step, this gradient is dominated by the specific batch content, not the true curvature optimum. Accumulating over 100 steps averages out batch-level noise.
3. **Combined with $\lambda = 0.1$ LR multiplier.** The effective curvature step size is $0.1 \times \text{base\_lr} / 100 = 10^{-3} \times \text{base\_lr}$ per underlying step — slow enough for stable geometry, fast enough to adapt over ~1000 steps.
4. **Typical training scale.** At 100K training steps, this gives ~1000 curvature updates — sufficient to learn per-layer curvature profiles while avoiding oscillation.

### Delta computation

```
# Accumulation (every step, inside loss.backward()):
c_grad_buffer += c_raw.grad.detach()     # fp32 accumulation
c_grad_count += 1
c_raw.grad = None                         # prevent main optimizer from stepping c_raw

# Application (every k steps):
avg_grad = c_grad_buffer / c_grad_count
c_raw.grad = avg_grad                     # set averaged gradient
optimizer.step()                           # optimizer steps c_raw with λ=0.1 LR
c_grad_buffer.zero_()
c_grad_count = 0
```

The gradient buffer is a single tensor of shape `(n_layers,)` — negligible memory. The accumulation is an `add_` operation on a small tensor — negligible compute.

### Integration with attention hot path

The curvature update path adds **zero operations** to the attention kernel's critical path:

1. $c_{\text{eff}}$ is computed once at the start of the forward pass (one `softplus` + one `max` per layer).
2. $c_{\text{eff}}$ is passed to the attention kernel as a **frozen scalar** (detached from the graph during the attention computation? No — gradients must flow through $c$ to reach $c_{\text{raw}}$. The scalar is part of the computation graph but adds only one multiply to the Lorentz inner product, which is already fused).
3. The gradient $\partial L / \partial c_{\text{raw}}$ is computed by standard autograd during `loss.backward()` — no additional backward pass needed.
4. The only "extra" work is the buffer accumulation (one `add_` on a small tensor) and the periodic optimizer step.

### Phase 3 QoS interface

```python
class CurvatureModule:
    def register_qos_hook(self, hook_fn: Callable) -> int:
        """Register a QoS hook called during curvature updates.

        hook_fn signature:
            (c_eff: Tensor[n_layers],
             c_grad_avg: Tensor[n_layers],
             N_eff: float,
             step: int) -> Optional[Tensor[n_layers]]

        Return value: additive adjustment to c_grad_avg, or None.

        Use cases:
        - QoS radial monitor detects tokens clustering at origin → increase c
          to expand R_max and push boundary tokens outward
        - QoS detects class separation degrading → decrease c to flatten
          the manifold and reduce distance distortion
        - QoS privilege monitor detects α_s dropping → increase c to
          strengthen origin privilege
        """
        hook_id = len(self._qos_hooks)
        self._qos_hooks.append(hook_fn)
        return hook_id
```

The hook is called during the curvature update step (every $k$ steps), NOT during the forward pass. QoS signals are accumulated between curvature updates and applied as gradient adjustments. This keeps the hot path clean while allowing QoS to steer curvature at the appropriate timescale.

---

## D7 — Implementation Sketch

### Module structure

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, List

class AdaptiveCurvatureModule(nn.Module):
    """Adaptive-curvature kernel enforcing R_max >= beta * ln(N).

    Extends DS-2 parameterization (c_eff = c_min + softplus(c_raw))
    with a dynamic curvature floor from DS-3 scaling constraint.
    ~80 LOC. GPU-native. Zero attention hot-path overhead.
    """

    def __init__(
        self,
        n_layers: int,
        c_init: float = 1.0,
        M: float = 1e7,
        beta: float = 1.0,        # R_max >= beta * ln(N); default=1 for N-independent bound
        N_max: int = 131072,       # 128K — static floor
        ema_decay: float = 0.99,
        update_every_k: int = 100,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.M = M
        self.beta = beta
        self.ema_decay = ema_decay
        self.update_every_k = update_every_k

        # Static floor from N_max (Mode A)
        c_floor_static = self._c_min_from_N(float(N_max))
        self.register_buffer('c_floor_static', torch.tensor(c_floor_static, dtype=torch.float32))

        # DS-2 global minimum (original, very small)
        c_min_global = N_max / (4.0 * M * M)
        self.register_buffer('c_min_global', torch.tensor(c_min_global, dtype=torch.float32))

        # Per-layer learnable curvature (DS-2 parameterization)
        # Initialize so that c_eff = c_init
        c_raw_init = torch.log(torch.expm1(torch.tensor(c_init - c_min_global)))
        self.c_raw = nn.Parameter(c_raw_init.expand(n_layers).clone())

        # EMA of effective sequence length
        self.register_buffer('N_eff', torch.tensor(128.0, dtype=torch.float32))
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))

        # Gradient accumulation buffer for low-frequency updates
        self.register_buffer('c_grad_buffer', torch.zeros(n_layers, dtype=torch.float32))
        self.register_buffer('c_grad_count', torch.tensor(0, dtype=torch.long))

        # QoS hooks (Phase 3 interface)
        self._qos_hooks: List[Callable] = []

    def _c_min_from_N(self, N: float) -> float:
        """Closed-form minimum curvature: cosh^2(beta * ln(N)) / M^2."""
        import math
        half_ln_N = self.beta * math.log(max(N, 2.0))
        return math.cosh(half_ln_N) ** 2 / (self.M ** 2)

    def _c_floor_dynamic(self, N_eff: torch.Tensor) -> torch.Tensor:
        """GPU-native dynamic floor computation."""
        half_ln_N = self.beta * torch.log(torch.clamp(N_eff, min=2.0))
        cosh_val = torch.cosh(half_ln_N)
        return (cosh_val * cosh_val) / (self.M * self.M)

    # ──────────────────────────────────────────────
    # Forward pass
    # ──────────────────────────────────────────────

    def forward(self, N_batch: int) -> torch.Tensor:
        """Compute per-layer effective curvature.

        Args:
            N_batch: maximum sequence length in the current batch.

        Returns:
            c_eff: Tensor of shape (n_layers,), fp32.
        """
        # --- Step 1: Update N_eff EMA (no grad, in-place) ---
        with torch.no_grad():
            N_t = torch.tensor(float(N_batch), device=self.N_eff.device, dtype=torch.float32)
            self.N_eff.mul_(self.ema_decay).add_(N_t, alpha=1.0 - self.ema_decay)
            self.step_count.add_(1)

        # --- Step 2: DS-2 learnable curvature (unchanged) ---
        c_learned = self.c_min_global + F.softplus(self.c_raw)    # (n_layers,)

        # --- Step 3: Dynamic floor from N_eff (Mode B) ---
        c_floor_dyn = self._c_floor_dynamic(self.N_eff)           # scalar

        # --- Step 4: Hybrid enforcement ---
        c_floor = torch.maximum(self.c_floor_static, c_floor_dyn) # scalar
        c_eff = torch.maximum(c_learned, c_floor.expand_as(c_learned))  # (n_layers,)

        return c_eff

    # ──────────────────────────────────────────────
    # Backward / gradient management
    # ──────────────────────────────────────────────

    def accumulate_and_gate_grad(self):
        """Call AFTER loss.backward(), BEFORE optimizer.step().

        Accumulates c_raw gradient into buffer and zeros it so the
        main optimizer does not step c_raw this iteration.

        Returns True when it's time to apply the curvature update.
        """
        if self.c_raw.grad is not None:
            self.c_grad_buffer.add_(self.c_raw.grad.detach())
            self.c_grad_count.add_(1)
            self.c_raw.grad = None  # gate: main optimizer skips c_raw

        return self.c_grad_count.item() >= self.update_every_k

    def apply_curvature_step(self, optimizer):
        """Apply accumulated curvature gradient. Call every k steps.

        Args:
            optimizer: the optimizer whose param group contains c_raw.
        """
        if self.c_grad_count.item() == 0:
            return

        avg_grad = self.c_grad_buffer / self.c_grad_count.float()

        # Phase 3 QoS hooks
        for hook_fn in self._qos_hooks:
            with torch.no_grad():
                c_eff = self.c_min_global + F.softplus(self.c_raw)
                adj = hook_fn(c_eff, avg_grad, self.N_eff.item(), self.step_count.item())
                if adj is not None:
                    avg_grad = avg_grad + adj

        # Set averaged gradient and let optimizer step
        self.c_raw.grad = avg_grad.clone()
        # Optimizer step for c_raw happens in the caller's optimizer.step()

        # Reset buffer
        self.c_grad_buffer.zero_()
        self.c_grad_count.zero_()

    # ──────────────────────────────────────────────
    # Phase 3 QoS interface
    # ──────────────────────────────────────────────

    def register_qos_hook(self, hook_fn: Callable) -> int:
        hook_id = len(self._qos_hooks)
        self._qos_hooks.append(hook_fn)
        return hook_id
```

### Training loop integration

```python
# --- Setup ---
curvature = AdaptiveCurvatureModule(n_layers=32, beta=1.0, N_max=131072)

# Separate param groups: main params at base LR, c_raw at 0.1x
optimizer = torch.optim.AdamW([
    {'params': model.parameters(), 'lr': 1e-4},
    {'params': [curvature.c_raw], 'lr': 1e-5},  # lambda=0.1 applied here
], weight_decay=0.01)

# Exclude c_raw from weight decay (D5-S3 mitigation)
optimizer.param_groups[1]['weight_decay'] = 0.0

# --- Training loop ---
for step, batch in enumerate(dataloader):
    N_batch = batch['input_ids'].shape[1]    # max seq len in batch

    # Forward: curvature module produces per-layer c_eff
    c_eff = curvature(N_batch)               # (n_layers,) tensor, in graph

    # c_eff is passed to each transformer layer's attention kernel
    logits = model(batch['input_ids'], c_eff=c_eff)
    loss = criterion(logits, batch['labels'])

    loss.backward()    # gradients flow through c_eff to c_raw

    # Gate curvature gradient: accumulate, don't step yet
    ready = curvature.accumulate_and_gate_grad()

    # Main optimizer steps everything EXCEPT c_raw (its grad is None)
    optimizer.step()
    optimizer.zero_grad()

    # Every k steps: apply accumulated curvature gradient
    if ready:
        curvature.apply_curvature_step(optimizer)
        # Now optimizer has c_raw.grad set; step just the curvature group
        optimizer.step()   # steps c_raw with accumulated avg gradient
        optimizer.zero_grad()
```

### LOC estimate and integration

| Component | LOC | Integrates with |
|-----------|-----|-----------------|
| `AdaptiveCurvatureModule` | ~80 | New module |
| Training loop modifications | ~15 | Existing training loop |
| Optimizer param group setup | ~5 | Existing optimizer config |
| **Total new code** | **~100** | |
| Existing custom optimizer stack (FA-3) | ~140 | Unchanged — curvature module is orthogonal |
| Existing custom exp/log maps (DS-4) | ~100 | Unchanged — receive $c_{\text{eff}}$ as input |

### GPU-native guarantees

| Constraint | How satisfied |
|-----------|--------------|
| No CPU round-trips | All tensors on GPU; `N_batch` is a Python int passed once per step |
| No Python-level per-token logic | `c_eff` is a per-layer scalar broadcast to all tokens via standard PyTorch ops |
| fp32 curvature arithmetic | `c_raw`, `c_eff`, and all floor computations are `dtype=torch.float32` |
| Inference-cheap | At inference: one `softplus` + one `max` per layer. No EMA update, no gradient accumulation |
| No attention hot-path ops | `c_eff` enters the attention kernel only as a scalar multiplier on the Lorentz inner product — fused into the existing GEMM |

---

## Appendix A — Summary of Design Decisions

| Decision | Choice | Alternatives considered | Justification |
|----------|--------|------------------------|---------------|
| Scaling exponent $\beta$ | 1.0 (default) | 0.5 (DS-3 original) | N-independent guarantee; curvature cost negligible |
| Enforcement mechanism | Output floor (`max`) | Softplus-input modulation | Preserves DS-2 parameterization; cleaner gradient semantics |
| Per-layer vs. global | Independent per-layer + shared floor | Shared base + offsets | Maximum expressivity; floor guarantees bound for all layers |
| Low-$c$ mitigation | Clamping + Taylor (existing) | Euclidean fallback | Low-$c$ is numerically benign; no additional code needed |
| Update frequency | $k = 100$ steps | Every step; every epoch | Balances noise averaging with adaptivity |
| Weight decay on $c_{\text{raw}}$ | Disabled | Enabled with small coefficient | Prevents drift during floor-active periods |
| EMA decay | $\alpha = 0.99$ (τ ≈ 100 steps) | 0.999 (τ ≈ 1000); 0.9 (τ ≈ 10) | Matches curvature update frequency for coherent tracking |

## Appendix B — Notation Reference

| Symbol | Definition |
|--------|-----------|
| $c$ | Lorentz curvature, $c > 0$ |
| $c_{\text{raw}}^{(l)}$ | Learnable unconstrained parameter for layer $l$ |
| $c_{\text{eff}}^{(l)}$ | Effective curvature after floor enforcement |
| $c_{\text{floor}}$ | Dynamic curvature floor from $N_{\text{eff}}$ |
| $M$ | Maximum temporal coordinate ($= 1/\varepsilon_{\text{norm}} = 10^7$) |
| $\beta$ | Scaling exponent in $R_{\max} \geq \beta \ln N$ |
| $N_{\text{eff}}$ | EMA of effective sequence length |
| $R_{\max}$ | Maximum scaled distance from origin: $\operatorname{arcosh}(\sqrt{c} \cdot M)$ |
| $D(x,y)$ | Scaled distance: $\operatorname{arcosh}(-c\langle x,y\rangle_{\mathcal{L}})$ |
| $\rho$ | Scaled distance of query from origin |
| $\alpha_s$ | Attention weight assigned to system token |
| $\lambda$ | LR multiplier for curvature ($= 0.1$) |
| $k$ | Curvature update frequency ($= 100$ steps) |
