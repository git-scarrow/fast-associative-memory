# Reference Catalog of Möbius/Manifold Operations

This document specifies the four fundamental Möbius/manifold operations required for the Hyperbolic Context Architecture (HCA) memory adapter. The conceptual space is the Poincaré ball $\mathbb{D}_c$, while implementations utilize the Lorentz hyperboloid model $\mathbb{H}_c$ for fp32 stability.

## Section 1 — Operation Specifications

### 1. Möbius Addition ($\oplus_c$)

| Field | Description |
|---|---|
| **Mathematical definition** | $x \oplus_c y = \frac{(1 + 2c\langle x, y\rangle + c\|y\|^2)x + (1 - c\|x\|^2)y}{1 + 2c\langle x, y\rangle + c^2\|x\|^2\|y\|^2}$ <br>*(Sources: Ungar 2005; Ganea et al. 2018)* |
| **Lorentz-model equivalent** | No direct arithmetic equivalent. In the Lorentz model, addition corresponds to pseudo-orthogonal transformations known as Lorentz boosts that preserve the Minkowski inner product *(Nickel & Kiela 2018)*. |
| **Numerical hazards** | The denominator $1 + 2c\langle x, y\rangle + c^2\|x\|^2\|y\|^2$ approaches $0$ locally when $x$ and $y$ are close to the boundary ($1/\sqrt{c}$) and point in opposite directions, causing fp32 $1/0$ infinities or catastrophic cancellation. The conformal factor $(1 - c\|x\|^2)$ degrades precision as norms near the boundary. |
| **Mitigations** | Restrict maximum representation norms using the strategy $r_{max} = 1 - 10^{-5}$. Add `eps_norm = 1e-7` safely to the denominator before the fp32 division. Since Lorentz boosts are preferred, spatial components must be explicitly clamped before time-component reconstruction. |
| **Adapter-layer usage** | **Not used** directly within the core adapter layers. While conceptually replacing vector addition in continuous RNN-style accumulations, the HCA attention kernel fundamentally relies on hyperbolic distances, bypassing explicit additions. |
| **Implementation notes** | Use `geoopt.PoincareBall.mobius_add` as a reference point only. A custom implementation is unnecessary since it does not participate directly in the core forward/backward computations of the four listed adapter stages. |

### 2. Möbius Matrix-Vector Multiplication ($\otimes_c$)

| Field | Description |
|---|---|
| **Mathematical definition** | $M \otimes_c x = \frac{1}{\sqrt{c}} \tanh\left( \frac{\|Mx\|}{\|x\|} \operatorname{artanh}(\sqrt{c}\|x\|) \right) \frac{Mx}{\|Mx\|}$ <br>*(Source: Ganea et al. 2018)* |
| **Lorentz-model equivalent** | Project the continuous linear transform $M$ onto the tangent space at the origin $T_0\mathbb{H}_c$, apply standard Euclidean matrix multiplication, and map the resultant vector back onto the hyperboloid using $\exp_0^c(Mx)$. |
| **Numerical hazards** | In fp32, dual division-by-zero risks occur if $x = 0$ (the origin) or if the matrix output $Mx = 0$. Additionally, $\operatorname{artanh}(z)$ generates `inf` as the argument $z \to 1$ (token encroaching upon the boundary). |
| **Mitigations** | Clamp the argument to $\operatorname{artanh}$ to $1 - 10^{-5}$ as dictated by the $r_{max}$ margin. Ensure denominators $\|x\|$ and $\|Mx\|$ are boosted with `eps_z = 1e-8`. If $x = 0$, explicitly bypass to $M \otimes_c 0 = 0$ to guarantee continuity. |
| **Adapter-layer usage** | **Required (fwd)** by the **(a) Euclidean→Poincaré projection layer**, serving as the initial continuous mapping of Euclidean KV input sequences into hyperbolic representations. |
| **Implementation notes** | Custom code needed. `geoopt` provides `mobius_matvec` for Poincaré manifolds, but utilizing tangent-space linear mapping (Euclidean matmul in $T_0$) followed by the Lorentz exponential map is mandatory to retain fp32 compatibility. |

### 3. Exponential Map ($\exp_x^c$)

| Field | Description |
|---|---|
| **Mathematical definition** | Maps a tangent vector $v \in T_x\mathbb{D}_c$ explicitly to a point on the manifold: <br> $\exp_x^c(v) = x \oplus_c \left( \tanh\left(\sqrt{c}\frac{\lambda_x^c\|v\|}{2}\right) \frac{v}{\sqrt{c}\|v\|} \right)$ where $\lambda_x^c = \frac{2}{1 - c\|x\|^2}$. <br>*(Sources: Ganea et al. 2018)* |
| **Lorentz-model equivalent** | For $x \in \mathbb{H}_c, v \in T_x\mathbb{H}_c$: <br> $\exp_x^c(v) = \cosh(\sqrt{c}\|v\|_\mathcal{L}) x + \frac{1}{\sqrt{c}}\sinh(\sqrt{c}\|v\|_\mathcal{L})\frac{v}{\|v\|_\mathcal{L}}$ <br> where $\|v\|_\mathcal{L} = \sqrt{\langle v, v\rangle_\mathcal{L}}$. *(Source: Nickel & Kiela 2018)* |
| **Numerical hazards** | In the Poincaré model, the conformal matrix $\lambda_x^c$ explodes into `inf` as $c\|x\|^2 \to 1$. In Lorentz, $\cosh$ and $\sinh$ exponentially overflow standard fp32 ranges beyond arguments of $\approx 88.7$. When $\|v\|_\mathcal{L} \to 0$, $v/\|v\|_\mathcal{L}$ results in $0/0$ yielding `NaN` in gradients. |
| **Mitigations** | Always execute via Lorentz parameterization. Explicitly clamp $v$ norms to prevent fp32 $\sinh/\cosh$ overflow. Add `eps_sinh = 1e-6` to the denominator of $\|v\|_\mathcal{L}$. Crucially, construct a custom analytical backward pass mapping the limit as $\|v\|_\mathcal{L} \to 0$ to avoid automatic `NaN` pollution. |
| **Adapter-layer usage** | **Required (fwd)** for the **(a) Euclidean→Poincaré projection layer** (specifically projecting from $T_0$ via $\exp_0^c$). <br> **Required (fwd+bwd)** for **(c) Radial position update (Hebbian pull)**, stepping position estimates across the manifold. |
| **Implementation notes** | Use custom code / `autograd.Function`. While `geoopt.Lorentz.expmap` is standard, an explicit bespoke forward/backward kernel is necessitated to override the unhandled $\sinh(0)$ denominator logic for fp32 environments. |

### 4. Logarithmic Map ($\log_x^c$)

| Field | Description |
|---|---|
| **Mathematical definition** | Maps a point $y \in \mathbb{D}_c$ back strictly onto a tangent vector $v \in T_x\mathbb{D}_c$: <br> $\log_x^c(y) = \frac{2}{\sqrt{c}\lambda_x^c} \operatorname{artanh}(\sqrt{c}\|-x \oplus_c y\|) \frac{-x \oplus_c y}{\|-x \oplus_c y\|}$. <br>*(Source: Ganea et al. 2018)* |
| **Lorentz-model equivalent** | For $x, y \in \mathbb{H}_c$: <br> $\log_x^c(y) = \frac{d_c^\mathcal{L}(x, y)}{\sinh(\sqrt{c} d_c^\mathcal{L}(x, y))} \left(y - \cosh(\sqrt{c} d_c^\mathcal{L}(x, y))x\right)$ <br> where $d_c^\mathcal{L}(x, y) = \frac{1}{\sqrt{c}} \operatorname{arcosh}(-c\langle x, y \rangle_\mathcal{L})$. *(Source: Nickel & Kiela 2018)* |
| **Numerical hazards** | **Most dangerous operation in HCA.** The term $\operatorname{arcosh}(z)$ exhibits infinite derivatives as $z \to 1$ ($x \approx y$), causing catastrophic gradient explosion and Adam `NaN` death. fp16 $\operatorname{arcosh}$ explicitly overflows its 65504 maximum. $\sinh(0) \to 0$ adds a secondary denominator explosion. |
| **Mitigations** | For the attention core, entirely circumvent the $\log_x^c$ map and use squared Lorentz distances $-( \langle x, y \rangle_\mathcal{L} + 1/c )$, eliminating $\operatorname{arcosh}$ completely. For true readout projections, employ `eps_sinh = 1e-6` in the denominator and strict clamping $-c\langle x, y \rangle_\mathcal{L} \ge 1 + 10^{-7}$. |
| **Adapter-layer usage** | **Required (fwd+bwd)** for the **(d) Poincaré→Euclidean readout layer**, projecting output hyperbolic hidden states inversely back to $T_0\mathbb{H}_c \cong \mathbb{R}^d$ for injection to subsequent MLPs. <br> **Not used** in **(b) Hyperbolic attention kernel**, which uses squared distance instead. |
| **Implementation notes** | Custom code purely. Due to `arcosh`'s backward explosion, a highly optimized `logmap0` (a specialized optimization mapping back implicitly to the origin) must be custom-written to decouple from `geoopt.Lorentz.logmap`. |

---

## Section 2 — Adapter-Layer $\to$ Operation Mapping Table

| Adapter Layer | Möbius Addition ($\oplus_c$) | Möbius Matvec ($\otimes_c$) | Exp Map ($\exp_x^c$) | Log Map ($\log_x^c$) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **(a) Euclidean→Poincaré projection** | Not used | Required (fwd) | Required (fwd) | Not used | Solved via $T_0$ Euclidean Matvec injected into $\exp_0^c$. |
| **(b) Hyperbolic attention kernel** | Not used | Not used | Not used | Not used | Explicitly replaced by Lorentz squared inner product. |
| **(c) Radial position update (Hebbian pull)** | Not used | Not used | Required (fwd+bwd) | Not used | Used iteratively by Riemannian SGD optimization loop. |
| **(d) Poincaré→Euclidean readout** | Not used | Not used | Not used | Required (fwd+bwd) | Projects back specifically to $T_0\mathbb{H}_c$ via $\log_0^c$. |

---

## Section 3 — Implementation Notes Summary

| Operation | `geoopt` primitive | Custom code needed | Rationale |
| :--- | :--- | :--- | :--- |
| **Möbius Add** | `geoopt.PoincareBall.mobius_add` | No | Operation avoided in main loop. `geoopt` suffices for offline inspection/testing. |
| **Möbius Matvec** | `geoopt.PoincareBall.mobius_matvec` | **Yes** | Crucial fp32 stability mandates tangent $T_0\mathbb{H}_c$ injection with pure Lorentz models rather than native Poincaré operations. |
| **Exp Map** | `geoopt.Lorentz.expmap` | **Yes** | Fp32 limitation guarantees $\sinh(0)/\|v\| = \text{NaN}$ in backward pass natively; requires `eps_sinh=1e-6` injection. |
| **Log Map** | `geoopt.Lorentz.logmap` | **Yes** | Catastrophic Adam explosion inside `arcosh(1)` backprop strictly mandates analytical fallback inside `logmap0`. |

---

> **Potential extensions:**
> - *Parallel Transport ($P_{x \to y}(v)$)*: If non-origin MLPs inside the manifold are desired before readout, parallel transport becomes mandatory to migrate tangent vectors dynamically between distinct token coordinate bases. 
> - *Fréchet / Einstein Midpoints*: Required strictly if mean pooling operations (e.g. sequence-to-vector compression) traverse the adapter later.
