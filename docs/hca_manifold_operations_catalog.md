# Hyperbolic Context Architecture (HCA): Möbius & Manifold Operations Catalog

Here is the precise, implementation-ready reference catalog of the Möbius and manifold operations required by the Hyperbolic Memory Adapter, adhering to your fp32 precision bounds and Lorentz-model design constraints.

## Section 1 — Operation Specifications

### 1. Möbius Addition ($\oplus_c$)

**Mathematical definition**
In the Poincaré ball model with curvature $c$:
$$ x \oplus_c y = \frac{(1 + 2c \langle x, y \rangle + c \|y\|^2)x + (1 - c \|x\|^2)y}{1 + 2c \langle x, y \rangle + c^2 \|x\|^2 \|y\|^2} $$

**Lorentz-model equivalent**
No direct algebraic equivalent exists for simple, associative vector addition that natively preserves the hyperboloid without projecting. In the Lorentz model, Möbius addition is functionally replaced by mapping the translated point to the tangent space and back, equivalent to parallel transport or applying a Lorentz boost that shifts the origin to $x$. Consequently, modern Lorentz implementations bypass direct "addition" in favor of exponential and logarithmic maps.

**Numerical hazards**
- **Denominator vanishing (cancellation):** In fp32, the denominator $1 + 2c \langle x, y \rangle + c^2 \|x\|^2 \|y\|^2$ can approach $0$ for boundary-proximate points having opposite vectors, causing explosive outputs.
- **Numerator vanishing (underflow):** As $x$ approaches the boundary, $\|x\|^2 \to 1/c$, causing $1 - c\|x\|^2 \to 0$ and artificially collapsing the vector influence of $y$.

**Mitigations**
- Avoid direct implementation where possible. If strictly required in Poincaré, clip the magnitude of input vectors such that $\|x\|, \|y\| \leq r_{\max} / \sqrt{c}$ (where $r_{\max} = 1 - 10^{-5}$) before performing the operation.
- Use the strictly stabler Lorentz model representation which structurally sidesteps the $1 - c\|x\|^2$ collapse.

**Adapter-layer usage**
- **Not used** directly in any of the four adapter layers, due to the preferential use of the Lorentz model, which relies on the combination of Exp/Log maps to accomplish Hebbian pulls and position updates.

**Implementation notes**
- **Custom code needed** if used, but practically bypassed. If a functional reference is required to validate equivalence during testing, use `geoopt.Manifold.mobius_add`.

---

### 2. Möbius Matrix-Vector Multiplication ($\otimes_c$)

**Mathematical definition**
In the Poincaré ball model with curvature $c$:
$$ M \otimes_c x = \frac{1}{\sqrt{c}} \tanh\left( \frac{\|Mx\|}{\|x\|} \text{artanh}(\sqrt{c}\|x\|) \right) \frac{Mx}{\|Mx\|} $$

**Lorentz-model equivalent**
A rigid linear transformation requires $M$ to safely preserve the Lorentz inner product (requiring an orthogonal Lorentz transformation/boost). For general dense weights, there is no direct equivalent. Instead, the operation is replaced by mapping to the origin's Euclidean tangent space: $\exp_0^c(M \log_0^c(x))$.

**Numerical hazards**
- **Division by zero:** $0/0$ `NaN` generation from terms $\frac{\|Mx\|}{\|x\|}$ and $\frac{Mx}{\|Mx\|}$ as either $\|x\| \to 0$ or $\|Mx\| \to 0$.
- **Boundary singularity:** $\text{artanh}(z)$ produces infinite gradients (or `NaN` in fp32) as $z \to 1$.

**Mitigations**
- Clamp the $\text{artanh}$ argument: restrict $\sqrt{c}\|x\|$ to a maximum of $r_{\max} = 1 - 10^{-5}$.
- Stabilize divisions by adding the safe norm buffer: replace $\|v\|$ with $\|v\| + \text{eps\_norm}$ where $\text{eps\_norm} = 10^{-7}$.

**Adapter-layer usage**
- **Not used** directly within Lorentz components. Conceptually, the combination of layer (d) Readout ($\log_0$) and layer (a) Projection ($\exp_0$) brackets a Euclidean linear matrix operation, which executes exactly the Lorentz equivalent mentioned above.

**Implementation notes**
- **Custom code** (Tangent-space dense layers). We rely on `exp_0` and `log_0` composability in custom modules rather than a direct Möbius matvec to ensure optimal computational graphing and fp32 stability. Reference implementations can use `geoopt.Manifold.mobius_matvec` for checks.

---

### 3. Exponential Map ($\exp_x$)

**Mathematical definition**
Projects a tangent vector $v \in \mathcal{T}_x\mathbb{H}_c^n$ onto the Poincaré ball manifold from base point $x$:
$$ \exp_x^c(v) = x \oplus_c \left( \tanh\left(\sqrt{c} \frac{\lambda_x^c}{2} \|v\|\right) \frac{v}{\sqrt{c}\|v\|} \right) \quad \text{where} \quad \lambda_x^c = \frac{2}{1 - c\|x\|^2} $$

**Lorentz-model equivalent**
For base $x \in \mathbb{H}_c^n$ and tangent $v \in \mathcal{T}_x\mathbb{H}_c^n$ (where Minkowski inner product $\langle x, v \rangle_{\mathcal{L}} = 0$):
$$ \exp_x^c(v) = \cosh(\sqrt{c} \|v\|_{\mathcal{L}}) x + \frac{1}{\sqrt{c}} \sinh(\sqrt{c} \|v\|_{\mathcal{L}}) \frac{v}{\|v\|_{\mathcal{L}}} $$

**Numerical hazards**
- **Overflow:** fp32 $\cosh$ and $\sinh$ exceed finite representation limits rapidly if the argument $\sqrt{c}\|v\|_{\mathcal{L}}$ exceeds $\sim 88$. 
- **Underflow $0/0$ division:** The scalar division $\frac{v}{\|v\|_{\mathcal{L}}}$ blows up as the tangent length vanishes ($v \to 0$).

**Mitigations**
- **Safe division:** Augment the denominator with the norm epsilon buffer: $\frac{v}{\|v\|_{\mathcal{L}} + \text{eps\_norm}}$ ($\text{eps\_norm} = 10^{-7}$). 
- **Taylor expansion for small $v$:** If $\|v\|_{\mathcal{L}} < \text{eps\_z}$ ($10^{-8}$), explicitly bypass to a tangent approximation to prevent precision loss.
- Limit max velocity $\|v\|_{\mathcal{L}}$ via gradient/output clipping before evaluating map.

**Adapter-layer usage**
- **(a) Euclidean→Poincaré projection layer:** *Required (fwd+bwd)* (specifically $\exp_0^c(v)$, embedding states).
- **(c) Radial position update (Hebbian pull):** *Required (fwd+bwd)* (translating position $x$ firmly along the tangent trajectory toward $y$).

**Implementation notes**
- **Custom code needed.** While `geoopt.Lorentz.expmap` exists, standard geoopt does not aggressively surface epsilon clipping bounds within deep inner passes. A custom $\exp$ formulation is strictly required to enforce adherence to $\text{eps\_norm}$ and $\text{eps\_z}$ limits during large-scale batched backward passes.

---

### 4. Logarithmic Map ($\log_x$)

**Mathematical definition**
Projects a point $y \in \mathbb{H}_c^n$ into the tangent space at base point $x \in \mathbb{H}_c^n$ (Poincaré representation):
$$ \log_x^c(y) = \frac{2}{\sqrt{c} \lambda_x^c} \text{artanh}(\sqrt{c} \|-x \oplus_c y\|) \frac{-x \oplus_c y}{\|-x \oplus_c y\|} $$

**Lorentz-model equivalent**
$$ \log_x^c(y) = \frac{d_c(x, y)}{\sinh(\sqrt{c} d_c(x, y))} \left( y - \cosh(\sqrt{c} d_c(x, y)) x \right) $$
where $d_c(x, y) = \frac{1}{\sqrt{c}} \text{arcosh}(-c \langle x, y \rangle_{\mathcal{L}})$.

**Numerical hazards**
- **Gradient explosion at origin equality:** The derivative of $\text{arcosh}(z)$ is $\frac{1}{\sqrt{z^2 - 1}}$, which explodes to infinity as $x \to y$ (meaning $-c \langle x, y \rangle_{\mathcal{L}} \to 1$).
- **Sinh collapse:** As distance $d_c(x, y) \to 0$, $\sinh(\sqrt{c} d_c(x, y)) \to 0$, creating a severe $0/0$ instability in the scaling fraction. 

**Mitigations**
- **Arcosh clamping:** Forcefully clamp the argument of $\text{arcosh}$ to minimum $1 + \text{eps\_z}$ ($1 + 10^{-8}$).
- **Sinh denominator buffer:** Clamp the evaluating string $\sinh(\sqrt{c} d_c(x, y))$ tightly from below with $\text{eps\_sinh} = 10^{-6}$.

**Adapter-layer usage**
- **(c) Radial position update (Hebbian pull):** *Required (fwd+bwd)* (to compute the tangent direction vector pointing from $x$ toward target system prompt memory $y$).
- **(d) Poincaré→Euclidean readout layer:** *Required (fwd+bwd)* (specifically $\log_0^c$ to map Lorentz coordinate activations safely back to a Euclidean layer for logits processing).
- **(b) Hyperbolic attention kernel:** *Not used.* By design priority, Arcosh breakdown limits are sidestepped by directly returning $d_{sq} = -c \langle x, y \rangle_{\mathcal{L}}$ (or a linear surrogate pseudo-distance).

**Implementation notes**
- **Custom code needed.** The exact, bounded injection of $\text{eps\_z}$ and $\text{eps\_sinh}$ inside backward-critical pathing necessitates a completely custom PyTorch `torch.autograd.Function` (or exact tensor equivalent) implementation. Standard references like `geoopt.Lorentz.logmap` lack the ruggedized fp32 stability hooks necessary for your hybrid optimizer flow.

---

## Section 2 — Adapter-Layer $\to$ Operation Mapping Table

| Adapter Layer | Möbius Add ($\oplus_c$) | Möbius Matvec ($\otimes_c$) | Exp Map ($\exp_x$) | Log Map ($\log_x$) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **(a) Euclidean→Poincaré projection** | Not used | Not used | Required (fwd+bwd) | Not used | Uses specialized $\exp_0^c$ to map ambient space vector. |
| **(b) Hyperbolic attention kernel** | Not used | Not used | Not used | Not used | Uses squared Lorentz inner product directly, sidestepping $\log$, arcosh, and distances entirely. |
| **(c) Radial position update** | Not used | Not used | Required (fwd+bwd) | Required (fwd+bwd) | Calculates direction via $\log_x(y)$ and translates point via $\exp_x$. Native Möbius equivalent replaced. |
| **(d) Poincaré→Euclidean readout** | Not used | Not used | Not used | Required (fwd+bwd) | Uses specialized $\log_0^c$ to map back to flat classification space. |

---

## Section 3 — Implementation Notes Summary

| Operation | geoopt primitive | Custom code needed | Rationale |
| :--- | :--- | :---: | :--- |
| **Möbius addition** | `geoopt.Manifold.mobius_add` | No | Not used in core computational pipeline; only valid as a unit-test mathematical reference. |
| **Möbius matvec** | `geoopt.Manifold.mobius_matvec` | No | Replaced functionally by Tangent-Euclidean layers bracketed by Exp/Log mapping. |
| **Exp map** | `geoopt.Lorentz.expmap` | **Yes** | Requires custom deep clamping for $\text{eps\_norm} = 10^{-7}$ and overflow avoidance for large $v$ in scale/batch runs. |
| **Log map** | `geoopt.Lorentz.logmap` | **Yes** | Absolute necessity to inject $\text{eps\_z} = 10^{-8}$ directly into `arcosh` argument and $\text{eps\_sinh} = 10^{-6}$ natively into denominator for fp32 safety. |

> *Potential extensions note:* If you introduce intermediate generic MLPs wholly inside the manifold down the line, these components *would* formally require Tangent-space approximations of Möbius operations using cascaded $\log_x \to \text{Euclidean MLP} \to \exp_x$ logic, but remain explicitly omitted inside the adapter's direct constraints.
