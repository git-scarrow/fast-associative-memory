# Curvature Selection Design Specification

## 1. Executive Summary
This document specifies the design constraints and parameterization of curvature $c$ for the Hyperbolic Context Architecture (HCA). The parameter $c$ effectively dictates the dynamic range of available hyperbolic volume, controlling the privilege gradient between origin-bound tokens and boundary tokens. We recommend a **layer-wise learnable curvature** bounded strictly away from zero via a Softplus-plus-epsilon parameterization mapping to the Lorentz model, optimizing convergence while preventing numerical collapse. Initialization is recommended at $c=1.0$, with a critical minimum bound $c^*$ tied explicitly to the target maximum context length $N_{\max}$ and the numerical bounding constant $M$.

## 2. Derivation: Curvature and Privilege Gradient
Let the attention distance be explicitly parameterized by the numerical curvature $c > 0$ under the Lorentz model mapping:
$$ d_c(u,v) = \operatorname{arcosh}(-c \langle u, v \rangle_\mathcal{L}) $$
For a query token $q$ near the boundary defined by a maximum numerical value $M$ (via the epsilon strategy, such that the temporal coordinate $x_0 \le M$), and an origin-bound system token $s = (1/\sqrt{c}, 0, \ldots, 0)$, the maximum available distance is:
$$ R_{\max} = d_c(q, s) = \operatorname{arcosh}(\sqrt{c} x_0) \approx \ln(2\sqrt{c}M) $$
Using the hyperbolic law of cosines, the typical distance between two boundary points $q$ and $t$ expands to approximately $2 R_{\max}$. The attention weight of the origin token over $N$ boundary tokens evaluates to:
$$ w_s(c, N) = \frac{e^{-R_{\max}}}{e^{-R_{\max}} + N e^{-2R_{\max}}} \approx \frac{1}{1 + N e^{-R_{\max}}} $$
As the context length $N$ grows, the competitive penalty $N e^{-R_{\max}}$ dilutes $w_s$. To guarantee the origin token's weight does not fall below a functional threshold of $\frac{1}{\sqrt{N}}$, we require:
$$ \frac{1}{1 + N e^{-R_{\max}}} \ge \frac{1}{\sqrt{N}} \implies N e^{-R_{\max}} \le \sqrt{N} \implies e^{R_{\max}} \ge \sqrt{N} $$
Substituting our approximation for $R_{\max}$, we obtain:
$$ \ln(2\sqrt{c}M) \ge \frac{1}{2} \ln N \implies 2\sqrt{c} M \ge \sqrt{N} $$
Solving for $c$, we identify the critical curvature **$c^*$**:
$$ c \ge c^* = \frac{N}{4 M^2} $$
If $c$ falls below $c^*$, the bounded numerical extent of the manifold shrinks the dynamic range $R_{\max}$ so severely that volume expansion halts and $w_s(c, N)$ decays past the $1/\sqrt{N}$ viability threshold.

## 3. Analysis: Fixed vs. Learnable Curvature

| Feature | Fixed Curvature ($c=1.0$) | Learnable Curvature ($c_l$) |
|---------|---------------------------|----------------------------|
| **Simplicity** | Requires no parameterization bounds; mathematically perfect. | Requires strict clipping and bounding protocols. |
| **Adaptability** | Suboptimal if the true semantic manifold dimension $d_{\text{eff}}$ differs from $d$. | Adapts the privilege gradient iteratively to fit the true token density. |
| **Layer Dynamics**| Enforces homogeneous gradients across all depth layers. | Allows early layers to maintain broader context ($c$ shifts lower) and deep layers to enforce strict extraction ($c$ shifts higher). |
| **Failure Risk** | Extremely safe, entirely immune to geometric degradation. | Vulnerable to $c \to 0$ (Euclidean collapse) and $c \to \infty$ (gradient vanishing). |

**Recommendation:** **Learnable Per-Layer Curvature**. The capacity for layers to independently negotiate origin privilege dynamics outweighs the logic complexity. Strict bounds must be applied to prevent the identified failure modes.

## 4. Curvature Selection Rule
The curvature $c$ must be bounded structurally. For a target max context length $N_{\max}$ and numeric epsilon bound defined by $M = 1/\epsilon_{\text{norm}}$:
1. **Critical Minimum ($c_{\min}$):** $c_{\min} = \frac{N_{\max}}{4 M^2}$. This is the hard lower bound.
2. **Initialization:** Start at $c_{\text{init}} = 1.0$ (standard baseline) everywhere to normalize initial loss and avoid immediate vanishing gradients.
3. **Learning Rate:** Assign a scaled learning rate multiplier of $\lambda = 0.1$ relative to conventional affine parameters to prevent catastrophic curvature jitter during early warm-up operations.

## 5. Failure Mode Catalog

| Failure Mode | Cause (Parameter Regime) | Observable Symptom | Mitigation |
|--------------|-------------------------|--------------------|------------|
| **Collapse to origin** | $c < 0$ (Negative curvature scalar) | The value $-c\langle u, v\rangle_\mathcal{L}$ drops below 1. $\operatorname{arcosh}$ clamps to 0; distance becomes 0 and attention uniformly flat. | Structurally bound $c > 0$ utilizing a strictly positive mapping function. |
| **Euclidean recovery (Too flat)** | $c \to 0$ (Approaching zero) | Exponential volume expansion terminates; temporal safety dilution resurfaces linearly. Loss of origin retrieval. | Hard clamp with $c_{\min} = \frac{N_{\max}}{4 M^2}$. |
| **Gradient Vanishing** | $c \to \infty$ (Extremely large $c$) | Inside $\operatorname{arcosh}$, $-c\langle u, v\rangle$ becomes massive. Derivative of $\operatorname{arcosh}(z)$ scales as $1/z$, saturating FP32 and killing gradients. | Apply gradient clipping and weight decay specifically targeting $c_{raw}$. |
| **Learnable collapse** | Optimizer driving $c$ aggressively smaller | Privilege gradient fails; cross-entropy loss stalls early. | Apply smaller LR multiplier ($\lambda=0.1$) for $c$ parameters. |

## 6. Implementation Parameterization

To integrate this properly with the Lorentz model and FP32 arithmetic while satisfying $c > c_{\min}$, define a scalar parameter $c_{\text{raw}}$ initialized such that the effective $c_{\text{eff}} = 1.0$. 

**Definition:**
$$ c_{\text{eff}} = c_{\min} + \operatorname{softplus}(c_{\text{raw}}) $$
where $c_{\min} = \frac{N_{\max}}{4 M^2}$. For typical values ($N_{\max} = 128,000, M = 10^7$), $c_{\min} \approx 3.2 \times 10^{-10}$.

**Initialization:**
$$ c_{\text{raw}} = \operatorname{softplus}^{-1}(1.0 - c_{\min}) \approx \ln(e^{1.0} - 1) \approx 0.5413 $$

**Integration Point:**
During the forward pass, apply $u_{\text{spatial}}, v_{\text{spatial}} \in \mathbb{R}^d$ via mapping onto the $c_{\text{eff}}$ Lorentz hyperboloid constraints:
```python
# Assume u_spatial, v_spatial are d-dimensional vectors in tangent space
c = c_min + F.softplus(c_raw)

# Enforce Lorentz geometric constraints onto spatial projections
u_0 = torch.sqrt(1.0 / c + torch.sum(u_spatial ** 2, dim=-1, keepdim=True))
v_0 = torch.sqrt(1.0 / c + torch.sum(v_spatial ** 2, dim=-1, keepdim=True))

# Lorentz inner product: -u_0*v_0 + u_spatial*v_spatial
lorentz_dot = - (u_0 * v_0) + torch.sum(u_spatial * v_spatial, dim=-1, keepdim=True)

# Compute hyperbolic distance
# Clamping required to prevent edge-case NaN where numeric drift forces lorentz_dot > -1/c
dist = torch.acosh(torch.clamp(-c * lorentz_dot, min=1.0 + 1e-6))
```

## 7. References
1. Nickel, M., & Kiela, D. (2018). *Learning Continuous Hierarchies in the Lorentz Model*.
2. Chami, I. et al. (2019). *Hyperbolic Graph Convolutional Neural Networks*.
3. HCA-LS-1 Literature Survey (Internal): Poincaré & Lorentz Embeddings.
4. HCA-FA-1 Numerical Stability Analysis (Internal): FP32 Lorentz Requirement & Epsilon strategy bindings.
