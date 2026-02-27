# Mathematical Analysis of Hyperbolic Attention Claim

## 1. Setup & Notation

Let $\mathbb{D}_c$ denote the Poincaré ball model of hyperbolic space with curvature $c > 0$. The generalized Poincaré distance between two points $u, v \in \mathbb{D}_c$ is:

$$ d_H(u, v) = \frac{1}{\sqrt{c}} \operatorname{arcosh}\left(1 + \frac{2c \|u - v\|^2}{(1 - c\|u\|^2)(1 - c\|v\|^2)}\right) $$

We consider $N$ total tokens:
- A "system-prompt" token $s$ placed near the origin ($r_s = \|s\| \approx 0$).
- Context tokens $t_1, \dots, t_{N-1}$ placed at radii $r_i = \|t_i\|$.
- A designated maximum boundary Euclidean radius $r_{max} = \max_i r_i$, strictly bounded away from the boundary of the ball ($r_{max} < 1/\sqrt{c}$).
- A query token $q \in \mathbb{D}_c$, placed at some radius $r_q \le r_{max}$.

The attention weight applied to the system prompt $s$ by query $q$ is defined as:

$$ \alpha_s = \frac{\exp(-d_H(q, s))}{\exp(-d_H(q, s)) + \sum_{i=1}^{N-1} \exp(-d_H(q, t_i))} $$

Let $R_{max}$ denote the maximum hyperbolic distance from the origin for any token in this bounded configuration. Since the radius $r_{max}$ is strictly bounded away from $1/\sqrt{c}$, $R_{max}$ is a finite positive constant dependent ONLY on $c$ and $r_{max}$:
$$ R_{max} = \frac{1}{\sqrt{c}} \ln\left( \frac{1 + \sqrt{c}r_{max}}{1 - \sqrt{c}r_{max}} \right) < \infty $$

## 2. Main Result

**Theorem:** For any fixed curvature $c$, fixed maximum radius $r_{max} < 1/\sqrt{c}$, and any query $q$ within this bounded domain, the attention weight $\alpha_s$ is strictly bounded from above by:
$$ \alpha_s \le \frac{1}{1 + (N-1) \exp(-2 R_{max})} $$
Consequently, $\lim_{N \to \infty} \alpha_s = 0$.

The claim that $\alpha_s$ is bounded below by a positive constant independent of $N$ is **FALSE**.

## 3. Proof

By definition, $(X, d_H)$ is a metric space satisfying the triangle inequality. For any two points $x, y$ in the domain bounded by Euclidean radius $r_{max}$:
$$ d_H(x, y) \le d_H(x, 0) + d_H(0, y) \le R_{max} + R_{max} = 2 R_{max} $$

Therefore, for any given query $q$ and context token $t_i$, the hyperbolic distance is bounded from above by $2 R_{max}$. Because the exponential function is monotonically decreasing with respect to its negated argument, we obtain a universal lower bound on the contribution of each context token:
$$ \exp(-d_H(q, t_i)) \ge \exp(-2 R_{max}) $$

Summing this lower bound over all $N-1$ context tokens yields the minimal total attention mass allocated to the context:
$$ \sum_{i=1}^{N-1} \exp(-d_H(q, t_i)) \ge (N-1) \exp(-2 R_{max}) $$

For the numerator, since $d_H(q, s) \ge 0$ for any metric space, its maximum possible value is $\exp(0) = 1$. The attention weight function $f(x, y) = \frac{x}{x+y}$ is monotonically increasing with respect to $x > 0$ and monotonically decreasing with respect to $y > 0$. By substituting the supremum of the numerator ($x \le 1$) and the infimum of the denominator's sum ($y \ge (N-1)\exp(-2R_{max})$), we compute a strict upper bound on $\alpha_s$:

$$ \alpha_s = \frac{\exp(-d_H(q, s))}{\exp(-d_H(q, s)) + \sum_{i=1}^{N-1} \exp(-d_H(q, t_i))} $$
$$ \alpha_s \le \frac{1}{1 + \sum_{i=1}^{N-1} \exp(-d_H(q, t_i))} \le \frac{1}{1 + (N-1) \exp(-2 R_{max})} $$

Since $R_{max}$ is a finite positive constant dependent entirely on $c$ and $r_{max}$, the term $\exp(-2 R_{max})$ is a fixed constant $> 0$. As the context length $N$ approaches infinity, the denominator grows linearly, forcing the inequality $\alpha_s \to 0$. This rigorously disproves the claim. $\blacksquare$

## 4. Conditions & Edge Cases

- **Non-zero System Radius ($r_s > 0$)**: If $s$ is placed at $r_s > 0$, the upper bound of the numerator drops from $1$ to $\sim \exp(-d_H(q, s))$. This worsens the weight dilution ratio, pushing $\alpha_s$ toward 0 even faster.
- **Clustered Context Tokens**: Even if tokens are closely clustered at the boundary, they cannot exceed the maximal inter-token distance $2 R_{max}$. The triangle inequality bound is universal and holds regardless of the spatial density of $t_i$.
- **Query Placed at a Context Token ($q = t_1$)**: If the query lands exactly on a token, $\exp(-d_H(t_1, t_1)) = 1$. The remaining $N-2$ tokens still each contribute at least $\exp(-2 R_{max})$, resulting in a similar strict upper bound: $\alpha_s \le \exp(R_{max}) / [1 + (N-2)\exp(-2R_{max})]$, which still evaluates to $\lim_{N\to\infty} \alpha_s = 0$.

### Structural Modification to Restore the Bound
The fundamental issue is that $\sum \exp(-d_H(q,t))$ grows linearly with $N$ in any bounded fixed-radius domain. To restore a positive lower bound independent of $N$, the maximum distance $R_{max}$ must structurally scale as a function of the context length $N$. Specifically, if the model parameterization enforces an asymptotic boundary convergence such that $R_{max}(N) \ge \frac{1}{2} \ln(N)$, the exponential penalty $\exp(-2 R_{max})$ suppresses the $O(N)$ growth. This can be practically implemented by **learnable curvature scaling** $c(N)$ or context-dependent temperature scaling in the kernel.

## 5. Verification Plan

To numerically falsify the analytical claim and confirm the dilution effect, we will execute a synthetic attention map experiment testing identical conditions.

**Experiment Specifications:**
- Coordinate System: Poincaré or Lorentz implementation (for optimal float64 numerical stability).
- Variables fixed: $c = 1.0$, $d = 64$, $r_s = 0.0$ (origin).
- $r_{max} \in \{0.9, 0.999\}$ strictly bound boundary token placements.
- Context Sweep N: $\{128, 1000, 8000, 32000, 128000\}$.
- Configuration: $q$ placed tangentially near the boundary, $s$ at origin, $t_i$ drawn randomly from a uniform Euclidean distribution placed onto a spherical manifold $r=r_{max}$.

**Expected Outcome:**
As $N$ scales from 128 to 128,000, the calculated $\alpha_s$ will be shown empirically dropping down toward effectively $0$, validating the $O(1/N)$ decay mathematically proven in section 3. The weight dilution ratio $\alpha_s(Hyperbolic) / \alpha_s(Euclidean)$ will show that while hyperbolic geometry defers the onset of attention sink collapse by pushing mass to the boundary, it does not achieve true asymptotic immunity for bounded radii.

## 6. Summary & Implications for HCA

Analytically, standard Poincaré attention cannot provide scale-invariant "privilege" to an origin token without context-length conditioning. The original HCA-DS-2 "privilege gradient" approximation empirically observed resilience only because $R_{max}$ happened to be large enough relative to practical $N$ to mask the linear degradation inside standard finite float precision. Rigorous implementation of Hyperbolic Context Architecture will require either explicit runtime length-bucketing scaling the curvature $c$, or allowing radial clipping constraints to scale algorithmically with $\ln(N)$.
