# Riemannian Optimization for HCA Adapter Training

## 1 — Riemannian SGD vs. Euclidean SGD + Retraction

### Conceptual distinction

**Riemannian SGD (RSGD)** computes the Riemannian gradient (the Euclidean gradient projected onto the tangent space at the current point), moves along a geodesic via the exponential map, then optionally applies parallel transport to carry momentum across steps:

```
x_{t+1} = exp_{x_t}( -η · grad_R f(x_t) )
grad_R f = grad_E f - ⟨grad_E f, x_t⟩_L · x_t    # Lorentz projection
```

The retraction `exp_x` keeps iterates exactly on the manifold by construction, at the cost of one exp-map call per step.

**Euclidean SGD + retraction correction** performs a standard Euclidean step in the ambient space (ℝ^{n+1} for Lorentz) and then projects back:

```
x̃_{t+1} = x_t - η · grad_E f(x_t)
x_{t+1} = retract(x̃_{t+1})          # e.g., normalize to hyperboloid
```

For the Lorentz model the cheapest retraction is the quadratic renormalization:

```
retract(x) = x / sqrt(|⟨x,x⟩_L|)
```

which is O(n) and differentiable, making it autograd-friendly.

### Trade-off analysis

| Criterion | RSGD (exp-map based) | Euclidean + Retraction |
|---|---|---|
| **Per-step cost** | ~2× (grad projection + exp-map) | ~1.1× (grad + one sqrt renorm) |
| **Manifold constraint satisfaction** | Exact (up to fp32 rounding) | Exact after retraction; drift between steps is bounded |
| **Numerical drift (long runs)** | Very low; exp-map is norm-preserving | Negligible with renorm retraction; can accumulate if retraction is skipped |
| **Adam / adaptive methods** | Riemannian Adam requires parallel transport of second moments — expensive and numerically fragile | Drop-in: standard Adam updates ambient coordinates, retract after each step |
| **Implementation complexity** | High: need exp, log, parallel transport; must handle `arcosh` domain | Low: retraction is one line; backward pass is automatic |
| **Gradient correctness** | Riemannian gradient is the correct manifold object | Euclidean gradient in ambient space; manifold geometry not respected during the step itself |
| **Interaction with learnable curvature** | Curvature enters exp-map formula explicitly; gradients through `c` flow correctly | Curvature only enters loss/attention kernels; retraction is curvature-agnostic |
| **Mixed precision safety** | exp-map uses `arcosh`; must be fp32. Tight | Retraction is a sqrt; easy to gate at fp32 |

### Lorentz-specific considerations

The Lorentz constraint `⟨x,x⟩_L = −1` is a *quadratic* surface. The retraction (rescaling by the pseudo-norm) is exact and costs a single division. This is cheaper than the Poincaré ball retraction (which involves a norm clamp) and far cheaper than computing a full geodesic. For the Lorentz model, the argument that "RSGD is necessary for correctness" is weaker than in, say, the SPD manifold case, because:

1. The quadratic constraint is so cheap to enforce that the gap between "ambient step + retract" and "true geodesic step" is small per iteration.
2. The Lorentz exp-map involves `sinh`/`cosh` of the tangent-vector norm and a `1/‖v‖` factor that becomes singular when `v → 0` (a common initialization state), requiring a guarded fallback.
3. Parallel transport on Lorentz is analytic but adds per-layer overhead and doesn't fit cleanly into PyTorch's optimizer abstraction without custom parameter classes.

**Bottom line:** For the HCA adapter (lightweight projection layers + attention head, frozen base), Euclidean Adam + per-step Lorentz retraction is the pragmatic choice. Full RSGD is theoretically cleaner but its practical advantages are negligible for low-depth adapters and its `arcosh`/parallel-transport machinery adds fragility.

---

## 2 — Library Comparison

### Comparison table

| Criterion | **geoopt** | **geomstats** | **hyptorch** |
|---|---|---|---|
| **Lorentz native support** | Yes — `Lorentz` manifold class, full implementation | Partial — `Hyperboloid` class exists but API is less ergonomic for PyTorch | Partial — implements `PoincareBall`; Lorentz available but thinner |
| **Riemannian optimizers** | RSGD, RAdam, RAdagrad (all manifold-aware) | RSGD via `RiemannianGradientDescent`; Adam support patchy | None built-in; must wrap geoopt or implement |
| **Exp map (Lorentz)** | `geoopt.manifolds.Lorentz.expmap` — tested, numerically guarded | `geomstats.geometry.hyperboloid.Hyperboloid.exp` — correct but slower (numpy heritage) | Custom, minimal guard code |
| **Log map (Lorentz)** | `geoopt.manifolds.Lorentz.logmap` — `arcosh`-guarded | Available; same caveats | Minimal implementation |
| **GPU/PyTorch integration** | Native PyTorch tensors; `ManifoldParameter` wraps `nn.Parameter` directly | Backends: numpy default, PyTorch via `geomstats.backend.set_backend("pytorch")` — some ops still go through numpy in 2025 | Pure PyTorch; lightweight |
| **Riemannian Adam** | `geoopt.optim.RiemannianAdam` — parallel transport implemented | Not production-ready as of early 2026 | Not available |
| **Maintenance (early 2026)** | Active; last commit <3 months; used in academic papers | Active but slower iteration; focus is on JAX/sklearn ecosystem | Effectively unmaintained (last meaningful commit ~2022) |
| **Mixed-precision compatibility** | Requires explicit `dtype=torch.float32` for manifold params; non-manifold params can be bf16/fp16 | Same restriction; backend switch adds risk | No explicit mixed-prec support |
| **Custom code surface for Lorentz** | ~0 lines for standard ops | ~50–100 lines (backend wrappers, manifold registration) | ~150–200 lines (exp/log, optimizer, parallel transport) |

### Prose evaluation

**geoopt** is the clear leader for this use case. Its `ManifoldParameter` integrates directly with PyTorch's `nn.Module`/`nn.Parameter` API: you declare a parameter as living on a manifold, and the optimizer automatically applies the Riemannian gradient and retraction. The `Lorentz` manifold class handles `expmap`, `logmap`, `inner` (Minkowski product), `transp`, and `dist` — every primitive the HCA adapter needs. `RiemannianAdam` is available and tested. The main caveat is that manifold parameters must be kept in fp32, which requires `autocast` regions to exclude them explicitly.

**geomstats** is better suited to scientific computing workflows (JAX, sklearn pipelines) than to training custom LLM adapters. The PyTorch backend is functional but the library was designed numpy-first; performance-sensitive inner loops sometimes round-trip through CPU tensors. For a production training loop this is a liability.

**hyptorch** served a purpose in 2019–2021 when geoopt was less mature. It is now effectively superseded and should not be used for new projects.

---

## 3 — Exp/Log Map Handling in the Adapter Graph

### Where maps are needed

```
Frozen LLM
  └─ KV hidden states  [B, H, T, d_model]   ← Euclidean, fp16/bf16 ok
        │
        ▼
  [EUC→LOR projection]  (Linear + Lorentz normalization)
        │  exp_map from origin (0-map): maps tangent vector at origin → hyperboloid
        │  forward: x_L = exp_o(W · h_E)
        │  backward: ∂L/∂W flows through exp_o autograd
        ▼
  Lorentz KV  [B, H, T, d_hyp]   ← fp32 required
        │
        ▼
  [Hyperbolic attention kernel]
        │  dist_L(q, k) = arcosh(-⟨q,k⟩_L)     ← numerical hazard #1
        │  attention weights = softmax(-β · dist²)
        ▼
  Lorentz context vector  (weighted Fréchet mean or Möbius aggregation)
        │  log_map back to tangent space for output projection
        │  log_q(v) = arcosh(−⟨q,v⟩_L) · (v − ⟨q,v⟩_L · q) / ‖…‖
        ▼
  [LOR→EUC output projection]  (Linear)
        ▼
  Downstream layers  [Euclidean]
```

### Numerical hazards and mitigations

**Hazard 1 — `arcosh` domain violation**
`arcosh(z)` requires `z ≥ 1`. The inner product `−⟨q,k⟩_L` can fall below 1.0 due to fp32 rounding after exp-map steps, especially at initialization.

*Mitigation:* Clamp before `arcosh`:
```python
inner = -lorentz_inner(q, k)          # should be ≥ 1
inner = torch.clamp(inner, min=1 + 1e-7)
dist = torch.acosh(inner)
```
geoopt's `Lorentz.dist` already does this; don't bypass it.

**Hazard 2 — tangent vector norm → 0 in exp-map**
`exp_x(v) = cosh(‖v‖_L)·x + sinh(‖v‖_L)·v/‖v‖_L` has a `0/0` form when `v → 0`.

*Mitigation:* Use the `sinch` approximation:
```python
norm = v_norm.clamp(min=1e-10)
exp_x = cosh(norm)*x + sinch(norm)*v   # sinch(z) = sinh(z)/z, series-stable near 0
```
geoopt implements this internally.

**Hazard 3 — constraint drift during Adam**
Adam's moving-average accumulators (`m`, `v`) live in Euclidean ambient space. After many steps, parameter iterates can drift off the hyperboloid even with per-step retraction if the retraction is applied post-`exp` but not post-gradient-accumulation.

*Mitigation:* In `RiemannianAdam`, apply the retraction *after* the full parameter update, not after the gradient step. geoopt does this correctly. Verify by logging `|⟨x,x⟩_L + 1|` as a training diagnostic metric.

**Hazard 4 — learnable curvature `c` interacts with exp-map scale**
The Lorentz model at curvature `c` uses `exp_x^c(v) = cosh(√c · ‖v‖)·x + sinh(√c · ‖v‖)·v/(√c · ‖v‖)`. When `c → 0` (flat limit), both branches approach 1 and `v` respectively — numerically fine. When `c` is large, `sinh/cosh` saturate → gradient vanishing through `c`.

*Mitigation:* The `softplus + ε` parameterization already prevents `c → 0`. Add a gradient clip specifically on `c_raw` (e.g., `max_norm=1.0`) and monitor `c_eff` per layer during training.

---

## 4 — Recommendation

### Recommended approach: Euclidean Adam + Lorentz retraction via geoopt

Use **`geoopt.optim.RiemannianAdam`** with `ManifoldParameter` wrapping the Lorentz-constrained parameters. This is not pure "Euclidean + retraction" — it is geoopt's Riemannian Adam, which applies Riemannian gradient correction + retraction in a single optimizer step — but it uses the Adam adaptive scaling rather than SGD momentum, which matters for convergence on the small adapter parameter count.

### Minimum viable library stack

```
geoopt >= 0.5.0          # manifold primitives + RiemannianAdam
torch >= 2.2             # for compile() + autocast compatibility
```

No geomstats. No hyptorch. One thin custom module:

```python
# hca_lorentz_utils.py  (~60 lines total)
# - lorentz_inner(u, v): -u[...,0]*v[...,0] + (u[...,1:]*v[...,1:]).sum(-1)
# - euc_to_lorentz(x, c): project Euclidean vector to hyperboloid at origin
# - lorentz_to_euc(x): log-map back to tangent at origin
# - LorentzNorm(d_in, d_hyp, c_param): nn.Module wrapping projection + exp_o
```

This thin layer exists because geoopt's `expmap0` (exp-map from the origin) is the canonical entry point but the curvature-parameterized version with `softplus` needs explicit wiring.

### Justification against upstream constraints

| Constraint | How the stack satisfies it |
|---|---|
| **Lorentz model, not Poincaré** | geoopt `Lorentz` is a first-class manifold; all primitives native |
| **Learnable curvature via softplus+ε** | `c_raw` is an ordinary `nn.Parameter`; gradients flow through `c_eff = c_min + F.softplus(c_raw)` into exp-map; use LR group with `lr_multiplier=0.1` in optimizer |
| **fp32 for manifold ops** | `ManifoldParameter` stays fp32; use `torch.autocast` with `dtype=torch.bfloat16` everywhere else; gate the projection modules with `@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)` |
| **Frozen base LLM** | geoopt optimizers only update parameters passed to them; exclude base model params trivially |
| **Lightweight adapter** | Small parameter count means Adam's per-parameter state overhead is negligible; convergence speed advantage over SGD justifies it |

---

## 5 — Appendix: Verification Sketch

A single-session synthetic experiment to validate the stack before wiring it into the full HCA:

### Setup

- Manifold: `geoopt.manifolds.Lorentz(c=1.0)` (fixed curvature for baseline)
- Parameter: `ManifoldParameter(torch.randn(128, 64), manifold=Lorentz())` after `expmap0` initialization
- Task: minimize `loss = dist_L(x, x_target).mean()` where `x_target` is a fixed set of 128 random Lorentz points
- Optimizer: `RiemannianAdam(lr=1e-3)`

### Metrics to collect per step

1. **Constraint violation magnitude:** `|⟨x,x⟩_L + 1|.max()` — should stay < 1e-5 throughout
2. **Optimization step wall time:** microseconds per step (baseline for production budget)
3. **Loss curve:** convergence to near-zero distance in ~500 steps validates exp-map gradients
4. **Delta vs. corrected-SGD baseline:** run same setup with SGD + manual `lorentz_retract`; compare loss curves at steps 100/500/1000

### Curvature gradient check

Add `c_raw = nn.Parameter(torch.tensor(0.0)); c_eff = 1e-4 + F.softplus(c_raw)`. Verify `c_raw.grad` is non-zero after `loss.backward()`, confirming end-to-end gradient flow through the curvature-parameterized exp-map.

### Pass/fail criteria

- Constraint violation < 1e-5 at all steps: **manifold integrity confirmed**
- Loss reaches < 0.01 by step 500: **optimization signal confirmed**
- `c_raw.grad` ≠ 0: **curvature gradient flow confirmed**
- Step time < 2× pure-Euclidean baseline: **overhead acceptable**

If constraint violation spikes after ~200 steps, the root cause is almost certainly the `arcosh` clamp being too tight; widen to `min=1 + 1e-6` and recheck.
