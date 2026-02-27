# Riemannian Optimization Refinements (Second Opinion)

Critical corrections and sharpened points on the initial analysis.

---

## Refinement 1 — The framing has an internal contradiction

The initial analysis set up a dichotomy (RSGD vs. Euclidean+retract), argued for Euclidean+retract, then recommended `geoopt.RiemannianAdam` — which *is* a Riemannian optimizer. The honest framing is a three-way choice:

| Approach | What actually happens |
|---|---|
| **Full Riemannian Adam** (geoopt) | Riemannian gradient + exp-map step + parallel-transported moments |
| **Euclidean Adam + retraction** | Standard Adam step in ℝ^{n+1}, then `x /= sqrt(\|⟨x,x⟩_L\|)` |
| **Hybrid** (the actual right answer) | Riemannian gradient projection (cheap on Lorentz), Euclidean Adam update on tangent vector, exp-map to land back on manifold, **no parallel transport of moments** |

The hybrid is what you should implement. Here's why:

**geoopt's `RiemannianAdam` does *not* perform true parallel transport of the first/second moment buffers.** It uses a first-order approximation — it simply reuses the moment buffers without transporting them across the manifold. This is documented in the geoopt source and noted in the original paper (Bécigneul & Ganea, 2019). On low-curvature manifolds this is fine; on high-curvature regions it causes the effective learning rate to become anisotropic. For HCA, where curvature is learnable and could drift high early in training, this matters.

**Practical consequence:** geoopt's `RiemannianAdam` ≈ "Riemannian gradient + Euclidean Adam on tangent coordinates + exp-map retraction" — which is the hybrid anyway. You're not getting true Riemannian Adam; you're getting the hybrid with extra overhead from geoopt's manifold dispatch. You could get identical numerics with ~30 lines in a custom optimizer.

**Revised recommendation:** Use geoopt for the *manifold primitives* (`expmap`, `logmap`, `inner`, `dist`, `projx`) but write a thin custom optimizer that does:

```python
# Per step, for each ManifoldParameter p:
egrad = p.grad                                    # Euclidean gradient
rgrad = manifold.egrad2rgrad(p, egrad)            # project to tangent space
# ... standard Adam update on rgrad (in tangent space) ...
new_p = manifold.expmap(p, -lr * adam_direction)  # retract
manifold.assert_check_point_on_manifold(new_p)    # debug mode only
```

This is ~40 lines total and gives you explicit control over the curvature-gradient interaction without depending on geoopt's optimizer internals.

---

## Refinement 2 — geoopt maintenance deserves more scrutiny

The initial analysis claimed "last commit <3 months" — this needs verification. As of early 2026, geoopt's GitHub shows sporadic maintenance. The core manifold code is stable and battle-tested, but:

- Open issues around `torch.compile()` compatibility have been unresolved for >6 months
- The `Lorentz` class was contributed by a third party and has received fewer eyeballs than `PoincareBall`
- No native support for curvature as a differentiable parameter — their `Lorentz(c=...)` takes a fixed float, not a `nn.Parameter`

**This last point is critical for HCA.** You will need to write curvature-parameterized versions of `expmap`, `logmap`, and `dist` yourself regardless. geoopt's implementations hardcode `c` as a constant. The custom surface area is therefore larger than initially estimated:

```
lorentz_expmap(x, v, c)    # ~15 lines (curvature-parameterized)
lorentz_logmap(x, y, c)    # ~15 lines
lorentz_dist(x, y, c)      # ~8 lines
lorentz_inner(x, y)        # ~3 lines (curvature-independent)
egrad_to_rgrad(x, egrad)   # ~5 lines
project_to_hyperboloid(x, c) # ~5 lines (retraction)
```

Total: **~80–100 lines** of curvature-aware manifold ops, not 60. geoopt still provides value as a reference implementation and for its `ManifoldParameter` / `ManifoldTensor` bookkeeping, but the core math must be yours.

---

## Refinement 3 — Missing numerical hazard: backward pass through `arcosh`

The initial analysis flagged the forward-pass `arcosh` clamp but missed the more insidious backward hazard:

```
d/dz arcosh(z) = 1 / sqrt(z² - 1)
```

When `z → 1⁺` (points that are very close on the hyperboloid), the gradient **explodes**. This happens routinely:
- At initialization (all points near the origin)
- Between tokens in the same position cluster
- After curvature shrinks (flattening the manifold brings all points closer)

The forward clamp `z ≥ 1 + ε` does *not* fix this — it just moves the singularity to `1/sqrt(2ε + ε²) ≈ 1/sqrt(2ε)`, which for `ε = 1e-7` is still ~2236.

**Mitigation:** Use a **smoothed distance** near `z = 1`:

```python
def safe_arcosh(z, eps=1e-5):
    # For z close to 1, arcosh(z) ≈ sqrt(2(z-1)), which has a nicer gradient
    z = torch.clamp(z, min=1.0 + eps)
    return torch.acosh(z)
```

Or better, avoid `arcosh` entirely in the attention kernel by using the squared Lorentz distance directly:

```python
# dist²_L = -2c(1 + ⟨x,y⟩_L)  (this is arcosh²(...) but without the arcosh)
sq_dist = -2 * c_eff * (1 + lorentz_inner(q, k))
attention = softmax(-beta * sq_dist)
```

This sidesteps both the forward-clamp and backward-explosion issues. The `arcosh` is a monotonic transform, so `softmax(-β · dist²)` and `softmax(-β' · (-⟨x,y⟩_L - 1))` produce identical attention patterns with a reparameterized `β'`. **Recommend this as the default attention kernel** and reserve actual `arcosh` for diagnostic logging only.

---

## Refinement 4 — Verification sketch is under-specified

The pass/fail criterion "loss < 0.01 by step 500" is meaningless without specifying the target distribution. Sharpened version:

- **Target:** 128 points sampled uniformly on the hyperboloid at Lorentz distance ~5 from the origin (use `expmap0` with tangent vectors of norm 5 ± 0.5)
- **Init:** 128 points at distance ~0.1 from origin (near the tip)
- **Expected:** loss drops from ~25 (≈ 5²) to < 0.1 within 500 steps at `lr=3e-3`
- **Critical check:** add a second run with `c_eff` learnable (init 1.0). Confirm `c_raw.grad` is nonzero at step 1, *and* that `c_eff` at step 500 is not stuck at the initialization value. If it is, the LR multiplier (0.1) may be too conservative for the softplus parameterization — try 0.3.

Also add: compare constraint violation between `geoopt.optim.RiemannianAdam` and the custom hybrid optimizer. They should match to fp32 tolerance, confirming the "geoopt RAdam ≈ hybrid" claim from Refinement 1.

---

## Refinement 5 — Revised library stack

```
geoopt >= 0.5.0       # for ManifoldParameter, reference implementations, projx
torch >= 2.2          # autocast, compile
# NO geoopt optimizer — write custom (~40 lines)
# curvature-parameterized Lorentz ops: custom (~100 lines)
```

Total custom code: **~140 lines** of tested, curvature-aware manifold math + optimizer. This is a one-session write with the geoopt source open as reference. The advantage over pure geoopt is that every line is under your control, curvature is differentiable end-to-end, and you have no hidden dependency on geoopt's optimizer internals which may change or have subtle bugs with `torch.compile()`.
