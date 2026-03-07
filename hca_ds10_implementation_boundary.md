# HCA-DS-10: Implementation Boundary Specification — Phase 5

## 1. Lorentz Attention Kernel with fp32 Stability Guarantees

### 1.1 Overview

The Lorentz attention kernel replaces standard scaled dot-product attention with a hyperbolic distance-based scoring function. The hot path uses the **squared Lorentz inner product** (no arcosh) with a log-cosh radial decay bias. The projection layer (Euclidean → hyperboloid) uses the exponential map at the origin; the readout layer (hyperboloid → Euclidean) uses the logarithmic map at the origin.

### 1.2 Lorentz Inner Product and Scoring

**Lorentz inner product** for points $x, y \in \mathbb{H}_c^n$:

$$\langle x, y \rangle_{\mathcal{L}} = -x_0 y_0 + \sum_{i=1}^{n} x_i y_i$$

**Attention score** (squared form, no arcosh — FA-3 critical finding):

$$\text{score}(q, k) = -\beta \cdot \bigl(-\langle q, k \rangle_{\mathcal{L}} - 1/c\bigr) - \gamma \cdot \ln(x_{0,k})$$

where $\beta > 0$ is the temperature, the first term is the squared Lorentz distance proxy, and the second term is the log-cosh radial decay (DS-6) reduced to a single log on the temporal coordinate (the $\frac{\gamma}{2}\ln c$ constant cancels in softmax).

**Standard attention equivalent:** $\text{score}_{\text{std}}(q, k) = q^T k / \sqrt{d}$, which is $2d$ FLOPs per pair. The Lorentz scoring is $\langle q, k \rangle_{\mathcal{L}} = -q_0 k_0 + q_{\text{sp}}^T k_{\text{sp}}$, costing $2d + 4$ FLOPs (one extra multiply for $-q_0 k_0$, one subtraction, one $-1/c$, one $\times(-\beta)$, plus one log for decay).

### 1.3 Projection: Euclidean → Hyperboloid

Given Euclidean embedding $e \in \mathbb{R}^d$, project via $\exp_o^c$:

```
FUNCTION project_to_hyperboloid(e: R^d, c: float) -> H_c^d:
    # Linear transform in tangent space at origin
    v = W_proj @ e + b_proj          # v in T_o H_c = R^d

    # Clamp tangent vector norm to prevent cosh/sinh overflow
    v_norm = max(||v||_2, eps_sinh)   # eps_sinh = 1e-6
    v_norm_clamped = min(v_norm, SINH_CLAMP / sqrt(c))  # SINH_CLAMP = 85.0
    v = v * (v_norm_clamped / v_norm) # rescale if clamped

    # Exponential map at origin (Lorentz model)
    sc = sqrt(c) * v_norm_clamped
    x_0 = cosh(sc) / sqrt(c)         # temporal coordinate
    x_sp = sinh(sc) / (sqrt(c) * v_norm_clamped) * v  # spatial coordinates

    return (x_0, x_sp)
```

### 1.4 Readout: Hyperboloid → Euclidean

```
FUNCTION project_to_euclidean(x: H_c^d, c: float) -> R^d:
    # Logarithmic map at origin
    x_0 = x[0]
    x_sp = x[1:]

    # Clamp argument to arcosh to prevent gradient explosion
    arg = max(sqrt(c) * x_0, 1.0 + eps_arcosh)  # eps_arcosh = 1e-7
    d_origin = arcosh(arg)                        # scaled distance from origin

    # Avoid 0/0 when x ≈ origin
    x_sp_norm = max(||x_sp||_2, eps_norm)         # eps_norm = 1e-7

    # Tangent vector at origin
    scale = d_origin / (sqrt(c) * x_sp_norm)
    v = scale * x_sp                               # v in T_o H_c = R^d

    # Linear readout
    return W_read @ v + b_read
```

### 1.5 fp32 Hazard Catalog

| ID | Hazard | Trigger Condition | Mitigation | Error Bound |
|---|---|---|---|---|
| H1 | cosh/sinh overflow in exp map | $\sqrt{c} \cdot \|v\| > 88.7$ (fp32 exp limit) | Clamp $\|v\|$ to $85/\sqrt{c}$ before exp map. This limits $R_{\max,\text{safe}} = \operatorname{arcosh}(\cosh(85)) \approx 85$, far above $R_{\max} = 16.12$ at $c=1$. | $|r_{\text{clamped}} - r_{\text{true}}| = 0$ at realistic operating points; max representable $r = 85$ |
| H2 | Lorentz inner product significance loss near origin | $x_0 \approx y_0 \approx 1/\sqrt{c}$, $x_{\text{sp}} \approx y_{\text{sp}} \approx 0$: cancellation in $-x_0 y_0 + x_{\text{sp}}^T y_{\text{sp}}$ | Compute in fp32 (not fp16). At $c=1$, origin has $x_0 = 1$; inner product $= -1 + 0 = -1 = -1/c$ exactly. Near origin: $|\langle x, y \rangle_{\mathcal{L}} + 1/c| \sim O(r^2)$, well above fp32 epsilon for $r > 10^{-3}$. | $\varepsilon_{\text{rel}} \leq 2^{-23} / r^2 \approx 10^{-7} / r^2$. For $r > 0.01$: $\varepsilon_{\text{rel}} < 10^{-3}$ |
| H3 | Softmax overflow from large distances | $-\langle q, k \rangle_{\mathcal{L}} - 1/c$ can reach $\sim 10^7$ (at $R_{\max}$), so $\beta \cdot \text{dist}^2$ overflows fp32 | Use standard softmax stabilization: subtract $\max_k \text{score}(q, k)$ before exp. This is already standard in PyTorch's `F.softmax`. No custom mitigation needed. | Exact (numerically stable softmax is standard) |
| H4 | arcosh gradient explosion in log map | $\frac{\partial}{\partial z}\operatorname{arcosh}(z) = 1/\sqrt{z^2 - 1} \to \infty$ as $z \to 1^+$ (x near origin) | **Bypass arcosh in attention hot path** (use squared inner product instead). For readout log map: clamp $z \geq 1 + 10^{-7}$. Custom backward: $\text{grad} = \min(1/\sqrt{z^2 - 1},\; 10^5)$ | Clamped gradient error $\leq \varepsilon_{\text{arcosh}} = 10^{-7}$ in distance, corresponding to $r_{\min} \approx 4.5 \times 10^{-4}$ |
| H5 | Division by zero in exp map ($v = 0$) | Zero tangent vector → $\text{sinh}(\|v\|)/\|v\| = 0/0$ | Check $\|v\| < \varepsilon_{\text{sinh}} = 10^{-6}$; if so, return origin $o = (1/\sqrt{c}, \mathbf{0})$. Custom backward: L'Hôpital gives $\lim_{v \to 0} \text{sinh}(\|v\|)/\|v\| = 1$, so use $\text{grad} = 1$ in this regime. | Exact (limit is well-defined) |
| H6 | fp16 arcosh overflow | fp16 max $= 65504$; $\operatorname{arcosh}(65504) \approx 11.8$, which is below $R_{\max} = 16.12$ | **Never use fp16 for distance computations.** All Lorentz kernel ops in fp32. Attention score computation: fp32. Softmax: may use fp16 after stabilization. | Not applicable (fp32 enforced) |
| H7 | Temporal coordinate drift | $x_0$ must satisfy $x_0 = \sqrt{1/c + \|x_{\text{sp}}\|^2}$ (hyperboloid constraint); optimizer updates may violate this | After each optimizer step on spatial coordinates, recompute $x_0 = \sqrt{1/c + \|x_{\text{sp}}\|^2}$. Alternatively, parameterize only $x_{\text{sp}}$ and derive $x_0$ in the forward pass. **Recommended: derive $x_0$, never store it.** | Exact (constraint satisfied by construction) |

### 1.6 Complete Kernel Pseudocode

```
FUNCTION hca_attention(Q, K, V, c, gamma, beta, mask):
    # Q, K, V: [B, H, N, d+1] — Lorentz embeddings (x_0 derived, not stored)
    # c: [L] per-layer curvature
    # gamma: [L] per-layer decay strength
    # beta: [L] temperature (or use 1/sqrt(d) default)

    # === Step 1: Compute Lorentz inner product [B, H, N, N] ===
    # Q, K have spatial dims [1..d] and temporal dim [0]
    Q_0 = sqrt(1/c + sum(Q_sp^2, dim=-1))   # [B,H,N] — derive x_0
    K_0 = sqrt(1/c + sum(K_sp^2, dim=-1))   # [B,H,N]

    # Inner product: -q_0*k_0 + q_sp . k_sp
    spatial_dot = einsum('bhid, bhjd -> bhij', Q_sp, K_sp)  # [B,H,N,N]
    temporal_dot = -Q_0.unsqueeze(-1) * K_0.unsqueeze(-2)   # [B,H,N,N]
    lorentz_ip = temporal_dot + spatial_dot                   # [B,H,N,N]

    # === Step 2: Squared distance proxy ===
    dist_sq = -lorentz_ip - 1/c   # ≥ 0 for points on H_c
    dist_sq = max(dist_sq, 0.0)   # clamp numerical noise

    # === Step 3: Radial decay bias (DS-6 log-cosh) ===
    # λ(r_k) = γ · ln(x_{0,k}), constant term γ/2·ln(c) cancels in softmax
    decay_bias = gamma * log(max(K_0, eps_norm))  # [B,H,N]

    # === Step 4: Attention score ===
    score = -beta * dist_sq - decay_bias.unsqueeze(-2)  # [B,H,N,N]

    # === Step 5: Mask and softmax ===
    IF mask is not None:
        score = score + mask   # -inf for masked positions

    alpha = softmax(score, dim=-1)  # standard stabilized softmax

    # === Step 6: Record α_s for QoS monitor (system token = index 0) ===
    alpha_s = alpha[:, :, :, 0].mean()  # mean over batch/head/query
    STORE alpha_s for QoS tracker (no grad)

    # === Step 7: Attention output ===
    out = einsum('bhij, bhjd -> bhid', alpha, V_sp)  # [B,H,N,d]

    return out, alpha_s
```

### 1.7 Summary Table

| Parameter | Value | Source |
|---|---|---|
| Scoring function | $-\beta(-\langle q, k \rangle_{\mathcal{L}} - 1/c) - \gamma \ln(x_{0,k})$ | DS-6, FA-3 |
| Temperature $\beta$ | $1/\sqrt{d}$ default, learnable | Standard |
| Decay $\gamma$ | $\text{softplus}(\gamma_{\text{raw}})$, init $\gamma = 0.6$ | DS-6 |
| Precision | fp32 for kernel, fp16 OK for softmax output | FA-1, H6 |
| arcosh in hot path | **No** — squared inner product only | FA-3 |
| Exp map clamp | $\|v\| \leq 85/\sqrt{c}$ | H1 |
| Log map clamp | $z \geq 1 + 10^{-7}$, grad clamp $10^5$ | H4 |
| $x_0$ storage | **Derived**, not stored | H7 |

---

## 2. RadialPositionTracker Class Specification

### 2.1 Overview

The RadialPositionTracker maintains per-token radial positions on the Lorentz hyperboloid and applies Hebbian pull updates at the $k = 100$ geometry update cadence. Tokens that receive high attention are pulled toward the origin (smaller radius), reinforcing their privilege. The tracker operates on spatial coordinates only; temporal coordinates are derived.

### 2.2 State Representation

```
CLASS RadialPositionTracker:
    # === State ===
    r_init: float = 8.0                    # DS-6 initialization radius
    r_reg_weight: float = 0.01             # DS-6 radial regularization μ
    pull_lr: float = 1e-3                  # Hebbian pull learning rate
    pull_threshold: float = 0.1            # attention weight threshold for pull
    pull_damping: float = 0.95             # exponential damping per update

    # === Storage ===
    # Spatial coordinates are part of the model parameters (optimized by Riemannian SGD)
    # The tracker maintains AUXILIARY state for the Hebbian pull signal:

    pull_accumulator: Tensor[L, H, N_max]  # accumulated pull signal, fp32
    pull_count: Tensor[L, H, N_max]        # number of contributions, int32

    # Memory layout: contiguous per-layer for cache-friendly access
    # Total auxiliary memory: L * H * N_max * 8 bytes (4 float + 4 int)
```

### 2.3 Initialization Strategy

```
FUNCTION init_positions(N: int, d: int, c: float) -> Tensor[N, d]:
    # DS-6 placement: push to r = r_init = 8.0
    # In Lorentz model: r = arcosh(sqrt(c) * x_0)
    # So x_0 = cosh(r_init) / sqrt(c) = cosh(8.0) / sqrt(c) ≈ 1490.5 at c=1

    # Initialize spatial coordinates on a unit sphere scaled to target radius
    x_sp = randn(N, d)                     # random direction
    x_sp = x_sp / ||x_sp||_2               # unit sphere

    # Scale to achieve target x_0:
    # x_0 = sqrt(1/c + ||x_sp||^2), so ||x_sp|| = sqrt(x_0^2 - 1/c)
    target_x0 = cosh(r_init) / sqrt(c)
    target_sp_norm = sqrt(target_x0^2 - 1/c)
    x_sp = x_sp * target_sp_norm

    return x_sp  # x_0 is derived in forward pass
```

### 2.4 Hebbian Pull Update Protocol

The pull is applied every $k = 100$ steps, after the optimizer step.

```
FUNCTION accumulate_pull(alpha: Tensor[B,H,N,N], layer: int):
    # Called during forward pass (no grad, detached)
    # alpha[b,h,i,j] = attention weight from query i to key j

    # System token pull signal: how much attention each key receives from all queries
    # pull_signal[j] = mean_over_queries(alpha[:,:,:,j])
    pull_signal = alpha.mean(dim=(0, 2))  # [H, N] — mean over batch and query

    # Only accumulate for keys receiving above-threshold attention
    mask = (pull_signal > pull_threshold)
    self.pull_accumulator[layer] += pull_signal * mask
    self.pull_count[layer] += mask.int()

FUNCTION apply_pull(x_sp: Tensor[N, d], layer: int, head: int, c: float):
    # Called every k=100 steps, per layer per head

    count = max(self.pull_count[layer, head], 1)
    avg_pull = self.pull_accumulator[layer, head] / count  # [N]

    # Pull strength: tokens receiving more attention get pulled harder
    # Direction: toward origin (decrease ||x_sp||)
    x_sp_norm = ||x_sp||_2  # [N]

    # Geodesic pull toward origin = scale down spatial coordinates
    # New radius: r_new = r_old - pull_lr * avg_pull * damping^(step/k)
    scale = 1.0 - pull_lr * avg_pull * pull_damping  # [N]
    scale = clamp(scale, 0.1, 1.0)  # prevent collapse past r=0.8 (≈10% of init)

    x_sp = x_sp * scale.unsqueeze(-1)  # scale spatial coordinates

    # Reset accumulators
    self.pull_accumulator[layer, head] = 0
    self.pull_count[layer, head] = 0

    return x_sp
```

### 2.5 Radial Regularization (DS-6)

Applied as a loss term every step (not just every $k$ steps):

$$\mathcal{L}_{\text{radial}} = \mu \cdot \frac{1}{N} \sum_{i=1}^{N} (x_{0,i} - x_{0,\text{target}})^2$$

where $x_{0,\text{target}} = \cosh(r_{\text{init}}) / \sqrt{c}$ and $\mu = 0.01$. This uses $x_0$ as a proxy (avoids arcosh in training loop), per DS-6.

### 2.6 Memory Layout

| Component | Shape | Dtype | Bytes |
|---|---|---|---|
| `x_sp` (model params) | $[L \times H \times N_{\max} \times d_{\text{head}}]$ | fp32 | $4 \cdot L \cdot H \cdot N \cdot d_h$ |
| `pull_accumulator` | $[L \times H \times N_{\max}]$ | fp32 | $4 \cdot L \cdot H \cdot N$ |
| `pull_count` | $[L \times H \times N_{\max}]$ | int32 | $4 \cdot L \cdot H \cdot N$ |

**Example:** $L = 12$, $H = 8$, $N = 8192$, $d_h = 64$:
- Model params: $12 \times 8 \times 8192 \times 64 \times 4 = 192$ MiB (same as standard attention KV)
- Pull accumulator: $12 \times 8 \times 8192 \times 4 = 3$ MiB
- Pull count: $12 \times 8 \times 8192 \times 4 = 3$ MiB
- **Tracker overhead: 6 MiB** (3.1% of model KV memory)

### 2.7 Thread Safety

Each attention head has independent pull accumulators. No cross-head synchronization needed during accumulation. The `apply_pull` function operates per-head, so parallel execution across heads is safe. The only synchronization point is the $k$-step barrier (all heads must complete before the next forward pass uses updated positions).

### 2.8 Summary Table

| Parameter | Value | Source |
|---|---|---|
| Init radius $r_{\text{init}}$ | 8.0 | DS-6 |
| Pull learning rate | $10^{-3}$ | Tunable; conservative default |
| Pull threshold | 0.1 (10% attention weight) | Tunable |
| Damping | 0.95 per geometry update | Prevents runaway pull |
| Clamp | $\text{scale} \in [0.1, 1.0]$ | Prevents radial collapse |
| Regularization $\mu$ | 0.01 | DS-6 |
| Update cadence | Every $k = 100$ steps | DS-5, DS-7 |
| Memory overhead | $8 \cdot L \cdot H \cdot N$ bytes | 3.1% of KV at reference config |

---

## 3. QoS Monitor Training Loop Integration

### 3.1 Overview

The GeometryQoSHub (DS-7) runs at $k = 100$ step cadence, collecting attention health metrics, evaluating the 4-state hysteresis machine, and applying corrective actions to $\gamma$ and $c$ via gradient injection. Integration point: **after optimizer step, before next forward pass**.

### 3.2 Training Loop Integration

```
FOR step = 1 TO max_steps:
    # === Forward pass ===
    output, alpha_s_per_layer = model.forward(batch)       # (1)

    # === Loss computation ===
    loss = task_loss(output, targets)
    loss += radial_regularization(model)                    # (2) DS-6 μ=0.01

    # === Backward pass ===
    loss.backward()                                         # (3)

    # === Optimizer step ===
    optimizer.step()                                        # (4) Riemannian + Adam
    optimizer.zero_grad()

    # === Hyperboloid constraint enforcement ===
    FOR each layer l:
        recompute_x0(model.layer[l])                       # (5) H7 mitigation

    # === QoS Monitor (every k steps) ===
    IF step % k == 0:                                       # (6)
        qos_hub.update(step, alpha_s_per_layer, model)

    # === Hebbian pull (every k steps) ===
    IF step % k == 0:                                       # (7)
        FOR each layer l, head h:
            model.layer[l].head[h].x_sp = tracker.apply_pull(
                model.layer[l].head[h].x_sp, l, h, c[l])
```

### 3.3 GeometryQoSHub Update Protocol

```
FUNCTION qos_hub.update(step, alpha_s_per_layer, model):
    FOR each layer l:
        # --- Collect metrics ---
        alpha_s = alpha_s_per_layer[l]
        alpha_s_ema[l] = rho * alpha_s_ema[l] + (1 - rho) * alpha_s     # ρ=0.99

        x0_values = get_temporal_coords(model.layer[l])                   # [H, N]
        r_median = median(arcosh(sqrt(c[l]) * x0_values))
        r_iqr = percentile(r, 75) - percentile(r, 25)

        c_eff = model.layer[l].c_eff
        gamma_val = softplus(model.layer[l].gamma_raw)

        grad_c = accumulated_grad_norm(model.layer[l].c_raw)
        grad_gamma = accumulated_grad_norm(model.layer[l].gamma_raw)

        # --- State machine transition (DS-7 Section 2.3) ---
        e = alpha_safe - alpha_s_ema[l]                    # privilege error

        prev_state = state[l]
        IF e < -eps_dead:                                  # α_s too HIGH
            state[l] = HEALTHY
        ELIF |e| <= eps_dead:                              # in dead zone
            state[l] = HEALTHY
        ELIF e > eps_dead AND e <= 0.15:
            state[l] = DEGRADED
        ELIF e > 0.15 AND e <= 0.3:
            state[l] = ALERT
        ELIF e > 0.3 OR (state[l] == CRITICAL for 5+ updates):
            state[l] = CRITICAL

        # Hysteresis: require eps_dead improvement to step DOWN
        IF prev_state > state[l]:
            IF e > prev_threshold - 0.05:                  # 0.05 hysteresis
                state[l] = prev_state                      # stay in worse state

        # --- Corrective actions ---
        IF state[l] == HEALTHY:
            # Relaxation: gently reduce γ toward 0 if over-decaying
            IF alpha_s_ema[l] > alpha_safe + eps_dead:
                delta_g_gamma = K_r * (alpha_s_ema[l] - alpha_safe - eps_dead)
                inject_gradient(model.layer[l].gamma_raw, +delta_g_gamma)  # decrease γ

        ELIF state[l] in {DEGRADED, ALERT}:
            # Proportional correction: increase γ
            delta_g_gamma = -K_p * (e - eps_dead)          # K_p = 0.5
            delta_g_gamma = clamp(delta_g_gamma, -dg_max, dg_max)  # dg_max = 1.0
            inject_gradient(model.layer[l].gamma_raw, delta_g_gamma)

        ELIF state[l] == CRITICAL:
            # Emergency: 3× multiplier + curvature check
            delta_g_gamma = -3 * K_p * (e - eps_dead)
            inject_gradient(model.layer[l].gamma_raw, delta_g_gamma)

            # Curvature compression check (DS-7 Section 5.2)
            IF r_iqr / r_median < 0.5:                     # radial compression
                inject_gradient(model.layer[l].c_raw, -0.1) # increase c

        # --- Logging ---
        log_metrics(step, l, {
            'alpha_s': alpha_s,
            'alpha_s_ema': alpha_s_ema[l],
            'state': state[l],
            'gamma': gamma_val,
            'c_eff': c_eff,
            'r_median': r_median,
            'r_iqr': r_iqr,
            'grad_c': grad_c,
            'grad_gamma': grad_gamma,
            'delta_g_gamma': delta_g_gamma,
        })
```

### 3.4 QoS Parameters

| Parameter | Value | Source |
|---|---|---|
| Update cadence $k$ | 100 steps | DS-5 |
| EMA decay $\rho$ | 0.99 | DS-7 |
| $\alpha_{\text{safe}}$ | 0.5 | DS-8 fixed point |
| Dead zone $\varepsilon_{\text{dead}}$ | 0.05 | DS-7 |
| Proportional gain $K_p$ | 0.5 | DS-7 |
| Relaxation gain $K_r$ | 0.1 | DS-7 |
| Max injection $\Delta g_{\max}$ | 1.0 | DS-7 |
| Emergency multiplier | 3× | DS-7 |
| Hysteresis band | 0.05 | DS-7 |
| Compression threshold | IQR/median < 0.5 | DS-7 |

### 3.5 Gradient Injection Mechanism

The QoS controller does NOT modify parameters directly. It injects a synthetic gradient into the `.grad` field of $\gamma_{\text{raw}}$ (or $c_{\text{raw}}$) **after** the optimizer step but **before** the next forward pass. This means:

1. Optimizer step uses task gradients only (step 4)
2. QoS evaluates state (step 6)
3. QoS modifies `.grad` on $\gamma_{\text{raw}}$ (synthetic injection)
4. A second mini-step applies the injection: $\gamma_{\text{raw}} \leftarrow \gamma_{\text{raw}} - \text{lr}_\gamma \cdot \Delta g_\gamma$

This two-phase approach keeps task gradient and QoS gradient cleanly separated.

### 3.6 QoS State Memory

| Component | Size | Dtype |
|---|---|---|
| `alpha_s_ema` | $L$ floats | fp32 |
| `state` | $L$ ints | int8 |
| `state_duration` | $L$ ints | int32 |
| `prev_threshold` | $L$ floats | fp32 |
| `grad_accum_c` | $L$ floats | fp32 |
| `grad_accum_gamma` | $L$ floats | fp32 |
| **Total** | $\sim 20 \cdot L$ bytes | |

For $L = 12$: **240 bytes**. Negligible.

---

## 4. Computational Cost Analysis

### 4.1 FLOP Analysis

**Standard scaled dot-product attention** for $N$ tokens, dimension $d$, $H$ heads ($d_h = d/H$):

| Operation | FLOPs |
|---|---|
| $QK^T$: $N \times N \times d_h$ multiply-add | $2 N^2 d_h$ |
| Softmax: $\sim 5N$ per row × $N$ rows | $5 N^2$ |
| $\alpha V$: $N \times N \times d_h$ | $2 N^2 d_h$ |
| **Total per head** | $4 N^2 d_h + 5 N^2$ |

**HCA Lorentz attention** per head:

| Operation | FLOPs | vs Standard |
|---|---|---|
| Derive $x_0$: $\sqrt{1/c + \|x_{\text{sp}}\|^2}$ per token | $2 N d_h + N$ | Extra: $2Nd_h + N$ |
| Lorentz IP: $-q_0 k_0 + q_{\text{sp}}^T k_{\text{sp}}$ | $2 N^2 d_h + 2 N^2$ | Extra: $2N^2$ |
| Distance proxy: $-\text{IP} - 1/c$ | $N^2$ | Extra: $N^2$ |
| Decay bias: $\gamma \cdot \ln(k_0)$ | $2N$ | Extra: $2N$ |
| Score assembly: $-\beta \cdot \text{dist} - \text{bias}$ | $2 N^2$ | Extra: $2N^2$ |
| Softmax | $5 N^2$ | Same |
| $\alpha V$ | $2 N^2 d_h$ | Same |
| **Total per head** | $4 N^2 d_h + 10 N^2 + 2Nd_h + 3N$ | |

**FLOP ratio:**

$$k = \frac{4N^2 d_h + 10N^2 + 2Nd_h + 3N}{4N^2 d_h + 5N^2}$$

For $N \gg d_h$ (attention-dominated regime):

$$k \approx \frac{4d_h + 10}{4d_h + 5}$$

At $d_h = 64$: $k = (256 + 10)/(256 + 5) = 266/261 = \mathbf{1.019}$

At $d_h = 128$: $k = (512 + 10)/(512 + 5) = 522/517 = \mathbf{1.010}$

**The Lorentz kernel adds 1–2% FLOPs to the attention computation.**

### 4.2 Amortized Overhead (QoS + Hebbian Pull)

Operations running every $k = 100$ steps:

| Operation | FLOPs per invocation | Amortized per step |
|---|---|---|
| EMA update: 1 multiply + 1 add per layer | $2L$ | $0.02L$ |
| Radial statistics: sort $N$ values, compute percentiles | $\sim N \log N$ per layer | $\sim L \cdot N \log N / k$ |
| State machine: $\sim 20$ comparisons per layer | $20L$ | $0.2L$ |
| Gradient injection | $2L$ | $0.02L$ |
| Hebbian pull: $N$ multiplies per head per layer | $L \cdot H \cdot N$ | $L \cdot H \cdot N / k$ |
| **Total amortized** | | $\sim L(H \cdot N + N \log N) / k$ |

For $L = 12$, $H = 8$, $N = 8192$, $k = 100$:
- Amortized: $12 \times (8 \times 8192 + 8192 \times 13) / 100 \approx 22{,}000$ FLOPs/step
- Attention FLOPs/step: $12 \times 8 \times (4 \times 8192^2 \times 64) \approx 1.65 \times 10^{12}$
- **Amortized overhead: $1.3 \times 10^{-8}$ — effectively zero**

### 4.3 Memory Overhead Summary

| Component | Formula | Example ($L$=12, $H$=8, $N$=8K, $d_h$=64) |
|---|---|---|
| Standard attention KV cache | $2 \cdot L \cdot H \cdot N \cdot d_h \cdot 4$ bytes | 384 MiB |
| HCA: same KV (spatial coords) | same | 384 MiB |
| Radial tracker auxiliary | $8 \cdot L \cdot H \cdot N$ bytes | 6 MiB |
| QoS state | $\sim 20 \cdot L$ bytes | 240 B |
| Per-layer $c_{\text{raw}}, \gamma_{\text{raw}}$ | $2 \cdot L \cdot 4$ bytes | 96 B |
| Curvature floor EMA ($N_{\text{eff}}$) | $4$ bytes | 4 B |
| **Total HCA overhead** | | **6 MiB (1.6%)** |

$$\text{HCA\_memory} = \text{Standard\_memory} + O(L \cdot H \cdot N)$$

The dominant term is the radial tracker ($8 \cdot L \cdot H \cdot N$ bytes). This is $1/(d_h/2)$ of the KV cache, so for $d_h = 64$: 3.1% overhead.

### 4.4 Benchmark Plan

| Task | Purpose | Success Criterion |
|---|---|---|
| WikiText-103 LM perplexity | Baseline competence — HCA must not degrade standard LM | Parity (±1 PPL) vs matched-param standard attention |
| Adversarial needle-in-haystack | Core HCA falsifiability — can origin privilege maintain system prompt fidelity at $N = 128$K? | $\alpha_s \geq 0.3$ at $N = 128$K; needle retrieval accuracy $\geq 95$% |
| Long-range retrieval (PG-19, SCROLLS) | Long-context generalization | Improvement over standard attention at $N \geq 16$K |
| GLUE/SuperGLUE | Downstream classification sanity | Parity (±0.5%) vs matched-param baseline |

**Comparison methodology:**
- **Matched parameters:** HCA model has same $d$, $L$, $H$ as baseline. Extra parameters: $2L$ scalars ($c_{\text{raw}}, \gamma_{\text{raw}}$) — negligible.
- **Matched FLOPs:** 1.02× overhead is within noise. No adjustment needed.
- **Matched wall-clock:** If fused CUDA kernel is used, expect ≤1.5× wall-clock overhead (FA-2 estimate). Without fusion: 2–3× due to fp32 enforcement on attention scores.
- **Ablation plan:** HCA full → HCA without decay ($\gamma = 0$) → HCA without QoS → Standard attention.

### 4.5 Cost Summary Table

| Metric | Value |
|---|---|
| FLOP ratio (attention only) | $\mathbf{1.02\times}$ at $d_h = 64$ |
| FLOP ratio (amortized QoS+pull) | $+1.3 \times 10^{-8}$ (negligible) |
| Memory overhead | $+O(L \cdot H \cdot N)$, $\sim 1.6$% at reference config |
| QoS state memory | 240 bytes (negligible) |
| Wall-clock estimate (fused) | $\leq 1.5\times$ (FA-2) |
| Wall-clock estimate (unfused) | $2$–$3\times$ (fp32 enforcement) |
| Extra parameters | $2L$ scalars ($c_{\text{raw}}, \gamma_{\text{raw}}$) |

---

## 5. Open Questions and DS-11 Handoff

### 5.1 Open Questions

**OQ-1: FlashAttention fusion.** The Lorentz inner product ($-q_0 k_0 + q_{\text{sp}}^T k_{\text{sp}}$) is structurally similar to standard dot product with one sign flip on the first dimension. Can this be implemented as a modified FlashAttention kernel with minimal changes to the tiling logic? The log-cosh decay bias is a per-key additive term, similar to ALiBi positional encoding, which FlashAttention already supports.

**OQ-2: Mixed-precision strategy.** The spec mandates fp32 for Lorentz kernel computations, but modern training uses bf16/fp16 extensively. Can the scoring computation use bf16 with the distance proxy accumulated in fp32? This would require testing whether bf16's 8-bit mantissa is sufficient for the Lorentz inner product.

**OQ-3: Hebbian pull vs. gradient-based radial update.** The Hebbian pull (Section 2.4) is a heuristic that operates outside the gradient graph. An alternative is to make radial position a differentiable function of attention weights, so task gradients naturally pull high-attention tokens toward the origin. This would unify the radial update with standard backprop but may be unstable.

### 5.2 DS-11 Handoff

DS-11 should address the **prototype implementation**:

1. **Custom `autograd.Function` for Lorentz ops:** `LorentzExpMap`, `LorentzLogMap`, `LorentzSquaredDistance` with analytical backward passes and all H1–H7 mitigations.
2. **`HCAAttention` module:** Drop-in replacement for `nn.MultiheadAttention` with Lorentz scoring + decay bias.
3. **Training harness:** Riemannian optimizer integration, QoS hub instantiation, Hebbian tracker lifecycle.
4. **Unit tests:** Numerical stability tests at boundary conditions (H1–H7 trigger conditions), gradient checking, determinism verification.
