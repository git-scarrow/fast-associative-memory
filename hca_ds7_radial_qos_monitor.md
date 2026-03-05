# HCA-DS-7: Radial QoS Monitor & Hebbian Pull Design Specification

## 1. Executive Summary

This document specifies the Quality-of-Service (QoS) monitoring and control system for HCA's radial privilege mechanism. The system addresses a critical operational gap: DS-5 and DS-6 provide the geometric machinery (adaptive curvature + log-cosh decay) to *establish* origin privilege, but no mechanism to *monitor* whether privilege is actually maintained during training or to *correct* drift when it isn't.

The design has four components:

1. **$\alpha_s$ Tracker** — Per-layer EMA of the system token's attention weight, computed from attention output with zero hot-path overhead.
2. **Radial Distribution Monitor** — Lightweight per-layer statistics on token placement (median, IQR of temporal coordinates), detecting separation erosion.
3. **$\gamma$ Controller** — A proportional controller that adjusts the DS-6 decay strength parameter $\gamma$ via gradient injection, increasing decay when $\alpha_s$ drops and relaxing it when privilege is safely maintained.
4. **Unified QoS Hook API** — Extension of DS-5's `register_qos_hook` to accept both curvature and decay parameter adjustments through a single callback interface.

The entire system adds **zero ops to the attention hot path**. All monitoring runs on attention weights already computed by the forward pass. All control actions are deferred to the k=100 geometry update cadence from DS-5/DS-6.

**FAM parallel:** This is the HCA analogue of FAM's adaptive eviction policy (MT-15). FAM uses `class_loss = 1 - (n_classes_present / n_classes_ever_seen)` to blend between LRU and coverage eviction. HCA uses $\alpha_s$ drop below threshold to blend between no-decay (pure distance attention) and full-decay (strong privilege). Both are reactive controllers that only intervene when a monitored invariant is threatened.

---

## 2. Component 1: $\alpha_s$ Tracker

### 2.1 Definition

For layer $l$ at training step $t$, let $\alpha_s^{(l)}(t)$ be the attention weight assigned to the system token by the system token's own query (self-attention weight at position 0):

$$\alpha_s^{(l)}(t) = \text{attn\_weights}^{(l)}[0, 0] \in [0, 1]$$

where `attn_weights` is the $[N_q, N_k]$ softmax output. We track the **head-averaged** value:

$$\bar{\alpha}_s^{(l)}(t) = \frac{1}{H} \sum_{h=1}^{H} \alpha_{s,h}^{(l)}(t)$$

### 2.2 EMA Specification

$$\hat{\alpha}_s^{(l)}(t) = \rho \cdot \hat{\alpha}_s^{(l)}(t-1) + (1 - \rho) \cdot \bar{\alpha}_s^{(l)}(t)$$

| Parameter | Symbol | Value | Justification |
|-----------|--------|-------|---------------|
| EMA decay | $\rho$ | 0.99 | Time constant $\tau = 100$ steps, matching k=100 geometry update cadence |
| Initial value | $\hat{\alpha}_s^{(l)}(0)$ | 0.5 | Neutral prior; avoids false alarms at startup |
| Alert threshold | $\alpha_{\text{low}}$ | 0.3 | From DS-6 Section 5.2: minimum $\alpha_s$ for meaningful privilege |
| Safety threshold | $\alpha_{\text{safe}}$ | 0.5 | From DS-5 D4 Corollary 2: the N-independent bound under $\beta = 1$ |
| Critical threshold | $\alpha_{\text{crit}}$ | 0.1 | Below this, privilege is effectively lost; trigger emergency response |

### 2.3 Privilege State Machine

Each layer $l$ is in one of four states, determined by $\hat{\alpha}_s^{(l)}$:

```
                 ┌──────────┐
                 │  HEALTHY  │  α̂_s ≥ α_safe (0.5)
                 └─────┬────┘
                       │ α̂_s drops below α_safe
                       ▼
                 ┌──────────┐
                 │ DEGRADED  │  α_low ≤ α̂_s < α_safe
                 └─────┬────┘
                       │ α̂_s drops below α_low
                       ▼
                 ┌──────────┐
                 │  ALERT    │  α_crit ≤ α̂_s < α_low
                 └─────┬────┘
                       │ α̂_s drops below α_crit
                       ▼
                 ┌──────────┐
                 │ CRITICAL  │  α̂_s < α_crit (0.1)
                 └──────────┘
```

**Hysteresis:** State transitions upward require $\hat{\alpha}_s$ to exceed the boundary by a margin of 0.05 (e.g., ALERT → DEGRADED requires $\hat{\alpha}_s \geq \alpha_{\text{low}} + 0.05 = 0.35$). This prevents oscillation at boundaries.

### 2.4 Computation Path

The tracker reads attention weights that are **already computed** by the forward pass. No additional matrix operations are needed.

```python
# Inside the attention layer's forward pass (post-softmax):
# attn_weights: [B, H, N_q, N_k] — already computed
# system_mask: [B, N] — True for system token positions

def update_alpha_tracker(self, attn_weights, system_mask, layer_idx):
    """Update α_s EMA. Called after attention computation, before value aggregation.

    Zero additional compute: reads one element from an existing tensor.
    """
    with torch.no_grad():
        # System token's self-attention weight, averaged over heads and batch
        # system_mask[b] is True at position 0 (or wherever the system token is)
        sys_idx = system_mask.nonzero(as_tuple=True)  # (batch_indices, token_indices)
        # α_s = attn_weights[b, h, sys_pos, sys_pos] for each batch/head
        alpha_s = attn_weights[sys_idx[0], :, sys_idx[1], sys_idx[1]]  # [B_sys, H]
        alpha_s_mean = alpha_s.mean()  # scalar

        # EMA update
        self.alpha_s_ema[layer_idx].mul_(self.rho).add_(alpha_s_mean, alpha=1 - self.rho)
```

**Cost:** One tensor index + one scalar mean + one FMA. Negligible.

---

## 3. Component 2: Radial Distribution Monitor

### 3.1 Purpose

The $\alpha_s$ tracker monitors the *effect* (attention weight) but not the *cause* (token placement). Two failure modes produce low $\alpha_s$ through different mechanisms:

- **Separation erosion:** Context tokens drift toward the origin, reducing $\delta$. Fix: increase radial regularization or $\gamma$.
- **Curvature collapse:** $c$ drops, shrinking $R_{\max}$ and compressing all tokens toward similar radii. Fix: increase curvature floor.

The radial distribution monitor distinguishes these cases by tracking the *distribution* of token radii.

### 3.2 Tracked Statistics

For each layer $l$, at each geometry update step (every k=100 steps), compute from the temporal coordinates $x_0^{(l)}$ of key embeddings:

| Statistic | Symbol | Computation | Cost |
|-----------|--------|-------------|------|
| System token $x_0$ | $x_{0,s}^{(l)}$ | Direct read | 1 access |
| Context token median $x_0$ | $\tilde{x}_0^{(l)}$ | `torch.median(x0_context)` | $O(N)$ |
| Context token 25th percentile | $x_{0,25}^{(l)}$ | `torch.quantile(x0_context, 0.25)` | $O(N)$ |
| Context token 75th percentile | $x_{0,75}^{(l)}$ | `torch.quantile(x0_context, 0.75)` | $O(N)$ |
| Separation ratio | $\sigma^{(l)}$ | $\tilde{x}_0^{(l)} / x_{0,s}^{(l)}$ | 1 div |

**Why $x_0$ instead of Lorentz radius $r$?** Computing $r = \operatorname{arcosh}(\sqrt{c} \cdot x_0)$ requires one arcosh per token. Since $r$ is monotonically increasing in $x_0$ (for fixed $c$), the temporal coordinate is a sufficient proxy for all ordering and ratio comparisons. The decay function itself operates on $\ln(x_0)$ (DS-6 Section 4), so $x_0$ is the natural coordinate.

### 3.3 Alert Conditions

| Alert | Condition | Diagnosis | Recommended Action |
|-------|-----------|-----------|-------------------|
| **Separation Erosion** | $x_{0,25}^{(l)} < 2 \cdot x_{0,s}^{(l)}$ | Context tokens' lower quartile is within 2× the system token's temporal coordinate; radial gap closing | Increase $\gamma$ (decay controller) + increase $\mu$ (radial regularization) |
| **Curvature Compression** | $x_{0,75}^{(l)} / x_{0,25}^{(l)} < 2.0$ AND $\hat{\alpha}_s < \alpha_{\text{safe}}$ | Low radial spread combined with privilege loss; tokens are compressed into a narrow band | Increase $c$ (curvature hook) to expand $R_{\max}$ |
| **System Drift** | $x_{0,s}^{(l)} > 1/\sqrt{c^{(l)}} + 0.1$ | System token has drifted away from the origin | Increase radial regularization on system token (or re-project to origin) |

### 3.4 Computation Cadence

Radial statistics are computed **only at geometry update steps** (every k=100 steps), not every forward pass. The quantile computation is $O(N)$ — negligible compared to the $O(N^2 d)$ attention computation that runs every step.

---

## 4. Component 3: $\gamma$ Controller (Hebbian Pull)

### 4.1 Design Philosophy

The name "Hebbian Pull" comes from the analogy to Hebbian learning: the system token's attention weight is a signal of "what fires together." When the system token loses influence ($\alpha_s$ drops), the controller *pulls* it back by strengthening the radial decay — making boundary tokens less influential so the system token regains dominance. This is a negative feedback loop: privilege loss triggers stronger decay, which restores privilege.

The FAM parallel is the adaptive eviction blend: `p = 0.2 + 0.8 * min(class_loss / 0.30, 1)` ramps coverage eviction when class diversity is threatened. Here, $\gamma$ ramps decay when origin privilege is threatened.

### 4.2 Controller Design

**Proportional controller** with asymmetric gains and dead zone.

Define the **privilege error** for layer $l$:

$$e^{(l)}(t) = \alpha_{\text{safe}} - \hat{\alpha}_s^{(l)}(t)$$

- $e > 0$: privilege below target (intervention needed)
- $e \leq 0$: privilege at or above target (no intervention, allow relaxation)

The controller output is a **gradient adjustment** $\Delta g_\gamma^{(l)}$ injected into the decay parameter's accumulated gradient buffer:

$$\Delta g_\gamma^{(l)} = \begin{cases}
0 & \text{if } |e^{(l)}| < \varepsilon_{\text{dead}} \quad \text{(dead zone)} \\
-K_p \cdot e^{(l)} & \text{if } e^{(l)} > \varepsilon_{\text{dead}} \quad \text{(increase } \gamma \text{)} \\
-K_r \cdot e^{(l)} & \text{if } e^{(l)} < -\varepsilon_{\text{dead}} \quad \text{(relax } \gamma \text{)}
\end{cases}$$

Note the sign: $e > 0$ means $\alpha_s$ is too low, so $\Delta g_\gamma < 0$ (negative gradient on $\gamma_{\text{raw}}$ → softplus increases → $\gamma$ increases → stronger decay → $\alpha_s$ rises). Wait — let's be precise about the sign convention.

### 4.3 Sign Analysis

The loss $L$ depends on $\gamma$ through the attention scores. When $\gamma$ increases, decay increases, boundary tokens are penalized more, and $\alpha_s$ increases. Therefore:

$$\frac{\partial \alpha_s}{\partial \gamma} > 0$$

We want to *increase* $\gamma$ when $\alpha_s$ is too low (i.e., when $e > 0$). The gradient injection adjusts $\gamma_{\text{raw}}$, and $\gamma = \text{softplus}(\gamma_{\text{raw}})$ is increasing in $\gamma_{\text{raw}}$.

To increase $\gamma_{\text{raw}}$ via the optimizer (which does $\theta \leftarrow \theta - \text{lr} \cdot g$), we need a **negative** gradient:

$$\Delta g_\gamma^{(l)} = \begin{cases}
0 & \text{if } |e^{(l)}| < \varepsilon_{\text{dead}} \\
-K_p \cdot (e^{(l)} - \varepsilon_{\text{dead}}) & \text{if } e^{(l)} > \varepsilon_{\text{dead}} \quad \text{(negative gradient → increase } \gamma_{\text{raw}} \text{)} \\
-K_r \cdot (e^{(l)} + \varepsilon_{\text{dead}}) & \text{if } e^{(l)} < -\varepsilon_{\text{dead}} \quad \text{(positive gradient → decrease } \gamma_{\text{raw}} \text{)}
\end{cases}$$

### 4.4 Controller Parameters

| Parameter | Symbol | Default | Justification |
|-----------|--------|---------|---------------|
| Proportional gain (increase) | $K_p$ | 0.5 | Moderate intervention; at $e = 0.2$ (α_s = 0.3), injection magnitude = 0.075 |
| Relaxation gain | $K_r$ | 0.1 | 5× slower relaxation than increase; asymmetric to avoid oscillation |
| Dead zone | $\varepsilon_{\text{dead}}$ | 0.05 | Prevents chatter around the target |
| Maximum injection | $\Delta g_{\max}$ | 1.0 | Clamp to prevent explosive adjustments |
| Emergency multiplier | $K_{\text{emerg}}$ | 3.0 | Applied when state = CRITICAL ($\hat{\alpha}_s < 0.1$) |

### 4.5 Stability Analysis

**Claim:** The proportional controller with dead zone and asymmetric gains is stable (no sustained oscillation) under the following conditions:

**Condition 1 — Monotone response:** $\partial \alpha_s / \partial \gamma > 0$ everywhere. This holds because increasing $\gamma$ monotonically increases the penalty on non-origin tokens (DS-6 Section 4.1, J6: $r + \gamma \ln(\cosh(r))$ is strictly increasing).

**Condition 2 — Bounded gain:** $K_p \cdot \text{lr}_\gamma \cdot \sigma'(\gamma_{\text{raw}}) < 1$ where $\sigma' = \text{sigmoid}(\gamma_{\text{raw}})$ is the softplus derivative. At default values: $0.5 \times 10^{-5} \times 0.65 = 3.25 \times 10^{-6} \ll 1$. The per-step $\gamma$ change is microscopic.

**Condition 3 — EMA damping:** The $\rho = 0.99$ EMA smooths noise with $\tau = 100$ steps. Combined with k=100 update cadence, each controller action is based on ~100 steps of signal. The controller effectively operates at $\tau_{\text{control}} = k = 100$ steps, while the system's response time (attention weight change from $\gamma$ change) is near-instantaneous (next forward pass). This gives a stability margin of $\tau_{\text{control}} / \tau_{\text{response}} \approx 100$.

**Condition 4 — Asymmetric gains prevent overshoot:** $K_r / K_p = 0.2$, so relaxation is 5× slower than tightening. After the controller increases $\gamma$ to restore $\alpha_s$, it takes 5× longer to relax back — ample time for the EMA to confirm the correction is stable.

**Potential instability: curvature-decay coupling.** If the curvature QoS hook simultaneously adjusts $c$ (which affects $\delta$, which affects the optimal $\gamma$), the two controllers could interact. Mitigation: the dead zone ($\varepsilon_{\text{dead}} = 0.05$) absorbs small perturbations from curvature changes. The curvature controller (DS-5) and decay controller (this spec) are gradient-decoupled (DS-6 Section 7.3), so their adjustments are additive, not multiplicative.

### 4.6 Emergency Protocol

When any layer enters CRITICAL state ($\hat{\alpha}_s < 0.1$):

1. Multiply $K_p$ by $K_{\text{emerg}} = 3.0$ for that layer.
2. Log warning: `[QoS CRITICAL] Layer {l}: α_s = {value:.3f}, γ = {gamma:.3f}`.
3. If CRITICAL persists for 5 consecutive geometry updates (500 steps): trigger a **radial re-initialization** — re-project all context tokens in that layer to $r_{\text{target}} = 8.0$ via the exp-map (DS-6 Section 5.3c). This is a hard reset, not a gradient-based correction.

### 4.7 Pseudocode

```python
class GammaController:
    """Proportional controller for radial decay strength.

    Adjusts gamma_raw gradient based on alpha_s privilege signal.
    Called every k=100 steps during the geometry update.
    """

    def __init__(
        self,
        n_layers: int,
        K_p: float = 0.5,
        K_r: float = 0.1,
        alpha_safe: float = 0.5,
        eps_dead: float = 0.05,
        delta_g_max: float = 1.0,
        K_emerg: float = 3.0,
    ):
        self.n_layers = n_layers
        self.K_p = K_p
        self.K_r = K_r
        self.alpha_safe = alpha_safe
        self.eps_dead = eps_dead
        self.delta_g_max = delta_g_max
        self.K_emerg = K_emerg
        self.critical_count = [0] * n_layers  # consecutive critical steps

    def compute_adjustment(
        self,
        alpha_s_ema: torch.Tensor,   # [n_layers]
        gamma_eff: torch.Tensor,      # [n_layers] current gamma values
    ) -> torch.Tensor:
        """Compute per-layer gradient adjustment for gamma_raw.

        Returns:
            delta_g: [n_layers] gradient adjustment (additive to accumulated grad)
        """
        error = self.alpha_safe - alpha_s_ema  # positive = privilege too low
        delta_g = torch.zeros_like(error)

        # Dead zone
        active_low = error > self.eps_dead      # need to increase gamma
        active_high = error < -self.eps_dead    # can relax gamma

        # Proportional control with asymmetric gains
        delta_g[active_low] = -self.K_p * (error[active_low] - self.eps_dead)
        delta_g[active_high] = -self.K_r * (error[active_high] + self.eps_dead)

        # Emergency multiplier for critical layers
        critical = alpha_s_ema < 0.1
        delta_g[critical] *= self.K_emerg

        # Update critical counters
        for l in range(self.n_layers):
            if critical[l].item():
                self.critical_count[l] += 1
            else:
                self.critical_count[l] = 0

        # Clamp
        delta_g.clamp_(-self.delta_g_max, self.delta_g_max)

        return delta_g

    def needs_reinit(self) -> list[int]:
        """Return layer indices that need emergency radial re-initialization."""
        return [l for l in range(self.n_layers) if self.critical_count[l] >= 5]
```

---

## 5. Component 4: Unified QoS Hook API

### 5.1 Problem

DS-5 defines `register_qos_hook` on `CurvatureModule` with signature:

```python
hook_fn(c_eff, c_grad_avg, N_eff, step) -> Optional[Tensor]  # curvature adjustment
```

DS-6 introduces `RadialDecayModule` with its own gradient accumulation buffer. The $\gamma$ controller needs to inject gradients into *both* modules based on the same $\alpha_s$ signal. Running two separate hook systems would be fragile and hard to coordinate.

### 5.2 Design: GeometryQoSHub

A single coordinator that:
1. Collects monitoring signals ($\alpha_s$ EMA, radial distribution stats).
2. Runs the $\gamma$ controller.
3. Dispatches gradient adjustments to both CurvatureModule and RadialDecayModule.

```python
class GeometryQoSHub:
    """Unified QoS coordinator for curvature and decay parameters.

    Sits between the monitoring signals and the DS-5/DS-6 parameter modules.
    Called every k=100 steps during the geometry update.
    """

    def __init__(
        self,
        curvature_module: AdaptiveCurvatureModule,
        decay_module: RadialDecayModule,
        n_layers: int,
        n_heads: int,
        rho: float = 0.99,         # EMA decay for alpha_s tracker
        alpha_safe: float = 0.5,
        alpha_low: float = 0.3,
        alpha_crit: float = 0.1,
    ):
        self.curvature = curvature_module
        self.decay = decay_module
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.rho = rho

        # Alpha_s tracker (Component 1)
        self.alpha_s_ema = torch.full((n_layers,), 0.5)

        # Radial distribution monitor (Component 2)
        self.radial_stats = {
            'x0_system': torch.ones(n_layers),
            'x0_median': torch.ones(n_layers),
            'x0_q25': torch.ones(n_layers),
            'x0_q75': torch.ones(n_layers),
        }

        # Gamma controller (Component 3)
        self.gamma_controller = GammaController(n_layers)

        # State tracking
        self.thresholds = {
            'safe': alpha_safe,
            'low': alpha_low,
            'crit': alpha_crit,
        }
        self.layer_states = ['HEALTHY'] * n_layers

        # Register as QoS hook on curvature module
        self.curvature.register_qos_hook(self._curvature_qos_hook)

    # ─── Called every forward pass (lightweight) ───

    def update_alpha_s(self, attn_weights: torch.Tensor, system_mask: torch.Tensor, layer_idx: int):
        """Update alpha_s EMA from attention weights.

        attn_weights: [B, H, N_q, N_k]
        system_mask: [B, N] boolean
        """
        with torch.no_grad():
            sys_indices = system_mask.nonzero(as_tuple=True)
            alpha_vals = attn_weights[sys_indices[0], :, sys_indices[1], sys_indices[1]]
            alpha_mean = alpha_vals.mean().item()
            self.alpha_s_ema[layer_idx] = (
                self.rho * self.alpha_s_ema[layer_idx] + (1 - self.rho) * alpha_mean
            )
            self._update_state(layer_idx)

    # ─── Called every k=100 steps (geometry update) ───

    def update_radial_stats(self, key_embeddings: torch.Tensor, system_mask: torch.Tensor, c_eff: torch.Tensor, layer_idx: int):
        """Update radial distribution statistics.

        key_embeddings: [B, H, N, d+1] Lorentz embeddings
        """
        with torch.no_grad():
            x0 = key_embeddings[..., 0]  # [B, H, N]
            x0_flat = x0.reshape(-1, x0.shape[-1])  # [B*H, N]
            sys_mask_expanded = system_mask.unsqueeze(1).expand_as(x0)

            # System token x0 (average over batch/heads)
            x0_sys = x0[sys_mask_expanded].mean()

            # Context token statistics
            ctx_mask = ~sys_mask_expanded
            x0_ctx = x0[ctx_mask]

            self.radial_stats['x0_system'][layer_idx] = x0_sys
            self.radial_stats['x0_median'][layer_idx] = x0_ctx.median()
            self.radial_stats['x0_q25'][layer_idx] = torch.quantile(x0_ctx, 0.25)
            self.radial_stats['x0_q75'][layer_idx] = torch.quantile(x0_ctx, 0.75)

    def _curvature_qos_hook(self, c_eff, c_grad_avg, N_eff, step):
        """DS-5 QoS hook: called during curvature update.

        Returns curvature gradient adjustment based on radial distribution analysis.
        """
        # Check for curvature compression
        q75 = self.radial_stats['x0_q75']
        q25 = self.radial_stats['x0_q25']
        spread = q75 / (q25 + 1e-8)

        compressed = (spread < 2.0) & (self.alpha_s_ema < self.thresholds['safe'])

        if compressed.any():
            # Inject negative curvature gradient (increase c to expand R_max)
            adj = torch.zeros_like(c_grad_avg)
            adj[compressed] = -0.1  # gentle push toward higher curvature
            return adj
        return None

    def compute_gamma_adjustment(self) -> torch.Tensor:
        """Compute gamma_raw gradient adjustment from the controller.

        Called by RadialDecayModule during its geometry update.
        """
        gamma_eff = self.decay()
        return self.gamma_controller.compute_adjustment(self.alpha_s_ema, gamma_eff)

    def check_emergency_reinit(self) -> list[int]:
        """Return layer indices needing emergency radial re-initialization."""
        return self.gamma_controller.needs_reinit()

    # ─── Internal ───

    def _update_state(self, layer_idx):
        alpha = self.alpha_s_ema[layer_idx].item()
        current = self.layer_states[layer_idx]
        hysteresis = 0.05

        if alpha >= self.thresholds['safe'] + (hysteresis if current != 'HEALTHY' else 0):
            self.layer_states[layer_idx] = 'HEALTHY'
        elif alpha >= self.thresholds['low'] + (hysteresis if current in ('ALERT', 'CRITICAL') else 0):
            self.layer_states[layer_idx] = 'DEGRADED'
        elif alpha >= self.thresholds['crit'] + (hysteresis if current == 'CRITICAL' else 0):
            self.layer_states[layer_idx] = 'ALERT'
        else:
            self.layer_states[layer_idx] = 'CRITICAL'
```

### 5.3 Extended RadialDecayModule

The DS-6 `RadialDecayModule` needs one addition — a QoS hook interface parallel to DS-5's:

```python
class RadialDecayModule(nn.Module):
    # ... (existing from DS-6 Section 9.3) ...

    def __init__(self, n_layers, gamma_init=0.6):
        super().__init__()
        # ... existing init ...
        self._qos_hooks: list[Callable] = []

    def register_qos_hook(self, hook_fn: Callable) -> int:
        """Register a QoS hook called during decay parameter updates.

        hook_fn signature:
            (gamma_eff: Tensor[n_layers],
             gamma_grad_avg: Tensor[n_layers],
             step: int) -> Optional[Tensor[n_layers]]

        Return value: additive adjustment to gamma_grad_avg, or None.
        """
        hook_id = len(self._qos_hooks)
        self._qos_hooks.append(hook_fn)
        return hook_id

    def apply_decay_step(self, step: int = 0):
        """Apply accumulated gradient every k=100 steps. Extended with QoS hooks."""
        if self.gamma_grad_count.item() == 0:
            return
        avg_grad = self.gamma_grad_buffer / self.gamma_grad_count.float()

        # QoS hooks
        for hook_fn in self._qos_hooks:
            adj = hook_fn(F.softplus(self.gamma_raw), avg_grad, step)
            if adj is not None:
                avg_grad = avg_grad + adj

        self.gamma_raw.grad = avg_grad.clone()
        self.gamma_grad_buffer.zero_()
        self.gamma_grad_count.zero_()
```

### 5.4 Training Loop Integration

```python
# Setup
curvature = AdaptiveCurvatureModule(n_layers=32, beta=1.0)
decay = RadialDecayModule(n_layers=32, gamma_init=0.6)
qos = GeometryQoSHub(curvature, decay, n_layers=32, n_heads=8)

# Register gamma controller as decay QoS hook
decay.register_qos_hook(
    lambda gamma_eff, gamma_grad_avg, step: qos.compute_gamma_adjustment()
)

# Training loop
for step, batch in enumerate(dataloader):
    c_eff = curvature(batch.seq_len)
    gamma = decay()

    # Forward pass with per-layer alpha_s tracking
    for layer_idx, layer in enumerate(model.layers):
        output, attn_weights = layer.attention(batch, c_eff[layer_idx], gamma[layer_idx])
        qos.update_alpha_s(attn_weights, batch.system_mask, layer_idx)

    loss = criterion(output, batch.labels)
    loss_radial = radial_regularization_loss(model.keys, c_eff)
    (loss + loss_radial).backward()

    c_ready = curvature.accumulate_and_gate_grad()
    g_ready = decay.accumulate_and_gate_grad()

    optimizer.step()
    optimizer.zero_grad()

    if c_ready or g_ready:
        # Radial stats update (every k=100 steps)
        with torch.no_grad():
            for l in range(32):
                qos.update_radial_stats(model.keys[l], batch.system_mask, c_eff, l)

        if c_ready:
            curvature.apply_curvature_step(optimizer)  # fires curvature QoS hook
        if g_ready:
            decay.apply_decay_step(step)  # fires decay QoS hook (gamma controller)

        optimizer.step()
        optimizer.zero_grad()

        # Emergency check
        reinit_layers = qos.check_emergency_reinit()
        if reinit_layers:
            radial_reinit(model, reinit_layers, r_target=8.0, c_eff=c_eff)
```

---

## 6. Failure Mode Catalog

| # | Failure Mode | Trigger | Observable Symptom | Mitigation |
|---|-------------|---------|-------------------|------------|
| Q1 | **Controller oscillation** | $K_p$ too large relative to EMA damping; $\gamma$ overshoots, then relaxes, then overshoots again | $\gamma$ oscillates with period $\sim 2k$ steps; $\hat{\alpha}_s$ fluctuates around $\alpha_{\text{safe}}$ | (a) Dead zone $\varepsilon_{\text{dead}} = 0.05$ absorbs small errors. (b) Asymmetric gains ($K_r = K_p / 5$) ensure slow relaxation. (c) Stability condition: $K_p \cdot \text{lr}_\gamma \cdot \sigma'(\gamma_{\text{raw}}) \ll 1$ (verified in Section 4.5). |
| Q2 | **False alarm from batch variance** | Short sequences in a batch produce naturally high $\alpha_s$; long sequences produce low $\alpha_s$; EMA alternates | Layer state oscillates between HEALTHY and DEGRADED across batches | EMA with $\rho = 0.99$ ($\tau = 100$) smooths batch-level variance. At k=100 update cadence, the controller acts on 100-step averages. |
| Q3 | **Curvature-decay runaway** | Both curvature and decay controllers react to the same $\alpha_s$ drop; combined adjustment overshoots | $c$ and $\gamma$ both increase simultaneously; $\alpha_s$ spikes to $\sim 1.0$; then both controllers relax; repeat | (a) Curvature hook is conservative ($\Delta c = -0.1$, only fires when radial compression detected). (b) Dead zone prevents chatter. (c) DS-6 Section 7.3 proves curvature and decay are gradient-decoupled — their adjustments are independent. |
| Q4 | **EMA initialization transient** | $\hat{\alpha}_s$ initialized to 0.5; actual $\alpha_s$ may start lower, triggering false DEGRADED/ALERT at step 0 | Controller injects $\gamma$ increase during warmup before the model has learned meaningful attention | Warmup period: disable controller for first 1000 steps (10 geometry updates). Allow EMA to converge before enabling control actions. |
| Q5 | **Quantile computation on variable-length batches** | Padded tokens (attention masked) included in $x_0$ statistics | Radial statistics contaminated by padding token embeddings | Apply the attention mask to exclude padded positions from quantile computation. Only compute over tokens where `attention_mask == True AND system_mask == False`. |
| Q6 | **Emergency reinit destroys learned representations** | 5 consecutive CRITICAL steps trigger radial re-initialization; good token placements in that layer are overwritten | Temporary accuracy drop after reinit; other layers compensate but overall quality degrades | (a) Reinit only affects radial position (temporal coordinate), not angular direction. Use scaled exp-map to re-project: $x' = \exp_o(r_{\text{target}} \cdot v / \|v\|)$ where $v = \log_o(x)$ preserves the direction. (b) Reinit is rare (requires 500+ steps of $\alpha_s < 0.1$) and indicates a fundamental geometry failure — the accuracy cost is acceptable vs. permanent privilege loss. |

---

## 7. FAM $\leftrightarrow$ HCA QoS Mapping

| FAM Concept | FAM Mechanism | HCA QoS Analogue |
|-------------|---------------|-----------------|
| `class_loss` signal | $1 - (n_{\text{present}} / n_{\text{ever\_seen}})$ | Privilege error: $e = \alpha_{\text{safe}} - \hat{\alpha}_s$ |
| Adaptive blend ramp | $p = 0.2 + 0.8 \cdot \min(\text{class\_loss} / 0.30, 1)$ | $\Delta g = -K_p \cdot (e - \varepsilon_{\text{dead}})$; clamped at $\Delta g_{\max}$ |
| LRU fallback (class_loss = 0) | Pure `last_seen` eviction | $\gamma \to 0$ when $\alpha_s \gg \alpha_{\text{safe}}$ (no decay needed) |
| Coverage eviction (class_loss > 0) | Evict most replaceable prototype | Increase $\gamma$ (penalize boundary tokens more) |
| `_classes_ever_seen` tracking | `set[int]` monotonically growing | $\hat{\alpha}_s$ EMA with $\rho = 0.99$ |
| 30% class-loss threshold | $p$ ramps when 30% of classes lost | $\alpha_{\text{safe}} = 0.5$ (50% privilege target) |
| Sole class representative protection | `score = inf` | System token at origin: $\lambda(0) = 0$ by construction |

---

## 8. Hot-Path Cost Summary

| Component | Per-step cost | Per-k-steps cost | Attention kernel overhead |
|-----------|--------------|-------------------|--------------------------|
| $\alpha_s$ tracker | 1 index + 1 mean per layer | — | **Zero** (reads existing attn_weights) |
| Radial distribution | — | $O(N)$ quantile per layer | **Zero** (reads existing embeddings) |
| $\gamma$ controller | — | $O(L)$ scalar ops | **Zero** (modifies gradient buffer) |
| Curvature QoS hook | — | $O(L)$ scalar ops | **Zero** (modifies gradient buffer) |
| **Total attention kernel overhead** | | | **Zero** |

All monitoring reads from tensors already computed by the forward pass. All control actions modify gradient buffers that are applied during the geometry update step (not during the forward pass).

---

## 9. Verification Plan (MT-2 Extension)

The following tests extend the MT-2 verification plan from DS-6 Section 10.3:

| Test | Metric | Pass criteria |
|------|--------|---------------|
| $\alpha_s$ EMA accuracy | $\|\hat{\alpha}_s - \bar{\alpha}_s\|$ over 1000 steps | Mean absolute error < 0.02 after 200-step warmup |
| Controller response time | Steps to restore $\alpha_s \geq 0.3$ after synthetic perturbation ($\gamma \to 0$) | $< 500$ steps ($= 5$ geometry updates) |
| Controller stability | $\text{std}(\gamma)$ over 1000-step window during steady state | $< 0.05$ (no oscillation) |
| Curvature-decay interaction | $\text{corr}(\Delta c, \Delta \gamma)$ over training | $|\text{corr}| < 0.3$ |
| Emergency reinit recovery | Accuracy 100 steps after reinit vs. 100 steps before | $\Delta \text{acc} > -2\%$ (limited damage) |
| End-to-end privilege maintenance | $\min_l \hat{\alpha}_s^{(l)}$ over full training run | $\geq 0.25$ after warmup (never enters CRITICAL) |
| Warmup transient | Controller injection magnitude during first 1000 steps | Identically 0 (disabled during warmup) |

---

## 10. Open Questions and DS-8 Handoff Notes

### 10.1 Open Questions

**OQ-1: Head-level vs. head-averaged $\alpha_s$.** The current design averages $\alpha_s$ across attention heads. Some heads may specialize in non-privileged patterns (e.g., local attention). Per-head tracking would allow the controller to be more selective, but adds $H \times L$ tracked scalars.

**Resolution:** Default to head-averaged. If MT-2 shows high inter-head variance in $\alpha_s$, revisit with per-head tracking and per-head $\gamma$.

**OQ-2: Integral term (PI controller).** A proportional-only controller has steady-state error when the system dynamics include a constant disturbance (e.g., a data distribution shift that permanently reduces $\alpha_s$). Adding an integral term eliminates steady-state error but risks integral windup.

**Resolution:** Defer to Phase 4. The proportional controller is simpler, and the dead zone absorbs small steady-state errors. If persistent DEGRADED states are observed in MT-2, add integral with anti-windup.

**OQ-3: Multi-system-token architectures.** Some designs may use multiple "privileged" tokens (e.g., task-specific system prompts). The current spec assumes a single system token per sequence.

**Resolution:** Generalize `system_mask` to mark multiple tokens. Track the *minimum* $\alpha_s$ across all system tokens. The controller then ensures the *weakest* privileged token maintains its guarantee.

**OQ-4: Interaction with learning rate scheduling.** The controller's effective gain depends on the optimizer's learning rate. During warmup (LR ramp), the controller's adjustments are amplified; during decay, they are attenuated.

**Resolution:** Normalize the controller output by the current LR: $\Delta g_\gamma^{(l)} \leftarrow \Delta g_\gamma^{(l)} / \text{lr}_\gamma$. This makes the physical $\gamma$ change independent of the LR schedule.

### 10.2 DS-8 Handoff: Equilibrium Analysis

DS-8 should analyze:

1. **Fixed-point existence:** Does the system $(\hat{\alpha}_s, \gamma, c)$ have a stable equilibrium under the combined curvature + decay controllers?
2. **Convergence rate:** How many geometry updates to reach equilibrium from arbitrary initial conditions?
3. **Basin of attraction:** What initial $(\gamma_0, c_0)$ configurations converge to the correct equilibrium vs. requiring emergency reinit?
4. **Sparse attention interaction:** How does the QoS monitor behave under block-sparse or sliding-window attention patterns where only a subset of keys are visible?

---

## Appendix A: Notation Reference

| Symbol | Definition |
|--------|-----------|
| $\alpha_s^{(l)}$ | System token attention weight in layer $l$ |
| $\hat{\alpha}_s^{(l)}$ | EMA of $\alpha_s$ for layer $l$ |
| $\rho$ | EMA decay rate (0.99) |
| $\alpha_{\text{safe}}$ | Safety threshold (0.5) |
| $\alpha_{\text{low}}$ | Alert threshold (0.3) |
| $\alpha_{\text{crit}}$ | Critical threshold (0.1) |
| $e^{(l)}$ | Privilege error: $\alpha_{\text{safe}} - \hat{\alpha}_s^{(l)}$ |
| $K_p$ | Proportional gain for $\gamma$ increase (0.5) |
| $K_r$ | Relaxation gain for $\gamma$ decrease (0.1) |
| $\varepsilon_{\text{dead}}$ | Dead zone width (0.05) |
| $\Delta g_\gamma$ | Gradient adjustment injected into $\gamma_{\text{raw}}$ |
| $\sigma^{(l)}$ | Separation ratio: $\tilde{x}_0 / x_{0,s}$ |
| $k$ | Geometry update cadence (100 steps, from DS-5) |
