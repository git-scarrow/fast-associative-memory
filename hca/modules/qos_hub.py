"""GeometryQoSHub: 4-state hysteresis monitor with gradient injection (DS-7)."""

import enum
import torch
import math
import logging

logger = logging.getLogger(__name__)


class QoSState(enum.IntEnum):
    HEALTHY = 0
    DEGRADED = 1
    ALERT = 2
    CRITICAL = 3


class GeometryQoSHub:
    """4-state hysteresis machine for monitoring and correcting HCA geometry health."""

    def __init__(
        self,
        n_layers: int,
        alpha_safe: float = 0.5,
        eps_dead: float = 0.05,
        K_p: float = 0.5,
        K_r: float = 0.1,
        dg_max: float = 1.0,
        emergency_mult: float = 3.0,
        hysteresis_band: float = 0.05,
        compression_threshold: float = 0.5,
        rho: float = 0.99,
        gamma_lr: float = 0.01,
    ):
        self.n_layers = n_layers
        self.alpha_safe = alpha_safe
        self.eps_dead = eps_dead
        self.K_p = K_p
        self.K_r = K_r
        self.dg_max = dg_max
        self.emergency_mult = emergency_mult
        self.hysteresis_band = hysteresis_band
        self.compression_threshold = compression_threshold
        self.rho = rho
        self.gamma_lr = gamma_lr

        # Per-layer state
        self.alpha_s_ema = [0.5] * n_layers
        self.state = [QoSState.HEALTHY] * n_layers
        self.state_duration = [0] * n_layers
        self.prev_threshold = [0.0] * n_layers

    def update(self, step: int, alpha_s_per_layer: list, model):
        """Run QoS update at k-step cadence.

        Args:
            step: current training step
            alpha_s_per_layer: list of alpha_s values per layer
            model: the model (for accessing gamma_raw, c_raw parameters)
        """
        for l in range(self.n_layers):
            alpha_s = float(alpha_s_per_layer[l])

            # EMA update
            self.alpha_s_ema[l] = self.rho * self.alpha_s_ema[l] + (1 - self.rho) * alpha_s

            # Privilege error
            e = self.alpha_safe - self.alpha_s_ema[l]

            # State transition
            prev_state = self.state[l]
            if e < -self.eps_dead:  # alpha_s too HIGH
                new_state = QoSState.HEALTHY
            elif abs(e) <= self.eps_dead:
                new_state = QoSState.HEALTHY
            elif e > self.eps_dead and e <= 0.15:
                new_state = QoSState.DEGRADED
            elif e > 0.15 and e <= 0.3:
                new_state = QoSState.ALERT
            else:
                new_state = QoSState.CRITICAL

            # Also escalate to CRITICAL if stuck
            if self.state[l] == QoSState.CRITICAL:
                self.state_duration[l] += 1
            else:
                self.state_duration[l] = 0
            if e > 0.3 or (self.state[l] >= QoSState.ALERT and self.state_duration[l] >= 5):
                new_state = QoSState.CRITICAL

            # Hysteresis: require improvement to step down
            if new_state < prev_state:
                if e > self.prev_threshold[l] - self.hysteresis_band:
                    new_state = prev_state

            self.state[l] = new_state
            self.prev_threshold[l] = e

            # Track duration in current state
            if new_state == prev_state:
                self.state_duration[l] += 1
            else:
                self.state_duration[l] = 0

            # Corrective actions via gradient injection
            layer_module = self._get_layer_module(model, l)
            if layer_module is None:
                continue

            gamma_raw = layer_module.gamma_raw if hasattr(layer_module, "gamma_raw") else None
            c_raw = layer_module.curvature.c_raw if hasattr(layer_module, "curvature") else None

            delta_g_gamma = 0.0

            if self.state[l] == QoSState.HEALTHY:
                if self.alpha_s_ema[l] > self.alpha_safe + self.eps_dead:
                    delta_g_gamma = self.K_r * (self.alpha_s_ema[l] - self.alpha_safe - self.eps_dead)
                    self._inject_gradient(gamma_raw, delta_g_gamma)

            elif self.state[l] in (QoSState.DEGRADED, QoSState.ALERT):
                delta_g_gamma = -self.K_p * (e - self.eps_dead)
                delta_g_gamma = max(-self.dg_max, min(self.dg_max, delta_g_gamma))
                self._inject_gradient(gamma_raw, delta_g_gamma)

            elif self.state[l] == QoSState.CRITICAL:
                delta_g_gamma = -self.emergency_mult * self.K_p * (e - self.eps_dead)
                self._inject_gradient(gamma_raw, delta_g_gamma)

                # Curvature compression check
                if c_raw is not None:
                    x0_vals = self._get_x0_values(model, l)
                    if x0_vals is not None:
                        r_vals = torch.acosh(x0_vals.clamp(min=1.0 + 1e-7))
                        r_median = r_vals.median().item()
                        if r_median > 0:
                            r_iqr = (torch.quantile(r_vals.float(), 0.75) - torch.quantile(r_vals.float(), 0.25)).item()
                            if r_iqr / r_median < self.compression_threshold:
                                self._inject_gradient(c_raw, -0.1)

            logger.debug(
                f"QoS layer={l} step={step} state={self.state[l].name} "
                f"alpha_s={alpha_s:.4f} ema={self.alpha_s_ema[l]:.4f} "
                f"delta_gamma={delta_g_gamma:.4f}"
            )

    def _inject_gradient(self, param, delta):
        """Inject synthetic gradient and apply mini-step."""
        if param is None:
            return
        with torch.no_grad():
            param.data -= self.gamma_lr * delta

    def _get_layer_module(self, model, layer_idx):
        """Get HCAAttention module for a given layer."""
        if hasattr(model, "layers"):
            if layer_idx < len(model.layers):
                layer = model.layers[layer_idx]
                if hasattr(layer, "self_attn"):
                    return layer.self_attn
        if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            if layer_idx < len(model.encoder.layers):
                layer = model.encoder.layers[layer_idx]
                if hasattr(layer, "self_attn"):
                    return layer.self_attn
        return None

    def _get_x0_values(self, model, layer_idx):
        """Get temporal coordinate values for compression check."""
        layer = self._get_layer_module(model, layer_idx)
        if layer is None:
            return None
        # x0 values would need to be cached during forward pass
        if hasattr(layer, "_cached_x0"):
            return layer._cached_x0
        return None

    def get_states(self):
        """Return current QoS states for all layers."""
        return [s.name for s in self.state]
