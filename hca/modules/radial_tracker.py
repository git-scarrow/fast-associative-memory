"""RadialPositionTracker: Hebbian pull updates and radial regularization (DS-6)."""

import torch
import torch.nn as nn
import math


class RadialPositionTracker(nn.Module):
    """Tracks per-token radial positions and applies Hebbian pull at k=100 cadence."""

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        max_seq_len: int,
        r_init: float = 8.0,
        r_reg_weight: float = 0.01,
        pull_lr: float = 1e-3,
        pull_threshold: float = 0.1,
        pull_damping: float = 0.95,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.r_init = r_init
        self.r_reg_weight = r_reg_weight
        self.pull_lr = pull_lr
        self.pull_threshold = pull_threshold
        self.pull_damping = pull_damping

        # Auxiliary state (not model parameters)
        self.register_buffer("pull_accumulator", torch.zeros(n_layers, n_heads, max_seq_len))
        self.register_buffer("pull_count", torch.zeros(n_layers, n_heads, max_seq_len, dtype=torch.int32))

    def accumulate_pull(self, alpha: torch.Tensor, layer: int):
        """Accumulate Hebbian pull signal from attention weights.

        Args:
            alpha: attention weights [B, H, N_q, N_k]
            layer: layer index
        """
        with torch.no_grad():
            N_k = alpha.shape[-1]
            # Mean attention each key receives from all queries
            pull_signal = alpha.mean(dim=(0, 2))  # [H, N_k]
            mask = (pull_signal > self.pull_threshold).float()
            self.pull_accumulator[layer, :, :N_k] += (pull_signal * mask).detach()
            self.pull_count[layer, :, :N_k] += mask.int().detach()

    def apply_pull(self, x_sp: torch.Tensor, layer: int, head: int, c: float) -> torch.Tensor:
        """Apply Hebbian pull toward origin for high-attention tokens.

        Args:
            x_sp: spatial coordinates [N, d]
            layer: layer index
            head: head index
            c: curvature

        Returns:
            Updated spatial coordinates [N, d]
        """
        N = x_sp.shape[0]
        count = self.pull_count[layer, head, :N].clamp(min=1).float()
        avg_pull = self.pull_accumulator[layer, head, :N] / count

        # Scale down spatial coordinates (pull toward origin)
        scale = 1.0 - self.pull_lr * avg_pull * self.pull_damping
        scale = scale.clamp(0.1, 1.0)  # prevent radial collapse

        x_sp_new = x_sp * scale.unsqueeze(-1)

        # Reset accumulators for this layer/head
        self.pull_accumulator[layer, head, :N] = 0.0
        self.pull_count[layer, head, :N] = 0

        return x_sp_new

    def radial_regularization(self, x0_values: torch.Tensor, c: float) -> torch.Tensor:
        """Compute radial regularization loss: mu * mean((x0 - x0_target)^2).

        Args:
            x0_values: temporal coordinates [...]
            c: curvature value

        Returns:
            Scalar regularization loss
        """
        sqrt_c = math.sqrt(c) if isinstance(c, float) else c.sqrt()
        x0_target = math.cosh(self.r_init) / sqrt_c if isinstance(c, float) else torch.cosh(torch.tensor(self.r_init)) / sqrt_c
        if isinstance(x0_target, float):
            x0_target = torch.tensor(x0_target, device=x0_values.device, dtype=x0_values.dtype)
        return self.r_reg_weight * ((x0_values - x0_target) ** 2).mean()
