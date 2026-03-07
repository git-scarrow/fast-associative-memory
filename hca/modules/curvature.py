"""CurvatureModule: per-layer learnable curvature with floor enforcement (H6)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CurvatureModule(nn.Module):
    """Per-layer curvature c_eff = max(c_min, c_min + softplus(c_raw)), c_init=1.0.

    The floor c_min(N) = (N + 2 + N^{-1}) / (4*M^2) adapts to sequence length.
    M is a model-dependent scale factor (default: d_model).
    """

    def __init__(self, c_init: float = 1.0, M: float = 128.0):
        super().__init__()
        # c_eff = c_min + softplus(c_raw) => softplus(c_raw) = c_init - c_min
        # At init with N unknown, set c_raw so softplus(c_raw) ~ c_init
        # softplus^{-1}(x) = log(exp(x) - 1)
        c_raw_init = torch.log(torch.tensor(max(torch.exp(torch.tensor(c_init)) - 1.0, 1e-6)))
        self.c_raw = nn.Parameter(torch.tensor(c_raw_init.item()))
        self.M = M
        self._n_eff_ema = 128.0  # initial EMA of effective sequence length
        self._ema_k = 100  # update cadence

    def c_min(self, N: float) -> float:
        """Curvature floor: c_min(N) = (N + 2 + 1/N) / (4*M^2)."""
        return (N + 2.0 + 1.0 / max(N, 1.0)) / (4.0 * self.M * self.M)

    def forward(self, N: int = None) -> torch.Tensor:
        """Return c_eff for current sequence length N."""
        if N is None:
            N = self._n_eff_ema

        c_min_val = self.c_min(float(N))
        # Hybrid mode: max(c_min(128K), c_min(N_eff))
        c_min_128k = self.c_min(131072.0)
        c_floor = max(c_min_128k, c_min_val)

        c_eff = c_floor + F.softplus(self.c_raw)
        return c_eff

    def update_n_eff(self, N: int):
        """Update EMA of effective sequence length (called every k steps)."""
        self._n_eff_ema = 0.99 * self._n_eff_ema + 0.01 * float(N)
