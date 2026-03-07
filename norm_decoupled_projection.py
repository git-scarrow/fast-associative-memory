"""
norm_decoupled_projection.py — Norm-Decoupled Hyperbolic Projection (HCA-DS-14).

Separates direction (angular position) from magnitude (radial position)
so that L2-normalized inputs can still produce meaningful radial variation
on the Lorentz hyperboloid.

The magnitude head learns to predict "hierarchical depth" from the feature
vector, enabling different classes to occupy different radii — the prerequisite
for hyperbolic volume expansion to encode hierarchy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lorentz_ops import exp_map_origin, EPS_NORM


class NormDecoupledProjection(nn.Module):
    """Project Euclidean features to the Lorentz hyperboloid with learned radial placement.

    Direction = L2-normalize(input) — preserves angular discriminability exactly.
    Magnitude = learned MLP head — maps features to a radial position.

    The tangent vector v = direction * magnitude is then exp-mapped to the hyperboloid.

    Args:
        d_in: Input feature dimension (e.g. 1024 for ViT-L/14).
        c_init: Initial curvature (positive).
        c_min: Curvature floor (from HCA-DS-5).
        mag_hidden: Hidden dim for the magnitude head.
        mag_min: Minimum radius (prevents collapse to origin).
        mag_max: Maximum radius (prevents numerical overflow in cosh).
        use_input_norm: If True, also feed the raw input norm to magnitude head.
    """

    def __init__(self, d_in: int, c_init: float = 1.0, c_min: float = 0.1,
                 mag_hidden: int = 128, mag_min: float = 0.5, mag_max: float = 2.5,
                 use_input_norm: bool = True):
        super().__init__()
        self.d_in = d_in
        self.mag_min = mag_min
        self.mag_max = mag_max
        self.use_input_norm = use_input_norm

        # No direction head — just L2-normalize the input to preserve
        # angular discriminability (the critical signal for classification).

        # Magnitude head: predicts scalar radius from features.
        # When use_input_norm=True, also gets the raw input norm as extra signal.
        mag_in = d_in + (1 if use_input_norm else 0)
        self.magnitude_head = nn.Sequential(
            nn.Linear(mag_in, mag_hidden),
            nn.ReLU(),
            nn.Linear(mag_hidden, 1),
        )

        # Learnable curvature (HCA-DS-2 parameterization)
        self.c_raw = nn.Parameter(torch.tensor(0.0))
        self.c_min = c_min

        self._init_weights()

    def _init_weights(self):
        # Magnitude: small init for stable starting point
        nn.init.xavier_uniform_(self.magnitude_head[0].weight, gain=0.5)
        nn.init.zeros_(self.magnitude_head[0].bias)
        nn.init.xavier_uniform_(self.magnitude_head[2].weight, gain=0.1)
        # Bias so initial sigmoid output ~ 0.5 -> mid-range radius
        nn.init.constant_(self.magnitude_head[2].bias, 0.0)

    @property
    def c_eff(self) -> torch.Tensor:
        return self.c_min + F.softplus(self.c_raw)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map Euclidean features to Lorentz hyperboloid.

        Args:
            z: (B, d_in) — input features (may or may not be L2-normalized).

        Returns:
            (B, d_in+1) — points on the Lorentz hyperboloid.
        """
        # Direction: L2-normalize input (preserves angular structure exactly)
        direction = F.normalize(z, dim=-1)

        # Magnitude: learned scalar radius in [mag_min, mag_max]
        if self.use_input_norm:
            input_norm = z.norm(dim=-1, keepdim=True)
            mag_input = torch.cat([z, input_norm], dim=-1)
        else:
            mag_input = z
        raw_mag = self.magnitude_head(mag_input)  # (B, 1)
        magnitude = self.mag_min + (self.mag_max - self.mag_min) * torch.sigmoid(raw_mag)

        # Tangent vector with learned norm
        v = direction * magnitude  # (B, d_in)

        # Exp-map to hyperboloid
        c = self.c_eff
        return exp_map_origin(v, c.item())

    def get_curvature(self) -> float:
        return self.c_eff.item()
