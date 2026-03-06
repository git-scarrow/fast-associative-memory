"""Gradient checks for all 3 Lorentz autograd.Function ops."""

import pytest
import torch
from torch.autograd import gradcheck

from hca.ops.lorentz_distance import LorentzSquaredDistance
from hca.ops.lorentz_exp_map import LorentzExpMap
from hca.ops.lorentz_log_map import LorentzLogMap


def _random_lorentz_spatial(shape, c=1.0, r_range=(0.5, 5.0)):
    """Generate random spatial coordinates on the Lorentz hyperboloid."""
    d = shape[-1]
    x_sp = torch.randn(*shape, dtype=torch.float64)
    # Normalize to random radii
    norms = x_sp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    target_r = torch.empty(*shape[:-1], 1, dtype=torch.float64).uniform_(*r_range)
    import math
    target_x0 = torch.cosh(target_r) / math.sqrt(c)
    target_sp_norm = torch.sqrt((target_x0 ** 2 - 1.0 / c).clamp(min=1e-8))
    x_sp = x_sp / norms * target_sp_norm
    return x_sp


class TestLorentzSquaredDistance:
    def test_gradcheck_3d(self):
        """Gradient check for [B, N, d] inputs."""
        B, N, d = 2, 3, 4
        c = 1.0
        x_sp = _random_lorentz_spatial((B, N, d), c).requires_grad_(True)
        y_sp = _random_lorentz_spatial((B, N, d), c).requires_grad_(True)
        assert gradcheck(LorentzSquaredDistance.apply, (x_sp, y_sp, c), eps=1e-4, atol=1e-3)

    def test_gradcheck_4d(self):
        """Gradient check for [B, H, N, d] multi-head inputs."""
        B, H, N, d = 2, 2, 3, 4
        c = 1.0
        x_sp = _random_lorentz_spatial((B, H, N, d), c).requires_grad_(True)
        y_sp = _random_lorentz_spatial((B, H, N, d), c).requires_grad_(True)
        assert gradcheck(LorentzSquaredDistance.apply, (x_sp, y_sp, c), eps=1e-4, atol=1e-3)

    def test_output_nonnegative(self):
        """Squared distance proxy should be >= 0."""
        x_sp = _random_lorentz_spatial((4, 8, 5))
        y_sp = _random_lorentz_spatial((4, 8, 5))
        dist = LorentzSquaredDistance.apply(x_sp, y_sp, 1.0)
        assert (dist >= 0).all()

    def test_self_distance_near_zero(self):
        """Distance from a point to itself should be ~0."""
        x_sp = _random_lorentz_spatial((4, 8, 5))
        dist = LorentzSquaredDistance.apply(x_sp, x_sp, 1.0)
        # Diagonal entries (i==j) should be near zero
        diag = torch.diagonal(dist, dim1=-2, dim2=-1)
        assert diag.max() < 1e-4


class TestLorentzExpMap:
    def test_gradcheck(self):
        """Gradient check for exp map."""
        B, N, d = 2, 3, 4
        c = 1.0
        v = torch.randn(B, N, d, dtype=torch.float64).requires_grad_(True)

        def exp_map_sp(v, c):
            x_sp, x0 = LorentzExpMap.apply(v, c)
            return x_sp

        assert gradcheck(exp_map_sp, (v, c), eps=1e-4, atol=1e-3)

    def test_gradcheck_multihead(self):
        """Gradient check for [B, H, N, d] inputs."""
        B, H, N, d = 2, 2, 3, 4
        c = 1.0
        v = torch.randn(B, H, N, d, dtype=torch.float64).requires_grad_(True)

        def exp_map_sp(v, c):
            x_sp, x0 = LorentzExpMap.apply(v, c)
            return x_sp

        assert gradcheck(exp_map_sp, (v, c), eps=1e-4, atol=1e-3)

    def test_origin_output(self):
        """Zero tangent vector maps to origin."""
        v = torch.zeros(2, 3, 4, dtype=torch.float64)
        c = 1.0
        x_sp, x0 = LorentzExpMap.apply(v, c)
        assert x_sp.abs().max() < 1e-6
        assert (x0 - 1.0 / c ** 0.5).abs().max() < 1e-6

    def test_hyperboloid_constraint(self):
        """Output satisfies x0^2 - ||x_sp||^2 = 1/c."""
        v = torch.randn(4, 8, 5, dtype=torch.float64)
        c = 2.0
        x_sp, x0 = LorentzExpMap.apply(v, c)
        constraint = x0 ** 2 - (x_sp ** 2).sum(dim=-1) - 1.0 / c
        assert constraint.abs().max() < 1e-5


class TestLorentzLogMap:
    def test_gradcheck(self):
        """Gradient check for log map."""
        B, N, d = 2, 3, 4
        c = 1.0
        x_sp = _random_lorentz_spatial((B, N, d), c, r_range=(0.5, 3.0)).requires_grad_(True)
        assert gradcheck(LorentzLogMap.apply, (x_sp, c), eps=1e-4, atol=1e-3)

    def test_roundtrip(self):
        """exp(log(x)) ≈ x (roundtrip through log map then exp map)."""
        c = 1.0
        x_sp_orig = _random_lorentz_spatial((4, 8, 5), c, r_range=(1.0, 4.0))
        v = LorentzLogMap.apply(x_sp_orig, c)
        x_sp_recovered, _ = LorentzExpMap.apply(v, c)
        assert (x_sp_orig - x_sp_recovered).abs().max() < 1e-4

    def test_output_finite(self):
        """Log map produces finite outputs for valid hyperboloid points."""
        x_sp = _random_lorentz_spatial((8, 16, 5))
        v = LorentzLogMap.apply(x_sp, 1.0)
        assert torch.isfinite(v).all()
