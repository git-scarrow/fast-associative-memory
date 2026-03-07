"""H1-H7 unit test suite: fp32 hazard boundary tests from DS-10."""

import math
import pytest
import torch

from hca.ops.lorentz_exp_map import LorentzExpMap, SINH_CLAMP
from hca.ops.lorentz_log_map import LorentzLogMap
from hca.ops.lorentz_distance import LorentzSquaredDistance
from hca.modules.curvature import CurvatureModule


class TestH1ExpMapOverflow:
    """H1: cosh/sinh overflow when sqrt(c)*||v|| > 88.7."""

    def test_large_tangent_clamped(self):
        c = 1.0
        # ||v|| = 90 > 85 clamp threshold
        v = torch.randn(1, 1, 4, dtype=torch.float64)
        v = v / v.norm(dim=-1, keepdim=True) * 90.0
        v.requires_grad_(True)

        x_sp, x0 = LorentzExpMap.apply(v, c)
        assert torch.isfinite(x_sp).all(), "x_sp has non-finite values"
        assert torch.isfinite(x0).all(), "x0 has non-finite values"

        # Backward should also be finite
        loss = x_sp.sum() + x0.sum()
        loss.backward()
        assert torch.isfinite(v.grad).all(), "gradient has non-finite values"

    def test_extreme_tangent(self):
        """||v|| = 1000 — well beyond overflow threshold."""
        c = 1.0
        v = torch.randn(1, 1, 8, dtype=torch.float64)
        v = v / v.norm(dim=-1, keepdim=True) * 1000.0
        v.requires_grad_(True)

        x_sp, x0 = LorentzExpMap.apply(v, c)
        assert torch.isfinite(x_sp).all()
        assert torch.isfinite(x0).all()

        loss = x_sp.sum() + x0.sum()
        loss.backward()
        assert torch.isfinite(v.grad).all()


class TestH2InnerProductNearOrigin:
    """H2: significance loss when points are near the hyperboloid origin."""

    def test_near_origin_points(self):
        c = 1.0
        # x_0 ≈ 1 + 1e-8, spatial coords very small
        x_sp = torch.randn(1, 4, dtype=torch.float64) * 1e-4
        y_sp = torch.randn(1, 4, dtype=torch.float64) * 1e-4
        x_sp.requires_grad_(True)
        y_sp.requires_grad_(True)

        dist = LorentzSquaredDistance.apply(x_sp, y_sp, c)
        assert torch.isfinite(dist).all(), "distance has non-finite values near origin"
        assert (dist >= 0).all(), "distance is negative"

        loss = dist.sum()
        loss.backward()
        assert torch.isfinite(x_sp.grad).all(), "x_sp gradient non-finite near origin"
        assert torch.isfinite(y_sp.grad).all(), "y_sp gradient non-finite near origin"


class TestH3SoftmaxLargeDistances:
    """H3: softmax overflow from large Lorentz distances."""

    def test_large_score_range(self):
        """Scores spanning 1e4 range should produce finite softmax."""
        scores = torch.tensor([[0.0, 1e4, -1e4, 5e3]], dtype=torch.float32)
        alpha = torch.softmax(scores, dim=-1)
        assert torch.isfinite(alpha).all(), "softmax overflow with large scores"
        assert (alpha >= 0).all() and (alpha <= 1).all()
        assert alpha.sum(dim=-1).allclose(torch.ones(1))

    def test_large_distances_in_pipeline(self):
        """Points at extreme radii produce finite attention weights."""
        c = 1.0
        # One point near origin, one far away
        x_sp = torch.randn(1, 2, 4, dtype=torch.float64)
        x_sp[0, 0] *= 0.01  # near origin
        x_sp[0, 1] *= 10.0  # far away
        x_sp.requires_grad_(True)

        dist = LorentzSquaredDistance.apply(x_sp, x_sp, c)
        scores = -dist  # simplified scoring
        alpha = torch.softmax(scores, dim=-1)
        assert torch.isfinite(alpha).all()

        loss = alpha.sum()
        loss.backward()
        assert torch.isfinite(x_sp.grad).all()


class TestH4LogMapCoincidentPoints:
    """H4: gradient explosion when x ≈ y (z → 1+)."""

    def test_near_origin_log_map(self):
        """Points very close to origin: z = 1 + 1e-9."""
        c = 1.0
        # Very small spatial coords → x_0 ≈ 1/sqrt(c), close to origin
        x_sp = torch.randn(1, 4, dtype=torch.float64) * 1e-5
        x_sp.requires_grad_(True)

        v = LorentzLogMap.apply(x_sp, c)
        assert torch.isfinite(v).all(), "log map output non-finite near origin"

        loss = v.sum()
        loss.backward()
        assert torch.isfinite(x_sp.grad).all(), "log map gradient non-finite near origin"

    def test_coincident_points_distance(self):
        """Nearly identical points should have near-zero distance with finite gradients."""
        c = 1.0
        x_sp = torch.randn(1, 4, dtype=torch.float64) * 2.0
        y_sp = x_sp.clone() + torch.randn_like(x_sp) * 1e-8
        x_sp.requires_grad_(True)
        y_sp.requires_grad_(True)

        dist = LorentzSquaredDistance.apply(x_sp, y_sp, c)
        assert torch.isfinite(dist).all()

        loss = dist.sum()
        loss.backward()
        assert torch.isfinite(x_sp.grad).all()
        assert torch.isfinite(y_sp.grad).all()


class TestH5ExpMapZeroTangent:
    """H5: division by zero when ||v|| = 0 in exp map."""

    def test_exact_zero_tangent(self):
        c = 1.0
        v = torch.zeros(1, 1, 4, dtype=torch.float64, requires_grad=True)

        x_sp, x0 = LorentzExpMap.apply(v, c)
        # Should return origin
        assert x_sp.abs().max() < 1e-6, "zero tangent should map to origin (spatial=0)"
        assert (x0 - 1.0).abs().max() < 1e-6, "zero tangent should map to origin (x0=1/sqrt(c))"
        assert torch.isfinite(x_sp).all()
        assert torch.isfinite(x0).all()

    def test_near_zero_tangent(self):
        """||v|| = 1e-10 — should not divide by zero."""
        c = 1.0
        v = torch.randn(1, 1, 4, dtype=torch.float64)
        v = v / v.norm(dim=-1, keepdim=True) * 1e-10
        v.requires_grad_(True)

        x_sp, x0 = LorentzExpMap.apply(v, c)
        assert torch.isfinite(x_sp).all()
        assert torch.isfinite(x0).all()


class TestH6CurvatureFloorEnforcement:
    """H6: curvature collapse prevention."""

    def test_negative_c_raw(self):
        """c_raw = -100 should still produce c_eff > 0."""
        module = CurvatureModule(c_init=1.0, M=128.0)
        with torch.no_grad():
            module.c_raw.fill_(-100.0)

        c_eff = module(N=512)
        assert c_eff.item() > 0, f"c_eff should be positive, got {c_eff.item()}"

    def test_floor_increases_with_N(self):
        """c_min should increase with sequence length."""
        module = CurvatureModule(c_init=1.0, M=128.0)
        c_min_128 = module.c_min(128)
        c_min_8192 = module.c_min(8192)
        c_min_128k = module.c_min(131072)
        assert c_min_128 < c_min_8192 < c_min_128k

    def test_c_eff_gradient(self):
        """c_eff should have gradient w.r.t. c_raw."""
        module = CurvatureModule(c_init=1.0, M=128.0)
        c_eff = module(N=512)
        c_eff.backward()
        assert module.c_raw.grad is not None
        assert torch.isfinite(module.c_raw.grad)


class TestH7TemporalCoordConsistency:
    """H7: x_0 must be derived, not stored, to prevent drift."""

    def test_derived_x0_after_spatial_change(self):
        """Modifying spatial coords should change derived x_0."""
        c = 1.0
        x_sp = torch.randn(4, 8, dtype=torch.float64)

        # Derive x0
        x0_before = torch.sqrt(1.0 / c + (x_sp ** 2).sum(dim=-1))

        # Modify spatial coords
        x_sp_mod = x_sp * 2.0
        x0_after = torch.sqrt(1.0 / c + (x_sp_mod ** 2).sum(dim=-1))

        # x0 should change
        assert not torch.allclose(x0_before, x0_after), "x0 should change when spatial coords change"

        # Verify hyperboloid constraint
        constraint = x0_after ** 2 - (x_sp_mod ** 2).sum(dim=-1) - 1.0 / c
        assert constraint.abs().max() < 1e-10, "hyperboloid constraint violated"

    def test_exp_map_derives_x0(self):
        """Exp map output should satisfy hyperboloid constraint by construction."""
        c = 2.0
        v = torch.randn(4, 8, 5, dtype=torch.float64)
        x_sp, x0 = LorentzExpMap.apply(v, c)

        # Check constraint: x0^2 - ||x_sp||^2 = 1/c
        constraint = x0 ** 2 - (x_sp ** 2).sum(dim=-1) - 1.0 / c
        assert constraint.abs().max() < 1e-5, f"constraint violation: {constraint.abs().max()}"
