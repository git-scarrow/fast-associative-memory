import pytest
import torch

from dynamic_vigilance import DynamicVigilance, RelativeVigilance
from associative_core import ContinuousCAM


def test_zero_margin_yields_base_vigilance():
    sims = torch.tensor([[0.8, 0.8]])  # best and strongest other equal
    labels = torch.tensor([0, 1])
    dv = DynamicVigilance(v_base=0.92, alpha=0.3, v_floor=0.30, v_ceiling=0.95)

    v_eff, margins = dv.compute(sims, labels)

    assert torch.allclose(margins, torch.zeros_like(margins))
    assert torch.allclose(v_eff, torch.full_like(v_eff, dv.v_base))


def test_small_margin_keeps_vigilance_near_base():
    sims = torch.tensor([[0.90, 0.89]])
    labels = torch.tensor([0, 1])
    dv = DynamicVigilance(v_base=0.92, alpha=0.3, v_floor=0.30, v_ceiling=0.95)

    v_eff, margins = dv.compute(sims, labels)
    expected_margin = torch.tensor([0.01])
    expected_v = dv.v_base - dv.alpha * expected_margin

    assert torch.allclose(margins, expected_margin, atol=1e-6)
    assert torch.allclose(v_eff, expected_v, atol=1e-6)


def test_large_margin_drives_vigilance_towards_floor():
    # Use exaggerated similarities to force a very large margin and test clamping
    sims = torch.tensor([[1.0, -2.0]])
    labels = torch.tensor([0, 1])
    dv = DynamicVigilance(v_base=0.92, alpha=0.3, v_floor=0.30, v_ceiling=0.95)

    v_eff, margins = dv.compute(sims, labels)

    assert torch.all(v_eff <= dv.v_base)
    # With such a large margin the unclamped value would fall below v_floor
    assert torch.allclose(v_eff, torch.full_like(v_eff, dv.v_floor))


def test_clamping_respects_ceiling():
    sims = torch.tensor([[0.5, 0.4]])
    labels = torch.tensor([0, 1])
    # Set v_base above ceiling and alpha=0 so margins do not matter
    dv = DynamicVigilance(v_base=1.0, alpha=0.0, v_floor=0.30, v_ceiling=0.95)

    v_eff, _ = dv.compute(sims, labels)
    assert torch.allclose(v_eff, torch.full_like(v_eff, dv.v_ceiling))


def test_backward_compat_when_dynamic_vigilance_none():
    cam = ContinuousCAM(key_dim=4, value_dim=3, max_entries=16, vigilance=0.85)
    x = torch.randn(8, 4)
    y = torch.nn.functional.one_hot(torch.randint(0, 3, (8,)), num_classes=3).float()

    cam.learn_local(x, y)

    # No dynamic vigilance was provided, so stats should remain at defaults
    stats = cam.get_stats()
    assert stats["mean_v_effective"] == pytest.approx(cam.vigilance)
    assert stats["mean_margin"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# RelativeVigilance tests
# ---------------------------------------------------------------------------

def test_relative_vigilance_output_shape_and_range():
    rv = RelativeVigilance(v_floor=0.30, v_ceiling=0.95, margin_guard=0.05, percentile=0.95)
    # Two-class setup: proto 0 is class 0, proto 1 is class 1
    sims = torch.tensor([[0.80, 0.40], [0.45, 0.85]])
    labels = torch.tensor([0, 1])

    v_eff, margins = rv.compute(sims, labels)

    assert v_eff.shape == (2,)
    assert margins.shape == (2,)
    assert (v_eff >= rv.v_floor).all()
    assert (v_eff <= rv.v_ceiling).all()


def test_relative_vigilance_anchors_to_percentile():
    rv = RelativeVigilance(v_floor=0.30, v_ceiling=0.95, margin_guard=0.05, percentile=1.0)
    # Off-class sims for both queries: 0.40 and 0.45; p100 = 0.45
    sims = torch.tensor([[0.80, 0.40], [0.45, 0.85]])
    labels = torch.tensor([0, 1])

    v_eff, _ = rv.compute(sims, labels)

    # p100 of [0.40, 0.45] = 0.45; v_eff = clamp(0.45 + 0.05, 0.30, 0.95) = 0.50
    expected = max(rv.v_floor, min(rv.v_ceiling, 0.45 + rv.margin_guard))
    assert v_eff[0].item() == pytest.approx(expected, abs=1e-5)
    assert v_eff[1].item() == pytest.approx(expected, abs=1e-5)


def test_relative_vigilance_floor_clamp():
    # margin_guard negative enough to push below floor
    rv = RelativeVigilance(v_floor=0.50, v_ceiling=0.95, margin_guard=-0.40, percentile=0.50)
    sims = torch.tensor([[0.80, 0.30]])
    labels = torch.tensor([0, 1])

    v_eff, _ = rv.compute(sims, labels)
    assert v_eff[0].item() == pytest.approx(rv.v_floor, abs=1e-5)


def test_relative_vigilance_ceiling_clamp():
    rv = RelativeVigilance(v_floor=0.30, v_ceiling=0.60, margin_guard=0.20, percentile=1.0)
    # Off-class sim is 0.70; 0.70 + 0.20 > v_ceiling → clamp to 0.60
    sims = torch.tensor([[0.90, 0.70]])
    labels = torch.tensor([0, 1])

    v_eff, _ = rv.compute(sims, labels)
    assert v_eff[0].item() == pytest.approx(rv.v_ceiling, abs=1e-5)


def test_relative_vigilance_ema_smoothing():
    rv = RelativeVigilance(v_floor=0.30, v_ceiling=0.95, margin_guard=0.0,
                           percentile=1.0, ema_beta=0.5)
    sims_a = torch.tensor([[0.80, 0.40]])
    sims_b = torch.tensor([[0.80, 0.60]])
    labels = torch.tensor([0, 1])

    rv.compute(sims_a, labels)                    # _rho_ema = 0.40
    assert rv._rho_ema == pytest.approx(0.40, abs=1e-5)

    rv.compute(sims_b, labels)                    # _rho_ema = 0.5*0.40 + 0.5*0.60 = 0.50
    assert rv._rho_ema == pytest.approx(0.50, abs=1e-5)


def test_relative_vigilance_no_cross_class_competitors():
    # All prototypes are same class → sim_other hits -1e9 sentinel; fallback applies.
    rv = RelativeVigilance(v_floor=0.30, v_ceiling=0.95, margin_guard=0.05, percentile=0.95)
    sims = torch.tensor([[0.90, 0.85, 0.80]])
    labels = torch.tensor([0, 0, 0])

    v_eff, margins = rv.compute(sims, labels)
    assert v_eff.shape == (1,)
    assert (v_eff >= rv.v_floor).all()
    assert (v_eff <= rv.v_ceiling).all()


def test_cam_with_relative_vigilance():
    rv = RelativeVigilance(v_floor=0.30, v_ceiling=0.95, margin_guard=0.05)
    cam = ContinuousCAM(key_dim=8, value_dim=4, max_entries=64, dynamic_vigilance=rv)

    x = torch.randn(16, 8)
    y = torch.nn.functional.one_hot(torch.randint(0, 4, (16,)), num_classes=4).float()
    cam.learn_local(x, y)

    stats = cam.get_stats()
    assert 0.0 <= stats["mean_v_effective"] <= 1.0
