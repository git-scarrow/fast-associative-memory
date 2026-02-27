import pytest
import torch

from dynamic_vigilance import DynamicVigilance
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
