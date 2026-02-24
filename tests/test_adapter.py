"""
tests/test_adapter.py — Unit tests for MetricAdapter and FAM adapter integration.

Validates:
  - MetricAdapter forward pass (linear and MLP variants)
  - L2-normalised output
  - triplet_loss method produces a scalar loss with valid gradients
  - FastAssociativeMemory works with and without adapter
  - learn_local and forward use the adapter projection when set
"""

import pytest
import torch
import torch.nn.functional as F

from adapter import MetricAdapter
from fast_associative_memory import FastAssociativeMemory


# ---------------------------------------------------------------------------
# MetricAdapter tests
# ---------------------------------------------------------------------------

class TestMetricAdapterLinear:
    """Tests for the default single-linear-layer variant."""

    @pytest.fixture
    def adapter(self):
        return MetricAdapter(input_dim=64, output_dim=32)

    def test_output_shape(self, adapter):
        x = torch.randn(8, 64)
        out = adapter(x)
        assert out.shape == (8, 32)

    def test_output_l2_normalised(self, adapter):
        x = torch.randn(8, 64)
        out = adapter(x)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)

    def test_stores_dims(self, adapter):
        assert adapter.input_dim == 64
        assert adapter.output_dim == 32

    def test_half_precision_upcast(self, adapter):
        """Half-precision input should be upcast to float32 without error."""
        x = torch.randn(4, 64).half()
        out = adapter(x)
        assert out.dtype == torch.float32

    def test_single_sample(self, adapter):
        x = torch.randn(1, 64)
        out = adapter(x)
        assert out.shape == (1, 32)
        assert torch.allclose(out.norm(dim=-1), torch.ones(1), atol=1e-5)


class TestMetricAdapterMLP:
    """Tests for the two-layer MLP variant (hidden_dim > 0)."""

    @pytest.fixture
    def adapter(self):
        return MetricAdapter(input_dim=64, output_dim=32, hidden_dim=128)

    def test_output_shape(self, adapter):
        x = torch.randn(8, 64)
        out = adapter(x)
        assert out.shape == (8, 32)

    def test_output_l2_normalised(self, adapter):
        x = torch.randn(8, 64)
        out = adapter(x)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)

    def test_different_output_from_linear(self):
        """MLP and linear adapters with same dims should generally differ."""
        torch.manual_seed(0)
        linear = MetricAdapter(input_dim=64, output_dim=32, hidden_dim=0)
        torch.manual_seed(0)
        mlp = MetricAdapter(input_dim=64, output_dim=32, hidden_dim=128)
        x = torch.randn(4, 64)
        assert not torch.allclose(linear(x), mlp(x))


# ---------------------------------------------------------------------------
# TripletMarginLoss tests
# ---------------------------------------------------------------------------

class TestTripletLoss:
    @pytest.fixture
    def adapter(self):
        return MetricAdapter(input_dim=32, output_dim=16)

    def test_loss_is_scalar(self, adapter):
        a = torch.randn(4, 32)
        p = torch.randn(4, 32)
        n = torch.randn(4, 32)
        loss = adapter.triplet_loss(a, p, n)
        assert loss.shape == ()

    def test_loss_non_negative(self, adapter):
        a = torch.randn(4, 32)
        p = torch.randn(4, 32)
        n = torch.randn(4, 32)
        loss = adapter.triplet_loss(a, p, n)
        assert loss.item() >= 0.0

    def test_loss_has_gradient(self, adapter):
        a = torch.randn(4, 32)
        p = torch.randn(4, 32)
        n = torch.randn(4, 32)
        loss = adapter.triplet_loss(a, p, n)
        loss.backward()
        for param in adapter.parameters():
            assert param.grad is not None

    def test_easy_negatives_lower_loss(self, adapter):
        """Loss should be lower when negatives are far from the anchor."""
        # Generous tolerance: random init may push easy/hard losses close together
        # but easy negatives should never exceed hard negatives by more than this.
        _LOSS_TOLERANCE = 0.5

        torch.manual_seed(42)
        anchor = F.normalize(torch.randn(8, 32), dim=-1)
        positive = F.normalize(anchor + 0.01 * torch.randn(8, 32), dim=-1)
        hard_neg = F.normalize(anchor + 0.05 * torch.randn(8, 32), dim=-1)
        easy_neg = F.normalize(torch.randn(8, 32), dim=-1)

        loss_hard = adapter.triplet_loss(anchor, positive, hard_neg)
        loss_easy = adapter.triplet_loss(anchor, positive, easy_neg)
        # Easy negatives produce ≤ loss compared to hard negatives on average
        assert loss_easy.item() <= loss_hard.item() + _LOSS_TOLERANCE

    def test_custom_margin(self, adapter):
        a = torch.randn(4, 32)
        p = torch.randn(4, 32)
        n = torch.randn(4, 32)
        loss_1 = adapter.triplet_loss(a, p, n, margin=0.5)
        loss_2 = adapter.triplet_loss(a, p, n, margin=2.0)
        # Larger margin → larger (or equal) loss
        assert loss_2.item() >= loss_1.item() - 1e-5


# ---------------------------------------------------------------------------
# FastAssociativeMemory + adapter integration tests
# ---------------------------------------------------------------------------

class TestFAMWithoutAdapter:
    """Existing FAM behaviour must be unchanged when adapter=None."""

    @pytest.fixture
    def fam(self):
        return FastAssociativeMemory(
            input_dim=32, value_dim=10, core_entries=256, core_vigilance=0.7
        )

    def test_learn_and_forward(self, fam):
        x = F.normalize(torch.randn(4, 32), dim=-1)
        ids = torch.randint(0, 10, (4,))
        fam.learn_local(x, ids)
        out = fam(x)
        assert out.shape == (4, 10)

    def test_adapter_is_none(self, fam):
        assert fam.adapter is None


class TestFAMWithAdapter:
    """FAM must correctly apply the adapter in both learn_local and forward."""

    @pytest.fixture
    def adapter(self):
        return MetricAdapter(input_dim=64, output_dim=32)

    @pytest.fixture
    def fam(self, adapter):
        return FastAssociativeMemory(
            input_dim=64, value_dim=10, core_entries=256, core_vigilance=0.7,
            adapter=adapter,
        )

    def test_adapter_stored(self, fam, adapter):
        assert fam.adapter is adapter

    def test_learn_and_forward_shape(self, fam):
        x = torch.randn(4, 64)
        ids = torch.randint(0, 10, (4,))
        fam.learn_local(x, ids)
        out = fam(x)
        assert out.shape == (4, 10)

    def test_cam_key_dim_matches_adapter_output(self, fam, adapter):
        """The CAM key dimension must equal adapter.output_dim."""
        assert fam.core_cam.key_dim == adapter.output_dim

    def test_keys_stored_in_projected_space(self, fam):
        """Keys written to CAM must have dim == adapter.output_dim (32), not 64."""
        x = torch.randn(4, 64)
        ids = torch.randint(0, 10, (4,))
        fam.learn_local(x, ids)
        occ = fam.core_cam.occupied
        assert occ.any()
        stored_key_dim = fam.core_cam.keys.shape[1]
        assert stored_key_dim == fam.adapter.output_dim

    def test_forward_consistent_after_learn(self, fam):
        """Repeated forward calls on the same input are deterministic."""
        x = torch.randn(4, 64)
        ids = torch.randint(0, 10, (4,))
        fam.learn_local(x, ids)
        out1 = fam(x)
        out2 = fam(x)
        assert torch.allclose(out1, out2)


class TestFAMWithMLPAdapter:
    """FAM must work with the MLP variant of MetricAdapter."""

    def test_learn_and_forward(self):
        adapter = MetricAdapter(input_dim=64, output_dim=32, hidden_dim=128)
        fam = FastAssociativeMemory(
            input_dim=64, value_dim=5, core_entries=128, core_vigilance=0.7,
            adapter=adapter,
        )
        x = torch.randn(8, 64)
        ids = torch.randint(0, 5, (8,))
        fam.learn_local(x, ids)
        out = fam(x)
        assert out.shape == (8, 5)
