"""Integration tests: 100-step micro-run with HCA Transformer."""

import pytest
import torch

from hca.training.harness import train_hca, HCATransformerLM
from hca.training.optimizer import build_optimizer


class TestMicroRun:
    @pytest.mark.slow
    def test_100_step_no_nan(self):
        """100-step micro-run completes without NaN/Inf."""
        metrics = train_hca(
            steps=100,
            d_model=64,
            n_heads=4,
            n_layers=2,
            seq_len=32,
            batch_size=4,
            use_wikitext=False,
            device="cpu",
            qos_cadence=50,
        )
        assert "kill_reason" not in metrics, f"Training killed: {metrics.get('kill_reason')}"
        assert all(torch.tensor(metrics["loss"]).isfinite()), "Non-finite loss detected"

    @pytest.mark.slow
    def test_loss_decreases(self):
        """Loss should decrease over 200 steps (learning signal present)."""
        metrics = train_hca(
            steps=200,
            d_model=64,
            n_heads=4,
            n_layers=2,
            seq_len=32,
            batch_size=8,
            lr=1e-3,
            use_wikitext=False,
            device="cpu",
            qos_cadence=100,
        )
        losses = metrics["loss"]
        # Compare average of first 20 steps to average of last 20 steps
        avg_first = sum(losses[:20]) / 20
        avg_last = sum(losses[-20:]) / 20
        assert avg_last < avg_first, (
            f"Loss did not decrease: avg_first_20={avg_first:.4f}, "
            f"avg_last_20={avg_last:.4f}"
        )

    @pytest.mark.slow
    def test_alpha_s_tracked(self):
        """alpha_s should be tracked and logged at QoS checkpoints."""
        metrics = train_hca(
            steps=100,
            d_model=64,
            n_heads=4,
            n_layers=2,
            seq_len=32,
            batch_size=4,
            use_wikitext=False,
            device="cpu",
            qos_cadence=50,
        )
        assert len(metrics["alpha_s"]) == 100, "alpha_s not tracked each step"
        # Check alpha_s values are in valid range
        for step_alphas in metrics["alpha_s"]:
            for a in step_alphas:
                assert 0 <= a <= 1, f"alpha_s={a} out of range"

    @pytest.mark.slow
    def test_qos_states_logged(self):
        """QoS state transitions should be logged."""
        metrics = train_hca(
            steps=100,
            d_model=64,
            n_heads=4,
            n_layers=2,
            seq_len=32,
            batch_size=4,
            use_wikitext=False,
            device="cpu",
            qos_cadence=50,
        )
        assert len(metrics["qos_state"]) >= 1, "No QoS states logged"


class TestModelConstruction:
    def test_model_creates(self):
        """HCATransformerLM can be instantiated."""
        model = HCATransformerLM(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        assert model is not None

    def test_forward_pass(self):
        """Forward pass produces correct output shapes."""
        model = HCATransformerLM(vocab_size=100, d_model=64, n_heads=4, n_layers=2, max_seq_len=32)
        ids = torch.randint(0, 100, (2, 16))
        logits, loss = model(ids, ids)
        assert logits.shape == (2, 16, 100)
        assert loss is not None
        assert torch.isfinite(loss)

    def test_optimizer_creates(self):
        """Hybrid optimizer can be built for HCA model."""
        model = HCATransformerLM(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        opt = build_optimizer(model)
        assert opt is not None
        assert len(opt.param_groups) == 2  # Euclidean + manifold

    def test_radial_reg_loss(self):
        """Radial regularization loss computes after forward pass."""
        model = HCATransformerLM(vocab_size=100, d_model=64, n_heads=4, n_layers=2, max_seq_len=32)
        ids = torch.randint(0, 100, (2, 16))
        model(ids)
        rad_loss = model.get_radial_reg_loss()
        assert torch.isfinite(rad_loss)
        assert rad_loss >= 0
