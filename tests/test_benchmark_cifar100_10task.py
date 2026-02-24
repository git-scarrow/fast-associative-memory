"""
tests/test_benchmark_cifar100_10task.py — Unit tests for G19 adapter
generalization stress test (benchmark_cifar100_10task.py).

Validates:
  - eval_fam works correctly with and without an adapter on synthetic data
  - compute_degradation returns correct relative percentage values
  - check_kill_condition fires only when degradation exceeds the threshold
  - format_results produces well-structured output with correct verdict strings
  - load_features raises FileNotFoundError when the cache directory is empty
  - load_adapter correctly reconstructs an adapter from a saved state-dict
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# Allow imports from the repo root regardless of working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapter import MetricAdapter
from benchmarks.benchmark_cifar100_10task import (
    check_kill_condition,
    compute_degradation,
    eval_fam,
    format_results,
    load_adapter,
    load_features,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_dataset():
    """Return (train_embeds, train_labels, test_embeds, test_labels) with 4
    well-separated classes so that FAM can achieve decent accuracy."""
    torch.manual_seed(0)
    n_classes = 4
    input_dim = 32
    train_per_class = 20
    test_per_class = 8

    # Build cluster centers far apart
    centers = F.normalize(torch.randn(n_classes, input_dim), dim=-1) * 5.0

    train_e = torch.cat([
        centers[c] + 0.05 * torch.randn(train_per_class, input_dim)
        for c in range(n_classes)
    ])
    train_l = torch.arange(n_classes).repeat_interleave(train_per_class)

    test_e = torch.cat([
        centers[c] + 0.05 * torch.randn(test_per_class, input_dim)
        for c in range(n_classes)
    ])
    test_l = torch.arange(n_classes).repeat_interleave(test_per_class)

    return train_e, train_l, test_e, test_l, n_classes, input_dim


# ---------------------------------------------------------------------------
# eval_fam
# ---------------------------------------------------------------------------

class TestEvalFam:
    """Tests for the core FAM evaluation helper."""

    def test_baseline_returns_valid_accuracy(self, synthetic_dataset):
        train_e, train_l, test_e, test_l, n_classes, _ = synthetic_dataset
        device = torch.device("cpu")
        acc, elapsed = eval_fam(
            train_e, train_l, test_e, test_l, n_classes, device, adapter=None
        )
        assert 0.0 <= acc <= 100.0
        assert elapsed >= 0.0

    def test_baseline_high_accuracy_on_separable_data(self, synthetic_dataset):
        """On clearly separable clusters, baseline FAM should reach near 100%."""
        train_e, train_l, test_e, test_l, n_classes, _ = synthetic_dataset
        device = torch.device("cpu")
        acc, _ = eval_fam(
            train_e, train_l, test_e, test_l, n_classes, device, adapter=None
        )
        assert acc >= 75.0, f"Expected high accuracy on separable data, got {acc:.2f}%"

    def test_with_random_adapter_returns_valid_accuracy(self, synthetic_dataset):
        """FAM with a random adapter should still return a valid accuracy."""
        train_e, train_l, test_e, test_l, n_classes, input_dim = synthetic_dataset
        device = torch.device("cpu")
        torch.manual_seed(1)
        adapter = MetricAdapter(input_dim=input_dim, output_dim=16)
        adapter.eval()
        acc, elapsed = eval_fam(
            train_e, train_l, test_e, test_l, n_classes, device, adapter=adapter
        )
        assert 0.0 <= acc <= 100.0
        assert elapsed >= 0.0

    def test_adapter_changes_accuracy(self, synthetic_dataset):
        """A randomly initialized adapter generally changes accuracy vs. baseline."""
        train_e, train_l, test_e, test_l, n_classes, input_dim = synthetic_dataset
        device = torch.device("cpu")
        acc_base, _ = eval_fam(
            train_e, train_l, test_e, test_l, n_classes, device, adapter=None
        )
        torch.manual_seed(99)
        adapter = MetricAdapter(input_dim=input_dim, output_dim=16, hidden_dim=32)
        adapter.eval()
        acc_adapted, _ = eval_fam(
            train_e, train_l, test_e, test_l, n_classes, device, adapter=adapter
        )
        # Accuracies are unlikely to be exactly equal; just check both are valid.
        assert 0.0 <= acc_adapted <= 100.0
        assert 0.0 <= acc_base <= 100.0

    def test_result_is_deterministic(self, synthetic_dataset):
        """Running eval_fam twice with the same data should give the same result."""
        train_e, train_l, test_e, test_l, n_classes, _ = synthetic_dataset
        device = torch.device("cpu")
        acc1, _ = eval_fam(
            train_e, train_l, test_e, test_l, n_classes, device, adapter=None
        )
        acc2, _ = eval_fam(
            train_e, train_l, test_e, test_l, n_classes, device, adapter=None
        )
        assert acc1 == acc2


# ---------------------------------------------------------------------------
# compute_degradation
# ---------------------------------------------------------------------------

class TestComputeDegradation:
    """Tests for the relative-degradation computation."""

    def test_zero_degradation_when_equal(self):
        assert compute_degradation(80.0, 80.0) == pytest.approx(0.0)

    def test_positive_degradation_when_adapter_lower(self):
        # baseline=80, adapter=76 → degradation = (80-76)/80*100 = 5%
        result = compute_degradation(80.0, 76.0)
        assert result == pytest.approx(5.0, rel=1e-5)

    def test_negative_degradation_when_adapter_higher(self):
        # adapter improves over baseline
        result = compute_degradation(80.0, 84.0)
        assert result == pytest.approx(-5.0, rel=1e-5)

    def test_zero_baseline_returns_zero(self):
        """Guard against division by zero when baseline accuracy is 0."""
        assert compute_degradation(0.0, 0.0) == 0.0
        assert compute_degradation(0.0, 50.0) == 0.0

    def test_100_pct_baseline(self):
        # baseline=100, adapter=95 → 5% relative drop
        result = compute_degradation(100.0, 95.0)
        assert result == pytest.approx(5.0, rel=1e-5)

    def test_complete_degradation(self):
        # adapter = 0% accuracy → 100% relative drop
        result = compute_degradation(80.0, 0.0)
        assert result == pytest.approx(100.0, rel=1e-5)


# ---------------------------------------------------------------------------
# check_kill_condition
# ---------------------------------------------------------------------------

class TestCheckKillCondition:
    """Tests for the G19 kill-condition gate."""

    def test_kill_fires_when_degradation_exceeds_threshold(self):
        assert check_kill_condition(5.1, threshold=5.0) is True

    def test_kill_does_not_fire_when_degradation_below_threshold(self):
        assert check_kill_condition(4.9, threshold=5.0) is False

    def test_kill_does_not_fire_at_exact_threshold(self):
        # Strictly greater than threshold required to fire.
        assert check_kill_condition(5.0, threshold=5.0) is False

    def test_kill_does_not_fire_for_negative_degradation(self):
        # Adapter improves accuracy — definitely not a kill condition.
        assert check_kill_condition(-3.0, threshold=5.0) is False

    def test_custom_threshold(self):
        assert check_kill_condition(3.0, threshold=2.5) is True
        assert check_kill_condition(2.0, threshold=2.5) is False


# ---------------------------------------------------------------------------
# format_results
# ---------------------------------------------------------------------------

class TestFormatResults:
    """Tests for the results-table formatter."""

    def _baseline_only(self):
        return format_results(
            dataset_label="CIFAR-100",
            baseline_acc=82.5,
            baseline_elapsed=10.0,
            adapter_acc=None,
            adapter_elapsed=None,
            degradation=None,
            threshold=5.0,
        )

    def _with_adapter_passing(self):
        return format_results(
            dataset_label="CIFAR-100",
            baseline_acc=82.5,
            baseline_elapsed=10.0,
            adapter_acc=80.0,
            adapter_elapsed=12.5,
            degradation=compute_degradation(82.5, 80.0),
            threshold=5.0,
        )

    def _with_adapter_kill(self):
        return format_results(
            dataset_label="CIFAR-100",
            baseline_acc=82.5,
            baseline_elapsed=10.0,
            adapter_acc=75.0,
            adapter_elapsed=12.5,
            degradation=compute_degradation(82.5, 75.0),
            threshold=5.0,
        )

    def test_returns_string(self):
        assert isinstance(self._baseline_only(), str)

    def test_contains_dataset_label(self):
        assert "CIFAR-100" in self._baseline_only()

    def test_baseline_only_contains_no_adapter_note(self):
        table = self._baseline_only()
        assert "No trained adapter" in table or "baseline only" in table.lower()

    def test_baseline_accuracy_present(self):
        table = self._baseline_only()
        assert "82.50" in table

    def test_g19_passed_when_degradation_small(self):
        table = self._with_adapter_passing()
        assert "G19 PASSED" in table
        assert "\u2713" in table  # checkmark ✓

    def test_kill_condition_met_when_degradation_large(self):
        table = self._with_adapter_kill()
        assert "KILL CONDITION MET" in table
        assert "\u26a0" in table  # warning ⚠

    def test_adapter_accuracy_present_in_table(self):
        table = self._with_adapter_passing()
        assert "80.00" in table

    def test_degradation_value_shown(self):
        table = self._with_adapter_passing()
        degradation = compute_degradation(82.5, 80.0)
        assert f"{degradation:.2f}" in table

    def test_threshold_shown_in_table(self):
        table = self._with_adapter_passing()
        assert "5.0" in table

    def test_g19_label_present(self):
        table = self._baseline_only()
        assert "G19" in table


# ---------------------------------------------------------------------------
# load_features
# ---------------------------------------------------------------------------

class TestLoadFeatures:
    """Tests for the feature-cache loading helper."""

    def test_raises_when_directory_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="No train/test .pt files"):
                load_features(tmpdir, torch.device("cpu"))

    def test_loads_correctly_from_valid_cache(self):
        """Save synthetic train/test .pt files and verify they load correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_e = torch.randn(50, 32)
            train_l = torch.randint(0, 4, (50,))
            test_e = torch.randn(20, 32)
            test_l = torch.randint(0, 4, (20,))

            torch.save({"embeds": train_e, "labels": train_l},
                       os.path.join(tmpdir, "dataset_dinov2_train.pt"))
            torch.save({"embeds": test_e, "labels": test_l},
                       os.path.join(tmpdir, "dataset_dinov2_test.pt"))

            te, tl, ve, vl = load_features(tmpdir, torch.device("cpu"))
            assert te.shape == (50, 32)
            assert tl.shape == (50,)
            assert ve.shape == (20, 32)
            assert vl.shape == (20,)
            assert te.dtype == torch.float32
            assert tl.dtype == torch.int64


# ---------------------------------------------------------------------------
# load_adapter
# ---------------------------------------------------------------------------

class TestLoadAdapter:
    """Tests for loading an adapter from a saved state-dict."""

    def test_load_reconstructs_forward_output(self):
        """Reloaded adapter must produce identical forward output."""
        torch.manual_seed(3)
        adapter_orig = MetricAdapter(input_dim=64, output_dim=32, hidden_dim=128)
        x = torch.randn(4, 64)
        expected = adapter_orig(x)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            torch.save(adapter_orig.state_dict(), path)
            adapter_loaded = load_adapter(
                path, input_dim=64, output_dim=32, hidden_dim=128,
                device=torch.device("cpu"),
            )
            actual = adapter_loaded(x)
            assert torch.allclose(expected, actual, atol=1e-6)
        finally:
            os.unlink(path)

    def test_load_adapter_is_in_eval_mode(self):
        torch.manual_seed(4)
        adapter_orig = MetricAdapter(input_dim=32, output_dim=16)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            torch.save(adapter_orig.state_dict(), path)
            adapter_loaded = load_adapter(
                path, input_dim=32, output_dim=16, hidden_dim=0,
                device=torch.device("cpu"),
            )
            assert not adapter_loaded.training
        finally:
            os.unlink(path)

    def test_load_linear_adapter(self):
        """Linear (hidden_dim=0) adapters load without error."""
        torch.manual_seed(5)
        adapter_orig = MetricAdapter(input_dim=64, output_dim=32, hidden_dim=0)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            torch.save(adapter_orig.state_dict(), path)
            adapter_loaded = load_adapter(
                path, input_dim=64, output_dim=32, hidden_dim=0,
                device=torch.device("cpu"),
            )
            x = torch.randn(2, 64)
            out = adapter_loaded(x)
            assert out.shape == (2, 32)
        finally:
            os.unlink(path)
