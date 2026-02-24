"""
tests/test_gauntlet_19_cross_domain.py — Unit tests for Gauntlet 19.

Validates the G19 cross-domain regression benchmark:
  - accuracy_relative_drop computes the correct percentage-point difference
  - check_kill_condition triggers only when drop > threshold
  - _format_table produces a well-formed string with the expected markers
  - eval_fam_accuracy evaluates correctly on synthetic data (no adapter)
  - An identity-like adapter introduces ≤ 5 pp accuracy drop (pass)
  - A heavily distorting adapter triggers the kill condition (fail)
  - load_features raises FileNotFoundError for an empty directory
"""

import os
import sys
import tempfile

import pytest
import torch
import torch.nn.functional as F

# Allow importing from the repo root regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.gauntlet_19_cross_domain import (
    accuracy_relative_drop,
    check_kill_condition,
    eval_fam_accuracy,
    load_features,
    _format_table,
)
from adapter import MetricAdapter


# ─────────────────────────────────────────────────── fixtures & helpers ───

_EMBED_DIM = 32
_NUM_CLASSES = 5
_SAMPLES_PER_CLASS = 40
_TOTAL_TRAIN = _NUM_CLASSES * _SAMPLES_PER_CLASS
_TOTAL_TEST = _NUM_CLASSES * 10


def _make_generic_data(seed: int = 0):
    """Return (train_embeds, train_labels, test_embeds, test_labels) for a
    well-separated synthetic generic-task dataset."""
    torch.manual_seed(seed)
    # Orthogonal class centers scaled to lie far apart on the hypersphere
    centers = F.normalize(torch.eye(_NUM_CLASSES, _EMBED_DIM), dim=-1) * 5.0

    def _cluster(center, n, noise=0.05):
        return center + noise * torch.randn(n, _EMBED_DIM)

    train_e = torch.cat([_cluster(centers[c], _SAMPLES_PER_CLASS) for c in range(_NUM_CLASSES)])
    train_l = torch.arange(_NUM_CLASSES).repeat_interleave(_SAMPLES_PER_CLASS)

    test_e = torch.cat([_cluster(centers[c], 10) for c in range(_NUM_CLASSES)])
    test_l = torch.arange(_NUM_CLASSES).repeat_interleave(10)

    # Shuffle training set
    perm = torch.randperm(len(train_e))
    return train_e[perm], train_l[perm], test_e, test_l


def _identity_adapter(embed_dim: int = _EMBED_DIM) -> MetricAdapter:
    """Return an adapter whose linear projection is initialised to the identity
    matrix (no structural distortion, only L2-normalisation)."""
    adapter = MetricAdapter(input_dim=embed_dim, output_dim=embed_dim)
    with torch.no_grad():
        adapter.net.weight.copy_(torch.eye(embed_dim))
        adapter.net.bias.zero_()
    return adapter


def _distorting_adapter(embed_dim: int = _EMBED_DIM, proj_dim: int = 4) -> MetricAdapter:
    """Return an adapter that projects to a much lower dimension, collapsing
    most of the discriminative structure and producing large accuracy drops."""
    torch.manual_seed(99)
    return MetricAdapter(input_dim=embed_dim, output_dim=proj_dim)


# ──────────────────────────────────────── accuracy_relative_drop tests ───

class TestAccuracyRelativeDrop:
    def test_no_drop(self):
        assert accuracy_relative_drop(75.0, 75.0) == pytest.approx(0.0)

    def test_positive_drop(self):
        assert accuracy_relative_drop(75.0, 70.0) == pytest.approx(5.0)

    def test_negative_drop_adapter_helps(self):
        # Adapter improves accuracy → drop is negative (not a degradation)
        assert accuracy_relative_drop(70.0, 72.0) == pytest.approx(-2.0)

    def test_full_accuracy_range(self):
        assert accuracy_relative_drop(100.0, 0.0) == pytest.approx(100.0)

    def test_fractional_pp(self):
        assert accuracy_relative_drop(80.5, 77.3) == pytest.approx(3.2, abs=1e-5)


# ───────────────────────────────────────── check_kill_condition tests ───

class TestCheckKillCondition:
    def test_below_threshold_passes(self):
        assert check_kill_condition(4.9, threshold=5.0) is False

    def test_exactly_at_threshold_passes(self):
        # Kill condition is strict (> threshold), so exactly equal should pass
        assert check_kill_condition(5.0, threshold=5.0) is False

    def test_above_threshold_kills(self):
        assert check_kill_condition(5.1, threshold=5.0) is True

    def test_zero_drop_passes(self):
        assert check_kill_condition(0.0) is False

    def test_negative_drop_passes(self):
        # Adapter helps → no kill
        assert check_kill_condition(-1.0) is False

    def test_custom_threshold(self):
        assert check_kill_condition(3.0, threshold=2.0) is True
        assert check_kill_condition(1.9, threshold=2.0) is False


# ──────────────────────────────────────────────── _format_table tests ───

class TestFormatTable:
    def _sample_results(self):
        return [
            ("FAM — no adapter (baseline)", 75.0, 10.0),
            ("FAM — Dogs adapter (adapted)", 72.0, 11.0),
        ]

    def test_returns_string(self):
        table = _format_table("cifar100", self._sample_results(), 3.0, 5.0)
        assert isinstance(table, str)

    def test_contains_method_names(self):
        table = _format_table("cifar100", self._sample_results(), 3.0, 5.0)
        assert "baseline" in table.lower()
        assert "adapted" in table.lower()

    def test_pass_condition_marker(self):
        # 3.0 pp drop < 5.0 pp threshold → pass
        table = _format_table("cifar100", self._sample_results(), 3.0, 5.0)
        assert "\u2713" in table  # ✓ checkmark
        assert "KILL CONDITION" not in table

    def test_kill_condition_marker(self):
        # 6.0 pp drop > 5.0 pp threshold → kill
        table = _format_table("cifar100", self._sample_results(), 6.0, 5.0)
        assert "\u26a0" in table  # ⚠ warning sign
        assert "KILL CONDITION MET" in table

    def test_kill_condition_at_boundary(self):
        # Exactly at threshold → pass (> is strict)
        table = _format_table("cifar100", self._sample_results(), 5.0, 5.0)
        assert "KILL CONDITION" not in table

    def test_dataset_name_in_table(self):
        table = _format_table("imagenet_r", self._sample_results(), 2.0, 5.0)
        assert "imagenet_r" in table

    def test_drop_value_in_table(self):
        table = _format_table("cifar100", self._sample_results(), 3.2, 5.0)
        assert "3.2" in table or "3.20" in table


# ──────────────────────────────────────────── load_features error test ───

class TestLoadFeatures:
    def test_raises_when_no_pt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="No train/test"):
                load_features(tmpdir, torch.device("cpu"))

    def test_raises_when_missing_train(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only a test file
            torch.save(
                {"embeds": torch.randn(10, 32), "labels": torch.zeros(10, dtype=torch.long)},
                os.path.join(tmpdir, "cifar100_test.pt"),
            )
            with pytest.raises(FileNotFoundError):
                load_features(tmpdir, torch.device("cpu"))

    def test_loads_correct_shapes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tr = {"embeds": torch.randn(50, 32), "labels": torch.zeros(50, dtype=torch.long)}
            te = {"embeds": torch.randn(10, 32), "labels": torch.zeros(10, dtype=torch.long)}
            torch.save(tr, os.path.join(tmpdir, "cifar100_train.pt"))
            torch.save(te, os.path.join(tmpdir, "cifar100_test.pt"))

            train_e, train_l, test_e, test_l = load_features(tmpdir, torch.device("cpu"))
            assert train_e.shape == (50, 32)
            assert test_e.shape == (10, 32)
            assert train_l.dtype == torch.long
            assert test_l.dtype == torch.long


# ───────────────────────────────────────── eval_fam_accuracy (no adapter) ───

class TestEvalFamAccuracyNoAdapter:
    """FAM without adapter should achieve high accuracy on well-separated data."""

    @pytest.fixture(scope="class")
    def generic_data(self):
        return _make_generic_data(seed=0)

    def test_returns_float_and_elapsed(self, generic_data):
        train_e, train_l, test_e, test_l = generic_data
        acc, elapsed = eval_fam_accuracy(
            train_e, train_l, test_e, test_l,
            device=torch.device("cpu"),
            num_classes=_NUM_CLASSES,
            adapter=None,
            core_entries=500,
            core_vigilance=0.7,
        )
        assert isinstance(acc, float)
        assert isinstance(elapsed, float)
        assert elapsed >= 0.0

    def test_accuracy_in_valid_range(self, generic_data):
        train_e, train_l, test_e, test_l = generic_data
        acc, _ = eval_fam_accuracy(
            train_e, train_l, test_e, test_l,
            device=torch.device("cpu"),
            num_classes=_NUM_CLASSES,
            adapter=None,
            core_entries=500,
            core_vigilance=0.7,
        )
        assert 0.0 <= acc <= 100.0

    def test_high_accuracy_on_separable_data(self, generic_data):
        """FAM should classify well-separated synthetic classes accurately."""
        train_e, train_l, test_e, test_l = generic_data
        acc, _ = eval_fam_accuracy(
            train_e, train_l, test_e, test_l,
            device=torch.device("cpu"),
            num_classes=_NUM_CLASSES,
            adapter=None,
            core_entries=500,
            core_vigilance=0.7,
        )
        assert acc >= 80.0, f"Expected ≥80% on well-separated data, got {acc:.1f}%"


# ─────────────────────────────────── identity adapter — no kill condition ───

class TestIdentityAdapterWithinThreshold:
    """An identity-initialised adapter must not trigger the kill condition."""

    @pytest.fixture(scope="class")
    def generic_data(self):
        return _make_generic_data(seed=1)

    def test_identity_adapter_small_drop(self, generic_data):
        """Identity adapter should produce ≤ 5 pp accuracy drop."""
        train_e, train_l, test_e, test_l = generic_data
        device = torch.device("cpu")
        kw = dict(
            device=device,
            num_classes=_NUM_CLASSES,
            core_entries=500,
            core_vigilance=0.7,
        )

        acc_baseline, _ = eval_fam_accuracy(train_e, train_l, test_e, test_l,
                                            adapter=None, **kw)
        adapter = _identity_adapter(embed_dim=_EMBED_DIM)
        adapter.eval()
        acc_adapted, _ = eval_fam_accuracy(train_e, train_l, test_e, test_l,
                                           adapter=adapter, **kw)

        drop = accuracy_relative_drop(acc_baseline, acc_adapted)
        assert not check_kill_condition(drop), (
            f"Identity adapter triggered kill condition: "
            f"baseline={acc_baseline:.1f}%, adapted={acc_adapted:.1f}%, "
            f"drop={drop:.1f} pp"
        )

    def test_identity_adapter_kill_condition_false(self, generic_data):
        """check_kill_condition must return False for the identity-adapter drop."""
        train_e, train_l, test_e, test_l = generic_data
        device = torch.device("cpu")
        kw = dict(
            device=device,
            num_classes=_NUM_CLASSES,
            core_entries=500,
            core_vigilance=0.7,
        )
        acc_baseline, _ = eval_fam_accuracy(train_e, train_l, test_e, test_l,
                                            adapter=None, **kw)
        adapter = _identity_adapter(embed_dim=_EMBED_DIM)
        adapter.eval()
        acc_adapted, _ = eval_fam_accuracy(train_e, train_l, test_e, test_l,
                                           adapter=adapter, **kw)
        drop = accuracy_relative_drop(acc_baseline, acc_adapted)
        assert check_kill_condition(drop) is False


# ─────────────────────────── distorting adapter — triggers kill condition ───

class TestDistortingAdapterTriggersKill:
    """An adapter that collapses most dimensions should fail the kill condition."""

    @pytest.fixture(scope="class")
    def generic_data(self):
        return _make_generic_data(seed=2)

    def test_high_drop_triggers_kill(self, generic_data):
        """An extremely lossy adapter should produce > 5 pp drop."""
        train_e, train_l, test_e, test_l = generic_data
        device = torch.device("cpu")

        acc_baseline, _ = eval_fam_accuracy(
            train_e, train_l, test_e, test_l,
            device=device,
            num_classes=_NUM_CLASSES,
            adapter=None,
            core_entries=500,
            core_vigilance=0.7,
        )

        # Project from 32-D to 2-D with random weights — almost all structure lost
        torch.manual_seed(77)
        bad_adapter = MetricAdapter(input_dim=_EMBED_DIM, output_dim=2)
        bad_adapter.eval()

        acc_adapted, _ = eval_fam_accuracy(
            train_e, train_l, test_e, test_l,
            device=device,
            num_classes=_NUM_CLASSES,
            adapter=bad_adapter,
            core_entries=500,
            core_vigilance=0.7,
        )

        drop = accuracy_relative_drop(acc_baseline, acc_adapted)
        # With 5 classes projected to 2-d, accuracy should collapse significantly
        assert check_kill_condition(drop), (
            f"Expected kill condition for highly distorting adapter: "
            f"baseline={acc_baseline:.1f}%, adapted={acc_adapted:.1f}%, "
            f"drop={drop:.1f} pp"
        )


# ─────────────────────────────────────── end-to-end drop & table test ───

class TestEndToEndDropAndTable:
    """Integration test: compute drop and format table from synthetic results."""

    def test_pass_scenario_table(self):
        results = [
            ("FAM — no adapter (baseline)", 90.0, 5.0),
            ("FAM — Dogs adapter (adapted)", 87.0, 5.5),
        ]
        drop = accuracy_relative_drop(90.0, 87.0)
        table = _format_table("cifar100", results, drop, threshold=5.0)
        assert "PASS" in table
        assert "\u2713" in table
        assert "3.00" in table or "3.0" in table

    def test_fail_scenario_table(self):
        results = [
            ("FAM — no adapter (baseline)", 90.0, 5.0),
            ("FAM — Dogs adapter (adapted)", 80.0, 5.5),
        ]
        drop = accuracy_relative_drop(90.0, 80.0)
        table = _format_table("cifar100", results, drop, threshold=5.0)
        assert "KILL CONDITION MET" in table
        assert "\u26a0" in table
