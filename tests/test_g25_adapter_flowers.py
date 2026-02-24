"""
tests/test_g25_adapter_flowers.py — Unit tests for Gauntlet 25.

Validates the G25 Oxford Flowers-102 adapter benchmark:
  - check_kill_condition triggers only when flowers_acc < dogs_ref
  - _format_report produces a well-formed string with expected markers
  - eval_full_stack evaluates correctly on synthetic data (no adapter)
  - An identity-like adapter does not crash eval_full_stack
  - _mine_hard_negatives returns valid triplet shapes
  - train_adapter runs without error on tiny synthetic data
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

# Allow importing from the repo root regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.g25_adapter_flowers import (
    _mine_hard_negatives,
    check_kill_condition,
    eval_full_stack,
    train_adapter,
    _format_report,
    _DOGS_REFERENCE_ACC,
    _NUM_CLASSES,
)
from adapter import MetricAdapter


# ─────────────────────────────────────────────────── fixtures & helpers ───

_EMBED_DIM = 1024   # matches ViT-L/14 output
_N_CLASSES = 5      # small subset for testing
_SAMPLES_PER_CLASS = 20


def _make_synthetic_data(seed: int = 0):
    """Return (train_embeds, train_labels, test_embeds, test_labels) for a
    well-separated synthetic dataset compatible with the 1024-d adapter."""
    torch.manual_seed(seed)
    # Orthogonal class centers on the hypersphere
    centers = F.normalize(torch.eye(_N_CLASSES, _EMBED_DIM), dim=-1)

    def _cluster(center, n, noise=0.05):
        return F.normalize(center + noise * torch.randn(n, _EMBED_DIM), dim=-1)

    train_e = torch.cat([_cluster(centers[c], _SAMPLES_PER_CLASS) for c in range(_N_CLASSES)])
    train_l = torch.arange(_N_CLASSES).repeat_interleave(_SAMPLES_PER_CLASS)

    test_e = torch.cat([_cluster(centers[c], 5) for c in range(_N_CLASSES)])
    test_l = torch.arange(_N_CLASSES).repeat_interleave(5)

    perm = torch.randperm(len(train_e))
    return train_e[perm], train_l[perm], test_e, test_l


def _identity_adapter_1024() -> MetricAdapter:
    """Return a MetricAdapter that passes through (nearly) unchanged."""
    # input=1024 → hidden=512 → output=256; just use random init and eval mode
    torch.manual_seed(42)
    adapter = MetricAdapter(input_dim=1024, output_dim=256, hidden_dim=512)
    adapter.eval()
    return adapter


# ──────────────────────────────────────── check_kill_condition tests ───

class TestCheckKillCondition:
    def test_above_reference_passes(self):
        # flowers_acc > dogs_ref → PASS
        assert check_kill_condition(99.72) is False

    def test_exactly_at_reference_passes(self):
        # Kill condition is strict (<), so exactly equal should PASS
        assert check_kill_condition(_DOGS_REFERENCE_ACC) is False

    def test_below_reference_kills(self):
        assert check_kill_condition(90.0) is True

    def test_zero_kills(self):
        assert check_kill_condition(0.0) is True

    def test_custom_dogs_ref(self):
        assert check_kill_condition(50.0, dogs_ref=60.0) is True
        assert check_kill_condition(60.0, dogs_ref=60.0) is False
        assert check_kill_condition(70.0, dogs_ref=60.0) is False


# ────────────────────────────────────────────── _format_report tests ───

class TestFormatReport:
    def _sample_rows(self):
        return [
            ("baseline (no adapter, v=0.92)", 55.0, 800, 0.40),
            ("flowers adapter (v=0.92)",      60.0, 700, 0.35),
            ("flowers adapter (v=0.85)",       65.0, 600, 0.30),
        ]

    def test_returns_string(self):
        report = _format_report(self._sample_rows(), 65.0, _DOGS_REFERENCE_ACC)
        assert isinstance(report, str)

    def test_contains_header(self):
        report = _format_report(self._sample_rows(), 65.0, _DOGS_REFERENCE_ACC)
        assert "G25" in report
        assert "Flowers-102" in report

    def test_pass_verdict_when_above_ref(self):
        # 65.0 < 99.71 → KILL
        report = _format_report(self._sample_rows(), 65.0, _DOGS_REFERENCE_ACC)
        assert "KILL" in report

    def test_pass_verdict_when_meets_or_exceeds_ref(self):
        report = _format_report(self._sample_rows(), 99.71, _DOGS_REFERENCE_ACC)
        assert "PASS" in report

    def test_contains_config_names(self):
        report = _format_report(self._sample_rows(), 65.0, _DOGS_REFERENCE_ACC)
        assert "baseline" in report
        assert "flowers adapter" in report

    def test_arrow_direction_marker(self):
        report = _format_report(self._sample_rows(), 65.0, _DOGS_REFERENCE_ACC)
        assert "→" in report

    def test_dogs_ref_value_shown(self):
        report = _format_report(self._sample_rows(), 65.0, _DOGS_REFERENCE_ACC)
        assert "99.71" in report


# ─────────────────────────────────── _mine_hard_negatives tests ───

class TestMineHardNegatives:
    def _make_batch(self, n_classes: int = 4, per_class: int = 5, dim: int = 32):
        torch.manual_seed(0)
        labels = torch.arange(n_classes).repeat_interleave(per_class)
        feats = F.normalize(torch.randn(len(labels), dim), dim=-1)
        proj = F.normalize(torch.randn(len(labels), 16), dim=-1)
        return proj, feats, labels

    def test_returns_three_tensors(self):
        proj, feats, labels = self._make_batch()
        anch, pos, neg = _mine_hard_negatives(proj, feats, labels, labels)
        assert anch.ndim == 2
        assert pos.ndim == 2
        assert neg.ndim == 2

    def test_shapes_match(self):
        proj, feats, labels = self._make_batch()
        anch, pos, neg = _mine_hard_negatives(proj, feats, labels, labels)
        assert anch.shape == pos.shape == neg.shape

    def test_feature_dim_preserved(self):
        proj, feats, labels = self._make_batch(dim=32)
        anch, _, _ = _mine_hard_negatives(proj, feats, labels, labels)
        assert anch.shape[1] == 32

    def test_empty_when_single_class(self):
        torch.manual_seed(0)
        feats = F.normalize(torch.randn(8, 16), dim=-1)
        proj = F.normalize(torch.randn(8, 8), dim=-1)
        labels = torch.zeros(8, dtype=torch.long)
        anch, pos, neg = _mine_hard_negatives(proj, feats, labels, labels)
        assert anch.shape[0] == 0

    def test_at_least_one_triplet_with_multi_class(self):
        proj, feats, labels = self._make_batch(n_classes=3, per_class=10)
        anch, pos, neg = _mine_hard_negatives(proj, feats, labels, labels)
        assert anch.shape[0] > 0


# ────────────────────────────────── eval_full_stack (no adapter) ───

class TestEvalFullStackNoAdapter:
    """eval_full_stack must return valid metrics on synthetic data."""

    @pytest.fixture(scope="class")
    def synthetic_data(self):
        return _make_synthetic_data(seed=0)

    def test_returns_tuple_of_three(self, synthetic_data):
        train_e, train_l, test_e, test_l = synthetic_data
        result = eval_full_stack(
            train_e, train_l, test_e, test_l,
            device=torch.device("cpu"),
            adapter=None,
            vigilance=0.85,
            core_entries=500,
        )
        assert len(result) == 3

    def test_accuracy_in_valid_range(self, synthetic_data):
        train_e, train_l, test_e, test_l = synthetic_data
        acc, _, _ = eval_full_stack(
            train_e, train_l, test_e, test_l,
            device=torch.device("cpu"),
            adapter=None,
            vigilance=0.85,
            core_entries=500,
        )
        assert 0.0 <= acc <= 100.0

    def test_prototype_count_positive(self, synthetic_data):
        train_e, train_l, test_e, test_l = synthetic_data
        _, protos, _ = eval_full_stack(
            train_e, train_l, test_e, test_l,
            device=torch.device("cpu"),
            adapter=None,
            vigilance=0.85,
            core_entries=500,
        )
        assert protos > 0

    def test_condensation_ratio_in_range(self, synthetic_data):
        train_e, train_l, test_e, test_l = synthetic_data
        n_train = len(train_l)
        _, protos, cond = eval_full_stack(
            train_e, train_l, test_e, test_l,
            device=torch.device("cpu"),
            adapter=None,
            vigilance=0.85,
            core_entries=500,
        )
        assert cond == pytest.approx(protos / n_train)


# ──────────────────────────────── eval_full_stack with adapter ───

class TestEvalFullStackWithAdapter:
    """eval_full_stack must not crash when an adapter is supplied."""

    @pytest.fixture(scope="class")
    def synthetic_data(self):
        return _make_synthetic_data(seed=1)

    def test_runs_without_error(self, synthetic_data):
        train_e, train_l, test_e, test_l = synthetic_data
        adapter = _identity_adapter_1024()
        acc, protos, cond = eval_full_stack(
            train_e, train_l, test_e, test_l,
            device=torch.device("cpu"),
            adapter=adapter,
            vigilance=0.85,
            core_entries=500,
        )
        assert 0.0 <= acc <= 100.0
        assert protos >= 0
        assert cond >= 0.0


# ──────────────────────────────────────── train_adapter smoke test ───

class TestTrainAdapter:
    """train_adapter must run without error on tiny synthetic data."""

    def test_returns_metric_adapter(self, tmp_path):
        torch.manual_seed(7)
        n_classes = 5
        per_class = 10
        feats = F.normalize(torch.randn(n_classes * per_class, 1024), dim=-1)
        labels = torch.arange(n_classes).repeat_interleave(per_class)

        out_path = str(tmp_path / "adapter_test.pt")
        adapter = train_adapter(
            train_embeds=feats,
            train_labels=labels,
            epochs=1,
            batch_size=16,
            lr=1e-4,
            margin=0.2,
            output=out_path,
            device=torch.device("cpu"),
        )
        assert isinstance(adapter, MetricAdapter)

    def test_saves_weights_file(self, tmp_path):
        torch.manual_seed(8)
        n_classes = 5
        per_class = 10
        feats = F.normalize(torch.randn(n_classes * per_class, 1024), dim=-1)
        labels = torch.arange(n_classes).repeat_interleave(per_class)

        out_path = str(tmp_path / "adapter_saved.pt")
        train_adapter(
            train_embeds=feats,
            train_labels=labels,
            epochs=1,
            batch_size=16,
            lr=1e-4,
            margin=0.2,
            output=out_path,
            device=torch.device("cpu"),
        )
        import os
        assert os.path.isfile(out_path)
