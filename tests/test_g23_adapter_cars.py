"""
tests/test_g23_adapter_cars.py — Unit tests for Gauntlet 23: Adapter Sweep on Stanford Cars.

Validates the G23 benchmark:
  - _check_kill_condition uses strict inequality (< reference) so exactly
    meeting the reference is a PASS.
  - _mine_hard_negatives returns valid triplets.
  - eval_full_stack returns plausible metrics on synthetic data.
  - train_adapter runs without error on a small synthetic feature bank.
"""

import os
import sys
import tempfile

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.g23_adapter_cars import (
    G11_DOGS_REFERENCE_ACC,
    VIGILANCE_SWEEP,
    _check_kill_condition,
    _mine_hard_negatives,
    _print_table,
    eval_full_stack,
    train_adapter,
)
from adapter import MetricAdapter


# ─────────────────────────────────────────── fixtures & helpers ───

_EMBED_DIM = 32
_NUM_CLASSES = 5
_SAMPLES_PER_CLASS = 20
_TOTAL_TRAIN = _NUM_CLASSES * _SAMPLES_PER_CLASS
_TOTAL_TEST = _NUM_CLASSES * 6


def _make_synthetic_data(seed: int = 0):
    """Return (train_embeds, train_labels, test_embeds, test_labels) for a
    well-separated synthetic dataset."""
    torch.manual_seed(seed)
    centers = F.normalize(torch.eye(_NUM_CLASSES, _EMBED_DIM), dim=-1) * 5.0

    def _cluster(center, n, noise=0.05):
        return center + noise * torch.randn(n, _EMBED_DIM)

    train_e = torch.cat([_cluster(centers[c], _SAMPLES_PER_CLASS) for c in range(_NUM_CLASSES)])
    train_l = torch.arange(_NUM_CLASSES).repeat_interleave(_SAMPLES_PER_CLASS)
    test_e = torch.cat([_cluster(centers[c], 6) for c in range(_NUM_CLASSES)])
    test_l = torch.arange(_NUM_CLASSES).repeat_interleave(6)

    perm = torch.randperm(len(train_e))
    return train_e[perm], train_l[perm], test_e, test_l


# ──────────────────────────────────────── _check_kill_condition ───

class TestCheckKillCondition:
    def test_exactly_at_reference_passes(self):
        # Strict < means exactly matching the reference is a PASS
        assert _check_kill_condition(G11_DOGS_REFERENCE_ACC) is False

    def test_above_reference_passes(self):
        assert _check_kill_condition(G11_DOGS_REFERENCE_ACC + 0.01) is False

    def test_below_reference_kills(self):
        assert _check_kill_condition(G11_DOGS_REFERENCE_ACC - 0.01) is True

    def test_zero_accuracy_kills(self):
        assert _check_kill_condition(0.0) is True

    def test_perfect_accuracy_passes(self):
        assert _check_kill_condition(100.0) is False


# ──────────────────────────────────────── _mine_hard_negatives ───

class TestMineHardNegatives:
    def _setup_batch(self, n_per_class: int = 8, n_classes: int = 3):
        """Create a mini-batch with well-separated classes."""
        torch.manual_seed(42)
        n = n_per_class * n_classes
        centers = F.normalize(torch.eye(n_classes, _EMBED_DIM), dim=-1) * 5.0
        feats = torch.cat([
            centers[c] + 0.05 * torch.randn(n_per_class, _EMBED_DIM)
            for c in range(n_classes)
        ])
        labels = torch.arange(n_classes).repeat_interleave(n_per_class)
        proj = F.normalize(feats, dim=-1)  # pretend adapter is identity
        return proj, feats, labels

    def test_returns_three_tensors(self):
        proj, feats, labels = self._setup_batch()
        anch, pos, neg = _mine_hard_negatives(proj, feats, labels, labels)
        assert isinstance(anch, torch.Tensor)
        assert isinstance(pos, torch.Tensor)
        assert isinstance(neg, torch.Tensor)

    def test_output_shapes_match(self):
        proj, feats, labels = self._setup_batch()
        anch, pos, neg = _mine_hard_negatives(proj, feats, labels, labels)
        T = anch.shape[0]
        assert pos.shape == (T, _EMBED_DIM)
        assert neg.shape == (T, _EMBED_DIM)

    def test_positive_triplets_found(self):
        """Should find at least one valid triplet."""
        proj, feats, labels = self._setup_batch(n_per_class=8, n_classes=3)
        anch, pos, neg = _mine_hard_negatives(proj, feats, labels, labels)
        assert anch.shape[0] > 0

    def test_single_class_returns_empty(self):
        """When only one class present, no valid triplets exist."""
        torch.manual_seed(0)
        feats = torch.randn(8, _EMBED_DIM)
        labels = torch.zeros(8, dtype=torch.long)
        proj = F.normalize(feats, dim=-1)
        anch, pos, neg = _mine_hard_negatives(proj, feats, labels, labels)
        assert anch.shape[0] == 0


# ──────────────────────────────────────── eval_full_stack ───

class TestEvalFullStack:
    @pytest.fixture(scope="class")
    def synthetic_data(self):
        return _make_synthetic_data(seed=7)

    def test_returns_correct_types(self, synthetic_data):
        train_e, train_l, test_e, test_l = synthetic_data
        acc, protos, cond = eval_full_stack(
            train_e, train_l, test_e, test_l,
            device=torch.device("cpu"),
            num_classes=_NUM_CLASSES,
            adapter=None,
            vigilance=0.7,
        )
        assert isinstance(acc, float)
        assert isinstance(protos, int)
        assert isinstance(cond, float)

    def test_accuracy_in_valid_range(self, synthetic_data):
        train_e, train_l, test_e, test_l = synthetic_data
        acc, protos, cond = eval_full_stack(
            train_e, train_l, test_e, test_l,
            device=torch.device("cpu"),
            num_classes=_NUM_CLASSES,
            adapter=None,
            vigilance=0.7,
        )
        assert 0.0 <= acc <= 100.0

    def test_protos_leq_train_samples(self, synthetic_data):
        train_e, train_l, test_e, test_l = synthetic_data
        _, protos, _ = eval_full_stack(
            train_e, train_l, test_e, test_l,
            device=torch.device("cpu"),
            num_classes=_NUM_CLASSES,
            adapter=None,
            vigilance=0.7,
        )
        assert protos <= _TOTAL_TRAIN

    def test_condensation_ratio_in_range(self, synthetic_data):
        train_e, train_l, test_e, test_l = synthetic_data
        _, protos, cond = eval_full_stack(
            train_e, train_l, test_e, test_l,
            device=torch.device("cpu"),
            num_classes=_NUM_CLASSES,
            adapter=None,
            vigilance=0.7,
        )
        assert cond == pytest.approx(protos / _TOTAL_TRAIN)
        assert 0.0 < cond <= 1.0

    def test_high_vigilance_fewer_protos(self, synthetic_data):
        """Higher vigilance → more selective ingestion → fewer prototypes."""
        train_e, train_l, test_e, test_l = synthetic_data
        _, protos_low_v, _ = eval_full_stack(
            train_e, train_l, test_e, test_l,
            device=torch.device("cpu"),
            num_classes=_NUM_CLASSES,
            adapter=None,
            vigilance=0.3,  # very lenient — admits many prototypes
        )
        _, protos_high_v, _ = eval_full_stack(
            train_e, train_l, test_e, test_l,
            device=torch.device("cpu"),
            num_classes=_NUM_CLASSES,
            adapter=None,
            vigilance=0.99,  # very strict — admits very few
        )
        assert protos_high_v <= protos_low_v

    def test_adapter_does_not_crash(self, synthetic_data):
        """Passing a random adapter should not raise."""
        train_e, train_l, test_e, test_l = synthetic_data
        torch.manual_seed(1)
        adapter = MetricAdapter(input_dim=_EMBED_DIM, output_dim=16, hidden_dim=32)
        adapter.eval()
        acc, protos, cond = eval_full_stack(
            train_e, train_l, test_e, test_l,
            device=torch.device("cpu"),
            num_classes=_NUM_CLASSES,
            adapter=adapter,
            vigilance=0.7,
        )
        assert 0.0 <= acc <= 100.0
        assert protos >= 0


# ──────────────────────────────────────── train_adapter ───

class TestTrainAdapter:
    def test_returns_metric_adapter(self):
        torch.manual_seed(0)
        train_e, train_l, _, _ = _make_synthetic_data(seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "adapter_test.pt")
            adapter = train_adapter(
                train_embeds=train_e,
                train_labels=train_l,
                epochs=1,
                batch_size=32,
                lr=1e-3,
                margin=0.2,
                output=out,
                device=torch.device("cpu"),
            )
        assert isinstance(adapter, MetricAdapter)

    def test_weights_saved(self):
        torch.manual_seed(1)
        train_e, train_l, _, _ = _make_synthetic_data(seed=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "adapter_test.pt")
            train_adapter(
                train_embeds=train_e,
                train_labels=train_l,
                epochs=1,
                batch_size=32,
                lr=1e-3,
                margin=0.2,
                output=out,
                device=torch.device("cpu"),
            )
            assert os.path.isfile(out)
            state = torch.load(out, weights_only=True)
            assert isinstance(state, dict)
            assert len(state) > 0

    def test_adapter_output_shape(self):
        torch.manual_seed(2)
        train_e, train_l, _, _ = _make_synthetic_data(seed=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "adapter_test.pt")
            adapter = train_adapter(
                train_embeds=train_e,
                train_labels=train_l,
                epochs=1,
                batch_size=32,
                lr=1e-3,
                margin=0.2,
                output=out,
                device=torch.device("cpu"),
            )
        adapter.eval()
        x = torch.randn(4, _EMBED_DIM)
        with torch.no_grad():
            out_tensor = adapter(x)
        # MetricAdapter with input_dim=_EMBED_DIM, output_dim=256, hidden_dim=512
        assert out_tensor.shape == (4, 256)


# ──────────────────────────────────────── _print_table (smoke) ───

class TestPrintTable:
    def test_does_not_raise(self, capsys):
        rows = [
            ("baseline (no adapter, v=0.92)", 80.0, 100, 0.50),
            ("cars adapter (v=0.92)", 85.0, 90, 0.45),
            ("cars adapter (v=0.85)", 87.0, 80, 0.40),
        ]
        _print_table(rows)
        captured = capsys.readouterr()
        assert "G23" in captured.out
        assert "Stanford Cars" in captured.out
        assert "baseline" in captured.out
        assert "cars adapter" in captured.out
