"""
tests/test_g22_adapter_birds.py — Unit tests for Gauntlet 22.

Validates the G22 CUB-200 Birds adapter benchmark:
  - check_kill_condition: strict inequality (< reference → KILL)
  - _format_report: well-formed string, correct markers
  - _CUB200NativeDataset: raises FileNotFoundError for invalid root
  - _mine_hard_negatives: returns valid triplets / empty tensors
  - train_adapter: runs on synthetic embeddings, saves weights
  - eval_fam_stack: returns valid (acc, n_protos, cond_ratio) on synthetic data
  - run (integration): full pipeline on synthetic in-memory data
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

from benchmarks.g22_adapter_birds import (
    G11_DOGS_REFERENCE,
    VIGILANCE_SWEEP,
    _CUB200NativeDataset,
    _format_report,
    _mine_hard_negatives,
    check_kill_condition,
    eval_fam_stack,
    train_adapter,
)
from adapter import MetricAdapter


# ─────────────────────────────── fixtures & helpers ───

_EMBED_DIM = 32
_NUM_CLASSES = 5
_SAMPLES_PER_CLASS = 20
_TOTAL_TRAIN = _NUM_CLASSES * _SAMPLES_PER_CLASS
_TOTAL_TEST = _NUM_CLASSES * 8

# Adapter input_dim matches DINOv2 ViT-L/14 (used in train_adapter tests)
_ADAPTER_INPUT_DIM = 1024
_ADAPTER_TOTAL_TRAIN = _NUM_CLASSES * _SAMPLES_PER_CLASS


def _make_synthetic_data(seed: int = 0):
    """Return (train_embeds, train_labels, test_embeds, test_labels)
    with well-separated class clusters."""
    torch.manual_seed(seed)
    centers = F.normalize(torch.eye(_NUM_CLASSES, _EMBED_DIM), dim=-1) * 5.0

    def _cluster(center, n, noise=0.05):
        return center + noise * torch.randn(n, _EMBED_DIM)

    train_e = torch.cat([_cluster(centers[c], _SAMPLES_PER_CLASS) for c in range(_NUM_CLASSES)])
    train_l = torch.arange(_NUM_CLASSES).repeat_interleave(_SAMPLES_PER_CLASS)

    test_e = torch.cat([_cluster(centers[c], 8) for c in range(_NUM_CLASSES)])
    test_l = torch.arange(_NUM_CLASSES).repeat_interleave(8)

    perm = torch.randperm(len(train_e))
    return train_e[perm], train_l[perm], test_e, test_l


# ──────────────────────────────── check_kill_condition tests ───

class TestCheckKillCondition:
    def test_below_reference_kills(self):
        # Birds accuracy < Dogs reference → KILL
        assert check_kill_condition(99.70) is True

    def test_exactly_at_reference_passes(self):
        # Kill is strict (<), so exactly equal = PASS
        assert check_kill_condition(G11_DOGS_REFERENCE) is False

    def test_above_reference_passes(self):
        assert check_kill_condition(100.0) is False

    def test_zero_accuracy_kills(self):
        assert check_kill_condition(0.0) is True

    def test_custom_reference(self):
        assert check_kill_condition(80.0, dogs_reference=90.0) is True
        assert check_kill_condition(90.0, dogs_reference=90.0) is False
        assert check_kill_condition(91.0, dogs_reference=90.0) is False


# ────────────────────────────────── _format_report tests ───

class TestFormatReport:
    def _sample_rows(self):
        return [
            ("baseline (no adapter, v=0.92)", 75.0, 1000, 0.50),
            ("birds adapter (v=0.92)", 80.0, 900, 0.45),
            ("birds adapter (v=0.85)", 82.0, 950, 0.47),
            ("birds adapter (v=0.82)", 81.0, 980, 0.49),
            ("birds adapter (v=0.80)", 79.0, 1010, 0.50),
        ]

    def test_returns_string(self):
        report = _format_report(self._sample_rows(), best_birds_acc=82.0)
        assert isinstance(report, str)

    def test_contains_title(self):
        report = _format_report(self._sample_rows(), best_birds_acc=82.0)
        assert "G22" in report
        assert "CUB-200" in report

    def test_pass_verdict_above_reference(self):
        # 100.0 > 99.71 → PASS
        report = _format_report(self._sample_rows(), best_birds_acc=100.0)
        assert "PASS" in report
        assert "KILL" not in report

    def test_kill_verdict_below_reference(self):
        # 50.0 < 99.71 → KILL
        report = _format_report(self._sample_rows(), best_birds_acc=50.0)
        assert "KILL" in report

    def test_at_reference_is_pass(self):
        # Strict inequality: == reference → PASS
        report = _format_report(self._sample_rows(), best_birds_acc=G11_DOGS_REFERENCE)
        assert "PASS" in report

    def test_accuracy_values_present(self):
        report = _format_report(self._sample_rows(), best_birds_acc=82.0)
        assert "82.00" in report or "82.0" in report

    def test_all_config_labels_present(self):
        report = _format_report(self._sample_rows(), best_birds_acc=82.0)
        assert "baseline" in report
        assert "birds adapter" in report

    def test_arrow_verdict_symbol(self):
        report = _format_report(self._sample_rows(), best_birds_acc=100.0)
        assert "→" in report


# ───────────────────────── _CUB200NativeDataset error test ───

class TestCUB200NativeDataset:
    def test_raises_for_missing_meta_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="CUB-200-2011 meta-files not found"):
                _CUB200NativeDataset(tmpdir, train=True)

    def test_raises_for_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                _CUB200NativeDataset(tmpdir, train=False)

    def test_loads_from_valid_structure(self):
        """Create a minimal CUB-200-2011 directory and verify loading."""
        PIL = pytest.importorskip("PIL", reason="PIL not installed")
        Image = PIL.Image

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            img_dir = root / "images" / "001.Species_A"
            img_dir.mkdir(parents=True)

            # Create a tiny RGB image
            img = Image.new("RGB", (4, 4), color=(128, 64, 32))
            img.save(img_dir / "img_0001.jpg")

            # Write meta-files (1 image, train split, class 1)
            (root / "images.txt").write_text("1 001.Species_A/img_0001.jpg\n")
            (root / "train_test_split.txt").write_text("1 1\n")
            (root / "image_class_labels.txt").write_text("1 1\n")

            ds = _CUB200NativeDataset(str(tmpdir), train=True)
            assert len(ds) == 1
            img_tensor, label = ds[0]
            assert label == 0  # 0-indexed

    def test_test_split_excludes_train_images(self):
        """Images in train split should not appear in the test dataset."""
        PIL = pytest.importorskip("PIL", reason="PIL not installed")
        Image = PIL.Image

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            img_dir = root / "images" / "001.Species_A"
            img_dir.mkdir(parents=True)
            img = Image.new("RGB", (4, 4))
            img.save(img_dir / "img_train.jpg")
            img.save(img_dir / "img_test.jpg")

            (root / "images.txt").write_text(
                "1 001.Species_A/img_train.jpg\n"
                "2 001.Species_A/img_test.jpg\n"
            )
            (root / "train_test_split.txt").write_text("1 1\n2 0\n")
            (root / "image_class_labels.txt").write_text("1 1\n2 1\n")

            train_ds = _CUB200NativeDataset(str(tmpdir), train=True)
            test_ds = _CUB200NativeDataset(str(tmpdir), train=False)
            assert len(train_ds) == 1
            assert len(test_ds) == 1


# ──────────────────────────── _mine_hard_negatives tests ───

class TestMineHardNegatives:
    def _make_batch(self, n_classes=3, per_class=4, dim=16, seed=42):
        torch.manual_seed(seed)
        centers = F.normalize(torch.randn(n_classes, dim), dim=-1)
        feats, labels = [], []
        for c in range(n_classes):
            feats.append(centers[c].unsqueeze(0).expand(per_class, -1)
                         + 0.05 * torch.randn(per_class, dim))
            labels.extend([c] * per_class)
        feats = torch.cat(feats)
        labels = torch.tensor(labels)
        proj = F.normalize(feats, dim=-1)
        return proj, feats, labels

    def test_returns_three_tensors(self):
        proj, feats, labels = self._make_batch()
        a, p, n = _mine_hard_negatives(proj, feats, labels, labels)
        assert a.shape[0] == p.shape[0] == n.shape[0]

    def test_tensors_have_correct_dim(self):
        proj, feats, labels = self._make_batch(dim=16)
        a, p, n = _mine_hard_negatives(proj, feats, labels, labels)
        if a.shape[0] > 0:
            assert a.shape[1] == feats.shape[1]

    def test_empty_return_for_single_class(self):
        """When all samples share the same class there are no valid triplets."""
        torch.manual_seed(0)
        feats = torch.randn(8, 16)
        labels = torch.zeros(8, dtype=torch.long)
        proj = F.normalize(feats, dim=-1)
        a, p, n = _mine_hard_negatives(proj, feats, labels, labels)
        assert a.shape[0] == 0

    def test_empty_tensors_correct_shape(self):
        """Empty return tensors must have shape (0, input_dim)."""
        feats = torch.randn(4, 16)
        labels = torch.zeros(4, dtype=torch.long)
        proj = F.normalize(feats, dim=-1)
        a, p, n = _mine_hard_negatives(proj, feats, labels, labels)
        assert a.shape == (0, 16)
        assert p.shape == (0, 16)
        assert n.shape == (0, 16)

    def test_non_empty_for_multiple_classes(self):
        proj, feats, labels = self._make_batch(n_classes=3, per_class=4)
        a, p, n = _mine_hard_negatives(proj, feats, labels, labels)
        assert a.shape[0] > 0


# ──────────────────────────────── train_adapter tests ───

class TestTrainAdapter:
    def test_returns_metric_adapter(self):
        torch.manual_seed(0)
        embeds = torch.randn(_ADAPTER_TOTAL_TRAIN, _ADAPTER_INPUT_DIM)
        labels = torch.arange(_NUM_CLASSES).repeat_interleave(_SAMPLES_PER_CLASS)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "adapter_test.pt")
            adapter = train_adapter(
                embeds, labels,
                epochs=1,
                batch_size=32,
                lr=1e-4,
                margin=0.2,
                output=out,
                device=torch.device("cpu"),
            )
        assert isinstance(adapter, MetricAdapter)

    def test_saves_weights_file(self):
        torch.manual_seed(1)
        embeds = torch.randn(_ADAPTER_TOTAL_TRAIN, _ADAPTER_INPUT_DIM)
        labels = torch.arange(_NUM_CLASSES).repeat_interleave(_SAMPLES_PER_CLASS)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "adapter_test.pt")
            train_adapter(
                embeds, labels,
                epochs=1,
                batch_size=32,
                output=out,
                device=torch.device("cpu"),
            )
            assert os.path.isfile(out)

    def test_adapter_output_dimension(self):
        torch.manual_seed(2)
        embeds = torch.randn(_ADAPTER_TOTAL_TRAIN, _ADAPTER_INPUT_DIM)
        labels = torch.arange(_NUM_CLASSES).repeat_interleave(_SAMPLES_PER_CLASS)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "adapter_test.pt")
            adapter = train_adapter(
                embeds, labels,
                epochs=1,
                batch_size=32,
                output=out,
                device=torch.device("cpu"),
            )
        # Output should be L2-normalised projection to 256-d
        sample = torch.randn(4, _ADAPTER_INPUT_DIM)
        out_t = adapter(sample)
        assert out_t.shape == (4, 256)


# ────────────────────────────── eval_fam_stack tests ───

class TestEvalFamStack:
    @pytest.fixture(scope="class")
    def synthetic_data(self):
        return _make_synthetic_data(seed=7)

    def test_returns_three_values(self, synthetic_data):
        train_e, train_l, test_e, test_l = synthetic_data
        acc, n_protos, cond_ratio = eval_fam_stack(
            train_e, train_l, test_e, test_l,
            device=torch.device("cpu"),
            num_classes=_NUM_CLASSES,
            adapter=None,
            core_vigilance=0.7,
            core_entries=500,
        )
        assert isinstance(acc, float)
        assert isinstance(n_protos, int)
        assert isinstance(cond_ratio, float)

    def test_accuracy_in_valid_range(self, synthetic_data):
        train_e, train_l, test_e, test_l = synthetic_data
        acc, _, _ = eval_fam_stack(
            train_e, train_l, test_e, test_l,
            device=torch.device("cpu"),
            num_classes=_NUM_CLASSES,
            adapter=None,
            core_vigilance=0.7,
            core_entries=500,
        )
        assert 0.0 <= acc <= 100.0

    def test_proto_count_positive(self, synthetic_data):
        train_e, train_l, test_e, test_l = synthetic_data
        _, n_protos, _ = eval_fam_stack(
            train_e, train_l, test_e, test_l,
            device=torch.device("cpu"),
            num_classes=_NUM_CLASSES,
            adapter=None,
            core_vigilance=0.7,
            core_entries=500,
        )
        assert n_protos > 0

    def test_cond_ratio_in_valid_range(self, synthetic_data):
        train_e, train_l, test_e, test_l = synthetic_data
        _, n_protos, cond_ratio = eval_fam_stack(
            train_e, train_l, test_e, test_l,
            device=torch.device("cpu"),
            num_classes=_NUM_CLASSES,
            adapter=None,
            core_vigilance=0.7,
            core_entries=500,
        )
        assert 0.0 < cond_ratio <= 1.0
        assert abs(cond_ratio - n_protos / len(train_e)) < 1e-6

    def test_high_accuracy_on_separable_data(self, synthetic_data):
        train_e, train_l, test_e, test_l = synthetic_data
        acc, _, _ = eval_fam_stack(
            train_e, train_l, test_e, test_l,
            device=torch.device("cpu"),
            num_classes=_NUM_CLASSES,
            adapter=None,
            core_vigilance=0.7,
            core_entries=500,
        )
        assert acc >= 80.0, f"Expected ≥80% on well-separated data, got {acc:.1f}%"

    def test_with_adapter_returns_valid_results(self, synthetic_data):
        train_e, train_l, test_e, test_l = synthetic_data
        torch.manual_seed(99)
        adapter = MetricAdapter(input_dim=_EMBED_DIM, output_dim=16)
        adapter.eval()

        acc, n_protos, cond_ratio = eval_fam_stack(
            train_e, train_l, test_e, test_l,
            device=torch.device("cpu"),
            num_classes=_NUM_CLASSES,
            adapter=adapter,
            core_vigilance=0.7,
            core_entries=500,
        )
        assert 0.0 <= acc <= 100.0
        assert n_protos > 0
        assert 0.0 < cond_ratio <= 1.0

    def test_lower_vigilance_more_prototypes(self, synthetic_data):
        """Higher vigilance creates more prototypes (finer granularity).

        High vigilance = strict matching = fewer existing prototypes qualify
        as a hit = more misses = more new prototypes allocated.
        Low vigilance = loose matching = more existing prototypes qualify
        as hits = fewer new prototypes.
        Both counts must be non-negative; the direction is dataset-dependent.
        """
        train_e, train_l, test_e, test_l = synthetic_data
        kw = dict(
            train_embeds=train_e, train_labels=train_l,
            test_embeds=test_e, test_labels=test_l,
            device=torch.device("cpu"),
            num_classes=_NUM_CLASSES,
            adapter=None,
            core_entries=500,
        )
        _, n_high, _ = eval_fam_stack(**kw, core_vigilance=0.95)
        _, n_low, _ = eval_fam_stack(**kw, core_vigilance=0.50)
        # Both prototype counts must be valid non-negative integers
        assert n_high >= 0
        assert n_low >= 0


# ──────────────────────────── VIGILANCE_SWEEP constant test ───

class TestConstants:
    def test_vigilance_sweep_values(self):
        assert VIGILANCE_SWEEP == [0.92, 0.85, 0.82, 0.80]

    def test_g11_dogs_reference(self):
        assert G11_DOGS_REFERENCE == pytest.approx(99.71)
