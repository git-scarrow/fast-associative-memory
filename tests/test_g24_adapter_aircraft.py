"""
tests/test_g24_adapter_aircraft.py — Unit tests for Gauntlet 24.

Validates the G24 FGVC Aircraft adapter sweep benchmark:
  - _mine_hard_negatives returns valid triplets or empty tensors
  - train_adapter runs without error on synthetic features
  - eval_full_stack returns correct shapes and types
  - _format_results produces a well-formed string
  - Kill-condition check: aircraft best < dogs reference → kill
  - extract_or_load_features loads from cache when files exist
"""

import os
import sys
import tempfile

import pytest
import torch
import torch.nn.functional as F

# Allow importing from the repo root regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.g24_adapter_aircraft import (
    _mine_hard_negatives,
    train_adapter,
    eval_full_stack,
    _format_results,
    extract_or_load_features,
    G11_DOGS_REFERENCE_ACC,
    CACHE_TRAIN,
    CACHE_TEST,
)
from adapter import MetricAdapter


# ─────────────────────────── synthetic data helpers ──────────────────────────

_EMBED_DIM = 64
_NUM_CLASSES = 10
_TRAIN_PER_CLASS = 30
_TEST_PER_CLASS = 10
_TOTAL_TRAIN = _NUM_CLASSES * _TRAIN_PER_CLASS
_TOTAL_TEST = _NUM_CLASSES * _TEST_PER_CLASS


def _make_synthetic_data(seed: int = 0):
    """Return well-separated (train_embeds, train_labels, test_embeds, test_labels)."""
    torch.manual_seed(seed)
    # Orthogonal class centers on the hypersphere, scaled for clear separation
    centers = F.normalize(torch.eye(_NUM_CLASSES, _EMBED_DIM), dim=-1) * 5.0

    def _cluster(center, n, noise=0.05):
        return F.normalize(center + noise * torch.randn(n, _EMBED_DIM), dim=-1)

    train_e = torch.cat([_cluster(centers[c], _TRAIN_PER_CLASS) for c in range(_NUM_CLASSES)])
    train_l = torch.arange(_NUM_CLASSES).repeat_interleave(_TRAIN_PER_CLASS)

    test_e = torch.cat([_cluster(centers[c], _TEST_PER_CLASS) for c in range(_NUM_CLASSES)])
    test_l = torch.arange(_NUM_CLASSES).repeat_interleave(_TEST_PER_CLASS)

    perm = torch.randperm(len(train_e))
    return train_e[perm], train_l[perm], test_e, test_l


# ──────────────────────────── _mine_hard_negatives ────────────────────────────

class TestMineHardNegatives:
    def test_returns_valid_triplets(self):
        torch.manual_seed(42)
        B = 32
        feats = F.normalize(torch.randn(B, _EMBED_DIM), dim=-1)
        labels = torch.arange(_NUM_CLASSES).repeat(B // _NUM_CLASSES + 1)[:B]
        proj = feats  # already normalised

        anch, pos, neg = _mine_hard_negatives(proj, feats, labels, labels)
        T = anch.shape[0]
        assert T > 0, "Expected at least one valid triplet"
        assert pos.shape == (T, _EMBED_DIM)
        assert neg.shape == (T, _EMBED_DIM)

    def test_returns_empty_when_single_class(self):
        """When all samples belong to the same class no triplets can be formed."""
        B = 8
        feats = F.normalize(torch.randn(B, _EMBED_DIM), dim=-1)
        labels = torch.zeros(B, dtype=torch.long)

        anch, pos, neg = _mine_hard_negatives(feats, feats, labels, labels)
        assert anch.shape[0] == 0
        assert pos.shape[0] == 0
        assert neg.shape[0] == 0

    def test_shapes_consistent(self):
        torch.manual_seed(7)
        B = 16
        feats = F.normalize(torch.randn(B, _EMBED_DIM), dim=-1)
        labels = torch.arange(B) % 4

        anch, pos, neg = _mine_hard_negatives(feats, feats, labels, labels)
        assert anch.shape[0] == pos.shape[0] == neg.shape[0]
        assert anch.shape[1] == _EMBED_DIM


# ─────────────────────────────── train_adapter ────────────────────────────────

class TestTrainAdapter:
    @pytest.fixture(scope="class")
    def synthetic_data(self):
        return _make_synthetic_data(seed=1)

    def test_returns_metric_adapter(self, synthetic_data, tmp_path):
        train_e, train_l, _, _ = synthetic_data
        output = str(tmp_path / "adapter_test.pt")
        result = train_adapter(
            train_embeds=train_e,
            train_labels=train_l,
            epochs=1,
            batch_size=64,
            lr=1e-4,
            margin=0.2,
            output=output,
            device=torch.device("cpu"),
        )
        assert isinstance(result, MetricAdapter)

    def test_weights_saved(self, synthetic_data, tmp_path):
        train_e, train_l, _, _ = synthetic_data
        output = str(tmp_path / "adapter_out.pt")
        train_adapter(
            train_embeds=train_e,
            train_labels=train_l,
            epochs=1,
            batch_size=64,
            lr=1e-4,
            margin=0.2,
            output=output,
            device=torch.device("cpu"),
        )
        assert os.path.isfile(output), f"Expected adapter weights at {output}"

    def test_adapter_dimensions(self, synthetic_data, tmp_path):
        """Adapter output_dim should be 256, input_dim matches embedding."""
        train_e, train_l, _, _ = synthetic_data
        output = str(tmp_path / "adapter_dim.pt")
        adapter = train_adapter(
            train_embeds=train_e,
            train_labels=train_l,
            epochs=1,
            batch_size=64,
            lr=1e-4,
            margin=0.2,
            output=output,
            device=torch.device("cpu"),
        )
        assert adapter.input_dim == _EMBED_DIM
        assert adapter.output_dim == 256

    def test_adapter_output_normalised(self, synthetic_data, tmp_path):
        """Adapter forward output should be L2-normalised."""
        train_e, train_l, _, _ = synthetic_data
        output = str(tmp_path / "adapter_norm.pt")
        adapter = train_adapter(
            train_embeds=train_e,
            train_labels=train_l,
            epochs=1,
            batch_size=64,
            lr=1e-4,
            margin=0.2,
            output=output,
            device=torch.device("cpu"),
        )
        adapter.eval()
        with torch.no_grad():
            x = F.normalize(torch.randn(8, _EMBED_DIM), dim=-1)
            out = adapter(x)
            norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ──────────────────────────────── eval_full_stack ─────────────────────────────

class TestEvalFullStack:
    @pytest.fixture(scope="class")
    def synthetic_data(self):
        return _make_synthetic_data(seed=2)

    def test_returns_tuple_of_three(self, synthetic_data):
        train_e, train_l, test_e, test_l = synthetic_data
        result = eval_full_stack(
            train_e, train_l, test_e, test_l,
            num_classes=_NUM_CLASSES,
            device=torch.device("cpu"),
            adapter=None,
            core_vigilance=0.7,
            core_entries=_TOTAL_TRAIN * 2,
        )
        assert len(result) == 3

    def test_accuracy_in_valid_range(self, synthetic_data):
        train_e, train_l, test_e, test_l = synthetic_data
        acc, protos, cond = eval_full_stack(
            train_e, train_l, test_e, test_l,
            num_classes=_NUM_CLASSES,
            device=torch.device("cpu"),
            adapter=None,
            core_vigilance=0.7,
            core_entries=_TOTAL_TRAIN * 2,
        )
        assert 0.0 <= acc <= 100.0
        assert protos >= 0
        assert cond >= 0.0

    def test_condensation_ratio_formula(self, synthetic_data):
        """condensation_ratio = num_prototypes / num_train_samples."""
        train_e, train_l, test_e, test_l = synthetic_data
        acc, protos, cond = eval_full_stack(
            train_e, train_l, test_e, test_l,
            num_classes=_NUM_CLASSES,
            device=torch.device("cpu"),
            adapter=None,
            core_vigilance=0.7,
            core_entries=_TOTAL_TRAIN * 2,
        )
        expected_cond = protos / _TOTAL_TRAIN
        assert abs(cond - expected_cond) < 1e-6, (
            f"Expected cond={expected_cond:.4f}, got {cond:.4f}"
        )

    def test_with_adapter(self, synthetic_data, tmp_path):
        """eval_full_stack should work with a MetricAdapter."""
        train_e, train_l, test_e, test_l = synthetic_data
        # Use an untrained adapter with matching input_dim
        adapter = MetricAdapter(input_dim=_EMBED_DIM, output_dim=32, hidden_dim=64)
        adapter.eval()
        acc, protos, cond = eval_full_stack(
            train_e, train_l, test_e, test_l,
            num_classes=_NUM_CLASSES,
            device=torch.device("cpu"),
            adapter=adapter,
            core_vigilance=0.7,
            core_entries=_TOTAL_TRAIN * 2,
        )
        assert 0.0 <= acc <= 100.0

    def test_high_accuracy_on_separable_data(self, synthetic_data):
        """FAM should classify well-separated data accurately."""
        train_e, train_l, test_e, test_l = synthetic_data
        acc, _, _ = eval_full_stack(
            train_e, train_l, test_e, test_l,
            num_classes=_NUM_CLASSES,
            device=torch.device("cpu"),
            adapter=None,
            core_vigilance=0.7,
            core_entries=_TOTAL_TRAIN * 2,
        )
        assert acc >= 80.0, f"Expected ≥80% on separable data, got {acc:.1f}%"


# ───────────────────────────────── _format_results ────────────────────────────

class TestFormatResults:
    def _sample_rows(self):
        return [
            ("baseline (no adapter, v=0.92)", 55.0, 300, 0.90),
            ("aircraft adapter (v=0.92)", 60.0, 280, 0.85),
            ("aircraft adapter (v=0.85)", 65.0, 250, 0.75),
            ("aircraft adapter (v=0.82)", 63.0, 240, 0.72),
            ("aircraft adapter (v=0.80)", 61.0, 230, 0.70),
        ]

    def test_returns_string(self):
        table = _format_results(self._sample_rows(), best_aircraft_acc=65.0)
        assert isinstance(table, str)

    def test_contains_g24_header(self):
        table = _format_results(self._sample_rows(), best_aircraft_acc=65.0)
        assert "G24" in table

    def test_contains_config_labels(self):
        table = _format_results(self._sample_rows(), best_aircraft_acc=65.0)
        assert "baseline" in table
        assert "aircraft adapter" in table

    def test_pass_when_aircraft_above_reference(self):
        """Aircraft best > Dogs reference (99.71) → PASS."""
        # Use a dogs_reference well below 65%
        table = _format_results(self._sample_rows(), best_aircraft_acc=65.0,
                                dogs_reference=50.0)
        assert "PASS" in table
        assert "KILL" not in table

    def test_kill_when_aircraft_below_reference(self):
        """Aircraft best < Dogs reference → KILL."""
        table = _format_results(self._sample_rows(), best_aircraft_acc=65.0,
                                dogs_reference=G11_DOGS_REFERENCE_ACC)
        assert "KILL" in table

    def test_exactly_at_reference_is_pass(self):
        """Aircraft best == Dogs reference is not strictly less → PASS."""
        table = _format_results(self._sample_rows(), best_aircraft_acc=99.71,
                                dogs_reference=99.71)
        assert "PASS" in table

    def test_dogs_reference_shown(self):
        table = _format_results(self._sample_rows(), best_aircraft_acc=65.0)
        assert "99.71" in table


# ──────────────────────── extract_or_load_features caching ───────────────────

class TestExtractOrLoadFeaturesCache:
    def test_loads_from_cache(self, tmp_path):
        """When cache files exist, features are loaded without DINOv2 extraction."""
        train_e = torch.randn(50, 1024).float()
        train_l = torch.arange(50).long()
        test_e = torch.randn(20, 1024).float()
        test_l = torch.arange(20).long()

        torch.save({"embeds": train_e, "labels": train_l},
                   tmp_path / CACHE_TRAIN)
        torch.save({"embeds": test_e, "labels": test_l},
                   tmp_path / CACHE_TEST)

        te, tl, se, sl = extract_or_load_features(
            root=str(tmp_path),
            batch_size=32,
            device=torch.device("cpu"),
        )
        assert te.shape == (50, 1024)
        assert tl.shape == (50,)
        assert se.shape == (20, 1024)
        assert sl.shape == (20,)
        assert te.dtype == torch.float32
        assert tl.dtype == torch.long

    def test_cache_returns_correct_values(self, tmp_path):
        """Loaded values must match the saved tensors exactly."""
        train_e = torch.ones(10, 1024).float() * 0.5
        train_l = torch.zeros(10, dtype=torch.long)
        test_e = torch.ones(5, 1024).float() * 0.25
        test_l = torch.ones(5, dtype=torch.long)

        torch.save({"embeds": train_e, "labels": train_l},
                   tmp_path / CACHE_TRAIN)
        torch.save({"embeds": test_e, "labels": test_l},
                   tmp_path / CACHE_TEST)

        te, tl, se, sl = extract_or_load_features(
            root=str(tmp_path),
            batch_size=32,
            device=torch.device("cpu"),
        )
        assert torch.allclose(te, train_e)
        assert torch.equal(tl, train_l)
        assert torch.allclose(se, test_e)
        assert torch.equal(sl, test_l)


# ──────────────────────────── kill-condition constant ─────────────────────────

class TestKillConditionConstant:
    def test_dogs_reference_value(self):
        """G11 Dogs reference must be exactly 99.71%."""
        assert G11_DOGS_REFERENCE_ACC == pytest.approx(99.71)

    def test_kill_condition_logic(self):
        """best < reference → kill; best >= reference → pass."""
        assert 65.0 < G11_DOGS_REFERENCE_ACC   # typical aircraft score → kill
        assert 99.71 >= G11_DOGS_REFERENCE_ACC  # at reference → pass
        assert 100.0 >= G11_DOGS_REFERENCE_ACC  # above reference → pass
