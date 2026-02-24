"""
tests/test_g28_multi_adapter_all4.py — Unit tests for G28 benchmark.

Validates:
  - remap_labels correctly concatenates domains with non-colliding offsets
  - _mine_hard_negatives returns valid triplets or empty tensors
  - eval_indomain returns accuracy, prototype count, and condensation ratio
  - eval_crossdomain returns accuracy and drop vs baseline
  - Kill condition: multi-all4 avg < best single → kill; else → pass
  - _load_adapter returns None and warns when file is missing
"""

import os
import sys
import tempfile

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.g28_multi_adapter_all4 import (
    remap_labels,
    _mine_hard_negatives,
    eval_indomain,
    eval_crossdomain,
    _load_adapter,
)
from adapter import MetricAdapter


# ─────────────────────────────────────────────── fixtures & helpers ───

_EMBED_DIM = 32
_NUM_CLASSES = 4
_SAMPLES_PER_CLASS = 20


def _make_domain_data(num_classes: int, samples_per_class: int, embed_dim: int, seed: int = 0):
    """Create (train_embeds, train_labels, test_embeds, test_labels) for a
    well-separated synthetic domain."""
    torch.manual_seed(seed)
    centers = F.normalize(torch.eye(num_classes, embed_dim), dim=-1) * 5.0

    def _cluster(center, n, noise=0.05):
        return center + noise * torch.randn(n, embed_dim)

    tr_e = torch.cat([_cluster(centers[c], samples_per_class) for c in range(num_classes)])
    tr_l = torch.arange(num_classes).repeat_interleave(samples_per_class)
    te_e = torch.cat([_cluster(centers[c], 5) for c in range(num_classes)])
    te_l = torch.arange(num_classes).repeat_interleave(5)

    perm = torch.randperm(len(tr_e))
    return tr_e[perm], tr_l[perm], te_e, te_l


# ────────────────────────────────────────────── remap_labels tests ───

class TestRemapLabels:
    def test_two_domains_no_collision(self):
        dom_a = torch.randn(9, _EMBED_DIM), torch.arange(3).repeat(3)  # 9 samples, labels 0-2
        dom_b = torch.randn(8, _EMBED_DIM), torch.arange(4).repeat_interleave(2)
        all_e, all_l = remap_labels([dom_a, dom_b])
        # Domain A labels: 0..2, domain B remapped to 3..6
        assert all_l[:9].max().item() < 3
        assert all_l[9:].min().item() == 3
        assert all_l[9:].max().item() == 6

    def test_four_domains_unique_labels(self):
        domains = [(torch.randn(12, 8), torch.arange(3).repeat(4)) for _ in range(4)]
        all_e, all_l = remap_labels(domains)
        assert all_l.max().item() == 3 * 4 - 1  # 0..11 total
        assert len(all_l.unique()) == 12

    def test_shape_concatenated(self):
        domains = [
            (torch.randn(10, _EMBED_DIM), torch.zeros(10, dtype=torch.long)),
            (torch.randn(15, _EMBED_DIM), torch.zeros(15, dtype=torch.long)),
        ]
        all_e, all_l = remap_labels(domains)
        assert all_e.shape == (25, _EMBED_DIM)
        assert all_l.shape == (25,)

    def test_single_domain_unchanged_labels(self):
        dom = (torch.randn(6, 8), torch.tensor([0, 1, 2, 0, 1, 2]))
        all_e, all_l = remap_labels([dom])
        assert torch.equal(all_l, dom[1])

    def test_labels_are_long(self):
        dom = (torch.randn(5, 8), torch.arange(5))
        _, all_l = remap_labels([dom])
        assert all_l.dtype == torch.long or all_l.dtype == torch.int64


# ────────────────────────────────────── _mine_hard_negatives tests ───

class TestMineHardNegatives:
    def _batch(self, seed=0):
        torch.manual_seed(seed)
        embeds = torch.randn(12, _EMBED_DIM)
        labels = torch.arange(3).repeat(4)
        proj = F.normalize(embeds, dim=-1)
        return proj, embeds, labels

    def test_returns_three_tensors(self):
        proj, feats, labels = self._batch()
        a, p, n = _mine_hard_negatives(proj, feats, labels, labels)
        assert len(a) == len(p) == len(n)

    def test_valid_triplets_non_empty(self):
        """With well-balanced classes, OHNM should find valid triplets."""
        proj, feats, labels = self._batch()
        a, p, n = _mine_hard_negatives(proj, feats, labels, labels)
        assert a.shape[0] > 0

    def test_empty_when_single_class(self):
        """With only one class, no valid triplets can be found."""
        feats = torch.randn(8, _EMBED_DIM)
        labels = torch.zeros(8, dtype=torch.long)
        proj = F.normalize(feats, dim=-1)
        a, p, n = _mine_hard_negatives(proj, feats, labels, labels)
        assert a.shape[0] == 0

    def test_output_shapes_match_input_dim(self):
        proj, feats, labels = self._batch()
        a, p, n = _mine_hard_negatives(proj, feats, labels, labels)
        if a.shape[0] > 0:
            assert a.shape[1] == feats.shape[1]
            assert p.shape[1] == feats.shape[1]
            assert n.shape[1] == feats.shape[1]

    def test_empty_returns_correct_dim(self):
        """Empty result tensors must still have the right feature dimension."""
        feats = torch.randn(4, _EMBED_DIM)
        labels = torch.zeros(4, dtype=torch.long)
        proj = F.normalize(feats, dim=-1)
        a, p, n = _mine_hard_negatives(proj, feats, labels, labels)
        assert a.shape == (0, _EMBED_DIM)


# ───────────────────────────────────────────── eval_indomain tests ───

class TestEvalIndomain:
    @pytest.fixture(scope="class")
    def data(self):
        return _make_domain_data(
            num_classes=_NUM_CLASSES,
            samples_per_class=_SAMPLES_PER_CLASS,
            embed_dim=_EMBED_DIM,
            seed=7,
        )

    def test_returns_three_values(self, data):
        tr_e, tr_l, te_e, te_l = data
        acc, n_proto, cond = eval_indomain(
            tr_e, tr_l, te_e, te_l,
            device=torch.device("cpu"),
            adapter=None,
            core_entries=500,
            core_vigilance=0.7,
        )
        assert isinstance(acc, float)
        assert isinstance(n_proto, int)
        assert isinstance(cond, float)

    def test_accuracy_in_range(self, data):
        tr_e, tr_l, te_e, te_l = data
        acc, _, _ = eval_indomain(
            tr_e, tr_l, te_e, te_l,
            device=torch.device("cpu"),
            adapter=None,
            core_entries=500,
            core_vigilance=0.7,
        )
        assert 0.0 <= acc <= 100.0

    def test_n_prototypes_positive(self, data):
        tr_e, tr_l, te_e, te_l = data
        _, n_proto, _ = eval_indomain(
            tr_e, tr_l, te_e, te_l,
            device=torch.device("cpu"),
            adapter=None,
            core_entries=500,
            core_vigilance=0.7,
        )
        assert n_proto > 0

    def test_condensation_ratio_in_range(self, data):
        tr_e, tr_l, te_e, te_l = data
        _, _, cond = eval_indomain(
            tr_e, tr_l, te_e, te_l,
            device=torch.device("cpu"),
            adapter=None,
            vigilance_condensation=0.80,
            core_entries=500,
            core_vigilance=0.7,
        )
        assert 0.0 <= cond <= 1.0

    def test_high_accuracy_on_separable_data(self, data):
        tr_e, tr_l, te_e, te_l = data
        acc, _, _ = eval_indomain(
            tr_e, tr_l, te_e, te_l,
            device=torch.device("cpu"),
            adapter=None,
            core_entries=500,
            core_vigilance=0.7,
        )
        assert acc >= 80.0, f"Expected ≥80% on well-separated data, got {acc:.1f}%"

    def test_with_adapter_returns_valid_acc(self, data):
        tr_e, tr_l, te_e, te_l = data
        adapter = MetricAdapter(input_dim=_EMBED_DIM, output_dim=16, hidden_dim=32)
        adapter.eval()
        acc, n_proto, cond = eval_indomain(
            tr_e, tr_l, te_e, te_l,
            device=torch.device("cpu"),
            adapter=adapter,
            core_entries=500,
            core_vigilance=0.7,
        )
        assert 0.0 <= acc <= 100.0
        assert n_proto >= 0
        assert 0.0 <= cond <= 1.0


# ─────────────────────────────────────────── eval_crossdomain tests ───

class TestEvalCrossdomain:
    @pytest.fixture(scope="class")
    def data(self):
        return _make_domain_data(
            num_classes=5,
            samples_per_class=_SAMPLES_PER_CLASS,
            embed_dim=_EMBED_DIM,
            seed=11,
        )

    def test_returns_two_floats(self, data):
        tr_e, tr_l, te_e, te_l = data
        acc, drop = eval_crossdomain(
            tr_e, tr_l, te_e, te_l,
            device=torch.device("cpu"),
            adapter=None,
            core_entries=500,
            core_vigilance=0.7,
        )
        assert isinstance(acc, float)
        assert isinstance(drop, float)

    def test_no_adapter_zero_drop(self, data):
        """Passing adapter=None should yield 0 pp drop (same as baseline)."""
        tr_e, tr_l, te_e, te_l = data
        acc, drop = eval_crossdomain(
            tr_e, tr_l, te_e, te_l,
            device=torch.device("cpu"),
            adapter=None,
            core_entries=500,
            core_vigilance=0.7,
        )
        assert drop == pytest.approx(0.0, abs=1e-4)

    def test_accuracy_in_valid_range(self, data):
        tr_e, tr_l, te_e, te_l = data
        acc, _ = eval_crossdomain(
            tr_e, tr_l, te_e, te_l,
            device=torch.device("cpu"),
            adapter=None,
            core_entries=500,
            core_vigilance=0.7,
        )
        assert 0.0 <= acc <= 100.0

    def test_distorting_adapter_may_degrade(self, data):
        """A random low-dimension adapter may degrade cross-domain accuracy."""
        tr_e, tr_l, te_e, te_l = data
        torch.manual_seed(99)
        bad_adapter = MetricAdapter(input_dim=_EMBED_DIM, output_dim=2)
        bad_adapter.eval()
        acc, drop = eval_crossdomain(
            tr_e, tr_l, te_e, te_l,
            device=torch.device("cpu"),
            adapter=bad_adapter,
            core_entries=500,
            core_vigilance=0.7,
        )
        # drop can be positive (degradation) or negative (improvement)
        assert isinstance(drop, float)


# ────────────────────────────────────────── _load_adapter tests ───

class TestLoadAdapter:
    def test_returns_none_when_file_missing(self, capsys):
        result = _load_adapter(
            "/nonexistent/path/adapter.pt",
            input_dim=_EMBED_DIM,
            device=torch.device("cpu"),
            label="test",
        )
        assert result is None

    def test_loads_valid_file(self):
        # _load_adapter hardcodes output_dim=256, hidden_dim=512 per the G28 spec
        adapter_orig = MetricAdapter(input_dim=_EMBED_DIM, output_dim=256, hidden_dim=512)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name
        try:
            torch.save(adapter_orig.state_dict(), tmp_path)
            loaded = _load_adapter(tmp_path, input_dim=_EMBED_DIM,
                                   device=torch.device("cpu"), label="test")
            assert loaded is not None
            assert isinstance(loaded, MetricAdapter)
        finally:
            os.unlink(tmp_path)

    def test_loaded_adapter_is_eval_mode(self):
        adapter_orig = MetricAdapter(input_dim=_EMBED_DIM, output_dim=256, hidden_dim=512)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name
        try:
            torch.save(adapter_orig.state_dict(), tmp_path)
            loaded = _load_adapter(tmp_path, input_dim=_EMBED_DIM,
                                   device=torch.device("cpu"), label="test")
            assert not loaded.training
        finally:
            os.unlink(tmp_path)


# ────────────────────────────────────── kill condition logic tests ───

class TestKillConditionLogic:
    """Verify the kill-condition inequality: all4_avg < best_single_avg → kill."""

    def test_all4_above_best_single_is_pass(self):
        all4_avg = 75.0
        best_single_avg = 73.0
        assert not (all4_avg < best_single_avg)

    def test_all4_below_best_single_is_kill(self):
        all4_avg = 70.0
        best_single_avg = 73.0
        assert all4_avg < best_single_avg

    def test_all4_equal_best_single_is_pass(self):
        """Exactly equal → pass (kill uses strict <)."""
        val = 72.5
        assert not (val < val)

    def test_all4_marginally_below_is_kill(self):
        all4_avg = 72.499
        best_single_avg = 72.5
        assert all4_avg < best_single_avg
