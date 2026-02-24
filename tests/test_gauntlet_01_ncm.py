"""
tests/test_gauntlet_01_ncm.py — Unit tests for Gauntlet 1 NCM baselines.

Tests StreamingClassMean and NCMMahalanobis for:
  - Correct incremental mean / variance updates
  - Cosine-similarity and Mahalanobis prediction
  - Memory accounting
  - Reproducibility under fixed seed
"""

import sys
import os

import pytest
import torch
import torch.nn.functional as F

# Allow import from repo root regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.gauntlet_01_ncm import (
    StreamingClassMean,
    NCMMahalanobis,
    _format_table,
)

torch.manual_seed(0)

DIM = 64


# ---------------------------------------------------------------------------
# StreamingClassMean
# ---------------------------------------------------------------------------

class TestStreamingClassMean:
    def test_first_sample_stored_as_mean(self):
        scm = StreamingClassMean()
        emb = torch.randn(DIM)
        scm.update(emb, class_id=0)
        assert torch.allclose(scm.means[0], emb)
        assert scm.counts[0] == 1

    def test_running_mean_matches_torch(self):
        scm = StreamingClassMean()
        samples = torch.randn(10, DIM)
        for s in samples:
            scm.update(s, class_id=7)
        expected = samples.mean(0)
        assert torch.allclose(scm.means[7], expected, atol=1e-5)
        assert scm.counts[7] == 10

    def test_multiple_classes_independent(self):
        scm = StreamingClassMean()
        for i in range(5):
            scm.update(torch.randn(DIM), class_id=i)
        assert set(scm.means.keys()) == set(range(5))
        for i in range(5):
            assert scm.counts[i] == 1

    def test_predict_returns_nearest_class(self):
        scm = StreamingClassMean()
        # Two well-separated class means
        scm.means[0] = torch.zeros(DIM)
        scm.means[1] = torch.zeros(DIM)
        scm.means[0][0] = 1.0   # class 0: positive x-axis
        scm.means[1][0] = -1.0  # class 1: negative x-axis
        scm.counts = {0: 1, 1: 1}

        # Query close to class 0
        q = torch.tensor([0.9] + [0.0] * (DIM - 1)).unsqueeze(0)
        pred = scm.predict_batch(q)
        assert pred.item() == 0

        # Query close to class 1
        q = torch.tensor([-0.9] + [0.0] * (DIM - 1)).unsqueeze(0)
        pred = scm.predict_batch(q)
        assert pred.item() == 1

    def test_memory_bytes(self):
        scm = StreamingClassMean()
        for c in range(10):
            scm.update(torch.randn(DIM), class_id=c)
        expected = 10 * DIM * 4  # 10 classes × DIM dims × float32
        assert scm.memory_bytes() == expected

    def test_memory_bytes_empty(self):
        scm = StreamingClassMean()
        assert scm.memory_bytes() == 0

    def test_predict_batch_multiple_queries(self):
        scm = StreamingClassMean()
        # Distinct orthogonal unit vectors as class means
        means = F.normalize(torch.eye(4, DIM), dim=-1)
        for c, m in enumerate(means):
            scm.means[c] = m.clone()
            scm.counts[c] = 1
        # Queries: slightly noisy versions of each mean
        queries = means + 0.01 * torch.randn(4, DIM)
        preds = scm.predict_batch(queries)
        assert (preds == torch.arange(4)).all()


# ---------------------------------------------------------------------------
# NCMMahalanobis
# ---------------------------------------------------------------------------

class TestNCMMahalanobis:
    def test_first_sample_sets_mean(self):
        ncm = NCMMahalanobis()
        emb = torch.randn(DIM)
        ncm.update(emb, class_id=3)
        assert torch.allclose(ncm.means[3], emb)
        assert ncm.counts[3] == 1

    def test_welford_mean_matches_torch(self):
        ncm = NCMMahalanobis()
        samples = torch.randn(20, DIM)
        for s in samples:
            ncm.update(s, class_id=0)
        expected_mean = samples.mean(0)
        assert torch.allclose(ncm.means[0], expected_mean, atol=1e-4)

    def test_welford_variance_matches_torch(self):
        ncm = NCMMahalanobis()
        samples = torch.randn(50, DIM)
        for s in samples:
            ncm.update(s, class_id=0)
        # Bessel-corrected variance (n-1 denominator)
        expected_var = samples.var(dim=0, unbiased=True)
        actual_var = ncm._variance(0)
        assert torch.allclose(actual_var, expected_var.clamp(min=ncm.var_eps), atol=1e-4)

    def test_single_sample_variance_falls_back_to_ones(self):
        ncm = NCMMahalanobis()
        ncm.update(torch.randn(DIM), class_id=5)
        var = ncm._variance(5)
        assert torch.allclose(var, torch.ones(DIM))

    def test_predict_selects_closest_class(self):
        ncm = NCMMahalanobis()
        # Class 0: tight cluster at +e0
        for _ in range(30):
            ncm.update(torch.tensor([1.0] + [0.0] * (DIM - 1)) + 0.01 * torch.randn(DIM),
                       class_id=0)
        # Class 1: tight cluster at -e0
        for _ in range(30):
            ncm.update(torch.tensor([-1.0] + [0.0] * (DIM - 1)) + 0.01 * torch.randn(DIM),
                       class_id=1)

        q0 = torch.tensor([[1.0] + [0.0] * (DIM - 1)])
        q1 = torch.tensor([[-1.0] + [0.0] * (DIM - 1)])
        assert ncm.predict_batch(q0).item() == 0
        assert ncm.predict_batch(q1).item() == 1

    def test_memory_bytes(self):
        ncm = NCMMahalanobis()
        for c in range(5):
            for _ in range(3):
                ncm.update(torch.randn(DIM), class_id=c)
        expected = 5 * 2 * DIM * 4  # 5 classes × (mean + M2) × DIM × float32
        assert ncm.memory_bytes() == expected

    def test_memory_bytes_empty(self):
        ncm = NCMMahalanobis()
        assert ncm.memory_bytes() == 0

    def test_multiple_classes_independent(self):
        ncm = NCMMahalanobis()
        for c in range(4):
            for _ in range(5):
                ncm.update(torch.randn(DIM), class_id=c)
        assert set(ncm.means.keys()) == set(range(4))
        for c in range(4):
            assert ncm.counts[c] == 5

    def test_predict_batch_multiple_queries(self):
        ncm = NCMMahalanobis()
        # Tight, well-separated clusters
        for c in range(3):
            center = torch.zeros(DIM)
            center[c] = 5.0
            for _ in range(20):
                ncm.update(center + 0.05 * torch.randn(DIM), class_id=c)
        queries = torch.eye(3, DIM) * 5.0
        preds = ncm.predict_batch(queries)
        assert (preds == torch.arange(3)).all()


# ---------------------------------------------------------------------------
# _format_table
# ---------------------------------------------------------------------------

class TestFormatTable:
    def _sample_results(self):
        return [
            ("SCM (Streaming Class Mean)", 72.50, 0.80),
            ("NCM + Mahalanobis (diagonal)", 71.00, 1.60),
            ("FAM (multi-prototype, 1000 slots)", 75.00, 200.0),
        ]

    def test_returns_string(self):
        table = _format_table("imagenet_r", "dinov2_vitl14",
                              self._sample_results(), 72.50, 71.00, 75.00)
        assert isinstance(table, str)

    def test_contains_method_names(self):
        table = _format_table("imagenet_r", "dinov2_vitl14",
                              self._sample_results(), 72.50, 71.00, 75.00)
        assert "SCM" in table
        assert "Mahalanobis" in table
        assert "FAM" in table

    def test_kill_condition_not_met(self):
        # delta = 75 - 72.5 = 2.5 > 1.0 → passes
        table = _format_table("imagenet_r", "dinov2_vitl14",
                              self._sample_results(), 72.50, 71.00, 75.00)
        assert "KILL CONDITION" not in table
        assert "\u2713" in table  # checkmark

    def test_kill_condition_met(self):
        # FAM only 0.5% above best NCM → kill condition fires
        results = [
            ("SCM (Streaming Class Mean)", 72.50, 0.80),
            ("NCM + Mahalanobis (diagonal)", 71.00, 1.60),
            ("FAM (multi-prototype, 1000 slots)", 73.00, 200.0),
        ]
        table = _format_table("imagenet_r", "dinov2_vitl14",
                              results, 72.50, 71.00, 73.00)
        assert "KILL CONDITION MET" in table
        assert "\u26a0" in table  # warning sign

    def test_kill_condition_exactly_at_boundary(self):
        # delta = exactly 1.0% → kill condition fires (FAM ≤ NCM + 1.0%)
        results = [
            ("SCM", 72.00, 0.80),
            ("NCM", 72.00, 1.60),
            ("FAM", 73.00, 200.0),
        ]
        table = _format_table("imagenet_r", "dinov2_vitl14",
                              results, 72.00, 72.00, 73.00)
        assert "KILL CONDITION MET" in table
