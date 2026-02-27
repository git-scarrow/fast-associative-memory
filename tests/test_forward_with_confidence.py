"""
tests/test_forward_with_confidence.py — Tests for FAM.forward_with_confidence().

Validates:
  - Output shapes and value ranges
  - Outputs are identical to forward() (same predictions)
  - Confidence is in [0, 1]
  - Higher accuracy queries tend toward higher confidence (calibration sanity)
  - Works with and without NSTP
  - Works with batch size 1 and larger batches
  - Works when store is empty (flood mode)
"""

import torch
import torch.nn.functional as F
import pytest

from fast_associative_memory import FastAssociativeMemory
from nstp import NSTPController


def _make_fam(n_classes=10, entries=200, with_nstp=False):
    nstp = NSTPController(sibling_threshold=0.85, depth_epsilon=0.10) if with_nstp else None
    return FastAssociativeMemory(
        input_dim=64,
        value_dim=n_classes,
        core_entries=entries,
        core_vigilance=0.85,
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=25,
        inference_temp=0.05,
        nstp=nstp,
    )


def _populate(fam, n_classes=10, n_per_class=20, dim=64, seed=0):
    torch.manual_seed(seed)
    for c in range(n_classes):
        feats = F.normalize(torch.randn(n_per_class, dim) + torch.randn(dim) * 3, dim=-1)
        labels = torch.full((n_per_class,), c, dtype=torch.long)
        fam.learn_local(feats, labels)


class TestForwardWithConfidenceShape:
    def test_output_shape(self):
        fam = _make_fam()
        _populate(fam)
        x = torch.randn(8, 64)
        outputs, conf = fam.forward_with_confidence(x)
        assert outputs.shape == (8, 10)
        assert conf.shape == (8,)

    def test_batch_size_1(self):
        fam = _make_fam()
        _populate(fam)
        x = torch.randn(1, 64)
        outputs, conf = fam.forward_with_confidence(x)
        assert outputs.shape == (1, 10)
        assert conf.shape == (1,)

    def test_confidence_in_unit_interval(self):
        fam = _make_fam()
        _populate(fam)
        x = torch.randn(32, 64)
        _, conf = fam.forward_with_confidence(x)
        assert (conf >= 0.0).all(), "confidence must be >= 0"
        assert (conf <= 1.0).all(), "confidence must be <= 1"


class TestForwardWithConfidenceConsistency:
    def test_outputs_match_forward(self):
        """forward_with_confidence outputs must be identical to forward()."""
        fam = _make_fam()
        _populate(fam)
        torch.manual_seed(1)
        x = torch.randn(16, 64)
        out_plain = fam.forward(x)
        out_conf, _ = fam.forward_with_confidence(x)
        assert torch.allclose(out_plain, out_conf, atol=1e-6), \
            "outputs differ between forward() and forward_with_confidence()"

    def test_predictions_match_forward(self):
        """Argmax predictions must be identical."""
        fam = _make_fam()
        _populate(fam)
        x = torch.randn(32, 64)
        preds_plain = fam.forward(x).argmax(dim=-1)
        preds_conf = fam.forward_with_confidence(x)[0].argmax(dim=-1)
        assert torch.equal(preds_plain, preds_conf)


class TestForwardWithConfidenceCalibration:
    def test_exact_prototype_has_higher_confidence_than_boundary(self):
        """A query identical to a stored prototype should have strictly higher
        confidence than a query halfway between two well-separated class centroids.

        This is a controlled contrast that doesn't depend on random geometry:
        the on-prototype query has a perfectly concentrated rejection histogram
        (maximum RVA and RE_inv), while the midpoint query has a mixed histogram
        (low RVA, high entropy → low RE_inv).
        """
        torch.manual_seed(7)
        dim = 64
        n_classes = 4
        fam = _make_fam(n_classes=n_classes, entries=200)

        # Build two maximally separated class centroids along orthogonal axes
        centroid_a = torch.zeros(dim)
        centroid_a[0] = 1.0  # unit vector along axis 0
        centroid_b = torch.zeros(dim)
        centroid_b[1] = 1.0  # unit vector along axis 1

        # Populate: many prototypes tightly around each centroid
        for _ in range(30):
            fa = F.normalize(centroid_a + torch.randn(dim) * 0.05, dim=-1)
            fb = F.normalize(centroid_b + torch.randn(dim) * 0.05, dim=-1)
            fam.learn_local(fa.unsqueeze(0), torch.tensor([0]))
            fam.learn_local(fb.unsqueeze(0), torch.tensor([1]))

        # Query 1: exactly on centroid_a (should be high confidence — rejection set
        # dominated by class-0 prototypes that didn't make top-25).
        q_on = centroid_a.unsqueeze(0)
        _, conf_on = fam.forward_with_confidence(q_on)

        # Query 2: midpoint between the two centroids (should be low confidence —
        # rejection set has mixed class-0 and class-1 prototypes).
        q_mid = F.normalize((centroid_a + centroid_b), dim=-1).unsqueeze(0)
        _, conf_mid = fam.forward_with_confidence(q_mid)

        assert conf_on.item() > conf_mid.item(), (
            f"On-prototype confidence ({conf_on.item():.3f}) should exceed "
            f"boundary confidence ({conf_mid.item():.3f})"
        )


class TestForwardWithConfidenceEdgeCases:
    def test_empty_store(self):
        """Empty store returns flood noise; confidence should still be valid scalars."""
        fam = _make_fam()
        x = torch.randn(4, 64)
        outputs, conf = fam.forward_with_confidence(x)
        assert outputs.shape == (4, 10)
        assert conf.shape == (4,)
        assert not conf.isnan().any()
        assert not conf.isinf().any()

    def test_with_nstp(self):
        """forward_with_confidence works correctly when NSTP is active."""
        fam = _make_fam(with_nstp=True)
        _populate(fam)
        x = torch.randn(8, 64)
        outputs, conf = fam.forward_with_confidence(x)
        assert outputs.shape == (8, 10)
        assert conf.shape == (8,)
        assert (conf >= 0.0).all()
        assert (conf <= 1.0).all()

    def test_no_nan_or_inf(self):
        """Confidence is never NaN or Inf under any normal conditions."""
        fam = _make_fam()
        _populate(fam)
        x = torch.randn(64, 64)
        outputs, conf = fam.forward_with_confidence(x)
        assert not conf.isnan().any(), "confidence contains NaN"
        assert not conf.isinf().any(), "confidence contains Inf"
        assert not outputs.isnan().any()

    def test_single_class(self):
        """Works when only one class is in the store."""
        fam = _make_fam(n_classes=5, entries=50)
        feats = F.normalize(torch.randn(10, 64), dim=-1)
        labels = torch.zeros(10, dtype=torch.long)  # only class 0
        fam.learn_local(feats, labels)
        x = torch.randn(4, 64)
        outputs, conf = fam.forward_with_confidence(x)
        assert conf.shape == (4,)
        assert not conf.isnan().any()
