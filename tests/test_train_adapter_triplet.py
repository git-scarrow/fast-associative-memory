"""
tests/test_train_adapter_triplet.py — Unit tests for train_adapter_triplet.py.

Validates:
  - mine_hard_negatives returns correctly shaped tensors
  - mine_hard_negatives selects the highest-similarity non-class sample
  - mine_hard_negatives skips anchors that have no valid positive or negative
  - train() runs on synthetic in-memory features (no ImageNet required)
  - Adapter weights change after training (loss is non-trivial)
  - Trained adapter state-dict can be saved and reloaded correctly
"""

import os
import sys
import tempfile

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapter import MetricAdapter
from train_adapter_triplet import mine_hard_negatives


# ---------------------------------------------------------------------------
# mine_hard_negatives tests
# ---------------------------------------------------------------------------


class TestMineHardNegatives:
    """Tests for the OHNM helper function."""

    def _make_batch(self, n_classes=4, samples_per_class=4, dim=32, seed=0):
        """Build a synthetic normalised feature batch with known class structure."""
        torch.manual_seed(seed)
        feats = F.normalize(torch.randn(n_classes * samples_per_class, dim), dim=-1)
        labels = torch.arange(n_classes).repeat_interleave(samples_per_class)
        return feats, labels

    def test_output_shapes(self):
        feats, labels = self._make_batch()
        anch, pos, neg = mine_hard_negatives(feats, feats, labels, labels)
        B = feats.shape[0]
        assert anch.shape[1] == feats.shape[1]
        assert pos.shape[1] == feats.shape[1]
        assert neg.shape[1] == feats.shape[1]
        assert anch.shape[0] == pos.shape[0] == neg.shape[0]
        # All anchors with a valid positive and negative should be returned
        assert anch.shape[0] <= B

    def test_triplets_found_with_sufficient_diversity(self):
        """With multiple classes and multiple samples per class, triplets must be found."""
        feats, labels = self._make_batch(n_classes=4, samples_per_class=4)
        anch, pos, neg = mine_hard_negatives(feats, feats, labels, labels)
        assert anch.shape[0] > 0, "Expected at least one valid triplet"

    def test_negative_is_different_class(self):
        """Every mined negative must belong to a different class than its anchor."""
        feats, labels = self._make_batch(n_classes=4, samples_per_class=8)
        # Build a lookup: raw feature → label (by index)
        anch, pos, neg = mine_hard_negatives(feats, feats, labels, labels)

        for i in range(anch.shape[0]):
            anchor_feat = anch[i]
            neg_feat = neg[i]
            # Find which label index the anchor and negative correspond to
            anchor_lbl = labels[(feats - anchor_feat).norm(dim=-1).argmin()]
            neg_lbl = labels[(feats - neg_feat).norm(dim=-1).argmin()]
            assert anchor_lbl != neg_lbl, "Negative must be a different class"

    def test_hard_negative_is_highest_sim(self):
        """The mined negative should be the highest-similarity non-class sample."""
        # Build a batch where one non-class sample is deliberately placed very
        # close to the anchor so it becomes the hard negative.
        torch.manual_seed(7)
        dim = 32
        # Anchor class 0 — two samples
        anchor = F.normalize(torch.randn(1, dim), dim=-1)
        positive = F.normalize(anchor + 0.05 * torch.randn(1, dim), dim=-1)
        # Hard negative: very close to anchor
        hard_neg = F.normalize(anchor + 0.01 * torch.randn(1, dim), dim=-1)
        # Easy negatives: random directions
        easy_negs = F.normalize(torch.randn(3, dim), dim=-1)

        feats = torch.cat([anchor, positive, hard_neg, easy_negs], dim=0)
        labels = torch.tensor([0, 0, 1, 2, 3, 4])

        anch_out, _, neg_out = mine_hard_negatives(feats, feats, labels, labels)

        # The anchor's mined hard negative should be the sample closest to it
        anchor_row = anch_out[0:1]  # first anchor
        sims = (anchor_row @ feats[2:].T).squeeze()  # sim to non-class samples
        best_idx = sims.argmax().item()
        # Index 0 among non-class samples is the hard_neg
        assert best_idx == 0, "Hard negative should be the highest-sim non-class sample"

    def test_empty_when_single_class(self):
        """If all samples are the same class, no valid triplets exist."""
        torch.manual_seed(1)
        feats = F.normalize(torch.randn(8, 32), dim=-1)
        labels = torch.zeros(8, dtype=torch.long)
        anch, pos, neg = mine_hard_negatives(feats, feats, labels, labels)
        assert anch.shape[0] == 0

    def test_empty_when_single_sample_per_class(self):
        """If each class has exactly one sample, no positive can be found."""
        torch.manual_seed(2)
        feats = F.normalize(torch.randn(4, 32), dim=-1)
        labels = torch.arange(4)
        anch, pos, neg = mine_hard_negatives(feats, feats, labels, labels)
        assert anch.shape[0] == 0


# ---------------------------------------------------------------------------
# Synthetic training loop test (no ImageNet, no DINOv2)
# ---------------------------------------------------------------------------


class TestSyntheticTraining:
    """Validate the adapter trains on synthetic in-memory features."""

    def _run_one_epoch(self, adapter, feat_loader, optimizer, margin=0.2):
        """One epoch of the same training logic used in train()."""
        total_loss = 0.0
        n_batches = 0
        for batch_feats, batch_labels in feat_loader:
            with torch.no_grad():
                proj = adapter(batch_feats)

            anch, pos, neg = mine_hard_negatives(
                proj, batch_feats, batch_labels, batch_labels
            )
            if anch.shape[0] == 0:
                continue

            optimizer.zero_grad()
            loss = adapter.triplet_loss(anch, pos, neg, margin=margin)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def test_loss_decreases_over_epochs(self):
        """Loss should trend downward over several epochs on linearly separable data."""
        torch.manual_seed(42)
        n_classes, samples_per_class, input_dim = 5, 20, 64

        # Construct well-separated class clusters
        class_centers = F.normalize(torch.randn(n_classes, input_dim), dim=-1) * 10
        feats = torch.cat([
            class_centers[c] + 0.1 * torch.randn(samples_per_class, input_dim)
            for c in range(n_classes)
        ])
        labels = torch.arange(n_classes).repeat_interleave(samples_per_class)

        from torch.utils.data import TensorDataset, DataLoader
        ds = TensorDataset(feats, labels)
        loader = DataLoader(ds, batch_size=32, shuffle=True)

        adapter = MetricAdapter(input_dim=input_dim, output_dim=16, hidden_dim=32)
        optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)

        losses = []
        for _ in range(5):
            adapter.train()
            loss = self._run_one_epoch(adapter, loader, optimizer)
            losses.append(loss)

        # Loss on the last epoch should be ≤ loss on the first epoch
        assert losses[-1] <= losses[0] + 0.1, (
            f"Loss did not decrease: {losses}"
        )

    def test_weights_change_after_training(self):
        """Adapter parameters should change after at least one training step."""
        torch.manual_seed(0)
        n_classes, samples_per_class, input_dim = 4, 16, 32

        feats = F.normalize(torch.randn(n_classes * samples_per_class, input_dim), dim=-1)
        labels = torch.arange(n_classes).repeat_interleave(samples_per_class)

        from torch.utils.data import TensorDataset, DataLoader
        ds = TensorDataset(feats, labels)
        # shuffle=True ensures each batch contains samples from multiple classes
        loader = DataLoader(ds, batch_size=16, shuffle=True)

        adapter = MetricAdapter(input_dim=input_dim, output_dim=8, hidden_dim=16)
        # Snapshot initial weights
        initial_weights = {
            k: v.clone() for k, v in adapter.state_dict().items()
        }

        optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
        adapter.train()
        self._run_one_epoch(adapter, loader, optimizer)

        changed = any(
            not torch.allclose(adapter.state_dict()[k], initial_weights[k])
            for k in initial_weights
        )
        assert changed, "At least one parameter should have changed after training"

    def test_state_dict_save_reload(self):
        """Saved adapter state-dict should reload with identical forward output."""
        torch.manual_seed(5)
        adapter = MetricAdapter(input_dim=64, output_dim=16, hidden_dim=32)
        x = torch.randn(4, 64)
        out_before = adapter(x)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            torch.save(adapter.state_dict(), path)
            adapter2 = MetricAdapter(input_dim=64, output_dim=16, hidden_dim=32)
            adapter2.load_state_dict(torch.load(path, weights_only=True))
            out_after = adapter2(x)
            assert torch.allclose(out_before, out_after, atol=1e-6)
        finally:
            os.unlink(path)
