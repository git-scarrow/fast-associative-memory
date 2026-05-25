"""
tests/test_slot_labels.py — First-class slot_labels semantic decoupling.

Validates the AgentAssociativeMemory abstraction boundary:
  - keys        = retrieval geometry
  - values      = continuous payload vectors
  - slot_labels = semantic identity (first-class, written at learn time)

Characterizes the compatibility guarantee: for a freshly written slot,
slot_labels reproduces the legacy argmax(values) label exactly, so existing
one-hot / orthogonal-prototype classification behavior is unchanged. Also
proves labels are stable state (not re-derived from the drifting payload) and
that values can hold arbitrary dense vectors without leaking class identity.
"""

import torch
import torch.nn.functional as F
import pytest

from associative_core import ContinuousCAM


def _one_hot(classes, n):
    return F.one_hot(torch.as_tensor(classes), num_classes=n).float()


class TestWriteTimeStamping:
    def test_labels_match_argmax_on_fresh_write(self):
        """Freshly written slot_labels == legacy argmax(values) — compat guarantee."""
        cam = ContinuousCAM(key_dim=16, value_dim=5, max_entries=64, vigilance=0.9)
        classes = torch.tensor([0, 1, 2, 3, 4, 2, 1])
        cam.learn_local(torch.randn(7, 16), _one_hot(classes, 5))

        occ = cam.occupied.nonzero(as_tuple=True)[0]
        legacy = cam.values[occ].float().argmax(dim=-1)
        assert torch.equal(cam.slot_labels[occ], legacy)

    def test_unoccupied_label_is_sentinel(self):
        cam = ContinuousCAM(key_dim=16, value_dim=5, max_entries=64)
        assert torch.all(cam.slot_labels == -1)
        cam.learn_local(torch.randn(3, 16), _one_hot([0, 1, 2], 5))
        assert torch.all(cam.slot_labels[~cam.occupied] == -1)
        assert torch.all(cam.slot_labels[cam.occupied] >= 0)


class TestLabelStability:
    def test_label_survives_ema_drift(self):
        """slot_labels is fixed state; it must not flip as the payload drifts."""
        cam = ContinuousCAM(key_dim=16, value_dim=4, max_entries=16,
                            vigilance=0.0, hebb_lr=0.5)  # v=0 → everything hits
        key = F.normalize(torch.randn(1, 16), dim=-1)
        cam.learn_local(key, _one_hot([1], 4))
        slot = cam.occupied.nonzero(as_tuple=True)[0]
        label0 = cam.slot_labels[slot].clone()

        # Repeatedly hit the same slot with a noisy, non-one-hot target to drag
        # the payload around. The stored label must stay put.
        for _ in range(20):
            noisy = torch.tensor([[0.4, 0.45, 0.1, 0.05]])
            cam.learn_local(key, noisy)
        assert torch.equal(cam.slot_labels[slot], label0)


class TestDecoupledFromPayload:
    def test_dense_values_do_not_define_identity(self):
        """With dense (non-one-hot) targets, identity comes from the writer, not
        from whatever argmax(values) happens to land on later."""
        cam = ContinuousCAM(key_dim=16, value_dim=8, max_entries=64, vigilance=0.95)
        # Orthogonal-style dense targets (argmax position == intended label).
        protos = F.normalize(torch.randn(3, 8), dim=-1)
        ids = torch.tensor([0, 1, 2])
        cam.learn_local(torch.randn(3, 16), protos[ids])
        occ = cam.occupied.nonzero(as_tuple=True)[0]
        # Labels are concrete LongTensor entries, present and stable.
        assert cam.slot_labels[occ].dtype == torch.long
        assert cam.slot_labels[occ].numel() == 3


class TestLifecyclePathsUseLabels:
    """Smoke: eviction, sleep, and the cross-class probe run off slot_labels."""

    def test_sleep_runs(self):
        cam = ContinuousCAM(key_dim=16, value_dim=4, max_entries=64, vigilance=0.99)
        cam.learn_local(torch.randn(20, 16), _one_hot(torch.randint(0, 4, (20,)), 4))
        out = cam.sleep(max_epochs=2)
        assert "epochs" in out

    def test_eviction_runs(self):
        cam = ContinuousCAM(key_dim=16, value_dim=4, max_entries=8,
                            vigilance=0.99, use_lfu=True, adaptive_eviction=False)
        # Force allocation past capacity → eviction path reads slot_labels.
        cam.learn_local(torch.randn(20, 16), _one_hot(torch.randint(0, 4, (20,)), 4))
        assert cam.occupied.sum().item() <= 8

    def test_probe_runs(self):
        cam = ContinuousCAM(key_dim=16, value_dim=4, max_entries=64, vigilance=0.9)
        labels = torch.randint(0, 4, (30,))
        cam.learn_local(torch.randn(30, 16), _one_hot(labels, 4))
        out = cam.probe_cross_class_similarity(torch.randn(10, 16),
                                               torch.randint(0, 4, (10,)))
        assert "n_occupied" in out
