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
        """Identity is stamped from the target at write time and does NOT track
        the payload: mutating values afterward must not change slot_labels."""
        cam = ContinuousCAM(key_dim=16, value_dim=8, max_entries=64, vigilance=0.95)
        # Dense (non-one-hot) targets; all misses → slots 0..2 in input order.
        protos = F.normalize(torch.randn(3, 8), dim=-1)
        targets = protos[torch.tensor([0, 1, 2])]
        cam.learn_local(torch.randn(3, 16), targets)
        occ = cam.occupied.nonzero(as_tuple=True)[0]

        # Labels equal what the compat writer derived from the written targets.
        expected = cam._labels_from_targets(targets)
        assert torch.equal(cam.slot_labels[occ], expected)

        # Now scramble the payload. Identity must be unaffected — it lives in
        # slot_labels, not in argmax(values).
        before = cam.slot_labels[occ].clone()
        cam.values[occ] = F.normalize(torch.randn(occ.numel(), 8), dim=-1)
        assert torch.equal(cam.slot_labels[occ], before)


class TestSentinelRepair:
    """Occupied slots populated WITHOUT learn_local keep label -1; consumers
    that need a non-negative index must repair them via argmax(values) rather
    than crash (PR #77 review)."""

    def _prepopulate_unstamped(self, value_dim=6, max_entries=32):
        cam = ContinuousCAM(key_dim=16, value_dim=value_dim, max_entries=max_entries)
        # Direct buffer fill — bypasses _labels_from_targets, so slot_labels==-1.
        n = 8
        cam.keys[:n] = torch.randn(n, 16)
        cam.values[:n] = _one_hot(torch.arange(n) % value_dim, value_dim)
        cam._keys_norm[:n] = F.normalize(cam.keys[:n], dim=-1)
        cam.occupied[:n] = True
        assert torch.all(cam.slot_labels[:n] == -1)  # precondition
        return cam, n

    def test_effective_labels_repairs_occupied_sentinels(self):
        cam, n = self._prepopulate_unstamped()
        occ = cam.occupied.nonzero(as_tuple=True)[0]
        eff = cam.effective_slot_labels(occ)
        assert torch.all(eff >= 0)
        assert torch.equal(eff, cam.values[occ].float().argmax(dim=-1))
        # The underlying buffer is NOT mutated by the read-time repair.
        assert torch.all(cam.slot_labels[occ] == -1)

    def test_probe_does_not_crash_on_unstamped(self):
        cam, n = self._prepopulate_unstamped()
        out = cam.probe_cross_class_similarity(torch.randn(5, 16),
                                               torch.randint(0, 6, (5,)))
        assert "n_occupied" in out

    def test_density_eviction_does_not_crash_on_unstamped(self):
        from shutter_deck.eviction import alloc_slots_density_aware
        cam, n = self._prepopulate_unstamped(max_entries=8)
        # Request more than free → forces the eviction path through slot_labels.
        slots = alloc_slots_density_aware(
            n=4, occupied=cam.occupied, usage=cam.usage,
            last_seen=cam.last_seen, values=cam.values,
            max_entries=cam.max_entries, min_per_class=1,
            slot_labels=cam.slot_labels,
        )
        assert slots.numel() <= 4


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
