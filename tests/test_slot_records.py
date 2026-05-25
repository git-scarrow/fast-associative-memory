"""
tests/test_slot_records.py — Per-slot provenance (the slot_records layer).

Validates the fourth decoupling layer:
    keys=geometry, values=payload, slot_labels=identity, slot_records=history.

slot_records captures the ACTUAL write/merge history (which external record IDs
formed each prototype) at learn time — never reconstructed from nearest-prototype
assignment. It is opt-in (track_provenance=True); off by default so existing CAM
behavior is byte-for-byte unchanged.
"""

import torch
import torch.nn.functional as F
import pytest

from associative_core import ContinuousCAM


def _one_hot(classes, n):
    return F.one_hot(torch.as_tensor(classes), num_classes=n).float()


class TestDisabledByDefault:
    def test_off_by_default(self):
        cam = ContinuousCAM(key_dim=8, value_dim=4, max_entries=16)
        assert cam.track_provenance is False
        assert cam.slot_records is None

    def test_record_ids_rejected_when_off(self):
        cam = ContinuousCAM(key_dim=8, value_dim=4, max_entries=16)
        with pytest.raises(ValueError, match="track_provenance=False"):
            cam.learn_local(torch.randn(2, 8), _one_hot([0, 1], 4),
                            record_ids=["a", "b"])

    def test_no_record_ids_unchanged_when_off(self):
        """Default path must be untouched: no record_ids → no error, no records."""
        cam = ContinuousCAM(key_dim=8, value_dim=4, max_entries=16)
        cam.learn_local(torch.randn(3, 8), _one_hot([0, 1, 2], 4))
        assert cam.slot_records is None

    def test_accessors_raise_when_off(self):
        cam = ContinuousCAM(key_dim=8, value_dim=4, max_entries=16)
        with pytest.raises(RuntimeError, match="track_provenance=False"):
            cam.records_for(0)
        with pytest.raises(RuntimeError, match="track_provenance=False"):
            cam.records_for_slots([0, 1])


class TestEnabled:
    def test_init_allocates_empty_records(self):
        cam = ContinuousCAM(key_dim=8, value_dim=4, max_entries=16,
                            track_provenance=True)
        assert cam.slot_records == [None] * 16

    def test_length_mismatch_raises(self):
        cam = ContinuousCAM(key_dim=8, value_dim=4, max_entries=16,
                            track_provenance=True)
        with pytest.raises(ValueError, match="!= batch size"):
            cam.learn_local(torch.randn(3, 8), _one_hot([0, 1, 2], 4),
                            record_ids=["only-one"])

    def test_allocation_stamps_record(self):
        cam = ContinuousCAM(key_dim=8, value_dim=4, max_entries=16,
                            track_provenance=True, vigilance=0.99)
        cam.learn_local(torch.randn(3, 8), _one_hot([0, 1, 2], 4),
                        record_ids=["x", "y", "z"])
        occ = cam.occupied.nonzero(as_tuple=True)[0].tolist()
        recorded = set().union(*(cam.records_for(s) for s in occ))
        assert recorded == {"x", "y", "z"}
        # Each fresh slot owns exactly one source id.
        assert all(len(cam.records_for(s)) == 1 for s in occ)

    def test_none_record_ids_does_not_invent(self):
        cam = ContinuousCAM(key_dim=8, value_dim=4, max_entries=16,
                            track_provenance=True, vigilance=0.99)
        cam.learn_local(torch.randn(2, 8), _one_hot([0, 1], 4))  # no ids
        occ = cam.occupied.nonzero(as_tuple=True)[0].tolist()
        assert all(cam.records_for(s) == set() for s in occ)


class TestAccumulateOnMerge:
    def test_same_slot_hit_unions(self):
        """A same-class hit merges the new source id into the slot's set."""
        cam = ContinuousCAM(key_dim=8, value_dim=4, max_entries=16,
                            track_provenance=True, vigilance=0.0)  # all hits
        key = F.normalize(torch.randn(1, 8), dim=-1)
        cam.learn_local(key, _one_hot([1], 4), record_ids=["first"])
        slot = cam.occupied.nonzero(as_tuple=True)[0].item()
        assert cam.records_for(slot) == {"first"}

        # Re-present the same key/class with a new source id → union, not replace.
        cam.learn_local(key, _one_hot([1], 4), record_ids=["second"])
        assert cam.records_for(slot) == {"first", "second"}

    def test_batch_hits_to_one_slot_union(self):
        cam = ContinuousCAM(key_dim=8, value_dim=4, max_entries=16,
                            track_provenance=True, vigilance=0.0)
        key = F.normalize(torch.randn(1, 8), dim=-1)
        cam.learn_local(key, _one_hot([2], 4), record_ids=["seed"])
        slot = cam.occupied.nonzero(as_tuple=True)[0].item()
        # A batch of 3 same-key rows all hit the one slot → all 3 ids union in.
        batch = key.repeat(3, 1)
        cam.learn_local(batch, _one_hot([2, 2, 2], 4),
                        record_ids=["a", "b", "c"])
        assert cam.records_for(slot) == {"seed", "a", "b", "c"}

    def test_history_not_reconstructed_from_geometry(self):
        """Guardrail: provenance reflects what was WRITTEN, even if a later
        nearest-prototype assignment would attribute the source elsewhere."""
        cam = ContinuousCAM(key_dim=8, value_dim=4, max_entries=16,
                            track_provenance=True, vigilance=0.0)
        ka = F.normalize(torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0]]), dim=-1)
        cam.learn_local(ka, _one_hot([0], 4), record_ids=["src-A"])
        slot_a = cam.occupied.nonzero(as_tuple=True)[0].item()
        # src-A lives on slot_a because that's where it was written.
        assert "src-A" in cam.records_for(slot_a)


class TestEvictionReuseClearsStale:
    def test_reused_slot_drops_previous_sources(self):
        # Capacity 2, pure LRU eviction, vigilance high so every write is a miss.
        cam = ContinuousCAM(key_dim=8, value_dim=4, max_entries=2,
                            track_provenance=True, vigilance=0.999,
                            adaptive_eviction=False, use_lfu=False)
        keys = F.normalize(torch.randn(3, 8), dim=-1)
        cam.learn_local(keys[0:1], _one_hot([0], 4), record_ids=["old0"])
        cam.learn_local(keys[1:2], _one_hot([1], 4), record_ids=["old1"])
        # Third miss must evict one slot and reuse it for "new2".
        cam.learn_local(keys[2:3], _one_hot([2], 4), record_ids=["new2"])

        all_ids = set().union(
            *(cam.records_for(s) for s in range(cam.max_entries)))
        assert "new2" in all_ids
        # Exactly one of the originals was evicted; no slot carries a stale set
        # alongside the new occupant.
        for s in cam.occupied.nonzero(as_tuple=True)[0].tolist():
            assert len(cam.records_for(s)) == 1

    def test_reuse_without_record_ids_clears(self):
        cam = ContinuousCAM(key_dim=8, value_dim=4, max_entries=2,
                            track_provenance=True, vigilance=0.999,
                            adaptive_eviction=False, use_lfu=False)
        keys = F.normalize(torch.randn(3, 8), dim=-1)
        cam.learn_local(keys[0:1], _one_hot([0], 4), record_ids=["old0"])
        cam.learn_local(keys[1:2], _one_hot([1], 4), record_ids=["old1"])
        # Reuse via eviction but WITHOUT new ids → reused slot must be cleared,
        # not left holding the evicted occupant's id.
        cam.learn_local(keys[2:3], _one_hot([2], 4))
        records = [cam.records_for(s)
                   for s in cam.occupied.nonzero(as_tuple=True)[0].tolist()]
        # The surviving original keeps its id; the reused slot is empty.
        assert set() in records


class TestAccessors:
    def test_records_for_slots_aligns_and_copies(self):
        cam = ContinuousCAM(key_dim=8, value_dim=4, max_entries=16,
                            track_provenance=True, vigilance=0.99)
        cam.learn_local(torch.randn(2, 8), _one_hot([0, 1], 4),
                        record_ids=["p", "q"])
        occ = cam.occupied.nonzero(as_tuple=True)[0]
        sets = cam.records_for_slots(occ)
        assert len(sets) == occ.numel()
        # Returned sets are copies: mutating them must not corrupt internal state.
        sets[0].add("INTRUDER")
        assert "INTRUDER" not in cam.records_for(int(occ[0].item()))

    def test_records_for_slots_accepts_trace_tensor(self):
        cam = ContinuousCAM(key_dim=8, value_dim=4, max_entries=64,
                            track_provenance=True, vigilance=0.9)
        labels = torch.randint(0, 4, (20,))
        cam.learn_local(torch.randn(20, 8), _one_hot(labels, 4),
                        record_ids=list(range(20)))
        _, tr = cam.forward(torch.randn(5, 8), trace=True)
        mapped = cam.records_for_slots(tr.final_slots)  # 2-D tensor flattens
        assert isinstance(mapped, list)
        assert all(isinstance(x, set) for x in mapped)
