"""
tests/test_eviction.py — Density-aware eviction correctness tests.

Validates:
  - Class floors are never violated
  - Density-normalized priority ordering is correct
  - Immune slots survive eviction
  - Free slots allocated first
  - Edge cases (n > max_entries, all immune, etc.)
"""

import torch
import pytest

from shutter_deck.eviction import alloc_slots_density_aware


def _make_cam_state(max_entries: int, value_dim: int = 10):
    """Helper: create blank CAM-like tensors."""
    return {
        "occupied": torch.zeros(max_entries, dtype=torch.bool),
        "usage": torch.zeros(max_entries, dtype=torch.float32),
        "last_seen": torch.zeros(max_entries, dtype=torch.float64),
        "values": torch.zeros(max_entries, value_dim, dtype=torch.float32),
        "max_entries": max_entries,
    }


def _fill_slot(state, slot: int, class_id: int, usage: float, last_seen: float):
    """Occupy a slot with given class, usage, and timestamp."""
    state["occupied"][slot] = True
    state["usage"][slot] = usage
    state["last_seen"][slot] = last_seen
    state["values"][slot] = 0.0
    state["values"][slot, class_id] = 1.0


class TestFreeSlotAllocation:
    def test_allocates_free_slots_first(self):
        state = _make_cam_state(20)
        # Occupy slots 0-9, leave 10-19 free
        for i in range(10):
            _fill_slot(state, i, class_id=0, usage=10.0, last_seen=1.0)

        result = alloc_slots_density_aware(n=5, **state)
        assert len(result) == 5
        # All allocated slots should be from free range (10-19)
        assert (result >= 10).all()
        assert (result < 20).all()

    def test_all_free(self):
        state = _make_cam_state(100)
        result = alloc_slots_density_aware(n=10, **state)
        assert len(result) == 10
        assert result.unique().numel() == 10  # no duplicates

    def test_partial_free_plus_eviction(self):
        state = _make_cam_state(10)
        # Fill slots 0-7 with class 0 (plenty of headroom above floor)
        for i in range(8):
            _fill_slot(state, i, class_id=0, usage=float(i), last_seen=float(i))
        # Slots 8-9 are free

        result = alloc_slots_density_aware(n=5, min_per_class=3, **state)
        assert len(result) == 5
        # Should include the 2 free slots (8, 9)
        free_in_result = (result >= 8).sum().item()
        assert free_in_result == 2


class TestClassFloorImmunity:
    def test_floor_protects_rare_class(self):
        """A class with exactly min_per_class slots must not lose any."""
        state = _make_cam_state(20, value_dim=5)
        # Class 0: 5 slots (at floor) — should be immune
        for i in range(5):
            _fill_slot(state, i, class_id=0, usage=1.0, last_seen=1.0)
        # Class 1: 15 slots (well above floor) — should provide victims
        for i in range(5, 20):
            _fill_slot(state, i, class_id=1, usage=1.0, last_seen=1.0)

        result = alloc_slots_density_aware(n=5, min_per_class=5, **state)
        assert len(result) == 5
        # None of the class-0 slots (0-4) should be evicted
        for slot in result:
            assert slot.item() >= 5, f"Slot {slot.item()} is class-0 (immune) but was evicted"

    def test_below_floor_also_immune(self):
        """Classes with fewer than min_per_class slots are also immune."""
        state = _make_cam_state(15, value_dim=5)
        # Class 0: 3 slots (below floor of 5) — immune
        for i in range(3):
            _fill_slot(state, i, class_id=0, usage=0.5, last_seen=0.5)
        # Class 1: 12 slots
        for i in range(3, 15):
            _fill_slot(state, i, class_id=1, usage=1.0, last_seen=1.0)

        result = alloc_slots_density_aware(n=3, min_per_class=5, **state)
        assert len(result) == 3
        for slot in result:
            assert slot.item() >= 3

    def test_all_immune_returns_empty(self):
        """If all slots are immune, no eviction occurs."""
        state = _make_cam_state(10, value_dim=3)
        # 3 classes, each with exactly 3 slots (floor=3 → all immune)
        for i in range(3):
            _fill_slot(state, i, class_id=0, usage=1.0, last_seen=1.0)
        for i in range(3, 6):
            _fill_slot(state, i, class_id=1, usage=1.0, last_seen=1.0)
        for i in range(6, 9):
            _fill_slot(state, i, class_id=2, usage=1.0, last_seen=1.0)
        # slot 9 is free

        # Request 5 slots: 1 free + need 4 evictions, but only 0 non-immune
        result = alloc_slots_density_aware(n=5, min_per_class=3, **state)
        # Should get at most 1 (the free slot)
        assert len(result) <= 1


class TestDensityNormalizedPriority:
    def test_high_usage_in_large_class_evicted_before_low_usage_in_small_class(self):
        """
        Sunset (class 0, pop=10): slot with usage=50 → priority=50/10=5.0
        Heron  (class 1, pop=2):  slot with usage=2  → priority=2/2=1.0
        The heron has lower raw usage but HIGHER density-normalized priority,
        so the sunset slot should be evicted first (lower priority = evict).
        Wait — the sunset slot has priority 5.0 vs heron 1.0.
        Lower priority = evict first. So heron (1.0) gets evicted before sunset (5.0)?
        
        No — re-read the spec: "A sunset prototype with usage=50 in a class of 500 
        scores 0.10. A heron prototype with usage=2 in a class of 5 scores 0.40 — 
        and is thus SAFER." Lower score = more redundant = evict first.
        
        So: sunset (usage=50, pop=500) → 0.10 (evict first)
            heron  (usage=2, pop=5)   → 0.40 (safer, evict later)
        """
        state = _make_cam_state(506, value_dim=5)
        
        # Class 0 (sunset): 500 slots, all usage=50
        for i in range(500):
            _fill_slot(state, i, class_id=0, usage=50.0, last_seen=1.0)
        
        # Class 1 (heron): 6 slots with usage=2 (above floor of 5, so 1 is evictable)
        for i in range(500, 506):
            _fill_slot(state, i, class_id=1, usage=2.0, last_seen=1.0)
        
        # All 506 slots occupied, no free → must evict
        # Sunset priority: 50/500 = 0.10
        # Heron priority: 2/6 = 0.333
        # Lowest priority = sunset → sunset should be evicted
        result = alloc_slots_density_aware(n=1, min_per_class=5, **state)
        assert len(result) == 1
        evicted = result[0].item()
        # Should evict a sunset slot (0-499), not a heron slot (500-505)
        assert evicted < 500, f"Expected sunset eviction, got slot {evicted}"

    def test_tie_broken_by_oldest_last_seen(self):
        """Same density-normalized priority → oldest (lowest last_seen) evicted first."""
        state = _make_cam_state(20, value_dim=3)
        # All class 0, same usage, but different timestamps
        for i in range(20):
            _fill_slot(state, i, class_id=0, usage=10.0, last_seen=float(i))

        result = alloc_slots_density_aware(n=3, min_per_class=5, **state)
        assert len(result) == 3
        # Should evict slots 0, 1, 2 (oldest)
        evicted = sorted(result.tolist())
        assert evicted == [0, 1, 2]


class TestEdgeCases:
    def test_n_greater_than_max_entries(self):
        """n > max_entries → return max_entries slots."""
        state = _make_cam_state(10)
        result = alloc_slots_density_aware(n=100, **state)
        assert len(result) <= 10

    def test_empty_table(self):
        state = _make_cam_state(10)
        result = alloc_slots_density_aware(n=5, **state)
        assert len(result) == 5

    def test_result_device_matches_occupied(self):
        state = _make_cam_state(10)
        result = alloc_slots_density_aware(n=3, **state)
        assert result.device == state["occupied"].device

    def test_no_duplicates_in_result(self):
        state = _make_cam_state(20, value_dim=3)
        for i in range(20):
            _fill_slot(state, i, class_id=i % 3, usage=float(i), last_seen=float(i))
        result = alloc_slots_density_aware(n=10, min_per_class=2, **state)
        assert result.unique().numel() == result.numel()


class TestInstallMonkeyPatch:
    def test_install_replaces_method(self):
        from associative_core import ContinuousCAM
        from shutter_deck.eviction import install_density_eviction

        cam = ContinuousCAM(key_dim=16, value_dim=4, max_entries=50)
        original_method = cam._alloc_slots_batch

        install_density_eviction(cam, min_per_class=3)

        # Method should be replaced
        assert cam._alloc_slots_batch is not original_method

        # Should still return correct number of free slots
        result = cam._alloc_slots_batch(5)
        assert len(result) == 5
