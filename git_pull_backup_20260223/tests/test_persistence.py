"""
tests/test_persistence.py — Sparse v2 CAM persistence tests.

Validates:
  - Sparse save/load round-trip for occupied slots
  - _keys_norm reconstruction from keys
  - v1 format rejection with clear fallback
  - Max-entry mismatch handling via slot-index clamping
  - Size scales with number of occupied prototypes (not max_entries)
"""

import struct

import pytest
import torch

from associative_core import ContinuousCAM
from shutter_deck.persistence import HEADER_FMT, MAGIC, VERSION, load_cam_state, save_cam_state


def _make_cam(key_dim=32, value_dim=8, max_entries=100):
    return ContinuousCAM(key_dim=key_dim, value_dim=value_dim, max_entries=max_entries)


@pytest.fixture
def cam():
    c = _make_cam()
    slots = [0, 1, 5, 9, 17, 23, 42, 56, 78, 99]
    for i, slot in enumerate(slots):
        c.keys[slot] = torch.randn(c.key_dim)
        c.values[slot] = torch.randn(c.value_dim)
        c.occupied[slot] = True
        c.last_seen[slot] = float(i * 100)
        c.usage[slot] = float(i + 1)
        c.hit_counts[slot] = i + 1
        c.prototype_density[slot] = float(i) / 10.0
    c._update_key_norm(c.occupied.nonzero(as_tuple=True)[0])
    # These are intentionally non-default to verify they are not persisted in sparse v2.
    c.class_means[:] = torch.randn_like(c.class_means)
    c.class_vars[:] = torch.rand_like(c.class_vars) + 0.25
    return c


@pytest.fixture
def state_path(tmp_path):
    return tmp_path / "test_state.bin"


class TestSparseRoundTrip:
    def test_occupied_slots_bitwise_identical(self, cam, state_path):
        save_cam_state(cam, state_path)
        cam2 = _make_cam()
        ok = load_cam_state(cam2, state_path)
        assert ok is True

        occ = cam.occupied
        for name in ["keys", "values", "occupied", "last_seen", "usage", "hit_counts", "prototype_density"]:
            orig = getattr(cam, name)
            loaded = getattr(cam2, name)
            if orig.ndim == 2:
                assert torch.equal(orig[occ], loaded[occ]), f"Mismatch in occupied rows for {name}"
            else:
                assert torch.equal(orig[occ], loaded[occ]), f"Mismatch in occupied rows for {name}"

    def test_non_occupied_slots_are_default_zero_false(self, cam, state_path):
        save_cam_state(cam, state_path)
        cam2 = _make_cam()
        assert load_cam_state(cam2, state_path)

        mask = ~cam2.occupied
        assert torch.count_nonzero(cam2.keys[mask]) == 0
        assert torch.count_nonzero(cam2.values[mask]) == 0
        assert torch.count_nonzero(cam2._keys_norm[mask]) == 0
        assert torch.count_nonzero(cam2.last_seen[mask]) == 0
        assert torch.count_nonzero(cam2.usage[mask]) == 0
        assert torch.count_nonzero(cam2.hit_counts[mask]) == 0
        assert torch.count_nonzero(cam2.prototype_density[mask]) == 0

    def test_key_norm_reconstructed_exactly(self, cam, state_path):
        save_cam_state(cam, state_path)
        cam2 = _make_cam()
        assert load_cam_state(cam2, state_path)

        occ_idx = cam.occupied.nonzero(as_tuple=True)[0]
        assert torch.equal(cam._keys_norm[occ_idx], cam2._keys_norm[occ_idx])
        cos = torch.nn.functional.cosine_similarity(cam._keys_norm[occ_idx], cam2._keys_norm[occ_idx], dim=-1)
        assert torch.allclose(cos, torch.ones_like(cos))

    def test_global_class_stats_reset_to_defaults(self, cam, state_path):
        save_cam_state(cam, state_path)
        cam2 = _make_cam()
        cam2.class_means[:] = 123.0
        cam2.class_vars[:] = 456.0
        assert load_cam_state(cam2, state_path)

        assert torch.count_nonzero(cam2.class_means) == 0
        assert torch.equal(cam2.class_vars, torch.ones_like(cam2.class_vars))

    def test_repeated_save_load_stable(self, cam, state_path):
        for _ in range(3):
            save_cam_state(cam, state_path)
            cam2 = _make_cam()
            assert load_cam_state(cam2, state_path)
            occ = cam.occupied
            assert torch.equal(cam.keys[occ], cam2.keys[occ])

    def test_bfloat16_cam_roundtrip(self, state_path):
        cam = ContinuousCAM(key_dim=32, value_dim=32, max_entries=64, use_bfloat16=True)
        for slot in [0, 3, 7]:
            cam.keys[slot] = torch.randn(32)
            cam.values[slot] = torch.randn(32)
            cam.occupied[slot] = True
            cam.last_seen[slot] = float(slot)
            cam.usage[slot] = 1.0
            cam.hit_counts[slot] = 1
            cam.prototype_density[slot] = 0.2
        cam._update_key_norm(cam.occupied.nonzero(as_tuple=True)[0])

        save_cam_state(cam, state_path)
        cam2 = ContinuousCAM(key_dim=32, value_dim=32, max_entries=64, use_bfloat16=True)
        assert load_cam_state(cam2, state_path)
        occ = cam.occupied.nonzero(as_tuple=True)[0]
        assert cam2.keys.dtype == torch.bfloat16
        assert cam2.values.dtype == torch.bfloat16
        assert torch.equal(cam.keys[occ], cam2.keys[occ])
        assert torch.equal(cam.values[occ], cam2.values[occ])


class TestMismatchAndVersionHandling:
    def test_wrong_key_dim_rejected(self, cam, state_path):
        save_cam_state(cam, state_path)
        cam2 = _make_cam(key_dim=64, value_dim=8, max_entries=100)
        assert load_cam_state(cam2, state_path) is False

    def test_wrong_value_dim_rejected(self, cam, state_path):
        save_cam_state(cam, state_path)
        cam2 = _make_cam(key_dim=32, value_dim=16, max_entries=100)
        assert load_cam_state(cam2, state_path) is False

    def test_v1_rejected_with_clear_message(self, state_path, caplog):
        # Minimal v1 header; loader should reject before attempting reads.
        state_path.write_bytes(struct.pack(HEADER_FMT, MAGIC, 1, 100, 32, 8, 0))
        cam = _make_cam()
        with caplog.at_level("WARNING"):
            ok = load_cam_state(cam, state_path)
        assert ok is False
        assert "Delete the old state file" in caplog.text

    def test_max_entries_mismatch_loads_and_clamps(self, state_path):
        cam = _make_cam(max_entries=100)
        # Put one row beyond the destination max_entries to verify clamping.
        for slot in [0, 5, 40, 99]:
            cam.keys[slot] = torch.randn(cam.key_dim)
            cam.values[slot] = torch.randn(cam.value_dim)
            cam.occupied[slot] = True
        cam._update_key_norm(cam.occupied.nonzero(as_tuple=True)[0])
        save_cam_state(cam, state_path)

        cam_small = _make_cam(max_entries=50)
        assert load_cam_state(cam_small, state_path) is True
        loaded_slots = cam_small.occupied.nonzero(as_tuple=True)[0].tolist()
        assert loaded_slots == [0, 5, 40]
        assert 99 not in loaded_slots


class TestGracefulFallback:
    def test_missing_file(self, state_path):
        cam = _make_cam()
        assert load_cam_state(cam, state_path) is False

    def test_truncated_file(self, cam, state_path):
        save_cam_state(cam, state_path)
        state_path.write_bytes(state_path.read_bytes()[:50])
        cam2 = _make_cam()
        assert load_cam_state(cam2, state_path) is False

    def test_bad_magic(self, state_path):
        state_path.write_bytes(b"XXXX" + b"\x00" * 64)
        cam = _make_cam()
        assert load_cam_state(cam, state_path) is False


class TestAtomicWrite:
    def test_no_tmp_file_remains(self, cam, state_path):
        save_cam_state(cam, state_path)
        assert not state_path.with_suffix(".tmp").exists()
        assert state_path.exists()


class TestSparseSizeScaling:
    def _fill_n(self, cam, n: int) -> None:
        for slot in range(n):
            cam.keys[slot] = torch.randn(cam.key_dim)
            cam.values[slot] = torch.randn(cam.value_dim)
            cam.occupied[slot] = True
            cam.last_seen[slot] = float(slot)
            cam.usage[slot] = 1.0
            cam.hit_counts[slot] = 1
            cam.prototype_density[slot] = 0.1
        cam._update_key_norm(cam.occupied.nonzero(as_tuple=True)[0])

    def test_one_prototype_file_is_small(self, tmp_path):
        # Autoassociative 768-d stores both keys and values, so expect ~6-8KB.
        cam = _make_cam(key_dim=768, value_dim=768, max_entries=2000)
        self._fill_n(cam, 1)
        p = tmp_path / "one.bin"
        save_cam_state(cam, p)
        assert p.stat().st_size < 50_000

    def test_1000_prototypes_size_scales_with_occupancy(self, tmp_path):
        # With 768-d keys + 768-d values, 1000 rows is ~6.2MB raw payload.
        cam = _make_cam(key_dim=768, value_dim=768, max_entries=2000)
        self._fill_n(cam, 1000)
        p = tmp_path / "thousand.bin"
        save_cam_state(cam, p)
        assert p.stat().st_size < 7_500_000
        assert p.stat().st_size > 5_000_000  # sanity check: values are also persisted
