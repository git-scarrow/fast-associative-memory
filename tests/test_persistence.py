"""
tests/test_persistence.py — Save/load round-trip tests.

Validates:
  - All tensors are bitwise identical after save → load
  - Dimension mismatch rejection
  - Missing file graceful fallback
  - Atomic rename (no corruption on partial write)
"""

import struct
from pathlib import Path

import torch
import pytest

from associative_core import ContinuousCAM
from shutter_deck.persistence import save_cam_state, load_cam_state, HEADER_FMT, MAGIC, VERSION


@pytest.fixture
def cam():
    """A small CAM with some occupied slots for testing."""
    c = ContinuousCAM(key_dim=32, value_dim=8, max_entries=100)
    # Fill 10 slots with known data
    for i in range(10):
        c.keys[i] = torch.randn(32)
        c.values[i] = torch.randn(8)
        c.occupied[i] = True
        # Explicit non-sentinel label pattern so the round-trip exercises real
        # occupied-slot label persistence, not just the -1 default.
        c.slot_labels[i] = i % 8
        c.last_seen[i] = float(i * 100)
        c.usage[i] = float(i + 1)
        c.hit_counts[i] = i + 1
    c._keys_norm[:10] = torch.nn.functional.normalize(c.keys[:10], dim=-1)
    return c


@pytest.fixture
def state_path(tmp_path):
    return tmp_path / "test_state.bin"


class TestSaveLoadRoundTrip:
    def test_bitwise_identical(self, cam, state_path):
        """Save → load into a fresh CAM → all buffers match."""
        save_cam_state(cam, state_path)
        assert state_path.exists()

        cam2 = ContinuousCAM(key_dim=32, value_dim=8, max_entries=100)
        ok = load_cam_state(cam2, state_path)
        assert ok is True

        # Check all tensor buffers
        for name in ["keys", "values", "slot_labels", "occupied", "last_seen",
                      "usage", "_keys_norm", "hit_counts"]:
            orig = getattr(cam, name)
            loaded = getattr(cam2, name)
            assert torch.equal(orig, loaded), f"Mismatch in buffer '{name}'"

    def test_occupied_count_preserved(self, cam, state_path):
        save_cam_state(cam, state_path)
        cam2 = ContinuousCAM(key_dim=32, value_dim=8, max_entries=100)
        load_cam_state(cam2, state_path)
        assert cam2.occupied.sum().item() == cam.occupied.sum().item()

    def test_repeated_save_load(self, cam, state_path):
        """Multiple save/load cycles remain stable."""
        for _ in range(3):
            save_cam_state(cam, state_path)
            cam2 = ContinuousCAM(key_dim=32, value_dim=8, max_entries=100)
            load_cam_state(cam2, state_path)
            assert torch.equal(cam.keys, cam2.keys)


class TestMismatchRejection:
    def test_wrong_max_entries(self, cam, state_path):
        save_cam_state(cam, state_path)
        cam2 = ContinuousCAM(key_dim=32, value_dim=8, max_entries=200)
        ok = load_cam_state(cam2, state_path)
        assert ok is False

    def test_wrong_key_dim(self, cam, state_path):
        save_cam_state(cam, state_path)
        cam2 = ContinuousCAM(key_dim=64, value_dim=8, max_entries=100)
        ok = load_cam_state(cam2, state_path)
        assert ok is False

    def test_wrong_value_dim(self, cam, state_path):
        save_cam_state(cam, state_path)
        cam2 = ContinuousCAM(key_dim=32, value_dim=16, max_entries=100)
        ok = load_cam_state(cam2, state_path)
        assert ok is False


class TestGracefulFallback:
    def test_missing_file(self, state_path):
        cam = ContinuousCAM(key_dim=32, value_dim=8, max_entries=100)
        ok = load_cam_state(cam, state_path)
        assert ok is False

    def test_truncated_file(self, cam, state_path):
        save_cam_state(cam, state_path)
        # Truncate the file
        data = state_path.read_bytes()
        state_path.write_bytes(data[:50])
        cam2 = ContinuousCAM(key_dim=32, value_dim=8, max_entries=100)
        ok = load_cam_state(cam2, state_path)
        assert ok is False

    def test_bad_magic(self, state_path):
        # Write garbage header
        state_path.write_bytes(b"XXXX" + b"\x00" * 100)
        cam = ContinuousCAM(key_dim=32, value_dim=8, max_entries=100)
        ok = load_cam_state(cam, state_path)
        assert ok is False


class TestAtomicWrite:
    def test_no_tmp_file_remains(self, cam, state_path):
        save_cam_state(cam, state_path)
        tmp_path = state_path.with_suffix(".tmp")
        assert not tmp_path.exists()
        assert state_path.exists()
