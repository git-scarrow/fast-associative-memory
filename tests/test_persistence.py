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
from shutter_deck.persistence import (
    save_cam_state, load_cam_state, HEADER_FMT, MAGIC, VERSION,
    save_provenance_sidecar, load_provenance_sidecar, _provenance_path,
)


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


@pytest.fixture
def prov_cam():
    """A provenance-tracking CAM with mixed-type record sets on a few slots."""
    c = ContinuousCAM(key_dim=32, value_dim=8, max_entries=100,
                      track_provenance=True)
    for i in range(5):
        c.keys[i] = torch.randn(32)
        c.values[i] = torch.randn(8)
        c.occupied[i] = True
        c.slot_labels[i] = i % 8
    c._keys_norm[:5] = torch.nn.functional.normalize(c.keys[:5], dim=-1)
    # Heterogeneous hashables: int, str, float, bool, and a (nested) tuple.
    c.slot_records[0] = {1, 2, 3}
    c.slot_records[1] = {"alpha", "beta"}
    c.slot_records[2] = {("img", 7), ("cap", ("nested", 9))}
    c.slot_records[3] = {3.5, True, "mixed", 42}
    # slot 4 occupied but left with no provenance (None) — must stay None.
    return c


class TestProvenanceSidecar:
    def test_roundtrip_preserves_records_and_types(self, prov_cam, state_path):
        save_cam_state(prov_cam, state_path, write_provenance=True)
        assert _provenance_path(state_path).exists()

        cam2 = ContinuousCAM(key_dim=32, value_dim=8, max_entries=100,
                             track_provenance=True)
        assert load_cam_state(cam2, state_path, read_provenance=True) is True
        assert cam2.slot_records == prov_cam.slot_records
        # Type fidelity: tuple stayed a tuple, int 42 did not become "42".
        assert ("img", 7) in cam2.slot_records[2]
        assert 42 in cam2.slot_records[3] and "42" not in cam2.slot_records[3]

    def test_unoccupied_slot_stays_none(self, prov_cam, state_path):
        save_cam_state(prov_cam, state_path, write_provenance=True)
        cam2 = ContinuousCAM(key_dim=32, value_dim=8, max_entries=100,
                             track_provenance=True)
        load_cam_state(cam2, state_path, read_provenance=True)
        assert cam2.slot_records[4] is None
        assert cam2.slot_records[50] is None

    def test_no_sidecar_written_without_flag(self, prov_cam, state_path):
        save_cam_state(prov_cam, state_path)  # write_provenance defaults False
        assert not _provenance_path(state_path).exists()

    def test_load_without_flag_leaves_cold_default(self, prov_cam, state_path):
        save_cam_state(prov_cam, state_path, write_provenance=True)
        cam2 = ContinuousCAM(key_dim=32, value_dim=8, max_entries=100,
                             track_provenance=True)
        load_cam_state(cam2, state_path)  # read_provenance defaults False
        assert cam2.slot_records == [None] * 100

    def test_missing_sidecar_graceful(self, cam, state_path):
        # Binary saved without a sidecar; explicit sidecar load returns False.
        save_cam_state(cam, state_path)
        c = ContinuousCAM(key_dim=32, value_dim=8, max_entries=100,
                          track_provenance=True)
        assert load_provenance_sidecar(c, state_path) is False
        assert c.slot_records == [None] * 100

    def test_digest_mismatch_rejected(self, prov_cam, state_path):
        """A sidecar paired to a different binary file must not be applied."""
        save_cam_state(prov_cam, state_path, write_provenance=True)
        # Re-save the binary (new random-free content is identical here, so
        # mutate a buffer to force a different digest) without refreshing sidecar.
        prov_cam.keys[0] += 1.0
        save_cam_state(prov_cam, state_path)  # binary only — sidecar now stale
        cam2 = ContinuousCAM(key_dim=32, value_dim=8, max_entries=100,
                             track_provenance=True)
        load_cam_state(cam2, state_path, read_provenance=True)
        assert cam2.slot_records == [None] * 100

    def test_corrupt_sidecar_graceful(self, prov_cam, state_path):
        save_cam_state(prov_cam, state_path, write_provenance=True)
        _provenance_path(state_path).write_text("{ not valid json")
        cam2 = ContinuousCAM(key_dim=32, value_dim=8, max_entries=100,
                             track_provenance=True)
        assert load_cam_state(cam2, state_path, read_provenance=True) is True
        assert cam2.slot_records == [None] * 100  # binary ok, provenance cold

    def test_max_entries_mismatch_rejected(self, prov_cam, tmp_path):
        sp = tmp_path / "state.bin"
        save_cam_state(prov_cam, sp, write_provenance=True)
        # A CAM of different size can't even pass the binary load; test the
        # sidecar guard directly by hand-loading against a mismatched CAM.
        c = ContinuousCAM(key_dim=32, value_dim=8, max_entries=200,
                          track_provenance=True)
        assert load_provenance_sidecar(c, sp) is False
        assert c.slot_records == [None] * 200

    def test_save_noop_when_provenance_disabled(self, cam, state_path):
        save_cam_state(cam, state_path)  # cam has track_provenance=False
        assert save_provenance_sidecar(cam, state_path) is False
        assert not _provenance_path(state_path).exists()

    def test_save_refuses_unbound_sidecar(self, prov_cam, state_path):
        """No binary state file → no sidecar (would be unbound / null digest)."""
        assert not state_path.exists()
        assert save_provenance_sidecar(prov_cam, state_path) is False
        assert not _provenance_path(state_path).exists()

    def test_load_rejects_missing_state_file(self, prov_cam, state_path):
        """A sidecar with no matching binary state file must not be applied."""
        save_cam_state(prov_cam, state_path, write_provenance=True)
        state_path.unlink()  # remove the binary, leave the sidecar in place
        assert _provenance_path(state_path).exists()
        c = ContinuousCAM(key_dim=32, value_dim=8, max_entries=100,
                          track_provenance=True)
        assert load_provenance_sidecar(c, state_path) is False
        assert c.slot_records == [None] * 100

    @pytest.mark.parametrize("bad", ["[]", '{"records": []}', '"a string"', "42"])
    def test_load_rejects_wrong_shape_json(self, prov_cam, state_path, bad):
        """Valid JSON of the wrong shape → graceful cold fallback, not a crash."""
        save_cam_state(prov_cam, state_path, write_provenance=True)
        _provenance_path(state_path).write_text(bad)
        c = ContinuousCAM(key_dim=32, value_dim=8, max_entries=100,
                          track_provenance=True)
        # Must return False (or the digest guard fires first) — never raise.
        assert load_provenance_sidecar(c, state_path) is False
        assert c.slot_records == [None] * 100

    def test_load_rejects_malformed_records_field(self, prov_cam, state_path):
        """Valid version/digest but non-dict 'records' → graceful, not a crash."""
        import json
        save_cam_state(prov_cam, state_path, write_provenance=True)
        sidecar = _provenance_path(state_path)
        payload = json.loads(sidecar.read_text())
        payload["records"] = []  # wrong shape; version/max_entries/digest intact
        sidecar.write_text(json.dumps(payload))
        c = ContinuousCAM(key_dim=32, value_dim=8, max_entries=100,
                          track_provenance=True)
        assert load_provenance_sidecar(c, state_path) is False
        assert c.slot_records == [None] * 100
