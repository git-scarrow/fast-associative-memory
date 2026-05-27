"""
shutter_deck/persistence.py — Versioned binary persistence for ContinuousCAM state.

Layout:
  [magic: 4 bytes "FCAM"]
  [version: 2 bytes uint16]
  [max_entries: 4 bytes uint32]
  [key_dim: 4 bytes uint32]
  [value_dim: 4 bytes uint32]
  [n_occupied: 4 bytes uint32]
  --- tensor blobs (contiguous float32 / float64 / bool as appropriate) ---
"""

import hashlib
import json
import logging
import struct
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

MAGIC = b"FCAM"
# v3: added slot_labels (int64) — first-class semantic identity decoupled from
# the `values` payload. v2 files are rejected by the version check and cold-start
# (existing graceful-fallback contract); a cold start re-stamps labels at write.
VERSION = 3
HEADER_FMT = "<4sHIIII"  # magic(4) + version(2) + max_entries(4) + key_dim(4) + value_dim(4) + n_occupied(4)
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# NOTE: ContinuousCAM.slot_records (per-slot provenance) is intentionally NOT
# part of the binary format above — it holds arbitrary Python hashables, not a
# fixed-width tensor. It is persisted *separately* in an opt-in JSON sidecar
# (see save_provenance_sidecar / load_provenance_sidecar and the write_provenance
# / read_provenance flags). On a binary load, slot_records is still reset to the
# cold default first, so stale provenance can never survive a load and mismatch
# the loaded tensors; the sidecar is only re-applied when explicitly requested
# AND its bound digest matches the binary file it was saved alongside.
#
# Ordered list of (buffer_name, numpy_dtype) for serialization
_TENSOR_FIELDS = [
    ("keys", np.float32),
    ("values", np.float32),
    ("slot_labels", np.int64),
    ("occupied", np.bool_),
    ("last_seen", np.float64),
    ("usage", np.float32),
    ("_keys_norm", np.float32),
    ("hit_counts", np.int32),
]


def _tensor_byte_size(cam, name: str, dtype) -> int:
    buf = getattr(cam, name)
    return int(np.prod(buf.shape)) * np.dtype(dtype).itemsize


def save_cam_state(cam, path: Path, write_provenance: bool = False) -> None:
    """Serialize full CAM state to a binary file with atomic rename.

    When ``write_provenance`` is True and the CAM tracks provenance, a JSON
    sidecar (``<path>.prov.json``) is written *after* the binary file so its
    bound digest covers the final on-disk bytes. Defaults to False, leaving the
    binary format and existing callers byte-for-byte unchanged.
    """
    path = Path(path)
    tmp_path = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)

    n_occupied = int(cam.occupied.sum().item())
    header = struct.pack(
        HEADER_FMT,
        MAGIC, VERSION,
        cam.max_entries, cam.key_dim, cam.value_dim, n_occupied,
    )

    with open(tmp_path, "wb") as f:
        f.write(header)
        for name, dtype in _TENSOR_FIELDS:
            buf = getattr(cam, name)
            # Convert to CPU numpy with target dtype
            arr = buf.detach().cpu().numpy().astype(dtype)
            f.write(arr.tobytes())

    # Atomic rename (POSIX guarantees)
    tmp_path.rename(path)
    logger.info("Saved CAM state: %d occupied slots → %s", n_occupied, path)

    if write_provenance:
        save_provenance_sidecar(cam, path)


def load_cam_state(cam, path: Path, read_provenance: bool = False) -> bool:
    """Load CAM state from a binary file. Returns True on success, False on mismatch/missing.

    When ``read_provenance`` is True and the CAM tracks provenance, the JSON
    sidecar is re-applied *after* the cold reset of ``slot_records`` — but only
    if its bound digest matches this binary file (otherwise slot_records is left
    at the cold default). Defaults to False, preserving the existing contract
    that a load always clears stale provenance.
    """
    path = Path(path)
    if not path.exists():
        logger.info("No state file at %s — starting cold.", path)
        return False

    with open(path, "rb") as f:
        raw_header = f.read(HEADER_SIZE)
        if len(raw_header) < HEADER_SIZE:
            logger.warning("State file too small — starting cold.")
            return False

        magic, version, max_entries, key_dim, value_dim, n_occupied = struct.unpack(
            HEADER_FMT, raw_header
        )

        if magic != MAGIC:
            logger.warning("Bad magic %r — starting cold.", magic)
            return False
        if version != VERSION:
            logger.warning("Version mismatch (file=%d, code=%d) — starting cold.", version, VERSION)
            return False
        if max_entries != cam.max_entries or key_dim != cam.key_dim or value_dim != cam.value_dim:
            logger.warning(
                "Dimension mismatch (file: %d/%d/%d, cam: %d/%d/%d) — starting cold.",
                max_entries, key_dim, value_dim,
                cam.max_entries, cam.key_dim, cam.value_dim,
            )
            return False

        for name, dtype in _TENSOR_FIELDS:
            buf = getattr(cam, name)
            n_elements = int(np.prod(buf.shape))
            n_bytes = n_elements * np.dtype(dtype).itemsize
            raw = f.read(n_bytes)
            if len(raw) < n_bytes:
                logger.warning("Truncated state file at field %s — starting cold.", name)
                return False
            arr = np.frombuffer(raw, dtype=dtype).reshape(buf.shape)
            buf.copy_(torch.from_numpy(arr.copy()))

    # slot_records is not part of the binary format. Reset it to the cold
    # default so stale in-memory provenance from prior use of this CAM cannot
    # survive a load and mismatch the freshly loaded tensor state.
    if getattr(cam, "track_provenance", False):
        cam.slot_records = [None] * cam.max_entries
        if read_provenance:
            load_provenance_sidecar(cam, path)

    logger.info("Loaded CAM state: %d occupied slots from %s", n_occupied, path)
    return True


# ----------------------------------------------------------------------------
# Provenance sidecar (slot_records) — JSON, kept out of the binary format.
# ----------------------------------------------------------------------------
PROVENANCE_SUFFIX = ".prov.json"
PROVENANCE_VERSION = 1


def _provenance_path(state_path: Path) -> Path:
    """Sidecar path for a binary state file: ``<state_path>.prov.json``.

    Appends rather than replacing the suffix so a ``foo.bin`` state and its
    ``foo.bin.prov.json`` sidecar stay unambiguously paired.
    """
    return Path(str(Path(state_path)) + PROVENANCE_SUFFIX)


def _file_digest(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes (binds a sidecar to its state file)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _encode_record(r):
    """Encode one hashable provenance record into a type-tagged JSON value.

    JSON collapses int/str/tuple distinctions that ``slot_records`` relies on
    (a tuple key is not its list form; ``5`` is not ``"5"``). Tagging the type
    preserves an exact round-trip. ``bool`` is checked before ``int`` because
    ``bool`` is an ``int`` subclass.
    """
    if isinstance(r, bool):
        return {"t": "bool", "v": r}
    if isinstance(r, int):
        return {"t": "int", "v": r}
    if isinstance(r, float):
        return {"t": "float", "v": r}
    if isinstance(r, str):
        return {"t": "str", "v": r}
    if isinstance(r, tuple):
        return {"t": "tuple", "v": [_encode_record(x) for x in r]}
    raise TypeError(
        f"Unsupported provenance record type {type(r).__name__!r}: {r!r}. "
        "slot_records must hold int/float/str/bool or (nested) tuples thereof."
    )


def _decode_record(obj):
    """Inverse of :func:`_encode_record`."""
    t = obj["t"]
    v = obj["v"]
    if t == "bool":
        return bool(v)
    if t == "int":
        return int(v)
    if t == "float":
        return float(v)
    if t == "str":
        return str(v)
    if t == "tuple":
        return tuple(_decode_record(x) for x in v)
    raise ValueError(f"Unknown provenance record tag {t!r}")


def save_provenance_sidecar(cam, state_path: Path) -> bool:
    """Write ``cam.slot_records`` to a JSON sidecar next to ``state_path``.

    No-op (returns False) when provenance tracking is disabled or when the
    binary state file does not yet exist — a sidecar is only meaningful bound to
    real on-disk bytes, so it is never written with a null digest. The sidecar
    stores a SHA-256 digest of the binary state file so a later load can refuse
    to apply provenance onto a state file it was not saved alongside. Written
    via tmp-file + atomic rename, mirroring the binary writer.
    """
    if not getattr(cam, "track_provenance", False) or cam.slot_records is None:
        return False

    state_path = Path(state_path)
    if not state_path.exists():
        # A sidecar is only meaningful bound to a binary state file. Refuse to
        # write one with a null digest that nothing could later validate against.
        logger.warning(
            "No binary state file at %s — refusing to write an unbound provenance sidecar.",
            state_path)
        return False
    sidecar = _provenance_path(state_path)
    digest = _file_digest(state_path)

    records = {
        str(slot): [_encode_record(r) for r in rec]
        for slot, rec in enumerate(cam.slot_records)
        if rec  # skip None and empty sets
    }
    payload = {
        "provenance_version": PROVENANCE_VERSION,
        "max_entries": cam.max_entries,
        "state_digest": digest,
        "records": records,
    }

    tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    tmp.rename(sidecar)
    logger.info("Saved provenance sidecar: %d slots → %s", len(records), sidecar)
    return True


def load_provenance_sidecar(cam, state_path: Path) -> bool:
    """Restore ``cam.slot_records`` from the JSON sidecar for ``state_path``.

    Returns True only when the sidecar exists, parses, and is consistent with
    the CAM (version + ``max_entries``) and the binary state file (digest). On
    any mismatch or error it logs a warning, leaves ``slot_records`` untouched
    (the caller's cold default), and returns False — the same graceful-fallback
    contract as :func:`load_cam_state`.
    """
    if not getattr(cam, "track_provenance", False):
        return False

    state_path = Path(state_path)
    if not state_path.exists():
        # No tensor state to bind to — a provenance set with no matching binary
        # cannot be validated, so never apply one.
        logger.warning(
            "No binary state file at %s — provenance stays cold.", state_path)
        return False
    sidecar = _provenance_path(state_path)
    if not sidecar.exists():
        logger.info("No provenance sidecar at %s — provenance stays cold.", sidecar)
        return False

    try:
        with open(sidecar, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Unreadable provenance sidecar %s (%s) — provenance stays cold.", sidecar, e)
        return False

    if not isinstance(payload, dict):
        logger.warning(
            "Provenance sidecar %s is not a JSON object — provenance stays cold.", sidecar)
        return False

    if payload.get("provenance_version") != PROVENANCE_VERSION:
        logger.warning(
            "Provenance sidecar version mismatch (file=%r, code=%d) — provenance stays cold.",
            payload.get("provenance_version"), PROVENANCE_VERSION)
        return False
    if payload.get("max_entries") != cam.max_entries:
        logger.warning(
            "Provenance sidecar max_entries mismatch (file=%r, cam=%d) — provenance stays cold.",
            payload.get("max_entries"), cam.max_entries)
        return False

    expected = payload.get("state_digest")
    actual = _file_digest(state_path)  # state_path existence checked above
    if expected != actual:
        logger.warning(
            "Provenance sidecar digest mismatch for %s — provenance stays cold.", state_path)
        return False

    records = payload.get("records", {})
    if not isinstance(records, dict):
        logger.warning(
            "Provenance sidecar %s has malformed 'records' — provenance stays cold.", sidecar)
        return False
    restored = [None] * cam.max_entries
    try:
        for slot_str, rec_list in records.items():
            slot = int(slot_str)
            if not (0 <= slot < cam.max_entries):
                logger.warning("Provenance slot %d out of range — provenance stays cold.", slot)
                return False
            restored[slot] = {_decode_record(x) for x in rec_list}
    except (ValueError, KeyError, TypeError) as e:
        logger.warning("Corrupt provenance record in %s (%s) — provenance stays cold.", sidecar, e)
        return False

    cam.slot_records = restored
    logger.info("Loaded provenance sidecar: %d slots from %s", len(records), sidecar)
    return True
