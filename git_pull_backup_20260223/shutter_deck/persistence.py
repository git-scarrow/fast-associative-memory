"""
shutter_deck/persistence.py — Versioned binary persistence for ContinuousCAM state.

v2 format is sparse for per-slot tensors: only occupied rows are serialized.

Layout (little-endian):
  [magic: 4 bytes "FCAM"]
  [version: 2 bytes uint16]
  [file_max_entries: 4 bytes uint32]  # informational only; loader uses current cam.max_entries
  [key_dim: 4 bytes uint32]
  [value_dim: 4 bytes uint32]
  [n_occupied: 4 bytes uint32]
  --- sparse tensor blobs ---
  keys            (n_occ, key_dim)   float32
  values          (n_occ, value_dim) float32
  last_seen       (n_occ,)           float64
  usage           (n_occ,)           float32
  hit_counts      (n_occ,)           int32
  prototype_density (n_occ,)         float32
  slot_indices    (n_occ,)           uint32

_keys_norm is not serialized; it is reconstructed from keys on load.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

MAGIC = b"FCAM"
VERSION = 2
HEADER_FMT = "<4sHIIII"  # magic + version + file_max_entries + key_dim + value_dim + n_occupied
HEADER_SIZE = struct.calcsize(HEADER_FMT)

_SPARSE_FIELDS = [
    ("keys", np.float32),
    ("values", np.float32),
    ("last_seen", np.float64),
    ("usage", np.float32),
    ("hit_counts", np.int32),
    ("prototype_density", np.float32),
]
_SLOT_INDEX_DTYPE = np.uint32


def _as_numpy(buf: torch.Tensor, dtype) -> np.ndarray:
    t = buf.detach().cpu()
    if t.dtype == torch.bfloat16:
        t = t.float()
    return t.numpy().astype(dtype, copy=False)


def _write_array(f, arr: np.ndarray) -> None:
    f.write(np.ascontiguousarray(arr).tobytes())


def _read_array(f, shape: tuple[int, ...], dtype) -> np.ndarray | None:
    n_elems = int(np.prod(shape))
    n_bytes = n_elems * np.dtype(dtype).itemsize
    raw = f.read(n_bytes)
    if len(raw) < n_bytes:
        return None
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def _zero_state(cam) -> None:
    """Reset mutable buffers so loading into a reused CAM is deterministic."""
    cam.keys.zero_()
    cam.values.zero_()
    cam.occupied.zero_()
    cam.last_seen.zero_()
    cam._keys_norm.zero_()
    cam.usage.zero_()
    cam.hit_counts.zero_()
    cam.prototype_density.zero_()
    # These are global statistics, not per-slot sparse tensors; reset to defaults.
    cam.class_means.zero_()
    cam.class_vars.fill_(1.0)


def save_cam_state(cam, path: Path) -> None:
    """Serialize sparse CAM state to a binary file with atomic rename."""
    path = Path(path)
    tmp_path = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)

    occ_idx = cam.occupied.nonzero(as_tuple=True)[0]
    n_occupied = int(occ_idx.numel())
    occ_idx_np = _as_numpy(occ_idx, _SLOT_INDEX_DTYPE)

    header = struct.pack(
        HEADER_FMT,
        MAGIC,
        VERSION,
        cam.max_entries,
        cam.key_dim,
        cam.value_dim,
        n_occupied,
    )

    with open(tmp_path, "wb") as f:
        f.write(header)
        for name, dtype in _SPARSE_FIELDS:
            arr = _as_numpy(getattr(cam, name)[occ_idx], dtype)
            _write_array(f, arr)
        _write_array(f, occ_idx_np)

    tmp_path.rename(path)
    logger.info("Saved CAM state (v2 sparse): %d occupied slots → %s", n_occupied, path)


def load_cam_state(cam, path: Path) -> bool:
    """Load CAM state from a binary file. Returns True on success, False on mismatch/missing."""
    path = Path(path)
    if not path.exists():
        logger.info("No state file at %s — starting cold.", path)
        return False

    with open(path, "rb") as f:
        raw_header = f.read(HEADER_SIZE)
        if len(raw_header) < HEADER_SIZE:
            logger.warning("State file too small — starting cold.")
            return False

        magic, version, file_max_entries, key_dim, value_dim, n_occupied = struct.unpack(HEADER_FMT, raw_header)

        if magic != MAGIC:
            logger.warning("Bad magic %r — starting cold.", magic)
            return False
        if version == 1:
            logger.warning(
                "Unsupported CAM state format v1 at %s. Delete the old state file and cold-start to regenerate v2.",
                path,
            )
            return False
        if version != VERSION:
            logger.warning("Version mismatch (file=%d, code=%d) — starting cold.", version, VERSION)
            return False
        if key_dim != cam.key_dim or value_dim != cam.value_dim:
            logger.warning(
                "Dimension mismatch (file key/value=%d/%d, cam=%d/%d) — starting cold.",
                key_dim, value_dim, cam.key_dim, cam.value_dim,
            )
            return False

        # Read sparse blobs into numpy first; don't mutate CAM on truncated/corrupt files.
        arrays: dict[str, np.ndarray] = {}
        for name, dtype in _SPARSE_FIELDS:
            buf = getattr(cam, name)
            # keys/values are 2D, others are 1D.
            if buf.ndim == 2:
                shape = (n_occupied, buf.shape[1])
            else:
                shape = (n_occupied,)
            arr = _read_array(f, shape, dtype)
            if arr is None:
                logger.warning("Truncated state file at field %s — starting cold.", name)
                return False
            arrays[name] = arr

        slot_indices = _read_array(f, (n_occupied,), _SLOT_INDEX_DTYPE)
        if slot_indices is None:
            logger.warning("Truncated state file at field slot_indices — starting cold.")
            return False

    # Reinitialize mutable state and load sparse rows.
    _zero_state(cam)

    valid = slot_indices < cam.max_entries
    if np.any(~valid):
        dropped = int((~valid).sum())
        logger.warning(
            "Dropping %d state rows with slot indices >= current max_entries (%d). "
            "(file max_entries=%d, current=%d)",
            dropped, cam.max_entries, file_max_entries, cam.max_entries,
        )

    slot_indices = slot_indices[valid].astype(np.int64, copy=False)
    slots_t = torch.from_numpy(slot_indices.copy()).long()

    for name, _dtype in _SPARSE_FIELDS:
        arr = arrays[name][valid]
        dest = getattr(cam, name)
        src = torch.from_numpy(arr.copy()).to(dtype=dest.dtype, device=dest.device)
        dest[slots_t] = src

    cam.occupied[slots_t] = True
    if slots_t.numel() > 0:
        cam._update_key_norm(slots_t)

    logger.info(
        "Loaded CAM state (v2 sparse): %d/%d occupied rows from %s (file max_entries=%d, current=%d)",
        int(slots_t.numel()),
        n_occupied,
        path,
        file_max_entries,
        cam.max_entries,
    )
    return True
