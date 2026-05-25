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


def save_cam_state(cam, path: Path) -> None:
    """Serialize full CAM state to a binary file with atomic rename."""
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

    logger.info("Loaded CAM state: %d occupied slots from %s", n_occupied, path)
    return True
