"""
shutter_deck/ingest.py — inotify-based ingestion pipeline.

Watches a directory for new image files, batches them through DINOv2,
writes embeddings to ContinuousCAM via learn_local(), and logs metadata
to an append-only SQLite table.
"""

import fnmatch
import logging
import sqlite3
import time
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# File patterns accepted
DEFAULT_PATTERNS = ("*.ARW", "*.arw", "*.JPG", "*.jpg", "*.jpeg")

METADATA_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingested_images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath    TEXT    NOT NULL,
    timestamp   REAL    NOT NULL,
    assigned_slot INTEGER,
    neighbor_1_path  TEXT,
    neighbor_1_sim   REAL,
    neighbor_2_path  TEXT,
    neighbor_2_sim   REAL,
    neighbor_3_path  TEXT,
    neighbor_3_sim   REAL
);
CREATE INDEX IF NOT EXISTS idx_ingested_ts ON ingested_images(timestamp);
"""


class EncoderProtocol(Protocol):
    """Minimal interface for the embedding encoder."""
    def embed_file(self, path: Path) -> torch.Tensor: ...
    def embed_images(self, images: list[Image.Image]) -> torch.Tensor: ...
    def preprocess_batch(self, images: list[Image.Image]) -> torch.Tensor: ...
    def embed(self, tensor: torch.Tensor) -> torch.Tensor: ...


def init_metadata_db(db_path: Path) -> sqlite3.Connection:
    """Open (or create) the metadata SQLite database."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(METADATA_SCHEMA)
    return conn


def _matches_patterns(filename: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(filename, p) for p in patterns)


def _load_image(path: Path) -> Image.Image:
    """Load an image file, handling ARW via rawpy."""
    suffix = path.suffix.lower()
    if suffix in (".arw", ".dng", ".cr2", ".nef"):
        import rawpy
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
        if rgb.ndim == 3 and rgb.shape[2] == 1:
            rgb = rgb.squeeze(2)
        img = Image.fromarray(rgb)
        return img.convert("RGB")
    return Image.open(path).convert("RGB")


def _lookup_filepath_by_slot(conn: sqlite3.Connection, slot: int) -> str | None:
    """Find the filepath for a given assigned_slot."""
    row = conn.execute(
        "SELECT filepath FROM ingested_images WHERE assigned_slot = ? ORDER BY timestamp DESC LIMIT 1",
        (slot,),
    ).fetchone()
    return row[0] if row else None


def _record_metadata(
    conn: sqlite3.Connection,
    filepath: str,
    timestamp: float,
    assigned_slot: int,
    neighbors: list[tuple[str | None, float]],
) -> None:
    """Insert a row into the ingested_images table."""
    # Pad neighbors to exactly 3
    while len(neighbors) < 3:
        neighbors.append((None, 0.0))
    conn.execute(
        "INSERT INTO ingested_images "
        "(filepath, timestamp, assigned_slot, "
        " neighbor_1_path, neighbor_1_sim, neighbor_2_path, neighbor_2_sim, "
        " neighbor_3_path, neighbor_3_sim) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            filepath, timestamp, assigned_slot,
            neighbors[0][0], neighbors[0][1],
            neighbors[1][0], neighbors[1][1],
            neighbors[2][0], neighbors[2][1],
        ),
    )
    conn.commit()


def _find_top_k_neighbors(
    cam, embedding: torch.Tensor, conn: sqlite3.Connection, k: int = 3,
) -> list[tuple[str | None, float]]:
    """Find top-k nearest prototypes and resolve their file paths from metadata."""
    if not cam.occupied.any():
        return []

    valid_idx = cam.occupied.nonzero(as_tuple=True)[0]
    q_norm = F.normalize(embedding.float(), dim=-1)
    sims = (q_norm @ cam._keys_norm[valid_idx].T).squeeze(0)  # (N_occ,)

    actual_k = min(k, sims.numel())
    top_sims, top_locs = sims.topk(actual_k)
    top_slots = valid_idx[top_locs]

    neighbors = []
    for i in range(actual_k):
        slot = top_slots[i].item()
        sim = top_sims[i].item()
        fpath = _lookup_filepath_by_slot(conn, slot)
        neighbors.append((fpath, sim))
    return neighbors


def ingest_batch(
    cam,
    encoder,
    file_paths: list[Path],
    metadata_conn: sqlite3.Connection,
) -> int:
    """Ingest a batch of image files into the CAM.

    Returns the number of successfully ingested images.
    """
    images = []
    valid_paths = []
    for fp in file_paths:
        try:
            images.append(_load_image(fp))
            valid_paths.append(fp)
        except Exception:
            logger.exception("Failed to load %s — skipping.", fp)

    if not images:
        return 0

    # Embed batch
    embeddings = encoder.embed_images(images)  # (B, 1024)

    now = time.time()
    ingested = 0
    for i, (emb, fp) in enumerate(zip(embeddings, valid_paths)):
        emb = emb.unsqueeze(0)  # (1, 1024)

        # Autoassociative mode: value = embedding itself
        target = emb.clone()

        # Find neighbors BEFORE learning (so we don't match ourselves)
        neighbors = _find_top_k_neighbors(cam, emb, metadata_conn, k=3)

        # Learn into CAM
        cam.learn_local(emb, target)

        # Find the slot that was just assigned (newest occupied slot with matching key)
        # Approximate: highest-similarity occupied slot to this embedding
        if cam.occupied.any():
            valid_idx = cam.occupied.nonzero(as_tuple=True)[0]
            q_norm = F.normalize(emb.float(), dim=-1)
            sims = (q_norm @ cam._keys_norm[valid_idx].T).squeeze(0)
            best_loc = sims.argmax()
            assigned_slot = valid_idx[best_loc].item()
        else:
            assigned_slot = -1

        _record_metadata(
            metadata_conn,
            filepath=str(fp),
            timestamp=now,
            assigned_slot=assigned_slot,
            neighbors=neighbors,
        )
        ingested += 1

    return ingested


def watch_and_ingest(
    cam,
    encoder,
    watch_dir: Path,
    metadata_conn: sqlite3.Connection,
    save_fn,
    batch_size: int = 4,
    batch_window_sec: float = 2.0,
    checkpoint_interval: int = 100,
    patterns: tuple[str, ...] = DEFAULT_PATTERNS,
) -> None:
    """Main inotify event loop. Blocks forever.

    Parameters
    ----------
    save_fn : callable
        Called every *checkpoint_interval* ingestions to persist CAM state.
    """
    from inotify_simple import INotify, flags as iflags

    watch_dir = Path(watch_dir)
    watch_dir.mkdir(parents=True, exist_ok=True)

    inotify = INotify()
    inotify.add_watch(str(watch_dir), iflags.CLOSE_WRITE)
    logger.info("Watching %s for new images …", watch_dir)

    pending: list[Path] = []
    total_ingested = 0
    last_event_time = 0.0

    while True:
        # Read events with a timeout so we can flush the batch window
        timeout_ms = int(batch_window_sec * 1000) if pending else -1
        events = inotify.read(timeout=timeout_ms)

        for event in events:
            name = event.name
            if name and _matches_patterns(name, patterns):
                fpath = watch_dir / name
                if fpath.is_file():
                    pending.append(fpath)
                    last_event_time = time.time()

        # Flush batch if: max batch size reached OR batch window expired
        should_flush = (
            len(pending) >= batch_size
            or (pending and (time.time() - last_event_time) >= batch_window_sec)
        )

        if should_flush and pending:
            batch = pending[:batch_size]
            pending = pending[batch_size:]
            logger.info("Ingesting batch of %d files …", len(batch))

            try:
                n = ingest_batch(cam, encoder, batch, metadata_conn)
                total_ingested += n
                logger.info("Ingested %d/%d (total: %d)", n, len(batch), total_ingested)
            except Exception:
                logger.exception("Batch ingestion failed.")

            # Periodic checkpoint
            if total_ingested > 0 and total_ingested % checkpoint_interval == 0:
                logger.info("Checkpoint at %d ingestions.", total_ingested)
                try:
                    save_fn()
                except Exception:
                    logger.exception("Checkpoint save failed.")


def scan_and_ingest(
    cam,
    encoder,
    scan_dir: Path,
    metadata_conn: sqlite3.Connection,
    save_fn,
    batch_size: int = 4,
    patterns: tuple[str, ...] = DEFAULT_PATTERNS,
) -> int:
    """One-shot scan: process all matching files in *scan_dir* and return.

    This is the non-inotify alternative for macOS testing and batch backfills.
    """
    scan_dir = Path(scan_dir)
    files = sorted(
        p for p in scan_dir.iterdir()
        if p.is_file() and _matches_patterns(p.name, patterns)
    )
    if not files:
        logger.warning("No matching files in %s", scan_dir)
        return 0

    logger.info("Found %d files to ingest in %s", len(files), scan_dir)
    total = 0
    for i in range(0, len(files), batch_size):
        batch = files[i : i + batch_size]
        logger.info("Batch %d–%d of %d …", i + 1, i + len(batch), len(files))
        n = ingest_batch(cam, encoder, batch, metadata_conn)
        total += n

    save_fn()
    logger.info("Scan complete: %d/%d files ingested.", total, len(files))
    return total


if __name__ == "__main__":
    import argparse
    import sys
    import tomllib
    from functools import partial

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Shutter-Deck ingestion CLI.")
    parser.add_argument("--scan-dir", type=Path, required=True,
                        help="Directory to scan for images (one-shot mode).")
    parser.add_argument("--config", type=Path, default=Path("config.toml"),
                        help="Path to config TOML file.")
    args = parser.parse_args()

    if not args.config.exists():
        logger.error("Config not found: %s", args.config)
        sys.exit(1)

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    from shutter_deck.service import build_cam
    from shutter_deck.persistence import load_cam_state, save_cam_state
    from shutter_deck.eviction import install_density_eviction
    from shutter_deck.encoder import DINOv2Encoder

    cam = build_cam(cfg)
    install_density_eviction(cam, min_per_class=cfg["eviction"]["min_per_class"])
    state_path = Path(cfg["paths"]["state_file"])
    load_cam_state(cam, state_path)

    save_fn = partial(save_cam_state, cam, state_path)

    encoder = DINOv2Encoder(
        model_name=cfg["encoder"]["model_name"],
        device=cfg["encoder"]["device"],
    )

    metadata_conn = init_metadata_db(Path(cfg["paths"]["metadata_db"]))

    n = scan_and_ingest(
        cam=cam,
        encoder=encoder,
        scan_dir=args.scan_dir,
        metadata_conn=metadata_conn,
        save_fn=save_fn,
        batch_size=cfg["ingest"]["batch_size"],
        patterns=tuple(cfg["ingest"]["file_extensions"]),
    )
    print(f"\nIngested {n} images. State saved to {state_path}")
    metadata_conn.close()
