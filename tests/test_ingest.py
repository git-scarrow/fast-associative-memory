"""
tests/test_ingest.py — Ingestion pipeline tests with mock encoder.

Validates:
  - SQLite metadata written correctly
  - Batch ingestion processes multiple files
  - File pattern matching works
  - Error handling for bad files
"""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import torch
import torch.nn.functional as F
import pytest
from PIL import Image

from associative_core import ContinuousCAM
from shutter_deck.eviction import install_density_eviction
from shutter_deck.ingest import (
    init_metadata_db,
    ingest_batch,
    _matches_patterns,
    DEFAULT_PATTERNS,
)


class MockEncoder:
    """Mock encoder that returns deterministic embeddings."""

    def __init__(self, dim: int = 32):
        self.dim = dim
        self._call_count = 0

    def embed_images(self, images: list[Image.Image]) -> torch.Tensor:
        B = len(images)
        self._call_count += 1
        # Deterministic but distinct embeddings per image
        embeddings = torch.randn(B, self.dim)
        return F.normalize(embeddings, dim=-1)

    def embed_file(self, path: Path) -> torch.Tensor:
        return self.embed_images([Image.new("RGB", (224, 224))])


@pytest.fixture
def cam():
    # Use key_dim=32 and value_dim=32 for autoassociative mode
    c = ContinuousCAM(key_dim=32, value_dim=32, max_entries=200, vigilance=0.85)
    install_density_eviction(c, min_per_class=2)
    return c


@pytest.fixture
def encoder():
    return MockEncoder(dim=32)


@pytest.fixture
def db_conn(tmp_path):
    return init_metadata_db(tmp_path / "test_meta.db")


@pytest.fixture
def sample_images(tmp_path):
    """Create temporary JPEG files."""
    paths = []
    for i in range(5):
        p = tmp_path / f"test_{i}.jpg"
        Image.new("RGB", (100, 100), color=(i * 50, 0, 0)).save(str(p), "JPEG")
        paths.append(p)
    return paths


class TestMetadataDB:
    def test_schema_created(self, db_conn):
        cursor = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ingested_images'"
        )
        assert cursor.fetchone() is not None

    def test_idempotent_init(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn1 = init_metadata_db(db_path)
        conn1.close()
        conn2 = init_metadata_db(db_path)
        cursor = conn2.execute("SELECT count(*) FROM ingested_images")
        assert cursor.fetchone()[0] == 0
        conn2.close()


class TestIngestBatch:
    def test_single_file(self, cam, encoder, db_conn, sample_images):
        n = ingest_batch(cam, encoder, [sample_images[0]], db_conn)
        assert n == 1
        assert cam.occupied.sum().item() >= 1

        # Metadata should be recorded
        rows = db_conn.execute("SELECT * FROM ingested_images").fetchall()
        assert len(rows) == 1
        assert str(sample_images[0]) in rows[0][1]  # filepath column

    def test_batch_of_multiple(self, cam, encoder, db_conn, sample_images):
        n = ingest_batch(cam, encoder, sample_images[:3], db_conn)
        assert n == 3
        rows = db_conn.execute("SELECT * FROM ingested_images").fetchall()
        assert len(rows) == 3

    def test_empty_batch(self, cam, encoder, db_conn):
        n = ingest_batch(cam, encoder, [], db_conn)
        assert n == 0

    def test_bad_file_skipped(self, cam, encoder, db_conn, tmp_path, sample_images):
        bad_file = tmp_path / "bad.jpg"
        bad_file.write_bytes(b"not an image")
        n = ingest_batch(cam, encoder, [bad_file, sample_images[0]], db_conn)
        # bad_file fails to load, sample_images[0] succeeds
        assert n == 1

    def test_metadata_has_neighbors(self, cam, encoder, db_conn, sample_images):
        # Ingest first file
        ingest_batch(cam, encoder, [sample_images[0]], db_conn)
        # Ingest second file — should have neighbor info
        ingest_batch(cam, encoder, [sample_images[1]], db_conn)

        rows = db_conn.execute("SELECT * FROM ingested_images ORDER BY id").fetchall()
        assert len(rows) == 2
        # Second row should have similarity score for neighbor 1
        # Column indices: 0=id, 1=filepath, 2=timestamp, 3=slot,
        #   4=n1_path, 5=n1_sim, 6=n2_path, 7=n2_sim, 8=n3_path, 9=n3_sim
        assert rows[1][5] is not None  # neighbor_1_sim should be set


class TestFilePatternMatching:
    @pytest.mark.parametrize("filename,expected", [
        ("photo.JPG", True),
        ("photo.jpg", True),
        ("photo.jpeg", True),
        ("photo.ARW", True),
        ("photo.arw", True),
        ("photo.png", False),
        ("photo.txt", False),
        ("photo.RAW", False),
    ])
    def test_patterns(self, filename, expected):
        assert _matches_patterns(filename, DEFAULT_PATTERNS) == expected
