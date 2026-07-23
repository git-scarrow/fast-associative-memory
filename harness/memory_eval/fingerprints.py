"""Canonical serialization and digest primitives for the input seal.

This is the single source of truth for how records, questions, and embeddings
are canonicalized before hashing. It is a leaf module — importable by both the
manifest (seal time) and the runner (run time) — so the fingerprints the
runner recomputes to police its inputs are byte-identical to the ones the
seal wrote, by construction rather than by convention.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from hashlib import sha256
import json
from typing import Any

import torch

from .models import MemoryQuestion
from .retrievers import TensorLike

__all__ = [
    "canonical_json",
    "fingerprint",
    "embedding_value",
    "query_side_fingerprints",
    "query_embedding_sha256",
]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def fingerprint(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def embedding_value(embeddings: Mapping[str, TensorLike]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for item_id in sorted(embeddings):
        vector = torch.as_tensor(embeddings[item_id], dtype=torch.float32).detach().cpu()
        if vector.ndim != 1 or not bool(torch.isfinite(vector).all()):
            raise ValueError(f"invalid embedding for {item_id}")
        result[str(item_id)] = [float(value) for value in vector.tolist()]
    return result


def query_side_fingerprints(
    questions: Sequence[MemoryQuestion],
    query_embeddings: Mapping[str, TensorLike],
) -> dict[str, str]:
    """Fingerprint ONLY the query-side inputs, byte-identical to the seal.

    ``verify_manifest`` binds the runner's construction inputs, but ``run()``
    takes questions and query embeddings as free parameters — the review
    demonstrated a foreign embedding table and a never-sealed question
    executing under a passing sealed digest. These two fingerprints let the
    runner recompute, at run time, exactly what the manifest sealed and
    refuse on mismatch.
    """
    return {
        "questions": fingerprint([asdict(question) for question in questions]),
        "query_embeddings": fingerprint(embedding_value(query_embeddings)),
    }


def query_embedding_sha256(embedding: TensorLike) -> str:
    """Canonical digest of ONE query embedding, for per-row provenance.

    Uses the seal's float canonicalization, so a row's recorded digest is
    directly comparable to the sealed query-embedding table after the one
    real run — an embedding swap between seal and execution (the
    darwin-seal/gentoo-run drift this project has been burned by) becomes
    auditable per row instead of vanishing.
    """
    vector = torch.as_tensor(embedding, dtype=torch.float32).detach().cpu()
    if vector.ndim != 1 or not bool(torch.isfinite(vector).all()):
        raise ValueError("invalid query embedding")
    return fingerprint([float(value) for value in vector.tolist()])
