"""Retrieval indexes that expose authoritative ledger record IDs only."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from associative_core import ContinuousCAM

from .models import MemoryRecord, RetrievedCandidate


TensorLike = torch.Tensor | Sequence[float]


def _as_vector(value: TensorLike, *, label: str) -> torch.Tensor:
    # clone(): torch.as_tensor may return a view sharing storage with the
    # caller's tensor, so an in-place mutation after verify_manifest would
    # silently change rerank order under a passing seal.
    vector = torch.as_tensor(value, dtype=torch.float32).detach().cpu().clone()
    if vector.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional vector")
    if vector.numel() == 0:
        raise ValueError(f"{label} must not be empty")
    if not bool(torch.isfinite(vector).all()):
        raise ValueError(f"{label} must contain finite values")
    if float(torch.linalg.vector_norm(vector)) == 0.0:
        raise ValueError(f"{label} must be non-zero")
    return vector


def _prepare_table(
    records: Sequence[MemoryRecord], embeddings: Mapping[str, TensorLike]
) -> tuple[tuple[MemoryRecord, ...], dict[str, torch.Tensor], int]:
    frozen_records = tuple(records)
    if not frozen_records:
        raise ValueError("at least one record is required")
    record_ids = [record.record_id for record in frozen_records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("duplicate record_id in retriever input")
    missing = sorted(set(record_ids) - set(embeddings))
    if missing:
        raise ValueError(f"missing embeddings for: {', '.join(missing)}")
    extra = sorted(set(embeddings) - set(record_ids))
    if extra:
        raise ValueError(f"embeddings without records: {', '.join(extra)}")

    vectors = {
        record_id: _as_vector(embeddings[record_id], label=f"embedding {record_id}")
        for record_id in record_ids
    }
    dimensions = {vector.numel() for vector in vectors.values()}
    if len(dimensions) != 1:
        raise ValueError("all embeddings must have the same dimension")
    return frozen_records, vectors, dimensions.pop()


class ExactVectorRetriever:
    """Exact cosine search used as the non-condensing retrieval baseline."""

    def __init__(
        self, records: Sequence[MemoryRecord], embeddings: Mapping[str, TensorLike]
    ) -> None:
        self.records, self._embeddings, self.dimension = _prepare_table(
            records, embeddings
        )
        self._matrix = F.normalize(
            torch.stack([self._embeddings[record.record_id] for record in self.records]),
            dim=1,
        )

    def query(self, query_embedding: TensorLike, k: int) -> tuple[RetrievedCandidate, ...]:
        query = _validated_query(query_embedding, self.dimension, k)
        scores = (self._matrix @ F.normalize(query, dim=0)).tolist()
        ranked = sorted(
            zip(self.records, scores),
            key=lambda pair: (-pair[1], pair[0].record_id),
        )[: min(k, len(self.records))]
        return tuple(
            RetrievedCandidate(record.record_id, float(score), rank)
            for rank, (record, score) in enumerate(ranked, start=1)
        )


class FAMRetriever:
    """FAM scope prototypes whose provenance maps back to ledger record IDs."""

    def __init__(
        self,
        records: Sequence[MemoryRecord],
        embeddings: Mapping[str, TensorLike],
        *,
        prototype_k: int = 3,
        max_entries: int | None = None,
    ) -> None:
        self.records, self._embeddings, self.dimension = _prepare_table(
            records, embeddings
        )
        if not isinstance(prototype_k, int) or isinstance(prototype_k, bool) or prototype_k < 1:
            raise ValueError("prototype_k must be positive")

        grouped: dict[str, list[MemoryRecord]] = defaultdict(list)
        for record in self.records:
            grouped[record.scope].append(record)
        self._by_id = {record.record_id: record for record in self.records}
        self._scope_labels = {
            scope: label for label, scope in enumerate(sorted(grouped))
        }
        capacity = len(grouped) if max_entries is None else max_entries
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError("max_entries must be positive")

        self.cam = ContinuousCAM(
            key_dim=self.dimension,
            value_dim=len(grouped),
            max_entries=capacity,
            vigilance=-1.0,
            immutable_keys=True,
            adaptive_eviction=False,
            inference_k=min(prototype_k, capacity),
            track_provenance=True,
        )
        for scope in sorted(grouped):
            members = sorted(grouped[scope], key=lambda record: record.record_id)
            centroid = torch.stack(
                [F.normalize(self._embeddings[item.record_id], dim=0) for item in members]
            ).mean(dim=0)
            if float(torch.linalg.vector_norm(centroid)) == 0.0:
                centroid = self._embeddings[members[0].record_id]
            centroid = F.normalize(centroid, dim=0).unsqueeze(0)
            label = self._scope_labels[scope]
            target = F.one_hot(
                torch.tensor([label]), num_classes=len(grouped)
            ).float()
            for member in members:
                self.cam.learn_local(centroid, target, record_ids=[member.record_id])

        # A max_entries below the scope count silently LRU-evicts whole scopes
        # during construction: their provenance is cleared and their records
        # become permanently unretrievable in the fam arms only, with no error.
        # A sealed fam_max_entries that quietly drops scopes attests nothing.
        if self.prototype_count != len(grouped):
            raise ValueError(
                f"fam capacity {capacity} holds {self.prototype_count} of "
                f"{len(grouped)} scopes; {len(grouped) - self.prototype_count} "
                "were evicted during construction and their records are "
                "unretrievable"
            )

    @property
    def prototype_count(self) -> int:
        return int(self.cam.occupied.sum().item())

    def provenance_for_scope(self, scope: str) -> set[str]:
        result: set[str] = set()
        for slot in self.cam.occupied.nonzero(as_tuple=True)[0].tolist():
            for record_id in self.cam.records_for(slot):
                if self._by_id[str(record_id)].scope == scope:
                    result.add(str(record_id))
        return result

    def query(self, query_embedding: TensorLike, k: int) -> tuple[RetrievedCandidate, ...]:
        query = _validated_query(query_embedding, self.dimension, k)
        _, trace = self.cam.forward(query.unsqueeze(0), trace=True)
        candidate_ids = set().union(*self.cam.records_for_slots(trace.final_slots))
        normalized_query = F.normalize(query, dim=0)
        scored = [
            (
                str(record_id),
                float(F.cosine_similarity(
                    normalized_query,
                    self._embeddings[str(record_id)],
                    dim=0,
                )),
            )
            for record_id in candidate_ids
        ]
        ranked = sorted(scored, key=lambda pair: (-pair[1], pair[0]))[:k]
        return tuple(
            RetrievedCandidate(record_id, score, rank)
            for rank, (record_id, score) in enumerate(ranked, start=1)
        )


def _validated_query(query_embedding: TensorLike, dimension: int, k: int) -> torch.Tensor:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be positive")
    query = _as_vector(query_embedding, label="query")
    if query.numel() != dimension:
        raise ValueError(
            f"query dimension {query.numel()} does not match index dimension {dimension}"
        )
    return query
