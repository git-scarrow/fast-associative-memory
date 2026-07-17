"""Retrieval indexes that expose authoritative ledger record IDs only."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from typing import Literal

import torch
import torch.nn.functional as F

from associative_core import ContinuousCAM

from .models import MemoryRecord, RetrievedCandidate


TensorLike = torch.Tensor | Sequence[float]


@dataclass(frozen=True, slots=True)
class CAMIndexSettings:
    max_entries: int
    prototype_k: int
    vigilance: float
    hebb_lr: float
    key_lr: float
    ema_beta: float
    inference_temp: float
    use_bfloat16: bool
    adaptive_eviction: bool
    use_lfu: bool

    def __post_init__(self) -> None:
        for name in ("max_entries", "prototype_k"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "vigilance",
            "hebb_lr",
            "key_lr",
            "ema_beta",
            "inference_temp",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            if not isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0 <= self.vigilance <= 1:
            raise ValueError("vigilance must be between 0 and 1")
        if self.inference_temp <= 0:
            raise ValueError("inference_temp must be positive")
        for name in ("use_bfloat16", "adaptive_eviction", "use_lfu"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class IndexBuildAttestation:
    mode: Literal["allocate-only", "condense"]
    written: int
    merged: int
    allocated: int
    dropped: int
    evicted: int
    prototype_count: int
    key_drifted_merges: int
    index_sha256: str


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


class _CAMRecordRetriever:
    """Sequential CAM index whose prototypes retain authoritative provenance."""

    def __init__(
        self,
        records: Sequence[MemoryRecord],
        embeddings: Mapping[str, TensorLike],
        *,
        settings: CAMIndexSettings,
        mode: Literal["allocate-only", "condense"],
    ) -> None:
        frozen_records = tuple(records)
        if settings.max_entries != len(frozen_records):
            raise ValueError(
                f"max_entries {settings.max_entries} must equal record count "
                f"{len(frozen_records)}"
            )
        self.records, self._embeddings, self.dimension = _prepare_table(
            frozen_records, embeddings
        )
        self._by_id = {record.record_id: record for record in self.records}
        self._scope_labels: dict[str, int] = {}
        for record in self.records:
            if record.scope not in self._scope_labels:
                self._scope_labels[record.scope] = len(self._scope_labels)

        self.cam = ContinuousCAM(
            key_dim=self.dimension,
            value_dim=len(self._scope_labels),
            max_entries=settings.max_entries,
            vigilance=settings.vigilance,
            hebb_lr=settings.hebb_lr,
            key_lr=settings.key_lr,
            ema_beta=settings.ema_beta,
            inference_temp=settings.inference_temp,
            use_bfloat16=settings.use_bfloat16,
            adaptive_eviction=settings.adaptive_eviction,
            use_lfu=settings.use_lfu,
            immutable_keys=(mode == "allocate-only"),
            inference_k=min(settings.prototype_k, settings.max_entries),
            track_provenance=True,
        )

        totals = {
            name: 0
            for name in ("written", "merged", "allocated", "dropped", "evicted")
        }
        key_drifted_merges = 0
        for record in self.records:
            before_keys = (
                self.cam.keys[self.cam.occupied].detach().clone()
                if mode == "condense"
                else None
            )
            query = self._embeddings[record.record_id].unsqueeze(0)
            label = self._scope_labels[record.scope]
            target = F.one_hot(
                torch.tensor([label]), num_classes=len(self._scope_labels)
            ).float()
            self.cam.learn_local(
                query,
                target,
                record_ids=[record.record_id],
                write_mode=mode,
            )
            stats = self.cam.last_write_stats
            for name in totals:
                totals[name] += int(stats[name])
            if mode == "condense":
                after_keys = self.cam.keys[self.cam.occupied].detach().clone()
                assert before_keys is not None
                if stats["merged"] and not torch.equal(before_keys, after_keys):
                    key_drifted_merges += 1

        if totals["dropped"] or totals["evicted"]:
            raise ValueError(
                "CAM index build lost writes: "
                f"dropped={totals['dropped']}, evicted={totals['evicted']}"
            )
        index_sha256 = self._validate_provenance_and_hash()
        self.attestation = IndexBuildAttestation(
            mode=mode,
            written=totals["written"],
            merged=totals["merged"],
            allocated=totals["allocated"],
            dropped=totals["dropped"],
            evicted=totals["evicted"],
            prototype_count=self.prototype_count,
            key_drifted_merges=key_drifted_merges,
            index_sha256=index_sha256,
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

    def _validate_provenance_and_hash(self) -> str:
        expected_ids = set(self._by_id)
        observed_ids: set[str] = set()
        rows: list[dict[str, object]] = []
        for slot in self.cam.occupied.nonzero(as_tuple=True)[0].tolist():
            provenance = {str(record_id) for record_id in self.cam.records_for(slot)}
            if not provenance:
                raise ValueError(f"occupied CAM slot {slot} has empty provenance")
            unknown = provenance - expected_ids
            if unknown:
                raise ValueError(
                    f"CAM provenance contains unknown record IDs: {sorted(unknown)}"
                )
            duplicate = provenance & observed_ids
            if duplicate:
                raise ValueError(
                    f"CAM provenance duplicates record IDs: {sorted(duplicate)}"
                )
            scopes = {self._by_id[record_id].scope for record_id in provenance}
            if len(scopes) != 1:
                raise ValueError(f"CAM slot {slot} mixes record scopes")
            scope = next(iter(scopes))
            semantic_label = int(self.cam.slot_labels[slot].item())
            if semantic_label != self._scope_labels[scope]:
                raise ValueError(f"CAM slot {slot} has a cross-scope semantic label")
            observed_ids.update(provenance)
            rows.append(
                {
                    "slot": slot,
                    "key": self.cam.keys[slot].detach().float().cpu().tolist(),
                    "semantic_label": semantic_label,
                    "provenance_ids": sorted(provenance),
                }
            )
        if observed_ids != expected_ids:
            missing = sorted(expected_ids - observed_ids)
            extra = sorted(observed_ids - expected_ids)
            raise ValueError(
                f"CAM provenance does not match inputs: missing={missing}, extra={extra}"
            )
        canonical = json.dumps(
            rows, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

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


class ExemplarCAMRetriever(_CAMRecordRetriever):
    """Matched CAM control that stores every record as an immutable exemplar."""

    def __init__(
        self,
        records: Sequence[MemoryRecord],
        embeddings: Mapping[str, TensorLike],
        *,
        settings: CAMIndexSettings,
    ) -> None:
        super().__init__(records, embeddings, settings=settings, mode="allocate-only")


class FAMRetriever(_CAMRecordRetriever):
    """Live FAM condensation whose provenance maps back to ledger record IDs."""

    def __init__(
        self,
        records: Sequence[MemoryRecord],
        embeddings: Mapping[str, TensorLike],
        *,
        settings: CAMIndexSettings,
    ) -> None:
        super().__init__(records, embeddings, settings=settings, mode="condense")


def _validated_query(query_embedding: TensorLike, dimension: int, k: int) -> torch.Tensor:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be positive")
    query = _as_vector(query_embedding, label="query")
    if query.numel() != dimension:
        raise ValueError(
            f"query dimension {query.numel()} does not match index dimension {dimension}"
        )
    return query
