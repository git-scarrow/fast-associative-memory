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
WriteMode = Literal["allocate-only", "condense"]
_SINGLETON_WRITE_STAT_KEYS = frozenset(
    {"written", "merged", "allocated", "dropped"}
)


def _validate_singleton_write_stats(stats: object) -> dict[str, int]:
    if not isinstance(stats, Mapping) or set(stats) != _SINGLETON_WRITE_STAT_KEYS:
        raise ValueError(
            "CAM singleton write accounting must have exactly the deployed "
            "four-field key set"
        )
    if any(
        not isinstance(stats[name], int) or isinstance(stats[name], bool)
        for name in _SINGLETON_WRITE_STAT_KEYS
    ):
        raise ValueError("CAM singleton write accounting values must be integers")

    validated = {name: stats[name] for name in _SINGLETON_WRITE_STAT_KEYS}
    if validated["written"] != 1:
        raise ValueError("CAM singleton write accounting must report written=1")
    outcomes = tuple(validated[name] for name in ("merged", "allocated", "dropped"))
    if any(value not in {0, 1} for value in outcomes) or sum(outcomes) != 1:
        raise ValueError(
            "CAM singleton write accounting must report exactly one binary outcome"
        )
    return validated


class _HarnessWriteCAM(ContinuousCAM):
    """Harness-only write adapter around the byte-frozen deployed CAM.

    The override is active only while :meth:`ingest_one` delegates a singleton
    static-vigilance write to ``ContinuousCAM.learn_local``.  Retrieval and all
    other core behavior remain inherited unchanged.
    """

    _active_write_mode: WriteMode | None = None
    _active_scope_label: int | None = None

    def ingest_one(
        self,
        query: torch.Tensor,
        target: torch.Tensor,
        *,
        record_id: object,
        write_mode: WriteMode,
    ) -> None:
        if write_mode not in {"allocate-only", "condense"}:
            raise ValueError(f"unsupported write_mode: {write_mode!r}")
        if query.ndim != 2 or target.ndim != 2:
            raise ValueError("ingest_one requires batched query and target tensors")
        if query.size(0) != 1 or target.size(0) != 1:
            raise ValueError("ingest_one requires exactly one record")
        if self.dynamic_vigilance is not None:
            raise ValueError("harness write modes require static vigilance")
        if self._active_write_mode is not None:
            raise RuntimeError("nested harness CAM ingest is not supported")

        scope_label = int(self._labels_from_targets(target)[0].item())
        self._active_write_mode = write_mode
        self._active_scope_label = scope_label
        try:
            super().learn_local(query, target, record_ids=[record_id])
        finally:
            self._active_write_mode = None
            self._active_scope_label = None

    def _get_nearest_batch(
        self, queries: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode = self._active_write_mode
        if mode is None:
            return super()._get_nearest_batch(queries)

        if queries.size(0) != 1:
            raise RuntimeError("harness write selection requires one query")
        if mode == "allocate-only" or not self.occupied.any():
            return (
                torch.full((1,), -1, dtype=torch.long, device=queries.device),
                torch.full((1,), -1.0, device=queries.device),
            )

        assert self._active_scope_label is not None
        occupied = self.occupied.nonzero(as_tuple=True)[0]
        same_scope = (
            self.effective_slot_labels(occupied) == self._active_scope_label
        )
        candidates = occupied[same_scope]
        if candidates.numel() == 0:
            return (
                torch.full((1,), -1, dtype=torch.long, device=queries.device),
                torch.full((1,), -1.0, device=queries.device),
            )

        normalized_query = F.normalize(self._cast(queries), dim=-1)
        similarities = normalized_query @ self._keys_norm[candidates].T
        best_sims, best_locs = similarities.max(dim=1)
        return candidates[best_locs], best_sims.float()


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

        self.cam = _HarnessWriteCAM(
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
            occupied_before = int(self.cam.occupied.sum().item())
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
            self.cam.ingest_one(
                query,
                target,
                record_id=record.record_id,
                write_mode=mode,
            )
            stats = _validate_singleton_write_stats(self.cam.last_write_stats)
            occupied_after = int(self.cam.occupied.sum().item())
            local_evicted = stats["allocated"] - max(
                0, occupied_after - occupied_before
            )
            if local_evicted not in {0, 1}:
                raise ValueError(
                    "CAM singleton local eviction accounting is invalid: "
                    f"evicted={local_evicted}"
                )
            if local_evicted:
                raise ValueError(
                    "CAM index build lost writes: "
                    f"dropped={stats['dropped']}, evicted={local_evicted}"
                )
            for name in ("written", "merged", "allocated", "dropped"):
                totals[name] += stats[name]
            totals["evicted"] += local_evicted
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
        # FAM read path (validated on text by FAM-G-40: FAISS top-K -> softmax
        # vote). Rank by the prototype vote, NOT by re-scoring candidates against
        # the original uncondensed embedding table. `final_weights` is the
        # softmax over query-to-prototype-KEY similarity, so a drifted prototype
        # key changes what is retrieved and in what order — the condensed
        # representation is load-bearing, and the exemplar control (immutable
        # per-record keys) diverges from fam exactly when condensation drifts a
        # key. Provenance expansion recovers the authoritative ledger records the
        # prototype was formed from; the served score is the prototype's vote
        # weight.
        final_slots = trace.final_slots.reshape(-1).tolist()
        final_weights = trace.final_weights.reshape(-1).tolist()
        # PRIMARY (and only) ranking signal: the prototype vote weight, from
        # query-to-prototype-KEY similarity. This is what makes condensation
        # load-bearing — the previous read path replaced it with a flat cosine
        # rerank over the original uncondensed table, which collapsed fam onto
        # the exact-vector arm. We deliberately do NOT re-score records by their
        # original embeddings anywhere in the served order: that mild rerank is
        # exactly what re-collapses fam onto exact. Within a merged prototype the
        # members are co-equal (they formed the prototype); they are emitted in
        # deterministic record_id order, leaving recency/lifecycle ordering to
        # the governed compiler so the raw arm stays a naive baseline. The served
        # score is the prototype's vote weight (the condensed signal).
        slots_by_vote = sorted(
            zip(final_slots, final_weights),
            key=lambda sw: (-sw[1], sw[0]),
        )
        ranked: list[tuple[str, float]] = []
        seen: set[str] = set()
        for slot, weight in slots_by_vote:
            for record_id in sorted(str(r) for r in self.cam.records_for(int(slot))):
                if record_id in seen:
                    continue
                seen.add(record_id)
                ranked.append((record_id, float(weight)))
        ranked = ranked[:k]
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
