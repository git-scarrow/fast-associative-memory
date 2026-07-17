"""Immutable values shared by the memory evaluation components."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal


def _nonempty(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def fact_scope(entity: str, relation: str) -> str:
    """Return a deterministic, separator-safe identity for a fact family."""

    separator = "\x1f"
    escaped_entity = _nonempty(entity, "entity").strip().replace(separator, separator * 2)
    escaped_relation = _nonempty(relation, "relation").strip().replace(
        separator, separator * 2
    )
    return f"{escaped_entity}{separator}{escaped_relation}"


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    record_id: str
    scope: str
    value: str
    content: str
    serial: int
    event_time: str | None = None
    source_id: str = "memory-eval"
    ingest_time: str | None = None

    def __post_init__(self) -> None:
        for field in (
            "record_id",
            "scope",
            "value",
            "content",
            "source_id",
        ):
            _nonempty(getattr(self, field), field)
        if not isinstance(self.serial, int) or isinstance(self.serial, bool) or self.serial < 0:
            raise ValueError("serial must be a non-negative integer")
        if self.event_time is not None:
            _nonempty(self.event_time, "event_time")
        if self.ingest_time is not None:
            _nonempty(self.ingest_time, "ingest_time")


@dataclass(frozen=True, slots=True)
class MemoryQuestion:
    query_id: str
    text: str
    scope: str
    answer: str

    def __post_init__(self) -> None:
        for field in ("query_id", "text", "scope", "answer"):
            _nonempty(getattr(self, field), field)


@dataclass(frozen=True, slots=True)
class RetrievedCandidate:
    record_id: str
    score: float
    rank: int

    def __post_init__(self) -> None:
        _nonempty(self.record_id, "record_id")
        if not isfinite(self.score):
            raise ValueError("score must be finite")
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank < 1:
            raise ValueError("rank must be a positive integer")


@dataclass(frozen=True, slots=True)
class AnswerObservation:
    status: Literal["answer", "abstain", "malformed"]
    answer: str | None

    def __post_init__(self) -> None:
        if self.status not in {"answer", "abstain", "malformed"}:
            raise ValueError(f"unsupported answer status: {self.status}")
        if self.status == "answer":
            if self.answer is None:
                raise ValueError("answer status requires an answer")
            _nonempty(self.answer, "answer")
        elif self.answer is not None:
            raise ValueError(f"{self.status} status cannot contain an answer")
