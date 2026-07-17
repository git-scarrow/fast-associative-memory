"""Strict normalized interchange format for FactConsolidation inputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .models import MemoryQuestion, MemoryRecord, fact_scope


_RECORD_REQUIRED = {
    "type",
    "record_id",
    "entity",
    "relation",
    "value",
    "content",
    "serial",
}
_RECORD_OPTIONAL = {"event_time", "ingest_time", "source_id"}
_QUESTION_REQUIRED = {
    "type",
    "query_id",
    "entity",
    "relation",
    "question",
    "answer",
}


@dataclass(frozen=True, slots=True)
class FactConsolidationCorpus:
    records: tuple[MemoryRecord, ...]
    questions: tuple[MemoryQuestion, ...]


def load_normalized_jsonl(path: str | Path) -> FactConsolidationCorpus:
    records: list[MemoryRecord] = []
    questions: list[MemoryQuestion] = []
    record_ids: set[str] = set()
    query_ids: set[str] = set()

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: row must be an object")
            try:
                row_type = row.get("type")
                if row_type == "record":
                    _check_fields(row, _RECORD_REQUIRED, _RECORD_OPTIONAL)
                    record = _record_from(row)
                    if record.record_id in record_ids:
                        raise ValueError(f"duplicate record_id: {record.record_id}")
                    record_ids.add(record.record_id)
                    records.append(record)
                elif row_type == "question":
                    _check_fields(row, _QUESTION_REQUIRED, set())
                    question = _question_from(row)
                    if question.query_id in query_ids:
                        raise ValueError(f"duplicate query_id: {question.query_id}")
                    query_ids.add(question.query_id)
                    questions.append(question)
                else:
                    raise ValueError(f"unknown row type: {row_type!r}")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"line {line_number}: {exc}") from exc

    if not records:
        raise ValueError("normalized corpus contains no records")
    if not questions:
        raise ValueError("normalized corpus contains no questions")
    return FactConsolidationCorpus(tuple(records), tuple(questions))


def _check_fields(
    row: dict[str, Any], required: set[str], optional: set[str]
) -> None:
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")
    extra = sorted(set(row) - required - optional)
    if extra:
        raise ValueError(f"unexpected fields: {', '.join(extra)}")


def _record_from(row: dict[str, Any]) -> MemoryRecord:
    return MemoryRecord(
        record_id=row["record_id"],
        scope=fact_scope(row["entity"], row["relation"]),
        value=row["value"],
        content=row["content"],
        serial=row["serial"],
        event_time=row.get("event_time"),
        ingest_time=row.get("ingest_time"),
        source_id=row.get("source_id", "memoryagentbench-normalized"),
    )


def _question_from(row: dict[str, Any]) -> MemoryQuestion:
    return MemoryQuestion(
        query_id=row["query_id"],
        text=row["question"],
        scope=fact_scope(row["entity"], row["relation"]),
        answer=row["answer"],
    )
