"""Append-only authoritative storage and deterministic lifecycle resolution."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable, Literal

from .models import MemoryRecord


LifecycleState = Literal["agent-readable", "superseded", "human-review"]


@dataclass(frozen=True, slots=True)
class ResolvedState:
    state: LifecycleState
    superseded_by: tuple[str, ...] = ()
    candidate_set_id: str | None = None


class MemoryLedger:
    """Own immutable records while resolving their current governance state."""

    def __init__(self, records: Iterable[MemoryRecord] = ()) -> None:
        self._records: list[MemoryRecord] = []
        self._by_id: dict[str, MemoryRecord] = {}
        for record in records:
            self.append(record)

    def append(self, record: MemoryRecord) -> None:
        if not isinstance(record, MemoryRecord):
            raise TypeError("ledger accepts MemoryRecord values only")
        if record.record_id in self._by_id:
            raise ValueError(f"duplicate record_id: {record.record_id}")
        self._records.append(record)
        self._by_id[record.record_id] = record

    @property
    def records(self) -> tuple[MemoryRecord, ...]:
        return tuple(self._records)

    def get(self, record_id: str) -> MemoryRecord:
        try:
            return self._by_id[record_id]
        except KeyError as exc:
            raise KeyError(f"unknown record_id: {record_id}") from exc

    def records_for_scope(self, scope: str) -> tuple[MemoryRecord, ...]:
        return tuple(
            sorted(
                (record for record in self._records if record.scope == scope),
                key=lambda record: (record.serial, record.record_id),
            )
        )

    def resolved_state(self, record_id: str) -> ResolvedState:
        record = self.get(record_id)
        family = self.records_for_scope(record.scope)
        maximum_serial = max(item.serial for item in family)
        latest = tuple(item for item in family if item.serial == maximum_serial)
        latest_ids = tuple(sorted(item.record_id for item in latest))

        if record.serial < maximum_serial:
            return ResolvedState(state="superseded", superseded_by=latest_ids)

        if len({item.value for item in latest}) > 1:
            return ResolvedState(
                state="human-review",
                candidate_set_id=self._fork_id(record.scope, maximum_serial, latest),
            )

        return ResolvedState(state="agent-readable")

    @staticmethod
    def _fork_id(
        scope: str, serial: int, latest: tuple[MemoryRecord, ...]
    ) -> str:
        members = "\n".join(
            f"{item.record_id}\x1f{item.value}"
            for item in sorted(latest, key=lambda item: item.record_id)
        )
        material = f"{scope}\n{serial}\n{members}".encode("utf-8")
        return f"fork-{sha256(material).hexdigest()[:16]}"
