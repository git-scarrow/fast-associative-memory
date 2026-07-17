"""Adapt authoritative ledger records into raw or governed context inputs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from .ledger import MemoryLedger


POLICY_VERSION = "memory-eval-state-v1+pr13-disposition-1.0"
ADAPTER_ID = "memory-eval-ledger-v1"


@dataclass(frozen=True, slots=True)
class RawContext:
    block: str
    rendered_item_ids: tuple[str, ...]
    token_count: int


def _item_id(source_id: str, record_id: str) -> str:
    return f"{source_id}:{record_id}"


def context_item_for_record(ledger: MemoryLedger, record_id: str) -> dict[str, Any]:
    record = ledger.get(record_id)
    resolution = ledger.resolved_state(record_id)
    family = ledger.records_for_scope(record.scope)
    native_to_item = {
        member.record_id: _item_id(member.source_id, member.record_id)
        for member in family
    }

    if resolution.state == "agent-readable":
        signal = "source_identity"
        value: Any = native_to_item[record.record_id]
    elif resolution.state == "superseded":
        signal = "superseded_by"
        value = [native_to_item[item] for item in resolution.superseded_by]
    else:
        signal = "contra_fork"
        value = sorted(
            native_to_item[member.record_id]
            for member in family
            if member.serial == record.serial
            and member.record_id != record.record_id
            and member.value != record.value
        )

    supersedes = sorted(
        native_to_item[member.record_id]
        for member in family
        if member.serial < record.serial and resolution.state != "superseded"
    )
    contradicts = sorted(
        native_to_item[member.record_id]
        for member in family
        if resolution.candidate_set_id is not None
        and member.serial == record.serial
        and member.record_id != record.record_id
        and member.value != record.value
    )
    return {
        "item_id": native_to_item[record.record_id],
        "source_id": record.source_id,
        "content": record.content,
        "content_kind": "source-native",
        "content_sha256": sha256(record.content.encode("utf-8")).hexdigest(),
        "event_time": record.event_time,
        "ingest_time": record.ingest_time,
        "evidence": [
            {
                "adapter_id": ADAPTER_ID,
                "signal": signal,
                "value": value,
                "evidence_ptr": f"ledger://{record.record_id}",
                "tier": "harness-heuristic",
            }
        ],
        "relations": {
            "supersedes": supersedes,
            "contradicts": contradicts,
            "candidate_set_id": resolution.candidate_set_id,
        },
        "state": resolution.state,
        "policy_version": POLICY_VERSION,
    }


def context_items_for_candidates(
    ledger: MemoryLedger, candidate_ids: Iterable[str]
) -> tuple[dict[str, Any], ...]:
    ids = tuple(candidate_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("candidate IDs must be unique")
    return tuple(context_item_for_record(ledger, record_id) for record_id in ids)


def render_raw(
    items: Iterable[Mapping[str, Any]],
    *,
    budget: int,
    count_tokens: Callable[[str], int],
) -> RawContext:
    if budget < 0:
        raise ValueError("budget must be non-negative")
    lines: list[str] = []
    rendered_ids: list[str] = []
    for item in items:
        proposed = "\n".join([*lines, str(item["content"])])
        if count_tokens(proposed) > budget:
            break
        lines.append(str(item["content"]))
        rendered_ids.append(str(item["item_id"]))
    block = "\n".join(lines)
    return RawContext(
        block=block,
        rendered_item_ids=tuple(rendered_ids),
        token_count=count_tokens(block),
    )
