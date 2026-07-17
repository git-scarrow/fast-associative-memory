from hashlib import sha256

from harness.memory_eval.context import (
    context_item_for_record,
    context_items_for_candidates,
    render_raw,
)
from harness.memory_eval.ledger import MemoryLedger
from harness.memory_eval.models import MemoryRecord


def record(record_id: str, *, serial: int, value: str) -> MemoryRecord:
    content = f"Ada's employer is {value}."
    return MemoryRecord(
        record_id=record_id,
        scope="Ada\x1femployer",
        value=value,
        content=content,
        serial=serial,
        event_time=f"2026-01-{serial + 1:02d}T00:00:00Z",
        ingest_time=f"2026-02-{serial + 1:02d}T00:00:00Z",
    )


def test_active_record_maps_to_schema_shaped_context_item():
    active = record("new", serial=2, value="B")
    old = record("old", serial=1, value="A")
    ledger = MemoryLedger([old, active])

    item = context_item_for_record(ledger, active.record_id)

    assert set(item) == {
        "item_id",
        "source_id",
        "content",
        "content_kind",
        "content_sha256",
        "event_time",
        "ingest_time",
        "evidence",
        "relations",
        "state",
        "policy_version",
    }
    assert item["item_id"] == "memory-eval:new"
    assert item["state"] == "agent-readable"
    assert item["evidence"][0]["signal"] == "source_identity"
    assert item["relations"]["supersedes"] == ["memory-eval:old"]
    assert item["content_sha256"] == sha256(active.content.encode()).hexdigest()


def test_superseded_record_carries_authoritative_successor_evidence():
    old = record("old", serial=1, value="A")
    active = record("new", serial=2, value="B")
    ledger = MemoryLedger([active, old])

    item = context_item_for_record(ledger, old.record_id)

    assert item["state"] == "superseded"
    assert item["evidence"][0]["signal"] == "superseded_by"
    assert item["evidence"][0]["value"] == ["memory-eval:new"]
    assert item["relations"]["supersedes"] == []


def test_equal_serial_fork_maps_to_shared_contradiction_set():
    left = record("left", serial=3, value="A")
    right = record("right", serial=3, value="B")
    ledger = MemoryLedger([right, left])

    items = context_items_for_candidates(ledger, ["left", "right"])

    assert {item["state"] for item in items} == {"human-review"}
    assert {item["evidence"][0]["signal"] for item in items} == {"contra_fork"}
    assert len({item["relations"]["candidate_set_id"] for item in items}) == 1
    assert items[0]["relations"]["contradicts"] == ["memory-eval:right"]
    assert items[1]["relations"]["contradicts"] == ["memory-eval:left"]


def test_raw_rendering_preserves_candidate_order_and_never_crosses_budget():
    records = [
        record("one", serial=1, value="one two"),
        MemoryRecord(
            record_id="two",
            scope="Grace\x1femployer",
            value="three four",
            content="three four five",
            serial=1,
            event_time="2026-01-01T00:00:00Z",
        ),
    ]
    ledger = MemoryLedger(records)
    items = context_items_for_candidates(ledger, ["two", "one"])

    rendered = render_raw(items, budget=5, count_tokens=lambda text: len(text.split()))

    assert rendered.block == "three four five"
    assert rendered.rendered_item_ids == ("memory-eval:two",)
    assert rendered.token_count == 3


def test_render_raw_skips_oversized_candidate_and_continues():
    items = [
        {"item_id": "a", "content": "short"},
        {"item_id": "b", "content": "far too long for this tiny budget now"},
        {"item_id": "c", "content": "ok"},
    ]

    rendered = render_raw(
        items, budget=7, count_tokens=lambda text: len(text.split())
    )

    assert rendered.block == "short\nok"
    assert rendered.rendered_item_ids == ("a", "c")
    assert rendered.token_count == 2
