import pytest

from harness.memory_eval.ledger import MemoryLedger
from harness.memory_eval.models import MemoryRecord


def record(record_id: str, *, serial: int, value: str, scope: str = "Ada\x1femployer"):
    return MemoryRecord(
        record_id=record_id,
        scope=scope,
        value=value,
        content=f"Ada's employer is {value}.",
        serial=serial,
        event_time=f"2026-01-{serial + 1:02d}T00:00:00Z",
    )


def test_ledger_is_append_only_and_rejects_duplicate_ids():
    first = record("r1", serial=0, value="A")
    ledger = MemoryLedger([first])

    with pytest.raises(ValueError, match="duplicate record_id"):
        ledger.append(record("r1", serial=1, value="B"))

    assert ledger.records == (first,)
    assert ledger.get("r1") is first


def test_unique_latest_record_is_active_and_history_is_superseded():
    old = record("r-old", serial=1, value="A")
    latest = record("r-new", serial=2, value="B")
    ledger = MemoryLedger([latest, old])

    assert [item.record_id for item in ledger.records_for_scope(old.scope)] == [
        "r-old",
        "r-new",
    ]
    assert ledger.resolved_state("r-new").state == "agent-readable"
    old_state = ledger.resolved_state("r-old")
    assert old_state.state == "superseded"
    assert old_state.superseded_by == ("r-new",)
    assert old_state.candidate_set_id is None


def test_equal_serial_distinct_values_form_deterministic_review_fork():
    left = record("r-b", serial=3, value="B")
    right = record("r-a", serial=3, value="A")
    older = record("r-old", serial=2, value="Old")

    first = MemoryLedger([left, older, right])
    second = MemoryLedger([right, left, older])

    left_state = first.resolved_state(left.record_id)
    right_state = first.resolved_state(right.record_id)
    assert left_state.state == right_state.state == "human-review"
    assert left_state.candidate_set_id == right_state.candidate_set_id
    assert left_state.candidate_set_id.startswith("fork-")
    assert left_state.candidate_set_id == second.resolved_state(left.record_id).candidate_set_id
    assert first.resolved_state("r-old").superseded_by == ("r-a", "r-b")


def test_equal_serial_equal_values_are_not_a_contradiction():
    left = record("r-a", serial=3, value="Same")
    right = record("r-b", serial=3, value="Same")
    ledger = MemoryLedger([right, left])

    assert ledger.resolved_state("r-a").state == "agent-readable"
    assert ledger.resolved_state("r-b").state == "agent-readable"
    assert ledger.resolved_state("r-a").candidate_set_id is None


def test_unknown_record_lookup_fails_clearly():
    ledger = MemoryLedger()
    with pytest.raises(KeyError, match="unknown record_id"):
        ledger.get("missing")
    with pytest.raises(KeyError, match="unknown record_id"):
        ledger.resolved_state("missing")
