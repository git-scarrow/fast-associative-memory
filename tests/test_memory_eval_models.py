from dataclasses import FrozenInstanceError

import pytest

from harness.memory_eval import ARM_NAMES, CONTEXT_BUDGET_TOKENS
from harness.memory_eval.models import (
    AnswerObservation,
    MemoryQuestion,
    MemoryRecord,
    RetrievedCandidate,
    fact_scope,
)


def test_protocol_is_fixed_to_five_arms_and_one_budget():
    assert ARM_NAMES == (
        "no_memory",
        "vector_raw",
        "vector_governed",
        "fam_raw",
        "fam_governed",
    )
    assert CONTEXT_BUDGET_TOKENS == 1500


def test_fact_scope_is_unambiguous_and_stable():
    assert fact_scope("Ada", "employer") == "Ada\x1femployer"
    assert fact_scope(" Ada ", " employer ") == "Ada\x1femployer"
    assert fact_scope("Ada\x1fLovelace", "employer") != fact_scope(
        "Ada", "Lovelace\x1femployer"
    )


def test_records_questions_and_observations_are_immutable():
    record = MemoryRecord(
        record_id="r1",
        scope=fact_scope("Ada", "employer"),
        value="Analytical Engines Ltd",
        content="Ada's employer is Analytical Engines Ltd.",
        serial=2,
        event_time="2026-01-02T00:00:00Z",
    )
    question = MemoryQuestion(
        query_id="q1",
        text="Who employs Ada?",
        scope=record.scope,
        answer="Analytical Engines Ltd",
    )
    observation = AnswerObservation(status="answer", answer=record.value)

    with pytest.raises(FrozenInstanceError):
        record.value = "Other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        question.answer = "Other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        observation.status = "abstain"  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"record_id": ""},
        {"scope": ""},
        {"value": ""},
        {"content": ""},
        {"serial": -1},
    ],
)
def test_record_validation(kwargs):
    base = dict(
        record_id="r1",
        scope="Ada\x1femployer",
        value="Analytical Engines Ltd",
        content="Ada's employer is Analytical Engines Ltd.",
        serial=0,
        event_time="2026-01-02T00:00:00Z",
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        MemoryRecord(**base)


def test_candidate_validation_and_order_fields():
    candidate = RetrievedCandidate(record_id="r1", score=0.75, rank=1)
    assert candidate == RetrievedCandidate(record_id="r1", score=0.75, rank=1)

    with pytest.raises(ValueError):
        RetrievedCandidate(record_id="r1", score=float("nan"), rank=1)
    with pytest.raises(ValueError):
        RetrievedCandidate(record_id="r1", score=0.1, rank=0)


@pytest.mark.parametrize("status", ["answer", "abstain", "malformed"])
def test_answer_observation_accepts_only_contract_statuses(status):
    observation = AnswerObservation(status=status, answer="x" if status == "answer" else None)
    assert observation.status == status


def test_answer_observation_rejects_inconsistent_answer():
    with pytest.raises(ValueError):
        AnswerObservation(status="answer", answer=None)
    with pytest.raises(ValueError):
        AnswerObservation(status="abstain", answer="unexpected")
