import pytest

from harness.memory_eval import ARM_NAMES, CONTEXT_BUDGET_TOKENS
from harness.memory_eval import manifest as manifest_module
from harness.memory_eval.ledger import MemoryLedger
from harness.memory_eval.models import AnswerObservation, MemoryQuestion, MemoryRecord
from harness.memory_eval.runner import ExperimentRow
from harness.memory_eval.scoring import (
    SCORING_VERSION,
    Rate,
    classify_scope,
    is_clean_scope,
    normalize_answer,
    score_rows,
)


def record(record_id, scope, value, serial):
    return MemoryRecord(
        record_id=record_id,
        scope=scope,
        value=value,
        content=f"The value is {value}.",
        serial=serial,
        event_time="2026-01-01T00:00:00Z",
    )


def row(
    query_id,
    arm,
    *,
    status="answer",
    answer=None,
    prompt_tokens=20,
    candidate_ids=(),
):
    if status == "answer" and answer is None:
        answer = "unused"
    observation = AnswerObservation(status=status, answer=answer)
    return ExperimentRow(
        query_id=query_id,
        arm=arm,
        budget=CONTEXT_BUDGET_TOKENS,
        candidate_ids=tuple(candidate_ids),
        rendered_item_ids=(),
        block_tokens=0,
        prompt_tokens=prompt_tokens,
        retrieval_ms=1.0,
        compilation_ms=2.0,
        consumer_ms=3.0,
        observation=observation,
        hedged=False if status == "answer" else None,
        extra_prose=False if status == "answer" else None,
        parse_reason="bad" if status == "malformed" else None,
        consumer_pin_id="test",
        block_sha256="0" * 64,
        prompt_sha256="1" * 64,
        raw_output="",
        audit_rows=(),
        audit_anomalies=(),
    )


def rows_for(questions, answers_by_arm):
    """Build one row per (question, arm) from {arm: ((status, answer), ...)}."""

    built = []
    for arm in ARM_NAMES:
        for question, (status, answer) in zip(questions, answers_by_arm[arm]):
            built.append(row(question.query_id, arm, status=status, answer=answer))
    return built


def test_answer_normalization_is_exact_after_unicode_case_and_space_cleanup():
    assert normalize_answer("  CAFÉ\nInc. ") == normalize_answer("cafe\u0301 inc.")
    assert normalize_answer("Inc.") != normalize_answer("Inc")


def test_clean_scope_requires_one_distinct_value_and_no_fork():
    ledger = MemoryLedger(
        [
            record("same-1", "clean", "A", 1),
            record("same-2", "clean", "A", 2),
            record("old", "evolving", "A", 1),
            record("new", "evolving", "B", 2),
            record("fork-a", "fork", "A", 3),
            record("fork-b", "fork", "B", 3),
        ]
    )

    assert is_clean_scope(ledger, "clean") is True
    assert is_clean_scope(ledger, "evolving") is False
    assert is_clean_scope(ledger, "fork") is False


def test_classify_scope_partitions_clean_stale_and_contested():
    ledger = MemoryLedger(
        [
            record("same-1", "clean", "A", 1),
            record("same-2", "clean", "A", 2),
            record("old", "evolving", "A", 1),
            record("new", "evolving", "B", 2),
            record("fork-a", "fork", "A", 3),
            record("fork-b", "fork", "B", 3),
            record("hist", "fork-with-history", "Old", 1),
            record("hist-fork-a", "fork-with-history", "A", 2),
            record("hist-fork-b", "fork-with-history", "B", 2),
        ]
    )

    assert classify_scope(ledger, "clean") == "clean"
    assert classify_scope(ledger, "evolving") == "stale_eligible"
    assert classify_scope(ledger, "fork") == "contested"
    # A fork with superseded history is contested, never stale-eligible:
    # the wrong-fork answer must not dilute the stale-adoption denominator.
    assert classify_scope(ledger, "fork-with-history") == "contested"
    with pytest.raises(ValueError, match="no ledger records"):
        classify_scope(ledger, "missing")


def test_scoring_reports_stale_adoption_clean_loss_and_operational_costs():
    ledger = MemoryLedger(
        [
            record("old", "evolving", "A", 1),
            record("new", "evolving", "B", 2),
            record("clean", "clean", "C", 1),
        ]
    )
    questions = [
        MemoryQuestion("q-evolving", "Current value?", "evolving", "B"),
        MemoryQuestion("q-clean", "Clean value?", "clean", "C"),
    ]
    answers = {
        "no_memory": (("abstain", None), ("malformed", None)),
        "vector_raw": (("answer", "A"), ("answer", "C")),
        "vector_governed": (("answer", "B"), ("answer", "C")),
        "fam_raw": (("answer", "A"), ("answer", "C")),
        "fam_governed": (("answer", "B"), ("abstain", None)),
    }
    rows = []
    for arm in ARM_NAMES:
        for question, (status, answer) in zip(questions, answers[arm]):
            rows.append(
                row(
                    question.query_id,
                    arm,
                    status=status,
                    answer=answer,
                    prompt_tokens=10 if question.query_id == "q-evolving" else 30,
                )
            )

    report = score_rows(rows, questions, ledger)

    assert report.scoring_version == SCORING_VERSION
    assert report.corpus.questions == 2
    assert report.corpus.clean_questions == 1
    assert report.corpus.stale_eligible_questions == 1
    assert report.corpus.contested_questions == 0
    assert report.corpus.ledger_scopes == 2
    assert report.corpus.contested_ledger_scopes == 0

    assert report.by_arm["vector_raw"].accuracy == Rate(value=0.5, n=2)
    assert report.by_arm["vector_raw"].stale_adoption_rate == Rate(value=1.0, n=1)
    assert report.by_arm["vector_raw"].current_adoption_rate == Rate(value=0.0, n=1)
    assert report.by_arm["vector_governed"].stale_adoption_rate == Rate(value=0.0, n=1)
    assert report.by_arm["vector_governed"].current_adoption_rate == Rate(
        value=1.0, n=1
    )
    assert report.by_arm["no_memory"].abstention_rate == Rate(value=0.5, n=2)
    assert report.by_arm["no_memory"].malformed_rate == Rate(value=0.5, n=2)
    # No contested scopes in this corpus: fork adoption has no data.
    assert report.by_arm["vector_raw"].fork_adoption_rate == Rate(value=None, n=0)
    assert report.by_arm["vector_raw"].mean_prompt_tokens == pytest.approx(20.0)
    assert report.by_arm["vector_raw"].mean_total_latency_ms == pytest.approx(6.0)

    assert report.clean_answer_loss["vector_governed_vs_raw"] == Rate(value=0.0, n=1)
    assert report.clean_answer_loss["fam_governed_vs_raw"] == Rate(value=1.0, n=1)
    assert report.stale_eligible_loss["vector_governed_vs_raw"] == Rate(
        value=0.0, n=1
    )
    assert report.stale_eligible_loss["fam_governed_vs_raw"] == Rate(value=0.0, n=1)

    stale = next(
        item
        for item in report.rows
        if item.query_id == "q-evolving" and item.arm == "fam_raw"
    )
    assert stale.correct is False
    assert stale.stale_adoption is True
    assert stale.scope_class == "stale_eligible"


def test_all_abstaining_governed_arm_registers_as_loss_on_evolving_corpus():
    """Blocker-1 regression: suppression can no longer win the value claim.

    Before this repair, a governed arm abstaining on every stale-eligible
    question satisfied BOTH registered criteria (stale adoption 0.0, clean
    loss 0.0 over zero clean questions) while destroying every correct
    current answer the raw arm delivered. It must now register as a loss.
    """

    ledger = MemoryLedger(
        [
            record("e1-old", "evolving-1", "A", 1),
            record("e1-new", "evolving-1", "B", 2),
            record("e2-old", "evolving-2", "X", 1),
            record("e2-new", "evolving-2", "Y", 2),
        ]
    )
    questions = [
        MemoryQuestion("q1", "Current value one?", "evolving-1", "B"),
        MemoryQuestion("q2", "Current value two?", "evolving-2", "Y"),
    ]
    answers = {
        "no_memory": (("abstain", None), ("abstain", None)),
        "vector_raw": (("answer", "B"), ("answer", "Y")),
        "vector_governed": (("abstain", None), ("abstain", None)),
        "fam_raw": (("answer", "B"), ("answer", "Y")),
        "fam_governed": (("abstain", None), ("abstain", None)),
    }

    report = score_rows(rows_for(questions, answers), questions, ledger)

    # Raw arms answered every current value; governed arms suppressed all.
    assert report.by_arm["vector_raw"].accuracy == Rate(value=1.0, n=2)
    assert report.by_arm["vector_governed"].accuracy == Rate(value=0.0, n=2)

    # The two originally registered criteria still look "won"...
    assert report.by_arm["vector_governed"].stale_adoption_rate == Rate(
        value=0.0, n=2
    )
    assert report.clean_answer_loss["vector_governed_vs_raw"] == Rate(
        value=None, n=0
    )

    # ...but suppression is now priced on the stale-eligible stratum.
    assert report.by_arm["vector_raw"].current_adoption_rate == Rate(value=1.0, n=2)
    assert report.by_arm["vector_governed"].current_adoption_rate == Rate(
        value=0.0, n=2
    )
    assert report.by_arm["fam_governed"].current_adoption_rate == Rate(
        value=0.0, n=2
    )
    assert report.stale_eligible_loss["vector_governed_vs_raw"] == Rate(
        value=1.0, n=2
    )
    assert report.stale_eligible_loss["fam_governed_vs_raw"] == Rate(value=1.0, n=2)


def test_zero_clean_scope_corpus_reports_clean_loss_as_no_data():
    """Blocker-2 regression: an empty denominator is None, never a 0.0."""

    ledger = MemoryLedger(
        [
            record("old", "evolving", "A", 1),
            record("new", "evolving", "B", 2),
        ]
    )
    questions = [MemoryQuestion("q1", "Current value?", "evolving", "B")]
    answers = {
        "no_memory": (("abstain", None),),
        "vector_raw": (("answer", "A"),),
        "vector_governed": (("answer", "B"),),
        "fam_raw": (("answer", "A"),),
        "fam_governed": (("answer", "B"),),
    }

    report = score_rows(rows_for(questions, answers), questions, ledger)

    assert report.corpus.clean_questions == 0
    for key in ("vector_governed_vs_raw", "fam_governed_vs_raw"):
        assert report.clean_answer_loss[key] == Rate(value=None, n=0)
        assert report.clean_answer_loss[key].value is None


def test_all_clean_corpus_is_scoreable_and_reports_stale_metrics_as_no_data():
    """The symmetric blocker-2 case: a serial-ordering bug that leaves zero
    stale-eligible questions must surface as no-data, not as a perfect 0.0
    stale-adoption rate in every arm. A legitimately all-clean corpus stays
    scoreable."""

    ledger = MemoryLedger(
        [
            record("c1", "clean-1", "A", 1),
            record("c2", "clean-2", "B", 1),
        ]
    )
    questions = [
        MemoryQuestion("q1", "Value one?", "clean-1", "A"),
        MemoryQuestion("q2", "Value two?", "clean-2", "B"),
    ]
    answers = {
        "no_memory": (("abstain", None), ("abstain", None)),
        "vector_raw": (("answer", "A"), ("answer", "B")),
        "vector_governed": (("answer", "A"), ("answer", "B")),
        "fam_raw": (("answer", "A"), ("answer", "B")),
        "fam_governed": (("answer", "A"), ("answer", "B")),
    }

    report = score_rows(rows_for(questions, answers), questions, ledger)

    assert report.corpus.stale_eligible_questions == 0
    assert report.corpus.clean_questions == 2
    for arm in ARM_NAMES:
        assert report.by_arm[arm].stale_adoption_rate == Rate(value=None, n=0)
        assert report.by_arm[arm].current_adoption_rate == Rate(value=None, n=0)
    for key in ("vector_governed_vs_raw", "fam_governed_vs_raw"):
        assert report.stale_eligible_loss[key] == Rate(value=None, n=0)
    assert report.clean_answer_loss["vector_governed_vs_raw"] == Rate(value=0.0, n=2)


def test_malformed_rows_leave_stale_denominator_but_abstain_stays():
    ledger = MemoryLedger(
        [
            record("e1-old", "evolving-1", "A", 1),
            record("e1-new", "evolving-1", "B", 2),
            record("e2-old", "evolving-2", "X", 1),
            record("e2-new", "evolving-2", "Y", 2),
        ]
    )
    questions = [
        MemoryQuestion("q1", "Current value one?", "evolving-1", "B"),
        MemoryQuestion("q2", "Current value two?", "evolving-2", "Y"),
    ]
    answers = {
        "no_memory": (("abstain", None), ("abstain", None)),
        "vector_raw": (("answer", "A"), ("answer", "X")),
        "vector_governed": (("malformed", None), ("abstain", None)),
        "fam_raw": (("answer", "A"), ("answer", "X")),
        "fam_governed": (("abstain", None), ("abstain", None)),
    }

    report = score_rows(rows_for(questions, answers), questions, ledger)

    # Malformed q1 row leaves the stale-adoption denominator (interface
    # failure, not stale-avoidance); the abstained q2 row stays in it as
    # non-adoption.
    assert report.by_arm["vector_governed"].stale_adoption_rate == Rate(
        value=0.0, n=1
    )
    assert report.by_arm["fam_governed"].stale_adoption_rate == Rate(value=0.0, n=2)
    assert report.by_arm["vector_raw"].stale_adoption_rate == Rate(value=1.0, n=2)
    # The anti-suppression floor keeps malformed rows in its denominator:
    # a malformed row delivered no current answer.
    assert report.by_arm["vector_governed"].current_adoption_rate == Rate(
        value=0.0, n=2
    )
    assert report.by_arm["vector_governed"].malformed_rate == Rate(value=0.5, n=2)


def test_contested_scopes_are_counted_and_kept_out_of_stale_stratum():
    ledger = MemoryLedger(
        [
            record("hist", "forked", "Old", 1),
            record("fork-a", "forked", "A", 2),
            record("fork-b", "forked", "B", 2),
        ]
    )
    questions = [MemoryQuestion("q1", "Value?", "forked", "A")]
    answers = {
        "no_memory": (("malformed", None),),
        "vector_raw": (("answer", "B"),),
        "vector_governed": (("abstain", None),),
        "fam_raw": (("answer", "Old"),),
        "fam_governed": (("answer", "A"),),
    }

    report = score_rows(rows_for(questions, answers), questions, ledger)

    assert report.corpus.contested_questions == 1
    assert report.corpus.contested_ledger_scopes == 1
    assert report.corpus.stale_eligible_questions == 0
    assert report.corpus.clean_questions == 0

    # The fork's superseded history does not smuggle it into the stale
    # stratum, where a wrong-fork answer would count as non-stale.
    for arm in ARM_NAMES:
        assert report.by_arm[arm].stale_adoption_rate == Rate(value=None, n=0)
        assert report.by_arm[arm].current_adoption_rate == Rate(value=None, n=0)

    # Fork adoption counts adopting ANY contested value — no adjudication
    # of which fork member is right, and no credit for abstention policy.
    assert report.by_arm["vector_raw"].fork_adoption_rate == Rate(value=1.0, n=1)
    assert report.by_arm["fam_governed"].fork_adoption_rate == Rate(value=1.0, n=1)
    assert report.by_arm["vector_governed"].fork_adoption_rate == Rate(
        value=0.0, n=1
    )
    assert report.by_arm["fam_raw"].fork_adoption_rate == Rate(value=0.0, n=1)
    # Malformed rows leave the fork denominator too.
    assert report.by_arm["no_memory"].fork_adoption_rate == Rate(value=None, n=0)

    contested_row = next(
        item for item in report.rows if item.arm == "fam_raw"
    )
    assert contested_row.scope_class == "contested"
    assert contested_row.stale_adoption is True  # row-level fact is retained
    assert contested_row.fork_adoption is False


def test_scoring_version_is_stamped_and_mismatch_raises():
    ledger = MemoryLedger([record("r", "clean", "C", 1)])
    questions = [MemoryQuestion("q", "Value?", "clean", "C")]
    rows = [row("q", arm, answer="C") for arm in ARM_NAMES]

    report = score_rows(
        rows, questions, ledger, expected_scoring_version=SCORING_VERSION
    )
    assert report.scoring_version == SCORING_VERSION
    assert manifest_module.SCORING_VERSION is SCORING_VERSION

    with pytest.raises(ValueError, match="scoring version mismatch"):
        score_rows(
            rows,
            questions,
            ledger,
            expected_scoring_version="memory-eval-scoring-v1",
        )


def test_paired_arms_with_divergent_candidate_ids_are_rejected():
    ledger = MemoryLedger([record("r", "clean", "C", 1)])
    questions = [MemoryQuestion("q", "Value?", "clean", "C")]
    rows = [
        row("q", "no_memory", answer="C"),
        row("q", "vector_raw", answer="C", candidate_ids=("r",)),
        row("q", "vector_governed", answer="C", candidate_ids=("r", "other")),
        row("q", "fam_raw", answer="C", candidate_ids=("r",)),
        row("q", "fam_governed", answer="C", candidate_ids=("r",)),
    ]

    with pytest.raises(ValueError, match="candidate_ids diverge"):
        score_rows(rows, questions, ledger)


def test_scoring_rejects_incomplete_duplicate_or_empty_inputs():
    ledger = MemoryLedger([record("r", "clean", "C", 1)])
    questions = [MemoryQuestion("q", "Value?", "clean", "C")]

    with pytest.raises(ValueError, match="exactly one row per fixed arm"):
        score_rows([row("q", "no_memory", status="abstain")], questions, ledger)
    duplicate = [row("q", arm, answer="C") for arm in ARM_NAMES]
    duplicate.append(row("q", "no_memory", status="abstain"))
    with pytest.raises(ValueError, match="exactly one row per fixed arm"):
        score_rows(duplicate, questions, ledger)
    with pytest.raises(ValueError, match="empty question set"):
        score_rows([], [], ledger)
