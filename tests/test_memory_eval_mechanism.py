from dataclasses import replace

import pytest

from harness.memory_eval.ledger import MemoryLedger
from harness.memory_eval.mechanism import mechanism_passes, score_mechanism
from harness.memory_eval.models import AnswerObservation, MemoryQuestion, MemoryRecord
from harness.memory_eval.retrievers import IndexBuildAttestation
from harness.memory_eval.runner import ExperimentRow


def record(record_id: str, scope: str, value: str, serial: int) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id,
        scope=scope,
        value=value,
        content=f"The value is {value}.",
        serial=serial,
    )


def row(query_id: str, arm: str, candidate_ids: tuple[str, ...]) -> ExperimentRow:
    return ExperimentRow(
        query_id=query_id,
        arm=arm,
        budget=1500,
        candidate_ids=candidate_ids,
        rendered_item_ids=candidate_ids,
        block_tokens=0,
        prompt_tokens=0,
        retrieval_ms=0.0,
        compilation_ms=0.0,
        consumer_ms=0.0,
        observation=AnswerObservation(status="abstain", answer=None),
        hedged=None,
        extra_prose=None,
        parse_reason=None,
        consumer_pin_id="test",
        block_sha256="0" * 64,
        prompt_sha256="1" * 64,
        raw_output="",
        audit_rows=(),
        audit_anomalies=(),
    )


def attestation(
    mode: str, *, prototype_count: int, merged: int = 0, key_drifted_merges: int = 0
) -> IndexBuildAttestation:
    written = 4
    return IndexBuildAttestation(
        mode=mode,
        written=written,
        merged=merged,
        allocated=written - merged,
        dropped=0,
        evicted=0,
        prototype_count=prototype_count,
        key_drifted_merges=key_drifted_merges,
        index_sha256="0" * 64,
    )


def test_authoritative_recall_uses_latest_matching_record_and_excludes_contested():
    ledger = MemoryLedger(
        [
            record("old", "evolving", "Old", 1),
            record("current", "evolving", " Current ", 2),
            record("fork-a", "contested", "A", 3),
            record("fork-b", "contested", "B", 3),
        ]
    )
    questions = [
        MemoryQuestion("q-current", "Current?", "evolving", "current"),
        MemoryQuestion("q-fork", "Fork?", "contested", "A"),
    ]
    rows = [
        row("q-current", "exemplar_raw", ("current",)),
        row("q-current", "fam_raw", ("old",)),
        row("q-fork", "exemplar_raw", ("fork-a",)),
        row("q-fork", "fam_raw", ("fork-b",)),
    ]

    report = score_mechanism(
        rows=rows,
        questions=questions,
        ledger=ledger,
        exemplar_attestation=attestation("allocate-only", prototype_count=4),
        fam_attestation=attestation(
            "condense", prototype_count=3, merged=1, key_drifted_merges=1
        ),
    )

    assert report.recall_n == 1
    assert report.exemplar_recall_count == 1
    assert report.fam_recall_count == 0
    assert report.recall_loss_count == 1


def test_mechanism_report_pairs_exemplar_and_fam_on_one_denominator():
    ledger = MemoryLedger(
        [
            record("q1-old", "s1", "old", 1),
            record("q1", "s1", "one", 2),
            record("q2", "s2", "two", 1),
            record("spare", "s3", "three", 1),
        ]
    )
    questions = [
        MemoryQuestion("q1", "One?", "s1", "one"),
        MemoryQuestion("q2", "Two?", "s2", "two"),
    ]
    rows = [
        row("q1", "exemplar_raw", ("q1",)),
        row("q1", "fam_raw", ("q1",)),
        row("q2", "exemplar_raw", ("q2",)),
        row("q2", "fam_raw", ("spare",)),
    ]

    report = score_mechanism(
        rows=rows,
        questions=questions,
        ledger=ledger,
        exemplar_attestation=attestation("allocate-only", prototype_count=4),
        fam_attestation=attestation(
            "condense", prototype_count=3, merged=1, key_drifted_merges=1
        ),
    )

    assert report.recall_n == 2
    assert report.exemplar_recall_count == 2
    assert report.fam_recall_count == 1
    assert report.recall_loss_count == 1
    assert report.record_n == 4
    assert report.exemplar_prototype_count == 4
    assert report.fam_prototype_count == 3
    assert report.prototype_reduction_count == 1


def test_mechanism_gate_uses_exact_integer_thresholds():
    assert mechanism_passes(
        reduction_count=2,
        record_n=10,
        reduction_margin=0.2,
        recall_loss_count=1,
        recall_n=10,
        recall_loss_bound=0.1,
    )
    assert not mechanism_passes(
        reduction_count=1,
        record_n=10,
        reduction_margin=0.2,
        recall_loss_count=1,
        recall_n=10,
        recall_loss_bound=0.1,
    )


def test_mechanism_gate_uses_registered_decimal_text_without_float_drift():
    assert mechanism_passes(
        reduction_count=7,
        record_n=50,
        reduction_margin=0.14,
        recall_loss_count=0,
        recall_n=50,
        recall_loss_bound=0.0,
    )
    assert mechanism_passes(
        reduction_count=0,
        record_n=50,
        reduction_margin=0.0,
        recall_loss_count=29,
        recall_n=50,
        recall_loss_bound=0.58,
    )
    assert not mechanism_passes(
        reduction_count=2,
        record_n=10,
        reduction_margin=0.2,
        recall_loss_count=2,
        recall_n=10,
        recall_loss_bound=0.1,
    )


def test_mechanism_gate_rejects_empty_denominators():
    with pytest.raises(ValueError, match="empty"):
        mechanism_passes(
            reduction_count=0,
            record_n=1,
            reduction_margin=0.0,
            recall_loss_count=0,
            recall_n=0,
            recall_loss_bound=0.0,
        )


@pytest.mark.parametrize("defect", ["missing", "duplicate"])
def test_mechanism_scorer_requires_one_raw_row_per_arm_and_question(defect):
    ledger = MemoryLedger([record("answer", "scope", "A", 1)])
    questions = [MemoryQuestion("q", "A?", "scope", "A")]
    exemplar = row("q", "exemplar_raw", ("answer",))
    fam = row("q", "fam_raw", ("answer",))
    rows = [exemplar] if defect == "missing" else [exemplar, fam, replace(fam)]

    with pytest.raises(ValueError, match="exactly one"):
        score_mechanism(
            rows=rows,
            questions=questions,
            ledger=ledger,
            exemplar_attestation=attestation("allocate-only", prototype_count=1),
            fam_attestation=attestation("condense", prototype_count=1),
        )


def test_mechanism_scorer_rejects_non_authoritative_candidate_ids():
    ledger = MemoryLedger([record("answer", "scope", "A", 1)])
    questions = [MemoryQuestion("q", "A?", "scope", "A")]

    with pytest.raises((KeyError, ValueError), match="unknown"):
        score_mechanism(
            rows=[
                row("q", "exemplar_raw", ("answer",)),
                row("q", "fam_raw", ("not-in-ledger",)),
            ],
            questions=questions,
            ledger=ledger,
            exemplar_attestation=attestation("allocate-only", prototype_count=1),
            fam_attestation=attestation("condense", prototype_count=1),
        )


def test_noncontested_question_requires_an_authoritative_answer_id():
    ledger = MemoryLedger([record("answer", "scope", "A", 1)])
    questions = [MemoryQuestion("q", "B?", "scope", "B")]

    with pytest.raises(ValueError, match="authoritative"):
        score_mechanism(
            rows=[
                row("q", "exemplar_raw", ("answer",)),
                row("q", "fam_raw", ("answer",)),
            ],
            questions=questions,
            ledger=ledger,
            exemplar_attestation=attestation("allocate-only", prototype_count=1),
            fam_attestation=attestation("condense", prototype_count=1),
        )
