"""Exact mechanism scoring for matched exemplar and live FAM retrieval."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from math import ceil, floor

from .ledger import MemoryLedger
from .models import MemoryQuestion
from .preregistration import experiment_verdict
from .retrievers import IndexBuildAttestation
from .runner import ExperimentRow
from .scoring import normalize_answer, validate_raw_with_invariant


_MECHANISM_ARMS = ("exemplar_raw", "fam_raw")


@dataclass(frozen=True, slots=True)
class MechanismReport:
    recall_n: int
    exemplar_recall_count: int
    fam_recall_count: int
    recall_loss_count: int
    paired_recall_loss_count: int
    record_n: int
    exemplar_prototype_count: int
    fam_prototype_count: int
    prototype_reduction_count: int


def score_mechanism(
    *,
    rows: Sequence[ExperimentRow],
    questions: Sequence[MemoryQuestion],
    ledger: MemoryLedger,
    exemplar_attestation: IndexBuildAttestation,
    fam_attestation: IndexBuildAttestation,
) -> MechanismReport:
    """Score E0/F0 authoritative-current recall and index condensation.

    Candidate IDs are interpreted only as references into ``ledger``. CAM
    values and generated answers are deliberately outside this scorer.
    """
    validate_raw_with_invariant(ledger.records)
    question_by_id: dict[str, MemoryQuestion] = {}
    for question in questions:
        if question.query_id in question_by_id:
            raise ValueError(f"duplicate query_id: {question.query_id}")
        question_by_id[question.query_id] = question

    mechanism_rows = tuple(row for row in rows if row.arm in _MECHANISM_ARMS)
    counts = Counter((row.query_id, row.arm) for row in mechanism_rows)
    expected = {
        (question.query_id, arm)
        for question in questions
        for arm in _MECHANISM_ARMS
    }
    if set(counts) != expected or any(count != 1 for count in counts.values()):
        raise ValueError(
            "expected exactly one exemplar_raw and fam_raw row per question"
        )
    rows_by_key = {(row.query_id, row.arm): row for row in mechanism_rows}

    for row in mechanism_rows:
        ids = tuple(row.candidate_ids)
        if len(ids) != len(set(ids)):
            raise ValueError(
                f"duplicate candidate ID in {row.arm} row for {row.query_id}"
            )
        for record_id in ids:
            ledger.get(record_id)

    exemplar_recall = 0
    fam_recall = 0
    paired_recall_loss = 0
    recall_n = 0
    for question in questions:
        records = ledger.records_for_scope(question.scope)
        if not records:
            raise ValueError(f"question {question.query_id} has no ledger scope")
        maximum_serial = max(record.serial for record in records)
        latest = tuple(
            record for record in records if record.serial == maximum_serial
        )
        if len({normalize_answer(record.value) for record in latest}) > 1:
            continue
        expected_answer = normalize_answer(question.answer)
        authoritative_ids = {
            record.record_id
            for record in latest
            if normalize_answer(record.value) == expected_answer
        }
        if not authoritative_ids:
            raise ValueError(
                f"question {question.query_id} has no authoritative answer ID"
            )
        recall_n += 1
        exemplar_ids = set(
            rows_by_key[(question.query_id, "exemplar_raw")].candidate_ids
        )
        fam_ids = set(rows_by_key[(question.query_id, "fam_raw")].candidate_ids)
        exemplar_hit = bool(exemplar_ids & authoritative_ids)
        fam_hit = bool(fam_ids & authoritative_ids)
        exemplar_recall += exemplar_hit
        fam_recall += fam_hit
        # Paired loss: questions where the un-condensed control recovered the
        # authoritative record but condensation did not. This is the honest M2
        # construct — net (exemplar_recall - fam_recall) lets an offsetting fam
        # gain elsewhere hide a real per-question recall loss (review FR-1).
        paired_recall_loss += exemplar_hit and not fam_hit

    if exemplar_attestation.mode != "allocate-only":
        raise ValueError("exemplar attestation must be allocate-only")
    if fam_attestation.mode != "condense":
        raise ValueError("FAM attestation must be condense")
    record_n = len(ledger.records)
    exemplar_prototypes = exemplar_attestation.prototype_count
    fam_prototypes = fam_attestation.prototype_count
    return MechanismReport(
        recall_n=recall_n,
        exemplar_recall_count=exemplar_recall,
        fam_recall_count=fam_recall,
        recall_loss_count=exemplar_recall - fam_recall,
        paired_recall_loss_count=paired_recall_loss,
        record_n=record_n,
        exemplar_prototype_count=exemplar_prototypes,
        fam_prototype_count=fam_prototypes,
        prototype_reduction_count=exemplar_prototypes - fam_prototypes,
    )


def mechanism_passes(
    *,
    reduction_count: int,
    record_n: int,
    reduction_margin: float,
    recall_loss_count: int,
    recall_n: int,
    recall_loss_bound: float,
) -> bool:
    """Apply M1/M2 using their exact integer ceil/floor thresholds."""
    if record_n <= 0 or recall_n <= 0:
        raise ValueError("mechanism is not evaluable on an empty denominator")
    exact_margin = _registered_rate(reduction_margin, "reduction_margin")
    exact_bound = _registered_rate(recall_loss_bound, "recall_loss_bound")
    return (
        reduction_count >= ceil(exact_margin * record_n)
        and recall_loss_count <= floor(exact_bound * recall_n)
    )


def _registered_rate(value: float, field: str) -> Fraction:
    """Recover the exact registered decimal represented by a JSON number."""
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"{field} must be a finite number in [0, 1]")
    try:
        decimal = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{field} must be a finite number in [0, 1]") from exc
    if not decimal.is_finite() or not Decimal(0) <= decimal <= Decimal(1):
        raise ValueError(f"{field} must lie in [0, 1]")
    return Fraction(decimal)


__all__ = [
    "MechanismReport",
    "experiment_verdict",
    "mechanism_passes",
    "score_mechanism",
]
