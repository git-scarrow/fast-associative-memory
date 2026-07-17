"""Outcome and cost metrics for the fixed five-arm experiment.

Scoring semantics (v2):

- Every question's scope is classified into exactly one stratum:
  ``clean`` (one distinct normalized value in the ledger), ``contested``
  (distinct normalized values at the maximum serial — an unresolved fork),
  or ``stale_eligible`` (a single latest value with at least one distinct
  superseded value). The strata are disjoint: contested scopes never enter
  the stale-eligible denominator.
- Every aggregate rate is a :class:`Rate` carrying its denominator ``n``.
  ``value`` is ``None`` when ``n == 0`` — "no data" is never reported as
  a measured ``0.0``.
- ``stale_adoption_rate`` and ``fork_adoption_rate`` exclude malformed
  rows from their denominators (a malformed row is an interface failure,
  not avoidance) while retaining abstained rows as non-adoption
  (abstention IS the governed non-adoption behavior under this project's
  deferral posture).
- ``current_adoption_rate`` keeps malformed rows in its denominator: it is
  the anti-suppression floor, and only a well-formed current answer earns
  credit. Together with ``stale_eligible_loss`` it prices blanket
  abstention as a loss instead of a win.
- Contested scopes are measured (counts, fork-adoption) but never
  adjudicated: whether contested questions are scoreable at all, and what
  the correct governed response there is, are pre-registration decisions
  reserved for the human.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Literal
import unicodedata

from . import ARM_NAMES
from .ledger import MemoryLedger
from .models import MemoryQuestion
from .runner import ExperimentRow


SCORING_VERSION = "memory-eval-scoring-v2"

ScopeClass = Literal["clean", "stale_eligible", "contested"]

_PAIRED_FAMILIES = (
    ("exemplar_raw", "exemplar_governed"),
    ("fam_raw", "fam_governed"),
)


@dataclass(frozen=True, slots=True)
class Rate:
    """A measured proportion that cannot be separated from its denominator.

    ``value`` is ``None`` if and only if ``n == 0``: an empty denominator is
    reported as "no data", never as a vacuous 0.0.
    """

    value: float | None
    n: int


@dataclass(frozen=True, slots=True)
class RowScore:
    query_id: str
    arm: str
    scope_class: ScopeClass
    correct: bool
    stale_adoption: bool
    fork_adoption: bool
    abstained: bool
    malformed: bool


@dataclass(frozen=True, slots=True)
class ArmMetrics:
    n: int
    accuracy: Rate
    stale_adoption_rate: Rate
    current_adoption_rate: Rate
    fork_adoption_rate: Rate
    abstention_rate: Rate
    malformed_rate: Rate
    mean_prompt_tokens: float
    mean_total_latency_ms: float


@dataclass(frozen=True, slots=True)
class CorpusCounts:
    """Stratum sizes a gate must read before trusting any rate."""

    questions: int
    clean_questions: int
    stale_eligible_questions: int
    contested_questions: int
    ledger_scopes: int
    contested_ledger_scopes: int


@dataclass(frozen=True, slots=True)
class ScoreReport:
    scoring_version: str
    rows: tuple[RowScore, ...]
    corpus: CorpusCounts
    by_arm: Mapping[str, ArmMetrics]
    clean_answer_loss: Mapping[str, Rate]
    stale_eligible_loss: Mapping[str, Rate]


def normalize_answer(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.split()).casefold()


def is_clean_scope(ledger: MemoryLedger, scope: str) -> bool:
    records = ledger.records_for_scope(scope)
    return bool(records) and len({normalize_answer(record.value) for record in records}) == 1


def classify_scope(ledger: MemoryLedger, scope: str) -> ScopeClass:
    """Assign a scope to exactly one scoring stratum.

    Contested is judged first: a fork with lower-serial history is contested,
    not stale-eligible, so wrong-fork answers can never dilute the
    stale-adoption denominator.
    """

    records = ledger.records_for_scope(scope)
    if not records:
        raise ValueError(f"scope has no ledger records: {scope}")
    latest_serial = max(record.serial for record in records)
    latest_values = {
        normalize_answer(record.value)
        for record in records
        if record.serial == latest_serial
    }
    if len(latest_values) > 1:
        return "contested"
    if len({normalize_answer(record.value) for record in records}) == 1:
        return "clean"
    return "stale_eligible"


def score_rows(
    rows: Sequence[ExperimentRow],
    questions: Sequence[MemoryQuestion],
    ledger: MemoryLedger,
    *,
    expected_scoring_version: str | None = None,
) -> ScoreReport:
    if (
        expected_scoring_version is not None
        and expected_scoring_version != SCORING_VERSION
    ):
        raise ValueError(
            "scoring version mismatch: manifest expects "
            f"{expected_scoring_version!r} but this scorer is {SCORING_VERSION!r}"
        )
    if not questions:
        raise ValueError("cannot score an empty question set")

    question_by_id: dict[str, MemoryQuestion] = {}
    for question in questions:
        if question.query_id in question_by_id:
            raise ValueError(f"duplicate query_id: {question.query_id}")
        question_by_id[question.query_id] = question
        _validate_expected_answer(question, ledger)

    counts = Counter((row.query_id, row.arm) for row in rows)
    expected = {
        (question.query_id, arm) for question in questions for arm in ARM_NAMES
    }
    if set(counts) != expected or any(count != 1 for count in counts.values()):
        raise ValueError("expected exactly one row per fixed arm for every question")

    rows_by_key = {(row.query_id, row.arm): row for row in rows}
    _validate_paired_candidates(rows_by_key, questions)

    scope_class_by_query = {
        question.query_id: classify_scope(ledger, question.scope)
        for question in questions
    }

    row_scores: list[RowScore] = []
    for row in rows:
        question = question_by_id[row.query_id]
        scope_class = scope_class_by_query[row.query_id]
        expected_answer = normalize_answer(question.answer)
        answer = (
            normalize_answer(row.observation.answer)
            if row.observation.status == "answer" and row.observation.answer is not None
            else None
        )
        stale_values = {
            normalize_answer(record.value)
            for record in ledger.records_for_scope(question.scope)
            if ledger.resolved_state(record.record_id).state == "superseded"
            and normalize_answer(record.value) != expected_answer
        }
        contested_values = (
            _latest_values(ledger, question.scope)
            if scope_class == "contested"
            else frozenset()
        )
        row_scores.append(
            RowScore(
                query_id=row.query_id,
                arm=row.arm,
                scope_class=scope_class,
                correct=answer == expected_answer,
                stale_adoption=answer is not None and answer in stale_values,
                fork_adoption=answer is not None and answer in contested_values,
                abstained=row.observation.status == "abstain",
                malformed=row.observation.status == "malformed",
            )
        )

    scores_by_key = {(score.query_id, score.arm): score for score in row_scores}
    rows_by_arm = {arm: [row for row in rows if row.arm == arm] for arm in ARM_NAMES}
    scores_by_arm = {
        arm: [score for score in row_scores if score.arm == arm] for arm in ARM_NAMES
    }
    metrics: dict[str, ArmMetrics] = {}
    for arm in ARM_NAMES:
        arm_scores = scores_by_arm[arm]
        arm_rows = rows_by_arm[arm]
        stale_rows = [
            score for score in arm_scores if score.scope_class == "stale_eligible"
        ]
        contested_rows = [
            score for score in arm_scores if score.scope_class == "contested"
        ]
        metrics[arm] = ArmMetrics(
            n=len(arm_scores),
            accuracy=_rate(score.correct for score in arm_scores),
            # Malformed rows are interface failures, not stale-avoidance:
            # they leave the denominator. Abstained rows stay in it as
            # non-adoption — abstention is the governed behavior under test.
            stale_adoption_rate=_rate(
                score.stale_adoption for score in stale_rows if not score.malformed
            ),
            # Anti-suppression floor: every stale-eligible row counts,
            # malformed included. Only a well-formed current answer scores.
            current_adoption_rate=_rate(score.correct for score in stale_rows),
            fork_adoption_rate=_rate(
                score.fork_adoption
                for score in contested_rows
                if not score.malformed
            ),
            abstention_rate=_rate(score.abstained for score in arm_scores),
            malformed_rate=_rate(score.malformed for score in arm_scores),
            mean_prompt_tokens=fmean(row.prompt_tokens for row in arm_rows),
            mean_total_latency_ms=fmean(
                row.retrieval_ms + row.compilation_ms + row.consumer_ms
                for row in arm_rows
            ),
        )

    clean_questions = [
        question
        for question in questions
        if scope_class_by_query[question.query_id] == "clean"
    ]
    stale_questions = [
        question
        for question in questions
        if scope_class_by_query[question.query_id] == "stale_eligible"
    ]
    contested_questions = [
        question
        for question in questions
        if scope_class_by_query[question.query_id] == "contested"
    ]
    clean_loss = {
        "exemplar_governed_vs_raw": _paired_loss(
            clean_questions, scores_by_key, "exemplar_raw", "exemplar_governed"
        ),
        "fam_governed_vs_raw": _paired_loss(
            clean_questions, scores_by_key, "fam_raw", "fam_governed"
        ),
    }
    stale_loss = {
        "exemplar_governed_vs_raw": _paired_loss(
            stale_questions, scores_by_key, "exemplar_raw", "exemplar_governed"
        ),
        "fam_governed_vs_raw": _paired_loss(
            stale_questions, scores_by_key, "fam_raw", "fam_governed"
        ),
    }
    ledger_scopes = {record.scope for record in ledger.records}
    corpus = CorpusCounts(
        questions=len(questions),
        clean_questions=len(clean_questions),
        stale_eligible_questions=len(stale_questions),
        contested_questions=len(contested_questions),
        ledger_scopes=len(ledger_scopes),
        contested_ledger_scopes=sum(
            1 for scope in ledger_scopes if classify_scope(ledger, scope) == "contested"
        ),
    )
    return ScoreReport(
        scoring_version=SCORING_VERSION,
        rows=tuple(row_scores),
        corpus=corpus,
        by_arm=metrics,
        clean_answer_loss=clean_loss,
        stale_eligible_loss=stale_loss,
    )


def _latest_values(ledger: MemoryLedger, scope: str) -> frozenset[str]:
    records = ledger.records_for_scope(scope)
    latest_serial = max(record.serial for record in records)
    return frozenset(
        normalize_answer(record.value)
        for record in records
        if record.serial == latest_serial
    )


def _validate_expected_answer(question: MemoryQuestion, ledger: MemoryLedger) -> None:
    family = ledger.records_for_scope(question.scope)
    if not family:
        raise ValueError(f"question {question.query_id} has no ledger scope")
    latest_serial = max(record.serial for record in family)
    latest_values = {
        normalize_answer(record.value)
        for record in family
        if record.serial == latest_serial
    }
    if normalize_answer(question.answer) not in latest_values:
        raise ValueError(
            f"question {question.query_id} answer is not a latest ledger value"
        )


def _validate_paired_candidates(
    rows_by_key: Mapping[tuple[str, str], ExperimentRow],
    questions: Sequence[MemoryQuestion],
) -> None:
    """Refuse to score rows whose paired arms saw different candidates.

    The causal comparisons require identical retrieved candidate IDs within
    each family; rows concatenated across a persistence boundary from
    different seals could otherwise score without error.
    """

    for question in questions:
        for raw_arm, governed_arm in _PAIRED_FAMILIES:
            raw_ids = tuple(rows_by_key[(question.query_id, raw_arm)].candidate_ids)
            governed_ids = tuple(
                rows_by_key[(question.query_id, governed_arm)].candidate_ids
            )
            if raw_ids != governed_ids:
                raise ValueError(
                    f"candidate_ids diverge between {raw_arm} and {governed_arm} "
                    f"for {question.query_id}; paired-arm comparison is invalid"
                )


def _paired_loss(
    questions: Sequence[MemoryQuestion],
    scores: Mapping[tuple[str, str], RowScore],
    raw_arm: str,
    governed_arm: str,
) -> Rate:
    return _rate(
        scores[(question.query_id, raw_arm)].correct
        and not scores[(question.query_id, governed_arm)].correct
        for question in questions
    )


def _rate(values) -> Rate:
    flags = tuple(bool(value) for value in values)
    if not flags:
        return Rate(value=None, n=0)
    return Rate(value=sum(flags) / len(flags), n=len(flags))
