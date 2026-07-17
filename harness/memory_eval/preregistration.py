"""Executable pre-registration protocol for the five-arm evaluation.

Normative artifact: ``docs/FIVE_ARM_PREREGISTRATION.md``. This module makes
that memo mechanically checkable: every decision it defers to the human has a
registry entry here, a schema field, and a validation rule, and the tests
assert those three sets are in bijection with the memo's own headings.

Scope boundary — this module is protocol only. Nothing here is wired into
``seal_manifest``; the seal extension and pre-execution verification (gate
G-I2) are separate, later work. Registering a value here authorizes nothing.

Gate arithmetic is INTEGER. Rates are ratios of counts, so a gate expressed
as a float comparison inherits the denominator's granularity and reads as
false precision. Every gate below converts its registered rate into an exact
row count first (``max_allowed_losses`` / ``min_required_successes``) and
compares integers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil, floor
from statistics import median
from typing import Any, Callable, Literal

from .context import context_items_for_candidates
from .ledger import MemoryLedger
from .models import MemoryQuestion
from .scoring import classify_scope


SENTINEL_PREFIX = "<<UNREGISTERED:"


def sentinel(decision_id: str) -> str:
    """The literal a memo carries where a decision is not yet registered.

    Keyed by decision id so that every sentinel maps to exactly one D-field
    and the count is mechanical. Revision 1 used a bare ``<<UNREGISTERED>>``
    for both decisions and prose, which made the count meaningless.
    """
    return f"{SENTINEL_PREFIX}{decision_id}>>"


# --------------------------------------------------------------------------
# Decision registry — one entry per D-field in the memo
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Decision:
    id: str
    title: str
    schema_field: str
    kind: Literal["rate", "count", "choice", "rate-or-null"]
    choices: tuple[str, ...] = ()
    requires: Mapping[str, tuple[str, ...]] | None = None
    """Conditional fields this decision's value makes mandatory:
    {value: (field, ...)}. Keeps a choice from being operationally inert."""


DECISIONS: tuple[Decision, ...] = (
    Decision("D-1", "Stale-reduction margin", "stale_reduction_margin", "rate"),
    Decision("D-2", "Clean-answer-loss bound", "clean_answer_loss_bound", "rate"),
    Decision("D-3", "Current-adoption floor", "current_adoption_floor", "rate"),
    Decision(
        "D-3b",
        "Additional abstention bound (null = none registered)",
        "abstention_bound",
        "rate-or-null",
    ),
    Decision(
        "D-4",
        "Scorer semantics",
        "scorer",
        "choice",
        ("exact", "containment", "split", "exact-with-hygiene"),
        {"exact-with-hygiene": ("slot_hygiene_bound",)},
    ),
    Decision(
        "D-5",
        "Raw-arm truncation semantics",
        "raw_truncation",
        "choice",
        ("break", "skip", "matched-arm"),
    ),
    Decision(
        "D-6",
        "Contested-question disposition",
        "contested_disposition",
        "choice",
        ("exploratory", "gated"),
        {"gated": ("contested_rule", "contested_bound")},
    ),
    Decision(
        "D-7",
        "Equivalence relation (ledger vs scorer)",
        "equivalence",
        "choice",
        ("raw-with-invariant", "normalized"),
    ),
    Decision("D-8a", "Minimum stale-eligible denominator", "min_stale_eligible_n", "count"),
    Decision("D-8b", "Minimum clean denominator", "min_clean_n", "count"),
    Decision(
        "D-9",
        "H1 denominator policy",
        "h1_denominator",
        "choice",
        ("paired-complete", "fixed-full"),
    ),
    Decision(
        "D-10",
        "Primary family",
        "primary_family",
        "choice",
        ("vector", "fam", "both"),
    ),
)

DECISION_BY_ID: Mapping[str, Decision] = {d.id: d for d in DECISIONS}
DECISION_FIELDS: frozenset[str] = frozenset(d.schema_field for d in DECISIONS)

#: Fields the registration carries that are derived, not decided.
DERIVED_FIELDS: frozenset[str] = frozenset({"memo_sha256"})

#: Fields mandatory only when some decision selects them (see Decision.requires).
CONDITIONAL_FIELDS: frozenset[str] = frozenset(
    field
    for decision in DECISIONS
    for fields in (decision.requires or {}).values()
    for field in fields
)


def validate_registration(registration: Mapping[str, Any]) -> tuple[str, ...]:
    """Return a tuple of human-readable errors; empty means registrable.

    Never raises on bad input and never partially accepts: the caller decides
    what to do. Not called by ``seal_manifest`` — see the module docstring.
    """
    errors: list[str] = []
    seen = set(registration)

    for field in sorted(DERIVED_FIELDS - seen):
        errors.append(f"missing derived field: {field}")

    for decision in DECISIONS:
        field = decision.schema_field
        if field not in registration:
            errors.append(f"{decision.id}: missing field {field!r}")
            continue
        value = registration[field]
        if isinstance(value, str) and value.startswith(SENTINEL_PREFIX):
            errors.append(f"{decision.id}: {field!r} is still unregistered")
            continue
        errors.extend(_validate_value(decision, value))

    for decision in DECISIONS:
        for trigger, required in (decision.requires or {}).items():
            if registration.get(decision.schema_field) != trigger:
                continue
            for field in required:
                if field not in registration:
                    errors.append(
                        f"{decision.id}: {field!r} is required when "
                        f"{decision.schema_field}={trigger!r}"
                    )

    unknown = seen - DECISION_FIELDS - DERIVED_FIELDS - CONDITIONAL_FIELDS
    for field in sorted(unknown):
        errors.append(f"unknown registration field: {field!r}")

    return tuple(errors)


def _validate_value(decision: Decision, value: Any) -> list[str]:
    field = decision.schema_field
    if decision.kind == "choice":
        if value not in decision.choices:
            return [
                f"{decision.id}: {field!r}={value!r} not one of "
                f"{list(decision.choices)}"
            ]
        return []
    if decision.kind == "rate-or-null":
        if value is None:
            return []
        return _validate_rate(decision, value)
    if decision.kind == "rate":
        return _validate_rate(decision, value)
    if decision.kind == "count":
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            return [f"{decision.id}: {field!r} must be a positive integer"]
        return []
    return [f"{decision.id}: unknown kind {decision.kind!r}"]


def _validate_rate(decision: Decision, value: Any) -> list[str]:
    field = decision.schema_field
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return [f"{decision.id}: {field!r} must be a number"]
    if not 0.0 <= float(value) <= 1.0:
        return [f"{decision.id}: {field!r} must lie in [0, 1]"]
    return []


# --------------------------------------------------------------------------
# Gate arithmetic — integer semantics on real denominators
# --------------------------------------------------------------------------


def max_allowed_losses(bound: float, n: int) -> int:
    """Largest loss count satisfying ``count / n <= bound``."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return floor(bound * n + 1e-9)


def min_required_successes(floor_rate: float, n: int) -> int:
    """Smallest success count satisfying ``count / n >= floor_rate``."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return ceil(floor_rate * n - 1e-9)


def min_required_margin(margin: float, n: int) -> int:
    """Smallest adoption-count difference satisfying ``diff / n >= margin``."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return ceil(margin * n - 1e-9)


def h1_passes(
    *, raw_adoptions: int, governed_adoptions: int, n_paired: int, margin: float
) -> bool:
    """H1 on a COMMON denominator (D-9). Integer margin, exact.

    ``n_paired`` must be the same denominator for both arms; the per-arm
    denominators the scorer reports today are not interchangeable here
    (see the memo's D-9), which is why per-arm is not a registrable choice.
    """
    if n_paired <= 0:
        raise ValueError("H1 is not evaluable on an empty denominator")
    return (raw_adoptions - governed_adoptions) >= min_required_margin(
        margin, n_paired
    )


def h2_passes(*, clean_losses: int, clean_n: int, bound: float) -> bool:
    if clean_n <= 0:
        raise ValueError("H2 is not evaluable on an empty denominator")
    return clean_losses <= max_allowed_losses(bound, clean_n)


def h3_passes(*, current_adoptions: int, stale_n: int, floor_rate: float) -> bool:
    if stale_n <= 0:
        raise ValueError("H3 is not evaluable on an empty denominator")
    return current_adoptions >= min_required_successes(floor_rate, stale_n)


# --------------------------------------------------------------------------
# Verdict — total and mutually exclusive
# --------------------------------------------------------------------------

BLOCKED = "blocked"
NOT_EVALUABLE = "not-evaluable"
GO = "governed-memory-GO"
NO_GO_NO_EFFECT = "NO-GO — no effect"
NO_GO_SUPPRESSION = "NO-GO — suppression"
NO_GO_COLLATERAL = "NO-GO — collateral"

VERDICTS: frozenset[str] = frozenset(
    {BLOCKED, NOT_EVALUABLE, GO, NO_GO_NO_EFFECT, NO_GO_SUPPRESSION, NO_GO_COLLATERAL}
)


def verdict(
    *, integrity_ok: bool, evaluable: bool, h1: bool, h2: bool, h3: bool
) -> str:
    """Map gate outcomes to exactly one verdict. Total; no overlaps.

    Precedence is registered, not incidental. Revision 1's table overlapped
    when H2 and H3 both failed; the ordering below resolves it:

      * integrity first — a failed integrity gate means the numbers are not
        evidence, so no value verdict may be read from them;
      * ``not h1`` → no effect: H1 is the effect claim, and without it the
        harm gates have nothing to qualify;
      * ``not h3`` outranks ``not h2`` — when H1 passes but H3 fails, the
        diagnostic fact is that the reduction was bought by not answering.
        That explains the H1 win, so it is reported over collateral damage.
    """
    if not integrity_ok:
        return BLOCKED
    if not evaluable:
        return NOT_EVALUABLE
    if not h1:
        return NO_GO_NO_EFFECT
    if not h3:
        return NO_GO_SUPPRESSION
    if not h2:
        return NO_GO_COLLATERAL
    return GO


# --------------------------------------------------------------------------
# Shape probe — whitelisted outputs, structurally incapable of effect sizes
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShapeProbe:
    """Everything the pilot is permitted to disclose. Nothing else.

    The whitelist is enforced structurally rather than by policy: this type
    carries no arm, no answer, and no outcome, and ``shape_probe`` takes no
    consumer, so no effect size can be computed from it. Reading effect sizes
    before registering thresholds is threshold-fitting and downgrades the run
    to exploratory-only (memo §2).
    """

    total_questions: int
    clean_questions: int
    stale_eligible_questions: int
    contested_questions: int
    ledger_scopes: int
    contested_ledger_scopes: int
    record_tokens_min: int
    record_tokens_median: float
    record_tokens_max: int
    budget_binding_questions: int
    candidate_k: int
    budget: int


SHAPE_PROBE_FIELDS: frozenset[str] = frozenset(
    {
        "total_questions",
        "clean_questions",
        "stale_eligible_questions",
        "contested_questions",
        "ledger_scopes",
        "contested_ledger_scopes",
        "record_tokens_min",
        "record_tokens_median",
        "record_tokens_max",
        "budget_binding_questions",
        "candidate_k",
        "budget",
    }
)


def shape_probe(
    *,
    ledger: MemoryLedger,
    questions: Sequence[MemoryQuestion],
    retriever: Any,
    query_embeddings: Mapping[str, Any],
    candidate_k: int,
    budget: int,
    count_tokens: Callable[[str], int],
) -> ShapeProbe:
    """Measure corpus shape only. Takes no consumer, by construction.

    ``budget_binding_questions`` counts questions whose full retrieved
    candidate set exceeds ``budget`` — the fact D-5 turns on — without
    rendering an arm or generating an answer.
    """
    if not questions:
        raise ValueError("shape probe requires at least one question")

    classes = [classify_scope(ledger, question.scope) for question in questions]
    scopes = {record.scope for record in ledger.records}

    record_tokens = sorted(
        count_tokens(record.content) for record in ledger.records
    )
    if not record_tokens:
        raise ValueError("shape probe requires at least one record")

    binding = 0
    for question in questions:
        candidates = retriever.query(query_embeddings[question.query_id], candidate_k)
        items = context_items_for_candidates(
            ledger, tuple(item.record_id for item in candidates)
        )
        full_block = "\n".join(str(item["content"]) for item in items)
        if count_tokens(full_block) > budget:
            binding += 1

    return ShapeProbe(
        total_questions=len(questions),
        clean_questions=classes.count("clean"),
        stale_eligible_questions=classes.count("stale_eligible"),
        contested_questions=classes.count("contested"),
        ledger_scopes=len(scopes),
        contested_ledger_scopes=sum(
            1 for scope in scopes if classify_scope(ledger, scope) == "contested"
        ),
        record_tokens_min=record_tokens[0],
        record_tokens_median=median(record_tokens),
        record_tokens_max=record_tokens[-1],
        budget_binding_questions=binding,
        candidate_k=candidate_k,
        budget=budget,
    )
