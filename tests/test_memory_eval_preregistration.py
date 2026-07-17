"""Protocol tests for the pre-registration memo.

These tests are the EVIDENCE for the memo's open decisions, not an
adjudication of them. Where a decision is unresolved (D-4, D-9), the tests
pin what each candidate option actually does on adversarial input so the
choice is made against measured behaviour rather than assertion. Revision 1
of the memo recommended a scorer on reasoning that turned out to be wrong;
the rule now is that no option is recommended unless a test here shows why.
"""

import inspect
import re
from dataclasses import asdict
from itertools import product
from pathlib import Path

import pytest

from harness.memory_eval.ledger import MemoryLedger
from harness.memory_eval.models import MemoryQuestion
from harness.memory_eval.preregistration import (
    BLOCKED,
    DECISION_FIELDS,
    DECISIONS,
    DERIVED_FIELDS,
    GO,
    NOT_EVALUABLE,
    NO_GO_COLLATERAL,
    NO_GO_NO_EFFECT,
    NO_GO_SUPPRESSION,
    SHAPE_PROBE_FIELDS,
    ShapeProbe,
    VERDICTS,
    h1_passes,
    h2_passes,
    h3_passes,
    max_allowed_losses,
    min_required_margin,
    min_required_successes,
    sentinel,
    shape_probe,
    validate_registration,
    verdict,
)
from harness.memory_eval import ARM_NAMES, CONTEXT_BUDGET_TOKENS
from harness.memory_eval.models import AnswerObservation, MemoryRecord
from harness.memory_eval.runner import ExperimentRow
from harness.memory_eval.scoring import Rate, normalize_answer, score_rows


MEMO = Path(__file__).resolve().parents[1] / "docs" / "FIVE_ARM_PREREGISTRATION.md"


def record(record_id, scope, value, serial):
    return MemoryRecord(
        record_id=record_id,
        scope=scope,
        value=value,
        content=f"The value is {value}.",
        serial=serial,
        event_time="2026-01-01T00:00:00Z",
    )


def rows_for(questions, answers_by_arm):
    built = []
    for arm in ARM_NAMES:
        for question, (status, answer) in zip(questions, answers_by_arm[arm]):
            built.append(
                ExperimentRow(
                    query_id=question.query_id,
                    arm=arm,
                    budget=CONTEXT_BUDGET_TOKENS,
                    candidate_ids=(),
                    rendered_item_ids=(),
                    block_tokens=0,
                    prompt_tokens=20,
                    retrieval_ms=1.0,
                    compilation_ms=2.0,
                    consumer_ms=3.0,
                    observation=AnswerObservation(status=status, answer=answer),
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
            )
    return built


def _complete_registration(**overrides):
    """A syntactically registrable block. The VALUES ARE PLACEHOLDERS chosen
    to exercise the validator; they are not proposed thresholds and carry no
    authority. The memo's sentinels remain the only registration."""
    base = {
        "memo_sha256": "0" * 64,
        "stale_reduction_margin": 0.1,
        "clean_answer_loss_bound": 0.1,
        "current_adoption_floor": 0.5,
        "abstention_bound": None,
        "scorer": "exact",
        "raw_truncation": "break",
        "contested_disposition": "exploratory",
        "equivalence": "raw-with-invariant",
        "min_stale_eligible_n": 40,
        "min_clean_n": 40,
        "h1_denominator": "paired-complete",
        "primary_family": "vector",
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# Finding 5 — one-to-one mapping across memo, schema, and validation
# --------------------------------------------------------------------------


def test_every_decision_has_a_unique_id_field_and_validation_rule():
    ids = [d.id for d in DECISIONS]
    fields = [d.schema_field for d in DECISIONS]
    assert len(ids) == len(set(ids))
    assert len(fields) == len(set(fields))
    assert DECISION_FIELDS == set(fields)
    assert not DECISION_FIELDS & DERIVED_FIELDS
    # Every declared field is actually enforced: dropping any one is an error.
    for decision in DECISIONS:
        broken = _complete_registration()
        del broken[decision.schema_field]
        errors = validate_registration(broken)
        assert any(decision.id in e for e in errors), decision.id


def test_memo_headings_and_sentinels_are_in_bijection_with_the_registry():
    text = MEMO.read_text(encoding="utf-8")
    heading_ids = set(re.findall(r"^### (D-[0-9]+[a-z]?) —", text, re.MULTILINE))
    sentinel_ids = set(re.findall(r"<<UNREGISTERED:(D-[0-9]+[a-z]?)>>", text))
    registry_ids = {d.id for d in DECISIONS}

    assert heading_ids == registry_ids, "memo D-headings drifted from the registry"
    assert sentinel_ids == registry_ids, "every decision needs exactly one sentinel"


def test_sentinel_count_equals_decision_count_and_prose_cannot_inflate_it():
    """Revision 1 had 14 literal markers but only 11 decisions: prose mentions
    of the bare sentinel were indistinguishable from real ones. Keying each
    sentinel to its decision id makes the count mechanical."""
    text = MEMO.read_text(encoding="utf-8")
    assert len(re.findall(r"<<UNREGISTERED:D-", text)) == len(DECISIONS)
    # A bare, unkeyed sentinel must not reappear.
    assert not re.search(r"<<UNREGISTERED>>", text)


def test_validator_rejects_sentinels_bad_choices_and_unknown_fields():
    assert validate_registration(_complete_registration()) == ()

    still_open = _complete_registration(stale_reduction_margin=sentinel("D-1"))
    assert any("still unregistered" in e for e in validate_registration(still_open))

    bad_choice = _complete_registration(scorer="fuzzy")
    assert any("not one of" in e for e in validate_registration(bad_choice))

    unknown = _complete_registration(favourite_arm="fam")
    assert any("unknown registration field" in e for e in validate_registration(unknown))

    out_of_range = _complete_registration(current_adoption_floor=1.5)
    assert any("must lie in [0, 1]" in e for e in validate_registration(out_of_range))

    bad_count = _complete_registration(min_clean_n=0)
    assert any("positive integer" in e for e in validate_registration(bad_count))


def test_conditional_fields_keep_a_choice_from_being_inert():
    """Finding 4: D-6 chose a policy no gate consumed. 'gated' must now drag
    its rule and bound in with it, so the choice has somewhere to land."""
    gated = _complete_registration(contested_disposition="gated")
    errors = validate_registration(gated)
    assert any("contested_rule" in e for e in errors)
    assert any("contested_bound" in e for e in errors)

    complete = _complete_registration(
        contested_disposition="gated",
        contested_rule="abstain-correct",
        contested_bound=0.1,
    )
    assert validate_registration(complete) == ()

    # 'exploratory' is a real registered choice, not a silent default.
    assert validate_registration(_complete_registration()) == ()


def test_abstention_bound_has_a_schema_field_and_null_is_an_explicit_choice():
    """Revision 1's floor-vs-abstention sub-decision had a sentinel but no
    field, so it could never be registered."""
    assert "abstention_bound" in DECISION_FIELDS
    assert validate_registration(_complete_registration(abstention_bound=None)) == ()
    assert validate_registration(_complete_registration(abstention_bound=0.2)) == ()
    assert validate_registration(_complete_registration(abstention_bound="x"))


# --------------------------------------------------------------------------
# Finding 1 — integer gate semantics on real denominators
# --------------------------------------------------------------------------


def test_gate_thresholds_convert_to_exact_row_counts():
    assert max_allowed_losses(0.05, 40) == 2
    assert max_allowed_losses(0.05, 10) == 0
    assert min_required_successes(0.5, 40) == 20
    assert min_required_successes(0.5, 41) == 21
    assert min_required_margin(0.1, 40) == 4
    # Exact at representative binary fractions; no float drift.
    assert max_allowed_losses(0.1, 10) == 1
    assert min_required_successes(0.1, 10) == 1


def test_threshold_below_one_over_n_is_a_strict_gate_not_an_unfalsifiable_one():
    """Corrects revision 1's G-I5. A bound under 1/n does not make the gate
    unfalsifiable — it makes it strict, and it stays perfectly falsifiable.
    The real defect is illusory precision: distinct sub-1/n bounds are
    indistinguishable."""
    n = 10
    # H2 at a bound below 1/n permits zero losses, and one loss still fails.
    assert max_allowed_losses(0.005, n) == 0
    assert h2_passes(clean_losses=0, clean_n=n, bound=0.005)
    assert not h2_passes(clean_losses=1, clean_n=n, bound=0.005)
    # H3 at a positive floor below 1/n requires one success, and zero fails.
    assert min_required_successes(0.005, n) == 1
    assert h3_passes(current_adoptions=1, stale_n=n, floor_rate=0.005)
    assert not h3_passes(current_adoptions=0, stale_n=n, floor_rate=0.005)
    # Illusory precision: two different registered bounds, identical gate.
    assert max_allowed_losses(0.005, n) == max_allowed_losses(0.09, n)


def test_h1_requires_a_common_denominator():
    assert h1_passes(raw_adoptions=8, governed_adoptions=4, n_paired=10, margin=0.4)
    assert not h1_passes(raw_adoptions=8, governed_adoptions=5, n_paired=10, margin=0.4)
    with pytest.raises(ValueError):
        h1_passes(raw_adoptions=0, governed_adoptions=0, n_paired=0, margin=0.1)


def test_gates_refuse_an_empty_denominator_rather_than_passing_it():
    with pytest.raises(ValueError):
        h2_passes(clean_losses=0, clean_n=0, bound=0.1)
    with pytest.raises(ValueError):
        h3_passes(current_adoptions=0, stale_n=0, floor_rate=0.1)


def test_per_arm_denominators_diverge_under_asymmetric_malformed_output():
    """Finding 1's core: the scorer removes malformed rows from each arm's
    stale denominator INDEPENDENTLY, so H1 currently differences two rates
    computed over different question sets. This is why D-9 exists and why
    'per-arm' is not a registrable option."""
    ledger = MemoryLedger(
        [
            record("a-old", "s-a", "A", 1),
            record("a-new", "s-a", "B", 2),
            record("b-old", "s-b", "X", 1),
            record("b-new", "s-b", "Y", 2),
        ]
    )
    questions = [
        MemoryQuestion("q1", "one?", "s-a", "B"),
        MemoryQuestion("q2", "two?", "s-b", "Y"),
    ]
    answers = {
        "no_memory": (("abstain", None), ("abstain", None)),
        "vector_raw": (("answer", "A"), ("answer", "X")),
        # Governed is malformed on exactly one stale-eligible question.
        "vector_governed": (("malformed", None), ("answer", "Y")),
        "fam_raw": (("answer", "A"), ("answer", "X")),
        "fam_governed": (("answer", "B"), ("answer", "Y")),
    }
    report = score_rows(rows_for(questions, answers), questions, ledger)

    raw = report.by_arm["vector_raw"].stale_adoption_rate
    governed = report.by_arm["vector_governed"].stale_adoption_rate
    assert raw == Rate(value=1.0, n=2)
    assert governed == Rate(value=0.0, n=1)
    # The two rates are over DIFFERENT denominators: their difference is not a
    # multiple of any single 1/n, so no exact integer margin exists for H1.
    assert raw.n != governed.n


def test_h1_denominator_policies_disagree_on_the_same_rows():
    """Demonstrates the D-9 trade-off. Governed is malformed on precisely the
    row where it would have adopted stale.

    paired-complete drops that row (flattering governed by hiding its worst
    case); fixed-full keeps it and books it as non-adoption (also flattering
    governed, by counting an interface failure as avoidance). Neither option
    is free — which is the point of registering the choice."""
    ledger = MemoryLedger(
        [
            record("a-old", "s-a", "A", 1),
            record("a-new", "s-a", "B", 2),
            record("b-old", "s-b", "X", 1),
            record("b-new", "s-b", "Y", 2),
        ]
    )
    questions = [
        MemoryQuestion("q1", "one?", "s-a", "B"),
        MemoryQuestion("q2", "two?", "s-b", "Y"),
    ]
    answers = {
        "no_memory": (("abstain", None), ("abstain", None)),
        "vector_raw": (("answer", "A"), ("answer", "X")),
        "vector_governed": (("malformed", None), ("answer", "X")),
        "fam_raw": (("answer", "A"), ("answer", "X")),
        "fam_governed": (("answer", "B"), ("answer", "Y")),
    }
    report = score_rows(rows_for(questions, answers), questions, ledger)
    rows = {(s.query_id, s.arm): s for s in report.rows}
    stale_qs = [
        q.query_id
        for q in questions
        if rows[(q.query_id, "vector_raw")].scope_class == "stale_eligible"
    ]

    def adoptions(arm, query_ids):
        return sum(1 for q in query_ids if rows[(q, arm)].stale_adoption)

    # paired-complete: keep only questions where NEITHER arm is malformed.
    paired = [
        q
        for q in stale_qs
        if not rows[(q, "vector_raw")].malformed
        and not rows[(q, "vector_governed")].malformed
    ]
    # fixed-full: every stale-eligible question, malformed counted as non-adoption.
    full = stale_qs

    assert len(paired) == 1 and len(full) == 2
    paired_diff = adoptions("vector_raw", paired) - adoptions("vector_governed", paired)
    full_diff = adoptions("vector_raw", full) - adoptions("vector_governed", full)
    assert paired_diff == 0  # governed's surviving row also adopted stale
    assert full_diff == 1  # the malformed row reads as governed avoidance

    # Same rows, same arms, different registered policy, opposite H1 outcome.
    assert not h1_passes(
        raw_adoptions=adoptions("vector_raw", paired),
        governed_adoptions=adoptions("vector_governed", paired),
        n_paired=len(paired),
        margin=0.5,
    )
    assert h1_passes(
        raw_adoptions=adoptions("vector_raw", full),
        governed_adoptions=adoptions("vector_governed", full),
        n_paired=len(full),
        margin=0.5,
    )


# --------------------------------------------------------------------------
# Finding 3 — scorer semantics under adversarial answers
# --------------------------------------------------------------------------


def _exact(expected, answer):
    return normalize_answer(answer) == normalize_answer(expected)


def _containment(expected, answer):
    return normalize_answer(expected) in normalize_answer(answer)


NEGATIONS = ("not NewCo", "the answer is not NewCo", "cannot confirm NewCo")
CAVEAT_ECHOES = ("NewCo (current)", "NewCo, per the latest record")


@pytest.mark.parametrize("answer", NEGATIONS)
def test_containment_scores_negations_of_the_expected_value_as_correct(answer):
    """Finding 3: containment admits FALSE WINS. Revision 1 claimed the split
    scorer 'avoids both failure directions'; keeping stale adoption exact does
    nothing about this, because the defect is in the correctness metric."""
    assert _containment("NewCo", answer)
    assert not _exact("NewCo", answer)


@pytest.mark.parametrize("answer", NEGATIONS)
def test_containment_correctness_would_defeat_the_anti_suppression_gate(answer):
    """The sharpest consequence: current_adoption_rate is computed from
    `correct`. Under containment, a governed arm that hedges its way out of
    answering still scores correct, raising the floor H3 exists to enforce."""
    stale_n = 4
    hedged_but_containment_correct = sum(1 for _ in range(stale_n) if _containment("NewCo", answer))
    assert hedged_but_containment_correct == stale_n
    assert h3_passes(
        current_adoptions=hedged_but_containment_correct,
        stale_n=stale_n,
        floor_rate=1.0,
    )
    # Under exact, the same arm scores zero and H3 fires as designed.
    assert not h3_passes(
        current_adoptions=sum(1 for _ in range(stale_n) if _exact("NewCo", answer)),
        stale_n=stale_n,
        floor_rate=0.25,
    )


@pytest.mark.parametrize("answer", CAVEAT_ECHOES)
def test_exact_scores_caveat_echoes_as_wrong(answer):
    """Exact's failure direction, for symmetry. It manufactures a FALSE LOSS:
    conservative against the governance hypothesis, never for it."""
    assert not _exact("NewCo", answer)
    assert _containment("NewCo", answer)


def test_stale_adoption_under_containment_would_book_a_mention_as_an_adoption():
    """Why 'split' does not rescue containment: exact stale adoption is the
    right call, but it leaves the correctness hole above wide open."""
    assert _containment("OldCo", "not OldCo — the current value is NewCo")
    assert not _exact("OldCo", "not OldCo — the current value is NewCo")


def test_scorer_failure_directions_are_asymmetric_and_that_is_the_argument():
    """The one test-backed claim available for D-4: exact can only manufacture
    a false LOSS, containment can manufacture a false WIN. Under this repo's
    posture a gate must be able to fail, so the conservative direction is the
    safe one. This argues the exact FAMILY, not a specific option — the
    choice between plain exact and exact-with-hygiene stays registered."""
    false_wins_under_containment = [a for a in NEGATIONS if _containment("NewCo", a)]
    false_wins_under_exact = [a for a in NEGATIONS if _exact("NewCo", a)]
    false_losses_under_exact = [a for a in CAVEAT_ECHOES if not _exact("NewCo", a)]

    assert false_wins_under_containment == list(NEGATIONS)
    assert false_wins_under_exact == []
    assert false_losses_under_exact == list(CAVEAT_ECHOES)


# --------------------------------------------------------------------------
# Finding 5 — verdict table is total and mutually exclusive
# --------------------------------------------------------------------------


def test_verdict_is_total_and_every_input_maps_to_exactly_one_verdict():
    seen = set()
    for integrity_ok, evaluable, h1, h2, h3 in product([True, False], repeat=5):
        result = verdict(
            integrity_ok=integrity_ok, evaluable=evaluable, h1=h1, h2=h2, h3=h3
        )
        assert result in VERDICTS
        seen.add(result)
    assert seen == VERDICTS  # every verdict is reachable; none is dead code


def test_h2_and_h3_both_failing_no_longer_overlaps():
    """Revision 1's table matched two rows here and named no winner."""
    assert (
        verdict(integrity_ok=True, evaluable=True, h1=True, h2=False, h3=False)
        == NO_GO_SUPPRESSION
    )
    assert (
        verdict(integrity_ok=True, evaluable=True, h1=True, h2=False, h3=True)
        == NO_GO_COLLATERAL
    )
    assert (
        verdict(integrity_ok=True, evaluable=True, h1=True, h2=True, h3=False)
        == NO_GO_SUPPRESSION
    )


def test_integrity_and_evaluability_outrank_every_value_gate():
    assert (
        verdict(integrity_ok=False, evaluable=True, h1=True, h2=True, h3=True)
        == BLOCKED
    )
    assert (
        verdict(integrity_ok=True, evaluable=False, h1=True, h2=True, h3=True)
        == NOT_EVALUABLE
    )
    assert verdict(integrity_ok=True, evaluable=True, h1=True, h2=True, h3=True) == GO
    assert (
        verdict(integrity_ok=True, evaluable=True, h1=False, h2=True, h3=True)
        == NO_GO_NO_EFFECT
    )


# --------------------------------------------------------------------------
# Finding 2 — the shape probe's whitelist is structural
# --------------------------------------------------------------------------


def test_shape_probe_cannot_be_given_a_consumer():
    """The whitelist is enforced by construction, not by policy: with no
    consumer there is no generation, hence no arm output and no effect size."""
    params = set(inspect.signature(shape_probe).parameters)
    assert "consumer" not in params
    assert not {p for p in params if "arm" in p or "answer" in p}


def test_shape_probe_output_is_exactly_the_whitelist(monkeypatch):
    ledger = MemoryLedger(
        [
            record("a-old", "s-a", "A", 1),
            record("a-new", "s-a", "B", 2),
            record("c-1", "s-c", "Solo", 1),
        ]
    )
    questions = [
        MemoryQuestion("q1", "one?", "s-a", "B"),
        MemoryQuestion("q2", "two?", "s-c", "Solo"),
    ]

    class StubRetriever:
        def query(self, embedding, k):
            class C:
                def __init__(self, rid):
                    self.record_id = rid

            return tuple(C(r.record_id) for r in ledger.records[:k])

    probe = shape_probe(
        ledger=ledger,
        questions=questions,
        retriever=StubRetriever(),
        query_embeddings={"q1": [1.0], "q2": [1.0]},
        candidate_k=2,
        budget=1500,
        count_tokens=lambda text: len(text.split()),
    )

    assert set(asdict(probe)) == SHAPE_PROBE_FIELDS
    # No field may carry an outcome, a rate, or an arm identity.
    forbidden = ("rate", "accuracy", "adoption", "loss", "arm", "answer", "correct")
    assert not [f for f in SHAPE_PROBE_FIELDS for k in forbidden if k in f]

    assert probe.total_questions == 2
    assert probe.stale_eligible_questions == 1
    assert probe.clean_questions == 1
    assert probe.contested_questions == 0
    assert probe.budget_binding_questions == 0


def test_shape_probe_reports_the_budget_binding_count_d5_turns_on():
    ledger = MemoryLedger([record("a-old", "s-a", "A", 1), record("a-new", "s-a", "B", 2)])
    questions = [MemoryQuestion("q1", "one?", "s-a", "B")]

    class StubRetriever:
        def query(self, embedding, k):
            class C:
                def __init__(self, rid):
                    self.record_id = rid

            return tuple(C(r.record_id) for r in ledger.records[:k])

    probe = shape_probe(
        ledger=ledger,
        questions=questions,
        retriever=StubRetriever(),
        query_embeddings={"q1": [1.0]},
        candidate_k=2,
        budget=3,  # deliberately tiny: the candidate set must exceed it
        count_tokens=lambda text: len(text.split()),
    )
    assert probe.budget_binding_questions == 1
    assert probe.record_tokens_max >= probe.record_tokens_min
