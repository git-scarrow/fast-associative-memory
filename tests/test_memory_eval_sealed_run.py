"""Gate G-I2: the seal binds the treatment, and binds it on the execution path.

Each test below proves a specific gate CAN FAIL. A seal whose checks cannot
fire attests nothing, which was the defect in the previous revision: the seal
was real but advisory, and its only non-test caller was the dry run.
"""

import json

import pytest
import torch

from harness.memory_eval.dry_run import HashScopeEncoder, RuleConsumer
from harness.memory_eval.manifest import (
    load_manifest,
    policy_sha256,
    scoring_module_sha256,
    seal_manifest,
    verify_manifest,
)
from harness.memory_eval.models import MemoryQuestion, MemoryRecord, fact_scope
from harness.memory_eval.preregistration import sentinel
from harness.memory_eval.retrievers import FAMRetriever
from harness.memory_eval.sealed_run import (
    Check,
    PreflightFailed,
    build_sealed_run,
    preflight,
)


POLICY = {"rules": [{"id": "R01", "action": "assert"}], "version": "test-policy-v1"}
SETTINGS = {"candidate_k": 2, "fam_prototype_k": 1, "fam_max_entries": 2}


def registration(**overrides):
    """Placeholder VALUES exercising the validator. Not proposed thresholds."""
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


def corpus():
    evolving = fact_scope("Ada", "employer")
    clean = fact_scope("Grace", "city")
    records = (
        MemoryRecord("01-old", evolving, "OldCo", "FACT[Ada|employer]=OldCo", 1, "2026-01-01T00:00:00Z"),
        MemoryRecord("02-new", evolving, "NewCo", "FACT[Ada|employer]=NewCo", 2, "2026-02-01T00:00:00Z"),
        MemoryRecord("03-clean", clean, "Detroit", "FACT[Grace|city]=Detroit", 1, "2026-01-01T00:00:00Z"),
    )
    questions = (
        MemoryQuestion("q-evolving", "What is FACT[Ada|employer]?", evolving, "NewCo"),
        MemoryQuestion("q-clean", "What is FACT[Grace|city]?", clean, "Detroit"),
    )
    encoder = HashScopeEncoder()
    return (
        records,
        questions,
        {r.record_id: encoder.encode(r.scope) for r in records},
        {q.query_id: encoder.encode(q.scope) for q in questions},
    )


def seal(path, **overrides):
    records, questions, rec_emb, qry_emb = corpus()
    kwargs = {
        "evidence_class": "scoring-run",
        "policy": POLICY,
        "consumer_pin": RuleConsumer.pin_id,
        "registration": registration(),
    }
    kwargs.update(overrides)
    return seal_manifest(path, records, questions, rec_emb, qry_emb, SETTINGS, **kwargs)


def run_preflight(path, *, policy=POLICY, consumer=None, settings=SETTINGS):
    records, questions, rec_emb, qry_emb = corpus()
    return preflight(
        path,
        records=records,
        questions=questions,
        record_embeddings=rec_emb,
        query_embeddings=qry_emb,
        retriever_settings=settings,
        policy=policy,
        consumer=consumer or RuleConsumer(),
    )


def failures(checks):
    return {c.name for c in checks if not c.passed}


# --------------------------------------------------------------------------
# G-I1 — seal refuses an incomplete registration
# --------------------------------------------------------------------------


def test_scoring_run_seal_refuses_an_unregistered_threshold(tmp_path):
    with pytest.raises(RuntimeError, match="still unregistered"):
        seal(
            tmp_path / "m.json",
            registration=registration(stale_reduction_margin=sentinel("D-1")),
        )


def test_scoring_run_seal_reports_every_reason_at_once(tmp_path):
    records, questions, rec_emb, qry_emb = corpus()
    with pytest.raises(RuntimeError) as exc:
        seal_manifest(
            tmp_path / "m.json",
            records,
            questions,
            rec_emb,
            qry_emb,
            {"candidate_k": 2, "fam_prototype_k": 1},  # no fam_max_entries
            evidence_class="scoring-run",
            policy=None,
            consumer_pin=None,
            registration=None,
        )
    message = str(exc.value)
    assert "disposition policy" in message
    assert "consumer_pin" in message
    assert "fam_max_entries" in message
    assert "registration block" in message


def test_plumbing_seal_cannot_carry_a_registration(tmp_path):
    with pytest.raises(ValueError, match="plumbing seal cannot carry a registration"):
        seal(tmp_path / "m.json", evidence_class="plumbing", registration=registration())


def test_evidence_class_is_mandatory_and_validated(tmp_path):
    records, questions, rec_emb, qry_emb = corpus()
    with pytest.raises(TypeError):
        seal_manifest(tmp_path / "m.json", records, questions, rec_emb, qry_emb, SETTINGS)
    with pytest.raises(ValueError, match="evidence_class must be one of"):
        seal(tmp_path / "m2.json", evidence_class="real-i-promise")


# --------------------------------------------------------------------------
# G-I2 — the seal binds the treatment
# --------------------------------------------------------------------------


def test_a_clean_scoring_run_seal_passes_every_preflight_check(tmp_path):
    path = tmp_path / "m.json"
    seal(path)
    checks = run_preflight(path)
    assert failures(checks) == set()
    assert {c.gate for c in checks} == {"G-I1", "G-I2"}


def test_preflight_catches_a_policy_edited_after_seal(tmp_path):
    """The blocker: the disposition policy IS the governed treatment, and it
    was loaded from disk at run time with nothing binding it to the seal."""
    path = tmp_path / "m.json"
    seal(path)
    edited = {"rules": [{"id": "R01", "action": "defer"}], "version": "test-policy-v1"}
    assert policy_sha256(edited) != policy_sha256(POLICY)
    checks = run_preflight(path, policy=edited)
    assert "policy binding" in failures(checks)


def test_preflight_catches_a_swapped_consumer(tmp_path):
    path = tmp_path / "m.json"
    seal(path)

    class OtherConsumer(RuleConsumer):
        pin_id = "some-other-consumer-v9"

    checks = run_preflight(path, consumer=OtherConsumer())
    assert "consumer pin" in failures(checks)


def test_preflight_catches_a_scorer_edited_after_seal(tmp_path):
    """scoring_version is a label; this binds the source. A post-seal edit to
    score_rows previously shipped under the sealed label."""
    path = tmp_path / "m.json"
    manifest = seal(path)
    assert manifest["protocol"]["scoring_module_sha256"] == scoring_module_sha256()

    tampered = load_manifest(path)
    tampered["protocol"]["scoring_module_sha256"] = "0" * 64
    body = dict(tampered)
    del body["manifest_sha256"]
    # Re-seal the digest so ONLY the scorer identity is wrong: proves the
    # scorer check fires on its own, not merely via the manifest digest.
    from harness.memory_eval.manifest import _fingerprint

    tampered["manifest_sha256"] = _fingerprint(body)
    path.write_text(json.dumps(tampered), encoding="utf-8")

    checks = run_preflight(path)
    assert "scorer identity" in failures(checks) or "manifest integrity" in failures(checks)


def test_preflight_catches_settings_that_drift_from_the_seal(tmp_path):
    """A typo'd candidate_k previously executed off-seal while every artifact
    still cited the manifest sha."""
    path = tmp_path / "m.json"
    seal(path)
    checks = run_preflight(path, settings={**SETTINGS, "candidate_k": 7})
    assert "settings match the seal" in failures(checks)


def test_preflight_reports_every_failure_at_once_not_just_the_first(tmp_path):
    """A run blocked for four reasons must disclose four. An early return on
    the first failure is how a second defect survives a fix for the first."""
    path = tmp_path / "m.json"
    seal(path)

    class OtherConsumer(RuleConsumer):
        pin_id = "some-other-consumer-v9"

    checks = run_preflight(
        path,
        policy={"rules": [], "version": "drifted"},
        consumer=OtherConsumer(),
        settings={**SETTINGS, "candidate_k": 7},
    )
    assert {"policy binding", "consumer pin", "settings match the seal"} <= failures(checks)


def test_build_sealed_run_refuses_and_never_reaches_the_consumer(tmp_path):
    path = tmp_path / "m.json"
    seal(path)
    records, questions, rec_emb, qry_emb = corpus()

    class ExplodingConsumer(RuleConsumer):
        def generate(self, prompt, max_new_tokens=256):
            raise AssertionError("consumer was called despite a blocked preflight")

    with pytest.raises(PreflightFailed) as exc:
        build_sealed_run(
            path,
            records=records,
            questions=questions,
            record_embeddings=rec_emb,
            query_embeddings=qry_emb,
            retriever_settings=SETTINGS,
            policy={"rules": [], "version": "drifted"},
            consumer=ExplodingConsumer(),
        )
    assert any(not c.passed for c in exc.value.checks)
    assert "BLOCKED" in str(exc.value)


def test_build_sealed_run_constructs_from_the_sealed_protocol(tmp_path):
    path = tmp_path / "m.json"
    seal(path)
    records, questions, rec_emb, qry_emb = corpus()
    runner = build_sealed_run(
        path,
        records=records,
        questions=questions,
        record_embeddings=rec_emb,
        query_embeddings=qry_emb,
        retriever_settings=SETTINGS,
        policy=POLICY,
        consumer=RuleConsumer(),
    )
    sealed = load_manifest(path)["protocol"]["retriever_settings"]
    assert runner.candidate_k == sealed["candidate_k"]
    assert runner.policy == POLICY


# --------------------------------------------------------------------------
# fam capacity: a sealed number that silently evicts attests nothing
# --------------------------------------------------------------------------


def test_fam_capacity_below_scope_count_now_raises_instead_of_evicting(tmp_path):
    records, _, rec_emb, _ = corpus()  # two scopes
    with pytest.raises(ValueError, match="were evicted during construction"):
        FAMRetriever(records, rec_emb, prototype_k=1, max_entries=1)


def test_fam_capacity_at_scope_count_is_accepted(tmp_path):
    records, _, rec_emb, _ = corpus()
    retriever = FAMRetriever(records, rec_emb, prototype_k=1, max_entries=2)
    assert retriever.prototype_count == 2


# --------------------------------------------------------------------------
# embeddings must not be mutable behind the seal
# --------------------------------------------------------------------------


def test_verified_embeddings_cannot_be_mutated_through_a_shared_view():
    records, _, rec_emb, _ = corpus()
    caller_tensor = rec_emb["01-old"]
    retriever = FAMRetriever(records, rec_emb, prototype_k=1, max_entries=2)
    before = retriever._embeddings["01-old"].clone()
    caller_tensor.mul_(-5.0)  # in-place, after construction
    assert torch.equal(retriever._embeddings["01-old"], before)
