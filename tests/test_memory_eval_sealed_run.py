"""Manifest v3 seals and rebuilds the realized matched CAM treatment."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import pytest

from harness.memory_eval.dry_run import RuleConsumer
from harness.memory_eval.manifest import load_manifest, seal_manifest
from harness.memory_eval.models import MemoryQuestion, MemoryRecord, fact_scope
from harness.memory_eval.preregistration import sentinel
from harness.memory_eval.retrievers import ExemplarCAMRetriever, FAMRetriever
from harness.memory_eval.sealed_run import (
    PreflightFailed,
    build_plumbing_run,
    build_sealed_run,
    preflight,
)


POLICY = {"rules": [{"id": "R01", "action": "assert"}], "version": "test-policy-v1"}
SETTINGS = {
    "candidate_k": 2,
    "cam_max_entries": 3,
    "cam_prototype_k": 1,
    "cam_vigilance": 0.85,
    "cam_hebb_lr": 0.1,
    "cam_key_lr": 0.05,
    "cam_ema_beta": 0.05,
    "cam_inference_temp": 0.05,
    "cam_use_bfloat16": False,
    "cam_adaptive_eviction": False,
    "cam_use_lfu": True,
    "cam_dynamic_vigilance": None,
    "cam_retrieval_floor": None,
    "cam_retrieval_truncation": None,
    "cam_nstp": None,
    "cam_sleep": False,
    "cam_ingest_order": "manifest-record-order",
    "exemplar_write_mode": "allocate-only",
    "fam_write_mode": "condense",
}


def registration(memo_path: Path, **overrides):
    """Complete mechanism/application registration with real memo bytes."""
    base = {
        "prototype_reduction_margin": 0.1,
        "mechanism_recall_loss_bound": 0.1,
        "min_mechanism_recall_n": 3,
        "stale_reduction_margin": 0.1,
        "clean_answer_loss_bound": 0.1,
        "current_adoption_floor": 0.5,
        "abstention_bound": None,
        "scorer": "exact",
        "raw_truncation": "skip",
        "contested_disposition": "exploratory",
        "equivalence": "raw-with-invariant",
        "min_stale_eligible_n": 40,
        "min_clean_n": 40,
        "h1_denominator": "fixed-full",
        "primary_family": "fam",
        "claim_order": "fam-mechanism-then-application",
        "memo_sha256": sha256(memo_path.read_bytes()).hexdigest(),
    }
    base.update(overrides)
    return base


def corpus():
    evolving = fact_scope("Ada", "employer")
    clean = fact_scope("Grace", "city")
    records = (
        MemoryRecord(
            "01-old",
            evolving,
            "OldCo",
            "FACT[Ada|employer]=OldCo",
            1,
            "2026-01-01T00:00:00Z",
        ),
        MemoryRecord(
            "02-new",
            evolving,
            "NewCo",
            "FACT[Ada|employer]=NewCo",
            2,
            "2026-02-01T00:00:00Z",
        ),
        MemoryRecord(
            "03-clean",
            clean,
            "Detroit",
            "FACT[Grace|city]=Detroit",
            1,
            "2026-01-01T00:00:00Z",
        ),
    )
    questions = (
        MemoryQuestion(
            "q-evolving", "What is FACT[Ada|employer]?", evolving, "NewCo"
        ),
        MemoryQuestion("q-clean", "What is FACT[Grace|city]?", clean, "Detroit"),
    )
    # Same-scope records exceed vigilance but are not identical: F0 realizes
    # both a merge and key drift. E0 still allocates three immutable exemplars.
    record_embeddings = {
        "01-old": [1.0, 0.0],
        "02-new": [0.9, 0.1],
        "03-clean": [0.0, 1.0],
    }
    query_embeddings = {
        "q-evolving": [0.9, 0.1],
        "q-clean": [0.0, 1.0],
    }
    return records, questions, record_embeddings, query_embeddings


def seal(path: Path, *, settings=SETTINGS, registration_overrides=None, **overrides):
    records, questions, rec_emb, qry_emb = corpus()
    memo_path = path.with_suffix(".registration.md")
    memo_path.write_bytes(b"# Fixed test preregistration\n")
    kwargs = {
        "evidence_class": "scoring-run",
        "policy": POLICY,
        "consumer_pin": RuleConsumer.pin_id,
        "registration": registration(memo_path, **(registration_overrides or {})),
        "registration_memo_path": memo_path,
    }
    kwargs.update(overrides)
    manifest = seal_manifest(
        path,
        records,
        questions,
        rec_emb,
        qry_emb,
        settings,
        **kwargs,
    )
    return manifest, memo_path


def run_preflight(path, *, policy=POLICY, consumer=None, settings=SETTINGS, records=None):
    corpus_records, questions, rec_emb, qry_emb = corpus()
    if records is None:
        records = corpus_records
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
    return {check.name for check in checks if not check.passed}


def reseal(path: Path, mutate):
    from harness.memory_eval.manifest import _fingerprint

    manifest = load_manifest(path)
    mutate(manifest)
    body = dict(manifest)
    del body["manifest_sha256"]
    manifest["manifest_sha256"] = _fingerprint(body)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_scoring_seal_contains_complete_realized_e0_and_f0_attestations(tmp_path):
    manifest, _ = seal(tmp_path / "m.json")
    attestations = manifest["protocol"]["index_attestations"]

    assert set(attestations) == {"exemplar", "fam"}
    assert attestations["exemplar"]["mode"] == "allocate-only"
    assert attestations["exemplar"]["prototype_count"] == 3
    assert attestations["fam"]["mode"] == "condense"
    assert attestations["fam"]["merged"] > 0
    assert attestations["fam"]["key_drifted_merges"] > 0
    assert len(attestations["exemplar"]["index_sha256"]) == 64
    assert len(attestations["fam"]["index_sha256"]) == 64


def test_scoring_seal_refuses_an_incomplete_mechanism_registration(tmp_path):
    with pytest.raises(RuntimeError, match="still unregistered"):
        seal(
            tmp_path / "m.json",
            registration_overrides={"prototype_reduction_margin": sentinel("D-M1")},
        )


def test_scoring_seal_requires_a_real_registered_memo_path(tmp_path):
    records, questions, rec_emb, qry_emb = corpus()
    with pytest.raises(RuntimeError, match="registration memo path"):
        seal_manifest(
            tmp_path / "m.json",
            records,
            questions,
            rec_emb,
            qry_emb,
            SETTINGS,
            evidence_class="scoring-run",
            policy=POLICY,
            consumer_pin=RuleConsumer.pin_id,
            registration={"memo_sha256": "0" * 64},
        )


def test_scoring_seal_rejects_a_memo_sha_not_matching_the_registered_bytes(tmp_path):
    with pytest.raises(RuntimeError, match="memo_sha256.*does not match"):
        seal(
            tmp_path / "m.json",
            registration_overrides={"memo_sha256": "0" * 64},
        )


def test_plumbing_seal_has_no_registration_or_confirmatory_attestations(tmp_path):
    records, questions, rec_emb, qry_emb = corpus()
    path = tmp_path / "m.json"
    manifest = seal_manifest(
        path,
        records,
        questions,
        rec_emb,
        qry_emb,
        SETTINGS,
        evidence_class="plumbing",
    )

    assert manifest["protocol"]["registration"] is None
    assert "registration_memo_path" not in manifest["protocol"]
    assert "index_attestations" not in manifest["protocol"]
    assert {"evidence class", "registration complete", "exemplar index", "FAM index"} <= failures(
        run_preflight(path)
    )


def seal_plumbing(path: Path):
    records, questions, rec_emb, qry_emb = corpus()
    return seal_manifest(
        path,
        records,
        questions,
        rec_emb,
        qry_emb,
        SETTINGS,
        evidence_class="plumbing",
        policy=POLICY,
        consumer_pin=RuleConsumer.pin_id,
    )


def build_plumbing(path: Path, *, settings=SETTINGS, consumer=None):
    records, questions, rec_emb, qry_emb = corpus()
    return build_plumbing_run(
        path,
        records=records,
        questions=questions,
        record_embeddings=rec_emb,
        query_embeddings=qry_emb,
        retriever_settings=settings,
        policy=POLICY,
        consumer=consumer or RuleConsumer(),
    )


def test_build_plumbing_run_rebuilds_only_from_verified_sealed_settings(tmp_path):
    path = tmp_path / "plumbing.json"
    manifest = seal_plumbing(path)

    runner = build_plumbing(path)

    assert manifest["protocol"]["evidence_class"] == "plumbing"
    assert "index_attestations" not in manifest["protocol"]
    assert runner.candidate_k == SETTINGS["candidate_k"]
    assert isinstance(runner.exemplar_retriever, ExemplarCAMRetriever)
    assert isinstance(runner.fam_retriever, FAMRetriever)
    assert runner.exemplar_retriever.attestation.prototype_count == 3
    assert runner.fam_retriever.attestation.merged == 1
    assert runner.fam_retriever.attestation.key_drifted_merges == 1


@pytest.mark.parametrize(
    ("block", "value"),
    [
        ("registration", {"not": "allowed"}),
        ("registration_memo_path", "/tmp/not-allowed.md"),
        ("index_attestations", {"exemplar": {}, "fam": {}}),
    ],
)
def test_build_plumbing_run_rejects_forbidden_confirmatory_blocks(
    tmp_path, block, value
):
    path = tmp_path / "plumbing.json"
    seal_plumbing(path)
    reseal(
        path,
        lambda manifest: manifest["protocol"].__setitem__(block, value),
    )

    with pytest.raises(RuntimeError, match=rf"plumbing.*{block}"):
        build_plumbing(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("registration", {"not": "allowed"}),
        ("registration_memo_path", "/tmp/not-allowed.md"),
        ("index_attestations", {"exemplar": {}, "fam": {}}),
    ],
)
def test_build_plumbing_run_refuses_digest_consistent_extra_top_level_fields(
    tmp_path, field, value
):
    path = tmp_path / "plumbing.json"
    seal_plumbing(path)
    reseal(path, lambda manifest: manifest.__setitem__(field, value))

    class ExplodingConsumer(RuleConsumer):
        generated = False

        def generate(self, prompt, max_new_tokens=256):
            self.generated = True
            raise AssertionError("consumer ran before manifest-body verification")

    consumer = ExplodingConsumer()
    with pytest.raises(RuntimeError, match=rf"manifest body mismatch.*{field}"):
        runner = build_plumbing(path, consumer=consumer)
        records, questions, rec_emb, qry_emb = corpus()
        runner.run(questions, qry_emb)
    assert consumer.generated is False


def test_build_plumbing_run_rejects_a_scoring_run(tmp_path):
    path = tmp_path / "scoring.json"
    seal(path)

    with pytest.raises(RuntimeError, match="requires a plumbing evidence class"):
        build_plumbing(path)


@pytest.mark.parametrize("drift", ["settings", "consumer-pin"])
def test_build_plumbing_run_rejects_live_drift_before_generation(tmp_path, drift):
    path = tmp_path / "plumbing.json"
    seal_plumbing(path)

    class ExplodingConsumer(RuleConsumer):
        generated = False
        pin_id = (
            "changed-rule-consumer-v1"
            if drift == "consumer-pin"
            else RuleConsumer.pin_id
        )

        def generate(self, prompt, max_new_tokens=256):
            self.generated = True
            raise AssertionError("consumer was called before plumbing verification")

    consumer = ExplodingConsumer()
    settings = (
        {**SETTINGS, "cam_key_lr": 0.01} if drift == "settings" else SETTINGS
    )

    with pytest.raises(RuntimeError, match="manifest fingerprint mismatch"):
        runner = build_plumbing(path, settings=settings, consumer=consumer)
        records, questions, rec_emb, qry_emb = corpus()
        runner.run(questions, qry_emb)
    assert consumer.generated is False


def test_clean_scoring_seal_passes_every_preflight_check(tmp_path):
    path = tmp_path / "m.json"
    seal(path)

    checks = run_preflight(path)

    assert failures(checks) == set()
    names = {check.name for check in checks}
    assert {
        "treatment settings",
        "exemplar index",
        "FAM index",
        "mechanism activity",
        "provenance and capacity integrity",
        "registration memo",
    } <= names


def test_preflight_rejects_a_changed_fam_numeric_setting(tmp_path):
    path = tmp_path / "m.json"
    seal(path)

    checks = run_preflight(path, settings={**SETTINGS, "cam_key_lr": 0.01})

    assert "treatment settings" in failures(checks)


def test_preflight_rejects_a_reordered_input_build(tmp_path):
    path = tmp_path / "m.json"
    seal(path)
    records, _, _, _ = corpus()

    checks = run_preflight(path, records=tuple(reversed(records)))

    assert {"input integrity", "exemplar index", "FAM index"} <= failures(checks)


def test_preflight_rejects_a_tampered_e0_attestation(tmp_path):
    path = tmp_path / "m.json"
    seal(path)
    reseal(
        path,
        lambda manifest: manifest["protocol"]["index_attestations"]["exemplar"].__setitem__(
            "allocated", 2
        ),
    )

    assert "exemplar index" in failures(run_preflight(path))


def test_preflight_rejects_a_tampered_f0_index_hash(tmp_path):
    path = tmp_path / "m.json"
    seal(path)
    reseal(
        path,
        lambda manifest: manifest["protocol"]["index_attestations"]["fam"].__setitem__(
            "index_sha256", "0" * 64
        ),
    )

    assert "FAM index" in failures(run_preflight(path))


def test_preflight_rejects_changed_memo_bytes_after_seal(tmp_path):
    path = tmp_path / "m.json"
    _, memo_path = seal(path)
    memo_path.write_bytes(b"changed after sealing\n")

    assert "registration memo" in failures(run_preflight(path))


def test_preflight_rejects_a_resealed_changed_memo_sha(tmp_path):
    path = tmp_path / "m.json"
    seal(path)
    reseal(
        path,
        lambda manifest: manifest["protocol"]["registration"].__setitem__(
            "memo_sha256", "0" * 64
        ),
    )

    assert "registration memo" in failures(run_preflight(path))


def test_preflight_rejects_a_resealed_incomplete_settings_schema(tmp_path):
    path = tmp_path / "m.json"
    seal(path)
    reseal(
        path,
        lambda manifest: manifest["protocol"]["retriever_settings"].pop(
            "cam_prototype_k"
        ),
    )

    assert "treatment settings" in failures(run_preflight(path))


@pytest.mark.parametrize(
    "failure_case",
    ["live-setting", "reordered-input", "exemplar-attestation", "memo-bytes"],
)
def test_every_preflight_failure_blocks_before_consumer_generation(
    tmp_path, failure_case
):
    path = tmp_path / "m.json"
    _, memo_path = seal(path)
    records, questions, rec_emb, qry_emb = corpus()
    settings = SETTINGS
    if failure_case == "live-setting":
        settings = {**SETTINGS, "cam_vigilance": 0.9}
    elif failure_case == "reordered-input":
        records = tuple(reversed(records))
    elif failure_case == "exemplar-attestation":
        reseal(
            path,
            lambda manifest: manifest["protocol"]["index_attestations"][
                "exemplar"
            ].__setitem__("written", 2),
        )
    else:
        memo_path.write_bytes(b"changed after seal\n")

    class ExplodingConsumer(RuleConsumer):
        generated = False

        def generate(self, prompt, max_new_tokens=256):
            self.generated = True
            raise AssertionError("consumer was called despite a blocked preflight")

    consumer = ExplodingConsumer()
    with pytest.raises(PreflightFailed):
        runner = build_sealed_run(
            path,
            records=records,
            questions=questions,
            record_embeddings=rec_emb,
            query_embeddings=qry_emb,
            retriever_settings=settings,
            policy=POLICY,
            consumer=consumer,
        )
        runner.run(questions, qry_emb)
    assert consumer.generated is False


def test_build_sealed_run_uses_only_sealed_settings_and_rebuilt_indexes(tmp_path):
    path = tmp_path / "m.json"
    manifest, _ = seal(path)
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

    attestations = manifest["protocol"]["index_attestations"]
    assert runner.candidate_k == SETTINGS["candidate_k"]
    assert isinstance(runner.exemplar_retriever, ExemplarCAMRetriever)
    assert isinstance(runner.fam_retriever, FAMRetriever)
    assert asdict(runner.exemplar_retriever.attestation) == attestations["exemplar"]
    assert asdict(runner.fam_retriever.attestation) == attestations["fam"]
    assert runner.exemplar_retriever.cam.inference_k == SETTINGS["cam_prototype_k"]
    assert runner.fam_retriever.cam.inference_k == SETTINGS["cam_prototype_k"]
