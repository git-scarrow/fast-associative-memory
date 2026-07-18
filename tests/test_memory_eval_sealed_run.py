"""Phase-A evidence boundaries and full-manifest preflight binding."""

from dataclasses import FrozenInstanceError
from hashlib import sha256
from pathlib import Path

import pytest

from harness.memory_eval import manifest as manifest_module
from harness.memory_eval import sealed_run as sealed_run_module
from harness.memory_eval.dry_run import RuleConsumer
from harness.memory_eval.manifest import canonical_json, load_manifest, seal_manifest
from harness.memory_eval.models import MemoryQuestion, MemoryRecord, fact_scope
from harness.memory_eval.preregistration import BLOCKED, experiment_verdict
from harness.memory_eval.retrievers import ExemplarCAMRetriever, FAMRetriever
from harness.memory_eval.sealed_run import (
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


def registration(memo_path: Path) -> dict[str, object]:
    """Synthetic values exercise schema only; they carry no experiment authority."""
    return {
        "prototype_reduction_margin": 0.1,
        "mechanism_recall_loss_bound": 0.1,
        "min_mechanism_recall_n": 3,
        "candidate_k": 2,
        "cam_prototype_k": 1,
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


def corpus():
    evolving = fact_scope("Ada", "employer")
    clean = fact_scope("Grace", "city")
    records = (
        MemoryRecord(
            "01-old", evolving, "OldCo", "FACT[Ada|employer]=OldCo", 1,
            "2026-01-01T00:00:00Z",
        ),
        MemoryRecord(
            "02-new", evolving, "NewCo", "FACT[Ada|employer]=NewCo", 2,
            "2026-02-01T00:00:00Z",
        ),
        MemoryRecord(
            "03-clean", clean, "Detroit", "FACT[Grace|city]=Detroit", 1,
            "2026-01-01T00:00:00Z",
        ),
    )
    questions = (
        MemoryQuestion("q-evolving", "What is FACT[Ada|employer]?", evolving, "NewCo"),
        MemoryQuestion("q-clean", "What is FACT[Grace|city]?", clean, "Detroit"),
    )
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


def seal_plumbing(path: Path):
    return seal_manifest(
        path,
        *corpus(),
        SETTINGS,
        evidence_class="plumbing",
        policy=POLICY,
        consumer_pin=RuleConsumer.pin_id,
    )


def handbuild_scoring_manifest(path: Path):
    """Model a digest-consistent external scoring envelope for refusal tests."""
    records, questions, rec_emb, qry_emb = corpus()
    memo_path = path.with_suffix(".registration.md")
    memo_path.write_bytes(b"# Synthetic schema exercise only\n")
    registered = registration(memo_path)
    body = manifest_module._manifest_body(
        records,
        questions,
        rec_emb,
        qry_emb,
        SETTINGS,
        evidence_class="scoring-run",
        policy=POLICY,
        consumer_pin=RuleConsumer.pin_id,
        registration=registered,
        registration_memo_path=str(memo_path.resolve()),
    )
    manifest = {**body, "manifest_sha256": manifest_module._fingerprint(body)}
    path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def reseal(path: Path, mutate) -> None:
    manifest = load_manifest(path)
    mutate(manifest)
    body = dict(manifest)
    del body["manifest_sha256"]
    manifest["manifest_sha256"] = manifest_module._fingerprint(body)
    path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")


def protocol_mutation(field: str, value):
    def mutate(manifest):
        protocol = manifest["protocol"]
        protocol[field] = value
        manifest["fingerprints"]["protocol"] = manifest_module._fingerprint(protocol)

    return mutate


def build_plumbing(path: Path, *, records=None, settings=SETTINGS, consumer=None):
    corpus_records, questions, rec_emb, qry_emb = corpus()
    return build_plumbing_run(
        path,
        records=corpus_records if records is None else records,
        questions=questions,
        record_embeddings=rec_emb,
        query_embeddings=qry_emb,
        retriever_settings=settings,
        policy=POLICY,
        consumer=consumer or RuleConsumer(),
    )


def run_preflight(path, *, settings=SETTINGS, consumer=None):
    records, questions, rec_emb, qry_emb = corpus()
    return preflight(
        path,
        records=records,
        questions=questions,
        record_embeddings=rec_emb,
        query_embeddings=qry_emb,
        retriever_settings=settings,
        policy=POLICY,
        consumer=consumer or RuleConsumer(),
    )


def failures(checks):
    return {check.name for check in checks if not check.passed}


def test_synthetic_inputs_and_rule_consumer_cannot_obtain_scoring_status(tmp_path):
    path = tmp_path / "scoring.json"
    records, questions, rec_emb, qry_emb = corpus()
    memo_path = tmp_path / "registration.md"
    memo_path.write_bytes(b"# Synthetic schema exercise only\n")

    with pytest.raises(
        RuntimeError,
        match="Phase B provenance/reconciliation envelope is not implemented",
    ):
        seal_manifest(
            path,
            records,
            questions,
            rec_emb,
            qry_emb,
            SETTINGS,
            evidence_class="scoring-run",
            policy=POLICY,
            consumer_pin=RuleConsumer.pin_id,
            registration=registration(memo_path),
            registration_memo_path=memo_path,
        )

    assert not path.exists()


def test_plumbing_bundle_preserves_nonconfirmatory_provenance(tmp_path):
    path = tmp_path / "plumbing.json"
    manifest = seal_plumbing(path)

    bundle = build_plumbing(path)

    assert isinstance(bundle, sealed_run_module.PlumbingRunBundle)
    assert bundle.manifest_sha256 == manifest["manifest_sha256"]
    assert bundle.evidence_class == "plumbing"
    assert bundle.admissible is False
    assert bundle.receipt.manifest_sha256 == manifest["manifest_sha256"]
    assert bundle.receipt.evidence_class == "plumbing"
    assert bundle.receipt.passed is True
    assert bundle.receipt.confirmatory is False
    assert isinstance(bundle.runner.exemplar_retriever, ExemplarCAMRetriever)
    assert isinstance(bundle.runner.fam_retriever, FAMRetriever)
    assert bundle.rebuilt_attestations.exemplar == bundle.runner.exemplar_retriever.attestation
    assert bundle.rebuilt_attestations.fam == bundle.runner.fam_retriever.attestation
    with pytest.raises(FrozenInstanceError):
        bundle.admissible = True


def test_plumbing_bundle_receipt_cannot_authorize_an_authoritative_go(tmp_path):
    path = tmp_path / "plumbing.json"
    seal_plumbing(path)
    bundle = build_plumbing(path)

    assert experiment_verdict(
        receipt=bundle.receipt,
        evaluable=True,
        mechanism_ok=True,
        application_h1=True,
        application_h2=True,
        application_h3=True,
        mechanism_active=True,
    ) == BLOCKED


@pytest.mark.parametrize(
    ("block", "value"),
    [
        ("registration", {"not": "allowed"}),
        ("registration_memo_path", "/tmp/not-allowed.md"),
        ("index_attestations", {"exemplar": {}, "fam": {}}),
    ],
)
def test_build_plumbing_run_rejects_confirmatory_blocks(tmp_path, block, value):
    path = tmp_path / "plumbing.json"
    seal_plumbing(path)
    reseal(path, lambda manifest: manifest["protocol"].__setitem__(block, value))

    with pytest.raises(RuntimeError, match=rf"plumbing.*{block}"):
        build_plumbing(path)


def test_build_plumbing_run_rejects_full_manifest_drift_before_generation(tmp_path):
    path = tmp_path / "plumbing.json"
    seal_plumbing(path)
    reseal(path, lambda manifest: manifest.__setitem__("future_envelope", {}))

    class ExplodingConsumer(RuleConsumer):
        generated = False

        def generate(self, prompt, max_new_tokens=256):
            self.generated = True
            raise AssertionError("consumer ran before full-manifest verification")

    consumer = ExplodingConsumer()
    with pytest.raises(RuntimeError, match="manifest body mismatch.*future_envelope"):
        build_plumbing(path, consumer=consumer)
    assert consumer.generated is False


@pytest.mark.parametrize(
    "mutation",
    [
        protocol_mutation("arms", ["changed"]),
        protocol_mutation("context_budget_tokens", 1),
        protocol_mutation("scoring_version", "changed"),
        lambda manifest: manifest.__setitem__("record_count", 999),
        lambda manifest: manifest.__setitem__("question_count", 999),
        lambda manifest: manifest.__setitem__("future_envelope", {}),
    ],
    ids=["arms", "budget", "protocol-identity", "record-count", "question-count", "unknown"],
)
def test_preflight_names_digest_consistent_full_manifest_drift(tmp_path, mutation):
    path = tmp_path / "handbuilt-scoring.json"
    handbuild_scoring_manifest(path)
    reseal(path, mutation)

    assert "full manifest binding" in failures(run_preflight(path))


def test_preflight_audits_handbuilt_scoring_but_phase_b_envelope_stays_closed(tmp_path):
    path = tmp_path / "handbuilt-scoring.json"
    handbuild_scoring_manifest(path)

    checks = run_preflight(path)

    assert "full manifest binding" not in failures(checks)
    assert "Phase B provenance/reconciliation envelope" in failures(checks)


@pytest.mark.parametrize(
    ("field", "value"),
    [("candidate_k", 99), ("cam_prototype_k", 99)],
)
def test_preflight_binds_registered_widths_to_the_verified_treatment(
    tmp_path, field, value
):
    path = tmp_path / "handbuilt-scoring.json"
    handbuild_scoring_manifest(path)

    def change_registration(manifest):
        protocol = manifest["protocol"]
        protocol["registration"][field] = value
        manifest["fingerprints"]["protocol"] = manifest_module._fingerprint(protocol)

    reseal(
        path,
        change_registration,
    )

    checks = run_preflight(path)

    assert "full manifest binding" not in failures(checks)
    assert "registration complete" in failures(checks)


def test_build_sealed_run_refuses_handbuilt_scoring_before_generation(tmp_path):
    path = tmp_path / "handbuilt-scoring.json"
    handbuild_scoring_manifest(path)
    records, questions, rec_emb, qry_emb = corpus()

    class ExplodingConsumer(RuleConsumer):
        generated = False

        def generate(self, prompt, max_new_tokens=256):
            self.generated = True
            raise AssertionError("consumer ran in Phase A")

    consumer = ExplodingConsumer()
    with pytest.raises(
        RuntimeError,
        match="Phase B provenance/reconciliation envelope is not implemented",
    ):
        build_sealed_run(
            path,
            records=records,
            questions=questions,
            record_embeddings=rec_emb,
            query_embeddings=qry_emb,
            retriever_settings=SETTINGS,
            policy=POLICY,
            consumer=consumer,
        )
    assert consumer.generated is False


def test_raw_normalized_collision_fails_plumbing_verification_before_generation(tmp_path):
    path = tmp_path / "plumbing.json"
    seal_plumbing(path)
    records, _, _, _ = corpus()
    collision = MemoryRecord(
        "04-collision",
        records[2].scope,
        "  DETROIT ",
        "FACT[Grace|city]=  DETROIT ",
        records[2].serial,
    )

    class ExplodingConsumer(RuleConsumer):
        generated = False

        def generate(self, prompt, max_new_tokens=256):
            self.generated = True
            raise AssertionError("consumer ran before raw-with-invariant validation")

    consumer = ExplodingConsumer()
    with pytest.raises(ValueError, match="raw-with-invariant"):
        build_plumbing(
            path,
            records=(*records, collision),
            settings={**SETTINGS, "cam_max_entries": 4},
            consumer=consumer,
        )
    assert consumer.generated is False
