import json
from hashlib import sha256

import pytest

from harness.memory_eval.manifest import (
    MANIFEST_VERSION,
    canonical_json,
    load_manifest,
    seal_manifest,
    verify_manifest,
)
from harness.memory_eval.models import MemoryQuestion, MemoryRecord, fact_scope


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


def inputs():
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
    record_embeddings = {
        "01-old": [1.0, 0.0],
        "02-new": [0.9, 0.1],
        "03-clean": [0.0, 1.0],
    }
    query_embeddings = {
        "q-evolving": [0.9, 0.1],
        "q-clean": [0.0, 1.0],
    }
    return records, questions, record_embeddings, query_embeddings, dict(SETTINGS)


def reseal_top_level(path, field, value):
    manifest = load_manifest(path)
    manifest[field] = value
    body = dict(manifest)
    del body["manifest_sha256"]
    manifest["manifest_sha256"] = sha256(
        canonical_json(body).encode("utf-8")
    ).hexdigest()
    path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")


def test_v3_plumbing_seal_is_deterministic_and_omits_confirmatory_evidence(tmp_path):
    values = inputs()
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    first = seal_manifest(first_path, *values, evidence_class="plumbing")
    second = seal_manifest(second_path, *values, evidence_class="plumbing")

    assert MANIFEST_VERSION == "memory-eval-manifest-v3"
    assert first == second
    assert len(first["manifest_sha256"]) == 64
    assert "index_attestations" not in first["protocol"]
    assert first["protocol"]["registration"] is None
    assert load_manifest(first_path) == first
    assert verify_manifest(first_path, *values, evidence_class="plumbing") == first


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("registration", {"not": "allowed"}),
        ("registration_memo_path", "/tmp/not-allowed.md"),
        ("index_attestations", {"exemplar": {}, "fam": {}}),
    ],
)
def test_verify_refuses_digest_consistent_extra_top_level_fields(
    tmp_path, field, value
):
    values = inputs()
    path = tmp_path / "manifest.json"
    seal_manifest(path, *values, evidence_class="plumbing")
    reseal_top_level(path, field, value)

    with pytest.raises(RuntimeError, match=rf"manifest body mismatch.*{field}"):
        verify_manifest(path, *values, evidence_class="plumbing")


def test_seal_rejects_a_missing_retriever_setting(tmp_path):
    records, questions, rec_emb, qry_emb, settings = inputs()
    del settings["cam_key_lr"]

    with pytest.raises(ValueError, match="missing retriever settings.*cam_key_lr"):
        seal_manifest(
            tmp_path / "m.json",
            records,
            questions,
            rec_emb,
            qry_emb,
            settings,
            evidence_class="plumbing",
        )


def test_seal_rejects_an_unknown_retriever_setting(tmp_path):
    records, questions, rec_emb, qry_emb, settings = inputs()
    settings["candidate_width_default"] = 10

    with pytest.raises(ValueError, match="unknown retriever settings.*candidate_width_default"):
        seal_manifest(
            tmp_path / "m.json",
            records,
            questions,
            rec_emb,
            qry_emb,
            settings,
            evidence_class="plumbing",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("candidate_k", True, "candidate_k must be a positive integer"),
        ("cam_max_entries", 3.0, "cam_max_entries must be a positive integer"),
        ("cam_prototype_k", False, "cam_prototype_k must be a positive integer"),
        ("cam_vigilance", "0.85", "cam_vigilance must be a finite number"),
        ("cam_use_bfloat16", 0, "cam_use_bfloat16 must be a boolean"),
        ("cam_adaptive_eviction", 1, "cam_adaptive_eviction must be a boolean"),
        ("cam_use_lfu", 1, "cam_use_lfu must be a boolean"),
        ("cam_sleep", 0, "cam_sleep must be a boolean"),
    ],
)
def test_seal_rejects_wrong_types_including_integer_boolean_impostors(
    tmp_path, field, value, message
):
    records, questions, rec_emb, qry_emb, settings = inputs()
    settings[field] = value

    with pytest.raises(ValueError, match=message):
        seal_manifest(
            tmp_path / "m.json",
            records,
            questions,
            rec_emb,
            qry_emb,
            settings,
            evidence_class="plumbing",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cam_dynamic_vigilance", False),
        ("cam_retrieval_floor", 0.0),
        ("cam_retrieval_truncation", 4),
        ("cam_nstp", {}),
        ("cam_adaptive_eviction", True),
        ("cam_sleep", True),
    ],
)
def test_seal_requires_optional_policies_to_be_explicitly_disabled(
    tmp_path, field, value
):
    records, questions, rec_emb, qry_emb, settings = inputs()
    settings[field] = value

    with pytest.raises(ValueError, match=field):
        seal_manifest(
            tmp_path / "m.json",
            records,
            questions,
            rec_emb,
            qry_emb,
            settings,
            evidence_class="plumbing",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cam_vigilance", float("nan")),
        ("cam_vigilance", 1.01),
        ("cam_hebb_lr", -0.01),
        ("cam_hebb_lr", 1.01),
        ("cam_key_lr", float("inf")),
        ("cam_key_lr", 1.01),
        ("cam_ema_beta", 1.01),
        ("cam_inference_temp", 0.0),
    ],
)
def test_seal_rejects_invalid_numeric_settings(tmp_path, field, value):
    records, questions, rec_emb, qry_emb, settings = inputs()
    settings[field] = value

    with pytest.raises(ValueError, match=field):
        seal_manifest(
            tmp_path / "m.json",
            records,
            questions,
            rec_emb,
            qry_emb,
            settings,
            evidence_class="plumbing",
        )


def test_seal_requires_cam_capacity_to_equal_the_three_record_fixture(tmp_path):
    records, questions, rec_emb, qry_emb, settings = inputs()
    settings["cam_max_entries"] = 2

    with pytest.raises(ValueError, match="cam_max_entries 2 must equal record count 3"):
        seal_manifest(
            tmp_path / "m.json",
            records,
            questions,
            rec_emb,
            qry_emb,
            settings,
            evidence_class="plumbing",
        )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("cam_ingest_order", "sorted-record-id", "manifest-record-order"),
        ("exemplar_write_mode", "condense", "allocate-only"),
        ("fam_write_mode", "allocate-only", "condense"),
    ],
)
def test_seal_requires_fixed_ingest_and_write_modes(tmp_path, field, value, expected):
    records, questions, rec_emb, qry_emb, settings = inputs()
    settings[field] = value

    with pytest.raises(ValueError, match=f"{field} must be {expected!r}"):
        seal_manifest(
            tmp_path / "m.json",
            records,
            questions,
            rec_emb,
            qry_emb,
            settings,
            evidence_class="plumbing",
        )


@pytest.mark.parametrize("changed", ["records", "questions", "record_embeddings", "settings"])
def test_verify_rejects_changed_inputs_or_treatment(tmp_path, changed):
    records, questions, record_embeddings, query_embeddings, settings = inputs()
    path = tmp_path / "manifest.json"
    seal_manifest(
        path,
        records,
        questions,
        record_embeddings,
        query_embeddings,
        settings,
        evidence_class="plumbing",
    )

    if changed == "records":
        records = tuple(reversed(records))
    elif changed == "questions":
        questions = tuple(reversed(questions))
    elif changed == "record_embeddings":
        record_embeddings = {**record_embeddings, "01-old": [0.8, 0.2]}
    else:
        settings = {**settings, "cam_vigilance": 0.9}

    with pytest.raises(RuntimeError, match="manifest fingerprint mismatch"):
        verify_manifest(
            path,
            records,
            questions,
            record_embeddings,
            query_embeddings,
            settings,
            evidence_class="plumbing",
        )


def test_load_refuses_unsealed_or_tampered_manifests(tmp_path):
    unsealed = tmp_path / "unsealed.json"
    unsealed.write_text(json.dumps({"sealed": False}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not sealed"):
        load_manifest(unsealed)

    values = inputs()
    path = tmp_path / "manifest.json"
    manifest = seal_manifest(path, *values, evidence_class="plumbing")
    manifest["fingerprints"]["records"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="seal digest mismatch"):
        load_manifest(path)
