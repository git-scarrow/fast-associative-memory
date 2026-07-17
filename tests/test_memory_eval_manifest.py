import json

import pytest

from harness.memory_eval.manifest import (
    load_manifest,
    seal_manifest,
    verify_manifest,
)
from harness.memory_eval.models import MemoryQuestion, MemoryRecord


def inputs():
    records = [
        MemoryRecord(
            record_id="r1",
            scope="Ada\x1femployer",
            value="B",
            content="Ada's employer is B.",
            serial=1,
            event_time="2026-01-01T00:00:00Z",
        )
    ]
    questions = [MemoryQuestion("q1", "Who employs Ada?", records[0].scope, "B")]
    record_embeddings = {"r1": [1.0, 0.0]}
    query_embeddings = {"q1": [0.9, 0.1]}
    settings = {"candidate_k": 5, "fam_prototype_k": 2, "fam_max_entries": 1}
    return records, questions, record_embeddings, query_embeddings, settings


def test_seal_is_deterministic_and_round_trips(tmp_path):
    values = inputs()
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    first = seal_manifest(first_path, *values, evidence_class="plumbing")
    second = seal_manifest(second_path, *values, evidence_class="plumbing")

    assert first == second
    assert len(first["manifest_sha256"]) == 64
    assert load_manifest(first_path) == first
    assert verify_manifest(first_path, *values, evidence_class="plumbing") == first


@pytest.mark.parametrize("changed", ["records", "questions", "record_embeddings", "settings"])
def test_verify_rejects_changed_inputs(tmp_path, changed):
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
        records = [
            MemoryRecord(
                record_id="r1",
                scope="Ada\x1femployer",
                value="changed",
                content="changed",
                serial=1,
                event_time="2026-01-01T00:00:00Z",
            )
        ]
    elif changed == "questions":
        questions = [MemoryQuestion("q1", "Changed?", "Ada\x1femployer", "B")]
    elif changed == "record_embeddings":
        record_embeddings = {"r1": [0.0, 1.0]}
    else:
        settings = {"candidate_k": 6, "fam_prototype_k": 2, "fam_max_entries": 1}

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
