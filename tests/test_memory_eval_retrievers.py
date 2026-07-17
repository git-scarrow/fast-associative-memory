from dataclasses import fields

import pytest
import torch

from harness.memory_eval.models import MemoryRecord
from harness.memory_eval.retrievers import (
    CAMIndexSettings,
    ExactVectorRetriever,
    ExemplarCAMRetriever,
    FAMRetriever,
)


def record(record_id: str, scope: str, serial: int = 0) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id,
        scope=scope,
        value=record_id.upper(),
        content=f"The value is {record_id.upper()}.",
        serial=serial,
        event_time="2026-01-01T00:00:00Z",
    )


def settings(max_entries=4, prototype_k=2):
    return CAMIndexSettings(
        max_entries=max_entries,
        prototype_k=prototype_k,
        vigilance=0.85,
        hebb_lr=0.1,
        key_lr=0.05,
        ema_beta=0.05,
        inference_temp=0.05,
        use_bfloat16=False,
        adaptive_eviction=False,
        use_lfu=True,
    )


def test_exact_vector_retriever_ranks_cosine_with_stable_ties():
    records = [record("b", "s1"), record("a", "s2"), record("c", "s3")]
    embeddings = {
        "a": torch.tensor([1.0, 0.0]),
        "b": torch.tensor([1.0, 0.0]),
        "c": torch.tensor([0.0, 1.0]),
    }
    retriever = ExactVectorRetriever(records, embeddings)

    found = retriever.query(torch.tensor([1.0, 0.0]), k=3)

    assert [item.record_id for item in found] == ["a", "b", "c"]
    assert [item.rank for item in found] == [1, 2, 3]
    assert found[0].score == pytest.approx(1.0)


def test_retrievers_reject_invalid_embedding_tables():
    records = [record("a", "s1"), record("b", "s2")]

    with pytest.raises(ValueError, match="missing embeddings"):
        ExactVectorRetriever(records, {"a": [1.0, 0.0]})
    with pytest.raises(ValueError, match="duplicate record_id"):
        ExactVectorRetriever([records[0], records[0]], {"a": [1.0, 0.0]})
    with pytest.raises(ValueError, match="same dimension"):
        ExactVectorRetriever(records, {"a": [1.0, 0.0], "b": [1.0]})
    with pytest.raises(ValueError, match="non-zero"):
        ExactVectorRetriever(records, {"a": [0.0, 0.0], "b": [1.0, 0.0]})


def test_query_validation_is_clear():
    retriever = ExactVectorRetriever([record("a", "s1")], {"a": [1.0, 0.0]})
    with pytest.raises(ValueError, match="query dimension"):
        retriever.query([1.0], k=1)
    with pytest.raises(ValueError, match="positive"):
        retriever.query([1.0, 0.0], k=0)


def test_live_fam_streams_actual_embeddings_and_merges_with_key_drift():
    records = [record("old", "scope", serial=1), record("new", "scope", serial=2)]
    embeddings = {"old": [1.0, 0.0], "new": [0.9, 0.1]}
    retriever = FAMRetriever(records, embeddings, settings=settings(max_entries=2))
    assert retriever.attestation.written == 2
    assert retriever.attestation.allocated == 1
    assert retriever.attestation.merged == 1
    assert retriever.attestation.key_drifted_merges == 1
    assert retriever.prototype_count == 1
    assert retriever.provenance_for_scope("scope") == {"old", "new"}


def test_below_vigilance_same_scope_allocates_second_fam_prototype():
    records = [record("a", "scope"), record("b", "scope", serial=1)]
    embeddings = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    retriever = FAMRetriever(records, embeddings, settings=settings(max_entries=2))
    assert retriever.attestation.merged == 0
    assert retriever.prototype_count == 2


def test_allocate_only_control_matches_envelope_but_never_merges():
    records = [record("a", "scope"), record("b", "scope", serial=1)]
    embeddings = {"a": [1.0, 0.0], "b": [0.9, 0.1]}
    retriever = ExemplarCAMRetriever(records, embeddings, settings=settings(max_entries=2))
    assert retriever.attestation.merged == 0
    assert retriever.attestation.allocated == 2
    assert retriever.prototype_count == 2


def test_cam_retriever_rejects_capacity_that_could_evict():
    records = [record("a", "s1"), record("b", "s2")]
    embeddings = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    with pytest.raises(ValueError, match="max_entries.*record count"):
        ExemplarCAMRetriever(records, embeddings, settings=settings(max_entries=1))
    with pytest.raises(ValueError, match="max_entries.*record count"):
        FAMRetriever(records, embeddings, settings=settings(max_entries=3))


def test_cam_retriever_reranks_authoritative_records_not_blended_values():
    records = [
        record("near", "scope-a"),
        record("far", "scope-a", serial=1),
    ]
    embeddings = {"near": [1.0, 0.0], "far": [0.8, 0.2]}

    exemplar = ExemplarCAMRetriever(records, embeddings, settings=settings(max_entries=2))
    fam = FAMRetriever(records, embeddings, settings=settings(max_entries=2))

    exemplar_found = exemplar.query([0.9, 0.1], k=1)
    fam_found = fam.query([0.9, 0.1], k=1)

    assert exemplar_found == fam_found
    assert fam_found[0].record_id == "near"
    assert fam_found[0].score > 0.9
    assert all(
        {field.name for field in fields(item)} == {"record_id", "score", "rank"}
        for item in fam_found
    )


def test_index_hash_is_deterministic_and_commits_to_record_order():
    ordered = [record("a", "scope"), record("b", "scope", serial=1)]
    reordered = list(reversed(ordered))
    embeddings = {"a": [1.0, 0.0], "b": [0.9, 0.1]}

    first = FAMRetriever(ordered, embeddings, settings=settings(max_entries=2))
    identical = FAMRetriever(ordered, embeddings, settings=settings(max_entries=2))
    reversed_build = FAMRetriever(reordered, embeddings, settings=settings(max_entries=2))

    assert first.attestation.index_sha256 == identical.attestation.index_sha256
    assert first.attestation.index_sha256 != reversed_build.attestation.index_sha256


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_entries": True}, "max_entries"),
        ({"prototype_k": 0}, "prototype_k"),
        ({"vigilance": float("nan")}, "vigilance"),
        ({"vigilance": 1.1}, "vigilance"),
        ({"hebb_lr": float("inf")}, "hebb_lr"),
        ({"key_lr": float("nan")}, "key_lr"),
        ({"ema_beta": float("inf")}, "ema_beta"),
        ({"inference_temp": 0.0}, "inference_temp"),
    ],
)
def test_cam_index_settings_reject_invalid_values(overrides, message):
    values = {
        "max_entries": 2,
        "prototype_k": 1,
        "vigilance": 0.85,
        "hebb_lr": 0.1,
        "key_lr": 0.05,
        "ema_beta": 0.05,
        "inference_temp": 0.05,
        "use_bfloat16": False,
        "adaptive_eviction": False,
        "use_lfu": True,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        CAMIndexSettings(**values)
