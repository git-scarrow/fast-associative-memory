from dataclasses import fields

import pytest
import torch

from harness.memory_eval.models import MemoryRecord
from harness.memory_eval.retrievers import ExactVectorRetriever, FAMRetriever


def record(record_id: str, scope: str, serial: int = 0) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id,
        scope=scope,
        value=record_id.upper(),
        content=f"The value is {record_id.upper()}.",
        serial=serial,
        event_time="2026-01-01T00:00:00Z",
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


def test_fam_condenses_each_scope_and_returns_only_source_record_ids():
    records = [
        record("old", "Ada\x1femployer", serial=1),
        record("new", "Ada\x1femployer", serial=2),
        record("other", "Grace\x1femployer", serial=1),
    ]
    embeddings = {
        "old": [1.0, 0.0, 0.0],
        "new": [0.8, 0.2, 0.0],
        "other": [0.0, 1.0, 0.0],
    }
    retriever = FAMRetriever(records, embeddings, prototype_k=1)

    assert retriever.prototype_count == 2
    found = retriever.query([1.0, 0.0, 0.0], k=2)

    assert [item.record_id for item in found] == ["old", "new"]
    assert all(
        {field.name for field in fields(item)} == {"record_id", "score", "rank"}
        for item in found
    )
    assert retriever.provenance_for_scope("Ada\x1femployer") == {"old", "new"}


def test_fam_reranks_authoritative_records_not_blended_values():
    records = [
        record("near", "scope-a"),
        record("far", "scope-a", serial=1),
    ]
    retriever = FAMRetriever(
        records,
        {"near": [1.0, 0.0], "far": [0.0, 1.0]},
        prototype_k=1,
    )

    found = retriever.query([0.9, 0.1], k=1)

    assert found[0].record_id == "near"
    assert found[0].score > 0.9
