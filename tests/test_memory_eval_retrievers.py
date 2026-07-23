from dataclasses import fields

import pytest
import torch

from associative_core import ContinuousCAM
from harness.memory_eval.models import MemoryRecord
from harness.memory_eval.retrievers import (
    CAMIndexSettings,
    ExactVectorRetriever,
    ExemplarCAMRetriever,
    FAMRetriever,
    _HarnessWriteCAM,
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
    slot = retriever.cam.occupied.nonzero(as_tuple=True)[0].item()
    # Allocation seeds hit_counts=1; this merge increments it to 2 before EMA.
    adaptive_key_alpha = 0.05 / (1.0 + 0.05 * 2)
    expected_key = torch.tensor([1.0, 0.0]) + adaptive_key_alpha * (
        torch.tensor([0.9, 0.1]) - torch.tensor([1.0, 0.0])
    )
    assert torch.allclose(retriever.cam.keys[slot], expected_key)


def test_closer_cross_scope_prototype_cannot_shield_a_valid_same_scope_merge():
    records = [
        record("a-seed", "scope-a", serial=1),
        record("b-seed", "scope-b", serial=1),
        record("a-update", "scope-a", serial=2),
    ]
    embeddings = {
        "a-seed": [1.0, 0.0],
        "b-seed": [0.95, 0.3122499],
        # Identical to the cross-scope slot, but still cosine 0.95 to scope-a.
        "a-update": [0.95, 0.3122499],
    }

    retriever = FAMRetriever(records, embeddings, settings=settings(max_entries=3))

    assert retriever.attestation.merged == 1
    assert retriever.prototype_count == 2
    assert retriever.provenance_for_scope("scope-a") == {"a-seed", "a-update"}
    assert retriever.provenance_for_scope("scope-b") == {"b-seed"}


def test_identical_different_scope_collision_allocates_separate_prototypes():
    records = [record("a", "scope-a"), record("b", "scope-b")]
    retriever = FAMRetriever(
        records,
        {"a": [1.0, 0.0], "b": [1.0, 0.0]},
        settings=settings(max_entries=2),
    )

    assert retriever.attestation.merged == 0
    assert retriever.prototype_count == 2
    occupied = retriever.cam.occupied.nonzero(as_tuple=True)[0].tolist()
    assert {frozenset(retriever.cam.records_for(slot)) for slot in occupied} == {
        frozenset({"a"}),
        frozenset({"b"}),
    }


def test_below_vigilance_same_scope_allocates_second_fam_prototype():
    records = [record("a", "scope"), record("b", "scope", serial=1)]
    embeddings = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    retriever = FAMRetriever(records, embeddings, settings=settings(max_entries=2))
    assert retriever.attestation.merged == 0
    assert retriever.prototype_count == 2


def test_allocate_only_identical_same_scope_rows_remain_immutable_exemplars():
    records = [record("a", "scope"), record("b", "scope", serial=1)]
    embeddings = {"a": [1.0, 0.0], "b": [1.0, 0.0]}
    retriever = ExemplarCAMRetriever(records, embeddings, settings=settings(max_entries=2))
    assert retriever.attestation.merged == 0
    assert retriever.attestation.allocated == 2
    assert retriever.prototype_count == 2
    assert retriever.cam.immutable_keys is True
    occupied = retriever.cam.occupied.nonzero(as_tuple=True)[0]
    assert torch.equal(
        retriever.cam.keys[occupied],
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    )


def test_harness_adapter_rejects_unknown_write_mode_before_mutation():
    cam = _HarnessWriteCAM(
        key_dim=2,
        value_dim=1,
        max_entries=1,
        track_provenance=True,
    )
    before_occupied = cam.occupied.clone()
    before_stats = cam.last_write_stats.copy()

    with pytest.raises(ValueError, match="write_mode"):
        cam.ingest_one(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0]]),
            record_id="a",
            write_mode="merge-ish",
        )

    assert torch.equal(cam.occupied, before_occupied)
    assert cam.last_write_stats == before_stats


@pytest.mark.parametrize(
    "corrupt",
    [
        {"written": 2, "merged": 2, "allocated": 0, "dropped": 0},
        {"written": True, "merged": 0, "allocated": 1, "dropped": 0},
        {"written": 1, "merged": 0.0, "allocated": 1, "dropped": 0},
        {"written": 1, "merged": -1, "allocated": 1, "dropped": 1},
        {
            "written": 1,
            "merged": 0,
            "allocated": 1,
            "dropped": 0,
            "evicted": 0,
        },
        {"written": 1, "merged": 0, "allocated": 1},
    ],
    ids=(
        "balanced-nonsingleton",
        "boolean",
        "float",
        "negative",
        "extra-key",
        "missing-key",
    ),
)
def test_cam_retriever_rejects_malformed_singleton_write_accounting(
    monkeypatch, corrupt
):
    original_ingest = _HarnessWriteCAM.ingest_one

    def corrupt_stats(self, *args, **kwargs):
        original_ingest(self, *args, **kwargs)
        self.last_write_stats = corrupt

    monkeypatch.setattr(_HarnessWriteCAM, "ingest_one", corrupt_stats)

    with pytest.raises(ValueError, match="write accounting"):
        ExemplarCAMRetriever(
            [record("a", "scope")],
            {"a": [1.0, 0.0]},
            settings=settings(max_entries=1),
        )


def test_cam_retriever_rejects_invalid_negative_local_eviction(monkeypatch):
    original_ingest = _HarnessWriteCAM.ingest_one

    def report_merge_after_allocation(self, *args, **kwargs):
        original_ingest(self, *args, **kwargs)
        self.last_write_stats = {
            "written": 1,
            "merged": 1,
            "allocated": 0,
            "dropped": 0,
        }

    monkeypatch.setattr(_HarnessWriteCAM, "ingest_one", report_merge_after_allocation)

    with pytest.raises(ValueError, match="local eviction"):
        ExemplarCAMRetriever(
            [record("a", "scope")],
            {"a": [1.0, 0.0]},
            settings=settings(max_entries=1),
        )


def test_cam_retriever_rejects_nonzero_local_eviction_immediately(monkeypatch):
    original_ingest = _HarnessWriteCAM.ingest_one
    calls = 0

    def hide_allocation(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        original_ingest(self, *args, **kwargs)
        if calls == 1:
            self.occupied.fill_(False)

    monkeypatch.setattr(_HarnessWriteCAM, "ingest_one", hide_allocation)

    with pytest.raises(ValueError, match="evicted=1"):
        ExemplarCAMRetriever(
            [record("a", "scope"), record("b", "scope", serial=1)],
            {"a": [1.0, 0.0], "b": [0.0, 1.0]},
            settings=settings(max_entries=2),
        )
    assert calls == 1


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

    assert type(exemplar.cam) is type(fam.cam) is _HarnessWriteCAM
    assert exemplar.cam.forward.__func__ is ContinuousCAM.forward
    assert fam.cam.forward.__func__ is ContinuousCAM.forward

    fam_found = fam.query([0.9, 0.1], k=2)

    # The served identifiers are AUTHORITATIVE ledger record IDs, never blended
    # CAM values. "near" and "far" condense into one drifted prototype; the FAM
    # read path recovers its provenance (both authoritative records) rather than
    # emitting the prototype's blended vector. It deliberately does NOT re-score
    # them by original-embedding proximity to the query — that flat rerank is
    # what previously collapsed fam onto the exact-vector arm — so within the
    # merged prototype the co-equal members appear in deterministic record_id
    # order, and the served score is the prototype's vote weight.
    served_ids = {c.record_id for c in fam_found}
    assert served_ids <= {r.record_id for r in records}
    assert served_ids == {"far", "near"}  # both provenance records recovered
    assert fam_found[0].record_id == "far"  # deterministic record_id order
    assert fam_found[0].score > 0.9
    assert all(
        {field.name for field in fields(item)} == {"record_id", "score", "rank"}
        for item in fam_found
    )


def test_fam_read_path_is_load_bearing_not_the_exact_vector_fallback():
    """Regression for the review's core blocker: fam must NOT collapse onto the
    exact-vector arm. When same-scope records condense (keys drift), the fam
    read path (vote over prototype keys) must diverge from exact cosine over the
    original embeddings. Pre-fix this diverged on 0/N queries."""
    # Three scopes with prototype_k=2 so the vote SELECTS a subset of prototypes
    # by their (drifted) keys — that selection is where condensation becomes
    # observable in retrieval. Same-scope cos ~0.97 > vigilance so keys drift.
    scopes = {
        "scope-a": [("a-old", [1.0, 0.0, 0.0, 0.0]), ("a-new", [0.97, 0.243, 0.0, 0.0])],
        "scope-b": [("b-old", [0.0, 1.0, 0.0, 0.0]), ("b-new", [0.0, 0.97, 0.243, 0.0])],
        "scope-c": [("c-old", [0.0, 0.0, 1.0, 0.0]), ("c-new", [0.243, 0.0, 0.97, 0.0])],
    }
    recs, emb = [], {}
    for scope, items in scopes.items():
        for rid, vec in items:
            recs.append(record(rid, scope, serial=0 if "old" in rid else 1))
            emb[rid] = vec
    fam = FAMRetriever(recs, emb, settings=settings(max_entries=6, prototype_k=2))
    exact = ExactVectorRetriever(recs, emb)

    # Condensation actually fired (same-scope cos > vigilance): keys drifted.
    assert fam.attestation.merged > 0
    assert fam.attestation.key_drifted_merges > 0
    assert fam.attestation.prototype_count < fam.attestation.written

    divergent = 0
    for scope, items in scopes.items():
        query = items[1][1]  # near the "new" record
        fam_ids = tuple(c.record_id for c in fam.query(query, k=3))
        exact_ids = tuple(c.record_id for c in exact.query(query, k=3))
        if fam_ids != exact_ids:
            divergent += 1
    assert divergent > 0, "fam collapsed onto the exact-vector fallback"


def test_index_hash_is_deterministic_and_commits_to_record_order():
    ordered = [record("a", "scope"), record("b", "scope", serial=1)]
    reordered = list(reversed(ordered))
    embeddings = {"a": [1.0, 0.0], "b": [0.9, 0.1]}

    first = FAMRetriever(ordered, embeddings, settings=settings(max_entries=2))
    identical = FAMRetriever(ordered, embeddings, settings=settings(max_entries=2))
    reversed_build = FAMRetriever(reordered, embeddings, settings=settings(max_entries=2))

    assert first.attestation == identical.attestation
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
