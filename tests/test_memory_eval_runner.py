import json

from harness.memory_eval import ARM_NAMES, CONTEXT_BUDGET_TOKENS
from harness.memory_eval.ledger import MemoryLedger
from harness.memory_eval.models import MemoryQuestion, MemoryRecord, RetrievedCandidate
from harness.memory_eval.runner import FiveArmRunner


class CountingRetriever:
    def __init__(self, ids):
        self.ids = ids
        self.calls = 0

    def query(self, query_embedding, k):
        self.calls += 1
        return tuple(
            RetrievedCandidate(record_id, 1.0 - rank / 10, rank)
            for rank, record_id in enumerate(self.ids[:k], start=1)
        )


class FakeConsumer:
    pin_id = "test-consumer"

    def __init__(self, outputs=None):
        self.outputs = iter(outputs) if outputs is not None else None
        self.prompts = []

    @staticmethod
    def count_tokens(text):
        return len(text.split())

    def generate(self, prompt, max_new_tokens=256):
        self.prompts.append(prompt)
        if self.outputs is not None:
            return next(self.outputs)
        return json.dumps({"answer": "B", "hedged": False})


class TickClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        self.value += 0.001
        return self.value


def make_ledger():
    return MemoryLedger(
        [
            MemoryRecord(
                record_id="old",
                scope="Ada\x1femployer",
                value="A",
                content="Ada's employer was A.",
                serial=1,
                event_time="2026-01-01T00:00:00Z",
            ),
            MemoryRecord(
                record_id="new",
                scope="Ada\x1femployer",
                value="B",
                content="Ada's employer is B.",
                serial=2,
                event_time="2026-02-01T00:00:00Z",
            ),
        ]
    )


def test_runner_executes_five_arms_with_shared_paired_candidates():
    vector = CountingRetriever(["old", "new"])
    fam = CountingRetriever(["new", "old"])
    consumer = FakeConsumer()
    runner = FiveArmRunner(
        ledger=make_ledger(),
        vector_retriever=vector,
        fam_retriever=fam,
        consumer=consumer,
        candidate_k=2,
        clock=TickClock(),
    )
    question = MemoryQuestion(
        query_id="q1",
        text="Who employs Ada?",
        scope="Ada\x1femployer",
        answer="B",
    )

    rows = runner.run([question], {"q1": [1.0, 0.0]})

    assert tuple(row.arm for row in rows) == ARM_NAMES
    assert vector.calls == fam.calls == 1
    assert len(consumer.prompts) == 5
    by_arm = {row.arm: row for row in rows}
    assert by_arm["vector_raw"].candidate_ids == ("old", "new")
    assert by_arm["vector_governed"].candidate_ids == ("old", "new")
    assert by_arm["fam_raw"].candidate_ids == ("new", "old")
    assert by_arm["fam_governed"].candidate_ids == ("new", "old")
    assert by_arm["no_memory"].candidate_ids == ()
    assert by_arm["no_memory"].rendered_item_ids == ()
    assert all(row.budget == CONTEXT_BUDGET_TOKENS for row in rows)
    assert all(row.block_tokens <= CONTEXT_BUDGET_TOKENS for row in rows)
    assert all(row.prompt_tokens >= row.block_tokens for row in rows)
    assert all(row.consumer_ms > 0 for row in rows)
    assert by_arm["vector_raw"].retrieval_ms == by_arm["vector_governed"].retrieval_ms
    assert by_arm["fam_raw"].retrieval_ms == by_arm["fam_governed"].retrieval_ms
    assert by_arm["vector_governed"].audit_rows


def test_runner_records_abstention_and_malformed_output():
    outputs = [
        '{"answer": "ABSTAIN", "hedged": true}',
        "not json",
        '{"answer": "B", "hedged": false}',
        '{"answer": "B", "hedged": false}',
        '{"answer": "B", "hedged": false}',
    ]
    consumer = FakeConsumer(outputs)
    runner = FiveArmRunner(
        ledger=make_ledger(),
        vector_retriever=CountingRetriever(["new"]),
        fam_retriever=CountingRetriever(["new"]),
        consumer=consumer,
        candidate_k=1,
        clock=TickClock(),
    )
    question = MemoryQuestion("q1", "Who employs Ada?", "Ada\x1femployer", "B")

    rows = runner.run([question], {"q1": [1.0, 0.0]})

    assert rows[0].observation.status == "abstain"
    assert rows[1].observation.status == "malformed"
    assert rows[1].parse_reason == "no_balanced_json_object"
    assert rows[2].observation.answer == "B"


def test_runner_rejects_missing_query_embeddings():
    runner = FiveArmRunner(
        ledger=make_ledger(),
        vector_retriever=CountingRetriever(["new"]),
        fam_retriever=CountingRetriever(["new"]),
        consumer=FakeConsumer(),
    )
    question = MemoryQuestion("q1", "Who employs Ada?", "Ada\x1femployer", "B")

    try:
        runner.run([question], {})
    except ValueError as exc:
        assert "missing query embedding" in str(exc)
    else:
        raise AssertionError("missing query embedding was accepted")
