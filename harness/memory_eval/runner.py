"""Paired execution of the fixed-budget five-arm memory experiment."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from typing import Any, Protocol

from harness.ctx.compile import compile as compile_context
from harness.ctx.compile import load_policy
from harness.ctx.output_contract import MAX_NEW_TOKENS, parse_consumer_output

from . import ARM_NAMES, CONTEXT_BUDGET_TOKENS
from .context import context_items_for_candidates, render_raw
from .ledger import MemoryLedger
from .models import (
    AnswerObservation,
    MemoryQuestion,
    RetrievedCandidate,
)
from .retrievers import TensorLike


class Retriever(Protocol):
    def query(
        self, query_embedding: TensorLike, k: int
    ) -> tuple[RetrievedCandidate, ...]: ...


class Consumer(Protocol):
    pin_id: str

    def count_tokens(self, text: str) -> int: ...

    def generate(self, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str: ...


@dataclass(frozen=True, slots=True)
class ExperimentRow:
    query_id: str
    arm: str
    budget: int
    candidate_ids: tuple[str, ...]
    rendered_item_ids: tuple[str, ...]
    block_tokens: int
    prompt_tokens: int
    retrieval_ms: float
    compilation_ms: float
    consumer_ms: float
    observation: AnswerObservation
    hedged: bool | None
    extra_prose: bool | None
    parse_reason: str | None
    consumer_pin_id: str | None
    block_sha256: str
    prompt_sha256: str
    raw_output: str
    audit_rows: tuple[Mapping[str, Any], ...]
    audit_anomalies: tuple[Mapping[str, Any], ...]


class FiveArmRunner:
    def __init__(
        self,
        *,
        ledger: MemoryLedger,
        vector_retriever: Retriever,
        fam_retriever: Retriever,
        consumer: Consumer,
        candidate_k: int = 10,
        clock: Callable[[], float] = perf_counter,
        policy: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(candidate_k, int) or isinstance(candidate_k, bool) or candidate_k < 1:
            raise ValueError("candidate_k must be positive")
        self.ledger = ledger
        self.vector_retriever = vector_retriever
        self.fam_retriever = fam_retriever
        self.consumer = consumer
        self.candidate_k = candidate_k
        self.clock = clock
        self.policy = dict(policy) if policy is not None else load_policy()

    def run(
        self,
        questions: Sequence[MemoryQuestion],
        query_embeddings: Mapping[str, TensorLike],
    ) -> tuple[ExperimentRow, ...]:
        rows: list[ExperimentRow] = []
        seen: set[str] = set()
        for question in questions:
            if question.query_id in seen:
                raise ValueError(f"duplicate query_id: {question.query_id}")
            seen.add(question.query_id)
            if question.query_id not in query_embeddings:
                raise ValueError(f"missing query embedding: {question.query_id}")
            rows.extend(self._run_question(question, query_embeddings[question.query_id]))
        return tuple(rows)

    def _run_question(
        self, question: MemoryQuestion, query_embedding: TensorLike
    ) -> tuple[ExperimentRow, ...]:
        vector_candidates, vector_ms = self._retrieve(
            self.vector_retriever, query_embedding
        )
        fam_candidates, fam_ms = self._retrieve(self.fam_retriever, query_embedding)
        candidates = {
            "vector": (vector_candidates, vector_ms),
            "fam": (fam_candidates, fam_ms),
        }

        result: list[ExperimentRow] = []
        for arm in ARM_NAMES:
            if arm == "no_memory":
                candidate_ids: tuple[str, ...] = ()
                retrieval_ms = 0.0
                block = ""
                rendered_ids: tuple[str, ...] = ()
                audit_rows: tuple[Mapping[str, Any], ...] = ()
                anomalies: tuple[Mapping[str, Any], ...] = ()
                compilation_ms = 0.0
            else:
                family, rendering = arm.split("_", maxsplit=1)
                retrieved, retrieval_ms = candidates[family]
                candidate_ids = tuple(item.record_id for item in retrieved)
                items = context_items_for_candidates(self.ledger, candidate_ids)
                started = self.clock()
                if rendering == "raw":
                    raw = render_raw(
                        items,
                        budget=CONTEXT_BUDGET_TOKENS,
                        count_tokens=self.consumer.count_tokens,
                    )
                    block = raw.block
                    rendered_ids = raw.rendered_item_ids
                    audit_rows = ()
                    anomalies = ()
                else:
                    block, audit = compile_context(
                        items,
                        self.policy,
                        CONTEXT_BUDGET_TOKENS,
                        {"turn_index": 1, "prior_rendered": {}},
                        self.consumer.count_tokens,
                        query_id=question.query_id,
                        authorization="five-arm-memory-eval",
                    )
                    audit_rows = tuple(audit["rows"])
                    anomalies = tuple(audit["anomalies"])
                    rendered_ids = _rendered_item_ids(audit_rows)
                compilation_ms = _milliseconds(started, self.clock())

            block_tokens = self.consumer.count_tokens(block)
            if block_tokens > CONTEXT_BUDGET_TOKENS:
                raise RuntimeError(
                    f"{arm} context is {block_tokens} tokens, over fixed budget "
                    f"{CONTEXT_BUDGET_TOKENS}"
                )
            prompt = build_prompt(question.text, block if arm != "no_memory" else None)
            consumer_started = self.clock()
            raw_output = self.consumer.generate(
                prompt, max_new_tokens=MAX_NEW_TOKENS
            )
            consumer_ms = _milliseconds(consumer_started, self.clock())
            parsed = parse_consumer_output(raw_output)
            observation = _observation(parsed)
            result.append(
                ExperimentRow(
                    query_id=question.query_id,
                    arm=arm,
                    budget=CONTEXT_BUDGET_TOKENS,
                    candidate_ids=candidate_ids,
                    rendered_item_ids=rendered_ids,
                    block_tokens=block_tokens,
                    prompt_tokens=self.consumer.count_tokens(prompt),
                    retrieval_ms=retrieval_ms,
                    compilation_ms=compilation_ms,
                    consumer_ms=consumer_ms,
                    observation=observation,
                    hedged=parsed.get("hedged"),
                    extra_prose=parsed.get("extra_prose"),
                    parse_reason=parsed.get("reason"),
                    consumer_pin_id=getattr(self.consumer, "pin_id", None),
                    block_sha256=_sha(block),
                    prompt_sha256=_sha(prompt),
                    raw_output=raw_output,
                    audit_rows=audit_rows,
                    audit_anomalies=anomalies,
                )
            )
        return tuple(result)

    def _retrieve(
        self, retriever: Retriever, query_embedding: TensorLike
    ) -> tuple[tuple[RetrievedCandidate, ...], float]:
        started = self.clock()
        candidates = tuple(retriever.query(query_embedding, self.candidate_k))
        elapsed = _milliseconds(started, self.clock())
        ids = [candidate.record_id for candidate in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("retriever returned duplicate candidate IDs")
        for record_id in ids:
            self.ledger.get(record_id)
        return candidates, elapsed


def build_prompt(question: str, context_block: str | None) -> str:
    context = context_block if context_block else "(no memory context)"
    return (
        "Answer the question using the supplied memory context. If the context "
        "does not support an answer, answer exactly ABSTAIN. Return only one JSON "
        "object with fields answer (string) and hedged (boolean).\n\n"
        f"Question:\n{question}\n\nMemory context:\n{context}"
    )


def _observation(parsed: Mapping[str, Any]) -> AnswerObservation:
    if parsed["status"] != "ok":
        return AnswerObservation(status="malformed", answer=None)
    answer = str(parsed["answer"])
    if answer.strip().casefold() == "abstain":
        return AnswerObservation(status="abstain", answer=None)
    return AnswerObservation(status="answer", answer=answer)


def _rendered_item_ids(
    audit_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    rendered = [
        row
        for row in audit_rows
        if row.get("budget_decision") in {"rendered", "summarized"}
    ]
    rendered.sort(key=lambda row: (row.get("render_index"), row["item_id"]))
    return tuple(str(row["item_id"]) for row in rendered)


def _milliseconds(start: float, end: float) -> float:
    return max(0.0, (end - start) * 1000.0)


def _sha(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()
