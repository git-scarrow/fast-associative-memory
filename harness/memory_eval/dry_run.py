"""Offline plumbing proof for the five-arm harness; never benchmark evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re

import torch

from . import ARM_NAMES, CONTEXT_BUDGET_TOKENS
from .ledger import MemoryLedger
from .manifest import seal_manifest, verify_manifest
from .models import MemoryQuestion, MemoryRecord, fact_scope
from .retrievers import ExactVectorRetriever, FAMRetriever
from .runner import FiveArmRunner
from .scoring import score_rows


class HashScopeEncoder:
    """Deterministic test encoder that intentionally carries no semantics."""

    def __init__(self, dimension: int = 32) -> None:
        self.dimension = dimension

    def encode(self, scope: str) -> torch.Tensor:
        digest = sha256(scope.encode("utf-8")).digest()
        values = [digest[index % len(digest)] for index in range(self.dimension)]
        return torch.tensor(
            [(value - 127.5) / 127.5 for value in values], dtype=torch.float32
        )


class RuleConsumer:
    """First-matching-fact parser used only to exercise the full pipeline."""

    pin_id = "memory-eval-rule-consumer-v1"
    _question = re.compile(r"What is FACT\[([^|]+)\|([^\]]+)\]\?")
    _fact = re.compile(r"FACT\[([^|]+)\|([^\]]+)\]=([A-Za-z0-9_-]+)")

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text.split())

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        target = self._question.search(prompt)
        answer = "ABSTAIN"
        if target is not None:
            target_key = target.group(1), target.group(2)
            for entity, relation, value in self._fact.findall(prompt):
                if (entity, relation) == target_key:
                    answer = value
                    break
        return json.dumps({"answer": answer, "hedged": answer == "ABSTAIN"})


def run_dry_run(output_dir: str | Path) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    records, questions = _synthetic_corpus()
    ledger = MemoryLedger(records)
    encoder = HashScopeEncoder()
    record_embeddings = {
        record.record_id: encoder.encode(record.scope) for record in records
    }
    query_embeddings = {
        question.query_id: encoder.encode(question.scope) for question in questions
    }
    settings = {"candidate_k": 2, "fam_prototype_k": 1, "fam_max_entries": 2}
    manifest_path = output_path / "dry_run_manifest.json"
    # evidence_class="plumbing": this seal carries no registration and is never
    # admissible. A scoring-run seal would be refused here — correctly, since
    # nothing about this corpus or this rule consumer is registered.
    manifest = seal_manifest(
        manifest_path,
        records,
        questions,
        record_embeddings,
        query_embeddings,
        settings,
        evidence_class="plumbing",
    )
    verify_manifest(
        manifest_path,
        records,
        questions,
        record_embeddings,
        query_embeddings,
        settings,
        evidence_class="plumbing",
    )

    vector = ExactVectorRetriever(records, record_embeddings)
    fam = FAMRetriever(
        records,
        record_embeddings,
        prototype_k=settings["fam_prototype_k"],
        max_entries=settings["fam_max_entries"],
    )
    rows = FiveArmRunner(
        ledger=ledger,
        vector_retriever=vector,
        fam_retriever=fam,
        consumer=RuleConsumer(),
        candidate_k=settings["candidate_k"],
    ).run(questions, query_embeddings)
    report = score_rows(
        rows,
        questions,
        ledger,
        expected_scoring_version=manifest["protocol"]["scoring_version"],
    )
    return {
        "evidence_status": "plumbing-only; not benchmark evidence",
        "arms": list(ARM_NAMES),
        "context_budget_tokens": CONTEXT_BUDGET_TOKENS,
        "question_count": len(questions),
        "row_count": len(rows),
        "fam_prototype_count": fam.prototype_count,
        "manifest_sha256": manifest["manifest_sha256"],
        "scoring_version": report.scoring_version,
        "corpus": asdict(report.corpus),
        "metrics": {
            arm: asdict(report.by_arm[arm]) for arm in ARM_NAMES
        },
        "clean_answer_loss": {
            key: asdict(rate) for key, rate in report.clean_answer_loss.items()
        },
        "stale_eligible_loss": {
            key: asdict(rate) for key, rate in report.stale_eligible_loss.items()
        },
    }


def _synthetic_corpus() -> tuple[tuple[MemoryRecord, ...], tuple[MemoryQuestion, ...]]:
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
    return records, questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_dry_run(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
