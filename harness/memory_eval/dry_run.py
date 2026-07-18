"""Offline plumbing proof for the five-arm harness; never benchmark evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re

import torch

from harness.ctx.compile import load_policy

from . import ARM_NAMES, CONTEXT_BUDGET_TOKENS
from .ledger import MemoryLedger
from .manifest import seal_manifest
from .mechanism import mechanism_passes, score_mechanism
from .models import MemoryQuestion, MemoryRecord, fact_scope
from .preregistration import experiment_verdict, h1_passes, h2_passes, h3_passes
from .scoring import score_rows
from .sealed_run import build_plumbing_run


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
    # These embeddings are a synthetic construction, not an encoder proposal.
    # The evolving pair has cosine similarity above 0.85 but is non-identical,
    # forcing one real FAM merge with key drift. The old-vector query keeps the
    # stale record first while candidate_k=2 still recalls the current record.
    record_embeddings = {
        "01-old": torch.tensor([1.0, 0.0]),
        "02-new": torch.tensor([0.9, 0.1]),
        "03-clean": torch.tensor([0.0, 1.0]),
    }
    query_embeddings = {
        "q-evolving": torch.tensor([1.0, 0.0]),
        "q-clean": torch.tensor([0.0, 1.0]),
    }
    settings = {
        "candidate_k": 2,
        "cam_prototype_k": 2,
        "cam_max_entries": len(records),
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
    policy = load_policy()
    consumer = RuleConsumer()
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
        policy=policy,
        consumer_pin=consumer.pin_id,
    )
    runner = build_plumbing_run(
        manifest_path,
        records=records,
        questions=questions,
        record_embeddings=record_embeddings,
        query_embeddings=query_embeddings,
        retriever_settings=settings,
        policy=policy,
        consumer=consumer,
    )
    exemplar = runner.exemplar_retriever
    fam = runner.fam_retriever
    rows = runner.run(questions, query_embeddings)
    report = score_rows(
        rows,
        questions,
        ledger,
        expected_scoring_version=manifest["protocol"]["scoring_version"],
    )

    # Synthetic-fixture assertions only. These tiny-corpus numbers are chosen
    # visibly at this call site, are surfaced below, and are never sealed as a
    # scoring-run registration or proposed as production defaults.
    synthetic_fixture_assertions = {
        "prototype_reduction_margin": 0.3,
        "mechanism_recall_loss_bound": 0.0,
        "min_mechanism_recall_n": 2,
        "stale_reduction_margin": 1.0,
        "clean_answer_loss_bound": 0.0,
        "current_adoption_floor": 1.0,
        "min_stale_eligible_n": 1,
        "min_clean_n": 1,
    }
    mechanism = score_mechanism(
        rows=rows,
        questions=questions,
        ledger=ledger,
        exemplar_attestation=exemplar.attestation,
        fam_attestation=fam.attestation,
    )
    mechanism_active = (
        fam.attestation.merged > 0 and fam.attestation.key_drifted_merges > 0
    )
    mechanism_evaluable = (
        mechanism.recall_n
        >= synthetic_fixture_assertions["min_mechanism_recall_n"]
    )
    mechanism_ok = mechanism_evaluable and mechanism_passes(
        reduction_count=mechanism.prototype_reduction_count,
        record_n=mechanism.record_n,
        reduction_margin=synthetic_fixture_assertions[
            "prototype_reduction_margin"
        ],
        recall_loss_count=mechanism.recall_loss_count,
        recall_n=mechanism.recall_n,
        recall_loss_bound=synthetic_fixture_assertions[
            "mechanism_recall_loss_bound"
        ],
    )

    scores = {(row.query_id, row.arm): row for row in report.rows}
    stale_queries = tuple(
        question
        for question in questions
        if scores[(question.query_id, "fam_raw")].scope_class == "stale_eligible"
    )
    clean_queries = tuple(
        question
        for question in questions
        if scores[(question.query_id, "fam_raw")].scope_class == "clean"
    )
    raw_stale_adoptions = sum(
        scores[(question.query_id, "fam_raw")].stale_adoption
        for question in stale_queries
    )
    governed_stale_adoptions = sum(
        scores[(question.query_id, "fam_governed")].stale_adoption
        for question in stale_queries
    )
    clean_losses = sum(
        scores[(question.query_id, "fam_raw")].correct
        and not scores[(question.query_id, "fam_governed")].correct
        for question in clean_queries
    )
    governed_current_adoptions = sum(
        scores[(question.query_id, "fam_governed")].correct
        for question in stale_queries
    )
    application_evaluable = (
        len(stale_queries) >= synthetic_fixture_assertions["min_stale_eligible_n"]
        and len(clean_queries) >= synthetic_fixture_assertions["min_clean_n"]
    )
    application_h1 = application_evaluable and h1_passes(
        raw_adoptions=raw_stale_adoptions,
        governed_adoptions=governed_stale_adoptions,
        n_paired=len(stale_queries),
        margin=synthetic_fixture_assertions["stale_reduction_margin"],
    )
    application_h2 = application_evaluable and h2_passes(
        clean_losses=clean_losses,
        clean_n=len(clean_queries),
        bound=synthetic_fixture_assertions["clean_answer_loss_bound"],
    )
    application_h3 = application_evaluable and h3_passes(
        current_adoptions=governed_current_adoptions,
        stale_n=len(stale_queries),
        floor_rate=synthetic_fixture_assertions["current_adoption_floor"],
    )
    integrity_ok = (
        exemplar.attestation.written == len(records)
        and exemplar.attestation.prototype_count == len(records)
        and fam.attestation.written == len(records)
        and not any(
            (
                exemplar.attestation.dropped,
                exemplar.attestation.evicted,
                fam.attestation.dropped,
                fam.attestation.evicted,
            )
        )
    )
    synthetic_verdict = experiment_verdict(
        integrity_ok=integrity_ok,
        evaluable=mechanism_evaluable and application_evaluable,
        mechanism_ok=mechanism_ok,
        application_h1=application_h1,
        application_h2=application_h2,
        application_h3=application_h3,
        mechanism_active=mechanism_active,
    )
    return {
        "evidence_status": "synthetic/plumbing",
        "admissible": False,
        "benchmark_evidence": False,
        "arms": list(ARM_NAMES),
        "context_budget_tokens": CONTEXT_BUDGET_TOKENS,
        "question_count": len(questions),
        "row_count": len(rows),
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest": {
            "version": manifest["manifest_version"],
            "evidence_class": manifest["protocol"]["evidence_class"],
            "registration": manifest["protocol"]["registration"],
            "confirmatory_index_attestations_sealed": (
                "index_attestations" in manifest["protocol"]
            ),
        },
        "rebuilt_index_attestations": {
            "exemplar": asdict(exemplar.attestation),
            "fam": asdict(fam.attestation),
        },
        "synthetic_fixture_assertions": synthetic_fixture_assertions,
        "outcome": {
            "evidence_status": "synthetic/plumbing",
            "admissible": False,
            "benchmark_evidence": False,
            "mechanism": {
                "passed": mechanism_ok,
                "active": mechanism_active,
                **asdict(mechanism),
            },
            "application": {
                "h1_stale_reduction": application_h1,
                "h2_clean_answer_loss": application_h2,
                "h3_current_adoption": application_h3,
                "passed": application_h1 and application_h2 and application_h3,
            },
            "synthetic_verdict": synthetic_verdict,
        },
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
