from harness.memory_eval import ARM_NAMES, CONTEXT_BUDGET_TOKENS
from harness.memory_eval.dry_run import run_dry_run
from harness.memory_eval.manifest import MANIFEST_VERSION
from harness.memory_eval.preregistration import GO
from harness.memory_eval.scoring import SCORING_VERSION


def test_dry_run_seals_executes_and_scores_all_five_arms(tmp_path):
    summary = run_dry_run(tmp_path)

    assert ARM_NAMES == (
        "no_memory",
        "exemplar_raw",
        "exemplar_governed",
        "fam_raw",
        "fam_governed",
    )
    assert summary["evidence_status"] == "synthetic/plumbing"
    assert summary["admissible"] is False
    assert summary["benchmark_evidence"] is False
    assert summary["arms"] == list(ARM_NAMES)
    assert summary["context_budget_tokens"] == CONTEXT_BUDGET_TOKENS
    assert summary["question_count"] == 2
    assert summary["row_count"] == 10
    assert len(summary["manifest_sha256"]) == 64
    assert (tmp_path / "dry_run_manifest.json").exists()
    assert summary["manifest"] == {
        "version": MANIFEST_VERSION,
        "evidence_class": "plumbing",
        "registration": None,
        "confirmatory_index_attestations_sealed": False,
    }
    assert summary["scoring_version"] == SCORING_VERSION
    attestations = summary["rebuilt_index_attestations"]
    assert set(attestations) == {"exemplar", "fam"}
    assert attestations["exemplar"]["mode"] == "allocate-only"
    assert attestations["exemplar"]["prototype_count"] == 3
    assert attestations["exemplar"]["allocated"] == 3
    assert attestations["fam"]["mode"] == "condense"
    assert attestations["fam"]["prototype_count"] == 2
    assert attestations["fam"]["merged"] == 1
    assert attestations["fam"]["key_drifted_merges"] == 1
    assert all(len(value["index_sha256"]) == 64 for value in attestations.values())
    assert summary["synthetic_fixture_assertions"] == {
        "prototype_reduction_margin": 0.3,
        "mechanism_recall_loss_bound": 0.0,
        "min_mechanism_recall_n": 2,
        "stale_reduction_margin": 1.0,
        "clean_answer_loss_bound": 0.0,
        "current_adoption_floor": 1.0,
        "min_stale_eligible_n": 1,
        "min_clean_n": 1,
    }
    outcome = summary["outcome"]
    assert outcome["evidence_status"] == "synthetic/plumbing"
    assert outcome["admissible"] is False
    assert outcome["benchmark_evidence"] is False
    assert outcome["mechanism"] == {
        "passed": True,
        "active": True,
        "recall_n": 2,
        "exemplar_recall_count": 2,
        "fam_recall_count": 2,
        "recall_loss_count": 0,
        "record_n": 3,
        "exemplar_prototype_count": 3,
        "fam_prototype_count": 2,
        "prototype_reduction_count": 1,
    }
    assert outcome["application"] == {
        "h1_stale_reduction": True,
        "h2_clean_answer_loss": True,
        "h3_current_adoption": True,
        "passed": True,
    }
    # A synthetic GO is allowed only inside this explicitly inadmissible block.
    assert outcome["synthetic_verdict"] == GO
    assert summary["corpus"] == {
        "questions": 2,
        "clean_questions": 1,
        "stale_eligible_questions": 1,
        "contested_questions": 0,
        "ledger_scopes": 2,
        "contested_ledger_scopes": 0,
    }
    # The specific values below are plumbing-only artifacts of the synthetic
    # two-question corpus and the rule consumer; they are not benchmark
    # evidence and carry no claim about governance.
    assert summary["metrics"]["exemplar_raw"]["stale_adoption_rate"] == {
        "value": 1.0,
        "n": 1,
    }
    assert summary["metrics"]["exemplar_governed"]["stale_adoption_rate"] == {
        "value": 0.0,
        "n": 1,
    }
    assert summary["metrics"]["fam_raw"]["stale_adoption_rate"] == {
        "value": 1.0,
        "n": 1,
    }
    assert summary["metrics"]["fam_governed"]["stale_adoption_rate"] == {
        "value": 0.0,
        "n": 1,
    }
    assert summary["metrics"]["exemplar_raw"]["current_adoption_rate"] == {
        "value": 0.0,
        "n": 1,
    }
    assert summary["metrics"]["exemplar_governed"]["current_adoption_rate"] == {
        "value": 1.0,
        "n": 1,
    }
    assert summary["metrics"]["fam_governed"]["current_adoption_rate"] == {
        "value": 1.0,
        "n": 1,
    }
    # No contested scopes in the synthetic corpus: no data, not 0.0.
    assert summary["metrics"]["exemplar_governed"]["fork_adoption_rate"] == {
        "value": None,
        "n": 0,
    }
    assert summary["clean_answer_loss"] == {
        "exemplar_governed_vs_raw": {"value": 0.0, "n": 1},
        "fam_governed_vs_raw": {"value": 0.0, "n": 1},
    }
    assert summary["stale_eligible_loss"] == {
        "exemplar_governed_vs_raw": {"value": 0.0, "n": 1},
        "fam_governed_vs_raw": {"value": 0.0, "n": 1},
    }
