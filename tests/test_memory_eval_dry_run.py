from harness.memory_eval import ARM_NAMES, CONTEXT_BUDGET_TOKENS
from harness.memory_eval.dry_run import run_dry_run
from harness.memory_eval.scoring import SCORING_VERSION


def test_dry_run_seals_executes_and_scores_all_five_arms(tmp_path):
    summary = run_dry_run(tmp_path)

    assert summary["evidence_status"] == "plumbing-only; not benchmark evidence"
    assert summary["arms"] == list(ARM_NAMES)
    assert summary["context_budget_tokens"] == CONTEXT_BUDGET_TOKENS
    assert summary["question_count"] == 2
    assert summary["row_count"] == 10
    assert summary["fam_prototype_count"] == 2
    assert len(summary["manifest_sha256"]) == 64
    assert (tmp_path / "dry_run_manifest.json").exists()
    assert summary["scoring_version"] == SCORING_VERSION
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
    assert summary["metrics"]["vector_raw"]["stale_adoption_rate"] == {
        "value": 1.0,
        "n": 1,
    }
    assert summary["metrics"]["vector_governed"]["stale_adoption_rate"] == {
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
    assert summary["metrics"]["vector_raw"]["current_adoption_rate"] == {
        "value": 0.0,
        "n": 1,
    }
    assert summary["metrics"]["vector_governed"]["current_adoption_rate"] == {
        "value": 1.0,
        "n": 1,
    }
    assert summary["metrics"]["fam_governed"]["current_adoption_rate"] == {
        "value": 1.0,
        "n": 1,
    }
    # No contested scopes in the synthetic corpus: no data, not 0.0.
    assert summary["metrics"]["vector_governed"]["fork_adoption_rate"] == {
        "value": None,
        "n": 0,
    }
    assert summary["clean_answer_loss"] == {
        "vector_governed_vs_raw": {"value": 0.0, "n": 1},
        "fam_governed_vs_raw": {"value": 0.0, "n": 1},
    }
    assert summary["stale_eligible_loss"] == {
        "vector_governed_vs_raw": {"value": 0.0, "n": 1},
        "fam_governed_vs_raw": {"value": 0.0, "n": 1},
    }
