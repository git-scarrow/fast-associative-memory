"""PR-7 G3 pins — the quarantine recoverability re-injection probe
(PR7_QUARANTINE_PROMOTION_GATE.md §4 G3).

Pins the analysis-only recovery probe's three separable claims and the committed
validation manifest:

  1. PROVENANCE RECOVERABLE (holds) — the ledger is a complete, lossless,
     reversible record of the diverted writes: ledger count == baseline
     supersession capture (router AND fork_events) == absorbed-count
     decomposition == label total, on every (pair, seed).
  2. CAPTURE RESTORABLE WITHIN BOUND (holds) — reinstatement (= the none
     baseline) restores capture to 192/seed from the quarantine arm's 0, with
     broken/stale not exceeding the ungoverned baseline.
  3. HARM-FREE RECOVERY (FAILS on every geometry) — the diverted writes are the
     stale supersessions, so reinstating them re-introduces the stale_wrong (and
     on D/E the broken) quarantine drained: capture and harm are the SAME writes.
     Per gate §3 the ledger is provenance-only and G3 does NOT clear promotion.

Imports no torch; reads only committed JSON/CSV.
"""
import json
from pathlib import Path

import pytest

from benchmarks.pr7_recovery_probe import build_recovery_validation

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "results/issue_failure_mode_blindness/pr7/recovery_validation.json"


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)


@pytest.fixture
def val():
    return json.loads(VAL.read_text())


def test_boundaries(val):
    assert val["engine_or_retrieval_change"] is False
    assert val["imports_torch"] is False
    assert val["reads_only_committed_artifacts"] is True
    assert val["cell"] == "merge_path_stale"


def test_reconstruction_faithful_every_arm(val):
    for pair, pv in val["per_pair"].items():
        for seed, v in pv["by_seed"].items():
            r = v["reconstruction"]
            assert r["faithful"] is True, (pair, seed)
            # the ledger count agrees with three independent baseline measurands.
            assert (r["ledger_quarantined_count"]
                    == r["baseline_router_merge_suspect"]
                    == r["baseline_fork_events_supersession"]
                    == r["absorbed_count_decomposition"]
                    == r["ledger_label_total"] == 192), (pair, seed)


def test_capture_restored_within_bound(val):
    for pair, pv in val["per_pair"].items():
        for seed, v in pv["by_seed"].items():
            c = v["capture_restoration"]
            assert v["capture"]["quarantine_active"] == 0
            assert v["capture"]["reinstated"] == 192
            assert c["capture_restored"] is True
            assert c["broken_not_beyond_baseline"] is True
            assert c["stale_not_beyond_baseline"] is True


def test_harm_free_recovery_fails_every_geometry(val):
    # The substantive G3 finding: recovery re-introduces the harm quarantine
    # removed on all four geometries (stale couples everywhere; broken on D/E).
    assert val["summary"]["harm_free_recovery_geometries"] == []
    assert val["summary"]["capture_harm_coupled_geometries"] == [
        "pairA", "pairB", "pairD", "pairE"]
    assert val["summary"]["g3_harm_free_clears"] is False
    assert val["g3_verdict"] == "provenance_recoverable_not_harm_free"
    # stale_wrong is re-introduced even on the benign pairs (no broken there).
    for seed, v in val["per_pair"]["pairA"]["by_seed"].items():
        assert v["harm_free_recovery"]["delta_stale_reintroduced"] > 0
        assert v["harm_free_recovery"]["harm_free"] is False


def test_g3_does_not_clear_promotion(val):
    s = val["summary"]
    # provenance recoverability holds, but harm-free reinstatement does not —
    # so per gate §3 the ledger is provenance-only and G3 is not cleared.
    assert s["provenance_recoverable"] is True
    assert s["capture_restorable_within_bound"] is True
    assert s["g3_harm_free_clears"] is False


def test_committed_manifest_matches_fresh_build(val):
    assert val == json.loads(json.dumps(build_recovery_validation("merge_path_stale")))
