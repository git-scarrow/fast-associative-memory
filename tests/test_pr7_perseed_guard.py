"""PR-7 prepared scorer step — G4 per-seed collateral guard + clean_control pairC
(PR7_QUARANTINE_PROMOTION_GATE.md §4 G1/G4).

This branch (feat/pr7-scorer-perseed-guard) PREPARES, but does not merge, the
scorer formalization the gate names as a later step:

  * clean_control is scored across BOTH pairs the gate requires (pairA AND pairC);
  * the G4 per-seed collateral bound (no seed, any cell, collateral Δ < −3) is
    implemented as a PROMOTION gate — separate from the aggregate cell verdict,
    which is unchanged. A breach is surfaced as a promotion blocker, never tuned.

The guard turns the documented pairD-s2 collateral −4 uptick (masked by the
aggregate guard in the gate branch) into a concrete, machine-checkable promotion
blocker. It does NOT flip any aggregate verdict (the read-time cells stay pass,
merge_path_stale stays needs_review).

Imports no torch; reads only committed JSON.
"""
import json
from pathlib import Path

import pytest

from benchmarks.pr7_twin_delta import (G4_PER_SEED_COLLATERAL_TOL,
                                       build_twin_delta)

ROOT = Path(__file__).resolve().parent.parent
DELTA = ROOT / "results/issue_failure_mode_blindness/pr7/twin_delta_quarantine.json"


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)


@pytest.fixture
def delta():
    return json.loads(DELTA.read_text())


def test_clean_control_scores_pairA_and_pairC(delta):
    cc = delta["cells"]["clean_control"]
    assert cc["pairs"] == ["pairA", "pairC"]
    assert set(cc["per_pair"]) == {"pairA", "pairC"}
    # clean traffic: quarantine is inert (0 merge-suspect events), so both pairs
    # are a zero-delta pass and carry no per-seed collateral breach.
    assert cc["verdict"] == "pass"
    assert cc["per_seed_collateral"]["g4_ok"] is True


def test_g4_per_seed_guard_flags_the_paird_s2_uptick(delta):
    assert delta["g4_tolerance"] == G4_PER_SEED_COLLATERAL_TOL == 3
    assert delta["g4_per_seed_collateral_ok"] is False
    breaches = delta["g4_per_seed_collateral_breaches"]
    # the same pairD-s2 −4 surfaces in both cells that score pairD stale-soft.
    keyed = {(b["cell"], b["pair"], b["seed"], b["collateral_delta"])
             for b in breaches}
    assert ("direct_harm", "pairD", 2, -4) in keyed
    assert ("merge_path_stale", "pairD", 2, -4) in keyed
    # no OTHER seed breaches (collateral_harm/pairE-s2 −2 is within the bound).
    assert all(b["collateral_delta"] < -3 for b in breaches)


def test_per_cell_per_seed_block_present_and_consistent(delta):
    for cell, cv in delta["cells"].items():
        if cv["verdict"] in ("observe_only", "inconclusive"):
            continue
        blk = cv["per_seed_collateral"]
        assert blk["tolerance"] == 3
        assert blk["g4_ok"] == (not blk["breaches"])
        assert blk["worst_seed_collateral_delta"] <= 0 or not blk["breaches"]


def test_aggregate_verdicts_unchanged_by_the_guard(delta):
    # the per-seed guard is additive: aggregate panel verdict and shape verdicts
    # are exactly as the gate branch reported.
    assert delta["overall_verdict"] == "needs_review"
    assert delta["both_shapes_ok"] is True
    for c in ("clean_control", "collateral_harm", "direct_harm"):
        assert delta["cells"][c]["verdict"] == "pass"
    assert delta["cells"]["merge_path_stale"]["verdict"] == "needs_review"


def test_promotion_blockers_and_scope(delta):
    blockers = delta["promotion_blockers_this_scorer"]
    assert any("G4 per-seed collateral breach" in b for b in blockers)
    assert delta["aggregate_panel_and_g4_clear"] is False
    # the scope note: this scorer is necessary-not-sufficient (G3 + completeness
    # live elsewhere), so an empty blocker list would NOT mean "promote".
    assert "G3 recoverability" in delta["promotion_note"]


def test_committed_manifest_matches_fresh_build(delta):
    assert delta == json.loads(json.dumps(build_twin_delta("quarantine")))
