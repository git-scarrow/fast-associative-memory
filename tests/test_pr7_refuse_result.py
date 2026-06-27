"""PR-7 step 5 pins — the committed pairD merge_path_stale REFUSE twin verdict,
under the acting-arm verdict semantics (PR7_DESIGN §8 step-5 addendum).

The first acting arm. `--govern refuse` skips the write-time merge_suspect
(supersession) write before it commits, against the SAME committed `none`
baseline the annotate twin used. `capture_stable` is the right HARD guard for a
null-action floor (annotate must preserve the 192/seed capture), but it is NOT a
pass/fail criterion for an ACTING arm: refuse is DESIGNED to consume that very
capture, so penalizing it for capture loss would penalize it for working. Refuse
is instead scored by refused-opportunity accounting + readout improvement +
collateral harm, and is never auto-passed (a human-review gate). The committed
result (twin_delta_refuse.json — a separate per-action manifest, so the annotate
floor in twin_delta.json is preserved byte-identical) records:

  * verdict = needs_review (NOT fail, NOT pass): refuse reduced read-time harm
    (broken_delta +111, stale_wrong ~30% lower) with no readout/collateral
    regression, while consuming the capture as designed (192→0 every seed);
  * full accounting: opportunity_count = refused_count = capture_delta = 576
    (every capturable merge-suspect event was refused — the capture loss is
    INTENDED, not a regression), plus readout/collateral deltas.

Boundaries held: engine byte-frozen (sha256 parity), geometry never a gate, the
refuse decision recorded in provenance (refused_events 192/seed), the none
baseline byte-identical and ungoverned.

Reads only committed artifacts; imports no torch and touches no cache.
"""
import json
from pathlib import Path

import pytest

from benchmarks.pr7_twin_delta import (
    BASELINE_GOVERN, CELL_SPEC, PRE_REGISTERED_MARGINS, _cell_verdict,
    build_twin_delta)

ROOT = Path(__file__).resolve().parent.parent
DELTA = ROOT / "results/issue_failure_mode_blindness/pr7/twin_delta_refuse.json"
CELL = ROOT / "results/issue_failure_mode_blindness/pr7/twin/merge_path_stale"


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)


@pytest.fixture
def delta():
    return json.loads(DELTA.read_text())


def test_refuse_is_needs_review_not_fail_not_pass(delta):
    assert delta["govern_action"] == "refuse"
    # an acting arm that helped and did not regress: surfaced for review, never
    # auto-passed, never discarded as a flat fail.
    assert delta["overall_verdict"] == "needs_review"
    assert delta["scored_cells"] == ["merge_path_stale"]
    cell = delta["cells"]["merge_path_stale"]
    assert cell["verdict"] == "needs_review"
    assert cell["guard"] == "capture_stable"
    assert cell["action_kind"] == "acting_arm"
    # capture preservation is N/A for an acting arm — reported, never a fail.
    assert cell["capture_stable"] is None
    assert cell["regressed"] is False


def test_refuse_accounting(delta):
    acc = delta["cells"]["merge_path_stale"]["acting_arm_accounting"]
    assert acc["action_kind"] == "acting_arm"
    assert acc["capture_preservation"] == "not_applicable_acting_arm"
    # every capturable merge-suspect event (192/seed × 3) was refused; the
    # capture delta equals that opportunity exactly — consumption is intended.
    assert acc["opportunity_count"] == 576
    assert acc["refused_count"] == 576
    assert acc["capture_delta_total"] == 576
    assert "consumed" in acc["expected_capture_effect"]
    # readout improved (baseline − governed > 0); collateral did not regress.
    assert acc["readout_broken_delta_total"] == 111
    assert acc["readout_stale_wrong_delta_total"] == 300
    assert acc["collateral_delta_total"] >= 0
    assert acc["direct_delta_total"] >= 0


def test_refuse_reduced_readout_and_consumed_capture(delta):
    cell = delta["cells"]["merge_path_stale"]
    assert cell["frozen_probe_broken_delta_total"] > 0
    assert cell["improved"] is True
    pairD = cell["per_pair"]["pairD"]
    assert pairD["seeds_scored"] == [0, 1, 2]
    for seed in ("0", "1", "2"):
        base = pairD["baseline_by_seed"][seed]
        gov = pairD["governed_by_seed"][seed]
        # capture: 192 (baseline) -> 0 (refused every supersession absorb).
        assert base["merge_suspect_events"] == 192
        assert gov["merge_suspect_events"] == 0
        # broken and stale_wrong strictly reduced on every seed.
        assert gov["frozen_probe_broken"] < base["frozen_probe_broken"]
        assert gov["stale_wrong"] < base["stale_wrong"]


def test_refuse_provenance_and_boundaries(delta):
    assert delta["engine_or_retrieval_change"] is False
    assert delta["geometry_used_as_gate"] is False
    assert delta["deployed_engine_sha256_parity"] is True
    gov = CELL / "refuse"
    for seed in (0, 1, 2):
        summ = json.loads(
            (gov / f"per_probe_stale-soft_s{seed}_pairD.summary.json").read_text())
        assert summ["govern"]["action"] == "refuse"
        assert summ["govern"]["refused_events"] == 192
        assert summ["govern"]["refused_event_class"] == "supersession"
        assert "reason" in summ["govern"]
        assert summ["payload_mode"] == "soft"
        assert summ["classes"] == [10, 28, 32, 95]


def test_baseline_none_unchanged_and_ungoverned():
    base = CELL / BASELINE_GOVERN
    for seed in (0, 1, 2):
        summ = json.loads(
            (base / f"per_probe_stale-soft_s{seed}_pairD.summary.json").read_text())
        assert "govern" not in summ


def test_committed_manifest_matches_fresh_build(delta):
    fresh = json.loads(json.dumps(build_twin_delta("refuse")))
    assert delta == fresh


# ---------------------------------------------------------------------------
# Verdict-semantics unit tests (PR7_DESIGN §8 step-5 addendum): capture_stable
# is a HARD guard for null-action floors, N/A for acting arms.
# ---------------------------------------------------------------------------
_SPEC = CELL_SPEC["merge_path_stale"]  # guard == "capture_stable"
_M = PRE_REGISTERED_MARGINS["refuse"]


def test_null_action_floor_keeps_capture_stable_hard_guard():
    # a null-action floor (is_acting=False) that LOSES capture fails — the hard
    # guard is preserved for annotate-like floors.
    verdict, improved, regressed, capture_ok = _cell_verdict(
        _SPEC, PRE_REGISTERED_MARGINS["annotate"], broken_delta=0,
        capture_delta=576, any_governed=True, is_acting=False)
    assert verdict == "fail"
    assert capture_ok is False
    # a floor that preserves capture (delta 0) and does not regress passes.
    v2, _, _, ok2 = _cell_verdict(
        _SPEC, PRE_REGISTERED_MARGINS["annotate"], broken_delta=0,
        capture_delta=0, any_governed=True, is_acting=False)
    assert v2 == "pass" and ok2 is True


def test_acting_arm_capture_loss_is_not_a_fail():
    # acting arm, capture consumed, readout improved, no collateral regression
    # -> needs_review (never auto-pass), capture_ok N/A (None).
    verdict, improved, regressed, capture_ok = _cell_verdict(
        _SPEC, _M, broken_delta=111, capture_delta=576, any_governed=True,
        is_acting=True, collateral_delta=26)
    assert verdict == "needs_review"
    assert improved is True and regressed is False
    assert capture_ok is None


def test_acting_arm_fails_on_readout_or_collateral_regression():
    # readout regressed -> fail even though capture loss is expected.
    v_read, _, reg, _ = _cell_verdict(
        _SPEC, _M, broken_delta=-5, capture_delta=576, any_governed=True,
        is_acting=True, collateral_delta=0)
    assert v_read == "fail" and reg is True
    # collateral regressed (governed collateral exceeded baseline) -> fail.
    v_col, _, _, _ = _cell_verdict(
        _SPEC, _M, broken_delta=50, capture_delta=576, any_governed=True,
        is_acting=True, collateral_delta=-3)
    assert v_col == "fail"
