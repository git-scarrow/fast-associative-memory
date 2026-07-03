"""PR-9B second-key desk scan pins (PR9B_SECOND_KEY_DESK_SCAN.md).

Pins the pre-registered admissibility boundary, the gate's structural
unsatisfiability for lemma-invariant candidates, the verdict mapping, and
the committed scan artifact's verdict. Analysis-only: nothing here touches
the engine, the reader contract, or any runtime policy.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

import pr9b_second_key_scan as scan  # noqa: E402
from failure_mode_probe import (  # noqa: E402
    PR92_INTRINSIC_CHECK_FIELDS,
    PR92_INTRINSIC_JOIN_KEY,
)

SCAN_JSON = (ROOT / "results" / "issue_failure_mode_blindness" / "pr9b"
             / "second_key_scan.json")
PANEL = ROOT / "results" / "issue_failure_mode_blindness" / "pr9_2" / "panel"


# ---------------------------------------------------------------------------
# §1 — the admissibility boundary IS the §9A-certified intrinsic set
# ---------------------------------------------------------------------------
def test_admissible_columns_are_exactly_the_certified_intrinsic_set():
    assert set(scan.ADMISSIBLE_COLUMNS) == (
        set(PR92_INTRINSIC_JOIN_KEY) | set(PR92_INTRINSIC_CHECK_FIELDS))


def test_preregistered_bounds_pinned():
    assert scan.CAPTURE_FLOOR == 0.5
    assert scan.FALSE_ACTION_CEILING == 0.05
    assert scan.TARGET_PAIRS == ("pairD", "pairE")
    assert scan.WRITE_CLEAN_PAIRS == ("pairA", "pairB")


# ---------------------------------------------------------------------------
# §5 — gate structure
# ---------------------------------------------------------------------------
def _score(capture: float, fa: float, second_key: bool = True) -> dict:
    return {"second_key": second_key,
            "capture_bound_ok": capture >= scan.CAPTURE_FLOOR,
            "false_action_bound_ok": fa <= scan.FALSE_ACTION_CEILING,
            "passes_gate": bool(second_key
                                and capture >= scan.CAPTURE_FLOOR
                                and fa <= scan.FALSE_ACTION_CEILING)}


def test_gate_unsatisfiable_for_lemma_invariant_candidates():
    """The invariance lemma forces capture == false-action; no such number
    can clear a 0.5 floor and a 0.05 ceiling simultaneously. Property-swept
    over the whole fired-fraction range at 1/192 resolution."""
    for fired in range(193):
        x = fired / 192.0
        assert not _score(capture=x, fa=x)["passes_gate"]


def test_gate_would_pass_a_genuinely_discriminating_key():
    """The bounds are not rigged: a key firing 60% on targets and 2% on
    write-clean pairs would GO. The gate fails candidates, not the idea."""
    assert _score(capture=0.6, fa=0.02)["passes_gate"]


def test_verdict_mapping():
    assert scan.apply_verdict([]) == scan.VERDICT_ABSENT
    only_k0 = [_score(1.0, 1.0, second_key=False)]
    assert scan.apply_verdict(only_k0) == scan.VERDICT_ABSENT
    failing = [_score(x, x) for x in (0.0, 1 / 6, 5 / 6, 1.0)]
    assert scan.apply_verdict(failing) == scan.VERDICT_FAILED
    assert scan.apply_verdict(
        failing + [_score(0.6, 0.02)]) == scan.VERDICT_CANDIDATE


# ---------------------------------------------------------------------------
# committed artifacts — denominator and scan result
# ---------------------------------------------------------------------------
needs_panel = pytest.mark.skipif(not PANEL.exists(),
                                 reason="committed §9A panel not present")


@needs_panel
def test_denominator_verification_on_committed_panel():
    runs = scan.load_denominator()
    v = scan.verify_denominator(runs)
    assert v["unique_stale_soft_runs"] == 12
    assert v["unique_clean_runs"] == 6
    assert v["total_events"] == 2304
    assert v["eligible_epochs"] == [6, 7, 8, 9, 10, 11]
    assert v["coordinate_multiset_identical_across_runs"]
    assert v["payload_label_equal_across_runs"]


@needs_panel
def test_committed_scan_artifact_verdict_and_lemma():
    rep = json.loads(SCAN_JSON.read_text())
    assert rep["invariance_lemma_holds"]
    assert rep["verdict"] == scan.VERDICT_FAILED
    by_id = {c["id"]: c for c in rep["candidates"]}
    # K0 is the committed refuse arm restated: constant, not a second key.
    assert by_id["K0"]["second_key"] is False
    assert by_id["K0"]["capture"] == {"pairD": 1.0, "pairE": 1.0}
    # K1 fires exactly the §9A incumbent-diagnostic prefix: 32/192.
    assert by_id["K1"]["capture"]["pairD"] == pytest.approx(32 / 192)
    # The lemma, realized: capture == false-action for every candidate.
    for c in rep["candidates"]:
        vals = set(c["capture"].values()) | set(c["false_action"].values())
        assert len(vals) == 1, f"{c['id']} violates the invariance lemma"
        assert not c["passes_gate"]
        assert not c["requires_threshold_movement"]
        assert c["clean_fired"] == 0


@needs_panel
def test_committed_citations_match_recorded_history():
    rep = json.loads(SCAN_JSON.read_text())
    cites = rep["committed_citations"]
    base = cites["baseline_3seed_sums"]
    assert base["pairD"]["frozen_probe_broken"] == 338
    assert base["pairE"]["frozen_probe_broken"] == 138
    assert base["pairA"]["frozen_probe_broken"] == 0
    assert base["pairB"]["frozen_probe_broken"] == 1
    # The K0 ceiling: acting on everything removed only ~1/3 of broken.
    gov = cites["refuse_pairD_governed_3seed_sums"]
    assert gov["frozen_probe_broken"] == 227
    assert gov["merge_suspect_events"] == 0
    assert cites["pr11_verdict"] == "negative"
    assert cites["pr11_post_pr10_soft_residual"]["pairD"] == {
        "0": 83, "1": 38, "2": 30}
