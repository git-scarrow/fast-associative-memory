"""PR-7 step 6/7 pins — the committed merge_path_stale QUARANTINE twin verdict,
across the pairD cell (step 6) and its pairE held-out replication (step 7).

`--govern quarantine` diverts the write-time merge_suspect (supersession) write
to a RECOVERABLE side ledger before it commits — excluded from the active memory
state / deployed read vote, but retained, not destroyed (PR7_DESIGN §4 quarantine
row). It runs against the SAME committed `none` baseline per geometry, with the
frozen readout applied to both arms.

In this engine-frozen harness the "side region" is the write-path quarantine
ledger (provenance only); the active ContinuousCAM the deployed read vote reads
never receives the diverted write. So quarantine's read-time effect equals
refuse's on the same geometry, and the acting-arm semantics apply: `capture_stable`
is N/A (quarantine is designed to remove that capture from the active state), the
cell is scored by refused-opportunity accounting + readout improvement +
collateral, and is never auto-passed.

The committed result (twin_delta_quarantine.json — a separate per-action manifest,
so the annotate floor in twin_delta.json and the refuse result in
twin_delta_refuse.json stay byte-identical) now aggregates BOTH committed
geometries of the merge_path_stale cell:

  * verdict = needs_review (NOT fail, NOT pass): quarantine reduced read-time harm
    on both geometries with no collateral regression, while removing the capture
    from the active state as designed (192→0 every seed, every pair);
  * pairD (step 6): broken_delta 111, stale_wrong 300, collateral net 26;
  * pairE (step 7, held-out): broken_delta 31, stale_wrong 40, collateral net 5 —
    the SAME pattern (improves readout, removes-but-retains capture, no collateral
    regression) at the smaller magnitude pairE's milder baseline hazard predicts;
  * aggregate accounting: opportunity = intercepted = capture_delta = 1152, all
    1152 diverted writes RETAINED recoverable in the ledger (retained_recoverable
    True, absorbed_into_active_memory False, payload histogram preserved) — where
    refuse discards them.

Boundaries held: engine byte-frozen (sha256 parity), geometry never a gate, the
quarantine decision recorded in provenance (quarantined_events 192/seed/pair +
ledger), each none baseline byte-identical and ungoverned.

Reads only committed artifacts; imports no torch and touches no cache.
"""
import json
from pathlib import Path

import pytest

from benchmarks.pr7_twin_delta import BASELINE_GOVERN, build_twin_delta

ROOT = Path(__file__).resolve().parent.parent
DELTA = ROOT / "results/issue_failure_mode_blindness/pr7/twin_delta_quarantine.json"
REFUSE = ROOT / "results/issue_failure_mode_blindness/pr7/twin_delta_refuse.json"
CELL = ROOT / "results/issue_failure_mode_blindness/pr7/twin/merge_path_stale"

# Per-geometry expected deltas (baseline − governed, summed over seeds 0/1/2),
# recorded from the committed twin arms — pinned, never re-tuned to force a pass.
PAIR_EXPECT = {
    "pairD": {"classes": [10, 28, 32, 95], "broken": 111, "stale": 300,
              "collateral": 26, "capture": 576},
    "pairE": {"classes": [47, 56, 61, 76], "broken": 31, "stale": 40,
              "collateral": 5, "capture": 576},
}


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)


@pytest.fixture
def delta():
    return json.loads(DELTA.read_text())


@pytest.fixture
def cell(delta):
    return delta["cells"]["merge_path_stale"]


def _pair_delta(cell, pair, key):
    pp = cell["per_pair"][pair]
    return sum(pp["delta_by_seed"][s][key] for s in pp["delta_by_seed"])


# ---------------------------------------------------------------------------
# Verdict + scope
# ---------------------------------------------------------------------------
def test_quarantine_is_needs_review_not_fail_not_pass(delta, cell):
    assert delta["govern_action"] == "quarantine"
    # an acting arm that helped and did not regress: surfaced for review, never
    # auto-passed, never discarded as a flat fail.
    assert delta["overall_verdict"] == "needs_review"
    assert delta["scored_cells"] == ["merge_path_stale"]
    assert cell["verdict"] == "needs_review"
    assert cell["guard"] == "capture_stable"
    assert cell["action_kind"] == "acting_arm"
    # capture preservation is N/A for an acting arm — reported, never a fail.
    assert cell["capture_stable"] is None
    assert cell["regressed"] is False
    assert cell["improved"] is True


def test_both_geometries_scored(cell):
    scored = sorted(p for p in cell["per_pair"]
                    if cell["per_pair"][p].get("governed_present"))
    assert scored == ["pairD", "pairE"]
    for pair in ("pairD", "pairE"):
        assert cell["per_pair"][pair]["seeds_scored"] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Per-geometry deltas + capture removed-from-active-state (held-out replication)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pair", ["pairD", "pairE"])
def test_per_pair_readout_and_capture(cell, pair):
    exp = PAIR_EXPECT[pair]
    assert _pair_delta(cell, pair, "frozen_probe_broken") == exp["broken"]
    assert _pair_delta(cell, pair, "stale_wrong") == exp["stale"]
    assert _pair_delta(cell, pair, "frozen_probe_collateral_br") == exp["collateral"]
    assert _pair_delta(cell, pair, "merge_suspect_events") == exp["capture"]
    pp = cell["per_pair"][pair]
    for s in ("0", "1", "2"):
        base, gov = pp["baseline_by_seed"][s], pp["governed_by_seed"][s]
        # capture: 192 (baseline) -> 0 (every supersession absorb diverted out of
        # the active state into the recoverable ledger), every seed both pairs.
        assert base["merge_suspect_events"] == 192
        assert gov["merge_suspect_events"] == 0
        # readout never worsens per seed: broken non-increasing, stale strictly
        # reduced (recorded actuals — pairE s2 broken is flat, not a regression).
        assert gov["frozen_probe_broken"] <= base["frozen_probe_broken"]
        assert gov["stale_wrong"] < base["stale_wrong"]


def test_pairE_replicates_pairD_at_smaller_magnitude(cell):
    """Held-out replication: pairE shows the SAME qualitative quarantine result
    as pairD — readout improves, capture removed-but-retained, no collateral
    regression — at the smaller magnitude pairE's milder baseline hazard predicts
    (recorded, not engineered: both improve, pairE < pairD)."""
    for key in ("frozen_probe_broken", "stale_wrong"):
        d = _pair_delta(cell, "pairD", key)
        e = _pair_delta(cell, "pairE", key)
        assert d > 0 and e > 0, (key, d, e)          # both improve
        assert e < d, (key, e, d)                    # pairE milder than pairD
    # neither geometry regresses collateral (net delta >= 0 both).
    assert _pair_delta(cell, "pairD", "frozen_probe_collateral_br") >= 0
    assert _pair_delta(cell, "pairE", "frozen_probe_collateral_br") >= 0


# ---------------------------------------------------------------------------
# Aggregate accounting + recoverable ledger (pairD + pairE)
# ---------------------------------------------------------------------------
def test_aggregate_accounting_and_recoverable_ledger(cell):
    acc = cell["acting_arm_accounting"]
    assert acc["action_kind"] == "acting_arm"
    assert acc["capture_preservation"] == "not_applicable_acting_arm"
    # every merge-suspect event across both geometries (192/seed × 3 × 2) diverted.
    assert acc["opportunity_count"] == 1152
    assert acc["refused_count"] == 1152          # intercepted (quarantined) count
    assert acc["capture_delta_total"] == 1152
    # quarantine RETAINS (does not destroy): expected-capture-effect + disposition.
    assert "retained" in acc["expected_capture_effect"]
    assert "consumed" not in acc["expected_capture_effect"]
    assert acc["disposition"] == "retained_recoverable_excluded_from_active_state"
    ledger = acc["quarantine_ledger"]
    assert ledger["opportunity_count"] == 1152
    assert ledger["quarantined_count"] == 1152
    assert ledger["retained_recoverable"] is True
    assert ledger["absorbed_into_active_memory"] is False
    assert sum(ledger["payload_label_histogram"].values()) == 1152
    # readout improved (baseline − governed > 0); collateral did not regress.
    assert acc["readout_broken_delta_total"] == 142    # 111 (D) + 31 (E)
    assert acc["readout_stale_wrong_delta_total"] == 340  # 300 (D) + 40 (E)
    assert acc["collateral_delta_total"] == 31         # 26 (D) + 5 (E), >= 0
    assert acc["direct_delta_total"] >= 0


# ---------------------------------------------------------------------------
# Provenance + boundaries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pair", ["pairD", "pairE"])
def test_quarantine_provenance(pair):
    for seed in (0, 1, 2):
        summ = json.loads(
            (CELL / "quarantine"
             / f"per_probe_stale-soft_s{seed}_{pair}.summary.json").read_text())
        g = summ["govern"]
        assert g["action"] == "quarantine"
        assert g["step"] == "pr7-step6-quarantine"
        assert g["quarantined_events"] == 192
        assert g["quarantined_event_class"] == "supersession"
        ledger = g["quarantine_ledger"]
        assert ledger["opportunity_count"] == 192
        assert ledger["quarantined_count"] == 192
        assert ledger["retained_recoverable"] is True
        assert ledger["absorbed_into_active_memory"] is False
        assert sum(ledger["payload_label_histogram"].values()) == 192
        assert "reason" in ledger and "reason" in g
        assert summ["payload_mode"] == "soft"
        assert summ["classes"] == PAIR_EXPECT[pair]["classes"]


def test_boundaries(delta):
    assert delta["engine_or_retrieval_change"] is False
    assert delta["geometry_used_as_gate"] is False
    assert delta["deployed_engine_sha256_parity"] is True


@pytest.mark.parametrize("pair", ["pairD", "pairE"])
def test_baseline_none_unchanged_and_ungoverned(pair):
    for seed in (0, 1, 2):
        summ = json.loads(
            (CELL / BASELINE_GOVERN
             / f"per_probe_stale-soft_s{seed}_{pair}.summary.json").read_text())
        assert "govern" not in summ


# ---------------------------------------------------------------------------
# Quarantine-vs-refuse comparison (pairD-scoped — refuse only ran pairD)
# ---------------------------------------------------------------------------
def test_pairD_quarantine_matches_refuse_but_retains_where_refuse_discards(cell):
    """On pairD, quarantine and refuse have the SAME active-memory effect (the
    engine-frozen harness keeps the diverted write out of the active CAM either
    way), so their pairD read-time deltas are identical; quarantine's added value
    is the recoverable ledger (refuse keeps only a count, no retention)."""
    refuse = json.loads(REFUSE.read_text())
    r_pair = refuse["cells"]["merge_path_stale"]["per_pair"]["pairD"]
    q_pair = cell["per_pair"]["pairD"]
    # identical per-seed deltas on pairD (broken, stale, collateral, capture).
    assert q_pair["delta_by_seed"] == r_pair["delta_by_seed"]
    # both acting arms reach needs_review; only quarantine retains recoverably.
    r_acc = refuse["cells"]["merge_path_stale"]["acting_arm_accounting"]
    q_acc = cell["acting_arm_accounting"]
    assert "quarantine_ledger" in q_acc and "quarantine_ledger" not in r_acc
    assert q_acc["quarantine_ledger"]["retained_recoverable"] is True
    assert "consumed" in r_acc["expected_capture_effect"]
    assert "retained" in q_acc["expected_capture_effect"]


def test_committed_manifest_matches_fresh_build(delta):
    fresh = json.loads(json.dumps(build_twin_delta("quarantine")))
    assert delta == fresh
