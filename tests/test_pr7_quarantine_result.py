"""PR-7 step 6 pins — the committed pairD merge_path_stale QUARANTINE twin
verdict, the comparator for the step-5 refuse arm (PR7_DESIGN §4 quarantine row /
§8 step-5 addendum).

The second acting arm. `--govern quarantine` diverts the write-time merge_suspect
(supersession) write to a RECOVERABLE side ledger before it commits — excluded
from the active memory state / deployed read vote, but retained, not destroyed
(PR7_DESIGN §4: "allocate to a side region excluded from the deployed read vote;
the write is retained and recoverable, not destroyed"). It runs against the SAME
committed `none` baseline the annotate and refuse twins used.

In this engine-frozen harness the "side region" is the write-path quarantine
ledger (provenance only); the active ContinuousCAM the deployed read vote reads
never receives the diverted write. So quarantine's read-time effect is IDENTICAL
to refuse's — both keep the supersession write out of the active state — and the
acting-arm semantics apply unchanged: `capture_stable` is N/A (quarantine is
designed to remove that capture from the active state), the cell is scored by
refused-opportunity accounting + readout improvement + collateral, and is never
auto-passed. The committed result (twin_delta_quarantine.json — a separate
per-action manifest, so the annotate floor in twin_delta.json stays byte-identical)
records:

  * verdict = needs_review (NOT fail, NOT pass): quarantine reduced read-time harm
    (broken_delta +111, stale_wrong −300) with no readout/collateral regression,
    while removing the capture from the active state as designed (192→0 every seed);
  * full accounting: opportunity = refused/intercepted = capture_delta = 576;
  * the DISTINCTION from refuse: the 576 diverted writes are RETAINED recoverable
    in the quarantine ledger (quarantined_count 576, retained_recoverable True,
    absorbed_into_active_memory False, payload histogram preserved) — where refuse
    discards them with only a count.

Boundaries held: engine byte-frozen (sha256 parity), geometry never a gate, the
quarantine decision recorded in provenance (quarantined_events 192/seed + ledger),
the none baseline byte-identical and ungoverned.

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


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)


@pytest.fixture
def delta():
    return json.loads(DELTA.read_text())


def test_quarantine_is_needs_review_not_fail_not_pass(delta):
    assert delta["govern_action"] == "quarantine"
    # an acting arm that helped and did not regress: surfaced for review, never
    # auto-passed, never discarded as a flat fail.
    assert delta["overall_verdict"] == "needs_review"
    # PR-7 step 8 (read-time panel) added the quarantine arms for the three
    # read-time cells, so the panel now scores all four (was merge_path_stale
    # only at step 6). The overall verdict is still needs_review: the read-time
    # cells pass on the ordinary improve/not_worsen guards while merge_path_stale
    # holds at needs_review (acting arm, capture consumed by design), and
    # verdicts <= {pass, needs_review} -> needs_review. Scorer logic unchanged.
    assert delta["scored_cells"] == [
        "clean_control", "collateral_harm", "direct_harm", "merge_path_stale"]
    assert delta["both_shapes_ok"] is True
    assert delta["both_shapes_detail"]["regressed_harm_shapes"] == []
    for c in ("clean_control", "collateral_harm", "direct_harm"):
        assert delta["cells"][c]["verdict"] == "pass"
    cell = delta["cells"]["merge_path_stale"]
    assert cell["verdict"] == "needs_review"
    assert cell["guard"] == "capture_stable"
    assert cell["action_kind"] == "acting_arm"
    # capture preservation is N/A for an acting arm — reported, never a fail.
    assert cell["capture_stable"] is None
    assert cell["regressed"] is False


def test_quarantine_accounting_and_recoverable_ledger(delta):
    acc = delta["cells"]["merge_path_stale"]["acting_arm_accounting"]
    assert acc["action_kind"] == "acting_arm"
    assert acc["capture_preservation"] == "not_applicable_acting_arm"
    # every merge-suspect event (192/seed × 3) was diverted; the capture delta
    # equals that opportunity exactly — removal from the active state is intended.
    assert acc["opportunity_count"] == 576
    assert acc["refused_count"] == 576          # intercepted (quarantined) count
    assert acc["capture_delta_total"] == 576
    # quarantine RETAINS (does not destroy): the expected-capture-effect and the
    # disposition both say retained/recoverable, unlike refuse's "consumed".
    assert "retained" in acc["expected_capture_effect"]
    assert "consumed" not in acc["expected_capture_effect"]
    assert acc["disposition"] == "retained_recoverable_excluded_from_active_state"
    # the recoverable ledger — the whole point vs refuse: payloads kept, not
    # discarded, with full per-label accounting.
    ledger = acc["quarantine_ledger"]
    assert ledger["opportunity_count"] == 576
    assert ledger["quarantined_count"] == 576
    assert ledger["retained_recoverable"] is True
    assert ledger["absorbed_into_active_memory"] is False
    assert sum(ledger["payload_label_histogram"].values()) == 576
    # readout improved (baseline − governed > 0); collateral did not regress.
    assert acc["readout_broken_delta_total"] == 111
    assert acc["readout_stale_wrong_delta_total"] == 300
    assert acc["collateral_delta_total"] >= 0
    assert acc["direct_delta_total"] >= 0


def test_quarantine_reduced_readout_and_removed_capture_from_active_state(delta):
    cell = delta["cells"]["merge_path_stale"]
    assert cell["frozen_probe_broken_delta_total"] > 0
    assert cell["improved"] is True
    pairD = cell["per_pair"]["pairD"]
    assert pairD["seeds_scored"] == [0, 1, 2]
    for seed in ("0", "1", "2"):
        base = pairD["baseline_by_seed"][seed]
        gov = pairD["governed_by_seed"][seed]
        # capture: 192 (baseline) -> 0 (every supersession absorb diverted out
        # of the active state into the recoverable ledger).
        assert base["merge_suspect_events"] == 192
        assert gov["merge_suspect_events"] == 0
        # broken and stale_wrong strictly reduced on every seed.
        assert gov["frozen_probe_broken"] < base["frozen_probe_broken"]
        assert gov["stale_wrong"] < base["stale_wrong"]


def test_quarantine_provenance_and_boundaries(delta):
    assert delta["engine_or_retrieval_change"] is False
    assert delta["geometry_used_as_gate"] is False
    assert delta["deployed_engine_sha256_parity"] is True
    gov = CELL / "quarantine"
    for seed in (0, 1, 2):
        summ = json.loads(
            (gov / f"per_probe_stale-soft_s{seed}_pairD.summary.json").read_text())
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
        assert summ["classes"] == [10, 28, 32, 95]


def test_baseline_none_unchanged_and_ungoverned():
    base = CELL / BASELINE_GOVERN
    for seed in (0, 1, 2):
        summ = json.loads(
            (base / f"per_probe_stale-soft_s{seed}_pairD.summary.json").read_text())
        assert "govern" not in summ


def test_quarantine_matches_refuse_readout_but_retains_where_refuse_discards(delta):
    """The comparator point: in this engine-frozen harness quarantine and refuse
    have the SAME active-memory effect, so their read-time deltas are identical;
    quarantine's added value is the recoverable ledger (refuse keeps only a
    count, with no retention/recoverability)."""
    refuse = json.loads(REFUSE.read_text())
    q_acc = delta["cells"]["merge_path_stale"]["acting_arm_accounting"]
    r_acc = refuse["cells"]["merge_path_stale"]["acting_arm_accounting"]
    # identical read-time / capture / collateral accounting.
    for k in ("opportunity_count", "refused_count", "capture_delta_total",
              "readout_broken_delta_total", "readout_stale_wrong_delta_total",
              "collateral_delta_total", "direct_delta_total"):
        assert q_acc[k] == r_acc[k], k
    # same overall verdict (both acting arms, helped, capture removed as designed).
    assert delta["overall_verdict"] == refuse["overall_verdict"] == "needs_review"
    # but only quarantine retains the diverted writes recoverable.
    assert "quarantine_ledger" in q_acc and "quarantine_ledger" not in r_acc
    assert q_acc["quarantine_ledger"]["retained_recoverable"] is True
    assert q_acc["disposition"] == "retained_recoverable_excluded_from_active_state"
    assert "consumed" in r_acc["expected_capture_effect"]
    assert "retained" in q_acc["expected_capture_effect"]


def test_committed_manifest_matches_fresh_build(delta):
    fresh = json.loads(json.dumps(build_twin_delta("quarantine")))
    assert delta == fresh
