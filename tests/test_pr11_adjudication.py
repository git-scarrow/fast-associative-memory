"""tests/test_pr11_adjudication.py — hermetic gates for PR-11.1.

The registered scorer change (PR11_ADJUDICATION_DESIGN.md §5): three
parameter-free set-membership policies appended to POLICIES —
`adjudicated-abstain` (P1: top-1 in quarantine(E) | deprecate(E)),
`merge-support-abstain` (P2: ANY surviving candidate in M) and
`pending-abstain` (P3: P1's sets plus ambiguous(E)) — merge-abstain's
certified action shape: no exclusions, no vote recomputation, no
tie/confidence/margin term, no new constants; where the trigger is
silent the `none` answer passes through untouched, so forced abstention
is structurally impossible.

Gates:
  1. registration is append-only (the pre-PR-11 prefix order unchanged);
  2. semantics on synthetic fixtures per policy: the trigger fires with
     the right per-trigger label, does not fire when the set member is
     not leading (P1/P3) or absent from the surviving support (P2),
     respects the verdict's epoch, and P1 does NOT act on an ambiguous
     verdict while P3 does;
  3. no tie trigger: an exact 0.5/0.5 tie with an empty router passes
     through on all three (the refuted unconditional abstain-tie is
     demonstrably not a component);
  4. structural sweep over BOTH re-emitted committed tables: for every
     cell, each new policy has abstained_forced == 0, fixed == broken ==
     tie_flips == 0, acted == abstained, an EXACT per-trigger
     decomposition (abstained == sum of its trigger counters), and zero
     actions on every clean cell (clean arms carry zero router state);
  5. regression pins: one pre-existing cell value per table (unchanged
     by the re-emission) and one new-policy cell value per table.

CPU-only, no GPU, no network.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.analyze_fork_governance import (  # noqa: E402
    ADJUDICATION_POLICIES, ADJUDICATION_TRIGGER_COUNTERS, POLICIES,
    READ_TIME_FORK_ONLY, TRUST_ROUTED, _vote, apply_policy)
from tests.test_pr4_governance_variants import (  # noqa: E402
    _mk_cands, _mk_router)

RESULTS = Path(__file__).resolve().parent.parent / \
    "results/issue_failure_mode_blindness"

PRE_PR11_POLICIES = [
    "none", "observe-only", "entropy-abstain", "abstain-tie",
    "recency-naive", "quarantine-naive",
    "mode-conditioned-observe", "mode-conditioned-abstain",
    "mode-conditioned-trust", "trust-downweight", "trust-guarded",
    "merge-abstain"]


def _apply(policy, cands, router, epoch=0):
    none_ans, _ = _vote(cands, 2)
    return none_ans, apply_policy(policy, cands, [], none_ans, 2,
                                  epoch, {}, True, router)


# ---------------------------------------------------------------------------
# 1. registration
# ---------------------------------------------------------------------------
def test_policies_appended_never_reordered():
    assert POLICIES[:len(PRE_PR11_POLICIES)] == PRE_PR11_POLICIES
    assert POLICIES[len(PRE_PR11_POLICIES):] == list(ADJUDICATION_POLICIES)
    for p in ADJUDICATION_POLICIES:
        assert p not in TRUST_ROUTED
        assert p not in READ_TIME_FORK_ONLY


# ---------------------------------------------------------------------------
# 2. semantics
# ---------------------------------------------------------------------------
def test_adjudicated_abstain_semantics():
    cands = _mk_cands((1, 0, 0.7), (2, 1, 0.3))  # surviving top-1 = slot 1
    # contradiction: both members quarantined; the leading member abstains
    router = _mk_router(I=1, O=2, verdict="contradiction")
    none_ans, (ans, acted, detail) = _apply("adjudicated-abstain",
                                            cands, router)
    assert (ans, acted) == (None, True)
    assert detail == {"trigger": "adjudicated"}
    # supersession: only the incumbent (I) is deprecated
    router = _mk_router(I=1, O=2, verdict="supersession")
    none_ans, (ans, acted, detail) = _apply("adjudicated-abstain",
                                            cands, router)
    assert (ans, acted, detail) == (None, True, {"trigger": "adjudicated"})
    # ...so when the NON-deprecated side leads, exact pass-through
    router = _mk_router(I=2, O=1, verdict="supersession")  # deprecates 2
    none_ans, (ans, acted, detail) = _apply("adjudicated-abstain",
                                            cands, router)
    assert (ans, acted, detail) == (none_ans, False, None)
    # ambiguous verdict: adjudicated-only policy does NOT act
    router = _mk_router(I=1, O=2, verdict="ambiguous")
    none_ans, (ans, acted, detail) = _apply("adjudicated-abstain",
                                            cands, router)
    assert (ans, acted, detail) == (none_ans, False, None)
    # a verdict recorded only at a later epoch must not fire at epoch 0
    router = _mk_router(I=1, O=2, verdict="contradiction")
    router["pairs"][0]["verdict_by_epoch"] = {
        1: router["pairs"][0]["verdict_by_epoch"][0]}
    none_ans, (ans, acted, detail) = _apply("adjudicated-abstain",
                                            cands, router, epoch=0)
    assert (ans, acted, detail) == (none_ans, False, None)


def test_pending_abstain_semantics():
    cands = _mk_cands((1, 0, 0.7), (2, 1, 0.3))
    # ambiguous verdict (the open window): pending-abstain holds the lead
    router = _mk_router(I=1, O=2, verdict="ambiguous")
    none_ans, (ans, acted, detail) = _apply("pending-abstain", cands, router)
    assert (ans, acted, detail) == (None, True, {"trigger": "pending"})
    # adjudicated verdicts fire with the ADJUDICATED trigger, not pending
    router = _mk_router(I=1, O=2, verdict="contradiction")
    none_ans, (ans, acted, detail) = _apply("pending-abstain", cands, router)
    assert (ans, acted, detail) == (None, True, {"trigger": "adjudicated"})
    # a pending pair whose members do not lead: exact pass-through
    router = _mk_router(I=5, O=6, verdict="ambiguous")
    none_ans, (ans, acted, detail) = _apply("pending-abstain", cands, router)
    assert (ans, acted, detail) == (none_ans, False, None)


def test_merge_support_abstain_semantics():
    cands = _mk_cands((1, 0, 0.7), (2, 1, 0.3))
    empty = {"pairs": [], "merge": []}
    # merge-suspect slot in the SUPPORT but not leading -> abstain
    none_ans, (ans, acted, detail) = _apply(
        "merge-support-abstain", cands, {"pairs": [], "merge": [(0, 2)]})
    assert (ans, acted, detail) == (None, True, {"trigger": "merge_support"})
    # leading -> also abstain (strict superset of merge-abstain's trigger)
    none_ans, (ans, acted, detail) = _apply(
        "merge-support-abstain", cands, {"pairs": [], "merge": [(0, 1)]})
    assert (ans, acted, detail) == (None, True, {"trigger": "merge_support"})
    # merge event only at a later epoch -> exact pass-through
    none_ans, (ans, acted, detail) = _apply(
        "merge-support-abstain", cands, {"pairs": [], "merge": [(1, 2)]},
        epoch=0)
    assert (ans, acted, detail) == (none_ans, False, None)
    # a NON-surviving candidate in M must not trigger
    cands_ns = _mk_cands((1, 0, 0.7), (2, 1, 0.3))
    cands_ns[1]["surviving"] = False
    none_ans, (ans, acted, detail) = _apply(
        "merge-support-abstain", cands_ns, {"pairs": [], "merge": [(0, 2)]})
    assert (ans, acted, detail) == (none_ans, False, None)
    # no merge evidence -> exact pass-through
    none_ans, (ans, acted, detail) = _apply(
        "merge-support-abstain", cands, empty)
    assert (ans, acted, detail) == (none_ans, False, None)


# ---------------------------------------------------------------------------
# 3. no tie trigger
# ---------------------------------------------------------------------------
def test_no_tie_trigger_on_any_pr11_policy():
    tie = _mk_cands((1, 0, 0.5), (2, 1, 0.5))
    empty = {"pairs": [], "merge": []}
    for policy in ADJUDICATION_POLICIES:
        none_ans, (ans, acted, detail) = _apply(policy, tie, empty)
        assert (ans, acted, detail) == (none_ans, False, None), policy


# ---------------------------------------------------------------------------
# 4. structural sweep over both re-emitted committed tables
# ---------------------------------------------------------------------------
def _governance_cells():
    pr4 = json.loads(
        (RESULTS / "pr4/pr4_geometry_table.json").read_text())["governance"]
    pr3c = json.loads(
        (RESULTS / "pr3c/pr3c_governance_table.json").read_text())[
            "governance"]
    for name, cell in list(pr4.items()) + list(pr3c.items()):
        yield name, cell


def test_structural_pins_on_both_tables():
    seen = 0
    for name, cell in _governance_cells():
        for policy in ADJUDICATION_POLICIES:
            row = cell[policy]
            assert row["abstained_forced"] == 0, (name, policy)
            assert row["fixed"] == row["broken"] == row["tie_flips"] == 0, \
                (name, policy)
            assert row["acted"] == row["abstained"], (name, policy)
            # exact per-trigger decomposition
            assert row["abstained"] == sum(
                row[c] for c in ADJUDICATION_TRIGGER_COUNTERS), \
                (name, policy)
            # P2 records only its own trigger; P1 never records pending
            if policy == "merge-support-abstain":
                assert row["abstained"] == row["abstained_merge_support"], \
                    name
            if policy == "adjudicated-abstain":
                assert row["abstained_pending"] == 0, name
            if "clean" in name:
                assert row["abstained"] == 0, (name, policy)
            seen += 1
    assert seen == 114 * 3  # 90 pr4 + 24 pr3c cells, 3 policies each


# ---------------------------------------------------------------------------
# 5. regression pins
# ---------------------------------------------------------------------------
def test_regression_pins():
    pr4 = json.loads(
        (RESULTS / "pr4/pr4_geometry_table.json").read_text())["governance"]
    pr3c = json.loads(
        (RESULTS / "pr3c/pr3c_governance_table.json").read_text())[
            "governance"]
    # pre-existing values, unchanged by the re-emission (PR-9/PR-10 pins)
    assert pr4["pairD/soft/s0"]["merge-abstain"]["abstained"] == 300
    assert pr4["pairD/soft/s0"]["merge-abstain"]["stale_wrong_abstained"] \
        == 292
    assert pr3c["contra_s0"]["none"]["abstained"] == 0
    # new-policy values (the PR-11.1 scan's committed table entries)
    d = pr4["pairD/soft/s0"]["merge-support-abstain"]
    assert d["abstained"] == 663 and d["abstain_on_correct"] == 288
    assert d["stale_wrong_abstained"] == 375  # 292 M-led + all 83 residual
    a = pr4["pairD/contra/s0"]["adjudicated-abstain"]
    assert a["abstained"] == a["abstained_adjudicated"] > 0
    p = pr4["pairD/oneshot/s0"]["pending-abstain"]
    assert p["abstained_pending"] > 0
