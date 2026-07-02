"""tests/test_pr9_merge_abstain.py — hermetic gates for PR-9.1(b).

The registered scorer change (PR9_ENVELOPE_DESK_CHECK.md §3.2): the
`merge-abstain` policy — abstain iff the deployed vote's surviving top-1
slot is merge-suspect, otherwise the `none` answer unchanged; no
exclusions, no vote recomputation, NO tie trigger, no parameters — plus
the previously-discarded `forced` flag and the per-trigger abstention
counters (`abstained_merge`, `abstained_forced`) as additive columns.

Gates:
  1. semantics on a synthetic fixture (merge-led -> abstain with the
     trigger recorded; merge not leading -> exact `none` pass-through;
     future merge events don't fire early; an exact vote tie does NOT
     trigger — unconditional abstain-tie was refuted, PR-9.1(a) §2);
  2. the forced flag is recorded where the desk check proved it was
     discarded (a vote whose surviving candidate set is fully excluded);
  3. integration on the tiny synthetic soft/one-shot caches: merge-path
     capture equals the mode-conditioned family's merge trigger, zero
     changed answers, and off the merge path the policy row is identical
     to `none`;
  4. regression pins on the two committed re-emitted tables: one
     PRE-EXISTING cell value each (unchanged by the re-emission) and one
     NEW-field cell each (the PR-9 additions).

CPU-only, no GPU, no network.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.analyze_fork_governance import (  # noqa: E402
    POLICIES, READ_TIME_FORK_ONLY, TRUST_ROUTED, _vote, apply_policy)
from tests.test_pr3c_shadow import _score  # noqa: E402
from tests.test_pr4_governance_variants import _mk_cands  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / \
    "results/issue_failure_mode_blindness"


def _apply_merge(cands, router, epoch=0):
    none_ans, _ = _vote(cands, 2)
    return none_ans, apply_policy("merge-abstain", cands, [], none_ans, 2,
                                  epoch, {}, True, router)


def test_merge_abstain_is_registered_additively():
    assert POLICIES[-1] == "merge-abstain"  # appended, never reordered
    assert "merge-abstain" not in TRUST_ROUTED
    assert "merge-abstain" not in READ_TIME_FORK_ONLY


def test_merge_abstain_semantics_on_fixture():
    cands = _mk_cands((1, 0, 0.7), (2, 1, 0.3))  # surviving top-1 = slot 1
    # merge-suspect leads the deployed vote -> abstain, trigger recorded
    none_ans, (ans, acted, detail) = _apply_merge(
        cands, {"pairs": [], "merge": [(0, 1)]})
    assert (ans, acted) == (None, True)
    assert detail == {"trigger": "merge"}
    # merge-suspect present but NOT leading -> exact `none` pass-through
    none_ans, (ans, acted, detail) = _apply_merge(
        cands, {"pairs": [], "merge": [(0, 2)]})
    assert (ans, acted, detail) == (none_ans, False, None)
    # a merge event at a LATER epoch must not fire at this probe's epoch
    none_ans, (ans, acted, detail) = _apply_merge(
        cands, {"pairs": [], "merge": [(1, 1)]}, epoch=0)
    assert (ans, acted, detail) == (none_ans, False, None)


def test_merge_abstain_has_no_tie_trigger():
    """Exact 0.5/0.5 tie, no merge event: merge-abstain answers the `none`
    answer; abstain-tie abstains on the same input (the refuted component
    is demonstrably NOT part of the new policy)."""
    tie = _mk_cands((1, 0, 0.5), (2, 1, 0.5))
    none_ans, (ans, acted, detail) = _apply_merge(
        tie, {"pairs": [], "merge": []})
    assert (ans, acted, detail) == (none_ans, False, None)
    ans_tie, acted_tie, _ = apply_policy("abstain-tie", tie, tie, none_ans,
                                         2, 0, {}, True, None)
    assert (ans_tie, acted_tie) == (None, True)


def test_forced_flag_is_recorded():
    """quarantine-naive dropping every surviving candidate: `_vote`
    force-abstains and the previously-discarded flag reaches `detail`."""
    cands = _mk_cands((1, 0, 0.5), (2, 1, 0.5))
    none_ans, _ = _vote(cands, 2)
    ans, acted, detail = apply_policy("quarantine-naive", cands, cands,
                                      none_ans, 2, 0, {}, True, None)
    assert (ans, acted) == (None, True)
    assert detail["trigger"] == "forced"


def test_merge_abstain_on_soft_arm_matches_merge_trigger(tmp_path):
    _, table, _ = _score("stale", tmp_path, "soft.csv", epochs=5,
                         supersede_epoch=3, payload_mode="soft")
    wrong = table["none"]["wrong_none"]
    m = table["merge-abstain"]
    assert m["stale_wrong_abstained"] == wrong > 0
    assert m["fixed"] == m["broken"] == m["tie_flips"] == 0
    assert m["abstained"] == m["abstained_merge"] == m["acted"] > 0
    assert m["abstained_forced"] == 0
    assert m["answered"] == m["n"] - m["abstained"]
    # the merge trigger is policy-invariant (desk check §1): identical
    # merge-triggered abstention count across the mode-conditioned family
    for pol in ("mode-conditioned-observe", "mode-conditioned-abstain",
                "mode-conditioned-trust"):
        assert table[pol]["abstained_merge"] == m["abstained_merge"], pol


def test_merge_abstain_is_none_off_the_merge_path(tmp_path):
    _, table, _ = _score("stale", tmp_path, "one.csv", epochs=7,
                         supersede_epoch=3, one_shot=True)
    assert table["_router"]["n_merge_suspect_events"] == 0
    assert table["merge-abstain"] == table["none"]


def test_regression_pin_pr3c_table():
    t = json.loads((RESULTS / "pr3c/pr3c_governance_table.json")
                   .read_text())["governance"]
    # PRE-EXISTING cells (PR3C_RESULT.md §2), unchanged by the re-emission
    assert t["mixed_s0"]["mode-conditioned-trust"]["fixed"] == 320
    assert t["contra_s0"]["none"]["n"] == 2822
    assert t["contra_s0"]["none"]["wrong_none"] == 264
    # NEW fields: merge-abstain on the required merge-path benchmark
    m = t["stale-soft_s0"]["merge-abstain"]
    assert m["stale_wrong_abstained"] == m["stale_wrong"] == 374
    assert m["abstained"] == m["abstained_merge"] == 374
    assert m["abstained_forced"] == 0
    assert m["fixed"] == m["broken"] == 0


def test_regression_pin_pr4_table():
    t = json.loads((RESULTS / "pr4/pr4_geometry_table.json")
                   .read_text())["governance"]
    # PRE-EXISTING cell (PR4_RESULT.md §4: pair D mixed s0 direct br 163,
    # collateral 77), unchanged by the re-emission
    trust = t["pairD/mixed/s0"]["mode-conditioned-trust"]
    assert trust["broken"] == 240
    assert trust["direct_br"] == 163
    assert trust["collateral_br"] == 77
    # NEW fields: the one registered envelope breach, recorded exactly —
    # pairD/soft/s0 capture 292/375 = 0.7787 < the 0.79 bound — and the
    # desk check's provable |M|+|F| split, now recorded (451 = 300 + 151)
    m = t["pairD/soft/s0"]["merge-abstain"]
    assert m["abstained"] == m["abstained_merge"] == 300
    assert m["abstained_forced"] == 0
    assert m["stale_wrong_abstained"] == 292
    assert m["stale_wrong"] == 375
    mc = t["pairD/soft/s0"]["mode-conditioned-observe"]
    assert mc["abstained"] == 451
    assert mc["abstained_merge"] == 300
    assert mc["abstained_forced"] == 151
