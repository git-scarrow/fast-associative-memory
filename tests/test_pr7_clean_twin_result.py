"""PR-7 step 3 pins — the committed pairA clean-control twin-run result.

Step 2 left ``twin_delta.json`` baseline-only (no governed arm committed). Step
3 ran the real pairA ``clean_control`` twin on the vitl14 manifold — the
ungoverned ``--govern none`` baseline against the ``--govern annotate``
null-action floor, 3 seeds, only the ``--govern`` flag differing — and committed
both arms. These pins fix that committed result so it cannot silently drift:

  * the null-action floor cost EXACTLY nothing — a zero delta on every measurand
    across all 3 seeds, so ``clean_control`` is a ``pass`` and the run added no
    harm to clean traffic (PR7_DESIGN §4/§8.4);
  * scope held — no governed arms for the other hazard cells, so they stay
    ``inconclusive``; one-shot ambiguity stays ``observe_only`` (never scored);
  * the boundaries held — engine byte-frozen (sha256 parity), geometry never a
    gate, no engine/retrieval change, govern action ``annotate``;
  * the committed manifest matches a fresh build over the committed twin arms.

Reads only committed artifacts; imports no torch and touches no cache.
"""
import json
from pathlib import Path

import pytest

from benchmarks.pr7_twin_delta import BASELINE_GOVERN, build_twin_delta

ROOT = Path(__file__).resolve().parent.parent
DELTA = ROOT / "results/issue_failure_mode_blindness/pr7/twin_delta.json"
TWIN = ROOT / "results/issue_failure_mode_blindness/pr7/twin"


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    # build_twin_delta reads the committed twin arms (and the frozen engine for
    # the sha256 parity field) via repo-relative paths; pin cwd so a fresh build
    # resolves the same artifacts the committed manifest was built from.
    monkeypatch.chdir(ROOT)


@pytest.fixture
def delta():
    return json.loads(DELTA.read_text())


def test_overall_is_zero_delta_pass(delta):
    assert delta["govern_action"] == "annotate"
    assert delta["overall_verdict"] == "pass"
    assert delta["both_shapes_ok"] is True
    # clean_control is one of the scored cells (step 4 adds merge_path_stale;
    # this test owns only the clean_control contribution).
    assert "clean_control" in delta["scored_cells"]


def test_clean_control_costs_nothing(delta):
    cell = delta["cells"]["clean_control"]
    assert cell["verdict"] == "pass"
    assert cell["frozen_probe_broken_delta_total"] == 0
    assert cell["merge_suspect_capture_delta_total"] == 0
    assert cell["improved"] is False
    assert cell["regressed"] is False
    pairA = cell["per_pair"]["pairA"]
    assert pairA["baseline_present"] and pairA["governed_present"]
    assert pairA["seeds_scored"] == [0, 1, 2]
    # the null-action floor: every measurand's baseline-minus-governed delta is
    # exactly zero on every seed.
    for seed_delta in pairA["delta_by_seed"].values():
        assert all(v == 0 for v in seed_delta.values())


def test_direct_collateral_stay_unscored(delta):
    # no acting-arm cells were run (those are quarantine/refuse, a later step);
    # direct/collateral harm have no governed arm and stay inconclusive.
    for cell in ("direct_harm", "collateral_harm"):
        assert delta["cells"][cell]["verdict"] == "inconclusive"
    # one-shot ambiguity is never scored pass/fail (PR7_DESIGN §12).
    assert delta["cells"]["one_shot_ambiguity"]["verdict"] == "observe_only"


def test_boundaries_held(delta):
    assert delta["engine_or_retrieval_change"] is False
    assert delta["geometry_used_as_gate"] is False
    # the two engine files stayed byte-frozen across the twin (PR7_DESIGN §1).
    assert delta["deployed_engine_sha256_parity"] is True
    assert delta["probe_policy"] == "mode-conditioned-trust"


def test_baseline_arm_carries_no_governance():
    # the none arm's summaries must record no govern block (the none-vote
    # bit-identity; PR7_DESIGN §10) — else the analyzer's provenance guard would
    # have raised rather than producing the committed pass.
    base = TWIN / "clean_control" / BASELINE_GOVERN
    for seed in (0, 1, 2):
        summ = json.loads(
            (base / f"per_probe_clean_s{seed}_pairA.summary.json").read_text())
        assert "govern" not in summ
        assert summ["classes"] == [0, 8, 19, 33]


def test_committed_manifest_matches_fresh_build(delta):
    # Round-trip the fresh build through JSON: the per-seed delta maps are keyed
    # by int seed in memory but stringified on disk, so the committed manifest
    # is compared against the serialized fresh build (the bytes that would be
    # written), exactly as the analyzer writes it.
    fresh = json.loads(json.dumps(build_twin_delta("annotate")))
    assert delta == fresh
