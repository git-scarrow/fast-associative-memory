"""PR-7 step 5 pins — the committed pairD merge_path_stale REFUSE twin verdict.

The first acting arm. `--govern refuse` skips the write-time merge_suspect
(supersession) write before it commits, against the SAME committed `none`
baseline the annotate twin used. The committed result (twin_delta_refuse.json,
a separate per-action manifest so the annotate floor in twin_delta.json is
preserved) records the honest, two-edged finding:

  * refuse REDUCES read-time harm — frozen-probe broken drops on every seed
    (baseline-minus-governed broken_delta_total > 0) and stale_wrong falls ~30%;
  * but it DESTROYS the write-time merge-suspect capture the cell guards on —
    you cannot capture an absorb you refused to perform, so capture goes 192→0
    every seed and the cell's `capture_stable` guard FAILS;
  * so merge_path_stale = fail and overall = fail, even though broken improved —
    refusal is not a free win here (PR7_DESIGN §8 stop condition).

Boundaries held: engine byte-frozen (sha256 parity), geometry never a gate, the
refuse decision recorded in provenance (refused_events 192/seed = the
supersession class), the none baseline byte-identical and ungoverned.

Reads only committed artifacts; imports no torch and touches no cache.
"""
import json
from pathlib import Path

import pytest

from benchmarks.pr7_twin_delta import BASELINE_GOVERN, build_twin_delta

ROOT = Path(__file__).resolve().parent.parent
DELTA = ROOT / "results/issue_failure_mode_blindness/pr7/twin_delta_refuse.json"
CELL = ROOT / "results/issue_failure_mode_blindness/pr7/twin/merge_path_stale"


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)


@pytest.fixture
def delta():
    return json.loads(DELTA.read_text())


def test_refuse_overall_fails_on_capture(delta):
    assert delta["govern_action"] == "refuse"
    assert delta["overall_verdict"] == "fail"
    assert delta["scored_cells"] == ["merge_path_stale"]
    cell = delta["cells"]["merge_path_stale"]
    assert cell["verdict"] == "fail"
    assert cell["guard"] == "capture_stable"
    # the cell fails because the write-time capture is destroyed, NOT because
    # read-time harm regressed.
    assert cell["capture_stable"] is False
    assert cell["merge_suspect_capture_delta_total"] == 576  # 192*3, all lost
    assert cell["regressed"] is False


def test_refuse_reduced_broken_but_killed_capture(delta):
    cell = delta["cells"]["merge_path_stale"]
    # refuse REDUCED read-time broken (baseline - governed > 0) — the two-edged
    # result: it helps read-time while breaking the capture invariant.
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
        # broken strictly reduced on every seed; stale_wrong reduced too.
        assert gov["frozen_probe_broken"] < base["frozen_probe_broken"]
        assert gov["stale_wrong"] < base["stale_wrong"]


def test_refuse_provenance_and_boundaries(delta):
    assert delta["engine_or_retrieval_change"] is False
    assert delta["geometry_used_as_gate"] is False
    assert delta["deployed_engine_sha256_parity"] is True
    # refused exactly the 192 supersession (merge_suspect) writes per seed.
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
    # the refuse twin reuses the committed step-4 none baseline, which must stay
    # ungoverned (no govern block) — the none-vote bit-identity.
    base = CELL / BASELINE_GOVERN
    for seed in (0, 1, 2):
        summ = json.loads(
            (base / f"per_probe_stale-soft_s{seed}_pairD.summary.json").read_text())
        assert "govern" not in summ


def test_committed_manifest_matches_fresh_build(delta):
    fresh = json.loads(json.dumps(build_twin_delta("refuse")))
    assert delta == fresh
