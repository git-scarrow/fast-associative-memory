"""PR-6 step 2 pins — seeding the merge_path_stale cell from PR-3c stale arms.

Pins the measured outcome of the analysis-only pass over the committed PR-3c
soft-payload ("merge-path") stale arms:

  (a) merge-path stale capture is MEASURED on the pair-A class set only, across
      3 seeds, and copied faithfully (no geometric input) from the committed
      governance JSON;
  (b) the frozen mode-conditioned-trust probe captures it via the WRITE-TIME
      merge-suspect trace, not read-time fork witness (broken=0);
  (c) pairs D/E never existed as geometries in PR-3c, so no committed artifact
      measures merge-path stale on D/E — the structural reason the cell cannot
      be fully seeded;
  (d) the cell therefore stays required-but-unseeded with a precise
      missing-evidence note, and no geometry is used to admit or exclude it.

Reads only committed artifacts; imports no torch and touches no cache.
"""
import json
from pathlib import Path

import pytest

from benchmarks.pr6_hazard_panel import (
    DE_CLASS_SETS, PROBE_POLICY, STALE_SOFT_ARMS, build_merge_path_stale_cell,
    build_panel, read_merge_path_stale_evidence)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results/issue_failure_mode_blindness"
PR3C = RESULTS / "pr3c"
POST = RESULTS / "pr5/hazard_postmortem.json"
PANEL = RESULTS / "pr6/panel.json"

PAIR_A = [0, 8, 19, 33]
_GEOMETRY_TOKENS = ("cos", "ratio", "confusion", "geometry", "centroid",
                    "attribution", "fork_pair")


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    # The builder resolves committed artifacts by repo-relative path.
    monkeypatch.chdir(ROOT)


@pytest.fixture
def cell():
    # Default (repo-relative) path so the cell's artifact paths match the
    # committed manifest; cwd is pinned to ROOT by the autouse fixture.
    return build_merge_path_stale_cell()


def test_cell_required_but_unseeded(cell):
    assert cell["required"] is True
    assert cell["status"] == "required_unseeded"
    assert cell["evidence_status"] == "partial_pairA_only"
    assert cell["seeds"] == []          # still unseeded: no D/E coverage
    assert cell["harm_shape"] == "write-time stale-capture"


def test_partial_evidence_is_pairA_only_three_seeds(cell):
    pe = cell["partial_evidence"]
    assert pe["covered_class_sets"] == [PAIR_A]      # pair A only
    assert pe["n_seeds"] == 3
    assert [s["seed"] for s in pe["per_seed"]] == [0, 1, 2]
    assert all(s["class_set"] == PAIR_A for s in pe["per_seed"])
    assert all(s["probe_policy"] == PROBE_POLICY for s in pe["per_seed"])


def test_partial_evidence_copies_committed_governance(cell):
    # The label is exactly the committed PR-3c governance numbers — measured,
    # not invented, and with no geometric field.
    for seed, stem in enumerate(STALE_SOFT_ARMS):
        gov = json.loads((PR3C / f"{stem}.governance.json").read_text())
        got = cell["partial_evidence"]["per_seed"][seed]
        assert got["stale_wrong"] == gov["none"]["stale_wrong"]
        assert got["wrong_total"] == gov["none"]["wrong_none"]
        mct = gov[PROBE_POLICY]
        assert got["frozen_probe_stale_abstained"] == mct["stale_wrong_abstained"]
        assert got["frozen_probe_stale_fixed"] == mct["stale_wrong_fixed"]
        assert got["frozen_probe_false_abstain"] == mct["abstain_on_correct"]
        assert got["frozen_probe_broken"] == mct["broken"]
        assert got["merge_suspect_events"] == gov["_router"]["n_merge_suspect_events"]
        assert got["read_time_witness_probes"] == mct["witness_probes"]


def test_s0_anchor_numbers(cell):
    # Explicit anchor so a silent artifact swap is caught (PR3C_RESULT.md §2).
    s0 = cell["partial_evidence"]["per_seed"][0]
    assert s0["stale_wrong"] == 374
    assert s0["frozen_probe_stale_abstained"] == 374
    assert s0["frozen_probe_stale_fixed"] == 0
    assert s0["frozen_probe_broken"] == 0
    assert s0["merge_suspect_events"] == 192


def test_capture_is_write_time_not_read_time(cell):
    # Every seed: the frozen probe breaks nothing and fixes nothing at read time;
    # the merge-suspect (write-time) trace is what fires. This is the cell's
    # defining property — merge-path stale is write-time-only evidence.
    for s in cell["partial_evidence"]["per_seed"]:
        assert s["frozen_probe_broken"] == 0
        assert s["frozen_probe_stale_fixed"] == 0
        assert s["merge_suspect_events"] == 192
        assert "write-time" in s["capture_via"]


def test_missing_evidence_names_DE_and_is_out_of_scope(cell):
    me = cell["missing_evidence"]
    assert me["uncovered_geometries"] == ["pairD", "pairE"]
    assert "D/E" in me["needed"]
    low = me["why_absent"].lower()
    assert "did not exist" in low and "pr-3c" in low
    # The completing run is named and explicitly fenced out of step-2 scope.
    rr = me["required_run"].lower()
    assert "no new cache runs" in rr or "out of" in rr
    assert "10" in me["required_run"] and "95" in me["required_run"]  # pairD set
    assert "47" in me["required_run"] and "76" in me["required_run"]  # pairE set


def test_DE_geometry_structurally_absent_from_committed_pr3c():
    # The empirical backbone of "cannot seed": no committed PR-3c arm touches the
    # pairD/pairE class sets at all (they were built later, PR-4/PR-5).
    de = set(DE_CLASS_SETS["pairD"]) | set(DE_CLASS_SETS["pairE"])
    for f in PR3C.glob("per_probe_*.summary.json"):
        classes = set(json.loads(f.read_text())["classes"])
        assert not (classes & de), f.name
    # And the merge-path (soft) arm exists only on the pair-A class set.
    ev = read_merge_path_stale_evidence(PR3C)
    assert ev["covered_class_sets"] == [PAIR_A]


def test_no_geometry_token_in_cell(cell):
    # No key anywhere in the cell (incl. partial_evidence) may be a geometric
    # property — the label is measured hazard only, never geometry.
    bad = []

    def scan(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if any(t in k.lower() for t in _GEOMETRY_TOKENS):
                    bad.append(f"{path}/{k}")
                scan(v, f"{path}/{k}")
        elif isinstance(obj, list):
            for x in obj:
                scan(x, path)

    scan(cell)
    assert bad == [], bad


def test_panel_invariants_hold_after_step2():
    panel = build_panel(json.loads(POST.read_text()), pr3c_dir=PR3C)
    assert panel["geometry_used_as_gate"] is False
    assert panel["new_cache_runs"] == 0
    assert panel["engine_or_retrieval_change"] is False
    # The cell's status keeps it out of the seeded set.
    assert panel["cell_status_summary"]["required_unseeded"] == ["merge_path_stale"]
    assert "merge_path_stale" not in panel["cell_status_summary"]["seeded"]


def test_committed_panel_matches_fresh_build():
    committed = json.loads(PANEL.read_text())["cells"]["merge_path_stale"]
    assert committed == build_merge_path_stale_cell()


def test_build_is_deterministic():
    assert build_merge_path_stale_cell() == build_merge_path_stale_cell()


def test_soft_payload_guard(tmp_path):
    # The reader refuses any arm that is not the soft merge path, so a wrong
    # artifact cannot silently seed the cell.
    gov = json.loads((PR3C / f"{STALE_SOFT_ARMS[0]}.governance.json").read_text())
    summ = json.loads((PR3C / f"{STALE_SOFT_ARMS[0]}.summary.json").read_text())
    for stem in STALE_SOFT_ARMS:
        (tmp_path / f"{stem}.governance.json").write_text(json.dumps(gov))
        (tmp_path / f"{stem}.summary.json").write_text(json.dumps(summ))
    # Valid as copied:
    assert read_merge_path_stale_evidence(tmp_path)["covered_class_sets"] == [PAIR_A]
    # Tamper one arm to a non-soft payload → refuse.
    bad = dict(summ, payload_mode="hard")
    (tmp_path / f"{STALE_SOFT_ARMS[1]}.summary.json").write_text(json.dumps(bad))
    with pytest.raises(RuntimeError):
        read_merge_path_stale_evidence(tmp_path)
