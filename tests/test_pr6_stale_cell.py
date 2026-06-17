"""PR-6 step 3 pins — seeding merge_path_stale with the measured D/E (+B) arms.

Step 2 left the cell ``required_unseeded`` because the only committed merge-path
arm (PR-3c soft-payload) was run on pair-A alone. Step 3 measured the dedicated
soft-payload stale arm on the geometries PR-3c never ran — pairD/pairE (the
required D/E component) and pairB (the step-2 residual) — 3 seeds each, scored
by the SAME frozen mode-conditioned-trust probe (committed under
``pr6/stale_de/``). These pins assert:

  (a) the cell is now ``seeded`` with the measured per-geometry labels, copied
      faithfully (no geometric input) from the committed governance JSON;
  (b) write-time merge-suspect capture is GEOMETRY-STABLE (192 events/seed on
      every arm) — the cell's defining write-time-only property;
  (c) the frozen probe's READ-TIME damage (broken) degrades on D/E — measured,
      not asserted — while it fixes ~0 stale-wrong rows on any geometry;
  (d) the residual pair-B note is drained (missing_evidence is None);
  (e) no geometry is used to admit or exclude the cell, and the analyzer falls
      back to required_unseeded when the step-3 arms are absent.

Reads only committed artifacts; imports no torch and touches no cache.
"""
import json
from pathlib import Path

import pytest

from benchmarks.pr6_hazard_panel import (
    DE_CLASS_SETS, PAIR_A_CLASS_SET, PROBE_POLICY, STALE_DE,
    STALE_DE_GEOMETRIES, STALE_GEOMETRY_ORDER, build_merge_path_stale_cell,
    build_panel, read_merge_path_stale_evidence, read_stale_de_evidence)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results/issue_failure_mode_blindness"
PR3C = RESULTS / "pr3c"
STALE_DE_DIR = RESULTS / "pr6/stale_de"
POST = RESULTS / "pr5/hazard_postmortem.json"
PANEL = RESULTS / "pr6/panel.json"

PAIR_A = [0, 8, 19, 33]
_GEOMETRY_TOKENS = ("cos", "ratio", "confusion", "geometry", "centroid",
                    "attribution", "fork_pair")

# Measured frozen-probe read-time damage (broken) per geometry, per seed — the
# explicit anchor so a silent artifact swap is caught. pairA is the committed
# PR-3c reference (broken 0); D/E degrade; pairB is benign like pairA.
BROKEN_BY_SEED = {
    "pairA": [0, 0, 0],
    "pairD": [112, 180, 46],
    "pairE": [74, 40, 24],
    "pairB": [0, 1, 0],
}
STALE_FIXED_TOTAL = {"pairA": 0, "pairD": 0, "pairE": 1, "pairB": 2}


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    # The builder resolves committed artifacts by repo-relative path.
    monkeypatch.chdir(ROOT)


@pytest.fixture
def cell():
    return build_merge_path_stale_cell()


def _gov(directory, stem):
    return json.loads((directory / f"{stem}.governance.json").read_text())


def _committed_stem(pair, seed):
    """The committed governance directory + stem for a (geometry, seed)."""
    if pair == "pairA":
        return PR3C, f"per_probe_stale-soft_s{seed}"
    return STALE_DE_DIR, f"per_probe_stale-soft_s{seed}_{pair}"


# --- (a) the cell is seeded -------------------------------------------------

def test_cell_is_seeded(cell):
    assert cell["required"] is True
    assert cell["status"] == "seeded"
    assert cell["evidence_status"] == "measured_pairA_pairD_pairE_pairB"
    assert cell["harm_shape"] == "write-time stale-capture"
    assert cell["seeds"]                                   # non-empty now


def test_seeds_in_canonical_order_and_cover_DE(cell):
    pairs = [s["pair"] for s in cell["seeds"]]
    assert pairs == list(STALE_GEOMETRY_ORDER)            # pairA, D, E, B
    by_pair = {s["pair"]: s for s in cell["seeds"]}
    assert by_pair["pairA"]["class_set"] == PAIR_A
    assert by_pair["pairD"]["class_set"] == DE_CLASS_SETS["pairD"]
    assert by_pair["pairE"]["class_set"] == DE_CLASS_SETS["pairE"]
    for s in cell["seeds"]:
        assert s["n_seeds"] == 3
        assert s["probe_policy"] == PROBE_POLICY


# --- (b) write-time capture is geometry-stable ------------------------------

def test_write_time_capture_geometry_stable(cell):
    # Every geometry, every seed: the merge-suspect (write-time) trace fires 192
    # events — the capture mechanism is intact regardless of geometry.
    for s in cell["seeds"]:
        assert s["merge_suspect_events_by_seed"] == [192, 192, 192], s["pair"]
        for ps in s["per_seed"]:
            assert ps["merge_suspect_events"] == 192
            assert "write-time" in ps["capture_via"]


# --- (c) read-time damage degrades on D/E (measured) ------------------------

def test_DE_read_time_degradation_measured(cell):
    by_pair = {s["pair"]: s for s in cell["seeds"]}
    for pair, expected in BROKEN_BY_SEED.items():
        assert by_pair[pair]["frozen_probe_broken_by_seed"] == expected, pair
        assert by_pair[pair]["frozen_probe_stale_fixed_total"] == \
            STALE_FIXED_TOTAL[pair], pair
    # The degradation is real and ordered: D worst, then E, with B/A benign.
    means = cell["measured_degradation"]["frozen_probe_broken_mean_by_pair"]
    assert means["pairD"] > means["pairE"] > means["pairB"]
    assert means["pairA"] == 0.0
    assert means["pairD"] == 112.67 and means["pairE"] == 46.0
    # Read-time "fixes" are negligible on every geometry (write-time-only).
    assert all(v <= 2 for v in
               cell["measured_degradation"]["frozen_probe_stale_fixed_total_by_pair"].values())


def test_measured_degradation_summary_states_the_finding(cell):
    summ = cell["measured_degradation"]["summary"].lower()
    assert "write-time" in summ and "read-time" in summ
    assert "degrade" in summ and "d/e" in summ
    assert "192" in summ


# --- faithfulness: labels are the committed numbers, not invented -----------

def test_seeds_copy_committed_governance(cell):
    by_pair = {s["pair"]: s for s in cell["seeds"]}
    for pair in STALE_GEOMETRY_ORDER:
        group = by_pair[pair]
        for seed in (0, 1, 2):
            directory, stem = _committed_stem(pair, seed)
            gov = _gov(directory, stem)
            got = group["per_seed"][seed]
            assert got["stale_wrong"] == gov["none"]["stale_wrong"]
            assert got["wrong_total"] == gov["none"]["wrong_none"]
            mct = gov[PROBE_POLICY]
            assert got["frozen_probe_broken"] == mct["broken"]
            assert got["frozen_probe_stale_abstained"] == mct["stale_wrong_abstained"]
            assert got["frozen_probe_stale_fixed"] == mct["stale_wrong_fixed"]
            assert got["frozen_probe_false_abstain"] == mct["abstain_on_correct"]
            assert got["read_time_witness_probes"] == mct["witness_probes"]
            assert got["merge_suspect_events"] == gov["_router"]["n_merge_suspect_events"]


def test_pairA_anchor_preserved(cell):
    # The step-2 pair-A anchor still holds, now inside seeds[pairA] (PR3C_RESULT §2).
    a = next(s for s in cell["seeds"] if s["pair"] == "pairA")
    s0 = a["per_seed"][0]
    assert s0["stale_wrong"] == 374
    assert s0["frozen_probe_stale_abstained"] == 374
    assert s0["frozen_probe_stale_fixed"] == 0
    assert s0["frozen_probe_broken"] == 0
    assert s0["merge_suspect_events"] == 192


# --- (d) residual pair-B note drained ---------------------------------------

def test_missing_evidence_drained(cell):
    assert cell["missing_evidence"] is None
    assert "none" in cell["additional_runs_needed"].lower()
    assert "pair-b residual drained" in cell["additional_runs_needed"].lower()


# --- (e) no geometry gate; provenance integrity; graceful fallback ----------

def test_no_geometry_token_in_cell(cell):
    # No key anywhere in the cell may be a geometric property — the label is
    # measured hazard only, never geometry (class_set/pair are provenance).
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


def test_DE_geometry_structurally_absent_from_committed_pr3c():
    # The empirical backbone of step 2: no committed PR-3c arm touches the
    # pairD/pairE class sets (built later, PR-4/PR-5); the D/E evidence comes
    # only from the step-3 stale_de arms, never from pr3c.
    de = set(DE_CLASS_SETS["pairD"]) | set(DE_CLASS_SETS["pairE"])
    for f in PR3C.glob("per_probe_*.summary.json"):
        classes = set(json.loads(f.read_text())["classes"])
        assert not (classes & de), f.name
    assert read_merge_path_stale_evidence(PR3C)["covered_class_sets"] == [PAIR_A]


def test_stale_de_evidence_reads_three_geometries():
    ev = read_stale_de_evidence(STALE_DE_DIR)
    assert ev["covered_class_sets"] == [
        STALE_DE_GEOMETRIES["pairD"],
        STALE_DE_GEOMETRIES["pairE"],
        STALE_DE_GEOMETRIES["pairB"],
    ]
    assert len(ev["source_artifacts"]) == 9
    assert set(ev["by_pair"]) == {"pairD", "pairE", "pairB"}


def test_stale_de_provenance_guards(tmp_path):
    # Copy the committed D/E/B arms; the reader accepts them...
    for pair in ("pairD", "pairE", "pairB"):
        for seed in (0, 1, 2):
            stem = f"per_probe_stale-soft_s{seed}_{pair}"
            for suf in ("governance.json", "summary.json"):
                (tmp_path / f"{stem}.{suf}").write_text(
                    (STALE_DE_DIR / f"{stem}.{suf}").read_text())
    assert len(read_stale_de_evidence(tmp_path)["covered_class_sets"]) == 3

    # ...refuses a non-soft payload (wrong arm cannot silently seed)...
    s = json.loads((tmp_path / "per_probe_stale-soft_s1_pairD.summary.json").read_text())
    (tmp_path / "per_probe_stale-soft_s1_pairD.summary.json").write_text(
        json.dumps(dict(s, payload_mode="hard")))
    with pytest.raises(RuntimeError):
        read_stale_de_evidence(tmp_path)

    # ...and refuses a stem whose name disagrees with its measured classes.
    (tmp_path / "per_probe_stale-soft_s1_pairD.summary.json").write_text(
        json.dumps(dict(s, classes=[1, 2, 3, 4])))
    with pytest.raises(RuntimeError):
        read_stale_de_evidence(tmp_path)


def test_unseeded_fallback_when_step3_arms_absent(tmp_path):
    # With no committed step-3 arms the cell degrades gracefully to the step-2
    # state: required_unseeded, pair-A partial evidence, D/E named as missing.
    cell = build_merge_path_stale_cell(stale_de_dir=tmp_path / "nope")
    assert cell["status"] == "required_unseeded"
    assert cell["evidence_status"] == "partial_pairA_only"
    assert cell["seeds"] == []
    assert cell["partial_evidence"]["covered_class_sets"] == [PAIR_A]
    assert cell["missing_evidence"]["uncovered_geometries"] == ["pairD", "pairE"]


# --- panel-level invariants + manifest pinning ------------------------------

def test_panel_invariants_after_step3():
    panel = build_panel(json.loads(POST.read_text()))
    assert panel["geometry_used_as_gate"] is False
    assert panel["engine_or_retrieval_change"] is False
    assert panel["new_cache_runs"] == 9            # the committed step-3 arms
    assert panel["cell_status_summary"]["seeded"] == [
        "clean_control", "collateral_harm", "direct_harm", "merge_path_stale"]
    assert "required_unseeded" not in panel["cell_status_summary"]


def test_committed_panel_matches_fresh_build():
    committed = json.loads(PANEL.read_text())["cells"]["merge_path_stale"]
    assert committed == build_merge_path_stale_cell()


def test_build_is_deterministic():
    assert build_merge_path_stale_cell() == build_merge_path_stale_cell()
