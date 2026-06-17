"""PR-6 step 1 pins — the empirical-hazard panel scaffold.

Pins (a) that the panel names all five required benchmark cell types, (b)
that the seeded cells copy the MEASURED frozen-probe hazard from the
committed PR-5 post-mortem faithfully and with no geometric input, (c) that
merge-path stale and one-shot ambiguity stay required-but-unseeded /
observe-only, and (d) that the committed manifest matches a fresh build — so
the scaffold's scope boundaries (no geometry gate, no new runs, no engine or
retrieval change) cannot drift after the fact.

Reads only committed artifacts; imports no torch and touches no cache.
"""
import json
from pathlib import Path

import pytest

from benchmarks.pr6_hazard_panel import (
    PROBE_POLICY, REQUIRED_CELL_TYPES, SEEDED_CELLS, build_panel, seed_label)

ROOT = Path(__file__).resolve().parent.parent
POST = ROOT / "results/issue_failure_mode_blindness/pr5/hazard_postmortem.json"
PANEL = ROOT / "results/issue_failure_mode_blindness/pr6/panel.json"

_GEOMETRY_TOKENS = ("cos", "ratio", "confusion", "geometry", "centroid",
                    "attribution", "fork_pair")


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    # build_panel reads the committed PR-3c stale arms (PR-6 step 2) via repo-
    # relative paths; pin cwd so the manifest's relative artifact paths match.
    monkeypatch.chdir(ROOT)


@pytest.fixture
def post():
    return json.loads(POST.read_text())


@pytest.fixture
def panel(post):
    return build_panel(post)


def test_required_cell_types_are_the_five(panel):
    assert REQUIRED_CELL_TYPES == (
        "clean_control", "direct_harm", "collateral_harm",
        "merge_path_stale", "one_shot_ambiguity")
    for cell in REQUIRED_CELL_TYPES:
        assert cell in panel["cells"], cell
        assert panel["cells"][cell]["required"] is True
    assert panel["required_cell_types"] == list(REQUIRED_CELL_TYPES)


def test_cell_role_assignment(panel):
    def pairs(cell):
        return [s["pair"] for s in panel["cells"][cell]["seeds"]]
    assert pairs("clean_control") == ["pairA", "pairC"]
    assert pairs("direct_harm") == ["pairD"]
    assert pairs("collateral_harm") == ["pairB", "pairE"]
    assert panel["cells"]["direct_harm"]["harm_shape"] == "direct"
    assert panel["cells"]["collateral_harm"]["harm_shape"] == "collateral"
    assert panel["cells"]["clean_control"]["harm_shape"] == "none"


def test_seeded_labels_copy_measured_hazard(post, panel):
    # The label is exactly the committed post-mortem's measured numbers.
    for cell, spec in SEEDED_CELLS.items():
        for pair in spec["pairs"]:
            got = next(s for s in panel["cells"][cell]["seeds"]
                       if s["pair"] == pair)
            assert got == seed_label(post, pair)
            gt = post["hazard_ground_truth"][pair]
            assert got["broken_mean"] == gt["broken_mean"]
            assert got["broken_by_seed"] == gt["broken_by_seed"]
            assert got["probe_policy"] == PROBE_POLICY


def test_pairD_direct_cell_numbers(panel):
    d = next(s for s in panel["cells"]["direct_harm"]["seeds"]
             if s["pair"] == "pairD")
    assert d["broken_mean"] == 271.67
    assert d["broken_total_3seed"] == 815
    assert d["direct_total_3seed"] == 533
    assert d["collateral_total_3seed"] == 282
    assert d["worst_true_class"] == 28
    assert d["worst_class_is_bystander"] is True


def test_both_harm_components_recorded_and_consistent(panel):
    # Both direct and collateral are recorded per seed (PR6_DESIGN §5: both
    # harm shapes are required) and the decomposition is internally consistent.
    for cell in ("clean_control", "direct_harm", "collateral_harm"):
        for s in panel["cells"][cell]["seeds"]:
            assert "direct_total_3seed" in s and "collateral_total_3seed" in s
            assert (s["direct_total_3seed"] + s["collateral_total_3seed"]
                    == s["broken_total_3seed"])


def test_merge_path_stale_seeded_after_step3(panel):
    # Step 3 measured the D/E (+pair-B) merge-path stale arms; the cell seeds.
    cell = panel["cells"]["merge_path_stale"]
    assert cell["required"] is True
    assert cell["status"] == "seeded"
    assert [s["pair"] for s in cell["seeds"]] == ["pairA", "pairD", "pairE", "pairB"]
    assert cell["missing_evidence"] is None          # residual pair-B drained
    # The cell's required label — the D/E degradation — is now measured.
    means = cell["measured_degradation"]["frozen_probe_broken_mean_by_pair"]
    assert means["pairD"] > means["pairE"] > means["pairA"]


def test_one_shot_ambiguity_observe_only(panel):
    cell = panel["cells"]["one_shot_ambiguity"]
    assert cell["required"] is True
    assert cell["status"] == "observe_only"
    assert cell["seeds"] == []
    assert cell["seed_artifact"] is None
    assert "provenance" in cell["additional_runs_needed"]


def test_no_geometry_used_as_gate(panel):
    assert panel["geometry_used_as_gate"] is False
    # Step 3 committed 9 new soft-payload stale arms (the measurand); the panel
    # BUILD still reads only committed JSON and changes no engine/retrieval code.
    assert panel["new_cache_runs"] == 9
    assert panel["engine_or_retrieval_change"] is False
    assert panel["scaffold"] is True
    # No seed label may carry a geometric field — the label is hazard only.
    for cell in panel["cells"].values():
        for s in cell["seeds"]:
            for key in s:
                assert not any(tok in key.lower() for tok in _GEOMETRY_TOKENS), key


def test_screening_forbids_geometry(panel):
    sp = panel["screening_procedure"]
    assert "geometry" in sp["geometry_forbidden"].lower()
    assert "measured" in sp["admit_cell_iff"].lower()
    assert sp["scope_statement"]
    assert "both" in sp["both_harm_shapes_required"].lower()


def test_inputs_are_postmortem_pr3c_and_stale_de_arms(panel):
    # Step 1 read only the post-mortem; step 2 added the committed PR-3c pair-A
    # soft-payload stale arms; step 3 adds the measured D/E + pair-B arms.
    base = "results/issue_failure_mode_blindness"
    assert panel["inputs_used"] == [
        f"{base}/pr5/hazard_postmortem.json",
        f"{base}/pr3c/per_probe_stale-soft_s0.governance.json",
        f"{base}/pr3c/per_probe_stale-soft_s1.governance.json",
        f"{base}/pr3c/per_probe_stale-soft_s2.governance.json",
        f"{base}/pr6/stale_de/per_probe_stale-soft_s0_pairD.governance.json",
        f"{base}/pr6/stale_de/per_probe_stale-soft_s1_pairD.governance.json",
        f"{base}/pr6/stale_de/per_probe_stale-soft_s2_pairD.governance.json",
        f"{base}/pr6/stale_de/per_probe_stale-soft_s0_pairE.governance.json",
        f"{base}/pr6/stale_de/per_probe_stale-soft_s1_pairE.governance.json",
        f"{base}/pr6/stale_de/per_probe_stale-soft_s2_pairE.governance.json",
        f"{base}/pr6/stale_de/per_probe_stale-soft_s0_pairB.governance.json",
        f"{base}/pr6/stale_de/per_probe_stale-soft_s1_pairB.governance.json",
        f"{base}/pr6/stale_de/per_probe_stale-soft_s2_pairB.governance.json",
    ]


def test_probe_policy_guard(post):
    bad = dict(post)
    bad["probe_policy"] = "some-deployment-candidate"
    with pytest.raises(RuntimeError):
        build_panel(bad)


def test_build_is_deterministic(post):
    assert build_panel(post) == build_panel(post)


def test_committed_manifest_matches_fresh_build(panel):
    committed = json.loads(PANEL.read_text())
    assert committed == panel
