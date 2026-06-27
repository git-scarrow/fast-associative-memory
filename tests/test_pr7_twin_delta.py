"""tests/test_pr7_twin_delta.py — PR-7 step 2: the write-path twin-run delta
analyzer (PR7_DESIGN.md §11 test 4 + §11 test 6 provenance guards).

These are written and green BEFORE any cache run (PR7_DESIGN §11). The analyzer
reads only committed JSON and imports no torch, so every case here drives it
over synthetic governance.json / summary.json fixtures laid out in the exact
twin-run directory shape (``pr7/twin/<cell>/<govern>/per_probe_<arm>_s{seed}_
<pair>.{summary,governance}.json``). Pinned behaviors:

  * analyzer discipline — imports no torch; ``geometry_used_as_gate`` and
    ``engine_or_retrieval_change`` are False (§11.4);
  * annotate FLOOR — a governed arm byte-equal to baseline yields a zero delta
    on every measurand and a ``pass`` (the null-action floor costs nothing);
  * the §8 BOTH-SHAPES rule — a governed result that fixes ``direct_harm`` while
    worsening ``collateral_harm`` is reported ``fail`` (§11.4);
  * ``one_shot_ambiguity`` is NEVER scored pass/fail (§11.4);
  * graceful baseline-only fallback when no governed arms are committed (§11.4);
  * provenance guards — stem ↔ classes, stem ↔ govern, baseline-carries-no-
    governance, and a non-soft payload refused on the merge-path cell (§11.6).
"""
import json
from pathlib import Path

import pytest

from benchmarks.pr7_twin_delta import (
    BASELINE_GOVERN, CELL_SPEC, PAIR_CLASS_SETS, build_twin_delta, read_arm,
    score_cell)

ROOT = Path(__file__).resolve().parent.parent
MODULE = ROOT / "benchmarks" / "pr7_twin_delta.py"

_ARM_NAME = {"clean_control": "clean", "direct_harm": "mixed",
             "collateral_harm": "mixed", "merge_path_stale": "stale-soft"}


def _measurands(*, broken, stale_wrong=0, collateral=0, direct=0, acted=0,
                conflict_pairs=0, merge_suspect=192):
    """One arm-seed's governance.json, in the frozen-scorer key shape the
    analyzer reads (none = deployed vote, mct = frozen probe, _router = trace)."""
    probe = {"broken": broken, "collateral_br": collateral, "direct_br": direct,
             "acted": acted, "stale_wrong": stale_wrong, "wrong_none": 0}
    return {
        "none": dict(probe, stale_wrong=stale_wrong),
        "mode-conditioned-trust": probe,
        "_router": {"n_conflict_pairs": conflict_pairs,
                    "n_merge_suspect_events": merge_suspect,
                    "final_epoch_verdicts": {}},
    }


def _write_arm(twin_root, cell, govern, pair, *, payload_mode=None,
               classes=None, govern_block=True, **measure):
    """Lay down all 3 seeds of one (cell, govern, pair) arm as committed JSON."""
    arm = _ARM_NAME[cell]
    classes = PAIR_CLASS_SETS[pair] if classes is None else classes
    d = twin_root / cell / govern
    d.mkdir(parents=True, exist_ok=True)
    for seed in (0, 1, 2):
        stem = f"per_probe_{arm}_s{seed}_{pair}"
        summ = {"arm": arm, "classes": classes, "seed": seed,
                "payload_mode": payload_mode}
        if govern != BASELINE_GOVERN and govern_block:
            summ["govern"] = {"action": govern, "step": f"pr7-{govern}"}
        (d / f"{stem}.summary.json").write_text(json.dumps(summ))
        (d / f"{stem}.governance.json").write_text(
            json.dumps(_measurands(**measure)))


def _twin(tmp_path):
    return tmp_path / "pr7" / "twin"


# ---------------------------------------------------------------------------
# Analyzer discipline (§11.4)
# ---------------------------------------------------------------------------
def test_module_imports_no_torch():
    src = MODULE.read_text()
    assert "import torch" not in src and "from torch" not in src


def test_geometry_and_engine_flags_are_false(tmp_path):
    d = build_twin_delta("annotate", _twin(tmp_path), repo_root=ROOT)
    assert d["geometry_used_as_gate"] is False
    assert d["engine_or_retrieval_change"] is False
    assert d["deployed_engine_sha256_parity"] is True  # repo engine unchanged


def test_baseline_govern_action_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="none"):
        build_twin_delta("none", _twin(tmp_path), repo_root=ROOT)


# ---------------------------------------------------------------------------
# Graceful baseline-only fallback (§11.4)
# ---------------------------------------------------------------------------
def test_no_governed_arms_is_baseline_only(tmp_path):
    d = build_twin_delta("annotate", _twin(tmp_path), repo_root=ROOT)
    assert d["overall_verdict"] == "baseline_only"
    assert d["scored_cells"] == []
    assert d["both_shapes_ok"] is True
    assert d["new_cache_runs"] == 0
    # Every real cell is inconclusive; one-shot stays observe-only.
    for cell in CELL_SPEC:
        assert d["cells"][cell]["verdict"] == "inconclusive"
    assert d["cells"]["one_shot_ambiguity"]["verdict"] == "observe_only"


# ---------------------------------------------------------------------------
# annotate floor: zero delta → pass (§11.4 / §4 / §8.4)
# ---------------------------------------------------------------------------
def test_annotate_floor_zero_delta_passes(tmp_path):
    tw = _twin(tmp_path)
    # Baseline and annotate arms byte-equal in every measurand → zero delta.
    _write_arm(tw, "clean_control", "none", "pairA", broken=16, acted=4)
    _write_arm(tw, "clean_control", "annotate", "pairA", broken=16, acted=4)
    d = build_twin_delta("annotate", tw, repo_root=ROOT)

    cell = d["cells"]["clean_control"]
    assert cell["frozen_probe_broken_delta_total"] == 0
    assert cell["verdict"] == "pass"
    assert cell["improved"] is False and cell["regressed"] is False
    # Every per-seed delta is exactly zero across measurands.
    for seed_delta in cell["per_pair"]["pairA"]["delta_by_seed"].values():
        assert set(seed_delta.values()) == {0}
    assert d["overall_verdict"] == "pass"
    assert d["both_shapes_ok"] is True
    assert d["new_cache_runs"] == 3


def test_annotate_increasing_harm_fails_floor(tmp_path):
    """If annotate ever cost something (governed broken > baseline), the floor is
    violated and the clean cell must report fail (worsen_tol 0)."""
    tw = _twin(tmp_path)
    _write_arm(tw, "clean_control", "none", "pairA", broken=16)
    _write_arm(tw, "clean_control", "annotate", "pairA", broken=20)  # +4 harm
    d = build_twin_delta("annotate", tw, repo_root=ROOT)
    cell = d["cells"]["clean_control"]
    assert cell["frozen_probe_broken_delta_total"] == -12  # 3 seeds x (16-20)
    assert cell["regressed"] is True
    assert cell["verdict"] == "fail"
    assert d["overall_verdict"] == "fail"


# ---------------------------------------------------------------------------
# The §8 both-shapes rule (§11.4)
# ---------------------------------------------------------------------------
def test_both_shapes_rule_fix_direct_worsen_collateral_fails(tmp_path):
    tw = _twin(tmp_path)
    # direct_harm: governed reduces broken (a fix).
    _write_arm(tw, "direct_harm", "none", "pairD", broken=100, direct=100)
    _write_arm(tw, "direct_harm", "quarantine", "pairD", broken=50, direct=50)
    # collateral_harm: governed increases broken (a regression on the OTHER shape).
    _write_arm(tw, "collateral_harm", "none", "pairB", broken=10, collateral=10)
    _write_arm(tw, "collateral_harm", "quarantine", "pairB", broken=30, collateral=30)

    d = build_twin_delta("quarantine", tw, repo_root=ROOT)
    assert d["cells"]["direct_harm"]["improved"] is True
    assert d["cells"]["direct_harm"]["verdict"] == "pass"
    assert d["cells"]["collateral_harm"]["regressed"] is True
    assert d["cells"]["collateral_harm"]["verdict"] == "fail"
    # Improving one shape while worsening another → fail, full stop.
    assert d["both_shapes_ok"] is False
    assert d["overall_verdict"] == "fail"
    assert d["both_shapes_detail"]["improved_harm_shapes"] == ["direct"]
    assert d["both_shapes_detail"]["regressed_harm_shapes"] == ["collateral"]


def test_clean_direct_improvement_alone_passes(tmp_path):
    """Improving direct with NO regression elsewhere passes the both-shapes rule."""
    tw = _twin(tmp_path)
    _write_arm(tw, "direct_harm", "none", "pairD", broken=100, direct=100)
    _write_arm(tw, "direct_harm", "quarantine", "pairD", broken=40, direct=40)
    d = build_twin_delta("quarantine", tw, repo_root=ROOT)
    assert d["cells"]["direct_harm"]["verdict"] == "pass"
    assert d["both_shapes_ok"] is True
    assert d["overall_verdict"] == "pass"


# ---------------------------------------------------------------------------
# one-shot ambiguity is never scored pass/fail (§11.4)
# ---------------------------------------------------------------------------
def test_one_shot_never_scored(tmp_path):
    tw = _twin(tmp_path)
    _write_arm(tw, "clean_control", "none", "pairA", broken=16)
    _write_arm(tw, "clean_control", "annotate", "pairA", broken=16)
    d = build_twin_delta("annotate", tw, repo_root=ROOT)
    one = d["cells"]["one_shot_ambiguity"]
    assert one["verdict"] == "observe_only"
    assert "one_shot_ambiguity" not in d["scored_cells"]


# ---------------------------------------------------------------------------
# merge-path capture stability (§7 / §8.4)
# ---------------------------------------------------------------------------
def test_merge_path_capture_regression_fails(tmp_path):
    tw = _twin(tmp_path)
    _write_arm(tw, "merge_path_stale", "none", "pairA",
               payload_mode="soft", broken=0, merge_suspect=192)
    _write_arm(tw, "merge_path_stale", "annotate", "pairA",
               payload_mode="soft", broken=0, merge_suspect=180)  # capture dropped
    d = build_twin_delta("annotate", tw, repo_root=ROOT)
    cell = d["cells"]["merge_path_stale"]
    assert cell["merge_suspect_capture_delta_total"] == 36  # 3 x (192-180)
    assert cell["capture_stable"] is False
    assert cell["verdict"] == "fail"


def test_merge_path_capture_stable_zero_delta_passes(tmp_path):
    tw = _twin(tmp_path)
    _write_arm(tw, "merge_path_stale", "none", "pairA",
               payload_mode="soft", broken=0, merge_suspect=192)
    _write_arm(tw, "merge_path_stale", "annotate", "pairA",
               payload_mode="soft", broken=0, merge_suspect=192)
    d = build_twin_delta("annotate", tw, repo_root=ROOT)
    cell = d["cells"]["merge_path_stale"]
    assert cell["capture_stable"] is True
    assert cell["verdict"] == "pass"


# ---------------------------------------------------------------------------
# Provenance guards (§11.6) — integrity, never a geometric gate
# ---------------------------------------------------------------------------
def test_stem_classes_mismatch_raises(tmp_path):
    tw = _twin(tmp_path)
    # pairA dir but the summary claims pairD's class set → integrity failure.
    _write_arm(tw, "clean_control", "none", "pairA",
               classes=[10, 28, 32, 95], broken=16)
    with pytest.raises(RuntimeError, match="classes"):
        read_arm("clean_control", "none", "pairA", tw)


def test_governed_arm_missing_govern_block_raises(tmp_path):
    tw = _twin(tmp_path)
    _write_arm(tw, "clean_control", "annotate", "pairA",
               broken=16, govern_block=False)  # governed dir, no govern provenance
    with pytest.raises(RuntimeError, match="govern"):
        read_arm("clean_control", "annotate", "pairA", tw)


def test_baseline_arm_with_govern_block_raises(tmp_path):
    tw = _twin(tmp_path)
    d = tw / "clean_control" / "none"
    d.mkdir(parents=True)
    stem = "per_probe_clean_s0_pairA"
    summ = {"arm": "clean", "classes": PAIR_CLASS_SETS["pairA"], "seed": 0,
            "payload_mode": None, "govern": {"action": "annotate"}}
    (d / f"{stem}.summary.json").write_text(json.dumps(summ))
    (d / f"{stem}.governance.json").write_text(json.dumps(_measurands(broken=1)))
    with pytest.raises(RuntimeError, match="ungoverned"):
        read_arm("clean_control", "none", "pairA", tw)


def test_merge_path_refuses_nonsoft_payload(tmp_path):
    tw = _twin(tmp_path)
    _write_arm(tw, "merge_path_stale", "none", "pairA",
               payload_mode="onehot", broken=0)  # not the soft merge path
    with pytest.raises(RuntimeError, match="payload_mode"):
        read_arm("merge_path_stale", "none", "pairA", tw)


def test_partial_seeds_treated_absent(tmp_path):
    """An arm missing any seed is treated absent (pr6's all-seeds-or-none rule),
    so a half-committed governed arm cannot produce a spurious verdict."""
    tw = _twin(tmp_path)
    _write_arm(tw, "clean_control", "none", "pairA", broken=16)
    d = tw / "clean_control" / "annotate"
    d.mkdir(parents=True)
    # only seed 0 committed for the governed arm
    stem = "per_probe_clean_s0_pairA"
    summ = {"arm": "clean", "classes": PAIR_CLASS_SETS["pairA"], "seed": 0,
            "payload_mode": None, "govern": {"action": "annotate"}}
    (d / f"{stem}.summary.json").write_text(json.dumps(summ))
    (d / f"{stem}.governance.json").write_text(json.dumps(_measurands(broken=16)))
    assert read_arm("clean_control", "annotate", "pairA", tw) is None
    cell = score_cell("clean_control", "annotate", tw)
    assert cell["verdict"] == "inconclusive"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_build_is_deterministic(tmp_path):
    tw = _twin(tmp_path)
    _write_arm(tw, "clean_control", "none", "pairA", broken=16)
    _write_arm(tw, "clean_control", "annotate", "pairA", broken=16)
    assert build_twin_delta("annotate", tw, repo_root=ROOT) == \
        build_twin_delta("annotate", tw, repo_root=ROOT)
