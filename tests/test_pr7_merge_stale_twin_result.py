"""PR-7 step 4 pins — the committed pairD merge_path_stale annotate stress twin.

Step 3 proved the annotate null-action floor costs nothing on the clean arm,
where it stamped ZERO events. Step 4 is the real stress: the soft-payload
("merge-path") stale arm on pairD HAS supersession (merge-suspect absorb)
events, so `--govern annotate` actually stamps them (annotated_events > 0) — and
the floor must STILL change nothing the writer commits (GovernanceHook.decide()
returns ALLOW regardless). These pins fix that committed result:

  * the floor held UNDER LOAD — a zero delta on every measurand across all 3
    seeds even though annotate stamped 192 merge-suspect events/seed, so
    merge_path_stale is a ``pass`` (PR7_DESIGN §4/§8);
  * the cell's ``capture_stable`` guard held — write-time merge-suspect capture
    stayed 192/seed in both arms (the PR-6 panel requirement) and read-time
    damage did not worsen;
  * the stress is real — the annotate summaries record annotated_events == 192
    (the floor acted), while the none baseline carries no governance at all;
  * boundaries held — engine byte-frozen, geometry never a gate, soft payload
    enforced as a provenance requirement on this cell.

Reads only committed artifacts; imports no torch and touches no cache.
"""
import json
from pathlib import Path

import pytest

from benchmarks.pr7_twin_delta import BASELINE_GOVERN

ROOT = Path(__file__).resolve().parent.parent
DELTA = ROOT / "results/issue_failure_mode_blindness/pr7/twin_delta.json"
TWIN = ROOT / "results/issue_failure_mode_blindness/pr7/twin"
CELL = TWIN / "merge_path_stale"

EXPECTED_CAPTURE = 192  # PR-6 panel: write-time merge-suspect capture / seed.


@pytest.fixture
def delta():
    return json.loads(DELTA.read_text())


def test_merge_path_stale_is_zero_delta_pass(delta):
    assert "merge_path_stale" in delta["scored_cells"]
    assert delta["overall_verdict"] == "pass"
    assert delta["both_shapes_ok"] is True
    cell = delta["cells"]["merge_path_stale"]
    assert cell["verdict"] == "pass"
    assert cell["guard"] == "capture_stable"
    assert cell["payload_mode"] == "soft"
    assert cell["frozen_probe_broken_delta_total"] == 0
    assert cell["merge_suspect_capture_delta_total"] == 0
    assert cell["improved"] is False
    assert cell["regressed"] is False
    assert cell["capture_stable"] is True


def test_pairD_floor_holds_under_load(delta):
    # pairD is the run geometry (pairA/pairE/pairB carry no governed stale arm
    # and stay unscored within the cell).
    pairD = delta["cells"]["merge_path_stale"]["per_pair"]["pairD"]
    assert pairD["baseline_present"] and pairD["governed_present"]
    assert pairD["seeds_scored"] == [0, 1, 2]
    # the null-action floor: every measurand's baseline-minus-governed delta is
    # exactly zero on every seed, despite annotate stamping events this time.
    for seed_delta in pairD["delta_by_seed"].values():
        assert all(v == 0 for v in seed_delta.values())
    # write-time merge-suspect capture held 192/seed in BOTH arms (the cell's
    # capture_stable requirement) and read-time broken matched arm-to-arm.
    for seed in ("0", "1", "2"):
        base = pairD["baseline_by_seed"][seed]
        gov = pairD["governed_by_seed"][seed]
        assert base["merge_suspect_events"] == EXPECTED_CAPTURE
        assert gov["merge_suspect_events"] == EXPECTED_CAPTURE
        assert base["frozen_probe_broken"] == gov["frozen_probe_broken"]


def test_annotate_actually_acted_but_changed_nothing():
    # the stress: on the stale arm annotate stamps the supersession events
    # (annotated_events > 0, unlike the step-3 clean arm's 0), yet every scored
    # artifact stays byte-identical (pinned in test_core_artifacts_byte_identical).
    gov = CELL / "annotate"
    for seed in (0, 1, 2):
        summ = json.loads(
            (gov / f"per_probe_stale-soft_s{seed}_pairD.summary.json").read_text())
        assert summ["govern"]["action"] == "annotate"
        assert summ["govern"]["annotated_events"] == EXPECTED_CAPTURE
        assert summ["classes"] == [10, 28, 32, 95]
        assert summ["payload_mode"] == "soft"


def test_baseline_arm_carries_no_governance():
    base = CELL / BASELINE_GOVERN
    for seed in (0, 1, 2):
        summ = json.loads(
            (base / f"per_probe_stale-soft_s{seed}_pairD.summary.json").read_text())
        assert "govern" not in summ
        assert summ["classes"] == [10, 28, 32, 95]
        assert summ["payload_mode"] == "soft"


def test_core_artifacts_byte_identical():
    # the null-action floor: every scored artifact is byte-identical between the
    # none baseline and the annotate arm (only the summary's govern block, tested
    # above, differs). topk is gzipped, so compare its decompressed bytes.
    import gzip

    base, gov = CELL / BASELINE_GOVERN, CELL / "annotate"
    for seed in (0, 1, 2):
        stem = f"per_probe_stale-soft_s{seed}_pairD"
        for ext in ("csv", "governance.json", "per_slot.csv", "fork_events.csv"):
            assert (base / f"{stem}.{ext}").read_bytes() \
                == (gov / f"{stem}.{ext}").read_bytes(), f"{stem}.{ext} differs"
        assert gzip.decompress((base / f"{stem}.topk.csv.gz").read_bytes()) \
            == gzip.decompress((gov / f"{stem}.topk.csv.gz").read_bytes()), \
            f"{stem}.topk content differs"
