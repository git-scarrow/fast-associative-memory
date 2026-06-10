"""tests/test_score_frozen_detector.py — PR-3a scoring guards.

Pins the two properties that make the PR-3a numbers trustworthy:

  1. the frozen #87 detector is REPRODUCED AND VERIFIED against its
     persisted study summary (both artifacts are git-tracked), and any
     mismatch RAISES instead of silently scoring with a different detector;
  2. score_arm's group/AUC/gate accounting behaves correctly on a tiny
     hand-built arm where every quantity is known.

CPU only; no caches, no GPU, no network — the fit inputs are repo-resident
result files.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.score_frozen_detector import (  # noqa: E402
    FIT_CSV, FIT_SUMMARY, reproduce_frozen_detector, score_arm)

REPO = Path(__file__).resolve().parent.parent


def test_frozen_detector_reproduces_persisted_summary():
    det, thr, prov = reproduce_frozen_detector(
        REPO / FIT_CSV, REPO / FIT_SUMMARY)
    with open(REPO / FIT_SUMMARY) as f:
        summary = json.load(f)
    assert prov["verified_heldout_parity_auc"] == \
        summary["heldout_auc"]["parity_vote"]
    assert round(thr, 6) == \
        summary["operating_point_parity_vote"]["threshold"]
    # the dumped parameters must be sufficient to re-apply the detector
    assert len(prov["logistic_coef"]) == 2
    assert len(prov["standardization_mean"]) == 2


def test_tampered_summary_raises(tmp_path):
    with open(REPO / FIT_SUMMARY) as f:
        summary = json.load(f)
    summary["heldout_auc"]["parity_vote"] += 0.01
    bad = tmp_path / "tampered_summary.json"
    bad.write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="reproduction FAILED"):
        reproduce_frozen_detector(REPO / FIT_CSV, bad)


class _StubDetector:
    """score == top1_top2_margin, so every score_arm quantity is hand-checkable."""

    def score(self, cols, mask):
        return np.asarray(cols["top1_top2_margin"], dtype=float)[mask]


def _row(margin, correct, *, bucket=0.0, c_len=0, s_len=0, cw=0.0, sw=0.0):
    return {"top1_top2_margin": margin, "effective_support": 5.0,
            "vote_correct": correct, "margin_bucket": bucket,
            "contradictory_lenient": c_len, "stale_lenient": s_len,
            "contradictory_strict": c_len, "stale_strict": s_len,
            "contra_vote_weight": cw, "stale_vote_weight": sw}


def test_score_arm_accounting_on_hand_built_rows():
    # 4 correct rows score {.8,.9} in-zone and {.6,.7} out-of-zone;
    # 2 contra-wrong rows score {.1,.2} in-zone; threshold .5 abstains both.
    rows = [
        _row(0.8, 1), _row(0.9, 1),
        _row(0.6, 1, bucket=1.0), _row(0.7, 1, bucket=1.0),
        _row(0.1, 0, c_len=1, cw=0.8), _row(0.2, 0, c_len=1, cw=0.6),
    ]
    out = score_arm(rows, _StubDetector(), thr=0.5)

    assert out["n_probes"] == 6 and out["wrong"] == 2
    zc = out["zone_composition"]
    assert zc["correct"]["n"] == 4 and zc["correct"]["n_forced_zone"] == 2
    assert zc["contra_wrong"]["forced_zone_share"] == 1.0
    # all wrong rows score below all correct rows -> perfectly oriented
    assert out["auc"]["all_rows"]["contra_wrong"] == 1.0
    assert out["auc"]["forced_zone"]["contra_wrong"] == 1.0
    assert out["auc"]["all_rows"]["stale_wrong"] is None
    # no wrong row exceeds the median correct score
    assert out["confidently_wrong"]["all_rows"]["contra_wrong"] == 0.0
    # zoned gate: out-of-zone rows retained unconditionally; both wrong rows
    # (in-zone, score < .5) abstained -> full error capture, no false abstain
    op = out["operating_point"]["zoned"]
    assert op["error_capture_contra_wrong"] == 1.0
    assert op["false_abstain_rate_on_correct"] == 0.0
    assert op["retained_coverage"] == round(4 / 6, 4)
    assert op["retained_accuracy"] == 1.0
    # dose-response: both wrong rows live in the (0.5, 0.75] mass bin
    bins = {b["bin"]: b for b in out["dose_response"]["contra_vote_weight"]}
    assert bins["(0.5,0.75]"]["n"] == 1 and bins["(0.5,0.75]"]["wrong_rate"] == 1.0
    assert bins["(0.75,1.0]"]["n"] == 1 and bins["(0.75,1.0]"]["wrong_rate"] == 1.0
    assert bins["0"]["n"] == 4 and bins["0"]["wrong_rate"] == 0.0


def test_score_arm_inverted_detector_reads_below_chance():
    # wrong rows scoring ABOVE correct rows must yield AUC < 0.5 (inversion
    # is reported, not clipped) and confidently-wrong = 1.0.
    rows = [_row(0.2, 1), _row(0.3, 1),
            _row(0.8, 0, s_len=1, sw=0.5), _row(0.9, 0, s_len=1, sw=0.5)]
    out = score_arm(rows, _StubDetector(), thr=0.5)
    assert out["auc"]["all_rows"]["stale_wrong"] == 0.0
    assert out["confidently_wrong"]["all_rows"]["stale_wrong"] == 1.0
    # gate retains every stale failure: zero error capture
    assert out["operating_point"]["zoned"]["error_capture_stale_wrong"] == 0.0
