#!/usr/bin/env python3
"""Score the frozen #87 two-axis detector per failure mode (PR-3a).

ANALYSIS ONLY. No new runs, no engine change, no detector refit on the
failure-mode data. This closes the pre-registered question from
results/issue_failure_mode_blindness/SCHEMA.md ("Planned analysis (PR-3)"):
per failure mode, what is the two-axis AUC under the FROZEN #87 detector,
the error capture at its frozen operating point, and the confidently-wrong
rate — evaluated against the redesign triggers stated before any PR-2 data
existed:

    CONTRADICTORY two-axis AUC <= 0.60 or confidently-wrong > 50%
        -> write-time contradiction detection required;
    STALE two-axis AUC <= 0.55
        -> recency must enter the readout.

What "frozen" means here. The #87 detector is a PROCEDURE persisted as a
result summary, not a coefficient file: heldout_abstention.py fits
standardization + logistic(rank_gap, manifold_support) on the parity-train
epochs of the #87 study's forced zone and freezes everything thereafter.
This script reproduces that exact fit from the #87 study's own per-probe
CSV (``--fit-csv``) using the same code path (AbstentionDetector,
split_epochs_parity, select_threshold_youden), then VERIFIES the
reproduction against the study's persisted summary JSON — held-out parity
AUC and Youden threshold must match to their recorded precision — and
RAISES on mismatch rather than scoring with an unverified detector. Only
then is the detector applied, with no refitting of any kind, to the PR-2
failure-mode CSVs. The fitted parameters (means, stds, coefficients,
intercept, threshold) are dumped into the report so later PRs can consume
the frozen detector without re-deriving it.

Scoping. The #87 detector is an in-zone gate: it was fit only on
forced-zone rows (margin_bucket == 0) and #85/#86 deployment semantics
retain out-of-zone rows without consulting it. The PR-2 arms are
stationary and mostly healthy, so mode failures need not land in the
forced zone at all. The report therefore separates:

  * zone composition — where each group's rows actually sit;
  * ``auc.all_rows`` (PRIMARY for the triggers) — score-separation of
    mode-wrong vs correct over every voting row, i.e. "does the frozen
    score rank mode failures below correct retrievals where they actually
    occur"; AUC convention matches analyze_failure_modes.py: > 0.5 means
    the detector is correctly oriented for that mode;
  * ``auc.forced_zone`` (secondary) — the same question restricted to the
    detector's fit domain;
  * the operational gate — abstain iff in-zone AND score < frozen
    threshold (the deployed rule), plus an unzoned variant for reference.

Dose-response (SCHEMA.md): error rate vs ``contra_vote_weight`` /
``stale_vote_weight`` bins, turning exposure labels into attribution.

Example (from repo root):
    python benchmarks/score_frozen_detector.py \
        results/issue_failure_mode_blindness/per_probe_vision_contra.csv \
        results/issue_failure_mode_blindness/per_probe_vision_stale.csv \
        results/issue_failure_mode_blindness/per_probe_vision_clean.csv \
        --json-out results/issue_failure_mode_blindness/pr3a_frozen_detector_scores.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.analyze_calibration import (  # noqa: E402
    auc_mann_whitney, load, smask)
from benchmarks.heldout_abstention import (  # noqa: E402
    FORCED_BUCKET, MANIFOLD_SUPPORT, RANK_GAP, AbstentionDetector,
    abstain_mask, epoch_mask, select_threshold_youden, split_epochs_parity)

FIT_CSV = "results/issue_vitl14_blend_confidence/per_probe.csv"
FIT_SUMMARY = ("results/issue_vitl14_blend_confidence/"
               "heldout_abstention_summary.json")

# Pre-registered redesign triggers (SCHEMA.md, stated before PR-2 data).
TRIGGERS = {
    "contra_auc_le": 0.60,
    "contra_confidently_wrong_gt": 0.50,
    "stale_auc_le": 0.55,
}

DOSE_BINS = [(0.0, 0.0), (0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]


def reproduce_frozen_detector(fit_csv: Path, fit_summary: Path):
    """Refit the #87 parity detector from its own study data and verify the
    reproduction against the persisted summary. Returns (detector, frozen
    threshold, provenance dict). Raises on any mismatch."""
    with open(fit_summary) as f:
        summary = json.load(f)
    if summary["main_model"] != "main:rank_gap+manifold_support":
        raise RuntimeError(f"unexpected main model in {fit_summary}: "
                           f"{summary['main_model']}")
    e_lo, e_hi = summary["study_window"]

    cols, _ = load(fit_csv)
    forced = smask(cols, e_lo=e_lo, e_hi=e_hi, bucket=FORCED_BUCKET)
    forced_epochs = sorted(set(int(e) for e in cols["epoch"][forced]))
    train_epochs, test_epochs = split_epochs_parity(forced_epochs)
    if train_epochs != summary["splits"]["parity"]["train_epochs"]:
        raise RuntimeError(
            f"parity train epochs drifted: recomputed {train_epochs}, "
            f"summary has {summary['splits']['parity']['train_epochs']}")

    det = AbstentionDetector([RANK_GAP, MANIFOLD_SUPPORT])
    train_mask = forced & epoch_mask(cols["epoch"], train_epochs)
    test_mask = forced & epoch_mask(cols["epoch"], test_epochs)
    det.fit(cols, train_mask, "vote_correct")

    # --- verification against the persisted #87 summary -------------------
    p_tr = det.score(cols, train_mask)
    y_tr = cols["vote_correct"][train_mask].astype(int)
    thr = select_threshold_youden(p_tr, y_tr)
    p_te = det.score(cols, test_mask)
    y_te = cols["vote_correct"][test_mask].astype(int)
    auc_te = auc_mann_whitney(p_te, y_te)

    want_auc = summary["heldout_auc"]["parity_vote"]
    want_thr = summary["operating_point_parity_vote"]["threshold"]
    if round(auc_te, 4) != want_auc:
        raise RuntimeError(
            f"frozen-detector reproduction FAILED: held-out parity AUC "
            f"{auc_te:.4f} != persisted {want_auc} — refusing to score with "
            f"an unverified detector.")
    if round(thr, 6) != want_thr:
        raise RuntimeError(
            f"frozen-detector reproduction FAILED: Youden threshold "
            f"{thr:.6f} != persisted {want_thr}.")

    provenance = {
        "fit_csv": str(fit_csv),
        "fit_summary": str(fit_summary),
        "features": [RANK_GAP, MANIFOLD_SUPPORT],
        "target": "vote_correct",
        "split": "parity (PRIMARY in heldout_abstention.py)",
        "train_epochs": train_epochs,
        "verified_heldout_parity_auc": round(auc_te, 4),
        "frozen_threshold_youden": round(thr, 6),
        "standardization_mean": [round(float(v), 6) for v in det.mean_],
        "standardization_std": [round(float(v), 6) for v in det.std_],
        "logistic_coef": [round(float(v), 6) for v in det.model_.coef_[0]],
        "logistic_intercept": round(float(det.model_.intercept_[0]), 6),
    }
    return det, thr, provenance


def load_arm_rows(paths):
    """PR-2 per-probe CSVs grouped by arm, with numeric columns parsed."""
    numeric = {"vote_correct", "contradictory_lenient", "stale_lenient",
               "contradictory_strict", "stale_strict"}
    floaty = {RANK_GAP, MANIFOLD_SUPPORT, "epoch", "margin_bucket",
              "contra_vote_weight", "stale_vote_weight"}
    by_arm: dict[str, list[dict]] = {}
    for path in paths:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                row = dict(r)
                for k in numeric:
                    row[k] = int(float(row[k]))
                for k in floaty:
                    row[k] = float(row[k])
                by_arm.setdefault(row["arm"], []).append(row)
    return by_arm


def _summ(vals: np.ndarray) -> dict:
    if vals.size == 0:
        return {"n": 0}
    return {"n": int(vals.size),
            "median": round(float(np.median(vals)), 4),
            "p10": round(float(np.percentile(vals, 10)), 4),
            "p90": round(float(np.percentile(vals, 90)), 4)}


def _auc_or_nan(score, group, correct):
    """AUC > 0.5 = correctly oriented (correct rows score higher)."""
    if not group.any() or not correct.any():
        return float("nan")
    sel = group | correct
    return round(auc_mann_whitney(score[sel], correct[sel].astype(int)), 4)


def _round_nan(x):
    return None if (isinstance(x, float) and math.isnan(x)) else x


def score_arm(rows: list[dict], det: AbstentionDetector, thr: float) -> dict:
    cols = {k: np.array([r[k] for r in rows], dtype=float)
            for k in (RANK_GAP, MANIFOLD_SUPPORT, "vote_correct",
                      "margin_bucket", "contra_vote_weight",
                      "stale_vote_weight", "contradictory_lenient",
                      "stale_lenient")}
    n = len(rows)
    score = det.score(cols, np.ones(n, dtype=bool))
    wrong = cols["vote_correct"] == 0
    correct = ~wrong
    groups = {
        "correct": correct,
        "contra_wrong": wrong & (cols["contradictory_lenient"] == 1),
        "stale_wrong": wrong & (cols["stale_lenient"] == 1),
        "other_wrong": wrong & (cols["contradictory_lenient"] == 0)
                             & (cols["stale_lenient"] == 0),
    }
    in_zone = cols["margin_bucket"] == FORCED_BUCKET

    out: dict = {"n_probes": n, "wrong": int(wrong.sum())}

    out["zone_composition"] = {
        name: {"n": int(m.sum()), "n_forced_zone": int((m & in_zone).sum()),
               "forced_zone_share":
                   round(float((m & in_zone).sum() / m.sum()), 4)
                   if m.any() else None}
        for name, m in groups.items()}

    out["score_distribution"] = {
        name: _summ(score[m]) for name, m in groups.items()}

    out["auc"] = {}
    for scope, scope_mask in (("all_rows", np.ones(n, dtype=bool)),
                              ("forced_zone", in_zone)):
        out["auc"][scope] = {
            mode: _round_nan(_auc_or_nan(score, groups[mode] & scope_mask,
                                         correct & scope_mask))
            for mode in ("contra_wrong", "stale_wrong", "other_wrong")}

    # confidently-wrong: mode-wrong rows scoring ABOVE the median correct
    # score (the SCHEMA.md definition, applied to the frozen detector score).
    out["confidently_wrong"] = {}
    for scope, scope_mask in (("all_rows", np.ones(n, dtype=bool)),
                              ("forced_zone", in_zone)):
        cm = correct & scope_mask
        block = {}
        for mode in ("contra_wrong", "stale_wrong"):
            mm = groups[mode] & scope_mask
            block[mode] = (round(float(
                (score[mm] > np.median(score[cm])).mean()), 4)
                if mm.any() and cm.any() else None)
        out["confidently_wrong"][scope] = block

    # operational gate at the frozen threshold. Deployed semantics
    # (#85/#86): out-of-zone rows are retained without consulting the
    # detector; in-zone rows are retained iff score >= threshold.
    retain_zoned = ~in_zone | abstain_mask(score, thr)
    retain_unzoned = abstain_mask(score, thr)
    out["operating_point"] = {}
    for name, retain in (("zoned", retain_zoned), ("unzoned", retain_unzoned)):
        op = {
            "threshold": round(thr, 6),
            "retained_coverage": round(float(retain.mean()), 4),
            "retained_accuracy":
                round(float(correct[retain].mean()), 4)
                if retain.any() else None,
            "false_abstain_rate_on_correct":
                round(float((~retain[correct]).mean()), 4)
                if correct.any() else None,
        }
        for mode in ("contra_wrong", "stale_wrong"):
            m = groups[mode]
            op[f"error_capture_{mode}"] = (
                round(float((~retain[m]).mean()), 4) if m.any() else None)
        out["operating_point"][name] = op

    # dose-response: wrong rate vs flagged vote mass (exposure -> attribution)
    out["dose_response"] = {}
    for col in ("contra_vote_weight", "stale_vote_weight"):
        w = cols[col]
        bins = []
        for lo, hi in DOSE_BINS:
            m = (w == 0.0) if lo == hi else (w > lo) & (w <= hi)
            bins.append({
                "bin": "0" if lo == hi else f"({lo},{hi if hi <= 1 else 1.0}]",
                "n": int(m.sum()),
                "wrong_rate": round(float(wrong[m].mean()), 4)
                              if m.any() else None})
        out["dose_response"][col] = bins
    return out


def evaluate_triggers(report: dict) -> dict:
    """Pre-registered triggers, evaluated on the PRIMARY (all_rows) scope of
    the arm that carries each mode; the forced-zone secondary is recorded
    alongside so a scope-dependent verdict is visible, not hidden."""
    def get(arm, path, default=None):
        node = report.get(arm)
        for k in path:
            if node is None:
                return default
            node = node.get(k)
        return node if node is not None else default

    contra_auc = get("contra", ["auc", "all_rows", "contra_wrong"])
    contra_cw = get("contra", ["confidently_wrong", "all_rows",
                               "contra_wrong"])
    stale_auc = get("stale", ["auc", "all_rows", "stale_wrong"])

    contra_fired = (contra_auc is not None
                    and contra_auc <= TRIGGERS["contra_auc_le"]) or \
                   (contra_cw is not None
                    and contra_cw > TRIGGERS["contra_confidently_wrong_gt"])
    stale_fired = (stale_auc is not None
                   and stale_auc <= TRIGGERS["stale_auc_le"])
    return {
        "definitions": TRIGGERS,
        "primary_scope": "all_rows",
        "contradictory": {
            "auc": contra_auc,
            "confidently_wrong": contra_cw,
            "auc_forced_zone_secondary":
                get("contra", ["auc", "forced_zone", "contra_wrong"]),
            "fired": bool(contra_fired),
            "consequence_if_fired":
                "write-time contradiction detection required",
        },
        "stale": {
            "auc": stale_auc,
            "confidently_wrong":
                get("stale", ["confidently_wrong", "all_rows", "stale_wrong"]),
            "auc_forced_zone_secondary":
                get("stale", ["auc", "forced_zone", "stale_wrong"]),
            "fired": bool(stale_fired),
            "consequence_if_fired": "recency must enter the readout",
        },
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+",
                    help="PR-2 per_probe_vision_*.csv files")
    ap.add_argument("--fit-csv", type=str, default=FIT_CSV)
    ap.add_argument("--fit-summary", type=str, default=FIT_SUMMARY)
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    det, thr, provenance = reproduce_frozen_detector(
        Path(args.fit_csv), Path(args.fit_summary))
    print(f"[frozen] #87 parity detector reproduced and verified: "
          f"heldout AUC {provenance['verified_heldout_parity_auc']}, "
          f"threshold {provenance['frozen_threshold_youden']}")

    by_arm = load_arm_rows(args.csvs)
    report: dict = {"frozen_detector": provenance}
    for arm, rows in sorted(by_arm.items()):
        report[arm] = score_arm(rows, det, thr)
    report["pre_registered_triggers"] = evaluate_triggers(report)

    print(json.dumps(report, indent=2))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[score] report -> {args.json_out}")


if __name__ == "__main__":
    main()
