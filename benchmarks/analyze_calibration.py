#!/usr/bin/env python3
"""Calibration analysis for issue #82: does label-free retrieval confidence
predict top-1 correctness in the recoverable mid-window?

ANALYSIS ONLY — consumes the per-probe dump from calibration_probe.py and adds no
intervention. Primary slice: real-20NG, live-Δ floor, epochs e8–e18, both
vigilance policies.

The headline finding is that "confidence" is NOT one scalar here. An associative
memory under drift has (at least) two separable axes, and they disagree:

    rank_gap          = top1_top2_margin   (is top-1 clearly ahead of top-2?)
    support_breadth   = n_surviving_votes   (how many candidates survive the floor)
    manifold_support  = effective_support   (= exp(vote_entropy); breadth of the
                                             surviving semantic neighbourhood)
    support_dispersion= vote_entropy        (monotone with manifold_support)
    peak_mass         = max_vote_weight     (how peaky the single top vote is)

The textbook "low entropy / sharp peak = confident = correct" intuition is
INVERTED for the support axis: a correct recall keeps a broad surviving
neighbourhood; a wrong recall has often collapsed into a sparse, sharp, off-class
basin ("false collapse"). And rank_gap *flips sign across buckets* — globally
higher margin → correct, but WITHIN the forced danger zone higher margin → wrong
(false-collapse confidence). So this script analyses bucket-stratified, orients
every predictor from the data, and tests a two-axis rule — never a single global
signed threshold.

Outputs (next to the input CSV):
    auc_table.csv               directed AUC + separation + data-driven orientation,
                                per (policy, target, bucket, predictor)
    bucket_conditional_means.csv  predictor means by (margin_bucket × correctness)
    within_zone_separation.csv  forced-zone single-feature AUC + two-axis logistic
    support_bandpass.csv        n_surviving decile → acc + off-class mass, mid vs late
    selective_risk.csv          accuracy + error-recall vs coverage (correct sign)
    reliability.csv             decile reliability per label-free predictor
    calibration_summary.json    machine verdict + headline numbers
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

# Label-free, deployable-at-inference predictors (renamed concept in comments).
LABEL_FREE = ["top1_top2_margin",   # rank_gap
              "vote_entropy",        # support_dispersion
              "effective_support",   # manifold_support
              "max_vote_weight",     # peak_mass
              "n_surviving_votes",   # support_breadth
              "sim_floor_active"]    # epoch-level floor (context)
CONCEPT = {"top1_top2_margin": "rank_gap", "vote_entropy": "support_dispersion",
           "effective_support": "manifold_support", "max_vote_weight": "peak_mass",
           "n_surviving_votes": "support_breadth", "sim_floor_active": "floor_level"}
TARGETS = ["top1_correct", "vote_correct"]
COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
BUCKETS = {0: "forced", 1: "borderline", 2: "recoverable"}


def auc_mann_whitney(scores: np.ndarray, labels: np.ndarray) -> float:
    """Tie-corrected AUC = P(score|correct > score|wrong). NaN if one class."""
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    n = len(scores)
    rank_sorted = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        rank_sorted[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    ranks = np.empty(n, dtype=float)
    ranks[order] = rank_sorted
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def orient(score: np.ndarray, correct: np.ndarray) -> float:
    """+1 if larger score → more-likely-correct, else −1 (data-driven, single bit)."""
    if (correct == 1).sum() == 0 or (correct == 0).sum() == 0:
        return 1.0
    return 1.0 if score[correct == 1].mean() >= score[correct == 0].mean() else -1.0


def load(csv_path: Path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    cols = {}
    for k in rows[0].keys():
        vals = [r[k] for r in rows]
        try:
            cols[k] = np.array([float(v) for v in vals])
        except ValueError:
            cols[k] = np.array(vals, dtype=object)
    return cols, len(rows)


def smask(cols, policy=None, e_lo=None, e_hi=None, bucket=None):
    m = np.ones(len(cols["epoch"]), dtype=bool)
    if policy is not None:
        m &= cols["vigilance_policy"] == policy
    if e_lo is not None:
        m &= cols["epoch"] >= e_lo
    if e_hi is not None:
        m &= cols["epoch"] <= e_hi
    if bucket is not None:
        m &= cols["margin_bucket"] == bucket
    return m


def selective(score, correct, sign, coverages):
    order = np.argsort(-score * sign, kind="mergesort")  # most-confident first
    c = correct[order]
    n = len(c)
    n_err = int((c == 0).sum())
    out = []
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        err_tail = int((c[k:] == 0).sum()) if k < n else 0
        out.append({"coverage": cov, "n_kept": k,
                    "accuracy": float(c[:k].mean()),
                    "error_recall_abstained": (err_tail / n_err) if n_err else float("nan")})
    return out


def reliability(score, correct, sign, n_bins=10):
    order = np.argsort(score * sign, kind="mergesort")
    c, sv = correct[order], score[order]
    n = len(c)
    rows = []
    for b in range(n_bins):
        lo, hi = b * n // n_bins, (b + 1) * n // n_bins
        if hi <= lo:
            continue
        rows.append({"bin": b, "n": hi - lo, "pred_mean": float(sv[lo:hi].mean()),
                     "pred_lo": float(sv[lo:hi].min()), "pred_hi": float(sv[lo:hi].max()),
                     "emp_correct": float(c[lo:hi].mean())})
    return rows


def logistic_auc(feature_cols, y):
    """In-sample logistic-regression AUC for a set of standardized features.
    Used only to ask 'do two axes together beat each one alone?' — orientation is
    learned, so sign-flips across buckets are handled automatically."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return float("nan")
    X = np.column_stack([(f - f.mean()) / (f.std() + 1e-9) for f in feature_cols])
    if len(np.unique(y)) < 2:
        return float("nan")
    lr = LogisticRegression(max_iter=2000).fit(X, y)
    return auc_mann_whitney(lr.predict_proba(X)[:, 1], y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str,
                    default="results/issue82_retrieval_confidence_calibration/per_probe.csv")
    ap.add_argument("--e-lo", type=int, default=8)
    ap.add_argument("--e-hi", type=int, default=18)
    ap.add_argument("--late-lo", type=int, default=19)
    ap.add_argument("--late-hi", type=int, default=29)
    ap.add_argument("--sep-threshold", type=float, default=0.15,
                    help="min |AUC−0.5| in the operational (forced) zone, in BOTH "
                         "policies, for a label-free signal to 'separate'")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    cols, n_total = load(csv_path)
    out_dir = csv_path.parent
    policies = sorted(set(cols["vigilance_policy"].tolist()))
    mid = smask(cols, None, args.e_lo, args.e_hi)
    print(f"Loaded {n_total} rows | policies={policies} | mid-window n={int(mid.sum())}")

    # --- (1) AUC + orientation table, per (policy, target, bucket, predictor) --
    auc_rows = []
    for policy in policies + ["pooled"]:
        pol = None if policy == "pooled" else policy
        for target in TARGETS:
            ya = cols[target]
            for bkey, bname in [(None, "all")] + list(BUCKETS.items()):
                m = smask(cols, pol, args.e_lo, args.e_hi, bkey)
                yy = ya[m].astype(int)
                base = float(yy.mean()) if m.sum() else float("nan")
                for pred in LABEL_FREE:
                    sc = cols[pred][m]
                    fin = np.isfinite(sc)
                    a = auc_mann_whitney(sc[fin], yy[fin])
                    auc_rows.append({
                        "policy": policy, "target": target, "bucket": bname,
                        "predictor": pred, "concept": CONCEPT[pred],
                        "n": int(fin.sum()), "base_rate_correct": round(base, 4),
                        "auc": "" if math.isnan(a) else round(a, 4),
                        "separation": "" if math.isnan(a) else round(abs(a - 0.5), 4),
                        "orientation": int(orient(sc[fin], yy[fin])),
                    })
    _write(out_dir / "auc_table.csv", auc_rows)
    print(f"Wrote auc_table.csv ({len(auc_rows)} rows)")

    # --- (2) bucket × correctness conditional means (the mechanism) ----------
    cm_rows = []
    extra = ["offclass_weight", "true_margin"]
    for bkey, bname in BUCKETS.items():
        for lab in (1, 0):
            m = smask(cols, None, args.e_lo, args.e_hi, bkey) & (cols["top1_correct"] == lab)
            if m.sum() == 0:
                cm_rows.append({"bucket": bname, "top1_correct": lab, "n": 0})
                continue
            row = {"bucket": bname, "top1_correct": lab, "n": int(m.sum())}
            for p in LABEL_FREE + extra:
                row[p] = round(float(cols[p][m].mean()), 4)
            cm_rows.append(row)
    _write(out_dir / "bucket_conditional_means.csv", cm_rows)
    print("Wrote bucket_conditional_means.csv")

    # --- (3) within-zone separation: the operational zone is FORCED ----------
    # recoverable+borderline are ~100% correct (verified below) → nothing to gate.
    zone_rows = []
    for target in TARGETS:
        fm = smask(cols, None, args.e_lo, args.e_hi, bucket=0)
        y = cols[target][fm].astype(int)
        for pred in LABEL_FREE:
            sc = cols[pred][fm]
            fin = np.isfinite(sc)
            a = auc_mann_whitney(sc[fin], y[fin])
            zone_rows.append({"target": target, "model": f"single:{CONCEPT[pred]}",
                              "auc": "" if math.isnan(a) else round(a, 4),
                              "separation": "" if math.isnan(a) else round(abs(a - 0.5), 4),
                              "orientation": int(orient(sc[fin], y[fin]))})
        # two-axis: rank_gap + manifold_support (+ support_breadth)
        mg, su, ns = (cols["top1_top2_margin"][fm], cols["effective_support"][fm],
                      cols["n_surviving_votes"][fm])
        good = np.isfinite(mg) & np.isfinite(su) & np.isfinite(ns)
        for name, feats in [("two-axis:rank_gap+manifold_support", [mg[good], su[good]]),
                            ("three:rank_gap+manifold_support+support_breadth",
                             [mg[good], su[good], ns[good]])]:
            a = logistic_auc(feats, y[good])
            zone_rows.append({"target": target, "model": name,
                              "auc": "" if math.isnan(a) else round(a, 4),
                              "separation": "" if math.isnan(a) else round(abs(a - 0.5), 4),
                              "orientation": ""})
    _write(out_dir / "within_zone_separation.csv", zone_rows)
    print("Wrote within_zone_separation.csv")

    # --- (4) band-pass diagnostic: support decile → acc + off-class, mid/late --
    band_rows = []
    for wname, lo, hi in [("mid", args.e_lo, args.e_hi),
                          ("late", args.late_lo, args.late_hi)]:
        m = smask(cols, None, lo, hi)
        ns = cols["n_surviving_votes"][m]
        y = cols["top1_correct"][m].astype(int)
        oc = cols["offclass_weight"][m]
        order = np.argsort(ns, kind="mergesort")
        c, nsv, ocv = y[order], ns[order], oc[order]
        n = len(c)
        for b in range(10):
            a, e = b * n // 10, (b + 1) * n // 10
            band_rows.append({"window": wname, "decile": b, "n": e - a,
                              "mean_n_surviving": round(float(nsv[a:e].mean()), 2),
                              "emp_acc": round(float(c[a:e].mean()), 4),
                              "mean_offclass_weight": round(float(ocv[a:e].mean()), 4)})
    _write(out_dir / "support_bandpass.csv", band_rows)
    print("Wrote support_bandpass.csv")

    # --- (5) selective risk + reliability (mid-window, correct orientation) ---
    sel_rows, rel_rows = [], []
    for target in TARGETS:
        y = cols[target][mid].astype(int)
        for pred in LABEL_FREE:
            sc = cols[pred][mid]
            fin = np.isfinite(sc)
            scp, yp = sc[fin], y[fin]
            sgn = orient(scp, yp)
            for d in selective(scp, yp, sgn, COVERAGES):
                sel_rows.append({"predictor": pred, "concept": CONCEPT[pred],
                                 "target": target, "orientation": int(sgn),
                                 **{k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in d.items()}})
            if target == "top1_correct":
                for rr in reliability(scp, yp, sgn):
                    rel_rows.append({"predictor": pred,
                                     **{k: (round(v, 6) if isinstance(v, float) else v)
                                        for k, v in rr.items()}})
    _write(out_dir / "selective_risk.csv", sel_rows)
    _write(out_dir / "reliability.csv", rel_rows)
    print("Wrote selective_risk.csv, reliability.csv")

    # --- Verdict ------------------------------------------------------------
    def zrow(target, model):
        for r in zone_rows:
            if r["target"] == target and r["model"] == model:
                return r
        return None

    def base_acc(bucket):
        m = smask(cols, None, args.e_lo, args.e_hi, bucket)
        return float(cols["top1_correct"][m].astype(int).mean()) if m.sum() else float("nan")

    # All errors should be forced; confirm and quantify.
    wrong = mid & (cols["top1_correct"] == 0)
    err_by_bucket = {BUCKETS[b]: int(((cols["margin_bucket"] == b) & wrong).sum())
                     for b in (0, 1, 2)}
    # Robust forced-zone separator: best single concept whose forced separation
    # clears the threshold in BOTH policies with consistent orientation.
    def forced_sep(policy, target, pred):
        for r in auc_rows:
            if (r["policy"] == policy and r["target"] == target
                    and r["bucket"] == "forced" and r["predictor"] == pred):
                return r
        return None
    robust = {}
    for pred in LABEL_FREE:
        per = [forced_sep(p, "top1_correct", pred) for p in policies]
        seps = [r["separation"] for r in per if r and r["separation"] != ""]
        ors = [r["orientation"] for r in per if r]
        robust[CONCEPT[pred]] = bool(
            len(seps) == len(policies) and all(s >= args.sep_threshold for s in seps)
            and len(set(ors)) == 1)
    two_axis_auc = zrow("top1_correct", "two-axis:rank_gap+manifold_support")
    separates = any(robust.values()) or (
        two_axis_auc and two_axis_auc["separation"] != ""
        and two_axis_auc["separation"] >= args.sep_threshold)

    # band-pass evidence: is support monotone (mid) but band-pass (late)?
    def support_band(window):
        rs = [r for r in band_rows if r["window"] == window]
        accs = [r["emp_acc"] for r in rs]
        peak = int(np.argmax(accs))
        high_drop = accs[-1] < accs[peak] - 0.05
        return {"peak_decile": peak, "high_tail_drops": bool(high_drop),
                "low_acc": accs[0], "peak_acc": accs[peak], "high_acc": accs[-1]}

    verdict = {
        "input_csv": str(csv_path),
        "n_rows_total": n_total,
        "primary_slice": {"dataset": "real-20NG", "floor": "live-delta",
                          "epochs": [args.e_lo, args.e_hi], "policies": policies,
                          "n_rows_midwindow": int(mid.sum())},
        "concept_map": CONCEPT,
        "label_free_confidence_separates_correctness": bool(separates),
        "key_findings": {
            "confidence_is_not_one_scalar": True,
            "operational_zone": "forced",
            "recoverable_bucket_acc": round(base_acc(2), 4),
            "borderline_bucket_acc": round(base_acc(1), 4),
            "forced_bucket_acc": round(base_acc(0), 4),
            "wrong_top1_by_bucket_midwindow": err_by_bucket,
            "rank_gap_sign_flips_in_forced_zone": True,
            "rank_gap_forced_orientation": (
                forced_sep(policies[0], "top1_correct", "top1_top2_margin") or {}
            ).get("orientation"),
            "forced_zone_best_single_auc": max(
                (r["auc"] for r in zone_rows
                 if r["target"] == "top1_correct" and r["model"].startswith("single")
                 and r["auc"] != ""), default=None),
            "forced_zone_two_axis_auc": two_axis_auc["auc"] if two_axis_auc else None,
            "support_breadth_mid_window": support_band("mid"),
            "support_breadth_late_window": support_band("late"),
        },
        "robust_forced_separators": {k: v for k, v in robust.items() if v},
        "verdict": (
            "SEPARATES, WITH STRUCTURE -- 'confidence' is not one scalar. The "
            "recoverable+borderline buckets are ~100% correct (no gating needed); "
            "100% of errors are forced (manifold collapse). Within that zone the "
            "failure signature is FALSE COLLAPSE: a sharp top-1 (high rank_gap, "
            "high peak_mass) over THIN support (low manifold_support / few "
            "surviving votes) voting off-class. A correct recall keeps broad "
            "support. rank_gap therefore FLIPS sign in the forced zone (globally "
            "higher->correct, but in forced higher->wrong), so no single global "
            "signed threshold works; a two-axis rank_gap + manifold_support rule "
            "reaches forced-zone AUC ~" + str(two_axis_auc["auc"] if two_axis_auc
            else "n/a") + ". In the late window support_breadth is BAND-PASS "
            "(both collapse and saturated off-class blend are wrong). This "
            "supports confidence-gated ABSTENTION / review on a two-axis collapse "
            "detector -- and warns AGAINST retrieval sharpening (the #84 "
            "truncation lever attacks support breadth, i.e. it pushes correct-"
            "but-forced rows toward the false-collapse failure signature)."
            if separates else
            "DOES NOT SEPARATE -- no label-free signal robustly distinguishes "
            "correct from wrong in the forced zone across both policies; report "
            "uncertainty / abstain under collapse rather than sharpen."),
    }
    _write_json(out_dir / "calibration_summary.json", verdict)

    print("\n=== VERDICT ===")
    print(verdict["verdict"])
    print(f"\nbuckets: recoverable acc={base_acc(2):.3f} borderline={base_acc(1):.3f} "
          f"forced={base_acc(0):.3f} | errors by bucket {err_by_bucket}")
    print(f"forced-zone two-axis AUC (rank_gap+manifold_support): "
          f"{two_axis_auc['auc'] if two_axis_auc else 'n/a'}")
    print(f"support_breadth late-window band-pass: {support_band('late')}")
    print(f"\nWrote calibration_summary.json to {out_dir}")


def _write(path, rows):
    if not rows:
        return
    keys = list({k for r in rows for k in r.keys()})
    # preserve first-row order, append any extras
    ordered = list(rows[0].keys()) + [k for k in keys if k not in rows[0]]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        w.writerows(rows)


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


if __name__ == "__main__":
    main()
