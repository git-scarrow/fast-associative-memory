#!/usr/bin/env python3
"""Held-out two-axis abstention calibration for forced retrieval collapse (#82/#85).

ABSTENTION ONLY — this driver adds **no retrieval intervention**. It consumes the
read-only per-probe dump from ``calibration_probe.py`` (regenerating it if absent)
and asks one operational question the #85 in-sample study left open:

    Does a two-axis collapse detector, fit ONLY on training epochs, still tell
    correct-from-wrong apart on *held-out epochs* inside the forced zone — well
    enough to abstain usefully on the forward() prediction?

Why this exists (the #85 verdict, restated):
  * "Confidence" is not one scalar. It decomposes into RANK confidence
    (``rank_gap`` = top1−top2 margin) and SUPPORT stability
    (``manifold_support`` = exp(vote_entropy); ``support_breadth`` =
    n_surviving_votes).
  * The recoverable + borderline buckets are ~100% correct, so there is nothing
    to gate there. 100% of observed errors are in the FORCED bucket, so the
    operational detector lives entirely inside that zone.
  * ``rank_gap`` FLIPS sign in the forced zone: globally a wider margin weakly
    predicts correctness, but inside forced rows a sharp top-1 over thin support
    is the *false-collapse* signature and predicts WRONGNESS. No single global
    threshold works; the two axes must be combined.
  * #85 reached forced-zone AUC ~0.79 (top-1) / 0.82 (vote) IN-SAMPLE. That
    number is not trustworthy on its own: ``support_breadth`` may be partly a
    readout of the live-Δ retrieval floor, which is an epoch-scalar. Splitting by
    row would let the held-out rows of an epoch borrow that epoch's floor context
    from its training rows. **So the split must be by epoch.**

What this script guarantees (the guardrails):
  * No leakage: orientation, feature standardisation, logistic coefficients, and
    the abstention threshold are ALL fit on train epochs only and then frozen.
  * Held-out epochs never touch any fit decision (including feature choice — the
    feature set is fixed a priori from the #85 in-sample study, not chosen here).
  * No correctness labels at inference: the detector score is a function of
    label-free predictors only; labels are used solely to fit/evaluate.
  * Abstention is a pure post-hoc mask. It NEVER changes the elected class for a
    retained row — ``forward()`` and its argmax are untouched. ``associative_core``
    is not modified by this branch at all.

Two epoch-split strategies are reported, because they probe different failure
modes of the #85 number:

  parity  (PRIMARY)  train = even epochs, test = odd epochs. Train and test span
                     the *same* drift trajectory, so this isolates the question
                     "does the two-axis SIGNAL generalise to unseen rows?" while
                     controlling for the drift-regime shift. A neighbouring
                     epoch's floor is similar, so this is the *charitable* test.
  contiguous         train = earliest fraction of epochs, test = latest. The
                     held-out floor regime is genuinely unseen, so this is the
                     *adversarial* test of the floor-readout worry. Degradation
                     here vs parity is the price of extrapolating across drift.

Example (from repo root):
    python benchmarks/heldout_abstention.py \
        --csv results/issue82_retrieval_confidence_calibration/per_probe.csv \
        --e-lo 8 --e-hi 29 --target-retained-acc 0.80
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.analyze_calibration import (  # noqa: E402
    auc_mann_whitney, load, orient)

# --- feature set: FIXED A PRIORI from the #85 in-sample study (not chosen on the
# held-out data). rank_gap = top1_top2_margin; the support axis primary is
# manifold_support = effective_support (the #85 two-axis winner), with
# support_breadth = n_surviving_votes carried as a secondary readout.
RANK_GAP = "top1_top2_margin"
MANIFOLD_SUPPORT = "effective_support"
SUPPORT_BREADTH = "n_surviving_votes"
FORCED_BUCKET = 0
# Operational target = the class forward() actually elects (vote); top1 alongside.
TARGETS = ["vote_correct", "top1_correct"]


# ----------------------------------------------------------------------------
# Epoch splits (by epoch, never by row). Both are deterministic and auditable.
# ----------------------------------------------------------------------------
def split_epochs_parity(epochs):
    """Interleaved split: train = even epochs, test = odd epochs."""
    uniq = sorted(set(int(e) for e in epochs))
    train = [e for e in uniq if e % 2 == 0]
    test = [e for e in uniq if e % 2 == 1]
    return train, test


def split_epochs_contiguous(epochs, train_frac=0.6):
    """Forward split: earliest ``train_frac`` of epochs train, the rest test."""
    uniq = sorted(set(int(e) for e in epochs))
    k = max(1, min(len(uniq) - 1, int(round(train_frac * len(uniq)))))
    return uniq[:k], uniq[k:]


def epoch_mask(epoch_col, keep_epochs):
    keep = set(int(e) for e in keep_epochs)
    return np.array([int(e) in keep for e in epoch_col], dtype=bool)


# ----------------------------------------------------------------------------
# Detector: standardise + logistic, fit on TRAIN only, then frozen.
# ----------------------------------------------------------------------------
class AbstentionDetector:
    """Two-axis (or single-axis / quadratic) logistic collapse detector.

    Everything that touches the data — feature means/stds for standardisation and
    the logistic coefficients — is computed in :meth:`fit` from the train arrays
    only. :meth:`score` is a pure function of the frozen parameters and so can be
    applied to held-out rows without any leakage.
    """

    def __init__(self, feature_names, quadratic=()):
        self.feature_names = list(feature_names)
        self.quadratic = list(quadratic)  # names to also add as x**2 columns
        self.mean_ = None
        self.std_ = None
        self.model_ = None
        self.single_sign_ = None  # set for 1-feature, no-quadratic detectors

    def _design(self, cols, mask):
        base = [np.asarray(cols[f], dtype=float)[mask] for f in self.feature_names]
        quad = [np.asarray(cols[f], dtype=float)[mask] ** 2 for f in self.quadratic]
        return np.column_stack(base + quad)

    def fit(self, cols, train_mask, target):
        X = self._design(cols, train_mask)
        y = np.asarray(cols[target], dtype=float)[train_mask].astype(int)
        finite = np.isfinite(X).all(axis=1)
        X, y = X[finite], y[finite]
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-9
        Xs = np.clip((X - self.mean_) / self.std_, -10.0, 10.0)
        if len(np.unique(y)) < 2:
            raise ValueError("train fold has a single class; cannot fit detector")
        if X.shape[1] == 1 and not self.quadratic:
            # auditable single-axis baseline: data-driven orientation, no sklearn
            self.single_sign_ = orient(Xs[:, 0], y)
        else:
            from sklearn.linear_model import LogisticRegression
            # silence cosmetic OpenBLAS matmul FP warnings; clipping already
            # bounds the logits, so no decision depends on the masked states.
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                self.model_ = LogisticRegression(max_iter=2000).fit(Xs, y)
        return self

    def score(self, cols, mask):
        """P(correct)-like score in [0,1], higher = more likely correct."""
        X = self._design(cols, mask)
        # clip standardised features: a near-constant feature in a fold gives a
        # tiny std and otherwise blows up the logit (matmul overflow). z=±10 is
        # already a saturated probability, so clipping changes no decision.
        Xs = np.clip((X - self.mean_) / self.std_, -10.0, 10.0)
        if self.model_ is not None:
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                return self.model_.predict_proba(Xs)[:, 1]
        # single-axis: monotone squashing of the oriented standardised feature
        return 1.0 / (1.0 + np.exp(-self.single_sign_ * Xs[:, 0]))


# ----------------------------------------------------------------------------
# Threshold selection (TRAIN ONLY) + held-out operating-point metrics.
# ----------------------------------------------------------------------------
def abstain_mask(scores, thr):
    """Pure mask. True = RETAIN (score >= thr); False = abstain. Never mutates."""
    return np.asarray(scores, dtype=float) >= thr


def _fold_metrics(scores, y, thr):
    """Raw retain/abstain counts at a threshold — shared by selection & report."""
    keep = abstain_mask(scores, thr)
    n, n_keep = len(y), int(keep.sum())
    n_err, n_cor = int((y == 0).sum()), int((y == 1).sum())
    err_abst = int(((~keep) & (y == 0)).sum())
    cor_abst = int(((~keep) & (y == 1)).sum())
    return {
        "coverage": n_keep / n if n else float("nan"),
        "retained_acc": float(y[keep].mean()) if n_keep else float("nan"),
        "error_capture": err_abst / n_err if n_err else float("nan"),
        "false_abstain": cor_abst / n_cor if n_cor else float("nan"),
        "n_keep": n_keep, "err_abst": err_abst,
    }


def select_threshold_youden(p_train, y_train):
    """PRIMARY: threshold maximising Youden's J for *error detection* on TRAIN.

    Framing abstention as a wrong-row detector, J = error_capture − false_abstain.
    Maximising it yields a balanced operating point (not the degenerate >99%-
    abstain point that a fixed retained-accuracy target forces in a zone whose
    base accuracy is far below the target). Selected on TRAIN only, then frozen.
    """
    p, y = np.asarray(p_train, float), np.asarray(y_train, int)
    cand = np.unique(p)
    best_thr, best_j = float(cand[0]), -np.inf
    for thr in cand:
        m = _fold_metrics(p, y, thr)
        if m["n_keep"] == 0:
            continue
        j = m["error_capture"] - m["false_abstain"]
        if j > best_j:
            best_thr, best_j = float(thr), j
    return best_thr


def select_threshold_target_acc(p_train, y_train, target_retained_acc):
    """Smallest abstention that lifts TRAIN retained accuracy to the target.

    Falls back to the best achievable train retained accuracy if the target is
    unreachable. Used for the train-selected coverage/accuracy sweep, NOT as the
    headline operating point (a target above the zone's base accuracy collapses
    coverage). Selected on TRAIN only, then frozen.
    """
    p, y = np.asarray(p_train, float), np.asarray(y_train, int)
    cand = np.unique(p)
    best_thr, best_acc = float(cand[0]), -1.0
    for thr in cand:
        m = _fold_metrics(p, y, thr)
        if m["n_keep"] == 0:
            continue
        if m["retained_acc"] >= target_retained_acc:
            return float(thr)
        if m["retained_acc"] > best_acc:
            best_thr, best_acc = float(thr), m["retained_acc"]
    return best_thr


def operating_point(scores, y, thr, preds=None, base_preds=None):
    """Held-out operating-point metrics for a frozen threshold.

    ``preds``/``base_preds`` (the elected class per row, both equal to forward()'s
    argmax) are only used to *assert* abstention does not alter predictions; they
    are never rewritten.
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    keep = abstain_mask(scores, thr)
    n = len(y)
    n_keep = int(keep.sum())
    n_err = int((y == 0).sum())
    n_correct = int((y == 1).sum())
    err_abstained = int(((~keep) & (y == 0)).sum())
    correct_abstained = int(((~keep) & (y == 1)).sum())
    if preds is not None and base_preds is not None:
        # invariant: the elected class for retained rows is exactly forward()'s.
        assert np.array_equal(np.asarray(preds)[keep], np.asarray(base_preds)[keep])
    return {
        "threshold": round(float(thr), 6),
        "n_total": n,
        "no_abstention_accuracy": round(float(y.mean()), 4) if n else float("nan"),
        "retained_coverage": round(n_keep / n, 4) if n else float("nan"),
        "retained_accuracy": round(float(y[keep].mean()), 4) if n_keep else float("nan"),
        "abstention_rate": round(1 - n_keep / n, 4) if n else float("nan"),
        "error_capture_rate": round(err_abstained / n_err, 4) if n_err else float("nan"),
        "false_abstain_rate_on_correct": (
            round(correct_abstained / n_correct, 4) if n_correct else float("nan")),
        "n_retained": n_keep,
        "n_errors_captured": err_abstained,
    }


# ----------------------------------------------------------------------------
# Dump regeneration (only if the CSV is absent / explicitly requested).
# ----------------------------------------------------------------------------
def ensure_dump(csv_path: Path, regenerate: bool):
    if csv_path.exists() and not regenerate:
        return
    if not csv_path.exists():
        print(f"[regen] {csv_path} not found — regenerating via calibration_probe.py")
    probe = Path(__file__).parent / "calibration_probe.py"
    cmd = [sys.executable, str(probe), "--epochs", "30", "--contraction-end", "0.9",
           "--held-out-per-class", "64", "--out", str(csv_path)]
    print("[regen] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


# ----------------------------------------------------------------------------
SWEEP_TARGETS = [0.5, 0.6, 0.7, 0.8]


def run_split(cols, forced_mask, split_name, train_epochs, test_epochs,
              sweep_model):
    """Fit every model on train epochs, evaluate on held-out epochs."""
    ep = cols["epoch"]
    train_mask = forced_mask & epoch_mask(ep, train_epochs)
    test_mask = forced_mask & epoch_mask(ep, test_epochs)
    assert not (set(int(e) for e in train_epochs) & set(int(e) for e in test_epochs)), \
        "epoch split leaked: train and test share an epoch"

    models = {
        "baseline:rank_gap": AbstentionDetector([RANK_GAP]),
        "baseline:support_breadth": AbstentionDetector([SUPPORT_BREADTH]),
        "baseline:manifold_support": AbstentionDetector([MANIFOLD_SUPPORT]),
        "main:rank_gap+manifold_support": AbstentionDetector([RANK_GAP, MANIFOLD_SUPPORT]),
        "diag:rank_gap+manifold_support+support^2": AbstentionDetector(
            [RANK_GAP, MANIFOLD_SUPPORT], quadratic=[MANIFOLD_SUPPORT]),
    }

    auc_rows, op_rows, sweep_rows = [], [], []
    for target in TARGETS:
        # the elected class per row (== forward().argmax); held only to prove the
        # abstention mask never rewrites a retained prediction.
        preds = np.asarray(cols["vote_pred_label"], dtype=float)
        for mname, det in models.items():
            try:
                det.fit(cols, train_mask, target)
            except ValueError as e:
                auc_rows.append({"split": split_name, "target": target, "model": mname,
                                 "n_train": 0, "n_test": 0,
                                 "train_auc": "", "heldout_auc": "", "note": str(e)})
                continue
            # in-sample (train) AUC for honesty, then the held-out AUC.
            ptr = det.score(cols, train_mask)
            ytr = np.asarray(cols[target], dtype=float)[train_mask].astype(int)
            pte = det.score(cols, test_mask)
            yte = np.asarray(cols[target], dtype=float)[test_mask].astype(int)
            ftr, fte = np.isfinite(ptr), np.isfinite(pte)
            ptr, ytr, pte, yte = ptr[ftr], ytr[ftr], pte[fte], yte[fte]
            preds_te = preds[test_mask][fte]
            a_tr = auc_mann_whitney(ptr, ytr)
            a_te = auc_mann_whitney(pte, yte)
            auc_rows.append({
                "split": split_name, "target": target, "model": mname,
                "n_train": int(ftr.sum()), "n_test": int(fte.sum()),
                "train_auc": "" if math.isnan(a_tr) else round(a_tr, 4),
                "heldout_auc": "" if math.isnan(a_te) else round(a_te, 4),
                "note": "",
            })
            # PRIMARY operating point: Youden-J threshold on TRAIN, applied held-out.
            thr = select_threshold_youden(ptr, ytr)
            op = operating_point(pte, yte, thr,
                                 preds=preds_te, base_preds=preds_te)
            op_rows.append({"split": split_name, "target": target, "model": mname,
                            "selection": "youden_J_train", **op})
            # train-selected coverage/accuracy sweep (headline model + target only)
            if mname == sweep_model and target == "vote_correct":
                for tgt in SWEEP_TARGETS:
                    t = select_threshold_target_acc(ptr, ytr, tgt)
                    sop = operating_point(pte, yte, t,
                                          preds=preds_te, base_preds=preds_te)
                    sweep_rows.append({"split": split_name, "model": mname,
                                       "train_target_acc": tgt, **sop})
    return auc_rows, op_rows, sweep_rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=str,
                    default="results/issue82_retrieval_confidence_calibration/per_probe.csv")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="output dir (default: alongside --csv)")
    ap.add_argument("--e-lo", type=int, default=8,
                    help="lowest epoch in the study window (post-onset)")
    ap.add_argument("--e-hi", type=int, default=29,
                    help="highest epoch in the study window")
    ap.add_argument("--train-frac", type=float, default=0.6,
                    help="fraction of epochs for the contiguous (adversarial) split")
    ap.add_argument("--regenerate", action="store_true",
                    help="force re-running calibration_probe.py even if CSV exists")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    ensure_dump(csv_path, args.regenerate)
    cols, n_total = load(csv_path)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent

    # study window AND forced zone — the only place gating matters.
    ep = cols["epoch"]
    window = (ep >= args.e_lo) & (ep <= args.e_hi)
    forced = window & (cols["margin_bucket"] == FORCED_BUCKET)
    forced_epochs = sorted(set(int(e) for e in ep[forced]))
    base_forced_acc_top1 = float(cols["top1_correct"][forced].astype(int).mean())
    base_forced_acc_vote = float(cols["vote_correct"][forced].astype(int).mean())
    print(f"Loaded {n_total} rows | window e{args.e_lo}-e{args.e_hi} | "
          f"forced rows={int(forced.sum())} over {len(forced_epochs)} epochs")
    print(f"forced-zone base accuracy: vote={base_forced_acc_vote:.3f} "
          f"top1={base_forced_acc_top1:.3f}")

    parity_train, parity_test = split_epochs_parity(forced_epochs)
    contig_train, contig_test = split_epochs_contiguous(forced_epochs, args.train_frac)
    print(f"parity split    : train(even)={parity_train}")
    print(f"                  test(odd)  ={parity_test}")
    print(f"contiguous split: train={contig_train}")
    print(f"                  test ={contig_test}")

    main_model = "main:rank_gap+manifold_support"
    all_auc, all_op, all_sweep = [], [], []
    for name, (tr, te) in [("parity", (parity_train, parity_test)),
                           ("contiguous", (contig_train, contig_test))]:
        a, o, s = run_split(cols, forced, name, tr, te, main_model)
        all_auc += a
        all_op += o
        all_sweep += s

    _write(out_dir / "heldout_auc.csv", all_auc)
    _write(out_dir / "heldout_operating_point.csv", all_op)
    _write(out_dir / "heldout_coverage_sweep.csv", all_sweep)
    print(f"\nWrote heldout_auc.csv ({len(all_auc)} rows), "
          f"heldout_operating_point.csv ({len(all_op)} rows), "
          f"heldout_coverage_sweep.csv ({len(all_sweep)} rows)")

    # --- verdict -----------------------------------------------------------
    def pick(rows, **kw):
        for r in rows:
            if all(r.get(k) == v for k, v in kw.items()):
                return r
        return None

    main_parity = pick(all_auc, split="parity", target="vote_correct",
                       model="main:rank_gap+manifold_support")
    main_contig = pick(all_auc, split="contiguous", target="vote_correct",
                       model="main:rank_gap+manifold_support")
    op_parity = pick(all_op, split="parity", target="vote_correct",
                     model="main:rank_gap+manifold_support")
    op_contig = pick(all_op, split="contiguous", target="vote_correct",
                     model="main:rank_gap+manifold_support")

    def auc_val(r):
        return r["heldout_auc"] if r and r["heldout_auc"] != "" else None

    # operationally useful = held-out main AUC clears chance with margin AND, at
    # the train-selected (Youden) operating point, the detector (a) retains a
    # non-degenerate slice of rows, (b) lifts retained accuracy above the no-
    # abstention base, and (c) captures more errors than the correct rows it
    # wrongly abstains (positive Youden J on held-out).
    def _num(x):
        return x if isinstance(x, (int, float)) and not math.isnan(x) else None

    def useful(auc_row, op_row, base_acc):
        if not auc_row or auc_row["heldout_auc"] == "" or not op_row:
            return False
        ra = _num(op_row["retained_accuracy"])
        cov = _num(op_row["retained_coverage"])
        ec = _num(op_row["error_capture_rate"])
        fa = _num(op_row["false_abstain_rate_on_correct"])
        if None in (ra, cov, ec, fa):
            return False
        return bool(auc_row["heldout_auc"] >= 0.65 and 0.05 <= cov <= 0.95
                    and ra > base_acc and ec > fa)

    parity_useful = useful(main_parity, op_parity, base_forced_acc_vote)
    contig_useful = useful(main_contig, op_contig, base_forced_acc_vote)

    verdict = {
        "input_csv": str(csv_path),
        "n_rows_total": n_total,
        "study_window": [args.e_lo, args.e_hi],
        "forced_zone": {
            "n_rows": int(forced.sum()), "n_epochs": len(forced_epochs),
            "base_accuracy_vote": round(base_forced_acc_vote, 4),
            "base_accuracy_top1": round(base_forced_acc_top1, 4),
        },
        "feature_set_fixed_a_priori": {
            "rank_gap": RANK_GAP, "manifold_support": MANIFOLD_SUPPORT,
            "support_breadth": SUPPORT_BREADTH},
        "splits": {
            "parity": {"train_epochs": parity_train, "test_epochs": parity_test},
            "contiguous": {"train_epochs": contig_train, "test_epochs": contig_test,
                           "train_frac": args.train_frac}},
        "headline_target": "vote_correct",
        "main_model": "main:rank_gap+manifold_support",
        "heldout_auc": {
            "parity_vote": auc_val(main_parity),
            "contiguous_vote": auc_val(main_contig)},
        "operating_point_parity_vote": op_parity,
        "operating_point_contiguous_vote": op_contig,
        "operationally_useful": {
            "parity": parity_useful, "contiguous": contig_useful},
        "verdict": _verdict_text(main_parity, main_contig, op_parity, op_contig,
                                 base_forced_acc_vote, parity_useful, contig_useful),
    }
    _write_json(out_dir / "heldout_abstention_summary.json", verdict)

    print("\n=== HELD-OUT VERDICT ===")
    print(verdict["verdict"])
    print(f"\nmain (vote) held-out AUC: parity={auc_val(main_parity)} "
          f"contiguous={auc_val(main_contig)}  (chance=0.5)")
    if op_parity:
        print(f"parity operating point (Youden-J on train): "
              f"coverage={op_parity['retained_coverage']} "
              f"retained_acc={op_parity['retained_accuracy']} "
              f"(base={base_forced_acc_vote:.3f}) "
              f"error_capture={op_parity['error_capture_rate']} "
              f"false_abstain={op_parity['false_abstain_rate_on_correct']}")
    print(f"\nWrote heldout_abstention_summary.json to {out_dir}")


def _verdict_text(mp, mc, op_p, op_c, base, pu, cu):
    pa = mp["heldout_auc"] if mp and mp["heldout_auc"] != "" else "n/a"
    ca = mc["heldout_auc"] if mc and mc["heldout_auc"] != "" else "n/a"
    head = (
        "Two axes — RANK confidence (rank_gap) and SUPPORT stability "
        "(manifold_support) — were combined in a logistic detector fit ONLY on "
        "train epochs and evaluated ONLY on held-out epochs inside the forced "
        f"zone (base accuracy {base:.3f}). Held-out forced-zone AUC for the main "
        f"two-axis model: parity={pa}, contiguous={ca} (chance 0.5). ")
    if pu and cu:
        tail = ("The detector SURVIVES held-out evaluation on both the charitable "
                "(parity) and adversarial (contiguous) epoch splits: at the train-"
                "selected operating point it lifts retained accuracy above the "
                "no-abstention base and captures more errors than correct rows it "
                "wrongly abstains. Abstention on probable FALSE COLLAPSE is "
                "operationally useful but NOT production-ready; "
                "sharpening/truncation remains hazardous "
                "because support breadth is part of the correctness signal.")
    elif pu and not cu:
        tail = ("The detector survives the charitable parity split but DEGRADES on "
                "the adversarial contiguous split — consistent with support_breadth "
                "being partly a readout of the per-epoch live-Δ floor, which the "
                "contiguous split holds out. Operationally useful only when the "
                "deployment drift regime resembles training; treat with caution and "
                "do NOT sharpen (support breadth carries correctness signal).")
    else:
        tail = ("The detector does NOT survive held-out epoch evaluation: the #85 "
                "in-sample AUC did not transfer once the per-epoch floor context "
                "was held out. Report uncertainty / abstain conservatively rather "
                "than trust the two-axis score, and do NOT sharpen.")
    return head + tail


def _write(path, rows):
    if not rows:
        return
    keys = list({k for r in rows for k in r.keys()})
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
