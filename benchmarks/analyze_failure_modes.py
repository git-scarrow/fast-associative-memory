#!/usr/bin/env python3
"""Failure-mode confidence analysis (PR-2b/PR-2c, issue: failure-mode blindness).

ANALYSIS ONLY. Reads ``per_probe_injected.csv`` files produced by
benchmarks/failure_mode_probe.py and answers, per arm, the memo questions
for the CONTRADICTORY (PR-2b) and STALE (PR-2c) modes symmetrically:

  * label rates — wrong rate, contradictory and stale strict/lenient rates
    (overall and as a share of wrong answers), failure_mode distribution,
    and the explicit contra∧stale lenient overlap (a probe implicated by
    both modes is counted in both AND reported as overlap — STALE is never
    silently folded into CONTRADICTORY);
  * confidence behavior — distribution of every existing label-free signal
    (vote_entropy, top1_top2_margin, top1_sim, effective_support,
    max_vote_weight, n_surviving_votes) on CORRECT vs CONTRA-WRONG
    (contradictory_lenient) vs STALE-WRONG (stale_lenient) vs OTHER-WRONG
    probes;
  * per-signal discriminability — rank AUC of each signal for separating
    each wrong group from correct, with the signal's confidence orientation
    fixed a priori (see ``SIGNALS``), so AUC > 0.5 always means "the signal
    flags the failure";
  * confidently-wrong rate — share of mode-wrong probes whose confidence
    exceeds the MEDIAN confidence of correct probes under the same signal
    (the SCHEMA.md pre-registered definition, applied per signal per mode).

No detector is refit and no operating point is chosen here: every number is
computed from the frozen telemetry columns the driver emitted. The frozen
#87 two-axis detector scoring stays in PR-3 as pre-registered.

Example:
    python benchmarks/analyze_failure_modes.py \
        results/issue_failure_mode_blindness/per_probe_contra.csv \
        results/issue_failure_mode_blindness/per_probe_clean.csv \
        --json-out results/issue_failure_mode_blindness/contra_analysis.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# (column, higher_means_more_confident). Orientations are fixed a priori from
# the signals' definitions, not fitted: entropy up = less confident, margin /
# similarity / weight up = more confident. effective_support and
# n_surviving_votes count voters; under collapse MORE off-class support is the
# failure signature (#85/#86), so larger = less confident there too.
SIGNALS = [
    ("vote_entropy", False),
    ("top1_top2_margin", True),
    ("top1_sim", True),
    ("max_vote_weight", True),
    ("effective_support", False),
    ("n_surviving_votes", False),
]


def rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg) by midrank (Mann-Whitney U / (n_pos * n_neg))."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    both = np.concatenate([pos, neg])
    order = both.argsort(kind="mergesort")
    ranks = np.empty_like(both)
    ranks[order] = np.arange(1, both.size + 1, dtype=float)
    # midranks for ties
    sorted_vals = both[order]
    i = 0
    while i < sorted_vals.size:
        j = i
        while j + 1 < sorted_vals.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    u = ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def summarize(vals: np.ndarray) -> dict:
    if vals.size == 0:
        return {"n": 0}
    return {"n": int(vals.size),
            "mean": round(float(vals.mean()), 4),
            "median": round(float(np.median(vals)), 4),
            "p10": round(float(np.percentile(vals, 10)), 4),
            "p90": round(float(np.percentile(vals, 90)), 4)}


def analyze_arm(rows: list[dict]) -> dict:
    n = len(rows)
    wrong = np.array([r["vote_correct"] == 0 for r in rows])
    c_strict = np.array([r["contradictory_strict"] == 1 for r in rows])
    c_lenient = np.array([r["contradictory_lenient"] == 1 for r in rows])
    s_strict = np.array([r["stale_strict"] == 1 for r in rows])
    s_lenient = np.array([r["stale_lenient"] == 1 for r in rows])
    correct = ~wrong
    contra_wrong = wrong & c_lenient
    stale_wrong = wrong & s_lenient
    other_wrong = wrong & ~c_lenient & ~s_lenient

    modes: dict[str, int] = {}
    for r in rows:
        modes[r["failure_mode"]] = modes.get(r["failure_mode"], 0) + 1

    out = {
        "n_probes": n,
        "wrong": int(wrong.sum()),
        "wrong_rate": round(float(wrong.mean()), 4) if n else float("nan"),
        "contradictory_strict": int(c_strict.sum()),
        "contradictory_lenient": int(c_lenient.sum()),
        "contra_strict_share_of_wrong":
            round(float(c_strict.sum() / wrong.sum()), 4)
            if wrong.any() else float("nan"),
        "contra_lenient_share_of_wrong":
            round(float(c_lenient.sum() / wrong.sum()), 4)
            if wrong.any() else float("nan"),
        "stale_strict": int(s_strict.sum()),
        "stale_lenient": int(s_lenient.sum()),
        "stale_strict_share_of_wrong":
            round(float(s_strict.sum() / wrong.sum()), 4)
            if wrong.any() else float("nan"),
        "stale_lenient_share_of_wrong":
            round(float(s_lenient.sum() / wrong.sum()), 4)
            if wrong.any() else float("nan"),
        # Explicit overlap: probes lenient-implicated by BOTH modes. STALE is
        # never folded into CONTRADICTORY — overlap is reported, not hidden.
        "contra_and_stale_lenient_overlap": int((c_lenient & s_lenient).sum()),
        "failure_mode_counts": dict(sorted(modes.items())),
        "contra_vote_weight_on_contra_wrong": summarize(np.array(
            [r["contra_vote_weight"] for r, m in zip(rows, contra_wrong) if m])),
        "stale_vote_weight_on_stale_wrong": summarize(np.array(
            [r["stale_vote_weight"] for r, m in zip(rows, stale_wrong) if m])),
        "groups": {}, "signals": {},
    }

    for name, mask in (("correct", correct), ("contra_wrong", contra_wrong),
                       ("stale_wrong", stale_wrong),
                       ("other_wrong", other_wrong)):
        out["groups"][name] = {
            col: summarize(np.array([r[col] for r, m in zip(rows, mask) if m]))
            for col, _ in SIGNALS}

    for col, hi_conf in SIGNALS:
        vals = np.array([r[col] for r in rows], dtype=float)
        # risk score: oriented so larger = MORE suspicious / less confident
        risk = -vals if hi_conf else vals
        sig = {
            # AUC of risk for flagging the failure group vs correct
            "auc_contra_wrong_vs_correct":
                round(rank_auc(risk[contra_wrong], risk[correct]), 4),
            "auc_stale_wrong_vs_correct":
                round(rank_auc(risk[stale_wrong], risk[correct]), 4),
            "auc_all_wrong_vs_correct":
                round(rank_auc(risk[wrong], risk[correct]), 4),
        }
        if correct.any():
            med_conf = np.median(vals[correct])
            if contra_wrong.any():
                conf_w = ((vals[contra_wrong] > med_conf) if hi_conf
                          else (vals[contra_wrong] < med_conf))
                sig["confidently_wrong_rate_contra"] = round(
                    float(conf_w.mean()), 4)
            if stale_wrong.any():
                conf_w = ((vals[stale_wrong] > med_conf) if hi_conf
                          else (vals[stale_wrong] < med_conf))
                sig["confidently_wrong_rate_stale"] = round(
                    float(conf_w.mean()), 4)
        out["signals"][col] = sig
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+", help="per_probe_injected.csv files")
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    numeric = {"vote_correct", "contradictory_strict", "contradictory_lenient",
               "stale_strict", "stale_lenient"}
    floaty = {c for c, _ in SIGNALS} | {"contra_vote_weight",
                                        "stale_vote_weight", "epoch"}
    by_arm: dict[str, list[dict]] = {}
    for path in args.csvs:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                row = dict(r)
                for k in numeric:
                    row[k] = int(float(row[k]))
                for k in floaty:
                    row[k] = float(row[k])
                by_arm.setdefault(row["arm"], []).append(row)

    report = {arm: analyze_arm(rows) for arm, rows in sorted(by_arm.items())}
    print(json.dumps(report, indent=2))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[analyze] report -> {args.json_out}")


if __name__ == "__main__":
    main()
