"""PR-11.1 — adjudication-window scan (PR11_ADJUDICATION_DESIGN.md §5-6).

Analysis-only reader over the two RE-EMITTED governance tables plus the
COMMITTED per-run artifacts of the PR-4 fresh grid (5 pairs x 6 arms x 3
seeds; the PR-3c anchors reused verbatim through the same pinned stem
mapping ``analyze_pr4_geometry.run_stem``). No engine import, no new run,
no policy parameter. It emits ``pr11/adjudication_scan.json`` containing,
per cell:

  * per-policy counters for the three registered PR-11.1 policies
    (`adjudicated-abstain` P1, `merge-support-abstain` P2,
    `pending-abstain` P3): capture of the policy's named class, false
    abstentions (``abstain_on_correct``) and their rate against correct
    traffic (the exact ``pr9_abstention_envelope.py`` definition:
    ``abstain_on_correct / (n - wrong_none)``), clean-arm actions, changed
    answers, forced abstentions, per-trigger decomposition;
  * the P2 residual ledger (soft arms): stale-wrong rows NOT captured by
    the certified `merge-abstain` trigger, and how many of those P2's
    support-membership trigger reaches;
  * the pending-nonredundancy computation (one-shot arms): wrong rows
    captured by ambiguous(E)-led abstention and by NEITHER P1's sets nor
    P2's trigger, as a fraction of P3's captured wrong mass;
  * the resolution-lag scan: per fork pair (fork epoch, epoch of first
    non-ambiguous verdict or null, final verdict, lag), and per harm
    class the split of harmed rows by their leading slot's pair status at
    the probe's epoch (operationalization fixed here, before any scan
    output was seen: `unpaired` = top-1 slot is a member of no routed
    pair; `pre_fork` = every pair containing it forks at a later epoch;
    `post_resolution` = at least one containing pair with an observed
    fork has a first non-ambiguous verdict at or before the probe's
    epoch; `pre_resolution` = otherwise, i.e. the window is open).

Every per-row policy decision is recomputed through the frozen scorer's
own ``apply_policy`` and cross-checked against the re-emitted
``pr4_geometry_table.json`` row for the cell — any counter mismatch
raises. The deployed ``vote_pred_label`` is used as the `none` answer;
its bit-fidelity to the shadow vote was raise-verified on every row of
every one of these runs during the table re-emission (§10).

Gate frame (§6): pairs B-E gate-scored on ALL seeds (the memo's residual
accounting is "fresh pairs B-E x seeds 0-2"); every pair A cell is a
report-only tainted anchor. The PR-4 `tainted_anchor` flag (which also
marks pair B s0 clean/contra/stale/mixed) is carried per cell for
transparency but does not remove pair B cells from PR-11 gating.

Usage:
  python benchmarks/pr11_adjudication_scan.py \
      --pr3c-dir results/issue_failure_mode_blindness/pr3c \
      --pr4-dir  results/issue_failure_mode_blindness/pr4 \
      --out results/issue_failure_mode_blindness/pr11/adjudication_scan.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.analyze_fork_governance import (  # noqa: E402
    ADJUDICATION_POLICIES, apply_policy, build_writetime_router, load_run)
from benchmarks.analyze_pr4_geometry import (  # noqa: E402
    ARMS, PAIRS, SEEDS, run_stem)

P1, P2, P3 = ADJUDICATION_POLICIES
SCAN_POLICIES = ("merge-abstain",) + ADJUDICATION_POLICIES
GATE_PAIRS = ("pairB", "pairC", "pairD", "pairE")
FALSE_ABSTAIN_MAX_FRAC = 0.05      # the standing program ceiling (§6)
CAPTURE_FLOOR = 0.5                # aggregate-over-seeds, per named pair
P2_RESIDUAL_FLOOR_AGG = 76         # of the 151 pairD soft residual rows
P2_RESIDUAL_FLOOR_S0 = 42          # of the 83 pairD/soft/s0 residual rows
NONREDUNDANT_FLOOR = 0.5           # of P3's captured one-shot wrong mass

# Counters cross-checked row-for-row against the re-emitted table.
CHECK_KEYS = ("n", "wrong_none", "abstained", "abstain_on_correct",
              "abstain_on_wrong", "abstained_forced", "acted",
              "stale_wrong", "contra_wrong", "stale_wrong_abstained",
              "contra_wrong_abstained", "fixed", "broken", "tie_flips",
              "abstained_adjudicated", "abstained_pending",
              "abstained_merge_support")
HARM_CLASSES = ("stale_wrong", "contra_wrong", "wrong_none")
SPLIT_BUCKETS = ("unpaired", "pre_fork", "pre_resolution",
                 "post_resolution")


def pair_lag_record(p: dict) -> dict:
    resolved = sorted(E for E, v in p["verdict_by_epoch"].items()
                      if v["verdict"] != "ambiguous")
    first = resolved[0] if resolved else None
    final = (p["verdict_by_epoch"][max(p["verdict_by_epoch"])]["verdict"]
             if p["verdict_by_epoch"] else None)
    return {"I": p["I"], "O": p["O"], "fork_epoch": p["epoch"],
            "first_resolved_epoch": first, "final_verdict": final,
            "lag": None if first is None else first - p["epoch"]}


def scan_cell(stem: Path, table_row: dict) -> dict:
    run = load_run(stem)
    router = build_writetime_router(run["events"], run["slot_obs"])
    lags = [pair_lag_record(p) for p in router["pairs"]]
    slot_pairs = defaultdict(list)
    for rec in lags:
        slot_pairs[rec["I"]].append(rec)
        slot_pairs[rec["O"]].append(rec)

    zero = {k: 0 for k in CHECK_KEYS}
    acc = {pol: dict(zero) for pol in SCAN_POLICIES}
    residual = {"residual": 0, "captured": 0}
    nonred = {"captured_wrong": 0, "pending_led": 0, "nonredundant": 0}
    harm = {cls: {b: 0 for b in SPLIT_BUCKETS} for cls in HARM_CLASSES}

    for prow in run["probes"]:
        epoch = int(float(prow["epoch"]))
        pi = int(prow["probe_index"])
        cands = run["topk"][(epoch, pi)]
        truth = int(float(prow["true_label"]))
        deployed = int(float(prow["vote_pred_label"]))
        none_correct = deployed == truth
        is_stale_wrong = prow["stale_lenient"] == "1" and not none_correct
        is_contra_wrong = (prow["contradictory_lenient"] == "1"
                           and not none_correct)
        decisions = {}
        for pol in SCAN_POLICIES:
            ans, acted, detail = apply_policy(
                pol, cands, [], deployed, run["value_dim"], epoch,
                run["slot_obs"], True, router)
            trigger = detail.get("trigger") if detail else None
            decisions[pol] = (ans, trigger)
            m = acc[pol]
            m["n"] += 1
            m["wrong_none"] += int(not none_correct)
            m["acted"] += int(acted)
            m["stale_wrong"] += int(is_stale_wrong)
            m["contra_wrong"] += int(is_contra_wrong)
            if ans is None:
                m["abstained"] += 1
                m["abstain_on_correct"] += int(none_correct)
                m["abstain_on_wrong"] += int(not none_correct)
                m["abstained_forced"] += int(trigger == "forced")
                m["abstained_adjudicated"] += int(trigger == "adjudicated")
                m["abstained_pending"] += int(trigger == "pending")
                m["abstained_merge_support"] += int(
                    trigger == "merge_support")
                m["stale_wrong_abstained"] += int(is_stale_wrong)
                m["contra_wrong_abstained"] += int(is_contra_wrong)
            else:
                m["fixed"] += int(ans == truth and not none_correct)
                m["broken"] += int(none_correct and ans != truth)

        # structural superset: P2's trigger contains merge-abstain's
        if (decisions["merge-abstain"][0] is None
                and decisions[P2][0] is not None):
            raise RuntimeError(f"P2 trigger not a superset of "
                               f"merge-abstain at {stem} epoch {epoch} "
                               f"probe {pi}")
        # P2 residual ledger (class-1 rows the certified trigger misses)
        if is_stale_wrong and decisions["merge-abstain"][0] is not None:
            residual["residual"] += 1
            residual["captured"] += int(decisions[P2][0] is None)
        # pending non-redundancy (evaluated on the one-shot arm)
        if not none_correct and decisions[P3][0] is None:
            nonred["captured_wrong"] += 1
            if decisions[P3][1] == "pending":
                nonred["pending_led"] += 1
                if (decisions[P1][0] is not None
                        and decisions[P2][0] is not None):
                    nonred["nonredundant"] += 1

        # harm split by the leading slot's pair status at the probe epoch
        if not none_correct:
            surv = [c for c in cands if c["surviving"]]
            top1 = max(surv, key=lambda c: float(c["weight"]))["slot"]
            recs = slot_pairs.get(top1)
            if not recs:
                bucket = "unpaired"
            elif all(r["fork_epoch"] > epoch for r in recs):
                bucket = "pre_fork"
            elif any(r["first_resolved_epoch"] is not None
                     and r["fork_epoch"] <= epoch
                     and r["first_resolved_epoch"] <= epoch for r in recs):
                bucket = "post_resolution"
            else:
                bucket = "pre_resolution"
            for cls, flag in (("stale_wrong", is_stale_wrong),
                              ("contra_wrong", is_contra_wrong),
                              ("wrong_none", True)):
                if flag:
                    harm[cls][bucket] += 1

    # cross-check every accumulated counter against the re-emitted table
    for pol in SCAN_POLICIES:
        row = table_row[pol]
        for k in CHECK_KEYS:
            if k not in row:
                if acc[pol][k] != 0:   # merge-abstain lacks PR-11 counters
                    raise RuntimeError(f"{stem}: {pol}.{k} nonzero but "
                                       f"absent from table row")
                continue
            if acc[pol][k] != row[k]:
                raise RuntimeError(f"{stem}: {pol}.{k} scan={acc[pol][k]} "
                                   f"table={row[k]}")

    none_row = table_row["none"]
    correct = none_row["n"] - none_row["wrong_none"]
    policies = {}
    for pol in SCAN_POLICIES:
        m = acc[pol]
        policies[pol] = {
            "abstained": m["abstained"],
            "abstained_adjudicated": m["abstained_adjudicated"],
            "abstained_pending": m["abstained_pending"],
            "abstained_merge_support": m["abstained_merge_support"],
            "abstained_forced": m["abstained_forced"],
            "abstain_on_correct": m["abstain_on_correct"],
            "abstain_on_wrong": m["abstain_on_wrong"],
            "false_abstain_rate": (round(m["abstain_on_correct"] / correct,
                                         6) if correct else None),
            "acted": m["acted"],
            "changed_answers": m["fixed"] + m["broken"],
            "stale_wrong_abstained": m["stale_wrong_abstained"],
            "contra_wrong_abstained": m["contra_wrong_abstained"],
            "capture_stale_wrong": (
                round(m["stale_wrong_abstained"] / m["stale_wrong"], 6)
                if m["stale_wrong"] else None),
            "capture_contra_wrong": (
                round(m["contra_wrong_abstained"] / m["contra_wrong"], 6)
                if m["contra_wrong"] else None),
            "capture_wrong_none": (
                round(m["abstain_on_wrong"] / m["wrong_none"], 6)
                if m["wrong_none"] else None),
        }

    lag_vals = [r["lag"] for r in lags if r["lag"] is not None]
    verdict_counts = defaultdict(int)
    for r in lags:
        verdict_counts[r["final_verdict"] or "none"] += 1
    return {
        "none": {"n": none_row["n"], "wrong_none": none_row["wrong_none"],
                 "correct_traffic": correct,
                 "stale_wrong": none_row["stale_wrong"],
                 "contra_wrong": none_row["contra_wrong"]},
        "policies": policies,
        "p2_residual": dict(residual),
        "pending_nonredundancy": dict(nonred),
        "harm_split": harm,
        "resolution_lag": {
            "n_pairs": len(lags),
            "n_resolved": len(lag_vals),
            "n_never_resolved": len(lags) - len(lag_vals),
            "median_lag": (float(np.median(lag_vals)) if lag_vals
                           else None),
            "max_lag": max(lag_vals) if lag_vals else None,
            "final_verdicts": dict(sorted(verdict_counts.items())),
            "pairs": lags,
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    base = Path("results/issue_failure_mode_blindness")
    ap.add_argument("--pr3c-dir", default=str(base / "pr3c"))
    ap.add_argument("--pr4-dir", default=str(base / "pr4"))
    ap.add_argument("--pr4-table",
                    default=str(base / "pr4/pr4_geometry_table.json"))
    ap.add_argument("--out",
                    default=str(base / "pr11/adjudication_scan.json"))
    args = ap.parse_args()
    pr3c, pr4 = Path(args.pr3c_dir), Path(args.pr4_dir)
    with open(args.pr4_table) as f:
        gov = json.load(f)["governance"]

    cells = {}
    for pair in PAIRS:
        for arm in ARMS:
            for seed in SEEDS:
                name = f"{pair}/{arm}/s{seed}"
                print(f"[pr11] {name}", flush=True)
                cell = scan_cell(run_stem(pair, arm, seed, pr3c, pr4),
                                 gov[name])
                cell["_cell"] = {
                    "pair": pair, "arm": arm, "seed": seed,
                    "pr4_tainted_anchor":
                        gov[name]["_cell"]["tainted_anchor"],
                    "gate_scored": pair != "pairA"}
                cells[name] = cell

    def gate_cells(**kw):
        for name, c in cells.items():
            cc = c["_cell"]
            if not cc["gate_scored"]:
                continue
            if any(cc[k] != v for k, v in kw.items()):
                continue
            yield name, c

    # -- instrumentation gates (hard) ------------------------------------
    clean_actions = {pol: {n: c["policies"][pol]["acted"]
                           + c["policies"][pol]["abstained"]
                           for n, c in cells.items()
                           if c["_cell"]["arm"] == "clean"}
                     for pol in ADJUDICATION_POLICIES}
    instrumentation = {
        "clean_arm_actions_zero": {
            pol: max(v.values()) == 0 for pol, v in clean_actions.items()},
        "changed_answers_zero": {
            pol: max(c["policies"][pol]["changed_answers"]
                     for c in cells.values()) == 0
            for pol in ADJUDICATION_POLICIES},
        "forced_abstentions_zero": {
            pol: max(c["policies"][pol]["abstained_forced"]
                     for c in cells.values()) == 0
            for pol in ADJUDICATION_POLICIES},
    }

    # -- per-policy GO gates (§6) ----------------------------------------
    def worst_false(pol):
        worst_name, worst = None, -1.0
        for name, c in gate_cells():
            r = c["policies"][pol]["false_abstain_rate"]
            if r is not None and r > worst:
                worst_name, worst = name, r
        return {"cell": worst_name, "rate": worst,
                "ok_everywhere": all(
                    (c["policies"][pol]["abstain_on_correct"]
                     <= FALSE_ABSTAIN_MAX_FRAC
                     * c["none"]["correct_traffic"])
                    for _, c in gate_cells())}

    def agg_capture(pol, arm, num_key, den_key):
        out = {}
        for pair in GATE_PAIRS:
            num = den = 0
            for _, c in gate_cells(pair=pair, arm=arm):
                num += c["policies"][pol][num_key]
                den += c["none"][den_key]
            out[pair] = {"captured": num, "class_total": den,
                         "capture": round(num / den, 6) if den else None,
                         "ok": den > 0 and num / den >= CAPTURE_FLOOR}
        return out

    p1_contra = agg_capture(P1, "contra", "contra_wrong_abstained",
                            "contra_wrong")
    p3_contra = agg_capture(P3, "contra", "contra_wrong_abstained",
                            "contra_wrong")
    p3_oneshot = agg_capture(P3, "oneshot", "abstain_on_wrong",
                             "wrong_none")

    p2_cells = {n: c["p2_residual"]
                for n, c in gate_cells(pair="pairD", arm="soft")}
    p2_agg = {"residual": sum(v["residual"] for v in p2_cells.values()),
              "captured": sum(v["captured"] for v in p2_cells.values())}
    p2_s0 = p2_cells["pairD/soft/s0"]
    p2_gate = {
        "per_cell": p2_cells,
        "aggregate": {**p2_agg,
                      "floor": P2_RESIDUAL_FLOOR_AGG,
                      "ok": p2_agg["captured"] >= P2_RESIDUAL_FLOOR_AGG},
        "s0": {**p2_s0, "floor": P2_RESIDUAL_FLOOR_S0,
               "ok": p2_s0["captured"] >= P2_RESIDUAL_FLOOR_S0},
    }

    false_gates = {pol: worst_false(pol) for pol in ADJUDICATION_POLICIES}
    go = {
        P1: (false_gates[P1]["ok_everywhere"]
             and all(v["ok"] for v in p1_contra.values())),
        P2: (false_gates[P2]["ok_everywhere"]
             and p2_gate["aggregate"]["ok"] and p2_gate["s0"]["ok"]),
        P3: (false_gates[P3]["ok_everywhere"]
             and all(v["ok"] for v in p3_oneshot.values())
             and all(v["ok"] for v in p3_contra.values())),
    }

    # -- pending non-redundancy + window verdict (§6) ---------------------
    nr_num = sum(c["pending_nonredundancy"]["nonredundant"]
                 for _, c in gate_cells(arm="oneshot"))
    nr_den = sum(c["pending_nonredundancy"]["captured_wrong"]
                 for _, c in gate_cells(arm="oneshot"))
    nonredundancy = {
        "nonredundant_rows": nr_num,
        "p3_captured_wrong_rows": nr_den,
        "fraction": round(nr_num / nr_den, 6) if nr_den else None,
        "floor": NONREDUNDANT_FLOOR,
        "ok": nr_den > 0 and nr_num / nr_den >= NONREDUNDANT_FLOOR,
    }
    if go[P3] and nonredundancy["ok"]:
        verdict = "window-GO"
    elif any(go.values()):
        verdict = "static-expansion"
    else:
        verdict = "negative"

    out = {
        "design": "PR11_ADJUDICATION_DESIGN.md §5-6 (PR-11.1 scan)",
        "policies": list(ADJUDICATION_POLICIES),
        "sources": {"pr4_table": args.pr4_table,
                    "pr3c_dir": args.pr3c_dir, "pr4_dir": args.pr4_dir},
        "gate_frame": "pairs B-E gate-scored (all seeds); pairA "
                      "report-only tainted anchor",
        "false_abstain_definition": "abstain_on_correct / (n - wrong_none)"
                                    " per run (pr9_abstention_envelope.py)",
        "gates": {
            "instrumentation": instrumentation,
            "false_abstain_worst": false_gates,
            "p1_contra_capture": p1_contra,
            "p2_paird_soft_residual": p2_gate,
            "p3_oneshot_capture": p3_oneshot,
            "p3_contra_capture": p3_contra,
            "go": go,
            "pending_nonredundancy": nonredundancy,
        },
        "verdict": verdict,
        "cells": dict(sorted(cells.items())),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"GO: {go}")
    print(f"pending non-redundancy: {nonredundancy}")
    print(f"verdict: {verdict}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
