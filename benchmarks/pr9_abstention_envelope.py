"""PR-9.1(b) — merge-abstain abstention envelope for PR-10 pre-registration.

Executes PR9_ENVELOPE_DESK_CHECK.md §3.2-3: after the registered scorer
change added `merge-abstain` (merge-suspect abstention ALONE) plus the
per-trigger abstention counters, this script reads ONLY the two re-emitted
governance tables — no per-run artifact, no cache, no engine import — and
extracts the exact envelope of the policy per (pair, arm, seed):

  * stale-wrong capture (``stale_wrong_abstained / stale_wrong``),
  * false abstentions on correct rows (``abstain_on_correct``) and the
    rate against correct traffic,
  * abstentions on wrong rows (``abstain_on_wrong``),
  * changed answers on answered rows — FIELD-VERIFIED: ``fixed == broken
    == tie_flips == 0``, ``answered == n - abstained`` and
    ``answered_correct == (n - wrong_none) - abstain_on_correct`` (every
    non-abstained row keeps the deployed answer and its correctness),
  * actions on clean arms (``acted``),
  * the trigger decomposition (``abstained == abstained_merge`` and
    ``abstained_forced == 0`` — merge-abstain never excludes, so a forced
    abstention is structurally impossible).

The verdict frame is the PR-4 grid (5 pairs x 6 arms x 3 seeds) from
``pr4_geometry_table.json``; `tainted_anchor` cells (all pair A, pair B s0
clean/contra/stale/mixed — PR4_DESIGN §7 leakage) are extracted but
EXCLUDED from bound verdicts and reported separately. The PR-3c-only runs
(pair A jitter arms) are reported in their own section, anchor-status
pair A. Every cell present in both tables is cross-checked for exact
equality of the merge-abstain row (same artifacts, same scorer — any
mismatch raises).

Pre-registered bounds evaluated (task order = memo order):
  (i)   0 changed answers everywhere;
  (ii)  soft-arm stale-wrong capture >= 0.95 on pairs A/C, >= 0.79 on
        pairs D/E (pair B soft has no pre-registered bound: reported,
        not scored);
  (iii) 0 actions on clean arms;
  (iv)  false abstentions <= 5% of correct traffic per run.

Usage:
  python benchmarks/pr9_abstention_envelope.py \
      --pr3c-table results/issue_failure_mode_blindness/pr3c/pr3c_governance_table.json \
      --pr4-table  results/issue_failure_mode_blindness/pr4/pr4_geometry_table.json \
      --out        results/issue_failure_mode_blindness/pr9/abstention_envelope.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

POLICY = "merge-abstain"

# PR-3c run-name -> (pair, arm) frame used by the PR-4 grid.
PR3C_ARM = {"clean": "clean", "contra": "contra", "stale": "stale",
            "mixed": "mixed", "stale-oneshot": "oneshot",
            "stale-soft": "soft", "stale-jitter0.05": "jitter0.05",
            "stale-jitter0.15": "jitter0.15"}
PR3C_RE = re.compile(r"^(?P<arm>.+)_s(?P<seed>\d+)(?P<pb>_pairB)?$")

CAPTURE_FLOOR = {"pairA": 0.95, "pairC": 0.95, "pairD": 0.79,
                 "pairE": 0.79}          # pairB: reported, not scored
FALSE_ABSTAIN_MAX_FRAC = 0.05


def cell_record(m: dict, none: dict, router: dict | None) -> dict:
    correct = m["n"] - m["wrong_none"]
    answered_ok = (m["answered"] == m["n"] - m["abstained"]
                   and m["answered_correct"] == correct
                   - m["abstain_on_correct"])
    return {
        "n": m["n"],
        "wrong_none": m["wrong_none"],
        "correct_traffic": correct,
        "stale_wrong": m["stale_wrong"],
        "stale_wrong_abstained": m["stale_wrong_abstained"],
        "capture": (round(m["stale_wrong_abstained"] / m["stale_wrong"], 6)
                    if m["stale_wrong"] else None),
        "abstained": m["abstained"],
        "abstained_merge": m["abstained_merge"],
        "abstained_forced": m["abstained_forced"],
        "abstain_on_correct": m["abstain_on_correct"],
        "abstain_on_wrong": m["abstain_on_wrong"],
        "false_abstain_rate": (round(m["abstain_on_correct"] / correct, 6)
                               if correct else None),
        "acted": m["acted"],
        "fixed": m["fixed"],
        "broken": m["broken"],
        "tie_flips": m["tie_flips"],
        "changed_answers": m["fixed"] + m["broken"],
        "field_verified_no_changed_answers": (
            m["fixed"] == 0 and m["broken"] == 0 and m["tie_flips"] == 0
            and answered_ok),
        "merge_only_trigger": (m["abstained"] == m["abstained_merge"]
                               and m["abstained_forced"] == 0),
        "none_row_consistent": (none["n"] == m["n"]
                                and none["wrong_none"] == m["wrong_none"]),
        "n_merge_suspect_events": (router or {}).get(
            "n_merge_suspect_events"),
    }


def bounds_for(cell: dict, pair: str, arm: str) -> dict:
    out = {"i_zero_changed_answers":
           cell["changed_answers"] == 0
           and cell["field_verified_no_changed_answers"]}
    if arm == "soft" and pair in CAPTURE_FLOOR:
        out["ii_soft_capture"] = (cell["capture"] is not None
                                  and cell["capture"]
                                  >= CAPTURE_FLOOR[pair])
    if arm == "clean":
        out["iii_zero_clean_actions"] = (cell["acted"] == 0
                                         and cell["abstained"] == 0)
    out["iv_false_abstain_le_5pct"] = (
        cell["abstain_on_correct"]
        <= FALSE_ABSTAIN_MAX_FRAC * cell["correct_traffic"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    base = Path("results/issue_failure_mode_blindness")
    ap.add_argument("--pr3c-table",
                    default=str(base / "pr3c/pr3c_governance_table.json"))
    ap.add_argument("--pr4-table",
                    default=str(base / "pr4/pr4_geometry_table.json"))
    ap.add_argument("--out",
                    default=str(base / "pr9/abstention_envelope.json"))
    args = ap.parse_args()

    with open(args.pr4_table) as f:
        pr4 = json.load(f)["governance"]
    with open(args.pr3c_table) as f:
        pr3c = json.load(f)["governance"]

    grid, anchors, jitter = {}, {}, {}
    for name, scored in pr4.items():
        c = scored["_cell"]
        cell = cell_record(scored[POLICY], scored["none"],
                           scored.get("_router"))
        cell["_cell"] = {k: c[k] for k in ("pair", "arm", "seed",
                                           "tainted_anchor")}
        cell["bounds"] = bounds_for(cell, c["pair"], c["arm"])
        (anchors if c["tainted_anchor"] else grid)[name] = cell

    # PR-3c runs: cross-check duplicates against the PR-4 anchors, keep
    # the jitter arms (absent from the PR-4 grid) as their own section.
    for name, scored in pr3c.items():
        mm = PR3C_RE.match(name)
        arm = PR3C_ARM[mm.group("arm")]
        pair = "pairB" if mm.group("pb") else "pairA"
        seed = int(mm.group("seed"))
        cell = cell_record(scored[POLICY], scored["none"],
                           scored.get("_router"))
        key = f"{pair}/{arm}/s{seed}"
        if arm.startswith("jitter"):
            cell["_cell"] = {"pair": pair, "arm": arm, "seed": seed,
                             "tainted_anchor": True}
            cell["bounds"] = bounds_for(cell, pair, arm)
            jitter[key] = cell
            continue
        ref = pr4[key][POLICY]
        if scored[POLICY] != ref:
            raise RuntimeError(
                f"pr3c/pr4 merge-abstain row mismatch on shared cell {key}")

    def verdict(cells: dict) -> dict:
        fails = {}
        for name, cell in sorted(cells.items()):
            bad = [b for b, ok in cell["bounds"].items() if not ok]
            if bad:
                fails[name] = bad
        return {"pass": not fails, "n_cells": len(cells),
                "failures": fails}

    fresh_soft = {n: c for n, c in grid.items()
                  if c["_cell"]["arm"] == "soft"}
    ceilings = {
        "per_pair_soft_capture_floor": {},
        "per_pair_soft_false_abstain_ceiling": {},
        "max_false_abstain_rate_any_fresh_cell": max(
            c["false_abstain_rate"] for c in grid.values()),
        "max_abstentions_off_soft_arms": max(
            c["abstained"] for c in grid.values()
            if c["_cell"]["arm"] != "soft"),
        "max_clean_actions": max(c["acted"] for c in grid.values()
                                 if c["_cell"]["arm"] == "clean"),
        "max_changed_answers": max(c["changed_answers"]
                                   for c in grid.values()),
        "max_forced_abstentions": max(c["abstained_forced"]
                                      for c in grid.values()),
    }
    for pair in ("pairB", "pairC", "pairD", "pairE"):
        caps = {n: c for n, c in fresh_soft.items()
                if c["_cell"]["pair"] == pair}
        ceilings["per_pair_soft_capture_floor"][pair] = min(
            c["capture"] for c in caps.values())
        ceilings["per_pair_soft_false_abstain_ceiling"][pair] = max(
            c["abstain_on_correct"] for c in caps.values())

    out = {
        "policy": POLICY,
        "sources": {"pr3c_table": args.pr3c_table,
                    "pr4_table": args.pr4_table},
        "bounds": {
            "i": "0 changed answers everywhere (field-verified)",
            "ii": ("soft-arm stale-wrong capture >= 0.95 (pairA/pairC), "
                   ">= 0.79 (pairD/pairE); pairB reported unscored"),
            "iii": "0 actions on clean arms",
            "iv": "false abstentions <= 5% of correct traffic per run",
        },
        "verdict_fresh": verdict(grid),
        "verdict_tainted_anchors_report_only": verdict(anchors),
        "verdict_pr3c_jitter_report_only": verdict(jitter),
        "cross_table_consistency": "pr3c==pr4 merge-abstain rows on all "
                                   "shared cells (raise on mismatch)",
        "pr10_ceilings_fresh_cells": ceilings,
        "cells_fresh": dict(sorted(grid.items())),
        "cells_tainted_anchors": dict(sorted(anchors.items())),
        "cells_pr3c_jitter": dict(sorted(jitter.items())),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"fresh: {out['verdict_fresh']}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
