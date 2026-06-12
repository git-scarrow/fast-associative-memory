"""PR-4 step 4 — geometry-sweep governance table + §6 criteria evaluation.

Scores every (pair, arm, seed) cell of the PR-4 sweep (PR4_DESIGN.md §4,
pair set fixed at gate 1, Addendum A.3) with the PR-3c shadow-governance
scorer — extended at step 3 with the guarded variants, the §3 collateral
decomposition and the §1 tie-flip column — and evaluates the
pre-registered §6 acceptance criteria mechanically. Pair A and pair B s0
cells are the PR-3c artifacts reused verbatim (leakage-tainted anchors,
§7: they anchor the sweep but are EXCLUDED from H1/H3 promotion).

Operationalizations of §6 phrases (fixed here, before looking at any
PR-4 output):

* collateral bar (per run): policy ``broken`` <= 1% of `none`'s correct
  rows (n - wrong_none), the same bar trust failed on pair B in PR-3c.
* contra containment: contra arm, contra_wrong_fixed - broken > 0, bar.
* mixed improvement: mixed arm, accuracy_effective > none's, bar.
* stale zero-harm: stale arm, broken == 0.
* merge-path capture: soft arm, stale_wrong_abstained >= 95% of
  stale_wrong, abstain_on_correct <= 5, broken == 0.
* one-shot honesty: oneshot arm, broken == 0 and tie_flips <= 204 (the
  PR-3c maximum side-effect level, PR3C_RESULT.md §2).
* clean zero-harm: clean arm, acted == 0 and broken == 0 and
  abstained == 0.
* strict domination (per mixed/contra run, vs each §6 baseline): on the
  joint (target-mode-fix, collateral) pair, fix = target-mode rows
  answered correctly (contra_wrong_fixed on contra, fixed on mixed) and
  collateral = broken + abstain_on_correct (abstaining policies harm by
  withholding correct answers — PR-3c counted false abstentions as harm
  throughout); the candidate weakly improves both and strictly improves
  at least one.

H2 monotonicity: mean mixed-arm collateral_br per pair (candidate
variants), checked non-decreasing in the gate-1 static compression index
over the pair sweep.

Usage:
  python benchmarks/analyze_pr4_geometry.py \
      --pr3c-dir results/issue_failure_mode_blindness/pr3c \
      --pr4-dir  results/issue_failure_mode_blindness/pr4 \
      --index    results/issue_failure_mode_blindness/pr4/compression_index.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.analyze_fork_governance import (  # noqa: E402
    load_run, score_run)
from benchmarks.score_frozen_detector import (  # noqa: E402
    FIT_CSV, FIT_SUMMARY, reproduce_frozen_detector)

PAIRS = ["pairA", "pairB", "pairC", "pairD", "pairE"]
ARMS = ["clean", "contra", "stale", "mixed", "oneshot", "soft"]
SEEDS = [0, 1, 2]
ARM_STEM = {"clean": "clean", "contra": "contra", "stale": "stale",
            "mixed": "mixed", "oneshot": "stale-oneshot",
            "soft": "stale-soft"}

# §7: cells the trust rule saw during PR-3c — anchors, never promotion.
TAINTED = {("pairA", arm, s) for arm in ARMS for s in SEEDS} | {
    ("pairB", arm, 0) for arm in ("clean", "contra", "stale", "mixed")}

CANDIDATES = ("mode-conditioned-trust", "trust-downweight", "trust-guarded")
BASELINES = ("mode-conditioned-observe", "recency-naive",
             "quarantine-naive", "entropy-abstain")
ONESHOT_FLIP_CEILING = 204  # PR-3c max (stale-oneshot s1)


def run_stem(pair: str, arm: str, seed: int,
             pr3c: Path, pr4: Path) -> Path:
    """Pinned cell -> artifact mapping (PR-3c anchors reused verbatim)."""
    a = ARM_STEM[arm]
    if pair == "pairA":
        return pr3c / f"per_probe_{a}_s{seed}.csv"
    if (pair, arm, seed) in TAINTED:                  # pairB s0 four arms
        return pr3c / f"per_probe_{a}_s{seed}_pairB.csv"
    return pr4 / f"per_probe_{a}_{pair}_s{seed}.csv"


def load_indices(index_json: Path) -> dict[str, float]:
    idx = json.loads(index_json.read_text())
    return {"pairA": idx["pair_A"]["index"],
            "pairB": idx["pair_B"]["index"],
            "pairC": idx["selected"]["pairC_alike"]["index"],
            "pairD": idx["selected"]["pairD_mid"]["index"],
            "pairE": idx["selected"]["pairE_compressed"]["index"]}


# ---------------------------------------------------------------------------
# §6 per-cell checks (each takes the policy metrics + the run's `none` row)
# ---------------------------------------------------------------------------
def bar_ok(m: dict, none: dict) -> bool:
    correct_rows = none["n"] - none["wrong_none"]
    return m["broken"] <= 0.01 * correct_rows


def cell_checks(arm: str, m: dict, none: dict) -> dict:
    """Raw §6 booleans for one (run, policy) cell, by arm."""
    out = {"bar": bar_ok(m, none), "broken": m["broken"]}
    if arm == "contra":
        out["contra_containment"] = (
            m["contra_wrong_fixed"] - m["broken"] > 0 and out["bar"])
    elif arm == "mixed":
        out["mixed_improvement"] = (
            m["accuracy_effective"] > none["accuracy_effective"]
            and out["bar"])
        out["collateral_br"] = m["collateral_br"]
        out["direct_br"] = m["direct_br"]
    elif arm == "stale":
        out["stale_zero_harm"] = m["broken"] == 0
    elif arm == "soft":
        cap = (m["stale_wrong_abstained"] / m["stale_wrong"]
               if m["stale_wrong"] else 1.0)
        out["merge_capture"] = (cap >= 0.95
                                and m["abstain_on_correct"] <= 5
                                and m["broken"] == 0)
        out["capture_rate"] = round(cap, 4)
        out["false_abstentions"] = m["abstain_on_correct"]
    elif arm == "oneshot":
        out["oneshot_honesty"] = (m["broken"] == 0
                                  and m["tie_flips"] <= ONESHOT_FLIP_CEILING)
        out["tie_flips"] = m["tie_flips"]
    elif arm == "clean":
        out["clean_zero_harm"] = (m["acted"] == 0 and m["broken"] == 0
                                  and m["abstained"] == 0)
    return out


def dominates(cand: dict, base: dict, arm: str) -> bool:
    """Strict domination on the joint (target-mode-fix, collateral) pair.

    Collateral = broken + abstain_on_correct, so abstain-heavy baselines
    (entropy-abstain withholds hundreds of correct rows per run) pay for
    their harm channel just as intervening policies pay for theirs.
    """
    fix_key = "contra_wrong_fixed" if arm == "contra" else "fixed"
    c_harm = cand["broken"] + cand["abstain_on_correct"]
    b_harm = base["broken"] + base["abstain_on_correct"]
    ge = cand[fix_key] >= base[fix_key] and c_harm <= b_harm
    gt = cand[fix_key] > base[fix_key] or c_harm < b_harm
    return ge and gt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pr3c-dir", required=True)
    ap.add_argument("--pr4-dir", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--fit-csv", default=FIT_CSV)
    ap.add_argument("--fit-summary", default=FIT_SUMMARY)
    args = ap.parse_args()
    pr3c, pr4 = Path(args.pr3c_dir), Path(args.pr4_dir)
    indices = load_indices(Path(args.index))
    det, thr, provenance = reproduce_frozen_detector(
        Path(args.fit_csv), Path(args.fit_summary))

    table, checks = {}, {}
    for pair in PAIRS:
        for arm in ARMS:
            for seed in SEEDS:
                stem = run_stem(pair, arm, seed, pr3c, pr4)
                name = f"{pair}/{arm}/s{seed}"
                print(f"[pr4] {name}", flush=True)
                scored = score_run(load_run(stem), det, thr)
                none = scored["none"]
                table[name] = scored
                table[name]["_cell"] = {
                    "pair": pair, "arm": arm, "seed": seed,
                    "tainted_anchor": (pair, arm, seed) in TAINTED,
                    "compression_index": indices[pair],
                    "witness_rate": round(
                        none["witness_probes"] / none["n"], 6)}
                checks[name] = {
                    pol: cell_checks(arm, scored[pol], none)
                    for pol in CANDIDATES + BASELINES}
                if arm in ("contra", "mixed"):
                    for pol in CANDIDATES:
                        checks[name][pol]["dominates"] = {
                            b: dominates(scored[pol], scored[b], arm)
                            for b in BASELINES}

    fresh = [(p, a, s) for p in PAIRS for a in ARMS for s in SEEDS
             if (p, a, s) not in TAINTED]

    def cells(pair=None, arm=None, pol=None, tainted_ok=False):
        for (p, a, s) in (fresh if not tainted_ok else
                          [(p, a, s) for p in PAIRS for a in ARMS
                           for s in SEEDS]):
            if (pair and p != pair) or (arm and a != arm):
                continue
            yield f"{p}/{a}/s{s}", checks[f"{p}/{a}/s{s}"][pol]

    # H1 — frozen trust on data it has never seen: pair C (the A-like
    # pair) all arms/seeds + the fresh pair-B completion seeds.
    h1_scope = [n for n, _ in cells(pair="pairC",
                                    pol="mode-conditioned-trust")]
    h1_fail = {}
    arm_key = {"contra": "contra_containment", "mixed": "mixed_improvement",
               "stale": "stale_zero_harm", "soft": "merge_capture",
               "oneshot": "oneshot_honesty", "clean": "clean_zero_harm"}
    for name in h1_scope:
        arm = name.split("/")[1]
        c = checks[name]["mode-conditioned-trust"]
        if not c[arm_key[arm]]:
            h1_fail[name] = c
    h1 = {"scope": h1_scope, "failures": h1_fail, "pass": not h1_fail}

    # H2 — (collateral br x index x variant) attribution table, mixed arm,
    # per-pair means over seeds (anchors included, marked).
    h2_rows = []
    for pair in PAIRS:
        row = {"pair": pair, "index": indices[pair],
               "anchor": pair in ("pairA", "pairB")}
        for pol in CANDIDATES:
            vals = [checks[f"{pair}/mixed/s{s}"][pol]["collateral_br"]
                    for s in SEEDS]
            row[pol] = {"collateral_br_by_seed": vals,
                        "mean": round(sum(vals) / len(vals), 2)}
        h2_rows.append(row)
    h2_rows.sort(key=lambda r: r["index"])
    seq = [r["mode-conditioned-trust"]["mean"] for r in h2_rows]
    h2 = {"rows": h2_rows,
          "trust_collateral_monotone_in_index": all(
              a <= b for a, b in zip(seq, seq[1:]))}

    # H3 — a fully pre-specified variant meeting every §6 criterion on
    # every FRESH cell (anchors excluded from promotion), plus strict
    # domination over every baseline on every fresh mixed/contra cell.
    h3 = {}
    for pol in CANDIDATES:
        fails = {}
        for name, c in cells(pol=pol):
            arm = name.split("/")[1]
            ok = c[arm_key[arm]]
            if arm in ("contra", "mixed"):
                ok = ok and all(c["dominates"].values())
            if not ok:
                fails[name] = c
        h3[pol] = {"pass": not fails, "failures": fails}

    out = {"design": "PR4_DESIGN.md §6 evaluation (step 4)",
           "frozen_detector": provenance,
           "indices": indices,
           "oneshot_flip_ceiling": ONESHOT_FLIP_CEILING,
           "tainted_anchors": sorted(f"{p}/{a}/s{s}"
                                     for (p, a, s) in TAINTED),
           "h1": h1, "h2": h2, "h3": h3,
           "checks": checks, "governance": table}
    out_path = pr4 / "pr4_geometry_table.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"H1 (frozen trust, fresh data): "
          f"{'PASS' if h1['pass'] else 'FAIL'} "
          f"({len(h1_fail)} failing cells)")
    print(f"H2 trust collateral monotone in index: "
          f"{h2['trust_collateral_monotone_in_index']}")
    for pol in CANDIDATES:
        print(f"H3 {pol}: {'PASS' if h3[pol]['pass'] else 'FAIL'} "
              f"({len(h3[pol]['failures'])} failing cells)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
