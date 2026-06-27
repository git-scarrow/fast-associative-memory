"""PR-7 step 2 — write-path twin-run delta analyzer (analysis-only).

PR7_DESIGN.md §5/§7/§10/§13.2. Reads the committed twin-run arms (an
*ungoverned* ``--govern none`` baseline and a *governed* ``--govern <action>``
arm over the same input stream — see §5) and emits the exact baseline-minus-
governed delta on the five §7 measurands, per panel cell, with a pass/fail
verdict scored against pre-registered margins and the §8 both-shapes rule.

What this module is, and is NOT (the safety boundary, all enforced here):

  * It IMPORTS NO TORCH and TOUCHES NO CACHE — it reads only committed JSON
    (``governance.json`` from the frozen ``analyze_fork_governance`` scorer and
    the run ``summary.json``), exactly like ``pr6_hazard_panel.py``. It runs no
    model, makes no write, changes no engine or retrieval code.
  * Geometry is NEVER a gate (``geometry_used_as_gate == False``). A class set's
    geometry is provenance only — used to check a stem's filename matches its
    measured ``classes``, never to admit, exclude, or certify a cell (PR-5
    step 1 closed static geometry as a safety predictor).
  * The §8 BOTH-SHAPES rule is enforced: a candidate that improves one harm
    shape (D-like *direct*) while worsening another (B/E-like *collateral*, or
    *clean* traffic) is reported ``fail`` — full stop (PR6_DESIGN §5).
  * ``one_shot_ambiguity`` is NEVER scored pass/fail — it stays observe-only
    (PR7_DESIGN §12); it is recorded so it is not silently dropped.
  * Provenance guards (integrity, never a geometric gate): a governed stem's
    filename must match its measured ``classes`` AND its ``govern`` action; a
    non-soft payload is refused where the soft (merge-path) arm is required;
    the baseline arm must carry no governance (the ``none``-vote bit-identity
    PR7_DESIGN §10 retains). With no committed governed arms the analyzer
    degrades gracefully to a "baseline-only, no verdict" state (mirroring the
    PR-6 step-2/step-3 unseeded fallback).

Step 2 runs ONE action (``annotate``, the null-action floor) on ONE cell
(pairA ``clean_control``); the analyzer is written cell-general so step 3 can
score the full panel without code change. With annotate the expected result is
a zero delta on every measurand — the floor proves the harness adds no harm.

Usage (runs on any host; reads committed JSON only, no cache/torch):
  python benchmarks/pr7_twin_delta.py \
      --govern annotate \
      --out results/issue_failure_mode_blindness/pr7/twin_delta.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RESULTS = Path("results/issue_failure_mode_blindness")
TWIN_ROOT = RESULTS / "pr7" / "twin"
PROBE_POLICY = "mode-conditioned-trust"  # the frozen measuring instrument
BASELINE_GOVERN = "none"                  # the ungoverned twin arm
# The write-path actions the twin can score against the none baseline (one at a
# time, PR7_DESIGN §4). `none` is the baseline itself, never a scored action.
CELL_SPEC_GOVERN_ALLOWED = ("annotate", "quarantine", "refuse")
SEEDS = (0, 1, 2)

# Class sets are PROVENANCE ONLY — used to verify a stem's name matches its
# measured `classes`, never to admit/exclude a cell by geometry (PR-5 step 1).
PAIR_CLASS_SETS = {
    "pairA": [0, 8, 19, 33],
    "pairB": [5, 27, 48, 86],
    "pairD": [10, 28, 32, 95],
    "pairE": [47, 56, 61, 76],
}

# The two engine files PR-7 keeps byte-frozen (PR7_DESIGN §1). The analyzer
# records sha256 parity as an integrity field — it gates nothing, but a drift
# here means the twin was scored against a changed engine.
ENGINE_SHA256_BASELINE = {
    "associative_core.py":
        "ce5a2785069905c0cebba43cf7642374a5c2834a60254bee007d5f0ae141c341",
    "fast_associative_memory.py":
        "38a601fc0819af0d19b877a07bc16c839839fb1dacf7b174ec0f2a573660d875",
}

# The frozen #87 two-axis detector's committed fit set (PR7_DESIGN §5 — the
# detector is reproduced from this, never refit from the run under test).
FROZEN_DETECTOR = {
    "fit_csv": "results/issue_failure_mode_blindness/pr3/frozen_fit.csv",
    "fit_summary": "results/issue_failure_mode_blindness/pr3/frozen_fit.summary.json",
}

# Per-cell guard (PR7_DESIGN §6/§8). `guard` is the acceptance direction on the
# frozen-probe harm; `payload_mode` (when set) is a hard provenance requirement.
#   improve       — governed must REDUCE frozen-probe broken by the margin (D)
#   not_worsen    — governed must NOT increase broken beyond tolerance (clean, B/E)
#   capture_stable— write-time capture must hold (192/seed) and D/E not worsen
CELL_SPEC = {
    "clean_control":    {"pairs": ["pairA"],
                         "harm_shape": "none",       "guard": "not_worsen",
                         "payload_mode": None},
    "direct_harm":      {"pairs": ["pairD"],
                         "harm_shape": "direct",     "guard": "improve",
                         "payload_mode": None},
    "collateral_harm":  {"pairs": ["pairB", "pairE"],
                         "harm_shape": "collateral", "guard": "not_worsen",
                         "payload_mode": None},
    "merge_path_stale": {"pairs": ["pairA", "pairD", "pairE", "pairB"],
                         "harm_shape": "write-time-stale-capture",
                         "guard": "capture_stable", "payload_mode": "soft"},
}

# Pre-registered acceptance margins, committed BEFORE any governed run and never
# re-tuned after seeing a result (the PR-4/PR-5 trap; PR7_DESIGN §7). All are on
# the baseline-minus-governed delta of the frozen-probe broken count.
#   improve_min   — a "fix" requires delta >= this (governed reduced harm)
#   worsen_tol    — a regression is delta < -this (governed increased harm)
# annotate is the null-action floor: it must cost EXACTLY nothing, so its
# worsen tolerance is 0 (any increase fails) and it claims no improvement.
PRE_REGISTERED_MARGINS = {
    "annotate":   {"improve_min": 0, "worsen_tol": 0,
                   "rationale": "null-action floor — must cost exactly nothing "
                                "(PR7_DESIGN §4/§8.4)"},
    "quarantine": {"improve_min": 1, "worsen_tol": 0,
                   "rationale": "acting decision — must drain direct harm with "
                                "no collateral/clean regression (step 3)"},
    "refuse":     {"improve_min": 1, "worsen_tol": 0,
                   "rationale": "ceiling action — tested last (step 3)"},
}


# ---------------------------------------------------------------------------
# Reading committed twin arms (provenance-guarded; no torch, no cache)
# ---------------------------------------------------------------------------
def _arm_measurands(gov: dict) -> dict:
    """The five §7 measurands for one committed arm-seed, read verbatim from the
    frozen scorer's governance.json. Deployed-vote quantities come from policy
    ``none`` (= the deployed ``forward()`` vote); harm quantities come from the
    frozen ``mode-conditioned-trust`` probe; capture from the router trace."""
    none = gov["none"]
    mct = gov[PROBE_POLICY]
    router = gov["_router"]
    return {
        # 1. stale-wrong: rows the deployed vote gets wrong on superseded keys.
        "stale_wrong": none["stale_wrong"],
        # 2. contradiction capture: conflict pairs the router classified.
        "contradiction_pairs": router["n_conflict_pairs"],
        # 3+4. frozen-probe harm (collateral / direct decomposition + total).
        "frozen_probe_broken": mct["broken"],
        "frozen_probe_collateral_br": mct["collateral_br"],
        "frozen_probe_direct_br": mct["direct_br"],
        # clean traffic: actions taken on benign rows (false-action proxy).
        "frozen_probe_acted": mct["acted"],
        # 5. merge-path capture: must hold 192/seed across geometry.
        "merge_suspect_events": router["n_merge_suspect_events"],
    }


def read_arm(cell: str, govern: str, pair: str,
             twin_root: Path = TWIN_ROOT) -> dict | None:
    """Read every committed seed of one (cell, govern, pair) twin arm.

    Returns ``{seed: measurands}`` or ``None`` if the arm is not fully
    committed (any seed missing → the whole arm is treated absent, mirroring
    pr6's "a geometry counts only if all its seeds are present" rule). Each
    committed seed is provenance-guarded: stem ↔ classes, stem ↔ govern, and —
    on a soft-required cell — payload_mode == soft (else RuntimeError)."""
    spec = CELL_SPEC[cell]
    arm_dir = twin_root / cell / govern
    if not arm_dir.exists():
        return None
    per_seed: dict[int, dict] = {}
    for seed in SEEDS:
        matches = sorted(arm_dir.glob(f"per_probe_*_s{seed}_{pair}.summary.json"))
        if not matches:
            return None
        if len(matches) > 1:
            raise RuntimeError(
                f"{arm_dir}: seed {seed} pair {pair} matches {len(matches)} "
                f"summary files; the (cell, govern, pair, seed) arm is ambiguous")
        spath = matches[0]
        gpath = spath.with_name(spath.name.replace(".summary.json",
                                                   ".governance.json"))
        if not gpath.exists():
            raise RuntimeError(f"{spath.name}: missing its governance.json twin")
        summ = json.loads(spath.read_text())
        gov = json.loads(gpath.read_text())
        _guard_provenance(cell, govern, pair, spec, summ, spath.name)
        per_seed[seed] = _arm_measurands(gov)
    return per_seed


def _guard_provenance(cell, govern, pair, spec, summ, stem):
    """Integrity guards (PR7_DESIGN §10) — never a geometric gate."""
    expected_classes = PAIR_CLASS_SETS[pair]
    if summ.get("classes") != expected_classes:
        raise RuntimeError(
            f"{stem}: measured classes {summ.get('classes')} != "
            f"{expected_classes} claimed by the stem name (pair {pair})")
    # A soft-required cell refuses a non-soft payload arm.
    if spec["payload_mode"] is not None \
            and summ.get("payload_mode") != spec["payload_mode"]:
        raise RuntimeError(
            f"{stem}: payload_mode {summ.get('payload_mode')!r} != required "
            f"{spec['payload_mode']!r} for cell {cell}")
    # Stem ↔ govern: the baseline arm carries NO governance (the none-vote
    # bit-identity); a governed arm must record its action.
    gprov = summ.get("govern")
    if govern == BASELINE_GOVERN:
        if gprov is not None:
            raise RuntimeError(
                f"{stem}: baseline (none) arm carries a govern block "
                f"{gprov!r}; the baseline must be ungoverned")
    else:
        if not gprov or gprov.get("action") != govern:
            raise RuntimeError(
                f"{stem}: govern provenance {gprov!r} does not match the "
                f"'{govern}' arm directory it lives in")


# ---------------------------------------------------------------------------
# Delta + verdict
# ---------------------------------------------------------------------------
def _seed_delta(base: dict, gov: dict) -> dict:
    """baseline − governed, per measurand, for one seed (PR7_DESIGN §7)."""
    return {k: base[k] - gov[k] for k in base}


def score_cell(cell: str, govern: str,
               twin_root: Path = TWIN_ROOT) -> dict:
    """Score one cell's twin: read both arms, compute per-seed deltas, verdict.

    Observe-only cells are never scored. With no governed arm (or no baseline)
    the cell is ``inconclusive`` with whatever evidence is present — the
    graceful baseline-only fallback."""
    spec = CELL_SPEC[cell]
    margins = PRE_REGISTERED_MARGINS.get(
        govern, {"improve_min": 1, "worsen_tol": 0, "rationale": "default"})

    per_pair: dict[str, dict] = {}
    broken_delta_total = 0
    capture_delta_total = 0
    any_governed = False
    for pair in spec["pairs"]:
        base = read_arm(cell, BASELINE_GOVERN, pair, twin_root)
        gov = read_arm(cell, govern, pair, twin_root)
        if base is None and gov is None:
            continue
        entry: dict = {"baseline_present": base is not None,
                       "governed_present": gov is not None}
        if base is not None and gov is not None:
            any_governed = True
            seeds = sorted(set(base) & set(gov))
            deltas = {s: _seed_delta(base[s], gov[s]) for s in seeds}
            entry["seeds_scored"] = seeds
            entry["delta_by_seed"] = deltas
            entry["baseline_by_seed"] = base
            entry["governed_by_seed"] = gov
            broken_delta_total += sum(deltas[s]["frozen_probe_broken"]
                                      for s in seeds)
            capture_delta_total += sum(deltas[s]["merge_suspect_events"]
                                       for s in seeds)
        elif base is not None:
            entry["baseline_by_seed"] = base
        per_pair[pair] = entry

    verdict, improved, regressed, capture_ok = _cell_verdict(
        spec, margins, broken_delta_total, capture_delta_total, any_governed)

    return {
        "harm_shape": spec["harm_shape"],
        "guard": spec["guard"],
        "pairs": spec["pairs"],
        "payload_mode": spec["payload_mode"],
        "pre_registered_margin": margins,
        "frozen_probe_broken_delta_total": broken_delta_total,
        "merge_suspect_capture_delta_total": capture_delta_total,
        "improved": improved,
        "regressed": regressed,
        "capture_stable": capture_ok,
        "verdict": verdict,
        "per_pair": per_pair,
    }


def _cell_verdict(spec, margins, broken_delta, capture_delta, any_governed):
    """Per-cell verdict from the aggregated deltas. Returns
    (verdict, improved, regressed, capture_ok)."""
    if not any_governed:
        return "inconclusive", False, False, None
    improved = broken_delta >= margins["improve_min"] and broken_delta > 0
    regressed = broken_delta < -margins["worsen_tol"]
    capture_ok = None
    if spec["guard"] == "improve":
        verdict = "pass" if improved else "fail"
    elif spec["guard"] == "not_worsen":
        verdict = "fail" if regressed else "pass"
    elif spec["guard"] == "capture_stable":
        capture_ok = capture_delta == 0
        verdict = "pass" if (capture_ok and not regressed) else "fail"
    else:  # pragma: no cover - guards are enumerated above
        verdict = "inconclusive"
    return verdict, improved, regressed, capture_ok


def _engine_sha256_parity(root: Path) -> bool:
    """True iff the two frozen engine files still hash to the PR-6 baseline.
    Integrity only — this gates nothing, but records that the twin was scored
    against an unchanged engine (PR7_DESIGN §1/§8.6)."""
    for fname, expected in ENGINE_SHA256_BASELINE.items():
        p = root / fname
        if not p.exists():
            return False
        if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            return False
    return True


def build_twin_delta(govern: str, twin_root: Path = TWIN_ROOT,
                     repo_root: Path | None = None) -> dict:
    """Assemble the twin-delta manifest for one governed action (PR7_DESIGN §10).

    Cell-general: scores every CELL_SPEC cell that has committed arms, records
    ``one_shot_ambiguity`` as observe-only (never pass/fail), enforces the §8
    both-shapes rule across cells, and degrades to a baseline-only / no-verdict
    state when no governed arms are committed."""
    if govern == BASELINE_GOVERN:
        raise RuntimeError(
            "twin-delta needs a non-baseline --govern action to compare against "
            "the 'none' baseline; got 'none'")
    if govern not in CELL_SPEC_GOVERN_ALLOWED:
        raise RuntimeError(
            f"--govern {govern!r} is not a scorable write-path action "
            f"{tuple(CELL_SPEC_GOVERN_ALLOWED)}")

    cells = {cell: score_cell(cell, govern, twin_root) for cell in CELL_SPEC}

    # one-shot ambiguity: recorded, NEVER scored pass/fail (PR7_DESIGN §12).
    cells["one_shot_ambiguity"] = {
        "harm_shape": "ambiguous-evidence (observe-only)",
        "guard": None,
        "verdict": "observe_only",
        "note": ("certified insufficient evidence; carries no hazard label and "
                 "must never be scored pass/fail (PR7_DESIGN §12)."),
    }

    scored = {c: v for c, v in cells.items()
              if v["verdict"] not in ("observe_only", "inconclusive")}
    any_governed = bool(scored)

    # §8 both-shapes rule: improving one harm shape while worsening another
    # (collateral or clean) fails, full stop.
    improved_shapes = sorted({cells[c]["harm_shape"] for c in scored
                              if cells[c].get("improved")})
    regressed_shapes = sorted({cells[c]["harm_shape"] for c in scored
                               if cells[c].get("regressed")})
    both_shapes_ok = not (improved_shapes and regressed_shapes)

    if not any_governed:
        overall = "baseline_only"
    elif not both_shapes_ok:
        overall = "fail"
    elif all(cells[c]["verdict"] == "pass" for c in scored):
        overall = "pass"
    else:
        overall = "fail"

    root = repo_root or Path(".")
    return {
        "design": "PR7_DESIGN.md §5 — write-path twin-run",
        "engine_or_retrieval_change": False,
        "geometry_used_as_gate": False,
        "deployed_engine_sha256_parity": _engine_sha256_parity(root),
        "frozen_detector": dict(FROZEN_DETECTOR),
        "govern_action": govern,
        "probe_policy": PROBE_POLICY,
        "new_cache_runs": _count_governed_arms(cells),
        "cells": cells,
        "scored_cells": sorted(scored),
        "both_shapes_ok": both_shapes_ok,
        "both_shapes_detail": {
            "improved_harm_shapes": improved_shapes,
            "regressed_harm_shapes": regressed_shapes,
            "rule": ("a candidate that improves one harm shape while worsening "
                     "another fails, full stop (PR7_DESIGN §8.5)."),
        },
        "overall_verdict": overall,
        "conclusions_enforced": [
            "static geometry cannot certify safety (PR-5 step 1)",
            "both D-like direct and B/E-like collateral harm remain required",
            "one-shot ambiguity stays observe-only (never scored pass/fail)",
            "merge-path stale capture must hold 192/seed across geometry",
            "slot-granularity / read-time trust stays closed; the frozen probe "
            "is only the measurand, never a deployment candidate",
            "deployed retrieval remains unchanged",
        ],
        "scope": ("certifies nothing beyond the enumerated cells; no "
                  "generalization to unseen geometry"),
    }


def _count_governed_arms(cells: dict) -> int:
    n = 0
    for cell, v in cells.items():
        for pair, entry in v.get("per_pair", {}).items():
            if entry.get("governed_present"):
                n += len(entry.get("seeds_scored", []))
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--govern", required=True, choices=CELL_SPEC_GOVERN_ALLOWED,
                    help="the governed action to compare against the 'none' "
                         "baseline (one action at a time; PR7_DESIGN §4)")
    ap.add_argument("--twin-root", default=str(TWIN_ROOT),
                    help="root of the committed twin arms (default: %(default)s)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    delta = build_twin_delta(args.govern, Path(args.twin_root))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(delta, indent=1, sort_keys=True) + "\n")

    print(f"govern={delta['govern_action']} "
          f"overall={delta['overall_verdict']} "
          f"both_shapes_ok={delta['both_shapes_ok']} "
          f"new_cache_runs={delta['new_cache_runs']} "
          f"engine_sha256_parity={delta['deployed_engine_sha256_parity']}")
    for cell in sorted(delta["cells"]):
        print(f"  {cell:18} {delta['cells'][cell]['verdict']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
