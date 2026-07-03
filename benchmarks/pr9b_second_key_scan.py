"""PR-9B second-key desk scan (PR9B_SECOND_KEY_DESK_SCAN.md §7).

Analysis-only adjudication of §9B condition (b): does a parameter-free
binary second key exist in committed write-time observables? Reads ONLY
committed artifacts (the §9A shadow ledgers, the PR-7 refuse twin-delta,
the PR-11.1 scan, the PR-6 panel); imports no engine; runs on darwin;
performs no run, no tuning, no policy change.

Everything scored here is defined in the pre-registered memo sections
(§1 denominator, §2 candidates, §4 scoring, §5 gate). The verdict mapping
is fixed: second-key-candidate / second-key-failed / second-key-absent.
Any denominator deviation (count drift, key collision, constancy failure)
is a STOP condition: the scan raises instead of emitting a verdict.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "issue_failure_mode_blindness"
PANEL = RESULTS / "pr9_2" / "panel"
REFUSE_TWIN = RESULTS / "pr7" / "twin_delta_refuse.json"
PR11_SCAN = RESULTS / "pr11" / "adjudication_scan.json"
OUT_PATH = RESULTS / "pr9b" / "second_key_scan.json"

# §1: the admissible key-input columns are EXACTLY the §9A-certified
# state-free set (PR92_INTRINSIC_JOIN_KEY + PR92_INTRINSIC_CHECK_FIELDS in
# benchmarks/failure_mode_probe.py). Pinned by test against those constants.
ADMISSIBLE_COLUMNS = ("epoch", "event_class", "batch_index", "payload_label")

# §3: pre-registered cell assignment (committed frozen-probe hazard,
# twin_delta_refuse.json baselines: broken D=338 / E=138 vs A=0 / B=1).
TARGET_PAIRS = ("pairD", "pairE")
WRITE_CLEAN_PAIRS = ("pairA", "pairB")
SEEDS = (0, 1, 2)

# §5: pre-registered bounds (PR-11 precedent floor/ceiling).
CAPTURE_FLOOR = 0.5
FALSE_ACTION_CEILING = 0.05

VERDICT_CANDIDATE = "second-key-candidate"
VERDICT_FAILED = "second-key-failed"
VERDICT_ABSENT = "second-key-absent"


class StopCondition(RuntimeError):
    """Denominator deviation — the scan must stop without a verdict."""


# ---------------------------------------------------------------------------
# §1 — denominator (deduplicated shadow-ledger event records)
# ---------------------------------------------------------------------------
def load_denominator(panel: Path = PANEL) -> dict:
    """One entry per unique run stem -> the ledger's flagged_event_records.

    The same stem committed under several hazard cells is the same run
    (byte-identical, certified in PR-9.2); dedup keeps the first and
    verifies the duplicates carry identical records.
    """
    runs: dict[str, list[dict]] = {}
    for p in sorted(panel.glob("*/shadow/*.summary.json")):
        stem = p.name.replace(".summary.json", "")
        recs = json.loads(p.read_text())["govern"]["quarantine_ledger"][
            "flagged_event_records"]
        if stem in runs:
            if runs[stem] != recs:
                raise StopCondition(
                    f"duplicate stem {stem} differs across cells")
            continue
        runs[stem] = recs
    if not runs:
        raise StopCondition(f"no shadow summaries under {panel}")
    return runs


def _pair_seed(stem: str) -> tuple[str, int] | None:
    """(pair, seed) for stale-soft stems; None for clean stems."""
    if "stale-soft" not in stem:
        return None
    pair = stem.rsplit("_", 1)[-1]
    seed = int(re.search(r"_s(\d+)_", stem).group(1))
    return pair, seed


def verify_denominator(runs: dict) -> dict:
    """§7 step 1: counts, uniqueness, constancy, lemma premise.

    Raises StopCondition on any deviation; returns the verification report.
    """
    stale = {s: r for s, r in runs.items() if _pair_seed(s) is not None}
    clean = {s: r for s, r in runs.items() if _pair_seed(s) is None}
    if len(stale) != 12 or len(clean) != 6:
        raise StopCondition(
            f"expected 12 stale-soft + 6 clean unique runs, got "
            f"{len(stale)} + {len(clean)}")
    for stem, recs in clean.items():
        if recs:
            raise StopCondition(f"clean run {stem} has {len(recs)} events")
    coord_sets, label_sets = set(), set()
    for stem, recs in stale.items():
        if len(recs) != 192:
            raise StopCondition(f"{stem}: {len(recs)} events, expected 192")
        keys = [(r["epoch"], r["event_class"], r["batch_index"])
                for r in recs]
        if len(set(keys)) != len(keys):
            raise StopCondition(f"{stem}: intrinsic-key collision")
        classes = {r["event_class"] for r in recs}
        if classes != {"supersession"}:
            raise StopCondition(f"{stem}: event_class not constant: {classes}")
        labels = frozenset(r["payload_label"] for r in recs)
        label_sets.add((stem, labels))
        coord_sets.add(frozenset((r["epoch"], r["batch_index"])
                                 for r in recs))
    lemma_premise = len(coord_sets) == 1
    payload_constant_per_run = all(len(ls) == 1 for _s, ls in label_sets)
    payload_equal_across_runs = len({ls for _s, ls in label_sets}) == 1
    some = next(iter(stale.values()))
    epochs = sorted({r["epoch"] for r in some})
    return {
        "unique_stale_soft_runs": len(stale),
        "unique_clean_runs": len(clean),
        "events_per_stale_run": 192,
        "total_events": 192 * len(stale),
        "eligible_epochs": epochs,
        "batch_indices": [min(r["batch_index"] for r in some),
                          max(r["batch_index"] for r in some)],
        "event_class_constant": True,
        "payload_label_constant_per_run": payload_constant_per_run,
        "payload_label_equal_across_runs": payload_equal_across_runs,
        "coordinate_multiset_identical_across_runs": lemma_premise,
    }


# ---------------------------------------------------------------------------
# §2 — candidates. Predicates take one ledger record; first/last eligible
# epoch are derived from the denominator (protocol-distinguished positions,
# not tuned constants).
# ---------------------------------------------------------------------------
def build_candidates(runs: dict) -> list[dict]:
    stale = {s: r for s, r in runs.items() if _pair_seed(s) is not None}
    all_epochs = sorted({r["epoch"] for recs in stale.values() for r in recs})
    first, last = all_epochs[0], all_epochs[-1]
    return [
        {"id": "K0", "definition": "true (act on every eligible event)",
         "second_key": False,
         "exclusion": "constant on the denominator — restates the first "
                      "key; committed acting outcome = PR-7 refuse",
         "mechanism": None,
         "fn": lambda r: True},
        {"id": "K1", "definition": f"epoch == {first} (first eligible epoch)",
         "second_key": True, "exclusion": None,
         "mechanism": "only self-transporting predicate (state identical "
                      "under acting divergence only on the first eligible "
                      "epoch — §9A 32/192 finding)",
         "fn": lambda r: r["epoch"] == first},
        {"id": "K2", "definition": f"epoch > {first} (complement of K1)",
         "second_key": True, "exclusion": None,
         "mechanism": "formal complement closure of K1",
         "fn": lambda r: r["epoch"] > first},
        {"id": "K3", "definition": f"epoch == {last} (last eligible epoch)",
         "second_key": True, "exclusion": None,
         "mechanism": "distinguished position only; no mechanism statable",
         "fn": lambda r: r["epoch"] == last},
        {"id": "K4", "definition": "batch_index == 0",
         "second_key": True, "exclusion": None,
         "mechanism": "distinguished position only; no mechanism statable",
         "fn": lambda r: r["batch_index"] == 0},
    ]


# ---------------------------------------------------------------------------
# §4 — scoring
# ---------------------------------------------------------------------------
def score_candidate(cand: dict, runs: dict) -> dict:
    per_cell: dict[str, dict] = {}
    fired_sets = set()
    for stem, recs in sorted(runs.items()):
        ps = _pair_seed(stem)
        fired = [r for r in recs if cand["fn"](r)]
        if ps is not None:
            fired_sets.add(frozenset(
                (r["epoch"], r["batch_index"]) for r in fired))
        per_cell[stem] = {"eligible": len(recs), "fired": len(fired)}
    agg = {p: {"fired": 0, "eligible": 0}
           for p in TARGET_PAIRS + WRITE_CLEAN_PAIRS}
    clean_fired = 0
    for stem, cell in per_cell.items():
        ps = _pair_seed(stem)
        if ps is None:
            clean_fired += cell["fired"]
            continue
        pair, _seed = ps
        agg[pair]["fired"] += cell["fired"]
        agg[pair]["eligible"] += cell["eligible"]

    def frac(p):
        e = agg[p]["eligible"]
        return agg[p]["fired"] / e if e else 0.0

    capture = {p: frac(p) for p in TARGET_PAIRS}
    false_action = {p: frac(p) for p in WRITE_CLEAN_PAIRS}
    capture_ok = all(capture[p] >= CAPTURE_FLOOR for p in TARGET_PAIRS)
    fa_ok = (all(false_action[p] <= FALSE_ACTION_CEILING
                 for p in WRITE_CLEAN_PAIRS) and clean_fired == 0)
    return {
        "id": cand["id"], "definition": cand["definition"],
        "second_key": cand["second_key"], "exclusion": cand["exclusion"],
        "mechanism": cand["mechanism"],
        "requires_threshold_movement": False,
        "redundancy_pr11": "acts on a subset of the merge-suspect event "
                           "class P2 merge-support-abstain keyed on"
                           + (" (K0 IS the committed PR-7 refuse arm)"
                              if cand["id"] == "K0" else ""),
        "per_cell": per_cell,
        "capture": capture, "false_action": false_action,
        "clean_fired": clean_fired,
        "capture_destruction_fraction": {
            p: frac(p) for p in TARGET_PAIRS + WRITE_CLEAN_PAIRS},
        "fired_set_identical_across_cells": len(fired_sets) == 1,
        "capture_bound_ok": capture_ok,
        "false_action_bound_ok": fa_ok,
        "passes_gate": bool(cand["second_key"] and capture_ok and fa_ok),
    }


# ---------------------------------------------------------------------------
# §2 F-blocked diagnostics (forbidden columns; reported, never eligible)
# ---------------------------------------------------------------------------
def blocked_diagnostics(panel: Path = PANEL) -> dict:
    """Per-pair payload_cos_incumbent distribution on the eligible
    (supersession) fork_events rows of the shadow arms — diagnostic only:
    these columns are state-contaminated past the first action."""
    seen: set[str] = set()
    by_pair: dict[str, list[float]] = {}
    vig_fired: dict[str, int] = {}
    for p in sorted(panel.glob("*/shadow/*stale-soft*.fork_events.csv")):
        stem = p.name.replace(".fork_events.csv", "")
        if stem in seen:
            continue
        seen.add(stem)
        pair = stem.rsplit("_", 1)[-1]
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                if row["event_class"] != "supersession":
                    continue
                by_pair.setdefault(pair, []).append(
                    float(row["payload_cos_incumbent"]))
                if (float(row["pre_sim"])
                        >= float(row["effective_vigilance"])):
                    vig_fired[pair] = vig_fired.get(pair, 0) + 1
    out = {}
    for pair, vals in sorted(by_pair.items()):
        n = len(vals)
        out[pair] = {
            "n": n,
            "payload_cos_mean": round(sum(vals) / n, 6),
            "payload_cos_min": round(min(vals), 6),
            "payload_cos_max": round(max(vals), 6),
            "sign_test_fired_fraction": round(
                sum(1 for v in vals if v <= 0.0) / n, 6),
            "vigilance_test_fired_fraction": round(
                vig_fired.get(pair, 0) / n, 6),
        }
    return out


# ---------------------------------------------------------------------------
# §4 committed citations (extracted programmatically, §7 step 4)
# ---------------------------------------------------------------------------
def committed_citations() -> dict:
    refuse = json.loads(REFUSE_TWIN.read_text())
    base = {}
    for cell in refuse["cells"].values():
        for pname, p in cell.get("per_pair", {}).items():
            b = p["baseline_by_seed"]
            base.setdefault(pname, {
                "frozen_probe_broken": sum(
                    b[s]["frozen_probe_broken"] for s in b),
                "stale_wrong": sum(b[s]["stale_wrong"] for s in b),
                "merge_suspect_events": sum(
                    b[s]["merge_suspect_events"] for s in b),
            })
    gd = refuse["cells"]["merge_path_stale"]["per_pair"]["pairD"][
        "governed_by_seed"]
    refuse_gov = {k: sum(gd[s][k] for s in gd)
                  for k in ("frozen_probe_broken", "stale_wrong",
                            "merge_suspect_events")}
    pr11 = json.loads(PR11_SCAN.read_text())
    residual = {
        pair: {s: pr11["cells"][f"{pair}/soft/s{s}"]["p2_residual"][
            "residual"] for s in SEEDS}
        for pair in TARGET_PAIRS}
    return {
        "baseline_3seed_sums": base,
        "refuse_pairD_governed_3seed_sums": refuse_gov,
        "refuse_overall_verdict": refuse.get("overall_verdict"),
        "pr11_verdict": pr11["verdict"],
        "pr11_post_pr10_soft_residual": residual,
    }


# ---------------------------------------------------------------------------
# §5 — gate / verdict
# ---------------------------------------------------------------------------
def apply_verdict(scores: list[dict]) -> str:
    admissible = [s for s in scores if s["second_key"]]
    if not admissible:
        return VERDICT_ABSENT
    if any(s["passes_gate"] for s in admissible):
        return VERDICT_CANDIDATE
    return VERDICT_FAILED


def run_scan() -> dict:
    runs = load_denominator()
    verification = verify_denominator(runs)
    candidates = build_candidates(runs)
    scores = [score_candidate(c, runs) for c in candidates]
    lemma_holds = (verification["coordinate_multiset_identical_across_runs"]
                   and verification["event_class_constant"]
                   and verification["payload_label_equal_across_runs"])
    report = {
        "design": "PR9B_SECOND_KEY_DESK_SCAN.md",
        "admissible_columns": list(ADMISSIBLE_COLUMNS),
        "bounds": {"capture_floor": CAPTURE_FLOOR,
                   "false_action_ceiling": FALSE_ACTION_CEILING},
        "denominator_verification": verification,
        "invariance_lemma_holds": lemma_holds,
        "candidates": [{k: v for k, v in s.items() if k != "per_cell"}
                       for s in scores],
        "per_cell": {s["id"]: s["per_cell"] for s in scores},
        "blocked_family_diagnostics": blocked_diagnostics(),
        "committed_citations": committed_citations(),
        "verdict": apply_verdict(scores),
    }
    return report


def main() -> int:
    report = run_scan()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT_PATH}")
    print("lemma holds:", report["invariance_lemma_holds"])
    for c in report["candidates"]:
        print(f"  {c['id']}: capture={c['capture']} "
              f"fa={c['false_action']} passes={c['passes_gate']}")
    print("VERDICT:", report["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
