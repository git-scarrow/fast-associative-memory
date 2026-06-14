"""PR-6 step 1 — empirical-hazard validation panel scaffold (analysis-only).

PR6_DESIGN.md §4 step 1 (+ §7 step 2). Assembles the empirical-hazard
validation panel the design memo recommends as the smallest next step after
PR-5 step 1 proved static geometry cannot certify safety. This is a SCAFFOLD,
not the panel run: it reads the committed PR-5 post-mortem artifact
(``pr5/hazard_postmortem.json`` — itself built from committed PR-3c/PR-4 runs)
and, for the merge-path stale cell (PR-6 step 2), the committed PR-3c
soft-payload stale arms (``pr3c/per_probe_stale-soft_s{0,1,2}.*``), and emits a
panel manifest that

  * names the benchmark cell types the memo requires — clean control, D-like
    *direct* harm, B/E-like *collateral* harm, merge-path stale, and one-shot
    ambiguity;
  * SEEDS each cell an existing artifact can label with that artifact's
    MEASURED frozen-probe hazard (broken counts), never with geometry;
  * states, per cell, what additional runs (if any) are still needed;
  * records the rule for screening future cells — by measured hazard only.

Scope (PR-6 step 1 boundaries, all enforced by construction here): no
full-matrix run, no new cache run, no engine/driver change, no retrieval
change, no write-path refusal, no record-granularity ledger, no policy
tuning, and — binding from PR-5 — no static geometry used as a certification
gate. This module imports no torch and touches no cache; it only reads
committed JSON. The frozen ``mode-conditioned-trust`` probe is the MEASURING
INSTRUMENT whose damage is the label; it is never a deployment candidate
(PR6_DESIGN §5).

The per-cell harm shape (direct vs collateral) is carried forward verbatim
from PR6_DESIGN §1's mechanistic conclusion. This scaffold RECORDS the
measured hazard decomposition per cell but does not re-derive harm shape and
never gates on it.

Usage (runs on any host; reads committed JSON only, no cache/torch):
  python benchmarks/pr6_hazard_panel.py \
      --out results/issue_failure_mode_blindness/pr6/panel.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path("results/issue_failure_mode_blindness")
POSTMORTEM = RESULTS / "pr5" / "hazard_postmortem.json"
PR3C = RESULTS / "pr3c"
PROBE_POLICY = "mode-conditioned-trust"  # frozen hazard probe = the measurand

# PR-6 step 2: the committed PR-3c soft-payload ("merge-path") stale arms. These
# are the only committed artifacts whose write-time evidence labels the
# merge-path stale failure (PR3C_RESULT.md §2; PR3_DESIGN.md §5/§8). They were
# run on the pair-A class set [0, 8, 19, 33] only; the coverage audit in
# ``read_merge_path_stale_evidence`` is what records that — and is why this cell
# cannot be fully seeded here.
STALE_SOFT_ARMS = ("per_probe_stale-soft_s0", "per_probe_stale-soft_s1",
                   "per_probe_stale-soft_s2")

# The PR-5 D/E geometries the merge-path cell must eventually cover. Named here
# ONLY to enumerate the missing measurement — never used to admit or exclude a
# cell by any geometric property (PR-5 step 1 closed geometry as a predictor).
DE_CLASS_SETS = {"pairD": [10, 28, 32, 95], "pairE": [47, 56, 61, 76]}

# Cell -> the spent pairs that seed it, with the harm shape carried forward
# from PR6_DESIGN §1/§4. This assignment is the memo's mechanistic conclusion,
# NOT a geometric property of the class set (PR-5 step 1 closed geometry as a
# safety predictor). The label attached to each pair below is its MEASURED
# frozen-probe hazard, read from the committed post-mortem.
SEEDED_CELLS = {
    "clean_control":   {"harm_shape": "none",       "pairs": ["pairA", "pairC"]},
    "direct_harm":     {"harm_shape": "direct",     "pairs": ["pairD"]},
    "collateral_harm": {"harm_shape": "collateral", "pairs": ["pairB", "pairE"]},
}

HARM_SHAPE_SOURCE = ("PR6_DESIGN.md §1 (mechanistic conclusion carried "
                     "forward; not derived from geometry)")

# The five benchmark cell types the panel must contain (PR6_DESIGN §4 step 1).
REQUIRED_CELL_TYPES = (
    "clean_control", "direct_harm", "collateral_harm",
    "merge_path_stale", "one_shot_ambiguity",
)


def seed_label(post: dict, pair: str) -> dict:
    """The MEASURED hazard label for one spent pair, from committed artifacts.

    The label is the frozen probe's broken counts and their per-true-class
    concentration — the empirical measurand, with no geometric input. Both the
    direct and collateral components are recorded so a future policy can be
    held to "must not worsen either" (PR6_DESIGN §5).
    """
    gt = post["hazard_ground_truth"][pair]
    pm = post["postmortem"][pair]
    direct_total = sum(c["direct"] for c in pm["by_true_class"].values())
    broken_total = pm["broken_total"]
    return {
        "pair": pair,
        "probe_policy": post["probe_policy"],
        "broken_mean": gt["broken_mean"],
        "broken_by_seed": gt["broken_by_seed"],
        "broken_total_3seed": broken_total,
        "direct_total_3seed": direct_total,
        "collateral_total_3seed": broken_total - direct_total,
        "worst_true_class": pm["worst_class"],
        "worst_class_is_bystander": pm["worst_class_is_bystander"],
        "worst_class_share": pm["worst_class_share"],
    }


def _stale_soft_seed_label(gov: dict, summ: dict, seed: int) -> dict:
    """MEASURED merge-path stale-capture label for one committed stale-soft arm.

    The hazard population is the soft-payload (merge-path) stale rows the
    deployed vote (policy ``none``) gets wrong; the label is how the frozen
    ``mode-conditioned-trust`` probe handles them and which evidence captured
    them. Read straight from the committed governance/summary JSON — no
    geometry, no new run. Capture is via the WRITE-TIME merge-suspect trace, not
    read-time fork witness (PR3C_RESULT.md §2: every read-time fork policy
    captures 0 of the stale-wrong rows)."""
    none = gov["none"]
    mct = gov[PROBE_POLICY]
    router = gov["_router"]
    return {
        "arm": summ["arm"],
        "seed": seed,
        "class_set": summ["classes"],
        "probe_policy": PROBE_POLICY,
        "stale_wrong": none["stale_wrong"],
        "wrong_total": none["wrong_none"],
        "frozen_probe_stale_abstained": mct["stale_wrong_abstained"],
        "frozen_probe_stale_fixed": mct["stale_wrong_fixed"],
        "frozen_probe_false_abstain": mct["abstain_on_correct"],
        "frozen_probe_broken": mct["broken"],
        "merge_suspect_events": router["n_merge_suspect_events"],
        "read_time_witness_probes": mct["witness_probes"],
        "capture_via": "write-time merge-suspect trace",
    }


def read_merge_path_stale_evidence(pr3c_dir: Path = PR3C) -> dict:
    """Analysis-only pass over the committed PR-3c soft-payload stale arms.

    Returns the measured per-seed stale-capture label plus a COVERAGE audit:
    which class sets the committed merge-path arms actually measured. The audit
    is provenance ("what was measured on what"), never a geometric gate — no
    class set is admitted to or excluded from the panel by any geometric
    property. Reads committed JSON only; imports no torch, touches no cache."""
    per_seed: list[dict] = []
    sources: list[str] = []
    covered: list[list[int]] = []
    for seed, stem in enumerate(STALE_SOFT_ARMS):
        gov = json.loads((pr3c_dir / f"{stem}.governance.json").read_text())
        summ = json.loads((pr3c_dir / f"{stem}.summary.json").read_text())
        if summ.get("arm") != "stale-soft" or summ.get("payload_mode") != "soft":
            raise RuntimeError(
                f"{stem}: arm/payload {summ.get('arm')!r}/"
                f"{summ.get('payload_mode')!r} is not the soft merge path")
        per_seed.append(_stale_soft_seed_label(gov, summ, seed))
        sources.append((pr3c_dir / f"{stem}.governance.json").as_posix())
        if summ["classes"] not in covered:
            covered.append(summ["classes"])
    return {"per_seed": per_seed, "covered_class_sets": covered,
            "source_artifacts": sources, "n_seeds": len(per_seed)}


def build_merge_path_stale_cell(pr3c_dir: Path = PR3C) -> dict:
    """PR-6 step 2 — seed merge_path_stale from committed PR-3c stale arms if the
    evidence covers the cell, else leave it required-unseeded with a precise
    missing-evidence note.

    Measured result: the committed soft-payload stale arms label merge-path
    stale capture on the pair-A class set ONLY (3 seeds). Pairs D/E did not exist
    as geometries in PR-3c (their class sets were constructed later, PR-4/PR-5),
    so no committed artifact measures merge-path stale on D/E. The cell therefore
    stays required-unseeded — its required label (step-1: "including its D/E
    degradation") is structurally not coverable from committed artifacts — but
    now carries the measured pair-A partial evidence and names the single gated
    run that would complete it. No new cache run is taken here."""
    ev = read_merge_path_stale_evidence(pr3c_dir)
    de_uncovered = sorted(name for name, cs in DE_CLASS_SETS.items()
                          if cs not in ev["covered_class_sets"])
    return {
        "harm_shape": "write-time stale-capture",
        "harm_shape_source": "PR6_DESIGN.md §2/§5 (PR-3c)",
        "required": True,
        "status": "required_unseeded",
        "evidence_status": "partial_pairA_only",
        "seed_artifact": pr3c_dir.as_posix(),
        "seeds": [],
        "partial_evidence": {
            "probe_policy": PROBE_POLICY,
            "covered_class_sets": ev["covered_class_sets"],
            "n_seeds": ev["n_seeds"],
            "per_seed": ev["per_seed"],
            "source_artifacts": ev["source_artifacts"],
            "summary": (
                "merge-path (soft-payload) stale capture is MEASURED on the "
                "pair-A class set only (3 seeds): the deployed vote is wrong on "
                "~374 stale rows/seed; the frozen mode-conditioned-trust probe "
                "abstains on all of them with broken=0, captured solely by the "
                "write-time merge-suspect trace (192 events/seed), not read-time "
                "fork witness (PR3C_RESULT.md §2)."),
        },
        "missing_evidence": {
            "needed": "measured merge-path stale capture on D/E geometry",
            "uncovered_geometries": de_uncovered,
            "why_absent": (
                "the PR-3c soft-payload (merge-path) stale arm was run only on "
                "the pair-A class set [0, 8, 19, 33]; pairs D and E did not exist "
                "as geometries in PR-3c (their class sets were constructed later, "
                "PR-4/PR-5), so no committed artifact measures merge-path stale on "
                "D/E. Pair-B merge-path stale was also never run."),
            "required_run": (
                "a dedicated soft-payload stale arm on the pairD {10,28,32,95} "
                "and pairE {47,56,61,76} class sets, >=3 seeds, scored by the "
                "frozen mode-conditioned-trust probe — OUT OF PR-6 step-2 scope "
                "(no new cache runs)."),
        },
        "additional_runs_needed": (
            "partially resolved: the committed PR-3c stale-soft arms now seed a "
            "MEASURED pair-A merge-path stale-capture label (recorded in "
            "partial_evidence). The cell stays unseeded because its required "
            "label includes D/E degradation, and no committed artifact measures "
            "merge-path stale on D/E geometry; completing it needs a dedicated "
            "stale-arm run on the D/E class sets (see missing_evidence), out of "
            "step-2 scope."),
        "note": (
            "merge-path stale is write-time-only evidence (PR-3c) and a required "
            "benchmark a future policy must not regress, not a solved problem "
            "(PR6_DESIGN §2/§5)."),
    }


def build_panel(post: dict, pr3c_dir: Path = PR3C) -> dict:
    """Assemble the panel manifest from the loaded post-mortem dict.

    PR-6 step 2: the merge_path_stale cell is now built by
    ``build_merge_path_stale_cell`` from the committed PR-3c stale arms; its
    measured source artifacts are added to ``inputs_used``."""
    if post.get("probe_policy") != PROBE_POLICY:
        raise RuntimeError(
            f"post-mortem probe_policy {post.get('probe_policy')!r} != "
            f"frozen hazard probe {PROBE_POLICY!r}; refusing to seed labels")

    cells: dict[str, dict] = {}
    for cell, spec in SEEDED_CELLS.items():
        cells[cell] = {
            "harm_shape": spec["harm_shape"],
            "harm_shape_source": HARM_SHAPE_SOURCE,
            "required": True,
            "status": "seeded",
            "seed_artifact": POSTMORTEM.as_posix(),
            "seeds": [seed_label(post, p) for p in spec["pairs"]],
            "additional_runs_needed":
                "none — measured labels present in the committed artifact",
        }

    # Required, write-time-only evidence. PR-6 step 2 reads the committed PR-3c
    # stale arms: merge-path stale is now MEASURED on pair-A geometry (recorded
    # as partial_evidence) but D/E geometry is structurally absent from the
    # committed artifacts, so the cell stays required-unseeded.
    cells["merge_path_stale"] = build_merge_path_stale_cell(pr3c_dir)

    # Required as a named cell, but observe-only by conclusion — carries no
    # hazard label and must never be scored pass/fail at read time.
    cells["one_shot_ambiguity"] = {
        "harm_shape": "ambiguous-evidence (observe-only)",
        "harm_shape_source": "PR6_DESIGN.md §5",
        "required": True,
        "status": "observe_only",
        "seed_artifact": None,
        "seeds": [],
        "additional_runs_needed": (
            "none at read time — one-shot ambiguity is certified insufficient "
            "evidence and stays observe-only (PR6_DESIGN §5). It becomes a "
            "pass/fail cell only via write-API provenance metadata "
            "(PR3_DESIGN §10), an engine change reserved to path 3 (PR-7), out "
            "of PR-6 scope."),
        "note": (
            "recorded as a panel cell so it is not silently dropped; it carries "
            "no hazard label and must not be scored as pass/fail."),
    }

    status_summary = {s: sorted(c for c in cells if cells[c]["status"] == s)
                      for s in sorted({cells[c]["status"] for c in cells})}

    return {
        "design": "PR6_DESIGN.md §4 step 1 — empirical-hazard validation panel "
                  "(scaffold)",
        "scaffold": True,
        "probe_policy": PROBE_POLICY,
        "probe_note": (
            "the frozen hazard probe is the measuring instrument whose damage "
            "is the label; it is never a deployment candidate (PR6_DESIGN §5)."),
        "inputs_used": (
            [POSTMORTEM.as_posix()]
            + cells["merge_path_stale"]["partial_evidence"]["source_artifacts"]),
        "source_cache_path": post.get("cache_path"),
        "geometry_used_as_gate": False,
        "new_cache_runs": 0,
        "engine_or_retrieval_change": False,
        "required_cell_types": list(REQUIRED_CELL_TYPES),
        "cells": cells,
        "cell_status_summary": status_summary,
        "screening_procedure": {
            "admit_cell_iff": (
                "the candidate class set's frozen mode-conditioned-trust hazard "
                "is MEASURED (mixed arm, >=3 seeds) and decomposed by true "
                "class; the measured broken counts ARE the cell's label."),
            "geometry_forbidden": (
                "a class set may NOT be admitted to or excluded from the panel "
                "by any static geometric property (attribution ratio, centroid "
                "confusion rate, fork contrast, etc.) — PR-5 step 1 closed "
                "static geometry as a safety predictor."),
            "scope_statement": (
                "the panel certifies nothing beyond its enumerated cells and "
                "must refuse to generalize to unseen geometries "
                "(PR6_DESIGN §3 path 1 risk)."),
            "both_harm_shapes_required": (
                "any policy validated on the panel must exercise BOTH direct "
                "(D-like) and collateral (B/E-like) cells; a policy that fixes "
                "one while worsening the other does not pass (PR6_DESIGN §5)."),
        },
        "additional_runs_summary": {
            "seeded_cells": "none — clean_control, direct_harm, collateral_harm "
                            "are fully labeled from committed artifacts.",
            "merge_path_stale": "PR-6 step 2 read the committed PR-3c stale-soft "
                                "arms: pair-A merge-path stale capture is now "
                                "MEASURED (partial_evidence); the cell stays "
                                "unseeded pending a D/E-geometry stale-arm run "
                                "(out of scope, no new cache runs).",
            "one_shot_ambiguity": "no read-time run; needs write-API provenance "
                                  "metadata (path 3 / PR-7), out of scope.",
            "optional_panel_widening": "additional empirically-screened cells per "
                                       "screening_procedure are optional, not "
                                       "required for the seeded panel.",
        },
        "conclusions_enforced": [
            "static geometry cannot certify safety (PR-5 step 1)",
            "both D-like direct and B/E-like collateral harm remain required "
            "benchmarks",
            "one-shot ambiguity stays observe-only",
            "merge-path stale stays a required benchmark",
            "slot-granularity trust deprecation stays closed; the frozen probe "
            "is only the measurand, never a baseline or deployment candidate",
            "deployed retrieval remains unchanged",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--postmortem", default=str(POSTMORTEM),
        help="committed PR-5 post-mortem JSON (default: %(default)s)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    post = json.loads(Path(args.postmortem).read_text())
    panel = build_panel(post)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(panel, indent=2, sort_keys=True) + "\n")

    summary = panel["cell_status_summary"]
    for status in sorted(summary):
        print(f"{status:18} {', '.join(summary[status])}")
    print(f"geometry_used_as_gate={panel['geometry_used_as_gate']} "
          f"new_cache_runs={panel['new_cache_runs']} "
          f"engine_or_retrieval_change={panel['engine_or_retrieval_change']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
