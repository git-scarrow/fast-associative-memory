"""PR-6 step 1 — empirical-hazard validation panel scaffold (analysis-only).

PR6_DESIGN.md §4 step 1. Assembles the empirical-hazard validation panel the
design memo recommends as the smallest next step after PR-5 step 1 proved
static geometry cannot certify safety. This is a SCAFFOLD, not the panel run:
it reads ONLY the committed PR-5 post-mortem artifact
(``pr5/hazard_postmortem.json`` — itself built from committed PR-3c/PR-4 runs)
and emits a panel manifest that

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
PROBE_POLICY = "mode-conditioned-trust"  # frozen hazard probe = the measurand

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


def build_panel(post: dict) -> dict:
    """Assemble the panel manifest from the loaded post-mortem dict."""
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

    # Required, but no committed single-artifact hazard label exists yet.
    cells["merge_path_stale"] = {
        "harm_shape": "write-time stale-capture",
        "harm_shape_source": "PR6_DESIGN.md §2/§5 (PR-3c)",
        "required": True,
        "status": "required_unseeded",
        "seed_artifact": (RESULTS / "pr3c").as_posix(),
        "seeds": [],
        "additional_runs_needed": (
            "an analysis-only pass over the committed PR-3c stale-arm artifacts "
            "to extract a per-cell merge-path stale-capture label, INCLUDING "
            "its measured degradation on D/E geometry; a dedicated stale-arm "
            "run is required only if that pass shows the committed artifacts do "
            "not cover a needed cell. Not produced in this scaffold."),
        "note": (
            "merge-path stale is write-time-only evidence (PR-3c) and a "
            "required benchmark a future policy must not regress, not a solved "
            "problem (PR6_DESIGN §2/§5)."),
    }

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
        "inputs_used": [POSTMORTEM.as_posix()],
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
            "merge_path_stale": "analysis-only pass over committed PR-3c stale "
                                "artifacts (a dedicated run only if a cell is "
                                "uncovered); deferred, not done here.",
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
