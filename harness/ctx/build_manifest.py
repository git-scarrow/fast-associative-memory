#!/usr/bin/env python3
"""Build the immutable PR-13 query manifest (memo §12 R-1).

    python harness/ctx/build_manifest.py [--out PATH]

Applies the registered sample to the 13 FAM cells, retains the organic
queries and multi-turn sessions in full, and seals the result. The rule
and its salt were committed in an earlier commit than the manifest this
script writes, so the selection cannot have been tuned to the strata.

Renders nothing: the disposition rule table is applied to items to
compute strata, no context block is produced, and no consumer is touched.
Deterministic — re-running must reproduce the same digest.
"""

import argparse
import hashlib
import json
import os

from harness.ctx import cells, loaders, replay, sample
from harness.ctx.compile import load_policy

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "manifests", "query_manifest_v1.json")
POLICY_PATH = os.path.join(HERE, "policy", "disposition_policy_v1.json")
SAMPLE_PATH = os.path.join(HERE, "policy", "sample_v1.json")


def _sha_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def build(out_path=DEFAULT_OUT, verbose=True):
    policy = load_policy()
    sample_policy = sample.load_sample_policy()

    fam_cells, flat, total_probe_queries = [], [], 0
    for cell_id, stem_cell, packet_dir, role in cells.FAM_CELLS:
        bundles = loaders.fam_cell_bundles(cell_id, stem_cell, packet_dir,
                                           policy)
        picked = sample.sample_cell(cell_id, role, bundles, policy,
                                    sample_policy)
        total_probe_queries += picked["n_total"]
        fam_cells.append(picked)
        flat.extend(picked["queries"])
        if verbose:
            print(f"  {cell_id:<20} {picked['n_total']:>6} -> "
                  f"{picked['n_selected']:>4}  strata="
                  + ",".join(f"{s['disposition_class']}:{s['n_selected']}"
                             f"/{s['n_total']}" for s in picked["strata"]))

    organic = loaders.organic_bundles(policy)
    organic_ids = sorted(organic)
    flat.extend(organic_ids)

    multiturn = loaders.multiturn_bundles(policy)
    sessions = {}
    for qid, bundle in multiturn.items():
        sessions.setdefault(bundle["session_id"],
                            {"session_id": bundle["session_id"],
                             "kind": bundle["kind"], "turns": []})
    mt_records = []
    for session_id in sorted(sessions):
        turns = sorted((q for q in multiturn if multiturn[q]["session_id"]
                        == session_id), key=lambda q: multiturn[q]["turn"])
        sessions[session_id]["turns"] = turns
        mt_records.append(sessions[session_id])
        flat.extend(turns)

    manifest = {
        "manifest_id": "pr13-query-manifest",
        "version": "1.0",
        "registered_by": ("docs/PR13_GOVERNED_CONTEXT_COMPILER.md section 12, "
                          "entry R-1; sample rule committed before this file"),
        "sample_policy": {
            "sample_id": sample_policy["sample_id"],
            "version": sample_policy["version"],
            "salt": sample_policy["salt"],
            "cap_per_cell": sample_policy["cap_per_cell"],
            "sha256": _sha_file(SAMPLE_PATH),
        },
        "disposition_policy": {
            "version": policy["version"],
            "sha256": _sha_file(POLICY_PATH),
        },
        "adapters": {"fam-v1": "1.0", "shutterdeck-v1": "1.0"},
        "witness_alt_policy": "pr12-frozen:pol_f1b",
        "arm_plan": [list(a) for a in replay.ARM_PLAN],
        "budgets": list(replay.BUDGETS),
        "cells": fam_cells,
        "organic": {
            "corpus": cells.ORGANIC_CORPUS_FLAG,
            "replay_clock": cells.ORGANIC_REPLAY_CLOCK,
            "ttl_seconds": cells.ORGANIC_TTL_SECONDS,
            "ledger_sha256": _sha_file(cells.SYNTHETIC_LEDGER),
            "n_queries": len(organic_ids),
            "queries": organic_ids,
        },
        "multiturn": {
            "clocks": list(cells.MULTITURN_CLOCKS),
            "n_sessions": len(mt_records),
            "sessions": mt_records,
        },
        "queries": flat,
        "totals": {
            "fam_queries_available": total_probe_queries,
            "fam_queries_selected": sum(c["n_selected"] for c in fam_cells),
            "organic_queries": len(organic_ids),
            "multiturn_turns": sum(len(s["turns"]) for s in mt_records),
            "queries": len(flat),
            "rows": len(flat) * len(replay.ARM_PLAN),
        },
    }

    manifest = replay.seal_manifest(manifest)
    replay.verify_manifest(manifest)
    if len(set(flat)) != len(flat):
        raise ValueError("manifest contains a duplicate query identity")
    replay.expand_rows(manifest)          # turn-order and arm-plan checks

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    m = build(args.out, verbose=not args.quiet)
    t = m["totals"]
    print(f"\nsealed {args.out}")
    print(f"  digest              {m['manifest_sha256']}")
    print(f"  FAM  {t['fam_queries_selected']:>6} of {t['fam_queries_available']}")
    print(f"  organic             {t['organic_queries']}")
    print(f"  multi-turn turns    {t['multiturn_turns']}")
    print(f"  queries             {t['queries']}")
    print(f"  rows (model calls)  {t['rows']}")


if __name__ == "__main__":
    main()
