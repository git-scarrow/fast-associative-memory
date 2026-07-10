#!/usr/bin/env python3
"""PR-13 sealed replay — EXECUTION (memo §8/§11 deliverable 2).

    <venv>/bin/python -m harness.ctx.execute_replay \
        --authorize pr13-scoring-run --out results/pr13_scoring_run/rows.jsonl

THIS IS THE FIRST EVALUATION RENDER over §7 material. Running it compiles
committed FAM / organic / multi-turn cells into context blocks and calls
the pinned Qwen3-8B consumer on them. After the first row, the §8.1
selection-timing kill and the §10 consumer/harness-motion kills are live:
no pin, decoding, manifest, compiler, sample, prompt, or scoring change
is legal any longer.

The run is sealed on both ends:
  * the committed query manifest fixes the exact 3,384 query identities;
  * the committed scoring manifest pins that query manifest by digest,
    the code, the consumer artifacts, greedy bfloat16 decoding, the
    prompts, and the device placement;
  * replay.run refuses a real consumer without both, verifies the seal
    before any compile or generation, holds the item multiset identical
    across governed and raw-matched, asserts G-C3 per row, and reconciles
    the executed rows against the manifest.

Resumable: re-running appends to the same log and continues, refusing a
log written under a different manifest. It issues no verdict — the gates
(§9) are evaluated from the row log afterwards, in a separate step.
"""

import argparse
import json
import os
import time

from harness.ctx import cells, loaders, replay
from harness.ctx.compile import load_policy
from harness.ctx.consumer_qwen3 import AUTHORIZATION_TOKEN, Qwen3Consumer

HERE = os.path.dirname(os.path.abspath(__file__))
QUERY_MANIFEST = os.path.join(HERE, "manifests", "query_manifest_v1.json")
SCORING_MANIFEST = os.path.join(HERE, "manifests", "scoring_manifest_v1.json")


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build_sources(policy, needed):
    """Construct the per-query bundles for exactly the manifest identities.

    Reads §7 cells through the registered adapters. This is a read of §7,
    not yet a render — the render happens when replay.run compiles a
    bundle into a context block. We build only what the manifest needs,
    and assert full coverage before returning so a missing source is a
    hard error, never a silently skipped row.
    """
    needed = set(needed)
    sources = {}

    for cell_id, stem_cell, packet_dir, _role in cells.FAM_CELLS:
        bundles = loaders.fam_cell_bundles(cell_id, stem_cell, packet_dir,
                                           policy)
        for qid, bundle in bundles.items():
            if qid in needed:
                sources[qid] = bundle

    for qid, bundle in loaders.organic_bundles(policy).items():
        if qid in needed:
            sources[qid] = bundle

    for qid, bundle in loaders.multiturn_bundles(policy).items():
        if qid in needed:
            sources[qid] = bundle

    missing = needed - set(sources)
    if missing:
        raise SystemExit(f"{len(missing)} manifest queries have no source; "
                         f"first: {sorted(missing)[0]}")
    return sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authorize", required=True,
                    help="must equal the registered scoring-run token")
    ap.add_argument("--out", default=os.path.join(
        "results", "pr13_scoring_run", "rows.jsonl"))
    ap.add_argument("--query-manifest", default=QUERY_MANIFEST)
    ap.add_argument("--scoring-manifest", default=SCORING_MANIFEST)
    args = ap.parse_args()

    if args.authorize != AUTHORIZATION_TOKEN:
        raise SystemExit("this is the one-shot scoring render; pass "
                         f"--authorize {AUTHORIZATION_TOKEN}")

    policy = load_policy()
    query_manifest = _load(args.query_manifest)
    scoring_manifest = _load(args.scoring_manifest)

    # Fail fast on a seal mismatch BEFORE loading 16 GB of weights.
    replay.verify_manifest(query_manifest)
    replay.verify_scoring_manifest(scoring_manifest, query_manifest)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"query manifest   {query_manifest['manifest_sha256']}")
    print(f"scoring manifest {scoring_manifest['manifest_sha256']}")
    print(f"queries {len(query_manifest['queries'])} -> rows "
          f"{query_manifest['totals']['rows']}")

    print("building §7 sources ...")
    sources = build_sources(policy, query_manifest["queries"])
    print(f"sources built: {len(sources)} queries")

    print("constructing the pinned consumer (loads weights, asserts "
          "placement) ...")
    consumer = Qwen3Consumer(authorize=args.authorize)
    print(f"consumer ready: placement {consumer.device_placement}, "
          f"pin {consumer.pin_id}")

    start = time.time()

    def progress(done, total, row_id):
        if done % 50 == 0 or done == total:
            rate = done / max(1e-9, time.time() - start)
            eta_h = (total - done) / rate / 3600 if rate else float("inf")
            print(f"[{done}/{total}] {rate*3600:.0f} rows/h, "
                  f"eta {eta_h:.1f} h  ({row_id})", flush=True)

    summary = replay.run(query_manifest, consumer, sources, args.out, policy,
                         scoring_manifest=scoring_manifest, progress=progress)
    summary["wall_seconds"] = time.time() - start
    print("\nREPLAY COMPLETE")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nNo verdict issued here; gates (§9) are evaluated from the row "
          "log in a separate step.")


if __name__ == "__main__":
    main()
