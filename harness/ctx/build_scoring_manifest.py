#!/usr/bin/env python3
"""Build the immutable PR-13 scoring manifest (memo §11 deliverable 2).

    python -m harness.ctx.build_scoring_manifest --runtime runtime.json

`runtime.json` is the output of `harness.ctx.host_probe`, run on the
machine that will execute the replay. This script binds it to the code,
the pins, and the sealed query manifest, and seals the result.

The scoring manifest is what `harness/ctx/replay.py` demands before it
will run a real consumer: it must be sealed, it must pin the exact query
manifest by digest, it must report `replay_ready`, and its output limit
must equal the §8.4 contract's. It re-attests the artifact-level pin and
introduces no new degree of freedom.

Contains, per the run deliverable:
    exact code commit and a per-file content hash of harness/ctx
    query manifest digest, sample manifest digest, and the sample salt
    model, tokenizer, config, index, and chat-template hashes
    prompt template hashes
    runtime and library versions from the scoring host
    precision (bfloat16), quantization (none)
    greedy decoding
    input budgets and the output-token limit
    parser contract version and module hash

Deterministic: no timestamp, no clock. Same code + pins + host state →
same digest.
"""

import argparse
import hashlib
import json
import os
import subprocess

from harness.ctx import replay
from harness.ctx.compile import load_policy

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_OUT = os.path.join(HERE, "manifests", "scoring_manifest_v1.json")
QUERY_MANIFEST = os.path.join(HERE, "manifests", "query_manifest_v1.json")

# Everything under harness/ctx that can change a number, hashed file by
# file. Skipped: `sealed/` (gitignored weights, pinned by consumer_pin),
# `manifests/` (this file's siblings, one of which pins the other by
# digest), and `cells_data/` (the synthetic ledger, pinned by sha inside
# the query manifest). Nothing that runs is unhashed.
_SKIP_DIRS = {"sealed", "manifests", "__pycache__", "cells_data"}


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args):
    try:
        return subprocess.run(("git",) + args, cwd=REPO, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def ctx_source_hashes():
    files = {}
    for root, dirs, names in os.walk(HERE):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        for name in sorted(names):
            if name.endswith((".pyc",)):
                continue
            path = os.path.join(root, name)
            files[os.path.relpath(path, REPO)] = _sha_file(path)
    rollup = hashlib.sha256(
        replay.canonical_json(files).encode("utf-8")).hexdigest()
    return files, rollup


def build(runtime, out_path=DEFAULT_OUT):
    with open(QUERY_MANIFEST, encoding="utf-8") as fh:
        query_manifest = json.load(fh)
    replay.verify_manifest(query_manifest)

    with open(os.path.join(HERE, "policy", "consumer_pin.json"),
              encoding="utf-8") as fh:
        pin = json.load(fh)
    with open(os.path.join(HERE, "policy", "consumer_output_contract_v1.json"),
              encoding="utf-8") as fh:
        contract = json.load(fh)

    policy = load_policy()
    files, rollup = ctx_source_hashes()
    seal = runtime["seal"]

    if seal["repository_id"] != pin["artifact"]["repository_id"] or \
            seal["revision"] != pin["artifact"]["revision"]:
        raise ValueError("scoring host sealed a different consumer artifact")

    decoding = pin["runtime"]["decoding"]
    manifest = {
        "manifest_id": "pr13-scoring-manifest",
        "version": "1.0",
        "registered_by": ("docs/PR13_GOVERNED_CONTEXT_COMPILER.md sections 8 "
                          "and 11; re-attests the artifact-level pin and adds "
                          "no degree of freedom"),

        "code": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            # Tracked changes only: untracked scratch cannot alter behavior,
            # and an untracked *module* would show up in `files` below.
            "worktree_clean": _git("status", "--porcelain",
                                   "--untracked-files=no") == "",
            "note": ("the commit that produced these inputs; this manifest is "
                     "committed on top of it"),
            "ctx_source_sha256": rollup,
            "files": files,
        },

        "query_manifest_sha256": query_manifest["manifest_sha256"],
        "query_manifest_path": os.path.relpath(QUERY_MANIFEST, REPO),
        "sample": {
            "salt": query_manifest["sample_policy"]["salt"],
            "cap_per_cell": query_manifest["sample_policy"]["cap_per_cell"],
            "sample_policy_sha256": query_manifest["sample_policy"]["sha256"],
            "queries": query_manifest["totals"]["queries"],
            "rows": query_manifest["totals"]["rows"],
        },
        "disposition_policy": query_manifest["disposition_policy"],

        "consumer": {
            "pin_id": pin["pin_id"],
            "family": pin["family"],
            "repository_id": pin["artifact"]["repository_id"],
            "revision": pin["artifact"]["revision"],
            "weights_sha256": pin["artifact"]["weights_sha256"],
            "tokenizer_sha256": pin["artifact"]["tokenizer_sha256"],
            "config_sha256": pin["artifact"]["config_sha256"],
            "index_sha256": pin["artifact"]["index_sha256"],
            "chat_template_sha256": pin["artifact"]["chat_template_sha256"],
            "replay_ready": seal["replay_ready"],
            "verified_on_host": runtime["host"],
        },

        "prompts": replay.prompt_shas(),

        "runtime": {
            "host": runtime["host"],
            "platform": runtime["platform"],
            "python": runtime["python"],
            "libraries": runtime["libraries"],
            "accelerator": runtime["accelerator"],
        },

        "precision": "bfloat16",
        "quantization": "none",
        "decoding": {
            "do_sample": decoding["do_sample"],
            "temperature": decoding["temperature"],
            "max_new_tokens": decoding["max_new_tokens"],
            "stopping": decoding["stopping"],
            "enable_thinking": decoding["enable_thinking"],
            "strategy": "greedy",
        },
        "limits": {
            "input_budgets": list(replay.BUDGETS),
            "raw_native_budget": replay.RAW_NATIVE_BUDGET,
            "max_new_tokens": contract["max_new_tokens"],
        },
        "arm_plan": [list(a) for a in replay.ARM_PLAN],

        "parser": {
            "contract_id": contract["contract_id"],
            "contract_version": contract["version"],
            "contract_sha256": _sha_file(os.path.join(
                HERE, "policy", "consumer_output_contract_v1.json")),
            "schema_sha256": _sha_file(os.path.join(
                HERE, "schema", "consumer_output.schema.json")),
            "module_sha256": _sha_file(os.path.join(HERE,
                                                    "output_contract.py")),
        },
    }

    if manifest["decoding"]["do_sample"] or manifest["decoding"]["temperature"]:
        raise ValueError("pin decoding is not greedy")
    if policy["version"] != manifest["disposition_policy"]["version"]:
        raise ValueError("disposition policy version drifted from the manifest")

    manifest = replay.seal_manifest(manifest)
    replay.verify_scoring_manifest(manifest, query_manifest)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True,
                    help="output of harness.ctx.host_probe on the scoring host")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    with open(args.runtime, encoding="utf-8") as fh:
        runtime = json.load(fh)
    m = build(runtime, args.out)
    print(f"sealed {args.out}")
    print(f"  digest             {m['manifest_sha256']}")
    print(f"  query manifest     {m['query_manifest_sha256']}")
    print(f"  code commit        {m['code']['commit']} "
          f"(clean={m['code']['worktree_clean']})")
    print(f"  ctx source rollup  {m['code']['ctx_source_sha256']}")
    print(f"  consumer           {m['consumer']['repository_id']}"
          f"@{m['consumer']['revision'][:12]} "
          f"replay_ready={m['consumer']['replay_ready']}")
    print(f"  host               {m['runtime']['host']} "
          f"py{m['runtime']['python']} "
          f"torch {m['runtime']['libraries']['torch']} "
          f"transformers {m['runtime']['libraries']['transformers']}")
    print(f"  rows               {m['sample']['rows']}")


if __name__ == "__main__":
    main()
