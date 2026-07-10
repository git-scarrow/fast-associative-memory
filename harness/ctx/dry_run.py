#!/usr/bin/env python3
"""PR-13 pre-flight: the whole replay pipeline on a throwaway synthetic
cell, driven by the deterministic fake consumer.

    python -m harness.ctx.dry_run

Exercises manifest sealing, row expansion, all four arms, both budgets,
multi-turn withdrawal threading, §8.4 parsing, resume after a crash, and
final reconciliation — end to end, in seconds, with no weights.

It deliberately uses a SYNTHETIC cell rather than the sealed query
manifest. Compiling a §7 cell would be a render over §7 material (§8.1),
and while the consumer pin was committed long before, that render would
foreclose the one thing still worth preserving until the scoring run is
authorized: the freedom to change the runtime if the model turns out not
to fit the scoring host. The dry run buys confidence in the harness; it
must not spend that option.

Renders nothing from §7. Constructs no real consumer. Issues no verdict.
"""

import json
import os
import shutil
import tempfile

from harness.ctx import replay
from harness.ctx.compile import load_policy
from harness.ctx.fake_consumer import FakeConsumer


def _item(native_id, content, state="agent-readable", signals=None,
          candidate_set=None):
    import hashlib
    return {
        "item_id": f"dry:{native_id}",
        "source_id": "dry-v1",
        "content": content,
        "content_kind": "source-native",
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "event_time": None, "ingest_time": None,
        "evidence": [{"adapter_id": "dry-v1", "signal": s, "value": True,
                      "evidence_ptr": f"ptr:{native_id}:{s}",
                      "tier": "core-certified" if s == "merge_suspect"
                              else "harness-heuristic"}
                     for s in (signals or ["source_identity"])],
        "relations": {"supersedes": [], "contradicts": [],
                      "candidate_set_id": candidate_set},
        "state": state, "policy_version": "1.0",
    }


def _bundle(items, text, turn=1):
    return {"items": items, "not_retrieved": [], "query_text": text,
            "turn_state": {"turn_index": turn, "prior_rendered": {}}}


def synthetic_sources(n_clean=6):
    """One query per disposition class the FAM cells actually realize,
    plus a long-content query that forces the budget loop, plus a
    three-turn withdrawal session."""
    long_text = " ".join(["lorem"] * 500) + "."
    sources = {
        "dry:assert:q0": _bundle([_item("a", "slot 3 decodes to class A.")],
                                 "Which class?"),
        "dry:caveat:q0": _bundle(
            [_item("b", "slot 4 decodes to class B.", state="stale",
                   signals=["superseded_by"])], "Which class?"),
        "dry:defer:q0": _bundle(
            [_item("c", "slot 5 decodes to class C.",
                   signals=["merge_suspect"]),
             _item("d", "slot 6 decodes to class D.")], "Which class?"),
        "dry:dual:q0": _bundle(
            [_item("e", "slot 7 decodes to class E.", signals=["oneshot_tie"],
                   candidate_set="cs1"),
             _item("f", "slot 8 decodes to class F.", signals=["oneshot_tie"],
                   candidate_set="cs1")], "Which class?"),
        "dry:budget:q0": _bundle(
            [_item(f"L{i}", long_text) for i in range(3)], "Which class?"),
    }
    for i in range(n_clean):
        sources[f"dry:clean:q{i}"] = _bundle(
            [_item(f"c{i}", f"slot {i} decodes to class A.")], "Which class?")

    served = _item("X", "the original fact.")
    superseded = _item("X", "the original fact.", state="superseded",
                       signals=["superseded_by"])
    fresh = _item("Y", "the replacement fact.")
    for turn, items in ((1, [served]), (2, [served]),
                        (3, [superseded, fresh])):
        sources[f"dry:mt:session#t{turn}"] = _bundle(
            items, "Which ingest is current?", turn=turn)
    return sources


def main():
    policy = load_policy()
    sources = synthetic_sources()
    manifest = replay.seal_manifest({
        "manifest_id": "pr13-dry-run-manifest",
        "version": "1.0",
        "note": "synthetic throwaway cell; no §7 material",
        "arm_plan": [list(a) for a in replay.ARM_PLAN],
        "budgets": list(replay.BUDGETS),
        "queries": sorted(sources),
    })

    workdir = tempfile.mkdtemp(prefix="pr13_dry_")
    out = os.path.join(workdir, "rows.jsonl")
    try:
        print(f"manifest digest {manifest['manifest_sha256']}")
        print(f"queries {len(manifest['queries'])} -> rows "
              f"{len(replay.expand_rows(manifest))}")

        # 1. crash halfway, to prove resume.
        class Crashing(FakeConsumer):
            def generate(self, prompt, max_new_tokens=256):
                if self.calls >= 17:
                    raise RuntimeError("simulated crash")
                return super().generate(prompt, max_new_tokens)

        try:
            replay.run(manifest, Crashing("mixed"), sources, out, policy)
        except RuntimeError as exc:
            print(f"crashed as designed after 17 rows: {exc}")

        # 2. resume and finish.
        summary = replay.run(manifest, FakeConsumer("mixed"), sources, out,
                             policy)
        print(f"\nreconciled: {json.dumps(summary, sort_keys=True)}")

        # 3. determinism: a clean second run must be byte-identical.
        out2 = os.path.join(workdir, "rows2.jsonl")
        replay.run(manifest, FakeConsumer("mixed"), sources, out2, policy)
        a = open(out, encoding="utf-8").read()
        b = open(out2, encoding="utf-8").read()
        print(f"byte-identical re-run: {a == b}")

        rows = [json.loads(line) for line in a.splitlines() if line.strip()]
        arms = {}
        for r in rows:
            key = (r["arm"], r["budget"])
            arms.setdefault(key, {"n": 0, "ok": 0, "prose": 0, "tok": 0})
            arms[key]["n"] += 1
            arms[key]["ok"] += r["status"] == "ok"
            arms[key]["prose"] += bool(r["extra_prose"])
            arms[key]["tok"] += r["block_tokens"]

        print("\narm            B     rows    ok  extra_prose  mean_block_tokens")
        for (arm, budget), s in sorted(arms.items(),
                                       key=lambda kv: (kv[0][0],
                                                       kv[0][1] or 0)):
            print(f"{arm:<14} {str(budget):<5} {s['n']:>4} {s['ok']:>5} "
                  f"{s['prose']:>12} {s['tok'] / s['n']:>17.1f}")

        withdrawal = next(r for r in rows
                          if r["row_id"] == "dry:mt:session#t3|governed|B800")
        print(f"\nmulti-turn t3 governed: hedged={withdrawal['hedged']} "
              f"rendered={withdrawal['rendered_item_ids']} "
              f"(withdrawn item is a notice, not a rendered item)")
        print("\nDRY RUN OK — harness exercised end to end, no §7 material, "
              "no real consumer, no verdict.")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
