#!/usr/bin/env python3
"""PR-13 sealed replay runner (memo §8).

Consumes an **immutable query manifest**: a sealed, self-hashing JSON
artifact naming the exact query identities to be replayed. The runner
never selects, filters, reorders, or extends that set; it expands each
identity into §8.3's six `(arm, B)` rows and executes them.

Structural guarantees, in the order they are enforced:

1. ``run()`` verifies the manifest seal **before** it compiles anything
   or touches a consumer. An unsealed or mutated manifest raises
   ``SealError`` with no context block rendered and no generation issued.
2. A **real** consumer (``is_real = True``) additionally requires a
   sealed scoring manifest that pins this exact query manifest and
   reports the consumer artifacts as verified. A fake consumer can never
   satisfy `is_real`, and a real one can never run unsealed — that is the
   §8.1 selection-timing kill made structural rather than procedural.
3. Governed and raw-matched at a given B are derived from **one**
   compilation, so their item multisets are equal by construction; the
   runner then re-checks the equality from the recorded rows, which
   catches a bug that a shared derivation would hide.
4. Every row's context block is asserted within its registered B (G-C3),
   the executed row set is reconciled against the expanded manifest
   (duplicates and omissions are errors, never warnings), and resumption
   refuses to append to a log written under a different manifest.

Deterministic given a consumer: no clock, no RNG, no network here.
"""

import functools
import hashlib
import json
import os

from harness.ctx.compile import compile as ctx_compile, render_raw_matched
from harness.ctx.output_contract import MAX_NEW_TOKENS, parse_consumer_output

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(HERE, "prompts")

# Memo §8.2/§8.3: governed and raw-matched at each registered B,
# raw-native once at B=1500 (exploratory, no gate reads it), none once.
BUDGETS = (800, 1500)
RAW_NATIVE_BUDGET = 1500
ARM_PLAN = (
    ("governed", 800),
    ("governed", 1500),
    ("raw_matched", 800),
    ("raw_matched", 1500),
    ("raw_native", RAW_NATIVE_BUDGET),
    ("none", None),
)
CONTEXT_ARMS = ("governed", "raw_matched", "raw_native")


class SealError(RuntimeError):
    """The manifest, the scoring manifest, or the consumer seal is not in
    a state that permits this render."""


class IntegrityError(RuntimeError):
    """The executed rows do not reconcile with the manifest."""


# --- canonical form and sealing --------------------------------------------

def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def manifest_digest(manifest):
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def seal_manifest(manifest):
    """Return a sealed copy. ``sealed`` is inside the digest preimage, so
    a manifest that claims a digest while unsealed cannot verify."""
    sealed = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    sealed["sealed"] = True
    sealed["manifest_sha256"] = manifest_digest(sealed)
    return sealed


def verify_manifest(manifest):
    if not manifest.get("sealed"):
        raise SealError("query manifest is not sealed; no render permitted")
    if "manifest_sha256" not in manifest:
        raise SealError("sealed manifest carries no digest")
    if manifest_digest(manifest) != manifest["manifest_sha256"]:
        raise SealError("query manifest digest mismatch — mutated after sealing")
    return manifest["manifest_sha256"]


def verify_scoring_manifest(scoring, query_manifest):
    if not scoring.get("sealed"):
        raise SealError("scoring manifest is not sealed")
    if manifest_digest(scoring) != scoring.get("manifest_sha256"):
        raise SealError("scoring manifest digest mismatch")
    if scoring.get("query_manifest_sha256") != query_manifest["manifest_sha256"]:
        raise SealError("scoring manifest pins a different query manifest")
    if not scoring.get("consumer", {}).get("replay_ready"):
        raise SealError("consumer artifacts are not sealed (replay_ready false)")
    if scoring.get("decoding", {}).get("max_new_tokens") != MAX_NEW_TOKENS:
        raise SealError("scoring manifest output limit diverges from §8.4")


def assert_render_permitted(consumer, manifest, scoring_manifest=None):
    """The only door. Called before any compile and before any generate."""
    verify_manifest(manifest)
    if getattr(consumer, "is_real", True):
        if scoring_manifest is None:
            raise SealError("a real consumer requires the sealed scoring "
                            "manifest (memo §8.1)")
        verify_scoring_manifest(scoring_manifest, manifest)


# --- row expansion ----------------------------------------------------------

def row_id(query_id, arm, budget):
    return f"{query_id}|{arm}|B{'na' if budget is None else budget}"


def expand_rows(manifest):
    """Manifest query identities → §8.3 rows, query-major, arm-minor.

    Query-major order is what lets a multi-turn session's turn *k* see the
    governed render of turn *k-1*; the manifest is required to list a
    session's turns in ascending order and that is checked here rather
    than assumed.

    The manifest also carries the arm plan and budget grid it was sealed
    against. If this module's registered constants have since moved, the
    sealed manifest and the code disagree about what a run *is* — that is
    arm or budget-grid motion (§10), and it stops the run.
    """
    plan = manifest.get("arm_plan")
    if plan is not None and [list(a) for a in plan] != [list(a) for a in ARM_PLAN]:
        raise IntegrityError("manifest arm plan diverges from the registered "
                             "ARM_PLAN — arm motion (§10)")
    grid = manifest.get("budgets")
    if grid is not None and list(grid) != list(BUDGETS):
        raise IntegrityError("manifest budget grid diverges from the "
                             "registered BUDGETS — budget-grid motion (§10)")

    seen_turn = {}
    rows = []
    for qid in manifest["queries"]:
        session_id, turn = _session_turn(qid)
        if session_id is not None:
            prev = seen_turn.get(session_id, 0)
            if turn != prev + 1:
                raise IntegrityError(
                    f"manifest lists {qid} out of turn order "
                    f"(expected turn {prev + 1})")
            seen_turn[session_id] = turn
        for arm, budget in ARM_PLAN:
            rows.append({"row_id": row_id(qid, arm, budget), "query_id": qid,
                         "arm": arm, "budget": budget,
                         "session_id": session_id, "turn": turn})
    return rows


def _session_turn(query_id):
    if "#t" not in query_id:
        return None, None
    session_id, turn = query_id.rsplit("#t", 1)
    return session_id, int(turn)


# --- arm rendering ----------------------------------------------------------

def render_raw_native(items, budget, count_tokens):
    """Memo §8.2: source-native order truncated to B by token count, with
    no evidence-derived selection or ordering of any kind — including the
    items governance would withhold. Exploratory; no gate reads it."""
    lines = []
    for item in items:
        candidate = lines + [item["content"]]
        if count_tokens("\n".join(candidate)) > budget:
            break
        lines = candidate
    return "\n".join(lines)


def _rendered_item_ids(audit):
    return sorted(row["item_id"] for row in audit["rows"]
                  if row["budget_decision"] in ("rendered", "summarized"))


@functools.lru_cache(maxsize=None)
def _load_prompt(name):
    """Read once. The templates are sealed artifacts; prompt motion after
    the first scoring run is a kill (§10), so caching cannot mask a change
    that would otherwise have been observed mid-run."""
    with open(os.path.join(PROMPT_DIR, name), encoding="utf-8") as fh:
        return fh.read()


def prompt_shas():
    """The sealed prompt templates, for the scoring manifest."""
    return {name: _sha(_load_prompt(name))
            for name in sorted(os.listdir(PROMPT_DIR)) if name.endswith(".txt")}


def build_prompt(arm, query_text, context_block):
    if arm == "none":
        return _load_prompt("none_arm_v1.txt").format(query=query_text)
    return _load_prompt("single_turn_v1.txt").format(
        context_block=context_block, query=query_text)


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- execution --------------------------------------------------------------

def _read_log(path, expected_ids, digest):
    """Resume: return the row_ids already executed, refusing a log that
    was written under a different manifest, carries a foreign row, or
    repeats one."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec["row_id"]
            if rec.get("manifest_sha256") != digest:
                raise IntegrityError(
                    f"{path}:{lineno}: row {rid} was written under manifest "
                    f"{rec.get('manifest_sha256')}, not {digest}")
            if rid not in expected_ids:
                raise IntegrityError(f"{path}:{lineno}: foreign row {rid} "
                                     "is not in the manifest")
            if rid in done:
                raise IntegrityError(f"{path}:{lineno}: duplicate row {rid}")
            done[rid] = rec
    return done


def _governed_for_budget(bundle, policy, count_tokens, query_id, prior_rendered):
    blocks = {}
    for budget in BUDGETS:
        turn_state = dict(bundle["turn_state"])
        turn_state["prior_rendered"] = dict(prior_rendered.get(budget, {}))
        block, audit = ctx_compile(bundle["items"], policy, budget, turn_state,
                                   count_tokens, query_id=query_id,
                                   not_retrieved=bundle["not_retrieved"])
        blocks[budget] = (block, audit)
    return blocks


def run(manifest, consumer, sources, out_path, policy,
        scoring_manifest=None, progress=None):
    """Execute the sealed replay. Returns a summary dict.

    ``sources``: ``{query_id: bundle}`` from harness.ctx.loaders (or, in
    tests, from throwaway fixtures). Every manifest query must be present.
    """
    # (1) and (2): the seal gate, before any render and any generation.
    assert_render_permitted(consumer, manifest, scoring_manifest)
    digest = manifest["manifest_sha256"]

    rows = expand_rows(manifest)
    expected_ids = [r["row_id"] for r in rows]
    if len(set(expected_ids)) != len(expected_ids):
        raise IntegrityError("manifest expands to duplicate rows")

    missing = [q for q in manifest["queries"] if q not in sources]
    if missing:
        raise IntegrityError(f"{len(missing)} manifest queries have no source, "
                             f"first: {missing[0]}")

    done = _read_log(out_path, set(expected_ids), digest)
    resumed = len(done)

    count_tokens = consumer.count_tokens
    prior = {}          # (session_id, budget) -> {item_id: turn}
    rendered_ids = {}   # (query_id, budget) -> sorted item ids, for check (3)
    for rid, rec in done.items():
        if rec.get("rendered_item_ids") is not None:
            rendered_ids[(rec["query_id"], rec["arm"], rec["budget"])] = \
                rec["rendered_item_ids"]

    executed = 0
    fh = open(out_path, "a", encoding="utf-8")
    try:
        by_query = {}
        for r in rows:
            by_query.setdefault(r["query_id"], []).append(r)

        for qid in manifest["queries"]:
            bundle = sources[qid]
            session_id, turn = _session_turn(qid)
            prior_for_query = {
                b: prior.get((session_id, b), {}) for b in BUDGETS
            } if session_id else {b: {} for b in BUDGETS}

            governed = _governed_for_budget(bundle, policy, count_tokens, qid,
                                            prior_for_query)

            # Thread prior_rendered forward for the next turn of this session.
            if session_id:
                for budget, (_blk, audit) in governed.items():
                    state = dict(prior.get((session_id, budget), {}))
                    for iid in _rendered_item_ids(audit):
                        state.setdefault(iid, turn)
                    prior[(session_id, budget)] = state

            for r in by_query[qid]:
                arm, budget = r["arm"], r["budget"]

                if arm == "governed":
                    block, audit = governed[budget]
                    ids = _rendered_item_ids(audit)
                elif arm == "raw_matched":
                    _gblock, audit = governed[budget]
                    block = render_raw_matched(bundle["items"], audit, budget,
                                               policy, count_tokens)
                    ids = _rendered_item_ids(audit)
                elif arm == "raw_native":
                    block = render_raw_native(bundle["items"], budget,
                                              count_tokens)
                    ids = None
                else:
                    block, ids = None, None

                if arm in CONTEXT_ARMS:
                    tokens = count_tokens(block)
                    if tokens > budget:
                        raise IntegrityError(
                            f"G-C3: {r['row_id']} block is {tokens} tokens "
                            f"over budget {budget}")
                    if ids is not None:
                        rendered_ids[(qid, arm, budget)] = ids
                else:
                    tokens = 0

                if r["row_id"] in done:
                    continue

                prompt = build_prompt(arm, bundle["query_text"], block)
                raw = consumer.generate(prompt, max_new_tokens=MAX_NEW_TOKENS)
                parsed = parse_consumer_output(raw)

                rec = {
                    "row_id": r["row_id"], "query_id": qid, "arm": arm,
                    "budget": budget, "session_id": session_id, "turn": turn,
                    "manifest_sha256": digest,
                    "consumer_pin_id": getattr(consumer, "pin_id", None),
                    "block_sha256": _sha(block) if block is not None else None,
                    "block_tokens": tokens,
                    "prompt_sha256": _sha(prompt),
                    "prompt_tokens": count_tokens(prompt),
                    "rendered_item_ids": ids,
                    "raw": raw,
                    "status": parsed["status"],
                    "answer": parsed.get("answer"),
                    "hedged": parsed.get("hedged"),
                    "extra_prose": parsed.get("extra_prose"),
                    "reason": parsed.get("reason"),
                }
                fh.write(canonical_json(rec) + "\n")
                fh.flush()
                executed += 1
                if progress:
                    progress(executed + resumed, len(expected_ids), r["row_id"])
    finally:
        fh.close()

    return _reconcile(out_path, expected_ids, digest, rendered_ids,
                      executed, resumed)


def _reconcile(out_path, expected_ids, digest, rendered_ids, executed, resumed):
    """(4): duplicates, omissions, and the cross-arm evidence-set equality
    that §8.2's arm-equivalence claim rests on."""
    final = _read_log(out_path, set(expected_ids), digest)
    missing = [rid for rid in expected_ids if rid not in final]
    if missing:
        raise IntegrityError(f"{len(missing)} manifest rows were never "
                             f"executed, first: {missing[0]}")
    if len(final) != len(expected_ids):
        raise IntegrityError("executed row count diverges from the manifest")

    mismatches = []
    for (qid, arm, budget), ids in rendered_ids.items():
        if arm != "governed":
            continue
        raw_ids = rendered_ids.get((qid, "raw_matched", budget))
        if raw_ids != ids:
            mismatches.append((qid, budget))
    if mismatches:
        raise IntegrityError(
            f"governed and raw-matched item multisets differ on "
            f"{len(mismatches)} (query, B) pairs, first: {mismatches[0]}")

    statuses = {}
    for rec in final.values():
        statuses[rec["status"]] = statuses.get(rec["status"], 0) + 1
    return {"rows": len(final), "executed": executed, "resumed": resumed,
            "manifest_sha256": digest, "status_counts": statuses,
            "evidence_set_pairs_checked": sum(
                1 for k in rendered_ids if k[1] == "governed")}
