#!/usr/bin/env python3
"""PR-12 minimal harness prototype — offline replay adapter.

Implements PR12_HARNESS_BOUNDARY_DESIGN.md §5 exactly: replays committed
PR-10 governed run artifacts through the harness boundary contract and
emits, per cell, a prompt-visible ``memory_packet.jsonl``, an
``audit_packet.jsonl``, and a flat ``decision_table.csv``. Analysis-only,
stdlib-only, darwin, no torch. No FAM-core file is imported or modified:
the harness consumes committed *artifacts* (boundary invariant I3), so the
frozen scorer's label-free router is mirrored here rather than imported,
each rule pinned to its source line in benchmarks/analyze_fork_governance.py.
Fidelity of the mirror is proven by the committed anchors, not assumed:
any mismatch is reported as a FAILED DESIGN ASSUMPTION (never silently
patched), per the §6 acceptance criteria.

Deliberate deviation from the memo's evidence-pointer wording: the memo
cites ``per_slot.is_merge_candidate`` as mechanism (a) evidence, but the
frozen scorer never loads the per_slot ``is_*`` diagnostic flags (its
policy-visible allowlist excludes them — they are driver diagnostics, not
label-free observables). States here derive from the label-free router
only; per_slot-flag agreement is reported as a diagnostic.

Reproducibility gate (run before merge):

    python3 harness/harness_boundary_sim.py --check

regenerates every emitted artifact from the committed inputs into a temp
dir and byte-compares against the committed ``pr12/`` outputs, exiting
nonzero on any drift (no normalization — byte identity is the contract).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# Frozen constants — analyze_fork_governance.py:119-120 (read-only mirror).
CONFLICT_COS = 0.5
MERGE_SUSPECT_COS = 0.9

CORE_TIER = "core-certified"
HARNESS_TIER = "harness-heuristic"
ABSTENTION_NOTICE = ("memory abstained (merge-suspect) — core-certified "
                     "PR-10 merge-abstain contract")


# ---------------------------------------------------------------------------
# Artifact loading (committed schemas only)
# ---------------------------------------------------------------------------
def load_cell(stem: Path) -> dict:
    with open(stem.with_suffix(".csv"), newline="") as f:
        probes = list(csv.DictReader(f))
    topk = defaultdict(list)
    with gzip.open(stem.with_suffix(".topk.csv.gz"), "rt", newline="") as f:
        for r in csv.DictReader(f):
            topk[(int(r["epoch"]), int(r["probe_index"]))].append({
                "rank": int(r["rank"]), "slot": int(r["slot"]),
                "sim": float(r["sim"]), "surviving": r["surviving"] == "1",
                "weight": r["weight"], "decode": int(r["decode"])})
    for cands in topk.values():
        cands.sort(key=lambda c: c["rank"])
    decode_snaps = defaultdict(dict)   # slot -> {epoch: decode}
    merge_flag = {}                    # (epoch, slot) -> is_merge_candidate
    with open(stem.with_suffix(".per_slot.csv"), newline="") as f:
        for r in csv.DictReader(f):
            e, s = int(r["epoch"]), int(r["slot"])
            decode_snaps[s][e] = int(r["decode"])
            merge_flag[(e, s)] = int(r["is_merge_candidate"])
    events = []
    with open(stem.with_suffix(".fork_events.csv"), newline="") as f:
        for r in csv.DictReader(f):
            events.append({
                "epoch": int(float(r["epoch"])),
                "record_seq": int(float(r["record_seq"])),
                "outcome": r["outcome"],
                "payload_cos_incumbent": float(r["payload_cos_incumbent"]),
                "incumbent_slot": int(float(r["incumbent_slot"])),
                "owner_slot": int(float(r["owner_slot"]))})
    events.sort(key=lambda e: e["record_seq"])
    with open(stem.with_suffix(".summary.json")) as f:
        summary = json.load(f)
    return {"probes": probes, "topk": topk, "decode_snaps": decode_snaps,
            "merge_flag": merge_flag, "events": events, "summary": summary}


# ---------------------------------------------------------------------------
# Label-free write-time router — mirror of analyze_fork_governance.py
# ---------------------------------------------------------------------------
def decode_at_fn(decode_snaps):
    # Mirror of _decode_at_fn (lines 820-833): forward-filled lookup.
    def decode_at(epoch, slot):
        snaps = decode_snaps.get(slot)
        if not snaps:
            return None
        best = max((e for e in snaps if e <= epoch), default=None)
        return None if best is None else snaps[best]
    return decode_at


def build_router(events, decode_at):
    # Mirror of build_writetime_router (lines 292-345).
    conflicts = [e for e in events
                 if e["outcome"] == "forked" and e["incumbent_slot"] >= 0
                 and math.isfinite(e["payload_cos_incumbent"])
                 and e["payload_cos_incumbent"] <= CONFLICT_COS]
    merges = [(e["epoch"], e["owner_slot"]) for e in events
              if e["outcome"] == "absorbed"
              and math.isfinite(e["payload_cos_incumbent"])
              and e["payload_cos_incumbent"] < MERGE_SUSPECT_COS]
    max_epoch = max((e["epoch"] for e in events), default=-1)
    pairs = []
    for c in conflicts:
        I, O = c["incumbent_slot"], c["owner_slot"]
        old_side, new_side = decode_at(c["epoch"], I), decode_at(c["epoch"], O)
        if old_side is None or new_side is None:
            continue
        later = [e for e in events
                 if e["record_seq"] > c["record_seq"]
                 and e["incumbent_slot"] in (I, O)
                 and e["outcome"] in ("absorbed", "forked")]
        verdicts = {}
        for E in range(c["epoch"], max_epoch + 1):
            old_n = new_n = 0
            for e in later:
                if e["epoch"] > E:
                    continue
                side = decode_at(
                    e["epoch"],
                    e["owner_slot"] if e["outcome"] == "forked"
                    else e["incumbent_slot"])
                if side == old_side:
                    old_n += 1
                elif side == new_side:
                    new_n += 1
            verdicts[E] = ("contradiction" if old_n > 0
                           else "supersession" if new_n > 0 else "ambiguous")
        pairs.append({"I": I, "O": O, "epoch": c["epoch"],
                      "old_side": old_side, "new_side": new_side,
                      "verdict_by_epoch": verdicts})
    return {"pairs": pairs, "merge": merges, "max_epoch": max_epoch}


def pair_counterparts(router):
    # Mirror of pair_counterparts (lines 371-378): static slot -> {counterpart
    # slots}; membership never changes across epochs, only verdicts do.
    cp = defaultdict(set)
    for p in router["pairs"]:
        cp[p["I"]].add(p["O"])
        cp[p["O"]].add(p["I"])
    return dict(cp)


def pair_alt_classes(router):
    """Slot -> decode classes of the OTHER side of every pair the slot is a
    member of (PR-12.1 C2 dual-present payload; router evidence only)."""
    alt = defaultdict(set)
    for p in router["pairs"]:
        alt[p["I"]].add(p["new_side"])
        alt[p["O"]].add(p["old_side"])
    return dict(alt)


def router_state(router, epoch):
    # Mirror of router_state (lines 348-368), trust=False path only.
    quarantine, deprecate, ambiguous = set(), set(), set()
    for p in router["pairs"]:
        v = p["verdict_by_epoch"].get(epoch)
        if v is None:
            continue
        if v == "contradiction":
            quarantine |= {p["I"], p["O"]}
        elif v == "supersession":
            deprecate.add(p["I"])
        else:
            ambiguous |= {p["I"], p["O"]}
    merge = {s for (e, s) in router["merge"] if e <= epoch}
    deprecate -= quarantine
    ambiguous -= (quarantine | deprecate)
    return {"quarantine": quarantine, "deprecate": deprecate,
            "ambiguous": ambiguous, "merge": merge}


# ---------------------------------------------------------------------------
# Harness decision for one probe row
# ---------------------------------------------------------------------------
def decide_probe(row, cands, st, allow_stale, policy_version, cell_name,
                 shape="prototype", counterpart=None, alt_classes=None):
    epoch = int(float(row["epoch"]))
    probe_index = int(row["probe_index"])
    qid = f"{cell_name}:e{epoch}:p{probe_index}"
    surv = [c for c in cands if c["surviving"]]
    led = (max(surv, key=lambda c: float(c["weight"]))["slot"]
           if surv else None)
    merge_support = any(c["slot"] in st["merge"] for c in surv)
    abstained = row["served_outcome"] == "abstain"

    def audit(item_id, state, disposition, reason, evidence, tier,
              authorization=None):
        return {"query_id": qid, "item_id": item_id, "state": state,
                "disposition": disposition, "reason_code": reason,
                "evidence_ptr": evidence, "certification_tier": tier,
                "policy_version": policy_version,
                "authorization": authorization}

    items, decisions = [], []
    if abstained:
        items.append({"type": "abstention_notice", "text": ABSTENTION_NOTICE,
                      "certification_tier": CORE_TIER,
                      "provenance": {"contract": "PR-10 merge-abstain",
                                     "abstain_reason": row["abstain_reason"]}})
        decisions.append(audit(
            f"served_answer@{qid}", "audit-only", "withheld",
            "core_abstention_passthrough",
            f"per_probe.csv: served_outcome=abstain, "
            f"abstain_reason={row['abstain_reason']}", CORE_TIER))
    else:
        answer = {"type": "memory_item",
                  "content": {"label": int(float(row["vote_pred_label"]))},
                  "provenance": {"top1_slot": led,
                                 "support_slots": sorted(
                                     c["slot"] for c in surv)},
                  "certification_tier": HARNESS_TIER, "caveats": []}
        if led in st["quarantine"]:
            # PR-12.1 §3 disposition shapes for quarantine-led served
            # answers ONLY; reason_code is shared (no new codes) so V1
            # (audit basis) is disposition-invariant. shape="prototype"
            # is byte-identical to PR-12.
            reason = "led_quarantined_contradiction"
            surv_slots = {c["slot"] for c in surv}
            witness_live = bool((counterpart or {}).get(led, set())
                                & surv_slots)
            escalate = (shape == "prototype"
                        or (shape == "C3" and witness_live))
            if escalate:
                state, disp = "quarantined", "escalated"
                evidence = (f"router(fork_events.csv): slot {led} in "
                            f"contradiction pair, unresolved @epoch {epoch}")
                if shape == "C3":
                    evidence += ("; witness-gated: counterpart co-surviving "
                                 "in this probe's support")
                items.append({"type": "unresolved_notice",
                              "text": "unresolved contradiction fork — "
                                      "answer withheld pending adjudication",
                              "certification_tier": HARNESS_TIER})
            else:
                state, disp = "agent-readable", "shown_with_caveat"
                evidence = (f"router(fork_events.csv): slot {led} in "
                            f"contradiction pair, unresolved @epoch {epoch}")
                if shape == "C3":
                    evidence += ("; witness-gated: no counterpart in this "
                                 "probe's support (degraded to caveat)")
                answer["caveats"].append(
                    "contradiction: led slot is party to an unresolved "
                    "contradiction fork (review pending)")
                if shape == "C2":
                    answer["alternatives"] = sorted(
                        (alt_classes or {}).get(led, set()))
                items.append(answer)
        elif led in st["ambiguous"]:
            state, disp, reason = ("human-review", "escalated",
                                   "led_pending_ambiguous")
            evidence = (f"router(fork_events.csv): slot {led} in "
                        f"ambiguous (pending) pair @epoch {epoch}")
            items.append({"type": "unresolved_notice",
                          "text": "unresolved fork — two candidates, "
                                  "unresolved tie",
                          "certification_tier": HARNESS_TIER})
        elif led in st["deprecate"]:
            state, reason = "superseded", "led_superseded_supersession"
            evidence = (f"router(fork_events.csv): slot {led} superseded "
                        f"(supersession verdict) @epoch {epoch}")
            if allow_stale:
                disp = "shown_with_caveat"
                answer["caveats"].append("superseded content (allow_stale)")
                items.append(answer)
            else:
                disp = "withheld"
        elif merge_support:
            state, disp, reason = ("agent-readable", "shown_with_caveat",
                                   "merge_support_member")
            msl = sorted(s for s in st["merge"]
                         if s in {c["slot"] for c in surv})
            evidence = (f"router(fork_events.csv): absorbed event(s) with "
                        f"payload_cos_incumbent<{MERGE_SUSPECT_COS} -> "
                        f"merge-suspect slot(s) {msl} in surviving support "
                        f"(topk.csv.gz) @epoch {epoch}")
            answer["caveats"].append(
                "stale-suspect: merge-suspect slot in surviving support")
            items.append(answer)
        else:
            state, disp, reason = "agent-readable", "shown", "no_adverse_flag"
            evidence = (f"topk.csv.gz + router: no adverse flag on support "
                        f"@epoch {epoch}")
            items.append(answer)
        decisions.append(audit(
            f"served_answer@{qid}", state, disp, reason, evidence,
            HARNESS_TIER,
            authorization="allow_stale" if (allow_stale
                                            and state == "superseded"
                                            and disp != "withheld")
            else None))

    for c in cands:
        if not c["surviving"]:
            decisions.append(audit(
                f"slot{c['slot']}@{qid}", "audit-only", "withheld",
                "not_surviving_engine",
                f"topk.csv.gz: surviving=0 at rank {c['rank']}",
                HARNESS_TIER))
    retrieval_scope = {"k": len(cands), "n_surviving": len(surv),
                       "note": "items absent from topk were not retrieved; "
                               "absence is evidenced by k, not enumerable"}
    return {"query_id": qid, "items": items, "decisions": decisions,
            "retrieval_scope": retrieval_scope, "led_slot": led,
            "merge_support": merge_support, "abstained": abstained}


# ---------------------------------------------------------------------------
# Invariant + anchor checks (§6). Failures are design-assumption reports.
# ---------------------------------------------------------------------------
def scrub_certified(obj, permitted_notice_types=("abstention_notice",)):
    """Remove the fields where 'certified' is permitted; return the rest."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "certification_tier":
                continue
            if (k == "text" and obj.get("type") in permitted_notice_types):
                continue
            out[k] = scrub_certified(v, permitted_notice_types)
        return out
    if isinstance(obj, list):
        return [scrub_certified(v, permitted_notice_types) for v in obj]
    return obj


def load_hazard(repo: Path, spec: str) -> dict:
    """Load a committed hazard-governance source. `path#key1#key2`
    navigates into an aggregate table, each #-segment one literal JSON
    key (keys may contain slashes, e.g. 'pairD/contra/s0' in
    pr4_geometry_table.json — PR-12.1 §4: contra-pairD has no per-run
    sibling). Missing file or key is a stop condition, never worked
    around."""
    path, *frags = spec.split("#")
    with open(repo / path) as f:
        src = json.load(f)
    for key in frags:
        if key not in src:
            raise FileNotFoundError(
                f"hazard key '{key}' not found in {path}")
        src = src[key]
    return src


def build_review_queue(router, led_rows, quarantine_led_by_slot):
    """PR-12.1 §2 V3 payload: one audit-only record per final-epoch
    contradiction pair — pair identity, per-side affected row counts,
    stable exemplar query IDs (explicit no_led_rows when a side never
    leads a served row). Derived from router + led mapping only, so it
    is disposition-shape-invariant by construction; V3 checks that."""
    queue = []
    for p in router["pairs"]:
        if p["verdict_by_epoch"].get(router["max_epoch"]) != "contradiction":
            continue
        sides = []
        for slot, side_cls, role in ((p["I"], p["old_side"], "old_side"),
                                     (p["O"], p["new_side"], "new_side")):
            rows = led_rows.get(slot, [])
            side = {"slot": slot, "role": role, "decode_class": side_cls,
                    "led_row_count": len(rows)}
            if rows:
                side["exemplars"] = {"first": rows[0], "last": rows[-1]}
            else:
                side["no_led_rows"] = True
            sides.append(side)
        queue.append({
            "record_type": "contradiction_pair_review",
            "pair": {"incumbent_slot": p["I"], "owner_slot": p["O"],
                     "onset_epoch": p["epoch"]},
            "quarantine_led_rows_total":
                quarantine_led_by_slot.get(p["I"], 0)
                + quarantine_led_by_slot.get(p["O"], 0),
            "sides": sides,
            "certification_tier": HARNESS_TIER})
    return queue


def run_cell(repo: Path, name: str, cfg: dict, policy: dict,
             allow_stale: bool, out_root: Path | None = None,
             shape: str = "prototype", policy_version: str | None = None,
             emit_review_queue: bool = False) -> dict:
    stem = repo / cfg["run_stem"]
    cell = load_cell(stem)
    router = build_router(cell["events"], decode_at_fn(cell["decode_snaps"]))
    counterpart = pair_counterparts(router)
    alt_classes = pair_alt_classes(router)
    hazard_src = load_hazard(repo, cfg["hazard_governance"])
    hz_router = hazard_src.get("_router", {})
    hz_none = hazard_src.get("none", {})
    hazard_tier = ("elevated" if (hz_router.get("n_merge_suspect_events", 0)
                                  or hz_none.get("stale_wrong", 0)
                                  or hz_none.get("contra_wrong", 0))
                   else "baseline")

    out_dir = (out_root if out_root is not None
               else repo / policy["output_root"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    pv = policy_version or policy["policy_version"]
    state_by_epoch = {}
    counters = defaultdict(int)
    table_rows = []
    led_rows = defaultdict(list)          # slot -> [query_id, ...] (served)
    quarantine_led_by_slot = defaultdict(int)
    exposure = {flag: {"caveated": 0, "unmarked": 0}
                for flag in ("stale_lenient", "contradictory_lenient",
                             "merge_support", "no_flag")}
    with open(out_dir / "memory_packet.jsonl", "w") as mem_f, \
            open(out_dir / "audit_packet.jsonl", "w") as aud_f:
        for row in cell["probes"]:
            epoch = int(float(row["epoch"]))
            if epoch not in state_by_epoch:
                state_by_epoch[epoch] = router_state(router, epoch)
            st = state_by_epoch[epoch]
            cands = cell["topk"][(epoch, int(row["probe_index"]))]
            d = decide_probe(row, cands, st, allow_stale, pv, name,
                             shape=shape, counterpart=counterpart,
                             alt_classes=alt_classes)
            answer_dec = d["decisions"][0]
            mem_f.write(json.dumps({"query_id": d["query_id"],
                                    "items": d["items"]}) + "\n")
            aud_f.write(json.dumps({
                "query_id": d["query_id"],
                "retrieval_scope": d["retrieval_scope"],
                "hazard_tier": {"tier": hazard_tier,
                                "certification_tier": HARNESS_TIER,
                                "evidence_ptr": cfg["hazard_governance"]},
                "decisions": d["decisions"]}) + "\n")
            table_rows.append({
                "query_id": d["query_id"], "epoch": epoch,
                "probe_index": int(row["probe_index"]),
                "led_slot": d["led_slot"],
                "merge_support_flag": int(d["merge_support"]),
                "abstained_core": int(d["abstained"]),
                "state": answer_dec["state"],
                "disposition": answer_dec["disposition"],
                "reason_code": answer_dec["reason_code"],
                "evidence_ptr": answer_dec["evidence_ptr"],
                "certification_tier": answer_dec["certification_tier"]})

            truth = int(float(row["true_label"]))
            deployed = int(float(row["vote_pred_label"]))
            stale_wrong = (row["stale_lenient"] == "1"
                           and deployed != truth)
            counters["n"] += 1
            counters["abstained"] += int(d["abstained"])
            counters["stale_wrong"] += int(stale_wrong)
            counters["stale_wrong_abstained"] += int(stale_wrong
                                                     and d["abstained"])
            counters["stale_wrong_flagged_served"] += int(
                stale_wrong and not d["abstained"] and d["merge_support"])
            counters["stale_wrong_unflagged_served"] += int(
                stale_wrong and not d["abstained"] and not d["merge_support"])
            counters["caveat_on_correct"] += int(
                not d["abstained"] and d["merge_support"]
                and deployed == truth)
            counters[f"state:{answer_dec['state']}"] += 1
            counters[f"disposition:{answer_dec['disposition']}"] += 1
            # G3-exactness self-check: certified abstain <=> led in merge.
            counters["abstain_led_merge_mismatch"] += int(
                d["abstained"] != (d["led_slot"] in st["merge"]))

            # ---- PR-12.1 scan tracking (counters only; never emitted on
            # ---- the prototype byte path, so pr12/ identity is untouched)
            correct = deployed == truth
            counters["wrong_none"] += int(not correct)
            if not d["abstained"]:
                led_rows[d["led_slot"]].append(d["query_id"])
                q_led = d["led_slot"] in st["quarantine"]
                if q_led:
                    quarantine_led_by_slot[d["led_slot"]] += 1
                    counters["quarantine_led_served"] += 1
                    # V1: audit basis retained, never no_adverse_flag
                    counters["v1_violations"] += int(
                        answer_dec["reason_code"]
                        != "led_quarantined_contradiction")
                    # V2: compiled item carries a contradiction marker
                    marker = any(
                        it.get("type") == "unresolved_notice"
                        or any("contradiction" in cv
                               for cv in it.get("caveats", []))
                        for it in d["items"])
                    counters["v2_violations"] += int(not marker)
                if answer_dec["disposition"] == "escalated":
                    counters["escalated_on_correct"] += int(correct)
                elif answer_dec["disposition"] == "withheld":
                    counters["withheld_on_correct"] += int(correct)
                elif not correct:  # wrong row entering the prompt (§5 E)
                    mark = ("caveated" if answer_dec["disposition"]
                            == "shown_with_caveat" else "unmarked")
                    flags = [f for f, on in (
                        ("stale_lenient", row["stale_lenient"] == "1"),
                        ("contradictory_lenient",
                         row["contradictory_lenient"] == "1"),
                        ("merge_support", d["merge_support"])) if on]
                    for f in flags or ["no_flag"]:
                        exposure[f][mark] += 1
                    counters["wrong_in_prompt"] += 1
                    counters["wrong_in_prompt_caveated"] += int(
                        mark == "caveated")

        review_queue = build_review_queue(router, led_rows,
                                          quarantine_led_by_slot)
        if emit_review_queue:
            for rec in review_queue:
                aud_f.write(json.dumps(rec) + "\n")

    with open(out_dir / "decision_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        w.writeheader()
        w.writerows(table_rows)

    # per_slot.is_merge_candidate agreement diagnostic (memo wording check)
    agree = mismatch = 0
    for (e, s), flag in cell["merge_flag"].items():
        router_says = s in {sl for (ee, sl) in router["merge"] if ee <= e}
        if bool(flag) == router_says:
            agree += 1
        else:
            mismatch += 1

    return {"counters": dict(counters), "out_dir": out_dir,
            "router_counts": {"n_conflict_pairs": len(router["pairs"]),
                              "n_merge_suspect_events": len(router["merge"])},
            "hazard_source_counts": {
                "n_conflict_pairs": hz_router.get("n_conflict_pairs"),
                "n_merge_suspect_events": hz_router.get(
                    "n_merge_suspect_events"),
                "n_rows": hz_none.get("n"),
                "stale_wrong_none": hz_none.get("stale_wrong"),
                "contra_wrong_none": hz_none.get("contra_wrong")},
            "merge_flag_diag": {"agree": agree, "mismatch": mismatch},
            "hazard_tier": hazard_tier,
            "review_queue": review_queue,
            "exposure": exposure}


def check(results: dict, policy: dict, allow_stale: bool) -> list[str]:
    failures = []

    def expect(cond, label, detail):
        if cond:
            print(f"  PASS  {label}")
        else:
            failures.append(label)
            print(f"  FAILED DESIGN ASSUMPTION  {label}: {detail}")

    for name, res in results.items():
        cfg = policy["cells"][name]
        a, c = cfg["anchors"], res["counters"]
        print(f"\n[{name}] ({cfg['role']}) -> {res['out_dir']}")
        expect(c["abstain_led_merge_mismatch"] == 0,
               "G3 exactness: certified abstain set == merge-led set",
               f"{c['abstain_led_merge_mismatch']} mismatched rows — the "
               "label-free router mirror does not reproduce the certified "
               "abstention trigger")
        expect(res["router_counts"]["n_merge_suspect_events"]
               == a.get("n_merge_suspect_events",
                        res["router_counts"]["n_merge_suspect_events"])
               and res["router_counts"]["n_merge_suspect_events"]
               == (res["hazard_source_counts"]["n_merge_suspect_events"]
                   or 0),
               "router rebuild matches committed governance.json counts",
               f"rebuilt {res['router_counts']} vs committed "
               f"{res['hazard_source_counts']}")
        expect(c["abstained"] == a["certified_abstentions"],
               f"certified abstentions == {a['certified_abstentions']}",
               f"got {c['abstained']}")
        if cfg["role"] == "primary":
            expect(c["stale_wrong"] == a["stale_wrong_total"],
                   f"stale-wrong rows == {a['stale_wrong_total']} "
                   "(committed none-policy count)",
                   f"got {c['stale_wrong']}")
            expect(c["stale_wrong_abstained"] == a["stale_wrong_abstained"],
                   f"mechanism (a): certified-abstained stale-wrong == "
                   f"{a['stale_wrong_abstained']}",
                   f"got {c['stale_wrong_abstained']}")
            expect(c["stale_wrong_flagged_served"]
                   == a["stale_wrong_residual_flagged"],
                   f"mechanism (a): residual stale-wrong flagged == "
                   f"{a['stale_wrong_residual_flagged']}",
                   f"got {c['stale_wrong_flagged_served']}")
            expect(c["stale_wrong_unflagged_served"] == 0,
                   "mechanism (a): zero stale-wrong rows escape "
                   "(375 = 292 + 83, P2 151/151 mechanism)",
                   f"{c['stale_wrong_unflagged_served']} stale-wrong served "
                   "rows carry no merge-support flag")
        if cfg["role"] == "control":
            adverse = sum(v for k, v in c.items()
                          if k.startswith("state:")
                          and k.split(":", 1)[1] not in
                          ("agent-readable",)) \
                + c.get("disposition:shown_with_caveat", 0) \
                + c.get("disposition:withheld", 0) \
                + c.get("disposition:escalated", 0)
            expect(adverse == 0 and c["abstained"] == 0,
                   "control emits zero adverse states",
                   f"adverse={adverse}, abstained={c['abstained']}")
        if not allow_stale:
            expect(c.get("state:stale", 0) == 0
                   and (c.get("state:superseded", 0) == 0
                        or c.get("disposition:withheld", 0) > 0),
                   "I5: no stale/superseded item compiled without "
                   "allow_stale",
                   "a stale/superseded item reached the memory packet")

        # 'certified' string containment (I1 / §6.7)
        bad = 0
        for fname in ("memory_packet.jsonl", "audit_packet.jsonl"):
            with open(res["out_dir"] / fname) as f:
                for line in f:
                    rec = scrub_certified(json.loads(line))
                    if "certified" in json.dumps(rec).lower():
                        bad += 1
        with open(res["out_dir"] / "decision_table.csv", newline="") as f:
            for r in csv.DictReader(f):
                r.pop("certification_tier", None)
                if "certified" in json.dumps(r).lower():
                    bad += 1
        expect(bad == 0,
               "I1: 'certified' appears only in permitted fields",
               f"{bad} records leak the word outside certification_tier / "
               "abstention notice")

        # I7 completeness: every decision carries every audit field
        incomplete = 0
        required = ("query_id", "item_id", "state", "disposition",
                    "reason_code", "evidence_ptr", "certification_tier",
                    "policy_version")
        with open(res["out_dir"] / "audit_packet.jsonl") as f:
            for line in f:
                for dec in json.loads(line)["decisions"]:
                    if any(not dec.get(k) for k in required):
                        incomplete += 1
        expect(incomplete == 0, "I7: every decision fully audited",
               f"{incomplete} incomplete decisions")
        print(f"  diag  per_slot.is_merge_candidate vs router: "
              f"{res['merge_flag_diag']} "
              f"{'(memo evidence-ptr wording holds)' if res['merge_flag_diag']['mismatch'] == 0 else '(memo evidence-ptr wording FAILED — report, see stdout)'}")
        print(f"  info  hazard_tier={res['hazard_tier']}; state counts: "
              + ", ".join(f"{k.split(':', 1)[1]}={v}"
                          for k, v in sorted(res["counters"].items())
                          if k.startswith("state:")))
    return failures


# ---------------------------------------------------------------------------
# PR-12.1 §8 — disposition re-shaping scan (gates scored with no discretion)
# ---------------------------------------------------------------------------
def packet_invariants(out_dir: Path) -> dict:
    """V4 file-level invariants for one emitted (shape, cell): I1
    certified-string containment and I7 audit completeness. Handles both
    per-probe records and review-queue records."""
    leaks = incomplete = 0
    required = ("query_id", "item_id", "state", "disposition",
                "reason_code", "evidence_ptr", "certification_tier",
                "policy_version")
    for fname in ("memory_packet.jsonl", "audit_packet.jsonl"):
        with open(out_dir / fname) as f:
            for line in f:
                rec = json.loads(line)
                if "certified" in json.dumps(
                        scrub_certified(rec)).lower():
                    leaks += 1
                for dec in rec.get("decisions", []):
                    if any(not dec.get(k) for k in required):
                        incomplete += 1
    with open(out_dir / "decision_table.csv", newline="") as f:
        for r in csv.DictReader(f):
            r.pop("certification_tier", None)
            if "certified" in json.dumps(r).lower():
                leaks += 1
    return {"certified_leaks": leaks, "incomplete_audits": incomplete}


def base_bytecheck(repo: Path, policy: dict, allow_stale: bool) -> bool:
    """PR-12 reproducibility gate, callable: regenerate the committed
    pr12/ cells into a temp dir and byte-compare. True iff all
    byte-identical."""
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="pr12_bytecheck_"))
    ok = True
    try:
        for name, cfg in policy["cells"].items():
            run_cell(repo, name, cfg, policy, allow_stale, out_root=tmp)
            for fn in ("memory_packet.jsonl", "audit_packet.jsonl",
                       "decision_table.csv"):
                fresh = (tmp / name / fn).read_bytes()
                committed = (repo / policy["output_root"] / name
                             / fn).read_bytes()
                ok &= fresh == committed
    finally:
        shutil.rmtree(tmp)
    return ok


def run_scan(repo: Path, policy: dict, allow_stale: bool) -> int:
    scan = policy["scan"]
    shapes = scan["shapes"]
    cells = scan["cells"]
    ceiling = scan["suppression_ceiling"]
    pv = scan["policy_version"]
    scan_root = repo / scan["output_root"]
    report = {"design_memo": scan["design_memo"], "policy_version": pv,
              "suppression_ceiling": ceiling, "shapes": shapes,
              "cells": list(cells), "gates": {}, "exposure": {},
              "counters": {}, "blocked": []}

    print("G-R prologue: PR-12 base byte-gate")
    before = base_bytecheck(repo, policy, allow_stale)
    report["base_bytecheck_before"] = before
    print(f"  {'PASS' if before else 'FAILED DESIGN ASSUMPTION'}  "
          f"pr12/ regeneration byte-identical (before scan)")

    results = {}
    for shape in shapes:
        for name, cfg in cells.items():
            try:
                results[(shape, name)] = run_cell(
                    repo, name, cfg, policy, allow_stale,
                    out_root=scan_root / shape, shape=shape,
                    policy_version=pv, emit_review_queue=True)
            except FileNotFoundError as e:
                report["blocked"].append(
                    {"shape": shape, "cell": name, "missing": str(e)})
    if report["blocked"]:
        report["verdict"] = "reshape-blocked"
        _write_report(scan_root, report)
        print(f"\nVERDICT: reshape-blocked — missing committed inputs: "
              f"{report['blocked']}")
        return 1

    # ---- instrumentation gates (shape-independent -> blocked on failure)
    blocked = not before
    for name, cfg in cells.items():
        res = results[("prototype", name)]
        c, rc, hz = res["counters"], res["router_counts"], \
            res["hazard_source_counts"]
        checks = {
            "router_pairs_match_committed":
                hz["n_conflict_pairs"] is None
                or rc["n_conflict_pairs"] == hz["n_conflict_pairs"],
            "router_merges_match_committed":
                hz["n_merge_suspect_events"] is None
                or rc["n_merge_suspect_events"]
                == hz["n_merge_suspect_events"],
            "row_count_matches_committed":
                hz["n_rows"] is None or c["n"] == hz["n_rows"],
            "abstain_set_exact": c["abstain_led_merge_mismatch"] == 0,
        }
        report["gates"].setdefault("instrumentation", {})[name] = checks
        for label, okc in checks.items():
            print(f"  {'PASS' if okc else 'FAILED DESIGN ASSUMPTION'}  "
                  f"[{name}] {label}")
            blocked |= not okc
    if blocked:
        report["verdict"] = "reshape-blocked"
        _write_report(scan_root, report)
        print("\nVERDICT: reshape-blocked — instrumentation contradiction "
              "(anchor break or base byte-gate failure); no candidate may "
              "be judged from a blocked run.")
        return 1

    # ---- per-candidate gates
    candidates_pass = []
    for shape in shapes:
        report["gates"][shape] = {}
        report["exposure"][shape] = {}
        report["counters"][shape] = {}
        shape_ok = True
        print(f"\n[{shape}]")
        for name, cfg in cells.items():
            res = results[(shape, name)]
            proto = results[("prototype", name)]
            c = res["counters"]
            denom = c["n"] - c["wrong_none"]
            supp = c.get("escalated_on_correct", 0) \
                + c.get("withheld_on_correct", 0)
            rate = supp / denom if denom else 0.0
            inv = packet_invariants(res["out_dir"])
            gates = {
                "G-S": {"suppressive_on_correct": supp,
                        "correct_traffic": denom,
                        "rate": round(rate, 6),
                        "pass": rate <= ceiling},
                "V1": {"quarantine_led_served":
                       c.get("quarantine_led_served", 0),
                       "violations": c.get("v1_violations", 0),
                       "pass": c.get("v1_violations", 0) == 0},
                "V2": {"violations": c.get("v2_violations", 0),
                       "pass": c.get("v2_violations", 0) == 0},
                "V3": {"pairs": len(res["review_queue"]),
                       "pass": res["review_queue"]
                       == proto["review_queue"]},
                "V4": {**inv,
                       "pass": inv["certified_leaks"] == 0
                       and inv["incomplete_audits"] == 0},
                "G-R_mech_d_superseded_identical": {
                    "pass": all(
                        c.get(k, 0) == proto["counters"].get(k, 0)
                        for k in ("state:human-review", "state:superseded",
                                  "abstained",
                                  "stale_wrong_flagged_served"))},
            }
            if cfg["role"] == "continuity":
                a = cfg["anchors"]
                gates["G-R_anchors"] = {"pass": (
                    c["abstained"] == a["certified_abstentions"]
                    and c["stale_wrong"] == a["stale_wrong_total"]
                    and c["stale_wrong_abstained"]
                    == a["stale_wrong_abstained"]
                    and c["stale_wrong_flagged_served"]
                    == a["stale_wrong_residual_flagged"]
                    and c["stale_wrong_unflagged_served"] == 0)}
            if cfg["role"] == "control":
                adverse = sum(v for k, v in c.items()
                              if k.startswith("state:")
                              and k.split(":", 1)[1] != "agent-readable") \
                    + c.get("disposition:shown_with_caveat", 0) \
                    + c.get("disposition:withheld", 0) \
                    + c.get("disposition:escalated", 0)
                gates["G-R_control_zero_adverse"] = {
                    "adverse": adverse, "pass": adverse == 0}
            report["gates"][shape][name] = gates
            report["exposure"][shape][name] = res["exposure"]
            report["counters"][shape][name] = res["counters"]
            cell_ok = all(g["pass"] for g in gates.values())
            shape_ok &= cell_ok
            print(f"  [{name}] " + "  ".join(
                f"{g}={'PASS' if v['pass'] else 'FAIL'}"
                + (f"({v['rate']:.3f})" if g == "G-S" else "")
                for g, v in gates.items()))
        if shape != "prototype" and shape_ok:
            candidates_pass.append(shape)

    print("\nG-R epilogue: PR-12 base byte-gate")
    after = base_bytecheck(repo, policy, allow_stale)
    report["base_bytecheck_after"] = after
    print(f"  {'PASS' if after else 'FAILED DESIGN ASSUMPTION'}  "
          f"pr12/ regeneration byte-identical (after scan)")
    if not after:
        report["verdict"] = "reshape-blocked"
        _write_report(scan_root, report)
        return 1

    report["candidates_pass"] = candidates_pass
    report["verdict"] = (f"reshape-evidence-GO({','.join(candidates_pass)})"
                         if candidates_pass else "reshape-negative")
    _write_report(scan_root, report)
    print(f"\nVERDICT: {report['verdict']}")
    print("Scope (PR12_1_DISPOSITION_RESHAPE.md §5): reshape evidence at "
          "the offline simulator layer only — not runtime prompt safety, "
          "not policy promotion.")
    return 0


def _write_report(scan_root: Path, report: dict):
    scan_root.mkdir(parents=True, exist_ok=True)
    with open(scan_root / "reshape_scan.json", "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path,
                    default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--allow-stale", action="store_true",
                    help="compile stale/superseded items WITH caveat and "
                         "logged authorization (I5 override)")
    ap.add_argument("--check", action="store_true",
                    help="reproducibility gate: regenerate every artifact "
                         "into a temp dir and byte-compare against the "
                         "committed pr12/ outputs; exit 1 on any drift. "
                         "No normalization is applied — byte identity is "
                         "the contract (stdlib json with ensure_ascii, "
                         "insertion-ordered dicts, sets sorted at "
                         "emission), so drift is a broken determinism "
                         "assumption, to be reported, never normalized "
                         "away silently.")
    ap.add_argument("--shape", choices=["prototype", "C1", "C2", "C3"],
                    help="PR-12.1 §3 disposition shape for quarantine-led "
                         "served answers; emits one shape over the scan "
                         "panel under pr12_1/<shape>/ without gate "
                         "scoring")
    ap.add_argument("--scan", action="store_true",
                    help="PR-12.1 §8 full scan: all shapes × all panel "
                         "cells, §5 gates scored with no discretion, "
                         "reshape_scan.json + verdict; includes the PR-12 "
                         "base byte-gate before and after")
    args = ap.parse_args()

    with open(Path(__file__).parent / "harness_policy.json") as f:
        policy = json.load(f)
    if args.scan:
        sys.exit(run_scan(args.repo_root, policy, args.allow_stale))
    if args.shape:
        scan = policy["scan"]
        for name, cfg in scan["cells"].items():
            res = run_cell(args.repo_root, name, cfg, policy,
                           args.allow_stale,
                           out_root=args.repo_root / scan["output_root"]
                           / args.shape,
                           shape=args.shape,
                           policy_version=scan["policy_version"],
                           emit_review_queue=True)
            print(f"[{args.shape}/{name}] -> {res['out_dir']} | "
                  + ", ".join(f"{k.split(':', 1)[1]}={v}"
                              for k, v in sorted(res["counters"].items())
                              if k.startswith("disposition:")))
        return
    out_root = None
    if args.check:
        import shutil
        import tempfile
        out_root = Path(tempfile.mkdtemp(prefix="pr12_bytecheck_"))
    results = {name: run_cell(args.repo_root, name, cfg, policy,
                              args.allow_stale, out_root=out_root)
               for name, cfg in policy["cells"].items()}
    failures = check(results, policy, args.allow_stale)
    if args.check:
        print("\nbyte-identity check vs committed artifacts:")
        committed_root = args.repo_root / policy["output_root"]
        for name in policy["cells"]:
            for fn in ("memory_packet.jsonl", "audit_packet.jsonl",
                       "decision_table.csv"):
                fresh = (out_root / name / fn).read_bytes()
                committed = (committed_root / name / fn).read_bytes()
                if fresh == committed:
                    print(f"  PASS  {name}/{fn} byte-identical")
                else:
                    failures.append(f"byte drift: {name}/{fn}")
                    print(f"  FAILED DESIGN ASSUMPTION  {name}/{fn}: "
                          f"regenerated output differs from committed "
                          f"bytes ({len(fresh)} vs {len(committed)} bytes)")
        shutil.rmtree(out_root)
    if failures:
        print(f"\nVERDICT: {len(failures)} FAILED DESIGN ASSUMPTION(S) — "
              "see above. These are boundary-design findings, not "
              "implementation bugs to be patched silently.")
        sys.exit(1)
    print("\nVERDICT: all anchors and invariants hold on the committed "
          "artifacts.")


if __name__ == "__main__":
    main()
