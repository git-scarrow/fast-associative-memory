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
                      "verdict_by_epoch": verdicts})
    return {"pairs": pairs, "merge": merges, "max_epoch": max_epoch}


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
def decide_probe(row, cands, st, allow_stale, policy_version, cell_name):
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
            state, disp, reason = ("quarantined", "escalated",
                                   "led_quarantined_contradiction")
            evidence = (f"router(fork_events.csv): slot {led} in "
                        f"contradiction pair, unresolved @epoch {epoch}")
            items.append({"type": "unresolved_notice",
                          "text": "unresolved contradiction fork — answer "
                                  "withheld pending adjudication",
                          "certification_tier": HARNESS_TIER})
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


def run_cell(repo: Path, name: str, cfg: dict, policy: dict,
             allow_stale: bool) -> dict:
    stem = repo / cfg["run_stem"]
    cell = load_cell(stem)
    router = build_router(cell["events"], decode_at_fn(cell["decode_snaps"]))
    with open(repo / cfg["hazard_governance"]) as f:
        hazard_src = json.load(f)
    hz_router = hazard_src.get("_router", {})
    hz_none = hazard_src.get("none", {})
    hazard_tier = ("elevated" if (hz_router.get("n_merge_suspect_events", 0)
                                  or hz_none.get("stale_wrong", 0)
                                  or hz_none.get("contra_wrong", 0))
                   else "baseline")

    out_dir = repo / policy["output_root"] / name
    out_dir.mkdir(parents=True, exist_ok=True)
    state_by_epoch = {}
    counters = defaultdict(int)
    table_rows = []
    with open(out_dir / "memory_packet.jsonl", "w") as mem_f, \
            open(out_dir / "audit_packet.jsonl", "w") as aud_f:
        for row in cell["probes"]:
            epoch = int(float(row["epoch"]))
            if epoch not in state_by_epoch:
                state_by_epoch[epoch] = router_state(router, epoch)
            st = state_by_epoch[epoch]
            cands = cell["topk"][(epoch, int(row["probe_index"]))]
            d = decide_probe(row, cands, st, allow_stale,
                             policy["policy_version"], name)
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
                "stale_wrong_none": hz_none.get("stale_wrong")},
            "merge_flag_diag": {"agree": agree, "mismatch": mismatch},
            "hazard_tier": hazard_tier}


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path,
                    default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--allow-stale", action="store_true",
                    help="compile stale/superseded items WITH caveat and "
                         "logged authorization (I5 override)")
    args = ap.parse_args()

    with open(Path(__file__).parent / "harness_policy.json") as f:
        policy = json.load(f)
    results = {name: run_cell(args.repo_root, name, cfg, policy,
                              args.allow_stale)
               for name, cfg in policy["cells"].items()}
    failures = check(results, policy, args.allow_stale)
    if failures:
        print(f"\nVERDICT: {len(failures)} FAILED DESIGN ASSUMPTION(S) — "
              "see above. These are boundary-design findings, not "
              "implementation bugs to be patched silently.")
        sys.exit(1)
    print("\nVERDICT: all anchors and invariants hold on the committed "
          "artifacts.")


if __name__ == "__main__":
    main()
