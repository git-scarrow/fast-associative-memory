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

# Frozen constants — analyze_fork_governance.py:117-120 (read-only mirror).
CONFLICT_COS = 0.5
MERGE_SUSPECT_COS = 0.9
WITNESS_SIM_WINDOW = 0.05  # frozen since PR-2; analyze_fork_governance.py:117

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
def ambiguous_pairs_at(router, epoch, st):
    """slot -> [{'onset': pair onset epoch, 'alt': counterpart decode
    class}] for pairs whose verdict at `epoch` is ambiguous, restricted
    to slots the router-state precedence left in the ambiguous set
    (PR-12.2 §3; router evidence only, no lookahead)."""
    out = defaultdict(list)
    for p in router["pairs"]:
        if p["verdict_by_epoch"].get(epoch) != "ambiguous":
            continue
        for slot, alt in ((p["I"], p["new_side"]), (p["O"], p["old_side"])):
            if slot in st["ambiguous"]:
                out[slot].append({"onset": p["epoch"], "alt": alt})
    return dict(out)


# ---------------------------------------------------------------------------
# PR-12.3 §3 — witness-window candidate-set construction (frozen-scorer mirror)
# ---------------------------------------------------------------------------
def _surv_class_mass(cands):
    """Surviving vote mass per decode class — the §3 width-bound ordering key.
    Mirror of analyze_fork_governance.py:282-286 (_class_mass), restricted to
    surviving candidates as §3 specifies ('descending surviving vote mass').
    Label-free: reads only topk weight/decode, never the registry."""
    mass = defaultdict(float)
    for c in cands:
        if c["surviving"]:
            mass[c["decode"]] += float(c["weight"])
    return mass


def witness_window_classes(cands):
    """Decode classes of the probe's fork witness window. EXACT reimplement of
    the frozen scorer's fork_witness (analyze_fork_governance.py:246-254):
    surviving candidates within WITNESS_SIM_WINDOW raw cosine of the surviving
    top-1 (top-1 by ``sim``), iff they span >= 2 decode classes; else empty.

    This is the probe's own read-time tie set (PR-12.3 §1); it reads only the
    probe's own topk rows — no lookahead (§4). PR-12.3 §6 names the fidelity
    of this mirror as the key assumption: a divergence from the frozen
    scorer's fork_witness is instrumentation, not evidence."""
    surv = [c for c in cands if c["surviving"]]
    if not surv:
        return set()
    top = max(c["sim"] for c in surv)
    win = [c for c in surv if c["sim"] >= top - WITNESS_SIM_WINDOW]
    classes = {c["decode"] for c in win}
    return set(classes) if len(classes) >= 2 else set()


def build_witness_present(cands, deployed_label, pairs_here):
    """PR-12.3 §3 candidate set + width bound for W1/W2 (parameter-free).

    Candidate classes = witness-window decode classes UNION the deployed
    answer; if the witness window is empty, fall back to the PR-12.2 D2 set
    (the leading ambiguous pair's counterpart classes), recorded
    ``fallback='pair_counterpart'``. Width bound (§3): at most 3 decode
    classes total — deployed always present, plus <= 2 alternatives selected
    by DESCENDING surviving vote mass per class, ties broken by ASCENDING
    decode-class index (fixed, label-free, not a new observable). Truncated
    alternatives are counted, never silently dropped.

    Empty-set scoring (§3): if the fallback set is nonetheless empty for some
    row (instrumentation fault), the presented set is {deployed} alone (width
    1) and the row is NEVER excluded from scoring — a small set only makes
    containment harder to achieve, never easier."""
    wclasses = witness_window_classes(cands)
    if wclasses:
        candidate_classes = set(wclasses) | {deployed_label}
        fallback = None
    else:
        candidate_classes = {p["alt"] for p in pairs_here} | {deployed_label}
        fallback = "pair_counterpart"
    mass = _surv_class_mass(cands)
    ordered_alts = sorted(candidate_classes - {deployed_label},
                          key=lambda cls: (-mass[cls], cls))
    kept = ordered_alts[:2]                    # deployed + <= 2 alternatives
    presented = [deployed_label, *kept]        # emission order: deployed first
    return {"presented": presented,
            "presented_set": sorted(set(presented)),
            "ordered_alts": kept,
            "fallback": fallback,
            "truncated": len(ordered_alts) - len(kept),
            "width_used": len(presented),
            "candidate_class_count": len(candidate_classes)}


def decide_probe(row, cands, st, allow_stale, policy_version, cell_name,
                 shape="prototype", counterpart=None, alt_classes=None,
                 amb_pairs=None):
    epoch = int(float(row["epoch"]))
    probe_index = int(row["probe_index"])
    qid = f"{cell_name}:e{epoch}:p{probe_index}"
    surv = [c for c in cands if c["surviving"]]
    led = (max(surv, key=lambda c: float(c["weight"]))["slot"]
           if surv else None)
    merge_support = any(c["slot"] in st["merge"] for c in surv)
    abstained = row["served_outcome"] == "abstain"
    dual_out = None
    dual_meta = None

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
            # PR-12.2 shapes D1/D2 and PR-12.3 shapes W1/W2 modify pending-led
            # rows ONLY (§2/§4); quarantine-led escalation keeps the prototype
            # disposition under them — C1/C2 behavior is NOT revived (a
            # quarantine-led disposition change would be a §7.5 kill).
            escalate = (shape in ("prototype", "D1", "D2", "W1", "W2")
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
            # PR-12.2 §3 shapes for pending-led served answers ONLY;
            # reason_code shared (V1' is disposition-invariant). Every
            # shape other than D1/D2 keeps the PR-12 prototype
            # escalation byte-identically. D2's never-resolving
            # classifier uses onset epoch vs current epoch only —
            # no future-epoch verdict is consulted (§4).
            reason = "led_pending_ambiguous"
            pairs_here = (amb_pairs or {}).get(led, [])
            fresh = any(p["onset"] == epoch for p in pairs_here)
            # PR-12.3 W1 dual-presents every pending-led row; W2 uses PR-12.2
            # D2's never-resolving selectivity (dual iff ambiguous strictly
            # past onset; fresh rows keep the prototype escalation). Neither
            # revives C1/C2/C3 nor alters quarantine/superseded handling (§4).
            dual = bool(pairs_here) and (
                shape == "D1" or (shape == "D2" and not fresh)
                or shape == "W1" or (shape == "W2" and not fresh))
            if not dual:
                state, disp = "human-review", "escalated"
                evidence = (f"router(fork_events.csv): slot {led} in "
                            f"ambiguous (pending) pair @epoch {epoch}")
                if shape == "D2":
                    evidence += ("; age-gated: pair inside its onset "
                                 "epoch (genuinely undecided)")
                elif shape == "W2":
                    evidence += ("; age-gated: fresh ambiguity inside onset "
                                 "epoch — prototype escalation retained")
                items.append({"type": "unresolved_notice",
                              "text": "unresolved fork — two candidates, "
                                      "unresolved tie",
                              "certification_tier": HARNESS_TIER})
            else:
                state, disp = "human-review", "shown_with_caveat"
                deployed_label = int(float(row["vote_pred_label"]))
                if shape in ("W1", "W2"):
                    # PR-12.3 §3 witness-window candidate set + width bound.
                    w = build_witness_present(cands, deployed_label,
                                              pairs_here)
                    dual_meta = {"fallback": w["fallback"],
                                 "truncated": w["truncated"],
                                 "width_used": w["width_used"],
                                 "empty_candidate_set":
                                     len(w["ordered_alts"]) == 0,
                                 "source": ("pair_counterpart"
                                            if w["fallback"] else "witness")}
                    src_basis = ("fork counterpart (router pair fallback)"
                                 if w["fallback"] else
                                 "witness co-resident (fork_witness)")
                    # Per-row fallback/truncation markers live inside the
                    # decision evidence (§8), audit-side.
                    evidence = (f"router(fork_events.csv): slot {led} in "
                                f"ambiguous (pending) pair @epoch {epoch}; "
                                f"witness-window dual-present "
                                f"[source={dual_meta['source']}, "
                                f"width={w['width_used']}, "
                                f"truncated={w['truncated']}]")
                    if shape == "W2":
                        evidence += ("; age-gated: ambiguous strictly past "
                                     "onset (onsets "
                                     f"{sorted({p['onset'] for p in pairs_here})})")
                    items.append({
                        "type": "unresolved_tie",
                        "text": "unresolved tie — two candidates, neither "
                                "asserted",
                        "candidates": [
                            {"decode_class": deployed_label, "slot": led,
                             "basis": "deployed vote"},
                            *[{"decode_class": a, "basis": src_basis}
                              for a in w["ordered_alts"]]],
                        "certification_tier": HARNESS_TIER})
                    answer = None  # never compiled as a single answer
                    dual_out = w["presented_set"]
                else:
                    # PR-12.2 D1/D2 (byte-frozen; not a PR-12.3 shape).
                    alts = sorted({p["alt"] for p in pairs_here})
                    dual_classes = sorted({deployed_label, *alts})
                    evidence = (f"router(fork_events.csv): slot {led} in "
                                f"ambiguous (pending) pair @epoch {epoch}; "
                                f"never-resolving classifier: ambiguous past "
                                f"onset (onsets "
                                f"{sorted({p['onset'] for p in pairs_here})})"
                                if shape == "D2" else
                                f"router(fork_events.csv): slot {led} in "
                                f"ambiguous (pending) pair @epoch {epoch}; "
                                f"dual-present-all")
                    items.append({
                        "type": "unresolved_tie",
                        "text": "unresolved tie — two candidates, neither "
                                "asserted",
                        "candidates": [
                            {"decode_class": deployed_label, "slot": led,
                             "basis": "deployed vote"},
                            *[{"decode_class": a,
                               "basis": "fork counterpart (router pair)"}
                              for a in alts]],
                        "certification_tier": HARNESS_TIER})
                    answer = None  # never compiled as a single answer
                    dual_out = dual_classes
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
            "merge_support": merge_support, "abstained": abstained,
            "pending_led": (not abstained and led in st["ambiguous"]),
            "dual_classes": dual_out, "dual_meta": dual_meta}


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


def build_ambiguous_queue(router, led_rows):
    """PR-12.2 §6 V3' payload: one audit-only record per final-epoch
    ambiguous pair — identity, the never-resolving classification
    (recorded for every shape), per-side counts and exemplars, explicit
    no_led_rows. Disposition-shape-invariant by construction."""
    queue = []
    for p in router["pairs"]:
        if p["verdict_by_epoch"].get(router["max_epoch"]) != "ambiguous":
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
            "record_type": "ambiguous_pair_review",
            "pair": {"incumbent_slot": p["I"], "owner_slot": p["O"],
                     "onset_epoch": p["epoch"]},
            "never_resolving": p["epoch"] < router["max_epoch"],
            "sides": sides,
            "certification_tier": HARNESS_TIER})
    return queue


def run_cell(repo: Path, name: str, cfg: dict, policy: dict,
             allow_stale: bool, out_root: Path | None = None,
             shape: str = "prototype", policy_version: str | None = None,
             emit_review_queue: bool = False,
             emit_ambiguous_queue: bool = False) -> dict:
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
    # PR-12.3 §5 counters live in a SEPARATE namespace so the shared run_cell
    # never perturbs the committed PR-12.1/12.2 scan reports (which dump
    # `counters` wholesale) — byte-identity of pr12_1/pr12_2 is the contract.
    pr123 = defaultdict(int)
    table_rows = []
    led_rows = defaultdict(list)          # slot -> [query_id, ...] (served)
    quarantine_led_by_slot = defaultdict(int)
    exposure = {flag: {"caveated": 0, "unmarked": 0, "dual_presented": 0}
                for flag in ("stale_lenient", "contradictory_lenient",
                             "merge_support", "no_flag")}
    amb_by_epoch = {}
    with open(out_dir / "memory_packet.jsonl", "w") as mem_f, \
            open(out_dir / "audit_packet.jsonl", "w") as aud_f:
        for row in cell["probes"]:
            epoch = int(float(row["epoch"]))
            if epoch not in state_by_epoch:
                state_by_epoch[epoch] = router_state(router, epoch)
                amb_by_epoch[epoch] = ambiguous_pairs_at(
                    router, epoch, state_by_epoch[epoch])
            st = state_by_epoch[epoch]
            cands = cell["topk"][(epoch, int(row["probe_index"]))]
            d = decide_probe(row, cands, st, allow_stale, pv, name,
                             shape=shape, counterpart=counterpart,
                             alt_classes=alt_classes,
                             amb_pairs=amb_by_epoch[epoch])
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
                if d["pending_led"]:
                    counters["pending_led_served"] += 1
                    # V1'/V2' (PR-12.2 §6): audit basis + prompt marker
                    counters["v1p_violations"] += int(
                        answer_dec["reason_code"]
                        != "led_pending_ambiguous")
                    marker = any(it.get("type") in ("unresolved_notice",
                                                    "unresolved_tie")
                                 for it in d["items"])
                    counters["v2p_violations"] += int(not marker)
                    if d["dual_classes"] is not None:
                        counters["pending_dual"] += 1
                        counters["pending_dual_on_correct"] += int(correct)
                        meta = d.get("dual_meta")
                        if meta is not None:   # PR-12.3 W1/W2 witness shapes
                            pr123["pending_dual_witness"] += int(
                                meta["fallback"] is None)
                            pr123["pending_dual_fallback_paircp"] += int(
                                meta["fallback"] == "pair_counterpart")
                            pr123["pending_dual_truncated_rows"] += int(
                                meta["truncated"] > 0)
                            pr123["pending_dual_truncated_classes"] += \
                                meta["truncated"]
                            pr123["pending_dual_full_width"] += int(
                                meta["width_used"] == 3)
                            pr123["pending_dual_width_gt3"] += int(
                                meta["width_used"] > 3)
                            pr123["pending_dual_empty_candidate_set"] += \
                                int(meta["empty_candidate_set"])
                        if not correct:
                            counters["pending_dual_wrong"] += 1
                            counters["pending_dual_truth_contained"] += int(
                                truth in d["dual_classes"])
                            # width_used per dual-presented WRONG row feeds the
                            # §5 chance-baseline (mean width / n_decode_classes)
                            if meta is not None:
                                pr123["pending_dual_wrong_width_sum"] += \
                                    meta["width_used"]
                    else:
                        counters["pending_escalated"] += 1
                        # PR-12.3 G-A: candidate-attributable suppression is
                        # the pending-led escalate/withhold on CORRECT traffic
                        # — the only suppression the W1/W2 candidate controls
                        # (frozen-baseline quarantine/superseded is separate).
                        pr123["pending_led_suppressed_on_correct"] += int(
                            correct)
                # PR-12.3 §5: frozen-baseline suppression (quarantine-led +
                # superseded; design-frozen per PR-12.2 §12) is reported on
                # every cell and gated NOWHERE — not the candidate's to
                # control. Recorded distinctly from the G-A numerator above.
                if (correct and answer_dec["disposition"] in ("escalated",
                                                              "withheld")
                        and answer_dec["reason_code"] in
                        ("led_quarantined_contradiction",
                         "led_superseded_supersession")):
                    pr123["frozen_baseline_suppressed_on_correct"] += 1
                if answer_dec["disposition"] == "escalated":
                    counters["escalated_on_correct"] += int(correct)
                elif answer_dec["disposition"] == "withheld":
                    counters["withheld_on_correct"] += int(correct)
                elif not correct:  # wrong row entering the prompt (§5 E)
                    mark = ("dual_presented"
                            if d["dual_classes"] is not None
                            else "caveated"
                            if answer_dec["disposition"]
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
        ambiguous_queue = build_ambiguous_queue(router, led_rows)
        if emit_review_queue:
            for rec in review_queue:
                aud_f.write(json.dumps(rec) + "\n")
        if emit_ambiguous_queue:
            for rec in ambiguous_queue:
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

    # §5 chance-baseline denominator: the cell's decode-class alphabet size
    # (distinct decode classes observed in topk; label-free, no registry).
    n_decode_classes = len({c["decode"]
                            for cs in cell["topk"].values() for c in cs})

    return {"counters": dict(counters), "pr123_counters": dict(pr123),
            "out_dir": out_dir, "n_decode_classes": n_decode_classes,
            "router_counts": {"n_conflict_pairs": len(router["pairs"]),
                              "n_merge_suspect_events": len(router["merge"])},
            "hazard_source_counts": {
                "n_conflict_pairs": hz_router.get("n_conflict_pairs"),
                "n_merge_suspect_events": hz_router.get(
                    "n_merge_suspect_events"),
                "n_rows": hz_none.get("n"),
                "stale_wrong_none": hz_none.get("stale_wrong"),
                "contra_wrong_none": hz_none.get("contra_wrong"),
                "ambiguous_pairs_final": hz_router.get(
                    "final_epoch_verdicts", {}).get("ambiguous", 0)},
            "merge_flag_diag": {"agree": agree, "mismatch": mismatch},
            "hazard_tier": hazard_tier,
            "review_queue": review_queue,
            "ambiguous_queue": ambiguous_queue,
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


def _write_report(scan_root: Path, report: dict,
                  fname: str = "reshape_scan.json"):
    scan_root.mkdir(parents=True, exist_ok=True)
    with open(scan_root / fname, "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
        f.write("\n")


# ---------------------------------------------------------------------------
# PR-12.2 §5 — pending dual-presentation scan (gates scored, no discretion)
# ---------------------------------------------------------------------------
def run_scan12_2(repo: Path, policy: dict, allow_stale: bool) -> int:
    scan = policy["scan12_2"]
    shapes, cells, pv = scan["shapes"], scan["cells"], scan["policy_version"]
    s_ceil, m_ceil = scan["suppression_ceiling"], \
        scan["presentation_mass_ceiling"]
    t_floor = scan["truth_containment_floor"]
    scan_root = repo / scan["output_root"]
    report = {"design_memo": scan["design_memo"], "policy_version": pv,
              "ceilings": {"G-S": s_ceil, "G-M": m_ceil, "G-T": t_floor},
              "shapes": shapes, "cells": list(cells), "gates": {},
              "exposure": {}, "counters": {}, "kill_conditions": []}

    def kill(cond, label, detail):
        ok = not cond
        print(f"  {'PASS' if ok else 'KILL'}  {label}"
              + ("" if ok else f": {detail}"))
        if cond:
            report["kill_conditions"].append({"label": label,
                                              "detail": detail})

    print("§8 prologue: PR-12 base byte-gate")
    before = base_bytecheck(repo, policy, allow_stale)
    report["base_bytecheck_before"] = before
    kill(not before, "kill-2: base byte-gate green (before)",
         "pr12/ regeneration drifted")

    results = {}
    for shape in shapes:
        for name, cfg in cells.items():
            try:
                results[(shape, name)] = run_cell(
                    repo, name, cfg, policy, allow_stale,
                    out_root=scan_root / shape, shape=shape,
                    policy_version=pv, emit_review_queue=True,
                    emit_ambiguous_queue=True)
            except FileNotFoundError as e:
                kill(True, f"kill-1: committed input present "
                     f"[{shape}/{name}]", str(e))
    if report["kill_conditions"]:
        report["verdict"] = "pending-blocked"
        _write_report(scan_root, report, "pending_scan.json")
        print(f"\nVERDICT: pending-blocked — {report['kill_conditions']}")
        return 1

    # ---- kill conditions 1 (cross-checks), 3 (byte-no-op), 4 (exactness)
    for name, cfg in cells.items():
        res = results[("prototype", name)]
        c, rc, hz = res["counters"], res["router_counts"], \
            res["hazard_source_counts"]
        kill(hz["n_conflict_pairs"] is not None
             and rc["n_conflict_pairs"] != hz["n_conflict_pairs"],
             f"kill-1: router pairs match committed [{name}]",
             f"rebuilt {rc['n_conflict_pairs']} vs {hz['n_conflict_pairs']}")
        kill(hz["n_merge_suspect_events"] is not None
             and rc["n_merge_suspect_events"]
             != hz["n_merge_suspect_events"],
             f"kill-1: router merges match committed [{name}]",
             f"rebuilt {rc['n_merge_suspect_events']} vs "
             f"{hz['n_merge_suspect_events']}")
        kill(hz["n_rows"] is not None and c["n"] != hz["n_rows"],
             f"kill-1: row count matches committed [{name}]",
             f"{c['n']} vs {hz['n_rows']}")
        n_amb = len(res["ambiguous_queue"])
        kill(n_amb != hz["ambiguous_pairs_final"],
             f"kill-1: final-epoch ambiguous pairs match committed "
             f"[{name}]",
             f"rebuilt {n_amb} vs committed {hz['ambiguous_pairs_final']}")
        if "expected_ambiguous_pairs" in cfg:
            kill(n_amb != cfg["expected_ambiguous_pairs"],
                 f"kill-1: one-shot ambiguous pairs == "
                 f"{cfg['expected_ambiguous_pairs']} [{name}]",
                 f"got {n_amb}")
        kill(c["abstain_led_merge_mismatch"] != 0,
             f"kill-4: certified abstain set == merge-led set [{name}]",
             f"{c['abstain_led_merge_mismatch']} mismatches")

    def audit_equal_modulo_e1(fresh_path: Path, committed_path: Path):
        """Erratum E1 (memo §11): record-by-record equality after
        excluding ONLY (1) per-record policy_version fields and (2)
        appended ambiguous_pair_review records; otherwise exact."""
        def norm(path, drop_ambiguous):
            out = []
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("record_type") == "ambiguous_pair_review":
                        if drop_ambiguous:
                            continue
                        return None  # unexpected on the committed side
                    for dec in rec.get("decisions", []):
                        dec.pop("policy_version", None)
                    out.append(rec)
            return out
        fresh = norm(fresh_path, drop_ambiguous=True)
        committed = norm(committed_path, drop_ambiguous=False)
        return committed is not None and fresh == committed

    shared_baseline = repo / scan["shared_prototype_baseline"]
    for name in cells:
        base_dir = shared_baseline / name
        if not base_dir.exists():
            continue  # one-shot cells: new baseline (§8)
        for fn in ("memory_packet.jsonl", "decision_table.csv"):
            fresh = (scan_root / "prototype" / name / fn).read_bytes()
            committed = (base_dir / fn).read_bytes()
            kill(fresh != committed,
                 f"kill-3: prototype byte-no-op vs pr12_1 [{name}/{fn}]",
                 f"{len(fresh)} vs {len(committed)} bytes (strict byte "
                 f"identity; NOT covered by erratum E1)")
        kill(not audit_equal_modulo_e1(
                scan_root / "prototype" / name / "audit_packet.jsonl",
                base_dir / "audit_packet.jsonl"),
             f"kill-3: prototype audit packet == pr12_1 modulo E1 "
             f"exclusions [{name}]",
             "differences beyond the two E1 exclusions "
             "(policy_version fields / appended ambiguous_pair_review "
             "records)")

    # ---- per-candidate gates (recorded even if killed; judged only if not)
    npend_keys = ("state:quarantined", "state:superseded",
                  "state:human-review", "abstained",
                  "stale_wrong_flagged_served", "quarantine_led_served")
    for shape in shapes:
        report["gates"][shape] = {}
        report["exposure"][shape] = {}
        report["counters"][shape] = {}
        print(f"\n[{shape}]")
        for name, cfg in cells.items():
            res, proto = results[(shape, name)], \
                results[("prototype", name)]
            c = res["counters"]
            denom = c["n"] - c["wrong_none"]
            supp = c.get("escalated_on_correct", 0) \
                + c.get("withheld_on_correct", 0)
            mass = c.get("pending_dual_on_correct", 0)
            dw = c.get("pending_dual_wrong", 0)
            tc = c.get("pending_dual_truth_contained", 0)
            unmarked = sum(v["unmarked"] for v in res["exposure"].values())
            unmarked_p = sum(v["unmarked"]
                             for v in proto["exposure"].values())
            inv = packet_invariants(res["out_dir"])
            gates = {
                "G-S": {"suppressive_on_correct": supp,
                        "correct_traffic": denom,
                        "rate": round(supp / denom, 6) if denom else 0.0,
                        "pass": (supp / denom if denom else 0) <= s_ceil},
                "G-M": {"dual_on_correct": mass,
                        "rate": round(mass / denom, 6) if denom else 0.0,
                        "pass": (mass / denom if denom else 0) <= m_ceil},
                "V1'": {"violations": c.get("v1p_violations", 0),
                        "pass": c.get("v1p_violations", 0) == 0},
                "V2'": {"violations": c.get("v2p_violations", 0),
                        "unmarked": unmarked,
                        "unmarked_prototype": unmarked_p,
                        "pass": c.get("v2p_violations", 0) == 0
                        and unmarked <= unmarked_p},
                "V3'": {"contradiction_pairs": len(res["review_queue"]),
                        "ambiguous_pairs": len(res["ambiguous_queue"]),
                        "pass": res["review_queue"] == proto["review_queue"]
                        and res["ambiguous_queue"]
                        == proto["ambiguous_queue"]},
                "V4'": {**inv, "pass": inv["certified_leaks"] == 0
                        and inv["incomplete_audits"] == 0},
                "G-R_non_pending_identical": {
                    "pass": all(c.get(k, 0) == proto["counters"].get(k, 0)
                                for k in npend_keys)},
            }
            if cfg["role"] in ("pending_burden", "harm_class"):
                gates["G-T"] = {"dual_wrong": dw, "truth_contained": tc,
                                "fraction": round(tc / dw, 6) if dw
                                else None,
                                "pass": (tc / dw >= t_floor) if dw
                                else True}
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
            print(f"  [{name}] " + "  ".join(
                f"{g}={'PASS' if v['pass'] else 'FAIL'}"
                + (f"({v['rate']:.3f})" if g in ("G-S", "G-M") else
                   f"({v['fraction']})" if g == "G-T" else "")
                for g, v in gates.items()))

    print("\n§8 epilogue: PR-12 base byte-gate")
    after = base_bytecheck(repo, policy, allow_stale)
    report["base_bytecheck_after"] = after
    kill(not after, "kill-2: base byte-gate green (after)",
         "pr12/ regeneration drifted")

    if report["kill_conditions"]:
        report["verdict"] = "pending-blocked"
        _write_report(scan_root, report, "pending_scan.json")
        print(f"\nVERDICT: pending-blocked — "
              f"{len(report['kill_conditions'])} kill condition(s); no "
              "candidate judged from a blocked run (memo §9).")
        return 1

    candidates_pass = [
        s for s in shapes if s != "prototype"
        and all(g["pass"] for name in cells
                for g in report["gates"][s][name].values())]
    report["candidates_pass"] = candidates_pass
    report["verdict"] = (f"pending-evidence-GO({','.join(candidates_pass)})"
                         if candidates_pass else "pending-negative")
    _write_report(scan_root, report, "pending_scan.json")
    print(f"\nVERDICT: {report['verdict']}")
    print("Scope (PR12_2_PENDING_DUAL_PRESENTATION.md §5): offline-"
          "simulator evidence only — no prompt-safety, promotion, memory-"
          "ingestion, or autonomous downstream-use claim.")
    return 0


# ---------------------------------------------------------------------------
# PR-12.3 §5 — suppression-attribution + witness-window scan (no discretion)
# ---------------------------------------------------------------------------
def run_scan12_3(repo: Path, policy: dict, allow_stale: bool) -> int:
    scan = policy["scan12_3"]
    shapes, cells, pv = scan["shapes"], scan["cells"], scan["policy_version"]
    a_ceil = scan["attributable_suppression_ceiling"]
    m_ceil = scan["presentation_mass_ceiling"]
    t_floor = scan["truth_containment_floor"]
    margin = scan["chance_baseline_margin"]
    width_bound = scan["width_bound_classes"]
    gated_roles = set(scan["gated_roles"])
    scan_root = repo / scan["output_root"]
    report = {"design_memo": scan["design_memo"], "policy_version": pv,
              "ceilings": {"G-A": a_ceil, "G-M": m_ceil, "G-T": t_floor,
                           "chance_baseline_margin": margin,
                           "width_bound_classes": width_bound},
              "gated_roles": sorted(gated_roles),
              "shapes": shapes, "cells": list(cells),
              "roles": {n: cells[n]["role"] for n in cells},
              "gates": {}, "report_only": {}, "exposure": {},
              "counters": {}, "n_decode_classes": {},
              "kill_conditions": []}

    def kill(cond, label, detail):
        ok = not cond
        print(f"  {'PASS' if ok else 'KILL'}  {label}"
              + ("" if ok else f": {detail}"))
        if cond:
            report["kill_conditions"].append({"label": label,
                                              "detail": detail})

    def merged(r):
        """Existing PR-12.2 counters plus the namespaced PR-12.3 counters
        (disjoint keys). run_scan12_3 scores from this merged view; the
        shared run_cell keeps them separate so scan12_2 stays byte-identical."""
        return {**r["counters"], **r["pr123_counters"]}

    print("§8 prologue: PR-12 base byte-gate")
    before = base_bytecheck(repo, policy, allow_stale)
    report["base_bytecheck_before"] = before
    kill(not before, "kill-2: base byte-gate green (before)",
         "pr12/ regeneration drifted")

    # ---- run every (shape, cell); missing committed input -> kill-1
    results = {}
    for shape in shapes:
        for name, cfg in cells.items():
            try:
                results[(shape, name)] = run_cell(
                    repo, name, cfg, policy, allow_stale,
                    out_root=scan_root / shape, shape=shape,
                    policy_version=pv, emit_review_queue=True,
                    emit_ambiguous_queue=True)
            except FileNotFoundError as e:
                kill(True, f"kill-1: committed input present "
                     f"[{shape}/{name}]", str(e))
    if report["kill_conditions"]:
        report["verdict"] = "attribution-blocked"
        _write_report(scan_root, report, "attribution_scan.json")
        print(f"\nVERDICT: attribution-blocked — {report['kill_conditions']}")
        return 1
    for name in cells:
        report["n_decode_classes"][name] = \
            results[("prototype", name)]["n_decode_classes"]

    # ---- kill-1 cross-checks + kill-4 (abstain exactness) on prototype
    for name, cfg in cells.items():
        res = results[("prototype", name)]
        c, rc, hz = res["counters"], res["router_counts"], \
            res["hazard_source_counts"]
        kill(hz["n_conflict_pairs"] is not None
             and rc["n_conflict_pairs"] != hz["n_conflict_pairs"],
             f"kill-1: router pairs match committed [{name}]",
             f"rebuilt {rc['n_conflict_pairs']} vs {hz['n_conflict_pairs']}")
        kill(hz["n_merge_suspect_events"] is not None
             and rc["n_merge_suspect_events"]
             != hz["n_merge_suspect_events"],
             f"kill-1: router merges match committed [{name}]",
             f"rebuilt {rc['n_merge_suspect_events']} vs "
             f"{hz['n_merge_suspect_events']}")
        kill(hz["n_rows"] is not None and c["n"] != hz["n_rows"],
             f"kill-1: row count matches committed [{name}]",
             f"{c['n']} vs {hz['n_rows']}")
        n_amb = len(res["ambiguous_queue"])
        kill(n_amb != hz["ambiguous_pairs_final"],
             f"kill-1: final-epoch ambiguous pairs match committed [{name}]",
             f"rebuilt {n_amb} vs committed {hz['ambiguous_pairs_final']}")
        if "expected_ambiguous_pairs" in cfg:
            kill(n_amb != cfg["expected_ambiguous_pairs"],
                 f"kill-1: one-shot ambiguous pairs == "
                 f"{cfg['expected_ambiguous_pairs']} [{name}]",
                 f"got {n_amb}")
        kill(c["abstain_led_merge_mismatch"] != 0,
             f"kill-4: certified abstain set == merge-led set [{name}]",
             f"{c['abstain_led_merge_mismatch']} mismatches")

    # ---- kill-3 prototype byte-no-op vs pr12_2/prototype (§5 rule)
    def audit_equal_modulo_pv(fresh_path: Path, committed_path: Path):
        """§5 comparison rule (adopted up front, the E1 lesson): audit packet
        equal record-by-record after excluding ONLY the per-record
        policy_version field — no other exclusion, since PR-12.3 appends no
        new record type beyond the 12.2 baseline (both carry the two
        review-queue record types)."""
        def norm(path):
            out = []
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    for dec in rec.get("decisions", []):
                        dec.pop("policy_version", None)
                    out.append(rec)
            return out
        return norm(fresh_path) == norm(committed_path)

    shared_baseline = repo / scan["shared_prototype_baseline"]
    for name in cells:
        base_dir = shared_baseline / name
        kill(not base_dir.exists(),
             f"kill-3: pr12_2 prototype baseline present [{name}]",
             f"{base_dir} missing")
        if not base_dir.exists():
            continue
        for fn in ("memory_packet.jsonl", "decision_table.csv"):
            fresh = (scan_root / "prototype" / name / fn).read_bytes()
            committed = (base_dir / fn).read_bytes()
            kill(fresh != committed,
                 f"kill-3: prototype byte-no-op vs pr12_2 [{name}/{fn}]",
                 f"{len(fresh)} vs {len(committed)} bytes (strict byte "
                 f"identity)")
        kill(not audit_equal_modulo_pv(
                scan_root / "prototype" / name / "audit_packet.jsonl",
                base_dir / "audit_packet.jsonl"),
             f"kill-3: prototype audit == pr12_2 modulo policy_version "
             f"[{name}]",
             "differences beyond the per-record policy_version field")

    # ---- kill-5/6/7/8 across candidate shapes (structural instrumentation)
    npend_keys = ("state:quarantined", "state:superseded",
                  "state:human-review", "abstained",
                  "stale_wrong_flagged_served", "quarantine_led_served")
    for shape in shapes:
        if shape == "prototype":
            continue
        for name in cells:
            res, proto = results[(shape, name)], results[("prototype", name)]
            c, pc = merged(res), merged(proto)
            kill(any(c.get(k, 0) != pc.get(k, 0) for k in npend_keys),
                 f"kill-5: no candidate diff outside pending-led rows "
                 f"[{shape}/{name}]",
                 "non-pending disposition counters differ from prototype")
            kill(c.get("pending_dual_width_gt3", 0) != 0,
                 f"kill-6: width bound (<= {width_bound} classes) holds "
                 f"[{shape}/{name}]",
                 f"{c.get('pending_dual_width_gt3', 0)} rows exceed "
                 f"{width_bound} presented classes")
            kill(res["review_queue"] != proto["review_queue"]
                 or res["ambiguous_queue"] != proto["ambiguous_queue"],
                 f"kill-7: review queues identical across shapes "
                 f"[{shape}/{name}]",
                 "contradiction/ambiguous review queue diverged")
            inv = packet_invariants(res["out_dir"])
            kill(inv["certified_leaks"] != 0,
                 f"kill-8: 'certified' only in permitted fields "
                 f"[{shape}/{name}]",
                 f"{inv['certified_leaks']} leaks")

    print("\n§8 epilogue: PR-12 base byte-gate")
    after = base_bytecheck(repo, policy, allow_stale)
    report["base_bytecheck_after"] = after
    kill(not after, "kill-2: base byte-gate green (after)",
         "pr12/ regeneration drifted")

    if report["kill_conditions"]:
        report["verdict"] = "attribution-blocked"
        _write_report(scan_root, report, "attribution_scan.json")
        print(f"\nVERDICT: attribution-blocked — "
              f"{len(report['kill_conditions'])} kill condition(s); no "
              "candidate judged from a blocked run (§7).")
        return 1

    # ---- economics measurements (G-A / G-M / G-T with chance baseline) -----
    def economics(res):
        c = merged(res)
        denom = c["n"] - c["wrong_none"]
        supp = c.get("pending_led_suppressed_on_correct", 0)
        mass = c.get("pending_dual_on_correct", 0)
        dw = c.get("pending_dual_wrong", 0)
        tc = c.get("pending_dual_truth_contained", 0)
        wsum = c.get("pending_dual_wrong_width_sum", 0)
        n_dec = res["n_decode_classes"]
        containment = (tc / dw) if dw else None
        chance = (wsum / (dw * n_dec)) if dw else None
        floor_pass = (containment >= t_floor) if dw else True
        # §5: a genuine G-T pass must beat its own same-width chance-baseline
        # by >= margin, not merely clear the fixed 0.5 floor. Clearing 0.5
        # without the margin is width-saturated and NOT a pass (kill-9).
        margin_pass = (containment >= chance + margin) if dw else True
        width_saturated = bool(dw) and floor_pass and not margin_pass
        genuine = floor_pass and margin_pass
        return {
            "denom_correct_traffic": denom,
            "G-A": {"attributable_suppressed_on_correct": supp,
                    "rate": round(supp / denom, 6) if denom else 0.0,
                    "ceiling": a_ceil,
                    "pass": (supp / denom if denom else 0) <= a_ceil},
            "G-M": {"dual_on_correct": mass,
                    "rate": round(mass / denom, 6) if denom else 0.0,
                    "ceiling": m_ceil,
                    "pass": (mass / denom if denom else 0) <= m_ceil},
            "G-T": {"dual_wrong": dw, "truth_contained": tc,
                    "containment_rate": round(containment, 6)
                    if containment is not None else None,
                    "floor": t_floor, "floor_pass": floor_pass,
                    "n_decode_classes": n_dec,
                    "chance_baseline_rate": round(chance, 6)
                    if chance is not None else None,
                    "chance_margin": margin,
                    "chance_plus_margin": round(chance + margin, 6)
                    if chance is not None else None,
                    "margin_pass": margin_pass,
                    "width_saturated": width_saturated,
                    "genuine": genuine, "pass": genuine}}

    def frozen_baseline(res):
        c = merged(res)
        denom = c["n"] - c["wrong_none"]
        fb = c.get("frozen_baseline_suppressed_on_correct", 0)
        return {"frozen_baseline_suppressed_on_correct": fb,
                "rate": round(fb / denom, 6) if denom else 0.0,
                "note": "quarantine-led + superseded; design-frozen, gated "
                        "nowhere (§5)"}

    def width_report(res):
        c = merged(res)
        dual = c.get("pending_dual", 0)
        dw = c.get("pending_dual_wrong", 0)
        wsum = c.get("pending_dual_wrong_width_sum", 0)
        n_dec = res["n_decode_classes"]
        return {"dual_presented": dual,
                "over_bound_rows": c.get("pending_dual_width_gt3", 0),
                "truncated_rows": c.get("pending_dual_truncated_rows", 0),
                "truncated_classes": c.get("pending_dual_truncated_classes", 0),
                "full_width_rows": c.get("pending_dual_full_width", 0),
                "full_width_fraction": round(
                    c.get("pending_dual_full_width", 0) / dual, 6)
                    if dual else 0.0,
                "fallback_pair_counterpart_rows":
                    c.get("pending_dual_fallback_paircp", 0),
                "witness_rows": c.get("pending_dual_witness", 0),
                "empty_candidate_set_rows":
                    c.get("pending_dual_empty_candidate_set", 0),
                "chance_baseline_rate_wrong": round(wsum / (dw * n_dec), 6)
                    if dw else None,
                "selection": "vote-mass-descending, tie-break ascending "
                             "decode-class index (§3)"}

    # ---- per-candidate gate scoring ---------------------------------------
    for shape in shapes:
        report["gates"][shape] = {}
        report["report_only"][shape] = {}
        report["exposure"][shape] = {}
        report["counters"][shape] = {}
        print(f"\n[{shape}]")
        for name, cfg in cells.items():
            res, proto = results[(shape, name)], results[("prototype", name)]
            c, proto_c = merged(res), merged(proto)
            role = cfg["role"]
            econ = economics(res)
            wrep = width_report(res)
            inv = packet_invariants(res["out_dir"])
            unmarked = sum(v["unmarked"] for v in res["exposure"].values())
            unmarked_p = sum(v["unmarked"]
                             for v in proto["exposure"].values())
            gates = {
                # G-W width integrity (hard, every cell)
                "G-W": {"over_bound_rows": c.get("pending_dual_width_gt3", 0),
                        "width_bound": width_bound, **wrep,
                        "pass": c.get("pending_dual_width_gt3", 0) == 0},
                # G-V visibility (hard, every cell)
                "V1": {"violations": c.get("v1p_violations", 0),
                       "pass": c.get("v1p_violations", 0) == 0},
                "V2": {"violations": c.get("v2p_violations", 0),
                       "unmarked": unmarked, "unmarked_prototype": unmarked_p,
                       "pass": c.get("v2p_violations", 0) == 0
                       and unmarked <= unmarked_p},
                "V3": {"contradiction_pairs": len(res["review_queue"]),
                       "ambiguous_pairs": len(res["ambiguous_queue"]),
                       "pass": res["review_queue"] == proto["review_queue"]
                       and res["ambiguous_queue"]
                       == proto["ambiguous_queue"]},
                "V4": {**inv, "pass": inv["certified_leaks"] == 0
                       and inv["incomplete_audits"] == 0},
                # G-R regression (hard, every cell)
                "G-R_non_pending_identical": {
                    "pass": all(c.get(k, 0) == proto_c.get(k, 0)
                                for k in npend_keys)},
            }
            if role in gated_roles:  # G-A/G-M/G-T hard on gated harm class
                gates["G-A"] = econ["G-A"]
                gates["G-M"] = econ["G-M"]
                gates["G-T"] = econ["G-T"]
            if role == "continuity":
                a = cfg["anchors"]
                gates["G-R_anchors"] = {"pass": (
                    c["abstained"] == a["certified_abstentions"]
                    and c["stale_wrong"] == a["stale_wrong_total"]
                    and c["stale_wrong_abstained"]
                    == a["stale_wrong_abstained"]
                    and c["stale_wrong_flagged_served"]
                    == a["stale_wrong_residual_flagged"]
                    and c["stale_wrong_unflagged_served"] == 0)}
            if role == "control":
                adverse = sum(v for k, v in c.items()
                              if k.startswith("state:")
                              and k.split(":", 1)[1] != "agent-readable") \
                    + c.get("disposition:shown_with_caveat", 0) \
                    + c.get("disposition:withheld", 0) \
                    + c.get("disposition:escalated", 0)
                gates["G-R_control_zero_adverse"] = {
                    "adverse": adverse, "pass": adverse == 0}
                gates["G-M_control_structural_zero"] = {
                    "dual_on_correct": econ["G-M"]["dual_on_correct"],
                    "pass": econ["G-M"]["dual_on_correct"] == 0}
            report["gates"][shape][name] = gates
            report["report_only"][shape][name] = {
                "role": role, "gated": role in gated_roles,
                "economics": econ,
                "frozen_baseline_suppression": frozen_baseline(res),
                "width": wrep}
            report["exposure"][shape][name] = res["exposure"]
            report["counters"][shape][name] = c  # merged view (§8 recompute)
            gt = gates.get("G-T")
            print(f"  [{name}] ({role}) " + "  ".join(
                f"{g}={'PASS' if v['pass'] else 'FAIL'}"
                for g, v in gates.items())
                + (f"  | cont={gt['containment_rate']} "
                   f"chance={gt['chance_baseline_rate']}"
                   + ("  WIDTH-SATURATED" if gt["width_saturated"] else "")
                   if gt else ""))

    # ---- verdict: candidate passes every hard gate on every applicable cell
    candidates_pass = [
        s for s in shapes if s != "prototype"
        and all(g["pass"] for name in cells
                for g in report["gates"][s][name].values())]

    # ---- kill-9: containment-inflation guard (§7.9). Mathematically a
    # width-saturated cell already fails G-T (genuine=False), so no such
    # candidate can be in candidates_pass; this asserts that invariant.
    for s in list(candidates_pass):
        for name, cfg in cells.items():
            if cfg["role"] in gated_roles \
                    and report["gates"][s][name].get(
                        "G-T", {}).get("width_saturated"):
                kill(True, f"kill-9: containment-inflation guard [{s}/{name}]",
                     "a GO candidate carries a width-saturated G-T (clears "
                     "0.5 but not chance-baseline + margin)")
    if report["kill_conditions"]:
        report["verdict"] = "attribution-blocked"
        _write_report(scan_root, report, "attribution_scan.json")
        print("\nVERDICT: attribution-blocked — containment-inflation "
              "contradiction (§7.9).")
        return 1

    report["candidates_pass"] = candidates_pass
    report["verdict"] = (
        f"attribution-evidence-GO({','.join(candidates_pass)})"
        if candidates_pass else "attribution-negative")
    _write_report(scan_root, report, "attribution_scan.json")
    print(f"\nVERDICT: {report['verdict']}")
    print("Scope (PR12_3_SUPPRESSION_ATTRIBUTION.md §5): offline-simulator "
          "evidence only — no global mechanism-(d) certification, agent "
          "prompting, promotion, memory ingestion, autonomous downstream "
          "use, or FAM-core change; PR-10 merge-abstain remains the only "
          "certified reader contract.")
    return 0


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
    ap.add_argument("--scan12-2", action="store_true",
                    help="PR-12.2 §5 full scan: shapes prototype/D1/D2 × "
                         "the six-cell panel, gates G-S/G-M/G-T/G-V'/G-R "
                         "scored with no discretion, pending_scan.json + "
                         "verdict; §9 kill conditions enforced")
    ap.add_argument("--scan12-3", action="store_true",
                    help="PR-12.3 §5 full scan: shapes prototype/W1/W2 × the "
                         "six-cell panel; gates G-A/G-M/G-T(+chance-baseline)/"
                         "G-W/G-V/G-R scored with no discretion, "
                         "attribution_scan.json + verdict; §7 kill conditions "
                         "enforced. Candidate economics (G-A/G-M/G-T) hard on "
                         "the two one-shot harm-class cells only; contra "
                         "cells report-only (§2)")
    ap.add_argument("--shape12-3", choices=["prototype", "W1", "W2"],
                    help="PR-12.3 §3 witness-window disposition shape for "
                         "pending-led served answers; emits one shape over "
                         "the panel under pr12_3/<shape>/ without gate "
                         "scoring")
    args = ap.parse_args()

    with open(Path(__file__).parent / "harness_policy.json") as f:
        policy = json.load(f)
    if args.scan12_3:
        sys.exit(run_scan12_3(args.repo_root, policy, args.allow_stale))
    if args.scan12_2:
        sys.exit(run_scan12_2(args.repo_root, policy, args.allow_stale))
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
    if args.shape12_3:
        scan = policy["scan12_3"]
        for name, cfg in scan["cells"].items():
            res = run_cell(args.repo_root, name, cfg, policy,
                           args.allow_stale,
                           out_root=args.repo_root / scan["output_root"]
                           / args.shape12_3,
                           shape=args.shape12_3,
                           policy_version=scan["policy_version"],
                           emit_review_queue=True, emit_ambiguous_queue=True)
            print(f"[{args.shape12_3}/{name}] -> {res['out_dir']} | "
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
