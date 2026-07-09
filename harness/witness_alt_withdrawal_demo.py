#!/usr/bin/env python3
"""PR-12.9 Stage I — synthetic-tripwire withdrawal-mechanics demonstration.

Artifact §5.2 of `PR12_9_READER_CONTRACT_CERTIFICATION.md` (main
`fe6248e`): demonstrates, on synthetic inputs only, that each
registered tripwire T1-T7 of `PR12_8_MONITORING_WITHDRAWAL.md`, when
injected, produces (a) the registered reversion/suspension behavior and
(b) a well-formed append-only event record hash-chained to its
predecessor (PR-12.9 §9). Demonstration only; nothing live, nothing
served, no committed packet consumed.

Registered semantics exercised (monitoring doc §3/§4, PR-12.9 §3):

* T1 envelope-multiset exactness (any byte)  -> candidate-wide
  suspension (`withdrawn-pending-review`);
* T2 composition exactness (I2 overlap row / incumbent-field deviation
  / non-`harness-heuristic` witness_alt)     -> candidate-wide
  suspension;
* T3 per-cell wrong-mass rate > 0.10 (truth-joined) -> cell reversion;
  on two or more cells of one engagement     -> candidate-wide
  suspension;
* T4 acting precision < 0.75 where acted >= 1 row  -> cell reversion
  (same multi-cell escalation);
* T5 aggregate wrong-mass rate > 0.05        -> candidate-wide
  suspension;
* T6 guard margin inside the unobserved corridor (-4, +23), open
  interval, evidence-gap fail-closed          -> cell reversion BEFORE
  any row is served (corridor cells are outside the evidence base);
* T7 any fail-closed event on a cell          -> cell reversion.

Cell reversion = the candidate is silent on the cell: every row it
would have served ``witness_alt`` is served ``defer``; incumbent
``abstain``/``answer`` rows are unaffected. Suspension = recorded
status ``withdrawn-pending-review``; no ``witness_alt`` on ANY cell;
no self-reinstatement (reinstatement only via a new pre-registration).
Withdrawal without a recorded event — or an event without the posture
change — is a conformance violation; the demo checks the coupling in
both directions. Truth labels appear ONLY as synthetic fixtures for
the truth-joined tripwires (T3/T4/T5) and only retrospectively, after
every serving decision is recorded, per monitoring doc §2.

Writes ONLY under ``results/issue_failure_mode_blindness/pr12_9/``:
``withdrawal_demo_events.jsonl`` (the append-only hash-chained event
log — the one artifact excluded from cross-run byte-identity, since
events accumulate by design) and ``withdrawal_demo_report.json``
(deterministic, byte-identical across runs).

**This program certifies nothing and serves nothing to anyone.** No
deployment, live acting, prompting use, promotion, memory ingestion,
FAM-core change, autonomous downstream use, or reader-contract change.
PR-10 merge-abstain remains the only certified reader contract; the
operational posture on witness-window rows remains deferral.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
REPO = HARNESS_DIR.parent
sys.path.insert(0, str(HARNESS_DIR))

import witness_alt_conformance_tests as ct  # noqa: E402  (§5.1 artifact)
import witness_alt_reference_reader as wr   # noqa: E402  (committed reader)

OUT_DIR = REPO / ct.BASE / "pr12_9"
LOG_FILE = OUT_DIR / "withdrawal_demo_events.jsonl"
REPORT_FILE = OUT_DIR / "withdrawal_demo_report.json"

DEMO_VERSION = "pr12_9-stage1-withdrawal-demo-v1"
READER_DESC = ("witness_alt_conformance_tests.serve_conformant "
               "(attested frozen policy block)")

# Registered tripwire constants (monitoring doc §3; the 12.6 constants
# and the corridor are NOT movable by this or any later document).
T3_PER_CELL_WRONG_CEILING = 0.10     # strict >
T4_PRECISION_FLOOR = 0.75            # strict <
T5_AGGREGATE_WRONG_CEILING = 0.05    # strict >
CORRIDOR = (-4, 23)                  # open interval, both ends excluded

GENESIS = hashlib.sha256(
    b"pr12_9 withdrawal-demo event-chain genesis").hexdigest()

EVENT_REQUIRED = ("seq", "prev_sha256", "scenario", "tripwire", "cell",
                  "packet_shas", "measured", "registered", "action",
                  "scope", "contract_id", "contract_version", "reader",
                  "monitor_version")


class EventLog:
    """Append-only, hash-chained (each line's sha256 is the next line's
    prev_sha256; the first-ever event chains to a fixed genesis hash).
    Accumulates across runs by design (PR-12.9 §9)."""

    def __init__(self, path: Path):
        self.path = path
        self.seq = 0
        self.prev = GENESIS
        if path.exists():
            for line in path.read_text().splitlines():
                self.prev = hashlib.sha256(line.encode()).hexdigest()
                self.seq += 1

    def append(self, ev: dict) -> dict:
        ev = dict(ev)
        ev["seq"] = self.seq
        ev["prev_sha256"] = self.prev
        line = json.dumps(ev, sort_keys=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")
        self.prev = hashlib.sha256(line.encode()).hexdigest()
        self.seq += 1
        return ev

    @staticmethod
    def verify_chain(path: Path):
        prev, n = GENESIS, 0
        for line in path.read_text().splitlines():
            ev = json.loads(line)
            if ev.get("prev_sha256") != prev or ev.get("seq") != n:
                return False, n
            prev = hashlib.sha256(line.encode()).hexdigest()
            n += 1
        return True, n


def event_well_formed(ev: dict) -> bool:
    return (all(k in ev for k in EVENT_REQUIRED)
            and ev["tripwire"] in {"T1", "T2", "T3", "T4", "T5", "T6", "T7"}
            and ev["action"] in {"cell-reversion", "candidate-suspension"}
            and ev["scope"] in {"cell", "candidate"}
            and isinstance(ev["packet_shas"], dict)
            and all(len(v) == 64 for v in ev["packet_shas"].values())
            and ev["measured"] != "" and ev["registered"] != ""
            and ev["contract_id"] == ct.CONTRACT_ID
            and ev["contract_version"] == ct.CONTRACT_VERSION)


def revert_records(records):
    """Registered cell reversion / suspension effect on served records:
    the candidate falls silent — witness_alt -> defer (candidate fields
    blanked); incumbent abstain/answer rows are untouched."""
    out = []
    for r in records:
        if r["served_outcome"] == "witness_alt":
            out.append(ct.blank_record(r["query_id"]))
        else:
            out.append(dict(r))
    return out


class Monitor:
    """The registered monitoring/withdrawal layer, exercised on
    synthetic cells. Posture state is per-run and in-memory; the event
    log is the durable, append-only record."""

    def __init__(self, log: EventLog, scenario: str):
        self.log = log
        self.scenario = scenario
        self.cell_posture = {}                     # cell -> "reverted"
        self.candidate_status = "active"
        self.trips = []

    def _trip(self, tw, cell, measured, registered, shas, scope,
              extra=None):
        if scope == "cell":
            self.cell_posture[cell] = "reverted"
            action = "cell-reversion"
        else:
            self.candidate_status = "withdrawn-pending-review"
            action = "candidate-suspension"
        ev = {"scenario": self.scenario, "tripwire": tw, "cell": cell,
              "packet_shas": shas, "measured": str(measured),
              "registered": str(registered), "action": action,
              "scope": scope, "contract_id": ct.CONTRACT_ID,
              "contract_version": ct.CONTRACT_VERSION,
              "reader": READER_DESC, "monitor_version": DEMO_VERSION}
        if extra:
            ev.update(extra)
        ev = self.log.append(ev)
        self.trips.append(ev)
        return ev

    def _apply_posture(self, cell, records):
        if self.candidate_status != "active" \
                or self.cell_posture.get(cell) == "reverted":
            return revert_records(records)
        return records

    def exercise(self, cell, pkt_lines, aud_lines, envelope=None,
                 inject_tier_violation=False,
                 inject_envelope_mutation=False):
        """One monitored exercise of the candidate on one synthetic
        cell. Watched quantities per monitoring doc §2; tripwire order:
        T6 pre-serving (evidence-gap fail-closed), then T7, T2, T1;
        posture applied last."""
        shas = {
            "memory_packet":
                hashlib.sha256("\n".join(pkt_lines).encode()).hexdigest(),
            "audit_packet":
                hashlib.sha256("\n".join(aud_lines).encode()).hexdigest()}
        serve = ct.serve_conformant(pkt_lines, aud_lines, "W2", cell)
        ctx = serve["ctx"]
        margin = (ctx.n_contradiction_pairs - ctx.n_ambiguous_pairs
                  if ctx else None)
        corridor = (margin is not None
                    and CORRIDOR[0] < margin < CORRIDOR[1])
        records = [dict(r) for r in serve["records"]]

        # T6 — pre-serving: a corridor cell is outside the candidate's
        # evidence base; revert before any witness_alt is served.
        if corridor:
            self._trip("T6", cell, f"guard margin {margin}",
                       f"open interval ({CORRIDOR[0]}, {CORRIDOR[1]})",
                       shas, "cell")

        # injections simulating a nonconformant/drifted reader
        wa_idx = [i for i, r in enumerate(records)
                  if r["served_outcome"] == "witness_alt"]
        if inject_tier_violation and wa_idx:
            records[wa_idx[0]]["certification_tier"] = "core-certified"
        if inject_envelope_mutation and wa_idx:
            records[wa_idx[0]]["witness_alt_class"] = 999

        # T7 — fail-closed events on the cell
        n_fc = len(serve["fail_closed_events"])
        if n_fc > 0:
            self._trip("T7", cell, f"n_fail_closed_events={n_fc}",
                       "> 0 (any instance)", shas, "cell",
                       {"events": serve["fail_closed_events"]})

        # T2 — composition exactness (any instance -> suspension)
        tier_bad = [r["query_id"] for r in records
                    if r["served_outcome"] == "witness_alt"
                    and r["certification_tier"] != "harness-heuristic"]
        if serve["i2_overlap"] or tier_bad:
            self._trip("T2", cell,
                       f"i2_overlap={serve['i2_overlap']} "
                       f"tier_violations={tier_bad}",
                       "exactness: any I2 overlap row / incumbent-field "
                       "deviation / non-harness-heuristic witness_alt",
                       shas, "candidate")

        # T1 — envelope-cell multiset exactness (any byte -> suspension)
        if envelope is not None:
            got = sorted((r["query_id"], str(r["witness_alt_class"]))
                         for r in records
                         if r["served_outcome"] == "witness_alt")
            if got != envelope:
                self._trip("T1", cell,
                           f"witness_alt multiset {got}",
                           f"frozen envelope entry {envelope}",
                           shas, "candidate")

        served = self._apply_posture(cell, records)
        return {"cell": cell, "margin": margin, "corridor": corridor,
                "packet_shas": shas,
                "n_witness_alt_decided": len(wa_idx),
                "n_witness_alt_served": sum(
                    1 for r in served
                    if r["served_outcome"] == "witness_alt"),
                "records": served, "pre_posture_records": records}

    def truth_join(self, engagement):
        """Retrospective truth-joined tripwires (monitoring doc §2/§3),
        computed only after every serving decision is recorded.
        engagement: cell -> (served_records, truth_map, n_in_scope)."""
        t34_cells, total_wrong, total_scope = [], 0, 0
        for cell, (records, truth, n_scope) in sorted(engagement.items()):
            acts = [r for r in records
                    if r["served_outcome"] == "witness_alt"]
            wrong = [r for r in acts
                     if truth.get(r["query_id"]) != r["witness_alt_class"]]
            rate = len(wrong) / n_scope if n_scope else 0.0
            precision = ((len(acts) - len(wrong)) / len(acts)
                         if acts else None)
            total_wrong += len(wrong)
            total_scope += n_scope
            shas = {"memory_packet": "0" * 64, "audit_packet": "0" * 64}
            tripped = False
            if rate > T3_PER_CELL_WRONG_CEILING:
                self._trip("T3", cell, f"wrong-mass rate {rate:.4f}",
                           f"> {T3_PER_CELL_WRONG_CEILING}", shas, "cell")
                tripped = True
            if acts and precision < T4_PRECISION_FLOOR:
                self._trip("T4", cell, f"acting precision {precision:.4f}",
                           f"< {T4_PRECISION_FLOOR} (acted >= 1 row)",
                           shas, "cell")
                tripped = True
            if tripped:
                t34_cells.append(cell)
        if len(t34_cells) >= 2:              # registered escalation
            self._trip("T3", "(engagement)",
                       f"T3/T4 tripped on {len(t34_cells)} cells of one "
                       f"engagement: {t34_cells}",
                       ">= 2 cells -> candidate-wide suspension",
                       {"memory_packet": "0" * 64, "audit_packet": "0" * 64},
                       "candidate",
                       {"escalation_rule":
                        "two-or-more-cells-one-engagement"})
        agg = total_wrong / total_scope if total_scope else 0.0
        if agg > T5_AGGREGATE_WRONG_CEILING:
            self._trip("T5", "(engagement)",
                       f"aggregate wrong-mass rate {agg:.4f}",
                       f"> {T5_AGGREGATE_WRONG_CEILING}",
                       {"memory_packet": "0" * 64, "audit_packet": "0" * 64},
                       "candidate")
        return {"t34_cells": t34_cells, "aggregate_rate": round(agg, 6)}


# ---------------------------------------------------------------------------
# synthetic cell fixtures (built from the §5.1 suite's builders)
# ---------------------------------------------------------------------------
def mk_cell(name, n_eligible, n_plain_ties=0, n_answers=1,
            n_contra=0, n_ambig=1):
    """A synthetic W2 cell: margin = n_contra - n_ambig; n_eligible
    eligible tie rows (in-scope, acted when guard open), n_plain_ties
    ineligible width-3 tie rows (in-scope, never acted), n_answers
    memory rows."""
    qids, rows = [], []
    for i in range(n_eligible):
        q = f"{name}:e{i}"
        qids.append(q)
        rows.append(ct.mk_row(q, [ct.mk_tie()]))
    for i in range(n_plain_ties):
        q = f"{name}:p{i}"
        qids.append(q)
        rows.append(ct.mk_row(q, [ct.mk_tie(
            alts=((3, ct.WITNESS), (5, ct.WITNESS)))]))
    for i in range(n_answers):
        q = f"{name}:a{i}"
        qids.append(q)
        rows.append(ct.mk_row(q, [ct.mk_memory_item()]))
    ambigs = tuple((2, 128, True) if j == 0 else (200 + j, 300 + j, False)
                   for j in range(n_ambig))
    aud = ct.mk_audit(qids, n_contra=n_contra,
                      ambigs=ambigs if n_ambig else ())
    return rows, aud


def truth_map(records, n_correct_acts):
    """Synthetic retrospective truth: the first n_correct_acts acted
    rows are 'correct' (truth = served class 3), the rest 'wrong'
    (truth = deployed class 0). Fixture only — no real label exists."""
    truth, seen = {}, 0
    for r in records:
        if r["served_outcome"] == "witness_alt":
            truth[r["query_id"]] = 3 if seen < n_correct_acts else 0
            seen += 1
    return truth


SCENARIOS = []


def record_scenario(name, tripwire, expected, monitor, observed, ok,
                    detail=None):
    events = [{k: e[k] for k in ("tripwire", "cell", "measured",
                                 "registered", "action", "scope")}
              for e in monitor.trips]
    wf = all(event_well_formed(e) for e in monitor.trips)
    # bidirectional coupling: every posture change (each reverted cell,
    # plus a suspension if any) has exactly one recorded event, and
    # every recorded event effected exactly one posture change
    n_posture = len(monitor.cell_posture) + (
        0 if monitor.candidate_status == "active" else 1)
    SCENARIOS.append({
        "scenario": name, "tripwire": tripwire, "expected": expected,
        "observed": observed, "events": events,
        "events_well_formed": wf,
        "posture_event_coupling": n_posture == len(monitor.trips),
        "candidate_status_after": monitor.candidate_status,
        "cells_reverted": sorted(monitor.cell_posture),
        "pass": bool(ok and wf), "detail": detail or {}})
    return ok and wf


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_before = ct.frozen_clean()
    att_ok, block_sha = ct.attest()
    if not att_ok:
        print("BLOCKED: policy-block attestation failed", file=sys.stderr)
        return 2
    log = EventLog(LOG_FILE)
    log_seq_at_start = log.seq

    # ---------- S0 baseline: healthy cell, no tripwire, no event
    pkt, aud = mk_cell("h1", n_eligible=3, n_contra=0, n_ambig=5)  # margin -5
    m = Monitor(log, "S0-baseline-no-trip")
    ex = m.exercise("h1", pkt, aud)
    record_scenario(
        "S0-baseline-no-trip", "(none)",
        "healthy open cell outside the corridor: witness_alt served, "
        "zero tripwire events, candidate active",
        m, f"served {ex['n_witness_alt_served']} witness_alt, "
           f"{len(m.trips)} events, status {m.candidate_status}",
        ex["n_witness_alt_served"] == 3 and not m.trips
        and m.candidate_status == "active",
        {"margin": ex["margin"]})

    # ---------- S1 T6 corridor: open-margin cell inside the corridor
    pkt, aud = mk_cell("c1", n_eligible=2, n_contra=1, n_ambig=2)  # margin -1
    m = Monitor(log, "S1-T6-corridor-fail-closed")
    ex = m.exercise("c1", pkt, aud)
    record_scenario(
        "S1-T6-corridor-fail-closed", "T6",
        "guard margin -1 inside (-4, +23): cell reverted BEFORE serving "
        "-> zero witness_alt served (2 were decided), event recorded",
        m, f"margin {ex['margin']}, decided "
           f"{ex['n_witness_alt_decided']}, served "
           f"{ex['n_witness_alt_served']}, events "
           f"{[e['tripwire'] for e in m.trips]}",
        ex["corridor"] and ex["n_witness_alt_decided"] == 2
        and ex["n_witness_alt_served"] == 0
        and [e["tripwire"] for e in m.trips] == ["T6"]
        and m.cell_posture.get("c1") == "reverted"
        and m.candidate_status == "active")

    # ---------- S2 T6 boundaries: the corridor is an OPEN interval
    boundary = []
    for name, n_c, n_a, expect_trip, note in (
            ("b-m4", 1, 5, False, "margin -4 = observed open boundary"),
            ("b-p23", 24, 1, False, "margin +23 = observed closed boundary"),
            ("b-m5", 0, 5, False, "margin -5 outside corridor"),
            ("b-p24", 25, 1, False, "margin +24 outside corridor"),
            ("b-zero", 0, 0, True, "margin 0: vacuous clean cell IS inside "
             "the registered corridor -> fail-closed (tighter than the "
             "evidence requires, zero utility lost: F1b never acts on "
             "clean cells)")):
        pkt, aud = mk_cell(name, n_eligible=1, n_contra=n_c, n_ambig=n_a)
        m2 = Monitor(log, f"S2-T6-boundary-{name}")
        ex2 = m2.exercise(name, pkt, aud)
        tripped = any(e["tripwire"] == "T6" for e in m2.trips)
        ok = tripped == expect_trip and all(
            event_well_formed(e) for e in m2.trips)
        boundary.append({"cell": name, "margin": ex2["margin"],
                         "expected_trip": expect_trip,
                         "observed_trip": tripped, "note": note,
                         "pass": ok})
    SCENARIOS.append({
        "scenario": "S2-T6-corridor-open-interval-boundaries",
        "tripwire": "T6",
        "expected": "margins -4/+23/-5/+24 do NOT trip (open interval, "
                    "committed observations at the ends); margin 0 DOES",
        "observed": boundary,
        "events_well_formed": True,
        "posture_event_coupling": True,
        "candidate_status_after": "per-sub-scenario",
        "cells_reverted": [b["cell"] for b in boundary
                           if b["observed_trip"]],
        "pass": all(b["pass"] for b in boundary), "detail": {}})

    # ---------- S3 T7: malformed row -> fail-closed event -> reversion
    pkt, aud = mk_cell("f1", n_eligible=2, n_contra=0, n_ambig=5)
    pkt = pkt + ["{malformed row"]
    m = Monitor(log, "S3-T7-fail-closed-event")
    ex = m.exercise("f1", pkt, aud)
    record_scenario(
        "S3-T7-fail-closed-event", "T7",
        "a fail-closed (schema) event on the cell trips T7: cell "
        "reverted, eligible rows' witness_alt suppressed, event recorded",
        m, f"decided {ex['n_witness_alt_decided']}, served "
           f"{ex['n_witness_alt_served']}, events "
           f"{[e['tripwire'] for e in m.trips]}",
        ex["n_witness_alt_decided"] == 2
        and ex["n_witness_alt_served"] == 0
        and [e["tripwire"] for e in m.trips] == ["T7"]
        and m.cell_posture.get("f1") == "reverted")

    # ---------- S4 T1: envelope-multiset drift -> candidate suspension
    pkt, aud = mk_cell("v1", n_eligible=2, n_contra=0, n_ambig=5)
    m0 = Monitor(log, "S4-T1-envelope-fixture")
    clean = m0.exercise("v1", pkt, aud)          # no envelope arg: no T1
    fixture = sorted((r["query_id"], str(r["witness_alt_class"]))
                     for r in clean["records"]
                     if r["served_outcome"] == "witness_alt")
    m = Monitor(log, "S4-T1-envelope-exactness")
    ex_ok = m.exercise("v1", pkt, aud, envelope=fixture)
    pre_trips = len(m.trips)
    ex_bad = m.exercise("v1", pkt, aud, envelope=fixture,
                        inject_envelope_mutation=True)
    pkt_h, aud_h = mk_cell("h2", n_eligible=2, n_contra=0, n_ambig=5)
    ex_other = m.exercise("h2", pkt_h, aud_h)    # healthy cell, but…
    ex_again = m.exercise("v1", pkt, aud, envelope=fixture)
    record_scenario(
        "S4-T1-envelope-exactness", "T1",
        "exact multiset -> no trip; one mutated class byte -> T1, "
        "candidate-wide withdrawn-pending-review; a HEALTHY other cell "
        "then serves zero witness_alt; no self-reinstatement on a later "
        "clean exercise",
        m, f"clean pass trips={pre_trips}; after mutation status="
           f"{m.candidate_status}; healthy cell served "
           f"{ex_other['n_witness_alt_served']}; clean re-exercise served "
           f"{ex_again['n_witness_alt_served']}",
        pre_trips == 0
        and [e["tripwire"] for e in m.trips] == ["T1"]
        and m.candidate_status == "withdrawn-pending-review"
        and ex_bad["n_witness_alt_served"] == 0
        and ex_other["n_witness_alt_decided"] == 2
        and ex_other["n_witness_alt_served"] == 0
        and ex_again["n_witness_alt_served"] == 0,
        {"fixture_multiset": fixture})

    # ---------- S5 T2: composition exactness (both arms) -> suspension
    pkt, aud = mk_cell("x1", n_eligible=2, n_contra=0, n_ambig=5)
    m = Monitor(log, "S5a-T2-tier-violation")
    m.exercise("x1", pkt, aud, inject_tier_violation=True)
    ok_a = record_scenario(
        "S5a-T2-tier-violation", "T2",
        "a witness_alt record at a tier other than harness-heuristic "
        "-> T2 -> candidate-wide suspension",
        m, f"events {[e['tripwire'] for e in m.trips]}, status "
           f"{m.candidate_status}",
        [e["tripwire"] for e in m.trips] == ["T2"]
        and m.candidate_status == "withdrawn-pending-review")
    dup_pkt = [ct.mk_row("x2:q0", [ct.mk_abstention()]),
               ct.mk_row("x2:q0", [ct.mk_tie()])]
    dup_aud = ct.mk_audit(["x2:q0"], n_contra=0,
                          ambigs=((2, 128, True), (201, 301, False),
                                  (202, 302, False), (203, 303, False),
                                  (204, 304, False)))
    m = Monitor(log, "S5b-T2-i2-overlap-row")
    ex = m.exercise("x2", dup_pkt, dup_aud)
    record_scenario(
        "S5b-T2-i2-overlap-row", "T2 (+T7)",
        "an I2 overlap row -> T2 candidate-wide suspension (the row "
        "itself was already fail-closed to defer, which also trips T7)",
        m, f"events {[e['tripwire'] for e in m.trips]}, status "
           f"{m.candidate_status}, served witness_alt "
           f"{ex['n_witness_alt_served']}",
        set(e["tripwire"] for e in m.trips) == {"T2", "T7"}
        and m.candidate_status == "withdrawn-pending-review"
        and ex["n_witness_alt_served"] == 0)

    # ---------- S6 T3 single cell (precision kept >= floor): reversion
    # cell A: 20 acts, 17 correct / 3 wrong of 25 in-scope -> rate 0.12,
    # precision 0.85; cell B healthy and large enough (40 in-scope, 0
    # wrong) that the engagement aggregate 3/65 ~ 0.046 stays under the
    # 0.05 ceiling -> T3 isolated; reversion is cell-scoped
    pkt_a, aud_a = mk_cell("t3a", n_eligible=20, n_plain_ties=5,
                           n_contra=0, n_ambig=5)
    pkt_b, aud_b = mk_cell("t3b", n_eligible=2, n_plain_ties=38,
                           n_contra=0, n_ambig=5)
    m = Monitor(log, "S6-T3-per-cell")
    ex_a = m.exercise("t3a", pkt_a, aud_a)
    ex_b = m.exercise("t3b", pkt_b, aud_b)
    tj = m.truth_join({
        "t3a": (ex_a["records"], truth_map(ex_a["records"], 17), 25),
        "t3b": (ex_b["records"], truth_map(ex_b["records"], 2), 40)})
    after_a = m._apply_posture("t3a", ex_a["records"])
    after_b = m._apply_posture("t3b", ex_b["records"])
    record_scenario(
        "S6-T3-per-cell-reversion", "T3",
        "wrong-mass rate 0.12 > 0.10 with precision 0.85 >= 0.75: T3 "
        "only, cell t3a reverted; healthy cell t3b untouched; candidate "
        "stays active (single cell)",
        m, f"events {[e['tripwire'] for e in m.trips]}, reverted "
           f"{sorted(m.cell_posture)}, status {m.candidate_status}",
        [e["tripwire"] for e in m.trips] == ["T3"]
        and m.cell_posture.get("t3a") == "reverted"
        and "t3b" not in m.cell_posture
        and m.candidate_status == "active"
        and all(r["served_outcome"] != "witness_alt" for r in after_a)
        and sum(1 for r in after_b
                if r["served_outcome"] == "witness_alt") == 2,
        {"aggregate_rate": tj["aggregate_rate"]})

    # ---------- S7 T4 single cell: precision < floor, wrong rate small
    # 2 acts, 1 wrong of 40 in-scope -> rate 0.025, precision 0.5
    pkt, aud = mk_cell("t4a", n_eligible=2, n_plain_ties=38,
                       n_contra=0, n_ambig=5)
    m = Monitor(log, "S7-T4-precision")
    ex = m.exercise("t4a", pkt, aud)
    m.truth_join({"t4a": (ex["records"], truth_map(ex["records"], 1), 40)})
    record_scenario(
        "S7-T4-precision-reversion", "T4",
        "acting precision 0.5 < 0.75 on an acted cell (wrong-mass rate "
        "0.025 stays under every ceiling): T4 only, cell reversion",
        m, f"events {[e['tripwire'] for e in m.trips]}, status "
           f"{m.candidate_status}",
        [e["tripwire"] for e in m.trips] == ["T4"]
        and m.cell_posture.get("t4a") == "reverted"
        and m.candidate_status == "active")

    # ---------- S8 T3/T4 on >= 2 cells of one engagement: suspension
    # m1/m2 each 3 wrong of 25 in-scope (rate 0.12); a third healthy
    # cell m3 (70 in-scope, 0 wrong) keeps the aggregate at 6/120 = 0.05
    # exactly, under the strict > ceiling -> the suspension is the
    # registered two-cell escalation, not a T5 co-trip
    m = Monitor(log, "S8-T3-multi-cell-escalation")
    cells = {}
    for cname in ("m1", "m2"):
        p, a = mk_cell(cname, n_eligible=20, n_plain_ties=5,
                       n_contra=0, n_ambig=5)
        exm = m.exercise(cname, p, a)
        cells[cname] = (exm["records"], truth_map(exm["records"], 17), 25)
    p, a = mk_cell("m3", n_eligible=20, n_plain_ties=50,
                   n_contra=0, n_ambig=5)
    exm = m.exercise("m3", p, a)
    cells["m3"] = (exm["records"], truth_map(exm["records"], 20), 70)
    m.truth_join(cells)
    record_scenario(
        "S8-T3-multi-cell-escalation", "T3 (escalation)",
        "T3 on two cells of one engagement: per-cell reversions PLUS the "
        "registered candidate-wide suspension",
        m, f"events {[e['tripwire'] for e in m.trips]}, status "
           f"{m.candidate_status}",
        [e["tripwire"] for e in m.trips] == ["T3", "T3", "T3"]
        and m.trips[-1]["scope"] == "candidate"
        and m.candidate_status == "withdrawn-pending-review")

    # ---------- S9 T5 aggregate: per-cell rates pass, aggregate fails
    # two cells: 40 acts each, 3 wrong of 50 in-scope -> per-cell 0.06
    # (< 0.10, precision 0.925), aggregate 6/100 = 0.06 > 0.05 -> T5
    m = Monitor(log, "S9-T5-aggregate")
    cells = {}
    for cname in ("g1", "g2"):
        p, a = mk_cell(cname, n_eligible=40, n_plain_ties=10,
                       n_contra=0, n_ambig=5)
        exg = m.exercise(cname, p, a)
        cells[cname] = (exg["records"], truth_map(exg["records"], 37), 50)
    tj = m.truth_join(cells)
    record_scenario(
        "S9-T5-aggregate-suspension", "T5",
        "per-cell wrong-mass 0.06 under the 0.10 per-cell ceiling on "
        "both cells, aggregate 0.06 > 0.05: T5 -> candidate-wide "
        "suspension",
        m, f"aggregate {tj['aggregate_rate']}, events "
           f"{[e['tripwire'] for e in m.trips]}, status "
           f"{m.candidate_status}",
        [e["tripwire"] for e in m.trips] == ["T5"]
        and m.candidate_status == "withdrawn-pending-review",
        {"aggregate_rate": tj["aggregate_rate"]})

    # ---------- S10 boundary honesty: registered constants are strict
    m = Monitor(log, "S10-strict-inequality-boundaries")
    outs = []
    # T3 boundary: 3 wrong of 30 in-scope = 0.10 exactly -> no trip
    # (a healthy 30-in-scope pad cell holds the aggregate at 0.05 exactly)
    p, a = mk_cell("bd1", n_eligible=20, n_plain_ties=10,
                   n_contra=0, n_ambig=5)
    exb = m.exercise("bd1", p, a)
    p2, a2 = mk_cell("bd1p", n_eligible=20, n_plain_ties=10,
                     n_contra=0, n_ambig=5)
    exb2 = m.exercise("bd1p", p2, a2)
    m.truth_join({"bd1": (exb["records"], truth_map(exb["records"], 17),
                          30),
                  "bd1p": (exb2["records"], truth_map(exb2["records"], 20),
                           30)})
    outs.append(("T3 at exactly 0.10 (aggregate held at exactly 0.05)",
                 len(m.trips) == 0))
    # T4 boundary: 4 acts, 1 wrong -> precision 0.75 exactly -> no trip
    p, a = mk_cell("bd2", n_eligible=4, n_plain_ties=36,
                   n_contra=0, n_ambig=5)
    exb = m.exercise("bd2", p, a)
    m.truth_join({"bd2": (exb["records"], truth_map(exb["records"], 3),
                          40)})
    outs.append(("T4 at exactly 0.75", len(m.trips) == 0))
    # T5 boundary: aggregate 5 wrong of 100 = 0.05 exactly -> no trip
    cells = {}
    for cname, n_corr in (("bd3", 38), ("bd4", 37)):
        p, a = mk_cell(cname, n_eligible=40, n_plain_ties=10,
                       n_contra=0, n_ambig=5)
        exb = m.exercise(cname, p, a)
        cells[cname] = (exb["records"], truth_map(exb["records"], n_corr),
                        50)
    m.truth_join(cells)
    outs.append(("T5 at exactly 0.05 (per-cell 0.04/0.06 both < 0.10)",
                 len(m.trips) == 0))
    SCENARIOS.append({
        "scenario": "S10-strict-inequality-boundaries",
        "tripwire": "T3/T4/T5",
        "expected": "the registered constants are strict (> / <): exact "
                    "boundary values do not trip",
        "observed": [{"case": c, "no_trip": ok} for c, ok in outs],
        "events_well_formed": True, "posture_event_coupling": True,
        "candidate_status_after": m.candidate_status,
        "cells_reverted": sorted(m.cell_posture),
        "pass": all(ok for _, ok in outs) and not m.trips, "detail": {}})

    # ---------- log-wide checks
    chain_ok, n_events = EventLog.verify_chain(LOG_FILE)
    events_this_run = n_events - log_seq_at_start
    total_trips = sum(len(s.get("events", [])) for s in SCENARIOS
                      if isinstance(s.get("events"), list))
    coupling_ok = all(s["posture_event_coupling"] for s in SCENARIOS)
    clean_after = ct.frozen_clean()

    all_pass = all(s["pass"] for s in SCENARIOS)
    report = {
        "artifact": "PR-12.9 §5.2 withdrawal-mechanics demo (Stage I)",
        "demo_version": DEMO_VERSION,
        "registration": "PR12_9_READER_CONTRACT_CERTIFICATION.md "
                        "(main fe6248e)",
        "monitoring_doc": "PR12_8_MONITORING_WITHDRAWAL.md",
        "contract_id": ct.CONTRACT_ID,
        "contract_version": ct.CONTRACT_VERSION,
        "policy_block_sha256": block_sha,
        "registered_constants": {
            "T3_per_cell_wrong_ceiling": T3_PER_CELL_WRONG_CEILING,
            "T4_precision_floor": T4_PRECISION_FLOOR,
            "T5_aggregate_wrong_ceiling": T5_AGGREGATE_WRONG_CEILING,
            "T6_corridor_open_interval": list(CORRIDOR)},
        "synthetic_only": "every cell, packet, truth map, and injection "
                          "is fabricated; no committed packet or truth "
                          "label is consumed",
        "scenarios": SCENARIOS,
        "tripwires_demonstrated": sorted(
            {e["tripwire"] for s in SCENARIOS
             for e in (s.get("events") or [])
             if isinstance(e, dict)}),
        "event_log": {
            "path": str(LOG_FILE.relative_to(REPO)),
            "hash_chain_verified": chain_ok,
            "append_only": "events accumulate across runs by design; "
                           "excluded from cross-run byte-identity "
                           "(PR-12.9 §9)",
            "coupling": "every posture change has exactly one recorded "
                        "event and vice versa",
            "coupling_verified": coupling_ok},
        "frozen_surfaces_clean_before": clean_before,
        "frozen_surfaces_clean_after": clean_after,
        "result": "all-scenarios-pass" if all_pass else "FAIL"}
    REPORT_FILE.write_text(json.dumps(report, indent=1, sort_keys=True)
                           + "\n")

    print(f"DEMO: {report['result']}")
    print(f"  scenarios={len(SCENARIOS)} "
          f"tripwires={report['tripwires_demonstrated']} "
          f"events_this_run={events_this_run} chain_ok={chain_ok}")
    print("Scope (PR-12.9 §11): Stage I withdrawal-mechanics evidence "
          "only, on synthetic tripwires — nothing live, nothing served, "
          "nothing certified; no FAM-core change; PR-10 merge-abstain "
          "remains the only certified reader contract; posture remains "
          "deferral.")
    return 0 if all_pass and chain_ok and coupling_ok else 1


if __name__ == "__main__":
    sys.exit(main())
