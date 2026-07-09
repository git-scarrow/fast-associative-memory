#!/usr/bin/env python3
"""PR-12.9 Stage I — hermetic conformance suite for the S1 contract definition.

Artifact §5.1 of `PR12_9_READER_CONTRACT_CERTIFICATION.md` (main
`fe6248e`): a hermetic, stdlib-only test suite for the contract
definition `PR12_8_S1_CONTRACT_CANDIDATE.md` (contract_id
``s1-witness-alt-batch``, version ``0.1-candidate``). Registered
coverage: the six §4 eligibility conditions exercised positively and
negatively on synthetic packets; precedence I1 (an abstention row is
never served ``witness_alt`` even if crafted to satisfy eligibility);
I2 fail-closed on overlap; I4 tier invariance; first-tie-item
semantics; W2-tree-only refusal; malformed-packet fail-closed (the T7
event feed); and determinism (double pass byte-identity).

Two subjects are exercised against every clause the contract text
determines:

* **committed** — ``read_cell`` of the merged reference reader
  (`harness/witness_alt_reference_reader.py`), the implementation that
  froze envelope v0.2 and that certification gate C-1 will re-run;
* **conformant** — a suite-internal batch pipeline implementing the
  contract §§2-7 text exactly, built on the SAME sha-attested frozen
  policy block (RowObs/CellCtx/pol_f1b imported from the committed
  reader after attestation against `action_boundary_score.py`). It
  exists because the committed batch proof tool has no seam for shape
  parameters, per-row malformed-input handling, or fail-closed event
  emission — behaviors PR-12.9 §5.1 explicitly orders tested. A
  divergence between subjects is a recorded finding, never hidden.

Hermetic: synthetic packets only — no committed packet is read,
modified, or regenerated; no truth label, registry CSV, or scoring
output is consumed. Stdlib + subprocess-git only (git is used solely
to verify that consumed source/document files match HEAD and that
frozen surfaces stay clean). Writes ONLY
``results/issue_failure_mode_blindness/pr12_9/conformance_results.json``.

**This program certifies nothing and serves nothing to anyone.** It is
Stage I evidence for a separately-authorized future Stage II
certification run. No deployment, live acting, prompting use,
promotion, memory ingestion, FAM-core change, autonomous downstream
use, or reader-contract change. PR-10 merge-abstain remains the only
certified reader contract; the operational posture on witness-window
rows remains deferral.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
REPO = HARNESS_DIR.parent
sys.path.insert(0, str(HARNESS_DIR))

import witness_alt_reference_reader as wr  # noqa: E402  (committed reader)

BASE = Path("results/issue_failure_mode_blindness")
OUT_DIR = REPO / BASE / "pr12_9"
OUT_FILE = OUT_DIR / "conformance_results.json"

SUITE_VERSION = "pr12_9-stage1-conformance-v1"
CONTRACT_ID = wr.CONTRACT_ID                  # "s1-witness-alt-batch"
CONTRACT_VERSION = wr.CONTRACT_VERSION        # "0.1-candidate"

# PR-12.9 §2: the policy block must attest to exactly this sha.
REGISTERED_BLOCK_SHA = \
    "2f009cf29ce64dc21e7bd392ceac4bbc189bc5b197489c2edaf8cd3ed1022c15"

# Files whose working-tree bytes are consumed (attestation sources and
# governing documents); each is verified byte-identical to HEAD.
CONSUMED_INPUTS = [
    "harness/action_boundary_score.py",
    "harness/witness_alt_reference_reader.py",
    str(BASE / "PR12_8_S1_CONTRACT_CANDIDATE.md"),
    str(BASE / "PR12_8_MONITORING_WITHDRAWAL.md"),
    str(BASE / "pr12_8" / "f1b_envelope.json"),
]
# The governing PR-12.9 registration memo is deliberately NOT in this
# manifest: its §12 results section records THIS suite's output hashes,
# so hashing the memo here would make the committed results
# irreproducible from the committed tree (self-reference). The memo's
# integrity is enforced by ordinary git history, not by this manifest.

# Frozen surfaces that must be git-clean before and after (PR-12.9
# §8.6): the reference reader's own list plus the reader and envelope.
FROZEN_SURFACES = [
    str(BASE / "pr12_3"), str(BASE / "pr12_4"), str(BASE / "pr12_6"),
    str(BASE / "pr12_7"), str(BASE / "pr12_7_holdout_cache"),
    str(BASE / "pr12_8_panel_cache"), str(BASE / "pr10"),
    str(BASE / "pr12_8"),
    "harness/harness_boundary_sim.py", "harness/action_boundary_score.py",
    "harness/reader_utility_score.py",
    "harness/action_boundary_holdout_score.py",
    "harness/witness_alt_reference_reader.py", "harness/harness_policy.json",
]

WITNESS = wr.WITNESS_BASIS   # "witness co-resident (fork_witness)"
TIE_TEXT = "unresolved tie — two candidates, neither asserted"


# ---------------------------------------------------------------------------
# synthetic packet builders (contract §2 schema, mirrored from the
# committed W2 emitter's shapes; every value below is fabricated)
# ---------------------------------------------------------------------------
def audit_ptr(qid: str) -> str:
    return (f"router(synthetic): {qid} witness-window dual-present "
            f"[synthetic conformance fixture]")


def mk_tie(deployed=(0, 2), alts=((3, WITNESS),), text=TIE_TEXT):
    cands = []
    if deployed is not None:
        cands.append({"decode_class": deployed[0], "slot": deployed[1],
                      "basis": "deployed vote"})
    for cls, basis in alts:
        cands.append({"decode_class": cls, "basis": basis})
    return {"type": "unresolved_tie", "text": text, "candidates": cands,
            "certification_tier": "harness-heuristic"}


def mk_abstention():
    return {"type": "abstention_notice",
            "text": "memory abstained (merge-suspect) — core-certified "
                    "PR-10 merge-abstain contract",
            "certification_tier": "core-certified",
            "provenance": {"contract": "PR-10 merge-abstain",
                           "abstain_reason": "merge_suspect_led"}}


def mk_memory_item(label=1):
    return {"type": "memory_item", "content": {"label": label},
            "provenance": {"top1_slot": 120, "support_slots": [120]},
            "certification_tier": "harness-heuristic", "caveats": []}


def mk_row(qid: str, items) -> str:
    return json.dumps({"query_id": qid, "items": items})


def mk_audit(qids, n_contra=0, ambigs=((2, 128, True),)):
    """Audit packet lines: n_contra contradiction reviews, one ambiguous
    review per (incumbent_slot, owner_slot, never_resolving) triple, and
    one per-row decision record (with evidence_ptr) per query id."""
    lines = []
    for i in range(n_contra):
        lines.append(json.dumps({
            "record_type": "contradiction_pair_review",
            "pair": {"incumbent_slot": 900 + i, "owner_slot": 950 + i,
                     "onset_epoch": 0},
            "quarantine_led_rows_total": 0,
            "certification_tier": "harness-heuristic"}))
    for inc, own, nr in ambigs:
        lines.append(json.dumps({
            "record_type": "ambiguous_pair_review",
            "pair": {"incumbent_slot": inc, "owner_slot": own,
                     "onset_epoch": 6},
            "never_resolving": nr,
            "certification_tier": "harness-heuristic"}))
    for qid in qids:
        lines.append(json.dumps({
            "query_id": qid,
            "decisions": [{"query_id": qid,
                           "item_id": f"served_answer@{qid}",
                           "state": "human-review",
                           "disposition": "shown_with_caveat",
                           "reason_code": "led_pending_ambiguous",
                           "evidence_ptr": audit_ptr(qid),
                           "certification_tier": "harness-heuristic",
                           "policy_version": "synthetic-fixture",
                           "authorization": None}]}))
    return lines


# ---------------------------------------------------------------------------
# subject 1 — the committed reference reader's read_cell
# ---------------------------------------------------------------------------
def serve_committed(pkt_lines, aud_lines, cell_key="syn"):
    anomalies = []
    try:
        ctx, records, abstain_qids, eligible_qids = wr.read_cell(
            "\n".join(pkt_lines).encode(), "\n".join(aud_lines).encode(),
            anomalies, cell_key)
        return {"ok": True, "records": records, "anomalies": anomalies,
                "abstain_qids": sorted(abstain_qids),
                "eligible_qids": sorted(eligible_qids),
                "i2_overlap": sorted(set(abstain_qids) & set(eligible_qids))}
    except Exception as ex:                       # crash = observed behavior
        return {"ok": False, "exception": f"{type(ex).__name__}: {ex}",
                "records": None, "anomalies": anomalies}


# ---------------------------------------------------------------------------
# subject 2 — suite-internal conformant pipeline (contract §§2-7 text,
# implemented exactly; policy decisions via the attested frozen block)
# ---------------------------------------------------------------------------
def blank_record(qid):
    return {"query_id": qid, "served_outcome": "defer", "abstain_reason": "",
            "witness_alt_class": "", "witness_alt_basis": "",
            "policy_id": "", "policy_block_sha256": "", "evidence_ptr": "",
            "certification_tier": "harness-heuristic",
            "contract_id": CONTRACT_ID, "contract_version": CONTRACT_VERSION}


def serve_conformant(pkt_lines, aud_lines, shape="W2", cell_key="syn"):
    """One served-decision record per packet line (contract §3), with the
    §7 fail-closed envelope: any schema failure, I2 overlap, or
    eligibility ambiguity -> the affected row is served defer (never
    witness_alt) and the anomaly is recorded as a fail-closed event (the
    monitoring layer's T7 feed). Batch semantics: the entire audit
    packet is ingested before any row is decided (contract §2)."""
    fail_closed_events, anomalies = [], []

    # -- batch ingest of the audit packet (contract §2)
    aud_records = []
    try:
        for line in aud_lines:
            aud_records.append(json.loads(line))
    except (json.JSONDecodeError, TypeError) as ex:
        # corrupt audit packet poisons the cell context: every row of the
        # cell is affected -> all-defer, one cell-level event (§7)
        fail_closed_events.append({
            "cell": cell_key, "query_id": "", "clause": "schema",
            "anomaly": f"audit packet malformed: {type(ex).__name__}"})
        records = []
        for i, line in enumerate(pkt_lines, 1):
            try:
                qid = json.loads(line)["query_id"]
            except Exception:
                qid = f"__fail_closed_line_{i}__"
            records.append(blank_record(qid))
        return {"ok": True, "records": records, "anomalies": anomalies,
                "fail_closed_events": fail_closed_events,
                "abstain_qids": [], "eligible_qids": [], "i2_overlap": [],
                "ctx": None}
    ctx = wr.CellCtx(aud_records)
    evidence = {}                       # qid -> audit decision evidence_ptr
    for rec in aud_records:
        decs = rec.get("decisions") or []
        if rec.get("query_id") and decs and "evidence_ptr" in decs[0]:
            evidence.setdefault(rec["query_id"], decs[0]["evidence_ptr"])

    # -- per-row decisions (contract §§3-5; §6 I1 precedence)
    records, abstain_qids, eligible_qids = [], set(), set()
    for i, line in enumerate(pkt_lines, 1):
        try:
            rec = json.loads(line)
            if not isinstance(rec, dict) or \
                    not isinstance(rec.get("query_id"), str):
                raise ValueError("row lacks a string query_id")
            qid = rec["query_id"]
            items = rec.get("items", [])
            if not isinstance(items, list):
                raise ValueError("items is not a list")
        except Exception as ex:                       # §7 fail-closed row
            fail_closed_events.append({
                "cell": cell_key, "query_id": f"__fail_closed_line_{i}__",
                "clause": "schema",
                "anomaly": f"packet line {i} malformed: "
                           f"{type(ex).__name__}: {ex}"})
            records.append(blank_record(f"__fail_closed_line_{i}__"))
            continue
        out = blank_record(qid)
        abst = [it for it in items if isinstance(it, dict)
                and it.get("type") == "abstention_notice"]
        ties = [it for it in items if isinstance(it, dict)
                and it.get("type") == "unresolved_tie"]
        mems = [it for it in items if isinstance(it, dict)
                and it.get("type") == "memory_item"]
        if abst:                                    # §6 I1: abstain first
            out["served_outcome"] = "abstain"
            out["abstain_reason"] = "merge_suspect_led"
            out["certification_tier"] = "core-certified"
            abstain_qids.add(qid)
            if ties:
                anomalies.append({"cell": cell_key, "query_id": qid,
                                  "anomaly": "abstention row carries an "
                                             "unresolved_tie item"})
        elif ties:
            try:                                    # §4.1: first tie item
                obs = wr.RowObs(shape, ties[0])
                act = wr.pol_f1b(obs, ctx)
                if act is not None and \
                        not set(act) <= set(obs.presented):
                    raise ValueError("ACT outside presented set")
            except Exception as ex:                 # §7 eligibility/schema
                fail_closed_events.append({
                    "cell": cell_key, "query_id": qid, "clause":
                    "eligibility", "anomaly": f"{type(ex).__name__}: {ex}"})
                records.append(out)                 # defer, never witness_alt
                continue
            if act is not None:
                out["served_outcome"] = "witness_alt"
                out["witness_alt_class"] = sorted(act)[0]
                out["witness_alt_basis"] = WITNESS
                out["policy_id"] = "W2:F1b"
                out["policy_block_sha256"] = REGISTERED_BLOCK_SHA
                # contract §3: the row's audit-packet decision
                # evidence_ptr, carried verbatim
                out["evidence_ptr"] = evidence.get(qid, "")
                eligible_qids.add(qid)
        elif mems:                                  # §5: packet answers
            out["served_outcome"] = "answer"
        records.append(out)

    # -- §6 I2 disjointness, fail-closed at row granularity (§7): a
    # query id served abstain anywhere may not also be served
    # witness_alt; the affected candidate-side row reverts to defer.
    overlap = sorted(abstain_qids & eligible_qids)
    if overlap:
        for out in records:
            if out["query_id"] in overlap \
                    and out["served_outcome"] == "witness_alt":
                fail_closed_events.append({
                    "cell": cell_key, "query_id": out["query_id"],
                    "clause": "i2-overlap",
                    "anomaly": "eligible row overlaps the certified "
                               "abstention set"})
                records[records.index(out)] = blank_record(out["query_id"])
        eligible_qids -= set(overlap)
    return {"ok": True, "records": records, "anomalies": anomalies,
            "fail_closed_events": fail_closed_events,
            "abstain_qids": sorted(abstain_qids),
            "eligible_qids": sorted(eligible_qids),
            "i2_overlap": overlap, "ctx": ctx}


# ---------------------------------------------------------------------------
# harness plumbing
# ---------------------------------------------------------------------------
def attest():
    """Contract §1 / PR-12.9 §2: the policy block in the committed reader
    must equal the frozen scorer's block byte-for-byte and hash to the
    registered sha. Sets wr.BLOCK_SHA exactly as the reader's main()."""
    mine = wr.extract_policy_block(
        (REPO / "harness" / "witness_alt_reference_reader.py").read_text())
    frozen = wr.extract_policy_block(
        (REPO / "harness" / "action_boundary_score.py").read_text())
    sha = hashlib.sha256(mine.encode()).hexdigest()
    ok = mine == frozen and sha == REGISTERED_BLOCK_SHA
    wr.BLOCK_SHA = sha
    return ok, sha


def verify_consumed_inputs():
    """Every consumed file must match its HEAD blob (no drift under us)."""
    manifest, ok = {}, True
    for rel in CONSUMED_INPUTS:
        data = (REPO / rel).read_bytes()
        head = subprocess.run(["git", "cat-file", "blob", f"HEAD:{rel}"],
                              cwd=REPO, capture_output=True)
        match = head.returncode == 0 and head.stdout == data
        manifest[rel] = {"sha256": hashlib.sha256(data).hexdigest(),
                         "matches_head": match}
        ok = ok and match
    return ok, manifest


def frozen_clean():
    r = subprocess.run(["git", "status", "--porcelain", "--",
                        *FROZEN_SURFACES], cwd=REPO,
                       capture_output=True, check=True)
    lines = [ln for ln in r.stdout.decode().splitlines()
             if not ln.endswith(".DS_Store")]
    return not lines


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------
CHECKS = []
ALL_RECORDS = []          # every record both subjects emit, for I4/§3 lint


def add_check(name, clause, expectation, committed, conformant, detail):
    CHECKS.append({"name": name, "clause": clause,
                   "expectation": expectation, "committed": committed,
                   "conformant": conformant, "detail": detail})


def collect(result):
    if result.get("records"):
        ALL_RECORDS.extend(result["records"])
    return result


def outcome_of(result, qid):
    if not result.get("records"):
        return None
    for r in result["records"]:
        if r["query_id"] == qid:
            return r
    return None


def verdict(cond):
    return "pass" if cond else "fail"


BASELINE_PKT = [mk_row("syn:q1", [mk_tie()]),
                mk_row("syn:q2", [mk_memory_item()]),
                mk_row("syn:q3", [])]
BASELINE_AUD = mk_audit(["syn:q1", "syn:q2", "syn:q3"])


def both(pkt, aud, key):
    return (collect(serve_committed(pkt, aud, key)),
            collect(serve_conformant(pkt, aud, "W2", key)))


def run_checks():
    # ---- baseline: the one eligible row class (contract §4 all-pass)
    cm, cf = both(BASELINE_PKT, BASELINE_AUD, "syn-baseline")
    for label, res in (("committed", cm), ("conformant", cf)):
        r = outcome_of(res, "syn:q1")
        fields_ok = (r is not None and r["served_outcome"] == "witness_alt"
                     and r["witness_alt_class"] == 3
                     and r["witness_alt_basis"] == WITNESS
                     and r["policy_id"] == "W2:F1b"
                     and r["policy_block_sha256"] == REGISTERED_BLOCK_SHA
                     and r["certification_tier"] == "harness-heuristic"
                     and r["contract_id"] == CONTRACT_ID
                     and r["contract_version"] == CONTRACT_VERSION
                     and r["witness_alt_class"] in (0, 3))
        if label == "committed":
            c_ok = fields_ok
        else:
            f_ok = fields_ok
    add_check(
        "baseline_eligible_served_witness_alt", "contract §4 (all six) + §3",
        "eligible W2 row -> witness_alt, class = sole alternative (in the "
        "presented set), all §3 constants stamped",
        verdict(c_ok), verdict(f_ok),
        {"row": "syn:q1", "committed": outcome_of(cm, "syn:q1"),
         "conformant": outcome_of(cf, "syn:q1")})
    add_check(
        "baseline_answer_and_defer_mapping", "contract §5",
        "memory_item row -> answer; empty row -> defer (both subjects)",
        verdict(outcome_of(cm, "syn:q2")["served_outcome"] == "answer"
                and outcome_of(cm, "syn:q3")["served_outcome"] == "defer"),
        verdict(outcome_of(cf, "syn:q2")["served_outcome"] == "answer"
                and outcome_of(cf, "syn:q3")["served_outcome"] == "defer"),
        {"q2_committed": outcome_of(cm, "syn:q2")["served_outcome"],
         "q3_committed": outcome_of(cm, "syn:q3")["served_outcome"]})

    # ---- finding F1 probe: §3 evidence_ptr sourcing
    exp_ptr = audit_ptr("syn:q1")
    add_check(
        "evidence_ptr_audit_decision_carriage", "contract §3 (evidence_ptr)",
        "witness_alt row carries the row's audit-packet decision "
        "evidence_ptr verbatim",
        verdict(outcome_of(cm, "syn:q1")["evidence_ptr"] == exp_ptr),
        verdict(outcome_of(cf, "syn:q1")["evidence_ptr"] == exp_ptr),
        {"expected": exp_ptr,
         "committed_observed": outcome_of(cm, "syn:q1")["evidence_ptr"],
         "conformant_observed": outcome_of(cf, "syn:q1")["evidence_ptr"],
         "finding": "F1 (committed reader carries the memory-packet tie "
                    "item text instead; identical divergence is present in "
                    "the committed pr12_8/served CSVs)"})

    # ---- §4 conditions, negated one at a time
    negatives = [
        ("cond1_no_tie_item", "§4.1",
         [mk_row("syn:n1", [])], ["syn:n1"], "syn:n1"),
        ("cond2_width_three", "§4.2",
         [mk_row("syn:n2", [mk_tie(alts=((3, WITNESS), (5, WITNESS)))])],
         ["syn:n2"], "syn:n2"),
        ("cond2_width_one", "§4.2",
         [mk_row("syn:n3", [mk_tie(deployed=(0, 2), alts=((0, WITNESS),))])],
         ["syn:n3"], "syn:n3"),
        ("cond3_foreign_basis", "§4.3",
         [mk_row("syn:n4", [mk_tie(alts=((3, "replay echo"),))])],
         ["syn:n4"], "syn:n4"),
        ("cond4_slot_not_never_resolving", "§4.4",
         [mk_row("syn:n5", [mk_tie(deployed=(0, 7))])], ["syn:n5"],
         "syn:n5"),
        ("cond4b_no_deployed_candidate", "§4.4 (overdetermined: also §4.5)",
         [mk_row("syn:n6", [mk_tie(deployed=None,
                                   alts=((3, WITNESS), (5, WITNESS)))])],
         ["syn:n6"], "syn:n6"),
        ("cond5_alt_multiplicity_two", "§4.5",
         [mk_row("syn:n7", [mk_tie(alts=((3, WITNESS), (3, WITNESS)))])],
         ["syn:n7"], "syn:n7"),
    ]
    for name, clause, pkt, qids, qid in negatives:
        cm_n, cf_n = both(pkt, mk_audit(qids), f"syn-{name}")
        add_check(
            name, f"contract {clause} (negative)",
            "condition violated -> row NOT eligible -> defer",
            verdict(outcome_of(cm_n, qid)["served_outcome"] == "defer"),
            verdict(outcome_of(cf_n, qid)["served_outcome"] == "defer"),
            {"row": qid,
             "committed_outcome": outcome_of(cm_n, qid)["served_outcome"],
             "conformant_outcome": outcome_of(cf_n, qid)["served_outcome"]})

    # cond6: quiet-cell guard — closed (2 contra > 1 ambiguous) vs the
    # boundary (1 == 1, guard opens on <=)
    cm_g, cf_g = both([mk_row("syn:g1", [mk_tie()])],
                      mk_audit(["syn:g1"], n_contra=2), "syn-guard-closed")
    add_check(
        "cond6_guard_closed_defers", "contract §4.6 (negative)",
        "n_contradiction_pairs(2) > n_ambiguous_pairs(1) -> guard closed "
        "-> defer",
        verdict(outcome_of(cm_g, "syn:g1")["served_outcome"] == "defer"),
        verdict(outcome_of(cf_g, "syn:g1")["served_outcome"] == "defer"),
        {"margins": "+1 (closed)"})
    cm_b, cf_b = both([mk_row("syn:g2", [mk_tie()])],
                      mk_audit(["syn:g2"], n_contra=1), "syn-guard-boundary")
    add_check(
        "cond6_guard_boundary_equal_opens", "contract §4.6 (positive, <=)",
        "n_contradiction_pairs(1) == n_ambiguous_pairs(1) -> guard open "
        "(<=) -> witness_alt",
        verdict(outcome_of(cm_b, "syn:g2")["served_outcome"]
                == "witness_alt"),
        verdict(outcome_of(cf_b, "syn:g2")["served_outcome"]
                == "witness_alt"),
        {"margins": "0 (open; the guard inequality is registered as <=)"})

    # ---- first-tie-item semantics (§4.1)
    tie_bad = mk_tie(alts=((3, WITNESS), (5, WITNESS)))       # width 3
    cm_t1, cf_t1 = both([mk_row("syn:t1", [mk_tie(), tie_bad])],
                        mk_audit(["syn:t1"]), "syn-firsttie-a")
    add_check(
        "first_tie_eligible_second_ignored", "contract §4.1",
        "first tie eligible, second ineligible -> witness_alt on the "
        "first tie's sole alternative (further ties ignored, not "
        "disqualifying)",
        verdict(outcome_of(cm_t1, "syn:t1")["served_outcome"]
                == "witness_alt"
                and outcome_of(cm_t1, "syn:t1")["witness_alt_class"] == 3),
        verdict(outcome_of(cf_t1, "syn:t1")["served_outcome"]
                == "witness_alt"
                and outcome_of(cf_t1, "syn:t1")["witness_alt_class"] == 3),
        {})
    cm_t2, cf_t2 = both([mk_row("syn:t2", [tie_bad, mk_tie()])],
                        mk_audit(["syn:t2"]), "syn-firsttie-b")
    add_check(
        "first_tie_ineligible_not_rescued", "contract §4.1",
        "first tie ineligible -> defer even though a later tie item "
        "would be eligible (only ties[0] is considered)",
        verdict(outcome_of(cm_t2, "syn:t2")["served_outcome"] == "defer"),
        verdict(outcome_of(cf_t2, "syn:t2")["served_outcome"] == "defer"),
        {})

    # ---- I1 precedence: abstention row crafted to satisfy eligibility
    cm_i1, cf_i1 = both(
        [mk_row("syn:i1", [mk_abstention(), mk_tie()])],
        mk_audit(["syn:i1"]), "syn-i1")
    for label, res in (("committed", cm_i1), ("conformant", cf_i1)):
        r = outcome_of(res, "syn:i1")
        ok = (r["served_outcome"] == "abstain"
              and r["abstain_reason"] == "merge_suspect_led"
              and r["certification_tier"] == "core-certified"
              and r["witness_alt_class"] == ""
              and len(res["anomalies"]) == 1)
        if label == "committed":
            i1_c = ok
        else:
            i1_f = ok
    add_check(
        "i1_abstention_precedence", "contract §6 I1",
        "row carrying the certified abstention notice AND a crafted "
        "eligible tie -> served abstain (core-certified), never "
        "witness_alt; the crafted tie is recorded as an anomaly",
        verdict(i1_c), verdict(i1_f),
        {"committed_record": outcome_of(cm_i1, "syn:i1"),
         "committed_anomalies": cm_i1["anomalies"]})

    # ---- I2 fail-closed on overlap (duplicate query id: one abstained
    # row and one crafted-eligible row share a join key)
    dup_pkt = [mk_row("syn:i2", [mk_abstention()]),
               mk_row("syn:i2", [mk_tie()])]
    cm_i2, cf_i2 = both(dup_pkt, mk_audit(["syn:i2"]), "syn-i2")
    cm_i2_rows = [r["served_outcome"] for r in (cm_i2["records"] or [])]
    cf_i2_rows = [r["served_outcome"] for r in cf_i2["records"]]
    add_check(
        "i2_overlap_fail_closed_row_level", "contract §6 I2 + §7",
        "eligible set overlapping the certified abstention set -> the "
        "affected row is served defer (never witness_alt) and the "
        "anomaly is recorded",
        verdict(bool(cm_i2["records"]) and cm_i2_rows == ["abstain", "defer"]
                and not cm_i2["i2_overlap"]),
        verdict(cf_i2_rows == ["abstain", "defer"]
                and cf_i2["i2_overlap"] == ["syn:i2"]
                and len(cf_i2["fail_closed_events"]) == 1),
        {"committed_outcomes": cm_i2_rows,
         "committed_overlap_detected": cm_i2.get("i2_overlap"),
         "conformant_outcomes": cf_i2_rows,
         "conformant_events": cf_i2["fail_closed_events"],
         "finding": "F2 (committed read_cell leaves the affected row at "
                    "witness_alt; the committed pipeline instead detects "
                    "the overlap in its composition proof and kills the "
                    "whole run — fail-closed by abort, not the registered "
                    "per-row defer record)"})

    # ---- W2-tree-only refusal (contract §2)
    tie = mk_tie()
    ctx = wr.CellCtx([json.loads(ln) for ln in mk_audit(["x"])])
    frozen_refuses = all(
        wr.pol_f1b(wr.RowObs(shape, tie), ctx) is None
        for shape in ("W1", "prototype"))
    cf_w1 = collect(serve_conformant([mk_row("syn:w1", [mk_tie()])],
                                     mk_audit(["syn:w1"]), "W1", "syn-w1"))
    cf_pr = collect(serve_conformant([mk_row("syn:w2", [mk_tie()])],
                                     mk_audit(["syn:w2"]), "prototype",
                                     "syn-proto"))
    add_check(
        "w2_tree_only_refusal", "contract §2",
        "a reader consuming a prototype or W1 tree MUST NOT emit "
        "witness_alt under any condition (frozen policy shape gate; the "
        "committed reader additionally only ever reads W2 trees)",
        verdict(frozen_refuses),
        verdict(outcome_of(cf_w1, "syn:w1")["served_outcome"] == "defer"
                and outcome_of(cf_pr, "syn:w2")["served_outcome"]
                == "defer"),
        {"frozen_pol_f1b_refuses_W1_and_prototype": frozen_refuses,
         "note": "committed subject exercised at the frozen-policy layer "
                 "(pol_f1b shape gate); its read_cell hardcodes W2 and "
                 "its panel consumes only committed /W2/ trees"})

    # ---- malformed-packet fail-closed (§7; the monitoring T7 feed)
    malformed = [
        ("malformed_json_line", [mk_row("syn:m1", [mk_tie()]),
                                 "{this is not json"], ["syn:m1"],
         "syn:m1", 1),
        ("malformed_missing_query_id",
         [json.dumps({"items": [mk_memory_item()]})], [], None, 1),
        ("malformed_tie_without_candidates",
         [mk_row("syn:m3", [{"type": "unresolved_tie",
                             "text": TIE_TEXT}])], ["syn:m3"], "syn:m3", 1),
        ("malformed_audit_packet", [mk_row("syn:m4", [mk_tie()])],
         "AUDIT_BAD", "syn:m4", 1),
    ]
    for name, pkt, qids, keep_qid, n_events in malformed:
        aud = (["{bad audit line"] if qids == "AUDIT_BAD"
               else mk_audit(qids if qids else ["syn:none"]))
        cm_m = collect(serve_committed(pkt, aud, f"syn-{name}"))
        cf_m = collect(serve_conformant(pkt, aud, "W2", f"syn-{name}"))
        no_wa = all(r["served_outcome"] != "witness_alt"
                    for r in cf_m["records"]
                    if keep_qid is None or r["query_id"] != keep_qid) \
            if name != "malformed_audit_packet" else all(
                r["served_outcome"] == "defer" for r in cf_m["records"])
        conformant_ok = (len(cf_m["fail_closed_events"]) == n_events
                         and no_wa
                         and len(cf_m["records"]) == len(pkt))
        committed_ok = cm_m["ok"] and cm_m["records"] is not None and all(
            r["served_outcome"] != "witness_alt" or r["query_id"] == keep_qid
            for r in cm_m["records"]) and len(cm_m["records"]) == len(pkt)
        add_check(
            name, "contract §7 fail-closed + PR-12.9 §5.1 (T7 feed)",
            "the affected row is served defer (never witness_alt), the "
            "anomaly is recorded as a fail-closed event, and unaffected "
            "rows are undisturbed",
            verdict(committed_ok), verdict(conformant_ok),
            {"committed_observed": (cm_m.get("exception")
                                    or "records emitted"),
             "conformant_events": cf_m["fail_closed_events"],
             "conformant_outcomes": [r["served_outcome"]
                                     for r in cf_m["records"]],
             "finding": ("F3 (committed read_cell aborts on malformed "
                         "input: fail-closed by crash — nothing is served, "
                         "never witness_alt, but the registered per-row "
                         "defer record and recorded anomaly are not "
                         "produced)" if not committed_ok else "")})

    # ---- I4 tier invariance + §3 field-domain lint over EVERY record
    # either subject emitted anywhere in this suite
    i4_bad, dom_bad = [], []
    for r in ALL_RECORDS:
        oc = r["served_outcome"]
        if oc == "abstain":
            if r["certification_tier"] != "core-certified" \
                    or r["abstain_reason"] != "merge_suspect_led":
                i4_bad.append(r)
        else:
            if r["certification_tier"] != "harness-heuristic":
                i4_bad.append(r)
        wa = oc == "witness_alt"
        if (bool(r["witness_alt_class"] != "") != wa
                or bool(r["witness_alt_basis"]) != wa
                or bool(r["policy_id"]) != wa
                or bool(r["policy_block_sha256"]) != wa
                or bool(r["abstain_reason"]) != (oc == "abstain")
                or r["contract_id"] != CONTRACT_ID
                or r["contract_version"] != CONTRACT_VERSION):
            dom_bad.append(r)
    add_check(
        "i4_tier_invariance_all_records", "contract §6 I4 + §3",
        "witness_alt/defer/answer records are always harness-heuristic; "
        "only abstain records are core-certified; no other path can mint "
        "authority",
        verdict(not i4_bad), verdict(not i4_bad),
        {"records_checked": len(ALL_RECORDS), "violations": i4_bad[:3]})
    add_check(
        "served_field_iff_domains_all_records", "contract §3",
        "witness_alt_class/basis/policy_id/policy_block_sha256 non-empty "
        "iff witness_alt; abstain_reason non-empty iff abstain; contract "
        "identity stamped on every record",
        verdict(not dom_bad), verdict(not dom_bad),
        {"records_checked": len(ALL_RECORDS), "violations": dom_bad[:3]})

    # ---- subject equivalence on every well-formed W2 cell, excluding
    # the already-failed evidence_ptr field (finding F1, counted once)
    diffs = []
    wellformed = [("syn-baseline", BASELINE_PKT, BASELINE_AUD)]
    wellformed += [(f"syn-{n}", p, mk_audit(q))
                   for n, _cl, p, q, _qid in negatives]
    for key, pkt, aud in wellformed:
        a = serve_committed(pkt, aud, key)["records"]
        b = serve_conformant(pkt, aud, "W2", key)["records"]
        for ra, rb in zip(a, b):
            for fld in wr.SERVED_FIELDS:
                if fld == "evidence_ptr":
                    continue
                if ra[fld] != rb[fld]:
                    diffs.append({"cell": key, "query_id": ra["query_id"],
                                  "field": fld, "committed": ra[fld],
                                  "conformant": rb[fld]})
    add_check(
        "subject_equivalence_excluding_evidence_ptr", "suite architecture",
        "on every well-formed W2 synthetic cell the two subjects emit "
        "field-identical records (evidence_ptr excluded: its divergence "
        "is finding F1, already a named failing check)",
        verdict(not diffs), verdict(not diffs),
        {"cells_compared": len(wellformed), "diffs": diffs[:5]})


def snapshot():
    """Canonical bytes of everything both subjects produce, for the
    registered double-pass determinism check."""
    cells = {"baseline": both(BASELINE_PKT, BASELINE_AUD, "det")}
    return json.dumps(
        {k: {"committed": v[0].get("records"),
             "conformant": v[1]["records"]} for k, v in cells.items()},
        sort_keys=True).encode()


def main() -> int:
    report = {"artifact": "PR-12.9 §5.1 conformance suite (Stage I)",
              "suite_version": SUITE_VERSION,
              "registration": "PR12_9_READER_CONTRACT_CERTIFICATION.md "
                              "(main fe6248e)",
              "contract_doc": "PR12_8_S1_CONTRACT_CANDIDATE.md",
              "contract_id": CONTRACT_ID,
              "contract_version": CONTRACT_VERSION,
              "hermetic": "synthetic packets only; no committed packet "
                          "read; no truth labels"}
    report["frozen_surfaces_clean_before"] = frozen_clean()
    ok_inputs, manifest = verify_consumed_inputs()
    report["consumed_input_manifest"] = manifest
    att_ok, sha = attest()
    report["policy_block_attestation"] = {
        "sha256": sha, "registered_sha256": REGISTERED_BLOCK_SHA,
        "committed_reader_equals_frozen_scorer_block": att_ok}
    if not (att_ok and ok_inputs):
        report["result"] = "suite-blocked (attestation or input drift)"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_FILE.write_text(json.dumps(report, indent=1, sort_keys=True)
                            + "\n")
        print("BLOCKED: attestation or consumed-input drift", file=sys.stderr)
        return 2

    run_checks()

    # registered determinism check: double pass, byte identity
    det = snapshot() == snapshot()
    add_check("determinism_double_pass", "PR-12.9 §5.1",
              "two full passes over identical synthetic inputs are "
              "byte-identical", verdict(det), verdict(det), {})

    report["checks"] = CHECKS
    fails = [(c["name"], s) for c in CHECKS
             for s in ("committed", "conformant") if c[s] == "fail"]
    findings = sorted({c["detail"].get("finding", "").split(" ")[0]
                       for c in CHECKS
                       if isinstance(c.get("detail"), dict)
                       and c["detail"].get("finding")} - {""})
    report["summary"] = {
        "n_checks": len(CHECKS),
        "n_subject_results": sum(1 for c in CHECKS
                                 for s in ("committed", "conformant")
                                 if c[s] in ("pass", "fail")),
        "failing_subject_results": [f"{n} [{s}]" for n, s in fails],
        "conformant_subject_failures": [n for n, s in fails
                                        if s == "conformant"],
        "committed_reader_findings": findings,
        "specification_ambiguities_found": 0,
        "ambiguity_note": "every crafted case has behavior the contract "
                          "text determines; no C-3 "
                          "certification-insufficient trigger was found"}
    report["frozen_surfaces_clean_after"] = frozen_clean()
    report["result"] = ("all-pass" if not fails else
                        f"{len(fails)} failing subject result(s); findings: "
                        + ", ".join(findings))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")

    print(f"SUITE: {report['result']}")
    print(f"  checks={len(CHECKS)} "
          f"failing={report['summary']['failing_subject_results']}")
    print(f"  spec ambiguities: 0; committed-reader findings: "
          f"{findings or 'none'}")
    print("Scope (PR-12.9 §11): Stage I conformance evidence only — "
          "nothing is certified or served to any consumer; no deployment, "
          "prompting use, promotion, ingestion, FAM-core change, or "
          "reader-contract change; PR-10 merge-abstain remains the only "
          "certified reader contract; posture remains deferral.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
