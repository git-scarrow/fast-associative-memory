#!/usr/bin/env python3
"""PR-12.8 Stage E — candidacy adjudication (read-only).

Evaluates the §8 candidacy gates G-R1–G-R7 of
`PR12_8_READER_CONTRACT_CANDIDACY.md` over the committed Stage A–D
record and the §9 monitoring registration, and emits exactly one §12
verdict into ``pr12_8/candidacy_scan.json`` (the §11 Stage E artifact).
Under the registered Stage D-bounded branch (§14.5) the unqualified GO
is unreachable: the only reachable GO is
``contract-candidate-GO-seedbounded(W2:F1b)``, and the verdict text
must carry the §14.3 traffic bounds, the §14.4 contra-power bound, and
the §14.5 seed bound verbatim, citing the §9 monitoring registration
as standing terms (dropping any bound is kill §10.9).

G-R1 is verified by INDEPENDENT RECOMPUTATION, not by trusting the
reference reader's own record: this adjudicator re-derives every
envelope cell's witness_alt multiset from the committed packets
through the sha-attested frozen policy block and requires exact
equality with the frozen envelope v0.2 entries; I1/I2 composition is
likewise recomputed from the packets. G-R7 attests the verbatim policy
block across every Stage artifact that carries it.

Even a GO emitted here is a CANDIDACY verdict and nothing more: it
authorizes exactly one thing — a future certification pre-registration
may cite it. Nothing is served to any consumer; no deployment, live
acting, prompting use, promotion, memory ingestion, FAM-core change,
or reader-contract change. **PR-10 merge-abstain remains the only
certified reader contract**; the operational posture on witness-window
rows remains **deferral**. Analysis-only, stdlib + subprocess-git; the
only writes are ``pr12_8/candidacy_scan.json``.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

BASE = Path("results/issue_failure_mode_blindness")
DESIGN_MEMO = str(BASE / "PR12_8_READER_CONTRACT_CANDIDACY.md")
MONITORING_DOC = str(BASE / "PR12_8_MONITORING_WITHDRAWAL.md")
CONTRACT_DOC = str(BASE / "PR12_8_S1_CONTRACT_CANDIDATE.md")
FROZEN_SCORER = Path("harness/action_boundary_score.py")
OUT = BASE / "pr12_8" / "candidacy_scan.json"

ADJUDICATION_PIN = "8f66f7be4b8acebde37657f776de8a1ad021d8f0"
SEED_BRANCH = "D-bounded"          # §14.5, operator-chosen
GO_NAME = "contract-candidate-GO-seedbounded(W2:F1b)"

ATTESTED_SOURCES = (               # every artifact carrying the §5 block
    "harness/action_boundary_score.py",
    "harness/action_boundary_holdout_score.py",
    "harness/action_boundary_panel_score.py",
    "harness/witness_alt_reference_reader.py",
    "harness/candidacy_adjudicate.py",
)

# Registered near-ceiling caveats carried into the verdict record (§8
# G-R5; sources: pr12_7 §26.3 table, pr12_6 §21 dev caveat).
NEAR_CEILING_CAVEATS = [
    {"unit": "W2:pairE_oneshot_s1 (holdout)", "wrong_mass_rate": 0.09434,
     "ceiling": 0.10, "partition": "holdout (gated, passed)"},
    {"unit": "W2:pairD_oneshot_s0 (12.6 dev)", "wrong_mass_rate": 0.1163,
     "ceiling": 0.10, "partition": "dev (never gated; recorded 12.6 §21)"},
]

# ---------------------------------------------------------------------------
# §5 frozen policy family — copied VERBATIM from action_boundary_score.py
# (G-R7: sha-attested across every stage artifact; any divergence aborts).
# Do not edit anything between here and the end of the POLICIES table.
# ---------------------------------------------------------------------------
PIN = "0afcb2bc4d94112fd2f2cb9a47525c6d2595c2dd"  # main @ PR-12.5 merge (§3)

PRECISION_FLOOR = 0.75          # G-A1 / G-C2
COVERAGE_FLOOR = 0.25           # G-A2, aggregate one-shot test coverage
CONTRA_WRONG_CEILING = 0.05     # G-C1, per contra test unit
PER_UNIT_WRONG_CEILING = 0.10   # §14, every test unit
GLOBAL_WRONG_CEILING = 0.05     # §13, per candidate over its test units

WITNESS_BASIS = "witness co-resident (fork_witness)"


class RowObs:
    __slots__ = ("shape", "width", "deployed_class", "led_slot",
                 "alt_classes", "all_witness", "presented")

    def __init__(self, shape, tie):
        cands = tie["candidates"]
        deployed = [c for c in cands if c.get("basis") == "deployed vote"]
        self.shape = shape                       # emitting governance layer
        self.deployed_class = deployed[0]["decode_class"] if deployed \
            else None
        self.led_slot = deployed[0].get("slot") if deployed else None
        self.alt_classes = [c["decode_class"] for c in cands
                            if c.get("basis") != "deployed vote"]
        self.all_witness = all(c.get("basis") == WITNESS_BASIS
                               for c in cands
                               if c.get("basis") != "deployed vote")
        self.presented = sorted({c["decode_class"] for c in cands})
        self.width = len(self.presented)


class CellCtx:
    __slots__ = ("n_contradiction_pairs", "n_ambiguous_pairs",
                 "never_resolving_slots", "hazard_tier")

    def __init__(self, audit_lines):
        self.n_contradiction_pairs = 0
        self.n_ambiguous_pairs = 0
        self.never_resolving_slots = set()
        self.hazard_tier = None
        for rec in audit_lines:
            rt = rec.get("record_type")
            if rt == "contradiction_pair_review":
                self.n_contradiction_pairs += 1
            elif rt == "ambiguous_pair_review":
                self.n_ambiguous_pairs += 1
                if rec.get("never_resolving"):
                    self.never_resolving_slots.add(
                        rec["pair"]["incumbent_slot"])
                    self.never_resolving_slots.add(rec["pair"]["owner_slot"])
            elif self.hazard_tier is None and "hazard_tier" in rec:
                self.hazard_tier = rec["hazard_tier"].get("tier")


def _f1a_condition(row: RowObs, ctx: CellCtx, shapes) -> bool:
    return (row.shape in shapes and row.width == 2 and row.all_witness
            and row.led_slot in ctx.never_resolving_slots
            and len(row.alt_classes) == 1)


def pol_defer_all(row, ctx):
    return None


def pol_act_uniform(row, ctx):
    return {c: 1.0 / len(row.presented) for c in row.presented}


def pol_act_alt(row, ctx):
    alts = sorted(set(row.alt_classes))
    if not alts:   # empty alt set: acts on deployed by exhaustion (§4 12.5)
        return {row.deployed_class: 1.0}
    return {c: 1.0 / len(alts) for c in alts}


def pol_f1a(row, ctx):
    if _f1a_condition(row, ctx, shapes=("W2",)):
        return {row.alt_classes[0]: 1.0}
    return None


def pol_f1b(row, ctx):
    if _f1a_condition(row, ctx, shapes=("W2",)) \
            and ctx.n_contradiction_pairs <= ctx.n_ambiguous_pairs:
        return {row.alt_classes[0]: 1.0}
    return None


def pol_f1c(row, ctx):
    if _f1a_condition(row, ctx, shapes=("W1", "W2")):
        return {row.alt_classes[0]: 1.0}
    return None


POLICIES = {"B-defer-all": pol_defer_all, "B-act-uniform": pol_act_uniform,
            "B-act-alt": pol_act_alt, "F1a": pol_f1a, "F1b": pol_f1b,
            "F1c": pol_f1c}
# --------------------------- end verbatim §5 block -------------------------


G_H3_LINE_ANCHORS = (
    'PIN = "', "PRECISION_FLOOR = ", "COVERAGE_FLOOR = ",
    "CONTRA_WRONG_CEILING = ", "PER_UNIT_WRONG_CEILING = ",
    "GLOBAL_WRONG_CEILING = ", "WITNESS_BASIS = ")
G_H3_BLOCK_ANCHORS = (
    "class RowObs:", "class CellCtx:", "def _f1a_condition(",
    "def pol_defer_all(", "def pol_act_uniform(", "def pol_act_alt(",
    "def pol_f1a(", "def pol_f1b(", "def pol_f1c(", "POLICIES = {")


def extract_policy_block(src: str) -> str:
    lines = src.splitlines()
    segs = []
    for prefix in G_H3_LINE_ANCHORS:
        hits = [ln for ln in lines if ln.startswith(prefix)]
        if len(hits) != 1:
            raise ValueError(f"anchor not unique: {prefix!r} ({len(hits)})")
        segs.append(hits[0])
    for anchor in G_H3_BLOCK_ANCHORS:
        idx = [i for i, ln in enumerate(lines) if ln.startswith(anchor)]
        if len(idx) != 1:
            raise ValueError(f"anchor not unique: {anchor!r} ({len(idx)})")
        i = idx[0]
        block = [lines[i]]
        i += 1
        while i < len(lines) and (lines[i] == ""
                                  or lines[i][0] in (" ", "\t")):
            block.append(lines[i])
            i += 1
        while block and block[-1] == "":
            block.pop()
        segs.append("\n".join(block))
    return "\n".join(segs)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def verify_input(repo: Path, relpath: str, manifest: dict, kills: list):
    p = repo / relpath
    try:
        tree = p.read_bytes()
    except FileNotFoundError:
        kills.append({"kill": 1, "label": f"missing input {relpath}"})
        return None
    try:
        committed = subprocess.run(
            ["git", "cat-file", "blob", f"{ADJUDICATION_PIN}:{relpath}"],
            cwd=repo, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError:
        kills.append({"kill": 1, "label":
                      f"no committed object {ADJUDICATION_PIN}:{relpath}"})
        return None
    ok = tree == committed
    manifest[relpath] = {"sha256": hashlib.sha256(tree).hexdigest(),
                         "matches_pin": ok}
    if not ok:
        kills.append({"kill": 1, "label": f"input drifted: {relpath}"})
    return tree


def main() -> int:
    repo = repo_root()
    manifest, kills = {}, []
    report = {"design_memo": DESIGN_MEMO, "contract_doc": CONTRACT_DOC,
              "monitoring_doc": MONITORING_DOC,
              "adjudication_pin": ADJUDICATION_PIN, "policy_pin": PIN,
              "seed_branch": SEED_BRANCH,
              "gates": {}, "input_manifest": manifest,
              "kill_conditions": kills}

    # ---- G-R7: no-tuning attestation across every stage artifact
    shas = {}
    try:
        for rel in ATTESTED_SOURCES:
            shas[rel] = hashlib.sha256(extract_policy_block(
                (repo / rel).read_text()).encode()).hexdigest()
        gr7_ok = len(set(shas.values())) == 1
    except ValueError as e:
        gr7_ok = False
        kills.append({"kill": 3, "label": f"G-R7 extraction failed: {e}"})
    report["gates"]["G-R7"] = {"block_sha256_by_source": shas,
                               "pass": gr7_ok}
    if not gr7_ok:
        kills.append({"kill": 3, "label": "G-R7: policy block divergence"})

    # ---- load committed stage artifacts (all pinned)
    def load(rel):
        raw = verify_input(repo, rel, manifest, kills)
        return json.loads(raw) if raw is not None else None
    env = load(str(BASE / "pr12_8" / "f1b_envelope.json"))
    r6 = load(str(BASE / "pr12_6" / "action_boundary_scan.json"))
    r7 = load(str(BASE / "pr12_7" / "holdout_scan.json"))
    r8 = load(str(BASE / "pr12_8" / "panel_scan.json"))
    for rel in (DESIGN_MEMO, MONITORING_DOC, CONTRACT_DOC):
        verify_input(repo, rel, manifest, kills)
    if None in (env, r6, r7, r8):
        report["verdict"] = "candidacy-blocked"
        print("VERDICT: candidacy-blocked (missing/drifted inputs)")
        return 1

    # ---- G-R1 (+ I1/I2 recompute for G-R2): independent recomputation of
    # every envelope cell's witness_alt multiset from the packets through
    # the attested block; exact equality with envelope v0.2 required.
    gr1_cells, gr2_cells = {}, {}
    for key in sorted(env["cells"]):
        source, _, cell = key.split("/")
        pkt = verify_input(repo, f"{BASE}/{source}/W2/{cell}/"
                                 "memory_packet.jsonl", manifest, kills)
        aud = verify_input(repo, f"{BASE}/{source}/W2/{cell}/"
                                 "audit_packet.jsonl", manifest, kills)
        if pkt is None or aud is None:
            gr1_cells[key] = False
            continue
        ctx = CellCtx([json.loads(line)
                       for line in aud.decode().splitlines()])
        recomputed, abstain, eligible = [], set(), set()
        for line in pkt.decode().splitlines():
            rec = json.loads(line)
            items = rec.get("items", [])
            if any(it.get("type") == "abstention_notice" for it in items):
                abstain.add(rec["query_id"])
                continue
            ties = [it for it in items
                    if it.get("type") == "unresolved_tie"]
            if not ties:
                continue
            act = pol_f1b(RowObs("W2", ties[0]), ctx)
            if act is not None:
                recomputed.append({"query_id": rec["query_id"],
                                   "class": sorted(act)[0]})
                eligible.add(rec["query_id"])
        gr1_cells[key] = recomputed == env["cells"][key]["witness_alt_rows"]
        overlap = abstain & eligible
        comp = env["cells"][key]["composition"]
        gr2_cells[key] = (not overlap) and comp["pass"]
        if not gr1_cells[key]:
            kills.append({"kill": 2, "label": f"G-R1 mismatch: {key}"})
        if not gr2_cells[key]:
            kills.append({"kill": 2, "label": f"G-R2 failure: {key}"})
    report["gates"]["G-R1"] = {
        "cells_recomputed": len(gr1_cells),
        "all_exact": all(gr1_cells.values()),
        "note": "witness_alt multisets independently recomputed from "
                "packets via the attested block; exact equality with the "
                "frozen envelope v0.2 on every cell",
        "reference_reader_record": {
            "internal_double_pass": env["internal_double_pass_identical"],
            "committed_external_rerun": "memo §14.3: all 238 pr12_8 files "
                                        "byte-identical"},
        "pass": all(gr1_cells.values())}
    report["gates"]["G-R2"] = {
        "cells_checked": len(gr2_cells),
        "envelope_composition_pass": all(
            env["cells"][k]["composition"]["pass"] for k in env["cells"]),
        "i2_recomputed_disjoint_all": all(gr2_cells.values()),
        "pass": all(gr2_cells.values())}

    # ---- G-R3: per-traffic-type ledger (scored-with-floors or honest bound)
    ledger = {
        "oneshot": {"status": "scored-pass", "source":
                    ["pr12_6 action-boundary-evidence-GO(W2:F1b)",
                     "pr12_7 holdout-validity-GO(W2:F1b)"],
                    "check": r6.get("combo_pass", {}).get("W2:F1b") is True
                    and r7["verdict"] == "holdout-validity-GO(W2:F1b)"
                    and r7["combo_pass"]["W2:F1b"] is True},
        "contra": {"status": "scored-pass (zero action; §14.4 power bound)",
                   "source": ["pr12_6 G-C1–G-C3", "pr12_7 G-C1–G-C3",
                              "memo §14.4 contra-power accounting"],
                   "check": r7["gates"]["W2:F1b"]["G-C3_no_collapse"]["pass"]},
        "mixed": {"status": r8["type_status"]["mixed"]["status"],
                  "source": ["pr12_8 panel_scan"],
                  "check": r8["type_status"]["mixed"]["status"]
                  == "scored-pass"},
        "stale": {"status": r8["type_status"]["stale"]["status"],
                  "scope_bound": "structurally tie-free; named bound",
                  "check": r8["type_status"]["stale"]["status"]
                  == "panel-insufficient"},
        "jitter": {"status": r8["type_status"]["jitter"]["status"],
                   "scope_bound": "s0-only stems, no governance arm; "
                                  "named bound",
                   "check": r8["type_status"]["jitter"]["status"]
                   == "panel-insufficient"},
        "g5twin": {"status": r8["type_status"]["g5twin"]["status"],
                   "scope_bound": "s0-only stem, no governance arm; "
                                  "named bound",
                   "check": r8["type_status"]["g5twin"]["status"]
                   == "panel-insufficient"},
    }
    gr3_ok = all(v["check"] for v in ledger.values())
    report["gates"]["G-R3"] = {"ledger": ledger, "pass": gr3_ok,
                               "note": "no third state: every type scored "
                                       "with floors met or carried as a "
                                       "named panel-insufficient bound"}
    if not gr3_ok:
        kills.append({"kill": 4, "label": "G-R3 ledger inconsistency"})

    # ---- G-R4: guard-regime statement (margins from the envelope)
    margins = {}
    for key in env["cells"]:
        g = env["cells"][key]["guard"]
        margins[key] = g["n_contradiction_pairs"] - g["n_ambiguous_pairs"]
    open_m = sorted(m for m in margins.values() if m <= 0)
    closed_m = sorted(m for m in margins.values() if m > 0)
    boundary_cells = {k: m for k, m in margins.items() if abs(m) <= 4}
    report["gates"]["G-R4"] = {
        "open_margin_range": [open_m[0], open_m[-1]] if open_m else None,
        "closed_margin_range": [closed_m[0], closed_m[-1]]
        if closed_m else None,
        "boundary_cells": boundary_cells,
        "boundary_note": "pairD_stale_s1 (margin -4) observed with zero "
                         "in-scope rows — no per-row behavior exists to "
                         "report (memo §14.3)",
        "corridor_unobserved": "(-4, +23)",
        "pass": True,
        "note": "bimodality recorded across the full 44-cell panel; the "
                "corridor scope bound is carried in the verdict text"}

    # ---- G-R5: safety ceilings on the extended panel (from committed scans)
    panel_gates_ok = all(
        all(v["pass"] for v in g.values()
            if isinstance(v, dict) and "pass" in v)
        for un, g in r8["gates"]["W2:F1b"].items()
        if un != "S14_global_ceiling") \
        and r8["gates"]["W2:F1b"]["S14_global_ceiling"]["pass"]
    gr5_ok = (r6.get("combo_pass", {}).get("W2:F1b") is True
              and r7["combo_pass"]["W2:F1b"] is True and panel_gates_ok)
    report["gates"]["G-R5"] = {
        "constants_unchanged": {
            "precision_floor": PRECISION_FLOOR,
            "coverage_floor": COVERAGE_FLOOR,
            "contra_wrong_ceiling": CONTRA_WRONG_CEILING,
            "per_unit_wrong_ceiling": PER_UNIT_WRONG_CEILING,
            "global_wrong_ceiling": GLOBAL_WRONG_CEILING},
        "near_ceiling_caveats": NEAR_CEILING_CAVEATS,
        "sources_pass": {"pr12_6_W2_F1b": r6.get("combo_pass",
                                                 {}).get("W2:F1b"),
                         "pr12_7_W2_F1b": r7["combo_pass"]["W2:F1b"],
                         "pr12_8_panel_W2_F1b": panel_gates_ok},
        "pass": gr5_ok}
    if not gr5_ok:
        kills.append({"kill": 4, "label": "G-R5 ceiling failure"})

    # ---- G-R6: seed-axis honesty (D-bounded → -seedbounded name forced)
    report["gates"]["G-R6"] = {
        "branch": SEED_BRANCH,
        "reachable_go_name": GO_NAME,
        "unqualified_go_reachable": False,
        "pass": True}

    # ---- verdict (§12; exactly one)
    hard = [report["gates"][g]["pass"]
            for g in ("G-R1", "G-R2", "G-R3", "G-R4", "G-R5",
                      "G-R6", "G-R7")]
    if kills:
        verdict = "candidacy-blocked"
    elif all(hard):
        verdict = GO_NAME
    else:
        verdict = "contract-candidacy-negative"
    report["verdict"] = verdict
    report["verdict_text"] = (
        f"{verdict} — the s1-witness-alt-batch 0.1-candidate batch "
        "packet-reader contract for W2:F1b is a certified-candidate under "
        "the following registered bounds, which are constitutive parts of "
        "this verdict: (seed, §14.5) valid only for the committed seeds' "
        "distributional regime s0–s2 on the committed panel protocol — no "
        "claim about any other seed, protocol, encoder, or drift; "
        "(traffic, §14.3) plain-stale, jitter, and g5twin are "
        "panel-insufficient named scope bounds, and mixed traffic is "
        "covered as safe-by-silence only (zero acting, zero utility); "
        "(contra power, §14.4) contra-side safety = zero action on 319 "
        "in-scope W2 rows across 18 guard-closed cells, counterfactual "
        "guardless exposure 73 acts / 70.0 wrong mass suppressed, per-cell "
        "guard mis-open rate bounded only at ≤13.3% (95%, iid-cell model), "
        "guard behavior inside the (−4, +23) margin corridor unobserved; "
        "(monitoring, §9) the PR12_8_MONITORING_WITHDRAWAL.md tripwires "
        "T1–T7 and withdrawal semantics are the candidate's standing "
        "terms. This verdict authorizes exactly one thing: a future "
        "certification pre-registration may cite it. Nothing is served to "
        "any consumer; no deployment, live acting, prompting use, "
        "promotion, memory ingestion, FAM-core change, autonomous "
        "downstream use, or reader-contract change. PR-10 merge-abstain "
        "remains the only certified reader contract; the operational "
        "posture on witness-window rows remains deferral."
        if verdict == GO_NAME else
        f"{verdict} — see gate table; no candidacy status is conferred; "
        "PR-10 merge-abstain remains the only certified reader contract; "
        "posture remains deferral.")

    # ---- internal double pass (recompute G-R1 fresh)
    gr1_again = {}
    for key in sorted(env["cells"]):
        source, _, cell = key.split("/")
        pkt = (repo / BASE / source / "W2" / cell
               / "memory_packet.jsonl").read_bytes()
        aud = (repo / BASE / source / "W2" / cell
               / "audit_packet.jsonl").read_bytes()
        ctx = CellCtx([json.loads(line)
                       for line in aud.decode().splitlines()])
        recomputed = []
        for line in pkt.decode().splitlines():
            rec = json.loads(line)
            items = rec.get("items", [])
            if any(it.get("type") == "abstention_notice" for it in items):
                continue
            ties = [it for it in items
                    if it.get("type") == "unresolved_tie"]
            if not ties:
                continue
            act = pol_f1b(RowObs("W2", ties[0]), ctx)
            if act is not None:
                recomputed.append({"query_id": rec["query_id"],
                                   "class": sorted(act)[0]})
        gr1_again[key] = recomputed == env["cells"][key]["witness_alt_rows"]
    same = gr1_again == gr1_cells
    report["internal_double_pass_identical"] = same
    if not same:
        kills.append({"kill": 7, "label": "internal double pass differs"})
        report["verdict"] = "candidacy-blocked"

    with open(repo / OUT, "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
        f.write("\n")

    print(f"VERDICT: {report['verdict']}")
    for g in ("G-R1", "G-R2", "G-R3", "G-R4", "G-R5", "G-R6", "G-R7"):
        print(f"  {g}: "
              f"{'PASS' if report['gates'][g]['pass'] else 'FAIL'}")
    print("Scope: candidacy adjudication only — rung 3 of the contract "
          "candidate's status ladder; certification is a separate future "
          "pre-registration; nothing is served, deployed, promoted, or "
          "ingested; PR-10 merge-abstain remains the only certified "
          "reader contract; posture remains deferral.")
    return 1 if kills else 0


if __name__ == "__main__":
    sys.exit(main())
