#!/usr/bin/env python3
"""PR-12.8 Stage C(ii)+(iii) — S1 reference reader, composition proof, envelope freeze.

Implements the conformance requirements of the registered contract
definition `PR12_8_S1_CONTRACT_CANDIDATE.md` (contract_id
``s1-witness-alt-batch``, version ``0.1-candidate``) and produces the
Stage C artifacts of `PR12_8_READER_CONTRACT_CANDIDACY.md` §7/§11:

* **C(ii) composition proof** — I1 (precedence), I2 (eligible set
  disjoint from the certified abstention set, checked exactly per
  cell, fail-closed), I3 (incumbent immutability: packets are read
  only; frozen-surface git-clean before and after), I4 (no new
  authority: ``witness_alt`` records are always ``harness-heuristic``)
  — proven on every committed W2 packet cell (pr12_3 s0, pr12_4 s1/s2,
  pr12_7_holdout_cache C/E).
* **C(iii) exactness envelope** — ``pr12_8/f1b_envelope.json``: per
  cell, the exact ``witness_alt`` row multiset (query_id + acted class
  in packet order), outcome counts, guard state, composition results,
  and — where a committed truth-joined table exists (pr12_6/pr12_7
  ``rows_W2_*_F1b.csv``) — the expected-correct mass copied from that
  committed table after an exact acted-set/act-class cross-check. The
  reader itself is label-free: every serving decision is made from the
  packet pair alone, before any committed table is opened.

Read-only over committed artifacts; stdlib + subprocess-git only; the
only writes are under ``pr12_8/`` and temp. No FAM-core import, no
emitter import, no truth CSV read. Label-freedom is structural: policy
code receives only RowObs/CellCtx built from the packets (the §5
policy block below is copied VERBATIM from the frozen
``action_boundary_score.py`` and sha-attested at runtime — any
divergence aborts). **This program serves nothing to anyone**: it
produces proof artifacts for a contract candidate that has no
operative force. PR-10 merge-abstain remains the only certified reader
contract; posture on witness-window rows remains deferral.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

BASE = Path("results/issue_failure_mode_blindness")
CONTRACT_DOC = str(BASE / "PR12_8_S1_CONTRACT_CANDIDATE.md")
DESIGN_MEMO = str(BASE / "PR12_8_READER_CONTRACT_CANDIDACY.md")
FROZEN_SCORER = Path("harness/action_boundary_score.py")
OUT_DIR = BASE / "pr12_8"

CONTRACT_ID = "s1-witness-alt-batch"
CONTRACT_VERSION = "0.1-candidate"

# Panel pin: main @ the Stage C(i) merge — every consumed input below is
# committed at this commit.
PANEL_PIN = "8cbf870d7de2d81747d1f6a8e9a32b3a09e0d6dc"

# The committed W2 evidence panel (every W2-shape packet cell in the repo)
# and, where one exists, the committed truth-joined F1b table whose ACT
# set the reader's witness_alt set must reproduce exactly.
PANEL = []  # (source_tree, cell, committed_f1b_table_or_None)
for _cell in ("clean_pairA_s0", "pairB_contra_s0", "pairB_oneshot_s0",
              "pairD_contra_s0", "pairD_oneshot_s0", "pairD_stale-soft_s0"):
    PANEL.append(("pr12_3", _cell, None))          # 12.6 dev; no table
for _cell in ("clean_pairA_s1", "clean_pairA_s2",
              "pairD_stale-soft_s1", "pairD_stale-soft_s2"):
    PANEL.append(("pr12_4", _cell, None))          # never truth-scored
for _cell in ("pairB_contra_s1", "pairB_contra_s2",
              "pairB_oneshot_s1", "pairB_oneshot_s2",
              "pairD_contra_s1", "pairD_contra_s2",
              "pairD_oneshot_s1", "pairD_oneshot_s2"):
    PANEL.append(("pr12_4", _cell,
                  str(BASE / "pr12_6" / f"rows_W2_{_cell}_F1b.csv")))
for _cell in ("clean_pairC_s1", "clean_pairE_s1",
              "pairC_contra_s1", "pairC_contra_s2",
              "pairC_oneshot_s1", "pairC_oneshot_s2",
              "pairE_contra_s1", "pairE_contra_s2",
              "pairE_oneshot_s1", "pairE_oneshot_s2"):
    PANEL.append(("pr12_7_holdout_cache", _cell,
                  str(BASE / "pr12_7" / f"rows_W2_{_cell}_F1b.csv")))

SHAPE = "W2"   # contract §2: W2-shape packet trees only

SERVED_FIELDS = ["query_id", "served_outcome", "abstain_reason",
                 "witness_alt_class", "witness_alt_basis", "policy_id",
                 "policy_block_sha256", "evidence_ptr",
                 "certification_tier", "contract_id", "contract_version"]

# ---------------------------------------------------------------------------
# §5 frozen policy family — copied VERBATIM from action_boundary_score.py
# (contract §1/§4: the sha-attested block is authoritative; any divergence
# aborts). Do not edit anything between here and the end of the POLICIES
# table.
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
            ["git", "cat-file", "blob", f"{PANEL_PIN}:{relpath}"],
            cwd=repo, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError:
        kills.append({"kill": 1,
                      "label": f"no committed object {PANEL_PIN}:{relpath}"})
        return None
    ok = tree == committed
    manifest[relpath] = {"sha256": hashlib.sha256(tree).hexdigest(),
                         "matches_pin": ok}
    if not ok:
        kills.append({"kill": 1, "label": f"input drifted: {relpath}"})
    return tree


BLOCK_SHA = None  # set in main() after attestation


def read_cell(pkt_bytes: bytes, aud_bytes: bytes, anomalies: list,
              cell_key: str):
    """Contract §4/§5/§6: one served-decision record per packet row.

    Decisions are made from the packet pair ALONE (label-free; no
    committed table is open at this point). Fail-closed: any anomaly
    serves defer, never witness_alt."""
    ctx = CellCtx([json.loads(line)
                   for line in aud_bytes.decode().splitlines()])
    records = []
    abstain_qids, eligible_qids = set(), set()
    for line in pkt_bytes.decode().splitlines():
        rec = json.loads(line)
        qid = rec["query_id"]
        items = rec.get("items", [])
        abst = [it for it in items if it.get("type") == "abstention_notice"]
        ties = [it for it in items if it.get("type") == "unresolved_tie"]
        mems = [it for it in items if it.get("type") == "memory_item"]
        out = {"query_id": qid, "abstain_reason": "",
               "witness_alt_class": "", "witness_alt_basis": "",
               "policy_id": "", "policy_block_sha256": "",
               "evidence_ptr": "",
               "certification_tier": "harness-heuristic",
               "contract_id": CONTRACT_ID,
               "contract_version": CONTRACT_VERSION}
        if abst:                                # §6 I1: abstain precedence
            out["served_outcome"] = "abstain"
            out["abstain_reason"] = "merge_suspect_led"
            out["certification_tier"] = "core-certified"
            abstain_qids.add(qid)
            if ties:                            # fail-closed + record (I1)
                anomalies.append({"cell": cell_key, "query_id": qid,
                                  "anomaly": "abstention row carries an "
                                             "unresolved_tie item"})
        elif ties:
            obs = RowObs(SHAPE, ties[0])        # §4.1: first tie item
            act = pol_f1b(obs, ctx)
            if act is not None and not set(act) <= set(obs.presented):
                anomalies.append({"cell": cell_key, "query_id": qid,
                                  "anomaly": "ACT outside presented set"})
                act = None                      # fail-closed (§7)
            if act is not None:
                out["served_outcome"] = "witness_alt"
                out["witness_alt_class"] = sorted(act)[0]
                out["witness_alt_basis"] = WITNESS_BASIS
                out["policy_id"] = "W2:F1b"
                out["policy_block_sha256"] = BLOCK_SHA
                out["evidence_ptr"] = ties[0].get("text", "")
                eligible_qids.add(qid)
            else:
                out["served_outcome"] = "defer"
        elif mems:
            out["served_outcome"] = "answer"
        else:
            out["served_outcome"] = "defer"
        records.append(out)
    return ctx, records, abstain_qids, eligible_qids


def main() -> int:
    global BLOCK_SHA
    repo = repo_root()
    out_root = repo / OUT_DIR
    manifest, kills, anomalies = {}, [], []
    report = {"contract_doc": CONTRACT_DOC, "design_memo": DESIGN_MEMO,
              "contract_id": CONTRACT_ID,
              "contract_version": CONTRACT_VERSION,
              "panel_pin": PANEL_PIN, "policy_pin": PIN,
              "shape": SHAPE, "cells": {},
              "input_manifest": manifest, "anomalies": anomalies,
              "kill_conditions": kills}

    def dirs_clean():
        r = subprocess.run(
            ["git", "status", "--porcelain", "--",
             str(BASE / "pr12_3"), str(BASE / "pr12_4"),
             str(BASE / "pr12_6"), str(BASE / "pr12_7"),
             str(BASE / "pr12_7_holdout_cache"), str(BASE / "pr10"),
             "harness/harness_boundary_sim.py",
             "harness/action_boundary_score.py",
             "harness/action_boundary_holdout_score.py",
             "harness/harness_policy.json"],
            cwd=repo, capture_output=True, check=True)
        lines = [ln for ln in r.stdout.decode().splitlines()
                 if not ln.endswith(".DS_Store")]
        return "\n".join(lines) == ""
    report["i3_dirs_clean_before"] = dirs_clean()
    if not report["i3_dirs_clean_before"]:
        kills.append({"kill": 6, "label": "frozen dirs dirty before run"})

    # ---- no-tuning attestation (contract §1; PR-12.8 G-R7 pattern)
    try:
        mine = extract_policy_block(
            (repo / "harness" / "witness_alt_reference_reader.py")
            .read_text())
        frozen = extract_policy_block((repo / FROZEN_SCORER).read_text())
        BLOCK_SHA = hashlib.sha256(mine.encode()).hexdigest()
        att_ok = mine == frozen
    except ValueError as e:
        att_ok = False
        kills.append({"kill": 3, "label": f"attestation extraction: {e}"})
    report["policy_block_attestation"] = {
        "copied_block_sha256": BLOCK_SHA,
        "frozen_source": str(FROZEN_SCORER), "pass": att_ok}
    if not att_ok:
        kills.append({"kill": 3, "label": "policy block != frozen source"})
        report["verdict"] = "stageC-blocked"
        print("KILL: attestation failed", file=sys.stderr)
        return 1

    # ---- decision pass (label-free) over the full W2 panel
    def one_pass():
        cells = {}
        for source, cell, table in PANEL:
            key = f"{source}/W2/{cell}"
            pkt = verify_input(repo, f"{BASE}/{source}/W2/{cell}/"
                                     "memory_packet.jsonl", manifest, kills)
            aud = verify_input(repo, f"{BASE}/{source}/W2/{cell}/"
                                     "audit_packet.jsonl", manifest, kills)
            if pkt is None or aud is None:
                continue
            ctx, records, abstain_qids, eligible_qids = read_cell(
                pkt, aud, anomalies, key)
            cells[key] = {"source": source, "cell": cell, "table": table,
                          "ctx": ctx, "records": records,
                          "abstain_qids": abstain_qids,
                          "eligible_qids": eligible_qids}
        return cells
    cells = one_pass()

    # ---- C(ii) composition proof, per cell (truth still not read)
    for key, c in sorted(cells.items()):
        overlap = sorted(c["abstain_qids"] & c["eligible_qids"])
        counts = {}
        for r in c["records"]:
            counts[r["served_outcome"]] = counts.get(
                r["served_outcome"], 0) + 1
        i4_bad = [r["query_id"] for r in c["records"]
                  if r["served_outcome"] == "witness_alt"
                  and r["certification_tier"] != "harness-heuristic"]
        i1_bad = [a for a in anomalies if a["cell"] == key
                  and "abstention row" in a["anomaly"]]
        comp = {"i1_abstain_precedence_violations": len(i1_bad),
                "i2_abstain_eligible_overlap": overlap,
                "i2_disjoint": not overlap,
                "i4_tier_violations": i4_bad,
                "pass": not overlap and not i4_bad and not i1_bad}
        if not comp["pass"]:
            kills.append({"kill": 2, "label": f"composition failed: {key}"})
        report["cells"][key] = {
            "n_rows": len(c["records"]),
            "outcome_counts": counts,
            "guard": {
                "n_contradiction_pairs": c["ctx"].n_contradiction_pairs,
                "n_ambiguous_pairs": c["ctx"].n_ambiguous_pairs,
                "open": c["ctx"].n_contradiction_pairs
                <= c["ctx"].n_ambiguous_pairs},
            "composition": comp,
            "witness_alt_rows": [
                {"query_id": r["query_id"], "class": r["witness_alt_class"]}
                for r in c["records"]
                if r["served_outcome"] == "witness_alt"]}

    # ---- C(iii) envelope: cross-check vs committed truth-joined tables
    # (post-decision only; expected-correct mass is COPIED, never computed)
    for key, c in sorted(cells.items()):
        entry = report["cells"][key]
        if c["table"] is None:
            entry["committed_truth_table"] = None
            entry["committed_act_set_match"] = None
            entry["expected_correct_mass"] = None
            continue
        tbl = verify_input(repo, c["table"], manifest, kills)
        if tbl is None:
            continue
        committed = list(csv.DictReader(tbl.decode().splitlines()))
        act_rows = [r for r in committed if r["decision"] == "ACT"]
        committed_set = {r["query_id"]:
                         sorted(json.loads(r["acted_class_mass"]))[0]
                         for r in act_rows}
        mine_set = {r["query_id"]: str(r["witness_alt_class"])
                    for r in c["records"]
                    if r["served_outcome"] == "witness_alt"}
        match = committed_set == mine_set
        entry["committed_truth_table"] = c["table"]
        entry["committed_act_set_match"] = match
        entry["expected_correct_mass"] = round(
            sum(float(r["expected_correct"]) for r in act_rows), 6)
        entry["acted_rows_committed"] = len(act_rows)
        if not match:
            kills.append({"kill": 5, "label":
                          f"witness_alt set != committed F1b ACT set: "
                          f"{key}"})

    # ---- I3 epilogue + determinism (internal double pass)
    report["i3_dirs_clean_after"] = dirs_clean()
    if not report["i3_dirs_clean_after"]:
        kills.append({"kill": 6, "label": "frozen dirs dirty after run"})
    cells2 = one_pass()

    def snap(cs):
        return json.dumps({k: c["records"] for k, c in sorted(cs.items())},
                          sort_keys=True)
    same = snap(cells) == snap(cells2)
    report["internal_double_pass_identical"] = same
    if not same:
        kills.append({"kill": 7, "label": "internal double pass differs"})

    # ---- summary + verdict
    total = {"cells": len(cells),
             "rows": sum(len(c["records"]) for c in cells.values()),
             "witness_alt": sum(len(report["cells"][k]["witness_alt_rows"])
                                for k in report["cells"]),
             "abstain": sum(report["cells"][k]["outcome_counts"]
                            .get("abstain", 0) for k in report["cells"]),
             "cross_checked_cells": sum(
                 1 for k in report["cells"]
                 if report["cells"][k].get("committed_act_set_match")),
             "composition_pass_cells": sum(
                 1 for k in report["cells"]
                 if report["cells"][k]["composition"]["pass"])}
    report["totals"] = total
    report["verdict"] = ("stageC-envelope-frozen" if not kills
                         else "stageC-blocked")
    report["scope_note"] = (
        "Envelope v0.1 covers the committed W2 panel at the panel pin "
        "(pr12_3 s0 dev, pr12_4 s1/s2, pr12_7 holdout C/E). Cells without "
        "a committed truth-joined table carry the exact witness_alt "
        "multiset with expected_correct_mass=null — no truth was joined "
        "by this stage. A future Stage A panel extension appends cells "
        "as an envelope version bump; it never rewrites these entries. "
        "This artifact confers no operative force on the candidate.")

    # ---- emit (writes under pr12_8/ ONLY)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "served").mkdir(exist_ok=True)
    for key, c in sorted(cells.items()):
        fname = out_root / "served" / (
            f"served_{c['source']}_W2_{c['cell']}.csv")
        with open(fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SERVED_FIELDS)
            w.writeheader()
            w.writerows(c["records"])
    with open(out_root / "f1b_envelope.json", "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
        f.write("\n")

    print(f"VERDICT: {report['verdict']}")
    print(f"  cells={total['cells']} rows={total['rows']} "
          f"witness_alt={total['witness_alt']} abstain={total['abstain']} "
          f"composition_pass={total['composition_pass_cells']}/"
          f"{total['cells']} "
          f"act_set_cross_checked={total['cross_checked_cells']}/18")
    print("Scope (PR12_8_S1_CONTRACT_CANDIDATE.md §11): proof artifacts "
          "for a rung-1 candidate with no operative force — nothing is "
          "served to any consumer; no deployment, prompting use, "
          "promotion, ingestion, FAM-core integration, or reader-contract "
          "change; PR-10 merge-abstain remains the only certified reader "
          "contract; posture remains deferral.")
    return 1 if kills else 0


if __name__ == "__main__":
    sys.exit(main())
