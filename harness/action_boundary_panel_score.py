#!/usr/bin/env python3
"""PR-12.8 Stage A — traffic-axis panel scorer (read-only).

Scores the byte-frozen W2:F1b policy over the committed Stage A panel
cache (`pr12_8_panel_cache/`: mixed and plain-stale, pairs B–E, seeds
s1/s2) under the PR-12.7 G-H gate structure, per
`PR12_8_READER_CONTRACT_CANDIDACY.md` §7 Stage A / §8 G-R3–G-R5.
Validity gates first; per-TYPE power floors (≥30 in-scope W2 rows
aggregated per traffic type, else that type is `panel-insufficient` —
an honest non-result carried as a named scope bound, never a pass).
The registered jitter (pairA) and g5twin (pairD) report-only anchors
are `panel-insufficient` by pre-flight determination (committed stems
exist only at s0; no pr4 geometry-table governance key exists for
either arm) and are recorded as such without generation or scoring.

Guard-regime measurement (§8 G-R4): every panel cell's
(n_contradiction_pairs, n_ambiguous_pairs) margin is recorded; any
boundary-regime cell (|margin| ≤ 4) is flagged for per-row reporting.

This is Stage A scoring, NOT Stage E adjudication: it emits a
`stage_a_status`, never a §12 candidacy verdict, and never a GO.
Analysis-only, stdlib + subprocess-git; the frozen emitter is imported
read-only solely for the G-H2 regeneration proof; writes only under
``pr12_8/`` and temp. Label-freedom is structural (policy code receives
only RowObs/CellCtx built from cache packets; truth joins
post-decision). The §5 policy block is copied VERBATIM from the frozen
``action_boundary_score.py`` and sha-attested at runtime. No
deployment, live acting, prompting use, promotion, ingestion, FAM-core
change, or reader-contract change; PR-10 merge-abstain remains the
only certified reader contract; posture remains deferral.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BASE = Path("results/issue_failure_mode_blindness")
DESIGN_MEMO = str(BASE / "PR12_8_READER_CONTRACT_CANDIDACY.md")
FROZEN_SCORER = Path("harness/action_boundary_score.py")
FROZEN_EMITTER = Path("harness/harness_boundary_sim.py")
POLICY_JSON = Path("harness/harness_policy.json")
CACHE = BASE / "pr12_8_panel_cache"
OUT_DIR = BASE / "pr12_8"

CACHE_PIN = "f3f5304a3fcdc38227d9a26ffd301156efc1d9ad"
EMITTER_SHA256_PIN = \
    "2539686a205fa03ba88fb4e222043720dc6acf97460f896a31bd58d1a11d32e5"

POWER_MIN_W2_ROWS_PER_TYPE = 30   # PR-12.8 §7 Stage A registered floor
BOUNDARY_REGIME_MARGIN = 4        # §8 G-R4: |n_contra - n_amb| <= 4

PANEL_CELLS = [  # (cell, pr10 run-stem base, traffic type)
    ("pairB_mixed_s1", "per_probe_mixed_pairB_s1", "mixed"),
    ("pairB_mixed_s2", "per_probe_mixed_pairB_s2", "mixed"),
    ("pairC_mixed_s1", "per_probe_mixed_pairC_s1", "mixed"),
    ("pairC_mixed_s2", "per_probe_mixed_pairC_s2", "mixed"),
    ("pairD_mixed_s1", "per_probe_mixed_pairD_s1", "mixed"),
    ("pairD_mixed_s2", "per_probe_mixed_pairD_s2", "mixed"),
    ("pairE_mixed_s1", "per_probe_mixed_pairE_s1", "mixed"),
    ("pairE_mixed_s2", "per_probe_mixed_pairE_s2", "mixed"),
    ("pairB_stale_s1", "per_probe_stale_pairB_s1", "stale"),
    ("pairB_stale_s2", "per_probe_stale_pairB_s2", "stale"),
    ("pairC_stale_s1", "per_probe_stale_pairC_s1", "stale"),
    ("pairC_stale_s2", "per_probe_stale_pairC_s2", "stale"),
    ("pairD_stale_s1", "per_probe_stale_pairD_s1", "stale"),
    ("pairD_stale_s2", "per_probe_stale_pairD_s2", "stale"),
    ("pairE_stale_s1", "per_probe_stale_pairE_s1", "stale"),
    ("pairE_stale_s2", "per_probe_stale_pairE_s2", "stale"),
]
UNGENERATED_TYPES = {
    "jitter": "committed stems exist only at s0 (dev exposure; excluded) "
              "and the pr4 geometry table has no 'jitter' governance arm "
              "— not generable under the registered provenance mechanism",
    "g5twin": "committed stem exists only at s0 (dev exposure; excluded) "
              "and the pr4 geometry table has no 'g5twin' governance arm "
              "— not generable under the registered provenance mechanism",
}
CANDIDATES = ("W1", "W2")
GO_POLICIES = ("F1a", "F1b", "F1c")   # context; gates evaluated for W2:F1b
GATED_COMBO = "W2:F1b"

# ---------------------------------------------------------------------------
# §5 frozen policy family — copied VERBATIM from action_boundary_score.py
# (sha-attested at runtime; any divergence aborts). Do not edit anything
# between here and the end of the POLICIES table.
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


def verify_input(repo: Path, relpath: str, pin: str, manifest: dict,
                 kills: list):
    p = repo / relpath
    try:
        tree = p.read_bytes()
    except FileNotFoundError:
        kills.append({"kill": 1, "label": f"missing input {relpath}"})
        return None
    try:
        committed = subprocess.run(
            ["git", "cat-file", "blob", f"{pin}:{relpath}"],
            cwd=repo, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError:
        kills.append({"kill": 1,
                      "label": f"no committed object {pin}:{relpath}"})
        return None
    ok = tree == committed
    manifest[relpath] = {"sha256": hashlib.sha256(tree).hexdigest(),
                         "matches_pin": ok, "pin": pin[:7]}
    if not ok:
        kills.append({"kill": 1, "label": f"input drifted: {relpath}"})
    return tree


def load_truth(csv_bytes: bytes) -> dict:
    rows = csv.DictReader(csv_bytes.decode().splitlines())
    return {(int(float(r["epoch"])), int(r["probe_index"])):
            (int(float(r["true_label"])), int(float(r["vote_pred_label"])))
            for r in rows}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    repo = repo_root()
    out_root = repo / OUT_DIR
    manifest, kills = {}, []
    report = {"design_memo": DESIGN_MEMO,
              "policy_pin": PIN, "cache_pin": CACHE_PIN,
              "gate_constants": {
                  "precision_floor": PRECISION_FLOOR,
                  "per_unit_wrong_ceiling": PER_UNIT_WRONG_CEILING,
                  "global_wrong_ceiling": GLOBAL_WRONG_CEILING,
                  "power_min_w2_rows_per_type": POWER_MIN_W2_ROWS_PER_TYPE,
                  "boundary_regime_margin": BOUNDARY_REGIME_MARGIN},
              "gate_note": "Panel types are neither one-shot nor contra "
                           "arms: the registered per-unit gates are the "
                           "universal §12.6 constants (precision floor "
                           "when acting; §15 per-unit 0.10 ceiling; §14 "
                           "global 0.05 over the powered panel units). "
                           "The one-shot coverage floor and contra 0.05 "
                           "per-unit ceiling do not apply; the 0.05 "
                           "figure is reported per unit for context.",
              "policies": list(POLICIES), "gated_combo": GATED_COMBO,
              "f2_registered": [],
              "ungenerated_types": UNGENERATED_TYPES,
              "validity_gates": {}, "units": {},
              "cell_context_features": {}, "guard_regime": {},
              "gates": {}, "type_status": {},
              "input_manifest": manifest, "kill_conditions": kills}

    def dirs_clean():
        r = subprocess.run(
            ["git", "status", "--porcelain", "--",
             str(BASE / "pr12"), str(BASE / "pr12_1"), str(BASE / "pr12_2"),
             str(BASE / "pr12_3"), str(BASE / "pr12_4"),
             str(BASE / "pr12_5"), str(BASE / "pr12_6"),
             str(BASE / "pr12_7"), str(BASE / "pr12_7_holdout_cache"),
             str(BASE / "pr10"), str(CACHE),
             str(FROZEN_EMITTER), str(FROZEN_SCORER), str(POLICY_JSON),
             "harness/action_boundary_holdout_score.py",
             "harness/reader_utility_score.py"],
            cwd=repo, capture_output=True, check=True)
        lines = [ln for ln in r.stdout.decode().splitlines()
                 if not ln.endswith(".DS_Store")]
        return "\n".join(lines) == ""
    report["dirs_clean_before"] = dirs_clean()
    if not report["dirs_clean_before"]:
        kills.append({"kill": 6, "label": "committed dirs dirty before run"})

    # ---- G-H3 / G-R7: no-tuning attestation
    try:
        mine = extract_policy_block(
            (repo / "harness" / "action_boundary_panel_score.py")
            .read_text())
        frozen = extract_policy_block((repo / FROZEN_SCORER).read_text())
        sha_mine = hashlib.sha256(mine.encode()).hexdigest()
        gh3_ok = mine == frozen
    except ValueError as e:
        sha_mine, gh3_ok = None, False
        kills.append({"kill": 3, "label": f"G-H3 extraction failed: {e}"})
    report["validity_gates"]["G-H3"] = {
        "copied_block_sha256": sha_mine,
        "frozen_source": str(FROZEN_SCORER), "pass": gh3_ok}
    if not gh3_ok:
        kills.append({"kill": 3, "label": "G-H3: policy block != frozen"})

    # ---- G-H1: unexposedness of the 16 panel cells (exact stem strings)
    stems = [stem for _, stem, _ in PANEL_CELLS]
    exposure_hits = []
    policy_cfg = json.loads((repo / POLICY_JSON).read_text())
    for block in ("cells", "scan", "scan12_2", "scan12_3", "scan12_4",
                  "scan12_7_holdout"):
        cells_cfg = policy_cfg[block] if block == "cells" \
            else policy_cfg[block]["cells"]
        blob = json.dumps(cells_cfg)
        for stem in stems:
            if stem in blob:
                exposure_hits.append(
                    {"where": f"harness_policy.json:{block}", "match": stem})
    for scandir in ("pr12_1", "pr12_2", "pr12_3", "pr12_4",
                    "pr12_5", "pr12_6", "pr12_7"):
        droot = repo / BASE / scandir
        if not droot.exists():
            continue
        for p in sorted(droot.rglob("*.json")):
            data = p.read_text()
            for stem in stems:
                if stem in data:
                    exposure_hits.append(
                        {"where": str(p.relative_to(repo)), "match": stem})
    report["validity_gates"]["G-H1"] = {
        "match_set": exposure_hits, "pass": not exposure_hits,
        "note": "the 16 panel run-stems are absent from every 12.1-12.7 "
                "scan config block and every committed pr12_1..pr12_7 "
                "aggregate JSON (scan12_8_panel itself excluded as the "
                "panel's own manifest)"}
    if exposure_hits:
        kills.append({"kill": 4, "label": f"G-H1 exposure: "
                                          f"{exposure_hits[:3]}..."})

    # ---- G-H2: cache regenerates byte-identically (unmodified emitter)
    emitter_sha_before = sha256_file(repo / FROZEN_EMITTER)
    gh2 = {"emitter_sha256_before": emitter_sha_before,
           "emitter_pin": EMITTER_SHA256_PIN, "cells": {}}
    if emitter_sha_before != EMITTER_SHA256_PIN:
        kills.append({"kill": 1, "label": "G-H2: emitter sha != pin"})
    sys.path.insert(0, str(repo / "harness"))
    from harness_boundary_sim import run_cell  # frozen import, read-only
    scan = policy_cfg["scan12_8_panel"]
    tmp = Path(tempfile.mkdtemp(prefix="pr12_8_panel_gh2_"))
    gh2_ok = True
    try:
        for shape in scan["shapes"]:
            for name, cfg in scan["cells"].items():
                run_cell(repo, name, cfg, policy_cfg, False,
                         out_root=tmp / shape, shape=shape,
                         policy_version=scan["policy_version"],
                         emit_review_queue=True, emit_ambiguous_queue=True)
                ok = True
                for fn in ("memory_packet.jsonl", "audit_packet.jsonl",
                           "decision_table.csv"):
                    ok &= (tmp / shape / name / fn).read_bytes() == \
                        (repo / CACHE / shape / name / fn).read_bytes()
                gh2["cells"][f"{shape}/{name}"] = ok
                gh2_ok &= ok
    finally:
        shutil.rmtree(tmp)
    gh2["emitter_sha256_after"] = sha256_file(repo / FROZEN_EMITTER)
    gh2["pass"] = (gh2_ok
                   and gh2["emitter_sha256_after"] == EMITTER_SHA256_PIN)
    report["validity_gates"]["G-H2"] = gh2
    if not gh2["pass"]:
        kills.append({"kill": 1, "label": "G-H2: panel cache does not "
                                          "regenerate byte-identically"})

    # ---- decision + join pass (12.6/12.7 mechanism)
    def one_pass():
        units = {}
        for cand in CANDIDATES:
            for cell, stem_base, ttype in PANEL_CELLS:
                unit = f"{cand}:{cell}"
                pkt = verify_input(
                    repo, str(CACHE / cand / cell / "memory_packet.jsonl"),
                    CACHE_PIN, manifest, kills)
                aud = verify_input(
                    repo, str(CACHE / cand / cell / "audit_packet.jsonl"),
                    CACHE_PIN, manifest, kills)
                if pkt is None or aud is None:
                    continue
                ctx = CellCtx([json.loads(line)
                               for line in aud.decode().splitlines()])
                rows = []
                for line in pkt.decode().splitlines():
                    rec = json.loads(line)
                    ties = [it for it in rec.get("items", [])
                            if it.get("type") == "unresolved_tie"]
                    if not ties:
                        continue
                    obs = RowObs(cand, ties[0])
                    decisions = {}
                    for pname, pf in POLICIES.items():
                        act = pf(obs, ctx)
                        if act is not None and \
                                not set(act) <= set(obs.presented):
                            kills.append({"kill": 5, "label":
                                          f"{unit}/{pname}: ACT outside "
                                          f"presented set "
                                          f"({rec['query_id']})"})
                            act = None
                        decisions[pname] = act
                    rows.append({"query_id": rec["query_id"],
                                 "obs": obs, "decisions": decisions})
                csv_rel = f"{BASE}/pr10/governed/{stem_base}.csv"
                csv_b = verify_input(repo, csv_rel, PIN, manifest, kills)
                truth = load_truth(csv_b) if csv_b is not None else {}
                for r in rows:
                    _, e, p = r["query_id"].split(":")
                    key = (int(e[1:]), int(p[1:]))
                    if key not in truth:
                        kills.append({"kill": 5, "label":
                                      f"{unit}: join miss "
                                      f"{r['query_id']}"})
                        r["truth"] = None
                        continue
                    t, vp = truth[key]
                    if r["obs"].deployed_class != vp:
                        kills.append({"kill": 5, "label":
                                      f"{unit}: deployed != "
                                      f"vote_pred ({r['query_id']})"})
                    r["truth"] = t
                units[unit] = {"cand": cand, "cell": cell, "type": ttype,
                               "ctx": ctx, "rows": rows}
        return units
    units = one_pass()

    # ---- G-H5 (structural) + G-H4 per-type power floors
    n_classified = sum(len(u["rows"]) for u in units.values())
    report["validity_gates"]["G-H5"] = {
        "rows_classified_from_packets": n_classified, "pass": True,
        "note": "structural: policy sees RowObs/CellCtx only; truth joins "
                "post-decision"}
    type_rows = {}
    for u in units.values():
        if u["cand"] == "W2":
            type_rows[u["type"]] = type_rows.get(u["type"], 0) \
                + len(u["rows"])
    power = {t: {"w2_rows_aggregated": type_rows.get(t, 0),
                 "floor": POWER_MIN_W2_ROWS_PER_TYPE,
                 "powered": type_rows.get(t, 0)
                 >= POWER_MIN_W2_ROWS_PER_TYPE}
             for t in ("mixed", "stale")}
    report["validity_gates"]["G-H4"] = power

    # ---- metrics (12.6 semantics)
    def unit_metrics(u):
        out = {}
        n = len(u["rows"])
        for pname in POLICIES:
            acted = correct = wrong = 0.0
            defer_width = []
            assert_correct_on_acted = 0.0
            for r in u["rows"]:
                act, t = r["decisions"][pname], r["truth"]
                if act is None:
                    defer_width.append(r["obs"].width)
                    continue
                acted += 1
                correct += act.get(t, 0.0)
                wrong += 1.0 - act.get(t, 0.0)
                assert_correct_on_acted += float(
                    t == r["obs"].deployed_class)
            out[pname] = {
                "n_rows": n, "acted_rows": int(acted),
                "coverage": round(acted / n, 6) if n else None,
                "correct_mass": round(correct, 6),
                "wrong_mass": round(wrong, 6),
                "wrong_mass_rate": round(wrong / n, 6) if n else None,
                "precision": round(correct / acted, 6) if acted else None,
                "assert_correct_rate_on_acted":
                    round(assert_correct_on_acted / acted, 6)
                    if acted else None}
        return out

    metrics = {unit: unit_metrics(u) for unit, u in units.items()}
    for unit, u in sorted(units.items()):
        report["units"][unit] = {
            "type": u["type"], "n_rows": len(u["rows"]),
            "policies": metrics[unit]}
        ctx = u["ctx"]
        margin = ctx.n_contradiction_pairs - ctx.n_ambiguous_pairs
        report["cell_context_features"][unit] = {
            "n_contradiction_pairs": ctx.n_contradiction_pairs,
            "n_ambiguous_pairs": ctx.n_ambiguous_pairs,
            "n_never_resolving_slots": len(ctx.never_resolving_slots),
            "hazard_tier": ctx.hazard_tier,
            "quiet_cell_guard_open": ctx.n_contradiction_pairs
            <= ctx.n_ambiguous_pairs}
        if u["cand"] == "W2":
            report["guard_regime"][u["cell"]] = {
                "margin_contra_minus_amb": margin,
                "boundary_regime": abs(margin) <= BOUNDARY_REGIME_MARGIN}

    # ---- gates: W2:F1b over the powered panel units (context: F1a/F1c)
    for cand in CANDIDATES:
        for pname in GO_POLICIES:
            key = f"{cand}:{pname}"
            gates = {}
            ok = True
            tot_wrong = tot_rows = 0.0
            for un in sorted(u for u, v in units.items()
                             if v["cand"] == cand
                             and power[v["type"]]["powered"]):
                m = metrics[un][pname]
                empty = m["n_rows"] == 0
                tot_wrong += m["wrong_mass"]
                tot_rows += m["n_rows"]
                g = {"precision_floor": {
                        "precision": m["precision"],
                        "floor": PRECISION_FLOOR,
                        "pass": m["acted_rows"] == 0
                        or m["precision"] >= PRECISION_FLOOR},
                     "S15_per_unit_ceiling": {
                        "wrong_mass_rate": m["wrong_mass_rate"],
                        "ceiling": PER_UNIT_WRONG_CEILING,
                        "pass": empty or m["wrong_mass_rate"]
                        <= PER_UNIT_WRONG_CEILING},
                     "context_contra_style_005": {
                        "wrong_mass_rate": m["wrong_mass_rate"],
                        "reference_ceiling": CONTRA_WRONG_CEILING,
                        "within": empty or m["wrong_mass_rate"]
                        <= CONTRA_WRONG_CEILING,
                        "gated": False}}
                gates[un] = g
                ok &= g["precision_floor"]["pass"] \
                    and g["S15_per_unit_ceiling"]["pass"]
            glob = tot_wrong / tot_rows if tot_rows else 0.0
            gates["S14_global_ceiling"] = {
                "global_wrong_mass_rate": round(glob, 6),
                "ceiling": GLOBAL_WRONG_CEILING,
                "pass": glob <= GLOBAL_WRONG_CEILING}
            ok &= gates["S14_global_ceiling"]["pass"]
            report["gates"][key] = gates
            if key == GATED_COMBO:
                gated_ok = ok

    # ---- per-type status
    for t in ("mixed", "stale"):
        if not power[t]["powered"]:
            report["type_status"][t] = {
                "status": "panel-insufficient",
                "reason": f"{power[t]['w2_rows_aggregated']} in-scope W2 "
                          f"rows aggregated < floor "
                          f"{POWER_MIN_W2_ROWS_PER_TYPE}; carried as a "
                          f"named scope bound",
                "structural_note": ("zero in-scope W2 rows: the traffic "
                                    "type produces no dual-present tie "
                                    "rows, so W2:F1b structurally cannot "
                                    "act on it (zero wrong-action risk, "
                                    "zero utility surface)"
                                    if power[t]["w2_rows_aggregated"] == 0
                                    else None)}
        else:
            t_units = [un for un, u in units.items()
                       if u["cand"] == "W2" and u["type"] == t]
            t_pass = all(
                all(v["pass"] for v in report["gates"][GATED_COMBO][un]
                    .values() if isinstance(v, dict) and "pass" in v)
                for un in t_units)
            report["type_status"][t] = {
                "status": "scored-pass" if t_pass else "scored-fail",
                "units": sorted(t_units)}
    for t, reason in UNGENERATED_TYPES.items():
        report["type_status"][t] = {"status": "panel-insufficient",
                                    "reason": reason}

    # ---- epilogue + internal double pass
    report["dirs_clean_after"] = dirs_clean()
    if not report["dirs_clean_after"]:
        kills.append({"kill": 6, "label": "committed dirs dirty after run"})
    units2 = one_pass()

    def snap(us):
        return json.dumps(
            {un: [{"q": r["query_id"], "t": r["truth"],
                   "d": {p: (sorted(a.items()) if a else None)
                         for p, a in r["decisions"].items()}}
                  for r in u["rows"]]
             for un, u in sorted(us.items())}, sort_keys=True)
    same = snap(units) == snap(units2)
    report["internal_double_pass_identical"] = same
    if not same:
        kills.append({"kill": 7, "label": "internal double pass differs"})

    report["stage_a_status"] = ("panel-blocked" if kills
                                else "panel-extension-scored")

    # ---- emit (writes under pr12_8/ ONLY)
    out_root.mkdir(parents=True, exist_ok=True)
    fields = ["query_id", "width", "all_witness", "led_never_resolving",
              "decision", "acted_class_mass", "expected_correct"]
    for unit, u in sorted(units.items()):
        for pname in POLICIES:
            fname = out_root / (f"rows_panel_{unit.replace(':', '_')}"
                                f"_{pname}.csv")
            with open(fname, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in u["rows"]:
                    act = r["decisions"][pname]
                    w.writerow({
                        "query_id": r["query_id"],
                        "width": r["obs"].width,
                        "all_witness": int(r["obs"].all_witness),
                        "led_never_resolving": int(
                            r["obs"].led_slot
                            in u["ctx"].never_resolving_slots),
                        "decision": "DEFER" if act is None else "ACT",
                        "acted_class_mass": "" if act is None else
                        json.dumps({str(k): round(v, 6)
                                    for k, v in sorted(act.items())}),
                        "expected_correct": "" if act is None else
                        round(act.get(r["truth"], 0.0), 6)})
    with open(out_root / "panel_scan.json", "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
        f.write("\n")

    print(f"STAGE A STATUS: {report['stage_a_status']}")
    for g in ("G-H1", "G-H2", "G-H3", "G-H5"):
        print(f"  {g}: "
              f"{'PASS' if report['validity_gates'][g]['pass'] else 'FAIL'}")
    print(f"  G-H4 power: " + "  ".join(
        f"{t}={power[t]['w2_rows_aggregated']} "
        f"({'powered' if power[t]['powered'] else 'insufficient'})"
        for t in ("mixed", "stale")))
    print("  type status: " + "  ".join(
        f"{t}:{report['type_status'][t]['status']}"
        for t in ("mixed", "stale", "jitter", "g5twin")))
    for unit in sorted(units):
        u = units[unit]
        if u["cand"] != "W2" or not power[u["type"]]["powered"] \
                or not u["rows"]:
            continue
        m = metrics[unit]["F1b"]
        print(f"  [{unit}] n={m['n_rows']} cov={m['coverage']} "
              f"prec={m['precision']} wrongmass={m['wrong_mass_rate']}")
    print("Scope: Stage A panel evidence only — no candidacy verdict, no "
          "GO, no serving, no deployment, prompting use, promotion, "
          "ingestion, FAM-core change, or reader-contract change; PR-10 "
          "merge-abstain remains the only certified reader contract; "
          "posture remains deferral.")
    return 1 if kills else 0


if __name__ == "__main__":
    sys.exit(main())
