#!/usr/bin/env python3
"""PR-12.7 Stage B — action-boundary holdout scorer (read-only).

Implements PR12_7_ACTION_BOUNDARY_HOLDOUT.md §§5-21 exactly: evaluates
the byte-frozen PR-12.6 act-versus-defer policy family over the
committed Stage A holdout cache (pairs C/E, seeds s1/s2 — the pair-axis
holdout registered in §3/§4) with the §16 holdout-validity gates
G-H1–G-H5 evaluated FIRST; any validity failure short-circuits to a
validity verdict, never to a GO. Analysis-only, stdlib-only
(+ subprocess-git for hash pinning). The only writes are under
``pr12_7/`` and temp directories. ``harness_boundary_sim.py``,
``action_boundary_score.py``, and ``reader_utility_score.py`` are never
modified; the frozen emitter is imported read-only (its sha256 pinned
and checked) solely for the G-H2 regeneration proof.

No-tuning attestation (G-H3): the §5 policy family below — the PIN and
gate constants, WITNESS_BASIS, RowObs, CellCtx, _f1a_condition, the six
policy functions, and the POLICIES table — is copied VERBATIM from the
frozen ``action_boundary_score.py``. At runtime this scorer extracts
that block from BOTH source files and requires sha256 identity; any
divergence is kill §17.3. No code path fits, selects, or thresholds
against any holdout row (all F1 policies are parameter-free; zero F2
policies are registered).

Label-freedom is enforced STRUCTURALLY (§6/§7): policy functions
receive only a RowObs and a CellCtx, both constructed exclusively from
the policy-visible cache packets before any truth file is opened.
Registry truth joins at scoring time only, after every policy decision
is recorded. No ``pr12_5/`` or ``pr12_6/`` file is read at all.
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
DESIGN_MEMO = str(BASE / "PR12_7_ACTION_BOUNDARY_HOLDOUT.md")
FROZEN_SCORER = Path("harness/action_boundary_score.py")
FROZEN_EMITTER = Path("harness/harness_boundary_sim.py")
POLICY_JSON = Path("harness/harness_policy.json")
CACHE = BASE / "pr12_7_holdout_cache"
OUT_DIR = BASE / "pr12_7"

# Pins. PIN (inside the verbatim §5 block below) is the 12.6 policy pin;
# CACHE_PIN is main @ the PR-12.7 Stage A merge, where the holdout cache
# and the scan12_7_holdout manifest are committed.
CACHE_PIN = "480cf1dd4dc4126b9f71064fe303cb21473c72f9"
EMITTER_SHA256_PIN = \
    "2539686a205fa03ba88fb4e222043720dc6acf97460f896a31bd58d1a11d32e5"

# §16 G-H4 registered power floor (not tunable).
POWER_MIN_ONESHOT_W2_ROWS = 30   # aggregated over W2 one-shot holdout units
POWER_MIN_CONTRA_W2_ROWS = 5     # >=1 contra unit at this size per pair

HOLDOUT_CELLS = [  # (cell, pr10 run-stem base, arm) — memo §4 table
    ("pairC_oneshot_s1", "per_probe_stale-oneshot_pairC_s1", "oneshot"),
    ("pairC_oneshot_s2", "per_probe_stale-oneshot_pairC_s2", "oneshot"),
    ("pairE_oneshot_s1", "per_probe_stale-oneshot_pairE_s1", "oneshot"),
    ("pairE_oneshot_s2", "per_probe_stale-oneshot_pairE_s2", "oneshot"),
    ("pairC_contra_s1", "per_probe_contra_pairC_s1", "contra"),
    ("pairC_contra_s2", "per_probe_contra_pairC_s2", "contra"),
    ("pairE_contra_s1", "per_probe_contra_pairE_s1", "contra"),
    ("pairE_contra_s2", "per_probe_contra_pairE_s2", "contra"),
    ("clean_pairC_s1", "per_probe_clean_pairC_s1", "control"),
    ("clean_pairE_s1", "per_probe_clean_pairE_s1", "control"),
]
HELD_OUT_PAIRS = ("pairC", "pairE")
CANDIDATES = ("W1", "W2")
GO_POLICIES = ("F1a", "F1b", "F1c")   # §5: only W2:F1b can receive a GO
GO_COMBO = "W2:F1b"
COMPARATORS = ("B-defer-all", "B-act-uniform", "B-act-alt")

# ---------------------------------------------------------------------------
# §5 frozen policy family — copied VERBATIM from action_boundary_score.py
# (G-H3: sha256-checked against the frozen source at runtime; kill §17.3
# on any divergence). Do not edit anything between here and the end of
# the POLICIES table.
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


# ---------------------------------------------------------------------------
# G-H3 attestation: extract the verbatim block from a source text. The
# extraction is anchor-based and applied identically to this file and to
# the frozen action_boundary_score.py; the two extractions must be
# sha256-identical.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Pinned IO (the PR-12.5/12.6 mechanism, reused; per-file pin selectable)
# ---------------------------------------------------------------------------
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
                  "coverage_floor": COVERAGE_FLOOR,
                  "contra_wrong_ceiling": CONTRA_WRONG_CEILING,
                  "per_unit_wrong_ceiling": PER_UNIT_WRONG_CEILING,
                  "global_wrong_ceiling": GLOBAL_WRONG_CEILING,
                  "power_min_oneshot_w2_rows": POWER_MIN_ONESHOT_W2_ROWS,
                  "power_min_contra_w2_rows": POWER_MIN_CONTRA_W2_ROWS},
              "policies": list(POLICIES), "go_policies": list(GO_POLICIES),
              "go_combo": GO_COMBO, "comparators": list(COMPARATORS),
              "f2_registered": [],
              "holdout_scope_note":
                  "pair-axis holdout at existing seeds s1/s2 (§3); pairs "
                  "C/E; s0 and pairs A/B/D are structurally excluded",
              "validity_gates": {}, "units": {},
              "cell_context_features": {}, "gates": {}, "exchange": {},
              "input_manifest": manifest, "kill_conditions": kills}

    def dirs_clean():
        r = subprocess.run(
            ["git", "status", "--porcelain", "--",
             str(BASE / "pr12"), str(BASE / "pr12_1"), str(BASE / "pr12_2"),
             str(BASE / "pr12_3"), str(BASE / "pr12_4"),
             str(BASE / "pr12_5"), str(BASE / "pr12_6"),
             str(BASE / "pr10"), str(CACHE),
             str(FROZEN_EMITTER), str(FROZEN_SCORER), str(POLICY_JSON),
             "harness/reader_utility_score.py"],
            cwd=repo, capture_output=True, check=True)
        lines = [ln for ln in r.stdout.decode().splitlines()
                 if not ln.endswith(".DS_Store")]
        return "\n".join(lines) == ""
    report["s17_6_dirs_clean_before"] = dirs_clean()
    if not report["s17_6_dirs_clean_before"]:
        kills.append({"kill": 6, "label": "committed dirs dirty before run"})

    # ---- G-H3: no-tuning attestation (verbatim §5 block sha identity)
    try:
        mine = extract_policy_block(
            (repo / "harness" / "action_boundary_holdout_score.py")
            .read_text())
        frozen = extract_policy_block(
            (repo / FROZEN_SCORER).read_text())
        sha_mine = hashlib.sha256(mine.encode()).hexdigest()
        sha_frozen = hashlib.sha256(frozen.encode()).hexdigest()
        gh3_ok = sha_mine == sha_frozen
    except ValueError as e:
        sha_mine = sha_frozen = None
        gh3_ok = False
        kills.append({"kill": 3, "label": f"G-H3 extraction failed: {e}"})
    report["validity_gates"]["G-H3"] = {
        "copied_block_sha256": sha_mine,
        "frozen_source_sha256": sha_frozen,
        "frozen_source": str(FROZEN_SCORER),
        "policy_pin": PIN, "pass": gh3_ok,
        "note": "parameter-free F1 policies; zero F2 registered; no "
                "fitting code path exists"}
    if not gh3_ok and sha_mine is not None:
        kills.append({"kill": 3, "label": "G-H3: copied policy block != "
                                          "frozen action_boundary_score.py"})

    # ---- G-H1: unexposedness (exposure grep re-run at scoring time)
    exposure_hits = []
    policy_cfg = json.loads((repo / POLICY_JSON).read_text())
    for block in ("cells", "scan", "scan12_2", "scan12_3", "scan12_4"):
        cells = policy_cfg[block] if block == "cells" \
            else policy_cfg[block]["cells"]
        blob = json.dumps(cells)
        for pair in HELD_OUT_PAIRS:
            if pair in blob:
                exposure_hits.append({"where": f"harness_policy.json:{block}",
                                      "match": pair})
    for scandir in ("pr12_1", "pr12_2", "pr12_3", "pr12_4",
                    "pr12_5", "pr12_6"):
        droot = repo / BASE / scandir
        if not droot.exists():
            continue
        for p in sorted(droot.rglob("*")):
            rel = str(p.relative_to(repo))
            for pair in HELD_OUT_PAIRS:
                if pair in p.name:
                    exposure_hits.append({"where": rel, "match": pair})
            if p.is_file() and p.suffix == ".json":
                data = p.read_text()
                for pair in HELD_OUT_PAIRS:
                    if pair in data:
                        exposure_hits.append({"where": rel,
                                              "match": f"content:{pair}"})
    report["validity_gates"]["G-H1"] = {
        "match_set": exposure_hits, "pass": not exposure_hits,
        "note": "holdout pairs absent from every 12.1-12.6 scan config "
                "and every committed pr12_1..pr12_6 aggregate/filename"}
    if exposure_hits:
        kills.append({"kill": 4, "label": f"G-H1 exposure: "
                                          f"{exposure_hits[:3]}..."})

    # ---- G-H2: provenance integrity (cache regenerates byte-identically
    # through the unmodified emitter; the scan_tree_bytecheck discipline)
    emitter_sha_before = sha256_file(repo / FROZEN_EMITTER)
    gh2 = {"emitter_sha256_before": emitter_sha_before,
           "emitter_pin": EMITTER_SHA256_PIN, "cells": {}}
    if emitter_sha_before != EMITTER_SHA256_PIN:
        kills.append({"kill": 1, "label": "G-H2: emitter sha != pin"})
    sys.path.insert(0, str(repo / "harness"))
    from harness_boundary_sim import run_cell  # frozen import, read-only
    scan = policy_cfg["scan12_7_holdout"]
    tmp = Path(tempfile.mkdtemp(prefix="pr12_7_gh2_"))
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
        kills.append({"kill": 1, "label": "G-H2: holdout cache does not "
                                          "regenerate byte-identically"})

    # ---- decision + join pass (§6-§10 semantics, 12.6 mechanism)
    def one_pass():
        units = {}
        for cand in CANDIDATES:
            for cell, stem_base, arm in HOLDOUT_CELLS:
                unit = f"{cand}:{cell}"
                # ---- policy-visible inputs ONLY
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
                # ---- decisions BEFORE any truth file is read (G-H5/§7)
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
                # ---- scoring join (truth enters only now)
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
                units[unit] = {"cand": cand, "cell": cell, "arm": arm,
                               "ctx": ctx, "rows": rows}
        return units
    units = one_pass()

    # ---- G-H5: feature-reconstructability (structural; every in-scope
    # row classified from packet fields alone — RowObs/CellCtx are the
    # only policy inputs, and every row above was classified without
    # exception before truth was read)
    n_classified = sum(len(u["rows"]) for u in units.values())
    report["validity_gates"]["G-H5"] = {
        "rows_classified_from_packets": n_classified,
        "pass": True,
        "note": "structural: policy functions receive only RowObs/CellCtx "
                "built from cache packets; truth joins post-decision; no "
                "pr12_5/, pr12_6/, PR-4 governance, scan-JSON, or "
                "identifier input is reachable from policy code"}

    # ---- G-H4: sufficiency / power floor (W2 in-scope rows, §16)
    w2_oneshot_rows = sum(len(u["rows"]) for u in units.values()
                          if u["cand"] == "W2" and u["arm"] == "oneshot")
    contra_by_pair = {pair: max((len(u["rows"]) for u in units.values()
                                 if u["cand"] == "W2"
                                 and u["arm"] == "contra"
                                 and u["cell"].startswith(pair)),
                                default=0)
                      for pair in HELD_OUT_PAIRS}
    powered = (w2_oneshot_rows >= POWER_MIN_ONESHOT_W2_ROWS
               and all(v >= POWER_MIN_CONTRA_W2_ROWS
                       for v in contra_by_pair.values()))
    report["validity_gates"]["G-H4"] = {
        "w2_oneshot_rows_aggregated": w2_oneshot_rows,
        "floor": POWER_MIN_ONESHOT_W2_ROWS,
        "max_contra_w2_rows_by_pair": contra_by_pair,
        "contra_floor_per_pair": POWER_MIN_CONTRA_W2_ROWS,
        "pass": powered}

    # ---- per-(unit, policy) accounting (§§8-11, §15)
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
                "deferral_rate": round(1 - acted / n, 6) if n else None,
                "correct_mass": round(correct, 6),
                "wrong_mass": round(wrong, 6),
                "wrong_mass_rate": round(wrong / n, 6) if n else None,
                "precision": round(correct / acted, 6) if acted else None,
                "assert_correct_rate_on_acted":
                    round(assert_correct_on_acted / acted, 6)
                    if acted else None,
                "deferred_mean_width": round(
                    sum(defer_width) / len(defer_width), 6)
                    if defer_width else None,
                "wrong_per_deferral_avoided":
                    round(wrong / acted, 6) if acted else None}
        return out

    metrics = {unit: unit_metrics(u) for unit, u in units.items()}
    for unit, u in sorted(units.items()):
        report["units"][unit] = {
            "arm": u["arm"], "partition":
                "holdout" if u["arm"] != "control" else "control",
            "n_rows": len(u["rows"]), "policies": metrics[unit]}
        report["cell_context_features"][unit] = {
            "source": "audit_packet.jsonl record counts (reader-visible "
                      "structural records only; §6)",
            "n_contradiction_pairs": u["ctx"].n_contradiction_pairs,
            "n_ambiguous_pairs": u["ctx"].n_ambiguous_pairs,
            "n_never_resolving_slots": len(u["ctx"].never_resolving_slots),
            "hazard_tier": u["ctx"].hazard_tier,
            "quiet_cell_guard_open": u["ctx"].n_contradiction_pairs
            <= u["ctx"].n_ambiguous_pairs}
        report["exchange"][unit] = {
            p: {"wrong_per_deferral_avoided":
                metrics[unit][p]["wrong_per_deferral_avoided"],
                "deferrals_avoided": metrics[unit][p]["acted_rows"]}
            for p in POLICIES}

    # ---- gates (§§12-15) over the 8 gated holdout units per candidate;
    # control units are report-only (never gated, never in ceilings)
    combo_pass = {}
    for cand in CANDIDATES:
        gated_units = [un for un, u in units.items()
                       if u["cand"] == cand and u["arm"] != "control"]
        for pname in GO_POLICIES:
            key = f"{cand}:{pname}"
            gates = {}
            ok = True
            tot_wrong = tot_rows = 0.0
            os_acted = os_rows = 0.0
            contra_included = []
            for un in sorted(gated_units):
                m = metrics[un][pname]
                arm = units[un]["arm"]
                empty = m["n_rows"] == 0
                tot_wrong += m["wrong_mass"]
                tot_rows += m["n_rows"]
                g = {}
                if arm == "oneshot":
                    os_acted += m["acted_rows"]
                    os_rows += m["n_rows"]
                    g["G-A1"] = {"precision": m["precision"],
                                 "floor": PRECISION_FLOOR,
                                 "pass": m["acted_rows"] == 0
                                 or m["precision"] >= PRECISION_FLOOR}
                    g["G-A3"] = {"correct_rate_on_acted":
                                 round(m["correct_mass"] / m["acted_rows"],
                                       6) if m["acted_rows"] else None,
                                 "assert_rate_on_acted":
                                 m["assert_correct_rate_on_acted"],
                                 "pass": m["acted_rows"] == 0
                                 or m["correct_mass"] / m["acted_rows"]
                                 > m["assert_correct_rate_on_acted"]}
                else:
                    contra_included.append(un)
                    g["G-C1"] = {"wrong_mass_rate": m["wrong_mass_rate"],
                                 "ceiling": CONTRA_WRONG_CEILING,
                                 "pass": empty or m["wrong_mass_rate"]
                                 <= CONTRA_WRONG_CEILING}
                    g["G-C2"] = {"precision": m["precision"],
                                 "floor": PRECISION_FLOOR,
                                 "pass": m["acted_rows"] == 0
                                 or m["precision"] >= PRECISION_FLOOR}
                g["S15_per_unit_ceiling"] = {
                    "wrong_mass_rate": m["wrong_mass_rate"],
                    "ceiling": PER_UNIT_WRONG_CEILING,
                    "pass": empty or m["wrong_mass_rate"]
                    <= PER_UNIT_WRONG_CEILING}
                gates[un] = g
                ok &= all(v["pass"] for v in g.values())
            cov = os_acted / os_rows if os_rows else 0.0
            gates["G-A2_aggregate"] = {
                "oneshot_holdout_coverage": round(cov, 6),
                "floor": COVERAGE_FLOOR, "pass": cov >= COVERAGE_FLOOR}
            glob = tot_wrong / tot_rows if tot_rows else 0.0
            gates["S14_global_ceiling"] = {
                "global_wrong_mass_rate": round(glob, 6),
                "ceiling": GLOBAL_WRONG_CEILING,
                "pass": glob <= GLOBAL_WRONG_CEILING,
                "units_counted": len(gated_units)}
            gates["G-C3_no_collapse"] = {
                "contra_holdout_units_included": sorted(contra_included),
                "expected": 4, "pass": len(contra_included) == 4}
            ok &= gates["G-A2_aggregate"]["pass"] \
                and gates["S14_global_ceiling"]["pass"] \
                and gates["G-C3_no_collapse"]["pass"]
            report["gates"][key] = gates
            combo_pass[key] = ok
    report["combo_pass"] = combo_pass

    # ---- §17.6 epilogue + §18 internal double pass
    report["s17_6_dirs_clean_after"] = dirs_clean()
    if not report["s17_6_dirs_clean_after"]:
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

    # ---- verdict (§20): validity first, then W2:F1b utility/safety
    validity_pass = all(report["validity_gates"][g]["pass"]
                        for g in ("G-H1", "G-H2", "G-H3", "G-H5"))
    if kills or not validity_pass:
        report["verdict"] = "holdout-validity-blocked"
    elif not report["validity_gates"]["G-H4"]["pass"]:
        report["verdict"] = "holdout-insufficient"
    elif combo_pass.get(GO_COMBO):
        report["verdict"] = f"holdout-validity-GO({GO_COMBO})"
    else:
        report["verdict"] = "holdout-validity-negative"

    # ---- emit (§18/§19; writes under pr12_7/ ONLY)
    out_root.mkdir(parents=True, exist_ok=True)
    fields = ["query_id", "width", "all_witness", "led_never_resolving",
              "decision", "acted_class_mass", "expected_correct"]
    for unit, u in sorted(units.items()):
        for pname in POLICIES:
            fname = out_root / (f"rows_{unit.replace(':', '_')}"
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
    with open(out_root / "holdout_scan.json", "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
        f.write("\n")

    print(f"VERDICT: {report['verdict']}")
    for g in ("G-H1", "G-H2", "G-H3", "G-H4", "G-H5"):
        print(f"  {g}: {'PASS' if report['validity_gates'][g]['pass'] else 'FAIL'}")
    for unit in sorted(units):
        if units[unit]["arm"] == "control" or units[unit]["cand"] != "W2":
            continue
        m = metrics[unit]
        print(f"  [{unit}] ({units[unit]['arm']}) n={len(units[unit]['rows'])} "
              + "  ".join(
                  f"{p}: cov={m[p]['coverage']} prec={m[p]['precision']} "
                  f"wrongmass={m[p]['wrong_mass_rate']}"
                  for p in GO_POLICIES))
    print("Scope (PR12_7_ACTION_BOUNDARY_HOLDOUT.md §21): offline holdout "
          "generalization evidence only — no deployment, live acting, "
          "prompting use, promotion, memory ingestion, autonomous "
          "downstream use, acting authorization, or reader-contract "
          "change; PR-10 merge-abstain remains the only certified reader "
          "contract; operational posture remains deferral.")
    return 1 if kills else 0


if __name__ == "__main__":
    sys.exit(main())
