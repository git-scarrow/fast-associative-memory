#!/usr/bin/env python3
"""PR-12.6 action-boundary scorer — read-only over committed artifacts.

Implements PR12_6_ACTION_BOUNDARY.md §§4-17 exactly: evaluates the
frozen act-versus-defer policy family over the pending-led
dual-presented rows of the committed PR-12.3 (s0) and PR-12.4 (s1/s2)
W1/W2 packets, one-shot AND contra cells. Analysis-only, stdlib-only
(+ subprocess-git for hash pinning). No packet is generated or
modified; the only writes are under ``pr12_6/``. ``harness_boundary_sim.py``
and ``reader_utility_score.py`` are neither imported nor touched.

Label-freedom is enforced STRUCTURALLY (§10, §16.2): policy functions
receive only a RowObs and a CellCtx, both constructed exclusively from
the policy-visible memory/audit packets before any truth file is
opened. Registry truth joins at scoring time only, after every policy
decision is recorded. Cell names, arm identifiers, file paths, seeds,
scan JSONs, and every ``pr12_5/`` file are unreachable from policy code
by construction; ``pr12_5/`` tables are opened only by the
reconciliation checker (§16.5).

F2 registration (§4): ZERO F2 policies are registered for this run —
the memo permits at most two; registering none removes the §5
split-weakness from the judged evidence entirely. Recorded in the scan
output as ``f2_registered: []``.

Train/test split (§5): s0 units are development (reported, never
gated); every gate binds on the s1/s2 test partition only. F1 policies
are parameter-free, so no fitting code path exists (§16.4 trivially
satisfied).
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PIN = "0afcb2bc4d94112fd2f2cb9a47525c6d2595c2dd"  # main @ PR-12.5 merge (§3)
BASE = Path("results/issue_failure_mode_blindness")
DESIGN_MEMO = str(BASE / "PR12_6_ACTION_BOUNDARY.md")

# §11-§14 registered gate constants (not tunable).
PRECISION_FLOOR = 0.75          # G-A1 / G-C2
COVERAGE_FLOOR = 0.25           # G-A2, aggregate one-shot test coverage
CONTRA_WRONG_CEILING = 0.05     # G-C1, per contra test unit
PER_UNIT_WRONG_CEILING = 0.10   # §14, every test unit
GLOBAL_WRONG_CEILING = 0.05     # §13, per candidate over its test units

CELLS = [  # (cell base, pr10 run-stem base, arm)
    ("pairD_oneshot", "per_probe_stale-oneshot_pairD", "oneshot"),
    ("pairB_oneshot", "per_probe_stale-oneshot_pairB", "oneshot"),
    ("pairD_contra", "per_probe_contra_pairD", "contra"),
    ("pairB_contra", "per_probe_contra_pairB", "contra"),
]
SEEDS = {"s0": ("pr12_3", "dev"), "s1": ("pr12_4", "test"),
         "s2": ("pr12_4", "test")}
CANDIDATES = ("W1", "W2")
GO_POLICIES = ("F1a", "F1b", "F1c")          # §4; comparators never GO
COMPARATORS = ("B-defer-all", "B-act-uniform", "B-act-alt")
WITNESS_BASIS = "witness co-resident (fork_witness)"


# ---------------------------------------------------------------------------
# §10 policy-visible views. RowObs/CellCtx are the ONLY objects policy code
# receives; they are built solely from packet fields. No cell name, path,
# seed, truth label, or scan-report value is present in either.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# §4 frozen policy family. Each returns None (DEFER) or {class: prob} (ACT
# by exact expectation; deterministic policies put mass 1 on one class).
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Pinned IO (G-U4-style, reused from PR-12.5)
# ---------------------------------------------------------------------------
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
            ["git", "cat-file", "blob", f"{PIN}:{relpath}"],
            cwd=repo, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError:
        kills.append({"kill": 1,
                      "label": f"no committed object {PIN}:{relpath}"})
        return None
    ok = tree == committed
    manifest[relpath] = {"sha256": hashlib.sha256(tree).hexdigest(),
                         "matches_pin": ok}
    if not ok:
        kills.append({"kill": 1, "label": f"input drifted: {relpath}"})
    return tree


def load_truth(csv_bytes: bytes) -> dict:
    rows = csv.DictReader(csv_bytes.decode().splitlines())
    return {(int(float(r["epoch"])), int(r["probe_index"])):
            (int(float(r["true_label"])), int(float(r["vote_pred_label"])))
            for r in rows}


def main() -> int:
    repo = repo_root()
    out_root = repo / BASE / "pr12_6"
    manifest, kills = {}, []
    report = {"design_memo": DESIGN_MEMO, "input_pin": PIN,
              "gate_constants": {
                  "precision_floor": PRECISION_FLOOR,
                  "coverage_floor": COVERAGE_FLOOR,
                  "contra_wrong_ceiling": CONTRA_WRONG_CEILING,
                  "per_unit_wrong_ceiling": PER_UNIT_WRONG_CEILING,
                  "global_wrong_ceiling": GLOBAL_WRONG_CEILING},
              "policies": list(POLICIES), "go_policies": list(GO_POLICIES),
              "comparators": list(COMPARATORS), "f2_registered": [],
              "f2_note": "zero of the permitted two F2 policies registered "
                         "(§4); no fitting code path exists, so §16.4 is "
                         "structurally satisfied",
              "split": {"dev": "s0", "test": ["s1", "s2"],
                        "note": "gates bind on test only (§5)"},
              "s13_interpretation_note":
                  "§13's text says 'across all 16 test units ... per "
                  "candidate'; verdicts are per (candidate,policy), so the "
                  "binding gate is computed over that candidate's 8 test "
                  "units, and the pooled 16-unit figure is also reported. "
                  "Observed-ambiguity resolution, recorded, not a "
                  "threshold change.",
              "units": {}, "cell_context_features": {}, "gates": {},
              "exchange": {}, "input_manifest": manifest,
              "kill_conditions": kills}

    def dirs_clean():
        r = subprocess.run(
            ["git", "status", "--porcelain", "--",
             str(BASE / "pr12_3"), str(BASE / "pr12_4"),
             str(BASE / "pr12_5"), str(BASE / "pr10")],
            cwd=repo, capture_output=True, check=True)
        return r.stdout.decode().strip() == ""
    report["s16_6_dirs_clean_before"] = dirs_clean()
    if not report["s16_6_dirs_clean_before"]:
        kills.append({"kill": 6, "label": "committed dirs dirty before run"})

    # pr12_5 reconciliation targets (§3/§16.5; NEVER policy-visible)
    ru_raw = verify_input(
        repo, str(BASE / "pr12_5" / "reader_utility_scan.json"),
        manifest, kills)
    ru = json.loads(ru_raw) if ru_raw is not None else None

    def one_pass():
        units = {}
        for cand in CANDIDATES:
            for cell_base, stem_base, arm in CELLS:
                for seed, (scandir, part) in SEEDS.items():
                    cell = f"{cell_base}_{seed}"
                    unit = f"{cand}:{cell}"
                    # ---- policy-visible inputs ONLY (§3)
                    pkt = verify_input(
                        repo, str(BASE / scandir / cand / cell
                                  / "memory_packet.jsonl"),
                        manifest, kills)
                    aud = verify_input(
                        repo, str(BASE / scandir / cand / cell
                                  / "audit_packet.jsonl"),
                        manifest, kills)
                    if pkt is None or aud is None:
                        continue
                    ctx = CellCtx([json.loads(line)
                                   for line in aud.decode().splitlines()])
                    # ---- decisions BEFORE any truth file is read (§10)
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
                    csv_rel = (f"{BASE}/pr10/governed/"
                               f"{stem_base}_{seed}.csv")
                    csv_b = verify_input(repo, csv_rel, manifest, kills)
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
                                   "part": part, "ctx": ctx, "rows": rows}
        return units
    units = one_pass()

    # ---- §16.5 reconciliation vs committed pr12_5 (row-level + n_act)
    if ru is not None:
        for unit, u in units.items():
            committed_n = ru["units"].get(unit, {}).get("n_act")
            if committed_n != len(u["rows"]):
                kills.append({"kill": 5, "label":
                              f"{unit}: n_act {len(u['rows'])} != committed "
                              f"{committed_n}"})
            tbl_rel = str(BASE / "pr12_5"
                          / f"rows_{unit.replace(':', '_')}.csv")
            tbl = verify_input(repo, tbl_rel, manifest, kills)
            if tbl is None:
                continue
            committed_rows = {r["query_id"]: r for r in
                              csv.DictReader(tbl.decode().splitlines())}
            for r in u["rows"]:
                cr = committed_rows.get(r["query_id"])
                obs, t = r["obs"], r["truth"]
                if cr is None or t is None:
                    kills.append({"kill": 5, "label":
                                  f"{unit}: {r['query_id']} absent from "
                                  f"committed pr12_5 table"})
                    continue
                ok = (int(cr["truth_in_set"]) == int(t in obs.presented)
                      and int(cr["truth_in_alts"])
                      == int(t in set(obs.alt_classes))
                      and int(cr["deployed_correct"])
                      == int(t == obs.deployed_class)
                      and int(cr["width"]) == obs.width)
                if not ok:
                    kills.append({"kill": 5, "label":
                                  f"{unit}: {r['query_id']} truth "
                                  f"recomputation mismatch vs pr12_5"})

    # ---- per-(unit, policy) accounting (§§6-9, §15)
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
            "arm": u["arm"], "partition": u["part"],
            "n_rows": len(u["rows"]),
            "policies": metrics[unit]}
        report["cell_context_features"][unit] = {
            "source": "audit_packet.jsonl record counts (reader-visible "
                      "structural records only; §10)",
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

    # ---- gates (§§11-14), test partition only
    combo_pass = {}
    for cand in CANDIDATES:
        test_units = [un for un, u in units.items()
                      if u["cand"] == cand and u["part"] == "test"]
        for pname in GO_POLICIES:
            key = f"{cand}:{pname}"
            gates = {}
            ok = True
            tot_wrong = tot_rows = 0.0
            os_acted = os_rows = 0.0
            contra_included = []
            for un in sorted(test_units):
                m = metrics[un][pname]
                arm = units[un]["arm"]
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
                                 "pass": m["wrong_mass_rate"]
                                 <= CONTRA_WRONG_CEILING}
                    g["G-C2"] = {"precision": m["precision"],
                                 "floor": PRECISION_FLOOR,
                                 "pass": m["acted_rows"] == 0
                                 or m["precision"] >= PRECISION_FLOOR}
                g["S14_per_unit_ceiling"] = {
                    "wrong_mass_rate": m["wrong_mass_rate"],
                    "ceiling": PER_UNIT_WRONG_CEILING,
                    "pass": m["wrong_mass_rate"] <= PER_UNIT_WRONG_CEILING}
                gates[un] = g
                ok &= all(v["pass"] for v in g.values())
            cov = os_acted / os_rows if os_rows else 0.0
            gates["G-A2_aggregate"] = {
                "oneshot_test_coverage": round(cov, 6),
                "floor": COVERAGE_FLOOR, "pass": cov >= COVERAGE_FLOOR}
            glob = tot_wrong / tot_rows if tot_rows else 0.0
            gates["S13_global_ceiling"] = {
                "global_wrong_mass_rate": round(glob, 6),
                "ceiling": GLOBAL_WRONG_CEILING,
                "pass": glob <= GLOBAL_WRONG_CEILING,
                "units_counted": len(test_units)}
            gates["G-C3_no_collapse"] = {
                "contra_test_units_included": sorted(contra_included),
                "expected": 4, "pass": len(contra_included) == 4}
            ok &= gates["G-A2_aggregate"]["pass"] \
                and gates["S13_global_ceiling"]["pass"] \
                and gates["G-C3_no_collapse"]["pass"]
            report["gates"][key] = gates
            combo_pass[key] = ok
    pooled_wrong = sum(metrics[un][p]["wrong_mass"]
                       for un, u in units.items() if u["part"] == "test"
                       for p in GO_POLICIES if p == "F1a")
    report["s13_pooled_16_unit_reference"] = {
        p: round(sum(metrics[un][p]["wrong_mass"] for un, u in units.items()
                     if u["part"] == "test")
                 / sum(len(u["rows"]) for u in units.values()
                       if u["part"] == "test"), 6)
        for p in GO_POLICIES}
    del pooled_wrong

    # ---- §16.6 epilogue + §17 internal double pass
    report["s16_6_dirs_clean_after"] = dirs_clean()
    if not report["s16_6_dirs_clean_after"]:
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

    # ---- verdict (§19)
    if kills:
        report["verdict"] = "action-boundary-blocked"
    else:
        passing = sorted(k for k, v in combo_pass.items() if v)
        report["combo_pass"] = combo_pass
        report["verdict"] = (
            f"action-boundary-evidence-GO({','.join(passing)})"
            if passing else "action-boundary-negative")

    # ---- emit (§18; writes under pr12_6/ ONLY)
    out_root.mkdir(parents=True, exist_ok=True)
    fields = ["query_id", "width", "all_witness", "led_never_resolving",
              "decision", "acted_class_mass", "expected_correct"]
    for unit, u in sorted(units.items()):
        if u["part"] != "test":
            continue
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
    with open(out_root / "action_boundary_scan.json", "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
        f.write("\n")

    print(f"VERDICT: {report['verdict']}")
    for unit in sorted(units):
        if units[unit]["part"] != "test":
            continue
        m = metrics[unit]
        print(f"  [{unit}] ({units[unit]['arm']}) n={len(units[unit]['rows'])} "
              + "  ".join(
                  f"{p}: cov={m[p]['coverage']} prec={m[p]['precision']} "
                  f"wrongmass={m[p]['wrong_mass_rate']}"
                  for p in GO_POLICIES))
    print("Scope (PR12_6_ACTION_BOUNDARY.md §20): offline action-boundary "
          "evidence only — no deployment, prompting use, promotion, memory "
          "ingestion, autonomous downstream use, acting authorization, or "
          "reader-contract change; PR-10 merge-abstain remains the only "
          "certified reader contract; operational posture remains deferral.")
    return 1 if kills else 0


if __name__ == "__main__":
    sys.exit(main())
