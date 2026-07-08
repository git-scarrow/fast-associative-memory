#!/usr/bin/env python3
"""PR-12.5 reader-utility scorer — read-only over committed artifacts.

Implements PR12_5_READER_UTILITY.md §§4-5 exactly: scores the frozen
parameter-free decision-policy family {P-abstain, P-assert, P-uniform,
P-alt-uniform} over the pending-led dual-presented rows of the committed
PR-12.3 (s0) and PR-12.4 (s1/s2) W1/W2 memory packets. Analysis-only,
stdlib-only, darwin, no torch. **This scorer generates and modifies no
packet**: every input is byte-pinned to the committed object at main
`2226d9d` (G-U4; the docs-only PR-12.5 pre-registration merge `cc1f3a5`
added no artifact, so the pinned objects are identical at both commits),
and the only writes are under `pr12_5/`. `harness_boundary_sim.py` is
neither imported nor touched (memo §9).

Policies are label-free: they read only packet fields
(`candidates[].decode_class`, `candidates[].basis`). Registry truth
labels enter at scoring time only, exactly as in every committed scorer.
Random policies are scored by exact expectation — no RNG, so the output
is deterministic (§8; an internal double pass plus an external re-run
both verify byte identity).
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PIN = "2226d9dfb070f2c6d168a8b5e8770ef475bd104e"  # main @ PR-12.4 merge (§3)
BASE = Path("results/issue_failure_mode_blindness")
DESIGN_MEMO = str(BASE / "PR12_5_READER_UTILITY.md")
CHANCE_MARGIN = 0.15          # §5 G-U3, adopted from the PR-12.3 discipline
GATED_POLICIES = ("P-uniform", "P-alt-uniform")   # §4; comparators never gated

# (cell base, pr10 run-stem base, gated?) — §2 scope; contra report-only.
CELLS = [
    ("pairD_oneshot", "per_probe_stale-oneshot_pairD", True),
    ("pairB_oneshot", "per_probe_stale-oneshot_pairB", True),
    ("pairD_contra", "per_probe_contra_pairD", False),
    ("pairB_contra", "per_probe_contra_pairB", False),
]
SEEDS = {"s0": "pr12_3", "s1": "pr12_4", "s2": "pr12_4"}
CANDIDATES = ("W1", "W2")
SCAN_REPORT = {"pr12_3": "attribution_scan.json",
               "pr12_4": "replication_scan.json"}


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def pinned_bytes(repo: Path, relpath: str) -> bytes:
    """The committed object at the §3 pin. Missing object is a kill-1."""
    return subprocess.run(
        ["git", "cat-file", "blob", f"{PIN}:{relpath}"],
        cwd=repo, capture_output=True, check=True).stdout


def verify_input(repo: Path, relpath: str, manifest: dict, kills: list):
    """G-U4: working-tree bytes == pinned object; sha256 recorded."""
    p = repo / relpath
    try:
        tree = p.read_bytes()
    except FileNotFoundError:
        kills.append({"kill": 1, "label": f"missing input {relpath}"})
        return None
    try:
        committed = pinned_bytes(repo, relpath)
    except subprocess.CalledProcessError:
        kills.append({"kill": 1,
                      "label": f"no committed object {PIN}:{relpath}"})
        return None
    ok = tree == committed
    manifest[relpath] = {"sha256": hashlib.sha256(tree).hexdigest(),
                         "matches_pin": ok}
    if not ok:
        kills.append({"kill": 1,
                      "label": f"input drifted from pin: {relpath}"})
    return tree


def load_truth(csv_bytes: bytes) -> dict:
    """(epoch, probe_index) -> (true_label, vote_pred_label)."""
    rows = csv.DictReader(csv_bytes.decode().splitlines())
    return {(int(float(r["epoch"])), int(r["probe_index"])):
            (int(float(r["true_label"])), int(float(r["vote_pred_label"])))
            for r in rows}


def score_unit(packet_bytes: bytes, truth: dict, kills: list, unit: str):
    """§4 policy family over one unit's in-scope rows, exact expectation.
    Returns per-unit sums plus the per-row evidence table (§9)."""
    n_act = dual_wrong = 0
    exp = {"P-assert": 0.0, "P-uniform": 0.0, "P-alt-uniform": 0.0}
    width_sum = 0
    guv2 = {"rows_missing_tie_structure": 0, "rows_with_compiled_answer": 0,
            "rows_without_exactly_one_deployed": 0}
    rows_out = []
    for line in packet_bytes.decode().splitlines():
        rec = json.loads(line)
        ties = [it for it in rec.get("items", [])
                if it.get("type") == "unresolved_tie"]
        if not ties:
            continue
        # ---- G-U2 non-assertion structure on every scored row (§5)
        if len(ties) != 1 or "neither asserted" not in ties[0].get("text", ""):
            guv2["rows_missing_tie_structure"] += 1
        if any(it.get("type") == "memory_item" for it in rec["items"]):
            guv2["rows_with_compiled_answer"] += 1
        deployed_cands = [c for c in ties[0]["candidates"]
                          if c.get("basis") == "deployed vote"]
        if len(deployed_cands) != 1:
            guv2["rows_without_exactly_one_deployed"] += 1
            kills.append({"kill": 4, "label":
                          f"{unit}: row without exactly one deployed "
                          f"candidate ({rec['query_id']})"})
            continue
        # ---- row join (kill-4 on miss); deployed cross-check (§6 assumption)
        _, e, p = rec["query_id"].split(":")
        key = (int(e[1:]), int(p[1:]))
        if key not in truth:
            kills.append({"kill": 4,
                          "label": f"{unit}: join miss {rec['query_id']}"})
            continue
        true_label, vote_pred = truth[key]
        deployed = deployed_cands[0]["decode_class"]
        if deployed != vote_pred:
            kills.append({"kill": 4, "label":
                          f"{unit}: deployed != vote_pred_label "
                          f"({rec['query_id']})"})
            continue
        presented = [c["decode_class"] for c in ties[0]["candidates"]]
        pset = set(presented)
        alts = pset - {deployed}
        width = len(pset)
        # ---- §4 exact expectations, label used for scoring only
        r_assert = 1.0 if true_label == deployed else 0.0
        r_uniform = (1.0 / width) if true_label in pset else 0.0
        r_alt = ((1.0 / len(alts)) if true_label in alts else 0.0) \
            if alts else 0.0
        n_act += 1
        dual_wrong += int(true_label != deployed)
        width_sum += width
        exp["P-assert"] += r_assert
        exp["P-uniform"] += r_uniform
        exp["P-alt-uniform"] += r_alt
        rows_out.append({"query_id": rec["query_id"], "width": width,
                         "truth_in_set": int(true_label in pset),
                         "truth_in_alts": int(true_label in alts),
                         "deployed_correct": int(r_assert),
                         "P-assert": r_assert,
                         "P-uniform": round(r_uniform, 6),
                         "P-alt-uniform": round(r_alt, 6)})
    return {"n_act": n_act, "dual_wrong": dual_wrong, "exp": exp,
            "width_sum": width_sum, "g_u2": guv2, "rows": rows_out}


def main() -> int:
    repo = repo_root()
    out_root = repo / BASE / "pr12_5"
    manifest, kills = {}, []
    report = {"design_memo": DESIGN_MEMO, "input_pin": PIN,
              "chance_margin": CHANCE_MARGIN,
              "policies": ["P-abstain", "P-assert",
                           "P-uniform", "P-alt-uniform"],
              "gated_policies": list(GATED_POLICIES),
              "candidates": list(CANDIDATES),
              "units": {}, "gates": {}, "g_u5_exchange": {},
              "input_manifest": manifest, "kill_conditions": kills}

    # ---- G-U4 prologue: committed dirs clean before scoring
    def dirs_clean():
        r = subprocess.run(
            ["git", "status", "--porcelain", "--",
             str(BASE / "pr12_3"), str(BASE / "pr12_4"),
             str(BASE / "pr10")],
            cwd=repo, capture_output=True, check=True)
        return r.stdout.decode().strip() == ""
    report["g_u4_dirs_clean_before"] = dirs_clean()
    if not report["g_u4_dirs_clean_before"]:
        kills.append({"kill": 3, "label": "pr12_3/pr12_4/pr10 not clean "
                      "before scoring"})

    # ---- pinned committed scan reports: n_decode + reconciliation targets
    scans = {}
    for scandir, fname in SCAN_REPORT.items():
        raw = verify_input(repo, str(BASE / scandir / fname),
                           manifest, kills)
        scans[scandir] = json.loads(raw) if raw is not None else None

    # ---- score every unit (2 candidates x 4 cell bases x 3 seeds)
    def one_pass():
        units = {}
        for cand in CANDIDATES:
            for cell_base, stem_base, gated in CELLS:
                for seed, scandir in SEEDS.items():
                    cell = f"{cell_base}_{seed}"
                    unit = f"{cand}:{cell}"
                    pkt = verify_input(
                        repo, str(BASE / scandir / cand / cell
                                  / "memory_packet.jsonl"),
                        manifest, kills)
                    csv_rel = (f"{BASE}/pr10/governed/"
                               f"{stem_base}_{seed}.csv")
                    csv_b = verify_input(repo, csv_rel, manifest, kills)
                    if pkt is None or csv_b is None or scans[scandir] is None:
                        continue
                    res = score_unit(pkt, load_truth(csv_b), kills, unit)
                    # ---- §5 reconciliation vs committed scan totals (kill-2)
                    sc = scans[scandir]
                    committed_dual = sc["report_only"][cand][cell][
                        "width"]["dual_presented"]
                    committed_dw = sc["report_only"][cand][cell][
                        "economics"]["G-T"]["dual_wrong"]
                    if res["n_act"] != committed_dual \
                            or res["dual_wrong"] != committed_dw:
                        kills.append({"kill": 2, "label":
                                      f"{unit}: recon fail n_act "
                                      f"{res['n_act']}/{committed_dual} dw "
                                      f"{res['dual_wrong']}/{committed_dw}"})
                    res["n_decode_classes"] = sc["n_decode_classes"][cell]
                    res["gated"] = gated
                    res["cand"], res["cell"] = cand, cell
                    units[unit] = res
        return units
    units = one_pass()

    # ---- per-unit metrics, gates (§5), G-U5 exchange (report-only)
    verdict_matrix = {f"{c}:{p}": True
                      for c in CANDIDATES for p in GATED_POLICIES}
    for unit, res in sorted(units.items()):
        n = res["n_act"]
        rates = {"P-abstain": {"act_rate": 0.0, "correct": 0.0, "wrong": 0.0,
                               "correct_rate": None, "wrong_rate": None}}
        for pol in ("P-assert", "P-uniform", "P-alt-uniform"):
            cor = res["exp"][pol]
            rates[pol] = {"act_rate": 1.0, "correct": round(cor, 6),
                          "wrong": round(n - cor, 6),
                          "correct_rate": round(cor / n, 6) if n else None,
                          "wrong_rate": round((n - cor) / n, 6)
                          if n else None}
        chance = 1.0 / res["n_decode_classes"]
        gates = {"G-U2": {**res["g_u2"],
                          "pass": all(v == 0 for v in res["g_u2"].values())}}
        exch = {}
        for pol in GATED_POLICIES:
            cor, wr = res["exp"][pol], n - res["exp"][pol]
            a_cor = res["exp"]["P-assert"]
            if res["gated"] and n:
                # G-U1 both clauses recorded; for always-acting policies
                # wrong = n - correct, so they coincide (noted in §12).
                gates[f"G-U1:{pol}"] = {
                    "correct_rate": round(cor / n, 6),
                    "assert_correct_rate": round(a_cor / n, 6),
                    "wrong_rate": round(wr / n, 6),
                    "assert_wrong_rate": round((n - a_cor) / n, 6),
                    "pass": cor / n > a_cor / n and wr / n < (n - a_cor) / n}
                gates[f"G-U3:{pol}"] = {
                    "correct_rate": round(cor / n, 6),
                    "alphabet_chance": round(chance, 6),
                    "threshold": round(chance + CHANCE_MARGIN, 6),
                    "pass": cor / n >= chance + CHANCE_MARGIN}
                ok = gates[f"G-U1:{pol}"]["pass"] \
                    and gates[f"G-U3:{pol}"]["pass"]
                verdict_matrix[f"{res['cand']}:{pol}"] &= ok
            exch[pol] = {"wrong_per_correct": round(wr / cor, 6)
                         if cor else None,
                         "deferrals_avoided": n,
                         "mean_width": round(res["width_sum"] / n, 6)
                         if n else None}
        report["units"][unit] = {
            "gated": res["gated"], "n_act": n,
            "dual_wrong": res["dual_wrong"],
            "n_decode_classes": res["n_decode_classes"],
            "policies": rates}
        report["gates"][unit] = gates
        report["g_u5_exchange"][unit] = exch
        if not gates["G-U2"]["pass"]:
            kills.append({"kill": 8, "label": f"{unit}: G-U2 structure"})

    # ---- G-U4 epilogue + internal determinism double pass (§8)
    report["g_u4_dirs_clean_after"] = dirs_clean()
    if not report["g_u4_dirs_clean_after"]:
        kills.append({"kill": 3, "label": "pr12_3/pr12_4/pr10 not clean "
                      "after scoring"})
    units2 = one_pass()
    same = json.dumps({u: {k: v for k, v in r.items() if k != "rows"}
                       for u, r in units.items()}, sort_keys=True) == \
        json.dumps({u: {k: v for k, v in r.items() if k != "rows"}
                    for u, r in units2.items()}, sort_keys=True) and \
        all(units[u]["rows"] == units2[u]["rows"] for u in units)
    report["internal_double_pass_identical"] = same
    if not same:
        kills.append({"kill": 7, "label": "internal double pass differs"})

    # ---- verdict (§10)
    if kills:
        report["verdict"] = "reader-utility-blocked"
    else:
        passing = sorted(k for k, v in verdict_matrix.items() if v)
        report["combo_pass"] = verdict_matrix
        report["verdict"] = (
            f"reader-utility-evidence-GO({','.join(passing)})"
            if passing else "reader-utility-negative")

    # ---- emit (writes under pr12_5/ ONLY, §5 G-U4 / §7.3)
    out_root.mkdir(parents=True, exist_ok=True)
    for unit, res in sorted(units.items()):
        fname = out_root / f"rows_{unit.replace(':', '_')}.csv"
        with open(fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(res["rows"][0].keys())
                               if res["rows"] else
                               ["query_id", "width", "truth_in_set",
                                "truth_in_alts", "deployed_correct",
                                "P-assert", "P-uniform", "P-alt-uniform"])
            w.writeheader()
            w.writerows(res["rows"])
    with open(out_root / "reader_utility_scan.json", "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
        f.write("\n")

    print(f"VERDICT: {report['verdict']}")
    for unit in sorted(units):
        u = report["units"][unit]
        pu = u["policies"]["P-uniform"]["correct_rate"]
        pa = u["policies"]["P-assert"]["correct_rate"]
        alt = u["policies"]["P-alt-uniform"]["correct_rate"]
        print(f"  [{unit}] {'GATED' if u['gated'] else 'report'} "
              f"n={u['n_act']} assert={pa} uniform={pu} alt={alt}")
    print("Scope (PR12_5_READER_UTILITY.md §10): offline expected-value "
          "evidence over committed packets only — no prompt contract, no "
          "reader-contract change, no acting-vs-abstention claim; PR-10 "
          "merge-abstain remains the only certified reader contract.")
    return 1 if kills else 0


if __name__ == "__main__":
    sys.exit(main())
