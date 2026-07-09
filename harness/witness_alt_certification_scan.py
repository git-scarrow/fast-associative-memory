#!/usr/bin/env python3
"""PR-12.9 Stage II — certification run for the S1 batch packet-reader contract.

The certification scanner registered in
`PR12_9_READER_CONTRACT_CERTIFICATION.md` §7 Stage II (main `fe6248e`;
Stage I record §12.1/§12.2 merged at `2d513c0`/`10b9335`): evaluates
the §6 certification gates C-1–C-7 over the committed record at the
certification pin and emits ``pr12_9/certification_scan.json`` with
exactly ONE §10 verdict.

* **C-1** envelope exactness by DUAL mechanism: the (remediated)
  reference reader's ``read_cell`` and the Stage E adjudicator's
  independent recomputation mechanism (imported from the committed
  ``candidacy_adjudicate.py``, its own verbatim policy-block copy) must
  both reproduce the frozen envelope v0.2 ``witness_alt`` multisets
  exactly on all 44 cells; the emitter, policy-block, envelope, and
  packet pins re-verified byte-identical.
* **C-2** composition re-proof (I1–I4) recomputed on every envelope
  cell; zero incumbent-field deviations; PR-10-served fields untouched.
* **C-3** the §5.1 conformance suite re-run: every test must pass; any
  specification ambiguity forces ``certification-insufficient``.
* **C-4** the §5.2 withdrawal demo re-run: every T1–T7 injection must
  produce the registered behavior and a well-formed hash-chained event.
* **C-5** bound carriage: the verdict text embeds the registration's §3
  verbatim (extracted from the PINNED memo blob, not the working tree)
  and cites the PR-12.8 verdict by its exact name.
* **C-6** determinism: internal double pass here; the external second
  invocation is performed and recorded by the run procedure (§12.3).
  The withdrawal demo's event log is the registered append-only
  exclusion (§9) — nothing from it that varies across runs is embedded.
* **C-7** approval separation: this run changes NOTHING. Even a GO
  verdict rests on its branch with no effect; the registry sentence
  (§4.2) and posture (§4.3) change only upon explicit human approval
  of the merge (Stage III, separately unauthorized).

Read-only over committed artifacts; stdlib + subprocess-git; writes
ONLY under ``results/issue_failure_mode_blindness/pr12_9/`` (this scan,
plus the registered outputs of the §5.1/§5.2 artifacts it invokes).
No FAM-core import. **Nothing is certified-with-effect, served,
deployed, promoted, or ingested by running this program.** PR-10
merge-abstain remains the only certified reader contract; the
operational posture on witness-window rows remains deferral.
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

import candidacy_adjudicate as adj          # noqa: E402  Stage E mechanism
import witness_alt_conformance_tests as ct  # noqa: E402  §5.1 artifact
import witness_alt_reference_reader as wr   # noqa: E402  reference reader

BASE = Path("results/issue_failure_mode_blindness")
OUT_DIR = REPO / BASE / "pr12_9"
OUT_FILE = OUT_DIR / "certification_scan.json"
MEMO = str(BASE / "PR12_9_READER_CONTRACT_CERTIFICATION.md")

# The certification pin: main at the Stage I remediation merge — every
# consumed input below is committed at this commit.
CERTIFICATION_PIN = "10b9335f3bf00b3a306e3dfeb869eea95b3842d3"

REGISTERED_BLOCK_SHA = \
    "2f009cf29ce64dc21e7bd392ceac4bbc189bc5b197489c2edaf8cd3ed1022c15"
EMITTER_SHA = \
    "2539686a205fa03ba88fb4e222043720dc6acf97460f896a31bd58d1a11d32e5"
CITED_VERDICT = "contract-candidate-GO-seedbounded(W2:F1b)"
GO_NAME = ("reader-contract-certified-seedbounded"
           "(s1-witness-alt-batch@1.0, W2:F1b)")


def pin_blob(relpath: str) -> bytes:
    return subprocess.run(
        ["git", "cat-file", "blob", f"{CERTIFICATION_PIN}:{relpath}"],
        cwd=REPO, capture_output=True, check=True).stdout


def verify_pin(relpath: str, manifest: dict, kills: list):
    """Working-tree bytes must equal the blob at the certification pin."""
    try:
        tree = (REPO / relpath).read_bytes()
        committed = pin_blob(relpath)
    except (FileNotFoundError, subprocess.CalledProcessError) as ex:
        kills.append({"kill": 3, "label": f"{relpath}: {type(ex).__name__}"})
        return None
    ok = tree == committed
    manifest[relpath] = {"sha256": hashlib.sha256(tree).hexdigest(),
                         "matches_certification_pin": ok}
    if not ok:
        kills.append({"kill": 3, "label": f"pin drift: {relpath}"})
    return tree


def extract_sec3(memo_bytes: bytes) -> str:
    """The registration's §3 body, verbatim, from the pinned memo blob."""
    lines = memo_bytes.decode().splitlines()
    start = next(i for i, ln in enumerate(lines)
                 if ln.startswith("## 3. Evidence base"))
    end = next(i for i, ln in enumerate(lines)
               if ln.startswith("## 4. What certification"))
    return "\n".join(lines[start:end]).rstrip()


def adjudicator_recompute(pkt: bytes, aud: bytes):
    """The Stage E adjudicator's G-R1 mechanism, re-run verbatim: derive
    the cell's witness_alt multiset from the packets through the
    adjudicator's OWN sha-attested policy-block copy."""
    ctx = adj.CellCtx([json.loads(line)
                       for line in aud.decode().splitlines()])
    recomputed, abstain, eligible = [], set(), set()
    for line in pkt.decode().splitlines():
        rec = json.loads(line)
        items = rec.get("items", [])
        if any(it.get("type") == "abstention_notice" for it in items):
            abstain.add(rec["query_id"])
            continue
        ties = [it for it in items if it.get("type") == "unresolved_tie"]
        if not ties:
            continue
        act = adj.pol_f1b(adj.RowObs("W2", ties[0]), ctx)
        if act is not None:
            recomputed.append({"query_id": rec["query_id"],
                               "class": sorted(act)[0]})
            eligible.add(rec["query_id"])
    return recomputed, abstain, eligible


def c1_c2_pass(env, manifest, kills):
    """C-1 (dual-mechanism envelope exactness) and C-2 (composition
    re-proof) over every envelope cell. Returns the two gate records."""
    reader_exact, adjud_exact, comp = {}, {}, {}
    incumbent_dev, i4_bad, fail_closed = 0, 0, 0
    for key in sorted(env["cells"]):
        source, _, cell = key.split("/")
        pkt = verify_pin(f"{BASE}/{source}/W2/{cell}/memory_packet.jsonl",
                         manifest, kills)
        aud = verify_pin(f"{BASE}/{source}/W2/{cell}/audit_packet.jsonl",
                         manifest, kills)
        if pkt is None or aud is None:
            reader_exact[key] = adjud_exact[key] = comp[key] = False
            continue
        frozen_rows = env["cells"][key]["witness_alt_rows"]

        # mechanism 1: the (remediated) reference reader
        anomalies = []
        _ctx, records, abstain_qids, eligible_qids = wr.read_cell(
            pkt, aud, anomalies, key)
        mine = [{"query_id": r["query_id"], "class": r["witness_alt_class"]}
                for r in records if r["served_outcome"] == "witness_alt"]
        reader_exact[key] = mine == frozen_rows

        # mechanism 2: the Stage E adjudicator, re-run
        recomputed, abstain2, eligible2 = adjudicator_recompute(pkt, aud)
        adjud_exact[key] = recomputed == frozen_rows

        # C-2: composition re-proof on the reader-mechanism record
        counts = {}
        for r in records:
            counts[r["served_outcome"]] = counts.get(
                r["served_outcome"], 0) + 1
        i1_bad = [a for a in anomalies if "abstention row" in a["anomaly"]]
        fail_closed += sum(1 for a in anomalies if a.get("fail_closed"))
        i4_here = [r for r in records
                   if (r["served_outcome"] == "abstain")
                   != (r["certification_tier"] == "core-certified")]
        i4_bad += len(i4_here)
        inc_here = [r for r in records if r["served_outcome"] == "abstain"
                    and r["abstain_reason"] != "merge_suspect_led"]
        incumbent_dev += len(inc_here)
        comp[key] = (not (abstain_qids & eligible_qids)
                     and not (abstain2 & eligible2)
                     and not i1_bad and not i4_here and not inc_here
                     and counts == env["cells"][key]["outcome_counts"]
                     and env["cells"][key]["composition"]["pass"])
        if not reader_exact[key] or not adjud_exact[key]:
            kills.append({"kill": 3, "label": f"C-1 mismatch: {key}"})
    c1 = {"cells": len(reader_exact),
          "reference_reader_exact_all": all(reader_exact.values()),
          "adjudicator_recompute_exact_all": all(adjud_exact.values()),
          "pass": all(reader_exact.values()) and all(adjud_exact.values())}
    c2 = {"cells": len(comp),
          "composition_recomputed_pass_all": all(comp.values()),
          "incumbent_field_deviations": incumbent_dev,
          "i4_tier_violations": i4_bad,
          "fail_closed_anomalies_on_committed_cells": fail_closed,
          "pass": all(comp.values()) and incumbent_dev == 0
          and i4_bad == 0 and fail_closed == 0}
    return c1, c2, reader_exact


def main() -> int:
    manifest, kills = {}, []
    report = {"artifact": "PR-12.9 §5.3 certification scan (Stage II)",
              "registration": MEMO,
              "certification_pin": CERTIFICATION_PIN,
              "policy_pin": adj.PIN,
              "evidence_base": f"PR-12.8 Stage E at a0e621d, verdict "
                               f"{CITED_VERDICT}",
              "gates": {}, "input_manifest": manifest,
              "kill_conditions": kills}

    frozen_before = ct.frozen_clean()
    report["frozen_surfaces_clean_before"] = frozen_before
    if not frozen_before:
        kills.append({"kill": 6, "label": "frozen surface dirty before run"})

    # ---- pins (C-1 second clause; kill §8.2/§8.3)
    att_ok, block_sha = ct.attest()          # also sets wr.BLOCK_SHA
    emitter = verify_pin("harness/harness_boundary_sim.py", manifest, kills)
    emitter_ok = (emitter is not None
                  and hashlib.sha256(emitter).hexdigest() == EMITTER_SHA)
    block_shas = {}
    try:
        for rel in adj.ATTESTED_SOURCES:
            block_shas[rel] = hashlib.sha256(adj.extract_policy_block(
                (REPO / rel).read_text()).encode()).hexdigest()
        blocks_ok = (set(block_shas.values()) == {REGISTERED_BLOCK_SHA})
    except ValueError as e:
        blocks_ok = False
        kills.append({"kill": 2, "label": f"block extraction: {e}"})
    env_raw = verify_pin(str(BASE / "pr12_8" / "f1b_envelope.json"),
                         manifest, kills)
    for rel in (str(BASE / "PR12_8_S1_CONTRACT_CANDIDATE.md"),
                str(BASE / "PR12_8_MONITORING_WITHDRAWAL.md"),
                str(BASE / "pr12_8" / "candidacy_scan.json")):
        verify_pin(rel, manifest, kills)
    report["pins"] = {
        "emitter_sha256_matches": emitter_ok,
        "policy_block_attestation": {"sha256": block_sha,
                                     "matches_registered": att_ok,
                                     "by_source": block_shas,
                                     "all_sources_identical": blocks_ok},
    }
    if not (att_ok and emitter_ok and blocks_ok):
        kills.append({"kill": 2, "label": "pin/attestation failure"})
    try:
        cand = json.loads(pin_blob(
            str(BASE / "pr12_8" / "candidacy_scan.json")))
        cited_ok = (cand["verdict"] == CITED_VERDICT
                    and CITED_VERDICT in cand["verdict_text"])
    except Exception as ex:
        cited_ok = False
        kills.append({"kill": 1, "label": f"evidence base unreadable: {ex}"})
    report["pins"]["evidence_base_verdict_intact"] = cited_ok
    if not cited_ok:
        kills.append({"kill": 1, "label": "cited verdict name not intact"})

    if env_raw is None or kills:
        report["verdict"] = "certification-blocked"
        report["verdict_text"] = ("certification-blocked — a §8 kill "
                                  "condition fired before gating; see "
                                  "kill_conditions. No status is "
                                  "conferred; PR-10 merge-abstain remains "
                                  "the only certified reader contract; "
                                  "posture remains deferral.")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_FILE.write_text(json.dumps(report, indent=1, sort_keys=True)
                            + "\n")
        print("VERDICT: certification-blocked (pre-gate kill)")
        return 1
    env = json.loads(env_raw)

    # ---- C-1 + C-2 (with internal double pass for C-6)
    c1a, c2a, first_pass = c1_c2_pass(env, manifest, kills)
    c1b, _c2b, second_pass = c1_c2_pass(env, {}, [])
    internal_same = first_pass == second_pass and c1a == c1b
    report["gates"]["C-1"] = c1a
    report["gates"]["C-2"] = c2a

    # ---- C-3: the §5.1 conformance suite, re-run
    r3 = subprocess.run([sys.executable,
                         "harness/witness_alt_conformance_tests.py"],
                        cwd=REPO, capture_output=True)
    conf = json.loads((OUT_DIR / "conformance_results.json").read_text())
    ambiguities = conf["summary"]["specification_ambiguities_found"]
    c3_ok = (r3.returncode == 0 and conf["result"] == "all-pass"
             and not conf["summary"]["failing_subject_results"]
             and ambiguities == 0)
    report["gates"]["C-3"] = {
        "suite_exit": r3.returncode,
        "result": conf["result"],
        "n_checks": conf["summary"]["n_checks"],
        "failing_subject_results":
            conf["summary"]["failing_subject_results"],
        "specification_ambiguities_found": ambiguities,
        "results_sha256": hashlib.sha256(
            (OUT_DIR / "conformance_results.json").read_bytes())
        .hexdigest(),
        "pass": c3_ok}

    # ---- C-4: the §5.2 withdrawal demo, re-run
    r4 = subprocess.run([sys.executable,
                         "harness/witness_alt_withdrawal_demo.py"],
                        cwd=REPO, capture_output=True)
    demo = json.loads((OUT_DIR / "withdrawal_demo_report.json").read_text())
    c4_ok = (r4.returncode == 0
             and demo["result"] == "all-scenarios-pass"
             and demo["tripwires_demonstrated"]
             == ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
             and demo["event_log"]["hash_chain_verified"]
             and demo["event_log"]["coupling_verified"]
             and all(s["events_well_formed"] for s in demo["scenarios"]))
    report["gates"]["C-4"] = {
        "demo_exit": r4.returncode,
        "result": demo["result"],
        "tripwires_demonstrated": demo["tripwires_demonstrated"],
        "hash_chain_verified": demo["event_log"]["hash_chain_verified"],
        "coupling_verified": demo["event_log"]["coupling_verified"],
        "report_sha256": hashlib.sha256(
            (OUT_DIR / "withdrawal_demo_report.json").read_bytes())
        .hexdigest(),
        "event_log_note": "append-only by design; excluded from cross-run "
                          "byte-identity (§9); no run-varying value from "
                          "it is embedded in this scan",
        "pass": c4_ok}

    # ---- C-6: determinism (internal here; external re-invocation is
    # part of the run procedure and recorded in memo §12.3)
    report["gates"]["C-6"] = {
        "internal_double_pass_identical": internal_same,
        "no_timestamps": True,
        "external_second_invocation": "performed by the run procedure; "
                                      "this file must be byte-identical "
                                      "across invocations",
        "pass": internal_same}
    if not internal_same:
        kills.append({"kill": 7, "label": "internal double pass differs"})

    # ---- C-7: approval separation (structural)
    report["gates"]["C-7"] = {
        "note": "this run changes nothing: the verdict below rests on its "
                "branch; the §4.2 registry sentence and §4.3 posture "
                "change only upon explicit human approval of the merge "
                "(Stage III, separately unauthorized)",
        "registry_sentence_changed": False,
        "posture_changed": False,
        "pass": True}

    # ---- C-5: bound carriage — §3 embedded verbatim from the PINNED memo
    sec3 = extract_sec3(pin_blob(MEMO))
    gates_pass = all(report["gates"][g]["pass"]
                     for g in ("C-1", "C-2", "C-3", "C-4", "C-6", "C-7"))
    if kills:
        verdict = "certification-blocked"
    elif ambiguities > 0:
        verdict = "certification-insufficient"
    elif gates_pass:
        verdict = GO_NAME
    else:
        verdict = "certification-negative"

    if verdict == GO_NAME:
        verdict_text = (
            f"{verdict} — the s1-witness-alt-batch batch packet-reader "
            f"contract for W2:F1b passes all certification gates C-1–C-7 "
            f"at certification pin {CERTIFICATION_PIN}, on the sole "
            f"adjudicated evidence base PR-12.8 Stage E at a0e621d, "
            f"verdict {CITED_VERDICT}, whose registered bounds are "
            "constitutive parts of this verdict and are embedded verbatim "
            "below (PR-12.9 §3). The -seedbounded qualifier is permanent "
            "under this registration. PER GATE C-7 THIS VERDICT HAS NO "
            "EFFECT: it rests on its branch; the contract registry "
            "sentence changes only per §4.2, and the posture only per "
            "§4.3, upon explicit human approval of the merge (Stage III, "
            "separately unauthorized). Until then nothing is served, "
            "deployed, promoted, or ingested; no FAM-core change; no "
            "S2/online claim; PR-10 merge-abstain remains the only "
            "certified reader contract and is never modified by any "
            "outcome here; the operational posture on witness-window "
            "rows remains deferral.\n\n" + sec3)
    else:
        verdict_text = (
            f"{verdict} — see gate table and kill_conditions; no "
            "certification status is conferred; the candidate remains at "
            "rung 3 under its registered bounds, cited by its exact name "
            f"{CITED_VERDICT} and never without them (PR-12.9 §3, "
            "embedded verbatim below). PR-10 merge-abstain remains the "
            "only certified reader contract; posture remains "
            "deferral.\n\n" + sec3)
    required = [CITED_VERDICT, "-seedbounded", "(−4, +23)", "≤13.3%",
                "319 in-scope W2 rows", "73 acts / 70.0 wrong mass",
                "panel-insufficient", "safe-by-silence",
                "s0 (dev), s1, s2", "T1", "T7", "withdrawn-pending-review",
                "Tighten-only, forever"]
    missing = [s for s in required if s not in verdict_text]
    c5_ok = not missing and sec3.startswith("## 3. Evidence base")
    report["gates"]["C-5"] = {
        "sec3_embedded_verbatim_from_pin": True,
        "required_phrases_missing": missing,
        "cited_name_exact": CITED_VERDICT in verdict_text,
        "pass": c5_ok}
    if not c5_ok:
        kills.append({"kill": 1, "label": f"C-5 carriage failure: "
                                          f"{missing}"})
        verdict = "certification-blocked"
        verdict_text = ("certification-blocked — scope-laundering guard: "
                        "bound carriage failed; see gates.C-5. PR-10 "
                        "merge-abstain remains the only certified reader "
                        "contract; posture remains deferral.")

    report["verdict"] = verdict
    report["verdict_text"] = verdict_text
    report["frozen_surfaces_clean_after"] = ct.frozen_clean()
    if not report["frozen_surfaces_clean_after"]:
        kills.append({"kill": 6, "label": "frozen surface dirty after run"})
        report["verdict"] = verdict = "certification-blocked"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")

    print(f"VERDICT: {verdict}")
    for g in ("C-1", "C-2", "C-3", "C-4", "C-5", "C-6", "C-7"):
        print(f"  {g}: {'PASS' if report['gates'][g]['pass'] else 'FAIL'}")
    print("Scope (PR-12.9 §11 / C-7): this run confers nothing — the "
          "verdict rests on its branch pending explicit approval; nothing "
          "is served, deployed, promoted, or ingested; no FAM-core "
          "change; PR-10 merge-abstain remains the only certified reader "
          "contract; the operational posture on witness-window rows "
          "remains deferral.")
    return 1 if kills else 0


if __name__ == "__main__":
    sys.exit(main())
