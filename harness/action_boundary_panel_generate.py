#!/usr/bin/env python3
"""PR-12.8 Stage A — standalone traffic-axis panel generator (memo §7).

Registered in PR12_8_READER_CONTRACT_CANDIDACY.md §7 Stage A and §11: the
PR-12.7 erratum-E1 generation pattern reused byte-for-byte as a new
driver file. It IMPORTS ``run_cell`` from the byte-frozen emitter
(sha256 pinned, checked before import and after every run; the emitter
is never edited) and calls it with the identical keyword signature the
registered PR-12.4 W2 runner uses. Before any panel packet may be
emitted, the same §23.4-mandated self-check must regenerate the
committed PR-12.4 W2 parity anchor
``results/issue_failure_mode_blindness/pr12_4/W2/pairD_oneshot_s1/``
byte-identically (full byte compare + sha256 + schema + ordering).

Panel scope (pre-flight determinations of record): mixed and
plain-stale, pairs B-E, seeds s1/s2 only (s0 is dev exposure). The
registered jitter (pairA) and g5twin (pairD) report-only anchors are
PANEL-INSUFFICIENT: their committed stems exist only at s0 and the pr4
geometry table carries no governance key for either arm, so they cannot
be generated under the registered provenance mechanism; they are
recorded as named scope bounds, never emitted.

Stage separation: emitting the panel cache authorizes no scoring; the
panel scorer is a separate program under its own approval. This driver
never reads truth labels or scoring outputs and scans its own output
for scoring signals. No FAM-core file is imported. PR-10 merge-abstain
remains the only certified reader contract.

stdlib + subprocess-git only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent

# §23.3 pin: the frozen emitter may not change across (or because of) any
# run of this driver.
EMITTER = HARNESS_DIR / "harness_boundary_sim.py"
EMITTER_SHA256_PIN = \
    "2539686a205fa03ba88fb4e222043720dc6acf97460f896a31bd58d1a11d32e5"
POLICY_PATH = HARNESS_DIR / "harness_policy.json"

# §23.4 parity anchor: committed PR-12.4 W2 emitter-output cell.
ANCHOR_SCAN_KEY = "scan12_4"
ANCHOR_CELL = "pairD_oneshot_s1"
ANCHOR_SHAPE = "W2"
ANCHOR_COMMITTED = ("results/issue_failure_mode_blindness/pr12_4/W2/"
                    "pairD_oneshot_s1")

PACKET_FILES = ("memory_packet.jsonl", "audit_packet.jsonl",
                "decision_table.csv")
# load_cell() input surface (harness_boundary_sim.py:56-89), hashed for
# provenance.
RUN_STEM_SUFFIXES = (".csv", ".topk.csv.gz", ".per_slot.csv",
                     ".fork_events.csv", ".summary.json")

# ``allow_stale`` is the emitter CLI's ``store_true`` default (False); no
# registered PR-12.x run passed --allow-stale. Pinned here; the anchor
# byte-equivalence check is what certifies the pin.
ALLOW_STALE = False

# ---- Stage A emission registration (§4 table via §23.3) -------------------
HOLDOUT_SCAN_KEY = "scan12_8_panel"
HOLDOUT_CACHE = "results/issue_failure_mode_blindness/pr12_8_panel_cache"
GOVERNED = "results/issue_failure_mode_blindness/pr10/governed"
GEOMETRY = ("results/issue_failure_mode_blindness/pr4/"
            "pr4_geometry_table.json#governance#")
# Exact §4 registered holdout cells. Any scan12_7_holdout cell deviating
# from this table in name, run-stem, or hazard-governance is kill §23.5.3.
REGISTERED_HOLDOUT_CELLS = {
    "pairB_mixed_s1": (f"{GOVERNED}/per_probe_mixed_pairB_s1",
                          f"{GEOMETRY}pairB/mixed/s1"),
    "pairB_mixed_s2": (f"{GOVERNED}/per_probe_mixed_pairB_s2",
                          f"{GEOMETRY}pairB/mixed/s2"),
    "pairC_mixed_s1": (f"{GOVERNED}/per_probe_mixed_pairC_s1",
                          f"{GEOMETRY}pairC/mixed/s1"),
    "pairC_mixed_s2": (f"{GOVERNED}/per_probe_mixed_pairC_s2",
                          f"{GEOMETRY}pairC/mixed/s2"),
    "pairD_mixed_s1": (f"{GOVERNED}/per_probe_mixed_pairD_s1",
                          f"{GEOMETRY}pairD/mixed/s1"),
    "pairD_mixed_s2": (f"{GOVERNED}/per_probe_mixed_pairD_s2",
                          f"{GEOMETRY}pairD/mixed/s2"),
    "pairE_mixed_s1": (f"{GOVERNED}/per_probe_mixed_pairE_s1",
                          f"{GEOMETRY}pairE/mixed/s1"),
    "pairE_mixed_s2": (f"{GOVERNED}/per_probe_mixed_pairE_s2",
                          f"{GEOMETRY}pairE/mixed/s2"),
    "pairB_stale_s1": (f"{GOVERNED}/per_probe_stale_pairB_s1",
                          f"{GEOMETRY}pairB/stale/s1"),
    "pairB_stale_s2": (f"{GOVERNED}/per_probe_stale_pairB_s2",
                          f"{GEOMETRY}pairB/stale/s2"),
    "pairC_stale_s1": (f"{GOVERNED}/per_probe_stale_pairC_s1",
                          f"{GEOMETRY}pairC/stale/s1"),
    "pairC_stale_s2": (f"{GOVERNED}/per_probe_stale_pairC_s2",
                          f"{GEOMETRY}pairC/stale/s2"),
    "pairD_stale_s1": (f"{GOVERNED}/per_probe_stale_pairD_s1",
                          f"{GEOMETRY}pairD/stale/s1"),
    "pairD_stale_s2": (f"{GOVERNED}/per_probe_stale_pairD_s2",
                          f"{GEOMETRY}pairD/stale/s2"),
    "pairE_stale_s1": (f"{GOVERNED}/per_probe_stale_pairE_s1",
                          f"{GEOMETRY}pairE/stale/s1"),
    "pairE_stale_s2": (f"{GOVERNED}/per_probe_stale_pairE_s2",
                          f"{GEOMETRY}pairE/stale/s2"),
}
# §23.3 parity: same shape set the registered W2 runner emits.
REGISTERED_SHAPES = ("prototype", "W1", "W2")

# §23.6 cache purity: none of these may appear anywhere in generated
# output (verified byte-absent from the committed anchor in advance).
FORBIDDEN_CACHE_TOKENS = (
    "true_label", "vote_pred", "registry",
    "action-boundary-", "holdout-validity-", "holdout-insufficient",
    "F1a", "F1b", "F1c", "f1a", "f1b", "f1c",
    "coverage", "precision", "wrong_mass", "wrong-action",
    "pr12_5/", "pr12_6/",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                          check=True, capture_output=True,
                          text=True).stdout.strip()


def check_emitter_pin(stage: str, provenance: dict) -> None:
    actual = sha256_file(EMITTER)
    provenance["emitter_sha256"][stage] = actual
    if actual != EMITTER_SHA256_PIN:
        die(provenance, f"kill §23.5.2: emitter sha256 {actual} != pin "
                        f"{EMITTER_SHA256_PIN} ({stage})")


def die(provenance: dict, reason: str) -> None:
    provenance["verdict"] = "stageA-blocked"
    provenance["kill"] = reason
    print(f"KILL  {reason}", file=sys.stderr)
    _flush_provenance(provenance)
    sys.exit(1)


_PROV_OUT: Path | None = None


def _flush_provenance(provenance: dict) -> None:
    provenance["recorded_utc"] = datetime.now(timezone.utc).isoformat()
    text = json.dumps(provenance, indent=2, sort_keys=True)
    if _PROV_OUT is not None:
        _PROV_OUT.parent.mkdir(parents=True, exist_ok=True)
        _PROV_OUT.write_text(text + "\n")
    print(text)


def jsonl_structure(path: Path) -> tuple[list[tuple[str, ...]], list[str]]:
    """(per-record sorted field tuples, per-record identity) for the
    explicit schema/ordering assertions of §23.4."""
    schemas, order = [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            schemas.append(tuple(sorted(rec.keys())))
            order.append(str(rec.get("query_id",
                                      rec.get("record_type", ""))))
    return schemas, order


def csv_structure(path: Path) -> tuple[list[str], list[str]]:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    return rows[0], [r[0] for r in rows[1:]]


def purity_scan(out_dir: Path) -> list[dict]:
    hits = []
    for fn in PACKET_FILES:
        data = (out_dir / fn).read_text()
        for tok in FORBIDDEN_CACHE_TOKENS:
            n = data.count(tok)
            if n:
                hits.append({"file": fn, "token": tok, "count": n})
    return hits


def input_hashes(cfg: dict) -> dict:
    """sha256 every committed input load_cell()/load_hazard() will read."""
    out = {}
    stem = REPO_ROOT / cfg["run_stem"]
    for suffix in RUN_STEM_SUFFIXES:
        p = stem.with_suffix(suffix)
        out[str(p.relative_to(REPO_ROOT))] = sha256_file(p)
    hazard_file = REPO_ROOT / cfg["hazard_governance"].split("#", 1)[0]
    out[str(hazard_file.relative_to(REPO_ROOT))] = sha256_file(hazard_file)
    out[str(POLICY_PATH.relative_to(REPO_ROOT))] = sha256_file(POLICY_PATH)
    return out


def frozen_surface_clean(provenance: dict) -> None:
    """§23.5.5: git dirtiness on frozen surfaces is a kill."""
    frozen = ["harness/harness_boundary_sim.py",
              "harness/harness_policy.json",
              "harness/action_boundary_score.py",
              "harness/reader_utility_score.py",
              "results/issue_failure_mode_blindness/pr10",
              "results/issue_failure_mode_blindness/pr12",
              "results/issue_failure_mode_blindness/pr12_1",
              "results/issue_failure_mode_blindness/pr12_2",
              "results/issue_failure_mode_blindness/pr12_3",
              "results/issue_failure_mode_blindness/pr12_4",
              "results/issue_failure_mode_blindness/pr12_5",
              "results/issue_failure_mode_blindness/pr12_6",
              "results/issue_failure_mode_blindness/pr12_7",
              "results/issue_failure_mode_blindness/pr12_7_holdout_cache",
              "harness/action_boundary_holdout_score.py",
              "harness/action_boundary_holdout_generate.py",
              "harness/witness_alt_reference_reader.py"]
    dirty = git("status", "--porcelain", "--", *frozen)
    dirty = "\n".join(line for line in dirty.splitlines()
                      if not line.endswith(".DS_Store"))
    provenance["frozen_surface_status"] = dirty
    if dirty:
        die(provenance, f"kill §23.5.5: frozen surface dirty:\n{dirty}")


def load_run_cell():
    """Import the frozen primitive AFTER the sha256 pin check."""
    sys.path.insert(0, str(HARNESS_DIR))
    from harness_boundary_sim import run_cell  # noqa: E402  (frozen import)
    return run_cell


def regenerate_cell(run_cell, name: str, cfg: dict, policy: dict,
                    out_root: Path, shape: str, policy_version: str):
    """§23.3 faithful-invocation contract: keyword-identical to the
    registered W2 runner (harness_boundary_sim.py run_scan12_4:1897)."""
    return run_cell(REPO_ROOT, name, cfg, policy, ALLOW_STALE,
                    out_root=out_root, shape=shape,
                    policy_version=policy_version,
                    emit_review_queue=True, emit_ambiguous_queue=True)


def compare_cell(fresh_dir: Path, committed_dir: Path) -> dict:
    """§23.4 proof: byte identity + sha256 + explicit schema/ordering."""
    proof, ok = {}, True
    for fn in PACKET_FILES:
        fresh, committed = fresh_dir / fn, committed_dir / fn
        fb, cb = fresh.read_bytes(), committed.read_bytes()
        if fn.endswith(".jsonl"):
            fs, fo = jsonl_structure(fresh)
            cs, co = jsonl_structure(committed)
        else:
            fs, fo = csv_structure(fresh)
            cs, co = csv_structure(committed)
        entry = {
            "committed_sha256": hashlib.sha256(cb).hexdigest(),
            "regenerated_sha256": hashlib.sha256(fb).hexdigest(),
            "byte_identical": fb == cb,
            "sha256_identical":
                hashlib.sha256(fb).digest() == hashlib.sha256(cb).digest(),
            "schema_identical": fs == cs,
            "record_order_identical": fo == co,
            "n_records": len(fo),
        }
        ok &= all(entry[k] for k in ("byte_identical", "sha256_identical",
                                     "schema_identical",
                                     "record_order_identical"))
        proof[fn] = entry
    proof["all_identical"] = ok
    return proof


def self_check(provenance: dict) -> None:
    """§23.4 mandatory pre-emission anchor byte-equivalence self-check."""
    policy = json.loads(POLICY_PATH.read_text())
    scan = policy[ANCHOR_SCAN_KEY]
    cfg = scan["cells"][ANCHOR_CELL]
    provenance["anchor"] = {
        "scan_key": ANCHOR_SCAN_KEY, "cell": ANCHOR_CELL,
        "shape": ANCHOR_SHAPE, "committed_dir": ANCHOR_COMMITTED,
        "policy_version": scan["policy_version"],
        "cell_config": cfg, "allow_stale": ALLOW_STALE,
    }
    provenance["anchor_input_sha256"] = input_hashes(cfg)

    run_cell = load_run_cell()
    tmp = Path(tempfile.mkdtemp(prefix="pr12_8_anchor_parity_"))
    try:
        regenerate_cell(run_cell, ANCHOR_CELL, cfg, policy,
                        out_root=tmp / ANCHOR_SHAPE, shape=ANCHOR_SHAPE,
                        policy_version=scan["policy_version"])
        fresh_dir = tmp / ANCHOR_SHAPE / ANCHOR_CELL
        proof = compare_cell(fresh_dir, REPO_ROOT / ANCHOR_COMMITTED)
        provenance["anchor_byte_equivalence"] = proof
        hits = purity_scan(fresh_dir)
        provenance["anchor_purity_scan"] = {
            "forbidden_tokens": FORBIDDEN_CACHE_TOKENS, "hits": hits}
        if hits:
            die(provenance, f"kill §23.6: scoring signal in output: {hits}")
        if not proof["all_identical"]:
            die(provenance, "kill §23.5.1: anchor regeneration is not "
                            "byte-identical to the committed cell")
    finally:
        shutil.rmtree(tmp)
    check_emitter_pin("after_self_check", provenance)
    print("PASS  §23.4 anchor byte-equivalence self-check "
          f"[{ANCHOR_SHAPE}/{ANCHOR_CELL}]", file=sys.stderr)


def emit_holdout(provenance: dict) -> None:
    """Stage A C/E emission — refuses unless separately configured (§23.7).

    Gated, in order, on: the anchor self-check having just passed in this
    same process; a scan12_7_holdout policy block whose cells byte-match
    the §4 registered table; double-emit determinism; cache purity."""
    policy = json.loads(POLICY_PATH.read_text())
    block = policy.get(HOLDOUT_SCAN_KEY)
    if block is None:
        die(provenance,
            f"refused: no '{HOLDOUT_SCAN_KEY}' block in harness_policy.json "
            "— Stage A panel emission is separately authorized and its "
            "additive config manifest does not exist")
    cells = block["cells"]
    registered = {n: {"run_stem": rs, "hazard_governance": hg}
                  for n, (rs, hg) in REGISTERED_HOLDOUT_CELLS.items()}
    got = {n: {"run_stem": c["run_stem"],
               "hazard_governance": c["hazard_governance"]}
           for n, c in cells.items()}
    if got != registered:
        die(provenance, "kill (scope drift): scan12_8_panel cells deviate from "
                        "the registered Stage A table")
    if tuple(block["shapes"]) != REGISTERED_SHAPES:
        die(provenance, f"kill §23.5.3: shapes {block['shapes']} != "
                        f"registered parity set {list(REGISTERED_SHAPES)}")
    pv = block["policy_version"]
    cache = REPO_ROOT / HOLDOUT_CACHE
    if cache.exists() and any(cache.iterdir()):
        die(provenance, "kill §23.5.5: pre-existing non-empty holdout cache")

    provenance["holdout_input_sha256"] = {}
    for name, cfg in cells.items():
        provenance["holdout_input_sha256"][name] = input_hashes(cfg)

    run_cell = load_run_cell()
    twin = Path(tempfile.mkdtemp(prefix="pr12_8_panel_twin_"))
    try:
        for shape in REGISTERED_SHAPES:
            for name, cfg in cells.items():
                regenerate_cell(run_cell, name, cfg, policy,
                                out_root=cache / shape, shape=shape,
                                policy_version=pv)
                regenerate_cell(run_cell, name, cfg, policy,
                                out_root=twin / shape, shape=shape,
                                policy_version=pv)
                proof = compare_cell(twin / shape / name,
                                     cache / shape / name)
                provenance.setdefault("holdout_double_emit", {})[
                    f"{shape}/{name}"] = proof
                if not proof["all_identical"]:
                    shutil.rmtree(cache)
                    die(provenance, f"kill §23.5.6: nondeterministic "
                                    f"emission [{shape}/{name}]")
                hits = purity_scan(cache / shape / name)
                if hits:
                    shutil.rmtree(cache)
                    die(provenance, f"kill §23.6: scoring signal in cache "
                                    f"[{shape}/{name}]: {hits}")
        provenance["holdout_output_sha256"] = {
            f"{shape}/{name}/{fn}":
                sha256_file(cache / shape / name / fn)
            for shape in REGISTERED_SHAPES for name in cells
            for fn in PACKET_FILES}
    finally:
        shutil.rmtree(twin)
    check_emitter_pin("after_emission", provenance)


def main() -> None:
    global _PROV_OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check", action="store_true",
                    help="run ONLY the §23.4 anchor byte-equivalence "
                         "self-check (no holdout emission)")
    ap.add_argument("--emit-holdout", action="store_true",
                    help="Stage A C/E emission — requires separate "
                         "authorization AND a scan12_7_holdout policy "
                         "block; always runs the self-check first")
    ap.add_argument("--provenance-out", type=Path, default=None,
                    help="where to write the provenance JSON (self-check "
                         "mode; must be OUTSIDE the repo). Emission mode "
                         "also writes into the holdout cache root.")
    args = ap.parse_args()
    if args.self_check == args.emit_holdout:
        ap.error("exactly one of --self-check / --emit-holdout")

    provenance = {
        "driver": "harness/action_boundary_panel_generate.py",
        "registration": "PR12_8_READER_CONTRACT_CANDIDACY.md §7 Stage A",
        "command": " ".join(sys.argv),
        "mode": "self-check" if args.self_check else "emit-holdout",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "git_head": git("rev-parse", "HEAD"),
        },
        "emitter_sha256": {"pin": EMITTER_SHA256_PIN},
        "stage_b_scoring_ran": False,
        "fam_core_imported": False,
        "reader_contract": "PR-10 merge-abstain (unchanged; the only "
                           "certified reader contract)",
    }
    if args.provenance_out is not None:
        _PROV_OUT = args.provenance_out.resolve()
        if (_PROV_OUT.is_relative_to(REPO_ROOT)
                and not _PROV_OUT.is_relative_to(REPO_ROOT / HOLDOUT_CACHE)):
            ap.error("--provenance-out inside the repo must be under the "
                     "declared holdout cache (§23.5.5 write boundary)")

    check_emitter_pin("before_run", provenance)
    frozen_surface_clean(provenance)
    self_check(provenance)
    if args.emit_holdout:
        emit_holdout(provenance)
        provenance["verdict"] = "stageA-emitted"
        if _PROV_OUT is None:   # §23.4: proof lives in the cache it gates
            _PROV_OUT = (REPO_ROOT / HOLDOUT_CACHE
                         / "stage_a_provenance.json")
    else:
        provenance["holdout_packets_emitted"] = False
        provenance["verdict"] = "anchor-parity-pass"
    _flush_provenance(provenance)


if __name__ == "__main__":
    main()
