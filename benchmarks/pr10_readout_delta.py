"""PR-10 step 1 — read-time abstention readout delta reader (analysis-only).

Scores the ``--read-govern merge-abstain`` serving seam against its committed
``--read-govern none`` baseline twins, per cell, on the exactness gates of
PR10_READTIME_ABSTENTION_GATE.md §3, and emits ``readout_delta.json`` with
verdict ``readout-certified`` | ``fail``. There is NO ``needs_review`` tier:
every expected value is pre-registered to exact equality, so nothing here
requires judgment (gate memo §3).

Gates (ALL must hold on EVERY manifest cell):

  G1 — write-stream byte-identity. ``fork_events.csv``, ``per_slot.csv``,
       ``topk.csv(.gz)`` of the governed run byte-identical to the baseline
       (topk compared on decompressed content when either side is gzipped);
       ``summary.json`` identical after removing the governed arm's
       ``read_govern`` block (both-absent, e.g. synthetic runs, is vacuous).
  G2 — answered-stream byte-identity. The governed per-probe CSV, after
       dropping the two appended READ_GOVERN_COLS, byte-identical to the
       baseline CSV on every row (every pre-existing byte of every line).
  G3 — abstention-set exactness. The multiset of abstained probe rows equals
       the frozen scorer's merge-abstain M-led row set, recomputed here from
       the run's own fork_events + topk artifacts; the count equals
       ``abstained_merge`` in the PR-9 envelope when the cell names an
       envelope cell; 0 abstentions on every non-soft cell.
  G4 — trigger purity. ``abstain_reason == merge_suspect_led`` on every
       abstained row and empty on every answered row; no forced / tie /
       other reason value exists in the schema's vocabulary at all.
  G5 — determinism. When the manifest supplies a same-seed ``governed_twin``
       stem, every governed artifact is byte-identical to it; the sha256 of
       every governed artifact is recorded either way (the cross-host
       gentoo/darwin comparison for step 2 is a hash comparison of this
       output).

Hard constraints (the pr8_shadow_audit_cert.py pattern):
  * imports NO torch and no scorer module — it reads committed CSV/JSON/gz
    text only. The scorer constants/rule mirrored below are pinned equal to
    the live ``analyze_fork_governance`` / ``failure_mode_probe`` values by
    tests/test_pr10_read_seam.py, so they cannot silently drift.
  * never rewrites an artifact; every file it touches is read-only.
  * manifest-driven, so it runs unchanged on the hermetic tests' tmp twins
    and on the full step-2 panel.

Manifest (JSON):

    {
      "artifact_root": "<dir>",              # optional; stems resolve under it
      "envelope": "<abstention_envelope.json>",   # optional (panel mode)
      "cells": [
        {"cell": "pairD/soft/s0",
         "soft": true,                       # false => 0 abstentions required
         "baseline": "<stem>",               # committed --read-govern none run
         "governed": "<stem>",               # --read-govern merge-abstain run
         "envelope_cell": "pairD/soft/s0",   # optional: exact count check
         "governed_twin": "<stem>"},         # optional same-seed re-run (G5)
        ...
      ]
    }

A *stem* is an artifact path with no suffix; the reader appends ``.csv``,
``.per_slot.csv``, ``.fork_events.csv``, ``.topk.csv``/``.topk.csv.gz``,
``.summary.json``.

Usage:
  python benchmarks/pr10_readout_delta.py \
      --manifest results/issue_failure_mode_blindness/pr10/panel_manifest.json \
      --out results/issue_failure_mode_blindness/pr10/readout_delta.json
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

# --- constants mirrored from the live modules so this reader imports no torch.
# tests/test_pr10_read_seam.py pins every one equal to the live
# analyze_fork_governance / failure_mode_probe values (the pr8 harness
# pattern), so they cannot silently drift from the frozen scorer.
MERGE_SUSPECT_COS = 0.9                      # analyze_fork_governance
ABSORBED_OUTCOME = "absorbed"                # failure_mode_probe.ABSORBED
READ_GOVERN_ACTION = "merge-abstain"         # the only PR-10 serving action
READ_GOVERN_COLS = ("served_outcome", "abstain_reason")
SERVED_ANSWER = "answer"
SERVED_ABSTAIN = "abstain"
ABSTAIN_REASON_MERGE_LED = "merge_suspect_led"

VERDICT_CERTIFIED = "readout-certified"
VERDICT_FAIL = "fail"

SUF_PER_PROBE = ".csv"
SUF_PER_SLOT = ".per_slot.csv"
SUF_FORK = ".fork_events.csv"
SUF_TOPK = ".topk.csv"
SUF_SUMMARY = ".summary.json"
WRITE_STREAM_SUFFIXES = (SUF_FORK, SUF_PER_SLOT)


# ---------------------------------------------------------------------------
# Artifact reading (read-only; bytes preserved exactly)
# ---------------------------------------------------------------------------
def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _topk_content(stem: Path) -> tuple[bytes, str]:
    """(decompressed content bytes, form) for ``<stem>.topk.csv[.gz]``."""
    plain = Path(f"{stem}{SUF_TOPK}")
    gz = Path(f"{stem}{SUF_TOPK}.gz")
    if plain.exists():
        return plain.read_bytes(), "plain"
    if gz.exists():
        return gzip.decompress(gz.read_bytes()), "gz"
    raise FileNotFoundError(f"{plain} (or .gz)")


def _csv_rows(data: bytes) -> list[dict]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")))


def _finite(text: str) -> bool:
    try:
        return math.isfinite(float(text))
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# The frozen scorer's merge-abstain rule, recomputed from the artifacts.
# Mirrors analyze_fork_governance byte-for-byte on the SAME csv text the
# scorer parses: merges = absorbed events with finite payload_cos_incumbent
# < MERGE_SUSPECT_COS (build_writetime_router); M at probe epoch E = merges
# with epoch <= E (router_state); abstain iff the surviving top-1 candidate
# (max float(weight), first-rank tie-break) is in M (apply_policy).
# ---------------------------------------------------------------------------
def merge_led_rows(stem: Path) -> set[tuple[int, int]]:
    """(epoch, probe_index) rows the frozen scorer's merge-abstain abstains
    on, recomputed from ``<stem>.fork_events.csv`` + ``<stem>.topk.csv``."""
    events = _csv_rows(_read_bytes(Path(f"{stem}{SUF_FORK}")))
    merges = [(int(float(e["epoch"])), int(float(e["owner_slot"])))
              for e in events
              if e["outcome"] == ABSORBED_OUTCOME
              and _finite(e["payload_cos_incumbent"])
              and float(e["payload_cos_incumbent"]) < MERGE_SUSPECT_COS]
    topk: dict[tuple[int, int], list[dict]] = {}
    for r in _csv_rows(_topk_content(stem)[0]):
        topk.setdefault((int(r["epoch"]), int(r["probe_index"])),
                        []).append(r)
    led = set()
    for (epoch, probe_index), cands in topk.items():
        m = {s for (e, s) in merges if e <= epoch}
        if not m:
            continue
        cands.sort(key=lambda c: int(c["rank"]))
        surv = [c for c in cands if c["surviving"] == "1"]
        top1 = max(surv, key=lambda c: float(c["weight"]))
        if int(top1["slot"]) in m:
            led.add((epoch, probe_index))
    return led


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def _gate_g1(base: Path, gov: Path) -> dict:
    """Write-stream byte-identity + summary identity modulo read_govern."""
    detail: dict = {}
    ok = True
    for suf in WRITE_STREAM_SUFFIXES:
        same = _read_bytes(Path(f"{base}{suf}")) == \
            _read_bytes(Path(f"{gov}{suf}"))
        detail[suf] = "byte-identical" if same else "DIFFERS"
        ok &= same
    b_topk, b_form = _topk_content(base)
    g_topk, g_form = _topk_content(gov)
    same = b_topk == g_topk
    detail[SUF_TOPK] = {"content_identical": same,
                        "baseline_form": b_form, "governed_form": g_form}
    ok &= same
    b_sum = Path(f"{base}{SUF_SUMMARY}")
    g_sum = Path(f"{gov}{SUF_SUMMARY}")
    if not b_sum.exists() and not g_sum.exists():
        detail[SUF_SUMMARY] = "absent-on-both (synthetic run; vacuous)"
    elif b_sum.exists() != g_sum.exists():
        detail[SUF_SUMMARY] = "PRESENT ON ONE SIDE ONLY"
        ok = False
    else:
        b = json.loads(b_sum.read_text())
        g = json.loads(g_sum.read_text())
        rg = g.pop("read_govern", None)
        block_ok = (isinstance(rg, dict)
                    and rg.get("action") == READ_GOVERN_ACTION)
        rest_ok = (g == b) and ("read_govern" not in b)
        detail[SUF_SUMMARY] = {
            "governed_has_read_govern_block": block_ok,
            "identical_after_removing_read_govern": rest_ok,
        }
        ok &= block_ok and rest_ok
    return {"pass": ok, "detail": detail}


def _gate_g2_g4(base: Path, gov: Path) -> tuple[dict, dict, list]:
    """G2 (answered-stream byte-identity after dropping the two appended
    columns) and G4 (trigger purity), one pass over the raw CSV bytes.

    Returns (g2, g4, served) where ``served`` is the emission-ordered list of
    (epoch, probe_index, served_outcome) rows parsed from the governed CSV.
    """
    base_lines = _read_bytes(Path(f"{base}{SUF_PER_PROBE}")).split(b"\r\n")
    gov_lines = _read_bytes(Path(f"{gov}{SUF_PER_PROBE}")).split(b"\r\n")
    g2_ok = len(base_lines) == len(gov_lines)
    g2_bad: list[int] = []
    g4_ok = True
    g4_bad: list[int] = []
    served: list[tuple[int, int, str]] = []
    header_ok = False
    if g2_ok:
        for i, (gl, bl) in enumerate(zip(gov_lines, base_lines)):
            if gl == b"" and bl == b"":  # trailing terminator
                continue
            parts = gl.rsplit(b",", 2)
            if len(parts) != 3 or parts[0] != bl:
                g2_ok = False
                g2_bad.append(i)
                continue
            core, outcome, reason = parts
            if i == 0:
                header_ok = (outcome, reason) == tuple(
                    c.encode() for c in READ_GOVERN_COLS)
                g4_ok &= header_ok
                continue
            if outcome == SERVED_ABSTAIN.encode():
                pure = reason == ABSTAIN_REASON_MERGE_LED.encode()
            elif outcome == SERVED_ANSWER.encode():
                pure = reason == b""
            else:
                pure = False
            if not pure:
                g4_ok = False
                g4_bad.append(i)
    # (epoch, probe_index, served_outcome) per data row, parsed with the csv
    # module so field positions never drift from the header.
    for r in _csv_rows(_read_bytes(Path(f"{gov}{SUF_PER_PROBE}"))):
        served.append((int(float(r["epoch"])), int(r["probe_index"]),
                       r[READ_GOVERN_COLS[0]]))
    g2 = {"pass": g2_ok,
          "detail": {"lines": len(gov_lines),
                     "appended_cols": list(READ_GOVERN_COLS),
                     "first_mismatched_lines": g2_bad[:5]}}
    g4 = {"pass": g4_ok,
          "detail": {"header_cols_ok": header_ok,
                     "reason_vocabulary": ["", ABSTAIN_REASON_MERGE_LED],
                     "first_impure_lines": g4_bad[:5]}}
    return g2, g4, served


def _gate_g3(gov: Path, served: list, soft: bool,
             envelope_cells: dict | None, envelope_cell: str | None) -> dict:
    """Abstention-set exactness vs the recomputed frozen-scorer M-led rows,
    the non-soft zero rule, and the envelope count (when named)."""
    actual = Counter((e, p) for (e, p, out) in served
                     if out == SERVED_ABSTAIN)
    expected = Counter(merge_led_rows(gov))
    set_ok = actual == expected
    n_abstained = sum(actual.values())
    detail: dict = {"abstained": n_abstained,
                    "merge_led_rows_recomputed": sum(expected.values()),
                    "set_equal": set_ok}
    ok = set_ok
    if not soft:
        zero_ok = n_abstained == 0
        detail["non_soft_zero"] = zero_ok
        ok &= zero_ok
    if envelope_cell is not None:
        if envelope_cells is None:
            detail["envelope_count"] = "ENVELOPE CELL NAMED BUT NO ENVELOPE"
            ok = False
        elif envelope_cell not in envelope_cells:
            detail["envelope_count"] = f"CELL {envelope_cell!r} NOT IN ENVELOPE"
            ok = False
        else:
            want = envelope_cells[envelope_cell]["abstained_merge"]
            detail["envelope_count"] = {"expected_abstained_merge": want,
                                        "equal": n_abstained == want}
            ok &= n_abstained == want
    return {"pass": ok, "detail": detail}


def _governed_artifacts(stem: Path) -> dict[str, bytes]:
    """Every governed artifact's content bytes (topk decompressed), keyed by
    suffix, for hashing / twin comparison."""
    out = {SUF_PER_PROBE: _read_bytes(Path(f"{stem}{SUF_PER_PROBE}")),
           SUF_PER_SLOT: _read_bytes(Path(f"{stem}{SUF_PER_SLOT}")),
           SUF_FORK: _read_bytes(Path(f"{stem}{SUF_FORK}")),
           SUF_TOPK: _topk_content(stem)[0]}
    summary = Path(f"{stem}{SUF_SUMMARY}")
    if summary.exists():
        out[SUF_SUMMARY] = summary.read_bytes()
    return out


def _gate_g5(gov: Path, twin: Path | None) -> dict:
    arts = _governed_artifacts(gov)
    detail: dict = {"governed_sha256": {s: _sha256(b)
                                        for s, b in sorted(arts.items())}}
    ok = True
    if twin is None:
        detail["twin"] = ("no same-seed twin in manifest; sha256 recorded "
                          "for the cross-host comparison")
    else:
        twin_arts = _governed_artifacts(twin)
        same = (sorted(arts) == sorted(twin_arts)
                and all(arts[s] == twin_arts[s] for s in arts))
        detail["twin"] = {"stem": str(twin), "byte_identical": same}
        ok = same
    return {"pass": ok, "detail": detail}


# ---------------------------------------------------------------------------
# Cell + manifest scoring
# ---------------------------------------------------------------------------
def score_cell(spec: dict, root: Path,
               envelope_cells: dict | None) -> dict:
    def _stem(key):
        if key not in spec or spec[key] is None:
            return None
        p = Path(spec[key])
        return p if p.is_absolute() else root / p

    base, gov, twin = _stem("baseline"), _stem("governed"), _stem("governed_twin")
    soft = bool(spec["soft"])
    gates: dict = {}
    try:
        gates["G1_write_stream_byte_identity"] = _gate_g1(base, gov)
        g2, g4, served = _gate_g2_g4(base, gov)
        gates["G2_answered_stream_byte_identity"] = g2
        gates["G3_abstention_set_exactness"] = _gate_g3(
            gov, served, soft, envelope_cells, spec.get("envelope_cell"))
        gates["G4_trigger_purity"] = g4
        gates["G5_determinism"] = _gate_g5(gov, twin)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        gates["artifact_error"] = {"pass": False, "detail": repr(exc)}
    ok = all(g.get("pass") for g in gates.values())
    return {"soft": soft, "baseline": str(base), "governed": str(gov),
            "gates": gates, "pass": ok}


def build_readout_delta(manifest: dict, manifest_dir: Path) -> dict:
    root = Path(manifest.get("artifact_root", "."))
    if not root.is_absolute():
        root = manifest_dir / root
    envelope_cells = None
    envelope_meta = None
    env = manifest.get("envelope")
    if env is not None:
        env_path = Path(env)
        if not env_path.is_absolute():
            env_path = manifest_dir / env_path
        data = env_path.read_bytes()
        envelope_cells = json.loads(data)["cells_fresh"]
        envelope_meta = {"path": str(env_path), "sha256": _sha256(data)}

    cells = {spec["cell"]: score_cell(spec, root, envelope_cells)
             for spec in manifest.get("cells", [])}
    all_pass = bool(cells) and all(c["pass"] for c in cells.values())
    return {
        "design": ("PR10_READTIME_ABSTENTION_GATE.md §3 — exactness gates "
                   "G1–G5; verdict readout-certified iff ALL hold on EVERY "
                   "cell, else fail; no needs_review tier"),
        "read_govern_action": READ_GOVERN_ACTION,
        "merge_suspect_cos": MERGE_SUSPECT_COS,
        "envelope": envelope_meta,
        "n_cells": len(cells),
        "cells": cells,
        "verdict": VERDICT_CERTIFIED if all_pass else VERDICT_FAIL,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True,
                    help="JSON manifest of (baseline, governed) twin cells")
    ap.add_argument("--out", required=True,
                    help="where to write readout_delta.json")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    delta = build_readout_delta(json.loads(manifest_path.read_text()),
                                manifest_path.parent)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(delta, indent=1, sort_keys=True) + "\n")

    for cell in sorted(delta["cells"]):
        c = delta["cells"][cell]
        gates = " ".join(f"{k.split('_')[0]}={'ok' if v.get('pass') else 'FAIL'}"
                         for k, v in c["gates"].items())
        print(f"  {cell:24} {'pass' if c['pass'] else 'FAIL'}  {gates}")
    print(f"cells={delta['n_cells']} verdict={delta['verdict']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
