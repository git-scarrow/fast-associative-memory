#!/usr/bin/env python3
"""PR-8 §9A shadow-quarantine certification harness — analysis-only.

This harness certifies the ``--govern shadow`` arm (PR-8 §9A) as *audit-only*
instrumentation: it flags the same supersession writes §8 quarantine would have
diverted, emits the quarantine-ledger shape with
``disposition: flagged_not_diverted``, and changes **no** retrieval behaviour.
It does **not** promote read-time quarantine, claim recovery, or move any frozen
margin (PR8_QUARANTINE_REPLACEMENT_GATE.md §4.9).

Hard constraints (enforced by construction here, pinned by tests):
  * imports no torch and trains nothing — it reads committed CSV/JSON/gz only;
  * depends on no scorer state (no ``feat/pr7-scorer-perseed-guard`` field);
  * never rewrites an artifact; every file it touches is read-only.

It consumes a *manifest* (JSON) describing matched runs:

    {
      "panel": "pr8-9a", "readout": "frozen-87", "mode": "panel"|"smoke",
      "artifact_root": "<dir>",            # stems resolve under here
      "seeds": [0, 1, 2],
      "runs": [
        {"cell": "merge_path_stale", "pair": "pairD", "seed": 0,
         "expected_flagged": 192,          # optional; else directional check
         "stems": {"none": "<stem>", "shadow": "<stem>",
                   "quarantine": "<stem>"}},
        ...
      ],
      "gentoo_root": "<dir>"               # optional second host tree
    }

A *stem* is an artifact path with no suffix; the harness appends ``.csv``,
``.per_slot.csv``, ``.fork_events.csv``, ``.topk.csv`` / ``.topk.csv.gz``,
``.summary.json``, ``.governance.json``.

Verdict semantics:
  * ``fail``       — a HARD invariant broke (retrieval not inert, ledger
                     malformed, clean-control changed, shadow self-inconsistent,
                     or a *provable* per-event identity contradiction);
  * ``pass``       — every dimension certified, including event-level flag-set
                     identity with an empty symmetric difference, a complete
                     frozen panel, cross-host sha256 determinism, and byte-equal
                     governance readouts;
  * ``incomplete`` — every check that *ran* passed, but a required dimension
                     could not be completed from the available artifacts
                     (identity unprovable, panel subset, single host, governance
                     readout absent). ``incomplete`` is NOT a certification.

``promoted-audit`` is asserted ONLY on ``pass``. Anything else certifies nothing.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

# --- engine constants mirrored here so the harness imports no torch ----------
# tests/test_pr8_shadow_audit_cert.py pins these equal to the live
# failure_mode_probe values, so they cannot silently drift.
SUPERSESSION_CLASS = "supersession"          # EVENT_SUPERSESSION
SHADOW_ACTION = "shadow"
SHADOW_STEP = "pr8-9a-shadow-quarantine-audit"
QUARANTINE_ACTION = "quarantine"
FLAGGED_DISPOSITION = "flagged_not_diverted"

# The per-event key tuple the LEGACY §9A flag-set-identity join used, present
# on BOTH the committed shadow fork_events.csv (as columns) and the quarantine
# ledger's diverted_events list. Mirrors
# failure_mode_probe.QUARANTINE_DIVERTED_JOIN_KEY (pinned equal by tests). A
# diverted write has no record_seq/owner_slot of its own, so events were
# bridged by the incumbent identity they supersede. STATE-CONTAMINATED: the
# incumbent_last_write_seq component diverges after the first diversion (the
# recorded §8 finding — identity smoke 8/24), so this key is retained as a
# reported DIAGNOSTIC, no longer the gating join.
JOIN_KEY = ("epoch", "event_class", "incumbent_slot", "incumbent_last_write_seq")

# PR-9.2: the pre-registered write-event-INTRINSIC identity key (gate memo §8's
# required registration). Mirrors failure_mode_probe.PR92_INTRINSIC_JOIN_KEY /
# PR92_INTRINSIC_CHECK_FIELDS (pinned equal by tests). Every component is
# protocol-determined for a given (seed, arm) — no memory state, no RNG, no
# dependence on whether earlier eligible writes committed or diverted — so it
# is identical across none/shadow/quarantine by construction. It is claimed
# UNIQUE per run: a duplicate key on either side is a STOP condition
# (identity-collision), never a joinable multiset.
INTRINSIC_JOIN_KEY = ("epoch", "event_class", "batch_index")
INTRINSIC_CHECK_FIELDS = ("payload_label",)

# --- artifact suffixes -------------------------------------------------------
SUF_PER_PROBE = ".csv"
SUF_PER_SLOT = ".per_slot.csv"
SUF_FORK = ".fork_events.csv"
SUF_TOPK = ".topk.csv"
SUF_TOPK_GZ = ".topk.csv.gz"
SUF_SUMMARY = ".summary.json"
SUF_GOVERNANCE = ".governance.json"

# Retrieval/readout artifacts that MUST be byte-inert between none and shadow.
# topk is handled specially (it may be gzipped); governance is compared if
# present on both sides; summary is compared modulo the additive govern block.
INERT_RAW_SUFFIXES = (SUF_PER_PROBE, SUF_PER_SLOT, SUF_FORK)

# --- frozen §9A panel matrix (PR8 §4.5) --------------------------------------
FROZEN_PANEL = {
    "direct_harm": ["pairD"],
    "collateral_harm": ["pairB", "pairE"],
    "clean_control": ["pairA", "pairC"],
    "merge_path_stale": ["pairA", "pairB", "pairD", "pairE"],
    "one_shot_ambiguity": [],            # observe-only: no acting/flagging arm
}
FROZEN_SEEDS = (0, 1, 2)
CLEAN_CELLS = {"clean_control"}
OBSERVE_ONLY_CELLS = {"one_shot_ambiguity"}

PASS, FAIL, INCOMPLETE = "pass", "fail", "incomplete"


# ---------------------------------------------------------------------------
# Low-level artifact readers (each records the file's sha256 in `manifest`)
# ---------------------------------------------------------------------------
def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class ShaLog:
    """Accumulates a path -> sha256 manifest for every artifact read."""

    def __init__(self) -> None:
        self.manifest: dict[str, str] = {}

    def bytes_of(self, path: Path) -> bytes:
        b = path.read_bytes()
        self.manifest[str(path)] = _sha256_bytes(b)
        return b

    def text_of(self, path: Path) -> str:
        return self.bytes_of(path).decode("utf-8")

    def json_of(self, path: Path) -> dict:
        return json.loads(self.text_of(path))

    def topk_content(self, stem: Path) -> bytes:
        """Decompressed topk bytes, accepting plain `.topk.csv` or gzipped
        `.topk.csv.gz` (the committed panel gzips topk; run_vision emits plain).
        Comparing *decompressed* content makes the inertness test robust to
        gzip-header mtime, matching the existing run-matrix `gunzip -c` compare."""
        gz = stem.with_name(stem.name + SUF_TOPK_GZ)
        plain = stem.with_name(stem.name + SUF_TOPK)
        if gz.exists():
            raw = self.bytes_of(gz)
            return gzip.decompress(raw)
        if plain.exists():
            return self.bytes_of(plain)
        raise FileNotFoundError(f"no topk artifact for {stem}")


def _stem_path(root: Path, stem: str) -> Path:
    p = Path(stem)
    return p if p.is_absolute() else (root / p)


def _artifact(stem: Path, suffix: str) -> Path:
    return stem.with_name(stem.name + suffix)


def _present_artifacts(stem: Path) -> dict[str, bool]:
    """Which of the required artifacts exist for one run stem (topk counts if
    either the plain or gzipped form is present)."""
    out = {}
    for suf in (SUF_PER_PROBE, SUF_PER_SLOT, SUF_FORK, SUF_SUMMARY):
        out[suf] = _artifact(stem, suf).exists()
    out[SUF_TOPK] = (_artifact(stem, SUF_TOPK).exists()
                     or _artifact(stem, SUF_TOPK_GZ).exists())
    out[SUF_GOVERNANCE] = _artifact(stem, SUF_GOVERNANCE).exists()
    return out


def _fork_supersession_rows(sha: ShaLog, fork_csv: Path) -> list[dict]:
    """Full committed supersession rows from a fork_events.csv (one dict per
    row). Empty for a quarantine run (its supersession writes were diverted and
    emit no row)."""
    text = sha.text_of(fork_csv)
    return [r for r in csv.DictReader(text.splitlines())
            if r.get("event_class") == SUPERSESSION_CLASS]


def _join_key(d: dict) -> tuple:
    """Normalize the JOIN_KEY tuple to strings so a CSV row (str fields) and a
    JSON ledger event (int fields) compare equal field-by-field."""
    return tuple(str(d.get(k)) for k in JOIN_KEY)


def _intrinsic_key(d: dict) -> tuple:
    """Normalize the PR-9.2 INTRINSIC_JOIN_KEY tuple to strings (both sides
    are JSON ledger events, but string-normalizing keeps the comparison robust
    to int/str drift)."""
    return tuple(str(d.get(k)) for k in INTRINSIC_JOIN_KEY)


def _mset_diff_to_list(counter: Counter) -> list:
    """A multiset difference rendered as sorted [{key, count}] records (the
    JOIN_KEY field names zipped back onto each tuple), for the JSON report."""
    out = []
    for key, n in sorted(counter.items()):
        out.append({"key": dict(zip(JOIN_KEY, key)), "count": n})
    return out


def _summary_minus_govern(summary: dict) -> dict:
    return {k: v for k, v in summary.items() if k != "govern"}


# ---------------------------------------------------------------------------
# Per-run checks
# ---------------------------------------------------------------------------
def check_inertness(sha: ShaLog, none_stem: Path, shadow_stem: Path) -> dict:
    """Every retrieval/readout artifact identical between none and shadow.

    Raw-bytes equal for per_probe/per_slot/fork_events; decompressed-content
    equal for topk; byte-equal for governance (the frozen scorer readout) when
    present on both sides; and JSON-equal for summary.json modulo the additive
    shadow ``govern`` block."""
    arts = {}
    ok = True
    for suf in INERT_RAW_SUFFIXES:
        nf, sf = _artifact(none_stem, suf), _artifact(shadow_stem, suf)
        if not (nf.exists() and sf.exists()):
            arts[suf] = {"status": INCOMPLETE, "reason": "artifact missing"}
            ok = False
            continue
        same = sha.bytes_of(nf) == sha.bytes_of(sf)
        arts[suf] = {"status": PASS if same else FAIL, "byte_identical": same}
        ok = ok and same

    # topk (decompressed-content compare)
    try:
        same_topk = sha.topk_content(none_stem) == sha.topk_content(shadow_stem)
        arts["topk"] = {"status": PASS if same_topk else FAIL,
                        "content_identical": same_topk}
        ok = ok and same_topk
    except FileNotFoundError:
        arts["topk"] = {"status": INCOMPLETE, "reason": "topk artifact missing"}
        ok = False

    # governance.json (compare only if present on both sides)
    ng, sg = (_artifact(none_stem, SUF_GOVERNANCE),
              _artifact(shadow_stem, SUF_GOVERNANCE))
    if ng.exists() and sg.exists():
        same_gov = sha.bytes_of(ng) == sha.bytes_of(sg)
        arts[SUF_GOVERNANCE] = {"status": PASS if same_gov else FAIL,
                                "byte_identical": same_gov}
        ok = ok and same_gov
    else:
        # Compare-if-present: a smoke run has no governance readout (run_vision
        # does not emit it; it needs the frozen scorer pass). Absence does NOT
        # downgrade inertness — panel mode requires governance via
        # check_panel_coverage, so the §4.7 cond.1 obligation is enforced there.
        arts[SUF_GOVERNANCE] = {"status": "absent",
                                "reason": "governance readout not present "
                                          "(needs the frozen scorer pass)"}

    # summary.json modulo the additive govern block
    ns, ss = (_artifact(none_stem, SUF_SUMMARY),
              _artifact(shadow_stem, SUF_SUMMARY))
    if ns.exists() and ss.exists():
        n_sum = _summary_minus_govern(sha.json_of(ns))
        s_sum = _summary_minus_govern(sha.json_of(ss))
        same_sum = n_sum == s_sum
        arts[SUF_SUMMARY] = {"status": PASS if same_sum else FAIL,
                             "identical_except_govern": same_sum}
        ok = ok and same_sum
    else:
        arts[SUF_SUMMARY] = {"status": INCOMPLETE, "reason": "summary missing"}
        ok = False

    # Roll up: any FAIL -> fail; any INCOMPLETE (no fail) -> incomplete.
    statuses = [a["status"] for a in arts.values()]
    if FAIL in statuses:
        status = FAIL
    elif INCOMPLETE in statuses:
        status = INCOMPLETE
    else:
        status = PASS
    return {"status": status, "artifacts": arts}


def check_ledger(sha: ShaLog, shadow_stem: Path, *, is_clean: bool,
                 expected_flagged: int | None) -> dict:
    """Validate the shadow govern/ledger block in summary.json."""
    sumf = _artifact(shadow_stem, SUF_SUMMARY)
    if not sumf.exists():
        return {"status": INCOMPLETE, "reason": "shadow summary.json missing"}
    gov = sha.json_of(sumf).get("govern")
    if not gov:
        return {"status": FAIL, "reason": "shadow summary has no govern block"}

    problems = []

    def want(cond, msg):
        if not cond:
            problems.append(msg)

    want(gov.get("action") == SHADOW_ACTION,
         f"action != {SHADOW_ACTION!r} (got {gov.get('action')!r})")
    want(gov.get("step") == SHADOW_STEP, f"step != {SHADOW_STEP!r}")
    want("quarantined_events" not in gov, "quarantined_events present on shadow")
    want("refused_events" not in gov, "refused_events present on shadow")

    flagged = gov.get("flagged_events")
    ledger = gov.get("quarantine_ledger") or {}
    want(ledger.get("disposition") == FLAGGED_DISPOSITION,
         f"disposition != {FLAGGED_DISPOSITION!r} (got "
         f"{ledger.get('disposition')!r})")
    want(ledger.get("quarantined_count") == 0,
         f"quarantined_count != 0 (got {ledger.get('quarantined_count')!r})")
    want(ledger.get("retained_recoverable") is False,
         "retained_recoverable is not False")
    want(ledger.get("absorbed_into_active_memory") is True,
         "absorbed_into_active_memory is not True")
    want(ledger.get("flagged_count") == flagged,
         "ledger.flagged_count != govern.flagged_events")
    want(ledger.get("opportunity_count") == flagged,
         "ledger.opportunity_count != flagged_events")
    hist = ledger.get("payload_label_histogram") or {}
    want(sum(hist.values()) == (flagged or 0),
         "payload_label_histogram does not sum to flagged_events")

    if is_clean:
        want(flagged == 0, f"clean cell flagged_events != 0 (got {flagged})")
        want(hist == {}, "clean cell has a non-empty payload histogram")
    else:
        want((flagged or 0) > 0, "non-clean cell flagged_events is not > 0")
        if expected_flagged is not None:
            want(flagged == expected_flagged,
                 f"flagged_events {flagged} != expected {expected_flagged}")

    return {"status": PASS if not problems else FAIL,
            "flagged_events": flagged,
            "payload_label_histogram": hist,
            "problems": problems}


def check_flag_set_identity(sha: ShaLog, none_stem: Path, shadow_stem: Path,
                            quarantine_stem: Path | None) -> dict:
    """The mandatory event-level identity check (PR8 §4 requirement 4).

    Derives a stable per-event key set on BOTH the §9A shadow side and the §8
    quarantine-diverted side, then reports set sizes, intersection, and both
    symmetric differences. Certification requires both differences empty AND that
    real per-event keys existed on both sides — equal counts are NOT sufficient.

    §8-diverted per-event keys come from the quarantine ledger's
    ``diverted_events`` list (the audit instrumentation). A diverted write never
    reaches ``write_fn`` so it has no record_seq/owner_slot of its own; both
    sides are bridged by the incumbent identity each supersession supersedes
    (:data:`JOIN_KEY`), which the shadow/none fork_events records as columns and
    the ledger records per diverted event. The two sides are compared as
    MULTISETS on that key, so equal counts alone never pass — both symmetric
    differences must be empty. A LEGACY aggregate-only ledger (no
    ``diverted_events``) yields ``identity-unprovable with current artifacts``.
    """
    shadow_rows = _fork_supersession_rows(sha, _artifact(shadow_stem, SUF_FORK))
    none_rows = _fork_supersession_rows(sha, _artifact(none_stem, SUF_FORK))
    # (record_tag, record_seq, owner_slot) anchors shadow self-consistency and
    # the provable shadow ≡ baseline sub-claim (shadow's own committed events).
    def _anchor(r):
        return (r.get("record_tag"), r.get("record_seq"), r.get("owner_slot"))
    shadow_keys = [_anchor(r) for r in shadow_rows]
    none_keys = [_anchor(r) for r in none_rows]

    shadow_gov = (sha.json_of(_artifact(shadow_stem, SUF_SUMMARY))
                  .get("govern") or {})
    shadow_ledger = shadow_gov.get("quarantine_ledger") or {}
    shadow_flagged_count = shadow_gov.get("flagged_events")
    shadow_hist = shadow_ledger.get("payload_label_histogram") or {}
    # PR-9.2: explicit per-event flagged records on the shadow side (the
    # mirror of the quarantine ledger's diverted_events). Empty on a
    # pre-instrumentation shadow run.
    shadow_records = shadow_ledger.get("flagged_event_records") or []

    self_consistent = (len(shadow_keys) == shadow_flagged_count)
    shadow_set, none_set = set(shadow_keys), set(none_keys)
    shadow_equals_baseline = (shadow_set == none_set)

    out = {
        "join_key": list(JOIN_KEY),
        "shadow_flagged_set_size": len(shadow_keys),
        "shadow_flagged_distinct_keys": len(shadow_set),
        "shadow_ledger_flagged_count": shadow_flagged_count,
        "shadow_self_consistent": self_consistent,
        "provable_subclaim_shadow_equals_baseline_supersession":
            shadow_equals_baseline,
        "baseline_supersession_set_size": len(none_keys),
    }

    # §8 quarantine-diverted per-event keys: from the ledger's diverted_events.
    q_diverted = []
    q_hist = {}
    q_count = None
    q_join_declared = None
    if quarantine_stem is not None:
        q_sum = _artifact(quarantine_stem, SUF_SUMMARY)
        if q_sum.exists():
            q_gov = sha.json_of(q_sum).get("govern") or {}
            q_ledger = q_gov.get("quarantine_ledger") or {}
            q_hist = q_ledger.get("payload_label_histogram") or {}
            q_count = q_gov.get("quarantined_events",
                                q_ledger.get("quarantined_count"))
            q_diverted = q_ledger.get("diverted_events") or []
            q_join_declared = q_ledger.get("diverted_event_join_key")

    diverted_keys_available = len(q_diverted) > 0
    out["quarantine_diverted_count"] = q_count
    out["quarantine_diverted_keys_available"] = diverted_keys_available
    out["quarantine_diverted_event_records"] = len(q_diverted)
    # Supporting (NON-certifying) evidence — never a basis for pass.
    out["non_certifying_histogram_equal"] = (shadow_hist == q_hist)
    out["non_certifying_count_equal"] = (shadow_flagged_count == q_count)
    if q_join_declared and list(q_join_declared) != list(JOIN_KEY):
        out["join_key_warning"] = (
            f"ledger declares join key {q_join_declared}; harness joins on "
            f"{list(JOIN_KEY)}")

    if not self_consistent:
        return {**out, "status": FAIL, "identity_status": "shadow-self-inconsistent",
                "reason": "shadow fork_events supersession-row count != shadow "
                          "ledger flagged_count; the ledger is not a faithful "
                          "record of committed events"}

    # Empty-set case (clean cells): nothing flagged on either side -> identity
    # is the trivial ∅ == ∅, NOT "unprovable".
    if (shadow_flagged_count or 0) == 0:
        out.update({"intersection_size": 0, "diverted_minus_flagged": [],
                    "shadow_flagged_minus_diverted": []})
        if q_count in (None, 0) and not q_diverted:
            return {**out, "status": PASS,
                    "identity_status": "identity-proven-empty"}
        return {**out, "status": FAIL, "identity_status": "identity-violated",
                "reason": f"shadow flagged 0 supersession writes but the §8 "
                          f"quarantine ledger reports {q_count} diverted"}

    # ---- PR-9.2: intrinsic-key join (the gating path) -----------------------
    # Explicit per-event records on BOTH sides -> join on the pre-registered
    # write-event-intrinsic key. Uniqueness is part of the claim: a duplicate
    # key on either side is a STOP condition (identity-collision), because the
    # key's identity semantics assume one eligible batch per (epoch, class).
    if shadow_records and diverted_keys_available:
        s_keys = [_intrinsic_key(e) for e in shadow_records]
        q_keys = [_intrinsic_key(e) for e in q_diverted]
        s_dups = [k for k, n in Counter(s_keys).items() if n > 1]
        q_dups = [k for k, n in Counter(q_keys).items() if n > 1]
        out.update({
            "intrinsic_join_key": list(INTRINSIC_JOIN_KEY),
            "intrinsic_check_fields": list(INTRINSIC_CHECK_FIELDS),
            "shadow_record_count": len(shadow_records),
            "quarantine_record_count": len(q_diverted),
        })
        if s_dups or q_dups:
            return {**out, "status": FAIL,
                    "identity_status": "identity-collision",
                    "collisions": {
                        "shadow": [dict(zip(INTRINSIC_JOIN_KEY, k))
                                   for k in sorted(s_dups)],
                        "quarantine": [dict(zip(INTRINSIC_JOIN_KEY, k))
                                       for k in sorted(q_dups)]},
                    "reason": (
                        "duplicate intrinsic key within a single arm's ledger: "
                        "the (epoch, event_class, batch_index) uniqueness claim "
                        "is violated, so per-event identity is undecidable on "
                        "this key. STOP condition — do not join as a multiset, "
                        "do not weaken the key.")}
        s_map = dict(zip(s_keys, shadow_records))
        q_map = dict(zip(q_keys, q_diverted))
        only_q = sorted(set(q_map) - set(s_map))
        only_s = sorted(set(s_map) - set(q_map))
        joined = sorted(set(s_map) & set(q_map))
        check_mismatch = []
        incumbent_agree = 0
        for k in joined:
            se, qe = s_map[k], q_map[k]
            for f in INTRINSIC_CHECK_FIELDS:
                if se.get(f) != qe.get(f):
                    check_mismatch.append(
                        {"key": dict(zip(INTRINSIC_JOIN_KEY, k)), "field": f,
                         "shadow": se.get(f), "quarantine": qe.get(f)})
            if (se.get("incumbent_slot") == qe.get("incumbent_slot")
                    and se.get("incumbent_last_write_seq")
                    == qe.get("incumbent_last_write_seq")):
                incumbent_agree += 1
        # Cross-link the shadow ledger to shadow's OWN committed fork_events
        # rows (the committed-schema anchor): per epoch, the ledger's record
        # count must equal the committed supersession-row count. (Labels are
        # NOT compared here: fork_events' injected_label is a contra-arm
        # field, -1 on supersession rows; payload consistency is checked
        # ledger-to-ledger via INTRINSIC_CHECK_FIELDS above.)
        ledger_by_epoch = Counter(str(e.get("epoch")) for e in shadow_records)
        fork_by_epoch = Counter(str(r.get("epoch")) for r in shadow_rows)
        fork_link_ok = ledger_by_epoch == fork_by_epoch
        out.update({
            "intersection_size": len(joined),
            "diverted_minus_flagged": [dict(zip(INTRINSIC_JOIN_KEY, k))
                                       for k in only_q],
            "shadow_flagged_minus_diverted": [dict(zip(INTRINSIC_JOIN_KEY, k))
                                              for k in only_s],
            "check_field_mismatches": check_mismatch,
            "shadow_ledger_fork_events_link_ok": fork_link_ok,
            "diagnostic_incumbent_key_agreement":
                {"agree": incumbent_agree, "joined": len(joined),
                 "note": ("state-contaminated legacy key; expected < joined "
                          "past the first diversion on acting pairs — "
                          "recorded, not gating")},
        })
        proven = (not only_q and not only_s and not check_mismatch
                  and fork_link_ok)
        if not fork_link_ok:
            return {**out, "status": FAIL,
                    "identity_status": "shadow-self-inconsistent",
                    "reason": "shadow flagged_event_records do not match "
                              "shadow's own committed fork_events supersession "
                              "rows (per-epoch counts or label multisets "
                              "differ)"}
        return {**out,
                "status": PASS if proven else FAIL,
                "identity_status": "identity-proven" if proven
                else "identity-violated",
                "reason": (
                    "shadow-flagged set == §8-diverted set on the "
                    "write-event-intrinsic key; every joined pair agrees on "
                    "the check fields; ledger anchored to committed "
                    "fork_events." if proven else
                    "decided on the write-event-intrinsic key: non-empty "
                    "symmetric difference or check-field contradiction — the "
                    "shadow-flagged and §8-diverted event sets provably "
                    "differ.")}

    if diverted_keys_available:
        # LEGACY path (pre-PR-9.2 shadow artifacts, no flagged_event_records):
        # multiset join on the state-contaminated incumbent key. Retained so
        # old artifacts (e.g. the committed pr8/identity_smoke) score exactly
        # as recorded; expected to report identity-violated past the first
        # diversion.
        shadow_mset = Counter(_join_key(r) for r in shadow_rows)
        q_mset = Counter(_join_key(e) for e in q_diverted)
        d_minus_f = q_mset - shadow_mset      # diverted but not flagged
        f_minus_d = shadow_mset - q_mset      # flagged but not diverted
        proven = (not d_minus_f) and (not f_minus_d)
        out.update({
            "shadow_join_multiset_size": sum(shadow_mset.values()),
            "quarantine_join_multiset_size": sum(q_mset.values()),
            "intersection_size": sum((shadow_mset & q_mset).values()),
            "diverted_minus_flagged": _mset_diff_to_list(d_minus_f),
            "shadow_flagged_minus_diverted": _mset_diff_to_list(f_minus_d),
        })
        return {**out,
                "status": PASS if proven else FAIL,
                "identity_status": "identity-proven" if proven
                else "identity-violated",
                "reason": (
                    "decided by multiset join on the incumbent-identity key; "
                    "non-empty symmetric difference means the shadow-flagged and "
                    "§8-diverted event sets differ (note: incumbent keys are "
                    "cross-arm stable only up to the first divergence)."
                    if not proven else
                    "shadow-flagged set == §8-diverted set on the incumbent "
                    "key, both symmetric differences empty.")}

    # Legacy aggregate-only ledger: no per-event keys -> unprovable, NOT pass.
    out.update({
        "intersection_size": None,
        "diverted_minus_flagged": None,
        "shadow_flagged_minus_diverted": None,
    })
    return {**out,
            "status": INCOMPLETE,
            "identity_status": "identity-unprovable with current artifacts",
            "reason": (
                "the quarantine ledger has no diverted_events list (a "
                "pre-instrumentation aggregate-only ledger records diverted "
                "events only as a payload_label_histogram). No per-event key "
                "exists on the §8 side to intersect against the shadow flagged "
                "set, so identity cannot be established (equal counts/histograms "
                "are explicitly non-certifying). Re-run quarantine with the "
                "diverted-key instrumentation (QUARANTINE_DIVERTED_JOIN_KEY) to "
                "make identity decidable.")}


# ---------------------------------------------------------------------------
# PR-9.2: instrumented quarantine re-run fidelity vs the committed §8 arm
# ---------------------------------------------------------------------------
def check_quarantine_rerun_fidelity(sha: ShaLog, q_stem: Path,
                                    q_committed_stem: Path) -> dict:
    """The instrumented quarantine RE-RUN must be the committed §8 quarantine
    arm plus nothing but the additive ledger keys.

    Retrieval artifacts (per_probe / per_slot / fork_events raw bytes, topk
    decompressed content) byte-identical to the committed stem; summary.json
    identical modulo ``govern``; the ``govern`` block identical after dropping
    the additive per-event instrumentation fields from the re-run side
    (``diverted_events`` / ``diverted_event_join_key`` — absent from
    pre-instrumentation §8 ledgers) and the REGISTRY-VERSION provenance field
    ``implemented_actions`` from both sides (it records the driver's action
    registry — which grew when PR-8 registered ``shadow`` — not anything the
    run did; every run-describing field: ``action``, ``step``,
    ``events_seen``, ``outcome_counts``, ``quarantined_events`` and the
    ledger counts/histogram, remains strictly compared). governance.json is
    NOT compared here: it is a deterministic function of the (byte-identical)
    CSVs, and the two sides were scored by different registered scorer
    versions (policy-registry growth since §8), so byte inequality there is
    expected and carries no information the CSV identity does not."""
    arts = {}
    ok = True
    for suf in INERT_RAW_SUFFIXES:
        rf, cf = _artifact(q_stem, suf), _artifact(q_committed_stem, suf)
        if not (rf.exists() and cf.exists()):
            arts[suf] = {"status": INCOMPLETE, "reason": "artifact missing"}
            ok = False
            continue
        same = sha.bytes_of(rf) == sha.bytes_of(cf)
        arts[suf] = {"status": PASS if same else FAIL, "byte_identical": same}
        ok = ok and same
    try:
        same_topk = (sha.topk_content(q_stem)
                     == sha.topk_content(q_committed_stem))
        arts["topk"] = {"status": PASS if same_topk else FAIL,
                        "content_identical": same_topk}
        ok = ok and same_topk
    except FileNotFoundError:
        arts["topk"] = {"status": INCOMPLETE, "reason": "topk missing"}
        ok = False
    rs, cs = (_artifact(q_stem, SUF_SUMMARY),
              _artifact(q_committed_stem, SUF_SUMMARY))
    if rs.exists() and cs.exists():
        r_sum, c_sum = sha.json_of(rs), sha.json_of(cs)
        same_body = (_summary_minus_govern(r_sum)
                     == _summary_minus_govern(c_sum))
        r_gov = dict(r_sum.get("govern") or {})
        c_gov = dict(c_sum.get("govern") or {})
        r_led = dict(r_gov.get("quarantine_ledger") or {})
        for additive in ("diverted_events", "diverted_event_join_key"):
            r_led.pop(additive, None)
        if "quarantine_ledger" in r_gov:
            r_gov["quarantine_ledger"] = r_led
        # registry-version provenance, not run behavior (see docstring)
        r_gov.pop("implemented_actions", None)
        c_gov.pop("implemented_actions", None)
        same_gov = r_gov == c_gov
        arts[SUF_SUMMARY] = {
            "status": PASS if (same_body and same_gov) else FAIL,
            "body_identical_except_govern": same_body,
            "govern_identical_minus_additive_keys": same_gov}
        ok = ok and same_body and same_gov
    else:
        arts[SUF_SUMMARY] = {"status": INCOMPLETE, "reason": "summary missing"}
        ok = False
    statuses = [a["status"] for a in arts.values()]
    status = (FAIL if FAIL in statuses
              else INCOMPLETE if INCOMPLETE in statuses else PASS)
    return {"status": status, "artifacts": arts}


# ---------------------------------------------------------------------------
# Cross-host determinism
# ---------------------------------------------------------------------------
def check_cross_host(sha: ShaLog, darwin_root: Path,
                     gentoo_root: Path | None,
                     gentoo_sha_manifest: Path | None = None) -> dict:
    """sha256-verify every read artifact against the canonical (gentoo) stack.

    Two accepted forms of the §4.7 cond-3 evidence:

    * ``gentoo_root`` — a literal second-host tree to byte-compare against
      (the original path; requires both trees on one filesystem);
    * ``gentoo_sha_manifest`` (PR-9.2) — a ``sha256sum``-format digest list
      computed ON gentoo over the panel tree before transfer. Every artifact
      the harness read is verified against the gentoo-computed digest of the
      same relative path. Combined with the run matrix's same-seed twin ON
      gentoo, this is the PR-10-consistent reading of "sha256-stable across
      darwin/gentoo": run artifacts are gentoo-canonical and darwin verifies
      the bytes it scored are the bytes gentoo produced. (Cross-ARCHITECTURE
      run replication was refuted program-wide in PR-10's result memo — float
      tails compound through the vigilance gate — so demanding a darwin
      recompute would fail for reasons §9A does not test.)

    Single host (neither given) -> INCOMPLETE: not required for a smoke, but
    a FINAL certification cannot pass without one of the two forms."""
    if gentoo_sha_manifest is not None:
        digests = {}
        for line in gentoo_sha_manifest.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            digest, _, rel = line.partition("  ")
            digests[rel.lstrip("./")] = digest
        mismatches = []
        checked = 0
        for darwin_path, digest in list(sha.manifest.items()):
            dp = Path(darwin_path)
            try:
                rel = str(dp.relative_to(darwin_root))
            except ValueError:
                rel = dp.name
            g = digests.get(rel)
            if g is None:
                # Committed baselines read from OUTSIDE the transferred panel
                # tree (e.g. pr7/twin none stems) have no gentoo-side digest;
                # they are covered by git object identity on both hosts.
                continue
            checked += 1
            if g != digest:
                mismatches.append({"artifact": rel, "reason": "sha256 differs"})
        return {"status": PASS if (checked and not mismatches) else
                (FAIL if mismatches else INCOMPLETE),
                "method": "gentoo-sha-manifest",
                "artifacts_checked": checked, "mismatches": mismatches}
    if gentoo_root is None:
        return {"status": INCOMPLETE,
                "reason": "single host: cross-host sha256 determinism not "
                          "verified (required before final certification)"}
    mismatches = []
    checked = 0
    for darwin_path, digest in list(sha.manifest.items()):
        dp = Path(darwin_path)
        try:
            rel = dp.relative_to(darwin_root)
        except ValueError:
            rel = Path(dp.name)
        gp = gentoo_root / rel
        if not gp.exists():
            mismatches.append({"artifact": str(rel), "reason": "absent on gentoo"})
            continue
        checked += 1
        if _sha256_bytes(gp.read_bytes()) != digest:
            mismatches.append({"artifact": str(rel), "reason": "sha256 differs"})
    return {"status": PASS if not mismatches else FAIL,
            "artifacts_checked": checked, "mismatches": mismatches}


# ---------------------------------------------------------------------------
# Panel coverage
# ---------------------------------------------------------------------------
def check_panel_coverage(runs: list[dict], seeds, mode: str,
                         present_by_run: list[dict]) -> dict:
    """Verify the manifest covers the frozen matrix and no declared run is
    missing a required artifact. In ``panel`` mode any gap is a FAIL; in
    ``smoke`` mode gaps are expected -> INCOMPLETE."""
    have = {(r["cell"], r["pair"], r["seed"]) for r in runs}
    missing_cells = []
    for cell, pairs in FROZEN_PANEL.items():
        if cell in OBSERVE_ONLY_CELLS:
            continue
        for pair in pairs:
            for seed in seeds:
                if (cell, pair, seed) not in have:
                    missing_cells.append(f"{cell}/{pair}/s{seed}")

    missing_artifacts = []
    for meta in present_by_run:
        for arm, present in meta["present"].items():
            for suf, ok in present.items():
                # governance readout is required for real certification (§4.7
                # cond.1) but absent in a smoke (no scorer pass); per_probe,
                # per_slot, fork_events, topk, summary are always required.
                if suf == SUF_GOVERNANCE and mode == "smoke":
                    continue
                if not ok:
                    missing_artifacts.append(
                        f"{meta['cell']}/{meta['pair']}/s{meta['seed']}/{arm}{suf}")

    complete = not missing_cells and not missing_artifacts
    if complete:
        status = PASS
    elif mode == "smoke":
        status = INCOMPLETE
    else:
        status = FAIL
    return {"status": status, "mode": mode,
            "missing_cells": missing_cells,
            "missing_artifacts": missing_artifacts}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_certification(manifest: dict, *, manifest_path: str | None = None,
                      out_json: str | None = None,
                      out_md: str | None = None) -> dict:
    sha = ShaLog()
    root = Path(manifest.get("artifact_root", "."))
    mode = manifest.get("mode", "panel")
    seeds = manifest.get("seeds", list(FROZEN_SEEDS))
    gentoo_root = (Path(manifest["gentoo_root"])
                   if manifest.get("gentoo_root") else None)
    runs = manifest.get("runs", [])

    per_run = []
    present_by_run = []
    for r in runs:
        cell, pair, seed = r["cell"], r["pair"], r["seed"]
        is_clean = cell in CLEAN_CELLS
        stems = r["stems"]
        none_stem = _stem_path(root, stems["none"])
        shadow_stem = _stem_path(root, stems["shadow"])
        q_stem = (_stem_path(root, stems["quarantine"])
                  if stems.get("quarantine") else None)

        present = {"none": _present_artifacts(none_stem),
                   "shadow": _present_artifacts(shadow_stem)}
        if q_stem is not None:
            present["quarantine"] = _present_artifacts(q_stem)
        present_by_run.append({"cell": cell, "pair": pair, "seed": seed,
                               "present": present})

        inert = check_inertness(sha, none_stem, shadow_stem)
        ledger = check_ledger(sha, shadow_stem, is_clean=is_clean,
                              expected_flagged=r.get("expected_flagged"))
        identity = check_flag_set_identity(sha, none_stem, shadow_stem, q_stem)
        # PR-9.2: optional fidelity check of the instrumented quarantine
        # re-run against the committed §8 arm (additive-only ledger growth).
        fidelity = None
        if q_stem is not None and stems.get("quarantine_committed"):
            fidelity = check_quarantine_rerun_fidelity(
                sha, q_stem, _stem_path(root, stems["quarantine_committed"]))
        clean = None
        if is_clean:
            clean = {"status": inert["status"],
                     "flagged_events": ledger.get("flagged_events"),
                     "byte_inert": inert["status"] == PASS,
                     "flagged_zero": ledger.get("flagged_events") == 0}
            if clean["byte_inert"] and clean["flagged_zero"]:
                clean["status"] = PASS
            elif inert["status"] == FAIL or ledger.get("flagged_events"):
                clean["status"] = FAIL
        per_run.append({
            "cell": cell, "pair": pair, "seed": seed, "is_clean": is_clean,
            "inertness": inert, "ledger": ledger,
            "flag_set_identity": identity,
            "quarantine_rerun_fidelity": fidelity, "clean_control": clean})

    panel = check_panel_coverage(runs, seeds, mode, present_by_run)
    gentoo_sha_manifest = (Path(manifest["gentoo_sha_manifest"])
                           if manifest.get("gentoo_sha_manifest") else None)
    cross_host = check_cross_host(sha, root, gentoo_root,
                                  gentoo_sha_manifest=gentoo_sha_manifest)

    # ---- roll up each dimension across runs --------------------------------
    def rollup(key, sub=None):
        sts = []
        for pr in per_run:
            node = pr[key]
            if node is None:
                continue
            sts.append(node["status"])
        if FAIL in sts:
            return FAIL
        if not sts:
            return INCOMPLETE
        if INCOMPLETE in sts:
            return INCOMPLETE
        return PASS

    inertness_status = rollup("inertness")
    ledger_status = rollup("ledger")
    identity_status = rollup("flag_set_identity")
    clean_runs = [pr for pr in per_run if pr["is_clean"]]
    if not clean_runs:
        clean_status = INCOMPLETE
    elif any(pr["clean_control"]["status"] == FAIL for pr in clean_runs):
        clean_status = FAIL
    elif all(pr["clean_control"]["status"] == PASS for pr in clean_runs):
        clean_status = PASS
    else:
        clean_status = INCOMPLETE

    # PR-9.2: fidelity rolls up over the runs where a committed §8 quarantine
    # counterpart was declared (a cell with no committed acting arm — e.g.
    # pairC clean — has nothing to compare and is not counted). The dimension
    # exists only when at least one run declares ``quarantine_committed``: it
    # is a PR-9.2 strengthening check, not a §4.7 condition, so manifests
    # that never declare it (pre-PR-9.2 smokes, fixtures) are not demoted.
    fid_sts = [pr["quarantine_rerun_fidelity"]["status"] for pr in per_run
               if pr["quarantine_rerun_fidelity"] is not None]

    dimensions = {
        "inertness": inertness_status,
        "ledger_coverage": ledger_status,
        "flag_set_identity": identity_status,
        "clean_control": clean_status,
        "panel_coverage": panel["status"],
        "cross_host_determinism": cross_host["status"],
    }
    if fid_sts:
        dimensions["quarantine_rerun_fidelity"] = (
            FAIL if FAIL in fid_sts
            else PASS if all(s == PASS for s in fid_sts) else INCOMPLETE)

    # ---- verdict -----------------------------------------------------------
    if FAIL in dimensions.values():
        verdict = FAIL
    elif all(v == PASS for v in dimensions.values()):
        verdict = PASS
    else:
        verdict = INCOMPLETE

    missing_artifacts = panel["missing_artifacts"]
    report = {
        "panel": manifest.get("panel", "pr8-9a"),
        "readout": manifest.get("readout"),
        "mode": mode,
        "verdict": verdict,
        "certifies": ("promoted-audit (detector/forensic role only; NOT "
                      "read-time remediation, recovery, or governance promotion)"
                      if verdict == PASS else None),
        "dimensions": dimensions,
        "flag_set_identity_status": (
            per_run[0]["flag_set_identity"]["identity_status"]
            if per_run else "no-runs"),
        "artifact_roots": {
            "darwin": str(root),
            "gentoo": str(gentoo_root) if gentoo_root else None,
        },
        "missing_artifacts": missing_artifacts,
        "panel_coverage": panel,
        "cross_host_determinism": cross_host,
        "runs": per_run,
        "commands": {
            "regenerate_panel": (
                "for arm in none shadow quarantine; do "
                "python benchmarks/failure_mode_probe.py --vision "
                "--vision-cache <#87-cache> --arm <arm> --govern $arm "
                "--seed <seed> --out <root>/<cell>/$arm/<stem>.csv; done"),
            "run_harness": (
                "python benchmarks/pr8_shadow_audit_cert.py "
                f"--manifest {manifest_path or '<manifest.json>'} "
                "--out-json <report.json> --out-md <report.md>"),
        },
        "sha256_manifest": sha.manifest,
    }

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(report, indent=2, sort_keys=False))
    if out_md:
        Path(out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(out_md).write_text(render_markdown(report))
    return report


def render_markdown(report: dict) -> str:
    d = report["dimensions"]
    L = []
    L.append(f"# PR-8 §9A shadow-quarantine certification — `{report['verdict'].upper()}`")
    L.append("")
    L.append(f"- **panel**: {report['panel']} · **readout**: {report['readout']} "
             f"· **mode**: {report['mode']}")
    if report["verdict"] != PASS:
        L.append(f"- **NOT a certification** — verdict is "
                 f"`{report['verdict']}`; `promoted-audit` is asserted only on "
                 f"`pass`.")
    else:
        L.append(f"- **certifies**: {report['certifies']}")
    L.append("")
    L.append("## Dimensions")
    L.append("")
    L.append("| dimension | status |")
    L.append("|---|---|")
    for k, v in d.items():
        L.append(f"| {k} | **{v}** |")
    L.append("")
    L.append(f"**Flag-set identity:** `{report['flag_set_identity_status']}`")
    if report["runs"]:
        fid = report["runs"][0]["flag_set_identity"]
        L.append("")
        L.append(f"- shadow flagged set (distinct per-event keys): "
                 f"{fid.get('shadow_flagged_distinct_keys')}")
        L.append(f"- §8 quarantine diverted count: "
                 f"{fid.get('quarantine_diverted_count')} "
                 f"(per-event keys available: "
                 f"{fid.get('quarantine_diverted_keys_available')})")
        L.append(f"- intersection: {fid.get('intersection_size')} · "
                 f"diverted−flagged: {fid.get('diverted_minus_flagged')} · "
                 f"flagged−diverted: {fid.get('shadow_flagged_minus_diverted')}")
        L.append(f"- provable sub-claim (shadow ≡ baseline supersession set): "
                 f"{fid.get('provable_subclaim_shadow_equals_baseline_supersession')}")
        if fid.get("reason"):
            L.append(f"- _reason_: {fid['reason']}")
    L.append("")
    L.append("## Artifact roots")
    L.append(f"- darwin: `{report['artifact_roots']['darwin']}`")
    L.append(f"- gentoo: `{report['artifact_roots']['gentoo']}`")
    L.append("")
    if report["missing_artifacts"]:
        L.append("## Missing artifacts")
        for m in report["missing_artifacts"]:
            L.append(f"- `{m}`")
        L.append("")
    L.append("## Commands")
    for name, cmd in report["commands"].items():
        L.append(f"- **{name}**: `{cmd}`")
    L.append("")
    L.append(f"_sha256 manifest: {len(report['sha256_manifest'])} artifacts "
             f"hashed._")
    L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="PR-8 §9A shadow-quarantine certification (analysis-only).")
    ap.add_argument("--manifest", required=True,
                    help="run manifest JSON (see module docstring)")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text())
    report = run_certification(manifest, manifest_path=args.manifest,
                               out_json=args.out_json, out_md=args.out_md)
    print(render_markdown(report))
    # exit non-zero on a HARD failure so CI can gate; incomplete is exit 0 (it
    # is an honest "not yet certifiable", not a defect).
    return 1 if report["verdict"] == FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
