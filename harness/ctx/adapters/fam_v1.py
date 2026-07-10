#!/usr/bin/env python3
"""PR-13 Adapter A — FAM run artifacts (`fam-v1`, memo §4.1).

Consumes the committed raw run artifacts for one cell, read-only:
    per_probe_<cell>.csv            (probe-level observables)
    per_probe_<cell>.topk.csv.gz    (rank-ordered retrieval)
    per_probe_<cell>.per_slot.csv   (engine slot flags per epoch)

Emits context items (one per surviving top-k slot per probe) plus
not-retrieved records (non-surviving ranks), with signals:
    merge_suspect   — per_slot.is_merge_candidate, tier core-certified
                      (PR-10 abstention passthrough; the ONLY signal the
                      adapter may emit at core-certified tier)
    stale_support   — a surviving supporting slot is stale-superseded
                      (attached to the rank-0 item)
    contra_fork     — per_slot.is_contra_fork
    superseded_by   — per_slot.is_stale_superseded
    oneshot_tie     — top1/top2 margin ≤ TIE_EPS (registered 1e-3);
                      rank-0/rank-1 items share a candidate_set_id
    source_identity — benign baseline (every item carries ≥1 record)

LABEL-BLINDNESS (G-C2, structural): all reads go through an explicit
POLICY-VISIBLE COLUMN WHITELIST. Truth columns (true_label, *_correct,
true_margin, failure_mode, contradictory_*, stale_strict/lenient,
n_contra_topk, n_stale_topk, *_vote_weight, role, record_tag,
injected_label, ...) are never accessed on any code path; the
poisoned-label canary test permutes them and asserts byte-identical
output. The normative enforcement is the canary; this whitelist is the
advisory, function-granular part.

WITNESS-ALT (memo §4.1 sixth signal): implemented in
``emit_witness_alt_from_packets`` but NOT wired into ``emit`` —
``pol_f1b``'s RowObs/CellCtx are defined over committed W2 packet
trees, while memo §7 says items re-derive from raw run artifacts "not
from the old packets". Whether packet-read-only signal derivation is
inside that boundary is a recorded specification question (memo §10
ambiguity discipline); the function ships, disabled, pending explicit
adjudication.

Deterministic: sorted iteration, string values carried verbatim from
the CSVs into rendered content (no float re-formatting), no clock, no
RNG, no truth file opened.
"""

import csv
import gzip
import hashlib
import os

TIE_EPS = 1e-3

# Policy-visible columns, per file. Nothing outside these is ever read.
VISIBLE_PER_PROBE = ("epoch", "probe_index", "top1_slot", "top1_top2_margin",
                     "top1_sim", "top2_sim", "n_surviving_votes",
                     "served_outcome", "abstain_reason")
VISIBLE_TOPK = ("epoch", "probe_index", "rank", "slot", "sim", "surviving",
                "weight", "decode")
VISIBLE_PER_SLOT = ("epoch", "slot", "decode", "hit_counts",
                    "last_write_seq", "usage", "n_records", "is_contra_fork",
                    "is_stale_superseded", "is_current_fork",
                    "is_merge_candidate")

ADAPTER_ID = "fam-v1"
TEMPLATE_ID = "ctx-fam-item-v1"


def _pick(row, visible):
    return {k: row[k] for k in visible if k in row}


def _read_csv(path, visible, open_fn=open):
    with open_fn(path, "rt", encoding="utf-8") as fh:
        return [_pick(row, visible) for row in csv.DictReader(fh)]


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _stem(runs_dir, cell):
    return os.path.join(runs_dir, f"per_probe_{cell}")


def emit(runs_dir, cell, policy):
    """Map one cell's raw run artifacts → adapter_output dict
    (schema: harness/ctx/schema/adapter_output.schema.json)."""
    stem = _stem(runs_dir, cell)
    per_probe = _read_csv(stem + ".csv", VISIBLE_PER_PROBE)
    topk = _read_csv(stem + ".topk.csv.gz", VISIBLE_TOPK, open_fn=gzip.open)
    per_slot = _read_csv(stem + ".per_slot.csv", VISIBLE_PER_SLOT)

    template = policy["render_templates"][TEMPLATE_ID]
    slot_flags = {(r["epoch"], r["slot"]): r for r in per_slot}
    probe_obs = {(r["epoch"], r["probe_index"]): r for r in per_probe}

    by_query = {}
    for row in topk:
        by_query.setdefault((row["epoch"], row["probe_index"]), []).append(row)

    items, not_retrieved = [], []
    for (epoch, probe) in sorted(by_query, key=lambda q: (int(q[0]), int(q[1]))):
        ranks = sorted(by_query[(epoch, probe)], key=lambda r: int(r["rank"]))
        obs = probe_obs.get((epoch, probe), {})
        margin = float(obs["top1_top2_margin"]) if obs.get("top1_top2_margin") else None
        tie = margin is not None and margin <= TIE_EPS
        tie_set = f"{ADAPTER_ID}:{cell}:e{epoch}:p{probe}:tie"

        surviving = [r for r in ranks if r["surviving"] == "1"]
        stale_in_support = any(
            slot_flags.get((epoch, r["slot"]), {}).get("is_stale_superseded") == "1"
            for r in surviving)

        for r in ranks:
            ptr = f"topk.csv.gz:{cell}:e{epoch}:p{probe}:rank{r['rank']}"
            if r["surviving"] != "1":
                not_retrieved.append({
                    "native_id": f"{cell}:e{epoch}:p{probe}:slot{r['slot']}",
                    "reason": "not_surviving_engine",
                })
                continue
            flags = slot_flags.get((epoch, r["slot"]), {})
            content = template.format(slot=r["slot"], decode=r["decode"],
                                      rank=r["rank"], sim=r["sim"])
            evidence = [{
                "adapter_id": ADAPTER_ID, "signal": "source_identity",
                "value": cell, "evidence_ptr": ptr,
                "tier": "harness-heuristic",
            }]
            state = "agent-readable"
            slot_ptr = f"per_slot.csv:{cell}:e{epoch}:slot{r['slot']}"
            if flags.get("is_merge_candidate") == "1":
                evidence.append({
                    "adapter_id": ADAPTER_ID, "signal": "merge_suspect",
                    "value": True, "evidence_ptr": slot_ptr,
                    "tier": "core-certified",
                })
            if flags.get("is_contra_fork") == "1":
                evidence.append({
                    "adapter_id": ADAPTER_ID, "signal": "contra_fork",
                    "value": True, "evidence_ptr": slot_ptr,
                    "tier": "harness-heuristic",
                })
            if flags.get("is_stale_superseded") == "1":
                state = "stale"
                evidence.append({
                    "adapter_id": ADAPTER_ID, "signal": "superseded_by",
                    "value": True, "evidence_ptr": slot_ptr,
                    "tier": "harness-heuristic",
                })
            if r["rank"] == "0" and stale_in_support:
                evidence.append({
                    "adapter_id": ADAPTER_ID, "signal": "stale_support",
                    "value": True,
                    "evidence_ptr": f"per_slot.csv:{cell}:e{epoch}:support",
                    "tier": "harness-heuristic",
                })
            in_tie = tie and r["rank"] in ("0", "1")
            if in_tie:
                evidence.append({
                    "adapter_id": ADAPTER_ID, "signal": "oneshot_tie",
                    "value": obs.get("top1_top2_margin"),
                    "evidence_ptr": f"per_probe.csv:{cell}:e{epoch}:p{probe}:margin",
                    "tier": "harness-heuristic",
                })
            items.append({
                "item_id": f"{ADAPTER_ID}:{cell}:e{epoch}:p{probe}:slot{r['slot']}",
                "source_id": ADAPTER_ID,
                "content": content,
                "content_kind": "adapter-rendered",
                "content_sha256": _sha(content),
                "event_time": None,
                "ingest_time": None,
                "evidence": evidence,
                "relations": {"supersedes": [], "contradicts": [],
                              "candidate_set_id": tie_set if in_tie else None},
                "state": state,
                "policy_version": f"{policy['version']}+{TEMPLATE_ID}",
            })

    return {
        "adapter_id": ADAPTER_ID,
        "adapter_version": "1.0",
        "source_id": ADAPTER_ID,
        "items": items,
        "not_retrieved": not_retrieved,
    }


def emit_witness_alt_from_packets(packet_dir, policy):  # pragma: no cover
    """Memo §4.1 sixth signal — NOT WIRED (see module docstring).

    Would import the frozen policy block read-only from the canonical
    source (harness/action_boundary_score.py, memo §0/§4.1) and emit
    witness_alt_candidate_set at tier harness-heuristic over completed
    W2 packet trees. Disabled pending adjudication of the recorded
    specification question: whether packet-read-only signal derivation
    is compatible with memo §7's "re-derived from run artifacts (not
    from the old packets)".
    """
    raise NotImplementedError(
        "witness_alt_candidate_set is registered but unwired: awaiting "
        "explicit adjudication of the packets-vs-run-artifacts referent "
        "(memo §10 ambiguity discipline).")
