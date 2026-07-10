#!/usr/bin/env python3
"""PR-13 registered query sources (memo §7, sample registration §12 R-1).

Turns each registered cell into ``{query_id: bundle}`` where a bundle is

    {"items": [...], "not_retrieved": [...], "query_text": str,
     "turn_state": {...}}

The sampler (``harness/ctx/sample.py``) and the sealed replay runner
(``harness/ctx/replay.py``) both consume these bundles, so the arms, the
strata, and the manifest are all defined over one construction of the
evidence — there is no second path by which a query could acquire a
different item set in a different arm.

Stable query identities are exactly those registered in §12 R-1:
    FAM        fam-v1:<cell_id>:e<epoch>:p<probe>
    organic    organic:<kind>:<path>
    multi-turn mt:<kind>:<path>#t<turn>

Note the FAM identity keys on the registered ``cell_id``, not on the raw
run stem's cell name (``stale-oneshot_pairB_s0``), which lives on inside
``item_id`` as adapter provenance.

Deterministic: no clock (the organic clocks are scripted constants), no
RNG, no network, no truth column. Nothing here renders a context block.
"""

import json
import os
import tempfile

from harness.ctx import cells
from harness.ctx.adapters import fam_v1, shutterdeck_v1


def _fam_epoch_probe(item_id):
    """``fam-v1:<stem_cell>:e<epoch>:p<probe>:slot<slot>`` → (epoch, probe).

    Stem-cell names carry ``-`` and ``_`` but never ``:``, so positional
    splitting is exact.
    """
    parts = item_id.split(":")
    return parts[2][1:], parts[3][1:]


def _fam_not_retrieved_key(native_id):
    """``<stem_cell>:e<epoch>:p<probe>:slot<slot>`` → (epoch, probe)."""
    parts = native_id.split(":")
    return parts[1][1:], parts[2][1:]


def fam_cell_bundles(cell_id, stem_cell, packet_dir, policy, runs_dir=None):
    """One registered FAM cell → its query bundles, keyed by query_id.

    The witness-alt signal is read from the committed W2 packet tree
    read-only (clarification C-1); items derive exclusively from the raw
    run artifacts.
    """
    runs_dir = cells.RUNS_DIR if runs_dir is None else runs_dir
    witness_alt = fam_v1.emit_witness_alt_from_packets(packet_dir)
    out = fam_v1.emit(runs_dir, stem_cell, policy, witness_alt=witness_alt)

    bundles = {}

    def _bundle(epoch, probe):
        qid = f"fam-v1:{cell_id}:e{epoch}:p{probe}"
        if qid not in bundles:
            bundles[qid] = {
                "items": [], "not_retrieved": [],
                "query_text": cells.FAM_QUERY_TEMPLATE.format(probe=probe,
                                                              epoch=epoch),
                "turn_state": {"turn_index": 1, "prior_rendered": {}},
            }
        return bundles[qid]

    for item in out["items"]:
        epoch, probe = _fam_epoch_probe(item["item_id"])
        _bundle(epoch, probe)["items"].append(item)
    for rec in out["not_retrieved"]:
        epoch, probe = _fam_not_retrieved_key(rec["native_id"])
        _bundle(epoch, probe)["not_retrieved"].append(rec)

    # A probe with no surviving slot has no items and cannot be compiled
    # (the item schema requires ≥1 evidence record on ≥1 item); it is not
    # a query. Dropped explicitly and counted by the caller, never silently.
    return {qid: b for qid, b in bundles.items() if b["items"]}


def _item_path(item):
    for ev in item["evidence"]:
        if ev["signal"] == "source_identity":
            return ev["value"]
    raise ValueError(f"item {item['item_id']} carries no source_identity")


def _by_path(adapter_out):
    grouped = {}
    for item in adapter_out["items"]:
        grouped.setdefault(_item_path(item), []).append(item)
    return grouped


def organic_bundles(policy, ledger=None, files_root=None):
    """The organic cell (corpus=synthetic): one Q-SD1 query per registered
    path; its context is that path's ingest events at the replay clock."""
    ledger = cells.SYNTHETIC_LEDGER if ledger is None else ledger
    files_root = cells.ORIGINAL_FILES_ROOT if files_root is None else files_root
    out = shutterdeck_v1.emit(ledger, files_root, cells.ORGANIC_REPLAY_CLOCK,
                              cells.ORGANIC_TTL_SECONDS, policy)
    grouped = _by_path(out)
    with open(cells.SYNTHETIC_QUERIES, encoding="utf-8") as fh:
        queries = json.load(fh)

    bundles = {}
    for q in queries:
        path = q["query_id"].split(":", 2)[2]
        bundles[q["query_id"]] = {
            "items": sorted(grouped[path], key=lambda i: i["item_id"]),
            "not_retrieved": [],
            "query_text": q["text"],
            "turn_state": {"turn_index": 1, "prior_rendered": {}},
            "kind": q["kind"],
        }
    return bundles


def multiturn_bundles(policy, ledger=None, files_root=None, workdir=None):
    """Multi-turn withdrawal/control sessions.

    Each turn gets a visibility snapshot of the ledger at that turn's
    scripted clock, so the re-ingest *arrives between turns* rather than
    being visible from turn 1. ``turn_state.prior_rendered`` is threaded
    by the runner (it depends on what the previous turn actually
    rendered), so it is left empty here and the runner fills it in.
    """
    ledger = cells.SYNTHETIC_LEDGER if ledger is None else ledger
    files_root = cells.ORIGINAL_FILES_ROOT if files_root is None else files_root
    with open(cells.SYNTHETIC_SESSIONS, encoding="utf-8") as fh:
        sessions = json.load(fh)

    tmp = workdir or tempfile.mkdtemp(prefix="pr13_mt_")
    snapshots = {}
    for clock in cells.MULTITURN_CLOCKS:
        snap = os.path.join(tmp, f"snap_{clock.replace(':', '')}.db")
        if not os.path.exists(snap):
            cells.build_ledger_snapshot(
                ledger, snap, shutterdeck_v1._parse_rfc3339(clock))
        snapshots[clock] = _by_path(
            shutterdeck_v1.emit(snap, files_root, clock,
                                cells.ORGANIC_TTL_SECONDS, policy))

    bundles = {}
    for s in sessions:
        for turn, clock in enumerate(s["turn_clocks"], start=1):
            qid = f"{s['session_id']}#t{turn}"
            items = snapshots[clock].get(s["target_path"], [])
            bundles[qid] = {
                "items": sorted(items, key=lambda i: i["item_id"]),
                "not_retrieved": [],
                "query_text": s["query_text"],
                "turn_state": {"turn_index": turn, "prior_rendered": {}},
                "session_id": s["session_id"],
                "turn": turn,
                "kind": s["kind"],
            }
    return bundles


def fam_all_bundles(policy, fam_cells=None, runs_dir=None):
    """Every registered FAM cell → ``{cell_id: (role, bundles)}``."""
    fam_cells = cells.FAM_CELLS if fam_cells is None else fam_cells
    out = {}
    for cell_id, stem_cell, packet_dir, role in fam_cells:
        out[cell_id] = (role, fam_cell_bundles(cell_id, stem_cell,
                                               packet_dir, policy,
                                               runs_dir=runs_dir))
    return out
