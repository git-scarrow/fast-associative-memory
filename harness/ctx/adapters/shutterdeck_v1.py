#!/usr/bin/env python3
"""PR-13 Adapter B — Shutter-Deck ingest ledger (`shutterdeck-v1`, memo §4.2).

Consumes the ingestion service's SQLite metadata ledger plus file
store, read-only. Shutter-Deck is an image pipeline (a defunct photo
project; the in-repo corpus is a single old test-drop fixture) — the
ledger carries no source text, so every item is
``content_kind = adapter-rendered`` via the registered template
``ctx-sd-item-v1`` bound into policy_version.

Signals:
    freshness           — benign; ingest within TTL of the REPLAY clock
    stale_ttl           — adverse; ingest older than TTL at the replay clock
    superseded_by_path  — same source path re-ingested later
    duplicate_content   — SHA-256 over file bytes matches another entry
                          (the ledger stores no content hash; memo §4.2)
    signal_unavailable  — the file is absent from the store: the
                          duplicate check is UNAVAILABLE, never false
    source_identity     — benign baseline

The replay clock is an argument (RFC3339), never the wall clock: the
compiler/adapter layer is forbidden real time (G-C1), so the corpus's
real-world age is invisible by construction.

``pin_corpus`` hashes the ledger and every referenced file so the cell
definition is independent of the fixture's continued existence in the
repo (it is defunct-project residue and may be cleaned up someday).

Label-blindness: the ledger has no truth column; ground truth for the
organic cell is constructed by the scripted re-ingest driver outside
this adapter and never read here. Deterministic: rows sorted, times
rendered in fixed UTC RFC3339, stdlib-only.
"""

import hashlib
import os
import sqlite3
from datetime import datetime, timezone

ADAPTER_ID = "shutterdeck-v1"
TEMPLATE_ID = "ctx-sd-item-v1"


def _rfc3339(epoch_seconds):
    return datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc) \
        .strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_rfc3339(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def _sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_ledger(db_path):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT id, filepath, timestamp, assigned_slot FROM ingested_images "
            "ORDER BY filepath, timestamp, id").fetchall()
    finally:
        conn.close()
    return rows


def emit(db_path, files_root, replay_clock, ttl_seconds, policy):
    """Map the ledger → adapter_output dict (adapter_output.schema.json).

    replay_clock: RFC3339 string — the scripted cell clock (memo §4.2).
    ttl_seconds: registered freshness TTL from cell policy.
    """
    template = policy["render_templates"][TEMPLATE_ID]
    now = _parse_rfc3339(replay_clock)
    rows = _read_ledger(db_path)

    # Later ingest of the same filepath supersedes earlier ones.
    latest_by_path = {}
    for rid, path, ts, slot in rows:
        key = latest_by_path.get(path)
        if key is None or (ts, rid) > key[:2]:
            latest_by_path[path] = (ts, rid)

    # Content-duplicate detection over file bytes (hash absent from ledger).
    file_sha = {}
    for rid, path, ts, slot in rows:
        full = os.path.join(files_root, os.path.basename(path))
        if path not in file_sha:
            file_sha[path] = _sha_file(full) if os.path.exists(full) else None
    sha_owners = {}
    for path, digest in sorted(file_sha.items()):
        if digest is not None:
            sha_owners.setdefault(digest, []).append(path)

    items = []
    for rid, path, ts, slot in rows:
        ingest_time = _rfc3339(ts)
        content = template.format(filepath=path, ingest_time=ingest_time,
                                  assigned_slot=slot)
        ledger_ptr = f"metadata.db:ingested_images:id={rid}"
        evidence = [{
            "adapter_id": ADAPTER_ID, "signal": "source_identity",
            "value": path, "evidence_ptr": ledger_ptr,
            "tier": "source-asserted",
        }]
        state = "agent-readable"

        if now - float(ts) > ttl_seconds:
            evidence.append({
                "adapter_id": ADAPTER_ID, "signal": "stale_ttl",
                "value": {"age_seconds": round(now - float(ts), 3),
                          "ttl_seconds": ttl_seconds},
                "evidence_ptr": ledger_ptr, "tier": "harness-heuristic",
            })
        else:
            evidence.append({
                "adapter_id": ADAPTER_ID, "signal": "freshness",
                "value": {"age_seconds": round(now - float(ts), 3),
                          "ttl_seconds": ttl_seconds},
                "evidence_ptr": ledger_ptr, "tier": "harness-heuristic",
            })

        superseded = latest_by_path[path][1] != rid
        supersedes = []
        if superseded:
            state = "superseded"
            evidence.append({
                "adapter_id": ADAPTER_ID, "signal": "superseded_by_path",
                "value": f"{ADAPTER_ID}:row{latest_by_path[path][1]}",
                "evidence_ptr": ledger_ptr, "tier": "harness-heuristic",
            })
        else:
            earlier = [r for r, p, t, s in rows
                       if p == path and r != rid]
            supersedes = [f"{ADAPTER_ID}:row{r}" for r in sorted(earlier)]

        digest = file_sha[path]
        if digest is None:
            evidence.append({
                "adapter_id": ADAPTER_ID, "signal": "signal_unavailable",
                "value": "duplicate_content",
                "evidence_ptr": ledger_ptr, "tier": "harness-heuristic",
            })
        elif len(sha_owners[digest]) > 1:
            evidence.append({
                "adapter_id": ADAPTER_ID, "signal": "duplicate_content",
                "value": sorted(p for p in sha_owners[digest] if p != path),
                "evidence_ptr": f"filestore:sha256:{digest}",
                "tier": "harness-heuristic",
            })

        items.append({
            "item_id": f"{ADAPTER_ID}:row{rid}",
            "source_id": ADAPTER_ID,
            "content": content,
            "content_kind": "adapter-rendered",
            "content_sha256": _sha_text(content),
            "event_time": None,
            "ingest_time": ingest_time,
            "evidence": evidence,
            "relations": {"supersedes": supersedes, "contradicts": [],
                          "candidate_set_id": None},
            "state": state,
            "policy_version": f"{policy['version']}+{TEMPLATE_ID}",
        })

    items.sort(key=lambda i: i["item_id"])
    return {
        "adapter_id": ADAPTER_ID,
        "adapter_version": "1.0",
        "source_id": ADAPTER_ID,
        "items": items,
        "not_retrieved": [],
    }


def pin_corpus(db_path, files_root):
    """Hash the ledger and every referenced file for the cell definition,
    making the scoring run independent of the fixture's survival."""
    rows = _read_ledger(db_path)
    files = {}
    for rid, path, ts, slot in rows:
        full = os.path.join(files_root, os.path.basename(path))
        files[path] = _sha_file(full) if os.path.exists(full) else None
    return {
        "metadata_db_sha256": _sha_file(db_path),
        "n_rows": len(rows),
        "files": dict(sorted(files.items())),
    }
