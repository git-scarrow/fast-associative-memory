#!/usr/bin/env python3
"""PR-13 §7 cell registry and instantiation (scoring-run preparation).

Three cell families, exactly as registered:

* FAM cells — the pr12_3/pr12_4 one-shot and contra packet-tree cells
  plus the clean/pairA/s0 control, re-derived from the committed raw
  run artifacts (pr10/governed run stems) through adapter A. The
  packet directory is used ONLY as the witness-alt evidence surface
  under clarification C-1 (read-only; no item or disposition recycled).
* Organic cell — a synthetic re-ingest ledger built from the original
  Shutter-Deck fixture's path list (the fixture itself is defunct-
  project residue with zero organic supersessions, so the registered
  fallback is certain: corpus=synthetic, carried into the verdict
  string, and G-U3 downgrades to a reported metric). The ORIGINAL
  test_data/metadata.db is never written; the synthetic ledger is a
  new file with scripted timestamps.
* Multi-turn withdrawal cells — scripted 3-turn sessions over the
  synthetic ledger: per-turn visibility snapshots make adverse
  evidence (a re-ingest) arrive between turn 1 and turn 3; control
  sessions use never-re-ingested paths at the same clocks.

Everything here is deterministic and clock-free: all times are
scripted constants. NOTHING in this module renders a compiled block —
instantiation produces inputs only; the first evaluation render
belongs to the sealed replay (memo §8.1 selection-timing kill).
"""

import os
import sqlite3

BASE = os.path.join("results", "issue_failure_mode_blindness")
RUNS_DIR = os.path.join(BASE, "pr10", "governed")

# --- FAM cells (memo §7 bullet 1) ------------------------------------------
# (cell_id, run-stem cell name for fam_v1.emit, packet dir for the
#  witness-alt surface (W2, clarification C-1), role)
FAM_CELLS = [
    ("clean_pairA_s0", "clean_pairA_s0",
     os.path.join(BASE, "pr12_3", "W2", "clean_pairA_s0"), "clean-control"),
    ("pairB_oneshot_s0", "stale-oneshot_pairB_s0",
     os.path.join(BASE, "pr12_3", "W2", "pairB_oneshot_s0"), "harm"),
    ("pairB_oneshot_s1", "stale-oneshot_pairB_s1",
     os.path.join(BASE, "pr12_4", "W2", "pairB_oneshot_s1"), "harm"),
    ("pairB_oneshot_s2", "stale-oneshot_pairB_s2",
     os.path.join(BASE, "pr12_4", "W2", "pairB_oneshot_s2"), "harm"),
    ("pairD_oneshot_s0", "stale-oneshot_pairD_s0",
     os.path.join(BASE, "pr12_3", "W2", "pairD_oneshot_s0"), "harm"),
    ("pairD_oneshot_s1", "stale-oneshot_pairD_s1",
     os.path.join(BASE, "pr12_4", "W2", "pairD_oneshot_s1"), "harm"),
    ("pairD_oneshot_s2", "stale-oneshot_pairD_s2",
     os.path.join(BASE, "pr12_4", "W2", "pairD_oneshot_s2"), "harm"),
    ("pairB_contra_s0", "contra_pairB_s0",
     os.path.join(BASE, "pr12_3", "W2", "pairB_contra_s0"), "harm"),
    ("pairB_contra_s1", "contra_pairB_s1",
     os.path.join(BASE, "pr12_4", "W2", "pairB_contra_s1"), "harm"),
    ("pairB_contra_s2", "contra_pairB_s2",
     os.path.join(BASE, "pr12_4", "W2", "pairB_contra_s2"), "harm"),
    ("pairD_contra_s0", "contra_pairD_s0",
     os.path.join(BASE, "pr12_3", "W2", "pairD_contra_s0"), "harm"),
    ("pairD_contra_s1", "contra_pairD_s1",
     os.path.join(BASE, "pr12_4", "W2", "pairD_contra_s1"), "harm"),
    ("pairD_contra_s2", "contra_pairD_s2",
     os.path.join(BASE, "pr12_4", "W2", "pairD_contra_s2"), "harm"),
]

FAM_QUERY_TEMPLATE = ("Which decode class does the memory support for "
                      "probe {probe} at epoch {epoch}? If the context marks "
                      "it unresolved or withdrawn, answer 'unresolved'.")

# --- Organic cell (memo §7 bullet 2; §4.2 registered fallback) --------------
# Scripted constants — every time below is a registered scalar, not a clock.
ORGANIC_BASE_T = 1767225600.0          # 2026-01-01T00:00:00Z
ORGANIC_REINGEST_OFFSET = 820800.0     # +9.5 days
ORGANIC_REPLAY_CLOCK = "2026-01-11T00:00:00Z"   # base + 10 days
ORGANIC_TTL_SECONDS = 604800           # 7 days: originals stale, re-ingests fresh
ORGANIC_REINGEST_STRIDE = 7            # every 7th path (by sorted order) re-ingests
ORGANIC_CORPUS_FLAG = "synthetic"      # certain, not merely likely (§4.2)

SD_QUERY_TEMPLATE = ("According to the ingest ledger, which ingest event "
                     "provides the current version of the item at path "
                     "{path}, and when was it ingested?")


def source_paths(original_db):
    """Read ONLY the path list from the original fixture, read-only."""
    conn = sqlite3.connect(f"file:{original_db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT DISTINCT filepath FROM ingested_images ORDER BY filepath"
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def build_organic_ledger(original_db, out_db):
    """Create the synthetic re-ingest ledger as a NEW sqlite file.

    Scripted events: path i (sorted order) ingested at BASE_T + 60*i,
    slot i; every STRIDE-th path re-ingested at BASE_T + 9.5d + 60*i,
    slot 1000+i. Returns the cell manifest (corpus flag, event lists,
    answer keys by construction).
    """
    if os.path.exists(out_db):
        raise FileExistsError(f"{out_db} exists; the ledger is built once")
    paths = source_paths(original_db)
    conn = sqlite3.connect(out_db)
    conn.execute("""CREATE TABLE ingested_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT, filepath TEXT NOT NULL,
        timestamp REAL NOT NULL, assigned_slot INTEGER,
        neighbor_1_path TEXT, neighbor_1_sim REAL, neighbor_2_path TEXT,
        neighbor_2_sim REAL, neighbor_3_path TEXT, neighbor_3_sim REAL)""")
    reingested, controls = [], []
    for i, path in enumerate(paths):
        conn.execute(
            "INSERT INTO ingested_images (filepath, timestamp, assigned_slot)"
            " VALUES (?,?,?)", (path, ORGANIC_BASE_T + 60.0 * i, i))
        if i % ORGANIC_REINGEST_STRIDE == 0:
            reingested.append(path)
        else:
            controls.append(path)
    for i, path in enumerate(paths):
        if i % ORGANIC_REINGEST_STRIDE == 0:
            conn.execute(
                "INSERT INTO ingested_images (filepath, timestamp, "
                "assigned_slot) VALUES (?,?,?)",
                (path, ORGANIC_BASE_T + ORGANIC_REINGEST_OFFSET + 60.0 * i,
                 1000 + i))
    conn.commit()
    conn.close()
    return {
        "corpus": ORGANIC_CORPUS_FLAG,
        "replay_clock": ORGANIC_REPLAY_CLOCK,
        "ttl_seconds": ORGANIC_TTL_SECONDS,
        "n_paths": len(paths),
        "reingested_paths": reingested,
        "control_paths": controls[:len(reingested)],
        # answer key by construction: latest event per path is the
        # re-ingest for reingested paths, the original for controls
    }


def organic_queries(manifest):
    """Q-SD1 instantiation: one query per re-ingested path (supersession
    present) and one per matched control path (none)."""
    queries = []
    for kind, paths in (("supersession", manifest["reingested_paths"]),
                        ("control", manifest["control_paths"])):
        for path in paths:
            queries.append({
                "query_id": f"organic:{kind}:{path}",
                "text": SD_QUERY_TEMPLATE.format(path=path),
                "kind": kind,
            })
    return queries


# --- Multi-turn withdrawal cells (memo §7 bullet 3) --------------------------
# Turn clocks (scripted): turn 1 before the re-ingest wave, turns 2-3
# after it. Visibility snapshots make the re-ingest "arrive" between
# turns for withdrawal sessions; control sessions share the clocks.
MULTITURN_CLOCKS = ("2026-01-03T00:00:00Z",   # day 2: originals fresh
                    "2026-01-10T14:00:00Z",   # day 9.58: re-ingests visible
                    "2026-01-10T16:00:00Z")   # day 9.67
MULTITURN_N_SESSIONS = 6                       # per arm (withdrawal/control)


def multiturn_sessions(manifest):
    sessions = []
    for kind, pool in (("withdrawal", manifest["reingested_paths"]),
                       ("control", manifest["control_paths"])):
        for path in pool[:MULTITURN_N_SESSIONS]:
            sessions.append({
                "session_id": f"mt:{kind}:{path}",
                "kind": kind,
                "target_path": path,
                "turn_clocks": list(MULTITURN_CLOCKS),
                "query_text": SD_QUERY_TEMPLATE.format(path=path),
            })
    return sessions


def build_ledger_snapshot(src_db, out_db, visible_until_epoch):
    """Per-turn visibility: copy rows with timestamp <= cutoff into a
    fresh snapshot db. The source ledger is opened read-only."""
    src = sqlite3.connect(f"file:{src_db}?mode=ro", uri=True)
    dst = sqlite3.connect(out_db)
    try:
        dst.execute("""CREATE TABLE ingested_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT, filepath TEXT NOT NULL,
            timestamp REAL NOT NULL, assigned_slot INTEGER,
            neighbor_1_path TEXT, neighbor_1_sim REAL, neighbor_2_path TEXT,
            neighbor_2_sim REAL, neighbor_3_path TEXT, neighbor_3_sim REAL)""")
        for rid, path, ts, slot in src.execute(
                "SELECT id, filepath, timestamp, assigned_slot FROM "
                "ingested_images WHERE timestamp <= ? ORDER BY id",
                (visible_until_epoch,)):
            dst.execute("INSERT INTO ingested_images (id, filepath, "
                        "timestamp, assigned_slot) VALUES (?,?,?,?)",
                        (rid, path, ts, slot))
        dst.commit()
    finally:
        src.close()
        dst.close()


def verify_registry():
    """Every registered FAM cell's run stem and packet surface must exist
    (existence check only — nothing is read into items here)."""
    missing = []
    for cell_id, stem_cell, packet_dir, _role in FAM_CELLS:
        for suffix in (".csv", ".topk.csv.gz", ".per_slot.csv"):
            p = os.path.join(RUNS_DIR, f"per_probe_{stem_cell}{suffix}")
            if not os.path.exists(p):
                missing.append(p)
        for name in ("memory_packet.jsonl", "audit_packet.jsonl"):
            p = os.path.join(packet_dir, name)
            if not os.path.exists(p):
                missing.append(p)
    return missing
