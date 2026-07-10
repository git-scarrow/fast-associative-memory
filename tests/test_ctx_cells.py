"""Tests for PR-13 §7 cell instantiation (harness/ctx/cells.py).

The registry checks read committed file PATHS (existence only); the
ledger/session tests run on synthetic tmp fixtures. Nothing here
renders a compiled block over §7 material — the end-to-end withdrawal
test uses a throwaway synthetic ledger, keeping the §8.1
selection-timing kill untripped.
"""

import hashlib
import json
import os
import sqlite3

import pytest

from harness.ctx import cells
from harness.ctx.adapters import shutterdeck_v1
from harness.ctx.compile import compile as ctx_compile, load_policy


def wtok(text):
    return len(text.split())


@pytest.fixture(scope="module")
def policy():
    return load_policy()


def make_source_db(path, n=9):
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE ingested_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT, filepath TEXT NOT NULL,
        timestamp REAL NOT NULL, assigned_slot INTEGER,
        neighbor_1_path TEXT, neighbor_1_sim REAL, neighbor_2_path TEXT,
        neighbor_2_sim REAL, neighbor_3_path TEXT, neighbor_3_sim REAL)""")
    for i in range(n):
        conn.execute("INSERT INTO ingested_images (filepath, timestamp, "
                     "assigned_slot) VALUES (?,?,?)",
                     (f"drop/img{i:02d}.arw", 1.0 + i, i))
    conn.commit()
    conn.close()
    return str(path)


def test_registry_complete_on_committed_artifacts():
    assert cells.verify_registry() == []
    assert len(cells.FAM_CELLS) == 13
    roles = [role for *_, role in cells.FAM_CELLS]
    assert roles.count("clean-control") == 1 and roles.count("harm") == 12


def test_build_organic_ledger_deterministic_and_source_untouched(tmp_path):
    src = make_source_db(tmp_path / "src.db")
    before = hashlib.sha256(open(src, "rb").read()).hexdigest()
    m1 = cells.build_organic_ledger(src, str(tmp_path / "led1.db"))
    m2 = cells.build_organic_ledger(src, str(tmp_path / "led2.db"))
    assert before == hashlib.sha256(open(src, "rb").read()).hexdigest()
    assert m1 == m2
    rows = lambda p: sqlite3.connect(p).execute(
        "SELECT filepath, timestamp, assigned_slot FROM ingested_images "
        "ORDER BY id").fetchall()
    assert rows(str(tmp_path / "led1.db")) == rows(str(tmp_path / "led2.db"))
    # stride 7 over 9 paths → img00 and img07 re-ingested
    assert m1["reingested_paths"] == ["drop/img00.arw", "drop/img07.arw"]
    assert len(m1["control_paths"]) == 2
    with pytest.raises(FileExistsError):
        cells.build_organic_ledger(src, str(tmp_path / "led1.db"))


def test_organic_queries_and_sessions_shape(tmp_path):
    src = make_source_db(tmp_path / "src.db")
    m = cells.build_organic_ledger(src, str(tmp_path / "led.db"))
    queries = cells.organic_queries(m)
    assert len(queries) == 4
    kinds = [q["kind"] for q in queries]
    assert kinds.count("supersession") == kinds.count("control") == 2
    sessions = cells.multiturn_sessions(m)
    assert {s["kind"] for s in sessions} == {"withdrawal", "control"}
    assert all(len(s["turn_clocks"]) == 3 for s in sessions)


def test_snapshot_visibility_cutoff(tmp_path):
    src = make_source_db(tmp_path / "src.db")
    cells.build_organic_ledger(src, str(tmp_path / "led.db"))
    # cutoff before the re-ingest wave: only the 9 originals visible
    cells.build_ledger_snapshot(str(tmp_path / "led.db"),
                                str(tmp_path / "snap1.db"),
                                cells.ORGANIC_BASE_T + 3600)
    n1 = sqlite3.connect(str(tmp_path / "snap1.db")).execute(
        "SELECT COUNT(*) FROM ingested_images").fetchone()[0]
    assert n1 == 9
    # cutoff after it: originals + 2 re-ingests
    cells.build_ledger_snapshot(str(tmp_path / "led.db"),
                                str(tmp_path / "snap2.db"),
                                cells.ORGANIC_BASE_T + cells.ORGANIC_REINGEST_OFFSET + 9999)
    n2 = sqlite3.connect(str(tmp_path / "snap2.db")).execute(
        "SELECT COUNT(*) FROM ingested_images").fetchone()[0]
    assert n2 == 11


def test_multiturn_withdrawal_end_to_end_synthetic(tmp_path, policy):
    """Turn 1 renders the original item; the re-ingest becomes visible
    before turn 3; the compiled turn-3 block carries the registered
    WITHDRAWN notice for the turn-1 item."""
    src = make_source_db(tmp_path / "src.db")
    cells.build_organic_ledger(src, str(tmp_path / "led.db"))
    target = "drop/img00.arw"   # re-ingested path
    clocks = cells.MULTITURN_CLOCKS

    def turn_items(cutoff_epoch, clock):
        snap = tmp_path / f"snap_{cutoff_epoch}.db"
        cells.build_ledger_snapshot(str(tmp_path / "led.db"), str(snap),
                                    cutoff_epoch)
        out = shutterdeck_v1.emit(str(snap), str(tmp_path / "nostore"),
                                  clock, cells.ORGANIC_TTL_SECONDS, policy)
        return [i for i in out["items"]
                if target in i["content"]]

    from harness.ctx.adapters.shutterdeck_v1 import _parse_rfc3339
    items1 = turn_items(_parse_rfc3339(clocks[0]), clocks[0])
    assert len(items1) == 1
    block1, packet1 = ctx_compile(items1, policy, 200,
                                  {"turn_index": 1, "prior_rendered": {}},
                                  wtok)
    rendered1 = [r["item_id"] for r in packet1["rows"]
                 if r["budget_decision"] == "rendered"]
    assert rendered1 == [items1[0]["item_id"]]

    items3 = turn_items(_parse_rfc3339(clocks[2]), clocks[2])
    assert len(items3) == 2   # original + re-ingest now visible
    turn_state = {"turn_index": 3,
                  "prior_rendered": {rendered1[0]: 1}}
    block3, packet3 = ctx_compile(items3, policy, 200, turn_state, wtok)
    assert f"WITHDRAWN: the item served at turn 1 ({rendered1[0]})" in block3
    row = next(r for r in packet3["rows"] if r["item_id"] == rendered1[0])
    assert row["disposition"] == "withdraw"


def test_synthetic_reingest_is_the_registered_corpus_location():
    """The directory is named for what it holds (scripted events), the
    original fixture is where the constants say, and the committed ledger
    matches the sha the manifest pins it to."""
    assert os.path.isdir(cells.SYNTHETIC_REINGEST_DIR)
    assert not os.path.exists(os.path.join(cells.CELLS_DATA, "organic"))
    assert os.path.exists(cells.ORIGINAL_FIXTURE_DB)
    with open(cells.SYNTHETIC_LEDGER, "rb") as fh:
        got = hashlib.sha256(fh.read()).hexdigest()
    assert got == json.load(open(cells.SYNTHETIC_MANIFEST))["ledger_sha256"]


def test_committed_organic_artifacts_match_builder():
    """The committed ledger/manifest must be exactly what the builder
    produces from the original fixture (row-level equality)."""
    committed = cells.SYNTHETIC_LEDGER
    manifest = json.load(open(cells.SYNTHETIC_MANIFEST))
    assert manifest["corpus"] == "synthetic"
    assert manifest["n_paths"] == 69
    assert len(manifest["reingested_paths"]) == 10
    conn = sqlite3.connect(f"file:{committed}?mode=ro", uri=True)
    n = conn.execute("SELECT COUNT(*) FROM ingested_images").fetchone()[0]
    conn.close()
    assert n == 69 + 10
