"""Hermetic tests for the PR-13 evidence adapters (memo §4) including
the poisoned-label canary — the NORMATIVE G-C2 mechanism.

SELECTION-TIMING KILL COMPLIANCE (memo §8.1): every fixture is
synthetic and built in tmp_path. No committed §7 material (no
pr10/governed run, no test_data/metadata.db row) is read, and nothing
is compiled for evaluation.
"""

import csv
import gzip
import json
import os
import sqlite3

import pytest

from harness.ctx.adapters import fam_v1, shutterdeck_v1
from harness.ctx.compile import load_policy


@pytest.fixture(scope="module")
def policy():
    return load_policy()


# --------------------------------------------------------------------------
# fam-v1 fixtures: a two-probe synthetic cell with a merge candidate, a
# stale-superseded slot, a contra fork, and a rank-0/1 tie.
# --------------------------------------------------------------------------

CELL = "synthetic_pairX_s9"

PER_PROBE_COLS = ["epoch", "probe_index", "top1_slot", "top1_top2_margin",
                  "top1_sim", "top2_sim", "n_surviving_votes",
                  "served_outcome", "abstain_reason",
                  # truth columns (must never influence output):
                  "true_label", "top1_correct", "vote_correct",
                  "failure_mode", "stale_strict", "contradictory_strict"]

def write_fam_fixture(root, poison=False):
    """Build a synthetic run-stem. With poison=True, every truth column
    is permuted/garbled; policy-visible columns are byte-identical."""
    t = (lambda v: {"CORRECT": "BROKEN", "0": "9", "1": "7"}.get(v, "POISON")) \
        if poison else (lambda v: v)
    os.makedirs(root, exist_ok=True)
    stem = os.path.join(str(root), f"per_probe_{CELL}")
    with open(stem + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(PER_PROBE_COLS)
        w.writerow(["0", "1", "5", "0.0005", "0.5001", "0.4996", "4",
                    "answer", "", t("1"), t("0"), t("0"), t("CORRECT"),
                    t("0"), t("0")])
        w.writerow(["0", "2", "7", "0.2100", "0.7100", "0.5000", "3",
                    "answer", "", t("0"), t("1"), t("1"), t("CORRECT"),
                    t("1"), t("0")])
    topk_rows = [
        # probe 1: tie between slot 5 (rank 0) and slot 6 (rank 1)
        ["0", "1", "0", "5", "0.5001", "1", "0.30", "3"],
        ["0", "1", "1", "6", "0.4996", "1", "0.29", "4"],
        ["0", "1", "2", "9", "0.3000", "0", "0.10", "3"],
        # probe 2: slot 7 leads; support includes stale slot 8
        ["0", "2", "0", "7", "0.7100", "1", "0.40", "2"],
        ["0", "2", "1", "8", "0.5000", "1", "0.20", "2"],
    ]
    with gzip.open(stem + ".topk.csv.gz", "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "epoch", "probe_index", "rank", "slot", "sim",
                    "surviving", "weight", "decode"])
        for row in topk_rows:
            w.writerow(["syn"] + row)
    with open(stem + ".per_slot.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "epoch", "slot", "decode", "hit_counts",
                    "last_write_seq", "usage", "n_records", "is_contra_fork",
                    "is_stale_superseded", "is_current_fork",
                    "is_merge_candidate", "role"])
        w.writerow(["syn", "0", "5", "3", "2", "10", "1.0", "2", "0", "0", "0", "1", t("clean")])
        w.writerow(["syn", "0", "6", "4", "1", "11", "1.0", "1", "1", "0", "0", "0", t("fork")])
        w.writerow(["syn", "0", "7", "2", "3", "12", "1.0", "3", "0", "0", "1", "0", t("clean")])
        w.writerow(["syn", "0", "8", "2", "1", "5", "0.5", "1", "0", "1", "0", "0", t("stale")])
    return root


def test_fam_v1_items_and_signals(tmp_path, policy):
    out = fam_v1.emit(str(write_fam_fixture(tmp_path)), CELL, policy)
    by_id = {i["item_id"]: i for i in out["items"]}
    assert len(out["items"]) == 4  # surviving ranks only
    sig = lambda i: {e["signal"] for e in by_id[i]["evidence"]}

    slot5 = f"fam-v1:{CELL}:e0:p1:slot5"
    slot6 = f"fam-v1:{CELL}:e0:p1:slot6"
    slot7 = f"fam-v1:{CELL}:e0:p2:slot7"
    slot8 = f"fam-v1:{CELL}:e0:p2:slot8"

    assert "merge_suspect" in sig(slot5) and "oneshot_tie" in sig(slot5)
    assert "contra_fork" in sig(slot6) and "oneshot_tie" in sig(slot6)
    assert by_id[slot5]["relations"]["candidate_set_id"] == \
           by_id[slot6]["relations"]["candidate_set_id"] is not None
    assert "stale_support" in sig(slot7)  # rank-0 with stale support
    assert "superseded_by" in sig(slot8)
    assert by_id[slot8]["state"] == "stale"
    # non-surviving rank audited as not-retrieved, never an item
    assert out["not_retrieved"] == [{"native_id": f"{CELL}:e0:p1:slot9",
                                     "reason": "not_surviving_engine"}]


def test_fam_v1_core_certified_only_merge_suspect(tmp_path, policy):
    out = fam_v1.emit(str(write_fam_fixture(tmp_path)), CELL, policy)
    for item in out["items"]:
        for ev in item["evidence"]:
            if ev["tier"] == "core-certified":
                assert ev["signal"] == "merge_suspect"


def test_fam_v1_content_is_registered_template(tmp_path, policy):
    out = fam_v1.emit(str(write_fam_fixture(tmp_path)), CELL, policy)
    item = next(i for i in out["items"] if i["item_id"].endswith("p2:slot7"))
    assert item["content"] == \
        "memory slot 7 decodes to class 2 (rank 0, sim 0.7100)"
    assert item["content_kind"] == "adapter-rendered"
    assert item["policy_version"].endswith("+ctx-fam-item-v1")


def test_fam_v1_poisoned_label_canary(tmp_path, policy):
    """G-C2 normative mechanism: permuting every truth column leaves the
    adapter output byte-identical."""
    clean = fam_v1.emit(str(write_fam_fixture(tmp_path / "clean")), CELL, policy)
    poisoned = fam_v1.emit(
        str(write_fam_fixture(tmp_path / "poison", poison=True)), CELL, policy)
    assert json.dumps(clean, sort_keys=True) == \
           json.dumps(poisoned, sort_keys=True)


def test_fam_v1_deterministic(tmp_path, policy):
    root = str(write_fam_fixture(tmp_path))
    a = fam_v1.emit(root, CELL, policy)
    b = fam_v1.emit(root, CELL, policy)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_fam_v1_witness_alt_is_unwired(policy):
    with pytest.raises(NotImplementedError):
        fam_v1.emit_witness_alt_from_packets("anywhere", policy)


# --------------------------------------------------------------------------
# shutterdeck-v1 fixtures: synthetic ledger with a re-ingested path, a
# content duplicate, and a missing file.
# --------------------------------------------------------------------------

def write_sd_fixture(root):
    files = root / "store"
    files.mkdir(parents=True)
    (files / "a.arw").write_bytes(b"AAA")
    (files / "b.arw").write_bytes(b"AAA")   # duplicate content of a.arw
    (files / "c.arw").write_bytes(b"CCC")
    # drop/d.arw deliberately missing from the store
    db = root / "metadata.db"
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE ingested_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT, filepath TEXT NOT NULL,
        timestamp REAL NOT NULL, assigned_slot INTEGER,
        neighbor_1_path TEXT, neighbor_1_sim REAL, neighbor_2_path TEXT,
        neighbor_2_sim REAL, neighbor_3_path TEXT, neighbor_3_sim REAL)""")
    rows = [
        ("drop/a.arw", 1000.0, 1),
        ("drop/b.arw", 1010.0, 2),
        ("drop/c.arw", 1020.0, 3),
        ("drop/c.arw", 5000.0, 4),   # re-ingest supersedes id 3
        ("drop/d.arw", 6000.0, 5),   # file missing from store
    ]
    for path, ts, slot in rows:
        conn.execute("INSERT INTO ingested_images (filepath, timestamp, "
                     "assigned_slot) VALUES (?,?,?)", (path, ts, slot))
    conn.commit()
    conn.close()
    return str(db), str(files)


REPLAY = "1970-01-01T02:00:00Z"   # epoch 7200 as the scripted replay clock


def test_shutterdeck_signals(tmp_path, policy):
    db, files = write_sd_fixture(tmp_path)
    out = shutterdeck_v1.emit(db, files, REPLAY, ttl_seconds=3000, policy=policy)
    by_id = {i["item_id"]: i for i in out["items"]}
    sig = lambda i: {e["signal"] for e in by_id[i]["evidence"]}

    assert len(out["items"]) == 5 and out["not_retrieved"] == []
    # superseded path: id 3 superseded by id 4; id 4 supersedes id 3
    assert "superseded_by_path" in sig("shutterdeck-v1:row3")
    assert by_id["shutterdeck-v1:row3"]["state"] == "superseded"
    assert by_id["shutterdeck-v1:row4"]["relations"]["supersedes"] == \
           ["shutterdeck-v1:row3"]
    # freshness vs stale_ttl at the scripted replay clock (7200):
    # ids 1-3 are older than TTL 3000; ids 4-5 within it
    assert "stale_ttl" in sig("shutterdeck-v1:row1")
    assert "freshness" in sig("shutterdeck-v1:row4")
    # duplicate content across a/b; missing file d → unavailable, not false
    assert "duplicate_content" in sig("shutterdeck-v1:row1")
    assert "duplicate_content" in sig("shutterdeck-v1:row2")
    assert "duplicate_content" not in sig("shutterdeck-v1:row5")
    assert "signal_unavailable" in sig("shutterdeck-v1:row5")


def test_shutterdeck_content_is_registered_template(tmp_path, policy):
    db, files = write_sd_fixture(tmp_path)
    out = shutterdeck_v1.emit(db, files, REPLAY, 3000, policy)
    item = next(i for i in out["items"] if i["item_id"] == "shutterdeck-v1:row1")
    assert item["content"] == ("image at drop/a.arw, ingested "
                               "1970-01-01T00:16:40.000000Z, slot 1")
    assert item["content_kind"] == "adapter-rendered"


def test_shutterdeck_deterministic_and_clock_injected(tmp_path, policy):
    db, files = write_sd_fixture(tmp_path)
    a = shutterdeck_v1.emit(db, files, REPLAY, 3000, policy)
    b = shutterdeck_v1.emit(db, files, REPLAY, 3000, policy)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    # a different scripted clock changes freshness — the clock is an
    # input, never the wall clock
    later = shutterdeck_v1.emit(db, files, "1970-01-01T05:00:00Z", 3000, policy)
    sig5 = {e["signal"] for i in later["items"]
            if i["item_id"] == "shutterdeck-v1:row5" for e in i["evidence"]}
    assert "stale_ttl" in sig5


def test_shutterdeck_pin_corpus(tmp_path):
    db, files = write_sd_fixture(tmp_path)
    pin = shutterdeck_v1.pin_corpus(db, files)
    assert pin["n_rows"] == 5
    assert pin["files"]["drop/a.arw"] == pin["files"]["drop/b.arw"]
    assert pin["files"]["drop/d.arw"] is None
    assert len(pin["metadata_db_sha256"]) == 64


# --------------------------------------------------------------------------
# adapter output feeds the compiler end-to-end (synthetic only)
# --------------------------------------------------------------------------

def test_adapter_items_compile_end_to_end(tmp_path, policy):
    from harness.ctx.compile import compile as ctx_compile
    db, files = write_sd_fixture(tmp_path)
    out = shutterdeck_v1.emit(db, files, REPLAY, 3000, policy)
    block, packet = ctx_compile(out["items"], policy, 220, None,
                                lambda s: len(s.split()))
    assert block.startswith("[governed context")
    assert len(packet["rows"]) == 5
    assert packet["anomalies"] == []
