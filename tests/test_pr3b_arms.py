"""tests/test_pr3b_arms.py — hermetic tests for the PR-3b arm variants and
side tables (mixed / one-shot / key-jitter / soft-payload, per_slot.csv,
fork_events.csv).

The gentoo PR-3b runs are only trustworthy if, BEFORE any cache run:

  1. the mixed arm runs both protocols in one memory with labels coexisting
     and contra injections excluded from the superseded group's rows;
  2. one-shot supersession leaves the boundary tie regime in place for every
     post-boundary epoch (the PR-2c prediction, now pinned);
  3. soft payloads take the EMA-merge path (absorbed, no phase-2 fork, slot
     keeps decoding the pre-update label -> STALE via merge);
  4. key jitter actually moves the phase-2 pre-write similarity off the
     exact-tie point (the artifact-control knob does something);
  5. per_slot.csv roles are consistent with the per-probe strict labels;
  6. fork_events.csv carries ground-truth event classes per protocol and NO
     registry-derived failure labels (feature/label separation, by schema);
  7. new-variant runs are byte-identical for a fixed seed (reproducible
     gentoo runs), side tables included.

All on the tiny synthetic cache FILE from test_failure_mode_vision (vision
wiring: identical batch re-written every epoch — the regime where merge vs
fork is decided by payload cosine, exactly like the real cache runs).
CPU-only, no GPU, no network.
"""
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.failure_mode_probe import (  # noqa: E402
    EVENT_COLS, EVENT_CONTRADICTION, EVENT_DUPLICATE, EVENT_INITIAL,
    EVENT_ONE_SHOT, EVENT_SUPERSESSION, FailureModeRegistry, SLOT_COLS,
    run_vision)
from tests.test_failure_mode_vision import (  # noqa: E402
    ATTRACTOR, CLASSES, _make_cache)

from associative_core import ContinuousCAM  # noqa: E402


def _run(arm, tmp_path, *, rate=0.5, epochs=6, seed=0, name="out.csv",
         supersede_epoch=3, **kw):
    cache = _make_cache(tmp_path)
    out = tmp_path / name
    n, summary = run_vision(
        arm, rate=rate, epochs=epochs, out_path=out, cache_path=str(cache),
        classes=CLASSES, attractor_class=ATTRACTOR, samples_per_class=8,
        held_out_per_class=8, contraction=0.0, seed=seed,
        supersede_epoch=supersede_epoch, **kw)
    return n, summary, out


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _tables(out):
    return (_read(out), _read(out.with_suffix(".per_slot.csv")),
            _read(out.with_suffix(".fork_events.csv")))


# ---------------------------------------------------------------------------
# mixed arm
# ---------------------------------------------------------------------------
def test_mixed_arm_runs_both_protocols_with_labels_coexisting(tmp_path):
    n, summary, out = _run("mixed", tmp_path, rate=0.5)
    rows, slots, events = _tables(out)
    assert n == len(rows) > 0
    assert {r["arm"] for r in rows} == {"mixed"}

    classes = {e["event_class"] for e in events}
    assert EVENT_CONTRADICTION in classes and EVENT_SUPERSESSION in classes
    # forks of both kinds materialized in the same memory
    roles = {s["role"] for s in slots}
    assert "contra_fork" in roles and "stale_superseded" in roles

    # both label families fire somewhere in the run (high injection rate +
    # boundary tie make this deterministic on the tiny manifold), and the
    # overlap is observable rather than folded.
    assert any(r["contradictory_lenient"] == "1" for r in rows)
    assert any(r["stale_lenient"] == "1" for r in rows)
    assert summary["fork_resolution"] in (
        "later-dominates", "persistent-split", "persistent-tie",
        "old-persists", "mixed")


def test_mixed_arm_excludes_superseded_rows_from_injection():
    """Unit pin of the eligible mask: with rate 1.0 the selection clamps to
    the eligible rows, and an all-False mask injects nothing."""
    dim, C = 16, 4
    mem = ContinuousCAM(key_dim=dim, value_dim=C, max_entries=64,
                        vigilance=0.8, track_provenance=True)
    reg = FailureModeRegistry()
    q = F.normalize(torch.eye(dim)[:8], dim=-1)
    lab = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    mem.learn_local(q, F.one_hot(lab, C).float(),
                    record_ids=reg.next_ids("clean", 8))

    rng = torch.Generator().manual_seed(0)
    ids = reg.inject_contradictions(mem, q, lab, C, rate=1.0, rng=rng,
                                    eligible=lab != 0)
    assert len(ids) == 6  # clamped to the 6 non-pre-label rows
    rng = torch.Generator().manual_seed(0)
    ids = reg.inject_contradictions(mem, q, lab, C, rate=1.0, rng=rng,
                                    eligible=torch.zeros(8, dtype=torch.bool))
    assert ids == []


# ---------------------------------------------------------------------------
# one-shot supersession
# ---------------------------------------------------------------------------
def test_one_shot_tie_regime_persists_to_end_of_run(tmp_path):
    n, summary, out = _run("stale", tmp_path, epochs=7, supersede_epoch=3,
                           one_shot=True)
    post = [e for e in summary["per_epoch"] if e["superseded"]]
    assert len(post) == 4
    assert summary["n_phase2_writes"] == post[0]["superseded_key_probes"] == 8
    for e in post:
        # exactly one A slot + one B fork per key at identical positions:
        # the split stays at 0.5/0.5 in EVERY post-boundary epoch and the
        # stale side keeps being elected by tie-break.
        assert abs(e["stale_weight_median_superseded"] - 0.5) < 1e-4, e
        assert e["stale_selected"] == e["superseded_key_probes"], e
        assert e["updated_selected"] == 0, e
        assert e["live_stale_slots"] == 8
    assert summary["fork_resolution"] == "persistent-tie"
    # exactly one phase-2 event per key, ground-truth class one-shot-ambiguous
    events = _read(out.with_suffix(".fork_events.csv"))
    sup = [e for e in events if e["event_class"] == EVENT_ONE_SHOT]
    assert len(sup) == 8
    assert all(e["outcome"] == "forked" for e in sup)
    assert not any(e["event_class"] == EVENT_SUPERSESSION for e in events)


# ---------------------------------------------------------------------------
# soft payloads (EMA-merge stale path)
# ---------------------------------------------------------------------------
def test_soft_payloads_take_merge_path_and_keep_decoding_stale(tmp_path):
    n, summary, out = _run("stale", tmp_path, epochs=5, supersede_epoch=3,
                           payload_mode="soft")
    rows, slots, events = _tables(out)

    sup = [e for e in events if e["event_class"] == EVENT_SUPERSESSION]
    assert sup and all(e["outcome"] == "absorbed" for e in sup), \
        "soft phase-2 writes must EMA-merge, never fork"
    # merge-path stale: the SAME slot carries phase-1 and phase-2 provenance
    flip = summary["supersede_epoch"]
    flip_slots = [s for s in slots if int(s["epoch"]) == flip]
    merge = [s for s in flip_slots if s["is_merge_candidate"] == "1"]
    assert len(merge) == 8
    # at the flip epoch the merged slots still decode to A -> stale role
    assert all(s["is_stale_superseded"] == "1" for s in merge)
    assert all(s["role"] == "stale_superseded" for s in merge)
    # no co-resident fork was allocated for the superseded keys at the flip:
    # phase-2 created zero current_fork slots there
    assert not any(s["is_current_fork"] == "1" for s in flip_slots)
    # and the failure surfaces as STALE on the superseded keys at the flip
    flip_rows = [r for r in rows if int(float(r["epoch"])) == flip]
    assert any(r["stale_strict"] == "1" for r in flip_rows)
    assert not any(r["contradictory_lenient"] == "1" for r in rows)


# ---------------------------------------------------------------------------
# key jitter
# ---------------------------------------------------------------------------
def test_key_jitter_moves_phase2_off_the_exact_tie(tmp_path):
    _, _, out0 = _run("stale", tmp_path, epochs=5, supersede_epoch=3,
                      name="eps0.csv", key_jitter=0.0)
    _, _, outj = _run("stale", tmp_path, epochs=5, supersede_epoch=3,
                      name="epsj.csv", key_jitter=0.2)

    def phase2_sims(out):
        events = _read(out.with_suffix(".fork_events.csv"))
        return [float(e["pre_sim"]) for e in events
                if e["event_class"] == EVENT_SUPERSESSION]

    s0, sj = phase2_sims(out0), phase2_sims(outj)
    assert s0 and min(s0) > 0.9999, "eps=0 re-writes the exact key"
    assert sj and max(sj) < 0.9999, "jitter must move the phase-2 key"
    assert min(sj) > 0.9, "eps=0.2 stays a near-duplicate, not a new key"


# ---------------------------------------------------------------------------
# side-table integrity
# ---------------------------------------------------------------------------
def test_per_slot_roles_consistent_with_strict_labels(tmp_path):
    _, _, out = _run("mixed", tmp_path, rate=0.5)
    rows, slots, _ = _tables(out)
    by_epoch = {}
    for s in slots:
        by_epoch.setdefault(int(s["epoch"]), {})[int(s["slot"])] = s
    checked = 0
    for r in rows:
        epoch, top1 = int(float(r["epoch"])), int(r["top1_slot"])
        if r["contradictory_strict"] == "1":
            assert by_epoch[epoch][top1]["is_contra_fork"] == "1"
            checked += 1
        if r["stale_strict"] == "1":
            assert by_epoch[epoch][top1]["is_stale_superseded"] == "1"
            checked += 1
    assert checked > 0, "no strict rows — the consistency check ran on nothing"


def test_fork_events_schema_and_class_domains(tmp_path):
    # schema: write-time observables + ground truth event class only — none
    # of the per-probe failure-label columns may leak in.
    assert not (set(EVENT_COLS)
                & {"failure_mode", "contradictory_strict",
                   "contradictory_lenient", "stale_strict", "stale_lenient",
                   "vote_correct"})
    _, _, out = _run("clean", tmp_path, rate=0.0, name="clean.csv")
    events = _read(out.with_suffix(".fork_events.csv"))
    assert {e["event_class"] for e in events} == \
        {EVENT_INITIAL, EVENT_DUPLICATE}, \
        "clean vision arm may emit only non-conflict event classes"
    # vision epoch>0 clean re-writes are exact duplicates
    assert all(e["event_class"] == EVENT_INITIAL for e in events
               if int(e["epoch"]) == 0)
    seqs = [(e["record_tag"], e["record_seq"]) for e in events]
    assert len(seqs) == len(set(seqs)), "record ids must be unique"


def test_new_variant_runs_are_deterministic(tmp_path):
    for arm, kw in (("mixed", {}), ("stale", {"one_shot": True}),
                    ("stale", {"payload_mode": "soft"}),
                    ("stale", {"key_jitter": 0.1})):
        outs = []
        for i in range(2):
            tag = f"{arm}-{'-'.join(kw)}-{i}"
            _, _, out = _run(arm, tmp_path, name=f"{tag}.csv", **kw)
            outs.append(out)
        a, b = outs
        assert a.read_bytes() == b.read_bytes()
        for suffix in (".per_slot.csv", ".fork_events.csv"):
            assert a.with_suffix(suffix).read_bytes() == \
                b.with_suffix(suffix).read_bytes()
