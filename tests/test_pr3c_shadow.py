"""tests/test_pr3c_shadow.py — hermetic gates for the PR-3c shadow basis.

Shadow governance (PR3_DESIGN.md §5 PR-3c) recomputes counterfactual
readouts OFFLINE from the logged top-k candidate composition; the deployed
``forward()`` is never altered. That is only meaningful if the logged
table reconstructs the deployed vote bit-exactly (§10 shadow-readout
fidelity) — one-shot rows sit at exact 0.5/0.5 ties, where anything short
of bit fidelity silently flips elections. Before any cache run:

  1. topk.csv re-aggregates to the deployed ``vote_pred_label`` on every
     probe row (policy-`none` == deployment), including the exact-tie
     one-shot arm and the mixed arm;
  2. the table is structurally sound: contiguous ranks, rank-ordered raw
     sims, non-surviving candidates carry exactly zero weight, surviving
     weights sum to 1;
  3. the one-shot tie is visible in the composition the policies read
     (equal surviving vote mass on the two disagreeing decode classes);
  4. no per-probe failure label leaks into the schema (policies must be
     label-free; registry truth is for scoring only).

All on the tiny synthetic cache FILE from test_failure_mode_vision.
CPU-only, no GPU, no network.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.failure_mode_probe import (  # noqa: E402
    TOPK_COLS, run_vision, vote_pred_from_candidates)
from tests.test_failure_mode_vision import (  # noqa: E402
    ATTRACTOR, CLASSES, _make_cache)


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


def _topk_by_probe(out):
    by_probe = defaultdict(list)
    for r in _read(out.with_suffix(".topk.csv")):
        by_probe[(int(r["epoch"]), int(r["probe_index"]))].append(r)
    for cands in by_probe.values():
        cands.sort(key=lambda r: int(r["rank"]))
    return by_probe


def test_topk_schema_has_no_label_leak():
    assert not (set(TOPK_COLS)
                & {"failure_mode", "contradictory_strict",
                   "contradictory_lenient", "stale_strict", "stale_lenient",
                   "vote_correct", "true_label", "event_class",
                   "injected_label"})


def test_topk_reaggregates_to_deployed_vote(tmp_path):
    """Policy-`none` fidelity: the vote recomputed from the written CSV text
    equals the deployed vote on EVERY probe row, in the mixed arm and in the
    exact-tie one-shot arm (the case where precision loss would flip it)."""
    for arm, name, kw in (("mixed", "mixed.csv", {}),
                          ("stale", "oneshot.csv", {"one_shot": True})):
        _, _, out = _run(arm, tmp_path, name=name, **kw)
        rows = _read(out)
        by_probe = _topk_by_probe(out)
        assert len(by_probe) == len(rows) > 0
        num_classes = max(int(r["decode"]) for c in by_probe.values()
                          for r in c) + 1
        for r in rows:
            cands = by_probe[(int(float(r["epoch"])), int(r["probe_index"]))]
            shadow = vote_pred_from_candidates(
                [c["weight"] for c in cands],
                [c["decode"] for c in cands], num_classes)
            assert shadow == int(float(r["vote_pred_label"])), r


def test_topk_structure(tmp_path):
    _, _, out = _run("mixed", tmp_path)
    by_probe = _topk_by_probe(out)
    for cands in by_probe.values():
        assert [int(c["rank"]) for c in cands] == list(range(len(cands)))
        sims = [float(c["sim"]) for c in cands]
        assert sims == sorted(sims, reverse=True)
        dead = [float(c["weight"]) for c in cands if c["surviving"] == "0"]
        live = [float(c["weight"]) for c in cands if c["surviving"] == "1"]
        assert all(w == 0.0 for w in dead)
        assert live and abs(sum(live) - 1.0) < 1e-5
        # rank-0 is the raw top-1 the per-probe telemetry quotes
        assert int(cands[0]["rank"]) == 0


def test_one_shot_tie_is_visible_in_topk_composition(tmp_path):
    """The fork-witness signal the policies read: on every post-boundary
    superseded probe, the surviving vote mass splits equally between the
    pre- and post-update decode classes. Only the tie is pinned — the
    elected side is platform-dependent."""
    _, summary, out = _run("stale", tmp_path, epochs=7, supersede_epoch=3,
                           one_shot=True)
    flip = summary["supersede_epoch"]
    pre, post = summary["stale_pre_label"], summary["stale_post_label"]
    by_probe = _topk_by_probe(out)
    tied = 0
    for (epoch, _), cands in by_probe.items():
        if epoch < flip:
            continue
        mass = defaultdict(float)
        for c in cands:
            if c["surviving"] == "1":
                mass[int(c["decode"])] += float(c["weight"])
        if (mass[pre] > 0.3 and mass[post] > 0.3
                and abs(mass[pre] - mass[post]) < 1e-3):
            tied += 1
    post_epochs = summary["epochs"] - flip
    assert tied == 8 * post_epochs, \
        f"expected every post-boundary superseded probe tied, got {tied}"
