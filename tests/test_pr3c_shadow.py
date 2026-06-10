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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.analyze_fork_governance import (  # noqa: E402
    POLICY_VISIBLE_EVENT, POLICY_VISIBLE_SLOT, POLICY_VISIBLE_TOPK,
    READ_TIME_FORK_ONLY, build_writetime_router, load_run, router_state,
    score_run)
from benchmarks.failure_mode_probe import (  # noqa: E402
    TOPK_COLS, run_vision, vote_pred_from_candidates)
from tests.test_failure_mode_vision import (  # noqa: E402
    ATTRACTOR, CLASSES, _make_cache)


class _RetainAll:
    """Frozen-detector stub: retain every probe (entropy-abstain == none)."""

    def score(self, cols, mask):
        return np.ones(int(np.asarray(mask).sum()))


def _score(arm, tmp_path, name, **kw):
    _, summary, out = _run(arm, tmp_path, name=name, **kw)
    run = load_run(out)
    return run, score_run(run, _RetainAll(), 0.0), summary


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


# ---------------------------------------------------------------------------
# shadow governance (offline, over the side tables)
# ---------------------------------------------------------------------------
def test_label_free_allowlists():
    labelish = {"failure_mode", "contradictory_strict",
                "contradictory_lenient", "stale_strict", "stale_lenient",
                "vote_correct", "true_label", "event_class",
                "injected_label", "record_tag", "role", "is_contra_fork",
                "is_stale_superseded", "is_current_fork",
                "is_merge_candidate"}
    for allow in (POLICY_VISIBLE_TOPK, POLICY_VISIBLE_SLOT,
                  POLICY_VISIBLE_EVENT):
        assert not (set(allow) & labelish)


def test_none_is_deployed_and_observe_only_matches(tmp_path):
    """score_run raises on any shadow-fidelity violation; this pins that it
    does NOT raise on real runs, and that observe-only changes nothing."""
    for arm, name, kw in (("mixed", "m.csv", {}),
                          ("stale", "o.csv", {"one_shot": True})):
        _, table, _ = _score(arm, tmp_path, name, **kw)
        n, obs = table["none"], table["observe-only"]
        assert obs == {**n}, "observe-only must be metric-identical to none"
        assert n["acted"] == n["abstained"] == n["fixed"] == 0
        # the stub detector retains everything: entropy-abstain == none too
        assert table["entropy-abstain"]["abstained"] == 0


def test_recency_resolves_oneshot_tie_to_newest_side(tmp_path):
    """Platform-safe pin: last_write_seq breaks the 0.5/0.5 tie toward the
    post-update side on every superseded probe, whatever side the deployed
    argmax elected — so it fixes ALL stale-wrong rows and breaks nothing."""
    _, table, _ = _score("stale", tmp_path, "one.csv", epochs=7,
                         supersede_epoch=3, one_shot=True)
    m = table["recency-naive"]
    assert m["stale_wrong"] > 0
    assert m["stale_wrong_fixed"] == m["stale_wrong"]
    assert m["broken"] == 0


def test_merge_path_is_invisible_to_read_time_fork_policies(tmp_path):
    """The required benchmark: merge-path stale leaves NO read-time fork
    topology, so every fork-topology policy fixes/abstains exactly zero of
    its errors, while the write-time merge-suspect route abstains on all of
    them without breaking a correct probe."""
    run, table, _ = _score("stale", tmp_path, "soft.csv", epochs=5,
                           supersede_epoch=3, payload_mode="soft")
    wrong = table["none"]["wrong_none"]
    assert wrong > 0 and table["none"]["stale_wrong"] == wrong
    for policy in READ_TIME_FORK_ONLY:
        m = table[policy]
        assert m["acted"] == 0, (policy, m["acted"])
        assert m["stale_wrong_fixed"] == m["stale_wrong_abstained"] == 0
    for policy in ("mode-conditioned-observe", "mode-conditioned-abstain",
                   "mode-conditioned-trust"):
        m = table[policy]
        assert m["stale_wrong_abstained"] == wrong
        assert m["broken"] == 0
    assert table["_router"]["n_merge_suspect_events"] > 0
    assert table["_router"]["n_conflict_pairs"] == 0


def test_router_routes_oneshot_to_ambiguous_and_repeated_to_supersession(
        tmp_path):
    _, _, out_one = _run("stale", tmp_path, name="r1.csv", epochs=7,
                         supersede_epoch=3, one_shot=True)
    one = load_run(out_one)
    router = build_writetime_router(one["events"], one["slot_obs"])
    last = max(e["epoch"] for e in one["events"])
    st = router_state(router, last)
    assert len(st["ambiguous"]) == 16  # 8 keys x both fork sides
    assert not st["quarantine"] and not st["deprecate"] and not st["merge"]

    _, _, out_rep = _run("stale", tmp_path, name="r2.csv", epochs=7,
                         supersede_epoch=3)
    rep = load_run(out_rep)
    router = build_writetime_router(rep["events"], rep["slot_obs"])
    st = router_state(router, max(e["epoch"] for e in rep["events"]))
    assert len(st["deprecate"]) == 8  # repeated K->B writes: deprecate old A
    assert not st["quarantine"]


def test_governance_scoring_is_deterministic(tmp_path):
    _, t1, _ = _score("mixed", tmp_path, "d1.csv")
    _, t2, _ = _score("mixed", tmp_path, "d2.csv")
    assert t1 == t2
