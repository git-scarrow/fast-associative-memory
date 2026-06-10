"""tests/test_failure_mode_probe.py — hermetic mechanism tests for PR-2a.

The failure-mode-labeled injection driver (benchmarks/failure_mode_probe.py)
is only trustworthy if its labels are MECHANICALLY valid: a CONTRADICTORY flag
must trace back to a real demotion fork (pre-write sim >= vigilance, payload
cosine <= 0.5, co-resident slot), and a STALE flag must trace back to a real
supersession (K→A matured, K→B written through the normal learn path, K still
retrieving A). These tests construct each mechanism on tiny synthetic
manifolds (CPU, no caches, no GPU, no network) and pin:

  1. fork vs plain-miss vs absorbed outcome classification;
  2. co-residency of the fork (key geometry + payload disagreement);
  3. both stale paths — EMA-freeze merge (payload cosine > 0.5) and
     supersession fork (payload cosine <= 0.5);
  4. STALE/CONTRADICTORY strict/lenient label semantics, strict => lenient;
  5. vote_pred_label in the labeled output is pinned to forward().argmax;
  6. the clean arm is a true negative control (no injection labels fire);
  7. provenance is required, never optional;
  8. the synthetic CLI harness end-to-end (schema + label domain).

No detector is fit, no GPU result is produced. Gentoo experiments are gated
on this file passing (PR-2b).
"""
import csv
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from associative_core import ContinuousCAM  # noqa: E402
from benchmarks.failure_mode_probe import (  # noqa: E402
    ABSORBED, FORKED, PLAIN_MISS, FailureModeRegistry, MODE_PRECEDENCE,
    OUT_COLS, build_synthetic_stream, label_probes, run_synthetic)


def _mem(dim=16, C=4, vigilance=0.85, **kw):
    """Static-vigilance memory (threshold known exactly) with provenance."""
    return ContinuousCAM(key_dim=dim, value_dim=C, max_entries=128,
                         vigilance=vigilance, track_provenance=True, **kw)


def _basis(dim, n):
    """n exactly-orthonormal keys."""
    return torch.eye(dim)[:n]


def _seed_class(mem, registry, key, label, C, n_writes=1):
    """Write key→one-hot(label) n times through the normal learn path."""
    y = F.one_hot(torch.tensor([label]), C).float()
    for _ in range(n_writes):
        ids = registry.next_ids("clean", 1)
        mem.learn_local(key.unsqueeze(0), y, record_ids=ids)
    return ids


# ---------------------------------------------------------------------------
# Contradiction mechanism
# ---------------------------------------------------------------------------
def test_contra_identical_key_forks_co_resident_slot():
    dim, C = 16, 4
    mem, reg = _mem(dim, C), FailureModeRegistry()
    k0 = _basis(dim, 1)[0]
    _seed_class(mem, reg, k0, label=0, C=C)
    assert int(mem.occupied.sum()) == 1

    rng = torch.Generator().manual_seed(0)
    ids = reg.inject_contradictions(
        mem, k0.unsqueeze(0), torch.tensor([0]), C, rate=1.0, rng=rng)
    assert len(ids) == 1
    meta = reg.contra[ids[0]]
    assert meta["outcome"] == FORKED

    # Co-residency: two occupied slots, keys identical (sim >= vigilance),
    # payloads disagreeing (cosine <= 0.5) — the silent fork, mechanically.
    occ = mem.occupied.nonzero(as_tuple=True)[0]
    assert occ.numel() == 2
    keys = F.normalize(mem.keys[occ].float(), dim=-1)
    assert float(keys[0] @ keys[1]) >= mem.vigilance
    vals = mem.values[occ].float()
    assert float(F.cosine_similarity(vals[0:1], vals[1:2])) <= 0.5

    forks = reg.contra_fork_slots(mem)
    assert len(forks) == 1
    fork_slot = next(iter(forks))
    assert int(mem.values[fork_slot].argmax()) == meta["label"]


def test_contra_far_key_is_plain_miss_not_fork():
    dim, C = 16, 4
    mem, reg = _mem(dim, C), FailureModeRegistry()
    b = _basis(dim, 2)
    _seed_class(mem, reg, b[0], label=0, C=C)

    # Orthogonal key: pre-write best sim = 0 < vigilance → plain miss.
    rng = torch.Generator().manual_seed(0)
    ids = reg.inject_contradictions(
        mem, b[1].unsqueeze(0), torch.tensor([1]), C, rate=1.0, rng=rng)
    assert reg.contra[ids[0]]["outcome"] == PLAIN_MISS
    assert reg.contra_fork_slots(mem) == set()
    assert int(mem.occupied.sum()) == 2  # new slot, but NOT gate-co-resident


def test_contra_agreeing_payload_is_absorbed():
    dim, C = 16, 4
    mem, reg = _mem(dim, C), FailureModeRegistry()
    k0 = _basis(dim, 1)[0]
    seed_ids = _seed_class(mem, reg, k0, label=0, C=C)
    slot = next(iter(mem.occupied.nonzero(as_tuple=True)[0].tolist()))

    # Force the injected "wrong" label to equal the stored class: with C
    # classes and true label C-1, generator seed picks shift such that... we
    # instead bypass randomness by injecting via the same code path with a
    # crafted true label so wrong == 0 is impossible; so call learn_local
    # semantics directly through the registry with rate 1 and check ABSORBED
    # via a same-class collision: true=1 forces wrong != 1, and with C=2
    # wrong is deterministically 0 == stored class.
    mem2, reg2 = _mem(dim, 2), FailureModeRegistry()
    _seed_class(mem2, reg2, k0, label=0, C=2)
    rng = torch.Generator().manual_seed(0)
    ids = reg2.inject_contradictions(
        mem2, k0.unsqueeze(0), torch.tensor([1]), 2, rate=1.0, rng=rng)
    assert reg2.contra[ids[0]]["outcome"] == ABSORBED
    assert reg2.contra_fork_slots(mem2) == set()
    assert int(mem2.occupied.sum()) == 1  # merged, no new slot
    # the merge unions provenance: seed id + injected id share the slot
    slot2 = int(mem2.occupied.nonzero(as_tuple=True)[0][0])
    assert len(mem2.records_for(slot2)) == 2
    del seed_ids, slot


def test_fork_slot_set_recomputed_from_live_provenance():
    """Eviction reuse replaces a slot's record set → fork flag must drop."""
    dim = 16
    mem, reg = _mem(dim, 4, vigilance=0.85), FailureModeRegistry()
    mem.max_entries = 128  # plenty; we evict manually below
    k0 = _basis(dim, 1)[0]
    _seed_class(mem, reg, k0, label=0, C=4)
    rng = torch.Generator().manual_seed(0)
    reg.inject_contradictions(mem, k0.unsqueeze(0), torch.tensor([0]), 4,
                              rate=1.0, rng=rng)
    forks = reg.contra_fork_slots(mem)
    assert len(forks) == 1
    slot = next(iter(forks))
    # Simulate eviction/reuse exactly as learn_local's miss path does:
    # new occupant, new record set.
    mem.slot_records[slot] = {("clean", 999_999)}
    assert reg.contra_fork_slots(mem) == set()


# ---------------------------------------------------------------------------
# Stale mechanism — supersession, not absence
# ---------------------------------------------------------------------------
def test_stale_merge_ema_freeze_strict():
    """K→A matured, K→B with payload cosine > 0.5 merges but barely moves the
    value (adaptive alpha decay): the slot still decodes to A → STALE_STRICT."""
    dim, C = 16, 4
    mem, reg = _mem(dim, C), FailureModeRegistry()
    b = _basis(dim, 2)
    K = b[0]
    # Soft payloads with cosine 0.96 > 0.5 but different argmax: A→class0, B→class1.
    A = F.normalize(torch.tensor([0.8, 0.6, 0.0, 0.0]), dim=-1).unsqueeze(0)
    B = F.normalize(torch.tensor([0.6, 0.8, 0.0, 0.0]), dim=-1).unsqueeze(0)
    group = reg.new_stale_group("K", pre_label=0, post_label=1)

    # Phase 1: mature the prototype (hit_counts -> 30, alpha ~ 0.04).
    for _ in range(30):
        reg.write_phase1(mem, group, K.unsqueeze(0), A)
    assert int(mem.occupied.sum()) == 1
    slot = int(mem.occupied.nonzero(as_tuple=True)[0][0])
    assert int(mem.hit_counts[slot]) == 30

    # A far class-1 prototype so the probe has a same-class candidate for
    # truth B=1 (probe rows need both a same- and other-class prototype).
    _seed_class(mem, reg, b[1], label=1, C=C)

    # Phase 2: supersession through the NORMAL learn path. cosine(A,B)=0.96
    # → hit → EMA merge into the mature slot, no new slot.
    reg.supersede(group, epoch=1)
    reg.write_phase2(mem, group, K.unsqueeze(0), B)
    assert int(mem.occupied.sum()) == 2
    assert int(mem.values[slot].float().argmax()) == 0  # still decodes to A

    stale = reg.stale_slots(mem)
    assert stale == {slot}

    # Probe K with CURRENT ground truth B=1: vote elects 0 (the mature slot
    # dominates), top-1 IS the stale slot, answer IS the pre-update payload.
    labeled = label_probes(mem, K.unsqueeze(0), torch.tensor([1]), reg)
    assert labeled["n_rows"] == 1
    assert labeled["failure_mode"][0] == "STALE_STRICT"
    assert int(labeled["stale_strict"][0]) == 1
    assert int(labeled["stale_lenient"][0]) == 1
    assert int(labeled["top1_slot"][0]) == slot


def test_stale_fork_path_co_resident_pre_update_support():
    """K→A matured (one-hot), K→B one-hot (cosine 0 <= 0.5) forks: the mature
    A slot stays co-resident and keeps winning for K → STALE_STRICT."""
    dim, C = 16, 4
    mem, reg = _mem(dim, C), FailureModeRegistry()
    b = _basis(dim, 2)
    K = b[0]
    A = F.one_hot(torch.tensor([0]), C).float()
    Bv = F.one_hot(torch.tensor([1]), C).float()
    group = reg.new_stale_group("K", pre_label=0, post_label=1)

    for _ in range(20):
        reg.write_phase1(mem, group, K.unsqueeze(0), A)
    a_slot = int(mem.occupied.nonzero(as_tuple=True)[0][0])

    reg.supersede(group, epoch=1)
    # B written with a slightly perturbed key so the probe at K proper still
    # ranks the matured A slot first (sims: A-slot ~1.0 > B-slot ~0.98).
    k_b = F.normalize(K + 0.20 * b[1], dim=-1)
    reg.write_phase2(mem, group, k_b.unsqueeze(0), Bv)
    occ = mem.occupied.nonzero(as_tuple=True)[0]
    assert occ.numel() == 2  # fork: pre-update support is co-resident

    assert reg.stale_slots(mem) == {a_slot}

    labeled = label_probes(mem, K.unsqueeze(0), torch.tensor([1]), reg)
    assert labeled["n_rows"] == 1
    assert labeled["failure_mode"][0] == "STALE_STRICT"
    assert int(labeled["top1_slot"][0]) == a_slot
    # the superseding B slot is in the top-k too: stale support competes
    # inside the same vote — exactly the contradiction-becomes-blend channel.
    assert int(labeled["n_stale_topk"][0]) == 1


def test_phase2_requires_supersede_first():
    mem, reg = _mem(), FailureModeRegistry()
    g = reg.new_stale_group("K", 0, 1)
    with pytest.raises(RuntimeError, match="supersede"):
        reg.write_phase2(mem, g, torch.eye(16)[:1],
                         F.one_hot(torch.tensor([1]), 4).float())


def test_stale_flag_drops_if_slot_absorbs_the_update():
    """If enough B writes flip the slot's decode to B, it is no longer stale."""
    dim, C = 16, 4
    mem, reg = _mem(dim, C), FailureModeRegistry()
    K = _basis(dim, 1)[0]
    A = F.normalize(torch.tensor([0.8, 0.6, 0.0, 0.0]), dim=-1).unsqueeze(0)
    B = F.normalize(torch.tensor([0.6, 0.8, 0.0, 0.0]), dim=-1).unsqueeze(0)
    g = reg.new_stale_group("K", 0, 1)
    reg.write_phase1(mem, g, K.unsqueeze(0), A)  # immature: hit_counts=1
    reg.supersede(g, epoch=1)
    slot = int(mem.occupied.nonzero(as_tuple=True)[0][0])
    assert reg.stale_slots(mem) == {slot}
    for _ in range(200):  # young prototype: alpha large enough to track B
        reg.write_phase2(mem, g, K.unsqueeze(0), B)
    assert int(mem.values[slot].float().argmax()) == 1
    assert reg.stale_slots(mem) == set()


# ---------------------------------------------------------------------------
# Label semantics on the probe side
# ---------------------------------------------------------------------------
def test_clean_arm_is_negative_control(tmp_path):
    out = tmp_path / "clean.csv"
    n = run_synthetic("clean", rate=0.0, epochs=4, supersede_epoch=-1,
                      out_path=out, seed=0)
    assert n > 0
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == n
    for r in rows:
        assert r["failure_mode"] in ("CORRECT", "BLENDED", "OTHER_WRONG")
        assert r["contradictory_strict"] == "0"
        assert r["contradictory_lenient"] == "0"
        assert r["stale_strict"] == "0"
        assert r["stale_lenient"] == "0"


def test_contra_arm_end_to_end(tmp_path):
    out = tmp_path / "contra.csv"
    n = run_synthetic("contra", rate=0.30, epochs=6, supersede_epoch=-1,
                      out_path=out, seed=0)
    assert n > 0
    rows = list(csv.DictReader(open(out)))
    assert set(rows[0].keys()) == set(OUT_COLS)
    allowed = set(MODE_PRECEDENCE) | {"CORRECT"}
    n_contra = 0
    for r in rows:
        assert r["failure_mode"] in allowed
        # strict => lenient, both modes
        assert int(r["contradictory_strict"]) <= int(r["contradictory_lenient"])
        assert int(r["stale_strict"]) <= int(r["stale_lenient"])
        # any wrong row with no flag must be BLENDED or OTHER_WRONG
        if r["vote_correct"] == "1":
            assert r["failure_mode"] == "CORRECT"
        n_contra += int(r["contradictory_lenient"])
        assert r["stale_lenient"] == "0"  # no stale protocol in this arm
    assert n_contra > 0, "30% injection over 6 epochs must produce at least " \
                         "one contradictory-flagged wrong probe"


def test_stale_arm_end_to_end(tmp_path):
    out = tmp_path / "stale.csv"
    n = run_synthetic("stale", rate=0.0, epochs=6, supersede_epoch=3,
                      out_path=out, seed=0)
    assert n > 0
    rows = list(csv.DictReader(open(out)))
    pre = [r for r in rows if int(r["epoch"]) < 3]
    post = [r for r in rows if int(r["epoch"]) >= 3]
    assert all(r["stale_lenient"] == "0" for r in pre), \
        "stale labels must not exist before supersession"
    assert any(int(r["stale_lenient"]) == 1 for r in post), \
        "post-supersession probes of the superseded class must surface " \
        "stale pre-update support"
    assert all(r["contradictory_lenient"] == "0" for r in rows)


def test_vote_pred_pinned_to_forward_argmax():
    """The labeled rows' vote_pred_label must equal forward().argmax for the
    same probes — the deployed prediction, not a re-derivation."""
    torch.manual_seed(3)
    batches, hq, hlab = build_synthetic_stream(epochs=3, seed=3)
    mem = _mem(dim=32, C=4)
    reg = FailureModeRegistry()
    rng = torch.Generator().manual_seed(3)
    for q, y, lab in batches:
        ids = reg.next_ids("clean", q.size(0))
        mem.learn_local(q, y, record_ids=ids)
        reg.inject_contradictions(mem, q, lab, 4, rate=0.25, rng=rng)
    labeled = label_probes(mem, hq, hlab, reg)
    assert labeled["n_rows"] > 0
    fwd_pred = mem.forward(hq).argmax(dim=-1)
    idx = labeled["probe_index"]
    assert torch.equal(labeled["per_probe"]["vote_pred_label"], fwd_pred[idx])


def test_label_probes_requires_provenance():
    mem = ContinuousCAM(key_dim=16, value_dim=4, track_provenance=False)
    mem.learn_local(torch.eye(16)[:2],
                    F.one_hot(torch.tensor([0, 1]), 4).float())
    with pytest.raises(RuntimeError, match="track_provenance"):
        label_probes(mem, torch.eye(16)[:1], torch.tensor([0]),
                     FailureModeRegistry())


def test_inject_requires_provenance():
    mem = ContinuousCAM(key_dim=16, value_dim=4, track_provenance=False)
    rng = torch.Generator().manual_seed(0)
    with pytest.raises(RuntimeError, match="track_provenance"):
        FailureModeRegistry().inject_contradictions(
            mem, torch.eye(16)[:1], torch.tensor([0]), 4, rate=1.0, rng=rng)


def test_label_probes_rejects_unreplicated_retrieval_config():
    """NSTP / truncation change candidate survival; labeling must refuse
    rather than silently mislabel."""
    mem = _mem()
    mem.nstp = object()  # any attached controller
    with pytest.raises(RuntimeError, match="NSTP"):
        label_probes(mem, torch.eye(16)[:1], torch.tensor([0]),
                     FailureModeRegistry())
