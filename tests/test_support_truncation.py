"""Tests for adaptive vote-support truncation (issue #82 intervention).

The intervention masks each probe's off-class vote tail (candidates more than
``window_steps`` softmax-temperature steps below that row's own top-1) before the
softmax vote, gated on a label-free top1-top2 confidence signal so the
collapsed/forced tail (top1 ≈ top2) is left untouched. These tests cover the
policy's masking semantics in isolation and its wiring into ``ContinuousCAM``
(off-by-default parity, blend reduction in a recoverable regime, and the gate's
no-gamble guard in a forced regime).
"""
import math

import pytest
import torch
import torch.nn.functional as F

from associative_core import ContinuousCAM
from dynamic_vigilance import RetrievalTruncationPolicy


# ---------------------------------------------------------------------------
# Policy masking semantics (pure tensor logic, deterministic)
# ---------------------------------------------------------------------------

def test_inactive_policy_is_total_noop():
    pol = RetrievalTruncationPolicy(window_steps=0.0)
    assert not pol.active
    sims = torch.tensor([[0.9, 0.5, 0.1], [0.5, 0.49, 0.48]])
    keep = pol.keep_mask(sims, temp=0.05)
    assert keep.all()


def test_window_drops_tail_relative_to_top1():
    # gate off (default) → window-only. temp=0.05 → window 2*temp=0.10.
    pol = RetrievalTruncationPolicy(window_steps=2.0, gate_steps=0.0)
    sims = torch.tensor([
        [0.90, 0.85, 0.40, 0.10],   # cutoff 0.80: keep 0.90, 0.85; drop 0.40, 0.10
        [0.50, 0.49, 0.48, 0.47],   # cutoff 0.40: all within window → keep all
    ])
    keep = pol.keep_mask(sims, temp=0.05)
    assert keep[0].tolist() == [True, True, False, False]
    assert keep[1].all()           # co-located row: self-gated no-op


def test_optional_gate_skips_near_tied_rows():
    # With gate_steps>0, a row whose top1≈top2 is left fully intact even if it
    # has a droppable tail (the conservative isolated-winner-only mode).
    pol = RetrievalTruncationPolicy(window_steps=2.0, gate_steps=1.0)
    sims = torch.tensor([
        [0.90, 0.50, 0.40, 0.10],   # gap 0.40 >= 0.05 → gated on; cutoff 0.80
        [0.90, 0.89, 0.40, 0.10],   # gap 0.01 <  0.05 → gate closed → keep all
    ])
    keep = pol.keep_mask(sims, temp=0.05)
    assert keep[0].tolist() == [True, False, False, False]
    assert keep[1].all()


def test_top1_always_survives_even_with_tiny_window():
    pol = RetrievalTruncationPolicy(window_steps=0.01, gate_steps=0.0)
    sims = torch.tensor([[0.9, 0.2, 0.1]])
    keep = pol.keep_mask(sims, temp=0.05)
    assert bool(keep[0, 0]) is True
    # everything far below top-1 is dropped
    assert keep[0].tolist() == [True, False, False]


def test_gate_zero_is_always_on():
    # gate_steps=0 → no confidence gate → truncate even a near-flat row.
    pol = RetrievalTruncationPolicy(window_steps=2.0, gate_steps=0.0)
    sims = torch.tensor([[0.50, 0.49, 0.30]])   # window 2*0.05=0.10 → cutoff 0.40
    keep = pol.keep_mask(sims, temp=0.05)
    assert keep[0].tolist() == [True, True, False]


def test_high_gate_disables_truncation_when_no_clear_winner():
    pol = RetrievalTruncationPolicy(window_steps=1.0, gate_steps=10.0)
    sims = torch.tensor([[0.90, 0.50, 0.10]])   # gap 0.40 < 10*0.05=0.50 → closed
    keep = pol.keep_mask(sims, temp=0.05)
    assert keep.all()


def test_fully_masked_row_yields_no_new_masking():
    # A row already all -inf (e.g. floor-masked) must stay all-True here so the
    # downstream fully-masked-row guard owns it; truncation adds nothing.
    pol = RetrievalTruncationPolicy(window_steps=4.0, gate_steps=1.0)
    neg = -float("inf")
    sims = torch.tensor([[neg, neg, neg], [0.9, 0.5, 0.1]])
    keep = pol.keep_mask(sims, temp=0.05)
    assert keep[0].all()                 # no new masking on the dead row
    assert keep[1].tolist() == [True, False, False]


def test_single_candidate_row_has_no_gate_but_keeps_top1():
    pol = RetrievalTruncationPolicy(window_steps=4.0, gate_steps=1.0)
    sims = torch.tensor([[0.9]])
    keep = pol.keep_mask(sims, temp=0.05)
    assert keep.all()


def test_validation_rejects_negative_params():
    with pytest.raises(ValueError):
        RetrievalTruncationPolicy(window_steps=-1.0)
    with pytest.raises(ValueError):
        RetrievalTruncationPolicy(gate_steps=-0.5)


# ---------------------------------------------------------------------------
# ContinuousCAM integration
# ---------------------------------------------------------------------------
#
# The synthetic drift manifold used elsewhere produces very flat neighbourhoods
# (effective support ~17/20), whose blend is a within-window cluster, not a
# separated tail — so it cannot exercise the truncation mechanism. We instead
# build a controlled memory with explicit cosine structure: a tight within-class
# core well above a *numerous* off-class tail (the recoverable geometry), which
# matches the real-20NG mid-window support (~4-6) where #82's premature blend
# lives. ``forced`` collapses the two into one co-located band.

_DIM = 64
_N_CLASSES = 6


def _anchors(n_classes=_N_CLASSES, dim=_DIM, seed=1):
    gen = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n_classes, dim, generator=gen), dim=-1)


def _with_cos(unit_dir, target, gen, dim=_DIM):
    """A unit vector whose cosine to ``unit_dir`` is ``target`` (random else)."""
    r = torch.randn(dim, generator=gen)
    r = r - (r @ unit_dir) * unit_dir
    r = F.normalize(r, dim=0)
    return target * unit_dir + math.sqrt(max(0.0, 1.0 - target ** 2)) * r


def _seed_memory(anchors, within_cos, off_cos, seed=1):
    """Keys/labels: 3 within-class protos at ``within_cos`` per class, plus a
    tail of off-class protos sitting at ``off_cos`` to each class's anchor."""
    gen = torch.Generator().manual_seed(seed)
    keys, labs = [], []
    n = anchors.size(0)
    for ci in range(n):
        for _ in range(3):
            keys.append(_with_cos(anchors[ci], within_cos, gen)); labs.append(ci)
    for ci in range(n):
        for cj in range(n):
            if cj == ci:
                continue
            for _ in range(4):
                keys.append(_with_cos(anchors[ci], off_cos, gen)); labs.append(cj)
    return F.normalize(torch.stack(keys), dim=-1), torch.tensor(labs)


def _probes(anchors, query_cos=0.92, per_class=20, seed=7):
    gen = torch.Generator().manual_seed(seed)
    pq, pl = [], []
    for ci in range(anchors.size(0)):
        for _ in range(per_class):
            pq.append(_with_cos(anchors[ci], query_cos, gen)); pl.append(ci)
    return F.normalize(torch.stack(pq), dim=-1), torch.tensor(pl)


def _make_cam(keys, labs, policy=None):
    cam = ContinuousCAM(key_dim=_DIM, value_dim=_N_CLASSES, max_entries=2048,
                        vigilance=0.999, inference_k=20, inference_temp=0.05,
                        retrieval_truncation_policy=policy)
    cam.learn_local(keys, F.one_hot(labs, _N_CLASSES).float())
    return cam


def test_off_by_default_is_bit_identical():
    """No policy attached → forward() and the probe match a vanilla CAM exactly."""
    anchors = _anchors()
    keys, labs = _seed_memory(anchors, within_cos=0.97, off_cos=0.78)
    pq, pl = _probes(anchors)
    cam_a, cam_b = _make_cam(keys, labs), _make_cam(keys, labs)
    assert torch.allclose(cam_a.forward(pq), cam_b.forward(pq), atol=0, rtol=0)
    pa = cam_a.probe_cross_class_similarity(pq, pl)
    pb = cam_b.probe_cross_class_similarity(pq, pl)
    assert pa["mean_offclass_weight"] == pb["mean_offclass_weight"]
    assert pa["mean_effective_support"] == pb["mean_effective_support"]


def test_truncation_reduces_blend_and_support_in_recoverable_regime():
    """Separated core above a numerous off-class tail (correct top-1): the window
    slices the tail, cutting off-class vote mass and deflating effective support,
    with accuracy preserved."""
    anchors = _anchors()
    keys, labs = _seed_memory(anchors, within_cos=0.97, off_cos=0.78)
    pq, pl = _probes(anchors)
    pol = RetrievalTruncationPolicy(window_steps=2.0, gate_steps=0.0)
    base, interv = _make_cam(keys, labs), _make_cam(keys, labs, pol)

    pb = base.probe_cross_class_similarity(pq, pl)
    pi = interv.probe_cross_class_similarity(pq, pl)

    assert pb["mean_offclass_weight"] > 0.10                      # baseline blends
    assert pi["mean_offclass_weight"] < 0.3 * pb["mean_offclass_weight"]
    assert pi["mean_effective_support"] < pb["mean_effective_support"]

    acc_b = (base.forward(pq).argmax(-1) == pl).float().mean().item()
    acc_i = (interv.forward(pq).argmax(-1) == pl).float().mean().item()
    assert acc_i >= acc_b - 1e-6


def test_window_does_not_rescue_forced_regime():
    """Co-located within/off (forced collapse): the window has no clean tail to
    slice, so it neither rescues accuracy nor de-blends the neighbourhood — the
    'do not claim to fix the forced tail' guarantee. (Accuracy stays collapsed
    and the vote stays heavily blended even with truncation on.)"""
    anchors = _anchors()
    keys, labs = _seed_memory(anchors, within_cos=0.80, off_cos=0.80)
    pq, pl = _probes(anchors)
    pol = RetrievalTruncationPolicy(window_steps=2.0, gate_steps=0.0)
    base, interv = _make_cam(keys, labs), _make_cam(keys, labs, pol)

    acc_b = (base.forward(pq).argmax(-1) == pl).float().mean().item()
    acc_i = (interv.forward(pq).argmax(-1) == pl).float().mean().item()
    # No rescue: accuracy is not meaningfully improved by truncation.
    assert acc_i <= acc_b + 0.02
    assert acc_i < 0.5                                            # stays collapsed
    # The neighbourhood remains deeply blended — not sharpened onto a winner.
    pi = interv.probe_cross_class_similarity(pq, pl)
    assert pi["mean_offclass_weight"] > 0.5


def test_high_gate_is_a_perfect_noop():
    """gate_steps so high it never opens → truncation disabled per-row → the vote
    is bit-identical to the baseline regardless of geometry."""
    anchors = _anchors()
    keys, labs = _seed_memory(anchors, within_cos=0.97, off_cos=0.78)
    pq, _ = _probes(anchors)
    pol = RetrievalTruncationPolicy(window_steps=2.0, gate_steps=1e6)
    base, interv = _make_cam(keys, labs), _make_cam(keys, labs, pol)
    assert torch.allclose(base.forward(pq), interv.forward(pq), atol=1e-7)


def test_trace_marks_truncation_victims_as_stage_3():
    anchors = _anchors()
    keys, labs = _seed_memory(anchors, within_cos=0.97, off_cos=0.78)
    pq, _ = _probes(anchors)
    pol = RetrievalTruncationPolicy(window_steps=2.0, gate_steps=0.0)
    cam = _make_cam(keys, labs, pol)
    _, tr = cam.forward(pq, trace=True)
    # Stage 3 (truncation) should appear among rejected candidates for at least
    # one probe, and survivors (stage 4) are never in the rejected set.
    assert (tr.rejection_stage == 3).any()
    assert not (tr.rejection_stage == 4).any()
