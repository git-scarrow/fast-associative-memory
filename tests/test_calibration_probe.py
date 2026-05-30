"""Per-probe calibration telemetry (issue #82 analysis branch).

`probe_cross_class_similarity(return_per_probe=True)` is read-only instrumentation
for the retrieval-confidence calibration study. These tests pin the two
properties the study relies on:

  1. The flag is side-effect-free: every aggregate already published is
     bit-identical with the flag on vs off, and the per-probe rows reduce back
     to those aggregates.
  2. `vote_pred_label` is the class `forward()` would actually elect for that
     probe -- so `vote_correct` is a faithful evaluation label for the deployed
     prediction, not a re-derivation that could silently drift from forward().
"""
import math

import torch
import torch.nn.functional as F

from associative_core import ContinuousCAM
from dynamic_vigilance import DynamicVigilance, RetrievalFloorPolicy


def _build(seed=0, floor=True, noise=0.25, n_per=40, held_per=30, C=4, D=32):
    torch.manual_seed(seed)
    centers = F.normalize(torch.randn(C, D), dim=-1)
    lab = torch.arange(C).repeat_interleave(n_per)
    q = F.normalize(centers[lab] + noise * torch.randn(C * n_per, D), dim=-1)
    y = F.one_hot(lab, C).float()
    mem = ContinuousCAM(
        key_dim=D, value_dim=C, max_entries=512,
        dynamic_vigilance=DynamicVigilance(),
        retrieval_floor_policy=RetrievalFloorPolicy() if floor else None)
    mem.learn_local(q, y)
    hl = torch.arange(C).repeat_interleave(held_per)
    hq = F.normalize(centers[hl] + noise * torch.randn(C * held_per, D), dim=-1)
    return mem, hq, hl


def test_flag_off_is_bit_identical():
    mem, hq, hl = _build()
    a = mem.probe_cross_class_similarity(hq, hl)
    b = mem.probe_cross_class_similarity(hq, hl, return_per_probe=True)
    assert "per_probe" not in a
    assert "per_probe" in b
    b_agg = {k: v for k, v in b.items() if k != "per_probe"}
    assert set(a) == set(b_agg)
    for k in a:
        av, bv = a[k], b_agg[k]
        if isinstance(av, float) and math.isnan(av):
            assert isinstance(bv, float) and math.isnan(bv)
        else:
            assert av == bv, f"{k}: {av} != {bv}"


def test_per_probe_reduces_to_aggregates():
    mem, hq, hl = _build()
    out = mem.probe_cross_class_similarity(hq, hl, return_per_probe=True)
    pp = out["per_probe"]
    V = pp["true_label"].numel()
    assert V == out["n_vote_probes"] > 0
    # all per-probe arrays share the voting-row count
    for k, v in pp.items():
        assert v.numel() == V, f"{k} has {v.numel()} rows, expected {V}"

    def close(x, y, t=1e-5):
        return abs(x - y) <= t

    assert close(pp["top1_top2_margin"].mean().item(), out["mean_top1_top2_margin"])
    assert close(pp["effective_support"].mean().item(), out["mean_effective_support"])
    assert close(pp["max_vote_weight"].mean().item(), out["mean_max_vote_weight"])
    assert close(pp["vote_entropy"].mean().item(), out["mean_vote_entropy"])
    assert close(pp["n_surviving_votes"].float().mean().item(),
                 out["mean_n_surviving_votes"])
    assert close((pp["margin_bucket"] == 2).float().mean().item(),
                 out["frac_margin_recoverable"])
    assert close((pp["margin_bucket"] == 1).float().mean().item(),
                 out["frac_margin_borderline"])
    assert close((pp["margin_bucket"] == 0).float().mean().item(),
                 out["frac_margin_forced"])
    blended = pp["is_blended"].bool()
    if blended.any():
        assert close(pp["top1_correct"][blended].float().mean().item(),
                     out["frac_blend_top1_correct"])


def test_vote_pred_matches_forward_row_for_row():
    # Floor OFF and all classes present => every held probe is valid AND votes,
    # so the per-probe rows are in held-set order and align with forward().
    mem, hq, hl = _build(floor=False)
    out = mem.probe_cross_class_similarity(hq, hl, return_per_probe=True)
    pp = out["per_probe"]
    assert pp["true_label"].numel() == hl.numel(), "expected no rows filtered"
    assert torch.equal(pp["true_label"], hl)
    fwd_pred = mem.forward(hq).argmax(dim=-1)
    assert torch.equal(pp["vote_pred_label"], fwd_pred), \
        "vote_pred_label must equal forward().argmax row-for-row"
    # and the label column agrees with the elected-class correctness
    assert torch.equal(pp["vote_correct"], (fwd_pred == hl).long())


def test_vote_pred_matches_forward_with_nstp_attached():
    # The per-probe vote must reproduce forward()'s DEPLOYED retrieval rule, which
    # includes NSTP lateral inhibition when self.nstp is set. forward() prunes the
    # top-k with NSTP before the floor/softmax; the probe must too, or
    # vote_pred_label would be labeled against a different rule than the one being
    # evaluated. Floor OFF + all classes present => rows align with forward().
    from nstp import NSTPController

    mem, hq, hl = _build(floor=False)
    # Probe with NO NSTP: floor is off and there is no truncation, so the ONLY
    # thing that can mask a candidate is NSTP — every row keeps all k survivors.
    surv_no_nstp = mem.probe_cross_class_similarity(
        hq, hl, return_per_probe=True)["per_probe"]["n_surviving_votes"]

    mem.nstp = NSTPController(sibling_threshold=0.5, depth_epsilon=0.10)
    fwd_pred = mem.forward(hq).argmax(dim=-1)
    out = mem.probe_cross_class_similarity(hq, hl, return_per_probe=True)
    pp = out["per_probe"]

    assert pp["true_label"].numel() == hl.numel(), \
        "NSTP fully pruned a row; pick a gentler controller for this alignment test"
    # THE PIN, under NSTP: the probe's elected class equals forward().argmax.
    assert torch.equal(pp["vote_pred_label"], fwd_pred), \
        "vote_pred_label must equal forward().argmax row-for-row WITH nstp"
    assert torch.equal(pp["vote_correct"], (fwd_pred == hl).long())
    # Regression guard: the probe must actually RUN NSTP. With the floor off and
    # no truncation, pruning is the only way a candidate can be masked, so a probe
    # that (re)skips NSTP would leave n_surviving_votes unchanged. Require it to
    # drop for at least one row.
    assert bool((pp["n_surviving_votes"] < surv_no_nstp).any()), \
        "probe did not apply NSTP — n_surviving_votes unchanged vs no-NSTP probe"
