"""Held-out two-axis abstention calibration (issue #82 / PR #85 follow-up).

These tests pin the four guarantees the abstention study rests on:

  1. The epoch split is BY EPOCH and leak-free — train and test never share an
     epoch (parity and contiguous strategies both).
  2. The detector (standardisation, logistic coefficients) and the abstention
     threshold are fit on TRAIN epochs ONLY — held-out rows cannot influence any
     fit decision.
  3. Abstention is a pure post-hoc MASK: it never rewrites the elected class of a
     retained row, and never mutates its inputs.
  4. ``forward()`` is untouched — the abstention path lives entirely in the
     benchmark module; ``ContinuousCAM`` gains no abstention API and its
     ``forward()`` output is bit-identical whether or not the analysis is run.
"""
import numpy as np
import torch
import torch.nn.functional as F

from associative_core import ContinuousCAM
from dynamic_vigilance import DynamicVigilance, RetrievalFloorPolicy
from benchmarks.heldout_abstention import (
    AbstentionDetector, RANK_GAP, MANIFOLD_SUPPORT, abstain_mask,
    operating_point, select_threshold_youden, split_epochs_contiguous,
    split_epochs_parity)


# --------------------------------------------------------------------------
# small, deterministic synthetic forced-zone dump
# --------------------------------------------------------------------------
def _synthetic(n_per_epoch=120, epochs=range(8, 30), seed=0):
    """A two-axis-separable forced-zone dataset spread across many epochs.

    rank_gap is anti-correlated with correctness (false collapse: sharp top-1 ->
    wrong) and manifold_support is positively correlated, so a two-axis detector
    has real signal to learn — but the labels are pure synthetic, never used at
    'inference' (only to fit/score)."""
    rng = np.random.default_rng(seed)
    rows = {"epoch": [], RANK_GAP: [], MANIFOLD_SUPPORT: [],
            "vote_correct": [], "top1_correct": [], "vote_pred_label": []}
    for e in epochs:
        for _ in range(n_per_epoch):
            correct = int(rng.random() < 0.35)
            # higher rank_gap when WRONG, higher support when CORRECT
            rg = rng.normal(0.6 - 0.4 * correct, 0.2)
            su = rng.normal(1.5 + 1.2 * correct, 0.5)
            rows["epoch"].append(e)
            rows[RANK_GAP].append(rg)
            rows[MANIFOLD_SUPPORT].append(max(su, 0.1))
            rows["vote_correct"].append(correct)
            rows["top1_correct"].append(correct)
            rows["vote_pred_label"].append(rng.integers(0, 4))
    return {k: np.asarray(v, dtype=float) for k, v in rows.items()}


# --------------------------------------------------------------------------
# 1. epoch split has no epoch overlap
# --------------------------------------------------------------------------
def test_epoch_split_has_no_overlap():
    epochs = list(range(8, 30))
    for tr, te in (split_epochs_parity(epochs),
                   split_epochs_contiguous(epochs, train_frac=0.6)):
        assert set(tr).isdisjoint(set(te)), "train/test share an epoch"
        assert set(tr) | set(te) == set(epochs), "split dropped/duplicated epochs"
        assert tr and te, "a fold is empty"
    # parity really is even/odd
    tr, te = split_epochs_parity(epochs)
    assert all(e % 2 == 0 for e in tr) and all(e % 2 == 1 for e in te)
    # contiguous really is a forward cut (all train epochs precede all test)
    tr, te = split_epochs_contiguous(epochs, train_frac=0.6)
    assert max(tr) < min(te)


# --------------------------------------------------------------------------
# 2. threshold (and the whole detector) is fit on TRAIN only
# --------------------------------------------------------------------------
def test_detector_and_threshold_fit_on_train_only():
    cols = _synthetic()
    ep = cols["epoch"]
    tr, te = split_epochs_parity(sorted(set(int(e) for e in ep)))
    train_mask = np.isin(ep.astype(int), tr)
    test_mask = np.isin(ep.astype(int), te)

    det = AbstentionDetector([RANK_GAP, MANIFOLD_SUPPORT]).fit(
        cols, train_mask, "vote_correct")
    mean0, std0 = det.mean_.copy(), det.std_.copy()
    coef0 = det.model_.coef_.copy()
    p_train = det.score(cols, train_mask)
    y_train = cols["vote_correct"][train_mask].astype(int)
    thr0 = select_threshold_youden(p_train, y_train)

    # corrupt EVERY held-out row's features and labels, leave train rows intact
    cols2 = {k: v.copy() for k, v in cols.items()}
    cols2[RANK_GAP][test_mask] = 999.0
    cols2[MANIFOLD_SUPPORT][test_mask] = -999.0
    cols2["vote_correct"][test_mask] = 1 - cols2["vote_correct"][test_mask]

    det2 = AbstentionDetector([RANK_GAP, MANIFOLD_SUPPORT]).fit(
        cols2, train_mask, "vote_correct")
    # the fit is identical because it only ever reads train rows
    assert np.allclose(det2.mean_, mean0) and np.allclose(det2.std_, std0)
    assert np.allclose(det2.model_.coef_, coef0)
    p_train2 = det2.score(cols2, train_mask)
    assert np.allclose(p_train2, p_train)
    thr1 = select_threshold_youden(p_train2, cols2["vote_correct"][train_mask].astype(int))
    assert thr1 == thr0, "threshold moved when only held-out rows changed"


# --------------------------------------------------------------------------
# 3. abstention does not alter predictions (pure mask)
# --------------------------------------------------------------------------
def test_abstention_is_a_pure_mask_over_predictions():
    rng = np.random.default_rng(1)
    scores = rng.random(200)
    y = (rng.random(200) < 0.4).astype(int)
    preds = rng.integers(0, 4, size=200)
    preds_before = preds.copy()
    scores_before = scores.copy()

    thr = 0.5
    keep = abstain_mask(scores, thr)
    # retained predictions are exactly the original elected classes, untouched
    assert np.array_equal(preds[keep], preds_before[keep])
    # mask never mutates its inputs
    assert np.array_equal(preds, preds_before)
    assert np.array_equal(scores, scores_before)

    # operating_point asserts the invariant; honest preds pass, a rewritten
    # prediction on a retained row must trip the guard.
    op = operating_point(scores, y, thr, preds=preds, base_preds=preds)
    assert op["n_retained"] == int(keep.sum())
    assert op["abstention_rate"] == round(1 - keep.sum() / len(y), 4)
    tampered = preds.copy()
    if keep.any():
        tampered[np.argmax(keep)] += 1  # rewrite a RETAINED prediction
        raised = False
        try:
            operating_point(scores, y, thr, preds=tampered, base_preds=preds)
        except AssertionError:
            raised = True
        assert raised, "operating_point must reject a rewritten retained prediction"


# --------------------------------------------------------------------------
# 4. forward() is unchanged when no abstention analysis is requested
# --------------------------------------------------------------------------
def test_forward_unchanged_and_no_abstention_api_on_model():
    torch.manual_seed(0)
    C, D = 4, 32
    centers = F.normalize(torch.randn(C, D), dim=-1)
    lab = torch.arange(C).repeat_interleave(40)
    q = F.normalize(centers[lab] + 0.25 * torch.randn(C * 40, D), dim=-1)
    mem = ContinuousCAM(
        key_dim=D, value_dim=C, max_entries=512,
        dynamic_vigilance=DynamicVigilance(),
        retrieval_floor_policy=RetrievalFloorPolicy())
    mem.learn_local(q, F.one_hot(lab, C).float())
    hl = torch.arange(C).repeat_interleave(30)
    hq = F.normalize(centers[hl] + 0.25 * torch.randn(C * 30, D), dim=-1)

    f_before = mem.forward(hq).clone()

    # run the full abstention pipeline on a synthetic dump — it touches arrays
    # only, never the model.
    cols = _synthetic()
    ep = cols["epoch"]
    tr, _ = split_epochs_parity(sorted(set(int(e) for e in ep)))
    train_mask = np.isin(ep.astype(int), tr)
    det = AbstentionDetector([RANK_GAP, MANIFOLD_SUPPORT]).fit(
        cols, train_mask, "vote_correct")
    _ = det.score(cols, ~train_mask)

    f_after = mem.forward(hq)
    assert torch.equal(f_before, f_after), "forward() output changed"

    # the abstention work added no method/attribute to the model class
    for name in ("abstain", "abstention", "abstention_mask", "set_abstention",
                 "abstain_mask", "two_axis_detector"):
        assert not hasattr(mem, name), f"ContinuousCAM unexpectedly exposes {name!r}"
