"""Tests for the retrieval-confidence / effective-support probe telemetry (#82).

These telemetry keys decompose late-epoch retrieval saturation into *premature*
(a usable within-class margin the soft vote is fumbling) vs *forced* (the
manifold has genuinely collapsed and no inference-time gate can recover the
signal). The label-free signals (top1_top2_margin, entropy, effective_support)
are deployable at inference; the label-aware ones (margin-regime split,
frac_blend_top1_correct) are diagnostic.
"""
import math

import torch
import torch.nn.functional as F

from associative_core import ContinuousCAM


_TELEMETRY_KEYS = (
    "mean_top1_sim", "mean_top2_sim", "mean_top1_top2_margin",
    "median_top1_top2_margin", "mean_vote_entropy", "mean_effective_support",
    "median_effective_support", "mean_max_vote_weight", "mean_n_surviving_votes",
    "frac_low_margin", "frac_high_entropy", "frac_margin_recoverable",
    "frac_margin_borderline", "frac_margin_forced", "frac_blend_top1_correct",
)


def _build_cam(centers, contraction, attractor, n_per_class, noise, seed,
               key_dim, n_classes, vigilance=0.5):
    """Populate a CAM with one mixed batch and return (cam, probe_q, probe_lab)."""
    gen = torch.Generator().manual_seed(seed)
    mixed = F.normalize((1.0 - contraction) * centers + contraction * attractor,
                        dim=-1)
    lab = torch.arange(n_classes).repeat_interleave(n_per_class)
    q = F.normalize(mixed[lab] + noise * torch.randn(lab.size(0), key_dim,
                                                      generator=gen), dim=-1)
    cam = ContinuousCAM(key_dim=key_dim, value_dim=n_classes,
                        max_entries=4096, vigilance=vigilance)
    cam.learn_local(q, F.one_hot(lab, n_classes).float())
    pq = F.normalize(mixed[lab] + noise * torch.randn(lab.size(0), key_dim,
                                                      generator=gen), dim=-1)
    return cam, pq, lab


def _centers(n_classes, key_dim, seed=0):
    gen = torch.Generator().manual_seed(seed)
    centers = F.normalize(torch.randn(n_classes, key_dim, generator=gen), dim=-1)
    attractor = F.normalize(torch.randn(1, key_dim, generator=gen), dim=-1)
    return centers, attractor


def test_telemetry_keys_present_and_finite():
    centers, attractor = _centers(6, 32)
    cam, pq, lab = _build_cam(centers, 0.0, attractor, 20, 0.1, 1, 32, 6)
    out = cam.probe_cross_class_similarity(pq, lab)
    for key in _TELEMETRY_KEYS:
        assert key in out, f"missing telemetry key {key}"
    # All except frac_blend_top1_correct (NaN when nothing blended) must be finite.
    for key in _TELEMETRY_KEYS:
        if key == "frac_blend_top1_correct":
            continue
        assert math.isfinite(out[key]), f"{key} is not finite: {out[key]}"


def test_effective_support_within_bounds():
    centers, attractor = _centers(6, 32)
    cam, pq, lab = _build_cam(centers, 0.3, attractor, 20, 0.1, 2, 32, 6)
    out = cam.probe_cross_class_similarity(pq, lab)
    k = min(cam.inference_k, int(out["n_occupied"]))
    # exp(entropy) is in [1, k]: 1 = winner-take-all, k = flat over survivors.
    assert 1.0 - 1e-5 <= out["mean_effective_support"] <= k + 1e-5
    assert 1.0 - 1e-5 <= out["median_effective_support"] <= k + 1e-5
    assert 0.0 <= out["mean_max_vote_weight"] <= 1.0 + 1e-6


def test_margin_regime_bins_partition_to_one():
    centers, attractor = _centers(6, 32)
    cam, pq, lab = _build_cam(centers, 0.6, attractor, 20, 0.1, 3, 32, 6)
    out = cam.probe_cross_class_similarity(pq, lab)
    total = (out["frac_margin_recoverable"]
             + out["frac_margin_borderline"]
             + out["frac_margin_forced"])
    assert abs(total - 1.0) < 1e-5


def test_separated_manifold_is_recoverable_not_forced():
    """At contraction 0 the manifold is well separated: no forced margins."""
    centers, attractor = _centers(6, 48)
    cam, pq, lab = _build_cam(centers, 0.0, attractor, 25, 0.08, 4, 48, 6)
    out = cam.probe_cross_class_similarity(pq, lab)
    assert out["frac_margin_recoverable"] > 0.95
    assert out["frac_margin_forced"] < 0.05
    assert out["mean_offclass_weight"] < 0.05


def test_collapsed_manifold_is_forced_and_top1_wrong():
    """Near-total collapse: most blended probes have a WRONG top-1 neighbour, so
    confidence-weighting cannot recover them (forced saturation)."""
    centers, attractor = _centers(6, 48)
    cam, pq, lab = _build_cam(centers, 0.97, attractor, 25, 0.1, 5, 48, 6,
                              vigilance=0.3)
    out = cam.probe_cross_class_similarity(pq, lab)
    assert out["frac_margin_forced"] > 0.5
    # Of the probes whose vote actually blends, few have the correct top-1.
    assert out["frac_blend_top1_correct"] < 0.5
    assert out["mean_offclass_weight"] > 0.3


def test_recoverable_vs_forced_orders_with_contraction():
    """frac_margin_forced rises monotonically-ish as the manifold contracts."""
    centers, attractor = _centers(6, 48)
    forced = []
    for c in (0.0, 0.5, 0.97):
        cam, pq, lab = _build_cam(centers, c, attractor, 25, 0.1, 6, 48, 6,
                                  vigilance=0.3)
        out = cam.probe_cross_class_similarity(pq, lab)
        forced.append(out["frac_margin_forced"])
    assert forced[0] <= forced[1] <= forced[2]
    assert forced[2] > forced[0]
