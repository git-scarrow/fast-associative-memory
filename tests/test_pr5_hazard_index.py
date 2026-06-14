"""PR-5 step 1 pins — candidate hazard indices and the promotion rule.

Hermetic: synthetic unit features only, no cache, no run artifacts. Pins
(a) the candidate definitions on constructed geometry where the right
answer is known by construction, and (b) the promotion rule's exact
admissibility logic (D strictly worst AND {A, C} strictly most benign),
including the stop recommendation when nothing is admissible — so the
rule cannot drift after the spent-pair values are known.
"""
import math

import torch
import torch.nn.functional as F

from benchmarks.pr5_hazard_index import (
    CANDIDATES, PAIRS, candidate_indices, evaluate)


def _clustered_features(centers, n=64, spread=0.05, seed=0):
    """Unit features around unit centers; labels = center index."""
    gen = torch.Generator().manual_seed(seed)
    feats, labels = [], []
    for i, c in enumerate(centers):
        c = F.normalize(torch.tensor(c, dtype=torch.float32), dim=-1)
        noise = torch.randn(n, c.shape[0], generator=gen) * spread
        feats.append(F.normalize(c + noise, dim=-1))
        labels.append(torch.full((n,), i, dtype=torch.long))
    return torch.cat(feats), torch.cat(labels)


def test_pair_registry_matches_run_matrix():
    # Verbatim from pr4_run_matrix.sh — the five spent configurations.
    assert PAIRS["pairD"] == {"classes": [10, 28, 32, 95], "attractor": 52}
    assert PAIRS["pairA"]["classes"] == [0, 8, 19, 33]
    assert list(PAIRS) == ["pairA", "pairC", "pairD", "pairB", "pairE"]
    assert CANDIDATES == ("attribution_ratio", "confusion_rate",
                          "fork_confusion_rate")


def test_candidates_on_separated_geometry_are_clean():
    # Four near-orthogonal clusters: no confusion, attribution ratio
    # governed by the (small, positive) cross-cosines.
    eye = torch.eye(8)
    feats, labels = _clustered_features(
        [eye[0], eye[2], eye[4], eye[6]], spread=0.02)
    v = candidate_indices(feats, labels, [0, 1, 2, 3])
    assert v["fork_pair"] == [0, 3]
    assert v["confusion_rate"] == 0.0
    assert v["fork_confusion_rate"] == 0.0
    assert v["n_samples"] == 256


def test_candidates_flag_bystander_pull_on_fork_side():
    # Class 1 (bystander) nearly coincides with class 0 (fork side I);
    # the fork pair (0, 3) is far apart. This is the pair-D shape:
    # high attribution ratio, confusion concentrated fork<->bystander.
    base = torch.eye(8)
    centers = [base[0], 0.9 * base[0] + 0.1 * base[1],
               base[4], base[6]]
    feats, labels = _clustered_features(centers, spread=0.05)
    v = candidate_indices(feats, labels, [0, 1, 2, 3])
    assert v["attribution_ratio"] > 5.0
    assert v["confusion_rate"] > 0.05
    # Every confusion involves exactly one fork class (0 vs bystander 1).
    assert math.isclose(v["fork_confusion_rate"], v["confusion_rate"],
                        rel_tol=1e-9)


def test_candidate_indices_deterministic():
    eye = torch.eye(8)
    feats, labels = _clustered_features(
        [eye[0], 0.9 * eye[0] + 0.1 * eye[1], eye[4], eye[6]])
    a = candidate_indices(feats, labels, [0, 1, 2, 3])
    b = candidate_indices(feats, labels, [0, 1, 2, 3])
    assert a == b


def _values(by_pair):
    # evaluate() consumes {pair: {candidate: value}}; fill all three
    # candidates with the same per-pair value unless overridden.
    return {p: {c: val for c in CANDIDATES} for p, val in by_pair.items()}


def test_promotion_requires_d_worst_and_ca_benign():
    good = _values({"pairD": 5.0, "pairE": 3.0, "pairB": 2.0,
                    "pairC": 1.0, "pairA": 0.5})
    ev = evaluate(good)
    assert ev["admissible"] == list(CANDIDATES)
    assert ev["promoted"] is not None
    assert ev["stop_recommended"] is False
    # D worst but A escapes the benign pair -> inadmissible.
    bad_benign = _values({"pairD": 5.0, "pairA": 3.0, "pairB": 2.0,
                          "pairC": 1.0, "pairE": 0.5})
    ev = evaluate(bad_benign)
    assert ev["admissible"] == []
    assert ev["promoted"] is None and ev["stop_recommended"] is True
    # C/A benign but E outranks D -> inadmissible.
    bad_d = _values({"pairE": 5.0, "pairD": 3.0, "pairB": 2.0,
                     "pairC": 1.0, "pairA": 0.5})
    ev = evaluate(bad_d)
    assert ev["admissible"] == []
    assert ev["stop_recommended"] is True


def test_promotion_picks_largest_d_margin():
    vals = _values({"pairD": 4.0, "pairB": 2.0, "pairE": 1.5,
                    "pairC": 1.0, "pairA": 0.9})
    # Give one candidate a wider D margin than the others.
    for p, v in {"pairD": 9.0, "pairB": 2.0, "pairE": 1.5,
                 "pairC": 1.0, "pairA": 0.9}.items():
        vals[p]["confusion_rate"] = v
    ev = evaluate(vals)
    assert ev["promoted"] == "confusion_rate"
    assert ev["candidates"]["confusion_rate"]["d_to_second_ratio"] == 4.5
