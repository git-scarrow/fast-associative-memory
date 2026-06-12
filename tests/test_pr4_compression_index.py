"""Hermetic pins for the PR-4 gate-1 compression index (no cache needed).

Pins the index math on synthetic geometry so the gentoo cache run is the
only non-hermetic step: pair_cos is the full cross-class mean cosine,
set_index averages the 6 within-set pairs, and a deliberately compressed
synthetic set scores above a separated one (the §4 ordering semantics).
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.pr4_compression_index import pair_cos, set_index


def _toy(vecs_per_class):
    """Build (embeds, labels) from {class: [vectors]}; rows unit-normalized."""
    embeds, labels = [], []
    for cls, vecs in vecs_per_class.items():
        for v in vecs:
            embeds.append(v)
            labels.append(cls)
    return F.normalize(torch.tensor(embeds, dtype=torch.float32), dim=-1), \
        torch.tensor(labels, dtype=torch.long)


def test_pair_cos_identical_and_orthogonal():
    embeds, labels = _toy({
        0: [[1.0, 0.0], [1.0, 0.0]],
        1: [[1.0, 0.0]],
        2: [[0.0, 1.0]],
    })
    assert abs(pair_cos(embeds, labels, 0, 1) - 1.0) < 1e-6
    assert abs(pair_cos(embeds, labels, 0, 2)) < 1e-6


def test_set_index_orders_compressed_above_separated():
    # Separated set: 4 near-orthogonal directions. Compressed set: 4 small
    # perturbations of one direction. Same attractor for both.
    e = torch.eye(8)
    base = e[4]
    sep = {i: [e[i].tolist()] for i in range(4)}
    comp = {i + 10: [(base + 0.05 * e[i]).tolist()] for i in range(4)}
    attractor = {99: [e[5].tolist()]}
    embeds, labels = _toy({**sep, **comp, **attractor})

    idx_sep = set_index(embeds, labels, [0, 1, 2, 3], 99)
    idx_comp = set_index(embeds, labels, [10, 11, 12, 13], 99)
    assert idx_comp["index"] > idx_sep["index"]
    assert len(idx_sep["within_set_pair_cos"]) == 6
    # stale-fork pair is (classes[0], classes[-1]) by construction
    assert idx_sep["stale_fork_pair_cos"] == \
        idx_sep["within_set_pair_cos"]["0,3"]
