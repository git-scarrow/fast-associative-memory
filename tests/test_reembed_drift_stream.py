"""Pilot gates for the A2b ``ReembedDriftStream`` (input-level re-embedding arm).

These pin the four questions the pilot must answer, using a fake model over a
tiny synthetic cache (no DINOv2, no ImageNet-R) so they run on any host:

  1. Selection consumption, not recomputation — the stream replays the exact
     tensors ``VisionDriftStream.selection_state`` hands it, even values no
     seeded RNG would produce, and clones them.
  2. Assignment / endpoint parity with ``VisionDriftStream``.
  3. ``c=0`` / ``c=1`` reproduce the cached endpoints at cosine > 0.999, through
     the real ``Resize→CenterCrop→ToTensor→blend in [0,1]→Normalize`` pipeline.
  4. When the input-level path is (near-)collinear with the cache-linear path
     (a linear model over the cross-fade), the verdict is INCONCLUSIVE, not PASS;
     a nonlinear model that genuinely bends the path yields PASS.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))

from probe_contraction import VisionDriftStream, VisionSelectionState  # noqa: E402
from reembed_drift_stream import (  # noqa: E402
    G1_COS_THRESHOLD, ReembedDriftStream, assignment_parity,
    g1_endpoint_min_cosine, g2_path_divergence, pilot_verdict, split_transform)

CATEGORIES = [0, 1]
ATTRACTOR = 2
SAMPLES_PER_CLASS = 2
HELD_PER_CLASS = 1
PER_CLASS_CACHE = 5            # > samples+held; selection has something to choose
SEED = 7
CLASSES = CATEGORIES + [ATTRACTOR]
N_ROWS = len(CLASSES) * PER_CLASS_CACHE


def _labels():
    labels = torch.empty(N_ROWS, dtype=torch.long)
    for ci, cat in enumerate(CLASSES):
        labels[ci * PER_CLASS_CACHE:(ci + 1) * PER_CLASS_CACHE] = cat
    return labels


# --------------------------------------------------------------------------
# Real-pipeline fixture: tiny PIL images + a fixed model define the cache, so
# re-embedding any selected row reproduces its cached vector exactly.
# --------------------------------------------------------------------------
SIZE = 8                      # post-crop spatial size
DIM = 16                      # embedding dim
_REAL_COMPOSE = T.Compose([
    T.Resize(SIZE, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(SIZE),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def _make_pil_images():
    """N_ROWS deterministic small RGB PILs of varied size (exercises resize/crop)."""
    g = torch.Generator().manual_seed(123)
    imgs = []
    for r in range(N_ROWS):
        h = 10 + (r % 3)
        w = 9 + ((r * 2) % 4)
        arr = (torch.rand(h, w, 3, generator=g) * 255).to(torch.uint8).numpy()
        imgs.append(Image.fromarray(arr, mode="RGB"))
    return imgs


def _linear_model(seed=99, out=DIM, in_dim=3 * SIZE * SIZE):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(out, in_dim, generator=g)

    def model(batch):
        return batch.flatten(1) @ W.T

    return model


def _nonlinear_model(seed=99, out=DIM, in_dim=3 * SIZE * SIZE):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(out, in_dim, generator=g)
    V = torch.randn(out, in_dim, generator=g)

    def model(batch):
        x = batch.flatten(1)
        # Quadratic term makes embed(pixel-blend) != linear blend of embeds,
        # so the input-level path genuinely bends away from the cache-linear path.
        return x @ W.T + (x * x) @ V.T

    return model


def _build_cache(tmp_path, model, geom, normalize, images):
    """Save a {'embeds','labels'} cache whose row r == model(pipeline(images[r]))."""
    embeds = torch.empty(N_ROWS, DIM)
    with torch.no_grad():
        for r in range(N_ROWS):
            x = normalize(geom(images[r])).unsqueeze(0)
            embeds[r] = model(x).squeeze(0)
    cache = tmp_path / "synthetic_cache.pt"
    torch.save({"embeds": embeds, "labels": _labels()}, cache)
    return cache


def _paired(tmp_path, model, geom, normalize, images):
    cache = _build_cache(tmp_path, model, geom, normalize, images)
    vision = VisionDriftStream(
        categories=CATEGORIES, attractor_category=ATTRACTOR,
        samples_per_class=SAMPLES_PER_CLASS, held_out_per_class=HELD_PER_CLASS,
        seed=SEED, cache_path=str(cache))
    reembed = ReembedDriftStream(
        vision, image_for_row=lambda r: images[r],
        geom_transform=geom, normalize=normalize, model=model)
    return vision, reembed


# --------------------------------------------------------------------------
# 1. Consumption, not recomputation
# --------------------------------------------------------------------------
def test_consumes_selection_state_verbatim_and_clones():
    # A fake vision stream exposing an *impossible* selection (non-contiguous,
    # unsorted rows no seeded randperm over this cache would yield in this order).
    train_idx = torch.tensor([97, 3, 55])
    held_idx = torch.tensor([12, 800])
    attr_idx = torch.tensor([41, 5, 63])
    train_assign = torch.tensor([2, 0, 1])
    held_assign = torch.tensor([1, 2])
    st = VisionSelectionState(
        train_indices=train_idx, held_indices=held_idx, attractor_indices=attr_idx,
        train_assign=train_assign, held_assign=held_assign)
    fake = SimpleNamespace(
        selection_state=st,
        train_lab=torch.tensor([0, 1, 0]), held_lab=torch.tensor([0, 1]),
        num_classes=2)

    # Construct using ONLY selection_state + labels (no seed / categories / cache).
    reembed = ReembedDriftStream(
        fake, image_for_row=lambda r: None,
        geom_transform=lambda x: x, normalize=lambda x: x, model=lambda b: b)

    assert torch.equal(reembed.train_indices, train_idx)
    assert torch.equal(reembed.held_indices, held_idx)
    assert torch.equal(reembed.attractor_indices, attr_idx)
    assert torch.equal(reembed.train_assign, train_assign)
    assert torch.equal(reembed.held_assign, held_assign)

    # Clones, not aliases: mutating the stream must not touch the source tensors.
    reembed.train_indices[0] = -1
    reembed.train_assign[0] = -1
    assert int(train_idx[0]) == 97
    assert int(train_assign[0]) == 2


# --------------------------------------------------------------------------
# 2. Assignment / endpoint parity
# --------------------------------------------------------------------------
def test_assignment_parity(tmp_path):
    images = _make_pil_images()
    geom, normalize = split_transform(_REAL_COMPOSE)
    vision, reembed = _paired(tmp_path, _linear_model(), geom, normalize, images)

    assert assignment_parity(reembed, vision)
    # ...and it really is the vision arm's selection.
    st = vision.selection_state
    assert torch.equal(reembed.train_indices, st.train_indices)
    assert torch.equal(reembed.train_assign, st.train_assign)
    assert torch.equal(reembed.attractor_indices, st.attractor_indices)

    # A tampered assignment must be detected as a parity break.
    reembed.train_assign[0] = (int(reembed.train_assign[0]) + 1) % reembed.attractor_indices.numel()
    assert not assignment_parity(reembed, vision)


# --------------------------------------------------------------------------
# 3. c=0 / c=1 endpoint identity through the real pixel pipeline
# --------------------------------------------------------------------------
def test_endpoint_cosine_through_real_pipeline(tmp_path):
    images = _make_pil_images()
    geom, normalize = split_transform(_REAL_COMPOSE)
    vision, reembed = _paired(tmp_path, _linear_model(), geom, normalize, images)

    min_cos = g1_endpoint_min_cosine(reembed, vision)
    assert min_cos > G1_COS_THRESHOLD, f"endpoint cosine {min_cos} <= {G1_COS_THRESHOLD}"

    # c=0 reproduces the sample endpoint; c=1 the assigned attractor endpoint.
    st = vision.selection_state
    (q0, _, _), _ = reembed.batch(0.0)
    (q1, _, _), _ = reembed.batch(1.0)
    assert torch.allclose(F.normalize(q0, dim=-1),
                          F.normalize(vision.train_X, dim=-1), atol=1e-5)
    assert torch.allclose(F.normalize(q1, dim=-1),
                          F.normalize(vision.attractor_feats[st.train_assign], dim=-1),
                          atol=1e-5)


def test_split_transform_orders_blend_before_normalize():
    geom, normalize = split_transform(_REAL_COMPOSE)
    img = _make_pil_images()[0]
    g = geom(img)
    # Geometry stage ends at ToTensor -> a [0,1] CHW float tensor.
    assert g.shape == (3, SIZE, SIZE)
    assert g.min() >= 0.0 and g.max() <= 1.0
    # The single Normalize lives in the normalize stage and shifts out of [0,1].
    n = normalize(g)
    assert n.min() < 0.0


# --------------------------------------------------------------------------
# 4. Verdict logic: trivial divergence -> INCONCLUSIVE; bent path -> PASS
# --------------------------------------------------------------------------
def _identity_unit_fixture(tmp_path):
    """Identity transforms + identity model + unit-norm endpoint 'pixels'.

    A linear (identity) model maps the [0,1] convex cross-fade to a convex combo
    in embedding space; equal-norm endpoints then make the input-level path
    collinear with the cache-linear path -> trivial divergence.
    """
    g = torch.Generator().manual_seed(321)
    vecs = [F.normalize(torch.randn(DIM, generator=g), dim=-1) for _ in range(N_ROWS)]
    embeds = torch.stack(vecs)
    cache = tmp_path / "identity_cache.pt"
    torch.save({"embeds": embeds, "labels": _labels()}, cache)
    vision = VisionDriftStream(
        categories=CATEGORIES, attractor_category=ATTRACTOR,
        samples_per_class=SAMPLES_PER_CLASS, held_out_per_class=HELD_PER_CLASS,
        seed=SEED, cache_path=str(cache))
    reembed = ReembedDriftStream(
        vision, image_for_row=lambda r: vecs[r],
        geom_transform=lambda x: x, normalize=lambda x: x, model=lambda b: b)
    return vision, reembed


def test_trivial_divergence_is_inconclusive_not_pass(tmp_path):
    vision, reembed = _identity_unit_fixture(tmp_path)

    # Parity and endpoints are fine; only the path is degenerate.
    assert assignment_parity(reembed, vision)
    assert g1_endpoint_min_cosine(reembed, vision) > G1_COS_THRESHOLD
    divs = g2_path_divergence(reembed, vision)
    assert max(divs) <= 1e-3, f"expected ~collinear path, got divergence {divs}"

    verdict = pilot_verdict(reembed, vision)
    assert verdict["verdict"] == "INCONCLUSIVE"
    assert verdict["gate"] == "g2_divergence"


def test_nonlinear_path_passes(tmp_path):
    images = _make_pil_images()
    geom, normalize = split_transform(_REAL_COMPOSE)
    vision, reembed = _paired(tmp_path, _nonlinear_model(), geom, normalize, images)

    # Endpoints still reproduce (cache == model(pipeline(image))), parity holds,
    # and the quadratic model bends the mid-path away from the linear arm.
    assert assignment_parity(reembed, vision)
    assert g1_endpoint_min_cosine(reembed, vision) > G1_COS_THRESHOLD
    divs = g2_path_divergence(reembed, vision)
    assert max(divs) > 1e-3, f"nonlinear path should diverge, got {divs}"

    verdict = pilot_verdict(reembed, vision)
    assert verdict["verdict"] == "PASS"
    assert verdict["gate"] is None


# --------------------------------------------------------------------------
# 5. Verdict logic: parity break -> FAIL
# --------------------------------------------------------------------------
def test_parity_break_fails(tmp_path):
    images = _make_pil_images()
    geom, normalize = split_transform(_REAL_COMPOSE)
    vision, reembed = _paired(tmp_path, _linear_model(), geom, normalize, images)

    # Sanity: starts as a valid stream.
    assert assignment_parity(reembed, vision)

    # Tamper: shift every train assignment by 1 mod the attractor pool size.
    n_attr = reembed.attractor_indices.numel()
    reembed.train_assign = (reembed.train_assign + 1) % n_attr

    assert not assignment_parity(reembed, vision)

    verdict = pilot_verdict(reembed, vision)
    assert verdict["verdict"] == "FAIL"
    assert verdict["gate"] == "assignment_parity"
