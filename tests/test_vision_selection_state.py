"""Read-only selection-state exposure on `VisionDriftStream` (A2b scaffolding).

The A2b paired-path arm (`ReembedDriftStream`, not built yet) must replay the
*exact same* cache rows `VisionDriftStream` selected — same train/held/attractor
rows and the same per-sample attractor assignment — without recomputing RNG,
class selection, or assignment. This pins the only thing that exposure must
guarantee: `selection_state` reports the internally selected rows, is stable
across calls, and hands back clones so a consumer cannot mutate the stream.

Uses a tiny synthetic `.pt` cache (this is selection-state plumbing, not
ImageNet-R endpoint recovery), so it runs without the host-only feature cache.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))

from probe_contraction import VisionDriftStream, VisionSelectionState  # noqa: E402

CATEGORIES = [0, 1, 2]
ATTRACTOR = 9
SAMPLES_PER_CLASS = 4
HELD_PER_CLASS = 3
SEED = 1234
D = 8
PER_CLASS_CACHE = 10  # > samples+held; gives selection something to choose among


def _write_synthetic_cache(path: Path) -> torch.Tensor:
    """Write a {'embeds','labels'} cache with deterministic, per-row-unique rows.

    Row r gets feature value r+1 broadcast across D, so `embeds[i]` is uniquely
    identifiable — letting the test prove indices point at the selected rows.
    Returns the raw (un-normalized) embeds for independent verification.
    """
    classes = CATEGORIES + [ATTRACTOR]
    n = len(classes) * PER_CLASS_CACHE
    embeds = torch.empty(n, D)
    labels = torch.empty(n, dtype=torch.long)
    for r in range(n):
        embeds[r] = float(r + 1)
    for ci, cat in enumerate(classes):
        labels[ci * PER_CLASS_CACHE:(ci + 1) * PER_CLASS_CACHE] = cat
    torch.save({"embeds": embeds, "labels": labels}, path)
    return embeds


def _build_stream(tmp_path: Path) -> tuple[VisionDriftStream, torch.Tensor]:
    cache = tmp_path / "synthetic_cache.pt"
    raw_embeds = _write_synthetic_cache(cache)
    stream = VisionDriftStream(
        categories=CATEGORIES, attractor_category=ATTRACTOR,
        samples_per_class=SAMPLES_PER_CLASS, held_out_per_class=HELD_PER_CLASS,
        seed=SEED, cache_path=str(cache))
    return stream, raw_embeds


def test_selection_state_exposure(tmp_path):
    stream, raw_embeds = _build_stream(tmp_path)
    norm_embeds = F.normalize(raw_embeds, dim=-1)
    st = stream.selection_state

    assert isinstance(st, VisionSelectionState)

    # 1. Exposed indices equal the internally selected indices.
    assert torch.equal(st.train_indices, stream._train_indices)
    assert torch.equal(st.held_indices, stream._held_indices)
    assert torch.equal(st.attractor_indices, stream._attractor_indices)
    assert torch.equal(st.train_assign, stream._train_assign)
    assert torch.equal(st.held_assign, stream._held_assign)

    # ...and those indices genuinely address the selected feature rows, in the
    # same order as the blended pools (the property's whole reason to exist).
    assert torch.equal(norm_embeds[st.train_indices], stream.train_X)
    assert torch.equal(norm_embeds[st.held_indices], stream.held_X)
    assert torch.equal(norm_embeds[st.attractor_indices], stream.attractor_feats)
    # Assignments index into the attractor pool.
    assert int(st.train_assign.max()) < st.attractor_indices.numel()
    assert int(st.held_assign.max()) < st.attractor_indices.numel()

    # 2. Repeated access is stable (equal values each time).
    st2 = stream.selection_state
    for f in ("train_indices", "held_indices", "attractor_indices",
              "train_assign", "held_assign"):
        assert torch.equal(getattr(st, f), getattr(st2, f))

    # 3. Mutating returned tensors does not mutate the stream (clone, not alias).
    snap_train = stream._train_indices.clone()
    snap_held = stream._held_indices.clone()
    snap_aidx = stream._attractor_indices.clone()
    snap_tassign = stream._train_assign.clone()
    snap_hassign = stream._held_assign.clone()
    st.train_indices[0] = 9999
    st.held_indices[0] = 9999
    st.attractor_indices[0] = 9999
    st.train_assign[0] = 9999
    st.held_assign[0] = 9999
    assert torch.equal(stream._train_indices, snap_train)
    assert torch.equal(stream._held_indices, snap_held)
    assert torch.equal(stream._attractor_indices, snap_aidx)
    assert torch.equal(stream._train_assign, snap_tassign)
    assert torch.equal(stream._held_assign, snap_hassign)
    # A fresh snapshot is unaffected by the mutation of the prior one.
    st3 = stream.selection_state
    assert torch.equal(st3.train_indices, snap_train)
    assert torch.equal(st3.train_assign, snap_tassign)

    # 4. The snapshot itself is frozen (dataclass(frozen=True)) — cannot rebind.
    try:
        st.train_indices = torch.zeros(1)
        raise AssertionError("expected frozen dataclass to reject field rebind")
    except Exception as e:  # FrozenInstanceError
        assert "Frozen" in type(e).__name__ or "frozen" in str(e).lower()
