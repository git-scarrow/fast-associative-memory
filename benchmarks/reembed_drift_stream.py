"""A2b · ``ReembedDriftStream`` — input-level (pixel cross-fade) re-embedding arm.

Pilot scaffolding only. This stream is the paired-path counterpart to
``VisionDriftStream``: it replays the **same** endpoint pairs and the **same**
per-sample attractor assignment, but interpolates in *pixel* space and
re-embeds through the model, instead of linearly blending cached features. The
only thing that differs from the cache-linear arm is the interpolation space
(pixels vs embeddings) — see
``results/issue_input_reembed_fidelity/REEMBED_DRIFT_STREAM_DESIGN.md``.

Design constraints honoured here:

* **Consumes, never re-derives, the selection.** Construction reads
  ``vision_stream.selection_state`` (cloned ``train``/``held``/``attractor``
  rows + the two assignment tensors) and the vision stream's labels. It takes
  no class list, seed, or split logic, so it is structurally incapable of
  recomputing RNG / class selection / assignment.
* **Raw model output to the engine.** ``batch(c)`` returns the raw DINOv2
  output (no unit-normalisation); unit norms appear only inside the cosine /
  path-divergence diagnostics below.
* **resize → crop → ToTensor → blend in [0,1] → Normalize → model.** The pixel
  cross-fade happens on the ``[0,1]`` cropped tensors, *before* ``Normalize``,
  which preserves byte-identical endpoint geometry at ``c=0`` / ``c=1``.

The heavy pieces (torchvision ``ImageFolder``, the live DINOv2 model) are
**injected**, never imported at module load, so the gate logic is unit-testable
with a fake model over a synthetic cache. The real wiring lives in
``ReembedDriftStream.from_extractor``.

This module does **not** touch ``forward()``, ``associative_core.py``, the
detector, retrieval, or ``VisionDriftStream``. It runs no calibration / subset
experiment; it exists to validate the pilot gates (see ``probe_reembed_pilot``).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Pilot gate thresholds (see REEMBED_DRIFT_STREAM_DESIGN.md §4/§6).
G1_COS_THRESHOLD = 0.999   # endpoint identity: cos(reembed@c, cached endpoint) must exceed this
G2_MIN_DIVERGENCE = 1e-3   # mid-path divergence below this is "trivial" -> INCONCLUSIVE


def split_transform(compose):
    """Split a ``Resize→CenterCrop→ToTensor→Normalize`` Compose into two stages.

    Returns ``(geometry, normalize)`` where ``geometry`` maps a PIL image to a
    cropped ``[0,1]`` CHW tensor (everything up to and including ``ToTensor``)
    and ``normalize`` applies the trailing ``Normalize``. The split point is the
    single ``Normalize`` in the pipeline; this is what lets the pixel cross-fade
    happen in ``[0,1]`` space before mean/std standardisation.
    """
    import torchvision.transforms as T

    ts = list(compose.transforms)
    norm_positions = [i for i, t in enumerate(ts) if isinstance(t, T.Normalize)]
    if len(norm_positions) != 1:
        raise ValueError(
            "expected exactly one torchvision Normalize in the transform "
            f"(the [0,1]->standardised split point), found {len(norm_positions)}")
    i = norm_positions[0]
    return T.Compose(ts[:i]), T.Compose(ts[i:])


class ReembedDriftStream:
    """Drop-in for ``VisionDriftStream.batch`` that re-embeds pixel cross-fades.

    Construct via :meth:`from_extractor` for the real ImageNet-R / DINOv2 wiring,
    or directly (with injected ``image_for_row`` / ``geom_transform`` /
    ``normalize`` / ``model``) for tests.
    """

    def __init__(self, vision_stream, *, image_for_row, geom_transform,
                 normalize, model, device="cpu"):
        # --- Consume the vision stream's selection verbatim (clones, no recompute) ---
        st = vision_stream.selection_state
        self.train_indices = st.train_indices.clone()
        self.held_indices = st.held_indices.clone()
        self.attractor_indices = st.attractor_indices.clone()
        self.train_assign = st.train_assign.clone()
        self.held_assign = st.held_assign.clone()
        # Shape/label parity straight off the vision stream — not re-derived.
        self.train_lab = vision_stream.train_lab.clone()
        self.held_lab = vision_stream.held_lab.clone()
        self.num_classes = vision_stream.num_classes

        self._image_for_row = image_for_row
        self._geom = geom_transform
        self._normalize = normalize
        self._model = model
        self._device = device
        self._geom_cache: dict[int, torch.Tensor] = {}

    # -- real wiring ---------------------------------------------------------
    @classmethod
    def from_extractor(cls, vision_stream, extractor, model, *,
                       device="cpu", split="train"):
        """Wire ``image_for_row`` / transforms from the cache's own extractor.

        Uses the extractor module that built the cache as the single source of
        truth for the dataset root, stratified split, loader, and transform —
        nothing is reimplemented from memory (mirrors the merged G0/G1 probe).
        """
        from torchvision.datasets import ImageFolder

        full_ds = ImageFolder(root=extractor.DATASET_ROOT, transform=None)
        train_sub, test_sub = extractor.stratified_split(
            full_ds, extractor.TRAIN_RATIO, extractor.SEED)
        sub = train_sub if split == "train" else test_sub
        split_indices = sub.indices            # cache row i -> full_ds[split_indices[i]]
        loader = full_ds.loader
        samples = full_ds.samples
        geom, normalize = split_transform(extractor.build_transform())

        # Force the real model into the expected inference state. Only here, in
        # the real-wiring path — direct __init__ stays untouched so tests can
        # inject fake (non-Module) models without an .eval()/.to() surface.
        model = model.to(device).eval()

        def image_for_row(row: int):
            ds_idx = split_indices[int(row)]
            path = samples[ds_idx][0]
            return loader(path)

        return cls(vision_stream, image_for_row=image_for_row,
                   geom_transform=geom, normalize=normalize,
                   model=model, device=device)

    # -- per-step behaviour --------------------------------------------------
    def _geom_for_row(self, row: int) -> torch.Tensor:
        """Cropped ``[0,1]`` CHW tensor for a cache row (cached; deterministic)."""
        r = int(row)
        if r not in self._geom_cache:
            self._geom_cache[r] = self._geom(self._image_for_row(r))
        return self._geom_cache[r]

    def _embed_blend(self, sample_rows: torch.Tensor, assign: torch.Tensor,
                     contraction: float) -> torch.Tensor:
        """Raw embeddings of the pixel cross-fade for one pool at contraction c.

        For sample row ``s`` assigned to attractor pool slot ``a`` (-> attractor
        cache row ``attractor_indices[a]``):
        ``img(c) = (1-c)·geom(s) + c·geom(attractor)`` in ``[0,1]``, then
        ``Normalize``, then ``model`` — returned raw (no unit normalisation).
        """
        c = float(contraction)
        imgs = []
        for s, a in zip(sample_rows.tolist(), assign.tolist()):
            arow = int(self.attractor_indices[a])
            gx = self._geom_for_row(s)
            ga = self._geom_for_row(arow)
            blended = (1.0 - c) * gx + c * ga      # pixel cross-fade in [0,1]
            imgs.append(self._normalize(blended))
        batch = torch.stack(imgs).to(self._device)
        with torch.no_grad():
            emb = self._model(batch)
        return emb.detach().cpu().float()          # RAW output to the engine

    def batch(self, contraction: float):
        """``VisionDriftStream``-shaped yield: ``(train_q, train_y, train_lab), (held_q, held_lab)``.

        ``train_q`` / ``held_q`` are RAW re-embeddings of the pixel cross-fade.
        """
        train_q = self._embed_blend(self.train_indices, self.train_assign, contraction)
        held_q = self._embed_blend(self.held_indices, self.held_assign, contraction)
        train_y = F.one_hot(self.train_lab, num_classes=self.num_classes).float()
        return (train_q, train_y, self.train_lab), (held_q, self.held_lab)


# --------------------------------------------------------------------------
# Pilot gates (pure diagnostics; consume both arms, mutate nothing)
# --------------------------------------------------------------------------
def assignment_parity(reembed: "ReembedDriftStream", vision) -> bool:
    """True iff the reembed arm replays the vision arm's exact selection."""
    st = vision.selection_state
    return bool(
        torch.equal(reembed.train_indices, st.train_indices)
        and torch.equal(reembed.held_indices, st.held_indices)
        and torch.equal(reembed.attractor_indices, st.attractor_indices)
        and torch.equal(reembed.train_assign, st.train_assign)
        and torch.equal(reembed.held_assign, st.held_assign))


def g1_endpoint_min_cosine(reembed: "ReembedDriftStream", vision) -> float:
    """Min cosine of reembed endpoints vs the cached endpoints the vision arm blends.

    ``c=0`` -> sample endpoint (cached ``train_X`` / ``held_X``);
    ``c=1`` -> attractor endpoint (cached ``attractor_feats[assign]``).
    Cosine is norm-invariant; diagnostics normalize both sides before comparison.
    """
    st = vision.selection_state
    (q0, _, _), (h0, _) = reembed.batch(0.0)
    (q1, _, _), (h1, _) = reembed.batch(1.0)

    def cos(a, b):
        return F.cosine_similarity(F.normalize(a, dim=-1),
                                   F.normalize(b, dim=-1), dim=-1)

    cosines = torch.cat([
        cos(q0, vision.train_X),
        cos(h0, vision.held_X),
        cos(q1, vision.attractor_feats[st.train_assign]),
        cos(h1, vision.attractor_feats[st.held_assign]),
    ])
    return float(cosines.min())


def g2_path_divergence(reembed: "ReembedDriftStream", vision,
                       contractions=(0.25, 0.5, 0.75)) -> list[float]:
    """Per-c max ``1 - cos(unit(q_linear), unit(q_reembed))`` over both pools.

    Measures how far the input-level path departs from the cache-linear path at
    each mid-contraction. Endpoints are shared (G1), so only ``0<c<1`` is probed.
    """
    divs = []
    for c in contractions:
        (qr, _, _), (hr, _) = reembed.batch(c)
        (ql, _, _), (hl, _) = vision.batch(c)
        d_tr = 1.0 - F.cosine_similarity(
            F.normalize(qr, dim=-1), F.normalize(ql, dim=-1), dim=-1)
        d_ho = 1.0 - F.cosine_similarity(
            F.normalize(hr, dim=-1), F.normalize(hl, dim=-1), dim=-1)
        divs.append(max(float(d_tr.max()), float(d_ho.max())))
    return divs


def pilot_verdict(reembed: "ReembedDriftStream", vision,
                  contractions=(0.25, 0.5, 0.75)) -> dict:
    """Run the three pilot gates and return a verdict dict.

    Answers, in order:
      1. parity preserved?            -> FAIL if not
      2. c=0/c=1 cos > 0.999?         -> FAIL if not
      3. mid-path diverges nontrivially? -> INCONCLUSIVE if not, else PASS
    """
    parity = assignment_parity(reembed, vision)
    if not parity:
        return {"verdict": "FAIL", "gate": "parity", "parity": False,
                "reason": "assignment/endpoint parity broken vs VisionDriftStream; "
                          "the paired comparison is void.",
                "g1_min_cosine": None, "g2_divergence": None}

    min_cos = g1_endpoint_min_cosine(reembed, vision)
    if min_cos <= G1_COS_THRESHOLD:
        return {"verdict": "FAIL", "gate": "g1_endpoint", "parity": True,
                "g1_min_cosine": min_cos, "g2_divergence": None,
                "reason": (f"endpoint cosine {min_cos:.6f} <= {G1_COS_THRESHOLD}; "
                           "re-embed pipeline does not reproduce the cached "
                           "endpoints. Do NOT loosen the threshold — diagnose the "
                           "resize/crop/ToTensor/Normalize ordering.")}

    divs = g2_path_divergence(reembed, vision, contractions)
    max_div = max(divs)
    if max_div <= G2_MIN_DIVERGENCE:
        return {"verdict": "INCONCLUSIVE", "gate": "g2_divergence", "parity": True,
                "g1_min_cosine": min_cos, "g2_divergence": divs,
                "reason": (f"mid-path divergence max {max_div:.3e} <= "
                           f"{G2_MIN_DIVERGENCE:.0e}; the input-level path is "
                           "(near-)collinear with the cache-linear path. Mark "
                           "INCONCLUSIVE and diagnose before any subset run.")}

    return {"verdict": "PASS", "gate": None, "parity": True,
            "g1_min_cosine": min_cos, "g2_divergence": divs,
            "reason": (f"parity holds, endpoints reproduce (min cos {min_cos:.6f} "
                       f"> {G1_COS_THRESHOLD}), and the input-level path diverges "
                       f"nontrivially mid-path (max {max_div:.3e} > "
                       f"{G2_MIN_DIVERGENCE:.0e}).")}
