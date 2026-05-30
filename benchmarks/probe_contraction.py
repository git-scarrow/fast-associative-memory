#!/usr/bin/env python3
"""Driver for the rho(t) cross-class contraction probe.

Runs a continual-learning loop on a *deliberately contracting* manifold and
collects the two telemetry curves added in PR #69:

  1. In-band per-epoch cross-class similarity (cheap; ContinuousCAM.get_stats()).
  2. Held-out, true-label probe (authoritative; probe_cross_class_similarity()).

The manifold is contracted with a coefficient that grows linearly across epochs.
This makes the inter-class cosine rho(t) rise, so we can observe the two failure
onsets VIGIL predicted on a frozen-manifold-assumption engine:

  * Retrieval-blend onset: off-class softmax vote mass crosses `blend_eps`,
    analytically near rho ~= Delta - inference_temp * ln((1-eps)/eps)
    (~0.79 for Delta~0.9, eps=0.10, temp=0.05).
  * Structural-chimera onset: rho crosses the dynamic-vigilance ceiling
    (v_ceiling, default 0.95), after which boundary writes are admitted and EMA
    bakes them into the prototype keys.

Two stream sources share the SAME loop, CSV columns, and onset report:

  * Synthetic (default, the controlled baseline): class centers are random
    gaussian directions; the manifold is contracted by interpolating every
    center toward a shared attractor *in vector space*, with gaussian
    within-class noise.
  * Real (--real, the transfer test for issue #71): real text from K
    20-Newsgroups topics embedded with a production model (all-MiniLM-L6-v2).
    Drift is produced at the TEXT level -- at contraction c, a fraction c of
    each document's sentences is swapped for sentences from a shared attractor
    pool, then the (real) model re-embeds. The vector trajectory is whatever the
    model emits, NOT a linear interpolation, so this tests whether the synthetic
    onset thresholds survive real anisotropic embedding geometry and real
    intra-class spread.

Memory PERSISTS across epochs (the realistic continual setting where EMA
contamination accumulates); only the per-epoch stats are reset.

Examples (run from repo root):
    # synthetic baseline
    python benchmarks/probe_contraction.py --epochs 30 --classes 8 --dim 64 \
        --contraction-end 0.97 --csv contraction.csv --plot
    # real-embedding transfer test
    python benchmarks/probe_contraction.py --real --epochs 30 \
        --contraction-end 0.9 --csv contraction_real.csv --plot
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from associative_core import ContinuousCAM  # noqa: E402
from dynamic_vigilance import (  # noqa: E402
    DynamicVigilance, RelativeVigilance, RetrievalFloorPolicy,
    RetrievalTruncationPolicy)


def make_epoch_batch(centers: torch.Tensor, attractor: torch.Tensor,
                     contraction: float, n_per_class: int, noise: float,
                     generator: torch.Generator):
    """Generate one epoch's (queries, one-hot targets, true labels).

    `contraction` in [0, 1) mixes each class center toward `attractor`; higher
    values pull all classes together, raising inter-class similarity.
    """
    C, D = centers.shape
    mixed = F.normalize((1.0 - contraction) * centers + contraction * attractor,
                        dim=-1)                                  # (C, D)
    labels = torch.arange(C).repeat_interleave(n_per_class)      # (C*n,)
    base = mixed[labels]                                         # (C*n, D)
    eps = torch.randn(base.shape, generator=generator) * noise
    queries = F.normalize(base + eps, dim=-1)                   # (C*n, D)
    targets = F.one_hot(labels, num_classes=C).float()          # (C*n, C)
    return queries, targets, labels


# --------------------------------------------------------------------------- #
# Real drifting embedding stream (issue #71 transfer test)
# --------------------------------------------------------------------------- #
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _to_sentences(doc: str, max_sentences: int) -> list[str]:
    """Split a raw document into a bounded list of non-trivial sentences."""
    parts = [s.strip() for s in _SENT_SPLIT.split(doc)]
    parts = [s for s in parts if len(s) >= 12]   # drop fragments/quote scars
    return parts[:max_sentences]


class RealDriftStream:
    """Epoch-ordered real-text stream whose inter-class similarity rises.

    K well-separated 20-Newsgroups topics are the classes; one further topic is
    the shared attractor. Each document is pre-split into sentences. At
    contraction `c`, ~c of every document's sentences are replaced with sentences
    drawn (deterministically) from the shared attractor pool, and the production
    model re-embeds the blended text. As c -> 1 every document collapses onto the
    attractor, so cross-class cosine rho(t) rises emergently from the real model.

    The same documents are reused every epoch (only the blend fraction changes),
    and train vs held-out documents are disjoint pools per class -- matching the
    synthetic driver's continual setting.
    """

    def __init__(self, categories: list[str], attractor_category: str,
                 samples_per_class: int, held_out_per_class: int,
                 model_name: str, max_sentences: int, seed: int,
                 batch_size: int = 256):
        from sklearn.datasets import fetch_20newsgroups
        from sentence_transformers import SentenceTransformer

        self.categories = categories
        self.num_classes = len(categories)
        self.batch_size = batch_size
        self._encode_kwargs = dict(convert_to_numpy=True,
                                   normalize_embeddings=False,
                                   show_progress_bar=False)
        rng = torch.Generator().manual_seed(seed)

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

        need = samples_per_class + held_out_per_class
        wanted = list(categories) + [attractor_category]
        raw = fetch_20newsgroups(subset="train", categories=wanted,
                                 remove=("headers", "footers", "quotes"),
                                 shuffle=True, random_state=seed)
        target_names = raw.target_names

        def docs_for(cat: str) -> list[list[str]]:
            ci = target_names.index(cat)
            out = []
            for doc, t in zip(raw.data, raw.target):
                if t != ci:
                    continue
                sents = _to_sentences(doc, max_sentences)
                if sents:
                    out.append(sents)
            return out

        # Attractor sentence pool (flattened, shared across all classes).
        self.attractor_sents = [s for d in docs_for(attractor_category) for s in d]
        if len(self.attractor_sents) < max_sentences:
            raise RuntimeError(
                f"attractor topic {attractor_category!r} yielded too few sentences")

        self.train_docs: list[list[list[str]]] = []   # [class][doc][sentence]
        self.held_docs: list[list[list[str]]] = []
        self.train_labels: list[int] = []
        self.held_labels: list[int] = []
        for ci, cat in enumerate(categories):
            pool = docs_for(cat)
            if len(pool) < need:
                raise RuntimeError(
                    f"topic {cat!r} has {len(pool)} usable docs, need {need}")
            tr = pool[:samples_per_class]
            ho = pool[samples_per_class:need]
            self.train_docs.append(tr)
            self.held_docs.append(ho)
            self.train_labels += [ci] * len(tr)
            self.held_labels += [ci] * len(ho)

        # Deterministic per-document attractor sentence assignment so a doc's
        # blend grows monotonically with c (the same attractor sentences are
        # progressively swapped in, never reshuffled).
        n_docs = sum(len(c) for c in self.train_docs) + \
                 sum(len(c) for c in self.held_docs)
        self._perm = [
            torch.randperm(len(self.attractor_sents), generator=rng).tolist()
            for _ in range(n_docs)
        ]

    def _blend(self, sents: list[str], contraction: float, doc_id: int) -> str:
        """Replace ~contraction of `sents` with shared attractor sentences."""
        L = len(sents)
        k = int(round(contraction * L))
        if k <= 0:
            return " ".join(sents)
        if k >= L:
            k = L
        order = self._perm[doc_id]
        swap_in = [self.attractor_sents[order[i % len(order)]] for i in range(k)]
        kept = sents[: L - k]
        return " ".join(kept + swap_in)

    def _embed(self, texts: list[str]) -> torch.Tensor:
        import numpy as np
        vecs = self.model.encode(texts, batch_size=self.batch_size,
                                 **self._encode_kwargs)
        return torch.from_numpy(np.asarray(vecs)).float()

    def batch(self, contraction: float):
        """Return (train_q, train_y, train_lab), (held_q, held_lab) at `c`.

        Queries are raw (un-normalized) model outputs; the engine and probe
        normalize internally, exactly as for the synthetic path.
        """
        doc_id = 0
        train_texts, held_texts = [], []
        for cls in self.train_docs:
            for sents in cls:
                train_texts.append(self._blend(sents, contraction, doc_id))
                doc_id += 1
        for cls in self.held_docs:
            for sents in cls:
                held_texts.append(self._blend(sents, contraction, doc_id))
                doc_id += 1

        train_q = self._embed(train_texts)
        held_q = self._embed(held_texts)
        train_lab = torch.tensor(self.train_labels, dtype=torch.long)
        held_lab = torch.tensor(self.held_labels, dtype=torch.long)
        train_y = F.one_hot(train_lab, num_classes=self.num_classes).float()
        return (train_q, train_y, train_lab), (held_q, held_lab)


# --------------------------------------------------------------------------- #
# Vision drifting feature stream (ViT-L/14 transfer test — A1)
# --------------------------------------------------------------------------- #
# Candidate locations for the CIFAR-100 DINOv2 ViT-L/14 feature cache. The
# repo-root `feature_cache_vitl14` symlink is the canonical entry; it points at a
# Linux /mnt path that resolves only on the compute host (it is a dead link on a
# macOS checkout). We probe a short list so the stream is robust to which copy of
# the cache happens to be mounted, rather than hard-failing on one absolute path.
_VISION_CACHE_CANDIDATES = [
    "feature_cache_vitl14/cifar100_dinov2_train.pt",
    "/mnt/storage/workspace-archive/campus-fabric/feature_cache_vitl14/cifar100_dinov2_train.pt",
    "/mnt/data/dev/projects/campus-fabric/feature_cache_vitl14/cifar100_dinov2_train.pt",
]


def _resolve_vision_cache(path: str | None) -> Path:
    """Return the first existing vision feature cache, preferring an explicit path."""
    tried = []
    candidates = ([path] if path else []) + _VISION_CACHE_CANDIDATES
    for c in candidates:
        tried.append(c)
        p = Path(c)
        if p.exists():
            return p
    raise FileNotFoundError(
        "no ViT-L/14 feature cache found; tried:\n  " + "\n  ".join(tried) +
        "\n(fix or recreate the `feature_cache_vitl14` symlink, or pass "
        "--vision-cache)")


@dataclass(frozen=True)
class VisionSelectionState:
    """Frozen, read-only snapshot of a ``VisionDriftStream``'s selection state.

    Every field is a fresh clone of the stream's internal tensors, so a caller
    (e.g. the A2b ``ReembedDriftStream``, which must replay the *same* cache rows
    back to source images) can consume the exact selection without recomputing
    RNG / class selection / assignment, and without being able to mutate the
    stream. Indices are cache rows (into the on-disk cache's ``embeds``/``labels``)
    in the same order as ``train_X`` / ``held_X`` / ``attractor_feats``.
    """
    train_indices: torch.Tensor       # (n_train,) cache rows behind train_X
    held_indices: torch.Tensor        # (n_held,)  cache rows behind held_X
    attractor_indices: torch.Tensor   # (A,)       cache rows behind attractor_feats
    train_assign: torch.Tensor        # (n_train,) per-train-sample -> attractor pool index
    held_assign: torch.Tensor         # (n_held,)  per-held-sample  -> attractor pool index


class VisionDriftStream:
    """Epoch-ordered DINOv2 ViT-L/14 feature stream whose inter-class similarity rises.

    The vision analogue of ``RealDriftStream`` for the A1 transfer test. K
    well-separated cached classes (default: CIFAR-100 via DINOv2 ViT-L/14, 1024-d)
    are the classes; one further class is the shared attractor. At contraction
    ``c`` every sample's (unit) feature is convex-blended toward a
    *deterministically assigned* real attractor-class feature and renormalized, so
    cross-class cosine rho(t) rises emergently as ``c`` -> 1. The same samples are
    reused every epoch (only the blend fraction changes) and train vs held-out
    pools are disjoint per class — matching ``RealDriftStream``'s continual setting.

    METHODOLOGICAL NOTE vs ``RealDriftStream``. Text drift is produced at the
    INPUT level (swap in attractor sentences, then *re-embed* with the real model),
    so the trajectory is the model's own nonlinear manifold. A precomputed feature
    cache has no cheap re-embed, so drift here is a LINEAR convex blend in feature
    space between two *real* endpoints — the sample and an assigned real attractor
    feature. Both endpoints carry real DINOv2 anisotropy and real intra-/inter-class
    spread; only the path between them is linear. This is the faithful cache-only
    analogue; input-level (pixel/patch) re-embedding is a future faithfulness check.
    The probe, engine, retrieval floor and vote are untouched — only the data
    source changes.
    """

    def __init__(self, categories: list[int], attractor_category: int,
                 samples_per_class: int, held_out_per_class: int, seed: int,
                 cache_path: str | None = None):
        cache = _resolve_vision_cache(cache_path)
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        embeds = blob["embeds"].float()                    # (N, D), raw features
        labels = blob["labels"].long()                     # (N,)
        # Work in unit features so the convex blend is a clean angular interpolation
        # knob; the engine/probe normalize internally, so this only fixes the
        # endpoints' geometry, it does not change retrieval.
        embeds = F.normalize(embeds, dim=-1)
        self.cache_path = str(cache)
        self.dim = int(embeds.shape[1])
        self.num_classes = len(categories)
        self.categories = list(categories)
        rng = torch.Generator().manual_seed(seed)

        need = samples_per_class + held_out_per_class
        train_blocks, held_blocks = [], []
        train_idx_blocks, held_idx_blocks = [], []
        self.train_labels: list[int] = []
        self.held_labels: list[int] = []
        for ci, cat in enumerate(categories):
            idx = (labels == cat).nonzero(as_tuple=True)[0]
            if idx.numel() < need:
                raise RuntimeError(
                    f"class {cat} has {idx.numel()} cached features, need {need}")
            perm = idx[torch.randperm(idx.numel(), generator=rng)]
            tr, ho = perm[:samples_per_class], perm[samples_per_class:need]
            train_blocks.append(embeds[tr])
            held_blocks.append(embeds[ho])
            train_idx_blocks.append(tr)
            held_idx_blocks.append(ho)
            self.train_labels += [ci] * samples_per_class
            self.held_labels += [ci] * held_out_per_class

        aidx = (labels == attractor_category).nonzero(as_tuple=True)[0]
        if aidx.numel() == 0:
            raise RuntimeError(f"attractor class {attractor_category} not in cache")
        aidx = aidx[torch.randperm(aidx.numel(), generator=rng)]
        self.attractor_feats = embeds[aidx]                # (A, D), unit
        self.attractor_category = attractor_category
        # Cache-row indices behind each selected pool (into the on-disk cache's
        # `embeds`/`labels`), in the SAME order as train_X / held_X / attractor_feats.
        # Stored only for read-only export via `selection_state`; not used in blend.
        self._train_indices = torch.cat(train_idx_blocks, dim=0)   # (n_train,)
        self._held_indices = torch.cat(held_idx_blocks, dim=0)     # (n_held,)
        self._attractor_indices = aidx                             # (A,)

        self.train_X = torch.cat(train_blocks, dim=0)      # (n_train, D)
        self.held_X = torch.cat(held_blocks, dim=0)        # (n_held, D)
        self.train_lab = torch.tensor(self.train_labels, dtype=torch.long)
        self.held_lab = torch.tensor(self.held_labels, dtype=torch.long)
        # Deterministic per-sample attractor assignment, stable across epochs, so a
        # sample's blend grows monotonically with c toward the SAME real attractor
        # feature (the direct analogue of the text driver's monotone sentence swap).
        A = self.attractor_feats.shape[0]
        self._train_assign = torch.randint(0, A, (self.train_X.shape[0],),
                                           generator=rng)
        self._held_assign = torch.randint(0, A, (self.held_X.shape[0],),
                                          generator=rng)

    def _blend(self, X: torch.Tensor, assign: torch.Tensor,
               contraction: float) -> torch.Tensor:
        a = self.attractor_feats[assign]                   # (n, D)
        mixed = (1.0 - contraction) * X + contraction * a
        return F.normalize(mixed, dim=-1)

    def batch(self, contraction: float):
        """Return (train_q, train_y, train_lab), (held_q, held_lab) at `c`."""
        train_q = self._blend(self.train_X, self._train_assign, contraction)
        held_q = self._blend(self.held_X, self._held_assign, contraction)
        train_y = F.one_hot(self.train_lab, num_classes=self.num_classes).float()
        return (train_q, train_y, self.train_lab), (held_q, self.held_lab)

    @property
    def selection_state(self) -> "VisionSelectionState":
        """Read-only snapshot of which cache rows this stream selected.

        Returns a frozen ``VisionSelectionState`` of **cloned** tensors, so a
        consumer can replay the identical selection (the A2b paired-path arm)
        without recomputing RNG/selection/assignment and without being able to
        mutate this stream's internal state. Purely an accessor over state fixed
        at construction — it does not change any selection or blend behavior.
        """
        return VisionSelectionState(
            train_indices=self._train_indices.clone(),
            held_indices=self._held_indices.clone(),
            attractor_indices=self._attractor_indices.clone(),
            train_assign=self._train_assign.clone(),
            held_assign=self._held_assign.clone(),
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--classes", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--samples-per-class", type=int, default=32)
    ap.add_argument("--held-out-per-class", type=int, default=32)
    ap.add_argument("--noise", type=float, default=0.10,
                    help="per-sample gaussian noise scale around the class center")
    ap.add_argument("--contraction-start", type=float, default=0.0)
    ap.add_argument("--contraction-end", type=float, default=0.97,
                    help="final attractor-mix coefficient (->1 collapses classes)")
    ap.add_argument("--max-entries", type=int, default=4096)
    ap.add_argument("--blend-eps", type=float, default=0.10,
                    help="off-class vote-mass threshold counted as a blend")
    ap.add_argument("--seed", type=int, default=0)
    # --- Vigilance policy ---
    ap.add_argument("--vigilance-policy", choices=["margin", "relative"],
                    default="margin",
                    help="margin=DynamicVigilance (G29 default); "
                         "relative=RelativeVigilance (issue #73 percentile-anchored)")
    ap.add_argument("--v-base", type=float, default=0.92,
                    help="baseline vigilance (margin policy only)")
    ap.add_argument("--alpha", type=float, default=0.30,
                    help="margin-to-vigilance slope (margin policy only)")
    ap.add_argument("--v-floor", type=float, default=0.30)
    ap.add_argument("--v-ceiling", type=float, default=0.95)
    ap.add_argument("--margin-guard", type=float, default=0.05,
                    help="safety margin above ρ-percentile anchor (relative policy)")
    ap.add_argument("--percentile", type=float, default=0.95,
                    help="off-class sim quantile to anchor on (relative policy)")
    # --- Retrieval-side floor policy (issue #74) ---
    ap.add_argument("--retrieval-floor-policy", choices=["static", "live-delta"],
                    default="static",
                    help="static=fixed inference_sim_floor (default 0=off); "
                         "live-delta=RetrievalFloorPolicy anchored to live Δ (#74)")
    ap.add_argument("--k-logit-steps", type=float, default=3.0,
                    help="floor = Δ_ema - k*inference_temp (live-delta policy)")
    ap.add_argument("--floor-ema-beta", type=float, default=0.9,
                    help="EMA smoothing for the live Δ estimate (live-delta policy)")
    # --- Adaptive vote-support truncation (issue #82 intervention) ---
    ap.add_argument("--support-truncation", action="store_true",
                    help="enable adaptive vote-support truncation: drop each "
                         "probe's off-class tail (> window steps below its top-1) "
                         "before the softmax vote (#82). The window is "
                         "self-gating — a co-located/forced neighbourhood has no "
                         "tail to cut. Composes with --retrieval-floor-policy.")
    ap.add_argument("--truncation-window-steps", type=float, default=2.0,
                    help="tail width in inference_temp steps below the per-row "
                         "top-1; candidates below top1 - w*temp are dropped "
                         "(smaller = more aggressive)")
    ap.add_argument("--truncation-gate-steps", type=float, default=0.0,
                    help="optional min top1-top2 gap (in inference_temp steps) to "
                         "truncate a row; 0=no gate (rely on window self-gating, "
                         "recommended). Higher = isolated-winner rows only.")
    ap.add_argument("--csv", type=str, default="contraction.csv")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot-path", type=str, default="contraction.png")
    # --- Real-embedding transfer test (#71) ---
    ap.add_argument("--real", action="store_true",
                    help="drive the loop from real 20NG text + a production "
                         "embedding model instead of the synthetic manifold")
    ap.add_argument("--real-categories", type=str,
                    default="sci.space,rec.sport.hockey,talk.politics.guns,comp.graphics",
                    help="comma-separated 20NG topics used as classes (real mode)")
    ap.add_argument("--attractor-category", type=str, default="sci.med",
                    help="20NG topic whose sentences are the shared attractor "
                         "blended into every class (real mode)")
    ap.add_argument("--embed-model", type=str, default="all-MiniLM-L6-v2",
                    help="sentence-transformers model name (real mode)")
    ap.add_argument("--max-sentences", type=int, default=24,
                    help="cap sentences per document before embedding (real mode)")
    ap.add_argument("--verdict-json", type=str, default=None,
                    help="if set, write the onset report + transfer verdict as JSON")
    ap.add_argument("--ref-blend-rho", type=float, default=0.79,
                    help="synthetic absolute blend-onset rho to test for transfer "
                         "(the ~0.79 figure validated in #69/#70)")
    args = ap.parse_args()

    # blend_eps is a vote-mass fraction and feeds log((1-eps)/eps); boundary
    # values 0 or 1 are degenerate (everything / nothing counts as a blend) and
    # would crash the analytic report. Fail fast rather than after the loop.
    if not (0.0 < args.blend_eps < 1.0):
        ap.error(f"--blend-eps must be in (0, 1), got {args.blend_eps}")
    if not (0.0 <= args.percentile <= 1.0):
        ap.error(f"--percentile must be in [0, 1], got {args.percentile}")
    if args.k_logit_steps < 0:
        ap.error(f"--k-logit-steps must be >= 0, got {args.k_logit_steps}")
    if not (0.0 <= args.floor_ema_beta < 1.0):
        ap.error(f"--floor-ema-beta must be in [0, 1), got {args.floor_ema_beta}")

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    # --- Build the epoch batch source: synthetic manifold or real text stream ---
    stream = None
    if args.real:
        cats = [c.strip() for c in args.real_categories.split(",") if c.strip()]
        if len(cats) < 2:
            ap.error("--real-categories needs at least 2 topics")
        if args.attractor_category in cats:
            ap.error("--attractor-category must differ from --real-categories")
        print(f"[real] loading {args.embed_model} and {len(cats)} 20NG topics "
              f"(attractor={args.attractor_category}) ...")
        stream = RealDriftStream(
            categories=cats, attractor_category=args.attractor_category,
            samples_per_class=args.samples_per_class,
            held_out_per_class=args.held_out_per_class,
            model_name=args.embed_model, max_sentences=args.max_sentences,
            seed=args.seed)
        num_classes, dim = stream.num_classes, stream.dim
        print(f"[real] embedding dim={dim}, classes={cats}")

        def get_batch(contraction: float):
            return stream.batch(contraction)
    else:
        num_classes, dim = args.classes, args.dim
        # Fixed, well-separated class centers and a shared attractor direction.
        centers = F.normalize(torch.randn(num_classes, dim, generator=gen), dim=-1)
        attractor = F.normalize(torch.randn(1, dim, generator=gen), dim=-1)

        def get_batch(contraction: float):
            q, y, lab = make_epoch_batch(centers, attractor, contraction,
                                         args.samples_per_class, args.noise, gen)
            hq, _, hlab = make_epoch_batch(centers, attractor, contraction,
                                           args.held_out_per_class, args.noise, gen)
            return (q, y, lab), (hq, hlab)

    if args.vigilance_policy == "relative":
        dv = RelativeVigilance(v_floor=args.v_floor, v_ceiling=args.v_ceiling,
                               margin_guard=args.margin_guard,
                               percentile=args.percentile)
        print(f"[policy] RelativeVigilance: percentile={args.percentile}, "
              f"margin_guard={args.margin_guard}, "
              f"v_floor={args.v_floor}, v_ceiling={args.v_ceiling}")
    else:
        dv = DynamicVigilance(v_base=args.v_base, alpha=args.alpha,
                              v_floor=args.v_floor, v_ceiling=args.v_ceiling)
        print(f"[policy] DynamicVigilance: v_base={args.v_base}, alpha={args.alpha}, "
              f"v_floor={args.v_floor}, v_ceiling={args.v_ceiling}")

    floor_policy = None
    if args.retrieval_floor_policy == "live-delta":
        floor_policy = RetrievalFloorPolicy(k_logit_steps=args.k_logit_steps,
                                            ema_beta=args.floor_ema_beta)
        print(f"[floor] RetrievalFloorPolicy: k_logit_steps={args.k_logit_steps}, "
              f"ema_beta={args.floor_ema_beta}")
    else:
        print("[floor] static inference_sim_floor (disabled, =0.0)")

    truncation_policy = None
    if args.support_truncation:
        truncation_policy = RetrievalTruncationPolicy(
            window_steps=args.truncation_window_steps,
            gate_steps=args.truncation_gate_steps)
        print(f"[truncation] RetrievalTruncationPolicy: "
              f"window_steps={args.truncation_window_steps}, "
              f"gate_steps={args.truncation_gate_steps}")
    else:
        print("[truncation] adaptive support truncation (disabled)")

    mem = ContinuousCAM(key_dim=dim, value_dim=num_classes,
                        max_entries=args.max_entries, dynamic_vigilance=dv,
                        retrieval_floor_policy=floor_policy,
                        retrieval_truncation_policy=truncation_policy)

    fields = ["epoch", "contraction",
              "rho_inband", "var_inband", "n_inband",
              "rho_probe", "var_probe", "within_probe", "margin_probe",
              "offclass_weight", "frac_blended", "n_vote",
              # Retrieval-confidence + effective-support telemetry (issue #82):
              # decompose late-epoch blend into premature (usable margin still
              # present) vs forced (manifold collapsed). top1_top2_margin and
              # entropy/support are label-free (deployable at inference); the
              # margin-regime split + blend_top1_correct are label-aware (which
              # confidence-weighting can/can't recover).
              "top1_top2_margin", "median_top1_top2_margin", "vote_entropy",
              "effective_support", "median_effective_support", "max_vote_weight",
              "n_surviving_votes", "frac_low_margin", "frac_high_entropy",
              "frac_margin_recoverable", "frac_margin_borderline",
              "frac_margin_forced", "frac_blend_top1_correct",
              "chimera_onset", "blend_onset", "mean_v_eff",
              "floor_delta_ema", "sim_floor_active",
              # Write-side metrics (issue #73): how the vigilance policy shapes
              # the prototype set. Counts are literal (from learn_local's
              # hits/misses + allocation), not occupancy deltas. occupied=slots
              # in use; alloc=misses that got a slot; merged=same-class hits
              # absorbed; dropped=misses dropped at capacity;
              # merge_rate=merged/n_written; acc=held-out top-1 accuracy.
              "occupied", "alloc", "merged", "dropped", "merge_rate", "acc"]
    rows = []

    first_blend = None
    first_chimera = None
    delta_ref = None  # within-class self-similarity at epoch 0 (the analytic Delta)

    for epoch in range(args.epochs):
        # Linear contraction schedule.
        if args.epochs > 1:
            frac = epoch / (args.epochs - 1)
        else:
            frac = 0.0
        contraction = args.contraction_start + frac * (
            args.contraction_end - args.contraction_start)

        mem.reset_dynamic_vigilance_stats()

        # --- Build this epoch's batch from the selected source ---
        (q, y, _), (hq, hlab) = get_batch(contraction)
        n_written = q.size(0)

        # --- Train (continual: memory persists across epochs) ---
        mem.learn_local(q, y)

        # --- Cheap in-band curve (lagged: reflects pre-call memory state) ---
        stats = mem.get_stats()

        # --- Write-side accounting: literal merge/alloc counts from the write ---
        # Read learn_local's per-call accounting, NOT occupancy deltas: the delta
        # proxy under-counts allocations once the table saturates or reuses
        # evicted slots (occupancy stops growing while writes keep allocating).
        occ = int(stats["n_occupied"])
        ws = mem.last_write_stats
        merged = ws["merged"]                    # same-class hits absorbed (EMA)
        alloc = ws["allocated"]                  # misses that obtained a slot
        dropped = ws["dropped"]                  # misses dropped at capacity
        merge_rate = merged / n_written if n_written else float("nan")

        # --- Held-out top-1 accuracy (read-only forward vote) ---
        pred = mem.forward(hq).argmax(dim=-1)
        acc = (pred == hlab.to(pred.device)).float().mean().item()

        # --- Authoritative held-out probe (post-write, true labels) ---
        p = mem.probe_cross_class_similarity(hq, hlab, blend_eps=args.blend_eps)

        if delta_ref is None and not math.isnan(p.get("mean_within_class_sim", float("nan"))):
            delta_ref = p["mean_within_class_sim"]

        if first_blend is None and p.get("blend_onset"):
            first_blend = (epoch, p["mean_cross_class_sim"], p["mean_offclass_weight"])
        if first_chimera is None and p.get("chimera_onset"):
            first_chimera = (epoch, p["mean_cross_class_sim"])

        rows.append({
            "epoch": epoch,
            "contraction": round(contraction, 4),
            "rho_inband": stats.get("mean_cross_class_sim", float("nan")),
            "var_inband": stats.get("var_cross_class_sim", float("nan")),
            "n_inband": stats.get("n_cross_class_obs", 0),
            "rho_probe": p.get("mean_cross_class_sim", float("nan")),
            "var_probe": p.get("var_cross_class_sim", float("nan")),
            "within_probe": p.get("mean_within_class_sim", float("nan")),
            "margin_probe": p.get("mean_true_margin", float("nan")),
            "offclass_weight": p.get("mean_offclass_weight", float("nan")),
            "frac_blended": p.get("frac_blended", float("nan")),
            "n_vote": p.get("n_vote_probes", 0),
            "top1_top2_margin": p.get("mean_top1_top2_margin", float("nan")),
            "median_top1_top2_margin": p.get("median_top1_top2_margin", float("nan")),
            "vote_entropy": p.get("mean_vote_entropy", float("nan")),
            "effective_support": p.get("mean_effective_support", float("nan")),
            "median_effective_support": p.get("median_effective_support", float("nan")),
            "max_vote_weight": p.get("mean_max_vote_weight", float("nan")),
            "n_surviving_votes": p.get("mean_n_surviving_votes", float("nan")),
            "frac_low_margin": p.get("frac_low_margin", float("nan")),
            "frac_high_entropy": p.get("frac_high_entropy", float("nan")),
            "frac_margin_recoverable": p.get("frac_margin_recoverable", float("nan")),
            "frac_margin_borderline": p.get("frac_margin_borderline", float("nan")),
            "frac_margin_forced": p.get("frac_margin_forced", float("nan")),
            "frac_blend_top1_correct": p.get("frac_blend_top1_correct", float("nan")),
            "chimera_onset": bool(p.get("chimera_onset", False)),
            "blend_onset": bool(p.get("blend_onset", False)),
            "mean_v_eff": stats.get("mean_v_effective", float("nan")),
            "floor_delta_ema": stats.get("floor_delta_ema", float("nan")),
            "sim_floor_active": stats.get("sim_floor_active", float("nan")),
            "occupied": occ,
            "alloc": alloc,
            "merged": merged,
            "dropped": dropped,
            "merge_rate": round(merge_rate, 4),
            "acc": round(acc, 4),
        })
        print(f"epoch {epoch:3d} | c={contraction:.3f} | "
              f"rho_probe={rows[-1]['rho_probe']:.3f} "
              f"within={rows[-1]['within_probe']:.3f} "
              f"offclass_w={rows[-1]['offclass_weight']:.3f} "
              f"frac_blend={rows[-1]['frac_blended']:.2f} "
              f"eff_supp={rows[-1]['effective_support']:.1f} "
              f"forced={rows[-1]['frac_margin_forced']:.2f} "
              f"blend_t1ok={rows[-1]['frac_blend_top1_correct']:.2f} "
              f"occ={occ} alloc={alloc} merged={merged} drop={dropped} "
              f"merge_rate={merge_rate:.2f} acc={acc:.3f} "
              f"| blend={rows[-1]['blend_onset']} chimera={rows[-1]['chimera_onset']}")

    # --- Write CSV ---
    csv_path = Path(args.csv)
    if csv_path.parent != Path(""):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} epochs to {args.csv}")

    # --- Onset report: observed vs analytic prediction ---
    print("\n=== Onset report ===")
    eps = args.blend_eps
    predicted_blend_rho = None
    if delta_ref is not None:
        predicted_blend_rho = delta_ref - mem.inference_temp * math.log((1 - eps) / eps)
        print(f"Delta (within-class self-sim, epoch 0): {delta_ref:.3f}")
        print(f"Predicted blend onset rho ~= Delta - temp*ln((1-eps)/eps) = {predicted_blend_rho:.3f} "
              f"(temp={mem.inference_temp}, eps={eps})")
    if first_blend is not None:
        e, rho, ow = first_blend
        print(f"Observed retrieval-blend onset: epoch {e}, rho={rho:.3f}, offclass_weight={ow:.3f}")
    else:
        print("Observed retrieval-blend onset: not reached")
    print(f"Predicted structural-chimera onset rho >= v_ceiling = {args.v_ceiling}")
    if first_chimera is not None:
        e, rho = first_chimera
        print(f"Observed structural-chimera onset: epoch {e}, rho={rho:.3f}")
    else:
        print("Observed structural-chimera onset: not reached")

    # --- Transfer verdict ---------------------------------------------------- #
    # Three separable questions, not one:
    #   (1) relative blend: does the Delta-anchored formula
    #       rho* = Delta - temp*ln((1-eps)/eps) predict the rho at which off-class
    #       vote mass crosses blend_eps?
    #   (2) absolute blend: does the synthetic ~0.79 onset rho transfer as an
    #       absolute number? (--ref-blend-rho)
    #   (3) chimera: is the absolute v_ceiling actually reachable on this manifold?
    max_rho = max((r["rho_probe"] for r in rows
                   if not math.isnan(r["rho_probe"])), default=float("nan"))
    # Left-censoring: if the blend onset fires on the very first epoch, the true
    # crossing rho is at or below the observed value -- the engine is already in
    # the blend regime at the manifold's resting state, before any drift.
    baseline_blended = first_blend is not None and first_blend[0] == 0
    chimera_reachable = (not math.isnan(max_rho)) and max_rho >= args.v_ceiling

    verdict = {"mode": "real" if args.real else "synthetic",
               "delta_ref": delta_ref,
               "predicted_blend_rho": predicted_blend_rho,
               "ref_blend_rho": args.ref_blend_rho,
               "v_ceiling": args.v_ceiling,
               "max_rho_observed": max_rho,
               "baseline_blended": baseline_blended,
               "chimera_reachable": chimera_reachable,
               "blend_onset": None, "chimera_onset": None,
               "blend_offset": None,
               "relative_transfer": "indeterminate",
               "absolute_transfer": "indeterminate",
               "chimera_transfer": "unreachable" if not chimera_reachable else "reachable",
               # Intervention config (issue #82) so each verdict records the exact
               # retrieval-side levers active for this run (baseline vs intervention).
               "retrieval_floor_policy": args.retrieval_floor_policy,
               "support_truncation": bool(args.support_truncation),
               "truncation_window_steps": (
                   args.truncation_window_steps if args.support_truncation else None),
               "truncation_gate_steps": (
                   args.truncation_gate_steps if args.support_truncation else None)}
    if args.real:
        verdict["categories"] = stream.categories
        verdict["attractor_category"] = args.attractor_category
        verdict["embed_model"] = args.embed_model
    if first_blend is not None:
        verdict["blend_onset"] = {"epoch": first_blend[0],
                                  "rho": first_blend[1],
                                  "offclass_weight": first_blend[2],
                                  "left_censored": baseline_blended}
    if first_chimera is not None:
        verdict["chimera_onset"] = {"epoch": first_chimera[0], "rho": first_chimera[1]}

    print(f"\n=== Transfer verdict ({verdict['mode']}) ===")
    if predicted_blend_rho is not None and first_blend is not None:
        offset = first_blend[1] - predicted_blend_rho   # observed - predicted
        verdict["blend_offset"] = offset
        # |offset| within one inference-temp logit step is "confirmed"; a stable,
        # quantified gap is "confirmed-with-offset"; anything larger is "refuted".
        tol = 2.0 * mem.inference_temp
        if abs(offset) <= tol:
            verdict["relative_transfer"] = "confirmed"
        elif abs(offset) <= 0.20:
            verdict["relative_transfer"] = "confirmed-with-offset"
        else:
            verdict["relative_transfer"] = "refuted"
        censor = " (LEFT-CENSORED: onset at epoch 0; true crossing rho <= this)" \
            if baseline_blended else ""
        print(f"  [relative] {verdict['relative_transfer']}: observed blend "
              f"rho={first_blend[1]:.3f} vs Delta-anchored {predicted_blend_rho:.3f} "
              f"(offset {offset:+.3f}, tol +/-{tol:.3f}){censor}")
        # Absolute transfer: does the REAL onset land near the synthetic
        # absolute onset rho? (Only meaningful off-synthetic -- on the synthetic
        # run itself the reference is the value being reproduced.)
        if args.real:
            abs_gap = first_blend[1] - args.ref_blend_rho
            verdict["absolute_transfer"] = (
                "transfers" if abs(abs_gap) <= 0.10 else "does-not-transfer")
            print(f"  [absolute] {verdict['absolute_transfer']}: observed onset "
                  f"rho={first_blend[1]:.3f} vs synthetic ref {args.ref_blend_rho:.3f} "
                  f"(gap {abs_gap:+.3f})")
        else:
            verdict["absolute_transfer"] = "n/a (synthetic reference run)"
    elif first_blend is None:
        print("  [blend] onset not reached in "
              f"{args.epochs} epochs (raise --contraction-end)")
    print(f"  [chimera] {verdict['chimera_transfer']}: max rho observed "
          f"{max_rho:.3f} vs v_ceiling {args.v_ceiling}")

    if args.verdict_json:
        import json
        vp = Path(args.verdict_json)
        if vp.parent != Path(""):
            vp.parent.mkdir(parents=True, exist_ok=True)
        with open(vp, "w") as f:
            json.dump(verdict, f, indent=2)
        print(f"Wrote verdict to {vp}")

    # --- Optional plot ---
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping plot")
            return
        ep = [r["epoch"] for r in rows]
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(ep, [r["rho_probe"] for r in rows], "o-", color="C0", label="rho_probe (true-label)")
        ax1.plot(ep, [r["rho_inband"] for r in rows], ".--", color="C0", alpha=0.5, label="rho_inband (cheap)")
        ax1.plot(ep, [r["within_probe"] for r in rows], "-", color="C2", alpha=0.6, label="within-class (Delta)")
        ax1.axhline(args.v_ceiling, color="C3", ls=":", label=f"v_ceiling={args.v_ceiling}")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("cosine similarity")
        ax1.set_ylim(0, 1.02)
        ax2 = ax1.twinx()
        ax2.plot(ep, [r["offclass_weight"] for r in rows], "s-", color="C1", label="off-class vote mass")
        ax2.axhline(args.blend_eps, color="C1", ls=":", alpha=0.6, label=f"blend_eps={args.blend_eps}")
        ax2.set_ylabel("off-class vote mass")
        ax2.set_ylim(0, 1.02)
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=8)
        ax1.set_title("rho(t) contraction and retrieval-blend onset")
        fig.tight_layout()
        plot_path = Path(args.plot_path)
        if plot_path.parent != Path(""):
            plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=120)
        print(f"Wrote plot to {plot_path}")


if __name__ == "__main__":
    main()
