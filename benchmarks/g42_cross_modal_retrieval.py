#!/usr/bin/env python3
"""
G-42 — FAM Cross-Modal Retrieval Benchmark

Phase A: Flickr30k 1K test split
Phase B: COCO 5K test split
Encoder: CLIP ViT-L/14 (768-d shared embedding space)
Baseline: CLIP kNN-20
FAM: shared prototype store for both modalities (interleaved writes)

Usage:
    PYTHONPATH=. .venv/bin/python benchmarks/g42_cross_modal_retrieval.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from fast_associative_memory import FastAssociativeMemory

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_CACHE = Path("data/g42_embeddings/.hf_cache")
EMB_CACHE = Path("data/g42_embeddings")
OUT_DIR = Path("results/g42_cross_modal")
MODEL_CACHE = Path(".model_cache")

# CLIP ViT-L/14 → 768-d shared space
EMBED_DIM = 768
IMG_BATCH = 64     # images per CLIP encode batch
TXT_BATCH = 256    # captions per CLIP encode batch

CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FAM hyperparameters (no adapters, no whitening)
FAM_VIGILANCE_DEFAULT = 0.85
FAM_K = 20
FAM_TEMP = 0.05


def set_seeds():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------------
# CLIP model loading
# ---------------------------------------------------------------------------

def load_clip():
    """Load CLIP ViT-L/14 from HuggingFace transformers."""
    from transformers import CLIPModel, CLIPProcessor
    print("  Loading CLIP ViT-L/14 ...")
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        cache_dir=str(MODEL_CACHE),
        torch_dtype=torch.float16,
    ).to(CLIP_DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14",
        cache_dir=str(MODEL_CACHE),
    )
    return model, processor


@torch.no_grad()
def encode_images(model, processor, images: list, batch_size: int = IMG_BATCH) -> torch.Tensor:
    """Encode a list of PIL images → (N, 768) normalized float32 on CPU."""
    all_embs = []
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i + batch_size]
        inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(CLIP_DEVICE)
        vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
        feats = model.visual_projection(vision_out.pooler_output)
        feats = F.normalize(feats.float(), dim=-1)
        all_embs.append(feats.cpu())
        if (i // batch_size) % 10 == 0:
            print(f"    images {i}/{len(images)}", end="\r")
    print()
    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def encode_texts(model, processor, texts: list, batch_size: int = TXT_BATCH) -> torch.Tensor:
    """Encode a list of strings → (N, 768) normalized float32 on CPU."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch_txt = texts[i:i + batch_size]
        inputs = processor(text=batch_txt, return_tensors="pt", padding=True, truncation=True,
                           max_length=77).to(CLIP_DEVICE)
        text_out = model.text_model(input_ids=inputs["input_ids"],
                                    attention_mask=inputs["attention_mask"])
        feats = model.text_projection(text_out.pooler_output)
        feats = F.normalize(feats.float(), dim=-1)
        all_embs.append(feats.cpu())
        if (i // batch_size) % 10 == 0:
            print(f"    texts {i}/{len(texts)}", end="\r")
    print()
    return torch.cat(all_embs, dim=0)


# ---------------------------------------------------------------------------
# Dataset loading and embedding extraction
# ---------------------------------------------------------------------------

def load_and_embed(dataset_name: str, split: str, model, processor) -> dict:
    """
    Load dataset, extract CLIP embeddings, cache to disk.

    Returns dict:
      img_embs:  (N_img, 768) — one per image
      cap_embs:  (N_cap, 768) — N_caps_per_img captions per image
      cap_imgid: (N_cap,) int  — image index for each caption
      N_img, N_cap, N_caps_per_img
    """
    cache_file = EMB_CACHE / f"{dataset_name.replace('/', '_')}_{split}.pt"
    if cache_file.exists():
        print(f"  Loading cached embeddings from {cache_file}")
        return torch.load(cache_file, map_location="cpu", weights_only=True)

    from datasets import load_dataset
    print(f"  Loading {dataset_name} {split} split ...")
    ds = load_dataset(dataset_name, split=split, cache_dir=str(HF_CACHE))

    images = []
    all_captions = []
    cap_imgid = []

    for i, ex in enumerate(ds):
        img = ex["jpg"].convert("RGB")
        images.append(img)
        caps = ex["txt"].strip().split("\n")
        # Some examples might have fewer than 5 captions — pad/truncate to 5
        caps = (caps + caps * 5)[:5]
        for c in caps:
            all_captions.append(c.strip())
            cap_imgid.append(i)
        if i % 500 == 0:
            print(f"  Parsed {i}/{len(ds)} items", end="\r")
    print(f"\n  {len(images)} images, {len(all_captions)} captions")

    print("  Encoding images ...")
    img_embs = encode_images(model, processor, images)
    print("  Encoding captions ...")
    cap_embs = encode_texts(model, processor, all_captions)

    data = {
        "img_embs": img_embs.half(),      # save as fp16 to reduce disk use
        "cap_embs": cap_embs.half(),
        "cap_imgid": torch.tensor(cap_imgid, dtype=torch.long),
        "N_img": len(images),
        "N_cap": len(all_captions),
        "N_caps_per_img": 5,
    }
    torch.save(data, cache_file)
    print(f"  Saved embeddings to {cache_file}")
    return data


# ---------------------------------------------------------------------------
# Recall@K evaluation helpers
# ---------------------------------------------------------------------------

def recall_at_k(scores: torch.Tensor, targets: torch.Tensor,
                ks=(1, 5, 10)) -> dict:
    """
    Generic R@K.

    Args:
        scores:  (N_queries, N_candidates) similarity matrix
        targets: (N_queries,) int or (N_queries, max_correct) int  — correct candidate index/indices
                 Use -1 as padding for variable-length correct sets.
        ks:      tuple of K values

    Returns: {k: recall_pct}
    """
    results = {}
    N_q = scores.size(0)
    # Sort candidates descending for each query
    ranked = scores.argsort(dim=1, descending=True)   # (N_q, N_candidates)

    # Normalise targets to (N_queries, max_correct) with -1 padding
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)   # (N_q, 1)

    for k in ks:
        top_k = ranked[:, :k]            # (N_q, k)
        # Vectorised: broadcast compare (N_q, k, 1) vs (N_q, 1, max_correct)
        hit = (top_k.unsqueeze(2) == targets.unsqueeze(1)).any(dim=(1, 2)).sum().item()
        results[k] = hit / N_q * 100.0
    return results


# ---------------------------------------------------------------------------
# kNN Baseline
# ---------------------------------------------------------------------------

def run_knn_baseline(img_embs: torch.Tensor, cap_embs: torch.Tensor,
                     cap_imgid: torch.Tensor, N_img: int,
                     device: str = DEVICE) -> dict:
    """
    CLIP kNN-20 retrieval baseline.

    text→image: for each caption, rank all images by cosine sim.
    image→text: for each image, rank all captions by cosine sim.
    """
    img = img_embs.float().to(device)       # (N_img, 768)
    cap = cap_embs.float().to(device)       # (N_cap, 768)
    # Already L2-normalised

    print("  kNN baseline: text→image ...")
    # (N_cap, N_img) similarity
    batch = 512
    scores_t2i = []
    for i in range(0, len(cap), batch):
        s = (cap[i:i + batch] @ img.T)      # (B, N_img)
        scores_t2i.append(s.cpu())
    scores_t2i = torch.cat(scores_t2i, dim=0)     # (N_cap, N_img)

    # Correct image for each caption
    t2i_targets = cap_imgid.long()         # (N_cap,)
    t2i_metrics = recall_at_k(scores_t2i, t2i_targets)

    print("  kNN baseline: image→text ...")
    # (N_img, N_cap) similarity
    scores_i2t = []
    for i in range(0, len(img), batch):
        s = (img[i:i + batch] @ cap.T)     # (B, N_cap)
        scores_i2t.append(s.cpu())
    scores_i2t = torch.cat(scores_i2t, dim=0)     # (N_img, N_cap)

    # For each image, correct captions are those with matching image_id
    # Build (N_img, N_cap) target boolean mask → then pack to (N_img, 5)
    cap_per_img = 5
    i2t_targets = torch.zeros(N_img, cap_per_img, dtype=torch.long) - 1
    for img_id in range(N_img):
        idxs = (cap_imgid == img_id).nonzero(as_tuple=True)[0]
        n = min(len(idxs), cap_per_img)
        i2t_targets[img_id, :n] = idxs[:n]

    i2t_metrics = recall_at_k(scores_i2t, i2t_targets)

    return {"t2i": t2i_metrics, "i2t": i2t_metrics}


# ---------------------------------------------------------------------------
# FAM setup and write
# ---------------------------------------------------------------------------

def build_fam(N_img: int, max_entries: int, vigilance: float = FAM_VIGILANCE_DEFAULT,
              device: str = DEVICE) -> FastAssociativeMemory:
    """Create a FAM with image_id as class labels (value_dim = N_img)."""
    fam = FastAssociativeMemory(
        input_dim=EMBED_DIM,
        value_dim=N_img,
        core_entries=max_entries,
        core_vigilance=vigilance,
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=FAM_K,
        inference_temp=FAM_TEMP,
        whitening_dim=0,        # no whitening
        use_lfu=False,
        adaptive_eviction=True,
        adapter=None,
        nstp=None,
    ).to(device)
    return fam


@torch.no_grad()
def write_fam_interleaved(fam: FastAssociativeMemory,
                          img_embs: torch.Tensor,
                          cap_embs: torch.Tensor,
                          cap_imgid: torch.Tensor,
                          N_img: int,
                          device: str = DEVICE):
    """
    Write all embeddings into FAM in interleaved order:
    for each image_id: image embedding, then its 5 captions.

    Targets are one-hot over image_ids.
    """
    # Build per-image list of caption indices
    caps_for_img = defaultdict(list)
    for ci, img_id in enumerate(cap_imgid.tolist()):
        caps_for_img[img_id].append(ci)

    # Interleaved sequence: (feat, image_id) pairs
    seq_feats = []
    seq_labels = []
    for img_id in range(N_img):
        seq_feats.append(img_embs[img_id])
        seq_labels.append(img_id)
        for ci in caps_for_img[img_id]:
            seq_feats.append(cap_embs[ci])
            seq_labels.append(img_id)

    seq_feats = torch.stack(seq_feats, dim=0)      # (N_total, 768)
    seq_labels = torch.tensor(seq_labels, dtype=torch.long)  # (N_total,)

    # One-hot encode targets
    targets = torch.zeros(len(seq_labels), N_img)
    targets.scatter_(1, seq_labels.unsqueeze(1), 1.0)

    # Write in mini-batches
    write_batch = 256
    N_total = len(seq_feats)
    for i in range(0, N_total, write_batch):
        bx = seq_feats[i:i + write_batch].to(device)
        bt = targets[i:i + write_batch].to(device)
        fam.core_cam.learn_local(fam._project(bx), bt)
        if i % 2000 == 0:
            n_occ = int(fam.core_cam.occupied.sum().item())
            print(f"    write {i}/{N_total} | prototypes: {n_occ}", end="\r")

    n_occ = int(fam.core_cam.occupied.sum().item())
    print(f"\n    Wrote {N_total} items → {n_occ} prototypes")
    return seq_feats, seq_labels, caps_for_img


# ---------------------------------------------------------------------------
# Provenance tracking
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_provenance(fam: FastAssociativeMemory,
                       img_embs: torch.Tensor,
                       cap_embs: torch.Tensor,
                       cap_imgid: torch.Tensor,
                       N_img: int,
                       device: str = DEVICE) -> dict:
    """
    For each occupied prototype, determine which items (images / captions)
    are assigned to it (nearest prototype in cosine space).

    Returns:
      proto_img_ids[slot]:  set of image_ids assigned (image modal origin)
      proto_cap_ids[slot]:  set of (cap_idx, image_id) tuples (text modal origin)
      item_to_proto[item_idx, modality_str]: slot index
    """
    valid_idx = fam.core_cam.occupied.nonzero(as_tuple=True)[0]  # (P,)
    P = len(valid_idx)
    proto_keys = fam.core_cam._keys_norm[valid_idx]   # (P, 768)

    # Assign each image to nearest prototype
    img = img_embs.float().to(device)     # (N_img, 768)
    batch = 512
    img_proto_assign = []
    for i in range(0, len(img), batch):
        sims = img[i:i + batch] @ proto_keys.T  # (B, P)
        best = sims.argmax(dim=1)                # (B,)
        img_proto_assign.append(valid_idx[best].cpu())
    img_proto_assign = torch.cat(img_proto_assign, dim=0)   # (N_img,)

    # Assign each caption to nearest prototype
    cap = cap_embs.float().to(device)     # (N_cap, 768)
    cap_proto_assign = []
    for i in range(0, len(cap), batch):
        sims = cap[i:i + batch] @ proto_keys.T
        best = sims.argmax(dim=1)
        cap_proto_assign.append(valid_idx[best].cpu())
    cap_proto_assign = torch.cat(cap_proto_assign, dim=0)   # (N_cap,)

    # Build provenance dicts
    proto_img_ids: dict[int, set] = defaultdict(set)
    proto_cap_ids: dict[int, set] = defaultdict(set)

    for img_id in range(N_img):
        slot = img_proto_assign[img_id].item()
        proto_img_ids[slot].add(img_id)

    for ci, img_id in enumerate(cap_imgid.tolist()):
        slot = cap_proto_assign[ci].item()
        proto_cap_ids[slot].add((ci, img_id))

    return {
        "proto_img_ids": dict(proto_img_ids),
        "proto_cap_ids": dict(proto_cap_ids),
        "img_proto_assign": img_proto_assign,
        "cap_proto_assign": cap_proto_assign,
        "valid_idx": valid_idx.cpu(),
    }


# ---------------------------------------------------------------------------
# FAM retrieval
# ---------------------------------------------------------------------------

@torch.no_grad()
def fam_retrieve(fam: FastAssociativeMemory,
                 queries: torch.Tensor,
                 N_img: int,
                 device: str = DEVICE,
                 batch_size: int = 256) -> torch.Tensor:
    """
    FAM retrieval: query → vote distribution over image_ids → (N_q, N_img) scores.

    Uses FAM's forward() (soft-kNN vote) as the scoring function.
    """
    queries = queries.float()
    all_scores = []
    for i in range(0, len(queries), batch_size):
        bq = queries[i:i + batch_size].to(device)
        bq_proj = fam._project(bq)
        scores = fam.core_cam(bq_proj)     # (B, N_img) — vote logits
        all_scores.append(scores.cpu())
    return torch.cat(all_scores, dim=0)    # (N_q, N_img)


# ---------------------------------------------------------------------------
# FAM diagnostics
# ---------------------------------------------------------------------------

def compute_fam_diagnostics(fam: FastAssociativeMemory,
                             provenance: dict,
                             N_img: int,
                             N_total_written: int) -> dict:
    """
    Compute FAM diagnostic telemetry:
    - Condensation ratio
    - Prototype count
    - Modality-origin breakdown
    - Cross-modal merge rate
    """
    n_proto = int(fam.core_cam.occupied.sum().item())
    cr = 1.0 - (n_proto / N_total_written)

    proto_img_ids = provenance["proto_img_ids"]
    proto_cap_ids = provenance["proto_cap_ids"]
    valid_idx = provenance["valid_idx"].tolist()

    n_image_only = 0
    n_text_only = 0
    n_cross_modal = 0
    n_image_origin = 0   # prototypes with at least one image item
    n_text_origin = 0    # prototypes with at least one text item

    for slot in valid_idx:
        has_img = slot in proto_img_ids and len(proto_img_ids[slot]) > 0
        has_cap = slot in proto_cap_ids and len(proto_cap_ids[slot]) > 0
        if has_img:
            n_image_origin += 1
        if has_cap:
            n_text_origin += 1
        if has_img and not has_cap:
            n_image_only += 1
        elif has_cap and not has_img:
            n_text_only += 1
        elif has_img and has_cap:
            n_cross_modal += 1

    cross_modal_merge_rate = n_cross_modal / n_proto if n_proto > 0 else 0.0

    return {
        "cr": cr,
        "n_proto": n_proto,
        "n_image_origin": n_image_origin,
        "n_text_origin": n_text_origin,
        "n_image_only": n_image_only,
        "n_text_only": n_text_only,
        "n_cross_modal": n_cross_modal,
        "cross_modal_merge_rate": cross_modal_merge_rate,
    }


# ---------------------------------------------------------------------------
# Confidence AUROC
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_confidence_auroc(fam: FastAssociativeMemory,
                              img_embs: torch.Tensor,
                              cap_embs: torch.Tensor,
                              cap_imgid: torch.Tensor,
                              provenance: dict,
                              N_img: int,
                              device: str = DEVICE,
                              n_sample: int = 500) -> float:
    """
    AUROC distinguishing cross-modal vs same-modal queries using composite
    confidence from forward_with_confidence().

    Positive = same-modal (image query → image-dominated prototype)
    Negative = cross-modal (text query → image-dominated prototype, or vice versa)

    Uses a random subsample of n_sample from each modality.
    """
    from sklearn.metrics import roc_auc_score

    img_proto_assign = provenance["img_proto_assign"]   # (N_img,)
    cap_proto_assign = provenance["cap_proto_assign"]   # (N_cap,)
    proto_img_ids = provenance["proto_img_ids"]
    proto_cap_ids = provenance["proto_cap_ids"]

    # Determine each prototype's dominant modality
    def dominant_modality(slot):
        n_img = len(proto_img_ids.get(slot, set()))
        n_cap = len(proto_cap_ids.get(slot, set()))
        if n_img > n_cap:
            return "image"
        elif n_cap > n_img:
            return "text"
        else:
            return "mixed"

    rng = np.random.RandomState(SEED)
    img_idx = rng.choice(N_img, size=min(n_sample, N_img), replace=False)
    cap_idx = rng.choice(len(cap_imgid), size=min(n_sample, len(cap_imgid)), replace=False)

    scores = []
    labels = []   # 1 = same-modal, 0 = cross-modal

    # Image queries
    img_q = img_embs[img_idx].float().to(device)
    _, conf = fam.forward_with_confidence(img_q)
    for i, idx in enumerate(img_idx):
        slot = img_proto_assign[idx].item()
        dom = dominant_modality(slot)
        labels.append(1 if dom == "image" else 0)
        scores.append(conf[i].item())

    # Caption queries
    cap_q = cap_embs[cap_idx].float().to(device)
    _, conf = fam.forward_with_confidence(cap_q)
    for i, idx in enumerate(cap_idx):
        slot = cap_proto_assign[idx].item()
        dom = dominant_modality(slot)
        labels.append(1 if dom == "text" else 0)
        scores.append(conf[i].item())

    scores = np.array(scores)
    labels = np.array(labels)

    # Guard: need both classes present
    if len(np.unique(labels)) < 2:
        return float("nan")

    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Failure mode classifier
# ---------------------------------------------------------------------------

def classify_failures(fam_scores: torch.Tensor,
                      knn_scores: torch.Tensor,
                      query_labels: torch.Tensor,     # (N_q,) correct image_id
                      provenance: dict,
                      mode: str = "t2i",
                      n_examples: int = 5) -> dict:
    """
    For incorrect FAM retrievals, classify failure mode:
    - modality-gap: correct item is cross-modal to query but ranked below same-modal distractors
    - condensation-fragmentation: correct concept exists but split into multiple prototype clusters
    - vote-collapse: top-K candidate set is mixed-modality → noisy softmax

    Returns:
      counts: {mode: count}
      examples: list of dicts with query info and failure classification
    """
    N_q = fam_scores.size(0)
    ranked = fam_scores.argsort(dim=1, descending=True)   # (N_q, N_img)

    proto_img_ids = provenance["proto_img_ids"]
    proto_cap_ids = provenance["proto_cap_ids"]
    valid_idx = set(provenance["valid_idx"].tolist())

    counts = {"modality-gap": 0, "condensation-fragmentation": 0, "vote-collapse": 0}
    examples = []

    incorrect_mask = ranked[:, 0] != query_labels
    incorrect_idxs = incorrect_mask.nonzero(as_tuple=True)[0].tolist()

    for qi in incorrect_idxs[:200]:   # cap analysis at 200
        correct_id = query_labels[qi].item()
        top1_id = ranked[qi, 0].item()

        # Find prototypes responsible for correct_id
        correct_proto_slots = [
            s for s in proto_img_ids if correct_id in proto_img_ids[s]
        ] if mode == "t2i" else [
            s for s in proto_cap_ids
            if any(img_id == correct_id for _, img_id in proto_cap_ids[s])
        ]

        top1_proto_slots = [
            s for s in proto_img_ids if top1_id in proto_img_ids[s]
        ] if mode == "t2i" else []

        # modality-gap: correct concept exists in store but was a cross-modal write
        # and ranked below a same-modal distractor
        if len(correct_proto_slots) == 0:
            # correct image never merged into any prototype for this direction
            failure = "condensation-fragmentation"
        elif len(correct_proto_slots) > 1:
            # correct concept fragmented across multiple prototypes
            failure = "condensation-fragmentation"
        else:
            # Determine if top-K region is mixed modality → vote-collapse
            top_k_ids = ranked[qi, :FAM_K].tolist()
            top_k_has_mixed = False
            for cid in top_k_ids:
                for s in proto_img_ids:
                    if cid in proto_img_ids[s] and s in proto_cap_ids:
                        top_k_has_mixed = True
                        break
            if top_k_has_mixed:
                failure = "vote-collapse"
            else:
                failure = "modality-gap"

        counts[failure] += 1

        if len(examples) < n_examples:
            examples.append({
                "query_idx": qi,
                "correct_id": correct_id,
                "top1_retrieved": top1_id,
                "failure_mode": failure,
                "n_correct_protos": len(correct_proto_slots),
                "knn_rank": (knn_scores[qi].argsort(descending=True) == correct_id).nonzero(
                    as_tuple=True)[0][0].item() + 1 if knn_scores is not None else -1,
            })

    return {"counts": counts, "examples": examples, "n_incorrect": len(incorrect_idxs)}


# ---------------------------------------------------------------------------
# Per-query telemetry (Phase B)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_phase_b_telemetry(fam: FastAssociativeMemory,
                               img_embs: torch.Tensor,
                               cap_embs: torch.Tensor,
                               cap_imgid: torch.Tensor,
                               provenance: dict,
                               N_img: int,
                               device: str = DEVICE,
                               sample_n: int = 200) -> dict:
    """
    Per-query diagnostic telemetry for Phase B.
    """
    proto_img_ids = provenance["proto_img_ids"]
    proto_cap_ids = provenance["proto_cap_ids"]
    valid_idx = provenance["valid_idx"]
    proto_keys = fam.core_cam._keys_norm[valid_idx].float()   # (P, 768)

    rng = np.random.RandomState(SEED)
    img_sample = rng.choice(N_img, size=min(sample_n, N_img), replace=False)
    cap_sample = rng.choice(len(cap_imgid), size=min(sample_n, len(cap_imgid)), replace=False)

    def proto_dominant_modality(slot):
        n_i = len(proto_img_ids.get(slot, set()))
        n_c = len(proto_cap_ids.get(slot, set()))
        return "image" if n_i >= n_c else "text"

    # For sample queries, compute top-K candidate modality distribution
    cross_modal_sims = []
    same_modal_sims = []
    cross_conf = []
    same_conf = []

    for modality, sample_idx, feats in [
        ("image", img_sample, img_embs),
        ("text", cap_sample, cap_embs),
    ]:
        qs = feats[sample_idx].float().to(device)
        qs_proj = fam._project(qs)
        _, conf = fam.forward_with_confidence(qs_proj)

        # Similarity to prototype keys
        sims = qs_proj @ proto_keys.to(device).T    # (B, P)
        topk_sims, topk_locs = sims.topk(FAM_K, dim=1)
        topk_slots = valid_idx[topk_locs.cpu()]     # (B, K)

        for bi, qi in enumerate(sample_idx):
            # modality of top-K prototypes
            top_k_modalities = [
                proto_dominant_modality(topk_slots[bi, k].item())
                for k in range(FAM_K)
            ]
            n_same = sum(1 for m in top_k_modalities if m == modality)
            n_cross = FAM_K - n_same

            # cosine similarities
            sims_bi = topk_sims[bi].tolist()

            for k, (sim, m) in enumerate(zip(sims_bi, top_k_modalities)):
                if m == modality:
                    same_modal_sims.append(sim)
                else:
                    cross_modal_sims.append(sim)

            c = conf[bi].item()
            if n_cross > n_same:
                cross_conf.append(c)
            else:
                same_conf.append(c)

    return {
        "cross_modal_sim_mean": float(np.mean(cross_modal_sims)) if cross_modal_sims else float("nan"),
        "cross_modal_sim_std": float(np.std(cross_modal_sims)) if cross_modal_sims else float("nan"),
        "same_modal_sim_mean": float(np.mean(same_modal_sims)) if same_modal_sims else float("nan"),
        "same_modal_sim_std": float(np.std(same_modal_sims)) if same_modal_sims else float("nan"),
        "cross_conf_mean": float(np.mean(cross_conf)) if cross_conf else float("nan"),
        "cross_conf_std": float(np.std(cross_conf)) if cross_conf else float("nan"),
        "same_conf_mean": float(np.mean(same_conf)) if same_conf else float("nan"),
        "same_conf_std": float(np.std(same_conf)) if same_conf else float("nan"),
        "n_cross_sims": len(cross_modal_sims),
        "n_same_sims": len(same_modal_sims),
    }


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

def run_phase(phase_name: str, dataset_name: str, split: str,
              model, processor, vigilance: float = FAM_VIGILANCE_DEFAULT) -> dict:
    """Run full evaluation for one phase."""
    print(f"\n{'='*60}")
    print(f"  {phase_name}: {dataset_name} {split}")
    print(f"{'='*60}")

    # 1. Load/extract embeddings
    data = load_and_embed(dataset_name, split, model, processor)
    img_embs = data["img_embs"].float()
    cap_embs = data["cap_embs"].float()
    cap_imgid = data["cap_imgid"]
    N_img = int(data["N_img"])
    N_cap = int(data["N_cap"])
    N_total = N_img + N_cap

    print(f"  {N_img} images, {N_cap} captions")

    # 2. kNN baseline
    print("\n  [Baseline] CLIP kNN-20")
    knn_metrics = run_knn_baseline(img_embs, cap_embs, cap_imgid, N_img)
    print(f"    t2i: R@1={knn_metrics['t2i'][1]:.1f} R@5={knn_metrics['t2i'][5]:.1f} R@10={knn_metrics['t2i'][10]:.1f}")
    print(f"    i2t: R@1={knn_metrics['i2t'][1]:.1f} R@5={knn_metrics['i2t'][5]:.1f} R@10={knn_metrics['i2t'][10]:.1f}")

    # 3. FAM evaluation
    print(f"\n  [FAM] vigilance={vigilance}")
    max_entries = min(N_total, max(N_img * 4, 4000))
    fam = build_fam(N_img, max_entries, vigilance=vigilance)

    set_seeds()
    seq_feats, seq_labels, caps_for_img = write_fam_interleaved(
        fam, img_embs, cap_embs, cap_imgid, N_img
    )

    # 4. Provenance tracking
    print("  Computing provenance ...")
    prov = compute_provenance(fam, img_embs, cap_embs, cap_imgid, N_img)

    # 5. FAM retrieval
    print("  FAM retrieval: text→image ...")
    fam_t2i_scores = fam_retrieve(fam, cap_embs, N_img)
    t2i_targets = cap_imgid.long()
    fam_t2i_metrics = recall_at_k(fam_t2i_scores, t2i_targets)

    print("  FAM retrieval: image→text ...")
    fam_i2t_scores = fam_retrieve(fam, img_embs, N_img)

    # For image→text: correct image_id for each query image = query image's own id
    # R@K: is the correct image_id in top-K positions?
    # (since each image_id maps to 5 captions, top-K image_ids expand to 5K captions)
    i2t_self_targets = torch.arange(N_img, dtype=torch.long)
    fam_i2t_metrics = recall_at_k(fam_i2t_scores, i2t_self_targets)

    print(f"    FAM t2i: R@1={fam_t2i_metrics[1]:.1f} R@5={fam_t2i_metrics[5]:.1f} R@10={fam_t2i_metrics[10]:.1f}")
    print(f"    FAM i2t: R@1={fam_i2t_metrics[1]:.1f} R@5={fam_i2t_metrics[5]:.1f} R@10={fam_i2t_metrics[10]:.1f}")

    # 6. Diagnostics
    print("  Computing FAM diagnostics ...")
    diag = compute_fam_diagnostics(fam, prov, N_img, N_total)

    # 7. Confidence AUROC
    print("  Computing confidence AUROC ...")
    try:
        auroc = compute_confidence_auroc(fam, img_embs, cap_embs, cap_imgid, prov, N_img)
    except Exception as e:
        print(f"    AUROC failed: {e}")
        auroc = float("nan")

    # 8. Failure mode analysis
    print("  Classifying failures (t2i) ...")
    knn_t2i_scores = (cap_embs.float() @ img_embs.float().T)   # (N_cap, N_img)
    failure_t2i = classify_failures(
        fam_t2i_scores, knn_t2i_scores, t2i_targets, prov, mode="t2i"
    )

    # 9. Phase B extra telemetry
    phase_b_telemetry = None
    if "COCO" in phase_name or "B" in phase_name:
        print("  Computing Phase B telemetry ...")
        phase_b_telemetry = compute_phase_b_telemetry(
            fam, img_embs, cap_embs, cap_imgid, prov, N_img
        )

        # CR at additional vigilance levels
        print("  CR at v=0.92 ...")
        fam_v92 = build_fam(N_img, max_entries, vigilance=0.92)
        set_seeds()
        write_fam_interleaved(fam_v92, img_embs, cap_embs, cap_imgid, N_img)
        n92 = int(fam_v92.core_cam.occupied.sum().item())
        cr_92 = 1.0 - (n92 / N_total)

        print("  CR at v=0.70 ...")
        fam_v70 = build_fam(N_img, max_entries, vigilance=0.70)
        set_seeds()
        write_fam_interleaved(fam_v70, img_embs, cap_embs, cap_imgid, N_img)
        n70 = int(fam_v70.core_cam.occupied.sum().item())
        cr_70 = 1.0 - (n70 / N_total)

        phase_b_telemetry["cr_v092"] = cr_92
        phase_b_telemetry["cr_v070"] = cr_70
        phase_b_telemetry["n_proto_v092"] = n92
        phase_b_telemetry["n_proto_v070"] = n70

    return {
        "knn": knn_metrics,
        "fam_t2i": fam_t2i_metrics,
        "fam_i2t": fam_i2t_metrics,
        "diagnostics": diag,
        "auroc": auroc,
        "failures": failure_t2i,
        "phase_b_telemetry": phase_b_telemetry,
        "N_img": N_img,
        "N_cap": N_cap,
        "N_total": N_total,
        "max_entries": max_entries,
        "vigilance": vigilance,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seeds()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EMB_CACHE.mkdir(parents=True, exist_ok=True)

    # Load CLIP once
    model, processor = load_clip()

    # ---- Phase A: Flickr30k 1K ----
    phase_a = run_phase(
        phase_name="Phase A — Flickr30k 1K",
        dataset_name="clip-benchmark/wds_flickr30k",
        split="test",
        model=model,
        processor=processor,
    )

    # Save intermediate results
    torch.save(phase_a, OUT_DIR / "phase_a_results.pt")

    # Gate decision
    knn_t2i_r10 = phase_a["knn"]["t2i"][10]
    knn_i2t_r10 = phase_a["knn"]["i2t"][10]
    fam_t2i_r10 = phase_a["fam_t2i"][10]
    fam_i2t_r10 = phase_a["fam_i2t"][10]

    fam_loses_t2i = fam_t2i_r10 < knn_t2i_r10
    fam_loses_i2t = fam_i2t_r10 < knn_i2t_r10
    phase_a_kill = fam_loses_t2i and fam_loses_i2t

    print(f"\n  Gate: FAM t2i R@10={fam_t2i_r10:.1f} vs kNN={knn_t2i_r10:.1f} → {'LOSE' if fam_loses_t2i else 'WIN'}")
    print(f"  Gate: FAM i2t R@10={fam_i2t_r10:.1f} vs kNN={knn_i2t_r10:.1f} → {'LOSE' if fam_loses_i2t else 'WIN'}")
    print(f"  Phase A decision: {'KILL' if phase_a_kill else 'PASS'}")

    # ---- Phase B: COCO 5K (only if Phase A passes) ----
    phase_b = None
    if not phase_a_kill:
        phase_b = run_phase(
            phase_name="Phase B — COCO 5K",
            dataset_name="clip-benchmark/wds_mscoco_captions",
            split="test",
            model=model,
            processor=processor,
        )
        torch.save(phase_b, OUT_DIR / "phase_b_results.pt")

        knn_t2i_r10_b = phase_b["knn"]["t2i"][10]
        knn_i2t_r10_b = phase_b["knn"]["i2t"][10]
        fam_t2i_r10_b = phase_b["fam_t2i"][10]
        fam_i2t_r10_b = phase_b["fam_i2t"][10]
        fam_loses_b = (fam_t2i_r10_b < knn_t2i_r10_b) and (fam_i2t_r10_b < knn_i2t_r10_b)
        print(f"\n  Phase B decision: {'KILL' if fam_loses_b else 'PASS'}")

    # ---- Print final report ----
    print_report(phase_a, phase_b, phase_a_kill)

    # Save JSON summary
    summary = {
        "phase_a_decision": "KILL" if phase_a_kill else "PASS",
        "phase_a": {
            "knn_t2i": phase_a["knn"]["t2i"],
            "knn_i2t": phase_a["knn"]["i2t"],
            "fam_t2i": {str(k): v for k, v in phase_a["fam_t2i"].items()},
            "fam_i2t": {str(k): v for k, v in phase_a["fam_i2t"].items()},
            "diagnostics": phase_a["diagnostics"],
            "auroc": phase_a["auroc"],
            "failures": {
                "counts": phase_a["failures"]["counts"],
                "n_incorrect": phase_a["failures"]["n_incorrect"],
            },
        },
    }
    if phase_b is not None:
        summary["phase_b_decision"] = "KILL" if fam_loses_b else "PASS"
        summary["phase_b"] = {
            "knn_t2i": phase_b["knn"]["t2i"],
            "knn_i2t": phase_b["knn"]["i2t"],
            "fam_t2i": {str(k): v for k, v in phase_b["fam_t2i"].items()},
            "fam_i2t": {str(k): v for k, v in phase_b["fam_i2t"].items()},
            "diagnostics": phase_b["diagnostics"],
            "auroc": phase_b["auroc"],
            "phase_b_telemetry": phase_b["phase_b_telemetry"],
        }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {OUT_DIR}/")


def print_report(phase_a: dict, phase_b: dict | None, phase_a_kill: bool):
    """Print the structured markdown report."""

    def fmt_metrics(t2i, i2t):
        return (
            f"| R@1 | R@5 | R@10 | R@1 | R@5 | R@10 |\n"
            f"|-----|-----|------|-----|-----|------|\n"
            f"| {t2i[1]:.1f} | {t2i[5]:.1f} | {t2i[10]:.1f} | {i2t[1]:.1f} | {i2t[5]:.1f} | {i2t[10]:.1f} |"
        )

    print("\n\n" + "="*70)
    print("FINAL REPORT")
    print("="*70)

    knn_t2i_r10 = phase_a["knn"]["t2i"][10]
    knn_i2t_r10 = phase_a["knn"]["i2t"][10]
    fam_t2i_r10 = phase_a["fam_t2i"][10]
    fam_i2t_r10 = phase_a["fam_i2t"][10]

    phase_a_reason = []
    if fam_t2i_r10 >= knn_t2i_r10:
        phase_a_reason.append(f"FAM wins text→image R@10 ({fam_t2i_r10:.1f} vs {knn_t2i_r10:.1f})")
    else:
        phase_a_reason.append(f"FAM loses text→image R@10 ({fam_t2i_r10:.1f} vs {knn_t2i_r10:.1f})")
    if fam_i2t_r10 >= knn_i2t_r10:
        phase_a_reason.append(f"FAM wins image→text R@10 ({fam_i2t_r10:.1f} vs {knn_i2t_r10:.1f})")
    else:
        phase_a_reason.append(f"FAM loses image→text R@10 ({fam_i2t_r10:.1f} vs {knn_i2t_r10:.1f})")

    d = phase_a["diagnostics"]
    f = phase_a["failures"]

    print(f"""
## Summary
- Phase A decision: {'KILL' if phase_a_kill else 'PASS'}
  - {'; '.join(phase_a_reason)}
{'- Phase B: not run (Phase A killed)' if phase_a_kill and phase_b is None else ''}
""")

    if phase_b is not None:
        knn_t2i_r10_b = phase_b["knn"]["t2i"][10]
        fam_t2i_r10_b = phase_b["fam_t2i"][10]
        knn_i2t_r10_b = phase_b["knn"]["i2t"][10]
        fam_i2t_r10_b = phase_b["fam_i2t"][10]
        fam_loses_b = (fam_t2i_r10_b < knn_t2i_r10_b) and (fam_i2t_r10_b < knn_i2t_r10_b)
        print(f"- Phase B decision: {'KILL' if fam_loses_b else 'PASS'}")
        if fam_t2i_r10_b >= knn_t2i_r10_b:
            print(f"  - FAM wins COCO text→image R@10 ({fam_t2i_r10_b:.1f} vs {knn_t2i_r10_b:.1f})")
        if fam_i2t_r10_b >= knn_i2t_r10_b:
            print(f"  - FAM wins COCO image→text R@10 ({fam_i2t_r10_b:.1f} vs {knn_i2t_r10_b:.1f})")

    print(f"""
## Phase A — Flickr30k 1K

### Retrieval metrics

| System | t→i R@1 | t→i R@5 | t→i R@10 | i→t R@1 | i→t R@5 | i→t R@10 |
|--------|---------|---------|----------|---------|---------|----------|
| Baseline | {phase_a['knn']['t2i'][1]:.1f} | {phase_a['knn']['t2i'][5]:.1f} | {phase_a['knn']['t2i'][10]:.1f} | {phase_a['knn']['i2t'][1]:.1f} | {phase_a['knn']['i2t'][5]:.1f} | {phase_a['knn']['i2t'][10]:.1f} |
| FAM    | {phase_a['fam_t2i'][1]:.1f} | {phase_a['fam_t2i'][5]:.1f} | {phase_a['fam_t2i'][10]:.1f} | {phase_a['fam_i2t'][1]:.1f} | {phase_a['fam_i2t'][5]:.1f} | {phase_a['fam_i2t'][10]:.1f} |

### FAM diagnostics
- Condensation ratio: {d['cr']:.4f}
- Prototype count: {d['n_proto']} / {phase_a['N_total']} items written
- Modality-origin breakdown: {d['n_image_origin']} image-origin, {d['n_text_origin']} text-origin
  - Image-only: {d['n_image_only']}, Text-only: {d['n_text_only']}, Cross-modal merged: {d['n_cross_modal']}
- Cross-modal merge rate: {d['cross_modal_merge_rate']:.4f}
- Confidence AUROC (same-modal vs cross-modal): {phase_a['auroc']:.4f}

### Error analysis
- Total incorrect at R@1 (t2i): {f['n_incorrect']} / {phase_a['N_cap']}
- Failure mode counts: {f['counts']}
- Representative examples:""")

    for ex in f["examples"][:5]:
        print(f"  - Query image_id={ex['correct_id']}, retrieved={ex['top1_retrieved']}, "
              f"mode={ex['failure_mode']}, knn_rank={ex['knn_rank']}, "
              f"correct_protos={ex['n_correct_protos']}")

    if phase_b is not None:
        pb_d = phase_b["diagnostics"]
        pb_t = phase_b["phase_b_telemetry"]
        print(f"""
## Phase B — COCO 5K

### Retrieval metrics

| System | t→i R@1 | t→i R@5 | t→i R@10 | i→t R@1 | i→t R@5 | i→t R@10 |
|--------|---------|---------|----------|---------|---------|----------|
| Baseline | {phase_b['knn']['t2i'][1]:.1f} | {phase_b['knn']['t2i'][5]:.1f} | {phase_b['knn']['t2i'][10]:.1f} | {phase_b['knn']['i2t'][1]:.1f} | {phase_b['knn']['i2t'][5]:.1f} | {phase_b['knn']['i2t'][10]:.1f} |
| FAM    | {phase_b['fam_t2i'][1]:.1f} | {phase_b['fam_t2i'][5]:.1f} | {phase_b['fam_t2i'][10]:.1f} | {phase_b['fam_i2t'][1]:.1f} | {phase_b['fam_i2t'][5]:.1f} | {phase_b['fam_i2t'][10]:.1f} |

### FAM diagnostics
- CR at v=0.92: {pb_t['cr_v092']:.4f} ({pb_t['n_proto_v092']} prototypes)
- CR at v=0.70: {pb_t['cr_v070']:.4f} ({pb_t['n_proto_v070']} prototypes)

### Telemetry summary
- Image queries, text queries mixed 50/50 in sample
- Cross-modal sim: {pb_t['cross_modal_sim_mean']:.4f} ± {pb_t['cross_modal_sim_std']:.4f}
- Same-modal sim: {pb_t['same_modal_sim_mean']:.4f} ± {pb_t['same_modal_sim_std']:.4f}
- forward_with_confidence() — cross-modal-dominated queries: {pb_t['cross_conf_mean']:.4f} ± {pb_t['cross_conf_std']:.4f}
- forward_with_confidence() — same-modal-dominated queries: {pb_t['same_conf_mean']:.4f} ± {pb_t['same_conf_std']:.4f}

### Error analysis
- Failures (t2i): {phase_b['failures']['counts']}""")


if __name__ == "__main__":
    main()
