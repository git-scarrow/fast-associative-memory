#!/usr/bin/env python3
"""
G-42 (Corrected) — FAM Cross-Modal Retrieval with Matched Comparisons

Fixes the original G-42's invalid i2t comparison by providing two matched
evaluation tracks:

  Track A (Classification): both kNN and FAM score over image_ids.
  Track B (Retrieval):      both kNN and FAM rank individual items.

Gate logic: cross-modal merge rate is the primary kill signal.

Dataset: Flickr30k 1K test split, CLIP ViT-L/14 768-d, no adapters.
Reuses G-42 embedding cache.

Usage:
    PYTHONPATH=. .venv/bin/python benchmarks/g42_cross_modal_corrected.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from fast_associative_memory import FastAssociativeMemory

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_CACHE = Path("data/g42_embeddings")  # reuse G-42 cache
MODEL_CACHE = Path(".model_cache")
HF_CACHE = Path("data/g42_embeddings/.hf_cache")
OUT_DIR = Path("results/g42_cross_modal_corrected")

EMBED_DIM = 768
IMG_BATCH = 64
TXT_BATCH = 256
CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FAM_VIGILANCE = 0.85
FAM_K = 20
FAM_TEMP = 0.05


def set_seeds():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------------
# CLIP model loading  (identical to G-42)
# ---------------------------------------------------------------------------

def load_clip():
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
def encode_images(model, processor, images: list) -> torch.Tensor:
    all_embs = []
    for i in range(0, len(images), IMG_BATCH):
        batch_imgs = images[i:i + IMG_BATCH]
        inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(CLIP_DEVICE)
        vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
        feats = model.visual_projection(vision_out.pooler_output)
        feats = F.normalize(feats.float(), dim=-1)
        all_embs.append(feats.cpu())
        if (i // IMG_BATCH) % 10 == 0:
            print(f"    images {i}/{len(images)}", end="\r")
    print()
    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def encode_texts(model, processor, texts: list) -> torch.Tensor:
    all_embs = []
    for i in range(0, len(texts), TXT_BATCH):
        batch_txt = texts[i:i + TXT_BATCH]
        inputs = processor(text=batch_txt, return_tensors="pt", padding=True,
                           truncation=True, max_length=77).to(CLIP_DEVICE)
        text_out = model.text_model(input_ids=inputs["input_ids"],
                                    attention_mask=inputs["attention_mask"])
        feats = model.text_projection(text_out.pooler_output)
        feats = F.normalize(feats.float(), dim=-1)
        all_embs.append(feats.cpu())
        if (i // TXT_BATCH) % 10 == 0:
            print(f"    texts {i}/{len(texts)}", end="\r")
    print()
    return torch.cat(all_embs, dim=0)


# ---------------------------------------------------------------------------
# Dataset loading / embedding extraction  (reuse G-42 cache)
# ---------------------------------------------------------------------------

def load_and_embed(dataset_name: str, split: str, model, processor) -> dict:
    cache_file = EMB_CACHE / f"{dataset_name.replace('/', '_')}_{split}.pt"
    if cache_file.exists():
        print(f"  Loading cached embeddings from {cache_file}")
        return torch.load(cache_file, map_location="cpu", weights_only=True)

    from datasets import load_dataset
    print(f"  Loading {dataset_name} {split} split ...")
    ds = load_dataset(dataset_name, split=split, cache_dir=str(HF_CACHE))

    images, all_captions, cap_imgid = [], [], []
    for i, ex in enumerate(ds):
        img = ex["jpg"].convert("RGB")
        images.append(img)
        caps = ex["txt"].strip().split("\n")
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
        "img_embs": img_embs.half(),
        "cap_embs": cap_embs.half(),
        "cap_imgid": torch.tensor(cap_imgid, dtype=torch.long),
        "N_img": len(images),
        "N_cap": len(all_captions),
        "N_caps_per_img": 5,
    }
    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_file)
    print(f"  Saved embeddings to {cache_file}")
    return data


# ---------------------------------------------------------------------------
# Recall@K helper
# ---------------------------------------------------------------------------

def recall_at_k(scores: torch.Tensor, targets: torch.Tensor,
                ks=(1, 5, 10)) -> dict:
    """
    R@K.  scores: (N_q, N_cand), targets: (N_q,) or (N_q, max_correct) with -1 pad.
    """
    ranked = scores.argsort(dim=1, descending=True)
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)
    results = {}
    for k in ks:
        top_k = ranked[:, :k]
        hit = (top_k.unsqueeze(2) == targets.unsqueeze(1)).any(dim=(1, 2)).sum().item()
        results[k] = hit / scores.size(0) * 100.0
    return results


# ---------------------------------------------------------------------------
# FAM setup
# ---------------------------------------------------------------------------

def build_fam(N_img: int, max_entries: int) -> FastAssociativeMemory:
    return FastAssociativeMemory(
        input_dim=EMBED_DIM,
        value_dim=N_img,
        core_entries=max_entries,
        core_vigilance=FAM_VIGILANCE,
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=FAM_K,
        inference_temp=FAM_TEMP,
        whitening_dim=0,
        use_lfu=False,
        adaptive_eviction=True,
        adapter=None,
        nstp=None,
        track_provenance=True,
    ).to(DEVICE)


@torch.no_grad()
def write_fam_interleaved(fam, img_embs, cap_embs, cap_imgid, N_img):
    caps_for_img = defaultdict(list)
    for ci, img_id in enumerate(cap_imgid.tolist()):
        caps_for_img[img_id].append(ci)

    seq_feats, seq_labels = [], []
    # Per-position provenance record ids, modality-tagged so the true write
    # history can be partitioned back into image vs caption origins:
    #   ("img", image_id)        — an image write
    #   ("cap", cap_idx, image_id) — a caption write
    seq_records: list[tuple] = []
    for img_id in range(N_img):
        seq_feats.append(img_embs[img_id])
        seq_labels.append(img_id)
        seq_records.append(("img", img_id))
        for ci in caps_for_img[img_id]:
            seq_feats.append(cap_embs[ci])
            seq_labels.append(img_id)
            seq_records.append(("cap", ci, img_id))

    seq_feats = torch.stack(seq_feats, dim=0)
    seq_labels = torch.tensor(seq_labels, dtype=torch.long)
    targets = torch.zeros(len(seq_labels), N_img)
    targets.scatter_(1, seq_labels.unsqueeze(1), 1.0)

    batch = 256
    for i in range(0, len(seq_feats), batch):
        bx = seq_feats[i:i + batch].to(DEVICE)
        bt = targets[i:i + batch].to(DEVICE)
        # Pass record_ids so slot_records captures the actual write/merge
        # history; provenance is then read natively, not reconstructed.
        fam.core_cam.learn_local(fam._project(bx), bt,
                                 record_ids=seq_records[i:i + batch])
        if i % 2000 == 0:
            n_occ = int(fam.core_cam.occupied.sum().item())
            print(f"    write {i}/{len(seq_feats)} | prototypes: {n_occ}", end="\r")

    n_occ = int(fam.core_cam.occupied.sum().item())
    print(f"\n    Wrote {len(seq_feats)} items -> {n_occ} prototypes")
    return len(seq_feats)


# ---------------------------------------------------------------------------
# Provenance tracking  (which prototypes hold image vs text items)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_provenance(fam):
    """Read each prototype's TRUE write/merge history from native slot_records.

    This replaces the former O(N*P) reconstruction that re-assigned every image
    and caption to its current nearest prototype (argmax cosine). That answered
    "where would this item retrieve to now"; slot_records answers "which writes
    actually formed this prototype" — the question the cross-modal merge metric
    intends. The two differ: a coincidental nearest-neighbour is not a merge,
    and an evicted item's record is dropped rather than reattributed.

    Records were written modality-tagged in write_fam_interleaved:
      ("img", image_id)          -> proto_img_ids[slot]
      ("cap", cap_idx, image_id) -> proto_cap_ids[slot]
    """
    valid_idx = fam.core_cam.occupied.nonzero(as_tuple=True)[0]

    proto_img_ids: dict[int, set] = defaultdict(set)
    proto_cap_ids: dict[int, set] = defaultdict(set)
    for slot in valid_idx.tolist():
        for rec in fam.core_cam.records_for(slot):
            kind = rec[0]
            if kind == "img":
                proto_img_ids[slot].add(rec[1])           # image_id
            elif kind == "cap":
                proto_cap_ids[slot].add((rec[1], rec[2]))  # (cap_idx, image_id)

    return {
        "proto_img_ids": dict(proto_img_ids),
        "proto_cap_ids": dict(proto_cap_ids),
        "valid_idx": valid_idx.cpu(),
    }


def compute_diagnostics(fam, provenance, N_img, N_total_written):
    n_proto = int(fam.core_cam.occupied.sum().item())
    cr = 1.0 - (n_proto / N_total_written)
    proto_img_ids = provenance["proto_img_ids"]
    proto_cap_ids = provenance["proto_cap_ids"]
    valid_idx = provenance["valid_idx"].tolist()

    n_image_only = n_text_only = n_cross_modal = 0
    for slot in valid_idx:
        has_img = slot in proto_img_ids and len(proto_img_ids[slot]) > 0
        has_cap = slot in proto_cap_ids and len(proto_cap_ids[slot]) > 0
        if has_img and has_cap:
            n_cross_modal += 1
        elif has_img:
            n_image_only += 1
        elif has_cap:
            n_text_only += 1

    return {
        "cr": cr,
        "n_proto": n_proto,
        "n_image_only": n_image_only,
        "n_text_only": n_text_only,
        "n_cross_modal": n_cross_modal,
        "cross_modal_merge_rate": n_cross_modal / n_proto if n_proto > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# FAM retrieval helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def fam_scores_over_imgids(fam, queries, N_img, batch_size=256):
    """FAM forward() -> (N_q, N_img) vote distribution over image_ids."""
    queries = queries.float()
    parts = []
    for i in range(0, len(queries), batch_size):
        bq = queries[i:i + batch_size].to(DEVICE)
        bq_proj = fam._project(bq)
        parts.append(fam.core_cam(bq_proj).cpu())
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Track A — Classification  (both systems score over image_ids)
# ---------------------------------------------------------------------------

def run_track_a(img_embs, cap_embs, cap_imgid, N_img, fam):
    """
    Track A: apples-to-apples on image_ids.

    t2i: caption -> rank image_ids
        kNN: cosine(caption, each image) -> per-image_id score
        FAM: forward(caption) -> vote distribution

    i2t: image -> rank image_ids
        kNN: cosine(image, each caption) -> aggregate per image_id (max)
        FAM: forward(image) -> vote distribution
    """
    img = img_embs.float()
    cap = cap_embs.float()

    # --- t2i: caption -> image_ids ---
    print("  Track A | kNN t2i ...")
    knn_t2i = _batched_matmul(cap, img)           # (N_cap, N_img)
    t2i_targets = cap_imgid.long()

    print("  Track A | FAM t2i ...")
    fam_t2i = fam_scores_over_imgids(fam, cap_embs, N_img)

    knn_t2i_metrics = recall_at_k(knn_t2i, t2i_targets)
    fam_t2i_metrics = recall_at_k(fam_t2i, t2i_targets)

    # --- i2t: image -> image_ids (kNN aggregates caption sims per image_id) ---
    print("  Track A | kNN i2t (aggregated to image_ids) ...")
    knn_img_cap = _batched_matmul(img, cap)        # (N_img, N_cap)
    # Aggregate: for each query image, score each image_id by max cosine
    # across that image_id's 5 captions.
    knn_i2t_imgid = _aggregate_cap_scores_to_imgids(
        knn_img_cap, cap_imgid, N_img
    )                                               # (N_img, N_img)
    i2t_targets = torch.arange(N_img, dtype=torch.long)

    print("  Track A | FAM i2t ...")
    fam_i2t = fam_scores_over_imgids(fam, img_embs, N_img)

    knn_i2t_metrics = recall_at_k(knn_i2t_imgid, i2t_targets)
    fam_i2t_metrics = recall_at_k(fam_i2t, i2t_targets)

    return {
        "knn_t2i": knn_t2i_metrics,
        "fam_t2i": fam_t2i_metrics,
        "knn_i2t": knn_i2t_metrics,
        "fam_i2t": fam_i2t_metrics,
    }


# ---------------------------------------------------------------------------
# Track B — Retrieval  (both systems rank individual items)
# ---------------------------------------------------------------------------

def run_track_b(img_embs, cap_embs, cap_imgid, N_img, fam):
    """
    Track B: oranges-to-oranges on individual items.

    t2i: same as Track A (images are 1:1 with image_ids).

    i2t: image -> rank individual captions
        kNN: cosine(image, each caption) -> rank captions directly
        FAM: forward(image) -> vote over image_ids -> each caption inherits
             its image_id's vote score -> rank captions
    """
    img = img_embs.float()
    cap = cap_embs.float()

    # --- t2i: identical to Track A ---
    print("  Track B | t2i (same as Track A) ...")
    knn_t2i = _batched_matmul(cap, img)
    t2i_targets = cap_imgid.long()
    fam_t2i = fam_scores_over_imgids(fam, cap_embs, N_img)

    knn_t2i_metrics = recall_at_k(knn_t2i, t2i_targets)
    fam_t2i_metrics = recall_at_k(fam_t2i, t2i_targets)

    # --- i2t: image -> rank individual captions ---
    print("  Track B | kNN i2t (rank captions) ...")
    knn_i2t_cap = _batched_matmul(img, cap)         # (N_img, N_cap)

    print("  Track B | FAM i2t (vote-then-expand to captions) ...")
    fam_i2t_imgid = fam_scores_over_imgids(fam, img_embs, N_img)  # (N_img, N_img)
    # Expand: each caption gets its image_id's vote score
    fam_i2t_cap = fam_i2t_imgid[:, cap_imgid.long()]              # (N_img, N_cap)

    # Targets: for each query image, correct captions are those with matching image_id
    caps_per_img = 5
    i2t_targets = torch.full((N_img, caps_per_img), -1, dtype=torch.long)
    for img_id in range(N_img):
        idxs = (cap_imgid == img_id).nonzero(as_tuple=True)[0]
        n = min(len(idxs), caps_per_img)
        i2t_targets[img_id, :n] = idxs[:n]

    knn_i2t_metrics = recall_at_k(knn_i2t_cap, i2t_targets)
    fam_i2t_metrics = recall_at_k(fam_i2t_cap, i2t_targets)

    return {
        "knn_t2i": knn_t2i_metrics,
        "fam_t2i": fam_t2i_metrics,
        "knn_i2t": knn_i2t_metrics,
        "fam_i2t": fam_i2t_metrics,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batched_matmul(a: torch.Tensor, b: torch.Tensor, batch=512) -> torch.Tensor:
    """(N, D) @ (M, D).T -> (N, M) in batches."""
    parts = []
    for i in range(0, len(a), batch):
        parts.append((a[i:i + batch] @ b.T).cpu())
    return torch.cat(parts, dim=0)


def _aggregate_cap_scores_to_imgids(
    scores: torch.Tensor, cap_imgid: torch.Tensor, N_img: int
) -> torch.Tensor:
    """
    (N_img, N_cap) caption scores -> (N_img, N_img) image_id scores.
    Aggregation: max cosine across each image_id's captions.
    """
    out = torch.full((scores.size(0), N_img), -1e9)
    for img_id in range(N_img):
        mask = cap_imgid == img_id
        if mask.any():
            out[:, img_id] = scores[:, mask].max(dim=1).values
    return out


# ---------------------------------------------------------------------------
# Failure mode tagging
# ---------------------------------------------------------------------------

def classify_failures(fam_scores, knn_scores, targets, provenance, mode="t2i"):
    ranked = fam_scores.argsort(dim=1, descending=True)
    proto_img_ids = provenance["proto_img_ids"]
    proto_cap_ids = provenance["proto_cap_ids"]

    counts = {"modality-gap": 0, "condensation-fragmentation": 0, "vote-collapse": 0}
    incorrect_mask = ranked[:, 0] != (targets if targets.dim() == 1 else targets[:, 0])
    incorrect_idxs = incorrect_mask.nonzero(as_tuple=True)[0].tolist()

    for qi in incorrect_idxs[:200]:
        correct_id = (targets[qi] if targets.dim() == 1 else targets[qi, 0]).item()
        correct_protos = [s for s in proto_img_ids if correct_id in proto_img_ids[s]]

        if len(correct_protos) == 0 or len(correct_protos) > 1:
            counts["condensation-fragmentation"] += 1
        else:
            counts["modality-gap"] += 1

    return {"counts": counts, "n_incorrect": len(incorrect_idxs)}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(track_a, track_b, diag, failures, kill_reason):
    def fmt(m):
        return f"R@1={m[1]:.1f}  R@5={m[5]:.1f}  R@10={m[10]:.1f}"

    print("\n" + "=" * 70)
    print("G-42 (CORRECTED) FINAL REPORT")
    print("=" * 70)

    print(f"\n## Decision: {kill_reason}")

    print(f"\n## Diagnostics")
    print(f"  CR:                    {diag['cr']:.4f}")
    print(f"  Prototypes:            {diag['n_proto']}")
    print(f"  Image-only:            {diag['n_image_only']}")
    print(f"  Text-only:             {diag['n_text_only']}")
    print(f"  Cross-modal merged:    {diag['n_cross_modal']}")
    print(f"  Cross-modal merge rate: {diag['cross_modal_merge_rate']:.4f}")

    print(f"\n## Track A — Classification (both score image_ids)")
    print(f"  {'':20s} {'t2i':30s} {'i2t':30s}")
    print(f"  {'kNN':20s} {fmt(track_a['knn_t2i']):30s} {fmt(track_a['knn_i2t']):30s}")
    print(f"  {'FAM':20s} {fmt(track_a['fam_t2i']):30s} {fmt(track_a['fam_i2t']):30s}")

    print(f"\n## Track B — Retrieval (both rank individual items)")
    print(f"  {'':20s} {'t2i (image_ids)':30s} {'i2t (captions)':30s}")
    print(f"  {'kNN':20s} {fmt(track_b['knn_t2i']):30s} {fmt(track_b['knn_i2t']):30s}")
    print(f"  {'FAM':20s} {fmt(track_b['fam_t2i']):30s} {fmt(track_b['fam_i2t']):30s}")

    print(f"\n## Failure modes (Track A t2i)")
    print(f"  {failures['counts']}")
    print(f"  Total incorrect: {failures['n_incorrect']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seeds()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EMB_CACHE.mkdir(parents=True, exist_ok=True)

    model, processor = load_clip()

    print("\n" + "=" * 60)
    print("  G-42 (Corrected): Flickr30k 1K — Corrected Cross-Modal Comparison")
    print("=" * 60)

    data = load_and_embed("clip-benchmark/wds_flickr30k", "test", model, processor)
    img_embs = data["img_embs"].float()
    cap_embs = data["cap_embs"].float()
    cap_imgid = data["cap_imgid"]
    N_img = int(data["N_img"])
    N_cap = int(data["N_cap"])
    N_total = N_img + N_cap
    print(f"  {N_img} images, {N_cap} captions")

    # Build and populate FAM
    max_entries = min(N_total, max(N_img * 4, 4000))
    fam = build_fam(N_img, max_entries)
    set_seeds()
    n_written = write_fam_interleaved(fam, img_embs, cap_embs, cap_imgid, N_img)

    # Provenance + diagnostics
    print("  Computing provenance ...")
    prov = compute_provenance(fam)
    diag = compute_diagnostics(fam, prov, N_img, n_written)

    # Primary kill signal: cross-modal merge rate
    merge_rate = diag["cross_modal_merge_rate"]
    if merge_rate == 0.0:
        kill_reason = (
            f"KILL — cross-modal merge rate = 0.0 "
            f"({diag['n_image_only']} image-only + {diag['n_text_only']} text-only prototypes, "
            f"0 cross-modal). FAM cannot bridge the modality gap at v={FAM_VIGILANCE}."
        )
    else:
        kill_reason = f"PASS — cross-modal merge rate = {merge_rate:.4f}"

    print(f"\n  Gate: {kill_reason}")

    # Run both tracks (even if killed, for the record)
    print("\n  Running Track A (classification) ...")
    track_a = run_track_a(img_embs, cap_embs, cap_imgid, N_img, fam)

    print("\n  Running Track B (retrieval) ...")
    track_b = run_track_b(img_embs, cap_embs, cap_imgid, N_img, fam)

    # Failure analysis on Track A t2i
    print("  Classifying failures ...")
    knn_t2i_scores = _batched_matmul(cap_embs.float(), img_embs.float())
    failures = classify_failures(
        fam_scores_over_imgids(fam, cap_embs, N_img),
        knn_t2i_scores,
        cap_imgid.long(),
        prov,
    )

    print_report(track_a, track_b, diag, failures, kill_reason)

    # Save results
    summary = {
        "decision": kill_reason,
        "diagnostics": diag,
        "track_a": {k: {str(kk): vv for kk, vv in v.items()} for k, v in track_a.items()},
        "track_b": {k: {str(kk): vv for kk, vv in v.items()} for k, v in track_b.items()},
        "failures": {"counts": failures["counts"], "n_incorrect": failures["n_incorrect"]},
        "config": {
            "vigilance": FAM_VIGILANCE,
            "inference_k": FAM_K,
            "inference_temp": FAM_TEMP,
            "max_entries": max_entries,
            "N_img": N_img,
            "N_cap": N_cap,
        },
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
