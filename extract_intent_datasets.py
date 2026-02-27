#!/usr/bin/env python3
"""
extract_intent_datasets.py — Extract sentence-transformer embeddings for
Banking77 and CLINC-150 intent detection datasets.

Uses all-MiniLM-L6-v2 (384-d), same as G-40 (20 Newsgroups).

Output:
    data/banking77/banking77_train.pt   — {"embeds": (N, 384), "labels": (N,)}
    data/banking77/banking77_test.pt
    data/clinc150/clinc150_train.pt
    data/clinc150/clinc150_test.pt

Usage:
    PYTHONPATH=. .venv/bin/python extract_intent_datasets.py
"""
from pathlib import Path

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


def extract_banking77(model, device):
    out_dir = Path("data/banking77")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n  Loading Banking77...")
    ds = load_dataset("banking77")
    print(f"    Train: {len(ds['train'])}, Test: {len(ds['test'])}")
    print(f"    Classes: 77")

    for split in ("train", "test"):
        texts = ds[split]["text"]
        labels = ds[split]["label"]
        print(f"    Encoding {split} ({len(texts)} samples)...")
        embeddings = model.encode(
            texts, batch_size=BATCH_SIZE, show_progress_bar=True,
            convert_to_numpy=True, device=device,
        )
        embeds_t = torch.from_numpy(embeddings).float()
        labels_t = torch.tensor(labels, dtype=torch.long)
        out_path = out_dir / f"banking77_{split}.pt"
        torch.save({"embeds": embeds_t, "labels": labels_t}, out_path)
        print(f"    Saved {out_path} — shape {embeds_t.shape}")


def extract_clinc150(model, device):
    out_dir = Path("data/clinc150")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n  Loading CLINC-150 (plus variant, includes OOS)...")
    ds = load_dataset("clinc_oos", "plus")
    print(f"    Train: {len(ds['train'])}, Val: {len(ds['validation'])}, Test: {len(ds['test'])}")

    # CLINC has 150 in-scope intents + 1 OOS (intent=42 is 'oos')
    # We include OOS as a class — FAM should handle it
    all_intents = sorted(set(ds["train"]["intent"]))
    n_classes = len(all_intents)
    print(f"    Total intent classes: {n_classes} (150 in-scope + OOS)")

    for split, split_name in [("train", "train"), ("test", "test")]:
        texts = ds[split]["text"]
        labels = ds[split]["intent"]
        print(f"    Encoding {split_name} ({len(texts)} samples)...")
        embeddings = model.encode(
            texts, batch_size=BATCH_SIZE, show_progress_bar=True,
            convert_to_numpy=True, device=device,
        )
        embeds_t = torch.from_numpy(embeddings).float()
        labels_t = torch.tensor(labels, dtype=torch.long)
        out_path = out_dir / f"clinc150_{split_name}.pt"
        torch.save({"embeds": embeds_t, "labels": labels_t}, out_path)
        print(f"    Saved {out_path} — shape {embeds_t.shape}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}")
    model = SentenceTransformer(MODEL_NAME)

    extract_banking77(model, device)
    extract_clinc150(model, device)
    print("\nDone.")


if __name__ == "__main__":
    main()
