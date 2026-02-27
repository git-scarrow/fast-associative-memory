#!/usr/bin/env python3
"""
extract_20newsgroups.py — Extract sentence-transformer embeddings for 20 Newsgroups.

Uses all-MiniLM-L6-v2 (384-d) to encode each document into a dense vector.
Strips headers, footers, and quotes (standard decontamination).

Output:
    data/20newsgroups/20newsgroups_train.pt  — {"embeds": (N, 384), "labels": (N,)}
    data/20newsgroups/20newsgroups_test.pt

Usage:
    PYTHONPATH=. .venv/bin/python extract_20newsgroups.py
"""
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups

MODEL_NAME = "all-MiniLM-L6-v2"
OUT_DIR = Path("data/20newsgroups")
BATCH_SIZE = 64


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading sentence-transformer: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    for split in ("train", "test"):
        print(f"\nFetching 20 Newsgroups ({split})...")
        data = fetch_20newsgroups(
            subset=split,
            remove=("headers", "footers", "quotes"),
        )
        texts = data.data
        labels = data.target
        print(f"  Documents: {len(texts)}, Classes: {len(data.target_names)}")
        print(f"  Categories: {data.target_names}")

        print(f"  Encoding with {MODEL_NAME}...")
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=device,
        )

        embeds_t = torch.from_numpy(embeddings).float()
        labels_t = torch.from_numpy(labels).long()
        print(f"  Embedding shape: {embeds_t.shape}")

        out_path = OUT_DIR / f"20newsgroups_{split}.pt"
        torch.save({"embeds": embeds_t, "labels": labels_t}, out_path)
        print(f"  Saved to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
