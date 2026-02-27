"""
query.py — CLI stub for retrieval validation.

Usage:
    python query.py <image_path> [--config config.toml] [--top-k 10]

Embeds the query image, runs cam.forward(), and prints top-10 nearest
stored images with similarity scores and file paths from the SQLite
metadata table.
"""

import argparse
import sqlite3
import sys
import tomllib
from pathlib import Path

import torch
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Shutter-Deck FAM index.")
    parser.add_argument("image", type=Path, help="Path to query image.")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    # Build CAM and load state
    from shutter_deck.service import build_cam
    from shutter_deck.persistence import load_cam_state
    from shutter_deck.eviction import install_density_eviction

    cam = build_cam(cfg)
    install_density_eviction(cam, min_per_class=cfg["eviction"]["min_per_class"])

    state_path = Path(cfg["paths"]["state_file"])
    if not load_cam_state(cam, state_path):
        print("No persisted state found. Index is empty.", file=sys.stderr)
        sys.exit(1)

    n_occ = cam.occupied.sum().item()
    print(f"Loaded index: {n_occ} prototypes.")

    # Embed query image
    from shutter_deck.encoder import DINOv2Encoder
    enc_cfg = cfg["encoder"]
    encoder = DINOv2Encoder(
        model_name=enc_cfg["model_name"],
        device=enc_cfg.get("device", "cpu"),
        model_path=enc_cfg.get("model_path"),
        backend=enc_cfg.get("backend", "onnxruntime"),
        num_threads=enc_cfg.get("num_threads", 4),
        input_size=enc_cfg.get("input_size", 224),
    )
    encoder.warmup()
    embedding = encoder.embed_file(args.image)  # (1, D)

    # Retrieve: find top-k nearest prototypes by cosine similarity
    valid_idx = cam.occupied.nonzero(as_tuple=True)[0]
    q_norm = F.normalize(embedding.float(), dim=-1)
    sims = (q_norm @ cam._keys_norm[valid_idx].float().T).squeeze(0)  # (N_occ,)

    k = min(args.top_k, sims.numel())
    top_sims, top_locs = sims.topk(k)
    top_slots = valid_idx[top_locs]

    # Look up file paths from metadata DB
    db_path = Path(cfg["paths"]["metadata_db"])
    if not db_path.exists():
        print("Metadata DB not found. Showing slot IDs only.", file=sys.stderr)
        for i in range(k):
            print(f"  {i+1:2d}. slot={top_slots[i].item():6d}  sim={top_sims[i].item():.4f}")
        return

    conn = sqlite3.connect(str(db_path))
    print(f"\nTop-{k} nearest neighbors:")
    print(f"{'Rank':>4}  {'Slot':>6}  {'Similarity':>10}  {'File Path'}")
    print("-" * 70)

    for i in range(k):
        slot = top_slots[i].item()
        sim = top_sims[i].item()
        row = conn.execute(
            "SELECT filepath FROM ingested_images WHERE assigned_slot = ? "
            "ORDER BY timestamp DESC LIMIT 1",
            (slot,),
        ).fetchone()
        fpath = row[0] if row else "<unknown>"
        print(f"{i+1:4d}  {slot:6d}  {sim:10.4f}  {fpath}")

    conn.close()


if __name__ == "__main__":
    main()
