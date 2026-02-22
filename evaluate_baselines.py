#!/usr/bin/env python3
"""
evaluate_baselines.py — Fair baseline comparison for Online Soft-kNN.

Compares FastAssociativeMemory (which condenses 50K training samples into
~48K prototypes via online class-conditional clustering) against:
  1. Standard K=1 Nearest Neighbor on the full 50K training set
  2. Weighted K=25, τ=0.05 k-NN on the full 50K training set

All methods use pre-cached DINOv2 ViT-L/14 (1024-d) features for CIFAR-100.
"""

import torch
import torch.nn.functional as F
import time

CACHE_DIR = "./feature_cache_vitl14"
EMBED_DIM = 1024
NUM_CLASSES = 100


def load_features(device):
    """Load pre-cached ViT-L/14 features for CIFAR-100."""
    train = torch.load(f"{CACHE_DIR}/cifar100_dinov2_train.pt",
                       map_location=device, weights_only=True)
    test = torch.load(f"{CACHE_DIR}/cifar100_dinov2_test.pt",
                      map_location=device, weights_only=True)
    return (train["embeds"].float(), train["labels"].long(),
            test["embeds"].float(), test["labels"].long())


@torch.no_grad()
def eval_1nn(train_embeds, train_labels, test_embeds, test_labels, device):
    """Standard K=1 Nearest Neighbor (cosine similarity)."""
    train_norm = F.normalize(train_embeds, dim=-1).to(device)
    test_norm = F.normalize(test_embeds, dim=-1).to(device)
    train_labs = train_labels.to(device)
    test_labs = test_labels.to(device)

    correct = 0
    for i in range(0, len(test_norm), 512):
        q = test_norm[i:i + 512]
        sims = q @ train_norm.T                             # (B, 50000)
        nn_idx = sims.argmax(dim=1)
        correct += (train_labs[nn_idx] == test_labs[i:i + 512]).sum().item()

    return 100.0 * correct / len(test_labels)


@torch.no_grad()
def eval_weighted_knn(train_embeds, train_labels, test_embeds, test_labels,
                      device, k=25, tau=0.05):
    """Weighted k-NN: softmax(sim/τ) weighted vote over one-hot labels."""
    train_norm = F.normalize(train_embeds, dim=-1).to(device)
    test_norm = F.normalize(test_embeds, dim=-1).to(device)
    train_labs = train_labels.to(device)
    test_labs = test_labels.to(device)

    # Pre-compute one-hot for all training labels
    train_onehot = F.one_hot(train_labs, NUM_CLASSES).float()  # (50000, 100)

    correct = 0
    for i in range(0, len(test_norm), 512):
        q = test_norm[i:i + 512]
        sims = q @ train_norm.T                                # (B, 50000)
        topk_sims, topk_idx = sims.topk(k, dim=1)             # (B, K)
        weights = F.softmax(topk_sims / tau, dim=-1)           # (B, K)
        nn_onehot = train_onehot[topk_idx]                     # (B, K, 100)
        voted = (weights.unsqueeze(-1) * nn_onehot).sum(dim=1) # (B, 100)
        preds = voted.argmax(dim=1)
        correct += (preds == test_labs[i:i + 512]).sum().item()

    return 100.0 * correct / len(test_labels)


@torch.no_grad()
def eval_fast_assoc(train_embeds, train_labels, test_embeds, test_labels, device):
    """FastAssociativeMemory: online ingest 50K, then evaluate on 10K test."""
    from fast_associative_memory import FastAssociativeMemory

    mem = FastAssociativeMemory(
        input_dim=EMBED_DIM,
        value_dim=NUM_CLASSES,
        core_entries=50000,
        core_vigilance=0.85,
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=25,
        inference_temp=0.05,
        use_lfu=True,
    ).to(device)

    # Online ingestion
    x_train = train_embeds.to(device)
    y_train = train_labels.to(device)
    for i in range(0, len(x_train), 256):
        mem.learn_local(x_train[i:i + 256], y_train[i:i + 256])

    n_occupied = mem.core_cam.occupied.sum().item()

    # Evaluate
    x_test = test_embeds.to(device)
    y_test = test_labels.to(device)
    correct = 0
    for i in range(0, len(x_test), 512):
        logits = mem(x_test[i:i + 512])
        correct += (logits.argmax(dim=1) == y_test[i:i + 512]).sum().item()

    acc = 100.0 * correct / len(test_labels)
    stats = mem.core_cam.get_stats()
    return acc, n_occupied, stats


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading features from {CACHE_DIR}...")
    train_e, train_l, test_e, test_l = load_features(device)
    print(f"  Train: {train_e.shape}, Test: {test_e.shape}")

    print("\n[1/3] 1-NN (K=1, cosine)...")
    t0 = time.time()
    acc_1nn = eval_1nn(train_e, train_l, test_e, test_l, device)
    t1 = time.time()
    print(f"  {acc_1nn:.2f}%  ({t1 - t0:.1f}s)")

    print("\n[2/3] Weighted k-NN (K=25, tau=0.05)...")
    t0 = time.time()
    acc_wknn = eval_weighted_knn(train_e, train_l, test_e, test_l, device)
    t1 = time.time()
    print(f"  {acc_wknn:.2f}%  ({t1 - t0:.1f}s)")

    print("\n[3/3] FastAssociativeMemory (online, 50K slots, adaptive EMA)...")
    t0 = time.time()
    acc_fam, n_protos, fam_stats = eval_fast_assoc(train_e, train_l, test_e, test_l, device)
    t1 = time.time()
    print(f"  {acc_fam:.2f}%  ({t1 - t0:.1f}s, {n_protos} prototypes)")
    print(f"  avg_hit_count={fam_stats['avg_hit_count']:.2f}  avg_class_var={fam_stats['avg_class_var']:.4f}")

    print(f"\n{'=' * 60}")
    print(f"  CIFAR-100 — ViT-L/14 Feature Baselines (full 50K train)")
    print(f"{'=' * 60}")
    print(f"  {'Method':<40} {'Accuracy':>8}")
    print(f"  {'-' * 50}")
    print(f"  {'1-NN (K=1, cosine)':<40} {acc_1nn:>7.2f}%")
    print(f"  {'Weighted k-NN (K=25, tau=0.05)':<40} {acc_wknn:>7.2f}%")
    print(f"  {'FastAssociativeMemory (adaptive EMA, 50K)':<40} {acc_fam:>7.2f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
