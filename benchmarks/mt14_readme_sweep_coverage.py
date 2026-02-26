"""
MT-14 — README Constrained Sweep (Coverage Eviction)

Re-run the README headline capacity sweep (1K, 5K, 10K, 25K, 50K) on CIFAR-100
with coverage-aware eviction (use_lfu=True on clean codebase).

Uses pre-extracted ViT-L/14 1024-d features from feature_cache_vitl14/.
"""
import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from fast_associative_memory import FastAssociativeMemory

# Determinism
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Old README numbers (LFU eviction, pre-fix)
OLD_LFU = {
    1000:  82.56,
    2000:  85.64,
    5000:  87.81,
    10000: 89.41,
    25000: 90.96,
    50000: 91.31,
}
CORESET_CEILING = 91.29  # offline soft k-NN at 50K

CAPACITIES = [1000, 5000, 10000, 25000, 50000]


def load_features():
    train = torch.load(os.path.join(ROOT, "feature_cache_vitl14/cifar100_dinov2_train.pt"),
                       weights_only=True)
    test = torch.load(os.path.join(ROOT, "feature_cache_vitl14/cifar100_dinov2_test.pt"),
                      weights_only=True)
    return train["embeds"], train["labels"], test["embeds"], test["labels"]


def run_sweep():
    train_feats, train_labels, test_feats, test_labels = load_features()
    print(f"Train: {train_feats.shape}, Test: {test_feats.shape}")
    print(f"Device: {DEVICE}\n")

    results = []

    for cap in CAPACITIES:
        # Reset seed for each capacity point
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        fam = FastAssociativeMemory(
            input_dim=1024, value_dim=100,
            core_entries=cap, core_vigilance=0.85,
            inference_k=25, inference_temp=0.05,
            use_lfu=True,  # coverage-aware eviction
        ).to(DEVICE)

        # Ingest (shuffled order)
        perm = torch.randperm(train_feats.size(0))
        fam.train()
        t0 = time.perf_counter()
        for i in range(0, len(perm), 256):
            batch_idx = perm[i:i+256]
            with torch.no_grad():
                fam.learn_local(
                    train_feats[batch_idx].to(DEVICE),
                    train_labels[batch_idx].to(DEVICE),
                )
        train_time = time.perf_counter() - t0

        # Eval on test set
        fam.eval()
        correct = 0
        total = test_feats.size(0)
        for i in range(0, total, 256):
            with torch.no_grad():
                preds = fam(test_feats[i:i+256].to(DEVICE)).argmax(1)
                correct += (preds == test_labels[i:i+256].to(DEVICE)).sum().item()

        acc = 100.0 * correct / total
        protos = fam.core_cam.occupied.sum().item()
        old = OLD_LFU.get(cap, None)
        delta = (acc - old) if old else None

        results.append((cap, acc, protos, train_time, old, delta))
        print(f"  Cap={cap:>6}  Acc={acc:.2f}%  Protos={protos:>6}  "
              f"Time={train_time:.2f}s  Old={old}%  Δ={delta:+.2f}pp" if delta else
              f"  Cap={cap:>6}  Acc={acc:.2f}%  Protos={protos:>6}  Time={train_time:.2f}s")

    # Summary table
    print(f"\n{'='*85}")
    print(f"  MT-14 — README Constrained Sweep: Coverage Eviction vs LFU")
    print(f"{'='*85}")
    print(f"  {'Capacity':>8} {'Coverage Acc':>12} {'Old LFU Acc':>12} {'Delta':>8} "
          f"{'Protos':>7} {'Ceiling Gap':>12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*8} {'-'*7} {'-'*12}")
    for cap, acc, protos, t, old, delta in results:
        gap = CORESET_CEILING - acc
        old_gap = CORESET_CEILING - old if old else None
        closure = ((old_gap - gap) / old_gap * 100) if old_gap and old_gap > 0 else None
        closure_str = f"{closure:.0f}%" if closure is not None else "N/A"
        print(f"  {cap:>8,} {acc:>11.2f}% {old:>11.2f}% {delta:>+7.2f}pp "
              f"{protos:>7,} {closure_str:>12}")
    print(f"{'='*85}")
    print(f"  Coreset ceiling (offline soft k-NN @ 50K): {CORESET_CEILING}%")


if __name__ == "__main__":
    run_sweep()
