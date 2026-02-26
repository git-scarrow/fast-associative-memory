"""
MT-15 — Adaptive Eviction Policy Validation

Compare three eviction policies across three regimes:
  1. CIFAR-100 (well-separated, 100 classes) at 1K, 5K, 10K
  2. Stanford Dogs (fine-grained, 120 classes) at 1K, 5K, 10K
  3. ImageNet-R long-tail (200 classes, Pareto-sampled) at 1552 slots

Policies: LRU, Coverage, Adaptive (blended by coverage pressure).
Goal: Adaptive should match or beat the best single policy in each regime.
"""
import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from collections import Counter
from fast_associative_memory import FastAssociativeMemory

SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def load_cifar100():
    train = torch.load(os.path.join(ROOT, "feature_cache_vitl14/cifar100_dinov2_train.pt"),
                       weights_only=True)
    test = torch.load(os.path.join(ROOT, "feature_cache_vitl14/cifar100_dinov2_test.pt"),
                      weights_only=True)
    return train["embeds"], train["labels"], test["embeds"], test["labels"]


def load_dogs():
    train = torch.load(os.path.join(ROOT, "data/stanford_dogs/stanford_dogs_train.pt"),
                       weights_only=True)
    test = torch.load(os.path.join(ROOT, "data/stanford_dogs/stanford_dogs_test.pt"),
                      weights_only=True)
    fkey = "features" if "features" in train else "embeds"
    return train[fkey], train["labels"], test[fkey], test["labels"]


def make_long_tail(feats, labels, imbalance_ratio=0.02):
    """Subsample to Pareto long-tail distribution."""
    labels_np = labels.numpy()
    classes = np.unique(labels_np)
    n_classes = len(classes)
    indices = []
    for i, c in enumerate(classes):
        ratio = imbalance_ratio ** (i / (n_classes - 1))
        class_idx = np.where(labels_np == c)[0]
        n_keep = max(1, int(len(class_idx) * ratio))
        keep = np.random.choice(class_idx, n_keep, replace=False)
        indices.extend(keep)
    indices = np.array(indices)
    return feats[indices], labels[indices]


def run_trial(train_feats, train_labels, test_feats, test_labels,
              capacity, policy, input_dim=1024, value_dim=100,
              sort_by_class=False):
    """Run a single ingest+eval trial. Returns (accuracy, n_classes_covered, pressure_at_end)."""
    seed_all()

    use_lfu = (policy == "coverage")
    adaptive = (policy == "adaptive")

    fam = FastAssociativeMemory(
        input_dim=input_dim, value_dim=value_dim,
        core_entries=capacity, core_vigilance=0.85,
        inference_k=25, inference_temp=0.05,
        use_lfu=use_lfu,
        adaptive_eviction=adaptive,
    ).to(DEVICE)

    # Ingest order
    if sort_by_class:
        perm = train_labels.argsort()
    else:
        perm = torch.randperm(train_feats.size(0))
    fam.train()
    for i in range(0, len(perm), 256):
        idx = perm[i:i+256]
        with torch.no_grad():
            fam.learn_local(train_feats[idx].to(DEVICE), train_labels[idx].to(DEVICE))

    # Measure coverage pressure at end
    occ = fam.core_cam.occupied
    vals = fam.core_cam.values[occ].float().argmax(1)
    unique_cls, counts = vals.unique(return_counts=True)
    n_classes = len(unique_cls)
    n_singleton = (counts == 1).sum().item()
    pressure = n_singleton / max(n_classes, 1)

    # Eval on test set
    fam.eval()
    correct = 0
    total = test_feats.size(0)
    for i in range(0, total, 256):
        with torch.no_grad():
            preds = fam(test_feats[i:i+256].to(DEVICE)).argmax(1)
            correct += (preds == test_labels[i:i+256].to(DEVICE)).sum().item()

    acc = 100.0 * correct / total
    return acc, n_classes, pressure


def main():
    policies = ["lru", "coverage", "adaptive"]
    capacities = [1000, 5000, 10000]

    # === CIFAR-100 ===
    print("=" * 90)
    print("  CIFAR-100 (well-separated, 100 classes, ViT-L/14 1024-d)")
    print("=" * 90)
    c_train_f, c_train_l, c_test_f, c_test_l = load_cifar100()
    print(f"  Train: {c_train_f.shape[0]}, Test: {c_test_f.shape[0]}")
    print(f"  {'Cap':>6} {'Policy':>10} {'Acc':>8} {'Classes':>8} {'Pressure':>9}")
    print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*8} {'-'*9}")
    cifar_results = []
    for cap in capacities:
        for pol in policies:
            acc, ncls, pres = run_trial(c_train_f, c_train_l, c_test_f, c_test_l,
                                        cap, pol, value_dim=100)
            cifar_results.append((cap, pol, acc, ncls, pres))
            print(f"  {cap:>6} {pol:>10} {acc:>7.2f}% {ncls:>8} {pres:>9.3f}")

    # === Stanford Dogs ===
    print(f"\n{'=' * 90}")
    print("  Stanford Dogs (fine-grained, 120 classes, ViT-L/14 1024-d)")
    print("=" * 90)
    d_train_f, d_train_l, d_test_f, d_test_l = load_dogs()
    n_dog_classes = len(d_train_l.unique())
    print(f"  Train: {d_train_f.shape[0]}, Test: {d_test_f.shape[0]}, Classes: {n_dog_classes}")
    print(f"  {'Cap':>6} {'Policy':>10} {'Acc':>8} {'Classes':>8} {'Pressure':>9}")
    print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*8} {'-'*9}")
    dogs_results = []
    for cap in capacities:
        for pol in policies:
            acc, ncls, pres = run_trial(d_train_f, d_train_l, d_test_f, d_test_l,
                                        cap, pol, value_dim=n_dog_classes)
            dogs_results.append((cap, pol, acc, ncls, pres))
            print(f"  {cap:>6} {pol:>10} {acc:>7.2f}% {ncls:>8} {pres:>9.3f}")

    # === Dogs class-sorted (adversarial for LRU) ===
    print(f"\n{'=' * 90}")
    print("  Stanford Dogs CLASS-SORTED ingestion (adversarial)")
    print("=" * 90)
    print(f"  {'Cap':>6} {'Policy':>10} {'Acc':>8} {'Classes':>8} {'Pressure':>9}")
    print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*8} {'-'*9}")
    sorted_results = []
    for cap in capacities:
        for pol in policies:
            acc, ncls, pres = run_trial(d_train_f, d_train_l, d_test_f, d_test_l,
                                        cap, pol, value_dim=n_dog_classes,
                                        sort_by_class=True)
            sorted_results.append((cap, pol, acc, ncls, pres))
            print(f"  {cap:>6} {pol:>10} {acc:>7.2f}% {ncls:>8} {pres:>9.3f}")

    # === Summary ===
    print(f"\n{'=' * 90}")
    print("  SUMMARY: Best policy per regime")
    print("=" * 90)
    all_datasets = [("CIFAR-100 (shuffled)", cifar_results),
                    ("Dogs (shuffled)", dogs_results),
                    ("Dogs (class-sorted)", sorted_results)]
    for name, results in all_datasets:
        print(f"\n  {name}:")
        for cap in capacities:
            cap_results = [(pol, acc) for c, pol, acc, _, _ in results if c == cap]
            best_pol, best_acc = max(cap_results, key=lambda x: x[1])
            adaptive_acc = [acc for pol, acc in cap_results if pol == "adaptive"][0]
            gap = adaptive_acc - best_acc
            marker = "***" if best_pol == "adaptive" else ""
            print(f"    {cap:>6}: best={best_pol:>10} ({best_acc:.2f}%)  "
                  f"adaptive={adaptive_acc:.2f}%  gap={gap:+.2f}pp {marker}")


if __name__ == "__main__":
    main()
