#!/usr/bin/env python3
"""
benchmark_sleep.py — Split-CIFAR-100 (10 tasks × 10 classes) sleep consolidation diagnostic.

Measures accuracy before/after sleep() and reports collision counts to determine
whether cross-class prototype collision is the accuracy bottleneck.

Decision rule:
  - collisions_initial < 50  → prototype placement hypothesis falsified → pivot to constrained RanPAC
  - collisions_initial in hundreds + accuracy jump → found the bottleneck
"""

import torch
import torch.nn.functional as F
import time

from fast_associative_memory import FastAssociativeMemory

CACHE_DIR = "./feature_cache_vitl14"
EMBED_DIM = 1024
NUM_CLASSES = 100
NUM_TASKS = 10
CLASSES_PER_TASK = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_features(device):
    train = torch.load(f"{CACHE_DIR}/cifar100_dinov2_train.pt",
                       map_location=device, weights_only=True)
    test = torch.load(f"{CACHE_DIR}/cifar100_dinov2_test.pt",
                      map_location=device, weights_only=True)
    return (train["embeds"].float(), train["labels"].long(),
            test["embeds"].float(), test["labels"].long())


@torch.no_grad()
def run_benchmark():
    device = torch.device(DEVICE)
    print(f"Device: {device}")
    print(f"Loading features from {CACHE_DIR}...")
    train_e, train_l, test_e, test_l = load_features(device)
    print(f"  Train: {train_e.shape}, Test: {test_e.shape}")

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

    # --- Incremental learning: 10 tasks × 10 classes ---
    task_splits = [(i * CLASSES_PER_TASK, (i + 1) * CLASSES_PER_TASK) for i in range(NUM_TASKS)]

    for tid, (c_lo, c_hi) in enumerate(task_splits):
        mask = (train_l >= c_lo) & (train_l < c_hi)
        x_task = train_e[mask].to(device)
        y_task = train_l[mask].to(device)

        t0 = time.time()
        for i in range(0, len(x_task), 256):
            mem.learn_local(x_task[i:i+256], y_task[i:i+256])
        t1 = time.time()
        print(f"  Task {tid} (classes {c_lo}-{c_hi-1}): {len(x_task)} samples, {t1-t0:.1f}s")

    n_occ = mem.core_cam.occupied.sum().item()
    print(f"\nPrototypes after all tasks: {n_occ}/{mem.core_cam.max_entries}")

    # --- Evaluate BEFORE sleep ---
    def evaluate():
        correct = 0
        for i in range(0, len(test_e), 512):
            logits = mem(test_e[i:i+512].to(device))
            correct += (logits.argmax(1) == test_l[i:i+512].to(device)).sum().item()
        return 100.0 * correct / len(test_l)

    acc_before = evaluate()
    print(f"\nAccuracy BEFORE sleep: {acc_before:.2f}%")

    # --- Class imbalance diagnostic ---
    occ_idx = mem.core_cam.occupied.nonzero(as_tuple=True)[0]
    occ_values = mem.core_cam.values[occ_idx].float()
    class_counts = occ_values.sum(dim=0)  # one-hot sums
    print(f"\nPrototype class distribution:")
    print(f"  mean={class_counts.mean():.1f}  std={class_counts.std():.1f}  "
          f"min={class_counts.min():.0f}  max={class_counts.max():.0f}")

    # --- Sleep consolidation ---
    print(f"\nRunning sleep(verbose=True)...")
    t0 = time.time()
    sleep_result = mem.core_cam.sleep(verbose=True)
    t1 = time.time()
    print(f"Sleep completed in {t1-t0:.1f}s")
    print(f"  epochs:             {sleep_result['epochs']}")
    print(f"  collisions_initial: {sleep_result['collisions_initial']}")
    print(f"  collisions_final:   {sleep_result['collisions_final']}")
    print(f"  keys_modified:      {sleep_result['keys_modified']}")

    # --- Evaluate AFTER sleep ---
    acc_after = evaluate()
    print(f"\nAccuracy AFTER sleep:  {acc_after:.2f}%")
    delta = acc_after - acc_before
    print(f"Accuracy delta:        {delta:+.2f}%")

    # --- Decision ---
    print(f"\n{'='*60}")
    ci = sleep_result['collisions_initial']
    if ci < 50:
        print(f"  collisions_initial={ci} < 50")
        print(f"  → Prototype placement hypothesis FALSIFIED")
        print(f"  → Pivot to constrained RanPAC")
    elif delta > 0.5:
        print(f"  collisions_initial={ci}, accuracy delta={delta:+.2f}%")
        print(f"  → Cross-class collision IS the bottleneck")
    else:
        print(f"  collisions_initial={ci}, accuracy delta={delta:+.2f}%")
        print(f"  → Collisions present but sleep didn't help much")
        print(f"  → Bottleneck may be elsewhere")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_benchmark()
