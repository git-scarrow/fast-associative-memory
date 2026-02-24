"""
benchmarks/benchmark_robustness.py — The Robustness Gauntlet (G8)
Tests Vigilance under Label Noise and Task Re-entry (ABA).

G8a: 20% of stream has randomized labels.
     Pass: Accuracy on clean data remains high; noise protos evicted via LFU.
     Fail: FAM memorizes noise, evicting good data.

G8b: Task A -> Task B -> Task A (re-entry).
     Pass: FAM recognizes old A prototypes (near-zero new protos on re-entry).
     Fail: FAM re-learns A from scratch (duplicate prototypes).
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from fast_associative_memory import FastAssociativeMemory


def generate_task_data(dim, n_samples, class_offset, n_classes=10):
    """Generates a synthetic task (normalized cluster blobs)."""
    X_list, Y_list = [], []
    for c in range(n_classes):
        center = torch.randn(dim)
        cluster = center + torch.randn(n_samples // n_classes, dim) * 0.05
        X_list.append(cluster)
        Y_list.extend([class_offset + c] * len(cluster))
    X = torch.cat(X_list)
    X = F.normalize(X, dim=-1)
    Y = torch.tensor(Y_list)
    return TensorDataset(X, Y)


def run_noise_test(device="cuda"):
    print("\n\U0001f50a  G8a: LABEL NOISE INJECTION  \U0001f50a")
    dim = 512
    n_clean = 4000
    n_noise = 1000  # 20% noise
    n_classes = 20

    clean_ds = generate_task_data(dim, n_clean, 0, n_classes)
    noise_X = F.normalize(torch.randn(n_noise, dim), dim=-1)
    noise_Y = torch.randint(0, n_classes, (n_noise,))
    noise_ds = TensorDataset(noise_X, noise_Y)

    full_ds = ConcatDataset([clean_ds, noise_ds])
    loader = DataLoader(full_ds, batch_size=256, shuffle=True)

    # Capacity = n_clean: noise MUST displace clean data if handled poorly
    model = FastAssociativeMemory(
        input_dim=dim, value_dim=n_classes,
        core_entries=n_clean, core_vigilance=0.90,
    ).to(device)

    print(f"Stream: {n_clean} Clean + {n_noise} Noise. Capacity: {n_clean}")

    for x, y in loader:
        model.learn_local(x.to(device), y.to(device))

    # Evaluate on CLEAN ONLY
    correct = 0
    total = 0
    clean_loader = DataLoader(clean_ds, batch_size=256)
    for x, y in clean_loader:
        preds = model(x.to(device)).argmax(dim=1)
        correct += (preds == y.to(device)).sum().item()
        total += x.size(0)

    acc = 100 * correct / total
    occupied = model.core_cam.occupied.sum().item()
    print(f"Clean Data Accuracy : {acc:.2f}%  (kill threshold: <90%)")
    print(f"Prototypes occupied : {occupied}/{n_clean}")
    if acc >= 90.0:
        print("\u2705  G8a PASS: Vigilance protected clean data from noise corruption.")
    else:
        print("\u26a0\ufe0f  G8a KILL: Noise displaced clean prototypes — vigilance insufficient.")


def run_aba_test(device="cuda"):
    print("\n\U0001f504  G8b: ABA RE-ENTRY TEST  \U0001f504")
    dim = 512
    model = FastAssociativeMemory(
        input_dim=dim, value_dim=30, core_entries=5000
    ).to(device)

    task_a = generate_task_data(dim, 1000, class_offset=0, n_classes=10)
    task_b = generate_task_data(dim, 1000, class_offset=10, n_classes=10)

    print("Phase 1: Learning Task A...")
    for x, y in DataLoader(task_a, batch_size=256, shuffle=True):
        model.learn_local(x.to(device), y.to(device))
    protos_a = model.core_cam.occupied.sum().item()
    print(f"  -> A Learned. Protos: {protos_a}")

    print("Phase 2: Learning Task B...")
    for x, y in DataLoader(task_b, batch_size=256, shuffle=True):
        model.learn_local(x.to(device), y.to(device))
    protos_ab = model.core_cam.occupied.sum().item()
    print(f"  -> B Learned. Total Protos: {protos_ab} (Delta: +{protos_ab - protos_a})")

    print("Phase 3: Re-entry of Task A...")
    protos_before_reentry = model.core_cam.occupied.sum().item()
    for x, y in DataLoader(task_a, batch_size=256, shuffle=True):
        model.learn_local(x.to(device), y.to(device))
    final_protos = model.core_cam.occupied.sum().item()
    new_protos = final_protos - protos_before_reentry

    print(f"  -> A Re-learned. Total Protos: {final_protos}")
    print(f"\nMetric: Duplicate Prototypes Created on Re-entry = {new_protos}")
    if new_protos < 100:
        print("\u2705  G8b PASS: Vigilance recognized old A prototypes. No massive duplication.")
    else:
        print(f"\u26a0\ufe0f  G8b KILL: FAM created {new_protos} duplicate prototypes — treated A as new data.")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run_noise_test(dev)
    run_aba_test(dev)
