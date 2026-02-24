import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from fast_associative_memory import FastAssociativeMemory


def generate_task(dim, n_samples, offset, n_classes=10):
    X_list, Y_list = [], []
    for c in range(n_classes):
        center = torch.randn(dim)
        cluster = center + torch.randn(n_samples // n_classes, dim) * 0.05
        X_list.append(cluster)
        Y_list.extend([offset + c] * len(cluster))
    X = torch.cat(X_list)
    Y = torch.tensor(Y_list)
    return TensorDataset(F.normalize(X, dim=-1), Y)


def run_robustness_gauntlet():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DIM = 512

    print(f"\n\U0001f6e1\ufe0f  ROBUSTNESS GAUNTLET (G8) \U0001f6e1\ufe0f")
    print(f"Device: {DEVICE}")

    # --- G8a: Label Noise ---
    print("\n[G8a] Label Noise Injection (20% Noise)")
    N_CLEAN = 4000
    N_NOISE = 1000
    CAPACITY = 4000

    clean_ds = generate_task(DIM, N_CLEAN, 0, 20)
    noise_X = F.normalize(torch.randn(N_NOISE, DIM), dim=-1)
    noise_Y = torch.randint(0, 20, (N_NOISE,))
    noise_ds = TensorDataset(noise_X, noise_Y)

    model = FastAssociativeMemory(DIM, 20, core_entries=CAPACITY, core_vigilance=0.9).to(DEVICE)
    model.core_cam.use_lfu = True

    loader = DataLoader(ConcatDataset([clean_ds, noise_ds]), batch_size=256, shuffle=True)
    for x, y in loader:
        model.learn_local(x.to(DEVICE), y.to(DEVICE))

    correct, total = 0, 0
    for x, y in DataLoader(clean_ds, batch_size=256):
        preds = model(x.to(DEVICE)).argmax(1)
        correct += (preds == y.to(DEVICE)).sum().item()
        total += x.size(0)

    acc = 100 * correct / total
    occ = model.core_cam.occupied.sum().item()
    print(f"Clean Acc: {acc:.2f}%  (kill threshold: <90%)")
    print(f"Memory:    {occ}/{CAPACITY}")
    if acc > 90:
        print("\u2705  G8a PASS: Noise evicted/suppressed.")
    else:
        print("\u26a0\ufe0f  G8a KILL: Noise displaced clean concepts.")

    # --- G8b: ABA Re-entry ---
    print("\n[G8b] ABA Task Re-entry")
    model = FastAssociativeMemory(DIM, 30, core_entries=5000).to(DEVICE)
    task_a = generate_task(DIM, 1000, 0, 10)
    task_b = generate_task(DIM, 1000, 10, 10)

    for x, y in DataLoader(task_a, batch_size=256):
        model.learn_local(x.to(DEVICE), y.to(DEVICE))
    c1 = model.core_cam.occupied.sum().item()
    print(f"After Task A:    {c1} protos")

    for x, y in DataLoader(task_b, batch_size=256):
        model.learn_local(x.to(DEVICE), y.to(DEVICE))
    c2 = model.core_cam.occupied.sum().item()
    print(f"After Task B:    {c2} protos")

    for x, y in DataLoader(task_a, batch_size=256):
        model.learn_local(x.to(DEVICE), y.to(DEVICE))
    c3 = model.core_cam.occupied.sum().item()
    print(f"After Task A(2): {c3} protos")

    dupes = c3 - c2
    print(f"Duplicates created on re-entry: {dupes}")
    if dupes < 50:
        print("\u2705  G8b PASS: Re-entry recognized.")
    else:
        print(f"\u26a0\ufe0f  G8b KILL: {dupes} duplicate prototypes created.")


if __name__ == "__main__":
    run_robustness_gauntlet()
