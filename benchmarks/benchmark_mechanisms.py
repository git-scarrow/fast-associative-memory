import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
from fast_associative_memory import FastAssociativeMemory
from associative_core import ContinuousCAM


# --- ABLATION VARIANTS ---

class RandomEvictionCAM(ContinuousCAM):
    """G4 Baseline: Random Eviction."""
    def _alloc_slots_batch(self, n: int):
        n = min(n, self.max_entries)
        free = (~self.occupied).nonzero(as_tuple=True)[0]
        if len(free) >= n:
            return free[:n]
        needed = n - len(free)
        occupied_idx = self.occupied.nonzero(as_tuple=True)[0]
        if len(free) > 0:
            mask = ~torch.isin(occupied_idx, free)
            occupied_idx = occupied_idx[mask]
        needed = min(needed, len(occupied_idx))
        if needed > 0:
            perm = torch.randperm(len(occupied_idx), device=self.occupied.device)
            victims = occupied_idx[perm[:needed]]
        else:
            victims = occupied_idx[:0]
        return torch.cat([free, victims]) if len(free) > 0 else victims


class FIFOEvictionCAM(ContinuousCAM):
    """G4 Baseline: FIFO (Ring Buffer)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("write_pointer", torch.tensor(0, dtype=torch.long))

    def _alloc_slots_batch(self, n: int):
        indices = []
        for _ in range(n):
            idx = self.write_pointer.item()
            indices.append(idx)
            self.write_pointer = (self.write_pointer + 1) % self.max_entries
        return torch.tensor(indices, device=self.occupied.device, dtype=torch.long)


class CoPE_CAM(ContinuousCAM):
    """G3 Baseline: CoPE-like (EMA Winner-Take-All, no LFU)."""
    def learn_local(self, queries: torch.Tensor, targets: torch.Tensor):
        queries = self._cast(queries)
        targets = self._cast(targets)
        best_slots, best_sims = self._get_nearest_batch(queries)

        hits = (best_slots >= 0) & (best_sims >= self.vigilance)

        if hits.any():
            hit_slots_all = best_slots[hits]
            stored_vals = self.values[hit_slots_all]
            payload_sims = F.cosine_similarity(targets[hits], stored_vals, dim=-1)
            same_class = payload_sims > 0.9
            if not same_class.all():
                hit_indices = hits.nonzero(as_tuple=True)[0]
                hits[hit_indices[~same_class]] = False

        misses = ~hits

        if hits.any():
            hit_slots = best_slots[hits]
            hit_queries = queries[hits]
            current_keys = self.keys[hit_slots]
            self.keys[hit_slots] = current_keys + self.hebb_lr * (hit_queries - current_keys)
            self._update_key_norm(hit_slots)
            self.values[hit_slots] = targets[hits]
            self.last_seen[hit_slots] = time.time()

        if misses.any():
            super().learn_local(queries[misses], targets[misses])


# --- HARNESS ---

def generate_multimodal_data(dim, n_samples, n_classes):
    """Generates 2 distinct clusters per class."""
    X_list, Y_list = [], []
    for c in range(n_classes):
        center_a = torch.randn(dim)
        cluster_a = center_a + torch.randn(n_samples // n_classes // 2, dim) * 0.1
        center_b = torch.randn(dim)
        cluster_b = center_b + torch.randn(n_samples // n_classes // 2, dim) * 0.1
        X_list.extend([cluster_a, cluster_b])
        Y_list.extend([c] * len(cluster_a))
        Y_list.extend([c] * len(cluster_b))
    X = torch.cat(X_list)
    Y = torch.tensor(Y_list)
    X = F.normalize(X, dim=-1)
    perm = torch.randperm(len(X))
    return X[perm], Y[perm]


def eval_model(model, loader, device):
    correct, total = 0, 0
    for x, y in loader:
        preds = model(x.to(device)).argmax(dim=1)
        correct += (preds == y.to(device)).sum().item()
        total += x.size(0)
    return 100 * correct / total


def run_mechanism_gauntlet():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DIM = 512
    N_CLASSES = 50
    N_TRAIN = 5000
    N_TEST = 1000
    CAPACITY = 2000  # 40% of train size -> forces eviction

    print(f"\n\u2694\ufe0f  MECHANISM GAUNTLET (G3/G4) \u2694\ufe0f")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {N_CLASSES} classes, multimodal (2 clusters/class)")
    print(f"Constraint: {CAPACITY} slots for {N_TRAIN} train samples")
    print(f"Evaluation: held-out test set ({N_TEST} samples)")

    X_train, Y_train = generate_multimodal_data(DIM, N_TRAIN, N_CLASSES)
    X_test, Y_test = generate_multimodal_data(DIM, N_TEST, N_CLASSES)
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=256)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=256)

    models = {
        "G3_FAM":     FastAssociativeMemory(DIM, N_CLASSES, core_entries=CAPACITY),
        "G3_CoPE":    FastAssociativeMemory(DIM, N_CLASSES, core_entries=CAPACITY),
        "G4_FAM_LFU": FastAssociativeMemory(DIM, N_CLASSES, core_entries=CAPACITY),
        "G4_Random":  FastAssociativeMemory(DIM, N_CLASSES, core_entries=CAPACITY),
        "G4_FIFO":    FastAssociativeMemory(DIM, N_CLASSES, core_entries=CAPACITY),
    }

    models["G3_CoPE"].core_cam = CoPE_CAM(DIM, N_CLASSES, max_entries=CAPACITY, vigilance=0.85).to(DEVICE)
    models["G4_Random"].core_cam = RandomEvictionCAM(DIM, N_CLASSES, max_entries=CAPACITY, vigilance=0.85).to(DEVICE)
    models["G4_FIFO"].core_cam = FIFOEvictionCAM(DIM, N_CLASSES, max_entries=CAPACITY, vigilance=0.85).to(DEVICE)
    for name in ["G3_FAM", "G4_FAM_LFU"]:
        models[name] = models[name].to(DEVICE)
        models[name].core_cam.use_lfu = True

    results = {}
    print(f"\n{'Model':14s} | {'Test Acc':>9} | {'Protos':>12} | {'Time':>6}")
    print("-" * 52)
    for name, model in models.items():
        start = time.time()
        for x, y in train_loader:
            model.learn_local(x.to(DEVICE), y.to(DEVICE))
        dur = time.time() - start
        acc = eval_model(model, test_loader, DEVICE)
        occ = model.core_cam.occupied.sum().item()
        results[name] = acc
        print(f"{name:14s} | {acc:8.2f}% | {occ:5d}/{CAPACITY} | {dur:.2f}s")

    print("\n--- KILL CONDITIONS ---")
    fam_acc  = results["G3_FAM"]
    cope_acc = results["G3_CoPE"]
    lfu_acc  = results["G4_FAM_LFU"]
    rand_acc = results["G4_Random"]
    fifo_acc = results["G4_FIFO"]

    print(f"G3 \u2014 FAM {fam_acc:.2f}% vs CoPE {cope_acc:.2f}%")
    if cope_acc >= fam_acc:
        print("\u26a0\ufe0f  G3 KILL: CoPE \u2265 FAM \u2014 Hebbian update adds nothing over EMA.")
    else:
        print(f"\u2705  G3 PASS: FAM beats CoPE by +{fam_acc - cope_acc:.2f}% on held-out test.")

    print(f"G4 \u2014 LFU {lfu_acc:.2f}% vs Random {rand_acc:.2f}% vs FIFO {fifo_acc:.2f}%")
    if rand_acc >= lfu_acc or fifo_acc >= lfu_acc:
        print("\u26a0\ufe0f  G4 KILL: LFU eviction provides no advantage.")
    else:
        print(f"\u2705  G4 PASS: LFU beats Random by +{lfu_acc - rand_acc:.2f}%, FIFO by +{lfu_acc - fifo_acc:.2f}%.")


if __name__ == "__main__":
    run_mechanism_gauntlet()
