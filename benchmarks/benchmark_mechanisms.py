"""
benchmarks/benchmark_mechanisms.py — The Mechanism Gauntlets (G3 & G4)
Tests Plasticity (FAM vs CoPE) and Eviction (LFU vs Random vs FIFO).
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import copy
import time
import numpy as np
from fast_associative_memory import FastAssociativeMemory
from associative_core import ContinuousCAM

# --- ABLATION VARIANTS ---

class RandomEvictionCAM(ContinuousCAM):
    """G4 Baseline: Random Eviction (The Null Hypothesis)."""
    def _alloc_slots_batch(self, n: int):
        n = min(n, self.max_entries)
        free = (~self.occupied).nonzero(as_tuple=True)[0]
        if len(free) >= n:
            return free[:n]
        
        # If full, pick RANDOM victims
        needed = n - len(free)
        occupied_idx = self.occupied.nonzero(as_tuple=True)[0]
        
        # Exclude slots we are already claiming as free
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
    """G4 Baseline: FIFO Eviction (Ring Buffer)."""
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
    """G3 Baseline: CoPE-like (Continual Prototype Evolution).
    
    Differences from FAM:
    1. No LFU usage tracking.
    2. Updates are P_new = P_old + alpha * (x - P_old).
    3. Allocation only happens if min_dist > T (Vigilance).
    4. Simple Momentum update (Standard CoPE).
    """
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
        
        # CoPE Update: Move Winner Closer (EMA)
        if hits.any():
            hit_slots = best_slots[hits]
            hit_queries = queries[hits]
            current_keys = self.keys[hit_slots]
            self.keys[hit_slots] = current_keys + self.hebb_lr * (hit_queries - current_keys)
            self._update_key_norm(hit_slots)
            self.values[hit_slots] = targets[hits]
            self.last_seen[hit_slots] = time.time()
            # No LFU update

        # CoPE Allocation: Random Eviction when full
        if misses.any():
            self.use_lfu = False
            super().learn_local(queries[misses], targets[misses])


# --- HARNESS ---

def run_mechanism_gauntlet(feature_dim=1024, n_samples=5000, n_classes=50, capacity=2000):
    print(f"\n\u2694\ufe0f  MECHANISM GAUNTLET (G3/G4) \u2694\ufe0f")
    print(f"Stream: {n_samples} samples, {n_classes} classes, Capacity constraint: {capacity} slots")
    
    # 1. Synthetic Multimodal Data (2 clusters per class to punish SCM)
    print("Generating multimodal synthetic stream...")
    X_list, Y_list = [], []
    for c in range(n_classes):
        center_a = torch.randn(feature_dim)
        cluster_a = center_a + torch.randn(n_samples // n_classes // 2, feature_dim) * 0.1
        center_b = torch.randn(feature_dim)
        cluster_b = center_b + torch.randn(n_samples // n_classes // 2, feature_dim) * 0.1
        X_list.append(cluster_a)
        X_list.append(cluster_b)
        Y_list.extend([c] * len(cluster_a))
        Y_list.extend([c] * len(cluster_b))
        
    X = torch.cat(X_list)
    Y = torch.tensor(Y_list)
    perm = torch.randperm(len(X))
    X, Y = X[perm], Y[perm]
    X = F.normalize(X, dim=-1)
    
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=256)
    
    # 2. Setup Contenders
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    models = {
        "G3_FAM":     FastAssociativeMemory(input_dim=feature_dim, value_dim=n_classes, core_entries=capacity),
        "G3_CoPE":    FastAssociativeMemory(input_dim=feature_dim, value_dim=n_classes, core_entries=capacity),
        "G4_FAM_LFU": FastAssociativeMemory(input_dim=feature_dim, value_dim=n_classes, core_entries=capacity),
        "G4_Random":  FastAssociativeMemory(input_dim=feature_dim, value_dim=n_classes, core_entries=capacity),
        "G4_FIFO":    FastAssociativeMemory(input_dim=feature_dim, value_dim=n_classes, core_entries=capacity),
    }
    
    models["G3_CoPE"].core_cam = CoPE_CAM(feature_dim, n_classes, max_entries=capacity, vigilance=0.85).to(device)
    models["G4_Random"].core_cam = RandomEvictionCAM(feature_dim, n_classes, max_entries=capacity, vigilance=0.85).to(device)
    models["G4_FIFO"].core_cam = FIFOEvictionCAM(feature_dim, n_classes, max_entries=capacity, vigilance=0.85).to(device)
    for name in ["G3_FAM", "G4_FAM_LFU"]:
        models[name] = models[name].to(device)

    # 3. Run Stream
    print(f"Streaming on {device}...")
    for name, model in models.items():
        start = time.time()
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            model.learn_local(x_batch, y_batch)
        print(f"[{name}] Trained in {time.time() - start:.2f}s")
        
    # 4. Evaluate
    print("\n--- RESULTS (Retention under Constraint) ---")
    print(f"{'Model':15s} | {'Acc':>8} | {'Protos':>12}")
    print("-" * 45)
    for name, model in models.items():
        correct = 0
        total = 0
        occupied = model.core_cam.occupied.sum().item()
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            correct += (preds.argmax(dim=1) == y_batch).sum().item()
            total += x_batch.size(0)
        acc = 100 * correct / total
        print(f"{name:15s} | {acc:7.2f}% | {occupied:5d}/{capacity}")

    # Kill condition verdicts
    print("\n--- KILL CONDITIONS ---")
    accs = {}
    for name, model in models.items():
        correct = 0
        total = 0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            correct += (preds.argmax(dim=1) == y_batch).sum().item()
            total += x_batch.size(0)
        accs[name] = 100 * correct / total

    fam_acc = accs["G3_FAM"]
    cope_acc = accs["G3_CoPE"]
    lfu_acc = accs["G4_FAM_LFU"]
    rand_acc = accs["G4_Random"]
    fifo_acc = accs["G4_FIFO"]

    print(f"G3 — FAM {fam_acc:.2f}% vs CoPE {cope_acc:.2f}%")
    if cope_acc >= fam_acc:
        print("\u26a0\ufe0f  G3 KILL CONDITION MET: CoPE >= FAM — vigilance+Hebbian adds nothing over EMA.")
    else:
        print(f"\u2705  G3 PASS: FAM beats CoPE by +{fam_acc - cope_acc:.2f}%")

    print(f"G4 — FAM(LFU) {lfu_acc:.2f}% vs Random {rand_acc:.2f}% vs FIFO {fifo_acc:.2f}%")
    if rand_acc >= lfu_acc or fifo_acc >= lfu_acc:
        print("\u26a0\ufe0f  G4 KILL CONDITION MET: LFU eviction provides no advantage.")
    else:
        print(f"\u2705  G4 PASS: LFU beats Random by +{lfu_acc - rand_acc:.2f}%, FIFO by +{lfu_acc - fifo_acc:.2f}%")


if __name__ == "__main__":
    # Capacity is 40% of dataset size -> forces heavy eviction
    run_mechanism_gauntlet(capacity=2000)
