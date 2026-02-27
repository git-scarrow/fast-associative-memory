#!/usr/bin/env python3
"""
G38: Conditional Negative Evidence Promotion
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from fast_associative_memory import FastAssociativeMemory
from nstp import NSTPController
from benchmarks.g30_adapter_sweep import load_domain_features as g30_load_domain_features
from benchmarks.g31_retrain_adapters import _load_g30_domain_adapter

# --------------------------
# Models
# --------------------------

class LinearGating(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Input: softmax_vote (V) + neg_hist (V) + neg_sim (V) = 3*V
        self.fc = nn.Linear(num_classes * 3, num_classes)
        
    def forward(self, x):
        return self.fc(x)

class MLPGating(nn.Module):
    def __init__(self, num_classes, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

# --------------------------
# Data Collection
# --------------------------

@torch.no_grad()
def collect_g38_dataset(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    adapter,
    device: torch.device,
    max_traces: int = 100_000,
):
    num_classes = int(train_labels.max().item()) + 1
    nstp = NSTPController(sibling_threshold=0.85, depth_epsilon=0.10)

    mem = FastAssociativeMemory(
        input_dim=int(train_feats.shape[1]),
        value_dim=num_classes,
        core_entries=50000,
        core_vigilance=0.85,
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=25,
        inference_temp=0.05,
        use_lfu=True,
        adapter=adapter,
        nstp=nstp,
    ).to(device)

    # Write training data
    x_train = train_feats.to(device)
    y_train = train_labels.to(device)
    for i in range(0, len(x_train), 256):
        mem.learn_local(x_train[i:i + 256], y_train[i:i + 256])

    print(f"    Prototypes written: {mem.core_cam.occupied.sum().item()}")

    # Collect traces
    query_feats = test_feats
    query_labels = test_labels
    if len(test_feats) < max_traces and len(train_feats) > len(test_feats):
        query_feats = torch.cat([test_feats, train_feats], dim=0)[:max_traces]
        query_labels = torch.cat([test_labels, train_labels], dim=0)[:max_traces]

    core = mem.core_cam
    occ_idx = core.occupied.nonzero(as_tuple=True)[0]
    proto_classes = core.values[occ_idx].float().argmax(dim=-1)

    x_q = query_feats.to(device)
    y_q = query_labels.to(device)

    all_softmax = []
    all_neg_hist = []
    all_neg_sim = []
    all_labels = []
    all_margins = []

    for i in range(0, min(len(x_q), max_traces), 256):
        batch_x = x_q[i:i + 256]
        batch_y = y_q[i:i + 256]
        B = batch_x.size(0)

        proj = mem._project(batch_x)
        outputs, tr = core(proj, nstp=nstp, trace=True)

        q_norm = F.normalize(proj.float(), dim=-1)
        proto_keys = core._keys_norm[occ_idx]
        sims = q_norm @ proto_keys.T
        pred_classes = outputs.argmax(dim=1)

        for b in range(B):
            true_class = batch_y[b].item()
            pred_class = pred_classes[b].item()

            row_sims = sims[b]
            best_sim = row_sims.max().item()
            same_class_mask = proto_classes == pred_class
            other_sims = row_sims.masked_fill(same_class_mask, -1e9)
            second_best = other_sims.max().item() if other_sims.max().item() > -1e8 else best_sim
            margin = best_sim - second_best

            # Normalized softmax vote
            sm_vote = F.softmax(outputs[b], dim=-1)

            # Rejection info
            rej_slots = tr.rejected_slots[b]
            valid_rej = (rej_slots >= 0) & (rej_slots < core.max_entries) & core.occupied[rej_slots]
            valid_slots = rej_slots[valid_rej]

            hist = torch.zeros(num_classes, device=device)
            mean_sims = torch.zeros(num_classes, device=device)

            if len(valid_slots) > 0:
                rej_classes = core.values[valid_slots].float().argmax(dim=-1)
                rej_keys = core._keys_norm[valid_slots]
                qb = q_norm[b]
                rej_cos = (qb @ rej_keys.T)

                hist.scatter_add_(0, rej_classes, torch.ones_like(rej_classes, dtype=torch.float32))
                
                sum_sims = torch.zeros(num_classes, device=device)
                sum_sims.scatter_add_(0, rej_classes, rej_cos)
                count_sims = torch.zeros(num_classes, device=device)
                count_sims.scatter_add_(0, rej_classes, torch.ones_like(rej_classes, dtype=torch.float32))
                
                mean_sims = torch.where(count_sims > 0, sum_sims / count_sims, torch.zeros_like(sum_sims))

            if hist.sum() > 0:
                hist = hist / hist.sum()

            all_softmax.append(sm_vote.cpu())
            all_neg_hist.append(hist.cpu())
            all_neg_sim.append(mean_sims.cpu())
            all_labels.append(true_class)
            all_margins.append(margin)

    dataset = {
        "softmax": torch.stack(all_softmax),
        "neg_hist": torch.stack(all_neg_hist),
        "neg_sim": torch.stack(all_neg_sim),
        "labels": torch.tensor(all_labels, dtype=torch.long),
        "margins": torch.tensor(all_margins, dtype=torch.float32),
        "num_classes": num_classes,
    }
    return dataset

# --------------------------
# Training
# --------------------------

def train_model(model, train_loader, epochs=50, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    return model

@torch.no_grad()
def evaluate_model(model, val_loader, device='cpu'):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    # Latency test on first batch (warmup) then measure per sample
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        
    acc = 100.0 * correct / total
    
    # Measure latency for B=1
    dummy_x = torch.zeros(1, x.size(1), device=device)
    # Warmup
    for _ in range(10):
        model(dummy_x)
        
    torch.cuda.synchronize() if torch.cuda.is_available() and device.type == 'cuda' else None
    t0 = time.perf_counter()
    iters = 1000
    for _ in range(iters):
        model(dummy_x)
    torch.cuda.synchronize() if torch.cuda.is_available() and device.type == 'cuda' else None
    t1 = time.perf_counter()
    
    latency_ms = (t1 - t0) * 1000.0 / iters
    return acc, latency_ms

# --------------------------
# Main
# --------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = Path('data')
    domain = 'dogs'
    
    print(f"Loading {domain} features...")
    tr_f, tr_l, te_f, te_l, n_classes = g30_load_domain_features(domain, data_root)
    print("Loading adapter...")
    adapter = _load_g30_domain_adapter(domain, input_dim=tr_f.shape[1], device=device)
    
    print("Collecting traces and extracting negative evidence features...")
    dataset = collect_g38_dataset(tr_f, tr_l, te_f, te_l, adapter, device)
    
    n_total = len(dataset['labels'])
    print(f"Total traces collected: {n_total}")
    
    # Sort by margin to find Q1 (bottom 25%)
    idx_sorted = torch.argsort(dataset['margins'])
    q1_size = n_total // 4
    q1_idx = idx_sorted[:q1_size]
    
    # Build feature matrix
    X = torch.cat([
        dataset['softmax'][q1_idx],
        dataset['neg_hist'][q1_idx],
        dataset['neg_sim'][q1_idx]
    ], dim=1) # (N, num_classes*3)
    y = dataset['labels'][q1_idx]
    
    print(f"Q1 subset size: {len(y)}")
    
    # Shuffle and split 80/20 train/val
    shuffled_idx = torch.randperm(len(y))
    train_size = int(0.8 * len(y))
    train_idx = shuffled_idx[:train_size]
    val_idx = shuffled_idx[train_size:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # Baseline: Uncorrected Softmax top-1 accuracy on VAL set
    uncorrected_val_preds = dataset['softmax'][q1_idx][val_idx].argmax(dim=1)
    uncorrected_acc = 100.0 * (uncorrected_val_preds == y_val).float().mean().item()
    
    print(f"Uncorrected softmax Top-1 (Val): {uncorrected_acc:.2f}%")
    
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    
    # Train Linear
    print("Training Linear Gating Model...")
    linear_model = LinearGating(dataset['num_classes'])
    train_model(linear_model, train_loader, epochs=50, lr=1e-3, device=device)
    linear_acc, linear_lat = evaluate_model(linear_model, val_loader, device=device)
    linear_delta = linear_acc - uncorrected_acc
    print(f"Linear Gating Top-1: {linear_acc:.2f}% (delta: {linear_delta:+.2f}pp) Latency: {linear_lat:.2f}ms")
    
    # Train MLP
    print("Training MLP Gating Model...")
    mlp_model = MLPGating(dataset['num_classes'], hidden_dim=128)
    train_model(mlp_model, train_loader, epochs=50, lr=1e-3, device=device)
    mlp_acc, mlp_lat = evaluate_model(mlp_model, val_loader, device=device)
    mlp_delta = mlp_acc - uncorrected_acc
    print(f"MLP Gating Top-1: {mlp_acc:.2f}% (delta: {mlp_delta:+.2f}pp) Latency: {mlp_lat:.2f}ms")
    
    # Results
    best_delta = max(linear_delta, mlp_delta)
    best_lat = linear_lat if linear_delta > mlp_delta else mlp_lat
    
    if best_delta > 1.0 and best_lat < 0.5:
        verdict = "PASS"
    elif best_delta < 0.5 or best_lat > 1.0:
        verdict = "KILL"
    else:
        verdict = "MARGINAL"
        
    print("\n" + "="*50)
    print("=== G38 RESULTS ===")
    print(f"| Q1 subset size: {q1_size}")
    print(f"| Uncorrected softmax top-1: {uncorrected_acc:.2f}%")
    print(f"| Linear gating top-1: {linear_acc:.2f}% (delta: {linear_delta:+.2f}pp)")
    print(f"| MLP gating top-1: {mlp_acc:.2f}% (delta: {mlp_delta:+.2f}pp)")
    print(f"| Best gating latency: {best_lat:.2f}ms/query")
    print(f"| PASS/KILL: {verdict}")
    print("="*50 + "\n")
    
    # Save checkpoints as requested
    Path("models").mkdir(exist_ok=True)
    torch.save(linear_model.state_dict(), "models/g38_linear_gating.pt")
    torch.save(mlp_model.state_dict(), "models/g38_mlp_gating.pt")

if __name__ == '__main__':
    main()
