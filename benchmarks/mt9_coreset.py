#!/usr/bin/env python3
"""
benchmarks/mt9_coreset.py — MT-9: Coreset Allocation vs. Online Write Path

Comparisons of standard FAM streaming ingestion vs. an offline greedy k-center
coreset selection. Determines if the online write path is suboptimal for coverage
and establishes the maximum condensation ceiling.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from fast_associative_memory import FastAssociativeMemory
from nstp import NSTPController
from benchmarks.g30_adapter_sweep import load_domain_features as g30_load_domain_features
from benchmarks.g31_retrain_adapters import _load_g30_domain_adapter

TAG = "MT-9"

CAPACITIES = [1000, 5000, 10000, 25000, 50000]
BATCH_SIZE = 256
SEED = 42

def set_all_seeds(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def k_center_greedy(features: torch.Tensor, n_centers: int) -> torch.Tensor:
    """Greedy k-center coreset selection on GPU.
    Selects points that maximize coverage (minimize max distance).
    Uses 1 - cosine_similarity as distance metric.
    """
    N = len(features)
    n_centers = min(n_centers, N)
    
    features_norm = F.normalize(features, dim=-1)
    
    selected = torch.zeros(n_centers, dtype=torch.long, device=features.device)
    
    # Pick first center randomly
    curr = 0 # Deterministic start for reproducibility. Could use random, but simple is fine.
    selected[0] = curr
    
    if n_centers == 1:
        return selected
    
    # Track max similarity to any selected center for all points
    # Min distance = Max similarity
    max_sim = torch.full((N,), -float('inf'), device=features.device)
    
    for i in range(1, n_centers):
        if i % 1000 == 0:
            print(f"    [k-center] Selected {i}/{n_centers}...")
        new_center_feat = features_norm[curr:curr+1]
        sims = (features_norm @ new_center_feat.T).squeeze(-1)
        
        # Update max similarity to nearest center
        max_sim = torch.maximum(max_sim, sims)
        
        # Next center is the one farthest from selected set (i.e. MIN max_sim)
        curr = max_sim.argmin().item()
        selected[i] = curr
        
    return selected

def make_fam(adapter, n_classes: int, capacity: int, device: torch.device) -> FastAssociativeMemory:
    nstp = NSTPController(sibling_threshold=0.85, depth_epsilon=0.10)
    fam = FastAssociativeMemory(
        input_dim=1024,
        value_dim=n_classes,
        core_entries=capacity,
        core_vigilance=0.85,
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=25,
        inference_temp=0.05,
        use_lfu=True,
        adapter=adapter,
        nstp=nstp,
    ).to(device)
    return fam

@torch.no_grad()
def populate_coreset_fam(fam: FastAssociativeMemory, train_feats: torch.Tensor, train_labels: torch.Tensor, selected_idx: torch.Tensor):
    """Directly loads the coreset into FAM without EMA or online updates."""
    # 1. Project all train feats
    q_float = fam._project(train_feats).float()
    
    cam = fam.core_cam
    
    # 2. Compute true global class means and vars for Mahalanobis
    for c in train_labels.unique():
        x_c = q_float[train_labels == c]
        if len(x_c) > 0:
            cam.class_means[c] = x_c.mean(0)
            cam.class_vars[c] = (x_c - x_c.mean(0)).pow(2).mean(0)
        
    # 3. Load selected prototypes
    n_sel = len(selected_idx)
    selected_q = q_float[selected_idx]
    selected_l = F.one_hot(train_labels[selected_idx], num_classes=cam.value_dim).float()
    
    cam.keys[:n_sel] = selected_q
    cam.values[:n_sel] = selected_l
    cam.occupied[:n_sel] = True
    cam._update_key_norm(torch.arange(n_sel))
    
    # 4. Compute prototype density for CSLS (offline)
    # Mean sim to top-100 nearest prototypes within the coreset
    sim_matrix = F.normalize(selected_q, dim=-1) @ F.normalize(selected_q, dim=-1).T
    k_density = min(100, n_sel)
    topk_sims, _ = sim_matrix.topk(k_density, dim=1)
    cam.prototype_density[:n_sel] = topk_sims.mean(dim=1)

@torch.no_grad()
def eval_accuracy_and_coverage(fam: FastAssociativeMemory, test_feats: torch.Tensor, test_labels: torch.Tensor) -> Tuple[float, float]:
    correct = 0
    total = len(test_feats)
    total_distance = 0.0
    
    cam = fam.core_cam
    valid_idx = cam.occupied.nonzero(as_tuple=True)[0]
    keys_norm = cam._keys_norm[valid_idx]
    
    for i in range(0, total, 256):
        batch_f = test_feats[i:i + 256]
        batch_l = test_labels[i:i + 256]
        
        # Full read path accuracy
        outputs = fam(batch_f)
        preds = outputs.argmax(dim=1)
        correct += (preds == batch_l).sum().item()
        
        # Coverage Radius (mean cosine distance to nearest prototype)
        q_proj = fam._project(batch_f).float()
        q_norm = F.normalize(q_proj, dim=-1)
        sims = q_norm @ keys_norm.T
        max_sims, _ = sims.max(dim=1)
        distances = 1.0 - max_sims
        total_distance += distances.sum().item()
        
    acc = correct / total * 100 if total > 0 else 0.0
    mean_dist = total_distance / total if total > 0 else 0.0
    return acc, mean_dist

def main():
    parser = argparse.ArgumentParser(description=f"{TAG}: Coreset Allocation Baseline")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-root", type=str, default="data")
    args = parser.parse_args()

    device = (torch.device("cuda") if args.device == "auto" and torch.cuda.is_available()
              else torch.device(args.device if args.device != "auto" else "cpu"))
    print(f"[{TAG}] Device: {device}", flush=True)

    set_all_seeds(SEED)

    print(f"[{TAG}] Loading Stanford Dogs features...", flush=True)
    tr_f, tr_l, te_f, te_l, n_classes = g30_load_domain_features("dogs", Path(args.data_root))
    input_dim = int(tr_f.shape[1])
    train_feats = tr_f.to(device)
    train_labels = tr_l.to(device)
    test_feats = te_f.to(device)
    test_labels = te_l.to(device)
    n_train = len(train_feats)
    print(f"[{TAG}] Train: {n_train}, Test: {len(test_feats)}, Classes: {n_classes}", flush=True)

    print(f"[{TAG}] Loading production Dogs adapter...", flush=True)
    adapter = _load_g30_domain_adapter("dogs", input_dim=input_dim, device=device)

    # Pre-compute coresets for requested capacities
    print(f"[{TAG}] Pre-computing k-center coresets...")
    coresets = {}
    max_cap = max(CAPACITIES)
    full_coreset_idx = k_center_greedy(adapter(train_feats).float(), min(max_cap, n_train))
    print(f"[{TAG}] Full coreset shape: {full_coreset_idx.shape}")

    results = []

    for cap in CAPACITIES:
        actual_cap = min(cap, n_train)
        print(f"\n--- Capacity: {cap} (Effective: {actual_cap}) ---")

        stats_row = {"cap": cap}

        # --- Condition A: Coreset ---
        set_all_seeds(SEED)
        fam_core = make_fam(adapter, n_classes, cap, device)
        coreset_idx = full_coreset_idx[:actual_cap]
        populate_coreset_fam(fam_core, train_feats, train_labels, coreset_idx)
        
        acc_core, cov_core = eval_accuracy_and_coverage(fam_core, test_feats, test_labels)
        stats_row["Coreset Acc"] = acc_core
        stats_row["Coreset Cov Radius"] = cov_core
        print(f"  Coreset: Acc={acc_core:.2f}%, CovRadius={cov_core:.4f}")

        # --- Condition B: FAM Online ---
        set_all_seeds(SEED)
        fam_online = make_fam(adapter, n_classes, cap, device)
        
        for i in range(0, n_train, BATCH_SIZE):
            fam_online.learn_local(train_feats[i:i+BATCH_SIZE], train_labels[i:i+BATCH_SIZE])
            
        acc_online, cov_online = eval_accuracy_and_coverage(fam_online, test_feats, test_labels)
        stats_row["FAM Acc"] = acc_online
        stats_row["FAM Cov Radius"] = cov_online
        print(f"  FAM:     Acc={acc_online:.2f}%, CovRadius={cov_online:.4f}")

        delta = acc_core - acc_online
        stats_row["Delta"] = delta
        print(f"  Delta:   {delta:+.2f}pp")
        
        results.append(stats_row)

    print("\n### Results Table\n")
    print("| N    | Coreset Acc | FAM Acc | Delta  | Coreset Cov Radius | FAM Cov Radius |")
    print("|------|------------|---------|--------|-------------------|----------------|")
    for r in results:
        print(f"| {r['cap']:<4} | {r['Coreset Acc']:>10.2f}% | {r['FAM Acc']:>6.2f}% | {r['Delta']:>5.2f}pp | {r['Coreset Cov Radius']:>17.4f} | {r['FAM Cov Radius']:>14.4f} |")

    # Generate writeup
    deltas = [r["Delta"] for r in results]
    writeup = "\n### Analysis\n\n"
    
    if all(d > 2.0 for d in deltas):
        writeup += "Coreset wins by >2pp at all N values. The online write path is significantly suboptimal for coverage; the architecture should explore coreset-inspired allocation or periodic offline re-selection.\n"
    elif all(d <= 1.0 and d >= -1.0 for d in deltas):
        writeup += "FAM wins or ties (within ±1pp) across all capacities. Online allocation is achieving near-optimal coverage despite chaotic positions; the architecture's strength is robust allocation/eviction dynamics.\n"
    else:
        # Mixed results
        drops_at_low = any(d > 1.0 for d, r in zip(deltas, results) if r['cap'] <= 5000)
        drops_at_high = any(d > 1.0 for d, r in zip(deltas, results) if r['cap'] >= 10000)
        
        if drops_at_low and not drops_at_high:
            writeup += "Coreset significantly outperforms FAM at low N but not high N. Coverage optimization matters most when the budget is tight, but becomes redundant when slots are abundant.\n"
        elif drops_at_high and not drops_at_low:
            writeup += "Coreset outperforms FAM at high N but not low N. At low N, both methods are forced to cover major modes; at high N, FAM may waste slots on redundant positions while the coreset doesn't.\n"
        else:
            writeup += "Mixed results depending on capacity. The online write path differs dynamically from greedy k-center coverage maximization.\n"

    writeup += "\n**Condensation Gap (Coreset - FAM Acc) [Maximum potential improvement from better placement]:**\n"
    for r in results:
        writeup += f"- N={r['cap']}: {r['Delta']:+.2f}pp\n"

    print(writeup)
    
    out_dir = Path("results/mt9")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "mt9_coreset_dogs.md"
    
    md_content = "# MT-9: Coreset Allocation vs. Online Write Path\n\n"
    md_content += "Dataset: Stanford Dogs (1024-d features)\n"
    md_content += "Adapter: Production Dogs\n\n"
    
    md_content += "### Results Table\n\n"
    md_content += "| N | Coreset Acc | FAM Acc | Delta | Coreset Cov Radius | FAM Cov Radius |\n"
    md_content += "|---|-------------|---------|-------|--------------------|----------------|\n"
    for r in results:
        md_content += f"| {r['cap']} | {r['Coreset Acc']:.2f}% | {r['FAM Acc']:.2f}% | {r['Delta']:+.2f}pp | {r['Coreset Cov Radius']:.4f} | {r['FAM Cov Radius']:.4f} |\n"
        
    md_content += writeup
    
    report_path.write_text(md_content)
    print(f"\n[{TAG}] Saved report to {report_path}")

if __name__ == "__main__":
    main()
