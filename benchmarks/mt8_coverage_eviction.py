#!/usr/bin/env python3
"""
benchmarks/mt8_coverage_eviction.py — MT-8: Coverage-Aware Eviction Analysis

Measures prototype load distribution and backup coverage to determine whether
LFU (Hit Count) is the right eviction policy or if a coverage-aware policy is needed.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from fast_associative_memory import FastAssociativeMemory
from nstp import NSTPController
from benchmarks.g30_adapter_sweep import load_domain_features as g30_load_domain_features
from benchmarks.g31_retrain_adapters import _load_g30_domain_adapter

TAG = "MT-8"

DOMAINS = ["dogs", "birds", "cars", "aircraft"]
MAX_ENTRIES = 50000  # Large enough to avoid eviction during ingestion
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

def compute_gini(array: np.ndarray) -> float:
    """Compute Gini coefficient of a 1D array."""
    if len(array) == 0:
        return 0.0
    array = array.flatten().astype(np.float64)
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 1e-12
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return float(((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))))

def compute_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Basic spearman correlation without scipy."""
    n = len(x)
    if n < 2:
        return 0.0
    # Simple rank: argsort twice. (Does not average ties).
    x_rank = np.argsort(np.argsort(x))
    y_rank = np.argsort(np.argsort(y))
    d = x_rank - y_rank
    spearman = 1.0 - (6.0 * np.sum(d**2)) / (n * (n**2 - 1.0))
    return float(spearman)

@torch.no_grad()
def get_top2_prototypes(fam: FastAssociativeMemory, queries: torch.Tensor):
    """Extracts top-1 and top-2 nearest prototypes post-Mahalanobis + CSLS + NSTP."""
    cam = fam.core_cam
    queries = fam._project(queries)
    
    valid_idx = cam.occupied.nonzero(as_tuple=True)[0]
    n_valid = len(valid_idx)
    # We need at least 2 prototypes to extract top-2
    final_k = min(cam.inference_k, n_valid)
    broad_k = min(100, n_valid)

    q_norm = F.normalize(cam._cast(queries), dim=-1)
    sims = q_norm @ cam._keys_norm[valid_idx].T
    if broad_k < 2:
        # Fallback if too few prototypes 
        return valid_idx[0].expand(queries.size(0)), valid_idx[0].expand(queries.size(0))

    broad_sims_raw, broad_locs = sims.topk(broad_k, dim=1)
    broad_slots = valid_idx[broad_locs]

    candidate_classes = cam.values[broad_slots].float().argmax(dim=-1)
    local_vars = cam.class_vars[candidate_classes].clamp(min=1e-4)
    inv_std = local_vars.rsqrt()
    candidate_keys = cam.keys[broad_slots].float()
    scaled_Q = queries.float().unsqueeze(1) * inv_std
    scaled_K = candidate_keys * inv_std
    reranked_sims = F.cosine_similarity(scaled_Q, scaled_K, dim=-1)

    candidate_density = cam.prototype_density[broad_slots]
    csls_sims = 2.0 * reranked_sims - cam.csls_weight * candidate_density

    final_k = max(2, final_k)
    _, topk_locs = csls_sims.topk(final_k, dim=1)
    topk_slots = broad_slots.gather(1, topk_locs)
    topk_sims = reranked_sims.gather(1, topk_locs)

    if cam.nstp is not None:
        topk_keys = cam.keys[topk_slots].float()
        topk_vals = cam.values[topk_slots].float()
        keep_mask, _ = cam.nstp.prune_batch(q_norm.float(), topk_keys, topk_vals, topk_sims)
        # Instead of masking to -inf, we just want top-2. 
        # But if NSTP masks out the true top-2, we need to pick the next one.
        topk_sims = topk_sims.masked_fill(~keep_mask, -float("inf"))

    sorted_sims, sorted_idx = topk_sims.sort(dim=1, descending=True)
    sorted_slots = topk_slots.gather(1, sorted_idx)
    
    # Return Top-1 and Top-2 slots
    return sorted_slots[:, 0], sorted_slots[:, 1]

def process_domain(domain: str, data_root: Path, device: torch.device):
    print(f"\n[{TAG}] Processing domain: {domain}")
    set_all_seeds(SEED)

    tr_f, tr_l, te_f, te_l, n_classes = g30_load_domain_features(domain, data_root)
    input_dim = int(tr_f.shape[1])
    train_feats = tr_f.to(device)
    train_labels = tr_l.to(device)
    test_feats = te_f.to(device)
    
    adapter = _load_g30_domain_adapter(domain, input_dim=input_dim, device=device)
    nstp = NSTPController(sibling_threshold=0.85, depth_epsilon=0.10)
    
    fam = FastAssociativeMemory(
        input_dim=1024,
        value_dim=n_classes,
        core_entries=MAX_ENTRIES,
        core_vigilance=0.85,
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=25,
        inference_temp=0.05,
        use_lfu=True,
        adapter=adapter,
        nstp=nstp,
    ).to(device)

    # Ingest entire training set
    n_train = len(train_feats)
    for i in range(0, n_train, BATCH_SIZE):
        batch_f = train_feats[i:i+BATCH_SIZE]
        batch_l = train_labels[i:i+BATCH_SIZE]
        fam.learn_local(batch_f, batch_l)
        
    cam = fam.core_cam
    occ_idx = cam.occupied.nonzero(as_tuple=True)[0]
    n_occ = len(occ_idx)
    print(f"  {n_occ} prototypes occupied after training.")

    # Step 1 & 2: Evaluate on test queries
    n_test = len(test_feats)
    loads = torch.zeros(MAX_ENTRIES, dtype=torch.long, device=device)
    irreplaceability = torch.zeros(MAX_ENTRIES, dtype=torch.long, device=device)
    
    fragility_count = 0
    
    for i in range(0, n_test, BATCH_SIZE):
        batch_q = test_feats[i:i+BATCH_SIZE]
        t1_slots, t2_slots = get_top2_prototypes(fam, batch_q)
        
        # Load count
        unique_t1, counts_t1 = t1_slots.unique(return_counts=True)
        loads[unique_t1] += counts_t1
        
        # Backup coverage
        t1_classes = cam.values[t1_slots].argmax(dim=-1)
        t2_classes = cam.values[t2_slots].argmax(dim=-1)
        
        fragile_mask = t1_classes != t2_classes
        fragility_count += fragile_mask.sum().item()
        
        # Irreplaceability: if prototype removed, query loses its only same-class top-1
        fragile_t1_slots = t1_slots[fragile_mask]
        if len(fragile_t1_slots) > 0:
            u_frag_t1, c_frag_t1 = fragile_t1_slots.unique(return_counts=True)
            irreplaceability[u_frag_t1] += c_frag_t1
            
    fragility_pct = 100.0 * fragility_count / n_test
    
    valid_loads = loads[occ_idx].cpu().numpy()
    zero_load_count = (valid_loads == 0).sum()
    gini_coeff = compute_gini(valid_loads)
    
    # Step 3: LFU vs Coverage Loss
    valid_hits = cam.hit_counts[occ_idx].cpu().numpy()
    valid_irrepl = irreplaceability[occ_idx].cpu().numpy()
    
    spearman_corr = compute_spearman(valid_hits, valid_irrepl)
    
    # Policy conflicts: LFU bottom 10% vs Irreplaceability top 10%
    lfu_targets = np.argsort(valid_hits)[:max(1, n_occ//10)] # Bottom 10% hit counts
    irrepl_vip = np.argsort(valid_irrepl)[-max(1, n_occ//10):] # Top 10% irreplaceability
    policy_conflicts = len(set(lfu_targets).intersection(set(irrepl_vip)))
    
    return {
        "Domain": domain,
        "Total_Protos": n_occ,
        "Zero_Load": int(zero_load_count),
        "Zero_Load_Pct": 100.0 * float(zero_load_count) / n_occ,
        "Fragility_Pct": fragility_pct,
        "Gini": gini_coeff,
        "Spearman": spearman_corr,
        "Conflicts": policy_conflicts,
        "Valid_Loads": valid_loads
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{TAG}] Target Device: {device}")
    
    data_root = Path(args.data_root)
    
    all_loads = []
    results = []
    
    for d in DOMAINS:
        try:
            res = process_domain(d, data_root, device)
            # Exclude Valid_Loads from tabular results
            loads = res.pop("Valid_Loads")
            all_loads.append((d, loads))
            results.append(res)
        except Exception as e:
            print(f"  Failed on {d}: {e}")
            import traceback
            traceback.print_exc()
            
    if HAS_MPL and all_loads:
        out_dir = Path("results/mt8")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (d, loads) in enumerate(all_loads):
            if idx >= len(axes): break
            ax = axes[idx]
            # Plot log scale histogram
            ax.hist(loads, bins=50, log=True, color='blue', alpha=0.7)
            ax.set_title(f"{d.capitalize()} Prototype Load Distribution")
            ax.set_xlabel("Load (# Eval Queries)")
            ax.set_ylabel("Number of Prototypes")
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        fig_path = out_dir / "mt8_load_distributions.png"
        plt.savefig(fig_path)
        print(f"\n[{TAG}] Saved load histograms to {fig_path}")

    print("\n### Summary Table\n")
    print("| Domain | Total Protos | Zero-Load | Fragility % | Gini | LFU-vs-Coverage Spearman | Policy Conflicts |")
    print("|--------|-------------|-----------|-------------|------|--------------------------|-----------------|")
    for r in results:
        zl_str = f"{r['Zero_Load']} ({r['Zero_Load_Pct']:.1f}%)"
        print(f"| {r['Domain'].capitalize():<6} | {r['Total_Protos']:<11} | {zl_str:<9} | {r['Fragility_Pct']:>10.1f}% | {r['Gini']:.3f} | {r['Spearman']:>24.3f} | {r['Conflicts']:>15} |")

    # Save Markdown Report
    out_dir = Path("results/mt8")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "mt8_coverage_eviction.md"
    
    md = "# MT-8: Coverage-Aware Eviction Analysis\n\n"
    md += "| Domain | Total Protos | Zero-Load | Fragility % | Gini | LFU-vs-Coverage Spearman | Policy Conflicts |\n"
    md += "|--------|-------------|-----------|-------------|------|--------------------------|-----------------|\n"
    for r in results:
        zl_str = f"{r['Zero_Load']} ({r['Zero_Load_Pct']:.1f}%)"
        md += f"| {r['Domain'].capitalize()} | {r['Total_Protos']} | {zl_str} | {r['Fragility_Pct']:.1f}% | {r['Gini']:.3f} | {r['Spearman']:.3f} | {r['Conflicts']} |\n"
        
    md += "\n\n### Analysis\n\n"
    md += "LFU tracks write-path access (nearest neighbor during training), while Irreplaceability tracks read-path coverage (sole same-class nearest neighbor during inference). "
    md += "The Gini coefficient measures the inequality of prototype loads, and the Spearman correlation shows how well LFU ranks align with coverage loss.\n"
    
    if HAS_MPL:
        md += "\n![Load Distributions](mt8_load_distributions.png)\n"
        
    report_path.write_text(md)
    print(f"\n[{TAG}] Saved report to {report_path}")

if __name__ == "__main__":
    main()
