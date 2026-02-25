#!/usr/bin/env python3
"""
benchmarks/mt5_mahalanobis_calibration.py — MT-5: Mahalanobis Calibration Surface.

Maps where Mahalanobis re-ranking disagrees with raw cosine along interpolation
paths between class centroid pairs, comparing S1 (forward-order store from MT-4)
against S3 (reversed-order store).

For each class pair (A, B):
  - Compute adapter-projected centroids cA, cB from training data.
  - Sample 101 points x(t) = (1-t)*cA + t*cB, t in [0,1].
  - At each x(t): get cosine top-10 and Mahalanobis top-10 from the store.
  - Record top-1 class under each metric, top-10 set overlap, flip indicator.

Output:
  results/mt5/mt5_paths.csv
  results/mt5/mt5_flip_summary_s1.png
  results/mt5/mt5_flip_summary_s3.png
  results/mt5/mt5_flip_diff.png
  results/mt5/mt5_exemplar_sweeps.png
  results/mt5/mt5_writeup.md

Usage:
    PYTHONPATH=. python benchmarks/mt5_mahalanobis_calibration.py [--device auto]
"""

from __future__ import annotations

import argparse
import itertools
import random
import subprocess
import sys
import time as _time_mod
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from fast_associative_memory import FastAssociativeMemory  # noqa: E402
from nstp import NSTPController  # noqa: E402
from benchmarks.g30_adapter_sweep import (  # noqa: E402
    load_domain_features as g30_load_domain_features,
)
from benchmarks.g31_retrain_adapters import _load_g30_domain_adapter  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 42
BATCH_SIZE = 256
MAX_ENTRIES = 8_000       # Match MT-4
N_T = 101                 # Interpolation points
TOP_K = 10                # Top-K for cosine vs Mahalanobis comparison
SEL_K = 25                # Actual pipeline selection boundary (inference_k)
N_PAIRS = 60              # Number of class pairs to probe
TAG = "MT-5"


# ---------------------------------------------------------------------------
# Helpers reused from MT-4
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_production_fam(adapter, n_classes: int, device: torch.device) -> FastAssociativeMemory:
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
    return fam


def clone_fam(src: FastAssociativeMemory, device: torch.device) -> FastAssociativeMemory:
    nstp = NSTPController(sibling_threshold=0.85, depth_epsilon=0.10)
    dst = FastAssociativeMemory(
        input_dim=src.input_dim,
        value_dim=src.value_dim,
        core_entries=src.core_cam.max_entries,
        core_vigilance=src.core_cam.vigilance,
        hebb_lr=src.core_cam.hebb_lr,
        key_lr=src.core_cam.key_lr,
        inference_k=src.core_cam.inference_k,
        inference_temp=src.core_cam.inference_temp,
        use_lfu=src.core_cam.use_lfu,
        adapter=src.adapter,
        nstp=nstp,
    ).to(device)
    dst.load_state_dict(src.state_dict())
    dst.core_cam.csls_weight = src.core_cam.csls_weight
    dst.core_cam.inference_sim_floor = src.core_cam.inference_sim_floor
    dst.core_cam.var_ema_alpha = src.core_cam.var_ema_alpha
    dst.core_cam.ema_beta = src.core_cam.ema_beta
    return dst


@torch.no_grad()
def feed_batches(fam, feats, labels, batch_indices, fixed_time):
    _orig = _time_mod.time
    clock = [fixed_time]
    def _dt():
        clock[0] += 1.0
        return clock[0]
    _time_mod.time = _dt
    try:
        for s, e in batch_indices:
            fam.learn_local(feats[s:e], labels[s:e])
    finally:
        _time_mod.time = _orig


# ---------------------------------------------------------------------------
# Core probing: cosine vs Mahalanobis top-K at a query point
# ---------------------------------------------------------------------------

@torch.no_grad()
def probe_query(cam, query_proj: torch.Tensor) -> Dict:
    """Return cosine vs Mahalanobis comparison at top-10 and top-25 boundaries."""
    valid_idx = cam.occupied.nonzero(as_tuple=True)[0]
    n_valid = len(valid_idx)
    k = min(TOP_K, n_valid)
    sel_k = min(SEL_K, n_valid)
    broad_k = min(100, n_valid)

    q = query_proj.unsqueeze(0).float()  # (1, D)
    q_norm = F.normalize(q, dim=-1)

    # Cosine similarities to all occupied
    sims_all = (q_norm @ cam._keys_norm[valid_idx].T).squeeze(0)  # (N_occ,)

    # Broad top-100 for Mahalanobis
    broad_sims, broad_locs = sims_all.topk(broad_k)
    broad_slots = valid_idx[broad_locs]

    # Cosine top-K and top-sel_k
    cos_topk_slots = broad_slots[:k]
    cos_topk_classes = cam.values[cos_topk_slots].float().argmax(dim=-1)
    cos_sel_slots = broad_slots[:sel_k]

    # Mahalanobis re-ranking on broad pool
    candidate_classes = cam.values[broad_slots].float().argmax(dim=-1)
    local_vars = cam.class_vars[candidate_classes].clamp(min=1e-4)
    inv_std = local_vars.rsqrt()
    candidate_keys = cam.keys[broad_slots].float()
    scaled_Q = q * inv_std          # (broad_k, D)  broadcast
    scaled_K = candidate_keys * inv_std
    mah_sims = F.cosine_similarity(scaled_Q, scaled_K, dim=-1)  # (broad_k,)

    _, mah_order = mah_sims.sort(descending=True)
    mah_topk_slots = broad_slots[mah_order[:k]]
    mah_topk_classes = cam.values[mah_topk_slots].float().argmax(dim=-1)
    mah_sel_slots = broad_slots[mah_order[:sel_k]]

    # Overlap at top-10 and top-25
    cos10_set = set(cos_topk_slots.tolist())
    mah10_set = set(mah_topk_slots.tolist())
    cos25_set = set(cos_sel_slots.tolist())
    mah25_set = set(mah_sel_slots.tolist())

    # Kendall rank correlation on the full broad-100 rankings
    cos_ranks = torch.arange(broad_k, device=broad_slots.device).float()
    mah_ranks_at_cos_order = torch.empty_like(cos_ranks)
    mah_ranks_at_cos_order[mah_order] = torch.arange(broad_k, device=broad_slots.device).float()
    # Simplified rank correlation: Spearman rho via torch (no scipy needed)
    cos_r = cos_ranks - cos_ranks.mean()
    mah_r = mah_ranks_at_cos_order - mah_ranks_at_cos_order.mean()
    rho = (cos_r * mah_r).sum() / (cos_r.norm() * mah_r.norm() + 1e-12)

    # Sim magnitude diagnostics
    sim_delta = (mah_sims - broad_sims).abs()

    return {
        "cos_top1_class": cos_topk_classes[0].item(),
        "mah_top1_class": mah_topk_classes[0].item(),
        "overlap_10": len(cos10_set & mah10_set),
        "overlap_25": len(cos25_set & mah25_set),
        "flip_top1": int(cos_topk_classes[0].item() != mah_topk_classes[0].item()),
        "rank_corr": rho.item(),
        "sim_delta_mean": sim_delta.mean().item(),
        "sim_delta_max": sim_delta.max().item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=f"{TAG}: Mahalanobis Calibration Surface")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-root", type=str, default="data")
    args = parser.parse_args()

    device = (torch.device("cuda") if args.device == "auto" and torch.cuda.is_available()
              else torch.device(args.device if args.device != "auto" else "cpu"))
    print(f"[{TAG}] Device: {device}", flush=True)
    set_all_seeds(SEED)

    # --- Load data ---
    print(f"[{TAG}] Loading Stanford Dogs features...", flush=True)
    tr_f, tr_l, te_f, te_l, n_classes = g30_load_domain_features("dogs", Path(args.data_root))
    train_feats = tr_f.to(device)
    train_labels = tr_l.to(device)

    print(f"[{TAG}] Loading production Dogs adapter...", flush=True)
    adapter = _load_g30_domain_adapter("dogs", input_dim=int(tr_f.shape[1]), device=device)

    # --- Build S1 (forward) and S3 (reversed) matching MT-4 ---
    n_samples = len(train_feats)
    forward_indices = [(i, min(i + BATCH_SIZE, n_samples))
                       for i in range(0, n_samples, BATCH_SIZE)]
    reverse_indices = list(reversed(forward_indices))

    print(f"[{TAG}] Building S1 (forward) and S3 (reversed)...", flush=True)
    S0 = make_production_fam(adapter, n_classes, device)
    S1 = clone_fam(S0, device)
    S3 = clone_fam(S0, device)
    del S0

    feed_batches(S1, train_feats, train_labels, forward_indices, 1_000_000.0)
    feed_batches(S3, train_feats, train_labels, reverse_indices, 1_000_000.0)

    s1_occ = S1.core_cam.occupied.sum().item()
    s3_occ = S3.core_cam.occupied.sum().item()
    print(f"[{TAG}] S1 prototypes: {s1_occ}, S3 prototypes: {s3_occ}", flush=True)

    # --- Compute adapter-projected centroids per class ---
    print(f"[{TAG}] Computing class centroids in adapter-projected space...", flush=True)
    with torch.no_grad():
        projected = S1._project(train_feats)  # (N, D) — adapter is shared & frozen
    centroids = {}
    for c in range(n_classes):
        mask = train_labels == c
        if mask.sum() > 0:
            centroids[c] = projected[mask].mean(dim=0)
    print(f"[{TAG}] Centroids for {len(centroids)} classes", flush=True)

    # --- Pick class pairs ---
    classes_with_centroids = sorted(centroids.keys())
    all_pairs = list(itertools.combinations(classes_with_centroids, 2))
    random.shuffle(all_pairs)
    pairs = all_pairs[:N_PAIRS]
    print(f"[{TAG}] Probing {len(pairs)} class pairs × {N_T} interpolation points × 2 stores", flush=True)

    # --- Sweep ---
    t_values = torch.linspace(0.0, 1.0, N_T, device=device)
    rows = []

    for pi, (cA, cB) in enumerate(pairs):
        if pi % 10 == 0:
            print(f"  [{pi}/{len(pairs)}]", flush=True)
        centA = centroids[cA]
        centB = centroids[cB]

        for ti, t in enumerate(t_values):
            xt = (1.0 - t) * centA + t * centB  # interpolated query

            for store_name, fam in [("S1", S1), ("S3", S3)]:
                res = probe_query(fam.core_cam, xt)
                rows.append({
                    "pair_idx": pi,
                    "class_a": cA,
                    "class_b": cB,
                    "t": t.item(),
                    "store": store_name,
                    "cos_top1": res["cos_top1_class"],
                    "mah_top1": res["mah_top1_class"],
                    "overlap_10": res["overlap_10"],
                    "overlap_25": res["overlap_25"],
                    "flip_top1": res["flip_top1"],
                    "rank_corr": res["rank_corr"],
                    "sim_delta_mean": res["sim_delta_mean"],
                    "sim_delta_max": res["sim_delta_max"],
                })

    df = pd.DataFrame(rows)
    out_dir = Path("results/mt5")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "mt5_paths.csv", index=False)
    print(f"[{TAG}] Saved {len(df)} rows to mt5_paths.csv", flush=True)

    # --- Class_vars diagnostics ---
    s1_vars = S1.core_cam.class_vars.float().cpu()  # (n_classes, D)
    s3_vars = S3.core_cam.class_vars.float().cpu()
    s1_var_dev = (s1_vars - 1.0).abs().mean(dim=1)  # mean |deviation from 1| per class
    s3_var_dev = (s3_vars - 1.0).abs().mean(dim=1)
    s1_var_spread = s1_vars.std(dim=1)  # per-class std across dimensions
    s3_var_spread = s3_vars.std(dim=1)
    # Cross-store: per-class cosine between class_vars vectors
    var_cos = F.cosine_similarity(s1_vars, s3_vars, dim=1)

    # --- Analysis & plots ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df_s1 = df[df["store"] == "S1"]
    df_s3 = df[df["store"] == "S3"]

    # ---- Plot 1: Rank correlation + overlap by t (the main surface) ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, sub, title in [(0, df_s1, "S1 (forward)"), (1, df_s3, "S3 (reversed)")]:
        # Row 0: rank correlation
        rc_by_t = sub.groupby("t")["rank_corr"].mean()
        axes[0, col].plot(rc_by_t.index, rc_by_t.values, linewidth=1.5, color="#1976d2")
        axes[0, col].fill_between(rc_by_t.index,
                                   sub.groupby("t")["rank_corr"].quantile(0.25).values,
                                   sub.groupby("t")["rank_corr"].quantile(0.75).values,
                                   alpha=0.2, color="#1976d2")
        axes[0, col].set_ylabel("Spearman ρ (cos vs mah rank)")
        axes[0, col].set_title(f"{title}: Rank Correlation")
        axes[0, col].set_ylim(0.5, 1.02)
        axes[0, col].axhline(rc_by_t.mean(), color="red", linestyle="--", alpha=0.5,
                              label=f"mean={rc_by_t.mean():.4f}")
        axes[0, col].legend(fontsize=8)

        # Row 1: top-10 and top-25 overlap
        ov10 = sub.groupby("t")["overlap_10"].mean()
        ov25 = sub.groupby("t")["overlap_25"].mean()
        axes[1, col].plot(ov10.index, ov10.values, linewidth=1.5, color="#7b1fa2", label="top-10")
        axes[1, col].plot(ov25.index, ov25.values, linewidth=1.5, color="#ff9800", label="top-25")
        axes[1, col].set_xlabel("t (interpolation)")
        axes[1, col].set_ylabel("Overlap (out of K)")
        axes[1, col].set_title(f"{title}: Set Overlap")
        axes[1, col].legend(fontsize=8)

    # Column 3: Diff in rank correlation
    rc_s1 = df_s1.groupby("t")["rank_corr"].mean()
    rc_s3 = df_s3.groupby("t")["rank_corr"].mean()
    diff_rc = rc_s3.values - rc_s1.values
    axes[0, 2].bar(rc_s1.index, diff_rc, width=0.01,
                    color=np.where(diff_rc > 0, "#2e7d32", "#d32f2f"))
    axes[0, 2].set_title("Rank corr diff (S3 − S1)")
    axes[0, 2].axhline(0, color="gray", linewidth=0.5)

    # Diff in overlap-25
    ov25_s1 = df_s1.groupby("t")["overlap_25"].mean()
    ov25_s3 = df_s3.groupby("t")["overlap_25"].mean()
    diff_ov = ov25_s3.values - ov25_s1.values
    axes[1, 2].bar(ov25_s1.index, diff_ov, width=0.01,
                    color=np.where(diff_ov > 0, "#2e7d32", "#d32f2f"))
    axes[1, 2].set_xlabel("t")
    axes[1, 2].set_title("Top-25 overlap diff (S3 − S1)")
    axes[1, 2].axhline(0, color="gray", linewidth=0.5)

    fig.suptitle(f"{TAG}: Mahalanobis Calibration Surface", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "mt5_flip_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Plot 2: class_vars diagnostic ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].bar(range(n_classes), s1_var_dev.numpy(), alpha=0.6, label="S1", color="#1976d2")
    axes[0].bar(range(n_classes), s3_var_dev.numpy(), alpha=0.6, label="S3", color="#d32f2f")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Mean |class_var − 1.0|")
    axes[0].set_title("Per-class variance deviation from unity")
    axes[0].legend(fontsize=8)

    axes[1].bar(range(n_classes), var_cos.numpy(), color="#7b1fa2", alpha=0.7)
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Cosine sim(S1 vars, S3 vars)")
    axes[1].set_title("S1 vs S3: class_vars agreement")
    axes[1].axhline(var_cos.mean().item(), color="red", linestyle="--",
                     label=f"mean={var_cos.mean().item():.4f}")
    axes[1].legend(fontsize=8)

    # Sim delta histogram
    axes[2].hist(df_s1["sim_delta_mean"].values, bins=50, alpha=0.6, label="S1", color="#1976d2")
    axes[2].hist(df_s3["sim_delta_mean"].values, bins=50, alpha=0.6, label="S3", color="#d32f2f")
    axes[2].set_xlabel("Mean |mah_sim − cos_sim| per query")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Magnitude of Mahalanobis rescoring")
    axes[2].legend(fontsize=8)

    fig.suptitle(f"{TAG}: Mahalanobis Calibration Diagnostics")
    fig.tight_layout()
    fig.savefig(out_dir / "mt5_class_vars_diagnostic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Plot 3: exemplar t-sweeps showing rank correlation ----
    # Pick 4 pairs with lowest mean rank correlation (most reordering)
    pair_rc = df_s1.groupby("pair_idx")["rank_corr"].mean().sort_values()
    exemplar_pairs = pair_rc.head(4).index.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, pidx in zip(axes.flat, exemplar_pairs):
        for store_name, color, ls in [("S1", "#1976d2", "-"), ("S3", "#d32f2f", "--")]:
            sub = df[(df["pair_idx"] == pidx) & (df["store"] == store_name)]
            ax.plot(sub["t"], sub["rank_corr"], color=color, linestyle=ls,
                    linewidth=1.2, label=f"{store_name} rank ρ", alpha=0.8)
            ax2 = ax.twinx()
            ax2.plot(sub["t"], sub["overlap_25"], color=color, linestyle=":",
                     linewidth=1.0, alpha=0.5)
            ax2.set_ylabel("top-25 overlap", fontsize=8, color="gray")
            ax2.set_ylim(15, 26)
        row0 = df[df["pair_idx"] == pidx].iloc[0]
        ax.set_title(f"Pair {pidx}: class {row0['class_a']} ↔ {row0['class_b']}")
        ax.set_xlabel("t")
        ax.set_ylabel("Rank correlation ρ")
        ax.legend(fontsize=8)
        ax.set_ylim(0.5, 1.02)
    fig.suptitle(f"{TAG}: Exemplar Sweeps (lowest mean rank correlation)")
    fig.tight_layout()
    fig.savefig(out_dir / "mt5_exemplar_sweeps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Summary stats ---
    s1_flip_rate = df_s1["flip_top1"].mean()
    s3_flip_rate = df_s3["flip_top1"].mean()
    s1_ov10 = df_s1["overlap_10"].mean()
    s3_ov10 = df_s3["overlap_10"].mean()
    s1_ov25 = df_s1["overlap_25"].mean()
    s3_ov25 = df_s3["overlap_25"].mean()
    s1_rc = df_s1["rank_corr"].mean()
    s3_rc = df_s3["rank_corr"].mean()
    s1_sd = df_s1["sim_delta_mean"].mean()
    s3_sd = df_s3["sim_delta_mean"].mean()
    s1_sd_max = df_s1["sim_delta_max"].mean()
    s3_sd_max = df_s3["sim_delta_max"].mean()

    # Where along t is rank correlation lowest?
    rc_by_t_s1 = df_s1.groupby("t")["rank_corr"].mean()
    rc_by_t_s3 = df_s3.groupby("t")["rank_corr"].mean()
    s1_worst_ts = rc_by_t_s1.nsmallest(int(0.1 * N_T))
    s3_worst_ts = rc_by_t_s3.nsmallest(int(0.1 * N_T))

    # class_vars summary
    s1_var_dev_mean = s1_var_dev.mean().item()
    s3_var_dev_mean = s3_var_dev.mean().item()
    var_cos_mean = var_cos.mean().item()

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()

    writeup = f"""# MT-5: Mahalanobis Calibration Surface

## Config
- **Commit**: `{commit}`
- **Seed**: {SEED}
- **Device**: {device}
- **Dataset**: Stanford Dogs ViT-L/14 1024-d, production adapter
- **Stores**: S1 (forward order) and S3 (reversed order), matching MT-4 (MAX_ENTRIES={MAX_ENTRIES})
- **Class pairs probed**: {len(pairs)} (sampled from {len(all_pairs)} possible pairs of {n_classes} classes)
- **Interpolation points**: {N_T} per pair (t in [0, 1])
- **Comparison**: cosine top-{TOP_K} / top-{SEL_K} vs Mahalanobis top-{TOP_K} / top-{SEL_K}

## Summary Table

| Metric | S1 (forward) | S3 (reversed) |
|---|---|---|
| Top-1 flip rate (cos top-1 != mah top-1) | {s1_flip_rate:.4f} | {s3_flip_rate:.4f} |
| Mean top-10 overlap | {s1_ov10:.2f}/10 | {s3_ov10:.2f}/10 |
| Mean top-25 overlap | {s1_ov25:.2f}/25 | {s3_ov25:.2f}/25 |
| Mean rank corr (Spearman rho, broad-100) | {s1_rc:.4f} | {s3_rc:.4f} |
| Mean |mah_sim - cos_sim| | {s1_sd:.6f} | {s3_sd:.6f} |
| Mean max |mah_sim - cos_sim| | {s1_sd_max:.6f} | {s3_sd_max:.6f} |
| class_vars mean |deviation from 1.0| | {s1_var_dev_mean:.6f} | {s3_var_dev_mean:.6f} |
| S1 vs S3 class_vars cosine (mean) | {var_cos_mean:.4f} | — |

## Key Findings

### 1. Mahalanobis effect magnitude
{"**Near-identity transform.** " + f"Mean sim delta = {s1_sd:.6f} (S1) / {s3_sd:.6f} (S3). The per-class `class_vars` have barely moved from their initial value of 1.0 (mean deviation: {s1_var_dev_mean:.6f} S1, {s3_var_dev_mean:.6f} S3). With EMA alpha=0.01 and ~100 samples per class, the diagonal scaling is essentially `inv_std ≈ 1.0` everywhere, making Mahalanobis re-ranking a near-no-op."
 if s1_sd < 0.01
 else f"**Measurable rescoring.** Mean sim delta = {s1_sd:.4f} (S1) / {s3_sd:.4f} (S3). The Mahalanobis re-ranking produces meaningful rescoring of the broad candidates."}

### 2. Localized vs global reordering?
{"**No reordering detected.** " + f"Rank correlation = {s1_rc:.4f} (S1) / {s3_rc:.4f} (S3) — the cosine and Mahalanobis rankings are essentially identical across all t. Top-10 overlap is {s1_ov10:.1f}/10 and top-25 overlap is {s1_ov25:.1f}/25."
 if s1_rc > 0.999
 else f"**Reordering is {'boundary-localized' if rc_by_t_s1[(rc_by_t_s1.index >= 0.3) & (rc_by_t_s1.index <= 0.7)].mean() < rc_by_t_s1.mean() * 0.98 else 'globally distributed'}.** Rank correlation dips to {s1_worst_ts.mean():.4f} in the worst 10% of t-values (S1). Top-25 overlap: {s1_ov25:.1f}/25."}

### 3. S3 vs S1 calibration?
The class_vars vectors between S1 and S3 have cosine similarity = {var_cos_mean:.4f}.
{"This means the per-class variance estimates are essentially identical despite completely different write order — the Mahalanobis is equally (in)effective in both stores."
 if var_cos_mean > 0.99
 else f"Write order produces meaningfully different class_vars (cos={var_cos_mean:.4f}), "
 + ("but the overall Mahalanobis effect is too small to change rankings." if s1_sd < 0.01 else "and this translates into different rescoring behavior.")}

### 4. Implication for MT-4's 47pp accuracy gap
{"Since Mahalanobis is a near-no-op (sim delta ~{:.1e}), the 47pp accuracy gap between S1 and S3 in MT-4 is **not caused by Mahalanobis miscalibration**. The gap comes entirely from different prototype populations (key drift + eviction) — the stores contain different prototypes for the same classes, leading to different cosine retrievals before Mahalanobis even runs.".format(s1_sd)
 if s1_sd < 0.01
 else "The Mahalanobis rescoring is significant and may contribute to the MT-4 accuracy gap."}

### 5. Where along t does most disagreement occur?
- S1 worst rank-corr region: t in [{s1_worst_ts.index.min():.2f}, {s1_worst_ts.index.max():.2f}] (rho={s1_worst_ts.mean():.4f})
- S3 worst rank-corr region: t in [{s3_worst_ts.index.min():.2f}, {s3_worst_ts.index.max():.2f}] (rho={s3_worst_ts.mean():.4f})

## Plots
- `mt5_flip_summary.png` — Rank correlation and overlap surfaces for S1, S3, and diff
- `mt5_class_vars_diagnostic.png` — Per-class variance deviation, S1/S3 agreement, rescoring magnitude
- `mt5_exemplar_sweeps.png` — 4 exemplar pair sweeps with lowest mean rank correlation

## Artifacts
- `mt5_paths.csv` — {len(df)} rows (pair x t x store)
"""

    (out_dir / "mt5_writeup.md").write_text(writeup)
    print(f"\n[{TAG}] Report saved to {out_dir / 'mt5_writeup.md'}")
    print(writeup)


if __name__ == "__main__":
    main()
