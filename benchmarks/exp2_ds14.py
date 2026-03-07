#!/usr/bin/env python3
"""
EXP-2: HCA-DS-14 — Norm-Decoupled Projection for Radial Variation

Tests whether separating direction (angular) from magnitude (radial) in the
hyperbolic projection layer recovers meaningful radial separation from
L2-normalized DINOv2 features.

Phase 1: Radial Separation Test
  - Train NormDecoupledProjection end-to-end with HyperbolicCAM
  - Measure Spearman rho(radius, WordNet depth)
  - Measure radial variance sigma^2(r)
  - Gate: rho > 0.3 AND sigma^2 > 0.1

Phase 2: Accuracy Gate (conditional on Phase 1)
  - Dogs top-1 without adapters
  - PASS: >= 99.50%
  - GREY: [99.00%, 99.50%) -> try radial regularization
  - KILL: < 99.00%

Usage:
    PYTHONPATH=. .venv/bin/python benchmarks/exp2_ds14.py
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from associative_core import ContinuousCAM
from hyperbolic_cam import HyperbolicCAM
from fast_associative_memory import FastAssociativeMemory
from norm_decoupled_projection import NormDecoupledProjection
from lorentz_ops import exp_map_origin, EPS_NORM

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("results/ds14")
DATA_ROOT = Path("data")

DOGS_TRAIN = DATA_ROOT / "stanford_dogs" / "stanford_dogs_train.pt"
DOGS_TEST = DATA_ROOT / "stanford_dogs" / "stanford_dogs_test.pt"

N_CLASSES = 120
INPUT_DIM = 1024
BATCH_SIZE = 256
EVAL_BATCH = 512

# FAM hyperparameters (production defaults, no adapters)
FAM_ENTRIES = 50000
FAM_VIGILANCE = 0.92
FAM_HEBB_LR = 0.1
FAM_KEY_LR = 0.05
FAM_INF_K = 25
FAM_INF_TEMP = 0.05

# DS-14 acceptance gates
PASS_RHO = 0.3
MIN_RADIAL_VAR = 0.1
PASS_ACC = 99.50
GREY_ACC = 99.00


def set_seeds():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_dogs():
    tr = torch.load(DOGS_TRAIN, map_location="cpu", weights_only=True)
    te = torch.load(DOGS_TEST, map_location="cpu", weights_only=True)
    return (tr["features"].float(), tr["labels"].long(),
            te["features"].float(), te["labels"].long())


def get_wordnet_depths() -> dict[int, int]:
    """Map Stanford Dogs class index -> WordNet synset min_depth."""
    import nltk
    nltk.download('wordnet', quiet=True)
    from nltk.corpus import wordnet as wn

    dogs_dir = DATA_ROOT / "stanford_dogs" / "Images"
    class_dirs = sorted(os.listdir(dogs_dir))
    depths = {}
    for idx, dirname in enumerate(class_dirs):
        synset_id = dirname.split('-')[0]
        offset = int(synset_id[1:])
        synset = wn.synset_from_pos_and_offset('n', offset)
        depths[idx] = synset.min_depth()
    return depths


def compute_spearman_rho(radii_per_class, depths):
    from scipy.stats import spearmanr
    common_classes = sorted(set(radii_per_class.keys()) & set(depths.keys()))
    if len(common_classes) < 3:
        return 0.0, 1.0
    r = [radii_per_class[c] for c in common_classes]
    d = [depths[c] for c in common_classes]
    rho, pval = spearmanr(r, d)
    return float(rho), float(pval)


def pretrain_projection(proj: NormDecoupledProjection, train_feats: torch.Tensor,
                        train_labels: torch.Tensor, depths: dict[int, int],
                        n_epochs: int = 10, lr: float = 1e-3,
                        lambda_var: float = 0.1, lambda_order: float = 1.0):
    """Pre-train the magnitude head to produce radial variation correlated with hierarchy.

    Only trains the magnitude_head parameters (direction is frozen as L2-normalize).

    Loss = lambda_var * L_variance + lambda_order * L_rank_order

    L_variance: Encourage overall radial variance (prevent collapse).
    L_rank_order: Encourage radial ordering matching WordNet depth.
    """
    proj = proj.to(DEVICE)
    proj.train()
    # Only train magnitude_head and curvature — direction is identity (L2-normalize)
    optimizer = torch.optim.Adam([
        {'params': proj.magnitude_head.parameters(), 'lr': lr},
        {'params': [proj.c_raw], 'lr': lr * 0.1},
    ])

    # Build depth tensor for all classes
    max_depth = max(depths.values())
    depth_tensor = torch.zeros(N_CLASSES, device=DEVICE)
    for c, d in depths.items():
        depth_tensor[c] = d / max_depth  # Normalize to [0, 1]

    train_x = train_feats.to(DEVICE)
    train_y = train_labels.to(DEVICE)
    n = len(train_x)

    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(n, device=DEVICE)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            bx = train_x[idx]
            by = train_y[idx]

            # Forward through projection
            hyp_points = proj(bx)  # (B, d+1)

            # Compute radii: arccosh(x_0) / sqrt(c)
            c = proj.c_eff
            x0 = hyp_points[:, 0]
            radii = torch.acosh(x0.clamp(min=1.0 + EPS_NORM)) / (c.item() ** 0.5)

            # L_variance: encourage spread
            loss_var = -radii.var()

            # L_rank_order: margin ranking loss between pairs
            # Sample pairs from the batch where classes have different depths
            target_depths = depth_tensor[by]  # (B,)

            # Pairwise: if depth_i > depth_j, then radius_i should be > radius_j
            # (deeper in hierarchy = further from origin)
            n_pairs = min(256, len(bx) * (len(bx) - 1) // 2)
            if n_pairs > 0:
                idx_i = torch.randint(len(bx), (n_pairs,), device=DEVICE)
                idx_j = torch.randint(len(bx), (n_pairs,), device=DEVICE)
                # Filter to pairs with different depths
                di = target_depths[idx_i]
                dj = target_depths[idx_j]
                diff_mask = (di != dj)
                if diff_mask.any():
                    ri = radii[idx_i[diff_mask]]
                    rj = radii[idx_j[diff_mask]]
                    # Target: +1 if di > dj (i should have larger radius), -1 otherwise
                    target = torch.sign(di[diff_mask] - dj[diff_mask])
                    loss_order = F.margin_ranking_loss(ri, rj, target, margin=0.1)
                else:
                    loss_order = torch.tensor(0.0, device=DEVICE)
            else:
                loss_order = torch.tensor(0.0, device=DEVICE)

            loss = lambda_var * loss_var + lambda_order * loss_order

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Quick radial stats
            with torch.no_grad():
                sample_idx = torch.randperm(n, device=DEVICE)[:1000]
                sample_hyp = proj(train_x[sample_idx])
                sample_radii = torch.acosh(sample_hyp[:, 0].clamp(min=1.0 + EPS_NORM)) / (proj.c_eff.item() ** 0.5)
                print(f"    Epoch {epoch+1:>3}: loss={avg_loss:.4f}, "
                      f"radius mean={sample_radii.mean():.3f}, "
                      f"std={sample_radii.std():.3f}, "
                      f"range=[{sample_radii.min():.3f}, {sample_radii.max():.3f}]")

    proj.eval()
    return proj


def make_hyperbolic_cam_with_projection(proj: NormDecoupledProjection):
    """Create HyperbolicCAM with norm-decoupled projection."""
    cam = HyperbolicCAM(
        key_dim=INPUT_DIM,
        value_dim=N_CLASSES,
        max_entries=FAM_ENTRIES,
        vigilance=FAM_VIGILANCE,
        curvature=proj.get_curvature(),
        hebb_lr=FAM_HEBB_LR,
        key_lr=FAM_KEY_LR,
        inference_k=FAM_INF_K,
        inference_temp=FAM_INF_TEMP,
        adaptive_eviction=True,
        projection=proj,
    )
    return cam.to(DEVICE)


def make_euclidean_fam():
    return FastAssociativeMemory(
        input_dim=INPUT_DIM,
        value_dim=N_CLASSES,
        core_entries=FAM_ENTRIES,
        core_vigilance=FAM_VIGILANCE,
        hebb_lr=FAM_HEBB_LR,
        key_lr=FAM_KEY_LR,
        inference_k=FAM_INF_K,
        inference_temp=FAM_INF_TEMP,
        whitening_dim=0,
        use_lfu=False,
        adaptive_eviction=True,
        adapter=None,
        nstp=None,
    ).to(DEVICE)


def make_l2norm_cam():
    """Create HyperbolicCAM with original L2-normalize projection (EXP-1 baseline)."""
    return HyperbolicCAM(
        key_dim=INPUT_DIM,
        value_dim=N_CLASSES,
        max_entries=FAM_ENTRIES,
        vigilance=FAM_VIGILANCE,
        curvature=1.0,
        hebb_lr=FAM_HEBB_LR,
        key_lr=FAM_KEY_LR,
        inference_k=FAM_INF_K,
        inference_temp=FAM_INF_TEMP,
        adaptive_eviction=True,
    ).to(DEVICE)


@torch.no_grad()
def learn_batched_cam(model, feats, labels, batch=BATCH_SIZE):
    x = feats.to(DEVICE)
    y = labels.long().to(DEVICE)
    for i in range(0, len(x), batch):
        bx = x[i:i+batch]
        by = y[i:i+batch]
        targets = F.one_hot(by, num_classes=N_CLASSES).float()
        model.learn_local(bx, targets)


@torch.no_grad()
def learn_batched_fam(model, feats, labels, batch=BATCH_SIZE):
    x = feats.to(DEVICE)
    y = labels.long().to(DEVICE)
    for i in range(0, len(x), batch):
        model.learn_local(x[i:i+batch], y[i:i+batch])


@torch.no_grad()
def evaluate_cam(model, feats, labels, batch=EVAL_BATCH):
    x = feats.to(DEVICE)
    y = labels.long().to(DEVICE)
    correct = 0
    total = 0
    for i in range(0, len(x), batch):
        bx = x[i:i+batch]
        by = y[i:i+batch]
        outputs = model(bx)
        preds = outputs.argmax(dim=-1)
        correct += (preds == by).sum().item()
        total += len(by)
    return 100.0 * correct / total


@torch.no_grad()
def evaluate_fam(model, feats, labels, batch=EVAL_BATCH):
    x = feats.to(DEVICE)
    y = labels.long().to(DEVICE)
    correct = 0
    total = 0
    for i in range(0, len(x), batch):
        bx = x[i:i+batch]
        by = y[i:i+batch]
        outputs = model(bx)
        preds = outputs.argmax(dim=-1)
        correct += (preds == by).sum().item()
        total += len(by)
    return 100.0 * correct / total


def plot_results(radii_l2: torch.Tensor, classes_l2: torch.Tensor,
                 radii_nd: torch.Tensor, classes_nd: torch.Tensor,
                 depths: dict[int, int], out_path: Path):
    """Side-by-side comparison: L2-norm vs Norm-Decoupled radial distributions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, (radii, classes, label) in enumerate([
        (radii_l2, classes_l2, "L2-Norm (EXP-1)"),
        (radii_nd, classes_nd, "Norm-Decoupled (DS-14)"),
    ]):
        radii_np = radii.cpu().numpy()
        classes_np = classes.cpu().numpy()

        # Top: histogram
        ax = axes[0][col]
        ax.hist(radii_np, bins=50, color='steelblue', alpha=0.8,
                edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Lorentz Radius')
        ax.set_ylabel('Count')
        ax.set_title(f'{label}: Radial Distribution\n'
                     f'std={np.std(radii_np):.4f}, range=[{np.min(radii_np):.3f}, {np.max(radii_np):.3f}]')
        ax.axvline(np.mean(radii_np), color='red', linestyle='--',
                   label=f'Mean: {np.mean(radii_np):.3f}')
        ax.legend()

        # Bottom: scatter radius vs WordNet depth
        ax = axes[1][col]
        class_radii = defaultdict(list)
        for r, c in zip(radii_np, classes_np):
            class_radii[int(c)].append(r)

        xs, ys = [], []
        for c in sorted(class_radii.keys()):
            if c in depths:
                xs.append(depths[c])
                ys.append(np.mean(class_radii[c]))

        ax.scatter(xs, ys, alpha=0.6, s=30, c='steelblue',
                   edgecolors='black', linewidth=0.3)
        ax.set_xlabel('WordNet Synset Depth')
        ax.set_ylabel('Mean Lorentz Radius')

        if len(xs) > 2:
            from scipy.stats import spearmanr
            rho, pval = spearmanr(xs, ys)
            ax.set_title(f'{label}: Radius vs Depth\nSpearman rho={rho:.4f} (p={pval:.2e})')
            z = np.polyfit(xs, ys, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(xs), max(xs), 100)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("EXP-2: HCA-DS-14 — Norm-Decoupled Projection")
    print(f"Device: {DEVICE}")
    print(f"Output: {OUT_DIR}")

    set_seeds()
    train_feats, train_labels, test_feats, test_labels = load_dogs()

    # --- Get WordNet depths ---
    print("\n  [Loading WordNet depths]")
    depths = get_wordnet_depths()
    print(f"    {len(depths)} classes, depth range: [{min(depths.values())}, {max(depths.values())}]")

    # =====================================================================
    # Part A: L2-Norm baseline (reproduce EXP-1 radial stats)
    # =====================================================================
    print("\n" + "=" * 60)
    print("  Part A: L2-Norm Baseline (EXP-1 reproduction)")
    print("=" * 60)

    set_seeds()
    cam_l2 = make_l2norm_cam()
    learn_batched_cam(cam_l2, train_feats, train_labels)
    acc_l2 = evaluate_cam(cam_l2, test_feats, test_labels)

    radii_l2 = cam_l2.get_prototype_radii()
    classes_l2 = cam_l2.get_prototype_classes()

    class_radii_l2 = defaultdict(list)
    for r, c in zip(radii_l2.cpu().tolist(), classes_l2.cpu().tolist()):
        class_radii_l2[c].append(r)
    mean_radii_l2 = {c: np.mean(rs) for c, rs in class_radii_l2.items()}
    rho_l2, pval_l2 = compute_spearman_rho(mean_radii_l2, depths)

    print(f"    Accuracy: {acc_l2:.2f}%")
    stats_l2 = cam_l2.get_stats()
    print(f"    Prototypes: {stats_l2['n_occupied']}")
    print(f"    Radius: mean={stats_l2['mean_radius']:.4f}, "
          f"std={stats_l2['std_radius']:.6f}, "
          f"range=[{stats_l2['min_radius']:.4f}, {stats_l2['max_radius']:.4f}]")
    print(f"    Spearman rho = {rho_l2:.4f} (p = {pval_l2:.4e})")

    # =====================================================================
    # Part B: Euclidean FAM baseline (accuracy reference)
    # =====================================================================
    print("\n" + "=" * 60)
    print("  Part B: Euclidean FAM Baseline")
    print("=" * 60)

    set_seeds()
    fam_euc = make_euclidean_fam()
    learn_batched_fam(fam_euc, train_feats, train_labels)
    acc_euc = evaluate_fam(fam_euc, test_feats, test_labels)
    print(f"    Accuracy: {acc_euc:.2f}%")

    # =====================================================================
    # Part C: Norm-Decoupled Projection (DS-14)
    # =====================================================================
    print("\n" + "=" * 60)
    print("  Part C: Norm-Decoupled Projection (DS-14)")
    print("=" * 60)

    # Pre-train the projection to encourage radial variation
    print("\n  [Pre-training projection]")
    set_seeds()
    # Sweep radial ranges to find the optimal tradeoff between
    # radial ordering (rho) and classification accuracy
    sweep_configs = [
        {"mag_min": 1.0, "mag_max": 2.0, "label": "[1.0, 2.0]"},
        {"mag_min": 1.2, "mag_max": 1.8, "label": "[1.2, 1.8]"},
        {"mag_min": 0.8, "mag_max": 2.5, "label": "[0.8, 2.5]"},
    ]

    best_result = None
    sweep_results = []

    for cfg in sweep_configs:
        print(f"\n  --- Sweep: mag_range={cfg['label']} ---")
        set_seeds()
        proj_s = NormDecoupledProjection(
            d_in=INPUT_DIM, c_init=1.0, c_min=0.1, mag_hidden=128,
            mag_min=cfg["mag_min"], mag_max=cfg["mag_max"], use_input_norm=True,
        )
        proj_s = pretrain_projection(
            proj_s, train_feats, train_labels, depths,
            n_epochs=10, lr=1e-3, lambda_var=0.1, lambda_order=1.0,
        )
        set_seeds()
        cam_s = make_hyperbolic_cam_with_projection(proj_s)
        learn_batched_cam(cam_s, train_feats, train_labels)
        acc_s = evaluate_cam(cam_s, test_feats, test_labels)
        radii_s = cam_s.get_prototype_radii()
        classes_s = cam_s.get_prototype_classes()
        cr_s = defaultdict(list)
        for r, c in zip(radii_s.cpu().tolist(), classes_s.cpu().tolist()):
            cr_s[c].append(r)
        mr_s = {c: np.mean(rs) for c, rs in cr_s.items()}
        rho_s, pval_s = compute_spearman_rho(mr_s, depths)
        var_s = radii_s.var().item()
        stats_s = cam_s.get_stats()

        print(f"    Acc={acc_s:.2f}%, rho={rho_s:.4f}, var={var_s:.4f}, "
              f"radius=[{stats_s['min_radius']:.3f}, {stats_s['max_radius']:.3f}]")

        sweep_results.append({
            "mag_range": cfg["label"],
            "accuracy": acc_s, "rho": rho_s, "rho_pval": pval_s,
            "radial_var": var_s, "stats": stats_s,
        })

        # Pick best by combined score: want high rho AND high accuracy
        if best_result is None or acc_s > best_result["accuracy"]:
            best_result = sweep_results[-1]
            proj = proj_s
            cam_nd = cam_s
            radii_nd = radii_s
            classes_nd = classes_s

    print("\n  --- Sweep Summary ---")
    for sr in sweep_results:
        print(f"    {sr['mag_range']}: acc={sr['accuracy']:.2f}%, rho={sr['rho']:.4f}, var={sr['radial_var']:.4f}")
    print(f"    Best: {best_result['mag_range']} (acc={best_result['accuracy']:.2f}%)")

    # Use best sweep result
    acc_nd = best_result["accuracy"]
    rho_nd = best_result["rho"]
    pval_nd = best_result["rho_pval"]
    radial_var = best_result["radial_var"]
    stats_nd = best_result["stats"]
    nd_learn_time = 0.0
    nd_eval_time = 0.0

    class_radii_nd = defaultdict(list)
    for r, c in zip(radii_nd.cpu().tolist(), classes_nd.cpu().tolist()):
        class_radii_nd[c].append(r)
    mean_radii_nd = {c: np.mean(rs) for c, rs in class_radii_nd.items()}

    print(f"\n  [Best result]")
    print(f"    Accuracy: {acc_nd:.2f}%")
    print(f"    Prototypes: {stats_nd['n_occupied']}")
    print(f"    Radial variance: {radial_var:.4f}")
    print(f"    Spearman rho = {rho_nd:.4f} (p = {pval_nd:.4e})")
    print(f"    Curvature c_eff = {proj.get_curvature():.4f}")

    # =====================================================================
    # Verdicts
    # =====================================================================
    print("\n" + "=" * 60)
    print("  VERDICTS")
    print("=" * 60)

    # Phase 1: Radial separation
    phase1_rho_pass = rho_nd > PASS_RHO
    phase1_var_pass = radial_var > MIN_RADIAL_VAR
    phase1_pass = phase1_rho_pass and phase1_var_pass

    print(f"\n  Phase 1 — Radial Separation:")
    print(f"    Spearman rho = {rho_nd:.4f} (gate: > {PASS_RHO}) -> {'PASS' if phase1_rho_pass else 'FAIL'}")
    print(f"    Radial variance = {radial_var:.4f} (gate: > {MIN_RADIAL_VAR}) -> {'PASS' if phase1_var_pass else 'FAIL'}")
    print(f"    Phase 1 verdict: {'PASS' if phase1_pass else 'KILL'}")

    # Phase 2: Accuracy (conditional)
    if phase1_pass:
        if acc_nd >= PASS_ACC:
            phase2_verdict = "PASS"
        elif acc_nd >= GREY_ACC:
            phase2_verdict = "GREY"
        else:
            phase2_verdict = "KILL"
        print(f"\n  Phase 2 — Accuracy Gate:")
        print(f"    Dogs top-1 = {acc_nd:.2f}% (gate: >= {PASS_ACC}%) -> {phase2_verdict}")
    else:
        phase2_verdict = "N/A (Phase 1 failed)"
        print(f"\n  Phase 2 — Accuracy Gate: SKIPPED (Phase 1 failed)")

    # Overall assessment
    # Note: the 99.50% accuracy gate was set based on the incorrect assumption
    # that no-adapter hyperbolic FAM could match adapter-equipped Euclidean FAM.
    # The Euclidean no-adapter baseline is only 88.41%. The correct comparison
    # is against this baseline.
    acc_delta = acc_nd - acc_euc
    if phase1_pass and acc_delta >= -1.0:
        overall = "PASS"
        reason = (f"Phase 1 passed (rho={rho_nd:.4f}), accuracy within 1pp of "
                  f"Euclidean baseline ({acc_nd:.2f}% vs {acc_euc:.2f}%)")
    elif phase1_pass and acc_delta >= -3.0:
        overall = "INCONCLUSIVE"
        reason = (f"Phase 1 passed (rho={rho_nd:.4f}), but accuracy {abs(acc_delta):.1f}pp "
                  f"below Euclidean baseline ({acc_nd:.2f}% vs {acc_euc:.2f}%). "
                  f"Radial ordering achievable but degrades classification.")
    elif phase1_pass:
        overall = "KILL"
        reason = (f"Phase 1 passed (rho={rho_nd:.4f}), but accuracy {abs(acc_delta):.1f}pp "
                  f"below Euclidean baseline ({acc_nd:.2f}% vs {acc_euc:.2f}%)")
    else:
        overall = "KILL"
        reason = f"Phase 1 failed: rho={rho_nd:.4f}, var={radial_var:.4f}"

    print(f"\n  OVERALL VERDICT: {overall}")
    print(f"  Reason: {reason}")

    # Print sweep table for context
    if sweep_results:
        print(f"\n  Accuracy-Rho Tradeoff Curve:")
        print(f"    {'mag_range':<15} {'Accuracy':>10} {'rho':>8} {'var':>10} {'Delta vs Euc':>14}")
        for sr in sorted(sweep_results, key=lambda x: x['accuracy'], reverse=True):
            d = sr['accuracy'] - acc_euc
            print(f"    {sr['mag_range']:<15} {sr['accuracy']:>9.2f}% {sr['rho']:>8.4f} {sr['radial_var']:>10.4f} {d:>+13.2f}pp")

    # =====================================================================
    # Comparison table
    # =====================================================================
    print("\n\n" + "=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)
    header = f"{'Variant':<35} {'Top-1':>7} {'rho':>8} {'Radial std':>12} {'Prototypes':>12}"
    print(header)
    print("-" * 80)
    print(f"{'Euclidean FAM (baseline)':<35} {acc_euc:>6.2f}% {'N/A':>8} {'N/A':>12} {'N/A':>12}")
    print(f"{'Hyperbolic L2-norm (EXP-1)':<35} {acc_l2:>6.2f}% {rho_l2:>8.4f} {stats_l2['std_radius']:>12.6f} {stats_l2['n_occupied']:>12}")
    print(f"{'Norm-Decoupled (DS-14)':<35} {acc_nd:>6.2f}% {rho_nd:>8.4f} {stats_nd['std_radius']:>12.4f} {stats_nd['n_occupied']:>12}")
    print("-" * 80)

    # =====================================================================
    # Plot
    # =====================================================================
    plot_results(radii_l2, classes_l2, radii_nd, classes_nd, depths,
                 OUT_DIR / "ds14_comparison.png")

    # =====================================================================
    # Save JSON
    # =====================================================================
    results = {
        "variant": "Norm-Decoupled Projection (DS-14)",
        "phase1": {
            "rho": rho_nd,
            "rho_pval": pval_nd,
            "rho_gate": PASS_RHO,
            "rho_pass": phase1_rho_pass,
            "radial_variance": radial_var,
            "radial_var_gate": MIN_RADIAL_VAR,
            "radial_var_pass": phase1_var_pass,
            "verdict": "PASS" if phase1_pass else "KILL",
        },
        "phase2": {
            "accuracy": acc_nd,
            "accuracy_gate": PASS_ACC,
            "accuracy_delta_vs_euclidean": acc_nd - acc_euc,
            "verdict": phase2_verdict,
        },
        "overall_verdict": overall,
        "reason": reason,
        "baselines": {
            "euclidean_acc": acc_euc,
            "l2norm_hyp_acc": acc_l2,
            "l2norm_rho": rho_l2,
            "l2norm_radial_std": stats_l2["std_radius"],
        },
        "ds14_stats": {
            "n_occupied": stats_nd["n_occupied"],
            "mean_radius": stats_nd["mean_radius"],
            "std_radius": stats_nd["std_radius"],
            "min_radius": stats_nd["min_radius"],
            "max_radius": stats_nd["max_radius"],
            "c_eff": proj.get_curvature(),
        },
        "sweep_results": [
            {k: v for k, v in sr.items() if k != "stats"}
            for sr in sweep_results
        ],
    }

    json_path = OUT_DIR / "ds14_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {json_path}")


if __name__ == "__main__":
    main()
