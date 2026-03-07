#!/usr/bin/env python3
"""
EXP-1: HFAM Phase 1 Validation — Hyperbolic Prototype Store on Stanford Dogs

Evaluates HyperbolicCAM (Lorentz hyperboloid prototype store) against the
Euclidean baseline (ContinuousCAM) with NO adapters enabled.

Metrics:
  - Top-1 accuracy (primary)
  - Compute overhead ratio vs Euclidean baseline (wall-clock)
  - Prototype radial distribution histogram
  - Spearman rho between prototype Lorentz radius and WordNet synset depth

Acceptance gates (from FAM-DS-1):
  PASS: top-1 >= 99.71% AND overhead <= 1.05x AND rho > 0.3
  GREY: top-1 in [99.50%, 99.71%) -> retry with hybrid radial decay
  KILL: top-1 < 99.50% after both attempts, OR overhead > 1.10x

Usage:
    PYTHONPATH=. .venv/bin/python benchmarks/exp1_hfam.py
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from associative_core import ContinuousCAM
from hyperbolic_cam import HyperbolicCAM
from fast_associative_memory import FastAssociativeMemory

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("results/exp1_hfam")
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

# Acceptance gates
PASS_ACC = 99.71
GREY_ACC = 99.50
MAX_OVERHEAD = 1.10
PASS_OVERHEAD = 1.05
PASS_RHO = 0.3


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


def make_euclidean_fam():
    """Create Euclidean baseline FAM (no adapters)."""
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


def make_hyperbolic_cam(curvature: float = 1.0):
    """Create HyperbolicCAM (no adapters, no whitening)."""
    return HyperbolicCAM(
        key_dim=INPUT_DIM,
        value_dim=N_CLASSES,
        max_entries=FAM_ENTRIES,
        vigilance=FAM_VIGILANCE,
        curvature=curvature,
        hebb_lr=FAM_HEBB_LR,
        key_lr=FAM_KEY_LR,
        inference_k=FAM_INF_K,
        inference_temp=FAM_INF_TEMP,
        adaptive_eviction=True,
    ).to(DEVICE)


@torch.no_grad()
def learn_batched(model, feats, labels, batch=BATCH_SIZE, is_cam=False):
    """Train in batches."""
    x = feats.to(DEVICE)
    y = labels.long().to(DEVICE)
    for i in range(0, len(x), batch):
        bx = x[i:i+batch]
        by = y[i:i+batch]
        if is_cam:
            targets = F.one_hot(by, num_classes=N_CLASSES).float()
            model.learn_local(bx, targets)
        else:
            model.learn_local(bx, by)


@torch.no_grad()
def evaluate(model, feats, labels, batch=EVAL_BATCH, is_cam=False):
    """Evaluate top-1 accuracy."""
    x = feats.to(DEVICE)
    y = labels.long().to(DEVICE)
    correct = 0
    total = 0
    for i in range(0, len(x), batch):
        bx = x[i:i+batch]
        by = y[i:i+batch]
        if is_cam:
            outputs = model(bx)
        else:
            outputs = model(bx)
        preds = outputs.argmax(dim=-1)
        correct += (preds == by).sum().item()
        total += len(by)
    return 100.0 * correct / total


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


def compute_spearman_rho(radii_per_class: dict[int, float],
                          depths: dict[int, int]) -> float:
    """Compute Spearman rank correlation between mean prototype radius and synset depth."""
    from scipy.stats import spearmanr

    common_classes = sorted(set(radii_per_class.keys()) & set(depths.keys()))
    if len(common_classes) < 3:
        return 0.0
    r = [radii_per_class[c] for c in common_classes]
    d = [depths[c] for c in common_classes]
    rho, pval = spearmanr(r, d)
    return float(rho), float(pval)


def plot_radial_distribution(radii: torch.Tensor, classes: torch.Tensor,
                              depths: dict[int, int], out_path: Path):
    """Plot prototype radial distribution histogram, colored by WordNet depth."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    radii_np = radii.cpu().numpy()
    classes_np = classes.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram of all radii
    ax = axes[0]
    ax.hist(radii_np, bins=50, color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Lorentz Radius (arccosh of time coordinate)')
    ax.set_ylabel('Count')
    ax.set_title('Prototype Radial Distribution')
    ax.axvline(np.mean(radii_np), color='red', linestyle='--', label=f'Mean: {np.mean(radii_np):.3f}')
    ax.legend()

    # Right: scatter of mean radius per class vs WordNet depth
    ax = axes[1]
    class_radii = defaultdict(list)
    for r, c in zip(radii_np, classes_np):
        class_radii[int(c)].append(r)

    xs, ys = [], []
    for c in sorted(class_radii.keys()):
        if c in depths:
            xs.append(depths[c])
            ys.append(np.mean(class_radii[c]))

    ax.scatter(xs, ys, alpha=0.6, s=30, c='steelblue', edgecolors='black', linewidth=0.3)
    ax.set_xlabel('WordNet Synset Depth')
    ax.set_ylabel('Mean Lorentz Radius')
    ax.set_title('Radius vs Semantic Depth')

    # Add trend line
    if len(xs) > 2:
        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(xs), max(xs), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.7, label=f'Linear fit')
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {out_path}")


def run_variant(name: str, curvature: float = 1.0,
                hybrid_decay: bool = False) -> dict:
    """Run a single evaluation variant."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    set_seeds()
    train_feats, train_labels, test_feats, test_labels = load_dogs()

    # --- Euclidean baseline ---
    print("\n  [Euclidean baseline]")
    set_seeds()
    fam_euc = make_euclidean_fam()

    t0 = time.perf_counter()
    learn_batched(fam_euc, train_feats, train_labels, is_cam=False)
    euc_learn_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    euc_acc = evaluate(fam_euc, test_feats, test_labels, is_cam=False)
    euc_eval_time = time.perf_counter() - t0
    euc_total = euc_learn_time + euc_eval_time

    print(f"    Accuracy: {euc_acc:.2f}%")
    print(f"    Learn: {euc_learn_time:.2f}s, Eval: {euc_eval_time:.2f}s, Total: {euc_total:.2f}s")
    euc_stats = fam_euc.core_cam.get_stats()
    print(f"    Prototypes: {euc_stats['n_occupied']}")

    # --- Hyperbolic ---
    print(f"\n  [Hyperbolic (c={curvature})]")
    set_seeds()
    cam_hyp = make_hyperbolic_cam(curvature=curvature)

    if hybrid_decay:
        # Hybrid radial decay: reduce curvature exponentially during training
        # so early prototypes are placed with high curvature (fine-grained
        # distinctions near the origin) and later prototypes use lower
        # curvature (broader grouping further out)
        print("    Hybrid radial decay enabled")

    t0 = time.perf_counter()
    train_x = train_feats.to(DEVICE)
    train_y = train_labels.long().to(DEVICE)
    n_batches = (len(train_x) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(train_x), BATCH_SIZE):
        bx = train_x[i:i+BATCH_SIZE]
        by = train_y[i:i+BATCH_SIZE]
        targets = F.one_hot(by, num_classes=N_CLASSES).float()

        if hybrid_decay:
            # Exponential decay of curvature from 1.0 to 0.1 over training
            progress = i / max(len(train_x) - 1, 1)
            cam_hyp.curvature = 1.0 * (0.1 ** progress)

        cam_hyp.learn_local(bx, targets)

    if hybrid_decay:
        cam_hyp.curvature = curvature  # Reset for eval

    hyp_learn_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    hyp_acc = evaluate(cam_hyp, test_feats, test_labels, is_cam=True)
    hyp_eval_time = time.perf_counter() - t0
    hyp_total = hyp_learn_time + hyp_eval_time

    overhead = hyp_total / max(euc_total, 1e-6)

    print(f"    Accuracy: {hyp_acc:.2f}%")
    print(f"    Learn: {hyp_learn_time:.2f}s, Eval: {hyp_eval_time:.2f}s, Total: {hyp_total:.2f}s")
    print(f"    Overhead: {overhead:.3f}x")
    hyp_stats = cam_hyp.get_stats()
    print(f"    Prototypes: {hyp_stats['n_occupied']}")
    print(f"    Vigilance distance: {hyp_stats.get('vigilance_distance', 'N/A')}")
    if 'mean_radius' in hyp_stats:
        print(f"    Mean radius: {hyp_stats['mean_radius']:.4f}")
        print(f"    Radius range: [{hyp_stats['min_radius']:.4f}, {hyp_stats['max_radius']:.4f}]")

    # --- Spearman rho ---
    print("\n  [Spearman rho: radius vs WordNet depth]")
    radii = cam_hyp.get_prototype_radii()
    classes = cam_hyp.get_prototype_classes()
    depths = get_wordnet_depths()

    class_radii = defaultdict(list)
    for r, c in zip(radii.cpu().tolist(), classes.cpu().tolist()):
        class_radii[c].append(r)
    mean_radii = {c: np.mean(rs) for c, rs in class_radii.items()}

    rho, pval = compute_spearman_rho(mean_radii, depths)
    print(f"    Spearman rho = {rho:.4f} (p = {pval:.4e})")

    # --- Verdict ---
    if overhead > MAX_OVERHEAD:
        verdict = "KILL"
        reason = f"overhead {overhead:.3f}x > {MAX_OVERHEAD}x"
    elif hyp_acc >= PASS_ACC and overhead <= PASS_OVERHEAD and rho > PASS_RHO:
        verdict = "PASS"
        reason = f"acc={hyp_acc:.2f}% >= {PASS_ACC}%, overhead={overhead:.3f}x <= {PASS_OVERHEAD}x, rho={rho:.4f} > {PASS_RHO}"
    elif hyp_acc >= GREY_ACC:
        verdict = "GREY"
        reason = f"acc={hyp_acc:.2f}% in [{GREY_ACC}%, {PASS_ACC}%)"
    else:
        verdict = "KILL"
        reason = f"acc={hyp_acc:.2f}% < {GREY_ACC}%"

    print(f"\n  VERDICT: {verdict} — {reason}")

    # --- Plot ---
    suffix = "_decay" if hybrid_decay else ""
    plot_path = OUT_DIR / f"radial_distribution{suffix}.png"
    plot_radial_distribution(radii, classes, depths, plot_path)

    return {
        "name": name,
        "hyp_acc": hyp_acc,
        "euc_acc": euc_acc,
        "overhead": overhead,
        "rho": rho,
        "rho_pval": pval,
        "verdict": verdict,
        "reason": reason,
        "hyp_stats": hyp_stats,
        "euc_stats": euc_stats,
        "euc_total_s": euc_total,
        "hyp_total_s": hyp_total,
        "curvature": curvature,
        "hybrid_decay": hybrid_decay,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("EXP-1: HFAM Phase 1 Validation")
    print(f"Device: {DEVICE}")
    print(f"Output: {OUT_DIR}")

    results = []

    # --- Run 1: Hyperbolic, no decay ---
    r1 = run_variant("Hyperbolic (no adapters, no decay)", curvature=1.0, hybrid_decay=False)
    results.append(r1)

    # --- Run 2: Grey zone retry (if applicable) ---
    if r1["verdict"] == "GREY":
        print("\n\n*** GREY ZONE: Retrying with hybrid radial decay ***")
        r2 = run_variant("Hyperbolic + hybrid radial decay", curvature=1.0, hybrid_decay=True)
        results.append(r2)
        final_verdict = r2["verdict"]
    else:
        final_verdict = r1["verdict"]

    # --- Results table ---
    print("\n\n" + "=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)
    header = f"{'Run Variant':<45} {'Top-1':>7} {'Overhead':>10} {'rho':>8} {'Verdict':>8}"
    print(header)
    print("-" * 80)
    for r in results:
        line = f"{r['name']:<45} {r['hyp_acc']:>6.2f}% {r['overhead']:>9.3f}x {r['rho']:>8.4f} {r['verdict']:>8}"
        print(line)
    print("-" * 80)
    print(f"Euclidean baseline accuracy: {results[0]['euc_acc']:.2f}%")
    print(f"\nFINAL VERDICT: {final_verdict}")

    # --- Save JSON ---
    json_path = OUT_DIR / "exp1_results.json"
    with open(json_path, "w") as f:
        # Convert non-serializable types
        def clean(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj
        json.dump([{k: clean(v) for k, v in r.items()} for r in results], f, indent=2, default=str)
    print(f"\nResults saved: {json_path}")


if __name__ == "__main__":
    main()
