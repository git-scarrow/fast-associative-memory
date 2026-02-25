#!/usr/bin/env python3
"""
benchmarks/mt6_lyapunov.py — MT-6: Write-Path Lyapunov Exponent.

Measures how fast prototype-store divergence grows as a function of
perturbation size (number of batch swaps).  Determines whether the
write path is chaotic (positive Lyapunov exponent) or controllable.

From a shared empty S0, builds a reference S1 (forward order) and a
series of perturbed stores S_k (k adjacent-pair swaps applied to the
forward order).  Measures divergence vs S1 at each perturbation level.

Usage:
    PYTHONPATH=. python benchmarks/mt6_lyapunov.py [--device auto] [--data-root data]
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time as _time_mod
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
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
K_LEVELS = [0, 1, 2, 5, 10, 20, 47]  # Plus "full_reverse" as separate condition
PERTURBATION_SEED = 12345  # Separate from torch seed
TAG = "MT-6"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Eviction-logging FAM
# ---------------------------------------------------------------------------

class EvictionLoggingFAM(FastAssociativeMemory):
    """FAM that logs eviction events from its inner ContinuousCAM."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._eviction_log: List[Tuple[int, int]] = []
        self._current_step = 0
        self._orig_alloc = self.core_cam._alloc_slots_batch
        self.core_cam._alloc_slots_batch = self._logging_alloc

    def _logging_alloc(self, n: int):
        cam = self.core_cam
        free = (~cam.occupied).nonzero(as_tuple=True)[0]
        n_clamped = min(n, cam.max_entries)
        if len(free) >= n_clamped:
            return free[:n_clamped]
        occupied_before = cam.occupied.clone()
        result = self._orig_alloc(n)
        for slot in result.tolist():
            if occupied_before[slot]:
                self._eviction_log.append((self._current_step, slot))
        return result


# ---------------------------------------------------------------------------
# Store helpers
# ---------------------------------------------------------------------------

def make_fam(adapter, n_classes: int, device: torch.device) -> EvictionLoggingFAM:
    nstp = NSTPController(sibling_threshold=0.85, depth_epsilon=0.10)
    fam = EvictionLoggingFAM(
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


def clone_fam(src: EvictionLoggingFAM, device: torch.device) -> EvictionLoggingFAM:
    nstp = NSTPController(sibling_threshold=0.85, depth_epsilon=0.10)
    dst = EvictionLoggingFAM(
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
    dst._eviction_log = list(src._eviction_log)
    dst._current_step = src._current_step
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
            fam._current_step += 1
            fam.learn_local(feats[s:e], labels[s:e])
    finally:
        _time_mod.time = _orig


# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------

def make_perturbed_indices(
    forward_indices: List[Tuple[int, int]],
    k_swaps: int,
    seed: int,
) -> List[Tuple[int, int]]:
    """Apply k random adjacent-pair swaps to the batch order."""
    rng = random.Random(seed)
    order = list(range(len(forward_indices)))
    for _ in range(k_swaps):
        i = rng.randint(0, len(order) - 2)
        order[i], order[i + 1] = order[i + 1], order[i]
    return [forward_indices[j] for j in order]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_cosine_sim(fam_a, fam_b) -> Dict:
    cam_a, cam_b = fam_a.core_cam, fam_b.core_cam
    both_occ = cam_a.occupied & cam_b.occupied
    n_both = both_occ.sum().item()
    if n_both == 0:
        return {"mean": 0.0, "min": 0.0, "p5": 0.0, "n_compared": 0}
    idx = both_occ.nonzero(as_tuple=True)[0]
    keys_a = F.normalize(cam_a.keys[idx].float(), dim=-1)
    keys_b = F.normalize(cam_b.keys[idx].float(), dim=-1)
    cos = (keys_a * keys_b).sum(dim=-1)
    return {
        "mean": cos.mean().item(),
        "min": cos.min().item(),
        "p5": cos.kthvalue(max(1, int(0.05 * n_both))).values.item(),
        "n_compared": n_both,
    }


@torch.no_grad()
def occupancy_agreement(fam_a, fam_b) -> float:
    return (fam_a.core_cam.occupied == fam_b.core_cam.occupied).float().mean().item()


@torch.no_grad()
def eval_accuracy(fam, test_feats, test_labels) -> float:
    correct = total = 0
    for i in range(0, len(test_feats), 256):
        out = fam(test_feats[i:i + 256])
        correct += (out.argmax(1) == test_labels[i:i + 256]).sum().item()
        total += min(256, len(test_feats) - i)
    return correct / total * 100 if total > 0 else 0.0


def eviction_jaccard(fam_a, fam_b) -> float:
    set_a = {s for _, s in fam_a._eviction_log}
    set_b = {s for _, s in fam_b._eviction_log}
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=f"{TAG}: Write-Path Lyapunov Exponent")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-root", type=str, default="data")
    args = parser.parse_args()

    device = (torch.device("cuda") if args.device == "auto" and torch.cuda.is_available()
              else torch.device(args.device if args.device != "auto" else "cpu"))
    print(f"[{TAG}] Device: {device}", flush=True)
    set_all_seeds(SEED)

    # Load data
    print(f"[{TAG}] Loading Stanford Dogs...", flush=True)
    tr_f, tr_l, te_f, te_l, n_classes = g30_load_domain_features("dogs", Path(args.data_root))
    train_feats = tr_f.to(device)
    train_labels = tr_l.to(device)
    test_feats = te_f.to(device)
    test_labels = te_l.to(device)

    adapter = _load_g30_domain_adapter("dogs", input_dim=int(tr_f.shape[1]), device=device)

    # Batch indices
    n_samples = len(train_feats)
    forward_indices = [(i, min(i + BATCH_SIZE, n_samples))
                       for i in range(0, n_samples, BATCH_SIZE)]
    n_batches = len(forward_indices)
    print(f"[{TAG}] {n_batches} batches ({n_samples} samples), MAX_ENTRIES={MAX_ENTRIES}", flush=True)

    # Build reference S1 (forward order)
    print(f"[{TAG}] Building reference S1 (forward)...", flush=True)
    S0 = make_fam(adapter, n_classes, device)
    S1 = clone_fam(S0, device)
    feed_batches(S1, train_feats, train_labels, forward_indices, 1_000_000.0)
    s1_acc = eval_accuracy(S1, test_feats[:2000], test_labels[:2000])
    print(f"[{TAG}] S1: {S1.core_cam.occupied.sum().item()} protos, acc={s1_acc:.2f}%", flush=True)

    # Build conditions
    conditions = []
    for k in K_LEVELS:
        k_capped = min(k, n_batches)
        indices = make_perturbed_indices(forward_indices, k_capped, PERTURBATION_SEED + k)
        conditions.append((f"swap_{k}", k, indices))
    # Full reversal
    conditions.append(("full_reverse", n_batches, list(reversed(forward_indices))))

    # Run each condition
    results = []
    for cond_name, k_swaps, batch_order in conditions:
        print(f"\n[{TAG}] Condition: {cond_name} (k={k_swaps})...", flush=True)
        S_k = clone_fam(S0, device)
        feed_batches(S_k, train_feats, train_labels, batch_order, 1_000_000.0)

        cos = compute_cosine_sim(S1, S_k)
        occ = occupancy_agreement(S1, S_k)
        acc_k = eval_accuracy(S_k, test_feats[:2000], test_labels[:2000])
        ej = eviction_jaccard(S1, S_k)

        row = {
            "condition": cond_name,
            "k_swaps": k_swaps,
            "cos_mean": cos["mean"],
            "cos_min": cos["min"],
            "cos_p5": cos["p5"],
            "occ_agreement": occ,
            "acc_s1": s1_acc,
            "acc_sk": acc_k,
            "acc_delta": abs(s1_acc - acc_k),
            "eviction_jaccard": ej,
            "n_evictions_s1": len(S1._eviction_log),
            "n_evictions_sk": len(S_k._eviction_log),
        }
        results.append(row)
        print(f"  cos={cos['mean']:.6f} occ={occ:.4f} acc={acc_k:.2f}% "
              f"(delta={abs(s1_acc - acc_k):.2f}pp) evict_J={ej:.4f}", flush=True)

        del S_k
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # Analysis: fit divergence curves
    # -------------------------------------------------------------------
    # Use 1 - cos_mean as divergence measure (0 = identical, ~1 = orthogonal)
    # Exclude k=0 from fitting (it's the sanity check)
    swap_results = [r for r in results if r["condition"].startswith("swap_") and r["k_swaps"] > 0]
    ks = np.array([r["k_swaps"] for r in swap_results], dtype=float)
    div_cos = np.array([1.0 - r["cos_mean"] for r in swap_results])
    div_acc = np.array([r["acc_delta"] for r in swap_results])

    # Power law fit: log(div) = log(a) + β * log(k)
    def fit_power(x, y):
        mask = (x > 0) & (y > 0)
        if mask.sum() < 2:
            return {"a": float("nan"), "beta": float("nan"), "r2": float("nan")}
        lx, ly = np.log(x[mask]), np.log(y[mask])
        coeffs = np.polyfit(lx, ly, 1)
        beta, log_a = coeffs
        a = np.exp(log_a)
        y_pred = np.exp(np.polyval(coeffs, lx))
        ss_res = ((y[mask] - y_pred) ** 2).sum()
        ss_tot = ((y[mask] - y[mask].mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-15)
        return {"a": float(a), "beta": float(beta), "r2": float(r2)}

    # Exponential fit: log(div) = log(a) + λ * k
    def fit_exp(x, y):
        mask = (x > 0) & (y > 0)
        if mask.sum() < 2:
            return {"a": float("nan"), "lambda": float("nan"), "r2": float("nan")}
        ly = np.log(y[mask])
        coeffs = np.polyfit(x[mask], ly, 1)
        lam, log_a = coeffs
        a = np.exp(log_a)
        y_pred = np.exp(np.polyval(coeffs, x[mask]))
        ss_res = ((y[mask] - y_pred) ** 2).sum()
        ss_tot = ((y[mask] - y[mask].mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-15)
        return {"a": float(a), "lambda": float(lam), "r2": float(r2)}

    cos_power = fit_power(ks, div_cos)
    cos_exp = fit_exp(ks, div_cos)
    acc_power = fit_power(ks, div_acc)
    acc_exp = fit_exp(ks, div_acc)

    # -------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path("results/mt6")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Divergence curve (the money plot)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: cosine divergence
    ax = axes[0]
    ax.scatter(ks, div_cos, color="#1976d2", zorder=5, s=60, label="data")
    if not np.isnan(cos_power["beta"]):
        x_fit = np.linspace(0.5, max(ks) * 1.1, 200)
        ax.plot(x_fit, cos_power["a"] * x_fit ** cos_power["beta"],
                "--", color="#d32f2f", label=f"power: β={cos_power['beta']:.2f}, R²={cos_power['r2']:.3f}")
    if not np.isnan(cos_exp["lambda"]):
        ax.plot(x_fit, cos_exp["a"] * np.exp(cos_exp["lambda"] * x_fit),
                ":", color="#7b1fa2", label=f"exp: λ={cos_exp['lambda']:.4f}, R²={cos_exp['r2']:.3f}")
    # Full reverse marker
    fr = next(r for r in results if r["condition"] == "full_reverse")
    ax.scatter([fr["k_swaps"]], [1.0 - fr["cos_mean"]], marker="x", s=100,
               color="#ff5722", zorder=6, label=f"full reverse")
    ax.set_xlabel("k (adjacent-pair swaps)")
    ax.set_ylabel("1 − cos_mean (divergence)")
    ax.set_title("Key Cosine Divergence vs Perturbation Size")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: accuracy delta
    ax = axes[1]
    ax.scatter(ks, div_acc, color="#1976d2", zorder=5, s=60, label="data")
    if not np.isnan(acc_power["beta"]):
        ax.plot(x_fit, np.clip(acc_power["a"] * x_fit ** acc_power["beta"], 0, 100),
                "--", color="#d32f2f", label=f"power: β={acc_power['beta']:.2f}, R²={acc_power['r2']:.3f}")
    ax.scatter([fr["k_swaps"]], [fr["acc_delta"]], marker="x", s=100,
               color="#ff5722", zorder=6, label="full reverse")
    ax.set_xlabel("k (adjacent-pair swaps)")
    ax.set_ylabel("Accuracy delta (pp)")
    ax.set_title("Accuracy Divergence vs Perturbation Size")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{TAG}: Write-Path Lyapunov — Divergence vs Perturbation Size")
    fig.tight_layout()
    fig.savefig(out_dir / "mt6_divergence_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Eviction curve
    fig, ax = plt.subplots(figsize=(8, 5))
    all_ks = [r["k_swaps"] for r in results if r["k_swaps"] > 0]
    all_ej = [r["eviction_jaccard"] for r in results if r["k_swaps"] > 0]
    ax.scatter(all_ks, all_ej, color="#1976d2", s=60)
    ax.set_xlabel("k (perturbation size)")
    ax.set_ylabel("Eviction Jaccard overlap with S1")
    ax.set_title(f"{TAG}: Eviction Overlap vs Perturbation Size")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "mt6_eviction_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------
    # Save metrics JSON
    # -------------------------------------------------------------------
    metrics = {
        "results": results,
        "fits": {
            "cos_divergence": {"power_law": cos_power, "exponential": cos_exp},
            "acc_divergence": {"power_law": acc_power, "exponential": acc_exp},
        },
    }
    with open(out_dir / "mt6_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # -------------------------------------------------------------------
    # Writeup
    # -------------------------------------------------------------------
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()

    # Determine verdict
    best_cos_fit = "power" if cos_power["r2"] >= cos_exp["r2"] else "exponential"
    best_acc_fit = "power" if acc_power["r2"] >= acc_exp["r2"] else "exponential"

    k0 = next((r for r in results if r["k_swaps"] == 0), None)
    k1 = next((r for r in results if r["condition"] == "swap_1"), None)

    writeup = f"""# MT-6: Write-Path Lyapunov Exponent

## Context

MT-4 showed full batch reversal produces 47pp accuracy gap. MT-5 proved the
divergence is in prototype positions, not Mahalanobis calibration. MT-6 measures
how divergence scales with perturbation size to determine if the write path is
**chaotic** (positive Lyapunov exponent) or **controllable** (sublinear/linear).

## Config
- **Commit**: `{commit}`
- **Seed**: {SEED} (perturbation seed: {PERTURBATION_SEED})
- **Device**: {device}
- **Dataset**: Stanford Dogs 1024-d, production adapter, MAX_ENTRIES={MAX_ENTRIES}
- **Reference S1 accuracy**: {s1_acc:.2f}%
- **Perturbation**: k adjacent-pair swaps applied to forward batch order
- **Levels**: k ∈ {{{', '.join(str(r['k_swaps']) for r in results)}}}

## Results

| Condition | k | Cos sim | 1−cos (div) | Occ agree | Acc S_k | Acc delta | Evict J |
|---|---|---|---|---|---|---|---|
"""
    for r in results:
        writeup += (
            f"| {r['condition']} | {r['k_swaps']} | {r['cos_mean']:.6f} | "
            f"{1 - r['cos_mean']:.6f} | {r['occ_agreement']:.4f} | "
            f"{r['acc_sk']:.2f}% | {r['acc_delta']:.2f}pp | {r['eviction_jaccard']:.4f} |\n"
        )

    writeup += f"""
## Sanity Check
- k=0 (identical to S1): cos={k0['cos_mean']:.6f}, acc_delta={k0['acc_delta']:.2f}pp {'PASS' if k0['cos_mean'] > 0.9999 else 'FAIL'}

## Curve Fits

### Cosine divergence (1 − cos_mean) vs k
- **Power law** `div = {cos_power['a']:.4e} × k^{cos_power['beta']:.2f}`: R² = {cos_power['r2']:.4f}
- **Exponential** `div = {cos_exp['a']:.4e} × exp({cos_exp['lambda']:.4f} × k)`: R² = {cos_exp['r2']:.4f}
- **Best fit**: {best_cos_fit}

### Accuracy delta vs k
- **Power law** `Δacc = {acc_power['a']:.4e} × k^{acc_power['beta']:.2f}`: R² = {acc_power['r2']:.4f}
- **Exponential** `Δacc = {acc_exp['a']:.4e} × exp({acc_exp['lambda']:.4f} × k)`: R² = {acc_exp['r2']:.4f}
- **Best fit**: {best_acc_fit}

## Verdict

"""

    if cos_power["beta"] > 0.8 and cos_power["r2"] > 0.8:
        if cos_power["beta"] > 1.5:
            verdict = "CHAOTIC"
            detail = (
                f"Cosine divergence grows **superlinearly** (β = {cos_power['beta']:.2f}). "
                "The write path has a positive Lyapunov exponent — small order perturbations "
                "amplify exponentially. Any arrival-order jitter (e.g., inotify timing in "
                "Shutter-Deck) will produce a materially different store after enough ingestion."
            )
        elif cos_power["beta"] > 0.8:
            verdict = "SENSITIVE BUT LINEAR"
            detail = (
                f"Cosine divergence grows **approximately linearly** (β = {cos_power['beta']:.2f}). "
                "The write path is order-sensitive but not chaotic — perturbation effects "
                "accumulate proportionally, not explosively. Single batch swaps produce bounded "
                "divergence."
            )
    elif cos_exp["lambda"] > 0 and cos_exp["r2"] > cos_power["r2"]:
        verdict = "CHAOTIC (EXPONENTIAL)"
        detail = (
            f"Cosine divergence fits exponential growth better (λ = {cos_exp['lambda']:.4f}, "
            f"R² = {cos_exp['r2']:.3f}). Positive Lyapunov exponent confirmed."
        )
    else:
        verdict = "SUBLINEAR / STABLE"
        detail = (
            f"Cosine divergence grows **sublinearly** (β = {cos_power['beta']:.2f}). "
            "The write path absorbs small perturbations — the system is controllable."
        )

    writeup += f"**{verdict}**\n\n{detail}\n"

    # Single-swap sensitivity
    if k1:
        writeup += f"""
## Single-Swap Sensitivity
- One adjacent batch swap (k=1) produces:
  - Cosine divergence: {1 - k1['cos_mean']:.6f}
  - Accuracy delta: {k1['acc_delta']:.2f}pp
  - Eviction Jaccard: {k1['eviction_jaccard']:.4f}
"""

    writeup += f"""
## Plots
- `mt6_divergence_curve.png` — Divergence vs k on log-log axes with power/exp fits
- `mt6_eviction_curve.png` — Eviction Jaccard vs k

## Artifacts
- `mt6_metrics.json` — all metrics and fit parameters
"""

    (out_dir / "mt6_lyapunov.md").write_text(writeup)
    print(f"\n[{TAG}] Report saved to {out_dir / 'mt6_lyapunov.md'}")
    print(writeup)


if __name__ == "__main__":
    main()
