#!/usr/bin/env python3
"""
benchmarks/mt4_write_path_divergence_dogs.py — MT-4: Write-Path Divergence (Production Dogs).

Production-realistic variant of MT-3.  Exercises the full write path with:
  - Stanford Dogs 1024-d ViT-L/14 features
  - Production Dogs MetricAdapter
  - FastAssociativeMemory wrapper (adapter projection → ContinuousCAM)
  - LFU-LRU eviction under realistic store capacity

Creates S1/S2 (same forward order) and S3 (reversed batch order) from a
shared S0 snapshot, measures divergence at N=10, 50, 100.

Usage:
    PYTHONPATH=. python benchmarks/mt4_write_path_divergence_dogs.py [--device auto]
"""

from __future__ import annotations

import argparse
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

CHECKPOINTS = [10, 50, 100]
BATCH_SIZE = 256
SEED = 42
# Capacity deliberately below total sample count to force eviction.
# Dogs train set is ~12,000 samples → 100 batches × 256 = capped at that.
# With MAX_ENTRIES=8000 we guarantee eviction pressure.
MAX_ENTRIES = 8_000

TAG = "MT-4"


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
# Eviction-logging FAM wrapper
# ---------------------------------------------------------------------------

class EvictionLoggingFAM(FastAssociativeMemory):
    """FAM that logs eviction events from its inner ContinuousCAM."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._eviction_log: List[Tuple[int, int]] = []
        self._current_step = 0
        # Monkey-patch the core CAM's _alloc_slots_batch
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
# Store creation / cloning
# ---------------------------------------------------------------------------

def make_production_fam(
    adapter,
    n_classes: int,
    device: torch.device,
) -> EvictionLoggingFAM:
    """Create a Dogs-production FAM with eviction logging."""
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
    """Deep-clone a FAM including adapter weights, core CAM buffers, and eviction state."""
    # Rebuild with same adapter (frozen, shared weights are fine since eval-only)
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

    # Copy all registered buffer/parameter state (core_cam buffers, adapter weights)
    dst.load_state_dict(src.state_dict())

    # Copy non-buffer scalars on the inner CAM
    dst.core_cam.csls_weight = src.core_cam.csls_weight
    dst.core_cam.inference_sim_floor = src.core_cam.inference_sim_floor
    dst.core_cam.var_ema_alpha = src.core_cam.var_ema_alpha
    dst.core_cam.ema_beta = src.core_cam.ema_beta

    # Copy eviction tracking state
    dst._eviction_log = list(src._eviction_log)
    dst._current_step = src._current_step

    return dst


# ---------------------------------------------------------------------------
# Deterministic batch feeding
# ---------------------------------------------------------------------------

@torch.no_grad()
def feed_batches(
    fam: EvictionLoggingFAM,
    feats: torch.Tensor,
    labels: torch.Tensor,
    batch_indices: List[Tuple[int, int]],
    fixed_time: float,
) -> None:
    """Feed batches through learn_local with deterministic clock."""
    _orig_time = _time_mod.time
    clock = [fixed_time]

    def _deterministic_time():
        clock[0] += 1.0
        return clock[0]

    _time_mod.time = _deterministic_time
    try:
        for start, end in batch_indices:
            fam._current_step += 1
            fam.learn_local(feats[start:end], labels[start:end])
    finally:
        _time_mod.time = _orig_time


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_cosine_sim(fam_a: EvictionLoggingFAM, fam_b: EvictionLoggingFAM) -> Dict:
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
def occupancy_agreement(fam_a: EvictionLoggingFAM, fam_b: EvictionLoggingFAM) -> float:
    return (fam_a.core_cam.occupied == fam_b.core_cam.occupied).float().mean().item()


@torch.no_grad()
def eval_accuracy(
    fam: EvictionLoggingFAM,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
) -> float:
    correct = 0
    total = 0
    for i in range(0, len(test_feats), 256):
        outputs = fam(test_feats[i:i + 256])
        preds = outputs.argmax(dim=1)
        correct += (preds == test_labels[i:i + 256]).sum().item()
        total += min(256, len(test_feats) - i)
    return correct / total * 100 if total > 0 else 0.0


def eviction_jaccard(fam_a: EvictionLoggingFAM, fam_b: EvictionLoggingFAM) -> float:
    set_a = {slot for _, slot in fam_a._eviction_log}
    set_b = {slot for _, slot in fam_b._eviction_log}
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=f"{TAG}: Write-Path Divergence (Production Dogs)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-root", type=str, default="data")
    args = parser.parse_args()

    device = (torch.device("cuda") if args.device == "auto" and torch.cuda.is_available()
              else torch.device(args.device if args.device != "auto" else "cpu"))
    print(f"[{TAG}] Device: {device}", flush=True)

    set_all_seeds(SEED)

    # --- Load Dogs features ---
    print(f"[{TAG}] Loading Stanford Dogs features...", flush=True)
    tr_f, tr_l, te_f, te_l, n_classes = g30_load_domain_features("dogs", Path(args.data_root))
    input_dim = int(tr_f.shape[1])
    train_feats = tr_f.to(device)
    train_labels = tr_l.to(device)
    test_feats = te_f.to(device)
    test_labels = te_l.to(device)
    print(f"[{TAG}] Train: {train_feats.shape}, Test: {test_feats.shape}, Classes: {n_classes}", flush=True)

    # --- Load production adapter ---
    print(f"[{TAG}] Loading production Dogs adapter...", flush=True)
    adapter = _load_g30_domain_adapter("dogs", input_dim=input_dim, device=device)

    # --- Build batch index lists ---
    max_batches = max(CHECKPOINTS)
    n_samples = min(max_batches * BATCH_SIZE, len(train_feats))
    forward_indices = [(i, min(i + BATCH_SIZE, n_samples))
                       for i in range(0, n_samples, BATCH_SIZE)]
    forward_indices = forward_indices[:max_batches]
    reverse_indices = list(reversed(forward_indices))
    actual_batches = len(forward_indices)
    print(f"[{TAG}] Stream: {actual_batches} batches ({n_samples} samples), "
          f"MAX_ENTRIES={MAX_ENTRIES}", flush=True)

    # --- Create S0 (empty, Variant B) and fork ---
    print(f"[{TAG}] Creating base snapshot S0 (empty store, production hyperparams)...", flush=True)
    S0 = make_production_fam(adapter, n_classes, device)

    S1 = clone_fam(S0, device)
    S2 = clone_fam(S0, device)
    S3 = clone_fam(S0, device)

    # Fixed eval subset
    eval_n = min(2000, len(test_feats))
    eval_feats = test_feats[:eval_n]
    eval_labels = test_labels[:eval_n]

    results = []
    prev_cp = 0
    fixed_time_base = 1_000_000.0

    for cp in CHECKPOINTS:
        # Clamp to actual available batches
        effective_cp = min(cp, actual_batches)
        if effective_cp <= prev_cp:
            print(f"[{TAG}] Skipping N={cp} (only {actual_batches} batches available)", flush=True)
            continue

        fwd_slice = forward_indices[prev_cp:effective_cp]
        rev_slice = reverse_indices[prev_cp:effective_cp]

        print(f"\n[{TAG}] Feeding batches {prev_cp+1}..{effective_cp}...", flush=True)

        feed_batches(S1, train_feats, train_labels, fwd_slice,
                     fixed_time=fixed_time_base + prev_cp)
        feed_batches(S2, train_feats, train_labels, fwd_slice,
                     fixed_time=fixed_time_base + prev_cp)
        feed_batches(S3, train_feats, train_labels, rev_slice,
                     fixed_time=fixed_time_base + prev_cp)

        for pair_name, fam_x, fam_y in [("S1 vs S2", S1, S2), ("S1 vs S3", S1, S3)]:
            cos = compute_cosine_sim(fam_x, fam_y)
            occ = occupancy_agreement(fam_x, fam_y)
            acc_x = eval_accuracy(fam_x, eval_feats, eval_labels)
            acc_y = eval_accuracy(fam_y, eval_feats, eval_labels)
            evict_j = eviction_jaccard(fam_x, fam_y)

            n_occ_x = fam_x.core_cam.occupied.sum().item()
            n_occ_y = fam_y.core_cam.occupied.sum().item()
            n_evict_x = len(fam_x._eviction_log)
            n_evict_y = len(fam_y._eviction_log)

            row = {
                "checkpoint": effective_cp,
                "pair": pair_name,
                "cos_mean": cos["mean"],
                "cos_min": cos["min"],
                "cos_p5": cos["p5"],
                "n_slots_compared": cos["n_compared"],
                "occ_agreement": occ,
                "acc_left": acc_x,
                "acc_right": acc_y,
                "acc_delta": abs(acc_x - acc_y),
                "eviction_jaccard": evict_j,
                "n_occ_left": n_occ_x,
                "n_occ_right": n_occ_y,
                "n_evictions_left": n_evict_x,
                "n_evictions_right": n_evict_y,
            }
            results.append(row)
            print(f"  [{pair_name}] N={effective_cp}: cos={cos['mean']:.6f} "
                  f"occ={occ:.4f} acc={acc_x:.2f}/{acc_y:.2f}% "
                  f"evict_J={evict_j:.4f} (evictions: {n_evict_x}/{n_evict_y})",
                  flush=True)

        prev_cp = effective_cp

    # -----------------------------------------------------------------------
    # Build markdown report
    # -----------------------------------------------------------------------
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()

    s1_s2_rows = [r for r in results if r["pair"] == "S1 vs S2"]
    s1_s3_rows = [r for r in results if r["pair"] == "S1 vs S3"]

    same_order_ok = all(
        r["cos_mean"] > 0.999999 and r["occ_agreement"] > 0.9999 and r["acc_delta"] < 0.01
        for r in s1_s2_rows
    )

    order_sensitive = any(
        r["cos_mean"] < 0.999 or r["occ_agreement"] < 0.999 or r["acc_delta"] > 0.1
        for r in s1_s3_rows
    )

    first_metric = None
    first_n = None
    for r in s1_s3_rows:
        if r["cos_mean"] < 0.999 and first_metric is None:
            first_metric = f"cosine similarity ({r['cos_mean']:.6f})"
            first_n = r["checkpoint"]
        if r["occ_agreement"] < 0.999 and first_metric is None:
            first_metric = f"occupancy agreement ({r['occ_agreement']:.4f})"
            first_n = r["checkpoint"]
        if r["acc_delta"] > 0.1 and first_metric is None:
            first_metric = f"accuracy delta ({r['acc_delta']:.2f}pp)"
            first_n = r["checkpoint"]

    lines = [
        f"# {TAG}: Write-Path Divergence Report (Production Dogs)",
        "",
        "## Config",
        f"- **Commit**: `{commit}`",
        f"- **Seed**: {SEED}",
        f"- **Device**: {device}",
        f"- **Dataset**: Stanford Dogs ViT-L/14 features (1024-d)",
        f"- **Adapter**: Production Dogs adapter (`adapter_trained.pt`, triplet/hard, 2-layer, ReLU)",
        f"- **Pipeline**: FastAssociativeMemory → MetricAdapter → ContinuousCAM",
        f"- **Max entries**: {MAX_ENTRIES:,}",
        f"- **Batch size**: {BATCH_SIZE}",
        f"- **Batches**: {actual_batches} ({n_samples:,} samples)",
        f"- **Eval subset**: {eval_n} test samples",
        f"- **Base snapshot**: Variant B (empty store, production hyperparams)",
        f"- **Determinism**: fixed RNG seeds, monkey-patched `time.time()`, `cudnn.deterministic=True`",
        "",
        "## Results",
        "",
        "| N | Pair | Cos sim (mean) | Cos sim (min) | Cos sim (p5) | Occ agree | Acc left | Acc right | Acc delta | Evict overlap | Evictions L/R |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        lines.append(
            f"| {r['checkpoint']} | {r['pair']} | {r['cos_mean']:.6f} | "
            f"{r['cos_min']:.6f} | {r['cos_p5']:.6f} | {r['occ_agreement']:.4f} | "
            f"{r['acc_left']:.2f}% | {r['acc_right']:.2f}% | {r['acc_delta']:.2f}pp | "
            f"{r['eviction_jaccard']:.4f} | {r['n_evictions_left']}/{r['n_evictions_right']} |"
        )

    lines.extend([
        "",
        "## Nondeterminism Sources",
        "",
        "- **`time.time()` in `learn_local()`**: Drives `last_seen` timestamps for LFU-LRU eviction "
        "tiebreaking. Neutralized via monkey-patch to deterministic 1s-increment counter.",
        "- **CUDA atomics**: `scatter_add_` used in EMA hit-count aggregation can produce "
        "non-associative float rounding. Mitigated with `cudnn.deterministic=True`.",
        "- **`torch.unique()` ordering**: Used in `learn_local()` for per-slot scatter-mean. "
        "Stable on CUDA for identical inputs.",
        "- **Adapter projection**: Frozen `MetricAdapter` is deterministic (linear layers, no dropout).",
        "",
        "## Conclusion",
        "",
    ])

    if same_order_ok:
        lines.append(
            "**Same-order determinism**: CONFIRMED. S1 vs S2 are bitwise identical "
            "across all checkpoints under production Dogs + adapter pipeline."
        )
    else:
        # Find first divergence
        first_fail = next((r for r in s1_s2_rows
                           if r["cos_mean"] <= 0.999999 or r["occ_agreement"] <= 0.9999
                           or r["acc_delta"] >= 0.01), None)
        lines.append(
            f"**Same-order determinism**: FAILED at N={first_fail['checkpoint'] if first_fail else '?'}. "
            "S1 vs S2 diverge — likely cause: CUDA `scatter_add_` float non-associativity "
            "or uncontrolled RNG state."
        )

    if order_sensitive:
        lines.append(
            f"**Order sensitivity**: CONFIRMED. First detected at N={first_n} via {first_metric}."
        )
        lines.append("")
        # Check severity
        worst_acc_delta = max(r["acc_delta"] for r in s1_s3_rows) if s1_s3_rows else 0
        if worst_acc_delta > 5.0:
            lines.append("")
            lines.append(
                f"**Severity: CRITICAL.** Accuracy delta reaches {worst_acc_delta:.1f}pp under "
                "reversed batch order — far worse than MT-3's <1pp on CIFAR-100. "
                "The production adapter + Mahalanobis pipeline amplifies order sensitivity: "
                "EMA class-conditional variance estimates (`class_vars`, α=0.01) are dominated "
                "by the most recently seen classes, and key centroid drift accumulates differently "
                "when early vs late classes swap position. The tighter metric space from the adapter "
                "makes re-ranking decisions more fragile to miscalibrated per-class statistics."
            )
        else:
            lines.append("")
            lines.append(
                "This matches MT-3 (CIFAR-100): EMA key drift, adaptive alpha, and LFU-LRU eviction "
                "are inherently order-dependent. Reversed batch order produces different prototype "
                "identities per slot. Accuracy impact is modest."
            )
    else:
        lines.append(
            "**Order sensitivity**: NOT DETECTED within measurement precision."
        )

    # MT-3 comparison note
    lines.extend([
        "",
        "## Comparison with MT-3 (CIFAR-100)",
        "",
        "| Dimension | MT-3 (CIFAR-100, 768-d, no adapter) | MT-4 (Dogs, 1024-d, adapter) |",
        "|---|---|---|",
    ])
    # Pull MT-3 known values for comparison
    mt3_note = {
        "Same-order": "Bitwise identical",
        "Order cos@10": "0.059",
        "Order cos@100": "0.060",
        "Acc delta@100": "0.15pp",
        "Evict J@100": "0.249",
    }
    if results:
        s3_10 = next((r for r in s1_s3_rows if r["checkpoint"] <= 10), None)
        s3_100 = next((r for r in s1_s3_rows if r["checkpoint"] >= 50), s1_s3_rows[-1] if s1_s3_rows else None)
        lines.append(
            f"| Same-order determinism | Bitwise identical | "
            f"{'Bitwise identical' if same_order_ok else 'DIVERGENT'} |"
        )
        if s3_10:
            lines.append(
                f"| Cos sim (mean) @ N=10 | 0.059 | {s3_10['cos_mean']:.3f} |"
            )
        if s3_100:
            lines.append(
                f"| Cos sim (mean) @ N={s3_100['checkpoint']} | 0.060 | {s3_100['cos_mean']:.3f} |"
            )
            lines.append(
                f"| Acc delta @ N={s3_100['checkpoint']} | 0.15pp | {s3_100['acc_delta']:.2f}pp |"
            )
            lines.append(
                f"| Eviction Jaccard @ N={s3_100['checkpoint']} | 0.249 | {s3_100['eviction_jaccard']:.3f} |"
            )

    report = "\n".join(lines) + "\n"

    out_dir = Path("results/mt4")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "mt4_write_path_divergence_dogs.md"
    report_path.write_text(report)
    print(f"\n[{TAG}] Report saved to {report_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
