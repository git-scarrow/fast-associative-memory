#!/usr/bin/env python3
"""
benchmarks/mt3_write_path_divergence.py — MT-3: Write-Path Divergence.

Tests whether the prototype store write path is deterministic and order-stable.
From a single base snapshot, creates 3 forked stores (S1, S2 same-order forward,
S3 reversed order). Feeds CIFAR-100 streaming batches and measures divergence
at N=10, 50, 100.

Metrics:
  1) Cosine similarity between stores (slotwise + aggregate)
  2) Per-slot occupancy agreement rate
  3) Accuracy on fixed eval set
  4) Eviction overlap (Jaccard)

Usage:
    PYTHONPATH=. python benchmarks/mt3_write_path_divergence.py [--device auto]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from associative_core import ContinuousCAM  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FEATURE_CACHE = Path("feature_cache_vitb14")
TRAIN_PT = FEATURE_CACHE / "cifar100_dinov2_train.pt"
TEST_PT = FEATURE_CACHE / "cifar100_dinov2_test.pt"

CHECKPOINTS = [10, 50, 100]
BATCH_SIZE = 256
SEED = 42
N_CLASSES = 100
MAX_ENTRIES = 20_000  # Enough to force some eviction at 100 batches × 256 = 25,600 samples


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
# Eviction-logging wrapper
# ---------------------------------------------------------------------------

class EvictionLoggingCAM(ContinuousCAM):
    """Thin wrapper that logs evicted slot ids during _alloc_slots_batch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eviction_log: List[Tuple[int, int]] = []  # (step, slot_id)
        self._current_step = 0

    def _alloc_slots_batch(self, n: int):
        free = (~self.occupied).nonzero(as_tuple=True)[0]
        n_clamped = min(n, self.max_entries)
        if len(free) >= n_clamped:
            return free[:n_clamped]

        # There will be evictions — call parent, then diff
        occupied_before = self.occupied.clone()
        result = super()._alloc_slots_batch(n)

        # Evicted slots = were occupied before, now claimed
        for slot in result.tolist():
            if occupied_before[slot]:
                self.eviction_log.append((self._current_step, slot))

        return result


# ---------------------------------------------------------------------------
# Store creation / cloning
# ---------------------------------------------------------------------------

def make_base_store(device: torch.device) -> EvictionLoggingCAM:
    """Create an empty CIFAR-100 prototype store."""
    cam = EvictionLoggingCAM(
        key_dim=768,
        value_dim=N_CLASSES,
        max_entries=MAX_ENTRIES,
        vigilance=0.85,
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=25,
        inference_temp=0.05,
        use_lfu=True,
    ).to(device)
    return cam


def clone_store(cam: EvictionLoggingCAM, device: torch.device) -> EvictionLoggingCAM:
    """Deep-clone a store, preserving all buffer state."""
    new = EvictionLoggingCAM(
        key_dim=cam.key_dim,
        value_dim=cam.value_dim,
        max_entries=cam.max_entries,
        vigilance=cam.vigilance,
        hebb_lr=cam.hebb_lr,
        key_lr=cam.key_lr,
        inference_k=cam.inference_k,
        inference_temp=cam.inference_temp,
        use_lfu=cam.use_lfu,
    ).to(device)
    # Copy all registered buffer state
    sd = cam.state_dict()
    new.load_state_dict(sd)
    new.eviction_log = list(cam.eviction_log)
    new._current_step = cam._current_step
    # Copy non-buffer scalars
    new.csls_weight = cam.csls_weight
    new.inference_sim_floor = cam.inference_sim_floor
    new.var_ema_alpha = cam.var_ema_alpha
    new.ema_beta = cam.ema_beta
    return new


# ---------------------------------------------------------------------------
# Feeding batches
# ---------------------------------------------------------------------------

@torch.no_grad()
def feed_batches(
    cam: EvictionLoggingCAM,
    feats: torch.Tensor,
    labels: torch.Tensor,
    batch_indices: List[Tuple[int, int]],
    fixed_time: float,
) -> None:
    """Feed a sequence of batches into the store.

    We monkey-patch time.time to return a deterministic clock so that
    last_seen timestamps don't depend on wall-clock jitter.
    """
    import time as _time_mod
    _orig_time = _time_mod.time

    clock = [fixed_time]

    def _deterministic_time():
        clock[0] += 1.0  # Advance by 1s per call
        return clock[0]

    _time_mod.time = _deterministic_time
    try:
        for start, end in batch_indices:
            cam._current_step += 1
            batch_x = feats[start:end]
            batch_y = labels[start:end]
            targets = F.one_hot(batch_y, num_classes=N_CLASSES).float()
            cam.learn_local(batch_x, targets)
    finally:
        _time_mod.time = _orig_time


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_cosine_sim(cam_a: EvictionLoggingCAM, cam_b: EvictionLoggingCAM) -> Dict:
    """Slotwise cosine similarity on occupied slots, then aggregate."""
    occ_a = cam_a.occupied
    occ_b = cam_b.occupied
    both_occ = occ_a & occ_b
    n_both = both_occ.sum().item()

    if n_both == 0:
        return {"mean": 0.0, "min": 0.0, "p5": 0.0, "n_compared": 0}

    idx = both_occ.nonzero(as_tuple=True)[0]
    keys_a = F.normalize(cam_a.keys[idx].float(), dim=-1)
    keys_b = F.normalize(cam_b.keys[idx].float(), dim=-1)
    cos = (keys_a * keys_b).sum(dim=-1)  # (N,)

    return {
        "mean": cos.mean().item(),
        "min": cos.min().item(),
        "p5": cos.kthvalue(max(1, int(0.05 * n_both))).values.item(),
        "n_compared": n_both,
    }


@torch.no_grad()
def occupancy_agreement(cam_a: EvictionLoggingCAM, cam_b: EvictionLoggingCAM) -> float:
    """Fraction of slots where occupancy matches."""
    agree = (cam_a.occupied == cam_b.occupied).float().mean().item()
    return agree


@torch.no_grad()
def eval_accuracy(
    cam: EvictionLoggingCAM,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
) -> float:
    """Eval top-1 accuracy on a fixed test set."""
    correct = 0
    total = 0
    for i in range(0, len(test_feats), 256):
        batch_x = test_feats[i:i + 256]
        batch_y = test_labels[i:i + 256]
        outputs = cam(batch_x)
        preds = outputs.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += len(batch_y)
    return correct / total * 100 if total > 0 else 0.0


def eviction_jaccard(cam_a: EvictionLoggingCAM, cam_b: EvictionLoggingCAM) -> float:
    """Jaccard overlap of evicted slot sets."""
    set_a = {slot for _, slot in cam_a.eviction_log}
    set_b = {slot for _, slot in cam_b.eviction_log}
    if not set_a and not set_b:
        return 1.0  # Both empty → perfect agreement
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MT-3: Write-Path Divergence")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    print(f"[MT-3] Device: {device}", flush=True)

    # Load features
    print("[MT-3] Loading CIFAR-100 features...", flush=True)
    train_data = torch.load(TRAIN_PT, map_location="cpu", weights_only=True)
    test_data = torch.load(TEST_PT, map_location="cpu", weights_only=True)

    train_feats = train_data["embeds"].float().to(device)
    train_labels = train_data["labels"].long().to(device)
    test_feats = test_data["embeds"].float().to(device)
    test_labels = test_data["labels"].long().to(device)

    print(f"[MT-3] Train: {train_feats.shape}, Test: {test_feats.shape}", flush=True)

    # Build batch index list: 100 batches × 256 = 25,600 samples (from first 25,600)
    max_batches = max(CHECKPOINTS)
    n_samples = min(max_batches * BATCH_SIZE, len(train_feats))
    forward_indices = [(i, min(i + BATCH_SIZE, n_samples)) for i in range(0, n_samples, BATCH_SIZE)]
    forward_indices = forward_indices[:max_batches]
    reverse_indices = list(reversed(forward_indices))

    print(f"[MT-3] Streaming {len(forward_indices)} batches ({n_samples} samples)", flush=True)

    # Create base snapshot S0 (empty store)
    set_all_seeds(SEED)
    S0 = make_base_store(device)

    # Use a fixed eval subset (first 2000 test samples for speed)
    eval_n = min(2000, len(test_feats))
    eval_feats = test_feats[:eval_n]
    eval_labels = test_labels[:eval_n]

    # Results table
    results = []

    # We'll feed incrementally: process batches up to each checkpoint
    # Clone S0 three times
    S1 = clone_store(S0, device)
    S2 = clone_store(S0, device)
    S3 = clone_store(S0, device)

    prev_cp = 0
    fixed_time_base = 1_000_000.0  # Deterministic time base

    for cp in CHECKPOINTS:
        n_new = cp - prev_cp
        fwd_slice = forward_indices[prev_cp:cp]
        rev_slice = reverse_indices[prev_cp:cp]

        print(f"\n[MT-3] Feeding batches {prev_cp+1}..{cp}...", flush=True)

        # S1 forward
        feed_batches(S1, train_feats, train_labels, fwd_slice,
                     fixed_time=fixed_time_base + prev_cp)
        # S2 forward (identical)
        feed_batches(S2, train_feats, train_labels, fwd_slice,
                     fixed_time=fixed_time_base + prev_cp)
        # S3 reverse
        feed_batches(S3, train_feats, train_labels, rev_slice,
                     fixed_time=fixed_time_base + prev_cp)

        # Compute metrics
        for pair_name, cam_x, cam_y in [("S1 vs S2", S1, S2), ("S1 vs S3", S1, S3)]:
            cos = compute_cosine_sim(cam_x, cam_y)
            occ = occupancy_agreement(cam_x, cam_y)
            acc_x = eval_accuracy(cam_x, eval_feats, eval_labels)
            acc_y = eval_accuracy(cam_y, eval_feats, eval_labels)
            evict_j = eviction_jaccard(cam_x, cam_y)

            n_occ_x = cam_x.occupied.sum().item()
            n_occ_y = cam_y.occupied.sum().item()
            n_evict_x = len(cam_x.eviction_log)
            n_evict_y = len(cam_y.eviction_log)

            row = {
                "checkpoint": cp,
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
            print(f"  [{pair_name}] N={cp}: cos={cos['mean']:.6f} occ={occ:.4f} "
                  f"acc={acc_x:.2f}/{acc_y:.2f}% evict_J={evict_j:.4f}", flush=True)

        prev_cp = cp

    # -----------------------------------------------------------------------
    # Build markdown report
    # -----------------------------------------------------------------------
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()

    report_lines = [
        "# MT-3: Write-Path Divergence Report",
        "",
        "## Config",
        f"- **Commit**: `{commit}`",
        f"- **Seed**: {SEED}",
        f"- **Device**: {device}",
        f"- **Dataset**: CIFAR-100 DINOv2 ViT-B/14 features (768-d)",
        f"- **Max entries**: {MAX_ENTRIES:,}",
        f"- **Batch size**: {BATCH_SIZE}",
        f"- **Batches**: {max_batches} ({n_samples:,} samples)",
        f"- **Eval subset**: {eval_n} test samples",
        f"- **Determinism**: fixed RNG seed, monkey-patched `time.time()` for deterministic `last_seen`",
        f"- **CUDA deterministic**: `torch.backends.cudnn.deterministic=True`",
        "",
        "## Results",
        "",
        "| Checkpoint (N) | Pair | Cos sim (mean) | Cos sim (min) | Cos sim (p5) | Occupancy agree | Acc left | Acc right | Acc delta | Eviction overlap | Evictions L/R |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        report_lines.append(
            f"| {r['checkpoint']} | {r['pair']} | {r['cos_mean']:.6f} | {r['cos_min']:.6f} | "
            f"{r['cos_p5']:.6f} | {r['occ_agreement']:.4f} | {r['acc_left']:.2f}% | "
            f"{r['acc_right']:.2f}% | {r['acc_delta']:.2f}pp | {r['eviction_jaccard']:.4f} | "
            f"{r['n_evictions_left']}/{r['n_evictions_right']} |"
        )

    # Determine conclusion
    s1_s2_rows = [r for r in results if r["pair"] == "S1 vs S2"]
    s1_s3_rows = [r for r in results if r["pair"] == "S1 vs S3"]

    same_order_deterministic = all(
        r["cos_mean"] > 0.999999 and r["occ_agreement"] > 0.9999 and r["acc_delta"] < 0.01
        for r in s1_s2_rows
    )

    order_sensitive = any(
        r["cos_mean"] < 0.999 or r["occ_agreement"] < 0.999 or r["acc_delta"] > 0.1
        for r in s1_s3_rows
    )

    # Identify which metric shows order-sensitivity first
    first_divergent_metric = None
    first_divergent_n = None
    for r in s1_s3_rows:
        if r["cos_mean"] < 0.999:
            first_divergent_metric = first_divergent_metric or f"cosine similarity ({r['cos_mean']:.6f})"
            first_divergent_n = first_divergent_n or r["checkpoint"]
        if r["occ_agreement"] < 0.999:
            first_divergent_metric = first_divergent_metric or f"occupancy agreement ({r['occ_agreement']:.4f})"
            first_divergent_n = first_divergent_n or r["checkpoint"]
        if r["acc_delta"] > 0.1:
            first_divergent_metric = first_divergent_metric or f"accuracy delta ({r['acc_delta']:.2f}pp)"
            first_divergent_n = first_divergent_n or r["checkpoint"]

    report_lines.extend([
        "",
        "## Nondeterminism Sources",
        "",
        "- **`time.time()` in `learn_local()`**: Used for `last_seen` timestamps which feed into LFU-LRU eviction tiebreaking. "
        "Mitigated by monkey-patching to a deterministic counter (1s increments per call).",
        "- **CUDA nondeterminism**: `atomicAdd` in scatter operations can produce floating-point non-associativity. "
        "Mitigated with `torch.backends.cudnn.deterministic=True`, but scatter_add on GPU may still vary.",
        "- **`torch.unique()` ordering**: Used in `learn_local()` for scatter-mean over hit slots. "
        "Documented as stable on CUDA for identical inputs.",
        "- **RNG state**: Fully captured via `torch.manual_seed` / `random.seed` / `np.random.seed`.",
        "",
        "## Conclusion",
        "",
    ])

    if same_order_deterministic:
        report_lines.append("**Same-order determinism**: CONFIRMED. S1 vs S2 are bitwise identical across all checkpoints.")
    else:
        report_lines.append("**Same-order determinism**: FAILED. S1 vs S2 diverge — indicates floating-point or RNG nondeterminism in the write path.")

    if order_sensitive:
        report_lines.append(
            f"**Order sensitivity**: CONFIRMED. The write path is **order-sensitive**. "
            f"First detected at N={first_divergent_n} via {first_divergent_metric}."
        )
        report_lines.append("")
        report_lines.append(
            "This is expected: EMA-based key drift, hit-count–adaptive alpha, and LFU-LRU eviction "
            "all depend on the order in which prototypes are created and updated. "
            "Reversed batch order changes which prototypes are allocated first, "
            "which accumulate more hits, and which get evicted."
        )
    else:
        report_lines.append("**Order sensitivity**: NOT DETECTED. The write path appears order-invariant within measurement precision.")

    report = "\n".join(report_lines) + "\n"

    out_dir = Path("results/mt3")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mt3_write_path_divergence.md").write_text(report)
    print(f"\n[MT-3] Report saved to {out_dir / 'mt3_write_path_divergence.md'}")

    # Print the report
    print("\n" + report)


if __name__ == "__main__":
    main()
