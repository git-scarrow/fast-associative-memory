#!/usr/bin/env python3
"""
benchmarks/verify_g36.py — G36: WriteProjector Verification on Real Data.

Trains a per-domain WriteProjector (binary gate MLP) on real trace-derived
pairs from the production prototype store, then evaluates condensation ratio
and accuracy against G31/G34 baselines.

The WriteProjector learns to replace the fixed vigilance threshold with a
learned write-gate: given (sample, nearest_prototype) features, it predicts
whether the sample should be written as a new prototype (1) or merged with
the existing one (0).

Six checks:
  A: CR Dogs ≤ 0.50
  B: Acc Dogs ≥ 87.98%
  C: CR Birds/Cars/Aircraft ≤ 0.60
  D: Acc Birds/Cars/Aircraft within 2pp of G31
  E: WriteProjector precision/recall on real holdout
  F: Side-by-side vs v=0.85 and v=0.70 on real data

Usage:
    PYTHONPATH=. python benchmarks/verify_g36.py --data-root data --device auto
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from fast_associative_memory import FastAssociativeMemory  # noqa: E402
from nstp import NSTPController  # noqa: E402
from benchmarks.g30_adapter_sweep import (  # noqa: E402
    load_domain_features as g30_load_domain_features,
)
from benchmarks.g31_retrain_adapters import _load_g30_domain_adapter  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

Domain4 = Literal["dogs", "birds", "cars", "aircraft"]
DOMAINS: list[Domain4] = ["dogs", "birds", "cars", "aircraft"]

G31_BASELINES: Dict[Domain4, float] = {
    "dogs": 89.98,
    "birds": 90.78,
    "cars": 89.80,
    "aircraft": 73.51,
}

G34_CR: Dict[Domain4, float] = {
    "dogs": 0.821,
    "birds": 0.947,
    "cars": 0.935,
    "aircraft": 0.958,
}

# Kill thresholds
CR_KILL_DOGS = 0.50
CR_KILL_OTHERS = 0.60
ACC_KILL_DOGS = 87.98
ACC_TOLERANCE_PP = 2.0

# WriteProjector architecture
WP_HIDDEN = 256
WP_LR = 1e-3
WP_EPOCHS = 15
WP_BATCH = 512
WP_HOLDOUT_FRAC = 0.2  # 20% holdout for precision/recall

# ---------------------------------------------------------------------------
# WriteProjector
# ---------------------------------------------------------------------------


class WriteProjector(nn.Module):
    """Binary gate MLP: (sample ⊕ prototype) → write probability.

    Input: concatenation of projected sample and nearest prototype features.
    Output: logit for P(should write new prototype).

    Label convention:
        0 = merge (same class as nearest proto → don't create new slot)
        1 = write (nearest proto is wrong class → must create new slot)
    """

    def __init__(self, key_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(key_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, sample: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (B, 1)."""
        x = torch.cat([sample, prototype], dim=-1)
        return self.net(x)

    def should_write(self, sample: torch.Tensor, prototype: torch.Tensor,
                     threshold: float = 0.5) -> torch.Tensor:
        """Returns bool tensor (B,) — True = write new, False = merge."""
        with torch.no_grad():
            logits = self.forward(sample, prototype).squeeze(-1)
            return torch.sigmoid(logits) >= threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_all_seeds(seed: int = 42) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


@torch.no_grad()
def build_prototype_store(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    adapter,
    device: torch.device,
    vigilance: float = 0.85,
) -> FastAssociativeMemory:
    """Write training set into FAM and return the populated memory."""
    num_classes = int(train_labels.max().item()) + 1
    nstp = NSTPController(sibling_threshold=0.85, depth_epsilon=0.10)
    mem = FastAssociativeMemory(
        input_dim=int(train_feats.shape[1]),
        value_dim=num_classes,
        core_entries=50000,
        core_vigilance=vigilance,
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=25,
        inference_temp=0.05,
        use_lfu=True,
        adapter=adapter,
        nstp=nstp,
    ).to(device)

    x_train = train_feats.to(device)
    y_train = train_labels.to(device)
    for i in range(0, len(x_train), 256):
        mem.learn_local(x_train[i:i + 256], y_train[i:i + 256])

    return mem


@torch.no_grad()
def generate_write_pairs(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    mem: FastAssociativeMemory,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate (sample_proj, nearest_proto, label) pairs from populated memory.

    For each training sample:
      - Project through adapter
      - Find nearest prototype
      - Label = 0 if same class (should merge), 1 if different class (should write)

    Returns: (samples_proj, nearest_protos, labels) all on CPU.
    """
    core = mem.core_cam
    occ_idx = core.occupied.nonzero(as_tuple=True)[0]
    proto_keys = core._keys_norm[occ_idx]  # (N_proto, D)
    proto_classes = core.values[occ_idx].float().argmax(dim=-1)  # (N_proto,)

    all_samples = []
    all_protos = []
    all_labels = []

    x_train = train_feats.to(device)
    y_train = train_labels.to(device)

    for i in range(0, len(x_train), 512):
        batch_x = x_train[i:i + 512]
        batch_y = y_train[i:i + 512]

        # Project through adapter
        proj = mem._project(batch_x)  # (B, D)
        proj_norm = F.normalize(proj.float(), dim=-1)

        # Find nearest prototype
        sims = proj_norm @ proto_keys.T  # (B, N_proto)
        best_sims, best_locs = sims.max(dim=1)
        nearest_classes = proto_classes[best_locs]

        # Label: 0=merge (same class), 1=write (different class)
        labels = (batch_y != nearest_classes).long()

        all_samples.append(proj_norm.cpu())
        all_protos.append(proto_keys[best_locs].cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_samples), torch.cat(all_protos), torch.cat(all_labels)


def train_write_projector(
    samples: torch.Tensor,
    protos: torch.Tensor,
    labels: torch.Tensor,
    key_dim: int,
    device: torch.device,
) -> Tuple[WriteProjector, dict]:
    """Train WriteProjector on real pairs, return model and holdout metrics.

    Splits data into train/holdout, trains MLP, evaluates precision/recall.
    """
    N = len(labels)
    perm = torch.randperm(N)
    holdout_n = max(1, int(N * WP_HOLDOUT_FRAC))
    train_idx = perm[holdout_n:]
    holdout_idx = perm[:holdout_n]

    s_train, p_train, y_train = samples[train_idx], protos[train_idx], labels[train_idx]
    s_hold, p_hold, y_hold = samples[holdout_idx], protos[holdout_idx], labels[holdout_idx]

    wp = WriteProjector(key_dim, WP_HIDDEN).to(device)
    optimizer = torch.optim.Adam(wp.parameters(), lr=WP_LR)
    criterion = nn.BCEWithLogitsLoss()

    # Class balance weighting
    n_write = (y_train == 1).sum().float()
    n_merge = (y_train == 0).sum().float()
    pos_weight = (n_merge / n_write.clamp(min=1.0)).clamp(max=10.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    s_train_d = s_train.to(device)
    p_train_d = p_train.to(device)
    y_train_d = y_train.float().to(device)

    for epoch in range(WP_EPOCHS):
        wp.train()
        perm_e = torch.randperm(len(s_train_d))
        epoch_loss = 0.0
        n_batches = 0
        for j in range(0, len(s_train_d), WP_BATCH):
            idx = perm_e[j:j + WP_BATCH]
            logits = wp(s_train_d[idx], p_train_d[idx]).squeeze(-1)
            loss = criterion(logits, y_train_d[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

    # Evaluate on holdout
    wp.eval()
    with torch.no_grad():
        h_logits = wp(s_hold.to(device), p_hold.to(device)).squeeze(-1)
        h_preds = (torch.sigmoid(h_logits) >= 0.5).long().cpu()
        h_labels = y_hold

        tp = ((h_preds == 1) & (h_labels == 1)).sum().item()
        fp = ((h_preds == 1) & (h_labels == 0)).sum().item()
        fn = ((h_preds == 0) & (h_labels == 1)).sum().item()
        tn = ((h_preds == 0) & (h_labels == 0)).sum().item()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_train": len(s_train),
        "n_holdout": len(s_hold),
        "class_balance_write_pct": 100.0 * (y_train == 1).sum().item() / len(y_train),
        "final_loss": epoch_loss / max(n_batches, 1),
    }
    return wp, metrics


@torch.no_grad()
def eval_with_write_projector(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    adapter,
    wp: WriteProjector,
    device: torch.device,
    wp_threshold: float = 0.5,
) -> Tuple[float, int, float]:
    """Write training set using WriteProjector gate, then evaluate accuracy.

    Instead of fixed vigilance, the WriteProjector decides per-sample whether
    to create a new prototype or merge with the nearest existing one.

    Returns: (accuracy%, num_prototypes, condensation_ratio).
    """
    num_classes = int(train_labels.max().item()) + 1
    nstp = NSTPController(sibling_threshold=0.85, depth_epsilon=0.10)

    # Build FAM with very low vigilance (will be overridden by WP gate)
    mem = FastAssociativeMemory(
        input_dim=int(train_feats.shape[1]),
        value_dim=num_classes,
        core_entries=50000,
        core_vigilance=0.0,  # Disabled — WP controls writes
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=25,
        inference_temp=0.05,
        use_lfu=True,
        adapter=adapter,
        nstp=nstp,
    ).to(device)

    core = mem.core_cam
    x_train = train_feats.to(device)
    y_train = train_labels.to(device)

    wp.eval()

    for i in range(0, len(x_train), 256):
        batch_x = x_train[i:i + 256]
        batch_y = y_train[i:i + 256]
        B = batch_x.size(0)

        # Project through adapter
        proj = mem._project(batch_x)
        proj_cast = core._cast(proj)
        targets = F.one_hot(batch_y, num_classes=num_classes).float()
        targets_cast = core._cast(targets)

        now = time.time()

        if not core.occupied.any():
            # First batch: write everything
            n_alloc = min(B, core.max_entries)
            new_slots = core._alloc_slots_batch(n_alloc)
            core.keys[new_slots] = proj_cast[:n_alloc]
            core.values[new_slots] = targets_cast[:n_alloc]
            core.occupied[new_slots] = True
            core.last_seen[new_slots] = now
            core.usage[new_slots] = 1
            core.hit_counts[new_slots] = 1
            core._update_key_norm(new_slots)
            continue

        # Find nearest prototype for each sample
        occ_idx = core.occupied.nonzero(as_tuple=True)[0]
        q_norm = F.normalize(proj_cast.float(), dim=-1)
        proto_keys = core._keys_norm[occ_idx]
        sims = q_norm @ proto_keys.T  # (B, N_occ)
        best_sims, best_locs = sims.max(dim=1)
        best_slots = occ_idx[best_locs]
        nearest_proto_keys = proto_keys[best_locs]  # (B, D)

        # WriteProjector gate: should we write new?
        wp_write = wp.should_write(q_norm, nearest_proto_keys, threshold=wp_threshold)

        # Class check: even if WP says merge, reject if class doesn't match
        stored_classes = core.values[best_slots].float().argmax(dim=-1)
        class_match = (batch_y == stored_classes)

        # Decision: merge only if WP says merge AND class matches
        merge_mask = (~wp_write) & class_match
        write_mask = ~merge_mask

        # EMA update for merges
        if merge_mask.any():
            hit_slots = best_slots[merge_mask]
            hit_targets = targets_cast[merge_mask]

            unique_slots, inverse = hit_slots.unique(return_inverse=True)
            slot_target_sum = torch.zeros(len(unique_slots), core.value_dim,
                                          device=device, dtype=targets_cast.dtype)
            slot_target_sum.scatter_add_(0, inverse.unsqueeze(1).expand_as(hit_targets),
                                         hit_targets)
            slot_counts = torch.zeros(len(unique_slots), device=device, dtype=targets_cast.dtype)
            slot_counts.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=targets_cast.dtype))
            slot_target_mean = slot_target_sum / slot_counts.unsqueeze(1)

            core.hit_counts[unique_slots] += slot_counts.int()
            adaptive_alpha = (core.hebb_lr /
                              (1.0 + core.ema_beta * core.hit_counts[unique_slots].float()))

            current_vals = core.values[unique_slots]
            core.values[unique_slots] = (current_vals +
                                         adaptive_alpha.unsqueeze(1) * (slot_target_mean - current_vals)
                                         ).to(core._mem_dtype)

            # Key centroid drift
            if not core.immutable_keys:
                hit_queries = proj_cast[merge_mask]
                slot_query_sum = torch.zeros(len(unique_slots), core.key_dim,
                                             device=device, dtype=proj_cast.dtype)
                slot_query_sum.scatter_add_(0, inverse.unsqueeze(1).expand_as(hit_queries),
                                            hit_queries)
                slot_query_mean = slot_query_sum / slot_counts.unsqueeze(1)
                adaptive_key_alpha = (core.key_lr /
                                      (1.0 + core.ema_beta * core.hit_counts[unique_slots].float()))
                current_keys = core.keys[unique_slots]
                core.keys[unique_slots] = (current_keys +
                                           adaptive_key_alpha.unsqueeze(1) * (slot_query_mean - current_keys)
                                           ).to(core._mem_dtype)
                core._update_key_norm(unique_slots)

            core.last_seen[hit_slots] = now
            core.usage[unique_slots] += 1

        # Allocate new slots for writes
        if write_mask.any():
            miss_queries = proj_cast[write_mask]
            miss_targets = targets_cast[write_mask]
            n_miss = miss_queries.size(0)

            new_slots = core._alloc_slots_batch(n_miss)
            n_alloc = len(new_slots)
            core.keys[new_slots] = miss_queries[:n_alloc]
            core.values[new_slots] = miss_targets[:n_alloc]
            core.occupied[new_slots] = True
            core.last_seen[new_slots] = now
            core.usage[new_slots] = 1
            core.hit_counts[new_slots] = 1
            core._update_key_norm(new_slots)

    n_protos = int(core.occupied.sum().item())
    cond = n_protos / max(len(train_feats), 1)

    # Evaluate on test set
    correct = 0
    x_test = test_feats.to(device)
    y_test = test_labels.to(device)
    for i in range(0, len(x_test), 512):
        logits = mem(x_test[i:i + 512])
        correct += (logits.argmax(dim=1) == y_test[i:i + 512]).sum().item()

    acc = 100.0 * correct / max(len(test_labels), 1)

    del mem
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return acc, n_protos, cond


@torch.no_grad()
def eval_static_vigilance(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    adapter,
    device: torch.device,
    vigilance: float,
) -> Tuple[float, int, float]:
    """Standard FAM eval with static vigilance. Returns (acc, n_protos, CR)."""
    num_classes = int(train_labels.max().item()) + 1
    nstp = NSTPController(sibling_threshold=0.85, depth_epsilon=0.10)
    mem = FastAssociativeMemory(
        input_dim=int(train_feats.shape[1]),
        value_dim=num_classes,
        core_entries=50000,
        core_vigilance=vigilance,
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=25,
        inference_temp=0.05,
        use_lfu=True,
        adapter=adapter,
        nstp=nstp,
    ).to(device)

    x_train = train_feats.to(device)
    y_train = train_labels.to(device)
    for i in range(0, len(x_train), 256):
        mem.learn_local(x_train[i:i + 256], y_train[i:i + 256])

    n_protos = int(mem.core_cam.occupied.sum().item())
    cond = n_protos / max(len(train_feats), 1)

    correct = 0
    x_test = test_feats.to(device)
    y_test = test_labels.to(device)
    for i in range(0, len(x_test), 512):
        logits = mem(x_test[i:i + 512])
        correct += (logits.argmax(dim=1) == y_test[i:i + 512]).sum().item()

    acc = 100.0 * correct / max(len(test_labels), 1)

    del mem
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return acc, n_protos, cond


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------


def run_domain(
    domain: Domain4,
    data_root: Path,
    device: torch.device,
) -> dict:
    """Full G36 pipeline for one domain. Returns results dict."""
    print(f"\n{'='*60}")
    print(f"  DOMAIN: {domain.upper()}")
    print(f"{'='*60}")

    tr_f, tr_l, te_f, te_l, n_classes = g30_load_domain_features(domain, data_root)
    adapter = _load_g30_domain_adapter(domain, input_dim=tr_f.shape[1], device=device)

    print(f"  Train: {len(tr_f)} samples, {n_classes} classes")
    print(f"  Test:  {len(te_f)} samples")

    # Step 1: Build prototype store with v=0.85
    print("\n  [1/5] Building prototype store (v=0.85)...")
    set_all_seeds(42)
    mem = build_prototype_store(tr_f, tr_l, adapter, device, vigilance=0.85)
    n_protos_base = int(mem.core_cam.occupied.sum().item())
    print(f"        Prototypes: {n_protos_base} (CR={n_protos_base/len(tr_f):.4f})")

    # Step 2: Generate training pairs from prototype store
    print("  [2/5] Generating write-gate training pairs...")
    samples, protos, pair_labels = generate_write_pairs(tr_f, tr_l, mem, device)
    n_write = (pair_labels == 1).sum().item()
    n_merge = (pair_labels == 0).sum().item()
    print(f"        Pairs: {len(pair_labels)} total, {n_write} write ({100*n_write/len(pair_labels):.1f}%), "
          f"{n_merge} merge ({100*n_merge/len(pair_labels):.1f}%)")

    del mem
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 3: Train WriteProjector
    print("  [3/5] Training WriteProjector...")
    set_all_seeds(42)
    key_dim = samples.shape[1]
    wp, wp_metrics = train_write_projector(samples, protos, pair_labels, key_dim, device)
    print(f"        Holdout precision={wp_metrics['precision']:.4f}  "
          f"recall={wp_metrics['recall']:.4f}  F1={wp_metrics['f1']:.4f}  "
          f"acc={wp_metrics['accuracy']:.4f}")
    print(f"        TP={wp_metrics['tp']} FP={wp_metrics['fp']} "
          f"FN={wp_metrics['fn']} TN={wp_metrics['tn']}")

    # Step 4: Re-write with WriteProjector gate
    print("  [4/5] Re-writing training set with WriteProjector gate...")
    set_all_seeds(42)
    acc_wp, n_protos_wp, cr_wp = eval_with_write_projector(
        tr_f, tr_l, te_f, te_l, adapter, wp, device
    )
    print(f"        WP: acc={acc_wp:.2f}%  protos={n_protos_wp}  CR={cr_wp:.4f}")

    # Step 5: Static vigilance baselines for comparison
    print("  [5/5] Static vigilance baselines...")
    set_all_seeds(42)
    acc_v85, n_v85, cr_v85 = eval_static_vigilance(tr_f, tr_l, te_f, te_l, adapter, device, 0.85)
    print(f"        v=0.85: acc={acc_v85:.2f}%  protos={n_v85}  CR={cr_v85:.4f}")

    set_all_seeds(42)
    acc_v70, n_v70, cr_v70 = eval_static_vigilance(tr_f, tr_l, te_f, te_l, adapter, device, 0.70)
    print(f"        v=0.70: acc={acc_v70:.2f}%  protos={n_v70}  CR={cr_v70:.4f}")

    del adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "domain": domain,
        "n_train": len(tr_f),
        "n_test": len(te_f),
        "n_classes": n_classes,
        "wp_acc": acc_wp,
        "wp_protos": n_protos_wp,
        "wp_cr": cr_wp,
        "v85_acc": acc_v85,
        "v85_protos": n_v85,
        "v85_cr": cr_v85,
        "v70_acc": acc_v70,
        "v70_protos": n_v70,
        "v70_cr": cr_v70,
        "g31_acc": G31_BASELINES[domain],
        "g34_cr": G34_CR[domain],
        "wp_metrics": wp_metrics,
    }


def print_checks(results: Dict[Domain4, dict]) -> bool:
    """Print all 6 checks and return True if all pass."""
    print("\n" + "=" * 70)
    print("  G36 VERIFICATION CHECKS (Real Data)")
    print("=" * 70)

    all_pass = True

    # Check A: CR Dogs ≤ 0.50
    dogs = results["dogs"]
    a_pass = dogs["wp_cr"] <= CR_KILL_DOGS
    a_tag = "PASS" if a_pass else "KILL"
    print(f"\n  CHECK A — CR Dogs ≤ {CR_KILL_DOGS}")
    print(f"    WriteProjector CR = {dogs['wp_cr']:.4f}  [{a_tag}]")
    print(f"    (v=0.85 CR = {dogs['v85_cr']:.4f}, v=0.70 CR = {dogs['v70_cr']:.4f}, "
          f"G34 CR = {dogs['g34_cr']:.4f})")
    all_pass &= a_pass

    # Check B: Acc Dogs ≥ 87.98%
    b_pass = dogs["wp_acc"] >= ACC_KILL_DOGS
    b_tag = "PASS" if b_pass else "KILL"
    print(f"\n  CHECK B — Acc Dogs ≥ {ACC_KILL_DOGS}%")
    print(f"    WriteProjector Acc = {dogs['wp_acc']:.2f}%  [{b_tag}]")
    print(f"    (G31 baseline = {dogs['g31_acc']:.2f}%)")
    all_pass &= b_pass

    # Check C: CR Birds/Cars/Aircraft ≤ 0.60
    print(f"\n  CHECK C — CR Birds/Cars/Aircraft ≤ {CR_KILL_OTHERS}")
    c_pass = True
    for d in ["birds", "cars", "aircraft"]:
        r = results[d]
        ok = r["wp_cr"] <= CR_KILL_OTHERS
        c_pass &= ok
        tag = "PASS" if ok else "KILL"
        print(f"    {d:10s}: WP CR = {r['wp_cr']:.4f}  [{tag}]  "
              f"(v=0.85={r['v85_cr']:.4f}, v=0.70={r['v70_cr']:.4f}, G34={r['g34_cr']:.4f})")
    all_pass &= c_pass

    # Check D: Acc within 2pp of G31
    print(f"\n  CHECK D — Acc Birds/Cars/Aircraft within {ACC_TOLERANCE_PP}pp of G31")
    d_pass = True
    for d in ["birds", "cars", "aircraft"]:
        r = results[d]
        delta = r["wp_acc"] - r["g31_acc"]
        ok = delta >= -ACC_TOLERANCE_PP
        d_pass &= ok
        tag = "PASS" if ok else "KILL"
        print(f"    {d:10s}: WP Acc = {r['wp_acc']:.2f}% vs G31 {r['g31_acc']:.2f}% "
              f"(Δ={delta:+.2f}pp)  [{tag}]")
    all_pass &= d_pass

    # Check E: WriteProjector precision/recall
    print(f"\n  CHECK E — WriteProjector Precision/Recall on Real Holdout")
    for d in DOMAINS:
        r = results[d]
        m = r["wp_metrics"]
        print(f"    {d:10s}: P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}  "
              f"(TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']})")
        print(f"              Train={m['n_train']}  Holdout={m['n_holdout']}  "
              f"Write%={m['class_balance_write_pct']:.1f}%")

    # Check F: Side-by-side vs static vigilance
    print(f"\n  CHECK F — Side-by-Side: WriteProjector vs v=0.85 vs v=0.70")
    print(f"    {'Domain':10s}  {'WP Acc':>8s}  {'WP CR':>8s}  "
          f"{'v85 Acc':>8s}  {'v85 CR':>8s}  {'v70 Acc':>8s}  {'v70 CR':>8s}  "
          f"{'G31 Acc':>8s}  {'G34 CR':>8s}")
    print(f"    {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for d in DOMAINS:
        r = results[d]
        print(f"    {d:10s}  {r['wp_acc']:8.2f}  {r['wp_cr']:8.4f}  "
              f"{r['v85_acc']:8.2f}  {r['v85_cr']:8.4f}  "
              f"{r['v70_acc']:8.2f}  {r['v70_cr']:8.4f}  "
              f"{r['g31_acc']:8.2f}  {r['g34_cr']:8.4f}")

    # Final verdict
    verdict = "PASS" if all_pass else "KILL"
    print(f"\n  {'='*70}")
    print(f"  G36 VERDICT: {verdict}")
    print(f"  {'='*70}")
    return all_pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="G36 — WriteProjector Verification on Real Data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = device_from_arg(args.device)

    print("G36 — WriteProjector Verification (Real Data)")
    print(f"Device:    {device}")
    print(f"Data root: {args.data_root}")

    results: Dict[Domain4, dict] = {}

    for domain in DOMAINS:
        results[domain] = run_domain(domain, args.data_root, device)

    passed = print_checks(results)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
