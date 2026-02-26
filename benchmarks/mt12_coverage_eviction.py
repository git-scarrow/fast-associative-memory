#!/usr/bin/env python3
"""
benchmarks/mt12_coverage_eviction.py — MT-12: Coverage-Aware Eviction (Minimal Intervention)

Swaps LFU eviction for coverage-aware eviction in the existing pipeline.
When a slot is needed, evicts the prototype whose nearest same-class neighbor
is closest (most replaceable), rather than the least-frequently-used.

Everything else is identical: vigilance gate, EMA centroid updates, adapter,
retrieval pipeline. This isolates the contribution of eviction policy to the
FAM-vs-coreset accuracy gap.

Baselines (from MT-7 / MT-9):
  LFU (MT-7):     1K=38.57%  5K=61.20%  10K=88.75%
  Coreset (MT-9): 1K=89.10%  5K=90.15%  10K=90.13%

Usage:
    PYTHONPATH=. python benchmarks/mt12_coverage_eviction.py [--device auto] [--data-root data]
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time as _time_mod
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from associative_core import ContinuousCAM  # noqa: E402
from fast_associative_memory import FastAssociativeMemory  # noqa: E402
from nstp import NSTPController  # noqa: E402
from benchmarks.g30_adapter_sweep import (  # noqa: E402
    load_domain_features as g30_load_domain_features,
)
from benchmarks.g31_retrain_adapters import _load_g30_domain_adapter  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TAG = "MT-12"
SEED = 42
BATCH_SIZE = 256
CAPACITIES = [1000, 5000, 10000]

# Baselines from MT-7 and MT-9
LFU_BASELINES = {1000: 38.57, 5000: 61.20, 10000: 88.75}
CORESET_BASELINES = {1000: 89.10, 5000: 90.15, 10000: 90.13}


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
# Coverage-Aware Eviction CAM
# ---------------------------------------------------------------------------

class CoverageEvictionCAM(ContinuousCAM):
    """ContinuousCAM with coverage-aware eviction replacing LFU.

    When eviction is needed, scores each prototype by distance to its nearest
    same-class neighbor.  Evicts the one with the smallest distance (most
    replaceable — another prototype of the same class is nearby).

    Prototypes that are the sole representative of their class get score=inf
    and are never evicted.
    """

    def _alloc_slots_batch(self, n: int):
        n = min(n, self.max_entries)
        free = (~self.occupied).nonzero(as_tuple=True)[0]
        if len(free) >= n:
            return free[:n]

        needed = n - len(free)
        occupied_idx = self.occupied.nonzero(as_tuple=True)[0]
        if len(free) > 0:
            mask = ~torch.isin(occupied_idx, free)
            occupied_idx = occupied_idx[mask]
        needed = min(needed, len(occupied_idx))

        if needed > 0:
            scores = self._coverage_eviction_score(occupied_idx)
            _, topk_idx = scores.topk(needed, largest=False)
            victims = occupied_idx[topk_idx]
        else:
            victims = occupied_idx[:0]

        return torch.cat([free, victims]) if len(free) > 0 else victims

    def _coverage_eviction_score(self, occupied_idx: torch.Tensor) -> torch.Tensor:
        """Score each prototype by distance to nearest same-class neighbor.

        Lower score → more replaceable → evict first.
        Score = 1 - cos(p_i, nearest same-class neighbor).
        Single class representatives get score = inf (protected).
        """
        keys_norm = self._keys_norm[occupied_idx]
        class_labels = self.values[occupied_idx].float().argmax(dim=-1)

        scores = torch.full(
            (len(occupied_idx),), float("inf"),
            device=keys_norm.device, dtype=torch.float32,
        )

        for c in class_labels.unique():
            c_mask = class_labels == c
            n_c = c_mask.sum().item()
            if n_c < 2:
                continue  # sole representative → protected

            idx_c = c_mask.nonzero(as_tuple=True)[0]
            keys_c = keys_norm[idx_c]
            sim_c = keys_c.float() @ keys_c.float().T
            sim_c.fill_diagonal_(-float("inf"))
            max_sim, _ = sim_c.max(dim=1)
            scores[idx_c] = 1.0 - max_sim

        return scores


# ---------------------------------------------------------------------------
# FAM construction
# ---------------------------------------------------------------------------

def make_fam(
    adapter, n_classes: int, capacity: int, device: torch.device,
    coverage_eviction: bool = False,
) -> FastAssociativeMemory:
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

    if coverage_eviction:
        # Swap core_cam with CoverageEvictionCAM (same init args)
        cam = fam.core_cam
        cov_cam = CoverageEvictionCAM(
            key_dim=cam.key_dim,
            value_dim=cam.value_dim,
            max_entries=cam.max_entries,
            vigilance=cam.vigilance,
            hebb_lr=cam.hebb_lr,
            aging_time=cam.aging_time,
            flood_scale=cam.flood_scale,
            immutable_keys=cam.immutable_keys,
            use_lfu=cam.use_lfu,
            use_bfloat16=cam.use_bfloat16,
            key_lr=cam.key_lr,
            inference_k=cam.inference_k,
            inference_temp=cam.inference_temp,
            ema_beta=cam.ema_beta,
            dynamic_vigilance=cam.dynamic_vigilance,
        ).to(device)
        cov_cam.csls_weight = cam.csls_weight
        cov_cam.inference_sim_floor = cam.inference_sim_floor
        cov_cam.var_ema_alpha = cam.var_ema_alpha
        fam.core_cam = cov_cam

    return fam


# ---------------------------------------------------------------------------
# Feed + Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def feed_all(fam, train_feats, train_labels):
    """Feed all training data with deterministic clock."""
    _orig = _time_mod.time
    clock = [1_000_000.0]

    def _dt():
        clock[0] += 1.0
        return clock[0]

    _time_mod.time = _dt
    try:
        for i in range(0, len(train_feats), BATCH_SIZE):
            fam.learn_local(
                train_feats[i:i + BATCH_SIZE],
                train_labels[i:i + BATCH_SIZE],
            )
    finally:
        _time_mod.time = _orig


@torch.no_grad()
def eval_accuracy(fam, test_feats, test_labels) -> float:
    correct = total = 0
    for i in range(0, len(test_feats), 256):
        out = fam(test_feats[i:i + 256])
        correct += (out.argmax(1) == test_labels[i:i + 256]).sum().item()
        total += min(256, len(test_feats) - i)
    return correct / total * 100 if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=f"{TAG}: Coverage-Aware Eviction")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-root", type=str, default="data")
    args = parser.parse_args()

    device = (
        torch.device("cuda")
        if args.device == "auto" and torch.cuda.is_available()
        else torch.device(args.device if args.device != "auto" else "cpu")
    )
    print(f"[{TAG}] Device: {device}", flush=True)

    # Verify eval contamination fix is present
    import associative_core, inspect
    src = inspect.getsource(associative_core.ContinuousCAM.forward)
    assert "last_seen" not in src or "NOT updated" in src, \
        "ABORT: eval contamination fix not present in associative_core.py"
    print(f"[{TAG}] Eval contamination fix: verified", flush=True)

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

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()

    results = []

    for cap in CAPACITIES:
        print(f"\n{'='*60}")
        print(f"  Capacity: {cap}")
        print(f"{'='*60}")

        # --- LFU baseline (re-run to confirm MT-7 numbers post-fix) ---
        set_all_seeds(SEED)
        fam_lfu = make_fam(adapter, n_classes, cap, device, coverage_eviction=False)
        t0 = _time_mod.time()
        feed_all(fam_lfu, train_feats, train_labels)
        t_lfu = _time_mod.time() - t0
        acc_lfu = eval_accuracy(fam_lfu, test_feats, test_labels)
        n_occ_lfu = fam_lfu.core_cam.occupied.sum().item()
        print(f"  LFU:      acc={acc_lfu:.2f}%  protos={n_occ_lfu}  ({t_lfu:.1f}s)", flush=True)

        # --- Coverage-aware eviction ---
        set_all_seeds(SEED)
        fam_cov = make_fam(adapter, n_classes, cap, device, coverage_eviction=True)
        t0 = _time_mod.time()
        feed_all(fam_cov, train_feats, train_labels)
        t_cov = _time_mod.time() - t0
        acc_cov = eval_accuracy(fam_cov, test_feats, test_labels)
        n_occ_cov = fam_cov.core_cam.occupied.sum().item()
        print(f"  Coverage: acc={acc_cov:.2f}%  protos={n_occ_cov}  ({t_cov:.1f}s)", flush=True)

        # --- Reference baselines ---
        acc_lfu_mt7 = LFU_BASELINES[cap]
        acc_coreset = CORESET_BASELINES[cap]

        gap_total = acc_coreset - acc_lfu
        gap_closure = (acc_cov - acc_lfu) / gap_total * 100 if gap_total > 0 else 0.0

        print(f"  LFU (MT-7 ref): {acc_lfu_mt7:.2f}%")
        print(f"  Coreset (MT-9): {acc_coreset:.2f}%")
        print(f"  Gap closure:    {gap_closure:.1f}%  "
              f"(+{acc_cov - acc_lfu:.2f}pp / {gap_total:.2f}pp total gap)")

        # --- Class distribution analysis ---
        cov_vals = fam_cov.core_cam.values[fam_cov.core_cam.occupied].float()
        cov_class_dist = cov_vals.argmax(dim=1).bincount(minlength=n_classes)
        lfu_vals = fam_lfu.core_cam.values[fam_lfu.core_cam.occupied].float()
        lfu_class_dist = lfu_vals.argmax(dim=1).bincount(minlength=n_classes)

        cov_classes_repr = (cov_class_dist > 0).sum().item()
        lfu_classes_repr = (lfu_class_dist > 0).sum().item()
        cov_min_per_class = cov_class_dist[cov_class_dist > 0].min().item() if cov_classes_repr > 0 else 0
        lfu_min_per_class = lfu_class_dist[lfu_class_dist > 0].min().item() if lfu_classes_repr > 0 else 0

        print(f"  Classes represented: LFU={lfu_classes_repr}/{n_classes} "
              f"Coverage={cov_classes_repr}/{n_classes}")
        print(f"  Min protos/class:    LFU={lfu_min_per_class} "
              f"Coverage={cov_min_per_class}")

        results.append({
            "capacity": cap,
            "acc_lfu": acc_lfu,
            "acc_lfu_mt7": acc_lfu_mt7,
            "acc_coverage": acc_cov,
            "acc_coreset": acc_coreset,
            "gap_closure_pct": gap_closure,
            "n_protos_lfu": n_occ_lfu,
            "n_protos_cov": n_occ_cov,
            "classes_repr_lfu": lfu_classes_repr,
            "classes_repr_cov": cov_classes_repr,
            "min_per_class_lfu": lfu_min_per_class,
            "min_per_class_cov": cov_min_per_class,
            "time_lfu_s": t_lfu,
            "time_cov_s": t_cov,
        })

        del fam_lfu, fam_cov
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------
    out_dir = Path("results/mt12")
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(out_dir / "mt12_metrics.json", "w") as f:
        json.dump({"commit": commit, "results": results}, f, indent=2)

    # Decision logic
    closures = [r["gap_closure_pct"] for r in results]
    low_cap_closures = [r["gap_closure_pct"] for r in results if r["capacity"] <= 5000]

    if all(c >= 50 for c in low_cap_closures):
        decision = "EVICTION IS THE BOTTLENECK"
        detail = (
            "Coverage-aware eviction closes >=50% of the gap at low capacity. "
            "Ship the eviction fix. MT-10 full rewrite is optional."
        )
    elif all(c < 20 for c in closures):
        decision = "ALLOCATION IS THE BOTTLENECK"
        detail = (
            "Coverage-aware eviction closes <20% of the gap. "
            "The allocation pathway (vigilance + EMA) is the primary problem. "
            "MT-10 full streaming k-center rewrite is necessary."
        )
    else:
        decision = "MIXED — BOTH CONTRIBUTE"
        detail = (
            "Coverage-aware eviction partially closes the gap. "
            "Eviction fix is worth shipping independently, but MT-10 is still needed."
        )

    # Writeup
    writeup = f"""# MT-12: Coverage-Aware Eviction (Minimal Intervention)

## Objective

Isolate the contribution of eviction policy to the FAM-vs-coreset accuracy gap.
Swap LFU for coverage-aware eviction; keep vigilance, EMA, everything else identical.

## Config
- **Commit**: `{commit}`
- **Eval contamination fix**: verified on main
- **Device**: {device}
- **Dataset**: Stanford Dogs 1024-d, production adapter
- **Capacities**: {', '.join(str(c) for c in CAPACITIES)}
- **Eviction rule**: Evict prototype with nearest same-class neighbor (most replaceable).
  Sole class representatives are protected (score = inf).

## Results

| Capacity | LFU (re-run) | Coverage Evict | Coreset (MT-9) | Gap Closure |
|----------|-------------|----------------|----------------|-------------|
"""
    for r in results:
        writeup += (
            f"| {r['capacity']:,} | {r['acc_lfu']:.2f}% | {r['acc_coverage']:.2f}% | "
            f"{r['acc_coreset']:.2f}% | {r['gap_closure_pct']:.1f}% |\n"
        )

    writeup += f"""
### LFU re-run vs MT-7 reference

| Capacity | MT-7 (pre-fix) | Re-run (post-fix) | Delta |
|----------|---------------|-------------------|-------|
"""
    for r in results:
        delta = r["acc_lfu"] - r["acc_lfu_mt7"]
        writeup += (
            f"| {r['capacity']:,} | {r['acc_lfu_mt7']:.2f}% | {r['acc_lfu']:.2f}% | "
            f"{delta:+.2f}pp |\n"
        )

    writeup += f"""
### Class distribution

| Capacity | Metric | LFU | Coverage |
|----------|--------|-----|----------|
"""
    for r in results:
        writeup += (
            f"| {r['capacity']:,} | Classes represented | "
            f"{r['classes_repr_lfu']}/{n_classes} | {r['classes_repr_cov']}/{n_classes} |\n"
            f"| | Min protos/class | {r['min_per_class_lfu']} | {r['min_per_class_cov']} |\n"
        )

    writeup += f"""
### Timing

| Capacity | LFU | Coverage |
|----------|-----|----------|
"""
    for r in results:
        writeup += f"| {r['capacity']:,} | {r['time_lfu_s']:.1f}s | {r['time_cov_s']:.1f}s |\n"

    writeup += f"""
## Decision

**{decision}**

{detail}

## Artifacts
- `mt12_metrics.json` — all metrics
"""

    report_path = out_dir / "mt12_coverage_eviction.md"
    report_path.write_text(writeup)
    print(f"\n[{TAG}] Report: {report_path}")
    print(writeup)


if __name__ == "__main__":
    main()
