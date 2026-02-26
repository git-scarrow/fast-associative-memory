#!/usr/bin/env python3
"""
benchmarks/mt11_curriculum_stress.py — MT-11: Curriculum Stress Test

Tests whether coverage-aware eviction handles pathological stream orderings.
MT-4/MT-6 showed class-sorted arrival is catastrophic under LFU eviction
(+55pp reversal delta). MT-12 showed coverage eviction closes 67–99% of
the coreset gap under standard (class-sorted) arrival.

Runs a 2×3 matrix:
  - 2 write paths:  LFU eviction (old) vs Coverage eviction (new/production)
  - 3 orderings:    class-sorted, IID shuffled, block-shuffled (realistic)
  - Fixed at 5K capacity (where MT-9 showed the largest gap)

Plus offline coreset as ceiling reference.

Usage:
    PYTHONPATH=. python benchmarks/mt11_curriculum_stress.py [--device auto] [--data-root data]
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time as _time_mod
from pathlib import Path

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

TAG = "MT-11"
SEED = 42
BATCH_SIZE = 256
CAPACITY = 5000
BLOCK_WINDOW = 300  # block-shuffle window size


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
# LFU-only CAM (restores old eviction for comparison)
# ---------------------------------------------------------------------------

class LFUEvictionCAM(ContinuousCAM):
    """ContinuousCAM with the old LFU-LRU hybrid eviction (pre-MT-12)."""

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
            occ_usage = self.usage[occupied_idx]
            occ_time = self.last_seen[occupied_idx].float()
            t_min = occ_time.min()
            t_range = occ_time.max() - t_min
            time_tiebreak = (occ_time - t_min) / (t_range + 1e-8)
            eviction_score = occ_usage + time_tiebreak
            _, topk_idx = eviction_score.topk(needed, largest=False)
            victims = occupied_idx[topk_idx]
        else:
            victims = occupied_idx[:0]
        return torch.cat([free, victims]) if len(free) > 0 else victims


# ---------------------------------------------------------------------------
# Stream orderings
# ---------------------------------------------------------------------------

def make_class_sorted_order(labels: torch.Tensor) -> torch.Tensor:
    """All class-0 first, then class-1, etc."""
    return labels.cpu().argsort(stable=True).to(labels.device)


def make_iid_shuffled_order(n: int, device: torch.device, seed: int = SEED) -> torch.Tensor:
    rng = torch.Generator()
    rng.manual_seed(seed)
    return torch.randperm(n, generator=rng).to(device)


def make_block_shuffled_order(
    labels: torch.Tensor, window_size: int = BLOCK_WINDOW, seed: int = SEED,
) -> torch.Tensor:
    """Class-clustered within windows, windows randomly interleaved.

    Models realistic Shutter-Deck ingestion: a burst of similar photos,
    then a burst of different ones, with bursts in arbitrary order.
    """
    labels_cpu = labels.cpu()
    n = len(labels_cpu)
    n_windows = (n + window_size - 1) // window_size
    window_indices = []
    for w in range(n_windows):
        start = w * window_size
        end = min(start + window_size, n)
        idx = torch.arange(start, end)
        sorted_order = labels_cpu[idx].argsort(stable=True)
        window_indices.append(idx[sorted_order])

    rng = random.Random(seed)
    rng.shuffle(window_indices)
    return torch.cat(window_indices).to(labels.device)


# ---------------------------------------------------------------------------
# FAM construction
# ---------------------------------------------------------------------------

def make_fam(
    adapter, n_classes: int, capacity: int, device: torch.device,
    use_lfu_eviction: bool = False,
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

    if use_lfu_eviction:
        # Swap core_cam with old LFU eviction
        cam = fam.core_cam
        lfu_cam = LFUEvictionCAM(
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
        lfu_cam.csls_weight = cam.csls_weight
        lfu_cam.inference_sim_floor = cam.inference_sim_floor
        lfu_cam.var_ema_alpha = cam.var_ema_alpha
        fam.core_cam = lfu_cam

    return fam


# ---------------------------------------------------------------------------
# Feed + Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def feed_ordered(fam, train_feats, train_labels, order: torch.Tensor):
    """Feed training data in the given index order with deterministic clock."""
    _orig = _time_mod.time
    clock = [1_000_000.0]

    def _dt():
        clock[0] += 1.0
        return clock[0]

    _time_mod.time = _dt
    try:
        for i in range(0, len(order), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            fam.learn_local(train_feats[idx], train_labels[idx])
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


@torch.no_grad()
def per_class_accuracy(fam, test_feats, test_labels, n_classes: int):
    """Returns per-class accuracy as a tensor of shape (n_classes,)."""
    correct = torch.zeros(n_classes, device=test_feats.device)
    total = torch.zeros(n_classes, device=test_feats.device)
    for i in range(0, len(test_feats), 256):
        batch_f = test_feats[i:i + 256]
        batch_l = test_labels[i:i + 256]
        out = fam(batch_f)
        preds = out.argmax(1)
        for c in range(n_classes):
            mask = batch_l == c
            if mask.any():
                total[c] += mask.sum()
                correct[c] += (preds[mask] == c).sum()
    return (correct / total.clamp(min=1) * 100).cpu()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=f"{TAG}: Curriculum Stress Test")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-root", type=str, default="data")
    args = parser.parse_args()

    device = (
        torch.device("cuda")
        if args.device == "auto" and torch.cuda.is_available()
        else torch.device(args.device if args.device != "auto" else "cpu")
    )
    print(f"[{TAG}] Device: {device}", flush=True)

    # Verify coverage eviction is present in production code
    import associative_core, inspect
    src = inspect.getsource(associative_core.ContinuousCAM._alloc_slots_batch)
    assert "coverage" in src.lower(), \
        "ABORT: coverage eviction not present in associative_core.py"
    print(f"[{TAG}] Coverage eviction: verified in production code", flush=True)

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

    # --- Generate orderings ---
    orderings = {
        "class-sorted": make_class_sorted_order(train_labels),
        "IID-shuffled": make_iid_shuffled_order(n_train, device, seed=SEED),
        "block-shuffled": make_block_shuffled_order(train_labels, BLOCK_WINDOW, seed=SEED),
    }

    # --- Write paths ---
    write_paths = {
        "Coverage (new)": False,    # use_lfu_eviction=False → production coverage
        "LFU (old)": True,          # use_lfu_eviction=True → old LFU
    }

    # --- Run matrix ---
    results = []
    per_class_results = {}

    for wp_name, use_lfu in write_paths.items():
        for ord_name, order in orderings.items():
            set_all_seeds(SEED)
            fam = make_fam(adapter, n_classes, CAPACITY, device, use_lfu_eviction=use_lfu)
            feed_ordered(fam, train_feats, train_labels, order)

            acc = eval_accuracy(fam, test_feats, test_labels)
            pc_acc = per_class_accuracy(fam, test_feats, test_labels, n_classes)

            n_occ = fam.core_cam.occupied.sum().item()
            vals = fam.core_cam.values[fam.core_cam.occupied].float()
            class_dist = vals.argmax(dim=1).bincount(minlength=n_classes)
            classes_repr = (class_dist > 0).sum().item()

            print(f"  [{wp_name:15s} × {ord_name:15s}]  acc={acc:.2f}%  "
                  f"protos={n_occ}  classes={classes_repr}/{n_classes}", flush=True)

            row = {
                "write_path": wp_name,
                "ordering": ord_name,
                "accuracy": acc,
                "n_protos": n_occ,
                "classes_represented": classes_repr,
            }
            results.append(row)

            # Store per-class accuracy for class-sorted condition
            if ord_name == "class-sorted":
                per_class_results[wp_name] = pc_acc

            del fam
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Coreset reference at 5K ---
    print(f"\n  [Coreset reference at {CAPACITY}]", flush=True)
    from benchmarks.mt9_coreset import k_center_greedy, make_fam as make_fam_mt9, populate_coreset_fam
    set_all_seeds(SEED)
    fam_core = make_fam_mt9(adapter, n_classes, CAPACITY, device)
    projected = adapter(train_feats).float()
    coreset_idx = k_center_greedy(projected, min(CAPACITY, n_train))
    populate_coreset_fam(fam_core, train_feats, train_labels, coreset_idx)
    acc_coreset = eval_accuracy(fam_core, test_feats, test_labels)
    print(f"  Coreset: {acc_coreset:.2f}%", flush=True)
    del fam_core
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------
    out_dir = Path("results/mt11")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the 2x3 matrix
    matrix = {}
    for r in results:
        matrix[(r["write_path"], r["ordering"])] = r

    # Per-write-path variance
    for wp_name in write_paths:
        accs = [matrix[(wp_name, o)]["accuracy"] for o in orderings]
        spread = max(accs) - min(accs)
        print(f"  {wp_name}: spread = {spread:.2f}pp  "
              f"(min={min(accs):.2f}%, max={max(accs):.2f}%)")

    # Per-class breakdown for class-sorted
    if per_class_results:
        n_early = min(10, n_classes)
        n_late = min(10, n_classes)
        for wp_name, pc in per_class_results.items():
            early = pc[:n_early].mean().item()
            mid_start = n_classes // 2 - 5
            mid = pc[mid_start:mid_start + 10].mean().item()
            late = pc[-n_late:].mean().item()
            print(f"  {wp_name} class-sorted per-class: "
                  f"early(0-9)={early:.1f}%  mid({mid_start}-{mid_start+9})={mid:.1f}%  "
                  f"late({n_classes-n_late}-{n_classes-1})={late:.1f}%")

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    with open(out_dir / "mt11_metrics.json", "w") as f:
        json.dump({
            "commit": commit,
            "capacity": CAPACITY,
            "coreset_acc": acc_coreset,
            "results": results,
        }, f, indent=2)

    # Determine curriculum robustness
    cov_accs = [matrix[("Coverage (new)", o)]["accuracy"] for o in orderings]
    lfu_accs = [matrix[("LFU (old)", o)]["accuracy"] for o in orderings]
    cov_spread = max(cov_accs) - min(cov_accs)
    lfu_spread = max(lfu_accs) - min(lfu_accs)

    if cov_spread < 5:
        verdict = "CURRICULUM-ROBUST"
        detail = (
            f"Coverage eviction spread is {cov_spread:.1f}pp across orderings "
            f"(vs LFU's {lfu_spread:.1f}pp). The write path handles class-sorted "
            f"arrival gracefully. No stream preprocessor needed."
        )
    elif cov_spread < 20:
        verdict = "PARTIALLY CURRICULUM-ROBUST"
        detail = (
            f"Coverage eviction spread is {cov_spread:.1f}pp — better than LFU's "
            f"{lfu_spread:.1f}pp but still ordering-sensitive. Consider lightweight "
            f"stream buffering for worst-case deployments."
        )
    else:
        verdict = "CURRICULUM-SENSITIVE"
        detail = (
            f"Coverage eviction spread is {cov_spread:.1f}pp — still catastrophically "
            f"ordering-sensitive. Architecture needs a stream preprocessor (reservoir "
            f"sampling, experience replay, or class-balanced buffers)."
        )

    # --- Writeup ---
    writeup = f"""# MT-11: Curriculum Stress Test

## Objective

Test whether coverage-aware eviction handles pathological stream orderings.
MT-4/MT-6 showed class-sorted arrival is catastrophic under LFU (+55pp delta).
MT-12 showed coverage eviction closes the coreset gap under default ordering.
Does it survive curriculum adversity?

## Config
- **Commit**: `{commit}`
- **Coverage eviction**: verified in production `associative_core.py`
- **Device**: {device}
- **Dataset**: Stanford Dogs 1024-d, production adapter
- **Capacity**: {CAPACITY}
- **Block-shuffle window**: {BLOCK_WINDOW}
- **Coreset reference**: {acc_coreset:.2f}%

## Results: 2x3 Accuracy Matrix

|                 | Class-sorted | IID Shuffled | Block-shuffled | Spread |
|-----------------|-------------|-------------|----------------|--------|
"""
    for wp_name in write_paths:
        accs = [matrix[(wp_name, o)]["accuracy"] for o in orderings]
        spread = max(accs) - min(accs)
        writeup += (
            f"| **{wp_name}** | "
            + " | ".join(f"{matrix[(wp_name, o)]['accuracy']:.2f}%" for o in orderings)
            + f" | {spread:.1f}pp |\n"
        )
    writeup += f"| **Coreset** | {acc_coreset:.2f}% | — | — | — |\n"

    writeup += """
## Class Distribution

|                 | Class-sorted | IID Shuffled | Block-shuffled |
|-----------------|-------------|-------------|----------------|
"""
    for wp_name in write_paths:
        writeup += (
            f"| **{wp_name}** | "
            + " | ".join(
                f"{matrix[(wp_name, o)]['classes_represented']}/{n_classes}"
                for o in orderings
            )
            + " |\n"
        )

    # Per-class breakdown for class-sorted
    writeup += "\n## Per-Class Accuracy (Class-Sorted Condition)\n\n"
    writeup += "| Write Path | Early (0–9) | Mid (55–64) | Late (110–119) |\n"
    writeup += "|------------|------------|------------|----------------|\n"
    for wp_name, pc in per_class_results.items():
        n_early = min(10, n_classes)
        mid_start = n_classes // 2 - 5
        n_late = min(10, n_classes)
        early = pc[:n_early].mean().item()
        mid = pc[mid_start:mid_start + 10].mean().item()
        late = pc[-n_late:].mean().item()
        writeup += f"| **{wp_name}** | {early:.1f}% | {mid:.1f}% | {late:.1f}% |\n"

    writeup += f"""
## Verdict

**{verdict}**

{detail}

## Artifacts
- `mt11_metrics.json` — all metrics and per-cell results
"""

    report_path = out_dir / "mt11_curriculum_stress.md"
    report_path.write_text(writeup)
    print(f"\n[{TAG}] Report: {report_path}")
    print(writeup)


if __name__ == "__main__":
    main()
