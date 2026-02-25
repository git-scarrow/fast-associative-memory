#!/usr/bin/env python3
"""
benchmarks/mt6_ema_disabled.py — MT-6: EMA-Disabled Configuration Test.

Tests the prototype store with EMA pathway entirely disabled:
- Hits are dropped entirely (no drift, value update, hit-count, etc.).
- Misses are allocated directly.
- Runs at multiple capacity caps on Stanford Dogs.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from associative_core import ContinuousCAM
from fast_associative_memory import FastAssociativeMemory
from nstp import NSTPController
from benchmarks.g30_adapter_sweep import load_domain_features as g30_load_domain_features
from benchmarks.g31_retrain_adapters import _load_g30_domain_adapter

TAG = "MT-6"

CAPACITIES = [1000, 5000, 10000, 25000, 50000]
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


class EMADisabledCAM(ContinuousCAM):
    """ContinuousCAM but with EMA updates completely disabled."""

    def learn_local(self, queries: torch.Tensor, targets: torch.Tensor):
        now = time.time()
        queries = self._cast(queries)
        targets = self._cast(targets)

        # Update per-class EMA mean and variance (needed for Mahalanobis read-path)
        q_float = queries.float()
        class_labels = targets.float().argmax(dim=-1)
        for c in class_labels.unique():
            x_c = q_float[class_labels == c]
            batch_mean = x_c.mean(0)
            deviation = x_c - self.class_means[c]
            batch_var = (deviation ** 2).mean(0)
            self.class_means[c] = ((1.0 - self.var_ema_alpha) * self.class_means[c]
                                   + self.var_ema_alpha * batch_mean)
            self.class_vars[c] = ((1.0 - self.var_ema_alpha) * self.class_vars[c]
                                  + self.var_ema_alpha * batch_var)

        best_slots, best_sims, query_density = self._get_nearest_batch(queries)

        if self.dynamic_vigilance is not None and self.occupied.any():
            # Dynamic vigilance calculation
            valid_idx = self.occupied.nonzero(as_tuple=True)[0]
            keys_occ = self._keys_norm[valid_idx]
            proto_labels = self.values[valid_idx].float().argmax(dim=-1)

            B = queries.size(0)
            best_classes = torch.full((B,), -1, dtype=torch.long, device=queries.device)
            has_best = best_slots >= 0
            if has_best.any():
                best_classes[has_best] = self.values[best_slots[has_best]].float().argmax(dim=-1)

            margins = torch.empty((B,), device=queries.device, dtype=torch.float32)
            v_eff = torch.empty((B,), device=queries.device, dtype=torch.float32)

            very_low = -1e9
            chunk_q = 16
            dv = self.dynamic_vigilance

            for s in range(0, B, chunk_q):
                e = min(B, s + chunk_q)
                q = F.normalize(queries[s:e].float(), dim=-1)
                sims = q @ keys_occ.T

                bc = best_classes[s:e]
                same_class = proto_labels.unsqueeze(0) == bc.unsqueeze(1)
                sims_other = sims.masked_fill(same_class, very_low)
                sim_other, _ = sims_other.max(dim=1)

                margin = best_sims[s:e] - sim_other
                margins[s:e] = margin

                v = dv.v_base - dv.alpha * margin
                v_eff[s:e] = torch.clamp(v, dv.v_floor, dv.v_ceiling)

            self._dynamic_vigilance_stats["sum_v"] += float(v_eff.sum().item())
            self._dynamic_vigilance_stats["sum_margin"] += float(margins.sum().item())
            self._dynamic_vigilance_stats["count"] += int(v_eff.numel())

            margin_log = getattr(self, "margin_log", None)
            if isinstance(margin_log, list):
                margin_log.append(margins.detach().cpu())
            v_log = getattr(self, "vigilance_log", None)
            if isinstance(v_log, list):
                v_log.append(v_eff.detach().cpu())

            vigilance_thresholds = v_eff.to(best_sims.device)
        else:
            vigilance_thresholds = torch.full_like(best_sims, self.vigilance)

        hits = (best_slots >= 0) & (best_sims >= vigilance_thresholds)

        # Bipartite class check: demote hits where stored class differs
        if hits.any():
            hit_slots_all = best_slots[hits]
            stored_vals = self.values[hit_slots_all]
            payload_sims = F.cosine_similarity(targets[hits], stored_vals, dim=-1)
            same_class = payload_sims > 0.5

            if not same_class.all():
                hit_indices = hits.nonzero(as_tuple=True)[0]
                hits[hit_indices[~same_class]] = False

        misses = ~hits

        # Condition B: Hits are completely ignored (dropped). No update block here!

        # Batch allocation for misses
        if misses.any():
            miss_queries = queries[misses]
            miss_targets = targets[misses]
            n_miss = miss_queries.size(0)

            new_slots = self._alloc_slots_batch(n_miss)
            n_alloc = len(new_slots)
            
            self.keys[new_slots] = miss_queries[:n_alloc]
            self.values[new_slots] = miss_targets[:n_alloc]
            self.occupied[new_slots] = True
            self.last_seen[new_slots] = now
            self.usage[new_slots] = 1
            self.hit_counts[new_slots] = 1
            self.prototype_density[new_slots] = query_density[misses][:n_alloc]
            self._update_key_norm(new_slots)


def make_fam(
    adapter,
    n_classes: int,
    capacity: int,
    device: torch.device,
    disable_ema: bool = False
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

    if disable_ema:
        # Replace the core_cam with our EMADisabledCAM
        # We must copy all the init args of ContinuousCAM that FastAssociativeMemory just set
        ema_disabled_cam = EMADisabledCAM(
            key_dim=fam.core_cam.key_dim,
            value_dim=fam.core_cam.value_dim,
            max_entries=fam.core_cam.max_entries,
            vigilance=fam.core_cam.vigilance,
            hebb_lr=fam.core_cam.hebb_lr,
            aging_time=fam.core_cam.aging_time,
            flood_scale=fam.core_cam.flood_scale,
            immutable_keys=fam.core_cam.immutable_keys,
            use_lfu=fam.core_cam.use_lfu,
            use_bfloat16=fam.core_cam.use_bfloat16,
            key_lr=fam.core_cam.key_lr,
            inference_k=fam.core_cam.inference_k,
            inference_temp=fam.core_cam.inference_temp,
            ema_beta=fam.core_cam.ema_beta,
            dynamic_vigilance=fam.core_cam.dynamic_vigilance
        ).to(device)
        # Assign it back
        fam.core_cam = ema_disabled_cam

    return fam


@torch.no_grad()
def eval_accuracy(fam: FastAssociativeMemory, test_feats: torch.Tensor, test_labels: torch.Tensor) -> float:
    correct = 0
    total = len(test_feats)
    for i in range(0, total, 256):
        batch = test_feats[i:i + 256]
        labels = test_labels[i:i + 256]
        outputs = fam(batch)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
    return correct / total * 100 if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description=f"{TAG}: EMA-Disabled Configuration Test")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-root", type=str, default="data")
    args = parser.parse_args()

    device = (torch.device("cuda") if args.device == "auto" and torch.cuda.is_available()
              else torch.device(args.device if args.device != "auto" else "cpu"))
    print(f"[{TAG}] Device: {device}", flush=True)

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

    results = []

    for cap in CAPACITIES:
        print(f"\n--- Capacity: {cap} ---")

        stats_row = {"cap": cap}

        for condition_name, disable_ema in [("EMA-On", False), ("EMA-Off", True)]:
            set_all_seeds(SEED)
            fam = make_fam(adapter, n_classes, cap, device, disable_ema=disable_ema)
            
            start_t = time.time()
            for i in range(0, n_train, BATCH_SIZE):
                fam.learn_local(train_feats[i:i+BATCH_SIZE], train_labels[i:i+BATCH_SIZE])
            
            acc = eval_accuracy(fam, test_feats, test_labels)
            occ = fam.core_cam.occupied.sum().item()
            cr = occ / n_train

            stats_row[f"{condition_name}_Acc"] = acc
            stats_row[f"{condition_name}_Occ"] = occ
            stats_row[f"{condition_name}_CR"] = cr

            print(f"  {condition_name}: Acc={acc:.2f}%, Occ={occ} (CR={cr:.3f})")

        delta = stats_row["EMA-On_Acc"] - stats_row["EMA-Off_Acc"]
        stats_row["Delta"] = delta
        results.append(stats_row)

    print("\n### Results Table\n")
    print("| Capacity | EMA-On Acc | EMA-Off Acc | Delta | EMA-On CR | EMA-Off CR | EMA-On Occ | EMA-Off Occ |")
    print("|----------|-----------|-------------|-------|-----------|------------|------------|-------------|")
    for r in results:
        print(f"| {r['cap']:<8} | {r['EMA-On_Acc']:>9.2f}% | {r['EMA-Off_Acc']:>10.2f}% | {r['Delta']:>4.2f}pp | {r['EMA-On_CR']:>9.3f} | {r['EMA-Off_CR']:>10.3f} | {r['EMA-On_Occ']:>10} | {r['EMA-Off_Occ']:>11} |")

    # Generate writeup logic
    writeup = "\n### Analysis\n\n"
    
    # Check max delta
    deltas = [r["Delta"] for r in results]
    max_delta = max(abs(d) for d in deltas)
    
    if max_delta <= 1.0:
        writeup += "Accuracy is statistically equivalent (all deltas within ±1pp) across all capacity points. "
        writeup += "The EMA pathway is confirmed architecturally inert and should be removed entirely. "
        writeup += "This validates that prototype tracking and blending adds no practical value, and raw exemplar insertion alone is sufficient."
    else:
        writeup += f"Significant differences detected (max delta {max_delta:.2f}pp). "
        # Check transition point
        drops_at_low = any(d > 1.0 for d, r in zip(deltas, results) if r['cap'] <= 5000)
        drops_at_high = any(d > 1.0 for d, r in zip(deltas, results) if r['cap'] >= 25000)
        
        if drops_at_low and not drops_at_high:
            writeup += "Accuracy drops at low capacity but not high capacity. EMA refinement matters when coverage is sparse and every prototype position counts."
        elif drops_at_high and not drops_at_low:
            writeup += "Accuracy drops at high capacity but not low capacity. EMA refinement matters when the store is saturated and blending provides denoising."
        else:
            writeup += "Accuracy decreases uniformly across capacities. EMA contributes real information at all scales."
            
        writeup += "\n\nTransition points exceeding 1pp delta:\n"
        for r in results:
            if r['Delta'] > 1.0:
                writeup += f"- Capacity {r['cap']}: {r['Delta']:.2f}pp drop without EMA\n"

    print(writeup)
    
    out_dir = Path("results/mt6")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "mt6_ema_disabled_dogs.md"
    
    md_content = "# MT-6: EMA-Disabled Architectual Test\n\n"
    md_content += "Dataset: Stanford Dogs (1024-d features)\n"
    md_content += "Adapter: Production Dogs\n\n"
    
    md_content += "### Results Table\n\n"
    md_content += "| Capacity | EMA-On Acc | EMA-Off Acc | Delta | EMA-On CR | EMA-Off CR | EMA-On Occ | EMA-Off Occ |\n"
    md_content += "|----------|-----------|-------------|-------|-----------|------------|------------|-------------|\n"
    for r in results:
        md_content += f"| {r['cap']} | {r['EMA-On_Acc']:.2f}% | {r['EMA-Off_Acc']:.2f}% | {r['Delta']:.2f}pp | {r['EMA-On_CR']:.4f} | {r['EMA-Off_CR']:.4f} | {r['EMA-On_Occ']} | {r['EMA-Off_Occ']} |\n"
        
    md_content += writeup
    
    report_path.write_text(md_content)
    print(f"\n[{TAG}] Saved report to {report_path}")

if __name__ == "__main__":
    main()
