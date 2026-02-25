#!/usr/bin/env python3
"""
benchmarks/g36_leave_one_out.py — G36: Leave-One-Out Candidate Influence Map.

For each eval query, removes each of its top-K broad retrieval candidates one
at a time and recomputes the full downstream pipeline (Mahalanobis re-rank →
CSLS selection → NSTP → floor → softmax vote).  Measures per-candidate
influence on final top-1 accuracy.

Produces:
  - results/g36_per_query_influence.csv
  - results/g36_candidate_summary.csv
  - results/g36_influence_histogram.png
  - results/g36_influence_by_rank.png
  - results/g36_class_concentration.png
  - results/g36_writeup.md

Usage:
    PYTHONPATH=. python benchmarks/g36_leave_one_out.py --data-root data --device auto
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from fast_associative_memory import FastAssociativeMemory  # noqa: E402
from nstp import NSTPController  # noqa: E402
from associative_core import ContinuousCAM  # noqa: E402
from benchmarks.g30_adapter_sweep import (  # noqa: E402
    load_domain_features as g30_load_domain_features,
)
from benchmarks.g31_retrain_adapters import _load_g30_domain_adapter  # noqa: E402


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


# ---------------------------------------------------------------------------
# Core: recompute pipeline from broad candidates with one removed
# ---------------------------------------------------------------------------

@torch.no_grad()
def _predict_from_broad(
    core: ContinuousCAM,
    nstp: NSTPController,
    q_norm: torch.Tensor,        # (1, D) — normalized, already projected
    queries_float: torch.Tensor,  # (1, D) — float, already projected
    broad_slots: torch.Tensor,    # (K,) — memory slot indices
    broad_sims_raw: torch.Tensor, # (K,) — raw cosine sims (unused, kept for API)
) -> int:
    """Recompute Mahalanobis → CSLS → top-k → NSTP → floor → softmax → argmax.

    Returns predicted class index.
    """
    K = broad_slots.shape[0]
    final_k = min(core.inference_k, K)

    # Mahalanobis re-ranking
    candidate_classes = core.values[broad_slots].float().argmax(dim=-1)  # (K,)
    local_vars = core.class_vars[candidate_classes].clamp(min=1e-4)      # (K, D)
    inv_std = local_vars.rsqrt()
    candidate_keys = core.keys[broad_slots].float()                       # (K, D)
    scaled_Q = queries_float.unsqueeze(0) * inv_std                       # (K, D)  broadcast (1,D)*(K,D)
    scaled_K = candidate_keys * inv_std
    reranked_sims = F.cosine_similarity(scaled_Q, scaled_K, dim=-1)      # (K,)

    # CSLS
    candidate_density = core.prototype_density[broad_slots]
    csls_sims = 2.0 * reranked_sims - core.csls_weight * candidate_density

    # Top-final_k selection
    _, topk_locs = csls_sims.topk(final_k)
    topk_slots = broad_slots[topk_locs]
    topk_sims = reranked_sims[topk_locs]

    # NSTP lateral inhibition
    topk_keys = core.keys[topk_slots].float()   # (final_k, D)
    topk_vals = core.values[topk_slots].float()  # (final_k, V)
    keep_mask, _ = nstp.prune_batch(
        q_norm.float(),
        topk_keys.unsqueeze(0),
        topk_vals.unsqueeze(0),
        topk_sims.unsqueeze(0),
    )
    topk_sims_masked = topk_sims.clone()
    topk_sims_masked[~keep_mask.squeeze(0)] = -float("inf")

    # Floor
    if core.inference_sim_floor > 0.0:
        topk_sims_masked[topk_sims_masked < core.inference_sim_floor] = -float("inf")

    # Softmax vote
    weights = F.softmax(topk_sims_masked / core.inference_temp, dim=-1)
    retrieved = core.values[topk_slots].float()
    output = (weights.unsqueeze(-1) * retrieved).sum(dim=0)  # (V,)
    return output.argmax().item()


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_leave_one_out(
    data_root: Path,
    device: torch.device,
    max_queries: int = 500,
    broad_k: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run leave-one-out occlusion on Stanford Dogs eval set."""

    print("[G36] Loading Stanford Dogs features...", flush=True)
    tr_f, tr_l, te_f, te_l, n_classes = g30_load_domain_features("dogs", data_root)
    input_dim = int(tr_f.shape[1])

    print("[G36] Loading production adapter...", flush=True)
    adapter = _load_g30_domain_adapter("dogs", input_dim=input_dim, device=device)

    nstp = NSTPController(sibling_threshold=0.85, depth_epsilon=0.10)

    mem = FastAssociativeMemory(
        input_dim=input_dim,
        value_dim=n_classes,
        core_entries=50000,
        core_vigilance=0.85,
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=25,
        inference_temp=0.05,
        use_lfu=True,
        adapter=adapter,
        nstp=nstp,
    ).to(device)

    # Write training data
    x_train = tr_f.to(device)
    y_train = tr_l.to(device)
    print("[G36] Writing training data to memory...", flush=True)
    for i in range(0, len(x_train), 256):
        mem.learn_local(x_train[i:i + 256], y_train[i:i + 256])
    n_protos = int(mem.core_cam.occupied.sum().item())
    print(f"[G36] Prototypes: {n_protos}", flush=True)

    core = mem.core_cam
    valid_idx = core.occupied.nonzero(as_tuple=True)[0]
    n_valid = len(valid_idx)
    actual_broad_k = min(broad_k, n_valid)

    # Select eval queries (up to max_queries from test set)
    x_test = te_f.to(device)
    y_test = te_l.to(device)
    n_eval = min(max_queries, len(x_test))
    print(f"[G36] Evaluating {n_eval} queries with K={actual_broad_k}...", flush=True)

    # Sanity check: compute full accuracy on the sample
    correct_full = 0
    rows = []

    t0 = time.time()
    for qi in range(n_eval):
        if qi % 50 == 0 and qi > 0:
            elapsed = time.time() - t0
            qps = qi / elapsed
            eta = (n_eval - qi) / qps
            print(f"  [{qi}/{n_eval}] {qps:.1f} q/s, ETA {eta:.0f}s", flush=True)

        raw_query = x_test[qi:qi + 1]  # (1, input_dim)
        true_label = y_test[qi].item()

        # Project through adapter
        proj = mem._project(raw_query)  # (1, D)
        q_norm = F.normalize(proj.float(), dim=-1)

        # Broad cosine search
        sims = q_norm @ core._keys_norm[valid_idx].T  # (1, N_occ)
        broad_sims_raw, broad_locs = sims.topk(actual_broad_k, dim=1)
        broad_slots = valid_idx[broad_locs.squeeze(0)]  # (K,)
        broad_sims_raw_1d = broad_sims_raw.squeeze(0)    # (K,)

        # Full pipeline prediction
        pred_full = _predict_from_broad(
            core, nstp, q_norm, proj.float().squeeze(0),
            broad_slots, broad_sims_raw_1d,
        )
        acc_full = int(pred_full == true_label)
        correct_full += acc_full

        # Determine stage labels for broad candidates
        # Run trace to get rejection stages
        outputs_tr, tr = core(proj, nstp=nstp, trace=True)

        # Build slot → stage mapping from trace
        slot_stage = {}
        for s in tr.final_slots[0].tolist():
            if s >= 0:
                slot_stage[s] = "final_survivor"
        for j, s in enumerate(tr.rejected_slots[0].tolist()):
            if s >= 0:
                stage_code = tr.rejection_stage[0][j].item()
                stage_names = {0: "rerank_drop", 1: "csls_survivor", 2: "nstp_kill", 3: "floor_kill"}
                slot_stage[s] = stage_names.get(stage_code, f"unknown_{stage_code}")

        # Leave-one-out: remove each candidate and recompute
        for ci in range(actual_broad_k):
            # Build mask: all except candidate ci
            mask = torch.ones(actual_broad_k, dtype=torch.bool, device=device)
            mask[ci] = False
            occluded_slots = broad_slots[mask]
            occluded_sims = broad_sims_raw_1d[mask]

            pred_minus_c = _predict_from_broad(
                core, nstp, q_norm, proj.float().squeeze(0),
                occluded_slots, occluded_sims,
            )
            acc_minus_c = int(pred_minus_c == true_label)
            delta = acc_full - acc_minus_c

            slot_id = broad_slots[ci].item()
            cand_class = core.values[slot_id].float().argmax().item()

            rows.append({
                "query_id": qi,
                "true_class": true_label,
                "candidate_slot": slot_id,
                "candidate_class": cand_class,
                "rank_in_broad_topK": ci,
                "stage": slot_stage.get(slot_id, "broad"),
                "acc_full": acc_full,
                "acc_minus_c": acc_minus_c,
                "delta": delta,
            })

    elapsed = time.time() - t0
    full_acc = correct_full / n_eval * 100
    print(f"[G36] Done in {elapsed:.1f}s. Full accuracy on sample: {full_acc:.2f}%", flush=True)

    # Build DataFrames
    df = pd.DataFrame(rows)

    # Candidate summary: aggregate across queries
    cand_summary = df.groupby("candidate_slot").agg(
        candidate_class=("candidate_class", "first"),
        n_queries=("delta", "count"),
        mean_delta=("delta", "mean"),
        sum_positive=("delta", lambda x: (x > 0).sum()),
        sum_negative=("delta", lambda x: (x < 0).sum()),
        sum_zero=("delta", lambda x: (x == 0).sum()),
        mean_rank=("rank_in_broad_topK", "mean"),
        stage_mode=("stage", lambda x: x.mode().iloc[0] if len(x) > 0 else "unknown"),
    ).reset_index()

    return df, cand_summary, full_acc, n_eval, n_classes


# ---------------------------------------------------------------------------
# Plotting and writeup
# ---------------------------------------------------------------------------

def generate_outputs(
    df: pd.DataFrame,
    cand_summary: pd.DataFrame,
    full_acc: float,
    n_eval: int,
    n_classes: int,
    out_dir: Path,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save CSVs
    df.to_csv(out_dir / "g36_per_query_influence.csv", index=False)
    cand_summary.to_csv(out_dir / "g36_candidate_summary.csv", index=False)
    print(f"[G36] Saved CSVs to {out_dir}", flush=True)

    # --- Plot 1: Influence distribution histogram ---
    fig, ax = plt.subplots(figsize=(8, 5))
    delta_counts = df["delta"].value_counts().sort_index()
    ax.bar(delta_counts.index, delta_counts.values, color=["#d32f2f", "#888", "#2e7d32"])
    ax.set_xlabel("Delta (acc_full - acc_minus_c)")
    ax.set_ylabel("Count")
    ax.set_title("G36: Leave-One-Out Influence Distribution")
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(["-1 (harmful)", "0 (redundant)", "+1 (load-bearing)"])
    for i, (x, y) in enumerate(zip(delta_counts.index, delta_counts.values)):
        ax.text(x, y + max(delta_counts.values) * 0.02, str(y), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "g36_influence_histogram.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: Mean influence by rank bucket ---
    df["rank_bucket"] = pd.cut(df["rank_in_broad_topK"], bins=10)
    rank_agg = df.groupby("rank_bucket", observed=True)["delta"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(rank_agg)), rank_agg.values, color="#1976d2")
    ax.set_xticks(range(len(rank_agg)))
    ax.set_xticklabels([str(b) for b in rank_agg.index], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Rank Bucket")
    ax.set_ylabel("Mean Delta")
    ax.set_title("G36: Mean Influence by Rank Bucket")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "g36_influence_by_rank.png", dpi=150)
    plt.close(fig)

    # --- Plot 3: Per-class influence concentration ---
    # For each class, compute share of total positive deltas from top-10 candidates
    positive_df = df[df["delta"] > 0].copy()
    class_concentration = []
    for cls in range(n_classes):
        cls_pos = positive_df[positive_df["true_class"] == cls]
        if len(cls_pos) == 0:
            class_concentration.append({"class": cls, "top10_share": 0.0, "total_positive": 0})
            continue
        cand_pos = cls_pos.groupby("candidate_slot")["delta"].sum().sort_values(ascending=False)
        total = cand_pos.sum()
        top10 = cand_pos.head(10).sum()
        class_concentration.append({
            "class": cls,
            "top10_share": top10 / total if total > 0 else 0.0,
            "total_positive": int(total),
        })
    conc_df = pd.DataFrame(class_concentration)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(conc_df["class"], conc_df["top10_share"], color="#7b1fa2", alpha=0.7)
    ax.set_xlabel("Dog Class")
    ax.set_ylabel("Top-10 Candidates' Share of Positive Deltas")
    ax.set_title("G36: Per-Class Influence Concentration")
    ax.axhline(conc_df["top10_share"].mean(), color="red", linestyle="--",
               label=f"Mean = {conc_df['top10_share'].mean():.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "g36_class_concentration.png", dpi=150)
    plt.close(fig)

    # --- Compute summary stats ---
    n_load_bearing = int((df["delta"] > 0).sum())
    n_harmful = int((df["delta"] < 0).sum())
    n_redundant = int((df["delta"] == 0).sum())
    total_pairs = len(df)

    # Per-query stats
    per_query = df.groupby("query_id").agg(
        n_positive=("delta", lambda x: (x > 0).sum()),
        n_negative=("delta", lambda x: (x < 0).sum()),
    )

    # Global load-bearing candidates (mean delta > 0.001)
    global_lb = cand_summary[cand_summary["mean_delta"] > 0.001]
    global_harmful = cand_summary[cand_summary["mean_delta"] < -0.001]
    global_redundant = cand_summary[cand_summary["mean_delta"].abs() < 0.0001]

    # Top 20 load-bearing
    top20_lb = cand_summary.nlargest(20, "mean_delta")
    top20_harmful = cand_summary.nsmallest(20, "mean_delta")

    # Stage breakdown
    stage_influence = df.groupby("stage")["delta"].agg(["mean", "count", "sum"])

    # --- Writeup ---
    writeup = f"""# G36: Leave-One-Out Candidate Influence Map

## Setup
- **Dataset**: Stanford Dogs (120 classes)
- **Eval queries**: {n_eval}
- **Broad K**: {df['rank_in_broad_topK'].max() + 1}
- **Full accuracy on sample**: {full_acc:.2f}%
- **Total (query, candidate) pairs**: {total_pairs:,}

## Key Takeaways

### Influence Distribution
- **Load-bearing** (delta > 0): {n_load_bearing:,} pairs ({n_load_bearing/total_pairs*100:.1f}%)
- **Harmful** (delta < 0): {n_harmful:,} pairs ({n_harmful/total_pairs*100:.1f}%)
- **Redundant** (delta = 0): {n_redundant:,} pairs ({n_redundant/total_pairs*100:.1f}%)

### Per-Query Summary
- Mean load-bearing candidates per query: {per_query['n_positive'].mean():.2f}
- Mean harmful candidates per query: {per_query['n_negative'].mean():.2f}
- Queries with at least one load-bearing candidate: {(per_query['n_positive'] > 0).sum()}/{n_eval}
- Queries with at least one harmful candidate: {(per_query['n_negative'] > 0).sum()}/{n_eval}

### Global Candidate Summary
- Distinct load-bearing candidates (mean delta > 0.001): {len(global_lb)}
- Distinct harmful candidates (mean delta < -0.001): {len(global_harmful)}
- Distinct redundant candidates (|mean delta| < 0.0001): {len(global_redundant)}

### Stage Breakdown
{stage_influence.to_string()}

### Per-Class Concentration
- Mean top-10 share of positive deltas: {conc_df['top10_share'].mean():.3f}
- Classes with top-10 share > 0.8: {(conc_df['top10_share'] > 0.8).sum()}/{n_classes}
- Classes with zero positive deltas: {(conc_df['total_positive'] == 0).sum()}/{n_classes}

## Top 20 Load-Bearing Candidates

{top20_lb[['candidate_slot', 'candidate_class', 'n_queries', 'mean_delta', 'sum_positive', 'mean_rank', 'stage_mode']].to_string(index=False)}

## Top 20 Harmful Candidates

{top20_harmful[['candidate_slot', 'candidate_class', 'n_queries', 'mean_delta', 'sum_negative', 'mean_rank', 'stage_mode']].to_string(index=False)}

## Plots
- `g36_influence_histogram.png` — Distribution of delta values
- `g36_influence_by_rank.png` — Mean influence by rank bucket
- `g36_class_concentration.png` — Per-class positive influence concentration

## Sanity Checks
- Full accuracy on {n_eval} sample queries: {full_acc:.2f}%
- Each occluded run recomputes Mahalanobis → CSLS → NSTP → floor → softmax from scratch
- No caching of downstream results between occlusion runs
"""

    (out_dir / "g36_writeup.md").write_text(writeup)
    print(f"[G36] Writeup saved to {out_dir / 'g36_writeup.md'}", flush=True)

    # Print summary
    print("\n" + "=" * 60)
    print("G36 SUMMARY")
    print("=" * 60)
    print(f"Full accuracy: {full_acc:.2f}%")
    print(f"Load-bearing pairs: {n_load_bearing:,} / {total_pairs:,} ({n_load_bearing/total_pairs*100:.1f}%)")
    print(f"Harmful pairs: {n_harmful:,} / {total_pairs:,} ({n_harmful/total_pairs*100:.1f}%)")
    print(f"Redundant pairs: {n_redundant:,} / {total_pairs:,} ({n_redundant/total_pairs*100:.1f}%)")
    print(f"Global load-bearing candidates: {len(global_lb)}")
    print(f"Global harmful candidates: {len(global_harmful)}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="G36: Leave-One-Out Influence Map")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--broad-k", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="results/g36")
    args = parser.parse_args()

    set_all_seeds(42)
    device = device_from_arg(args.device)
    print(f"[G36] Device: {device}", flush=True)

    df, cand_summary, full_acc, n_eval, n_classes = run_leave_one_out(
        data_root=Path(args.data_root),
        device=device,
        max_queries=args.max_queries,
        broad_k=args.broad_k,
    )

    generate_outputs(df, cand_summary, full_acc, n_eval, n_classes, Path(args.out_dir))


if __name__ == "__main__":
    main()
