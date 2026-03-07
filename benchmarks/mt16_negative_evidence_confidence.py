#!/usr/bin/env python3
"""
MT-16 — Negative Evidence Confidence Calibration

Tests whether rejection-based metrics (RAS, RVA, RE) serve as confidence
estimators, complementing or beating softmax margin.

Read-only analysis on G37-style traces. No model modifications.

Usage:
    PYTHONPATH=. .venv/bin/python benchmarks/mt16_negative_evidence_confidence.py
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from fast_associative_memory import FastAssociativeMemory
from nstp import NSTPController
from benchmarks.g30_adapter_sweep import load_domain_features as g30_load_domain_features
from benchmarks.g31_retrain_adapters import _load_g30_domain_adapter

SEED = 42
P_THRESHOLD = 0.01


def set_seeds(seed=SEED):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Trace collection (reused from G37/G39)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_traces(train_feats, train_labels, test_feats, test_labels,
                   adapter, device, max_traces=100_000):
    num_classes = int(train_labels.max().item()) + 1
    nstp = NSTPController(sibling_threshold=0.85, depth_epsilon=0.10)

    mem = FastAssociativeMemory(
        input_dim=int(train_feats.shape[1]),
        value_dim=num_classes,
        core_entries=50000, core_vigilance=0.85,
        hebb_lr=0.1, key_lr=0.05,
        inference_k=25, inference_temp=0.05,
        use_lfu=True, adapter=adapter, nstp=nstp,
    ).to(device)

    x_train = train_feats.to(device)
    y_train = train_labels.to(device)
    for i in range(0, len(x_train), 256):
        mem.learn_local(x_train[i:i + 256], y_train[i:i + 256])
    print(f"    Prototypes: {mem.core_cam.occupied.sum().item()}")

    query_feats, query_labels = test_feats, test_labels
    if len(test_feats) < max_traces and len(train_feats) > len(test_feats):
        query_feats = torch.cat([test_feats, train_feats], dim=0)[:max_traces]
        query_labels = torch.cat([test_labels, train_labels], dim=0)[:max_traces]

    core = mem.core_cam
    occ_idx = core.occupied.nonzero(as_tuple=True)[0]
    proto_classes = core.values[occ_idx].float().argmax(dim=-1)

    x_q = query_feats.to(device)
    y_q = query_labels.to(device)

    records = []
    for i in range(0, min(len(x_q), max_traces), 256):
        bx = x_q[i:i + 256]
        by = y_q[i:i + 256]
        B = bx.size(0)

        proj = mem._project(bx)
        outputs, tr = core(proj, nstp=nstp, trace=True)

        q_norm = F.normalize(proj.float(), dim=-1)
        proto_keys = core._keys_norm[occ_idx]
        sims = q_norm @ proto_keys.T
        pred_classes = outputs.argmax(dim=1)

        for b in range(B):
            true_cls = by[b].item()
            pred_cls = pred_classes[b].item()

            row_sims = sims[b]
            best_sim = row_sims.max().item()
            same_mask = proto_classes == pred_cls
            other_sims = row_sims.masked_fill(same_mask, -1e9)
            second = other_sims.max().item() if other_sims.max().item() > -1e8 else best_sim
            margin = best_sim - second

            # Softmax vote distribution
            sm_vote = F.softmax(outputs[b], dim=-1).cpu()

            # Accepted classes
            acc_cls = []
            for s in tr.final_slots[b].tolist():
                if 0 <= s < core.max_entries and core.occupied[s]:
                    acc_cls.append(core.values[s].float().argmax().item())

            # Rejected classes
            rej_cls = []
            for s in tr.rejected_slots[b].tolist():
                if 0 <= s < core.max_entries and core.occupied[s]:
                    rej_cls.append(core.values[s].float().argmax().item())

            records.append({
                "true_class": true_cls,
                "pred_class": pred_cls,
                "correct": true_cls == pred_cls,
                "margin": margin,
                "sm_vote": sm_vote,
                "accepted_classes": acc_cls,
                "rejected_classes": rej_cls,
                "num_classes": num_classes,
            })

    del mem
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


# ---------------------------------------------------------------------------
# Confidence metrics
# ---------------------------------------------------------------------------

def compute_metrics(records):
    """Compute RAS, RVA, RE, and margin for each query."""
    margins = []
    ras_scores = []   # Cramér's V
    rva_scores = []   # cosine(rej_hist, sm_vote)
    re_scores = []    # normalized rejection entropy
    correct = []

    for r in records:
        nc = r["num_classes"]
        margins.append(r["margin"])
        correct.append(r["correct"])

        rej = r["rejected_classes"]
        acc = r["accepted_classes"]

        # Rejection histogram
        rej_hist = np.zeros(nc)
        for c in rej:
            rej_hist[c] += 1

        # --- RAS: Cramér's V from contingency table ---
        if len(rej) >= 5 and len(acc) >= 2:
            all_cls = sorted(set(rej) | set(acc))
            if len(all_cls) >= 2:
                acc_counts = Counter(acc)
                rej_counts = Counter(rej)
                acc_row = [acc_counts.get(c, 0) for c in all_cls]
                rej_row = [rej_counts.get(c, 0) for c in all_cls]
                table = np.array([acc_row, rej_row])
                nonzero = table.sum(axis=0) > 0
                table = table[:, nonzero]
                if table.shape[1] >= 2 and table.sum() >= 10:
                    chi2, p, dof, _ = sp_stats.chi2_contingency(table)
                    n = table.sum()
                    k = min(table.shape)
                    v = float(np.sqrt(chi2 / (n * (k - 1)))) if n > 0 else 0.0
                    ras_scores.append(v)
                else:
                    ras_scores.append(0.0)
            else:
                ras_scores.append(0.0)
        else:
            ras_scores.append(0.0)

        # --- RVA: cosine(rej_hist_norm, sm_vote) ---
        sm = r["sm_vote"].numpy().astype(np.float64)
        if rej_hist.sum() > 0:
            rh_norm = rej_hist / rej_hist.sum()
            dot = np.dot(rh_norm, sm)
            norm_rh = np.linalg.norm(rh_norm)
            norm_sm = np.linalg.norm(sm)
            rva = dot / (norm_rh * norm_sm + 1e-12)
            rva_scores.append(float(rva))
        else:
            rva_scores.append(0.0)

        # --- RE: normalized Shannon entropy of rejection distribution ---
        if rej_hist.sum() > 0:
            rh_norm = rej_hist / rej_hist.sum()
            rh_norm = rh_norm[rh_norm > 0]
            entropy = -np.sum(rh_norm * np.log(rh_norm))
            max_entropy = np.log(nc)
            re_scores.append(float(entropy / max_entropy))
        else:
            re_scores.append(1.0)  # max uncertainty

    return {
        "margin": np.array(margins),
        "ras": np.array(ras_scores),
        "rva": np.array(rva_scores),
        "re": np.array(re_scores),
        "correct": np.array(correct, dtype=bool),
    }


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def auroc(scores, labels):
    """Manual AUROC (no sklearn dependency)."""
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Wilcoxon-Mann-Whitney
    n_pos, n_neg = len(pos), len(neg)
    all_scores = np.concatenate([pos, neg])
    all_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    order = np.argsort(-all_scores)
    sorted_labels = all_labels[order]
    cum_pos = np.cumsum(sorted_labels)
    # Count concordant pairs
    auc = np.sum((1 - sorted_labels) * cum_pos) / (n_pos * n_neg)
    return float(auc)


def ece(scores, labels, n_bins=10):
    """Expected Calibration Error."""
    # Normalize scores to [0,1] range
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-12:
        return 0.0
    conf = (scores - s_min) / (s_max - s_min)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    total_ece = 0.0
    for i in range(n_bins):
        mask = (conf >= bin_edges[i]) & (conf < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (conf >= bin_edges[i]) & (conf <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = conf[mask].mean()
        total_ece += mask.sum() / len(labels) * abs(bin_acc - bin_conf)
    return float(total_ece)


def precision_at_k(scores, labels, k):
    """Precision at K for misclassification detection (lowest confidence = most suspicious)."""
    order = np.argsort(scores)  # ascending — lowest confidence first
    top_k = order[:k]
    # We want to detect INCORRECT predictions, so label=False is "positive"
    n_incorrect_in_topk = (~labels[top_k]).sum()
    return float(n_incorrect_in_topk / k) if k > 0 else 0.0


def calibration_table(scores, labels, n_bins=10):
    """Return (bin_center, bin_acc, bin_count) for reliability diagram."""
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-12:
        return [], [], []
    conf = (scores - s_min) / (s_max - s_min)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    centers, accs, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        centers.append((lo + hi) / 2)
        accs.append(float(labels[mask].mean()))
        counts.append(int(mask.sum()))
    return centers, accs, counts


# ---------------------------------------------------------------------------
# Composite metric
# ---------------------------------------------------------------------------

def fit_composite(metrics, correct):
    """Simple rank-averaging composite (no learned weights)."""
    # Rank each metric (higher = more confident)
    rank_margin = sp_stats.rankdata(metrics["margin"])
    rank_rva = sp_stats.rankdata(metrics["rva"])
    rank_re_inv = sp_stats.rankdata(-metrics["re"])  # invert: low entropy = high confidence
    rank_ras = sp_stats.rankdata(metrics["ras"])

    composite = (rank_margin + rank_rva + rank_re_inv) / 3.0
    return composite


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_domain(domain, data_root, device):
    print(f"\n  Loading {domain} features and adapter...")
    tr_f, tr_l, te_f, te_l, nc = g30_load_domain_features(domain, data_root)
    adapter = _load_g30_domain_adapter(domain, input_dim=tr_f.shape[1], device=device)
    print(f"  Train: {len(tr_f)}, Test: {len(te_f)}, Classes: {nc}")

    print("  Collecting traces...")
    set_seeds()
    records = collect_traces(tr_f, tr_l, te_f, te_l, adapter, device)
    print(f"  Collected {len(records)} traces")

    acc = 100.0 * sum(r["correct"] for r in records) / len(records)
    misclass_rate = 100.0 - acc
    print(f"  Baseline accuracy: {acc:.2f}% (misclass rate: {misclass_rate:.1f}%)")

    # Compute metrics
    M = compute_metrics(records)
    n_total = len(M["correct"])
    n_incorrect = (~M["correct"]).sum()

    # Composite
    composite = fit_composite(M, M["correct"])

    # Adjust P@K to available misclassifications
    pk_values = [100, 500] if n_incorrect >= 100 else [50, min(200, n_incorrect)]

    # --- Metric comparison table ---
    print(f"\n  Confidence Metric Comparison:")
    header = f"  {'':24s} {'AUROC':>7} {'ECE':>7}"
    for k in pk_values:
        header += f" {'P@' + str(k):>7}"
    print(header)

    metric_names = [
        ("Softmax Margin", M["margin"]),
        ("RAS (Cramér's V)", M["ras"]),
        ("RVA (cosine align)", M["rva"]),
        ("RE (1-entropy)", 1.0 - M["re"]),  # invert so high = confident
        ("Composite (rank avg)", composite),
    ]

    for name, scores in metric_names:
        auc = auroc(scores, M["correct"])
        e = ece(scores, M["correct"])
        row = f"  {name:24s} {auc:>7.3f} {e:>7.3f}"
        for k in pk_values:
            p = precision_at_k(scores, M["correct"], k)
            row += f" {100 * p:>6.1f}%"
        print(row)

    # --- Margin-stratified AUROC ---
    print(f"\n  Margin-Stratified AUROC (within-quartile):")
    sorted_idx = np.argsort(M["margin"])
    q_size = n_total // 4
    quartile_names = ["Q1(low)", "Q2", "Q3", "Q4(high)"]

    print(f"  {'':18s}", end="")
    for qn in quartile_names:
        print(f" {qn:>8}", end="")
    print()

    for name, scores in [("Margin", M["margin"]),
                          ("RVA", M["rva"]),
                          ("Composite", composite)]:
        print(f"  {name:18s}", end="")
        for qi in range(4):
            start = qi * q_size
            end = (qi + 1) * q_size if qi < 3 else n_total
            idx = sorted_idx[start:end]
            if M["correct"][idx].all() or (~M["correct"][idx]).all():
                print(f" {'N/A':>8}", end="")
            else:
                auc = auroc(scores[idx], M["correct"][idx])
                print(f" {auc:>8.3f}", end="")
        print()

    # --- Correlations ---
    print(f"\n  Correlation between metrics:")
    for label, arr in [("RAS", M["ras"]), ("RVA", M["rva"]), ("RE", M["re"])]:
        r, p = sp_stats.pearsonr(M["margin"], arr)
        print(f"    {label:3s} vs Margin: r = {r:+.3f} (p = {p:.2e})")

    # --- Calibration curves (ASCII) ---
    print(f"\n  Calibration Curves (10 bins, predicted_conf → observed_acc):")
    for name, scores in [("Margin", M["margin"]),
                          ("RVA", M["rva"]),
                          ("1-RE", 1.0 - M["re"]),
                          ("Composite", composite)]:
        centers, accs, counts = calibration_table(scores, M["correct"])
        print(f"\n    {name}:")
        print(f"    {'Bin':>5} {'Acc':>7} {'Count':>6}")
        for c, a, n in zip(centers, accs, counts):
            bar = "#" * int(a * 30)
            print(f"    {c:>5.2f} {a:>6.1%} {n:>6} |{bar}")

    return M, composite


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path("data")

    print("=" * 65)
    print("  MT-16 — Negative Evidence Confidence Calibration")
    print("=" * 65)

    # Dogs (primary)
    print("\n" + "=" * 65)
    print("  DOMAIN: DOGS")
    print("=" * 65)
    M_dogs, comp_dogs = run_domain("dogs", data_root, device)

    # Aircraft (secondary, if available)
    try:
        print("\n" + "=" * 65)
        print("  DOMAIN: AIRCRAFT")
        print("=" * 65)
        M_air, comp_air = run_domain("aircraft", data_root, device)
    except Exception as e:
        print(f"  Aircraft skipped: {e}")

    print("\n" + "=" * 65)
    print("  MT-16 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
