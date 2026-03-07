#!/usr/bin/env python3
"""
G39 — Negative Evidence on FGVC Aircraft

Re-runs the G37→G38 pipeline on Aircraft instead of Dogs.
Aircraft has Q1 baseline ~50%, providing correction headroom that Dogs (~87%) lacked.

Phase A: G37-style contingency analysis (chi-squared, Cramér's V, margin quartiles)
Phase B: G38-style gated correction (linear, MLP, multiplicative)

Kill conditions:
  Phase A: ≥20% Q1 queries with significant asymmetry (p<0.01)
  Phase B: gated correction improves Q1 accuracy by ≥1pp

Usage:
    PYTHONPATH=. python benchmarks/g39_negative_evidence_aircraft.py
"""
from __future__ import annotations

import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from fast_associative_memory import FastAssociativeMemory
from nstp import NSTPController
from benchmarks.g30_adapter_sweep import load_domain_features as g30_load_domain_features
from benchmarks.g31_retrain_adapters import _load_g30_domain_adapter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DOMAIN = "aircraft"
P_THRESHOLD = 0.01
MARGIN_QUARTILE = 0.25
SEED = 42

STAGE_NAMES = {
    0: "rerank_drop",
    1: "csls_survivor",
    2: "nstp_kill",
    3: "floor_kill",
    4: "final_survivor",
}


def set_seeds(seed: int = SEED):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Shared: build memory and collect raw traces
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_memory_and_traces(
    train_feats, train_labels, test_feats, test_labels,
    adapter, device, max_traces=100_000,
) -> Tuple[List[dict], dict, FastAssociativeMemory]:
    """Build FAM, collect per-query traces AND G38-style feature vectors."""
    num_classes = int(train_labels.max().item()) + 1
    nstp = NSTPController(sibling_threshold=0.85, depth_epsilon=0.10)

    mem = FastAssociativeMemory(
        input_dim=int(train_feats.shape[1]),
        value_dim=num_classes,
        core_entries=50000,
        core_vigilance=0.85,
        hebb_lr=0.1, key_lr=0.05,
        inference_k=25, inference_temp=0.05,
        use_lfu=True,
        adapter=adapter, nstp=nstp,
    ).to(device)

    # Ingest training data
    x_train = train_feats.to(device)
    y_train = train_labels.to(device)
    for i in range(0, len(x_train), 256):
        mem.learn_local(x_train[i:i + 256], y_train[i:i + 256])
    n_protos = int(mem.core_cam.occupied.sum().item())
    print(f"    Prototypes: {n_protos}")

    # Query set: combine test+train if needed
    query_feats = test_feats
    query_labels = test_labels
    if len(test_feats) < max_traces and len(train_feats) > len(test_feats):
        query_feats = torch.cat([test_feats, train_feats], dim=0)[:max_traces]
        query_labels = torch.cat([test_labels, train_labels], dim=0)[:max_traces]

    core = mem.core_cam
    occ_idx = core.occupied.nonzero(as_tuple=True)[0]
    proto_classes = core.values[occ_idx].float().argmax(dim=-1)

    x_q = query_feats.to(device)
    y_q = query_labels.to(device)

    traces = []
    all_softmax, all_neg_hist, all_neg_sim = [], [], []
    all_labels, all_margins = [], []

    for i in range(0, min(len(x_q), max_traces), 256):
        batch_x = x_q[i:i + 256]
        batch_y = y_q[i:i + 256]
        B = batch_x.size(0)

        proj = mem._project(batch_x)
        outputs, tr = core(proj, nstp=nstp, trace=True)

        q_norm = F.normalize(proj.float(), dim=-1)
        proto_keys = core._keys_norm[occ_idx]
        sims = q_norm @ proto_keys.T
        pred_classes = outputs.argmax(dim=1)

        for b in range(B):
            true_class = batch_y[b].item()
            pred_class = pred_classes[b].item()

            row_sims = sims[b]
            best_sim = row_sims.max().item()
            same_class_mask = proto_classes == pred_class
            other_sims = row_sims.masked_fill(same_class_mask, -1e9)
            second_best = other_sims.max().item() if other_sims.max().item() > -1e8 else best_sim
            margin = best_sim - second_best

            # --- G37-style trace record ---
            final_slots = tr.final_slots[b]
            rejected_slots_raw = tr.rejected_slots[b]
            rejection_stages = tr.rejection_stage[b]

            accepted_classes = []
            for s in final_slots.tolist():
                if 0 <= s < core.max_entries and core.occupied[s]:
                    accepted_classes.append(core.values[s].float().argmax().item())

            rejected_classes = []
            rejected_stages_list = []
            for j, s in enumerate(rejected_slots_raw.tolist()):
                if 0 <= s < core.max_entries and core.occupied[s]:
                    rejected_classes.append(core.values[s].float().argmax().item())
                    rejected_stages_list.append(
                        rejection_stages[j].item() if j < len(rejection_stages) else -1
                    )

            traces.append({
                "true_class": true_class,
                "pred_class": pred_class,
                "correct": true_class == pred_class,
                "margin": margin,
                "best_sim": best_sim,
                "accepted_classes": accepted_classes,
                "rejected_classes": rejected_classes,
                "rejected_stages": rejected_stages_list,
            })

            # --- G38-style feature vectors ---
            sm_vote = F.softmax(outputs[b], dim=-1)

            rej_slots = tr.rejected_slots[b]
            valid_rej = (rej_slots >= 0) & (rej_slots < core.max_entries) & core.occupied[rej_slots]
            valid_slots = rej_slots[valid_rej]

            hist = torch.zeros(num_classes, device=device)
            mean_sims_vec = torch.zeros(num_classes, device=device)

            if len(valid_slots) > 0:
                rej_cls = core.values[valid_slots].float().argmax(dim=-1)
                rej_keys = core._keys_norm[valid_slots]
                rej_cos = q_norm[b] @ rej_keys.T
                hist.scatter_add_(0, rej_cls, torch.ones_like(rej_cls, dtype=torch.float32))
                sum_sims = torch.zeros(num_classes, device=device)
                sum_sims.scatter_add_(0, rej_cls, rej_cos)
                count_sims = torch.zeros(num_classes, device=device)
                count_sims.scatter_add_(0, rej_cls, torch.ones_like(rej_cls, dtype=torch.float32))
                mean_sims_vec = torch.where(count_sims > 0, sum_sims / count_sims, torch.zeros_like(sum_sims))

            if hist.sum() > 0:
                hist = hist / hist.sum()

            all_softmax.append(sm_vote.cpu())
            all_neg_hist.append(hist.cpu())
            all_neg_sim.append(mean_sims_vec.cpu())
            all_labels.append(true_class)
            all_margins.append(margin)

    g38_dataset = {
        "softmax": torch.stack(all_softmax),
        "neg_hist": torch.stack(all_neg_hist),
        "neg_sim": torch.stack(all_neg_sim),
        "labels": torch.tensor(all_labels, dtype=torch.long),
        "margins": torch.tensor(all_margins, dtype=torch.float32),
        "num_classes": num_classes,
    }

    return traces, g38_dataset, mem


# ---------------------------------------------------------------------------
# Phase A: G37-style analysis
# ---------------------------------------------------------------------------

def compute_cramers_v(table: np.ndarray) -> float:
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.sum()
    r, k = table.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1)))) if n > 0 else 0.0


def phase_a_analysis(traces: List[dict], num_classes: int) -> dict:
    """Run G37-style statistical tests across all quartiles."""
    margins = np.array([t["margin"] for t in traces])
    sorted_idx = np.argsort(margins)
    q_size = len(traces) // 4

    quartile_results = {}
    for qi, q_name in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        start = qi * q_size
        end = (qi + 1) * q_size if qi < 3 else len(traces)
        q_traces = [traces[sorted_idx[j]] for j in range(start, end)]

        q_acc = 100.0 * sum(t["correct"] for t in q_traces) / max(len(q_traces), 1)

        # Test A: per-query chi-squared on rejected class distribution
        sig_count = 0
        modal_neq_true = 0
        cramers_vs = []

        for t in q_traces:
            rej = t["rejected_classes"]
            acc = t["accepted_classes"]
            if len(rej) < 5 or len(acc) < 2:
                continue

            all_cls = sorted(set(rej) | set(acc))
            if len(all_cls) < 2:
                continue

            acc_counts = Counter(acc)
            rej_counts = Counter(rej)
            acc_row = [acc_counts.get(c, 0) for c in all_cls]
            rej_row = [rej_counts.get(c, 0) for c in all_cls]
            table = np.array([acc_row, rej_row])

            nonzero = table.sum(axis=0) > 0
            table = table[:, nonzero]
            if table.shape[1] < 2 or table.sum() < 10:
                continue

            chi2, p, dof, _ = stats.chi2_contingency(table)
            if p < P_THRESHOLD:
                sig_count += 1
            cramers_vs.append(compute_cramers_v(table))

            # Test B: modal rejected class
            if rej:
                modal_rej = Counter(rej).most_common(1)[0][0]
                if modal_rej != t["true_class"]:
                    modal_neq_true += 1

        n_tested = len([t for t in q_traces if len(t["rejected_classes"]) >= 5 and len(t["accepted_classes"]) >= 2])
        test_a_pct = 100.0 * sig_count / max(n_tested, 1)
        test_b_pct = 100.0 * modal_neq_true / max(n_tested, 1)
        mean_v = float(np.mean(cramers_vs)) if cramers_vs else 0.0

        quartile_results[q_name] = {
            "n_traces": len(q_traces),
            "n_tested": n_tested,
            "accuracy": q_acc,
            "test_a_pct": test_a_pct,
            "test_a_sig": sig_count,
            "test_b_pct": test_b_pct,
            "test_b_count": modal_neq_true,
            "cramers_v": mean_v,
            "margin_range": (float(margins[sorted_idx[start]]),
                             float(margins[sorted_idx[min(end - 1, len(traces) - 1)]])),
        }

    return quartile_results


def print_phase_a(qr: dict) -> bool:
    """Print Phase A results. Returns True if proceed threshold met."""
    print("\n=== PHASE A: G37-style Analysis ===")
    print(f"Domain: FGVC Aircraft (100 classes)")

    total_rerank = sum(
        len(t.get("rejected_classes", [])) for q in qr.values() for t in [q]
    )

    print(f"\n  {'Quartile':>8} {'N':>6} {'Tested':>6} {'Acc':>7} "
          f"{'TestA%':>7} {'TestB%':>7} {'V':>6} {'Margin':>16}")
    print(f"  {'-' * 8} {'-' * 6} {'-' * 6} {'-' * 7} "
          f"{'-' * 7} {'-' * 7} {'-' * 6} {'-' * 16}")

    for q_name in ["Q1", "Q2", "Q3", "Q4"]:
        q = qr[q_name]
        mr = q["margin_range"]
        print(f"  {q_name:>8} {q['n_traces']:>6} {q['n_tested']:>6} "
              f"{q['accuracy']:>6.2f}% {q['test_a_pct']:>6.1f}% "
              f"{q['test_b_pct']:>6.1f}% {q['cramers_v']:>5.3f} "
              f"[{mr[0]:.4f},{mr[1]:.4f}]")

    q1 = qr["Q1"]
    proceed = q1["test_a_pct"] >= 20.0
    tag = "PROCEED" if proceed else "STOP"
    print(f"\n  Q1 asymmetry rate: {q1['test_a_pct']:.1f}% (threshold: 20%)")
    print(f"  Phase A VERDICT: {tag}")
    return proceed


def print_qualitative_examples(traces: List[dict], n: int = 5):
    """Print n qualitative examples from Q1 with strongest asymmetry."""
    margins = [t["margin"] for t in traces]
    threshold = np.quantile(margins, MARGIN_QUARTILE)
    q1 = [t for t in traces if t["margin"] <= threshold and len(t["rejected_classes"]) >= 5]

    # Sort by number of rejected candidates (most informative)
    q1.sort(key=lambda t: len(t["rejected_classes"]), reverse=True)

    print(f"\n  --- {n} Qualitative Examples (Q1, most rejected candidates) ---")
    for i, t in enumerate(q1[:n]):
        rej_counter = Counter(t["rejected_classes"])
        acc_counter = Counter(t["accepted_classes"])
        modal_rej = rej_counter.most_common(1)[0] if rej_counter else (None, 0)
        modal_acc = acc_counter.most_common(1)[0] if acc_counter else (None, 0)

        corrective = (t["pred_class"] != t["true_class"] and
                      modal_rej[0] is not None and
                      modal_rej[0] == t["pred_class"])

        tag = "CORRECTIVE" if corrective else "CONFIRMATORY"
        mark = ">>>" if corrective else "   "

        print(f"\n  {mark} Example {i + 1}: true={t['true_class']} pred={t['pred_class']} "
              f"correct={t['correct']} margin={t['margin']:.4f} [{tag}]")
        print(f"      Accepted: {acc_counter.most_common(3)}")
        print(f"      Rejected: {rej_counter.most_common(3)} (total {len(t['rejected_classes'])})")
        if corrective:
            print(f"      ^^ Rejected class {modal_rej[0]} matches wrong prediction — "
                  f"incorporating would suppress the error")


# ---------------------------------------------------------------------------
# Phase B: G38-style gated correction
# ---------------------------------------------------------------------------

class LinearGating(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Linear(num_classes * 3, num_classes)
    def forward(self, x):
        return self.fc(x)


class MLPGating(nn.Module):
    def __init__(self, num_classes, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, x):
        return self.net(x)


class MultiplicativeGating(nn.Module):
    """softmax_vote * (1 - beta * neg_hist)"""
    def __init__(self, num_classes):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(0.1))
    def forward(self, x):
        V = x.size(1) // 3
        sm = x[:, :V]
        neg = x[:, V:2*V]
        corrected = sm * (1 - self.beta * neg)
        return corrected


def train_model(model, loader, epochs=50, lr=1e-3, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
    return model


@torch.no_grad()
def eval_model(model, loader, device="cpu"):
    model.to(device).eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def phase_b(dataset: dict, device: torch.device) -> dict:
    """Train gated correction models on Q1 subset."""
    nc = dataset["num_classes"]
    n_total = len(dataset["labels"])
    idx_sorted = torch.argsort(dataset["margins"])
    q1_size = n_total // 4

    # Q1 features
    q1_idx = idx_sorted[:q1_size]
    X_q1 = torch.cat([
        dataset["softmax"][q1_idx],
        dataset["neg_hist"][q1_idx],
        dataset["neg_sim"][q1_idx],
    ], dim=1)
    y_q1 = dataset["labels"][q1_idx]

    # Full-set features (for full eval)
    X_all = torch.cat([dataset["softmax"], dataset["neg_hist"], dataset["neg_sim"]], dim=1)
    y_all = dataset["labels"]

    # Q1 train/val split
    perm = torch.randperm(len(y_q1))
    split = int(0.8 * len(y_q1))
    train_idx, val_idx = perm[:split], perm[split:]

    X_train, y_train = X_q1[train_idx], y_q1[train_idx]
    X_val, y_val = X_q1[val_idx], y_q1[val_idx]

    # Uncorrected baseline
    baseline_preds = dataset["softmax"][q1_idx][val_idx].argmax(1)
    baseline_acc = 100.0 * (baseline_preds == y_val).float().mean().item()

    # Full uncorrected
    full_baseline_preds = dataset["softmax"].argmax(1)
    full_baseline_acc = 100.0 * (full_baseline_preds == y_all).float().mean().item()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256)
    full_loader = DataLoader(TensorDataset(X_all, y_all), batch_size=256)

    results = {"baseline_q1": baseline_acc, "baseline_full": full_baseline_acc}

    for name, model in [
        ("linear", LinearGating(nc)),
        ("mlp", MLPGating(nc, hidden_dim=128)),
        ("multiplicative", MultiplicativeGating(nc)),
    ]:
        torch.manual_seed(SEED)
        train_model(model, train_loader, epochs=50, lr=1e-3, device=device)
        q1_acc = eval_model(model, val_loader, device=device)
        full_acc = eval_model(model, full_loader, device=device)
        results[name] = {
            "q1_acc": q1_acc,
            "q1_delta": q1_acc - baseline_acc,
            "full_acc": full_acc,
            "full_delta": full_acc - full_baseline_acc,
        }

    return results


def print_phase_b(results: dict) -> bool:
    print("\n=== PHASE B: G38-style Gated Correction ===")
    print(f"\n  Aircraft Q1 baseline: {results['baseline_q1']:.2f}%")
    print(f"  Aircraft full baseline: {results['baseline_full']:.2f}%")

    best_q1_delta = -999
    for name in ["linear", "mlp", "multiplicative"]:
        r = results[name]
        label = {"linear": "Linear gate (additive)", "mlp": "MLP gate",
                 "multiplicative": "Multiplicative gate"}[name]
        print(f"  {label:30s}: Q1 {r['q1_acc']:.2f}% ({r['q1_delta']:+.2f}pp)  "
              f"Full {r['full_acc']:.2f}% ({r['full_delta']:+.2f}pp)")
        if r["q1_delta"] > best_q1_delta:
            best_q1_delta = r["q1_delta"]

    passed = best_q1_delta >= 1.0
    verdict = "PASS" if passed else "KILL"
    if passed:
        reason = "Gated correction improves Q1 accuracy by ≥1pp"
    else:
        reason = "Negative evidence is confirmatory, not corrective"
    print(f"\n  Best Q1 delta: {best_q1_delta:+.2f}pp (threshold: +1.0pp)")
    print(f"  Phase B VERDICT: {verdict} — {reason}")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path("data")

    print("=" * 60)
    print("  G39 — Negative Evidence on FGVC Aircraft")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Domain: {DOMAIN}")

    set_seeds()

    # Load data + adapter
    print("\n  Loading features and adapter...")
    tr_f, tr_l, te_f, te_l, n_classes = g30_load_domain_features(DOMAIN, data_root)
    adapter = _load_g30_domain_adapter(DOMAIN, input_dim=tr_f.shape[1], device=device)
    print(f"  Train: {len(tr_f)}, Test: {len(te_f)}, Classes: {n_classes}")

    # Build memory and collect both trace formats in one pass
    print("\n  Building memory and collecting traces...")
    traces, g38_data, mem = build_memory_and_traces(
        tr_f, tr_l, te_f, te_l, adapter, device,
    )
    print(f"  Collected {len(traces)} traces")

    total_rejected = sum(len(t["rejected_classes"]) for t in traces)
    print(f"  Total rerank_drop: {total_rejected} "
          f"(mean {total_rejected / max(len(traces), 1):.1f}/query)")

    del mem
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Phase A
    qr = phase_a_analysis(traces, n_classes)
    proceed = print_phase_a(qr)
    print_qualitative_examples(traces)

    # Phase B (run regardless for completeness, but gate verdict on Phase A)
    print("\n" + "-" * 60)
    b_results = phase_b(g38_data, device)
    b_passed = print_phase_b(b_results)

    # Final verdict
    print("\n" + "=" * 60)
    if not proceed:
        print("  FINAL VERDICT: KILL — Phase A signal too weak (<20% Q1 asymmetry)")
    elif not b_passed:
        print("  FINAL VERDICT: KILL — Phase A signal exists but Phase B correction <1pp")
    else:
        print("  FINAL VERDICT: PASS — Negative evidence provides corrective signal on Aircraft")
    print("=" * 60)


if __name__ == "__main__":
    main()
