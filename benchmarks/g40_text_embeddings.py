#!/usr/bin/env python3
"""
G-40 — FAM on Text Embeddings: 20 Newsgroups Existence Proof

Tests whether the FAM pipeline (prototype store → softmax vote) works on
sentence-transformer embeddings (all-MiniLM-L6-v2, 384-d).

Kill condition: FAM accuracy < best kNN accuracy on the same embeddings.
Pass condition: FAM accuracy >= kNN AND condensation ratio < 0.90.

Usage:
    PYTHONPATH=. .venv/bin/python benchmarks/g40_text_embeddings.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from fast_associative_memory import FastAssociativeMemory

SEED = 42
DATA_DIR = Path("data/20newsgroups")
OUT_DIR = Path("results/g40_text_embeddings")

CATEGORIES = [
    "alt.atheism", "comp.graphics", "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.windows.x",
    "misc.forsale", "rec.autos", "rec.motorcycles", "rec.sport.baseball",
    "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med",
    "sci.space", "soc.religion.christian", "talk.politics.guns",
    "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc",
]


def set_seeds():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def load_embeddings():
    train = torch.load(DATA_DIR / "20newsgroups_train.pt",
                       map_location="cpu", weights_only=True)
    test = torch.load(DATA_DIR / "20newsgroups_test.pt",
                      map_location="cpu", weights_only=True)
    return (train["embeds"].float(), train["labels"].long(),
            test["embeds"].float(), test["labels"].long())


# ---------------------------------------------------------------------------
# Phase 1: kNN baselines
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_knn(train_feats, train_labels, test_feats, test_labels, k_values,
            device):
    """Flat kNN with majority vote. Returns {k: accuracy%}."""
    print("\n  Phase 1: kNN baselines")
    x_tr = F.normalize(train_feats.to(device), dim=-1)
    x_te = F.normalize(test_feats.to(device), dim=-1)
    y_tr = train_labels.to(device)
    n_test = len(x_te)
    results = {}

    # Batch similarity to avoid OOM
    batch_sz = 256
    for k in k_values:
        correct = 0
        for i in range(0, n_test, batch_sz):
            batch = x_te[i:i + batch_sz]
            sims = batch @ x_tr.T                         # (B, N_train)
            topk_idx = sims.topk(k, dim=-1).indices       # (B, k)
            topk_labels = y_tr[topk_idx]                  # (B, k)
            # Majority vote
            for b in range(len(batch)):
                counts = topk_labels[b].bincount(minlength=20)
                pred = counts.argmax().item()
                if pred == test_labels[i + b].item():
                    correct += 1
        acc = 100.0 * correct / n_test
        results[k] = acc
        print(f"    kNN (K={k:>3}): {acc:.2f}%")

    return results


# ---------------------------------------------------------------------------
# Phase 2: FAM baseline + sweep
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_fam(train_feats, train_labels, test_feats, test_labels,
            vigilance, inference_k, inference_temp, device):
    """Single FAM configuration. Returns (accuracy%, CR, n_prototypes)."""
    n_classes = 20
    dim = train_feats.shape[1]
    set_seeds()

    mem = FastAssociativeMemory(
        input_dim=dim,
        value_dim=n_classes,
        core_entries=50000,
        core_vigilance=vigilance,
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=inference_k,
        inference_temp=inference_temp,
        use_lfu=True,
    ).to(device)

    x_tr = train_feats.to(device)
    y_tr = train_labels.to(device)
    for i in range(0, len(x_tr), 256):
        mem.learn_local(x_tr[i:i + 256], y_tr[i:i + 256])

    n_proto = int(mem.core_cam.occupied.sum().item())
    cr = n_proto / len(train_feats)

    # Eval
    x_te = test_feats.to(device)
    correct = 0
    batch_sz = 256
    for i in range(0, len(x_te), batch_sz):
        out = mem.forward(x_te[i:i + batch_sz])
        preds = out.argmax(dim=-1)
        correct += (preds == test_labels[i:i + batch_sz].to(device)).sum().item()

    acc = 100.0 * correct / len(test_feats)
    del mem
    return acc, cr, n_proto


def run_fam_sweep(train_feats, train_labels, test_feats, test_labels, device):
    """Sweep FAM hyperparameters."""
    print("\n  Phase 2: FAM parameter sweep")

    vigilances = [0.70, 0.80, 0.85, 0.90, 0.95]
    inf_ks = [15, 25, 50]
    temps = [0.05, 0.10]

    results = []
    header = f"    {'v':>5} {'k':>4} {'temp':>5} {'Acc%':>7} {'CR':>6} {'Protos':>7}"
    print(header)
    print(f"    {'-' * 40}")

    for v in vigilances:
        for k in inf_ks:
            for t in temps:
                acc, cr, n_proto = run_fam(
                    train_feats, train_labels, test_feats, test_labels,
                    vigilance=v, inference_k=k, inference_temp=t, device=device)
                row = {
                    "vigilance": v, "inference_k": k, "inference_temp": t,
                    "accuracy": acc, "cr": cr, "n_prototypes": n_proto,
                }
                results.append(row)
                print(f"    {v:>5.2f} {k:>4} {t:>5.2f} {acc:>6.2f}% {cr:>5.3f} {n_proto:>7}")

    return results


# ---------------------------------------------------------------------------
# Phase 3: Per-class breakdown (best config)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_per_class_analysis(train_feats, train_labels, test_feats, test_labels,
                           vigilance, inference_k, inference_temp, device):
    """Per-class accuracy and prototype count for the best FAM config."""
    print("\n  Phase 3: Per-class analysis")
    n_classes = 20
    dim = train_feats.shape[1]
    set_seeds()

    mem = FastAssociativeMemory(
        input_dim=dim, value_dim=n_classes, core_entries=50000,
        core_vigilance=vigilance, hebb_lr=0.1, key_lr=0.05,
        inference_k=inference_k, inference_temp=inference_temp,
        use_lfu=True,
    ).to(device)

    x_tr = train_feats.to(device)
    y_tr = train_labels.to(device)
    for i in range(0, len(x_tr), 256):
        mem.learn_local(x_tr[i:i + 256], y_tr[i:i + 256])

    # Prototype count per class
    core = mem.core_cam
    occ = core.occupied.nonzero(as_tuple=True)[0]
    proto_classes = core.values[occ].float().argmax(dim=-1)

    # Eval per class
    x_te = test_feats.to(device)
    y_te = test_labels.to(device)
    all_preds = []
    for i in range(0, len(x_te), 256):
        out = mem.forward(x_te[i:i + 256])
        all_preds.append(out.argmax(dim=-1))
    all_preds = torch.cat(all_preds)

    # Confidence analysis
    print("\n  Phase 3b: Confidence calibration on text")
    all_confs = []
    all_corrects = []
    for i in range(0, len(x_te), 256):
        out, conf = mem.forward_with_confidence(x_te[i:i + 256])
        preds = out.argmax(dim=-1)
        corrects = (preds == y_te[i:i + 256])
        all_confs.append(conf.cpu())
        all_corrects.append(corrects.cpu())
    all_confs = torch.cat(all_confs).numpy()
    all_corrects = torch.cat(all_corrects).numpy()

    # AUROC (manual)
    pos = all_confs[all_corrects]
    neg = all_confs[~all_corrects]
    if len(pos) > 0 and len(neg) > 0:
        all_s = np.concatenate([pos, neg])
        all_l = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        order = np.argsort(-all_s)
        sorted_l = all_l[order]
        cum_pos = np.cumsum(sorted_l)
        auroc = float(np.sum((1 - sorted_l) * cum_pos) / (len(pos) * len(neg)))
    else:
        auroc = 0.5
    print(f"    Confidence AUROC: {auroc:.4f}")
    print(f"    Mean confidence (correct):   {pos.mean():.4f}")
    print(f"    Mean confidence (incorrect): {neg.mean():.4f}")

    per_class = []
    print(f"\n    {'Class':>3} {'Category':<35} {'Acc%':>7} {'Protos':>7} {'Train':>6}")
    print(f"    {'-' * 65}")
    for c in range(n_classes):
        c_mask = y_te == c
        c_correct = (all_preds[c_mask] == c).sum().item()
        c_total = c_mask.sum().item()
        c_acc = 100.0 * c_correct / c_total if c_total > 0 else 0.0

        c_protos = (proto_classes == c).sum().item()
        c_train = (train_labels == c).sum().item()

        per_class.append({
            "class_id": c,
            "category": CATEGORIES[c],
            "accuracy": c_acc,
            "n_prototypes": c_protos,
            "n_train": c_train,
        })
        print(f"    {c:>3} {CATEGORIES[c]:<35} {c_acc:>6.1f}% {c_protos:>7} {c_train:>6}")

    del mem
    return per_class, auroc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  G-40 — FAM on Text Embeddings: 20 Newsgroups")
    print(f"  Device: {device}")
    print("=" * 65)

    tr_f, tr_l, te_f, te_l = load_embeddings()
    print(f"  Train: {len(tr_f)}, Test: {len(te_f)}, Dim: {tr_f.shape[1]}")

    # Phase 1: kNN
    knn_results = run_knn(tr_f, tr_l, te_f, te_l,
                          k_values=[1, 5, 15, 25, 50], device=device)
    best_knn_k = max(knn_results, key=knn_results.get)
    best_knn_acc = knn_results[best_knn_k]
    print(f"\n  Best kNN: K={best_knn_k}, Accuracy={best_knn_acc:.2f}%")

    # Phase 2: FAM sweep
    fam_results = run_fam_sweep(tr_f, tr_l, te_f, te_l, device)

    # Find best FAM config
    best_fam = max(fam_results, key=lambda r: r["accuracy"])
    print(f"\n  Best FAM: v={best_fam['vigilance']}, k={best_fam['inference_k']}, "
          f"T={best_fam['inference_temp']}")
    print(f"    Accuracy: {best_fam['accuracy']:.2f}%, CR: {best_fam['cr']:.4f}, "
          f"Prototypes: {best_fam['n_prototypes']}")

    # Phase 3: Per-class analysis on best config
    per_class, conf_auroc = run_per_class_analysis(
        tr_f, tr_l, te_f, te_l,
        vigilance=best_fam["vigilance"],
        inference_k=best_fam["inference_k"],
        inference_temp=best_fam["inference_temp"],
        device=device,
    )

    # ---- Verdict ----
    print("\n" + "=" * 65)
    print("  G-40 VERDICT")
    print("=" * 65)
    delta = best_fam["accuracy"] - best_knn_acc
    print(f"  Best kNN:  {best_knn_acc:.2f}% (K={best_knn_k})")
    print(f"  Best FAM:  {best_fam['accuracy']:.2f}% "
          f"(v={best_fam['vigilance']}, k={best_fam['inference_k']}, T={best_fam['inference_temp']})")
    print(f"  Delta:     {delta:+.2f}pp")
    print(f"  CR:        {best_fam['cr']:.4f} ({best_fam['n_prototypes']} prototypes)")
    print(f"  Conf AUROC: {conf_auroc:.4f}")

    killed = best_fam["accuracy"] < best_knn_acc
    passed = not killed and best_fam["cr"] < 0.90

    if killed:
        verdict = "KILL — FAM accuracy below kNN baseline on text embeddings"
    elif passed:
        verdict = "PASS — FAM >= kNN with condensation (CR < 0.90)"
    else:
        verdict = f"INCONCLUSIVE — FAM >= kNN but CR = {best_fam['cr']:.4f} (no condensation)"

    print(f"\n  {verdict}")
    print("=" * 65)

    # ---- Save ----
    output = {
        "knn_results": {str(k): v for k, v in knn_results.items()},
        "best_knn": {"k": best_knn_k, "accuracy": best_knn_acc},
        "fam_sweep": fam_results,
        "best_fam": best_fam,
        "per_class": per_class,
        "confidence_auroc": conf_auroc,
        "verdict": verdict,
        "delta_vs_knn": delta,
    }
    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
