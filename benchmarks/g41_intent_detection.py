#!/usr/bin/env python3
"""
G-41 — FAM on Fine-Grained Intent Detection: Banking77 + CLINC-150

Tests FAM on many-class intent classification (77 and 151 classes) using
sentence-transformer embeddings (all-MiniLM-L6-v2, 384-d).

Builds on G-40 (20 Newsgroups existence proof, PASSED).

Kill condition: FAM accuracy < best kNN accuracy on either dataset.
Pass condition: FAM >= kNN on both datasets.

Usage:
    PYTHONPATH=. .venv/bin/python benchmarks/g41_intent_detection.py
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
OUT_DIR = Path("results/g41_intent_detection")

DATASETS = {
    "banking77": {
        "dir": Path("data/banking77"),
        "train": "banking77_train.pt",
        "test": "banking77_test.pt",
        "n_classes": 77,
    },
    "clinc150": {
        "dir": Path("data/clinc150"),
        "train": "clinc150_train.pt",
        "test": "clinc150_test.pt",
        "n_classes": 151,
    },
}


def set_seeds():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def load_embeddings(ds_info):
    train = torch.load(ds_info["dir"] / ds_info["train"],
                       map_location="cpu", weights_only=True)
    test = torch.load(ds_info["dir"] / ds_info["test"],
                      map_location="cpu", weights_only=True)
    return (train["embeds"].float(), train["labels"].long(),
            test["embeds"].float(), test["labels"].long())


# ---------------------------------------------------------------------------
# kNN baseline
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_knn(tr_f, tr_l, te_f, te_l, k_values, device):
    x_tr = F.normalize(tr_f.to(device), dim=-1)
    x_te = F.normalize(te_f.to(device), dim=-1)
    y_tr = tr_l.to(device)
    n_test = len(x_te)
    n_classes = int(tr_l.max().item()) + 1
    results = {}
    for k in k_values:
        correct = 0
        for i in range(0, n_test, 256):
            batch = x_te[i:i + 256]
            sims = batch @ x_tr.T
            topk_idx = sims.topk(k, dim=-1).indices
            topk_labels = y_tr[topk_idx]
            for b in range(len(batch)):
                counts = topk_labels[b].bincount(minlength=n_classes)
                pred = counts.argmax().item()
                if pred == te_l[i + b].item():
                    correct += 1
        acc = 100.0 * correct / n_test
        results[k] = acc
    return results


# ---------------------------------------------------------------------------
# FAM run
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_fam(tr_f, tr_l, te_f, te_l, n_classes, vigilance, inference_k,
            inference_temp, device):
    dim = tr_f.shape[1]
    set_seeds()
    mem = FastAssociativeMemory(
        input_dim=dim, value_dim=n_classes,
        core_entries=50000, core_vigilance=vigilance,
        hebb_lr=0.1, key_lr=0.05,
        inference_k=inference_k, inference_temp=inference_temp,
        use_lfu=True,
    ).to(device)

    x_tr = tr_f.to(device)
    y_tr = tr_l.to(device)
    for i in range(0, len(x_tr), 256):
        mem.learn_local(x_tr[i:i + 256], y_tr[i:i + 256])

    n_proto = int(mem.core_cam.occupied.sum().item())
    cr = n_proto / len(tr_f)

    # Eval
    x_te = te_f.to(device)
    correct = 0
    for i in range(0, len(x_te), 256):
        out = mem.forward(x_te[i:i + 256])
        preds = out.argmax(dim=-1)
        correct += (preds == te_l[i:i + 256].to(device)).sum().item()
    acc = 100.0 * correct / len(te_f)

    # Confidence
    all_confs, all_corrects = [], []
    for i in range(0, len(x_te), 256):
        out, conf = mem.forward_with_confidence(x_te[i:i + 256])
        preds = out.argmax(dim=-1)
        all_confs.append(conf.cpu())
        all_corrects.append((preds == te_l[i:i + 256].to(device)).cpu())
    all_confs = torch.cat(all_confs).numpy()
    all_corrects = torch.cat(all_corrects).numpy()

    # AUROC
    pos, neg = all_confs[all_corrects], all_confs[~all_corrects]
    if len(pos) > 0 and len(neg) > 0:
        all_s = np.concatenate([pos, neg])
        all_l = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        order = np.argsort(-all_s)
        sorted_l = all_l[order]
        cum_pos = np.cumsum(sorted_l)
        auroc = float(np.sum((1 - sorted_l) * cum_pos) / (len(pos) * len(neg)))
    else:
        auroc = 0.5

    # Per-class prototype counts
    core = mem.core_cam
    occ = core.occupied.nonzero(as_tuple=True)[0]
    proto_classes = core.values[occ].float().argmax(dim=-1).cpu()
    class_proto_counts = [int((proto_classes == c).sum().item()) for c in range(n_classes)]

    # Per-class accuracy
    all_preds = []
    for i in range(0, len(x_te), 256):
        out = mem.forward(x_te[i:i + 256])
        all_preds.append(out.argmax(dim=-1).cpu())
    all_preds = torch.cat(all_preds)

    per_class_acc = []
    for c in range(n_classes):
        mask = te_l == c
        if mask.sum() == 0:
            per_class_acc.append(0.0)
        else:
            per_class_acc.append(100.0 * (all_preds[mask] == c).float().mean().item())

    del mem
    return {
        "accuracy": acc, "cr": cr, "n_prototypes": n_proto,
        "conf_auroc": auroc,
        "conf_mean_correct": float(pos.mean()) if len(pos) > 0 else 0.0,
        "conf_mean_incorrect": float(neg.mean()) if len(neg) > 0 else 0.0,
        "class_proto_counts": class_proto_counts,
        "per_class_acc": per_class_acc,
    }


# ---------------------------------------------------------------------------
# Per-dataset benchmark
# ---------------------------------------------------------------------------

def run_dataset(name, ds_info, device):
    print(f"\n{'=' * 65}")
    print(f"  DATASET: {name.upper()}")
    print(f"{'=' * 65}")

    tr_f, tr_l, te_f, te_l = load_embeddings(ds_info)
    n_classes = ds_info["n_classes"]
    print(f"  Train: {len(tr_f)}, Test: {len(te_f)}, Classes: {n_classes}, Dim: {tr_f.shape[1]}")

    # kNN
    print("\n  kNN baselines:")
    knn = run_knn(tr_f, tr_l, te_f, te_l, [1, 5, 15, 25, 50], device)
    for k, acc in sorted(knn.items()):
        print(f"    K={k:>3}: {acc:.2f}%")
    best_knn_k = max(knn, key=knn.get)
    best_knn_acc = knn[best_knn_k]
    print(f"  Best kNN: K={best_knn_k}, {best_knn_acc:.2f}%")

    # FAM sweep (focused: G-40 showed T=0.10 > T=0.05 on text)
    print("\n  FAM parameter sweep:")
    vigilances = [0.70, 0.80, 0.85, 0.90, 0.95]
    inf_ks = [15, 25, 50]
    temps = [0.05, 0.10]

    print(f"    {'v':>5} {'k':>4} {'temp':>5} {'Acc%':>7} {'CR':>6} {'Protos':>7} "
          f"{'AUROC':>6} {'Delta':>7}")
    print(f"    {'-' * 55}")

    fam_results = []
    for v in vigilances:
        for inf_k in inf_ks:
            for t in temps:
                r = run_fam(tr_f, tr_l, te_f, te_l, n_classes,
                            v, inf_k, t, device)
                r.update({"vigilance": v, "inference_k": inf_k, "inference_temp": t})
                fam_results.append(r)
                delta = r["accuracy"] - best_knn_acc
                print(f"    {v:>5.2f} {inf_k:>4} {t:>5.2f} {r['accuracy']:>6.2f}% "
                      f"{r['cr']:>5.3f} {r['n_prototypes']:>7} "
                      f"{r['conf_auroc']:>5.3f} {delta:>+6.2f}pp")

    best_fam = max(fam_results, key=lambda r: r["accuracy"])
    print(f"\n  Best FAM: v={best_fam['vigilance']}, k={best_fam['inference_k']}, "
          f"T={best_fam['inference_temp']}")
    print(f"    Accuracy: {best_fam['accuracy']:.2f}%, CR: {best_fam['cr']:.4f}, "
          f"Protos: {best_fam['n_prototypes']}")
    print(f"    Confidence AUROC: {best_fam['conf_auroc']:.4f}")
    print(f"    Mean conf (correct/incorrect): "
          f"{best_fam['conf_mean_correct']:.4f} / {best_fam['conf_mean_incorrect']:.4f}")

    # Per-class analysis on best config (top/bottom 5)
    pca = best_fam["per_class_acc"]
    sorted_cls = sorted(range(n_classes), key=lambda c: pca[c])
    print(f"\n  Hardest 5 classes (best FAM config):")
    for c in sorted_cls[:5]:
        print(f"    Class {c:>3}: {pca[c]:>5.1f}% ({best_fam['class_proto_counts'][c]} protos)")
    print(f"  Easiest 5 classes:")
    for c in sorted_cls[-5:]:
        print(f"    Class {c:>3}: {pca[c]:>5.1f}% ({best_fam['class_proto_counts'][c]} protos)")

    # Verdict
    delta = best_fam["accuracy"] - best_knn_acc
    killed = best_fam["accuracy"] < best_knn_acc
    return {
        "dataset": name,
        "n_classes": n_classes,
        "n_train": len(tr_f),
        "n_test": len(te_f),
        "knn_results": {str(k): v for k, v in knn.items()},
        "best_knn": {"k": best_knn_k, "accuracy": best_knn_acc},
        "fam_sweep": [{k: v for k, v in r.items()
                       if k not in ("class_proto_counts", "per_class_acc")}
                      for r in fam_results],
        "best_fam": {k: v for k, v in best_fam.items()
                     if k not in ("class_proto_counts", "per_class_acc")},
        "delta_vs_knn": delta,
        "killed": killed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  G-41 — FAM on Fine-Grained Intent Detection")
    print(f"  Device: {device}")
    print("=" * 65)

    all_results = []
    for name, ds_info in DATASETS.items():
        result = run_dataset(name, ds_info, device)
        all_results.append(result)

    # Summary
    print("\n" + "=" * 65)
    print("  G-41 SUMMARY")
    print("=" * 65)
    any_killed = False
    for r in all_results:
        status = "KILL" if r["killed"] else "PASS"
        if r["killed"]:
            any_killed = True
        print(f"  {r['dataset']:<12}: kNN={r['best_knn']['accuracy']:.2f}%, "
              f"FAM={r['best_fam']['accuracy']:.2f}%, "
              f"delta={r['delta_vs_knn']:+.2f}pp, "
              f"CR={r['best_fam']['cr']:.4f}, "
              f"AUROC={r['best_fam']['conf_auroc']:.4f} → {status}")

    if any_killed:
        verdict = "KILL — FAM below kNN on at least one dataset"
    else:
        verdict = "PASS — FAM >= kNN on both fine-grained intent datasets"
    print(f"\n  VERDICT: {verdict}")
    print("=" * 65)

    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump({"results": all_results, "verdict": verdict}, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
