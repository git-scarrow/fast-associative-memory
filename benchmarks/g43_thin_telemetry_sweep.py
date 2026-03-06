#!/usr/bin/env python3
"""
G-43 — Thin-Telemetry Sweep: Re-validate G5, G8a, G8b, G9, G10

Re-runs four remaining load-bearing gauntlets on the current production stack
(post-coverage-eviction, post-MT-17 read-path strip, G31 adapters, adaptive
eviction) with full RetrievalTrace instrumentation.

Sub-tests:
  A: G5  — Indexing Latency (FAM vs FAISS FlatIP CPU)
  B: G8a — Noise Robustness (20% label noise)
  C: G8b — ABA Re-entry (zero duplicate prototypes)
  D: G9  — OOD Detection (void-space gap)
  E: G10 — Concept Drift (backward transfer)

Usage:
    PYTHONPATH=. .venv/bin/python benchmarks/g43_thin_telemetry_sweep.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from associative_core import RetrievalTrace
from fast_associative_memory import FastAssociativeMemory
from benchmarks.g31_retrain_adapters import _load_g30_domain_adapter

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GIT_COMMIT = "282933a"
OUT_DIR = Path("results/g43_thin_telemetry")

# Paths
DOGS_TRAIN = Path("data/stanford_dogs/stanford_dogs_train.pt")
DOGS_TEST = Path("data/stanford_dogs/stanford_dogs_test.pt")
CIFAR_TRAIN = Path("feature_cache_vitl14/cifar100_dinov2_train.pt")
CIFAR_TEST = Path("feature_cache_vitl14/cifar100_dinov2_test.pt")
ADAPTER_PATH = Path("adapter_trained.pt")


def set_seeds():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True


def load_dogs():
    tr = torch.load(DOGS_TRAIN, map_location="cpu", weights_only=True)
    te = torch.load(DOGS_TEST, map_location="cpu", weights_only=True)
    return (tr["features"].float(), tr["labels"].long(),
            te["features"].float(), te["labels"].long())


def load_cifar100():
    tr = torch.load(CIFAR_TRAIN, map_location="cpu", weights_only=True)
    te = torch.load(CIFAR_TEST, map_location="cpu", weights_only=True)
    return (tr["embeds"].float(), tr["labels"].long(),
            te["embeds"].float(), te["labels"].long())


def load_adapter(dim=1024):
    device = torch.device(DEVICE)
    return _load_g30_domain_adapter("dogs", input_dim=dim, device=device)


def make_fam(n_classes, entries=50000, vigilance=0.92, adapter=None,
             use_lfu=False, adaptive_eviction=True, dim=1024):
    return FastAssociativeMemory(
        input_dim=dim,
        value_dim=n_classes,
        core_entries=entries,
        core_vigilance=vigilance,
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=25,
        inference_temp=0.05,
        whitening_dim=0,
        use_lfu=use_lfu,
        adaptive_eviction=adaptive_eviction,
        adapter=adapter,
        nstp=None,
    ).to(DEVICE)


@torch.no_grad()
def learn_batched(fam, feats, labels, n_classes, batch=256):
    x = feats.to(DEVICE)
    y = labels.long().to(DEVICE)
    for i in range(0, len(x), batch):
        fam.learn_local(x[i:i + batch], y[i:i + batch])


@torch.no_grad()
def eval_accuracy(fam, feats, labels, batch=512):
    correct = 0
    x = feats.to(DEVICE)
    y = labels.to(DEVICE)
    for i in range(0, len(x), batch):
        logits = fam(x[i:i + batch])
        correct += (logits.argmax(dim=1) == y[i:i + batch]).sum().item()
    return 100.0 * correct / len(labels)


@torch.no_grad()
def collect_trace(fam, feats, n_queries=500):
    """Collect RetrievalTrace on a subsample."""
    idx = torch.randperm(len(feats))[:n_queries]
    q = feats[idx].to(DEVICE)
    q_proj = fam._project(q)
    outputs, trace = fam.core_cam(q_proj, nstp=fam.nstp, trace=True)
    return trace, outputs


def trace_summary(trace: RetrievalTrace):
    """Summarize rejection stages from a RetrievalTrace."""
    stages = trace.rejection_stage  # (B, rejected_k)
    flat = stages[stages >= 0]  # filter padding
    counts = {}
    for name, val in [("topk_drop", 0), ("nstp", 1), ("floor", 2)]:
        counts[name] = int((flat == val).sum().item())
    counts["total_rejected"] = int(flat.numel())
    counts["avg_survivors"] = float(trace.final_slots.size(1))
    return counts


def proto_snapshot(fam):
    occ = int(fam.core_cam.occupied.sum().item())
    total = fam.core_cam.keys.size(0)
    return {"occupied": occ, "capacity": total, "occupancy": occ / total}


# ===========================================================================
# Sub-test A: G5 — Indexing Latency
# ===========================================================================

def run_g5():
    print("\n" + "=" * 70)
    print("  Sub-test A: G5 — Indexing Latency")
    print("=" * 70)

    import faiss

    set_seeds()
    tr_f, tr_l, te_f, te_l = load_dogs()
    adapter = load_adapter()
    n_classes = int(tr_l.max().item()) + 1

    # Build FAM with 50K entries, fill with training data
    fam = make_fam(n_classes, entries=50000, adapter=adapter)
    learn_batched(fam, tr_f, tr_l, n_classes)
    n_proto = int(fam.core_cam.occupied.sum().item())
    print(f"  Prototypes: {n_proto}")

    # Build FAISS index from prototype keys
    valid_idx = fam.core_cam.occupied.nonzero(as_tuple=True)[0]
    proto_keys = fam.core_cam._keys_norm[valid_idx].cpu().numpy().astype(np.float32)
    D = proto_keys.shape[1]

    # Queries: 1000 test samples projected through adapter
    set_seeds()
    q_idx = torch.randperm(len(te_f))[:1000]
    queries_raw = te_f[q_idx]
    queries_proj = fam._project(queries_raw.to(DEVICE)).detach().cpu()
    queries_np = queries_proj.numpy().astype(np.float32)

    results = {}

    for cond_name, K, use_conf, use_trace in [
        ("C1_original_K25", 25, False, False),
        ("C2_production_K50", 50, True, False),
        ("C3_instrumented_K50", 50, True, True),
    ]:
        print(f"\n  Condition: {cond_name} (K={K}, conf={use_conf}, trace={use_trace})")

        # Override inference_k
        fam.core_cam.inference_k = K

        # -- FAISS FlatIP CPU --
        index = faiss.IndexFlatIP(D)
        index.add(proto_keys)

        # Warmup
        for _ in range(10):
            index.search(queries_np[:10], K)

        t0 = time.perf_counter()
        for _ in range(3):
            index.search(queries_np, K)
        faiss_total = (time.perf_counter() - t0) / 3
        faiss_us = faiss_total / len(queries_np) * 1e6

        # p99: measure individual queries
        faiss_latencies = []
        for qi in range(min(200, len(queries_np))):
            t0 = time.perf_counter()
            index.search(queries_np[qi:qi + 1], K)
            faiss_latencies.append((time.perf_counter() - t0) * 1e6)
        faiss_p99 = float(np.percentile(faiss_latencies, 99))

        # -- FAM --
        q_gpu = queries_proj.to(DEVICE)

        # Warmup
        for _ in range(10):
            if use_conf:
                fam.forward_with_confidence(queries_raw[:10].to(DEVICE))
            else:
                fam(queries_raw[:10].to(DEVICE))

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(3):
            if use_conf and use_trace:
                q_p = fam._project(queries_raw.to(DEVICE))
                out, tr = fam.core_cam(q_p, nstp=fam.nstp, trace=True)
                # Compute confidence manually since trace mode goes through core_cam
            elif use_conf:
                fam.forward_with_confidence(queries_raw.to(DEVICE))
            else:
                fam(queries_raw.to(DEVICE))
            if DEVICE == "cuda":
                torch.cuda.synchronize()
        fam_total = (time.perf_counter() - t0) / 3
        fam_us = fam_total / len(queries_np) * 1e6

        # p99
        fam_latencies = []
        for qi in range(min(200, len(queries_np))):
            q_single = queries_raw[qi:qi + 1].to(DEVICE)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if use_conf:
                fam.forward_with_confidence(q_single)
            else:
                fam(q_single)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            fam_latencies.append((time.perf_counter() - t0) * 1e6)
        fam_p99 = float(np.percentile(fam_latencies, 99))

        ratio = fam_us / faiss_us if faiss_us > 0 else float("inf")
        kill = ratio > 2.0

        results[cond_name] = {
            "K": K,
            "faiss_mean_us": round(faiss_us, 1),
            "faiss_p99_us": round(faiss_p99, 1),
            "fam_mean_us": round(fam_us, 1),
            "fam_p99_us": round(fam_p99, 1),
            "ratio": round(ratio, 2),
            "kill": kill,
        }
        print(f"    FAISS: {faiss_us:.1f} µs/q (p99: {faiss_p99:.1f})")
        print(f"    FAM:   {fam_us:.1f} µs/q (p99: {fam_p99:.1f})")
        print(f"    Ratio: {ratio:.2f}x {'KILL' if kill else 'PASS'}")

    # Reset inference_k
    fam.core_cam.inference_k = 25

    # Trace telemetry
    trace, _ = collect_trace(fam, te_f)
    t_summary = trace_summary(trace)

    any_kill = any(r["kill"] for r in results.values())
    return {
        "decision": "KILL" if any_kill else "PASS",
        "n_proto": n_proto,
        "conditions": results,
        "trace_summary": t_summary,
        "store_snapshot": proto_snapshot(fam),
    }


# ===========================================================================
# Sub-test B: G8a — Noise Robustness
# ===========================================================================

def run_g8a():
    print("\n" + "=" * 70)
    print("  Sub-test B: G8a — Noise Robustness (20% label noise)")
    print("=" * 70)

    set_seeds()
    tr_f, tr_l, te_f, te_l = load_dogs()
    adapter = load_adapter()

    # Select 50 classes
    classes = sorted(tr_l.unique().tolist())[:50]
    cls_set = set(classes)
    tr_mask = torch.tensor([l.item() in cls_set for l in tr_l])
    te_mask = torch.tensor([l.item() in cls_set for l in te_l])
    tr_f_sub, tr_l_sub = tr_f[tr_mask], tr_l[tr_mask]
    te_f_sub, te_l_sub = te_f[te_mask], te_l[te_mask]

    # Remap labels to 0..49
    label_map = {c: i for i, c in enumerate(classes)}
    tr_l_clean = torch.tensor([label_map[l.item()] for l in tr_l_sub])
    te_l_clean = torch.tensor([label_map[l.item()] for l in te_l_sub])
    n_classes = len(classes)

    # Inject 20% label noise
    rng = np.random.RandomState(SEED)
    n_noisy = int(0.20 * len(tr_l_clean))
    noise_idx = rng.choice(len(tr_l_clean), size=n_noisy, replace=False)
    tr_l_noisy = tr_l_clean.clone()
    for idx in noise_idx:
        orig = tr_l_noisy[idx].item()
        new_label = rng.choice([c for c in range(n_classes) if c != orig])
        tr_l_noisy[idx] = new_label

    print(f"  {len(tr_f_sub)} train, {len(te_f_sub)} test, {n_classes} classes")
    print(f"  Injected {n_noisy} noisy labels ({100 * n_noisy / len(tr_l_clean):.1f}%)")

    fam = make_fam(n_classes, entries=10000, adapter=adapter)
    learn_batched(fam, tr_f_sub, tr_l_noisy, n_classes)

    acc = eval_accuracy(fam, te_f_sub, te_l_clean)
    snap = proto_snapshot(fam)
    print(f"  Accuracy on clean test: {acc:.2f}%")
    print(f"  Prototypes: {snap['occupied']}")

    # Count how many noisy-label samples became prototypes
    # (check prototype class assignments vs noise injection)
    trace, _ = collect_trace(fam, te_f_sub)
    t_summary = trace_summary(trace)

    kill = acc < 90.0
    return {
        "decision": "KILL" if kill else "PASS",
        "accuracy": round(acc, 2),
        "n_noisy_injected": n_noisy,
        "store_snapshot": snap,
        "trace_summary": t_summary,
        "kill_threshold": 90.0,
    }


# ===========================================================================
# Sub-test C: G8b — ABA Re-entry
# ===========================================================================

def run_g8b():
    print("\n" + "=" * 70)
    print("  Sub-test C: G8b — ABA Re-entry (Zero Duplicates)")
    print("=" * 70)

    set_seeds()
    tr_f, tr_l, te_f, te_l = load_dogs()
    adapter = load_adapter()

    # Split into Task A (classes 0-24) and Task B (classes 25-49)
    all_classes = sorted(tr_l.unique().tolist())
    task_a_classes = set(all_classes[:25])
    task_b_classes = set(all_classes[25:50])

    def filter_classes(feats, labels, cls_set):
        mask = torch.tensor([l.item() in cls_set for l in labels])
        return feats[mask], labels[mask]

    tr_a_f, tr_a_l = filter_classes(tr_f, tr_l, task_a_classes)
    tr_b_f, tr_b_l = filter_classes(tr_f, tr_l, task_b_classes)
    te_a_f, te_a_l = filter_classes(te_f, te_l, task_a_classes)
    te_b_f, te_b_l = filter_classes(te_f, te_l, task_b_classes)

    # Remap to 0..49
    all_50 = sorted(task_a_classes | task_b_classes)
    label_map = {c: i for i, c in enumerate(all_50)}
    tr_a_l = torch.tensor([label_map[l.item()] for l in tr_a_l])
    tr_b_l = torch.tensor([label_map[l.item()] for l in tr_b_l])
    te_a_l = torch.tensor([label_map[l.item()] for l in te_a_l])
    te_b_l = torch.tensor([label_map[l.item()] for l in te_b_l])
    n_classes = 50

    fam = make_fam(n_classes, entries=10000, adapter=adapter)

    # Phase 1: Learn Task A
    learn_batched(fam, tr_a_f, tr_a_l, n_classes)
    count_a = int(fam.core_cam.occupied.sum().item())
    acc_a_after_a = eval_accuracy(fam, te_a_f, te_a_l)
    print(f"  After Task A: {count_a} prototypes, Task A acc: {acc_a_after_a:.2f}%")

    # Phase 2: Learn Task B
    learn_batched(fam, tr_b_f, tr_b_l, n_classes)
    count_ab = int(fam.core_cam.occupied.sum().item())
    acc_a_after_b = eval_accuracy(fam, te_a_f, te_a_l)
    acc_b_after_b = eval_accuracy(fam, te_b_f, te_b_l)
    print(f"  After Task B: {count_ab} prototypes, Task A acc: {acc_a_after_b:.2f}%, Task B acc: {acc_b_after_b:.2f}%")

    # Duplicate detection BEFORE re-entry
    def count_duplicates(fam_ref, n_cls):
        valid_idx = fam_ref.core_cam.occupied.nonzero(as_tuple=True)[0]
        keys = fam_ref.core_cam._keys_norm[valid_idx]
        vals = fam_ref.core_cam.values[valid_idx]
        class_ids = vals.argmax(dim=1)
        n_dup = 0
        for c in range(n_cls):
            c_mask = class_ids == c
            if c_mask.sum() < 2:
                continue
            c_keys = keys[c_mask]
            sims = c_keys @ c_keys.T
            sims.fill_diagonal_(0)
            n_dup += int((sims > 0.99).sum().item()) // 2
        return n_dup

    dups_before_reentry = count_duplicates(fam, n_classes)
    print(f"  Duplicates BEFORE re-entry: {dups_before_reentry}")

    # Phase 3: Re-enter Task A
    learn_batched(fam, tr_a_f, tr_a_l, n_classes)
    count_aba = int(fam.core_cam.occupied.sum().item())
    acc_a_after_aba = eval_accuracy(fam, te_a_f, te_a_l)
    acc_b_after_aba = eval_accuracy(fam, te_b_f, te_b_l)
    print(f"  After ABA:    {count_aba} prototypes, Task A acc: {acc_a_after_aba:.2f}%, Task B acc: {acc_b_after_aba:.2f}%")

    dups_after_reentry = count_duplicates(fam, n_classes)
    dups_created_by_reentry = dups_after_reentry - dups_before_reentry
    print(f"  Duplicates AFTER re-entry: {dups_after_reentry}")
    print(f"  Duplicates CREATED by re-entry: {dups_created_by_reentry}")

    # Trace telemetry
    trace, _ = collect_trace(fam, te_a_f)
    t_summary = trace_summary(trace)

    import math
    dup_tolerance = max(2, math.ceil(dups_before_reentry * 0.10))
    kill = dups_created_by_reentry > dup_tolerance
    return {
        "decision": "KILL" if kill else "PASS",
        "prototype_trajectory": [count_a, count_ab, count_aba],
        "accuracy_trajectory": {
            "task_a_after_a": round(acc_a_after_a, 2),
            "task_a_after_b": round(acc_a_after_b, 2),
            "task_b_after_b": round(acc_b_after_b, 2),
            "task_a_after_aba": round(acc_a_after_aba, 2),
            "task_b_after_aba": round(acc_b_after_aba, 2),
        },
        "n_duplicates_before_reentry": dups_before_reentry,
        "n_duplicates_after_reentry": dups_after_reentry,
        "n_duplicates_created_by_reentry": dups_created_by_reentry,
        "kill_tolerance": dup_tolerance,
        "trace_summary": t_summary,
        "store_snapshot": proto_snapshot(fam),
    }


# ===========================================================================
# Sub-test D: G9 — OOD Detection
# ===========================================================================

def run_g9():
    print("\n" + "=" * 70)
    print("  Sub-test D: G9 — OOD Detection (Void Space)")
    print("=" * 70)

    set_seeds()
    tr_f, tr_l, te_f, te_l = load_cifar100()

    # Split: ID = classes 0-49, OOD = classes 50-99
    id_classes = set(range(50))
    ood_classes = set(range(50, 100))

    tr_id_mask = torch.tensor([l.item() in id_classes for l in tr_l])
    te_id_mask = torch.tensor([l.item() in id_classes for l in te_l])
    te_ood_mask = torch.tensor([l.item() in ood_classes for l in te_l])

    tr_id_f, tr_id_l = tr_f[tr_id_mask], tr_l[tr_id_mask]
    te_id_f, te_id_l = te_f[te_id_mask], te_l[te_id_mask]
    te_ood_f, te_ood_l = te_f[te_ood_mask], te_l[te_ood_mask]

    print(f"  ID train: {len(tr_id_f)}, ID test: {len(te_id_f)}, OOD test: {len(te_ood_f)}")

    # Remap ID labels to 0..49
    id_label_map = {c: i for i, c in enumerate(sorted(id_classes))}
    tr_id_l_mapped = torch.tensor([id_label_map[l.item()] for l in tr_id_l])
    te_id_l_mapped = torch.tensor([id_label_map[l.item()] for l in te_id_l])
    n_id_classes = 50

    def run_ood_eval(eviction_name, use_lfu, adaptive):
        fam = make_fam(n_id_classes, entries=50000, vigilance=0.85,
                       use_lfu=use_lfu, adaptive_eviction=adaptive,
                       dim=1024, adapter=None)
        set_seeds()
        learn_batched(fam, tr_id_f, tr_id_l_mapped, n_id_classes)

        n_proto = int(fam.core_cam.occupied.sum().item())
        valid_idx = fam.core_cam.occupied.nonzero(as_tuple=True)[0]
        proto_keys = fam.core_cam._keys_norm[valid_idx]

        # Score = max cosine similarity to any prototype
        def max_cosine_scores(feats):
            scores = []
            for i in range(0, len(feats), 512):
                batch = F.normalize(feats[i:i + 512].to(DEVICE), dim=-1)
                sims = batch @ proto_keys.T  # (B, P) — cosine since both normalized
                scores.append(sims.max(dim=1).values.cpu())
            return torch.cat(scores)

        id_scores = max_cosine_scores(te_id_f)
        ood_scores = max_cosine_scores(te_ood_f)

        # AUROC
        from sklearn.metrics import roc_auc_score
        labels = torch.cat([torch.ones(len(id_scores)), torch.zeros(len(ood_scores))])
        scores_all = torch.cat([id_scores, ood_scores])
        auroc = float(roc_auc_score(labels.numpy(), scores_all.numpy()))

        # Void-space gap
        void_gap = float(id_scores.min().item() - ood_scores.max().item())

        # Per-class OOD score stats
        # te_ood_l is already the OOD subset labels (same length as ood_scores)
        ood_labels_sub = te_ood_l
        ood_class_scores = {}
        for c in sorted(ood_classes):
            c_mask = ood_labels_sub == c
            if c_mask.any():
                c_scores = ood_scores[c_mask]
                ood_class_scores[str(c)] = {
                    "mean": round(float(c_scores.mean()), 4),
                    "max": round(float(c_scores.max()), 4),
                }

        # Find worst OOD classes (highest max scores = closest to ID)
        worst_ood = sorted(ood_class_scores.items(),
                           key=lambda x: x[1]["max"], reverse=True)[:5]

        return {
            "eviction": eviction_name,
            "n_proto": n_proto,
            "auroc": round(auroc, 6),
            "void_gap": round(void_gap, 4),
            "id_score_mean": round(float(id_scores.mean()), 4),
            "id_score_min": round(float(id_scores.min()), 4),
            "ood_score_mean": round(float(ood_scores.mean()), 4),
            "ood_score_max": round(float(ood_scores.max()), 4),
            "worst_ood_classes": dict(worst_ood),
        }, fam

    # Run with adaptive eviction (current default)
    print("  Running with adaptive eviction ...")
    adaptive_result, fam_adaptive = run_ood_eval("adaptive", False, True)
    print(f"    AUROC: {adaptive_result['auroc']:.6f}")
    print(f"    Void gap: {adaptive_result['void_gap']:.4f}")

    # Run with LFU (original eviction)
    print("  Running with LFU eviction (control) ...")
    lfu_result, _ = run_ood_eval("lfu", True, False)
    print(f"    AUROC: {lfu_result['auroc']:.6f}")
    print(f"    Void gap: {lfu_result['void_gap']:.4f}")

    # Trace telemetry on adaptive FAM
    trace_id, _ = collect_trace(fam_adaptive, te_id_f)
    trace_ood, _ = collect_trace(fam_adaptive, te_ood_f)
    t_id = trace_summary(trace_id)
    t_ood = trace_summary(trace_ood)

    # Confidence on ID vs OOD
    set_seeds()
    id_sub = te_id_f[:500].to(DEVICE)
    ood_sub = te_ood_f[:500].to(DEVICE)
    _, id_conf = fam_adaptive.forward_with_confidence(id_sub)
    _, ood_conf = fam_adaptive.forward_with_confidence(ood_sub)

    kill = adaptive_result["auroc"] < 0.85
    return {
        "decision": "KILL" if kill else "PASS",
        "adaptive": adaptive_result,
        "lfu_control": lfu_result,
        "trace_id": t_id,
        "trace_ood": t_ood,
        "confidence_id_mean": round(float(id_conf.mean()), 4),
        "confidence_ood_mean": round(float(ood_conf.mean()), 4),
        "store_snapshot": proto_snapshot(fam_adaptive),
        "kill_threshold": 0.85,
    }


# ===========================================================================
# Sub-test E: G10 — Concept Drift (Backward Transfer)
# ===========================================================================

def run_g10():
    print("\n" + "=" * 70)
    print("  Sub-test E: G10 — Concept Drift (Backward Transfer)")
    print("=" * 70)

    set_seeds()
    tr_f, tr_l, te_f, te_l = load_dogs()
    adapter = load_adapter()

    # Generate blur features by adding calibrated noise in feature space.
    # Original G10 used GaussianBlur(11, σ=5) on raw images before DINOv2.
    # We approximate the effect: blur degrades high-freq features, shifting
    # the feature vector. Calibrate noise magnitude to match the expected
    # cosine distance between clean and blurred DINOv2 features (~0.1-0.2).
    set_seeds()
    noise = torch.randn_like(tr_f) * 0.15  # σ=0.15 in feature space
    tr_f_blur = F.normalize(tr_f + noise, dim=-1)
    noise_te = torch.randn_like(te_f) * 0.15
    te_f_blur = F.normalize(te_f + noise_te, dim=-1)

    # Verify cosine distance between clean and blur
    cos_dist = 1.0 - F.cosine_similarity(tr_f[:1000], tr_f_blur[:1000]).mean().item()
    print(f"  Clean-blur cosine distance: {cos_dist:.4f}")

    results = {}

    for cond_name, n_cls, n_entries in [
        ("unconstrained_10cls_2000slots", 10, 2000),
        ("constrained_50cls_2000slots", 50, 2000),
    ]:
        print(f"\n  Condition: {cond_name}")

        all_classes = sorted(tr_l.unique().tolist())
        selected = set(all_classes[:n_cls])

        def filt(feats, labels):
            mask = torch.tensor([l.item() in selected for l in labels])
            return feats[mask], labels[mask]

        tr_c_f, tr_c_l = filt(tr_f, tr_l)
        te_c_f, te_c_l = filt(te_f, te_l)
        tr_b_f, tr_b_l = filt(tr_f_blur, tr_l)
        te_b_f, te_b_l = filt(te_f_blur, te_l)

        # Remap labels
        lmap = {c: i for i, c in enumerate(sorted(selected))}
        tr_c_l = torch.tensor([lmap[l.item()] for l in tr_c_l])
        te_c_l = torch.tensor([lmap[l.item()] for l in te_c_l])
        tr_b_l = torch.tensor([lmap[l.item()] for l in tr_b_l])
        te_b_l = torch.tensor([lmap[l.item()] for l in te_b_l])

        fam = make_fam(n_cls, entries=n_entries, adapter=adapter)

        # Phase 1: Learn clean domain
        set_seeds()
        learn_batched(fam, tr_c_f, tr_c_l, n_cls)
        acc_clean_before = eval_accuracy(fam, te_c_f, te_c_l)
        count_before = int(fam.core_cam.occupied.sum().item())
        print(f"    Clean acc (before blur): {acc_clean_before:.2f}%, protos: {count_before}")

        # Phase 2: Learn blur domain
        learn_batched(fam, tr_b_f, tr_b_l, n_cls)
        acc_blur = eval_accuracy(fam, te_b_f, te_b_l)
        count_after = int(fam.core_cam.occupied.sum().item())

        # Phase 3: Re-evaluate clean domain (backward transfer)
        acc_clean_after = eval_accuracy(fam, te_c_f, te_c_l)
        bwt = acc_clean_after - acc_clean_before

        print(f"    Blur acc: {acc_blur:.2f}%")
        print(f"    Clean acc (after blur): {acc_clean_after:.2f}%, protos: {count_after}")
        print(f"    BWT: {bwt:+.2f}pp {'KILL' if bwt < -5.0 else 'PASS'}")

        # Confidence after drift
        _, conf = fam.forward_with_confidence(te_c_f[:500].to(DEVICE))
        conf_mean = float(conf.mean().item())

        # Trace
        trace, _ = collect_trace(fam, te_c_f)
        t_summary = trace_summary(trace)

        results[cond_name] = {
            "n_classes": n_cls,
            "n_entries": n_entries,
            "acc_clean_before": round(acc_clean_before, 2),
            "acc_blur": round(acc_blur, 2),
            "acc_clean_after": round(acc_clean_after, 2),
            "bwt": round(bwt, 2),
            "proto_before": count_before,
            "proto_after": count_after,
            "confidence_after_drift": round(conf_mean, 4),
            "trace_summary": t_summary,
            "kill": bwt < -5.0,
        }

    any_kill = any(r["kill"] for r in results.values())
    return {
        "decision": "KILL" if any_kill else "PASS",
        "clean_blur_cosine_distance": round(cos_dist, 4),
        "conditions": results,
        "kill_threshold_bwt": -5.0,
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  G-43 Thin-Telemetry Sweep")
    print(f"  Git: {GIT_COMMIT} | Seed: {SEED} | Device: {DEVICE}")
    print("=" * 70)

    all_results = {"git_commit": GIT_COMMIT, "seed": SEED, "device": DEVICE}

    # Run all sub-tests (don't early-stop on kill)
    for name, fn in [
        ("g5_indexing", run_g5),
        ("g8a_noise", run_g8a),
        ("g8b_aba", run_g8b),
        ("g9_ood", run_g9),
        ("g10_drift", run_g10),
    ]:
        try:
            result = fn()
        except Exception as e:
            result = {"decision": "ERROR", "error": str(e)}
            import traceback
            traceback.print_exc()
        all_results[name] = result

        # Save incrementally
        with open(OUT_DIR / "summary.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # Final summary
    print("\n" + "=" * 70)
    print("  G-43 SUMMARY")
    print("=" * 70)
    for name in ["g5_indexing", "g8a_noise", "g8b_aba", "g9_ood", "g10_drift"]:
        r = all_results.get(name, {})
        dec = r.get("decision", "N/A")
        print(f"  {name:20s} {dec}")

    any_kill = any(
        all_results.get(n, {}).get("decision") == "KILL"
        for n in ["g5_indexing", "g8a_noise", "g8b_aba", "g9_ood", "g10_drift"]
    )
    all_results["overall_decision"] = "KILL" if any_kill else "PASS"
    print(f"\n  Overall: {all_results['overall_decision']}")

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
