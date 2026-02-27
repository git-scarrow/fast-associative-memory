#!/usr/bin/env python3
"""
MT-17 — Read-Path Dead-Weight Ablation (Mahalanobis + CSLS)

Tests whether Mahalanobis rerank and CSLS hubness correction contribute
anything across 7 domains. If both are dead weight everywhere, they can
be stripped from the read path.

Conditions:
  1. Full pipeline (control)
  2. No Mahalanobis (class_vars → all-ones)
  3. No CSLS (csls_weight → 0)
  4. Neither (both disabled)

Kill threshold: >0.5pp accuracy drop OR >1% top-1 flip rate in any domain.

Usage:
    PYTHONPATH=. .venv/bin/python benchmarks/mt17_deadweight_ablation.py
"""
from __future__ import annotations

import random
import sys
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
ADAPTER_DOMAINS = {"dogs", "birds", "cars", "aircraft"}


def set_seeds():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------------
# Data loading for all 7 domains
# ---------------------------------------------------------------------------

def load_domain(domain: str, data_root: Path):
    """Returns (train_feats, train_labels, test_feats, test_labels, n_classes, adapter_or_None)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if domain in ADAPTER_DOMAINS:
        tr_f, tr_l, te_f, te_l, nc = g30_load_domain_features(domain, data_root)
        adapter = _load_g30_domain_adapter(domain, input_dim=tr_f.shape[1], device=device)
        return tr_f, tr_l, te_f, te_l, nc, adapter

    if domain == "cifar100":
        root = Path("feature_cache_vitl14")
        train = torch.load(root / "cifar100_dinov2_train.pt", map_location="cpu", weights_only=True)
        test = torch.load(root / "cifar100_dinov2_test.pt", map_location="cpu", weights_only=True)
        return (train["embeds"].float(), train["labels"].long(),
                test["embeds"].float(), test["labels"].long(),
                100, None)

    if domain == "flowers":
        root = data_root / "flowers102" / "feature_cache"
        train = torch.load(root / "flowers102_train.pt", map_location="cpu", weights_only=True)
        test = torch.load(root / "flowers102_test.pt", map_location="cpu", weights_only=True)
        fk = "embeds" if "embeds" in train else "features"
        return (train[fk].float(), train["labels"].long(),
                test[fk].float(), test["labels"].long(),
                int(train["labels"].max().item() + 1), None)

    if domain == "imagenet_r":
        root = Path("feature_cache_inr_vitl14")
        train = torch.load(root / "imagenetr_dinov2_train.pt", map_location="cpu", weights_only=True)
        test = torch.load(root / "imagenetr_dinov2_test.pt", map_location="cpu", weights_only=True)
        fk = "embeds" if "embeds" in train else "features"
        return (train[fk].float(), train["labels"].long(),
                test[fk].float(), test["labels"].long(),
                int(train["labels"].max().item() + 1), None)

    raise ValueError(f"Unknown domain: {domain}")


# ---------------------------------------------------------------------------
# Build FAM + eval under different ablation conditions
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_and_eval(train_feats, train_labels, test_feats, test_labels,
                   n_classes, adapter, device,
                   disable_mahalanobis=False, disable_csls=False):
    """Build FAM from train, eval on test under ablation. Returns (preds, acc, margins)."""
    set_seeds()
    nstp = NSTPController(sibling_threshold=0.85, depth_epsilon=0.10)

    mem = FastAssociativeMemory(
        input_dim=int(train_feats.shape[1]),
        value_dim=n_classes,
        core_entries=50000, core_vigilance=0.85,
        hebb_lr=0.1, key_lr=0.05,
        inference_k=25, inference_temp=0.05,
        use_lfu=True, adapter=adapter, nstp=nstp,
    ).to(device)

    # Ingest
    x_train = train_feats.to(device)
    y_train = train_labels.to(device)
    for i in range(0, len(x_train), 256):
        mem.learn_local(x_train[i:i + 256], y_train[i:i + 256])

    core = mem.core_cam

    # Apply ablations
    if disable_mahalanobis:
        core.class_vars.fill_(1.0)  # identity transform
    if disable_csls:
        core.csls_weight = 0.0

    # Eval
    mem.eval()
    all_preds = []
    all_margins = []
    x_test = test_feats.to(device)
    y_test = test_labels.to(device)

    occ_idx = core.occupied.nonzero(as_tuple=True)[0]
    proto_classes = core.values[occ_idx].float().argmax(dim=-1)

    for i in range(0, len(x_test), 256):
        bx = x_test[i:i + 256]
        outputs = mem(bx)
        preds = outputs.argmax(1)
        all_preds.append(preds.cpu())

        # Compute margins for quartile analysis
        proj = mem._project(bx)
        q_norm = F.normalize(proj.float(), dim=-1)
        sims = q_norm @ core._keys_norm[occ_idx].T
        for b in range(bx.size(0)):
            pred_cls = preds[b].item()
            row_sims = sims[b]
            best = row_sims.max().item()
            same_mask = proto_classes == pred_cls
            other = row_sims.masked_fill(same_mask, -1e9)
            second = other.max().item() if other.max().item() > -1e8 else best
            all_margins.append(best - second)

    preds_t = torch.cat(all_preds)
    acc = 100.0 * (preds_t == test_labels).float().mean().item()
    n_protos = int(core.occupied.sum().item())

    del mem
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return preds_t, acc, np.array(all_margins), n_protos


def spearman_rank_corr(preds_full, preds_ablated):
    """Spearman rank correlation — but preds are class labels, not rankings.
    Use agreement rate instead since we don't have candidate orderings easily.
    Actually, let's compute flip rate which is more informative."""
    pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path("data")

    DOMAINS = ["dogs", "birds", "cars", "aircraft", "cifar100", "flowers", "imagenet_r"]
    CONDITIONS = [
        ("Full", False, False),
        ("No-Mah", True, False),
        ("No-CSLS", False, True),
        ("Neither", True, True),
    ]

    print("=" * 90)
    print("  MT-17 — Read-Path Dead-Weight Ablation (Mahalanobis + CSLS)")
    print("=" * 90)
    print(f"  Device: {device}")

    results = {}  # domain -> {condition_name: (preds, acc, margins, n_protos)}

    for domain in DOMAINS:
        print(f"\n  --- {domain.upper()} ---")
        tr_f, tr_l, te_f, te_l, nc, adapter = load_domain(domain, data_root)
        print(f"  Train: {len(tr_f)}, Test: {len(te_f)}, Classes: {nc}, "
              f"Adapter: {'yes' if adapter else 'no'}")

        results[domain] = {}
        for cond_name, dis_mah, dis_csls in CONDITIONS:
            preds, acc, margins, n_protos = build_and_eval(
                tr_f, tr_l, te_f, te_l, nc, adapter, device,
                disable_mahalanobis=dis_mah, disable_csls=dis_csls,
            )
            results[domain][cond_name] = (preds, acc, margins, n_protos)
            print(f"    {cond_name:10s}: {acc:.2f}% ({n_protos} protos)")

    # === Summary Table ===
    print(f"\n{'=' * 105}")
    print(f"  SUMMARY TABLE")
    print(f"{'=' * 105}")
    print(f"  {'Domain':12s} | {'Full':>7} | {'No-Mah':>7} {'Δ':>7} {'Flips':>6} | "
          f"{'No-CSLS':>7} {'Δ':>7} {'Flips':>6} | {'Neither':>7} {'Δ':>7} {'Flips':>6}")
    print(f"  {'-' * 12}-+-{'-' * 7}-+-{'-' * 7}-{'-' * 7}-{'-' * 6}-+-"
          f"{'-' * 7}-{'-' * 7}-{'-' * 6}-+-{'-' * 7}-{'-' * 7}-{'-' * 6}")

    any_kill = False
    for domain in DOMAINS:
        r = results[domain]
        full_preds, full_acc, full_margins, _ = r["Full"]
        n_test = len(full_preds)

        parts = [f"  {domain:12s} | {full_acc:>6.2f}%"]
        for cond in ["No-Mah", "No-CSLS", "Neither"]:
            preds, acc, margins, _ = r[cond]
            delta = acc - full_acc
            flips = (preds != full_preds).float().mean().item() * 100
            parts.append(f" | {acc:>6.2f}% {delta:>+6.2f} {flips:>5.1f}%")
            if abs(delta) > 0.5 or flips > 1.0:
                any_kill = True
        print("".join(parts))

    # === Per-Quartile Table ===
    print(f"\n{'=' * 105}")
    print(f"  PER-QUARTILE ACCURACY (by Full-pipeline margin)")
    print(f"{'=' * 105}")

    for cond_name in ["No-Mah", "No-CSLS", "Neither"]:
        print(f"\n  Condition: {cond_name}")
        print(f"  {'Domain':12s} | {'Q1 Full':>7} {'Q1 Abl':>7} {'ΔQ1':>6} | "
              f"{'Q2 Full':>7} {'Q2 Abl':>7} {'ΔQ2':>6} | "
              f"{'Q3 Full':>7} {'Q3 Abl':>7} {'ΔQ3':>6} | "
              f"{'Q4 Full':>7} {'Q4 Abl':>7} {'ΔQ4':>6}")

        for domain in DOMAINS:
            r = results[domain]
            full_preds, full_acc, full_margins, _ = r["Full"]
            abl_preds, abl_acc, abl_margins, _ = r[cond_name]
            test_labels_cpu = torch.load(
                _get_test_path(domain, data_root), map_location="cpu", weights_only=True
            )["labels"].long() if False else None

            # Re-derive test labels from full_preds correctness — we need actual labels
            # Use the stored data
            _, _, te_f, te_l, _, _ = load_domain(domain, data_root)

            sorted_idx = np.argsort(full_margins)
            q_size = len(full_margins) // 4

            parts = [f"  {domain:12s}"]
            for qi in range(4):
                start = qi * q_size
                end = (qi + 1) * q_size if qi < 3 else len(full_margins)
                idx = sorted_idx[start:end]

                full_q_acc = 100.0 * (full_preds[idx] == te_l[idx]).float().mean().item()
                abl_q_acc = 100.0 * (abl_preds[idx] == te_l[idx]).float().mean().item()
                delta = abl_q_acc - full_q_acc
                parts.append(f" | {full_q_acc:>6.2f}% {abl_q_acc:>6.2f}% {delta:>+5.2f}")
            print("".join(parts))

    # === Verdict ===
    print(f"\n{'=' * 105}")
    if any_kill:
        print("  VERDICT: At least one domain exceeded kill threshold.")
        print("  Stage(s) with signal survive as domain-conditional.")
    else:
        print("  VERDICT: CONFIRMED DEAD WEIGHT — Both Mahalanobis and CSLS are")
        print("  no-ops across all 7 domains. Safe to strip from read path.")
    print(f"{'=' * 105}")


def _get_test_path(domain, data_root):
    """Unused — kept for reference."""
    pass


if __name__ == "__main__":
    main()
