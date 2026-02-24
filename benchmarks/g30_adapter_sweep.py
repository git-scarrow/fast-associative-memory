#!/usr/bin/env python3
"""
benchmarks/g30_adapter_sweep.py — G30: Adapter Hyperparameter Sweep (Per-Domain)

Phases
------
1. Axis sensitivity scan:
   Sweep one axis at a time over 4 domains (Dogs, Birds, Cars, Aircraft),
   holding all other hyperparameters at the G30 defaults.

2. Focused 3×3×3 grid on top-3 axes per domain:
   Use Phase 1 CSV to pick the 3 most sensitive axes (largest Δ accuracy),
   pick the top-3 values for each, and run a full grid.

3. Cross-domain regression:
   Take each domain's Phase 2 winner (best in-domain accuracy), retrain that
   config, and evaluate the resulting adapter on all 4 domains, yielding a
   4×4 accuracy matrix.

All phases train adapters on pre-extracted DINOv2 ViT-L/14 (1024-d) features,
then evaluate via FastAssociativeMemory + NSTP (same stack used in G22–G25).

Feature caches (existing)
-------------------------
  Dogs:     data/stanford_dogs/stanford_dogs_{train,test}.pt
  Birds:    data/cub200/cub200_{train,test}.pt
  Cars:     data/stanford_cars/stanford_cars_{train,test}.pt
  Aircraft: data/fgvc_aircraft/fgvc_aircraft_{train,test}.pt

CSV schema
----------
Axis-level runs (Phase 1) and grid runs (Phase 2) share this schema:

  phase,domain,axis,value,accuracy,n_prototypes,condensation_ratio,
  training_time_s,proj_dim,nonlinearity,layers,loss_type,triplet_margin,
  infonce_temp,ms_alpha,ms_beta,ms_base,lr,epochs,mining_strategy,batch_size

Phase 3 emits:

  phase=3, train_domain, eval_domain, accuracy, n_prototypes, condensation_ratio
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from adapter import MetricAdapter
from fast_associative_memory import FastAssociativeMemory
from nstp import NSTPController


################################################################################
# Types and configuration
################################################################################


LossType = Literal["triplet", "infonce", "multisim"]
MiningStrategy = Literal["semi-hard", "hard", "all"]
Nonlinearity = Literal["relu", "gelu", "silu", "identity"]
DomainName = Literal["dogs", "birds", "cars", "aircraft"]


@dataclass
class AdapterConfig:
    proj_dim: int = 1024
    nonlinearity: Nonlinearity = "relu"
    layers: int = 2
    loss_type: LossType = "triplet"
    triplet_margin: float = 0.3
    infonce_temp: float = 0.07
    ms_alpha: float = 2.0
    ms_beta: float = 50.0
    ms_base: float = 0.5
    lr: float = 1e-3
    epochs: int = 10
    mining_strategy: MiningStrategy = "semi-hard"
    batch_size: int = 256


@dataclass
class RunRow:
    phase: int
    domain: str
    axis: str
    value: str
    accuracy: float
    n_prototypes: int
    condensation_ratio: float
    training_time_s: float
    proj_dim: int
    nonlinearity: str
    layers: int
    loss_type: str
    triplet_margin: float
    infonce_temp: float
    ms_alpha: float
    ms_beta: float
    ms_base: float
    lr: float
    epochs: int
    mining_strategy: str
    batch_size: int


CSV_FIELDNAMES = [
    "phase",
    "domain",
    "axis",
    "value",
    "accuracy",
    "n_prototypes",
    "condensation_ratio",
    "training_time_s",
    "proj_dim",
    "nonlinearity",
    "layers",
    "loss_type",
    "triplet_margin",
    "infonce_temp",
    "ms_alpha",
    "ms_beta",
    "ms_base",
    "lr",
    "epochs",
    "mining_strategy",
    "batch_size",
]


AXES: Dict[str, Iterable] = {
    "proj_dim": [256, 512, 1024, 2048],
    "nonlinearity": ["relu", "gelu", "silu", "identity"],
    "layers": [1, 2, 3, 4],
    "loss_type": ["triplet", "infonce", "multisim"],
    "triplet_margin": [0.1, 0.2, 0.3, 0.5, 1.0],
    "infonce_temp": [0.01, 0.05, 0.07, 0.1, 0.2],
    "lr": [1e-4, 5e-4, 1e-3, 5e-3],
    "epochs": [5, 10, 20, 40],
    "mining_strategy": ["semi-hard", "hard", "all"],
    "batch_size": [64, 128, 256, 512],
}


################################################################################
# Feature loaders
################################################################################


def _device_from_str(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _load_pt(path: Path, features_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    feats = obj[features_key].float()
    labels = obj["labels"].long()
    return feats, labels


def load_domain_features(
    domain: DomainName,
    root_dir: Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Load pre-extracted DINOv2 features for a domain."""
    if domain == "dogs":
        train_pt = root_dir / "stanford_dogs" / "stanford_dogs_train.pt"
        test_pt = root_dir / "stanford_dogs" / "stanford_dogs_test.pt"
        feats_key = "features"
    elif domain == "birds":
        train_pt = root_dir / "cub200" / "cub200_train.pt"
        test_pt = root_dir / "cub200" / "cub200_test.pt"
        feats_key = "embeds"
    elif domain == "cars":
        train_pt = root_dir / "stanford_cars" / "stanford_cars_train.pt"
        test_pt = root_dir / "stanford_cars" / "stanford_cars_test.pt"
        feats_key = "embeds"
    elif domain == "aircraft":
        train_pt = root_dir / "fgvc_aircraft" / "fgvc_aircraft_train.pt"
        test_pt = root_dir / "fgvc_aircraft" / "fgvc_aircraft_test.pt"
        feats_key = "embeds"
    else:
        raise ValueError(f"Unknown domain {domain!r}")

    if not train_pt.is_file() or not test_pt.is_file():
        raise FileNotFoundError(
            f"Expected feature caches at {train_pt} and {test_pt}. "
            "Run the corresponding extraction scripts first."
        )

    train_feats, train_labels = _load_pt(train_pt, feats_key)
    test_feats, test_labels = _load_pt(test_pt, feats_key)
    n_classes = int(train_labels.max().item() + 1)
    return train_feats, train_labels, test_feats, test_labels, n_classes


################################################################################
# Losses and mining
################################################################################


def _mine_triplets(
    proj: torch.Tensor,
    feats: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
    strategy: MiningStrategy,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triplet mining over a batch.

    * semi-hard:  margin < d(a,n) - d(a,p) < 2*margin
    * hard:       hardest negative (highest cosine similarity)
    * all:        all possible (a,p,n) with at least one positive and negative
                  per anchor; this can be heavy on large batches.
    """
    device = feats.device
    B = feats.size(0)
    anchors: List[torch.Tensor] = []
    positives: List[torch.Tensor] = []
    negatives: List[torch.Tensor] = []

    sim = proj @ proj.T  # (B, B), cosine since proj is L2-normalised

    for i in range(B):
        lbl = labels[i]
        same_mask = labels == lbl
        diff_mask = labels != lbl

        same_idx = same_mask.nonzero(as_tuple=True)[0]
        same_idx = same_idx[same_idx != i]
        if same_idx.numel() == 0 or diff_mask.sum() == 0:
            continue

        if strategy == "all":
            # Use all (anchor, positive, negative) triplets for this anchor.
            a = feats[i].expand(same_idx.numel(), -1)
            p = feats[same_idx]
            diff_idx = diff_mask.nonzero(as_tuple=True)[0]
            a_rep = a.unsqueeze(1).expand(-1, diff_idx.numel(), -1)
            p_rep = p.unsqueeze(1).expand(-1, diff_idx.numel(), -1)
            n_rep = feats[diff_idx].unsqueeze(0).expand(same_idx.numel(), -1, -1)
            anchors.append(a_rep.reshape(-1, feats.size(1)))
            positives.append(p_rep.reshape(-1, feats.size(1)))
            negatives.append(n_rep.reshape(-1, feats.size(1)))
            continue

        # Random positive for semi-hard / hard.
        pos_idx = same_idx[
            torch.randint(len(same_idx), (1,), device=device).item()
        ]

        if strategy == "hard":
            sim_row = sim[i].clone()
            sim_row[~diff_mask] = -2.0
            neg_idx = sim_row.argmax()
        else:  # semi-hard
            a_proj = proj[i].unsqueeze(0)
            dists = torch.norm(a_proj - proj, dim=1)
            d_ap = dists[pos_idx]
            cand_idx = diff_mask.nonzero(as_tuple=True)[0]
            d_an = dists[cand_idx]
            mask = (d_an - d_ap > margin) & (d_an - d_ap < 2 * margin)
            if mask.any():
                chosen = torch.where(mask)[0][0]
                neg_idx = cand_idx[chosen]
            else:
                sim_row = sim[i].clone()
                sim_row[~diff_mask] = -2.0
                neg_idx = sim_row.argmax()

        anchors.append(feats[i])
        positives.append(feats[pos_idx])
        negatives.append(feats[neg_idx])

    if not anchors:
        empty = torch.empty(0, feats.size(1), device=device)
        return empty, empty, empty

    if strategy == "all":
        return (
            torch.cat(anchors, dim=0),
            torch.cat(positives, dim=0),
            torch.cat(negatives, dim=0),
        )

    return (
        torch.stack(anchors),
        torch.stack(positives),
        torch.stack(negatives),
    )


def _triplet_loss(
    adapter: MetricAdapter,
    feats: torch.Tensor,
    labels: torch.Tensor,
    cfg: AdapterConfig,
) -> torch.Tensor:
    with torch.no_grad():
        proj = adapter(feats)
    a, p, n = _mine_triplets(
        proj,
        feats,
        labels,
        margin=cfg.triplet_margin,
        strategy=cfg.mining_strategy,
    )
    if a.numel() == 0:
        return torch.zeros((), device=feats.device)
    return adapter.triplet_loss(a, p, n, margin=cfg.triplet_margin)


def _info_nce_loss(
    adapter: MetricAdapter,
    feats: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Supervised InfoNCE: each anchor uses all same-class samples as positives."""
    z = adapter(feats)  # (B, D), L2-normalised
    sim = z @ z.T
    sim = sim / temperature

    mask = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    sim.masked_fill_(mask, float("-inf"))

    labels = labels.view(-1, 1)
    same = labels.eq(labels.T) & ~mask
    log_denom = torch.logsumexp(sim, dim=1)
    sim_pos = sim.masked_fill(~same, float("-inf"))
    log_num = torch.logsumexp(sim_pos, dim=1)

    valid = torch.isfinite(log_num)
    if not valid.any():
        return torch.zeros((), device=sim.device)

    loss = -(log_num[valid] - log_denom[valid]).mean()
    return loss


def _multisimilarity_loss(
    adapter: MetricAdapter,
    feats: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    beta: float,
    base: float,
) -> torch.Tensor:
    """Multi-Similarity Loss (Wang et al., CVPR 2019).

    This is a straightforward implementation adapted to our cosine-similarity
    setting; no mining strategy is applied beyond using all positive/negative
    pairs in the batch.
    """
    z = adapter(feats)  # (B, D), L2-normalised
    sim = z @ z.T
    B = sim.size(0)
    mask = torch.eye(B, dtype=torch.bool, device=sim.device)

    labels = labels.view(-1, 1)
    same = labels.eq(labels.T) & ~mask
    diff = ~same & ~mask

    loss_list: List[torch.Tensor] = []
    for i in range(B):
        s_pos = sim[i][same[i]]
        s_neg = sim[i][diff[i]]
        if s_pos.numel() == 0 or s_neg.numel() == 0:
            continue
        pos_term = (1.0 / alpha) * torch.log1p(
            torch.sum(torch.exp(-alpha * (s_pos - base)))
        )
        neg_term = (1.0 / beta) * torch.log1p(
            torch.sum(torch.exp(beta * (s_neg - base)))
        )
        loss_list.append(pos_term + neg_term)

    if not loss_list:
        return torch.zeros((), device=sim.device)
    return torch.stack(loss_list).mean()


################################################################################
# Training and evaluation
################################################################################


def make_adapter(input_dim: int, cfg: AdapterConfig, device: torch.device) -> MetricAdapter:
    hidden_dim = cfg.proj_dim if cfg.layers >= 2 else 0
    adapter = MetricAdapter(
        input_dim=input_dim,
        output_dim=cfg.proj_dim,
        hidden_dim=hidden_dim,
        nonlinearity=cfg.nonlinearity,
        proj_dim=cfg.proj_dim,
        layers=cfg.layers,
        residual=(cfg.layers >= 2),
    )
    adapter.to(device)
    return adapter


def train_adapter(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    cfg: AdapterConfig,
    device: torch.device,
) -> Tuple[MetricAdapter, float]:
    """Train MetricAdapter under cfg; returns (adapter, training_time_s)."""
    input_dim = train_feats.size(1)
    adapter = make_adapter(input_dim, cfg, device)

    ds = TensorDataset(train_feats, train_labels)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.Adam(adapter.parameters(), lr=cfg.lr)

    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        adapter.train()
        for feats, labels in loader:
            feats = feats.to(device)
            labels = labels.to(device)

            opt.zero_grad()
            if cfg.loss_type == "triplet":
                loss = _triplet_loss(adapter, feats, labels, cfg)
            elif cfg.loss_type == "infonce":
                loss = _info_nce_loss(adapter, feats, labels, cfg.infonce_temp)
            else:
                loss = _multisimilarity_loss(
                    adapter,
                    feats,
                    labels,
                    cfg.ms_alpha,
                    cfg.ms_beta,
                    cfg.ms_base,
                )

            if loss.item() == 0.0:
                continue

            loss.backward()
            opt.step()

    training_time_s = float(time.perf_counter() - t0)
    return adapter, training_time_s


@torch.no_grad()
def eval_full_stack(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    adapter: MetricAdapter,
    device: torch.device,
) -> Tuple[float, int, float]:
    """Evaluate adapter+NSTP+FAM on a single domain."""
    nstp = NSTPController(sibling_threshold=0.85, depth_epsilon=0.10)

    mem = FastAssociativeMemory(
        input_dim=train_feats.size(1),
        value_dim=num_classes,
        core_entries=50000,
        core_vigilance=0.92,
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=25,
        inference_temp=0.05,
        use_lfu=True,
        adapter=adapter,
        nstp=nstp,
    ).to(device)

    x_tr = train_feats.to(device)
    y_tr = train_labels.to(device)
    for i in range(0, len(x_tr), 256):
        mem.learn_local(x_tr[i : i + 256], y_tr[i : i + 256])

    n_protos = int(mem.core_cam.occupied.sum().item())
    condensation = n_protos / float(len(train_feats))

    correct = 0
    x_te = test_feats.to(device)
    y_te = test_labels.to(device)
    for i in range(0, len(x_te), 512):
        logits = mem(x_te[i : i + 512])
        correct += (logits.argmax(dim=1) == y_te[i : i + 512]).sum().item()

    acc = 100.0 * correct / float(len(test_labels))
    return acc, n_protos, condensation


################################################################################
# CSV helpers
################################################################################


def load_existing_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.is_file():
        return []
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def append_rows(csv_path: Path, rows: List[RunRow]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.is_file()
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "phase": r.phase,
                    "domain": r.domain,
                    "axis": r.axis,
                    "value": r.value,
                    "accuracy": f"{r.accuracy:.4f}",
                    "n_prototypes": r.n_prototypes,
                    "condensation_ratio": f"{r.condensation_ratio:.6f}",
                    "training_time_s": f"{r.training_time_s:.3f}",
                    "proj_dim": r.proj_dim,
                    "nonlinearity": r.nonlinearity,
                    "layers": r.layers,
                    "loss_type": r.loss_type,
                    "triplet_margin": r.triplet_margin,
                    "infonce_temp": r.infonce_temp,
                    "ms_alpha": r.ms_alpha,
                    "ms_beta": r.ms_beta,
                    "ms_base": r.ms_base,
                    "lr": r.lr,
                    "epochs": r.epochs,
                    "mining_strategy": r.mining_strategy,
                    "batch_size": r.batch_size,
                }
            )


################################################################################
# Phase 1 — Axis sensitivity
################################################################################


def run_phase1_for_domain(
    domain: DomainName,
    data_root: Path,
    base_cfg: AdapterConfig,
    device: torch.device,
    csv_path: Path,
) -> None:
    existing = load_existing_rows(csv_path)
    done_keys = {
        (int(row["phase"]), row["domain"], row["axis"], row["value"])
        for row in existing
        if row.get("phase")
    }

    tr_f, tr_l, te_f, te_l, n_classes = load_domain_features(domain, data_root)

    all_new_rows: List[RunRow] = []

    for axis, values in AXES.items():
        for v in values:
            key = (1, domain, axis, str(v))
            if key in done_keys:
                continue

            cfg = AdapterConfig(**asdict(base_cfg))
            setattr(cfg, axis, v)

            # Axis-dependent rules:
            if axis == "triplet_margin":
                cfg.loss_type = "triplet"
            if axis == "infonce_temp":
                cfg.loss_type = "infonce"
            if axis in ("triplet_margin", "mining_strategy") and cfg.loss_type != "triplet":
                continue
            if axis == "infonce_temp" and cfg.loss_type != "infonce":
                continue

            print(f"[G30][Phase1] {domain} axis={axis} value={v} …", flush=True)
            adapter, train_time = train_adapter(tr_f, tr_l, cfg, device)
            adapter.eval()
            acc, n_protos, cond = eval_full_stack(
                tr_f, tr_l, te_f, te_l, n_classes, adapter, device
            )
            print(
                f"[G30][Phase1] {domain} axis={axis} value={v} → "
                f"acc={acc:.2f}% protos={n_protos} cond={cond:.3f}",
                flush=True,
            )

            all_new_rows.append(
                RunRow(
                    phase=1,
                    domain=domain,
                    axis=axis,
                    value=str(v),
                    accuracy=acc,
                    n_prototypes=n_protos,
                    condensation_ratio=cond,
                    training_time_s=train_time,
                    proj_dim=cfg.proj_dim,
                    nonlinearity=cfg.nonlinearity,
                    layers=cfg.layers,
                    loss_type=cfg.loss_type,
                    triplet_margin=cfg.triplet_margin,
                    infonce_temp=cfg.infonce_temp,
                    ms_alpha=cfg.ms_alpha,
                    ms_beta=cfg.ms_beta,
                    ms_base=cfg.ms_base,
                    lr=cfg.lr,
                    epochs=cfg.epochs,
                    mining_strategy=cfg.mining_strategy,
                    batch_size=cfg.batch_size,
                )
            )

            # Flush incrementally to allow resume on interruption.
            append_rows(csv_path, all_new_rows)
            all_new_rows.clear()


################################################################################
# Phase 2 — Focused grid on top-3 axes
################################################################################


def _parse_csv_float(s: str) -> float:
    try:
        return float(s)
    except Exception:
        return float("nan")


def select_top3_axes_for_domain(
    domain: DomainName,
    csv_path: Path,
) -> List[str]:
    rows = load_existing_rows(csv_path)
    # axis -> value -> list[acc]
    stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if int(row.get("phase", 0)) != 1 or row.get("domain") != domain:
            continue
        axis = row["axis"]
        value = row["value"]
        acc = _parse_csv_float(row["accuracy"])
        if not (acc == acc):  # NaN check
            continue
        stats[axis][value].append(acc)

    deltas: List[Tuple[str, float]] = []
    for axis, val_dict in stats.items():
        means = [sum(vs) / len(vs) for vs in val_dict.values() if vs]
        if not means:
            continue
        d = max(means) - min(means)
        deltas.append((axis, d))

    deltas.sort(key=lambda t: t[1], reverse=True)
    return [axis for axis, _ in deltas[:3]]


def select_top3_values_for_axis(
    domain: DomainName,
    axis: str,
    csv_path: Path,
) -> List[str]:
    rows = load_existing_rows(csv_path)
    # value -> list[acc]
    stats: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        if int(row.get("phase", 0)) != 1 or row.get("domain") != domain:
            continue
        if row["axis"] != axis:
            continue
        acc = _parse_csv_float(row["accuracy"])
        if not (acc == acc):  # NaN
            continue
        stats[row["value"]].append(acc)

    scored: List[Tuple[str, float]] = []
    for v, accs in stats.items():
        if not accs:
            continue
        scored.append((v, sum(accs) / len(accs)))
    scored.sort(key=lambda t: t[1], reverse=True)
    return [v for v, _ in scored[:3]]


def run_phase2_for_domain(
    domain: DomainName,
    data_root: Path,
    base_cfg: AdapterConfig,
    device: torch.device,
    csv_path: Path,
) -> None:
    rows = load_existing_rows(csv_path)
    existing_keys = {
        (int(row["phase"]), row["domain"], row["axis"], row["value"])
        for row in rows
        if row.get("phase")
    }

    top_axes = select_top3_axes_for_domain(domain, csv_path)
    if len(top_axes) < 3:
        print(f"[G30][Phase2] Domain {domain}: fewer than 3 axes with signal; skipping.")
        return

    axis_vals: Dict[str, List[str]] = {}
    for axis in top_axes:
        axis_vals[axis] = select_top3_values_for_axis(domain, axis, csv_path)
        print(f"[G30][Phase2] {domain} axis={axis} values={axis_vals[axis]}")

    tr_f, tr_l, te_f, te_l, n_classes = load_domain_features(domain, data_root)

    all_new: List[RunRow] = []

    a1, a2, a3 = top_axes
    for v1 in axis_vals[a1]:
        for v2 in axis_vals[a2]:
            for v3 in axis_vals[a3]:
                combo_label = f"{a1}={v1},{a2}={v2},{a3}={v3}"
                key = (2, domain, "combo", combo_label)
                if key in existing_keys:
                    continue

                cfg = AdapterConfig(**asdict(base_cfg))
                for axis, val in ((a1, v1), (a2, v2), (a3, v3)):
                    # Cast back to correct type
                    if axis in ("proj_dim", "layers", "epochs", "batch_size"):
                        setattr(cfg, axis, int(float(val)))
                    elif axis in ("lr", "triplet_margin", "infonce_temp"):
                        setattr(cfg, axis, float(val))
                    elif axis == "nonlinearity":
                        setattr(cfg, axis, val)
                    elif axis == "loss_type":
                        setattr(cfg, axis, val)
                    elif axis == "mining_strategy":
                        setattr(cfg, axis, val)
                print(f"[G30][Phase2] {domain} combo {combo_label} …", flush=True)
                adapter, train_time = train_adapter(tr_f, tr_l, cfg, device)
                adapter.eval()
                acc, n_protos, cond = eval_full_stack(
                    tr_f, tr_l, te_f, te_l, n_classes, adapter, device
                )
                print(
                    f"[G30][Phase2] {domain} combo {combo_label} → "
                    f"acc={acc:.2f}% protos={n_protos} cond={cond:.3f}",
                    flush=True,
                )

                all_new.append(
                    RunRow(
                        phase=2,
                        domain=domain,
                        axis="combo",
                        value=combo_label,
                        accuracy=acc,
                        n_prototypes=n_protos,
                        condensation_ratio=cond,
                        training_time_s=train_time,
                        proj_dim=cfg.proj_dim,
                        nonlinearity=cfg.nonlinearity,
                        layers=cfg.layers,
                        loss_type=cfg.loss_type,
                        triplet_margin=cfg.triplet_margin,
                        infonce_temp=cfg.infonce_temp,
                        ms_alpha=cfg.ms_alpha,
                        ms_beta=cfg.ms_beta,
                        ms_base=cfg.ms_base,
                        lr=cfg.lr,
                        epochs=cfg.epochs,
                        mining_strategy=cfg.mining_strategy,
                        batch_size=cfg.batch_size,
                    )
                )
                append_rows(csv_path, all_new)
                all_new.clear()


################################################################################
# Phase 3 — Cross-domain regression
################################################################################


def best_phase2_config_for_domain(
    domain: DomainName,
    csv_path: Path,
) -> AdapterConfig | None:
    rows = load_existing_rows(csv_path)
    best_acc = float("-inf")
    best_row: Dict[str, str] | None = None
    for row in rows:
        if int(row.get("phase", 0)) != 2 or row.get("domain") != domain:
            continue
        acc = _parse_csv_float(row["accuracy"])
        if acc > best_acc:
            best_acc = acc
            best_row = row
    if best_row is None:
        return None

    cfg = AdapterConfig()
    cfg.proj_dim = int(best_row["proj_dim"])
    cfg.nonlinearity = best_row["nonlinearity"]  # type: ignore[assignment]
    cfg.layers = int(best_row["layers"])
    cfg.loss_type = best_row["loss_type"]  # type: ignore[assignment]
    cfg.triplet_margin = float(best_row["triplet_margin"])
    cfg.infonce_temp = float(best_row["infonce_temp"])
    cfg.ms_alpha = float(best_row["ms_alpha"])
    cfg.ms_beta = float(best_row["ms_beta"])
    cfg.ms_base = float(best_row["ms_base"])
    cfg.lr = float(best_row["lr"])
    cfg.epochs = int(best_row["epochs"])
    cfg.mining_strategy = best_row["mining_strategy"]  # type: ignore[assignment]
    cfg.batch_size = int(best_row["batch_size"])
    return cfg


def run_phase3(
    domains: List[DomainName],
    data_root: Path,
    base_cfg: AdapterConfig,
    device: torch.device,
    csv_path: Path,
) -> None:
    # Preload features for all domains once.
    features: Dict[DomainName, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]] = {}
    for d in domains:
        features[d] = load_domain_features(d, data_root)

    rows: List[RunRow] = []

    for train_domain in domains:
        cfg = best_phase2_config_for_domain(train_domain, csv_path)
        if cfg is None:
            print(f"[G30][Phase3] No Phase2 config for {train_domain}; skipping.")
            continue

        tr_f, tr_l, _, _, _ = features[train_domain]
        adapter, train_time = train_adapter(tr_f, tr_l, cfg, device)
        adapter.eval()

        for eval_domain in domains:
            _, _, te_f, te_l, n_classes = features[eval_domain]
            acc, n_protos, cond = eval_full_stack(
                tr_f, tr_l, te_f, te_l, n_classes, adapter, device
            )
            print(
                f"[G30][Phase3] train={train_domain} eval={eval_domain} → "
                f"acc={acc:.2f}% protos={n_protos} cond={cond:.3f}",
                flush=True,
            )
            rows.append(
                RunRow(
                    phase=3,
                    domain=train_domain,
                    axis="eval_domain",
                    value=eval_domain,
                    accuracy=acc,
                    n_prototypes=n_protos,
                    condensation_ratio=cond,
                    training_time_s=train_time,
                    proj_dim=cfg.proj_dim,
                    nonlinearity=cfg.nonlinearity,
                    layers=cfg.layers,
                    loss_type=cfg.loss_type,
                    triplet_margin=cfg.triplet_margin,
                    infonce_temp=cfg.infonce_temp,
                    ms_alpha=cfg.ms_alpha,
                    ms_beta=cfg.ms_beta,
                    ms_base=cfg.ms_base,
                    lr=cfg.lr,
                    epochs=cfg.epochs,
                    mining_strategy=cfg.mining_strategy,
                    batch_size=cfg.batch_size,
                )
            )

    append_rows(csv_path, rows)


################################################################################
# CLI
################################################################################


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="G30: Adapter hyperparameter sweep (per-domain).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="Which phase to run (1: axis scan, 2: focused grid, 3: cross-domain).",
    )
    p.add_argument(
        "--domains",
        default="dogs,birds,cars,aircraft",
        help="Comma-separated subset of domains to run (dogs,birds,cars,aircraft).",
    )
    p.add_argument(
        "--data-root",
        default="data",
        help="Root directory containing domain feature caches.",
    )
    p.add_argument(
        "--output-dir",
        default="results",
        help="Directory to store CSV files.",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Torch device string or 'auto' to prefer CUDA.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = _device_from_str(args.device)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    csv_path = output_dir / "g30_adapter_sweep.csv"

    base_cfg = AdapterConfig()

    domains: List[DomainName] = [
        d.strip()
        for d in args.domains.split(",")
        if d.strip()
    ]  # type: ignore[assignment]

    if args.phase == 1:
        for d in domains:
            run_phase1_for_domain(d, data_root, base_cfg, device, csv_path)
    elif args.phase == 2:
        for d in domains:
            run_phase2_for_domain(d, data_root, base_cfg, device, csv_path)
    else:
        run_phase3(domains, data_root, base_cfg, device, csv_path)


if __name__ == "__main__":
    main()

