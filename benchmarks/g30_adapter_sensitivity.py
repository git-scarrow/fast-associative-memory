#!/usr/bin/env python3
"""
benchmarks/g30_adapter_sensitivity.py — G30: Adapter Hyperparameter Sweep (Per-Domain)

Phase 1: Axis sensitivity scan for domain-specific MetricAdapters on
fine-grained visual domains:

  * Dogs     (ImageNet G6 — Stanford Dogs subset, classes 151–200)
  * Birds    (CUB-200-2011)
  * Cars     (Stanford Cars)
  * Aircraft (FGVC Aircraft)

For each domain we:
  1. Load (or extract) frozen DINOv2 ViT-L/14 embeddings.
  2. Train a MetricAdapter on the train split under a given hyperparameter
     configuration.
  3. Evaluate the full stack (adapter + NSTP + FAM) on the in-domain test set
     at vigilance ``v=0.92``, recording:
        * Top-1 accuracy (%)
        * Number of prototypes
        * Condensation ratio (prototypes / n_train)
  4. Repeat while sweeping ONE axis at a time, holding all other axes at a
     domain-independent default configuration.

Axes
----
  1. proj_dim       ∈ {256, 512, 768}
  2. nonlinearity   ∈ {"none", "relu", "gelu"}
  3. layers         ∈ {1, 2}   (2 uses a residual MLP when proj_dim == hidden_dim)
  4. loss_type      ∈ {"triplet", "infonce"}
  5. margin         ∈ {0.1, 0.2, 0.4}   (triplet only)
  6. temperature    ∈ {0.05, 0.1, 0.2}  (InfoNCE only)
  7. lr             ∈ {1e-4, 5e-4, 1e-3}
  8. epochs         ∈ {5, 10, 20}
  9. mining         ∈ {"random", "semi-hard", "hard"}  (triplet only)
 10. batch_size     ∈ {32, 64, 128}

The resulting CSV file for each domain contains one row per run; this script
also prints an aggregated axis sensitivity table to stdout, ranking axes by
Δ (max-min) in-domain accuracy.

Phase 2/3 (focused grid + cross-domain regression) are *not* implemented here;
they should build on top of the raw CSV produced by this script.
"""

from __future__ import annotations

import argparse
import csv
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
# Domain feature loaders
################################################################################


def _device_from_str(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _load_dogs_features(
    root: str,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """ImageNet G6 (dog breeds) feature bank for the Dogs domain."""
    from torchvision import datasets, transforms

    cache_dir = Path(root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = cache_dir / "g30_dogs_train.pt"
    test_cache = cache_dir / "g30_dogs_test.pt"

    if train_cache.is_file() and test_cache.is_file():
        tr = torch.load(train_cache, map_location="cpu", weights_only=True)
        te = torch.load(test_cache, map_location="cpu", weights_only=True)
        n_classes = int(tr["labels"].max().item() + 1)
        return (
            tr["embeds"].float(),
            tr["labels"].long(),
            te["embeds"].float(),
            te["labels"].long(),
            n_classes,
        )

    tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    ds = datasets.ImageFolder(root, transform=tfm)

    start_cls, end_cls = 151, 200
    indices = [i for i, t in enumerate(ds.targets) if start_cls <= t <= end_cls]
    subset = torch.utils.data.Subset(ds, indices)

    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    class _DINOv2Extractor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitl14", verbose=False
            )
            self.model.eval()

        @torch.no_grad()
        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.model(x)

    extractor = _DINOv2Extractor().to(device)

    all_feats: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    with torch.no_grad():
        for imgs, lbls in loader:
            feats = extractor(imgs.to(device))
            all_feats.append(feats.cpu())
            # Relabel to 0..C-1 contiguous for stability.
            all_labels.append(lbls - start_cls)

    feats = torch.cat(all_feats, 0)
    labels = torch.cat(all_labels, 0)
    n = feats.shape[0]
    n_train = int(0.8 * n)

    perm = torch.randperm(n)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_embeds = feats[train_idx]
    train_labels = labels[train_idx]
    test_embeds = feats[test_idx]
    test_labels = labels[test_idx]

    torch.save(
        {"embeds": train_embeds, "labels": train_labels},
        train_cache,
    )
    torch.save(
        {"embeds": test_embeds, "labels": test_labels},
        test_cache,
    )

    n_classes = int(labels.max().item() + 1)
    return train_embeds, train_labels, test_embeds, test_labels, n_classes


def _load_birds_features(
    root: str,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """CUB-200-2011 embeddings via g22 helpers."""
    from benchmarks.g22_adapter_birds import load_or_extract_features

    tr_e, tr_l, te_e, te_l = load_or_extract_features(root, batch_size, device)
    n_classes = int(tr_l.max().item() + 1)
    return tr_e, tr_l, te_e, te_l, n_classes


def _load_cars_features(
    root: str,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Stanford Cars embeddings via g23 helpers."""
    from benchmarks.g23_adapter_cars import extract_and_cache

    tr_e, tr_l, te_e, te_l = extract_and_cache(root, batch_size, device)
    n_classes = int(tr_l.max().item() + 1)
    return tr_e, tr_l, te_e, te_l, n_classes


def _load_aircraft_features(
    root: str,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """FGVC Aircraft embeddings via g24 helpers."""
    from benchmarks.g24_adapter_aircraft import extract_or_load_features

    tr_e, tr_l, te_e, te_l = extract_or_load_features(root, batch_size, device)
    n_classes = int(tr_l.max().item() + 1)
    return tr_e, tr_l, te_e, te_l, n_classes


################################################################################
# Training utilities
################################################################################


LossType = Literal["triplet", "infonce"]
MiningStrategy = Literal["random", "semi-hard", "hard"]
Nonlinearity = Literal["none", "relu", "gelu"]


@dataclass
class AdapterConfig:
    proj_dim: int = 512
    nonlinearity: Nonlinearity = "none"
    layers: int = 1
    loss_type: LossType = "triplet"
    margin: float = 0.2
    temperature: float = 0.1
    lr: float = 5e-4
    epochs: int = 5
    mining: MiningStrategy = "random"
    batch_size: int = 64


def _make_adapter(input_dim: int, cfg: AdapterConfig) -> MetricAdapter:
    hidden_dim = cfg.proj_dim if cfg.layers == 2 else 0
    return MetricAdapter(
        input_dim=input_dim,
        output_dim=cfg.proj_dim,
        hidden_dim=hidden_dim,
        nonlinearity=cfg.nonlinearity,
        proj_dim=cfg.proj_dim,
        layers=cfg.layers if hidden_dim > 0 else 1,
        residual=(cfg.layers == 2),
    )


def _mine_triplets(
    proj: torch.Tensor,
    feats: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
    mining: MiningStrategy,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triplet mining for a mini-batch."""
    device = feats.device
    B = feats.size(0)
    anchors: List[torch.Tensor] = []
    positives: List[torch.Tensor] = []
    negatives: List[torch.Tensor] = []

    sim = proj @ proj.T  # (B, B)

    for i in range(B):
        lbl = labels[i]
        same_mask = labels == lbl
        diff_mask = labels != lbl

        same_idx = same_mask.nonzero(as_tuple=True)[0]
        same_idx = same_idx[same_idx != i]
        if same_idx.numel() == 0 or diff_mask.sum() == 0:
            continue

        pos_idx = same_idx[torch.randint(len(same_idx), (1,), device=device).item()]

        if mining == "random":
            diff_idx = diff_mask.nonzero(as_tuple=True)[0]
            neg_idx = diff_idx[
                torch.randint(len(diff_idx), (1,), device=device).item()
            ]
        elif mining == "hard":
            sim_row = sim[i].clone()
            sim_row[~diff_mask] = -2.0
            neg_idx = sim_row.argmax()
        else:  # semi-hard
            a = proj[i].unsqueeze(0)
            dists = torch.norm(a - proj, dim=1)
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
    a, p, n = _mine_triplets(proj, feats, labels, cfg.margin, cfg.mining)
    if a.numel() == 0:
        return torch.zeros((), device=feats.device)
    return adapter.triplet_loss(a, p, n, margin=cfg.margin)


def _info_nce_loss(
    adapter: MetricAdapter,
    feats: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Supervised InfoNCE over a mini-batch."""
    z = adapter(feats)
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


def train_adapter_on_embeddings(
    train_embeds: torch.Tensor,
    train_labels: torch.Tensor,
    cfg: AdapterConfig,
    device: torch.device,
) -> MetricAdapter:
    """Train a MetricAdapter on pre-extracted embeddings under a given config."""
    input_dim = train_embeds.size(1)
    adapter = _make_adapter(input_dim, cfg).to(device)

    ds = TensorDataset(train_embeds, train_labels)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    opt = torch.optim.Adam(adapter.parameters(), lr=cfg.lr)

    for epoch in range(1, cfg.epochs + 1):
        adapter.train()
        total_loss = 0.0
        n_batches = 0

        for feats, labels in loader:
            feats = feats.to(device)
            labels = labels.to(device)

            opt.zero_grad()
            if cfg.loss_type == "triplet":
                loss = _triplet_loss(adapter, feats, labels, cfg)
            else:
                loss = _info_nce_loss(adapter, feats, labels, cfg.temperature)

            if loss.item() == 0.0:
                continue

            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(
            f"    epoch {epoch}/{cfg.epochs} loss={avg_loss:.6f}",
            flush=True,
        )

    return adapter


@torch.no_grad()
def eval_full_stack(
    train_embeds: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeds: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    adapter: MetricAdapter,
    device: torch.device,
    vigilance: float = 0.92,
) -> Tuple[float, int, float]:
    """Evaluate adapter + NSTP + FAM at a fixed vigilance."""
    nstp = NSTPController(sibling_threshold=0.85, depth_epsilon=0.10)

    mem = FastAssociativeMemory(
        input_dim=train_embeds.size(1),
        value_dim=num_classes,
        core_entries=50000,
        core_vigilance=vigilance,
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=25,
        inference_temp=0.05,
        use_lfu=True,
        adapter=adapter,
        nstp=nstp,
    ).to(device)

    x_tr = train_embeds.to(device)
    y_tr = train_labels.to(device)
    for i in range(0, len(x_tr), 256):
        mem.learn_local(x_tr[i : i + 256], y_tr[i : i + 256])

    n_proto = int(mem.core_cam.occupied.sum().item())
    condensation = n_proto / float(len(train_embeds))

    correct = 0
    x_te = test_embeds.to(device)
    y_te = test_labels.to(device)
    for i in range(0, len(x_te), 512):
        logits = mem(x_te[i : i + 512])
        correct += (logits.argmax(dim=1) == y_te[i : i + 512]).sum().item()

    acc = 100.0 * correct / float(len(test_labels))
    return acc, n_proto, condensation


################################################################################
# Phase 1 axis sensitivity
################################################################################


AXES: Dict[str, Iterable] = {
    "proj_dim": [256, 512, 768],
    "nonlinearity": ["none", "relu", "gelu"],
    "layers": [1, 2],
    "loss_type": ["triplet", "infonce"],
    "margin": [0.1, 0.2, 0.4],
    "temperature": [0.05, 0.1, 0.2],
    "lr": [1e-4, 5e-4, 1e-3],
    "epochs": [5, 10, 20],
    "mining": ["random", "semi-hard", "hard"],
    "batch_size": [32, 64, 128],
}


@dataclass
class RunResult:
    domain: str
    axis: str
    value: str
    accuracy: float
    prototypes: int
    condensation: float
    proj_dim: int
    nonlinearity: str
    layers: int
    loss_type: str
    margin: float
    temperature: float
    lr: float
    epochs: int
    mining: str
    batch_size: int


def _sweep_axis(
    domain: str,
    axis: str,
    values: Iterable,
    base_cfg: AdapterConfig,
    train_embeds: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeds: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
) -> List[RunResult]:
    results: List[RunResult] = []

    for v in values:
        cfg = AdapterConfig(**asdict(base_cfg))
        setattr(cfg, axis, v)

        if axis in ("margin", "mining") and cfg.loss_type != "triplet":
            continue
        if axis == "temperature" and cfg.loss_type != "infonce":
            continue

        print(f"\n[{domain}] Axis={axis} Value={v} — training adapter …", flush=True)
        adapter = train_adapter_on_embeddings(train_embeds, train_labels, cfg, device)
        adapter.eval()
        acc, n_proto, cond = eval_full_stack(
            train_embeds,
            train_labels,
            test_embeds,
            test_labels,
            num_classes=num_classes,
            adapter=adapter,
            device=device,
            vigilance=0.92,
        )
        print(
            f"[{domain}] Axis={axis} Value={v} → acc={acc:.2f}%, "
            f"protos={n_proto}, cond={cond:.3f}",
            flush=True,
        )

        results.append(
            RunResult(
                domain=domain,
                axis=axis,
                value=str(v),
                accuracy=acc,
                prototypes=n_proto,
                condensation=cond,
                proj_dim=cfg.proj_dim,
                nonlinearity=cfg.nonlinearity,
                layers=cfg.layers,
                loss_type=cfg.loss_type,
                margin=cfg.margin,
                temperature=cfg.temperature,
                lr=cfg.lr,
                epochs=cfg.epochs,
                mining=cfg.mining,
                batch_size=cfg.batch_size,
            )
        )

    return results


def _aggregate_sensitivity(domain: str, rows: List[RunResult]) -> None:
    print(f"\n\n=== G30 Phase 1 — Axis Sensitivity ({domain}) ===")
    print(
        "┌────────────┬──────────────┬─────────┬─────────┬─────────┬───────────┐\n"
        "│ Domain     │ Axis         │ Value 1 │ Value 2 │ Value 3 │ Δ max-min │\n"
        "├────────────┼──────────────┼─────────┼─────────┼─────────┼───────────┤"
    )

    for axis, _values in AXES.items():
        vals = sorted({r.value for r in rows if r.axis == axis})
        if not vals:
            continue
        accs = {val: None for val in vals}
        for r in rows:
            if r.axis == axis:
                accs[r.value] = r.accuracy

        ordered_vals = list(vals)[:3]
        acc_list = [accs[v] for v in ordered_vals]
        finite_accs = [a for a in acc_list if a is not None]
        if finite_accs:
            delta = max(finite_accs) - min(finite_accs)
        else:
            delta = 0.0

        def fmt(a: float | None) -> str:
            return f"{a:.2f}%" if a is not None else "   N/A "

        v1 = fmt(acc_list[0]) if len(acc_list) > 0 else "   N/A "
        v2 = fmt(acc_list[1]) if len(acc_list) > 1 else "   N/A "
        v3 = fmt(acc_list[2]) if len(acc_list) > 2 else "   N/A "

        print(
            f"│ {domain:<10} │ {axis:<12} │ {v1:>7} │ {v2:>7} │ {v3:>7} │ "
            f"{delta:>6.2f}pp │"
        )

    print(
        "└────────────┴──────────────┴─────────┴─────────┴─────────┴───────────┘"
    )


def run_phase1(
    domain: str,
    root: str,
    base_cfg: AdapterConfig,
    device: torch.device,
    output_csv: Path,
) -> None:
    if domain == "dogs":
        loader = _load_dogs_features
    elif domain == "birds":
        loader = _load_birds_features
    elif domain == "cars":
        loader = _load_cars_features
    elif domain == "aircraft":
        loader = _load_aircraft_features
    else:
        raise ValueError(f"Unknown domain {domain!r}")

    print(f"[G30] Loading {domain} features from {root!r} …", flush=True)
    tr_e, tr_l, te_e, te_l, n_classes = loader(root, base_cfg.batch_size, device)
    print(
        f"[G30] {domain}: train={tr_e.shape[0]} test={te_e.shape[0]} "
        f"classes={n_classes}",
        flush=True,
    )

    all_results: List[RunResult] = []
    for axis, values in AXES.items():
        all_results.extend(
            _sweep_axis(
                domain,
                axis,
                values,
                base_cfg,
                tr_e,
                tr_l,
                te_e,
                te_l,
                n_classes,
                device,
            )
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "domain",
                "axis",
                "value",
                "accuracy",
                "prototypes",
                "condensation",
                "proj_dim",
                "nonlinearity",
                "layers",
                "loss_type",
                "margin",
                "temperature",
                "lr",
                "epochs",
                "mining",
                "batch_size",
            ],
        )
        writer.writeheader()
        for r in all_results:
            writer.writerow(
                {
                    "domain": r.domain,
                    "axis": r.axis,
                    "value": r.value,
                    "accuracy": r.accuracy,
                    "prototypes": r.prototypes,
                    "condensation": r.condensation,
                    "proj_dim": r.proj_dim,
                    "nonlinearity": r.nonlinearity,
                    "layers": r.layers,
                    "loss_type": r.loss_type,
                    "margin": r.margin,
                    "temperature": r.temperature,
                    "lr": r.lr,
                    "epochs": r.epochs,
                    "mining": r.mining,
                    "batch_size": r.batch_size,
                }
            )

    _aggregate_sensitivity(domain, all_results)


################################################################################
# CLI
################################################################################


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="G30 Phase 1: Adapter hyperparameter axis sensitivity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--domain",
        choices=["dogs", "birds", "cars", "aircraft"],
        required=True,
        help="Fine-grained domain to sweep.",
    )
    p.add_argument(
        "--root",
        required=True,
        help="Data root for the chosen domain.",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Torch device string or 'auto' to prefer CUDA.",
    )
    p.add_argument(
        "--output-csv",
        default=None,
        help="Path to write raw CSV results. "
        "Defaults to results/g30_phase1_<domain>.csv under the repo root.",
    )
    p.add_argument("--base-proj-dim", type=int, default=512)
    p.add_argument(
        "--base-loss-type", choices=["triplet", "infonce"], default="triplet"
    )
    p.add_argument("--base-margin", type=float, default=0.2)
    p.add_argument("--base-temperature", type=float, default=0.1)
    p.add_argument("--base-lr", type=float, default=5e-4)
    p.add_argument("--base-epochs", type=int, default=5)
    p.add_argument(
        "--base-mining",
        choices=["random", "semi-hard", "hard"],
        default="random",
    )
    p.add_argument("--base-batch-size", type=int, default=64)
    p.add_argument(
        "--base-nonlinearity",
        choices=["none", "relu", "gelu"],
        default="none",
    )
    p.add_argument("--base-layers", type=int, choices=[1, 2], default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = _device_from_str(args.device)

    base_cfg = AdapterConfig(
        proj_dim=args.base_proj_dim,
        nonlinearity=args.base_nonlinearity,
        layers=args.base_layers,
        loss_type=args.base_loss_type,  # type: ignore[arg-type]
        margin=args.base_margin,
        temperature=args.base_temperature,
        lr=args.base_lr,
        epochs=args.base_epochs,
        mining=args.base_mining,  # type: ignore[arg-type]
        batch_size=args.base_batch_size,
    )

    if args.output_csv is not None:
        output_csv = Path(args.output_csv)
    else:
        repo_root = Path(__file__).parent.parent
        output_csv = repo_root / "results" / f"g30_phase1_{args.domain}.csv"

    run_phase1(
        domain=args.domain,
        root=args.root,
        base_cfg=base_cfg,
        device=device,
        output_csv=output_csv,
    )


if __name__ == "__main__":
    main()

