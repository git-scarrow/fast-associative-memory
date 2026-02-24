#!/usr/bin/env python3
"""
benchmarks/g23_adapter_cars.py — Gauntlet 23: Adapter Sweep on Stanford Cars.

Self-contained training + evaluation script that:
1. Downloads Stanford Cars (196 classes, ~8.1k train / ~8.0k test images).
2. Extracts DINOv2 ViT-L/14 embeddings (1024d) for all images.
3. Trains a domain-specific MetricAdapter on Cars train embeddings using OHNM
   triplet loss.
4. Evaluates the full stack (adapter + NSTP + vigilance sweep) on the Cars test
   set.
5. Checks the kill condition: Cars best accuracy vs G11 Dogs reference (99.71%).

Usage
-----
    python benchmarks/g23_adapter_cars.py [--root DATA_DIR] [--epochs 5]
        [--batch-size 128] [--lr 1e-4] [--margin 0.2] [--device auto]

CLI args
--------
  --root         Data directory for Stanford Cars download/cache.
                 (default: ./data/stanford_cars)
  --epochs       Number of training epochs.  (default: 5)
  --batch-size   Mini-batch size.  (default: 128)
  --lr           Adam learning rate.  (default: 1e-4)
  --margin       Triplet margin.  (default: 0.2)
  --device       Torch device string, e.g. ``cuda`` or ``cpu``.
                 ``auto`` selects CUDA when available.  (default: auto)

Kill Condition (G23)
--------------------
In-domain top-1 accuracy on Stanford Cars with Cars-trained adapter is lower
than the G11 Dogs adapter reference (99.71%) → **G23 Kill Condition Met**.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T
from torchvision.datasets import StanfordCars

# Allow importing from the repo root regardless of the working directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapter import MetricAdapter  # noqa: E402
from fast_associative_memory import FastAssociativeMemory  # noqa: E402
from nstp import NSTPController  # noqa: E402

# ─────────────────────────────────────────────── constants ───

G11_DOGS_REFERENCE_ACC = 99.71  # G11 reference accuracy (%)
VIGILANCE_SWEEP = [0.92, 0.85, 0.82, 0.80]
EMBED_DIM = 1024        # DINOv2 ViT-L/14 output dimension
ADAPTER_OUTPUT_DIM = 256
ADAPTER_HIDDEN_DIM = 512
TRAIN_CACHE_NAME = "stanford_cars_train.pt"
TEST_CACHE_NAME = "stanford_cars_test.pt"
ADAPTER_WEIGHTS_NAME = "adapter_stanford_cars.pt"


# ─────────────────────────────────────────────── data ───

def _build_transform() -> T.Compose:
    return T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def _load_dinov2(device: torch.device) -> nn.Module:
    """Load DINOv2 ViT-L/14 backbone (frozen)."""
    model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitl14", verbose=False
    )
    model = model.to(device).eval()
    return model


@torch.no_grad()
def _extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract DINOv2 features for all images in *loader*."""
    all_embeds, all_labels = [], []
    for images, labels in loader:
        embeds = model(images.to(device))
        all_embeds.append(embeds.cpu())
        all_labels.append(labels)
    return torch.cat(all_embeds), torch.cat(all_labels)


def extract_and_cache(
    root: str,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Download Stanford Cars, extract DINOv2 features, and cache to disk.

    Cached files are written next to the data root directory.  If cache files
    already exist they are loaded directly without re-extracting.

    Returns
    -------
    train_embeds, train_labels, test_embeds, test_labels
    """
    cache_dir = Path(root)
    train_cache = cache_dir / TRAIN_CACHE_NAME
    test_cache = cache_dir / TEST_CACHE_NAME

    if train_cache.is_file() and test_cache.is_file():
        print("[G23] Loading cached embeddings …")
        train_data = torch.load(train_cache, weights_only=True)
        test_data = torch.load(test_cache, weights_only=True)
        return (
            train_data["embeds"].float(), train_data["labels"].long(),
            test_data["embeds"].float(), test_data["labels"].long(),
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    transform = _build_transform()

    print("[G23] Loading Stanford Cars dataset …")
    train_ds = StanfordCars(root=root, split="train", transform=transform,
                            download=True)
    test_ds = StanfordCars(root=root, split="test", transform=transform,
                           download=True)
    print(f"[G23]   Train: {len(train_ds)} images, Test: {len(test_ds)} images, "
          f"{len(train_ds.classes)} classes")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4,
        pin_memory=True,
    )

    print("[G23] Loading DINOv2 ViT-L/14 …")
    model = _load_dinov2(device)

    print("[G23] Extracting train features …")
    train_embeds, train_labels = _extract_features(model, train_loader, device)
    print(f"[G23]   train embeds: {train_embeds.shape}")

    print("[G23] Extracting test features …")
    test_embeds, test_labels = _extract_features(model, test_loader, device)
    print(f"[G23]   test embeds: {test_embeds.shape}")

    torch.save({"embeds": train_embeds.float(), "labels": train_labels.long()},
               train_cache)
    torch.save({"embeds": test_embeds.float(), "labels": test_labels.long()},
               test_cache)
    print(f"[G23] Cached embeddings to {cache_dir}/")

    return (
        train_embeds.float(), train_labels.long(),
        test_embeds.float(), test_labels.long(),
    )


# ─────────────────────────────────────────────── training ───

def _mine_hard_negatives(
    anchor_proj: torch.Tensor,
    batch_feats: torch.Tensor,
    batch_labels: torch.Tensor,
    anchor_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Online Hard Negative Mining over a mini-batch.

    For each anchor selects:
    * **Positive** — a randomly chosen same-class sample (≠ anchor itself).
    * **Hard negative** — the non-same-class sample with the highest cosine
      similarity to the *projected* anchor embedding.

    Returns
    -------
    Tuple of ``(anchors, positives, negatives)`` raw feature tensors.
    """
    anchors_out, positives_out, negatives_out = [], [], []
    sim = anchor_proj @ anchor_proj.T  # (B, B) — anchor_proj already L2-normed

    for i, lbl in enumerate(anchor_labels):
        same_mask = batch_labels == lbl
        diff_mask = batch_labels != lbl

        same_indices = same_mask.nonzero(as_tuple=True)[0]
        same_indices = same_indices[same_indices != i]
        if len(same_indices) == 0 or diff_mask.sum() == 0:
            continue

        pos_idx = same_indices[torch.randint(len(same_indices), (1,)).item()]

        sim_row = sim[i].clone()
        sim_row[~diff_mask] = -2.0
        neg_idx = sim_row.argmax()

        anchors_out.append(batch_feats[i])
        positives_out.append(batch_feats[pos_idx])
        negatives_out.append(batch_feats[neg_idx])

    if not anchors_out:
        empty = torch.empty(0, batch_feats.shape[1], device=batch_feats.device)
        return empty, empty, empty

    return (
        torch.stack(anchors_out),
        torch.stack(positives_out),
        torch.stack(negatives_out),
    )


def train_adapter(
    train_embeds: torch.Tensor,
    train_labels: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    margin: float,
    output: str,
    device: torch.device,
) -> MetricAdapter:
    """Train MetricAdapter on Stanford Cars embeddings using OHNM triplet loss.

    Parameters
    ----------
    train_embeds:  Pre-extracted DINOv2 features, shape ``(N, 1024)``.
    train_labels:  Class labels, shape ``(N,)``.
    epochs:        Number of training epochs.
    batch_size:    Mini-batch size.
    lr:            Adam learning rate.
    margin:        Triplet margin.
    output:        File path to save adapter weights.
    device:        Compute device.

    Returns
    -------
    Trained :class:`~adapter.MetricAdapter`.
    """
    input_dim = train_embeds.size(1)
    adapter = MetricAdapter(
        input_dim=input_dim,
        output_dim=ADAPTER_OUTPUT_DIM,
        hidden_dim=ADAPTER_HIDDEN_DIM,
    ).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)

    feat_ds = TensorDataset(train_embeds, train_labels)
    feat_loader = DataLoader(feat_ds, batch_size=batch_size, shuffle=True)

    print(
        f"\n[G23] Training adapter: {epochs} epoch(s), batch={batch_size},"
        f" lr={lr}, margin={margin}"
    )
    for epoch in range(1, epochs + 1):
        adapter.train()
        total_loss = 0.0
        n_batches = 0

        for batch_feats, batch_labels in feat_loader:
            batch_feats = batch_feats.to(device)
            batch_labels = batch_labels.to(device)

            with torch.no_grad():
                proj = adapter(batch_feats)  # (B, 256) L2-normalised

            anch, pos, neg = _mine_hard_negatives(
                proj, batch_feats, batch_labels, batch_labels
            )
            if anch.shape[0] == 0:
                continue

            optimizer.zero_grad()
            loss = adapter.triplet_loss(anch, pos, neg, margin=margin)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch}/{epochs}  loss={avg_loss:.6f}")

    torch.save(adapter.state_dict(), output)
    print(f"\n[G23] Adapter saved → {output}")

    return adapter


# ─────────────────────────────────────────────── evaluation ───

@torch.no_grad()
def eval_full_stack(
    train_embeds: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeds: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
    num_classes: int,
    adapter: MetricAdapter | None,
    vigilance: float,
) -> tuple[float, int, float]:
    """Evaluate full stack (adapter + NSTP) at a given vigilance level.

    Parameters
    ----------
    adapter:    :class:`~adapter.MetricAdapter` or ``None`` for baseline.
    vigilance:  Vigilance threshold for the continuous CAM.

    Returns
    -------
    accuracy : float
        Top-1 accuracy in %.
    n_prototypes : int
        Number of stored prototypes after ingestion.
    condensation_ratio : float
        ``n_prototypes / len(train_embeds)``.
    """
    embed_dim = train_embeds.size(1)
    nstp = NSTPController(sibling_threshold=0.85, depth_epsilon=0.10)

    mem = FastAssociativeMemory(
        input_dim=embed_dim,
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

    x_train = train_embeds.to(device)
    y_train = train_labels.to(device)
    for i in range(0, len(x_train), 256):
        mem.learn_local(x_train[i:i + 256], y_train[i:i + 256])

    n_prototypes = int(mem.core_cam.occupied.sum().item())
    n_train = len(train_embeds)
    condensation_ratio = n_prototypes / n_train

    correct = 0
    x_test = test_embeds.to(device)
    y_test = test_labels.to(device)
    for i in range(0, len(x_test), 512):
        logits = mem(x_test[i:i + 512])
        correct += (logits.argmax(dim=1) == y_test[i:i + 512]).sum().item()

    accuracy = 100.0 * correct / len(test_labels)
    return accuracy, n_prototypes, condensation_ratio


# ─────────────────────────────────────────────── reporting ───

def _print_table(rows: list[tuple[str, float, int, float]]) -> None:
    sep = "=" * 67
    print(f"\nG23 — Adapter Sweep: Stanford Cars")
    print(sep)
    print(f"  {'Config':<36} {'Accuracy':>9}  {'Protos':>6}  {'Cond.Ratio':>10}")
    print(f"  {'-' * 63}")
    for config, acc, protos, cond in rows:
        print(f"  {config:<36} {acc:>8.2f}%  {protos:>6}  {cond:>10.2f}")
    print(sep)


def _check_kill_condition(best_cars_acc: float) -> bool:
    """Return ``True`` when kill condition is met (Cars < Dogs reference)."""
    return best_cars_acc < G11_DOGS_REFERENCE_ACC


# ─────────────────────────────────────────────── main ───

def run(
    root: str,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-4,
    margin: float = 0.2,
    device_str: str = "auto",
) -> None:
    """Full G23 pipeline: extract → train → evaluate → report."""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[G23] Device: {device}")

    # ── 1. Extract / load embeddings ──────────────────────────────────────────
    train_embeds, train_labels, test_embeds, test_labels = extract_and_cache(
        root, batch_size, device
    )
    num_classes = int(train_labels.max().item()) + 1
    print(
        f"[G23] num_classes={num_classes}, "
        f"train={len(train_embeds)}, test={len(test_embeds)}"
    )

    # ── 2. Train adapter ──────────────────────────────────────────────────────
    adapter_path = os.path.join(root, ADAPTER_WEIGHTS_NAME)
    adapter = train_adapter(
        train_embeds=train_embeds,
        train_labels=train_labels,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        margin=margin,
        output=adapter_path,
        device=device,
    )
    adapter.eval()

    # ── 3. Evaluate ────────────────────────────────────────────────────────────
    print("\n[G23] Evaluating full stack …")
    rows: list[tuple[str, float, int, float]] = []

    # Baseline: no adapter, v=0.92
    print("  [baseline] no adapter, v=0.92 …")
    acc, protos, cond = eval_full_stack(
        train_embeds, train_labels, test_embeds, test_labels,
        device=device, num_classes=num_classes,
        adapter=None, vigilance=0.92,
    )
    rows.append(("baseline (no adapter, v=0.92)", acc, protos, cond))

    # Cars adapter at each vigilance level
    for v in VIGILANCE_SWEEP:
        label = f"cars adapter (v={v:.2f})"
        print(f"  [{label}] …")
        acc, protos, cond = eval_full_stack(
            train_embeds, train_labels, test_embeds, test_labels,
            device=device, num_classes=num_classes,
            adapter=adapter, vigilance=v,
        )
        rows.append((label, acc, protos, cond))

    # ── 4. Report ──────────────────────────────────────────────────────────────
    _print_table(rows)

    # Best accuracy achieved by Cars adapter (skip baseline row at index 0)
    adapter_rows = rows[1:]
    best_acc = max(r[1] for r in adapter_rows)

    kill = _check_kill_condition(best_acc)
    verdict = "KILL" if kill else "PASS"
    print(
        f"\nKill condition: Cars best accuracy {best_acc:.2f}%"
        f" vs Dogs reference {G11_DOGS_REFERENCE_ACC:.2f}%"
    )
    print(f"→ {verdict}")


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G23 — Adapter Sweep: Stanford Cars",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        default="./data/stanford_cars",
        metavar="DIR",
        help="Data directory for Stanford Cars download/cache.",
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--margin", type=float, default=0.2,
        help="Triplet margin.",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Torch device string, e.g. cuda or cpu (auto = detect).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    run(
        root=args.root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        margin=args.margin,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
