#!/usr/bin/env python3
"""
benchmarks/g25_adapter_flowers.py — Gauntlet 25: Adapter Sweep on Oxford Flowers-102.

Downloads Oxford Flowers-102, extracts DINOv2 ViT-L/14 embeddings, trains a
domain-specific MetricAdapter with Online Hard Negative Mining (OHNM) triplet
loss, then evaluates the full stack (adapter + NSTP + vigilance sweep) against
the G11 Dogs-adapter reference baseline.

Protocol
--------
1. Download Flowers-102 via ``torchvision.datasets.Flowers102``; combine train
   and val splits for training, use test split for evaluation.
2. Extract DINOv2 ViT-L/14 embeddings (1024-d); cache to
   ``flowers102_train.pt`` / ``flowers102_test.pt``.
3. Train ``MetricAdapter(input_dim=1024, output_dim=256, hidden_dim=512)`` with
   OHNM triplet loss (5 epochs, batch_size=128, lr=1e-4, margin=0.2).
4. Evaluate full stack (adapter + NSTP, vigilance ∈ {0.92, 0.85, 0.82, 0.80})
   plus a no-adapter baseline (v=0.92) on the Flowers-102 test set.
5. Kill condition: Flowers best top-1 accuracy < G11 Dogs reference (99.71%).

Kill Condition (G25)
--------------------
Flowers best top-1 accuracy < 99.71% → **G25 Kill Condition Met**.

Usage
-----
    python benchmarks/g25_adapter_flowers.py
    python benchmarks/g25_adapter_flowers.py --root ./data/flowers102 --epochs 5

CLI args
--------
  --root        Data directory for Flowers-102 download/cache.
                (default: ./data/flowers102)
  --epochs      Number of training epochs.  (default: 5)
  --batch-size  Mini-batch size.  (default: 128)
  --lr          Adam learning rate.  (default: 1e-4)
  --margin      Triplet margin.  (default: 0.2)
  --device      Torch device string, e.g. ``cuda`` or ``cpu``.  Auto-detected
                when omitted or ``auto``.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

# Allow importing from the repo root regardless of the working directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapter import MetricAdapter  # noqa: E402
from fast_associative_memory import FastAssociativeMemory  # noqa: E402
from nstp import NSTPController  # noqa: E402

# ─────────────────────────────────────────────────── constants ───

_DOGS_REFERENCE_ACC = 99.71   # G11 reference: Dogs adapter top-1 accuracy (%)
_VIGILANCE_SWEEP = [0.92, 0.85, 0.82, 0.80]
_NUM_CLASSES = 102             # Oxford Flowers-102


# ─────────────────────────────────────────── DINOv2 feature extraction ───

class _DINOv2Extractor(nn.Module):
    """Thin wrapper around DINOv2 ViT-L/14 (1024-dim output)."""

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitl14", verbose=False
        )
        self.model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)   # (B, 1024)


def _build_transform():
    from torchvision import transforms as T
    return T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


@torch.no_grad()
def _extract(model: nn.Module, loader: DataLoader, device: torch.device
             ) -> tuple[torch.Tensor, torch.Tensor]:
    all_embeds, all_labels = [], []
    for images, labels in loader:
        embeds = model(images.to(device))
        all_embeds.append(embeds.cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_embeds), torch.cat(all_labels)


def extract_flowers_features(
    root: str,
    cache_dir: str,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Download Flowers-102 and extract DINOv2 ViT-L/14 embeddings.

    Combines the official *train* and *val* splits into a single training set.
    Uses the official *test* split for evaluation.  Embedding files are cached
    to ``cache_dir/flowers102_train.pt`` and ``cache_dir/flowers102_test.pt``
    so subsequent runs skip the extraction step.

    Parameters
    ----------
    root:       Directory where Flowers-102 data is downloaded.
    cache_dir:  Directory where ``.pt`` embedding caches are stored.
    batch_size: DataLoader batch size for feature extraction.
    device:     Compute device for the DINOv2 backbone.

    Returns
    -------
    (train_embeds, train_labels, test_embeds, test_labels)
    """
    from torchvision.datasets import Flowers102

    train_cache = Path(cache_dir) / "flowers102_train.pt"
    test_cache = Path(cache_dir) / "flowers102_test.pt"

    if train_cache.is_file() and test_cache.is_file():
        print("[G25] Loading cached embeddings …")
        tr = torch.load(train_cache, weights_only=True)
        te = torch.load(test_cache, weights_only=True)
        return (
            tr["embeds"].float(), tr["labels"].long(),
            te["embeds"].float(), te["labels"].long(),
        )

    print("[G25] Downloading / verifying Flowers-102 …")
    tfm = _build_transform()
    ds_train = Flowers102(root=root, split="train", transform=tfm, download=True)
    ds_val   = Flowers102(root=root, split="val",   transform=tfm, download=True)
    ds_test  = Flowers102(root=root, split="test",  transform=tfm, download=True)

    # Combine train + val for a larger training pool.
    ds_trainval = ConcatDataset([ds_train, ds_val])

    loader_tv = DataLoader(
        ds_trainval, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    loader_te = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    print("[G25] Loading DINOv2 ViT-L/14 …")
    extractor = _DINOv2Extractor().to(device)

    print("[G25] Extracting train+val embeddings …")
    train_embeds, train_labels = _extract(extractor, loader_tv, device)
    print(f"[G25]   train+val: {train_embeds.shape}")

    print("[G25] Extracting test embeddings …")
    test_embeds, test_labels = _extract(extractor, loader_te, device)
    print(f"[G25]   test: {test_embeds.shape}")

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        {"embeds": train_embeds.float(), "labels": train_labels.long()},
        train_cache,
    )
    torch.save(
        {"embeds": test_embeds.float(), "labels": test_labels.long()},
        test_cache,
    )
    print(f"[G25] Embeddings cached → {cache_dir}/")

    return train_embeds.float(), train_labels.long(), test_embeds.float(), test_labels.long()


# ──────────────────────────────── Online Hard Negative Mining (OHNM) ───

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

    Triplets where no valid positive or negative can be found are silently
    skipped so the caller can handle empty returns gracefully.
    """
    anchors_out, positives_out, negatives_out = [], [], []

    sim = anchor_proj @ anchor_proj.T   # (B, B)

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


# ─────────────────────────────────────────────── adapter training ───

def train_adapter(
    train_embeds: torch.Tensor,
    train_labels: torch.Tensor,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-4,
    margin: float = 0.2,
    output: str = "adapter_flowers102.pt",
    device: torch.device | None = None,
) -> MetricAdapter:
    """Train a MetricAdapter on Flowers-102 embeddings using OHNM triplet loss.

    Parameters
    ----------
    train_embeds:  Pre-extracted training embeddings, shape ``(N, 1024)``.
    train_labels:  Corresponding class labels, shape ``(N,)``.
    epochs:        Number of training epochs.
    batch_size:    Mini-batch size.
    lr:            Adam learning rate.
    margin:        Triplet margin.
    output:        File path to save the trained adapter state-dict.
    device:        Compute device.  Auto-detected when ``None``.

    Returns
    -------
    The trained :class:`~adapter.MetricAdapter`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adapter = MetricAdapter(input_dim=1024, output_dim=256, hidden_dim=512)
    adapter.to(device)

    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)

    feat_ds = TensorDataset(train_embeds, train_labels)
    feat_loader = DataLoader(feat_ds, batch_size=batch_size, shuffle=True)

    print(
        f"\n[G25] Training MetricAdapter: {epochs} epoch(s),"
        f" batch={batch_size}, lr={lr}, margin={margin}"
    )
    for epoch in range(1, epochs + 1):
        adapter.train()
        total_loss = 0.0
        n_batches = 0

        for batch_feats, batch_labels in feat_loader:
            batch_feats  = batch_feats.to(device)
            batch_labels = batch_labels.to(device)

            with torch.no_grad():
                proj = adapter(batch_feats)   # (B, 256) L2-normalised

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
    print(f"\n[G25] Trained adapter saved → {output}")
    return adapter


# ───────────────────────────────────────────────────── evaluation ───

@torch.no_grad()
def eval_full_stack(
    train_embeds: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeds: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
    adapter: MetricAdapter | None,
    vigilance: float,
    core_entries: int = 10000,
) -> tuple[float, int, float]:
    """Ingest training embeddings into FAM and evaluate top-1 accuracy.

    Parameters
    ----------
    train_embeds, train_labels: Training set tensors.
    test_embeds, test_labels:   Test set tensors.
    device:         Compute device.
    adapter:        Trained :class:`~adapter.MetricAdapter`, or ``None`` for
                    the no-adapter baseline.
    vigilance:      FAM vigilance threshold.
    core_entries:   FAM prototype capacity.

    Returns
    -------
    (accuracy_pct, num_prototypes, condensation_ratio)
    """
    n_train = len(train_labels)
    nstp = NSTPController(sibling_threshold=0.85, depth_epsilon=0.10)

    mem = FastAssociativeMemory(
        input_dim=1024,
        value_dim=_NUM_CLASSES,
        core_entries=core_entries,
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

    correct = 0
    x_test = test_embeds.to(device)
    y_test = test_labels.to(device)
    for i in range(0, len(x_test), 512):
        logits = mem(x_test[i:i + 512])
        correct += (logits.argmax(dim=1) == y_test[i:i + 512]).sum().item()

    num_protos = int(mem.core_cam.occupied.sum().item())
    acc = 100.0 * correct / len(test_labels)
    cond_ratio = num_protos / n_train
    return acc, num_protos, cond_ratio


# ────────────────────────────────────────── kill-condition helpers ───

def check_kill_condition(flowers_acc: float, dogs_ref: float = _DOGS_REFERENCE_ACC) -> bool:
    """Return ``True`` when the kill condition is met.

    Kill condition: Flowers best top-1 accuracy is strictly less than the Dogs
    adapter reference (99.71%).  Exactly meeting the threshold = pass.

    Parameters
    ----------
    flowers_acc:  Best Flowers-102 adapter top-1 accuracy (%).
    dogs_ref:     G11 Dogs-adapter reference accuracy (%).  Default 99.71%.

    Returns
    -------
    bool
        ``True`` if kill condition is met (flowers_acc < dogs_ref).
    """
    return flowers_acc < dogs_ref


# ──────────────────────────────────────────────── output formatting ───

def _format_report(
    rows: list[tuple[str, float, int, float]],
    flowers_best_acc: float,
    dogs_ref: float,
) -> str:
    """Return the formatted results table as a string."""
    sep = "=" * 67
    lines = [
        "\nG25 \u2014 Adapter Sweep: Oxford Flowers-102",
        sep,
        f"  {'Config':<38} {'Accuracy':>10} {'Protos':>7} {'Cond.Ratio':>10}",
        f"  {'-' * 63}",
    ]
    for config, acc, protos, cond in rows:
        lines.append(
            f"  {config:<38} {acc:>9.2f}% {protos:>7d} {cond:>10.2f}"
        )
    lines.append(sep)

    killed = check_kill_condition(flowers_best_acc, dogs_ref)
    verdict = "KILL" if killed else "PASS"
    lines.append(
        f"\nKill condition: Flowers best accuracy {flowers_best_acc:.2f}%"
        f" vs Dogs reference {dogs_ref:.2f}%"
    )
    lines.append(f"\u2192 {verdict}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────── main ───

def run(
    root: str = "./data/flowers102",
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-4,
    margin: float = 0.2,
    device_str: str | None = None,
) -> dict:
    """End-to-end Gauntlet 25 run: extract → train → evaluate → report.

    Parameters
    ----------
    root:        Data directory for Flowers-102 download and embedding cache.
    epochs:      Adapter training epochs.
    batch_size:  Mini-batch size (extraction + training).
    lr:          Adam learning rate.
    margin:      Triplet margin.
    device_str:  Torch device string or ``None`` / ``"auto"`` for auto-detect.

    Returns
    -------
    dict with keys:
        ``rows``              — list of (config, acc, protos, cond_ratio) tuples
        ``flowers_best_acc``  — best Flowers adapter top-1 accuracy (%)
        ``kill``              — bool, True if kill condition met
    """
    if device_str is None or device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"[G25] Device: {device}")

    cache_dir = os.path.join(root, "feature_cache")

    # ── Feature extraction ────────────────────────────────────────────────
    train_embeds, train_labels, test_embeds, test_labels = extract_flowers_features(
        root=root,
        cache_dir=cache_dir,
        batch_size=batch_size,
        device=device,
    )
    print(
        f"[G25] Train: {train_embeds.shape}  Test: {test_embeds.shape}"
        f"  Classes: {_NUM_CLASSES}"
    )

    # ── Adapter training ─────────────────────────────────────────────────
    adapter_path = os.path.join(root, "adapter_flowers102.pt")
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

    # ── Evaluation ────────────────────────────────────────────────────────
    rows: list[tuple[str, float, int, float]] = []

    # Baseline: no adapter, v=0.92
    print("\n[G25] Evaluating baseline (no adapter, v=0.92) …")
    acc, protos, cond = eval_full_stack(
        train_embeds, train_labels, test_embeds, test_labels,
        device=device, adapter=None, vigilance=0.92,
    )
    rows.append(("baseline (no adapter, v=0.92)", acc, protos, cond))

    # Flowers adapter at each vigilance level
    flowers_accs = []
    for v in _VIGILANCE_SWEEP:
        print(f"[G25] Evaluating flowers adapter (v={v}) …")
        acc, protos, cond = eval_full_stack(
            train_embeds, train_labels, test_embeds, test_labels,
            device=device, adapter=adapter, vigilance=v,
        )
        rows.append((f"flowers adapter (v={v})", acc, protos, cond))
        flowers_accs.append(acc)

    flowers_best_acc = max(flowers_accs)

    # ── Report ────────────────────────────────────────────────────────────
    report = _format_report(rows, flowers_best_acc, _DOGS_REFERENCE_ACC)
    print(report)

    return {
        "rows": rows,
        "flowers_best_acc": flowers_best_acc,
        "kill": check_kill_condition(flowers_best_acc, _DOGS_REFERENCE_ACC),
    }


# ─────────────────────────────────────────────────────────── CLI ───

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="G25 — Adapter Sweep: Oxford Flowers-102",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        default="./data/flowers102",
        metavar="DIR",
        help="Data directory for Flowers-102 download/cache.",
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Number of adapter training epochs.",
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
        help="Torch device string, e.g. cuda or cpu (default: auto-detect).",
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
