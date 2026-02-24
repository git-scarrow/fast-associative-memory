"""
train_adapter_triplet.py — G11: Train MetricAdapter with Online Hard Negative Mining.

Trains a :class:`~adapter.MetricAdapter` on the ImageNet G6 subset (dog breeds,
classes 151–200) using :class:`torch.nn.TripletMarginLoss` with Online Hard
Negative Mining (OHNM).  After training, the adapter weights are saved to
``adapter_trained.pt`` (or a user-specified path).

Usage
-----
    python train_adapter_triplet.py --root /path/to/imagenet/train
    python train_adapter_triplet.py --root /path/to/imagenet/train \\
        --epochs 10 --output adapter_trained.pt

CLI args
--------
  --root        Path to ImageNet ``train`` directory (ImageFolder format).
  --epochs      Number of training epochs.  (default: 5)
  --batch-size  Mini-batch size.  (default: 128)
  --lr          Adam learning rate.  (default: 1e-4)
  --margin      Triplet margin.  (default: 0.2)
  --output      Output path for trained adapter weights.  (default: adapter_trained.pt)
  --device      Torch device string, e.g. ``cuda`` or ``cpu``.  Auto-detected when omitted.

Kill Condition (G11)
--------------------
Condensation Ratio > 0.50 on the G6 (Dogs/Birds) subset after Triplet Loss
training → **G11 Kill Condition Met**.

Success Criteria
----------------
Condensation Ratio < 0.50 with trained adapter → **G11 Passed**.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adapter import MetricAdapter


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
        return self.model(x)  # (B, 1024)


def mine_hard_negatives(
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

    Args:
        anchor_proj:   L2-normalised projected embeddings of anchors, shape
                       ``(B, proj_dim)``.  Used *only* for similarity ranking.
        batch_feats:   Raw (pre-projection) features of the entire batch,
                       shape ``(B, input_dim)``.
        batch_labels:  Class labels of the batch, shape ``(B,)``.
        anchor_labels: Class labels of the anchors, shape ``(B,)``.

    Returns:
        Tuple of ``(anchors, positives, negatives)`` raw feature tensors, each
        of shape ``(T, input_dim)`` where *T* ≤ *B* is the number of valid
        triplets found.
    """
    anchors_out, positives_out, negatives_out = [], [], []

    # Pre-compute full similarity matrix between anchors and all batch samples.
    # anchor_proj is already L2-normalised; batch projection is computed by caller.
    proj_batch = anchor_proj  # shape (B, proj_dim)
    sim = anchor_proj @ proj_batch.T  # (B, B)

    for i, lbl in enumerate(anchor_labels):
        same_mask = batch_labels == lbl
        diff_mask = batch_labels != lbl

        # Need at least one positive and one negative
        same_indices = same_mask.nonzero(as_tuple=True)[0]
        # Exclude the anchor itself from the positive pool
        same_indices = same_indices[same_indices != i]
        if len(same_indices) == 0 or diff_mask.sum() == 0:
            continue

        # Random positive
        pos_idx = same_indices[torch.randint(len(same_indices), (1,)).item()]

        # Hard negative: highest cosine sim among different-class samples
        sim_row = sim[i].clone()
        sim_row[~diff_mask] = -2.0  # exclude same-class
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


def train(
    root_dir: str,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-4,
    margin: float = 0.2,
    output: str = "adapter_trained.pt",
    device: str | None = None,
) -> MetricAdapter:
    """Train MetricAdapter on the ImageNet G6 subset using triplet loss with OHNM.

    DINOv2 features are pre-extracted once and cached in CPU RAM to avoid
    redundant backbone passes.  The adapter is then trained on the cached
    feature bank for the requested number of epochs.

    Args:
        root_dir:   Path to ImageNet ``train`` directory (ImageFolder format).
        epochs:     Number of training epochs.  Defaults to 5.
        batch_size: Mini-batch size for both extraction and training.
                    Defaults to 128.
        lr:         Adam learning rate.  Defaults to 1e-4.
        margin:     Triplet margin.  Defaults to 0.2.
        output:     File path where the trained adapter state-dict is saved.
                    Defaults to ``"adapter_trained.pt"``.
        device:     Torch device string.  Auto-detected when ``None``.

    Returns:
        The trained :class:`~adapter.MetricAdapter` instance.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    print(f"[G11] Device: {dev}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(root_dir, transform=tfm)

    # G6 subset: classes 151–200 (dog breeds in ImageNet-1k)
    start_cls, end_cls = 151, 200
    indices = [i for i, t in enumerate(ds.targets) if start_cls <= t <= end_cls]
    subset = Subset(ds, indices)
    print(f"[G11] G6 subset: classes {start_cls}–{end_cls}, {len(indices)} samples")

    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    # ── DINOv2 feature extractor (frozen backbone) ────────────────────────────
    print("[G11] Loading DINOv2 ViT-L/14 …")
    extractor = _DINOv2Extractor().to(dev)

    # ── Pre-extract all DINOv2 features once ─────────────────────────────────
    print("[G11] Pre-extracting DINOv2 features …")
    all_feats, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            feats = extractor(imgs.to(dev))
            all_feats.append(feats.cpu())
            all_labels.append(lbls)

    all_feats = torch.cat(all_feats, 0)    # (N, 1024)
    all_labels = torch.cat(all_labels, 0)  # (N,)
    print(f"[G11] Feature bank: {all_feats.shape}")

    # ── Adapter ───────────────────────────────────────────────────────────────
    adapter = MetricAdapter(input_dim=1024, output_dim=256, hidden_dim=512)
    adapter.to(dev)

    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)

    feat_ds = TensorDataset(all_feats, all_labels)
    feat_loader = DataLoader(feat_ds, batch_size=batch_size, shuffle=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(
        f"\n[G11] Training: {epochs} epoch(s), batch={batch_size},"
        f" lr={lr}, margin={margin}"
    )
    for epoch in range(1, epochs + 1):
        adapter.train()
        total_loss = 0.0
        n_batches = 0

        for batch_feats, batch_labels in feat_loader:
            batch_feats = batch_feats.to(dev)
            batch_labels = batch_labels.to(dev)

            # Project the batch for similarity ranking (no gradient needed here)
            with torch.no_grad():
                proj = adapter(batch_feats)  # (B, 256) L2-normalised

            # Mine hard triplets
            anch, pos, neg = mine_hard_negatives(
                proj, batch_feats, batch_labels, batch_labels
            )
            if anch.shape[0] == 0:
                continue

            # Forward + loss with gradients
            optimizer.zero_grad()
            loss = adapter.triplet_loss(anch, pos, neg, margin=margin)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch}/{epochs}  loss={avg_loss:.6f}")

    # ── Save weights ──────────────────────────────────────────────────────────
    torch.save(adapter.state_dict(), output)
    print(f"\n[G11] Trained adapter saved → {output}")

    return adapter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="G11: Train MetricAdapter with Online Hard Negative Mining."
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("IMAGENET_ROOT", "data/imagenet-r"),
        help="Path to ImageNet directory (env: IMAGENET_ROOT)",
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Mini-batch size (default: 128)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Adam learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--margin", type=float, default=0.2,
        help="Triplet margin (default: 0.2)",
    )
    parser.add_argument(
        "--output", default="adapter_trained.pt",
        help="Output path for trained adapter weights (default: adapter_trained.pt)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device string, e.g. cuda or cpu (default: auto-detect)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print(
            f"[G11] ERROR: --root directory not found: {args.root!r}",
            file=sys.stderr,
        )
        sys.exit(1)

    train(
        root_dir=args.root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        margin=args.margin,
        output=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
