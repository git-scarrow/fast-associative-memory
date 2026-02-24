#!/usr/bin/env python3
"""
benchmarks/g24_adapter_aircraft.py — Gauntlet 24: Adapter Sweep: FGVC Aircraft.

Downloads FGVC-Aircraft, extracts DINOv2 ViT-L/14 embeddings (1024d),
trains a domain-specific MetricAdapter using Online Hard Negative Mining,
and evaluates the full stack (adapter + NSTP + vigilance sweep).

Kill Condition (G24)
--------------------
Aircraft adapter in-domain top-1 accuracy is lower than the G11 Dogs
adapter baseline of 99.71% on Stanford Dogs → KILL.

Usage
-----
    python benchmarks/g24_adapter_aircraft.py [--root DATA_DIR] \\
        [--epochs 5] [--batch-size 128] [--lr 1e-4] [--margin 0.2] \\
        [--device auto]

CLI args
--------
  --root        Data directory for FGVC Aircraft download/cache.
                (default: ./data/fgvc_aircraft)
  --epochs      Number of adapter training epochs.  (default: 5)
  --batch-size  Mini-batch size.  (default: 128)
  --lr          Adam learning rate.  (default: 1e-4)
  --margin      Triplet margin.  (default: 0.2)
  --device      Torch device string, e.g. ``cuda`` or ``cpu``.
                ``auto`` selects CUDA when available.  (default: auto)
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

# Allow importing from the repo root regardless of working directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapter import MetricAdapter  # noqa: E402
from fast_associative_memory import FastAssociativeMemory  # noqa: E402
from nstp import NSTPController  # noqa: E402


# ─────────────────────────────────────── constants ───────────────────────────

#: G11 reference: Dogs adapter full-stack accuracy (adapter + NSTP + v=0.80).
G11_DOGS_REFERENCE_ACC = 99.71

#: Vigilance values to sweep during evaluation.
VIGILANCE_SWEEP = [0.92, 0.85, 0.82, 0.80]

#: Default adapter weight output file.
ADAPTER_OUTPUT = "adapter_fgvc_aircraft.pt"

#: Cache file names (written to --root / cache sub-dir).
CACHE_TRAIN = "fgvc_aircraft_train.pt"
CACHE_TEST = "fgvc_aircraft_test.pt"


# ────────────────────────────────────── DINOv2 extractor ──────────────────────

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


# ─────────────────────────────────── dataset helpers ──────────────────────────

def _build_transform():
    """Standard DINOv2 pre-processing transform."""
    import torchvision.transforms as T
    return T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def _load_fgvc_aircraft(root: str):
    """Load FGVC-Aircraft train/test splits via torchvision.

    Uses ``torchvision.datasets.FGVCAircraft`` (torchvision ≥ 0.13).
    The dataset is downloaded automatically if not already present.

    Args:
        root: Directory where the dataset will be downloaded / cached.

    Returns:
        Tuple ``(train_ds, test_ds)`` of torchvision datasets.
    """
    try:
        from torchvision.datasets import FGVCAircraft
    except ImportError as exc:
        raise ImportError(
            "FGVCAircraft requires torchvision ≥ 0.13. "
            "Upgrade with: pip install --upgrade torchvision"
        ) from exc

    tfm = _build_transform()
    os.makedirs(root, exist_ok=True)
    train_ds = FGVCAircraft(root=root, split="train", annotation_level="variant",
                             transform=tfm, download=True)
    test_ds = FGVCAircraft(root=root, split="test", annotation_level="variant",
                            transform=tfm, download=True)
    return train_ds, test_ds


# ─────────────────────────── embedding extraction & caching ───────────────────

@torch.no_grad()
def _extract_embeddings(
    model: nn.Module,
    dataset,
    batch_size: int,
    device: torch.device,
    desc: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract DINOv2 embeddings for *dataset* and return ``(embeds, labels)``."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=(device.type == "cuda"),
    )
    all_embeds, all_labels = [], []
    for imgs, lbls in loader:
        embeds = model(imgs.to(device))
        all_embeds.append(embeds.cpu())
        all_labels.append(lbls if isinstance(lbls, torch.Tensor) else torch.tensor(lbls))
    return torch.cat(all_embeds), torch.cat(all_labels)


def extract_or_load_features(
    root: str,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cached (or freshly extracted) DINOv2 embeddings.

    If ``fgvc_aircraft_train.pt`` / ``fgvc_aircraft_test.pt`` already exist
    inside *root*, they are loaded directly.  Otherwise the FGVC-Aircraft
    dataset is downloaded, DINOv2 features are extracted, and the cache files
    are written for future runs.

    Returns:
        ``(train_embeds, train_labels, test_embeds, test_labels)``
    """
    cache_dir = Path(root)
    train_path = cache_dir / CACHE_TRAIN
    test_path = cache_dir / CACHE_TEST

    if train_path.is_file() and test_path.is_file():
        print("[G24] Loading cached embeddings …")
        tr = torch.load(train_path, map_location="cpu", weights_only=True)
        te = torch.load(test_path, map_location="cpu", weights_only=True)
        return (
            tr["embeds"].float(), tr["labels"].long(),
            te["embeds"].float(), te["labels"].long(),
        )

    # ── Download + extract ────────────────────────────────────────────────────
    print("[G24] Loading FGVC-Aircraft dataset …")
    train_ds, test_ds = _load_fgvc_aircraft(root)
    print(f"[G24]   Train: {len(train_ds)} images, "
          f"Test: {len(test_ds)} images, "
          f"Classes: {len(train_ds.classes)}")

    print("[G24] Loading DINOv2 ViT-L/14 …")
    extractor = _DINOv2Extractor().to(device)

    print("[G24] Extracting train embeddings …")
    train_embeds, train_labels = _extract_embeddings(
        extractor, train_ds, batch_size, device, desc="train"
    )
    print(f"[G24]   train embeds: {train_embeds.shape}")

    print("[G24] Extracting test embeddings …")
    test_embeds, test_labels = _extract_embeddings(
        extractor, test_ds, batch_size, device, desc="test"
    )
    print(f"[G24]   test embeds: {test_embeds.shape}")

    # ── Cache to disk ─────────────────────────────────────────────────────────
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"embeds": train_embeds.float(), "labels": train_labels.long()}, train_path)
    torch.save({"embeds": test_embeds.float(), "labels": test_labels.long()}, test_path)
    print(f"[G24] Embeddings cached → {cache_dir}/")

    return train_embeds.float(), train_labels.long(), test_embeds.float(), test_labels.long()


# ──────────────────────────── OHNM triplet mining ─────────────────────────────

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
      similarity to the projected anchor embedding.

    Args:
        anchor_proj:   L2-normalised projected embeddings of anchors, ``(B, proj_dim)``.
        batch_feats:   Raw pre-projection features, ``(B, input_dim)``.
        batch_labels:  Class labels of the batch, ``(B,)``.
        anchor_labels: Class labels of the anchors, ``(B,)``.

    Returns:
        Tuple ``(anchors, positives, negatives)`` raw feature tensors, each
        of shape ``(T, input_dim)`` where *T* ≤ *B*.
    """
    anchors_out, positives_out, negatives_out = [], [], []

    sim = anchor_proj @ anchor_proj.T  # (B, B)

    for i, lbl in enumerate(anchor_labels):
        same_mask = batch_labels == lbl
        diff_mask = batch_labels != lbl

        same_indices = same_mask.nonzero(as_tuple=True)[0]
        same_indices = same_indices[same_indices != i]
        if len(same_indices) == 0 or diff_mask.sum() == 0:
            continue

        # Random positive
        pos_idx = same_indices[torch.randint(len(same_indices), (1,)).item()]

        # Hard negative: highest cosine sim among different-class samples
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


# ───────────────────────────────── adapter training ───────────────────────────

def train_adapter(
    train_embeds: torch.Tensor,
    train_labels: torch.Tensor,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-4,
    margin: float = 0.2,
    output: str = ADAPTER_OUTPUT,
    device: torch.device = torch.device("cpu"),
) -> MetricAdapter:
    """Train a MetricAdapter on FGVC Aircraft embeddings using OHNM triplet loss.

    Args:
        train_embeds: Pre-extracted DINOv2 embeddings, ``(N, 1024)``.
        train_labels: Class labels, ``(N,)``.
        epochs:       Number of training epochs.
        batch_size:   Mini-batch size.
        lr:           Adam learning rate.
        margin:       Triplet loss margin.
        output:       Path to save adapter weights.
        device:       Compute device.

    Returns:
        Trained :class:`~adapter.MetricAdapter`.
    """
    input_dim = train_embeds.shape[1]
    adapter = MetricAdapter(input_dim=input_dim, output_dim=256, hidden_dim=512)
    adapter.to(device)

    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)

    feat_ds = TensorDataset(train_embeds, train_labels)
    feat_loader = DataLoader(feat_ds, batch_size=batch_size, shuffle=True)

    print(
        f"\n[G24] Training adapter: {epochs} epoch(s), "
        f"batch={batch_size}, lr={lr}, margin={margin}"
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
    print(f"\n[G24] Adapter saved → {output}")
    return adapter


# ─────────────────────────────────────── evaluation ───────────────────────────

@torch.no_grad()
def eval_full_stack(
    train_embeds: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeds: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
    adapter: MetricAdapter | None = None,
    core_vigilance: float = 0.85,
    core_entries: int = 50000,
) -> tuple[float, int, float]:
    """Evaluate FAM with optional adapter and NSTP on FGVC Aircraft test set.

    Args:
        train_embeds, train_labels: Training embeddings and labels.
        test_embeds, test_labels:   Test embeddings and labels.
        num_classes:  Number of aircraft variants.
        device:       Compute device.
        adapter:      Optional trained MetricAdapter.
        core_vigilance: Vigilance threshold for FAM.
        core_entries: FAM prototype capacity.

    Returns:
        Tuple ``(accuracy_pct, num_prototypes, condensation_ratio)``.
    """
    nstp = NSTPController(sibling_threshold=0.85, depth_epsilon=0.10)

    mem = FastAssociativeMemory(
        input_dim=train_embeds.shape[1],
        value_dim=num_classes,
        core_entries=core_entries,
        core_vigilance=core_vigilance,
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=25,
        inference_temp=0.05,
        use_lfu=True,
        adapter=adapter,
        nstp=nstp,
    ).to(device)

    # Ingest training embeddings
    x_train = train_embeds.to(device)
    y_train = train_labels.to(device)
    for i in range(0, len(x_train), 256):
        mem.learn_local(x_train[i:i + 256], y_train[i:i + 256])

    # Evaluate on test set
    correct = 0
    x_test = test_embeds.to(device)
    y_test = test_labels.to(device)
    for i in range(0, len(x_test), 512):
        logits = mem(x_test[i:i + 512])
        correct += (logits.argmax(dim=1) == y_test[i:i + 512]).sum().item()

    num_prototypes = int(mem.core_cam.occupied.sum().item())
    num_train = len(train_labels)
    condensation_ratio = num_prototypes / num_train if num_train > 0 else 0.0
    accuracy = 100.0 * correct / len(test_labels)

    return accuracy, num_prototypes, condensation_ratio


# ───────────────────────────────── output formatting ──────────────────────────

def _format_results(
    rows: list[tuple[str, float, int, float]],
    best_aircraft_acc: float,
    dogs_reference: float = G11_DOGS_REFERENCE_ACC,
) -> str:
    """Format the G24 results table as a string.

    Args:
        rows: List of ``(config_label, accuracy, num_protos, cond_ratio)``.
        best_aircraft_acc: Best top-1 accuracy achieved by the Aircraft adapter.
        dogs_reference: G11 Dogs reference accuracy (default 99.71 %).

    Returns:
        Formatted multi-line string.
    """
    sep = "=" * 67
    lines = [
        "\nG24 \u2014 Adapter Sweep: FGVC Aircraft",
        sep,
        f"  {'Config':<38} {'Accuracy':>9}  {'Protos':>6}  {'Cond.Ratio':>10}",
        f"  {'-' * 63}",
    ]
    for config, acc, protos, cond in rows:
        lines.append(
            f"  {config:<38} {acc:>8.2f}%  {protos:>6d}  {cond:>10.2f}"
        )
    lines.append(sep)

    kill = best_aircraft_acc < dogs_reference
    verdict = "KILL" if kill else "PASS"
    lines.append(
        f"\nKill condition: Aircraft best accuracy {best_aircraft_acc:.2f}% "
        f"vs Dogs reference {dogs_reference:.2f}%"
    )
    lines.append(f"\u2192 {verdict}")
    return "\n".join(lines)


# ──────────────────────────────────────────── main ────────────────────────────

def run(
    root: str = "./data/fgvc_aircraft",
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-4,
    margin: float = 0.2,
    device_str: str = "auto",
    adapter_output: str = ADAPTER_OUTPUT,
    vigilance_sweep: list[float] | None = None,
    dogs_reference: float = G11_DOGS_REFERENCE_ACC,
) -> dict:
    """End-to-end G24 pipeline: extract → train adapter → evaluate → report.

    Args:
        root:          Data directory for FGVC Aircraft.
        epochs:        Adapter training epochs.
        batch_size:    Mini-batch size for extraction and training.
        lr:            Adam learning rate.
        margin:        Triplet loss margin.
        device_str:    ``"auto"`` auto-detects CUDA; otherwise a device string.
        adapter_output: Path where the trained adapter is saved.
        vigilance_sweep: Vigilance values to evaluate.  Defaults to
                         ``[0.92, 0.85, 0.82, 0.80]``.
        dogs_reference: G11 Dogs reference accuracy (default 99.71 %).

    Returns:
        Dict with keys ``"rows"``, ``"best_aircraft_acc"``, ``"kill"``.
    """
    if vigilance_sweep is None:
        vigilance_sweep = VIGILANCE_SWEEP

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"[G24] Device: {device}")

    # ── Step 1: Extract or load DINOv2 embeddings ─────────────────────────────
    train_embeds, train_labels, test_embeds, test_labels = extract_or_load_features(
        root=root, batch_size=batch_size, device=device
    )
    num_classes = int(train_labels.max().item()) + 1
    num_train = len(train_labels)
    print(f"[G24] num_classes={num_classes}, train={num_train}, test={len(test_labels)}")

    core_entries = max(num_train * 2, 50000)

    # ── Step 2: Train domain-specific MetricAdapter ────────────────────────────
    adapter = train_adapter(
        train_embeds=train_embeds,
        train_labels=train_labels,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        margin=margin,
        output=adapter_output,
        device=device,
    )
    adapter.eval()

    rows: list[tuple[str, float, int, float]] = []

    # ── Step 3a: Baseline (no adapter, v=0.92) ─────────────────────────────────
    print("\n[G24] Evaluating baseline (no adapter, v=0.92) …")
    acc_b, protos_b, cond_b = eval_full_stack(
        train_embeds, train_labels, test_embeds, test_labels,
        num_classes=num_classes, device=device,
        adapter=None, core_vigilance=0.92, core_entries=core_entries,
    )
    rows.append(("baseline (no adapter, v=0.92)", acc_b, protos_b, cond_b))
    print(f"  accuracy={acc_b:.2f}%  protos={protos_b}  cond={cond_b:.2f}")

    # ── Step 3b: Aircraft adapter at each vigilance level ─────────────────────
    best_aircraft_acc = 0.0
    for v in vigilance_sweep:
        label = f"aircraft adapter (v={v:.2f})"
        print(f"\n[G24] Evaluating {label} …")
        acc, protos, cond = eval_full_stack(
            train_embeds, train_labels, test_embeds, test_labels,
            num_classes=num_classes, device=device,
            adapter=adapter, core_vigilance=v, core_entries=core_entries,
        )
        rows.append((label, acc, protos, cond))
        print(f"  accuracy={acc:.2f}%  protos={protos}  cond={cond:.2f}")
        if acc > best_aircraft_acc:
            best_aircraft_acc = acc

    # ── Step 4: Format and print results ──────────────────────────────────────
    table = _format_results(rows, best_aircraft_acc, dogs_reference)
    print(table)

    kill = best_aircraft_acc < dogs_reference
    return {"rows": rows, "best_aircraft_acc": best_aircraft_acc, "kill": kill}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="G24 — Adapter Sweep: FGVC Aircraft",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        default="./data/fgvc_aircraft",
        metavar="DIR",
        help="Data directory for FGVC Aircraft download/cache.",
    )
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of adapter training epochs.")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Adam learning rate.")
    parser.add_argument("--margin", type=float, default=0.2,
                        help="Triplet margin.")
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
