#!/usr/bin/env python3
"""
benchmarks/g22_adapter_birds.py — Gauntlet 22: Adapter Sweep: CUB-200 Birds.

Self-contained training + evaluation script for fine-grained bird recognition
using the CUB-200-2011 dataset with a domain-specific MetricAdapter.

Protocol
--------
1. Loads CUB-200-2011 (200 bird species, ~6k train / ~5.8k test images) via
   ``torchvision.datasets.CUB200`` when available, or a native CUB-200-2011
   reader that understands the canonical ``train_test_split.txt`` format, with
   ImageFolder (``train/`` / ``test/`` sub-directories) as a final fallback.
2. Extracts DINOv2 ViT-L/14 embeddings (1024-d) for train and test; caches to
   ``<root>/cub200_train.pt`` and ``<root>/cub200_test.pt`` so the expensive
   backbone pass only runs once.
3. Trains a domain-specific ``MetricAdapter`` on the CUB-200 train embeddings
   using Online Hard Negative Mining (OHNM) triplet loss — same architecture
   and hyper-parameters as ``train_adapter_triplet.py``.
4. Evaluates the full stack (adapter + NSTP + vigilance sweep) on the CUB-200
   test set.  Vigilance levels ``[0.92, 0.85, 0.82, 0.80]`` match the G11
   ablation.
5. Checks the kill condition: Birds best accuracy vs. the G11 Dogs reference
   (99.71 %).

Kill Condition (G22)
--------------------
Best Birds-adapter top-1 accuracy < G11 Dogs reference (99.71 %) → **KILL**.

G11 Reference
-------------
Full stack (adapter + NSTP + v=0.80) achieved 99.71 % accuracy,
condensation ratio 0.47 on Stanford Dogs subset (ImageNet classes 151–200).

Usage
-----
    python benchmarks/g22_adapter_birds.py
    python benchmarks/g22_adapter_birds.py --root ./data/cub200 --epochs 5

CLI args
--------
  --root        Data directory for CUB-200 download / cache.
                (default: ./data/cub200)
  --epochs      Number of training epochs.  (default: 5)
  --batch-size  Mini-batch size for extraction and training.  (default: 128)
  --lr          Adam learning rate.  (default: 1e-4)
  --margin      Triplet margin.  (default: 0.2)
  --device      Torch device string or ``auto``.  (default: auto)
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Allow imports from the repo root regardless of working directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapter import MetricAdapter  # noqa: E402
from fast_associative_memory import FastAssociativeMemory  # noqa: E402
from nstp import NSTPController  # noqa: E402


# ─────────────────────────────────────────────────── constants ───

#: G11 Dogs-adapter reference accuracy (full stack, v=0.80).
G11_DOGS_REFERENCE: float = 99.71

#: Vigilance levels to sweep — matches the G11 ablation.
VIGILANCE_SWEEP: list[float] = [0.92, 0.85, 0.82, 0.80]

#: Default output path for the trained CUB-200 adapter.
_ADAPTER_SAVE = "adapter_cub200_birds.pt"

#: CUB-200-2011 archive download URL.
_CUB200_URL = (
    "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
)


# ─────────────────────────────────────────────── dataset helpers ───

def _build_transform():
    """Standard DINOv2 image pre-processing (matches extract_imagenet_r_vitl14.py)."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


class _CUB200NativeDataset(Dataset):
    """CUB-200-2011 dataset reader for the canonical archive layout.

    Reads ``images.txt``, ``train_test_split.txt``, and
    ``image_class_labels.txt`` from ``<cub_root>/CUB_200_2011/`` (or
    directly from ``<cub_root>/`` when the files are in the root).

    Args:
        cub_root: Directory that *contains* ``images.txt``,
            ``train_test_split.txt``, ``image_class_labels.txt``, and the
            ``images/`` sub-directory.
        train: If ``True`` load the official training split; otherwise the
            test split.
        transform: Optional image transform applied at ``__getitem__``.
    """

    def __init__(self, cub_root: str, train: bool = True, transform=None):
        self.transform = transform

        root = Path(cub_root)

        # Locate the meta-files (they may sit directly in root or one level down)
        for candidate in (root, root / "CUB_200_2011"):
            if (candidate / "images.txt").is_file():
                root = candidate
                break
        else:
            raise FileNotFoundError(
                f"CUB-200-2011 meta-files not found under '{cub_root}'. "
                "Expected 'images.txt', 'train_test_split.txt', "
                "'image_class_labels.txt' alongside an 'images/' directory."
            )

        # Parse meta-files
        id2file = {}
        with open(root / "images.txt") as fh:
            for line in fh:
                img_id, fname = line.strip().split()
                id2file[int(img_id)] = fname

        id2split = {}
        with open(root / "train_test_split.txt") as fh:
            for line in fh:
                img_id, is_train = line.strip().split()
                id2split[int(img_id)] = int(is_train)

        id2label = {}
        with open(root / "image_class_labels.txt") as fh:
            for line in fh:
                img_id, cls_id = line.strip().split()
                id2label[int(img_id)] = int(cls_id) - 1  # 0-indexed

        images_dir = root / "images"
        target_split = 1 if train else 0

        self.samples: list[tuple[Path, int]] = [
            (images_dir / id2file[img_id], id2label[img_id])
            for img_id in sorted(id2file)
            if id2split[img_id] == target_split
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image  # lazy import — only needed when images are loaded

        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _load_cub200_datasets(root: str):
    """Load CUB-200-2011 train/test datasets.

    Tries in order:
    1. ``torchvision.datasets.CUB200`` (when present in the installed version).
    2. :class:`_CUB200NativeDataset` — native CUB-200-2011 split-file reader.
    3. ``torchvision.datasets.ImageFolder`` with ``<root>/train/`` and
       ``<root>/test/`` sub-directories.

    Parameters
    ----------
    root:
        Directory where the dataset is stored (or will be searched for).

    Returns
    -------
    train_ds, test_ds : Dataset
    """
    from torchvision import datasets

    transform = _build_transform()

    # ── 1. torchvision native CUB200 (if available) ──────────────────────
    try:
        train_ds = datasets.CUB200(root=root, train=True,
                                   transform=transform, download=True)
        test_ds = datasets.CUB200(root=root, train=False,
                                  transform=transform, download=True)
        print(f"[G22] Loaded CUB-200 via torchvision.datasets.CUB200")
        return train_ds, test_ds
    except AttributeError:
        pass  # CUB200 not available in this torchvision version
    except Exception as exc:
        print(f"[G22] torchvision.datasets.CUB200 failed ({exc}); trying fallbacks …")

    # ── 2. Native CUB-200-2011 reader ─────────────────────────────────────
    try:
        train_ds = _CUB200NativeDataset(root, train=True, transform=transform)
        test_ds = _CUB200NativeDataset(root, train=False, transform=transform)
        print(f"[G22] Loaded CUB-200 via native split-file reader ({root})")
        return train_ds, test_ds
    except FileNotFoundError:
        pass

    # ── 3. ImageFolder fallback (user-organised train/ test/ dirs) ────────
    train_dir = os.path.join(root, "train")
    test_dir = os.path.join(root, "test")
    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        train_ds = datasets.ImageFolder(train_dir, transform=transform)
        test_ds = datasets.ImageFolder(test_dir, transform=transform)
        print(f"[G22] Loaded CUB-200 via ImageFolder ({train_dir} / {test_dir})")
        return train_ds, test_ds

    raise FileNotFoundError(
        f"CUB-200-2011 dataset not found under '{root}'.\n"
        "Options:\n"
        f"  (a) Download from {_CUB200_URL} and\n"
        "      extract so that 'images.txt' lives directly under --root or under\n"
        "      <root>/CUB_200_2011/.\n"
        "  (b) Organise images as ImageFolder with <root>/train/ and <root>/test/\n"
        "      sub-directories.\n"
        "  (c) Install a torchvision version that includes datasets.CUB200."
    )


# ─────────────────────────────────────── feature extraction / caching ───

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


@torch.no_grad()
def _extract_split(
    model: nn.Module,
    dataset,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``model`` over *dataset* and return (embeds, labels)."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    all_embeds, all_labels = [], []
    for imgs, lbls in loader:
        embeds = model(imgs.to(device))
        all_embeds.append(embeds.cpu())
        all_labels.append(lbls)
    return torch.cat(all_embeds), torch.cat(all_labels)


def load_or_extract_features(
    root: str,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (train_embeds, train_labels, test_embeds, test_labels).

    Features are extracted using DINOv2 ViT-L/14 and cached to
    ``<root>/cub200_train.pt`` / ``<root>/cub200_test.pt``.  Subsequent
    calls load from the cache and skip the expensive backbone pass.

    Parameters
    ----------
    root:       CUB-200 data / cache directory.
    batch_size: Batch size for the DINOv2 extraction pass.
    device:     Compute device for the backbone forward pass.

    Returns
    -------
    train_embeds : (N_train, 1024) float32
    train_labels : (N_train,) int64
    test_embeds  : (N_test, 1024) float32
    test_labels  : (N_test,) int64
    """
    cache_dir = Path(root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = cache_dir / "cub200_train.pt"
    test_cache = cache_dir / "cub200_test.pt"

    if train_cache.is_file() and test_cache.is_file():
        print(f"[G22] Loading cached features from {cache_dir} …")
        tr = torch.load(train_cache, map_location="cpu", weights_only=True)
        te = torch.load(test_cache, map_location="cpu", weights_only=True)
        return (
            tr["embeds"].float(), tr["labels"].long(),
            te["embeds"].float(), te["labels"].long(),
        )

    print("[G22] Loading CUB-200-2011 …")
    train_ds, test_ds = _load_cub200_datasets(root)
    print(f"[G22]   Train: {len(train_ds)} | Test: {len(test_ds)}")

    print("[G22] Loading DINOv2 ViT-L/14 …")
    extractor = _DINOv2Extractor().to(device)

    print("[G22] Extracting train features …")
    train_embeds, train_labels = _extract_split(extractor, train_ds, batch_size, device)
    print(f"[G22]   train embeds: {tuple(train_embeds.shape)}")

    print("[G22] Extracting test features …")
    test_embeds, test_labels = _extract_split(extractor, test_ds, batch_size, device)
    print(f"[G22]   test embeds: {tuple(test_embeds.shape)}")

    torch.save({"embeds": train_embeds.float(), "labels": train_labels.long()},
               train_cache)
    torch.save({"embeds": test_embeds.float(), "labels": test_labels.long()},
               test_cache)
    print(f"[G22] Feature caches written to {cache_dir}/")

    return train_embeds.float(), train_labels.long(), test_embeds.float(), test_labels.long()


# ──────────────────────────────────────────────── OHNM training ───

def _mine_hard_negatives(
    anchor_proj: torch.Tensor,
    batch_feats: torch.Tensor,
    batch_labels: torch.Tensor,
    anchor_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Online Hard Negative Mining over a mini-batch.

    Identical protocol to ``train_adapter_triplet.py``:
    * **Positive** — a randomly chosen same-class sample (≠ anchor itself).
    * **Hard negative** — the non-same-class sample with highest cosine
      similarity to the *projected* anchor embedding.

    Args:
        anchor_proj:   L2-normalised projected embeddings, shape ``(B, proj_dim)``.
        batch_feats:   Raw pre-projection features, shape ``(B, input_dim)``.
        batch_labels:  Class labels of the batch, shape ``(B,)``.
        anchor_labels: Class labels of the anchors, shape ``(B,)``.

    Returns:
        Tuple of ``(anchors, positives, negatives)`` raw feature tensors,
        each of shape ``(T, input_dim)`` where *T* ≤ *B*.
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
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-4,
    margin: float = 0.2,
    output: str = _ADAPTER_SAVE,
    device: torch.device | None = None,
) -> MetricAdapter:
    """Train a ``MetricAdapter`` on CUB-200 train embeddings using OHNM.

    Architecture and hyper-parameters are identical to the G11 Dogs adapter:
    ``MetricAdapter(input_dim=1024, output_dim=256, hidden_dim=512)``.

    Parameters
    ----------
    train_embeds:  Pre-extracted DINOv2 features, shape ``(N, 1024)``.
    train_labels:  Corresponding class labels, shape ``(N,)``.
    epochs:        Number of training epochs.
    batch_size:    Mini-batch size.
    lr:            Adam learning rate.
    margin:        Triplet margin.
    output:        File path to save trained adapter weights.
    device:        Compute device.  Auto-detected when ``None``.

    Returns
    -------
    MetricAdapter — trained adapter (already in eval mode, weights saved).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adapter = MetricAdapter(input_dim=1024, output_dim=256, hidden_dim=512)
    adapter.to(device)

    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)
    feat_ds = TensorDataset(train_embeds, train_labels)
    feat_loader = DataLoader(feat_ds, batch_size=batch_size, shuffle=True)

    print(
        f"\n[G22] Training: {epochs} epoch(s), batch={batch_size},"
        f" lr={lr}, margin={margin}"
    )
    for epoch in range(1, epochs + 1):
        adapter.train()
        total_loss, n_batches = 0.0, 0

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
    print(f"\n[G22] Birds adapter saved → {output}")
    adapter.eval()
    return adapter


# ──────────────────────────────────────── evaluation helpers ───

@torch.no_grad()
def eval_fam_stack(
    train_embeds: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeds: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
    num_classes: int,
    adapter: MetricAdapter | None,
    core_vigilance: float,
    core_entries: int = 50000,
) -> tuple[float, int, float]:
    """Ingest train embeddings into FAM and evaluate top-1 accuracy.

    Constructs a ``FastAssociativeMemory`` with NSTP lateral inhibition and
    the given *adapter* (``None`` = baseline, no adapter), ingests the full
    training set, and evaluates on the test set.

    Parameters
    ----------
    train_embeds, train_labels:
        Training embeddings and labels.
    test_embeds, test_labels:
        Test embeddings and labels.
    device:
        Compute device.
    num_classes:
        Number of output classes.
    adapter:
        Trained ``MetricAdapter`` (or ``None`` for no-adapter baseline).
    core_vigilance:
        Vigilance threshold for the continuous CAM.
    core_entries:
        Maximum prototype capacity.

    Returns
    -------
    acc : float
        Top-1 accuracy in %.
    n_protos : int
        Number of prototypes stored after ingestion.
    cond_ratio : float
        Condensation ratio = n_protos / len(train_embeds).
    """
    embed_dim = train_embeds.size(1)

    nstp = NSTPController()
    mem = FastAssociativeMemory(
        input_dim=embed_dim,
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

    x_train = train_embeds.to(device)
    y_train = train_labels.to(device)
    for i in range(0, len(x_train), 256):
        mem.learn_local(x_train[i : i + 256], y_train[i : i + 256])

    n_protos = int(mem.core_cam.occupied.sum().item())
    cond_ratio = n_protos / len(train_embeds)

    correct = 0
    x_test = test_embeds.to(device)
    y_test = test_labels.to(device)
    for i in range(0, len(x_test), 512):
        logits = mem(x_test[i : i + 512])
        correct += (logits.argmax(dim=1) == y_test[i : i + 512]).sum().item()

    acc = 100.0 * correct / len(test_labels)
    return acc, n_protos, cond_ratio


def check_kill_condition(best_birds_acc: float, dogs_reference: float = G11_DOGS_REFERENCE) -> bool:
    """Return ``True`` when the kill condition is met.

    Kill condition: Birds best accuracy < Dogs reference (strict inequality).

    Parameters
    ----------
    best_birds_acc : float
        Best top-1 accuracy (%) achieved by the Birds adapter.
    dogs_reference : float
        G11 Dogs-adapter reference accuracy.  Defaults to ``G11_DOGS_REFERENCE``.

    Returns
    -------
    bool
        ``True`` if kill condition met (Birds accuracy is strictly less than
        the Dogs reference).
    """
    return best_birds_acc < dogs_reference


# ───────────────────────────────────────────────── reporting ───

def _format_report(
    rows: list[tuple[str, float, int, float]],
    best_birds_acc: float,
    dogs_reference: float = G11_DOGS_REFERENCE,
) -> str:
    """Return the formatted results table as a string.

    Parameters
    ----------
    rows : list of (config_label, accuracy, n_protos, cond_ratio)
    best_birds_acc : Best Birds-adapter accuracy across all vigilance levels.
    dogs_reference : G11 Dogs reference accuracy.
    """
    sep = "=" * 67
    lines = [
        "\nG22 \u2014 Adapter Sweep: CUB-200 Birds",
        sep,
        f"  {'Config':<36} {'Accuracy':>10} {'Protos':>7} {'Cond.Ratio':>11}",
        f"  {'-' * 65}",
    ]
    for label, acc, n_protos, cond_ratio in rows:
        lines.append(
            f"  {label:<36} {acc:>9.2f}% {n_protos:>7d} {cond_ratio:>11.2f}"
        )
    lines.append(sep)

    killed = check_kill_condition(best_birds_acc, dogs_reference)
    verdict = "KILL" if killed else "PASS"
    lines.append(
        f"\nKill condition: Birds best accuracy {best_birds_acc:.2f}%"
        f" vs Dogs reference {dogs_reference:.2f}%"
    )
    lines.append(f"\u2192 {verdict}")
    return "\n".join(lines)


# ────────────────────────────────────────────── main runner ───

def run(
    root: str = "./data/cub200",
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-4,
    margin: float = 0.2,
    device: str | None = None,
    adapter_output: str = _ADAPTER_SAVE,
) -> dict:
    """Run the full G22 pipeline: extract → train → evaluate → report.

    Parameters
    ----------
    root:           CUB-200 data / cache directory.
    epochs:         Adapter training epochs.
    batch_size:     Batch size for extraction and training.
    lr:             Adam learning rate.
    margin:         Triplet margin.
    device:         Torch device string.  Auto-detected when ``None``.
    adapter_output: Path to save trained adapter weights.

    Returns
    -------
    dict with keys:
      ``rows``            — list of (label, acc, n_protos, cond_ratio) tuples.
      ``best_birds_acc``  — best Birds-adapter top-1 accuracy (%).
      ``kill``            — bool, True if kill condition met.
    """
    if device is None or device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    print(f"[G22] Device: {dev}")

    # ── Feature extraction (with caching) ────────────────────────────────
    train_e, train_l, test_e, test_l = load_or_extract_features(
        root, batch_size, dev
    )
    num_classes = int(train_l.max().item()) + 1
    print(f"[G22] num_classes={num_classes}  train={len(train_e)}  test={len(test_e)}")

    # ── Adapter training ─────────────────────────────────────────────────
    adapter = train_adapter(
        train_e, train_l,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        margin=margin,
        output=adapter_output,
        device=dev,
    )
    adapter.eval()

    # ── Evaluation ───────────────────────────────────────────────────────
    print("\n[G22] Evaluating …")
    rows: list[tuple[str, float, int, float]] = []

    # Baseline: no adapter, v=0.92
    label = "baseline (no adapter, v=0.92)"
    print(f"  {label} …")
    acc, n_protos, cond_ratio = eval_fam_stack(
        train_e, train_l, test_e, test_l, dev, num_classes,
        adapter=None, core_vigilance=0.92,
    )
    rows.append((label, acc, n_protos, cond_ratio))

    best_birds_acc = 0.0
    for v in VIGILANCE_SWEEP:
        label = f"birds adapter (v={v:.2f})"
        print(f"  {label} …")
        acc, n_protos, cond_ratio = eval_fam_stack(
            train_e, train_l, test_e, test_l, dev, num_classes,
            adapter=adapter, core_vigilance=v,
        )
        rows.append((label, acc, n_protos, cond_ratio))
        if acc > best_birds_acc:
            best_birds_acc = acc

    # ── Report ───────────────────────────────────────────────────────────
    report = _format_report(rows, best_birds_acc)
    print(report)

    return {
        "rows": rows,
        "best_birds_acc": best_birds_acc,
        "kill": check_kill_condition(best_birds_acc),
    }


# ─────────────────────────────────────────────────── CLI ───

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="G22 — Adapter Sweep: CUB-200 Birds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root", default="./data/cub200", metavar="DIR",
        help="Data directory for CUB-200 download / cache.",
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Mini-batch size for extraction and training.",
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
        help="Torch device string or 'auto' for automatic detection.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    result = run(
        root=args.root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        margin=args.margin,
        device=args.device,
    )
    sys.exit(1 if result["kill"] else 0)


if __name__ == "__main__":
    main()
