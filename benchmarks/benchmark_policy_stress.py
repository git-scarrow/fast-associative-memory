"""
benchmark_policy_stress.py — Phase 3: G6 (Granularity) & G7 (Tail Eviction)

Usage
-----
  # Without adapter (raw DINOv2 baseline):
  python benchmarks/benchmark_policy_stress.py --root /path/to/imagenet/train

  # With trained MetricAdapter (G11 evaluation):
  python benchmarks/benchmark_policy_stress.py --root /path/to/imagenet/train \\
      --use-adapter --adapter-path adapter_trained.pt

CLI args
--------
  --root          Path to ImageNet train directory (ImageFolder format).
  --use-adapter   Apply a trained MetricAdapter before committing to FAM.
  --adapter-path  Path to saved adapter state-dict (default: adapter_trained.pt).
  --device        Torch device string (default: auto-detect).
  --g6-only       Run G6 (granularity) benchmark only; skip G7.
  --g7-only       Run G7 (tail eviction) benchmark only; skip G6.
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import numpy as np
from collections import Counter

from fast_associative_memory import FastAssociativeMemory
from adapter import MetricAdapter


class DINOv2Extractor(nn.Module):
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


# ==========================================
# G7: IMAGENET LONG-TAIL (Simulated)
# ==========================================
def make_long_tail_indices(targets, imbalance_ratio=0.01):
    """
    Subsamples indices to create a Pareto distribution (Long-Tail).
    Head classes get 100% of samples. Tail classes get imbalance_ratio * samples.
    """
    targets = np.array(targets)
    classes = np.unique(targets)
    n_classes = len(classes)

    indices = []
    counts = []
    print(f"Generating Long-Tail splits (Imbalance Ratio: {imbalance_ratio})...")

    for i, c in enumerate(classes):
        ratio = imbalance_ratio ** (i / (n_classes - 1))
        class_indices = np.where(targets == c)[0]
        n_keep = max(1, int(len(class_indices) * ratio))
        keep = np.random.choice(class_indices, n_keep, replace=False)
        indices.extend(keep)
        counts.append(n_keep)

    print(f"  Head Class Count: {counts[0]}")
    print(f"  Tail Class Count: {counts[-1]}")
    print(f"  Total Samples: {len(indices)} (Original: {len(targets)})")

    return indices, counts


def run_g7_tail_eviction(root_dir, device="cuda"):
    print("\n\U0001f480  G7: TAIL EVICTION TEST (ImageNet-LT)  \U0001f480")

    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ds = datasets.ImageFolder(root_dir, transform=tfm)

    lt_indices, cls_counts = make_long_tail_indices(ds.targets, imbalance_ratio=0.02)
    train_set = Subset(ds, lt_indices)

    cls_counts = np.array(cls_counts)
    many_shot   = np.where(cls_counts > 100)[0]
    medium_shot = np.where((cls_counts <= 100) & (cls_counts >= 20))[0]
    few_shot    = np.where(cls_counts < 20)[0]

    print(f"  Many (>100): {len(many_shot)} cls | Medium (20-100): {len(medium_shot)} cls | Few (<20): {len(few_shot)} cls")

    extractor = DINOv2Extractor().to(device)
    capacity = int(len(train_set) * 0.2)
    print(f"  Constraint: {capacity} slots for {len(train_set)} samples.")

    fam = FastAssociativeMemory(input_dim=1024, value_dim=len(ds.classes),
                                core_entries=capacity, core_vigilance=0.90).to(device)
    fam.core_cam.use_lfu = True

    loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
    fam.train()
    print("  Streaming data...")
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        with torch.no_grad():
            feats = extractor(imgs)
            fam.learn_local(feats, lbls)

    print("  Evaluating Retention...")
    correct_counts = Counter()
    total_counts = Counter()

    fam.eval()
    eval_loader = DataLoader(train_set, batch_size=128, num_workers=4)
    with torch.no_grad():
        for imgs, lbls in eval_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            feats = extractor(imgs)
            preds = fam(feats).argmax(dim=1)
            for l, p in zip(lbls.cpu().numpy(), preds.cpu().numpy()):
                total_counts[l] += 1
                if l == p:
                    correct_counts[l] += 1

    def get_acc(classes):
        if len(classes) == 0:
            return 0.0
        corr = sum(correct_counts[c] for c in classes)
        tot  = sum(total_counts[c]   for c in classes)
        return (corr / tot * 100) if tot > 0 else 0.0

    acc_many = get_acc(many_shot)
    acc_med  = get_acc(medium_shot)
    acc_few  = get_acc(few_shot)

    print(f"\n--- G7 Results ---")
    print(f"Many-Shot Acc:   {acc_many:.2f}% (Expect High)")
    print(f"Medium-Shot Acc: {acc_med:.2f}%")
    print(f"Few-Shot Acc:    {acc_few:.2f}% (The Kill Zone)")

    if acc_few < 20.0 and acc_many > 80.0:
        print("\u274c KILL: LFU massacred the tail. (Few < 20%)")
    elif acc_few > 50.0:
        print("\u2705 PASS: LFU preserved rare classes.")
    else:
        print("\u26a0\ufe0f  WARN: Mixed results.")


# ==========================================
# G6: GRANULARITY (Dogs subset of ImageNet)
# ==========================================
def run_g6_granularity(root_dir, device="cuda", adapter: MetricAdapter | None = None):
    print("\n\U0001f985  G6: GRANULARITY TORTURE TEST (Dogs/Birds)  \U0001f985")
    if adapter is not None:
        print("  MetricAdapter: ACTIVE (trained metric projection)")
    else:
        print("  MetricAdapter: NONE (raw DINOv2 baseline)")

    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ds = datasets.ImageFolder(root_dir, transform=tfm)

    # Classes 151-200: 50 dog breeds in ImageNet-1k
    start_cls, end_cls = 151, 200
    indices = [i for i, t in enumerate(ds.targets) if start_cls <= t <= end_cls]

    subset = Subset(ds, indices)
    print(f"  Fine-Grained Subset: Classes {start_cls}-{end_cls} ({len(indices)} samples)")

    extractor = DINOv2Extractor().to(device)

    # When an adapter is provided, FAM input_dim stays 1024 (backbone output)
    # and the adapter handles the projection to its output_dim internally.
    fam = FastAssociativeMemory(
        input_dim=1024, value_dim=1000,
        core_entries=10000, core_vigilance=0.92,
        adapter=adapter,
    ).to(device)

    loader = DataLoader(subset, batch_size=128, shuffle=True, num_workers=4)
    print("  Ingesting...")
    for imgs, lbls in loader:
        with torch.no_grad():
            fam.learn_local(extractor(imgs.to(device)), lbls.to(device))

    correct = 0
    total   = 0
    fam.eval()
    for imgs, lbls in loader:
        with torch.no_grad():
            preds = fam(extractor(imgs.to(device))).argmax(1)
            correct += (preds == lbls.to(device)).sum().item()
            total   += imgs.size(0)

    acc    = 100 * correct / total
    protos = fam.core_cam.occupied.sum().item()
    ratio  = protos / total

    print(f"\n--- G6 Results ---")
    print(f"Accuracy:           {acc:.2f}%")
    print(f"Prototypes:         {protos} / {total} samples")
    print(f"Condensation Ratio: {ratio:.2f} (Lower is better)")

    if adapter is not None:
        if ratio > 0.50:
            print("\u274c G11 KILL: Condensation Ratio > 0.50 — adapter did not improve granularity.")
        else:
            print("\u2705 G11 PASS: Condensation Ratio \u2264 0.50 with trained adapter.")

    if acc > 90.0 and ratio < 0.5:
        print("\u2705 PASS: High accuracy with efficient condensation.")
    elif ratio > 0.8:
        print("\u274c FAIL: Overfitting. FAM memorized the dataset.")
    elif acc < 70.0:
        print("\u274c FAIL: Underfitting. FAM merged distinct breeds.")
    else:
        print("\u26a0\ufe0f  WARN: Moderate accuracy/condensation trade-off.")


def _load_adapter(adapter_path: str) -> MetricAdapter:
    """Load a trained MetricAdapter from a saved state-dict file.

    Args:
        adapter_path: Path to the ``adapter_trained.pt`` file produced by
                      ``train_adapter_triplet.py``.

    Returns:
        :class:`~adapter.MetricAdapter` with trained weights loaded and set to
        eval mode.

    Raises:
        SystemExit: If the file does not exist.
    """
    if not os.path.isfile(adapter_path):
        print(
            f"[benchmark] ERROR: adapter file not found: {adapter_path!r}\n"
            "  Run train_adapter_triplet.py first to generate adapter_trained.pt",
            file=sys.stderr,
        )
        sys.exit(1)

    adapter = MetricAdapter(input_dim=1024, output_dim=256, hidden_dim=512)
    adapter.load_state_dict(torch.load(adapter_path, weights_only=True))
    adapter.eval()
    return adapter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="G6 (Granularity) & G7 (Tail Eviction) benchmarks."
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("IMAGENET_ROOT", "data/imagenet-r"),
        help="Path to ImageNet directory (env: IMAGENET_ROOT)",
    )
    parser.add_argument(
        "--use-adapter", action="store_true",
        help="Apply a trained MetricAdapter before committing to FAM (G11 eval)",
    )
    parser.add_argument(
        "--adapter-path", default="adapter_trained.pt",
        help="Path to trained adapter weights (default: adapter_trained.pt)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device string, e.g. cuda or cpu (default: auto-detect)",
    )
    parser.add_argument(
        "--g6-only", action="store_true", help="Run G6 only; skip G7"
    )
    parser.add_argument(
        "--g7-only", action="store_true", help="Run G7 only; skip G6"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print(
            f"Please provide a valid --root directory (currently: {args.root!r})"
        )
        sys.exit(1)

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    adapter = _load_adapter(args.adapter_path) if args.use_adapter else None
    if adapter is not None:
        adapter = adapter.to(dev)

    if not args.g6_only:
        run_g7_tail_eviction(args.root, dev)
    if not args.g7_only:
        run_g6_granularity(args.root, dev, adapter=adapter)
