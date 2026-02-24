#!/usr/bin/env python3
"""
benchmarks/gauntlet_01_ncm.py — Gauntlet 1: "Is It Just the Features?" (NCM Variants)

Red-team phase 1. Determines whether FAM's multi-prototype architecture provides
any value beyond what frozen DINOv2 features alone can achieve using trivial
classifiers.

Kill condition: If FAM ≤ NCM + 1.0% top-1 accuracy, the multi-prototype
machinery is redundant — the backbone is doing all the work.

Baselines implemented:
  1a. SCM  — Streaming Class Mean: one running mean per class, cosine similarity.
  1b. NCM + Mahalanobis: per-class diagonal covariance (Welford online), predict
      via argmin diagonal Mahalanobis distance.

Usage:
  python benchmarks/gauntlet_01_ncm.py [--dataset DATASET] [--backbone BACKBONE]
                                        [--seed SEED] [--output OUTPUT]
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

# Allow running from the repo root or from the benchmarks/ subdirectory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Feature cache configuration
# ---------------------------------------------------------------------------

_CACHE_CONFIG = {
    # (dataset, backbone) -> (cache_dir, train_filename, test_filename)
    ("imagenet_r", "dinov2_vitl14"): (
        "./feature_cache_inr_vitl14",
        "imagenetr_dinov2_train.pt",
        "imagenetr_dinov2_test.pt",
    ),
    ("cifar100", "dinov2_vitb14"): (
        "./feature_cache_vitb14",
        "cifar100_dinov2_train.pt",
        "cifar100_dinov2_test.pt",
    ),
}


def _cache_paths(dataset: str, backbone: str):
    """Return (cache_dir, train_path, test_path) for the given dataset/backbone pair."""
    key = (dataset, backbone)
    if key in _CACHE_CONFIG:
        cache_dir, train_file, test_file = _CACHE_CONFIG[key]
    else:
        cache_dir = f"./feature_cache_{dataset}_{backbone}"
        train_file = f"{dataset}_dinov2_train.pt"
        test_file = f"{dataset}_dinov2_test.pt"
    return cache_dir, os.path.join(cache_dir, train_file), os.path.join(cache_dir, test_file)


# ---------------------------------------------------------------------------
# Feature loading / extraction
# ---------------------------------------------------------------------------

def load_or_extract_features(dataset: str, backbone: str, seed: int,
                              device: torch.device):
    """Return (train_embeds, train_labels, test_embeds, test_labels) on *device*.

    Loads from disk cache when available; otherwise runs the frozen backbone once
    and writes the result to disk for future runs.
    """
    cache_dir, train_path, test_path = _cache_paths(dataset, backbone)

    if os.path.isfile(train_path) and os.path.isfile(test_path):
        print(f"Loading cached features from {cache_dir} ...")
        train = torch.load(train_path, map_location=device, weights_only=True)
        test = torch.load(test_path, map_location=device, weights_only=True)
        return (
            train["embeds"].float(),
            train["labels"].long(),
            test["embeds"].float(),
            test["labels"].long(),
        )

    print(f"No cached features found at {cache_dir}. Extracting with {backbone} ...")
    return _extract_and_cache(dataset, backbone, seed, device,
                              cache_dir, train_path, test_path)


def _stratified_split(dataset, train_ratio: float, seed: int):
    """Deterministic stratified split (mirrors extract_imagenet_r_vitl14.py)."""
    from torch.utils.data import Subset

    targets = [dataset.targets[i] for i in range(len(dataset))]
    classes = sorted(set(targets))
    gen = torch.Generator().manual_seed(seed)
    train_indices, test_indices = [], []
    for cls in classes:
        cls_indices = [i for i, t in enumerate(targets) if t == cls]
        n_train = int(len(cls_indices) * train_ratio)
        perm = torch.randperm(len(cls_indices), generator=gen).tolist()
        train_indices.extend(cls_indices[p] for p in perm[:n_train])
        test_indices.extend(cls_indices[p] for p in perm[n_train:])
    return Subset(dataset, train_indices), Subset(dataset, test_indices)


def _extract_and_cache(dataset: str, backbone: str, seed: int,
                       device: torch.device, cache_dir: str,
                       train_path: str, test_path: str):
    """Extract frozen DINOv2 features and write them to *cache_dir*."""
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)

    model = torch.hub.load("facebookresearch/dinov2", backbone)
    model = model.to(device).eval()

    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    if dataset == "cifar100":
        from torchvision.datasets import CIFAR100
        train_ds = CIFAR100(root="./data", train=True, download=True, transform=transform)
        test_ds = CIFAR100(root="./data", train=False, download=True, transform=transform)
    elif dataset == "imagenet_r":
        from torchvision.datasets import ImageFolder
        full_ds = ImageFolder(root="./data/imagenet-r", transform=transform)
        train_ds, test_ds = _stratified_split(full_ds, 0.8, seed)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Supported: 'cifar100', 'imagenet_r'."
        )

    @torch.no_grad()
    def extract(loader):
        embeds, labels = [], []
        for images, lbls in loader:
            embeds.append(model(images.to(device)).cpu())
            labels.append(lbls)
        return torch.cat(embeds).float(), torch.cat(labels).long()

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print("  Extracting train features ...")
    train_embeds, train_labels = extract(train_loader)
    print(f"    {train_embeds.shape}")

    print("  Extracting test features ...")
    test_embeds, test_labels = extract(test_loader)
    print(f"    {test_embeds.shape}")

    os.makedirs(cache_dir, exist_ok=True)
    torch.save({"embeds": train_embeds, "labels": train_labels}, train_path)
    torch.save({"embeds": test_embeds, "labels": test_labels}, test_path)
    print(f"  Features cached to {cache_dir}")

    return (
        train_embeds.to(device),
        train_labels.to(device),
        test_embeds.to(device),
        test_labels.to(device),
    )


# ---------------------------------------------------------------------------
# 1a — Streaming Class Mean (SCM)
# ---------------------------------------------------------------------------

class StreamingClassMean:
    """SimpleCIL-style streaming mean: one running mean vector per class.

    Memory: O(num_classes × embed_dim).  No replay buffer — only the current
    class mean is kept per class.
    """

    def __init__(self):
        self.means: dict[int, torch.Tensor] = {}
        self.counts: dict[int, int] = {}

    def update(self, emb: torch.Tensor, class_id: int):
        """Incremental mean update for a single float embedding vector."""
        if class_id not in self.means:
            self.means[class_id] = emb.clone()
            self.counts[class_id] = 1
        else:
            n = self.counts[class_id]
            self.means[class_id] = (self.means[class_id] * n + emb) / (n + 1)
            self.counts[class_id] += 1

    def predict_batch(self, embs: torch.Tensor) -> torch.Tensor:
        """Return predicted class ids (argmax cosine similarity to class means)."""
        class_ids = sorted(self.means.keys())
        mean_matrix = torch.stack([self.means[c] for c in class_ids])  # (C, D)
        embs_norm = F.normalize(embs.float(), dim=-1)
        means_norm = F.normalize(mean_matrix.to(embs.device), dim=-1)
        sims = embs_norm @ means_norm.T                                 # (B, C)
        pred_local = sims.argmax(dim=1)
        id_tensor = torch.tensor(class_ids, device=embs.device)
        return id_tensor[pred_local]

    def memory_bytes(self) -> int:
        """Bytes consumed by stored class means (float32)."""
        if not self.means:
            return 0
        dim = next(iter(self.means.values())).numel()
        return len(self.means) * dim * 4  # float32


# ---------------------------------------------------------------------------
# 1b — NCM + Mahalanobis (diagonal covariance, Welford online)
# ---------------------------------------------------------------------------

class NCMMahalanobis:
    """NCM with per-class diagonal variance estimated via Welford's online algorithm.

    Prediction: argmin_c  sum_d  (x_d - μ_cd)² / σ²_cd
    (diagonal Mahalanobis; reduces to normalized Euclidean per dimension).

    Memory: O(2 × num_classes × embed_dim).  No replay buffer — only the
    running mean and M2 accumulator are kept.
    """

    def __init__(self, var_eps: float = 1e-4):
        self.means: dict[int, torch.Tensor] = {}
        self.M2: dict[int, torch.Tensor] = {}   # Welford M2 accumulator
        self.counts: dict[int, int] = {}
        self.var_eps = var_eps

    def update(self, emb: torch.Tensor, class_id: int):
        """Welford online mean and M2 update for a single embedding."""
        if class_id not in self.means:
            self.means[class_id] = emb.clone()
            self.M2[class_id] = torch.zeros_like(emb)
            self.counts[class_id] = 1
        else:
            n = self.counts[class_id] + 1
            delta = emb - self.means[class_id]
            self.means[class_id] = self.means[class_id] + delta / n
            delta2 = emb - self.means[class_id]
            self.M2[class_id] = self.M2[class_id] + delta * delta2
            self.counts[class_id] = n

    def _variance(self, class_id: int) -> torch.Tensor:
        """Per-dimension variance (Bessel-corrected).  Falls back to 1 for n < 2."""
        n = self.counts[class_id]
        if n < 2:
            return torch.ones_like(self.means[class_id])
        return (self.M2[class_id] / (n - 1)).clamp(min=self.var_eps)

    def predict_batch(self, embs: torch.Tensor) -> torch.Tensor:
        """Return predicted class ids (argmin diagonal Mahalanobis distance).

        Vectorised: builds (C, D) mean and variance matrices then computes the
        full (B, C) distance matrix in one shot.
        """
        class_ids = sorted(self.means.keys())
        dev = embs.device
        means = torch.stack([self.means[c] for c in class_ids]).to(dev)   # (C, D)
        vars_ = torch.stack([self._variance(c) for c in class_ids]).to(dev)  # (C, D)

        # (B, C, D) = embs[:, None, :] - means[None, :, :]
        diff = embs.float().unsqueeze(1) - means.unsqueeze(0)             # (B, C, D)
        dists = (diff ** 2 / vars_.unsqueeze(0)).sum(dim=-1)              # (B, C)
        pred_local = dists.argmin(dim=1)
        id_tensor = torch.tensor(class_ids, device=dev)
        return id_tensor[pred_local]

    def memory_bytes(self) -> int:
        """Bytes consumed by means + M2 accumulators (float32)."""
        if not self.means:
            return 0
        dim = next(iter(self.means.values())).numel()
        return len(self.means) * 2 * dim * 4  # float32


# ---------------------------------------------------------------------------
# Evaluation runners
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_scm(train_embeds: torch.Tensor, train_labels: torch.Tensor,
            test_embeds: torch.Tensor, test_labels: torch.Tensor,
            device: torch.device):
    """Stream all training samples through SCM, then evaluate on the test set."""
    scm = StreamingClassMean()

    # Streaming update — one sample at a time (matches issue pseudocode)
    n_train = train_embeds.size(0)
    for i in range(n_train):
        scm.update(train_embeds[i].to(device), int(train_labels[i].item()))

    # Batched evaluation
    correct = 0
    batch_size = 512
    n_test = test_embeds.size(0)
    for i in range(0, n_test, batch_size):
        embs = test_embeds[i:i + batch_size].to(device)
        lbls = test_labels[i:i + batch_size].to(device)
        preds = scm.predict_batch(embs)
        correct += (preds == lbls).sum().item()

    acc = 100.0 * correct / n_test
    mem_mb = scm.memory_bytes() / (1024 ** 2)
    return acc, mem_mb


@torch.no_grad()
def run_ncm_mahalanobis(train_embeds: torch.Tensor, train_labels: torch.Tensor,
                        test_embeds: torch.Tensor, test_labels: torch.Tensor,
                        device: torch.device):
    """Stream all training samples through NCM+Mahalanobis, then evaluate."""
    ncm = NCMMahalanobis()

    n_train = train_embeds.size(0)
    for i in range(n_train):
        ncm.update(train_embeds[i].to(device), int(train_labels[i].item()))

    correct = 0
    # Smaller batch: (B, C, D) intermediate tensor grows with num_classes
    batch_size = 256
    n_test = test_embeds.size(0)
    for i in range(0, n_test, batch_size):
        embs = test_embeds[i:i + batch_size].to(device)
        lbls = test_labels[i:i + batch_size].to(device)
        preds = ncm.predict_batch(embs)
        correct += (preds == lbls).sum().item()

    acc = 100.0 * correct / n_test
    mem_mb = ncm.memory_bytes() / (1024 ** 2)
    return acc, mem_mb


@torch.no_grad()
def run_fam(train_embeds: torch.Tensor, train_labels: torch.Tensor,
            test_embeds: torch.Tensor, test_labels: torch.Tensor,
            device: torch.device, num_classes: int):
    """Run FastAssociativeMemory and return (accuracy, memory_mb, n_prototypes)."""
    from fast_associative_memory import FastAssociativeMemory

    embed_dim = train_embeds.size(1)
    n_train = train_embeds.size(0)

    mem = FastAssociativeMemory(
        input_dim=embed_dim,
        value_dim=num_classes,
        core_entries=min(n_train, 50000),
        core_vigilance=0.85,
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=25,
        inference_temp=0.05,
        use_lfu=True,
    ).to(device)

    x_train = train_embeds.to(device)
    y_train = train_labels.to(device)
    for i in range(0, n_train, 256):
        mem.learn_local(x_train[i:i + 256], y_train[i:i + 256])

    n_occupied = int(mem.core_cam.occupied.sum().item())

    x_test = test_embeds.to(device)
    y_test = test_labels.to(device)
    correct = 0
    for i in range(0, len(x_test), 512):
        logits = mem(x_test[i:i + 512])
        correct += (logits.argmax(dim=1) == y_test[i:i + 512]).sum().item()

    acc = 100.0 * correct / len(test_labels)

    # Allocated prototype storage (keys + values tensors only)
    cam = mem.core_cam
    mem_bytes = (cam.keys.numel() * cam.keys.element_size()
                 + cam.values.numel() * cam.values.element_size())
    mem_mb = mem_bytes / (1024 ** 2)

    return acc, mem_mb, n_occupied


# ---------------------------------------------------------------------------
# Results table helpers
# ---------------------------------------------------------------------------

def _format_table(dataset: str, backbone: str,
                  results: list[tuple[str, float, float]],
                  acc_scm: float, acc_ncm: float, acc_fam: float) -> str:
    """Return the formatted results table as a string."""
    sep = "=" * 72
    lines = [
        f"\n{sep}",
        f"  {dataset.upper()} — {backbone} — Gauntlet 1: NCM Baselines",
        sep,
        f"  {'Method':<44} {'Top-1 Acc':>10} {'Memory':>10}",
        f"  {'-' * 66}",
    ]
    for method, acc, mem_mb in results:
        lines.append(f"  {method:<44} {acc:>9.2f}% {mem_mb:>8.2f} MB")
    lines.append(sep)

    best_ncm = max(acc_scm, acc_ncm)
    delta = acc_fam - best_ncm
    lines.append(f"\n  FAM vs best NCM: {delta:+.2f}%  "
                 f"(kill condition: FAM \u2264 NCM + 1.0%)")
    if delta <= 1.0:
        lines.append("  \u26a0  KILL CONDITION MET: multi-prototype machinery is redundant.")
    else:
        lines.append("  \u2713  FAM exceeds the best NCM baseline by more than 1%.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gauntlet 1: NCM Baseline — Is It Just the Features?"
    )
    parser.add_argument(
        "--dataset", default="imagenet_r",
        choices=["imagenet_r", "cifar100"],
        help="Evaluation dataset (default: imagenet_r)",
    )
    parser.add_argument(
        "--backbone", default="dinov2_vitl14",
        choices=["dinov2_vitb14", "dinov2_vitl14"],
        help="Frozen DINOv2 backbone variant (default: dinov2_vitl14)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional file path to save the printed results table",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Dataset : {args.dataset}  |  Backbone: {args.backbone}  |  Seed: {args.seed}")

    train_embeds, train_labels, test_embeds, test_labels = load_or_extract_features(
        args.dataset, args.backbone, args.seed, device
    )
    num_classes = int(train_labels.max().item()) + 1
    print(f"Train   : {train_embeds.shape}  |  Test: {test_embeds.shape}"
          f"  |  Classes: {num_classes}")

    results: list[tuple[str, float, float]] = []

    # 1a — SCM
    print("\n[1/3] Streaming Class Mean (SCM) ...")
    t0 = time.time()
    acc_scm, mem_scm = run_scm(train_embeds, train_labels,
                                test_embeds, test_labels, device)
    t1 = time.time()
    print(f"  top-1: {acc_scm:.2f}%  memory: {mem_scm:.2f} MB  ({t1 - t0:.1f}s)")
    results.append(("SCM (Streaming Class Mean)", acc_scm, mem_scm))

    # 1b — NCM + Mahalanobis
    print("\n[2/3] NCM + Mahalanobis (diagonal covariance) ...")
    t0 = time.time()
    acc_ncm, mem_ncm = run_ncm_mahalanobis(train_embeds, train_labels,
                                            test_embeds, test_labels, device)
    t1 = time.time()
    print(f"  top-1: {acc_ncm:.2f}%  memory: {mem_ncm:.2f} MB  ({t1 - t0:.1f}s)")
    results.append(("NCM + Mahalanobis (diagonal)", acc_ncm, mem_ncm))

    # FAM
    print("\n[3/3] FastAssociativeMemory (FAM) ...")
    t0 = time.time()
    acc_fam, mem_fam, n_protos = run_fam(
        train_embeds, train_labels, test_embeds, test_labels, device, num_classes
    )
    t1 = time.time()
    print(f"  top-1: {acc_fam:.2f}%  memory: {mem_fam:.2f} MB"
          f"  ({t1 - t0:.1f}s, {n_protos} prototypes)")
    results.append((f"FAM (multi-prototype, {n_protos} slots)", acc_fam, mem_fam))

    # Print table
    table = _format_table(args.dataset, args.backbone, results,
                          acc_scm, acc_ncm, acc_fam)
    print(table)

    if args.output:
        with open(args.output, "w") as f:
            f.write(table + "\n")
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
