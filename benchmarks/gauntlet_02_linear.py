#!/usr/bin/env python3
"""
benchmarks/gauntlet_02_linear.py — Gauntlet 2: Parametric Floor.

Red-team test: does a single linear layer match or beat FastAssociativeMemory?

Implements two streaming linear baselines trained on the same frozen DINOv2
embeddings used by Gauntlet 1 (evaluate_baselines.py):

  2a  Streaming AdamW linear probe with a fixed-capacity ring-buffer replay
  2b  Streaming Ridge Regression with Sherman-Morrison rank-1 matrix inverse
      updates (no replay buffer, O(D²) memory)

Kill condition: if any linear baseline achieves ≥ FAM top-1 accuracy, the
associative-memory mechanism (storing instances) is unnecessary overhead.

Usage
-----
  python benchmarks/gauntlet_02_linear.py \\
      --dataset ./feature_cache_inr_vitl14 \\
      --backbone dinov2_vitl14 \\
      --seed 42 --buffer-size 256 --lambda 1.0

CLI args
--------
  --dataset     Path to the feature cache directory produced by
                extract_imagenet_r_vitl14.py or extract_dinov2_vitb14.py.
                The script auto-discovers *train*.pt / *test*.pt files inside.
                (default: ./feature_cache_inr_vitl14)
  --backbone    Backbone name — informational only, printed in the header.
                (default: dinov2_vitl14)
  --seed        Global random seed for reproducibility.  (default: 42)
  --buffer-size Replay-buffer capacity for baseline 2a.  (default: 256)
  --lambda      Ridge regularisation λ for baseline 2b.  (default: 1.0)
  --output      Optional path to write a CSV results file.
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow importing from the repo root regardless of working directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from fast_associative_memory import FastAssociativeMemory  # noqa: E402


# ──────────────────────────────────────────────────────────── helpers ───

def set_seed(seed: int) -> None:
    """Set global seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)


def _tensor_mb(tensors) -> float:
    """Return total memory in MB for an iterable of tensors."""
    return sum(t.numel() * t.element_size() for t in tensors) / (1024 ** 2)


def load_features(cache_dir: str, device: torch.device):
    """Load pre-cached embeddings from *cache_dir*.

    Looks for files matching ``*train*.pt`` and ``*test*.pt``.  Each file must
    contain a dict with ``"embeds"`` (float32 tensor) and ``"labels"`` (int64).

    Raises
    ------
    FileNotFoundError
        When no matching ``.pt`` files are found in *cache_dir*.
    """
    cdir = Path(cache_dir)
    pt_files = list(cdir.glob("*.pt"))
    train_files = [f for f in pt_files if "train" in f.name]
    test_files = [f for f in pt_files if "test" in f.name]

    if not train_files or not test_files:
        raise FileNotFoundError(
            f"No train/test .pt files found in '{cache_dir}'.\n"
            "Run extract_imagenet_r_vitl14.py (or extract_dinov2_vitb14.py) first."
        )

    train = torch.load(train_files[0], map_location=device, weights_only=True)
    test = torch.load(test_files[0], map_location=device, weights_only=True)
    return (
        train["embeds"].float(), train["labels"].long(),
        test["embeds"].float(), test["labels"].long(),
    )


# ─────────────────────────────────────── 2a: AdamW streaming linear probe ───

class _RingBuffer:
    """Fixed-capacity ring buffer for experience replay.

    Memory footprint: ``capacity × embed_dim × 4`` bytes (float32) plus
    ``capacity × 8`` bytes (int64 labels).
    """

    def __init__(self, capacity: int, embed_dim: int, device: torch.device) -> None:
        self.capacity = capacity
        self.size = 0
        self._ptr = 0
        self._embeds = torch.zeros(capacity, embed_dim, device=device)
        self._labels = torch.zeros(capacity, dtype=torch.long, device=device)

    def add(self, embeds: torch.Tensor, labels: torch.Tensor) -> None:
        """Insert a mini-batch using random replacement once full."""
        for i in range(embeds.size(0)):
            self._embeds[self._ptr] = embeds[i]
            self._labels[self._ptr] = labels[i]
            self._ptr = (self._ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int):
        """Sample *n* random entries with replacement from the current buffer."""
        n = min(n, self.size)
        idx = torch.randint(0, self.size, (n,), device=self._embeds.device)
        return self._embeds[idx], self._labels[idx]

    def tensors(self):
        return (self._embeds, self._labels)


def train_adamw_probe(
    train_embeds: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeds: torch.Tensor,
    test_labels: torch.Tensor,
    embed_dim: int,
    num_classes: int,
    buffer_size: int,
    device: torch.device,
    seed: int,
    batch_size: int = 32,
):
    """Baseline 2a — streaming AdamW linear probe with replay buffer.

    Architecture : ``nn.Linear(embed_dim, num_classes, bias=True)``
    Optimizer    : AdamW, lr=1e-3
    Replay buffer: ring buffer, capacity=*buffer_size*, random replacement
    Each step    : gradient step on current mini-batch ∪ replay sample

    Returns
    -------
    acc : float
        Top-1 test accuracy (%).
    mem_mb : float
        Memory consumed by linear layer + replay buffer (MB).
    train_time : float
        Wall-clock training time (seconds).
    """
    set_seed(seed)
    linear = nn.Linear(embed_dim, num_classes, bias=True).to(device)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=1e-3)
    buf = _RingBuffer(buffer_size, embed_dim, device)

    t0 = time.time()
    for i in range(0, len(train_embeds), batch_size):
        x_cur = train_embeds[i: i + batch_size]
        y_cur = train_labels[i: i + batch_size]

        buf.add(x_cur, y_cur)

        # Wait until the buffer has at least one full batch before training.
        if buf.size < batch_size:
            continue

        x_rep, y_rep = buf.sample(batch_size)
        x_all = torch.cat([x_cur, x_rep], dim=0)
        y_all = torch.cat([y_cur, y_rep], dim=0)

        optimizer.zero_grad()
        loss = F.cross_entropy(linear(x_all), y_all)
        loss.backward()
        optimizer.step()

    train_time = time.time() - t0

    # Evaluate
    linear.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(test_embeds), 512):
            logits = linear(test_embeds[i: i + 512])
            correct += (logits.argmax(dim=1) == test_labels[i: i + 512]).sum().item()

    acc = 100.0 * correct / len(test_labels)
    mem_mb = _tensor_mb(list(linear.parameters())) + _tensor_mb(buf.tensors())
    return acc, mem_mb, train_time


# ──────────────────────────────── 2b: Streaming Ridge / Sherman-Morrison ───

def train_ridge_probe(
    train_embeds: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeds: torch.Tensor,
    test_labels: torch.Tensor,
    embed_dim: int,
    num_classes: int,
    lam: float,
    device: torch.device,
    seed: int,
):
    """Baseline 2b — streaming ridge regression via Sherman-Morrison updates.

    Maintains the Woodbury / Sherman-Morrison incremental inverse of
    ``A = X_aug^T X_aug + λI``, where ``X_aug`` appends a bias column of 1s
    to the embedding so that no separate bias term is needed.

    Memory footprint: ``(D+1)² × 4`` bytes for ``A⁻¹`` plus
    ``(D+1) × num_classes × 4`` bytes for ``b = X_aug^T Y``.

    Update rule (rank-1 Sherman-Morrison each new sample)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Let ``v = A⁻¹ x_aug``.  Then::

        A⁻¹  ←  A⁻¹  −  outer(v, v) / (1 + x_aug · v)
        b    ←  b    +  outer(x_aug, y_onehot)

    Prediction: ``W = A⁻¹ b``,  ``logits = x_feat @ W[:-1] + W[-1]``

    Returns
    -------
    acc : float
        Top-1 test accuracy (%).
    mem_mb : float
        Memory consumed by ``A⁻¹`` and ``b`` (MB).
    train_time : float
        Wall-clock training time (seconds).
    """
    set_seed(seed)
    D = embed_dim + 1  # augmented dimension (+1 for implicit bias)

    # Initialise: A = λI  ⟹  A⁻¹ = (1/λ) I
    A_inv = torch.eye(D, device=device) / lam
    b = torch.zeros(D, num_classes, device=device)

    t0 = time.time()
    for i in range(len(train_embeds)):
        x = train_embeds[i]                              # (embed_dim,)
        x_aug = torch.cat([x, x.new_ones(1)])            # (D,)
        y = F.one_hot(train_labels[i], num_classes=num_classes).float()  # (num_classes,)
        # Sherman-Morrison rank-1 update of A⁻¹
        v = A_inv @ x_aug                                # (D,)
        denom = 1.0 + x_aug @ v                          # scalar tensor
        A_inv -= torch.outer(v, v) / denom               # avoids CPU–GPU sync

        # Accumulate right-hand side
        b.addr_(x_aug, y)                                # in-place outer add

    train_time = time.time() - t0

    # Closed-form solution: W = A⁻¹ b  shape (D, num_classes)
    W = A_inv @ b
    W_feat = W[:-1]   # (embed_dim, num_classes)
    W_bias = W[-1]    # (num_classes,)

    # Evaluate
    correct = 0
    with torch.no_grad():
        for i in range(0, len(test_embeds), 512):
            logits = test_embeds[i: i + 512] @ W_feat + W_bias
            correct += (logits.argmax(dim=1) == test_labels[i: i + 512]).sum().item()

    acc = 100.0 * correct / len(test_labels)
    mem_mb = _tensor_mb([A_inv, b])
    return acc, mem_mb, train_time


# ──────────────────────────────────────────────────── FAM reference eval ───

def eval_fam(
    train_embeds: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeds: torch.Tensor,
    test_labels: torch.Tensor,
    embed_dim: int,
    num_classes: int,
    device: torch.device,
):
    """Reference evaluation of FastAssociativeMemory (Gauntlet 1 settings).

    Uses the same hyper-parameters as ``evaluate_baselines.py`` so that the
    comparison is fair.

    Returns
    -------
    acc : float
    mem_mb : float
        Approximate memory for the prototype keys + values tensors (MB).
    train_time : float
        Combined ingest + eval wall-clock time (seconds).
    """
    mem = FastAssociativeMemory(
        input_dim=embed_dim,
        value_dim=num_classes,
        core_entries=50000,
        core_vigilance=0.85,
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=25,
        inference_temp=0.05,
        use_lfu=True,
    ).to(device)

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(train_embeds), 256):
            mem.learn_local(train_embeds[i: i + 256], train_labels[i: i + 256])

        correct = 0
        for i in range(0, len(test_embeds), 512):
            logits = mem(test_embeds[i: i + 512])
            correct += (logits.argmax(dim=1) == test_labels[i: i + 512]).sum().item()

    train_time = time.time() - t0
    acc = 100.0 * correct / len(test_labels)

    cam = mem.core_cam
    mem_mb = _tensor_mb([cam.keys, cam.values])
    return acc, mem_mb, train_time


# ─────────────────────────────────────────────────────────────── CLI ───

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Gauntlet 2 — Parametric Floor: streaming linear probes vs FAM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="./feature_cache_inr_vitl14",
        metavar="DIR",
        help="Directory containing pre-cached *train*.pt and *test*.pt feature files.",
    )
    parser.add_argument(
        "--backbone",
        default="dinov2_vitl14",
        help="Backbone name (informational; used in the printed header).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=256,
        metavar="N",
        help="Replay-buffer capacity for baseline 2a (samples).",
    )
    parser.add_argument(
        "--lambda",
        dest="lam",
        type=float,
        default=1.0,
        metavar="λ",
        help="Ridge regularisation λ for baseline 2b.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Optional path to write a CSV results file.",
    )
    return parser.parse_args(argv)


def _print_table(results):
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {'Method':<35} {'top-1 acc':>10} {'memory (MB)':>12} {'train time (s)':>14}")
    print(f"  {'-' * 66}")
    for method, acc, mem, t in results:
        print(f"  {method:<35} {acc:>9.2f}% {mem:>11.1f} {t:>13.1f}")
    print(sep)


def _write_csv(results, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method", "top1_accuracy_pct", "memory_mb", "train_time_s"])
        for method, acc, mem, t in results:
            writer.writerow([method, f"{acc:.4f}", f"{mem:.4f}", f"{t:.4f}"])
    print(f"\nResults written to {path}")


def main(argv=None) -> None:
    args = _parse_args(argv)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Backbone    : {args.backbone}")
    print(f"Cache dir   : {args.dataset}")
    print(f"Seed        : {args.seed}")
    print(f"Buffer size : {args.buffer_size}  (2a)")
    print(f"Lambda      : {args.lam}  (2b)")
    print()

    print("Loading features ...")
    train_e, train_l, test_e, test_l = load_features(args.dataset, device)
    embed_dim = train_e.size(1)
    num_classes = int(train_l.max().item()) + 1
    print(f"  Train : {tuple(train_e.shape)}  Test : {tuple(test_e.shape)}")
    print(f"  embed_dim={embed_dim}  num_classes={num_classes}")
    print()

    results = []

    # ── 2a: AdamW linear probe ───────────────────────────────────────────
    print("[1/3] 2a — Streaming AdamW linear probe (replay buffer) ...")
    acc_a, mem_a, t_a = train_adamw_probe(
        train_e, train_l, test_e, test_l,
        embed_dim, num_classes, args.buffer_size, device, args.seed,
    )
    results.append(("2a AdamW linear probe", acc_a, mem_a, t_a))
    print(f"  top-1={acc_a:.2f}%  memory={mem_a:.1f} MB  time={t_a:.1f}s")

    # ── 2b: Ridge regression ─────────────────────────────────────────────
    print("[2/3] 2b — Streaming Ridge Regression (Sherman-Morrison) ...")
    acc_b, mem_b, t_b = train_ridge_probe(
        train_e, train_l, test_e, test_l,
        embed_dim, num_classes, args.lam, device, args.seed,
    )
    results.append(("2b Ridge Regression", acc_b, mem_b, t_b))
    print(f"  top-1={acc_b:.2f}%  memory={mem_b:.1f} MB  time={t_b:.1f}s")

    # ── FAM reference ─────────────────────────────────────────────────────
    print("[3/3] FAM — FastAssociativeMemory (Gauntlet 1 reference) ...")
    acc_fam, mem_fam, t_fam = eval_fam(
        train_e, train_l, test_e, test_l, embed_dim, num_classes, device,
    )
    results.append(("FAM (reference)", acc_fam, mem_fam, t_fam))
    print(f"  top-1={acc_fam:.2f}%  memory={mem_fam:.1f} MB  time={t_fam:.1f}s")

    # ── Results table ────────────────────────────────────────────────────
    _print_table(results)

    # Kill-condition check
    for method, acc, _m, _t in results[:-1]:
        if acc >= acc_fam:
            print(
                f"\n⚠  KILL CONDITION MET: {method} (top-1={acc:.2f}%) "
                f"≥ FAM ({acc_fam:.2f}%)"
            )
            print(
                "   Associative memory mechanism may be unnecessary overhead."
            )

    if args.output:
        _write_csv(results, args.output)


if __name__ == "__main__":
    main()
