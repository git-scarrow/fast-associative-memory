#!/usr/bin/env python3
"""
benchmarks/gauntlet_19_cross_domain.py — Gauntlet 19: Cross-Domain Regression.

Red-team test: does the Triplet Adapter trained on "Dogs" (ImageNet classes
151–200) overfit so severely that it destroys the backbone's general utility on
unrelated tasks (CIFAR-100 / ImageNet-1k)?

Protocol
--------
1. Evaluate FAM **without** any adapter on the generic task → ``acc_baseline``.
2. Load the trained Dogs adapter; evaluate FAM **with** the adapter on the same
   generic task → ``acc_adapted``.
3. Compute accuracy drop: ``drop = acc_baseline − acc_adapted`` (percentage
   points).

Kill Condition (G19)
--------------------
``drop > threshold`` (default 5.0 pp).  If triggered, the adapter is too
domain-specific and requires regularisation or must be kept task-specific.

Usage
-----
    python benchmarks/gauntlet_19_cross_domain.py \\
        --dataset ./feature_cache_vitb14 \\
        --adapter adapter_trained.pt

CLI args
--------
  --dataset    Directory of generic-task feature cache containing ``*train*.pt``
               and ``*test*.pt`` files.
               (default: ./feature_cache_vitb14  — CIFAR-100 DINOv2 ViT-B/14)
  --adapter    Path to trained adapter state-dict (``.pt`` file).
               (default: adapter_trained.pt)
  --input-dim  Backbone embedding dimension that matches the adapter's
               ``input_dim``.  (default: 1024)
  --output-dim Adapter projection dimension; matches the adapter's
               ``output_dim``.  (default: 256)
  --hidden-dim Hidden layer size for the MLP adapter; 0 = linear.
               (default: 512)
  --threshold  Kill-condition threshold in percentage points.  (default: 5.0)
  --seed       Global random seed for reproducibility.  (default: 42)
  --output     Optional path to write a CSV results file.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Allow importing from the repo root regardless of the working directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapter import MetricAdapter  # noqa: E402
from fast_associative_memory import FastAssociativeMemory  # noqa: E402


# ─────────────────────────────────────────────────────────── feature I/O ───

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
            "Run extract_dinov2_vitb14.py (or extract_imagenet_r_vitl14.py) first."
        )

    train = torch.load(train_files[0], map_location=device, weights_only=True)
    test = torch.load(test_files[0], map_location=device, weights_only=True)
    return (
        train["embeds"].float(), train["labels"].long(),
        test["embeds"].float(), test["labels"].long(),
    )


# ─────────────────────────────────────────────────────────── evaluation ───

@torch.no_grad()
def eval_fam_accuracy(
    train_embeds: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeds: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
    num_classes: int,
    adapter: MetricAdapter | None = None,
    core_entries: int = 50000,
    core_vigilance: float = 0.85,
) -> tuple[float, float]:
    """Ingest training embeddings into FAM and evaluate top-1 accuracy.

    The optional *adapter* is passed directly to
    :class:`~fast_associative_memory.FastAssociativeMemory`.  When ``None`` the
    raw backbone embeddings are used (identity baseline).

    Parameters
    ----------
    train_embeds, train_labels:
        Training set (device-resident or CPU tensors; moved to *device*
        internally).
    test_embeds, test_labels:
        Test set.
    device:
        Target compute device.
    num_classes:
        Number of output classes.
    adapter:
        Optional :class:`~adapter.MetricAdapter` to apply before storing keys.
    core_entries:
        FAM prototype capacity.  Defaults to 50 000.
    core_vigilance:
        Vigilance threshold for the continuous CAM.  Defaults to 0.85.

    Returns
    -------
    acc : float
        Top-1 accuracy in %.
    elapsed : float
        Wall-clock time in seconds (ingest + eval).
    """
    embed_dim = train_embeds.size(1)

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
    ).to(device)

    t0 = time.time()

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

    elapsed = time.time() - t0
    acc = 100.0 * correct / len(test_labels)
    return acc, elapsed


# ──────────────────────────────────────────────── kill-condition helpers ───

def accuracy_relative_drop(baseline_acc: float, adapted_acc: float) -> float:
    """Return the absolute accuracy drop in percentage points.

    A positive value indicates degradation (adapted < baseline).

    Parameters
    ----------
    baseline_acc : float
        Top-1 accuracy (%) of the FAM **without** the adapter.
    adapted_acc : float
        Top-1 accuracy (%) of the FAM **with** the Dogs adapter.

    Returns
    -------
    float
        ``baseline_acc − adapted_acc`` (pp).
    """
    return baseline_acc - adapted_acc


def check_kill_condition(drop_pct: float, threshold: float = 5.0) -> bool:
    """Return ``True`` when the accuracy drop exceeds *threshold* pp.

    Parameters
    ----------
    drop_pct : float
        Accuracy drop in percentage points (output of
        :func:`accuracy_relative_drop`).
    threshold : float
        Kill-condition threshold.  Defaults to 5.0 pp.

    Returns
    -------
    bool
        ``True`` if the kill condition is met (adapter too domain-specific).
    """
    return drop_pct > threshold


# ────────────────────────────────────────────────────────── output helpers ───

def _format_table(
    dataset: str,
    results: list[tuple[str, float, float]],
    drop_pct: float,
    threshold: float,
) -> str:
    """Return the formatted results table as a string."""
    sep = "=" * 72
    lines = [
        f"\n{sep}",
        f"  G19 — Cross-Domain Regression  |  dataset: {dataset}",
        sep,
        f"  {'Method':<40} {'Top-1 Acc':>10} {'Time (s)':>10}",
        f"  {'-' * 62}",
    ]
    for method, acc, t in results:
        lines.append(f"  {method:<40} {acc:>9.2f}% {t:>9.1f}")
    lines.append(sep)

    lines.append(
        f"\n  Accuracy drop (baseline − adapted): {drop_pct:+.2f} pp  "
        f"(kill threshold: {threshold:.1f} pp)"
    )
    if check_kill_condition(drop_pct, threshold):
        lines.append(
            "  \u26a0  KILL CONDITION MET: adapter is too domain-specific — "
            "accuracy dropped by more than the allowed threshold."
        )
    else:
        lines.append(
            "  \u2713  PASS: accuracy drop is within the allowed threshold."
        )
    return "\n".join(lines)


def _write_csv(
    results: list[tuple[str, float, float]],
    drop_pct: float,
    threshold: float,
    path: str,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method", "top1_accuracy_pct", "time_s"])
        for method, acc, t in results:
            writer.writerow([method, f"{acc:.4f}", f"{t:.4f}"])
        writer.writerow(["accuracy_drop_pp", f"{drop_pct:.4f}", ""])
        writer.writerow(["kill_condition_threshold_pp", f"{threshold:.4f}", ""])
        writer.writerow(
            ["kill_condition_met", str(check_kill_condition(drop_pct, threshold)), ""]
        )
    print(f"\nResults written to {path}")


# ─────────────────────────────────────────────────────────────── CLI ───

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Gauntlet 19 — Cross-Domain Regression: Dogs adapter vs generic task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="./feature_cache_vitb14",
        metavar="DIR",
        help="Directory containing pre-cached *train*.pt and *test*.pt feature files.",
    )
    parser.add_argument(
        "--adapter",
        default="adapter_trained.pt",
        metavar="FILE",
        help="Path to the trained Dogs adapter state-dict (.pt).",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=1024,
        metavar="D",
        help="Backbone embedding dimension (adapter input_dim).",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=256,
        metavar="D",
        help="Adapter projection dimension (adapter output_dim).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        metavar="D",
        help="Hidden layer size for MLP adapter; 0 = linear adapter.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        metavar="PP",
        help="Kill-condition accuracy-drop threshold (percentage points).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Optional path to write a CSV results file.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Dataset     : {args.dataset}")
    print(f"Adapter     : {args.adapter}")
    print(f"Threshold   : {args.threshold} pp")
    print(f"Seed        : {args.seed}")
    print()

    # ── Load features ──────────────────────────────────────────────────────
    print("Loading features ...")
    train_e, train_l, test_e, test_l = load_features(args.dataset, device)
    num_classes = int(train_l.max().item()) + 1
    print(f"  Train : {tuple(train_e.shape)}  Test : {tuple(test_e.shape)}")
    print(f"  num_classes={num_classes}")
    print()

    results: list[tuple[str, float, float]] = []

    # ── Baseline: FAM without adapter ──────────────────────────────────────
    print("[1/2] Baseline — FAM (no adapter) ...")
    acc_baseline, t_baseline = eval_fam_accuracy(
        train_e, train_l, test_e, test_l, device, num_classes, adapter=None
    )
    results.append(("FAM — no adapter (baseline)", acc_baseline, t_baseline))
    print(f"  top-1={acc_baseline:.2f}%  time={t_baseline:.1f}s")

    # ── Adapted: FAM with trained Dogs adapter ─────────────────────────────
    print(f"[2/2] Adapted — FAM + Dogs adapter ({args.adapter}) ...")
    adapter = MetricAdapter(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
    )
    adapter_path = Path(args.adapter)
    if not adapter_path.is_file():
        print(
            f"  WARNING: adapter file not found at '{args.adapter}'. "
            "Using randomly-initialised adapter — results are not meaningful.",
            file=sys.stderr,
        )
    else:
        adapter.load_state_dict(
            torch.load(args.adapter, map_location=device, weights_only=True)
        )
        print(f"  Loaded adapter weights from '{args.adapter}'")
    adapter.eval()

    acc_adapted, t_adapted = eval_fam_accuracy(
        train_e, train_l, test_e, test_l, device, num_classes, adapter=adapter
    )
    results.append(("FAM — Dogs adapter (adapted)", acc_adapted, t_adapted))
    print(f"  top-1={acc_adapted:.2f}%  time={t_adapted:.1f}s")

    # ── Kill-condition evaluation ──────────────────────────────────────────
    drop = accuracy_relative_drop(acc_baseline, acc_adapted)
    table = _format_table(args.dataset, results, drop, args.threshold)
    print(table)

    if args.output:
        _write_csv(results, drop, args.threshold, args.output)


if __name__ == "__main__":
    main()
