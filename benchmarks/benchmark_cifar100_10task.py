#!/usr/bin/env python3
"""
benchmarks/benchmark_cifar100_10task.py — G19: Adapter Generalization Stress Test.

Verifies that a Triplet Adapter trained on the Dogs subset (G6, ImageNet classes
151–200) has not overfit and destroyed the backbone's general utility on generic
tasks (CIFAR-100 or ImageNet-style datasets).

Two conditions are compared:

  Baseline  FastAssociativeMemory with no adapter (Identity — raw backbone
            embeddings fed directly to FAM).
  Trained   FastAssociativeMemory with the Dogs-trained Triplet Adapter applied
            before FAM ingestion.

Kill Condition (G19)
--------------------
Accuracy on the generic task drops > 5 % *relative* to the baseline (no adapter).

Success Criteria
----------------
Accuracy degradation < 5 % relative to baseline → adapter has not overfit.

Usage
-----
  # Baseline only (no adapter path supplied):
  python benchmarks/benchmark_cifar100_10task.py

  # Compare against a trained adapter:
  python benchmarks/benchmark_cifar100_10task.py --adapter adapter_trained.pt

  # Full options:
  python benchmarks/benchmark_cifar100_10task.py \\
      --dataset ./feature_cache_vitb14 \\
      --adapter adapter_trained.pt \\
      --adapter-input-dim 1024 \\
      --adapter-output-dim 256 \\
      --adapter-hidden-dim 512 \\
      --seed 42 \\
      --output results_g19.csv

CLI args
--------
  --dataset            Path to a feature-cache directory containing
                       ``*train*.pt`` and ``*test*.pt`` files with keys
                       ``"embeds"`` and ``"labels"``.
                       (default: ./feature_cache_vitb14)
  --adapter            Optional path to a saved adapter state-dict
                       (``adapter_trained.pt``).  When omitted only the
                       baseline (no adapter) is evaluated.
  --adapter-input-dim  Input dimensionality of the adapter (must match the
                       backbone embedding).  (default: 768)
  --adapter-output-dim Output dimensionality of the adapter projection.
                       (default: 256)
  --adapter-hidden-dim Hidden layer size (0 → linear adapter).  (default: 512)
  --threshold          Kill-condition degradation threshold in percentage
                       points.  (default: 5.0)
  --seed               Global random seed.  (default: 42)
  --output             Optional path to write a CSV results file.
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Allow importing from the repo root regardless of working directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapter import MetricAdapter  # noqa: E402
from fast_associative_memory import FastAssociativeMemory  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set global seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)


def load_features(cache_dir: str, device: torch.device):
    """Load pre-cached embeddings from *cache_dir*.

    Looks for files matching ``*train*.pt`` and ``*test*.pt``.  Each file
    must contain a dict with ``"embeds"`` (float32) and ``"labels"`` (int64).

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
            "Run extract_dinov2_vitb14.py (or similar) first."
        )

    if len(train_files) > 1:
        import warnings
        warnings.warn(
            f"Multiple train .pt files found; using '{train_files[0].name}'."
        )
    if len(test_files) > 1:
        import warnings
        warnings.warn(
            f"Multiple test .pt files found; using '{test_files[0].name}'."
        )

    train = torch.load(train_files[0], map_location=device, weights_only=True)
    test = torch.load(test_files[0], map_location=device, weights_only=True)
    return (
        train["embeds"].float(), train["labels"].long(),
        test["embeds"].float(), test["labels"].long(),
    )


def load_adapter(
    path: str,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    device: torch.device,
) -> MetricAdapter:
    """Load a :class:`~adapter.MetricAdapter` from a saved state-dict.

    Args:
        path:       File path to the saved ``torch.save`` state-dict.
        input_dim:  Must match the backbone embedding dimensionality.
        output_dim: Must match the output dimensionality used during training.
        hidden_dim: 0 for a linear adapter; > 0 for two-layer MLP.
        device:     Target device.

    Returns:
        Loaded :class:`~adapter.MetricAdapter` in eval mode.
    """
    adapter = MetricAdapter(input_dim=input_dim, output_dim=output_dim,
                            hidden_dim=hidden_dim)
    state = torch.load(path, map_location=device, weights_only=True)
    adapter.load_state_dict(state)
    adapter.to(device).eval()
    return adapter


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def eval_fam(
    train_embeds: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeds: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
    adapter: MetricAdapter | None = None,
) -> tuple[float, float]:
    """Evaluate FastAssociativeMemory with an optional adapter.

    Parameters
    ----------
    train_embeds, train_labels : Tensor
        Training set embeddings and class labels.
    test_embeds, test_labels : Tensor
        Test set embeddings and class labels.
    num_classes : int
        Total number of classes in the dataset.
    device : torch.device
        Compute device.
    adapter : MetricAdapter or None
        When ``None``, raw backbone embeddings are fed to FAM (identity /
        baseline condition).  When provided, the adapter projection is applied
        before FAM ingestion and retrieval.

    Returns
    -------
    acc : float
        Top-1 test accuracy (%).
    elapsed : float
        Wall-clock time for ingest + evaluation (seconds).
    """
    embed_dim = train_embeds.size(1)
    n_train = train_embeds.size(0)

    # When an adapter is used, FAM's internal key dimension is output_dim;
    # otherwise it equals embed_dim.
    fam_input_dim = adapter.output_dim if adapter is not None else embed_dim

    mem = FastAssociativeMemory(
        input_dim=fam_input_dim,
        value_dim=num_classes,
        core_entries=min(n_train, 50000),
        core_vigilance=0.85,
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=25,
        inference_temp=0.05,
        use_lfu=True,
    ).to(device)

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n_train, 256):
            x_batch = train_embeds[i: i + 256].to(device)
            y_batch = train_labels[i: i + 256].to(device)
            if adapter is not None:
                x_batch = adapter(x_batch)
            mem.learn_local(x_batch, y_batch)

        correct = 0
        for i in range(0, len(test_embeds), 512):
            x_batch = test_embeds[i: i + 512].to(device)
            if adapter is not None:
                x_batch = adapter(x_batch)
            logits = mem(x_batch)
            correct += (
                logits.argmax(dim=1) == test_labels[i: i + 512].to(device)
            ).sum().item()

    elapsed = time.time() - t0
    acc = 100.0 * correct / len(test_labels)
    return acc, elapsed


# ---------------------------------------------------------------------------
# Kill-condition logic
# ---------------------------------------------------------------------------

def compute_degradation(baseline_acc: float, adapter_acc: float) -> float:
    """Return the *relative* accuracy drop introduced by the adapter (%).

    A positive value means the adapter lowered accuracy.  A negative value
    means the adapter improved accuracy over the baseline.

    Formula: ``(baseline_acc - adapter_acc) / baseline_acc * 100``

    Args:
        baseline_acc: Top-1 accuracy of the no-adapter (identity) condition.
        adapter_acc:  Top-1 accuracy of the trained-adapter condition.

    Returns:
        Relative degradation in percentage points (0–100).
    """
    if baseline_acc <= 0.0:
        return 0.0
    return (baseline_acc - adapter_acc) / baseline_acc * 100.0


def check_kill_condition(degradation: float, threshold: float = 5.0) -> bool:
    """Return ``True`` iff the G19 kill condition is met.

    The kill condition fires when *degradation* (relative accuracy drop)
    exceeds *threshold* percent.

    Args:
        degradation: Output of :func:`compute_degradation`.
        threshold:   Kill-condition threshold in percentage points (default 5.0).

    Returns:
        ``True`` when the adapter degrades accuracy beyond the threshold.
    """
    return degradation > threshold


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_results(
    dataset_label: str,
    baseline_acc: float,
    baseline_elapsed: float,
    adapter_acc: float | None,
    adapter_elapsed: float | None,
    degradation: float | None,
    threshold: float,
) -> str:
    """Return a formatted results summary string for G19.

    Args:
        dataset_label:    Short label for the dataset (e.g. ``"CIFAR-100"``).
        baseline_acc:     Top-1 accuracy of the identity/no-adapter condition.
        baseline_elapsed: Wall-clock time for the baseline (seconds).
        adapter_acc:      Top-1 accuracy of the trained adapter condition,
                          or ``None`` when no adapter was evaluated.
        adapter_elapsed:  Wall-clock time for the adapter condition (seconds),
                          or ``None``.
        degradation:      Relative accuracy degradation (%), or ``None``.
        threshold:        Kill-condition threshold (%).

    Returns:
        Multi-line formatted string.
    """
    sep = "=" * 72
    lines = [
        f"\n{sep}",
        f"  G19 — Adapter Generalization Stress Test  |  Dataset: {dataset_label}",
        sep,
        f"  {'Condition':<38} {'Top-1 Acc':>10} {'Time (s)':>10}",
        f"  {'-' * 60}",
        f"  {'Baseline (no adapter)':<38} {baseline_acc:>9.2f}% {baseline_elapsed:>9.1f}",
    ]

    if adapter_acc is not None and adapter_elapsed is not None:
        lines.append(
            f"  {'Trained adapter':<38} {adapter_acc:>9.2f}% {adapter_elapsed:>9.1f}"
        )

    lines.append(sep)

    if degradation is not None:
        lines.append(
            f"\n  Relative accuracy degradation: {degradation:.2f}%"
            f"  (kill threshold: {threshold:.1f}%)"
        )
        if check_kill_condition(degradation, threshold):
            lines.append(
                f"  \u26a0  KILL CONDITION MET (G19): adapter degrades accuracy by "
                f"{degradation:.2f}% > {threshold:.1f}%."
            )
            lines.append(
                "     The adapter is too specific; consider regularization or"
                " task-specific deployment."
            )
        else:
            lines.append(
                f"  \u2713  G19 PASSED: degradation {degradation:.2f}% \u2264"
                f" {threshold:.1f}% — adapter preserves general utility."
            )
    else:
        lines.append(
            "  (No trained adapter supplied — baseline only.)"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "G19 — Adapter Generalization Stress Test: "
            "verify trained Dogs adapter does not break generic tasks."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="./feature_cache_vitb14",
        metavar="DIR",
        help="Directory containing *train*.pt and *test*.pt feature cache files.",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        metavar="FILE",
        help="Path to trained adapter state-dict (.pt).  Omit for baseline only.",
    )
    parser.add_argument(
        "--adapter-input-dim",
        type=int,
        default=768,
        metavar="D",
        help="Input dimensionality of the adapter (must match backbone).",
    )
    parser.add_argument(
        "--adapter-output-dim",
        type=int,
        default=256,
        metavar="D",
        help="Output dimensionality of the adapter projection.",
    )
    parser.add_argument(
        "--adapter-hidden-dim",
        type=int,
        default=512,
        metavar="H",
        help="Hidden layer size (0 → linear adapter; > 0 → two-layer MLP).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        metavar="PCT",
        help="Kill-condition degradation threshold (%%relative).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Optional path to write a CSV results file.",
    )
    return parser.parse_args(argv)


def _write_csv(path: str, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["condition", "top1_accuracy_pct", "elapsed_s", "degradation_pct"]
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {path}")


def main(argv=None) -> None:  # pragma: no cover
    args = _parse_args(argv)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Cache dir   : {args.dataset}")
    print(f"Threshold   : {args.threshold}%")
    if args.adapter:
        print(f"Adapter     : {args.adapter}")
    print()

    print("Loading features ...")
    train_e, train_l, test_e, test_l = load_features(args.dataset, device)
    embed_dim = train_e.size(1)
    num_classes = int(train_l.max().item()) + 1
    print(f"  Train : {tuple(train_e.shape)}  Test : {tuple(test_e.shape)}")
    print(f"  embed_dim={embed_dim}  num_classes={num_classes}")
    print()

    csv_rows: list[dict] = []

    # ── Baseline: no adapter ──────────────────────────────────────────────
    print("[1/2] Baseline — FAM with no adapter (identity) ...")
    baseline_acc, baseline_elapsed = eval_fam(
        train_e, train_l, test_e, test_l, num_classes, device, adapter=None
    )
    print(f"  top-1={baseline_acc:.2f}%  time={baseline_elapsed:.1f}s")
    csv_rows.append({
        "condition": "baseline_no_adapter",
        "top1_accuracy_pct": f"{baseline_acc:.4f}",
        "elapsed_s": f"{baseline_elapsed:.2f}",
        "degradation_pct": "0.0000",
    })

    # ── Trained adapter (optional) ────────────────────────────────────────
    adapter_acc: float | None = None
    adapter_elapsed: float | None = None
    degradation: float | None = None

    if args.adapter:
        print(f"[2/2] Trained adapter — loading from {args.adapter} ...")
        adapter = load_adapter(
            args.adapter,
            input_dim=args.adapter_input_dim,
            output_dim=args.adapter_output_dim,
            hidden_dim=args.adapter_hidden_dim,
            device=device,
        )
        adapter_acc, adapter_elapsed = eval_fam(
            train_e, train_l, test_e, test_l, num_classes, device, adapter=adapter
        )
        degradation = compute_degradation(baseline_acc, adapter_acc)
        print(f"  top-1={adapter_acc:.2f}%  time={adapter_elapsed:.1f}s")
        print(f"  degradation={degradation:.2f}%")
        csv_rows.append({
            "condition": "trained_adapter",
            "top1_accuracy_pct": f"{adapter_acc:.4f}",
            "elapsed_s": f"{adapter_elapsed:.2f}",
            "degradation_pct": f"{degradation:.4f}",
        })
    else:
        print("[2/2] No adapter path supplied — skipping trained-adapter evaluation.")

    # ── Results ───────────────────────────────────────────────────────────
    # Infer a dataset label from the cache directory name.
    dataset_label = Path(args.dataset).name
    table = format_results(
        dataset_label,
        baseline_acc, baseline_elapsed,
        adapter_acc, adapter_elapsed,
        degradation, args.threshold,
    )
    print(table)

    if args.output:
        _write_csv(args.output, csv_rows)


if __name__ == "__main__":
    main()
