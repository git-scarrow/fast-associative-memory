#!/usr/bin/env python3
"""
benchmarks/g26_cross_domain_matrix.py — Gauntlet 26: Cross-Domain Regression Matrix.

Extends the G19 cross-domain regression protocol to all 4 domain-specific adapters
trained in G22–G25 (Birds, Cars, Aircraft, Flowers).

Protocol
--------
For each adapter × each dataset (4 × 2 = 8 evaluations):
1. Evaluate FAM **without** any adapter on the dataset → ``acc_baseline``.
2. Load the trained adapter; evaluate FAM **with** the adapter → ``acc_adapted``.
3. Compute accuracy drop: ``drop = acc_baseline − acc_adapted`` (percentage points).
4. Check kill condition: ``drop > threshold`` (default 5.0 pp).

Kill Condition (G26)
--------------------
Any single-domain adapter (Birds, Cars, Aircraft, Flowers) drops accuracy > 5pp on
CIFAR-100 or ImageNet-R relative to the no-adapter baseline.

Usage
-----
    python benchmarks/g26_cross_domain_matrix.py \\
        --cifar-cache ./feature_cache_vitb14 \\
        --imagenet-r-cache ./feature_cache_imagenet_r_vitl14 \\
        --adapter-birds adapter_cub200_birds.pt \\
        --adapter-cars adapter_stanford_cars.pt \\
        --adapter-aircraft adapter_fgvc_aircraft.pt \\
        --adapter-flowers adapter_flowers102.pt

CLI args
--------
  --cifar-cache        Directory of CIFAR-100 feature cache (default: ./feature_cache_vitb14)
  --imagenet-r-cache   Directory of ImageNet-R feature cache
                       (default: ./feature_cache_imagenet_r_vitl14)
  --adapter-birds      Path to CUB-200 Birds adapter state-dict (.pt file).
  --adapter-cars       Path to Stanford Cars adapter state-dict (.pt file).
  --adapter-aircraft   Path to FGVC Aircraft adapter state-dict (.pt file).
  --adapter-flowers    Path to Flowers-102 adapter state-dict (.pt file).
  --input-dim          Backbone embedding dimension (default: 1024).
  --output-dim         Adapter projection dimension (default: 256).
  --hidden-dim         Hidden layer size for MLP adapter (default: 512).
  --threshold          Kill-condition threshold in pp (default: 5.0).
  --seed               Global random seed (default: 42).
"""

import argparse
import sys
from pathlib import Path

import torch

# Allow importing from the repo root regardless of the working directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapter import MetricAdapter  # noqa: E402
from benchmarks.gauntlet_19_cross_domain import (  # noqa: E402
    accuracy_relative_drop,
    check_kill_condition,
    eval_fam_accuracy,
    load_features,
)

# ─────────────────────────────────────────────────── adapter definitions ───

_ADAPTERS = [
    ("Birds", "--adapter-birds"),
    ("Cars", "--adapter-cars"),
    ("Aircraft", "--adapter-aircraft"),
    ("Flowers", "--adapter-flowers"),
]

_DATASETS = [
    ("CIFAR-100", "cifar_cache"),
    ("ImageNet-R", "imagenet_r_cache"),
]


# ──────────────────────────────────────────────────────── output helpers ───

def _format_matrix(
    dataset_results: dict[str, dict],
    adapter_names: list[str],
    threshold: float,
) -> tuple[str, list[str]]:
    """Format the full regression matrix table.

    Parameters
    ----------
    dataset_results : dict
        Mapping of dataset_name → dict with keys 'baseline' (float acc) and
        adapter_name → float acc for each adapter.
    adapter_names : list[str]
        Names of adapters in display order (only those that were evaluated).
    threshold : float
        Kill-condition threshold in percentage points.

    Returns
    -------
    table : str
        Formatted table string.
    failures : list[str]
        List of failure description strings (empty if all pass).
    """
    sep = "=" * 73
    col_w = 10

    # Build header
    header_adapters = "".join(f"{name:<{col_w}}" for name in adapter_names)
    lines = [
        "",
        "G26 — Cross-Domain Regression Matrix",
        sep,
        f"  {'Dataset':<14}{'Baseline':<{col_w}}" + header_adapters,
        f"  {'-' * 71}",
    ]

    failures = []

    for ds_name, _ in _DATASETS:
        if ds_name not in dataset_results:
            continue
        ds = dataset_results[ds_name]
        baseline = ds["baseline"]

        # Accuracy row
        acc_row = f"  {ds_name:<14}{baseline:.2f}%   "
        for name in adapter_names:
            if name in ds:
                acc_row += f"{ds[name]:.2f}%   "
            else:
                acc_row += f"{'N/A':<{col_w}}"
        lines.append(acc_row)

        # Drop row
        drop_row = f"  {'  drop (pp)':<14}{'':<{col_w}}"
        for name in adapter_names:
            if name in ds:
                drop = accuracy_relative_drop(baseline, ds[name])
                drop_str = f"{drop:+.2f}"
                drop_row += f"{drop_str:<{col_w}}"
                if check_kill_condition(drop, threshold):
                    failures.append(
                        f"{ds_name} / {name}: drop={drop:+.2f} pp (>{threshold:.1f} pp)"
                    )
            else:
                drop_row += f"{'N/A':<{col_w}}"
        lines.append(drop_row)

    lines.append(sep)
    lines.append("")

    # Kill condition summary
    if failures:
        lines.append(f"Kill condition (> {threshold:.1f} pp drop):")
        for f in failures:
            lines.append(f"  ⚠  {f}")
        lines.append("→ KILL")
    else:
        lines.append(
            f"Kill condition (> {threshold:.1f} pp drop): "
            "NONE — all adapters pass"
        )
        lines.append("→ PASS")

    return "\n".join(lines), failures


# ─────────────────────────────────────────────────────────────── CLI ───

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Gauntlet 26 — Cross-Domain Regression Matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cifar-cache",
        default="./feature_cache_vitl14",
        metavar="DIR",
        help="Directory of CIFAR-100 feature cache (*train*.pt / *test*.pt).",
    )
    parser.add_argument(
        "--imagenet-r-cache",
        default="./feature_cache_inr_vitl14",
        metavar="DIR",
        help="Directory of ImageNet-R feature cache (*train*.pt / *test*.pt).",
    )
    parser.add_argument(
        "--adapter-birds",
        default="adapter_cub200_birds.pt",
        metavar="FILE",
        help="Path to CUB-200 Birds adapter state-dict.",
    )
    parser.add_argument(
        "--adapter-cars",
        default="./data/stanford_cars/adapter_stanford_cars.pt",
        metavar="FILE",
        help="Path to Stanford Cars adapter state-dict.",
    )
    parser.add_argument(
        "--adapter-aircraft",
        default="adapter_fgvc_aircraft.pt",
        metavar="FILE",
        help="Path to FGVC Aircraft adapter state-dict.",
    )
    parser.add_argument(
        "--adapter-flowers",
        default="./data/flowers102/adapter_flowers102.pt",
        metavar="FILE",
        help="Path to Flowers-102 adapter state-dict.",
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
    return parser.parse_args(argv)


def _load_adapter(
    path: str,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    device: torch.device,
) -> MetricAdapter | None:
    """Load a MetricAdapter from *path*, returning None if the file is missing."""
    p = Path(path)
    if not p.is_file():
        print(
            f"  WARNING: adapter file not found at '{path}' — skipping.",
            file=sys.stderr,
        )
        return None
    adapter = MetricAdapter(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
    )
    adapter.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    adapter.eval()
    print(f"  Loaded adapter from '{path}'")
    return adapter


def main(argv=None) -> None:
    args = _parse_args(argv)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}")
    print(f"Threshold : {args.threshold} pp")
    print(f"Seed      : {args.seed}")
    print()

    # Map adapter label → CLI path
    adapter_paths = {
        "Birds": args.adapter_birds,
        "Cars": args.adapter_cars,
        "Aircraft": args.adapter_aircraft,
        "Flowers": args.adapter_flowers,
    }

    # Map dataset label → cache directory
    dataset_caches = {
        "CIFAR-100": args.cifar_cache,
        "ImageNet-R": args.imagenet_r_cache,
    }

    # ── Load adapters ──────────────────────────────────────────────────────
    print("Loading adapters ...")
    adapters: dict[str, MetricAdapter] = {}
    for label, path in adapter_paths.items():
        a = _load_adapter(
            path,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            hidden_dim=args.hidden_dim,
            device=device,
        )
        if a is not None:
            adapters[label] = a
    print()

    # Only include adapter names that were successfully loaded
    adapter_names = [name for name in ["Birds", "Cars", "Aircraft", "Flowers"]
                     if name in adapters]

    # ── Evaluate each dataset ──────────────────────────────────────────────
    dataset_results: dict[str, dict] = {}

    for ds_name, cache_dir in dataset_caches.items():
        print(f"─── Dataset: {ds_name} (cache: {cache_dir}) ───")

        try:
            train_e, train_l, test_e, test_l = load_features(cache_dir, device)
        except FileNotFoundError as exc:
            print(f"  WARNING: {exc} — skipping {ds_name}.", file=sys.stderr)
            print()
            continue

        num_classes = int(train_l.max().item()) + 1
        print(f"  train={tuple(train_e.shape)}  test={tuple(test_e.shape)}"
              f"  num_classes={num_classes}")

        ds: dict[str, float] = {}

        # Baseline
        print("  [baseline] FAM — no adapter ...")
        acc_baseline, t_baseline = eval_fam_accuracy(
            train_e, train_l, test_e, test_l,
            device=device,
            num_classes=num_classes,
            adapter=None,
        )
        ds["baseline"] = acc_baseline
        print(f"    top-1={acc_baseline:.2f}%  time={t_baseline:.1f}s")

        # Each adapter
        for label in adapter_names:
            adapter = adapters[label]
            print(f"  [{label}] FAM + {label} adapter ...")
            acc, t = eval_fam_accuracy(
                train_e, train_l, test_e, test_l,
                device=device,
                num_classes=num_classes,
                adapter=adapter,
            )
            ds[label] = acc
            drop = accuracy_relative_drop(acc_baseline, acc)
            killed = check_kill_condition(drop, args.threshold)
            status = "⚠  KILL" if killed else "✓  pass"
            print(f"    top-1={acc:.2f}%  drop={drop:+.2f}pp  {status}"
                  f"  time={t:.1f}s")

        dataset_results[ds_name] = ds
        print()

    # ── Print matrix ───────────────────────────────────────────────────────
    table, failures = _format_matrix(dataset_results, adapter_names, args.threshold)
    print(table)

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
