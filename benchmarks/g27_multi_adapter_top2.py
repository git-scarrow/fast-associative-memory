#!/usr/bin/env python3
"""
benchmarks/g27_multi_adapter_top2.py — Gauntlet 27: Multi-Domain Adapter: Top-2 Combination.

Trains a MetricAdapter on the combined training data of the top-2 single-domain
winners from G22–G25, then evaluates it on all domains (in-domain and cross-domain).

Kill Condition (G27)
--------------------
If the multi-domain adapter's average accuracy across all evaluation datasets
is lower than the best single-domain adapter's average, the kill condition is met.

Usage
-----
    python benchmarks/g27_multi_adapter_top2.py \\
        --domains birds cars \\
        --cache-birds ./data/cub200 \\
        --cache-cars ./data/stanford_cars \\
        --cache-aircraft ./data/fgvc_aircraft \\
        --cache-flowers ./data/flowers102 \\
        --cifar-cache ./feature_cache_vitb14 \\
        --imagenet-r-cache ./feature_cache_imagenet_r_vitl14 \\
        --adapter-birds adapter_cub200_birds.pt \\
        --adapter-cars adapter_stanford_cars.pt

CLI args
--------
  --domains          Exactly 2 domain names from {birds, cars, aircraft, flowers}.
  --cache-{domain}   Path to the feature cache directory for each of the 4 domains.
  --cifar-cache      Feature cache directory for CIFAR-100 cross-domain evaluation.
  --imagenet-r-cache Feature cache directory for ImageNet-R cross-domain evaluation.
  --adapter-{domain} Paths to single-domain adapter weights for comparison.
  --epochs           Training epochs (default: 5).
  --batch-size       Mini-batch size (default: 128).
  --lr               Adam learning rate (default: 1e-4).
  --margin           Triplet margin (default: 0.2).
  --vigilance        FAM vigilance for in-domain evaluation (default: 0.80).
  --seed             Global random seed (default: 42).
  --device           Torch device string, e.g. ``cuda`` or ``cpu`` (default: auto).
  --output           Path to save multi-domain adapter weights (default: adapter_multi_top2.pt).
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapter import MetricAdapter  # noqa: E402
from fast_associative_memory import FastAssociativeMemory  # noqa: E402
from benchmarks.gauntlet_19_cross_domain import load_features  # noqa: E402

# All supported fine-grained domains
_ALL_DOMAINS = ("birds", "cars", "aircraft", "flowers")


# ─────────────────────────────────────────────────── training helpers ───

def _mine_hard_negatives(
    anchor_proj: torch.Tensor,
    batch_feats: torch.Tensor,
    batch_labels: torch.Tensor,
    anchor_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Online Hard Negative Mining (OHNM) over a mini-batch.

    Identical to the OHNM logic in ``train_adapter_triplet.py``.
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


def train_multi_adapter(
    train_embeds: torch.Tensor,
    train_labels: torch.Tensor,
    input_dim: int = 1024,
    output_dim: int = 256,
    hidden_dim: int = 512,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-4,
    margin: float = 0.2,
    device: torch.device | None = None,
    output: str = "adapter_multi_top2.pt",
) -> MetricAdapter:
    """Train a MetricAdapter on multi-domain combined training data.

    Labels from different domains must already be remapped so they do not
    collide (e.g. domain A: 0..N-1, domain B: N..N+M-1).

    Args:
        train_embeds: Combined (N, input_dim) embeddings from both domains.
        train_labels: Remapped class labels, shape (N,).
        input_dim:    Adapter input dimension.
        output_dim:   Adapter output dimension.
        hidden_dim:   Hidden layer size (> 0 → MLP).
        epochs:       Training epochs.
        batch_size:   Mini-batch size.
        lr:           Adam learning rate.
        margin:       Triplet margin.
        device:       Compute device.
        output:       Path to save trained weights.

    Returns:
        Trained :class:`~adapter.MetricAdapter`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adapter = MetricAdapter(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)
    adapter.to(device)

    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)
    feat_ds = TensorDataset(train_embeds, train_labels)
    feat_loader = DataLoader(feat_ds, batch_size=batch_size, shuffle=True)

    print(
        f"\n[G27] Training multi-domain adapter: {epochs} epoch(s),"
        f" batch={batch_size}, lr={lr}, margin={margin}"
    )
    for epoch in range(1, epochs + 1):
        adapter.train()
        total_loss = 0.0
        n_batches = 0

        for batch_feats, batch_labels in feat_loader:
            batch_feats = batch_feats.to(device)
            batch_labels = batch_labels.to(device)

            with torch.no_grad():
                proj = adapter(batch_feats)

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
    print(f"\n[G27] Multi-domain adapter saved → {output}")
    return adapter


# ─────────────────────────────────────────────────── evaluation helpers ───

@torch.no_grad()
def eval_fam_with_adapter(
    train_embeds: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeds: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
    adapter: MetricAdapter | None,
    core_vigilance: float = 0.80,
    core_entries: int = 50000,
) -> tuple[float, int, float]:
    """Evaluate FAM accuracy with optional adapter.

    Returns
    -------
    acc : float
        Top-1 accuracy in %.
    n_prototypes : int
        Number of prototypes written into FAM.
    condensation_ratio : float
        n_prototypes / n_train_samples.
    """
    num_classes = int(train_labels.max().item()) + 1
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

    x_train = train_embeds.to(device)
    y_train = train_labels.to(device)
    for i in range(0, len(x_train), 256):
        mem.learn_local(x_train[i:i + 256], y_train[i:i + 256])

    n_prototypes = int(mem.core_cam.occupied.sum().item())
    condensation_ratio = n_prototypes / len(train_labels)

    correct = 0
    x_test = test_embeds.to(device)
    y_test = test_labels.to(device)
    for i in range(0, len(x_test), 512):
        logits = mem(x_test[i:i + 512])
        correct += (logits.argmax(dim=1) == y_test[i:i + 512]).sum().item()

    acc = 100.0 * correct / len(test_labels)
    return acc, n_prototypes, condensation_ratio


def _load_single_domain_adapter(
    path: str,
    device: torch.device,
    input_dim: int = 1024,
    output_dim: int = 256,
    hidden_dim: int = 512,
) -> MetricAdapter | None:
    """Load a single-domain adapter from disk, or return None if path missing.

    Supports both legacy G27/G28 adapters (typically 1024→256 with hidden=512)
    and newer G30/G31 sweep-trained adapters with different projection sizes
    and layer counts by inferring the architecture from the checkpoint shapes.
    """
    p = Path(path)
    if not p.is_file():
        print(f"  WARNING: adapter not found at '{path}' — skipping.", file=sys.stderr)
        return None
    state = torch.load(str(p), map_location=device, weights_only=True)

    if "net.weight" in state:
        # Single linear projection
        w = state["net.weight"]
        inferred_input_dim = int(w.shape[1])
        inferred_output_dim = int(w.shape[0])
        inferred_hidden_dim = 0
        inferred_layers = 1
    else:
        # Sequential MLP: collect linear layers in order: net.<idx>.weight
        linear_keys = [
            k for k in state.keys()
            if k.startswith("net.") and k.endswith(".weight")
        ]
        linear_keys = sorted(linear_keys, key=lambda k: int(k.split(".")[1]))
        if not linear_keys:
            print(
                f"  WARNING: could not infer adapter architecture for '{path}' — skipping.",
                file=sys.stderr,
            )
            return None
        first_w = state[linear_keys[0]]
        last_w = state[linear_keys[-1]]
        inferred_input_dim = int(first_w.shape[1])
        inferred_hidden_dim = int(first_w.shape[0])
        inferred_output_dim = int(last_w.shape[0])
        inferred_layers = len(linear_keys)

    if inferred_input_dim != input_dim:
        print(
            f"  NOTE: adapter '{path}' input_dim inferred as {inferred_input_dim} "
            f"(eval input_dim={input_dim}).",
            file=sys.stderr,
        )

    adapter = MetricAdapter(
        input_dim=inferred_input_dim,
        output_dim=inferred_output_dim,
        hidden_dim=inferred_hidden_dim,
        nonlinearity="relu",
        proj_dim=inferred_output_dim,
        layers=inferred_layers,
        residual=(inferred_layers >= 2),
    )
    adapter.load_state_dict(state)
    adapter.eval()
    return adapter


# ────────────────────────────────────────────────────── output helpers ───

def _format_table(
    domain_a: str,
    domain_b: str,
    rows: list[tuple[str, float, float]],
    avg_best_single: float,
    avg_multi: float,
    kill: bool,
) -> str:
    """Return the formatted results table as a string."""
    sep = "=" * 73
    lines = [
        f"\nG27 \u2014 Multi-Domain Adapter: Top-2 Combination",
        sep,
        f"  {'Dataset':<20} {'Best Single':>12} {'Multi-Top2':>12} {'\u0394 (pp)':>8}",
        f"  {'-' * 55}",
    ]
    for label, best_single, multi in rows:
        delta = multi - best_single
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"  {label:<20} {best_single:>11.2f}% {multi:>11.2f}% {sign}{delta:>7.2f}"
        )
    lines.append(f"  {'-' * 55}")
    avg_delta = avg_multi - avg_best_single
    sign = "+" if avg_delta >= 0 else ""
    lines.append(
        f"  {'AVERAGE':<20} {avg_best_single:>11.2f}% {avg_multi:>11.2f}% {sign}{avg_delta:>7.2f}"
    )
    lines.append(sep)
    lines.append(
        f"\nKill condition: Multi-top2 avg {avg_multi:.2f}% vs best single avg {avg_best_single:.2f}%"
    )
    if kill:
        lines.append("\u2192 KILL")
    else:
        lines.append("\u2192 PASS")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────── CLI ───

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Gauntlet 27 — Multi-Domain Adapter: Top-2 Combination",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--domains", nargs=2, required=True,
        metavar="DOMAIN",
        help="Exactly 2 domain names from {birds, cars, aircraft, flowers}.",
    )
    # Per-domain cache paths
    _CACHE_DEFAULTS = {
        "birds":    "./data/cub200",
        "cars":     "./data/stanford_cars",
        "aircraft": "./data/fgvc_aircraft",
        "flowers":  "./data/flowers102/feature_cache",
    }
    for d in _ALL_DOMAINS:
        parser.add_argument(
            f"--cache-{d}", default=_CACHE_DEFAULTS[d], metavar="DIR",
            help=f"Feature cache directory for the '{d}' domain.",
        )
    # Cross-domain cache paths
    parser.add_argument(
        "--cifar-cache", default="./feature_cache_vitl14", metavar="DIR",
        help="Feature cache directory for CIFAR-100 cross-domain evaluation.",
    )
    parser.add_argument(
        "--imagenet-r-cache", default="./feature_cache_inr_vitl14", metavar="DIR",
        help="Feature cache directory for ImageNet-R cross-domain evaluation.",
    )
    # Per-domain single-domain adapter paths
    _ADAPTER_DEFAULTS = {
        "birds":    "adapter_cub200_birds.pt",
        "cars":     "./data/stanford_cars/adapter_stanford_cars.pt",
        "aircraft": "adapter_fgvc_aircraft.pt",
        "flowers":  "./data/flowers102/adapter_flowers102.pt",
    }
    for d in _ALL_DOMAINS:
        parser.add_argument(
            f"--adapter-{d}", default=_ADAPTER_DEFAULTS[d], metavar="FILE",
            help=f"Single-domain adapter weights for '{d}' (used for comparison).",
        )
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--vigilance", type=float, default=0.80,
                        help="FAM vigilance for in-domain evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="auto",
        help="Torch device string (e.g. 'cuda', 'cpu') or 'auto'.",
    )
    parser.add_argument(
        "--output", default="adapter_multi_top2.pt",
        help="Output path for the trained multi-domain adapter weights.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    torch.manual_seed(args.seed)

    # Validate domains
    domains = args.domains
    for d in domains:
        if d not in _ALL_DOMAINS:
            print(
                f"ERROR: unknown domain '{d}'. Must be one of {_ALL_DOMAINS}.",
                file=sys.stderr,
            )
            sys.exit(2)
    domain_a, domain_b = domains[0], domains[1]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"G27 \u2014 Multi-Domain Adapter: Top-2 Combination")
    print(f"Device   : {device}")
    print(f"Domains  : {domain_a}, {domain_b}")
    print(f"Seed     : {args.seed}")
    print()

    # ── Load in-domain training/test features ──────────────────────────────
    cache_map = {d: getattr(args, f"cache_{d}") for d in _ALL_DOMAINS}
    adapter_map = {d: getattr(args, f"adapter_{d}") for d in _ALL_DOMAINS}

    print(f"[1/4] Loading in-domain features ...")
    tr_a_e, tr_a_l, te_a_e, te_a_l = load_features(cache_map[domain_a], device)
    tr_b_e, tr_b_l, te_b_e, te_b_l = load_features(cache_map[domain_b], device)
    print(f"  {domain_a}: train={tuple(tr_a_e.shape)}, test={tuple(te_a_e.shape)}")
    print(f"  {domain_b}: train={tuple(tr_b_e.shape)}, test={tuple(te_b_e.shape)}")

    # ── Load cross-domain features ──────────────────────────────────────────
    print(f"[2/4] Loading cross-domain features ...")
    cifar_tr_e, cifar_tr_l, cifar_te_e, cifar_te_l = load_features(args.cifar_cache, device)
    inr_tr_e, inr_tr_l, inr_te_e, inr_te_l = load_features(args.imagenet_r_cache, device)
    print(f"  CIFAR-100   : train={tuple(cifar_tr_e.shape)}, test={tuple(cifar_te_e.shape)}")
    print(f"  ImageNet-R  : train={tuple(inr_tr_e.shape)}, test={tuple(inr_te_e.shape)}")

    # ── Combine training data with remapped labels ─────────────────────────
    # Domain A labels: 0..N_A-1; Domain B labels: N_A..N_A+N_B-1
    n_classes_a = int(tr_a_l.max().item()) + 1
    tr_b_l_remapped = tr_b_l + n_classes_a

    combined_embeds = torch.cat([tr_a_e, tr_b_e], dim=0).cpu()
    combined_labels = torch.cat([tr_a_l, tr_b_l_remapped], dim=0).cpu()

    # Shuffle combined training set
    perm = torch.randperm(len(combined_embeds), generator=torch.Generator().manual_seed(args.seed))
    combined_embeds = combined_embeds[perm]
    combined_labels = combined_labels[perm]

    input_dim = combined_embeds.shape[1]

    # ── Train multi-domain adapter ─────────────────────────────────────────
    print(f"[3/4] Training multi-domain adapter on {len(combined_embeds)} samples ...")
    multi_adapter = train_multi_adapter(
        train_embeds=combined_embeds,
        train_labels=combined_labels,
        input_dim=input_dim,
        output_dim=256,
        hidden_dim=512,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        margin=args.margin,
        device=device,
        output=args.output,
    )
    multi_adapter.eval()

    # ── Evaluate ──────────────────────────────────────────────────────────
    print(f"\n[4/4] Evaluating ...")

    # Load single-domain adapters for comparison
    sd_adapter_a = _load_single_domain_adapter(adapter_map[domain_a], device, input_dim)
    sd_adapter_b = _load_single_domain_adapter(adapter_map[domain_b], device, input_dim)

    # Cross-domain baseline (no adapter) — needed to compute single-domain adapter
    # accuracy on cross-domain sets for the "best single" column.
    # We evaluate each single-domain adapter on all datasets.

    results: dict[str, dict[str, float]] = {}  # dataset → {multi, sd_a, sd_b}

    def _eval_all_adapters(label, tr_e, tr_l, te_e, te_l, vigilance=args.vigilance):
        """Evaluate all three adapters (multi, sd_a, sd_b) on one dataset."""
        acc_multi, _, _ = eval_fam_with_adapter(
            tr_e, tr_l, te_e, te_l, device, multi_adapter, core_vigilance=vigilance
        )
        acc_sda = None
        if sd_adapter_a is not None:
            acc_sda, _, _ = eval_fam_with_adapter(
                tr_e, tr_l, te_e, te_l, device, sd_adapter_a, core_vigilance=vigilance
            )
        acc_sdb = None
        if sd_adapter_b is not None:
            acc_sdb, _, _ = eval_fam_with_adapter(
                tr_e, tr_l, te_e, te_l, device, sd_adapter_b, core_vigilance=vigilance
            )
        results[label] = {"multi": acc_multi, "sd_a": acc_sda, "sd_b": acc_sdb}
        print(
            f"  {label}: multi={acc_multi:.2f}%"
            + (f"  sd_{domain_a}={acc_sda:.2f}%" if acc_sda is not None else "")
            + (f"  sd_{domain_b}={acc_sdb:.2f}%" if acc_sdb is not None else "")
        )

    # In-domain evaluations
    _eval_all_adapters(
        f"{domain_a.capitalize()} (in)",
        tr_a_e, tr_a_l, te_a_e, te_a_l,
        vigilance=args.vigilance,
    )
    _eval_all_adapters(
        f"{domain_b.capitalize()} (in)",
        tr_b_e, tr_b_l, te_b_e, te_b_l,
        vigilance=args.vigilance,
    )

    # Cross-domain evaluations
    cross_vigilance = 0.85  # matches G19 default
    _eval_all_adapters(
        "CIFAR-100 (cross)",
        cifar_tr_e, cifar_tr_l, cifar_te_e, cifar_te_l,
        vigilance=cross_vigilance,
    )
    _eval_all_adapters(
        "ImageNet-R (cross)",
        inr_tr_e, inr_tr_l, inr_te_e, inr_te_l,
        vigilance=cross_vigilance,
    )

    # ── Compute averages and kill condition ────────────────────────────────
    dataset_labels = list(results.keys())

    avg_multi = sum(results[k]["multi"] for k in dataset_labels) / len(dataset_labels)

    # Best single-domain adapter: average across all datasets
    def _safe_avg(key: str) -> float | None:
        vals = [results[k][key] for k in dataset_labels if results[k][key] is not None]
        return sum(vals) / len(vals) if vals else None

    avg_sd_a = _safe_avg("sd_a")
    avg_sd_b = _safe_avg("sd_b")

    if avg_sd_a is None and avg_sd_b is None:
        avg_best_single = float("nan")
    elif avg_sd_a is None:
        avg_best_single = avg_sd_b
    elif avg_sd_b is None:
        avg_best_single = avg_sd_a
    else:
        avg_best_single = max(avg_sd_a, avg_sd_b)

    # Build table rows: Best Single = max(sd_a, sd_b) per dataset
    table_rows: list[tuple[str, float, float]] = []
    for label in dataset_labels:
        sd_a_acc = results[label]["sd_a"]
        sd_b_acc = results[label]["sd_b"]
        if sd_a_acc is None and sd_b_acc is None:
            best_single_acc = float("nan")
        elif sd_a_acc is None:
            best_single_acc = sd_b_acc
        elif sd_b_acc is None:
            best_single_acc = sd_a_acc
        else:
            best_single_acc = max(sd_a_acc, sd_b_acc)
        table_rows.append((label, best_single_acc, results[label]["multi"]))

    # Kill condition: multi avg < best single avg
    kill = (not math.isnan(avg_best_single)) and (avg_multi < avg_best_single)

    table = _format_table(
        domain_a, domain_b, table_rows, avg_best_single, avg_multi, kill
    )
    print(table)

    sys.exit(1 if kill else 0)


if __name__ == "__main__":
    main()
