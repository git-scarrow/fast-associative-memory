#!/usr/bin/env python3
"""
benchmarks/g28_multi_adapter_all4.py — Gauntlet 28: Multi-Domain Adapter All-4 Combination.

Trains a MetricAdapter on the combined training data of ALL 4 fine-grained
domains (CUB-200 Birds, Stanford Cars, FGVC Aircraft, Oxford Flowers-102) and
evaluates it on both in-domain and cross-domain datasets.

Kill Condition (G28)
--------------------
Average accuracy of the all-4 multi-domain adapter across all 6 evaluation
datasets (4 in-domain + 2 cross-domain) is less than the best individual
single-domain adapter average → kill condition met.

CLI Interface
-------------
    python benchmarks/g28_multi_adapter_all4.py \\
        --cache-birds ./data/cub200 \\
        --cache-cars ./data/stanford_cars \\
        --cache-aircraft ./data/fgvc_aircraft \\
        --cache-flowers ./data/flowers102 \\
        --cifar-cache ./feature_cache_vitb14 \\
        --imagenet-r-cache ./feature_cache_imagenet_r_vitl14 \\
        --adapter-birds adapter_cub200_birds.pt \\
        --adapter-cars adapter_stanford_cars.pt \\
        --adapter-aircraft adapter_fgvc_aircraft.pt \\
        --adapter-flowers adapter_flowers102.pt \\
        --adapter-top2 adapter_multi_top2.pt
"""

import argparse
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


# ─────────────────────────────────────────────── label remapping helpers ───

def remap_labels(
    domains_train: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate multi-domain training data with non-colliding label offsets.

    Parameters
    ----------
    domains_train:
        List of ``(embeds, labels)`` tuples — one per domain.

    Returns
    -------
    all_embeds : torch.Tensor
        Concatenated embeddings.
    all_labels : torch.Tensor
        Remapped labels with no inter-domain collisions.
    """
    all_embeds = []
    all_labels = []
    offset = 0
    for embeds, labels in domains_train:
        all_embeds.append(embeds)
        all_labels.append(labels + offset)
        offset += int(labels.max().item()) + 1
    return torch.cat(all_embeds, dim=0), torch.cat(all_labels, dim=0)


# ─────────────────────────────────────────────────────────── OHNM miner ───

def _mine_hard_negatives(
    anchor_proj: torch.Tensor,
    batch_feats: torch.Tensor,
    batch_labels: torch.Tensor,
    anchor_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Online Hard Negative Mining — same protocol as train_adapter_triplet.py."""
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


# ─────────────────────────────────────────────────────────── training ───

def train_multi_adapter(
    all_embeds: torch.Tensor,
    all_labels: torch.Tensor,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-4,
    margin: float = 0.2,
    output: str = "adapter_multi_all4.pt",
    device: torch.device | None = None,
) -> MetricAdapter:
    """Train MetricAdapter on combined multi-domain features via OHNM triplet loss.

    Parameters
    ----------
    all_embeds:
        Concatenated training embeddings from all domains, shape ``(N, D)``.
    all_labels:
        Remapped labels with no inter-domain collisions, shape ``(N,)``.
    epochs, batch_size, lr, margin, output:
        Training hyper-parameters; mirror ``train_adapter_triplet.py``.
    device:
        Compute device.  Auto-detected when ``None``.

    Returns
    -------
    MetricAdapter
        Trained adapter instance.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = all_embeds.size(1)
    adapter = MetricAdapter(input_dim=input_dim, output_dim=256, hidden_dim=512)
    adapter.to(device)

    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)
    feat_ds = TensorDataset(all_embeds, all_labels)
    feat_loader = DataLoader(feat_ds, batch_size=batch_size, shuffle=True)

    print(
        f"\n[G28] Training all-4 multi-domain adapter: {epochs} epoch(s),"
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
    print(f"[G28] All-4 multi-domain adapter saved → {output}")
    return adapter


# ─────────────────────────────────────────────────────── in-domain eval ───

@torch.no_grad()
def eval_indomain(
    train_embeds: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeds: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
    adapter: MetricAdapter | None,
    vigilance_condensation: float = 0.80,
    core_entries: int = 50000,
    core_vigilance: float = 0.85,
) -> tuple[float, int, float]:
    """Evaluate in-domain accuracy, prototype count, and condensation ratio.

    Parameters
    ----------
    vigilance_condensation:
        Vigilance used for the condensation-ratio measurement (default 0.80).

    Returns
    -------
    acc : float
        Top-1 accuracy in %.
    n_prototypes : int
        Number of prototypes stored at *core_vigilance*.
    condensation_ratio : float
        Fraction of training samples condensed at *vigilance_condensation*.
    """
    num_classes = int(train_labels.max().item()) + 1
    embed_dim = train_embeds.size(1)

    # ── Top-1 accuracy at the requested core vigilance ──────────────────────
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

    correct = 0
    x_test = test_embeds.to(device)
    y_test = test_labels.to(device)
    for i in range(0, len(x_test), 512):
        logits = mem(x_test[i:i + 512])
        correct += (logits.argmax(dim=1) == y_test[i:i + 512]).sum().item()
    acc = 100.0 * correct / len(test_labels)

    # ── Condensation ratio at vigilance_condensation ─────────────────────────
    mem_cond = FastAssociativeMemory(
        input_dim=embed_dim,
        value_dim=num_classes,
        core_entries=core_entries,
        core_vigilance=vigilance_condensation,
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=25,
        inference_temp=0.05,
        use_lfu=True,
        adapter=adapter,
    ).to(device)

    for i in range(0, len(x_train), 256):
        mem_cond.learn_local(x_train[i:i + 256], y_train[i:i + 256])

    n_cond = int(mem_cond.core_cam.occupied.sum().item())
    condensation_ratio = n_cond / max(len(train_labels), 1)

    return acc, n_prototypes, condensation_ratio


# ────────────────────────────────────────────────────── cross-domain eval ───

@torch.no_grad()
def eval_crossdomain(
    train_embeds: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeds: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
    adapter: MetricAdapter | None,
    core_entries: int = 50000,
    core_vigilance: float = 0.85,
) -> tuple[float, float]:
    """Evaluate cross-domain: return (accuracy, drop_vs_baseline).

    Parameters
    ----------
    adapter:
        Adapter to test.  ``None`` is used to compute the baseline.

    Returns
    -------
    acc_adapted : float
        Top-1 accuracy with the adapter in %.
    drop_pp : float
        Accuracy drop relative to the no-adapter baseline (pp).
    """
    num_classes = int(train_labels.max().item()) + 1
    embed_dim = train_embeds.size(1)

    def _run(adp):
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
            adapter=adp,
        ).to(device)

        x_tr = train_embeds.to(device)
        y_tr = train_labels.to(device)
        for i in range(0, len(x_tr), 256):
            mem.learn_local(x_tr[i:i + 256], y_tr[i:i + 256])

        correct = 0
        x_te = test_embeds.to(device)
        y_te = test_labels.to(device)
        for i in range(0, len(x_te), 512):
            logits = mem(x_te[i:i + 512])
            correct += (logits.argmax(dim=1) == y_te[i:i + 512]).sum().item()
        return 100.0 * correct / len(test_labels)

    acc_baseline = _run(None)
    acc_adapted = _run(adapter)
    drop_pp = acc_baseline - acc_adapted
    return acc_adapted, drop_pp


# ─────────────────────────────────── single-domain adapter loader helper ───

def _load_adapter(
    path: str,
    input_dim: int,
    device: torch.device,
    label: str,
) -> MetricAdapter | None:
    """Load a MetricAdapter from *path*; return ``None`` if file is missing.

    Supports legacy G27/G28 adapters (typically 1024→256 with hidden=512) and
    newer G30/G31 sweep-trained adapters by inferring architecture from
    checkpoint tensor shapes.
    """
    p = Path(path)
    if not p.is_file():
        print(f"  WARNING: adapter file not found: '{path}' — skipping {label}.",
              file=sys.stderr)
        return None
    state = torch.load(path, map_location=device, weights_only=True)

    if "net.weight" in state:
        # Single linear projection checkpoint.
        w = state["net.weight"]
        inferred_input_dim = int(w.shape[1])
        inferred_output_dim = int(w.shape[0])
        inferred_hidden_dim = 0
        inferred_layers = 1
    else:
        # Sequential MLP checkpoint: infer layer count and dims from Linear weights.
        linear_keys = [
            k for k in state.keys()
            if k.startswith("net.") and k.endswith(".weight")
        ]
        linear_keys = sorted(linear_keys, key=lambda k: int(k.split(".")[1]))
        if not linear_keys:
            print(
                f"  WARNING: could not infer adapter architecture for '{path}' "
                f"— skipping {label}.",
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


# ─────────────────────────────────────────────────────────── CLI ───

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="G28 — Multi-Domain Adapter: All-4 Combination",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Fine-grained domain caches
    parser.add_argument("--cache-birds", default="./data/cub200", metavar="DIR",
                        help="Feature cache directory for CUB-200 Birds.")
    parser.add_argument("--cache-cars", default="./data/stanford_cars", metavar="DIR",
                        help="Feature cache directory for Stanford Cars.")
    parser.add_argument("--cache-aircraft", default="./data/fgvc_aircraft", metavar="DIR",
                        help="Feature cache directory for FGVC Aircraft.")
    parser.add_argument("--cache-flowers", default="./data/flowers102/feature_cache", metavar="DIR",
                        help="Feature cache directory for Oxford Flowers-102.")
    # Cross-domain caches
    parser.add_argument("--cifar-cache", default="./feature_cache_vitl14", metavar="DIR",
                        help="Feature cache directory for CIFAR-100.")
    parser.add_argument("--imagenet-r-cache", default="./feature_cache_inr_vitl14", metavar="DIR",
                        help="Feature cache directory for ImageNet-R.")
    # Single-domain adapter paths (optional — warning printed if missing)
    parser.add_argument("--adapter-birds", default="adapter_cub200_birds.pt",
                        metavar="FILE", help="Single-domain Birds adapter weights.")
    parser.add_argument("--adapter-cars", default="./data/stanford_cars/adapter_stanford_cars.pt",
                        metavar="FILE", help="Single-domain Cars adapter weights.")
    parser.add_argument("--adapter-aircraft", default="adapter_fgvc_aircraft.pt",
                        metavar="FILE", help="Single-domain Aircraft adapter weights.")
    parser.add_argument("--adapter-flowers", default="./data/flowers102/adapter_flowers102.pt",
                        metavar="FILE", help="Single-domain Flowers adapter weights.")
    # Top-2 multi-domain adapter (informational, optional)
    parser.add_argument("--adapter-top2", default="adapter_multi_top2.pt", metavar="FILE",
                        help="G27 multi-top2 adapter weights (informational).")
    # Training hyper-parameters
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto",
                        help="Torch device string or 'auto'.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("G28 — Multi-Domain Adapter: All-4 Combination")
    print(f"Device : {device}  |  Seed : {args.seed}")
    print()

    # ── Load all feature caches ──────────────────────────────────────────────
    domain_cfg = [
        ("Birds",    args.cache_birds,    args.adapter_birds),
        ("Cars",     args.cache_cars,     args.adapter_cars),
        ("Aircraft", args.cache_aircraft, args.adapter_aircraft),
        ("Flowers",  args.cache_flowers,  args.adapter_flowers),
    ]

    domain_data: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]] = []
    for name, cache_dir, adp_path in domain_cfg:
        print(f"Loading {name} features from '{cache_dir}' …")
        tr_e, tr_l, te_e, te_l = load_features(cache_dir, device)
        print(f"  train={tuple(tr_e.shape)}  test={tuple(te_e.shape)}"
              f"  classes={int(tr_l.max().item()) + 1}")
        domain_data.append((name, tr_e, tr_l, te_e, te_l, adp_path))

    print("\nLoading cross-domain features …")
    cifar_tr_e, cifar_tr_l, cifar_te_e, cifar_te_l = load_features(
        args.cifar_cache, device
    )
    print(f"  CIFAR-100 train={tuple(cifar_tr_e.shape)}  test={tuple(cifar_te_e.shape)}")

    inr_tr_e, inr_tr_l, inr_te_e, inr_te_l = load_features(
        args.imagenet_r_cache, device
    )
    print(f"  ImageNet-R train={tuple(inr_tr_e.shape)}  test={tuple(inr_te_e.shape)}")

    # ── Train all-4 multi-domain adapter ────────────────────────────────────
    print("\nConcatenating training data with remapped labels …")
    all_embeds, all_labels = remap_labels([
        (tr_e.cpu(), tr_l.cpu()) for _, tr_e, tr_l, _, _, _ in domain_data
    ])
    print(f"  Combined: {tuple(all_embeds.shape)}"
          f"  total classes={int(all_labels.max().item()) + 1}")

    adapter_all4 = train_multi_adapter(
        all_embeds, all_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        margin=args.margin,
        output="adapter_multi_all4.pt",
        device=device,
    )
    adapter_all4.eval()

    # Infer adapter input dim from the combined features
    input_dim = all_embeds.size(1)

    # ── Load single-domain adapters (optional) ───────────────────────────────
    single_adapters: dict[str, MetricAdapter | None] = {}
    for name, _, _, _, _, adp_path in domain_data:
        single_adapters[name] = _load_adapter(adp_path, input_dim, device, name)

    # ── Load top-2 multi adapter (optional, informational) ───────────────────
    adapter_top2: MetricAdapter | None = None
    if args.adapter_top2 is not None:
        adapter_top2 = _load_adapter(args.adapter_top2, input_dim, device, "top2")

    # ── Evaluate on all datasets ─────────────────────────────────────────────
    # Rows: (dataset_name, best_single_acc, top2_acc, all4_acc)
    print("\nEvaluating …")

    # Helper: evaluate a set of adapters on an in-domain split
    def _eval_set_indomain(name, tr_e, tr_l, te_e, te_l):
        results = {}
        single_adp = single_adapters.get(name)

        # All-4 adapter
        t0 = time.time()
        acc4, n_proto, cond = eval_indomain(tr_e, tr_l, te_e, te_l, device, adapter_all4)
        results["all4"] = (acc4, n_proto, cond, time.time() - t0)

        # Single-domain adapter for this domain
        if single_adp is not None:
            acc_s, _, _ = eval_indomain(tr_e, tr_l, te_e, te_l, device, single_adp)
        else:
            acc_s = float("nan")
        results["single"] = acc_s

        # Top-2 informational
        if adapter_top2 is not None:
            acc_t2, _, _ = eval_indomain(tr_e, tr_l, te_e, te_l, device, adapter_top2)
        else:
            acc_t2 = float("nan")
        results["top2"] = acc_t2

        return results

    # Helper: evaluate a set of adapters on a cross-domain split
    def _eval_set_crossdomain(tr_e, tr_l, te_e, te_l):
        results = {}
        acc4, drop4 = eval_crossdomain(tr_e, tr_l, te_e, te_l, device, adapter_all4)
        results["all4"] = (acc4, drop4)

        # For cross-domain "best single" we use the single-domain Birds adapter
        # (first loaded; if none are available, we fall back to NaN)
        # The spec compares all4 to the *best* single-domain adapter — we pick
        # the one with highest accuracy on this cross-domain split.
        best_single_acc = float("nan")
        for name, single_adp in single_adapters.items():
            if single_adp is None:
                continue
            acc_s, _ = eval_crossdomain(tr_e, tr_l, te_e, te_l, device, single_adp)
            if best_single_acc != best_single_acc or acc_s > best_single_acc:
                best_single_acc = acc_s
        results["single"] = best_single_acc

        if adapter_top2 is not None:
            acc_t2, _ = eval_crossdomain(tr_e, tr_l, te_e, te_l, device, adapter_top2)
        else:
            acc_t2 = float("nan")
        results["top2"] = acc_t2

        return results

    # In-domain evaluations
    indomain_rows: list[tuple[str, float, float, float, int, float]] = []
    for name, tr_e, tr_l, te_e, te_l, _ in domain_data:
        print(f"  [{name}] in-domain …")
        res = _eval_set_indomain(name, tr_e, tr_l, te_e, te_l)
        acc4, n_proto, cond, _ = res["all4"]
        indomain_rows.append((name, res["single"], res["top2"], acc4, n_proto, cond))

    # Cross-domain evaluations
    print("  [CIFAR-100] cross-domain …")
    cifar_res = _eval_set_crossdomain(cifar_tr_e, cifar_tr_l, cifar_te_e, cifar_te_l)
    cifar_acc4, cifar_drop4 = cifar_res["all4"]

    print("  [ImageNet-R] cross-domain …")
    inr_res = _eval_set_crossdomain(inr_tr_e, inr_tr_l, inr_te_e, inr_te_l)
    inr_acc4, inr_drop4 = inr_res["all4"]

    # ── Kill-condition computation ────────────────────────────────────────────
    all4_accs = [r[3] for r in indomain_rows] + [cifar_acc4, inr_acc4]
    avg_all4 = sum(all4_accs) / len(all4_accs)

    # Best single-domain adapter average: each domain uses its own adapter for
    # in-domain; for cross-domain we pick the best cross-domain accuracy.
    # Collect per-dataset best-single accuracies.
    best_single_accs: list[float] = []
    for _, best_s, _, _, _, _ in indomain_rows:
        best_single_accs.append(best_s)
    best_single_accs.append(cifar_res["single"])
    best_single_accs.append(inr_res["single"])

    # Only include datasets where at least one single-domain adapter was present
    valid_pairs = [(a4, bs) for a4, bs in zip(all4_accs, best_single_accs)
                   if not (isinstance(bs, float) and bs != bs)]  # exclude NaN
    if valid_pairs:
        avg_best_single = sum(bs for _, bs in valid_pairs) / len(valid_pairs)
    else:
        avg_best_single = float("nan")

    # ── Print results table ───────────────────────────────────────────────────
    sep = "=" * 73
    header = f"  {'Dataset':<20} {'Best Single':>11} {'Multi-Top2':>11} {'Multi-All4':>11} {'Δ vs Best':>10}"
    inner_sep = f"  {'-' * 67}"

    def _fmt(val: float, ref: float | None = None) -> str:
        if val != val:  # NaN
            return f"{'N/A':>10}"
        s = f"{val:>9.2f}%"
        if ref is not None and ref == ref:
            delta = val - ref
            s += f"  {delta:+.2f}"
        return s

    print(f"\n{sep}")
    print(f"  G28 — Multi-Domain Adapter: All-4 Combination")
    print(sep)
    print(header)
    print(inner_sep)

    for name, best_s, top2_acc, acc4, n_proto, cond in indomain_rows:
        delta_vs_best = (acc4 - best_s) if best_s == best_s else float("nan")
        s_single = f"{best_s:>10.2f}%" if best_s == best_s else f"{'N/A':>10}"
        s_top2 = f"{top2_acc:>10.2f}%" if top2_acc == top2_acc else f"{'N/A':>10}"
        s_all4 = f"{acc4:>10.2f}%"
        s_delta = f"{delta_vs_best:>+9.2f}" if delta_vs_best == delta_vs_best else f"{'N/A':>9}"
        print(f"  {name + ' (in)':<20} {s_single} {s_top2} {s_all4} {s_delta}")
        print(f"  {'':20}  prototypes={n_proto}  condensation@0.80={cond:.4f}")

    # CIFAR-100 row
    cifar_best_s = cifar_res["single"]
    cifar_top2 = cifar_res["top2"]
    cifar_delta = (cifar_acc4 - cifar_best_s) if cifar_best_s == cifar_best_s else float("nan")
    s_single = f"{cifar_best_s:>10.2f}%" if cifar_best_s == cifar_best_s else f"{'N/A':>10}"
    s_top2 = f"{cifar_top2:>10.2f}%" if cifar_top2 == cifar_top2 else f"{'N/A':>10}"
    s_delta = f"{cifar_delta:>+9.2f}" if cifar_delta == cifar_delta else f"{'N/A':>9}"
    print(f"  {'CIFAR-100 (cross)':<20} {s_single} {s_top2} {cifar_acc4:>10.2f}% {s_delta}")
    print(f"  {'':20}  drop vs no-adapter={cifar_drop4:+.2f} pp")

    # ImageNet-R row
    inr_best_s = inr_res["single"]
    inr_top2 = inr_res["top2"]
    inr_delta = (inr_acc4 - inr_best_s) if inr_best_s == inr_best_s else float("nan")
    s_single = f"{inr_best_s:>10.2f}%" if inr_best_s == inr_best_s else f"{'N/A':>10}"
    s_top2 = f"{inr_top2:>10.2f}%" if inr_top2 == inr_top2 else f"{'N/A':>10}"
    s_delta = f"{inr_delta:>+9.2f}" if inr_delta == inr_delta else f"{'N/A':>9}"
    print(f"  {'ImageNet-R (cross)':<20} {s_single} {s_top2} {inr_acc4:>10.2f}% {s_delta}")
    print(f"  {'':20}  drop vs no-adapter={inr_drop4:+.2f} pp")

    print(inner_sep)

    s_avg_single = f"{avg_best_single:>10.2f}%" if avg_best_single == avg_best_single else f"{'N/A':>10}"
    avg_top2_accs = [r[2] for r in indomain_rows] + [cifar_res["top2"], inr_res["top2"]]
    valid_top2 = [v for v in avg_top2_accs if v == v]
    avg_top2 = sum(valid_top2) / len(valid_top2) if valid_top2 else float("nan")
    s_avg_top2 = f"{avg_top2:>10.2f}%" if avg_top2 == avg_top2 else f"{'N/A':>10}"
    avg_delta = (avg_all4 - avg_best_single) if avg_best_single == avg_best_single else float("nan")
    s_avg_delta = f"{avg_delta:>+9.2f}" if avg_delta == avg_delta else f"{'N/A':>9}"
    print(f"  {'AVERAGE':<20} {s_avg_single} {s_avg_top2} {avg_all4:>10.2f}% {s_avg_delta}")
    print(sep)

    # ── Kill-condition verdict ────────────────────────────────────────────────
    print(f"\nKill condition: Multi-all4 avg {avg_all4:.2f}%"
          f" vs best single avg {avg_best_single:.2f}%")

    kill = avg_best_single == avg_best_single and avg_all4 < avg_best_single
    if kill:
        print("→ KILL  (multi-all4 underperforms best single-domain adapter)")
        sys.exit(1)
    else:
        print("→ PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
