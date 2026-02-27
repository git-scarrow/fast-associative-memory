#!/usr/bin/env python3
"""benchmark_dynamic_vigilance.py — G29 Dynamic Vigilance (Margin-Based).

Benchmarks static vs margin-based dynamic vigilance across multiple datasets
using DINOv2 ViT-L/14 features and domain-specific adapters (where available).

Datasets (feature caches)
-------------------------
- Dogs
- Birds
- Cars
- Aircraft
- Flowers
- CIFAR-100
- ImageNet-R

Conditions
----------
- Static  : v = 0.92
- Dynamic : alpha in {0.2, 0.3, 0.5}

For each run we report:
- Accuracy (top-1 %)
- Condensation ratio (n_train / n_prototypes)
- Prototype count
- Mean effective vigilance (aggregated over writes)
- Mean margin

Margin distributions per dataset are also logged so we can verify that
fine-grained domains such as Dogs yield small margins (high vig), while
coarser domains such as Aircraft yield large margins (low vig).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

# Allow importing from the repo root regardless of the working directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapter import MetricAdapter  # noqa: E402
from dynamic_vigilance import DynamicVigilance  # noqa: E402
from fast_associative_memory import FastAssociativeMemory  # noqa: E402
from nstp import NSTPController  # noqa: E402


@dataclass
class Condition:
    name: str
    core_vigilance: float
    dynamic_alpha: float | None  # None = static


DATASET_SPECS: Dict[str, Dict[str, str | None]] = {
    "Dogs": {
        "train": "data/stanford_dogs/stanford_dogs_train.pt",
        "test": "data/stanford_dogs/stanford_dogs_test.pt",
        "adapter": "adapter_trained.pt",
    },
    "Birds": {
        "train": "data/cub200/cub200_train.pt",
        "test": "data/cub200/cub200_test.pt",
        "adapter": "adapter_cub200_birds.pt",
    },
    "Cars": {
        "train": "data/stanford_cars/stanford_cars_train.pt",
        "test": "data/stanford_cars/stanford_cars_test.pt",
        "adapter": "data/stanford_cars/adapter_stanford_cars.pt",
    },
    "Aircraft": {
        "train": "data/fgvc_aircraft/fgvc_aircraft_train.pt",
        "test": "data/fgvc_aircraft/fgvc_aircraft_test.pt",
        "adapter": "adapter_fgvc_aircraft.pt",
    },
    "Flowers": {
        "train": "data/flowers102/feature_cache/flowers102_train.pt",
        "test": "data/flowers102/feature_cache/flowers102_test.pt",
        "adapter": "data/flowers102/adapter_flowers102.pt",
    },
    "CIFAR-100": {
        "train": "/mnt/data/dev/projects/campus-fabric/feature_cache_vitl14/cifar100_dinov2_train.pt",
        "test": "/mnt/data/dev/projects/campus-fabric/feature_cache_vitl14/cifar100_dinov2_test.pt",
        "adapter": None,
    },
    "ImageNet-R": {
        "train": "feature_cache_inr_vitl14/imagenetr_dinov2_train.pt",
        "test": "feature_cache_inr_vitl14/imagenetr_dinov2_test.pt",
        "adapter": None,
    },
}


CONDITIONS: List[Condition] = [
    Condition(name="static_v0.92", core_vigilance=0.92, dynamic_alpha=None),
    Condition(name="dynamic_alpha0.2", core_vigilance=0.92, dynamic_alpha=0.2),
    Condition(name="dynamic_alpha0.3", core_vigilance=0.92, dynamic_alpha=0.3),
    Condition(name="dynamic_alpha0.5", core_vigilance=0.92, dynamic_alpha=0.5),
]


def _load_feature_pt(path: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    obj = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict in {path}, got {type(obj)}")

    # Support both historical key conventions.
    if "embeds" in obj:
        x = obj["embeds"]
    elif "features" in obj:
        x = obj["features"]
    else:
        raise KeyError(f"{path} missing 'embeds'/'features' keys (has {list(obj.keys())})")

    if "labels" in obj:
        y = obj["labels"]
    elif "targets" in obj:
        y = obj["targets"]
    else:
        raise KeyError(f"{path} missing 'labels'/'targets' keys (has {list(obj.keys())})")

    return x.float(), y.long()


def _load_adapter(adapter_path: str, embed_dim: int, device: torch.device) -> MetricAdapter:
    """Load MetricAdapter weights, inferring output_dim/hidden_dim."""
    sd = torch.load(adapter_path, map_location=device, weights_only=True)

    # Linear adapter: net.weight (out, in)
    if "net.weight" in sd:
        out_dim, in_dim = sd["net.weight"].shape
        if in_dim != embed_dim:
            raise ValueError(f"Adapter input_dim mismatch: expected {embed_dim}, got {in_dim}")
        adapter = MetricAdapter(input_dim=in_dim, output_dim=out_dim, hidden_dim=0)
        adapter.load_state_dict(sd)
        return adapter

    # MLP adapter: net.0.weight (hidden, in) and net.2.weight (out, hidden)
    if "net.0.weight" in sd and "net.2.weight" in sd:
        hidden_dim, in_dim = sd["net.0.weight"].shape
        out_dim, hidden_dim2 = sd["net.2.weight"].shape
        if in_dim != embed_dim:
            raise ValueError(f"Adapter input_dim mismatch: expected {embed_dim}, got {in_dim}")
        if hidden_dim2 != hidden_dim:
            raise ValueError("Adapter hidden_dim mismatch in state dict")
        adapter = MetricAdapter(input_dim=in_dim, output_dim=out_dim, hidden_dim=hidden_dim)
        adapter.load_state_dict(sd)
        return adapter

    raise ValueError(f"Unrecognized MetricAdapter state dict keys in {adapter_path}")


@torch.no_grad()
def eval_condition(
    train_embeds: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeds: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
    num_classes: int,
    adapter_path: str | None,
    condition: Condition,
    core_entries: int = 50000,
    dataset_name: str | None = None,
    margin_out_dir: str | None = None,
    train_batch: int = 256,
    test_batch: int = 128,
) -> Tuple[float, int, float, float, float]:
    """Run FAM under one vigilance condition and return metrics.

    Returns
    -------
    acc : float
        Top-1 accuracy in %.
    n_prototypes : int
        Number of occupied prototypes in the core CAM.
    condensation_ratio : float
        n_train / n_prototypes.
    mean_v_eff : float
        Mean effective vigilance as reported by the core CAM.
    mean_margin : float
        Mean margin across all writes.
    """

    embed_dim = train_embeds.size(1)

    # Optional adapter
    adapter: MetricAdapter | None = None
    if adapter_path is not None:
        ap = Path(adapter_path)
        if ap.is_file():
            adapter = _load_adapter(str(ap), embed_dim=embed_dim, device=device)
            adapter = adapter.to(device).eval()
        else:
            print(f"  WARNING: adapter file not found at '{adapter_path}', using identity.")

    dynamic_v = None
    if condition.dynamic_alpha is not None:
        dynamic_v = DynamicVigilance(v_base=condition.core_vigilance, alpha=condition.dynamic_alpha)

    # Spec: NSTP on
    nstp = NSTPController()

    fam = FastAssociativeMemory(
        input_dim=embed_dim,
        value_dim=num_classes,
        core_entries=core_entries,
        core_vigilance=condition.core_vigilance,
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=25,
        inference_temp=0.05,
        whitening_dim=0,
        use_lfu=True,
        use_bfloat16=False,
        immutable_keys=False,
        adapter=adapter,
        nstp=nstp,
        dynamic_vigilance=dynamic_v,
    ).to(device)

    x_train = train_embeds.to(device)
    y_train = train_labels.to(device)
    for i in range(0, len(x_train), train_batch):
        fam.learn_local(x_train[i : i + train_batch], y_train[i : i + train_batch])

    # Optional: log full margin/vigilance distributions per dataset×condition.
    if margin_out_dir and dynamic_v is not None and dataset_name is not None:
        core = fam.core_cam
        occ = core.occupied.nonzero(as_tuple=True)[0]
        if len(occ) > 0:
            proj_train = fam._project(x_train).float()  # type: ignore[attr-defined]
            keys_occ = core._keys_norm[occ]
            proto_labels = core.values[occ].float().argmax(dim=-1)

            # Chunked to avoid allocating (N_train, N_occ).
            margins_chunks = []
            v_chunks = []
            chunk_q = 256
            for s in range(0, proj_train.size(0), chunk_q):
                e = min(proj_train.size(0), s + chunk_q)
                q = F.normalize(proj_train[s:e], dim=-1)
                sims = q @ keys_occ.T  # (C, N_occ)
                v_eff_c, margins_c = dynamic_v.compute(sims, proto_labels)
                margins_chunks.append(margins_c.detach().cpu())
                v_chunks.append(v_eff_c.detach().cpu())

            margins_all = torch.cat(margins_chunks, dim=0)
            v_eff_all = torch.cat(v_chunks, dim=0)

            out_dir = Path(margin_out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{dataset_name}_{condition.name}.pt"
            torch.save(
                {
                    "dataset": dataset_name,
                    "condition": condition.name,
                    "v_base": dynamic_v.v_base,
                    "alpha": dynamic_v.alpha,
                    "v_floor": dynamic_v.v_floor,
                    "v_ceiling": dynamic_v.v_ceiling,
                    "margins": margins_all,
                    "v_effective": v_eff_all,
                },
                out_path,
            )

    # Accuracy
    # Help fragmented/busy GPUs by clearing cached blocks before eval.
    if device.type == "cuda":
        torch.cuda.empty_cache()

    correct = 0
    x_test = test_embeds.to(device)
    y_test = test_labels.to(device)
    for i in range(0, len(x_test), test_batch):
        logits = fam(x_test[i : i + test_batch])
        correct += (logits.argmax(dim=1) == y_test[i : i + test_batch]).sum().item()
    acc = 100.0 * correct / len(test_labels)

    # Core stats
    core = fam.core_cam
    stats = core.get_stats()
    n_prototypes = int(stats["n_occupied"])
    condensation_ratio = float(len(train_labels)) / max(1, n_prototypes)
    mean_v_eff = float(stats.get("mean_v_effective", condition.core_vigilance))
    mean_margin = float(stats.get("mean_margin", 0.0))

    return acc, n_prototypes, condensation_ratio, mean_v_eff, mean_margin


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="G29 — Dynamic Vigilance (Margin-Based)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--root",
        default=".",
        metavar="DIR",
        help="Repo root containing feature_cache_* directories.",
    )
    p.add_argument(
        "--core-entries",
        type=int,
        default=50000,
        help="Prototype capacity for the core CAM.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Override device string, e.g. 'cuda:0' or 'cpu'. Defaults to CUDA if available.",
    )
    p.add_argument(
        "--output-json",
        default=None,
        metavar="FILE",
        help="Optional path to write JSON results.",
    )
    p.add_argument(
        "--train-batch",
        type=int,
        default=256,
        help="Batch size for ingest (learn_local).",
    )
    p.add_argument(
        "--test-batch",
        type=int,
        default=128,
        help="Batch size for evaluation (smaller avoids GPU OOM).",
    )
    p.add_argument(
        "--margin-out-dir",
        default="outputs/margins_g29",
        metavar="DIR",
        help="When set, save per-dataset margin/vigilance distributions here.",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(args.root)
    print(f"Device   : {device}")
    print(f"Root     : {root}")
    print(f"Core cap : {args.core_entries}")
    print()

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for dataset_name, spec in DATASET_SPECS.items():
        train_path = spec["train"]
        test_path = spec["test"]
        adapter_path = spec.get("adapter")

        # Resolve relative paths against repo root; allow absolute paths.
        train_path = train_path if Path(train_path).is_absolute() else str(root / train_path)
        test_path = test_path if Path(test_path).is_absolute() else str(root / test_path)
        if adapter_path is not None:
            adapter_path = adapter_path if Path(adapter_path).is_absolute() else str(root / adapter_path)

        print(f"=== Dataset: {dataset_name} ===")
        print(f"  train   : {train_path}")
        print(f"  test    : {test_path}")
        if adapter_path is not None:
            print(f"  adapter : {adapter_path}")
        else:
            print("  adapter : (none)")

        # Annotate backbone variant from feature paths.
        paths_str = f"{train_path} {test_path}"
        if "vitb14" in paths_str:
            print("  backbone: DINOv2 ViT-B/14")
        else:
            print("  backbone: DINOv2 ViT-L/14")

        train_e, train_l = _load_feature_pt(train_path, device)
        test_e, test_l = _load_feature_pt(test_path, device)

        num_classes = int(train_l.max().item()) + 1
        print(f"  train   : {tuple(train_e.shape)}  test: {tuple(test_e.shape)}  num_classes={num_classes}")

        ds_results: Dict[str, Dict[str, float]] = {}

        for cond in CONDITIONS:
            print(f"  -> Condition: {cond.name}")
            acc, n_proto, cr, mean_v, mean_margin = eval_condition(
                train_e,
                train_l,
                test_e,
                test_l,
                device,
                num_classes,
                adapter_path,
                cond,
                core_entries=args.core_entries,
                dataset_name=dataset_name,
                margin_out_dir=args.margin_out_dir,
                train_batch=args.train_batch,
                test_batch=args.test_batch,
            )
            ds_results[cond.name] = {
                "accuracy_top1_pct": acc,
                "n_prototypes": n_proto,
                "condensation_ratio": cr,
                "mean_v_effective": mean_v,
                "mean_margin": mean_margin,
            }
            print(
                f"     acc={acc:6.2f}%  n_proto={n_proto:6d}  "
                f"cond_ratio={cr:7.2f}  v_eff_mean={mean_v:5.3f}  margin_mean={mean_margin:6.3f}"
            )

        all_results[dataset_name] = ds_results
        print()

    if args.output_json is not None:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"Results written to {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
