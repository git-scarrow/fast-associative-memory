"""
benchmark_open_world.py — Phase 3: G9 (OOD Detection) & G10 (Concept Drift)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from sklearn.metrics import roc_auc_score
import numpy as np

from fast_associative_memory import FastAssociativeMemory
from adapter import MetricAdapter
from nstp import NSTPController


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


def run_g9_ood(root_dir, device="cuda"):
    print("\n\U0001f47d  G9: OPEN WORLD OOD DETECTION  \U0001f47d")

    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ds = datasets.ImageFolder(root_dir, transform=tfm)

    # Fix v2: split dynamically at n_classes // 2 so the benchmark works
    # regardless of whether the folder has 200, 500, or 1000 classes.
    # Hardcoding 500 caused OOD Samples: 0 when the dataset had <500 classes.
    n_classes = len(ds.classes)
    split = n_classes // 2
    print(f"  Dataset: {n_classes} classes, splitting at class {split}")

    id_indices  = [i for i, t in enumerate(ds.targets) if t < split][:5000]
    ood_indices = [i for i, t in enumerate(ds.targets) if t >= split][:5000]

    print(f"  ID Samples:  {len(id_indices)} (Classes 0-{split-1})")
    print(f"  OOD Samples: {len(ood_indices)} (Classes {split}+)")

    if len(ood_indices) == 0:
        print("  ERROR: No OOD samples found. Check IMAGENET_ROOT structure.")
        return

    extractor = DINOv2Extractor().to(device)
    fam = FastAssociativeMemory(input_dim=1024, value_dim=n_classes,
                                core_entries=5000).to(device)

    train_loader = DataLoader(Subset(ds, id_indices), batch_size=128,
                              shuffle=True, num_workers=4)
    print("  Learning ID knowledge...")
    for imgs, lbls in train_loader:
        with torch.no_grad():
            fam.learn_local(extractor(imgs.to(device)), lbls.to(device))

    print("  Testing OOD rejection...")
    fam.eval()
    confs  = []
    labels = []  # 1=ID, 0=OOD

    # ID confidence
    for imgs, _ in train_loader:
        with torch.no_grad():
            feats = extractor(imgs.to(device))
            qs = fam.core_cam._cast(feats)
            # _get_nearest_batch returns (best_slots, best_sims, query_density)
            _, best_sims, _ = fam.core_cam._get_nearest_batch(qs)
            confs.extend(best_sims.cpu().numpy())
            labels.extend([1] * len(imgs))

    # OOD confidence
    ood_loader = DataLoader(Subset(ds, ood_indices), batch_size=128, num_workers=4)
    for imgs, _ in ood_loader:
        with torch.no_grad():
            feats = extractor(imgs.to(device))
            qs = fam.core_cam._cast(feats)
            _, best_sims, _ = fam.core_cam._get_nearest_batch(qs)
            confs.extend(best_sims.cpu().numpy())
            labels.extend([0] * len(imgs))

    auroc = roc_auc_score(labels, confs)
    print(f"\n--- G9 Results ---")
    print(f"Dataset: {n_classes} classes, ID=0-{split-1}, OOD={split}+")
    print(f"OOD Detection AUROC: {auroc:.4f}")

    if auroc > 0.85:
        print("\u2705 PASS: FAM knows what it doesn't know.")
    elif auroc > 0.70:
        print("\u26a0\ufe0f  WARN: Partial OOD rejection (0.70 < AUROC \u2264 0.85).")
    else:
        print("\u274c FAIL: FAM is hallucinating confidence on OOD data.")


def run_g10_drift(root_dir, device="cuda"):
    print("\n\U0001f32a\ufe0f  G10: CONCEPT DRIFT (Gaussian Blur)  \U0001f32a")

    tfm_clean = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ds_clean = datasets.ImageFolder(root_dir, transform=tfm_clean)
    indices = [i for i, t in enumerate(ds_clean.targets) if t < 10][:1000]
    subset  = Subset(ds_clean, indices)
    loader  = DataLoader(subset, batch_size=128, shuffle=True)

    extractor = DINOv2Extractor().to(device)
    fam = FastAssociativeMemory(input_dim=1024, value_dim=10, core_entries=2000).to(device)

    # Phase 1: Train Clean
    print("  Phase 1: Learning Clean Domain...")
    for imgs, lbls in loader:
        with torch.no_grad():
            fam.learn_local(extractor(imgs.to(device)), lbls.to(device))

    acc_clean_1 = evaluate(fam, extractor, loader, device)
    print(f"  Clean Accuracy (T1): {acc_clean_1:.2f}%")

    # Phase 2: Train Drift (heavy Gaussian blur applied in-loop)
    print("  Phase 2: Learning Drift Domain (Blur)...")
    blur = transforms.GaussianBlur(kernel_size=11, sigma=5.0)
    for imgs, lbls in loader:
        imgs = blur(imgs)
        with torch.no_grad():
            fam.learn_local(extractor(imgs.to(device)), lbls.to(device))

    # Backward Transfer: does Clean still work?
    acc_clean_2 = evaluate(fam, extractor, loader, device)
    print(f"  Clean Accuracy (T2 - Backward Transfer): {acc_clean_2:.2f}%")

    bwt = acc_clean_2 - acc_clean_1
    print(f"\n--- G10 Results ---")
    print(f"Backward Transfer: {bwt:+.2f}%")

    if bwt > -5.0:
        print("\u2705 PASS: FAM supports multi-modal concepts (Clean + Blur coexist).")
    else:
        print("\u274c FAIL: Catastrophic Forgetting/Interference.")


def _eval_embeds(fam, embeds, labels, batch_size=128):
    """Evaluate FAM accuracy on pre-extracted embeddings, return accuracy (%)."""
    fam.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(embeds), batch_size):
            preds = fam(embeds[i:i + batch_size]).argmax(1)
            correct += (preds == labels[i:i + batch_size]).sum().item()
    return 100.0 * correct / len(labels)


def run_g10_drift_full_stack(root_dir, device="cuda"):
    """G10 Concept Drift benchmark with Quantized Storage + NSTP + Adapter.

    Confirms that the full-stack combination does not break the non-destructive
    slot allocation property (Success criterion: BWT > -5%).
    """
    print("\n\U0001f32a\ufe0f  G10 FULL STACK: Concept Drift — Quantized + NSTP + Adapter  \U0001f32a")

    tfm_clean = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ds_clean = datasets.ImageFolder(root_dir, transform=tfm_clean)
    indices = [i for i, t in enumerate(ds_clean.targets) if t < 10][:1000]
    subset  = Subset(ds_clean, indices)
    loader  = DataLoader(subset, batch_size=128, shuffle=True)

    extractor = DINOv2Extractor().to(device)
    adapter = MetricAdapter(input_dim=1024, output_dim=512)
    nstp = NSTPController(sibling_threshold=0.85, depth_epsilon=0.10)
    fam = FastAssociativeMemory(
        input_dim=1024, value_dim=10, core_entries=2000,
        use_bfloat16=True, adapter=adapter, nstp=nstp,
    ).to(device)

    # Phase 1: Train Clean
    print("  Phase 1: Learning Clean Domain...")
    for imgs, lbls in loader:
        with torch.no_grad():
            fam.learn_local(extractor(imgs.to(device)), lbls.to(device))

    acc_clean_1 = evaluate(fam, extractor, loader, device)
    print(f"  Clean Accuracy (T1): {acc_clean_1:.2f}%")

    # Phase 2: Train Drift (heavy Gaussian blur applied in-loop)
    print("  Phase 2: Learning Drift Domain (Blur)...")
    blur = transforms.GaussianBlur(kernel_size=11, sigma=5.0)
    for imgs, lbls in loader:
        imgs = blur(imgs)
        with torch.no_grad():
            fam.learn_local(extractor(imgs.to(device)), lbls.to(device))

    # Backward Transfer: does Clean still work?
    acc_clean_2 = evaluate(fam, extractor, loader, device)
    print(f"  Clean Accuracy (T2 - Backward Transfer): {acc_clean_2:.2f}%")

    bwt = acc_clean_2 - acc_clean_1
    print(f"\n--- G10 Full Stack Results ---")
    print(f"Backward Transfer: {bwt:+.2f}%")

    if bwt > -5.0:
        print("\u2705 PASS: Full stack preserves multi-modal concepts (BWT > -5%).")
    else:
        print("\u274c FAIL: Catastrophic Forgetting/Interference with full stack.")

    return bwt


def evaluate(model, extractor, loader, device):
    correct = 0
    total   = 0
    model.eval()
    for imgs, lbls in loader:
        with torch.no_grad():
            preds = model(extractor(imgs.to(device))).argmax(1)
            correct += (preds == lbls.to(device)).sum().item()
            total   += imgs.size(0)
    return 100 * correct / total


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="G9 (OOD) & G10 (Drift) benchmarks.")
    parser.add_argument(
        "--root",
        default=os.environ.get("IMAGENET_ROOT", "data/imagenet-r"),
        help="Path to ImageNet directory (env: IMAGENET_ROOT)",
    )
    args = parser.parse_args()
    IMAGENET_ROOT = args.root
    if os.path.exists(IMAGENET_ROOT):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        run_g9_ood(IMAGENET_ROOT, dev)
        run_g10_drift(IMAGENET_ROOT, dev)
        run_g10_drift_full_stack(IMAGENET_ROOT, dev)
    else:
        print(f"Please set --root or IMAGENET_ROOT (currently: {IMAGENET_ROOT!r}).")
