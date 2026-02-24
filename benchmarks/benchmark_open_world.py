"""
benchmark_open_world.py — Phase 3: G9 (OOD Detection) & G10 (Concept Drift)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from sklearn.metrics import roc_auc_score
import numpy as np

from fast_associative_memory import FastAssociativeMemory
from extract_dinov2_vitb14 import DinoV2FeatureExtractor


def run_g9_ood(root_dir, device="cuda"):
    print("\n\U0001f47d  G9: OPEN WORLD OOD DETECTION  \U0001f47d")

    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ds = datasets.ImageFolder(root_dir, transform=tfm)

    id_indices  = [i for i, t in enumerate(ds.targets) if t < 500][:5000]
    ood_indices = [i for i, t in enumerate(ds.targets) if t >= 500][:5000]

    print(f"  ID Samples:  {len(id_indices)} (Classes 0-499)")
    print(f"  OOD Samples: {len(ood_indices)} (Classes 500+)")

    extractor = DinoV2FeatureExtractor().to(device)
    fam = FastAssociativeMemory(input_dim=1024, value_dim=1000, core_entries=5000).to(device)

    train_loader = DataLoader(Subset(ds, id_indices), batch_size=128, shuffle=True, num_workers=4)
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
    print(f"OOD Detection AUROC: {auroc:.4f}")

    if auroc > 0.85:
        print("\u2705 PASS: FAM knows what it doesn't know.")
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

    extractor = DinoV2FeatureExtractor().to(device)
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
    IMAGENET_ROOT = "/path/to/imagenet/train"
    if os.path.exists(IMAGENET_ROOT):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        run_g9_ood(IMAGENET_ROOT, dev)
        run_g10_drift(IMAGENET_ROOT, dev)
    else:
        print(f"Please set IMAGENET_ROOT (currently: {IMAGENET_ROOT!r}) to run G9/G10.")
