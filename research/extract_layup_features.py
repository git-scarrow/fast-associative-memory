#!/usr/bin/env python3
"""
extract_layup_features.py — Extract multi-layer DINOv2 CLS tokens (LayUP).

Concatenates CLS tokens from ViT-L/14 blocks {8, 16, 24} (1-indexed)
to produce 3072-d features for CIFAR-100 train and test splits.
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

CACHE_DIR = "./feature_cache_vitl14"
LAYERS = [7, 15, 23]  # 0-indexed block indices for layers 8, 16, 24


@torch.no_grad()
def extract_layup(dataset, model, batch_size=128):
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    all_embeds, all_labels = [], []
    for imgs, labs in loader:
        imgs = imgs.to(device)
        out = model.get_intermediate_layers(imgs, n=LAYERS,
                                            return_class_token=True, norm=True)
        # Per-layer L2 normalization before concatenation to equalize
        # contributions and prevent narrow-cone cosine inflation
        cls_cat = torch.cat([F.normalize(cls, dim=-1) for _, cls in out], dim=-1)  # (B, 3072)
        all_embeds.append(cls_cat.cpu())
        all_labels.append(labs)
    return torch.cat(all_embeds), torch.cat(all_labels)


def main():
    device = torch.device("cuda")
    print("Loading DINOv2 ViT-L/14...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print("Extracting train features...")
    train_ds = datasets.CIFAR100(root="./data", train=True, download=True, transform=tfm)
    train_embeds, train_labels = extract_layup(train_ds, model)
    print(f"  Train: {train_embeds.shape}")

    print("Extracting test features...")
    test_ds = datasets.CIFAR100(root="./data", train=False, download=True, transform=tfm)
    test_embeds, test_labels = extract_layup(test_ds, model)
    print(f"  Test:  {test_embeds.shape}")

    train_path = f"{CACHE_DIR}/cifar100_dinov2_layup_train.pt"
    test_path = f"{CACHE_DIR}/cifar100_dinov2_layup_test.pt"
    torch.save({"embeds": train_embeds, "labels": train_labels}, train_path)
    torch.save({"embeds": test_embeds, "labels": test_labels}, test_path)
    print(f"Saved to {train_path} and {test_path}")


if __name__ == "__main__":
    main()
