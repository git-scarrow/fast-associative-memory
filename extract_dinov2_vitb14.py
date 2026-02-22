"""Extract DINOv2 ViT-B/14 features for CIFAR-100 and save to disk."""

import os
import torch
import torchvision.transforms as T
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

CACHE_DIR = "./feature_cache_vitb14"
BATCH_SIZE = 256
NUM_WORKERS = 4


def build_transform():
    return T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225)),
    ])


@torch.no_grad()
def extract(model, loader, device):
    all_embeds, all_labels = [], []
    for images, labels in loader:
        embeds = model(images.to(device))
        all_embeds.append(embeds.cpu())
        all_labels.append(labels)
    return torch.cat(all_embeds), torch.cat(all_labels)


def main():
    device = torch.device("cuda")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model = model.to(device).eval()

    transform = build_transform()
    train_ds = CIFAR100(root="./data", train=True, download=True, transform=transform)
    test_ds = CIFAR100(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print("Extracting train features...")
    train_embeds, train_labels = extract(model, train_loader, device)
    print(f"  train embeds: {train_embeds.shape}")

    print("Extracting test features...")
    test_embeds, test_labels = extract(model, test_loader, device)
    print(f"  test embeds: {test_embeds.shape}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save({"embeds": train_embeds.float(), "labels": train_labels.long()},
               f"{CACHE_DIR}/cifar100_dinov2_train.pt")
    torch.save({"embeds": test_embeds.float(), "labels": test_labels.long()},
               f"{CACHE_DIR}/cifar100_dinov2_test.pt")
    print(f"Saved to {CACHE_DIR}/")


if __name__ == "__main__":
    main()
