"""Extract DINOv2 ViT-L/14 features for ImageNet-R and save to disk."""

import os
import tarfile
import urllib.request

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

CACHE_DIR = "./feature_cache_inr_vitl14"
DATA_DIR = "./data"
TAR_URL = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
TAR_PATH = os.path.join(DATA_DIR, "imagenet-r.tar")
DATASET_ROOT = os.path.join(DATA_DIR, "imagenet-r")

BATCH_SIZE = 128
NUM_WORKERS = 8
SEED = 42
TRAIN_RATIO = 0.8


def download_and_extract():
    """Download ImageNet-R tarball and extract if not already present."""
    if os.path.isdir(DATASET_ROOT):
        print(f"Dataset already exists at {DATASET_ROOT}, skipping download.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.isfile(TAR_PATH):
        print(f"Downloading ImageNet-R from {TAR_URL} ...")
        urllib.request.urlretrieve(TAR_URL, TAR_PATH)
        print("Download complete.")

    print("Extracting tarball ...")
    with tarfile.open(TAR_PATH, "r") as tar:
        tar.extractall(path=DATA_DIR)
    print(f"Extracted to {DATASET_ROOT}")


def stratified_split(dataset, train_ratio, seed):
    """Deterministic stratified 80/20 split using fixed seed."""
    targets = [dataset.targets[i] for i in range(len(dataset))]
    classes = sorted(set(targets))

    gen = torch.Generator().manual_seed(seed)
    train_indices, test_indices = [], []

    for cls in classes:
        cls_indices = [i for i, t in enumerate(targets) if t == cls]
        n_train = int(len(cls_indices) * train_ratio)
        perm = torch.randperm(len(cls_indices), generator=gen).tolist()
        train_indices.extend(cls_indices[p] for p in perm[:n_train])
        test_indices.extend(cls_indices[p] for p in perm[n_train:])

    return Subset(dataset, train_indices), Subset(dataset, test_indices)


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
    download_and_extract()

    device = torch.device("cuda")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    model = model.to(device).eval()

    transform = build_transform()
    full_ds = ImageFolder(root=DATASET_ROOT, transform=transform)
    print(f"ImageNet-R: {len(full_ds)} images, {len(full_ds.classes)} classes")

    train_ds, test_ds = stratified_split(full_ds, TRAIN_RATIO, SEED)
    print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}")

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
               os.path.join(CACHE_DIR, "imagenetr_dinov2_train.pt"))
    torch.save({"embeds": test_embeds.float(), "labels": test_labels.long()},
               os.path.join(CACHE_DIR, "imagenetr_dinov2_test.pt"))
    print(f"Saved to {CACHE_DIR}/")


if __name__ == "__main__":
    main()
