"""Extract DINOv2 ViT-L/14 features for CUB-200-2011 and save to disk.

Reads the official CUB-200-2011 dataset from a local directory using the
native split/label files (images.txt, train_test_split.txt,
image_class_labels.txt).  Outputs feature tensors in the format expected by
benchmarks/g22_adapter_birds.py.

Usage
-----
    python extract_cub200_vitl14.py [--root ./data/cub200]

Expected layout under --root
-----------------------------
    CUB_200_2011/
        images/
            001.Black_footed_Albatross/
                ...
        images.txt
        image_class_labels.txt
        train_test_split.txt

Download
--------
    wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
    tar -xzf CUB_200_2011.tgz -C <root>/

Output
------
    <root>/cub200_train.pt  — {"embeds": Tensor[N_train, 1024], "labels": Tensor[N_train]}
    <root>/cub200_test.pt   — {"embeds": Tensor[N_test,  1024], "labels": Tensor[N_test]}
"""

import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset


BATCH_SIZE = 128
NUM_WORKERS = 4


# ─────────────────────────────────────────────────── dataset ───

class CUB200Dataset(Dataset):
    """Minimal CUB-200-2011 reader using the native annotation files."""

    def __init__(self, root: str, train: bool, transform=None):
        base = Path(root) / "CUB_200_2011"

        # image index → relative path
        image_paths = {}
        with open(base / "images.txt") as f:
            for line in f:
                img_id, rel_path = line.strip().split()
                image_paths[int(img_id)] = rel_path

        # image index → label (0-indexed)
        labels = {}
        with open(base / "image_class_labels.txt") as f:
            for line in f:
                img_id, cls = line.strip().split()
                labels[int(img_id)] = int(cls) - 1  # 1-indexed → 0-indexed

        # image index → split flag (1 = train, 0 = test)
        split_flag = {}
        with open(base / "train_test_split.txt") as f:
            for line in f:
                img_id, is_train = line.strip().split()
                split_flag[int(img_id)] = int(is_train)

        self.samples = [
            (str(base / "images" / image_paths[i]), labels[i])
            for i in sorted(image_paths)
            if split_flag[i] == (1 if train else 0)
        ]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ─────────────────────────────────────────────── extraction ───

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


# ──────────────────────────────────────────────────── main ───

def main():
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 ViT-L/14 features for CUB-200-2011.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", default="./data/cub200",
                        help="Directory containing the CUB_200_2011/ subdirectory.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Root  : {args.root}")

    transform = build_transform()
    train_ds = CUB200Dataset(args.root, train=True,  transform=transform)
    test_ds  = CUB200Dataset(args.root, train=False, transform=transform)
    print(f"CUB-200-2011: {len(train_ds)} train / {len(test_ds)} test images")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    print("Loading DINOv2 ViT-L/14 …")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14",
                           verbose=False)
    model = model.to(device).eval()

    print("Extracting train features …")
    train_embeds, train_labels = extract(model, train_loader, device)
    print(f"  train: {tuple(train_embeds.shape)}")

    print("Extracting test features …")
    test_embeds, test_labels = extract(model, test_loader, device)
    print(f"  test : {tuple(test_embeds.shape)}")

    out_dir = Path(args.root)
    torch.save({"embeds": train_embeds.float(), "labels": train_labels.long()},
               out_dir / "cub200_train.pt")
    torch.save({"embeds": test_embeds.float(),  "labels": test_labels.long()},
               out_dir / "cub200_test.pt")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
