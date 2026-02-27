"""Extract DINOv2 ViT-L/14 features for Stanford Cars and save to disk.

Downloads Stanford Cars via the HuggingFace dataset `tanganke/stanford_cars`
(the de facto standard mirror after the original Kaggle source was taken down).
Outputs feature tensors in the format expected by benchmarks/g23_adapter_cars.py.

Usage
-----
    python extract_stanford_cars_vitl14.py [--root ./data/stanford_cars]

    Set HF_HOME to a path with sufficient disk space if your home partition
    is small (the dataset is ~500 MB):

        HF_HOME=/mnt/data/.cache/huggingface python extract_stanford_cars_vitl14.py

Output
------
    <root>/stanford_cars_train.pt  — {"embeds": Tensor[8144, 1024], "labels": Tensor[8144]}
    <root>/stanford_cars_test.pt   — {"embeds": Tensor[8041, 1024], "labels": Tensor[8041]}
"""

import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit(
        "The 'datasets' package is required: pip install datasets"
    ) from exc


BATCH_SIZE = 128
NUM_WORKERS = 4


# ─────────────────────────────────────────────────── dataset ───

class StanfordCarsHFDataset(Dataset):
    """Thin wrapper around tanganke/stanford_cars (HuggingFace)."""

    def __init__(self, hf_split: str, transform=None):
        self._ds = load_dataset("tanganke/stanford_cars", split=hf_split)
        self.transform = transform

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        sample = self._ds[idx]
        img = sample["image"].convert("RGB")
        label = sample["label"]
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
        description="Extract DINOv2 ViT-L/14 features for Stanford Cars via HuggingFace.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", default="./data/stanford_cars",
                        help="Output directory for cached .pt feature files.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Root  : {args.root}")
    print("Source: tanganke/stanford_cars (HuggingFace)")

    transform = build_transform()

    print("Loading Stanford Cars train split …")
    train_ds = StanfordCarsHFDataset("train", transform=transform)
    print("Loading Stanford Cars test split …")
    test_ds  = StanfordCarsHFDataset("test",  transform=transform)
    print(f"Stanford Cars: {len(train_ds)} train / {len(test_ds)} test images")

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
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"embeds": train_embeds.float(), "labels": train_labels.long()},
               out_dir / "stanford_cars_train.pt")
    torch.save({"embeds": test_embeds.float(),  "labels": test_labels.long()},
               out_dir / "stanford_cars_test.pt")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
