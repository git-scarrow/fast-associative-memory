"""Extract DINOv2 ViT-L/14 features for Stanford Dogs and save to disk.

Downloads the official Stanford Dogs archives and uses the official MATLAB
split files (`train_list.mat`, `test_list.mat`) to create train/test subsets.

Usage
-----
    python extract_stanford_dogs_vitl14.py --root ./data/stanford_dogs

Output
------
    <root>/stanford_dogs_train.pt  — {"features": Tensor[N_train, 1024], "labels": Tensor[N_train]}
    <root>/stanford_dogs_test.pt   — {"features": Tensor[N_test,  1024], "labels": Tensor[N_test]}
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

try:
    from scipy.io import loadmat
except ImportError as exc:  # pragma: no cover
    raise SystemExit("The 'scipy' package is required: pip install scipy") from exc


BATCH_SIZE = 128
NUM_WORKERS = 4
BASE_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs"
IMAGES_TAR_URL = f"{BASE_URL}/images.tar"
LISTS_TAR_URL = f"{BASE_URL}/lists.tar"


def _build_transform() -> T.Compose:
    return T.Compose(
        [
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


@torch.no_grad()
def _extract(model, loader: DataLoader, device: torch.device):
    all_features, all_labels = [], []
    for images, labels in loader:
        features = model(images.to(device))
        all_features.append(features.cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_features), torch.cat(all_labels)


def _download_if_missing(url: str, dest: Path) -> None:
    if dest.exists():
        return
    print(f"Downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def _extract_tar_if_missing(tar_path: Path, expected_path: Path) -> None:
    if expected_path.exists():
        return
    print(f"Extracting {tar_path} -> {tar_path.parent}")
    with tarfile.open(tar_path, "r:*") as tf:
        # Python 3.14 changes tar extraction defaults; use the safer filter now
        # when supported, while remaining compatible with older versions.
        try:
            tf.extractall(path=tar_path.parent, filter="data")
        except TypeError:  # pragma: no cover
            tf.extractall(path=tar_path.parent)


def _mat_file_list_to_paths(mat_path: Path) -> list[str]:
    mat = loadmat(str(mat_path))
    if "file_list" not in mat:
        raise RuntimeError(f"'file_list' not found in {mat_path}")

    raw = mat["file_list"]
    paths: list[str] = []
    # train_list.mat/test_list.mat store file_list as an object array of strings.
    for item in raw.reshape(-1):
        s = str(item[0]) if hasattr(item, "__len__") and not isinstance(item, str) else str(item)
        s = s.strip()
        if not s:
            continue
        paths.append(s.replace("\\", "/"))
    return paths


class IndexedSubset(Dataset):
    """Subset that preserves a custom index ordering."""

    def __init__(self, base: Dataset, indices: list[int]):
        self.base = base
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base[self.indices[idx]]


def _build_stanford_dogs_splits(root: Path, transform: T.Compose):
    images_tar = root / "images.tar"
    lists_tar = root / "lists.tar"
    images_dir = root / "Images"
    lists_dir = root / "lists"

    _download_if_missing(IMAGES_TAR_URL, images_tar)
    _download_if_missing(LISTS_TAR_URL, lists_tar)
    _extract_tar_if_missing(images_tar, images_dir)
    _extract_tar_if_missing(lists_tar, lists_dir)

    if not images_dir.exists():
        raise RuntimeError(f"Expected extracted images directory at {images_dir}")
    # The official lists.tar commonly extracts .mat files directly into <root>
    # (no "lists/" folder), but some mirrors may preserve a "lists/" directory.
    if lists_dir.exists():
        lists_base = lists_dir
    elif (root / "train_list.mat").exists() and (root / "test_list.mat").exists():
        lists_base = root
    else:
        raise RuntimeError(
            f"Expected split files under {lists_dir} or directly under {root}"
        )

    full_ds = ImageFolder(root=str(images_dir), transform=transform)

    # ImageFolder stores absolute sample paths; map to Stanford Dogs relative paths
    # (e.g., 'n02085620-Chihuahua/n02085620_10074.jpg').
    rel_to_idx = {
        Path(path).relative_to(images_dir).as_posix(): i
        for i, (path, _label) in enumerate(full_ds.samples)
    }

    train_rel = _mat_file_list_to_paths(lists_base / "train_list.mat")
    test_rel = _mat_file_list_to_paths(lists_base / "test_list.mat")

    missing_train = [p for p in train_rel if p not in rel_to_idx]
    missing_test = [p for p in test_rel if p not in rel_to_idx]
    if missing_train or missing_test:
        missing = (missing_train + missing_test)[:10]
        raise RuntimeError(
            "Split files reference images missing from ImageFolder. "
            f"Examples: {missing}"
        )

    train_indices = [rel_to_idx[p] for p in train_rel]
    test_indices = [rel_to_idx[p] for p in test_rel]
    return IndexedSubset(full_ds, train_indices), IndexedSubset(full_ds, test_indices)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Extract DINOv2 ViT-L/14 features for Stanford Dogs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--root",
        default="./data/stanford_dogs",
        help="Output directory for cached .pt feature files.",
    )
    args = p.parse_args(argv)

    out_dir = Path(args.root)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Root  : {out_dir}")

    transform = _build_transform()
    train_ds, test_ds = _build_stanford_dogs_splits(out_dir, transform)
    print(f"Stanford Dogs: {len(train_ds)} train / {len(test_ds)} test images")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    print("Loading DINOv2 ViT-L/14 …")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", verbose=False)
    model = model.to(device).eval()

    print("Extracting train features …")
    train_features, train_labels = _extract(model, train_loader, device)
    print(f"  train: {tuple(train_features.shape)}")

    print("Extracting test features …")
    test_features, test_labels = _extract(model, test_loader, device)
    print(f"  test : {tuple(test_features.shape)}")

    torch.save(
        {"features": train_features.float(), "labels": train_labels.long()},
        out_dir / "stanford_dogs_train.pt",
    )
    torch.save(
        {"features": test_features.float(), "labels": test_labels.long()},
        out_dir / "stanford_dogs_test.pt",
    )
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":  # pragma: no cover
    main()
