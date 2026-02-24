"""
sweep_vigilance_floor.py — Extract-once vigilance floor sweep for G6.

1. Extract DINOv2 features once, cache to benchmarks/cache/g6_features.pt
2. Sweep vigilance levels twice:
   Pass A: full stack (adapter + NSTP)
   Pass B: baseline (no adapter, no NSTP)
3. Print combined side-by-side table.

Early-stops a pass if accuracy drops below 99.0%.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

from fast_associative_memory import FastAssociativeMemory
from adapter import MetricAdapter
from nstp import NSTPController

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "g6_features.pt")

VIGILANCE_LEVELS = [0.80, 0.75, 0.70, 0.65, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00]


class DINOv2Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitl14", verbose=False
        )
        self.model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def extract_features(root_dir, device):
    """Extract DINOv2 features for G6 subset. Cache to disk."""
    if os.path.exists(CACHE_FILE):
        print(f"[cache] Loading {CACHE_FILE}")
        data = torch.load(CACHE_FILE, weights_only=True)
        print(f"[cache] {data['features'].shape[0]} samples, "
              f"{data['features'].shape[1]}-d")
        return data["features"], data["labels"]

    print("[extract] Running DINOv2 on G6 subset...")
    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    ds = datasets.ImageFolder(root_dir, transform=tfm)
    start_cls, end_cls = 151, 200
    indices = [i for i, t in enumerate(ds.targets)
               if start_cls <= t <= end_cls]
    subset = Subset(ds, indices)
    loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=4)

    extractor = DINOv2Extractor().to(device)
    all_feats, all_lbls = [], []
    for imgs, lbls in loader:
        with torch.no_grad():
            all_feats.append(extractor(imgs.to(device)).cpu())
            all_lbls.append(lbls)

    features = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_lbls, dim=0)

    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save({"features": features, "labels": labels}, CACHE_FILE)
    print(f"[cache] Saved {CACHE_FILE} ({features.shape[0]} samples)")
    return features, labels


def run_g6(features, labels, device, vigilance,
           adapter=None, nstp=None):
    """Replay cached features through a fresh FAM. Returns (acc, ratio, protos)."""
    fam = FastAssociativeMemory(
        input_dim=1024, value_dim=1000,
        core_entries=10000, core_vigilance=vigilance,
        adapter=adapter, nstp=nstp,
    ).to(device)

    # Ingest with shuffled order
    perm = torch.randperm(features.size(0))
    for i in range(0, len(perm), 128):
        batch_idx = perm[i:i + 128]
        with torch.no_grad():
            fam.learn_local(features[batch_idx].to(device),
                            labels[batch_idx].to(device))

    # Evaluate
    fam.eval()
    correct = 0
    total = features.size(0)
    for i in range(0, total, 128):
        with torch.no_grad():
            preds = fam(features[i:i + 128].to(device)).argmax(1)
            correct += (preds == labels[i:i + 128].to(device)).sum().item()

    acc = 100.0 * correct / total
    protos = fam.core_cam.occupied.sum().item()
    ratio = protos / total
    return acc, ratio, protos


def sweep(features, labels, device, adapter=None, nstp=None, label=""):
    """Run vigilance sweep. Returns list of (vig, acc, ratio, protos) tuples."""
    results = []
    for vig in VIGILANCE_LEVELS:
        acc, ratio, protos = run_g6(features, labels, device, vig,
                                    adapter=adapter, nstp=nstp)
        results.append((vig, acc, ratio, protos))
        tag = "PASS" if acc > 90.0 and ratio < 0.5 else "----"
        print(f"  [{label}] vig={vig:.2f}  acc={acc:.2f}%  "
              f"ratio={ratio:.2f}  protos={protos}  {tag}")
        if acc < 99.0:
            print(f"  [{label}] FLOOR HIT: accuracy {acc:.2f}% < 99.0% "
                  f"at vigilance={vig}")
            break
    return results


def load_adapter(path, device):
    if not os.path.isfile(path):
        print(f"ERROR: adapter not found: {path}", file=sys.stderr)
        sys.exit(1)
    adapter = MetricAdapter(input_dim=1024, output_dim=256, hidden_dim=512)
    adapter.load_state_dict(torch.load(path, weights_only=True))
    adapter.eval()
    return adapter.to(device)


def main():
    parser = argparse.ArgumentParser(
        description="G6 vigilance floor sweep (extract-once)")
    parser.add_argument(
        "--root",
        default=os.environ.get("IMAGENET_ROOT", "data/imagenet-r"),
        help="Path to ImageNet directory (env: IMAGENET_ROOT)",
    )
    parser.add_argument(
        "--adapter-path", default="adapter_trained.pt",
        help="Path to trained adapter weights",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device (default: auto-detect)",
    )
    args = parser.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- Step 1: Extract once ---
    features, labels = extract_features(args.root, dev)
    n_samples = features.shape[0]

    # --- Step 2: Sweep ---
    print(f"\n{'='*70}")
    print(f"  Pass A: full stack (adapter + NSTP)")
    print(f"{'='*70}")
    adapter = load_adapter(args.adapter_path, dev)
    nstp = NSTPController(root_strategy="proximity")
    results_stack = sweep(features, labels, dev,
                          adapter=adapter, nstp=nstp, label="stack")

    print(f"\n{'='*70}")
    print(f"  Pass B: baseline (no adapter, no NSTP)")
    print(f"{'='*70}")
    results_base = sweep(features, labels, dev, label="baseline")

    # --- Step 3: Combined table ---
    # Pad shorter list with None rows
    max_len = max(len(results_stack), len(results_base))

    print(f"\n{'='*80}")
    print(f"  G6 VIGILANCE FLOOR SWEEP — {n_samples} samples")
    print(f"{'='*80}")
    print(f"  {'':>9}  {'--- adapter + NSTP ---':^30}  {'--- baseline ---':^30}")
    print(f"  {'Vigilance':>9}  {'Acc':>7} {'Ratio':>6} {'Protos':>7} {'':>5}  "
          f"{'Acc':>7} {'Ratio':>6} {'Protos':>7} {'':>5}")
    print(f"  {'-'*9}  {'-'*7} {'-'*6} {'-'*7} {'-'*5}  "
          f"{'-'*7} {'-'*6} {'-'*7} {'-'*5}")

    for i in range(max_len):
        vig = VIGILANCE_LEVELS[i]

        if i < len(results_stack):
            sv, sa, sr, sp = results_stack[i]
            s_status = "PASS" if sa > 90.0 and sr < 0.5 else "FAIL"
            s_str = f"{sa:>6.2f}% {sr:>6.2f} {sp:>7} {s_status:>5}"
        else:
            s_str = f"{'---':>7} {'---':>6} {'---':>7} {'':>5}"

        if i < len(results_base):
            bv, ba, br, bp = results_base[i]
            b_status = "PASS" if ba > 90.0 and br < 0.5 else "FAIL"
            b_str = f"{ba:>6.2f}% {br:>6.2f} {bp:>7} {b_status:>5}"
        else:
            b_str = f"{'---':>7} {'---':>6} {'---':>7} {'':>5}"

        print(f"  {vig:>9.2f}  {s_str}  {b_str}")

    print(f"{'='*80}")

    # Find floors
    stack_floor = None
    for vig, acc, _, _ in results_stack:
        if acc < 99.0:
            stack_floor = vig
            break
    base_floor = None
    for vig, acc, _, _ in results_base:
        if acc < 99.0:
            base_floor = vig
            break

    if stack_floor:
        print(f"  Stack floor: vigilance={stack_floor} (accuracy dropped below 99%)")
    else:
        print(f"  Stack floor: not reached (accuracy >= 99% across all levels)")
    if base_floor:
        print(f"  Baseline floor: vigilance={base_floor} (accuracy dropped below 99%)")
    else:
        print(f"  Baseline floor: not reached (accuracy >= 99% across all levels)")


if __name__ == "__main__":
    main()
