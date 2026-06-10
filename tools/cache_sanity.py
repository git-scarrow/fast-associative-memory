"""cache_sanity.py — read-only feature-cache inventory + integrity manifest (A1, PR-1).

Inspects every feature cache the repo's benchmarks reference and writes a JSON
manifest of what exists on THIS host: path resolution (including the tracked
`feature_cache_vitl14` symlink), tensor shapes/dtypes, NaN/Inf counts, L2-norm
stats, label histograms, and the specific class/count requirements of the A1
(#87, CIFAR-100) and A2 (ImageNet-R breadth) collapse studies.

Strictly metadata inspection: nothing is written except the manifest, no cache
is modified, no retrieval code is imported. Loads use ``weights_only=True``.

Usage:
    python tools/cache_sanity.py --out results/issue_a1_cache_sanity/manifest_<host>.json
    python tools/cache_sanity.py --hash ...   # also SHA256 each cache file (slow on multi-GB)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

# The cache writers are not uniform: extract_stanford_dogs_vitl14.py saves the
# feature tensor under "features"; every other extractor uses "embeds". Detect
# rather than assume, and report which key was found.
FEATURE_KEYS = ("embeds", "features")

# Fallbacks mirrored from benchmarks/probe_contraction.py::_VISION_CACHE_CANDIDATES
# so this manifest reports the same resolution order VisionDriftStream uses.
VITL14_CIFAR_FALLBACKS = [
    "/mnt/storage/workspace-archive/campus-fabric/feature_cache_vitl14/cifar100_dinov2_train.pt",
    "/mnt/data/dev/projects/campus-fabric/feature_cache_vitl14/cifar100_dinov2_train.pt",
]

# Every cache file referenced by committed code, with the study-specific
# requirements where one exists. `required_classes` = (class ids, min samples
# per class) that the named study's stream construction would demand of THIS
# file; attractor classes only need >= 1 sample.
CACHE_SPECS = [
    {
        "id": "vitl14_cifar100_train",
        "tier": "a-track",
        "path": "feature_cache_vitl14/cifar100_dinov2_train.pt",
        "fallbacks": VITL14_CIFAR_FALLBACKS,
        "consumers": [
            "benchmarks/probe_contraction.py:VisionDriftStream (A1 #87)",
            "benchmarks/calibration_probe.py --vision",
            "benchmark_sleep.py:load_features",
            "benchmarks/mt15_adaptive_eviction.py",
        ],
        # A1/#87: --vision-classes 0,8,19,33 --vision-attractor-class 71,
        # samples_per_class 32 + held_out_per_class 64 = 96 per class.
        "required_classes": {"study": "A1 (#87)", "classes": [0, 8, 19, 33],
                             "min_per_class": 96, "attractor": 71},
    },
    {
        "id": "vitl14_cifar100_test",
        "tier": "a-track",
        "path": "feature_cache_vitl14/cifar100_dinov2_test.pt",
        "consumers": ["benchmark_sleep.py:load_features",
                      "benchmarks/mt15_adaptive_eviction.py"],
    },
    {
        "id": "inr_vitl14_train",
        "tier": "a-track",
        "path": "feature_cache_inr_vitl14/imagenetr_dinov2_train.pt",
        "consumers": [
            "benchmarks/calibration_probe.py --vision-cache (A2 INR breadth)",
            "benchmarks/benchmark_dynamic_vigilance.py:DATASET_SPECS[ImageNet-R]",
            "evaluate_baselines.py",
            "probe_g0g1_reembed.py",
        ],
        # A2 INR breadth: --vision-classes 166,63,77,156 --vision-attractor-class 134.
        "required_classes": {"study": "A2 (INR breadth)",
                             "classes": [166, 63, 77, 156],
                             "min_per_class": 96, "attractor": 134},
    },
    {
        "id": "inr_vitl14_test",
        "tier": "a-track",
        "path": "feature_cache_inr_vitl14/imagenetr_dinov2_test.pt",
        "consumers": ["benchmarks/benchmark_dynamic_vigilance.py",
                      "evaluate_baselines.py"],
    },
    {
        "id": "vitb14_cifar100_train",
        "tier": "secondary",
        "path": "feature_cache_vitb14/cifar100_dinov2_train.pt",
        "consumers": ["extract_dinov2_vitb14.py (writer)",
                      "benchmarks/benchmark_dynamic_vigilance.py (CIFAR spec on some hosts)"],
    },
    {
        "id": "vitb14_cifar100_test",
        "tier": "secondary",
        "path": "feature_cache_vitb14/cifar100_dinov2_test.pt",
        "consumers": ["benchmarks/benchmark_dynamic_vigilance.py"],
    },
    # Adapter-domain caches (benchmark_dynamic_vigilance.py DATASET_SPECS).
    # Not on the A-track critical path; inventoried so the manifest is the
    # single picture of what compute a given host can support.
    {"id": "dogs_train", "tier": "adapter",
     "path": "data/stanford_dogs/stanford_dogs_train.pt",
     "consumers": ["benchmarks/benchmark_dynamic_vigilance.py"]},
    {"id": "dogs_test", "tier": "adapter",
     "path": "data/stanford_dogs/stanford_dogs_test.pt",
     "consumers": ["benchmarks/benchmark_dynamic_vigilance.py"]},
    {"id": "cub200_train", "tier": "adapter",
     "path": "data/cub200/cub200_train.pt",
     "consumers": ["benchmarks/benchmark_dynamic_vigilance.py"]},
    {"id": "cub200_test", "tier": "adapter",
     "path": "data/cub200/cub200_test.pt",
     "consumers": ["benchmarks/benchmark_dynamic_vigilance.py"]},
    {"id": "cars_train", "tier": "adapter",
     "path": "data/stanford_cars/stanford_cars_train.pt",
     "consumers": ["benchmarks/benchmark_dynamic_vigilance.py"]},
    {"id": "cars_test", "tier": "adapter",
     "path": "data/stanford_cars/stanford_cars_test.pt",
     "consumers": ["benchmarks/benchmark_dynamic_vigilance.py"]},
    {"id": "aircraft_train", "tier": "adapter",
     "path": "data/fgvc_aircraft/fgvc_aircraft_train.pt",
     "consumers": ["benchmarks/benchmark_dynamic_vigilance.py"]},
    {"id": "aircraft_test", "tier": "adapter",
     "path": "data/fgvc_aircraft/fgvc_aircraft_test.pt",
     "consumers": ["benchmarks/benchmark_dynamic_vigilance.py"]},
    {"id": "flowers_train", "tier": "adapter",
     "path": "data/flowers102/feature_cache/flowers102_train.pt",
     "consumers": ["benchmarks/benchmark_dynamic_vigilance.py"]},
    {"id": "flowers_test", "tier": "adapter",
     "path": "data/flowers102/feature_cache/flowers102_test.pt",
     "consumers": ["benchmarks/benchmark_dynamic_vigilance.py"]},
]


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _symlink_report(rel_path: str) -> dict | None:
    """Report on the first symlinked component of ``rel_path`` (if any)."""
    p = Path(rel_path)
    base = Path(p.anchor) if p.is_absolute() else REPO_ROOT
    cur = base
    for part in p.parts if p.is_absolute() else p.parts:
        if part == p.anchor:
            continue
        cur = cur / part
        if cur.is_symlink():
            try:
                shown = str(cur.relative_to(REPO_ROOT))
            except ValueError:
                shown = str(cur)
            return {
                "symlink": shown,
                "target": str(cur.readlink()),
                "target_exists": cur.resolve().exists(),
            }
    return None


def inspect_cache(spec: dict, do_hash: bool = False) -> dict:
    """Inspect one cache spec; never raises (failures land in the record)."""
    rec: dict = {
        "id": spec["id"],
        "tier": spec["tier"],
        "path": spec["path"],
        "consumers": spec["consumers"],
    }
    link = _symlink_report(spec["path"])
    if link is not None:
        rec["symlink"] = link

    resolved = None
    candidates = [spec["path"]] + list(spec.get("fallbacks", []))
    tried = []
    for c in candidates:
        p = Path(c) if Path(c).is_absolute() else REPO_ROOT / c
        tried.append(str(p))
        if p.exists() and p.is_file():
            resolved = p
            break
    rec["tried"] = tried
    if resolved is None:
        if link is not None and not link["target_exists"]:
            rec["status"] = "broken-symlink"
        else:
            rec["status"] = "missing"
        return rec

    rec["resolved_path"] = str(resolved)
    st = resolved.stat()
    rec["size_bytes"] = st.st_size
    rec["mtime"] = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
    if do_hash:
        rec["sha256"] = _sha256(resolved)

    try:
        blob = torch.load(resolved, map_location="cpu", weights_only=True)
    except Exception as e:  # corrupt file, pickle refusal, ...
        rec["status"] = "load-error"
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec

    if not isinstance(blob, dict):
        rec["status"] = "format-error"
        rec["error"] = f"expected dict, got {type(blob).__name__}"
        return rec
    rec["keys"] = sorted(blob.keys())

    feat_key = next((k for k in FEATURE_KEYS if k in blob), None)
    if feat_key is None or "labels" not in blob:
        rec["status"] = "format-error"
        rec["error"] = (f"need one of {FEATURE_KEYS} plus 'labels'; "
                        f"found {rec['keys']}")
        return rec
    rec["feature_key"] = feat_key

    feats = blob[feat_key]
    labels = blob["labels"]
    norms = feats.float().norm(dim=-1)
    rec["features"] = {
        "shape": list(feats.shape),
        "dtype": str(feats.dtype),
        "nan_count": int(torch.isnan(feats).sum().item()),
        "inf_count": int(torch.isinf(feats).sum().item()),
        "norm_mean": float(norms.mean().item()),
        "norm_std": float(norms.std().item()),
        "norm_min": float(norms.min().item()),
        "norm_max": float(norms.max().item()),
    }
    counts = torch.bincount(labels.long())
    nonzero = counts[counts > 0]
    rec["labels"] = {
        "shape": list(labels.shape),
        "dtype": str(labels.dtype),
        "n_classes_present": int((counts > 0).sum().item()),
        "label_min": int(labels.min().item()),
        "label_max": int(labels.max().item()),
        "per_class_min": int(nonzero.min().item()),
        "per_class_median": float(nonzero.float().median().item()),
        "per_class_max": int(nonzero.max().item()),
    }

    checks = {
        "rows_match": feats.shape[0] == labels.shape[0],
        "finite": rec["features"]["nan_count"] == 0
                  and rec["features"]["inf_count"] == 0,
    }
    req = spec.get("required_classes")
    if req is not None:
        per_class = {int(c): int(counts[c].item()) if c < len(counts) else 0
                     for c in req["classes"]}
        attr = req["attractor"]
        attr_n = int(counts[attr].item()) if attr < len(counts) else 0
        checks["study"] = req["study"]
        checks["required_class_counts"] = per_class
        checks["attractor_count"] = {str(attr): attr_n}
        checks["study_requirements_met"] = (
            all(n >= req["min_per_class"] for n in per_class.values())
            and attr_n >= 1)
    rec["checks"] = checks
    rec["status"] = "ok" if checks["rows_match"] and checks["finite"] else "integrity-error"
    return rec


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def build_manifest(do_hash: bool = False, specs=None) -> dict:
    records = [inspect_cache(s, do_hash=do_hash) for s in (specs or CACHE_SPECS)]
    by_status: dict[str, int] = {}
    for r in records:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "repo_commit": _git_commit(),
        "hashed": do_hash,
        "summary": by_status,
        "caches": records,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=str, default=None,
                    help="manifest output path (default: stdout)")
    ap.add_argument("--hash", action="store_true",
                    help="SHA256 each cache file (slow on multi-GB caches)")
    ap.add_argument("--root", type=str, default=None,
                    help="repo root to resolve relative cache paths against "
                         "(default: the parent of this script's directory)")
    args = ap.parse_args()

    if args.root:
        global REPO_ROOT
        REPO_ROOT = Path(args.root).resolve()

    manifest = build_manifest(do_hash=args.hash)
    text = json.dumps(manifest, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print(f"wrote {out}")
    else:
        print(text)

    for r in manifest["caches"]:
        flag = "" if r["status"] == "ok" else "   <-- ATTENTION"
        print(f"  [{r['status']:>14}] {r['id']:<22} {r['path']}{flag}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
