#!/usr/bin/env python3
"""
Standalone ARM encoder benchmark for Shutter-Deck.

Run this on the Pi before starting the ingestion pipeline. It times only the
DINOv2 forward pass on a synthetic tensor so you can choose the encoder tier
without any file I/O noise.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelSpec:
    name: str
    dim: int


MODEL_SPECS = [
    ModelSpec("dinov2_vitl14", 1024),
    ModelSpec("dinov2_vitb14", 768),
    ModelSpec("dinov2_vits14", 384),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark DINOv2 encoder latency on CPU.")
    p.add_argument("--runs", type=int, default=5, help="Timed runs per model (default: 5)")
    p.add_argument("--warmup", type=int, default=1, help="Warmup runs per model (default: 1)")
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads (default: 4)")
    p.add_argument(
        "--models",
        nargs="*",
        default=[m.name for m in MODEL_SPECS],
        help="Subset of model names to benchmark (default: vitl14 vitb14 vits14)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    selected = [m for m in MODEL_SPECS if m.name in set(args.models)]
    if not selected:
        raise SystemExit("No valid models selected.")

    print("Torch:", torch.__version__)
    print("Device: cpu")
    print("Threads:", torch.get_num_threads())

    dummy = torch.randn(1, 3, 224, 224)
    summaries: list[tuple[str, int, float, float, float, str]] = []

    for spec in selected:
        print(f"\nBenchmarking {spec.name} ({spec.dim}-d)")
        load_t0 = time.perf_counter()
        try:
            model = torch.hub.load("facebookresearch/dinov2", spec.name, pretrained=True)
            model = model.eval()
            load_t1 = time.perf_counter()
            print(f"  Load time: {load_t1 - load_t0:.2f}s")

            with torch.inference_mode():
                for i in range(args.warmup):
                    _ = model(dummy)
                    print(f"  Warmup {i + 1}/{args.warmup}: done")

            times: list[float] = []
            out_shape = "unknown"
            for i in range(args.runs):
                t0 = time.perf_counter()
                with torch.inference_mode():
                    out = model(dummy)
                dt = time.perf_counter() - t0
                out_shape = str(tuple(out.shape))
                times.append(dt)
                print(f"  Run {i + 1}/{args.runs}: {dt:.2f}s")

            median = statistics.median(times)
            avg = statistics.mean(times)
            p95 = max(times) if len(times) < 20 else statistics.quantiles(times, n=20)[18]
            print(f"  Median: {median:.2f}s | Mean: {avg:.2f}s | Max/P95: {p95:.2f}s | Output: {out_shape}")
            summaries.append((spec.name, spec.dim, median, avg, p95, out_shape))

            del model
        except Exception as e:
            print(f"  FAILED: {e}")
            summaries.append((spec.name, spec.dim, float("inf"), float("inf"), float("inf"), "FAILED"))

    print("\nSummary")
    print(f"{'Model':16} {'Dim':>5} {'Median(s)':>10} {'Mean(s)':>10} {'P95/Max(s)':>12} {'Output'}")
    print("-" * 72)
    for name, dim, median, avg, p95, out_shape in summaries:
        if median == float("inf"):
            print(f"{name:16} {dim:5d} {'FAILED':>10} {'FAILED':>10} {'FAILED':>12} {out_shape}")
        else:
            print(f"{name:16} {dim:5d} {median:10.2f} {avg:10.2f} {p95:12.2f} {out_shape}")


if __name__ == "__main__":
    main()
