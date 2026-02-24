#!/usr/bin/env python3
"""
benchmarks/benchmark_full_stack.py — G18: Full Stack Latency Benchmark.

Verifies that the "Fast" in FAM holds up when all optional modules are active:

  * MetricAdapter         (use_adapter=True)
  * QuantizedCore         (use_quantization=True → bfloat16 ContinuousCAM)
  * NSTPController        (use_nstp=True, K=50)

Measures wall-clock time for the full forward pass *excluding* the DINOv2
backbone (constant cost, not part of the FAM pipeline).

Kill Condition (G18)
--------------------
Mean total forward() time (Adapter + QuantizedCore + NSTP) > 2.0 ms per query.

Benchmark stages
----------------
For each of N timing trials (single-query, B=1):

  Stage 1 — MetricAdapter:
    Linear projection (1024 → 512) + L2 normalisation.

  Stage 2 — QuantizedCore:
    ContinuousCAM forward pass with bfloat16 storage, K=50 retrieval.
    Includes broad cosine search, Mahalanobis re-ranking, CSLS correction,
    and softmax voting.

  Stage 3 — NSTP lateral inhibition:
    Query-to-candidate cosine similarity + NSTPController.prune_batch()
    on the K=50 retrieved candidates.

Usage
-----
  python benchmarks/benchmark_full_stack.py
  python benchmarks/benchmark_full_stack.py --k 50 --trials 500 --seed 42
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Allow importing from repo root regardless of working directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapter import MetricAdapter          # noqa: E402
from associative_core import ContinuousCAM  # noqa: E402
from nstp import NSTPController            # noqa: E402


# ─────────────────────────────────────────────────────────── constants ───

_KILL_THRESHOLD_MS: float = 2.0   # G18 kill condition
_DEFAULT_DIM_IN: int = 1024       # DINOv2 ViT-L/14 output dimensionality
_DEFAULT_DIM_PROJ: int = 512      # MetricAdapter output / CAM key dimension
_DEFAULT_VALUE_DIM: int = 100     # CAM value dimension (number of classes)
_DEFAULT_CAM_ENTRIES: int = 5000  # Number of stored CAM prototypes
_DEFAULT_K: int = 50              # Retrieval width for CAM inference and NSTP
_DEFAULT_TRIALS: int = 500        # Timing trials (after warmup)
_DEFAULT_WARMUP: int = 50         # Warmup iterations (excluded from statistics)
_DEFAULT_SEED: int = 42


# ──────────────────────────────────────────────────── module builders ───

def _build_stack(
    dim_in: int,
    dim_proj: int,
    value_dim: int,
    cam_entries: int,
    k: int,
    seed: int,
) -> tuple:
    """Construct the full module stack: Adapter + QuantizedCore + NSTP.

    The ContinuousCAM is pre-filled with *cam_entries* random unit-norm
    prototypes so that its forward pass has realistic work to perform.

    Returns
    -------
    adapter : MetricAdapter
    cam     : ContinuousCAM (bfloat16, inference_k=k, pre-filled)
    nstp    : NSTPController
    """
    torch.manual_seed(seed)

    # Stage 1 — MetricAdapter (linear projection + L2 normalise)
    adapter = MetricAdapter(input_dim=dim_in, output_dim=dim_proj).eval()

    # Stage 2 — QuantizedCore: bfloat16 ContinuousCAM pre-filled with random prototypes
    cam = ContinuousCAM(
        key_dim=dim_proj,
        value_dim=value_dim,
        max_entries=cam_entries,
        inference_k=k,
        use_bfloat16=True,
    )
    rand_keys = F.normalize(torch.randn(cam_entries, dim_proj), dim=-1).to(torch.bfloat16)
    cam.keys.copy_(rand_keys)
    class_indices = torch.arange(cam_entries) % value_dim
    cam.values.copy_(F.one_hot(class_indices, num_classes=value_dim).to(torch.bfloat16))
    cam.occupied.fill_(True)
    cam._update_key_norm(torch.arange(cam_entries))

    # Stage 3 — NSTPController
    nstp = NSTPController(sibling_threshold=0.85)

    return adapter, cam, nstp


# ────────────────────────────────────────────────── benchmark runner ───

def _run_benchmark(
    dim_in: int = _DEFAULT_DIM_IN,
    dim_proj: int = _DEFAULT_DIM_PROJ,
    value_dim: int = _DEFAULT_VALUE_DIM,
    cam_entries: int = _DEFAULT_CAM_ENTRIES,
    k: int = _DEFAULT_K,
    trials: int = _DEFAULT_TRIALS,
    warmup: int = _DEFAULT_WARMUP,
    seed: int = _DEFAULT_SEED,
) -> tuple[dict[str, list[float]], list[float]]:
    """Run the G18 full-stack latency benchmark (single-query, B=1).

    Each trial measures one end-to-end forward pass:
    Adapter → QuantizedCore → NSTP.

    Returns
    -------
    stage_timings : dict[str, list[float]]
        Per-trial latencies in ms for each stage.
    total_timings : list[float]
        Per-trial end-to-end latencies in ms.
    """
    adapter, cam, nstp = _build_stack(dim_in, dim_proj, value_dim, cam_entries, k, seed)

    # Pre-extract K candidate keys/values from the CAM for NSTP input.
    # These represent the top-K prototypes that cam.forward() would retrieve
    # internally; they are fixed here so all timing overhead falls inside
    # the NSTP stage (cosine-sim computation + prune_batch).
    candidate_keys = cam.keys[:k].float().unsqueeze(0)    # (1, K, dim_proj)
    candidate_values = cam.values[:k].float().unsqueeze(0)  # (1, K, value_dim)

    stage_timings: dict[str, list[float]] = {
        "adapter": [],
        "quantized_core": [],
        "nstp": [],
    }
    total_timings: list[float] = []

    for trial in range(-warmup, trials):
        x = F.normalize(torch.randn(1, dim_in), dim=-1)  # single DINOv2 query

        t_total_start = time.perf_counter()

        # Stage 1 — Adapter
        t0 = time.perf_counter()
        with torch.no_grad():
            x_adapted = adapter(x)                         # (1, dim_proj)
        t1 = time.perf_counter()

        # Stage 2 — QuantizedCore (bfloat16 ContinuousCAM)
        t2 = time.perf_counter()
        with torch.no_grad():
            _ = cam(x_adapted)                             # (1, value_dim)
        t3 = time.perf_counter()

        # Stage 3 — NSTP lateral inhibition
        # Cosine similarity of the query to the K candidates (part of NSTP cost)
        t4 = time.perf_counter()
        cand_sims = F.cosine_similarity(
            x_adapted.unsqueeze(1), candidate_keys, dim=-1
        )                                                  # (1, K)
        _masks, _sims_out = nstp.prune_batch(
            x_adapted, candidate_keys, candidate_values, cand_sims
        )
        t5 = time.perf_counter()

        t_total_end = time.perf_counter()

        if trial < 0:
            continue  # discard warmup trials

        ms = 1000.0  # seconds → milliseconds
        stage_timings["adapter"].append((t1 - t0) * ms)
        stage_timings["quantized_core"].append((t3 - t2) * ms)
        stage_timings["nstp"].append((t5 - t4) * ms)
        total_timings.append((t_total_end - t_total_start) * ms)

    return stage_timings, total_timings


# ──────────────────────────────────────────────────────── reporting ───

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    """Return the *pct*-th percentile of *values* (linear interpolation)."""
    if not values:
        return 0.0
    sv = sorted(values)
    n = len(sv)
    idx = (pct / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    return sv[lo] + (idx - lo) * (sv[hi] - sv[lo])


def _print_report(
    stage_timings: dict[str, list[float]],
    total_timings: list[float],
    k: int,
    trials: int,
) -> bool:
    """Print the results table and return True if the benchmark passes."""
    sep = "=" * 72
    labels = {
        "adapter":        "Stage 1 — MetricAdapter (linear + L2-norm)",
        "quantized_core": f"Stage 2 — QuantizedCore (bfloat16 CAM, K={k})",
        "nstp":           f"Stage 3 — NSTP lateral inhibition (K={k})",
    }

    print(f"\n{sep}")
    print(f"  G18 Full-Stack Latency Benchmark  |  K={k}  |  N={trials} trials")
    print(sep)
    print(f"  {'Stage':<46} {'Mean (ms)':>10} {'p99 (ms)':>10}")
    print(f"  {'-' * 68}")

    for key, label in labels.items():
        vals = stage_timings[key]
        mean = _mean(vals)
        p99 = _percentile(vals, 99)
        print(f"  {label:<46} {mean:>10.3f} {p99:>10.3f}")

    total_mean = _mean(total_timings)
    total_p99 = _percentile(total_timings, 99)

    print(f"  {'-' * 68}")
    print(
        f"  {'TOTAL (Adapter + QuantizedCore + NSTP)':<46} "
        f"{total_mean:>10.3f} {total_p99:>10.3f}"
    )
    print(sep)

    passed = total_mean < _KILL_THRESHOLD_MS
    if passed:
        print(
            f"\n  ✓  G18 PASS — mean total = {total_mean:.3f} ms  "
            f"< {_KILL_THRESHOLD_MS:.1f} ms kill threshold.\n"
            f"       The 'Fast' in FAM holds up with all optional modules active."
        )
    else:
        print(
            f"\n  ✗  G18 FAIL — mean total = {total_mean:.3f} ms  "
            f"≥ {_KILL_THRESHOLD_MS:.1f} ms kill threshold.\n"
            f"       Full-stack forward() exceeds the latency budget."
        )

    return passed


# ──────────────────────────────────────────────────────────── CLI ───

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "G18: Full-stack latency benchmark — "
            "Adapter + QuantizedCore + NSTP < 2 ms per query on CPU."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dim-in", type=int, default=_DEFAULT_DIM_IN, metavar="D",
        help="DINOv2 backbone output dimension.",
    )
    parser.add_argument(
        "--dim-proj", type=int, default=_DEFAULT_DIM_PROJ, metavar="D",
        help="MetricAdapter output / CAM key dimension.",
    )
    parser.add_argument(
        "--value-dim", type=int, default=_DEFAULT_VALUE_DIM, metavar="V",
        help="CAM value dimension (number of classes).",
    )
    parser.add_argument(
        "--cam-entries", type=int, default=_DEFAULT_CAM_ENTRIES, metavar="N",
        help="Number of prototypes stored in the CAM.",
    )
    parser.add_argument(
        "--k", type=int, default=_DEFAULT_K, metavar="K",
        help="Retrieval width for CAM inference and NSTP.",
    )
    parser.add_argument(
        "--trials", type=int, default=_DEFAULT_TRIALS, metavar="N",
        help="Number of timing trials (after warmup).",
    )
    parser.add_argument(
        "--warmup", type=int, default=_DEFAULT_WARMUP, metavar="N",
        help="Warmup iterations (excluded from statistics).",
    )
    parser.add_argument(
        "--seed", type=int, default=_DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> bool:
    """Run the G18 full-stack latency benchmark. Returns True if it passes."""
    args = _parse_args(argv)

    print(
        f"G18 Full-Stack Benchmark\n"
        f"  Config : use_adapter=True  use_quantization=True  use_nstp=True\n"
        f"  DIM_IN={args.dim_in}  DIM_PROJ={args.dim_proj}  K={args.k}\n"
        f"  CAM entries={args.cam_entries}  trials={args.trials}  seed={args.seed}\n"
        f"Running {args.trials} trials (+ {args.warmup} warmup) …"
    )

    stage_timings, total_timings = _run_benchmark(
        dim_in=args.dim_in,
        dim_proj=args.dim_proj,
        value_dim=args.value_dim,
        cam_entries=args.cam_entries,
        k=args.k,
        trials=args.trials,
        warmup=args.warmup,
        seed=args.seed,
    )
    return _print_report(stage_timings, total_timings, k=args.k, trials=args.trials)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
