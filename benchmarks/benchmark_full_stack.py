#!/usr/bin/env python3
"""
benchmarks/benchmark_full_stack.py — G18: Full Stack E2E Latency Benchmark.

Verifies that the "Fast" in FAM holds up when all optional modules are active:

  - MetricAdapter      — learned metric projection between backbone and CAM core.
  - QuantizedCore      — bfloat16 ContinuousCAM (reduced-precision key/value store).
  - NSTPController     — lateral inhibition on the K=50 retrieval subgraph.

Measures wall-clock time for the full forward pass, **excluding** the DINOv2
backbone (backbone inference time is constant and independent of FAM).

Benchmark stages (per query)
----------------------------
  Stage 1 — Adapter + QuantizedCore:
    fam.forward(query) — MetricAdapter projection followed by bfloat16 CAM
    retrieval (broad cosine search → Mahalanobis re-rank → softmax vote).

  Stage 2 — NSTP:
    NSTPController.prune_batch() on a K=50 candidate subgraph — lateral
    inhibition to resolve sibling conflicts (Granularity Collapse fix).

Pass criterion (G18)
--------------------
Mean total latency < 2 ms per query on CPU.

Usage
-----
  python benchmarks/benchmark_full_stack.py
  python benchmarks/benchmark_full_stack.py --k 50 --trials 1000 --seed 42
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Allow importing from the repo root regardless of working directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapter import MetricAdapter                        # noqa: E402
from fast_associative_memory import FastAssociativeMemory  # noqa: E402
from nstp import NSTPController                          # noqa: E402


# ──────────────────────────────────────────────────────────── constants ───

_KILL_THRESHOLD_MS = 2.0      # 2 ms per query (G18 kill condition)
_DEFAULT_INPUT_DIM = 1024     # DINOv2 ViT-L/14 post-backbone embedding dimension
_DEFAULT_ADAPTER_DIM = 512    # adapter output dimension fed into the CAM core
_DEFAULT_N_CLASSES = 100      # number of prototype value classes
_DEFAULT_K = 50               # inference top-K retrieval and NSTP subgraph size
_DEFAULT_N_ENTRIES = 5000     # prototypes pre-loaded into the CAM core
_DEFAULT_TRIALS = 1000        # number of timing trials for stable statistics
_DEFAULT_SEED = 42
_DEFAULT_BATCH = 1            # single-query latency (worst case for latency)


# ─────────────────────────────────────────────────────────────── setup ───

def _build_full_stack(
    input_dim: int,
    adapter_dim: int,
    n_classes: int,
    k: int,
    n_entries: int,
    seed: int,
):
    """Construct and pre-populate the adapter + quantized-core + NSTP stack.

    Parameters
    ----------
    input_dim   : Post-backbone embedding dimension (e.g. 1024 for ViT-L/14).
    adapter_dim : Adapter output dimension, also the CAM key dimension.
    n_classes   : Number of value classes stored in the CAM.
    k           : Top-K retrieval count (inference_k) and NSTP subgraph size.
    n_entries   : Number of prototypes to pre-load into the CAM core.
    seed        : Random seed for reproducibility.

    Returns
    -------
    fam  : FastAssociativeMemory configured with adapter + bfloat16 core.
    nstp : NSTPController for lateral inhibition.
    """
    torch.manual_seed(seed)

    adapter = MetricAdapter(input_dim=input_dim, output_dim=adapter_dim)
    fam = FastAssociativeMemory(
        input_dim=input_dim,
        value_dim=n_classes,
        core_entries=n_entries,
        inference_k=k,
        use_bfloat16=True,   # quantization: bfloat16 key/value memory
        adapter=adapter,
    )
    nstp = NSTPController()

    # Pre-populate core with n_entries synthetic prototypes so that retrieval
    # exercises real occupied-slot code paths (non-trivial similarity search).
    with torch.no_grad():
        proto_keys = F.normalize(
            torch.randn(n_entries, adapter_dim), dim=-1
        ).to(fam.core_cam.keys.dtype)
        proto_values = F.one_hot(
            torch.randint(0, n_classes, (n_entries,)), num_classes=n_classes
        ).float().to(fam.core_cam.values.dtype)
        fam.core_cam.keys[:n_entries] = proto_keys
        fam.core_cam.values[:n_entries] = proto_values
        fam.core_cam.occupied[:n_entries] = True
        fam.core_cam._update_key_norm(torch.arange(n_entries))

    return fam, nstp


# ──────────────────────────────────────────────────────── timing ───

def _run_benchmark(
    input_dim: int,
    adapter_dim: int,
    n_classes: int,
    k: int,
    n_entries: int,
    trials: int,
    seed: int,
    batch: int = 1,
):
    """Run the full-stack E2E latency benchmark.

    Each trial measures:
      * Stage 1: ``fam.forward(query)``  — adapter projection + bfloat16 CAM retrieval.
      * Stage 2: ``nstp.prune_batch(...)`` — lateral inhibition on K candidates.

    The trial total is the sum of Stage 1 and Stage 2 (tensor-generation
    scaffolding between stages is excluded from all timings).

    Parameters
    ----------
    input_dim   : Post-backbone embedding dimension.
    adapter_dim : Adapter output / CAM key dimension.
    n_classes   : Number of value classes.
    k           : Top-K retrieval count and NSTP subgraph size.
    n_entries   : Number of pre-loaded core prototypes.
    trials      : Number of timing trials.
    seed        : Random seed.
    batch       : Query batch size (default 1 for worst-case latency).

    Returns
    -------
    stage_timings : dict[str, list[float]]
        Per-trial latencies in **milliseconds**, keyed by stage name.
    total_timings : list[float]
        Per-trial total latency in milliseconds (Stage 1 + Stage 2).
    """
    fam, nstp = _build_full_stack(
        input_dim, adapter_dim, n_classes, k, n_entries, seed
    )

    # Warmup: stabilise CPU caches and any lazy initialisation.
    with torch.no_grad():
        warmup_q = F.normalize(torch.randn(batch, input_dim), dim=-1)
        for _ in range(10):
            _ = fam(warmup_q)

    stage_timings: dict[str, list[float]] = {
        "adapter_quantized_core": [],
        "nstp": [],
    }
    total_timings: list[float] = []

    with torch.no_grad():
        for _ in range(trials):
            query = F.normalize(torch.randn(batch, input_dim), dim=-1)

            # ── Stage 1: Adapter + QuantizedCore ──────────────────────────
            t0 = time.perf_counter()
            _ = fam(query)
            t1 = time.perf_counter()
            t_s1 = (t1 - t0) * 1e3   # ms
            stage_timings["adapter_quantized_core"].append(t_s1)

            # ── Stage 2: NSTP ──────────────────────────────────────────────
            # Build NSTP inputs: projected query (adapter output) and
            # synthetic K-candidate tensors that represent realistic CAM output.
            # _project() mirrors the adapter step already timed in Stage 1 —
            # it is called a second time here only to obtain the projected query
            # vector needed as the NSTP root-election input, and this call is
            # NOT included in either stage or total timings.
            query_proj = fam._project(query)   # (B, adapter_dim)
            top_k_keys = F.normalize(
                torch.randn(batch, k, adapter_dim), dim=-1
            )
            top_k_values = F.one_hot(
                torch.randint(0, n_classes, (batch, k)), num_classes=n_classes
            ).float()
            top_k_sims = torch.rand(batch, k)

            t0 = time.perf_counter()
            _, _ = nstp.prune_batch(
                query_proj, top_k_keys, top_k_values, top_k_sims
            )
            t1 = time.perf_counter()
            t_s2 = (t1 - t0) * 1e3   # ms
            stage_timings["nstp"].append(t_s2)

            total_timings.append(t_s1 + t_s2)

    return stage_timings, total_timings


# ──────────────────────────────────────────────── helpers ───

def _mean(values: list[float]) -> float:
    """Return the arithmetic mean of *values*, or 0.0 if empty."""
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    """Return the *pct*-th percentile of *values* (linear interpolation)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    idx = (pct / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo])


# ────────────────────────────────────────────── reporting ───

def _print_report(
    stage_timings: dict[str, list[float]],
    total_timings: list[float],
    k: int,
    trials: int,
) -> bool:
    """Print the results table and return True if the benchmark passes."""
    sep = "=" * 72
    stage_labels = {
        "adapter_quantized_core": "Stage 1 — Adapter + QuantizedCore",
        "nstp":                   f"Stage 2 — NSTP (K={k} candidates)",
    }

    print(f"\n{sep}")
    print(f"  G18 Full Stack E2E Latency  |  K={k}  |  N={trials} trials")
    print(sep)
    print(f"  {'Stage':<44} {'Mean (ms)':>10} {'p99 (ms)':>10}")
    print(f"  {'-' * 66}")

    for key, label in stage_labels.items():
        vals = stage_timings[key]
        mean = _mean(vals)
        p99 = _percentile(vals, 99)
        print(f"  {label:<44} {mean:>10.3f} {p99:>10.3f}")

    total_mean = _mean(total_timings)
    total_p99 = _percentile(total_timings, 99)

    print(f"  {'-' * 66}")
    print(f"  {'TOTAL (Stage 1 + Stage 2)':<44} {total_mean:>10.3f} {total_p99:>10.3f}")
    print(sep)

    passed = total_mean < _KILL_THRESHOLD_MS
    if passed:
        print(
            f"\n  ✓  PASS — mean total = {total_mean:.3f} ms  "
            f"< {_KILL_THRESHOLD_MS:.1f} ms kill threshold."
        )
        print(
            "       Full stack (Adapter + QuantizedCore + NSTP) "
            "satisfies the FAM latency SLA."
        )
    else:
        print(
            f"\n  ✗  FAIL — mean total = {total_mean:.3f} ms  "
            f"≥ {_KILL_THRESHOLD_MS:.1f} ms kill threshold."
        )
        print(
            "       Full stack latency exceeds the 2 ms SLA; "
            "the 'Fast' claim requires investigation."
        )

    return passed


# ────────────────────────────────────────────── CLI entry point ───

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "G18: Full Stack E2E Latency Benchmark "
            "(Adapter + QuantizedCore + NSTP)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dim", type=int, default=_DEFAULT_INPUT_DIM, metavar="D",
        help="Input embedding dimension (post-backbone, pre-adapter).",
    )
    parser.add_argument(
        "--adapter-dim", type=int, default=_DEFAULT_ADAPTER_DIM, metavar="D",
        help="Adapter output dimension fed into the CAM core.",
    )
    parser.add_argument(
        "--n-classes", type=int, default=_DEFAULT_N_CLASSES, metavar="C",
        help="Number of value classes in the CAM core.",
    )
    parser.add_argument(
        "--k", type=int, default=_DEFAULT_K, metavar="K",
        help="Inference top-K retrieval count and NSTP subgraph size.",
    )
    parser.add_argument(
        "--n-entries", type=int, default=_DEFAULT_N_ENTRIES, metavar="N",
        help="Number of prototypes pre-loaded into the CAM core.",
    )
    parser.add_argument(
        "--trials", type=int, default=_DEFAULT_TRIALS, metavar="N",
        help="Number of timing trials for stable statistics.",
    )
    parser.add_argument(
        "--seed", type=int, default=_DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--batch", type=int, default=_DEFAULT_BATCH, metavar="B",
        help="Query batch size (default 1 for worst-case single-query latency).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> bool:
    """Run the G18 full-stack latency benchmark and return True if it passes."""
    args = _parse_args(argv)

    print(
        f"Input dim   : {args.input_dim}   "
        f"Adapter dim : {args.adapter_dim}   "
        f"K           : {args.k}"
    )
    print(
        f"n_entries   : {args.n_entries}   "
        f"Trials      : {args.trials}   "
        f"Batch       : {args.batch}   "
        f"Seed        : {args.seed}"
    )
    print(f"Modules     : use_adapter=True  use_quantization=True  use_nstp=True")
    print(f"Running {args.trials} trials on CPU …")

    stage_timings, total_timings = _run_benchmark(
        input_dim=args.input_dim,
        adapter_dim=args.adapter_dim,
        n_classes=args.n_classes,
        k=args.k,
        n_entries=args.n_entries,
        trials=args.trials,
        seed=args.seed,
        batch=args.batch,
    )
    passed = _print_report(stage_timings, total_timings, k=args.k, trials=args.trials)
    return passed


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(0 if main() else 1)
