# BCL-EXP-3: Log-Structured Merge EWC — Experiment Report

## 1. Implementation Summary

LogStructuredEWC implements a write-ahead log for diagonal Fisher information matrices. After each task, the per-task Fisher is appended to a log (O(1) bookkeeping) rather than immediately merged into a consolidated store. Every C tasks, a compaction step merges all log entries into the consolidated matrix using one of three weighting strategies: uniform (sum), recency-weighted (linear ramp favoring recent tasks), or magnitude-weighted (proportional to Fisher L2 norm). The read path computes `F_consolidated + sum(log_entries)` on each regularization call. StandardEWC serves as a faithful baseline with immediate `F += F_t` after every task. Both implementations share identical model architectures, training loops, and Fisher computation (empirical diagonal via gradient-squared averaging over 2000 samples).

Device: CUDA (PyTorch 2.10+cu128). Timing via `torch.cuda.synchronize()` + `time.perf_counter()`. Seeds: 42, 43, 44.

## 2. Comparison Tables

### Split-MNIST (5 tasks, MLP 784-256-256-2)

| Method                | BWT            | Avg Acc        | Consol. Time/Task | Peak Mem |
|-----------------------|----------------|----------------|--------------------|----------|
| Standard EWC          | -41.47%+/-0.29 | 65.00%+/-0.21  | 0.1727ms           | 1        |
| Log-EWC C=2 uniform   | -41.47%+/-0.29 | 65.00%+/-0.21  | 0.2167ms           | 2        |
| Log-EWC C=5 uniform   | -41.47%+/-0.29 | 65.00%+/-0.21  | 0.2009ms           | 4        |
| **Log-EWC C=10 uniform** | -41.47%+/-0.29 | 65.00%+/-0.21  | **0.1667ms**       | 5        |
| Log-EWC C=20 uniform  | -41.47%+/-0.29 | 65.00%+/-0.21  | 0.1632ms           | 5        |
| Log-EWC C=10 recency  | -41.47%+/-0.29 | 65.00%+/-0.21  | 0.1514ms           | 5        |
| Log-EWC C=10 magnitude| -41.47%+/-0.29 | 65.00%+/-0.21  | 0.1439ms           | 5        |

All 12 configurations produce **identical** accuracy and BWT to standard EWC on Split-MNIST (approx quality = 0.000000 at all compaction boundaries). This is expected: with only 5 tasks and C >= 5, no compaction occurs before the final evaluation; with uniform weighting, the sum is mathematically identical to standard accumulation.

### Split-CIFAR-10 (5 tasks, SmallCNN)

| Method                | BWT            | Avg Acc        | Consol. Time/Task | Peak Mem |
|-----------------------|----------------|----------------|--------------------|----------|
| Standard EWC          | -6.90%+/-1.45  | 62.59%+/-2.05  | 0.3482ms           | 1        |
| Log-EWC C=2 uniform   | -6.95%+/-1.48  | 62.54%+/-2.05  | 0.5571ms           | 2        |
| Log-EWC C=5 uniform   | -6.93%+/-1.48  | 62.59%+/-2.08  | 0.3964ms           | 4        |
| Log-EWC C=10 uniform  | -6.86%+/-1.48  | 62.63%+/-2.09  | 0.2897ms           | 5        |
| **Log-EWC C=20 uniform** | -6.92%+/-1.52 | 62.62%+/-2.07 | **0.2408ms**       | 5        |
| Log-EWC C=10 recency  | -6.89%+/-1.46  | 62.59%+/-2.07  | 0.3136ms           | 5        |
| Log-EWC C=20 magnitude| -6.90%+/-1.45  | 62.59%+/-2.04  | 0.2747ms           | 5        |

Accuracy differences are within noise (<0.1pp). CIFAR-10 shows slight non-zero approximation error (max 0.009) for non-uniform strategies due to the re-weighting changing the effective Fisher sum.

### Permuted-MNIST (5 tasks, MLP 784-256-256-10)

| Method                | BWT            | Avg Acc        | Consol. Time/Task | Peak Mem |
|-----------------------|----------------|----------------|--------------------|----------|
| Standard EWC          | -9.37%+/-1.18  | 84.82%+/-0.90  | 0.1882ms           | 1        |
| Log-EWC C=2 uniform   | -9.37%+/-1.17  | 84.82%+/-0.90  | 0.2585ms           | 2        |
| Log-EWC C=5 uniform   | -9.37%+/-1.18  | 84.82%+/-0.90  | 0.2736ms           | 4        |
| **Log-EWC C=10 uniform** | -9.37%+/-1.18 | 84.82%+/-0.90 | **0.1414ms**       | 5        |
| Log-EWC C=20 uniform  | -9.37%+/-1.18  | 84.82%+/-0.90  | 0.1613ms           | 5        |
| Log-EWC C=10 recency  | -9.37%+/-1.18  | 84.82%+/-0.90  | 0.1604ms           | 5        |
| Log-EWC C=10 magnitude| -9.37%+/-1.18  | 84.82%+/-0.90  | 0.1414ms           | 5        |

## 3. Scaling Experiment (Permuted-MNIST, C=5, uniform)

| N  | Std EWC Time/Task | Log-EWC Time/Task | Ratio | Acc Delta |
|----|-------------------|--------------------|-------|-----------|
| 5  | 0.1835ms          | 0.2291ms           | 0.801 | +0.00pp   |
| 10 | 0.1645ms          | 0.2737ms           | 0.601 | +0.00pp   |
| 20 | 0.1705ms          | 0.2505ms           | 0.681 | -0.04pp   |
| 50 | 0.1743ms          | 0.2596ms           | 0.671 | +0.17pp   |

The amortization ratio (std_time / log_time) is consistently **below 1.0** across all N, meaning log-structured EWC is *slower* than standard EWC. The ratio does not grow with N; it slightly decreases (0.801 -> 0.671).

Note: C=10/C=20 configs on individual benchmarks occasionally showed ratios >1.0 (e.g., `log_C10_uniform` at 1.331x on permuted_mnist) because they never compact during small task counts, eliminating compaction overhead entirely. This is degenerate behavior (the log IS the store).

## 4. Approximation Quality

| Benchmark      | Config         | ||F_log - F_std|| / ||F_std|| |
|----------------|----------------|-------------------------------|
| Split-MNIST    | All configs    | 0.000000                      |
| Split-CIFAR-10 | C=2, uniform   | 0.001, 0.002, 0.005           |
| Split-CIFAR-10 | C=2, recency   | 0.001, 0.002, 0.009           |
| Split-CIFAR-10 | C=20, magnitude| 0.007                         |
| Permuted-MNIST | C=2, magnitude | 0.000, 0.005, 0.007           |
| Permuted-MNIST | C>=5, uniform  | 0.000                         |

For **uniform** weighting, approximation quality is near-perfect (< 0.005) because `sum(log) = consolidated` by construction — the only source of divergence is that different regularization strengths during intermediate training steps cause the models to follow slightly different optimization paths. Non-uniform strategies (recency, magnitude) introduce genuine approximation error but it remains below 1%.

## 5. Kill Condition Check

- **KILL-1** (>2pp degradation with no cost reduction): **NOT TRIGGERED.** No configuration showed >0.1pp accuracy degradation vs standard EWC.
- **KILL-2** (>5pp degradation despite cost reduction): **NOT TRIGGERED.** No configuration showed >0.2pp accuracy degradation.

## 6. Summary Verdict

The log-structured merge pattern does **NOT** amortize EWC consolidation cost for diagonal Fisher matrices. The fundamental reason: standard EWC's consolidation step (`F_consolidated += F_task`) is a single elementwise tensor addition — already O(p) with minimal constant factor on GPU. There is nothing meaningful to amortize. The log-structured variant adds overhead from tensor cloning (to preserve log entries), list management, and — critically — a more expensive read path that must sum all pending log entries during each regularization computation (called every training step). This read-path amplification dominates any savings from deferred writes.

The pattern produces **identical accuracy** to standard EWC when using uniform weighting (mathematically equivalent), and negligible accuracy differences (<0.1pp) with non-uniform strategies. No kill conditions were triggered.

**Recommended configuration:** If the log-structured pattern is used regardless (e.g., for auditing/rollback of per-task Fisher contributions), use C=10 or C=20 with uniform weighting. These configurations never compact on short task sequences, eliminating overhead, and match standard EWC exactly.

**Hypothesis 2 is NOT SUPPORTED** for diagonal EWC. The LSM-tree pattern would be more relevant for: (a) full Fisher matrices where consolidation is O(p^2), (b) block-diagonal or Kronecker-factored approximations (K-FAC) where merging involves non-trivial matrix operations, or (c) systems where the consolidation involves compression/quantization that benefits from batching. Diagonal EWC's consolidation is too cheap to benefit from amortization.
