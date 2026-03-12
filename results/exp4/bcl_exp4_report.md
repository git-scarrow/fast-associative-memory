# BCL-EXP-4: Per-Task-Partitioned Accumulator Scaling Gauntlet

## 1. Configuration

- N values: [5, 10, 20, 30, 40, 50]
- Seeds: [42, 43, 44]
- Conditions: vanilla GEM (memory_size=256/task) vs partitioned-accumulator (budget=1280, hot_frac=0.5, flush=100)
- Batch size: 128
- Benchmark: Permuted-MNIST
- Total runs: 36

## 2. Results

| N | Condition | Seed | BWT (%) | Avg Acc (%) |
|---|-----------|------|---------|-------------|
| 5 | partitioned | 42 | -1.99 | 94.04 |
| 5 | partitioned | 43 | -2.56 | 93.37 |
| 5 | partitioned | 44 | -2.92 | 93.51 |
| 5 | vanilla | 42 | -2.03 | 94.01 |
| 5 | vanilla | 43 | -2.57 | 93.67 |
| 5 | vanilla | 44 | -2.68 | 93.51 |
| 10 | partitioned | 42 | -8.54 | 88.27 |
| 10 | partitioned | 43 | -7.87 | 88.85 |
| 10 | partitioned | 44 | -8.32 | 88.35 |
| 10 | vanilla | 42 | -3.82 | 92.41 |
| 10 | vanilla | 43 | -3.91 | 92.45 |
| 10 | vanilla | 44 | -5.41 | 91.00 |
| 20 | partitioned | 42 | -21.24 | 75.86 |
| 20 | partitioned | 43 | -18.99 | 78.06 |
| 20 | partitioned | 44 | -21.14 | 75.84 |
| 20 | vanilla | 42 | -8.86 | 87.55 |
| 20 | vanilla | 43 | -7.61 | 88.72 |
| 20 | vanilla | 44 | -7.83 | 88.46 |
| 30 | partitioned | 42 | -32.15 | 65.01 |
| 30 | partitioned | 43 | -30.41 | 66.56 |
| 30 | partitioned | 44 | -31.47 | 65.58 |
| 30 | vanilla | 42 | -9.38 | 86.94 |
| 30 | vanilla | 43 | -9.29 | 86.98 |
| 30 | vanilla | 44 | -9.60 | 86.71 |
| 40 | partitioned | 42 | -37.03 | 59.90 |
| 40 | partitioned | 43 | -35.79 | 61.06 |
| 40 | partitioned | 44 | -37.62 | 59.34 |
| 40 | vanilla | 42 | -10.96 | 85.31 |
| 40 | vanilla | 43 | -10.80 | 85.37 |
| 40 | vanilla | 44 | -10.97 | 85.20 |
| 50 | partitioned | 42 | -44.74 | 52.18 |
| 50 | partitioned | 43 | -43.90 | 52.98 |
| 50 | partitioned | 44 | -43.43 | 53.38 |
| 50 | vanilla | 42 | -11.89 | 84.27 |
| 50 | vanilla | 43 | -11.79 | 84.34 |
| 50 | vanilla | 44 | -12.01 | 84.09 |

## 3. Δ(BWT) vs N Regression

- **Slope**: -0.7233 pp/task
- **95% CI**: [-0.7749, -0.6717]
- **R²**: 0.9822
- **p-value**: 0.0000
- **EXP-2 shared-pool slope**: −0.60 pp/task (for comparison)

## 4. Per-Task Exemplar Counts (Partitioned)

- N=5: min=128, max=704, mean=243.2, partition_size=128
- N=10: min=64, max=640, mean=121.6, partition_size=64
- N=20: min=32, max=608, mean=60.8, partition_size=32
- N=30: min=21, max=597, mean=40.2, partition_size=21
- N=40: min=16, max=592, mean=30.4, partition_size=16
- N=50: min=12, max=588, mean=23.5, partition_size=12

## 5. Verdict

**KILL**: Slope CI [-0.7749, -0.6717] entirely below zero. Scaling failure is inherent to accumulator pattern.

The scaling failure is **inherent** to the two-tier accumulator pattern. Partitioning the cold zone does not rescue the negative Δ(BWT) slope. BCL scaling narrative should be archived.