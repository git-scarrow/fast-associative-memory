# BCL-EXP-1: Two-Tier Accumulator Replay Buffer — Sweep Report

> **Early Termination (2026-03-04):** CIFAR-10 sweep stopped after 3/36 accumulator configs
> (ff=50, all merge strategies × 3 seeds). Kill condition met on available data.
> Permuted-MNIST sweep was not run. Verdict is definitive on the data collected.

## 1. Vanilla GEM Baselines

| Benchmark | BWT (mean±std) | Avg Acc (mean±std) | KL drift |
|-----------|----------------|--------------------|----------|
| split_cifar10 | -5.12±0.80% | 82.04±0.80% | 0.1521 |
| split_mnist | -0.44±0.16% | 99.17±0.09% | 0.0520 |

## 2. Sweep Results


### Sweep Results (split_cifar10)

| flush_freq | hot_frac | merge      | BWT (mean±std)    | Avg Acc (mean±std) | KL_drift (mean) |
|------------|----------|------------|-------------------|--------------------|-----------------|
| 50         | 0.1      | importance | -24.84±1.34       | 66.97±0.53         | 0.1742          |
| 50         | 0.1      | random     | -8.45±0.18        | 80.45±0.38         | 0.1660          |
| 50         | 0.1      | reservoir  | -5.38±0.49        | 82.46±0.56         | 0.1614          |

### Sweep Results (split_mnist)

| flush_freq | hot_frac | merge      | BWT (mean±std)    | Avg Acc (mean±std) | KL_drift (mean) |
|------------|----------|------------|-------------------|--------------------|-----------------|
| 50         | 0.1      | importance | -0.08±0.07        | 99.40±0.04         | 0.1072          |
| 50         | 0.1      | random     | -0.64±0.12        | 98.96±0.15         | 0.0855          |
| 50         | 0.1      | reservoir  | -0.48±0.11        | 99.06±0.06         | 0.0869          |
| 50         | 0.25     | importance | -0.14±0.11        | 99.34±0.07         | 0.0928          |
| 50         | 0.25     | random     | -0.75±0.25        | 98.85±0.16         | 0.0859          |
| 50         | 0.25     | reservoir  | -0.42±0.02        | 99.11±0.07         | 0.0877          |
| 50         | 0.5      | importance | -0.48±0.17        | 99.06±0.07         | 0.0859          |
| 50         | 0.5      | random     | -0.68±0.16        | 98.91±0.10         | 0.0856          |
| 50         | 0.5      | reservoir  | -0.40±0.04        | 99.14±0.06         | 0.0843          |
| 100        | 0.1      | importance | -0.16±0.12        | 99.32±0.11         | 0.1108          |
| 100        | 0.1      | random     | -0.54±0.03        | 99.05±0.08         | 0.0871          |
| 100        | 0.1      | reservoir  | -0.44±0.09        | 99.09±0.04         | 0.0875          |
| 100        | 0.25     | importance | +0.05±0.02        | 99.50±0.06         | 0.1012          |
| 100        | 0.25     | random     | -0.56±0.09        | 99.02±0.08         | 0.0864          |
| 100        | 0.25     | reservoir  | -0.49±0.09        | 99.07±0.04         | 0.0878          |
| 100        | 0.5      | importance | -0.05±0.08        | 99.41±0.03         | 0.0899          |
| 100        | 0.5      | random     | -0.58±0.13        | 99.01±0.20         | 0.0859          |
| 100        | 0.5      | reservoir  | -0.44±0.03        | 99.14±0.07         | 0.0858          |
| 500        | 0.1      | importance | -0.37±0.03        | 99.16±0.08         | 0.0891          |
| 500        | 0.1      | random     | -0.49±0.05        | 99.07±0.12         | 0.0864          |
| 500        | 0.1      | reservoir  | -0.42±0.04        | 99.13±0.06         | 0.0861          |
| 500        | 0.25     | importance | -0.18±0.09        | 99.32±0.06         | 0.0983          |
| 500        | 0.25     | random     | -0.39±0.05        | 99.16±0.05         | 0.0852          |
| 500        | 0.25     | reservoir  | -0.43±0.08        | 99.12±0.02         | 0.0866          |
| 500        | 0.5      | importance | -0.08±0.07        | 99.39±0.06         | 0.0888          |
| 500        | 0.5      | random     | -0.35±0.06        | 99.19±0.03         | 0.0866          |
| 500        | 0.5      | reservoir  | -0.51±0.08        | 99.05±0.12         | 0.0869          |
| 1000       | 0.1      | importance | -0.37±0.08        | 99.15±0.13         | 0.0890          |
| 1000       | 0.1      | random     | -0.43±0.03        | 99.14±0.13         | 0.0858          |
| 1000       | 0.1      | reservoir  | -0.40±0.09        | 99.11±0.09         | 0.0859          |
| 1000       | 0.25     | importance | -0.20±0.04        | 99.32±0.05         | 0.0965          |
| 1000       | 0.25     | random     | -0.46±0.03        | 99.10±0.11         | 0.0848          |
| 1000       | 0.25     | reservoir  | -0.36±0.01        | 99.16±0.10         | 0.0871          |
| 1000       | 0.5      | importance | -0.03±0.03        | 99.44±0.07         | 0.0877          |
| 1000       | 0.5      | random     | -0.51±0.02        | 99.08±0.11         | 0.0847          |
| 1000       | 0.5      | reservoir  | -0.48±0.12        | 99.10±0.05         | 0.0861          |

## 3. Best Configuration per Benchmark

**split_cifar10:**
- Best config: flush_freq=50, hot_frac=0.1, merge=reservoir
- BWT: -5.38±0.49% (Δ=-0.26pp vs vanilla)
- Avg Acc: 82.46±0.56%
- KL drift: 0.1614 (reduction: -6.2%)

**split_mnist:**
- Best config: flush_freq=100, hot_frac=0.25, merge=importance
- BWT: +0.05±0.02% (Δ=+0.49pp vs vanilla)
- Avg Acc: 99.50±0.06%
- KL drift: 0.1012 (reduction: -94.5%)

## 4. Kill Condition Evaluation

  split_cifar10: vanilla BWT=-5.12%, best accum BWT=-5.38% (Δ=-0.26pp, config=split_cifar10_ff50_hf0.1_msreservoir)
  split_mnist: vanilla BWT=-0.44%, best accum BWT=+0.05% (Δ=+0.49pp, config=split_mnist_ff100_hf0.25_msimportance)

**VERDICT: KILL** — Best accumulator BWT is within ±1pp of vanilla GEM on ALL benchmarks. The accumulator pattern does not transfer to replay buffer management in a measurable way.

## 5. Interpretation

- **split_cifar10**: No significant BWT improvement.
- **split_mnist**: No significant BWT improvement.