# MT-12: Coverage-Aware Eviction (Minimal Intervention)

## Objective

Isolate the contribution of eviction policy to the FAM-vs-coreset accuracy gap.
Swap LFU for coverage-aware eviction; keep vigilance, EMA, everything else identical.

## Config
- **Commit**: `d515f09`
- **Eval contamination fix**: verified on main
- **Device**: cuda
- **Dataset**: Stanford Dogs 1024-d, production adapter
- **Capacities**: 1000, 5000, 10000
- **Eviction rule**: Evict prototype with nearest same-class neighbor (most replaceable).
  Sole class representatives are protected (score = inf).

## Results

| Capacity | LFU (re-run) | Coverage Evict | Coreset (MT-9) | Gap Closure |
|----------|-------------|----------------|----------------|-------------|
| 1,000 | 36.56% | 71.69% | 89.10% | 66.9% |
| 5,000 | 59.09% | 89.84% | 90.15% | 99.0% |
| 10,000 | 88.78% | 90.09% | 90.13% | 97.3% |

### LFU re-run vs MT-7 reference

| Capacity | MT-7 (pre-fix) | Re-run (post-fix) | Delta |
|----------|---------------|-------------------|-------|
| 1,000 | 38.57% | 36.56% | -2.01pp |
| 5,000 | 61.20% | 59.09% | -2.11pp |
| 10,000 | 88.75% | 88.78% | +0.03pp |

### Class distribution

| Capacity | Metric | LFU | Coverage |
|----------|--------|-----|----------|
| 1,000 | Classes represented | 48/120 | 117/120 |
| | Min protos/class | 2 | 1 |
| 5,000 | Classes represented | 79/120 | 120/120 |
| | Min protos/class | 2 | 10 |
| 10,000 | Classes represented | 118/120 | 120/120 |
| | Min protos/class | 15 | 15 |

### Timing

| Capacity | LFU | Coverage |
|----------|-----|----------|
| 1,000 | 0.3s | 0.3s |
| 5,000 | 0.1s | 0.3s |
| 10,000 | 0.1s | 0.1s |

## Decision

**EVICTION IS THE BOTTLENECK**

Coverage-aware eviction closes >=50% of the gap at low capacity. Ship the eviction fix. MT-10 full rewrite is optional.

## Artifacts
- `mt12_metrics.json` — all metrics
