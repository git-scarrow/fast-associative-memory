# MT-11: Curriculum Stress Test

## Objective

Test whether coverage-aware eviction handles pathological stream orderings.
MT-4/MT-6 showed class-sorted arrival is catastrophic under LFU (+55pp delta).
MT-12 showed coverage eviction closes the coreset gap under default ordering.
Does it survive curriculum adversity?

## Config
- **Commit**: `d515f09`
- **Coverage eviction**: verified in production `associative_core.py`
- **Device**: cuda
- **Dataset**: Stanford Dogs 1024-d, production adapter
- **Capacity**: 5000
- **Block-shuffle window**: 300
- **Coreset reference**: 90.15%

## Results: 2x3 Accuracy Matrix

|                 | Class-sorted | IID Shuffled | Block-shuffled | Spread |
|-----------------|-------------|-------------|----------------|--------|
| **Coverage (new)** | 89.84% | 89.93% | 89.49% | 0.4pp |
| **LFU (old)** | 59.09% | 89.93% | 59.08% | 30.9pp |
| **Coreset** | 90.15% | — | — | — |

## Class Distribution

|                 | Class-sorted | IID Shuffled | Block-shuffled |
|-----------------|-------------|-------------|----------------|
| **Coverage (new)** | 120/120 | 120/120 | 120/120 |
| **LFU (old)** | 79/120 | 120/120 | 79/120 |

## Per-Class Accuracy (Class-Sorted Condition)

| Write Path | Early (0–9) | Mid (55–64) | Late (110–119) |
|------------|------------|------------|----------------|
| **Coverage (new)** | 91.6% | 93.3% | 88.0% |
| **LFU (old)** | 27.7% | 28.0% | 89.0% |

## Verdict

**CURRICULUM-ROBUST**

Coverage eviction spread is 0.4pp across orderings (vs LFU's 30.9pp). The write path handles class-sorted arrival gracefully. No stream preprocessor needed.

## Artifacts
- `mt11_metrics.json` — all metrics and per-cell results
