# MT-6: EMA-Disabled Architectual Test

Dataset: Stanford Dogs (1024-d features)
Adapter: Production Dogs

### Results Table

| Capacity | EMA-On Acc | EMA-Off Acc | Delta | EMA-On CR | EMA-Off CR | EMA-On Occ | EMA-Off Occ |
|----------|-----------|-------------|-------|-----------|------------|------------|-------------|
| 1000 | 38.57% | 15.41% | 23.16pp | 0.0833 | 0.0833 | 1000 | 1000 |
| 5000 | 61.20% | 51.15% | 10.05pp | 0.4167 | 0.4167 | 5000 | 5000 |
| 10000 | 88.75% | 88.72% | 0.03pp | 0.8333 | 0.8333 | 10000 | 10000 |
| 25000 | 90.08% | 90.05% | 0.03pp | 0.8566 | 0.8566 | 10279 | 10279 |
| 50000 | 90.08% | 90.05% | 0.03pp | 0.8566 | 0.8566 | 10279 | 10279 |

### Analysis

Significant differences detected (max delta 23.16pp). Accuracy drops at low capacity but not high capacity. EMA refinement matters when coverage is sparse and every prototype position counts.

Transition points exceeding 1pp delta:
- Capacity 1000: 23.16pp drop without EMA
- Capacity 5000: 10.05pp drop without EMA
