# MT-9: Coreset Allocation vs. Online Write Path

Dataset: Stanford Dogs (1024-d features)
Adapter: Production Dogs

### Results Table

| N | Coreset Acc | FAM Acc | Delta | Coreset Cov Radius | FAM Cov Radius |
|---|-------------|---------|-------|--------------------|----------------|
| 1000 | 89.10% | 38.57% | +50.54pp | 0.1917 | 0.3692 |
| 5000 | 90.15% | 61.20% | +28.95pp | 0.1321 | 0.2356 |
| 10000 | 90.13% | 88.75% | +1.38pp | 0.1095 | 0.1140 |
| 25000 | 90.03% | 90.08% | -0.05pp | 0.1061 | 0.1093 |
| 50000 | 90.03% | 90.08% | -0.05pp | 0.1061 | 0.1093 |

### Analysis

Mixed results depending on capacity. The online write path differs dynamically from greedy k-center coverage maximization.

**Condensation Gap (Coreset - FAM Acc) [Maximum potential improvement from better placement]:**
- N=1000: +50.54pp
- N=5000: +28.95pp
- N=10000: +1.38pp
- N=25000: -0.05pp
- N=50000: -0.05pp
