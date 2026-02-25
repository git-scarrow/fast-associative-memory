# G36: Leave-One-Out Candidate Influence Map

## Setup
- **Dataset**: Stanford Dogs (120 classes)
- **Eval queries**: 500
- **Broad K**: 100
- **Full accuracy on sample**: 91.60%
- **Total (query, candidate) pairs**: 50,000

## Key Takeaways

### Influence Distribution
- **Load-bearing** (delta > 0): 29 pairs (0.1%)
- **Harmful** (delta < 0): 11 pairs (0.0%)
- **Redundant** (delta = 0): 49,960 pairs (99.9%)

### Per-Query Summary
- Mean load-bearing candidates per query: 0.06
- Mean harmful candidates per query: 0.02
- Queries with at least one load-bearing candidate: 5/500
- Queries with at least one harmful candidate: 3/500

### Global Candidate Summary
- Distinct load-bearing candidates (mean delta > 0.001): 28
- Distinct harmful candidates (mean delta < -0.001): 10
- Distinct redundant candidates (|mean delta| < 0.0001): 2430

### Stage Breakdown
                    mean  count  sum
stage                               
csls_survivor   0.000000   1947    0
final_survivor  0.001455  12371   18
nstp_kill       0.000000    129    0
rerank_drop     0.000000  35553    0

### Per-Class Concentration
- Mean top-10 share of positive deltas: 0.028
- Classes with top-10 share > 0.8: 3/120
- Classes with zero positive deltas: 116/120

## Top 20 Load-Bearing Candidates

 candidate_slot  candidate_class  n_queries  mean_delta  sum_positive  mean_rank     stage_mode
             63                0         46    0.021739             1  69.826087    rerank_drop
            424                4         83    0.012048             1  56.361446    rerank_drop
            435                4        100    0.010000             1  48.590000    rerank_drop
            205                2        144    0.006944             1  45.291667    rerank_drop
            208                2        145    0.006897             1  27.000000    rerank_drop
            223                2        145    0.006897             1  51.006897    rerank_drop
            240                2        145    0.006897             1  54.296552    rerank_drop
            255                2        145    0.006897             1  41.937931    rerank_drop
            204                2        146    0.006849             1   9.349315 final_survivor
            210                2        146    0.006849             1  35.568493    rerank_drop
            213                2        146    0.006849             1   9.178082 final_survivor
            231                2        146    0.006849             1  24.390411 final_survivor
            235                2        146    0.006849             1  18.280822 final_survivor
            241                2        146    0.006849             1  25.643836 final_survivor
            253                2        146    0.006849             1  22.308219 final_survivor
            212                2        147    0.006803             1  15.258503 final_survivor
            220                2        147    0.006803             1  30.598639    rerank_drop
            239                2        147    0.006803             1  16.183673 final_survivor
            200                2        148    0.006757             1  11.614865 final_survivor
            209                2        148    0.006757             1  19.006757 final_survivor

## Top 20 Harmful Candidates

 candidate_slot  candidate_class  n_queries  mean_delta  sum_negative  mean_rank     stage_mode
            348                3         48   -0.020833             1  40.708333    rerank_drop
            276                3         52   -0.019231             1  51.769231 final_survivor
            345                3         52   -0.019231             1  43.365385 final_survivor
            310                3         58   -0.017241             1  60.413793    rerank_drop
            323                3         58   -0.017241             1  38.344828    rerank_drop
            273                3         61   -0.016393             1  66.868852    rerank_drop
            355                3         67   -0.014925             1  62.149254    rerank_drop
            183                1         82   -0.012195             1  58.158537 final_survivor
           4574               53         85   -0.011765             1  63.470588    rerank_drop
            352                3        101   -0.009901             1  69.148515    rerank_drop
              0                0         47    0.000000             0  37.382979    rerank_drop
              1                0         51    0.000000             0  21.764706 final_survivor
              2                0         48    0.000000             0  23.583333 final_survivor
              3                0         47    0.000000             0  71.765957    rerank_drop
              4                0         49    0.000000             0  40.204082    rerank_drop
              5                0         47    0.000000             0  75.957447    rerank_drop
              6                0         45    0.000000             0  73.422222    rerank_drop
              7                0         48    0.000000             0  78.854167    rerank_drop
              8                0         49    0.000000             0  62.959184    rerank_drop
              9                0         49    0.000000             0  43.367347    rerank_drop

## Plots
- `g36_influence_histogram.png` — Distribution of delta values
- `g36_influence_by_rank.png` — Mean influence by rank bucket
- `g36_class_concentration.png` — Per-class positive influence concentration

## Sanity Checks
- Full accuracy on 500 sample queries: 91.60%
- Each occluded run recomputes Mahalanobis → CSLS → NSTP → floor → softmax from scratch
- No caching of downstream results between occlusion runs
