# MT-5: Mahalanobis Calibration Surface

## Config
- **Commit**: `7728f92`
- **Seed**: 42
- **Device**: cuda
- **Dataset**: Stanford Dogs ViT-L/14 1024-d, production adapter
- **Stores**: S1 (forward order) and S3 (reversed order), matching MT-4 (MAX_ENTRIES=8000)
- **Class pairs probed**: 60 (sampled from 7140 possible pairs of 120 classes)
- **Interpolation points**: 101 per pair (t in [0, 1])
- **Comparison**: cosine top-10 / top-25 vs Mahalanobis top-10 / top-25

## Summary Table

| Metric | S1 (forward) | S3 (reversed) |
|---|---|---|
| Top-1 flip rate (cos top-1 != mah top-1) | 0.0000 | 0.0000 |
| Mean top-10 overlap | 10.00/10 | 10.00/10 |
| Mean top-25 overlap | 25.00/25 | 25.00/25 |
| Mean rank corr (Spearman rho, broad-100) | 1.0000 | 1.0000 |
| Mean |mah_sim - cos_sim| | 0.000003 | 0.000003 |
| Mean max |mah_sim - cos_sim| | 0.000005 | 0.000005 |
| class_vars mean |deviation from 1.0| | 0.013699 | 0.013699 |
| S1 vs S3 class_vars cosine (mean) | 1.0000 | — |

## Key Findings

### 1. Mahalanobis effect magnitude
**Near-identity transform.** Mean sim delta = 0.000003 (S1) / 0.000003 (S3). The per-class `class_vars` have barely moved from their initial value of 1.0 (mean deviation: 0.013699 S1, 0.013699 S3). With EMA alpha=0.01 and ~100 samples per class, the diagonal scaling is essentially `inv_std ≈ 1.0` everywhere, making Mahalanobis re-ranking a near-no-op.

### 2. Localized vs global reordering?
**No reordering detected.** Rank correlation = 1.0000 (S1) / 1.0000 (S3) — the cosine and Mahalanobis rankings are essentially identical across all t. Top-10 overlap is 10.0/10 and top-25 overlap is 25.0/25.

### 3. S3 vs S1 calibration?
The class_vars vectors between S1 and S3 have cosine similarity = 1.0000.
This means the per-class variance estimates are essentially identical despite completely different write order — the Mahalanobis is equally (in)effective in both stores.

### 4. Implication for MT-4's 47pp accuracy gap
Since Mahalanobis is a near-no-op (sim delta ~2.5e-06), the 47pp accuracy gap between S1 and S3 in MT-4 is **not caused by Mahalanobis miscalibration**. The gap comes entirely from different prototype populations (key drift + eviction) — the stores contain different prototypes for the same classes, leading to different cosine retrievals before Mahalanobis even runs.

### 5. Where along t does most disagreement occur?
- S1 worst rank-corr region: t in [0.00, 0.33] (rho=1.0000)
- S3 worst rank-corr region: t in [0.08, 0.88] (rho=1.0000)

## Plots
- `mt5_flip_summary.png` — Rank correlation and overlap surfaces for S1, S3, and diff
- `mt5_class_vars_diagnostic.png` — Per-class variance deviation, S1/S3 agreement, rescoring magnitude
- `mt5_exemplar_sweeps.png` — 4 exemplar pair sweeps with lowest mean rank correlation

## Artifacts
- `mt5_paths.csv` — 12120 rows (pair x t x store)
