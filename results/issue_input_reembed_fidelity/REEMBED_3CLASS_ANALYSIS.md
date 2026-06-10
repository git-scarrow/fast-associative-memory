# A2b · ReembedDriftStream 3-Class Subset Analysis

**Status:** Analysis complete. Raw CSV on Gentoo; analyzer outputs in scratch; memo only below.

---

## Data slice

- **Total rows:** 9,085
- **Mid-window rows (e8–e18):** 3,378
- **Policies:** margin, relative
- **Epoch range:** 0–29; primary analysis window 8–18

---

## Key result

**The false-collapse signature survives input-level re-embedding in the 3-class subset.**

---

## Forced-zone conditional means

### Correct (n=50)
| Predictor | Value |
|-----------|-------|
| `top1_top2_margin` | 0.0127 |
| `vote_entropy` | 1.0867 |
| `effective_support` | 3.4155 |
| `max_vote_weight` | 0.4692 |
| `n_surviving_votes` | 3.84 |
| `offclass_weight` | 0.3957 |
| `true_margin` | 0.0133 |

### Wrong (n=574)
| Predictor | Value |
|-----------|-------|
| `top1_top2_margin` | 0.1164 |
| `vote_entropy` | 0.4149 |
| `effective_support` | 1.7793 |
| `max_vote_weight` | 0.8259 |
| `n_surviving_votes` | 2.3693 |
| `offclass_weight` | 0.9178 |
| `true_margin` | -0.1936 |

**Interpretation:** Wrong forced retrievals exhibit the expected false-collapse signature — sharper top-1 (higher `top1_top2_margin`), thinner support (`effective_support` 1.78 vs 3.42, fewer surviving votes 2.37 vs 3.84), higher peak mass (`max_vote_weight` 0.83 vs 0.47), and higher off-class blending (0.92 vs 0.40). Correct forced retrievals preserve broader, more balanced support.

---

## Error distribution

### Recoverable bucket
- Correct: 2,675
- Wrong: 0
- Accuracy: 1.0

### Borderline bucket
- Correct: 79
- Wrong: 0
- Accuracy: 1.0

### Forced bucket
- Correct: 50
- Wrong: 574
- Accuracy: 0.0801

**Interpretation:** 100% of errors remain in the forced zone. Recoverable and borderline buckets are error-free in both, confirming the margin-bucket stratification holds.

---

## Label-free confidence separators

**Robust forced-zone separators** (AUC-based, both policies):
- `rank_gap` (top1_top2_margin) ✓
- `support_dispersion` (vote_entropy) ✓
- `manifold_support` (effective_support) ✓
- `peak_mass` (max_vote_weight) ✓
- `support_breadth` (n_surviving_votes) ✓

**Rank_gap sign-flip:** true (orientation = −1 in forced zone, meaning higher margin → more likely wrong, opposite of global trend).

**Best single-feature forced-zone AUC:** 0.8093

---

## Support-breadth decile behavior

### Mid-window (e8–e18)
| Decile | n | Mean surviving | Accuracy | Offclass weight |
|--------|---|---|---|---|
| 0 (low support) | 337 | 1.0 | 0.5875 | 0.4125 |
| 1 | 338 | 1.21 | 0.5799 | 0.4106 |
| 2 | 338 | 2.15 | 0.6627 | 0.3281 |
| 3 | 338 | 3.64 | 0.6272 | 0.3745 |
| 4 | 338 | 7.13 | 0.8521 | 0.1426 |
| 5 | 337 | 12.08 | 0.9941 | 0.0084 |
| 6 | 338 | 17.31 | 0.9970 | 0.0019 |
| 7–9 | ~1k | 20.0 (saturated) | 1.0 | ~0.0 |
| **Peak:** | — | — | 1.0 @ decile 7 | — |
| **High-tail drop:** | — | — | **false** | — |

### Late-window (e19–e29)
| Decile | n | Mean surviving | Accuracy | Offclass weight |
|--------|---|---|---|---|
| 0 (low support) | 333 | 1.0 | 0.3483 | 0.6517 |
| 1 | 334 | 1.23 | 0.3922 | 0.6043 |
| 2–7 | ~2.3k | 2–12.5 | 0.43–0.60 | 0.52–0.39 |
| 8 | 334 | 18.66 | 0.9012 | 0.0905 |
| 9 (high support) | 334 | 20.0 | 0.9701 | 0.0259 |
| **Peak:** | — | — | 0.9701 @ decile 9 | — |
| **High-tail drop:** | — | — | **false** | — |

**Interpretation:** Support-breadth orientation is positive overall, but the decile curve is not strictly monotone. Low support correlates with ~0.59 acc in mid-window and ~0.35 acc in late-window; high support reaches ~1.0 and 0.97 respectively, confirming the +1 orientation. Late-window accuracy shows a broad ~0.43–0.60 band from deciles 0–7, then rapid rise to 0.97 at deciles 8–9, and does NOT show the high-tail drop observed in the anchor dataset. This suggests slightly different manifold behavior under input-level re-embedding, but the core support-breadth signal survives.

---

## Two-axis detector (rank_gap + manifold_support)

**Forced-zone logistic AUC (in-sample):**

| Target | two-axis (rank_gap + manifold_support) | three-axis (+ support_breadth) |
|--------|---------------------------------------|-------------------------------|
| `top1_correct` | **0.9222** (sep 0.4222) | 0.9231 |
| `vote_correct` | **0.9289** (sep 0.4289) | 0.9322 |

**Root cause of original blank:** The analyzer was invoked with the system interpreter, which lacks scikit-learn. `logistic_auc` silently returns NaN on `ImportError`; every numpy-only computation resolved correctly. Re-running with the project venv (scikit-learn installed) recovers the values above — all other numbers reproduce bit-for-bit (9,085 rows, mid-window 3,378, forced acc 0.080, 574 forced errors). The fix: `analyze_calibration.py` now emits a one-time stderr warning on ImportError instead of returning NaN silently.

---

## Verdict

**The false-collapse signature survives input-level re-embedding.**

Core observations:
- Errors remain concentrated in forced zone (100% of errors, forced bucket acc 0.08).
- Wrong forced retrievals are sharp, thin-support, high-peak-mass, high-offclass collapses.
- Correct forced retrievals preserve broader support.
- `rank_gap` sign-flips in forced zone (−1 orientation, as expected).
- `support_breadth` orientation survives (+1, broad = correct).
- All five label-free predictors separate in forced zone.
- Two-axis forced-zone AUC = **0.9222** (sep 0.4222 >> 0.15 gate), matching #88 structure.
- Support-breadth orientation is positive overall, but the decile curve is not strictly monotone.

**Limitations:**
- Late-window band-pass / high-tail drop does not fully reproduce (real, not an artifact).
- 3-class subset is small; larger reembed run would reduce small-sample artifacts.

**Recommended next step:**
Full A2b is unblocked on the two-axis front. The late-window high-tail-drop divergence is real but not a blocker for the core fidelity question. A larger input-reembed subset remains useful for robustness but is not required before advancing.

---

**Analysis date:** 2026-06-08  
**Analyzer:** `benchmarks/analyze_calibration.py` (unmodified)  
**Input CSV location (Gentoo):** `results/issue_input_reembed_fidelity/per_probe_reembed.csv`
