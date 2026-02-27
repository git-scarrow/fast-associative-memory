Loading dogs features...
Loading adapter...
Collecting traces...
    Prototypes: 10279
Eval contamination fix active: Yes (Verified in associative_core.py)
Adaptive Eviction: True, Use LFU: True

# G37 Offline Negative Evidence Analysis

## Aggregate Stats
| Quartile | Pop Size | Test A Sig (p<0.01) | Test B Mismatch | Class Cov (Mean/Med/Range) | Cramer's V (Mean) |
|---|---|---|---|---|---|
| Q1 (Bottom) | 5145 | 36.7% | 33.2% | 4.3 / 3.0 / [1-21] | 0.524 |
| Q2 | 5145 | 31.5% | 20.3% | 4.7 / 3.0 / [1-27] | 0.592 |
| Q3 | 5145 | 27.0% | 14.9% | 4.8 / 3.0 / [1-27] | 0.609 |
| Q4 | 5145 | 15.6% | 6.4% | 4.1 / 2.0 / [1-26] | 0.593 |

**Kill Condition Check**: On bottom-quartile-by-margin, rejected candidates' class distribution must show significant asymmetry (p<0.01) vs accepted set.
**Result**: PASS. 36.7% of border queries show significant asymmetry (>> 1% null expectation). Dark matter is NON-EPIPHENOMENAL.

## Qualitative Examples (Bottom Quartile)
### Example 1
- **Ground Truth**: Class 0
- **Standard Prediction**: Class 0
- **Asymmetry p-value**: 2.39e-06
- **Survivor Vote**: {110: 2, 0: 23}
- **Rejected Distribution**: {86: 24, 110: 6, 72: 2, 44: 14, 7: 3, 102: 2, 0: 15, 94: 1}
- **Would adjusted prediction change?**: No (Adj Pred: 0)

### Example 2
- **Ground Truth**: Class 0
- **Standard Prediction**: Class 0
- **Asymmetry p-value**: 2.54e-15
- **Survivor Vote**: {35: 2, 0: 23}
- **Rejected Distribution**: {42: 13, 111: 6, 35: 40, 34: 2, 76: 1}
- **Would adjusted prediction change?**: No (Adj Pred: 0)

### Example 3
- **Ground Truth**: Class 2
- **Standard Prediction**: Class 2
- **Asymmetry p-value**: 6.54e-10
- **Survivor Vote**: {2: 25}
- **Rejected Distribution**: {114: 4, 113: 50, 2: 18}
- **Would adjusted prediction change?**: No (Adj Pred: 2)

### Example 4
- **Ground Truth**: Class 2
- **Standard Prediction**: Class 2
- **Asymmetry p-value**: 1.15e-06
- **Survivor Vote**: {2: 25}
- **Rejected Distribution**: {49: 42, 2: 26, 93: 1, 53: 3}
- **Would adjusted prediction change?**: No (Adj Pred: 2)

### Example 5
- **Ground Truth**: Class 2
- **Standard Prediction**: Class 2
- **Asymmetry p-value**: 4.28e-06
- **Survivor Vote**: {2: 25}
- **Rejected Distribution**: {53: 8, 50: 2, 100: 25, 45: 1, 36: 4, 107: 1, 2: 16, 49: 6, 1: 1, 4: 1}
- **Would adjusted prediction change?**: No (Adj Pred: 2)

