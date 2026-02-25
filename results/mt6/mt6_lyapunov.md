# MT-6: Write-Path Lyapunov Exponent

## Context

MT-4 showed full batch reversal produces orthogonal stores (cos ~0.005) with a
reported 47pp accuracy gap. MT-5 proved the divergence is entirely in prototype
positions, not Mahalanobis calibration (class_vars identical, sim delta ~2.5e-6).

MT-6 measures how divergence scales with perturbation size to determine if the
write path is **chaotic** (positive Lyapunov exponent) or **controllable**
(sublinear/linear). Instead of binary forward-vs-reverse, we apply k adjacent-pair
swaps to the batch order and measure divergence as a function of k.

## Config
- **Commit**: `7728f92`
- **Seed**: 42 (perturbation seed: 12345)
- **Device**: cuda
- **Dataset**: Stanford Dogs 1024-d, production adapter, MAX_ENTRIES=8000
- **Reference S1 accuracy**: 35.00%
- **Perturbation**: k adjacent-pair swaps applied to forward batch order
- **Levels**: k ∈ {0, 1, 2, 5, 10, 20, 47} + full_reverse

## Results

| Condition | k | Cos sim | 1−cos (div) | Occ agree | Acc S_k | Acc delta | Evict J |
|---|---|---|---|---|---|---|---|
| swap_0 | 0 | 1.000000 | 0.000000 | 1.0000 | 35.00% | 0.00pp | 1.0000 |
| swap_1 | 1 | 0.770348 | 0.229652 | 1.0000 | 35.00% | 0.00pp | 0.9882 |
| swap_2 | 2 | 0.530863 | 0.469137 | 1.0000 | 35.10% | 0.10pp | 0.9321 |
| swap_5 | 5 | 0.401125 | 0.598875 | 1.0000 | 35.00% | 0.00pp | 0.9922 |
| swap_10 | 10 | 0.235356 | 0.764644 | 1.0000 | 35.00% | 0.00pp | 0.8972 |
| swap_20 | 20 | 0.166907 | 0.833093 | 1.0000 | 35.10% | 0.10pp | 0.8326 |
| swap_47 | 47 | 0.113806 | 0.886194 | 1.0000 | 35.05% | 0.05pp | 0.8219 |
| full_reverse | 47 | 0.004263 | 0.995737 | 1.0000 | 90.85% | 55.85pp | 0.8615 |

## Sanity Check
- k=0 (identical to S1): cos=1.000000, acc_delta=0.00pp **PASS**

## Curve Fits

### Cosine divergence (1 − cos_mean) vs k
- **Power law** `div = 3.10e-01 × k^0.32`: R² = 0.78
- **Exponential** `div = 4.39e-01 × exp(0.019 × k)`: R² = 0.40
- **Best fit**: power law (sublinear)

### Accuracy delta vs k
- Effectively zero across all swap levels (max 0.10pp)
- Full reverse is the only condition with material accuracy shift (+55.85pp)
- Curve fitting on the swap data is meaningless — there is no trend to fit

## Verdict

**MICRO-CHAOTIC, MACRO-STABLE**

The write path exhibits two distinct regimes:

1. **Prototype positions are highly sensitive** — a single adjacent batch swap
   (k=1) produces 23% cosine divergence. Divergence grows sublinearly with k
   (β = 0.32), saturating near orthogonality (~0.89) by k=47. This means the
   exact prototype centroids are *not* reproducible under any ordering variation.

2. **Accuracy is completely insensitive** to local perturbations — across all
   swap levels from k=1 to k=47, accuracy stays within 0.10pp of the reference.
   The decision surface is robust even when the underlying prototypes have
   diverged to near-orthogonal positions.

The write path has a **positive but sublinear Lyapunov exponent** for prototype
positions (β = 0.32 < 1.0, ruling out exponential chaos). However, the downstream
accuracy signal is **zero-Lyapunov** — the system absorbs perturbations entirely.

## The Full-Reverse Outlier

Full reversal is **not** "47 swaps" — it is a qualitatively different perturbation.
The Stanford Dogs training set is class-sorted (classes 1–120 in sequence). Under
forward order with MAX_ENTRIES=8000 and ~12,000 samples, late-arriving classes
evict early-class prototypes via LFU. This creates a systematic bias toward the
last ~67% of classes, yielding 35% accuracy on a balanced test set.

Full reversal inverts the class-arrival order entirely: early classes (by label)
now arrive *last* and dominate the store. This happens to produce 90.85% accuracy
— not because the store is "better," but because the reversed class distribution
happens to align more favorably with the test set's class balance.

Adjacent-pair swaps cannot produce this effect. Even k=47 adjacent swaps barely
perturbs the global class-arrival order — a class that was batch 5 might move to
batch 6, but it won't move to batch 40. The macro-structure of "which classes
dominate the final store" is preserved under local perturbation but shattered
under global reversal.

This explains the discontinuity: swap_47 (cos=0.11, acc_delta=0.05pp) vs
full_reverse (cos=0.004, acc_delta=55.85pp). The two conditions have the same
nominal "k" but completely different perturbation geometry.

## MT-4 Accuracy Discrepancy

MT-4 reported S1 accuracy of **91.80%** vs MT-6's **35.00%** for the same
forward-order pipeline. The cause: **eval contamination**.

MT-4 evaluates between feeding chunks (at N=10 and N=47). `ContinuousCAM.forward()`
updates `self.last_seen[topk_slots[:, 0]] = now` at line 278 of `associative_core.py`
using wall-clock `time.time()`. During eval, this sets `last_seen` to ~1.74×10⁹ for
whichever prototypes match test queries. When subsequent batches are fed, these
test-relevant prototypes have enormously inflated `last_seen` values, making them
immune to LFU-LRU eviction. This selectively protects the prototypes that matter
for the test set, inflating accuracy from the true ~35% to 91.80%.

MT-6 feeds all 47 batches in one shot before any eval, with a deterministic
monkey-patched clock, so it reports the true accuracy.

**Action item**: MT-4's `eval_accuracy` calls should use `torch.no_grad()` with a
temporarily frozen `last_seen` buffer, or eval should only happen after all feeding
is complete. The 47pp "order sensitivity" reported by MT-4 is a composite of two
effects: genuine class-arrival bias (~56pp, as MT-6 shows) and eval contamination
(~36pp spurious inflation of S1's score).

## Single-Swap Sensitivity
- One adjacent batch swap (k=1) produces:
  - Cosine divergence: 0.230 (23% of keyspace has moved)
  - Accuracy delta: 0.00pp (zero functional impact)
  - Eviction Jaccard: 0.988 (98.8% of same slots evicted)

## Implications for Shutter-Deck

The practical question: does inotify arrival jitter in the Shutter-Deck ingestion
loop produce materially different stores?

**Answer: No.** Inotify jitter produces local perturbations equivalent to small-k
adjacent swaps. MT-6 shows these produce zero accuracy impact despite significant
prototype drift. The store's decision surface is robust to the kind of ordering
noise that real deployment introduces.

The only dangerous perturbation is a **systematic reordering of class arrival** —
e.g., if the data source switches from alphabetical to reverse-alphabetical class
ordering between runs. This is a data pipeline concern, not a store concern.

## Plots
- `mt6_divergence_curve.png` — Divergence vs k on log-log axes with power/exp fits
- `mt6_eviction_curve.png` — Eviction Jaccard vs k

## Artifacts
- `mt6_metrics.json` — all metrics and fit parameters
