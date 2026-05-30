# #82 calibration study — does label-free retrieval confidence predict top-1 correctness?

**Analysis only.** No new retrieval intervention; `forward()` is untouched; top-1
correctness is used **only as an evaluation label**. This increment follows the
#84 truncation null (a qualified null: sharpening the vote is accuracy-neutral
and cannot excise off-class mass without eroding accuracy) and the #84 review
note that reframed the next step as *a calibration study, not a gate*.

> **Verdict (one line):** A label-free signal **does** separate correct from
> wrong retrievals — but "confidence" is **not one scalar**. The recoverable and
> borderline buckets are already ~100% correct; **100% of errors are forced**
> (manifold collapse). Inside that forced zone the failure signature is **false
> collapse** — a *sharp* top-1 over *thin* support, voting off-class — so
> `rank_gap` (top1−top2 margin) **flips sign** there and no single global
> threshold works. A two-axis **rank_gap + manifold_support** rule reaches
> forced-zone AUC ≈ **0.79** (top-1) / **0.82** (vote). This supports
> confidence-gated **abstention / "report uncertainty under collapse"**, and
> warns **against** retrieval *sharpening* — sharpening attacks the very support
> breadth that distinguishes a correct-but-forced recall.

## How to reproduce

```bash
# 1. per-probe dump: real-20NG, live-Δ floor, both vigilance policies, 30 epochs
python benchmarks/calibration_probe.py \
    --epochs 30 --contraction-end 0.9 --held-out-per-class 64 \
    --out results/issue82_retrieval_confidence_calibration/per_probe.csv
# 2. analysis (bucket-stratified, data-driven orientation, two-axis rule)
python benchmarks/analyze_calibration.py
```

The per-probe telemetry is emitted by the **opt-in, read-only**
`ContinuousCAM.probe_cross_class_similarity(return_per_probe=True)` (default off →
every existing aggregate is bit-identical; verified in
`tests/test_calibration_probe.py`, which also pins `vote_pred_label` to
`forward().argmax` row-for-row).

## Primary slice

real-20NG · live-Δ retrieval floor (#74) · epochs **e8–e18** · both vigilance
policies (`margin` = DynamicVigilance, `relative` = RelativeVigilance). The two
policies run over the **same** embedding stream, so they differ only in write-side
prototype shaping. 13,341 per-probe rows total; **4,835** in the mid-window.

## Concept map — two axes, not one scalar

| telemetry column   | concept            | reading                                              |
|--------------------|--------------------|------------------------------------------------------|
| `top1_top2_margin` | **rank_gap**       | is top-1 clearly ahead of top-2?                     |
| `n_surviving_votes`| **support_breadth**| how many candidates survive the floor                |
| `effective_support`| **manifold_support**| `exp(vote_entropy)` — breadth of surviving neighbourhood |
| `vote_entropy`     | support_dispersion | monotone with manifold_support                       |
| `max_vote_weight`  | **peak_mass**      | how peaky the single top vote is                     |

All are **label-free** (computable at inference). The textbook intuition
"low entropy / sharp peak = confident = correct" is **inverted** on the support
axis here: a healthy associative recall keeps a *broad* surviving neighbourhood;
a wrong recall has often collapsed into a *sparse, sharp, off-class* basin.

## Finding 1 — all errors are forced; the recoverable bucket needs no rescue

Mid-window, by label-aware margin bucket (`true_margin` vs `inference_temp`):

| bucket       |   n  | top-1 acc | wrong |
|--------------|-----:|----------:|------:|
| recoverable (margin ≥ temp)        | 3391 | **1.000** | 0 |
| borderline  (½temp ≤ margin < temp)|  347 | **1.000** | 0 |
| forced      (margin < ½temp)       | 1097 | **0.248** | **825** |

Once the true margin clears `temp`, top-1 is **always** correct. There are **no
recoverable-but-wrong rows to rescue**. The operational decision lives entirely
in the **forced** zone — so a pooled, all-rows AUC can look meaningful while
hiding that it only matters where the manifold has collapsed.

## Finding 2 — the forced-zone failure signature is *false collapse*

Conditional means **within the forced bucket** (`bucket_conditional_means.csv`):

| forced rows   |  n  | rank_gap | manifold_support | support_breadth | peak_mass | off-class mass |
|---------------|----:|---------:|-----------------:|----------------:|----------:|---------------:|
| **correct**   | 272 |   0.011  | **5.73**         | **6.78**        |  0.365    | 0.36           |
| **wrong**     | 825 | **0.038**|   3.64           |   4.42          | **0.582** | **0.82**       |

A wrong-but-forced recall is **sharper** (higher rank_gap, higher peak_mass) over
**thinner** support (lower manifold_support, fewer survivors), and it votes
off-class. The correct-but-forced recall keeps broad support. This is exactly
"the system became *wrong by becoming sharper*" — sharpness is **false-collapse
confidence**, not correctness.

## Finding 3 — `rank_gap` flips sign across buckets

`rank_gap` is **not monotone** in correctness:

| slice                | `rank_gap` AUC | orientation     |
|----------------------|---------------:|-----------------|
| pooled (all rows)    | 0.576          | higher → correct |
| **forced zone only** | **0.228**      | **higher → WRONG** (separation 0.272) |

Globally, more rank separation means a healthy (recoverable) row; inside the
collapse zone, more rank separation means a confident off-class spike. **No single
global signed threshold on `rank_gap` can work** — which is also why its pooled
AUC is weak (the two regimes partly cancel) and why high pooled AUC for the
support axis came with a small naive selective-lift before stratifying.

## Finding 4 — within the forced zone, a *two-axis* rule separates collapse from healthy recall

`within_zone_separation.csv` (forced bucket only):

| model (forced zone)                         | AUC top-1 | AUC vote |
|---------------------------------------------|----------:|---------:|
| single: rank_gap                            | 0.228 (inv) | 0.238 (inv) |
| single: peak_mass                           | 0.265 (inv) | 0.206 (inv) |
| single: support_breadth                     | 0.657     | 0.733    |
| single: manifold_support (≡ dispersion)     | 0.690     | 0.764    |
| **two-axis: rank_gap + manifold_support**   | **0.788** | **0.825**|
| three: + support_breadth                    | 0.793     | 0.825    |

Combining the **rank** and **support** axes (logistic, orientation learned)
clearly beats either alone: forced-zone AUC ≈ 0.79 / 0.82. The third axis adds
little (support_breadth and manifold_support are near-collinear; `effective_support
= exp(vote_entropy)`, so dispersion and manifold_support are rank-identical).

## Finding 5 — support is *band-pass*, not monotone (late window)

`support_bandpass.csv` — `n_surviving_votes` decile → empirical accuracy:

| window | low support | peak | saturated-high support |
|--------|------------:|-----:|-----------------------:|
| **mid (e8–18)**  | 0.62 (d0) | 0.99 (d8) | 0.97 (d9) — ~monotone |
| **late (e19–29)**| 0.50 (d0) | 0.65 (d7) | **0.41 (d9)**, off-class 0.64 — **band-pass** |

In the mid-window support is ~monotone (the blend tail hasn't formed). In the
late window there are **two** failure modes: low support = **false collapse**, and
*saturated* high support with high off-class mass = **unresolved blend** (the
manifold mushed; all candidates clear the floor but they're off-class). A
deployable rule must reject **both** tails — a band-pass on support, not a single
signed threshold.

## Finding 6 — selective prediction (mid-window, correct orientation)

`selective_risk.csv` — abstain on the lowest-`support_breadth` rows:

| coverage kept | top-1 acc | vote acc | errors caught in abstained tail |
|---------------|----------:|---------:|--------------------------------:|
| 100% (base)   | 0.829     | 0.839    | —    |
| 60%           | 0.907     | 0.925    | 67%  |
| **50%**       | **0.924** | **0.943**| **78% / 82%** |
| 20%           | 0.982     | 0.996    | —    |

Abstaining the lowest-support ~40–50% of mid-window retrievals removes ~80% of
the errors and lifts retained accuracy from 0.83 to ~0.92–0.94. (A two-axis rule
does better still in the forced sub-zone; this single-axis curve is the
conservative lower bound.)

## Decision

The decision rule from the task was: *if label-free confidence separates
correctness → next PR can implement confidence-gated retrieval or abstention; if
not → report uncertainty / abstain under collapse rather than sharpen.*

Both arms **converge**: a label-free signal separates correctness, **and** the
honest mechanism is abstention. Concretely, the next increment should:

1. **Abstain / flag, do not sharpen.** Gate on a **two-axis collapse detector**
   (low `manifold_support` *or* `support_breadth` despite high `rank_gap` /
   `peak_mass`; plus the saturated-high-support + high-off-class blend tail) —
   not a single signed threshold, and **never** `rank_gap` globally (it flips).
2. **Treat sharpening as a hazard.** The #84 truncation lever reduces support
   breadth; applied naively it pushes *correct-but-forced* rows toward the
   false-collapse signature. Any future use of it must be guarded by the support
   axis, or it will convert recoverable recalls into confident errors.
3. **Scope honestly.** `support_breadth` is partly a readout of the live-Δ floor's
   action; the forced-zone two-axis AUC (~0.79) is in-sample (orientation and the
   logistic fit are not held out across a calibration split). A clean follow-up
   would fit the two-axis rule on a held-out epoch split and report the
   abstention operating point (e.g. coverage at a target retained-accuracy).

## Files

| file | what |
|------|------|
| `per_probe.csv` | 13,341 per-probe rows: label-free predictors + evaluation labels + epoch context |
| `auc_table.csv` | directed AUC + separation + data-driven orientation, per (policy, target, **bucket**, predictor) |
| `bucket_conditional_means.csv` | predictor means by (margin_bucket × correctness) — the false-collapse mechanism |
| `within_zone_separation.csv` | forced-zone single-feature AUC + two-axis logistic AUC |
| `support_bandpass.csv` | `n_surviving_votes` decile → accuracy + off-class mass, mid vs late window |
| `selective_risk.csv` | accuracy + error-recall vs coverage (correctly oriented) |
| `reliability.csv` | decile reliability per label-free predictor |
| `calibration_summary.json` | machine-readable verdict + headline numbers |
| `heldout_auc.csv` | **held-out** forced-zone AUC, per (split, target, model), train vs held-out |
| `heldout_operating_point.csv` | held-out abstention metrics at the train-selected (Youden-J) threshold |
| `heldout_coverage_sweep.csv` | held-out coverage/accuracy tradeoff at train-selected retained-accuracy targets |
| `heldout_abstention_summary.json` | machine-readable held-out verdict + operating point |

---

# Held-out two-axis abstention — does the #85 number survive? (PR follow-up)

The §Decision point 3 above flagged the open risk: the forced-zone two-axis AUC
(~0.79) was **in-sample** — orientation and the logistic fit were not held out
across a calibration split, and `support_breadth` is partly a readout of the
per-epoch live-Δ floor. This section closes that out.

> **Held-out verdict (one line):** The two-axis collapse detector — **RANK
> confidence** (`rank_gap`) + **SUPPORT stability** (`manifold_support`), fit on
> train epochs only — **survives held-out epoch evaluation**: forced-zone
> AUC ≈ **0.75** (parity split) / **0.74** (contiguous split) for the deployed
> `vote_correct` target, vs a no-abstention base accuracy of **0.305**. At a
> train-selected operating point it **lifts retained accuracy to ~0.44 while
> capturing ~53% of errors** and wrongly abstaining only ~14% of correct rows.
> Abstention on probable **false collapse** is **operationally useful but not
> production-ready**; **sharpening stays hazardous** because the support axis
> carries correctness signal.

## "Confidence" = rank confidence × support stability

The detector makes the #85 two-axis story concrete and **label-free**. Fit on
*train epochs only* inside the forced zone, the single-axis orientations are:

| axis | telemetry | learned forced-zone orientation | reading |
|------|-----------|-------------------------------|---------|
| **rank confidence** | `rank_gap` (top1−top2 margin) | higher → **WRONG** (−1) | a *sharp* top-1 over thin support is **false collapse** |
| **support stability** | `manifold_support` (`exp(vote_entropy)`) | higher → **CORRECT** (+1) | a *broad* surviving neighbourhood is a healthy recall |

The two axes pull in opposite directions, so no single global threshold works —
they must be combined. That is the whole point: low entropy / few survivors is
**not** confidence here, it is often false collapse.

## Held-out protocol (the guardrails)

* **Operational zone only.** Recoverable + borderline buckets are ~100% correct
  (nothing to gate); the detector is fit and evaluated **entirely inside the
  forced bucket**, post-onset window e8–e29.
* **Split by epoch, never by row.** Rows in one epoch share that epoch's floor
  scalar, so a by-row split would leak the floor context. Two epoch splits are
  reported:
  * **parity** (primary, charitable) — train = even epochs, test = odd. Isolates
    "does the *signal* generalise to unseen rows?" while controlling for the
    drift-regime shift (a neighbour epoch's floor is similar).
  * **contiguous** (adversarial) — train = earliest 60% of epochs, test = latest
    40%. The held-out floor regime is genuinely unseen; this is the direct test
    of the floor-readout worry.
* **No leakage.** Standardisation means/stds, logistic coefficients, the
  single-axis orientation, *and* the abstention threshold are all fit on train
  epochs and then frozen before held-out is scored. The feature set is fixed a
  priori from the #85 in-sample study, not chosen on the held-out data.
* **No labels at inference.** The detector score is a function of label-free
  predictors only; correctness is used solely to fit/evaluate.
* **No retrieval intervention.** Abstention is a pure post-hoc mask — it never
  changes the elected class of a retained row. `forward()` and `associative_core`
  are untouched (pinned by `tests/test_heldout_abstention.py`).

## Held-out AUC (`heldout_auc.csv`, target = `vote_correct`)

| model | parity train → **held-out** | contiguous train → **held-out** |
|-------|----------------------------:|--------------------------------:|
| baseline: `rank_gap` only | 0.736 → 0.720 | 0.768 → 0.705 |
| baseline: `support_breadth` only | 0.640 → 0.625 | 0.728 → 0.607 |
| baseline: `manifold_support` only | 0.689 → 0.677 | 0.761 → 0.680 |
| **main: `rank_gap` + `manifold_support`** | **0.768 → 0.748** | **0.827 → 0.740** |
| diag: + `manifold_support²` | 0.772 → 0.751 | 0.828 → 0.741 |

Two axes beat either alone on held-out in every split. The quadratic band-pass
term adds only ~+0.003 held-out — **kept as a diagnostic, not adopted**: the
simple two-axis logistic is the auditable model. The biggest train→held-out gap
is on the contiguous split (0.827 → 0.740), exactly the floor-readout shrinkage
the split was designed to expose — but the signal still clears 0.74, so it
**survives** the adversarial split, not just the charitable one.

## Operating point (`heldout_operating_point.csv`, main model, train-selected Youden-J)

A fixed retained-accuracy *target* is the wrong knob in a zone whose base
accuracy is 0.305 — targeting 0.80 collapses coverage to <1% (see
`heldout_coverage_sweep.csv`). So the headline threshold maximises Youden's J
(error_capture − false_abstain) **on train**, then is applied to held-out:

| split | retained coverage | retained acc (base 0.305) | error capture | false-abstain (correct) |
|-------|------------------:|--------------------------:|--------------:|------------------------:|
| **parity** | 0.59 | **0.44** | **0.53** | 0.14 |
| **contiguous** | 0.69 | **0.41** | 0.41 | 0.07 |

Both lift retained accuracy well above the no-abstention base and capture far
more errors than the correct rows they sacrifice (positive held-out Youden J).
The `heldout_coverage_sweep.csv` curve shows the full coverage/accuracy tradeoff
for callers who want a different operating point.

## What this does and does not license

* **Licenses** a confidence-gated **abstention / "report uncertainty under
  collapse"** layer on the forward prediction, driven by the two-axis detector.
  It is read-only and does not touch retrieval selection.
* **Does not license sharpening/truncation.** Support breadth is *part of the
  correctness signal* (orientation +1). The #84 truncation lever attacks it, so
  it pushes correct-but-forced rows toward the false-collapse signature — a
  confident-error hazard. The held-out result reinforces, not relaxes, this.
* **Validity scope.** The claim is only as strong as held-out epoch performance,
  which is reported above for both a charitable and an adversarial split. If a
  future stream's drift regime departs from training, re-fit and re-check the
  contiguous split before trusting the score.

## Reproduce

```bash
# 1. (if needed) regenerate the per-probe dump
python benchmarks/calibration_probe.py \
    --epochs 30 --contraction-end 0.9 --held-out-per-class 64 \
    --out results/issue82_retrieval_confidence_calibration/per_probe.csv
# 2. held-out two-axis abstention calibration (auto-regenerates the dump if absent)
python benchmarks/heldout_abstention.py --e-lo 8 --e-hi 29
```
