# A1 · does the #85/#86 blend / false-collapse structure transfer to the ViT-L/14 vision manifold?

**Analysis only.** No new retrieval intervention; `forward()`, the live-Δ floor,
the vote, and `associative_core` are untouched (the 14 pinning tests in
`tests/test_calibration_probe.py`, `tests/test_probe_telemetry.py`,
`tests/test_heldout_abstention.py` still pass bit-for-bit). The **only** change is
the embedding stream: the 20-Newsgroups text source is swapped for a cached
**DINOv2 ViT-L/14** feature manifold (CIFAR-100, 1024-d). Top-1 / vote correctness
is used **only** as an evaluation label, never to steer retrieval.

> **Verdict (one line):** The #86 result is **not text-specific** — it transfers
> to a *second, non-text* manifold. The scope of "general" here is exactly **two
> demonstrated points** (20NG/MiniLM text and CIFAR-100/DINOv2 ViT-L/14 vision),
> **not** all vision and **not** all embedding spaces; it is a transfer existence
> proof, not a sweep (see "Methodological caveats"). On that second manifold every
> load-bearing finding of the text forced-zone study
> reproduces — *all errors are forced*, the
> forced-zone failure signature is *false collapse* (a **sharp** top-1 over
> **thin** support, voting off-class), `rank_gap` **flips sign** in the forced
> zone, and a two-axis **rank_gap + manifold_support** detector **fit on train
> epochs survives held-out epoch evaluation**: forced-zone held-out AUC ≈
> **0.804** (parity) / **0.799** (contiguous) for the deployed `vote_correct`
> target — *at or above* the text numbers (0.748 / 0.740). Abstention on probable
> false collapse is **operationally useful but not production-ready**, and
> **sharpening stays hazardous** because the support axis carries correctness
> signal on the vision manifold too.

## What was run

```bash
# 1. per-probe dump on the vision manifold (both vigilance policies, 30 epochs).
#    The contraction schedule is shifted up (--contraction-start 0.35) so the
#    DINOv2 collapse onset lands near e8, aligning the operational window with the
#    text study's e8–e18 / e8–e29; the unshifted (start 0.0) run shows the same
#    structure, just later (see "Methodological caveats").
python benchmarks/calibration_probe.py --vision \
    --epochs 30 --contraction-start 0.35 --contraction-end 0.9 \
    --held-out-per-class 64 --vision-classes 0,8,19,33 --vision-attractor-class 71 \
    --out results/issue_vitl14_blend_confidence/per_probe.csv
# 2. SAME analysis scripts as the text study, unchanged, pointed at the vision CSV
python benchmarks/analyze_calibration.py --csv results/issue_vitl14_blend_confidence/per_probe.csv
python benchmarks/heldout_abstention.py  --csv results/issue_vitl14_blend_confidence/per_probe.csv --e-lo 8 --e-hi 29
```

The new `VisionDriftStream` (in `benchmarks/probe_contraction.py`) is the vision
analogue of `RealDriftStream`: it loads cached DINOv2 ViT-L/14 features, takes K
well-separated CIFAR-100 classes + one attractor class, and at contraction `c`
convex-blends every (unit) sample feature toward a **deterministically assigned
real attractor-class feature**, renormalizing. Drift therefore *emerges in feature
space* from real anisotropic endpoints (see caveats for the linear-path vs
re-embed distinction). Train/held pools are disjoint per class.

**Manifold:** CIFAR-100 · DINOv2 ViT-L/14 (1024-d) · classes `{0,8,19,33}` +
attractor `71` · live-Δ retrieval floor (#74) · epochs **e8–e29** · both vigilance
policies. 14,645 per-probe rows; **5,364** in the mid-window, **5,083** forced
rows over 22 post-onset epochs. Resting (e0) state is healthy — top-1 acc 0.98,
ρ≈0.35 — so the collapse is *induced*, not baseline (no left-censoring).

## Finding 1 — all errors are forced (✔ transfers)

Mid-window, by label-aware margin bucket:

| bucket       |   n  | top-1 acc | wrong |
|--------------|-----:|----------:|------:|
| recoverable  | 3652 | **1.000** | 0 |
| borderline   |  426 | **1.000** | 0 |
| forced       | 1286 | **0.241** | **976** |

Identical to text: once the true margin clears `temp`, top-1 is **always**
correct; **100% of errors live in the forced bucket** (`{forced: 976,
borderline: 0, recoverable: 0}`). The operational decision is entirely inside the
collapse zone.

## Finding 2 — the forced-zone signature is *false collapse* (✔ transfers)

Conditional means within the forced bucket (`bucket_conditional_means.csv`):

| forced rows | n | rank_gap | manifold_support | support_breadth | peak_mass | off-class mass |
|-------------|--:|---------:|-----------------:|----------------:|----------:|---------------:|
| **correct** | 310 | 0.0084 | **9.02** | **11.10** | 0.250 | 0.32 |
| **wrong**   | 976 | **0.0594** | 6.07 | 9.61 | **0.500** | **0.67** |

Same direction and ordering as text: a wrong-but-forced recall is **sharper**
(7× the rank_gap, 2× the peak_mass) over **thinner** support (lower
manifold_support, fewer survivors) and votes off-class; the correct-but-forced
recall keeps **broad** support. The absolute support magnitudes are larger than
text (~9–11 vs ~4–7) because DINOv2 cosine geometry is differently scaled — the
*signature*, not the scale, is what transfers.

## Finding 3 — `rank_gap` flips sign (✔ transfers, even harder)

`rank_gap` is **not** monotone in correctness (`auc_table.csv`, margin policy):

| slice                | rank_gap AUC (vote) | orientation |
|----------------------|--------------------:|-------------|
| pooled (all rows)    | 0.301 | higher → **WRONG** (−1) |
| **forced zone only** | **0.145** | higher → **WRONG** (−1), separation 0.355 |

The flip is *cleaner* than text (where pooled weakly favored correct, 0.576):
because vision's recoverable + borderline buckets are **perfectly** correct, all
`rank_gap` variance lives in the collapse zone, where a sharp top-1 is the
false-collapse spike. **No single global signed threshold on `rank_gap` works.**

## Finding 4 — within the forced zone, a two-axis rule separates collapse (✔ transfers)

`within_zone_separation.csv` (forced bucket only):

| model (forced zone)                       | AUC top-1 | AUC vote |
|-------------------------------------------|----------:|---------:|
| single: rank_gap                          | 0.141 (inv) | 0.145 (inv) |
| single: peak_mass                         | 0.203 (inv) | 0.110 (inv) |
| single: support_breadth                   | 0.572     | 0.723    |
| single: manifold_support (≡ dispersion)   | 0.689     | 0.839    |
| **two-axis: rank_gap + manifold_support** | **0.860** | **0.889**|
| three: + support_breadth                  | 0.869     | 0.889    |

Combining the **rank** and **support** axes beats either alone — forced-zone AUC
≈ **0.86 / 0.89**, *higher* than the text in-sample numbers (0.79 / 0.83). The
third axis again adds almost nothing (support_breadth and manifold_support are
near-collinear).

## Finding 5 — support is *band-pass*, not monotone (✔ transfers)

`support_bandpass.csv` — `n_surviving_votes` decile → empirical accuracy:

| window | low support | peak | saturated-high support |
|--------|------------:|-----:|-----------------------:|
| **mid (e8–18)**  | 0.69 (d0) | 0.93 (d7) | 0.88 (d9) — ~monotone |
| **late (e19–29)**| 0.34 (d0) | 0.55 (d7) | **0.32 (d9)**, off-class 0.69 — **band-pass** |

Same two-tailed late-window failure as text: low support = false collapse, *and*
saturated-high support with high off-class mass = unresolved blend. A deployable
rule must reject **both** tails.

## Finding 6 — selective prediction (✔ transfers)

`selective_risk.csv` — abstain on the lowest-`support_breadth` rows (vote target):

| coverage kept | vote acc | errors caught in abstained tail |
|---------------|---------:|--------------------------------:|
| 100% (base)   | 0.874    | —    |
| 60%           | 0.937    | 70%  |
| **50%**       | **0.948**| **79%** |
| 20%           | 0.972    | 96%  |

Abstaining the lowest-support ~40–50% removes ~80% of the errors and lifts
retained accuracy — the same curve shape as text.

## The decisive test — held-out two-axis abstention (the #86 question)

Detector = **RANK confidence** (`rank_gap`) + **SUPPORT stability**
(`manifold_support`), standardization, logistic coefficients, single-axis
orientation, and abstention threshold **all fit on train epochs only**, then frozen
and scored on held-out epochs inside the forced zone. Two epoch splits:
**parity** (train even / test odd — charitable) and **contiguous** (train earliest
60% / test latest 40% — adversarial, the floor-readout stress test).

### Held-out AUC (`heldout_auc.csv`, target = `vote_correct`)

| model | parity train → **held-out** | contiguous train → **held-out** |
|-------|----------------------------:|--------------------------------:|
| baseline: `rank_gap` only          | 0.790 → 0.781 | 0.854 → 0.760 |
| baseline: `support_breadth` only   | 0.558 → 0.557 | 0.678 → 0.568 |
| baseline: `manifold_support` only  | 0.743 → 0.744 | 0.820 → 0.744 |
| **main: `rank_gap` + `manifold_support`** | **0.811 → 0.804** | **0.880 → 0.799** |
| diag: + `manifold_support²`        | 0.816 → 0.809 | 0.888 → 0.794 |

Two axes beat either alone on held-out in every split. The biggest train→held-out
shrinkage is on the contiguous split (0.880 → 0.799) — exactly the floor-readout
shrinkage that split is built to expose — but the signal still clears **0.80**, so
it **survives the adversarial split**, not just the charitable one. The quadratic
band-pass term adds ≤ +0.01 held-out → kept as a diagnostic, not adopted.

### Operating point (`heldout_operating_point.csv`, main model, train-selected Youden-J)

| split | retained coverage | retained acc (base ≈ 0.30) | error capture | false-abstain (correct) |
|-------|------------------:|---------------------------:|--------------:|------------------------:|
| **parity**     | 0.55 | **0.48** | **0.60** | 0.10 |
| **contiguous** | 0.59 | **0.37** | 0.52 | 0.07 |

Both lift retained accuracy well above the no-abstention base (0.29) and capture
far more errors than the correct rows they sacrifice (positive held-out Youden J).
`heldout_coverage_sweep.csv` gives the full coverage/accuracy tradeoff.

## Text ↔ vision, side by side

| quantity | text (20NG · MiniLM-L6, 384-d) | **vision (CIFAR-100 · DINOv2 ViT-L/14, 1024-d)** |
|----------|-------------------------------:|-------------------------------------------------:|
| all errors forced? | yes (100%) | **yes (100%)** |
| forced wrong: sharper over thinner support? | yes | **yes** |
| `rank_gap` sign-flip in forced zone? | yes (pooled 0.576 → forced 0.228) | **yes (pooled 0.30 → forced 0.145)** |
| forced-zone two-axis AUC (in-sample, vote) | 0.825 | **0.889** |
| held-out two-axis AUC, parity / contiguous (vote) | 0.748 / 0.740 | **0.804 / 0.799** |
| held-out parity operating point (cov / ret-acc / capture) | 0.59 / 0.44 / 0.53 | **0.55 / 0.48 / 0.60** |
| support band-pass in late window? | yes | **yes** |

The vision manifold does not merely echo the text result — it reproduces the full
two-axis structure and, if anything, the collapse detector is **cleaner** there
(perfectly correct recoverable/borderline buckets, sharper rank_gap inversion,
higher held-out AUC).

## Methodological caveats (read before trusting the magnitudes)

1. **Linear feature-space drift vs nonlinear re-embed.** Text drift is produced at
   the *input* level (swap attractor sentences, then re-embed), so the trajectory
   is the model's own nonlinear manifold. A precomputed feature cache has no cheap
   re-embed, so vision drift is a **convex blend in feature space** between two
   *real* DINOv2 endpoints (a sample and an assigned real attractor feature). Both
   endpoints carry real anisotropy and real intra-/inter-class spread; only the
   **path** between them is linear. This is the faithful cache-only analogue; the
   clean next faithfulness check is **input-level re-embedding** (image/patch
   mixup re-encoded through DINOv2). The engine, probe, floor and vote are
   byte-identical to the text run — only the stream differs.
2. **Single dataset / encoder / class set.** One manifold (CIFAR-100), one encoder
   (DINOv2 ViT-L/14), one 4-class + attractor selection (`0,8,19,33` / `71`). This
   is a *transfer existence proof*, not a sweep. ImageNet-R features
   (`feature_cache_inr_vitl14`, 200 classes) are cached and ready for a breadth
   follow-up.
3. **Schedule shift is cosmetic.** `--contraction-start 0.35` moves the collapse
   onset to ≈ e8 so the operational window matches the text study; the unshifted
   run (start 0.0) shows the **same** structure with the onset at ≈ e17 (margin) /
   e22 (relative). It changes *which epochs* are forced, not the forced-zone
   structure.
4. **Same floor-readout caveat as text.** `support_breadth` is partly a readout of
   the per-epoch live-Δ floor; the contiguous split is the direct test of that
   worry and the signal survives it (held-out 0.799).

## What this does and does not license

* **Licenses** the narrower conclusion that the #86 collapse detector is **not an
  artifact of text embedding geometry specifically** — it also holds on the
  CIFAR-100 DINOv2 ViT-L/14 manifold. It does **not** license a claim over all
  vision encoders, datasets, or embedding spaces. Within that scope, the honest
  mechanism on a collapsing manifold is the same as text — *report uncertainty
  under collapse*, read-only, never changing the elected class of a retained row,
  and still a **diagnostic, not a production-ready gate**.
* **Does not license sharpening/truncation.** Support breadth is **part of the
  correctness signal on vision too** (forced correct manifold_support 9.02 vs wrong
  6.07; orientation +1). The #84 truncation lever attacks exactly that axis, so it
  would push correct-but-forced rows toward the false-collapse signature — a
  confident-error hazard. Per the standing decision, truncation is **not** reopened
  here; it may only be revisited as a *controlled ablation against support-breadth
  preservation*, which this read-only study does not perform.

## Files

| file | what |
|------|------|
| `per_probe.csv` | 14,645 vision per-probe rows: label-free predictors + evaluation labels + epoch context |
| `auc_table.csv` | directed AUC + separation + data-driven orientation per (policy, target, bucket, predictor) |
| `bucket_conditional_means.csv` | predictor means by (margin_bucket × correctness) — the false-collapse mechanism |
| `within_zone_separation.csv` | forced-zone single-feature AUC + two-axis logistic AUC |
| `support_bandpass.csv` | `n_surviving_votes` decile → accuracy + off-class mass, mid vs late window |
| `selective_risk.csv` | accuracy + error-recall vs coverage (correctly oriented) |
| `reliability.csv` | decile reliability per label-free predictor |
| `calibration_summary.json` | machine-readable verdict + headline numbers |
| `heldout_auc.csv` | held-out forced-zone AUC per (split, target, model), train vs held-out |
| `heldout_operating_point.csv` | held-out abstention metrics at the train-selected (Youden-J) threshold |
| `heldout_coverage_sweep.csv` | held-out coverage/accuracy tradeoff at train-selected retained-accuracy targets |
| `heldout_abstention_summary.json` | machine-readable held-out verdict + operating point |
