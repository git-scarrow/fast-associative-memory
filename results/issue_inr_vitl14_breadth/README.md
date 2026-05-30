# A2 · ViT-L/14 false-collapse breadth — does the #87 result survive ImageNet-R?

**Analysis only.** No new retrieval intervention; `forward()`, the live-Δ floor,
the vote, and `associative_core` are untouched (the same 14 pinning tests that
guard #87 still pass bit-for-bit; this branch adds **no** code beyond what #87
merged — it only points the existing `--vision-cache` flag at a different cache).
Top-1 / vote correctness is used **only** as an evaluation label.

This is a **breadth / domain-shift check, not a fidelity fix.** #87 showed the
text false-collapse structure transfers to a CIFAR-100 DINOv2 ViT-L/14 manifold;
A2 asks whether it *also* survives a **broader, more distribution-shifted** visual
manifold — **ImageNet-R** (200 ImageNet classes rendered as art, cartoons,
sculptures, etc.), same DINOv2 ViT-L/14 encoder. It still uses **cache-level
linear drift** (a convex blend between real feature endpoints). It does **not**
address the input-level re-embedding caveat from #87 — that remains a separate,
still-open fidelity check.

> **Verdict (one line):** **All four signatures survive ImageNet-R.** Across a
> harder, more distribution-shifted manifold (forced-zone base accuracy **0.175**
> vs CIFAR's 0.305), the structure holds: *100% of errors are forced*, the
> forced-zone failure signature is *false collapse* (sharp top-1 over thin
> support, off-class), **support breadth keeps positive orientation** (broad =
> correct), `rank_gap` **flips sign** in the forced zone, and the two-axis
> `rank_gap + manifold_support` detector **fit on train epochs survives held-out
> epoch evaluation** — forced-zone held-out AUC ≈ **0.838** (parity) / **0.831**
> (contiguous) for `vote_correct`, *above* the CIFAR numbers (0.804 / 0.799) even
> though the zone is harder. Scope remains **two visual manifolds + one text**,
> one encoder — a breadth confirmation, not a universal claim, and still a
> **diagnostic, not a production-ready gate**.

## What was run

```bash
# Same #87 machinery; only --vision-cache + class ids differ. Both vigilance
# policies, 30 epochs. Contraction schedule is IDENTICAL to #87 (see "Dataset
# calibration" — no retune was needed).
python benchmarks/calibration_probe.py --vision \
    --vision-cache feature_cache_inr_vitl14/imagenetr_dinov2_train.pt \
    --vision-classes 166,63,77,156 --vision-attractor-class 134 \
    --epochs 30 --contraction-start 0.35 --contraction-end 0.9 \
    --held-out-per-class 64 --out results/issue_inr_vitl14_breadth/per_probe.csv
# SAME analysis scripts, unchanged, pointed at the ImageNet-R CSV
python benchmarks/analyze_calibration.py --csv results/issue_inr_vitl14_breadth/per_probe.csv
python benchmarks/heldout_abstention.py  --csv results/issue_inr_vitl14_breadth/per_probe.csv --e-lo 8 --e-hi 29
```

**Manifold:** ImageNet-R · DINOv2 ViT-L/14 (1024-d) · classes `{166,63,77,156}` +
attractor `134` · live-Δ floor (#74) · epochs **e8–e29** · both policies. 12,539
per-probe rows; **3,520** forced rows over the held-out window.

### Class selection (data-driven, reported for reproducibility)

ImageNet-R has 200 classes with **uneven** per-class counts (40–344, median 109).
Of the 118 classes with ≥ 96 samples (the `samples_per_class 32 + held_out 64`
requirement), four were chosen by **greedy maximum mutual separation** of unit
class centroids, and the attractor was the eligible class with the **highest mean
similarity** to those four (a natural central puller). Resting pairwise cosines
among the four are **≤ 0.056** — at least as well separated as the CIFAR-100 set.
Selection used only the feature geometry, never correctness labels.

## Dataset calibration (explicit: this is *not* detector optimization)

The #87 (CIFAR-100) schedule was `--contraction-start 0.35 --contraction-end 0.9`.
A **smoke sweep** on ImageNet-R at that exact setting produced a **comparable
forced-zone** — overall forced fraction ≈ 0.30, forced-bucket accuracy ≈ 0.11,
recoverable/borderline intact, gradual onset — i.e. **neither too few forced rows
nor near-total collapse**. So the schedule was **kept identical to #87; no
dataset-specific retune was applied.** The only difference is the collapse onset
lands ≈ e10 (ImageNet-R) vs ≈ e8 (CIFAR-100), reflecting ImageNet-R's slightly
greater class separation under the same blend. This paragraph documents the
schedule decision as **dataset calibration of the analysis window**, not tuning of
any detector parameter — the detector's feature set, orientation, standardization,
logistic fit, and threshold are all learned downstream from train epochs only,
exactly as in #87.

## The four signatures (ImageNet-R vs CIFAR-100)

### 1 · all errors forced / forced-zone concentration — ✔ survives

| bucket (ImageNet-R, mid-window) | n | top-1 acc | wrong |
|---------------------------------|--:|----------:|------:|
| recoverable | 3654 | **1.000** | 0 |
| borderline  |  122 | **1.000** | 0 |
| forced      |  874 | **0.137** | **754** |

`errors by bucket = {forced: 754, borderline: 0, recoverable: 0}` — **100% of
errors are forced**, exactly as on CIFAR-100 and text.

### 2 · false-collapse signature — ✔ survives (sharper)

Forced-bucket conditional means (`bucket_conditional_means.csv`):

| forced rows | n | rank_gap | manifold_support | support_breadth | peak_mass | off-class mass |
|-------------|--:|---------:|-----------------:|----------------:|----------:|---------------:|
| **correct** | 120 | 0.0111 | **5.24** | **6.45** | 0.373 | 0.31 |
| **wrong**   | 754 | **0.0788** | 2.99 | 4.00 | **0.703** | **0.84** |

Same direction as CIFAR/text, with an *even starker* contrast: wrong-but-forced
recalls are **7× sharper** (rank_gap) and nearly **2× peakier** over **~half** the
support, voting **84%** off-class.

### 3 · support-breadth orientation stays positive — ✔ survives

`within_zone_separation.csv` (forced zone): `manifold_support` orientation **+1**
(AUC 0.769 top-1 / **0.845** vote), `support_breadth` orientation **+1** (0.717 /
0.790). **Broad support = correct** still holds — so sharpening/truncation remains
hazardous on this manifold too.

### 4 · rank_gap sign-flip + held-out two-axis AUC — ✔ survives

`rank_gap` (margin policy, `auc_table.csv` / `within_zone_separation.csv`):

| slice                | rank_gap AUC (vote) | orientation |
|----------------------|--------------------:|-------------|
| pooled (all rows)    | 0.382 | higher → **WRONG** (−1) |
| **forced zone only** | **0.139** | higher → **WRONG** (−1), separation 0.362 |

Same inversion as CIFAR (where the perfectly-correct recoverable/borderline
buckets push the pooled orientation negative too). Forced-zone single vs two-axis:

| model (forced zone)                       | AUC top-1 | AUC vote |
|-------------------------------------------|----------:|---------:|
| single: rank_gap                          | 0.149 (inv) | 0.139 (inv) |
| single: manifold_support                  | 0.769     | 0.845    |
| **two-axis: rank_gap + manifold_support** | **0.851** | **0.861**|
| three: + support_breadth                  | 0.855     | 0.865    |

**Held-out** (`heldout_auc.csv`, target = `vote_correct`, detector fit on train
epochs only):

| model | parity train → **held-out** | contiguous train → **held-out** |
|-------|----------------------------:|--------------------------------:|
| baseline: `rank_gap` only          | 0.844 → 0.832 | 0.865 → 0.824 |
| baseline: `manifold_support` only  | 0.798 → 0.776 | 0.831 → 0.769 |
| **main: `rank_gap` + `manifold_support`** | **0.850 → 0.838** | **0.865 → 0.831** |
| diag: + `manifold_support²`        | 0.853 → 0.841 | 0.890 → 0.828 |

Two axes beat either alone on held-out in both splits; the signal clears **0.83**
even on the adversarial contiguous split. Late-window `support_breadth` is again
**band-pass** (`support_bandpass.csv`: low d0 0.37, peak d8 0.58, saturated-high
d9 0.50 with 0.54 off-class — both tails wrong).

### Held-out operating point (`heldout_operating_point.csv`, main, train Youden-J)

| split | retained coverage | retained acc (base ≈ 0.17) | error capture | false-abstain (correct) |
|-------|------------------:|---------------------------:|--------------:|------------------------:|
| **parity**     | 0.42 | **0.38** | **0.69** | 0.05 |
| **contiguous** | 0.44 | **0.34** | 0.66 | 0.03 |

Both more than **double** retained accuracy over the no-abstention base and capture
~⅔ of errors while wrongly abstaining < 5% of correct rows.

## CIFAR-100 ↔ ImageNet-R, side by side (both DINOv2 ViT-L/14, 1024-d)

| quantity | CIFAR-100 (#87) | **ImageNet-R (A2)** |
|----------|----------------:|--------------------:|
| forced-zone base acc (vote) | 0.305 | **0.175** (harder, more shifted) |
| 100% of errors forced? | yes | **yes** |
| forced-wrong sharper over thinner support? | yes | **yes (starker)** |
| support-breadth orientation | **+1** | **+1** |
| `rank_gap` forced-zone orientation | −1 (flip) | **−1 (flip)** |
| forced-zone two-axis AUC (in-sample, vote) | 0.889 | 0.861 |
| **held-out two-axis AUC parity / contiguous (vote)** | 0.804 / 0.799 | **0.838 / 0.831** |
| held-out parity op point (cov / ret-acc / capture) | 0.55 / 0.48 / 0.60 | 0.42 / 0.38 / 0.69 |
| late-window support band-pass? | yes | **yes** |

The forced zone on ImageNet-R is **harder** (lower base accuracy, more errors), yet
the two-axis detector separates correct-from-wrong **at least as well on held-out**
— in-sample AUC is marginally lower but the train→held-out gap is *smaller*, so the
deployed held-out number is actually higher. Domain shift does not break the
signature; if anything it sharpens the false-collapse contrast.

## Scope and caveats (unchanged from #87 + the breadth boundary)

1. **Breadth, not fidelity.** A2 widens the manifold (200-class, art/rendition
   domain shift) but uses the **same cache-level linear convex drift** between real
   DINOv2 endpoints as #87. It does **not** address input-level (pixel/patch)
   re-embedding — that remains the open fidelity check, untouched by this PR.
2. **Scope of the claim.** The collapse detector now holds on **three manifolds**
   — 20NG/MiniLM text and two DINOv2 ViT-L/14 vision manifolds (CIFAR-100,
   ImageNet-R). This is a **breadth confirmation across one encoder**, not a claim
   over all encoders, datasets, or embedding spaces.
3. **Diagnostic, not production.** Held-out abstention is operationally useful but
   **not a production-ready gate**; the support axis carries correctness signal, so
   sharpening/truncation stays hazardous and is **not** reopened here.
4. **Same floor-readout caveat.** `support_breadth` is partly a readout of the
   per-epoch live-Δ floor; the adversarial contiguous split is the direct test and
   the signal survives it (held-out 0.831).

## Files

Same layout as `results/issue_vitl14_blend_confidence/`: `per_probe.csv`,
`auc_table.csv`, `bucket_conditional_means.csv`, `within_zone_separation.csv`,
`support_bandpass.csv`, `selective_risk.csv`, `reliability.csv`,
`calibration_summary.json`, `heldout_auc.csv`, `heldout_operating_point.csv`,
`heldout_coverage_sweep.csv`, `heldout_abstention_summary.json`.
