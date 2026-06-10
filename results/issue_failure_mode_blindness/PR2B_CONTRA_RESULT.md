# PR-2b — cache-backed CONTRADICTORY arm result (gentoo)

First empirical result of the failure-mode blindness study: the PR-2a
injection/labeling machinery run against the verified
`vitl14_cifar100_train` cache. Clean arm = negative control.

## Protocol

| | |
|---|---|
| host / cache | gentoo, `feature_cache_vitl14/cifar100_dinov2_train.pt` (sha256 `7734…b5b3`, manifest_gentoo.json, 16/16 ok) |
| stream | `VisionDriftStream` (exact #87 A1 stream), classes {0, 8, 19, 33}, attractor 71, 32 train / 64 held per class, dim 1024 |
| contraction | **fixed 0.0 (stationary)** — deliberate: the calibration sequence already characterizes confidence under drift (BLENDED); this arm isolates contradiction on an otherwise-healthy manifold |
| memory | margin `DynamicVigilance` + live-Δ `RetrievalFloorPolicy`, `max_entries=4096`, `track_provenance=True` — #87 configuration, no retrieval change, no detector refit |
| arms | `clean` (rate 0) and `contra` (rate 0.15 → 19 injections/epoch), 12 epochs, seed 0 |
| commands | `python benchmarks/failure_mode_probe.py --vision --arm clean --epochs 12 --out results/issue_failure_mode_blindness/per_probe_vision_clean.csv` ; same with `--arm contra --rate 0.15 --out …per_probe_vision_contra.csv` ; then `python benchmarks/analyze_failure_modes.py …contra.csv …clean.csv --json-out …/contra_analysis.json` |
| gates | 20/20 hermetic tests (PR-2a 15 + PR-2b vision 5) passing on gentoo before the run; analyzer output byte-identical re-run on darwin |

## Headline numbers

**Negative control (clean):** 2 532 probes, **0 wrong**, 0 contradictory/stale
flags, accuracy flat at 0.980 for all 12 epochs. The manifold itself produces
no failures; everything below is injection-caused.

**Injections:** 228/228 classified `forked` — on a real, well-separated
manifold every hallucinated re-write of a seen key cleared vigilance with a
disagreeing payload and was silently allocated as a co-resident prototype.
Zero absorbed, zero plain-miss, zero dropped.

**Contra arm:** 2 822 probes.

| quantity | value |
|---|---|
| wrong rate | 264 / 2 822 = **9.4 %** (acc 0.973 → 0.719 as forks accumulate — clean dose-response, +19 fork slots/epoch) |
| CONTRADICTORY_STRICT | 161 (**61.0 %** of wrong) |
| CONTRADICTORY_LENIENT | 255 (**96.6 %** of wrong) |
| `contra_vote_weight` on contra-wrong probes | median **0.667** (p10 0.50, p90 0.75) — fork mass is the majority of the wrong vote, not incidental exposure |
| failure modes | 161 C_STRICT, 94 C_LENIENT, 9 BLENDED, 0 STALE, 0 OTHER_WRONG |

## Confidence behavior on contradictory failures

Medians, contra-wrong (n=255) vs correct (n=2 558); AUC is rank-AUC of the
signal as a risk score with its a-priori orientation (higher = flags failure):

| signal | correct | contra-wrong | AUC (contra-wrong vs correct) | confidently-wrong rate |
|---|---|---|---|---|
| `vote_entropy` | 2.37 | 1.92 | **0.356** (inverted) | **0.710** |
| `effective_support` | 10.7 | 6.9 | **0.356** (inverted) | **0.710** |
| `n_surviving_votes` | 20 | 8 | **0.305** (inverted) | 0.643 |
| `max_vote_weight` | 0.220 | 0.201 | 0.531 (chance) | 0.424 |
| `top1_top2_margin` | 0.0 | 0.0 | 0.728 | 0.028 |
| `top1_sim` | 0.646 | 0.506 | 0.740 | 0.216 |

Confidently-wrong rate = share of contra-wrong probes more confident than the
**median correct probe** under that signal (SCHEMA.md definition).

## Interpretation

1. **Contradiction is confident wrong recall on the collapse axes.** The
   entropy/support family — the signals the #85/#86 detector geometry is
   built around, where failure = diffuse, broad, high-entropy votes — is
   *anti-predictive* here (AUC 0.31–0.36). A contradictory failure retrieves a
   tight cluster of co-resident forks at high similarity: **fewer** surviving
   votes (8 vs 20), **lower** entropy (1.92 vs 2.37), **less** support than a
   correct retrieval. 71 % of contradictory failures are more confident than
   the median correct probe by entropy/support. This is the VIGIL prediction
   (session 732dc595) realized in data: the blindness is structural, not a
   threshold problem.
2. **Partial visibility on the similarity axis, not the vote axis.**
   `top1_sim` (0.740) and `top1_top2_margin` (0.728) carry some signal —
   forks sit slightly off the clean prototype and compete with it — but at
   AUC ≈ 0.73 with heavily overlapping distributions, and `top1_sim` is the
   axis the held-out study (#86) found drift-fragile. Nothing here approaches
   the ≥ 0.80 two-axis performance on BLENDED.
3. **The labels are causally loaded, not just exposure.** Median fork vote
   mass on contra-wrong probes is 2/3, strict (leading-support) covers 61 %
   of wrong answers, and the clean arm fires zero flags — so the wrong
   answers are attributable to the injected forks, not background noise.
4. **Pre-registered trigger status** (SCHEMA.md, stated before data): the
   *formal* trigger is the frozen #87 two-axis detector score (PR-3), but
   the per-signal evidence — confidently-wrong 71 % ≫ 50 % on the entropy
   axis, inverted AUC on the entire collapse family — points squarely at
   **write-time contradiction governance**: the fork is mechanically known at
   write time (`pre-write sim ≥ vigilance ∧ payload cosine ≤ 0.5` — exactly
   the condition the registry classifies), while at read time it is
   indistinguishable from a confident correct recall on the deployed
   confidence axes. Retrieval-time confidence-gating with the existing
   signals would *preferentially keep* these failures.

## Next (not done here, per scope guardrails)

- PR-2c: stale arm (supersession) on the same cache.
- PR-3: score the frozen #87 two-axis detector per failure mode against the
  pre-registered triggers; dose-response analysis over `contra_vote_weight`.
- Open question this run cannot answer: whether contradiction under
  *concurrent drift* stays inverted or migrates into the BLENDED signature
  (the 9 BLENDED rows hint at a fork-becomes-blend channel).

## Files

- `per_probe_vision_clean.csv` / `.summary.json` — negative control (2 532 rows)
- `per_probe_vision_contra.csv` / `.summary.json` — contradiction arm (2 822 rows)
- `contra_analysis.json` — analyzer output (verified identical darwin/gentoo)
