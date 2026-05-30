# Issue #82 — Late-epoch retrieval saturation: premature vs forced

Instrument-first increment. Adds retrieval-confidence + effective-support
telemetry to `probe_cross_class_similarity()` and reproduces the #80 live-Δ
floor baseline on real 20-Newsgroups to answer the one question that decides the
next intervention: **is the late-epoch blend recoverable (a usable signal the
soft vote is fumbling) or forced (the manifold has genuinely collapsed)?**

## Setup

```
python benchmarks/probe_contraction.py --real --epochs 30 --contraction-end 0.9 \
    --retrieval-floor-policy live-delta --vigilance-policy {margin,relative}
```

4 well-separated 20NG topics (sci.space, rec.sport.hockey, talk.politics.guns,
comp.graphics), `sci.med` attractor, `all-MiniLM-L6-v2`. Memory persists across
epochs (continual). Matches #80 exactly: blend onset e4 (margin) / e6 (relative),
epoch-0 off-class vote mass 0.088.

## New telemetry

| key | meaning | available at inference? |
|---|---|---|
| `top1_top2_margin` | gap between best two cosine neighbours | yes (label-free) |
| `vote_entropy` / `effective_support` | spread of the softmax vote (`exp(H)` ∈ [1,k]) | yes |
| `max_vote_weight`, `n_surviving_votes` | vote concentration / post-floor support | yes |
| `frac_low_margin`, `frac_high_entropy` | low-confidence flags (normalized entropy ≥ 0.5) | yes |
| `frac_margin_{recoverable,borderline,forced}` | split on **true** `ss−so` margin, temp-relative | no (diagnostic) |
| `frac_blend_top1_correct` | of blended probes, share whose top-1 is the correct class | no (diagnostic) |

`recoverable: ss−so ≥ temp · borderline: 0.5·temp ≤ ss−so < temp · forced: < 0.5·temp` (temp=0.05).

## Result (epoch 29, c=0.90)

| gate | frac_blend | forced | recoverable | blend_top1_correct | Δ−ρ gap | acc |
|---|---|---|---|---|---|---|
| margin   | 0.91 | **0.74** | 0.13 | **0.28** | **−0.023** | 0.38 |
| relative | 0.98 | **0.79** | 0.13 | **0.33** | **−0.020** | 0.41 |

**The late tail is forced.** By epoch 29 the within-class Δ (~0.43) is *below*
off-class ρ (~0.45) — the gap has gone **negative**: off-class neighbours are on
average closer than same-class. ~74–79% of probes have a sub-`0.5·temp` true
margin, and only ~28–33% of *blended* probes still have a correct top-1. No
retrieval-time mechanism (sharper temp, top-k truncation, stronger floor) can
restore correct retrieval here — they would only sharpen onto a wrong winner.
The honest move for the collapsed tail is **confidence reporting / abstention**,
not recovery.

**But there is a recoverable mid-window.** Through ~epoch 16 (c≈0.50) the gap is
still healthy (+0.09), `forced ≤ 0.28`, `recoverable ≥ 0.60`, and `blend_top1_correct
≈ 0.50–0.56` while `frac_blend` has already climbed to 0.40–0.45 and
`effective_support` inflated from ~4 → ~6. In this band the blend runs **ahead**
of the forced fraction — a meaningful slice is *premature*: the soft vote is
diffusing a still-resolvable signal across more prototypes than it should. acc is
still 0.81–0.84 here. This is where **confidence-weighted retrieval / adaptive
support truncation** can suppress blend without harming early accuracy.

The forced fraction overtakes recoverable around epoch 18–20 (c≈0.55–0.62),
exactly where Δ−ρ drops below ~0.05 and held-out accuracy begins its steep
decline.

## Verdict → next increment

Two regimes, two levers:

1. **Mid-window (recoverable, ~e8–e18):** confidence-weighted retrieval / adaptive
   support truncation — sharpen / truncate when the vote is diffuse but top-1 is
   still correct. Buys *delay* of saturation without harming early accuracy.
2. **Late tail (forced, ~e20+):** confidence **reporting / abstention** — `Δ−ρ`
   is negative, signal is gone; the engine should surface low confidence rather
   than emit a confident wrong vote.

A stronger live-Δ floor schedule (candidate from #82) is **not** indicated: the
floor already de-censored onset (#80), and at the tail it would have to mask
within-class neighbours too (Δ < ρ), harming the same-class vote.

Caveat (rigor): the margin-regime bins use fixed `temp`-relative thresholds;
`frac_blend − frac_margin_forced` is a loose upper bound on what is recoverable.
`frac_blend_top1_correct` is the tighter signal — and it says ~half the mid-window
blend and only ~30% of the e29 blend is top-1-recoverable.

## Files

- `real_{margin,relative}_livedelta.csv` — per-epoch telemetry (36 columns)
- `real_{margin,relative}_livedelta_verdict.json` — onset / transfer verdict
- `real_{margin,relative}_livedelta.png` — ρ(t) / off-class-mass plot
