# Issue #82 — Intervention 1: adaptive vote-support truncation

First #82 intervention (follows the #83 instrumentation increment). Adds a
retrieval-time `RetrievalTruncationPolicy` and evaluates it against the #83
live-Δ baseline on real 20-Newsgroups, using the #83 telemetry as the harness.

**This is an intervention-only increment, evaluated honestly. The headline is a
qualified-null:** adaptive support truncation removes the *premature
effective-support inflation* #83 identified, but it is **accuracy-neutral, not
accuracy-improving**, and it cannot excise off-class vote mass without
proportionally eroding accuracy. The reason is structural and was predicted by
#83's `frac_blend_top1_correct ≈ 0.49`.

## The lever

`RetrievalTruncationPolicy` (`dynamic_vigilance.py`) applies a per-probe,
*relative-margin* cut to the top-k candidate cosines **before** the softmax vote:

```
keep[j]  iff  sim[j] >= top1 - window_steps · inference_temp
```

- **Adaptive / self-gating.** The cut is relative to *each probe's own top-1*,
  so the survivor count tracks local geometry. A separated within-class core
  above an off-class tail gets the tail sliced; a co-located (forced)
  neighbourhood has nothing beyond the window, so the cut is a near no-op — the
  intervention leaves the collapsed tail alone without needing labels.
- **Top-1 always survives** → the cut can only sharpen toward the local winner.
- **Composes** with the #74 live-Δ floor (runs on the floor survivors) and is
  **off by default** (no policy attached → retrieval bit-identical to baseline,
  verified: the margin/relative baselines below reproduce #83 to the digit).
- Optional `gate_steps` (default 0) restricts the cut to isolated-winner rows;
  left off here because a top1−top2 gate closes on within-class *siblings* —
  exactly the recoverable rows we want to act on.

```
python benchmarks/probe_contraction.py --real --epochs 30 --contraction-end 0.9 \
    --retrieval-floor-policy live-delta --vigilance-policy {margin,relative} \
    --support-truncation --truncation-window-steps 2.0 --truncation-gate-steps 0.0
```

## Result vs the #83 live-Δ baseline (window=2.0, gate=0)

Recoverable mid-window, mean over e8–e18 (B = #83 baseline, I = intervention):

| metric | margin B | margin I | Δ | relative B | relative I | Δ |
|---|---|---|---|---|---|---|
| `effective_support` | 5.32 | 4.02 | **−1.30** | 4.81 | 3.61 | **−1.20** |
| `vote_entropy` | 1.36 | 1.11 | −0.26 | 1.30 | 1.03 | −0.27 |
| `max_vote_weight` | 0.508 | 0.550 | +0.042 | 0.527 | 0.571 | +0.044 |
| `offclass_weight` | 0.224 | 0.218 | −0.006 | 0.209 | 0.205 | −0.005 |
| `frac_blended` | 0.386 | 0.368 | −0.018 | 0.341 | 0.332 | −0.009 |
| `acc` | 0.805 | 0.803 | −0.002 | 0.787 | 0.787 | 0.000 |

Forced tail (e29) and whole-run accuracy:

| metric | margin B | margin I | relative B | relative I |
|---|---|---|---|---|
| `effective_support` (e29) | 9.05 | 7.71 | 12.98 | 9.60 |
| `offclass_weight` (e29) | 0.654 | 0.656 | 0.680 | 0.676 |
| `acc` (e29) | 0.375 | 0.375 | 0.414 | 0.391 |
| **mean acc, all epochs** | 0.7547 | 0.7529 | 0.7453 | 0.7435 |

**What the lever does:** it cleanly deflates the diffuse-vote signal —
`effective_support` drops ~1.2–1.3 in the mid-window and the vote sharpens
(`max_vote_weight ↑`, `vote_entropy ↓`). The premature support inflation #83
flagged (~4→6 across the mid-window) is exactly what gets reabsorbed.

**What it does not do:** it barely moves net off-class vote mass (−0.006) and it
does **not** improve accuracy (−0.002, within determinism noise). The forced tail
is untouched, as designed — no claim to fix e29.

## Window scan (margin gate) — the accuracy ↔ blend-reduction tradeoff

| window | mid `offclass_weight` | mid `effective_support` | mid `acc` | mean acc (all) | e29 `acc` |
|---|---|---|---|---|---|
| baseline | 0.224 | 5.32 | 0.805 | 0.7547 | 0.375 |
| 2.0 | 0.218 | 4.02 | 0.803 | 0.7529 | 0.375 |
| 1.5 | 0.214 | 3.31 | 0.805 | 0.7484 | 0.359 |
| 1.0 | 0.209 | 2.49 | 0.803 | 0.7474 | 0.359 |
| 0.5 | 0.199 | 1.65 | 0.802 | 0.7437 | 0.367 |

Tightening the window buys *more* off-class-mass reduction (0.224→0.199) and
much harder support deflation (5.32→1.65), but accuracy slides monotonically
(0.7547→0.7437). There is **no window that reduces blend and improves accuracy
simultaneously.** `window=2.0` is shipped as the default because it is the
accuracy-neutral end of this curve while still reabsorbing the premature support
inflation; CSVs at 0.5/1.0/1.5 are in `window_scan/` for full reproducibility.

## Why truncation alone cannot recover the mid-window

#83 already measured the ceiling: in the mid-window `frac_blend_top1_correct ≈
0.49` — of the probes that blend, **only about half have a correct top-1**. The
off-class mass is not a separable tail sitting below a clean within-class core;
on the real all-MiniLM manifold the within- and off-class neighbours are
interleaved in one narrow similarity band (the within–off gap is ~0.09 ≈
1.8·temp). A similarity-relative window therefore slices within- and off-class
neighbours in near-equal proportion:

- For the ~half of blended probes whose top-1 *is* correct, sharpening helps.
- For the ~half whose top-1 is an off-class neighbour, sharpening toward that
  top-1 **hurts** — and truncation, which always keeps the top-1, makes the
  confident-wrong vote *more* confident.

The two effects cancel: net accuracy is a wash. This is the honest boundary of a
pure retrieval-time sharpening lever, and it is consistent with #83's own caveat
that `frac_blend_top1_correct` — not `frac_blend − frac_margin_forced` — is the
true recoverability signal.

## Verdict → what this changes for the next increment

1. **Adaptive support truncation is sound infrastructure but a weak lever here.**
   It is correct, tested (off-by-default parity, self-gating in the forced
   regime, stage-3 trace accounting), and composes with the floor — keep it.
   But on this manifold it deflates support without improving correctness, so it
   ships off by default and makes no accuracy claim.
2. **The open question is a *calibration* question, not another gate.** Whether
   truncation (or any selective sharpening) could ever pay off hinges on a prior
   unknown: **can label-free confidence telemetry predict top-1 correctness in
   the recoverable mid-window?** Note `top1_top2_margin` alone does not (it is
   ~0.045–0.051 in *both* the recoverable and forced regimes) — but it has not
   been tested jointly with the other label-free signals. The next increment is
   therefore a **calibration study**, not an intervention: regress top-1
   correctness on `top1_top2_margin`, `vote_entropy`, `effective_support`,
   `max_vote_weight`, and `n_surviving_votes`, using top-1 correctness **only as
   the evaluation label** — never as an inference-time gate. Gating on true
   correctness would be an oracle and is explicitly out of scope. Only if a
   label-free predictor is shown to be calibrated does a selective lever become
   worth building.
3. **The forced tail still wants abstention, not recovery** (#83's second
   lever), unchanged by this result.

## Files

- `real_{margin,relative}_livedelta_baseline.{csv,png,_verdict.json}` — #83
  baseline, reproduced with this branch's code (off-by-default parity check).
- `real_{margin,relative}_livedelta_truncation.{csv,png,_verdict.json}` —
  intervention, window=2.0 / gate=0. Verdict JSONs record the exact levers.
- `window_scan/real_margin_w{0.5,1.0,1.5}.csv` — the window sweep above.
