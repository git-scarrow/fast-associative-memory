# PR-2c — cache-backed STALE arm result (gentoo)

Second empirical result of the failure-mode blindness study: the PR-2a
supersession protocol run against the verified `vitl14_cifar100_train`
cache. STALE is measured separately from CONTRADICTORY throughout: the
question here is whether a superseded association remains reachable or
dominant after replacement, not whether competing claims collapse
confidence. The PR-2b clean arm is reused as the negative control.

## Protocol

| | |
|---|---|
| host / cache | gentoo, `feature_cache_vitl14/cifar100_dinov2_train.pt` — cache sanity re-run before use (16/16 ok, A1 class/count requirements met at commit `3ceaa42`) |
| stream | `VisionDriftStream` (exact #87 A1 stream), classes {0, 8, 19, 33}, attractor 71, 32 train / 64 held per class, dim 1024 |
| contraction | **fixed 0.0 (stationary)** — same rationale as PR-2b: isolate the injected mode from drift/BLENDED |
| memory | margin `DynamicVigilance` + live-Δ `RetrievalFloorPolicy`, `max_entries=4096`, `track_provenance=True` — #87 configuration, no retrieval change, no detector refit |
| stale protocol | cached class 0 (remapped label A=0) superseded by cached class 33 (remapped label B=3). Phase 1 (epochs 0–5): K→A through the normal learn path (192 phase-1 writes). Phase 2 (epochs 6–11): the SAME keys K→B; held-out ground truth for A-keys flips to B at epoch 6. Merge-vs-fork is decided by the engine. No contra injections run in this arm. |
| run | 12 epochs, supersede epoch 6, seed 0. `python benchmarks/failure_mode_probe.py --vision --arm stale --epochs 12 --supersede-epoch 6 --out results/issue_failure_mode_blindness/per_probe_vision_stale.csv` ; then `python benchmarks/analyze_failure_modes.py …stale.csv …clean.csv --json-out …/stale_analysis.json` |
| gates | 22/22 hermetic tests (PR-2a 15 + vision 7, incl. stale end-to-end + stale determinism) passing on gentoo before the run; analyzer output byte-identical re-run on darwin |

## Headline numbers

**Stale arm:** 2 192 probes, 60 wrong (2.7 %), **all 60 in a single epoch** —
the supersession boundary. Clean arm (PR-2b control): 0 wrong, 0 flags.

| quantity | value |
|---|---|
| stale wrong recall | **found, transient**: epoch 6 accuracy 0.980 → 0.742, epochs 7–11 back to ≥ 0.984 |
| stale-selection rate at the boundary | **60/60** superseded-key voting probes elect the stale value A; 0 elect the updated B |
| stale-selection rate after the boundary | **0/60** from epoch 7 onward (all 60 elect B) |
| label coverage | STALE_LENIENT covers **100 %** of wrong (60/60), STALE_STRICT 55 % (33/60); all 60 also carry `is_blended=1` |
| overlap with contradiction | `contra ∧ stale` lenient overlap = **0**; no contra protocol runs in this arm and no contra flag fired (pinned by test) |
| supersession path | **fork, never merge**: one-hot class payloads have cosine 0 ≤ 0.5, so every phase-2 write takes the contradiction-fork path. 32 stale slots (one per superseded train key) appear at epoch 6 and remain live through epoch 11 |
| persistent reachability | epochs 7–11: 297/930 voting probes still hold stale slots in their surviving top-k (median 4 slots, median **0.20** vote mass, max 0.35) — all correct. Stale support is permanently co-resident, outvoted but never removed |
| `stale_vote_weight` on stale-wrong probes | **exactly 0.500** on all 60 (p10 = p90 = 0.5) |

## Why the failure window is one epoch (mechanism, verified)

Replicated with instrumented slot counts on the hermetic tiny cache: every
phase-2 epoch forks **again** — the nearest slot to each superseded key
remains (or ties with) the mature A slot, the payload disagrees, and a new
B copy is allocated. So B-slot copies grow by one per key per epoch while
the A slots persist unchanged:

* **At the flip epoch** each key has exactly one A slot and one B fork *at
  an identical key vector* → the vote splits **exactly 0.5 / 0.5** (the
  measured stale vote weight) and `top1_top2_margin = 0`. The elected class
  is then decided by argmax tie-breaking, which selects the lower label
  index — here A (0) over B (3). The observed 100 % stale selection at the
  boundary is therefore the deterministic tie-break direction, **not**
  evidence that stale support is stronger; the protocol-independent fact is
  that the readout assigns *equal* mass to the superseded and current
  associations.
* **From the next epoch** B copies outnumber A 2:1 and rising, so the
  current value dominates — at the cost of one redundant fork per key per
  rewrite epoch (silent write amplification), while stale slots stay
  reachable indefinitely.

A corollary worth stating: the window closed here only because the
protocol re-writes the superseding fact every epoch. A **one-shot**
supersession would leave the retrieval in the 0.5/0.5 tie regime
indefinitely.

## Confidence behavior on stale failures

Medians, stale-wrong (n=60) vs correct (n=2 132); AUC is rank-AUC of the
signal as a risk score with its a-priori orientation (higher = flags
failure). PR-2b contra values shown for contrast:

| signal | correct | stale-wrong | AUC stale | conf-wrong stale | AUC contra (PR-2b) | conf-wrong contra |
|---|---|---|---|---|---|---|
| `vote_entropy` | 1.52 | 2.50 | **0.833** | 0.050 | 0.356 (inv.) | 0.710 |
| `effective_support` | 4.6 | 12.2 | **0.833** | 0.050 | 0.356 (inv.) | 0.710 |
| `n_surviving_votes` | 7 | 20 | **0.866** | 0.050 | 0.305 (inv.) | 0.643 |
| `top1_top2_margin` | 0.026 | 0.0 | **0.930** | 0.000 | 0.728 | 0.028 |
| `max_vote_weight` | 0.451 | 0.188 | 0.818 | 0.050 | 0.531 | 0.424 |
| `top1_sim` | 0.669 | 0.784 | **0.292 (inverted)** | **0.717** | 0.740 | 0.216 |

(The clean-vs-correct baselines differ between the PR-2b and PR-2c tables
because each arm's correct population is its own run's.)

## Interpretation — calibrated

1. **Stale wrong recall: found.** Superseding a mature association does not
   replace it: the old slots survive as permanently co-resident,
   still-decoding stale support (32/32 live through end of run), and at the
   supersession boundary the readout is at an exact stale/current tie, with
   the elected answer decided by tie-breaking. Wrong recall is confined to
   the window between the first and second superseding writes.
2. **Confidence inversion on the collapse axes: NOT found — the signature
   is the opposite of contradiction.** A contradictory failure is sharp and
   narrow (PR-2b: lower entropy/support than correct, collapse-axis AUC
   0.31–0.36, 71 % confidently wrong). A stale failure is broad and
   ambivalent: ~2× the entropy/support/surviving votes of a correct probe,
   zero margin, every stale-wrong probe also blended-flagged. The collapse
   axes flag it at AUC 0.83–0.93 with only 5 % confidently wrong. The one
   inverted signal is `top1_sim` (0.29; 72 % confidently wrong by that
   axis) — a stale probe sits *right on* a mature slot — which is the
   reverse of PR-2b, where `top1_sim` was the best signal. **No single
   existing signal is correctly oriented for both failure modes.**
3. **STALE and CONTRADICTORY are label-disjoint here but mechanically
   adjacent at write time.** Overlap by label is zero (by design, no contra
   protocol ran), but every phase-2 supersession write took *exactly* the
   contradiction-fork path (`pre-write sim ≥ vigilance ∧ payload cosine ≤
   0.5`). At write time the engine cannot distinguish "hallucinated
   contradiction" from "legitimate update": the difference is which side is
   true, which only temporal/version metadata can supply. Write-time fork
   detection (the PR-2b implication) would therefore flag both — but the
   correct *policy* differs (reject vs. deprecate-the-old), so fork
   detection alone is not a resolution mechanism.
4. **Pre-registered trigger status** (SCHEMA.md: STALE AUC ≤ 0.55 → recency
   must enter the readout). The formal test is the frozen #87 two-axis
   detector (PR-3), but per-signal evidence is far above the trigger
   (collapse-family AUC 0.83–0.93), so this run does **not** support
   forcing recency into the readout on detectability grounds. What
   detectability does not give is *correction*: at the boundary the
   detector can at best abstain on a 50/50 tie; electing the *current*
   value requires knowing which fork is newer.

## What this run does not establish

- **The EMA-freeze (merge-path) stale is untested.** One-hot class payloads
  force the fork path; the originally hypothesized failure — a mature slot
  absorbing the update but still decoding to the old label — needs
  non-orthogonal payloads (e.g. soft/embedding targets) and remains open.
- Single seed, single supersession group, one class pair, stationary
  manifold; no dose-response over stale strength is available because the
  boundary tie is exact by construction (identical re-written keys). With
  near-identical real-world keys the tie becomes a near-tie; direction of
  the elected answer at the boundary is then data- not tie-break-driven.
- k/τ tuning, detector scoring at an operating point, and any readout
  redesign remain unsettled (PR-3 territory).
- The clean-arm control is the PR-2b run (same stream/config/seed); no
  stale-specific control was rerun.

## Next (not done here, per scope guardrails)

- PR-3: score the frozen #87 two-axis detector per failure mode (now with
  CONTRADICTORY and STALE both labeled) against the pre-registered
  triggers; the cross-mode orientation conflict in point 2 is the central
  question for any single-detector design.
- Open: merge-path stale (soft payloads), one-shot supersession (does the
  tie regime persist as predicted), supersession under concurrent drift.

## Files

- `per_probe_vision_stale.csv` / `.summary.json` — stale arm (2 192 rows)
- `stale_analysis.json` — analyzer output (verified byte-identical darwin/gentoo)
- `per_probe_vision_clean.csv` (PR-2b) — negative control, reused
