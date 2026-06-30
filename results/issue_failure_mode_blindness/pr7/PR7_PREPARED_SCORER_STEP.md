# PR-7 — prepared scorer step (clean_control pairC + per-seed collateral guard)

**Status: PREPARED, NOT MERGED.** Branch `feat/pr7-scorer-perseed-guard`, off the
gate branch `feat/pr7-quarantine-promotion-gate`. This implements the scorer
formalization the promotion gate names as a later step
(PR7_QUARANTINE_PROMOTION_GATE.md §4 G1/G4, §7 step 3). It is held unmerged so the
gate branch keeps its current negative result under the frozen aggregate scorer;
this branch is the proposal that would formalize the per-seed bound.

No engine/probe change; engine sha256 == frozen baseline; `--govern` opt-in;
deployed retrieval unchanged. No margin or quarantine parameter tuned — the −3
bound is the gate's pre-registered threshold, only now *implemented*.

## What changed in `benchmarks/pr7_twin_delta.py`

1. **clean_control pairC (G1).** `PAIR_CLASS_SETS` gains `pairC {10,29,42,67}`;
   `clean_control` is scored across **pairA and pairC** (the gate's G1 clean
   control), not pairA alone. New arms: `twin/clean_control/{none,quarantine}/
   per_probe_clean_s{0,1,2}_pairC.*` (clean arm, attractor 69). Quarantine is
   inert on pairC clean traffic exactly as on pairA (0 merge-suspect events →
   zero delta), so clean_control stays a pass with no per-seed breach.

2. **G4 per-seed collateral guard.** New frozen constant
   `G4_PER_SEED_COLLATERAL_TOL = 3`. Each cell now carries a
   `per_seed_collateral` block (worst seed delta, breaches, `g4_ok`); the
   manifest carries `g4_per_seed_collateral_ok`, `g4_per_seed_collateral_breaches`,
   `promotion_blockers_this_scorer`, `aggregate_panel_and_g4_clear`, and a
   `promotion_note`. A breach is a single seed with collateral Δ < −3.

The guard is a **promotion gate, separate from the aggregate cell verdict** — it
never flips a cell's pass/needs_review (those are pinned by the committed
per-action manifests). It only records whether the per-seed bound holds.

## Result on the committed panel

* `g4_per_seed_collateral_ok = False`. Breaches: **direct_harm/pairD/s2 (−4)** and
  **merge_path_stale/pairD/s2 (−4)** — the same pairD-s2 uptick, surfaced in both
  cells that score the pairD stale-soft arm. No other seed breaches
  (collateral_harm/pairE-s2 −2 is within the −3 bound; clean_control flat).
* Aggregate verdicts **unchanged**: clean_control / collateral_harm / direct_harm
  `pass`, merge_path_stale `needs_review`, overall `needs_review`,
  `both_shapes_ok` True.
* `aggregate_panel_and_g4_clear = False`; `promotion_blockers_this_scorer` names
  the two G4 breaches.

The guard does exactly what the gate intended: it turns the aggregate-masked
pairD-s2 −4 into a concrete, machine-checkable promotion blocker, without
disturbing the aggregate signal.

## Scope (necessary-not-sufficient)

`aggregate_panel_and_g4_clear` asserts only the aggregate panel + both-shapes +
G4 per-seed. It does **not** represent **G3** recoverability (a separate probe —
`pr7_recovery_probe.py`, which independently blocks: provenance-only) or
full-panel completeness. An empty blocker list from this scorer would still not
mean "promote." Promotion (`promoted` verdict, gate sequencing step 4) remains
unbuilt.

## Why unmerged

The gate branch records the negative result under the frozen aggregate scorer, as
pre-registered. Merging the per-seed guard would change the scored manifest and
the gate's verdict surface mid-evaluation. Held as a proposal: when the explicit
promotion-scoring step runs, this is the scorer it would use — and on today's
panel it would report **not promotable** (G4 breach), consistent with G3
(provenance-only recoverability). Tests: `tests/test_pr7_perseed_guard.py`
(6 pins); full PR-7 + mechanism suite green (69).
