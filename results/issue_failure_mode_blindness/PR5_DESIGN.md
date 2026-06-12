# PR-5 design memo — the smallest next step after PR-4's negative result

**Status: design only. No code, no cache runs, no engine change. Written
after PR-4 closed (main `eb69cfb`); all numbers quoted are from
`PR4_RESULT.md` / `pr4/pr4_geometry_table.json`. PR-4 is treated as a
successful negative result — the goal here is not to patch the failed
policy but to choose what, if anything, is worth building next.**

## 0. The design question

> **Given PR-4's negative result, what is the smallest next step that can
> make governance safer without repeating slot-granularity trust
> deprecation?**

Hard constraint, stated up front and repeated in §8: **PR-5 must not
revive slot-granularity trust deprecation.** PR-4 falsified it across
three fully pre-specified variants on 68 fresh cells; the only thing that
reopens that family is new evidence that changes the *unit of action*
(per-record contribution instead of whole slots) — which is path 2 below,
and which is currently blocked on observables the engine does not emit.

## 1. Inventory — what PR-5 starts with

* `pr4/pr4_geometry_table.json` — 90 scored cells (5 pairs × 6 arms × 3
  seeds × 11 policies), §6 check booleans, per-cell witness rates,
  byte-identical darwin/gentoo. Includes the **§3 per-slot
  collateral-exposure blocks (registry exposure beside the label-free
  proxy) which PR-4 emitted but never analyzed** — free evidence for §3
  path comparison, no compute needed.
* `pr4/compression_index.json` — the falsified static index, plus the
  recorded-but-unpromoted secondary diagnostics (stale-fork pair cosine,
  attractor cosine) and the 256-candidate scan. Pair D's full within-set
  cosine structure is in here.
* `pr3c/` + `pr4/` raw artifacts (per-probe, per-slot, fork-events, topk)
  for all 92 runs — any *read-time* counterfactual or any new *static*
  index can be evaluated against them without a single new cache run.
* `feature_cache_vitl14` (gentoo) — 50k × 1024-d features, 100 classes;
  76 classes untouched by any run (24 spent across the five pairs +
  attractors; C and D share class 10) ⇒ a large held-out pool for
  prospective index validation.
* `benchmarks/analyze_fork_governance.py` — scorer with fidelity gate,
  variants, decomposition, tie-flips; `analyze_pr4_geometry.py` — §6
  evaluator; both pinned by hermetic tests.
* Engine (`associative_core.py`): `slot_records` lineage is in-memory
  only; the value EMA (`values[slot] += α(t)·(target − values[slot])`,
  adaptive α from `hit_counts`) is **not logged per write** — confirmed
  at gate 1 (Addendum A.4) as the reason trust-record was dropped.

## 2. What PR-4 closed (and the one positive)

Closed: slot-granularity trust deprecation, in all three pre-specified
action flavors (exclude / attenuate λ=0.25 / guard θ=0.8). Decisive
counterexample: **pair D** — the pre-registered compression index scored
it within 1% of pair A, yet it was the worst geometry in the sweep
(mixed-arm broken 240–330/run, mostly **direct** wrong-side deprecation
at the contested key region, 163–206/run). Pair B's harm was the milder
*collateral* channel (neighboring-key support). Pair E exceeded the
one-shot tie-flip ceiling (286–294 vs 204). The guard proxy saturated
(≥ θ for essentially every deprecated slot — a ±1-row no-op);
attenuation was strictly worse than exclusion. H1 also failed zero-harm
on a third of seeds even on A-like geometry.

The positive finding PR-5 builds on: **router verdicts were
geometry-stable** (supersession exactly 160 in every mixed run across
all five pairs; contradiction 209–237; conflict-pair counts 423–454).
The write-time evidence and its classification are sound. *Acting on
that classification by deprecating whole slots at read time is the
defect.* Also carried forward: merge-path stale is catchable only at
write time (PR-3c), and its capture itself degraded on D/E (pair D
capture 0.79–0.93; false abstentions up to 128 on D and 143 on E) — so
even the current "safe set" has a geometry caveat worth recording.

## 3. The three candidate paths

### Path 1 — a geometry index that orders pair D, before any policy test

Build and pre-register a static hazard index whose job is purely
predictive: given a class set, order it correctly against the five spent
pairs (D worst, C/A benign), then survive prospective validation on
held-out class sets.

* **Cost:** lowest of the three. Pure analysis against existing cache
  features + existing run artifacts; new compute only for a small
  prospective-validation matrix (mixed arm is the discriminating arm —
  D vs C separate by 50× there).
* **What it buys:** a gate every future governance experiment needs
  anyway. PR-4's sweep design assumed the index worked; without a
  D-ordering index, *any* future policy claim ("safe on benign
  geometry") is unfalsifiable, including paths 2 and 3.
* **Risks:** index-shopping. Five pairs is a tiny training set and the
  candidate space of geometry statistics is huge; anything fit to the
  spent pairs must be validated only on held-out sets, predicted
  *before* their runs (the gate-1 discipline, now with a real
  counterexample to order). Honest possible outcome: no static feature
  of the class set orders D — hazard is emergent from write dynamics —
  which is itself decision-relevant (it would force per-class-set
  empirical validation onto every future proposal and strengthen the
  case for path 3, which does not condition on geometry).
* **Evidence already pointing somewhere:** D's failure was *direct*
  wrong-side deprecation — the rule's reinforcement-side attribution
  breaks where same-class traffic lands in multiple co-resident slots.
  The unanalyzed §3 exposure blocks and D's fork-events give a
  zero-compute first look at what distinguishes D's conflict structure
  before any new index is invented.

### Path 2 — record-granularity action (new engine observables)

Deprecate a fork-attributed record's *contribution* inside a slot rather
than the slot. Gate 1 established this is not reconstructable from
existing artifacts; it requires a **write ledger** (§5) and re-running
caches to regenerate artifacts with it.

* **Cost:** medium-high. Engine/driver logging change (read path
  untouched), schema addition, full re-run of at least one pair grid,
  plus the ledger's determinism burden.
* **What it buys:** a smaller blast radius for the same action.
* **Why it is NOT the smallest next step:** PR-4's decisive failure mode
  argues against it. On pair D the harm was mostly **direct** — the rule
  chose the *wrong side* at the contested region. Shrinking the unit of
  action from slot to record does not fix side-*selection*; it shrinks
  the damage of a correct selection wrongly scoped (the pair-B
  collateral channel, 53–59/run — real but the smaller problem).
  Building new engine observables to mitigate the minor channel while
  the major channel is untouched is the wrong order. Record granularity
  becomes justified only if a future result shows selection is sound and
  scope is the residual defect.

### Path 3 — write-path refusal / write-time governance (PR3_DESIGN §11)

Act where the evidence lives: refuse, quarantine, or annotate the
*incoming conflicting write* (or the merge-suspect absorb) instead of
re-weighting reads. This is the reservation PR-3 made for exactly this
contingency, and it leans on PR-4's positive finding (verdict stability)
plus PR-3c's (merge-path stale is write-time-only evidence).

* **Cost:** highest. A refused write changes all subsequent engine
  state, so it **cannot be shadow-scored from existing read logs** — it
  needs a twin-run protocol (governed vs ungoverned writer over the same
  stream, compared on held-out probes), i.e. a new harness and new cache
  runs per cell. It also changes deployed *write* behavior if it ever
  ships, which is an engine change with its own gates.
* **What it buys:** the only path that addresses merge-path stale at its
  source and the only one that does not depend on read-time geometry at
  all.
* **Why not first:** twin-run governance experiments inherit the same
  validation problem PR-4 just exposed — "works on the pairs we ran" is
  exactly the claim pair D falsified. Running path 3 before an index (or
  a demonstrated impossibility result from path 1) repeats PR-4's
  structural mistake with a more expensive protocol.

## 4. Recommendation — smallest safe PR-5

**PR-5 = path 1, scoped to one falsifiable claim: a pre-registered
static hazard index that orders the five spent pairs correctly and
survives prospective validation on held-out class sets.** No policy, no
engine change, retrieval untouched. Paths 2 and 3 are explicitly *not*
killed — they are sequenced behind the gate they both need: path 3
(write-path, as PR-6) if an index exists or is proven impossible; path 2
only if some future result isolates scope (not selection) as the defect.

Smallest implementation, in landing order (each step on main before the
next):

1. **Post-mortem, zero compute:** analyze the already-emitted PR-4
   evidence — pair D's fork-events/per-slot structure, the §3 exposure
   blocks, witness-window composition — and the cache features, to
   identify what static property separates D from C/A. Candidate
   indices are *constructed* here against the five spent pairs (this is
   training data and is named as such). Result: a short addendum with
   ≤ 3 candidate indices, fully specified.
2. **Pre-registration:** for each candidate index, score the 90-class
   held-out pool, select K ≥ 4 validation class sets spanning
   predicted-hazard (≥ 2 predicted hazardous, ≥ 2 predicted benign,
   selection by index quantile only), and record the predictions —
   hazard ordering and a pass/fail margin — in the addendum *before any
   run*.
3. **Prospective validation, minimal matrix:** mixed arm only (the
   discriminating arm), 3 seeds, K sets — shadow-scored with the
   existing PR-4 scorer; the "policy" under measurement is frozen
   `mode-conditioned-trust` used purely as a **hazard probe** (its
   broken/collateral counts are the measurand, never a deployment
   candidate). Gentoo compute, darwin byte-verification, as always.
4. **Result memo:** index validated / falsified / impossible, and the
   consequent go/no-go for PR-6 (path 3).

## 5. Required engine observables IF path 2 is ever chosen (recorded now)

For completeness and to prevent re-derivation: record-granularity shadow
reconstruction needs a per-write ledger emitted at write time —
`(record_seq, epoch, slot, outcome, adaptive_alpha_at_write,
payload_target_decode, value_decode_before, value_decode_after)` per
written row — sufficient to replay `values[slot]`'s EMA composition and
subtract a record's contribution offline. Constraints: protocol-time
only (no wall-clock, the `last_seen` lesson), read path untouched,
float32 round-trip text (the topk fidelity lesson), pinned label-free.
This is a schema + driver change and a full artifact regeneration; it is
not part of PR-5.

## 6. Acceptance criteria (pre-registered for PR-5 = path 1)

* **Retrospective (necessary, not sufficient):** the chosen index orders
  the spent pairs with D strictly worst and C/A strictly most benign on
  mixed-arm trust broken counts (Spearman against the five-pair hazard
  ranking = 1.0 on the D/C/A extremes; B/E may interleave).
* **Prospective (the real test):** on the K held-out sets, every
  predicted-benign set shows mixed-arm trust broken within the PR-4
  collateral bar (≤ 1 pp of none's correct rows) on all seeds, and every
  predicted-hazardous set exceeds it on all seeds — i.e., the index
  separates with no inversions at the chosen quantiles.
* **Honesty rails:** predictions recorded before runs; one index
  promoted, others reported; no index revision after the validation runs
  (a failed index fails — revision means a new pre-registration and new
  held-out sets).

## 7. Stop conditions

* **No candidate index orders pair D retrospectively** → stop before any
  run; report that static class-set geometry does not predict the
  hazard. Consequence recorded in the memo: every future governance
  claim requires per-class-set empirical validation, and PR-6/path 3
  must be designed without geometry gating.
* **Retrospective pass, prospective fail (any inversion)** → report the
  index as overfit to the spent pairs; same consequence as above; do not
  iterate indices against the now-spent validation sets.
* **Ambiguous separation** (margins straddle the bar across seeds) →
  report as ambiguous; no promotion; widening the validation set is a
  new pre-registered study, not a continuation.

## 8. Constraints carried forward (binding on PR-5 and successors)

* **Deployed retrieval remains unchanged** until some later PR earns a
  deployment proposal through its own gates; PR-5 contains no mechanism
  that could change it.
* **Merge-path stale stays a required benchmark** for any future policy
  PR — including its newly measured degradation on D/E geometry, which a
  PR-6 candidate must not worsen.
* **One-shot ambiguity stays observe-only.** The protocol certifies the
  evidence as insufficient; nothing in PR-4 changed that. It becomes
  classifiable only via provenance metadata at the write API
  (PR3_DESIGN.md §10) — path-3 territory, not PR-5.
* **Slot-granularity trust deprecation is not revived** — not as a
  baseline, not as a "calibrated" variant, not with new thresholds. Its
  only sanctioned use is §4 step 3: a frozen hazard *probe* whose damage
  is the measurand. The family reopens only on evidence that changes the
  unit of action, with the §5 observables in place.
