# PR-4 design memo — geometry safety of mode-conditioned trust governance

**Status: design only. No code, no cache runs, no engine change. Written
after PR-3c landed (main `66ad253`); everything below is stated before any
PR-4 data exists. Numbers quoted are from `PR3C_RESULT.md`.**

## 0. The design question

> **Can the exploratory trust refinement be made geometry-safe and
> pre-registered, or is it only a pair-A artifact?**

PR-3c's `mode-conditioned-trust` (contradiction → deprecate the side
post-fork traffic did NOT reinforce; quarantine-both only when both sides
keep being written) was the only policy that improved both failure modes
while breaking essentially nothing on pair A: contra fixed 176–297 with
broken 1–10, mixed fixed 289–320 with broken 1–24, stale broken 0
everywhere, merge-path stale 374–380 captured by abstention, clean
untouched. It strictly dominated every baseline on the joint
(stale-fix, contra-harm) pair in every pair-A run.

Two things stop that from being a result rather than a lead:

1. **It is post-hoc.** The trust rule was written after, and motivated by,
   the PR-3c comparison table. Nothing about it has been tested against
   data it did not see.
2. **It failed its own collateral bar the one time geometry changed.** On
   mixed_s0_pairB it broke 77 correct probes (vs 1–24 on pair A) —
   deprecated slots also carried support for neighboring correct keys.
   Net accuracy still improved (0.7883 → 0.9034), but the 1-pp
   non-inferiority bar on correct traffic was exceeded, and the collateral
   was measured exactly once, on one pair, one seed.

PR-3c gives a strong lead, not a deployment license. PR-4's job is to
turn the lead into a pre-registered, geometry-swept result — or to
establish that slot-granularity trust deprecation is inherently
geometry-conditioned and report that plainly.

Scope notes against earlier framings: PR3_DESIGN.md §11 named write-path
governance as the PR-4 question *if H3 failed*; H3 instead produced an
exploratory pass with a geometry caveat, so PR-4 interrogates that caveat
first. Drift × fork (PR3C_RESULT.md §8) remains deferred — every run here
is stationary, one encoder, one cache, as before. Provenance-metadata
write-API design (PR3_DESIGN.md §10) also stays out of scope.

## 1. What PR-4 inherits as fixed constraints

* **Shadow only.** The deployed `forward()` is unchanged everywhere;
  interventions apply only to votes recomputed offline from logged top-k
  composition, with the PR-3c fidelity gate (policy-`none` shadow vote
  bit-identical to the deployed vote on every probe row,
  raise-on-mismatch) carried forward unchanged.
* **Merge-path stale is a required benchmark.** The soft-payload arm is
  the one place write-time evidence was the *only* evidence any policy
  could use (fork-topology policies fixed/abstained exactly 0 of 374
  stale-wrong rows). Any PR-4 candidate must preserve the PR-3c
  merge-suspect route: capture ≥ 95% of merge-path stale-wrong rows by
  abstention with ≤ 5 false abstentions per run, with broken = 0. A
  variant that improves contra collateral by losing merge-path capture
  fails outright.
* **One-shot ambiguity is an abstain/observe-only outcome, not a
  forced-choice target.** The protocol certifies the evidence is
  insufficient (PR3_DESIGN.md §10); electing either side of a one-shot
  tie is policy fiat, not inference. Any PR-4 policy that elects a side
  on a protocol-certified-ambiguous pair fails regardless of headline
  accuracy — including via the side-effect channel PR-3c measured (trust
  flipping 162–204 tied votes by deprecating background forks). The
  side-effect flip count is reported per run and counts as a defect to
  minimize, not a fix.
* **Read-time mode classification stays closed.** H1 ended with a
  measured pair-B inversion and a 3-way classifier that never predicts
  stale-wrong. No PR-4 policy may route on a read-time mode classifier;
  routing evidence is write-time (router verdicts, merge-suspect marks)
  plus read-time fork topology only, exactly as in PR-3c.
* **Retrieval stays unchanged unless the §6 criteria are met** — and even
  then, this memo authorizes only *proposing* a deployment PR, not making
  the change.

## 2. Hypotheses

* **H1 (pre-registered replication).** The trust rule, frozen exactly as
  specified in PR3C_RESULT.md §1 (no parameter changes, no refit —
  fork witness ≥ 2 decode classes within 0.05 raw cosine of surviving
  top-1; contradiction → deprecate the unreinforced side; merge-suspect
  → abstain when leading), reproduces its pair-A result on data it has
  never seen: new seeds and at least one new class pair of comparable
  geometry. This is the same rule promoted from exploratory to
  pre-registered status; the prediction is stated here, before the runs.
* **H2 (collateral attribution).** The pair-B broken-77 has exactly one
  of three explanations, separable by design:
  * **geometry-specific:** collateral on correct traffic scales with
    class-pair compression (a pre-registered geometry index, §4) and is
    low whenever compression is pair-A-like — trust is safe on a
    measurable subset of geometries;
  * **policy-specific:** softer or guarded deprecation (§5 variants)
    achieves a comparable fix rate with collateral within the bar even
    on compressed pairs — the harm is in *exclusion at slot
    granularity*, not in trust-side selection itself;
  * **general:** every trust-side deprecation variant exceeds the
    collateral bar on compressed geometry at any useful fix rate —
    slot-granularity governance is inherently unsafe as geometry
    compresses, and the safe policy set collapses back to
    {observe-only, merge-suspect abstention, abstain-tie}.
* **H3 (geometry-safe variant).** At least one fully pre-specified
  guarded variant (§5) meets the target-mode improvement AND the
  collateral bar on *every* pair in the geometry sweep, replicated
  across seeds. Failure of H3 with confirmation of H2-geometry is the
  informative-failure outcome, not a wasted study.

Pre-registered directional predictions (falsifiable): unguarded trust
collateral (broken on correct traffic) is monotone in the §4 compression
index across the pair sweep; the PR-3c rule router's verdicts themselves
(contradiction/supersession/ambiguous assignment) are geometry-stable
(they were pair-B-unchanged in PR-3c, unlike every read-time signal) — so
any collateral growth is attributable to the *intervention*, not the
*routing*.

## 3. Collateral must be measured, not just counted

PR-3c reported broken-77 as a row count. PR-4 decomposes it. New
per-policy outcome columns in the governance table, per run:

* **direct br:** broken correct probes whose top-1 slot was itself a
  deprecated fork side for the probe's own key region;
* **collateral br:** broken correct probes whose elected-vote change came
  from deprecating a slot whose fork verdict concerned a *different* key
  — the pair-B mechanism (deprecated slots carrying support for
  neighboring correct keys);
* **per-slot collateral exposure:** for every deprecated slot, the
  fraction of its surviving top-k appearances (over the run) on probes
  whose registry key is NOT a party to the slot's fork — computable from
  the existing topk + per_slot + registry artifacts, label-free at
  policy time (registry used only for scoring).

H2's three branches are distinguished on the (collateral br × geometry
index × variant) table, not on net accuracy: net accuracy improved even
on pair B and is therefore the wrong safety metric.

## 4. Geometry sweep (the new experimental axis)

* **Compression index, pre-registered before pair selection:** for a
  class pair, the mean pairwise cosine between the encoder keys of the
  two classes' probe sets (computed once from the existing vitl14 cache
  features, before any cache run), reported alongside the realized
  fork-witness co-residency rate per run as a check that the static
  index predicts the dynamic crowding. Pair A and pair B are scored on
  the same index first; the design expects pair B to score as more
  compressed — if it does not, the index is wrong and must be revised
  *before* the runs (revision recorded in this memo's changelog).
* **Pair set:** ≥ 4 class pairs spanning the index — the PR-3 pair A and
  pair B as anchors, plus ≥ 1 pair more compressed than B and ≥ 1
  between A and B. Selection rule (by index quantile, not by eyeballing
  results) is fixed when the index is computed and recorded here before
  any governance run.
* **Seeds:** 3 seeds on every headline (pair × arm) cell.
* **Arms per pair:** `mixed` (the headline collateral arm), `contra`,
  `stale`, `stale --soft-payloads` (required benchmark), `stale
  --one-shot` (ambiguity benchmark), `clean` (zero-harm control). Jitter
  arms are not repeated — the PR-3b jitter question (tie-artifact
  sensitivity) is closed and orthogonal to geometry.
* All runs on the verified vitl14 cache, #87 config, stationary; gentoo
  compute, darwin byte-identity verification, per host-role convention.

## 5. Policy set

Baselines, unchanged from PR-3c (same code paths, pinned by the existing
tests): `none`, `observe-only`, `entropy-abstain` (frozen #87),
`abstain-tie`, `recency-naive`, `quarantine-naive`, and **`mode-cond`
(the PR-3c pre-registered rule routing, quarantine-both on
contradiction)** — kept as the it-must-be-beaten governance baseline so
the trust refinement is always compared against the policy that *was*
pre-registered last time.

Candidates (all fully specified here; any parameter not fixed below is
fixed in this memo before the first cache run, never tuned on PR-4 data):

* **`trust`** — the PR-3c exploratory rule, frozen verbatim. H1's
  subject.
* **`trust-downweight(λ)`** — identical routing, but the deprecated
  side's vote weight is multiplied by λ instead of excluded. λ = 0.25,
  fixed now. Tests whether exclusion-vs-attenuation is the collateral
  driver (H2-policy branch).
* **`trust-guarded(θ)`** — identical routing, but a slot is deprecated
  only if its collateral exposure proxy is low: the fraction of the
  slot's surviving top-k appearances (within the shadow log up to that
  probe) on fork-party keys is ≥ θ; otherwise the slot is downweighted
  at λ = 0.25 instead. θ = 0.8, fixed now. This is the explicit
  geometry-safety guard: it spends fix-rate to cap collateral, using
  only engine observables already logged.
* **`trust-record`** — *conditional candidate*: deprecate only the
  fork-attributed records' contribution within a slot rather than the
  slot's whole vote, **iff** that contribution is reconstructable from
  the logged topk decode composition plus `slot_records` lineage without
  new engine observables. Feasibility is determined from the existing
  PR-3c artifacts before the run matrix is finalized; if it is not
  reconstructable shadow-side, the variant is dropped here, not
  approximated.

Every candidate inherits the merge-suspect → abstain route and the
ambiguous → observe-only route unchanged. No candidate may consume a
read-time mode classification (§1).

## 6. Acceptance criteria (pre-registered)

Collateral bar, everywhere below: broken-on-correct-traffic ≤ 1 pp
absolute of `none`'s correct rows, per run — the same bar trust failed on
pair B.

* **H1 success:** frozen `trust` on new seeds / the new pair-A-like pair
  reproduces the PR-3c pair-A envelope — contra target-mode improvement
  with collateral within the bar, stale broken 0, merge-path capture per
  §1, clean untouched. **Failure:** collateral bar exceeded or
  target-mode improvement absent on pair-A-like geometry → the PR-3c
  table overfit even its own geometry; report and stop (H2/H3 still run,
  but no variant can be promoted on a failed H1).
* **H2 is attribution, not pass/fail:** the (collateral br × index ×
  variant) table assigns the pair-B caveat to geometry-specific /
  policy-specific / general per §2. An ambiguous table (non-monotone
  collateral, or variants splitting inconsistently across seeds) is
  reported as ambiguous and blocks H3 promotion.
* **H3 success:** ≥ 1 fully pre-specified variant meets target-mode
  improvement (contra error containment AND mixed improvement) with the
  collateral bar satisfied on **every pair** in the sweep, all seeds,
  while preserving merge-path capture (§1), one-shot abstention (§1,
  side-effect flips reported and ≤ PR-3c levels), and clean zero-harm —
  and strictly dominates `mode-cond`, `recency-naive`,
  `quarantine-naive`, and `entropy-abstain` on the joint
  (target-mode-fix, collateral) pair. **Informative failure:** every
  variant trades fix rate against collateral such that no point meets
  both on compressed pairs → slot-granularity trust deprecation is
  geometry-bounded; the result memo states the measured safe-geometry
  region and the safe fallback set, and record-granularity or write-path
  refusal becomes the explicitly justified PR-5 question. **Failure:**
  guards cost fix rate without reducing collateral (guard proxy doesn't
  track the §3 exposure measure) → the collateral mechanism is not
  observable from engine state; governance at this granularity is
  closed.
* **Only on H1+H3 success** may a deployment proposal be drafted — as a
  separate, subsequent PR with its own gates; PR-4 itself changes no
  retrieval behavior under any outcome.

## 7. Risks and confounds

* **Pre-registration leakage (the central risk).** The trust rule has
  seen pair A seeds 0–2 and pair B mixed/contra/stale s0. Those runs can
  anchor the sweep but cannot count toward H1/H3 replication; all
  promotion decisions rest on new seeds and new pairs only.
* **Index validity.** If the static compression index fails to order
  pair A vs pair B as expected (§4), the geometry axis is rebuilt before
  any governance run; proceeding with a broken index would make
  H2-geometry unfalsifiable.
* **Guard circularity.** `trust-guarded`'s exposure proxy is computed
  from the same shadow log it governs; the §3 scoring measure uses the
  registry. The proxy must be label-free (pinned by test, as in PR-3c)
  and is validated *against* the registry measure, never built from it.
* **Variant garden.** Three candidate variants × failure-to-prespecify =
  a tuning exercise. Every parameter is in this memo (λ = 0.25, θ = 0.8,
  witness window 0.05 carried from PR-3c); any change after the first
  cache run invalidates the affected variant's pre-registered status.
* **One-shot side-effect flips.** Trust's tied-vote flips are benchmark-
  flattering (truth = latest write) but epistemically empty; they are
  excluded from target-mode improvement accounting and reported
  separately, so a variant cannot pass on fiat fixes.
* **Stationarity, still.** Nothing here transfers to drift × fork;
  PR-4's conclusions are scoped to the stationary cache exactly as
  PR-2/PR-3's were.

## 8. Smallest safe implementation path (when PR-4 is greenlit)

1. Compute the §4 compression index from the existing cache features;
   record pair selection and the pair-A/pair-B index check as a short
   addendum to this memo (no cache runs yet).
2. Determine `trust-record` shadow-feasibility from existing PR-3c
   artifacts; record keep/drop in the same addendum.
3. Extend `analyze_fork_governance.py` with the three variants and the
   §3 collateral decomposition + hermetic tests (label-free pins,
   observe-only ≡ none, fidelity gate) — analysis code only.
4. Gentoo run matrix (pairs × arms × seeds), darwin byte-identity
   verification, then the comparison table and result memo answering
   H1/H2/H3 against §6.

Each step lands on main before the next begins; this memo is step 0.

## Addendum A — gate 1: compression index, pair selection, trust-record feasibility

**Recorded 2026-06-12, before any PR-4 governance/cache run. Steps 1–2 of
§8. Artifact: `pr4/compression_index.json` (computed on gentoo from the
existing vitl14 cache, sha256
`bb2d9fde06147136363f27e50acdb52654de5ee5ae993b07faaed8f0d84ee43e`,
byte-verified on darwin). Analyzer: `benchmarks/pr4_compression_index.py`,
hermetic pins in `tests/test_pr4_compression_index.py`.**

### A.1 Index operationalization (primary, fixed now)

For an unordered class pair (i, j): mean cosine over ALL cached row pairs
of the two classes, on unit-normalized features (the normalization
`VisionDriftStream` applies — the geometry retrieval actually sees).
For a 4-class set: the mean of the 6 within-set pair cosines. Static and
seed-free by construction (full cache rows, not the per-seed 32/64 probe
subsets). Higher = more compressed. **This set-level index is the primary
geometry axis for every §2/§6 H2 decision.** Two secondary diagnostics
are recorded per config, descriptively only (fixed now so they cannot be
adopted post hoc): the stale-fork pair cosine (classes[0], classes[-1] —
the supersession locus) and the mean class-to-attractor cosine.

### A.2 §4 validity check: pair A vs pair B

* pair A (0,8,19,33 / attr 71): index **0.082731**
* pair B (5,27,48,86 / attr 13): index **0.084472**

Pair B scores above pair A — the pre-registered direction holds, so the
index stands and the geometry axis is NOT rebuilt. Honest caveat,
recorded before any run: the set-level margin is small (+0.0017, ~2%
relative), while the stale-fork pair cosine separates more strongly
(A 0.0698 vs B 0.0841, +20% relative), consistent with PR-3c's pair-B
collateral concentrating at the supersession locus. No margin threshold
was pre-registered — direction only — and none is added now. If the §2
monotonicity prediction comes out ambiguous on the primary index, that
is reported as ambiguous per §6; the secondary diagnostic does not get
promoted after the fact.

### A.3 Pair selection (rule fixed in `pr4_compression_index.py`)

Candidates: 256 fixed-seed (seed 0) 4-class+attractor draws over the 90
classes unused by pairs A/B; candidate index range 0.0455–0.1238.
Selection by index only, no results consulted:

| config | classes | attr | index | role |
|---|---|---|---|---|
| pair A | 0,8,19,33 | 71 | 0.082731 | anchor (leakage-tainted, §7) |
| pair C | 10,29,42,67 | 69 | 0.082754 | nearest A — H1's new pair-A-like pair |
| pair D | 10,28,32,95 | 52 | 0.083550 | nearest A/B midpoint |
| pair B | 5,27,48,86 | 13 | 0.084472 | anchor (leakage-tainted, §7) |
| pair E | 47,56,61,76 | 1 | 0.123812 | highest-index candidate above B |

Five configs spanning the index, satisfying §4 (≥ 4 pairs; ≥ 1 between A
and B; ≥ 1 more compressed than B). Caveat: with the A–B gap this small,
pair D's "between" role is nearly degenerate; pair E carries the real
compression extension (+47% over B). The realized fork-witness
co-residency rate per run remains the §4 dynamic check on this static
index.

### A.4 `trust-record` feasibility: DROPPED

The §5 conditional candidate is **not reconstructable shadow-side** from
the PR-3c artifacts:

* `topk.csv` logs per candidate (slot, sim, surviving, weight, decode)
  where `decode` is the slot's argmax decode only — the slot's payload
  vector and its per-record composition are never logged.
* `per_slot.csv` logs `n_records` (a count); the `slot_records` record-id
  lineage is in-memory engine state (`associative_core.py`), not exported.
* `fork_events.csv` attributes record→slot only for conflict events;
  clean reinforcement writes — the bulk of a slot's records — have no
  logged record→slot assignment, and the EMA blend order/coefficients
  that would apportion `values[slot]` among records are not recoverable
  from any logged table.

Deprecating "only the fork-attributed records' contribution" would mean
reconstructing `values[slot]` minus those records, which the logged
evidence cannot support. Per §5: dropped here, not approximated. The
PR-4 candidate set is `trust`, `trust-downweight(λ=0.25)`,
`trust-guarded(θ=0.8, λ=0.25)` plus the §5 baselines.

### A.5 Gate verdict

The §7 index-validity risk is retired (direction confirmed); pair
selection and the trust-record keep/drop are recorded before any
governance run. PR-4 is cleared to proceed to §8 step 3 (analyzer
variants + collateral decomposition, analysis code only) and then step 4
(the gentoo run matrix over pairs A–E). Pair A and pair B runs remain
leakage-tainted anchors per §7: no H1/H3 promotion may rest on them.
