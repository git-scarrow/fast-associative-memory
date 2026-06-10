# PR-3 design memo — mode-conditioned failure detection and fork governance

**Status: PR-3a executed (`PR3A_RESULT.md`) — audit PASS, both triggers
fired. PR-3b executed (`PR3B_RESULT.md`) — signatures are additionally
class-geometry-conditioned (pair-B sign flips); merge-path stale
confirmed persistent; one-shot tie confirmed permanent; H1 near-falsified
as geometry-general, H2 strengthened. PR-3c executed (PR3C_RESULT.md) —
H1 failure confirmed (pair-B collapse; 3-way never predicts stale-wrong);
H2 separable but forced-choice classifiers misroute one-shot ambiguity and
absorbed supersession (rule router with explicit ambiguity is what works);
H3: pre-registered quarantine-both mode-conditioning fails non-inferiority
on contra, exploratory trust refinement (deprecate the unreinforced side)
passes on pair A with a measured pair-B collateral caveat. Shadow only —
no deployed retrieval change.**
Written after PR-2a/2b/2c landed (main `2d264c4`). Everything below is
stated before any PR-3 data exists; numbers quoted are from the PR-2 memos.

Revision 2 changes (before implementation): the PR-2 signal profiles are
demoted from classifier grammar to hypotheses with named artifact risks; a
temporal/provenance invariant audit is added to PR-3a as a gate for PR-3b;
ambiguity gets an explicit observe-only governance outcome; cross-mode harm
gets a non-inferiority acceptance criterion against simple baselines; the
label taxonomies distinguish ground-truth labels from observational ones.

## 1. Repo / artifact inventory (what PR-3 builds on)

| Artifact | Role |
|---|---|
| `benchmarks/failure_mode_probe.py` (857 ln) | PR-2a injection driver: `clean`/`contra`/`stale` arms, `FailureModeRegistry` (fork outcomes, stale groups), `label_probes` with runtime-verified vote replication, synthetic + `--vision` modes |
| `benchmarks/analyze_failure_modes.py` (208 ln) | per-arm label rates, per-signal a-priori-oriented AUC, confidently-wrong rates |
| `benchmarks/heldout_abstention.py` | the frozen #87 two-axis detector (rank_gap = `top1_top2_margin`; support axis = `effective_support`, secondary `n_surviving_votes`; logistic fit, epoch-split, orientation/standardization/coefficients/threshold frozen) |
| `results/issue_failure_mode_blindness/SCHEMA.md` | label semantics (exposure vs causality), CSV schema, pre-registered PR-3 triggers |
| `PR2B_CONTRA_RESULT.md`, `PR2C_STALE_RESULT.md` | empirical results (below) |
| `per_probe_vision_{clean,contra,stale}.csv` + summaries + analyses | the data PR-3a scores without re-running anything |
| `tests/test_failure_mode_probe.py` (15) + `tests/test_failure_mode_vision.py` (7) | hermetic mechanism gates; cache runs are invalid unless green |

Engine state per slot, label-free, **without modification**: `last_seen`,
`hit_counts`, `usage`, `slot_records` (write lineage), payload argmax
decode. Driver-side, record ids are a monotone write sequence.

Preliminary code reading (not yet a pinned invariant) says all mutation
sites for these fields live in `learn_local`'s hit/miss blocks
(associative_core.py:806–881) and `forward()` is documented read-only
(associative_core.py:536). PR-3 treats this as a **hypothesis to audit**
(§5, PR-3a), not a foundation: every temporal/provenance feature below is
conditional on the audit passing, and the feature set shrinks — the theory
does not get forced — if any field turns out to be read-mutated.

## 2. PR-2b summary — contradiction (gentoo, vitl14_cifar100_train, stationary)

228/228 injections forked (zero absorbed/missed/dropped); wrong rate 9.4%
with clean dose-response; clean control 0 wrong, 0 flags. Lenient coverage
96.6% of wrong, strict 61%, median fork vote mass 0.667 — causally loaded.

Signature observed: **confident wrong recall on the collapse axes.**
Entropy / effective_support / n_surviving_votes are *anti-predictive*
(AUC 0.31–0.36; 71% confidently wrong by entropy/support): a contradictory
failure is a tight, sharp, low-entropy retrieval of co-resident forks.
Partial signal on `top1_sim` (0.74) and margin (0.73) only. The fork is
mechanically known at write time (`pre-write sim ≥ vigilance ∧ payload
cosine ≤ 0.5`); at read time it imitates a confident correct recall.

## 3. PR-2c summary — stale / supersession (same cache, stationary)

Wrong rate 2.7%, **all 60 errors in the single supersession-boundary epoch**:
each superseded key holds one mature A slot and one fresh B fork at an
identical key position, the vote splits exactly 0.5/0.5
(`stale_vote_weight` = 0.500 on all 60, margin = 0), and argmax tie-breaking
— not evidence — elects A. From the next epoch B copies outnumber A and all
probes elect B, while 32 stale slots remain live and reachable through the
end of the run (epochs 7–11: 297/930 voting probes carry stale slots in
surviving top-k at median 0.20 mass). One redundant fork per key per
re-write epoch (write amplification). A one-shot supersession would leave
the tie regime in place indefinitely — predicted, not yet tested.

Signature observed: **broad ambivalent vote.** Collapse axes well oriented
(AUC 0.83–0.93, 5% confidently wrong); `top1_sim` inverts (AUC 0.29, 72%
confidently wrong) because a stale probe sits *on* a mature slot. Every
phase-2 write took exactly the contradiction-fork path: at write time the
engine cannot tell hallucinated contradiction from legitimate update.
Merge-path (EMA-freeze) stale never occurred — one-hot payloads force the
fork path — so that hypothesized mode is **untested**, not refuted.

## 4. The exact scientific problem

The two arms produced structurally different, partially *opposed* failure
signatures on the same frozen telemetry:

| signal family | contra orientation (observed) | stale orientation (observed) |
|---|---|---|
| entropy / support / n_surviving | inverted (failure = LOW) | correct (failure = HIGH) |
| `top1_sim` | correct (failure = LOW) | inverted (failure = HIGH) |
| `top1_top2_margin` | weakly correct (0.73) | correct (0.93) — degenerate (exact tie) |

**These profiles are hypotheses, not settled classifier grammar.** Each
element carries a named artifact risk from the PR-2 protocol:

* the stale margin-0 / 0.5-vote-split signature follows mechanically from
  *identical* re-written keys — the jitter arm (§5) tests whether any
  margin/tie signal survives realistic key dispersion;
* the stale high-entropy/high-support signature was measured in a single
  boundary epoch (n=60), one seed, one class pair — replication arms test
  whether it is a property of supersession or of this run;
* the contra low-entropy signature was measured under a monotone fork
  ramp on a stationary, well-separated manifold; co-occurrence with
  supersession (mixed arm) and other class geometries may dilute it;
* both arms exclude drift by design, so nothing here is claimed about how
  these profiles interact with the BLENDED signature.

What survives all of those checks becomes classifier input; what does not
is reported as protocol-specific. The orientation conflict itself — no
single signal correctly oriented for both modes — is the finding PR-3 must
either confirm beyond the PR-2 setup or narrow to an artifact.

Independent of read-time signatures: both modes traverse the *same*
write-time fork event but demand opposite governance, and recency points in
opposite truth-directions (stale: newer write correct; contra: newer write
is the hallucination — blind recency would *elect the hallucination*).
That is a pre-registered prediction PR-3 tests, not an assumption.

PR-3 therefore tests three separable hypotheses:

* **H1 (read-time mode identification).** The signal *profile* — not any
  single signal — identifies the failure mode, so a two-stage readout
  (mode classifier → mode-conditioned orientation) can recover
  discriminability no single-oriented detector has. Falsifiers, in order
  of strength: held-out mode-classification near chance ⇒ orientation
  conflict is fundamental at read time and governance must move to the
  write path (question 7 answered empirically); classification that works
  in-protocol but collapses under jitter / unseen seeds / unseen class
  pairs / the mixed arm ⇒ the profiles were protocol fingerprints.
* **H2 (write-time fork classification).** Fork events plus temporal
  observables (conflict timeline per key region: does the conflicting
  payload *replace* the old writer or *interleave* with it?) separate
  contradiction from supersession — with one honest exception: a one-shot
  conflicting write is observationally identical to a one-shot
  supersession and a single hallucination. Success for that case is a
  calibrated **ambiguous** output, not a forced guess.
* **H3 (governance without cross-mode harm).** Deprecating older fork
  sides may fix the stale boundary; quarantining conflicted forks may
  contain contradiction; doing either *unconditionally* is predicted to
  harm the other mode. Mode-conditioned governance is useful only if it
  improves its target mode while remaining **non-inferior on the other
  mode** relative to no governance — and only if it beats the simple
  baselines (§9). Doing nothing on an unresolved fork is a legitimate
  outcome, not a failure of the framework.

PR-2b and PR-2c are NOT collapsed into one "memory conflict" story: they
share a write mechanism but differ in readout signature, error dynamics
(accumulating dose-response vs one-epoch transient + permanent latent
reachability), and correct policy. Stale is not weaker contradiction; it is
a tie/argmax failure plus a persistence problem.

## 5. Proposed protocol

### PR-3a — invariant audit + frozen-detector scoring (no new cache runs)

**(i) Temporal/provenance invariant audit — gate for everything downstream.**
A small hermetic test (and a read-only audit helper) that snapshots
`last_seen`, `hit_counts`, `usage`, and `slot_records` and asserts they are
bit-identical across `forward()`, `probe_cross_class_similarity()`,
`label_probes()`, and the eviction-score paths — i.e., that no field later
used as temporal or provenance evidence is mutated by any read/inference
path. Expected to pass per §1's code reading; **if any field fails, that
field is removed from the PR-3 feature set and the design proceeds with
what remains** (the audit result is recorded in the PR-3a memo either way,
and the passing fields become pinned invariants future engine changes
cannot silently break).

**(ii) Frozen #87 detector scoring.** Score the frozen two-axis detector
(orientation, standardization, coefficients, threshold — refit nothing) on
the existing PR-2b/2c CSVs, per failure mode, against the pre-registered
SCHEMA.md triggers:

* per mode: two-axis AUC (mode-wrong vs correct), error capture at the
  frozen operating point, confidently-wrong rate under the detector score;
* dose-response: error rate vs `contra_vote_weight` / `stale_vote_weight`
  (exposure → attribution, as SCHEMA.md planned);
* trigger evaluation: CONTRADICTORY AUC ≤ 0.60 or confidently-wrong > 50%
  → write-time contradiction governance required; STALE AUC ≤ 0.55 →
  recency must enter the readout. (Per-signal evidence predicts the contra
  trigger fires and the stale one does not; the frozen-score run is the
  formal test.)

### PR-3b — new arms (all on the verified vitl14 cache, stationary, #87 config)

These arms exist primarily to test whether the PR-2 signatures are
properties of the modes or of the protocol:

| Arm | Purpose |
|---|---|
| `mixed` | contra injections AND a supersession group in the same memory. The only arm where fork-event classification (H2) and cross-mode governance harm (H3) can be tested under co-occurrence; also tests whether labels stay disjoint when both modes are live (lenient overlap is reported, never folded), and whether each mode's signature survives the other's presence. |
| `stale --one-shot` | phase 2 writes K→B exactly once. Tests the prediction that the 0.5/0.5 tie regime persists indefinitely (highest-severity stale scenario) and supplies the honest ambiguous case for H2. |
| `stale --key-jitter ε` | phase-2 keys perturbed by ε. Breaks the exact-tie protocol artifact: dose-response of boundary election direction and margin vs tie tightness. Any margin/tie-based claim that does not survive jitter is rejected as artifact. |
| `stale --soft-payloads` | non-orthogonal payload targets (cosine > 0.5) to force the merge path: first test of EMA-freeze stale (mature slot absorbs the update, keeps decoding A). Currently untested; its read-time signature is unknown and may resemble neither PR-2 profile. |
| seeds / classes | ≥ 3 seeds and a second class pair for the headline arms; PR-2 was seed 0, one pair. Signature claims are quoted only across this replication set. |

Same guardrails as PR-2: every write through `learn_local`, provenance
required, runtime-verified vote replication, no engine modification, no
detector refit, clean negative control per configuration.

### PR-3c — classifiers + shadow governance (analysis + shadow readout)

1. **Mode classifier (H1):** logistic (or equivalently simple) classifier
   over the six existing label-free signals, contra-wrong vs stale-wrong,
   fit on train epochs of one seed-class config, scored on held-out epochs
   AND unseen seed/class configs AND the jitter and mixed arms (the #86
   epoch-split discipline, extended across protocols). Also report the
   3-way version including correct, since deployment never knows a probe
   is wrong.
2. **Fork-event classifier (H2):** input = write-time observables only
   (event: pre-write sim, payload cosine, incumbent `hit_counts` /
   `last_seen` / record count, write sequence position; timeline: whether
   subsequent writes to the same key region carry the old payload, the new
   payload, or both). Target = registry ground truth (§7), with one-shot
   events expected **ambiguous**. No registry label may appear in the
   feature set (pinned by test). Features drawing on audited fields are
   conditional on the PR-3a audit.
3. **Shadow governance (H3):** counterfactual readout computed in the
   driver from the logged top-k slot composition — the deployed
   `forward()` is never altered (the PR-2a replication machinery already
   verifies vote fidelity at runtime; interventions are applied to the
   replicated vote only). Policies, each vs no-governance, on every arm:
   * `none`: no governance — the mandatory baseline row;
   * `observe-only`: forks detected and logged, **no intervention**. This
     is the explicit "fork detection is observation, not governance"
     policy, and the default routing for unresolved/ambiguous forks; its
     value is measured (it should match `none` on accuracy while making
     latent fork load visible);
   * `entropy-abstain`: abstain on the frozen #87 two-axis score — the
     existing-detector-as-governance baseline a mode-conditioned policy
     must beat to justify its complexity;
   * `abstain-tie`: abstain when near-tied conflicting decode classes
     split the vote (margin < δ among fork-witness slots);
   * `recency-naive`: exclude the older side of any detected fork
     (pre-registered prediction: fixes stale boundary, elects the
     hallucination in contra);
   * `quarantine-naive`: exclude *both* sides of any detected fork, fall
     back to remaining support (prediction: contains contra, breaks
     superseded keys entirely);
   * `downweight-graded`: age-weighted penalty on older fork sides
     instead of exclusion — the soft variant between `none` and
     `recency-naive`;
   * `mode-conditioned`: H2 classifier routes — supersession → deprecate
     or downweight older side; contradiction → quarantine both;
     ambiguous/unresolved → observe-only (primary) and abstain
     (secondary variant), both reported.
   Metrics per arm and policy: held-out accuracy, stale-boundary
   correction rate, hallucination-election rate on contra keys, abstention
   cost, latent stale reachability after governance. The deliverable is a
   **shadow-governance comparison table** (arm × policy × metric); no
   deployed retrieval change is in scope for PR-3, and none would be
   proposed without this table existing first.

## 6. Detector features to compute (all label-free at inference)

Read-time, per probe (new columns): co-resident disagreement count
(distinct decode classes among surviving top-k within sim-window of top-1),
fork-witness flag (∃ slot pair with key sim above a near-duplicate
threshold and disagreeing decodes), tie score (vote-mass split between top
two decode classes among near-tied slots), top1-slot maturity
(`hit_counts`), top1-slot write recency (`last_seen` delta), top1-slot
record count, age spread between the two sides of a witnessed fork.
Maturity/recency/lineage features are conditional on the PR-3a audit.

Write-time, per fork event (new `fork_events.csv`): see H2 input list.

## 7. Labels

Two grades, kept distinct throughout: **ground-truth labels** (derivable
from the injection registry / protocol — the only labels classifiers are
scored against) and **observational labels** (computed from engine state or
longitudinal outcomes; descriptive, never used as classifier ground truth
where the protocol cannot certify them).

* **Probe level** — unchanged PR-2a flags/precedence, plus
  `governance_*` counterfactual outcome columns from the shadow readout.
* **Slot level** — new `per_slot.csv`, one row per occupied slot per probe
  epoch: slot id, epoch, decode, `hit_counts`, `last_seen`, record count,
  evicted/reused flag, and a role label:
  * ground-truth roles (registry-certified): `current-supported` (clean or
    current-fork support for live ground truth), `stale-superseded`,
    `contradiction-candidate` (forked contra slot), `duplicate-equivalent`
    (slot formed by protocol re-writes of the same key with an agreeing
    payload — the clean epoch-rewrite gives this ground truth for free);
  * observational roles: `merge-candidate` (slot whose provenance shows
    absorbed conflicting writes — only instantiable in the soft-payload
    arm), `unresolved-ambiguous` (slot on either side of a one-shot
    conflict, where the protocol itself certifies that the evidence is
    insufficient).
  This is what PR-2 lacked: slot-level ground truth to score slot-targeted
  governance (does deprecation hit the right slots?) rather than only
  probe-level outcomes.
* **Fork-event level** — new `fork_events.csv`: event observables plus
  * ground-truth event class: `supersession`, `contradiction`,
    `duplicate-rewrite`, `one-shot-ambiguous` (the protocol certifies the
    ambiguity, so "ambiguous" is itself ground truth there);
  * reserved observational classes, **not instantiable in PR-3b** and
    therefore not scored: `payload-drift`, `key-drift` — named now so the
    schema does not need breaking changes when a drift-crossed study
    (PR-4 territory) introduces them, but no PR-3 classifier may claim to
    detect them.
* **Fork-resolution level** — observational, longitudinal, per fork group:
  computed from the epochs after the event, one label per group per run:
  `later-dominates`, `old-reasserts`, `persistent-split`,
  `collapse-to-wrong`, `collapse-to-current`, `unresolved` (run ended
  inside the tie window). These describe what the memory *did* with the
  fork absent intervention — the natural-history baseline any governance
  effect is measured against — and are never classifier targets.

## 8. Tests (hermetic, synthetic, gate the cache runs as in PR-2)

* **invariant audit (PR-3a):** read/inference paths leave `last_seen`,
  `hit_counts`, `usage`, `slot_records` bit-identical — the pinned form of
  §1's code reading; doubles as `last_seen` write-only semantics pin.
* mixed arm: contra and stale labels coexist, overlap reported, neither
  folded; clean control still fires zero flags.
* one-shot: tie regime persists across all post-boundary epochs (the PR-2c
  prediction, pinned).
* soft payloads: phase-2 write takes the merge path (absorbed, no fork)
  and the slot keeps decoding A while ground truth is B → STALE_STRICT;
  slot carries the observational `merge-candidate` role.
* key jitter: boundary margin grows monotonically with ε.
* per_slot.csv roles consistent with registry sets at every epoch;
  eviction/reuse drops the role; ground-truth and observational roles
  never conflated in one column value.
* fork-event features contain no registry-derived values (schema-level
  check).
* fork-resolution labels recomputed from the per-epoch slot tables match
  the driver's online accounting.
* shadow readout with policy `none` reproduces the deployed vote exactly
  (extends the existing runtime verification); `observe-only` matches
  `none` on every elected class.
* governance policies change only the replicated vote, never engine state.

## 9. Acceptance criteria (pre-registered)

* **PR-3a** is reporting, not pass/fail: audit outcome and trigger
  outcomes recorded as-is. The audit gates the *feature set*, not the PR.
* **H1 success:** held-out mode-classification AUC ≥ 0.80 (contra-wrong vs
  stale-wrong) that survives ALL of: the unseen seed/class split, the
  key-jitter arm, and the mixed arm. **Failure:** ≤ 0.60 anywhere in that
  set, or collapse under jitter (classifier was reading the exact-tie
  artifact). **Ambiguous:** 0.60–0.80, or pair-discrimination that does
  not survive the 3-way (with correct) setting → more seeds/configs before
  any architecture claim.
* **H2 success:** ≥ 0.90 accuracy on repeated-write fork events
  (supersession vs contradiction vs duplicate-rewrite) with one-shot
  events routed to ambiguous at a calibrated rate; **failure:** systematic
  confident misclassification of any repeated-write class; one-shot events
  landing confidently in either conflict class is a failure even if
  headline accuracy looks good.
* **H3 — comparative, with explicit non-inferiority.** Judged from the
  shadow comparison table against the full baseline set
  {`none`, `observe-only`, `entropy-abstain`, `recency-naive`,
  `quarantine-naive`}:
  * **success:** `mode-conditioned` improves its target-mode metric
    (stale-boundary correction ≥ 80% of errors; and/or contra error
    containment) while remaining **non-inferior on the other mode** —
    hallucination-election rate and other-mode accuracy within 1 pp
    absolute of `none` — and strictly dominating every baseline on the
    joint (stale-fix, contra-harm) pair, replicated across seeds;
  * **informative failure:** a naive baseline (including
    `entropy-abstain`) matches mode-conditioned governance → the
    classification stage is unjustified complexity; report that plainly;
  * **failure:** no policy achieves target-mode improvement with
    other-mode non-inferiority → read-time governance insufficient;
    escalate to write-path governance (refuse/route the fork at write
    time) as the PR-4 question;
  * **ambiguous:** dominance that does not replicate across seeds.
* Pre-registered directional predictions (falsifiable): recency-naive
  elects the hallucination on contra forks; quarantine-naive zeroes
  accuracy on superseded keys; one-shot tie persists indefinitely;
  `observe-only` matches `none` on accuracy.

## 10. Risks and confounds

* **Signature-as-artifact (the central risk).** PR-2c's margin-0 signature
  is partly protocol (identical re-written keys); PR-2c's collapse-axis
  orientation rests on n=60 from one epoch/seed/class pair; PR-2b's
  low-entropy signature was measured under a monotone fork ramp. The
  jitter, replication, and mixed arms exist to separate mode properties
  from protocol fingerprints; claims that fail there are reported as
  artifacts, not rescued.
* **Stationarity.** Both PR-2 arms exclude drift by design; BLENDED
  cross-talk (the fork-becomes-blend channel, 9 rows in PR-2b) is out of
  scope. A mode classifier validated here may not survive drift —
  explicitly deferred, not claimed. `payload-drift`/`key-drift` event
  classes stay reserved-observational for the same reason.
* **Invariant-audit failure.** If any of `last_seen` / `hit_counts` /
  `usage` / `slot_records` is mutated by a read path, every feature built
  on it is invalid: the audit runs first (PR-3a) and the feature set is
  cut to the surviving fields rather than the theory being forced.
* **Label circularity.** Detector/classifier features must be computable
  from engine state alone; registry ground truth appears only as targets.
  Enforced by test, not convention.
* **One-shot ambiguity is real, not a bug.** Without source identity or an
  explicit supersession assertion, a single conflicting write is
  undecidable. PR-3 measures the size of that undecidable region and
  routes it to observe-only/abstain; closing it requires provenance trust
  metadata (source identity, explicit deprecation markers) — a write-API
  design question, deliberately out of scope here. Minimal metadata answer
  so far: write order + slot maturity + lineage (present, pending audit)
  suffice for repeated-write cases; one-shot cases need metadata the
  engine does not have.
* **Shadow-readout fidelity.** Counterfactual accuracy is only meaningful
  if the policy-`none` shadow vote is bit-identical to deployment;
  runtime-verified, raise-on-mismatch (PR-2a pattern).
* **Taxonomy overreach.** The §7 taxonomies are scaffolding for this
  study, not an ontology claim; labels whose ground truth the protocol
  cannot certify are kept observational, and reserved classes are not
  scored. If PR-3b data shows a label is not cleanly instantiable, it is
  weakened or dropped in the result memo rather than defended.

## 11. Smallest safe implementation path

1. **PR-3a** — invariant-audit test + `benchmarks/score_frozen_detector.py`
   + result memo. The audit is a hermetic test (no cache); the scoring is
   pure analysis of existing CSVs and closes the pre-registered trigger
   question already promised in SCHEMA.md. No new cache runs, no schema
   change. Land first; PR-3b's feature plan is finalized against the audit
   outcome.
2. **PR-3b** — driver extensions (mixed / one-shot / jitter / soft-payload
   arms, `per_slot.csv`, `fork_events.csv`, fork-resolution accounting),
   SCHEMA.md additions, hermetic tests; then the gentoo cache runs
   (compute on gentoo, verify analyzer byte-identical on darwin, per
   host-role convention). Result memo per PR-2 format, including which
   PR-2 signatures survived replication.
3. **PR-3c** — `benchmarks/analyze_fork_governance.py` (mode classifier,
   fork-event classifier, shadow-governance comparison table) + tests +
   result memo answering H1/H2/H3 against §9.

Each stage is independently falsifiable and lands on main before the next
begins (follow-on PRs base on main). Engine code is untouched throughout;
the shadow comparison table is a hard prerequisite for ever proposing a
deployed retrieval change; if H3 fails, write-path governance becomes the
explicitly justified PR-4.
