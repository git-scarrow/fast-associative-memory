# PR-12.1 — mechanism (c) disposition re-shaping (pre-registration)

Date: 2026-07-04. Main @ `adb3aee`. **Status: pre-registration (§0–§8)
committed BEFORE any scan run; results appended after (§9+), PR-9.2
appendix discipline.** Harness-track only: no FAM-core file, no engine
byte, no threshold, no committed PR-10/PR-12 artifact is touched; no new
observables are introduced (every candidate below reads evidence the
committed artifacts already carry). PR-10's merge-abstain remains the
only certified reader contract; nothing here is a promotion claim.

**The question.** PR-12 §8 measured that mechanism (c) — quarantine-led
escalation — suppresses 30.7% of correct traffic on the prototype cell,
6× the proposed 5% suppression ceiling, while mechanism (a)'s caveats
cost 1.35%. This memo pre-registers the test of whether quarantine-led
escalation can be re-shaped to cheaper dispositions **without losing
adverse-state visibility** — with "visibility" given a hard, checkable
definition (§2) so the answer is falsifiable rather than rhetorical.

## 0. Constraints (all inherited, none new)

Engine/scorer/driver byte-frozen; harness consumes committed artifacts
only; state taxonomy, reason-code vocabulary shape, audit fields, and
invariants I1–I7 of PR12_HARNESS_BOUNDARY_DESIGN.md §§2–3 unchanged.
Candidates are parameter-free dispositions over evidence the router
already emits. Mechanisms (a) (merge-support caveat), (d) (pending-led
dual-presentation), and the PR-10 abstention pass-through are out of
scope and must be byte-preserved in behavior (§5 G-R). The committed
`pr12/` prototype outputs are immutable; all scan outputs land under
`pr12_1/`.

## 1. Committed baseline (the problem, in numbers)

Prototype cell pairD/stale-soft/s0 (`pr12/pairD_stale-soft_s0/`,
decision economics recomputed from the committed decision table +
per-probe CSV; correct-traffic denominator `n − wrong_none` =
2,939 − 490 = 2,449):

| disposition (prototype) | rows | on correct | on wrong |
|---|---|---|---|
| escalated (quarantine-led 850 + pending-led 42) | 892 | 743 | 149 |
| withheld (superseded-led) | 11 | 8 | 3 |
| shown_with_caveat (mechanism (a)) | 46 | 33 | 13 |
| suppressive total on correct | — | **751 (30.7%)** | — |

Decomposition of the 149 escalated-on-wrong rows — what escalation's
suppression actually buys beyond the other mechanisms: 70 are
stale-lenient, 70 already carry the merge-support flag, and **79 carry
no other flag**. Mechanism (c)'s *unique* suppression benefit on this
cell is therefore 79 wrong rows, purchased with 743 suppressed correct
rows (≈ 9.4 : 1). The contradiction-arm prior is worse and committed:
PR-11.1 P1 (quarantine ∪ deprecate-led abstention) on pairD/contra/s0
suppressed 1,470 of 2,747 rows — 1,024 of them correct — to capture
1,048 of 1,265 contra-wrong; pairB/contra capture 0.889 (849/955). P1
is an upper bound on quarantine-led counts (it adds deprecate-led).

## 2. Definition: adverse-state visibility (hard gate material)

A re-shaped disposition **preserves adverse-state visibility** iff all
of the following hold for every quarantine-led served row:

* **V1 (audit basis retained).** The audit decision and decision-table
  row keep `reason_code = led_quarantined_contradiction` with its
  evidence pointer; zero rows are downgraded to `no_adverse_flag`.
* **V2 (prompt marker).** The memory-packet item for that row carries
  an explicit contradiction marker (caveat text or notice); no
  quarantine-led answer compiles unmarked.
* **V3 (review-queue completeness, with adjudication payload).** The
  set of contradiction pairs emitted for human review is identical to
  the prototype's at pair granularity (28 pairs on the primary cell) —
  reducing *row-level* escalation must not shrink *pair-level*
  adjudication demand. Additionally, each audit-only
  contradiction-pair record must carry enough payload to adjudicate
  from, not merely to count: pair identity (`I`/`O` slots, onset
  epoch, `old_side`/`new_side` decode classes), **affected row counts
  per side** (served rows whose led slot is that party, plus the
  pair's total quarantine-led row attribution), and **stable exemplar
  row IDs per side where available** — deterministic `(epoch,
  probe_index)` query IDs (e.g. first and last led row per side), with
  a side that never leads a served row recorded explicitly as
  `no_led_rows`, never omitted.
* **V4 (invariants intact).** I1–I7 all hold, including the
  `certified`-string containment and the withheld ≠ not_retrieved
  distinction.

What visibility deliberately does **not** require: prompt-level
suppression of the answer. Whether a caveat is adequate protection at
the agent runtime is a behavioral question no artifact in this repo can
answer; it is this memo's named decision-relevant unknown (§6), and the
reason a `reshape-evidence-GO` verdict certifies escalation-reduction
evidence only, never prompt safety (§5, verdict scope).

## 3. Candidates (frozen; all parameter-free, existing evidence only)

For **quarantine-led served answers only** (all other dispositions
unchanged, including pending-led escalation and superseded-led
withholding):

* **C1 — contradiction-caveat.** Compile the answer
  `shown_with_caveat` with an explicit contradiction caveat; the fork
  pair still goes to the review queue (V3). Zero row-level
  suppression from mechanism (c).
* **C2 — dual-present-inline.** As C1, plus the item names the fork
  counterpart's decode class as an explicit alternative (from the
  router pair's `old_side`/`new_side` — already computed). Same
  suppression profile as C1; maximal in-prompt disclosure.
* **C3 — witness-gated escalation.** Escalate only when the
  contradiction is *locally live*: a pair counterpart of the led slot
  is itself a surviving candidate in this probe's support (static
  `pair_counterparts` membership × the probe's own `topk` survivors —
  the co-residence notion the frozen scorer's GuardIndex already
  uses). Otherwise degrade to C1's caveat. Partial suppression,
  targeted at the rows where the disagreement is present in the
  retrieval itself.

## 4. Panel (committed inputs only; no new runs)

| cell | run artifacts | hazard-tier source | role |
|---|---|---|---|
| clean/pairA/s0 | `pr10/governed/per_probe_clean_pairA_s0.*` | `pr3c/per_probe_clean_s0.governance.json` | control (zero-adverse must hold) |
| pairD/stale-soft/s0 | `pr10/governed/per_probe_stale-soft_pairD_s0.*` | `pr6/stale_de/...s0_pairD.governance.json` | continuity with PR-12 |
| pairD/contra/s0 | `pr10/governed/per_probe_contra_pairD_s0.*` | committed `pr4/pr4_geometry_table.json` row (no per-run sibling exists — checked before this pre-registration) | harm-bearing stress (contra_wrong lives here) |
| pairB/contra/s0 | `pr10/governed/per_probe_contra_pairB_s0.*` | `pr3c/per_probe_contra_s0_pairB.governance.json` | geometry diversity (uncompressed pair) |

Any missing/mismatched committed input at implementation time is a stop
condition recorded as a failed design assumption, never worked around
silently.

## 5. Gates (frozen; no discretion at scoring time)

* **G-S — suppression ceiling (hard, per cell, every panel cell).**
  Suppressive dispositions (`escalated` + `withheld`) on correct
  traffic ≤ **0.05** of `n − wrong_none`. This adopts the program's
  precedent ceiling as the harness track's own pre-registered bound
  (on the primary cell: ≤ 122 rows). Clean control must be 0
  structurally.
* **G-V — visibility (hard, per cell).** V1–V4 of §2, all of them.
* **G-R — regression (hard).** PR-12 anchors byte-stable: 300
  certified abstentions pass through; mechanism (a) coverage
  375 = 292 + 83 with zero escapes on the continuity cell; clean
  control zero adverse states; the committed `pr12/` outputs untouched
  and `--check` green; mechanisms (a)/(d) and superseded handling
  behaviorally identical.
* **E — exposure report (report-only, NOT a gate).** Per candidate ×
  cell: wrong served rows entering the prompt, split by
  (stale-lenient, contradictory-lenient, merge-support-flagged, no
  flag) × (caveated, unmarked). Report-only because gating it requires
  the agent-behavior evidence named in §2; recording it is what makes
  a future prompt-safety pre-registration possible.

**Verdict vocabulary (exactly one):**
`reshape-evidence-GO(<candidates>)` — at least one candidate passes
G-S, G-V, and G-R on every panel cell; `reshape-negative` — none does;
`reshape-blocked` — an instrumentation contradiction (missing committed
input, anchor break, invariant failure independent of candidates).

**Scope of every verdict, stated in advance.** PR-12.1 can certify
**reshape evidence only**: that a disposition shape meets the
suppression ceiling and the §2 visibility definition on the committed
panel, at the offline simulator layer. No verdict — `GO` included — is
a runtime prompt-safety claim, a policy promotion, a change to any
policy version, or an extension of any FAM-core certification. Prompt
safety would additionally require the agent-behavior evidence named in
§2 (does a caveat change downstream agent actions?), which no artifact
in this repository provides; any future claim of that kind needs its
own pre-registration with its own gates. A `GO` here may be *cited* by
such a pre-registration; it may not be *substituted* for it.

## 6. Predictions and falsification analysis

*This section is inference, recorded before the scan; §1 is committed
measurement.*

* C1/C2 are expected to pass G-S **structurally** on the primary cell:
  residual suppression = pending-led escalations (≤ 42) +
  superseded-withheld (8 on correct) ≈ 2.0% ≤ 5%. Their genuine
  failure modes are G-V V3 (review-queue completeness is an
  implementation obligation, not a given) and G-S on the **contra
  cells**, where the deprecate-led (superseded) and pending-led
  fractions are unmeasured — if supersession-withholding alone exceeds
  the ceiling there, even zero-escalation candidates fail, which would
  be a finding about superseded handling, not about (c).
* C3 is the genuinely open measurement: the witness-gated fraction of
  the 850 quarantine-led rows is unknown. C3 passes only if
  locally-live contradictions on correct traffic fit within the
  ceiling's remainder; PR-4's slot-granularity falsification is the
  honest prior *against* it.
* The exposure report's expected shape: C1/C2 move ~79 (primary) /
  order-10² (contra, from the §1 prior) previously-suppressed wrong
  rows into the prompt **caveated**. Versus the ungoverned baseline
  (all wrong rows served unmarked) this is strictly more marked; versus
  the prototype it is more exposed. Both comparisons are recorded; the
  memo takes no position on which matters until the behavioral unknown
  is resolved.
* Key assumption carried: pair-granularity review-queue demand (V3) is
  the right unit of "nothing lost" for human adjudication. If review
  actually needs row-granularity exemplars, V3 is too weak — a
  design-assumption risk recorded now, checkable in review-queue UX
  work later.

## 7. Consequences (pre-registered)

* **`reshape-evidence-GO`** → the winning candidate becomes *eligible*
  to be proposed as harness policy version 0.2 in a follow-up
  implementation PR — a separate, reviewable step, not an automatic
  consequence of the verdict. That follow-up remains prototype-level
  simulator policy; PR-12 §8's non-promotion statement stands, and no
  prompt-safety or reader-facing claim may be made until an
  agent-behavior pre-registration addresses the caveat-adequacy
  unknown, which is outside this repo's current evidence (§5, verdict
  scope).
* **`reshape-negative`** → quarantine-led escalation stands as the
  recorded, quantified cost of contradiction visibility at the prompt
  layer; the harness track's next fork is pair-granularity review UX
  or explicit acceptance — not threshold motion, not gate re-scoping.
* **`reshape-blocked`** → fix the instrumentation contradiction,
  re-run; no candidate may be judged from a blocked run.

## 8. Implementation prompt (bounded; the only implementation this memo authorizes)

> Extend `harness/harness_boundary_sim.py` with a
> `--shape {prototype|C1|C2|C3}` flag implementing §3 exactly, and add
> the two contra cells of §4 to `harness/harness_policy.json` (policy
> version `pr12.1-scan-0.1`; hazard sources as tabled, missing input =
> stop condition reported as a failed design assumption). Emit the
> three standard files per (candidate × cell) under
> `results/issue_failure_mode_blindness/pr12_1/<shape>/<cell>/`, plus
> one `reshape_scan.json` recording every G-S/G-V/G-R check and the §5
> E exposure report. The committed `pr12/` outputs must remain
> byte-untouched (`--check` green before and after). Make the review
> queue explicit and checkable: emit contradiction pairs as audit-only
> records carrying the full §2 V3 payload — pair identity, per-side
> affected row counts, and stable `(epoch, probe_index)` exemplar IDs
> per side (explicit `no_led_rows` marker when a side never leads) —
> so V3 is a set-and-payload comparison, prototype vs candidate.
> Analysis-only, darwin, stdlib, no torch, no FAM-core change, no new
> observables, no new states or reason codes, no threshold motion.
> Score the §5 gates with no discretion and append the §9 results
> section to this memo with the verdict. Stop there: no policy-version
> promotion, no agent integration, no additional cells or candidates.
