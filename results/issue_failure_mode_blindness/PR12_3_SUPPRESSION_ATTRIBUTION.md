# PR-12.3 — Suppression attribution + witness-window dual-presentation for the one-shot harm class (pre-registration DRAFT)

Date: 2026-07-05. Main @ `ab35874`. **Status: DRAFT — not committed;
awaiting explicit review and authorization before any branch, commit, or
implementation.** Harness-track only; every gate is defined in full here
(nothing inherited from PR-12.1/12.2 by interpretation). PR-12.1
(`reshape-negative`) and PR-12.2 (`pending-negative`) stand unmodified —
gates, thresholds, outputs, §9/§12 untouched; neither is reinterpreted
as a pass; C1/C2/C3 are not revived. PR-10's merge-abstain remains the
only certified reader contract. FAM-core is untouched at every layer.

## 1. Hypothesis

On the **one-shot harm class** — the closure memo's residual of 3,834
wrong none-rows and 32 permanently-ambiguous pairs per cell, where
PR-12.2 measured pending-led rows to be predominantly wrong (pairD
one-shot: 203 pending-led, 153 wrong; pairB: 340, 330 wrong) and the
economics of acting on them to be nearly free (dual-presentation mass on
correct traffic 0.021 / 0.004; candidate-attributable suppression ≈ 0)
— a **probe-local, width-bounded** dual-presentation candidate set can
clear the truth-containment floor that PR-12.2's *pair-derived* sets
failed (D1 0.203 / 0.494, D2 0.160 / 0.493 vs the 0.5 floor), while
candidate-attributable suppression and presentation mass each stay
within 5% of correct traffic per gated cell, adverse-state visibility is
preserved, and non-pending dispositions are byte-untouched. Mechanism
basis, committed: PR-12.2 §12 showed the wrong rows' *leading pairs* are
frequently not the probe's true-class fork — the failure is set
construction, not selectivity — while the frozen scorer's witness window
(surviving candidates within 0.05 raw cosine of the surviving top-1,
spanning ≥ 2 decode classes; `WITNESS_SIM_WINDOW` frozen since PR-2) is
exactly the probe's own read-time tie set, and the one-shot signature is
an exact vote tie.

## 2. Target mechanism, and the pre-registered scope decision

Target: PR-12 mechanism (d) — pending-led served answers
(`led_pending_ambiguous`) — on the **one-shot harm class**. Frozen and
out of scope: mechanisms (a)/(c), superseded handling, PR-10 abstention
pass-through (all at the PR-12 prototype disposition), and every
FAM-core layer.

**Scope decision, stated before any run.** The contra cells' pending-led
correct mass is ~11% (242/2,136 pairD, 256/2,476 pairB; committed in
`pr12_2/pending_scan.json`). At 5%/5% ceilings this is a structural
double-bind: dual-presenting them all fails any 5% mass ceiling
(PR-12.2 D1: 0.113/0.103), and escalating most of them fails any 5%
attribution ceiling (PR-12.2 D2 escalated 285/303 and 292/303 there —
≈ 0.11 of correct traffic). Gating contra cells on candidate economics
would therefore pre-determine `attribution-negative` by gate
composition — the exact design flaw PR-12.2 §12 recorded. Accordingly:
**candidate-economics gates (G-A, G-M, G-T) apply to the one-shot
harm-class cells; the contra cells carry the same measurements
report-only**, with the double-bind arithmetic above as the recorded
justification. Visibility and regression gates (G-V, G-R) remain hard on
**every** cell. This narrows PR-12.3's claim to the one-shot class —
recorded openly as a coverage reduction, not hidden: contra-arm
pending-led rows remain ungoverned by any candidate here, and their
harm class (contra-wrong, 86% post-resolution) is quarantine-frozen
territory in any case.

## 3. Allowed intervention family

Parameter-free disposition shapes for pending-led served answers, using
only evidence the committed artifacts already carry:

* **W1 — dual-present-all, witness-window set.** Every pending-led
  served answer dual-presents. The candidate set is **probe-local**:
  the decode classes of the probe's witness window (surviving
  candidates within 0.05 raw cosine of the surviving top-1, iff they
  span ≥ 2 decode classes — the frozen scorer's `fork_witness`
  definition, reimplemented artifact-side), union the deployed answer.
  If the witness window is empty, fall back to the PR-12.2 D2 set (the
  leading ambiguous pair's counterpart classes), recorded per row as
  `fallback=pair_counterpart`.
* **W2 — age-gated, witness-window set.** PR-12.2 D2's never-resolving
  selectivity (a pair still ambiguous strictly past its onset epoch;
  lookahead-free; exact under the committed max-lag-1 bound) with W1's
  candidate-set construction; fresh-ambiguity rows keep the prototype
  escalation.

**Width bound (both candidates, by construction):** the presented set
is at most **3 decode classes total** (deployed + ≤ 2 alternatives),
selected by descending surviving vote mass per class (deterministic,
label-free); ties in vote mass are broken by ascending decode-class
index (fixed, label-free, not a new observable). Truncation counts are
reported per cell — no silent caps. The bound exists because the panel
protocol has ~5 decode classes per cell: an unbounded set saturates
truth-containment trivially and the claim would be empty. **This bound
is pre-registered here and is not tunable after any run** — widening or
narrowing it after seeing G-T results is a threshold-motion violation
under §4, not a permitted refinement.

**Empty-set scoring.** A pending-led row has a leading ambiguous pair
by construction (that is what `led_pending_ambiguous` means); if the
fallback pair-counterpart set is nonetheless empty for some row
(instrumentation fault, not evidence), that row counts as a **G-T
failure** for that row — it is never excluded from the containment
denominator. No row is dropped from scoring for having a small or
empty candidate set; a small set only makes containment harder to
achieve, never easier.

## 4. Forbidden interventions

* Any FAM-core change; any change to committed `pr12/`, `pr12_1/`,
  `pr12_2/` artifacts or to the PR-12.1/12.2 memos beyond nothing at
  all (they are closed).
* Any change to non-pending-led dispositions (kill condition; counter
  comparison per G-R).
* Reviving C1/C2/C3 in any form; any quarantine-led disposition change.
* Lookahead in W2's classifier or in witness construction (the witness
  window reads only the probe's own `topk` rows).
* Exceeding the 3-class width bound, or any candidate-set selection
  rule other than the pre-registered vote-mass ordering.
* Registry labels anywhere except scoring (G-T), exactly as in every
  committed scorer.
* Threshold motion or gate re-scoping after the first judging run; new
  reason codes; new observables.

## 5. Exact pass/fail gates

Panel (identical cells to PR-12.2; committed inputs; missing input =
kill): control `clean_pairA_s0`; continuity `pairD_stale-soft_s0`;
report-only candidate economics `pairD_contra_s0`, `pairB_contra_s0`;
**gated harm class** `pairD_oneshot_s0`, `pairB_oneshot_s0` (32/32
final-epoch ambiguous pairs, cross-checked). Shapes:
`{prototype, W1, W2}`.

* **G-A — attributable-suppression ceiling (hard on gated cells).**
  Candidate-attributable suppression = pending-led rows the candidate
  escalates or withholds, on correct traffic, / (`n − wrong_none`)
  ≤ **0.05**. Frozen-baseline suppression (quarantine-led + superseded,
  design-frozen per PR-12.2 §12) is reported on every cell and gated
  nowhere — it is not the candidate's to control. Adopted as this
  memo's own bound, not inherited.
* **G-M — presentation-mass ceiling (hard on gated cells).**
  Dual-presented items on correct traffic / (`n − wrong_none`)
  ≤ **0.05**. Report-only on contra cells (§2). Clean control must be 0
  structurally.
* **G-T — truth-containment floor (hard on gated cells).** Over
  pending-led **wrong** rows the candidate dual-presents: fraction
  whose true label is among the presented (width-bounded) classes
  ≥ **0.5** per cell. Registry scoring-only. The §1 committed baseline
  it must beat: 0.203/0.494 (D1) and 0.160/0.493 (D2) with pair-derived
  sets.
  **Chance-baseline control (containment inflation, pre-registered
  here — this is the reason the 0.5 floor alone is not sufficient at
  width 3 over a ~5-class alphabet).** For every dual-presented wrong
  row, `chance_i = width_used_i / n_decode_classes(cell)` (both
  quantities already available from committed panel/decode data — no
  new observable); the cell's chance-baseline rate is the mean of
  `chance_i` over those rows. A cell's G-T only counts as a **genuine**
  pass toward `attribution-evidence-GO` if the measured containment
  rate exceeds that cell's chance-baseline rate by **≥ 0.15 absolute**.
  A cell that clears 0.5 without clearing its chance-baseline-plus-0.15
  margin is reported as `width-saturated` in the scan output and does
  **not** count as a G-T pass for that cell — the witness window must
  be shown to outperform a same-width random class draw, not merely
  outperform a fixed number picked before the class alphabet's size was
  taken into account.
* **G-W — width integrity (hard, every cell).** Zero rows exceed 3
  presented classes; selection order is vote-mass-descending with the
  §3 tie-break; truncation, `fallback=pair_counterpart` counts, the
  full-width-usage fraction, and the chance-baseline rate (G-T) are all
  reported per cell.
* **G-V — visibility (hard, every cell).** V1: every pending-led row
  keeps `reason_code = led_pending_ambiguous` + evidence pointer, zero
  `no_adverse_flag` downgrades. V2: every compiled pending-led item is
  marked (`unresolved_notice` or `unresolved_tie`); unmarked
  wrong-in-prompt ≤ prototype per cell. V3: contradiction-pair and
  ambiguous-pair review queues (full payload: identity, never-resolving
  classification, per-side counts, exemplars, explicit `no_led_rows`)
  identical across shapes. V4: invariants I1–I7 including
  `certified`-string containment.
* **G-R — regression (hard, every cell).** Continuity anchors (300
  certified abstentions; 375 = 292 + 83, zero escapes); non-pending
  counters identical to prototype; control zero-adverse; PR-12 base
  byte-gate green before and after; committed `pr12/`, `pr12_1/`,
  `pr12_2/` outputs byte-untouched. **Prototype byte-no-op vs the
  committed `pr12_2/prototype/<cell>/`, with the audit-packet
  comparison rule pre-registered here** (the E1 lesson, adopted up
  front): memory packet and decision table strictly byte-identical;
  audit packet identical record-by-record after excluding only the
  per-record `policy_version` field — no other exclusion, since
  PR-12.3 introduces no new appended record type beyond those already
  in the 12.2 baseline.

**Verdict vocabulary (exactly one):**
`attribution-evidence-GO(<candidates>)` — at least one candidate passes
every hard gate on every cell where that gate applies;
`attribution-negative` — none does; `attribution-blocked` —
instrumentation contradiction (§7). **Scope of every verdict:**
offline-simulator evidence only. **Downstream-use boundary:** no
verdict — `GO` included — may be claimed as: certification of
mechanism (d) globally (beyond the two gated one-shot cells named in
§5); direct suitability for agent prompting, promotion to any policy
version, memory ingestion or write-back of dual-presented content, or
any autonomous downstream use of the emitted packets; or any FAM-core
policy, threshold, or reader-contract change — PR-10's merge-abstain
remains the only certified reader contract, unmodified by any outcome
here. Each of the above requires its own pre-registration. A `GO` here
additionally claims nothing about contra-arm pending-led rows (§2
scope decision).

## 6. Predictions and falsification analysis (inference, recorded before the run)

* **G-A and G-M are expected to pass structurally on the gated cells**
  (committed one-shot economics: mass ≤ 0.021, attributable
  suppression ≈ 0 for W1; W2's fresh-window escalations were 74/66
  rows, mostly on wrong traffic). The pre-registration is honest about
  this: the *only* genuinely open gate on the gated cells is **G-T** —
  PR-12.3 is a single-question experiment by design, and that is
  recorded rather than padded.
* **G-T, the open question.** For: the one-shot signature is an exact
  vote tie, so the witness window should contain both tie parties'
  classes on direct one-shot rows — precisely the rows where PR-12.2's
  leading-pair sets pointed at the wrong fork. Against: wrong rows that
  are unpaired/collateral may have an empty or single-class witness
  window (fallback to the failed pair-derived set), and the 3-class
  width bound may truncate the true class on multi-class ties. Committed
  prior on the failure side: PR-11.1 recorded that soft-arm harm is
  mostly outside the fork-pair structure; whether one-shot wrong rows'
  ties are probe-visible is exactly what this measures. Either outcome
  is informative: a pass gives the harness its first truth-bearing
  disposition for the harm class; a fail closes mechanism (d) the way
  PR-9B closed write-event authority — the residual joins the
  accepted column.
* **Key assumption named:** witness-window reconstruction from
  committed `topk` artifacts (surviving rows, 0.05 window, ≥ 2 decode
  classes) exactly mirrors the frozen scorer's `fork_witness`; the
  implementation must pin this by comment and the anchors must hold. A
  mismatch is instrumentation (kill), not evidence.

## 7. Kill conditions (any → `attribution-blocked`)

1. Missing committed input; router/hazard cross-check failure (pairs,
   merges, rows, ambiguous counts incl. 32/32 one-shot).
2. PR-12 base byte-gate red before or after; any byte drift in
   committed `pr12/`, `pr12_1/`, `pr12_2/`.
3. Prototype not byte-identical per the §5 pre-registered comparison
   rule.
4. Certified-abstain set ≠ merge-led set on any cell.
5. Any candidate diff outside pending-led rows.
6. Width-bound violation, non-pre-registered selection order or
   tie-break, or lookahead in W2/witness construction.
7. Review-queue set or payload differing across shapes.
8. "certified" outside permitted fields.
9. Containment inflation: a `GO` claimed on a gated cell whose G-T pass
   does not clear its chance-baseline-plus-0.15 margin (§5) — such a
   cell is `width-saturated`, not a pass, and cannot be counted toward
   `attribution-evidence-GO`.

## 8. Expected artifacts and filenames

* This memo, committed before any run:
  `results/issue_failure_mode_blindness/PR12_3_SUPPRESSION_ATTRIBUTION.md`
  (results appended as §9+, append-only).
* Implementation (separately authorized): `--scan12-3` +
  `--shape12-3 {prototype|W1|W2}` in `harness/harness_boundary_sim.py`;
  a `scan12_3` block in `harness/harness_policy.json` (policy version
  `pr12.3-scan-0.1`); prior scan blocks untouched.
* Per (shape × cell) under
  `results/issue_failure_mode_blindness/pr12_3/<shape>/<cell>/`: the
  three standard files (audit packets carry both review-queue record
  types plus per-row `fallback`/truncation markers inside the decision
  evidence). 54 artifact files.
* Gate scoring:
  `results/issue_failure_mode_blindness/pr12_3/attribution_scan.json` —
  every G-A/G-M/G-T/G-W/G-V/G-R check, frozen-baseline suppression
  report, contra report-only economics, exposure report, both byte-gate
  results, verdict; every §9 number recomputable from it.
