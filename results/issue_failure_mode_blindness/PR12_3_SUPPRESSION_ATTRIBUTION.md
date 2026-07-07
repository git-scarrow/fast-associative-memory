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

## 9. Results (append-only; §§1–8 are the frozen pre-registration snapshot)

Run date 2026-07-07, this section committed on branch
`feat/pr12.3-witness-window-scan` (append-only; the §§1–8 pre-registration
above — including the DRAFT header, committed unchanged on docs branch
`2ad4bdc`, merged to main `8ab0ee0` — is never rewritten). Implementation:
`--scan12-3` + `--shape12-3 {prototype|W1|W2}` and the `scan12_3` policy
block (policy_version `pr12.3-scan-0.1`, shared prototype baseline
`pr12_2/prototype`) in the two harness files only. Every number below is a
verbatim field of `attribution_scan.json`.

### 9.1 Verdict

**`attribution-evidence-GO(W1,W2)`** — the first GO in the PR-12 series.
Both W1 (dual-present-all) and W2 (age-gated) pass every hard gate on every
cell where that gate applies. `kill_conditions: []` (zero); PR-12 base
byte-gate `base_bytecheck_before = true`, `base_bytecheck_after = true`.

### 9.2 Exact G-A / G-M / G-T on the two gated harm-class cells

Ceilings: G-A ≤ 0.05, G-M ≤ 0.05, G-T ≥ 0.5 floor **and** ≥ chance+0.15.
Denominator for G-A/G-M is correct traffic `n − wrong_none`.

| shape · cell | G-A rate (rows) | G-M rate (rows) | G-T dual_wrong | truth_contained | containment | verdict |
|---|---|---|---|---|---|---|
| W1 · pairD_oneshot_s0 | 0.000000 (0) | 0.020517 (50) | 153 | 141 | 0.921569 | **PASS** |
| W1 · pairB_oneshot_s0 | 0.000000 (0) | 0.004252 (10) | 330 | 330 | 1.000000 | **PASS** |
| W2 · pairD_oneshot_s0 | 0.016414 (40) | 0.004103 (10) | 119 | 109 | 0.915966 | **PASS** |
| W2 · pairB_oneshot_s0 | 0.004252 (10) | 0.000000 (0)  | 274 | 274 | 1.000000 | **PASS** |

G-A/G-M/G-T `pass = true` on all four (cell,candidate) pairs. The wrong-row
sets are the same rows PR-12.2's D1/D2 scored (dual_wrong pairD 153/119,
pairB 330/274); only the candidate-set construction changed, lifting
containment from D1/D2's 0.203/0.494 (pairD/pairB) to 0.92/1.00 — recorded
here as confirmation of §1's "the failure is set construction, not
selectivity."

### 9.3 Chance-baseline values and chance+0.15 thresholds (containment inflation control, §5 / kill-9)

`n_decode_classes = 4` on every cell (decode alphabet {0,1,2,3}, contiguous;
NOT the ~5 the §3/§5 prose assumed — recorded as an observed correction, not
a spec change: the control is defined per-cell from committed decode data,
so it self-adjusts). `chance_i = width_used_i / 4`; cell chance rate = mean
over dual-presented wrong rows. A cell counts as a **genuine** G-T pass only
if containment ≥ chance+0.15; else `width-saturated` (not a pass).

| shape · cell | chance_baseline_rate | chance+0.15 threshold | containment | margin over threshold | genuine | width_saturated |
|---|---|---|---|---|---|---|
| W1 · pairD_oneshot_s0 | 0.526144 | 0.676144 | 0.921569 | +0.245 | true | false |
| W1 · pairB_oneshot_s0 | 0.500000 | 0.650000 | 1.000000 | +0.350 | true | false |
| W2 · pairD_oneshot_s0 | 0.527311 | 0.677311 | 0.915966 | +0.239 | true | false |
| W2 · pairB_oneshot_s0 | 0.500000 | 0.650000 | 1.000000 | +0.350 | true | false |

All four clear their own same-width chance-baseline-plus-0.15 margin;
`floor_pass = margin_pass = genuine = true`, `width_saturated = false`;
kill-9 does not fire.

### 9.4 G-W, V1–V4, G-R, and kill conditions — all pass

* **G-W (width integrity, hard every cell):** `over_bound_rows = 0`,
  `pending_dual_width_gt3 = 0` on every (shape,cell); selection
  `vote-mass-descending, tie-break ascending decode-class index (§3)`. Width
  usage on the gated cells is witness-carried: W1 pairD 165 witness / 38
  fallback-pair-counterpart rows, W1 pairB 330 witness / 10 fallback, W2
  pairD 129 witness / 0 fallback, W2 pairB 274 witness / 0 fallback;
  full-width fractions ≤ 0.14 on the gated cells (truncation never fires on
  them: `truncated_rows = 0`). `empty_candidate_set_rows = 0` everywhere.
* **V1–V4 (visibility, hard every cell):** all `pass = true`; V1/V2
  violations = 0; V2 unmarked ≤ prototype per cell (identical); V3 review
  queues identical across shapes; V4 `certified_leaks = 0`,
  `incomplete_audits = 0` on every cell.
* **G-R (regression, hard every cell):** continuity anchors hold
  (`G-R_anchors` pass — 300 certified abstentions; 375 = 292 + 83, zero
  escapes); `G-R_non_pending_identical` pass on every cell; control
  `G-R_control_zero_adverse` pass (adverse = 0); PR-12 base byte-gate green
  before and after; committed `pr12/`, `pr12_1/`, `pr12_2/` byte-untouched.
  Quarantine-led rows are byte-identical across prototype/W1/W2 under the §5
  audit-packet rule (policy_version excluded): 850/1470/1375/923/296 records
  on pairD_stale-soft / pairD_contra / pairB_contra / pairD_oneshot /
  pairB_oneshot respectively, IDENTICAL on both candidates — direct evidence
  the quarantine-caveat downgrade was not revived (§9.7).
* **Kill conditions 1–9:** none fired (`kill_conditions: []`). Determinism
  re-verified: a second `--scan12-3` reproduces the entire `pr12_3/` tree
  byte-identically and the same verdict.

### 9.5 Contra cells remain report-only — no GO claim, the §2 double-bind confirmed

Per §2 the candidate-economics gates are **not** applied to the contra
cells; the same measurements are carried report-only (`gated: false`,
`role: contra_report_only`). They confirm the pre-registered structural
double-bind exactly — and would have pre-determined `attribution-negative`
by gate composition had §2 not scoped them out:

| shape · contra cell | G-A rate | G-M rate | would-fail-if-gated |
|---|---|---|---|
| W1 · pairD_contra_s0 | 0.000000 | 0.113296 | **G-M** (dual-present-all blows the mass ceiling) |
| W1 · pairB_contra_s0 | 0.000000 | 0.103393 | **G-M** |
| W2 · pairD_contra_s0 | 0.106273 | 0.007022 | **G-A** (escalate-most blows the attribution ceiling) |
| W2 · pairB_contra_s0 | 0.098950 | 0.004443 | **G-A** |

The GO makes **no claim** about contra-arm pending-led rows; that harm class
is out of PR-12.3's scope (§2, §5 downstream boundary).

### 9.6 Frozen-baseline suppression — reported, gated nowhere (§5)

Quarantine-led + superseded suppression is design-frozen (mechanism (a)/(c),
PR-12.2 §12) and gated nowhere; it is reported identically under
prototype/W1/W2: pairD_oneshot 719 (rate 0.295035), pairB_oneshot 272
(0.115646), pairD_contra 1024 (0.479401), pairB_contra 1075 (0.434168),
pairD_stale-soft 716 (0.292364), clean 0. G-A measures only
**candidate-attributable** suppression (§9.2), which is ≤ 0.0164.

### 9.7 Implementation incidents (both caught and resolved before the judging run)

1. **kill-5 caught a forbidden C1/quarantine-caveat inheritance before any
   outcome gate was computed.** W1/W2 initially fell outside the quarantine
   `escalate` allow-list, so quarantine-led rows would have taken C1's
   caveat downgrade (`state:quarantined` 923 → 0 on pairD_oneshot) — an
   unauthorized C1 revival (§4). The kills-1–8 sweep returns
   `attribution-blocked` (`harness_boundary_sim.py:1582`) **before**
   `economics()` is even defined (`:1591`), so G-A/G-M/G-T were never
   computed or printed on that blocked run — no outcome was observed. Fixed
   by adding `"W1","W2"` to the allow-list (`:331`), with the §4 rationale
   pinned in the adjacent comment. Same defect class PR-12.2 hit with D1/D2;
   the recurring lesson: any new pending-led shape must join the quarantine
   escalate allow-list or it silently revives C1.
2. **PR-12.3 counters namespaced under `pr123` to preserve PR-12.2 byte
   identity.** The ten new counters would otherwise have leaked into
   `run_scan12_2`'s wholesale `res["counters"]` dump and modified the
   committed `pr12_2/pending_scan.json` (a §4/§7.2 byte-identity violation).
   They are routed to a separate `pr123` defaultdict returned as
   `res["pr123_counters"]`; `run_scan12_3` scores from a merged view;
   `run_scan12_2` is literally untouched. Re-verified this run: `--scan12-2`
   reproduces byte-identically and its verdict remains **`pending-negative`**
   (G-S FAIL, G-T FAIL 0.159664 / 0.492701, G-M PASS).

### 9.8 Scope and downstream boundary (restated per §5 — nothing here is enlarged)

Offline-simulator evidence only. This GO does **not** certify mechanism (d)
globally (beyond the two gated one-shot cells), nor authorize agent
prompting, promotion to any policy version, memory ingestion or write-back
of dual-presented content, autonomous downstream use of the emitted packets,
or any FAM-core / policy / threshold / reader-contract change. **PR-10's
merge-abstain remains the only certified reader contract, unmodified by this
outcome.** PR-12.1 (`reshape-negative`) and PR-12.2 (`pending-negative`)
verdicts stand unchanged; C1/C2/C3 are not revived.

### 9.9 Emitted artifacts (54 per-(shape × cell) files + the scan report)

Gate report:
`results/issue_failure_mode_blindness/pr12_3/attribution_scan.json`

Per (shape × cell) under `results/issue_failure_mode_blindness/pr12_3/`, the
three standard files `audit_packet.jsonl`, `decision_table.csv`,
`memory_packet.jsonl` (3 shapes × 6 cells × 3 = 54):

```
pr12_3/{prototype,W1,W2}/clean_pairA_s0/{audit_packet.jsonl,decision_table.csv,memory_packet.jsonl}
pr12_3/{prototype,W1,W2}/pairD_stale-soft_s0/{audit_packet.jsonl,decision_table.csv,memory_packet.jsonl}
pr12_3/{prototype,W1,W2}/pairD_contra_s0/{audit_packet.jsonl,decision_table.csv,memory_packet.jsonl}
pr12_3/{prototype,W1,W2}/pairB_contra_s0/{audit_packet.jsonl,decision_table.csv,memory_packet.jsonl}
pr12_3/{prototype,W1,W2}/pairD_oneshot_s0/{audit_packet.jsonl,decision_table.csv,memory_packet.jsonl}
pr12_3/{prototype,W1,W2}/pairB_oneshot_s0/{audit_packet.jsonl,decision_table.csv,memory_packet.jsonl}
```
