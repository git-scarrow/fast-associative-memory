# PR-12.4 — Witness-window replication across seed variation (pre-registration DRAFT)

Date: 2026-07-07. Main @ `aece0d4`. **Status: DRAFT — design-only;
awaiting explicit review and authorization before any implementation,
run, or `pr12_4/` artifact.** Harness-track only; every gate is defined
in full here (nothing inherited from PR-12.1/12.2/12.3 by
interpretation). Standing record, restated up front and unmodified by
any outcome of this memo:

* **PR-12.1 remains `reshape-negative` and PR-12.2 remains
  `pending-negative`** — gates, thresholds, outputs, and memo sections
  untouched; neither is reinterpreted; C1/C2/C3 are not revived.
* **PR-12.3 remains narrow seed-0 evidence**:
  `attribution-evidence-GO(W1,W2)` on exactly two gated one-shot cells
  (`pairD_oneshot_s0`, `pairB_oneshot_s0`), offline simulator only.
  This memo does not enlarge that claim; a PR-12.4 pass would be a
  *separate* replication claim, and a PR-12.4 fail would bound the
  PR-12.3 GO as a seed-0 artifact without altering its recorded verdict.
* **PR-10's merge-abstain remains the only certified reader contract.**
* **FAM-core is untouched at every layer**, and no FAM-core work is
  authorized by this memo or by any verdict it can produce.
* No claim of reader utility, prompting safety, promotion safety,
  memory-ingestion safety, downstream-use safety, or a new reader
  contract is made here or is available as an outcome (§10).

## 1. Hypothesis

The PR-12.3 witness-window result is a property of the mechanism, not
of seed 0. Concretely: on seed-varied replications of the one-shot
harm-class cells (pairs B and D at seeds **s1 and s2**), the frozen W1
and W2 candidates (§4, copied verbatim from PR-12.3 §3) pass the same
hard gate family — candidate-attributable suppression ≤ 0.05 and
presentation mass ≤ 0.05 on correct traffic, truth-containment ≥ 0.5
**and** ≥ that cell's own chance-baseline-plus-0.15 margin computed per
seed/cell from that cell's measured decode alphabet — with visibility
preserved and non-pending dispositions byte-identical to the prototype.
Committed basis at s0: containment 0.9216/1.0000 (W1) and 0.9160/1.0000
(W2) against chance+0.15 thresholds of 0.676/0.650 (PR-12.3 §9.2–9.3).
Falsification is symmetric and pre-accepted: failure of any hard gate
on any gated cell yields `replication-negative`, in which case the
PR-12.3 GO is recorded as seed-0-bounded evidence and nothing
downstream may be proposed on it until the bound is understood.

## 2. Scope

Target: PR-12 mechanism (d) — pending-led served answers
(`led_pending_ambiguous`) — on the **one-shot harm class**, at seeds
**s1 and s2 only** (s0 is closed under PR-12.3 and is not re-judged;
its committed numbers are carried as reference values only). Frozen and
out of scope: mechanisms (a)/(c), superseded handling, PR-10 abstention
pass-through, contra-led and quarantine-led disposition changes, every
FAM-core layer, and every committed artifact under `pr12/`, `pr12_1/`,
`pr12_2/`, and `pr12_3/` (all closed; byte-drift in any of them is a
kill, §7).

**Contra cells: report-only, carried under the PR-12.3 §2 double-bind.**
The recorded arithmetic (pending-led correct mass ~11% of correct
traffic vs 5% ceilings; PR-12.3 §9.5 measured both horns over ceiling:
W1 G-M 0.113/0.103, W2 G-A 0.106/0.099) established that this gate
family cannot pass on the contra cells under the current 5%
attribution/mass ceilings; a contra claim would require a different
pre-registered accounting model or intervention family, neither of
which is proposed here. Accordingly the candidate-economics gates
(G-A, G-M, G-T) apply to the one-shot harm-class cells only; the contra
cells carry the same measurements report-only (`gated: false`,
`role: contra_report_only`), and their s1/s2 economics are recorded as
data for any future, separately registered contra design. Visibility
and regression gates (G-V, G-R) remain hard on **every** cell.

## 3. Seed-input state (verified on main `aece0d4` before this draft; classification A)

Committed seed-varied panel inputs already exist; no generation step is
required. Verified: all six panel run stems exist at s1 and s2 under
`results/issue_failure_mode_blindness/pr10/governed/` with all five
file types (`.csv`, `.fork_events.csv`, `.per_slot.csv`,
`.summary.json`, `.topk.csv.gz`), git-tracked and clean (committed in
PR-10 step 2, `8c953aa`). The `.topk.csv.gz` per seed is the input the
witness-window reconstruction reads, so `fork_witness` reconstruction
is available for every seed; decode alphabets are computable per
seed/cell from the same committed run data (as in PR-12.3 §9.3, which
computed `n_decode_classes = 4` from committed decode data at s0 —
s1/s2 alphabets are measured per cell at scoring time, not assumed).

Pre-registered input mapping (mirrors the committed `scan12_3` policy
block's s0 mapping; recorded decisions marked):

| cell (per seed s ∈ {s1, s2}) | run_stem (`pr10/governed/`) | hazard_governance |
|---|---|---|
| `clean_pairA_<s>` | `per_probe_clean_pairA_<s>` | `pr3c/per_probe_clean_<s>.governance.json` (unsuffixed per-seed file, same source family as s0) |
| `pairD_stale-soft_<s>` | `per_probe_stale-soft_pairD_<s>` | `pr6/stale_de/per_probe_stale-soft_<s>_pairD.governance.json` (verified present for s1, s2) |
| `pairD_contra_<s>` | `per_probe_contra_pairD_<s>` | `pr4/pr4_geometry_table.json#governance#pairD/contra/<s>` (verified present, `merge-abstain` populated) |
| `pairB_contra_<s>` | `per_probe_contra_pairB_<s>` | `pr4/pr4_geometry_table.json#governance#pairB/contra/<s>` — **recorded decision:** pr3c's pair-suffixed contra governance exists only at s0 (`per_probe_contra_s0_pairB.governance.json`); at s1/s2 the committed hazard evidence for (pairB, contra, seed) lives in the pr4 geometry table, the same cross-run pattern the committed policy already uses for `pairD_contra_s0`. Cross-checked against the rebuilt router per §5 G-R; mismatch is a kill, not evidence. |
| `pairD_oneshot_<s>` | `per_probe_stale-oneshot_pairD_<s>` | `pr4/pr4_geometry_table.json#governance#pairD/oneshot/<s>` (verified present) |
| `pairB_oneshot_<s>` | `per_probe_stale-oneshot_pairB_<s>` | `pr4/pr4_geometry_table.json#governance#pairB/oneshot/<s>` (verified present) |

Panel: **12 cells** = the six cell roles above × {s1, s2}. **Gated harm
class (4 cells): `pairD_oneshot_s1`, `pairB_oneshot_s1`,
`pairD_oneshot_s2`, `pairB_oneshot_s2`.** Control and continuity cells
(`clean_pairA_*`, `pairD_stale-soft_*`) and contra cells per §2.
Shapes: `{prototype, W1, W2}`. Missing committed input = kill.

## 4. Intervention family — frozen, copied verbatim from PR-12.3 §3

No new shapes; introducing any shape beyond the three below is a kill.

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
  lookahead-free) with W1's candidate-set construction; fresh-ambiguity
  rows keep the prototype escalation. **Seed-variation note, recorded
  up front:** PR-12.2 established the classifier exact under the
  committed **s0** max-lag-1 bound; the resolution-lag bound at s1/s2
  is an empirical property measured and reported per seed
  (`max_resolution_lag` report field). The classifier itself is
  unchanged and lookahead-free by construction; whether the exactness
  property carries is a reported finding, not a gate.

**Width bound (both candidates, by construction):** the presented set
is at most **3 decode classes total** (deployed + ≤ 2 alternatives),
selected by descending surviving vote mass per class (deterministic,
label-free); ties in vote mass broken by ascending decode-class index.
Truncation counts reported per cell — no silent caps. **This bound is
pre-registered here and is not tunable after any run** — widening or
narrowing it after seeing G-T results is a threshold-motion violation
under §6, not a permitted refinement.

**Empty-set scoring.** A pending-led row has a leading ambiguous pair
by construction; if the fallback pair-counterpart set is nonetheless
empty for some row (instrumentation fault, not evidence), that row
counts as a **G-T failure** for that row — never excluded from the
containment denominator. No row is dropped from scoring for having a
small or empty candidate set.

## 5. Exact pass/fail gates (copied from PR-12.3 §5; every deviation explicitly marked)

* **G-A — attributable-suppression ceiling (hard on gated cells).**
  Candidate-attributable suppression = pending-led rows the candidate
  escalates or withholds, on correct traffic, / (`n − wrong_none`)
  ≤ **0.05** per gated cell. Frozen-baseline suppression (quarantine-led
  + superseded) reported on every cell and gated nowhere. Report-only on
  contra cells (§2).
* **G-M — presentation-mass ceiling (hard on gated cells).**
  Dual-presented items on correct traffic / (`n − wrong_none`)
  ≤ **0.05** per gated cell. Report-only on contra cells. Clean control
  must be 0 structurally.
* **G-T — truth-containment floor (hard on gated cells).** Over
  pending-led **wrong** rows the candidate dual-presents: fraction whose
  true label is among the presented (width-bounded) classes ≥ **0.5**
  per cell, **and** the chance-baseline control: for every dual-presented
  wrong row, `chance_i = width_used_i / n_decode_classes(cell)` with
  `n_decode_classes` **measured per seed/cell from that cell's committed
  decode data** (never assumed; the PR-12.3 §9.3 lesson); the cell's
  chance-baseline rate is the mean of `chance_i` over those rows; a cell
  counts as a **genuine** G-T pass only if measured containment exceeds
  chance-baseline by **≥ 0.15 absolute**, else it is reported
  `width-saturated` and does not count as a pass.
  **Deviation marked — vacuous-cell rule (new, forced by seed
  variation):** the s1/s2 wrong-row populations are unmeasured; a gated
  cell whose candidate dual-presents **zero** wrong rows has an undefined
  containment rate and is reported `vacuous` — it neither passes nor
  fails G-T, and a `replication-GO` cannot be claimed unless **all four
  gated cells** produce genuine (non-vacuous, non-width-saturated) G-T
  passes. Vacuity on any gated cell caps the verdict at
  `replication-negative` (recorded as `vacuous`, distinguished from a
  measured failure in the scan output).
* **G-W — width integrity (hard, every cell).** Zero rows exceed 3
  presented classes; selection order vote-mass-descending with the §4
  tie-break; truncation, `fallback=pair_counterpart` counts, full-width
  usage fraction, and chance-baseline rate reported per cell.
* **G-V — visibility (hard, every cell).** V1: every pending-led row
  keeps `reason_code = led_pending_ambiguous` + evidence pointer, zero
  `no_adverse_flag` downgrades. V2: every compiled pending-led item is
  marked (`unresolved_notice` or `unresolved_tie`); unmarked
  wrong-in-prompt ≤ prototype per cell. V3: contradiction-pair and
  ambiguous-pair review queues (full payload) identical across shapes.
  V4: invariants I1–I7 including `certified`-string containment.
* **G-R — regression (hard, every cell).**
  **Deviation marked — s0 anchor constants inapplicable as numbers:**
  PR-12.3's continuity anchors (300 certified abstentions; 375 = 292 +
  83; 32/32 final-epoch ambiguous pairs) are seed-0 measurements and do
  not transfer as constants. They are retained **structurally**: zero
  certified-abstention escapes; certified-abstain set = merge-led set on
  every cell (kill-4); router/hazard cross-check consistency on pairs,
  merges, rows, and ambiguous counts per seed (kill-1); the measured
  per-seed counts are recorded in the scan output and become the
  registered anchors for any future s1/s2 work.
  **Deviation marked — prototype baseline definition:** no committed
  prototype artifacts exist at s1/s2 (`pr12_2/prototype/` is s0-only),
  so a byte-comparison against a committed prototype is impossible.
  Instead: the prototype shape is generated in-run per seed as the
  intra-run baseline; non-pending counters and quarantine-led audit
  records must be identical to prototype within each cell (audit-packet
  comparison record-by-record excluding only the per-record
  `policy_version` field — the E1/§9.7 rule, adopted verbatim); and
  byte-reproducibility is established by the §8 double-run requirement
  rather than by comparison to a prior commit.
  Unchanged and hard: control cells zero-adverse; PR-12 base byte-gate
  green before and after; committed `pr12/`, `pr12_1/`, `pr12_2/`,
  `pr12_3/` outputs byte-untouched.

## 6. Predictions and falsification analysis (inference, recorded before any run)

* G-A/G-M are expected to pass on the gated cells if the one-shot
  economics generalize (s0: mass ≤ 0.021, attributable suppression
  ≤ 0.0164) — but unlike PR-12.3, this is **not** structurally assured,
  because the s1/s2 pending-led populations are unmeasured. PR-12.4 has
  two genuinely open gate families, and that is recorded rather than
  padded: the economics gates (do the harm-class economics replicate?)
  and G-T (does witness-carried containment replicate?).
* For G-T, for: the one-shot signature (exact vote tie) is a protocol
  property, not a seed property, so the witness window should carry the
  tie parties' classes at any seed. Against: witness-window emptiness
  rates, fallback rates, decode-alphabet sizes, and wrong-row
  compositions can all shift with seed; PR-11.1's prior (soft-arm harm
  mostly outside the fork-pair structure) could reassert at another
  seed. Either outcome is informative: a pass upgrades PR-12.3's
  attribution evidence toward a mechanism property; a fail bounds the
  GO to a seed-0 artifact before anything is built on it — the PR-12
  version of PR-9B's closure discipline.
* **Key assumption named:** witness-window reconstruction from the
  committed s1/s2 `topk` artifacts exactly mirrors the frozen scorer's
  `fork_witness` (0.05 window, survivors only, ≥ 2 decode classes) at
  those seeds, as it was verified to at s0. A mismatch is
  instrumentation (kill), not evidence.
* **Most decision-relevant unknown:** whether the s1/s2 gated cells
  contain enough pending-led wrong rows for G-T to be well-powered
  (s0 had 153/330 dual-presented wrong rows on pairD/pairB). The
  vacuous-cell rule (§5) pre-commits the handling so this cannot become
  a post-hoc judgment call.

## 7. Kill conditions (any → `replication-blocked`)

1. Missing committed input (§3 table); router/hazard cross-check
   failure per seed (pairs, merges, rows, ambiguous counts — counts
   recorded per seed, no s0 constant assumed).
2. PR-12 base byte-gate red before or after; any byte drift in
   committed `pr12/`, `pr12_1/`, `pr12_2/`, **or `pr12_3/`**.
3. Prototype/W1/W2 quarantine-led or non-pending divergence per the §5
   audit-packet comparison rule.
4. Certified-abstain set ≠ merge-led set on any cell.
5. Any candidate diff outside pending-led rows.
6. Width-bound violation, non-pre-registered selection order or
   tie-break, or lookahead in W2/witness construction.
7. Review-queue set or payload differing across shapes.
8. "certified" outside permitted fields.
9. Containment inflation: a GO counted on a gated cell that is
   `width-saturated` (or `vacuous`) rather than a genuine
   chance-margin pass.
10. Any intervention shape beyond `{prototype, W1, W2}`; any change to
    the quarantine `escalate` allow-list beyond the committed PR-12.3
    entries (`W1`, `W2` are already members — the §9.7 defect class
    cannot recur because no new shape exists to omit); any counter
    leakage into committed scan outputs (PR-12.4 counters namespaced
    `pr124`, mirroring the committed `pr123` isolation, so that
    `--scan12-2` and `--scan12-3` reproduce byte-identically).

## 8. Byte-reproducibility requirement

A second, independent invocation of the (separately authorized) scan
must reproduce the entire `pr12_4/` tree byte-identically and the same
verdict, and `--scan12-2` and `--scan12-3` must still reproduce their
committed outputs byte-identically after the PR-12.4 implementation
lands. All three checks are recorded in the scan output; failure of any
is kill-2/kill-10 territory.

## 9. Expected artifacts and filenames (none created by this memo)

* This memo, committed before any implementation or run:
  `results/issue_failure_mode_blindness/PR12_4_WITNESS_WINDOW_REPLICATION.md`
  (results appended as §12+, append-only; §§1–11 are the frozen
  pre-registration snapshot).
* Implementation (separately authorized, not part of this PR):
  `--scan12-4` + `--shape12-4 {prototype|W1|W2}` in
  `harness/harness_boundary_sim.py`; a `scan12_4` block in
  `harness/harness_policy.json` (policy version `pr12.4-scan-0.1`);
  prior scan blocks untouched.
* Per (shape × cell) under
  `results/issue_failure_mode_blindness/pr12_4/<shape>/<cell>/`: the
  three standard files (`audit_packet.jsonl`, `decision_table.csv`,
  `memory_packet.jsonl`). 3 shapes × 12 cells × 3 files = **108
  artifact files**.
* Gate scoring:
  `results/issue_failure_mode_blindness/pr12_4/replication_scan.json` —
  every G-A/G-M/G-T/G-W/G-V/G-R check, per-seed anchor counts,
  chance-baseline and vacuity/width-saturation status per gated cell,
  contra report-only economics, frozen-baseline suppression report,
  `max_resolution_lag` per seed, all three byte-gate results, verdict;
  every results-section number recomputable from it.

## 10. Verdict vocabulary and downstream-use boundary

**Verdict (exactly one):**
`replication-GO(<candidates>)` — at least one candidate passes every
hard gate on every cell where that gate applies, with genuine
(non-vacuous, non-width-saturated) G-T passes on all four gated cells;
`replication-negative` — no candidate does (including by vacuity, §5);
`replication-blocked` — instrumentation contradiction (§7).

**Scope of every verdict:** offline-simulator evidence only, at seeds
s1/s2 on pairs B/D. **Downstream-use boundary — no verdict, `GO`
included, may be claimed as:** certification of mechanism (d) globally;
reader utility of dual-presentation in any form; suitability for agent
prompting; promotion to any policy version; memory ingestion or
write-back of dual-presented content; any autonomous downstream use of
the emitted packets; safety of any downstream use; or any FAM-core
policy, threshold, or reader-contract change — **PR-10's merge-abstain
remains the only certified reader contract, unmodified by any outcome
here.** Each such step requires its own pre-registration. A
`replication-GO` additionally claims nothing about contra-arm
pending-led rows (§2) and does not retroactively widen PR-12.3's
seed-0 claim — it stands as its own registered result alongside it.

## 11. Standing-record restatement

PR-12.1 remains `reshape-negative`. PR-12.2 remains `pending-negative`.
PR-12.3 remains `attribution-evidence-GO(W1,W2)` — narrow seed-0
evidence on two gated one-shot cells, not enlarged here. PR-10
merge-abstain remains the only certified reader contract. C1/C2/C3
remain closed. FAM-core remains untouched, and nothing in this memo
authorizes FAM-core work, implementation, or any run; implementation of
§9 requires explicit separate authorization after review of this
pre-registration.

## 12. Results (reserved; append-only after an authorized run)

Intentionally empty at pre-registration. §§1–11 above are the frozen
snapshot and are never rewritten.

Run date 2026-07-07, this section committed on branch
`feat/pr12.4-witness-window-replication-scan` (append-only; the §§1–11
pre-registration above — including its DRAFT header, committed unchanged
on docs branch `4345cb6`, merged to main `7779297` — is never rewritten).
Implementation authorized separately after the pre-registration merge:
`--scan12-4` + `--shape12-4 {prototype|W1|W2}` and the `scan12_4` policy
block (policy_version `pr12.4-scan-0.1`) in the two harness files only;
`decide_probe`, the W1/W2 shapes, the witness-window builder, and every
per-row counter are byte-reused from the committed PR-12.3
implementation — no new shape, observable, or per-row counter was
introduced (see §12.8). Every number below is a verbatim field of
`replication_scan.json`.

### 12.1 Verdict

**`replication-GO(W1,W2)`** — the witness-window result replicates
across seed variation. Both W1 (dual-present-all) and W2 (age-gated)
pass every hard gate on every cell where that gate applies, with
genuine (non-vacuous, non-width-saturated) G-T passes on all four gated
one-shot cells. `kill_conditions: []` (zero);
`base_bytecheck_before = base_bytecheck_after = true`; reproducibility
checks all true (§12.4).

### 12.2 Exact G-A / G-M / G-T on the four gated harm-class cells

Ceilings: G-A ≤ 0.05, G-M ≤ 0.05, G-T ≥ 0.5 floor **and** ≥ that
cell's chance-baseline+0.15. Denominator for G-A/G-M is correct traffic
`n − wrong_none`. s0 reference (PR-12.3 §9.2, closed, not re-judged):
W1 0.921569/1.0, W2 0.915966/1.0 (pairD/pairB).

| shape · cell | G-A rate (rows) | G-M rate (rows) | dual_wrong | contained | containment | verdict |
|---|---|---|---|---|---|---|
| W1 · pairD_oneshot_s1 | 0.000000 (0) | 0.020408 (52) | 178 | 178 | 1.000000 | **PASS** |
| W1 · pairB_oneshot_s1 | 0.000000 (0) | 0.002931 (6) | 278 | 278 | 1.000000 | **PASS** |
| W1 · pairD_oneshot_s2 | 0.000000 (0) | 0.015690 (38) | 213 | 212 | 0.995305 | **PASS** |
| W1 · pairB_oneshot_s2 | 0.000000 (0) | 0.006748 (15) | 330 | 330 | 1.000000 | **PASS** |
| W2 · pairD_oneshot_s1 | 0.015699 (40) | 0.004710 (12) | 142 | 142 | 1.000000 | **PASS** |
| W2 · pairB_oneshot_s1 | 0.002931 (6) | 0.000000 (0) | 231 | 231 | 1.000000 | **PASS** |
| W2 · pairD_oneshot_s2 | 0.015277 (37) | 0.000413 (1) | 170 | 170 | 1.000000 | **PASS** |
| W2 · pairB_oneshot_s2 | 0.006748 (15) | 0.000000 (0) | 275 | 275 | 1.000000 | **PASS** |

Containment at s1/s2 (0.9953–1.0000) is at or above the s0 values it
replicates. Exactly one dual-presented wrong row in the whole scan is
uncontained (W1 · pairD_oneshot_s2). Independently re-scored from the
emitted memory packets + committed pr10 CSVs (bypassing every harness
counter): dw/tc/containment exact-match on all 8 (candidate × gated
cell) pairs. The witness window, not the fallback, carries the result:
W2 is pure witness (0 fallback rows on every gated cell); W1 fallback
rows are 30/6/38/15 of 230/284/251/345 dual-presented.

### 12.3 Chance-baseline values (containment-inflation control, §5 / kill-9)

`n_decode_classes = 4` on every cell at both seeds (as at s0; measured
per cell from committed decode data, §5). All eight gated
(candidate, cell) pairs clear chance+0.15 with margin ≥ +0.30;
`width_saturated = false`, `vacuous = false`, `genuine = true`
everywhere; kill-9 does not fire.

| shape · cell | chance_baseline_rate | chance+0.15 | containment | margin |
|---|---|---|---|---|
| W1 · pairD_oneshot_s1 | 0.546348 | 0.696348 | 1.000000 | +0.304 |
| W1 · pairB_oneshot_s1 | 0.500000 | 0.650000 | 1.000000 | +0.350 |
| W1 · pairD_oneshot_s2 | 0.521127 | 0.671127 | 0.995305 | +0.324 |
| W1 · pairB_oneshot_s2 | 0.500000 | 0.650000 | 1.000000 | +0.350 |
| W2 · pairD_oneshot_s1 | 0.545775 | 0.695775 | 1.000000 | +0.304 |
| W2 · pairB_oneshot_s1 | 0.500000 | 0.650000 | 1.000000 | +0.350 |
| W2 · pairD_oneshot_s2 | 0.522059 | 0.672059 | 1.000000 | +0.328 |
| W2 · pairB_oneshot_s2 | 0.500000 | 0.650000 | 1.000000 | +0.350 |

The §5 vacuous-cell rule never had to fire on a candidate: every gated
cell carried a substantial dual-presented wrong population (142–330
rows). (The prototype shape's G-T is `vacuous` by construction — it
never dual-presents — and the prototype is not a judged candidate.)

### 12.4 G-W, V1–V4, G-R, reproducibility, and kill conditions — all pass

* **G-W (hard, every cell):** `over_bound_rows = 0`,
  `truncated_rows = 0`, `empty_candidate_set_rows = 0` on every
  (shape, cell); full-width fractions ≤ 0.169 on gated cells (0 on both
  pairB cells — every pairB dual-wrong row is width 2, which is why its
  chance baseline is exactly 0.5).
* **V1–V4 (hard, every cell):** all pass; V1/V2 violations = 0; V3
  review queues identical across shapes; V4 `certified_leaks = 0`,
  `incomplete_audits = 0`.
* **G-R (hard, every cell):** kill-3 (audit-record-level non-pending
  identity vs the in-run prototype, modulo per-record policy_version —
  the §5 baseline deviation, stronger than the kill-5 counter check)
  passes on all 24 candidate (shape, cell) pairs; kill-5 counter
  identity passes; controls zero-adverse with structurally-zero G-M;
  continuity cells pass the structural anchor gate (zero
  certified-abstention escapes, §12.6); PR-12 base byte-gate green
  before and after.
* **§8 reproducibility, recorded in the scan output:**
  `pr12_4_double_run = true` (embedded re-emission byte-identical);
  `scan12_2_tree = true` and `scan12_3_tree = true` (committed
  pr12_2/pr12_3 artifact trees regenerate byte-identically under the
  PR-12.4 implementation, checked in temp dirs without touching them).
  Externally verified in addition: a second full `--scan12-4`
  invocation reproduces all 109 `pr12_4/` files byte-identically
  (sha256 over the tree) with the same verdict; full in-place re-runs
  of `--scan12-2` and `--scan12-3` leave `git status` clean and their
  verdicts unchanged (**`pending-negative`**,
  **`attribution-evidence-GO(W1,W2)`**).
* **Kill conditions 1–10:** none fired (`kill_conditions: []`),
  including the new structural kill-10 (shape set frozen to
  {prototype, W1, W2} — asserted before anything runs).

### 12.5 Contra cells remain report-only — the §2 double-bind replicates at s1/s2

Candidate-economics gates were not applied to contra cells (§2); the
same measurements, carried report-only, confirm the recorded structural
double-bind at both new seeds — the same two horns as s0:

| shape · contra cell | G-A rate | G-M rate | would-fail-if-gated |
|---|---|---|---|
| W1 · pairD_contra_s1 | 0.000000 | 0.110972 | **G-M** |
| W1 · pairB_contra_s1 | 0.000000 | 0.090502 | **G-M** |
| W1 · pairD_contra_s2 | 0.000000 | 0.106599 | **G-M** |
| W1 · pairB_contra_s2 | 0.000000 | 0.081649 | **G-M** |
| W2 · pairD_contra_s1 | 0.089615 | 0.021357 | **G-A** |
| W2 · pairB_contra_s1 | 0.086918 | 0.003584 | **G-A** |
| W2 · pairD_contra_s2 | 0.089255 | 0.017343 | **G-A** |
| W2 · pairB_contra_s2 | 0.079705 | 0.001944 | **G-A** |

Per §2/§3 (corrected framing): this gate family cannot pass on the
contra cells under the current 5% ceilings; a contra claim would
require a different pre-registered accounting model or intervention
family. The GO makes no claim about contra-arm pending-led rows.

### 12.6 Registered per-seed anchors and resolution lag (§5 G-R deviation)

The measured per-seed counts below are now the registered anchors for
any future s1/s2 work (`anchor_counts` in `replication_scan.json`;
recomputable). Continuity structural anchor — zero certified-abstention
escapes — holds at both seeds with seed-specific totals (the s0
constants 300/375 = 292+83 were measurements, not laws): stale-soft s1
= 280 abstained, 318 = 280 + 38 flagged, 0 escapes; s2 = 296 abstained,
326 = 296 + 30 flagged, 0 escapes. Router/hazard cross-checks proved
genuinely per-seed: final-epoch ambiguous pairs are 32 on every
one-shot cell at s1/s2 (the 32-pair one-shot signature is
protocol-structural, not a seed-0 accident) but 30/25 on
pairD/pairB_contra_s2 — and the committed hazard evidence matched both
departures exactly.

**W2 exactness carries (§4 report field):** `max_resolution_lag = 1` on
every cell with pairs, at both seeds — the committed s0 max-lag-1 bound
under which PR-12.2 proved the never-resolving classifier exact is an
empirical invariant of this protocol at s1/s2 as well, so W2's
selectivity is exact on this panel (reported, never gated).

### 12.7 Frozen-baseline suppression — reported, gated nowhere (§5)

Identical under prototype/W1/W2 per cell: pairD_oneshot 820 (0.321821)
s1 / 636 (0.262593) s2; pairB_oneshot 418 (0.204201) s1 / 327
(0.147099) s2; pairD_contra 963 (0.403266) / 1018 (0.430626);
pairB_contra 771 (0.345430) / 918 (0.356921); pairD_stale-soft 816
(0.319499) / 640 (0.263050); clean 0 / 0. G-A measures only
candidate-attributable suppression (§12.2), which is ≤ 0.0157.

### 12.8 Implementation notes

1. **No `pr124` counter namespace was needed.** §7.10 required new
   counters to be namespaced to protect committed byte identity; in the
   event, PR-12.4 introduced **zero** new per-row counters — the frozen
   W1/W2 shapes and the existing namespaced `pr123` counters already
   carry every §5 measurement, so there was nothing to leak. The one
   new measurement (`resolution_lag`, §12.6) is a top-level `run_cell`
   return key computed from the rebuilt router — it touches nothing
   emitted and no counter dict. The §7.10 protection was then verified
   rather than assumed: committed pr12_2/pr12_3 trees regenerate
   byte-identically (§12.4).
2. **The recurring PR-12.2/12.3 allow-list defect class could not
   recur:** no new shape exists to omit from the quarantine escalate
   allow-list (W1/W2 were added there by the committed PR-12.3 fix),
   and kill-10 asserts the frozen shape set before any cell runs.
3. Change surface: `harness/harness_boundary_sim.py` +
   `harness/harness_policy.json` only; `pr12_4/` new (108 artifacts +
   `replication_scan.json`); committed `pr12/`, `pr12_1/`, `pr12_2/`,
   `pr12_3/`, FAM-core, and the frozen scorer untouched (verified by
   git status and the embedded byte-checks).

### 12.9 Scope and downstream boundary (restated per §10 — nothing enlarged)

Offline-simulator evidence only, seeds s1/s2 on pairs B/D. This GO does
**not** certify mechanism (d) globally, nor make any reader-utility,
prompting, promotion, memory-ingestion, or downstream-use-safety claim,
nor authorize any FAM-core / policy / threshold / reader-contract
change. It does not enlarge PR-12.3's seed-0 claim — the two results
stand side by side as registered evidence. **PR-10's merge-abstain
remains the only certified reader contract.** PR-12.1
(`reshape-negative`) and PR-12.2 (`pending-negative`) verdicts stand
unchanged; C1/C2/C3 are not revived; contra-arm pending-led rows remain
outside every candidate-economics claim (§12.5). What a GO here adds to
the record is exactly one thing: witness-window truth-containment on
the one-shot harm class is now evidenced at three seeds on two pairs,
upgrading PR-12.3's attribution evidence toward a mechanism property —
each §11-listed step toward any use of it still requires its own
pre-registration.

### 12.10 Emitted artifacts (108 per-(shape × cell) files + the scan report)

Gate report:
`results/issue_failure_mode_blindness/pr12_4/replication_scan.json`

Per (shape × cell) under `results/issue_failure_mode_blindness/pr12_4/`,
the three standard files (3 shapes × 12 cells × 3 = 108):

```
pr12_4/{prototype,W1,W2}/{clean_pairA,pairD_stale-soft,pairD_contra,pairB_contra,pairD_oneshot,pairB_oneshot}_{s1,s2}/
    {audit_packet.jsonl,decision_table.csv,memory_packet.jsonl}
```
