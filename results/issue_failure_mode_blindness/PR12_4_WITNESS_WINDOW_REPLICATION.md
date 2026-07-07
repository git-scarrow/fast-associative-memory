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
