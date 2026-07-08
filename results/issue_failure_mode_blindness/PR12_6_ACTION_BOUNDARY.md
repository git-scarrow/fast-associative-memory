# PR-12.6 — Disposition-scoped action boundary for witness-window reader policies (pre-registration DRAFT)

Date: 2026-07-07. Main @ `0afcb2b`. **Status: DRAFT — design-only;
awaiting explicit review and authorization before any implementation,
scoring run, or `pr12_6/` artifact.** Analysis-track only: like
PR-12.5, this PR proposes no harness run, no new packets, and no
disposition change — it evaluates act-versus-defer policies read-only
over the committed `pr12_3/`/`pr12_4/` packets. Standing record,
restated up front and unmodified by any outcome here:

* PR-12.1 `reshape-negative`; PR-12.2 `pending-negative`; C1/C2/C3
  closed. PR-12.3 (s0) and PR-12.4 (s1/s2) remain narrow containment
  evidence; PR-12.5 remains reader-utility evidence for always-acting
  policies, with its **contra inversion carried here as a binding
  safety constraint** (P-alt-uniform 0.66–1.00 on one-shot rows but
  ≤ 0.13 on contra rows — PR-12.5 §12.4), not as noise.
* **PR-10's merge-abstain remains the only certified reader contract.**
* **FAM-core is untouched at every layer**; nothing here authorizes
  FAM-core work, deployment, prompting use, promotion, memory
  ingestion, autonomous downstream use, or a new reader contract (§20).
* "Reader policy" means a pre-registered, label-free decision rule
  evaluated offline over emitted packet fields — never an LLM, a
  prompt, or a live agent.

## 1. Hypothesis

**Core question (as authorized):** can a reader policy decide when to
act versus defer on W1/W2 witness-window rows while maintaining a
pre-registered wrong-action risk ceiling across both one-shot and
contra traffic?

**Hypothesis:** at least one policy from the §4 frozen families,
scoping itself by **read-time packet observables only** (§10 — never
by arm, cell name, or file path), achieves on the held-out test
partition (s1/s2, §5): acting precision ≥ 0.75 on every one-shot unit
where it acts, aggregate one-shot coverage ≥ 0.25, wrong-action mass
≤ 0.10 on every unit and ≤ 0.05 globally, and wrong-action mass
≤ 0.05 with precision ≥ 0.75 on every **contra** unit — i.e., the
value asymmetry PR-12.5 measured between harm classes is *separable at
read time*. Falsification is symmetric and pre-accepted: if no
observable separates the classes, the contra safety gates fail, the
verdict is `action-boundary-negative`, and the registered conclusion
is that acting on witness-window rows cannot yet be bounded — deferral
remains the only defensible posture.

## 2. Scope

In-scope rows: the pending-led **dual-presented** rows of the
committed W1/W2 packets on the one-shot AND contra cells, pairs B/D,
seeds s0/s1/s2 — the same row universe PR-12.5 scored (24 units =
2 candidates × 4 cell bases × 3 seeds), now partitioned into
development (s0, 8 units) and test (s1/s2, 16 units) per §5. Frozen
and out of scope: every non-pending disposition, quarantine/superseded
handling, clean/stale-soft cells (no dual rows), every FAM-core layer,
every committed artifact under `pr12/`–`pr12_5/` (byte drift in any is
a kill), and any online, prompted, or agent-embodied evaluation.
**Contra rows are never collapsed into the one-shot claim** (§12, §16.8):
they are gated separately, as a safety constraint, and a GO must pass
both gate families simultaneously — excluding or reweighting contra
units is a kill, not a scope choice.

## 3. Exact input artifacts (all committed; pin = main `0afcb2b`)

* **Policy-visible inputs (the only bytes a policy may read):**
  `pr12_3/{W1,W2}/<cell>_s0/{memory_packet.jsonl,audit_packet.jsonl}`
  and `pr12_4/{W1,W2}/<cell>_s{1,2}/{memory_packet.jsonl,audit_packet.jsonl}`
  for `<cell>` ∈ {pairD_oneshot, pairB_oneshot, pairD_contra,
  pairB_contra}.
* **Scoring-only inputs (never policy-visible):** the committed pr10
  run-stem CSVs (`true_label`, `vote_pred_label`), used exactly as in
  every committed scorer.
* **Reconciliation inputs from PR-12.5 (never policy-visible):**
  `pr12_5/reader_utility_scan.json` and `pr12_5/rows_*.csv` — they
  define the row universe (`n_act` per unit must reconcile exactly) and
  carry truth-derived columns (`truth_in_set`, `truth_in_alts`,
  `deployed_correct`) that this PR's scorer must reproduce per row;
  **any policy input path that touches a `pr12_5/` file is a label
  leak** (§16.2), because those tables encode correctness.
* All input files hash-pinned to `0afcb2b` with a per-file manifest in
  the scan output (the PR-12.5 G-U4 mechanism, reused).

## 4. Allowed policy families (frozen at this memo; complete list)

Every policy maps each in-scope row to exactly one of `ACT(c)` with
`c` in that row's presented set, or `DEFER` (§6). Deterministic, or
stochastic only via exact expectation over a uniform choice (no RNG).

* **Comparators (never GO candidates):** `B-defer-all` (defer every
  row — the null that passes every safety gate and must be beaten on
  coverage); `B-act-alt` and `B-act-uniform` (PR-12.5's always-acting
  policies, re-scored under §7–§9 accounting as reference rows).
* **F1 — structural predicates (parameter-free).** `ACT` iff a
  conjunction of §10 observables holds, else `DEFER`; the acted class
  is fixed by the predicate. The complete enumerated set:
  * **F1a (sole-witness-veto):** ACT(the single alternative) iff
    shape = W2, width = 2, source = witness (no fallback), and the
    row's leading pair is never-resolving at the final epoch per the
    cell's own `ambiguous_pair_review` records; else DEFER.
  * **F1b (F1a + quiet-cell guard):** F1a, additionally requiring the
    row's cell-context contradiction-pair count (§10) to be ≤ the
    cell's final-epoch ambiguous-pair count; else DEFER. (Rationale,
    fixed in advance: contra cells carry a large resolved-contradiction
    population — quarantine-heavy context is the candidate read-time
    signature of contra-class traffic.)
  * **F1c (F1a restricted to W1 additionally):** as F1a but allowing
    shape ∈ {W1, W2}; measures whether the W2 age-gate is load-bearing
    for the boundary.
* **F2 — tuned thresholds (train/test split mandatory, §5).** Rules of
  the form "ACT(single alternative) iff width = 2 ∧ source = witness ∧
  g(cell-context features) ≥ θ", with g and θ selected **only** on the
  s0 development partition and frozen in the implementation before any
  s1/s2 row is evaluated. At most **two** F2 policies may be
  registered at implementation time (named and frozen in the scorer
  before the test run); more is a kill (§16.3).

## 5. Train/test split (pre-registered, with an honesty note)

Development partition: all s0 units (8). Test partition: all s1/s2
units (16) — judged **once**; every §11–§14 gate binds on the test
partition only; s0 results are reported as development reference.
**Known limitation, recorded rather than hidden:** s1/s2 *aggregate*
outcomes for always-acting policies are already public
(PR-12.4/PR-12.5), so the test partition is not pristine at the
aggregate level. Mitigations, both binding: (a) the F1 family and the
F2 selection procedure are frozen in this memo before any per-row
action-boundary evaluation exists anywhere; (b) F2 fitting may consume
s0 rows only, enforced by the scorer's input manifest (an s1/s2 packet
read during a fitting code path is a kill, §16.4). Residual risk — the
designer's own knowledge of the published aggregates — cannot be
mechanically removed and is acknowledged as the reason F1's
parameter-free predicates are the primary candidates.

## 6. Acting versus deferral semantics

Per in-scope row the policy emits exactly one of:

* **ACT(c):** the reader would rely on decode class `c`, which must be
  a member of that row's presented set (acting outside the presented
  set is structurally impossible and coded as a kill if observed).
  Acting is an offline label applied at scoring time — no packet, no
  disposition, and no downstream system is touched.
* **DEFER:** the row keeps the committed dual-present/escalation
  posture (human review with the width-narrowed candidate set). DEFER
  is never penalized as wrong; its cost is counted only in the §9
  deferral accounting.

## 7. Wrong-action definition

A wrong action on a row is `ACT(c)` with `c ≠ true_label` (registry
truth joined at scoring time from the committed run-stem CSV, exactly
as in PR-12.5). For an exact-expectation policy, the row contributes
its wrong-probability mass. **Wrong-action mass of a unit** = summed
wrong-action expectation over that unit's in-scope rows, divided by
the unit's in-scope row count (deferred rows stay in the denominator —
a policy can never reduce its wrong-action mass by acting less
accurately on fewer rows without that showing up in §9 coverage).

## 8. Correct-action definition

A correct action is `ACT(c)` with `c = true_label`; expectation mass
for stochastic choices. **Acting precision** of a unit = correct-action
mass / (correct + wrong action mass) — i.e., accuracy among acted
rows only. Precision is undefined on a unit where the policy acts on
zero rows; such a unit passes no precision gate and contributes zero
coverage (§9), so all-defer behavior can never back into a GO.

## 9. Deferral accounting

Per unit: `coverage` = acted-row fraction; `deferral_rate` =
1 − coverage; deferred-row mean presented width (the residual human
search space); and the counterfactual columns for the §4 comparators.
**Coverage floor for GO (non-triviality):** aggregate coverage over
the four one-shot **test** units per candidate ≥ **0.25**. `B-defer-all`
therefore passes every safety gate and can never GO — by construction,
the null is safe and useless, and the floor is what a GO must clear
above it.

## 10. Disposition-scoping requirement (the crux)

A policy may condition **only** on read-time observables present in
the committed packets for the row's own cell:

* Row-local: shape (W1/W2 — a property of the emitting governance
  layer), presented width, candidate classes, deployed class, source
  basis (witness co-resident vs fork-counterpart fallback), the
  evidence-string age-gate markers (onsets; "strictly past onset").
* Cell-context (label-free, from the same cell's audit packet):
  counts of `contradiction_pair_review` and `ambiguous_pair_review`
  records, `never_resolving` flags, per-side led-row counts,
  quarantine-led row counts, certified-abstention counts, hazard-tier
  field.

**Forbidden scoping inputs (each use is a kill, §16.2):** cell names,
arm identifiers, file paths, seed identifiers, scan-report JSONs,
`pr12_5/` outputs, registry/true labels, `vote_pred` correctness, any
post-hoc correctness signal, and any quantity derived from them. The
point of PR-12.6 is precisely whether the one-shot/contra distinction
is recoverable from what a reader would actually see; naming the arm
assumes the answer.

## 11. Required one-shot performance gates (hard, per one-shot test unit)

* **G-A1 (precision floor):** acting precision ≥ **0.75** on every
  one-shot test unit where the policy acts ≥ 1 row. Rationale for
  0.75, fixed in advance: strictly above the best full-coverage
  uniform policy (~0.50) and 3× the 4-class alphabet chance; below
  the W2 sole-witness signature (~0.95–1.0) so a genuinely selective
  boundary can pass while an unselective one cannot.
* **G-A2 (coverage floor, aggregate):** §9 floor — one-shot test
  coverage ≥ 0.25 per candidate.
* **G-A3 (dominance carried):** on acted rows, expected correct rate
  strictly above what `P-assert` scores on those same rows (the
  certified-only reader must remain dominated on the acted subset).

## 12. Required contra safety gates (hard, per contra test unit; the PR-12.5 inversion as constraint)

* **G-C1 (contra wrong-action ceiling):** wrong-action mass ≤ **0.05**
  per contra test unit. Small units are not exempt (W2 contra units
  have 5–53 in-scope rows; one wrong action out of 8 rows = 0.125 =
  FAIL) — strictness on small n is the safety posture.
* **G-C2 (contra precision when acting):** if the policy acts on ≥ 1
  contra row, acting precision ≥ 0.75 there too — acting on contra is
  not forbidden, acting *unreliably* on contra is.
* **G-C3 (no-collapse):** the verdict computation must include every
  contra test unit for the candidate; any exclusion, reweighting, or
  averaging of contra into one-shot aggregates is kill §16.8.

## 13. Global wrong-action ceiling (hard, per candidate)

Total wrong-action mass across **all 16 test units** (one-shot +
contra, both pairs, both seeds) / total in-scope test rows ≤ **0.05**.

## 14. Per-cell wrong-action ceiling (hard, every test unit)

Wrong-action mass ≤ **0.10** per unit (one-shot units included —
coverage × (1 − precision) must stay under it; at the G-A1 floor this
binds coverage-precision combinations, e.g. full coverage requires
precision ≥ 0.90).

## 15. Abstention/coverage reporting

Per unit and policy: coverage, deferral rate, precision, correct/wrong
action mass, deferred-row mean width, and the same quantities for the
three comparators; per candidate: the aggregate one-shot coverage and
global wrong-action mass; a development-vs-test table (s0 columns
marked as development, never gated); and an explicit
`B-defer-all`-vs-GO-policy exchange line per unit (wrong actions
incurred per deferral avoided) extending the PR-12.5 G-U5 register.

## 16. Kill conditions (any → `action-boundary-blocked`)

1. Input drift from the `0afcb2b` pin; missing input.
2. Label leak: any policy code path reading truth labels, run-stem
   CSVs, `pr12_5/` files, scan JSONs, cell/arm/file/seed identifiers,
   or any §10-forbidden input.
3. Family motion: any policy added, removed, or modified after the
   first test-partition evaluation; more than two F2 policies; any F2
   threshold not frozen before the test run.
4. Split violation: any s1/s2 packet read during F2 fitting.
5. An ACT outside the row's presented set; a row join miss; a row
   whose `truth_in_set`/`deployed_correct` recomputation disagrees
   with the committed `pr12_5/` row table (reconciliation).
6. Writes outside `pr12_6/`; `git status` dirty on `pr12_3/`,
   `pr12_4/`, `pr12_5/`, `pr10/` before or after.
7. Nondeterminism: internal double pass or external re-run differing
   in any byte of `pr12_6/`.
8. Contra collapse: any contra test unit excluded from a candidate's
   verdict; any GO text claiming one-shot performance without the
   contra gates; any aggregation mixing contra rows into one-shot
   gate denominators.
9. Any output language claiming deployment readiness, prompting use,
   promotion, ingestion, autonomous use, or a reader contract.

## 17. Byte-reproducibility requirements

Internal double pass identical; an external second invocation
reproduces every `pr12_6/` file byte-identically (sha256 over the
tree); both recorded in the scan output. Committed `pr12/`–`pr12_5/`
protected by hash pinning + the §16.6 clean-tree checks (the PR-12.5
mechanism, reused).

## 18. Artifact paths (none created by this memo)

* This memo:
  `results/issue_failure_mode_blindness/PR12_6_ACTION_BOUNDARY.md`
  (results appended as §21+, append-only; §§1–20 frozen).
* Implementation (separately authorized): standalone read-only scorer
  `harness/action_boundary_score.py` (stdlib + subprocess-git;
  `harness_boundary_sim.py` and `reader_utility_score.py` stay
  byte-frozen).
* Output: `results/issue_failure_mode_blindness/pr12_6/action_boundary_scan.json`
  (gates, per-unit per-policy §15 tables, input manifest,
  reproducibility flags, development/test partition marks, verdict)
  plus per-row decision tables
  `pr12_6/rows_<candidate>_<cell>_<policy>.csv` for test units — every
  results-section number recomputable from `pr12_6/` alone.

## 19. Pass/fail verdict names (exactly one)

`action-boundary-evidence-GO(<candidate:policy,...>)` — every listed
(candidate, policy) passes G-A1–G-A3 on all one-shot test units,
G-C1–G-C3 on all contra test units, and the §13/§14 ceilings, with
§16 clean; `action-boundary-negative` — no (candidate, policy) does;
`action-boundary-blocked` — instrumentation contradiction (§16).

## 20. Downstream-use boundary

Offline expected-value evidence about pre-registered act-versus-defer
policies over committed packets — nothing else. **A GO authorizes
offline evidence for an action-boundary policy and nothing beyond
it:** no deployment, no FAM-core integration, no prompting use, no
promotion to any policy version, no memory ingestion or write-back, no
autonomous downstream use, no LLM/agent reader certification, no
acting authorization in any live system, and no reader-contract change
— **PR-10's merge-abstain remains the only certified reader
contract**, and the operational posture on witness-window rows remains
deferral unless and until a separate pre-registration proposes
otherwise with this PR's evidence as one input. A GO claims nothing
about contra traffic beyond the safety gates it passed, and nothing
about any cell, pair, seed, manifold, or mechanism outside the 16 test
units. PR-12.1–12.5 verdicts stand unchanged.

## 21. Results (reserved; append-only after an authorized run)

Intentionally empty at pre-registration. §§1–20 above are the frozen
snapshot and are never rewritten.
