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

Run date 2026-07-07, this section committed on branch
`feat/pr12.6-action-boundary-scan` (append-only; §§1–20 — committed on
docs branch `ef59a75`, merged to main `5646f00` — are never rewritten).
Implementation exactly per §18: standalone read-only scorer
`harness/action_boundary_score.py`; `harness_boundary_sim.py` and
`reader_utility_score.py` byte-untouched. **F2 registration: zero** of
the permitted two (§4) — no fitting code path exists, so §16.4 is
structurally satisfied and the §5 split weakness never enters the
judged evidence. Every number below is a verbatim field of
`action_boundary_scan.json`.

### 21.1 Verdict

**`action-boundary-evidence-GO(W2:F1b)`** — exactly one
(candidate, policy) combo passes every gate on the test partition.
`kill_conditions: []`; all 85 input files match the `0afcb2b` pin;
committed dirs clean before and after; internal double pass and an
external second invocation byte-identical (97 `pr12_6/` files,
sha256-verified). Label-freedom is structural: policy code receives
only RowObs/CellCtx objects built from the policy-visible packets
before any truth file is opened; `pr12_5/` tables are opened only by
the §16.5 reconciliation checker, which verified every row's
`truth_in_set`/`truth_in_alts`/`deployed_correct`/`width` against the
committed values and every unit's `n_act` — zero mismatches.

### 21.2 The quiet-cell guard is the read-time separator (§10 features, measured)

The §10 cell-context features, computed **only** from
`contradiction_pair_review` / `ambiguous_pair_review` record counts in
the row's own audit packet (reader-visible structural records; values
recorded per unit in `cell_context_features` with their source):

* one-shot cells: 19–28 contradiction pairs vs 32 ambiguous pairs →
  **guard open** (contradiction ≤ ambiguous) on all 12 one-shot units;
* contra cells: 207–209 contradiction pairs vs 25–32 ambiguous pairs →
  **guard closed** on all 12 contra units.

The guard separates the harm classes perfectly on this panel without
ever seeing an arm name, cell name, path, seed, or label.

### 21.3 Gate table on the test partition (all values verbatim)

**W2:F1b — the passing combo.** One-shot units: pairD_s1 coverage
0.831169, precision 0.906250, wrong mass 0.077922; pairD_s2 0.912281 /
0.993590 / 0.005848; pairB_s1 1.0 / 1.0 / 0.0; pairB_s2 1.0 / 1.0 /
0.0. G-A1 (≥ 0.75) pass ×4; G-A3 (dominates P-assert on acted rows)
pass ×4; §14 (≤ 0.10) pass ×4. Contra units: coverage 0 and wrong mass
0 on all four → G-C1 (≤ 0.05), G-C2, §14 pass ×4; G-C3 all four contra
test units included. G-A2 aggregate one-shot test coverage
**0.950662** ≥ 0.25. §13 global wrong-mass rate **0.013800** ≤ 0.05
over the candidate's 8 test units (see §21.6 for the §13 wording
note). The null is beaten, not disguised: `B-defer-all` covers 0 and
cannot GO (§9); W2:F1b resolves 790 of 831 one-shot test deferrals at
a wrong-per-deferral-avoided exchange of 0.0 (pairB, both seeds),
0.09375 (pairD_s1), and 0.00641 (pairD_s2).

**Failing combos — each failure is the registered question answering
itself:**

| combo | first failing gates | reading |
|---|---|---|
| W2:F1a | G-C1/G-C2/§14 on pairB_contra_s2 (wrong mass 0.2), pairD_contra_s1 (0.207547), pairD_contra_s2 (0.4) | the row-local witness signature (W2 · width 2 · witness · never-resolving lead) does NOT separate harm classes — contra W2 rows carry the same row-local shape and the veto is ~0% precise there |
| W2:F1c | same contra failures as F1a | ditto (F1c ⊇ F1a on W2) |
| W1:F1c | G-C1/G-C2 on pairD_contra_s1/s2, G-C2 on pairB_contra_s2 (precision 0–0.14 acting on contra) | extending to W1 without the guard imports the same contra exposure |
| W1:F1a, W1:F1b | G-A2 (coverage 0) | structurally W2-only policies under a W1 candidate; expected at design time |

### 21.4 Development-partition reference (s0; reported, never gated — §5), including the honest exceedance

s0 mirrors the test structure — with one registered caveat that must
not be buried: on `W2:pairD_oneshot_s0`, F1a/F1b/F1c coverage 0.860465
at precision 0.864865 gives wrong mass **0.116279, above the 0.10 §14
ceiling** (and W1:F1c on `pairD_oneshot_s0` is 0.108374, likewise
above). The gates bind on the test partition per the §5 registration,
and on test the same policy scores 0.077922/0.005848 — but the s0
figures show the §14 ceiling is *tight* for pairD one-shot traffic,
not comfortably cleared everywhere in principle. Any future
registration building on W2:F1b should treat pairD-class wrong mass
near 0.08–0.12 as the realistic operating band, not the pairB-class
zeros. Contra behavior at s0 matches test: guard closed, F1b coverage
0 on both contra cells; F1a/F1c act on `pairD_contra_s0` at 0.5
coverage with 0.0 precision (wrong mass 0.5) — the same separation
failure the test partition shows.

### 21.5 Kill report and reproducibility (§§16–17)

Kills 1–9: none fired. Specifically: 85/85 inputs pinned to
`0afcb2b`; zero label-leak paths (structural, §21.1); zero family
motion (the §4 list is the code's `POLICIES` dict, F2 empty); zero
split violations (no fitting path); zero ACTs outside a presented set;
zero join misses; zero pr12_5 reconciliation mismatches; zero writes
outside `pr12_6/` (`git status` clean on `pr12_3/`, `pr12_4/`,
`pr12_5/`, `pr10/` before and after); internal double pass identical;
external re-run byte-identical over all 97 emitted files; no contra
unit excluded from any combo's verdict (G-C3 ×6 combos); no forbidden
claim language in the output.

### 21.6 Observed §13 wording note (recorded, no threshold motion)

§13's frozen text reads "across all 16 test units … per candidate" —
internally inconsistent, since a (candidate, policy) verdict spans
that candidate's 8 test units. Resolved as registered in the scan
output (`s13_interpretation_note`): the binding gate is per-candidate
over its 8 test units (W2:F1b = 0.013800 ≤ 0.05), and the pooled
16-unit figures are also reported (F1a 0.013866, F1b **0.004192**,
F1c 0.038375) — W2:F1b passes under either reading, so the ambiguity
is immaterial to this verdict, and it is recorded here rather than
silently resolved.

### 21.7 Scope and downstream boundary (restated per §20 — nothing enlarged)

Offline action-boundary evidence only, over the 16 test units (plus s0
development reference). This GO establishes exactly one thing: **a
registered, label-free, reader-visible, disposition-scoped policy
(W2:F1b) can decide act-versus-defer on witness-window rows while
holding every registered wrong-action ceiling across one-shot AND
contra traffic on this panel** — because the contra harm class is
separable at read time by contradiction-pair density, not because
acting is safe in general. It authorizes **no** deployment, FAM-core
integration, prompting use, promotion, memory ingestion, autonomous
downstream use, live acting, or reader-contract change. **PR-10's
merge-abstain remains the only certified reader contract, and the
operational posture on witness-window rows remains deferral.**
PR-12.1–12.5 verdicts stand unchanged.

### 21.8 Emitted artifacts

`results/issue_failure_mode_blindness/pr12_6/action_boundary_scan.json`
(gate tables, §15 accounting incl. comparators, cell-context feature
values with source, exchange register, 85-file input manifest,
reproducibility flags, F2 registration, §13 note, verdict) plus 96
per-row decision tables `pr12_6/rows_<candidate>_<cell>_<policy>.csv`
(16 test units × 6 policies) — every §21 number recomputable from
`pr12_6/` alone.
