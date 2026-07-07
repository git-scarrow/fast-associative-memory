# PR-12.5 — Reader-utility evidence for witness-window dual-presentation (pre-registration DRAFT)

Date: 2026-07-07. Main @ `2226d9d`. **Status: DRAFT — design-only;
awaiting explicit review and authorization before any implementation,
scoring run, or `pr12_5/` artifact.** Analysis-track only: this PR
proposes **no harness run, no new packets, no disposition change** — it
scores decision policies over the already-committed `pr12_3/` and
`pr12_4/` artifacts, read-only. Standing record, restated up front and
unmodified by any outcome here:

* **PR-12.1 remains `reshape-negative`; PR-12.2 remains
  `pending-negative`;** C1/C2/C3 are not revived.
* **PR-12.3 (`attribution-evidence-GO(W1,W2)`, s0) and PR-12.4
  (`replication-GO(W1,W2)`, s1/s2) remain containment evidence only** —
  three seeds, two pairs, offline simulator. Neither is enlarged here.
* **PR-10's merge-abstain remains the only certified reader contract.**
* **FAM-core is untouched at every layer**; no FAM-core work is
  authorized by this memo or any verdict it can produce.
* **No prompt contract is proposed, evaluated, or certifiable here**
  (§10). "Reader" in this memo means a pre-registered, parameter-free
  decision policy evaluated offline over emitted packet fields — never
  an LLM, a prompt, or a live agent.

## 1. Core question and hypothesis

**Core question (as authorized):** can a reader use the W1/W2
witness-window candidate sets to make safer or more accurate decisions
than merge-abstain, without increasing wrong-action risk or creating a
new unsafe prompt contract?

**Made precise (§2 fixes every term):** on the one-shot harm-class
rows that W1/W2 dual-present, the incumbent certified reader — PR-10
merge-abstain plus nothing else — receives the deployed answer
*asserted* (merge-abstain governs only merge-suspect-led rows; the
pending-led rows pass through). Its measured accuracy on those rows is
0.000–0.246 (§6). The hypothesis: a **parameter-free, label-free
decision policy over the presented candidate set strictly dominates
that incumbent on both axes at once** — strictly fewer wrong actions
AND strictly more correct actions, per gated cell — while clearing a
chance-baseline control, preserving abstention structurally, and
claiming nothing about acting versus deferral (that comparison is
report-only, §5 G-U5). Falsification: any gated cell where no gated
policy dominates → `reader-utility-negative`; the presented sets would
then be truth-*bearing* but not decision-*improving* under
parameter-free reading, and the residual joins the accepted column.

## 2. Scope, comparators, and the meaning of every term

* **Rows in scope:** pending-led dual-presented rows in the committed
  W1/W2 packets on the six gated one-shot cells:
  `pair{D,B}_oneshot_s{0,1,2}` (s0 from `pr12_3/`, s1/s2 from
  `pr12_4/`). 12 evaluation units = 2 candidates × 6 cells. Everything
  else — quarantine-led, superseded, merge-support, abstained, clean
  traffic — is untouched by construction: **no packet is generated or
  modified; the committed artifacts are read-only inputs** (byte-pinned,
  §7.1).
* **Incumbent comparator A (`P-assert`): the certified-only reader.**
  Acts on every in-scope row by asserting the deployed class (the
  candidate whose `basis` is `deployed vote`). This is what a reader
  operating under PR-10 merge-abstain alone receives today on these
  rows; it is the harm class.
* **Incumbent comparator B (`P-abstain`): the deferral posture.** Acts
  on no in-scope row (the PR-12 prototype escalation semantics). Zero
  wrong actions, zero correct actions; every row costs one deferral.
* **Contra cells: report-only, again.** The same policy metrics are
  computed on the committed W1/W2 contra-cell packets
  (`pair{D,B}_contra_s{0,1,2}`) and carried report-only — pending-led
  traffic there is correct-dominant, so `P-assert` may well win, which
  is data for the recorded double-bind, not a gate. No contra claim of
  any kind (PR-12.3 §2 / PR-12.4 §2 carry over verbatim).
* **"Safer / more accurate ... without increasing wrong-action risk"
  binds against comparator A**, per gated cell: wrong-action rate
  strictly below `P-assert`'s AND correct-action rate strictly above
  `P-assert`'s. **Against comparator B no such claim is possible or
  made:** any acting policy has wrong-action rate > 0 = `P-abstain`'s.
  The acting-vs-deferral trade is reported as an exact exchange rate
  (§5 G-U5) and left to a future, separately registered decision — it
  is a policy question the harness measures but does not answer.
* **"No new unsafe prompt contract":** structural. The committed
  packets assert neither candidate (`unresolved_tie`, "neither
  asserted"); this PR adds no packet, no prompt, no reader
  implementation, and its verdict vocabulary cannot certify any prompt
  contract (§10). Unsafety-by-contract cannot be created by a read-only
  scorer; the gate G-U2 verifies the non-assertion structure held on
  every scored row.

## 3. Inputs (all committed on main `2226d9d`; no generation, no run)

| input | role |
|---|---|
| `pr12_3/{W1,W2}/pair{D,B}_oneshot_s0/memory_packet.jsonl` | s0 presented sets (gated) |
| `pr12_4/{W1,W2}/pair{D,B}_oneshot_s{1,2}/memory_packet.jsonl` | s1/s2 presented sets (gated) |
| `pr12_3/`, `pr12_4/` contra-cell memory packets | report-only metrics |
| `pr10/governed/per_probe_stale-oneshot_pair{D,B}_s{0,1,2}.csv` (+ contra stems) | truth + deployed labels, **scoring only** |
| `pr12_3/attribution_scan.json`, `pr12_4/replication_scan.json` | cross-check totals (dual, dual_wrong must reconcile exactly) |

Row join: `query_id = <cell>:e<epoch>:p<probe_index>` →
`(epoch, probe_index)` in the run-stem CSV — the same join the
committed independent re-scores used (PR-12.3 §9.2 / PR-12.4 §12.2).
Policies read **only** packet fields (`candidates[].decode_class`,
`candidates[].basis`); registry truth labels enter at scoring time
only, exactly as in every committed scorer.

## 4. Frozen policy family (parameter-free; the complete list)

Evaluated deterministically; random policies are scored by **exact
expectation** (no RNG, preserving determinism). Adding, removing, or
modifying a policy after the first scoring run is a kill (§7.6).

* **P-abstain** — never act. Comparator B; never gated.
* **P-assert** — act: deployed class. Comparator A; never gated.
* **P-uniform** *(gated)* — act: uniform draw over the presented set
  (deployed + alternatives). Expected correct on a row =
  `1[truth ∈ presented] / width`. The dumbest reader that uses the set
  at all: no ordering information, no side information.
* **P-alt-uniform** *(gated)* — act: uniform draw over the presented
  **alternatives only** (presented minus deployed); rationale, fixed
  in advance: on this harm class the deployed vote is measured to be
  predominantly wrong, so vetoing it is a plausible fixed rule. Its
  cost on correct-deployed rows (it is then wrong with certainty) is
  part of what is measured, not worked around.

A policy "acts" on every in-scope row (act rate 1.0) except P-abstain
(0.0); partial-acting policies are out of scope for this first
pre-registration — one fewer degree of freedom.

## 5. Exact pass/fail gates

Per evaluation unit (candidate × gated cell), over its in-scope rows;
`n_act = dual_presented` (cross-checked against the committed scan
totals, kill on mismatch). For a policy P: `correct(P)` = expected
correct actions, `wrong(P) = n_act − correct(P)` (P always acts),
rates over `n_act`.

* **G-U1 — double dominance over the certified-only reader (hard, per
  gated unit, per gated policy):**
  `correct_rate(P) > correct_rate(P-assert)` **and**
  `wrong_rate(P) < wrong_rate(P-assert)`. Strict on both axes; no
  margin parameter (the strictness is the claim; measured margins are
  reported).
* **G-U2 — non-assertion structure (hard, every scored unit):** every
  in-scope row's tie item asserts neither candidate (type
  `unresolved_tie`, deployed and alternatives carried as candidates,
  no compiled single answer), zero rows missing the marker. This is
  the "no unsafe prompt contract" structure check on the actual scored
  rows.
* **G-U3 — chance-baseline control (hard, per gated unit, per gated
  policy; the PR-12.3 §5 discipline transposed to decisions):**
  `correct_rate(P) ≥ 1/n_decode_classes(cell) + 0.15` — the policy
  must beat a uniform draw over the cell's full decode alphabet
  (`n_decode_classes` measured per cell from committed data, = 0.25 at
  4 classes) by ≥ 0.15 absolute. A dominance pass that cannot beat
  alphabet-chance is `chance-saturated`, not a pass.
* **G-U4 — read-only integrity (hard, global):** every input file's
  content hash matches the committed object at main `2226d9d`
  (recorded per file in the scan output); `git status` clean on
  `pr12_3/`, `pr12_4/`, `pr10/` before and after; **zero files
  written** outside `pr12_5/`.
* **G-U5 — acting-vs-deferral exchange rate (report-only, never
  gated):** per unit and policy: wrong actions incurred per correct
  action gained relative to P-abstain (`wrong(P) / correct(P)`), the
  deferral volume avoided (`n_act`), and the residual-width report
  (mean presented width vs alphabet — what a human reviewer's search
  space becomes under deferral-with-narrowed-options). Recorded so the
  future acting-vs-deferral registration starts from measured numbers.

**Reconciliation (kill, §7.2):** per unit, the scorer's `n_act` and
recomputed `dual_wrong` must equal the committed scan values exactly
(they already survived one independent re-score each; a third
independent path must agree).

## 6. Predictions and falsification analysis (inference, recorded before any scoring run)

* **Committed basis.** Incumbent accuracy (`P-assert`) on the 12 gated
  units, from committed dual/dual_wrong: W1 0.246/0.029 (s0),
  0.226/0.021 (s1), 0.151/0.043 (s2); W2 0.078/0.000 (s0), 0.078/0.000
  (s1), 0.006/0.000 (s2) (pairD/pairB). On three W2 units the
  incumbent is **never** right. Containment is 0.92–1.00 and measured
  mean width ≈ 2–2.2, so P-uniform's expected accuracy is ≈ 0.45–0.50
  per unit — G-U1 and G-U3 are predicted to pass with wide margins,
  and P-alt-uniform to pass at least where deployed-correct mass is
  near zero.
* **What is genuinely open:** (a) whether P-uniform's dominance holds
  on **every** unit including the W1 pairD units where the incumbent
  is least bad (0.15–0.25) — width >2 rows dilute per-row expected
  correctness even at containment 1; (b) whether P-alt-uniform's veto
  cost on correct-deployed rows breaks its dominance on exactly those
  units — informative either way, since it measures how much of the
  set's value depends on *not* discarding the deployed vote; (c) the
  G-U5 exchange rates — the number the eventual acting-vs-deferral
  decision will actually turn on, unknown until computed.
* **Honesty note:** G-U1's direction is strongly predicted by the
  committed numbers; PR-12.5's evidential weight is therefore less in
  whether dominance holds than in **registering the exact magnitudes
  under frozen policies before any reader-contract discussion** — the
  same role PR-12.2's G-M played for PR-12.3. The failure modes worth
  the registration are (a) and (b), and the deliverable is G-U5.
* **Key assumption named:** the packet-to-CSV row join and the
  deployed-candidate identification (`basis = "deployed vote"`) are
  faithful; both already validated by two committed independent
  re-scores. A join failure or a row with zero/two deployed candidates
  is instrumentation (kill), not evidence.

## 7. Kill conditions (any → `reader-utility-blocked`)

1. Any input file hash differing from the committed object at
   `2226d9d`; any missing input.
2. Reconciliation failure: scorer `n_act`/`dual_wrong` ≠ committed
   scan values on any unit.
3. Any write outside `pr12_5/`; any modification of committed
   artifacts (G-U4 red).
4. A row whose tie item lacks exactly one deployed candidate, or a
   join miss between packet and CSV.
5. A policy consulting anything beyond the §4 packet fields
   (label-leak; policies are label-free by construction and by review).
6. Adding/removing/modifying a policy, gate, threshold, or comparator
   after the first scoring run; partial-acting policies smuggled in.
7. Nondeterminism: a second scoring run differing in any byte of
   `pr12_5/`.
8. Any claim in the output beyond the §10 vocabulary (in particular
   any prompt-contract, promotion, or ingestion language).

## 8. Byte-reproducibility requirement

A second, independent invocation of the (separately authorized) scorer
must reproduce `pr12_5/` byte-identically, recorded in the scan output.
Since the scorer is read-only over committed inputs, `pr12/`–`pr12_4/`
byte-identity is enforced as G-U4 (hash pinning) rather than
regeneration.

## 9. Expected artifacts and filenames (none created by this memo)

* This memo, committed before any implementation:
  `results/issue_failure_mode_blindness/PR12_5_READER_UTILITY.md`
  (results appended as §12+, append-only; §§1–11 frozen).
* Implementation (separately authorized): a **standalone read-only
  scorer** `harness/reader_utility_score.py` (stdlib-only; does not
  import or modify `harness_boundary_sim.py` — the sim stays byte-
  frozen this PR; no `--scan12-5` flag is added to it).
* Output: `results/issue_failure_mode_blindness/pr12_5/reader_utility_scan.json`
  — per-unit per-policy correct/wrong/act rates, G-U1/G-U2/G-U3/G-U4
  results, G-U5 exchange table, contra report-only block, input hash
  manifest, reproducibility flag, verdict. Optionally a per-row
  `pr12_5/rows_<unit>.csv` evidence table (row id, truth-in-set, width,
  per-policy expected correctness) — decided in the memo now: **yes,
  emitted**, so every §12 number is recomputable from `pr12_5/` alone.

## 10. Verdict vocabulary and downstream-use boundary

**Verdict (exactly one):**
`reader-utility-evidence-GO(<candidate:policy,...>)` — every listed
(candidate, policy) passes G-U1+G-U3 on all six gated cells with G-U2/
G-U4 green; `reader-utility-negative` — no (candidate, policy) does;
`reader-utility-blocked` — instrumentation contradiction (§7).

**Scope of every verdict:** offline expected-value evidence about
parameter-free decision policies over committed packets — nothing
else. **No verdict, GO included, may be claimed as:** a reader
contract, a prompt contract, or a change to PR-10 merge-abstain; a
claim that acting is safer than abstention/deferral (G-U5 is
report-only); certification of any LLM/agent reader behavior; prompting
use, promotion, memory ingestion, write-back, or autonomous downstream
use; mechanism-(d) certification beyond the six gated cells; any
contra-cell claim; or any FAM-core change. **PR-10's merge-abstain
remains the only certified reader contract.** A GO here contributes
exactly one thing to the record: registered, frozen-policy evidence
that the witness-window sets are decision-improving (not merely
truth-bearing) against the certified-only reader on the one-shot harm
class — an input to, never a substitute for, a future reader-contract
pre-registration.

## 11. Standing-record restatement

PR-12.1 `reshape-negative`; PR-12.2 `pending-negative`; PR-12.3
`attribution-evidence-GO(W1,W2)` (s0, narrow); PR-12.4
`replication-GO(W1,W2)` (s1/s2, narrow); contra cells report-only and
double-bound under the 5% ceilings; C1/C2/C3 closed; PR-10
merge-abstain the only certified reader contract; FAM-core untouched.
Implementation of §9 requires explicit separate authorization after
review of this pre-registration.

## 12. Results (reserved; append-only after an authorized run)

Intentionally empty at pre-registration. §§1–11 above are the frozen
snapshot and are never rewritten.
