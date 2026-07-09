# PR-12.9 — Rung-4 certification pre-registration for the S1 batch packet-reader contract (W2:F1b) (pre-registration DRAFT, design-only)

*Design-only. No implementation, no test suite, no certification run,
no serving, no artifact beyond this memo is authorized by this
document. It does not certify, serve, deploy, promote, or ingest
anything; it changes no FAM-core file and does not alter the PR-10
reader contract. **PR-10 merge-abstain remains the only certified
reader contract**, and the operational posture on witness-window rows
remains **deferral**, unless and until a certification verdict under
this registration is emitted, merged, and **explicitly approved** —
and not before. §§1–11 are frozen at commit; §12 is reserved
append-only for separately-authorized stage results. Every stage
requires its own explicit approval.*

---

## 1. Question and hypotheses

PR-12.8 ended at rung 3 of the candidate's status ladder:
`PR12_8_READER_CONTRACT_CANDIDACY.md` §14.7, main `a0e621d`, verdict
**`contract-candidate-GO-seedbounded(W2:F1b)`** — the sole adjudicated
evidence base of this registration, cited by that exact name and never
without its bounds (§3). This memo pre-registers the rung-4 question:

**H0 (null, the honest default):** the candidate cannot survive
contract-grade *operational* discipline — a hermetic conformance suite
exposes a specification ambiguity in the contract definition; the
fail-closed and withdrawal mechanics cannot be demonstrated to execute
as registered; or certification-time re-verification finds drift.

**H1 (what a certification would support):** a conformant reference
implementation, exercised under a hermetic conformance suite and a
demonstrated tripwire/withdrawal mechanism, reproduces the frozen
envelope exactly at certification time — upon which
`s1-witness-alt-batch` may be certified as a **second, additive,
opt-in, seed-bounded** batch packet-reader contract beside (never in
place of, never modifying) PR-10 merge-abstain.

## 2. Scope

In scope: the certification gates (§6), required artifacts (§5), the
staged plan (§7), failure/withdrawal semantics (§8 with §4.4), and the
exact verdict vocabulary (§10) — all fixed here before any run. Out of
scope and forbidden: any policy or threshold motion (the W2:F1b block
stays at sha
`2f009cf29ce64dc21e7bd392ceac4bbc189bc5b197489c2edaf8cd3ed1022c15`,
pin `0afcb2b`); any envelope mutation; any new evidence generation
(the evidence base is closed at `a0e621d`); any S2/online claim; any
FAM-core import or change; any seed beyond s0–s2; any serving to any
consumer before an explicitly-approved certification verdict; any
change to the PR-10 contract ever.

## 3. Evidence base — carried verbatim (constitutive; dropping any item is kill §8.1)

The single adjudicated input is PR-12.8 Stage E at main `a0e621d`:
verdict **`contract-candidate-GO-seedbounded(W2:F1b)`** with all seven
candidacy gates G-R1–G-R7 passing (`pr12_8/candidacy_scan.json`, whose
`verdict_text` embeds the bounds below as constitutive parts). This
registration carries those bounds forward verbatim; every §10 verdict
under this memo must restate them in its own verdict text:

* **Traffic bounds (PR-12.8 §14.3):** plain-stale, jitter, and g5twin
  are `panel-insufficient` named scope bounds (plain-stale is
  structurally tie-free; jitter/g5twin have s0-only stems and no
  governance arm); mixed traffic is covered as **safe-by-silence only**
  (guard-closed everywhere, zero acting, zero utility on the type);
  the acting surface is exactly the one-shot harm class.
* **Contra-power bound (PR-12.8 §14.4, verbatim):** "contra-side
  safety = zero action on 319 in-scope W2 rows across 18 guard-closed
  cells (21 including the tie-free stale-soft cells), with a
  counterfactual guardless exposure of 73 acts / 70.0 wrong mass
  suppressed; per-cell guard mis-open rate bounded only at ≤13.3%
  (95%, iid-cell model); guard behavior inside the (−4, +23) margin
  corridor unobserved."
* **Seed bound (PR-12.8 §14.5, D-bounded):** valid only for the
  committed seeds' distributional regime — s0 (dev), s1, s2 — on the
  committed panel protocol (vision, 12 epochs, supersede-epoch 6); no
  claim about any other seed, write protocol, encoder, or drift. The
  `-seedbounded` suffix is permanent under this registration: no
  verdict here may drop it; only a future D-strong adjudication under
  its own registrations could ever support an unqualified name.
* **Monitoring terms (PR-12.8 §9 / `PR12_8_MONITORING_WITHDRAWAL.md`),
  T1–T7 verbatim:** T1 envelope-cell `witness_alt` multiset ≠ frozen
  envelope entry (exactness, any byte); T2 any I2 overlap row,
  incumbent-field deviation, or `witness_alt` at a tier other than
  `harness-heuristic` (exactness, any instance); T3 per-cell
  wrong-mass rate > 0.10 (truth-joined); T4 acting precision < 0.75
  where acted ≥ 1 row (truth-joined); T5 aggregate wrong-mass rate
  > 0.05 (truth-joined); T6 exercised cell's guard margin inside
  (−4, +23) (evidence-gap fail-closed, any instance); T7
  `n_fail_closed_events` > 0 on a cell (conformance, any instance).
  Withdrawal: automatic cell-level reversion to the committed
  dual-present/escalation posture as the registered minimum;
  candidate-wide `withdrawn-pending-review` for any T1/T2, any T5, or
  T3/T4 on two or more cells of one engagement; mandatory append-only
  event record; reinstatement only via a new pre-registration.
  Tighten-only, forever.

## 4. What certification means here (and does not)

**4.1 Relation to PR-10.** Certification under this memo would make
`s1-witness-alt-batch` a **second** certified reader contract:
additive, opt-in, batch-only, `harness-heuristic` tier, at the §3
bounds. It composes with — and never modifies — PR-10 merge-abstain:
the candidate's I3 invariant (incumbent immutability) is a
certification gate (§6 C-2) and a standing tripwire (T2). The PR-10
contract's own text, envelope, fields, and certification are untouched
by any outcome here.

**4.2 The contract-registry sentence.** The standing sentence "PR-10
merge-abstain is the only certified reader contract" changes **only**
upon an explicitly-approved `reader-contract-certified-…` verdict
(§10), and then only to: "PR-10 merge-abstain is the only
core-certified reader contract; `s1-witness-alt-batch@1.0` is a
certified opt-in **batch, harness-heuristic-tier, seed-bounded**
packet-reader contract at its registered bounds." No other wording is
authorized.

**4.3 Post-certification posture (registered now, effective only on
approval).** Blanket deferral remains the **default** posture on
witness-window rows. A certification verdict authorizes `witness_alt`
**only** for consumers that explicitly opt in to the contract, only on
completed W2 packet trees, only under the §3 monitoring terms with the
T1–T7 event log active. Non-opt-in consumers see no change whatsoever.

**4.4 Failure/withdrawal after certification.** The §3 monitoring
terms are the contract's standing conditions: any tripwire event
executes the registered withdrawal (cell reversion or candidate-wide
suspension), the event log is append-only, and reinstatement requires
a new pre-registration. A withdrawn contract does not revert the
registry sentence silently: withdrawal is recorded append-only, and
the contract remains listed as `withdrawn-pending-review` until a
reinstatement or revocation registration disposes of it.

## 5. Required artifacts (none exist; each produced only under §7 stage approval)

1. `harness/witness_alt_conformance_tests.py` — hermetic, stdlib-only
   test suite for the contract definition
   (`PR12_8_S1_CONTRACT_CANDIDATE.md`): eligibility conditions
   (each of the six §4 conditions exercised positively and
   negatively on synthetic packets), precedence I1 (abstention row
   never served `witness_alt` even if crafted to satisfy eligibility),
   I2 fail-closed on overlap, I4 tier invariance, first-tie-item
   semantics, W2-tree-only refusal, malformed-packet fail-closed
   (T7 event emission), and determinism (double pass byte-identity).
   Synthetic packets only — no committed packet is modified, no truth
   labels used.
2. `harness/witness_alt_withdrawal_demo.py` + demo artifacts under
   `pr12_9/` — a synthetic-tripwire demonstration that each T1–T7
   condition, when injected on synthetic inputs, produces (a) the
   registered reversion/suspension behavior and (b) a well-formed
   append-only event record. Demonstration only; nothing live.
3. `pr12_9/certification_scan.json` — the certification run record:
   gate table, re-verification hashes, conformance and demo results,
   verdict, and verdict text (which must embed §3 verbatim).
4. Contract re-issue `s1-witness-alt-batch@1.0` — **the one permitted
   transformation** of the contract document, explicitly authorized by
   this registration (satisfying its §10 change control and the
   append-only rule): a new file
   `PR12_9_S1_CONTRACT_V1.md` whose §§2–8 normative content is
   byte-carried from the 0.1-candidate document unchanged, with only
   the identity block updated (version `1.0`, status per ladder,
   certification citation). Any semantic difference from
   0.1-candidate voids certification (§8.5). Produced only if the
   certification verdict is the §10 GO **and** is explicitly approved.

## 6. Certification gates (hard; all must hold)

* **C-1 (envelope exactness at certification time):** the reference
  reader and an independent recomputation (the Stage E adjudicator
  mechanism, re-run) both reproduce the frozen envelope v0.2
  `witness_alt` multisets exactly on all 44 cells at the certification
  pin; all pins (emitter `2539686a…`, policy block `2f009cf2…`,
  envelope, packets) re-verified byte-identical.
* **C-2 (composition):** I1–I4 re-proven exactly on every envelope
  cell; zero incumbent-field deviations; PR-10-served fields
  byte-untouched.
* **C-3 (conformance suite):** every test of artifact §5.1 passes;
  any specification ambiguity discovered (a test whose expected
  behavior the contract text does not determine) is
  `certification-insufficient`, not a pass — the contract returns to
  its own change control (new version, rung 1).
* **C-4 (withdrawal mechanics demonstrated):** every T1–T7 injection
  in artifact §5.2 produces the registered behavior and a well-formed
  event record; withdrawal without a record, or a record without
  withdrawal, is a fail.
* **C-5 (bound carriage):** the §10 verdict text embeds §3 verbatim —
  traffic, contra-power, seed, and monitoring terms — and cites the
  PR-12.8 verdict by its exact name
  `contract-candidate-GO-seedbounded(W2:F1b)`.
* **C-6 (determinism):** internal double pass and an external
  second invocation of the certification run reproduce every `pr12_9/`
  output byte-identically; no timestamps in gated artifacts.
* **C-7 (approval separation):** a passing run yields the §10 GO
  verdict **on a branch**; the registry sentence (§4.2) and posture
  (§4.3) change only upon explicit human approval of that merge —
  the run itself changes nothing.

## 7. Staged plan (each stage separately authorized; none runs now)

* **Stage I — conformance + withdrawal mechanics.** Implement and run
  artifacts §5.1 and §5.2; record results append-only in §12. Kills
  §8 apply.
* **Stage II — certification run.** Implement the certification
  scanner (C-1/C-2/C-5/C-6 checks + assembly of C-3/C-4 results);
  emit `pr12_9/certification_scan.json` with exactly one §10 verdict.
* **Stage III — disposition.** Only on an explicitly-approved GO
  merge: produce artifact §5.4 and update the registry sentence per
  §4.2. Absent approval, the verdict rests on its branch with no
  effect.

## 8. Kill conditions (any → `certification-blocked`)

1. **Scope laundering:** any output dropping the `-seedbounded`
   suffix, omitting any §3 bound from a verdict text, citing the
   PR-12.8 verdict by any name other than
   `contract-candidate-GO-seedbounded(W2:F1b)`, claiming unqualified
   generality, or claiming S2/online applicability from S1 evidence.
2. Policy/threshold motion: the attested block sha differing anywhere;
   any 12.6 constant or T1–T7 tripwire relaxed.
3. Pin drift: any §6 C-1 input failing byte-identity at the
   certification pin; envelope or packet mutation.
4. Serving, deploying, promoting, or ingesting anything before an
   explicitly-approved GO merge; any live-acting claim at any time.
5. Contract-text drift: any semantic difference between
   `s1-witness-alt-batch@1.0` (§5.4) and the adjudicated
   0.1-candidate.
6. Writes outside `pr12_9/` and the named §5 harness artifacts; any
   FAM-core import; git dirtiness on any frozen surface.
7. Nondeterminism in any gated artifact.
8. Registry/posture change without the §6 C-7 explicit approval, or
   any wording change beyond §4.2's authorized sentence.

## 9. Byte-reproducibility

All §5 artifacts deterministic; certification scan double-passed
internally and reproduced externally; conformance tests hermetic
(synthetic inputs only); the withdrawal demo's event log is the only
append-only artifact and is excluded from byte-identity comparisons
across runs (events accumulate by design), with each event
individually well-formed and hash-chained to its predecessor.

## 10. Verdict vocabulary (exactly one, at Stage II)

* `reader-contract-certified-seedbounded(s1-witness-alt-batch@1.0, W2:F1b)`
  — all §6 gates pass. Even this verdict has **no effect until its
  merge is explicitly approved** (§6 C-7); upon approval, §4.2/§4.3
  take effect and rung 4 is reached.
* `certification-negative` — valid run, a §6 gate failed on its
  merits. The candidate remains at rung 3; the failure is recorded
  append-only.
* `certification-insufficient` — a required artifact cannot be
  completed as specified (e.g., C-3 specification ambiguity); names
  the deficiency and the registration path back.
* `certification-blocked` — any §8 kill.

## 11. Downstream-use boundary

This registration and every stage under it produce offline/batch
certification evidence about one adjudicated candidate — nothing else.
Until a §10 GO is merged with explicit approval: nothing is served,
deployed, promoted, or ingested; no reader contract changes; **PR-10
merge-abstain remains the only certified reader contract**; the
operational posture on witness-window rows remains **deferral**. After
such approval, §4.2–§4.4 apply exactly as written and nothing more: no
FAM-core integration, no S2/online seam, no prompting-use or
autonomous-use authorization, no LLM/agent reader certification, and
no change to PR-10 — ever, under this registration. PR-12.1–12.8
verdicts and registrations stand unchanged.

## 12. Results (reserved; append-only after separately-authorized stage runs)

Intentionally empty at pre-registration. §§1–11 above are the frozen
snapshot and are never rewritten.
