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

### 12.1 Stage I — conformance suite + withdrawal mechanics (run record, 2026-07-09)

*Separately authorized (Stage I only; explicit user approval,
2026-07-09). This subsection records results append-only per §7. It
issues **no §10 verdict** — verdicts exist only at Stage II, which
remains separately unauthorized. Evidence base unchanged: PR-12.8
Stage E at main `a0e621d`, verdict
`contract-candidate-GO-seedbounded(W2:F1b)`, carried with every §3
bound.*

**Artifacts produced (the §5.1 and §5.2 artifacts; nothing else was
written outside `pr12_9/`):**

| artifact | sha256 |
|---|---|
| `harness/witness_alt_conformance_tests.py` (§5.1) | `8c517fb5fbd6749c5770af5ab309bc29ea72c8802e0766b155b8a3e1cb7bfbcd` |
| `harness/witness_alt_withdrawal_demo.py` (§5.2) | `9a9c7e239b8cb163e1432ac5a1bde22b6137f109d2242509085fff627a173715` |
| `pr12_9/conformance_results.json` | `4423b315b91254b370fbb9a71d9d79721c787bb0a0a4f715c49b343f3fa25472` |
| `pr12_9/withdrawal_demo_report.json` | `003585525d61a01d67fc320fd779551c8c7459a43c5e673c9b65e5e90d0c3b90` |
| `pr12_9/withdrawal_demo_events.jsonl` (append-only event log, 26 events over two runs, hash-chained from genesis, chain verified) | `16e970a65c23b77c69d47892cd1b66a87ec6335d47b53207c38247d181eb87ad` (at commit; grows by design, §9) |

Policy-block attestation passed: the committed reference reader's
block equals the frozen scorer's byte-for-byte at the registered sha
`2f009cf29ce64dc21e7bd392ceac4bbc189bc5b197489c2edaf8cd3ed1022c15`.
Both reports are byte-identical across independent re-runs; frozen
surfaces git-clean before and after every run; synthetic inputs only
(no committed packet read, no truth label consumed — truth appears
solely as fabricated fixtures for the T3–T5 demonstrations).

**§5.1 conformance suite — architecture and result.** Every clause is
exercised against two subjects: *committed* (the merged reference
reader's `read_cell`, the implementation that froze envelope v0.2 and
that C-1 will re-run) and *conformant* (a suite-internal batch
pipeline implementing the contract §§2–7 text exactly, on the same
attested policy block). 25 checks, 50 subject-results, covering: all
six §4 eligibility conditions positively and negatively (including
the `≤` guard boundary at equal pair counts), I1 precedence on a
crafted abstain+eligible row, I2 fail-closed on a duplicate-join-key
overlap, I4 tier invariance and the §3 iff-field domains over every
emitted record, first-tie-item semantics in both directions,
W2-tree-only refusal (frozen `pol_f1b` refuses `W1` and `prototype`
at the shape gate), four malformed-packet variants (the T7 feed), and
double-pass determinism.

Result: **the conformant subject passes 25/25 — zero specification
ambiguities were found** (every crafted case has behavior the
contract text determines; the C-3 `certification-insufficient`
trigger is absent). **The committed reference reader passes 19/25**;
the 6 failing subject-results reduce to three findings, none touching
eligibility, the envelope multisets, composition, or any adjudicated
Stage E gate:

* **F1 — §3 `evidence_ptr` sourcing.** The contract registers
  "the row's audit-packet decision `evidence_ptr`, carried verbatim";
  the committed reader carries the memory-packet tie item's `text`
  (the fixed string "unresolved tie — two candidates, neither
  asserted") instead. The same divergence is present in the committed
  `pr12_8/served/*.csv` artifacts (e.g. row `pairD_oneshot_s1:e7:p4`,
  whose audit decision pointer reads "router(fork_events.csv): slot 2
  in ambiguous (pending) pair @epoch 7; …"). Non-envelope served
  field only.
* **F2 — §6 I2/§7 row-level fail-closed.** On a crafted overlap the
  committed `read_cell` leaves the affected row at `witness_alt`; the
  committed pipeline instead detects the overlap in its composition
  proof and kills the whole run (fail-closed by abort — safe
  direction, nothing is served as valid, but the registered per-row
  `defer` record is not produced).
* **F3 — §7 malformed-packet fail-closed (4 sub-checks).** The
  committed reader crashes on malformed packet or audit input
  (`JSONDecodeError`/`KeyError`): fail-closed by crash — nothing is
  served, never `witness_alt`, but the registered per-row `defer`
  record and recorded anomaly (the T7 event feed) are not produced.

On all committed panels these three paths never trigger (all 44
envelope cells are well-formed, overlap-free, and were proven so at
Stages C/E); the findings concern the contract's *operational*
envelope, exactly what rung 4 exists to test — H0's prediction
partially realized.

**§5.2 withdrawal demo — result: all 12 scenarios pass; every
tripwire T1–T7 demonstrated** with the registered behavior and a
well-formed, hash-chained event record (cell identity, packet shas,
tripwire id, measured vs registered value, contract/reader/monitor
versions): T1 envelope-byte mutation → candidate-wide
`withdrawn-pending-review`, after which a *healthy* other cell serves
zero `witness_alt` and a later clean exercise does not
self-reinstate; T2 (both a tier violation and an I2 overlap row) →
suspension; T3 per-cell 0.12 → cell reversion that leaves a healthy
sibling cell untouched; T3 on two cells of one engagement → the
registered candidate-wide escalation; T4 precision 0.5 → cell
reversion; T5 aggregate 0.06 with both per-cell rates under 0.10 →
suspension; T6 margin −1 → reversion *before* serving (2 decided, 0
served); T7 fail-closed event → reversion. Boundary honesty: T3/T4/T5
do **not** trip at exactly 0.10/0.75/0.05 (the constants are strict),
and the corridor is enforced as the open interval — margins −4 and
+23 do not trip. Posture↔event coupling verified bidirectionally.
Observation of record: a vacuous clean cell (margin 0, no pairs) lies
*inside* the registered corridor, so a faithful T6 monitor
fail-closes clean cells — tighter than the evidence requires, as the
monitoring registration permits, with zero utility lost (F1b never
acts on clean cells).

**Stage I determination (no §10 verdict).** The withdrawal-mechanics
half is discharged in full. The conformance half ran to completion
and establishes that the contract text is determinate and
implementable, but records 6 failing committed-reader results
(findings F1–F3). As registered, gate C-3 ("every test of artifact
§5.1 passes") cannot pass while these stand: a Stage II run today
would yield `certification-negative` on its merits. The registered
paths back are outside Stage I's authority and are **not** taken
here: remediating the reference reader and/or the contract text is a
§5.4/§10 change-control matter (any semantic change produces a
0.2-candidate and returns the candidate to rung 1; kill §8.5 guards
the re-issue), each step under its own explicit authorization.

**Boundary.** Stage I evidence only. Nothing is certified, served,
deployed, promoted, or ingested; no FAM-core change; no registry or
posture change; Stages II and III remain separately unauthorized.
**PR-10 merge-abstain remains the only certified reader contract**;
the operational posture on witness-window rows remains **deferral**.
PR-12.1–12.8 verdicts and registrations stand unchanged.

### 12.2 Stage I remediation — reference reader brought to §7 conformance (run record, 2026-07-09)

*Separately authorized (explicit user approval, 2026-07-09): remediate
the committed reference reader only; re-version the contract only if
the fix reveals the text actually underspecified. Target behavior,
fixed in advance: exact audit-packet `evidence_ptr` passthrough;
per-row `defer` for I2 overlap; malformed packets handled as per-row
`defer` with an append-only anomaly/T7 record; no `witness_alt`
assertion on malformed input; no whole-run abort except
non-recoverable harness/environment failure. This subsection issues
**no §10 verdict**.*

**What changed.** `harness/witness_alt_reference_reader.py`
`read_cell` only (plus a `_defer_record` helper). The §5 sha-attested
policy block is byte-untouched (attestation re-passes at
`2f009cf2…`); `main()`, the panel, the pins, and every kill gate are
unchanged. The remediated `read_cell`: (F1) populates `evidence_ptr`
from the row's audit-packet **serving decision**, carried verbatim;
(F2) downgrades any I2-overlap `witness_alt` row to `defer` with a
recorded `fail_closed` anomaly; (F3) wraps per-row parsing and
eligibility so malformed packet content yields a per-row `defer`
record plus a `fail_closed` anomaly (the T7 feed) — an unparseable
*audit* packet fail-closes the whole cell to `defer` (batch
semantics, §2) with one recorded cell-level anomaly. Packet content
can no longer abort a run.

**The referent question (why the contract is NOT re-versioned).**
Governed audit rows carry multiple `decisions` entries — one
row-level serving decision (`item_id = served_answer@<query_id>`)
plus slot-level `withheld` records (decisions about items, not about
the row's serving). "The row's audit-packet decision `evidence_ptr`"
(contract §3) therefore has a unique natural referent — the serving
decision — and the §2 provenance clause (governed emission only)
makes that uniqueness part of the contract's input domain. Verified
empirically: **all 116,991 audit rows across the committed 44-cell
panel carry exactly one serving decision, in first position, with a
non-empty pointer** (in particular all 2,052 eligible rows). Zero or
multiple serving decisions, or an empty pointer, on an
otherwise-eligible row is off-domain and fail-closes that row (§7).
The text is determinate on its domain; no underspecification found;
`0.1-candidate` stands unchanged. The suite's conformant pipeline,
which had used a first-decision shortcut, was corrected to the same
serving-decision referent, and two new checks pin it (multi-decision
referent selection; missing-serving-decision fail-closed).

**Verification.**

* Conformance suite: **all-pass — 27 checks (25 prior + 2 new), both
  subjects, zero failures; findings F1–F3 cleared**; still zero
  specification ambiguities; results byte-identical across runs. The
  F1/F2 finding annotations in the suite were made conditional on an
  observed failure (annotation-only; no expectation changed).
* Envelope reproduction (throwaway git worktree; the committed
  `pr12_8/` artifacts in the main tree are untouched): the remediated
  reader emits verdict `stageC-envelope-frozen`, totals identical
  (44 cells / 116,991 rows / 2,052 `witness_alt` / 876 abstain /
  composition 44/44 / act-set cross-check 34/34), **all 44 envelope
  cell entries byte-identical**, prior-version preservation gate
  passed; the only envelope-file difference is the known
  self-referential `prior_version_cells_checked` counter (28→44,
  deterministic-given-HEAD, documented at Stage A).
* Served-decision delta, field-precise over all 116,991 rows ×
  44 files: **exactly 2,052 differing values — every one on a
  `witness_alt` row, every one in `evidence_ptr`** (tie text →
  serving-decision pointer), zero diffs in any other field, zero
  diffs on any non-`witness_alt` row. The committed `pr12_8/served/`
  CSVs are intentionally left as the historical Stage C record
  produced at their pins; corrected pointers materialize in any
  future served output (Stage II C-1 re-run, under its own
  authorization, at its own pin).
* Withdrawal demo report byte-identical (the demo does not exercise
  `read_cell`); frozen surfaces git-clean after the run.

**Artifacts at this remediation:**

| artifact | sha256 |
|---|---|
| `harness/witness_alt_reference_reader.py` (remediated) | `b8219f78e3943fef7949e468c755848582e3854c8dfd990317c376b51e7fd7c8` |
| `harness/witness_alt_conformance_tests.py` (referent + annotations) | `c8d92abddbaa640e7b0482f431f2d44ba94fa3e2c58d8a943be803ec4035458f` |
| `pr12_9/conformance_results.json` (all-pass state) | `7e15a240803294cfb3e3d4f134f2901c4ca9b4ca075e167ee7763abdce98f88e` |

The §12.1 record and its sha table remain the Stage I finding record
as run; this subsection supersedes nothing and is append-only.
**Consequence:** with F1–F3 remediated and the suite all-pass, the
C-3 obstacle recorded in §12.1 is cleared on this branch; gate C-1's
envelope-exactness precondition is re-evidenced (multisets and cell
entries reproduce). Stage II remains **separately unauthorized**.

**Boundary.** Remediation evidence only. Nothing is certified,
served, deployed, promoted, or ingested; no FAM-core change; no
registry or posture change; the contract text is unchanged at
`0.1-candidate`. **PR-10 merge-abstain remains the only certified
reader contract**; the operational posture on witness-window rows
remains **deferral**. PR-12.1–12.8 verdicts and registrations stand
unchanged.

### 12.3 Stage II — certification run (run record, 2026-07-09)

*Separately authorized (explicit user approval, 2026-07-09: "Stage II
authorized"). Implements the §7 Stage II certification scanner
(`harness/witness_alt_certification_scan.py`, the §7-ordered
implementation whose registered output is artifact §5.3) and runs it.
Per gate C-7, the verdict below **has no effect**: it rests on this
branch; the §4.2 registry sentence and §4.3 posture change only upon
explicit human approval of the merge (Stage III, separately
unauthorized).*

**Certification pin:** `10b9335` (main at the Stage I remediation
merge). Every consumed input — 93 manifest entries: all 88 packet
files, the envelope, the contract and monitoring documents, the
PR-12.8 candidacy scan, and the emitter — verified byte-identical to
the pin; emitter at `2539686a…`; the policy block sha-identical at
`2f009cf2…` across all five attested sources; the evidence base's
verdict intact under its exact name
`contract-candidate-GO-seedbounded(W2:F1b)`.

**Gate results — all seven pass:**

| gate | result |
|---|---|
| C-1 envelope exactness | PASS — dual mechanism: the remediated reference reader (`read_cell`) **and** the Stage E adjudicator's independent recomputation (imported from committed `candidacy_adjudicate.py`, its own policy-block copy) each reproduce the frozen envelope v0.2 `witness_alt` multisets exactly on all 44 cells |
| C-2 composition | PASS — I1–I4 recomputed on every cell; outcome counts equal the envelope; 0 incumbent-field deviations; 0 tier violations; 0 fail-closed anomalies on committed cells |
| C-3 conformance suite | PASS — re-run at the pin: all-pass, 27 checks, zero failing subject-results, **zero specification ambiguities** (the `certification-insufficient` trigger is absent) |
| C-4 withdrawal mechanics | PASS — re-run: all 12 scenarios, T1–T7 all demonstrated, hash chain verified, posture↔event coupling verified |
| C-5 bound carriage | PASS — the verdict text embeds this registration's §3 **verbatim** (extracted from the pinned memo blob) and cites the PR-12.8 verdict by its exact name; zero required phrases missing |
| C-6 determinism | PASS — internal double pass identical; **external second full invocation reproduces `certification_scan.json`, `conformance_results.json`, and `withdrawal_demo_report.json` byte-identically** (verified by sha256 comparison); no timestamps; the demo event log is the registered §9 append-only exclusion (26 committed → 52 after the two C-4 exercises, chain verified from genesis over all 52) |
| C-7 approval separation | PASS — structural: this run changed no registry sentence and no posture |

**Verdict (exactly one, §10):**
**`reader-contract-certified-seedbounded(s1-witness-alt-batch@1.0, W2:F1b)`**
— emitted **on this branch only**, with the §3 traffic, contra-power,
seed, and monitoring bounds embedded verbatim in its `verdict_text`
as constitutive parts and the `-seedbounded` qualifier permanent. No
kill condition fired.

**Artifacts:**

| artifact | sha256 |
|---|---|
| `harness/witness_alt_certification_scan.py` | `31d8d3511e4aaa916822ef5b8e2e49e1bf158780d62dca5d0c39c009840357a5` |
| `pr12_9/certification_scan.json` (§5.3) | `431ef56367aa9fe8a7c38701230ec6bcb91f1c47ff5d1597fcddff59efcf2213` |
| `pr12_9/withdrawal_demo_events.jsonl` (grown append-only, 52 events) | `4f63105f394103a1d02bcb098716174c5c410566cf57d5f6fb9d8e4df8fff5b3` (at commit; grows by design) |

**What this verdict does and does not do.** Per C-7 and §10: it does
**nothing** until its merge is explicitly approved. Upon such
approval — and only then — Stage III produces the §5.4 contract
re-issue `PR12_9_S1_CONTRACT_V1.md` (byte-carrying the 0.1-candidate
normative text) and the registry sentence changes to exactly the §4.2
wording; the §4.3 opt-in posture takes effect under the standing
T1–T7 monitoring terms. Absent approval, the verdict rests here with
no effect of any kind.

**Boundary.** Stage II run record only. Nothing is served, deployed,
promoted, or ingested; no FAM-core change; no registry or posture
change; Stage III remains separately unauthorized. **PR-10
merge-abstain remains the only certified reader contract**; the
operational posture on witness-window rows remains **deferral**.
PR-12.1–12.8 verdicts and registrations stand unchanged.

### 12.4 Stage III — disposition: contract re-issue produced (run record, 2026-07-09; effect pending merge approval)

*Separately authorized (explicit user instruction, 2026-07-09). The
approval of record for the Stage II merge (`8875b72`) was **scoped**:
"limited to preserving the Stage II certification-run evidence and §10
GO verdict on main. It does not authorize Stage III, registry
re-issue, serving, deployment, ingestion, FAM-core change, posture
change, or any opt-in consumer enablement" — with Stage III production
then separately ordered to begin. Accordingly: the §10 GO verdict now
sits on main **as evidence without operative effect**, and this
subsection records the Stage III artifact production. **Rung 4, the
§4.2 registry sentence, and the §4.3 posture take effect only upon
explicit approval of THIS Stage III branch's merge** — the C-7
separation is preserved one more step, at the user's direction.*

**Artifact §5.4 produced:** `PR12_9_S1_CONTRACT_V1.md`
(sha256 `8cc3e80f00fea09d0eac21e4dffe60b649a86814396c72a112dc39eef49df2f9`)
— the one permitted transformation of the adjudicated 0.1-candidate:

* §§2–8 normative content **byte-carried unchanged**, extracted
  programmatically from the pinned committed blob
  (`8875b72:PR12_8_S1_CONTRACT_CANDIDATE.md`) and byte-verified
  identical (carried-span sha256
  `1a2b80a716f232ee3eb94d94b8614d9dfb6c3a3506aed3e5d46757cf24f50a78`);
  the §8.5 kill (semantic drift from the adjudicated candidate) is
  structurally untrippable by construction.
* Identity block updated exactly as §5.4 permits: version `1.0`,
  status `certified-seedbounded, pending effect` (rung 4 per ladder,
  effective only on approved merge), certification citation
  (Stage II verdict, scan artifact, pin `10b9335`, evidence merge
  `8875b72`), evidence-base chain extended through PR-12.9.
* Non-normative frame (§9 ladder, §10 change control, §11 boundary)
  updated to the certified re-issue's position: any modification →
  `1.1`, voids certification, rung 1, new pre-registration; boundary
  restates the §4.3 opt-in-only posture and the permanent
  prohibitions.

**The registry sentence (§4.2, exact wording, effective only on
approved merge of this branch):** "PR-10 merge-abstain is the only
core-certified reader contract; `s1-witness-alt-batch@1.0` is a
certified opt-in **batch, harness-heuristic-tier, seed-bounded**
packet-reader contract at its registered bounds." No other wording is
authorized; historical documents are not rewritten (append-only repo —
their boundary statements stand as records of their time).

**Explicitly not done, per the scoped approval:** no consumer is
enrolled or enabled; nothing is served, deployed, promoted, or
ingested; no FAM-core change; no posture change — blanket deferral
remains the operational posture on witness-window rows, and even
after any future approval it remains the **default**, with
`witness_alt` reaching only explicit opt-in batch consumers under the
standing T1–T7 monitoring terms and append-only event log.

**Boundary.** Until this branch's merge is explicitly approved:
**PR-10 merge-abstain remains the only certified reader contract**;
the operational posture on witness-window rows remains **deferral**;
rung 4 is not reached. PR-12.1–12.8 verdicts and registrations stand
unchanged.
