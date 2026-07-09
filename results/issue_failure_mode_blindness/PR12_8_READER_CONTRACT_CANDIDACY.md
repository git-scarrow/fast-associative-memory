# PR-12.8 — Reader-contract candidacy pathway for W2:F1b (pre-registration DRAFT, design-only)

*Design-only. No implementation, no scan, no packet generation, no
scoring, no serving seam, no envelope artifact, no harness-code or
FAM-core change is authorized by this memo. `harness_boundary_sim.py`,
`action_boundary_score.py`, `action_boundary_holdout_score.py`,
`reader_utility_score.py`, and every PR-12.1–12.7
artifact/verdict/gate/output/memo section remain byte-frozen. **PR-10
merge-abstain remains the only certified reader contract**; the
operational posture on witness-window rows remains **deferral**. This
memo converts the 2026-07-08 contract-readiness analysis into a
registered design: it records determinations, registers a staged
evidence plan whose every stage requires separate explicit approval,
and fixes the candidacy verdict vocabulary in advance. §§1–13 are
frozen at commit; §14 is reserved append-only for separately-authorized
stage results. Even a full `contract-candidate-GO` under this memo is
NOT a certification and NOT a contract change — certification, if ever,
is a further separate pre-registration taking this PR's output as one
input.*

---

## 1. Question and hypotheses

PR-12.3–12.7 (all merged to main, latest `7e98518`) end at
`holdout-validity-GO(W2:F1b)`: offline, byte-reproducible,
pair-axis-generalized evidence that one frozen, parameter-free,
label-free act-versus-defer policy is useful on the one-shot harm class
and safe on contra traffic. This memo pre-registers the pathway from
that evidence tier to **reader-contract candidacy** — the highest
status short of certification — under the certification anatomy the
repo already set with PR-10.

**H0 (null, the honest default):** carrying W2:F1b onto a serving
surface under PR-10-grade discipline exposes a leak the offline chain
could not see — the quiet-cell guard misbehaves off its observed
bimodal regime or on mixed traffic; the acted-row envelope is not
exactly reproducible by a reference reader; composition with
merge-abstain perturbs a certified field; or the panel extension
reveals wrong-action mass above a registered ceiling.

**H1 (what a GO would support, and only offline/batch):** a batch
packet-reader contract candidate
`{answer | abstain(merge_suspect_led) | witness_alt(c, provenance)}`
exists whose acted-row envelope is frozen to exactness, whose
composition with the incumbent contract is invariant-clean, whose
traffic/seed scope is stated honestly, and which passes every §8
candidacy gate with §10 kill conditions clean.

## 2. Scope

In scope: design determinations (§4–§6), a staged evidence plan (§7),
candidacy gates (§8), verdict vocabulary (§12), all offline/batch over
committed or deterministically-regenerable artifacts. Out of scope and
explicitly forbidden here: any policy change, any threshold change, any
new policy family, any fitting on any row, any FAM-core/engine
execution or change, any online/driver serving seam (§4 defers S2
entirely), any new random seed without the §7 Stage D registration, any
deployment/prompting/promotion/ingestion/autonomous/live-acting claim,
any reader-contract change. PR-12.1–12.7 verdicts stand unchanged.

## 3. Evidence base (committed; verbatim from the 12.3–12.7 scan JSONs)

| PR | verdict | what it established |
|---|---|---|
| 12.3 `aece0d4` | attribution-evidence-GO(W1,W2) | witness-window set construction recovers truth (containment 0.92–1.0, chance-adjusted); the prior failure was set construction, not selectivity |
| 12.4 `2226d9d` | replication-GO(W1,W2) | containment replicates per-seed s1/s2 (0.995–1.0); max resolution lag 1 |
| 12.5 `0afcb2b` | reader-utility-evidence-GO | frozen parameter-free readers double-dominate the certified-only reader (P-assert 0.000–0.246); deployed vote is anti-signal on one-shot; contra inverts → acting must be disposition-scoped |
| 12.6 `9a7537e` | action-boundary-evidence-GO(W2:F1b) | exactly one act-vs-defer combo passes; the quiet-cell guard separates harm classes perfectly on B/D |
| 12.7 `7e98518` | holdout-validity-GO(W2:F1b) | the boundary generalizes on the pair axis (C/E): G-H1–G-H5 valid, guard separation perfect again, sole passing combo, zero contra action |

Aggregate scored footprint: **1,728 one-shot in-scope W2 rows** (8
units, 4 pairs, 2 test/holdout seeds; coverage 0.94–1.0, precision
0.90–1.0) and **135 contra in-scope rows** with **zero actions**
(12.6: 8/5/53/45; 12.7: 7/0/12/5). Governance placement: this is the
harness-layer answer to the closure memo §5 "one-shot ambiguity"
residual (3,834 wrong none-rows; 32 permanently-ambiguous pairs per
cell), and it evades PR-11.1's rejected abstention trigger (33–36%
false abstention on correct traffic) by construction — it acts on the
witness alternative on dual-present rows instead of abstaining. The
contradiction-fork residual remains untouched by design (guard closed;
F1b defers).

## 4. Determination D1 — serving surface is S1 (batch packet-reader); S2 is excluded

**Finding of record (the retrospective-feature problem):** as scored in
12.6/12.7, `CellCtx` is end-of-run information. The emitter writes the
`contradiction_pair_review`/`ambiguous_pair_review` records the
quiet-cell guard counts **after** the probe loop, and `never_resolving`
means "never resolves by the final epoch". Two candidate surfaces
therefore differ in validity, not merely in engineering:

* **S1 — batch packet-reader contract (harness layer).** The reader
  consumes a *completed* cell's `memory_packet.jsonl` /
  `audit_packet.jsonl` and may rely on the witness alternative on
  qualifying rows. Look-ahead is legitimate — the packet is a finished
  artifact — and the 12.3–12.7 evidence is directly on-surface. This is
  also the layer the governance closure §7 fork assigned lifecycle
  semantics to.
* **S2 — online driver read seam (FAM-core layer, PR-10-style).** At
  serve time inside a run, the guard's inputs are unknowable without
  seeing the future. Any online surrogate (running pair counts at epoch
  *t*, "unresolved so far") is a **different policy** with no committed
  evidence.

**Registered determination: the candidate surface is S1.** S2 is
excluded from PR-12.8 entirely; pursuing it would require a new policy
definition and a full re-scoring chain under its own pre-registration,
and nothing in this memo or its stages may be cited as S2 evidence.
The candidacy verdict (§12), if GO, names a **batch packet-reader
contract candidate** and nothing else.

## 5. Determination D2 — candidate served vocabulary (registered, not certified)

The candidate contract extends the certified vocabulary additively; the
incumbent's fields are untouched:

* Per in-scope row the reader serves exactly one of:
  * `answer` — the deployed answer, unchanged (incumbent semantics);
  * `abstain(merge_suspect_led)` — incumbent PR-10 semantics, unchanged;
  * `witness_alt(c, provenance)` — **candidate**: rely on decode class
    `c` = the sole witness alternative of a qualifying W2 dual-present
    row, carrying (i) the dual-present evidence pointer (the row's
    `unresolved_tie` item), (ii) the certification tier already present
    in the packets (`harness-heuristic` — explicitly NOT
    `core-certified`), and (iii) the policy identifier `W2:F1b` at its
    frozen sha (§6).
* `witness_alt` is emitted only where the byte-frozen `pol_f1b`
  (policy block sha `2f009cf29ce64dc21e7bd392ceac4bbc189bc5b197489c2edaf8cd3ed1022c15`,
  attested against `action_boundary_score.py` per 12.7 G-H3) returns an
  action; every other in-scope row keeps its committed
  dual-present/escalation posture (DEFER).
* Encoding is a registration deliverable of Stage C (§7): a served
  field specification in the PR-10 style (named columns/fields, additive
  only), fixed *before* any envelope gate is written.

## 6. Determination D3 — composition invariants with the incumbent contract

Registered invariants (to be *proven*, not assumed, in Stage C):

* **I1 (precedence):** `abstain ≻ witness_alt ≻ defer`. A row abstained
  under the certified contract is never eligible for `witness_alt`.
* **I2 (disjointness):** the F1b in-scope row set (pending-led served
  dual-present W2 rows carrying an `unresolved_tie` item) is disjoint
  from the certified abstention set (merge-suspect-led; the emitters'
  kill-4 pins certified-abstain == merge-led, and abstained rows carry
  no `unresolved_tie` item). Stage C must verify I2 as an exact
  set-level check on every panel cell, not cite it as structure.
* **I3 (incumbent immutability):** no field the PR-10 contract serves
  changes value, position, or encoding on any row; the candidate is
  additive-only. Any violation is kill §10.2.
* **I4 (no new authority):** `witness_alt` rows remain
  `harness-heuristic` tier; nothing is promoted to `core-certified`.

## 7. Registered staged evidence plan (each stage separately authorized later; none runs now)

**Stage A — traffic-axis panel extension (generation + scoring).**
Extend the witness-window panel to the committed-but-never-W2-scored
traffic types: mixed (pairs B–E), plain-stale (B–E), stale-soft
co-occurrence cells (the merge-abstain interaction case: cells where
certified abstentions fire *and* dual-present rows exist), with jitter
(pairA) and g5twin (pairD) as report-only anchors. Mechanism: the
PR-12.7 erratum-E1 pattern reused byte-for-byte — an additive
`scan12_8_panel` manifest of committed `pr10/governed` run-stems, the
merged `action_boundary_holdout_generate.py` invocation discipline
gated on the committed anchor
(`pr12_4/W2/pairD_oneshot_s1`), a new cache
`pr12_8_panel_cache/`, and a standalone scorer reusing the 12.7 G-H
gate structure with the same sha-attested policy block. Power floors
(registered now): ≥30 in-scope W2 rows aggregated per new traffic
type, else that type returns `panel-insufficient` and is recorded as a
scope bound, not a pass. **Guard-regime measurement:** every new cell's
`(n_contradiction_pairs, n_ambiguous_pairs)` is recorded; the committed
observation to beat is strict bimodality (one-shot 0–28 vs 25–32;
contra 207–209 vs 25–32; nearest flip-point margin 4 at
`pairD_oneshot_s1`). If any new cell lands near the boundary
(|margin| ≤ 4), its F1b behavior is measured and reported per-row; if
no committed input produces such a cell, the candidacy carries the
registered scope bound **"guard certified only in the observed bimodal
regime"** — stated, never silently dropped.

**Stage B — contra-power accounting.** No new mechanism: Stage A's
mixed/plain-stale cells either raise the contra-side in-scope row count
materially above the current 135 or they do not. The outcome is
recorded either way; a thin result is a **named scope bound** on the
candidacy (contra safety = zero action on N rows, N stated), not a
blocker, because F1b's contra posture is structural (guard closed) and
12.6/12.7 already gate it hard on every scored unit.

**Stage C — served-field registration + composition proof + envelope
freeze.** (i) Fix the §5 encoding as a schema document; (ii) prove
I1–I4 exactly on every panel cell (12.6 + 12.7 + Stage A); (iii) freeze
the **exactness envelope**: a committed artifact listing, per cell, the
exact acted-row multiset, acted class per row, and expected-correct
mass — sourced from the committed `rows_*_F1b.csv` tables and Stage A's
equivalents, hash-pinned. From that point the envelope is a PR-10-style
frozen expected value: not a floor to re-earn, not a margin to move; a
reference reader (stdlib, read-only) must reproduce it byte-for-byte
twice (internal + external double pass).

**Stage D — seed-axis branch decision (two registered branches; one
must be chosen explicitly, neither is default).**
* **D-strong:** new-seed (s3+) engine runs of the governed panel under
  the established compute/custody discipline (gentoo compute, darwin
  byte-verify), then witness-window generation + scoring on the new
  seeds. This is the only branch that can support a seed-general
  candidacy. It is a FAM-engine execution and sits outside every
  offline authorization in this memo — it requires its own explicit
  approval with its own run matrix.
* **D-bounded:** no new seeds; the candidacy verdict carries the
  registered bound **"certified-candidate for the committed seeds'
  distributional regime (s0–s2) only"** — a materially weaker contract
  candidate, named as such in the verdict string itself
  (`contract-candidate-GO-seedbounded`), never silently equated with
  D-strong.

**Stage E — candidacy adjudication.** Evaluate the §8 gates over
everything above; emit exactly one §12 verdict. Monitoring/withdrawal
semantics (§9) must be registered before Stage E runs.

## 8. Candidacy gates (hard; all must hold for any GO)

* **G-R1 (envelope exactness):** the Stage C reference reader
  reproduces the frozen acted-row envelope byte-for-byte on every panel
  cell; internal and external double passes identical.
* **G-R2 (composition):** I1–I4 proven exactly on every cell; zero
  overlap rows; zero incumbent-field deviations.
* **G-R3 (traffic coverage or honest bound):** every §7-Stage-A traffic
  type is either scored with its power floor met and every 12.6-§12–§15
  ceiling/floor satisfied by W2:F1b on it, or recorded as
  `panel-insufficient` **and** carried as a named scope bound in the
  verdict; no third state.
* **G-R4 (guard-regime statement):** the bimodality observation and the
  flip-point margin table are recorded across the full panel; any
  boundary-regime cell's behavior is reported per-row; absent such a
  cell, the regime scope bound is present in the verdict text.
* **G-R5 (safety ceilings on the extended panel):** the registered
  12.6 constants — precision floor 0.75, coverage floor 0.25 (one-shot
  aggregate), contra wrong-action ceiling 0.05/unit, per-unit ceiling
  0.10, global ceiling 0.05 — bind unchanged on every newly scored
  unit. No threshold moves; near-ceiling margins (the committed 0.0943
  on `pairE_oneshot_s1`, dev-s0 0.1163 breach) are reported alongside.
* **G-R6 (seed-axis honesty):** the Stage D branch actually taken is
  named in the verdict string; D-bounded can never emit the unqualified
  GO.
* **G-R7 (no-tuning attestation):** the policy block sha
  (`2f009cf2…`) is attested against the frozen 12.6 source at every
  stage; zero code path fits, selects, or thresholds against any row.

## 9. Monitoring and withdrawal semantics (registration deliverable, pre-Stage-E)

PR-10 needed no monitoring clause because its gates are exactness — 
nothing drifts without failing G3. An acting candidate whose safety
gates are inequalities cannot borrow that: before Stage E, a
monitoring registration must fix (i) which quantities are watched when
the contract candidate is exercised on new packets (per-cell wrong-mass
rate, guard margin, acted-row precision where truth later becomes
joinable), (ii) the tripwire values (registered now as the 12.6
constants themselves — any measured breach on any exercised cell), and
(iii) what withdrawal means for an opt-in batch contract: the candidate
reverts that cell to the committed dual-present/escalation posture and
the event is recorded append-only. No tripwire may be moved by the
monitoring registration; it may only bind tighter.

## 10. Kill / contamination conditions (any → `candidacy-blocked`)

1. Input drift from the pinned commits; a panel packet not regenerable
   byte-identically through the unmodified emitter (12.7 G-H2
   discipline; emitter sha `2539686a…`).
2. Any incumbent-served field changed on any row (I3), or any
   abstained row served `witness_alt` (I1/I2).
3. Policy/threshold motion: any change to the sha-attested policy
   block, any constant moved, any new/removed policy family, any fit
   against any row (G-R7).
4. Label leak: any policy-visible path reading truth labels, run-stem
   CSVs, scan JSONs, `pr12_5/`–`pr12_7/` scoring outputs, PR-4
   governance rows, or cell/pair/arm/seed identifiers.
5. An ACT outside a row's presented set; a truth-join miss.
6. Writes outside the declared PR-12.8 artifact paths (§11);
   `git status` dirtiness on any frozen surface.
7. Nondeterminism: any internal double pass or external re-run
   differing in any byte of any PR-12.8 output.
8. Contra collapse: any contra unit excluded, reweighted, or averaged
   into one-shot aggregates; any GO text omitting the contra gates.
9. Scope laundering: any output language equating D-bounded with
   D-strong, dropping a G-R3/G-R4 scope bound from the verdict text,
   claiming certification, deployment readiness, live acting, prompting
   use, promotion, ingestion, autonomous use, or a reader-contract
   change.
10. Any S2/online claim sourced from S1 evidence (§4).

## 11. Artifact paths (none created by this memo)

* This memo:
  `results/issue_failure_mode_blindness/PR12_8_READER_CONTRACT_CANDIDACY.md`
  (results appended as §14+, append-only; §§1–13 frozen).
* Stage A (separately authorized): additive `scan12_8_panel` block in
  `harness/harness_policy.json`; cache
  `results/issue_failure_mode_blindness/pr12_8_panel_cache/`; standalone
  generator invocation via the merged
  `harness/action_boundary_holdout_generate.py` pattern (a PR-12.8
  driver file if needed, byte-equivalence-gated on the same committed
  anchor); scorer outputs under
  `results/issue_failure_mode_blindness/pr12_8/`.
* Stage C (separately authorized): served-field schema document;
  envelope artifact `pr12_8/f1b_envelope.json`; reference reader
  `harness/witness_alt_reference_reader.py` (stdlib, read-only).
* Stage D-strong (separately authorized, engine execution): its own
  run-matrix registration — not specifiable further here.
* Stage E (separately authorized): `pr12_8/candidacy_scan.json`.
* The 2026-07-08 readiness analysis this memo supersedes is absorbed
  here in full (evidence inventory §3, surface finding §4, gap
  registrations §5–§9); the standalone analysis file is retired rather
  than committed, to keep one source of truth.

## 12. Verdict vocabulary (exactly one, at Stage E)

* `contract-candidate-GO(W2:F1b)` — all §8 gates pass under Stage
  D-strong. W2:F1b becomes a **certified-candidate batch packet-reader
  contract** — a status that authorizes exactly one thing: a future
  certification pre-registration may cite this verdict as its evidence
  base. Nothing is served to anyone by this verdict.
* `contract-candidate-GO-seedbounded(W2:F1b)` — as above under Stage
  D-bounded, carrying the seed-regime bound and any G-R3/G-R4 scope
  bounds in the verdict text itself.
* `contract-candidacy-negative` — panel valid and powered, but W2:F1b
  fails a §8 gate on it: the offline chain does not survive
  contract-grade discipline. A real, informative result; the 12.3–12.7
  verdicts stand unchanged.
* `panel-insufficient` — Stage A power floors unmet across the board:
  no candidacy claim, registers the next input-generation step.
* `candidacy-blocked` — any §10 kill.

## 13. Downstream-use boundary

This memo and every stage under it produce offline/batch evidence about
a candidate contract over committed and deterministically-regenerable
packets — nothing else. No stage authorizes deployment, live acting,
prompting use, promotion to any policy version, memory ingestion or
write-back, autonomous downstream use, LLM/agent reader certification,
an online/driver seam, or any reader-contract change. Even
`contract-candidate-GO(W2:F1b)` changes no contract and no posture:
**PR-10 merge-abstain remains the only certified reader contract**, and
the operational posture on witness-window rows remains **deferral**
unless and until a separate certification pre-registration — with its
own gates, its own approval, and this PR's output as merely one input —
proposes otherwise. PR-12.1–12.7 verdicts stand unchanged.

## 14. Results (reserved; append-only after separately-authorized stage runs)

Intentionally empty at pre-registration. §§1–13 above are the frozen
snapshot and are never rewritten. Implementation of every stage
(A through E) is **not** authorized by this memo and requires separate
explicit approval per stage.

### 14.1 Stage C(i) discharged — S1 contract-candidate definition registered (append-only; 2026-07-08)

Separately authorized, definition-only. The served-field/contract
registration deliverable of §7 Stage C(i) is discharged by
`results/issue_failure_mode_blindness/PR12_8_S1_CONTRACT_CANDIDATE.md`
— contract_id `s1-witness-alt-batch`, version `0.1-candidate`, status
**defined** (rung 1 of its §9 status ladder; no operative force). It
fixes the §5/§6 determinations as normative contract clauses: the
served-decision record schema (additive-only; `witness_alt` always
`harness-heuristic` tier), the §4 eligibility conditions as a normative
restatement with the sha-attested policy block
(`2f009cf2…`) authoritative on any divergence, the I1–I4 composition
clauses, batch-only semantics with the retrospective-feature basis
restated, fail-closed conformance requirements, prohibitions, and
change control (any edit → new version → all Stage C–E work void).
Nothing else ran: no reference reader exists, no composition proof
(C(ii)), no envelope freeze (C(iii)), no Stage A/B/D/E work, no
scoring, no packet generation, no FAM-core change. Even fully
adjudicated, the candidate authorizes serving nothing to anyone —
certification is a separate future pre-registration (§13). PR-10
merge-abstain remains the only certified reader contract; posture
remains deferral.

### 14.2 Stage C(ii)+(iii) discharged — composition proven, envelope frozen (append-only; 2026-07-08)

Separately authorized. New `harness/witness_alt_reference_reader.py`
(stdlib + subprocess-git; label-free by construction — every serving
decision from the packet pair alone, committed truth-joined tables
opened only post-decision for cross-check; §5 policy block copied
verbatim and runtime-attested sha `2f009cf2…` against the frozen 12.6
scorer) run twice over the full committed W2 panel at panel pin
`8cbf870` (the Stage C(i) merge): 28 cells — pr12_3 s0 (6, dev),
pr12_4 s1/s2 (12), pr12_7_holdout_cache C/E (10) — 75,631 packet rows,
74 pinned inputs, zero kills, zero anomalies.

**C(ii) composition proof — PASS on all 28 cells.** I1: zero
precedence violations (no abstention row carries a tie item). I2:
eligible-set ∩ certified-abstention-set = ∅ exactly, per cell —
proven on real co-occurrence material: the three `pairD_stale-soft`
cells carry 300/280/296 incumbent abstentions alongside 903/943/763
dual-present defers, and the quiet-cell guard is CLOSED on all three,
so F1b emits zero `witness_alt` on the incumbent's home traffic. I3:
frozen surfaces git-clean before and after; packets read-only. I4:
every `witness_alt` record `harness-heuristic`; `core-certified`
appears only on incumbent abstain records.

**C(iii) envelope — FROZEN as `pr12_8/f1b_envelope.json`** (verdict
`stageC-envelope-frozen`) + 28 per-cell served-decision tables under
`pr12_8/served/`. Totals: **2,052 witness_alt rows** across the panel;
876 abstain rows co-scored. On all **18/18** cells with a committed
truth-joined table (12.6 test + 12.7 holdout) the reader's
`witness_alt` set matches the committed F1b ACT set **exactly**
(query_id and class), and expected-correct mass is copied verbatim
(e.g. pairB_oneshot 231/231 + 275/275; pairD_oneshot 116/128 +
155/156; pairE_oneshot 225/250 + 280/280; all contra and clean cells
0 acted). The 10 cells without committed truth (s0 dev, clean_pairA
s1/s2, stale-soft s1/s2) carry their exact label-free `witness_alt`
multisets (s0: pairB_oneshot 274, pairD_oneshot 111; all others 0)
with `expected_correct_mass=null` — no truth was joined by this stage.
Internal double pass identical; external re-run reproduces all 29
`pr12_8/` files byte-identically.

The candidate `s1-witness-alt-batch 0.1-candidate` thereby holds the
evidence for **rung 2** of its status ladder (composition-proven +
envelope-frozen). Stages A/B/D/E remain separately unauthorized; the
envelope scope note records that it covers the committed panel only
and confers no operative force. PR-10 merge-abstain remains the only
certified reader contract; posture remains deferral.

### 14.3 Stage A discharged — traffic-axis panel extension scored; envelope v0.2 (append-only; 2026-07-08)

Separately authorized (goal directive: extend the candidate panel
across the registered traffic-axis cells). Three artifacts, all under
the §7 Stage A registration:

**Pre-flight determinations of record.** mixed and plain-stale: 16
cells (pairs B–E × s1/s2), all 80 committed input files and all 16
`pr4_geometry_table` governance keys present. **jitter and g5twin:
`panel-insufficient` without generation** — their committed stems exist
only at s0 (dev exposure, excluded by the standing discipline) and the
pr4 geometry table has no governance arm for either, so they are not
generable under the registered provenance mechanism; carried as named
scope bounds per §7/§8, not silently dropped.

**Generation.** Additive `scan12_8_panel` manifest (+90/−0, every
pre-existing key byte-equal) + new driver
`harness/action_boundary_panel_generate.py` (the 12.7 erratum-E1
pattern as a new file, per §11). Anchor gate re-passed; 16 cells × 3
shapes emitted to `pr12_8_panel_cache/` (145 files, committed
`f3f5304`); 48/48 double-emits byte-identical; purity zero hits;
emitter sha = pin `2539686a…` throughout.

**Scoring** (`harness/action_boundary_panel_score.py`, 12.7 G-H
structure, sha-attested block `2f009cf2…`, cache pin `f3f5304`):
STAGE A STATUS `panel-extension-scored`, zero kills, internal double
pass identical, external re-run byte-identical (193 files). G-H1
exposure empty; G-H2 48/48; G-H4 per-type power: **mixed 155 W2 rows
POWERED; plain-stale 0 — `panel-insufficient`** (the type is
structurally tie-free: guard-open cells, zero dual-present rows, so
F1b has no action surface there — zero utility, zero wrong-action
risk). **The mixed finding:** every mixed cell is quiet-cell-guard
CLOSED (contradiction 209–237 vs ambiguous 54–63; margins +155…+183) —
mixed traffic presents as loud cells, and W2:F1b acts on **zero** of
the 155 in-scope mixed rows: `scored-pass` **safe-by-silence** (wrong
mass 0.0 on every unit, global 0.0), the same posture class as contra,
with the same honest caveat: no utility on this type. **Guard-regime
(G-R4):** `pairD_stale_s1` (margin −4) is the **first boundary-regime
cell observed** (|margin| ≤ 4) — recorded; it carries zero in-scope
rows, so no per-row behavior exists to report; the guard's three
observed clusters are now one-shot open (−4…−32), contra closed
(+176…+184), mixed closed (+155…+183).

**Envelope v0.2** (`pr12_8/f1b_envelope.json`, verdict
`stageC-envelope-frozen`, panel pin `6236382`): 44 W2 cells, 116,991
rows, composition I1–I4 **44/44**, act-set cross-check **34/34**
against committed truth-joined tables, `witness_alt` total unchanged
at 2,052 (the 16 new cells contribute zero), **all 28 v0.1 entries
reproduced identically** (in-run assertion against the committed v0.1
envelope; version-bump-never-rewrite honored); 122 pinned inputs; both
double passes identical; external re-run reproduces all 238 `pr12_8/`
files byte-identically.

**Consequence for the candidacy.** The served-surface envelope now
covers every traffic type the committed corpus can produce packets
for. W2:F1b's acting surface is confirmed to be exactly the one-shot
harm class: it acts nowhere else (contra 0, mixed 0, stale
structurally 0, clean 0). G-R3's ledger after Stage A: one-shot
scored-pass (12.6/12.7), mixed scored-pass (safe-by-silence), contra
scored-pass (12.6/12.7, thin panels), plain-stale/jitter/g5twin
`panel-insufficient` scope bounds. Stages B/D/E remain separately
unauthorized; no candidacy verdict is emitted here; nothing is served
to anyone. PR-10 merge-abstain remains the only certified reader
contract; posture remains deferral.

### 14.4 Stage B discharged — contra-power accounting (append-only; 2026-07-08)

Separately authorized (goal directive: quantify whether the contra
evidence is powered enough to support later adjudication). Analysis
over committed artifacts only — every number below is recomputed from
the committed `pr12_6/action_boundary_scan.json`,
`pr12_7/holdout_scan.json`, `pr12_8/panel_scan.json`, and
`pr12_8/f1b_envelope.json` (v0.2); no new packet, no new scoring run,
no truth join beyond what those artifacts already contain. **No
candidacy verdict, no GO, no certification, no served output.**

**Inventory (the safety-relevant surface = guard-closed cells).**
Envelope v0.2 contains **21 guard-closed W2 cells**: 8 contra (B/D
s0–s2, C/E s1/s2), 8 mixed (B–E s1/s2), 3 pairD stale-soft (s0–s2).
The 3 stale-soft cells carry **zero** in-scope tie rows (the stale
family is tie-free), leaving 18 closed cells with an action surface:
**319 in-scope W2 rows, on which F1b acted 0 times** (Stage A raised
the closed-cell row count from 135 to 319, +136%).

**Counterfactual guardless exposure (what the guard is measurably
preventing).** On those same 319 rows the guardless F1a row-signature
would have acted **73 times with wrong mass 70.0** (precision 0.041) —
e.g. pairD_contra_s2: 21 acts, 18 wrong; pairD_mixed_s1: 16 acts, all
wrong. The quiet-cell guard is therefore not decorative: it suppresses
an acting surface that is ~96% wrong where it exists, and it is the
sole protector (the row-local signature alone passes no contra/mixed
gate).

**Power quantification (model assumptions stated).**
* *Row granularity (iid-row model):* 0 acts / 319 rows → one-sided
  95% upper bound on the per-row act rate on closed cells **0.93%**
  (99%: 1.43%). Caveat: rows within a cell share the cell guard, so
  iid-rows overstates independence.
* *Cell granularity (iid-cell model — the honest thin number):* 0
  guard mis-opens / 21 closed cells → 95% upper bound on the per-cell
  mis-open rate **13.3%** (18 scored cells only: 15.3%). Cell
  granularity is the right unit for guard failure, and this bound is
  wide: the evidence cannot exclude a ~1-in-8 mis-open rate at 95%
  confidence under that model.
* *Margin separation (distribution-free, descriptive):* closed-cell
  margins are themselves bimodal — contra/mixed +155…+186, stale-soft
  +23…+28 — and open-cell margins are −4…−32. **The unobserved guard
  corridor is (−4, +23)**: no committed cell has ever landed in it.
  This refines the §14.3 G-R4 record (the +23 stale-soft margins are
  closer to the flip than any contra/mixed cell).
* *Detection sensitivity of the registered gates:* at the 0.05
  per-unit ceiling, a **single** wrong act breaches the gate in any
  unit with n < 20 — true of 13 of the 18 scored closed units; the
  five larger units (n = 28–61) detect at 2–4 wrong acts. The panels'
  thinness therefore limits estimation precision, not violation
  detection: any minimal leakage would have failed a gate.

**Determination (registered §7 Stage B vocabulary; not a verdict).**
The contra evidence is **sufficient at row granularity and gate
granularity, thin at cell granularity**. It supports later Stage E
adjudication **only together with the registered named scope bound**,
which this accounting fixes as: *"contra-side safety = zero action on
319 in-scope W2 rows across 18 guard-closed cells (21 including the
tie-free stale-soft cells), with a counterfactual guardless exposure
of 73 acts / 70.0 wrong mass suppressed; per-cell guard mis-open rate
bounded only at ≤13.3% (95%, iid-cell model); guard behavior inside
the (−4, +23) margin corridor unobserved."* Stage E must carry this
bound in any verdict text (§8 G-R3/G-R4; dropping it is kill §10.9).
Enlarging cell-level power is not reachable from the committed corpus
(every governed run-stem is now either in the panel or
panel-insufficient); it would require Stage D-strong new-seed engine
runs or new traffic protocols, both outside this memo's offline
authorizations. Stages D and E remain separately unauthorized; PR-10
merge-abstain remains the only certified reader contract; posture
remains deferral.

### 14.5 Stage D discharged — seed-axis branch chosen: D-bounded (append-only; 2026-07-08)

Separately authorized; the branch choice was made **explicitly by the
operator** (per §7 Stage D: one branch must be chosen explicitly,
neither is default). **Chosen: D-bounded.** Registration-only; nothing
executed.

**What D-bounded fixes.** No new seed is minted; no FAM-engine run
occurs. The candidacy's seed scope is the committed seeds'
distributional regime — **s0 (dev), s1, s2** on the committed panel
protocol (vision, 12 epochs, supersede-epoch 6) — and nothing beyond
it. Consequences, registered in advance and now operative for Stage E:

* Any Stage E GO **must** be named
  `contract-candidate-GO-seedbounded(W2:F1b)` — the unqualified
  `contract-candidate-GO(W2:F1b)` is **unreachable** under this branch
  (§8 G-R6; emitting it would be kill §10.9, scope laundering).
* The verdict text must carry the seed bound together with the
  standing scope bounds it joins: the §14.3 traffic bounds
  (plain-stale/jitter/g5twin `panel-insufficient`; mixed
  safe-by-silence) and the §14.4 contra-power bound (zero action on
  319 closed-cell rows; per-cell mis-open ≤13.3% at 95%, iid-cell
  model; guard corridor (−4, +23) unobserved).
* No seed-generalization claim of any kind is created: the candidate,
  even fully adjudicated, asserts nothing about s3+ seeds, other write
  protocols, other encoders, or drift — the same class of bound PR-10
  certification carries (stationary, one encoder), stated rather than
  implied.

**What D-bounded does not foreclose.** D-strong (new-seed engine runs:
gentoo compute, darwin byte-verify, its own run-matrix
pre-registration and execution approval) remains available as a
*future, separate* registration. If ever run and passed, it would
support a fresh adjudication under the unqualified verdict name;
nothing from this branch choice would need to be unwound — the
D-bounded verdict simply remains the honest record of what the s0–s2
corpus supported.

**Stage E readiness ledger after this section.** Registered inputs
complete: Stage A (§14.3), Stage B (§14.4), Stage C (§14.1/§14.2),
Stage D (this section). The single remaining prerequisite before
Stage E may be authorized is the **§9 monitoring/withdrawal
registration** (tripwires = the registered 12.6 constants, tightening
only). Stage E itself remains separately unauthorized. No candidacy
verdict, GO, certification, or served output is issued here; PR-10
merge-abstain remains the only certified reader contract; the
operational posture on witness-window rows remains deferral.
