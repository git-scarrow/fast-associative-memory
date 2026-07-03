# PR-12 — Harness-boundary design memo: FAM-core × constructive-forgetting harness

Date: 2026-07-03. Main @ `f5e7105`. **Design only.** This memo opens the
harness track recommended by `FAM_GOVERNANCE_CLOSURE.md` §7 (fork (a):
accept residuals, design the external constructive-forgetting harness
boundary). It adds no FAM-core policy, moves no threshold, changes no
schema, and does not reopen PR-11.1 (read-time expansion, `negative`) or
PR-9B (write-event authority, `second-key-failed`). No implementation is
performed in this memo; §7 ends with the single bounded implementation
prompt for the follow-on step.

**Scope of claims.** Everything cited as "certified" or "measured" below
is a committed result of the closed escalation sequence (PR-2 … PR-11.1,
PR-9B), scoped exactly as those memos scope it: the committed panel
protocol, one encoder, stationary evidence, gentoo-canonical run
artifacts. Everything in the harness column is *design*, carrying no
certification until its own gates run. The memo keeps these strictly
separated (§3, invariant I1).

**The one structural idea.** All three intra-core escalations died on the
same pre-registered gate: false abstention — the cost of *suppressing a
served answer*. PR-11.1's finding was that router evidence *localizes*
(names the right slots) but does not *discriminate* (acting on those slots
punishes correct traffic) at the only action the core reader has:
abstain. A harness above the core has a different action space —
annotate, caveat, dual-present, quarantine from prompt, escalate to a
human — whose failure cost is a caveat or an escalation, not a lost
answer. Evidence that failed the core's suppression gate can therefore be
*re-used, not re-litigated*, for non-suppressive dispositions. This does
not reopen PR-11.1: no read-time enforcement is added to the core, the
served stream stays byte-identical, and harness dispositions get their own
acceptance metrics (§6), not an inherited exemption from the core's
ceiling.

## 1. FAM-core responsibilities

**Certified to do (and nothing more).** Serve
`{answer | abstain(merge_suspect_led)}` under `--read-govern
merge-abstain` — the PR-10 contract (`readout-certified`, 92/92 cells,
gates G1–G5), at its exact recorded envelope: capture floors
C 1.000 / B 0.994667 / E 0.969466 / D 0.778667; worst false-abstain
0.327% of correct traffic; 5,118 abstentions all on soft arms, 0 on all
74 non-soft cells; 0 changed answers; abstention set equal to the frozen
scorer's merge-suspect set. This contract is immutable here as in every
memo since PR-10.

**Evidence FAM-core exports (all existing, all committed schemas).**

| artifact | fields the harness consumes | granularity |
|---|---|---|
| served per-probe CSV | `served_outcome`, `abstain_reason`, `top1_label`, `top1_slot`, `top1_sim`, `top1_top2_margin`, `vote_entropy`, `effective_support`, `probe_index`, `epoch` | per query |
| `*.topk.csv.gz` | `(probe_index, rank, slot, sim, surviving, weight, decode)` | per query × support member |
| `*.per_slot.csv` | `is_contra_fork`, `is_stale_superseded`, `is_current_fork`, `is_merge_candidate`, `role`, `last_write_seq`, `n_records`, `decode` | per slot × epoch (router state) |
| `*.fork_events.csv` | `epoch`, `event_class`, `record_seq`, `outcome`, `incumbent_slot`, `owner_slot`, `payload_cos_incumbent` | per write event |
| PR-9.2 event ledger | intrinsic key `(epoch, event_class, batch_index)` + `payload_label` (`identity-certified`) | per eligible write event |
| `summary.json` / `.governance.json` | protocol metadata; frozen-scorer per-arm hazard counters (`analyze_fork_governance.py`) | per run |

**What FAM-core must not claim to govern** (closure memo §5, verbatim
residuals): pairD/pairE stale-soft rows surviving abstention (151 / 14,
3-seed); frozen-probe broken rows on D/E (338 / 138); contradiction forks
(4,491 contra-wrong none-rows, B–E); one-shot ambiguity (3,834 wrong
none-rows; 32 permanently ambiguous pairs per cell); and the entire
freshness / tombstone / retention column, for which no certified behavior
exists. FAM-core additionally must not claim: deletion semantics, time or
version semantics, cross-session identity, or anything under drift,
re-embedding, or other encoders.

## 2. Harness responsibilities

The harness owns lifecycle semantics and prompt compilation. It assigns
every memory item it handles exactly one **exposure state**:

* **`agent-readable`** — compiled into the prompt-visible memory packet.
  Default only for items with no adverse flag from any §1 evidence source.
* **`hidden`** — harness-internal state (hazard priors, policy version,
  counters, per-arm statistics). Never prompt-visible, never presented as
  memory content; exists so that harness reasoning is itself inspectable
  in the audit stream.
* **`stale`** — support intersects a `is_merge_candidate` slot, or the
  item's slot is flagged `is_stale_superseded` while still decodable.
  Excluded from the prompt unless the caller sets an explicit
  `allow_stale` flag, in which case it compiles *with an inline
  staleness caveat* and the authorization is logged.
* **`superseded`** — a newer item with the same identity key exists
  (harness ledger, keyed by the PR-9.2 intrinsic key or by
  harness-stamped ingest identity). Same exposure rule as `stale`; the
  superseding item is what compiles by default.
* **`quarantined`** — a party to an unresolved contradiction fork
  (`is_contra_fork` / `is_current_fork` pair with no adjudicated winner).
  Never compiles. Both parties are held, neither is blended, and the
  pair is emitted on the human-review queue.
* **`audit-only`** — items retained solely as evidence (diverted-event
  records, tombstoned items inside their retention window, withheld
  packet members). Never compiles; always queryable in the audit stream.
* **`human-review`** — items whose disposition the harness is not
  permitted to decide alone: unresolved contradiction pairs, permanently
  ambiguous one-shot pairs, and any item where two evidence sources
  disagree. Compiles only as an explicit "unresolved — N candidates"
  notice (dual-presentation), never as a single asserted answer.

**Prompt-compilation rules.**

1. Only `agent-readable` items, staleness-caveated `stale`/`superseded`
   items under a logged `allow_stale`, and `human-review`
   dual-presentation notices may appear in the prompt.
2. A FAM-core abstention compiles as an explicit line — "memory abstained
   (merge-suspect)" — never as silence. The agent must be able to
   distinguish *no memory* from *withheld memory* (invariant I6).
3. Caveats attach inline to the item they qualify; the compiler never
   merges two items' content into one statement (no silent blending).
4. Certification tier is part of the compiled text's provenance: only the
   PR-10 abstention may be described to the agent as certified behavior;
   every harness flag is labeled as harness-level.
5. Compilation is deterministic given (memory packet, policy version):
   same inputs, byte-identical prompt block.

## 3. Boundary contract

**Inputs (FAM-core → harness), read-only, per query.** Exactly the §1
export table: the served row (`served_outcome`, `abstain_reason`, top-1
fields), that probe's `topk` support rows, the current-epoch `per_slot`
router state, and — at session/run granularity — the `fork_events`
stream, the PR-9.2 intrinsic-key ledger, and `summary.json` /
`.governance.json`. The harness opens these artifacts; it never calls
into engine internals, never sets a `--govern`/`--read-govern` flag, and
never writes to any FAM-core artifact.

**Outputs (harness → agent runtime), per query.**

* **Memory packet** (prompt-visible): ordered list of compiled items,
  each `{content, state, caveat?, provenance, certification_tier}`, plus
  the abstention notice when the core abstained.
* **Audit packet** (never prompt-visible): the full decision record —
  every candidate the harness saw, its assigned state, disposition, and
  the audit fields below.

**Invariants.**

* **I1 — no certification laundering.** The only behavior the harness may
  label "certified" is the PR-10 merge-abstain contract at its recorded
  envelope. Every harness-assigned state is labeled harness-level in both
  packets. The harness never presents its own flags as FAM-core
  guarantees, and never extrapolates the envelope beyond its scope (one
  encoder, stationary, committed protocol).
* **I2 — no abstention override.** `served_outcome = abstain` is final:
  the harness must not reconstruct or substitute an answer for an
  abstained query from support rows or its own ledger.
* **I3 — core frozen.** The harness consumes committed artifact schemas
  only; FAM-core engine, driver, scorer, and thresholds are untouched.
  (Mechanically: the harness has no write path into the repo's engine or
  benchmark files.)
* **I4 — retrieval-faithful accounting.** "Not retrieved" (absent from
  `topk`) and "retrieved but withheld" (present, disposition ≠ shown) are
  distinct audit states; the audit packet records `k` and the support set
  so absence is itself evidenced.
* **I5 — stale/superseded exclusion by default.** No `stale` or
  `superseded` item compiles without an explicit, logged `allow_stale`
  authorization, and never without its caveat.
* **I6 — contradictions never blend.** Unresolved fork parties are
  quarantined or dual-presented for adjudication; the compiler has no
  code path that merges them.
* **I7 — total auditability.** Every exposure decision — including
  "shown, no adverse flag" — emits the audit fields below.

**Required audit fields, every harness decision.**

`query_id` · `item_id` (slot + record identity; PR-9.2 intrinsic key
where applicable) · `state` (§2 taxonomy) · `disposition`
(`shown | shown_with_caveat | withheld | escalated | not_retrieved`) ·
`reason_code` (closed vocabulary, e.g. `merge_support_member`,
`contra_fork_unresolved`, `oneshot_permanent_tie`, `superseded_by`,
`no_adverse_flag`) · `evidence_ptr` (artifact, field, value that
triggered the state — e.g. `per_slot.csv: is_merge_candidate=1 @ slot 17,
epoch 9`) · `certification_tier` (`core-certified | harness-heuristic`) ·
`policy_version` · `authorization` (present iff an override flag was
used).

## 4. Residual mapping

For each uncovered class in closure memo §5: can the harness govern it,
by what mechanism, on what evidence. "Govern" here means *non-suppressive
disposition* (§0's structural idea), never a change to the served stream.

**(a) pairD/pairE stale-soft residuals (151 / 14 rows, 3-seed) —
governable, existing evidence.** PR-11.1 §1 confirmed the mechanism
exactly: the merged slot sits in the surviving support of **every**
residual row (P2 captured 151/151 aggregate, 83/83 at pairD/s0); it is
merely outvoted. In-core, acting on support membership failed the
suppression gate on compressed pairs (false-abstain 0.095–0.142 on D,
0.068–0.101 on E — 231–363 suppressed correct answers per pairD run).
Harness disposition: any served answer whose surviving support intersects
an `is_merge_candidate` slot is compiled as `stale`-suspect —
`shown_with_caveat` by default. The identical evidence
(`topk.surviving=1` rows ∩ `per_slot.is_merge_candidate=1`) that cost
231–363 *lost answers* in-core costs 231–363 *caveats* here, while
covering all 165 residual rows. No new observables.

**(b) Frozen-probe broken rows (D 338 / E 138) — not governable on
current evidence; accepted.** This is store corruption: PR-9B showed
every currently logged write observable is hazard-blind or
state-contaminated at the event level, and total diversion (PR-7 refuse)
removed only ~33% of broken rows while consuming all 576 capture events.
A read-side harness has no row-level signal that distinguishes a broken
answer from a correct one. What it *can* do honestly is coarse exposure
bounding: attach arm/run-level hazard tiers from the frozen scorer's
`governance.json` to whole memory regions — a prior, labeled as such,
never a row claim. Anything better requires event-local write-time signal
that PR-9B demonstrated the current logs do not contain — i.e. this class,
and only this class, is where the boundary design points back at the
closure memo's fork (b) new-observables pre-registration. Per this memo's
constraint, no new observable is designed here; the requirement is merely
*named*: a per-event geometric observable with demonstrably different
distributions across D/E vs A/C, where `payload_cos_incumbent` was
identical to six decimals.

**(c) Contradiction forks (4,491 contra-wrong none-rows) — governable,
existing evidence.** The router adjudicates ~208/235 fork pairs, with
resolution lag ≤ 1 epoch wherever it resolves (median 1.0, max 1); the
core could not act on this because adjudicated-abstain cost up to 58.2%
false abstention. Harness disposition: fork-party slots
(`is_contra_fork` / `is_current_fork`) map to states, not suppression —
the adjudicated winner compiles `agent-readable` tagged
`contra-adjudicated (harness-level)`; the loser becomes `superseded`;
unresolved pairs (343 contra pairs never resolve) are `quarantined` and
emitted on the human-review queue as dual-presentation candidates. Never
blended (I6). Evidence: `per_slot` flags per epoch + fork-pair records;
no new observables.

**(d) One-shot ambiguity (3,834 wrong none-rows; 32 permanently ambiguous
pairs per cell) — partially governable, existing evidence.** PR-11.1 §3:
79% of one-shot harm occurs while the leading slot's pair is unresolved,
and those pairs *never* resolve — the pending state localizes exactly this
class (nonredundancy 0.807) but acting on it in-core cost 33–36% false
abstention. Harness disposition: items led by a permanently-pending pair
compile as `human-review` dual-presentation — "two candidates, unresolved
tie" — rather than one asserted answer. The 21% of one-shot harm not
localized by pending state remains uncovered and is recorded as accepted.
Evidence: router pending/resolution state reconstructed from `per_slot`
across epochs; no new observables.

**(e) Freshness / tombstone / retention — harness-native; needs no
FAM-core evidence at all.** The closure memo records this column as
outside every certified artifact's scope, and that is the point: time,
version, and deletion semantics belong to the layer that observes wall
clock and source lineage. The harness stamps its own ingest-time metadata
(timestamp, source id, version, TTL) in a harness-side ledger, joined to
FAM-core records by the PR-9.2 `identity-certified` intrinsic key where
events are eligible, or by harness-assigned ids otherwise. Tombstoning is
a harness-ledger state transition (`audit-only` within the retention
window, then eligible for true deletion *from the harness ledger*);
FAM-core is never mutated, so a tombstoned item that FAM-core still
retrieves is simply withheld with `reason_code=tombstoned` — which is
precisely the "retrieved but withheld" state I4 exists to make honest.

Observed vs inferred, and the load-bearing assumption: mechanisms (a),
(c), (d) rest on *committed* localization measurements (P2 151/151;
adjudication 208/235; pending nonredundancy 0.807) — the inference is
only that caveat/dual-presentation costs are acceptable at the agent
layer, which §6 makes a measurable gate rather than an assumption. The
decision-relevant unknown the prototype must answer: the **caveat
rate** — what fraction of correct traffic gets flagged (the in-core
numbers bound it: e.g. ~9–14% on pairD soft arms for mechanism (a)) — and
whether an agent's downstream behavior degrades under it. That is a
harness-track measurement, not a FAM-core one.

## 5. Minimal harness prototype

Smallest non-invasive step: an **offline replay adapter** — analysis-only,
darwin, no torch, exactly the discipline of the sequence's registered
readers.

* **Location:** new top-level `harness/` directory —
  `harness/harness_boundary_sim.py` + `harness/harness_policy.json`
  (the closed reason-code vocabulary, state-assignment rules, and
  policy version). Nothing under `benchmarks/` or the engine is touched.
* **Input:** one committed PR-10 governed cell replayed from its five
  artifacts (`per_probe_*.csv`, `.topk.csv.gz`, `.per_slot.csv`,
  `.fork_events.csv`, `summary.json`) plus its `.governance.json`.
  Primary cell `pairD/soft/s0` (richest residual: 300 certified
  abstentions + 83 residual stale-wrong rows); control cell
  `clean/pairA/s0` (must produce zero adverse states).
* **Output, per probe row:** `memory_packet.jsonl` (prompt-visible
  compilation, §2 rules) and `audit_packet.jsonl` (§3 audit fields for
  every candidate, including `not_retrieved` accounting), plus a single
  `decision_table.csv` — one row per (probe, support member) with state,
  disposition, reason code, and evidence pointer — so every stale /
  quarantine / supersession decision is directly inspectable with no
  tooling.
* **Mechanisms implemented:** (a) merge-support stale-suspect caveats,
  (c) fork-party quarantine/supersession with a human-review queue file,
  (d) permanent-pending dual-presentation. Mechanism (b) appears only as
  the labeled run-level hazard tier; (e) as ledger schema stubs
  (no live ingest exists to stamp).
* **Verification:** a hermetic test asserting the §6 criteria on the
  emitted packets (invariants I1–I7 checked structurally: no abstain
  override, no uncaveated stale item in any memory packet, every audit
  row complete, disposition vocabulary closed), plus the two headline
  counts against committed numbers: mechanism (a) flags exactly the
  P2-captured row set at pairD/soft/s0 (375 = 292 certified-abstained +
  83 residual), and the clean control emits zero adverse states.

## 6. Acceptance criteria

1. **No FAM-core behavior change:** no engine, driver, scorer, threshold,
   or committed-artifact byte differs; the prototype is read-only over
   committed files (I3, mechanically checkable via `git status`).
2. **PR-10 untouched:** abstained rows pass through as explicit
   abstention notices; zero packets convert an abstain to an answer (I2).
3. **Total audit coverage:** every item in every packet — hidden or
   exposed — carries a complete §3 audit record; audit-row count equals
   candidate count plus not-retrieved records (I7, I4).
4. **Stale/superseded exclusion:** zero `stale`/`superseded` items in any
   memory packet without both an `allow_stale` authorization field and an
   inline caveat (I5).
5. **No silent blending:** every unresolved contradiction pair is
   `quarantined` or `human-review`-surfaced; no compiled item derives
   from more than one fork party (I6).
6. **Withheld ≠ absent:** `not_retrieved` and `withheld` are distinct
   dispositions in the decision table, each evidenced (I4).
7. **No certification overstatement:** grep-clean guarantee — the string
   "certified" appears in packets only on the PR-10 abstention notice;
   all other flags carry `harness-heuristic` (I1).

## 7. Next implementation prompt

> Implement the PR-12 minimal harness prototype exactly as specified in
> `results/issue_failure_mode_blindness/PR12_HARNESS_BOUNDARY_DESIGN.md`
> §5. Create `harness/harness_boundary_sim.py` and
> `harness/harness_policy.json` only — no FAM-core file may change.
> Replay the committed cells
> `pr10/governed/per_probe_stale-soft_pairD_s0.*` (primary) and
> `pr10/governed/per_probe_clean_pairA_s0.*` (control), consuming only
> the five run artifacts plus `.governance.json`. Emit
> `memory_packet.jsonl`, `audit_packet.jsonl`, and `decision_table.csv`
> under `results/issue_failure_mode_blindness/pr12/`, implementing
> mechanisms (a), (c), (d) of §4 with the §2 state taxonomy and the §3
> audit fields. Add a hermetic test
> (`tests/test_pr12_harness_boundary.py`) asserting the seven §6
> acceptance criteria, including the two committed anchors: mechanism (a)
> flags exactly 375 rows at pairD/soft/s0 (the 292 certified abstentions
> plus the 83 residual stale-wrong rows) and the clean control emits zero
> adverse states. Analysis-only, darwin, no torch, no new observables, no
> threshold motion, no change to any committed baseline. This is a
> simulator, not a production system: stop after the two cells and the
> test; do not add live ingest, deletion, or agent integration.

## 8. Prototype findings / errata (post-merge appendix)

*Appended 2026-07-03 after the §5 prototype merged (design memo
`ca79d4c`, prototype `ec62381` + `3ca3d07`, merges `66147f9`/`b8cab28`).
§§1–7 above are the pre-registration snapshot and are unmodified. This
appendix records what the prototype run falsified, corrected, or
measured. No FAM-core change, no new observable, no policy-semantics
change, and no promotion claim is made here.*

**Failed / corrected design assumptions (three).**

* **Missing `.governance.json` sibling.** §5 assumed each pr10 governed
  run has a `.governance.json` sibling; none exists under
  `pr10/governed/`. The committed hazard evidence for the same
  (pair, arm, seed) lives in the pr6 panel (`pr6/stale_de/`) and the
  pr3c baseline family; the prototype consumes those, justifies the
  cross-run join by PR-10 G1 write-stream byte-identity, and
  cross-checks it numerically (192 merge-suspect events / 28 conflict
  pairs / 375 stale-wrong all match). Recorded in
  `harness/harness_policy.json`.
* **`core_certified_abstention` naming, caught by the memo's own grep
  invariant.** The prototype's first run failed §6 criterion 7: the
  harness-chosen reason-code name itself contained the string
  "certified", leaking into 300 audit records. Resolved by renaming the
  vocabulary term to `core_abstention_passthrough`, keeping the grep
  invariant maximally strict rather than widening its exemptions. This
  is the certification-laundering guard working as designed.
* **`per_slot.is_merge_candidate` oracle-hygiene substitution.** §§3–4
  cite the per_slot flag as mechanism (a) evidence, but the frozen
  scorer never loads the per_slot `is_*` diagnostic flags (its
  policy-visible allowlist excludes them). The prototype derives every
  state from the label-free write-time router instead and reports
  per_slot-flag agreement as a diagnostic: **1536/1536 (epoch, slot)
  cells agree on both cells**, so the memo's evidence-pointer wording
  holds empirically but is downgraded to corroboration, not basis.

**Committed prototype results (both cells re-verifiable byte-exactly via
`python3 harness/harness_boundary_sim.py --check`).**

* **clean/pairA/s0 control: zero adverse states** — 2,532 probes, all
  `agent-readable`/`shown`, zero abstentions, zero router state.
* **pairD/stale-soft/s0: exactly 375 adverse stale-wrong rows covered**
  — 292 certified-abstained (pass-through, `core-certified`) + 83
  residual merge-support-flagged, zero escapes: the PR-11.1 P2
  mechanism reproduced from committed artifacts alone. G3-exactness
  self-check: the certified abstain set equals the merge-led set
  row-for-row.
* **Caveat economics: 1.35% of correct traffic** (33 caveats on correct
  answers / 2,449 correct-traffic denominator `n − wrong_none`; 46
  caveats total). Mechanism (a)'s cost is trivial at this cell.
* **Suppressive-disposition economics: 30.7% of correct traffic** (743
  escalated + 8 withheld-superseded on correct / 2,449). Mechanism (c)
  as prototyped replaces the answer with an unresolved-notice —
  suppression in effect — and PR-11.1's granularity problem reappears
  at the harness as escalation volume.

**Consequence, stated explicitly: mechanism (c) is not promotion-ready
under the proposed 5% suppression ceiling** (the program-precedent gate,
suppressive dispositions on correct traffic ≤ 0.05 per cell). At 30.7%
it fails by 6×. Any re-shaping (e.g. contradiction-caveat instead of
escalation) is a disposition change within the existing policy
semantics and requires its own pre-registered gate before any promotion
claim; nothing in this appendix promotes anything.
