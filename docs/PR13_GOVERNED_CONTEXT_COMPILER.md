# PR-13 — Governed context compiler (pre-registration, revision 3)

Date: 2026-07-09. Main @ `142fc49`. **Status: r3 — registration text final;
build checkpoint authorized 2026-07-09.** Revision 3 adds the
machine-readable consumer output contract (§8.4), narrows the
arm-equivalence claim to its registered five components (§8.2), and splits
the family-level consumer pin (fixed here) from the artifact-level pin
(fixed at the build checkpoint) (§6/§8.1). Revision 2 applied the
2026-07-09 red-team review: all seven blocking edits and recommended edits
8–12.
Raw-matched is adopted as the **primary comparator** (§8.2); raw-native is
retained as an **exploratory reported metric only**, read by no gate. This
memo registers the first milestone of the governed-context track motivated
by `docs/WHAT_FAM_HAS_BECOME.md` §10: a source-agnostic context compiler
with a bounded budget, evaluated by a sealed replay experiment against a
live-model consumer treated as **experimental subject, never as a certified
reader**.

**Standing record, restated and unmodified by any outcome here:**

* PR-10 merge-abstain remains the only core-certified reader contract; its
  text, envelope, and certification are untouched.
* s1-witness-alt-batch@1.0 remains a certified opt-in batch contract at its
  registered bounds; this memo neither enrolls a consumer in it nor modifies
  it. Here, witness-alt is consumed as **one adapter-level evidence signal**
  (§4.1), not as the center of anything.
* FAM-core is untouched at every layer (engine, driver, scorer, thresholds).
* No deployment, no live serving, no promotion, no memory ingestion into any
  agent, no autonomous downstream use. The consumer model runs only inside
  the sealed replay of §8.
* The PR-12.9 permanent prohibitions are not relaxed. This memo does not
  certify an LLM/agent reader and its verdict vocabulary cannot (§10). It
  registers a new experiment in which model behavior is a *measured
  outcome*, which no standing prohibition forbids.

**Explicit non-goals (structural, not just scoping):**

* **No certification ladder.** No rungs, no candidacy, no re-issue stages.
  One registration, one build checkpoint, one scoring run, one verdict.
  Rationale: the rung apparatus exists to graduate *acting policies* toward
  standing authority; nothing here acquires standing authority. A GO verdict
  is evidence, not a contract.
* **No new acting policy over FAM rows.** The compiler re-renders committed
  dispositions; it does not invent new act/defer decisions on witness rows.
* **No prose-only schema.** Every normative structure in this memo ships as
  a machine-readable JSON Schema committed alongside the code (lesson of
  PR-12.9 findings F1/F3: prose referents drift; typed schemas cannot).
  That now includes the *decision function*, not only the vocabularies: the
  state×evidence→disposition mapping is a committed policy table (§3).
* **No sha-attested copy-paste, no mirrored logic.** One shared library;
  frozen upstream logic is imported read-only, never re-implemented
  (supersedes the PR-12 mirror-not-import reading of invariant I3; the
  invariant's content — no mutation of frozen artifacts — is preserved by
  filesystem read-onlyness, not by re-typing code). The canonical import
  source is registered in §4.1: `harness/action_boundary_score.py` at its
  pinned sha, and **no existing sha-attested mirror elsewhere in the repo is
  refactored, edited, or deleted in this PR** — the four historical mirrors
  are untouched record; their cleanup, if ever, is its own registration.

---

## 1. Core question and hypotheses

**Core question:** does deterministic, evidence-carrying, budget-bounded
compilation of context measurably improve a live consumer's decisions over
the same underlying material served raw — and does the consumer actually
*use* the epistemic structure (caveats, dual-presentation, withdrawal
notices) the compiler emits?

**H1 (structure utility):** on harm-class rows, the governed arm strictly
reduces wrong-action mass versus **raw-matched** — the identical item
multiset at the identical budget with all governance structure stripped
(§8.2) — without exceeding the registered suppression ceiling on clean
rows. H1 is deliberately a claim about *presentation structure at fixed
selection*: on rows where the compiler's protection is withholding an item,
both arms lack it and H1 gains nothing there; the joint selection+structure
effect is reported exploratorily against raw-native and is **not claimed**
by this registration.

**H2 (compliance):** the consumer's behavior is measurably sensitive to
dispositions — caveats shift adoption, dual-presentation is answered from
the presented set, withdrawal notices are honored — each against a
chance-baseline control.

**H0 (the honest default):** compilation is decoration — the consumer's
behavior is statistically indistinguishable across governed and raw-matched
arms given identical item multisets and token budgets, at **every**
registered budget point (§6). If H0 survives, the governed-context thesis,
not just this compiler, is in trouble; that is recorded as `compiler-moot`.

---

## 2. Context-item schema (source-agnostic; normative artifact is the JSON Schema)

Committed as `harness/ctx/schema/context_item.schema.json` (draft 2020-12).
Prose below is descriptive only; on any divergence the schema wins.

Required fields per item:

| field | type | semantics |
|---|---|---|
| `item_id` | string | globally unique; `<source_id>:<native_id>` |
| `source_id` | string | registered adapter identity (§4) |
| `content` | string | the candidate text; never synthesized by the compiler |
| `content_kind` | enum | `source-native` (text carried byte-derived from the source) or `adapter-rendered` (text produced from source metadata by a **registered deterministic template**, whose template id is bound into `policy_version`; free-form generation is structurally impossible — templates are enumerated, versioned artifacts). Makes metadata-only sources (§4.2) representable without violating the anti-synthesis rule |
| `content_sha256` | string | binds content to audit trail |
| `event_time` / `ingest_time` | RFC3339 or null | source-asserted vs adapter-stamped; null is legal and *rendered* as unknown, never defaulted |
| `evidence[]` | array | ≥1 records: `{adapter_id, signal, value, evidence_ptr, tier}`; `tier ∈ {core-certified, harness-heuristic, source-asserted}` |
| `relations` | object | `supersedes[]`, `contradicts[]`, `candidate_set_id?` — item-id references, source-agnostic |
| `state` | enum | the PR-12 exposure taxonomy, unchanged: `agent-readable, hidden, stale, superseded, quarantined, audit-only, human-review` |
| `policy_version` | string | of the assignment policy that produced `state`, of the disposition policy table (§3), and of any rendering templates used |

Removed relative to r1: **`budget_cost` is not an item field.** Token cost
is a compiler-computed quantity under the §6 pinned tokenizer, not source
provenance; it lives in the audit row (§5). This keeps the item schema — and
`content_sha256`'s meaning — independent of any consumer.

Design rule carried over verbatim from PR-12: certification tier is part of
provenance; the only tier the compiler may render as `core-certified` is
PR-10 abstention passthrough (the grep guard is retained as a schema-level
enum plus a test, not a grep).

## 3. Dispositions (closed vocabulary, fixed precedence, committed decision function)

`disposition ∈ {assert, caveat, dual_present, withhold, summarize, defer,
withdraw}` — committed as an enum in the schema. Semantics:

* **assert** — rendered plainly; permitted only when no adverse evidence
  record exists on the item.
* **caveat** — rendered with an inline qualifier naming the evidence signal
  (e.g. "possibly stale: merged support"); the qualifier template is part of
  the frozen policy, one template per reason code.
* **dual_present** — the item's `candidate_set` is rendered as N unasserted
  candidates ("unresolved — neither asserted"); the compiler never orders
  candidates by anything but a registered deterministic key.
* **withhold** — not rendered; audit row records reason + evidence_ptr.
  Distinct from *not retrieved*, which is also audited (PR-12 I4 unchanged).
* **summarize** — rendered lossily under budget pressure, marked
  "(condensed)" with an audit pointer to the full item. Summaries are
  produced by a deterministic extractive rule (registered: first
  `k`-sentence prefix under the token cap), **never by a model** — a model
  summarizer would put unattested content into the governed arm.
* **defer** — an explicit unresolved notice without candidates ("1 item
  unresolved; escalation available"); the compiled line preserves the
  committed escalation posture for rows that carry it.
* **withdraw** — a retroactive notice revoking an item compiled at an
  earlier turn of the same replayed session ("the note served at turn j is
  withdrawn; do not rely on it"), emitted when adverse evidence arrives
  between turns. In this milestone, withdrawal is exercised only inside the
  scripted multi-turn cells of §8.3; the record only grows (no prior packet
  is edited).

Precedence on conflicting assignments (highest wins): `withdraw ≻ withhold ≻
defer ≻ dual_present ≻ caveat ≻ summarize ≻ assert`. PR-10 abstention
passthrough renders before all item dispositions, unchanged.

**The disposition function is a committed artifact, not prose.** The mapping
(state, evidence records) → disposition ships as a third schema,
`harness/ctx/schema/disposition_policy.schema.json`, plus one versioned
instance, `harness/ctx/policy/disposition_policy_v1.json`: an ordered rule
table whose rows are `{match: {state?, signal?, tier?}, disposition,
reason_code}`. Each evidence record is mapped by its first matching rule;
conflicting outputs across records resolve by the precedence above; an item
matched by **no** rule fail-closes to `defer` with an anomaly audit record
(the PR-12.9 F1/F3 lesson applied to the verb, not just the nouns: two
implementers must not be able to resolve state×evidence→disposition
differently). The instance is frozen at the build checkpoint; motion after
first scoring is a kill (§10).

## 4. Evidence adapters (two registered; the adapter contract is the point)

Adapter contract (committed as `harness/ctx/schema/adapter_output.schema.json`):
an adapter maps source-native artifacts → context items + evidence records.
Adapters are label-blind (no truth column may be read on any code path — the
PR-12.6 label-leak rule generalized), deterministic, and read-only over
their sources. Adding an adapter never changes compiler code — only policy
registration.

**4.1 Adapter A — FAM artifacts (`fam-v1`).** Consumes the committed run
artifacts (per-probe CSV, topk, per_slot, fork_events, ledger) exactly as
the PR-12 harness did, emitting signals: `merge_suspect` (PR-10 abstention
passthrough, tier core-certified), `stale_support`, `contra_fork`,
`superseded_by`, `oneshot_tie`, and `witness_alt_candidate_set` — the last
computed by importing the frozen policy block read-only and emitted at tier
harness-heuristic **as one signal among six, feeding `dual_present` like any
other candidate-set evidence.** The frozen policy block is imported from
exactly one canonical source: `harness/action_boundary_score.py` at its
pinned sha (verified import-safe: stdlib-only imports, module-level policy
definitions, `__main__`-guarded execution). No rung machinery, no envelope
reproduction; the s1 contract itself is neither invoked nor claimed.

**4.2 Adapter B — Shutter-Deck ingest ledger (`shutterdeck-v1`).** The
non-FAM source, chosen because it is in-repo, idle, and organic (no
injection driver): the ingestion service's SQLite metadata + file store.
Shutter-Deck is an *image* pipeline — the ledger carries no source text —
so every item it emits carries `content_kind = adapter-rendered`, produced
by the registered deterministic template `ctx-sd-item-v1` (rendering
`filepath`, `ingest_time`, `assigned_slot` into one fixed sentence form),
bound into `policy_version` like every other template. Signals: `freshness`
(ingest_time vs replay clock, TTL from registered policy),
`superseded_by_path` (same source path re-ingested later),
`duplicate_content` (identical file content across entries), and
`source_identity`. The ledger stores **no content hash**, so
`duplicate_content` is computed by the adapter as **SHA-256 over file bytes**
read from the file store at adapter run time; when the underlying file is
absent, the signal is emitted as `unavailable` with an anomaly audit record
— never as false. This adapter exercises exactly the column the closure memo
recorded as "outside the scope of every certified artifact" —
freshness/retention — and gives the schema its first test against evidence
FAM never produced. If the live Shutter-Deck corpus is too thin at build
time, the fallback (registered now, not chosen later) is a re-ingest of
`test_data/` with scripted re-ingestion events; the fallback flips a
`corpus=synthetic` flag that is carried into the verdict string. **Registered
now:** under `corpus=synthetic`, gate G-U3 downgrades from verdict-bearing
to a reported metric (§9) — scripted supersession events are an injection
driver wearing a different hat, and the record must not present them as an
organic pass.

## 5. The compiler (deterministic; the certifiable object)

`harness/ctx/compile.py`, pure function:

```
compile(items: [ContextItem], policy: PolicyVersion, budget: int, turn_state)
  -> (context_block: str, audit_packet: [AuditRow])
```

* Byte-determinism: same inputs → byte-identical `context_block` and audit
  packet; enforced by a double-run gate (§9 G-C1). No wall clock, no RNG,
  no model calls anywhere in the compiler.
* The block opens with a status header (I6 generalized): counts of shown /
  caveated / withheld / unresolved items and any abstention or withdrawal
  notices — the consumer always sees *that* governance happened, never
  silently curated context.
* Every audit row carries the PR-12 field set (query_id, item_id, state,
  disposition, reason_code, evidence_ptr, tier, policy_version,
  authorization) plus `budget_cost` (this item's rendered token cost under
  the §6 pinned tokenizer — moved here from the item schema) and
  `budget_decision` (§6). Audit-row count = candidate count + not-retrieved
  records, exactly as I4 requires.
* All I1–I7 invariants are re-asserted structurally by hermetic tests over
  emitted packets, with I3 in its corrected form (§0 non-goals).

## 6. Bounded context budget (tokenizer pinned here; budget grid, not point)

* **Tokenizer pin, split like the consumer pin (§8.1):** family-level is
  fixed in this memo — the tokenizer of the Qwen3 dense 8B family;
  artifact-level is fixed at the build checkpoint, where the exact
  tokenizer files (`tokenizer.json` + config) are sha-pinned inside
  `harness/ctx/policy/consumer_pin.json`. From the build checkpoint on,
  `budget_cost`, the byte-meaning of every B, and the G-C3 hermetic tests
  are fixed — before any §7 material is compiled for evaluation (§8.1
  selection-timing kill). The split resolves the r1 circularity in which B
  was denominated in tokens of an unpinned tokenizer.
* **Registered budget grid: `B ∈ {800, 1500}` tokens** (the minimal
  two-point grid; a single point cannot distinguish "structure doesn't
  help" from "the budget regime was mischosen"). 1,500 is the primary
  operating point; 800 is chosen to guarantee budget pressure on the
  largest cell, so the `summarize` and `withhold:budget_exceeded` branches
  of §3 are exercised somewhere in the record. All verdict-bearing gates
  are evaluated per B; `compiler-moot` requires its condition to hold at
  **every** registered B (§10). Changing or extending the grid after the
  first scoring run is a kill.
* **Budget-fill statistic (registered):** per cell × arm × B, the
  percentage of candidates budget-withheld and the percentage summarized is
  computed and reported, so "the budget never bound" is detectable in the
  record rather than inferable after the fact.
* Deterministic allocation: items are ranked by the registered priority key
  `(tier rank, adverse-evidence-free first, recency, item_id)` — no learned
  scoring. Overflow handling in registered order: `summarize` the lowest-
  priority summarizable items, then `withhold` with reason
  `budget_exceeded`, each with a full audit row. **No silent truncation of
  any kind**; a compiled block that hits B must still contain the status
  header, and the header must state how many items were budget-withheld.
* Budget integrity gate: for every row, tokens(context_block) ≤ that row's
  B, and every non-compiled candidate has an audit row (G-C3).

## 7. Corpus and cells

* **FAM cells (committed, replayed):** the pr12_3/pr12_4 one-shot and
  contra packet-tree cells plus `clean/pairA/s0` control — re-derived from
  run artifacts through adapter A (not from the old packets, which remain
  untouched historical record).
* **Organic cell:** one Shutter-Deck corpus cell through adapter B, with
  scripted supersession/freshness events (re-ingested files) providing
  ground truth by construction. **Task registered now (was r1's missing
  query surface):** query family `Q-SD1`, one template — "According to the
  ingest ledger, which ingest event provides the current version of the
  item at path `<P>`, and when was it ingested?" — instantiated once per
  path with ≥1 supersession event and once per control path with none.
  Answer key by construction: the latest ingest event for `<P>` at the
  replay clock. A **stale adoption** (G-U3's unit of measure) is a parsed
  answer that asserts a superseded ingest event (its timestamp or slot) as
  current.
* **Multi-turn withdrawal cells:** scripted 3-turn sessions in which
  adverse evidence for a turn-1 item arrives before turn 3; ground truth is
  whether the consumer's turn-3 answer still relies on the withdrawn item.

## 8. The replay experiment (sealed; consumer = experimental subject)

**8.1 Consumer.** One pinned open-weights model, pinned at two levels:

* **Family-level pin (fixed in this memo; motion = kill):** **Qwen3 dense
  8B, instruction-capable, run in non-thinking mode.** The mode is part of
  the pin, fixed in the committed prompt-template artifacts (thinking
  disabled via the chat template's `enable_thinking=False` path); flipping
  it after the first scoring run is prompt motion and therefore a kill.
  Non-thinking mode also removes the reasoning-trace surface from the
  G-C1 temperature-0 determinism check — the parsed answer is the whole
  output.
* **Artifact-level pin (fixed at the build checkpoint; committed as
  `harness/ctx/policy/consumer_pin.json`):** exact repository ID, revision,
  weights, tokenizer, chat template, runtime, precision, and their hashes.
  The scoring-run manifest re-attests these pins and introduces no new
  degree of freedom.

Temperature 0, fixed decoding parameters, one pinned prompt template per
task shape, output bounded by the §8.4 contract. Runs offline on
repo-controlled hardware. No fine-tuning, no system-prompt iteration
after the first scoring run (prompt motion after first scoring = kill, the
G-R7 no-tuning rule applied to prompts). **Selection-timing kill (registered
now):** every consumer degree of freedom above must be committed to the
repo **before the first compiled block over any §7 cell is rendered for
evaluation purposes** — structurally guaranteed by fixing the artifact
pins at the build checkpoint, which precedes all scoring. Hermetic-test
renders over synthetic fixtures do not count; any render over §7 material
does. Rendering real compiled output first and choosing the consumer after
is a quiet forking path the prompt-motion kill does not cover; it is now
its own kill.

**8.2 Arms (four: two verdict-bearing, one exploratory, one floor; all
budget-matched where budgeted).**

* **governed** — the §5 compiled block; run at every registered B.
* **raw-matched** — **the primary comparator; every utility and behavioral
  gate that compares arms reads this one.** Identical item multiset to the
  governed render at the same B. **Withheld-item rule (registered):** items
  the compiler withholds — whether for adverse evidence or for budget — are
  absent from *both* arms; withholding is selection, and raw-matched holds
  selection fixed. Same render order as governed; all governance structure
  stripped: no status header, no caveat qualifiers, dual-present candidate
  sets rendered as plain unmarked items, no defer or withdrawal notices;
  summarized items replaced by the deterministic raw prefix at the same
  token cost. This isolates *presentation structure* from both information
  quantity and item selection.
* **raw-native** — **exploratory, reported only; no gate reads it; run at
  B = 1,500 only.** Source-native top-k order truncated to B by token
  count, no evidence-derived selection or ordering of any kind, including
  items governance would withhold. Reported alongside the gated results to
  show the joint selection+structure effect; promoting it to a
  verdict-bearing comparator is a matter for a future registration.
* **none** — no context block; floor control; one run per row
  (budget-independent).

**Registered arm-equivalence claim (narrower than "equal compute", which
is claimed nowhere in this memo):** governed and raw-matched share exactly
five things — the same admissible evidence set (the adapter output for the
row), the same input-token budget B, the same consumer artifacts (§8.1),
the same decoding configuration, and the same output-token limit (§8.4).
Nothing in this registration asserts that the two arms induce equal
computation in the consumer, and no gate depends on such a claim.

**8.3 Rows.** Each cell row = (query, arm, B where applicable) → one model
call → parsed answer under the §8.4 output contract (fail-closed to
`unparseable`, which scores as wrong for the arm that produced it — the
malformed-input lesson F3, applied to model output). Per query: governed
and raw-matched at each of the two Bs, raw-native once, none once — six
calls.

**8.4 Consumer output contract (machine-readable; registered now).** Ships
as a fourth schema, `harness/ctx/schema/consumer_output.schema.json`, plus
the versioned constants instance
`harness/ctx/policy/consumer_output_contract_v1.json`; motion after first
scoring is a kill.

* **Allowed response shape (exact):** one JSON object,
  `{"answer": <non-empty string>, "hedged": <boolean>}`, both fields
  required, `additionalProperties: false`. The typed `hedged` field
  replaces r2's prose hedge-marker rubric in G-B2: unqualified assertion
  is `hedged = false`, measured directly rather than judged.
* **Output-token limit:** `max_new_tokens = 256`, identical across all
  arms, cells, and budgets.
* **Stopping rules:** generation ends at the model's EOS token or the
  256-token cap, whichever comes first; no other stop sequences.
* **Extra prose:** the parser extracts the first balanced JSON object in
  the raw output; any non-whitespace characters outside that object set a
  per-row `extra_prose` flag, recorded and reported per arm × B, but do
  not by themselves fail the row — the tolerance is arm-neutral, and
  strictness here would only burn power.
* **Parse failure:** no balanced JSON object, an object left unbalanced by
  the token cap, or an object failing schema validation (missing or extra
  fields, wrong types, empty answer) → the row is `unparseable`.
* **Scoring of malformed output:** `unparseable` scores as **wrong for the
  arm that produced it**, and additionally feeds a reported malformed-rate
  per arm × B, so a parser-hostile arm is visible in the record rather
  than only penalized.

## 9. Gates (all fixed here; exact counts where possible)

**Integrity gates (any failure → `blocked`, no verdict):**

* **G-C1 determinism** — double-run byte-identity of every compiled block
  and audit packet, all arms and Bs; consumer calls replayed twice must
  agree on parsed answers (temperature-0 check; any nondeterminism is
  quarantined to a recorded row list and that list must be < 1% of rows or
  the run blocks).
* **G-C2 label-blindness** — no adapter or compiler code path reads truth
  columns. **Normative mechanism: the poisoned-label canary cell** (labels
  permuted; compiled output must be byte-identical) — it catches any actual
  read path, imported or not. The import audit is retained as **advisory
  and function-granular only**: the canonical policy-block source is a
  scorer module whose other functions read truth columns by design, so a
  module-granularity audit cannot distinguish importing the frozen policy
  function from being able to reach label-reading code.
* **G-C3 budget integrity** — 100% of rows within their registered B, at
  every B; zero silent drops.
* **G-C4 invariant suite** — I1/I2/I4/I5/I6/I7 structural checks all-pass
  on every emitted packet.

**Utility gates (H1; all comparisons are governed vs raw-matched, evaluated
per registered B):**

* **G-U1** — on harm-class FAM cells: wrong-action mass (governed) ≤
  wrong-action mass (raw-matched) on **every** such cell, with strict `<`
  on **at least one** cell, at each B. Tie handling registered now: cells
  where raw-matched wrong-action mass = 0 count toward the ≤ clause and are
  excluded from the strict clause — a 0–0 cell is a pass of ≤, not a
  failure of `<`.
* **G-U2** — on clean cells: correct answers lost by governance (withheld/
  deferred/caveated-into-wrong) ≤ **5%** of correct traffic per cell, at
  each B (the program-precedent suppression ceiling, now applied at the
  consumer).
* **G-U3** — on the organic cell: stale adoptions (consumer asserts a
  superseded fact) strictly fewer under governed than raw-matched.
  Verdict-bearing only under `corpus=organic`; under `corpus=synthetic`
  this gate is a **reported metric**, excluded from the §10 decision table
  (§4.2).

**Behavioral gates (H2; each against a registered chance baseline):**

* **G-B1 caveat sensitivity** — on matched item pairs rendered with vs
  without caveat (same content, counterbalanced across rows), the
  consumer's adoption rate of caveated-wrong items must be lower than
  uncaveated-wrong by a margin exceeding the permutation-null 95% bound;
  AND the correctness guard, now **relative** (r1's absolute 80% gated on
  consumer strength, not caveat design): adoption of caveated-*correct*
  items ≥ adoption of uncaveated-correct items − **15 percentage points**.
  The 15-pp constant is a first-registration heuristic, not derived;
  revising it requires re-issue of this memo.
* **G-B2 dual-present discipline** — on dual-present rows: answers outside
  the presented candidate set ≤ 2% of rows; unqualified assertion
  (`hedged = false` per the §8.4 output contract — a typed field, not a
  prose rubric) ≤ 10% of rows. Both constants
  are first-registration heuristics, not derived from data; they are fixed
  in advance and revisable only by re-issue.
* **G-B3 withdrawal compliance** — on the multi-turn cells: turn-3 reliance
  on the withdrawn item < turn-3 reliance in a control session where no
  withdrawal notice was compiled, margin over permutation null.

## 10. Kill conditions and verdict decision table

Kills (any → run stops, memo records the trigger): schema, disposition
policy table, rendering template, prompt, output-contract (§8.4), or
budget-grid motion after first scoring; consumer selection or motion after
the first evaluation render
(§8.1); any adapter reading labels; any compiler code path invoking a
model; any edit to a committed FAM artifact or to either certified
contract; discovery of a specification ambiguity that two implementers
resolve differently (recorded, then this memo is re-issued — ambiguity is a
defect of the registration, not something to adjudicate ad hoc).

**Verdict — total, mutually exclusive decision table.** The verdict is the
**first matching row** (top wins); the final row is a residual, so every
gate outcome lands in exactly one row. All conditions are evaluated over
all registered Bs. If `corpus=synthetic` was triggered, it is carried as a
suffix on whatever verdict is issued.

| # | condition | verdict |
|---|---|---|
| 1 | any integrity gate (G-C1–G-C4) fails | `blocked` |
| 2 | G-U2 fails at any B, **or** wrong-action (governed) > wrong-action (raw-matched) on any harm-class cell at any B | `context-compiler-negative` |
| 3 | at **every** registered B: G-U1's strict-improvement clause fails **and** G-B1 fails | `compiler-moot` |
| 4 | every utility and behavioral gate passes at every B | `context-compiler-evidence-GO(H1,H2)` |
| 5 | anything else | `context-compiler-evidence-partial(<failed gates>)` |

Row 3 is H0 confirmed — structure adds nothing this consumer uses — and is
the single most decision-relevant possible outcome. Row 5 replaces r1's
`GO(H1)-consumer-limited`, which was not total; the old verdict survives as
the special case of row 5 where the failed-gate list is all-behavioral. Row
5 must name every failed gate in the verdict string.

**No verdict, GO included, may be claimed as:** certification of any reader
or model; authorization for live serving, prompting use in production, or
enrollment in any contract; evidence beyond the pinned model, prompts,
budgets, cells, and corpora named here. A GO earns exactly one thing: the
right to register the *next* experiment (second consumer model, or live
S2-style session replay) as its own pre-registration.

## 11. Deliverables and checkpoints (two, not rungs)

1. **Build checkpoint:** `harness/ctx/` (shared library; **four** schemas
   — context item, adapter output, disposition policy, consumer output —
   plus the versioned disposition-policy and output-contract instances and
   registered rendering templates; two adapters; compiler; deterministic
   summarizer; budget allocator) + the artifact-level consumer pin
   `consumer_pin.json` (§8.1, subsuming the §6 tokenizer shas) + hermetic
   tests for §9's integrity gates + this memo's schemas frozen. Merges on
   its own review; produces no verdict.
2. **Scoring run:** sealed replay, gates evaluated, results appended to
   this memo append-only (§12+), verdict issued. One run; re-runs only
   under a recorded re-issue of this memo.

Estimated blast radius: zero engine files, zero `results/` mutations, new
code confined to `harness/ctx/` and `tests/`.

## 12. Build-phase clarifications (append-only)

**C-1 (2026-07-09, explicitly approved): scope of §7's packet
prohibition.** §7's "re-derived from run artifacts (not from the old
packets)" prohibits recycling prior packet *items or dispositions* as
PR-13 context items. It does not prohibit read-only use of the committed
W2 packet trees as the evidence surface required by the
already-registered `witness_alt_candidate_set` signal (§4.1), whose
frozen policy block is defined over those trees and over nothing else.
Implementation constraints, registered with the clarification: the
adapter imports the frozen policy block and its registered input
constructors from the §4.1 canonical source
(`harness/action_boundary_score.py`), reading only the minimum committed
packet fields those constructors require; every emitted signal carries
provenance to the packet-tree artifact and the frozen policy version;
the PR-12 envelope is not reproduced, packet generation is not
recomputed, and no prior disposition is carried forward into any item.
This is a narrow clarification of registered text, not a re-issue: no
gate, bound, arm, kill, or verdict condition changes.

**A-1 (2026-07-10): build-checkpoint amendment — the withdrawal
trigger.** The build checkpoint merged at `6c7b6a2` under-implemented
§3's withdrawal rule. This entry amends the checkpoint; it does not
re-issue this memo. (§12 accordingly carries clarifications,
amendments, and pre-scoring registrations — all append-only.)

*Exact behavior change.* `compile()` emitted a withdrawal notice for a
prior-rendered item iff its resolved disposition was `withhold` or
`defer`. It now does so iff that holds **or** the item's `state` is
`superseded` or `quarantined`. No other line of the compiler changes.

*Why it is an amendment and not a design change.* §3 registers
`withdraw` for adverse evidence about an already-served item arriving
between turns. The frozen policy table maps exactly that evidence to
`caveat` — `state: superseded` via R12, `signal: superseded_by` and
`superseded_by_path` via R10/R11 — and `caveat` is lower precedence
than `withhold`/`defer`, so the merged predicate could not see it. The
registered multi-turn cells are built on ledger supersession; without
this amendment no withdrawal notice would ever be compiled in them and
**G-B3 would be unmeasurable by construction**. Nothing in §3's
vocabulary, §3's precedence list, the policy artifact, §5's signature,
or any gate, bound, arm, kill, or verdict condition changes.

*Scope, bounded in both directions.* The `prior_rendered` guard
short-circuits first, so an item never served at an earlier turn is
untouched; turn-1 compiles have an empty `prior_rendered` and are
therefore byte-identical before and after; every single-turn cell (all
13 FAM cells and the organic cell) is byte-identical before and after.
Only turn ≥ 2 of a multi-turn session can differ, and only for an item
that session already served. Prior-rendered `quarantined` items already
withdrew through the disposition clause (R01 → `withhold`); the state
clause is redundant for them and is written anyway so the predicate
reads on state directly rather than through a policy-table coincidence.

*Test coverage.* Three hermetic tests in `tests/test_ctx_compiler.py`
pin the amended predicate and both boundaries:
`test_withdrawal_fires_on_superseded_state_at_caveat_disposition` (the
new behavior — it fails against the `6c7b6a2` compiler, which returns
`caveat`), `test_state_transition_never_withdraws_an_unserved_item`
(turn-1 boundary), and `test_prior_rendered_clean_item_is_not_withdrawn`
(benign-state boundary). The pre-existing
`test_withdrawal_notice_on_multiturn` covers the unchanged quarantined
path, and `tests/test_ctx_cells.py` exercises the amendment end-to-end
over a throwaway synthetic ledger. The change ships as one isolated
commit ("fix(ctx): withdrawal trigger fires on superseded/quarantined
state transitions") carrying nothing else.
