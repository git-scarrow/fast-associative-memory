# FAM Condensation with Constructive Forgetting — Experimental Design

**Date:** 2026-07-17

**Status:** Approved design; ready for implementation planning after written-spec review

**Branch:** `codex/five-arm-memory-eval`

## 1. Goal and claim hierarchy

The experiment restores the repository's historical ordering of contributions:

1. **FAM is the mechanism.** The primary claim concerns static-vigilance,
   online, class-conditional condensation: each record embedding is written
   once; a same-scope hit at or above vigilance is absorbed through EMA key
   drift and provenance union; otherwise a new prototype is allocated.
2. **Constructive forgetting is an application.** The secondary claim concerns
   a lifecycle-aware compiler that governs which records retrieved from a
   demonstrated FAM index may reach a consumer.

The primary experiment must therefore establish the FAM mechanism before the
constructive-forgetting result may be read as confirmatory. A favorable
governance result cannot rescue a failed or inert FAM treatment.

The bounded headline is:

> Under a fixed, non-binding index capacity and a matched non-condensing CAM
> control, online FAM condenses the retrieval index by a registered amount
> while keeping authoritative-current-record recall within a registered loss
> bound. Conditional on that mechanism claim passing, constructive forgetting
> over the same FAM candidates reduces stale adoption without unacceptable
> clean-answer loss or answer suppression.

This design does not claim that constructive forgetting is intrinsic to FAM,
that FAM solves governance inside its core, or that governance has a larger
effect on FAM than on other retrievers.

## 2. Alternatives considered

### 2.1 Keep exact vector retrieval as the mechanism control — rejected

An all-record exact-vector index changes the storage representation, query
path, capacity semantics, provenance expansion, and reranking path at once. It
is a useful retrieval ceiling, but it cannot isolate online condensation. It
remains an exploratory, consumer-free diagnostic.

### 2.2 Execute only the three verdict-bearing arms — valid but not selected

`exemplar_raw`, `fam_raw`, and `fam_governed` are sufficient for the two
registered claims. Keeping two additional exploratory arms preserves the
existing five-arm runner shape and provides diagnostic floors without adding
confirmatory hypotheses.

### 2.3 Full 2×2 FAM-by-governance interaction — deferred

The five executed arms make a difference-in-differences estimate possible, but
this design does not register it. A claim that constructive forgetting works
especially well with FAM would require a separately registered interaction:

`(fam_governed − fam_raw) − (exemplar_governed − exemplar_raw)`.

That is a distinct claim family with additional multiplicity and power
requirements. It remains exploratory.

## 3. Executed arms

| Arm | Retrieval | Context | Claim status |
|---|---|---|---|
| `no_memory` | none | empty memory block | exploratory floor |
| `exemplar_raw` | matched allocate-only CAM | lifecycle-blind raw renderer | primary control E0 |
| `exemplar_governed` | exact E0 candidate IDs | constructive-forgetting compiler | exploratory |
| `fam_raw` | live online FAM condensation | same raw renderer as E0 | primary treatment F0 |
| `fam_governed` | exact F0 candidate IDs | constructive-forgetting compiler | secondary treatment F1 |

The exact-vector ceiling is evaluated at retrieval time only and does not add a
sixth consumer arm.

Within each retrieval family, raw and governed arms reuse the same ordered
candidate-ID tuple. Between E0 and F0, every component is matched except the
condensation package described in §4.

## 4. Retrieval treatments

### 4.1 Shared CAM envelope

E0 and F0 share:

- the sealed record and query embeddings;
- one-hot scope labels and their deterministic label map;
- manifest record order and one write per `learn_local` call;
- `max_entries = sealed normalized record_count`, making capacity non-binding;
- float32 keys and values with `use_bfloat16 = false`;
- `adaptive_eviction = false` and `use_lfu = true` (both inert while capacity
  remains non-binding, but sealed so the treatment has no implicit defaults);
- `inference_temp = 0.05`, `inference_sim_floor = 0.0`, no retrieval-floor
  policy, no retrieval-truncation policy, and no NSTP controller;
- identical key normalization, candidate `k`, prototype search width, and tie
  rules; `candidate_k` and `prototype_k` are mandatory preregistration fields
  with no implementation defaults;
- identical provenance-to-record expansion and exact record-embedding rerank;
- zero sleep/replay and `dynamic_vigilance = null`;
- the same raw renderer, 1,500-token context budget, consumer, prompt,
  decoding configuration, and structured output parser.

The normalized JSONL byte order is the online ingestion order. The transformer
must preserve the official source order, emit stable record IDs, and record its
source-to-normalized reconciliation. The manifest fingerprints the ordered
record list, so reordering invalidates the seal.

Records are written individually. Batch ingestion is prohibited because
`ContinuousCAM.learn_local` classifies every member of a batch against the
pre-call state; a cold-start batch could allocate multiple records that a true
online sequence would merge.

### 4.2 F0: live FAM condensation

F0 uses actual record embeddings, not precomputed scope centroids. Its fixed v1
mechanism is:

- static `vigilance = 0.85`;
- `hebb_lr = 0.1`;
- `key_lr = 0.05`;
- `ema_beta = 0.05`;
- `immutable_keys = false`;
- `track_provenance = true`;
- `dynamic_vigilance = null`;
- sleep/replay disabled;
- capacity non-binding, so any drop or eviction is invalid.

On a same-scope nearest-prototype hit at or above vigilance, F0 applies the
core's adaptive EMA key update and unions the record ID into that prototype's
provenance. A miss or class collision allocates a new prototype.

`hebb_lr` is sealed even though identical one-hot values make it effectively
inert in this adapter and prototype values are never authoritative facts.

### 4.3 E0: matched noncondensing exemplar CAM

E0 uses the same CAM storage, labels, ordering, capacity, query path,
provenance expansion, and exact rerank as F0. Its only treatment difference is
an explicit `write_mode = "allocate-only"`:

- every input record allocates one exemplar prototype;
- same-scope absorption and EMA key drift are disabled;
- every prototype retains exactly one record ID as provenance.

The implementation must use an explicit write mode rather than an out-of-range
vigilance value, so the control cannot silently become condensation if core
validation changes.

### 4.4 Raw rendering

`exemplar_raw` and `fam_raw` use the same deterministic skip-not-stop renderer:
an over-budget candidate is skipped and later candidates are still considered.
Stopping at the first oversized candidate is prohibited because it makes
candidate order an unregistered, systematically pro-governed truncation
treatment.

## 5. Treatment-fidelity and integrity gates

All gates below run before any value verdict is read.

### 5.1 Build accounting

For both E0 and F0:

- `written == record_count`;
- `merged + allocated + dropped == written`;
- `dropped == 0`;
- `evicted == 0`;
- occupied provenance sets are disjoint and scope-consistent;
- the union of occupied provenance equals the complete sealed record-ID set;
- rebuilding from the seal produces the same index SHA-256.

For E0 specifically, `merged == 0`, `allocated == record_count`, and occupied
prototype count equals record count.

For F0 specifically, at least one write must merge and at least one merged key
must change. If the live path executes but neither occurs, the FAM mechanism is
`not-evaluable`; the run cannot produce the headline claim.

Prototype count is not used to detect eviction. Occupancy can remain constant
while a full CAM reuses a slot and destroys provenance.

### 5.2 Sealed treatment identity

The scoring-run seal contains a strict, closed FAM/exemplar schema. It rejects
missing, unknown, defaulted, or wrongly typed treatment fields. The sealed
builder constructs both retrievers exclusively from that schema and compares
their realized attestations before returning a runner.

Any integrity failure produces `blocked`, occurs before `consumer.generate`,
and makes all outcome rows non-evidence.

## 6. Primary mechanism claim

The primary FAM claim is a conjunction evaluated on E0 versus F0:

### M1 — index condensation

F0's occupied-prototype reduction relative to E0 must meet a numeric rate
registered before outcome inspection. The implementation represents the rate
as an exact count at the sealed record denominator and records the implied
integer threshold.

### M2 — authoritative-current retrieval fidelity

For each non-contested question, the authoritative record set consists of the
maximum-serial ledger records whose normalized value equals the annotated
answer. Recall@`candidate_k` is one when the candidate tuple contains any
member of this set.

The paired F0-minus-E0 recall loss must remain within a registered bound on a
registered minimum denominator. Questions are paired by query and inference is
clustered by ledger scope.

The mechanism verdict is `FAM-condensation-GO` only when every integrity gate,
M1, and M2 passes. If it fails, all constructive-forgetting results are
exploratory regardless of their values.

Numeric M1/M2 bounds are human preregistration decisions. Code supplies no
defaults and the seal refuses to exist while either is absent. Corpus-shape
probing may disclose denominators but never recall differences or condensation
outcomes.

## 7. Secondary constructive-forgetting claim

Only after `FAM-condensation-GO`, compare F1 with F0 using their identical FAM
candidate IDs:

- **A1:** stale adoptions decrease by the registered integer margin;
- **A2:** clean-answer losses stay within the registered integer bound;
- **A3:** current adoption clears the registered anti-suppression floor.

The existing exact structured-answer scorer remains primary. Containment is
exploratory because negations and hedges can produce false wins. H1 uses the
fixed-full stale-eligible denominator. Ledger equality remains raw string
equality plus a sealed invariant rejecting normalized-equal/raw-unequal
same-serial values. Contested questions remain exploratory unless separately
registered.

The application verdict is `FAM-constructive-forgetting-GO` only when A1, A2,
and A3 all pass. Fixed-sequence gatekeeping—mechanism first, application
second—prevents the secondary claim from surviving a failed FAM treatment.

## 8. Manifest and preflight

The next manifest version binds:

1. **Corpus:** upstream repository/dataset identity and immutable revision,
   `Conflict_Resolution/fact_sh` configuration, source hashes and declared
   cardinalities, transformer source hash/version, normalized schema and JSONL
   hashes, loaded counts, and reconciliation report.
2. **Embeddings:** model and immutable revision, tokenizer/artifact hashes,
   pooling, normalization, dtype, record/query prefixes, encoder source hash,
   runtime versions, and the executed vector fingerprints.
3. **Consumer:** complete consumer-pin file hash and artifact hashes, tokenizer,
   chat template, weights/config/index, decoding, precision/quantization,
   placement policy, prompt template, parser schema, and parser source hash.
4. **Code:** commit and clean-worktree assertion plus a source rollup covering
   the CAM core, retrievers, runner, ledger, compiler, scorer, preregistration,
   manifest, sealed builder, schemas, parser, and policy.
5. **Treatments:** every shared and differing E0/F0 field from §4, raw rendering
   semantics, candidate budget, and the complete registration block.
6. **Realized indexes:** build accounting, occupied count, sorted provenance
   partition, and deterministic index hash for E0 and F0.

Preflight reports all failures together where possible and raises before any
consumer call. Registration fields are not merely validated: scorer,
equivalence, raw truncation, denominator policy, claim hierarchy, derived
integer gates, corpus reconciliation, and caveat integrity must each drive the
executing path or verdict.

## 9. Data flow

1. Transform and reconcile official source rows into ordered normalized JSONL.
2. Produce and seal record/query embeddings.
3. Build E0 and F0 deterministically from the ordered records.
4. Verify build accounting, provenance partitions, mechanism activity, and
   index hashes.
5. Run the outcome-incapable shape probe to expose denominators and budget
   binding only.
6. Complete numeric preregistration without viewing retrieval differences or
   consumer outcomes.
7. Seal the full treatment and immediately run preflight.
8. Execute each question once per retrieval family, reusing candidate IDs for
   raw/governed pairs.
9. Evaluate the FAM mechanism gates; only on GO evaluate the application gates.
10. Preserve all other comparisons as exploratory.

## 10. Verification strategy

Implementation follows test-first slices:

1. **Live FAM writes:** prove one actual embedding per call, manifest order,
   above-threshold merge and key drift, below-threshold allocation, class
   collision allocation, and provenance union.
2. **Matched exemplar control:** prove allocate-only behavior and byte-matched
   query/rerank plumbing.
3. **Capacity integrity:** reproduce silent eviction with constant occupancy,
   then prove the provenance-partition/no-loss gate catches it.
4. **Mechanism scoring:** test authoritative-record-set construction, paired
   recall, exact integer compression/recall gates, empty denominators, and
   fixed-sequence verdicts.
5. **Strict manifest:** tamper every corpus, embedding, consumer, code, treatment,
   registration, and realized-index field independently.
6. **Sealed builder:** prove every setting is consumed from the manifest and
   every failure occurs before an exploding test consumer can run.
7. **Corpus reconciliation:** fixture-test missing, duplicated, filtered,
   ambiguous, reordered, and revision-drifted source rows.
8. **End-to-end:** run a sealed deterministic test double before any separately
   authorized model-host smoke test.

The existing targeted baseline is 96 passing memory-evaluation tests before
this design's implementation.

## 11. Explicit non-claims

This experiment does not establish:

- that FAM learns new representations;
- that EMA-blended values are authoritative facts;
- that constructive forgetting is part of FAM core;
- that governance works generically across retrieval systems;
- that FAM and governance have a positive statistical interaction;
- that exact-vector retrieval is inferior as an unconstrained ceiling;
- that dynamic vigilance, sleep replay, adaptive eviction, or capacity pressure
  contributes to the result;
- that a synthetic dry run is benchmark evidence.
