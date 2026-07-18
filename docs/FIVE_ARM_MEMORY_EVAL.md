# Five-Arm FAM-First Memory Evaluation

## Status

Phase A is implemented only as plumbing: the local harness can seal manifest-v3 plumbing inputs, rebuild matched E0/F0 CAM indexes, execute the explicitly synthetic five-arm fixture under one 1,500-token context budget, enforce paired candidate reuse, and calculate non-authoritative mechanism/application diagnostics.

Confirmatory capability is not implemented. Public scoring-run sealing and `build_sealed_run` fail closed before generation—even for a digest-consistent hand-built manifest—because the Phase B source-provenance and reconciliation envelope does not exist. Preflight remains available for audit, performs full manifest binding, and reports that missing Phase B gate.

The included dry run is a deterministic `synthetic/plumbing` proof with `admissible: false`. It is not benchmark evidence. Its numerical gates and retrieval widths are visibly named fixture assertions returned in the artifact; they are neither production defaults nor a scoring-run registration. The plumbing manifest contains no registration or confirmatory index attestations. The artifact separately reports attestations from the verified rebuild, reports synthetic gate booleans as non-authoritative diagnostics, and shows the receipt-bound authoritative verdict as `blocked`.

## The five fixed arms

| Arm | Retrieval | Context treatment | What it identifies |
|---|---|---|---|
| `no_memory` | None | None | Exploratory consumer-only floor |
| `exemplar_raw` (E0) | Matched allocate-only CAM | Verbatim, budget-truncated | Primary noncondensing mechanism control |
| `exemplar_governed` (E1) | Same candidate IDs as E0 | Lifecycle-aware compiler | Exploratory governance diagnostic |
| `fam_raw` (F0) | Live FAM condensation plus provenance-to-record reranking | Same raw renderer as E0 | Primary mechanism treatment and application control |
| `fam_governed` (F1) | Same candidate IDs as F0 | Lifecycle-aware compiler | Secondary constructive-forgetting treatment |

The fixed claim order is mechanism first, application second. E0 versus F0 tests whether FAM reduces occupied prototypes while retaining authoritative-current recall at `candidate_k`. Only if that mechanism is active and passes may F0 versus F1 support the constructive-forgetting application claim. A favorable F1 result cannot rescue an inert or failed FAM mechanism. The runner queries each retriever once per question and reuses the exact ordered candidate-ID tuple for its raw and governed arms.

The registered v1 treatment identity is exact: vigilance `0.85`, Hebbian learning rate `0.1`, key learning rate `0.05`, EMA beta `0.05`, inference temperature `0.05`, float32 (`use_bfloat16 = false`), adaptive eviction off, and LFU on. Dynamic vigilance, retrieval floor/truncation, NSTP, and sleep are explicitly disabled; ingest order and the two write modes are fixed. `candidate_k` and `cam_prototype_k` remain mandatory D-M4/D-M5 human choices rather than fixed values.

Exact-vector retrieval remains a consumer-free exploratory ceiling. It is useful for diagnosing retrieval headroom, but it changes the representation, query path, and capacity semantics, so it is not a matched mechanism control and is never a sixth consumer arm.

## Why the agent harness fits the memory model

The fit depends on a strict ownership boundary:

1. The append-only ledger owns immutable text, values, serials, and source identity.
2. Lifecycle resolution labels records as current, superseded, or unresolved forks. Constructive forgetting changes what is asserted; it does not erase history.
3. E0 and F0 share the same CAM capacity, ingestion order, query path, provenance expansion, and exact record-embedding rerank. E0 allocates every write; F0 may condense above-vigilance same-scope writes. On this fixed static-vigilance path, F0 selects the nearest occupied prototype with the matching scope label before applying vigilance, so a closer cross-scope key cannot shield a valid merge.
4. FAM stores the record IDs that formed each prototype as provenance.
5. After either CAM retrieves prototypes, original record embeddings rerank the recovered ledger IDs. The consumer only sees ledger payloads.

This preserves FAM's useful online condensation while avoiding a category error: an EMA-blended FAM value is not an authoritative evolving fact. If an agentic framework requires its memory object to own and rewrite answer payloads, it does not yet fit this model. It must first separate authoritative records from disposable retrieval indexes and expose provenance IDs at retrieval time.

## Constructive forgetting behavior

For a unique latest serial, the latest record is `agent-readable` and older versions are `superseded`. Raw arms receive the original candidate text. Governed arms pass the same candidates through the existing context compiler, which prioritizes current evidence and visibly caveats superseded content.

Distinct values with the same maximum serial form a deterministic contradiction set. They are marked `human-review` and fail closed to an unresolved notice under the current compiler policy. A raw-distinct same-scope/same-serial pair that normalizes equal under the scorer is ambiguous rather than genuinely contested: sealing, verification, application scoring, and mechanism scoring all reject it.

No record is deleted. The ledger remains auditable, while assertion policy changes what the consumer can safely rely on.

## Metrics that show value

Every question's ledger scope is scored in exactly one stratum:

- **Clean**: one distinct normalized value across the scope's records.
- **Stale-eligible**: a single latest value plus at least one distinct superseded value.
- **Contested**: distinct values at the maximum serial (an unresolved fork). Contested scopes never enter the stale-eligible stratum, even when they carry superseded history, so a wrong-fork answer cannot dilute the stale-adoption denominator.

Every reported rate carries its denominator: rates serialize as `{"value": ..., "n": ...}` and `value` is `null` whenever `n` is 0. "No data" is never reported as a measured 0.0, and a gate must read `n` before trusting any rate. The report also carries corpus-level stratum counts (total, clean, stale-eligible, and contested question counts; total and contested ledger-scope counts).

- Accuracy: exact match after Unicode normalization, case folding, and whitespace normalization.
- Stale-adoption rate: among stale-eligible questions, the fraction of well-formed responses (answer or abstain) adopting a distinct superseded value. Malformed rows leave this denominator — an interface failure is not stale-avoidance. Abstained rows stay in it as non-adoption, because abstention is the governed behavior under test.
- Current-adoption rate: among all stale-eligible questions (malformed rows included), the fraction answered with the expected current value. This is the anti-suppression floor: a governed arm that abstains on every evolving question scores zero here, so blanket suppression cannot satisfy the value claim.
- Clean-answer loss (paired): among clean questions, the fraction where raw is correct and its paired governed arm is not.
- Stale-eligible loss (paired): the same paired structure over stale-eligible questions — the current answers that raw delivered and governance destroyed.
- Fork-adoption rate: among contested questions with well-formed responses, the fraction adopting any contested value. Contested scopes are counted and reported, never adjudicated in Phase A. The Phase A registration schema accepts only `contested_disposition = "exploratory"`; confirmatory contested rules remain Phase B evaluator work.
- Abstention and malformed-output rates: safety and interface costs.
- Prompt tokens and total latency: operational costs by arm.

The secondary application claim requires all three of the following on F0 versus F1, after the E0/F0 mechanism gate passes:

1. A reduction in stale-adoption rate — margin UNREGISTERED; pre-registration required.
2. A bounded clean-answer loss — bound UNREGISTERED; pre-registration required.
3. A current-adoption floor on the stale-eligible stratum — floor UNREGISTERED; pre-registration required.

Criterion 3 exists so the claim can come out against governance: an arm that wins criteria 1 and 2 by suppressing every evolving answer fails criterion 3. Overall accuracy alone can hide any of these failures. None of the mechanism or application thresholds is registered yet; they are human pre-registration decisions. `candidate_k` and `cam_prototype_k` are also mandatory human choices (D-M4/D-M5), not public runner defaults. Phase A accepts only `abstention_bound = null` and `contested_disposition = "exploratory"` because no evaluator implements the alternatives. Confirmatory integer thresholds are derived and recorded only during future Phase B sealing. A rate whose `n` is 0 satisfies no criterion—it is missing data, and that criterion is not evaluable on that corpus.

## Normalized FactConsolidation input

The official MemoryAgentBench release stores each example as a long `context` with parallel `questions`, `answers`, and `metadata`. The project loader intentionally accepts a smaller, reviewable JSONL interchange format so the extraction of entities, relations, serials, and current answers can be audited before a run.

Record row:

```json
{"type":"record","record_id":"r-001","entity":"Ada","relation":"employer","value":"OldCo","content":"Ada works at OldCo (serial 1).","serial":1,"event_time":null,"source_id":"mab-fact-sh"}
```

Question row:

```json
{"type":"question","query_id":"q-001","entity":"Ada","relation":"employer","question":"Where does Ada work?","answer":"NewCo"}
```

Required record fields are `type`, `record_id`, `entity`, `relation`, `value`, `content`, and non-negative integer `serial`. Required question fields are `type`, `query_id`, `entity`, `relation`, `question`, and `answer`. Optional record fields are `event_time`, `ingest_time`, and `source_id`. Unexpected fields and duplicate IDs fail closed.

## Phase B seal, run, and score workflow

The following workflow is a Phase B readiness contract, not a callable Phase A path. Today steps 4–6 are deliberately disabled before generation.

1. Pin the official `fact_sh` source revision, transform it into normalized JSONL, and pass the G-I6 source-to-normalized reconciliation gate.
2. Pin the semantic record/query encoder and consumer artifacts, including revisions, hashes, preprocessing, decoding, and code rollups. Materialize embeddings before sealing.
3. Run only the outcome-incapable real-corpus shape probe, then complete the human numeric registration without inspecting outcomes.
4. Seal the complete manifest-v3 scoring treatment, including the registration memo and realized E0/F0 attestations.
5. On the scoring host, rebuild through `build_sealed_run`; preflight must validate inputs, treatment, artifacts, registration, and exact rebuilt attestations before generation.
6. Execute the single authorized real run, retain every row and audit packet, score E0/F0 mechanism counts first, and evaluate F0/F1 only after a mechanism GO.

Run the offline plumbing proof with:

```bash
python -m harness.memory_eval.dry_run --output-dir /tmp/fam-memory-eval-dry-run
```

The output is explicitly labeled `synthetic/plumbing`, `admissible: false`, and `benchmark_evidence: false`. Synthetic mechanism/application gate checks are reported as non-authoritative diagnostics; the authoritative verdict for the immutable plumbing receipt is `blocked`, never GO.

## Phase B blockers

Phase A ends at the sealed deterministic synthetic proof. It establishes plumbing behavior, not benchmark performance. Phase B requires a separate plan and authorization for all of the following:

- implement and review the official `Conflict_Resolution/fact_sh` parquet transformer without guessing entity/relation scopes from natural-language facts;
- implement G-I6 reconciliation from every pinned source row through normalized records, questions, answer lists, and `qa_pair_ids`;
- pin complete semantic encoder, consumer, tokenizer, prompt/parser, decoding, code, and runtime artifacts rather than reusing the synthetic vectors or rule consumer;
- run an outcome-incapable real-corpus shape probe and complete human registration for D-M4/D-M5, M1/M2, and A1/A2/A3;
- perform scoring-host preflight and the single authorized real execution.

Planning inspection found eight Conflict Resolution rows: `factconsolidation_sh_{6k,32k,64k,262k}` each contains one numbered fact context plus 100 questions, answer lists, and `qa_pair_ids`. Those observations guide the Phase B transformer and reconciliation work; they do not authorize Phase A code to infer fact scopes.
