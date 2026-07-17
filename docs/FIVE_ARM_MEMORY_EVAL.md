# Five-Arm Governed Memory Evaluation

## Status

The local experiment harness is implemented. It can seal inputs, execute all five arms under one 1,500-token context budget, enforce paired candidate reuse, and score both memory benefit and governance harm.

The included dry run is a plumbing proof, not scientific evidence. A real benchmark run remains on hold until the official dataset transformation, real embedding pin, consumer seal, and scoring manifest are reviewed and frozen.

## The five fixed arms

| Arm | Retrieval | Context treatment | What it identifies |
|---|---|---|---|
| `no_memory` | None | None | Consumer-only lower bound |
| `vector_raw` | Exact cosine over every record | Verbatim, budget-truncated | Ordinary RAG baseline |
| `vector_governed` | Same candidate IDs as `vector_raw` | Lifecycle-aware compiler | Value of constructive forgetting with retrieval held fixed |
| `fam_raw` | FAM scope prototypes, then provenance-to-record reranking | Verbatim, budget-truncated | Effect of FAM condensation without governance |
| `fam_governed` | Same candidate IDs as `fam_raw` | Lifecycle-aware compiler | Full proposed memory design |

The primary causal comparisons are `vector_governed - vector_raw` and `fam_governed - fam_raw`. The runner queries each retriever once per question and reuses that exact ordered candidate-ID list for its raw and governed arms.

## Why the agent harness fits the memory model

The fit depends on a strict ownership boundary:

1. The append-only ledger owns immutable text, values, serials, and source identity.
2. Lifecycle resolution labels records as current, superseded, or unresolved forks. Constructive forgetting changes what is asserted; it does not erase history.
3. Exact vector search indexes every record.
4. FAM condenses scope keys and stores the record IDs that formed each prototype as provenance.
5. After FAM retrieval, original record embeddings rerank the recovered ledger IDs. The consumer only sees ledger payloads.

This preserves FAM's useful online condensation while avoiding a category error: an EMA-blended FAM value is not an authoritative evolving fact. If an agentic framework requires its memory object to own and rewrite answer payloads, it does not yet fit this model. It must first separate authoritative records from disposable retrieval indexes and expose provenance IDs at retrieval time.

## Constructive forgetting behavior

For a unique latest serial, the latest record is `agent-readable` and older versions are `superseded`. Raw arms receive the original candidate text. Governed arms pass the same candidates through the existing context compiler, which prioritizes current evidence and visibly caveats superseded content.

Distinct values with the same maximum serial form a deterministic contradiction set. They are marked `human-review` and fail closed to an unresolved notice under the current compiler policy.

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
- Fork-adoption rate: among contested questions with well-formed responses, the fraction adopting any contested value. Contested scopes are counted and reported, never adjudicated: whether contested questions are scoreable at all, and whether the correct governed response there is abstention rather than the annotated value, are pre-registration decisions reserved for the human.
- Abstention and malformed-output rates: safety and interface costs.
- Prompt tokens and total latency: operational costs by arm.

The value claim requires all three of the following, evaluated per family (`vector_governed` vs `vector_raw`, `fam_governed` vs `fam_raw`):

1. A reduction in stale-adoption rate — margin UNREGISTERED; pre-registration required.
2. A bounded clean-answer loss — bound UNREGISTERED; pre-registration required.
3. A current-adoption floor on the stale-eligible stratum — floor UNREGISTERED; pre-registration required.

Criterion 3 exists so the claim can come out against governance: an arm that wins criteria 1 and 2 by suppressing every evolving answer fails criterion 3. Overall accuracy alone can hide any of these failures. None of the three thresholds is registered yet; they are the human's pre-registration decision, and until numeric values are registered, any run is exploratory-only. A rate whose `n` is 0 satisfies no criterion — it is missing data, and that criterion is not evaluable on that corpus.

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

## Seal, run, and score workflow

1. Pin an official MemoryAgentBench dataset revision and transform `fact_sh` into the normalized JSONL format. Validate every extracted serial and answer against the source row.
2. Pin the record/query embedding model and revision. Materialize embeddings before sealing.
3. Freeze candidate count, FAM prototype count, FAM capacity, consumer pin, arm order, 1,500-token budget, and scoring version.
4. Seal records, questions, embeddings, and protocol with `seal_manifest`.
5. Reload with `verify_manifest` immediately before execution.
6. Execute with `FiveArmRunner`, retain every row and audit packet, then score with `score_rows`.

Run the offline plumbing proof with:

```bash
python -m harness.memory_eval.dry_run --output-dir /tmp/fam-memory-eval-dry-run
```

The output is explicitly labeled `plumbing-only; not benchmark evidence`.

## Remaining work before a real run

- Implement and review the official `Conflict_Resolution/fact_sh` to normalized-JSONL transformer. The upstream benchmark was updated in 2025–2026, so pin a commit or dataset revision rather than relying on a floating `main`.
- Choose and pin a semantic embedding model. The dry run's hash encoder carries no semantics.
- Bind the real consumer artifact/revision and decoding settings into `retriever_settings` before sealing; the existing Qwen3 consumer can satisfy the runner interface once its weights are sealed on the scoring host.
- Pre-register the query set, primary metrics, the numeric stale-adoption margin, clean-answer-loss bound, and current-adoption floor, the contested-question scoring policy, and failure rules.
- Run a small sealed pilot for data-shape and resource checks, discard it, then seal and execute the scoring run once.

The official benchmark repository documents FactConsolidation as its conflict-resolution task and currently scores it with `substring_exact_match`: [MemoryAgentBench repository](https://github.com/HUST-AI-HYZ/MemoryAgentBench) and [dataset release](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench).
