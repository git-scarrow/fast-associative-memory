# Five-Arm Governed Memory Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, fixed-budget five-arm experiment that tests whether governed constructive forgetting improves an agent's use of evolving memory, and whether FAM can serve as a provenance-preserving retrieval index without becoming the source of truth.

**Architecture:** An append-only ledger owns immutable memory records and resolves their lifecycle state. Two retrievers—exact-vector and FAM—return only ledger record IDs. Raw and governed arms share each retriever's candidate IDs; governed arms compile those candidates through the existing context governance compiler. A consumer adapter answers questions, and an independent scorer reports stale adoption, clean-answer loss, abstention, token use, and latency. The existing sealed PR13 replay runner remains unchanged.

**Tech Stack:** Python 3.11+, dataclasses, PyTorch, the repository's `ContinuousCAM`, existing `harness.ctx` schemas/compiler/output parser, pytest, JSON/JSONL.

## Global Constraints

- The five arms are exactly `no_memory`, `vector_raw`, `vector_governed`, `fam_raw`, and `fam_governed`.
- Every arm has an input-context budget of 1,500 tokens. Output is limited by the existing consumer contract.
- Raw and governed variants for one retriever receive an identical ordered candidate-ID list.
- The ledger is authoritative. Retrievers may condense keys but may return only ledger IDs; answer payloads always come from the ledger.
- Experiment manifests seal records, questions, embeddings, arm definitions, budget, and scoring policy before execution.
- Production changes follow test-driven development: write a failing focused test, run it, add the minimum implementation, rerun it.
- Do not modify `harness/ctx/replay.py`, PR13 manifests, or any existing sealed result artifact.

## Task 1: Define the experiment domain model and fixed protocol

**Files:**
- Create: `harness/memory_eval/__init__.py`
- Create: `harness/memory_eval/models.py`
- Test: `tests/test_memory_eval_models.py`

- [x] Add a failing test asserting the fixed arm order, the 1,500-token budget, immutable validated records/questions, and stable scope identity.
- [x] Run `python -m pytest -q tests/test_memory_eval_models.py` and confirm import/behavior failures.
- [x] Implement frozen dataclasses `MemoryRecord`, `MemoryQuestion`, `RetrievedCandidate`, and `AnswerObservation`.
- [x] Validate non-empty IDs/scopes/content, non-negative serials, finite scores, and positive ranks.
- [x] Export `ARM_NAMES` and `CONTEXT_BUDGET_TOKENS` from the package.
- [x] Rerun the focused test and confirm it passes.

## Task 2: Add the append-only lifecycle ledger

**Files:**
- Create: `harness/memory_eval/ledger.py`
- Test: `tests/test_memory_eval_ledger.py`

- [x] Add failing tests for unique record IDs, append-only behavior, deterministic ordering, active/latest resolution, superseded history, and equal-serial contradictory forks.
- [x] Run the ledger tests and confirm expected failures.
- [x] Implement `MemoryLedger.append`, `get`, `records`, `records_for_scope`, and `resolved_state`.
- [x] Resolve a unique maximum-serial record as `agent-readable`; mark lower serials `superseded`; classify distinct equal-maximum values as a `human-review` contradiction set with a deterministic candidate-set ID.
- [x] Keep original record objects immutable and never delete or rewrite superseded payloads.
- [x] Rerun the ledger tests and confirm they pass.

## Task 3: Implement exact-vector and FAM provenance retrievers

**Files:**
- Create: `harness/memory_eval/retrievers.py`
- Test: `tests/test_memory_eval_retrievers.py`

- [x] Add failing tests showing exact cosine ranking, deterministic tie breaking, FAM retrieval returning ledger IDs, same-scope FAM condensation, and no synthesized payload/value in retriever output.
- [x] Run the retriever tests and confirm expected failures.
- [x] Implement `ExactVectorRetriever.fit/query` using normalized PyTorch dot products and record-ID tie breaking.
- [x] Implement `FAMRetriever.fit/query` using one-hot scope classes, `ContinuousCAM(track_provenance=True)`, record-ID provenance unions, and exact cosine reranking of the recovered ledger candidates.
- [x] Reject missing/duplicate embeddings and dimension mismatches before fitting.
- [x] Rerun the retriever tests and confirm they pass.

## Task 4: Adapt ledger candidates to governed context items

**Files:**
- Create: `harness/memory_eval/context.py`
- Test: `tests/test_memory_eval_context.py`

- [x] Add failing tests for active, superseded, and contradiction-fork ContextItem mappings and for deterministic raw rendering.
- [x] Run the context tests and confirm expected failures.
- [x] Implement a deterministic content hash and ContextItem conversion compatible with `context_item.schema.json`.
- [x] Map active records to `agent-readable`/`source_identity`, older versions to `superseded`/`superseded_by`, and equal-serial forks to `human-review`/`contra_fork` with a shared candidate-set ID.
- [x] Implement raw rendering that preserves candidate order and stops before the fixed token budget.
- [x] Rerun the context tests and confirm they pass.

## Task 5: Build the five-arm runner with paired candidate reuse

**Files:**
- Create: `harness/memory_eval/runner.py`
- Test: `tests/test_memory_eval_runner.py`

- [x] Add failing tests with fake retrievers and a fake consumer that prove: exactly five rows per question; each retriever is queried once; raw/governed candidate IDs match within retriever; `no_memory` has no candidates; governed compilation respects 1,500 tokens; audit rows contain rendered IDs, token count, and timing.
- [x] Run the runner tests and confirm expected failures.
- [x] Define small `Retriever` and `Consumer` protocols so real models and deterministic test doubles share the same runner.
- [x] Query vector and FAM once per question and cache their ordered candidate IDs.
- [x] Render `no_memory`, both raw arms, and both governed arms; use the existing `harness.ctx.compile.compile` for governed context and the existing consumer-output parser.
- [x] Measure retrieval, compilation, and consumer latency separately with an injectable clock.
- [x] Emit immutable `ExperimentRow` values containing answer status, answer text, candidate IDs, rendered IDs, token use, and audit data.
- [x] Rerun runner tests and confirm they pass.

## Task 6: Implement value-oriented scoring

**Files:**
- Create: `harness/memory_eval/scoring.py`
- Test: `tests/test_memory_eval_scoring.py`

- [x] Add failing tests for exact normalized correctness, stale-answer detection, abstention, malformed output, clean-scope identification, and paired clean-answer loss.
- [x] Run the scoring tests and confirm expected failures.
- [x] Score each row using authoritative ledger values only.
- [x] Define stale adoption as answering a superseded value instead of the current expected value.
- [x] Define clean cases as scopes with one distinct ledger value and no contradiction fork.
- [x] Aggregate by arm: accuracy, stale-adoption rate, abstention rate, malformed rate, mean prompt tokens, and mean latency.
- [x] For `vector_governed` versus `vector_raw` and `fam_governed` versus `fam_raw`, report paired clean-answer loss as the fraction of clean questions where raw is correct and governed is not.
- [x] Rerun scoring tests and confirm they pass.

## Task 7: Seal and verify experiment inputs

**Files:**
- Create: `harness/memory_eval/manifest.py`
- Test: `tests/test_memory_eval_manifest.py`

- [x] Add failing tests for deterministic SHA-256 fingerprints, save/load round trips, rejection of changed records/questions/embeddings/protocol, and refusal to run an unsealed manifest.
- [x] Run the manifest tests and confirm expected failures.
- [x] Implement canonical JSON serialization and fingerprints for ledger records, questions, embedding tensors, arm order, budget, retriever settings, and scoring version.
- [x] Implement `seal_manifest`, `load_manifest`, and `verify_manifest` independently of the sealed PR13 runner.
- [x] Rerun manifest tests and confirm they pass.

## Task 8: Add a normalized FactConsolidation loader and deterministic dry run

**Files:**
- Create: `harness/memory_eval/fact_consolidation.py`
- Create: `harness/memory_eval/dry_run.py`
- Test: `tests/test_memory_eval_fact_consolidation.py`
- Test: `tests/test_memory_eval_dry_run.py`

- [x] Add failing loader tests for normalized JSONL records/questions, duplicate IDs, invalid serials, and missing answer fields.
- [x] Add a failing end-to-end dry-run test that seals a tiny evolving-fact corpus, executes all five arms with deterministic embeddings/consumer behavior, and returns all required score fields.
- [x] Run both focused tests and confirm expected failures.
- [x] Implement a strict normalized JSONL loader whose record rows contain `type=record`, `record_id`, `entity`, `relation`, `value`, `content`, `serial`, and optional timestamps/source; question rows contain `type=question`, `query_id`, `entity`, `relation`, `question`, and `answer`.
- [x] Implement a deterministic hash encoder and rule-based consumer for local plumbing verification only; label dry-run output as non-benchmark evidence.
- [x] Rerun loader and dry-run tests and confirm they pass.

## Task 9: Document real-run readiness and execute verification

**Files:**
- Create: `docs/FIVE_ARM_MEMORY_EVAL.md`
- Modify: `README.md`

- [x] Document the causal claim each comparison supports, the ledger/FAM trust boundary, data normalization contract, seal/run/score workflow, metrics, and the distinction between dry-run plumbing and benchmark evidence.
- [x] Add concise README links and commands without changing PR13 instructions.
- [x] Run all `tests/test_memory_eval_*.py` tests.
- [x] Run the relevant existing context/compiler/output-contract tests.
- [x] Run the dry-run command and inspect its JSON summary.
- [x] Run the broader test suite; record the known clean-worktree fixture failure separately if it remains the only failure.
- [x] Inspect `git diff --check`, `git status --short`, and the complete diff to verify that no sealed PR13 file or user artifact changed.

