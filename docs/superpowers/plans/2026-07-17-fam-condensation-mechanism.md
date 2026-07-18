# FAM Condensation Mechanism — Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the inert scope-centroid arm with genuine online FAM condensation, add a matched allocate-only CAM control, and make the FAM mechanism claim gate the constructive-forgetting application claim under a sealed synthetic experiment.

**Architecture:** Five consumer arms remain, but `exemplar_raw`/`exemplar_governed` replace the old exact-vector arms. E0 and F0 share CAM capacity, ordering, query, provenance expansion, reranking, and rendering; only `write_mode` differs. A new mechanism scorer evaluates prototype reduction and authoritative-record recall before the existing FAM governed-vs-raw application gates, and the manifest seals every live treatment field plus deterministic index attestations.

**Tech Stack:** Python 3.11+, PyTorch, dataclasses, pytest, canonical JSON/SHA-256, existing `ContinuousCAM` and `harness.memory_eval` modules.

## Global Constraints

- Work only in `/private/tmp/fast-associative-memory-five-arm` on `codex/five-arm-memory-eval`.
- Preserve authoritative immutable ledger records; CAM values never become answer payloads.
- Execute exactly `no_memory`, `exemplar_raw`, `exemplar_governed`, `fam_raw`, and `fam_governed` as consumer arms.
- Only E0 (`exemplar_raw`), F0 (`fam_raw`), and F1 (`fam_governed`) are verdict-bearing.
- E0 and F0 use one record per `learn_local` call in manifest record order with `max_entries == record_count`.
- F0 uses `vigilance=0.85`, `hebb_lr=0.1`, `key_lr=0.05`, `ema_beta=0.05`, mutable float32 keys, static vigilance, no sleep, no NSTP, and no retrieval-floor/truncation policy.
- E0 uses the same CAM/query envelope with explicit `write_mode="allocate-only"`; never emulate it with an out-of-range vigilance.
- Any dropped/evicted write or incomplete/duplicated/cross-scope provenance blocks the run.
- Exact-vector retrieval remains consumer-free and exploratory.
- Raw rendering is deterministic skip-not-stop under the fixed 1,500-token context budget.
- Mechanism verdict precedes application verdict; a failed mechanism makes all application outcomes exploratory.
- Numeric prototype-reduction, recall-loss, and application bounds remain mandatory preregistration fields with no code defaults.
- Do not modify `harness/ctx/`, sealed PR-13 manifests, or existing result artifacts.
- Use TDD: observe each named test fail for the intended missing behavior before implementing it.

---

### Task 1: Add an explicit allocate-only CAM write mode and eviction accounting

**Files:**
- Modify: `associative_core.py:45-920`
- Modify: `tests/test_write_accounting.py`
- Modify: `tests/test_slot_records.py`

**Interfaces:**
- Consumes: existing `ContinuousCAM.learn_local(queries, targets, record_ids=None)`.
- Produces: `ContinuousCAM.learn_local(..., *, write_mode: Literal["condense", "allocate-only"] = "condense")` and `last_write_stats["evicted"]`.

- [ ] **Step 1: Write failing core tests for allocate-only and eviction accounting**

Add tests with these exact behaviors:

```python
def test_allocate_only_keeps_identical_same_class_writes_as_exemplars():
    cam = ContinuousCAM(
        key_dim=2, value_dim=1, max_entries=2,
        vigilance=0.0, immutable_keys=False,
        adaptive_eviction=False, use_lfu=True,
        track_provenance=True,
    )
    key = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([[1.0]])
    cam.learn_local(key, target, record_ids=["a"], write_mode="allocate-only")
    cam.learn_local(key, target, record_ids=["b"], write_mode="allocate-only")
    assert int(cam.occupied.sum()) == 2
    assert cam.last_write_stats == {
        "written": 1, "merged": 0, "allocated": 1,
        "dropped": 0, "evicted": 0,
    }


def test_write_accounting_reports_slot_reuse_as_eviction():
    cam = ContinuousCAM(
        key_dim=2, value_dim=2, max_entries=1,
        adaptive_eviction=False, use_lfu=False,
        track_provenance=True,
    )
    cam.learn_local(torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]), record_ids=["old"])
    cam.learn_local(
        torch.tensor([[0.0, 1.0]]), torch.tensor([[0.0, 1.0]]),
        record_ids=["new"], write_mode="allocate-only",
    )
    assert cam.last_write_stats["evicted"] == 1
    assert cam.records_for(0) == {"new"}


def test_learn_local_rejects_unknown_write_mode():
    cam = ContinuousCAM(key_dim=2, value_dim=1)
    with pytest.raises(ValueError, match="write_mode"):
        cam.learn_local(torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0]]), write_mode="merge-ish")
```

Update existing exact-dict assertions for `last_write_stats` to include `"evicted": 0`.

- [ ] **Step 2: Run the focused tests and verify the intended failures**

Run:

```bash
pytest -q tests/test_write_accounting.py tests/test_slot_records.py
```

Expected: failures report the unexpected `write_mode` keyword and missing `evicted` field.

- [ ] **Step 3: Implement the minimal core behavior**

In `associative_core.py`:

```python
from typing import Literal

WriteMode = Literal["condense", "allocate-only"]
```

Extend `learn_local` with a keyword-only `write_mode`. Validate it before any mutation. After the normal vigilance and same-class calculation, force `hits` to all-false only for allocate-only mode:

```python
if write_mode not in {"condense", "allocate-only"}:
    raise ValueError(f"unsupported write_mode: {write_mode!r}")

if write_mode == "allocate-only":
    hits = torch.zeros_like(hits)
```

Immediately before `_alloc_slots_batch`, capture free capacity and derive reused victims without relying on occupancy delta:

```python
free_before = int((~self.occupied).sum().item())
new_slots = self._alloc_slots_batch(n_miss)
n_alloc = len(new_slots)
n_evicted = max(0, n_alloc - min(n_miss, free_before))
```

Initialize and emit the five-field accounting dict everywhere:

```python
{"written": 0, "merged": 0, "allocated": 0, "dropped": 0, "evicted": 0}
```

- [ ] **Step 4: Run core regression tests**

Run:

```bash
pytest -q tests/test_write_accounting.py tests/test_slot_records.py tests/test_read_path_invariants.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add associative_core.py tests/test_write_accounting.py tests/test_slot_records.py
git commit -m "feat: add allocate-only CAM writes"
```

---

### Task 2: Build matched exemplar and live FAM retrievers with attestations

**Files:**
- Modify: `harness/memory_eval/retrievers.py`
- Rewrite focused expectations: `tests/test_memory_eval_retrievers.py`

**Interfaces:**
- Consumes: Task 1 `write_mode` and five-field `last_write_stats`.
- Produces:
  - `CAMIndexSettings`
  - `IndexBuildAttestation`
  - `ExemplarCAMRetriever(records, embeddings, *, settings)`
  - `FAMRetriever(records, embeddings, *, settings)`
  - shared `.query(...)`, `.prototype_count`, `.attestation`, and `.provenance_for_scope(...)`.

- [ ] **Step 1: Write failing retriever tests**

Define a helper with explicit settings:

```python
def settings(max_entries=4, prototype_k=2):
    return CAMIndexSettings(
        max_entries=max_entries,
        prototype_k=prototype_k,
        vigilance=0.85,
        hebb_lr=0.1,
        key_lr=0.05,
        ema_beta=0.05,
        inference_temp=0.05,
        use_bfloat16=False,
        adaptive_eviction=False,
        use_lfu=True,
    )
```

Add tests proving:

```python
def test_live_fam_streams_actual_embeddings_and_merges_with_key_drift():
    records = [record("old", "scope", serial=1), record("new", "scope", serial=2)]
    embeddings = {"old": [1.0, 0.0], "new": [0.9, 0.1]}
    retriever = FAMRetriever(records, embeddings, settings=settings(max_entries=2))
    assert retriever.attestation.written == 2
    assert retriever.attestation.allocated == 1
    assert retriever.attestation.merged == 1
    assert retriever.attestation.key_drifted_merges == 1
    assert retriever.prototype_count == 1
    assert retriever.provenance_for_scope("scope") == {"old", "new"}


def test_below_vigilance_same_scope_allocates_second_fam_prototype():
    records = [record("a", "scope"), record("b", "scope", serial=1)]
    embeddings = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    retriever = FAMRetriever(records, embeddings, settings=settings(max_entries=2))
    assert retriever.attestation.merged == 0
    assert retriever.prototype_count == 2


def test_allocate_only_control_matches_envelope_but_never_merges():
    records = [record("a", "scope"), record("b", "scope", serial=1)]
    embeddings = {"a": [1.0, 0.0], "b": [0.9, 0.1]}
    retriever = ExemplarCAMRetriever(records, embeddings, settings=settings(max_entries=2))
    assert retriever.attestation.merged == 0
    assert retriever.attestation.allocated == 2
    assert retriever.prototype_count == 2


def test_cam_retriever_rejects_capacity_that_could_evict():
    records = [record("a", "s1"), record("b", "s2")]
    embeddings = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    with pytest.raises(ValueError, match="max_entries.*record count"):
        ExemplarCAMRetriever(records, embeddings, settings=settings(max_entries=1))
```

Retain the authoritative-record rerank assertion and add deterministic index-hash equality for two identical builds and inequality after record reordering.

- [ ] **Step 2: Run the focused retriever tests and verify failure**

```bash
pytest -q tests/test_memory_eval_retrievers.py
```

Expected: imports for `CAMIndexSettings` and `ExemplarCAMRetriever` fail.

- [ ] **Step 3: Implement settings and attestation dataclasses**

Use frozen, slotted dataclasses:

```python
@dataclass(frozen=True, slots=True)
class CAMIndexSettings:
    max_entries: int
    prototype_k: int
    vigilance: float
    hebb_lr: float
    key_lr: float
    ema_beta: float
    inference_temp: float
    use_bfloat16: bool
    adaptive_eviction: bool
    use_lfu: bool


@dataclass(frozen=True, slots=True)
class IndexBuildAttestation:
    mode: Literal["allocate-only", "condense"]
    written: int
    merged: int
    allocated: int
    dropped: int
    evicted: int
    prototype_count: int
    key_drifted_merges: int
    index_sha256: str
```

Validate positive integer capacity/width, finite numeric fields, `0 <= vigilance <= 1`, and positive temperature. Require `max_entries == len(records)` in both public constructors.

- [ ] **Step 4: Implement one shared sequential CAM build/query path**

Factor the current FAM query behavior into a private `_CAMRecordRetriever` taking `mode`. Build a `ContinuousCAM` with actual settings, `immutable_keys=(mode == "allocate-only")`, `track_provenance=True`, static vigilance, and no optional policies. Iterate `self.records` without sorting and call `learn_local` once per record using its actual embedding and scope one-hot.

Before and after each condense write, clone occupied keys; when the write reports a merge and any occupied key changes, increment `key_drifted_merges`. Accumulate `last_write_stats` immediately after each call.

After the build, reject dropped/evicted writes and validate that occupied provenance sets are nonempty, disjoint, scope-consistent, and union exactly to the input record IDs. Hash canonical JSON rows containing occupied slot number, float32 key values, semantic label, and sorted provenance IDs.

Keep prototype-trace → provenance union → exact authoritative-record embedding rerank byte-equivalent between E0 and F0.

- [ ] **Step 5: Run retriever and core tests**

```bash
pytest -q tests/test_memory_eval_retrievers.py tests/test_write_accounting.py tests/test_slot_records.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add harness/memory_eval/retrievers.py tests/test_memory_eval_retrievers.py
git commit -m "feat: execute live FAM condensation"
```

---

### Task 3: Replace vector arms with exemplar arms and symmetrize raw rendering

**Files:**
- Modify: `harness/memory_eval/__init__.py`
- Modify: `harness/memory_eval/context.py`
- Modify: `harness/memory_eval/runner.py`
- Modify: `harness/memory_eval/dry_run.py`
- Modify: `harness/memory_eval/scoring.py`
- Modify tests: `tests/test_memory_eval_context.py`, `tests/test_memory_eval_dry_run.py`, `tests/test_memory_eval_runner.py`, `tests/test_memory_eval_scoring.py`

**Interfaces:**
- Consumes: Task 2 retrievers.
- Produces: five renamed arms and paired candidate reuse for `exemplar_*` and `fam_*`.

- [ ] **Step 1: Write failing arm and renderer tests**

Change expected arm names to:

```python
(
    "no_memory",
    "exemplar_raw",
    "exemplar_governed",
    "fam_raw",
    "fam_governed",
)
```

Add a raw-render regression where item 2 is oversized but item 3 fits:

```python
def test_render_raw_skips_oversized_candidate_and_continues():
    items = [
        {"item_id": "a", "content": "short"},
        {"item_id": "b", "content": "far too long for this tiny budget now"},
        {"item_id": "c", "content": "ok"},
    ]
    rendered = render_raw(items, budget=7, count_tokens=lambda text: len(text.split()))
    assert rendered.rendered_item_ids == ("a", "c")
```

Update runner pairing tests to require identical candidate IDs within the exemplar and FAM families and to permit differences between families.

- [ ] **Step 2: Run the selected tests and verify old names/break behavior fail**

```bash
pytest -q tests/test_memory_eval_context.py tests/test_memory_eval_runner.py tests/test_memory_eval_scoring.py tests/test_memory_eval_dry_run.py
```

Expected: arm-name assertions and skip-not-stop test fail.

- [ ] **Step 3: Rename and rewire the runner**

Replace `vector_*` with `exemplar_*` in `ARM_NAMES`, `_PAIRED_FAMILIES`, metrics keys, fixtures, dry run, and docs strings. Rename runner constructor field `vector_retriever` to `exemplar_retriever` and build the candidates mapping with `"exemplar"` and `"fam"`.

Do not add exact-vector as a consumer arm. Retain `ExactVectorRetriever` for later exploratory retrieval scoring.

- [ ] **Step 4: Implement skip-not-stop raw rendering**

In `render_raw`, replace `break` with `continue` when the proposed block exceeds budget. Preserve original ordering and recompute the proposal against only accepted lines.

- [ ] **Step 5: Run the four test modules**

```bash
pytest -q tests/test_memory_eval_context.py tests/test_memory_eval_runner.py tests/test_memory_eval_scoring.py tests/test_memory_eval_dry_run.py
```

Expected: all selected tests pass with exactly five renamed arms.

- [ ] **Step 6: Commit Task 3**

```bash
git add harness/memory_eval/__init__.py harness/memory_eval/context.py harness/memory_eval/runner.py harness/memory_eval/dry_run.py harness/memory_eval/scoring.py tests/test_memory_eval_context.py tests/test_memory_eval_runner.py tests/test_memory_eval_scoring.py tests/test_memory_eval_dry_run.py
git commit -m "feat: add matched exemplar experiment arms"
```

---

### Task 4: Add mechanism scoring and fixed-sequence preregistration

**Files:**
- Create: `harness/memory_eval/mechanism.py`
- Modify: `harness/memory_eval/preregistration.py`
- Modify: `docs/FIVE_ARM_PREREGISTRATION.md`
- Create: `tests/test_memory_eval_mechanism.py`
- Modify: `tests/test_memory_eval_preregistration.py`

**Interfaces:**
- Consumes: E0/F0 rows, ledger, Task 2 attestations, existing `Rate` and normalization.
- Produces:
  - `MechanismReport`
  - `score_mechanism(...)`
  - `mechanism_passes(...)`
  - fixed-sequence `experiment_verdict(...)`.

- [ ] **Step 1: Write failing authoritative-recall and mechanism-gate tests**

Create tests covering:

```python
def test_authoritative_recall_uses_latest_matching_record_and_excludes_contested():
    # evolving scope: old serial 1, current serial 2; candidate must contain serial 2
    # contested max-serial scope is omitted from the mechanism denominator


def test_mechanism_report_pairs_exemplar_and_fam_on_one_denominator():
    # E0 hits q1/q2, F0 hits q1 only => recall_loss_count=1, n=2
    # E0 prototypes=4, F0 prototypes=3 => reduction_count=1, record_n=4


def test_mechanism_gate_uses_exact_integer_thresholds():
    assert mechanism_passes(
        reduction_count=2, record_n=10, reduction_margin=0.2,
        recall_loss_count=1, recall_n=10, recall_loss_bound=0.1,
    )
    assert not mechanism_passes(
        reduction_count=1, record_n=10, reduction_margin=0.2,
        recall_loss_count=1, recall_n=10, recall_loss_bound=0.1,
    )


def test_application_is_exploratory_when_mechanism_fails():
    assert experiment_verdict(
        integrity_ok=True, evaluable=True, mechanism_ok=False,
        application_h1=True, application_h2=True, application_h3=True,
    ) == "NO-GO — FAM mechanism"
```

Also test empty mechanism denominator, duplicate/missing arm rows, candidate pairing, and a live path with zero merges returning `not-evaluable` rather than GO.

- [ ] **Step 2: Run the new tests and verify missing module/functions**

```bash
pytest -q tests/test_memory_eval_mechanism.py tests/test_memory_eval_preregistration.py
```

Expected: import failures for `harness.memory_eval.mechanism` and new registration fields.

- [ ] **Step 3: Implement `MechanismReport` and scorer**

Use exact counts as the canonical representation:

```python
@dataclass(frozen=True, slots=True)
class MechanismReport:
    recall_n: int
    exemplar_recall_count: int
    fam_recall_count: int
    recall_loss_count: int
    record_n: int
    exemplar_prototype_count: int
    fam_prototype_count: int
    prototype_reduction_count: int
```

For every non-contested question, compute maximum serial, filter records at that serial whose normalized value equals the annotated answer, and require at least one authoritative ID. Read E0/F0 candidate tuples from exactly one `exemplar_raw` and `fam_raw` row per question. Define:

```python
recall_loss_count = exemplar_recall_count - fam_recall_count
prototype_reduction_count = exemplar_prototype_count - fam_prototype_count
```

Gate with:

```python
prototype_reduction_count >= ceil(reduction_margin * record_n)
recall_loss_count <= floor(recall_loss_bound * recall_n)
```

- [ ] **Step 4: Extend registration as a closed schema**

Add required fields:

```python
"prototype_reduction_margin"  # D-M1, float in [0, 1]
"mechanism_recall_loss_bound" # D-M2, float in [0, 1]
"min_mechanism_recall_n"      # D-M3, positive int
"claim_order"                 # literal "fam-mechanism-then-application"
```

Restrict `primary_family` to literal `"fam"`; retain exact scorer, fixed-full H1 denominator, raw-with-invariant equivalence, and conditional contested fields. Add derived integer threshold fields only at seal construction, not as human choices.

Implement a total experiment verdict: integrity failure → `blocked`; missing denominators or no realized merge/key drift → `not-evaluable`; M1/M2 failure → `NO-GO — FAM mechanism`; only then delegate to the existing application H1/H2/H3 verdict.

- [ ] **Step 5: Revise the preregistration memo to revision 3**

Make the design hierarchy, renamed arms, D-M1/D-M2/D-M3 fields, treatment-fidelity gates, fixed-sequence verdict table, exact scorer, fixed-full denominator, raw invariant, and exploratory nonclaims match the committed design spec. Keep numeric choices as keyed sentinels and ensure the memo-heading/schema bijection test includes them.

- [ ] **Step 6: Run mechanism and preregistration tests**

```bash
pytest -q tests/test_memory_eval_mechanism.py tests/test_memory_eval_preregistration.py tests/test_memory_eval_scoring.py
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 4**

```bash
git add harness/memory_eval/mechanism.py harness/memory_eval/preregistration.py docs/FIVE_ARM_PREREGISTRATION.md tests/test_memory_eval_mechanism.py tests/test_memory_eval_preregistration.py
git commit -m "feat: gate application evidence on FAM mechanism"
```

---

### Task 5: Seal the complete CAM treatment and realized indexes

**Files:**
- Modify: `harness/memory_eval/manifest.py`
- Modify: `harness/memory_eval/sealed_run.py`
- Modify: `tests/test_memory_eval_manifest.py`
- Modify: `tests/test_memory_eval_sealed_run.py`

**Interfaces:**
- Consumes: Task 2 settings/attestations and Task 4 registration.
- Produces: manifest v3 closed retriever schema, sealed E0/F0 attestations, preflight rebuild/compare, runner built exclusively from sealed settings.

Use a dedicated three-record manifest fixture in this task so the registered
`cam_max_entries` value below equals the fixture's record count exactly.

- [ ] **Step 1: Write failing strict-schema and attestation tests**

Define one complete settings fixture with exactly these keys:

```python
SETTINGS = {
    "candidate_k": 2,
    "cam_max_entries": 3,
    "cam_prototype_k": 1,
    "cam_vigilance": 0.85,
    "cam_hebb_lr": 0.1,
    "cam_key_lr": 0.05,
    "cam_ema_beta": 0.05,
    "cam_inference_temp": 0.05,
    "cam_use_bfloat16": False,
    "cam_adaptive_eviction": False,
    "cam_use_lfu": True,
    "cam_dynamic_vigilance": None,
    "cam_retrieval_floor": None,
    "cam_retrieval_truncation": None,
    "cam_nstp": None,
    "cam_sleep": False,
    "cam_ingest_order": "manifest-record-order",
    "exemplar_write_mode": "allocate-only",
    "fam_write_mode": "condense",
}
```

Add tests that independently reject a missing key, unknown key, wrong type, `cam_max_entries != record_count`, changed FAM numeric setting, reordered inputs, tampered E0 attestation, tampered F0 index hash, and a changed memo SHA. Add an exploding consumer test proving every preflight failure occurs before generation.

- [ ] **Step 2: Run manifest/sealed-run tests and verify failures**

```bash
pytest -q tests/test_memory_eval_manifest.py tests/test_memory_eval_sealed_run.py
```

Expected: strict-schema and attestation assertions fail against manifest v2.

- [ ] **Step 3: Implement the closed treatment schema and manifest v3**

Set `MANIFEST_VERSION = "memory-eval-manifest-v3"`. Add a validator that compares the settings key set exactly, validates booleans without accepting integers, validates finite numeric values/ranges, requires every optional feature to be explicit `None`/`False`, requires fixed write modes/order, and requires `cam_max_entries == len(records)`.

Before sealing, build E0 and F0 from the records/embeddings and serialize both attestations into:

```python
"index_attestations": {
    "exemplar": asdict(exemplar.attestation),
    "fam": asdict(fam.attestation),
}
```

For a scoring-run seal, require complete mechanism/application registration and a real memo SHA matching the bytes at the registered memo path. Plumbing seals may carry settings but no registration or confirmatory index attestation.

- [ ] **Step 4: Make preflight rebuild and compare both indexes**

In `preflight`, validate the schema, rebuild E0/F0 from sealed settings, and compare every realized attestation field. Emit separate checks for treatment settings, exemplar index, FAM index, mechanism activity (`merged > 0` and `key_drifted_merges > 0`), and provenance/capacity integrity.

In `build_sealed_run`, construct `CAMIndexSettings` only from the sealed protocol, pass the rebuilt `ExemplarCAMRetriever` and `FAMRetriever` into `FiveArmRunner`, and never use caller defaults.

- [ ] **Step 5: Run manifest, sealed-run, and retriever tests**

```bash
pytest -q tests/test_memory_eval_manifest.py tests/test_memory_eval_sealed_run.py tests/test_memory_eval_retrievers.py
```

Expected: all selected tests pass and tamper cases identify their named gates.

- [ ] **Step 6: Commit Task 5**

```bash
git add harness/memory_eval/manifest.py harness/memory_eval/sealed_run.py tests/test_memory_eval_manifest.py tests/test_memory_eval_sealed_run.py
git commit -m "feat: seal live CAM treatment and indexes"
```

---

### Task 6: Complete synthetic end-to-end verification and documentation

**Files:**
- Modify: `harness/memory_eval/dry_run.py`
- Modify: `docs/FIVE_ARM_MEMORY_EVAL.md`
- Modify: `README.md`
- Modify: `tests/test_memory_eval_dry_run.py`
- Modify: `tests/test_memory_eval_fact_consolidation.py` only if renamed arms affect fixtures

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: deterministic sealed synthetic proof that the mechanism-first/application-second workflow executes; updated honest documentation.

- [ ] **Step 1: Write a failing end-to-end test with both mechanism and application outcomes**

Construct a synthetic corpus where:

- two same-scope embeddings exceed vigilance and therefore merge in F0;
- E0 allocates one prototype per record;
- both retrievers recover the current record at registered `candidate_k`;
- F0 raw adopts a stale value under the deterministic consumer;
- F1 governed answers the current value;
- at least one clean scope remains correct.

Assert the returned artifact contains five arms, both attestations, a passing mechanism verdict, and a passing application verdict, while labeling itself synthetic/plumbing rather than benchmark evidence.

- [ ] **Step 2: Run the end-to-end test and verify it fails before fixture updates**

```bash
pytest -q tests/test_memory_eval_dry_run.py
```

Expected: old arm/settings/result-shape assertions fail.

- [ ] **Step 3: Update the deterministic dry run and docs**

Build both CAM retrievers with capacity equal to synthetic record count and explicit settings. Seal/rebuild through manifest v3. Score mechanism before application. Do not promote synthetic numbers into README performance claims.

Update `FIVE_ARM_MEMORY_EVAL.md` and the README section to state:

- FAM condensation is the primary mechanism;
- constructive forgetting is the secondary application;
- exact vector is an exploratory ceiling;
- the official `fact_sh` transformer, G-I6 reconciliation, pinned semantic encoder/consumer artifacts, numeric registration, and real scoring-host execution remain Phase B blockers.

- [ ] **Step 4: Run the complete targeted suite**

```bash
pytest -q tests/test_memory_eval_*.py tests/test_write_accounting.py tests/test_slot_records.py tests/test_read_path_invariants.py
```

Expected: all selected tests pass with no deselection.

- [ ] **Step 5: Run repository hygiene checks**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended Phase A files differ from the task baseline.

- [ ] **Step 6: Commit Task 6**

```bash
git add harness/memory_eval/dry_run.py docs/FIVE_ARM_MEMORY_EVAL.md README.md tests/test_memory_eval_dry_run.py tests/test_memory_eval_fact_consolidation.py
git commit -m "docs: document FAM-first memory experiment"
```

## Compatibility amendment — Task 7

Full-repository verification after Tasks 1-6 established that
`associative_core.py` is a byte-frozen deployed-engine contract. This amendment
supersedes Task 1's prescription to add `write_mode` and eviction accounting to
`ContinuousCAM.learn_local`, and Task 2's prescription to consume that extended
core API. The earlier RED/GREEN chronology above is retained as the historical
implementation record; its core changes were subsequently reverted to the
deployed bytes at commit `60abd33`.

The approved treatment semantics now live entirely in a private
`harness.memory_eval.retrievers` adapter. It accepts exactly one record and a
validated `write_mode` in `{"allocate-only", "condense"}`. During static-
vigilance ingest only, it overrides nearest selection: `condense` searches only
occupied prototypes with the incoming scope label, and `allocate-only` forces
the ordinary allocation path. It delegates all mutation to the unchanged core,
inherits `forward(..., trace=True)` without override, and computes eviction
attestation locally as `allocated - max(0, occupied_after - occupied_before)`.
The core's four-field `last_write_stats` contract and all deployed query
semantics remain unchanged.

## Phase B boundary

Phase A ends with a sealed, deterministic synthetic experiment and no benchmark claim. A separate implementation plan must cover the official parquet transformer and reconciliation (G-I6), complete encoder/consumer/code artifact rollups, outcome-incapable real-corpus shape probing, human numeric registration, scoring-host preflight, and the single authorized real execution. The authoritative dataset checked during planning has eight Conflict Resolution rows; `factconsolidation_sh_{6k,32k,64k,262k}` each contains one numbered fact context plus 100 questions, answer lists, and `qa_pair_ids`. No Phase A code may guess entity/relation scopes from those natural-language facts.
