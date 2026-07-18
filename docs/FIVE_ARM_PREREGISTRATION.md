# Five-arm FAM evaluation — preregistration (revision 3, DRAFT)

**Status:** Phase A draft; confirmatory execution is disabled. Every keyed
marker below remains a human Phase B decision. Even a completed memo, policy,
consumer pin, and digest-consistent manifest are insufficient today because the
Phase B provenance/reconciliation envelope is not implemented. Corpus-shape
probes may disclose denominators, but never retrieval differences, prototype
reduction, or application outcomes.

## 1. Design hierarchy and claims

The fixed arms are:

| Arm | Retrieval | Rendering | Confirmatory role |
|---|---|---|---|
| `no_memory` | none | none | negative control only |
| `exemplar_raw` (E0) | matched allocate-only CAM | deterministic raw skip-not-stop | primary mechanism control |
| `exemplar_governed` (E1) | same E0 candidate tuple | governed compiler | diagnostic control |
| `fam_raw` (F0) | live online FAM condensation | same raw renderer as E0 | primary mechanism treatment and application control |
| `fam_governed` (F1) | same F0 candidate tuple | governed compiler | application treatment |

The confirmatory hierarchy is fixed. First, E0 versus F0 must demonstrate an
active, fidelity-preserving FAM mechanism. Only after that conjunction passes
may F0 versus F1 support the constructive-forgetting application claim. E1,
cross-family answer comparisons, latency, token use, and all contested-question
outcomes are exploratory. Phase A accepts no confirmatory contested gate.

The primary scorer is exact normalized structured-answer equality. H1 uses the
fixed-full stale-eligible denominator. Ledger lifecycle identity remains raw
string equality, protected by an invariant rejecting any raw-unequal values
that normalize equal at the same scope and serial.

The closed v1 treatment fixes vigilance `0.85`, Hebbian LR `0.1`, key LR
`0.05`, EMA beta `0.05`, inference temperature `0.05`, float32
(`use_bfloat16 = false`), adaptive eviction off, and LFU on. It also requires
explicitly disabled dynamic vigilance, retrieval policies, NSTP, and sleep,
plus the fixed ingest/write modes. Only the two retrieval widths remain human
choices (D-M4/D-M5).

## 2. Decisions requiring registration

Numeric registrations have no code defaults. The displayed values are keyed
sentinels, not recommendations. Fixed design literals are also sealed fields so
schema drift cannot silently change the claim.

### D-M1 — Prototype-reduction margin

`prototype_reduction_margin = <<UNREGISTERED:D-M1>>`, a number in `[0, 1]`.
M1 compares the exact prototype reduction with `ceil(margin * record_n)`.

### D-M2 — Mechanism recall-loss bound

`mechanism_recall_loss_bound = <<UNREGISTERED:D-M2>>`, a number in `[0, 1]`.
M2 compares paired E0-minus-F0 authoritative recall loss with
`floor(bound * recall_n)`.

### D-M3 — Minimum mechanism recall denominator

`min_mechanism_recall_n = <<UNREGISTERED:D-M3>>`, a positive integer.
Contested maximum-serial scopes never enter this denominator.

### D-M4 — Candidate retrieval width

`candidate_k = <<UNREGISTERED:D-M4>>`, a positive integer selected by the human
registrant. Phase A synthetic fixture values are not a proposed real-run width.

### D-M5 — CAM prototype retrieval width

`cam_prototype_k = <<UNREGISTERED:D-M5>>`, a positive integer selected by the
human registrant. Phase A synthetic fixture values are not a proposed real-run
width.

### D-1 — Stale-reduction margin

`stale_reduction_margin = <<UNREGISTERED:D-1>>`, a number in `[0, 1]`.
A1 uses the exact F0-minus-F1 stale-adoption count on the fixed-full denominator.

### D-2 — Clean-answer-loss bound

`clean_answer_loss_bound = <<UNREGISTERED:D-2>>`, a number in `[0, 1]`.
A2 limits paired clean rows where F0 is correct and F1 is not.

### D-3 — Current-adoption floor

`current_adoption_floor = <<UNREGISTERED:D-3>>`, a number in `[0, 1]`.
A3 requires current-answer successes on stale-eligible questions.

### D-3b — Additional abstention bound

`abstention_bound = <<UNREGISTERED:D-3b>>`. Phase A accepts only null, explicitly
recording that no additional abstention gate is operational. A numeric bound is
a Phase B evaluator feature, not a Phase A registration choice.

### D-4 — Scorer semantics

`scorer = <<UNREGISTERED:D-4>>`. The only confirmatory literal is `exact`.
Containment, split scoring, and answer hygiene variants are exploratory.

### D-5 — Raw-arm truncation semantics

`raw_truncation = <<UNREGISTERED:D-5>>`. The only confirmatory literal is
`skip`: an over-budget record is skipped and later records are still considered.
E0 and F0 use the identical deterministic renderer and budget.

### D-6 — Contested-question disposition

`contested_disposition = <<UNREGISTERED:D-6>>`. Phase A accepts only
`exploratory`; contested counts and outcomes are reported but read by no value
gate. `gated`, `contested_rule`, and `contested_bound` remain Phase B evaluator
work and are rejected by the Phase A closed schema.

### D-7 — Equivalence relation (ledger vs scorer)

`equivalence = <<UNREGISTERED:D-7>>`. The only confirmatory literal is
`raw-with-invariant`: lifecycle forks use raw ledger equality, while answer
scoring uses normalization, and the seal forbids their ambiguous overlap.

### D-8a — Minimum stale-eligible denominator

`min_stale_eligible_n = <<UNREGISTERED:D-8a>>`, a positive integer.

### D-8b — Minimum clean denominator

`min_clean_n = <<UNREGISTERED:D-8b>>`, a positive integer.

### D-9 — H1 denominator policy

`h1_denominator = <<UNREGISTERED:D-9>>`. The only confirmatory literal is
`fixed-full`; every stale-eligible question stays in the application denominator,
including malformed output as non-adoption.

### D-10 — Primary family

`primary_family = <<UNREGISTERED:D-10>>`. The only confirmatory literal is
`fam`. The exemplar family is a mechanism control, not a second application
claim family.

### D-11 — Fixed claim order

`claim_order = <<UNREGISTERED:D-11>>`. The only accepted literal is
`fam-mechanism-then-application`.

## 3. Exact mechanism scorer

For each non-contested question, find the maximum ledger serial and then the
records at that serial whose normalized value equals the annotated answer. At
least one such authoritative record ID must exist. Recall@`candidate_k` is one
when the raw arm's candidate-ID tuple contains any authoritative ID. CAM values
and generated answer payloads are never inputs to mechanism recall.

Exactly one `exemplar_raw` and one `fam_raw` row must exist per query. Duplicate
or missing rows, duplicate candidate IDs, candidate IDs absent from the ledger,
or a non-contested annotation without an authoritative latest ID invalidate
scoring. A contested maximum serial is omitted from both E0 and F0 recall using
one shared denominator.

Canonical counts are:

```text
recall_loss_count          = exemplar_recall_count - fam_recall_count
prototype_reduction_count = exemplar_prototype_count - fam_prototype_count
M1 threshold              = ceil(D-M1 * record_n)
M2 threshold              = floor(D-M2 * recall_n)
```

Threshold multiplication uses exact rational arithmetic recovered from the
registered decimal text. It never multiplies binary floating-point values.
The two integer thresholds are not human registration choices. Because Phase A
cannot construct a scoring seal, deriving and recording confirmatory integer
thresholds is explicitly a Phase B sealing obligation.

## 4. Integrity, activity, and evaluability gates

Phase A preflight calls full manifest verification before using any protocol
field. Digest-consistent changes to arms, budget, protocol identities, counts,
or unknown envelope fields fail its named full-manifest-binding check. Any
failure yields `blocked` and no value claim. Additional checks include closed
registration schema and memo hash, immutable corpus and embeddings, the exact
v1 treatment settings, successful rebuild of E0/F0, exact realized index
attestations, no dropped/evicted writes, complete one-to-one provenance over
ledger record IDs, capacity equality, raw renderer identity, and F0/F1
candidate-tuple identity. D-M4/D-M5 must equal the verified treatment widths.
The missing Phase B provenance/reconciliation
envelope always remains a failed confirmatory check in Phase A.

FAM mechanism activity requires both `fam_attestation.merged > 0` and
`fam_attestation.key_drifted_merges > 0`. Prototype occupancy alone cannot
establish activity. Zero realized merges, zero key-drifted merges, an empty
mechanism denominator, `recall_n < D-M3`, or missing application denominators
yields `not-evaluable`, never GO.

## 5. Fixed-sequence gates and verdict

The following fixed sequence is the Phase B verdict design and remains
available only through pure diagnostic helpers. In Phase A the authoritative
API unconditionally returns `blocked`: `PreflightReceipt` is publicly
constructible, so even caller-asserted `passed = true`, `confirmatory = true`,
and `evidence_class = "scoring-run"` fields grant no authority. After Phase B
implements a trusted issuer, integrity, and evaluability:

1. M1 requires `prototype_reduction_count >= ceil(D-M1 * record_n)`.
2. M2 requires `recall_loss_count <= floor(D-M2 * recall_n)`.
3. Only if M1 and M2 pass, A1/A2/A3 evaluate F0 versus F1 using the existing
   exact integer application gates.

| First matching condition | Experiment verdict | Application outputs |
|---|---|---|
| any integrity failure | `blocked` | non-evidence |
| inactive FAM or any required denominator absent/too small | `not-evaluable` | exploratory |
| M1 or M2 fails | `NO-GO — FAM mechanism` | exploratory regardless of values |
| A1 fails | `NO-GO — no effect` | confirmatory diagnostic |
| A1 passes, A3 fails | `NO-GO — suppression` | confirmatory diagnostic |
| A1 and A3 pass, A2 fails | `NO-GO — collateral` | confirmatory diagnostic |
| M1, M2, A1, A2, and A3 pass | `governed-memory-GO` | confirmatory |

The ordering is semantic, not presentational: a failed mechanism cannot be
rescued by favorable application outcomes.

## 6. Closed registration schema

A complete human registration contains exactly these decision fields plus the
derived `memo_sha256`:

```text
prototype_reduction_margin, mechanism_recall_loss_bound,
min_mechanism_recall_n, candidate_k, cam_prototype_k, stale_reduction_margin,
clean_answer_loss_bound, current_adoption_floor, abstention_bound,
scorer, raw_truncation, contested_disposition, equivalence,
min_stale_eligible_n, min_clean_n, h1_denominator, primary_family,
claim_order, memo_sha256
```

Phase A accepts `abstention_bound = null` and
`contested_disposition = "exploratory"` only. `contested_rule` and
`contested_bound` are unknown fields. Unknown, missing, wrongly typed,
out-of-range, or still-sentinel fields are rejected. No numeric bound, minimum,
or retrieval width has a default.

## 7. Exploratory nonclaims

No confirmatory inference is made from E1, no-memory accuracy, contested
questions, cross-family generated-answer deltas,
latency, token cost, prototype identities, per-scope examples, effect sizes
viewed before registration, or any application result when the mechanism is
blocked, inactive, under-denominated, or fails M1/M2. Synthetic and dry-run
fixtures prove plumbing only and are never benchmark evidence.

## 8. Registration and execution sequence

Phase A may freeze and verify plumbing inputs, run the outcome-blind shape
probe, rebuild E0/F0, and execute the explicitly synthetic dry-run fixture. Its
typed bundle and receipt remain `plumbing`, `admissible = false`, and
non-confirmatory; the authoritative verdict is `blocked`.
Caller-constructed receipt fields cannot change that result.

Phase B must first implement and review source provenance/reconciliation. Only
then may it replace every keyed sentinel (including D-M4/D-M5), seal the memo
and confirmatory integer thresholds, construct a scoring manifest, run a
passing confirmatory preflight before generation, execute the fixed arms once,
score mechanism first, and evaluate F0/F1 only after mechanism GO.
