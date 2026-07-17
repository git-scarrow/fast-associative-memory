# Five-arm FAM evaluation — preregistration (revision 3, DRAFT)

**Status:** not registered. Every keyed marker below must be replaced and the
memo bytes sealed before confirmatory outcome inspection. Corpus-shape probes
may disclose denominators, but never retrieval differences, prototype
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
outcomes are exploratory unless the conditional contested gate is registered.

The primary scorer is exact normalized structured-answer equality. H1 uses the
fixed-full stale-eligible denominator. Ledger lifecycle identity remains raw
string equality, protected by a sealed invariant rejecting normalized-equal but
raw-unequal values tied at maximum serial.

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

`abstention_bound = <<UNREGISTERED:D-3b>>`, either null or a number in `[0, 1]`.
Null explicitly registers no additional abstention gate.

### D-4 — Scorer semantics

`scorer = <<UNREGISTERED:D-4>>`. The only confirmatory literal is `exact`.
Containment, split scoring, and answer hygiene variants are exploratory.

### D-5 — Raw-arm truncation semantics

`raw_truncation = <<UNREGISTERED:D-5>>`. The only confirmatory literal is
`skip`: an over-budget record is skipped and later records are still considered.
E0 and F0 use the identical deterministic renderer and budget.

### D-6 — Contested-question disposition

`contested_disposition = <<UNREGISTERED:D-6>>`, either `exploratory` or `gated`.
Selecting `gated` additionally requires `contested_rule` and `contested_bound`;
otherwise contested counts and outcomes are reported but read by no value gate.

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

The two integer thresholds are derived only while constructing the seal. They
are recorded for audit but are not human registration choices.

## 4. Integrity, activity, and evaluability gates

All integrity checks run before generation. Any failure yields `blocked` and no
value claim. They include closed manifest schema and memo hash, immutable corpus
and embeddings, identical treatment settings except write mode, successful
rebuild of E0/F0, exact realized index attestations, no dropped/evicted writes,
complete one-to-one provenance over ledger record IDs, capacity equality, raw
renderer identity, and F0/F1 candidate-tuple identity.

FAM mechanism activity requires both `fam_attestation.merged > 0` and
`fam_attestation.key_drifted_merges > 0`. Prototype occupancy alone cannot
establish activity. Zero realized merges, zero key-drifted merges, an empty
mechanism denominator, `recall_n < D-M3`, or missing application denominators
yields `not-evaluable`, never GO.

## 5. Fixed-sequence gates and verdict

After integrity and evaluability:

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
min_mechanism_recall_n, stale_reduction_margin,
clean_answer_loss_bound, current_adoption_floor, abstention_bound,
scorer, raw_truncation, contested_disposition, equivalence,
min_stale_eligible_n, min_clean_n, h1_denominator, primary_family,
claim_order, memo_sha256
```

When `contested_disposition` is `gated`, `contested_rule` and
`contested_bound` are additionally mandatory. Unknown, missing, wrongly typed,
out-of-range, or still-sentinel fields are rejected. No numeric bound or
minimum has a default.

## 7. Exploratory nonclaims

No confirmatory inference is made from E1, no-memory accuracy, contested
questions unless conditionally gated, cross-family generated-answer deltas,
latency, token cost, prototype identities, per-scope examples, effect sizes
viewed before registration, or any application result when the mechanism is
blocked, inactive, under-denominated, or fails M1/M2. Synthetic and dry-run
fixtures prove plumbing only and are never benchmark evidence.

## 8. Registration and execution sequence

1. Freeze code, corpus transformer, renderer, scorer, and treatment schema.
2. Run the outcome-blind shape probe only.
3. Replace every keyed sentinel and seal the revision-3 memo SHA.
4. Build and attest E0/F0 exclusively from sealed inputs.
5. Run all integrity and treatment-fidelity checks before generation.
6. Execute the fixed five arms once per query with paired candidate reuse.
7. Score mechanism counts, determine activity/evaluability, then apply M1/M2.
8. Only after mechanism GO, score the F0/F1 application gates.
9. Publish all counts, denominators, thresholds, failures, and exploratory labels.
