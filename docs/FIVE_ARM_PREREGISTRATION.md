# Five-arm governed memory evaluation — pre-registration (revision 2, DRAFT)

**STATUS: DRAFT. NOT REGISTERED. This memo does not authorize a run.**

Every decision in §3 carries an `UNREGISTERED` sentinel keyed to its own
D-field. While any sentinel remains, the registration is incomplete (§8) and
any run executed anyway is **exploratory-only**, read by no gate.

This memo registers the decision rules *before* the data exists. PR-13 — the
base commit of this branch — passed its integrity tier but could not issue its
confirmatory H1/H2 because the rubric was underspecified at seal time. The same
defect class, one iteration later, would void this run. Checklist discipline
already failed here once, so §8's completeness rule is machine-checked
(`validate_registration`) and every claim this memo makes about its own gates is
pinned by a test in `tests/test_memory_eval_preregistration.py`.

## Revision 2 changelog — what revision 1 got wrong

Revision 1 was reviewed and five protocol defects were found and confirmed.
They are recorded here because the memo's own errors are part of its evidence:

1. **G-I5 was mathematically wrong.** It claimed a threshold below `1/n` is
   "unfalsifiable". It is not: a sub-`1/n` bound on H2 is a strict
   *zero-losses-permitted* gate, and a positive sub-`1/n` floor on H3 requires
   one success — both perfectly falsifiable. The real defect is *illusory
   precision*. Worse, H1 had **no single `1/n`** to speak of, because the
   scorer removes malformed rows from each arm's denominator independently
   (`scoring.py:243`), so H1 differenced two rates over different question
   sets. §2 now specifies integer gate arithmetic and §3 adds **D-9**.
2. **The sequence was impossible.** §9 registered every decision before a pilot
   whose whitelist did not disclose what D-5 needs, and resolved FAM
   attribution after the primary-family field it gates. §9 is re-phased.
3. **The recommended scorer permitted false wins.** Containment scores
   "not NewCo" and "cannot confirm NewCo" as correct, and because
   `current_adoption_rate` is computed from `correct`, it would **defeat H3**,
   the anti-suppression gate. Revision 1's claim that the split option "avoids
   both failure directions" was asserted without a test. D-4 is rebuilt on
   demonstrated behaviour.
4. **D-6 was inert.** It registered a contested policy no gate consumed. It is
   now a disposition with conditional fields.
5. **Decisions and schema were not one-to-one**, the abstention sub-decision
   had no field, and the verdict table overlapped when H2 and H3 both failed.
   All three are now machine-checked.

**Rule adopted in response to defect 3:** no option in this memo is marked
RECOMMENDED unless a committed test demonstrates the reason.

---

## 1. Core question and hypotheses

**Core question.** At fixed retrieval, does lifecycle-aware constructive
forgetting reduce a consumer's adoption of superseded facts without paying for
it in suppressed current answers?

The retriever is queried once per family per question and the ordered
candidate-ID list reused for both arms in that family, so within a family the
governed-minus-raw difference isolates the compiler. Nothing below compares
across families.

For family `F` ∈ {`vector`, `fam`}, over the **stale-eligible stratum** (§2):

**H1 (governance suppresses stale adoption).** Governed stale adoptions are
fewer than raw's by at least the D-1 margin, on the D-9 common denominator.
Loses to: no reduction, or a reduction inside the margin.

**H2 (governance does not pay for it on clean facts).** Clean-answer loss is
within the D-2 bound. Loses to: governance breaking uncontested answers.

**H3 (governance does not win by suppression).** Governed current adoption
meets the D-3 floor. Loses to: an arm reaching H1 by abstaining rather than by
choosing correctly.

**The value claim for family F is H1 ∧ H2 ∧ H3.** H3 is load-bearing: without
it the claim is winnable by pure suppression, since an arm abstaining on every
evolving question scores stale adoption 0.0 and clean-answer loss over a
disjoint stratum. That path was reproduced by execution and is pinned as a
regression test. Because deferral is this project's default posture, H3 is the
gate most likely to fire.

---

## 2. Strata, denominators, and gate arithmetic

`classify_scope` partitions every question's scope into exactly one of:

| Stratum | Definition | Gated by |
|---|---|---|
| `clean` | one distinct value | H2 |
| `stale_eligible` | superseded history, no fork at max serial | H1, H3 |
| `contested` | distinct values share the max serial | D-6 |

Contested is judged first, so a fork carrying superseded history is contested,
never stale-eligible — this keeps wrong-fork answers out of the H1 denominator.

**Every rate carries its denominator** (`Rate(value, n)`), and `value is None`
iff `n == 0`. A rate with `n == 0` is missing data: **it satisfies no criterion
and fails no criterion.** It is never read as a pass.

**Gate arithmetic is integer.** A rate is a ratio of counts, so a gate written
as a float comparison inherits its denominator's granularity. Every gate
converts its registered rate into an exact row count first and compares
integers:

| Gate | Registered as | Evaluated as |
|---|---|---|
| H1 | margin D-1 | `raw_adoptions − governed_adoptions ≥ ceil(D-1 × n_paired)` |
| H2 | bound D-2 | `clean_losses ≤ floor(D-2 × clean_n)` |
| H3 | floor D-3 | `current_adoptions ≥ ceil(D-3 × stale_n)` |

Implemented in `preregistration.py` (`h1_passes`, `h2_passes`, `h3_passes`).
This makes each gate exact and interpretable — "at most 2 losses out of 40"
rather than "≤ 0.05" — and it makes the granularity visible at registration
time instead of after the run.

**Precision note (replaces revision 1's G-I5).** A threshold below `1/n` is
*not* unfalsifiable; it is strict, and it remains falsifiable. What it is, is
imprecise: at `n = 10`, bounds of 0.005 and 0.09 specify the identical gate
(`floor(bound × n) == 0`). Registering a threshold finer than `1/n` therefore
conveys precision the corpus cannot support. This is a **reporting obligation,
not an integrity gate**: §8 requires the integer count each threshold implies
to be recorded alongside it, so the reader sees the gate that will actually run.

**Anti-forking-paths, enforced structurally.** The shape probe (§9 phase 2)
takes **no consumer** and its output type carries no arm, answer, or outcome
(`ShapeProbe`, whitelist pinned by test). It therefore *cannot* disclose an
effect size. Because the disclosure boundary is structural rather than a
discipline about which fields to read, any decision made from probe output is
uncontaminated by outcomes **by construction**. Revision 1 tried to enforce
this with a prose rule and produced an impossible sequence; the whitelist
replaces the rule.

---

## 3. Decisions requiring registration

### D-1 — Stale-reduction margin

Minimum `raw − governed` stale-adoption rate difference on the D-9 denominator:
**`<<UNREGISTERED:D-1>>`**

*Controls:* how much reduction counts as an effect. *At 0.0:* any single row of
difference passes. *Too high:* a real but modest effect reads as failure.
*Depends on:* `min_stale_eligible_n` (D-8a) for its integer resolution.

### D-2 — Clean-answer-loss bound

Maximum `clean_answer_loss`: **`<<UNREGISTERED:D-2>>`**

*Controls:* acceptable collateral damage on uncontested facts. *At 0.0:* a
single clean regression kills the claim — defensible, but note D-4 can
manufacture such a regression from a caveat echo. *Depends on:* `min_clean_n`
(D-8b).

### D-3 — Current-adoption floor

Minimum governed `current_adoption_rate` on the stale-eligible stratum:
**`<<UNREGISTERED:D-3>>`**

*Controls:* the anti-suppression gate — the fraction of evolving questions the
governed arm must answer *correctly*, not merely avoid answering wrongly.
*At 0.0:* H3 is inert and the claim reverts to suppression-winnable. *At 1.0:*
any single abstention kills the claim, contradicting the deferral posture.
*Note:* malformed rows are **inside** this denominator by design, so the floor
also bounds interface failure.

### D-3b — Additional abstention bound

An optional second anti-suppression constraint on `abstention_rate`; `null`
registers "none". **`<<UNREGISTERED:D-3b>>`**

*Why it is not a substitute for D-3:* an abstention bound alone does not
constrain malformed output, so a malformed-heavy arm could satisfy it while
answering nothing. D-3 requires correctness and subsumes that hole; this field
exists so that adding a bound *on top* is an explicit registered choice rather
than an omission. (Revision 1 raised this as a sentinel with no schema field,
making it unregistrable.)

### D-4 — Scorer semantics

**Premise correction.** Revision 1 justified containment by comparability with
upstream `substring_exact_match`. That premise is weak: upstream scores **free
text** in which the answer is embedded in prose, whereas this harness scores a
**structured `answer` slot** under the PR-13 consumer output contract
(`parse_consumer_output`), which already performs the extraction substring
matching approximates. Cross-paper comparability is confounded by the differing
output contract regardless of which scorer is chosen.

Demonstrated behaviour (`test_memory_eval_preregistration.py`):

| Answer, expected `NewCo` | exact | containment |
|---|---|---|
| `NewCo` | correct | correct |
| `NewCo (current)` | **wrong** (false loss) | correct |
| `not NewCo` | wrong | **correct** (false win) |
| `cannot confirm NewCo` | wrong | **correct** (false win) |

- **Option A — exact.** Only failure mode is the caveat echo: a **false loss**,
  conservative against the governance hypothesis.
- **Option B — containment.** Scores negations and hedges as correct: **false
  wins**. Because `current_adoption_rate` derives from `correct`, this
  **defeats H3** — a governed arm that hedges its way out of answering still
  clears the anti-suppression floor.
- **Option C — split** (containment correctness, exact stale adoption). Exact
  stale adoption is right, but the hole in Option B is in the *correctness*
  metric, so C inherits it. Revision 1's claim that C "avoids both failure
  directions" is **withdrawn as false**.
- **Option D — exact-with-hygiene.** Option A plus a registered bound on
  answer-slot hygiene (rows whose slot is not a bare value), so the caveat-echo
  false loss is measured rather than absorbed. Requires `slot_hygiene_bound`.

**RECOMMENDED: the exact family (A or D)** — test-backed. Exact can only
manufacture a false *loss*; containment can manufacture a false *win* and
specifically disables H3. Under this repo's posture a gate must be able to
fail, so the conservative failure direction is the safe one. The recommendation
covers the family only; **A vs D remains open** and is not test-decided.

Registered choice: **`<<UNREGISTERED:D-4>>`** (`exact` | `containment` |
`split` | `exact-with-hygiene`)

Whichever is chosen, both variants should be **reported**; only the registered
one is read by a gate. Switching scorers after the run downgrades it to
exploratory (PR-13 precedent).

### D-5 — Raw-arm truncation semantics

`render_raw` stops at the first over-budget candidate (`break`); the governed
compiler summarizes and continues. When the budget binds this is asymmetric and
systematically **pro-governed**, because the compiler's priority key protects
recency — exactly the correct FactConsolidation answer.

- **Option A — symmetrize** (skip-not-stop in `render_raw`).
- **Option B — raw-matched arm** using the existing `render_raw_matched`
  (`harness/ctx/compile.py:399`, currently unused). Adds a sixth arm; "exactly
  five arms" would need amending.
- **Option C — register `break` as part of the raw-baseline definition** and
  rename the delta accordingly (a weaker but honest claim: governance vs naive
  prefix-truncated RAG).

Registered choice: **`<<UNREGISTERED:D-5>>`** (`skip` | `matched-arm` | `break`)

*Resolved at phase 3 from `ShapeProbe.budget_binding_questions`.* If the budget
never binds on the real corpus, all three options coincide and the choice is
recorded as moot rather than assumed moot.

### D-6 — Contested-question disposition

Contested questions are currently counted and reported but scored into no
stratum, and `fork_adoption_rate` is read by no gate — so in revision 1 this
decision changed nothing. It is now a disposition:

- **Option A — `exploratory`.** Contested questions are excluded from every
  gate; `contested_n` and `fork_adoption_rate` are reported only. This is a
  *registered* exclusion, not a silent default.
- **Option B — `gated`.** Adds gate G-V4. Requires `contested_rule` (whether
  the correct governed answer on a fork is `abstain` or the dataset annotation)
  and `contested_bound`.

Registered choice: **`<<UNREGISTERED:D-6>>`** (`exploratory` | `gated`)

*Note:* `_validate_expected_answer` accepts a fork question whose annotated
answer is one fork member — the annotation silently adjudicates a fork the
governance layer is designed to refuse. Option B with
`contested_rule = annotation` would score *against* the thesis's own position;
that is a legitimate choice, but it must be made knowingly.

### D-7 — Equivalence relation (ledger vs scorer)

`ledger.py:67` forks on **raw string** equality; scoring classifies scopes on
**normalized** equality. A same-serial pair differing only in case or
whitespace is `human-review` to the compiler (governed fails closed) while
scoring calls the scope clean — the designed abstention then lands in H2's harm
number. Bias is exclusively anti-governed.

- **Option A — `normalized`:** one relation in both places.
- **Option B — `raw-with-invariant`:** keep the conservative raw-equality fork
  test and add a sealed-input invariant rejecting same-serial values that are
  normalize-equal but raw-unequal, making the disagreement unrepresentable.

Registered choice: **`<<UNREGISTERED:D-7>>`** (`normalized` | `raw-with-invariant`)

### D-8a — Minimum stale-eligible denominator

**`<<UNREGISTERED:D-8a>>`** — below this, H1 and H3 report `not-evaluable`,
never `pass`.

### D-8b — Minimum clean denominator

**`<<UNREGISTERED:D-8b>>`** — below this, H2 reports `not-evaluable`.

### D-9 — H1 denominator policy

H1 differences two arms' stale-adoption rates, but the scorer removes malformed
rows from each arm's denominator **independently** (`scoring.py:243`). Measured:
on a corpus where the governed arm is malformed on one stale-eligible question,
raw reports `n=2` and governed `n=1` — two rates over different question sets,
whose difference is a multiple of no single `1/n`. H1 as written in revision 1
was therefore not a paired comparison and had no exact integer margin.

- **Option A — `paired-complete`:** denominator is the stale-eligible questions
  where **neither** arm in the family is malformed. *Cost:* an arm malformed on
  precisely the rows it would have adopted stale gets them dropped, hiding its
  worst case.
- **Option B — `fixed-full`:** denominator is every stale-eligible question,
  malformed counted as non-adoption. *Cost:* an interface failure reads as
  stale avoidance — the deflation defect the malformed exclusion was added to
  fix, reintroduced into H1 specifically.

Both costs are demonstrated by test on identical rows, where the two policies
return **opposite H1 outcomes**. Neither is free; that is why this is
registered rather than defaulted. *Not registrable:* the current per-arm
behaviour, which supports no exact margin.

Registered choice: **`<<UNREGISTERED:D-9>>`** (`paired-complete` | `fixed-full`)

*If `paired-complete` is chosen,* consider pairing it with D-3b or a malformed
bound so the dropped set cannot grow large enough to game.

### D-10 — Primary family

Two families × three criteria are available, and reporting whichever family
wins is a garden of forking paths. The non-primary family is reported and read
by no gate unless `both` is registered.

Registered choice: **`<<UNREGISTERED:D-10>>`** (`vector` | `fam` | `both`)

**Blocked on the FAM attribution.** Measured during review: no FAM-distinctive
mechanism executes — `vigilance=-1.0` never rejects, `immutable_keys=True`
skips EMA drift, blended values are discarded, and CAM keys are identical to
the harness-precomputed centroids to `0.0`. Until that is exercised or
relabeled, the `fam` family measures a **harness-computed scope-centroid index
with FAM provenance storage**, not FAM condensation. Registering `fam` or
`both` while the arm labels claim condensation would attribute the result to a
mechanism that never ran. **§9 phase 0 resolves the attribution before this
field is registered.**

---

## 4. Gates

### Integrity gates — any failure → `blocked`, no verdict, no value claim

- **G-I1 registration complete** — `validate_registration` returns no errors:
  every D-field present, no sentinel surviving, every conditional field
  supplied. Machine-checked.
- **G-I2 seal binds the treatment** — `policy_sha256`, `consumer_pin`,
  `fam_max_entries`, and scorer identity present in the sealed protocol, each
  asserted before the first `generate` call. *(Open: not built. Cannot pass
  today.)*
- **G-I3 pairing holds** — raw and governed `candidate_ids` identical within
  each family, per question. Enforced in `score_rows`.
- **G-I4 budget integrity** — every row's `block_tokens` ≤ 1,500 under the
  registered tokenizer.
- **G-I5 thresholds are recorded with their integer effect** — each of D-1,
  D-2, D-3 is recorded alongside the exact row count it implies at the
  registered minimum denominator (§2). *(Replaces revision 1's incorrect
  resolution rule; this is a completeness check, not a numeric constraint.)*
- **G-I6 corpus reconciliation** — loaded record and question counts match the
  pinned upstream revision's declared size. *(Open: not built.)*
- **G-I7 caveat integrity** — count of audit rows with `disposition == caveat`
  **and** `budget_decision == summarized` is **0**. Under budget pressure the
  sealed compiler strips the caveat from exactly the superseded items
  governance must label, re-serving stale content unlabeled; scoring would then
  book that as a governance failure. A run-validity gate, not a metric.

### Value gates — read only if every integrity gate passes

- **G-V1** — H1 at margin D-1 on the D-9 denominator, primary family.
- **G-V2** — H2 at bound D-2.
- **G-V3** — H3 at floor D-3 (and D-3b, if registered).
- **G-V4** — contested rule at bound D-6, *only if* `contested_disposition`
  registers `gated`.

---

## 5. Verdict

Implemented as a total function (`preregistration.verdict`) and tested
exhaustively over all gate combinations; exactly one verdict is reachable for
any input. Precedence is registered, not incidental:

| Condition | Verdict |
|---|---|
| any integrity gate fails | `blocked` |
| a gated stratum is empty or below its D-8 minimum | `not-evaluable` |
| H1 fails | `NO-GO — no effect` |
| H1 passes, H3 fails | `NO-GO — suppression` |
| H1 passes, H3 passes, H2 fails | `NO-GO — collateral` |
| H1 ∧ H2 ∧ H3 pass | `governed-memory-GO(<family>)` |

Rows are evaluated top-down and the first match wins. **H3 outranks H2:** when
H1 passes and both harm gates fail, the diagnostic fact is that the reduction
was bought by not answering — that explains the H1 win, so it is reported over
collateral damage. (Revision 1's table matched two rows in this case and named
no winner.)

**Kill conditions** — any of these voids the confirmatory tier and preserves
the run as exploratory-only: re-scoring with an unregistered scorer; re-running
after inspecting outcomes; editing the disposition policy, consumer pin, or
scorer between seal and execution; disclosing anything beyond the §9 phase-2
whitelist before registration is complete.

---

## 6. Exploratory-only — reported, read by no gate

`accuracy`, `abstention_rate` (unless D-3b registers a bound), `malformed_rate`,
`fork_adoption_rate` (unless D-6 registers `gated`), `mean_prompt_tokens`,
`mean_total_latency_ms`, the non-primary family, the non-registered scorer
variant, and the `no_memory` arm (a consumer-only lower bound; no hypothesis
references it).

Named here so a favorable exploratory number cannot be promoted into the claim
after the fact.

---

## 7. Registration schema

`seal_manifest` must refuse to seal unless `protocol["registration"]` passes
`validate_registration` with zero errors. The sentinel is keyed per decision so
the check is mechanical and the count cannot be inflated by prose.

```
"registration": {
  "memo_sha256":             "<sha256 of this file at seal time>",   # derived
  "stale_reduction_margin":  <rate>,   # D-1
  "clean_answer_loss_bound": <rate>,   # D-2
  "current_adoption_floor":  <rate>,   # D-3
  "abstention_bound":        <rate|null>,                            # D-3b
  "scorer":                  "exact" | "containment" | "split" | "exact-with-hygiene",  # D-4
  "raw_truncation":          "break" | "skip" | "matched-arm",       # D-5
  "contested_disposition":   "exploratory" | "gated",                # D-6
  "equivalence":             "raw-with-invariant" | "normalized",    # D-7
  "min_stale_eligible_n":    <int>,    # D-8a
  "min_clean_n":             <int>,    # D-8b
  "h1_denominator":          "paired-complete" | "fixed-full",       # D-9
  "primary_family":          "vector" | "fam" | "both",              # D-10
}
```

Conditional fields, required only when selected: `slot_hygiene_bound` (D-4
`exact-with-hygiene`); `contested_rule`, `contested_bound` (D-6 `gated`).

`memo_sha256` binds this text to the run: the registration cannot be edited
afterward without the mismatch being detectable. This is the standard PR-13
applies to its scoring manifest, which pins the disposition policy by sha256 —
a standard the five-arm seal does not yet meet (G-I2).

---

## 8. Sequence

Phases are ordered so that no decision precedes the information it needs.
Revision 1's ordering was impossible; this one is not.

**Phase 0 — decisions that need no data.** D-4 (scorer), D-6 (contested
disposition), D-7 (equivalence), D-9 (H1 denominator). Resolve the FAM
attribution — exercise real condensation or relabel the arms and strike the
condensation claims — **then** register D-10.

**Phase 1 — build the enforcement.** Wire `validate_registration` into
`seal_manifest` (G-I1). Land the seal extension (G-I2) and corpus
reconciliation (G-I6). Land the transformer from official
`Conflict_Resolution/fact_sh` and pin the embedding model.

**Phase 2 — shape probe.** Run `shape_probe` on the real corpus. It takes no
consumer, so it cannot emit an effect size; its whitelisted output is the only
thing that leaves this phase. Discard the rest.

**Phase 3 — decisions the probe informs.** D-5 from `budget_binding_questions`;
D-8a/D-8b from the stratum counts; D-1, D-2, D-3, D-3b calibrated against those
denominators' integer resolution (§2). Record each threshold's implied row
count (G-I5).

**Phase 4 — register and seal.** No sentinel remains; `validate_registration`
returns clean; stamp `memo_sha256`; seal; verify immediately before execution.

**Phase 5 — execute once.** Score with the registered scorer. Issue the §5
verdict.

---

## 9. What this memo does not cover

The `fact_sh` transformer does not exist yet, so record lengths, serial-tie
frequency, and clean-scope count are **unknown**. D-5 and D-8 are resolvable
only at phase 3, and whether the contested stratum is populated at all — which
decides whether D-6 is live or moot — is a fact to be **measured at the probe,
not assumed**. This remains the most decision-relevant unknown in the design:
it determines whether several of these gates can fire at all.

G-I2 and G-I6 are registered here but not built. They cannot pass today, and
their absence blocks any run — deliberately.
