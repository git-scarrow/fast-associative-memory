# S1 batch packet-reader contract candidate for W2:F1b — registered definition (design-only)

*Definition only. This document discharges exactly one deliverable of
PR-12.8 (`PR12_8_READER_CONTRACT_CANDIDACY.md`, main `0167125`): the
Stage C(i) served-field/contract registration. It defines — it does not
implement, prove, score, adjudicate, or certify. No reference reader is
written by this document; the Stage C(ii) composition proof, the
Stage C(iii) exactness-envelope freeze, Stage A/B panel work, Stage D
seed work, and Stage E adjudication each remain separately
unauthorized. **This candidate has no operative force**: nothing may
serve `witness_alt` to any consumer under it; no deployment, live
acting, prompting use, promotion, memory ingestion, autonomous
downstream use, FAM-core integration, or reader-contract change is
authorized or implied. **PR-10 merge-abstain remains the only certified
reader contract**; the operational posture on witness-window rows
remains **deferral**. This definition is frozen at commit; any change
is a new candidate version requiring full re-adjudication (§10).*

---

## 1. Contract identity

| field | value |
|---|---|
| contract_id | `s1-witness-alt-batch` |
| version | `0.1-candidate` |
| status | **defined** (see the §9 status ladder — lowest rung) |
| surface | S1 batch packet-reader (PR-12.8 §4, determination D1; S2/online excluded) |
| policy | `W2:F1b` exactly as merged at PR-12.6 (`9a7537e`), policy pin `0afcb2bc4d94112fd2f2cb9a47525c6d2595c2dd` |
| policy block sha256 | `2f009cf29ce64dc21e7bd392ceac4bbc189bc5b197489c2edaf8cd3ed1022c15` (the G-H3/G-R7-attested verbatim block: constants, `WITNESS_BASIS`, `RowObs`, `CellCtx`, `_f1a_condition`, the six policy functions, `POLICIES`) |
| emitter pin | `harness_boundary_sim.py` sha256 `2539686a205fa03ba88fb4e222043720dc6acf97460f896a31bd58d1a11d32e5` |
| evidence base | PR-12.3 `aece0d4` → 12.4 `2226d9d` → 12.5 `0afcb2b` → 12.6 `9a7537e` → 12.7 `7e98518` → 12.8 `0167125` (verbatim inventory: PR-12.8 §3) |
| incumbent contract | PR-10 merge-abstain (`readout-certified`), untouched (§6) |

## 2. Consumed inputs (exhaustive; anything else is a §8 violation)

The reader under this contract consumes, per cell, exactly:

1. `memory_packet.jsonl` — the prompt-visible packet of one **completed**
   cell emitted at shape **W2** by the byte-frozen emitter;
2. `audit_packet.jsonl` — the same cell's audit packet (same directory).

Both must originate from a governed emission whose provenance satisfies
the PR-12.7 G-H2 discipline (regenerable byte-identically through the
pinned emitter from pinned committed inputs). The contract applies to
**W2-shape packet trees only**: a reader consuming a `prototype` or
`W1` tree MUST NOT emit `witness_alt` under any condition.

**Batch semantics, made explicit:** the reader MUST ingest the cell's
*entire* audit packet before deciding any row. The cell-context
features (§4) are end-of-run quantities — this is the registered basis
of the S1 surface determination (PR-12.8 §4) and the reason this
contract is *batch-only*: consuming a partial or streaming packet under
this contract is a §8 violation, and no S2/online claim may cite it.

## 3. Output: the served-decision record

For every row (line) of `memory_packet.jsonl`, the reader emits exactly
one served-decision record:

| field | type / domain | semantics |
|---|---|---|
| `query_id` | string, verbatim from the packet row | join key |
| `served_outcome` | one of `answer` \| `abstain` \| `witness_alt` \| `defer` | §5 mapping |
| `abstain_reason` | `""` \| `merge_suspect_led` | non-empty iff `served_outcome == abstain`; incumbent vocabulary, unchanged |
| `witness_alt_class` | int decode class \| `""` | non-empty iff `served_outcome == witness_alt`; the §4 sole alternative |
| `witness_alt_basis` | `"witness co-resident (fork_witness)"` \| `""` | fixed string; non-empty iff `witness_alt` |
| `policy_id` | `"W2:F1b"` \| `""` | non-empty iff `witness_alt` |
| `policy_block_sha256` | the §1 block sha \| `""` | non-empty iff `witness_alt` |
| `evidence_ptr` | string \| `""` | iff `witness_alt`: the row's audit-packet decision `evidence_ptr`, carried verbatim |
| `certification_tier` | `"core-certified"` \| `"harness-heuristic"` | `core-certified` **only** on `abstain` rows (the incumbent's tier); `witness_alt` and `defer` rows are always `harness-heuristic` |
| `contract_id`, `contract_version` | §1 constants | stamped on every record |

Encoding is additive-only: no packet field is rewritten, reordered, or
suppressed; the served-decision records are a *new* artifact beside the
packets, never a mutation of them.

## 4. Eligibility — when a row is served `witness_alt` (normative restatement of the frozen policy; the sha-attested code is authoritative on any divergence)

Let P be the cell's packets. Compute once per cell, from
`audit_packet.jsonl` alone:

* `n_contradiction_pairs` — count of records with
  `record_type == "contradiction_pair_review"`;
* `n_ambiguous_pairs` — count of records with
  `record_type == "ambiguous_pair_review"`;
* `never_resolving_slots` — the union of `pair.incumbent_slot` and
  `pair.owner_slot` over ambiguous-pair records flagged
  `never_resolving`.

A memory-packet row R is **eligible** iff ALL of:

1. R carries at least one item with `type == "unresolved_tie"`; the
   **first** such item in R's item order is the tie considered (exactly
   the frozen scorer's `ties[0]`; any further tie items are ignored,
   not disqualifying);
2. the tie's candidate set presents exactly **2** distinct decode
   classes (width 2);
3. every candidate whose `basis` is not `"deployed vote"` has
   `basis == "witness co-resident (fork_witness)"`;
4. the deployed-vote candidate exists and its `slot` ∈
   `never_resolving_slots`;
5. exactly **one** alternative decode class exists beside the deployed
   class;
6. **quiet-cell guard:** `n_contradiction_pairs ≤ n_ambiguous_pairs`.

If eligible, `served_outcome = witness_alt` and `witness_alt_class` =
the sole alternative class (which is by construction a member of the
row's presented set; serving any class outside the presented set is a
§8 violation, mirroring 12.6 kill-5). If any condition fails, the row
is NOT eligible and §5 applies.

## 5. Outcome mapping for every other row (committed posture, unchanged)

* Rows carrying the certified abstention notice (the packets'
  `abstention_notice` item; merge-suspect-led) → `served_outcome =
  abstain`, `abstain_reason = merge_suspect_led`. Incumbent semantics,
  byte-untouched.
* Rows the packet serves as a plain or caveated answer → `served_outcome
  = answer` (the deployed answer, unchanged; caveats carried by the
  packet remain the reader's to display, not this contract's to alter).
* Every remaining row — dual-present-but-ineligible, escalated,
  withheld, quarantine- or supersession-disposed — → `served_outcome =
  defer`: the committed dual-present/escalation posture, unchanged.

The candidate therefore modifies the reader outcome of **exactly one
row class**: eligible W2 unresolved-tie rows, `defer` → `witness_alt`.
Nothing else moves.

## 6. Composition with the incumbent contract (PR-12.8 D3 invariants, as contract clauses)

* **I1 (precedence):** `abstain ≻ witness_alt ≻ defer`. A row carrying
  the certified abstention notice MUST be served `abstain` even if it
  were somehow to satisfy §4 (structurally impossible — abstained rows
  carry no `unresolved_tie` item — but the precedence is normative, not
  an assumption).
* **I2 (disjointness):** the §4 eligible set MUST be disjoint from the
  certified abstention set on every cell. Stage C(ii) proves this
  exactly per cell; under this definition it is a conformance check
  every reader MUST perform and fail closed on.
* **I3 (incumbent immutability):** no field, value, or encoding the
  PR-10 contract serves changes on any row.
* **I4 (no new authority):** `witness_alt` is `harness-heuristic` tier,
  always. Nothing under this contract is, or may be presented as,
  `core-certified`.

## 7. Conformance requirements for any future reference reader (Stage C; not built here)

Stdlib-only, read-only over the packets, no network, no FAM-core
import; deterministic — identical inputs MUST yield byte-identical
served-decision records across internal double pass and external
re-run; label-free by construction — the §2 inputs are the *only*
inputs, and the §8 prohibitions are enforced structurally (policy-
visible code receives packet-derived views only, in the 12.6/12.7
scorer pattern); fail-closed — any packet failing schema expectations,
any I2 overlap, any eligibility ambiguity → the affected row is served
`defer` (never `witness_alt`) and the anomaly is recorded.

## 8. Prohibitions (each a §10-kill under PR-12.8 if breached by any stage artifact)

Reading truth labels, registry CSVs, `pr12_5/`–`pr12_8/` scoring
outputs, PR-4 governance rows, scan-report JSONs, or cell/pair/arm/
seed/file identifiers into any eligibility decision; consuming partial
or streaming packets (§2); emitting `witness_alt` from non-W2 trees;
serving any class outside a row's presented set; presenting
`witness_alt` as certified, deployed, promoted, or FAM-core-integrated;
citing this definition as S2/online evidence; moving any §1 pin or §4
condition without a version bump and re-adjudication (§10).

## 9. Status ladder (this document sits at the lowest rung)

1. **defined** (this document) — vocabulary and semantics fixed; no
   operative force.
2. **composition-proven + envelope-frozen** — Stage C(ii)/(iii):
   I1–I4 proven exactly on every panel cell; the per-cell exact
   acted-row envelope committed (`pr12_8/f1b_envelope.json`); a
   conformant reference reader reproduces it byte-for-byte.
3. **adjudicated** — Stage E emits one PR-12.8 §12 verdict over gates
   G-R1–G-R7 (with Stage A/B panel coverage and the Stage D seed
   branch honestly named).
4. **certified** — outside PR-12.8 entirely: a future certification
   pre-registration with its own gates and approval. Only at this rung
   could any consumer ever be served `witness_alt` — and only then
   under whatever narrower terms that registration sets.

Each rung requires separate explicit approval; skipping a rung is a
kill (PR-12.8 §10.9, scope laundering).

## 10. Change control

This definition is frozen at commit. Any modification — a pin, a §4
condition, a §3 field, a §6 clause — produces `0.2-candidate` (or
later) and returns the candidate to rung 1 with all Stage C–E work
void. Errata follow the repo's append-only convention and may only
tighten or correct the record, never relax a condition.

## 11. Boundary (restated)

This document defines a candidate and nothing more. No deployment, no
live acting, no prompting use, no promotion, no memory ingestion or
write-back, no autonomous downstream use, no LLM/agent reader
certification, no online/driver seam, no FAM-core integration or
change, and no reader-contract change. **PR-10 merge-abstain remains
the only certified reader contract**; the operational posture on
witness-window rows remains **deferral**. PR-12.1–12.8 verdicts and
registrations stand unchanged.
