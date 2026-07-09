# S1 batch packet-reader contract `s1-witness-alt-batch@1.0` — certified re-issue (seed-bounded; effective only on approved Stage III merge)

*The §5.4 contract re-issue of `PR12_9_READER_CONTRACT_CERTIFICATION.md`
(main `fe6248e`) — the one permitted transformation of the adjudicated
0.1-candidate definition, produced under the separately-authorized
Stage III after the Stage II verdict
`reader-contract-certified-seedbounded(s1-witness-alt-batch@1.0, W2:F1b)`
(all gates C-1–C-7 pass; evidence preserved on main at merge
`8875b72`). **This document has no effect until the Stage III merge is
explicitly approved.** Upon that approval and not before: the contract
registry sentence becomes exactly the §4.2 wording — "PR-10
merge-abstain is the only core-certified reader contract;
`s1-witness-alt-batch@1.0` is a certified opt-in **batch,
harness-heuristic-tier, seed-bounded** packet-reader contract at its
registered bounds." — and the §4.3 posture takes effect: blanket
deferral remains the **default** on witness-window rows; `witness_alt`
may be served **only** to consumers that explicitly opt in, only on
completed W2 packet trees, only under the standing
`PR12_8_MONITORING_WITHDRAWAL.md` T1–T7 tripwires with the append-only
event log active; non-opt-in consumers see no change, and **no consumer
is enrolled by this document**. The `-seedbounded` qualifier is
permanent. The §§2–8 normative content below is **byte-carried
unchanged** from `PR12_8_S1_CONTRACT_CANDIDATE.md`; any semantic
difference voids certification (PR-12.9 §8.5). No FAM-core
integration, no S2/online seam, no prompting or autonomous use, no
LLM/agent reader certification, and no change to PR-10 — ever, under
this registration.*

---

## 1. Contract identity

| field | value |
|---|---|
| contract_id | `s1-witness-alt-batch` |
| version | `1.0` |
| status | **certified-seedbounded, pending effect** (rung 4 of the §9 ladder; status and registry sentence take effect only upon explicit approval of the PR-12.9 Stage III merge) |
| certification | PR-12.9 Stage II: `reader-contract-certified-seedbounded(s1-witness-alt-batch@1.0, W2:F1b)` — `pr12_9/certification_scan.json`, gates C-1–C-7 all pass, certification pin `10b9335`, evidence merged at `8875b72` |
| surface | S1 batch packet-reader (PR-12.8 §4, determination D1; S2/online excluded) |
| policy | `W2:F1b` exactly as merged at PR-12.6 (`9a7537e`), policy pin `0afcb2bc4d94112fd2f2cb9a47525c6d2595c2dd` |
| policy block sha256 | `2f009cf29ce64dc21e7bd392ceac4bbc189bc5b197489c2edaf8cd3ed1022c15` (the G-H3/G-R7/C-1-attested verbatim block: constants, `WITNESS_BASIS`, `RowObs`, `CellCtx`, `_f1a_condition`, the six policy functions, `POLICIES`) |
| emitter pin | `harness_boundary_sim.py` sha256 `2539686a205fa03ba88fb4e222043720dc6acf97460f896a31bd58d1a11d32e5` |
| evidence base | PR-12.3 `aece0d4` → 12.4 `2226d9d` → 12.5 `0afcb2b` → 12.6 `9a7537e` → 12.7 `7e98518` → 12.8 Stage E `a0e621d` (verdict `contract-candidate-GO-seedbounded(W2:F1b)`) → PR-12.9 `fe6248e` Stages I–II |
| bounds (constitutive) | PR-12.9 §3 verbatim: traffic (PR-12.8 §14.3), contra-power (§14.4), seed (§14.5, `-seedbounded` permanent), monitoring T1–T7 (`PR12_8_MONITORING_WITHDRAWAL.md`) |
| incumbent contract | PR-10 merge-abstain (core-certified), untouched (§6) |
| normative text | §§2–8 below byte-carried unchanged from `PR12_8_S1_CONTRACT_CANDIDATE.md` (0.1-candidate) |

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

## 9. Status ladder (this re-issue sits at rung 4, seed-bounded)

1. **defined** — Stage C(i), merged `8cbf870`.
2. **composition-proven + envelope-frozen** — Stage C(ii)/(iii), merged
   `ddcc38a`; envelope v0.2 at `d25cd2d`.
3. **adjudicated** — PR-12.8 Stage E, merged `a0e621d`, verdict
   `contract-candidate-GO-seedbounded(W2:F1b)`.
4. **certified** (this re-issue) — PR-12.9 Stage II verdict on
   certification evidence merged at `8875b72`; **reached in effect only
   upon explicit approval of the PR-12.9 Stage III merge**, per gate
   C-7 approval separation.

Each rung required its own explicit approval; skipping a rung is a
kill (PR-12.8 §10.9 / PR-12.9 §8.1, scope laundering).

## 10. Change control

This re-issue is frozen at commit. Any modification — a pin, a §4
condition, a §3 field, a §6 clause — produces version `1.1` (or
later), **voids this certification** (PR-12.9 §8.5), returns the
contract to rung 1 of the §9 ladder, and requires full
re-adjudication and a new certification pre-registration with its own
gates and approvals. Errata follow the repo's append-only convention
and may only tighten or correct the record, never relax a condition.
The monitoring terms are tighten-only, forever; withdrawal and
reinstatement follow `PR12_8_MONITORING_WITHDRAWAL.md` §4 exactly
(reinstatement only via a new pre-registration).

## 11. Boundary (restated)

Until the PR-12.9 Stage III merge is explicitly approved, this
document confers nothing: **PR-10 merge-abstain remains the only
certified reader contract**, and the operational posture on
witness-window rows remains **deferral**. Upon approval and thereafter:
the contract is opt-in and batch-only; every `witness_alt` record is
`harness-heuristic` tier, always; the seed bound is permanent; the
T1–T7 tripwires and withdrawal semantics are standing conditions with
a mandatory append-only event log; and there is no deployment, live
acting, prompting use, promotion, memory ingestion or write-back,
autonomous downstream use, LLM/agent reader certification, S2/online
seam, or FAM-core integration or change — ever, under this
registration. PR-10 merge-abstain is never modified by any outcome
here. PR-12.1–12.9 verdicts and registrations stand unchanged.
