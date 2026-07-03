# PR-9B — Second-Key Desk Scan (analysis-only pre-registration)

Date: 2026-07-02. Branch: `feat/pr9b-second-key-desk-scan`. Base: `main` @ `83f6ba1`
(PR-9.2 identity-certified).

## 0. Claim under test and scope

§9B write-event authority is gated on two conditions
(PR8_QUARANTINE_REPLACEMENT_GATE.md §9B):

- **(a)** a certified, complete, inert, event-addressable audit denominator —
  **met** by PR-9.2 (`identity-certified`, main `83f6ba1`); and
- **(b)** a **parameter-free binary second key** in **committed write-time
  observables** that discriminates harm-linked eligible write events from
  write-clean ones. If no such key exists, §9B is falsified before any
  acting run.

This desk scan adjudicates condition (b) and nothing else.

**Scope (verbatim constraints, all binding):** analysis-only; committed
artifacts only; no new gentoo runs; no runtime policy changes; no
`--read-govern` changes; no threshold tuning; no pair-specific exception
rules; no post-hoc use of answer correctness as a key input; no use of
future read outcomes as a key input; no implementation of write-event
authority. PR-10 `merge-abstain` remains the only certified reader contract
and is untouched. The read-time expansion and pending-window policy space
killed by PR-11.1 (verdict `negative`, committed in
`pr11/adjudication_scan.json`) is not re-opened.

**Stop conditions:** the scan stops immediately if it would require a
non-committed artifact, a tuned threshold, or an outcome-leaking column as
a key input.

**Integrity note (confirmatory scan).** This is a desk scan over
already-published artifacts. Some quantities scored below are derivable in
advance by any reader of the committed §9A/PR-7 results (e.g. the
constant-true key's acting outcome *is* the committed PR-7 refuse arm; the
first-epoch key's fired count is the §9A incumbent-diagnostic count). The
pre-registration therefore freezes **definitions, admissibility rules, the
candidate list, the bounds, and the verdict mapping** — none of which are
functions of the scores — not ignorance of the data. No bound or rule below
was chosen after computing a candidate's score.

## 1. Event denominator

The denominator is the **PR-9.2 certified event-addressable set**: the
quarantine-eligible supersession write events recorded per-event in the
shadow-arm quarantine ledgers of the §9A panel, addressed by

- `PR92_INTRINSIC_JOIN_KEY = (epoch, event_class, batch_index)` with
- `PR92_INTRINSIC_CHECK_FIELDS = (payload_label,)`

(`benchmarks/failure_mode_probe.py`; certified by
`PR9_2_IDENTITY_CERT.md`: 21 stale-soft cells at exactly 192/192
`identity-proven`, 6 clean cells proven-empty, zero collisions).

**Admissible artifacts** (all committed on `main`, read-only):

| artifact | role |
|---|---|
| `pr9_2/panel/*/shadow/*.summary.json` → `govern.quarantine_ledger.flagged_event_records` | the denominator itself (per-event intrinsic records) |
| `pr9_2/panel/*/shadow/*.fork_events.csv` | column-classification evidence + blocked-family diagnostics only |
| `pr7/twin_delta_refuse.json` | committed acting-arm reference (the constant-true key) |
| `pr11/adjudication_scan.json` | committed post-PR-10 residual counts (target-harm definition) |
| `pr6/panel.json` | committed per-pair frozen-probe hazard (target/benign cell assignment) |

The panel contains 27 cell-stems; identical stems appearing in more than
one hazard cell are the **same run** (committed byte-identical, per
PR-9.2), so the denominator deduplicates to **12 unique stale-soft runs**
(pairs A/B/D/E × seeds 0–2, 192 events each = 2,304 events) plus **6 clean
runs** (pairA, pairC × seeds 0–2, 0 events each). The scan re-verifies
counts and intrinsic uniqueness and treats any deviation as a stop
condition (unexplained row-count drift).

**Admissible key-input columns** — exactly the §9A-certified state-free
set, nothing else:

| column | why intrinsic / write-time / state-free |
|---|---|
| `epoch` | protocol schedule position; certified identical across none/shadow/quarantine |
| `event_class` | protocol ground truth stamped by the driver; constant `supersession` on the denominator by eligibility |
| `batch_index` | row position inside the write call's batch; certified identical across arms |
| `payload_label` | argmax of the incoming payload — intrinsic to the write content, not to memory. **Nominal** coordinate: class indices carry no order/magnitude, so only pair-agnostic predicates are admissible over it |

**Forbidden columns**, by reason (the user-specified taxonomy):

| column(s) | reason |
|---|---|
| `pre_sim`, `payload_cos_incumbent`, `effective_vigilance`, `incumbent_slot`, `incumbent_hit_counts`, `incumbent_last_write_seq`, `incumbent_n_records` | **runtime state contamination** — every one is computed against the pre-write memory state (`_pre_write_observables`). Under an acting arm the memory diverges after the first action, so the value seen at decision time diverges from the committed desk value. This is not hypothetical: it is the §9A-certified failure of the old join key (smoke 8/24; panel incumbent diagnostic exactly 32/192 on every cell — agreement only on the first eligible epoch). A key scored here on these columns would not be the key the acting run evaluates. |
| `outcome`, `owner_slot`, `record_seq` | **outcome-leaking / post-commit** — assigned by or after `write_fn`; a §9B decision must be taken *before* the write commits (`logged_learn` consults the hook pre-write). `record_seq` is additionally state-contaminated (the global sequence counter shifts once any earlier write is diverted). |
| `injected_label` | **scorer-only** ground-truth channel (the experimenter's injected wrong label; `-1` on every denominator row). |
| `arm`, `record_tag` | **pair/arm identity** — any predicate reading them is a pair-specific exception rule by construction. |
| everything in `per_slot.csv`, `topk.csv`, per-probe CSVs, `governance.json` | **read-time / probe-time / scorer-side** — not write-time observables; using them leaks future read outcomes. |

## 2. Candidate second keys

A candidate must be: **binary**, **parameter-free** (no tuned constant; only
protocol-distinguished positions such as *first*/*last* are admitted as
constants), **write-time available** (computable from admissible columns at
the pre-write decision point), **non-leaking**, **non-pair-specific**
(pair-agnostic definition), and — because the eligibility rule
`event_class == supersession` is the *first* key — **non-constant on the
denominator** (a constant restates eligibility; it refines nothing).

Since `event_class` and (empirically, verified by the scan)
`payload_label` are constant on the denominator, every candidate reduces to
a predicate over `(epoch, batch_index)`.

**Enumerated candidates** (scored):

| id | predicate | mechanism hypothesis |
|---|---|---|
| K0 | `true` (act on every eligible event) | none — this is the first key restated. **Excluded by the non-constancy rule**; reported anyway because its acting outcome is already committed (PR-7 step 5 `refuse`, `twin_delta_refuse.json`): the reference point every refinement must beat. |
| K1 | `epoch == first eligible epoch` | the only self-transporting predicate: on the first eligible epoch — and only there — the acting run's pre-write state is provably identical to the committed desk state (the §9A 32/192 finding). Acting later epochs is conditioned on earlier actions. |
| K2 | `epoch > first eligible epoch` | complement of K1 (formal closure; no independent mechanism). |
| K3 | `epoch == last eligible epoch` | distinguished position; no mechanism statable (harm accrues during the window; acting last prevents least). Listed to close the distinguished-position family. |
| K4 | `batch_index == 0` | distinguished position; no mechanism (within-batch order is data-loader order under the seed). Listed for closure. |

**Blocked family F-blocked** (the *natural* candidates — enumerated,
forbidden, not eligible for GO): `payload_cos_incumbent <= 0` (sign test —
parameter-free, canonical zero), `pre_sim >= effective_vigilance` (the
vigilance test itself), `payload_label != incumbent dominant label` (the
semantically meaningful "this write changes the slot's meaning" test).
Every member reads memory-state-derived columns → the certified
contamination class in §1. This is the honest headline of the enumeration:
**the discriminating candidates all live in forbidden columns.** The scan
reports their committed-run distributions as *diagnostics only* (valid only
up to the first action of any hypothetical acting run), to document
whether they would have discriminated even if admissible — PR-5's committed
finding (B/E geometrically indistinguishable from benign A/C) predicts no.

**Excluded family F-arbitrary** (enumerable, not scored, not eligible):
all remaining parameter-free binary predicates over the admissible columns
— parities, equalities to non-distinguished positions, unions thereof, and
any pair-agnostic function of the nominal `payload_label`. Excluded by
rule, twice over: (i) they carry no mechanism hypothesis, so selecting one
*by its score* is threshold tuning by enumeration (forbidden); (ii) a
predicate on a nominal coordinate that happened to separate D/E from A/B
would encode those pairs' identities — a pair-specific rule in effect. No
score could make an F-arbitrary member eligible, so scoring them is
omitted by design (and §5's invariance lemma covers them regardless).

**K-null**: declared the sole candidate if the admissible list above turns
out empty. (It is not empty: K1–K4 satisfy the form constraints.)

## 3. Target harms

§9B may target only **write-linked residual harm on the merge/supersession
path** — harm carried by denominator events. Committed evidence
(3-seed sums, frozen probe, `none` policy, from `twin_delta_refuse.json`
baselines):

| pair (stale-soft) | frozen_probe_broken | stale_wrong | eligible events | assignment |
|---|---|---|---|---|
| D | 338 | 1019 | 576 | **target** (direct harm) |
| E | 138 | 1164 | 576 | **target** (collateral harm) |
| A | 0 | 1126 | 576 | **write-clean** (flagged, no damage) |
| B | 1 | 1076 | 576 | **write-clean** (flagged, no damage) |
| clean A/C | 0 | 0 | 0 | write-clean (no eligible events) |

Post-PR-10 residuals that motivate §9B at all (committed,
`pr11/adjudication_scan.json`): pairD soft residual stale-wrong 151
aggregate (83/38/30 by seed); pairE soft 14 (12/1/1); plus the
frozen-probe broken rows above, which **no reader policy can remove**
(they are memory corruption, not served answers).

**Explicitly out of §9B's reach — by design, not by choice:**

- **one-shot ambiguity**: observe-only, never quarantine-eligible
  (PR7_DESIGN §12 / PR-8 §9A) → not in the denominator;
- **contradiction forks**: `event_class == contradiction` is not the
  eligible class (`GOVERN_QUARANTINE_EVENT_CLASS = supersession`) → not in
  the denominator;
- **everything read-time**: PR-10 `merge-abstain` immutable; PR-11.1's
  killed policy space stays closed.

**Read-side ceiling (committed, binding context for §6):** acting on 100%
of pairD's eligible events (K0 = PR-7 refuse) removed only ~33% of broken
(338→227) and ~29% of stale-wrong (1019→719) while consuming all 576
write-time capture events (capture 576→0, verdict `needs_review`). Any
refinement K ⊂ K0 acts on fewer events; its gross benefit is bounded by
this ceiling.

## 4. Scoring (pre-registered definitions)

For each scored candidate K, over the deduplicated denominator:

1. **Fired set** — the subset of the 192 protocol positions per run where
   K = 1; reported per (pair, seed) cell.
2. **Capture** — fired fraction on each **target** pair (D, E), 3-seed
   aggregate: `fired / 576`. (The user-specified "capture of target
   residual write-events": every eligible event on a target run is a
   residual-harm-linked write event — per-event finer attribution is
   impossible from committed artifacts and any claim otherwise would be
   outcome leakage.)
3. **False-action exposure** — fired fraction on each **write-clean**
   stale-soft pair (A, B), 3-seed aggregate: `fired / 576`; plus fired
   count on clean runs (structurally 0 — no eligible events; reported for
   completeness).
4. **Capture destruction** — fired fraction per run, reported as
   write-time capture consumed (committed PR-7 fact: every acted event
   consumes its capture 1:1).
5. **Pair/arm breakdown** — the full per-cell table.
6. **Redundancy vs PR-11.1** — whether K acts on the same event class the
   failed read-time policies triggered on (all candidates act on subsets
   of the merge-suspect class that P2 `merge-support-abstain` keyed on;
   K0 additionally *is* the committed refuse arm).
7. **Tuning exposure** — whether K would require any threshold movement or
   post-hoc selection to be stated (by construction of §2: no for K1–K4;
   the F-arbitrary exclusion exists because *selection itself* would be
   tuning).

**Invariance lemma (pre-stated, verified by the scan before scoring).**
If the multiset of admissible coordinates `(epoch, batch_index)` is
identical across all 12 stale-soft runs — which §9A certification already
implies (192 events, 6 epochs × 32 batch rows, every cell) — then every
pair-agnostic candidate fires on the **same** subset of protocol positions
in every run, hence

> capture(D) = capture(E) = false-action(A) = false-action(B) = |fired|/192
> — one number per candidate, identical on every cell.

The scan verifies the premise empirically (multiset equality across all
runs and both constancy claims) and reports the lemma as
`holds / does-not-hold`. If it holds, no pair-agnostic candidate can
satisfy §5's two bounds simultaneously, and the failure is **exhaustive
over the entire admissible class** (including F-arbitrary), not merely
over the enumerated members.

## 5. Gate (pre-registered bounds and verdict mapping)

A candidate **passes** only if all of:

- **form**: binary, parameter-free, write-time available, non-leaking,
  non-pair-specific, non-constant on the denominator (§2);
- **capture bound**: capture ≥ 0.5 on **each** target pair (D and E),
  3-seed aggregate (i.e. ≥ 288/576) — the PR-11 precedent floor;
- **false-action bound**: false-action ≤ 0.05 on **each** write-clean
  stale-soft pair (A and B), 3-seed aggregate (i.e. ≤ 28/576) — the PR-11
  precedent ceiling; clean runs must be 0 (structural).

**Verdict mapping** (fixed before scoring; exactly one is emitted):

- `second-key-candidate` — ≥ 1 candidate passes form + both bounds.
- `second-key-failed` — the admissible candidate list is non-empty but no
  member passes the bounds.
- `second-key-absent` — the admissible candidate list is empty (only
  K-null remains).

Bounds are not movable after scores are seen; there are no per-pair
exceptions; a candidate passing on one target pair but failing the other
fails.

## 6. Consequence (pre-registered)

- `second-key-candidate` permits a **future** §9B acting-arm
  pre-registration. It does **not** permit implementation inside this PR.
- `second-key-absent` or `second-key-failed` **closes §9B** unless new
  write-time observables are introduced under a new pre-registered
  experiment (a schema change to `fork_events.csv` / the ledger, with its
  own inertness proof).
- **Acceptance of the residuals remains an allowed outcome** in every
  branch.

## 7. Procedure

`benchmarks/pr9b_second_key_scan.py` (analysis-only; stdlib only; reads
only the §1 artifacts; runs on darwin; no engine import, no cache, no
gentoo):

1. Load and deduplicate the denominator; **verify** counts
   (12 × 192 + 6 × 0), per-run intrinsic-key uniqueness, `event_class` /
   `payload_label` constancy, and the lemma premise (coordinate-multiset
   equality across runs). Any deviation → stop condition, no verdict.
2. Score K0–K4 per §4; evaluate §5 bounds.
3. Compute F-blocked diagnostics from the shadow `fork_events.csv`
   supersession rows (per-pair `payload_cos_incumbent` distribution and
   sign-test fired fractions), labeled diagnostic-only.
4. Extract the committed citations programmatically (refuse deltas,
   PR-11.1 residuals, baseline harm table) — no hand-copied numbers.
5. Emit `results/issue_failure_mode_blindness/pr9b/second_key_scan.json`
   and the §8 results below; apply the §5 mapping; end the memo with the
   single verdict line.

Pinned by `tests/test_pr9b_second_key.py`: the admissible-column set
equals the §9A-certified intrinsic set; the gate is unsatisfiable for any
lemma-invariant candidate (property test over all fired fractions); the
verdict mapping; the committed scan artifact's verdict.

---

*Sections above this line are the pre-registration and were committed
before the scan ran. Sections below are appended by the scan run.*
