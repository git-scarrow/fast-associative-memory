# PR-12.2 — Dual-presentation economics for never-resolving ambiguous pairs (pre-registration DRAFT)

Date: 2026-07-05. Main @ `fdd83a3`. **Status: DRAFT — not committed, not
scheduled; awaiting explicit review and authorization before any branch,
commit, or implementation.** Harness-track only. This draft defines every
gate in full; nothing is inherited from PR-12.1 by reference or by
interpretation. PR-12.1's verdict is `reshape-negative` and stands
unmodified: its gates, thresholds, outputs, and §9 are untouched by this
document, and no PR-12.1 candidate is promoted or revived here. PR-10's
merge-abstain remains the only certified reader contract.

## 1. Hypothesis

Pending-led escalation — the disposition the PR-12.1 scan measured as the
*sole* residual suppressor once quarantine-led escalation is removed
(pending-led rows on correct traffic: 242 of 2,136 pairD/contra, 256 of
2,476 pairB/contra, 35 of 2,449 pairD/stale-soft; committed in
`pr12_1/reshape_scan.json`) — can be replaced, for served answers led by
**never-resolving ambiguous pairs**, with an explicit two-candidate
dual-presentation that (a) keeps total suppression on correct traffic
within a 5% per-cell ceiling, (b) dual-presents **selectively** — no
more than 5% of correct traffic per cell (§5 G-M, the primary economics
gate) — (c) preserves adverse-state visibility as defined in §6, and
(d) converts wrong single answers into candidate sets that contain the
truth at a pre-registered floor (§5 G-T, a sanity gate). The mechanism
basis is committed: PR-11.1 §3 measured that fork pairs either adjudicate
at the next epoch boundary or never do (resolution lag median 1.0, max
1), and that the permanently-ambiguous class is the one-shot signature
(32 pairs per one-shot cell; 79% of one-shot wrong-none harm occurs while
those pairs are unresolved).

## 2. Target mechanism

PR-12 boundary-memo **mechanism (d)** only: the disposition of served
answers whose surviving top-1 slot is in the router's `ambiguous` set at
the probe's epoch (`led_pending_ambiguous`). Out of scope and frozen at
the PR-12 prototype disposition: mechanism (a) merge-support caveats,
mechanism (c) quarantine-led escalation (PR-12.1 `reshape-negative`
means C1/C2 were **not** promoted; the baseline here is the prototype
shape, not C1), superseded-led withholding, and the PR-10 abstention
pass-through. FAM-core is untouched at every layer: engine, driver,
scorer, thresholds, committed artifacts.

## 3. Allowed intervention family

Parameter-free disposition shapes for pending-led served answers,
computed from evidence the committed artifacts already carry:

* **D1 — dual-present-all.** Every pending-led served answer compiles as
  a two-candidate dual-presentation: the deployed answer plus the fork
  counterpart's decode class(es) (from the router pair's
  `old_side`/`new_side`, already computed by the harness), each with
  provenance, under an explicit `unresolved-tie` notice type. Zero
  pending-led suppression.
* **D2 — age-gated dual-presentation.** The never-resolving classifier,
  state-free and lookahead-free: a pair whose verdict is still
  `ambiguous` at any epoch strictly greater than its onset epoch is
  classified never-resolving (the committed lag bound — max 1 — makes
  this exact on the panel protocol: anything unresolved one boundary
  after onset never resolves). Pending-led rows whose leading pair is
  classified never-resolving dual-present as in D1; rows led by a pair
  still inside its first epoch (genuinely undecided) keep the prototype
  escalation.

The scan runs shapes `{prototype, D1, D2}` — prototype is the baseline
and must be a byte-no-op (§8).

## 4. Forbidden interventions

* Any change to FAM-core files, thresholds, or committed baselines.
* Any change to the disposition of non-pending-led rows (mechanisms (a),
  (c), superseded, abstention pass-through) — candidate diffs must be
  confined to `led_pending_ambiguous` rows.
* **Lookahead / oracle use:** classifying a pair by consulting verdicts
  at epochs later than the probe's epoch. D2's classifier may use only
  the pair's onset epoch and the current epoch. (Registry labels remain
  scoring-only, exactly as in every committed scorer.)
* Reviving any PR-12.1 candidate (C1/C2/C3) in any form, alone or
  composed — that requires its own new pre-registration.
* New observables, new FAM-core-facing schema, new reason codes beyond
  the existing `led_pending_ambiguous` (dual-presentation is a
  disposition + item shape, not a new audit vocabulary).
* Threshold motion or gate re-scoping after the first scan run.
* Blending: a dual-presented item must carry the candidates as separate
  entries; no code path may merge them into one asserted answer.

## 5. Exact pass/fail gates (all hard unless marked report-only; no discretion at scoring time)

Panel (all committed inputs; missing input = kill condition §9):

| cell | run stem (`pr10/governed/`) | hazard source | role |
|---|---|---|---|
| clean/pairA/s0 | `per_probe_clean_pairA_s0` | `pr3c/per_probe_clean_s0.governance.json` | control |
| pairD/stale-soft/s0 | `per_probe_stale-soft_pairD_s0` | `pr6/stale_de/per_probe_stale-soft_s0_pairD.governance.json` | continuity |
| pairD/contra/s0 | `per_probe_contra_pairD_s0` | `pr4/pr4_geometry_table.json#governance#pairD/contra/s0` | pending burden |
| pairB/contra/s0 | `per_probe_contra_pairB_s0` | `pr3c/per_probe_contra_s0_pairB.governance.json` | pending burden |
| pairD/one-shot/s0 | `per_probe_stale-oneshot_pairD_s0` | `pr4/pr4_geometry_table.json#governance#pairD/oneshot/s0` | harm class (32 ambiguous pairs) |
| pairB/one-shot/s0 | `per_probe_stale-oneshot_pairB_s0` | `pr4/pr4_geometry_table.json#governance#pairB/oneshot/s0` | harm class (32 ambiguous pairs) |

* **G-S — suppression ceiling (per cell, every cell).** Suppressive
  dispositions (`escalated` + `withheld`) on correct traffic, divided by
  correct traffic (`n − wrong_none`), ≤ **0.05**. Stated here in full as
  this pre-registration's own bound (numerically the program precedent;
  adopted, not inherited). Clean control must be 0 structurally.
* **G-M — presentation-mass ceiling (per cell, every cell; the PRIMARY
  economics gate).** Dual-presented items on **correct** traffic,
  divided by correct traffic (`n − wrong_none`), ≤ **0.05**. A
  dual-presentation replaces a correct single answer with a
  two-candidate notice — a real cost to the reader even though nothing
  is suppressed — so a candidate may not pass by brute-force
  dual-presenting broadly. Pre-registered prediction from committed
  data: pending-led correct mass is 242/2,136 = 11.3% (pairD/contra)
  and 256/2,476 = 10.3% (pairB/contra), so **D1 is expected to FAIL
  G-M on both contra cells** — D1 is retained as the recorded
  brute-force control, and D2's never-resolving selectivity is the
  genuine candidate under test. Clean control must be 0 structurally.
* **G-T — truth-containment floor (per harm-class and pending-burden
  cell; a SANITY gate, secondary to G-M).** Over pending-led **wrong**
  rows that a candidate dual-presents: the fraction whose true label is
  among the presented candidate classes ≥ **0.5** (aggregate per cell;
  registry used for scoring only). Triviality analysis, recorded in
  advance: G-T is *not* structurally trivial in general — the presented
  classes are pair-derived (`old_side`/`new_side`), not
  registry-derived, so a wrong row led by a pair whose parties do not
  include the true class fails containment (collateral-led rows on the
  contra cells are the expected failure population). On the one-shot
  cells, however, the protocol constructs fork pairs between the true
  and injected classes, so containment there is plausibly
  near-saturated and a pass is weak evidence. G-T therefore gates
  against a candidate that dual-presents the *wrong pairs* (a genuine
  failure mode), but a G-T pass carries no economics weight of its own
  — selectivity (G-M) is where a candidate earns a GO. Per-cell G-T
  values are reported alongside the saturation caveat in the results
  section.
* **G-V — adverse-state visibility.** §6, all four clauses, per cell.
* **G-R — regression.** (i) PR-12 continuity anchors on
  pairD/stale-soft/s0: 300 certified abstentions; stale-wrong
  375 = 292 + 83 with zero escapes. (ii) Non-pending dispositions
  byte-equal in counters to the prototype shape on every cell.
  (iii) Committed `pr12/` and `pr12_1/` outputs byte-untouched;
  prototype shape a byte-no-op (§8). (iv) Contradiction-pair review
  queue identical to prototype in set and payload.
* **E — report-only (never gates):** dual-presentation volume on correct
  traffic per cell; wrong-in-prompt exposure by flag class ×
  (dual-presented / caveated / unmarked); unmarked exposure must be
  *reported against* the prototype but is gated only via §6 V2′.

**Verdict vocabulary (exactly one):** `pending-evidence-GO(<candidates>)`
— at least one candidate passes G-S, G-M, G-T, G-V, G-R on every cell;
`pending-negative` — none does; `pending-blocked` — instrumentation
contradiction (§9). **Scope of every verdict:** offline-simulator reshape
evidence only — never runtime prompt safety, never policy promotion,
never an extension of any FAM-core certification. **Downstream-use
boundary, stated in advance:** no PR-12.2 verdict — `GO` included — may
be claimed as direct suitability for agent prompting, for promotion to
any policy version, for memory ingestion or write-back of dual-presented
content, or for any autonomous downstream use of the emitted packets.
Each of those is a separate claim requiring its own pre-registration
with its own evidence; a PR-12.2 `GO` may be *cited* by such a
pre-registration, never *substituted* for it.

## 6. Adverse-state visibility requirements (G-V clauses)

* **V1′ — audit basis retained.** Every pending-led served row keeps
  `reason_code = led_pending_ambiguous` with its evidence pointer; zero
  downgrades to `no_adverse_flag`, regardless of disposition.
* **V2′ — prompt marker.** Every pending-led item that compiles is
  marked: either the prototype `unresolved_notice` or the
  dual-presentation `unresolved-tie` item type. No pending-led answer
  may compile as an unmarked single answer; unmarked wrong-in-prompt
  must not exceed the prototype's count on any cell.
* **V3′ — ambiguous-pair review queue.** A new audit-only record per
  final-epoch ambiguous pair (`ambiguous_pair_review`), disposition-
  shape-invariant, carrying: pair identity (`I`/`O` slots, onset epoch,
  decode sides), never-resolving classification (D2's classifier,
  recorded for every shape), per-side led-row counts, stable
  `(epoch, probe_index)` exemplar query IDs per side, explicit
  `no_led_rows` for a side that never leads. Candidate queues must be
  identical to the prototype's in set and payload. The existing
  contradiction-pair queue is unchanged (G-R iv).
* **V4′ — invariants.** PR-12 boundary invariants I1–I7 hold, including
  `certified`-string containment (the word appears only on the PR-10
  abstention pass-through) and the withheld ≠ not_retrieved distinction.

## 7. Prompt-safety requirements

Structural requirements on the emitted packets (checkable now), plus an
explicit scope boundary (not checkable now, therefore never claimed):

* A dual-presented item must present **both** candidates as distinct
  entries with per-candidate provenance (slot, decode class), under a
  notice text that asserts *neither* — phrasing must present an
  unresolved tie, not a ranked recommendation.
* No blending (§4); no dual-presented item may be distinguishable from
  a single asserted answer only by inspection of provenance — the item
  *type* itself must differ.
* Every dual-presentation carries `certification_tier =
  harness-heuristic`.
* The §5 E exposure report must decompose all wrong rows entering the
  prompt so a future prompt-safety study has its denominator.
* **Boundary:** whether dual-presentation is adequate protection at the
  agent runtime — whether an agent treats a two-candidate notice more
  safely than a wrong single answer — is the named decision-relevant
  unknown. No PR-12.2 verdict addresses it. Claiming it requires a
  separate pre-registration with agent-behavior evidence.
* **Downstream-use boundary (restated from §5, binding on every
  verdict):** PR-12.2 evidence is offline-simulator evidence about
  disposition economics. It confers no suitability claim for (i) direct
  agent prompting with the emitted packets, (ii) promotion to any
  harness or reader policy version, (iii) memory ingestion or
  write-back of dual-presented content into any store, or (iv) any
  autonomous downstream consumption of PR-12.2 outputs. Any such use
  needs its own pre-registration and gates.

## 8. Byte-reproducibility requirements

* The committed `pr12/` outputs (6 files) and `pr12_1/` outputs
  (49 files) must be byte-untouched; the PR-12 base byte-gate
  (`--check`) must run green **before and after** the scan.
* The `prototype` shape under the PR-12.2 panel must be a byte-no-op
  for the four cells shared with PR-12.1: identical bytes to the
  committed `pr12_1/prototype/<cell>/` files for the three per-cell
  artifacts (the two one-shot cells are new; their prototype outputs
  become the committed baseline).
* Every emitted PR-12.2 artifact must regenerate byte-identically from
  committed inputs by one command (`--shape <s>` per shape or the full
  scan runner); no normalization — determinism is structural (stdlib
  json, `ensure_ascii`, insertion-ordered dicts, sets sorted at
  emission). Byte drift is a broken determinism assumption, reported,
  never normalized away.

## 9. Kill conditions (any → `pending-blocked`; fix instrumentation, re-run; no candidate judged from a blocked run)

1. Any §5 panel input missing or failing its committed-count cross-check
   (router rebuild vs hazard source: conflict pairs, merge events, row
   counts; ambiguous-pair count vs the committed `final_epoch_verdicts`
   — one-shot cells must show 32).
2. PR-12 base byte-gate red before or after, or any byte drift in
   committed `pr12/` / `pr12_1/` outputs.
3. Prototype shape not byte-identical on the four shared cells.
4. Certified-abstain set ≠ merge-led set on any cell (G3-exactness
   self-check).
5. Any candidate diff outside pending-led rows (detected by counter
   comparison per G-R ii).
6. D2 classifier found to require lookahead on any row (implementation
   audit; §4 oracle prohibition).
7. Contradiction-pair or ambiguous-pair review queue differing across
   shapes in set or payload.
8. Any occurrence of "certified" outside the permitted fields.

## 10. Expected artifacts and filenames

* This memo, committed before any scan run:
  `results/issue_failure_mode_blindness/PR12_2_PENDING_DUAL_PRESENTATION.md`
  (results appended after as §11+, append-only).
* Harness changes (implementation step, separately authorized):
  `harness/harness_boundary_sim.py` (shapes D1/D2 + ambiguous-pair
  queue + one-shot cells), `harness/harness_policy.json` (a `scan12_2`
  block, policy version `pr12.2-scan-0.1`; the PR-12.1 `scan` block is
  untouched).
* Per (shape × cell), under
  `results/issue_failure_mode_blindness/pr12_2/<shape>/<cell>/`:
  `memory_packet.jsonl`, `audit_packet.jsonl` (per-probe records +
  `contradiction_pair_review` + `ambiguous_pair_review` records),
  `decision_table.csv`. Shapes: `prototype`, `D1`, `D2`; cells: the six
  §5 names (`clean_pairA_s0`, `pairD_stale-soft_s0`, `pairD_contra_s0`,
  `pairB_contra_s0`, `pairD_oneshot_s0`, `pairB_oneshot_s0`) — 54
  artifact files.
* Gate scoring: `results/issue_failure_mode_blindness/pr12_2/
  pending_scan.json` — every G-S/G-M/G-T/G-V/G-R check, the E report,
  both byte-gate results, and the verdict; every number in the appended
  results section must be recomputable from it.
