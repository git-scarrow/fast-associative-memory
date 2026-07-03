# FAM Governance Escalation Sequence — Closure Memo

Date: 2026-07-02. Main @ `1859044`. Documentation only: this memo opens no
implementation PR, adds no policy variant, moves no threshold, and changes
no schema. It records where the pre-registered escalation sequence ended,
what is and is not governed, and the recommended next fork.

**Scope of every claim below.** All negative results are scoped to the
committed artifacts, the committed panel protocol (vision, 12 epochs,
supersede-epoch 6, soft-payload construction), and the currently logged
write-time observables. Nothing here claims universal impossibility, and
nothing here claims FAM solves memory governance. What is claimed is that
the pre-registered hypothesis space *on the current evidence* has been
exhausted, with each closure carrying its own committed falsification.

## 1. What is certified: merge-abstain is the only reader contract (PR-10)

`--read-govern merge-abstain` — the reader serves {answer |
abstain(merge_suspect_led)} — is FAM's only certified reader-facing
contract (`readout-certified`, main `76fe7eb`; 92/92 cells pass gates
G1–G5). Its envelope is exact, not estimated: capture floors
C 1.000 / B 0.9947 / E 0.9695 / D 0.7787; worst false-abstain 0.327%;
5,118 abstentions all on soft arms, 0 on all 74 non-soft cells; 0 changed
answers; abstention set equal to the frozen scorer's merge-suspect set.
Run artifacts are gentoo-canonical (recorded cross-architecture FP
limitation); the engine is byte-frozen. This contract is immutable in
every subsequent branch of this memo.

## 2. Read-time expansion and pending-window enforcement: rejected (PR-11.1)

Verdict `negative` (main `8569ff0`), on the current artifacts. Every
candidate expansion passed its capture gate and failed its pre-registered
5%-of-correct-traffic false-abstention ceiling: adjudicated-abstain worst
0.582 (1,470 of 2,747 rows abstained, 1,024 of them correct);
pending-abstain worst 0.707; merge-support-abstain failed on the
compressed pairs D (0.095–0.142) and E (0.068–0.101) while passing B/C.
The pending-window hypothesis was refuted independently of any policy:
router resolution lag is ≤ 1 epoch everywhere (median 1.0, max 1), contra
harm is 86% post-resolution, and the only genuinely pending harm class
(one-shot, 79% pre-resolution, ties that never resolve) has a trigger
costing 33–36% false abstention. The finding: router evidence *localizes*
(fork sets name the right slots) but does not *discriminate* (acting on
those slots punishes correct traffic); merge-suspect-led abstention is
unique in doing both. Timing is not the bottleneck — granularity is.

## 3. Write-event authority: rejected (PR-9B)

Verdict `second-key-failed` (main `1859044`), on the currently logged
observables. §9A certification (PR-9.2, `identity-certified`) supplied the
event-addressable denominator; the desk scan then showed §9B's required
second key cannot exist within it: the admissible state-free columns
reduce to protocol coordinates (`epoch`, `event_class`, `batch_index`,
`payload_label`), and on the certified denominator two are constants while
the `(epoch, batch_index)` multiset is identical across every pair, arm,
and seed. Every pair-agnostic parameter-free key therefore has capture
equal to false-action — one number per key — and no number clears a
0.5 floor and a 0.05 ceiling at once. The forbidden columns would not have
helped: the per-event `payload_cos_incumbent` distribution is identical to
six decimals across all four pairs, and both natural binary tests are
constants on the denominator. The committed acting ceiling bounds any
would-have-been benefit: total diversion (PR-7 refuse) removed only ~33%
of broken rows while consuming all 576 write-time capture events.

## 4. Characterization: the residual harms appear trajectory-level

*This section is inference; §§1–3 are committed results.* Five independent
committed measurements point the same way:

- PR-5: static write-time geometry cannot separate hazardous pairs from
  benign ones (B/E indistinguishable from A/C).
- PR-6: write-time capture is geometry-stable on every arm (192/seed
  identical); read-time damage nonetheless differs by an order of
  magnitude across arms.
- PR-7: acting on 100% of eligible events removes only ~30% of read-time
  harm — most harm is not attributable to the acted events individually.
- PR-11.1: read-time fork evidence localizes without discriminating;
  resolution timing is not the failure axis.
- PR-9B: at per-event resolution, every write-time observable is either
  hazard-blind (identical distributions across harm and benign) or
  state-contaminated.

The consistent picture: harm on this protocol is an emergent property of
the *trajectory* — payload construction × accumulation across epochs ×
how subsequent reads traverse the resulting geometry — and is not present
in any single write event's observables at the time of that write. The
key assumption this characterization rests on: the committed panel
protocol (one eligible batch per epoch, fixed-cosine soft payloads) is
representative of the deployment write pattern. A materially different
write protocol could relocate information into event-level observables;
that is precisely why no universal impossibility is claimed.

## 5. The current FAM-core governance envelope

**Governed** (certified, exact): registered soft/blend merge-path cases
caught by merge-suspect-led abstention — the PR-10 contract of §1, at its
recorded capture floors and false-abstain ceiling.

**Not governed** (measured, uncovered):

| residual | committed magnitude (3-seed) | why uncovered |
|---|---|---|
| pairD stale-soft rows surviving abstention | 151 rows (83/38/30) | merged slot present in support but outvoted; every discriminating expansion failed its FP gate (PR-11.1) |
| pairE stale-soft rows surviving abstention | 14 rows (12/1/1) | same |
| frozen-probe broken rows on D/E | D 338, E 138 | memory corruption — no reader policy can remove it; write-side keys hazard-blind (PR-9B) |
| contradiction forks | 4,491 contra-wrong none-rows (B–E) | not quarantine-eligible by design; contra-led read policies failed FP gates; router adjudicates (~208/235 pairs) but adjudication-led action punishes correct traffic |
| one-shot ambiguity | 3,834 wrong none-rows; 32 permanently ambiguous pairs per cell | observe-only by design (PR-7 §12 / PR-8 §9A); its abstention trigger costs 33–36% false abstention |
| freshness / tombstone / retention behavior | not measured | outside the scope of every certified artifact; no certified behavior exists |

## 6. Paths forward

Both paths below are permitted by the pre-registered consequences already
on record (PR-11.1 §6, PR-9B §6); neither is an implementation step today.

**(a) Accept the residuals; move governance up into an external
constructive-forgetting harness.** FAM-core remains what the evidence
shows it is: an evidence-emitting, abstention-capable memory component
with one certified contract. Deletion, freshness, tombstones, retention
windows, and trajectory-level repair become the responsibility of a
harness *around* the core — the layer where lifecycle semantics naturally
live, and the only layer positioned to see the trajectory that §4 says
carries the harm. The design work is a boundary specification: what
FAM-core exports (abstain reasons, fork verdicts, the certified event
ledger, router state), what the harness owns, and what invariants the
boundary preserves (engine byte-frozen, contract untouched).

**(b) Open a new-observables pre-registration.** A new instrumentation
experiment logging richer write-time geometry than the current
single-scalar nearest-incumbent view (the only geometric observable the
driver captures per event today), under the established discipline: schema
extension to `fork_events`/the ledger, an inertness proof before any
scoring, a stated mechanism hypothesis, and a falsifiable gate fixed in
advance. The honest prior from committed evidence is unfavorable — PR-5,
PR-6, and PR-9B's blocked-family diagnostic all found no event-local
signal in the geometry actually measured — so a credible pre-registration
must state *why* a richer observable would differ where
`payload_cos_incumbent` does not (the decision-relevant unknown this memo
leaves open).

These are not mutually exclusive over time: (a) does not foreclose a
later (b) if harness design surfaces a concrete mechanism hypothesis.

## 7. Recommended fork

**Accept residuals and design harness boundary.**

Rationale: the residuals are measured, bounded, seed-stable, and sit on a
certified denominator — acceptance here is a quantified engineering
decision, not resignation. All three intra-core escalations died on
pre-registered false-positive gates for the same structural reason (harm
is not localizable at the granularity each mechanism acts on), so
continuing inside the core means searching a space with no surviving
mechanism hypothesis. Meanwhile three of the ungoverned rows in §5 —
one-shot ambiguity, contradiction adjudication, and the entire
freshness/tombstone column — are lifecycle concerns that only a
trajectory-aware layer can own. Designing the harness boundary is the
next falsifiable step that uses everything this sequence certified
(the contract, the event ledger, the router evidence) without disturbing
any of it; the new-observables pre-registration remains open as the
fallback if the boundary design uncovers a mechanism that genuinely needs
event-level write-time signal.
