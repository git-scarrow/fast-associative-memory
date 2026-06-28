# PR-7 — Quarantine promotion gate (pre-registration, design-only)

**Status:** design-only. This memo *pre-registers* the criteria that would
promote the write-path `quarantine` action above its current `needs_review`
verdict. It **implements nothing**: no engine, probe, scorer, or policy code
changes; no cache run; deployed retrieval is unchanged; main is untouched. The
gate is fixed here **before** the remaining §8 quarantine arms are run, so the
acceptance criteria cannot be retrofit to favorable results — the discipline
PR-4/PR-5 violated and PR-7 §7/§8 reinstated.

It is the companion to the step-7 review (branch
`feat/pr7-step7-quarantine-merge-stale-paire` @ `33cbc33`, tagged
`pr7-step7-evidence`), which upheld `needs_review` and listed promotion's
prerequisites. This memo turns that list into a frozen, checkable gate.

## 1. Why a promotion gate, and why now

`needs_review` is a **holding state**, not a defect (PR7_DESIGN §8 step-5
addendum): an acting arm that helped and did not regress, "promising but not
certified; `pass` is reserved for a later, explicit promotion step." That step is
this memo. Two facts force the gate to be written *before* the arms it scores:

1. **The anti-tuning rule (PR7_DESIGN §7).** Margins are committed before any run
   and never re-tuned after seeing a result. A promotion criterion authored after
   pairD+pairE results, to capture pairD+pairE results, is exactly the post-hoc
   tuning the project forbids. The gate is therefore frozen here, in advance.
2. **`needs_review` is geometry-local to one cell.** Quarantine's `needs_review`
   is specifically the `merge_path_stale` capture-removal verdict (the acting-arm
   `capture_stable: null` path). On the read-time harm-shape cells
   (`direct_harm`, `collateral_harm`, `clean_control`) an acting arm is scored by
   the ordinary `improve` / `not_worsen` guards — it has **no quarantine arm run
   yet**, so those verdicts are `inconclusive`. Promotion cannot precede the
   panel.

## 2. What is already established (do not re-litigate)

Quarantine on the `merge_path_stale` cell, soft (merge-path) arm, frozen #87
readout, seeds 0/1/2, two independent geometries:

| geometry | classes (attractor) | broken Δ | stale_wrong Δ | collateral Δ (net) | capture | ledger |
|---|---|---|---|---|---|---|
| pairD (step 6) | `{10,28,32,95}` (52) | **+111** | +300 | +26 | 192→0/seed | 576 retained |
| pairE (step 7) | `{47,56,61,76}` (1) | **+31** | +40 | +5 | 192→0/seed | 576 retained |

Established and not in question here: quarantine reduces read-time broken/stale
on both geometries; removes the write-time merge-suspect capture from the active
state (192→0) **but retains all 1152 diverted writes recoverable in the ledger**
(`retained_recoverable: true`, `absorbed_into_active_memory: false`); on pairD its
active-memory effect is **identical to refuse**, with the recoverable ledger the
sole difference. pairE replicates pairD at the smaller magnitude its milder
baseline hazard predicts (PR-6: E broken_mean ≈46 vs D ≈112) — geometry-consistent,
not a pairD artifact.

## 3. The gap between "helps" and "promotable"

Three things stand between this evidence and a `pass`:

* **Panel incompleteness.** `direct_harm`, `collateral_harm` (pairB/pairE
  read-time), and `clean_control` (pairA/pairC) have **no quarantine arm**.
  `merge_path_stale` itself is run on pairD/pairE only (pairA/pairB pending).
  PR7_DESIGN §8 requires *all* of: direct improves, collateral does not worsen,
  clean does not inflate, both-shapes holds. None of those are tested for
  quarantine.
* **Recoverability is a provenance claim, not a demonstrated mechanism.** The
  ledger records *that* writes were diverted and retains their payload
  accounting; no run has shown the quarantined writes can be **re-injected or
  audited without read-time harm**. Promoting now would certify recoverability
  the harness has never exercised.
* **The win is non-uniform.** Both pairD-s2 (collateral 0→4) and pairE-s2 (7→9)
  show a per-seed collateral uptick, absorbed only by the aggregate
  `collateral_delta_total ≥ 0` guard. The aggregate guard can mask a per-pair or
  per-seed regression when another seed over-compensates.

## 4. The pre-registered promotion gate

Quarantine is promoted above `needs_review` **iff all** of G1–G7 hold. Every
margin below is frozen by this memo. The unit is the *cell* / *harm shape*, never
the geometry (geometry is never a gate — PR-5).

### G1 — Full §8 panel, run as quarantine arms

Run `--govern none` vs `--govern quarantine` twins on every §8 cell, ≥3 seeds
{0,1,2}, frozen #87 readout, engine byte-frozen:

| cell | pairs (arms) | guard | pre-registered margin (baseline − governed) |
|---|---|---|---|
| `direct_harm` | pairD | improve | broken Δ **≥ +20** total over 3 seeds, **no** seed with broken Δ < 0 |
| `collateral_harm` | pairB **and** pairE | not_worsen | broken Δ **≥ −0** total **and** per-seed broken Δ **≥ −3** each (no seed worse than 3) |
| `clean_control` | pairA **and** pairC | not_worsen | broken Δ **≥ −0** total **and** false-action (acted-on-benign) Δ **≥ −0** total |
| `merge_path_stale` | pairA, pairB (complete the cell) | capture_stable→acting | see G2 |

`direct_harm`'s `≥ +20` is the pre-registered analogue of "drains direct harm";
it is deliberately below the pairD merge-cell broken Δ (111) because the
`direct_harm` cell measures a different (read-time, non-merge) quantity and must
be earned on its own arm, not inferred from the merge cell.

### G2 — Complete `merge_path_stale` and characterize capture removal

* Run quarantine arms on pairA and pairB of `merge_path_stale` (pairD/pairE
  already committed), ≥3 seeds.
* Across **all four** geometries: aggregate broken Δ **> 0**, aggregate
  stale_wrong Δ **> 0**, aggregate collateral Δ **≥ 0**, and capture diverted =
  opportunity on every seed (the acting arm removes what it intercepts).
* Capture removal stays recorded as *intended* (`capture_stable: null`), never a
  fail — unchanged from the §8 addendum.

### G3 — Recoverability validation (the quarantine-specific criterion)

This is what `quarantine` must prove that `refuse` cannot. Pre-register an
**opt-in, design-only recovery probe** (no deployed-path change) that:

* reads the committed quarantine ledger for a scored cell;
* reconstructs the diverted (quarantined) writes from the retained payload
  accounting; and
* demonstrates, under the frozen readout, that **reinstating** the quarantined
  writes recovers the baseline capture (192/seed) **without** re-inflating
  read-time broken/stale beyond the ungoverned baseline.

Pass condition: reinstatement restores capture to 192/seed on every seed **and**
post-reinstatement broken/stale ≤ ungoverned baseline + pre-registered tolerance
(broken per-seed **≤ +0**, i.e. no worse than baseline). If reinstatement cannot
restore capture, or restores it only by re-introducing the harm quarantine
removed, "recoverable" is falsified and the ledger is provenance-only — **no
promotion**. (Whether the recovery probe is built is a later *named* step, not
this memo; the criterion is frozen now.)

### G4 — Collateral uniformity bound

Promotion requires the per-seed collateral guard, not only the aggregate:
**no seed, in any cell, may have collateral Δ < −3** (governed collateral exceeds
baseline by more than 3 on a single seed). This closes the aggregate-masking gap
the pairD-s2 / pairE-s2 upticks exposed. (Implementing a per-seed guard is a
scorer change and therefore a later step; the *threshold* is pre-registered here.)

### G5 — Both-shapes rule across the full panel

A candidate that improves one harm shape (D-like direct) while worsening another
(B/E-like collateral, or clean) **fails**, full stop (PR6 §5 / PR7_DESIGN §8.5),
evaluated over the completed panel, not a subset.

### G6 — Boundary invariants (unchanged, re-asserted as gate conditions)

Engine `associative_core.py` / `fast_associative_memory.py` sha256 unchanged;
deployed `forward()` / `learn_local` byte-identical; `--govern` stays opt-in and
unreached by deployed retrieval; geometry never a gate; both arms byte-stable
across darwin/gentoo; baseline `none` vote bit-identical to deployment. Any
breach voids promotion regardless of G1–G5.

### G7 — Anti-tuning

The margins in G1–G4 are frozen by this commit. If a run misses them, the result
is recorded as a negative (quarantine not promotable on that shape) — **the
margins are not moved to manufacture a pass** (the PR-4/PR-5 trap).

## 5. Promoted-state semantics

If G1–G7 all hold, the verdict vocabulary gains exactly one state above
`needs_review`:

* **`promoted`** — the panel passed and recoverability is validated; quarantine is
  certified *for the enumerated cells only*, with no claim on unseen geometry
  (the scope refusal PR-5 retired). `pass` remains reserved for the ordinary
  null-action / read-time guards; acting arms reach `promoted`, never a bare
  `pass`, to keep the capture-removal provenance legible.

Promotion authorizes a **deployment-proposal PR with its own gates** — *not*
deployment, and *not* any deployed-retrieval change. This mirrors PR7_DESIGN §8's
existing "passing the panel authorizes a proposal, not deployment."

## 6. Stop conditions (record a negative, do not tune)

Promotion is abandoned — recorded as a documented negative, like PR-5 recorded
geometry's failure — if any fire:

* the §8 panel cannot be passed by a parameter-free quarantine without regressing
  collateral or clean (G1/G5);
* recoverability is falsified — reinstatement cannot restore capture, or only by
  re-introducing harm (G3);
* a per-seed collateral regression beyond −3 appears on any cell (G4);
* promotion would require touching deployed engine code or admitting a geometry
  gate (G6) — out of bounds, stop and redesign.

In every stop case quarantine remains a **documented manual-review success at
`needs_review`** (its current, earned status), not a failure.

## 7. Sequencing (named, not built)

Each lands on main before the next, mirroring PR-6/PR-7 discipline:

1. **Complete `merge_path_stale`** — quarantine arms on pairA, pairB (G2).
2. **Read-time panel** — quarantine arms on `direct_harm`, `collateral_harm`,
   `clean_control` (G1), scored against the frozen margins.
3. **Recovery probe** — the opt-in, design-only ledger-reinstatement validation
   (G3); per-seed collateral guard (G4) added to the scorer at this step.
4. **Promotion scoring** — only if 1–3 all meet this memo's frozen gate, emit the
   `promoted` verdict; else record the negative.

This memo plans the line; it implements none of it.

## 8. Explicit non-goals (hard boundaries, unchanged from PR7_DESIGN §12)

No deployed retrieval change; no read-time trust revival; no static geometry
gate; no slot-granularity trust; no one-shot forced classification; no
record-granularity ledger beyond the quarantine provenance already committed; no
margin tuning to manufacture a pass; no broad class-pair search beyond the
enumerated §8 cells; **no implementation in this branch** — the gate, the recovery
criterion, and the promoted-state semantics are specified here, not written.
