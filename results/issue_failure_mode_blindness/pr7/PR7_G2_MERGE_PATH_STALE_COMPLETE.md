# PR-7 G2 — complete the `merge_path_stale` cell (all four geometries)

**Gate item:** PR7_QUARANTINE_PROMOTION_GATE.md **G2** — run quarantine arms on
pairA and pairB (pairD/pairE already produced), so `merge_path_stale` is scored
across all four geometries `{A, B, D, E}`. Reports each geometry separately, then
the aggregate G2 checks.

**Branch:** `feat/pr7-quarantine-promotion-gate` (unmerged). No engine/probe/
scorer-logic change; engine sha256 == frozen baseline; `--govern` opt-in;
deployed retrieval unchanged. No margin or quarantine-parameter tuning.

## What ran (this step)

* **pairA** `{0,8,19,33}` / attractor 71 — stale-soft, none + quarantine, seeds 0-2.
* **pairB** `{5,27,48,86}` / attractor 13 — stale-soft, none + quarantine, seeds 0-2.
* **pairE** `{47,56,61,76}` / attractor 1 — re-run here for branch self-containment
  (previously only on the archived step-7 branch `pr7-step7-evidence`), so the
  scored cell holds all four geometries on one branch.

pairD was already committed (step 6). Compute gentoo, verified darwin.

### Determinism cross-checks
* pairB none == committed `pr6/stale_de` pairB (all exts, byte-identical).
* pairE none == committed `pr6/stale_de` pairE (all exts, byte-identical);
  pairE quarantine summary == the step-7 tag `pr7-step7-evidence` (deterministic
  across branches).
* pairA raw probe outputs (csv / summary / per_slot / fork_events) byte-identical
  to committed `pr3c` stale-soft; only `governance.json` differs, and **solely by
  scorer enrichment** added since pr3c (2026-06-10): new policies
  `trust-downweight` / `trust-guarded` and fields `collateral_exposure`,
  `direct_br`, `collateral_br` (all `None`/absent in the pr3c era). `_router`
  (merge-suspect capture, conflict pairs) is identical. The probe is
  deterministic; pairA is scored by the **current** frozen detector consistently
  with every other current arm, so the delta is internally consistent.

## Per-geometry result (baseline `none` − governed `quarantine`)

| pair | shape | brokenΔ | stale_wrongΔ | collateralΔ | capture (opp → after) | worst per-seed collΔ |
|---|---|---|---|---|---|---|
| pairA | benign | +0 | +43 | +0 | 576 → 0 | 0 |
| pairB | collateral-like | +1 | +167 | +0 | 576 → 0 | 0 |
| pairD | direct-like | +111 | +300 | +26 | 576 → 0 | **−4 (s2)** |
| pairE | collateral-like | +31 | +40 | +5 | 576 → 0 | −2 (s2) |
| **agg** | | **+143** | **+550** | **+31** | **2304 → 0** | |

## G2 checks (PR7_QUARANTINE_PROMOTION_GATE §4 G2)

* aggregate broken Δ **> 0** — yes (+143).
* aggregate stale_wrong Δ **> 0** — yes (+550).
* aggregate collateral Δ **≥ 0** — yes (+31).
* capture diverted == opportunity on every seed — yes (192/seed every pair, all
  diverted; `capture_after` 0 everywhere; all 2304 retained recoverable in the
  ledger, `absorbed_into_active_memory: false`).
* capture removal recorded as **intended** (`capture_stable: null`), never a fail
  — unchanged from the §8 step-5 addendum.

**G2 is satisfied.** The cell is complete: quarantine drains stale_wrong on every
geometry, improves or holds broken, never regresses aggregate collateral, and
diverts exactly the merge-suspect capture it intercepts — retained recoverable.

## G4 per-seed audit across the completed cell

The only per-seed collateral regression beyond −3 remains **pairD s2 collΔ = −4**
(the documented blocker). pairE s2 collΔ = −2 is within the −3 bound; pairA and
pairB are flat. G2 completing the cell adds **no new** per-seed breach.

## Status (unchanged)

`merge_path_stale` verdict = **`needs_review`** (acting arm; capture consumed by
design, readout improved, aggregate collateral not regressed). Overall manifest
verdict `needs_review`. G2 strengthens the aggregate signal across all four
geometries but does **not** clear the G4 per-seed blocker and does **not**
promote. Branch stays unmerged.

Pending: **G3** (recoverability re-injection probe — the quarantine-specific
criterion) and the prepared-not-merged scorer step (clean_control pairC +
per-seed collateral guard).
