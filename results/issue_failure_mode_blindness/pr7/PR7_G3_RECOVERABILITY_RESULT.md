# PR-7 G3 — quarantine recoverability re-injection probe (result)

**Gate item:** PR7_QUARANTINE_PROMOTION_GATE.md **G3** — the quarantine-specific
criterion. `refuse` destroys the writes it blocks; `quarantine` retains them in a
recoverable side ledger. G3 asks whether "recoverable" is a real, exercised
mechanism — can the diverted writes be reconstructed and reinstated to recover
the baseline capture (192/seed) **without re-inflating read-time broken/stale
beyond the ungoverned baseline** — or whether it is a provenance-only label.

**Probe:** `benchmarks/pr7_recovery_probe.py` — analysis-only, imports no torch,
reads only committed twin artifacts (`governance.json`, the `summary.json`
quarantine ledger, `fork_events.csv`). Engine byte-frozen; deployed retrieval
untouched. Output: `pr7/recovery_validation.json`. Cell: `merge_path_stale`, all
four geometries {A,B,D,E}, 3 seeds.

**How "reinstatement" is realized (verified, not assumed).** In this
engine-frozen harness the quarantine arm differs from the `none` arm by *exactly*
the diverted supersession writes, so reinstating them reproduces the ungoverned
(`none`) run — the committed `none` arm **is** the reinstated state. This is
checked, not asserted: on every (pair, seed) `none` `absorbed` − `quarantine`
`absorbed` == the ledger's quarantined count, and the baseline's
`event_class == 'supersession'` count == it too.

## Three separable claims (never collapsed)

### 1. Reconstruction fidelity — **HOLDS** (every arm)
The ledger is a complete, lossless record of exactly the diverted writes:
`ledger.quarantined_count == baseline router merge-suspect == baseline
fork_events supersession == absorbed-count decomposition == ledger label total ==
192` on every (pair, seed); `retained_recoverable: true`,
`absorbed_into_active_memory: false`. The diverted set is independently
reconstructable from the baseline write log — the ledger loses nothing.

### 2. Capture restoration within the frozen bound — **HOLDS**
Reinstatement (= `none`) restores active-memory capture to 192/seed from the
quarantine arm's 0, and post-reinstatement broken/stale do **not exceed** the
ungoverned baseline (reinstated == baseline; the frozen numeric G3 bound
"per-seed broken ≤ baseline + 0" is met).

### 3. Harm-free recovery — **FAILS on every geometry**
The diverted writes *are* the stale supersessions, so reinstating them
re-introduces the harm quarantine removed:

| pair | reinstated stale Δ vs quarantine | reinstated broken Δ vs quarantine | harm-free? |
|---|---|---|---|
| pairA | +14 / +15 / +14 | +0 / +0 / +0 | no (stale) |
| pairB | +42 / +72 / +53 | +0 / +0 / +0 | no (stale) |
| pairD | +127 / +69 / +104 | +41 / +56 / +14 | no (stale+broken) |
| pairE | +5 / +17 / +18 | +16 / +15 / +0 | no (stale+broken) |

Even on the benign pairs (A/B, where baseline broken ≈ 0) recovery re-introduces
the `stale_wrong` quarantine drained. **Capture and harm are the same writes** on
all four geometries.

## G3 verdict: `provenance_recoverable_not_harm_free`

Recoverability is validated as **provenance / auditability** — a faithful,
complete, reversible record a reviewer can audit — but **not** as harm-free
reinstatement into active memory. Per gate §3 ("restores it only by
re-introducing the harm quarantine removed → 'recoverable' is falsified, the
ledger is provenance-only, no promotion"), **G3 does not clear quarantine for
promotion.**

This is a **second independent non-promotion reason**, distinct from the G4
per-seed collateral breach:
* **G4** — one per-seed collateral regression (pairD s2 −4 < −3).
* **G3** — recoverability is provenance-only; restoring capture re-incurs the
  harm, so the "recoverable" advantage over refuse is an *audit-trail* advantage,
  not a *reverse-without-cost* advantage.

## What this says about keeping quarantine

The finding is not "quarantine is useless" — it sharpens what quarantine **is**.
Its real, validated value over refuse is a **complete, reversible audit trail**
of the diverted suspect writes (refuse keeps only a count). That is the correct
shape for a quarantine: a *safe one-way diversion with forensics*, where a human
or later policy reviews the ledger — **not** an auto-restore store. Quarantine is
worth keeping as that, at `needs_review`. It is **not** promotable on this
registered panel: blocked independently by G3 (provenance-only recoverability)
and G4 (per-seed collateral). No margin or parameter tuned; branch unmerged.
