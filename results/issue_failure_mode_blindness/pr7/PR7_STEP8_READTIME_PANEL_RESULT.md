# PR-7 step 8 — read-time quarantine panel (result)

**Sequencing:** promotion-gate step 2 (PR7_QUARANTINE_PROMOTION_GATE.md §7) — the
read-time panel. Runs `--govern quarantine` twins for the three read-time §8
cells that had no quarantine arm: `direct_harm`, `collateral_harm`,
`clean_control`. `merge_path_stale` (pairD) was already committed at step 6.

**Question answered:** does quarantine remain beneficial and non-destructive
across the full hazard/control panel, or was it only validated for stale-merge
geometries?

**Boundaries (held, re-verified):** engine `associative_core.py` /
`fast_associative_memory.py` sha256 == frozen PR-6 baseline; `--govern` opt-in,
deployed retrieval unchanged; scorer (`pr7_twin_delta.py`), engine, and probe
code byte-frozen — this step only *runs arms* and regenerates the per-action
manifest. No margin was tuned (G7). Not merged to main.

## What ran

| cell | arm | pairs (classes / attractor) | seeds | twin arms |
|---|---|---|---|---|
| `direct_harm` | stale-soft | pairD `10,28,32,95` / 52 | 0,1,2 | none + quarantine |
| `collateral_harm` | stale-soft | pairB `5,27,48,86` / 13; pairE `47,56,61,76` / 1 | 0,1,2 | none + quarantine |
| `clean_control` | clean | pairA `0,8,19,33` / 71 | 0,1,2 | quarantine (committed none reused) |

Compute on gentoo (cache `feature_cache_vitl14/`), verified on darwin.

### Determinism / schema cross-checks (all byte-identical)
* `direct_harm/pairD` (none + quarantine, all seeds/exts) == committed
  `merge_path_stale/pairD` — same flags, different cell dir → identical output.
  Proves determinism, schema conformance, and no cell/path leakage in
  `summary.json`.
* `collateral_harm/none` (pairB, pairE) == committed `pr6/stale_de` none arms.
* `clean_control` none re-run == committed none (baseline invariant).
* `clean_control/quarantine` readout == `clean_control/none` (csv, topk,
  governance, per_slot, fork_events) — **quarantine is inert on clean traffic**
  (0 merge-suspect events → 0 diverted; only `summary.json` adds the govern
  provenance block).

## Per-cell result (frozen scorer, aggregate guards)

| cell | guard | brokenΔ (3-seed) | staleΔ | collateral Δ | capture | verdict |
|---|---|---|---|---|---|---|
| `direct_harm` | improve | **+111** | +300 | +26 | 576→0 | **pass** |
| `collateral_harm` | not_worsen | +32 | +207 | +5 | 1152→0 | **pass** |
| `clean_control` | not_worsen | 0 | 0 | 0 | 0→0 | **pass** |
| `merge_path_stale` | capture_stable→acting | +111 | +300 | +26 | 576→0 | needs_review |

`both_shapes_ok = True` — quarantine improves `direct`, `collateral`, and
write-capture; **no aggregate cell regresses; one per-seed collateral regression
breaches G4** (direct_harm s2, below). Overall verdict `needs_review`
(merge_path_stale acting arm holds the verdict; read-time cells pass).

**Answer:** quarantine is **not** stale-merge-only. **Quarantine generalizes
under the frozen aggregate read-time scorer** to the direct and collateral harm
shapes (drains broken +111 / +32 and stale_wrong +300 / +207, no aggregate cell
worse) and is provably inert on clean traffic. In aggregate it is a useful
read-time signal; the per-seed view (G4) carries a documented blocker.

## G4 per-seed collateral audit (pre-registered: no seed, any cell, collΔ < −3)

| cell / pair | seed | collateral Δ | G4 |
|---|---|---|---|
| direct_harm / pairD | 0 | +26 | ok |
| direct_harm / pairD | 1 | +4 | ok |
| direct_harm / pairD | 2 | **−4** | **BREACH (< −3)** |
| collateral_harm / pairB | 0,1,2 | 0,0,0 | ok |
| collateral_harm / pairE | 0,1,2 | +2,+5,−2 | ok |
| clean_control / pairA | 0,1,2 | 0,0,0 | ok |

**One breach:** `direct_harm` seed 2, collateral Δ = −4 (baseline 0 → governed
4). This is byte-identical to — and is — the already-reviewed `merge_path_stale`
pairD-s2 uptick the gate memo §3 pre-identified as the reason G4 exists. It is
masked by the aggregate guard (`direct_harm` aggregate collΔ +26) and the cell's
`improve` guard scores it `pass`. No *new* per-seed breach appeared in the
read-time cells.

## Conclusion (documented negative — not tuned)

* Under the **existing aggregate scorer**, the full panel passes
  (`needs_review` overall, both-shapes holds). **Quarantine generalizes under the
  frozen aggregate read-time scorer** beyond stale-merge and does not corrupt
  default retrieval.
* Against the **pre-registered promotion gate G4** (per-seed collateral bound
  −3), the panel **fails on one seed** (`direct_harm` s2, −4 < −3). Per gate §6
  / G7, **promotion is unavailable** and this is recorded as a documented
  negative on G4. The margin is **not** moved to manufacture a pass. Quarantine
  **is not promotable as-is on this registered panel and frozen G4 bound.**

Quarantine therefore remains at its earned status — **`needs_review`**, a
documented manual-review success — and is **not promoted**. The per-seed guard
(gate sequencing step 3) is where this breach would be formalized in the scorer;
this memo records the breach against the frozen threshold now.
