# PR-10 step 2 — read-time abstention panel (result): **`readout-certified`**

Executes PR10_READTIME_ABSTENTION_GATE.md §2/§7 step 2. The acting readout
twin serves the frozen scorer's `merge-abstain` policy —
`{answer | abstain(merge_suspect_led)}` — and reproduces the PR-9 envelope
**exactly** on every cell, perturbing nothing else. Verdict from the
registered reader (`benchmarks/pr10_readout_delta.py`, analysis-only, no
torch): **`readout-certified`**, 92/92 cells pass all five gates
(`pr10/readout_delta.json`).

## What ran

`pr10/pr10_step2_run_matrix.sh` on gentoo (cache `feature_cache_vitl14/`,
torch 2.10.0+cu128): 93 governed runs — full A–E grids (six arms × seeds
0–2 = 90) + two pairA jitter anchors + the kept G5 same-seed twin
(stale-soft pairD s0). Governed arms ONLY; the committed PR-4/PR-3c
baselines were never regenerated (gate §6). Prologue gates both passed
before the matrix:

* **protocol byte-identity (seam OFF)**: the merged driver with no
  `--read-govern` flag reproduces the pinned PR-3c `mixed_s0` pairA/pairB
  CSVs bit-exactly on gentoo — the seam's presence in the binary changes
  nothing when inactive;
* **G5 same-seed stability**: the governed pairD/soft/s0 twin is
  byte-identical to its sibling on every artifact.

Manifest: `pr10/panel_manifest.json` — 92 cells (68 fresh with
`envelope_cell` exact-count checks against `pr9/abstention_envelope.json`
`cells_fresh`; pairA grid, pairB-s0 basics, and jitter as report-only
anchors, gates enforced but no envelope naming). Analysis ran on darwin
over the transferred artifacts (both-host discipline: gentoo computes,
darwin scores).

## Result

* **G1** write-stream byte-identity: pass on every cell —
  `fork_events`/`per_slot` byte-identical, `topk` content-identical,
  `summary.json` identical after removing the `read_govern` block.
* **G2** answered-stream byte-identity: pass on every cell — dropping the
  two appended columns (`served_outcome`, `abstain_reason`) recovers the
  committed baseline CSV byte-for-byte, abstained rows included.
* **G3** abstention-set exactness: pass on every cell. The served abstain
  set equals the frozen scorer's M-led row set recomputed from the run's
  own artifacts, and equals the envelope count on all 68 fresh cells
  (pairD/soft/s0: 300 == `abstained_merge` 300). **Zero abstentions on all
  74 non-soft cells** (clean/contra/stale/mixed/one-shot/jitter).
* **G4** trigger purity: pass — `merge_suspect_led` on every abstained
  row, empty elsewhere; no forced/tie/other reason exists.
* **G5** determinism: pass — same-seed governed twin byte-identical;
  per-artifact sha256 recorded in `readout_delta.json`.

Soft-arm abstentions served, per pair (s0/s1/s2 — the envelope's
`abstained_merge`, reproduced exactly): pairA† 374/368/380, pairB
373/340/361, pairC 304/299/285, pairD 300/280/296, pairE 387/394/377 —
**5,118 total, all on soft arms** († = report-only anchors). Capture and
false-abstention economics are unchanged from PR-9.1(b) §5 by G3 exactness:
capture floors (min over seeds) pairC 1.000 / pairB 0.994667 / pairE
0.969466 / pairD 0.778667; false-abstain ceilings 2/0/6/8 per run (worst
rate 0.327% of correct traffic); changed answers 0 everywhere.

## Cross-architecture note (documented limitation, not a seam defect)

An additional, beyond-the-reader replication was attempted: the full matrix
re-run on darwin (arm64 CPU, torch 2.10.0) against the same cache. It
**failed the seam-OFF prologue**: darwin's `mixed_s0` pairA re-run differs
from the pinned PR-3c CSV in float tails (e.g. `top1_top2_margin`
0.008692443… vs 0.008692204…) and — decisively — in row count (2550 vs
2531): sub-ULP accumulation differences compound through the vigilance gate
into different write trajectories. Because the prologue contains no
`--read-govern` flag, this divergence pre-exists PR-10 and applies equally
to the entire PR-2..PR-10 protocol: **run artifacts are canonical on
gentoo's stack; darwin's role is analysis recomputation and byte
verification of transferred artifacts**, which is what G5 implements
(same-host twin + hash manifest + darwin-side scoring). Recorded here so
no future PR mistakes cross-architecture run replication for a regression
— and as a standing warning that a gentoo torch/hardware upgrade will
break byte-comparability with committed baselines and must be treated as a
re-baselining event.

## What `readout-certified` means (gate §4, verbatim scope)

Merge-abstain is a **certified opt-in served readout** at exactly the PR-9
envelope costs — the first reader-facing governance contract this program
has promoted. It changes nothing about deployed `forward()`; it claims
nothing under drift, re-embedding, or other encoders (all evidence
stationary, one encoder); it does not address the residual stale-wrong rows
whose top-1 is not the merged slot (pairD/s0: 83 rows, 22.1%) nor one-shot
ties or contradiction forks — those are the named targets of **PR-11**
(adjudication-window design), and the parallel audit track remains
**PR-9.2** (§9A shadow certification with a write-event-intrinsic identity
key; see `pr8/identity_smoke/`).

## Files

* `pr10/pr10_step2_run_matrix.sh` — the gentoo matrix (prologue gates + 93 runs)
* `pr10/governed/` — 93 governed runs × 5 artifacts (gentoo-computed)
* `pr10/panel_manifest.json` — 92-cell manifest (68 fresh + 24 anchors)
* `pr10/readout_delta.json` — the registered reader's verdict (this memo's
  every number is recomputable from it)
