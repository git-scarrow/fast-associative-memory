# PR-9.1(b) — merge-abstain scorer change + the exact PR-10 envelope

Executes PR9_ENVELOPE_DESK_CHECK.md §3.2 (the pre-registered follow-up):
`merge-abstain` added to `benchmarks/analyze_fork_governance.py` POLICIES
as a first-class policy — **abstain iff the deployed vote's surviving
top-1 slot is merge-suspect; otherwise the `none` answer unchanged. No
exclusions, no vote recomputation, no tie trigger, no parameters** — plus
the previously-discarded `forced` flag and per-trigger abstention
counters (`abstained_merge`, `abstained_forced`) on every policy row,
strictly additive. Both governance tables re-emitted from the committed
per-run artifacts; no engine file, no benchmark driver, no run matrix
touched; no new runs.

## 0. Environment fidelity gate (step 0, before any edit): PASS

* `pr4_geometry_table.json` re-emitted with the UNMODIFIED committed
  analyzer: **byte-identical** to HEAD (sha256 `a09c9a04f597…`), headline
  H1/H2/H3 verdicts identical to PR4_RESULT.md.
* `pr3c_governance_table.json` at HEAD (sha256 `cbdda6c9a57e…`) was last
  written by the analyzer at commit `804af19` (the 3-way-H1 re-emission;
  the `d8ca6bc3…` sha in PR3C_RESULT.md §0 is the superseded `036b578`
  version). Re-emitted with that exact analyzer version against today's
  environment: **byte-identical**, and all 24 per-run `.governance.json`
  **byte-identical**. The sklearn-fit paths (frozen-detector
  reproduction, H1/H2 studies) are exactly reproducible in this
  environment (numpy 2.4.2, sklearn 1.8.0, python 3.13.5, darwin).

## 1. Regression guard (scripted, additive-only): PASS

Recursive comparison of every pre-existing `(run, policy, counter)` value
in both re-emitted tables and all 24 per-run tables against their git
HEAD baselines: **changed = 0, removed = 0** everywhere. Additions only:

* pr3c table: `abstained_merge`/`abstained_forced` on all 216 existing
  rows, the `merge-abstain` row ×24, plus the PR-4-era columns the
  current frozen scorer emits that the 2026-era table predates
  (`tie_flips`, `direct_br`, `collateral_br`, `collateral_exposure` ×216;
  `trust-downweight`/`trust-guarded` rows ×24; two frozen constants);
  `policies` list appended, never reordered.
* pr4 table: `abstained_merge`/`abstained_forced` ×990,
  `merge-abstain` row ×90. `checks`/`h1`/`h2`/`h3` bit-unchanged.

Cross-table consistency: on every cell present in both tables (pair A
grid, pair B s0 clean/contra/stale/mixed) the merge-abstain rows are
exactly equal (enforced by raise in `pr9_abstention_envelope.py`).

## 2. What the recorded triggers settle (desk-check obstruction 1)

The mode-conditioned family's abstention count now decomposes exactly,
e.g. pairD/soft/s0: `abstained 451 = 300 merge + 151 forced` (observe and
trust identical; `-abstain` adds 118 ambiguous). The desk check could
only prove |F| ≥ 1 there; |F| = 151 is now a recorded column. For
`merge-abstain` itself `abstained == abstained_merge` and
`abstained_forced == 0` on **every** cell in both tables — the policy
never excludes, so forced abstention is structurally impossible, and its
abstention set is exactly the policy-invariant M.

## 3. The envelope (merge-abstain ALONE, per (pair, seed))

Soft arms — the ONLY arms where the policy ever acts (abstentions on
every clean/contra/stale/mixed/oneshot/jitter cell in both tables: 0;
`n_merge_suspect_events` = 192 on all 18 soft cells, 0 elsewhere):

| cell | capture (stale-wrong abst / stale-wrong) | false abst (aC) | rate of correct | changed answers |
|---|---|---|---|---|
| pairA/soft/s0 † | 374/374 = 1.000 | 0 | 0 | 0 |
| pairA/soft/s1 † | 368/372 = 0.9892 | 0 | 0 | 0 |
| pairA/soft/s2 † | 380/380 = 1.000 | 0 | 0 | 0 |
| pairB/soft/s0 | 373/375 = 0.9947 | 0 | 0 | 0 |
| pairB/soft/s1 | 340/340 = 1.000 | 0 | 0 | 0 |
| pairB/soft/s2 | 361/361 = 1.000 | 0 | 0 | 0 |
| pairC/soft/s0 | 304/304 = 1.000 | 0 | 0 | 0 |
| pairC/soft/s1 | 297/297 = 1.000 | 2 | 0.078% | 0 |
| pairC/soft/s2 | 285/285 = 1.000 | 0 | 0 | 0 |
| **pairD/soft/s0** | **292/375 = 0.7787** | 8 | 0.327% | 0 |
| pairD/soft/s1 | 280/318 = 0.8805 | 0 | 0 | 0 |
| pairD/soft/s2 | 296/326 = 0.9080 | 0 | 0 | 0 |
| pairE/soft/s0 | 381/393 = 0.9695 | 6 | 0.247% | 0 |
| pairE/soft/s1 | 392/393 = 0.9975 | 2 | 0.086% | 0 |
| pairE/soft/s2 | 377/378 = 0.9974 | 0 | 0 | 0 |

† = `tainted_anchor` (PR4_DESIGN §7 leakage) — reported, excluded from
verdicts. Zero-changed-answers is field-verified on every cell (90 grid
cells + 2 jitter runs): `fixed == broken == tie_flips == 0`,
`answered == n − abstained`,
`answered_correct == correct − abstain_on_correct`.

## 4. Verdicts against the pre-registered bounds (fresh cells, 68)

| bound | outcome |
|---|---|
| (i) 0 changed answers everywhere | **PASS** (all 90 + 2 cells, field-verified) |
| (ii) soft capture ≥ 0.95 A/C, ≥ 0.79 D/E | **FAIL on exactly one cell: pairD/soft/s0 = 0.778667 (292/375), 0.0113 below the 0.79 floor.** pairC 1.0/1.0/1.0, pairD s1–2 0.8805/0.9080, pairE 0.9695/0.9975/0.9974 all pass; pairA (anchor, report-only) 1.0/0.9892/1.0; pairB (no pre-registered bound) 0.9947/1.0/1.0 |
| (iii) 0 actions on clean arms | **PASS** (0 acted, 0 abstained — on every clean cell and in fact every non-soft cell) |
| (iv) false abstentions ≤ 5% of correct traffic per run | **PASS** everywhere; worst 0.327% (pairD/soft/s0, 8 rows) |

The (ii) breach is recorded, not tuned. Context, not excuse: the 0.79
draft floor was inherited from PR-4's *trust* capture floor on pair D —
and trust's 298/375 = 0.7947 at that cell includes 6 stale-wrong rows
captured by **forced** abstentions (credit merge-abstain by construction
cannot earn), at a cost of 128 false abstentions and 112 broken rows
where merge-abstain has 8 and 0. The pure-M capture on pair D s0 is
0.7787; the 83 uncaptured rows are stale-wrong probes whose surviving
top-1 is not the merged slot (the PR-3c §2 mechanism, amplified by
pair-D compression).

## 5. Exact ceilings to pre-register for PR-10 (from the re-emitted table
— per desk check §3, NOT from the draft intervals)

Per-cell exact values live in `pr9/abstention_envelope.json`
(`cells_fresh`); the acting readout twin must reproduce them exactly on
its baseline-identical arms. Summary ceilings (fresh cells):

* **Abstention set**: per cell, abstentions == `abstained_merge` of this
  envelope (soft arms only; exact counts per cell, e.g. pairD/s0: 300).
  **0 abstentions on every non-soft arm.**
* **Capture floors (min over seeds, soft)**: pairC **1.000**, pairD
  **0.778667** (292/375 at s0 — the measured envelope replaces the
  refuted 0.79 draft), pairE **0.969466**, pairB **0.994667** (newly
  scored in PR-10).
* **False-abstention ceilings (max per run, soft)**: pairC **2**, pairD
  **8**, pairE **6**, pairB **0**; rate ceiling **0.327%** of correct
  traffic (60× inside the 5% bound).
* **Changed answers: 0. Forced abstentions: 0. Clean-arm actions: 0.
  Tie-triggered anything: 0** (the policy has no tie path; pinned by
  test).
* pairA cells remain anchors: report-only in PR-10 as here.

## 6. Files

* `benchmarks/analyze_fork_governance.py` — registered scorer change
  (`merge-abstain` + trigger recording; minimal diff, no pre-existing
  field renamed/reordered/recomputed)
* `results/issue_failure_mode_blindness/pr3c/pr3c_governance_table.json`
  + 24 per-run `.governance.json`, `pr4/pr4_geometry_table.json` —
  re-emitted, additive-only vs HEAD (scripted guard, §1)
* `benchmarks/pr9_abstention_envelope.py` →
  `results/issue_failure_mode_blindness/pr9/abstention_envelope.json` —
  reads ONLY the two re-emitted tables
* `tests/test_pr9_merge_abstain.py` — semantics fixture (incl. the
  no-tie-trigger pin), forced-flag recording, soft/one-shot integration,
  regression pins on one pre-existing + one new cell per table
