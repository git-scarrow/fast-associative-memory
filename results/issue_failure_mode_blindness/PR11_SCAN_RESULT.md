# PR-11.1 — adjudication-window scan (result): **`negative`**

Executes PR11_ADJUDICATION_DESIGN.md §5 exactly as pre-registered; scored
against the §6 gates with no bound moved. **Verdict: `negative` — no policy
GOes.** Every capture gate passed; every false-abstention gate failed. The
pre-registered conclusion applies verbatim: *read-time enforcement over the
existing write-time evidence cannot cover the residuals; the recorded
escalation options are §9B write-event-intrinsic authority (via PR-9.2's key)
or explicit acceptance of the residual — not threshold motion.*

Analysis-only: no engine file, no benchmark driver, no committed baseline, no
PR-10 artifact touched; no new runs; darwin only. All numbers below are
recomputable from `pr11/adjudication_scan.json` and the two re-emitted tables.

## 0. Instrumentation gates (all hard): PASS

* **Regression guard, byte-grade.** Stripping exactly the three new policy
  rows and the three appended `policies` names from the re-emitted
  `pr4_geometry_table.json` and `pr3c_governance_table.json` reproduces the
  git-HEAD files **byte-identically** under the emitter's own serialization
  (per-run `.governance.json` spot-checked the same way). Field-level sweep:
  changed = 0, removed = 0 on every pre-existing (run, policy, counter);
  additions = 342 rows (3 policies × 114 cells); `merge-abstain` rows
  identical everywhere; `policies` list append-only; every non-governance
  block (`checks`/`h1`/`h2`/`h3`/`indices`/…) identical. This subsumes the
  PR-9.1(b) step-0 environment-fidelity check: the extended analyzer in this
  environment reproduces every committed value exactly.
* **Structural pins** (test-enforced on every cell of both tables, all three
  policies): forced abstentions 0; changed answers 0
  (`fixed == broken == tie_flips == 0`); `acted == abstained`; **exact
  per-trigger decomposition** (`abstained == abstained_adjudicated +
  abstained_pending + abstained_merge_support`); clean-arm actions 0 — the
  design memo's structural prediction (clean arms carry zero router state)
  held with no exceptions.
* Scan cross-check: every per-row policy decision recomputed through the
  frozen scorer's own `apply_policy` and raise-verified against the
  re-emitted table row for its cell.

## 1. Captures: every capture gate PASSED

| gate (≥ 0.5 aggregate/pair) | pairB | pairC | pairD | pairE |
|---|---|---|---|---|
| P1 `adjudicated-abstain`, contra `contra_wrong` | 0.889 (849/955) | 0.863 (767/889) | 0.828 (1048/1265) | 0.868 (1199/1382) |
| P3 `pending-abstain`, contra `contra_wrong` | 0.999 (954/955) | 0.978 (869/889) | 0.926 (1171/1265) | 0.960 (1327/1382) |
| P3 `pending-abstain`, oneshot `wrong_none` | 1.000 (986/986) | 0.965 (655/679) | 0.948 (970/1023) | 0.988 (1132/1146) |

P2 `merge-support-abstain` captured the pairD soft residual **completely**:
151/151 aggregate, 83/83 at s0 (floors were 76 and 42) — at pairD/soft/s0 its
`stale_wrong_abstained` is 375/375, the certified policy's 292 plus every one
of the 83 residual rows. The design memo's mechanism hypothesis is
**confirmed: the merged slot sits in the surviving support of every residual
row; it is merely outvoted.** Pending-nonredundancy also passed: 3,019 of
P3's 3,743 captured one-shot wrong rows (0.807 ≥ 0.5) are reached by the
pending trigger and by neither P1's sets nor P2's trigger.

## 2. False abstention: every GO FAILED (ceiling 0.05 of correct traffic/run)

Worst rate per arm (fresh pairs; rate = `abstain_on_correct / (n −
wrong_none)`, the exact PR-9 definition; clean = 0.000 for all three,
structurally):

| policy | contra | mixed | oneshot | stale | soft |
|---|---|---|---|---|---|
| P1 `adjudicated-abstain` | **0.479** (pairD/s0) | **0.582** (pairD/s0) | **0.330** (pairE/s0) | **0.319** (pairE/s0) | **0.327** (pairE/s0) |
| P2 `merge-support-abstain` | 0 | 0 | 0 | 0 | **0.142** (pairD/s1) |
| P3 `pending-abstain` | **0.593** (pairD/s0) | **0.707** (pairD/s0) | **0.358** (pairE/s1) | **0.365** (pairE/s1) | **0.342** (pairE/s1) |

In raw terms: P1 at pairD/contra/s0 abstains **1,470 of 2,747 rows — 1,024 of
them correct answers** — to capture 1,048 wrong ones. The adjudicated sets
*localize* the harm (capture ≥ 0.83 everywhere) but do not *discriminate*:
correct traffic on fork-party slots dwarfs the harm at slot granularity. This
is PR-4's slot-granularity-trust falsification reproduced for the router's
adjudicated verdicts under the certified led-only action shape — the failure
that killed quarantine-naive and mode-conditioned-abstain was their exclusion
*action*, but removing the exclusion does not rescue the *granularity*.

P2 is the closest miss and fails differently: it passes the ceiling on pairB
(0.005–0.015) and pairC (0.000–0.003) but fails on pairD (0.095–0.142) and
pairE (0.068–0.101) — 231–363 false abstentions per pairD run against the
certified policy's 0–8. Support-membership widening is nearly free on B/C
geometry and breaks under D/E compression: the same degradation-with-
compression pattern recorded for abstain-tie (PR-9.1(a) §2). The gate is
every-cell by pre-registration; the B/C pass is recorded, not promoted, and
any narrower re-scoped variant would need its own pre-registration.

## 3. The window itself: refuted on independent grounds

The resolution-lag scan closes the question the policies' failure leaves
open — whether a *better-discriminating* future policy should still be
window-shaped:

* **The window is at most one epoch wide wherever it closes.** Across every
  fresh cell, first non-ambiguous verdict lag: median 1.0, **max 1**.
  Fork pairs either adjudicate at the very next epoch boundary or never do
  (contra: 2,499 resolved vs 343 never; oneshot: 197 vs 384; stale: 2,117 vs
  384 — the never-resolved being the design memo's permanently-ambiguous
  one-shot class).
* **Where verdicts resolve, the harm comes after.** Contra-arm `contra_wrong`
  splits 86% post-resolution (3,863), 10% pre-resolution (458), 2% pre-fork
  (86), 2% unpaired (84). A perfect pending-hold would address ~10% of the
  class; enforcement on *resolved* verdicts is what P1 tested, and it fails
  on discrimination, not on verdict availability or timing.
* **Where harm is genuinely pending, it is one-shot** — 79% of one-shot
  `wrong_none` (3,019/3,834) occurs while the leading slot's pair is
  unresolved, and those pairs never resolve. The pending state does localize
  exactly this class (nonredundancy 0.807) — but acting on it costs 33–36%
  false abstention, so the only harm the window uniquely reaches cannot be
  served within the program's ceiling.
* **Soft-arm harm is mostly outside the fork-pair structure entirely**
  (3,079/4,503 `wrong_none` unpaired — led by merge-absorbed slots; M is not
  a pair). Clean-arm wrong rows are 100% unpaired, as predicted.

Observed vs inferred: the counters above are observed; the inference is that
the residuals are a **granularity** problem (slot-level state cannot separate
row-level guilt), not a **timing** problem (the window has no room to act
where verdicts resolve, and no affordable action where they don't). The key
assumption carried from the design memo — that led-only action shape would
tame fork-set exposure the way it tames M exposure — is what the scan
falsified: merge-abstain survives certification because M-led is both
localizing *and* discriminating (0–8 false/run); no router fork set has that
second property on any conflicted arm.

## 4. Verdict and consequences (per §6, no discretion)

* `window-GO` — **no** (P3 failed GO).
* `static-expansion` — **no** (P1 and P2 both failed GO).
* **`negative` — recorded.** Read-time enforcement over the existing
  write-time router evidence is refuted for all three residual classes at
  the program's false-abstention ceiling. The certified merge-abstain
  contract remains the *only* certified reader-facing governance, unchanged.
  Escalation options as pre-registered: **PR-9.2 §9A** (write-event-intrinsic
  identity key; the parallel audit track, unaffected by this negative) toward
  §9B write-path authority, or **explicit acceptance** of the residuals as
  uncovered. No threshold motion, no re-scoped variant without a new
  pre-registration.

## 5. Files

* `benchmarks/analyze_fork_governance.py` — registered scorer change
  (3 policies appended; per-trigger counters on the new rows only; minimal
  additive diff)
* `results/issue_failure_mode_blindness/pr4/pr4_geometry_table.json`,
  `pr3c/pr3c_governance_table.json` + 24 per-run `.governance.json` —
  re-emitted, additive-only vs HEAD (byte-grade guard, §0)
* `benchmarks/pr11_adjudication_scan.py` →
  `results/issue_failure_mode_blindness/pr11/adjudication_scan.json` — the
  scan reader (recomputes every decision through the frozen scorer's
  `apply_policy`, raise-verified against the tables)
* `tests/test_pr11_adjudication.py` — hermetic gates (registration,
  per-policy semantics, no-tie-trigger, structural sweep over all 342 new
  rows, regression pins); `tests/test_pr9_merge_abstain.py` — registration
  pin updated intent-preserving (merge-abstain fixed at its PR-9.1(b)
  position; prefix order now pinned by the PR-11 test)
