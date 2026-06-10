# PR-3a — invariant audit + frozen #87 detector scored per failure mode

Executes the pre-registered "Planned analysis (PR-3)" from SCHEMA.md and
PR3_DESIGN.md §5 PR-3a, on the existing PR-2b/2c artifacts. **Analysis
only**: no new cache runs, no engine change, no detector refit on the
failure-mode data. Audit outcome and trigger results are recorded as-is.

## 1. Invariant audit — PASS

`tests/test_read_path_invariants.py` (hermetic, CPU): after exercising
every read path the failure-mode study uses — `forward()` (plain and
traced), `probe_cross_class_similarity()`, `get_stats()`,
`effective_slot_labels()`, `_get_nearest_batch()`, `_active_sim_floor()`,
`_coverage_eviction_score()`, `records_for(_slots)()`, the driver's
`label_probes()`, and the registry slot-set recomputations, each twice —
`last_seen`, `hit_counts`, `usage`, and `slot_records` (plus
keys/values/occupied) are bit-identical to their pre-read snapshot. A
positive control pins that `learn_local` does move the audited fields, so
the pass is not vacuous, and that `last_seen` moves on writes (the
write-recency semantics PR-3 features need).

**Consequence:** no field is cut from the PR-3 feature set. Write recency
(`last_seen`), maturity (`hit_counts`), and lineage (`slot_records`) are
now pinned invariants; any future engine change that breaks them fails
this test rather than silently invalidating temporal evidence.

## 2. The frozen detector, reproduced and verified

The #87 detector is a procedure, not a coefficient file, so
`benchmarks/score_frozen_detector.py` refits it from the #87 study's own
per-probe CSV (`results/issue_vitl14_blend_confidence/per_probe.csv`,
parity-train epochs of the forced zone, target `vote_correct`) using the
original `heldout_abstention.py` code path, then verifies the reproduction
against the persisted study summary — held-out parity AUC 0.804 and Youden
threshold 0.292748 both match exactly (mismatch raises; pinned by
`tests/test_score_frozen_detector.py`, including a tampered-summary test).
The frozen parameters are dumped in the report for later PRs:
standardized-logistic coefficients **margin −3.546, support +0.411**,
intercept −2.293.

Note the coefficient signs: inside its fit domain (forced zone, under
drift) the detector learned *wrong = sharp margin over thin support* — so
**broad support reads as healthy**. That single learned orientation drives
everything below.

## 3. Results

Scope note (stated honestly): SCHEMA.md pre-registered the triggers but
not the evaluation scope. The scorer designates **all voting rows** as
primary (deployment-relevant: failures vs all correct retrievals) and
reports the forced-zone restriction as secondary. 100% of both modes'
failures sit inside the forced zone, so the gate had domain over every
failure it is scored on.

### Contradiction arm (264 wrong / 2 822)

| quantity | value |
|---|---|
| AUC, all rows (primary) | **0.454** — at/below chance |
| AUC, forced zone (secondary) | 0.671 — weak |
| confidently wrong (score > median correct) | 0.467 all rows / 0.263 in zone |
| error capture at frozen operating point (zoned gate) | **2.75 %**, at 0 % false abstain |
| median score | correct 0.563, contra-wrong 0.555 |

The frozen score barely ranks contradictory failures below correct
retrievals anywhere, and at its own operating point the gate lets
**97 % of contradictory failures through**. The PR-2b per-signal
prediction (collapse-axis inversion) carries over to the composite frozen
instrument.

### Stale arm (60 wrong / 2 192)

| quantity | value |
|---|---|
| AUC, all rows (primary) | **0.120 — strongly inverted** |
| AUC, forced zone (secondary) | 0.775 (see flip explanation below) |
| confidently wrong | **1.000** all rows / 0.133 in zone |
| error capture at frozen operating point | **0.0 %** (zoned and unzoned) |
| median score | correct 0.248, stale-wrong **0.654** |

The frozen detector does not merely miss stale failures — it **actively
prefers them**. A stale boundary failure (margin 0, broad split support)
is precisely the picture the #87 fit learned to call maximally healthy:
every one of the 60 stale failures scores above the median correct probe,
and no threshold on this score can capture any of them without abstaining
on essentially everything (unzoned gate: 0 % error capture at 54 %
false abstain).

**The zone-scope flip, explained and verified.** In-zone AUC looks decent
(0.775) while the global AUC is inverted (0.120). The stale arm's
forced-zone correct rows (n=297) were verified row-for-row to be exactly
the post-boundary probes still carrying stale slots in their surviving
top-k (epochs 7–11, all with `stale_vote_weight` > 0, PR-2c's persistent
reachability): they have margin 0 and even *broader* support (median 16.8)
than the failures (12.2), so within the zone the support axis happens to
rank them above the failures. Both groups, however, score far above
typical correct rows — the zone-conditional ranking is an artifact of
conditioning on rows the detector already misreads, not evidence of stale
discriminability.

### Negative control (clean arm, 2 532 rows, 0 wrong)

Zero forced-zone rows; the zoned gate retains everything at accuracy 1.0.
The unzoned variant would abstain on **56 %** of perfectly correct
stationary rows (median clean score 0.231 < threshold 0.293) — the
detector's score scale does not transfer outside its zone even on healthy
data, which is why the zoned semantics are the only deployable ones and
why the all-rows AUCs above are about *ranking*, not the gate.

### Dose-response (exposure → attribution, SCHEMA.md)

| flagged vote mass | contra arm wrong rate | stale arm wrong rate |
|---|---|---|
| 0 | 0.067 | 0.0 |
| (0, 0.25] | 0.0 | **0.0** (n=237) |
| (0.25, 0.5] | 0.045 | 0.5 (= the 60 boundary ties among 120 rows) |
| (0.5, 0.75] | 0.172 | — |
| (0.75, 1] | **1.0** | — |

Contradiction errors ramp monotonically with fork mass (deterministic
above 0.75). Stale errors exist only at the 0.5 boundary tie; the
persistent post-boundary reachability (mass ≤ 0.25, 237 rows) produces
**zero** errors in this protocol — latent, currently harmless, with the
PR-2c caveat that a one-shot supersession would leave keys at the 0.5 tie
indefinitely.

## 4. Pre-registered triggers — both FIRED (primary scope)

| trigger (SCHEMA.md, stated before PR-2 data) | observed | fired |
|---|---|---|
| CONTRADICTORY: two-axis AUC ≤ 0.60 **or** confidently-wrong > 50 % | AUC 0.454 (cw 0.467) | **yes** → write-time contradiction detection required |
| STALE: two-axis AUC ≤ 0.55 | AUC 0.120 | **yes** → recency must enter the readout |

Recorded honestly: under the forced-zone secondary scope **neither**
trigger fires (0.671 / 0.775), and the scope was not pre-specified. The
scope-independent facts are the operating-point numbers — the frozen gate
captures **2.75 %** of contradictory and **0 %** of stale failures while
they pass at full confidence — and the clean-arm evidence that the
detector's score scale only means anything inside its zone. The
operational-blindness conclusion does not depend on the scope choice; the
formal trigger verdict does, and both readings are in the JSON.

The PR-2c memo predicted the stale trigger would *not* fire, based on
per-signal a-priori-oriented AUCs of 0.83–0.93. Both are right: the
*signals* separate stale failures well, but under the **opposite
orientation** from the one the frozen detector learned in the drift
regime (`analyze_failure_modes.py` orients support as failure-when-HIGH
per #85/#86 collapse semantics; the frozen in-zone fit learned
correct-when-high, +0.411). The orientation conflict the design memo
centers is therefore not only cross-mode but **cross-regime for the same
instrument**: a detector trained where broad support means health
(drift/false-collapse) is anti-trained for a regime where broad support
means a contested fork (stationary supersession). PR-2b/2c/3a now jointly
show no fixed orientation of the existing signals serves contradiction,
supersession, and drift-collapse simultaneously.

On "recency must enter the readout": per PR3_DESIGN.md, this is routed
through H3 (mode-conditioned governance with cross-mode non-inferiority),
not applied blindly — the standing prediction that naive recency elects
the hallucination on contra forks is unchanged by this result.

## 5. What this does not establish

- Nothing here impugns #87 in its own regime: the detector remains
  validated for drift/BLENDED false collapse; PR-3a shows it does not
  *transfer* to injected stationary fork modes, not that it is wrong.
- Single seed, single class pair, stationary manifold, fork-path stale
  only — every PR-2c limitation carries over verbatim.
- No alternative detector is fit and no operating point is chosen here;
  whether a mode-conditioned readout can recover the per-signal
  separability is exactly PR-3b/3c (H1–H3), not this PR.

## 6. Files

- `tests/test_read_path_invariants.py` — invariant audit (3 tests, PASS)
- `benchmarks/score_frozen_detector.py` — reproduce/verify/score driver
- `tests/test_score_frozen_detector.py` — reproduction + accounting guards
- `pr3a_frozen_detector_scores.json` — full report (detector params, per-arm
  scores, both scopes, dose-response, trigger verdicts)
