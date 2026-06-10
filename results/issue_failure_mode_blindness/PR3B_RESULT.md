# PR-3b — fork/mixed/jitter artifacts: orientation is mode-, regime-, AND geometry-conditioned

Executes PR3_DESIGN.md §5 PR-3b on gentoo (vitl14_cifar100_train, #87
config, stationary, 12 epochs, supersede epoch 6, rate 0.15). 21 runs
(`pr3b_run_matrix.sh`): mixed/one-shot/jitter/soft variants, PR-2
replication seeds, and a second class pair. All hermetic gates (33 tests)
green on both hosts before the runs; cache sanity 16/16; all 21 analyzer
outputs byte-identical darwin/gentoo. **No deployed governance anywhere**:
shadow labels (`per_slot.csv`, `fork_events.csv`) and analysis outputs
only. PR-3a is the pinned baseline. Exact counts accompany every rate.

## 1. What was asked

Whether the PR-2 detector-orientation conflict is **mode-conditioned**
(contradiction vs supersession need opposite orientations),
**regime-conditioned** (the frozen drift-trained #87 detector is
anti-oriented for stationary fork modes), or **artifact-specific** (the
signatures are fingerprints of the exact PR-2 protocol). Answer: **all
three layers are real**, and the third is stronger than the design memo
anticipated.

## 2. Replication of the PR-2 baselines (pair A = classes {0,8,19,33})

A-priori-oriented AUC (analyze_failure_modes convention; > 0.5 = signal
flags the failure), wrong counts in parentheses:

| run | wrong | entropy→stale | top1_sim→stale | margin→stale | frozen #87→stale |
|---|---|---|---|---|---|
| stale s0 (PR-2c) | 60/2192 | 0.833 | 0.292 (inv) | 0.930 | 0.120 (inv) |
| stale s1 | 62/2116 (60 stale-len) | 0.838 | 0.314 (inv) | 0.929 | 0.116 (inv) |
| stale s2 | 62/2268 | 0.821 | 0.320 (inv) | 0.932 | 0.116 (inv) |

| run | wrong | entropy→contra | top1_sim→contra | frozen #87→contra |
|---|---|---|---|---|
| contra s0 (PR-2b) | 264/2822 | 0.356 (inv) | 0.740 | 0.454 |
| contra s1 | 343/2758 | 0.448 (inv) | 0.617 | 0.386 (inv) |
| contra s2 | 422/2860 | 0.494 (chance) | 0.547 | 0.363 (inv) |

Within pair A the PR-2 stale signature replicates cleanly across seeds.
The contra signature replicates in *direction* but with substantial drift
in magnitude (entropy inversion 0.36 → 0.49 across seeds; top1_sim 0.74 →
0.55). Clean controls: 0 wrong (s2, pair-B s0) and 12/2508 wrong with
zero injection labels (s1) — the manifold is near-noise-free.

## 3. The four variant arms

### One-shot supersession — the tie regime persists, as predicted

3 seeds: wrong = 360/2337, 366/2285, 372/2449 (~16 %, vs 2.7 % for
repeated-write stale). Every post-boundary epoch (6 of 6, all seeds):
superseded-key median stale vote mass **exactly 0.5**, stale elected
60/60 per epoch on this host, `fork_resolution = persistent-tie`. The
PR-2c prediction is confirmed: one superseding write leaves retrieval at
a permanent 50/50 tie. **Platform caveat (pinned by test):** at an exact
tie the *elected* side flips across BLAS implementations (darwin vs
gentoo differ on the tiny fixture) — the tie is the stable fact, the
election direction is not.

The frozen #87 detector on one-shot stale: AUC **0.015, 0.009, 0.015** —
its worst numbers anywhere; the highest-severity stale variant is the one
it prefers most.

### Soft payloads — merge-path (EMA-freeze) stale is real, persistent, and confident

The originally hypothesized mode, untested until now. All 192 phase-2
writes per run took the merge path (`absorbed` 192/192, zero forks); all
32 merged slots carry both phases' provenance (`merge_candidate`) and
**still decode A at the end of the run** (`fork_resolution =
old-persists`, 3/3 seeds). Wrong = 385/2463, 374/2412, 389/2548 (~15 %),
sustained at 60–63 per epoch for all 6 post-boundary epochs — unlike
fork-path stale, more writes do NOT self-correct it within the run.

Its signature is a **third profile**, resembling neither PR-2 mode:
median stale vote mass **1.0** (the merged slot is the only voter for the
key — no tie, no fork competitor), margin AUC ≈ chance (0.578, 0.510,
0.582), top1_sim inverted (0.255, 0.267, 0.272), entropy mildly positive
(0.723, 0.684, 0.753). A merge-path stale failure is a *confident,
single-slot, mature* wrong recall — closer to PR-2b contradiction than to
PR-2c boundary stale, but with no co-resident fork for any topology
detector to find. **Write-time evidence is the only trace it leaves**
(the absorbed conflicting write in fork_events).

### Key jitter — the pair-A stale signature is NOT the exact-tie artifact

eps 0.05 / 0.15 (seed 0): wrong 57/2195, 61/2204; phase-2 pre-write sims
move off 1.0 (pinned by test) and the boundary tie becomes a near-tie.
Orientation survives: entropy→stale 0.839 / 0.822, top1_sim inverted
0.269 / 0.301, margin degrades gracefully 0.883 / 0.803 (from 0.930
exact), frozen #87 still inverted 0.123 / 0.146, resolution
later-dominates. On pair A, the stale signature is a property of the
supersession mechanism, not of identical re-written keys.

### Mixed arm — labels stay disjoint; co-occurrence dilutes the stale signal

Pair A, 3 seeds: wrong 489/2530, 501/2479, 466/2638; contra-lenient 424,
441, 403; stale-lenient 63, 60, 63; **contra∧stale lenient overlap = 0 in
every run** (also pair B) — the modes remain label-separable under
co-occurrence. Each mode keeps its pair-A orientation inside the mixed
memory (entropy→contra 0.43–0.47 inverted, top1_sim→contra 0.65–0.71;
top1_sim→stale 0.29–0.32 inverted), but the stale collapse-axis signal is
**diluted** (entropy→stale 0.60–0.67 vs 0.82–0.84 pure): the "correct"
comparison population is now fork-contaminated. Any mode classifier
calibrated on pure arms would meet weaker separations in realistic mixed
traffic.

## 4. The second class pair breaks the signatures (the central PR-3b finding)

Pair B = classes {5,27,48,86}, attractor 13, seed 0 — same cache, same
protocol, same engine config:

| signal (a-priori orientation) | pair A stale (s0/s1/s2) | pair B stale | pair A mixed contra (3 seeds) | pair B mixed contra |
|---|---|---|---|---|
| entropy→mode | 0.82–0.84 | **0.534 (chance)** | 0.43–0.47 (inv) | **0.712 (flipped sign)** |
| top1_sim→mode | 0.29–0.32 (inv) | **0.658 (flipped sign)** | 0.65–0.71 | **0.347 (flipped sign)** |
| margin→stale | 0.93 | 0.949 | — | — |
| frozen #87→mode | 0.12 (inv) | 0.148 (inv) | 0.37–0.41 (inv) | 0.227 (inv) |

Counts: pair B stale 60/2602 wrong; pair B mixed 572/2702 wrong (513
contra-lenient, 57 stale-lenient); pair B pure contra 348/2824 wrong
(entropy 0.505 chance, top1_sim 0.604).

The medians show these are real geometry effects, not noise: pair A
stale-wrong sits *on* its mature slot (top1_sim 0.755 vs correct 0.661 —
hence inversion); pair B stale-wrong sits slightly *off* it (0.630 vs
0.694 — hence the flip). Pair B mixed contradiction is high-entropy AND
high-sim (2.70 vs 2.21; 0.768 vs 0.677) — a blended-like profile, the
opposite of pair A's tight low-entropy fork retrieval.

**Consequence for H1 (read-time mode classification):** the PR-2 signal
*profiles* do not survive a change of class pair, in either mode, even
with mechanism, protocol, engine, cache, and regime held fixed. They are
not artifacts of the exact-tie protocol (jitter preserved them) — they
are contingent on local class geometry. A read-time profile classifier
fit on pair A would be anti-oriented on pair B for the two strongest
signals. Per the design memo's own falsifier ("classification that works
in-protocol but collapses under unseen class pairs ⇒ the profiles were
protocol fingerprints"), H1's premise now requires per-geometry
calibration to survive at all — which deployment cannot do, because it
would need failure labels per geometry.

## 5. What IS stable, across all 21 runs

1. **The frozen #87 detector is inverted or at-chance for every
   stationary fork-mode failure population in every run** — all 22
   mode-AUCs across the 18 failure runs are ≤ 0.484, spanning both modes,
   all four variants, both class pairs (worst: one-shot 0.009;
   best-for-it: pair-B mixed stale 0.484). Regime conditioning is
   unconditional in this data: a detector trained where failure = diffuse
   collapse cannot rank stationary fork failures, whatever their mode or
   geometry.
2. **Write-time mechanical ground truth never degraded.** Every conflict
   event was captured with its observables; injection outcomes stayed
   deterministic (mixed-arm contra: 228/228 `forked` in each of 4 runs;
   soft supersession: 192/192 `absorbed` per seed; one-shot: 32/32
   `forked` exactly once). Labels stayed disjoint (overlap 0 in all
   runs).
3. **Tie/near-tie structure behaves lawfully**: margin AUC 0.93 at exact
   ties → 0.88/0.80 under jitter → ~0.5 (chance) when no fork exists
   (merge path). The signal tracks fork topology, not failure per se.
4. **Fork resolutions sort cleanly** by protocol: repeated-write →
   later-dominates (6/6 such runs incl. jitter and pair B), one-shot →
   persistent-tie (3/3), merge-path → old-persists (3/3).

## 6. Implications for PR-3c

- **H1 (read-time mode classifier) is near-falsified as a
  geometry-general mechanism.** If attempted at all, it must be scored
  across class pairs with the pair-B flip in hand; the pre-registered
  acceptance criterion (AUC ≥ 0.80 surviving unseen class pairs) looks
  unreachable for profiles built on entropy/top1_sim. PR-3c should
  test it honestly and expect the documented failure outcome.
- **H2 (write-time fork classification) is strengthened twice over**:
  the only mode-stable, geometry-stable evidence in this study is at
  write time, and merge-path stale — the most persistent failure found
  (old-persists, ~15 % sustained wrong) — is *invisible* to read-time
  fork topology (single slot, no tie, no co-resident competitor) while
  being trivially visible at write time (an absorbed conflicting write).
- **H3 baselines now have sharper predictions**: one-shot ties give
  abstain-tie a real target (360+/run); merge-path gives every
  fork-topology-based policy a guaranteed miss; recency evidence
  (`last_write_seq`, incumbent maturity in fork_events) is logged and
  byte-deterministic, ready for shadow governance.

## 7. What this does not establish

- Still one encoder, one cache, stationary; drift × fork remains
  untested (BLENDED cross-talk deferred, as designed).
- Pair B is one additional geometry at seed 0 — enough to break
  generality claims, not enough to characterize the geometry dependence.
- Soft-payload cosine was fixed at 0.6; the merge/fork boundary
  (cosine ≈ 0.5) is unexplored.
- No governance was simulated; nothing here measures cross-mode harm —
  that is PR-3c, with the comparison table mandated before any deployed
  change.
- The one-shot election direction is platform-contingent at exact ties;
  any future analysis must treat tie elections as undefined, not as a
  tie-break rule.

## 8. Files

- `pr3b_run_matrix.sh` — the exact 21-run protocol
- `pr3b/per_probe_<run>.csv` + `.per_slot.csv` + `.fork_events.csv` +
  `.summary.json` — 21 runs × 4 artifacts (gentoo-computed)
- `pr3b/per_probe_<run>.analysis.json` — per-signal analyses
  (byte-identical darwin/gentoo, 21/21)
- `pr3b/per_probe_<run>.frozen.json` — frozen #87 scores per run
- `pr3b/pr3b_orientation_table.json` — the collated cross-run table
- `tests/test_pr3b_arms.py` — 8 hermetic gates (33 total with PR-2)
