# PR-6 design memo — governance after static geometry was ruled out

**Status: design only. No code, no cache runs, no engine change. Written
after PR-5 step 1 closed (main `7c595ca`); all numbers quoted are from
`pr5/hazard_postmortem.json` and `pr4/pr4_geometry_table.json`. PR-5 is
treated as a successful negative result: it did not produce a hazard
index, and that absence — not a tuning failure — is what sets PR-6's
problem.**

## 0. The design question

> **Given PR-5's failure to certify benign geometry statically, what is
> the smallest next step toward governance that relies on write dynamics
> or new observables rather than static class-set geometry?**

PR-5 asked whether a static property of the class set could predict
hazard. The answer was no (§1). Every option below therefore drops
geometry as a gate and conditions on something the *write process*
produces — either a measured behavior or a newly logged observable.

## 1. What PR-5 closed (the binding negative)

PR-5 step 1 built three pre-specified static indices (attribution ratio,
within-set centroid confusion rate, fork-bystander confusion rate) from
the cache features of the five spent pairs — training data, named as
such — and scored them against the frozen-probe mixed-arm hazard
(`mode-conditioned-trust` broken counts, the PR-4 measurand):

| | A | C | D | B | E |
|--|--:|--:|--:|--:|--:|
| **hazard (mean broken/run)** | 16.0 | 5.3 | **271.7** | 122.0 | 142.3 |
| attribution_ratio | 1.79 | 1.59 | **3.23** | 1.51 | 1.38 |
| confusion_rate | .0040 | .0115 | **.0635** | .0045 | .0065 |
| fork_confusion_rate | .0025 | .0015 | **.0620** | .0020 | .0045 |

Every index ranks pair D strictly worst — sample-level confusion
separates D by an order of magnitude (127 / 2000 confused samples vs
8–23 for all other pairs). **None recovers the benign end:** pairs B and
E, empirically ~8× pair A's hazard and ~25× pair C's, are
geometrically indistinguishable from the benign pairs in every
candidate. The retrospective gate failed *on training data*, so no
held-out prediction was justified; no index was promoted; no
pre-registration or validation run was produced (PR5_DESIGN §7, first
stop condition).

**Conclusion carried forward — static geometry cannot certify safety.**
Benign geometry is not statically recognizable: the B/E collateral
channel (slot-sharing under ordinary geometry) is emergent from write
dynamics — vigilance, slot allocation, EMA absorption — and leaves no
usable trace in the feature statistics of the class set alone. Any
future "safe on benign geometry" claim is unfalsifiable and is not
admissible.

What PR-5's post-mortem *did* localize (zero compute, from committed
artifacts): in every hazardous pair the harm concentrates on the
**bystander class nearest a fork side** — D 93% on class 28
(cos(10,28)=0.169 against a fork contrast of only 0.052); B 89% on class
48; E 75% on class 56. D's harm is mostly *direct* (wrong-side
deprecation at the contested region — a side-*selection* error); B/E's
is mostly *collateral* (a correct verdict scoped too wide). Both harm
shapes are real and distinct, and both must remain benchmarks (§4).

## 2. The substrate PR-6 inherits

* **The frozen hazard probe is a working measuring instrument.** It is
  not a deployment candidate and never will be; but run over a class set
  it produces a reproducible, decomposable hazard signal (direct vs
  collateral broken, by true class). PR-4/PR-5 established it as
  byte-deterministic across hosts. This is the one thing PR-5 leaves
  that *does* discriminate hazard — empirically, not geometrically.
* **Router verdicts are geometry-stable** (PR-4: supersession exactly
  160 every mixed run; contradiction 209–237). Write-time classification
  is sound; acting on it at read time over whole slots is the closed
  defect.
* **Merge-path stale is write-time-only evidence** (PR-3c) and its
  capture itself degrades on D/E geometry — a benchmark, not a solved
  problem.
* **No write ledger exists.** `slot_records` lineage is in-memory only;
  the value EMA is not logged per write (PR-4 gate-1 Addendum A.4). Any
  record-granularity reconstruction needs the §5 schema PR-5 recorded.

## 3. The three candidate paths

### Path 1 — empirical-hazard validation matrix (no new observables)

Replace the falsified *geometric* gate with an *empirical* one: a fixed,
documented matrix of class sets labeled hazardous/benign **by measured
frozen-probe behavior**, not by any property of the class set. Concretely
— a curated panel that already contains D-like *direct* cells (pair D)
and B/E-like *collateral* cells (pairs B, E) plus the benign anchors
(A, C), with the hazard label being the probe's decomposed broken counts
themselves. Future governance PRs validate against this panel.

* **Cost:** lowest. The scorer, the frozen probe, and the five spent
  pairs already exist; the only new compute is optionally widening the
  panel with a few more empirically-screened sets (mixed arm, the
  discriminating arm). No engine change, no ledger, retrieval untouched.
* **What it buys:** the gate every subsequent path needs and currently
  lacks. PR-4 falsified "works on the pairs we ran"; the answer to that
  is not a cleverer predictor (PR-5 showed there isn't one) but an
  honest, fixed empirical benchmark that *names* its cells and their
  measured hazard, so a later policy's "safe" claim is falsifiable
  against direct *and* collateral harm at once.
* **Risk:** it certifies nothing by itself — it is measurement, not
  mechanism. It also cannot claim coverage of unseen geometries (that is
  exactly the claim PR-5 retired); the panel must state its scope and
  refuse to generalize beyond it. Selection of new panel cells must be
  by *measured* hazard, never by geometry, or it reintroduces the closed
  assumption.

### Path 2 — record-granularity governance (new write-ledger observables)

Deprecate a fork-attributed record's *contribution* inside a slot rather
than the slot. Requires the §5 write ledger and full artifact
regeneration.

* **Cost:** medium-high. Engine/driver logging change (read path
  untouched), schema addition, re-run of at least one pair grid, plus the
  ledger's determinism burden.
* **What it buys:** a smaller blast radius — relevant to the B/E
  *collateral* channel, where a correct verdict was scoped too wide.
* **Why still not first:** PR-5's post-mortem sharpened, but did not
  overturn, the PR-4 verdict. On pair D the harm is *direct* —
  side-*selection* at the contested region — and record granularity does
  not fix selection. It would mitigate the collateral channel (B/E) while
  leaving the larger direct channel (D) untouched. Building new
  observables to address the smaller half first is the wrong order. It
  becomes justified only if a future result isolates *scope*, not
  selection, as the residual defect — which is the condition PR-5
  already set, unchanged.

### Path 3 — write-path refusal / write-time governance (twin-run harness)

Act where the evidence lives: refuse, quarantine, or annotate the
*incoming conflicting write* (or the merge-suspect absorb) instead of
re-weighting reads. The PR-3 §11 reservation, now reached: PR-5's
impossibility result is exactly the "index proven impossible" branch
that PR5_DESIGN §4 named as PR-6's trigger for path 3 — and §7 specified
it must be designed **without geometry gating**.

* **Cost:** highest. A refused write changes all subsequent engine
  state, so it **cannot be shadow-scored from existing read logs** — it
  needs a twin-run protocol (governed vs ungoverned writer over the same
  stream, compared on held-out probes), i.e. a new harness and new cache
  runs per cell. If it ever ships it changes deployed *write* behavior,
  an engine change with its own gates.
* **What it buys:** the only path that addresses merge-path stale at its
  source, the only one independent of read-time geometry, and the only
  one that acts on the sound write-time verdict rather than on whole
  slots after the fact.
* **Why not yet the whole of PR-6:** a twin-run experiment inherits the
  same validation problem PR-4 exposed — "works on the cells we ran" is
  unfalsifiable without a hazard benchmark that spans both harm shapes.
  That benchmark is path 1. Running the twin-run harness *before* the
  empirical panel exists repeats PR-4's structural mistake with a more
  expensive protocol.

## 4. Recommendation — smallest next step

**PR-6 = path 1: build and freeze the empirical-hazard validation
matrix.** It is the lowest-cost step, it is the gate paths 2 and 3 both
require and PR-5 proved cannot be supplied by geometry, and it changes no
engine or retrieval behavior. It conditions hazard on *measured write
behavior* — directly answering §0's "write dynamics rather than static
geometry."

Sequencing (each landing on main before the next; not implemented here):

1. **PR-6 (this recommendation):** assemble the empirical panel — benign
   anchors A, C; direct-harm cell D; collateral-harm cells B, E; their
   frozen-probe decomposed hazard recorded as the label — plus a stated
   procedure for screening additional cells *by measured hazard only*.
   The panel and its scope statement are the deliverable; the frozen
   probe is the measuring instrument, never a deployed policy.
2. **PR-7 = path 3 (write-path twin-run):** the governance mechanism,
   validated against the path-1 panel, geometry-gating forbidden, with
   D-like direct and B/E-like collateral cells both required to pass and
   merge-path stale required not to regress.
3. **Path 2 (record-granularity)** remains deferred behind the §5 ledger
   until a result isolates scope, not selection, as the residual defect.

This memo plans only the next implementable PR; PR-7/path-3 details are
named for sequencing, not specified here.

## 5. Conclusions preserved (binding on PR-6 and successors)

* **Static geometry cannot certify safety.** Benign geometry is not
  statically recognizable (PR-5 step 1); no future governance claim may
  gate on, or assume coverage from, a static property of the class set.
* **Both harm shapes remain required benchmarks.** D-like *direct*
  (wrong-side deprecation / side-selection) and B/E-like *collateral*
  (over-wide correct verdict) are distinct failure channels; any PR-6+
  validation must exercise both, and a policy that fixes one while
  worsening the other does not pass.
* **One-shot ambiguity stays observe-only.** The protocol certifies the
  evidence as insufficient; nothing in PR-4/PR-5 changed that. It becomes
  classifiable only via provenance metadata at the write API
  (PR3_DESIGN §10) — path-3 territory, not a read-time fix.
* **Merge-path stale stays a required benchmark** for any future policy
  PR, including its measured capture degradation on D/E geometry, which a
  candidate must not worsen.
* **Slot-granularity trust deprecation remains closed** unless the unit
  of action changes (per-record contribution, with the §5 observables in
  place). Its only sanctioned use is as the frozen hazard *probe* whose
  damage is the measurand — never as a baseline, a "calibrated" variant,
  or a deployment candidate.
* **Deployed retrieval remains unchanged.** PR-6 contains no mechanism
  that could change it; a deployment proposal must earn its own gates in
  a later PR.

## 6. PR-6 step 1 — empirical-hazard panel scaffold (analysis-only)

Smallest implementable slice of the §4 recommendation. **Scaffold, not the
panel run:** it assembles the panel *manifest* from committed artifacts and
names what is missing; it runs no matrix, adds no cache run, and changes no
engine or retrieval code. The artifact is
`benchmarks/pr6_hazard_panel.py` → `pr6/panel.json`, pinned by
`tests/test_pr6_hazard_panel.py`. The analyzer reads exactly one input —
`pr5/hazard_postmortem.json` — imports no torch, and touches no cache, so it
reproduces on the canonical (darwin) host and is structurally incapable of
consulting geometry.

**The five required cell types, and how each is seeded — by *measured*
frozen-probe hazard, never geometry:**

| cell | harm shape | seeded by | status | label = |
|--|--|--|--|--|
| `clean_control` | none | pairA, pairC | seeded | broken_mean 16.0 / 5.33 |
| `direct_harm` | direct (D-like) | pairD | seeded | broken_mean 271.67 |
| `collateral_harm` | collateral (B/E-like) | pairB, pairE | seeded | broken_mean 122.0 / 142.33 |
| `merge_path_stale` | write-time stale-capture | PR-3c artifacts | **required, unseeded** | — |
| `one_shot_ambiguity` | ambiguous evidence | — | **required, observe-only** | — |

Each seeded label is the frozen `mode-conditioned-trust` probe's broken
counts copied verbatim from the post-mortem, with **both** the direct and
collateral components recorded per cell (the §5 "both harm shapes required"
constraint, made checkable). Harm shape is the §1 mechanistic label carried
forward; it is recorded, not re-derived, and never gates.

**Additional runs needed.** The direct, collateral, and control cells need
*none* — they are fully labeled from committed artifacts. `merge_path_stale`
needs an analysis-only pass over the committed PR-3c stale-arm artifacts to
extract a per-cell stale-capture label *including its D/E degradation*, with
a dedicated stale-arm run only if a cell turns out uncovered — deferred, not
done here. `one_shot_ambiguity` needs no read-time run at all: it stays
observe-only and could become a pass/fail cell only via write-API
provenance metadata (path 3 / PR-7). Widening the panel with more
empirically-screened cells is optional.

**Screening rule for future cells** (recorded in the manifest): admit a
class set iff its frozen-probe hazard is *measured* and decomposed by true
class — the broken counts are the label; geometric properties may never
admit or exclude a cell, and the panel refuses to generalize beyond its
enumerated cells. This step revives nothing PR-5 closed (no geometry gate,
no slot-granularity trust, no record-granularity ledger, no write-path
refusal, no policy tuning) and leaves deployed retrieval untouched.

## 7. PR-6 step 2 — seeding merge_path_stale from committed PR-3c stale arms

The §6 scaffold left `merge_path_stale` *required, unseeded* with one
deferred action: an analysis-only pass over the committed PR-3c stale-arm
artifacts to extract a merge-path stale-capture label *including its D/E
degradation*, escalating to a dedicated stale-arm run only if a cell turned
out uncovered. Step 2 runs that pass. The extension lives in
`benchmarks/pr6_hazard_panel.py` (`read_merge_path_stale_evidence` /
`build_merge_path_stale_cell`), reads only the committed
`pr3c/per_probe_stale-soft_s{0,1,2}.{governance,summary}.json`, and is pinned
by `tests/test_pr6_stale_cell.py`. No cache run was taken.

**What the committed artifacts measure.** The soft-payload ("merge-path")
stale arm is the PR-3c required benchmark (PR3C_RESULT.md §2; PR3_DESIGN.md
§5/§8): the phase-2 write is absorbed on the merge path, the mature slot keeps
decoding A while ground truth is B. The arm is measured across three seeds and
is now recorded as the cell's `partial_evidence` — the measured frozen
`mode-conditioned-trust` probe behaviour, copied verbatim, with no geometric
input:

| seed | class set | stale-wrong (`none`) | frozen-probe stale abstained | broken | merge-suspect events |
|--|--|--|--|--|--|
| s0 | pair-A `[0,8,19,33]` | 374 | 374 | 0 | 192 |
| s1 | pair-A `[0,8,19,33]` | 372 | 368 | 0 | 192 |
| s2 | pair-A `[0,8,19,33]` | 380 | 380 | 0 | 192 |

The capture is **write-time only**: the probe abstains on (essentially) all
stale-wrong rows via the merge-suspect trace while breaking nothing
(`broken=0`), and the read-time fork witness fires on ~4–12 probes, none
merge-related. This confirms the cell's harm shape (`write-time stale-capture`)
from measurement, not assertion.

**Why the cell stays unseeded.** The label the §6 panel requires *includes
D/E degradation*, and that component is **structurally absent** from every
committed PR-3c artifact. The merge-path arm was run on the pair-A class set
only; pairs D and E did not exist as geometries in PR-3c — their class sets
(`pairD {10,28,32,95}`, `pairE {47,56,61,76}`) were constructed later
(PR-4/PR-5) — so no committed artifact measures merge-path stale on D/E (a test
pins that no PR-3c arm touches those classes at all; pair-B merge-path was also
never run). Composing the pair-A merge-path label with the D/E *mixed-arm*
broken counts would be exactly the cross-geometry generalization the panel
forbids. The cell therefore keeps `status: required_unseeded` with
`evidence_status: partial_pairA_only`, carries the measured pair-A evidence,
and names the single completing action in `missing_evidence`: a dedicated
soft-payload stale arm on the pairD/pairE class sets, ≥3 seeds, scored by the
frozen probe — **out of step-2 scope (no new cache runs)**.

This is a measurement-coverage result, never a geometric one: the class set is
used as provenance ("what was measured on what"), and no cell is admitted or
excluded by any geometric property. `geometry_used_as_gate` stays `false`,
`new_cache_runs` stays `0`, and deployed retrieval is untouched.
`clean_control` remains a *lower-hazard empirical control under the frozen
probe* (measured broken_mean 16.0 / 5.33 on pairs A/C), not zero-risk clean
traffic. Path 2 (record-granularity ledger), path 3 (write-path refusal /
twin-run), and slot-granularity trust all stay closed; the merge-path D/E run
is sequenced behind whatever future work constructs it, not opened here.

## 8. PR-6 step 3 — completing merge_path_stale (D/E + pair-B): run scaffold

Step 2 left `merge_path_stale` `required_unseeded` because the only committed
merge-path arm (PR-3c soft-payload) was run on pair-A alone. Step 3 is the
single, scoped cache run that closes it. The runnable matrix is
`pr6_step3_run_matrix.sh`; **this commit is scaffold only — no cache run is
taken here.**

**What it measures.** The soft-payload ("merge-path") stale arm on the three
geometries PR-3c never ran: `pairD {10,28,32,95}` (attractor 52),
`pairE {47,56,61,76}` (attractor 1) — the D/E component the §6 panel's required
label demands — and `pairB {5,27,48,86}` (attractor 13), which drains the
residual pair-B note in `missing_evidence` in the same pass. Three seeds each
(0,1,2), matching the committed pair-A arms: **9 runs total**.

**Why it is the smallest faithful run, not a governance experiment.** It reuses
the frozen `mode-conditioned-trust` probe already emitted inside
`failure_mode_probe.py`'s `governance.json` (top-level `none`,
`mode-conditioned-trust`, `_router`) — no new policy, no policy sweep, no
threshold tuning, no new observable, no retrieval change. Protocol/config are
identical to `pr3c_run_matrix.sh` (`--vision --epochs 12 --supersede-epoch 6`,
`--arm stale --payload-mode soft`); only the class set / attractor differ —
which *is* the missing measurement. PR-3c stays byte-for-byte untouched; the new
arms land in `pr6/stale_de/`. Geometry is provenance only ("what was measured on
what"), never an admission gate (PR-5 step 1 closed geometry as a predictor);
`geometry_used_as_gate` stays `false`.

**Seeding wiring (applied once the artifacts exist).** The panel's
`_stale_soft_seed_label` reader is geometry-agnostic and consumes these arms
unchanged. `read_merge_path_stale_evidence` is extended to read the D/E + B arms
from `pr6/stale_de/`; when `covered_class_sets ⊇ {pairA, pairD, pairE}` the cell
flips to `status: seeded` with `seeds` populated, and `missing_evidence`
collapses (pair-B included → nulled). Until those artifacts are committed the
cell stays `required_unseeded` exactly as step 2 left it.

**Acceptance (verify on the Darwin host after copy-back).** (1) Each of the 9
arms emits `governance.json` with `{none, mode-conditioned-trust, _router}` and
`_router.n_merge_suspect_events`, and `summary.json` with `arm==stale-soft`,
`payload_mode==soft`, `classes` == the pair's set — the script's own
SCHEMA/PROVENANCE self-check. (2) `read_merge_path_stale_evidence` reports
`de_uncovered == []`. (3) Panel rebuild yields `merge_path_stale.status ==
"seeded"`. (4) `pytest tests/test_pr6_stale_cell.py tests/test_pr6_hazard_panel.py`
green; deployed retrieval untouched. Per the host split, the 9 runs execute on
gentoo (SSH compute, venv active); artifacts are copied back and re-verified on
Darwin. `new_cache_runs` for *this* scaffold commit stays `0`.

## 9. PR-6 step 3 — completing merge_path_stale (D/E + pair-B): the measured run

The §8 scaffold is now executed. The 9 soft-payload ("merge-path") stale arms —
`pairD {10,28,32,95}` (attractor 52), `pairE {47,56,61,76}` (attractor 1),
`pairB {5,27,48,86}` (attractor 13), 3 seeds each — were measured on gentoo with
the frozen engine and the frozen `mode-conditioned-trust` probe, and committed
under `pr6/stale_de/`. **`merge_path_stale` is now `status: seeded`.** Deployed
retrieval is unchanged: the five engine/probe/scorer source files that determine
the run output were sha256-verified byte-identical between the canonical (Darwin)
checkout and the gentoo compute host before the run, and no engine, driver, or
retrieval code was touched — only the class set / attractor differ from the
committed pair-A arms (PR-4/5/6 are analysis-only; `f1a5cad..HEAD` changes no
engine file).

**The measured result — the cell's required label, now empirical.** Write-time
capture is **geometry-stable**: the merge-suspect trace fires exactly **192
events/seed on every arm** (pairA/D/E/B alike), so the capture *mechanism* is
intact regardless of geometry. The frozen probe's **read-time damage degrades on
D/E**, exactly as §2/§5 anticipated but had not measured:

| geometry | `broken` mean (read-time damage) | `stale_wrong_fixed` total | merge-suspect |
|--|--:|--:|--:|
| pairA (PR-3c ref) | 0.0 | 0 | 192/seed |
| pairB | 0.33 | 2 | 192/seed |
| pairE | 46.0 | 1 | 192/seed |
| pairD | 112.67 | 0 | 192/seed |

The probe **fixes ~0 stale-wrong rows on any geometry** (`stale_wrong_fixed`
total 0–2 of ~1100 stale-wrong rows): read-time application cannot repair
merge-path stale, and on D/E it *only* inflicts collateral damage on correct
rows. This confirms, from measurement, that merge-path stale is **write-time-only
evidence whose read-time use degrades on D/E geometry** — a required benchmark a
future policy must not regress, not a solved problem. pair-B is benign on this
axis (like pair-A), draining the step-2 residual note in the same pass.

**A protocol gap found and closed.** The §8 scaffold matrix ran only
`failure_mode_probe.py`, which emits `per_probe`/`summary`/`topk` but **not**
`governance.json` — the per-stem policy table (`none` / `mode-conditioned-trust`
/ `_router`) the panel reads is produced by the separate frozen scorer
`analyze_fork_governance.py`, exactly as PR-3c did. The scaffold's own
SCHEMA/PROVENANCE self-check would therefore have `FileNotFoundError`'d. Step 3
adds the missing stage to the matrix: it calls the scorer's **own per-stem path**
(`reproduce_frozen_detector` + `load_run` + `score_run`, byte-identical to
`analyze_fork_governance.main()`'s per-stem write loop, `indent=1`,
`sort_keys=True`) and **skips that scorer's cross-run H1/H2 read-time/fork
classifier studies**, which require a `mixed_s0` arm, are undefined for a
stale-only directory, and are not consumed by the panel. The frozen detector is
reproduced from the committed #87 fit set
(`score_frozen_detector.FIT_CSV/FIT_SUMMARY`), not from this directory, so the
per-stem scoring is independent of which arms are present — the measurement is a
faithful per-geometry extension of the PR-3c stale-soft scoring, not a new
protocol.

**Seeding wiring (now applied).** `read_stale_de_evidence` reads the D/E + B arms
from `pr6/stale_de/`, guarded so a stem's name must match its measured `classes`
(provenance integrity, never a geometric gate) and a non-soft payload is refused.
`build_merge_path_stale_cell` flips to `seeded` with `seeds` ordered
pairA→pairD→pairE→pairB, records the measured degradation, and nulls
`missing_evidence` (pair-B drained). With no committed step-3 arms the analyzer
falls back to the exact step-2 `required_unseeded` cell, so the change degrades
gracefully. `pr6_hazard_panel.py` still imports no torch and reads only committed
JSON; `geometry_used_as_gate` stays `false`; `new_cache_runs` is `9` (the
committed arms — the panel *build* itself takes none); `engine_or_retrieval_change`
stays `false`.

**Scope held.** This step revives nothing PR-5/earlier closed: no deployed
retrieval change, no slot-granularity trust (the frozen probe remains only the
measurand whose damage *is* the label — never a baseline or deployment
candidate), no trust-parameter tuning, no static geometry gate, no record-
granularity ledger (path 2), no write-path refusal / twin-run (path 3), and no
broad class-pair search. The panel still certifies nothing beyond its enumerated
cells. Tests: `tests/test_pr6_stale_cell.py` (rewritten to pin the seeded state,
the geometry-stable write-time capture, the measured D/E degradation, the
provenance guards, and the unseeded fallback) and `tests/test_pr6_hazard_panel.py`
(updated for the seeded cell, `new_cache_runs == 9`, and the 13 committed inputs)
— full suite **556 passed**.
