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
