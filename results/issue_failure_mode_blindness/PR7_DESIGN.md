# PR-7 design memo — the smallest safe write-path twin-run harness

**Status: design only. No code, no cache runs, no engine change, no
implementation in this branch.** Written after PR-6 closed (main
`91e2fa4`); every number quoted is read from committed artifacts
(`pr6/panel.json`, `pr5/hazard_postmortem.json`, `PR3A_RESULT.md`). PR-6
is treated as a successful, binding result: it built and froze the
empirical-hazard panel that PR-5 proved geometry could not supply, and
that panel — not a new predictor — is the gate PR-7 must answer to.

This memo names the harness, the protocol, the artifacts, the tests, and
the boundaries. It builds none of them. PR-7's implementation is
sequenced in §13 as steps that land on main one at a time; this branch
contains this file only.

## 0. The design question

> **What is the smallest safe twin-run harness that can test write-path
> governance without mutating deployed retrieval or repeating the failed
> read-time / slot-level interventions?**

"Safe" here is load-bearing and has three parts, each a hard boundary
(§12): the harness must (a) leave deployed retrieval byte-frozen, (b) act
only where prior PRs proved the signal is sound — the write-time verdict —
and never revive a read-time or slot-granularity trust action, and (c)
remain falsifiable against the PR-6 panel's *both* harm shapes, so a
candidate that helps one channel while hurting another is reported as a
failure, not a win.

## 1. Repo / artifact inventory

**Deployed engine — frozen, must not change (the safety invariant).**
The source files that determine run output were sha256-verified
byte-identical across the canonical (Darwin) and compute (gentoo) hosts
in PR-6 step 3 and must stay so:

| file | role PR-7 must not touch |
|---|---|
| `associative_core.py` | `ContinuousCAM.learn_local` bipartite contradiction check (`payload_sims > 0.5`), EMA-freeze merge path, slot allocation/eviction |
| `fast_associative_memory.py` | deployed `forward()` vote, top-k softmax readout |
| `query.py`, `dynamic_vigilance.py`, `adapter.py` | retrieval / vigilance / projection surface |

**Experimental writer (the twin-run substrate).**
`benchmarks/failure_mode_probe.py` — injects through the *normal* learn
path and modifies no engine state (SCHEMA.md §"Mechanisms"); arms
`clean / contra / stale / mixed`; `--payload-mode soft` selects the
EMA-merge ("merge-path") stale mode; CLI surface `--vision
--epochs --supersede-epoch --arm --payload-mode --vision-classes
--vision-attractor-class --seed`. It already classifies every write at
write time into the event classes a governance policy would act on —
`contradiction` (`forked` outcome), `supersession` (merge-suspect
absorb), `one-shot-ambiguous` — and logs the write-time observables in
`<out>.fork_events.csv` (`pre_sim`, `payload_cos_incumbent`,
`effective_vigilance`, incumbent maturity/recency/lineage, `outcome`,
`owner_slot`). **No new observable is required for write-path governance;
that is exactly the path-2 ledger PR-6 §3 deferred.**

**Frozen readout / scorer (the measurand after both runs).**
`benchmarks/analyze_fork_governance.py` — recomputes per-stem policy
tables from the logged `topk.csv` shadow basis; policy `none` is
bit-identical to the deployed `forward()` vote (re-derived from emitted
float32 text and raised-on-mismatch, SCHEMA.md §topk). Emits
`governance.json` with the `none` / `mode-conditioned-trust` / `_router`
block (`_router.n_merge_suspect_events`), `indent=1, sort_keys=True`. The
frozen #87 two-axis detector is reproduced from the committed fit set
(`benchmarks/score_frozen_detector.py` `FIT_CSV` / `FIT_SUMMARY`), never
refit from the run under test.

**The PR-6 panel (the gate).**
`benchmarks/pr6_hazard_panel.py` → `results/issue_failure_mode_blindness/pr6/panel.json`
— 4 seeded cells + 1 observe-only (§6). Pinned by
`tests/test_pr6_hazard_panel.py`, `tests/test_pr6_stale_cell.py`.

**Seed artifacts.** `pr5/hazard_postmortem.json` (clean/direct/collateral
labels); `pr3c/per_probe_stale-soft_s{0,1,2}.*` (pair-A merge-path stale);
`pr6/stale_de/per_probe_stale-soft_s{0,1,2}_pair{D,E,B}.*` (D/E/B
merge-path stale). Mechanism tests: `tests/test_failure_mode_probe.py`,
`tests/test_failure_mode_vision.py`. Schema: `SCHEMA.md`.

## 2. What PR-6 (and its predecessors) closed — carried forward, binding

Every item below is a *settled* result; PR-7 may not reopen any of them.

* **Static geometry cannot certify safety** (PR-5 step 1). No static
  property of a class set (attribution ratio, centroid/fork confusion
  rate) recovers the benign end: pairs B/E are ~8× pair A's hazard yet
  geometrically indistinguishable from the benign anchors. Geometry is
  provenance only — it may never admit, exclude, or certify a cell.
* **Read-time detector orientation is not globally stable** (PR-3a). The
  frozen #87 two-axis detector ran *inverted* per mode — contra AUC
  0.454, stale AUC 0.120 — so the orientation that separates one failure
  mode mis-orders the other. No single read-time readout is correct for
  both modes at once.
* **Slot-granularity trust deprecation is unsafe and closed** (PR-4).
  Acting on the sound write-time verdict at read time *over whole slots*
  is the defect, not the fix. The `mode-conditioned-trust` probe survives
  **only** as the frozen measurand whose damage *is* the hazard label —
  never a baseline, a "calibrated" variant, or a deployment candidate.
* **Merge-path stale is now measured across A/D/E/B** (PR-6 step 3). It is
  a required benchmark, not a solved problem.
* **Write-time merge-suspect capture is geometry-stable.** The
  merge-suspect trace fires exactly **192 events/seed on every arm**
  (pairA/D/E/B alike) — the capture *mechanism* is intact regardless of
  geometry.
* **Read-time use of merge-suspect evidence degrades on D/E.** The frozen
  probe's read-time `broken` mean is pairA 0.0, pairB 0.33, pairE 46.0,
  pairD 112.67, while it fixes ~0 stale-wrong rows on any geometry
  (`stale_wrong_fixed` total 0–2 of ~1100). Read-time application of this
  write-time evidence is all downside on D/E.
* **One-shot ambiguity stays observe-only.** Certified insufficient
  evidence; carries no hazard label; must never be scored pass/fail. It
  could become classifiable only via write-API provenance metadata —
  named here as a *possible* path-3 observable, but **out of PR-7 scope**
  (§12).
* **Both harm shapes remain required benchmarks.** D-like **direct**
  (wrong-side deprecation / side-selection at the contested region) and
  B/E-like **collateral** (a correct verdict scoped too wide) are
  distinct channels; a policy that fixes one while worsening the other
  does not pass.
* **Deployed retrieval is unchanged** and stays so until a later PR earns
  its own deployment gates.

## 3. Why write-path, and why it cannot be shadow-scored

Read-time and slot-level levers are exhausted: PR-3a closed a stable
read-time orientation, PR-4 closed slot-granularity read-time trust, PR-5
closed static geometry. What survives all of them is the **write-time
verdict**: router classifications are geometry-stable (supersession
exactly 160/run, contradiction 209–237 in PR-4) and merge-suspect capture
is geometry-stable (192/seed, §2). The only remaining lever that acts on
a *sound* signal is the write path itself — refuse, quarantine, or
annotate the conflicting write at the moment the engine already labels it,
before it corrupts subsequent state.

That same property is why it **cannot be shadow-scored**.
`analyze_fork_governance.py`'s read-time policies are counterfactual
re-readouts of one fixed `topk.csv`: the write history is identical across
policies, so they recompute offline. A refused or quarantined write
changes the *write history* — every later allocation, EMA absorption, and
eviction downstream of it differs — so there is no single logged basis to
recompute from. Testing it requires two genuinely separate runs over the
same input stream: an **ungoverned** writer and a **governed** writer,
each producing its own `topk.csv`, then the frozen readout applied to
both. This is the PR-3 §11 reservation and the PR5_DESIGN §4 "index proven
impossible → path 3" trigger, which PR5_DESIGN §7 required be designed
**without geometry gating**.

## 4. Candidate write-path governance actions

The governed writer acts on the *existing* write-time event class — no new
observable, no ledger (path 2 stays deferred). Three candidate actions,
ordered by blast radius, each **parameter-free first** (a binary decision
on the already-emitted event class, so there is no threshold to tune —
the trap PR-4/PR-5 fell into):

| action | intervenes on | what it does | reversible? |
|---|---|---|---|
| **annotate** | `supersession` (merge-suspect absorb) | stamp a write-time `merge_suspect` flag on the slot record; **change nothing at read time** | yes — null-action upper bound on safety |
| **quarantine** | `contradiction` (`forked`) and/or merge-suspect | allocate to a side region excluded from the deployed read vote; the write is retained and recoverable, not destroyed | yes |
| **refuse** | `contradiction` (`forked`) | do not allocate the co-resident contradictory slot at all | no (the most aggressive; tested last) |

`annotate` is the floor: it must cost nothing, proving the harness itself
adds no harm. `refuse` is the ceiling. The harness compares baseline
(action = `none`) against **one** candidate action at a time, never a
sweep.

**Where the action lives is the whole safety argument.** Governance is
implemented as an *opt-in experimental hook inside the writer driver*
(`failure_mode_probe.py`), gated behind a new `--govern {none,annotate,
quarantine,refuse}` flag, intercepting at the point where the driver has
*already classified* the write. It is **not** a change to
`associative_core.py` / `fast_associative_memory.py`. The deployed
`forward()` and `learn_local` stay byte-identical; the deployed retrieval
path never reaches the hook. The governance behavior is exercised only
through the experimental flag, which production never sets. This is what
makes a write-path test "without mutating deployed retrieval" possible at
all.

## 5. Twin-run protocol

Per panel cell, run the same input stream twice and read both out with the
frozen scorer:

1. **Baseline write path.** `failure_mode_probe.py` exactly as committed,
   `--govern none` (≡ no flag), over the cell's class set / arm / attractor.
2. **Candidate governed write path.** The *same* driver, `--govern
   <action>` for one action from §4, intercepting at the already-classified
   write event and otherwise calling the identical learn path.
3. **Identical seeds / class sets / protocol.** Both runs share
   `--vision --epochs 12 --supersede-epoch 6`, the cell's `--arm` /
   `--payload-mode` / `--vision-classes` / `--vision-attractor-class`, and
   the same 3 seeds `{0,1,2}` — mirroring `pr3c_run_matrix.sh` /
   `pr6_step3_run_matrix.sh`. **Only the `--govern` flag differs.**
4. **Frozen readout / scoring after both runs.** `analyze_fork_governance.py`
   per-stem policy `none` (= the deployed `forward()` vote) applied to
   **both** runs' `topk.csv`. The readout is frozen and identical across the
   twin; only the upstream *write state* differs, so any delta is
   attributable to the write-path action, not to a read-time policy. The
   #87 detector is reproduced from the committed fit set, independent of
   either run's contents.
5. **Exact delta accounting.** Baseline − governed, per cell per seed, on
   the five quantities in §7, with sign conventions and pre-registered
   margins fixed *before* any run.

Determinism is a precondition: both arms must be byte-deterministic and
sha256-stable across darwin/gentoo (the PR-6 host split), and the
baseline `none`-policy vote must remain bit-identical to deployment.

## 6. Benchmark cells from the PR-6 panel

PR-7 validates against the frozen panel — it adds no cells, screens none
by geometry, and certifies nothing beyond these enumerated cells.

| cell | class set (attractor) | harm shape | baseline label (frozen probe) | role in PR-7 |
|---|---|---|---|---|
| `direct_harm` | pairD `{10,28,32,95}` (52) | **direct** (D-like) | broken_mean **271.67** | **required to improve**, no collateral regression |
| `collateral_harm` | pairB `{5,27,48,86}` (13); pairE `{47,56,61,76}` (1) | **collateral** (B/E-like) | broken_mean **122.0 / 142.33** | **required not to worsen** (both shapes rule) |
| `clean_control` | pairA `{0,8,19,33}`; pairC | none (lower-hazard empirical control, *not* zero-risk) | broken_mean **16.0 / 5.33** | governance must not inflate clean traffic |
| `merge_path_stale` | pairA / pairD / pairE / pairB, soft arm | write-time stale-capture | capture **192/seed all arms**; read-time broken_mean A 0.0 / B 0.33 / E 46.0 / D 112.67; fixes ~0 stale-wrong | capture must stay 192/seed; D/E read-time damage must not worsen |
| `one_shot_ambiguity` | — | ambiguous (observe-only) | no label | **stays observe-only** — never forced to pass/fail (§12) |

Both `direct_harm` (D) and `collateral_harm` (B/E) are required to pass:
a candidate that drains direct harm while inflating collateral fails, and
vice-versa.

## 7. Exact delta accounting

Each accounting quantity maps to a measurand already emitted by the
existing artifacts; PR-7 invents no metric. All deltas are **baseline −
governed**, per cell per seed.

| accounting quantity | measurand (source) | desired sign of (baseline − governed) | guard |
|---|---|---|---|
| **stale-wrong** | `stale_wrong` count (per_probe / summary) | ≥ 0 (governed fixes rows read-time left at ~0-fixed) | must not increase on any cell |
| **contradiction capture** | `_router` contradiction events / `forked` outcomes captured (fork_events + governance `_router`) | ≥ 0 (governed quarantines/refuses the contradictory writes) | capture, do not silently drop clean writes |
| **collateral harm** | frozen-probe `broken` on bystander/correct rows, collateral component (governance.json) | ≥ 0 on B/E; **= 0 minimum** | a negative delta on B/E is a hard fail |
| **clean traffic** | `clean_control` broken_mean + false-action rate on benign rows | ≈ 0 (no new harm on benign) | any increase beyond pre-registered tolerance fails |
| **merge-path stale** | `merge_suspect_events` (must hold 192/seed) and frozen-probe broken_mean on D/E | capture delta = 0; D/E read-time broken not worsened | regressing 192/seed capture is a hard fail |

The delta table, with pre-registered acceptance margins, is committed
*before* the first governed run as a frozen expectation file — the run
either meets the pre-registration or it does not. Margins are never
re-tuned after seeing the result (PR-4/PR-5 trap).

## 8. Acceptance criteria

A candidate write-path action **passes the panel** iff *all* hold:

1. **Direct (D):** governed reduces `direct_harm` broken by the
   pre-registered margin, ≥3 seeds, with no collateral regression.
2. **Collateral (B/E):** governed does not increase `collateral_harm`
   broken on pairB *or* pairE; ideally reduces it.
3. **Clean (A/C):** governed does not increase `clean_control` broken or
   introduce new false-action on benign traffic beyond the pre-registered
   tolerance.
4. **Merge-path stale:** write-time capture stays geometry-stable
   (192/seed on every arm); D/E read-time damage (broken_mean 112.67 /
   46.0) is not worsened; a write-time `refuse`/`quarantine` that drains
   stale-wrong rows read-time left unfixed is the target win.
5. **Both-shapes rule:** a candidate fixing one shape while worsening
   another **fails** (PR-6 §5), full stop.
6. **Determinism / boundary:** governed and baseline both byte-stable
   across darwin/gentoo; baseline `none` vote bit-identical to deployment;
   deployed-engine sha256 unchanged from the PR-6 baseline.
7. **Scope refusal:** the result certifies nothing beyond the enumerated
   cells and makes no claim on unseen geometries (the claim PR-5 retired).

Passing the panel authorizes a *deployment-proposal* PR (its own gates),
not deployment itself.

## 9. Stop conditions

PR-7 halts and records a negative result — without tuning to manufacture
a pass — if any of these fire:

* **No parameter-free action improves direct harm without regressing
  collateral or clean.** Then write-path refusal is not the mechanism;
  stop and record it, exactly as PR-5 recorded geometry's failure. Do not
  introduce a threshold to rescue it.
* **The governed writer cannot be expressed without touching deployed
  engine code.** The experimental-only boundary is then violated; stop and
  redesign, or defer to the path-2 ledger.
* **The twin-run is non-deterministic** (governed run not reproducible
  across seeds/hosts). Determinism is a precondition; stop until it holds.
* **Acceptance would require crossing a §12 boundary** — forcing
  `one_shot_ambiguity` to classify, reviving slot-granularity or read-time
  trust, or admitting a geometry gate. Stop; out of bounds.
* **The action turns out shadow-scorable** (no actual downstream state
  divergence between twin runs). Then it is a read-time policy in disguise
  and belongs to the closed read-time line; stop and route it back there.

## 10. Artifact schema

New artifacts live under `results/issue_failure_mode_blindness/pr7/`,
following PR-6 conventions (`indent=1, sort_keys=True`, gzip `-n`,
governance.json reuse). No existing artifact is modified.

**Twin run dirs.**
`pr7/twin/<cell>/<govern>/per_probe_<arm>_s{0,1,2}_<pair>.{csv,summary.json,
topk.csv.gz,per_slot.csv,fork_events.csv,governance.json}` — same emit set
as `pr6/stale_de/`, one subtree per `<govern> ∈ {none, annotate,
quarantine, refuse}`. `summary.json` carries a new provenance field
`govern` recording the action applied (`none` for baseline); `classes`,
`arm`, `payload_mode`, `attractor` as today.

**Delta artifact.** `pr7/twin_delta.json` (top-level), built by an
analysis-only `benchmarks/pr7_twin_delta.py` that imports no torch and
reads only committed JSON (mirroring `pr6_hazard_panel.py`). Keys:

```
{
  "design": "PR7_DESIGN.md §5 — write-path twin-run",
  "engine_or_retrieval_change": false,
  "geometry_used_as_gate": false,
  "deployed_engine_sha256_parity": true,
  "new_cache_runs": <int>,
  "frozen_detector": { "fit_csv": ..., "fit_summary": ... },
  "govern_action": "<one of annotate|quarantine|refuse>",
  "cells": {
     "<cell>": {
        "baseline": { <five §7 measurands> },
        "governed": { <five §7 measurands> },
        "delta":    { <baseline-minus-governed, per quantity, per seed> },
        "pre_registered_margin": { ... },
        "verdict": "pass|fail|inconclusive",
        "both_shapes_ok": <bool>
     }, ...
  },
  "conclusions_enforced": [ ... §2 list ... ],
  "scope": "certifies nothing beyond the enumerated cells; no generalization to unseen geometry"
}
```

**Provenance guards (integrity, never a geometric gate).** A governed
stem's filename must match its measured `classes` *and* its `govern`
action; a non-soft payload is refused where the soft (merge-path) arm is
required; the baseline `none`-vote bit-identity check is retained. With no
committed governed arms, the analyzer degrades gracefully to a
"baseline-only, no verdict" state (mirroring the step-2/step-3 unseeded
fallback).

## 11. Tests needed before any run

Test-first, exactly as PR-6 did — these are written and green *before* any
cache run:

1. **`tests/test_pr7_twin_harness.py` — no-op identity.** `--govern none`
   produces a run **byte-identical** to plain `failure_mode_probe.py`
   (same seed → identical `per_probe`/`topk`/`summary`). Proves the
   experimental hook adds nothing on the baseline path and cannot leak
   into deployment.
2. **Engine-frozen guard.** Assert the governance hook lives only in the
   experimental driver and that deployed `associative_core.py` /
   `fast_associative_memory.py` sha256 is unchanged from the PR-6 baseline
   (and that the hook imports nothing the deployed `forward()` path
   imports back).
3. **Determinism / deployment-parity.** Governed run reproducible (same
   seed → identical artifacts on re-run); baseline `none`-policy vote
   bit-identical to deployment on every arm.
4. **`tests/test_pr7_twin_delta.py` — analyzer discipline.** Imports no
   torch; reads only committed JSON; `geometry_used_as_gate == false` and
   `engine_or_retrieval_change == false`; the **both-shapes rule** is
   enforced (a synthetic governed result that fixes D but worsens B is
   reported `fail`); `one_shot_ambiguity` is never scored pass/fail;
   graceful baseline-only fallback when no governed arms are committed.
5. **Mechanism regression.** Existing `tests/test_failure_mode_probe.py`
   and `tests/test_failure_mode_vision.py` stay green — the harness must
   not perturb the unguarded contradiction/supersession mechanisms.
6. **Provenance-guard tests.** Stem-name ↔ `classes` ↔ `govern` match;
   non-soft payload refused on the merge-path cell.

## 12. Explicit non-goals (hard boundaries)

PR-7 contains **none** of the following; each would reopen a closed
result:

* **No deployed retrieval change.** Engine source stays byte-frozen;
  governance is experimental-only behind `--govern` until a later PR earns
  deployment gates of its own.
* **No read-time trust revival.** No read-time slot deprecation, no
  read-time re-weighting; `mode-conditioned-trust` stays the frozen
  measurand, never a policy.
* **No static geometry gate.** Geometry is provenance only; no cell is
  admitted, excluded, or certified by any geometric property.
* **No slot-granularity trust deprecation.** The unit of action is the
  *incoming write*, not a mature slot at read time.
* **No one-shot forced classification.** `one_shot_ambiguity` stays
  observe-only; write-API provenance metadata for it is noted as a
  *possible future* observable but is explicitly **out of PR-7 scope**.
* **No record-granularity ledger** (path 2) — deferred behind the §5
  write-ledger until a result isolates *scope*, not *selection*, as the
  residual defect.
* **No policy / threshold tuning** to manufacture a pass; actions are
  parameter-free and pre-registered.
* **No broad class-pair search**; only the enumerated panel cells.
* **No implementation in this branch.** The harness, the `--govern` hook,
  the delta analyzer, and the tests are specified here, not written.

## 13. Sequencing — what PR-7 implementation would be (named, not built)

Each step lands on main before the next, mirroring PR-6's stepwise
discipline:

1. **PR-7 step 1 — boundary scaffold (zero behavior change).** Add the
   `--govern {none,annotate,quarantine,refuse}` flag wired to a hook that,
   for every action *including the real ones*, is initially a recorded
   no-op, plus tests 1–3 and 5–6 of §11. Proves the experimental boundary
   and determinism with no governed behavior yet. No cache run.
2. **PR-7 step 2 — one action, one cell.** Implement the cheapest action
   (`annotate`, the null-action floor) and run the twin on a single cell
   (pairA clean control), establishing the delta artifact and §11 test 4.
   Smallest run that exercises the protocol end-to-end.
3. **PR-7 step 3 — full panel, one action at a time.** Run baseline vs
   governed across `direct_harm`, `collateral_harm`, `clean_control`, and
   `merge_path_stale`, scoring against §8. `refuse` (the ceiling) tested
   last.

This memo plans the line; it implements none of it. Deployed retrieval
remains unchanged.
