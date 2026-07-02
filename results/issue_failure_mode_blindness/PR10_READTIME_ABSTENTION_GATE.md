# PR-10 — read-time abstention gate (pre-registration, design-only)

**Status: design-only. No implementation in this branch.** This memo
pre-registers the first **acting read-time** governance twin: serving the
`merge-abstain` policy — abstain iff the deployed vote's surviving top-1 slot
is merge-suspect; otherwise the deployed answer unchanged — as a reader-facing
outcome, and certifying that the serving seam perturbs **nothing** else. The
engine stays byte-frozen; deployed `forward()` / `learn_local` are untouched;
the write-path `--govern` seam is `none` in every arm.

**Why this is the next gate.** The program's negatives converge on one design
rule: *write-time evidence, read-time enforcement*. Write-path acting arms are
blocked (G3 `provenance_recoverable_not_harm_free`; G4 per-seed collateral;
PR-8 counterfactual identity refuted past the first divergence). Read-time
*confidence* gating is closed (PR-3a inversion; PR-3c `entropy-abstain` worst
policy in every stationary run). Read-time *slot deprecation* is closed
(PR-4). What survives, with **zero recorded negatives on any non-target arm**,
is per-query abstention driven by the geometry-stable write-time merge-suspect
evidence — measured exactly in PR-9.1(b)
(`PR9_ENVELOPE_RESULT.md`, `pr9/abstention_envelope.json`). PR-10 turns that
shadow-scored envelope into a served contract, or documents where the seam
leaks.

**What PR-10 decides.** Not the envelope's magnitudes — those are already
measured and frozen (PR-9.1(b) §5, including the documented pairD/soft/s0
capture 292/375 = 0.778667, recorded not tuned). PR-10 decides whether an
**acting** readout can serve exactly that envelope: every gate below is an
**exactness** condition against committed artifacts, not a floor. There is no
tuning path inside PR-10; a deviation in either direction is a `fail`.

---

## 1. Mechanism (frozen)

A new opt-in read seam in `benchmarks/failure_mode_probe.py`:
`--read-govern {none, merge-abstain}`, consulted AFTER the deployed
`forward()` vote is computed and scored, BEFORE the per-probe row is emitted.

* **Eligibility rule (parameter-free, committed):** abstain iff the surviving
  top-1 slot of the deployed vote is in the write-time router's merge-suspect
  set — byte-for-byte the rule the frozen scorer's `merge-abstain` policy uses
  (`benchmarks/analyze_fork_governance.py`; the policy-invariant M set; the
  same absorbed-conflict evidence that is geometry-stable at 192/seed on every
  soft arm). No tie trigger, no confidence term, no exclusion, no vote
  recomputation, no threshold.
* **Served-outcome encoding (additive schema):** the governed arm's per-probe
  CSV appends exactly two columns — `served_outcome` ∈ {`answer`, `abstain`}
  and `abstain_reason` ∈ {``, `merge_suspect_led`} — and preserves every
  pre-existing column's value on every row (the deployed vote remains recorded
  so the frozen scorers and the twin comparison still read identical numbers;
  the *reader contract* is the served field). The baseline arm's schema is
  unchanged (the deployment-bit-identity invariant).
* **Summary provenance:** the governed arm's `summary.json` gains one
  `read_govern` block (action, per-arm abstention counts, reason histogram).
  No other artifact changes shape.
* The write path is byte-unaffected by construction (no read-path mutation —
  pinned already by `tests/test_read_path_invariants.py`); G1 makes this a
  measured gate, not an assumption.

## 2. Required cells (frozen)

Governed arms only — the baselines are the **committed** PR-4/PR-3c artifacts;
re-running them is neither needed nor allowed (determinism ties the governed
arm to them directly):

* the PR-4 fresh grid: pairs **B, C, D, E** × arms
  {clean, contra, stale, soft, one-shot, mixed} × seeds {0, 1, 2} —
  identical driver, cache, config, epochs as `pr4_run_matrix.sh`;
* the pairA grid and jitter runs as **report-only anchors** (excluded from
  verdicts, as in PR-4/PR-9);
* compute on gentoo (`feature_cache_vitl14/`), byte-verification on darwin
  (both-host sha256 discipline).

## 3. Gates (all exactness; ALL must hold for certification)

* **G1 — write-stream byte-identity.** `fork_events.csv`, `per_slot.csv`,
  `topk.csv(.gz)` of every governed run byte-identical to the committed
  baseline run; `summary.json` identical after removing the `read_govern`
  block.
* **G2 — answered-stream byte-identity.** The governed per-probe CSV, after
  dropping the two new columns, byte-identical to the committed baseline CSV
  on every run (every pre-existing field of every row, abstained or not).
* **G3 — abstention-set exactness.** Per cell, the multiset of abstained probe
  rows equals the frozen scorer's M set, and the counts equal
  `abstained_merge` in `pr9/abstention_envelope.json` `cells_fresh` (e.g.
  pairD/soft/s0: 300 abstentions — 292 on stale-wrong rows, 8 on correct);
  **0 abstentions on every non-soft arm** (all clean/contra/stale/one-shot/
  mixed cells; envelope: 0 across all 72 non-soft cells and both jitter runs).
* **G4 — trigger purity.** `abstain_reason == merge_suspect_led` on every
  abstained row; forced abstentions 0; tie-triggered anything 0 (the policy
  has no tie path — pinned by `tests/test_pr9_merge_abstain.py`).
* **G5 — determinism.** Same seed → byte-identical governed artifacts;
  gentoo/darwin sha256-stable.

Scoring is by a new analysis-only reader (`benchmarks/pr10_readout_delta.py`,
no torch, reads committed artifacts + the governed runs only), with hermetic
tests. It emits `pr10/readout_delta.json`; verdict `readout-certified` iff
G1–G5 all hold on every required cell, else `fail`. No partial credit, no
`needs_review` tier: nothing here requires judgment, because every expected
value is pre-registered to exact equality.

## 4. What certification means (and does not)

`readout-certified` asserts: **merge-abstain is a certified opt-in served
readout at exactly the PR-9 envelope costs** — capture floors (min over seeds,
soft arms) pairC 1.000 / pairB 0.994667 / pairE 0.969466 / pairD 0.778667;
false-abstention ceilings per run pairC 2 / pairB 0 / pairE 6 / pairD 8
(worst rate 0.327% of correct traffic); zero changed answers; zero actions
outside soft arms. The reader contract becomes
`{answer | abstain(merge_suspect_led)}` for callers that opt in.

It does **not** assert: any change to deployed `forward()`; any claim under
drift, re-embedding, or a different encoder/cache (all PR2–PR10 evidence is
stationary, one encoder); any handling of the residual stale-wrong rows whose
top-1 is not the merged slot (pairD/s0: 83 rows, 22.1% — the PR-3c §2
mechanism amplified by compression); any handling of one-shot ties or
contradiction forks. The residual and the tie/fork classes are the named
targets of **PR-11** (adjudication-window design), not of any PR-10 follow-up
tuning.

## 5. Failure conditions (record the negative; do not tune)

* any write-stream artifact differs (G1) — the seam leaks into state;
* any pre-existing per-probe field differs (G2) — the seam leaks into scoring;
* any abstention outside soft arms, any count ≠ the envelope, any row-set ≠ M
  (G3) — the served policy is not the scored policy;
* any forced or non-merge-reason abstention (G4);
* any cross-host or same-seed instability (G5);
* any wording — artifact, memo, or verdict — implying deployed retrieval
  changed, a floor was re-negotiated, or drift/re-embed validity was earned.

A G1/G2 failure is an instrumentation defect: fix the seam and re-run; the
envelope and this gate do not move. A G3 failure with correct instrumentation
falsifies the claim that the shadow-scored M set is servable — that would be a
substantive negative for the read-time-enforcement architecture and must be
recorded as such.

## 6. Boundary & anti-tuning invariants

Engine `associative_core.py` / `fast_associative_memory.py` sha256 unchanged
(the `pr7_twin_delta.py` baselines); deployed `forward()` / `learn_local`
byte-identical; `--read-govern` opt-in and unreached by deployed retrieval;
`--govern` (write seam) `none` everywhere; geometry never a gate; every action
parameter-free; the PR-9 envelope values are **frozen expected values** — not
floors to re-earn, not margins to move; no seed exclusion; no pairD
special-casing; committed baselines never regenerated. Any breach voids
certification.

## 7. Sequencing

1. **PR-10 step 1** — implement the read seam + `pr10_readout_delta.py` +
   hermetic tests (synthetic + tiny-cache vision, mirroring the PR-8 test
   pattern), own branch, no panel runs.
2. **PR-10 step 2** — governed-arm matrix on gentoo, darwin byte-verify,
   result memo (`PR10_READTIME_ABSTENTION_RESULT.md`), verdict.
3. Parallel track, independent of PR-10: **PR-9.2** — §9A shadow
   certification with a pre-registered write-event-intrinsic identity key
   (the committed identity smoke, `pr8/identity_smoke/`, shows the state-free
   key components match 24/24 — the intrinsic key is viable).
4. After PR-10, regardless of verdict: **PR-11 design memo** — pending-write
   adjudication for the contradiction/tie classes and the pairD residual.

## 8. Explicit non-goals (hard boundaries)

No tie or one-shot trigger; no confidence/entropy/two-axis term anywhere in
the acting path; no write-path action; no engine or deployed-retrieval change;
no slot exclusion or down-weighting; no recovery or §9B/§9C claims; no
re-embedding or drift claims; no capture-semantics re-opening; no new
geometry index; **no implementation in this branch** — the seam, reader, and
verdict semantics are specified here, not written.
