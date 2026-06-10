# Failure-mode blindness study — injection driver schema (PR-2a)

**Status: PR-2b contradiction-arm result landed** (gentoo,
`vitl14_cifar100_train` — see `PR2B_CONTRA_RESULT.md`). The PR-2c stale arm
has no results yet. Cache-backed runs are gated on
`tests/test_failure_mode_probe.py` + `tests/test_failure_mode_vision.py`
passing — the labels below are only meaningful if the fork and supersession
mechanisms are mechanically valid.

## Question

The two-axis detector (#85/#86/#87, INR breadth, A2b) was developed and
validated entirely on BLENDED/false-collapse failures under drift. VIGIL's
code-level predictions (session 732dc595) say STALE and CONTRADICTORY
failures are structurally different: a stale recall looks like a *confident
correct* retrieval, and a contradictory write is silently forked into a
co-resident prototype. This study measures, per failure mode, whether the
**frozen** #87 detector (orientations, standardization, coefficients,
threshold — no refitting) has any discriminative power.

## Mechanisms (pre-existing engine behavior; the driver modifies nothing)

* **Contradiction fork** — `learn_local`'s bipartite check
  (`payload_sims > 0.5`, associative_core.py) demotes a vigilance hit whose
  payload disagrees into a miss, allocating a co-resident prototype.
* **Provenance** — `track_provenance=True` + `record_ids` stamp every write;
  `slot_records` maps slots back to the writes that formed them. The driver
  stamps **every** write, so a fresh allocation is exactly a slot whose
  record set is `{id}` after the call.
* **EMA freeze** — adaptive alpha `hebb_lr / (1 + ema_beta * hit_counts)`
  means a mature slot barely absorbs a same-slot update: supersession with
  payload cosine > 0.5 merges but keeps decoding to the old label.

## Arms

| Arm | Protocol |
|---|---|
| `clean` | Negative control. No injections; no contra/stale label may fire. |
| `contra` | After each clean epoch write, re-write `rate` of the batch rows with the true key but a uniformly chosen **wrong**-class one-hot payload (hallucinating-writer model). Each injection is classified at write time: `forked` (pre-write nearest sim ≥ effective vigilance AND payload disagreed → co-resident slot), `plain-miss` (below vigilance), `absorbed` (payload agreed; merged), `dropped` (no slot). Only `forked` writes create contradictory slots. |
| `stale` | **Supersession, not absence.** Phase 1: write K→A through the normal learn path until mature. Phase 2 (from `supersede_epoch`): write the same keys K→B through the normal learn path; held-out ground truth for K flips to B. Whether phase 2 merges (payload cosine > 0.5: EMA-freeze stale) or forks (≤ 0.5: co-resident stale) is decided by the engine. Stale = K still retrieves A while B is current ground truth. |

## Slot sets (recomputed from live state at every probe epoch)

* **Contradictory fork slot** — occupied; provenance contains ≥ 1 `forked`
  contra record id; payload still argmax-decodes to the injected wrong label.
* **Stale slot** — occupied; provenance intersects a superseded group's
  phase-1 ids; payload still argmax-decodes to the pre-update label A.

Recomputing each epoch makes the sets self-invalidating: eviction reuse
replaces a slot's record set, and a slot that absorbs enough updates to flip
its decode drops out (both pinned by tests).

## Per-probe labels

Let *wrong* = `vote_correct == 0` (the deployed `forward().argmax` answer,
pinned by test). "Top-1" is the best **raw cosine** candidate before floor
masking — the same convention as the existing `top1_label`/`top1_sim`
telemetry. "Surviving top-k" = candidates with finite post-floor similarity
(the `n_surviving_votes` set).

| Label | Definition |
|---|---|
| `CONTRADICTORY_STRICT` | wrong AND top-1 slot is a contradictory fork slot. |
| `CONTRADICTORY_LENIENT` | wrong AND any surviving top-k slot is a contradictory fork slot. |
| `STALE_STRICT` | wrong AND top-1 slot is a stale slot AND the vote elects that slot's pre-update label (K still retrieves A). |
| `STALE_LENIENT` | wrong AND any surviving top-k slot is a stale slot. |
| `BLENDED` | wrong AND `is_blended == 1` (off-class vote mass > `blend_eps`, the established #82 definition). |
| `OTHER_WRONG` | wrong, none of the above. |
| `CORRECT` | `vote_correct == 1`. |

Strict ⟹ lenient by construction (pinned by test). The flag columns are
**not mutually exclusive** — a wrong probe can be both contradictory-lenient
and blended (the fork-becomes-blend channel is itself a finding). The single
`failure_mode` column collapses via the documented precedence
`CONTRADICTORY_STRICT > STALE_STRICT > CONTRADICTORY_LENIENT >
STALE_LENIENT > BLENDED > OTHER_WRONG`; analyses should prefer the flags.

### Exposure vs causality — what the labels do and do not assert

The labels are **exposure** classifications, not counterfactual causality
proofs. The vote is an aggregate over up to `inference_k` candidates, so a
wrong answer can draw mass from slots outside the flagged set even when a
flagged slot is present.

* **Strict = leading-support exposure.** The raw-cosine top-1 is, on every
  emitted row, also the maximum-weight voter: the floor only masks sims
  below threshold, a row whose top-1 (its max) is masked has no vote and is
  excluded entirely, and softmax weight is monotone in similarity. So a
  strict flag asserts that the single strongest contributor to the wrong
  vote belongs to the fork/stale set.
  * `STALE_STRICT` additionally requires the elected class to BE the stale
    slot's pre-update label (K still retrieves A), which ties the answer's
    identity to the flagged slot — the closest the labels come to causal
    attribution.
  * `CONTRADICTORY_STRICT` does **not** require the elected class to equal
    the fork's injected label; it asserts leading-support exposure plus a
    wrong answer. (`vote_pred_label` and the fork slot's decode are both in
    the row, so the stricter identity-matched subset is recoverable in
    analysis without re-running.)
* **Lenient = top-k implication only.** A lenient flag asserts the wrong
  vote's surviving candidate set *contained* fork/stale support — nothing
  about how much that support contributed.
* **Causal weight is quantified, not asserted:** `contra_vote_weight` /
  `stale_vote_weight` record the actual softmax mass on each flagged set,
  so PR-3 can run dose-response analysis (error rate vs flagged mass)
  instead of treating exposure as attribution.

## CSV schema (`per_probe_injected.csv`)

All columns of the issue-#82 `per_probe.csv` (context, label-free predictors,
evaluation labels — unchanged, byte-compatible with
`benchmarks/analyze_calibration.py` and `benchmarks/heldout_abstention.py`),
plus:

| Column | Meaning |
|---|---|
| `arm` | `clean` / `contra` / `stale` |
| `injection_rate` | contra arm rate (0.0 otherwise) |
| `supersede_epoch` | stale arm flip epoch (−1 otherwise) |
| `probe_index` | row index into the held-out probe set |
| `top1_slot` | raw-cosine top-1 slot id |
| `failure_mode` | precedence-collapsed label (see above) |
| `contradictory_strict` / `contradictory_lenient` | 0/1 flags |
| `stale_strict` / `stale_lenient` | 0/1 flags |
| `n_contra_topk` / `n_stale_topk` | surviving top-k members in each slot set |
| `contra_vote_weight` / `stale_vote_weight` | softmax vote mass on each slot set |

## Integrity guards baked into the driver

* Predictors and evaluation labels come **unchanged** from
  `probe_cross_class_similarity(return_per_probe=True)`; the driver only adds
  slot-composition columns. Its replication of candidate selection is
  verified at runtime (row count, `top1_sim` allclose, `vote_pred` equality)
  and **raises** on mismatch rather than mislabeling.
* NSTP / support-truncation configs are refused (not replicated; not used
  anywhere in the calibration sequence).
* `track_provenance=True` is required; injection without it raises.
* No retrieval intervention, no detector refitting, anywhere in this driver.

## Planned analysis (PR-3, for pre-registration)

Per failure mode: error share; two-axis AUC under the frozen #87 detector;
error-capture at the frozen operating point; confidently-wrong rate
(mode-M errors scoring above the median confidence of correct retrievals).
Redesign triggers (stated before data exists): CONTRADICTORY two-axis
AUC ≤ 0.60 or confidently-wrong > 50% → write-time contradiction detection
required; STALE AUC ≤ 0.55 → recency must enter the readout.
