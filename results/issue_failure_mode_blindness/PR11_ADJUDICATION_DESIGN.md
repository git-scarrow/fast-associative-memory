# PR-11 — pending-write adjudication window (design memo, no implementation)

**Status: design-only. No policy code in this branch.** This memo designs the
adjudication window for the three residual harm classes PR-10 explicitly did
not certify, and pre-registers the one bounded experiment that decides whether
the window is worth implementing. The claim under investigation is NOT "write
governance is now needed." It is: **"the residual harms excluded by PR-10 are
better addressed by a pending-write adjudication window than by expanding
reader-time abstention"** — and the experiment below is constructed so that
either answer is a decisive, recordable outcome.

Every number in this memo is read from committed artifacts
(`pr4/pr4_geometry_table.json` `governance` block unless cited otherwise);
nothing here is a new measurement.

---

## 1. The PR-10 boundary

**What PR-10 certified** (`PR10_READTIME_ABSTENTION_RESULT.md`,
verdict `readout-certified`, 92/92 cells): `merge-abstain` — abstain iff the
deployed vote's surviving top-1 slot is in the write-time merge-suspect set M —
is a served, opt-in reader contract `{answer | abstain(merge_suspect_led)}` at
exactly the PR-9 envelope costs: 5,118 abstentions all on soft arms, 0 on all
74 non-soft cells, 0 changed answers, capture floors C 1.000 / B 0.994667 /
E 0.969466 / D 0.778667, worst false-abstention 0.327% of correct traffic.

**What PR-10 explicitly did not certify** (gate §4; result memo, verbatim
targets of this PR). Residual harm magnitudes, from the committed `none` rows,
fresh pairs B–E × seeds 0–2:

| residual class | where | answered-wrong rows left standing |
|---|---|---|
| 1. soft-arm stale-wrong whose top-1 is NOT the merged slot | soft arms | 167 total — pairD **83/38/30** (22.1% of stale-wrong at s0), pairE 12/1/1, pairB 2/0/0, pairC 0 (PR-3c §2 mechanism, compression-amplified) |
| 2. one-shot supersession (tie/near-tie ambiguity) | oneshot arms | 3,834 `wrong_none` total (per-run 211–402; B 986 / C 679 / D 1,023 / E 1,146) |
| 3. contradiction forks (co-resident incompatible memory) | contra arms | 4,491 `contra_wrong` total (per-run 259–587; B 955 / C 889 / D 1,265 / E 1,382) — plus the contra component of mixed arms |
| (adjacent) one-hot stale supersession | stale arms | 688 `stale_wrong` total — listed for completeness; same evidence family as class 1, secondary lens only |

**Why expanding merge-abstain cannot reach these.** Three independent reasons:

1. **The evidence is absent where the harm is.** `n_merge_suspect_events` = 192
   on all 18 soft cells and **0 on every other cell** of both committed tables.
   Classes 2 and 3 live entirely on arms where M is empty; no re-scoping of
   merge evidence can act there. A different evidence class is required.
2. **M's definition is threshold-frozen.** M = absorbed writes with
   `payload_cos_incumbent < MERGE_SUSPECT_COS` (0.9, frozen since PR-3). The
   residual-1 rows' top-1 slots took no absorbed write at all — the harm
   arrives via a *different* co-resident slot outvoting the merged one
   (PR3C_RESULT.md §2). Widening the cosine bound is tuning (banned) and would
   not put those slots into M anyway.
3. **Trigger-weakening is expressible without new evidence** — abstain when M
   intersects the *support* rather than leads it. That IS "expanding
   reader-time abstention", and this memo does not assume it fails: it is
   policy P2 below, included precisely so the claim under investigation is
   tested against its strongest rival rather than declared.

**Why the failed policies do not close the question.** The committed tables
already contain contradiction-driven policies, and their negatives are
action-shaped, not evidence-shaped: `quarantine-naive` acts *only* through
forced abstentions (`abstained == abstained_forced` on every contra cell —
e.g. pairB/contra/s0: 232 abstained, 232 forced, 142 on correct rows) because
it excludes candidates until nothing survives; `mode-conditioned-abstain`
abstains ~1,900–2,500 rows/run on contra arms with ~1,600–2,100 of them false.
Both fail by **candidate exclusion**, the same mechanism the PR-9.1(a) desk
check identified behind every forced abstention. Meanwhile the router verdicts
those policies consumed are, at final epoch, largely *correct in aggregate
direction*: ~208 of ~235 contra-arm fork pairs adjudicate `contradiction`,
160 of 192 stale-arm supersessions adjudicate `supersession`, clean arms have
zero router state of any kind. **No committed policy has ever paired the
adjudicated verdicts with merge-abstain's certified action** (led-only
abstention, no exclusion, no vote recomputation). That untested pairing is
PR-11's design space.

## 2. The pending-write model

States are a **read-side overlay on slots, per epoch**, computed from the
write-event stream. The engine stays byte-frozen: every write lands exactly as
today (G1 write-stream byte-identity is inherited as a hard gate); what changes
per state is only what the reader is served. This is the same architecture
rule PR-10 certified — write-time evidence, read-time enforcement — extended
from one static set (M) to a state machine with a time axis.

| state | a slot enters when… | reader sees |
|---|---|---|
| **candidate** | an incoming write is between arrival and the vigilance-gate outcome (new-slot / absorbed / forked). Instantaneous in the current engine; never stored, never reader-visible. Named here so the vocabulary covers a future write-path hold, which PR-11 does NOT propose. | nothing (state does not persist) |
| **readable** | default: occupied slot in no overlay set | normal answers (`served_outcome = answer`) |
| **pending** | the slot is a member of a fork pair whose verdict at the current epoch is `ambiguous` — fork observed (`outcome == forked`, `payload_cos_incumbent ≤ CONFLICT_COS`), no corroborating same-side traffic yet. **The adjudication window is the epochs a pair spends in this state.** | pending-led top-1 → held: `served_outcome = abstain`, reason `fork_pending` (the benchmark encoding of "deferred until adjudicated") |
| **quarantined** | the slot's pair adjudicates `contradiction` (old side receives corroborating traffic after the fork — `old_n > 0`). Both members enter; contradiction is absorbing (corroboration counts only accumulate). | quarantined-led top-1 → `abstain`, reason `contradiction_unresolved`. Both sides blocked from *leading*; neither is excluded from the candidate set (no forced-abstention path, structurally) |
| **superseded** | either (a) the slot is the incumbent of a pair adjudicated `supersession` (`new_n > 0, old_n == 0` — router `deprecate` set), or (b) the slot absorbed a merge-suspect write (M — the degenerate in-place supersession, already certified) | superseded-led top-1 → `abstain`; reason `merge_suspect_led` for (b) — the PR-10 contract, unchanged — or `superseded_led` for (a) |
| **audit-only** | any recorded evidence with no reader-visible consequence: annotations, the shadow ledger, `slot_records` provenance, per-slot flags | nothing. **Audit-only is by definition not governance** (this program's standing rule); no PR-11 claim of "governed" may attach to evidence that only lands here |

Transition graph (per fork pair): `pending → quarantined` (old-side
corroboration), `pending → superseded` (new-side-only corroboration),
`superseded → quarantined` (late old-side corroboration — supersession is NOT
absorbing; `router_state`'s conservative precedence already encodes this),
`pending → pending` (no traffic: the one-shot case, a window that never
closes). All transitions are already computed by the frozen scorer
(`analyze_fork_governance.py:267-343`, `verdict_by_epoch`); PR-11 invents no
new classifier.

## 3. Evidence inputs

**Write-time evidence (runtime-available, label-free).** The
`POLICY_VISIBLE_EVENT` allowlist (`analyze_fork_governance.py:116-120`):
`epoch, record_seq, outcome, pre_sim, payload_cos_incumbent,
effective_vigilance, incumbent_slot, incumbent_hit_counts,
incumbent_last_write_seq, incumbent_n_records, owner_slot` — the
`fork_events.csv` stream minus ground truth. Everything the router consumes is
on this list, and PR-10 step 1 proved the *driver can build the router live*
over its own event stream (the seam imports the frozen scorer's
`build_writetime_router`/`router_state` — `failure_mode_probe.py`, merged
`6391654`). Consequence: **all four router sets (merge, quarantine, deprecate,
ambiguous) are runtime-available today at zero new instrumentation.** Only M
is currently consulted.

**Read-time evidence (runtime-available).** The vote's rank-ordered surviving
candidate list (`POLICY_VISIBLE_TOPK`: rank, slot, sim, surviving, weight,
decode) and quantities derived from it (top-1 slot identity — the certified
trigger input; `top1_top2_margin`; decode-class mass gap). The advisory
confidence scalar (`forward_with_confidence`) exists but is refuted as a
failure-zone signal (PR-3a AUC 0.454/0.120) and is banned from acting paths.

**Provenance evidence (recorded, NOT reachable at retrieval).**
`slot_records` is opt-in and never read at retrieval; the FAM wrapper's
`learn_local` does not thread `record_ids`, so per-record provenance is
unreachable from the serving path without an interface change (out of scope).
Eviction leaves zero audit residue — router pairs whose members evict before a
snapshot are unroutable (`decode_at → None`, "nothing to route"), a standing
coverage hole. Cross-run event identity breaks after the first supersession
epoch (PR-8: 8/24 key match; state-free key components 24/24 —
`pr8/identity_smoke/`), which is why PR-9.2, not PR-11, owns identity claims.

**Scorer-only evidence (NOT available to any runtime policy).** Ground-truth
`event_class`/`injected_label` in `fork_events.csv`; per-probe true labels and
the label-derived `margin_bucket`/`true_margin` (the PR-3 forced-zone lesson);
registry roles in `per_slot.csv` (`contra_fork`, `stale_superseded`). These
score policies; they may never feed one — pinned by the allowlists.

**Unavailable, full stop.** Payload tensors of write-seam-quarantined records
(the ledger stores argmax labels; `retained_recoverable` overstates); any
drift/re-embed/other-encoder signal (all evidence stationary, one encoder).

## 4. Policy options

All three candidates are **parameter-free set-membership triggers on the
surviving top-1 slot** — merge-abstain's certified action shape — differing
only in which overlay sets they consult. None excludes a candidate,
recomputes a vote, or contains a tie/confidence/margin term (no unconditional
tie abstention, per the PR-9.1(a) refutation: 23.6–36.8% false abstention on
contra/mixed, 24 clean-arm actions/run on D/E). Because clean arms carry zero
router state (0 conflict pairs, 0 merge events, every clean cell, both
tables), all three have **structurally zero clean-arm actions** — a clean-arm
action in the experiment is an instrumentation bug, not a finding.

**P1 — adjudicated-only led-abstain (the conservative policy).**
Abstain iff top-1 ∈ `quarantine(E) ∪ deprecate(E)` at the probe's epoch —
act only on *resolved* verdicts, never during the window.
*Catches:* post-adjudication contradiction leads (class 3; the ~208/run
adjudicated contra pairs) and fork-half supersession leads (the stale-arm
lens). *Risks:* blind during the window (all pre-resolution harm uncovered)
and blind to one-shot forever (its pairs never adjudicate); false abstention
on correct rows led by quarantined slots — contra arms interleave correct
traffic on fork-party slots, and the mode-conditioned rows show that exposure
can be large when the action is wrong; whether the led-only action tames it
is exactly what has never been measured. *Falsified by:* the committed
contra/mixed cells — false abstention > 5% of correct traffic on any run, or
aggregate `contra_wrong` capture < 0.5 on any fresh pair, refutes it; any
clean-arm action voids the run.

**P2 — merge-in-support abstain (the pairD-residual policy, and the
"expanding reader-time abstention" rival).**
Abstain iff **any surviving candidate** ∈ M (support-membership instead of
led). Same frozen evidence as the certified contract; strictly weaker trigger;
zero new sets. *Catches:* residual-1 rows **if** their mechanism is "merged
slot present but outvoted" — the PR-3c §2 reading. *Risks:* the merged slot
sits in the support of many correctly-answered soft-arm rows, so false
abstention may explode; or the residual rows may not contain the merged slot
in support at all, in which case the mechanism hypothesis itself is falsified
(informative either way). *Falsified by:* the committed soft cells — capture
gain < 42 of the 83 pairD/s0 residual rows, or false abstention > 5% on any
soft run, refutes it. **If P2 passes on class 1, the adjudication window is
NOT needed for that class** — that is the honest test of this memo's claim.

**P3 — pending-window abstain (the contradiction-fork + one-shot policy).**
Abstain iff top-1 ∈ `quarantine(E) ∪ deprecate(E) ∪ ambiguous(E)` — P1 plus
the pending state: forks are held while unadjudicated, which also covers the
window-that-never-closes. One-shot arms carry exactly **32 permanently
ambiguous pairs per run** (every B–E oneshot cell); stale arms carry the same
count of never-corroborated pairs alongside their 160 adjudicated
supersessions — whether those 32 are right-censored windows or structurally
dead keys is a question the scan's resolution-lag readout settles.
*Catches:* one-shot supersession leads (class 2, 3,834 rows in scope) plus the
pre-resolution contra harm P1 misses. *Risks:* pending-led false abstention on
*benign* forks — a legitimately new fact's slot is pending until corroborated,
so early traffic to honest new content gets held; this is the direct,
measurable cost of the window, and the recorded bar is high: unconditional
abstain-tie already achieves 354/354 wrong-row capture with 0 false
abstentions on pairB/oneshot/s0, degrading to ≤44 false/run on D/E. P3 must
beat tie's one-shot capture-to-false ratio without tie's clean/contra
failures, or the window buys nothing over a (banned but instructive)
tie trigger. *Falsified by:* the committed oneshot cells — capture < 0.5 of
`wrong_none` aggregate on any fresh pair, or false abstention > 5% on any
run (any arm), refutes it.

P1 ⊂ P3 by construction: their delta isolates the value of the pending state
itself, which is the memo's actual question.

## 5. The minimal experiment (PR-11.1 — one PR, analysis-only, no new runs)

**Form.** A registered scorer extension under the exact PR-9.1(b) protocol:
add P1/P2/P3 to `POLICIES` in `benchmarks/analyze_fork_governance.py`
(set-membership triggers; no new constants — `CONFLICT_COS`,
`MERGE_SUSPECT_COS` frozen; `TIE_MARGIN` untouched and unused), re-emit both
governance tables from the **committed** PR-4/PR-3c per-run artifacts, plus
one sidecar: a **resolution-lag scan** reading `verdict_by_epoch` to report,
per fork pair, (fork epoch, epoch of first non-ambiguous verdict, final
verdict), and per harm class the fraction of harmed rows occurring before vs
after their leading slot's pair resolves.

**Fixed inputs.** The committed per-run artifacts (per-probe CSV, topk,
fork_events, per_slot) of the PR-4 fresh grid and PR-3c runs at current main
(`76fe7eb`) — nothing regenerated, no gentoo required (PR-9.1(b) proved this
path byte-reproducible on darwin).

**Fixed outputs.** Re-emitted `pr4_geometry_table.json` /
`pr3c_governance_table.json` (additive-only, scripted regression guard:
changed = 0, removed = 0 on every pre-existing `(run, policy, counter)`;
merge-abstain rows byte-identical), plus
`pr11/adjudication_scan.json` (per-cell capture / false-abstention /
clean-action counters for P1–P3, per-class before/after-resolution harm
split, resolution-lag histogram) and `PR11_SCAN_RESULT.md`.

**No tuning path.** All three policies are parameter-free; the acceptance
bounds below are fixed in this memo before any execution; a failed policy is
recorded, not adjusted. Any variant trigger conceived after seeing the scan
requires a new pre-registration on a new branch.

## 6. Acceptance criteria (pre-registered; success and failure both decisive)

**Instrumentation gates (hard; violation = defect, fix and re-run, bounds do
not move):** regression guard passes; merge-abstain rows byte-identical
everywhere; clean-arm actions = 0 for all three policies; changed answers = 0
and forced abstentions = 0 for all three policies on every cell (structural
for led-triggers — no exclusion path exists).

**Per-policy GO** iff on fresh pairs (B–E; pairA remains a report-only tainted
anchor), all of:

* false abstention ≤ **5% of correct traffic per run on every cell, every
  arm** (the standing program ceiling; the certified contract sits at 0.327%);
* aggregate capture over seeds ≥ **0.5 of the policy's named class on each
  named pair** — P1: `contra_wrong` on all four pairs; P2: the pairD soft
  residual (≥ 76 of 151 rows; per-cell reported, s0 ≥ 42/83 named
  individually); P3: oneshot `wrong_none` on all four pairs AND the P1
  criterion.

**Window verdict** (the PR-11 question, answered by comparison, not by any
single GO):

* **Adjudication window worth implementing** iff P3 GOes AND its pending-led
  component does non-redundant work: rows captured by `ambiguous(E)`-led
  abstention and by neither P1's sets nor P2's trigger ≥ **0.5 of the one-shot
  captured mass**. Then PR-11.2 is a PR-10-shape certification of a new
  opt-in mode (working name `--read-govern fork-hold`), on its own branch,
  with exact-count envelopes drawn from the re-emitted table — the
  PR-9.1(b) → PR-10 pipeline, reused verbatim.
* **Static-expansion outcome** iff P1 and/or P2 GO but the pending component
  is redundant under the bound above: the residuals are addressable by
  reader-time set expansion; the window claim is recorded as unsupported;
  any deployment path certifies the static policies instead.
* **Negative outcome** iff no policy GOes: read-time enforcement over the
  existing write-time evidence cannot cover the residuals; the recorded
  escalation options are §9B write-event-intrinsic authority (via PR-9.2's
  key) or explicit acceptance of the residual — not threshold motion.

**PR-10 invariance (absolute).** PR-11.1 touches no driver, no seam, no
engine file, no committed baseline; the certified merge-abstain contract and
its envelope are never edited. Any policy this program later serves is a NEW
`--read-govern` mode behind its own pre-registered certification branch;
`merge-abstain` semantics are immutable. A PR-11 claim of "governed" attaches
only to reader-visible outcome changes (`served_outcome`/`abstain_reason`);
evidence that lands in audit-only state supports no such claim.

## 7. Implementation-agent prompt

> Implement PR-11.1 exactly as pre-registered in
> `PR11_ADJUDICATION_DESIGN.md` §5–6 on branch `feat/pr11-adjudication-scan`.
> Scope: (1) add three parameter-free policies to `POLICIES` in
> `benchmarks/analyze_fork_governance.py` — `adjudicated-abstain` (top-1 ∈
> quarantine ∪ deprecate), `merge-support-abstain` (any surviving candidate ∈
> M), `pending-abstain` (top-1 ∈ quarantine ∪ deprecate ∪ ambiguous) — reusing
> the existing `router_state` sets and the merge-abstain code path (no
> exclusions, no vote recomputation, no new constants, no tie/confidence
> terms); (2) re-emit both governance tables from committed artifacts with the
> PR-9.1(b) scripted regression guard (changed=0/removed=0, merge-abstain rows
> byte-identical); (3) emit `pr11/adjudication_scan.json` +
> `PR11_SCAN_RESULT.md` scoring the pre-registered gates of §6, including the
> resolution-lag scan and the pending-nonredundancy bound. Hermetic tests in
> the `test_pr9_merge_abstain.py` style (semantics fixtures per policy;
> structural pins: forced=0, clean actions=0, changed answers=0). Do NOT touch
> `benchmarks/failure_mode_probe.py`, any engine file, any committed baseline,
> or the PR-10 artifacts; do NOT run anything on gentoo; do NOT adjust any
> bound in §6 — a failed gate is a recorded result. STOP and report if the
> regression guard fails or any structural pin fires. Deliverable: one branch,
> one result memo, verdict ∈ {window-GO, static-expansion, negative} per §6.

## Non-goals (hard boundaries)

No implementation in this branch; no engine or deployed-retrieval change; no
write-path action of any kind (candidate-holding is vocabulary, not proposal);
no unconditional tie abstention; no confidence/entropy/margin term in any
acting path; no threshold introduction or motion; no drift/re-embed claims;
no record-granularity or provenance-interface change; no reopening of
merge-abstain's certified semantics or envelope.
