# PR-3c — shadow governance: write-time event evidence governs; read-time classification does not

Executes PR3_DESIGN.md §5 PR-3c. Question: **can write-time event evidence
support safer shadow governance across contradiction, supersession,
one-shot ambiguity, and merge-path stale than naive global policies?**
Answer: **yes — with one refinement to the pre-registered routing — and
nothing read-time comes close.** Everything below is shadow analysis: the
deployed `forward()` is unchanged everywhere; interventions are applied
only to votes recomputed offline from the logged candidate composition.

## 0. Setup and integrity

* 24 gentoo runs (`pr3c_run_matrix.sh`): the exact 21-run PR-3b protocol
  plus the three pair-A seed-0 PR-2 baselines, re-executed with the new
  `<out>.topk.csv` side table (per-probe top-k candidate composition:
  slot, sim, surviving, weight, decode at float32 round-trip precision).
* **Baseline byte-identity: OK** — all 21 per-probe CSVs byte-identical
  to the merged PR-3b artifacts and all 3 seed-0 CSVs byte-identical to
  the PR-2 `per_probe_vision_*.csv` baselines. The new logging changed no
  run behavior.
* **Shadow-readout fidelity:** the policy-`none` vote recomputed from the
  emitted topk text equals the deployed `vote_pred_label` on every one of
  the 59,980 probe rows (raise-on-mismatch, enforced at generation in the
  driver and again in the analyzer; pinned by test including the exact
  0.5/0.5 one-shot ties).
* `pr3c_governance_table.json` is byte-identical recomputed on darwin and
  gentoo (sha256 `d8ca6bc3…`, sklearn fits included).
* Policies see ONLY engine observables (topk composition, per-slot
  `hit_counts`/`last_write_seq`/`usage`/`n_records`, write-event
  observables, frozen #87 score). Registry truth scores outcomes; the
  allowlists are pinned by test.

## 1. The policy set

Pre-registered (§5): `none`, `observe-only`, `entropy-abstain` (frozen
#87 at its Youden threshold), `abstain-tie` (witness-window class-mass
margin < 0.10), `recency-naive` (drop the older `last_write_seq` side of
any witnessed fork), `quarantine-naive` (drop all witnessed-fork sides),
`mode-conditioned` with ambiguous→observe-only (primary) and
ambiguous→abstain (secondary). The write-time router behind
mode-conditioned classifies each conflict-fork pair (incumbent I, fork O)
per epoch from post-fork traffic: old payload side written again →
contradiction → quarantine both; only new side reinforced → supersession
→ deprecate I; neither → ambiguous. Absorbed conflicts
(`payload_cos_incumbent` < 0.9 on a merge) mark the slot merge-suspect →
abstain when it leads the vote.

**Exploratory addition (not pre-registered), motivated by the table
itself:** `mode-conditioned-trust` replaces contradiction →
quarantine-both with contradiction → deprecate the side subsequent
traffic did NOT reinforce (quarantine-both only when both sides keep
being written). Read-time fork witness: ≥ 2 decode classes among
surviving candidates within 0.05 raw cosine of the surviving top-1.

## 2. The comparison table (headline runs; exact counts)

n = probes; W = wrong under `none`; fx = wrong→right; br = right→wrong;
aC/aW = abstentions on would-be-correct / would-be-wrong rows. Full table
for all 24 runs × 9 policies: `pr3c_governance_table.json` and per-run
`*.governance.json`.

**contra_s0 (n=2822, W=264)**
| policy | fx | br | aC | aW | acc 0.9064 → |
|---|---|---|---|---|---|
| observe-only | 0 | 0 | 0 | 0 | 0.9064 |
| entropy-abstain | 0 | 0 | 607 | 16 | 0.6914 |
| abstain-tie | 0 | 0 | 835 | 171 | 0.6106 |
| recency-naive | 157 | **605** | 0 | 0 | 0.7477 |
| quarantine-naive | 72 | 146 | 137 | 114 | 0.8317 |
| mode-cond-observe | 56 | 260 | 74 | 74 | 0.8079 |
| mode-cond-abstain | 26 | 1 | **2099** | 216 | 0.1715 |
| mode-cond-trust | **176** | **1** | 0 | 16 | **0.9685** |

**stale_s0 (n=2192, W=60)**: recency-naive fixes 60/60 (acc 1.0, br 0);
quarantine-naive fixes 0, abstains on 93 correct + 2 wrong, breaks 5
(acc 0.9279); abstain-tie abstains on exactly the 60 wrong (aC=0);
entropy-abstain abstains on 1153 rows of which 1153 correct;
mode-cond-trust fixes 27/60, br 0, aC 5 (acc 0.9827).

**stale-soft_s0 — merge-path stale, the required benchmark (n=2463,
W=385, 374 stale-wrong)**: fork witness fires on 12 probes (4, 5 in
s1/s2) — none of them supersession-related. abstain-tie: aC=1, fixes 0.
recency-naive: acted 12, fixes 0. quarantine-naive: acted 12, fixes 0,
br 2. **Every read-time fork-topology policy fixes or abstains exactly 0
of the 374 stale-wrong rows.** All three mode-conditioned variants
abstain on **374/374** stale-wrong with **3** false abstentions and br 0
(s1: 370/374 captured, 3 false; s2: 380/389 of W, 1 false; the handful
uncaptured are wrong rows whose top-1 is not the merged slot). The
write-time trace (192 absorbed conflicts/run → 32 merge-suspect slots)
is the ONLY evidence any policy successfully used.

**stale-oneshot_s0 (n=2337, W=360, all at the permanent 0.5/0.5 tie)**:
abstain-tie abstains on exactly the 360 tied-wrong rows, aC=0 — the
honest action, perfectly targeted (replicates 360/360, 372/372).
recency-naive answers the post-update side on every tied probe: fx
360/360, br 0 (s1 360, s2 372) — but see §4. quarantine-naive fixes 0
(drops both sides; remaining same-key support re-elects or
force-abstains, aW=12). mode-cond-observe/trust fix 162 (s1 204, s2 162)
with br 0 — a side effect: deprecating background-conflict forks removes
mass from exactly-tied votes and happens to flip them toward the
post-update side. mode-cond-abstain abstains on all 360 tied rows + 18
correct.

**mixed_s0 (n=2530, W=489: 424 contra-wrong + 63 stale-wrong, overlap
0)**: trust fixes 320 (291 contra + 29 stale) with br **1**, aW 34 (acc
0.8067 → 0.9328). recency fixes 313 but breaks 448. Pre-registered
mode-cond-observe: fx 102, br 169. Replication — trust fx/br: s1
296/23, s2 289/24, pairB 398/**77**.

**clean s0/s1/s2 + pairB (n=2532/2508/2688/2568, W=0/12/0/0)**: every
fork policy and the router act 0 times (witness 0, pairs 0) — except
entropy-abstain, which abstains on 1656/2496 correct rows in clean_s1
and 1572/2688 in clean_s2. Zero-harm on clean traffic holds for
everything except the frozen detector.

## 3. Per-policy verdicts against the pre-registered predictions (§9)

* **`observe-only` matches `none` on every metric in every run** —
  prediction confirmed (also pinned by test). Fork load becomes visible
  (witness counts, router verdicts) at zero accuracy cost.
* **`recency-naive` elects the hallucination on contra forks** —
  confirmed and quantified: br 605/495/521 (s0/s1/s2) and 675 (pairB)
  correct probes broken per contra run (21–27 % of correct traffic),
  against 157–248 fixed — it breaks 2–4× more than it fixes. On pure
  supersession it is perfect (60/60, 62/62, jitter 57/57 and 61/61,
  br 0) — and useless on merge-path (fixes 0–4 of ~374).
* **`quarantine-naive` breaks superseded keys** — confirmed: fixes 0
  stale errors anywhere, abstains/breaks 93+5, 60+3, 102+6 correct rows
  on stale s0/s1/s2, and still misses merge-path entirely.
* **`entropy-abstain` (frozen #87) is the worst policy in every
  stationary run** — abstains on 500–1656 CORRECT rows per run while
  capturing 0–16 contra errors (its stale "capture" in soft runs, 157–
  227, comes with ~1300 false abstentions). The PR-3a/3b inversion,
  realized as governance: the existing-detector baseline is decisively
  beaten by doing nothing.
* **`abstain-tie`** is the surgical one-shot policy (360/360 wrong
  abstained, 0 false, all seeds) but indiscriminate under contradiction
  ramps (835/805/764 false abstentions on contra) — tie topology without
  mode evidence cannot tell a live conflict from a tie.
* **Pre-registered `mode-conditioned` (quarantine-both on contradiction)
  fails its own non-inferiority bar on contra**: br 260/193/202 — worse
  than `none` by 5–10 pp. Quarantining both sides discards the
  corroborated true slot along with the fork. The secondary
  ambiguous→abstain variant is unusable under fork ramps (2099 false
  abstentions on contra_s0): ambiguous-pair slots accumulate and touch
  most of the candidate pool.
* **Exploratory `mode-conditioned-trust`** — deprecate the unreinforced
  side instead — is the only policy that improves BOTH modes while
  breaking essentially nothing on pair A: contra fx 176/244/297 with br
  1/3/10; mixed fx 320/296/289 with br 1/23/24; stale br 0 everywhere;
  merge-path 374/370/380 captured; clean untouched. Accuracy vs `none`:
  +6.2/+8.7/+10.0 pp (contra), +12.6/+11.0/+10.0 pp (mixed), +1.0 pp
  (stale), ±0.1 pp (soft, via abstention), 0 (clean). It strictly
  dominates every baseline on the joint (stale-fix, contra-harm) pair in
  every pair-A run.

## 4. Two honest caveats on the headline wins

* **One-shot "fixes" are not resolutions.** recency-naive's 360/360 on
  one-shot is correct only because the benchmark defines truth as the
  latest write; the protocol itself certifies the evidence is
  insufficient (PR3_DESIGN.md §10). The same goes for trust's 162–204
  side-effect fixes. The defensible one-shot outcomes remain abstain-tie
  (abstain on exactly the tied rows) or observe-only — electing either
  side is policy fiat, not inference.
* **Pair-B collateral (br 77).** In mixed_s0_pairB, trust breaks 77
  correct probes (vs 1–24 elsewhere): under pair-B's tighter class
  geometry, deprecated slots also carried support for neighboring
  correct keys. Slot-granularity governance has collateral cost that
  grows as class geometry compresses — net accuracy still improves
  (0.7883 → 0.9034), but the 1-pp non-inferiority bar is not met on this
  run's correct traffic.

## 5. H1 — read-time mode classifier: pre-registered failure confirmed

Binary (contra-wrong vs stale-wrong, logistic over the six label-free
signals, fit mixed_s0 even epochs): within pair A it generalizes (mixed
s1 0.895 n=441/60, s2 0.879 n=403/63; pure-arm pools 0.824–0.956;
one-shot 0.951, soft 0.933, jitter 0.943–0.956) — and **collapses on
pair B: mixed 0.299 (n=513/57, inverted), pure 0.562 (n=347/60)**. The
§9 failure condition ("≤ 0.60 anywhere in the set") fires exactly as
PR-3b predicted.

3-way (with `correct`, which deployment cannot exclude): the classifier
**never predicts stale-wrong — 0 rows in every split**; stale-wrong is
routed to "correct" 59/60 (s1), 61/63 (s2), 51/57 (pairB), and
contra-wrong to "correct" 301/441 and 247/403. The VIGIL prediction that
a stale recall is indistinguishable from a confident correct retrieval
is now a measured confusion row. H1 is closed as a deployment
mechanism on this evidence.

## 6. H2 — write-time event classifier: classes are separable, but the
rule router is what actually works

Multinomial logistic (duplicate-rewrite / contradiction / supersession),
fit on mixed_s0 conflict-relevant events:

* **Repeated-write accuracy meets the ≥ 0.90 bar in-protocol**: with
  timeline features 0.999–1.0 on every repeated-write split (mixed
  s1/s2/pairB, pure pools, jitter) — and the event-local variant is
  already at 0.995–1.0, which means the separation rides substantially
  on protocol
  timing (`seq_gap`: injections re-write immediately, supersessions a
  full epoch later), not only on mode semantics. Geometry-stable
  (pair B unchanged), unlike every read-time signal.
* **Both §9 failure conditions also fire, instructively.** (1) One-shot
  routing: all 96 protocol-certified-ambiguous events are routed
  CONFIDENTLY into a conflict class (event-local: 96/96 supersession,
  median max-prob 0.785; with timeline: 96/96 contradiction, 0.803) — a
  forced-choice classifier has no ambiguous outcome. (2) Absorbed
  supersession: the soft runs' 576 merge-path supersessions are
  misrouted as duplicate-rewrite 521/576 (event-local) and **576/576**
  (timeline) — the fork-trained classifier inherits its protocol's
  blind spot exactly where governance needs it most.
* The rule-based router, by contrast, holds every one-shot pair
  ambiguous (32/32 per run; the 3–7 extra "contradiction" pairs per run
  are background cross-class boundary forks, present in clean-free form
  in all arms) and catches all 192 absorbed conflicts per soft run via
  the merge-suspect threshold. **The governance-relevant H2 lesson:
  write-time evidence is sufficient, but it must be consumed by a
  procedure with explicit ambiguity and an absorbed-conflict route, not
  by a forced-choice classifier.**

## 7. H3 verdict (§9)

* **Pre-registered mode-conditioned routing: informative failure** on
  the contradiction side (quarantine-both breaks 193–260 correct probes
  per contra run; ambiguous→abstain floods under ramps).
* **The exploratory trust refinement passes the success criteria on
  pair A, replicated across seeds**: target-mode improvement on both
  modes (contra error containment 176–297 fixed; merge-stale capture
  374–380/374–389; stale-boundary 27–35/60–62 fixed — bounded by §4's
  causality, the flip-epoch evidence is genuinely one-shot), other-mode
  non-inferiority (clean identical to `none`; stale br 0; contra br
  ≤ 10 ≈ 0.4 pp), and strict dominance over every baseline on the joint
  pair. On pair-B mixed it improves net accuracy but exceeds the 1-pp
  collateral bar (br 77) — geometry-conditioned collateral, consistent
  with PR-3b's theme, and the reason this is a *shadow* result, not a
  deployment proposal.
* **Ambiguity held as a real outcome**: one-shot conflicts route to
  observe-only/abstain, never to a conflict class; `observe-only`
  itself costs nothing and surfaces the latent fork load.

## 8. What this does not establish

* No deployed retrieval change is proposed or justified here; the trust
  refinement is post-hoc relative to the design memo and needs
  pre-registered replication (new seeds/pairs at minimum) before any
  write-path or read-path engineering.
* All conclusions are stationary, one encoder, one cache; drift × fork
  remains PR-4 territory.
* The router's timeline evidence needs ≥ 1 post-fork epoch; boundary-
  epoch supersession errors (the 60-row transients) are undecidable at
  the epoch they occur — only recency-by-fiat fixes those, at proven
  cross-mode cost.
* Pair-B trust collateral (br 77) is measured once; characterizing how
  collateral scales with class-geometry compression is open.
* H2's event-local separability is partly protocol timing (`seq_gap`);
  a deployment claim would need timing-randomized conflict traffic.

## 9. Files

* `pr3c_run_matrix.sh` — 24-run protocol + baseline byte-identity check
* `pr3c/per_probe_<run>.csv` + `.per_slot.csv` + `.fork_events.csv` +
  `.topk.csv.gz` + `.summary.json` — gentoo-computed artifacts
* `pr3c/per_probe_<run>.governance.json` — per-run policy table
* `pr3c/pr3c_governance_table.json` — collated table + H1 + H2
  (byte-identical darwin/gentoo)
* `benchmarks/analyze_fork_governance.py` — policies, router, studies
* `tests/test_pr3c_shadow.py` — 10 hermetic gates (43 failure-mode
  gates total)
