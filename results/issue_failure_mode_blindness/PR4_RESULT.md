# PR-4 result memo — geometry safety of mode-conditioned trust governance

**Status: step 4 of PR4_DESIGN.md complete. All §6 verdicts are
mechanical outputs of `benchmarks/analyze_pr4_geometry.py` over the
sweep artifacts in `pr4/` (68 new gentoo runs + the PR-3c anchors);
`pr4/pr4_geometry_table.json` is byte-identical on darwin and gentoo
(sha256 `a09c9a04f597…`). Every operationalization of a §6 phrase was
fixed in the analyzer docstring before any PR-4 output was read, except
one scoring-axis correction noted in §6 below. Retrieval was never
touched; everything here is shadow governance.**

## 1. Verdicts

* **H1 (pre-registered replication): FAIL.** The frozen trust rule does
  not reproduce the PR-3c pair-A envelope on the new pair-A-like pair C:
  at seed 0 it breaks correct probes on all three stale-protocol arms
  (stale br 9, soft br 12 with 7 false abstentions, one-shot br 10),
  violating the "stale broken 0 / merge capture with broken 0 / one-shot
  honesty" clauses. 15/18 pair-C cells pass, target-mode improvement and
  the 1-pp bar held on every pair-C cell — but the §6 success criterion
  is the full envelope, and per §6 **no variant can be promoted on a
  failed H1**.
* **H2 (collateral attribution): the pre-registered geometry index is
  falsified as a predictor.** Mean mixed-arm collateral br by ascending
  static index: A 6.7 → C 1.7 → **D 94.0** → B 55.3 → E 56.3. Collateral
  is grossly non-monotone: pair D, whose set-level index (0.0836) is
  within 1% of pair A's, is the most dangerous geometry in the sweep,
  far worse than pair E (index 47% higher). Per §6 this is an ambiguous
  attribution table and **blocks H3 promotion**; per §7 the geometry
  axis must be rebuilt before any "safe-geometry region" can be claimed.
* **H3 (geometry-safe variant): FAIL for all three candidates** —
  mode-conditioned-trust 32/68 fresh cells failing, trust-guarded 32/68,
  trust-downweight 36/68. This is the informative-failure outcome of §6,
  with one sharpening: the failure is not even cleanly geometry-*bounded*,
  because the pre-registered index cannot say where the safe region is.
* **Consequence (per §6): no deployment proposal. The deployed
  `forward()` remains unchanged. The safe policy set collapses back to
  {observe-only, merge-suspect abstention, abstain-tie}** — and
  record-granularity deprecation or write-path refusal becomes the
  explicitly justified PR-5 question, together with a rebuilt geometry
  index (pair D is the test case any future index must order correctly).

## 2. Protocol

68 new runs (pair-B completion s1–2 + soft/one-shot s0–2; pairs C/D/E
full grids: 6 arms × 3 seeds), `pr4_run_matrix.sh`, identical driver,
cache, config, epochs as PR-3c. Before spending the matrix, the driver
re-ran two pinned PR-3c cells (mixed_s0 pair A and pair B) and
byte-compared: **PROTOCOL BYTE-IDENTITY: OK**. All 341 artifacts
sha256-identical gentoo↔darwin. Pair A (all cells) and pair B s0
(clean/contra/stale/mixed) are the PR-3c artifacts reused verbatim and
are marked `tainted_anchor` in the table: excluded from every H1/H3
verdict (§7 leakage), shown only for continuity.

The §4 dynamic check: realized fork-witness co-residency rates sit in a
narrow band (0.68–0.78 on mixed) across ALL pairs — the static index
also fails to predict realized crowding differences, consistent with §1.

## 3. H1 detail — replication on pair C (index 0.0828, nearest A)

Pair C contra/mixed replicate PR-3c pair A cleanly: contra fx 175/196/248
with br 6/1/3 (bar ≈ 22–25); mixed fx 413/353/325 with br 8/0/8; clean
0/0/0 everywhere. The misses are all stale-protocol arms at seed 0:

| cell | trust outcome | §6 clause violated |
|---|---|---|
| pairC/stale/s0 | br 9 (all collateral) | stale broken 0 |
| pairC/soft/s0 | capture 1.0 but br 12, 7 false abst | merge capture with broken 0, ≤5 false |
| pairC/oneshot/s0 | br 10, 38 tie flips | broken 0 |

Seeds 1–2 pass all six arms. So even on deliberately A-like geometry the
rule's zero-harm tail is seed-dependent — PR-3c's "stale broken 0
everywhere" was partly seed luck, which is precisely what pre-registered
replication exists to expose.

## 4. H2 detail — the index is the casualty

Mixed-arm collateral br (mean over seeds; anchors marked †):

| pair | index | trust | trust-downweight | trust-guarded |
|---|---|---|---|---|
| A† | 0.0827 | 6.7 | 8.7 | 6.7 |
| C | 0.0828 | 1.7 | 1.7 | 1.7 |
| D | 0.0836 | **94.0** | 102.7 | 94.0 |
| B† | 0.0845 | 55.3 | 74.3 | 55.3 |
| E | 0.1238 | 56.3 | 75.7 | 56.3 |

The §2 directional prediction (collateral monotone in the index) is
**falsified**. Worse for the index: pair D's harm is mostly **direct**
breakage (mixed direct br 163–206 vs collateral 77–124) — the rule
deprecates the *wrong side at the contested key region itself*, a
failure channel pair B never showed. Pair B's fresh seeds also worsen
its anchor numbers (mixed br 144/145 at s1/s2 vs 77 at s0 — the PR-3c
collateral measurement was the mild seed).

The second §2 prediction **holds**: router verdicts are geometry-stable
(supersession exactly 160 in every mixed run; contradiction 209–237;
conflict pairs 423–454 across all five pairs). The collateral growth is
attributable to the intervention, not the routing — the rule *routes*
correctly and then *acts* destructively where class structure is
entangled in ways the static mean-cosine index does not measure.

Honest accounting against gate 1 (Addendum A.2): the recorded secondary
diagnostic (stale-fork pair cosine) also fails to order pair D (0.0522,
the LOWEST in the sweep, yet the worst collateral). Neither the primary
nor the recorded secondary axis survives contact with pair D. No
post-hoc index is proposed here; building one that orders D correctly is
PR-5 work and must be pre-registered against held-out pairs.

## 5. H3 detail — no variant survives, and the §5 mechanisms resolve

* **Exclusion-vs-attenuation is not the collateral driver** (H2-policy
  branch refuted): trust-downweight(λ=0.25) is *worse* than exclusion on
  every compressed mixed cell (e.g. pair B s1 br 212 vs 144; pair D s1
  380 vs 330) — the attenuated wrong-side mass keeps perturbing
  neighboring elections that full exclusion would at least settle.
* **The guard is a near-no-op at θ=0.8**: trust-guarded differs from
  trust by at most ±1 broken/fixed row in every cell where it differs at
  all. The label-free fork-party proxy sits above θ for essentially every
  deprecated slot in deployment-shaped traffic, so the guard almost
  never selects attenuation. Per §6 this is the "guards cost fix rate
  without reducing collateral" failure mode (degenerately: they cost
  nothing and reduce nothing) — the collateral mechanism is not
  observable through this proxy at this threshold.
* **The required merge-path benchmark itself degrades on D/E**: soft-arm
  capture falls to 0.79–0.93 on pair D with 38–128 false abstentions and
  br 46–180; pair E captures 0.97–1.0 but with 69–143 false abstentions
  and br 24–74. Per §1 this alone fails any candidate outright on those
  pairs.
* **One-shot honesty fails two ways**: pair D oneshot br 127–243; pair E
  tie flips 286–294 per run — above the 204 PR-3c ceiling — i.e. the
  side-effect channel §1 required to be minimized *grows* with
  compression even where br is moderate.
* **Clean stays clean everywhere** (every pair, every seed, all three
  candidates: acted 0, broken 0) — the rule never invents work on
  healthy traffic; the danger is exclusively in how it acts on real
  conflicts.
* Contra containment is the one robust transfer: trust passes the bar on
  contra in 10/11 fresh contra cells (exception pair D s2, br 34 vs bar
  22) with fx 175–379.

## 6. Deviations and limitations

* **One operationalization was corrected after first contact with PR-4
  output** (recorded per §7 discipline): the strict-domination axis
  initially compared `broken` only, which made `entropy-abstain`
  (broken 0 by construction, harm entirely in 500+ correct-row
  abstentions) unbeatable. Collateral was corrected to
  `broken + abstain_on_correct` — the same harm accounting PR-3c used in
  prose — and pinned by test. H1/H2 and every per-arm §6 check were
  untouched by this; only the domination column changed (and no verdict
  flipped from pass to fail or back at the H1/H2/H3 level: all were
  already failing).
* The §3 per-slot collateral-exposure block (registry exposure vs
  label-free proxy) is emitted per policy per run in the table for
  PR-5's guard post-mortem; it was not further analyzed here because H3
  failed before the guard-validation question became decision-relevant.
* Stationary, one encoder, one cache, as scoped. Nothing here transfers
  to drift × fork.
* All verdicts rest on 68 fresh cells over 3 new class sets + pair-B
  completion; pair D may itself be an outlier class-set — but one
  counterexample is sufficient against "geometry-safe", and the burden
  per §6 was on the variants to pass every pair.

## 7. What PR-4 settles

The PR-3c trust lead does not survive pre-registration. Slot-granularity
trust deprecation: replicates its contra/mixed gains on near-A geometry,
fails zero-harm even there on a third of seeds, becomes net-destructive
on a class set the pre-registered compression index scores as benign,
and none of the fully-specified guards help. Write-time rule routing
remains geometry-stable and is worth keeping; *acting on it by excluding
or attenuating whole slots is not deployable*. The safe set remains
{observe-only, merge-suspect abstention (with its own D/E degradation
noted), abstain-tie}. PR-5, if opened, owns: (a) a geometry index that
orders pair D, pre-registered on held-out pairs; (b) record-granularity
action, which requires new engine observables (gate 1 showed the current
logs cannot support it); (c) write-path refusal per PR3_DESIGN.md §11.
