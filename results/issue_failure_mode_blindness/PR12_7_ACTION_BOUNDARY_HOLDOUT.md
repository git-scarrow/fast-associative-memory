# PR-12.7 — Action-boundary holdout validity for W2:F1b (pre-registration DRAFT)

*Design-only. No implementation, no scan, no artifact, no harness-code or
FAM-core change is authorized by this memo. `harness_boundary_sim.py`,
`reader_utility_score.py`, `action_boundary_score.py`, and every
PR-12.1–12.6 artifact/verdict/gate/output/memo section remain
byte-frozen. PR-10 merge-abstain remains the only certified reader
contract; the operational posture on witness-window rows remains
deferral. §§1–21 are frozen at commit; §22 is reserved append-only for a
separately-authorized run.*

---

## 1. Hypothesis

PR-12.6 established, on a 16-unit test panel drawn from **pairs B and D**
at seeds s1/s2, that the registered label-free policy **W2:F1b** acts on
W2 witness-window one-shot rows and defers on contra rows while clearing
every registered wrong-action ceiling
(`action-boundary-evidence-GO(W2:F1b)`, merged to main `9a7537e`).

**H0 (null, the honest default):** W2:F1b's boundary is an artifact of the
pairs (B, D) it was frozen against. On genuinely held-out fork
configurations it either loses coverage below the §11 floor, drops
acting precision below 0.75, or leaks wrong-action mass over a §12–§15
ceiling — i.e., the 12.6 evidence does **not** generalize.

**H1 (what a GO would support, and only offline):** on a held-out panel
that no PR-12.4/12.5/12.6 aggregate has seen, the *same byte-frozen*
W2:F1b — with *no* threshold change and *no* fitting on any holdout row —
still satisfies G-A1–G-A3 on one-shot holdout units and G-C1–G-C3 on
contra holdout units under the §13/§14 ceilings.

This memo pre-registers the test **and** the prior question it depends
on: whether a valid holdout for W2:F1b exists at all, and if not, what
the smallest input-generation/provenance step to obtain one is.

## 2. Scope

Offline expected-value evidence about the *generalization* of a single
already-registered act-versus-defer policy over *committed* packets and
*deterministically regenerable* packets — nothing else. In scope: the
W2:F1b decision on held-out one-shot and contra rows; the validity of the
holdout itself. Out of scope and explicitly forbidden here: any policy
change, any threshold change, any new policy family, any fitting on
holdout rows, any FAM-core/engine execution, any new random seed, any
deployment/prompting/promotion/ingestion/autonomous/live-acting claim,
any new reader contract. PR-12.1–12.6 verdicts stand unchanged.

## 3. Holdout source classification — determined **C**

Classification is one of: **A** committed held-out packets already exist
and were unused in 12.4–12.6; **B** committed inputs exist but were
indirectly exposed through prior aggregate results (weak holdout); **C**
new held-out inputs must be *generated* under a pre-registered
cache/provenance step; **D** the harness cannot generate or address the
needed holdout without capability work.

**Determination: C** (a *strong* holdout is reachable, but only via a
pre-registered generation/provenance step; it does not yet exist as
packets). Basis, from the read-only inspection below:

* **Paths inspected.**
  `results/issue_failure_mode_blindness/pr10/governed/` (registry truth,
  279 CSVs); `harness/harness_policy.json` (all scan cell configs
  `scan`/`scan12_2`/`scan12_3`/`scan12_4`); the witness-window packet
  trees `pr12_1/`, `pr12_2/`, `pr12_3/`, `pr12_4/`, `pr12/`
  (`audit_packet.jsonl` / `memory_packet.jsonl`); the gated scorers
  `harness/reader_utility_score.py` (12.5) and
  `harness/action_boundary_score.py` (12.6, `CELLS`/`SEEDS`); the
  committed hazard-governance sources `pr4/pr4_geometry_table.json` and
  the `pr4/` per-probe grids; `pr10/pr10_step2_run_matrix.sh`.

* **Committed vs untracked.** All governed truth, all witness-window
  packets, all scan configs, and all hazard-governance sources are
  committed at the pin. The only untracked paths in the tree are
  unrelated local junk (`.DS_Store`, `proxmox-jobs.sqlite3*`,
  `pr9_2/runs/`) — none of it a holdout input.

* **Available panels/seeds.** Registry truth exists for **pairs A–E** ×
  traffic {clean, contra, mixed, stale, stale-oneshot, stale-soft,
  stale-jitter0.05/0.15 (pairA only), g5twin (pairD only)} × seeds
  {s0, s1, s2}. Witness-window packets, however, exist for only **pairs
  {A (clean), B (contra, oneshot), D (contra, oneshot, stale-soft)}** ×
  {s0, s1, s2} × windows {W1, W2} (plus the earlier prototype/C1–C3/D1–D2
  design iterations at s0). No witness-window packet exists for **pair C
  or pair E**, nor for traffic {mixed, plain-stale, jitter, g5twin}.

* **One-shot and contra traffic both present.** Yes, for the un-run
  pairs: `per_probe_stale-oneshot_pair{C,E}_s{0,1,2}.csv` (one-shot) and
  `per_probe_contra_pair{C,E}_s{0,1,2}.csv` (contra) are all committed in
  `pr10/governed/`. Both harm classes W2:F1b must handle (act vs defer)
  are therefore available on the held-out pairs.

* **W2 witness-window features reconstructable.** Yes — but only by
  *running the frozen emitter*. The W2 features W2:F1b reads (shape,
  presented width, candidate `decode_class`, `basis` = witness
  co-resident vs deployed vote, sole-witness structure) are emitted into
  `memory_packet.jsonl`/`audit_packet.jsonl` by `harness_boundary_sim.py`
  from a governed run-stem. The emitter is fully **data-driven from
  `harness_policy.json`** (`run_cell` iterates `policy["cells"]`, each
  cell = `{run_stem → a committed governed path, hazard_governance → a
  committed source}`), so addressing pairs C/E is a *config-only* cell
  addition pointing at inputs that already exist — no Python change. This
  is why the class is C, not D.

* **F1b quiet-cell features computable without forbidden labels.** Yes.
  `CellCtx` derives `n_contradiction_pairs`/`n_ambiguous_pairs` from
  `record_type` tallies (`contradiction_pair_review` /
  `ambiguous_pair_review`) and `never_resolving_slots` from
  `pair.incumbent_slot`/`owner_slot` — all structural router/witness
  fields the emitter produces independent of any truth label, exactly as
  audited for 12.6. The quiet-cell guard
  `n_contradiction_pairs ≤ n_ambiguous_pairs` is therefore computable on
  pairs C/E from the generated packets alone.

* **Prior-aggregate exposure.** **None.** A grep of every committed scan
  cell config (`harness_policy.json`) and every `pr12_*` result JSON for
  `pairC|pairE|per_probe_mixed|per_probe_stale_pair|jitter|g5twin`
  returned empty. Pairs C/E and the un-run traffic types were never a
  cell in any 12.1–12.6 scan, never entered a 12.4/12.5/12.6 aggregate,
  and played no role in freezing any W2:F1b threshold. (They do appear in
  the PR-4 geometry table as *hazard-governance* rows — `pairC/clean|
  contra|oneshot/s{0,1,2}` etc. — which is the structural cross-run
  instrumentation source, **not** a 12.x aggregate and **not** a policy
  input; §7 forbids the policy from reading it.)

* **Strong / weak / invalid.** The **existing** witness-window packets
  (pairs A/B/D, s1/s2) are **weak-to-invalid** as a holdout: s1/s2 are
  the very units 12.4/12.5/12.6 gated on, and s0 is the 12.6 development
  partition — all exposed. A **strong** holdout is obtainable on **pairs
  C and E** (unexposed pairs, both harm classes present, real committed
  hazard-governance, label-free features reconstructable) — but its
  witness-window packets **do not yet exist** and must be generated
  (§4). No genuinely-new *seed* is offline-reachable: minting s3+ requires
  re-running the FAM engine (`--seed` in the pr10 matrix), which is
  forbidden here and is not a stdlib-offline operation. The holdout is
  therefore held-out on the **pair axis** at the existing seeds, which is
  stated plainly as its scope and its limit.

**Consequence for honesty (per the standing rule):** because no strong
holdout exists *as packets today*, this memo does **not** claim holdout
evidence and does **not** dress the weak existing packets up as a
holdout. It registers the input-generation/provenance step (§4) as the
gating prerequisite, and registers `holdout-insufficient` (§20) as a
first-class outcome if the generated panel lacks power (§16 G-H4).

## 4. Exact input artifacts and generation plan (pre-registered cache/provenance step)

Two stages, both separately authorized later; neither runs now.

**Stage A — holdout packet generation (provenance-controlled).** Extend
`harness/harness_policy.json` with a new `scan12_7_holdout` block whose
cells point *only* at already-committed governed run-stems for the
held-out pairs, and run the **byte-frozen** `harness_boundary_sim.py`
emitter (no code edit; the file's sha256 must be unchanged across the
run) to emit witness-window packets into a new cache directory
`results/issue_failure_mode_blindness/pr12_7_holdout_cache/`. Registered
holdout cells (exact run-stems, all committed at the pin):

| cell | run-stem (`pr10/governed/…`) | arm | hazard-governance source (committed) |
|---|---|---|---|
| `pairC_oneshot_s1` | `per_probe_stale-oneshot_pairC_s1` | oneshot | `pr4/pr4_geometry_table.json#governance#pairC/oneshot/s1` |
| `pairC_oneshot_s2` | `per_probe_stale-oneshot_pairC_s2` | oneshot | `…#pairC/oneshot/s2` |
| `pairE_oneshot_s1` | `per_probe_stale-oneshot_pairE_s1` | oneshot | `…#pairE/oneshot/s1` |
| `pairE_oneshot_s2` | `per_probe_stale-oneshot_pairE_s2` | oneshot | `…#pairE/oneshot/s2` |
| `pairC_contra_s1` | `per_probe_contra_pairC_s1` | contra | `…#pairC/contra/s1` |
| `pairC_contra_s2` | `per_probe_contra_pairC_s2` | contra | `…#pairC/contra/s2` |
| `pairE_contra_s1` | `per_probe_contra_pairE_s1` | contra | `…#pairE/contra/s1` |
| `pairE_contra_s2` | `per_probe_contra_pairE_s2` | contra | `…#pairE/contra/s2` |
| `clean_pairC_s1` | `per_probe_clean_pairC_s1` | control | `…#pairC/clean/s1` |
| `clean_pairE_s1` | `per_probe_clean_pairE_s1` | control | `…#pairE/clean/s1` |

The emitter runs under the same `W1`/`W2` shapes as 12.4 (frozen). Seeds
s1/s2 mirror the 12.6 *test* partition seeds; s0 is **not** used (dev
partition; would re-expose). The `.governance.json`/geometry-table
cross-run pattern is exactly the one `scan12_4` already uses for pairs
B/D — no new provenance mechanism is introduced.

**Stage B — holdout scoring (frozen policy, new standalone scorer).**
Because `harness/action_boundary_score.py` hard-codes its `CELLS`/`SEEDS`
to pairs B/D and `pr12_3`/`pr12_4`, it must stay **byte-frozen** (a 12.6
artifact). Scoring the holdout is a *new* standalone read-only scorer
`harness/action_boundary_holdout_score.py` (stdlib + subprocess-git only)
that (i) reads the Stage-A cache + the held-out run-stem CSVs, (ii)
contains the §5 policy family **copied verbatim** from
`action_boundary_score.py` — `RowObs`, `CellCtx`, `_f1a_condition`,
`pol_f1b`, and the §11–§15 constants — with a registered check that the
copied block's sha256 matches the frozen source, and (iii) emits only
into `results/issue_failure_mode_blindness/pr12_7/`.

**Forbidden shortcuts (each is a §17 kill):** editing any `.jsonl`
packet by hand; generating packets from anything other than the pinned
committed governed inputs through the unmodified emitter; running the FAM
engine or minting any new seed; importing FAM-core; modifying the frozen
emitter or the 12.6 scorer; scoring any pair-B/D or s0 unit as if it were
a holdout.

## 5. Frozen W2:F1b policy definition (copied verbatim from PR-12.6; no change permitted)

The policy under test is **exactly** the code merged at `9a7537e`. It is
reproduced here so the holdout scorer can be checked byte-for-byte
against it. **No line may change**; any deviation is kill §17.3.

```python
PIN = "0afcb2bc4d94112fd2f2cb9a47525c6d2595c2dd"  # main @ PR-12.5 merge (§3)
PRECISION_FLOOR = 0.75          # G-A1 / G-C2
COVERAGE_FLOOR = 0.25           # G-A2, aggregate one-shot test coverage
CONTRA_WRONG_CEILING = 0.05     # G-C1, per contra test unit
PER_UNIT_WRONG_CEILING = 0.10   # §14, every test unit
GLOBAL_WRONG_CEILING = 0.05     # §13, per candidate over its test units
WITNESS_BASIS = "witness co-resident (fork_witness)"

class RowObs:
    __slots__ = ("shape", "width", "deployed_class", "led_slot",
                 "alt_classes", "all_witness", "presented")
    def __init__(self, shape, tie):
        cands = tie["candidates"]
        deployed = [c for c in cands if c.get("basis") == "deployed vote"]
        self.shape = shape
        self.deployed_class = deployed[0]["decode_class"] if deployed else None
        self.led_slot = deployed[0].get("slot") if deployed else None
        self.alt_classes = [c["decode_class"] for c in cands
                            if c.get("basis") != "deployed vote"]
        self.all_witness = all(c.get("basis") == WITNESS_BASIS
                               for c in cands
                               if c.get("basis") != "deployed vote")
        self.presented = sorted({c["decode_class"] for c in cands})
        self.width = len(self.presented)

class CellCtx:
    __slots__ = ("n_contradiction_pairs", "n_ambiguous_pairs",
                 "never_resolving_slots", "hazard_tier")
    def __init__(self, audit_lines):
        self.n_contradiction_pairs = 0
        self.n_ambiguous_pairs = 0
        self.never_resolving_slots = set()
        self.hazard_tier = None
        for rec in audit_lines:
            rt = rec.get("record_type")
            if rt == "contradiction_pair_review":
                self.n_contradiction_pairs += 1
            elif rt == "ambiguous_pair_review":
                self.n_ambiguous_pairs += 1
                if rec.get("never_resolving"):
                    self.never_resolving_slots.add(rec["pair"]["incumbent_slot"])
                    self.never_resolving_slots.add(rec["pair"]["owner_slot"])
            elif self.hazard_tier is None and "hazard_tier" in rec:
                self.hazard_tier = rec["hazard_tier"].get("tier")

def _f1a_condition(row, ctx, shapes):
    return (row.shape in shapes and row.width == 2 and row.all_witness
            and row.led_slot in ctx.never_resolving_slots
            and len(row.alt_classes) == 1)

def pol_f1b(row, ctx):
    if _f1a_condition(row, ctx, shapes=("W2",)) \
            and ctx.n_contradiction_pairs <= ctx.n_ambiguous_pairs:
        return {row.alt_classes[0]: 1.0}
    return None
```

The single candidate under test is **W2:F1b**. F1a/F1c and the B-*
comparators may be reported for context (as in 12.6) but only W2:F1b can
receive a GO. No F2/new family is registered (§17.3).

## 6. Permitted structural features (read-time observables; unchanged from PR-12.6 §10)

The policy may condition **only** on read-time observables present in the
holdout cell's own committed/generated packets:

* Row-local (`RowObs`): shape (W1/W2 — a property of the emitting
  governance layer), presented width, candidate decode classes, deployed
  class, source basis (witness co-resident vs deployed vote), sole-alt
  structure.
* Cell-context (`CellCtx`, label-free, from the same cell's audit
  packet): `contradiction_pair_review` / `ambiguous_pair_review` counts,
  `never_resolving` slot flags, hazard-tier field.

These are byte-identical to the features 12.6 audited as truth-free, and
§3 confirms every one is reconstructable on the pairs-C/E holdout from
the generated packets without any label.

## 7. Forbidden scoping inputs (unchanged from PR-12.6 §10; each use is kill §17.2)

Cell names, pair identifiers, arm identifiers, file paths, seed
identifiers, scan-report JSONs, `pr12_5/` or `pr12_6/` outputs, the PR-4
geometry/governance rows, registry/true labels, `vote_pred`/`deployed`
correctness, any post-hoc correctness signal, the `pr12_5`/`pr12_6`
correctness-coded row tables, and any quantity derived from them. Pair
identity in particular is **not** a policy input — a held-out pair is
just another cell; if the boundary needs to know it is on pair C, it has
already failed the disposition-scoping requirement.

## 8. Acting versus deferral semantics (unchanged from PR-12.6 §6)

Per in-scope holdout row the policy emits exactly one of **ACT(c)** — the
reader would rely on decode class `c`, which must be a member of that
row's presented set (acting outside it is structurally impossible and
coded as a kill if observed) — or **DEFER** — the row keeps its committed
dual-present/escalation posture. Acting is an offline label applied at
scoring time; no packet, disposition, or downstream system is touched.
DEFER is never penalized as wrong; its cost is counted only in §11.

## 9. Wrong-action definition (unchanged from PR-12.6 §7)

A wrong action on a row is `ACT(c)` with `c ≠ true_label` (registry truth
joined at scoring time from the committed held-out run-stem CSV, exactly
as in PR-12.5/12.6 — truth enters only after every decision is recorded).
For the deterministic W2:F1b the row contributes its full wrong mass when
it acts. **Wrong-action mass of a unit** = summed wrong-action
expectation over that unit's in-scope rows / the unit's in-scope row
count (deferred rows remain in the denominator).

## 10. Correct-action definition (unchanged from PR-12.6 §8)

A correct action is `ACT(c)` with `c = true_label`. **Acting precision**
of a unit = correct-action mass / (correct + wrong action mass) —
accuracy among acted rows only; undefined where the policy acts on zero
rows (such a unit passes no precision gate and contributes zero
coverage, so all-defer can never back into a GO).

## 11. Deferral accounting and coverage floor (unchanged from PR-12.6 §9)

Per unit: `coverage` = acted-row fraction; `deferral_rate` = 1 −
coverage; deferred-row mean presented width; the counterfactual columns
for the B-* comparators. **Coverage floor for GO (non-triviality):**
aggregate coverage over the one-shot **holdout** units ≥ **0.25**.
`B-defer-all` passes every safety gate and can never GO — the null is
safe and useless, and the floor is what a GO must clear above it.

## 12. Required one-shot utility gates (hard, per one-shot holdout unit)

* **G-A1 (precision floor):** acting precision ≥ **0.75** on every
  one-shot holdout unit where W2:F1b acts ≥ 1 row.
* **G-A2 (coverage floor, aggregate):** §11 floor — one-shot holdout
  coverage ≥ 0.25.
* **G-A3 (dominance carried):** on acted rows, expected correct rate
  strictly above what the certified-only `P-assert` reader scores on
  those same rows.

## 13. Required contra safety gates (hard, per contra holdout unit)

* **G-C1 (contra wrong-action ceiling):** wrong-action mass ≤ **0.05**
  per contra holdout unit; small units are not exempt.
* **G-C2 (contra precision when acting):** if W2:F1b acts on ≥ 1 contra
  row, acting precision ≥ **0.75** there too.
* **G-C3 (no-collapse):** the verdict must include every contra holdout
  unit; any exclusion, reweighting, or averaging of contra into one-shot
  aggregates is kill §17.8.

## 14. Global wrong-action ceiling (hard, per candidate)

Total wrong-action mass across **all** holdout units (one-shot + contra,
both held-out pairs, both seeds) / total in-scope holdout rows ≤
**0.05**.

## 15. Per-cell wrong-action ceiling (hard, every holdout unit)

Wrong-action mass ≤ **0.10** per holdout unit (one-shot units included;
at the G-A1 floor this binds coverage-precision combinations, e.g. full
coverage requires precision ≥ 0.90).

## 16. Holdout validity gates (hard — the crux of PR-12.7)

The utility/safety verdict is meaningful only if the holdout is valid.
These gates are evaluated **first**; any failure short-circuits to a
validity verdict (§20), never to a GO.

* **G-H1 (unexposedness):** every holdout cell's `(pair, arm, seed)` and
  its governed run-stem must be **absent** from every 12.1–12.6 scan
  config and every `pr12_*` aggregate result. Verified automatically by
  re-running the §3 exposure grep at scoring time and recording the
  (empty) match set. Any hit is kill §17.1.
* **G-H2 (provenance integrity):** the holdout packets must regenerate
  **byte-identically** from the pinned committed governed inputs through
  the **unmodified** emitter (emitter sha256 recorded and checked; a
  double emit is byte-identical). No hand-edited packet, no engine run,
  no new seed. Failure is kill §17.
* **G-H3 (no-tuning attestation):** the §5 policy block embedded in the
  holdout scorer is sha256-identical to the frozen
  `action_boundary_score.py` source; zero code path fits, selects, or
  thresholds against any holdout row; the pin is `0afcb2b`. Failure is
  kill §17.3.
* **G-H4 (sufficiency / power floor):** the holdout must contain at least
  a pre-registered minimum of in-scope W2 rows for the verdict to be
  *powered* — **≥ 30 in-scope W2 one-shot rows aggregated** and **≥ 1
  contra holdout unit with ≥ 5 in-scope W2 rows per held-out pair**.
  Below either threshold the run returns **`holdout-insufficient`** (§20)
  — an honest non-result that registers input-generation/provenance as
  the next step, **not** a pass and **not** a repackaging of a weak
  holdout as strong. (The floor is fixed in advance so a thin panel
  cannot be spun as confirmation.)
* **G-H5 (feature-reconstructability):** every in-scope holdout row's W2
  witness features and F1b quiet-cell features must be computable from
  packet fields alone (no label); any row requiring a forbidden field to
  classify is kill §17.2.

## 17. Leak / contamination kill conditions (any → `holdout-validity-blocked`)

1. Input drift from the `0afcb2b` pin; a missing committed input; a
   holdout packet not regenerable byte-identically from pinned inputs
   through the unmodified emitter (G-H2).
2. Label leak: any policy path reading truth labels, run-stem CSVs,
   `pr12_5/` or `pr12_6/` files, PR-4 governance rows, scan JSONs,
   cell/pair/arm/file/seed identifiers, or any §6/§7-forbidden input
   (G-H5).
3. Policy/threshold motion: any change to the §5 block, any threshold
   edit, any new/removed/modified policy family, any F2, any fit against
   a holdout row (G-H3).
4. Split/exposure violation: any holdout cell that appears in a 12.1–12.6
   config or aggregate (G-H1); any use of s0 or of pairs A/B/D as a
   "holdout"; any s1/s2 pair-B/D packet scored as held-out.
5. An ACT outside the row's presented set; a row join miss.
6. Writes outside `pr12_7/` and the declared `pr12_7_holdout_cache/`;
   `git status` dirty on `pr12/`–`pr12_6/`, `pr10/`, or any harness file
   other than the newly-added holdout scorer + the additive
   `scan12_7_holdout` config block, before or after.
7. Nondeterminism: internal double pass or external re-run differing in
   any byte of `pr12_7/` or the holdout cache.
8. Contra collapse: any contra holdout unit excluded from the verdict;
   any GO text claiming one-shot performance without the contra gates.
9. Any output language claiming deployment readiness, live acting,
   prompting use, promotion, ingestion, autonomous use, or a reader
   contract.

## 18. Byte-reproducibility requirements

The holdout cache regenerates byte-identically on a second emit (sha256
over the cache tree; emitter unmodified). The holdout scorer's internal
double pass is identical, and an external second invocation reproduces
every `pr12_7/` file byte-identically (sha256 over the tree); all three
recorded in the scan output. Committed `pr12/`–`pr12_6/`, `pr10/`, and
the frozen harness files are protected by hash-pinning + the §17.6
clean-tree checks (the PR-12.5/12.6 mechanism, reused).

## 19. Artifact paths (none created by this memo)

* This memo:
  `results/issue_failure_mode_blindness/PR12_7_ACTION_BOUNDARY_HOLDOUT.md`
  (results appended as §22+, append-only; §§1–21 frozen).
* Generation (separately authorized): additive `scan12_7_holdout` block
  in `harness/harness_policy.json`; holdout cache
  `results/issue_failure_mode_blindness/pr12_7_holdout_cache/`.
* Scoring (separately authorized): standalone read-only scorer
  `harness/action_boundary_holdout_score.py` (stdlib + subprocess-git;
  `harness_boundary_sim.py`, `reader_utility_score.py`, and
  `action_boundary_score.py` stay byte-frozen).
* Output: `results/issue_failure_mode_blindness/pr12_7/holdout_scan.json`
  (validity gates, per-unit per-policy §12–§15 tables, exposure-grep
  match set, provenance/reproducibility hashes, verdict) plus per-row
  decision tables `pr12_7/rows_<candidate>_<cell>_<policy>.csv` — every
  results-section number recomputable from `pr12_7/` + the cache alone.

## 20. Pass/fail verdict names (exactly one)

* `holdout-validity-GO(W2:F1b)` — the holdout is valid (G-H1–G-H5 pass,
  powered) **and** W2:F1b clears G-A1–G-A3 on all one-shot holdout units,
  G-C1–G-C3 on all contra holdout units, and the §14/§15 ceilings, with
  §17 clean. Establishes offline holdout evidence that the 12.6 boundary
  generalizes to the held-out pairs — **and nothing beyond that**.
* `holdout-validity-negative` — holdout valid and powered, but W2:F1b
  fails a utility or safety gate on it: the 12.6 evidence does **not**
  generalize. A real, informative result.
* `holdout-insufficient` — G-H4 not met: the generated panel lacks the
  in-scope W2 one-shot/contra rows to decide. Registers a larger
  input-generation/provenance step as the next action; asserts nothing
  about generalization.
* `holdout-validity-blocked` — a §17 provenance/instrumentation
  contradiction (leak, drift, tuning, or exposure).

## 21. Downstream-use boundary

Offline expected-value evidence about the generalization of one
pre-registered act-versus-defer policy over committed and
deterministically-regenerable packets — nothing else. **A
`holdout-validity-GO` authorizes offline generalization evidence and
nothing beyond it:** no deployment, no FAM-core integration, no prompting
use, no promotion to any policy version, no memory ingestion or
write-back, no autonomous downstream use, no LLM/agent reader
certification, no acting authorization in any live system, and no
reader-contract change — **PR-10's merge-abstain remains the only
certified reader contract**, and the operational posture on
witness-window rows remains **deferral** unless and until a separate
pre-registration proposes otherwise with this PR's evidence as one input.
A GO claims nothing about contra traffic beyond the safety gates it
passed, and nothing about any cell, pair, seed, manifold, or mechanism
outside the registered holdout units. A `negative` or `insufficient`
verdict likewise changes no contract and no posture. PR-12.1–12.6
verdicts stand unchanged.

## 22. Results (reserved; append-only after an authorized run)

Intentionally empty at pre-registration. §§1–21 above are the frozen
snapshot and are never rewritten. Implementation of Stage A (generation)
and Stage B (scoring) is **not** authorized by this memo and requires a
separate explicit approval.

## 23. Erratum E1 — Stage A mechanism correction (append-only; §§1–22 frozen)

*Design-only. This section is appended under the memo's own append-only
convention (§19/§22, the PR-9.2 / PR-12.2-E1 / PR-12.3-E1 precedent). It
does not rewrite §§1–22; those remain the historical registration
snapshot of what was originally proposed. This erratum **supersedes the
Stage A generation mechanism registered in §4** and **refines the
classification note in §3**, and registers nothing executable: no
generator is implemented, no packet is generated, no scoring runs, no
threshold/policy/verdict vocabulary changes. `harness_boundary_sim.py`,
`reader_utility_score.py`, `action_boundary_score.py`, all FAM-core
files, and every PR-12.1–12.6 artifact/verdict/gate/output/memo section
remain byte-frozen. PR-10 merge-abstain remains the only certified reader
contract; operational posture remains deferral.*

### 23.1 What was found at Stage A execution (why this erratum exists)

Stage A was **authorized and attempted on 2026-07-08 and halted clean
before any mutation** — no packets generated, no scoring, no config edit,
no branch/commit, `harness_boundary_sim.py` sha256 unchanged
(`2539686a205fa03ba88fb4e222043720dc6acf97460f896a31bd58d1a11d32e5`),
no `pr12_7/` or `pr12_7_holdout_cache/` created. The registered §4 Stage A
mechanism — *"add a `scan12_7_holdout` block to `harness_policy.json` and
run the byte-frozen emitter, no code edit"* — is **not executable as
written**. Read-only inspection established that the witness-window
(W1/W2) **emission** path in `harness_boundary_sim.py` is dispatched by
**hardcoded literal scan-block keys**: `run_scan`/`run_scan12_2`/
`run_scan12_3`/`run_scan12_4` each bind `scan = policy["scan12_N"]`, and
the emitting CLI paths in `main()` (`--shape`, `--shape12-3`,
`--shape12-4`) each hardcode their `policy["scan12_N"]`. There is **no
`--scan12-7`/`--shape12-7` flag, no `run_scan12_7`, and no generic
"run an arbitrary named block" CLI path** (the single indirection,
`scan = policy[scan_key]` at line 1819, is an internal
`scan_tree_bytecheck` parameter with hardcoded call-sites, not
CLI-reachable). The only generic path — the default `policy["cells"]`
loop used by `--check` — covers exactly the two base cells
`{pairD_stale-soft_s0, clean_pairA_s0}` emitted to `pr12/`, **not** pairs
C/E, **not** the W1/W2 shapes, **not** the cache directory. Emitting C/E
W2 holdout packets under the §4 mechanism would therefore require adding a
flag + a `main()` dispatch branch reading `policy["scan12_7_holdout"]` —
an **edit to the frozen emitter's logic** (changing its sha256), which
§4 forbade ("no code edit; sha256 unchanged") and which fired the
registered stop condition ("generation requires changing registered
config or emitter logic").

### 23.2 Refinement to the §3 classification (correction of record)

§3 concluded classification **C** on the reasoning that "the emitter is
fully data-driven from `harness_policy.json` (`run_cell` iterates
`policy["cells"]`), so addressing pairs C/E is a config-only cell
addition." That reasoning is **correct for the `--check` byte-gate path
only** and **does not hold for the witness-window shape-emission path
that Stage A actually needs**, which is per-block hardcoded. The
classification is refined (not overturned): **strong pair-axis holdout
candidates on pairs C/E still exist** (unexposed to any 12.1–12.6
aggregate; both harm classes committed in `pr10/governed`; committed
hazard-governance in the PR-4 geometry table; label-free features
reconstructable), **but Stage A requires a registered generation
capability** because W1/W2 emission is not config-only generic. In the
memo's own A/B/C/D vocabulary this is still **C with a named capability
prerequisite**, not **D** — no *new* engine or FAM-core work is needed;
the required primitives already exist and are frozen; what is missing is a
registered, byte-verified *invocation* of them. This erratum supplies that
registration.

### 23.3 Corrected Stage A mechanism — separately-authorized standalone generator

The Stage A generation mechanism registered in §4 is superseded by a
**standalone generation driver**, mirroring the Stage B standalone-scorer
pattern (a new file that reuses frozen primitives without editing them):

* **Artifact (future, separately authorized; not created by this
  erratum):** `harness/action_boundary_holdout_generate.py` — stdlib +
  `subprocess`-git only; it **imports** the frozen emitter
  (`from harness_boundary_sim import run_cell`, and any read-only helpers
  it needs) and **must not edit `harness_boundary_sim.py`** (whose sha256
  must be unchanged across the run, pinned to
  `2539686a205fa03ba88fb4e222043720dc6acf97460f896a31bd58d1a11d32e5`).
* **Additive config:** a `scan12_7_holdout` block in
  `harness_policy.json` is retained as the *data manifest* for the C/E
  cells (run-stems + hazard-governance, per §4's table), but it is
  **read by the standalone generator**, not by any emitter CLI flag — so
  no emitter dispatch branch is added.
* **Faithful-invocation contract:** the generator must construct W2
  packets by invoking the **same primitive the registered W2 runner
  uses** — `run_cell(repo, name, cfg, policy, allow_stale, out_root=…/
  <shape>, shape="W2", policy_version=<the scan12_7_holdout policy
  version>, emit_review_queue=True, emit_ambiguous_queue=True)` — with
  the identical keyword signature and shape semantics as the PR-12.4 W2
  emission path (`main()` `--shape12-4` / `run_scan12_4`). It reuses,
  never reimplements, the packet-construction logic. (F1b is W2-only;
  the generator emits the same shape set the registered runner emits for
  parity, and the holdout scoring target remains W2.)
* **Output:** witness-window holdout packets only, under
  `results/issue_failure_mode_blindness/pr12_7_holdout_cache/` (the §19
  cache path). No `pr12_7/` scoring directory is created by Stage A.

### 23.4 Mandatory pre-emission byte-equivalence self-check (the crux of this erratum)

Before emitting any C/E holdout packet, the generator **must** prove it
reproduces the registered runner's output byte-for-byte on an
**already-committed known W2 cell**:

* **Parity anchor:** a committed W2 **emitter-output** cell from
  **PR-12.4** — canonically
  `results/issue_failure_mode_blindness/pr12_4/W2/pairD_oneshot_s1/`
  (`memory_packet.jsonl`, `audit_packet.jsonl`, `decision_table.csv` all
  committed). *Precision recorded:* PR-12.6 produced **no** emitter
  packets (it is a read-only scoring stage over 12.3/12.4 packets), so
  the parity anchor is a PR-12.4 (or PR-12.3) W2 cell, not a 12.6 output;
  where §4/point-5 says "PR-12.4/12.6", read it as "the committed W2
  emitter output, which lives in PR-12.4/12.3."
* **Procedure:** regenerate the anchor cell into a temp directory via the
  imported `run_cell` at `shape="W2"` with the anchor cell's own
  committed config, then compare against the committed bytes — exactly
  the `scan_tree_bytecheck`/`base_bytecheck` discipline already in the
  frozen emitter (reused, not reimplemented).
* **Proof required:** for each of the three artifact files —
  **byte-identical** content (full compare), **identical sha256**,
  **identical schema** (field/column set), and **identical record
  ordering** — recorded in a provenance JSON. Byte-identity is the
  contract; schema/ordering are asserted explicitly and are implied by
  it.
* **Gate:** only **after** the self-check passes on the anchor cell may
  the generator emit the C/E holdout packets. The anchor comparison and
  the C/E emission use the *same* `run_cell` invocation, so a passing
  anchor certifies the C/E packets are constructed by the registered
  primitive.

### 23.5 Registered kill conditions for the corrected Stage A (any → generation aborts, cache discarded)

1. **Byte-equivalence failure** — the anchor-cell regeneration differs
   from the committed bytes in any file, sha256, schema field, or record
   order (§23.4). Generation must not proceed to C/E; the run is a kill.
2. **Emitter-edit requirement** — if faithful generation is found to
   require editing `harness_boundary_sim.py` (its sha256 would change),
   that is a kill **unless a separate pre-registration explicitly
   authorizes an emitter extension** with its own byte-gate. This erratum
   does **not** authorize an emitter edit.
3. **Config/scope drift** — any `scan12_7_holdout` cell pointing at
   anything other than the §4-registered committed C/E governed
   run-stems; any use of s0 or pairs A/B/D as "holdout"; any exposure hit
   under the §16 G-H1 grep.
4. **Label / outcome leak into the cache (§23.6).**
5. **Writes outside `pr12_7_holdout_cache/`**; any creation of `pr12_7/`
   scoring artifacts by Stage A; `git` dirtiness on `pr12/`–`pr12_6/`,
   `pr10/`, or any frozen harness file.
6. **Nondeterminism** — the holdout cache not byte-identical across two
   generation runs.

### 23.6 Cache purity requirement (no scoring outcomes in Stage A output)

Generated C/E holdout packets **must not** contain — and the generator
must scan its own cache and abort if it finds — any `true_label`,
registry label, `vote_pred`/post-hoc correctness field, action-boundary
verdict token (`action-boundary-*`, `holdout-validity-*`,
`holdout-insufficient`), F1a/F1b/F1c decision, coverage/precision/
wrong-mass figure, or any reference to `pr12_5/`/`pr12_6/` scoring
outputs. The packets are the emitter's `memory_packet.jsonl` /
`audit_packet.jsonl` / `decision_table.csv` only — the same
label-free artifacts whose truth-freedom PR-12.6's provenance audit
already established. Stage A produces read-time observables for a future
Stage B; it must carry **no** scoring signal.

### 23.7 Scope, stage separation, and boundary (unchanged, restated)

Stage A (generation) remains **separate from and prior to** Stage B
(scoring), which is **still unauthorized** and unchanged by this erratum.
This erratum authorizes **nothing to execute**: implementing
`action_boundary_holdout_generate.py`, running it, generating any packet,
and running any scoring each require **separate explicit approval**. FAM-
core is untouched; no emitter edit is authorized; the W2:F1b policy,
gates, thresholds, holdout-validity gates (G-H1–G-H5), and verdict
vocabulary (§20) are unchanged; the pair-axis (not fresh-seed) holdout
scope (§3) is unchanged. No deployment, live acting, prompting use,
promotion, memory ingestion, autonomous downstream use, or reader-contract
change is created or implied — **PR-10 merge-abstain remains the only
certified reader contract**, and the operational posture remains
deferral. PR-12.1–12.6 verdicts stand unchanged.
