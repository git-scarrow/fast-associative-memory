# PR-9.1(a) — abstention-composite envelope desk check (documented stop + partial refutation)

**Question.** Can the composite read-time policy
`{abstain if the vote-leading slot is merge-suspect; abstain on witnessed-fork
tie margin < 0.10; otherwise observe-only}` — drawn from PR-4's certified-safe
set — be scored exactly from the two committed shadow-governance tables
(`pr3c/pr3c_governance_table.json`, `pr4/pr4_geometry_table.json`), to produce
pre-registration bounds for an acting-readout twin (PR-10)?

**Answer: no (stop condition), and the desk check already refutes two of the
draft PR-10 bounds.** Analysis-only; nothing was tuned; no artifact modified.

## 1. Why exact extraction fails (the stop condition)

The merge-suspect component fires only on soft arms (`n_merge_suspect_events`
= 192 on all 15 PR-4 soft cells and all 3 PR-3c soft runs; 0 elsewhere), and on
14 of those 15 cells the extraction is provably inexact, for two independent
reasons:

1. **Forced-exclusion abstentions are folded into `abstained`.** For
   mode-conditioned policies `abstained = |M| + |F|`, where F = probes whose
   entire surviving candidate set was excluded (`_vote` returns
   `(None, True)`); the `forced` flag is discarded by the frozen scorer
   (`benchmarks/analyze_fork_governance.py:469`) and recorded nowhere. |F| ≥ 1
   is provable in 7 soft cells from strict counter divergence between nested
   policies (e.g. pairD/s0 450 vs 451); in the other CP>0 cells the F sets are
   provably equal, not provably empty.
2. **The merge∩tie overlap is recorded nowhere.** The composite abstains on
   M ∪ T; both components fire with overlapping stale-wrong mass exactly on the
   hard cells (pairD/soft/s0: tie captures 23 stale-wrong vs the policy's 298 —
   capture bounds only to [0.795, 0.856], an interval, not a number).

The merge-abstain component's *construction* is sound: the frozen scorer
evaluates merge-suspect-led on the unmodified deployed candidate set
(`analyze_fork_governance.py:441-447`), so the abstain row set M is
policy-invariant. What is missing is recording, not semantics.

Only pairC/soft/s2 escapes both obstructions: composite capture **285/285 =
1.000, 0 false abstentions, fixed=broken=0** — the merge-abstain component's
one exactly-scorable soft cell, and it is clean.

## 2. What IS exact (96/114 cells) — and two draft bounds already refuted

Everywhere `n_merge_suspect_events` = 0, the composite ≡ abstain-tie, which is
exactly recorded, and zero-changed-answers is field-verified
(`fixed=broken=tie_flips=0`, `acted==abstained`, `observe-only == none` on
every scalar in all 114 cells). Against the draft PR-10 bounds:

| draft bound | outcome on exact cells |
|---|---|
| (i) 0 changed answers | PASS everywhere scorable |
| (iii) one-shot 100% wrong-abstain, 0 false | FAIL on C/D/E — coverage floor 0.600 (pairD/s0), false abstentions up to 44/run |
| (iv) 0 actions on clean arms | **REFUTED** — 24 tie abstentions/run on pairD clean s0/s1 and pairE clean s0 |
| (v) false abstain ≤ 5% of correct traffic | **REFUTED on all 30 contra+mixed cells** — 23.6–36.8% (worst pairD/contra/s0: 786/2136) |

The refutations are entirely the **unconditional abstain-tie** component: tie
topology without mode evidence cannot tell a live conflict from an honest tie
(PR-3c §3's warning, now quantified per pair), and its "tie == one-shot wrong
row" identity is pairA/pairB geometry luck that degrades with D/E compression
(echoing PR-4). The merge-abstain component has **zero recorded negatives**
anywhere in either table.

## 3. Consequences (design decisions, recorded before any PR-10 run)

1. **PR-10's acting-readout policy is merge-suspect abstention ALONE.**
   Unconditional tie-abstention is not deployable; one-shot/tie handling is
   deferred to the contradiction-adjudication stage (PR-11), where ties are
   held pending rather than answered or blanket-abstained.
2. **Smallest sufficient scorer addition** (pre-registered follow-up, own
   branch, not a retrofit): add the merge-abstain policy as a first-class
   member of `POLICIES` in `benchmarks/analyze_fork_governance.py` (no
   exclusions, no vote recomputation: `if top1 in st["merge"] → abstain, else
   the none answer`), record the discarded `forced` flag and per-trigger
   abstention counters, and re-emit both governance tables from the committed
   per-run artifacts. Deterministic offline re-analysis — no new benchmark
   arms, no engine change — but it changes the scorer sha, so it is registered
   as new analysis, and the re-emitted tables must reproduce every existing
   policy row byte-identically (regression guard) while adding the new rows.
3. **PR-10's bounds are pre-registered from the re-emitted table**, not from
   this memo's intervals.

Provenance: extraction attempted against the committed tables at main after
the PR-9.0 custody merges; the implementation branch
(`feat/pr9-abstention-envelope`) was left empty per stop-condition discipline.
