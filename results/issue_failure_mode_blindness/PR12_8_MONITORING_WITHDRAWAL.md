# PR-12.8 §9 monitoring/withdrawal registration for `s1-witness-alt-batch` (design-only)

*Registration only. This document discharges the §9 deliverable of
`PR12_8_READER_CONTRACT_CANDIDACY.md` (main `fb2914f`): it fixes, in
advance of Stage E, what is watched whenever the contract candidate is
ever exercised, the tripwire values, and what withdrawal means. It
authorizes **nothing to run**: no exercise of the candidate, no
Stage E adjudication, no serving of `witness_alt` to any consumer, no
deployment, prompting use, promotion, memory ingestion, FAM-core
change, or reader-contract change. The candidate remains at rung 2
(defined, composition-proven, envelope-frozen) with **no operative
force**. **PR-10 merge-abstain remains the only certified reader
contract**; the operational posture on witness-window rows remains
**deferral**. Frozen at commit; per the memo §9 rule, no tripwire
here may ever be relaxed — future amendments may only bind tighter,
by append.*

---

## 1. Scope of application

This registration governs **every exercise** of the
`s1-witness-alt-batch` contract candidate (any version) on any packet
cell, in any context in which such exercise is ever separately
authorized — Stage E adjudication computations, future panel
extensions, D-strong reruns, and any post-certification use. Exercise
means: a conformant reader (contract §7) computing served-decision
records over a cell's packet pair. Nothing here authorizes an
exercise; absent separate authorization the candidate is not exercised
at all.

## 2. Watched quantities (per exercised cell)

**Label-free, computable at exercise time from the packet pair and the
served-decision records alone:**

* `guard_margin` = `n_contradiction_pairs − n_ambiguous_pairs`, and
  the open/closed state;
* `corridor_flag` — margin inside the **unobserved guard corridor
  (−4, +23)** (open interval; committed observations to date: open
  cells −4…−32, closed cells +23…+186);
* `n_in_scope_rows`, `n_witness_alt`, per-cell coverage;
* `n_fail_closed_events` — rows served `defer` by the contract §7
  fail-closed clause (schema anomaly, eligibility ambiguity);
* composition results — I2 overlap row count (must be 0), incumbent
  field deviations (must be 0), tier violations (must be 0);
* `envelope_membership` — whether the cell is in the frozen envelope
  (v0.2 at registration time; later versions by append), and if so,
  whether the exercise's `witness_alt` multiset (query_id + class)
  matches the envelope entry **exactly**.

**Truth-joinable (computed retrospectively, only where registry truth
exists and only after every serving decision is recorded):**

* per-cell wrong-mass rate and acting precision on acted rows;
* aggregate wrong-mass rate over the exercised cells of the engagement.

## 3. Tripwires (registered values; exactness or the 12.6 constants; tighten-only)

| id | condition | class | value |
|---|---|---|---|
| T1 | envelope-cell `witness_alt` multiset ≠ frozen envelope entry | exactness | any byte |
| T2 | any I2 overlap row, incumbent-field deviation, or `witness_alt` at a tier other than `harness-heuristic` | exactness | any instance |
| T3 | per-cell wrong-mass rate (truth-joined) | 12.6 §15 constant | > 0.10 |
| T4 | acting precision (truth-joined) where the candidate acted ≥ 1 row on the cell | 12.6 constant | < 0.75 |
| T5 | aggregate wrong-mass rate over exercised cells (truth-joined) | 12.6 §14 constant | > 0.05 |
| T6 | `corridor_flag` — exercised cell's guard margin inside (−4, +23) | evidence-gap fail-closed | any instance |
| T7 | `n_fail_closed_events` > 0 on a cell | conformance | any instance |

Notes of record: T1/T2 are exactness conditions in the PR-10 style —
no tolerance, no averaging. T3–T5 are exactly the registered 12.6
constants; they are not movable by this or any later monitoring
document. T6 binds **tighter** than the evidence requires, by design:
the guard's behavior inside the corridor has never been observed
(§14.4), so a cell landing there is treated as outside the candidate's
evidence base. T7 restates the contract's fail-closed clause as a
monitored, recorded event rather than a silent one.

## 4. Withdrawal semantics

* **Cell-level reversion (T6, T7, and any single-cell T3/T4):** the
  affected cell is served entirely at the committed
  dual-present/escalation posture — the candidate is **silent** on it
  (every row `defer`; incumbent `abstain`/`answer` rows unaffected).
  This is the registered §9 minimum and is automatic and immediate.
* **Candidate-wide suspension (any T1 or T2; any T5; T3/T4 on two or
  more cells of one engagement):** the candidate as a whole reverts to
  a recorded status `withdrawn-pending-review`; no `witness_alt` is
  emitted on **any** cell by any conformant reader until reinstatement.
  Exactness and composition breaches indict the mechanism, and
  multi-cell or aggregate ceiling breaches falsify the safety-evidence
  class, so suspension is global — tighter than the registered
  minimum, as permitted.
* **Event record (append-only, mandatory):** every tripwire event is
  recorded with cell identity, packet shas, tripwire id, measured
  value vs registered value, and the reader/contract versions, in an
  append-only event log kept beside the exercising context's
  artifacts. Withdrawal without a recorded event is itself a
  conformance violation.
* **Reinstatement:** only via a new pre-registration that examines the
  recorded events; neither the monitoring layer nor any reader may
  self-reinstate a withdrawn candidate. Withdrawal never edits
  packets, the envelope, prior verdicts, or any committed artifact —
  posture reverts; the record only grows.

## 5. Relationship to Stage E

With this registration committed, the §14.5 readiness ledger is
complete: Stages A–D discharged and the §9 prerequisite met. Stage E
adjudication becomes **authorizable** — it remains **unauthorized**
until separately approved, and when run it must carry verbatim the
§14.5 seed bound (any GO strictly named
`contract-candidate-GO-seedbounded(W2:F1b)`), the §14.3 traffic
bounds, and the §14.4 contra-power bound, with this document's
tripwires cited as the candidate's standing monitoring terms.

## 6. Boundary

Registration only. Nothing is exercised, adjudicated, certified,
served, deployed, promoted, or ingested; FAM-core is untouched;
Stage E remains separately unauthorized. **PR-10 merge-abstain remains
the only certified reader contract**; the operational posture on
witness-window rows remains **deferral**. PR-12.1–12.8 verdicts and
registrations stand unchanged.
