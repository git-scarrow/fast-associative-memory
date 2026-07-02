# PR-8 — §9 quarantine-replacement program · **§9A shadow-quarantine / audit-only gate** (pre-registration, design-only)

**Status: design-only. DRAFT, uncommitted, unmerged.** This memo pre-registers
**only §9A**. The later stages §9B/§9C/§9D are *named and staged* here so the
program is legible, but they are **explicitly NOT pre-registered** — each has an
entry condition that must be met, and its own margins, before it is registered in
a separate memo. Nothing here promotes read-time quarantine, claims recovery, or
moves any frozen margin. Engine byte-frozen; deployed retrieval unchanged; main
untouched; the prepared per-seed scorer (`feat/pr7-scorer-perseed-guard`) stays
unmerged and is not a dependency of §9A.

**Revision note.** This supersedes the earlier broad §9 draft (which registered a
single gate over four candidates A/B/C/D at once). Per review, the program is
**staged**: §9A certifies *only* the one thing §8 actually validated — forensic /
audit value with complete provenance — in isolation, at zero read-time risk. The
read-time-benefit hypothesis (B), the recovery-coherence question (C), and the
per-seed scorer discipline (D) are deferred behind explicit gates.

---

## 1. What §8 established (settled evidence — inputs, not questions)

Quarantine, write-path `--govern quarantine`, frozen #87 readout, engine
byte-frozen, seeds {0,1,2}, the §8 panel:

* **Useful aggregate signal** (read-time drain on every hazard shape; inert on
  clean; `both_shapes_ok = True`).
* **Complete reversible provenance** — G3 reconstruction fidelity HOLDS on every
  arm: the diverted set is a lossless, independently reconstructable record.
* **G3 FAILURE** — recoverability is *provenance-only, not harm-free*: the diverted
  writes **are** the stale supersessions, so reinstating them re-introduces the
  harm on all four geometries. `g3_verdict = provenance_recoverable_not_harm_free`.
* **G4 FAILURE** — per-seed collateral breach `direct_harm`/pairD/s2 collateral
  Δ = −4 (< frozen −3), masked by the aggregate guard (+26).
* **Therefore** quarantine is **not promotable as automatic governance** and is
  retained at `needs_review` — a *safe one-way diversion with a complete audit
  trail*.

The **one thing §8 validated outright** is the *audit trail* (complete, lossless,
reversible-as-a-record). §9A certifies exactly and only that.

---

## 2. Why staged, and the program shape

The two §8 blockers are **orthogonal, with different roots** (full recap §3):
G3 is a *semantics* defect (a read-time suppressor and a reversible store fused on
the same unit); G4 is a *granularity* defect (a binary divert/keep on a dual-use
write, masked by aggregate scoring). Neither is resolved by a single combined
gate. Only the **forensic value** is established without caveat, so it is the only
thing that can be certified now — alone, and provably inert.

| stage | certifies | entry condition | pre-registered now? |
|---|---|---|---|
| **§9A** shadow / audit-only | the forensic/audit value is first-class **and provably inert** (zero retrieval effect) | none — this memo | **YES** |
| **§9B** corroborated read-time | a *parameter-free one-way suppressor* that drains harm **and** clears the −3 per-seed bound | §9A passes **AND** a parameter-free **binary** second key is found in committed write-time observables | **no — blocked** |
| **§9C** recovery / re-injection coherence | whether a *reversible-store* claim is even coherent in this engine | the line still wants to test a reversible store | **no — separate semantic question** |
| **§9D** scorer discipline / per-seed guard | infrastructure: per-seed *reporting* | needed as reporting infra | **no — informs §9A reporting, is not its product claim** |

§9A's product claim is narrow on purpose: **audit-only inertness**, nothing about
read-time governance, recovery, or promotion of quarantine as a policy.

---

## 3. Mechanism recap (why §9A is the right first stage)

| §8 failure | nature | root |
|---|---|---|
| **G3** provenance-only recovery | semantics | quarantine merges a *read-time suppressor* (its benefit) and a *reversible store* (its claimed edge over refuse) on the **same unit**; reinstatement is byte-identical replay with no re-validation → *audit-vs-read-time conflation* + *lack of re-injection semantics* |
| **G4** per-seed collateral | granularity | a binary divert/keep on a **dual-use** write (harmful to its stale target, useful to a benign neighbour) can't separate the uses; aggregate guard masks the cost → *routing ambiguity* + *scorer blindness* |

§9A removes the conflation by construction: it takes the **detector / audit** half
of quarantine and runs it with **zero suppression**. That makes G3 and G4
non-applicable to §9A (nothing is removed, nothing is reinstated, retrieval is
untouched) — and isolates the audit value as a thing that can be certified on its
own terms.

---

## 4. §9A — shadow-quarantine / audit-only gate (the pre-registration)

### 4.1 Purpose
**Not** to promote read-time quarantine. To certify whether the forensic/audit
value §8 validated can be made **first-class and provably inert**: a governance
mode that *flags and records* every quarantine-eligible event while changing no
retrieval behaviour at all, producing an audit trail a human or later policy can
review — and that later §9B can use as a denominator.

### 4.2 Mechanism
`--govern shadow` (the existing `--govern` seam in
`benchmarks/failure_mode_probe.py`; engine untouched): detect+ledger
quarantine-eligible events **using the same committed eligibility rule as §8
quarantine**, but **divert nothing** (byte-identical to `none`). **Expected
coverage, derived from §8, is 192/seed on stale arms and 0 on clean.** No new
record-granularity observable — the per-event facts already exist in committed
`fork_events.csv` (`pre_sim`, `payload_cos_incumbent`, `effective_vigilance`,
incumbent maturity/recency/lineage, `outcome`, `owner_slot`); §9A emits the
aggregate ledger in the committed quarantine-ledger shape with
`disposition: flagged_not_diverted`, and **certifies** `fork_events.csv` labels
the eligible class completely and deterministically. Denominator for §9B = that
certified existing per-event provenance, not a new ledger.

### 4.3 Scope (frozen)
1. detect and ledger quarantine-eligible events;
2. divert nothing;
3. change no retrieval behaviour;
4. produce a complete audit trail usable as a **denominator for later §9B**;
5. measure the **residual harm** that shadow quarantine would have flagged but not
   suppressed (§4.6).

### 4.4 Non-goals (frozen)
* no read-time suppression;
* no recovery claim;
* no promotion of quarantine as governance;
* no scorer-threshold changes;
* no pairD/s2 special-casing;
* no seed exclusion;
* no margin movement.

### 4.5 Required cells and per-seed audit
The **same §8 panel**, no new cell, no geometry screen, no class-pair search:
`direct_harm` (pairD); `collateral_harm` (pairB, pairE); `clean_control` (pairA,
pairC); `merge_path_stale` (pairA, pairB, pairD, pairE); `one_shot_ambiguity`
observe-only. Seeds {0,1,2}, frozen #87 readout. A **per-seed audit report** is
produced for every (cell, pair, seed). **No per-seed retrieval regression is
possible** — shadow does not affect retrieval — so the per-seed report is an audit
artifact, not a pass/fail surface (contrast §9B, where it would be).

### 4.6 Residual-harm measurement (the one new figure §9A reports)
Because shadow is inert, its read-time readout **is** the baseline. §9A therefore
reports, without acting:
* baseline read-time broken/stale per (cell, seed) (= shadow's own readout, by the
  inertness requirement); and
* the **counterfactual ceiling** = the committed §8 `baseline − quarantine` drain
  per (cell, seed) (e.g. pairD broken +111 / stale +300), **bound to shadow's
  flagged ledger** by verifying shadow's flagged set == the §8 quarantine diverted
  set (192/seed). This is the harm a later read-time candidate (§9B) could in
  principle recover, and the cost §9B must justify against the −3 bound.

§9A does **not** re-derive the counterfactual by suppressing — it inherits it from
the committed §8 arms and certifies the flag-set identity. A finer per-event
attribution (joining broken rows to originating write events) is **optional**; if
produced it must be deterministic and read only committed artifacts, and it is
**not** required for the §9A verdict.

### 4.7 Promotion condition → verdict `promoted-audit`
§9A is certified (`promoted-audit`) **iff all** hold:
1. **byte-inert retrieval** versus baseline — every readout artifact (per_probe,
   topk, governance, per_slot, fork_events) identical to `--govern none` on every
   cell and seed;
2. **complete ledger coverage** — shadow flags every event the old quarantine
   would have diverted (flagged count == committed quarantine diverted count:
   192/seed on each merge-path-stale arm, 0 on clean);
3. **deterministic replay** — re-run byte-identical (same seed → identical
   artifacts) and sha256-stable across darwin/gentoo;
4. **no clean-control change** — `clean_control` (pairA, pairC) byte-identical to
   baseline on every artifact and seed;
5. **per-seed audit reports produced** — but no per-seed retrieval regression is
   possible, because shadow mode does not affect retrieval.

`promoted-audit` certifies the **detector / forensic role only**. It explicitly
does **not** assert read-time remediation, recovery, or any read-time governance.

### 4.8 Failure conditions (record the negative; do not tune)
* any retrieval output changes under shadow mode;
* ledger incompleteness (flagged < diverted, or any eligible event unrecorded);
* nondeterministic event labeling;
* hidden dependency on future scorer state (the §9A verdict must be computable from
  committed artifacts + the shadow run alone — **no** dependency on the unmerged
  per-seed guard or any unbuilt scorer field);
* any wording — in artifact, memo, or verdict — that implies read-time quarantine
  has been promoted.

### 4.9 Certification harness (constraints)
§9A is scored by an analysis-only reader that imports no torch and reads only
committed artifacts plus the shadow run. It must **not** depend on
`feat/pr7-scorer-perseed-guard` or any future scorer state (that dependency is a
§4.8 failure). Determinism across darwin/gentoo is a precondition.

---

## 5. Deferred stages (named, NOT pre-registered here)

### §9B — corroborated read-time quarantine *(the next hypothesis, blocked)*
Divert a merge-suspect write only when a second, already-emitted, already-binary
write-time signal also fires (strict AND, parameter-free). Targets G4 by sparing
dual-use writes. **Blocked** — not registered, not run — **unless both**: (a) §9A
passes (a stable, complete, inert ledger exists as the denominator), **and** (b) a
**parameter-free binary** second key is found in the committed write-time
observables (`fork_events.csv`). If no such key exists, §9B is falsified before any
run and the line consolidates on §9A. §9B's margins (it must clear the inherited §8
aggregate margins **and** the frozen −3 per-seed bound on *every* seed) will be
pre-registered in a separate memo **only** once both entry conditions hold. §9B is
**not** part of §9A and shares none of its claims.

### §9C — recovery / re-injection coherence *(separate semantic question)*
Whether a quarantined write can ever be reinstated harm-free (e.g. recovery gated
on re-running the existing admission check at recovery time). This is a **distinct
semantic question**, pursued only if the line still wants a reversible-store claim.
Its most likely outcome — `recovery-incoherent` under the frozen static incumbent —
is a finding of §9C, **not** a property of an audit-only pass. **Do not fold
"recovery incoherent" into a §9A certification**; §9A makes no recovery claim in
either direction.

### §9D — scorer discipline / per-seed guard *(infrastructure)*
The prepared `feat/pr7-scorer-perseed-guard` per-seed collateral guard. It may
**inform §9A reporting** (the per-seed audit artifacts of §4.5) but is **not** the
§9A product claim and **not** a §9A gate condition — §9A must certify without
depending on it (§4.8). It becomes a *gate* only where a stage acts on retrieval
(§9B), at which point it is registered as that stage's scoring discipline.

---

## 6. Boundary & anti-tuning invariants (all stages)

Engine `associative_core.py` / `fast_associative_memory.py` sha256 unchanged;
deployed `forward()` / `learn_local` byte-identical; `--govern` opt-in and
unreached by deployed retrieval; geometry never a gate; both arms byte-stable
across darwin/gentoo; baseline `none` vote bit-identical to deployment; every
action parameter-free; the §8 panel, its margins, and the **frozen −3 per-seed
bound** are **not moved**; no seed exclusion; no pairD special-casing. Any breach
voids certification.

## 7. Explicit non-goals (hard boundaries)

All of PR7_DESIGN §12 and PR7_QUARANTINE_PROMOTION_GATE §8, plus: §9A performs **no
read-time suppression, no recovery claim, and no promotion of quarantine as
governance**; no §8 re-opening or re-scoring; no movement of the −3 bound; no
"make quarantine pass" by seed exclusion / margin relaxation / pairD special-
casing; **no implementation in this branch** — §9A's mechanism, gate, and
verdict semantics are specified here, not written; §9B/§9C/§9D are named, not
registered.

## 8. Instrumentation note — identity-key semantics (post-registration)

*Added with the diverted-key instrumentation PR (probe + §9A harness, no §9A
certification). Records a known limitation a future certification attempt must
heed; it does not move any registered gate, margin, or boundary.*

Persisted diverted-event keys close the observability gap but do not by
themselves prove logical event identity across divergent runs. The current
strongest committed common key is incumbent-state-based; because shadow commits
flagged writes and quarantine diverts them, incumbent state diverges after the
first diversion. Therefore real-run identity may be decidable as violated even
when detector eligibility is logically similar. A future certification attempt
must either accept this as a refutation or pre-register a write-event-intrinsic
identity key that is present in both shadow and quarantine artifacts without
breaking shadow byte-inertness.

Empirically (hermetic smoke, supersede_epoch 3 / 6 epochs): the incumbent join
key matches 8/24 events (the first supersession epoch, before divergence) and
differs on 16/24 (epochs 4–5), so the harness reports `identity-violated`. This
is the "or refute" half of *decide identity* — a valid finding, not a defect.
The fixture tests retain the controlled `identity-proven` case; the join key is
`QUARANTINE_DIVERTED_JOIN_KEY = (epoch, event_class, incumbent_slot,
incumbent_last_write_seq)`, the strongest key common to both committed artifacts
that does not add a `fork_events` column (which the byte-inertness invariant
forbids).
