# PR-9.2 — §9A shadow certification with the write-event-intrinsic identity key

**Status: pre-registration (§1–§4) committed BEFORE any panel run; results
(§5–§8) appended after.** Shadow artifacts only: no runtime policy change, no
reader-contract change, no new `--read-govern` mode, no threshold movement, no
residual-fixing attempt. PR-10's `merge-abstain` remains the only certified
reader contract, untouched. This PR does **not** implement or register §9B
write-event authority; it certifies (or refutes) the identity foundation §9B's
entry condition requires (`PR8_QUARANTINE_REPLACEMENT_GATE.md` §8: "pre-register
a write-event-intrinsic identity key … or accept this refutation").

Verdict vocabulary (exactly one, §8): **identity-certified** — the §9A gate
passes end-to-end on the frozen panel with the intrinsic key;
**identity-blocked** — a provable contradiction (identity mismatch, key
collision, inertness break, non-additive rewrite, unexplained row-count
drift: all stop conditions); **identity-inconclusive** — every check that ran
passed but a required dimension could not be completed.

## 1. The old identity basis, and why it is state-contaminated

The §8-era join key is `QUARANTINE_DIVERTED_JOIN_KEY = (epoch, event_class,
incumbent_slot, incumbent_last_write_seq)` — the strongest key that existed on
both committed sides (a diverted write never reaches `write_fn`, so it has no
`record_seq`/`owner_slot` of its own; events were bridged by the incumbent
they supersede). Its last component is **incumbent state, not event
identity**: shadow commits every flagged write (bumping the incumbent's write
seq) while quarantine diverts it (freezing the incumbent), so the two arms'
values diverge after the first diversion epoch. The committed identity smoke
(`pr8/identity_smoke/`) measures exactly this: 8/24 events join (the first
supersession epoch only); all 16 failures differ **only** in
`incumbent_last_write_seq`; `(epoch, event_class, incumbent_slot)` matches
24/24. The harness verdict `identity-violated` was the honest "or refute" half
of the gate memo §8 — a property of the key, not of the detector.

## 2. The write-event-intrinsic key (the pre-registration)

```
PR92_INTRINSIC_JOIN_KEY    = (epoch, event_class, batch_index)   # unique
PR92_INTRINSIC_CHECK_FIELDS = (payload_label,)                   # consistency
```

**State-free components, and why.** For a given (seed, arm) every component is
protocol-determined before any governance decision: `epoch` and `event_class`
come from the protocol schedule; `batch_index` is the row's position inside
its write call's batch (the injected write stream is generated upstream of the
memory, so batch composition and order are identical whether earlier eligible
writes were committed by shadow or diverted by quarantine); `payload_label` is
the argmax of the incoming payload — intrinsic to the record, carried as a
consistency check rather than a join component. The incumbent fields stay in
the per-event records as **diagnostics only**: `incumbent_slot` was
empirically stable 24/24 in the smoke but is read from pre-write memory state
(not guaranteed intrinsic); `incumbent_last_write_seq` is the recorded §8
contaminant.

**Uniqueness is part of the claim.** The key assumes one eligible
(supersession) batch per epoch — true of every committed protocol (192/seed =
32 × epochs 6–11 on the panel arms; 24 = 8 × epochs 3–5 in the smoke). The
harness treats a duplicate key on either side as a **stop condition**
(`identity-collision`), never a joinable multiset.

**Instrumentation (additive, byte-inert).** The quarantine ledger already
persists all key components per diverted event (`diverted_events`, committed
instrumentation `d2d893c`). PR-9.2 adds the mirror on the shadow side:
`record_shadow_flag` gains an optional `flagged_events` list captured from the
same pre-write observables at the same point in the write path; the shadow
summary ledger gains `flagged_event_join_key` / `flagged_event_check_fields` /
`flagged_event_records`. No memory read, no RNG, no write: every retrieval
artifact stays byte-identical to `--govern none` (measured by G-style
inertness, pinned by test). `fork_events.csv` schema is untouched (the
byte-inertness invariant forbids new columns).

## 3. Certification procedure (frozen; the §9A gate of PR8 §4.7, completed)

The harness (`benchmarks/pr8_shadow_audit_cert.py`, analysis-only, no torch)
decides identity on the intrinsic key when per-event records exist on both
sides; legacy artifacts still take the incumbent path (the committed smoke
must keep scoring exactly as recorded). Additional PR-9.2 checks: per-joined-
event `payload_label` consistency; shadow-ledger ↔ shadow-`fork_events`
per-epoch count cross-link (the committed-schema anchor); incumbent-key
agreement reported as a diagnostic; instrumented-quarantine-re-run fidelity
(below); cross-host evidence via a gentoo-computed sha256 manifest.

**Panel (the frozen §8 panel, PR8 §4.5)** — `pr9_2/pr9_2_panel_run_matrix.sh`
on gentoo (`feature_cache_vitl14/`), seeds {0,1,2}:

* `none` arms: the **committed** baselines (pr7/twin; pairC clean from the
  pr4 grid), copied byte-verified into the panel tree — never re-generated;
* `shadow` arms: new runs (expected flagged: **192/seed** on every
  stale-soft cell, **0** on clean);
* `quarantine` arms: instrumented **re-runs** whose retrieval artifacts must
  be byte-identical to the committed §8 quarantine arms
  (`quarantine_rerun_fidelity`; the ledger may differ ONLY by the additive
  per-event fields) — this is how the committed §8 evidence and the new
  intrinsic-key records are proven to describe the same runs;
* prologue drift-guard: fresh `none` re-runs of pairD stale-soft s0 and pairA
  clean s0 byte-compared against the committed stems (a gentoo stack change
  since §8 is a stop condition, per the PR-10 re-baselining warning);
* same-seed shadow twin (pairD stale-soft s0) byte-compared on gentoo (the
  determinism half of §4.7 cond. 3);
* `sha256sum` manifest computed on gentoo over the panel tree; the darwin-side
  harness verifies every artifact it scores against those digests
  (the PR-10-consistent reading of "sha256-stable across darwin/gentoo":
  run artifacts are gentoo-canonical, cross-ARCHITECTURE recompute is
  refuted program-wide — PR10 result memo — so it is not demanded here).

**governance.json note.** The frozen-scorer readout is re-scored with the
CURRENT registered scorer for all three arms inside the panel tree (same
scorer on both sides of every comparison). It is a deterministic function of
the byte-compared CSVs; the committed §8-era governance files were written by
an older policy registry and are not byte-comparable across scorer versions —
they are left untouched and uncompared.

## 4. Acceptance (frozen before the run)

* PR-10 merge-abstain behavior untouched: no reader-path file changes, no
  changed answers, no new abstentions (nothing here runs `--read-govern`).
* Committed artifacts: zero modified/deleted; every addition under `pr9_2/`
  (plus this memo, the driver/harness diffs, and tests).
* §9A gate (PR8 §4.7), all five conditions, with identity decided on the
  intrinsic key: byte-inert retrieval on every cell/seed; complete ledger
  coverage (192/seed stale-soft, 0 clean); determinism (same-seed twin +
  gentoo sha manifest); clean control; per-seed audit report (the harness
  per-run blocks).
* Identity: intrinsic join 1:1 with empty symmetric difference and zero
  check-field mismatches on every non-clean cell; `identity-proven-empty` on
  clean cells; **any** mismatch, collision, non-additive rewrite, or
  unexplained row-count drift is a stop condition → `identity-blocked`.
* Full suite green on darwin (the canonical analysis host); the run stack is
  gentoo (canonical compute host), guarded by the prologue byte-checks.

---

## 5. Identity smoke, re-run with the intrinsic key (result: as registered)

`pr9_2/identity_smoke/` — the exact `pr8/identity_smoke/` divergent trio,
instrumented driver: `flag_set_identity` **pass, `identity-proven`,
intersection 24/24**, both symmetric differences empty, zero `payload_label`
mismatches, ledger anchored to shadow's own committed `fork_events.csv`. The
incumbent diagnostic reproduces the committed refutation **exactly: 8/24**
(first supersession epoch only). What refuted the legacy key is now a recorded
property; the intrinsic key is immune to it on the same artifacts. Inertness
and ledger coverage pass; smoke verdict `incomplete` (clean/panel/cross-host
absent by construction — correct smoke semantics).

## 6. Panel results (`pr9_2/cert_report.json`): **verdict `pass` → `promoted-audit`**

The full frozen §8 panel — 27 (cell, pair, seed) runs, gentoo
(`feature_cache_vitl14/`, torch 2.10.0+cu128), harness scored on darwin over
sha-verified transfers. Prologue drift-guard **passed** (fresh `none` re-runs
of pairD stale-soft s0 and pairA clean s0 byte-identical to the committed §8
stems — today's gentoo stack still reproduces the §8-era bytes); same-seed
shadow twin **byte-identical** (summary included).

All seven dimensions pass:

| dimension | result |
|---|---|
| inertness | pass — every shadow retrieval/readout artifact byte/content-identical to the committed `none` baseline on every cell/seed (governance re-scored same-scorer both sides) |
| ledger_coverage | pass — flagged **exactly 192/seed on all 21 stale-soft cells, 0 on all 6 clean cells**; disposition `flagged_not_diverted` everywhere |
| flag_set_identity | pass — **21 cells `identity-proven` (192/192 joined, 4,032 events total), 6 clean cells `identity-proven-empty`**; zero symmetric-difference entries, zero `payload_label` mismatches, zero collisions; fork_events anchor OK on every cell |
| clean_control | pass — byte-inert and zero flags on pairA and pairC, all seeds |
| panel_coverage | pass — the frozen §4.5 matrix complete, all artifacts present |
| cross_host_determinism | pass — 447 darwin-scored artifacts verified against the gentoo-computed sha256 manifest, 0 mismatches (method `gentoo-sha-manifest` + the on-gentoo same-seed twin; the PR-10-consistent reading of §4.7 cond. 3) |
| quarantine_rerun_fidelity | pass — all 24 instrumented re-runs byte-identical to the committed §8 quarantine arms on every retrieval artifact; summaries identical except the two declared additive ledger fields and one registry-provenance field (below) |

**The incumbent-key diagnostic, panel scale.** On every one of the 21
stale-soft cells the legacy key agrees on **exactly 32 of 192** joined events
— the first supersession epoch (epoch 6) — and diverges on all 160 events of
epochs 7–11, entirely in `incumbent_last_write_seq` (672/4,032 = 16.7%
agreement overall). The §8 refutation mechanism is thus fully regular at
panel scale, and fully attributable to the state component, exactly as §1
records.

**One post-registration comparison-spec correction (documented, not tuned).**
The first fidelity pass failed on all 24 comparisons with every retrieval
artifact byte-identical: the `govern` blocks differ in `implemented_actions`
(§8-era runs predate the `shadow` action's registration, so their recorded
registry lists 4 actions vs today's 5). That field records the driver's
action REGISTRY, not anything the run did — the same version-skew class as
governance.json across scorer versions (§3), so it was excluded from the
comparison and the exclusion pinned by test
(`test_fidelity_ignores_registry_version_but_not_run_fields`: registry skew
alone passes; any run-describing field — `action`, `step`, `events_seen`,
`outcome_counts`, `quarantined_events`, ledger counts/histogram — still
gates). No expected value, margin, or identity semantics moved.

## 7. Answers to the certification questions

1. **Old basis & contamination** — §1: `(epoch, event_class, incumbent_slot,
   incumbent_last_write_seq)`; the last component is incumbent state and
   diverges after the first diversion (8/24 smoke, 32/192 panel — first
   epoch only, mechanically regular).
2. **Proposed key** — §2: `(epoch, event_class, batch_index)` unique join,
   `payload_label` consistency check.
3. **State-free components** — all three join components are
   protocol-determined (schedule, schedule, batch position); `payload_label`
   intrinsic to the record; incumbent fields carried as diagnostics only.
4. **Identity-smoke expectation** — holds: intrinsic 24/24
   `identity-proven` on the recorded refutation trio; the 24/24 state-free
   component agreement of the pr8 smoke README is superseded by a true keyed
   join, and the 8/24 legacy behavior is reproduced as a diagnostic.
5. **Joins to existing artifacts** — clean: shadow ledger ↔ shadow's own
   committed `fork_events.csv` anchored per epoch on every cell; shadow ↔
   committed §8 quarantine evidence bridged via instrumented re-runs proven
   byte-identical to the committed arms (fidelity, 24/24).
6. **Existing certified rows unchanged** — zero committed artifacts modified
   (all additions under `pr9_2/` + driver/harness/tests/memo); PR-10's
   merge-abstain contract untouched (no reader-path change; no
   `--read-govern` anywhere in this PR); no changed answers, no new
   abstentions; full suite green.
7. **Ambiguity / collision / non-additive artifacts** — none: zero intrinsic
   key collisions on any of the 27 runs (the uniqueness claim held at
   192/seed panel scale); zero join ambiguity; the single non-additive-
   looking difference (`implemented_actions`) is registry-version provenance,
   fully explained and comparison-spec-corrected (§6). No unexplained
   row-count drift anywhere (the prologue and fidelity checks byte-pin the
   row streams).
8. **§9B sufficiency** — see §8.

## 8. Verdict: **`identity-certified`**

The §9A gate (PR8 §4.7) passes end-to-end — `promoted-audit` — with event
identity decided on the pre-registered write-event-intrinsic key: the shadow
ledger is a complete, inert, deterministic, event-addressable audit trail
whose flagged set provably equals the §8 quarantine diverted set on every
panel cell.

**What this makes possible (and does not).** §9B's entry conditions
(gate memo §5) are: (a) §9A passes — **now met**; (b) a parameter-free binary
second key exists in committed write-time observables — **still open, and
untouched by this PR**. §9B write-event authority is therefore now eligible
to be considered as a FUTURE pre-registered design — its own memo, its own
margins (the inherited §8 aggregate margins AND the frozen −3 per-seed bound
on every seed), its own branch — and nothing more: this PR implements no
write-event authority, changes no reader contract, and promotes no
quarantine behavior. `promoted-audit` certifies the detector/forensic role
only. The PR-11.1 negative stands unchanged: the read-time path over router
sets remains refuted; the escalation fork is now concretely §9B-shaped or
explicit acceptance of the residuals.
