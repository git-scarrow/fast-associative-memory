# PR-8 §9A identity smoke — committed artifact for the gate memo §8 note

This directory commits the hermetic divergent-run smoke whose result is
recorded in `PR8_QUARANTINE_REPLACEMENT_GATE.md` §8 ("Instrumentation note —
identity-key semantics"): the incumbent join key matches **8/24** events across
a shadow-vs-quarantine pair, so the certification harness reports
**`identity-violated`**. The memo previously carried this as prose only; this
directory makes it checkable.

## What ran

Three `run_vision` arms (`none`, `shadow`, `quarantine`) on a tiny hermetic
#87-shaped cache — the same construction as
`tests/test_pr8_diverted_keys.py::_make_cache` (dim 24, classes `[0, 8]`,
attractor 71, 24 rows/class, noise 0.05, torch generator seed 0) — with
`epochs=6, supersede_epoch=3, samples_per_class=8, held_out_per_class=8,
contraction=0.0, seed=0, payload_mode="soft"`, summaries written with
`json.dump(..., indent=2)` (the probe CLI's writer). Supersessions fire at
epochs 3–5 × 8 events/epoch = 24 quarantine-eligible events per arm.

Then `benchmarks/pr8_shadow_audit_cert.py --manifest manifest.json
--out-json cert_report.json` (run from this directory; `artifact_root` is
`"."`).

## Result (see `cert_report.json`)

* `inertness` **pass**, `ledger_coverage` **pass** — shadow is byte-inert and
  flags all 24 events (`disposition: flagged_not_diverted`).
* `flag_set_identity` **fail** → `identity-violated`: intersection **8/24**
  (the first supersession epoch, before divergence); epochs 4–5 differ on all
  16 remaining events.
* Overall verdict **fail** — the honest "or refute" half of *decide identity*,
  exactly as the gate memo §8 records. NOT a §9A certification either way
  (`mode: smoke`; clean-control/panel/cross-host dimensions incomplete by
  construction).

## The decision-relevant detail the prose could not carry

Every one of the 16 divergent events differs **only** in
`incumbent_last_write_seq`: the quarantine arm's incumbents stay frozen at
seq 32–39 (their epoch-3 writes were diverted), while the shadow arm's advance
to 48–55 (epoch 4) and 64–71 (epoch 5) because shadow commits the flagged
writes. `(epoch, event_class, incumbent_slot)` matches **24/24**. The
refutation is therefore entirely attributable to the state-dependent
`incumbent_last_write_seq` component of `QUARANTINE_DIVERTED_JOIN_KEY` — which
is direct evidence that a **write-event-intrinsic** identity key (one that does
not reference post-divergence incumbent state) can decide detector-eligibility
identity across these arms. A future §9A certification attempt must
pre-register such a key (gate memo §8) or accept this refutation.

Caveat: 24/24 agreement of the state-free key components is a property of this
smoke's geometry (no eviction, no slot reallocation between arms); it is
support for the intrinsic-key design, not a certification of it.
