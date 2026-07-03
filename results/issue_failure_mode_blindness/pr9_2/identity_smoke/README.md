# PR-9.2 identity smoke — the intrinsic key on the §8 refutation trio

The exact `pr8/identity_smoke/` construction, re-run with the PR-9.2
instrumented driver (shadow now persists per-event `flagged_event_records`).
Same hermetic #87-shaped cache (dim 24, classes `[0, 8]`, attractor 71, 24
rows/class, noise 0.05, torch generator seed 0 — recipe in this README; the
cache file itself is not committed), same arms (`none`, `shadow`,
`quarantine`), same protocol (`epochs=6, supersede_epoch=3,
samples_per_class=8, held_out_per_class=8, contraction=0.0, seed=0,
payload_mode="soft"`): supersessions at epochs 3–5 × 8/epoch = 24 eligible
events per arm. Harness: `benchmarks/pr8_shadow_audit_cert.py --manifest
manifest.json --out-json cert_report.json` (run from this directory).

## Result (`cert_report.json`)

* `inertness` **pass**, `ledger_coverage` **pass** — shadow remains byte-inert
  with the per-event instrumentation active, and flags all 24 events.
* `flag_set_identity` **pass** → **`identity-proven`** on the pre-registered
  write-event-intrinsic key `(epoch, event_class, batch_index)`:
  intersection **24/24**, both symmetric differences empty, zero
  `payload_label` check-field mismatches, shadow ledger anchored to its own
  committed `fork_events.csv` (per-epoch counts equal).
* The incumbent-key **diagnostic** reproduces the committed pr8 smoke's
  refutation exactly: agreement **8/24** (the first supersession epoch only;
  epochs 4–5 diverge in `incumbent_last_write_seq` because shadow commits the
  flagged writes while quarantine diverts them). What was the gating failure
  in `pr8/identity_smoke/` is now a recorded property of the legacy key.
* Overall verdict **`incomplete`** — correct for a smoke (`mode: smoke`):
  clean-control, panel coverage, and cross-host dimensions are absent by
  construction. NOT a §9A certification; the panel run is
  (`pr9_2/pr9_2_panel_run_matrix.sh`).

## What this settles (and what it does not)

The gate memo §8's open question — whether a write-event-intrinsic key can
decide detector-eligibility identity across divergent shadow/quarantine runs —
is answered **yes on this geometry**: the same trio that refuted the incumbent
key proves clean under the intrinsic key. The caveat from the pr8 README
carries over unchanged: no eviction or slot reallocation occurs in this smoke;
panel-scale certification (192 events/seed, five pairs, real #87 geometry) is
the §6 result in `PR9_2_IDENTITY_CERT.md`.
