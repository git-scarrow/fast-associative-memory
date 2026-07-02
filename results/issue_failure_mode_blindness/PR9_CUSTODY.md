# PR-9.0 — evidence custody (no science)

**Purpose.** Before PR-9 opens any new question, bring every piece of evidence
the PR-8 program's premises rest on onto `main`. Prior to this PR the entire §8
quarantine-promotion-gate evidence (full read-time panel, G2, G3, G4) lived
only on unmerged branches, in a repository with a documented history of
main-divergence losing work. No margin, verdict, or policy changes here.

## Merged to main

1. `feat/pr7-step7-quarantine-merge-stale-paire` (tag `pr7-step7-evidence`) —
   pairE merge_path_stale quarantine held-out replication (replicates pairD,
   milder; `needs_review`).
2. `feat/pr7-quarantine-promotion-gate` — `PR7_QUARANTINE_PROMOTION_GATE.md`
   plus its full evidence: step-8 read-time panel
   (`pr7/PR7_STEP8_READTIME_PANEL_RESULT.md`; aggregate pass on
   direct/collateral/clean; **G4** per-seed collateral breach pairD/s2 −4 < −3,
   masked by aggregate +26), G2 merge_path_stale completion
   (`pr7/PR7_G2_MERGE_PATH_STALE_COMPLETE.md`), and the G3 recoverability probe
   (`pr7/PR7_G3_RECOVERABILITY_RESULT.md`, `pr7/recovery_validation.json`;
   verdict `provenance_recoverable_not_harm_free`). Quarantine's earned status
   remains **`needs_review`**; nothing here promotes it.
   Merge-conflict resolution: the pairE twin artifacts existed on both branches
   — decompressed content verified identical (gzip headers differed); the
   promotion-gate branch's `twin_delta_quarantine.json` and
   `tests/test_pr7_quarantine_result.py` supersets were taken.
3. `feat/pr8-shadow-audit-diverted-keys` — the §9A pre-registration
   (`PR8_QUARANTINE_REPLACEMENT_GATE.md`), the `--govern shadow` arm, the
   analysis-only certification harness (`benchmarks/pr8_shadow_audit_cert.py`),
   and the diverted-event join-key instrumentation. §9A certification has NOT
   run. (The earlier `feat/pr8-quarantine-replacement-gate` branch is a
   superseded pre-rebase duplicate of the same work; retained as an archive
   branch per repo policy — do not prune — but not merged.)

## Repairs in this PR

* **Twin manifests regenerated** (`pr7/twin_delta.json`,
  `pr7/twin_delta_refuse.json`) by deterministic re-run of the byte-frozen
  analyzer after the merges made the pairB/pairE baseline and read-time-cell
  arms visible on main. Coverage-only: per_pair baseline blocks added; every
  pre-existing number and verdict byte-unchanged (verified field-by-field
  before regeneration). Restores the committed-manifest == fresh-build
  invariant (`tests/test_pr7_refuse_result.py`). `twin_delta_quarantine.json`
  already matched a fresh build (it was regenerated on the promotion-gate
  branch) and is untouched.
* **§8 identity smoke committed** — `pr8/identity_smoke/` (see its README):
  the gate memo §8's "8/24, `identity-violated`" note is now a machine-checkable
  artifact, with the additional finding that all 16 divergent events differ
  only in `incumbent_last_write_seq` (state-free key components match 24/24).
* **Frozen-detector citation resolved** —
  `pr3/FROZEN_DETECTOR_PROVENANCE.md` documents where the twin manifests'
  dangling `pr3/frozen_fit.*` paths actually resolve (the frozen analyzer
  constant is NOT edited; that would be a scorer change).

## Known remaining custody gaps (not fixed here)

* `results/issue_input_reembed_fidelity` 3-class re-embed per-probe CSV
  (9,085 rows) and analyzer outputs remain uncommitted on gentoo scratch —
  explicitly "pending review" per `REEMBED_3CLASS_SUBSET.md`; committing them
  is a review decision, not a mechanical custody fix, and needs the gentoo
  host.
* The #86/#87-era calibration artifacts referenced from other issue trees are
  unaffected by this PR.

## Verification

Full PR-7/PR-8 test set on merged main: 102 passed
(`test_pr7_quarantine_result`, `test_pr7_recovery_probe`,
`test_pr7_twin_delta`, `test_pr7_govern_noop`, `test_pr7_refuse_behavior`,
`test_pr7_refuse_result`, `test_pr8_diverted_keys`,
`test_pr8_shadow_audit_cert`).
