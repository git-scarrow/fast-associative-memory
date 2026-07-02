# Frozen #87 detector — provenance note for the twin-manifest `frozen_detector` paths

The committed twin manifests (`pr7/twin_delta.json`, `pr7/twin_delta_refuse.json`,
`pr7/twin_delta_quarantine.json`) record, verbatim from the byte-frozen analyzer
constant (`benchmarks/pr7_twin_delta.py` `FROZEN_DETECTOR`):

```
fit_csv:     results/issue_failure_mode_blindness/pr3/frozen_fit.csv
fit_summary: results/issue_failure_mode_blindness/pr3/frozen_fit.summary.json
```

Those two files were never committed. This note records how the citation
actually resolves, without modifying the frozen analyzer (whose byte-identity
is an integrity field of the PR-7/PR-8 program) or any certified manifest.

**The frozen detector is a procedure, not a coefficient file** (PR-3a,
`benchmarks/score_frozen_detector.py` module docstring): it is reproduced by
refitting standardization + logistic(rank_gap, manifold_support) on the parity
-train epochs of the #87 study's forced zone, from the #87 study's own
committed per-probe CSV, then VERIFIED against the study's persisted summary
(held-out parity AUC and Youden threshold must match to recorded precision;
the scorer RAISES on mismatch rather than scoring with an unverified detector).

The actually-committed fit set is therefore:

* fit CSV — `results/issue_vitl14_blend_confidence/per_probe.csv`
* fit summary — `results/issue_vitl14_blend_confidence/heldout_abstention_summary.json`
* verified fitted parameters (means, stds, coefficients, intercept, threshold)
  — dumped in `results/issue_failure_mode_blindness/pr3a_frozen_detector_scores.json`

The `pr3/frozen_fit.*` paths in the manifests are the intended commit location
that was never populated; they should be read as pointers to the three files
above. Any future change to the analyzer constant is a scorer change and must
be treated as such (new sha, new pre-registration), not slipped in as a docs
fix — hence this note instead.
