# A2b ReembedDriftStream 3-Class Subset

## Scope

Host-only Gentoo/CUDA run using PR #94 driver (`probe_reembed_subset.py`).

**Input path:** Re-embedded ImageNet-R pixel cross-fade → fresh DINOv2 ViT-L/14 embeddings

**Selection state:**
- Classes: 166, 63, 77
- Attractor: 134
- Split: train
- Samples per class: 32
- Held-out per class: 64
- Seed: 0

**Calibration schedule:**
- Epochs: 30
- Contraction: 0.35 → 0.9
- Output CSV: `results/issue_input_reembed_fidelity/per_probe_reembed.csv`

## Environment

- Host: gentoo
- Main HEAD: 13c3706 (Merge PR #94)
- CUDA device: NVIDIA GeForce RTX 4080 SUPER
- Note: xFormers warnings were performance-only, not correctness blockers

## Gate Result

**Pilot verdict: PASS**

The `pilot_verdict()` check passed before CSV write. The CSV was written only after a successful PASS verdict; no data written on FAIL/ABORT.

## CSV Summary

- **Rows:** 9085 (+ 1 header)
- **Fields:** 27
  - Core: vigilance_policy, retrieval_floor_policy, epoch, contraction
  - Probe state: rho_probe, within_probe, offclass_weight_mean, acc_epoch, sim_floor_active, floor_delta_ema
  - Top-1: top1_top2_margin, top1_sim, top2_sim
  - Voting: vote_entropy, effective_support, max_vote_weight, n_surviving_votes, vote_pred_label, vote_correct
  - Ground truth: true_label, top1_label, top1_correct
  - Fidelity: is_blended, offclass_weight, true_margin, margin_bucket
- **Vigilance policies:** margin, relative
- **Epochs:** 0–29 (30 total)
- **Contraction range:** 0.35–0.9

## Policy Summary

### margin (live-delta floor)
- Rows: 4422
- vote_correct_mean: 0.763682
- top1_correct_mean: 0.759611

### relative
- Rows: 4663
- vote_correct_mean: 0.753378
- top1_correct_mean: 0.748660

## Coverage

### margin
- Epoch 0 rows: 152
- Epoch 29 rows: 137
- n_epochs: 30

### relative
- Epoch 0 rows: 152
- Epoch 29 rows: 146
- n_epochs: 30

## Bucket and Blend Distributions

### margin
- margin_bucket: {0: 1140, 1: 70, 2: 3212}
- is_blended: {0: 3200, 1: 1222}

### relative
- margin_bucket: {0: 1275, 1: 107, 2: 3281}
- is_blended: {0: 3241, 1: 1422}

## Interpretation

This run validates that the input-reembedded pixel-interpolation path can complete the same subset calibration protocol after passing the paired-path gates from PR #93.

The observed accuracy (margin: 0.764, relative: 0.753 vote_correct) is in the same broad regime as the earlier cache-linear anchor, indicating the re-embed path does not degrade the learning signal in this subset selection.

**Scope limitations:**
- This is a three-class subset, not a full A2b calibration.
- This does not include the cache-linear anchor comparison unless separately authorized.
- Geometry diagnostics (arc length, deviation-vs-c) are deferred to a later PR.
- Raw CSV remains uncommitted pending review.

**Next steps contingent on authorization.**
