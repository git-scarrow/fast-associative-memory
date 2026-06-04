# A2b · ReembedDriftStream pilot gates

**Verdict: PASS**

_parity holds, endpoints reproduce (min cos 0.999998 > 0.999), and the input-level path diverges nontrivially mid-path (max 6.780e-01 > 1e-03)._

## Run metadata

- Host: `gentoo` · Device: `cuda`
- Command: `probe_reembed_pilot.py --split train --category 166 --attractor 134 --samples-per-class 8 --held-out-per-class 4 --seed 0 --contractions 0.25,0.5,0.75`
- Cache: `./feature_cache_inr_vitl14/imagenetr_dinov2_train.pt` (split=`train`)
- Model: `torch.hub facebookresearch/dinov2 :: dinov2_vitl14`
- Pilot: class 166 + attractor 134, 8 train / 4 held, seed=0
- Split/transform source of truth: `extract_imagenet_r_vitl14` (SEED=42, train_ratio=0.8)

## Gate results

- **Parity** (same endpoints + assignment as VisionDriftStream): OK
- **G1** endpoint identity (c=0/c=1): min cosine 0.999998 (threshold > 0.999)
- **G2** mid-path divergence vs cache-linear: c=0.25: 5.757e-01, c=0.5: 6.780e-01, c=0.75: 6.556e-01 (nontrivial > 1e-03)

## What this answers

1. Parity preserved? — see Parity above.
2. c=0/c=1 reproduce cached endpoints at cos > 0.999? — see G1.
3. Input-level path diverges nontrivially mid-path? — see G2.
4. If not, marked INCONCLUSIVE (not PASS). — see Verdict.

## Next-step decision

- ✅ Gates pass. Proceed to the 3-class subset run (separate, authorized step).
