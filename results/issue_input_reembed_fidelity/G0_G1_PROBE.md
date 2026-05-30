# A2b · G0/G1 endpoint-recovery feasibility probe

**Verdict: PASS**

_all sampled endpoint cosines > 0.999 and all labels match._

Invariant tested: `cache row i -> ImageFolder sample train_indices[i] -> exact image path -> fresh DINOv2 embedding -> cosine(cached, fresh) > 0.999`.

## Run metadata

- Host: `gentoo`
- Command: `python probe_g0g1_reembed.py --split train`
- Cache path: `./feature_cache_inr_vitl14/imagenetr_dinov2_train.pt` (split=`train`)
- Model/checkpoint: `torch.hub facebookresearch/dinov2 :: dinov2_vitl14`
- Device: `cuda`
- Split/transform source of truth: `extract_imagenet_r_vitl14` (SEED=42, train_ratio=0.8)

## G0 — split replay & label recovery

- Cache rows: 23918; replayed split length: 23918 (OK)
- Sampled rows: 35 (classes [166, 63, 77, 156] + attractor 134 + 5 negatives, seed=0)
- Label-match rate: 1.0000 (35/35)

## G1 — fresh re-embed endpoint identity

- cosine(cached, reembedded): min=1.000000 median=1.000000 mean=1.000000 max=1.000000
- passing cos > 0.999: 35/35
- cache feature L2 norm: min=39.173416 mean=45.564987 max=48.670437
- extractor saved **normalized** features: **False** (norms ~1.0 => yes; cosine is norm-invariant so this does not affect the G1 check, but it dictates how the re-embed arm must feed the engine)

## Next-step decision

- ✅ Proceed to drafting `ReembedDriftStream` (endpoints provably shared between cache-linear and input-level arms).
