# A2b · `ReembedDriftStream` — design note (DESIGN ONLY, no implementation)

**Status: design only. No code written. No run started.** Follows the merged
G0/G1 probe (PR #89) and the paired-experiment scope (`SCOPE.md`). This note
fixes the class contract for `ReembedDriftStream` so implementation, when
authorized, is mechanical. It does **not** modify `forward()`,
`associative_core.py`, the detector, `calibration_probe.py`, or
`VisionDriftStream`.

---

## 0. What G0/G1 already settled (inputs to this design)

- **Endpoints recover bit-exactly.** `cache row i → stratified_split(SEED=42,
  0.8) → ImageFolder.samples[idx] → path → fresh DINOv2 ViT-L/14 → cosine 1.0`
  (35/35, raw vs raw). The row↔image map is the replay mechanism this stream
  implements — no new cache needed (Open Decision 1 in SCOPE resolves to
  "replay works", not "write new path cache").
- **Cache features are RAW**, L2 norm ~39–49, *not* unit. Therefore the
  re-embed arm must feed **raw DINOv2 output** to the engine (engine normalizes
  internally, as the text `RealDriftStream` does). Unit-normalization is for
  geometry diagnostics only.
- This clears **endpoint identity only**. The open risk this stream exists to
  probe is whether the *path between* shared endpoints carries the #82→#88
  signal or is an artifact of moving linearly through embedding space.

---

## 1. The single invariant the class must guarantee

> For every (sample, contraction `c`) pair, `ReembedDriftStream` returns the
> embedding of an **image-space** blend of the **same two source images** whose
> cached embeddings `VisionDriftStream` blends linearly — same per-sample
> attractor assignment, same endpoints, same `c` schedule. **Only the
> interpolation space differs (pixels vs embeddings).**

Everything below is in service of that one invariant. If any design choice
breaks endpoint-sharing or per-sample-pairing parity with `VisionDriftStream`,
the comparison is void.

---

## 2. Contract (shape parity with `VisionDriftStream`)

`ReembedDriftStream` must be a **drop-in** for `VisionDriftStream` in the
existing `calibration_probe.py` per-policy run function — same constructor
surface, same per-epoch yield contract, same emitted `per_probe.csv` schema
(`PER_PROBE_KEYS`). The driver imports the existing run function verbatim; it is
not duplicated.

**Construction must reuse, not re-derive, the selection state.** `VisionDriftStream`
must expose read-only accessors for `tr` / `ho` (train/held-out sample rows), `aidx`
(attractor row pool), `_train_assign` / `_held_assign` (per-sample → attractor
assignment). `ReembedDriftStream` **consumes those exact tensors**. Same `seed`
→ identical selection is necessary but not sufficient; consuming the same objects
is the guard against silent drift if either side's RNG order changes.
`ReembedDriftStream` must not recompute class selection, RNG order, or assignments.

```
ReembedDriftStream(
    vision_stream,          # a constructed VisionDriftStream — source of tr/ho/aidx/assign + endpoints
    extractor=ex,           # extract_imagenet_r_vitl14 — split/transform/loader source of truth
    model,                  # live dinov2_vitl14, .eval(), on device
    blend="pixel_crossfade",# the only operator in the pilot (see §3)
    device,
)
```

It must **not** take its own class list / seed / split logic. It derives 100% of
selection from `vision_stream`. This is what makes "shared endpoints" structural
rather than hopeful.

---

## 3. Per-step behavior

For sample row `i` assigned to attractor row `a = assign[i]`, at contraction `c`:

1. Map both rows back to source images via the **G0 replay** (already proven):
   `X_img = loader(samples[split_indices[i]][0])`,
   `A_img = loader(samples[split_indices[a_global]][0])`.
   (`a_global` is the attractor's full-dataset index via the same `split_indices`
   table; the attractor pool `aidx` indexes cache rows, so it goes through the
   identical map.)
2. **Pixel cross-fade** on resized/cropped ToTensor'd `[0,1]` tensors *before* Normalize:
   - Apply `Resize(224,BICUBIC)→CenterCrop→ToTensor` to each endpoint image independently
   - Blend in cropped `[0,1]` space: `img(c) = (1−c)·toT(X_img) + c·toT(A_img)`
   - Then `Normalize`, then `model(·)`
   This order preserves G1 endpoint identity at `c=0`/`c=1` (byte-identical to cache pipeline)
   while making the blend shape-defined across differently-sized PILs.
3. Return the **raw** `model(img(c))` output (no normalize) to the engine.

`c=0 ⇒ embed(X_img)` and `c=1 ⇒ embed(A_img)`, both provably equal to the cached
raw endpoints (G1). Only `0<c<1` differs from the linear arm — the experiment.

---

## 4. Build-time gates (must pass before any subset run)

Re-assert at stream-construction / pilot time, not just trust PR #89:

- **G1 re-check inside the stream** — first call at `c=0` and `c=1` must hit
  cosine > 0.999 against the cached endpoint. If the resize-before-blend ordering
  in §3 perturbs the endpoint pipeline, this catches it immediately.
- **G2 paths diverge** — mid-path `1 − cos(q_linear(c), q_reembed(c))` must be
  non-trivially > 0 at some `c`. If divergence is not meaningfully observed,
  mark the pilot **INCONCLUSIVE** (not "passed"). Do not continue to the subset
  until the cause is understood.
- **Assignment-parity assertion** — `assert torch.equal(reembed.assign,
  vision.assign)` (and tr/ho/aidx). Cheap, and the only thing standing between
  "shared endpoints" and a subtly mispaired comparison.

---

## 5. Explicitly out of scope for this stream

- Perturbation-severity drift (blur/noise) — SCOPE §2 secondary; deferred.
- CIFAR / any non-ImageNet-R, non-ViT-L/14 source (encoder-variable; SCOPE §3).
- Any change to retrieval, the detector, held-out logic, or analysis scripts —
  they run unchanged on the shared `per_probe.csv` schema.
- Writing a new path/index cache — unnecessary; G0 replay suffices.

## 6. Build-time gates (pilot scale)

Before the 3-class subset run, execute a tiny pilot (1 class + 1 attractor,
~8 samples, ~6 `c` values) purely to validate:
- G1 endpoint identity (byte-identical to cache at `c=0`/`c=1`)
- Assignment parity (`assert torch.equal(reembed.assign, vision.assign)`)
- G2 path divergence (non-trivial mid-path divergence)

The pilot is only to validate these three gates. If G2 fails to show meaningful
divergence, mark the pilot **INCONCLUSIVE** and diagnose before proceeding.

— design note only; awaiting authorization to implement.
