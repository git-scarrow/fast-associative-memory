# Transfer benchmark: ρ(t) onset criteria on real drifting LLM-memory vectors

Closes #71. Tests whether the two failure-onset criteria validated on the
**synthetic** contracting manifold (#69 telemetry, #70 driver) transfer to a
**real** production embedding manifold.

## Method

Driver: `benchmarks/probe_contraction.py` (now dual-source). Both paths feed the
*identical* loop — `learn_local` → `get_stats` → `probe_cross_class_similarity`
→ onset flags — and emit the same CSV columns and onset report.

- **Synthetic baseline** (`--synthetic`, default): 8 random gaussian class
  centers in 96-d, contracted toward a shared attractor *in vector space*, with
  gaussian within-class noise. Tuned so within-class Δ ≈ 0.92.
- **Real transfer test** (`--real`): 4 well-separated 20-Newsgroups topics
  (`rec.sport.baseball, sci.space, talk.politics.mideast, misc.forsale`)
  embedded with **all-MiniLM-L6-v2** (384-d). The shared attractor is `sci.med`.
  Drift is produced at the **text level**: at contraction `c`, a fraction `c` of
  each document's sentences is swapped for sentences from the shared attractor
  pool, then the real model re-embeds. The vector trajectory is whatever the
  model emits — *not* a linear vector interpolation — so this exercises real
  anisotropic geometry and real intra-topic spread. Train/held-out documents are
  disjoint pools per class; memory persists across epochs.

Reproduce:

```bash
# synthetic baseline
python benchmarks/probe_contraction.py --epochs 36 --classes 8 --dim 96 \
  --noise 0.02 --contraction-end 0.999 \
  --csv results/probe_real_transfer/synthetic.csv \
  --verdict-json results/probe_real_transfer/synthetic_verdict.json --plot \
  --plot-path results/probe_real_transfer/synthetic.png

# real-embedding transfer test
HF_HUB_OFFLINE=1 python benchmarks/probe_contraction.py --real \
  --epochs 24 --contraction-end 0.97 --samples-per-class 40 --held-out-per-class 40 \
  --real-categories "rec.sport.baseball,sci.space,talk.politics.mideast,misc.forsale" \
  --attractor-category "sci.med" \
  --csv results/probe_real_transfer/real.csv \
  --verdict-json results/probe_real_transfer/real_verdict.json --plot \
  --plot-path results/probe_real_transfer/real.png
```

## Results

| quantity | synthetic | real (all-MiniLM-L6-v2) |
|---|---|---|
| within-class Δ (epoch 0) | 0.92 | **0.44** |
| Δ-anchored predicted blend ρ* | 0.862 | 0.329 |
| observed blend-onset ρ | 0.896 | **0.242** (epoch 0, **left-censored**) |
| blend offset (obs − pred) | +0.033 | −0.087 |
| max ρ at full contraction | 0.976 | **0.481** |
| structural-chimera onset (ρ ≥ 0.95) | reached (epoch 31, ρ=0.963) | **never reached** |

## Verdict

**Confirmed-with-quantified-offset for the relative criterion; the absolute
thresholds do not transfer.**

1. **Relative blend criterion transfers (with offset).** The Δ-anchored formula
   `ρ* = Δ − temp·ln((1−ε)/ε)` predicts the blend onset within ≈0.09 cosine on
   both manifolds (synthetic +0.033, real −0.087, both inside the ±2·temp = ±0.10
   band). The sign flips on real data: real embeddings blend *slightly earlier*
   than predicted, consistent with the formula modelling a single off-class
   competitor while real space has many moderately-similar neighbours that
   aggregate vote mass.

2. **The absolute ρ thresholds do NOT transfer.** The synthetic blend onset
   (~0.79–0.90 ρ) is meaningless on real embeddings: the off-class vote mass
   already exceeds `blend_eps` at **ρ ≈ 0.24 with zero drift** (the onset is
   left-censored at epoch 0). Real all-MiniLM-L6-v2 keeps within-class Δ ≈ 0.44,
   so the Δ−ρ margin is small at rest and the engine begins in the blend regime
   before any manifold contraction.

3. **The structural-chimera ceiling (v_ceiling = 0.95) is unreachable on this
   real manifold.** Even at full text-level collapse (c = 0.97), real cross-class
   ρ plateaus at **0.481** — far below 0.95 — because the embedding model's own
   within-class similarity floor (≈0.44) caps how tightly classes can converge.
   A fixed absolute 0.95 ceiling never admits boundary writes here, so EMA
   chimera contamination via that gate does not occur under natural drift; it is
   an artefact of the synthetic manifold's ability to collapse ρ → 1.

## Implication (out of scope for #71, motivates the follow-up)

Both findings argue against absolute ρ anchors and for VIGIL's proposed
**relative / percentile** anchor: the Δ-relative blend formula is the part that
survives transfer, and a fixed 0.95 chimera ceiling is simultaneously
unreachable on real embeddings and reachable on the synthetic toy. A
drift-compensated vigilance floor anchored to the live Δ (or a running ρ
percentile) is the natural next step — tracked separately, not changed here.
