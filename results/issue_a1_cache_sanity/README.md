# PR-1 · A1 cache sanity — feature-cache inventory & integrity manifest

**Analysis only.** No retrieval intervention, no benchmark code touched:
`forward()`, the live-Δ floor, the vote, and `associative_core` are byte-for-byte
unchanged. This PR adds a read-only inventory tool (`tools/cache_sanity.py`), its
hermetic tests, and per-host manifests.

> **Verdict (one line):** **A1 cache sanity is CLOSED.** The gentoo (compute
> host) manifest confirms all 16 referenced caches load with full integrity and
> that both decisive caches meet their study requirements —
> `vitl14_cifar100_train` satisfies A1/#87 (classes `{0,8,19,33}` at 500 rows
> each, attractor `71` present) and `inr_vitl14_train` satisfies the INR breadth
> study (classes `{166,63,77,156}` at 131–250 rows each, ≥ 96 required;
> attractor `134` present). The darwin checkout has **no** caches (analysis/push
> host only); that inventory is likewise closed. SHA256 fingerprints are
> recorded for cross-host cache identity.

## Why

Every A-track collapse study (#85/#86/#87, INR breadth) and several MT
benchmarks read precomputed DINOv2 feature caches that are **gitignored and
host-specific**. Resolution is fragile by construction:

* `feature_cache_vitl14` is a **git-tracked symlink** (mode `120000`) whose
  target is a machine-specific Linux path
  (`/mnt/data/dev/projects/campus-fabric/feature_cache_vitl14`) — a dead link on
  any macOS checkout, a valid one on the compute host.
* `benchmarks/probe_contraction.py::_VISION_CACHE_CANDIDATES` papers over this
  with a fallback list of absolute `/mnt/...` paths.
* `benchmarks/benchmark_dynamic_vigilance.py:89-90` hardcodes one of those
  absolute paths directly in `DATASET_SPECS` (CIFAR-100 entry).
* Cache writers are **not key-uniform**: `extract_stanford_dogs_vitl14.py`
  saves the feature tensor under `"features"`; every other extractor uses
  `"embeds"`.

Until now, no artifact recorded which caches a given host actually has, whether
they are internally consistent, or whether they satisfy the exact class/count
requirements the A1/A2 streams impose. That record is a precondition for the
upcoming (k, τ) sweep and risk-coverage runs (roadmap A2/A3).

## What the tool checks

`tools/cache_sanity.py` inspects every cache file referenced by committed code
(A-track: CIFAR-100 ViT-L/14, ImageNet-R ViT-L/14, CIFAR-100 ViT-B/14;
adapter tier: Dogs/Birds/Cars/Aircraft/Flowers) and reports, per file:

| check | meaning |
|---|---|
| `status` | `ok` / `missing` / `broken-symlink` / `load-error` / `format-error` / `integrity-error` |
| `symlink` | first symlinked path component, its target, and whether the target resolves |
| `tried` | the resolution order, mirroring `_resolve_vision_cache()` (primary + `/mnt` fallbacks) |
| `feature_key` | `embeds` vs `features` (auto-detected; the writers disagree) |
| `features` | shape, dtype, NaN/Inf counts, L2-norm mean/std/min/max |
| `labels` | classes present, label range, per-class min/median/max counts |
| `checks.study_requirements_met` | A1 (#87): classes `{0,8,19,33}` ≥ 96 rows each + attractor `71` present; INR breadth: classes `{166,63,77,156}` ≥ 96 each + attractor `134` present |
| `sha256` (opt-in `--hash`) | content hash for cross-host cache identity |

Loads use `weights_only=True` (the caches are plain tensor dicts; note
`VisionDriftStream` currently loads with `weights_only=False` — recorded here as
an observation, not changed).

## How to run

```bash
# this host's inventory (fast; metadata only)
python tools/cache_sanity.py --out results/issue_a1_cache_sanity/manifest_$(uname -s | tr A-Z a-z).json

# from outside the repo (e.g. a copied script), pin the resolution root:
python /tmp/cache_sanity.py --root /path/to/fast-associative-memory --hash --out manifest.json

# hermetic tests (no real caches needed)
python -m pytest tests/test_cache_sanity.py -q
```

## Findings — darwin (analysis/push host), 2026-06-10 · **inventory closed**

`manifest_darwin.json`, repo @ `c3d87e7` (origin/main, Codeberg canonical):

| cache | status |
|---|---|
| `feature_cache_vitl14/cifar100_dinov2_{train,test}.pt` | **broken-symlink** — tracked link → `/mnt/data/dev/projects/campus-fabric/feature_cache_vitl14`, target absent; both `/mnt` fallbacks also absent |
| `feature_cache_inr_vitl14/…`, `feature_cache_vitb14/…`, all `data/*` adapter caches | **missing** |

This darwin checkout can run *no* cache-dependent benchmark — it is the
analysis/push host (consistent with the established split: compute on gentoo,
push from darwin). All A2/A3 sweep compute must be scheduled on gentoo.

## Findings — gentoo (compute host), 2026-06-10 · **A1 cache sanity closed**

`manifest_gentoo.json` (hashed), repo checkout @ `13c3706`; caches are
gitignored so the manifest is commit-independent. Run twice: first without
`--hash` (resolution + load integrity), then with `--hash` (largest file is
196 MB, so fingerprinting is cheap).

| cache | status | study check |
|---|---|---|
| `vitl14_cifar100_train` (50k × 1024) | **ok** — resolves through the tracked symlink (`target_exists: true`) | **A1 (#87) met**: classes `0/8/19/33` → 500 rows each; attractor `71` → 500 |
| `vitl14_cifar100_test` | ok | — |
| `inr_vitl14_train` | **ok** (real directory at repo root) | **INR breadth met**: classes `166/63/77/156` → 184/250/131/163 rows (≥ 96 required); attractor `134` → 232 |
| `inr_vitl14_test` | ok | — |
| `vitb14_cifar100_{train,test}` | ok | — |
| all 10 `data/*` adapter caches (Dogs/Birds/Cars/Aircraft/Flowers) | ok | — |

**Summary: 16/16 ok** — every cache loads with `weights_only=True`, rows match
labels, zero NaN/Inf. The exact published class selections of #87 and the INR
breadth study are reusable as-is for the upcoming (k, τ) sweep and risk-coverage
runs.

## Deferred host-coupling issues (documented, deliberately NOT fixed here)

1. The **git-tracked machine-specific symlink** `feature_cache_vitl14 →
   /mnt/data/dev/projects/campus-fabric/...` (valid on gentoo, dead elsewhere).
2. The **hardcoded absolute `/mnt` path** in
   `benchmarks/benchmark_dynamic_vigilance.py:89-90` (CIFAR-100 spec).

Fixing either changes benchmark resolution order on the compute host; both are
out of scope for this analysis-only PR and should be addressed, if at all, in a
dedicated change with gentoo verification.

## Files

| file | what |
|---|---|
| `tools/cache_sanity.py` | the inventory tool (stdlib + torch only; no new dependencies) |
| `tests/test_cache_sanity.py` | 10 hermetic tests over synthetic caches (ok / missing / broken-symlink / NaN / wrong-keys / class-requirements / fallback order / hashing) |
| `manifest_darwin.json` | analysis/push host inventory (generated; no caches present) |
| `manifest_gentoo.json` | compute host inventory with SHA256 fingerprints (16/16 ok; both study checks met) |
