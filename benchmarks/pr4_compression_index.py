"""PR-4 gate 1 — static class-set compression index (PR4_DESIGN.md §4, §8 step 1).

Computes the pre-registered geometry index for class-set configurations
BEFORE any PR-4 cache/governance run, from the existing vitl14 cache
features only. No engine, no retrieval, no governance code is touched.

Operationalization (recorded in the PR4_DESIGN.md gate-1 addendum):

* Features are the cached DINOv2 ViT-L/14 embeddings, unit-normalized
  exactly as ``VisionDriftStream`` normalizes them (the engine sees unit
  features; the index must describe the geometry retrieval actually sees).
* For an unordered class pair (i, j): ``pair_cos(i, j)`` = mean cosine
  over ALL cached row pairs (full cross-class mean), seed-free and
  static — deliberately NOT the per-seed 32/64 probe subsets, so the
  index exists once per pair, before and independent of any run.
* For a 4-class set: ``set_index`` = mean of the 6 within-set pair
  cosines. Secondary diagnostics (reported, not selective): the
  stale-fork pair cosine (classes[0], classes[-1]) — the supersession
  locus — and the mean class-to-attractor cosine.
* Higher index = more compressed geometry. The §4 validity check is that
  pair B scores strictly above pair A; if it does not, the index is
  wrong and must be revised before any governance run.

Candidate scan + selection rule (fixed here, by index, not by results):
candidate 4-class sets + attractor are drawn with a fixed-seed generator
from classes unused by pair A / pair B; from the scored candidates the
script selects

* ``pairC_alike``  — index nearest pair A (H1's new pair-A-like pair),
* ``pairD_mid``    — index nearest the A/B midpoint,
* ``pairE_compressed`` — the highest-index candidate strictly above
  pair B (absent if no candidate exceeds B).

Usage (gentoo, where the cache lives):
  python benchmarks/pr4_compression_index.py \
      --out results/issue_failure_mode_blindness/pr4/compression_index.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.probe_contraction import _resolve_vision_cache  # noqa: E402

# The PR-2/PR-3 anchor configurations, verbatim from pr3c_run_matrix.sh.
PAIR_A = {"classes": [0, 8, 19, 33], "attractor": 71}
PAIR_B = {"classes": [5, 27, 48, 86], "attractor": 13}

CANDIDATE_SEED = 0
N_CANDIDATES = 256


def load_unit_features(cache_path: str | None):
    """Cache embeddings unit-normalized as VisionDriftStream normalizes them."""
    cache = _resolve_vision_cache(cache_path)
    blob = torch.load(cache, map_location="cpu", weights_only=False)
    embeds = F.normalize(blob["embeds"].float(), dim=-1)
    labels = blob["labels"].long()
    return embeds, labels, str(cache)


def pair_cos(embeds: torch.Tensor, labels: torch.Tensor,
             a: int, b: int) -> float:
    """Mean cosine over all cross-class row pairs of classes a and b."""
    fa = embeds[labels == a]
    fb = embeds[labels == b]
    if fa.numel() == 0 or fb.numel() == 0:
        raise ValueError(f"class {a if fa.numel() == 0 else b} not in cache")
    return float((fa @ fb.T).mean())


def set_index(embeds: torch.Tensor, labels: torch.Tensor,
              classes: list[int], attractor: int) -> dict:
    """Set-level compression index + secondary diagnostics for one config."""
    pairs = {f"{a},{b}": pair_cos(embeds, labels, a, b)
             for a, b in itertools.combinations(classes, 2)}
    vals = list(pairs.values())
    return {
        "classes": list(classes),
        "attractor": attractor,
        "index": sum(vals) / len(vals),
        "within_set_pair_cos": pairs,
        "stale_fork_pair_cos": pair_cos(embeds, labels,
                                        classes[0], classes[-1]),
        "attractor_mean_cos": sum(
            pair_cos(embeds, labels, c, attractor) for c in classes
        ) / len(classes),
    }


def candidate_configs(used: set[int], n_classes: int) -> list[dict]:
    """Fixed-seed candidate 4-class sets + attractor over unused classes."""
    free = [c for c in range(n_classes) if c not in used]
    gen = torch.Generator().manual_seed(CANDIDATE_SEED)
    out = []
    for _ in range(N_CANDIDATES):
        pick = torch.randperm(len(free), generator=gen)[:5].tolist()
        out.append({"classes": sorted(free[i] for i in pick[:4]),
                    "attractor": free[pick[4]]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vision-cache", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    embeds, labels, cache_path = load_unit_features(args.vision_cache)
    n_classes = int(labels.max()) + 1

    anchor_a = set_index(embeds, labels, **{
        "classes": PAIR_A["classes"], "attractor": PAIR_A["attractor"]})
    anchor_b = set_index(embeds, labels, **{
        "classes": PAIR_B["classes"], "attractor": PAIR_B["attractor"]})
    order_ok = anchor_b["index"] > anchor_a["index"]

    used = set(PAIR_A["classes"] + PAIR_B["classes"]
               + [PAIR_A["attractor"], PAIR_B["attractor"]])
    candidates = []
    seen = set()
    for cfg in candidate_configs(used, n_classes):
        key = (tuple(cfg["classes"]), cfg["attractor"])
        if key in seen:
            continue
        seen.add(key)
        candidates.append(set_index(embeds, labels, **cfg))

    ia, ib = anchor_a["index"], anchor_b["index"]
    mid = (ia + ib) / 2
    sel = {
        "pairC_alike": min(candidates, key=lambda c: abs(c["index"] - ia)),
        "pairD_mid": min(candidates, key=lambda c: abs(c["index"] - mid)),
    }
    above_b = [c for c in candidates if c["index"] > ib]
    sel["pairE_compressed"] = (max(above_b, key=lambda c: c["index"])
                               if above_b else None)

    result = {
        "design": "PR4_DESIGN.md §4 static compression index (gate 1)",
        "cache_path": cache_path,
        "n_cache_rows": int(labels.numel()),
        "n_classes": n_classes,
        "candidate_seed": CANDIDATE_SEED,
        "n_candidates_requested": N_CANDIDATES,
        "n_candidates_scored": len(candidates),
        "pair_A": anchor_a,
        "pair_B": anchor_b,
        "index_orders_A_below_B": order_ok,
        "selected": sel,
        "candidate_index_range": [
            min(c["index"] for c in candidates),
            max(c["index"] for c in candidates)],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(f"pair A index: {ia:.6f}   pair B index: {ib:.6f}")
    print("INDEX ORDER (A < B): " + ("OK" if order_ok else "FAILED"))
    for name, cfg in sel.items():
        if cfg is None:
            print(f"{name}: NONE (no candidate above pair B)")
        else:
            print(f"{name}: classes={cfg['classes']} "
                  f"attractor={cfg['attractor']} index={cfg['index']:.6f}")
    print(f"wrote {out}")
    return 0 if order_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
