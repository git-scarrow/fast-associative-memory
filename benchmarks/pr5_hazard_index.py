"""PR-5 step 1 — zero-compute post-mortem + candidate hazard indices.

PR5_DESIGN.md §4 step 1. Constructs the candidate static hazard indices
from PR-4's spent-pair evidence ONLY: the committed pr3c/pr4 run
artifacts (post-mortem) and the existing vitl14 cache features (the
candidate index values). No new cache/governance run, no engine code, no
retrieval change, no held-out validation — the five spent pairs are
TRAINING DATA and are named as such; anything promoted here goes to
PR-5 step 2 (pre-registered held-out prediction) before it is believed.

Part 1 — post-mortem (committed artifacts, runs on either host):
re-scores the mixed-arm runs with frozen ``mode-conditioned-trust`` (the
PR-4 hazard probe, never a deployment candidate) and decomposes its
broken rows by TRUE probe class. PR-4 reported the direct/collateral
split per §3; this adds the per-class location of the harm. Finding that
motivates the candidates: in every pair the harm concentrates on the
BYSTANDER class most similar to a fork side — the bystander's traffic
co-resides with fork-party slots, so deprecating a fork side breaks the
bystander, and weak fork contrast (pair D: cos(I,O)=0.052) makes the
side-attribution itself unstable.

Part 2 — candidate indices (cache features, gentoo). All static,
seed-free, label-free at run time (class ids only), parameter-free.
With fork classes F = {classes[0], classes[-1]} (the supersession locus,
driver convention) and bystanders Y = the middle two:

* C1 ``attribution_ratio``  — max over (s in F, b in Y) of
  pair_cos(s, b) / pair_cos(I, O). Class-MEAN cosines, computable from
  the committed compression_index.json alone (cross-checked here).
  Captures bystander pull relative to fork contrast.
* C2 ``confusion_rate``     — fraction of all set samples whose highest
  within-set centroid cosine is NOT their own class. Sample-level tails
  that class means hide (pair A vs B: mean cosines within 2%, hazard 8x
  apart).
* C3 ``fork_confusion_rate``— C2 restricted to attribution-relevant
  confusions: sample's top centroid is cross-class AND exactly one of
  (own, top) is a fork class. Bystander<->fork confusion is the §3
  harm channel; bystander<->bystander confusion is not.

Promotion rule (fixed before any value was computed): a candidate is
admissible iff pair D is strictly worst AND pairs A and C are strictly
the two most benign (PR5_DESIGN.md §6 retrospective criterion); among
admissible candidates promote the one with the largest D-to-second-worst
ratio. If none is admissible, ``promoted`` is null and
``stop_recommended`` is true (PR5_DESIGN.md §7 first stop condition).

Usage (gentoo, where the cache lives; artifacts are committed on both):
  python benchmarks/pr5_hazard_index.py \
      --out results/issue_failure_mode_blindness/pr5/hazard_postmortem.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.pr4_compression_index import (  # noqa: E402
    load_unit_features, pair_cos)
import benchmarks.analyze_fork_governance as G  # noqa: E402

RESULTS = Path("results/issue_failure_mode_blindness")

# The five spent configurations, verbatim from pr4_run_matrix.sh. Fork
# pair = (classes[0], classes[-1]) — the driver's supersession locus.
PAIRS = {
    "pairA": {"classes": [0, 8, 19, 33], "attractor": 71},
    "pairC": {"classes": [10, 29, 42, 67], "attractor": 69},
    "pairD": {"classes": [10, 28, 32, 95], "attractor": 52},
    "pairB": {"classes": [5, 27, 48, 86], "attractor": 13},
    "pairE": {"classes": [47, 56, 61, 76], "attractor": 1},
}
SEEDS = (0, 1, 2)
PROBE_POLICY = "mode-conditioned-trust"  # frozen hazard probe (PR-5 §4)

CANDIDATES = ("attribution_ratio", "confusion_rate", "fork_confusion_rate")


def mixed_stem(pair: str, seed: int) -> Path:
    """Committed mixed-arm per-probe CSV for (pair, seed) — pr3c anchors
    for pair A and pair B s0, pr4 for everything else (pr4_run_matrix.sh)."""
    if pair == "pairA":
        return RESULTS / f"pr3c/per_probe_mixed_s{seed}.csv"
    if pair == "pairB" and seed == 0:
        return RESULTS / "pr3c/per_probe_mixed_s0_pairB.csv"
    return RESULTS / f"pr4/per_probe_mixed_{pair}_s{seed}.csv"


# ---------------------------------------------------------------------------
# Part 1 — post-mortem of the frozen probe's harm, by true class
# ---------------------------------------------------------------------------
def postmortem_pair(pair: str) -> dict:
    """Broken/direct counts by true (remapped) class, summed over seeds,
    for the frozen hazard probe on the committed mixed-arm artifacts."""
    classes = PAIRS[pair]["classes"]
    broken, direct, totals = Counter(), Counter(), Counter()
    for seed in SEEDS:
        run = G.load_run(mixed_stem(pair, seed))
        router = G.build_writetime_router(run["events"], run["slot_obs"])
        counterpart = G.pair_counterparts(router)
        guard = G.GuardIndex(run["topk"], counterpart)
        for p in run["probes"]:
            epoch = int(float(p["epoch"]))
            probe_index = int(p["probe_index"])
            cands = run["topk"][(epoch, probe_index)]
            truth = int(float(p["true_label"]))
            deployed = int(float(p["vote_pred_label"]))
            witness = G.fork_witness(cands)
            none_answer, _ = G._vote(cands, run["value_dim"])
            ans, _, detail = G.apply_policy(
                PROBE_POLICY, cands, witness, none_answer,
                run["value_dim"], epoch, run["slot_obs"], True,
                router, guard=guard, counterpart=counterpart,
                probe_index=probe_index)
            totals["broken_total"] += int(
                ans is not None and deployed == truth and ans != truth)
            if ans is None or deployed != truth or ans == truth:
                continue
            broken[truth] += 1
            surv = {c["slot"] for c in cands if c["surviving"]}
            top1 = max((c for c in cands if c["surviving"]),
                       key=lambda c: float(c["weight"]))["slot"]
            if detail and top1 in detail["dropped"]:
                cp = detail["counterpart"]
                if cp is None or cp.get(top1, set()) & surv:
                    direct[truth] += 1
    by_class = {str(classes[k]): {"remapped_label": k,
                                  "broken": broken.get(k, 0),
                                  "direct": direct.get(k, 0)}
                for k in range(len(classes))}
    worst = max(range(len(classes)), key=lambda k: broken.get(k, 0))
    return {"by_true_class": by_class,
            "broken_total": totals["broken_total"],
            "worst_class": classes[worst],
            "worst_class_is_bystander": worst in (1, 2),
            "worst_class_share": round(
                broken.get(worst, 0) / max(totals["broken_total"], 1), 6)}


# ---------------------------------------------------------------------------
# Part 2 — candidate static indices from cache features
# ---------------------------------------------------------------------------
def candidate_indices(embeds: torch.Tensor, labels: torch.Tensor,
                      classes: list[int]) -> dict:
    """The three candidate values + supporting geometry for one class set."""
    fork = (classes[0], classes[-1])
    bystanders = classes[1:-1]
    fork_pair = pair_cos(embeds, labels, *fork)
    bf = {f"{s},{b}": pair_cos(embeds, labels, s, b)
          for s in fork for b in bystanders}
    pairs = {f"{a},{b}": pair_cos(embeds, labels, a, b)
             for a, b in itertools.combinations(classes, 2)}

    # Sample-level within-set centroid confusion.
    feats, owner = [], []
    cents = []
    for c in classes:
        fc = embeds[labels == c]
        if fc.numel() == 0:
            raise ValueError(f"class {c} not in cache")
        feats.append(fc)
        owner.append(torch.full((fc.shape[0],), len(cents),
                                dtype=torch.long))
        cents.append(torch.nn.functional.normalize(
            fc.mean(dim=0), dim=-1))
    feats = torch.cat(feats)
    owner = torch.cat(owner)
    sims = feats @ torch.stack(cents).T            # (N_set, 4)
    top = sims.argmax(dim=1)
    confused = top != owner
    fork_idx = {0, len(classes) - 1}
    one_fork = torch.tensor(
        [(int(o) in fork_idx) != (int(t) in fork_idx)
         for o, t in zip(owner.tolist(), top.tolist())])
    n = feats.shape[0]
    return {
        "fork_pair": list(fork),
        "fork_pair_cos": fork_pair,
        "bystander_fork_cos": bf,
        "within_set_pair_cos": pairs,
        "n_samples": int(n),
        "n_confused": int(confused.sum()),
        "n_fork_confused": int((confused & one_fork).sum()),
        "attribution_ratio": max(bf.values()) / fork_pair,
        "confusion_rate": float(confused.float().mean()),
        "fork_confusion_rate": float((confused & one_fork).float().mean()),
    }


def hazard_ground_truth() -> dict:
    """Frozen-probe mixed-arm broken counts from the committed PR-4 table."""
    table = json.loads(
        (RESULTS / "pr4/pr4_geometry_table.json").read_text())
    out = {}
    for pair in PAIRS:
        by_seed = [table["governance"][f"{pair}/mixed/s{s}"]
                   [PROBE_POLICY]["broken"] for s in SEEDS]
        out[pair] = {"broken_by_seed": by_seed,
                     "broken_mean": round(sum(by_seed) / len(by_seed), 2)}
    return out


def evaluate(values: dict[str, dict]) -> dict:
    """Promotion per the rule frozen in the module docstring."""
    verdicts = {}
    for cand in CANDIDATES:
        v = {p: values[p][cand] for p in PAIRS}
        order = sorted(v, key=v.get, reverse=True)  # worst (highest) first
        d_worst = order[0] == "pairD" and v["pairD"] > v[order[1]]
        benign = set(order[-2:]) == {"pairA", "pairC"}
        verdicts[cand] = {
            "values": {p: round(v[p], 6) for p in order},
            "order_worst_first": order,
            "d_strictly_worst": d_worst,
            "ca_strictly_most_benign": benign,
            "admissible": d_worst and benign,
            "d_to_second_ratio": round(v["pairD"] / v[order[1]], 4)
            if order[0] == "pairD" else None,
        }
    admissible = [c for c in CANDIDATES if verdicts[c]["admissible"]]
    promoted = (max(admissible,
                    key=lambda c: verdicts[c]["d_to_second_ratio"])
                if admissible else None)
    return {"candidates": verdicts, "admissible": admissible,
            "promoted": promoted, "stop_recommended": promoted is None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vision-cache", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    embeds, labels, cache_path = load_unit_features(args.vision_cache)

    post = {p: postmortem_pair(p) for p in PAIRS}
    values = {p: candidate_indices(embeds, labels, PAIRS[p]["classes"])
              for p in PAIRS}

    # Cross-check the class-mean cosines against the committed gate-1 JSON
    # (same cache, same operationalization — must agree to float precision).
    gate1 = json.loads(
        (RESULTS / "pr4/compression_index.json").read_text())
    gate1_sets = {"pairA": gate1["pair_A"], "pairB": gate1["pair_B"],
                  "pairC": gate1["selected"]["pairC_alike"],
                  "pairD": gate1["selected"]["pairD_mid"],
                  "pairE": gate1["selected"]["pairE_compressed"]}
    for pair, ref in gate1_sets.items():
        for key, want in ref["within_set_pair_cos"].items():
            got = values[pair]["within_set_pair_cos"][key]
            if abs(got - want) > 1e-6:
                raise RuntimeError(
                    f"pair_cos mismatch vs gate-1 JSON: {pair} {key} "
                    f"got {got} want {want}")

    result = {
        "design": "PR5_DESIGN.md §4 step 1 (post-mortem + candidates)",
        "cache_path": cache_path,
        "n_cache_rows": int(labels.numel()),
        "probe_policy": PROBE_POLICY,
        "hazard_ground_truth": hazard_ground_truth(),
        "postmortem": post,
        "pair_geometry": values,
        "evaluation": evaluate(values),
        "gate1_cross_check": "within_set_pair_cos matches "
                             "pr4/compression_index.json to 1e-6",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    ev = result["evaluation"]
    for cand in CANDIDATES:
        v = ev["candidates"][cand]
        print(f"{cand:22} order: {' > '.join(v['order_worst_first'])}   "
              f"D-worst={v['d_strictly_worst']} "
              f"C/A-benign={v['ca_strictly_most_benign']}")
    print(f"promoted: {ev['promoted']}   "
          f"stop_recommended: {ev['stop_recommended']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
