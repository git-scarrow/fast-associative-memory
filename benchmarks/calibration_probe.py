#!/usr/bin/env python3
"""Per-probe retrieval-confidence dump for the issue #82 calibration study.

ANALYSIS ONLY. This driver introduces **no retrieval intervention**: it runs the
exact #83/#84 contraction loop (real 20NG, live-Δ retrieval floor) and, at each
epoch, reads the held-out probe with ``return_per_probe=True`` to emit one row
per voting probe. ``forward()`` is untouched; ``vote_pred_label`` is pinned to
``forward().argmax`` row-for-row by tests/test_calibration_probe.py.

The question the dump exists to answer (analysed separately by
analyze_calibration.py): does any *label-free* per-probe signal
(``top1_top2_margin``, ``vote_entropy``, ``effective_support``,
``max_vote_weight``, ``n_surviving_votes``, plus the epoch-scalar
``sim_floor_active``) separate a correct top-1 / correct vote from a wrong one in
the recoverable mid-window (e8–e18)? top-1 correctness is used ONLY as an
evaluation label here, never to steer retrieval.

Both vigilance policies (margin = DynamicVigilance, relative = RelativeVigilance)
are run over the *same* embedding stream — the stream depends only on the
contraction schedule and seed, not on memory — so the only difference between the
two passes is write-side prototype shaping. Held-out docs are never written, so
``--held-out-per-class`` scales calibration sample size with no effect on the
manifold dynamics or onsets.

Example (from repo root):
    python benchmarks/calibration_probe.py \
        --epochs 30 --contraction-end 0.9 --held-out-per-class 64 \
        --out results/issue82_retrieval_confidence_calibration/per_probe.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from associative_core import ContinuousCAM  # noqa: E402
from dynamic_vigilance import (  # noqa: E402
    DynamicVigilance, RelativeVigilance, RetrievalFloorPolicy)
from benchmarks.probe_contraction import (  # noqa: E402
    RealDriftStream, VisionDriftStream)


# Per-probe columns: epoch-scalar context first, then the label-free predictors,
# then the evaluation labels. Keep this list in sync with PER_PROBE_KEYS below.
CONTEXT_COLS = ["vigilance_policy", "retrieval_floor_policy", "epoch",
                "contraction", "rho_probe", "within_probe", "offclass_weight_mean",
                "acc_epoch", "sim_floor_active", "floor_delta_ema"]
# Label-free, deployable-at-inference predictors (true labels unknown):
PREDICTOR_COLS = ["top1_top2_margin", "top1_sim", "top2_sim", "vote_entropy",
                  "effective_support", "max_vote_weight", "n_surviving_votes"]
# Evaluation labels / label-aware context (NOT available at inference):
LABEL_COLS = ["true_label", "top1_label", "top1_correct", "vote_pred_label",
              "vote_correct", "is_blended", "offclass_weight", "true_margin",
              "margin_bucket"]
PER_PROBE_KEYS = (["true_label", "top1_label", "top1_correct", "vote_pred_label",
                   "vote_correct", "is_blended", "offclass_weight",
                   "top1_top2_margin", "top1_sim", "top2_sim", "vote_entropy",
                   "effective_support", "max_vote_weight", "n_surviving_votes",
                   "true_margin", "margin_bucket"])
ALL_COLS = CONTEXT_COLS + PREDICTOR_COLS + LABEL_COLS


def build_policy(name: str, args):
    if name == "relative":
        return RelativeVigilance(v_floor=args.v_floor, v_ceiling=args.v_ceiling,
                                 margin_guard=args.margin_guard,
                                 percentile=args.percentile)
    return DynamicVigilance(v_base=args.v_base, alpha=args.alpha,
                            v_floor=args.v_floor, v_ceiling=args.v_ceiling)


def run_policy(policy_name, batches, num_classes, dim, args, writer):
    """Run the continual loop for one vigilance policy, append per-probe rows."""
    dv = build_policy(policy_name, args)
    floor = RetrievalFloorPolicy(k_logit_steps=args.k_logit_steps,
                                 ema_beta=args.floor_ema_beta)
    mem = ContinuousCAM(key_dim=dim, value_dim=num_classes,
                        max_entries=args.max_entries, dynamic_vigilance=dv,
                        retrieval_floor_policy=floor)
    n_rows = 0
    for epoch, (contraction, q, y, _lab, hq, hlab) in enumerate(batches):
        mem.reset_dynamic_vigilance_stats()
        mem.learn_local(q, y)
        stats = mem.get_stats()
        acc = (mem.forward(hq).argmax(dim=-1) == hlab).float().mean().item()
        p = mem.probe_cross_class_similarity(hq, hlab, blend_eps=args.blend_eps,
                                             return_per_probe=True)
        pp = p.get("per_probe")
        if pp is None or pp["true_label"].numel() == 0:
            print(f"[{policy_name}] epoch {epoch:3d}: no voting probes, skipped")
            continue
        ctx = {
            "vigilance_policy": policy_name,
            "retrieval_floor_policy": "live-delta",
            "epoch": epoch,
            "contraction": round(contraction, 4),
            "rho_probe": round(p.get("mean_cross_class_sim", float("nan")), 6),
            "within_probe": round(p.get("mean_within_class_sim", float("nan")), 6),
            "offclass_weight_mean": round(p.get("mean_offclass_weight", float("nan")), 6),
            "acc_epoch": round(acc, 6),
            "sim_floor_active": round(float(stats.get("sim_floor_active", float("nan"))), 6),
            "floor_delta_ema": round(float(stats.get("floor_delta_ema", float("nan"))), 6),
        }
        V = pp["true_label"].numel()
        for i in range(V):
            row = dict(ctx)
            for k in PER_PROBE_KEYS:
                v = pp[k][i].item()
                row[k] = v
            writer.writerow(row)
            n_rows += 1
        print(f"[{policy_name}] epoch {epoch:3d} | c={contraction:.3f} "
              f"rho={ctx['rho_probe']:.3f} acc={acc:.3f} "
              f"floor={ctx['sim_floor_active']:.3f} probes={V}")
    return n_rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--samples-per-class", type=int, default=32,
                    help="train docs/class (kept at 32 to match #83/#84 dynamics)")
    ap.add_argument("--held-out-per-class", type=int, default=64,
                    help="held probe docs/class; never written, so larger only "
                         "grows calibration sample size (no effect on onsets)")
    ap.add_argument("--contraction-start", type=float, default=0.0)
    ap.add_argument("--contraction-end", type=float, default=0.9)
    ap.add_argument("--max-entries", type=int, default=4096)
    ap.add_argument("--blend-eps", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policies", type=str, default="margin,relative",
                    help="comma-separated vigilance policies to run")
    # vigilance params
    ap.add_argument("--v-base", type=float, default=0.92)
    ap.add_argument("--alpha", type=float, default=0.30)
    ap.add_argument("--v-floor", type=float, default=0.30)
    ap.add_argument("--v-ceiling", type=float, default=0.95)
    ap.add_argument("--margin-guard", type=float, default=0.05)
    ap.add_argument("--percentile", type=float, default=0.95)
    # live-Δ retrieval floor (#74) — the primary-slice floor
    ap.add_argument("--k-logit-steps", type=float, default=3.0)
    ap.add_argument("--floor-ema-beta", type=float, default=0.9)
    # real 20NG stream
    ap.add_argument("--real-categories", type=str,
                    default="sci.space,rec.sport.hockey,talk.politics.guns,comp.graphics")
    ap.add_argument("--attractor-category", type=str, default="sci.med")
    ap.add_argument("--embed-model", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--max-sentences", type=int, default=24)
    # vision ViT-L/14 stream (A1 transfer test) — swaps the 20NG text source for a
    # cached DINOv2 ViT-L/14 feature manifold. Everything downstream (engine, probe,
    # floor, vote, analysis) is identical; only the embedding stream changes.
    ap.add_argument("--vision", action="store_true",
                    help="drive the loop from a cached DINOv2 ViT-L/14 feature "
                         "manifold instead of 20NG text (A1 transfer test)")
    ap.add_argument("--vision-cache", type=str, default=None,
                    help="path to the *_train.pt feature cache; if unset, probe the "
                         "feature_cache_vitl14 symlink + known archive locations")
    ap.add_argument("--vision-classes", type=str, default="0,8,19,33",
                    help="comma-separated cached class ids used as classes (vision)")
    ap.add_argument("--vision-attractor-class", type=int, default=71,
                    help="cached class id whose features are the shared attractor "
                         "blended into every class (vision)")
    ap.add_argument("--out", type=str,
                    default="results/issue82_retrieval_confidence_calibration/per_probe.csv")
    args = ap.parse_args()

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    if args.vision:
        vcats = [int(c.strip()) for c in args.vision_classes.split(",") if c.strip()]
        print(f"[vision] loading DINOv2 ViT-L/14 cache, {len(vcats)} classes "
              f"(attractor={args.vision_attractor_class}) ...")
        stream = VisionDriftStream(
            categories=vcats, attractor_category=args.vision_attractor_class,
            samples_per_class=args.samples_per_class,
            held_out_per_class=args.held_out_per_class,
            seed=args.seed, cache_path=args.vision_cache)
        num_classes, dim = stream.num_classes, stream.dim
        print(f"[vision] cache={stream.cache_path} dim={dim}, classes={vcats}, "
              f"held/class={args.held_out_per_class}")
    else:
        cats = [c.strip() for c in args.real_categories.split(",") if c.strip()]
        print(f"[real] loading {args.embed_model} and {len(cats)} 20NG topics "
              f"(attractor={args.attractor_category}) ...")
        stream = RealDriftStream(
            categories=cats, attractor_category=args.attractor_category,
            samples_per_class=args.samples_per_class,
            held_out_per_class=args.held_out_per_class,
            model_name=args.embed_model, max_sentences=args.max_sentences,
            seed=args.seed)
        num_classes, dim = stream.num_classes, stream.dim
        print(f"[real] dim={dim}, classes={cats}, held/class={args.held_out_per_class}")

    # Precompute every epoch's batch ONCE (identical across policies).
    print(f"[embed] precomputing {args.epochs} epoch batches ...")
    batches = []
    for epoch in range(args.epochs):
        frac = epoch / (args.epochs - 1) if args.epochs > 1 else 0.0
        contraction = args.contraction_start + frac * (
            args.contraction_end - args.contraction_start)
        (q, y, lab), (hq, hlab) = stream.batch(contraction)
        batches.append((contraction, q, y, lab, hq, hlab))
    print(f"[embed] done ({sum(b[1].size(0) + b[4].size(0) for b in batches)} encodes)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLS)
        writer.writeheader()
        for pol in policies:
            print(f"\n=== vigilance policy: {pol} ===")
            total += run_policy(pol, batches, num_classes, dim, args, writer)
    print(f"\nWrote {total} per-probe rows to {out_path}")


if __name__ == "__main__":
    main()
