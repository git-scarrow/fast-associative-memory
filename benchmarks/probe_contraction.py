#!/usr/bin/env python3
"""Driver for the rho(t) cross-class contraction probe.

Runs a continual-learning loop on a *deliberately contracting* synthetic
manifold and collects the two telemetry curves added in PR #69:

  1. In-band per-epoch cross-class similarity (cheap; ContinuousCAM.get_stats()).
  2. Held-out, true-label probe (authoritative; probe_cross_class_similarity()).

The manifold is contracted by interpolating every class center toward a shared
attractor direction with a coefficient that grows linearly across epochs. This
makes the inter-class cosine rho(t) rise by construction, so we can observe the
two failure onsets VIGIL predicted on a frozen-manifold-assumption engine:

  * Retrieval-blend onset: off-class softmax vote mass crosses `blend_eps`,
    analytically near rho ~= Delta - inference_temp * ln((1-eps)/eps)
    (~0.79 for Delta~0.9, eps=0.10, temp=0.05).
  * Structural-chimera onset: rho crosses the dynamic-vigilance ceiling
    (v_ceiling, default 0.95), after which boundary writes are admitted and EMA
    bakes them into the prototype keys.

Memory PERSISTS across epochs (the realistic continual setting where EMA
contamination accumulates); only the per-epoch stats are reset.

Example (run from repo root):
    python benchmarks/probe_contraction.py --epochs 30 --classes 8 --dim 64 \
        --contraction-end 0.97 --csv contraction.csv --plot
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from associative_core import ContinuousCAM  # noqa: E402
from dynamic_vigilance import DynamicVigilance  # noqa: E402


def make_epoch_batch(centers: torch.Tensor, attractor: torch.Tensor,
                     contraction: float, n_per_class: int, noise: float,
                     generator: torch.Generator):
    """Generate one epoch's (queries, one-hot targets, true labels).

    `contraction` in [0, 1) mixes each class center toward `attractor`; higher
    values pull all classes together, raising inter-class similarity.
    """
    C, D = centers.shape
    mixed = F.normalize((1.0 - contraction) * centers + contraction * attractor,
                        dim=-1)                                  # (C, D)
    labels = torch.arange(C).repeat_interleave(n_per_class)      # (C*n,)
    base = mixed[labels]                                         # (C*n, D)
    eps = torch.randn(base.shape, generator=generator) * noise
    queries = F.normalize(base + eps, dim=-1)                   # (C*n, D)
    targets = F.one_hot(labels, num_classes=C).float()          # (C*n, C)
    return queries, targets, labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--classes", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--samples-per-class", type=int, default=32)
    ap.add_argument("--held-out-per-class", type=int, default=32)
    ap.add_argument("--noise", type=float, default=0.10,
                    help="per-sample gaussian noise scale around the class center")
    ap.add_argument("--contraction-start", type=float, default=0.0)
    ap.add_argument("--contraction-end", type=float, default=0.97,
                    help="final attractor-mix coefficient (->1 collapses classes)")
    ap.add_argument("--max-entries", type=int, default=4096)
    ap.add_argument("--blend-eps", type=float, default=0.10,
                    help="off-class vote-mass threshold counted as a blend")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--v-base", type=float, default=0.92)
    ap.add_argument("--alpha", type=float, default=0.30)
    ap.add_argument("--v-floor", type=float, default=0.30)
    ap.add_argument("--v-ceiling", type=float, default=0.95)
    ap.add_argument("--csv", type=str, default="contraction.csv")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot-path", type=str, default="contraction.png")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    # Fixed, well-separated class centers and a shared attractor direction.
    centers = F.normalize(torch.randn(args.classes, args.dim, generator=gen), dim=-1)
    attractor = F.normalize(torch.randn(1, args.dim, generator=gen), dim=-1)

    dv = DynamicVigilance(v_base=args.v_base, alpha=args.alpha,
                          v_floor=args.v_floor, v_ceiling=args.v_ceiling)
    mem = ContinuousCAM(key_dim=args.dim, value_dim=args.classes,
                        max_entries=args.max_entries, dynamic_vigilance=dv)

    fields = ["epoch", "contraction",
              "rho_inband", "var_inband", "n_inband",
              "rho_probe", "var_probe", "within_probe", "margin_probe",
              "offclass_weight", "frac_blended", "n_vote",
              "chimera_onset", "blend_onset", "mean_v_eff"]
    rows = []

    first_blend = None
    first_chimera = None
    delta_ref = None  # within-class self-similarity at epoch 0 (the analytic Delta)

    for epoch in range(args.epochs):
        # Linear contraction schedule.
        if args.epochs > 1:
            frac = epoch / (args.epochs - 1)
        else:
            frac = 0.0
        contraction = args.contraction_start + frac * (
            args.contraction_end - args.contraction_start)

        mem.reset_dynamic_vigilance_stats()

        # --- Train (continual: memory persists across epochs) ---
        q, y, _ = make_epoch_batch(centers, attractor, contraction,
                                   args.samples_per_class, args.noise, gen)
        mem.learn_local(q, y)

        # --- Cheap in-band curve (lagged: reflects pre-call memory state) ---
        stats = mem.get_stats()

        # --- Authoritative held-out probe (post-write, true labels) ---
        hq, _, hlab = make_epoch_batch(centers, attractor, contraction,
                                       args.held_out_per_class, args.noise, gen)
        p = mem.probe_cross_class_similarity(hq, hlab, blend_eps=args.blend_eps)

        if delta_ref is None and not math.isnan(p.get("mean_within_class_sim", float("nan"))):
            delta_ref = p["mean_within_class_sim"]

        if first_blend is None and p.get("blend_onset"):
            first_blend = (epoch, p["mean_cross_class_sim"], p["mean_offclass_weight"])
        if first_chimera is None and p.get("chimera_onset"):
            first_chimera = (epoch, p["mean_cross_class_sim"])

        rows.append({
            "epoch": epoch,
            "contraction": round(contraction, 4),
            "rho_inband": stats.get("mean_cross_class_sim", float("nan")),
            "var_inband": stats.get("var_cross_class_sim", float("nan")),
            "n_inband": stats.get("n_cross_class_obs", 0),
            "rho_probe": p.get("mean_cross_class_sim", float("nan")),
            "var_probe": p.get("var_cross_class_sim", float("nan")),
            "within_probe": p.get("mean_within_class_sim", float("nan")),
            "margin_probe": p.get("mean_true_margin", float("nan")),
            "offclass_weight": p.get("mean_offclass_weight", float("nan")),
            "frac_blended": p.get("frac_blended", float("nan")),
            "n_vote": p.get("n_vote_probes", 0),
            "chimera_onset": bool(p.get("chimera_onset", False)),
            "blend_onset": bool(p.get("blend_onset", False)),
            "mean_v_eff": stats.get("mean_v_effective", float("nan")),
        })
        print(f"epoch {epoch:3d} | c={contraction:.3f} | "
              f"rho_probe={rows[-1]['rho_probe']:.3f} "
              f"within={rows[-1]['within_probe']:.3f} "
              f"offclass_w={rows[-1]['offclass_weight']:.3f} "
              f"frac_blend={rows[-1]['frac_blended']:.2f} "
              f"| blend={rows[-1]['blend_onset']} chimera={rows[-1]['chimera_onset']}")

    # --- Write CSV ---
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} epochs to {args.csv}")

    # --- Onset report: observed vs analytic prediction ---
    print("\n=== Onset report ===")
    if delta_ref is not None:
        eps = args.blend_eps
        predicted_blend_rho = delta_ref - mem.inference_temp * math.log((1 - eps) / eps)
        print(f"Delta (within-class self-sim, epoch 0): {delta_ref:.3f}")
        print(f"Predicted blend onset rho ~= Delta - temp*ln((1-eps)/eps) = {predicted_blend_rho:.3f} "
              f"(temp={mem.inference_temp}, eps={eps})")
    if first_blend is not None:
        e, rho, ow = first_blend
        print(f"Observed retrieval-blend onset: epoch {e}, rho={rho:.3f}, offclass_weight={ow:.3f}")
    else:
        print("Observed retrieval-blend onset: not reached")
    print(f"Predicted structural-chimera onset rho >= v_ceiling = {args.v_ceiling}")
    if first_chimera is not None:
        e, rho = first_chimera
        print(f"Observed structural-chimera onset: epoch {e}, rho={rho:.3f}")
    else:
        print("Observed structural-chimera onset: not reached")

    # --- Optional plot ---
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping plot")
            return
        ep = [r["epoch"] for r in rows]
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(ep, [r["rho_probe"] for r in rows], "o-", color="C0", label="rho_probe (true-label)")
        ax1.plot(ep, [r["rho_inband"] for r in rows], ".--", color="C0", alpha=0.5, label="rho_inband (cheap)")
        ax1.plot(ep, [r["within_probe"] for r in rows], "-", color="C2", alpha=0.6, label="within-class (Delta)")
        ax1.axhline(args.v_ceiling, color="C3", ls=":", label=f"v_ceiling={args.v_ceiling}")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("cosine similarity"); ax1.set_ylim(0, 1.02)
        ax2 = ax1.twinx()
        ax2.plot(ep, [r["offclass_weight"] for r in rows], "s-", color="C1", label="off-class vote mass")
        ax2.axhline(args.blend_eps, color="C1", ls=":", alpha=0.6)
        ax2.set_ylabel("off-class vote mass"); ax2.set_ylim(0, 1.02)
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=8)
        ax1.set_title("rho(t) contraction and retrieval-blend onset")
        fig.tight_layout(); fig.savefig(args.plot_path, dpi=120)
        print(f"Wrote plot to {args.plot_path}")


if __name__ == "__main__":
    main()
