#!/usr/bin/env python3
"""A2b · ReembedDriftStream pilot gates (host-only, analysis-only).

Tiny paired-path pilot (1 class + 1 attractor, a handful of samples, a few `c`
values) that validates ONLY the three build-time gates from
`results/issue_input_reembed_fidelity/REEMBED_DRIFT_STREAM_DESIGN.md` §4/§6:

  1. Assignment / endpoint parity with VisionDriftStream.
  2. G1 — c=0 / c=1 reproduce the cached endpoints at cosine > 0.999.
  3. G2 — the input-level (pixel cross-fade) path diverges nontrivially from the
     cache-linear path mid-path. If it does not, the pilot is INCONCLUSIVE
     (NOT a pass) and must be diagnosed before any subset run.

It does NOT run the subset/full A2b experiment, touch forward()/
associative_core.py/the detector/retrieval, or modify VisionDriftStream. It is a
read-only consumer of the merged feature cache + the live DINOv2 model.

Host-only: requires the ImageNet-R dataset, the DINOv2 ViT-L/14 weights, and the
feature cache the extractor wrote. The gate *logic* is covered host-independently
by tests/test_reembed_drift_stream.py.
"""
import argparse
import json
import os
import socket
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "benchmarks"))

import extract_imagenet_r_vitl14 as ex  # noqa: E402
from probe_contraction import VisionDriftStream  # noqa: E402
from reembed_drift_stream import (  # noqa: E402
    G1_COS_THRESHOLD, G2_MIN_DIVERGENCE, ReembedDriftStream, pilot_verdict)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--category", type=int, default=166,
                    help="single content class for the pilot")
    ap.add_argument("--attractor", type=int, default=134,
                    help="attractor class (must differ from --category)")
    ap.add_argument("--samples-per-class", type=int, default=8)
    ap.add_argument("--held-out-per-class", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--contractions", type=str, default="0.25,0.5,0.75",
                    help="comma-separated mid-path c values for the G2 probe")
    ap.add_argument("--command", default="", help="verbatim command (for the report)")
    ap.add_argument("--out",
                    default="results/issue_input_reembed_fidelity/REEMBED_PILOT.md")
    args = ap.parse_args()

    if args.category == args.attractor:
        ap.error("--category and --attractor must differ")
    contractions = tuple(float(x) for x in args.contractions.split(",") if x.strip())

    host = socket.gethostname()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_path = os.path.join(ex.CACHE_DIR, f"imagenetr_dinov2_{args.split}.pt")

    # Cache-linear arm.
    vision = VisionDriftStream(
        categories=[args.category], attractor_category=args.attractor,
        samples_per_class=args.samples_per_class,
        held_out_per_class=args.held_out_per_class,
        seed=args.seed, cache_path=cache_path)

    # Live model + input-level arm (replays the SAME selection through pixels).
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    model = model.to(device).eval()
    reembed = ReembedDriftStream.from_extractor(
        vision, ex, model, device=device, split=args.split)

    verdict = pilot_verdict(reembed, vision, contractions=contractions)

    summary = {
        "verdict": verdict["verdict"], "gate": verdict["gate"],
        "reason": verdict["reason"], "host": host, "device": str(device),
        "cache_path": cache_path, "split": args.split,
        "category": args.category, "attractor": args.attractor,
        "samples_per_class": args.samples_per_class,
        "held_out_per_class": args.held_out_per_class, "seed": args.seed,
        "contractions": list(contractions),
        "parity": verdict["parity"],
        "g1_min_cosine": verdict["g1_min_cosine"],
        "g1_threshold": G1_COS_THRESHOLD,
        "g2_divergence_per_c": verdict["g2_divergence"],
        "g2_min_divergence": G2_MIN_DIVERGENCE,
    }
    print(json.dumps(summary, indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    g2 = verdict["g2_divergence"]
    g2_str = ", ".join(f"c={c}: {d:.3e}" for c, d in zip(contractions, g2)) if g2 else "n/a"
    g1_str = f"{verdict['g1_min_cosine']:.6f}" if verdict["g1_min_cosine"] is not None else "n/a"
    lines = [
        "# A2b · ReembedDriftStream pilot gates\n",
        f"**Verdict: {verdict['verdict']}**"
        f"{f' (gate: {verdict['gate']})' if verdict['gate'] else ''}\n",
        f"_{verdict['reason']}_\n",
        "## Run metadata\n",
        f"- Host: `{host}` · Device: `{device}`",
        f"- Command: `{args.command}`",
        f"- Cache: `{cache_path}` (split=`{args.split}`)",
        "- Model: `torch.hub facebookresearch/dinov2 :: dinov2_vitl14`",
        f"- Pilot: class {args.category} + attractor {args.attractor}, "
        f"{args.samples_per_class} train / {args.held_out_per_class} held, seed={args.seed}",
        f"- Split/transform source of truth: `extract_imagenet_r_vitl14` "
        f"(SEED={ex.SEED}, train_ratio={ex.TRAIN_RATIO})\n",
        "## Gate results\n",
        f"- **Parity** (same endpoints + assignment as VisionDriftStream): "
        f"{'OK' if verdict['parity'] else 'BROKEN'}",
        f"- **G1** endpoint identity (c=0/c=1): min cosine {g1_str} "
        f"(threshold > {G1_COS_THRESHOLD})",
        f"- **G2** mid-path divergence vs cache-linear: {g2_str} "
        f"(nontrivial > {G2_MIN_DIVERGENCE:.0e})\n",
        "## What this answers\n",
        "1. Parity preserved? — see Parity above.",
        "2. c=0/c=1 reproduce cached endpoints at cos > 0.999? — see G1.",
        "3. Input-level path diverges nontrivially mid-path? — see G2.",
        "4. If not, marked INCONCLUSIVE (not PASS). — see Verdict.\n",
        "## Next-step decision\n",
    ]
    if verdict["verdict"] == "PASS":
        lines.append("- ✅ Gates pass. Proceed to the 3-class subset run "
                     "(separate, authorized step).")
    elif verdict["verdict"] == "INCONCLUSIVE":
        lines.append("- ⏸ INCONCLUSIVE. Do NOT proceed to the subset. Diagnose why "
                     "the input-level path is collinear with the cache-linear path.")
    else:
        lines.append("- ⛔ FAIL. Stop and diagnose per the reason above before any "
                     "further work.")
    lines.append("")
    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {args.out}", file=sys.stderr)

    # Non-zero exit on non-PASS so a CI/automation caller cannot mistake an
    # INCONCLUSIVE/FAIL pilot for a green light.
    sys.exit(0 if verdict["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
