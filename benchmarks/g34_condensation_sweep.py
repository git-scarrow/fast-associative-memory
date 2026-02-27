#!/usr/bin/env python3
"""
benchmarks/g34_condensation_sweep.py — G34: Per-Domain Vigilance + NSTP Condensation Sweep.

Three phases per domain:
  Phase 1: Vigilance sweep (no NSTP) to find accuracy-condensation tradeoff
  Phase 2: NSTP sweep at best vigilance from Phase 1
  Phase 3: Fine-tuned vigilance around Phase 1 winner with best NSTP from Phase 2

Run:
    PYTHONPATH=. python benchmarks/g34_condensation_sweep.py --data-root data --device auto
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fast_associative_memory import FastAssociativeMemory  # noqa: E402
from nstp import NSTPController  # noqa: E402
from benchmarks.g30_adapter_sweep import (  # noqa: E402
    load_domain_features as g30_load_domain_features,
)
from benchmarks.g31_retrain_adapters import _load_g30_domain_adapter  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

Domain4 = Literal["dogs", "birds", "cars", "aircraft"]
DOMAINS: List[Domain4] = ["dogs", "birds", "cars", "aircraft"]

G31_BASELINES: Dict[Domain4, float] = {
    "dogs": 89.98,
    "birds": 90.78,
    "cars": 89.80,
    "aircraft": 73.51,
}

ACCURACY_TOLERANCE_PP = 2.0

TARGET_CONDENSATION: Dict[Domain4, float] = {
    "dogs": 0.50,
    "birds": 0.70,
    "cars": 0.70,
    "aircraft": 0.90,
}

STRETCH_CONDENSATION: Dict[Domain4, float] = {
    "dogs": 0.40,
    "birds": 0.50,
    "cars": 0.50,
    "aircraft": 0.70,
}

PHASE1_VIGILANCES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]
PHASE2_NSTP_ITERS = [1, 3, 5, 10]

CSV_COLUMNS = [
    "phase", "domain", "vigilance", "nstp", "accuracy",
    "condensation_ratio", "num_prototypes", "num_train", "delta_vs_g31",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_all_seeds(seed: int = 42) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_csv_results(csv_path: Path) -> List[dict]:
    """Load existing CSV results for resume support."""
    if not csv_path.is_file():
        return []
    with open(csv_path, "r") as f:
        return list(csv.DictReader(f))


def run_key(phase: int, domain: str, vigilance: float, nstp: int) -> str:
    return f"{phase}|{domain}|{vigilance:.4f}|{nstp}"


def existing_keys(rows: List[dict]) -> set:
    keys = set()
    for r in rows:
        keys.add(run_key(
            int(r["phase"]), r["domain"],
            float(r["vigilance"]), int(r["nstp"]),
        ))
    return keys


def append_csv_row(csv_path: Path, row: dict) -> None:
    write_header = not csv_path.is_file()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(row)


@torch.no_grad()
def eval_run(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    device: torch.device,
    adapter,
    vigilance: float,
    nstp_iters: int = 0,
) -> Tuple[float, int, float]:
    """Run a single FAM evaluation with given vigilance and optional NSTP.

    Returns (accuracy%, num_prototypes, condensation_ratio).
    """
    num_classes = int(train_labels.max().item()) + 1
    nstp = NSTPController(sibling_threshold=0.85, depth_epsilon=0.10) if nstp_iters > 0 else None

    mem = FastAssociativeMemory(
        input_dim=int(train_feats.shape[1]),
        value_dim=num_classes,
        core_entries=50000,
        core_vigilance=vigilance,
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=25,
        inference_temp=0.05,
        use_lfu=True,
        adapter=adapter,
        nstp=nstp,
    ).to(device)

    # Write training data
    x_train = train_feats.to(device)
    y_train = train_labels.to(device)
    for i in range(0, len(x_train), 256):
        mem.learn_local(x_train[i:i + 256], y_train[i:i + 256])

    n_protos = int(mem.core_cam.occupied.sum().item())
    cond = n_protos / max(len(train_feats), 1)

    # Evaluate
    correct = 0
    x_test = test_feats.to(device)
    y_test = test_labels.to(device)
    for i in range(0, len(x_test), 512):
        logits = mem(x_test[i:i + 512])
        correct += (logits.argmax(dim=1) == y_test[i:i + 512]).sum().item()

    acc = 100.0 * correct / max(len(test_labels), 1)

    # Cleanup
    del mem
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return acc, n_protos, cond


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def phase1(
    data_root: Path,
    device: torch.device,
    csv_path: Path,
    done_keys: set,
) -> Dict[Domain4, dict]:
    """Phase 1: Vigilance sweep (no NSTP)."""
    print("\n" + "=" * 60)
    print("PHASE 1 — Vigilance Sweep (no NSTP)")
    print("=" * 60)

    best_per_domain: Dict[Domain4, dict] = {}

    for domain in DOMAINS:
        print(f"\n--- {domain.upper()} ---")
        tr_f, tr_l, te_f, te_l, _ = g30_load_domain_features(domain, data_root)
        adapter = _load_g30_domain_adapter(domain, input_dim=tr_f.shape[1], device=device)

        baseline = G31_BASELINES[domain]
        best = None

        for v in PHASE1_VIGILANCES:
            key = run_key(1, domain, v, 0)
            if key in done_keys:
                # Load from existing CSV
                print(f"[G34][Phase1] {domain} v={v} → (cached, skipping)")
                continue

            set_all_seeds(42)
            acc, n_protos, cond = eval_run(
                tr_f, tr_l, te_f, te_l,
                device=device,
                adapter=adapter,
                vigilance=v,
                nstp_iters=0,
            )
            delta = acc - baseline
            row = {
                "phase": 1, "domain": domain, "vigilance": f"{v:.4f}",
                "nstp": 0, "accuracy": f"{acc:.4f}",
                "condensation_ratio": f"{cond:.6f}",
                "num_prototypes": n_protos, "num_train": len(tr_f),
                "delta_vs_g31": f"{delta:.4f}",
            }
            append_csv_row(csv_path, row)
            done_keys.add(key)
            print(
                f"[G34][Phase1] {domain} v={v} → acc={acc:.2f}% "
                f"ratio={cond:.4f} protos={n_protos}/{len(tr_f)}"
            )

        # Find best from all Phase 1 results (including cached)
        all_rows = load_csv_results(csv_path)
        domain_p1 = [
            r for r in all_rows
            if int(r["phase"]) == 1 and r["domain"] == domain
        ]
        for r in domain_p1:
            acc_r = float(r["accuracy"])
            cond_r = float(r["condensation_ratio"])
            delta_r = float(r["delta_vs_g31"])
            within_tol = delta_r >= -ACCURACY_TOLERANCE_PP
            if within_tol:
                if best is None or cond_r < float(best["condensation_ratio"]):
                    best = r

        if best is None:
            # Fallback: pick highest accuracy regardless
            domain_p1.sort(key=lambda x: float(x["accuracy"]), reverse=True)
            best = domain_p1[0] if domain_p1 else None

        if best:
            best_per_domain[domain] = best
            print(
                f"  >> Best: v={float(best['vigilance']):.2f} → "
                f"ratio={float(best['condensation_ratio']):.4f}, "
                f"acc={float(best['accuracy']):.2f}% "
                f"(Δ={float(best['delta_vs_g31']):+.2f}pp)"
            )

        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_per_domain


def phase2(
    data_root: Path,
    device: torch.device,
    csv_path: Path,
    done_keys: set,
    phase1_best: Dict[Domain4, dict],
) -> Dict[Domain4, dict]:
    """Phase 2: NSTP sweep at best vigilance from Phase 1."""
    print("\n" + "=" * 60)
    print("PHASE 2 — NSTP Sweep at Best Vigilance")
    print("=" * 60)

    best_per_domain: Dict[Domain4, dict] = {}

    for domain in DOMAINS:
        if domain not in phase1_best:
            print(f"\n--- {domain.upper()} --- (skipped, no Phase 1 result)")
            continue

        best_v = float(phase1_best[domain]["vigilance"])
        print(f"\n--- {domain.upper()} (v={best_v}) ---")

        tr_f, tr_l, te_f, te_l, _ = g30_load_domain_features(domain, data_root)
        adapter = _load_g30_domain_adapter(domain, input_dim=tr_f.shape[1], device=device)
        baseline = G31_BASELINES[domain]

        best = None
        # Include the Phase 1 result (nstp=0) as the baseline to beat
        p1_cond = float(phase1_best[domain]["condensation_ratio"])

        for nstp_iter in PHASE2_NSTP_ITERS:
            key = run_key(2, domain, best_v, nstp_iter)
            if key in done_keys:
                print(f"[G34][Phase2] {domain} v={best_v} nstp={nstp_iter} → (cached, skipping)")
                continue

            set_all_seeds(42)
            acc, n_protos, cond = eval_run(
                tr_f, tr_l, te_f, te_l,
                device=device,
                adapter=adapter,
                vigilance=best_v,
                nstp_iters=nstp_iter,
            )
            delta = acc - baseline
            row = {
                "phase": 2, "domain": domain, "vigilance": f"{best_v:.4f}",
                "nstp": nstp_iter, "accuracy": f"{acc:.4f}",
                "condensation_ratio": f"{cond:.6f}",
                "num_prototypes": n_protos, "num_train": len(tr_f),
                "delta_vs_g31": f"{delta:.4f}",
            }
            append_csv_row(csv_path, row)
            done_keys.add(key)
            print(
                f"[G34][Phase2] {domain} v={best_v} nstp={nstp_iter} → "
                f"acc={acc:.2f}% ratio={cond:.4f} protos={n_protos}/{len(tr_f)}"
            )

        # Find best from Phase 2 results
        all_rows = load_csv_results(csv_path)
        domain_p2 = [
            r for r in all_rows
            if int(r["phase"]) == 2 and r["domain"] == domain
        ]
        for r in domain_p2:
            acc_r = float(r["accuracy"])
            cond_r = float(r["condensation_ratio"])
            delta_r = float(r["delta_vs_g31"])
            within_tol = delta_r >= -ACCURACY_TOLERANCE_PP
            if within_tol:
                if best is None or cond_r < float(best["condensation_ratio"]):
                    best = r

        if best is None:
            # NSTP didn't help within tolerance — keep Phase 1 result
            best_per_domain[domain] = phase1_best[domain]
            print(f"  >> NSTP did not improve condensation within tolerance. Keeping Phase 1 result.")
        else:
            best_per_domain[domain] = best
            print(
                f"  >> Best: nstp={best['nstp']} → "
                f"ratio={float(best['condensation_ratio']):.4f}, "
                f"acc={float(best['accuracy']):.2f}% "
                f"(Δ={float(best['delta_vs_g31']):+.2f}pp)"
            )

        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_per_domain


def phase3(
    data_root: Path,
    device: torch.device,
    csv_path: Path,
    done_keys: set,
    phase1_best: Dict[Domain4, dict],
    phase2_best: Dict[Domain4, dict],
) -> Dict[Domain4, dict]:
    """Phase 3: Fine-tuned vigilance around Phase 1 winner with best NSTP."""
    print("\n" + "=" * 60)
    print("PHASE 3 — Fine-Tuned Vigilance + NSTP")
    print("=" * 60)

    best_per_domain: Dict[Domain4, dict] = {}

    for domain in DOMAINS:
        if domain not in phase2_best:
            print(f"\n--- {domain.upper()} --- (skipped)")
            continue

        p2 = phase2_best[domain]
        p1 = phase1_best.get(domain, {})
        p2_nstp = int(p2.get("nstp", 0))
        p2_cond = float(p2["condensation_ratio"])
        p1_cond = float(p1["condensation_ratio"]) if p1 else p2_cond

        # Only run Phase 3 if Phase 2 improved condensation
        if p2_nstp == 0 or p2_cond >= p1_cond:
            print(f"\n--- {domain.upper()} --- (skipped, NSTP did not improve)")
            best_per_domain[domain] = p2
            continue

        p1_v = float(p1["vigilance"])
        print(f"\n--- {domain.upper()} (base v={p1_v}, nstp={p2_nstp}) ---")

        tr_f, tr_l, te_f, te_l, _ = g30_load_domain_features(domain, data_root)
        adapter = _load_g30_domain_adapter(domain, input_dim=tr_f.shape[1], device=device)
        baseline = G31_BASELINES[domain]

        # Fine sweep: ±0.05 around Phase 1 winner in 0.01 steps
        fine_vs = [round(p1_v + offset, 2) for offset in [-0.05, -0.03, -0.01, 0.01, 0.03]]
        # Remove duplicates and out-of-range
        fine_vs = sorted(set(v for v in fine_vs if 0.50 <= v <= 0.99))

        best = p2  # Start with Phase 2 best as baseline

        for v in fine_vs:
            key = run_key(3, domain, v, p2_nstp)
            if key in done_keys:
                print(f"[G34][Phase3] {domain} v={v} nstp={p2_nstp} → (cached, skipping)")
                continue

            set_all_seeds(42)
            acc, n_protos, cond = eval_run(
                tr_f, tr_l, te_f, te_l,
                device=device,
                adapter=adapter,
                vigilance=v,
                nstp_iters=p2_nstp,
            )
            delta = acc - baseline
            row = {
                "phase": 3, "domain": domain, "vigilance": f"{v:.4f}",
                "nstp": p2_nstp, "accuracy": f"{acc:.4f}",
                "condensation_ratio": f"{cond:.6f}",
                "num_prototypes": n_protos, "num_train": len(tr_f),
                "delta_vs_g31": f"{delta:.4f}",
            }
            append_csv_row(csv_path, row)
            done_keys.add(key)
            print(
                f"[G34][Phase3] {domain} v={v} nstp={p2_nstp} → "
                f"acc={acc:.2f}% ratio={cond:.4f} protos={n_protos}/{len(tr_f)}"
            )

        # Find overall best across Phase 3
        all_rows = load_csv_results(csv_path)
        domain_p3 = [
            r for r in all_rows
            if int(r["phase"]) == 3 and r["domain"] == domain
        ]
        for r in domain_p3:
            cond_r = float(r["condensation_ratio"])
            delta_r = float(r["delta_vs_g31"])
            within_tol = delta_r >= -ACCURACY_TOLERANCE_PP
            if within_tol and cond_r < float(best["condensation_ratio"]):
                best = r

        best_per_domain[domain] = best
        print(
            f"  >> Best: v={float(best['vigilance']):.2f} nstp={best['nstp']} → "
            f"ratio={float(best['condensation_ratio']):.4f}, "
            f"acc={float(best['accuracy']):.2f}% "
            f"(Δ={float(best['delta_vs_g31']):+.2f}pp)"
        )

        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_per_domain


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------


def print_final_verdict(
    p1_best: Dict[Domain4, dict],
    p2_best: Dict[Domain4, dict],
    p3_best: Dict[Domain4, dict],
) -> bool:
    print("\n" + "=" * 60)
    print("=== G34 FINAL VERDICT ===")
    print("=" * 60)

    # Phase 1 summary
    print("\nPhase 1 best per domain (vigilance only):")
    for d in DOMAINS:
        if d in p1_best:
            r = p1_best[d]
            print(
                f"  {d:10s}: v={float(r['vigilance']):.2f} → "
                f"ratio={float(r['condensation_ratio']):.4f}, "
                f"acc={float(r['accuracy']):.2f}% "
                f"(Δ={float(r['delta_vs_g31']):+.2f}pp)"
            )
        else:
            print(f"  {d:10s}: NO RESULT")

    # Phase 2 summary
    print("\nPhase 2 best per domain (vigilance + NSTP):")
    for d in DOMAINS:
        if d in p2_best:
            r = p2_best[d]
            print(
                f"  {d:10s}: v={float(r['vigilance']):.2f} nstp={r['nstp']} → "
                f"ratio={float(r['condensation_ratio']):.4f}, "
                f"acc={float(r['accuracy']):.2f}% "
                f"(Δ={float(r['delta_vs_g31']):+.2f}pp)"
            )
        else:
            print(f"  {d:10s}: NO RESULT")

    # Phase 3 summary
    has_p3 = any(
        d in p3_best and int(p3_best[d].get("phase", 0)) == 3
        for d in DOMAINS
    )
    if has_p3:
        print("\nPhase 3 best per domain (fine-tuned):")
        for d in DOMAINS:
            if d in p3_best and int(p3_best[d].get("phase", 0)) == 3:
                r = p3_best[d]
                print(
                    f"  {d:10s}: v={float(r['vigilance']):.2f} nstp={r['nstp']} → "
                    f"ratio={float(r['condensation_ratio']):.4f}, "
                    f"acc={float(r['accuracy']):.2f}% "
                    f"(Δ={float(r['delta_vs_g31']):+.2f}pp)"
                )

    # Determine overall best per domain (from latest phase)
    print("\n--- VERDICT ---")
    overall_pass = False  # Need at least one domain to meet target

    for d in DOMAINS:
        # Pick best result across all phases
        best = None
        for phase_results in [p3_best, p2_best, p1_best]:
            if d in phase_results:
                r = phase_results[d]
                cond = float(r["condensation_ratio"])
                delta = float(r["delta_vs_g31"])
                if delta >= -ACCURACY_TOLERANCE_PP:
                    if best is None or cond < float(best["condensation_ratio"]):
                        best = r

        if best is None:
            print(f"  {d:10s}: NO VALID RESULT")
            continue

        cond = float(best["condensation_ratio"])
        target = TARGET_CONDENSATION[d]
        stretch = STRETCH_CONDENSATION[d]
        met_target = cond < target
        met_stretch = cond < stretch

        if met_target:
            overall_pass = True

        tag = "STRETCH" if met_stretch else ("TARGET" if met_target else "MISSED")
        print(
            f"  {d:10s}: ratio={cond:.4f} target<{target:.2f} "
            f"stretch<{stretch:.2f} → {tag}"
        )

    verdict = "PASS" if overall_pass else "KILL"
    print(f"\nVERDICT: {verdict}")
    return overall_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="G34 — Per-Domain Vigilance + NSTP Condensation Sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--csv", type=Path, default=Path("results/g34_condensation_sweep.csv"),
        help="CSV output path (supports resume).",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = device_from_arg(args.device)

    # Ensure output directory exists
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results for resume
    existing = load_csv_results(args.csv)
    done = existing_keys(existing)
    print(f"Loaded {len(existing)} cached results from {args.csv}")
    print(f"Device: {device}")

    # Phase 1
    p1_best = phase1(args.data_root, device, args.csv, done)

    # Phase 2
    p2_best = phase2(args.data_root, device, args.csv, done, p1_best)

    # Phase 3
    p3_best = phase3(args.data_root, device, args.csv, done, p1_best, p2_best)

    # Final verdict
    passed = print_final_verdict(p1_best, p2_best, p3_best)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
