#!/usr/bin/env python3
"""
benchmarks/g31_retrain_adapters.py — G31: Retrain Adapters with G30 Winners.

Phases
------
A) Retrain 4 production adapters (Dogs/Birds/Cars/Aircraft) with G30 Phase-2
   winning configs, overwrite adapter checkpoints, and verify each retrained
   adapter lands within ±2 pp of the G30 target on its own domain.

B) Re-run a G27-style Top-2 multi-domain evaluation (Birds+Cars) using the G27
   defaults (epochs/batch/lr/margin/vigilance), then compare the average lift
   over no-adapter to the recorded G27 reference (+1.62 pp over no-adapter).

C) Run a 7-domain FAM regression sweep (Dogs, Birds, Cars, Aircraft, Flowers,
   CIFAR-100, ImageNet-R), using the newly retrained adapters where available
   and no adapter elsewhere.

Usage
-----
    PYTHONPATH=. python benchmarks/g31_retrain_adapters.py --data-root data --device auto
"""

from __future__ import annotations

import argparse
import random
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Literal

import numpy as np
import torch

# Allow running from repo root or directly via file path.
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapter import MetricAdapter  # noqa: E402
from benchmarks.g30_adapter_sweep import (  # noqa: E402
    AdapterConfig,
    eval_full_stack,
    load_domain_features as g30_load_domain_features,
    make_adapter as g30_make_adapter,
    train_adapter,
)


Domain4 = Literal["dogs", "birds", "cars", "aircraft"]

G30_PHASE2_TARGETS: Dict[Domain4, float] = {
    "aircraft": 72.85,
    "cars": 89.95,
    "birds": 90.94,
    "dogs": 90.20,
}

PRODUCTION_ADAPTER_PATHS: Dict[Domain4, Path] = {
    "dogs": Path("adapter_trained.pt"),
    "birds": Path("adapter_cub200_birds.pt"),
    "cars": Path("data/stanford_cars/adapter_stanford_cars.pt"),
    "aircraft": Path("adapter_fgvc_aircraft.pt"),
}

# All domains share these base defaults in the prompt.
BASE_CFG = AdapterConfig(
    proj_dim=1024,
    layers=2,
    lr=1e-3,
    epochs=10,
    batch_size=256,
)

DOMAIN_CFGS: Dict[Domain4, AdapterConfig] = {
    "dogs": AdapterConfig(
        **{
            **asdict(BASE_CFG),
            "loss_type": "triplet",
            "mining_strategy": "hard",
            "infonce_temp": 0.05,
            "triplet_margin": 0.3,
            "nonlinearity": "relu",
        }
    ),
    "birds": AdapterConfig(
        **{
            **asdict(BASE_CFG),
            "loss_type": "multisim",
            "mining_strategy": "hard",
            "triplet_margin": 0.3,
            "nonlinearity": "relu",
        }
    ),
    "cars": AdapterConfig(
        **{
            **asdict(BASE_CFG),
            "loss_type": "multisim",
            "mining_strategy": "semi-hard",
            "triplet_margin": 0.3,
            "nonlinearity": "relu",
        }
    ),
    "aircraft": AdapterConfig(
        **{
            **asdict(BASE_CFG),
            "loss_type": "multisim",
            "mining_strategy": "hard",
            "infonce_temp": 0.1,
            "triplet_margin": 0.3,
            "nonlinearity": "relu",
        }
    ),
}

G27_REFERENCE_AVG_TOP2 = 81.92
G27_REFERENCE_DELTA_VS_NO_ADAPTER_PP = 1.62


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


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_g30_domain_adapter(
    domain: Domain4,
    input_dim: int,
    device: torch.device,
) -> MetricAdapter:
    cfg = DOMAIN_CFGS[domain]
    adapter = g30_make_adapter(input_dim=input_dim, cfg=cfg, device=device)
    state = torch.load(
        PRODUCTION_ADAPTER_PATHS[domain], map_location=device, weights_only=True
    )
    adapter.load_state_dict(state)
    adapter.eval()
    return adapter


def _load_feature_cache(path: Path, key: str) -> tuple[torch.Tensor, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    return obj[key].float(), obj["labels"].long()


def load_7domain_features(
    data_root: Path,
) -> Dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Load all 7 domains into CPU tensors."""
    out: Dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    # Reuse G30 loader for the 4 domains retrained in Phase A.
    for d in ("dogs", "birds", "cars", "aircraft"):
        tr_f, tr_l, te_f, te_l, _ = g30_load_domain_features(d, data_root)
        out[d] = (tr_f, tr_l, te_f, te_l)

    # Flowers / CIFAR-100 / ImageNet-R use generic "embeds" caches.
    flowers_tr = data_root / "flowers102" / "feature_cache" / "flowers102_train.pt"
    flowers_te = data_root / "flowers102" / "feature_cache" / "flowers102_test.pt"
    out["flowers"] = (*_load_feature_cache(flowers_tr, "embeds"), *_load_feature_cache(flowers_te, "embeds"))

    cifar_tr = Path("feature_cache_vitb14") / "cifar100_dinov2_train.pt"
    cifar_te = Path("feature_cache_vitb14") / "cifar100_dinov2_test.pt"
    out["cifar100"] = (*_load_feature_cache(cifar_tr, "embeds"), *_load_feature_cache(cifar_te, "embeds"))

    inr_tr = Path("feature_cache_inr_vitl14") / "imagenetr_dinov2_train.pt"
    inr_te = Path("feature_cache_inr_vitl14") / "imagenetr_dinov2_test.pt"
    out["imagenet-r"] = (*_load_feature_cache(inr_tr, "embeds"), *_load_feature_cache(inr_te, "embeds"))

    return out


def run_phase_a(data_root: Path, device: torch.device):
    results: dict[str, dict[str, float | bool | int]] = {}
    all_pass = True

    for domain in ("dogs", "birds", "cars", "aircraft"):
        cfg = DOMAIN_CFGS[domain]
        target = G30_PHASE2_TARGETS[domain]

        set_all_seeds(42)
        tr_f, tr_l, te_f, te_l, n_classes = g30_load_domain_features(domain, data_root)

        adapter, train_time_s = train_adapter(tr_f, tr_l, cfg, device)
        adapter.eval()
        acc, n_protos, cond = eval_full_stack(
            tr_f, tr_l, te_f, te_l, n_classes, adapter, device
        )

        out_path = PRODUCTION_ADAPTER_PATHS[domain]
        _ensure_parent(out_path)
        torch.save(adapter.state_dict(), out_path)

        delta = acc - target
        ok = abs(delta) <= 2.0
        all_pass = all_pass and ok
        tag = "PASS" if ok else "FAIL"
        print(
            f"[G31][PhaseA] {domain} → acc={acc:.2f}% target={target:.2f}% "
            f"delta={delta:+.2f}pp {tag}",
            flush=True,
        )

        results[domain] = {
            "acc": acc,
            "target": target,
            "delta_pp": delta,
            "pass": ok,
            "train_time_s": train_time_s,
            "n_prototypes": n_protos,
            "condensation_ratio": cond,
        }

        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results, all_pass


def run_phase_b(device: torch.device):
    """Run the G27 script (birds+cars) and parse per-dataset/average Multi-Top2."""
    set_all_seeds(42)
    print(
        "[G31][PhaseB] Running benchmarks/g27_multi_adapter_top2.py (birds cars, defaults)",
        flush=True,
    )

    cmd = [
        sys.executable,
        "benchmarks/g27_multi_adapter_top2.py",
        "--domains",
        "birds",
        "cars",
        "--device",
        str(device),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
        check=False,
    )
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")

    row_re = re.compile(
        r"^\s+(?P<label>Birds \(in\)|Cars \(in\)|CIFAR-100 \(cross\)|ImageNet-R \(cross\))\s+"
        r"(?P<best>\d+\.\d+)%\s+(?P<multi>\d+\.\d+)%\s+(?P<delta>[+-]?\s*\d+\.\d+)\s*$"
    )
    avg_re = re.compile(
        r"^\s+AVERAGE\s+(?P<best>\d+\.\d+)%\s+(?P<multi>\d+\.\d+)%\s+(?P<delta>[+-]?\s*\d+\.\d+)\s*$"
    )

    rows: dict[str, dict[str, float]] = {}
    avg_multi = None
    avg_best_single = None
    avg_delta_vs_best_single = None
    for line in proc.stdout.splitlines():
        m = row_re.match(line)
        if m:
            rows[m.group("label")] = {
                "best_single": float(m.group("best")),
                "multi_top2": float(m.group("multi")),
                "delta_pp": float(m.group("delta").replace(" ", "")),
            }
            continue
        m = avg_re.match(line)
        if m:
            avg_best_single = float(m.group("best"))
            avg_multi = float(m.group("multi"))
            avg_delta_vs_best_single = float(m.group("delta").replace(" ", ""))

    required_labels = (
        "Birds (in)",
        "Cars (in)",
        "CIFAR-100 (cross)",
        "ImageNet-R (cross)",
    )
    missing = [k for k in required_labels if k not in rows]
    if missing or avg_multi is None or avg_best_single is None or avg_delta_vs_best_single is None:
        raise RuntimeError(
            "Failed to parse G27 output for Phase B. "
            f"missing_rows={missing}, parsed_avg={avg_multi}"
        )

    for label in required_labels:
        r = rows[label]
        print(
            f"[G31][PhaseB] {label} → acc={r['multi_top2']:.2f}% "
            f"best_single={r['best_single']:.2f}% delta={r['delta_pp']:+.2f}pp",
            flush=True,
        )

    delta_vs_g27_abs = avg_multi - G27_REFERENCE_AVG_TOP2
    kill = delta_vs_g27_abs < -2.0
    print(
        f"[G31][PhaseB] avg_multi={avg_multi:.2f}% vs G27 ref={G27_REFERENCE_AVG_TOP2:.2f}% "
        f"(Δ={delta_vs_g27_abs:+.2f}pp; G27 ref was {G27_REFERENCE_DELTA_VS_NO_ADAPTER_PP:+.2f}pp "
        f"over no-adapter) {'KILL' if kill else 'PASS'}",
        flush=True,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "rows": rows,
        "avg_multi": avg_multi,
        "avg_best_single": avg_best_single,
        "avg_delta_vs_best_single": avg_delta_vs_best_single,
        "g27_reference_avg": G27_REFERENCE_AVG_TOP2,
        "g27_reference_delta_vs_no_adapter": G27_REFERENCE_DELTA_VS_NO_ADAPTER_PP,
        "delta_vs_g27_reference_avg_pp": delta_vs_g27_abs,
        "g27_script_exit_code": proc.returncode,
        "kill": kill,
    }


def run_phase_c(data_root: Path, device: torch.device):
    """Uniform 7-domain FAM regression check using optional adapters."""
    set_all_seeds(42)
    feats = load_7domain_features(data_root)
    results: dict[str, float] = {}

    for domain in ("dogs", "birds", "cars", "aircraft", "flowers", "cifar100", "imagenet-r"):
        tr_e, tr_l, te_e, te_l = feats[domain]
        adapter = None
        if domain in PRODUCTION_ADAPTER_PATHS:
            adapter = _load_g30_domain_adapter(domain, input_dim=tr_e.shape[1], device=device)

        n_classes = int(tr_l.max().item()) + 1
        acc, _, _ = eval_full_stack(
            tr_e, tr_l, te_e, te_l, n_classes, adapter, device
        )
        print(f"[G31][PhaseC] {domain} → acc={acc:.2f}%", flush=True)
        results[domain] = acc

        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def _print_final_verdict(phase_a, phase_a_ok: bool, phase_b, phase_c) -> bool:
    overall_kill = (not phase_a_ok) or bool(phase_b["kill"])
    verdict = "KILL" if overall_kill else "PASS"

    print("\n=== G31 FINAL VERDICT ===")
    print("Phase A: [per-domain accuracy vs G30 Phase 2 targets]")
    for domain in ("dogs", "birds", "cars", "aircraft"):
        row = phase_a[domain]
        print(
            f"  - {domain}: {row['acc']:.2f}% vs {row['target']:.2f}% "
            f"({row['delta_pp']:+.2f}pp) {'PASS' if row['pass'] else 'FAIL'}"
        )

    print("Phase B: [Top-2 multi-domain avg vs G27 reference]")
    print(
        f"  - avg_multi={phase_b['avg_multi']:.2f}% "
        f"avg_best_single={phase_b['avg_best_single']:.2f}% "
        f"delta_vs_best_single={phase_b['avg_delta_vs_best_single']:+.2f}pp "
        f"vs G27 ref avg {phase_b['g27_reference_avg']:.2f}% "
        f"(Δ={phase_b['delta_vs_g27_reference_avg_pp']:+.2f}pp; "
        f"ref lift vs no-adapter={phase_b['g27_reference_delta_vs_no_adapter']:+.2f}pp)"
    )

    print("Phase C: [7-domain regression check]")
    for domain in ("dogs", "birds", "cars", "aircraft", "flowers", "cifar100", "imagenet-r"):
        print(f"  - {domain}: {phase_c[domain]:.2f}%")

    print(f"VERDICT: {verdict}")
    return not overall_kill


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="G31 — Retrain G30-winning adapters and re-run multi-domain evaluations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default="data", type=Path)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = device_from_arg(args.device)

    print("G31 — Retrain Adapters with G30-Optimal Configs + Multi-Domain Re-eval")
    print(f"Device   : {device}")
    print(f"Data root : {args.data_root}")

    phase_a, phase_a_ok = run_phase_a(args.data_root, device)
    phase_b = run_phase_b(device)
    phase_c = run_phase_c(args.data_root, device)

    passed = _print_final_verdict(phase_a, phase_a_ok, phase_b, phase_c)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
