#!/usr/bin/env python3
"""
benchmarks/g33_regression_suite.py — G33: Prior Gauntlet Regression Suite.

Re-evaluates selected prior gauntlets using the current codebase and G31
production adapters, then applies the G33 directive's kill/pass rules.

Run:
    PYTHONPATH=. python benchmarks/g33_regression_suite.py --data-root data --device auto
"""

from __future__ import annotations

import argparse
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fast_associative_memory import FastAssociativeMemory  # noqa: E402
from nstp import NSTPController  # noqa: E402
from benchmarks.gauntlet_19_cross_domain import load_features  # noqa: E402
from benchmarks.g30_adapter_sweep import (  # noqa: E402
    eval_full_stack as g30_eval_full_stack,
    load_domain_features as g30_load_domain_features,
)
from benchmarks.g31_retrain_adapters import _load_g30_domain_adapter  # noqa: E402


G24_KILL_THRESHOLD_ACC = 75.0  # Per G33 directive.
G25_PASS_FLOOR_ACC = 99.50     # Per G33 directive.

G28_G32_CITED_AVG = 75.83
G28_G32_CITED_AIRCRAFT_REGRESSION_PP = -40.87


@dataclass
class GauntletResult:
    code: str
    status: str
    summary: str
    details: dict[str, Any]


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


@torch.no_grad()
def eval_fam_stack(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    device: torch.device,
    adapter=None,
    vigilance: float = 0.80,
    use_nstp: bool = True,
    core_entries: int = 50000,
) -> tuple[float, int, float]:
    """Evaluate FAM (+ optional adapter/NSTP) on a single dataset split."""
    num_classes = int(train_labels.max().item()) + 1
    nstp = NSTPController(sibling_threshold=0.85, depth_epsilon=0.10) if use_nstp else None

    mem = FastAssociativeMemory(
        input_dim=int(train_feats.shape[1]),
        value_dim=num_classes,
        core_entries=core_entries,
        core_vigilance=vigilance,
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=25,
        inference_temp=0.05,
        use_lfu=True,
        adapter=adapter,
        nstp=nstp,
    ).to(device)

    x_train = train_feats.to(device)
    y_train = train_labels.to(device)
    for i in range(0, len(x_train), 256):
        mem.learn_local(x_train[i:i + 256], y_train[i:i + 256])

    n_protos = int(mem.core_cam.occupied.sum().item())
    cond = n_protos / max(len(train_feats), 1)

    correct = 0
    x_test = test_feats.to(device)
    y_test = test_labels.to(device)
    for i in range(0, len(x_test), 512):
        logits = mem(x_test[i:i + 512])
        correct += (logits.argmax(dim=1) == y_test[i:i + 512]).sum().item()

    acc = 100.0 * correct / max(len(test_labels), 1)
    return acc, n_protos, cond


def run_g11(data_root: Path, device: torch.device) -> GauntletResult:
    set_all_seeds(42)
    tr_f, tr_l, te_f, te_l, _ = g30_load_domain_features("dogs", data_root)
    adapter = _load_g30_domain_adapter("dogs", input_dim=tr_f.shape[1], device=device)
    acc, n_protos, cond = eval_fam_stack(
        tr_f, tr_l, te_f, te_l,
        device=device,
        adapter=adapter,
        vigilance=0.80,
        use_nstp=True,
    )
    passed = cond < 0.50
    status = "PASS (still)" if passed else "REGRESSED"
    return GauntletResult(
        code="G11",
        status=status,
        summary=f"{status} — ratio={cond:.2f}, acc={acc:.2f}%",
        details={"acc": acc, "condensation_ratio": cond, "n_prototypes": n_protos},
    )


def run_g22_or_g23(code: str, domain: str, data_root: Path, device: torch.device) -> GauntletResult:
    set_all_seeds(42)
    tr_f, tr_l, te_f, te_l, n_classes = g30_load_domain_features(domain, data_root)
    adapter = _load_g30_domain_adapter(domain, input_dim=tr_f.shape[1], device=device)

    baseline_acc, baseline_n_protos, baseline_cond = g30_eval_full_stack(
        tr_f, tr_l, te_f, te_l, n_classes, None, device
    )
    adapted_acc, n_protos, cond = g30_eval_full_stack(
        tr_f, tr_l, te_f, te_l, n_classes, adapter, device
    )
    passed = adapted_acc > baseline_acc
    status = "PASS (still)" if passed else "REGRESSED"
    return GauntletResult(
        code=code,
        status=status,
        summary=f"{status} — acc={adapted_acc:.2f}% vs baseline {baseline_acc:.2f}%",
        details={
            "acc": adapted_acc,
            "baseline_acc": baseline_acc,
            "n_prototypes": n_protos,
            "condensation_ratio": cond,
            "baseline_n_prototypes": baseline_n_protos,
            "baseline_condensation_ratio": baseline_cond,
        },
    )


def run_g24(data_root: Path, device: torch.device) -> GauntletResult:
    set_all_seeds(42)
    tr_f, tr_l, te_f, te_l, n_classes = g30_load_domain_features("aircraft", data_root)
    adapter = _load_g30_domain_adapter("aircraft", input_dim=tr_f.shape[1], device=device)

    baseline_acc, baseline_n_protos, baseline_cond = g30_eval_full_stack(
        tr_f, tr_l, te_f, te_l, n_classes, None, device
    )
    adapted_acc, n_protos, cond = g30_eval_full_stack(
        tr_f, tr_l, te_f, te_l, n_classes, adapter, device
    )
    passed = adapted_acc >= G24_KILL_THRESHOLD_ACC
    status = "PASS (new)" if passed else "KILL (still)"
    return GauntletResult(
        code="G24",
        status=status,
        summary=f"{status} — acc={adapted_acc:.2f}% vs kill threshold {G24_KILL_THRESHOLD_ACC:.2f}%",
        details={
            "acc": adapted_acc,
            "baseline_acc": baseline_acc,
            "threshold_acc": G24_KILL_THRESHOLD_ACC,
            "n_prototypes": n_protos,
            "condensation_ratio": cond,
            "baseline_n_prototypes": baseline_n_protos,
            "baseline_condensation_ratio": baseline_cond,
        },
    )


def run_g25(device: torch.device) -> GauntletResult:
    set_all_seeds(42)
    tr_f, tr_l, te_f, te_l = load_features("data/flowers102/feature_cache", device)
    acc, n_protos, cond = eval_fam_stack(
        tr_f, tr_l, te_f, te_l,
        device=device,
        adapter=None,
        vigilance=0.80,
        use_nstp=True,
    )
    passed = acc >= G25_PASS_FLOOR_ACC
    status = "PASS (still)" if passed else "REGRESSED"
    return GauntletResult(
        code="G25",
        status=status,
        summary=f"{status} — acc={acc:.2f}%",
        details={"acc": acc, "n_prototypes": n_protos, "condensation_ratio": cond},
    )


def run_g27(repo_root: Path, device: torch.device) -> GauntletResult:
    set_all_seeds(42)
    cmd = [
        sys.executable,
        "benchmarks/g27_multi_adapter_top2.py",
        "--domains",
        "birds",
        "cars",
        "--device",
        str(device),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError(
            f"G27 subprocess failed unexpectedly (exit={proc.returncode}).\nSTDERR:\n{proc.stderr}"
        )

    avg_re = re.compile(
        r"^\s+AVERAGE\s+(?P<best>\d+\.\d+)%\s+(?P<multi>\d+\.\d+)%\s+(?P<delta>[+-]?\s*\d+\.\d+)\s*$"
    )
    avg_best_single = None
    avg_multi = None
    avg_delta = None
    for line in proc.stdout.splitlines():
        m = avg_re.match(line)
        if m:
            avg_best_single = float(m.group("best"))
            avg_multi = float(m.group("multi"))
            avg_delta = float(m.group("delta").replace(" ", ""))
            break

    if avg_best_single is None or avg_multi is None:
        raise RuntimeError(
            "Failed to parse G27 output average row.\n"
            f"Exit={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    passed = avg_multi >= avg_best_single
    status = "PASS (still)" if passed else "REGRESSED"
    return GauntletResult(
        code="G27",
        status=status,
        summary=f"{status} — avg={avg_multi:.2f}% vs best-single {avg_best_single:.2f}%",
        details={
            "avg_multi": avg_multi,
            "avg_best_single": avg_best_single,
            "avg_delta_pp": avg_delta,
            "g27_exit_code": proc.returncode,
        },
    )


def run_g28_cited() -> GauntletResult:
    return GauntletResult(
        code="G28",
        status="KILL (still)",
        summary=(
            "KILL (still) — cited from G32 "
            f"({G28_G32_CITED_AVG:.2f}% avg, Aircraft {G28_G32_CITED_AIRCRAFT_REGRESSION_PP:+.2f}pp)"
        ),
        details={
            "avg": G28_G32_CITED_AVG,
            "aircraft_regression_pp": G28_G32_CITED_AIRCRAFT_REGRESSION_PP,
            "source": "G32 cited result (no re-run)",
        },
    )


def print_report(results: list[GauntletResult]) -> bool:
    by_code = {r.code: r for r in results}
    regression = any(by_code[g].status == "REGRESSED" for g in ("G11", "G22", "G23", "G25", "G27"))
    verdict = "KILL (regression detected)" if regression else "PASS (no regressions)"

    print("=== G33 REGRESSION SUITE ===")
    print(f"G11: {by_code['G11'].summary}")
    print(f"G22: {by_code['G22'].summary}")
    print(f"G23: {by_code['G23'].summary}")
    print(f"G24: {by_code['G24'].summary}")
    print(f"G25: {by_code['G25'].summary}")
    print(f"G27: {by_code['G27'].summary}")
    print(f"G28: {by_code['G28'].summary}")
    print(f"VERDICT: {verdict}")
    return not regression


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="G33 — Prior Gauntlet Regression Suite with G31 adapters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    device = device_from_arg(args.device)

    results: list[GauntletResult] = []
    results.append(run_g11(args.data_root, device))
    results.append(run_g22_or_g23("G22", "birds", args.data_root, device))
    results.append(run_g22_or_g23("G23", "cars", args.data_root, device))
    results.append(run_g24(args.data_root, device))
    results.append(run_g25(device))
    results.append(run_g27(repo_root, device))
    results.append(run_g28_cited())

    ok = print_report(results)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
