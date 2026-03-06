#!/usr/bin/env python3
"""
BCL-EXP-3: Log-Structured Merge EWC Benchmark

Compares StandardEWC vs LogStructuredEWC across:
  - Split-MNIST (5 tasks)
  - Split-CIFAR-10 (5 tasks)
  - Permuted-MNIST (5, 10, 20, 50 tasks)

Sweep: compaction_interval x merge_weights (4 x 3 = 12 configs)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bcl_gem'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import json
import traceback

from benchmarks import set_seed, get_split_mnist, get_split_cifar10, get_permuted_mnist, FastTensorDataLoader
from model import MLP, SmallCNN
from ewc import StandardEWC, LogStructuredEWC, compute_fisher_diagonal


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 43, 44]
COMPACTION_INTERVALS = [2, 5, 10, 20]
MERGE_STRATEGIES = ['uniform', 'recency', 'magnitude']
EWC_LAMBDA = 400.0
FISHER_SAMPLES = 2000
TRAIN_EPOCHS = 1
LR = 0.01


def train_ewc(model, ewc_module, tasks_train, tasks_test, n_tasks,
               num_classes, lr=LR, epochs=TRAIN_EPOCHS, ewc_lambda=EWC_LAMBDA,
               fisher_samples=FISHER_SAMPLES, device=DEVICE):
    """
    Train model with EWC regularization.
    Returns: acc_matrix, per-task consolidation times, per-task fisher dicts, peak_mem per task
    """
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    acc_matrix = np.zeros((n_tasks, n_tasks))
    consol_times = []
    fisher_snapshots = []  # for approximation quality comparison
    peak_mem_entries = []

    for t in range(n_tasks):
        model.train()
        loader = tasks_train[t]

        for ep in range(epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)

                # EWC penalty
                ewc_pen = ewc_module.penalty(lambd=ewc_lambda)
                if isinstance(ewc_pen, torch.Tensor):
                    loss = loss + ewc_pen
                elif ewc_pen != 0.0:
                    loss = loss + ewc_pen

                loss.backward()
                optimizer.step()

        # Compute Fisher for this task
        fisher = compute_fisher_diagonal(model, tasks_train[t], device, n_samples=fisher_samples)
        fisher_snapshots.append({n: f.clone() for n, f in fisher.items()})

        # Record consolidation timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ewc_module.after_task(fisher)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        consol_times.append(t1 - t0)

        # Peak memory (number of Fisher matrices held)
        if isinstance(ewc_module, LogStructuredEWC):
            peak_mem_entries.append(ewc_module.peak_memory_entries())
        else:
            peak_mem_entries.append(1)  # standard always holds 1 consolidated

        # Evaluate on all tasks seen so far
        model.eval()
        with torch.no_grad():
            for eval_t in range(t + 1):
                correct, total = 0, 0
                for x, y in tasks_test[eval_t]:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    pred = out.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += x.size(0)
                acc_matrix[t, eval_t] = correct / total

    return acc_matrix, consol_times, fisher_snapshots, peak_mem_entries


def compute_bwt(acc_matrix):
    T = acc_matrix.shape[0]
    if T <= 1:
        return 0.0
    return np.mean([acc_matrix[T-1, i] - acc_matrix[i, i] for i in range(T-1)])


def compute_avg_acc(acc_matrix):
    T = acc_matrix.shape[0]
    return np.mean(acc_matrix[T-1, :])


def compute_approx_quality(std_fishers, log_fishers, compaction_interval):
    """
    Compute ||F_log - F_std|| / ||F_std|| at compaction boundaries.
    Both lists are cumulative sum snapshots after each task.
    We check at task indices that are multiples of compaction_interval.
    """
    qualities = []
    # Build cumulative Fishers
    std_cumul = None
    log_cumul_uniform = None  # uniform weighting = same as standard sum

    for t in range(len(std_fishers)):
        if std_cumul is None:
            std_cumul = {n: f.clone() for n, f in std_fishers[t].items()}
        else:
            for n in std_cumul:
                std_cumul[n] = std_cumul[n] + std_fishers[t][n]

        if log_cumul_uniform is None:
            log_cumul_uniform = {n: f.clone() for n, f in log_fishers[t].items()}
        else:
            for n in log_cumul_uniform:
                log_cumul_uniform[n] = log_cumul_uniform[n] + log_fishers[t][n]

        # Check at compaction boundaries
        if (t + 1) % compaction_interval == 0 or t == len(std_fishers) - 1:
            diff_norm = 0.0
            std_norm = 0.0
            for n in std_cumul:
                diff_norm += (std_cumul[n] - log_cumul_uniform[n]).norm().item() ** 2
                std_norm += std_cumul[n].norm().item() ** 2
            diff_norm = diff_norm ** 0.5
            std_norm = std_norm ** 0.5
            if std_norm > 0:
                qualities.append(diff_norm / std_norm)
            else:
                qualities.append(0.0)

    return qualities


def get_benchmark(name, n_tasks=5, seed=42):
    """Get benchmark data and model factory."""
    set_seed(seed)
    if name == "split_mnist":
        tasks_train, tasks_test, nc = get_split_mnist(batch_size=64)
        def make_model():
            return MLP(28*28, 256, nc).to(DEVICE)
        return tasks_train, tasks_test, nc, 5, make_model
    elif name == "split_cifar10":
        tasks_train, tasks_test, nc = get_split_cifar10(batch_size=64)
        def make_model():
            return SmallCNN(nc).to(DEVICE)
        return tasks_train, tasks_test, nc, 5, make_model
    elif name == "permuted_mnist":
        tasks_train, tasks_test, nc = get_permuted_mnist(batch_size=64, n_tasks=n_tasks, seed=seed)
        def make_model():
            return MLP(28*28, 256, nc).to(DEVICE)
        return tasks_train, tasks_test, nc, n_tasks, make_model
    else:
        raise ValueError(f"Unknown benchmark: {name}")


def run_single_config(benchmark_name, method, compaction_interval, merge_strategy, seed, n_tasks=None):
    """Run a single (benchmark, method, C, strategy, seed) configuration."""
    nt = n_tasks if n_tasks else 5
    tasks_train, tasks_test, nc, nt_actual, make_model = get_benchmark(
        benchmark_name, n_tasks=nt, seed=seed
    )

    set_seed(seed)
    model = make_model()

    if method == "standard":
        ewc = StandardEWC(model)
    else:
        ewc = LogStructuredEWC(model, compaction_interval=compaction_interval, merge_weights=merge_strategy)

    acc_matrix, consol_times, fisher_snapshots, peak_mem = train_ewc(
        model, ewc, tasks_train, tasks_test, nt_actual, nc
    )

    bwt = compute_bwt(acc_matrix)
    avg_acc = compute_avg_acc(acc_matrix)
    avg_consol_time = np.mean(consol_times)
    max_peak_mem = max(peak_mem)

    return {
        "bwt": bwt,
        "avg_acc": avg_acc,
        "avg_consol_time": avg_consol_time,
        "total_consol_time": sum(consol_times),
        "consol_times": consol_times,
        "max_peak_mem": max_peak_mem,
        "acc_matrix": acc_matrix,
        "fisher_snapshots": fisher_snapshots,
    }


def format_table_row(cells, widths):
    return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"


def check_kill_conditions(std_result, log_result, config_str):
    """
    Check kill conditions. Returns (killed, message) tuple.
    """
    acc_delta = log_result["avg_acc"] - std_result["avg_acc"]
    time_ratio = std_result["avg_consol_time"] / max(log_result["avg_consol_time"], 1e-12)

    # KILL 1: >2pp degradation AND no consolidation cost reduction
    if acc_delta < -0.02 and time_ratio <= 1.0:
        return True, (f"KILL-1 triggered [{config_str}]: "
                      f"Acc delta={acc_delta*100:.2f}pp, time ratio={time_ratio:.2f} "
                      f"(log-EWC is strictly worse)")

    # KILL 2: consolidation reduced but >5pp accuracy degradation
    if acc_delta < -0.05 and time_ratio > 1.0:
        return True, (f"KILL-2 triggered [{config_str}]: "
                      f"Acc delta={acc_delta*100:.2f}pp, time ratio={time_ratio:.2f} "
                      f"(approximation too lossy)")

    return False, ""


def main():
    print(f"BCL-EXP-3: Log-Structured EWC Benchmark")
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print(f"EWC lambda: {EWC_LAMBDA}, Fisher samples: {FISHER_SAMPLES}")
    print(f"LR: {LR}, Epochs: {TRAIN_EPOCHS}")
    print("=" * 80)

    all_results = {}
    kill_messages = []

    # ========== PART 3: Three benchmarks ==========
    for bench_name in ["split_mnist", "split_cifar10", "permuted_mnist"]:
        print(f"\n{'='*80}")
        print(f"BENCHMARK: {bench_name}")
        print(f"{'='*80}")

        bench_results = {}

        # Run standard EWC baseline across seeds
        print(f"\n--- Standard EWC ---")
        std_runs = []
        for seed in SEEDS:
            r = run_single_config(bench_name, "standard", None, None, seed)
            std_runs.append(r)
            print(f"  seed={seed}: avg_acc={r['avg_acc']*100:.2f}%, bwt={r['bwt']*100:.2f}%, "
                  f"consol_time/task={r['avg_consol_time']*1000:.4f}ms")

        std_mean = {
            "bwt": np.mean([r["bwt"] for r in std_runs]),
            "bwt_std": np.std([r["bwt"] for r in std_runs]),
            "avg_acc": np.mean([r["avg_acc"] for r in std_runs]),
            "avg_acc_std": np.std([r["avg_acc"] for r in std_runs]),
            "avg_consol_time": np.mean([r["avg_consol_time"] for r in std_runs]),
            "avg_consol_time_std": np.std([r["avg_consol_time"] for r in std_runs]),
            "max_peak_mem": 1,
        }
        bench_results["standard"] = std_mean

        # Run log-structured EWC for each (C, strategy) combo
        for strategy in MERGE_STRATEGIES:
            for C in COMPACTION_INTERVALS:
                key = f"log_C{C}_{strategy}"
                print(f"\n--- Log-EWC C={C}, strategy={strategy} ---")
                log_runs = []
                for seed in SEEDS:
                    r = run_single_config(bench_name, "log", C, strategy, seed)
                    log_runs.append(r)
                    print(f"  seed={seed}: avg_acc={r['avg_acc']*100:.2f}%, bwt={r['bwt']*100:.2f}%, "
                          f"consol_time/task={r['avg_consol_time']*1000:.4f}ms, peak_mem={r['max_peak_mem']}")

                log_mean = {
                    "bwt": np.mean([r["bwt"] for r in log_runs]),
                    "bwt_std": np.std([r["bwt"] for r in log_runs]),
                    "avg_acc": np.mean([r["avg_acc"] for r in log_runs]),
                    "avg_acc_std": np.std([r["avg_acc"] for r in log_runs]),
                    "avg_consol_time": np.mean([r["avg_consol_time"] for r in log_runs]),
                    "avg_consol_time_std": np.std([r["avg_consol_time"] for r in log_runs]),
                    "max_peak_mem": np.mean([r["max_peak_mem"] for r in log_runs]),
                }
                bench_results[key] = log_mean

                # Kill condition check
                killed, msg = check_kill_conditions(
                    {"avg_acc": std_mean["avg_acc"], "avg_consol_time": std_mean["avg_consol_time"]},
                    {"avg_acc": log_mean["avg_acc"], "avg_consol_time": log_mean["avg_consol_time"]},
                    f"{bench_name}/C={C}/{strategy}"
                )
                if killed:
                    kill_messages.append(msg)
                    print(f"  *** {msg}")

                # Approximation quality (use seed=42 runs)
                if len(std_runs) > 0 and len(log_runs) > 0:
                    quals = compute_approx_quality(
                        std_runs[0]["fisher_snapshots"],
                        log_runs[0]["fisher_snapshots"],
                        C
                    )
                    log_mean["approx_quality"] = quals
                    print(f"  Approx quality at compaction boundaries: "
                          f"{[f'{q:.6f}' for q in quals]}")

        all_results[bench_name] = bench_results

        # Print comparison table for this benchmark
        print(f"\n\nComparison Results ({bench_name}):")
        print(f"| {'Method':<30} | {'BWT':>12} | {'Avg Acc':>12} | {'Consol. Time/Task':>20} | {'Peak Mem':>10} |")
        print(f"|{'-'*32}|{'-'*14}|{'-'*14}|{'-'*22}|{'-'*12}|")

        sm = bench_results["standard"]
        print(f"| {'Standard EWC':<30} | {sm['bwt']*100:>+6.2f}%{chr(177)}{sm['bwt_std']*100:.2f} "
              f"| {sm['avg_acc']*100:>6.2f}%{chr(177)}{sm['avg_acc_std']*100:.2f} "
              f"| {sm['avg_consol_time']*1000:>12.4f}ms       "
              f"| {sm['max_peak_mem']:>10} |")

        for strategy in MERGE_STRATEGIES:
            for C in COMPACTION_INTERVALS:
                key = f"log_C{C}_{strategy}"
                lm = bench_results[key]
                label = f"Log-EWC C={C} {strategy[:4]}"
                print(f"| {label:<30} | {lm['bwt']*100:>+6.2f}%{chr(177)}{lm['bwt_std']*100:.2f} "
                      f"| {lm['avg_acc']*100:>6.2f}%{chr(177)}{lm['avg_acc_std']*100:.2f} "
                      f"| {lm['avg_consol_time']*1000:>12.4f}ms       "
                      f"| {lm['max_peak_mem']:>10.1f} |")

    # ========== PART 4: Scaling experiment ==========
    print(f"\n\n{'='*80}")
    print("SCALING EXPERIMENT: Permuted-MNIST")
    print(f"{'='*80}")

    scaling_Ns = [5, 10, 20, 50]
    # Use best config from permuted_mnist results: uniform C=5 as default
    best_strategy = "uniform"
    best_C = 5

    scaling_results = {}
    for N in scaling_Ns:
        print(f"\n--- N={N} tasks ---")
        std_times = []
        log_times = []
        std_accs = []
        log_accs = []

        for seed in SEEDS:
            sr = run_single_config("permuted_mnist", "standard", None, None, seed, n_tasks=N)
            lr = run_single_config("permuted_mnist", "log", best_C, best_strategy, seed, n_tasks=N)
            std_times.append(sr["avg_consol_time"])
            log_times.append(lr["avg_consol_time"])
            std_accs.append(sr["avg_acc"])
            log_accs.append(lr["avg_acc"])

        scaling_results[N] = {
            "std_time_mean": np.mean(std_times),
            "std_time_std": np.std(std_times),
            "log_time_mean": np.mean(log_times),
            "log_time_std": np.std(log_times),
            "ratio_mean": np.mean(std_times) / max(np.mean(log_times), 1e-12),
            "std_acc_mean": np.mean(std_accs),
            "log_acc_mean": np.mean(log_accs),
            "acc_delta": np.mean(log_accs) - np.mean(std_accs),
        }

        sr = scaling_results[N]
        print(f"  Std EWC: time/task={sr['std_time_mean']*1000:.4f}ms, acc={sr['std_acc_mean']*100:.2f}%")
        print(f"  Log EWC: time/task={sr['log_time_mean']*1000:.4f}ms, acc={sr['log_acc_mean']*100:.2f}%")
        print(f"  Ratio: {sr['ratio_mean']:.3f}, Acc delta: {sr['acc_delta']*100:+.2f}pp")

    # Scaling table
    print(f"\n\nScaling (Permuted-MNIST, C={best_C}, {best_strategy}):")
    print(f"| {'N':>4} | {'Std EWC Time/Task':>18} | {'Log-EWC Time/Task':>18} | {'Ratio':>8} | {'Acc Delta':>10} |")
    print(f"|{'-'*6}|{'-'*20}|{'-'*20}|{'-'*10}|{'-'*12}|")
    for N in scaling_Ns:
        sr = scaling_results[N]
        print(f"| {N:>4} | {sr['std_time_mean']*1000:>12.4f}ms     "
              f"| {sr['log_time_mean']*1000:>12.4f}ms     "
              f"| {sr['ratio_mean']:>8.3f} "
              f"| {sr['acc_delta']*100:>+8.2f}pp |")

    # ========== KILL CONDITION SUMMARY ==========
    print(f"\n\n{'='*80}")
    print("KILL CONDITION CHECK")
    print(f"{'='*80}")
    if kill_messages:
        for msg in kill_messages:
            print(f"  TRIGGERED: {msg}")
        print("\nEXPERIMENT HALTED DUE TO KILL CONDITIONS.")
    else:
        print("  KILL-1 (strict worse): NOT triggered")
        print("  KILL-2 (too lossy): NOT triggered")

    # ========== SAVE RESULTS ==========
    # Serialize what we can
    serializable = {}
    for bench, br in all_results.items():
        serializable[bench] = {}
        for key, val in br.items():
            sv = {}
            for k, v in val.items():
                if isinstance(v, (np.floating, np.integer)):
                    sv[k] = float(v)
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], float):
                    sv[k] = v
                elif isinstance(v, (float, int)):
                    sv[k] = v
            serializable[bench][key] = sv

    serializable["scaling"] = {}
    for N, sr in scaling_results.items():
        serializable["scaling"][str(N)] = {k: float(v) for k, v in sr.items()}

    with open(os.path.join(os.path.dirname(__file__), "bcl_exp3_results.json"), "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to bcl_ewc/bcl_exp3_results.json")

    # ========== VERDICT ==========
    print(f"\n\n{'='*80}")
    print("SUMMARY VERDICT")
    print(f"{'='*80}")

    # Determine if scaling ratio grows
    ratios = [scaling_results[N]["ratio_mean"] for N in scaling_Ns]
    ratio_trend = ratios[-1] - ratios[0]

    print(f"\nAmortization ratios across N: {[f'{r:.3f}' for r in ratios]}")
    print(f"Ratio trend (N=50 - N=5): {ratio_trend:+.3f}")

    # Find best config across permuted_mnist
    if "permuted_mnist" in all_results:
        pm = all_results["permuted_mnist"]
        std_acc = pm["standard"]["avg_acc"]
        best_key = None
        best_score = -999
        for key, val in pm.items():
            if key == "standard":
                continue
            # Score: minimize acc loss, maximize time savings
            acc_delta = val["avg_acc"] - std_acc
            if abs(acc_delta) < 0.02:  # within 2pp
                time_ratio = pm["standard"]["avg_consol_time"] / max(val["avg_consol_time"], 1e-12)
                if time_ratio > best_score:
                    best_score = time_ratio
                    best_key = key
        if best_key:
            print(f"\nBest config (within 2pp of standard): {best_key} (time ratio: {best_score:.3f})")
        else:
            print(f"\nNo config within 2pp of standard EWC accuracy.")

    if ratio_trend > 0.05:
        print("\nVERDICT: The log-structured merge pattern DOES amortize EWC consolidation cost. "
              "The amortization ratio grows with task count N, confirming that the LSM-tree "
              "pattern provides increasing benefits as the number of tasks scales. "
              "Hypothesis 2 is SUPPORTED.")
    elif ratio_trend > -0.05:
        print("\nVERDICT: The log-structured merge pattern shows NEUTRAL scaling behavior. "
              "The amortization ratio is roughly constant across N. The pattern works but "
              "does not provide increasing returns with scale. Hypothesis 2 is PARTIALLY SUPPORTED.")
    else:
        print("\nVERDICT: The log-structured merge pattern does NOT amortize effectively. "
              "The amortization ratio decreases with N. Hypothesis 2 is NOT SUPPORTED.")


if __name__ == "__main__":
    main()
