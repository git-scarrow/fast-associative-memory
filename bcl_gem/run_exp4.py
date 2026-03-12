"""BCL-EXP-4: Per-task-partitioned accumulator buffer scaling gauntlet.

36-run gauntlet: 6 N-values × 2 conditions × 3 seeds.
Tests whether EXP-2 scaling failure is shared-pool starvation or inherent.

Run: cd bcl_gem && python run_exp4.py
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from benchmarks import set_seed, get_permuted_mnist
from model import MLP
from gem import GEM
from accumulator_gem import AccumulatorGEM
from partitioned_buffer import PartitionedAccumulatorBuffer
from metrics import compute_accuracy, compute_bwt, compute_avg_acc, compute_drift_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# --- Config ---
N_VALUES = [5, 10, 20, 30, 40, 50]
SEEDS = [42, 43, 44]
TOTAL_BUDGET = 1280
HOT_FRACTION = 0.5
FLUSH_FREQ = 100
MERGE_STRATEGY = "random"
LR = 0.1
MARGIN = 0.5
BATCH_SIZE = 128
MEMORY_SIZE_PER_TASK = 256  # vanilla GEM
EPOCHS = 1
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "exp4"


class PartitionedAccumulatorGEM(AccumulatorGEM):
    """AccumulatorGEM with per-task-partitioned cold zone."""

    def __init__(self, model, n_tasks, total_budget=1280,
                 hot_fraction=0.5, flush_freq=100,
                 merge_strategy="random", lr=0.1, margin=0.5):
        # Skip AccumulatorGEM.__init__ to avoid creating shared-pool buffer
        # Call GEM.__init__ directly
        GEM.__init__(self, model, n_tasks, memory_size=0, lr=lr, margin=margin)
        del self.memory_x
        del self.memory_y

        self._total_budget = total_budget
        self._hot_fraction = hot_fraction
        self._flush_freq = flush_freq
        self._merge_strategy = merge_strategy
        self._max_tasks = n_tasks

        self.buffer = None
        self._batch_counter = 0

    def observe(self, x, y, task_id):
        if self.buffer is None:
            self.buffer = PartitionedAccumulatorBuffer(
                total_budget=self._total_budget,
                hot_fraction=self._hot_fraction,
                flush_freq=self._flush_freq,
                merge_strategy=self._merge_strategy,
                x_shape=x.shape[1:],
                device=self.device,
                max_tasks=self._max_tasks,
            )
        # Delegate to parent (AccumulatorGEM.observe handles ingestion, flush, QP)
        AccumulatorGEM.observe(self, x, y, task_id)


def run_single(n_tasks, condition, seed):
    """Run a single experiment: train on n_tasks Permuted-MNIST tasks, return metrics."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tasks_train, tasks_test, num_classes = get_permuted_mnist(
        batch_size=BATCH_SIZE, n_tasks=n_tasks, seed=seed
    )
    model = MLP(28 * 28, 256, num_classes).to(device)

    if condition == "vanilla":
        agent = GEM(model, n_tasks=n_tasks, memory_size=MEMORY_SIZE_PER_TASK,
                     lr=LR, margin=MARGIN)
    elif condition == "partitioned":
        agent = PartitionedAccumulatorGEM(
            model, n_tasks=n_tasks, total_budget=TOTAL_BUDGET,
            hot_fraction=HOT_FRACTION, flush_freq=FLUSH_FREQ,
            merge_strategy=MERGE_STRATEGY, lr=LR, margin=MARGIN,
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")

    acc_matrix = np.zeros((n_tasks, n_tasks))
    kl_drift = []
    batches_seen = 0

    for t in range(n_tasks):
        for ep in range(EPOCHS):
            for x, y in tasks_train[t]:
                x, y = x.to(device), y.to(device)
                agent.observe(x, y, t)
                batches_seen += 1

                if batches_seen % 100 == 0:
                    kl, js = compute_drift_metrics(agent, y, num_classes)
                    kl_drift.append({"batch": batches_seen, "task": t, "kl": float(kl)})

        agent.end_task(tasks_train[t], t)

        for eval_t in range(t + 1):
            acc = compute_accuracy(model, tasks_test[eval_t], eval_t)
            acc_matrix[t, eval_t] = acc

    bwt = compute_bwt(acc_matrix)
    avg_acc = compute_avg_acc(acc_matrix)

    # Per-task exemplar counts (partitioned only)
    exemplar_counts = {}
    if condition == "partitioned" and agent.buffer is not None:
        exemplar_counts = agent.buffer.get_per_task_counts()

    return {
        "n_tasks": n_tasks,
        "condition": condition,
        "seed": seed,
        "bwt": float(bwt),
        "avg_acc": float(avg_acc),
        "acc_matrix": acc_matrix.tolist(),
        "kl_drift": kl_drift,
        "exemplar_counts": {str(k): v for k, v in exemplar_counts.items()},
    }


def compute_regression(results):
    """Compute Δ(BWT) vs N regression for partitioned-accumulator vs vanilla."""
    # Group by (n_tasks, seed)
    vanilla = {}
    partitioned = {}
    for r in results:
        key = (r["n_tasks"], r["seed"])
        if r["condition"] == "vanilla":
            vanilla[key] = r["bwt"]
        else:
            partitioned[key] = r["bwt"]

    # Compute Δ(BWT) = partitioned_bwt - vanilla_bwt for each (N, seed)
    ns = []
    deltas = []
    for key in sorted(vanilla.keys()):
        if key in partitioned:
            delta = (partitioned[key] - vanilla[key]) * 100  # pp
            ns.append(key[0])
            deltas.append(delta)

    ns = np.array(ns, dtype=float)
    deltas = np.array(deltas, dtype=float)

    if len(ns) < 3:
        return {"error": "Not enough data points for regression"}

    slope, intercept, r_value, p_value, std_err = stats.linregress(ns, deltas)

    # 95% CI for slope
    t_crit = stats.t.ppf(0.975, df=len(ns) - 2)
    ci_low = slope - t_crit * std_err
    ci_high = slope + t_crit * std_err

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "n_points": len(ns),
        "ns": ns.tolist(),
        "deltas": deltas.tolist(),
    }


def evaluate_verdict(regression):
    """Apply kill/pass/inconclusive logic."""
    if "error" in regression:
        return "ERROR", regression["error"]

    ci_low = regression["ci_95_low"]
    ci_high = regression["ci_95_high"]
    slope = regression["slope"]

    if ci_high < 0:
        return "KILL", (
            f"Slope CI [{ci_low:.4f}, {ci_high:.4f}] entirely below zero. "
            f"Scaling failure is inherent to accumulator pattern."
        )
    elif ci_low >= 0:
        return "PASS", (
            f"Slope CI [{ci_low:.4f}, {ci_high:.4f}] entirely non-negative. "
            f"Partitioned buffer rescues scaling thesis."
        )
    elif ci_low < 0 < ci_high and slope < 0:
        return "INCONCLUSIVE", (
            f"Slope CI [{ci_low:.4f}, {ci_high:.4f}] overlaps zero but "
            f"point estimate is negative ({slope:.4f}). Rerun with 5 seeds."
        )
    else:
        return "PASS", (
            f"Slope CI [{ci_low:.4f}, {ci_high:.4f}] includes zero with "
            f"non-negative point estimate ({slope:.4f})."
        )


def generate_scaling_plot(regression, out_path):
    """Generate scaling curves plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available, skipping plot")
        return

    ns = np.array(regression["ns"])
    deltas = np.array(regression["deltas"])

    # Group by N for mean/std
    unique_ns = sorted(set(ns))
    means = [np.mean(deltas[ns == n]) for n in unique_ns]
    stds = [np.std(deltas[ns == n]) for n in unique_ns]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Scatter individual points
    ax.scatter(ns, deltas, alpha=0.5, color="steelblue", label="Individual runs", zorder=3)

    # Mean ± std
    ax.errorbar(unique_ns, means, yerr=stds, fmt="o-", color="navy",
                capsize=4, label="Mean ± std", zorder=4)

    # Regression line
    x_fit = np.linspace(min(ns) - 2, max(ns) + 2, 100)
    y_fit = regression["slope"] * x_fit + regression["intercept"]
    ax.plot(x_fit, y_fit, "--", color="red", alpha=0.7,
            label=f"OLS: slope={regression['slope']:.4f} (p={regression['p_value']:.3f})")

    # CI band
    ci_low_line = regression["ci_95_low"] * x_fit + regression["intercept"]
    ci_high_line = regression["ci_95_high"] * x_fit + regression["intercept"]
    ax.fill_between(x_fit, ci_low_line, ci_high_line, alpha=0.1, color="red", label="95% CI")

    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Number of Tasks (N)")
    ax.set_ylabel("Δ(BWT) [pp] (partitioned - vanilla)")
    ax.set_title("BCL-EXP-4: Partitioned Accumulator Scaling")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # EXP-2 reference line
    exp2_slope = -0.60
    y_exp2 = exp2_slope * x_fit
    ax.plot(x_fit, y_exp2, ":", color="orange", alpha=0.6,
            label=f"EXP-2 shared-pool slope ({exp2_slope})")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Plot saved to {out_path}")


def generate_report(results, regression, verdict, reason, out_dir):
    """Generate markdown report."""
    lines = ["# BCL-EXP-4: Per-Task-Partitioned Accumulator Scaling Gauntlet\n"]

    lines.append("## 1. Configuration\n")
    lines.append(f"- N values: {N_VALUES}")
    lines.append(f"- Seeds: {SEEDS}")
    lines.append(f"- Conditions: vanilla GEM (memory_size={MEMORY_SIZE_PER_TASK}/task) "
                 f"vs partitioned-accumulator (budget={TOTAL_BUDGET}, "
                 f"hot_frac={HOT_FRACTION}, flush={FLUSH_FREQ})")
    lines.append(f"- Batch size: {BATCH_SIZE}")
    lines.append(f"- Benchmark: Permuted-MNIST")
    lines.append(f"- Total runs: {len(results)}\n")

    # Results table
    lines.append("## 2. Results\n")
    lines.append("| N | Condition | Seed | BWT (%) | Avg Acc (%) |")
    lines.append("|---|-----------|------|---------|-------------|")
    for r in sorted(results, key=lambda x: (x["n_tasks"], x["condition"], x["seed"])):
        lines.append(f"| {r['n_tasks']} | {r['condition']} | {r['seed']} "
                     f"| {r['bwt']*100:+.2f} | {r['avg_acc']*100:.2f} |")

    # Δ(BWT) summary
    lines.append("\n## 3. Δ(BWT) vs N Regression\n")
    if "error" not in regression:
        lines.append(f"- **Slope**: {regression['slope']:.4f} pp/task")
        lines.append(f"- **95% CI**: [{regression['ci_95_low']:.4f}, {regression['ci_95_high']:.4f}]")
        lines.append(f"- **R²**: {regression['r_squared']:.4f}")
        lines.append(f"- **p-value**: {regression['p_value']:.4f}")
        lines.append(f"- **EXP-2 shared-pool slope**: −0.60 pp/task (for comparison)")
    else:
        lines.append(f"- Error: {regression['error']}")

    # Per-task exemplar counts
    lines.append("\n## 4. Per-Task Exemplar Counts (Partitioned)\n")
    for r in sorted(results, key=lambda x: x["n_tasks"]):
        if r["condition"] == "partitioned" and r["seed"] == SEEDS[0]:
            counts = r.get("exemplar_counts", {})
            if counts:
                vals = list(counts.values())
                lines.append(f"- N={r['n_tasks']}: min={min(vals)}, max={max(vals)}, "
                             f"mean={np.mean(vals):.1f}, partition_size={TOTAL_BUDGET // 2 // r['n_tasks']}")

    # Verdict
    lines.append(f"\n## 5. Verdict\n")
    lines.append(f"**{verdict}**: {reason}\n")

    if verdict == "KILL":
        lines.append("The scaling failure is **inherent** to the two-tier accumulator pattern. "
                     "Partitioning the cold zone does not rescue the negative Δ(BWT) slope. "
                     "BCL scaling narrative should be archived.")
    elif verdict == "PASS":
        lines.append("The partitioned buffer **rescues** the scaling thesis. "
                     "The EXP-2 failure was caused by shared-pool allocation starvation, "
                     "not a fundamental flaw in the accumulator architecture.")

    report_text = "\n".join(lines)
    report_path = out_dir / "bcl_exp4_report.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    log.info(f"Report saved to {report_path}")
    print(report_text)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    total_runs = len(N_VALUES) * 2 * len(SEEDS)
    run_idx = 0

    for n_tasks in N_VALUES:
        for condition in ["vanilla", "partitioned"]:
            for seed in SEEDS:
                run_idx += 1
                log.info(f"=== Run {run_idx}/{total_runs}: "
                         f"N={n_tasks}, {condition}, seed={seed} ===")

                t0 = time.time()
                result = run_single(n_tasks, condition, seed)
                elapsed = time.time() - t0
                result["elapsed_s"] = round(elapsed, 1)

                log.info(f"  BWT={result['bwt']*100:+.2f}%, "
                         f"AvgAcc={result['avg_acc']*100:.2f}%, "
                         f"time={elapsed:.1f}s")

                all_results.append(result)

                # Checkpoint after each run
                raw_path = RESULTS_DIR / "raw_results.json"
                with open(raw_path, "w") as f:
                    json.dump(all_results, f, indent=2)

    # Regression analysis
    regression = compute_regression(all_results)
    verdict, reason = evaluate_verdict(regression)

    log.info(f"=== VERDICT: {verdict} ===")
    log.info(f"  {reason}")

    # Save summary
    summary = {
        "regression": regression,
        "verdict": verdict,
        "reason": reason,
        "config": {
            "n_values": N_VALUES,
            "seeds": SEEDS,
            "total_budget": TOTAL_BUDGET,
            "hot_fraction": HOT_FRACTION,
            "flush_freq": FLUSH_FREQ,
            "merge_strategy": MERGE_STRATEGY,
            "memory_size_per_task": MEMORY_SIZE_PER_TASK,
        },
        "exp2_reference_slope": -0.60,
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot
    if "error" not in regression:
        generate_scaling_plot(regression, RESULTS_DIR / "scaling_curves.png")

    # Report
    generate_report(all_results, regression, verdict, reason, RESULTS_DIR)


if __name__ == "__main__":
    main()
