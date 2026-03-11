"""BCL-EXP-2 Gauntlet: Characterize accumulator-buffer scaling with task count N.

Run matrix: 6 N-values × 2 conditions × 3 seeds = 36 runs on Permuted-MNIST.
Conditions:
  1. Vanilla GEM (uniform per-task buffer, memory_per_task = B // N)
  2. Accumulator-GEM (best EXP-1 config: random merge, 50% hot, flush@100)

Total buffer budget B = 1280 held constant for both conditions.
Training knobs pinned to EXP-0/EXP-1: lr=0.1, epochs=1, batch_size=10, margin=0.5.

Features:
  - Incremental checkpointing: each completed run saved immediately to disk.
  - Resume: re-running skips already-completed (N, condition, seed) combos.
  - All task data pre-loaded to GPU to reduce transfer overhead.
"""

import sys
import os
import json
import logging
import time
import numpy as np
import torch
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks import set_seed, get_permuted_mnist, FastTensorDataLoader
from model import MLP
from gem import GEM
from accumulator_gem import AccumulatorGEM
from metrics import compute_accuracy, compute_bwt, compute_avg_acc, compute_drift_metrics

# ── Fixed experiment parameters ──────────────────────────────────────────────
TASK_COUNTS = [5, 10, 15, 20, 30, 50]
SEEDS = [42, 43, 44]
TOTAL_BUDGET = 1280          # B: same for both conditions
LR = 0.1
MARGIN = 0.5
BATCH_SIZE = 10
EPOCHS = 1
NUM_CLASSES = 10             # Permuted-MNIST: shared 10-class label space

# Accumulator config (best from EXP-1, not retuned)
HOT_FRACTION = 0.5
FLUSH_FREQ = 100
MERGE_STRATEGY = "random"

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "exp2")
CHECKPOINT_FILE = os.path.join(OUT_DIR, "checkpoint.json")

LOG = logging.getLogger("exp2")


def _run_key(n_tasks, condition, seed):
    return f"{n_tasks}_{condition}_{seed}"


def _load_checkpoint():
    """Load completed runs from checkpoint file."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}


def _save_checkpoint(completed):
    """Save completed runs to checkpoint file."""
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(completed, f, indent=2)


def _gpu_loaders(tasks_train, tasks_test, device):
    """Re-wrap loaders with data pre-moved to GPU."""
    gpu_train = []
    gpu_test = []
    for loader in tasks_train:
        gx = loader.tensors[0].to(device)
        gy = loader.tensors[1].to(device)
        gpu_train.append(FastTensorDataLoader(gx, gy, batch_size=loader.batch_size,
                                              shuffle=loader.shuffle))
    for loader in tasks_test:
        gx = loader.tensors[0].to(device)
        gy = loader.tensors[1].to(device)
        gpu_test.append(FastTensorDataLoader(gx, gy, batch_size=loader.batch_size,
                                             shuffle=loader.shuffle))
    return gpu_train, gpu_test


def run_single(n_tasks, condition, seed):
    """Execute one (N, condition, seed) configuration. Returns metrics dict."""
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tasks_train, tasks_test, num_classes = get_permuted_mnist(
        batch_size=BATCH_SIZE, n_tasks=n_tasks, seed=seed,
    )

    # Pre-load all data to GPU to eliminate per-batch transfer overhead
    if device.type == "cuda":
        try:
            tasks_train, tasks_test = _gpu_loaders(tasks_train, tasks_test, device)
        except torch.cuda.OutOfMemoryError:
            LOG.warning(f"  GPU OOM preloading N={n_tasks} — falling back to CPU transfers")
            torch.cuda.empty_cache()

    model = MLP(28 * 28, 256, num_classes).to(device)
    if condition == "vanilla":
        mem_per_task = max(1, TOTAL_BUDGET // n_tasks)
        agent = GEM(model, n_tasks=n_tasks, memory_size=mem_per_task,
                     lr=LR, margin=MARGIN)
    else:
        agent = AccumulatorGEM(
            model, n_tasks=n_tasks,
            total_budget=TOTAL_BUDGET,
            hot_fraction=HOT_FRACTION,
            flush_freq=FLUSH_FREQ,
            merge_strategy=MERGE_STRATEGY,
            lr=LR, margin=MARGIN,
        )

    acc_matrix = np.zeros((n_tasks, n_tasks))
    kl_at_boundaries = []

    for t in range(n_tasks):
        for _ep in range(EPOCHS):
            for x, y in tasks_train[t]:
                x, y = x.to(device), y.to(device)
                agent.observe(x, y, t)

        agent.end_task(tasks_train[t], t)

        # KL divergence at task boundary (first batch as stream proxy)
        for x, y in tasks_train[t]:
            x, y = x.to(device), y.to(device)
            kl, js = compute_drift_metrics(agent, y, num_classes)
            kl_at_boundaries.append({"task": t, "kl": float(kl), "js": float(js)})
            break

        # Evaluate on all tasks seen so far
        for eval_t in range(t + 1):
            acc = compute_accuracy(model, tasks_test[eval_t], eval_t)
            acc_matrix[t, eval_t] = acc

        if (t + 1) % 5 == 0 or t == n_tasks - 1:
            LOG.info(f"    task {t+1}/{n_tasks} done")

    bwt = compute_bwt(acc_matrix)
    avg_acc = compute_avg_acc(acc_matrix)

    return {
        "bwt": float(bwt),
        "avg_acc": float(avg_acc),
        "acc_matrix": acc_matrix.tolist(),
        "kl_boundaries": kl_at_boundaries,
    }


def analyse(results):
    """Aggregate results, fit regression, print tables, save plots."""
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Raw dump ─────────────────────────────────────────────────────────
    with open(os.path.join(OUT_DIR, "raw_results.json"), "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    # ── Per-N aggregation ────────────────────────────────────────────────
    rows = []
    Ns = []
    deltas = []

    for N in TASK_COUNTS:
        if N not in results:
            continue
        v_bwts = [r["bwt"] for r in results[N]["vanilla"]]
        a_bwts = [r["bwt"] for r in results[N]["accumulator"]]
        if not v_bwts or not a_bwts:
            continue

        v_mean, v_std = np.mean(v_bwts), np.std(v_bwts)
        a_mean, a_std = np.mean(a_bwts), np.std(a_bwts)
        delta = a_mean - v_mean

        rows.append(dict(N=N, v_mean=v_mean, v_std=v_std,
                         a_mean=a_mean, a_std=a_std, delta=delta))
        Ns.append(N)
        deltas.append(delta)

    if len(Ns) < 3:
        LOG.warning("Fewer than 3 N-values complete — skipping regression.")
        return None

    # ── Regression: Δ(BWT) = a·N + b ────────────────────────────────────
    Ns_arr = np.array(Ns, dtype=float)
    deltas_arr = np.array(deltas)
    slope, intercept, r_val, p_val, std_err = sp_stats.linregress(Ns_arr, deltas_arr)

    t_crit = sp_stats.t.ppf(0.975, df=max(1, len(Ns_arr) - 2))
    ci_lo = slope - t_crit * std_err
    ci_hi = slope + t_crit * std_err

    if ci_lo > 0:
        interpretation = "scaling thesis SUPPORTED (positive slope, CI excludes zero)"
    elif ci_hi < 0:
        interpretation = "scaling thesis REJECTED (negative slope, CI excludes zero)"
    else:
        interpretation = "INCONCLUSIVE (CI includes zero)"

    soft_kill = slope <= 0 and ci_hi < 0

    # ── Print results ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("BCL-EXP-2 GAUNTLET RESULTS")
    print("=" * 80)

    print("\nScaling Curve Results:")
    print("| N  | Vanilla BWT (mean±std) | Accum BWT (mean±std) | Δ(BWT) |")
    print("|----|------------------------|----------------------|--------|")
    for r in rows:
        print(f"| {r['N']:<2} | {r['v_mean']*100:+.2f}% ± {r['v_std']*100:.2f}%"
              f"       | {r['a_mean']*100:+.2f}% ± {r['a_std']*100:.2f}%"
              f"     | {r['delta']*100:+.2f}% |")

    print(f"\nRegression: Δ(BWT) = a·N + b")
    print(f"  a = {slope * 100:.4f}pp/task (95% CI: [{ci_lo * 100:.4f}, {ci_hi * 100:.4f}])")
    print(f"  b = {intercept * 100:.4f}pp")
    print(f"  R² = {r_val ** 2:.4f}")
    print(f"  Interpretation: {interpretation}")

    if soft_kill:
        print("\n*** SOFT KILL: Δ(BWT) is not increasing with N. "
              "Scaling narrative weakened. ***")

    # ── KL summary ───────────────────────────────────────────────────────
    print("\nKL Divergence at Task Boundaries (mean over seeds):")
    print("| N  | Vanilla KL (mean) | Accum KL (mean) |")
    print("|----|-------------------|-----------------|")
    kl_rows = []
    for N in Ns:
        v_kls = [np.mean([b["kl"] for b in r["kl_boundaries"]])
                 for r in results[N]["vanilla"]]
        a_kls = [np.mean([b["kl"] for b in r["kl_boundaries"]])
                 for r in results[N]["accumulator"]]
        v_kl_m, a_kl_m = np.mean(v_kls), np.mean(a_kls)
        kl_rows.append(dict(N=N, v_kl=v_kl_m, a_kl=a_kl_m))
        print(f"| {N:<2} | {v_kl_m:.4f}            | {a_kl_m:.4f}          |")

    # ── Average accuracy summary ─────────────────────────────────────────
    print("\nAverage Accuracy (mean over seeds):")
    print("| N  | Vanilla AvgAcc | Accum AvgAcc |")
    print("|----|----------------|--------------|")
    for N in Ns:
        v_acc = np.mean([r["avg_acc"] for r in results[N]["vanilla"]])
        a_acc = np.mean([r["avg_acc"] for r in results[N]["accumulator"]])
        print(f"| {N:<2} | {v_acc*100:.2f}%          | {a_acc*100:.2f}%        |")

    # ── Save structured summary ──────────────────────────────────────────
    summary = {
        "config": {
            "total_budget": TOTAL_BUDGET,
            "task_counts": TASK_COUNTS,
            "seeds": SEEDS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "accumulator": {
                "hot_fraction": HOT_FRACTION,
                "flush_freq": FLUSH_FREQ,
                "merge_strategy": MERGE_STRATEGY,
            },
        },
        "rows": rows,
        "kl_rows": kl_rows,
        "regression": {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_val ** 2),
            "slope_ci_95": [float(ci_lo), float(ci_hi)],
            "p_value": float(p_val),
            "std_err": float(std_err),
            "interpretation": interpretation,
            "soft_kill": bool(soft_kill),
        },
    }
    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            return super().default(obj)

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, cls=_NumpyEncoder)

    # ── Plots ────────────────────────────────────────────────────────────
    _save_plots(rows, kl_rows, slope, intercept)

    return summary


def _save_plots(rows, kl_rows, slope, intercept):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOG.warning("matplotlib not available — skipping plots.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    Ns = [r["N"] for r in rows]

    # 1. BWT vs N
    ax = axes[0]
    ax.errorbar(Ns, [r["v_mean"] * 100 for r in rows],
                yerr=[r["v_std"] * 100 for r in rows],
                marker="o", capsize=4, label="Vanilla GEM")
    ax.errorbar(Ns, [r["a_mean"] * 100 for r in rows],
                yerr=[r["a_std"] * 100 for r in rows],
                marker="s", capsize=4, label="Accumulator-GEM")
    ax.set_xlabel("Number of Tasks (N)")
    ax.set_ylabel("BWT (%)")
    ax.set_title("BWT vs Task Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Δ(BWT) vs N
    ax = axes[1]
    delta_pcts = [r["delta"] * 100 for r in rows]
    ax.plot(Ns, delta_pcts, "ro-", markersize=8)
    fit_x = np.linspace(min(Ns), max(Ns), 100)
    fit_y = (slope * fit_x + intercept) * 100
    ax.plot(fit_x, fit_y, "b--", alpha=0.7,
            label=f"Fit: a={slope * 100:.4f}pp/task")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Number of Tasks (N)")
    ax.set_ylabel("Δ(BWT) = Accum − Vanilla (%)")
    ax.set_title("Δ(BWT) vs Task Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. KL divergence vs N
    ax = axes[2]
    ax.plot([r["N"] for r in kl_rows], [r["v_kl"] for r in kl_rows],
            "o-", label="Vanilla GEM")
    ax.plot([r["N"] for r in kl_rows], [r["a_kl"] for r in kl_rows],
            "s-", label="Accumulator-GEM")
    ax.set_xlabel("Number of Tasks (N)")
    ax.set_ylabel("Mean KL Divergence")
    ax.set_title("Buffer-Stream KL Divergence vs Task Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "scaling_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nPlots saved to {plot_path}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    # Load checkpoint for resume support
    completed = _load_checkpoint()
    LOG.info(f"Checkpoint: {len(completed)} runs already completed.")

    total_runs = len(TASK_COUNTS) * 2 * len(SEEDS)
    run_idx = 0

    # Reconstruct results dict from checkpoint
    results = {}
    for N in TASK_COUNTS:
        results[N] = {"vanilla": [], "accumulator": []}

    for key, res in completed.items():
        parts = key.split("_")
        N = int(parts[0])
        condition = parts[1]
        results[N][condition].append(res)

    for N in TASK_COUNTS:
        for condition in ["vanilla", "accumulator"]:
            for seed in SEEDS:
                run_idx += 1
                key = _run_key(N, condition, seed)

                if key in completed:
                    LOG.info(f"[{run_idx}/{total_runs}] N={N}, {condition}, seed={seed} "
                             f"— CACHED (BWT={completed[key]['bwt']*100:.2f}%)")
                    continue

                LOG.info(f"[{run_idx}/{total_runs}] N={N}, {condition}, seed={seed}")
                t0 = time.time()
                res = run_single(N, condition, seed)
                elapsed = time.time() - t0
                LOG.info(f"  BWT={res['bwt'] * 100:.2f}%, "
                         f"AvgAcc={res['avg_acc'] * 100:.2f}%, "
                         f"time={elapsed:.1f}s")

                results[N][condition].append(res)

                # Incremental checkpoint
                completed[key] = res
                _save_checkpoint(completed)
                LOG.info(f"  Checkpointed ({len(completed)}/{total_runs} complete)")

    analyse(results)


if __name__ == "__main__":
    main()
