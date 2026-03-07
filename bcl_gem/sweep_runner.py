"""BCL-EXP-1: Hyperparameter sweep for two-tier accumulator replay buffer.

Run with: cd bcl_gem && python sweep_runner.py --benchmark split_mnist
          cd bcl_gem && python sweep_runner.py --benchmark all
"""
import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from benchmarks import set_seed, get_split_mnist, get_split_cifar10, get_permuted_mnist
from model import MLP, SmallCNN
from gem import GEM
from accumulator_gem import AccumulatorGEM
from metrics import compute_accuracy, compute_bwt, compute_avg_acc, compute_drift_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%H:%M:%S'
)

BENCHMARK_N_TASKS = {
    "split_mnist": 5,
    "split_cifar10": 5,
    "permuted_mnist": 20,
}

SWEEP_FLUSH_FREQ = [50, 100, 500, 1000]
SWEEP_HOT_FRACTION = [0.1, 0.25, 0.5]
SWEEP_MERGE_STRATEGY = ["random", "reservoir", "importance"]
SWEEP_SEEDS = [0, 1, 2]


@dataclass
class SweepConfig:
    benchmark: str
    seed: int
    flush_freq: int
    hot_fraction: float
    merge_strategy: str
    total_budget: int
    is_vanilla: bool = False
    epochs: int = 1
    lr: float = 0.1
    margin: float = 0.5
    batch_size: int = 10
    memory_per_task: int = 256

    @property
    def config_id(self):
        if self.is_vanilla:
            return f"{self.benchmark}_vanilla"
        return (f"{self.benchmark}_ff{self.flush_freq}"
                f"_hf{self.hot_fraction}_ms{self.merge_strategy}")

    @property
    def run_id(self):
        return f"{self.config_id}_s{self.seed}"


def generate_configs(benchmarks, merge_strategies=None, seeds=None):
    """Generate all sweep configs for given benchmarks."""
    if merge_strategies is None:
        merge_strategies = SWEEP_MERGE_STRATEGY
    if seeds is None:
        seeds = SWEEP_SEEDS

    configs = []
    for bench in benchmarks:
        n_tasks = BENCHMARK_N_TASKS[bench]
        total_budget = 256 * n_tasks

        # Vanilla GEM controls
        for seed in seeds:
            configs.append(SweepConfig(
                benchmark=bench, seed=seed,
                flush_freq=0, hot_fraction=0.0,
                merge_strategy="none", total_budget=total_budget,
                is_vanilla=True
            ))

        # Accumulator configs
        for ff in SWEEP_FLUSH_FREQ:
            for hf in SWEEP_HOT_FRACTION:
                for ms in merge_strategies:
                    for seed in seeds:
                        configs.append(SweepConfig(
                            benchmark=bench, seed=seed,
                            flush_freq=ff, hot_fraction=hf,
                            merge_strategy=ms, total_budget=total_budget
                        ))

    return configs


def setup_benchmark(config):
    """Create model, data loaders, and n_tasks for a benchmark."""
    if config.benchmark == "split_mnist":
        tasks_train, tasks_test, num_classes = get_split_mnist(batch_size=config.batch_size)
        model = MLP(28*28, 256, num_classes, num_tasks=5)
        n_tasks = 5
    elif config.benchmark == "split_cifar10":
        tasks_train, tasks_test, num_classes = get_split_cifar10(batch_size=config.batch_size)
        model = SmallCNN(num_classes, num_tasks=5)
        n_tasks = 5
    elif config.benchmark == "permuted_mnist":
        tasks_train, tasks_test, num_classes = get_permuted_mnist(
            batch_size=config.batch_size, n_tasks=20, seed=config.seed)
        model = MLP(28*28, 256, num_classes)
        n_tasks = 20
    else:
        raise ValueError(f"Unknown benchmark: {config.benchmark}")
    return model, tasks_train, tasks_test, num_classes, n_tasks


def run_single(config):
    """Run a single experiment (vanilla GEM or AccumulatorGEM)."""
    set_seed(config.seed)

    model, tasks_train, tasks_test, num_classes, n_tasks = setup_benchmark(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if config.is_vanilla:
        gem = GEM(model, n_tasks=n_tasks, memory_size=config.memory_per_task,
                  lr=config.lr, margin=config.margin)
    else:
        gem = AccumulatorGEM(
            model, n_tasks=n_tasks, total_budget=config.total_budget,
            hot_fraction=config.hot_fraction, flush_freq=config.flush_freq,
            merge_strategy=config.merge_strategy,
            lr=config.lr, margin=config.margin
        )

    metrics = {
        "acc_matrix": np.zeros((n_tasks, n_tasks)),
        "kl_drift": [],
        "js_drift": []
    }
    batches_seen = 0

    for t in range(n_tasks):
        for ep in range(config.epochs):
            for x, y in tasks_train[t]:
                x, y = x.to(device), y.to(device)
                gem.observe(x, y, t)
                batches_seen += 1

                if batches_seen % 100 == 0:
                    kl, js = compute_drift_metrics(gem, y, num_classes)
                    metrics["kl_drift"].append({
                        "batch": batches_seen, "task": t,
                        "kl": float(kl)
                    })
                    metrics["js_drift"].append({
                        "batch": batches_seen, "task": t,
                        "js": float(js)
                    })

        gem.end_task(tasks_train[t], t)

        for eval_t in range(t + 1):
            acc = compute_accuracy(model, tasks_test[eval_t], eval_t)
            metrics["acc_matrix"][t, eval_t] = acc

    bwt = compute_bwt(metrics["acc_matrix"])
    avg_acc = compute_avg_acc(metrics["acc_matrix"])

    return {
        "run_id": config.run_id,
        "config_id": config.config_id,
        "benchmark": config.benchmark,
        "seed": config.seed,
        "is_vanilla": config.is_vanilla,
        "flush_freq": config.flush_freq,
        "hot_fraction": config.hot_fraction,
        "merge_strategy": config.merge_strategy,
        "total_budget": config.total_budget,
        "bwt": float(bwt),
        "avg_acc": float(avg_acc),
        "acc_matrix": metrics["acc_matrix"].tolist(),
        "kl_drift": metrics["kl_drift"],
        "js_drift": metrics["js_drift"],
    }


def run_sweep(configs, results_dir):
    """Run all configs, saving results incrementally with resume support."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Resume: check completed runs
    completed = set()
    for p in results_dir.glob("*.json"):
        if p.name != "summary.json":
            completed.add(p.stem)

    remaining = [c for c in configs if c.run_id not in completed]
    logging.info(f"{len(completed)} complete, {len(remaining)} remaining out of {len(configs)}")

    for i, config in enumerate(remaining):
        t0 = time.time()
        logging.info(f"[{i+1}/{len(remaining)}] {config.run_id}")

        result = run_single(config)
        elapsed = time.time() - t0

        # Save individual result
        out_path = results_dir / f"{config.run_id}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        bwt_str = f"{result['bwt']*100:+.2f}%"
        acc_str = f"{result['avg_acc']*100:.2f}%"
        logging.info(f"  BWT={bwt_str} AvgAcc={acc_str} ({elapsed:.1f}s)")

    # Generate summary
    all_results = []
    for p in sorted(results_dir.glob("*.json")):
        if p.name != "summary.json":
            with open(p) as f:
                all_results.append(json.load(f))

    with open(results_dir / "summary.json", "w") as f:
        # Strip large fields for summary
        summary = []
        for r in all_results:
            summary.append({
                "run_id": r["run_id"],
                "config_id": r["config_id"],
                "benchmark": r["benchmark"],
                "seed": r["seed"],
                "is_vanilla": r["is_vanilla"],
                "flush_freq": r["flush_freq"],
                "hot_fraction": r["hot_fraction"],
                "merge_strategy": r["merge_strategy"],
                "bwt": r["bwt"],
                "avg_acc": r["avg_acc"],
            })
        json.dump(summary, f, indent=2)

    logging.info(f"Summary saved to {results_dir / 'summary.json'}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="BCL-EXP-1 sweep runner")
    parser.add_argument("--benchmark", required=True,
                        choices=["split_mnist", "split_cifar10", "permuted_mnist", "all"])
    parser.add_argument("--merge-strategy", default="all",
                        choices=["random", "reservoir", "importance", "all"])
    parser.add_argument("--seed", type=int, default=None,
                        help="Run single seed (default: all 3)")
    parser.add_argument("--vanilla-only", action="store_true",
                        help="Only run vanilla GEM baselines")
    parser.add_argument("--results-dir", default="../results/bcl_sweep")
    args = parser.parse_args()

    if args.benchmark == "all":
        benchmarks = ["split_mnist", "split_cifar10", "permuted_mnist"]
    else:
        benchmarks = [args.benchmark]

    merge_strategies = (SWEEP_MERGE_STRATEGY if args.merge_strategy == "all"
                        else [args.merge_strategy])
    seeds = [args.seed] if args.seed is not None else SWEEP_SEEDS

    configs = generate_configs(benchmarks, merge_strategies, seeds)

    if args.vanilla_only:
        configs = [c for c in configs if c.is_vanilla]

    logging.info(f"Total configs: {len(configs)}")
    run_sweep(configs, args.results_dir)


if __name__ == "__main__":
    main()
