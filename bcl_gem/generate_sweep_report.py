"""BCL-EXP-1: Generate sweep report with kill-condition evaluation.

Run with: cd bcl_gem && python generate_sweep_report.py --results-dir ../results/bcl_sweep
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_results(results_dir):
    """Load all individual JSON result files."""
    results = []
    for p in sorted(Path(results_dir).glob("*.json")):
        if p.name == "summary.json":
            continue
        with open(p) as f:
            results.append(json.load(f))
    return results


def aggregate(results):
    """Group by config_id, compute mean/std across seeds."""
    groups = defaultdict(list)
    for r in results:
        groups[r["config_id"]].append(r)

    agg = {}
    for config_id, runs in groups.items():
        bwts = [r["bwt"] for r in runs]
        accs = [r["avg_acc"] for r in runs]

        # Mean KL drift across all drift measurements and seeds
        all_kl = []
        for r in runs:
            kl_vals = [d["kl"] for d in r["kl_drift"]]
            if kl_vals:
                all_kl.append(np.mean(kl_vals))

        agg[config_id] = {
            "config_id": config_id,
            "benchmark": runs[0]["benchmark"],
            "is_vanilla": runs[0]["is_vanilla"],
            "flush_freq": runs[0]["flush_freq"],
            "hot_fraction": runs[0]["hot_fraction"],
            "merge_strategy": runs[0]["merge_strategy"],
            "n_seeds": len(runs),
            "bwt_mean": np.mean(bwts),
            "bwt_std": np.std(bwts),
            "avg_acc_mean": np.mean(accs),
            "avg_acc_std": np.std(accs),
            "kl_drift_mean": np.mean(all_kl) if all_kl else 0.0,
        }
    return agg


def format_table(rows, benchmark):
    """Format sweep results as a Markdown table."""
    lines = [
        f"\n### Sweep Results ({benchmark})\n",
        "| flush_freq | hot_frac | merge      | BWT (mean±std)    | Avg Acc (mean±std) | KL_drift (mean) |",
        "|------------|----------|------------|-------------------|--------------------|-----------------|",
    ]
    for r in sorted(rows, key=lambda x: (x["flush_freq"], x["hot_fraction"], x["merge_strategy"])):
        bwt = f"{r['bwt_mean']*100:+.2f}±{r['bwt_std']*100:.2f}"
        acc = f"{r['avg_acc_mean']*100:.2f}±{r['avg_acc_std']*100:.2f}"
        kl = f"{r['kl_drift_mean']:.4f}"
        lines.append(
            f"| {r['flush_freq']:<10} | {r['hot_fraction']:<8} | {r['merge_strategy']:<10} "
            f"| {bwt:<17} | {acc:<18} | {kl:<15} |"
        )
    return "\n".join(lines)


def check_kill_condition(agg, benchmarks):
    """Check if best accumulator BWT is within ±1pp of vanilla on ALL benchmarks."""
    kill = True
    details = []

    for bench in benchmarks:
        vanilla_key = f"{bench}_vanilla"
        vanilla = agg.get(vanilla_key)
        if not vanilla:
            details.append(f"  {bench}: NO VANILLA BASELINE FOUND")
            kill = False
            continue

        vanilla_bwt = vanilla["bwt_mean"]

        # Find best accumulator config for this benchmark
        accum_rows = [v for v in agg.values()
                      if v["benchmark"] == bench and not v["is_vanilla"]]
        if not accum_rows:
            details.append(f"  {bench}: NO ACCUMULATOR RESULTS")
            kill = False
            continue

        best = max(accum_rows, key=lambda x: x["bwt_mean"])
        delta = (best["bwt_mean"] - vanilla_bwt) * 100

        if abs(delta) > 1.0:
            kill = False

        details.append(
            f"  {bench}: vanilla BWT={vanilla_bwt*100:+.2f}%, "
            f"best accum BWT={best['bwt_mean']*100:+.2f}% "
            f"(Δ={delta:+.2f}pp, config={best['config_id']})"
        )

    return kill, details


def generate_report(results_dir):
    """Generate the full sweep report."""
    results = load_results(results_dir)
    if not results:
        print("No results found.")
        return

    agg = aggregate(results)
    benchmarks = sorted(set(r["benchmark"] for r in results))

    report = ["# BCL-EXP-1: Two-Tier Accumulator Replay Buffer — Sweep Report\n"]

    # 1. Vanilla baselines
    report.append("## 1. Vanilla GEM Baselines\n")
    report.append("| Benchmark | BWT (mean±std) | Avg Acc (mean±std) | KL drift |")
    report.append("|-----------|----------------|--------------------|----------|")
    for bench in benchmarks:
        key = f"{bench}_vanilla"
        v = agg.get(key)
        if v:
            report.append(
                f"| {bench} | {v['bwt_mean']*100:+.2f}±{v['bwt_std']*100:.2f}% "
                f"| {v['avg_acc_mean']*100:.2f}±{v['avg_acc_std']*100:.2f}% "
                f"| {v['kl_drift_mean']:.4f} |"
            )

    # 2. Sweep results per benchmark
    report.append("\n## 2. Sweep Results\n")
    for bench in benchmarks:
        accum_rows = [v for v in agg.values()
                      if v["benchmark"] == bench and not v["is_vanilla"]]
        if accum_rows:
            report.append(format_table(accum_rows, bench))

    # 3. Best config per benchmark
    report.append("\n## 3. Best Configuration per Benchmark\n")
    for bench in benchmarks:
        vanilla = agg.get(f"{bench}_vanilla")
        accum_rows = [v for v in agg.values()
                      if v["benchmark"] == bench and not v["is_vanilla"]]
        if not accum_rows or not vanilla:
            continue

        best = max(accum_rows, key=lambda x: x["bwt_mean"])
        bwt_delta = (best["bwt_mean"] - vanilla["bwt_mean"]) * 100
        kl_reduction = ((vanilla["kl_drift_mean"] - best["kl_drift_mean"])
                        / max(vanilla["kl_drift_mean"], 1e-8) * 100)

        report.append(f"**{bench}:**")
        report.append(f"- Best config: flush_freq={best['flush_freq']}, "
                      f"hot_frac={best['hot_fraction']}, merge={best['merge_strategy']}")
        report.append(f"- BWT: {best['bwt_mean']*100:+.2f}±{best['bwt_std']*100:.2f}% "
                      f"(Δ={bwt_delta:+.2f}pp vs vanilla)")
        report.append(f"- Avg Acc: {best['avg_acc_mean']*100:.2f}±{best['avg_acc_std']*100:.2f}%")
        report.append(f"- KL drift: {best['kl_drift_mean']:.4f} "
                      f"(reduction: {kl_reduction:+.1f}%)")
        report.append("")

    # 4. Kill condition
    report.append("## 4. Kill Condition Evaluation\n")
    kill, details = check_kill_condition(agg, benchmarks)
    for d in details:
        report.append(d)
    report.append("")
    if kill:
        report.append("**VERDICT: KILL** — Best accumulator BWT is within ±1pp of vanilla "
                      "GEM on ALL benchmarks. The accumulator pattern does not transfer "
                      "to replay buffer management in a measurable way.")
    else:
        report.append("**VERDICT: PASS** — At least one benchmark shows >1pp BWT improvement.")

    # 5. Interpretation
    report.append("\n## 5. Interpretation\n")
    for bench in benchmarks:
        vanilla = agg.get(f"{bench}_vanilla")
        accum_rows = [v for v in agg.values()
                      if v["benchmark"] == bench and not v["is_vanilla"]]
        if not accum_rows or not vanilla:
            continue
        best = max(accum_rows, key=lambda x: x["bwt_mean"])
        bwt_improved = best["bwt_mean"] > vanilla["bwt_mean"] + 0.01
        kl_improved = best["kl_drift_mean"] < vanilla["kl_drift_mean"]

        if bwt_improved and not kl_improved:
            report.append(
                f"- **{bench}**: BWT improves but KL drift does NOT decrease. "
                f"The mechanism may work through a different channel than hypothesized "
                f"(e.g., gradient diversity rather than distribution alignment)."
            )
        elif bwt_improved and kl_improved:
            report.append(
                f"- **{bench}**: Both BWT and KL drift improve. Consistent with hypothesis "
                f"that the accumulator buffer reduces buffer-stream distribution divergence."
            )
        else:
            report.append(
                f"- **{bench}**: No significant BWT improvement."
            )

    report_text = "\n".join(report)

    out_path = Path(results_dir) / "sweep_report.md"
    with open(out_path, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to {out_path}")
    return report_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results/bcl_sweep")
    args = parser.parse_args()
    generate_report(args.results_dir)


if __name__ == "__main__":
    main()
