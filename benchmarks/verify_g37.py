#!/usr/bin/env python3
"""
benchmarks/verify_g37.py — G37: Offline Negative Evidence Analysis.

Collects RetrievalTrace records on real data, filters to the bottom-quartile
by margin (most ambiguous retrievals), and tests whether rejected candidates'
class distributions carry statistically significant asymmetry vs the accepted
set.

Kill condition: rejected candidates' class distribution must show p < 0.01
asymmetry vs accepted set uncertainty. Kill if rejected mirrors accepted
(dark matter is epiphenomenal).

Statistical approach inspired by:
  - scipy.stats.chi2_contingency (BSD) for independence testing
  - ART literature's "match tracking" as first-class rejection signal
  - Positive-Negative Prototypes Fusion (Sci. Rep. 2025) — negative
    prototypes carry non-redundant class-discriminative signal

Usage:
    PYTHONPATH=. python benchmarks/verify_g37.py --data-root data --device auto
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

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
DOMAINS: list[Domain4] = ["dogs", "birds", "cars", "aircraft"]

P_VALUE_THRESHOLD = 0.01
MARGIN_QUARTILE = 0.25  # Bottom 25% by margin = most ambiguous

# Rejection stage labels from RetrievalTrace
STAGE_NAMES = {
    0: "rerank_drop",
    1: "csls_survivor",
    2: "nstp_kill",
    3: "floor_kill",
    4: "final_survivor",
}


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


# ---------------------------------------------------------------------------
# Phase 1: Build memory and collect traces
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_traces(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    adapter,
    device: torch.device,
    max_traces: int = 100_000,
) -> Tuple[List[dict], FastAssociativeMemory]:
    """Build FAM from training data, then collect RetrievalTrace on test set.

    Returns list of per-query trace dicts and the populated memory.
    """
    num_classes = int(train_labels.max().item()) + 1
    nstp = NSTPController(sibling_threshold=0.85, depth_epsilon=0.10)

    mem = FastAssociativeMemory(
        input_dim=int(train_feats.shape[1]),
        value_dim=num_classes,
        core_entries=50000,
        core_vigilance=0.85,
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
    print(f"    Prototypes: {n_protos}")

    # Collect traces on test set (or train set if test is too small)
    # Use whichever set gets us closest to max_traces
    query_feats = test_feats
    query_labels = test_labels
    if len(test_feats) < max_traces and len(train_feats) > len(test_feats):
        # Combine train + test for more traces
        query_feats = torch.cat([test_feats, train_feats], dim=0)[:max_traces]
        query_labels = torch.cat([test_labels, train_labels], dim=0)[:max_traces]

    core = mem.core_cam
    occ_idx = core.occupied.nonzero(as_tuple=True)[0]
    proto_classes = core.values[occ_idx].float().argmax(dim=-1)  # (N_proto,)

    traces = []
    x_q = query_feats.to(device)
    y_q = query_labels.to(device)

    for i in range(0, min(len(x_q), max_traces), 256):
        batch_x = x_q[i:i + 256]
        batch_y = y_q[i:i + 256]
        B = batch_x.size(0)

        # Project through adapter
        proj = mem._project(batch_x)

        # Forward with trace
        outputs, tr = core(proj, nstp=nstp, trace=True)

        # Compute margin for each query: sim_best - sim_second_best_other_class
        q_norm = F.normalize(proj.float(), dim=-1)
        proto_keys = core._keys_norm[occ_idx]
        sims = q_norm @ proto_keys.T  # (B, N_occ)

        pred_classes = outputs.argmax(dim=1)  # (B,)

        for b in range(B):
            true_class = batch_y[b].item()
            pred_class = pred_classes[b].item()

            # Margin: best sim - best other-class sim
            row_sims = sims[b]
            best_sim = row_sims.max().item()
            same_class_mask = proto_classes == pred_class
            other_sims = row_sims.masked_fill(same_class_mask, -1e9)
            if other_sims.max().item() > -1e8:
                second_best = other_sims.max().item()
                margin = best_sim - second_best
            else:
                margin = best_sim  # No other class → huge margin

            # Get class labels for final (accepted) and rejected candidates
            final_slots = tr.final_slots[b]
            rejected_slots = tr.rejected_slots[b]
            rejection_stages = tr.rejection_stage[b]

            # Map slots back to class labels
            # final_slots are indices into the full memory, not occ_idx
            accepted_classes = []
            for s in final_slots.tolist():
                if s >= 0 and s < core.max_entries and core.occupied[s]:
                    c = core.values[s].float().argmax().item()
                    accepted_classes.append(c)

            rejected_classes = []
            rejected_stages_list = []
            for j, s in enumerate(rejected_slots.tolist()):
                if s >= 0 and s < core.max_entries and core.occupied[s]:
                    c = core.values[s].float().argmax().item()
                    rejected_classes.append(c)
                    if j < len(rejection_stages):
                        rejected_stages_list.append(rejection_stages[j].item())
                    else:
                        rejected_stages_list.append(-1)

            traces.append({
                "true_class": true_class,
                "pred_class": pred_class,
                "correct": true_class == pred_class,
                "margin": margin,
                "best_sim": best_sim,
                "accepted_classes": accepted_classes,
                "rejected_classes": rejected_classes,
                "rejected_stages": rejected_stages_list,
            })

    return traces, mem


# ---------------------------------------------------------------------------
# Phase 2: Filter to bottom quartile by margin
# ---------------------------------------------------------------------------


def filter_bottom_quartile(traces: List[dict], quantile: float = MARGIN_QUARTILE) -> List[dict]:
    """Return the most ambiguous traces (lowest margin)."""
    margins = [t["margin"] for t in traces]
    threshold = np.quantile(margins, quantile)
    bottom = [t for t in traces if t["margin"] <= threshold]
    return bottom


# ---------------------------------------------------------------------------
# Phase 3: Build contingency tables and run statistical tests
# ---------------------------------------------------------------------------


def build_contingency_analysis(traces: List[dict]) -> dict:
    """Analyze class distributions of accepted vs rejected candidates.

    For each trace in the ambiguous set:
    - Accepted set: classes of the final survivors
    - Rejected set: classes of the killed candidates

    Tests:
    1. Overall: do rejected and accepted carry different class distributions?
    2. Per-stage: breakdown by rejection stage
    3. Conditional: given ambiguity between classes A and B, do rejected
       candidates disproportionately carry one of them?
    """
    results = {}

    # ---- Test 1: Overall rejected vs accepted class distribution ----
    all_accepted_classes = Counter()
    all_rejected_classes = Counter()

    for t in traces:
        for c in t["accepted_classes"]:
            all_accepted_classes[c] += 1
        for c in t["rejected_classes"]:
            all_rejected_classes[c] += 1

    # Build contingency table over all classes present in either set
    all_classes = sorted(set(all_accepted_classes.keys()) | set(all_rejected_classes.keys()))

    if len(all_classes) < 2:
        results["overall"] = {
            "p_value": 1.0,
            "statistic": 0.0,
            "signal": False,
            "note": "Fewer than 2 classes in combined set",
        }
    else:
        # Contingency table: rows = [accepted, rejected], cols = classes
        accepted_row = [all_accepted_classes.get(c, 0) for c in all_classes]
        rejected_row = [all_rejected_classes.get(c, 0) for c in all_classes]
        table = np.array([accepted_row, rejected_row])

        # Filter out columns that are all-zero
        nonzero_cols = table.sum(axis=0) > 0
        table = table[:, nonzero_cols]

        if table.shape[1] < 2 or table.sum() < 10:
            results["overall"] = {
                "p_value": 1.0,
                "statistic": 0.0,
                "signal": False,
                "note": "Insufficient data for chi-squared test",
            }
        else:
            # Inspired by scipy.stats.chi2_contingency (BSD license)
            chi2, p_value, dof, expected = stats.chi2_contingency(table)
            results["overall"] = {
                "chi2": float(chi2),
                "p_value": float(p_value),
                "dof": int(dof),
                "n_classes": int(table.shape[1]),
                "n_accepted": int(sum(accepted_row)),
                "n_rejected": int(sum(rejected_row)),
                "signal": p_value < P_VALUE_THRESHOLD,
            }

    # ---- Test 2: Per-rejection-stage class distribution ----
    stage_classes: Dict[int, Counter] = defaultdict(Counter)
    for t in traces:
        for c, s in zip(t["rejected_classes"], t["rejected_stages"]):
            stage_classes[s][c] += 1

    stage_results = {}
    for stage_id in sorted(stage_classes.keys()):
        stage_counter = stage_classes[stage_id]
        stage_name = STAGE_NAMES.get(stage_id, f"unknown_{stage_id}")

        # Compare this stage's class distribution against accepted set
        stage_row = [stage_counter.get(c, 0) for c in all_classes]
        table_s = np.array([accepted_row, stage_row])
        nonzero_s = table_s.sum(axis=0) > 0
        table_s = table_s[:, nonzero_s]

        if table_s.shape[1] >= 2 and table_s.sum() >= 10 and table_s[1].sum() > 0:
            chi2_s, p_s, dof_s, _ = stats.chi2_contingency(table_s)
            stage_results[stage_name] = {
                "chi2": float(chi2_s),
                "p_value": float(p_s),
                "dof": int(dof_s),
                "n_rejected_this_stage": int(sum(stage_row)),
                "signal": p_s < P_VALUE_THRESHOLD,
            }
        else:
            stage_results[stage_name] = {
                "p_value": 1.0,
                "n_rejected_this_stage": int(sum(stage_row)),
                "signal": False,
                "note": "Insufficient data",
            }

    results["per_stage"] = stage_results

    # ---- Test 3: Conditional analysis — when prediction is ambiguous ----
    # For traces where top-2 accepted classes differ, check if rejected
    # candidates disproportionately carry one of those two classes
    conditional_signal_count = 0
    conditional_total = 0

    for t in traces:
        acc = t["accepted_classes"]
        rej = t["rejected_classes"]
        if len(acc) < 2 or len(rej) < 2:
            continue

        # Top-2 classes in accepted set
        acc_counter = Counter(acc)
        top2 = acc_counter.most_common(2)
        if len(top2) < 2:
            continue  # All accepted are same class — not ambiguous

        class_a, count_a = top2[0]
        class_b, count_b = top2[1]

        # Filter rejected to only these two classes
        rej_a = sum(1 for c in rej if c == class_a)
        rej_b = sum(1 for c in rej if c == class_b)

        if rej_a + rej_b < 3:
            continue

        conditional_total += 1

        # Under null: rejected should mirror accepted proportion
        # 2x2 contingency: [accepted_a, accepted_b], [rejected_a, rejected_b]
        table_c = np.array([[count_a, count_b], [rej_a, rej_b]])

        if table_c.min() == 0:
            # Use Fisher's exact test for small counts
            _, p_c = stats.fisher_exact(table_c)
        else:
            chi2_c, p_c, _, _ = stats.chi2_contingency(table_c, correction=True)

        if p_c < P_VALUE_THRESHOLD:
            conditional_signal_count += 1

    if conditional_total > 0:
        # Bonferroni-ish: what fraction of ambiguous traces show signal?
        signal_rate = conditional_signal_count / conditional_total

        # Global test: is the signal rate above chance?
        # Under null (no signal), each test has P_VALUE_THRESHOLD chance of
        # significance. Use binomial test.
        binom_p = stats.binom_test(
            conditional_signal_count,
            conditional_total,
            P_VALUE_THRESHOLD,
            alternative="greater",
        ) if hasattr(stats, "binom_test") else stats.binomtest(
            conditional_signal_count,
            conditional_total,
            P_VALUE_THRESHOLD,
            alternative="greater",
        ).pvalue

        results["conditional"] = {
            "n_ambiguous_traces": conditional_total,
            "n_significant": conditional_signal_count,
            "signal_rate": signal_rate,
            "expected_rate_under_null": P_VALUE_THRESHOLD,
            "binomial_p": float(binom_p),
            "signal": binom_p < P_VALUE_THRESHOLD,
        }
    else:
        results["conditional"] = {
            "n_ambiguous_traces": 0,
            "signal": False,
            "note": "No ambiguous traces with sufficient rejected candidates",
        }

    # ---- Supplementary: Top rejected classes in ambiguous traces ----
    ambiguous_rejected = Counter()
    ambiguous_accepted = Counter()
    for t in traces:
        acc = t["accepted_classes"]
        acc_counter = Counter(acc)
        if len(acc_counter) < 2:
            continue
        for c in t["rejected_classes"]:
            ambiguous_rejected[c] += 1
        for c in t["accepted_classes"]:
            ambiguous_accepted[c] += 1

    results["top_rejected_classes"] = ambiguous_rejected.most_common(10)
    results["top_accepted_classes"] = ambiguous_accepted.most_common(10)

    # ---- Distribution stats ----
    margins = [t["margin"] for t in traces]
    results["margin_stats"] = {
        "mean": float(np.mean(margins)),
        "std": float(np.std(margins)),
        "min": float(np.min(margins)),
        "max": float(np.max(margins)),
        "median": float(np.median(margins)),
    }

    correct_count = sum(1 for t in traces if t["correct"])
    results["accuracy_in_subset"] = 100.0 * correct_count / max(len(traces), 1)
    results["n_traces"] = len(traces)

    return results


# ---------------------------------------------------------------------------
# Phase 4: Per-domain pipeline
# ---------------------------------------------------------------------------


def run_domain(
    domain: Domain4,
    data_root: Path,
    device: torch.device,
) -> dict:
    """Full G37 pipeline for one domain."""
    print(f"\n{'='*60}")
    print(f"  DOMAIN: {domain.upper()}")
    print(f"{'='*60}")

    tr_f, tr_l, te_f, te_l, n_classes = g30_load_domain_features(domain, data_root)
    adapter = _load_g30_domain_adapter(domain, input_dim=tr_f.shape[1], device=device)

    print(f"  Train: {len(tr_f)} samples, Test: {len(te_f)} samples, {n_classes} classes")

    # Collect traces
    print("  [1/3] Building memory & collecting traces...")
    set_all_seeds(42)
    traces, mem = collect_traces(tr_f, tr_l, te_f, te_l, adapter, device)
    print(f"    Collected {len(traces)} traces")

    # Filter to bottom quartile
    print("  [2/3] Filtering to bottom-quartile by margin...")
    ambiguous = filter_bottom_quartile(traces)
    all_margins = [t["margin"] for t in traces]
    threshold = np.quantile(all_margins, MARGIN_QUARTILE)
    print(f"    Margin threshold: {threshold:.6f}")
    print(f"    Ambiguous traces: {len(ambiguous)} / {len(traces)}")
    print(f"    Full-set accuracy: {100 * sum(t['correct'] for t in traces) / len(traces):.2f}%")
    print(f"    Ambiguous-set accuracy: {100 * sum(t['correct'] for t in ambiguous) / max(len(ambiguous), 1):.2f}%")

    # Statistical analysis
    print("  [3/3] Running contingency analysis...")
    analysis = build_contingency_analysis(ambiguous)

    del mem
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "domain": domain,
        "n_traces_total": len(traces),
        "n_traces_ambiguous": len(ambiguous),
        "margin_threshold": float(threshold),
        "analysis": analysis,
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def print_verdict(results: Dict[Domain4, dict]) -> bool:
    """Print G37 results and determine PASS/KILL."""
    print("\n" + "=" * 70)
    print("  G37 — OFFLINE NEGATIVE EVIDENCE ANALYSIS")
    print("=" * 70)

    any_signal = False

    for domain in DOMAINS:
        r = results[domain]
        a = r["analysis"]
        print(f"\n  --- {domain.upper()} ---")
        print(f"  Traces: {r['n_traces_total']} total, {r['n_traces_ambiguous']} ambiguous (bottom 25%)")
        print(f"  Margin threshold: {r['margin_threshold']:.6f}")
        print(f"  Margin stats: mean={a['margin_stats']['mean']:.6f} "
              f"std={a['margin_stats']['std']:.6f} "
              f"median={a['margin_stats']['median']:.6f}")
        print(f"  Accuracy in ambiguous subset: {a['accuracy_in_subset']:.2f}%")

        # Overall test
        ov = a["overall"]
        if "chi2" in ov:
            tag = "SIGNAL" if ov["signal"] else "no signal"
            print(f"\n  Overall chi2 test:")
            print(f"    χ²={ov['chi2']:.2f}  p={ov['p_value']:.2e}  dof={ov['dof']}  "
                  f"classes={ov['n_classes']}  accepted={ov['n_accepted']}  "
                  f"rejected={ov['n_rejected']}  [{tag}]")
            if ov["signal"]:
                any_signal = True
        else:
            print(f"\n  Overall test: {ov.get('note', 'N/A')}")

        # Per-stage
        print(f"\n  Per-rejection-stage:")
        for stage_name, sr in a["per_stage"].items():
            n = sr["n_rejected_this_stage"]
            if "chi2" in sr:
                tag = "SIGNAL" if sr["signal"] else "—"
                print(f"    {stage_name:15s}: n={n:6d}  χ²={sr['chi2']:8.2f}  "
                      f"p={sr['p_value']:.2e}  [{tag}]")
                if sr["signal"]:
                    any_signal = True
            else:
                print(f"    {stage_name:15s}: n={n:6d}  {sr.get('note', 'N/A')}")

        # Conditional
        cond = a["conditional"]
        print(f"\n  Conditional test (ambiguous top-2 accepted classes):")
        if cond.get("n_ambiguous_traces", 0) > 0:
            tag = "SIGNAL" if cond["signal"] else "no signal"
            print(f"    Ambiguous traces tested: {cond['n_ambiguous_traces']}")
            print(f"    Significant (p<{P_VALUE_THRESHOLD}): {cond['n_significant']} "
                  f"({cond['signal_rate']:.1%} vs {cond['expected_rate_under_null']:.1%} expected)")
            print(f"    Binomial test p={cond['binomial_p']:.2e}  [{tag}]")
            if cond["signal"]:
                any_signal = True
        else:
            print(f"    {cond.get('note', 'N/A')}")

        # Top classes
        if a["top_rejected_classes"]:
            print(f"\n  Top rejected classes (ambiguous traces): "
                  f"{a['top_rejected_classes'][:5]}")
        if a["top_accepted_classes"]:
            print(f"  Top accepted classes (ambiguous traces):  "
                  f"{a['top_accepted_classes'][:5]}")

    # Final verdict
    verdict = "PASS" if any_signal else "KILL"
    print(f"\n{'='*70}")
    print(f"  G37 VERDICT: {verdict}")
    if any_signal:
        print("  At least one domain shows statistically significant asymmetry")
        print("  (p < 0.01) in rejected candidate class distributions.")
        print("  → Dark matter is NON-EPIPHENOMENAL. Proceed to G38.")
    else:
        print("  No domain shows significant asymmetry in rejected candidates.")
        print("  → Dark matter is EPIPHENOMENAL. Hypothesis dead.")
    print(f"{'='*70}")

    return any_signal


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="G37 — Offline Negative Evidence Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--device", default="auto")
    p.add_argument("--max-traces", type=int, default=100_000,
                   help="Target number of RetrievalTrace records per domain")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = device_from_arg(args.device)

    print("G37 — Offline Negative Evidence Analysis (Real Data)")
    print(f"Device:     {device}")
    print(f"Data root:  {args.data_root}")
    print(f"Max traces: {args.max_traces}")
    print(f"P-value:    {P_VALUE_THRESHOLD}")
    print(f"Margin Q:   bottom {MARGIN_QUARTILE:.0%}")

    results: Dict[Domain4, dict] = {}
    for domain in DOMAINS:
        results[domain] = run_domain(domain, args.data_root, device)

    passed = print_verdict(results)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
