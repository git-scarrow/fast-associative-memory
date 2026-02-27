#!/usr/bin/env python3
"""
Jacobian Effective Dimensionality Analysis
==========================================
Measures the local sensitivity of retrieval logits to perturbations in the
1024-d query key space via Jacobian SVD analysis.

Answers: how many independent directions does the memory actually use?
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from associative_core import ContinuousCAM
from fast_associative_memory import FastAssociativeMemory
from adapter import MetricAdapter

# ── Config ──────────────────────────────────────────────────────────────────
N_QUERIES = 500
TOP_K_CANDIDATES = 100  # fixed candidate set size for Jacobian
RANK_THRESHOLD = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "stanford_dogs"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "jacobian_effective_dim"
ADAPTER_PATH = Path(__file__).resolve().parent.parent / "adapters"


def load_dogs_data():
    train = torch.load(DATA_DIR / "stanford_dogs_train.pt", weights_only=False)
    test = torch.load(DATA_DIR / "stanford_dogs_test.pt", weights_only=False)
    return train["features"], train["labels"], test["features"], test["labels"]


def build_fam(num_classes, adapter=None):
    """Build FAM with production-like settings."""
    fam = FastAssociativeMemory(
        input_dim=1024,
        value_dim=num_classes,
        core_entries=15000,
        core_vigilance=0.85,
        hebb_lr=0.1,
        key_lr=0.05,
        inference_k=25,
        inference_temp=0.05,
        whitening_dim=0,
        use_lfu=True,
        use_bfloat16=False,
        adapter=adapter,
    )
    return fam


def train_fam(fam, features, labels):
    """Ingest all training data into the prototype store."""
    fam.eval()
    bs = 256
    for i in range(0, len(features), bs):
        batch_f = features[i:i+bs].to(DEVICE)
        batch_l = labels[i:i+bs].to(DEVICE)
        fam.learn_local(batch_f, batch_l)
    n_occ = fam.core_cam.occupied.sum().item()
    print(f"  Prototypes stored: {n_occ}")
    return n_occ


def differentiable_retrieval_logits(cam: ContinuousCAM, query: torch.Tensor,
                                     adapter=None, K=100):
    """
    Differentiable proxy for retrieval logits.

    1. Detached pass: find top-K candidate indices.
    2. Differentiable pass: compute Mahalanobis-reranked similarities
       for those fixed candidates, apply CSLS, then softmax-weighted vote
       to produce class logits.

    Returns: (logits, ) where logits is (K,) re-ranked similarity scores
             (pre-softmax, which are the retrieval logits).
    """
    # query: (1, key_dim), requires_grad=True
    valid_idx = cam.occupied.nonzero(as_tuple=True)[0]
    n_valid = len(valid_idx)
    K = min(K, n_valid)

    # Step 1: Detached candidate selection
    with torch.no_grad():
        q_det = F.normalize(query.detach().float(), dim=-1)
        sims_det = q_det @ cam._keys_norm[valid_idx].T  # (1, N)
        _, broad_locs = sims_det.topk(K, dim=1)  # (1, K)
        broad_slots = valid_idx[broad_locs.squeeze(0)]  # (K,)

    # Step 2: Differentiable scoring against fixed candidates
    # Get candidate data (detached, treated as constants)
    candidate_keys = cam.keys[broad_slots].float().detach()  # (K, key_dim)
    candidate_classes = cam.values[broad_slots].float().detach().argmax(dim=-1)  # (K,)
    candidate_density = cam.prototype_density[broad_slots].detach()  # (K,)

    # Per-class variance (detached)
    local_vars = cam.class_vars[candidate_classes].clamp(min=1e-4).detach()  # (K, key_dim)
    inv_std = local_vars.rsqrt()  # (K, key_dim)

    # Differentiable: scale query by per-candidate inv_std
    q_expanded = query.expand(K, -1)  # (K, key_dim) - grad flows through query
    scaled_Q = q_expanded * inv_std  # (K, key_dim)
    scaled_K = candidate_keys * inv_std  # (K, key_dim)

    # Cosine similarity in Mahalanobis-scaled space
    reranked_sims = F.cosine_similarity(scaled_Q, scaled_K, dim=-1)  # (K,)

    # CSLS correction
    csls_sims = 2.0 * reranked_sims - cam.csls_weight * candidate_density  # (K,)

    # Return the CSLS-corrected scores as the "retrieval logits"
    return csls_sims


def compute_jacobian_svd(cam, query, adapter=None, K=100):
    """Compute Jacobian of retrieval logits w.r.t. query and return SVD."""
    query = query.clone().detach().requires_grad_(True)  # (1, D)

    logits = differentiable_retrieval_logits(cam, query, adapter=adapter, K=K)
    # logits: (K,)

    # Compute Jacobian row-by-row
    D = query.shape[1]
    K_actual = logits.shape[0]
    J = torch.zeros(K_actual, D, device=DEVICE)

    for i in range(K_actual):
        if query.grad is not None:
            query.grad.zero_()
        logits[i].backward(retain_graph=(i < K_actual - 1))
        J[i] = query.grad.squeeze(0).clone()

    # SVD
    S = torch.linalg.svdvals(J.float())
    # Numerical rank
    rank = (S > RANK_THRESHOLD * S[0]).sum().item() if S[0] > 0 else 0

    return S.cpu().numpy(), rank


def run_analysis(cam, queries, labels, adapter_name="none", K=100):
    """Run Jacobian analysis on sampled queries."""
    N = len(queries)
    all_svs = []
    all_ranks = []

    t0 = time.time()
    for i in range(N):
        q = queries[i:i+1].to(DEVICE).float()
        svs, rank = compute_jacobian_svd(cam, q, K=K)
        all_svs.append(svs)
        all_ranks.append(rank)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N - i - 1)
            print(f"  [{adapter_name}] {i+1}/{N}  "
                  f"median_rank={np.median(all_ranks):.0f}  "
                  f"ETA={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"  [{adapter_name}] Done in {elapsed:.1f}s")

    # Pad SVD arrays to same length
    max_len = max(len(s) for s in all_svs)
    padded = np.zeros((N, max_len))
    for i, s in enumerate(all_svs):
        padded[i, :len(s)] = s

    return {
        "singular_values": padded,  # (N, max_sv_dim)
        "ranks": np.array(all_ranks),
        "adapter": adapter_name,
    }


def plot_results(results_list, out_dir):
    """Generate all plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Rank histogram ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for res in results_list:
        ax.hist(res["ranks"], bins=30, alpha=0.6, label=res["adapter"],
                edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Numerical Rank (threshold=1e-4)")
    ax.set_ylabel("Count")
    ax.set_title(f"Effective Dimensionality Distribution (N={len(results_list[0]['ranks'])})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "rank_hist.png", dpi=150)
    plt.close(fig)

    # ── 2. Mean singular value spectrum ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for res in results_list:
        svs = res["singular_values"]
        mean_svs = svs.mean(axis=0)
        # Clip zeros for log plot
        mean_svs_clipped = np.clip(mean_svs, 1e-12, None)
        ax.semilogy(mean_svs_clipped, label=res["adapter"], linewidth=1.5)
    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Singular Value (log scale)")
    ax.set_title("Mean Singular Value Spectrum of Retrieval Jacobian")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "singular_spectrum_mean.png", dpi=150)
    plt.close(fig)

    # ── 3. Example per-query spectra ──
    ex_dir = out_dir / "singular_spectrum_examples"
    ex_dir.mkdir(exist_ok=True)
    for res in results_list:
        for idx in [0, 50, 100, 250, 499]:
            if idx >= len(res["singular_values"]):
                continue
            svs = res["singular_values"][idx]
            svs_clipped = np.clip(svs, 1e-12, None)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.semilogy(svs_clipped, linewidth=1)
            rank = res["ranks"][idx]
            ax.set_title(f"Query {idx} — rank={rank} [{res['adapter']}]")
            ax.set_xlabel("SV Index")
            ax.set_ylabel("Singular Value")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(ex_dir / f"query_{idx}_{res['adapter']}.png", dpi=100)
            plt.close(fig)

    # ── 4. Cumulative energy plot ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for res in results_list:
        svs = res["singular_values"]
        mean_svs = svs.mean(axis=0)
        energy = np.cumsum(mean_svs ** 2)
        if energy[-1] > 0:
            energy /= energy[-1]
        ax.plot(energy, label=res["adapter"], linewidth=1.5)
    ax.axhline(0.95, color="red", linestyle="--", alpha=0.5, label="95% energy")
    ax.axhline(0.99, color="orange", linestyle="--", alpha=0.5, label="99% energy")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Energy Fraction")
    ax.set_title("Cumulative Energy of Jacobian Singular Values")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_energy.png", dpi=150)
    plt.close(fig)


def save_results_json(results_list, out_dir):
    out = {}
    for res in results_list:
        name = res["adapter"]
        ranks = res["ranks"]
        svs = res["singular_values"]
        # Top 128 SVs per query
        top128 = svs[:, :128].tolist()

        # 95% energy dimension
        mean_svs = svs.mean(axis=0)
        energy = np.cumsum(mean_svs ** 2)
        if energy[-1] > 0:
            energy_norm = energy / energy[-1]
            dim_95 = int(np.searchsorted(energy_norm, 0.95)) + 1
            dim_99 = int(np.searchsorted(energy_norm, 0.99)) + 1
        else:
            dim_95 = dim_99 = 0

        out[name] = {
            "n_queries": len(ranks),
            "mean_rank": float(ranks.mean()),
            "median_rank": float(np.median(ranks)),
            "std_rank": float(ranks.std()),
            "min_rank": int(ranks.min()),
            "max_rank": int(ranks.max()),
            "effective_dim_ratio_mean": float(ranks.mean() / 1024),
            "dim_95pct_energy": dim_95,
            "dim_99pct_energy": dim_99,
            "per_query_ranks": ranks.tolist(),
            "per_query_top128_svs": top128,
        }
    with open(out_dir / "results.json", "w") as f:
        json.dump(out, f, indent=2)


def try_load_adapter(name="stanford_dogs"):
    """Try to load a saved adapter for the domain.

    Known checkpoints (from G34):
      Dogs:     adapter_trained.pt
      Birds:    adapter_cub200_birds.pt
      Cars:     data/stanford_cars/adapter_stanford_cars.pt
      Aircraft: adapter_fgvc_aircraft.pt
    """
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "adapter_trained.pt",  # Dogs adapter (G31)
        root / f"adapter_{name}.pt",
        root / f"{name}_adapter.pt",
        root / "data" / name / f"adapter_{name}.pt",
    ]
    for p in candidates:
        if p.exists():
            print(f"  Found adapter: {p}")
            state = torch.load(p, weights_only=False)
            if isinstance(state, dict) and "net.0.weight" in state:
                in_dim = state["net.0.weight"].shape[1]
                # Find the last layer's output dim
                weight_keys = sorted([k for k in state if "weight" in k])
                last_weight = state[weight_keys[-1]]
                out_dim = last_weight.shape[0]
                # Count linear layers
                n_layers = len(weight_keys)
                if n_layers == 1:
                    adapter = MetricAdapter(in_dim, out_dim)
                else:
                    hidden = state["net.0.weight"].shape[0]
                    adapter = MetricAdapter(in_dim, out_dim, hidden_dim=hidden)
                adapter.load_state_dict(state)
                return adapter
            elif isinstance(state, MetricAdapter):
                return state
    return None


def main():
    print("=" * 60)
    print("Jacobian Effective Dimensionality Analysis")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading Stanford Dogs data...")
    train_f, train_l, test_f, test_l = load_dogs_data()
    num_classes = int(train_l.max().item()) + 1
    print(f"  Train: {train_f.shape}, Test: {test_f.shape}, Classes: {num_classes}")

    # Sample N queries uniformly
    N = min(N_QUERIES, len(test_f))
    perm = torch.randperm(len(test_f))[:N]
    query_features = test_f[perm]
    query_labels = test_l[perm]
    print(f"  Sampled {N} queries for analysis")

    results_list = []

    # ── Run 1: No adapter (baseline) ──
    print("\n--- Condition: No Adapter ---")
    fam_base = build_fam(num_classes, adapter=None).to(DEVICE)
    train_fam(fam_base, train_f, train_l)

    res_base = run_analysis(fam_base.core_cam, fam_base._project(query_features.to(DEVICE)),
                            query_labels, adapter_name="no_adapter", K=TOP_K_CANDIDATES)
    results_list.append(res_base)
    print(f"  Mean rank: {res_base['ranks'].mean():.1f} / 1024  "
          f"(ratio: {res_base['ranks'].mean()/1024:.3f})")
    print(f"  Median rank: {np.median(res_base['ranks']):.1f}")

    # ── Run 2: With adapter (if available) ──
    adapter = try_load_adapter("stanford_dogs")
    if adapter is not None:
        print("\n--- Condition: With Adapter ---")
        adapter = adapter.to(DEVICE).eval()
        fam_adapt = build_fam(num_classes, adapter=adapter).to(DEVICE)
        train_fam(fam_adapt, train_f, train_l)

        # Project queries through adapter
        with torch.no_grad():
            projected_queries = fam_adapt._project(query_features.to(DEVICE))

        res_adapt = run_analysis(fam_adapt.core_cam, projected_queries,
                                 query_labels, adapter_name="with_adapter", K=TOP_K_CANDIDATES)
        results_list.append(res_adapt)
        print(f"  Mean rank: {res_adapt['ranks'].mean():.1f} / 1024  "
              f"(ratio: {res_adapt['ranks'].mean()/1024:.3f})")
        print(f"  Median rank: {np.median(res_adapt['ranks']):.1f}")
    else:
        print("\n  No saved adapter found — skipping adapter condition.")

    # ── Generate outputs ──
    print("\nGenerating plots...")
    plot_results(results_list, OUT_DIR)

    print("Saving results.json...")
    save_results_json(results_list, OUT_DIR)

    # ── Summary table ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Condition':<20} {'Mean Rank':>10} {'Median':>8} {'Ratio':>8} "
          f"{'95% E':>7} {'99% E':>7}")
    print("-" * 60)
    for res in results_list:
        name = res["adapter"]
        ranks = res["ranks"]
        svs = res["singular_values"]
        mean_svs = svs.mean(axis=0)
        energy = np.cumsum(mean_svs ** 2)
        if energy[-1] > 0:
            en = energy / energy[-1]
            d95 = int(np.searchsorted(en, 0.95)) + 1
            d99 = int(np.searchsorted(en, 0.99)) + 1
        else:
            d95 = d99 = 0
        print(f"{name:<20} {ranks.mean():>10.1f} {np.median(ranks):>8.1f} "
              f"{ranks.mean()/1024:>8.3f} {d95:>7d} {d99:>7d}")
    print("=" * 60)

    print(f"\nArtifacts saved to: {OUT_DIR}")
    print("  rank_hist.png")
    print("  singular_spectrum_mean.png")
    print("  cumulative_energy.png")
    print("  singular_spectrum_examples/")
    print("  results.json")


if __name__ == "__main__":
    main()
