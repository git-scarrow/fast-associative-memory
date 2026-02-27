#!/usr/bin/env python3
"""
MT-18 — VP-Tree Candidate Set Augmentation

Test whether a vantage-point tree over the prototype key space produces a
different and more informative candidate set than flat exhaustive cosine.

This is a retrieval quality experiment, not a latency optimization.
Hypothesis: tree traversal encodes manifold structure (bridge avoidance,
neighborhood coherence) that raw cosine discards, and that augmenting flat
candidates with tree-selected candidates improves accuracy, particularly on
low-confidence (Q1) queries.

Usage:
    PYTHONPATH=. .venv/bin/python benchmarks/mt18_vptree_augmentation.py

Stop condition: All 3 tables populated for all 7 domains. One session.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from fast_associative_memory import FastAssociativeMemory
from benchmarks.g30_adapter_sweep import load_domain_features as g30_load_domain_features
from benchmarks.g31_retrain_adapters import _load_g30_domain_adapter

SEED = 42
ADAPTER_DOMAINS = {"dogs", "birds", "cars", "aircraft"}
BROAD_K = 100       # candidate set size (matches flat pipeline)
FINAL_K = 25        # softmax vote window (matches production inference_k)
TEMP = 0.05         # softmax temperature (matches production)
MAX_QUERIES = 5000  # cap per domain to keep runtime reasonable


def set_seeds():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------------
# VP-Tree — pure Python/numpy, no external libraries
# ---------------------------------------------------------------------------

class _VPNode:
    __slots__ = ("vp_idx", "vp_vec", "threshold", "left", "right", "leaf_indices")

    def __init__(self):
        self.vp_idx: int = -1
        self.vp_vec: np.ndarray = None
        self.threshold: float = 0.0
        self.left: Optional[_VPNode] = None
        self.right: Optional[_VPNode] = None
        self.leaf_indices: Optional[np.ndarray] = None


class VPTree:
    """Vantage-point tree for cosine-distance nearest-neighbor search.

    Distance metric: d(a, b) = 1 - (a · b)   (assumes unit vectors).

    Build: O(N log N).  Query: O(log N) average, O(N) worst case.
    """

    LEAF_SIZE = 32  # switch to linear scan below this node size

    def __init__(self, keys: torch.Tensor, indices: torch.Tensor):
        """
        keys:    (N, D) normalized key vectors for occupied prototype slots.
        indices: (N,)  original slot indices into the prototype store.
        """
        keys_np = keys.cpu().float().numpy()       # (N, D)
        idx_np = indices.cpu().numpy().astype(np.int64)
        self._keys = keys_np
        self._slot_indices = idx_np
        self._root = self._build(np.arange(len(idx_np)))

    # -- build ---------------------------------------------------------------

    def _build(self, node_pts: np.ndarray) -> _VPNode:
        node = _VPNode()
        if len(node_pts) <= self.LEAF_SIZE:
            node.leaf_indices = node_pts
            return node

        # Pick a random vantage point
        vp_pos = int(random.randrange(len(node_pts)))
        vp_local = node_pts[vp_pos]
        node.vp_idx = int(vp_local)
        node.vp_vec = self._keys[vp_local]          # (D,)

        # Compute cosine distances from VP to all other points
        rest = node_pts[node_pts != vp_local]
        if len(rest) == 0:
            node.leaf_indices = node_pts
            return node

        dots = self._keys[rest] @ node.vp_vec       # (M,)
        dists = 1.0 - dots                           # cosine distance

        # Split at median
        median_d = float(np.median(dists))
        node.threshold = median_d

        left_mask = dists <= median_d
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            node.leaf_indices = node_pts
            return node

        node.left = self._build(rest[left_mask])
        node.right = self._build(rest[right_mask])
        return node

    # -- query ---------------------------------------------------------------

    def query(self, q: torch.Tensor, k: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (slot_indices, cosine_distances) of k nearest neighbors.

        q: (D,) normalized query vector.
        """
        q_np = q.cpu().float().numpy()

        # Max-heap: store (-distance, local_array_index) so we can pop the
        # farthest candidate when we have more than k.
        heap: List[Tuple[float, int]] = []   # (neg_dist, local_idx)

        self._search(self._root, q_np, k, heap)

        # Sort ascending by distance
        heap.sort(key=lambda x: -x[0])       # negate back to sort ascending
        local_indices = np.array([h[1] for h in heap], dtype=np.int64)
        dists_out = np.array([-h[0] for h in heap], dtype=np.float32)

        slot_indices = torch.from_numpy(self._slot_indices[local_indices])
        dists_tensor = torch.from_numpy(dists_out)
        return slot_indices, dists_tensor

    def _search(self, node: _VPNode, q: np.ndarray, k: int,
                heap: List[Tuple[float, int]]):
        if node is None:
            return

        # Leaf node — linear scan
        if node.leaf_indices is not None:
            for li in node.leaf_indices:
                d = float(1.0 - float(self._keys[li] @ q))
                self._heap_push(heap, d, int(li), k)
            return

        # Distance from query to vantage point
        d_vp = float(1.0 - float(node.vp_vec @ q))
        self._heap_push(heap, d_vp, node.vp_idx, k)

        tau = heap[0][0] * -1 if heap else float("inf")  # farthest in heap (positive dist)
        if len(heap) < k:
            tau = float("inf")

        if d_vp < node.threshold:
            # query is closer to left (inner ball)
            self._search(node.left, q, k, heap)
            # recompute tau
            tau = -heap[0][0] if heap else float("inf")
            if len(heap) < k:
                tau = float("inf")
            if d_vp - node.threshold <= tau:
                self._search(node.right, q, k, heap)
        else:
            self._search(node.right, q, k, heap)
            tau = -heap[0][0] if heap else float("inf")
            if len(heap) < k:
                tau = float("inf")
            if node.threshold - d_vp <= tau:
                self._search(node.left, q, k, heap)

    @staticmethod
    def _heap_push(heap: list, dist: float, idx: int, k: int):
        """Push (-dist, idx) into a max-heap of size k (keeps k smallest dists)."""
        import heapq
        heapq.heappush(heap, (-dist, idx))
        if len(heap) > k:
            heapq.heappop(heap)


# ---------------------------------------------------------------------------
# Data loading (mirrors mt17_deadweight_ablation.py)
# ---------------------------------------------------------------------------

def load_domain(domain: str, data_root: Path):
    """Returns (train_feats, train_labels, test_feats, test_labels, n_classes, adapter_or_None)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if domain in ADAPTER_DOMAINS:
        tr_f, tr_l, te_f, te_l, nc = g30_load_domain_features(domain, data_root)
        adapter = _load_g30_domain_adapter(domain, input_dim=tr_f.shape[1], device=device)
        return tr_f, tr_l, te_f, te_l, nc, adapter

    if domain == "cifar100":
        root = Path("feature_cache_vitl14")
        train = torch.load(root / "cifar100_dinov2_train.pt", map_location="cpu", weights_only=True)
        test = torch.load(root / "cifar100_dinov2_test.pt", map_location="cpu", weights_only=True)
        return (train["embeds"].float(), train["labels"].long(),
                test["embeds"].float(), test["labels"].long(),
                100, None)

    if domain == "flowers":
        root = data_root / "flowers102" / "feature_cache"
        train = torch.load(root / "flowers102_train.pt", map_location="cpu", weights_only=True)
        test = torch.load(root / "flowers102_test.pt", map_location="cpu", weights_only=True)
        fk = "embeds" if "embeds" in train else "features"
        return (train[fk].float(), train["labels"].long(),
                test[fk].float(), test["labels"].long(),
                int(train["labels"].max().item() + 1), None)

    if domain == "imagenet_r":
        root = Path("feature_cache_inr_vitl14")
        train = torch.load(root / "imagenetr_dinov2_train.pt", map_location="cpu", weights_only=True)
        test = torch.load(root / "imagenetr_dinov2_test.pt", map_location="cpu", weights_only=True)
        fk = "embeds" if "embeds" in train else "features"
        return (train[fk].float(), train["labels"].long(),
                test[fk].float(), test["labels"].long(),
                int(train["labels"].max().item() + 1), None)

    raise ValueError(f"Unknown domain: {domain!r}")


# ---------------------------------------------------------------------------
# Softmax vote helper (standalone, no ContinuousCAM)
# ---------------------------------------------------------------------------

@torch.no_grad()
def softmax_vote(
    q_norm: torch.Tensor,           # (D,)
    cand_keys: torch.Tensor,        # (C, D) normalized
    cand_values: torch.Tensor,      # (C, nc) one-hot float
    temp: float = TEMP,
    top_k: int = FINAL_K,
) -> torch.Tensor:
    """Compute the softmax-weighted class vote for a single query."""
    sims = cand_keys @ q_norm                        # (C,)
    k = min(top_k, len(sims))
    topk_vals, topk_idx = sims.topk(k)
    weights = F.softmax(topk_vals / temp, dim=0)     # (k,)
    vote = (weights.unsqueeze(-1) * cand_values[topk_idx]).sum(0)  # (nc,)
    return vote


# ---------------------------------------------------------------------------
# Per-domain benchmark
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_domain(domain: str, data_root: Path, device: torch.device,
               out_dir: Path) -> dict:
    print(f"\n{'=' * 65}")
    print(f"  DOMAIN: {domain.upper()}")
    print(f"{'=' * 65}")

    tr_f, tr_l, te_f, te_l, nc, adapter = load_domain(domain, data_root)
    print(f"  Train: {len(tr_f)}, Test: {len(te_f)}, Classes: {nc}")

    # ---- Build FAM (for projection + prototype population) ----
    mem = FastAssociativeMemory(
        input_dim=int(tr_f.shape[1]),
        value_dim=nc,
        core_entries=50000, core_vigilance=0.85,
        hebb_lr=0.1, key_lr=0.05,
        inference_k=FINAL_K, inference_temp=TEMP,
        use_lfu=True, adapter=adapter,
    ).to(device)

    x_train = tr_f.to(device)
    y_train = tr_l.to(device)
    for i in range(0, len(x_train), 256):
        mem.learn_local(x_train[i:i + 256], y_train[i:i + 256])

    core = mem.core_cam
    n_proto = int(core.occupied.sum().item())
    print(f"  Prototypes stored: {n_proto}")

    # ---- Prototype key / value arrays ----
    occ_mask = core.occupied                              # (max_entries,)
    occ_slots = occ_mask.nonzero(as_tuple=True)[0]        # (N,)
    proto_keys_norm = core._keys_norm[occ_slots].float()  # (N, D)
    proto_values = core.values[occ_slots].float()         # (N, nc)
    proto_slot_idx = occ_slots                            # (N,) original indices

    # ---- Build VP-tree ----
    print("  Building VP-tree...", end="", flush=True)
    vptree = VPTree(proto_keys_norm, proto_slot_idx)
    print(" done.")

    # ---- Eval queries ----
    query_feats = te_f
    query_labels = te_l
    n_q = min(len(query_feats), MAX_QUERIES)
    query_feats = query_feats[:n_q]
    query_labels = query_labels[:n_q]

    # ---- Per-query results ----
    results = []
    for qi in range(n_q):
        raw_q = query_feats[qi].to(device).unsqueeze(0)
        proj_q = mem._project(raw_q).squeeze(0)
        q_norm = F.normalize(proj_q.float(), dim=-1)   # (D,)
        true_cls = int(query_labels[qi].item())

        # --- 1. Flat top-100 by cosine ---
        flat_sims = proto_keys_norm @ q_norm             # (N,)
        flat_topk_idx = flat_sims.topk(min(BROAD_K, len(flat_sims))).indices  # local indices
        flat_slot_ids = set(proto_slot_idx[flat_topk_idx].cpu().tolist())

        # Softmax margin for quartile stratification (use flat top-FINAL_K)
        flat_top_sims = flat_sims[flat_topk_idx]
        k_vote = min(FINAL_K, len(flat_topk_idx))
        top_vote_sims, _ = flat_top_sims.topk(k_vote)
        w = F.softmax(top_vote_sims / TEMP, dim=0)
        sm_top2 = w.topk(min(2, len(w))).values
        margin = float((sm_top2[0] - sm_top2[1]).item()) if len(sm_top2) > 1 else float(sm_top2[0].item())

        # --- 2. VP-tree top-100 ---
        tree_slots, tree_dists = vptree.query(q_norm, k=BROAD_K)  # (M,), (M,)
        tree_slot_ids = set(tree_slots.cpu().tolist())

        # --- 3. Overlap ---
        union_ids = flat_slot_ids | tree_slot_ids
        inter_ids = flat_slot_ids & tree_slot_ids
        jaccard_100 = len(inter_ids) / len(union_ids) if union_ids else 1.0

        # Jaccard at top-25 and top-5
        flat_top25_ids = set(proto_slot_idx[flat_topk_idx[:25]].cpu().tolist())
        tree_top25_ids = set(tree_slots[:25].cpu().tolist())
        j25 = len(flat_top25_ids & tree_top25_ids) / len(flat_top25_ids | tree_top25_ids) \
              if flat_top25_ids | tree_top25_ids else 1.0

        flat_top5_ids = set(proto_slot_idx[flat_topk_idx[:5]].cpu().tolist())
        tree_top5_ids = set(tree_slots[:5].cpu().tolist())
        j5 = len(flat_top5_ids & tree_top5_ids) / len(flat_top5_ids | tree_top5_ids) \
             if flat_top5_ids | tree_top5_ids else 1.0

        # --- 4. Votes for each candidate set ---
        def _vote_from_slots(slot_ids_set):
            local_idx = [i for i, s in enumerate(proto_slot_idx.cpu().tolist()) if s in slot_ids_set]
            if not local_idx:
                return torch.zeros(nc, device=device)
            local_t = torch.tensor(local_idx, device=device)
            cand_k = proto_keys_norm[local_t]
            cand_v = proto_values[local_t]
            return softmax_vote(q_norm, cand_k, cand_v)

        flat_vote = _vote_from_slots(flat_slot_ids)
        tree_vote = _vote_from_slots(tree_slot_ids)
        union_vote = _vote_from_slots(union_ids)

        flat_pred = int(flat_vote.argmax().item())
        tree_pred = int(tree_vote.argmax().item())
        union_pred = int(union_vote.argmax().item())

        results.append({
            "true_cls": true_cls,
            "flat_pred": flat_pred,
            "tree_pred": tree_pred,
            "union_pred": union_pred,
            "margin": margin,
            "jaccard_100": jaccard_100,
            "jaccard_25": j25,
            "jaccard_5": j5,
            "flat_top5_classes": sorted(flat_top5_ids),
            "tree_top5_classes": sorted(tree_top5_ids),
        })

    # ---- Aggregate ----
    margins = np.array([r["margin"] for r in results])
    q_bounds = np.percentile(margins, [25, 50, 75])

    def quartile(m):
        if m <= q_bounds[0]:
            return 0
        elif m <= q_bounds[1]:
            return 1
        elif m <= q_bounds[2]:
            return 2
        return 3

    # Per-quartile accuracy
    q_stats = {cond: [{"n": 0, "corr": 0} for _ in range(4)]
               for cond in ("flat", "tree", "union")}

    for r in results:
        q = quartile(r["margin"])
        for cond in ("flat", "tree", "union"):
            q_stats[cond][q]["n"] += 1
            if r[f"{cond}_pred"] == r["true_cls"]:
                q_stats[cond][q]["corr"] += 1

    def qacc(cond, q):
        s = q_stats[cond][q]
        return 100.0 * s["corr"] / s["n"] if s["n"] > 0 else float("nan")

    def overall_acc(cond):
        total = sum(q_stats[cond][q]["n"] for q in range(4))
        corr = sum(q_stats[cond][q]["corr"] for q in range(4))
        return 100.0 * corr / total if total > 0 else float("nan")

    # Jaccard averages
    j100_all = np.mean([r["jaccard_100"] for r in results])
    j25_all = np.mean([r["jaccard_25"] for r in results])
    j5_all = np.mean([r["jaccard_5"] for r in results])

    print(f"\n  Table 1 — Candidate set overlap (mean Jaccard):")
    print(f"  {'top-100':>10} {'top-25':>10} {'top-5':>10}")
    print(f"  {j100_all:>10.4f} {j25_all:>10.4f} {j5_all:>10.4f}")

    print(f"\n  Table 2 — Accuracy by confidence quartile:")
    header = f"  {'Condition':>8}  {'Q1':>7} {'Q2':>7} {'Q3':>7} {'Q4':>7} {'Overall':>8}"
    print(header)
    for cond in ("flat", "tree", "union"):
        row = f"  {cond:>8}  "
        for q in range(4):
            row += f"{qacc(cond, q):>6.1f}% "
        row += f"{overall_acc(cond):>7.1f}%"
        print(row)

    # ---- Aircraft class 38/37 qualitative analysis ----
    bridge_rows = []
    if domain == "aircraft":
        print(f"\n  Table 3 — Aircraft class 38/37 bridge analysis:")
        target_classes = {37, 38}
        cands = [r for r in results if r["true_cls"] in target_classes]
        # Look for queries where flat and tree top-5 differ
        diff_cands = [r for r in cands
                      if set(r["flat_top5_classes"]) != set(r["tree_top5_classes"])]
        print(f"  Queries in classes 37/38: {len(cands)}, "
              f"with differing top-5: {len(diff_cands)}")
        print(f"\n  {'True':>5} {'FlatPred':>9} {'TreePred':>9} {'UnionPred':>10} "
              f"{'FlatTop5Cls':>20} {'TreeTop5Cls':>20}")
        for r in diff_cands[:20]:
            flat5 = [core.values[s].float().argmax().item()
                     for s in r["flat_top5_classes"]
                     if 0 <= s < core.max_entries and core.occupied[s].item()]
            tree5 = [core.values[s].float().argmax().item()
                     for s in r["tree_top5_classes"]
                     if 0 <= s < core.max_entries and core.occupied[s].item()]
            print(f"  {r['true_cls']:>5} {r['flat_pred']:>9} {r['tree_pred']:>9} "
                  f"{r['union_pred']:>10} {str(sorted(flat5)):>20} {str(sorted(tree5)):>20}")
            bridge_rows.append({
                "true_cls": r["true_cls"],
                "flat_pred": r["flat_pred"],
                "tree_pred": r["tree_pred"],
                "union_pred": r["union_pred"],
                "flat_top5_cls": flat5,
                "tree_top5_cls": tree5,
            })

    domain_result = {
        "domain": domain,
        "n_queries": n_q,
        "n_prototypes": n_proto,
        "jaccard_100": float(j100_all),
        "jaccard_25": float(j25_all),
        "jaccard_5": float(j5_all),
        "accuracy": {
            cond: {
                "overall": float(overall_acc(cond)),
                "Q1": float(qacc(cond, 0)),
                "Q2": float(qacc(cond, 1)),
                "Q3": float(qacc(cond, 2)),
                "Q4": float(qacc(cond, 3)),
            }
            for cond in ("flat", "tree", "union")
        },
        "bridge_analysis": bridge_rows,
    }

    del mem
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return domain_result


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def print_summary(all_results: list):
    print("\n" + "=" * 80)
    print("  MT-18 SUMMARY")
    print("=" * 80)

    # Table 1: Jaccard overlap
    print("\n  Table 1: Per-domain candidate set overlap (mean Jaccard)")
    print(f"  {'Domain':<14} {'Jaccard@100':>12} {'Jaccard@25':>11} {'Jaccard@5':>10}")
    print(f"  {'-' * 50}")
    for r in all_results:
        print(f"  {r['domain']:<14} {r['jaccard_100']:>12.4f} {r['jaccard_25']:>11.4f} "
              f"{r['jaccard_5']:>10.4f}")

    # Table 2: Accuracy comparison
    print("\n  Table 2: Accuracy comparison (flat vs tree vs union) by quartile")
    print(f"  {'Domain':<14} {'Cond':>6}  {'Q1':>7} {'Q2':>7} {'Q3':>7} {'Q4':>7} {'Overall':>8}")
    print(f"  {'-' * 65}")
    for r in all_results:
        for cond in ("flat", "tree", "union"):
            a = r["accuracy"][cond]
            print(f"  {r['domain']:<14} {cond:>6}  "
                  f"{a['Q1']:>6.1f}% {a['Q2']:>6.1f}% {a['Q3']:>6.1f}% {a['Q4']:>6.1f}% "
                  f"{a['overall']:>7.1f}%")
        print()

    # Verdict
    print("\n  VERDICT:")
    for r in all_results:
        j100 = r["jaccard_100"]
        delta_union_q1 = r["accuracy"]["union"]["Q1"] - r["accuracy"]["flat"]["Q1"]
        delta_union_overall = r["accuracy"]["union"]["overall"] - r["accuracy"]["flat"]["overall"]
        if j100 > 0.95:
            verdict = "Flat sufficient (J@100 > 0.95)"
        elif delta_union_q1 >= 0.5:
            verdict = f"Signal real: union Q1 +{delta_union_q1:.2f}pp"
        elif delta_union_q1 < 0.0:
            verdict = f"Tree lossy: union Q1 {delta_union_q1:.2f}pp"
        else:
            verdict = f"Marginal: union Q1 {delta_union_q1:+.2f}pp, overall {delta_union_overall:+.2f}pp"
        print(f"    {r['domain']:<14}: {verdict}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path("data")

    out_dir = Path("results/mt18_vptree_augmentation")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  MT-18 — VP-Tree Candidate Set Augmentation")
    print(f"  Device: {device}")
    print("=" * 65)

    DOMAINS = ["dogs", "birds", "cars", "aircraft", "cifar100", "flowers", "imagenet_r"]

    all_results = []
    for domain in DOMAINS:
        try:
            result = run_domain(domain, data_root, device, out_dir)
            all_results.append(result)
        except Exception as e:
            print(f"\n  {domain} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print_summary(all_results)

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
