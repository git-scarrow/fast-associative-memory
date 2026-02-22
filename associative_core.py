"""
associative_core.py — Online prototype memory with EMA updates,
temperature-scaled soft-kNN retrieval, and LFU-LRU hybrid eviction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


def _make_orthogonal_prototypes(num_vectors: int, dim: int) -> torch.Tensor:
    """Creates fixed unit-norm vectors with near-orthogonal rows."""
    if num_vectors <= dim:
        q, _ = torch.linalg.qr(torch.randn(dim, dim))
        return q[:, :num_vectors].T.contiguous()

    # If more vectors than dimensions, keep vectors normalized and diverse.
    vecs = torch.randn(num_vectors, dim)
    return F.normalize(vecs, dim=-1)


class ContinuousCAM(nn.Module):
    """Prototype memory with EMA plasticity.

    All operations use batched matrix multiplications for GPU efficiency.
    Buffers follow nn.Module device placement (register_buffer), so
    calling `.to(device)` or `.cuda()` moves the entire table to VRAM.
    """
    def __init__(self, key_dim: int, value_dim: int, max_entries: int = 2048,
                 vigilance: float = 0.85, hebb_lr: float = 0.1,
                 aging_time: float = 1e9, flood_scale: float = 0.15,
                 immutable_keys: bool = False, use_lfu: bool = False,
                 use_bfloat16: bool = False, key_lr: float = 0.05,
                 inference_k: int = 20, inference_temp: float = 0.05,
                 ema_beta: float = 0.05):
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.max_entries = max_entries
        self.vigilance = vigilance
        self.hebb_lr = hebb_lr
        self.aging_time = aging_time
        self.flood_scale = flood_scale
        self.immutable_keys = immutable_keys
        self.use_lfu = use_lfu
        self.key_lr = key_lr
        self.inference_k = inference_k
        self.inference_temp = inference_temp
        self.use_bfloat16 = use_bfloat16
        self.ema_beta = ema_beta
        self.var_ema_alpha: float = 0.01
        # Similarity floor for inference: top-K entries below this threshold are
        # masked to -inf before softmax, forcing hard winner-take-all when set high.
        # 0.0 = disabled (legacy behaviour).
        self.inference_sim_floor: float = 0.0

        mem_dtype = torch.bfloat16 if use_bfloat16 else torch.float32

        # Memory storage — all on same device via register_buffer
        self.register_buffer("keys", torch.zeros(max_entries, key_dim, dtype=mem_dtype))
        self.register_buffer("values", torch.zeros(max_entries, value_dim, dtype=mem_dtype))
        self.register_buffer("occupied", torch.zeros(max_entries, dtype=torch.bool))
        self.register_buffer("last_seen", torch.zeros(max_entries, dtype=torch.float64))
        # Pre-normalized key cache for fast cosine similarity
        self.register_buffer("_keys_norm", torch.zeros(max_entries, key_dim, dtype=mem_dtype))
        # Per-slot usage count for LFU eviction
        self.register_buffer("usage", torch.zeros(max_entries, dtype=torch.float32))
        # Per-slot hit count for adaptive EMA alpha decay
        self.register_buffer("hit_counts", torch.zeros(max_entries, dtype=torch.int32))
        # Per-class running statistics for diagonal Mahalanobis re-ranking
        # Shape: (value_dim, key_dim) — value_dim doubles as num_classes
        self.register_buffer("class_means", torch.zeros(value_dim, key_dim))
        self.register_buffer("class_vars", torch.ones(value_dim, key_dim))

    @property
    def _mem_dtype(self):
        return self.keys.dtype

    def _cast(self, t: torch.Tensor) -> torch.Tensor:
        """Cast tensor to memory dtype if needed."""
        return t.to(self._mem_dtype) if t.dtype != self._mem_dtype else t

    def _update_key_norm(self, slots):
        """Update cached normalized keys for given slot indices."""
        self._keys_norm[slots] = F.normalize(self.keys[slots], dim=-1)

    # ------------------------------------------------------------------
    # Single-query API (kept for compatibility)
    # ------------------------------------------------------------------
    def _get_nearest(self, query: torch.Tensor):
        """Finds the most similar stored prototype (single query)."""
        if not self.occupied.any():
            return None, -1.0
        valid_idx = self.occupied.nonzero(as_tuple=True)[0]
        q_norm = F.normalize(self._cast(query).unsqueeze(0), dim=-1)
        sims = (q_norm @ self._keys_norm[valid_idx].T).squeeze(0)
        best_sim, best_loc = sims.max(dim=0)
        return valid_idx[best_loc].item(), best_sim.item()

    # ------------------------------------------------------------------
    # Batched core — single matmul for all queries
    # ------------------------------------------------------------------
    def _get_nearest_batch(self, queries: torch.Tensor):
        """Returns (best_slots, best_sims) tensors of shape (B,).
        Slots are -1 where table is empty."""
        B = queries.size(0)
        if not self.occupied.any():
            return (torch.full((B,), -1, dtype=torch.long, device=queries.device),
                    torch.full((B,), -1.0, device=queries.device))

        valid_idx = self.occupied.nonzero(as_tuple=True)[0]
        q_norm = F.normalize(self._cast(queries), dim=-1)
        sim_matrix = q_norm @ self._keys_norm[valid_idx].T
        best_sims, best_locs = sim_matrix.max(dim=1)
        best_slots = valid_idx[best_locs]
        return best_slots, best_sims.float()

    def _alloc_slots_batch(self, n: int):
        """Allocate n slots: free first, then LFU-LRU hybrid eviction.

        Eviction priority: lowest usage first. Ties broken by oldest last_seen.
        The last_seen tie-breaker is normalized to [0, 1) so it never overrides usage.

        When n > max_entries, only max_entries slots are returned (last-write-wins
        for the excess — matches sequential _alloc_slot semantics).
        """
        n = min(n, self.max_entries)
        free = (~self.occupied).nonzero(as_tuple=True)[0]
        if len(free) >= n:
            return free[:n]

        # Use all free slots, evict LFU-LRU for the rest
        needed = n - len(free)
        occupied_idx = self.occupied.nonzero(as_tuple=True)[0]
        # Exclude slots we're already claiming as free
        if len(free) > 0:
            mask = ~torch.isin(occupied_idx, free)
            occupied_idx = occupied_idx[mask]
        needed = min(needed, len(occupied_idx))
        if needed > 0:
            if self.use_lfu:
                # LFU-LRU hybrid: usage is primary, last_seen is tie-breaker < 1.0
                occ_usage = self.usage[occupied_idx]
                occ_time = self.last_seen[occupied_idx].float()
                t_min = occ_time.min()
                t_range = occ_time.max() - t_min
                time_tiebreak = (occ_time - t_min) / (t_range + 1e-8)  # [0, 1)
                eviction_score = occ_usage + time_tiebreak  # lowest = evict first
                _, topk_idx = eviction_score.topk(needed, largest=False)
            else:
                # Pure LRU: evict oldest by last_seen
                _, topk_idx = self.last_seen[occupied_idx].topk(needed, largest=False)
            victims = occupied_idx[topk_idx]
        else:
            victims = occupied_idx[:0]  # empty tensor
        return torch.cat([free, victims]) if len(free) > 0 else victims

    # ------------------------------------------------------------------
    # Forward / Learn — fully batched
    # ------------------------------------------------------------------
    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """Broad cosine search → diagonal Mahalanobis re-ranking → soft-kNN vote.

        1. Broad search: top-100 candidates by cosine similarity (single matmul).
        2. Re-rank: scale query and candidate keys by per-class inverse-std, then
           recompute cosine similarity in the whitened feature space.
        3. Vote: softmax-weighted sum over the top-inference_k re-ranked slots.
        """
        now = time.time()
        if not self.occupied.any():
            return torch.randn(queries.size(0), self.value_dim,
                               device=queries.device) * self.flood_scale

        valid_idx = self.occupied.nonzero(as_tuple=True)[0]
        n_valid = len(valid_idx)
        final_k = min(self.inference_k, n_valid)
        broad_k = min(100, n_valid)

        # --- Step 1: Broad cosine search ---
        q_norm = F.normalize(self._cast(queries), dim=-1)
        sims = q_norm @ self._keys_norm[valid_idx].T           # (B, N_occ)
        _, broad_locs = sims.topk(broad_k, dim=1)              # (B, broad_k)
        broad_slots = valid_idx[broad_locs]                    # (B, broad_k)

        # --- Step 2: Diagonal Mahalanobis re-ranking ---
        # Determine per-candidate class from stored one-hot values
        candidate_classes = self.values[broad_slots].float().argmax(dim=-1)  # (B, broad_k)

        # Gather per-class running variance; clamp to prevent div-by-zero
        local_vars = self.class_vars[candidate_classes].clamp(min=1e-4)      # (B, broad_k, key_dim)
        inv_std = local_vars.rsqrt()                                          # (B, broad_k, key_dim)

        # Scale both query and candidate keys by the per-class inverse std
        candidate_keys = self.keys[broad_slots].float()                       # (B, broad_k, key_dim)
        scaled_Q = queries.float().unsqueeze(1) * inv_std                     # (B, broad_k, key_dim)
        scaled_K = candidate_keys * inv_std                                   # (B, broad_k, key_dim)

        reranked_sims = F.cosine_similarity(scaled_Q, scaled_K, dim=-1)      # (B, broad_k)

        # --- Step 3: Top-final_k and softmax vote ---
        topk_sims, topk_locs = reranked_sims.topk(final_k, dim=1)            # (B, final_k)
        topk_slots = broad_slots.gather(1, topk_locs)                         # (B, final_k)

        # Optional similarity floor
        if self.inference_sim_floor > 0.0:
            topk_sims = topk_sims.masked_fill(topk_sims < self.inference_sim_floor,
                                              -float("inf"))

        weights = F.softmax(topk_sims / self.inference_temp, dim=-1)          # (B, final_k)
        retrieved = self.values[topk_slots].float()                           # (B, final_k, V)
        outputs = (weights.unsqueeze(-1) * retrieved).sum(dim=1)              # (B, V)

        # Touch only Top-1 winner for LRU bookkeeping
        self.last_seen[topk_slots[:, 0]] = now

        return outputs

    def learn_local(self, queries: torch.Tensor, targets: torch.Tensor):
        """Two-pathway learning with class-match check.

        1. Hit (sim >= vigilance, same class) → EMA update + key centroid drift
        2. Miss (below vigilance OR class collision) → allocate new slot via LFU
        """
        now = time.time()
        queries = self._cast(queries)
        targets = self._cast(targets)

        # --- Update per-class EMA mean and variance (all samples, before hit/miss) ---
        q_float = queries.float()
        class_labels = targets.float().argmax(dim=-1)  # (B,) — class index per sample
        for c in class_labels.unique():
            x_c = q_float[class_labels == c]           # (N_c, key_dim)
            batch_mean = x_c.mean(0)
            deviation = x_c - self.class_means[c]
            batch_var = (deviation ** 2).mean(0)
            self.class_means[c] = ((1.0 - self.var_ema_alpha) * self.class_means[c]
                                   + self.var_ema_alpha * batch_mean)
            self.class_vars[c] = ((1.0 - self.var_ema_alpha) * self.class_vars[c]
                                  + self.var_ema_alpha * batch_var)

        best_slots, best_sims = self._get_nearest_batch(queries)

        # Flat vigilance check (no per-engram thresholds)
        hits = (best_slots >= 0) & (best_sims >= self.vigilance)

        # Bipartite class check: demote hits where stored class differs
        if hits.any():
            hit_slots_all = best_slots[hits]
            stored_vals = self.values[hit_slots_all]
            payload_sims = F.cosine_similarity(targets[hits], stored_vals, dim=-1)
            same_class = payload_sims > 0.5

            if not same_class.all():
                hit_indices = hits.nonzero(as_tuple=True)[0]
                hits[hit_indices[~same_class]] = False

        misses = ~hits

        # --- EMA update for same-class hits ---
        if hits.any():
            hit_slots = best_slots[hits]
            hit_targets = targets[hits]

            # Scatter-mean: average target per unique slot, then apply delta
            unique_slots, inverse = hit_slots.unique(return_inverse=True)
            slot_target_sum = torch.zeros(len(unique_slots), self.value_dim,
                                          device=targets.device, dtype=targets.dtype)
            slot_target_sum.scatter_add_(0, inverse.unsqueeze(1).expand_as(hit_targets),
                                         hit_targets)
            slot_counts = torch.zeros(len(unique_slots), device=targets.device, dtype=targets.dtype)
            slot_counts.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=targets.dtype))
            slot_target_mean = slot_target_sum / slot_counts.unsqueeze(1)

            # Adaptive EMA: decay alpha as prototypes mature
            self.hit_counts[unique_slots] += slot_counts.int()
            adaptive_alpha = (self.hebb_lr /
                              (1.0 + self.ema_beta * self.hit_counts[unique_slots].float()))

            # Value EMA: adaptive_alpha * (mean_target - current_value)
            current_vals = self.values[unique_slots]
            self.values[unique_slots] = (current_vals +
                                         adaptive_alpha.unsqueeze(1) * (slot_target_mean - current_vals))

            # Key centroid drift (skip if keys are frozen)
            if not self.immutable_keys:
                hit_queries = queries[hits]
                slot_query_sum = torch.zeros(len(unique_slots), self.key_dim,
                                             device=queries.device, dtype=queries.dtype)
                slot_query_sum.scatter_add_(0, inverse.unsqueeze(1).expand_as(hit_queries),
                                            hit_queries)
                slot_query_mean = slot_query_sum / slot_counts.unsqueeze(1)
                adaptive_key_alpha = (self.key_lr /
                                      (1.0 + self.ema_beta * self.hit_counts[unique_slots].float()))
                current_keys = self.keys[unique_slots]
                self.keys[unique_slots] = (current_keys +
                                           adaptive_key_alpha.unsqueeze(1) * (slot_query_mean - current_keys))
                self._update_key_norm(unique_slots)

            self.last_seen[hit_slots] = now
            self.usage[unique_slots] += 1

        # --- Batch allocation for misses ---
        if misses.any():
            miss_queries = queries[misses]
            miss_targets = targets[misses]
            n_miss = miss_queries.size(0)

            new_slots = self._alloc_slots_batch(n_miss)
            n_alloc = len(new_slots)
            self.keys[new_slots] = miss_queries[:n_alloc]
            self.values[new_slots] = miss_targets[:n_alloc]
            self.occupied[new_slots] = True
            self.last_seen[new_slots] = now
            self.usage[new_slots] = 1
            self.hit_counts[new_slots] = 1
            self._update_key_norm(new_slots)


    def get_stats(self) -> dict:
        """Return telemetry about the current memory state."""
        n_occ = self.occupied.sum().item()
        stats = {
            "n_occupied": n_occ,
            "capacity": self.max_entries,
            "fill_pct": 100.0 * n_occ / self.max_entries,
        }
        if n_occ > 0:
            occ_mask = self.occupied
            stats["avg_hit_count"] = self.hit_counts[occ_mask].float().mean().item()
        else:
            stats["avg_hit_count"] = 0.0
        stats["avg_class_var"] = self.class_vars.mean().item()
        return stats


class CAMNet_Continuous(nn.Module):
    """Network utilizing continuous CAMs for lifelong learning."""
    def __init__(self, input_dim: int = 784, proj_dim: int = 128, cam_dim: int = 64,
                 num_classes: int = 10, num_tasks: int = 5):
        super().__init__()

        # Fixed random projection to compress input cleanly
        self.proj = nn.Linear(input_dim, proj_dim, bias=False)
        self.proj.weight.requires_grad = False

        # One independent memory module per task
        self.cam_modules = nn.ModuleList([
            ContinuousCAM(key_dim=proj_dim, value_dim=cam_dim, max_entries=2000)
            for _ in range(num_tasks)
        ])

        # Fixed Orthogonal Class Prototypes
        self.register_buffer("class_protos", _make_orthogonal_prototypes(num_classes, cam_dim))

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        with torch.no_grad():
            keys = self.proj(x)
            cam_out = self.cam_modules[task_id](keys)

            # Predict by checking similarity to known class prototypes
            cam_out_norm = F.normalize(cam_out, dim=-1)
            protos_norm = F.normalize(self.class_protos, dim=-1)
            return F.linear(cam_out_norm, protos_norm) * 10.0

    def learn_local(self, x: torch.Tensor, class_ids: torch.Tensor, task_id: int):
        with torch.no_grad():
            keys = self.proj(x)
            targets = self.class_protos[class_ids]
            self.cam_modules[task_id].learn_local(keys, targets)
