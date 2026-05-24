"""
associative_core.py — Online prototype memory with EMA updates,
temperature-scaled soft-kNN retrieval, and LFU-LRU hybrid eviction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from dataclasses import dataclass
from typing import Union, Tuple

@dataclass
class RetrievalTrace:
    broad_slots: torch.Tensor          # (broad_k,) — top-100 indices by raw cosine
    broad_sims_raw: torch.Tensor       # (broad_k,) — cosine similarities
    final_slots: torch.Tensor          # (final_k,) — survivor indices after NSTP + floor
    final_weights: torch.Tensor        # (final_k,) — softmax vote weights
    rejected_slots: torch.Tensor       # (broad_k - final_k,) — indices of killed candidates
    rejection_stage: torch.Tensor      # (broad_k - final_k,) categorical: 0=topk_drop, 1=nstp, 2=floor


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
                 adaptive_eviction: bool = True,
                 use_bfloat16: bool = False, key_lr: float = 0.05,
                 inference_k: int = 20, inference_temp: float = 0.05,
                 ema_beta: float = 0.05,
                 dynamic_vigilance=None):
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.max_entries = max_entries
        self.vigilance = vigilance
        self.dynamic_vigilance = dynamic_vigilance
        self._dynamic_vigilance_stats = {
            "sum_v": 0.0, "sum_margin": 0.0, "count": 0,
            # rho(t) probe: top-1 cross-class cosine (manifold-contraction signal).
            # Accumulated as sum + sum-of-squares so per-epoch mean AND variance
            # are recoverable. See get_stats() and reset_dynamic_vigilance_stats().
            "sum_sim_other": 0.0, "sumsq_sim_other": 0.0,
        }
        self.hebb_lr = hebb_lr
        self.aging_time = aging_time
        self.flood_scale = flood_scale
        self.immutable_keys = immutable_keys
        self.use_lfu = use_lfu
        self.adaptive_eviction = adaptive_eviction
        # Track total distinct classes ever written (for adaptive eviction)
        self._classes_ever_seen: set[int] = set()
        self.key_lr = key_lr
        self.inference_k = inference_k
        self.inference_temp = inference_temp
        self.use_bfloat16 = use_bfloat16
        self.ema_beta = ema_beta
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
        self.nstp = None  # Optional NSTPController for lateral inhibition

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

        Slots are -1 where table is empty.
        """
        B = queries.size(0)
        if not self.occupied.any():
            return (torch.full((B,), -1, dtype=torch.long, device=queries.device),
                    torch.full((B,), -1.0, device=queries.device))

        valid_idx = self.occupied.nonzero(as_tuple=True)[0]
        q_norm = F.normalize(self._cast(queries), dim=-1)
        sim_matrix = q_norm @ self._keys_norm[valid_idx].T        # (B, N_occ)
        best_sims, best_locs = sim_matrix.max(dim=1)
        best_slots = valid_idx[best_locs]

        return best_slots, best_sims.float()

    def _alloc_slots_batch(self, n: int):
        """Allocate n slots: free first, then eviction for the rest.

        When use_lfu is True, uses coverage-aware eviction: evicts the
        prototype whose nearest same-class neighbor is closest (most
        replaceable).  Sole class representatives are protected.
        Replaced the prior LFU-LRU hybrid after MT-12 showed coverage
        eviction closes 67–99% of the coreset gap.

        When use_lfu is False, falls back to pure LRU.

        When n > max_entries, only max_entries slots are returned (last-write-wins
        for the excess — matches sequential _alloc_slot semantics).
        """
        n = min(n, self.max_entries)
        free = (~self.occupied).nonzero(as_tuple=True)[0]
        if len(free) >= n:
            return free[:n]

        needed = n - len(free)
        occupied_idx = self.occupied.nonzero(as_tuple=True)[0]
        if len(free) > 0:
            mask = ~torch.isin(occupied_idx, free)
            occupied_idx = occupied_idx[mask]
        needed = min(needed, len(occupied_idx))
        if needed > 0:
            if self.adaptive_eviction:
                eviction_score = self._adaptive_eviction_score(occupied_idx)
                _, topk_idx = eviction_score.topk(needed, largest=False)
            elif self.use_lfu:
                # Coverage-aware eviction: evict the most replaceable prototype
                # (smallest distance to nearest same-class neighbor).
                eviction_score = self._coverage_eviction_score(occupied_idx)
                _, topk_idx = eviction_score.topk(needed, largest=False)
            else:
                # Pure LRU: evict oldest by last_seen
                _, topk_idx = self.last_seen[occupied_idx].topk(needed, largest=False)
            victims = occupied_idx[topk_idx]
        else:
            victims = occupied_idx[:0]  # empty tensor
        return torch.cat([free, victims]) if len(free) > 0 else victims

    def _coverage_eviction_score(self, occupied_idx: torch.Tensor) -> torch.Tensor:
        """Score each prototype by distance to nearest same-class neighbor.

        Lower score = more replaceable = evict first.
        Score = 1 - cos(prototype, nearest same-class neighbor).
        Sole class representatives get score = inf (protected from eviction).
        """
        keys_norm = self._keys_norm[occupied_idx]
        class_labels = self.values[occupied_idx].float().argmax(dim=-1)
        scores = torch.full(
            (len(occupied_idx),), float("inf"),
            device=keys_norm.device, dtype=torch.float32,
        )
        for c in class_labels.unique():
            c_mask = class_labels == c
            if c_mask.sum().item() < 2:
                continue
            idx_c = c_mask.nonzero(as_tuple=True)[0]
            keys_c = keys_norm[idx_c].float()
            sim_c = keys_c @ keys_c.T
            sim_c.fill_diagonal_(-float("inf"))
            max_sim, _ = sim_c.max(dim=1)
            scores[idx_c] = 1.0 - max_sim
        return scores

    def _adaptive_eviction_score(self, occupied_idx: torch.Tensor) -> torch.Tensor:
        """Blend coverage and LRU eviction based on class loss rate.

        Tracks classes ever seen vs classes currently in store.  When no
        classes have been lost, delegates to pure LRU (preserving EMA-refined
        centroids).  When classes start disappearing, blends in coverage
        eviction to protect diversity.

        Score semantics: lower = evict first.
        """
        class_labels = self.values[occupied_idx].float().argmax(dim=-1)
        unique_classes, class_counts = class_labels.unique(return_counts=True)
        n_classes_present = len(unique_classes)
        n_classes_seen = len(self._classes_ever_seen) if self._classes_ever_seen else n_classes_present

        # Class loss rate: fraction of previously-seen classes no longer in store.
        # 0.0 = all classes retained. 0.9 = 90% of classes lost.
        if n_classes_seen > 0:
            class_loss = 1.0 - (n_classes_present / n_classes_seen)
        else:
            class_loss = 0.0

        # Pure LRU when no classes have been lost.
        if class_loss < 1e-9:
            return self.last_seen[occupied_idx]

        # Any class loss immediately activates coverage blending.
        # Linear ramp: 0% → p=0.2, 30%+ → p=1.0 (full coverage mode).
        p = min(1.0, 0.2 + 0.8 * min(class_loss / 0.30, 1.0))

        # Coverage score: lower = more replaceable = evict first
        coverage_raw = self._coverage_eviction_score(occupied_idx)
        finite_mask = coverage_raw.isfinite()
        if finite_mask.any():
            cmin = coverage_raw[finite_mask].min()
            cmax = coverage_raw[finite_mask].max()
            crange = cmax - cmin
            if crange > 0:
                coverage_norm = (coverage_raw - cmin) / crange
            else:
                coverage_norm = torch.ones_like(coverage_raw) * 0.5
            coverage_norm[~finite_mask] = float("inf")
        else:
            return self.last_seen[occupied_idx]

        # LRU score: normalized [0, 1]
        ls = self.last_seen[occupied_idx].float()
        ls_min, ls_max = ls.min(), ls.max()
        ls_range = ls_max - ls_min
        if ls_range > 0:
            lru_norm = (ls - ls_min) / ls_range
        else:
            lru_norm = torch.ones_like(ls) * 0.5

        blended = p * coverage_norm + (1.0 - p) * lru_norm
        return blended

    # ------------------------------------------------------------------
    # Forward / Learn — fully batched
    # ------------------------------------------------------------------
    def forward(self, queries: torch.Tensor, nstp=None, trace: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, RetrievalTrace]]:
        """Broad cosine search → soft-kNN vote.

        1. Broad search: top-100 candidates by cosine similarity (single matmul).
        2. Vote: softmax-weighted sum over the top-inference_k candidates.

        Args:
            queries: Input query tensor, shape ``(B, key_dim)``.
            nstp:    Optional :class:`~nstp.NSTPController` for post-retrieval
                     lateral inhibition.  When provided, suppressed candidates
                     are masked to ``-inf`` before the final softmax vote so
                     that they contribute zero weight.  ``None`` disables NSTP
                     (default, preserves existing behaviour exactly).
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
        broad_sims_raw, broad_locs = sims.topk(broad_k, dim=1)              # (B, broad_k)
        broad_slots = valid_idx[broad_locs]                    # (B, broad_k)

        # --- Step 2: Top-final_k and softmax vote ---
        _, topk_locs = broad_sims_raw.topk(final_k, dim=1)                    # (B, final_k)
        topk_slots = broad_slots.gather(1, topk_locs)                         # (B, final_k)
        topk_sims = broad_sims_raw.gather(1, topk_locs)                       # (B, final_k)

        # --- Step 4b: NSTP lateral inhibition (optional) ---
        # Per-call `nstp` parameter takes priority; falls back to instance self.nstp.
        _nstp = nstp if nstp is not None else self.nstp
        keep_mask = None
        if _nstp is not None:
            topk_keys = self.keys[topk_slots].float()                         # (B, final_k, key_dim)
            topk_vals = self.values[topk_slots].float()                       # (B, final_k, value_dim)
            keep_mask, _ = _nstp.prune_batch(
                q_norm.float(), topk_keys, topk_vals, topk_sims
            )
            topk_sims = topk_sims.masked_fill(~keep_mask, -float("inf"))

        # Optional similarity floor
        if self.inference_sim_floor > 0.0:
            topk_sims = topk_sims.masked_fill(topk_sims < self.inference_sim_floor,
                                              -float("inf"))

        weights = F.softmax(topk_sims / self.inference_temp, dim=-1)          # (B, final_k)
        retrieved = self.values[topk_slots].float()                           # (B, final_k, V)
        outputs = (weights.unsqueeze(-1) * retrieved).sum(dim=1)              # (B, V)

        # NOTE: last_seen is NOT updated here. Inference is read-only.
        # learn_local() handles all last_seen bookkeeping for both new and
        # existing prototypes. Updating last_seen during forward() caused
        # eval contamination: wall-clock timestamps from inference protected
        # test-relevant prototypes from eviction (see MT-6 writeup).

        if not trace:
            return outputs

        # === TRACE COLLECTION ===
        B = queries.size(0)
        _broad_slots = broad_slots.detach()
        _broad_sims_raw = broad_sims_raw.detach()
        _weights = weights.detach()
        _topk_locs = topk_locs.detach()

        # 0=topk_drop, survivors start at 4
        rej_stages = torch.zeros(B, broad_k, dtype=torch.long, device=queries.device)
        rej_stages.scatter_(1, _topk_locs, 4)

        # Now mark NSTP and floor victims
        _base_topk_sims = broad_sims_raw.gather(1, _topk_locs)
        
        if keep_mask is not None:
            _keep_mask = keep_mask.detach()
        else:
            _keep_mask = torch.ones_like(_base_topk_sims, dtype=torch.bool)
            
        floor_mask = _base_topk_sims < self.inference_sim_floor
        _floor_rejected = floor_mask & _keep_mask
        
        final_slots_list = []
        final_weights_list = []
        rejected_slots_list = []
        rejection_stage_list = []
        
        final_k_max = 0
        rejected_k_max = 0
        
        for b in range(B):
            nstp_mask_b = ~_keep_mask[b]
            if nstp_mask_b.any():
                rej_stages[b, _topk_locs[b][nstp_mask_b]] = 1

            floor_mask_b = _floor_rejected[b]
            if floor_mask_b.any():
                rej_stages[b, _topk_locs[b][floor_mask_b]] = 2
                
            stages = rej_stages[b]
            survivor_mask = stages == 4
            rejected_mask = stages != 4
            
            row_survivor_slots = _broad_slots[b][survivor_mask]
            
            surv_in_topk_mask = _keep_mask[b] & ~floor_mask[b]
            row_survivor_weights = _weights[b][surv_in_topk_mask]
            
            row_rejected_slots = _broad_slots[b][rejected_mask]
            row_rejection_stages = stages[rejected_mask]
            
            final_slots_list.append(row_survivor_slots)
            final_weights_list.append(row_survivor_weights)
            rejected_slots_list.append(row_rejected_slots)
            rejection_stage_list.append(row_rejection_stages)
            
            if len(row_survivor_slots) > final_k_max:
                final_k_max = len(row_survivor_slots)
            if len(row_rejected_slots) > rejected_k_max:
                rejected_k_max = len(row_rejected_slots)
                
        def _pad(lst, max_len):
            padded = []
            for t in lst:
                if len(t) == max_len:
                    padded.append(t)
                elif len(t) == 0:
                    padded.append(torch.full((max_len,), -1, dtype=t.dtype, device=t.device))
                else:
                    padded.append(F.pad(t, (0, max_len - len(t)), value=t[-1].item()))
            return torch.stack(padded)
            
        tr = RetrievalTrace(
            broad_slots=_broad_slots,
            broad_sims_raw=_broad_sims_raw,
            final_slots=_pad(final_slots_list, final_k_max) if final_k_max > 0 else torch.empty((B, 0), dtype=torch.long, device=queries.device),
            final_weights=_pad(final_weights_list, final_k_max) if final_k_max > 0 else torch.empty((B, 0), dtype=torch.float32, device=queries.device),
            rejected_slots=_pad(rejected_slots_list, rejected_k_max) if rejected_k_max > 0 else torch.empty((B, 0), dtype=torch.long, device=queries.device),
            rejection_stage=_pad(rejection_stage_list, rejected_k_max) if rejected_k_max > 0 else torch.empty((B, 0), dtype=torch.long, device=queries.device)
        )
        return outputs, tr

    def learn_local(self, queries: torch.Tensor, targets: torch.Tensor):
        """Two-pathway learning with class-match check.

        1. Hit (sim >= vigilance, same class) → EMA update + key centroid drift
        2. Miss (below vigilance OR class collision) → allocate new slot via LFU
        """
        now = time.time()
        queries = self._cast(queries)
        targets = self._cast(targets)

        best_slots, best_sims = self._get_nearest_batch(queries)

        # Flat or margin-based dynamic vigilance check
        if self.dynamic_vigilance is not None and self.occupied.any():
            # We already computed best match via _get_nearest_batch.
            # To avoid allocating a second full (B, N_occ) sim matrix, compute
            # the best *other-class* competitor in small query chunks.
            valid_idx = self.occupied.nonzero(as_tuple=True)[0]
            keys_occ = self._keys_norm[valid_idx]  # (N_occ, D)
            # TD(AgentAssociativeMemory): argmax assumes one-hot / softmax-class
            # value vectors. Works for the current classifier/probe branch but will
            # not generalise to dense arbitrary-value slots. The refactor should
            # introduce an explicit label store or a pluggable label extractor.
            proto_labels = self.values[valid_idx].float().argmax(dim=-1)  # (N_occ,)

            B = queries.size(0)
            best_classes = torch.full((B,), -1, dtype=torch.long, device=queries.device)
            has_best = best_slots >= 0
            if has_best.any():
                best_classes[has_best] = self.values[best_slots[has_best]].float().argmax(dim=-1)

            margins = torch.empty((B,), device=queries.device, dtype=torch.float32)
            v_eff = torch.empty((B,), device=queries.device, dtype=torch.float32)
            # rho(t) probe: retain the per-query top-1 cross-class cosine that the
            # margin computation discards. NOTE: "cross-class" here is masked
            # against the query's BEST-MATCH class (best_classes), not its true
            # label, so this is a slight under-estimate of true cross-class
            # similarity. For label-true rho(t) use probe_cross_class_similarity().
            sim_other_all = torch.empty((B,), device=queries.device, dtype=torch.float32)

            very_low = -1e9
            chunk_q = 16
            dv = self.dynamic_vigilance

            for s in range(0, B, chunk_q):
                e = min(B, s + chunk_q)
                q = F.normalize(queries[s:e].float(), dim=-1)  # (C, D)
                sims = q @ keys_occ.T  # (C, N_occ)

                bc = best_classes[s:e]
                same_class = proto_labels.unsqueeze(0) == bc.unsqueeze(1)
                sims_other = sims.masked_fill(same_class, very_low)
                sim_other, _ = sims_other.max(dim=1)
                sim_other_all[s:e] = sim_other

                margin = best_sims[s:e] - sim_other
                margins[s:e] = margin

                v = dv.v_base - dv.alpha * margin
                v_eff[s:e] = torch.clamp(v, dv.v_floor, dv.v_ceiling)

            # Track running means for benchmarking/telemetry
            self._dynamic_vigilance_stats["sum_v"] += float(v_eff.sum().item())
            self._dynamic_vigilance_stats["sum_margin"] += float(margins.sum().item())
            self._dynamic_vigilance_stats["count"] += int(v_eff.numel())

            # rho(t) probe accumulation. Exclude queries that had NO cross-class
            # competitor (sim_other == very_low sentinel) so the contraction
            # signal is not poisoned by single-class batches. Counted under a
            # separate denominator (count_sim_other) for correctness.
            # Exclude only the very_low sentinel (no cross-class competitor); a
            # legitimate cosine of -1.0 (antipodal competitor) must still count.
            valid_other = sim_other_all > (very_low / 2)
            n_valid = int(valid_other.sum().item())
            if n_valid > 0:
                so = sim_other_all[valid_other]
                self._dynamic_vigilance_stats["sum_sim_other"] += float(so.sum().item())
                self._dynamic_vigilance_stats["sumsq_sim_other"] += float((so * so).sum().item())
                self._dynamic_vigilance_stats["count_sim_other"] = (
                    self._dynamic_vigilance_stats.get("count_sim_other", 0) + n_valid)

            # Opt-in raw log (enabled when a list attribute `cross_class_sim_log`
            # is attached), mirroring the margin_log / vigilance_log pattern.
            cc_log = getattr(self, "cross_class_sim_log", None)
            if isinstance(cc_log, list):
                cc_log.append(sim_other_all[valid_other].detach().cpu())

            # Optional per-call logging for external analysis (enabled when
            # a list attribute `margin_log` / `vigilance_log` is attached).
            margin_log = getattr(self, "margin_log", None)
            if isinstance(margin_log, list):
                margin_log.append(margins.detach().cpu())
            v_log = getattr(self, "vigilance_log", None)
            if isinstance(v_log, list):
                v_log.append(v_eff.detach().cpu())

            vigilance_thresholds = v_eff.to(best_sims.device)
        else:
            vigilance_thresholds = torch.full_like(best_sims, self.vigilance)

        hits = (best_slots >= 0) & (best_sims >= vigilance_thresholds)

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
                                         adaptive_alpha.unsqueeze(1) * (slot_target_mean - current_vals)
                                         ).to(self._mem_dtype)

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
                                           adaptive_key_alpha.unsqueeze(1) * (slot_query_mean - current_keys)
                                           ).to(self._mem_dtype)
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

        # Track all classes seen for adaptive eviction (covers hits + misses)
        if self.adaptive_eviction:
            all_classes = targets.float().argmax(dim=-1)
            self._classes_ever_seen.update(all_classes.cpu().tolist())

    def sleep(self, anti_lr=0.3, max_epochs=10, collision_threshold=0.5,
              chunk_size=1024, verbose=False) -> dict:
        """NREM sleep consolidation: anti-Hebbian repulsion of cross-class engrams.

        Replays stored keys offline and pushes collision-prone engrams from
        different classes apart in key-space, fixing retrieval theft without
        touching the write path.
        """
        occ_idx = self.occupied.nonzero(as_tuple=True)[0]
        n_occ = len(occ_idx)
        if n_occ < 2:
            return {"epochs": 0, "collisions_initial": 0,
                    "collisions_final": 0, "keys_modified": 0}

        class_labels = self.values[occ_idx].argmax(dim=-1)   # (N_occ,)
        all_keys_norm = self._keys_norm[occ_idx]              # (N_occ, D)

        collisions_initial = None
        collisions_final = 0
        keys_modified_total = 0

        for epoch in range(max_epochs):
            delta = torch.zeros_like(self.keys[occ_idx])       # (N_occ, D)
            counts = torch.zeros(n_occ, device=self.keys.device)
            epoch_collisions = 0

            for c_start in range(0, n_occ, chunk_size):
                c_end = min(c_start + chunk_size, n_occ)
                chunk_keys = all_keys_norm[c_start:c_end]       # (C, D)
                chunk_labels = class_labels[c_start:c_end]       # (C,)

                sim_matrix = chunk_keys @ all_keys_norm.T        # (C, N_occ)

                # Mask out self-similarity
                self_idx = torch.arange(c_end - c_start, device=sim_matrix.device)
                sim_matrix[self_idx, self_idx + c_start] = -float("inf")

                best_sims, best_locs = sim_matrix.max(dim=1)    # (C,)

                different_class = chunk_labels != class_labels[best_locs]
                above_thresh = best_sims > collision_threshold
                collisions = different_class & above_thresh

                n_col = collisions.sum().item()
                epoch_collisions += n_col

                if n_col == 0:
                    continue

                col_idx = collisions.nonzero(as_tuple=True)[0]
                a_global = col_idx + c_start
                b_global = best_locs[col_idx]

                key_a = self.keys[occ_idx[a_global]]
                key_b = self.keys[occ_idx[b_global]]
                repulsion = anti_lr * (key_a - key_b)

                delta[a_global] += repulsion
                counts[a_global] += 1
                delta[b_global] -= repulsion
                counts[b_global] += 1

            if collisions_initial is None:
                collisions_initial = epoch_collisions

            if verbose:
                print(f"    sleep epoch {epoch}: {epoch_collisions} collisions")

            if epoch_collisions == 0:
                collisions_final = 0
                break

            collisions_final = epoch_collisions

            # Apply averaged deltas (skip if keys are frozen)
            active = counts > 0
            if active.any() and not self.immutable_keys:
                avg_delta = delta[active] / counts[active].unsqueeze(1)
                active_slots = occ_idx[active]
                self.keys[active_slots] += avg_delta
                self._update_key_norm(active_slots)
                all_keys_norm = self._keys_norm[occ_idx]
                keys_modified_total += active.sum().item()

        return {
            "epochs": epoch + 1 if n_occ >= 2 else 0,
            "collisions_initial": collisions_initial or 0,
            "collisions_final": collisions_final,
            "keys_modified": keys_modified_total,
        }

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
        if self._dynamic_vigilance_stats["count"] > 0:
            count = float(self._dynamic_vigilance_stats["count"])
            stats["mean_v_effective"] = self._dynamic_vigilance_stats["sum_v"] / count
            stats["mean_margin"] = self._dynamic_vigilance_stats["sum_margin"] / count
        else:
            stats["mean_v_effective"] = float(self.vigilance)
            stats["mean_margin"] = 0.0

        # rho(t): empirical mean + variance of top-1 cross-class cosine. This is
        # the manifold-contraction signal. Predicted onsets to watch:
        #   mean_cross_class_sim >= v_ceiling (0.95) -> structural chimeras.
        #   (retrieval-blend onset ~0.79 is measured via probe_cross_class_similarity)
        n_cc = int(self._dynamic_vigilance_stats.get("count_sim_other", 0))
        if n_cc > 0:
            n = float(n_cc)
            mean_cc = self._dynamic_vigilance_stats["sum_sim_other"] / n
            # Population variance: E[x^2] - E[x]^2, clamped at 0 for fp safety.
            var_cc = max(0.0, self._dynamic_vigilance_stats["sumsq_sim_other"] / n - mean_cc * mean_cc)
            stats["mean_cross_class_sim"] = mean_cc
            stats["var_cross_class_sim"] = var_cc
            stats["n_cross_class_obs"] = n_cc
        else:
            stats["mean_cross_class_sim"] = float("nan")
            stats["var_cross_class_sim"] = float("nan")
            stats["n_cross_class_obs"] = 0
        return stats

    def reset_dynamic_vigilance_stats(self) -> None:
        """Zero the running vigilance / rho(t) accumulators.

        Call at each epoch boundary so get_stats() reports per-epoch means
        (and so the rho(t) degradation curve has one clean point per epoch).
        Does NOT touch any opt-in raw logs (margin_log, vigilance_log,
        cross_class_sim_log) — clear those externally if used.
        """
        self._dynamic_vigilance_stats = {
            "sum_v": 0.0, "sum_margin": 0.0, "count": 0,
            "sum_sim_other": 0.0, "sumsq_sim_other": 0.0, "count_sim_other": 0,
        }

    @torch.no_grad()
    def probe_cross_class_similarity(self, probe_queries, probe_labels,
                                     blend_eps: float = 0.10) -> dict:
        """Read-only rho(t) probe against a held-out, TRUE-LABELED set.

        This is the gold-standard manifold-contraction measurement: unlike the
        in-band sim_other (masked against the best-MATCH class), this masks
        against the true label, so it captures cross-class competitors even when
        they are the top match. Use it to plot the degradation curve and to test
        the two predicted onsets:

          * Retrieval-blend onset (~0.79 for Delta~0.9): begins when a non-trivial
            share of the inference softmax mass lands on off-class prototypes.
            Reported as `mean_offclass_weight` (mean fraction of vote mass on
            wrong-class prototypes) and `frac_blended` (fraction of probes whose
            off-class mass exceeds `blend_eps`). VIGIL's analytic gap of ~0.11
            corresponds to off-class weight ~`blend_eps`.
          * Structural-chimera onset (0.95 = v_ceiling): begins when
            `mean_cross_class_sim` crosses the dynamic-vigilance ceiling, after
            which boundary writes are admitted and EMA bakes them into keys.

        Args:
            probe_queries: (P, key_dim) raw (un-normalized) query vectors.
            probe_labels:  (P,) long tensor of TRUE class ids.
            blend_eps:     off-class vote-mass threshold counted as a "blend".

        Returns dict with mean+variance of top-1 cross-class cosine, mean
        within-class cosine, the true margin, retrieval-blend metrics, and the
        predicted onset flags.
        Returns {"n_occupied": 0} if memory is empty (nothing to probe).
        """
        if not self.occupied.any():
            return {"n_occupied": 0}

        device = self.keys.device
        q = F.normalize(probe_queries.to(device).float(), dim=-1)        # (P, D)
        labels = probe_labels.to(device).long()                          # (P,)

        valid_idx = self.occupied.nonzero(as_tuple=True)[0]
        keys_occ = self._keys_norm[valid_idx].float()                    # (N, D)
        # TD(AgentAssociativeMemory): same argmax assumption as in learn_local().
        proto_labels = self.values[valid_idx].float().argmax(dim=-1)     # (N,)

        sims = q @ keys_occ.T                                            # (P, N)
        same_class = proto_labels.unsqueeze(0) == labels.unsqueeze(1)    # (P, N)
        very_low = -1e9

        # Top-1 cross-class and within-class cosine, per probe.
        sims_other = sims.masked_fill(same_class, very_low)
        sim_other, _ = sims_other.max(dim=1)
        sims_same = sims.masked_fill(~same_class, very_low)
        sim_same, _ = sims_same.max(dim=1)

        # Only probes that have BOTH a same-class and an other-class prototype
        # yield a meaningful margin / contraction reading.
        # Validity is "did such a prototype exist", i.e. not the sentinel — a
        # legitimate cosine of -1.0 must still count as a real competitor.
        has_other = sim_other > (very_low / 2)
        has_same = sim_same > (very_low / 2)
        valid = has_other & has_same

        out = {"n_occupied": int(valid_idx.numel()), "n_probes": int(valid.sum().item())}
        if out["n_probes"] == 0:
            return out

        so = sim_other[valid]
        ss = sim_same[valid]
        margin = ss - so
        out["mean_cross_class_sim"] = float(so.mean().item())
        out["var_cross_class_sim"] = float(so.var(unbiased=False).item())
        out["mean_within_class_sim"] = float(ss.mean().item())
        out["mean_true_margin"] = float(margin.mean().item())

        # --- Retrieval-blend metric: replicate the inference softmax vote ---
        # (matches forward(): top-inference_k sims, softmax(./inference_temp)).
        k = min(self.inference_k, keys_occ.size(0))
        topk_sims, topk_locs = sims[valid].topk(k, dim=1)                # (Pv, k)
        if self.inference_sim_floor > 0.0:
            topk_sims = topk_sims.masked_fill(
                topk_sims < self.inference_sim_floor, -float("inf"))
        # Guard: if the floor masks every candidate in a row, softmax(-inf...)
        # is NaN and would silently corrupt the telemetry. Neutralize those rows
        # before softmax and exclude them from the vote-mass averages.
        row_has_vote = torch.isfinite(topk_sims).any(dim=1)              # (Pv,)
        safe_sims = topk_sims.masked_fill(~row_has_vote.unsqueeze(1), 0.0)
        weights = F.softmax(safe_sims / self.inference_temp, dim=-1)     # (Pv, k)
        topk_labels = proto_labels[topk_locs]                           # (Pv, k)
        offclass = topk_labels != labels[valid].unsqueeze(1)
        offclass_weight = (weights * offclass.float()).sum(dim=1)[row_has_vote]
        n_vote = int(row_has_vote.sum().item())
        out["n_vote_probes"] = n_vote
        if n_vote > 0:
            out["mean_offclass_weight"] = float(offclass_weight.mean().item())
            out["frac_blended"] = float((offclass_weight > blend_eps).float().mean().item())
        else:
            out["mean_offclass_weight"] = float("nan")
            out["frac_blended"] = float("nan")

        # --- Predicted onset flags (read against live dynamic-vigilance config) ---
        dv = self.dynamic_vigilance
        ceiling = float(dv.v_ceiling) if dv is not None else float("nan")
        out["v_ceiling"] = ceiling
        out["chimera_onset"] = bool(out["mean_cross_class_sim"] >= ceiling)
        out["blend_onset"] = bool(out["mean_offclass_weight"] >= blend_eps)
        return out


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
