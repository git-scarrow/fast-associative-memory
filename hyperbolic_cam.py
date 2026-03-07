"""
hyperbolic_cam.py — Hyperbolic prototype memory (Lorentz hyperboloid).

Drop-in replacement for ContinuousCAM with the same external API,
but prototypes live on the Lorentz hyperboloid and distances use
geodesic (arccosh-based) metrics instead of cosine similarity.

Feature flag: use `geometry='lorentz'` in FastAssociativeMemory to activate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from lorentz_ops import (
    exp_map_origin,
    log_map_origin,
    lorentz_inner,
    lorentz_distance_batch,
    lorentz_similarity_batch,
    sign_flip,
    project_to_hyperboloid,
    EPS_NORM,
)


class HyperbolicCAM(nn.Module):
    """Prototype memory on the Lorentz hyperboloid.

    Keys are (d+1)-dimensional points on the hyperboloid.
    Input queries are d-dimensional Euclidean vectors, projected via exp-map.
    Similarity uses negative Lorentz distance.
    EMA updates use tangent-space averaging with exp-map retraction.

    The sign-flip trick stores negated temporal coordinates in _keys_flipped
    so that standard inner-product search yields correct Lorentz nearest
    neighbors (zero-cost ANN compatibility with FAISS).
    """

    def __init__(self, key_dim: int, value_dim: int, max_entries: int = 2048,
                 vigilance: float = 0.85, hebb_lr: float = 0.1,
                 curvature: float = 1.0,
                 aging_time: float = 1e9, flood_scale: float = 0.15,
                 immutable_keys: bool = False, use_lfu: bool = False,
                 adaptive_eviction: bool = True,
                 key_lr: float = 0.05,
                 inference_k: int = 20, inference_temp: float = 0.05,
                 ema_beta: float = 0.05,
                 projection: nn.Module = None):
        super().__init__()
        self.key_dim = key_dim          # Euclidean input dimension
        self.hyp_dim = key_dim + 1      # Lorentz dimension (d+1)
        self.projection = projection    # Optional custom projection (e.g. NormDecoupledProjection)
        self.value_dim = value_dim
        self.max_entries = max_entries
        self.vigilance = vigilance
        self.curvature = curvature
        self.hebb_lr = hebb_lr
        self.aging_time = aging_time
        self.flood_scale = flood_scale
        self.immutable_keys = immutable_keys
        self.use_lfu = use_lfu
        self.adaptive_eviction = adaptive_eviction
        self._classes_ever_seen: set[int] = set()
        self.key_lr = key_lr
        self.inference_k = inference_k
        self.inference_temp = inference_temp
        self.ema_beta = ema_beta
        self.inference_sim_floor: float = 0.0
        self.nstp = None

        # Vigilance is specified as a Lorentz distance threshold.
        # Points closer than this distance are "hits".
        # We convert the cosine-similarity-style vigilance to a distance threshold:
        # A cosine sim of 0.85 corresponds roughly to an angle of ~32 degrees.
        # In Lorentz space, we use a distance threshold directly.
        # The vigilance parameter here is reinterpreted as a distance threshold:
        # hit if distance < vigilance_distance.
        # We'll calibrate this adaptively based on the observed distance distribution.
        self._vigilance_distance = None  # Set during first learn batch

        # Memory storage — keys are on the hyperboloid (d+1 dimensional)
        self.register_buffer("keys", torch.zeros(max_entries, self.hyp_dim))
        self.register_buffer("values", torch.zeros(max_entries, value_dim))
        self.register_buffer("occupied", torch.zeros(max_entries, dtype=torch.bool))
        self.register_buffer("last_seen", torch.zeros(max_entries, dtype=torch.float64))
        # Sign-flipped keys for efficient IP-based search
        self.register_buffer("_keys_flipped", torch.zeros(max_entries, self.hyp_dim))
        self.register_buffer("usage", torch.zeros(max_entries, dtype=torch.float32))
        self.register_buffer("hit_counts", torch.zeros(max_entries, dtype=torch.int32))

    def _to_hyperboloid(self, x: torch.Tensor) -> torch.Tensor:
        """Project Euclidean vectors to the Lorentz hyperboloid.

        If a custom projection module is set, uses it (e.g. NormDecoupledProjection
        which separates direction from magnitude to preserve radial variation).
        Otherwise falls back to L2-normalize + exp-map (original behavior).
        """
        if self.projection is not None:
            return self.projection(x)
        x_normed = F.normalize(x, dim=-1)
        return exp_map_origin(x_normed, self.curvature)

    def _update_flipped(self, slots: torch.Tensor):
        """Update sign-flipped cache for given slots."""
        self._keys_flipped[slots] = sign_flip(self.keys[slots])

    def _get_nearest_batch(self, queries_hyp: torch.Tensor):
        """Find nearest prototypes using sign-flip trick for IP search.

        Args:
            queries_hyp: Already on hyperboloid, shape (B, d+1).

        Returns:
            (best_slots, best_dists): slots are -1 where empty; dists are Lorentz distances.
        """
        B = queries_hyp.size(0)
        if not self.occupied.any():
            return (torch.full((B,), -1, dtype=torch.long, device=queries_hyp.device),
                    torch.full((B,), float('inf'), device=queries_hyp.device))

        valid_idx = self.occupied.nonzero(as_tuple=True)[0]

        # Use sign-flip trick: IP(sign_flip(q), key) = <q, key>_L
        # Maximize Lorentz IP = minimize distance
        q_flipped = sign_flip(queries_hyp)  # (B, d+1)
        valid_keys = self.keys[valid_idx]   # (N_occ, d+1)

        # Lorentz inner product via sign-flipped query and regular keys
        # <q, k>_L = IP(q_flipped, k)
        lorentz_ip = q_flipped @ valid_keys.T  # (B, N_occ)

        # Maximum Lorentz IP = minimum distance
        best_ip, best_locs = lorentz_ip.max(dim=1)
        best_slots = valid_idx[best_locs]

        # Convert to actual distances for vigilance check
        best_dists = torch.acosh((-best_ip).clamp(min=1.0 + EPS_NORM)) / (self.curvature ** 0.5)

        return best_slots, best_dists

    def _alloc_slots_batch(self, n: int):
        """Allocate n slots, same eviction logic as ContinuousCAM."""
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
                eviction_score = self._coverage_eviction_score(occupied_idx)
                _, topk_idx = eviction_score.topk(needed, largest=False)
            else:
                _, topk_idx = self.last_seen[occupied_idx].topk(needed, largest=False)
            victims = occupied_idx[topk_idx]
        else:
            victims = occupied_idx[:0]
        return torch.cat([free, victims]) if len(free) > 0 else victims

    def _coverage_eviction_score(self, occupied_idx: torch.Tensor) -> torch.Tensor:
        """Score by Lorentz distance to nearest same-class neighbor."""
        keys_occ = self.keys[occupied_idx]
        class_labels = self.values[occupied_idx].float().argmax(dim=-1)
        scores = torch.full((len(occupied_idx),), float("inf"),
                           device=keys_occ.device, dtype=torch.float32)

        for c in class_labels.unique():
            c_mask = class_labels == c
            if c_mask.sum().item() < 2:
                continue
            idx_c = c_mask.nonzero(as_tuple=True)[0]
            keys_c = keys_occ[idx_c]
            # Pairwise Lorentz distance
            dist_c = lorentz_distance_batch(keys_c, keys_c, self.curvature)
            dist_c.fill_diagonal_(float("inf"))
            min_dist, _ = dist_c.min(dim=1)
            # Lower distance = more replaceable = lower score = evict first
            scores[idx_c] = min_dist
        return scores

    def _adaptive_eviction_score(self, occupied_idx: torch.Tensor) -> torch.Tensor:
        """Blend coverage and LRU eviction based on class loss rate."""
        class_labels = self.values[occupied_idx].float().argmax(dim=-1)
        unique_classes = class_labels.unique()
        n_classes_present = len(unique_classes)
        n_classes_seen = len(self._classes_ever_seen) if self._classes_ever_seen else n_classes_present

        if n_classes_seen > 0:
            class_loss = 1.0 - (n_classes_present / n_classes_seen)
        else:
            class_loss = 0.0

        if class_loss < 1e-9:
            return self.last_seen[occupied_idx]

        p = min(1.0, 0.2 + 0.8 * min(class_loss / 0.30, 1.0))

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

        ls = self.last_seen[occupied_idx].float()
        ls_min, ls_max = ls.min(), ls.max()
        ls_range = ls_max - ls_min
        if ls_range > 0:
            lru_norm = (ls - ls_min) / ls_range
        else:
            lru_norm = torch.ones_like(ls) * 0.5

        return p * coverage_norm + (1.0 - p) * lru_norm

    def forward(self, queries: torch.Tensor, nstp=None, trace: bool = False):
        """Retrieve via soft-kNN voting in Lorentz space.

        1. Project queries to hyperboloid.
        2. Find top-K by Lorentz distance (via sign-flip IP trick).
        3. Softmax-weighted vote using negative distance as similarity.
        """
        if not self.occupied.any():
            return torch.randn(queries.size(0), self.value_dim,
                              device=queries.device) * self.flood_scale

        valid_idx = self.occupied.nonzero(as_tuple=True)[0]
        n_valid = len(valid_idx)
        final_k = min(self.inference_k, n_valid)
        broad_k = min(100, n_valid)

        # Project to hyperboloid
        q_hyp = self._to_hyperboloid(queries.float())

        # Sign-flip trick for IP-based search
        q_flipped = sign_flip(q_hyp)
        valid_keys = self.keys[valid_idx]

        # Lorentz IP via sign-flip: IP(q_flipped, k) = <q, k>_L
        lorentz_ip = q_flipped @ valid_keys.T  # (B, N_occ)

        # Top-broad_k by Lorentz IP (higher = closer)
        broad_ips, broad_locs = lorentz_ip.topk(broad_k, dim=1)
        broad_slots = valid_idx[broad_locs]

        # Top-final_k
        _, topk_locs = broad_ips.topk(final_k, dim=1)
        topk_slots = broad_slots.gather(1, topk_locs)
        topk_ips = broad_ips.gather(1, topk_locs)

        # Convert Lorentz IP to distances for similarity proxy
        # d = arccosh(-<x,y>_L) / sqrt(c)
        topk_dists = torch.acosh((-topk_ips).clamp(min=1.0 + EPS_NORM)) / (self.curvature ** 0.5)
        # Use negative distance as similarity (higher = closer)
        topk_sims = -topk_dists

        # NSTP lateral inhibition (optional)
        _nstp = nstp if nstp is not None else self.nstp
        if _nstp is not None:
            # For NSTP, we need keys in a format it can use.
            # Use the spatial part of hyperbolic keys (d-dimensional).
            topk_keys = self.keys[topk_slots][..., 1:].float()
            topk_vals = self.values[topk_slots].float()
            keep_mask, _ = _nstp.prune_batch(
                queries.float(), topk_keys, topk_vals, topk_sims
            )
            topk_sims = topk_sims.masked_fill(~keep_mask, -float("inf"))

        # Softmax vote — use adapted temperature for distance-based similarity.
        # Cosine sims are ~[0.8, 1.0] range; negative distances are ~[-1.5, -0.1].
        # Scale to match effective sharpness of the Euclidean softmax.
        weights = F.softmax(topk_sims / self.inference_temp, dim=-1)
        retrieved = self.values[topk_slots].float()
        outputs = (weights.unsqueeze(-1) * retrieved).sum(dim=1)

        return outputs

    def learn_local(self, queries: torch.Tensor, targets: torch.Tensor):
        """Two-pathway learning in Lorentz space.

        1. Project queries to hyperboloid.
        2. Hit (distance < threshold, same class) -> tangent-space EMA update.
        3. Miss -> allocate new slot.
        """
        now = time.time()
        queries = queries.float()
        targets = targets.float()

        # Project to hyperboloid
        q_hyp = self._to_hyperboloid(queries)

        best_slots, best_dists = self._get_nearest_batch(q_hyp)

        # Curvature-adaptive vigilance: distance threshold
        # We calibrate the vigilance threshold from the cosine-similarity
        # vigilance parameter. For the exp-map, a cosine sim of v maps roughly
        # to a Lorentz distance of arccosh(1 / (1 - arccos(v)/pi)).
        # Simpler: use a percentile-based calibration on first batch.
        if self._vigilance_distance is None:
            # Use a fixed distance threshold calibrated to match the
            # Euclidean vigilance behavior. A cosine sim of 0.85 in
            # high-dimensional space corresponds to a small angle;
            # we map it to the corresponding Lorentz distance.
            # For points near the origin, exp-map preserves relative distances.
            # Empirically calibrated: vigilance 0.85 ~ distance 0.55
            self._vigilance_distance = self._calibrate_vigilance(q_hyp)

        # Hit detection: distance below threshold
        hits = (best_slots >= 0) & (best_dists < self._vigilance_distance)

        # Class check: demote hits where stored class differs
        if hits.any():
            hit_slots_all = best_slots[hits]
            stored_vals = self.values[hit_slots_all]
            payload_sims = F.cosine_similarity(targets[hits], stored_vals, dim=-1)
            same_class = payload_sims > 0.5
            if not same_class.all():
                hit_indices = hits.nonzero(as_tuple=True)[0]
                hits[hit_indices[~same_class]] = False

        misses = ~hits

        # --- Tangent-space EMA update for hits ---
        if hits.any():
            hit_slots = best_slots[hits]
            hit_targets = targets[hits]
            hit_q_hyp = q_hyp[hits]

            unique_slots, inverse = hit_slots.unique(return_inverse=True)
            slot_target_sum = torch.zeros(len(unique_slots), self.value_dim,
                                         device=targets.device, dtype=targets.dtype)
            slot_target_sum.scatter_add_(0, inverse.unsqueeze(1).expand_as(hit_targets),
                                        hit_targets)
            slot_counts = torch.zeros(len(unique_slots), device=targets.device, dtype=targets.dtype)
            slot_counts.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=targets.dtype))
            slot_target_mean = slot_target_sum / slot_counts.unsqueeze(1)

            # Adaptive EMA alpha
            self.hit_counts[unique_slots] += slot_counts.int()
            adaptive_alpha = (self.hebb_lr /
                             (1.0 + self.ema_beta * self.hit_counts[unique_slots].float()))

            # Value EMA (same as Euclidean — values are still Euclidean one-hot)
            current_vals = self.values[unique_slots]
            self.values[unique_slots] = (current_vals +
                                         adaptive_alpha.unsqueeze(1) * (slot_target_mean - current_vals))

            # Key update: tangent-space averaging with exp-map retraction
            if not self.immutable_keys:
                # For each unique slot, compute mean query in tangent space at the current prototype
                current_keys = self.keys[unique_slots]  # (U, d+1) on hyperboloid

                # Compute mean of hit queries per slot in tangent space at origin
                slot_q_sum = torch.zeros(len(unique_slots), self.hyp_dim,
                                        device=queries.device)
                slot_q_sum.scatter_add_(0, inverse.unsqueeze(1).expand(-1, self.hyp_dim),
                                       hit_q_hyp)
                slot_q_mean = slot_q_sum / slot_counts.unsqueeze(1)

                # Project mean queries to hyperboloid (they may have drifted off)
                slot_q_mean = project_to_hyperboloid(slot_q_mean, self.curvature)

                # Tangent-space EMA: interpolate in tangent space at origin
                # log both current and target to tangent space, interpolate, exp back
                current_tangent = log_map_origin(current_keys, self.curvature)
                target_tangent = log_map_origin(slot_q_mean, self.curvature)

                adaptive_key_alpha = (self.key_lr /
                                     (1.0 + self.ema_beta * self.hit_counts[unique_slots].float()))

                updated_tangent = (current_tangent +
                                  adaptive_key_alpha.unsqueeze(1) * (target_tangent - current_tangent))
                updated_keys = exp_map_origin(updated_tangent, self.curvature)

                self.keys[unique_slots] = updated_keys
                self._update_flipped(unique_slots)

            self.last_seen[hit_slots] = now
            self.usage[unique_slots] += 1

        # --- Batch allocation for misses ---
        if misses.any():
            miss_q_hyp = q_hyp[misses]
            miss_targets = targets[misses]
            n_miss = miss_q_hyp.size(0)

            new_slots = self._alloc_slots_batch(n_miss)
            n_alloc = len(new_slots)
            self.keys[new_slots] = miss_q_hyp[:n_alloc]
            self.values[new_slots] = miss_targets[:n_alloc]
            self.occupied[new_slots] = True
            self.last_seen[new_slots] = now
            self.usage[new_slots] = 1
            self.hit_counts[new_slots] = 1
            self._update_flipped(new_slots)

        # Track classes for adaptive eviction
        if self.adaptive_eviction:
            all_classes = targets.float().argmax(dim=-1)
            self._classes_ever_seen.update(all_classes.cpu().tolist())

    def _calibrate_vigilance(self, q_hyp: torch.Tensor) -> float:
        """Calibrate distance threshold from the cosine-similarity vigilance.

        For L2-normalized vectors exp-mapped with c=1, two unit vectors at
        angle theta = arccos(vigilance) map to Lorentz distance:
            d = arccosh(cosh(1)^2 - sinh(1)^2 * cos(theta))

        This is computed exactly.
        """
        import math
        v = min(self.vigilance, 1.0 - 1e-7)
        c = self.curvature
        # For unit-norm vectors exp-mapped at origin with curvature c:
        # Points are at (cosh(sqrt(c)), sinh(sqrt(c)) * v_hat) on the hyperboloid.
        # Lorentz IP between two such points at angle theta:
        # <p1, p2>_L = -cosh^2(sqrt(c)) + sinh^2(sqrt(c)) * cos(theta)
        # Distance = arccosh(-<p1,p2>_L) / sqrt(c)
        sc = c ** 0.5
        ch = math.cosh(sc)
        sh = math.sinh(sc)
        cos_theta = v
        neg_ip = ch * ch - sh * sh * cos_theta  # = -<p1,p2>_L
        dist_threshold = math.acosh(max(neg_ip, 1.0 + 1e-12)) / sc
        return dist_threshold

    def get_stats(self) -> dict:
        n_occ = self.occupied.sum().item()
        stats = {
            "n_occupied": n_occ,
            "capacity": self.max_entries,
            "fill_pct": 100.0 * n_occ / self.max_entries,
            "vigilance_distance": self._vigilance_distance,
        }
        if n_occ > 0:
            occ_mask = self.occupied
            stats["avg_hit_count"] = self.hit_counts[occ_mask].float().mean().item()
            # Radial distribution: time coordinate of prototypes
            radii = self.keys[occ_mask, 0]
            stats["mean_radius"] = radii.mean().item()
            stats["std_radius"] = radii.std().item()
            stats["min_radius"] = radii.min().item()
            stats["max_radius"] = radii.max().item()
        else:
            stats["avg_hit_count"] = 0.0
        return stats

    def get_prototype_radii(self) -> torch.Tensor:
        """Return the Lorentz radius (arccosh of time coordinate) for each occupied prototype."""
        occ_idx = self.occupied.nonzero(as_tuple=True)[0]
        if len(occ_idx) == 0:
            return torch.tensor([])
        # Radius from origin = arccosh(x_0) for c=1
        time_coords = self.keys[occ_idx, 0]
        radii = torch.acosh(time_coords.clamp(min=1.0 + EPS_NORM)) / (self.curvature ** 0.5)
        return radii

    def get_prototype_classes(self) -> torch.Tensor:
        """Return class labels for occupied prototypes."""
        occ_idx = self.occupied.nonzero(as_tuple=True)[0]
        if len(occ_idx) == 0:
            return torch.tensor([], dtype=torch.long)
        return self.values[occ_idx].float().argmax(dim=-1)
