"""
shutter_deck/eviction.py — Density-Aware Eviction with Class-Balanced Floors.

Replaces the LFU-LRU hybrid in ContinuousCAM._alloc_slots_batch() with a
policy that protects rare-class prototypes from being drowned by power-law
class distributions (e.g., 500-image sunset burst vs. 1-shot heron).
"""

import logging
import torch

logger = logging.getLogger(__name__)


def alloc_slots_density_aware(
    n: int,
    occupied: torch.Tensor,
    usage: torch.Tensor,
    last_seen: torch.Tensor,
    values: torch.Tensor,
    max_entries: int,
    min_per_class: int = 5,
) -> torch.Tensor:
    """Allocate *n* slots using density-aware eviction with class-balanced floors.

    Pure tensor operations — no Python loops over slots.

    Parameters
    ----------
    n : int
        Number of slots requested.
    occupied : Tensor (max_entries,) bool
        Per-slot occupancy mask.
    usage : Tensor (max_entries,) float32
        Per-slot cumulative usage counts.
    last_seen : Tensor (max_entries,) float64
        Per-slot last-access timestamps.
    values : Tensor (max_entries, value_dim) float
        Per-slot class/cluster label vectors (one-hot or soft).
    max_entries : int
        Total table capacity.
    min_per_class : int
        Minimum prototypes per class (eviction-immune floor).

    Returns
    -------
    Tensor (<=n,) long — allocated slot indices, same device as *occupied*.
    """
    device = occupied.device
    n = min(n, max_entries)

    # Step 1: allocate from free slots first
    free = (~occupied).nonzero(as_tuple=True)[0]
    if free.numel() >= n:
        return free[:n]

    # Need eviction for the remainder
    needed = n - free.numel()
    occ_idx = occupied.nonzero(as_tuple=True)[0]

    # Exclude slots already claimed as free (shouldn't overlap, but be safe)
    if free.numel() > 0:
        mask = ~torch.isin(occ_idx, free)
        occ_idx = occ_idx[mask]

    if occ_idx.numel() == 0:
        return free[:n]

    # Step 2a: compute class labels from stored value vectors
    class_id = values[occ_idx].argmax(dim=-1)  # (N_occ,)

    # Step 2b: per-class population counts
    # Use scatter to count in a single pass
    max_class = class_id.max().item() + 1
    class_pop = torch.zeros(max_class, device=device, dtype=torch.long)
    class_pop.scatter_add_(0, class_id, torch.ones_like(class_id))

    per_slot_pop = class_pop[class_id]  # (N_occ,) — population of each slot's class

    # Step 2c: eviction-immunity mask — immune if class population == min_per_class
    immune = per_slot_pop <= min_per_class  # (N_occ,) bool

    non_immune_mask = ~immune
    n_non_immune = non_immune_mask.sum().item()

    if n_non_immune == 0:
        logger.warning(
            "All %d occupied slots are immune (floor=%d). "
            "Cannot evict any prototypes.",
            occ_idx.numel(), min_per_class,
        )
        return free if free.numel() > 0 else torch.empty(0, dtype=torch.long, device=device)

    if n_non_immune < needed:
        logger.warning(
            "Only %d non-immune slots available but %d needed. "
            "Evicting all non-immune slots (floor=%d preserved).",
            n_non_immune, needed, min_per_class,
        )
        needed = n_non_immune

    # Step 2d: density-normalized usage priority among non-immune slots
    non_immune_idx = occ_idx[non_immune_mask]  # global slot indices
    ni_usage = usage[non_immune_idx].float()
    ni_pop = per_slot_pop[non_immune_mask].float()
    priority = ni_usage / ni_pop  # density-normalized: higher = safer

    # Step 2e: tie-break with last_seen (oldest first → lower normalized value)
    ni_time = last_seen[non_immune_idx].float()
    t_min = ni_time.min()
    t_range = ni_time.max() - t_min
    time_tiebreak = (ni_time - t_min) / (t_range + 1e-8)  # [0, 1)

    eviction_score = priority + time_tiebreak  # lowest = most redundant = evict first

    # Step 2f: evict the *needed* slots with lowest priority
    _, topk_idx = eviction_score.topk(needed, largest=False)
    victims = non_immune_idx[topk_idx]

    return torch.cat([free, victims]) if free.numel() > 0 else victims


def install_density_eviction(cam, min_per_class: int = 5):
    """Monkey-patch a ContinuousCAM instance to use density-aware eviction.

    This replaces ``cam._alloc_slots_batch`` with a closure that calls
    :func:`alloc_slots_density_aware` with the CAM's live buffers.
    """
    import types

    def _alloc_slots_batch(self, n: int) -> torch.Tensor:
        return alloc_slots_density_aware(
            n=n,
            occupied=self.occupied,
            usage=self.usage,
            last_seen=self.last_seen,
            values=self.values,
            max_entries=self.max_entries,
            min_per_class=min_per_class,
        )

    cam._alloc_slots_batch = types.MethodType(_alloc_slots_batch, cam)
