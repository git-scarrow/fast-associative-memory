import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class AccumulatorBuffer:
    """Two-tier replay buffer inspired by bcachefs disk accounting.

    Hot zone: FIFO circular buffer for recent examples (high throughput, high recency).
    Cold zone: consolidated representatives from prior tasks (stable, coverage-optimized).
    Periodic flush moves candidates from hot -> cold via configurable merge strategy.

    All operations are vectorized — no per-element Python loops on the hot path.
    """

    def __init__(self, total_budget, hot_fraction, flush_freq, merge_strategy,
                 x_shape, device):
        self.B = total_budget
        self.alpha = hot_fraction
        self.flush_freq = flush_freq
        self.merge_strategy = merge_strategy
        self.device = device

        self.hot_capacity = max(1, int(self.alpha * self.B))
        self.cold_capacity = self.B - self.hot_capacity

        # Hot zone: FIFO circular buffer
        self.hot_x = torch.zeros(self.hot_capacity, *x_shape, device=device)
        self.hot_y = torch.full((self.hot_capacity,), -1, dtype=torch.long, device=device)
        self.hot_task = torch.full((self.hot_capacity,), -1, dtype=torch.long, device=device)
        self.hot_count = 0
        self.hot_head = 0  # FIFO write pointer

        # Cold zone: consolidated storage
        self.cold_x = torch.zeros(self.cold_capacity, *x_shape, device=device)
        self.cold_y = torch.full((self.cold_capacity,), -1, dtype=torch.long, device=device)
        self.cold_task = torch.full((self.cold_capacity,), -1, dtype=torch.long, device=device)
        self.cold_count = 0

        # Reservoir sampling state
        self._reservoir_n = 0

        # Importance-weighted: cold utility scores (decayed loss)
        self.cold_utility = torch.zeros(self.cold_capacity, device=device)

    def ingest(self, x, y, task_id):
        """Add a batch of examples to the hot zone. Vectorized FIFO circular write."""
        batch_size = x.size(0)

        # Compute write positions (may wrap around)
        positions = torch.arange(batch_size, device=self.device)
        positions = (self.hot_head + positions) % self.hot_capacity

        # Batch write
        self.hot_x[positions] = x
        self.hot_y[positions] = y
        self.hot_task[positions] = task_id

        # Advance FIFO pointer
        self.hot_head = (self.hot_head + batch_size) % self.hot_capacity
        self.hot_count = min(self.hot_count + batch_size, self.hot_capacity)

    def flush(self, model=None, criterion=None):
        """Move candidates from hot zone to cold zone using configured strategy."""
        if self.hot_count == 0:
            return

        if self.merge_strategy == "random":
            self._flush_random()
        elif self.merge_strategy == "reservoir":
            self._flush_reservoir()
        elif self.merge_strategy == "importance":
            self._flush_importance(model, criterion)
        else:
            raise ValueError(f"Unknown merge strategy: {self.merge_strategy}")

    def _flush_random(self):
        """Select random subset from valid hot entries, merge into cold zone."""
        valid_mask = self.hot_y >= 0
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        if valid_indices.numel() == 0:
            return

        n_valid = valid_indices.numel()
        n_transfer = min(n_valid, max(1, self.cold_capacity // 10))

        # Random selection from hot zone
        perm = torch.randperm(n_valid, device=self.device)[:n_transfer]
        selected = valid_indices[perm]

        self._transfer_to_cold(selected)

        # Clear transferred hot entries
        self.hot_y[selected] = -1
        self.hot_task[selected] = -1

    def _flush_reservoir(self):
        """Vitter's Algorithm R across the hot->cold boundary."""
        valid_mask = self.hot_y >= 0
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        if valid_indices.numel() == 0:
            return

        # Process each valid hot entry through reservoir sampling
        for idx in valid_indices:
            idx_item = idx.item()
            self._reservoir_n += 1

            if self.cold_count < self.cold_capacity:
                cold_pos = self.cold_count
                self.cold_x[cold_pos] = self.hot_x[idx_item]
                self.cold_y[cold_pos] = self.hot_y[idx_item]
                self.cold_task[cold_pos] = self.hot_task[idx_item]
                self.cold_count += 1
            else:
                j = random.randint(0, self._reservoir_n - 1)
                if j < self.cold_capacity:
                    self.cold_x[j] = self.hot_x[idx_item]
                    self.cold_y[j] = self.hot_y[idx_item]
                    self.cold_task[j] = self.hot_task[idx_item]

        # Hot entries NOT cleared — FIFO overwrites them naturally

    def _flush_importance(self, model, criterion):
        """Score hot examples by loss, promote highest-loss to cold."""
        valid_mask = self.hot_y >= 0
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        if valid_indices.numel() == 0:
            return
        if model is None or criterion is None:
            self._flush_random()
            return

        n_valid = valid_indices.numel()

        # Compute per-example loss, grouped by task_id
        all_losses = torch.zeros(n_valid, device=self.device)
        tasks_in_hot = self.hot_task[valid_indices].unique().tolist()

        model.eval()
        with torch.no_grad():
            for t in tasks_in_hot:
                task_mask = self.hot_task[valid_indices] == t
                task_local = task_mask.nonzero(as_tuple=True)[0]
                hot_global = valid_indices[task_local]

                tx = self.hot_x[hot_global]
                ty = self.hot_y[hot_global]

                # Batched forward
                batch_losses = []
                for start in range(0, tx.size(0), 256):
                    end = min(start + 256, tx.size(0))
                    out = model(tx[start:end], task_id=t) if (
                        hasattr(model, 'num_tasks') and model.num_tasks is not None
                    ) else model(tx[start:end])
                    batch_losses.append(F.cross_entropy(out, ty[start:end], reduction='none'))

                all_losses[task_local] = torch.cat(batch_losses)
        model.train()

        # Promote top-k highest-loss
        n_promote = min(n_valid, max(1, self.cold_capacity // 10))
        _, top_k = all_losses.topk(min(n_promote, n_valid))
        promote_global = valid_indices[top_k]

        # Decay cold utility
        self.cold_utility[:self.cold_count] *= 0.95

        for i, hot_idx in enumerate(promote_global):
            hot_idx = hot_idx.item()
            loss_val = all_losses[top_k[i]].item()

            if self.cold_count < self.cold_capacity:
                cold_pos = self.cold_count
                self.cold_count += 1
            else:
                cold_pos = self.cold_utility[:self.cold_count].argmin().item()

            self.cold_x[cold_pos] = self.hot_x[hot_idx]
            self.cold_y[cold_pos] = self.hot_y[hot_idx]
            self.cold_task[cold_pos] = self.hot_task[hot_idx]
            self.cold_utility[cold_pos] = loss_val

        # Clear transferred hot entries
        self.hot_y[promote_global] = -1
        self.hot_task[promote_global] = -1

    def _transfer_to_cold(self, hot_indices):
        """Transfer selected hot entries to cold zone. Random replacement when full."""
        n = hot_indices.numel()

        for i in range(n):
            hi = hot_indices[i].item()

            if self.cold_count < self.cold_capacity:
                cold_pos = self.cold_count
                self.cold_count += 1
            else:
                cold_pos = random.randint(0, self.cold_capacity - 1)

            self.cold_x[cold_pos] = self.hot_x[hi]
            self.cold_y[cold_pos] = self.hot_y[hi]
            self.cold_task[cold_pos] = self.hot_task[hi]
            self.cold_utility[cold_pos] = 0.0

    def sample_task(self, task_id):
        """Return (x, y) for all examples belonging to task_id across both zones."""
        parts_x = []
        parts_y = []

        # Hot zone: mask by task_id
        hot_mask = self.hot_task == task_id
        if hot_mask.any():
            parts_x.append(self.hot_x[hot_mask])
            parts_y.append(self.hot_y[hot_mask])

        # Cold zone: mask by task_id (only occupied slots)
        if self.cold_count > 0:
            cold_mask = self.cold_task[:self.cold_count] == task_id
            if cold_mask.any():
                parts_x.append(self.cold_x[:self.cold_count][cold_mask])
                parts_y.append(self.cold_y[:self.cold_count][cold_mask])

        if not parts_x:
            x_shape = self.hot_x.shape[1:]
            return (torch.empty(0, *x_shape, device=self.device),
                    torch.empty(0, dtype=torch.long, device=self.device))

        return torch.cat(parts_x, dim=0), torch.cat(parts_y, dim=0)

    def get_all_labels(self):
        """Return all valid labels across both zones."""
        parts = []

        hot_valid = self.hot_y >= 0
        if hot_valid.any():
            parts.append(self.hot_y[hot_valid])

        if self.cold_count > 0:
            cold_valid = self.cold_y[:self.cold_count] >= 0
            if cold_valid.any():
                parts.append(self.cold_y[:self.cold_count][cold_valid])

        if not parts:
            return torch.empty(0, dtype=torch.long, device=self.device)

        return torch.cat(parts)

    def get_task_ids_present(self):
        """Return sorted list of unique task IDs with at least one example."""
        parts = []

        hot_valid = self.hot_task[self.hot_y >= 0]
        if hot_valid.numel() > 0:
            parts.append(hot_valid)

        if self.cold_count > 0:
            cold_valid = self.cold_task[:self.cold_count]
            valid_mask = cold_valid >= 0
            if valid_mask.any():
                parts.append(cold_valid[valid_mask])

        if not parts:
            return []

        all_tasks = torch.cat(parts).unique().tolist()
        return sorted(int(t) for t in all_tasks)
