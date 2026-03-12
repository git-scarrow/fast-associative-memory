"""Per-task-partitioned accumulator buffer for BCL-EXP-4.

Fork of accumulator_buffer.py with static per-task cold zone partitions.
Each task gets floor(cold_capacity / max_tasks) slots, guaranteeing minimum
exemplar counts even at high N. Hot zone remains shared.
"""
import torch
import random


class PartitionedAccumulatorBuffer:
    """Two-tier replay buffer with per-task-partitioned cold zone.

    Hot zone: shared FIFO circular buffer (identical to AccumulatorBuffer).
    Cold zone: statically partitioned into N equal slots per task.
    Flush routes exemplars from hot → cold into the correct task partition.
    """

    def __init__(self, total_budget, hot_fraction, flush_freq, merge_strategy,
                 x_shape, device, max_tasks):
        self.B = total_budget
        self.alpha = hot_fraction
        self.flush_freq = flush_freq
        self.merge_strategy = merge_strategy
        self.device = device
        self.max_tasks = max_tasks

        self.hot_capacity = max(1, int(self.alpha * self.B))
        self.cold_capacity = self.B - self.hot_capacity

        # Per-task partition size (static allocation)
        self.partition_size = max(1, self.cold_capacity // self.max_tasks)

        # Hot zone: shared FIFO circular buffer
        self.hot_x = torch.zeros(self.hot_capacity, *x_shape, device=device)
        self.hot_y = torch.full((self.hot_capacity,), -1, dtype=torch.long, device=device)
        self.hot_task = torch.full((self.hot_capacity,), -1, dtype=torch.long, device=device)
        self.hot_count = 0
        self.hot_head = 0

        # Cold zone: partitioned storage (max_tasks × partition_size)
        self.cold_x = torch.zeros(self.max_tasks, self.partition_size, *x_shape, device=device)
        self.cold_y = torch.full((self.max_tasks, self.partition_size), -1,
                                 dtype=torch.long, device=device)
        # Per-partition occupancy count
        self.cold_counts = [0] * self.max_tasks

    def ingest(self, x, y, task_id):
        """Add a batch of examples to the hot zone. Vectorized FIFO circular write."""
        batch_size = x.size(0)
        positions = torch.arange(batch_size, device=self.device)
        positions = (self.hot_head + positions) % self.hot_capacity

        self.hot_x[positions] = x
        self.hot_y[positions] = y
        self.hot_task[positions] = task_id

        self.hot_head = (self.hot_head + batch_size) % self.hot_capacity
        self.hot_count = min(self.hot_count + batch_size, self.hot_capacity)

    def flush(self, model=None, criterion=None):
        """Move candidates from hot zone to cold zone, routing by task partition."""
        if self.hot_count == 0:
            return

        valid_mask = self.hot_y >= 0
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        if valid_indices.numel() == 0:
            return

        if self.merge_strategy == "random":
            self._flush_random(valid_indices)
        elif self.merge_strategy == "reservoir":
            self._flush_reservoir(valid_indices)
        else:
            self._flush_random(valid_indices)

    def _flush_random(self, valid_indices):
        """Random subset from hot, routed to per-task partitions."""
        n_valid = valid_indices.numel()
        n_transfer = min(n_valid, max(1, self.cold_capacity // 10))

        perm = torch.randperm(n_valid, device=self.device)[:n_transfer]
        selected = valid_indices[perm]

        for idx in selected:
            idx_item = idx.item()
            task_id = self.hot_task[idx_item].item()
            if task_id < 0 or task_id >= self.max_tasks:
                continue
            self._insert_cold(task_id, self.hot_x[idx_item], self.hot_y[idx_item])

        # Clear transferred hot entries
        self.hot_y[selected] = -1
        self.hot_task[selected] = -1

    def _flush_reservoir(self, valid_indices):
        """Reservoir sampling into per-task partitions."""
        for idx in valid_indices:
            idx_item = idx.item()
            task_id = self.hot_task[idx_item].item()
            if task_id < 0 or task_id >= self.max_tasks:
                continue
            self._insert_cold(task_id, self.hot_x[idx_item], self.hot_y[idx_item])

    def _insert_cold(self, task_id, x, y):
        """Insert a single example into the task's cold partition."""
        count = self.cold_counts[task_id]
        if count < self.partition_size:
            self.cold_x[task_id, count] = x
            self.cold_y[task_id, count] = y
            self.cold_counts[task_id] = count + 1
        else:
            # Partition full: random replacement within partition
            pos = random.randint(0, self.partition_size - 1)
            self.cold_x[task_id, pos] = x
            self.cold_y[task_id, pos] = y

    def sample_task(self, task_id):
        """Return (x, y) for all examples belonging to task_id across both zones."""
        parts_x = []
        parts_y = []

        # Hot zone
        hot_mask = self.hot_task == task_id
        if hot_mask.any():
            parts_x.append(self.hot_x[hot_mask])
            parts_y.append(self.hot_y[hot_mask])

        # Cold zone: task's partition
        if task_id < self.max_tasks:
            count = self.cold_counts[task_id]
            if count > 0:
                parts_x.append(self.cold_x[task_id, :count])
                parts_y.append(self.cold_y[task_id, :count])

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

        for t in range(self.max_tasks):
            count = self.cold_counts[t]
            if count > 0:
                valid = self.cold_y[t, :count]
                mask = valid >= 0
                if mask.any():
                    parts.append(valid[mask])

        if not parts:
            return torch.empty(0, dtype=torch.long, device=self.device)
        return torch.cat(parts)

    def get_task_ids_present(self):
        """Return sorted list of unique task IDs with at least one example."""
        present = set()

        hot_valid = self.hot_task[self.hot_y >= 0]
        if hot_valid.numel() > 0:
            present.update(hot_valid.unique().tolist())

        for t in range(self.max_tasks):
            if self.cold_counts[t] > 0:
                present.add(t)

        return sorted(int(t) for t in present)

    def get_per_task_counts(self):
        """Return dict of {task_id: total_exemplar_count} across both zones."""
        counts = {}
        for t in range(self.max_tasks):
            cold_n = self.cold_counts[t]
            hot_n = (self.hot_task == t).sum().item()
            total = cold_n + hot_n
            if total > 0:
                counts[t] = total
        return counts
