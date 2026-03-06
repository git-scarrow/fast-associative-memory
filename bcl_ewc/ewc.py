"""
BCL-EXP-3: Log-Structured EWC vs Standard EWC

Implements:
  - StandardEWC: immediate consolidation after every task
  - LogStructuredEWC: append-only log with periodic compaction
"""

import torch
import torch.nn as nn
import time
def _clone_param_dict(d):
    """Clone a dict of tensors safely (detach from graph)."""
    return {n: t.clone().detach() for n, t in d.items()}


def compute_fisher_diagonal(model, data_loader, device, n_samples=None):
    """
    Compute diagonal Fisher information matrix via empirical Fisher
    (gradient squared, averaged over samples).
    """
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}

    total = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        out = model(x)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += (p.grad.data ** 2) * x.size(0)

        total += x.size(0)
        if n_samples is not None and total >= n_samples:
            break

    for n in fisher:
        fisher[n] /= total

    model.train()
    return fisher


class StandardEWC:
    """
    Faithful standard EWC baseline: immediate consolidation after every task.
    F_consolidated = sum of all per-task Fishers seen so far.
    """

    def __init__(self, model):
        self.model = model
        self.consolidated_fisher = None
        self.optimal_params = {}  # params snapshot after each task
        self.task_params = []     # list of param snapshots per task
        self._consolidation_times = []

    def after_task(self, task_fisher):
        """Immediately merge task_fisher into consolidated store."""
        t0 = time.perf_counter()
        if self.consolidated_fisher is None:
            self.consolidated_fisher = {n: f.clone() for n, f in task_fisher.items()}
        else:
            for n in self.consolidated_fisher:
                self.consolidated_fisher[n] = self.consolidated_fisher[n] + task_fisher[n]
        t1 = time.perf_counter()
        self._consolidation_times.append(t1 - t0)

        # Snapshot optimal params
        self.optimal_params = {n: p.clone() for n, p in self.model.named_parameters() if p.requires_grad}
        self.task_params.append(_clone_param_dict(self.optimal_params))

    def get_importance(self):
        """Return consolidated Fisher."""
        return self.consolidated_fisher

    def penalty(self, lambd=100.0):
        """EWC regularization loss."""
        if self.consolidated_fisher is None:
            return 0.0
        loss = 0.0
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.consolidated_fisher:
                loss += (self.consolidated_fisher[n] * (p - self.optimal_params[n]) ** 2).sum()
        return lambd * loss

    def get_consolidation_time(self):
        """Average consolidation time per task."""
        if not self._consolidation_times:
            return 0.0
        return sum(self._consolidation_times) / len(self._consolidation_times)


class LogStructuredEWC:
    """
    Log-structured EWC: append per-task Fishers to a write-ahead log,
    compact every C tasks into a consolidated store.
    """

    def __init__(self, model, compaction_interval=5, merge_weights='uniform'):
        self.model = model
        self.consolidated_fisher = None
        self.fisher_log = []
        self.compaction_interval = compaction_interval
        self.merge_weights = merge_weights
        self.tasks_since_compaction = 0
        self.optimal_params = {}
        self.task_params = []
        self._consolidation_times = []  # time spent in compact()
        self._append_times = []         # time spent in after_task (just appending)

    def after_task(self, task_fisher):
        """Append to log, compact if interval reached."""
        t0 = time.perf_counter()
        self.fisher_log.append({n: f.clone() for n, f in task_fisher.items()})
        self.tasks_since_compaction += 1
        t1 = time.perf_counter()
        self._append_times.append(t1 - t0)

        # Snapshot optimal params
        self.optimal_params = {n: p.clone() for n, p in self.model.named_parameters() if p.requires_grad}
        self.task_params.append(_clone_param_dict(self.optimal_params))

        if self.tasks_since_compaction >= self.compaction_interval:
            self.compact()

    def compact(self):
        """Merge log entries into consolidated store using selected weighting strategy."""
        if not self.fisher_log:
            return

        t0 = time.perf_counter()

        n_entries = len(self.fisher_log)
        weights = self._compute_weights(n_entries)

        # Weighted merge of log entries
        merged = {}
        param_names = list(self.fisher_log[0].keys())
        for n in param_names:
            merged[n] = torch.zeros_like(self.fisher_log[0][n])
            for i, entry in enumerate(self.fisher_log):
                merged[n] += weights[i] * entry[n]

        # Merge into consolidated
        if self.consolidated_fisher is None:
            self.consolidated_fisher = merged
        else:
            for n in self.consolidated_fisher:
                self.consolidated_fisher[n] = self.consolidated_fisher[n] + merged[n]

        # Clear the log
        self.fisher_log = []
        self.tasks_since_compaction = 0

        t1 = time.perf_counter()
        self._consolidation_times.append(t1 - t0)

    def _compute_weights(self, n_entries):
        """Compute merge weights for log entries."""
        if self.merge_weights == 'uniform':
            return [1.0] * n_entries
        elif self.merge_weights == 'recency':
            # Linear ramp: oldest=1, newest=n_entries
            raw = [float(i + 1) for i in range(n_entries)]
            total = sum(raw)
            return [w / total * n_entries for w in raw]  # normalize so sum = n_entries
        elif self.merge_weights == 'magnitude':
            # Weight by L2 norm of each Fisher diagonal
            norms = []
            for entry in self.fisher_log:
                norm = sum(f.norm().item() ** 2 for f in entry.values()) ** 0.5
                norms.append(norm)
            total = sum(norms)
            if total == 0:
                return [1.0] * n_entries
            return [n / total * n_entries for n in norms]  # normalize so sum = n_entries
        else:
            raise ValueError(f"Unknown merge_weights: {self.merge_weights}")

    def get_importance(self):
        """Read path: consolidated + all pending log entries."""
        if self.consolidated_fisher is None and not self.fisher_log:
            return None

        result = {}
        if self.consolidated_fisher is not None:
            result = {n: f.clone() for n, f in self.consolidated_fisher.items()}

        for entry in self.fisher_log:
            for n, f in entry.items():
                if n in result:
                    result[n] = result[n] + f
                else:
                    result[n] = f.clone()

        return result

    def penalty(self, lambd=100.0):
        """EWC regularization loss using read-path importance."""
        importance = self.get_importance()
        if importance is None:
            return 0.0
        loss = 0.0
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in importance:
                loss += (importance[n] * (p - self.optimal_params[n]) ** 2).sum()
        return lambd * loss

    def get_consolidation_time(self):
        """Average compaction time per compaction event."""
        if not self._consolidation_times:
            return 0.0
        return sum(self._consolidation_times) / len(self._consolidation_times)

    def get_total_consolidation_time(self):
        """Total time spent in compaction + appending."""
        return sum(self._consolidation_times) + sum(self._append_times)

    def peak_memory_entries(self):
        """Number of Fisher matrices held simultaneously (log + consolidated)."""
        return len(self.fisher_log) + (1 if self.consolidated_fisher is not None else 0)
