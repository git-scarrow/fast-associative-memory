import torch
from gem import GEM, store_grad, overwrite_grad, project2cone2
from accumulator_buffer import AccumulatorBuffer


class AccumulatorGEM(GEM):
    """GEM with two-tier accumulator replay buffer.

    Replaces GEM's per-task episodic memory with a hot/cold zone buffer.
    Examples are ingested continuously during observe(), and periodic flushes
    consolidate hot zone entries into the cold zone.
    """

    def __init__(self, model, n_tasks, total_budget=1280,
                 hot_fraction=0.5, flush_freq=100,
                 merge_strategy="random", lr=0.1, margin=0.5):
        super().__init__(model, n_tasks, memory_size=0, lr=lr, margin=margin)
        # Remove parent's memory_x/memory_y instance attrs so __getattr__ handles them
        del self.memory_x
        del self.memory_y

        self._total_budget = total_budget
        self._hot_fraction = hot_fraction
        self._flush_freq = flush_freq
        self._merge_strategy = merge_strategy

        self.buffer = None  # Lazy init on first observe()
        self._batch_counter = 0

    def observe(self, x, y, task_id):
        # Lazy init buffer on first call
        if self.buffer is None:
            self.buffer = AccumulatorBuffer(
                total_budget=self._total_budget,
                hot_fraction=self._hot_fraction,
                flush_freq=self._flush_freq,
                merge_strategy=self._merge_strategy,
                x_shape=x.shape[1:],
                device=self.device
            )

        # 1. Continuous ingestion into hot zone
        self.buffer.ingest(x, y, task_id)

        # 2. Periodic flush
        self._batch_counter += 1
        if self._batch_counter % self._flush_freq == 0:
            if self._merge_strategy == "importance":
                self.buffer.flush(model=self.model, criterion=self.criterion)
            else:
                self.buffer.flush()

        # 3. Forward current batch
        self.opt.zero_grad()
        out = self.model(x, task_id=task_id) if (
            hasattr(self.model, 'num_tasks') and self.model.num_tasks is not None
        ) else self.model(x)
        loss = self.criterion(out, y)
        loss.backward()

        # 4. QP constraint against past tasks in buffer
        past_tasks = [t for t in self.buffer.get_task_ids_present() if t < task_id]

        if past_tasks:
            # Store current gradient
            store_grad(self.model.parameters, self.grads, self.grad_dims, task_id)
            current_grad = self.grads[:, task_id].clone()

            for idx, t in enumerate(past_tasks):
                self.opt.zero_grad()
                ptx, pty = self.buffer.sample_task(t)

                if ptx.numel() == 0:
                    # No examples for this task, store zero gradient
                    self.grads[:, idx].fill_(0.0)
                    continue

                pout = self.model(ptx, task_id=t) if (
                    hasattr(self.model, 'num_tasks') and self.model.num_tasks is not None
                ) else self.model(ptx)
                ploss = self.criterion(pout, pty)
                ploss.backward()

                # Map past task to contiguous column idx
                store_grad(self.model.parameters, self.grads, self.grad_dims, idx)

            # Build G matrix from columns 0..len(past_tasks)-1
            G = self.grads[:, :len(past_tasks)]
            dotp = G.t() @ current_grad

            if (dotp < 0).any():
                projected_g = project2cone2(current_grad, G, margin=self.margin)
                overwrite_grad(self.model.parameters, projected_g.to(self.device), self.grad_dims)
            else:
                overwrite_grad(self.model.parameters, current_grad, self.grad_dims)

        # 5. Optimizer step
        self.opt.step()

    def end_task(self, loader, task_id):
        """Trigger final flush to consolidate hot zone after task completion."""
        if self.buffer is not None:
            if self._merge_strategy == "importance":
                self.buffer.flush(model=self.model, criterion=self.criterion)
            else:
                self.buffer.flush()

    def __getattr__(self, name):
        """Intercept memory_y and memory_x access for metrics compatibility."""
        if name == 'memory_y':
            buffer = self.__dict__.get('buffer')
            if buffer is None:
                return []
            labels = buffer.get_all_labels()
            if labels.numel() == 0:
                return []
            return [labels]
        if name == 'memory_x':
            return []
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
