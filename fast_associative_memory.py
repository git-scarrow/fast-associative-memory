"""
fast_associative_memory.py — Online Soft-kNN with optional PCA whitening.

Receives raw DINOv2 features, optionally whitens them, then stores/retrieves
via a single prototype memory using temperature-scaled soft-kNN voting with
class-conditional centroid drift.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from associative_core import ContinuousCAM
from adapter import MetricAdapter


class WhiteningLayer(nn.Module):
    """Frozen PCA whitening: (B, input_dim) → (B, output_dim).

    Fit once on training embeddings (burn-in), then frozen for all subsequent
    reads and writes.  Decorrelates the DINOv2 embedding manifold, eliminating
    the narrow-cone geometry that inflates inter-class cosine baselines.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.register_buffer("mean", torch.zeros(input_dim))
        self.register_buffer("W", torch.eye(output_dim, input_dim))  # (out, in)

    def fit(self, X: torch.Tensor, eps: float = 1e-5):
        """Compute PCA whitening from (N, input_dim) training embeddings."""
        X = X.float()
        mu = X.mean(0)
        X_c = X - mu
        cov = (X_c.T @ X_c) / (len(X) - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)   # ascending
        # Top output_dim components (largest eigenvalues = descending sort)
        idx = eigenvalues.argsort(descending=True)[:self.output_dim]
        V = eigenvectors[:, idx]                              # (D, d')
        D_vals = eigenvalues[idx]                             # (d',)
        W = (V / (D_vals + eps).sqrt()).T                     # (d', D)
        self.mean.copy_(mu)
        self.W.copy_(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.float() - self.mean, self.W)       # (B, d')


class FastAssociativeMemory(nn.Module):
    """Online soft-kNN associative memory with optional PCA whitening.

    Receives raw DINOv2 features (default 1024-d), optionally whitens them,
    then stores/retrieves via a single ContinuousCAM using Top-K
    softmax-weighted voting with class-conditional centroid drift.

    An optional :class:`~adapter.MetricAdapter` can be supplied to apply a
    learned metric projection before keys are committed to memory.  When
    provided, the adapter is applied in both :meth:`forward` and
    :meth:`learn_local` so that retrieval and storage share the same feature
    space.  FAM remains fully functional when ``adapter=None``.
    """

    def __init__(self, input_dim: int = 1024, value_dim: int = 100,
                 core_entries: int = 50000, core_vigilance: float = 0.85,
                 hebb_lr: float = 0.1, key_lr: float = 0.05,
                 inference_k: int = 25, inference_temp: float = 0.05,
                 whitening_dim: int = 0, use_lfu: bool = True,
                 use_bfloat16: bool = False, immutable_keys: bool = False,
                 adapter: MetricAdapter | None = None):
        super().__init__()
        self.input_dim = input_dim
        self.value_dim = value_dim

        # Optional metric adapter: applied before whitening (if both are set)
        self.adapter = adapter
        pre_dim = adapter.output_dim if adapter is not None else input_dim

        # Optional PCA whitening
        if whitening_dim > 0:
            self.whitening: WhiteningLayer | None = WhiteningLayer(pre_dim, whitening_dim)
            cam_key_dim = whitening_dim
        else:
            self.whitening = None
            cam_key_dim = pre_dim

        # Single core memory
        self.core_cam = ContinuousCAM(
            key_dim=cam_key_dim,
            value_dim=value_dim,
            max_entries=core_entries,
            vigilance=core_vigilance,
            hebb_lr=hebb_lr,
            key_lr=key_lr,
            inference_k=inference_k,
            inference_temp=inference_temp,
            use_lfu=use_lfu,
            use_bfloat16=use_bfloat16,
            immutable_keys=immutable_keys,
        )

    def fit_whitening(self, X: torch.Tensor):
        """Fit PCA whitening on training embeddings. No-op if whitening disabled."""
        if self.whitening is not None:
            self.whitening.fit(X)

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adapter then whitening (each is a no-op when disabled)."""
        if self.adapter is not None:
            x = self.adapter(x)
        if self.whitening is not None:
            x = self.whitening(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retrieve class predictions via soft-kNN voting."""
        with torch.no_grad():
            return self.core_cam(self._project(x))

    def learn_local(self, x: torch.Tensor, class_ids: torch.Tensor):
        """Online learning: project (adapter + whiten), form one-hot target, commit to core CAM."""
        with torch.no_grad():
            x = self._project(x)
            targets = F.one_hot(class_ids, num_classes=self.value_dim).float()
            self.core_cam.learn_local(x, targets)
