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
from nstp import NSTPController


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

    An optional :class:`~nstp.NSTPController` can be supplied to apply
    post-retrieval lateral inhibition during inference.  NSTP suppresses
    sibling candidates that share high inter-prototype similarity but belong
    to different classes, improving prediction sharpness in fine-grained
    scenarios.  Learning (``learn_local``) is unaffected by NSTP.
    FAM remains fully functional when ``nstp=None``.
    """

    def __init__(self, input_dim: int = 1024, value_dim: int = 100,
                 core_entries: int = 50000, core_vigilance: float = 0.85,
                 hebb_lr: float = 0.1, key_lr: float = 0.05,
                 inference_k: int = 25, inference_temp: float = 0.05,
                 whitening_dim: int = 0, use_lfu: bool = True,
                 use_bfloat16: bool = False, immutable_keys: bool = False,
                 adaptive_eviction: bool = True,
                 adapter: MetricAdapter | None = None,
                 nstp: NSTPController | None = None,
                  dynamic_vigilance=None):
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

        # Optional NSTP lateral-inhibition controller (inference-only)
        self.nstp = nstp

        # Optional dynamic vigilance scheduler (injected into core CAM)
        self.dynamic_vigilance = dynamic_vigilance

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
            adaptive_eviction=adaptive_eviction,
            use_bfloat16=use_bfloat16,
            immutable_keys=immutable_keys,
            dynamic_vigilance=dynamic_vigilance,
        )
        if nstp is not None:
            self.core_cam.nstp = nstp

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
        """Retrieve class predictions via soft-kNN voting.

        When an :class:`~nstp.NSTPController` was supplied at construction,
        NSTP lateral inhibition is applied after the top-K candidate selection
        and before the final softmax vote.
        """
        with torch.no_grad():
            return self.core_cam(self._project(x), nstp=self.nstp)

    @torch.no_grad()
    def forward_with_confidence(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve class predictions with a per-query composite confidence score.

        Runs trace=True internally and computes three orthogonal signals
        validated in MT-16 across 7 production domains:

        - **Margin**: top-1 minus top-2 vote probability (from outputs).
        - **RVA** (Rejection-Vote Alignment): cosine similarity between the
          rejection class histogram and the softmax vote distribution.  High
          RVA means rejected candidates confirm the vote (confirmatory signal).
        - **RE_inv**: 1 minus the normalized Shannon entropy of the rejection
          class distribution.  High RE_inv means rejections are concentrated
          in a few classes (structured signal).

        Composite = (margin + RVA + RE_inv) / 3, in [0, 1].
        Higher = more confident.  Calibration curve is monotonically increasing
        on both Dogs and Aircraft (MT-16 post-strip validation).

        On Aircraft (hardest domain), composite identifies errors 31% better
        than margin alone at P@100 (67% vs 51%).  On Dogs, margin and composite
        are similar in AUROC; composite adds value in the lowest-confidence
        quartile (Q1 AUROC 0.753 vs margin 0.774).

        Args:
            x: Input features, shape ``(B, input_dim)``.

        Returns:
            outputs:    Class vote logits, shape ``(B, value_dim)``.
                        Identical to :meth:`forward`.
            confidence: Per-query composite confidence, shape ``(B,)``,
                        values in [0, 1].
        """
        proj = self._project(x)

        # Empty store: core_cam returns flood noise without a trace.
        # Return zero confidence — no retrieval signal available.
        if not self.core_cam.occupied.any():
            outputs = self.core_cam(proj, nstp=self.nstp)
            return outputs, torch.zeros(outputs.size(0), device=outputs.device)

        outputs, tr = self.core_cam(proj, nstp=self.nstp, trace=True)

        B = outputs.size(0)
        n_classes = self.value_dim
        core = self.core_cam
        device = outputs.device

        # --- Margin from vote distribution ---
        # outputs is a softmax-weighted sum of one-hot value vectors, so it is
        # a proper probability distribution over classes (sums to 1).
        top2_vals = outputs.topk(min(2, n_classes), dim=-1).values  # (B, 2)
        if top2_vals.size(1) == 2:
            margin = (top2_vals[:, 0] - top2_vals[:, 1]).clamp(0.0, 1.0)
        else:
            margin = top2_vals[:, 0].clamp(0.0, 1.0)

        # --- Rejection class histograms (vectorized) ---
        # tr.rejected_slots: (B, rejected_k_max), padded by repeating last slot.
        # Clamp indices to [0, max_entries-1] for safe lookup; mask by occupied.
        if tr.rejected_slots.size(1) == 0:
            rej_hists = torch.zeros(B, n_classes, device=device)
        else:
            rej_slots = tr.rejected_slots.clamp(min=0, max=core.max_entries - 1)
            occupied_mask = core.occupied[rej_slots]                        # (B, rej_k)
            # effective_slot_labels repairs occupied-but-unstamped slots (label
            # -1, e.g. directly prepopulated buffers) via argmax(values) so
            # one_hot never sees a negative index. Unoccupied slots stay -1 and
            # are masked to 0 below.
            rej_classes = core.effective_slot_labels(rej_slots)             # (B, rej_k)
            rej_classes = rej_classes.masked_fill(~occupied_mask, 0)
            rej_one_hot = F.one_hot(rej_classes, n_classes).float()         # (B, rej_k, nc)
            rej_hists = (
                rej_one_hot * occupied_mask.unsqueeze(-1).float()
            ).sum(dim=1)                                                     # (B, nc)

        rej_totals = rej_hists.sum(dim=-1)                                  # (B,)
        has_rej = rej_totals > 0
        rej_norm = rej_hists / (rej_totals.unsqueeze(-1) + 1e-12)           # (B, nc)

        # --- RVA: cosine(rejection histogram, vote distribution) ---
        dot = (rej_norm * outputs).sum(dim=-1)                              # (B,)
        rva = dot / (
            rej_norm.norm(dim=-1) * outputs.norm(dim=-1) + 1e-12
        )
        rva = rva.masked_fill(~has_rej, 0.0)

        # --- RE_inv: 1 - normalized Shannon entropy of rejection distribution ---
        log_p = torch.where(
            rej_norm > 0,
            rej_norm.clamp(min=1e-12).log(),
            torch.zeros_like(rej_norm),
        )
        entropy = -(rej_norm * log_p).sum(dim=-1)                           # (B,)
        log_nc = torch.tensor(float(n_classes), device=device).log()
        re_inv = (1.0 - (entropy / log_nc).clamp(0.0, 1.0)).masked_fill(
            ~has_rej, 0.0
        )

        # --- Composite: equal-weight average of three orthogonal signals ---
        confidence = (margin + rva + re_inv) / 3.0                          # (B,)

        return outputs, confidence

    def learn_local(self, x: torch.Tensor, class_ids: torch.Tensor):
        """Online learning: project (adapter + whiten), form one-hot target, commit to core CAM."""
        with torch.no_grad():
            x = self._project(x)
            targets = F.one_hot(class_ids, num_classes=self.value_dim).float()
            self.core_cam.learn_local(x, targets)
