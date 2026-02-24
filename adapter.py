"""
adapter.py — MetricAdapter module with Triplet Loss for Granularity Collapse (G11).

Provides a lightweight learned projection (linear or 2-layer MLP) that sits between
the DINOv2 backbone output and FAM Core input.  After projection the output is
L2-normalised, so it lives on the unit hypersphere — the same geometry FAM's
cosine-vigilance gate assumes.

Training the adapter via ``triplet_loss`` pushes hard-negative pairs (e.g. Husky
vs. Wolf) further apart in the projected space without changing FAM's write path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetricAdapter(nn.Module):
    """Lightweight learned projection for metric-space shaping.

    Architecture is controlled by ``hidden_dim``:
    * ``hidden_dim == 0`` (default): single ``nn.Linear(input_dim, output_dim)``
    * ``hidden_dim > 0``: two-layer MLP ``input_dim → hidden_dim → output_dim``
      with a ReLU non-linearity between the layers.

    The ``forward`` pass always L2-normalises the projection output so that
    downstream cosine-based vigilance comparisons remain meaningful.

    Args:
        input_dim:  Dimensionality of the backbone embedding (e.g. 1024 for
                    DINOv2 ViT-L/14).
        output_dim: Dimensionality of the projected embedding fed to FAM Core.
        hidden_dim: If > 0, insert a hidden layer of this size with ReLU
                    activation.  Defaults to 0 (linear projection only).
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if hidden_dim > 0:
            self.net: nn.Module = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalise embeddings.

        Args:
            x: Input tensor of shape ``(B, input_dim)``.

        Returns:
            L2-normalised tensor of shape ``(B, output_dim)``.
        """
        return F.normalize(self.net(x.float()), dim=-1)

    def triplet_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        margin: float = 1.0,
    ) -> torch.Tensor:
        """Compute Triplet Margin Loss in the projected space.

        Passes ``anchor``, ``positive``, and ``negative`` through the adapter
        projection, then applies ``torch.nn.TripletMarginLoss``.  The adapter
        gradients flow through all three arms so this can be used directly in a
        training loop.

        Args:
            anchor:   Anchor embeddings, shape ``(B, input_dim)``.
            positive: Same-class embeddings, shape ``(B, input_dim)``.
            negative: Hard-negative embeddings (different-but-similar class),
                      shape ``(B, input_dim)``.
            margin:   Margin for the triplet loss.  Defaults to 1.0.

        Returns:
            Scalar loss tensor.
        """
        criterion = nn.TripletMarginLoss(margin=margin, reduction="mean")
        return criterion(self(anchor), self(positive), self(negative))
