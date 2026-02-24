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

    Architecture is controlled by ``hidden_dim`` and optional keyword-only knobs
    used by the G30 adapter sweeps:

    * ``hidden_dim == 0`` (or ``layers == 1``): single linear projection
      ``nn.Linear(input_dim, proj_dim)``.
    * ``hidden_dim > 0`` and ``layers == 2``: two-layer MLP
      ``input_dim → hidden_dim → proj_dim`` with a configurable non-linearity
      between the layers. When ``residual=True`` and shapes match, a residual
      connection is applied: ``y ← y + x`` where ``x`` is the first linear
      output.

    The ``forward`` pass always L2-normalises the projection output so that
    downstream cosine-based vigilance comparisons remain meaningful.

    Args:
        input_dim:  Dimensionality of the backbone embedding (e.g. 1024 for
                    DINOv2 ViT-L/14).
        output_dim: Dimensionality of the projected embedding fed to FAM Core.
        hidden_dim: If > 0, insert a hidden layer of this size with ReLU
                    activation (default).  Defaults to 0 (linear projection).
        nonlinearity: Activation between layers for 2-layer adapters. One of
                    ``"relu"``, ``"gelu"``, or ``"none"``.
        proj_dim:    Optional explicit projection dimension. When ``None``,
                    defaults to ``output_dim``.
        layers:      Number of learnable layers (1 or 2).
        residual:    Enable residual connection for 2-layer adapters when the
                    first-layer output and final output shapes match.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 0,
        *,
        nonlinearity: str = "relu",
        proj_dim: int | None = None,
        layers: int = 2,
        residual: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.residual = residual

        self.proj_dim = proj_dim if proj_dim is not None else output_dim

        nl = nonlinearity.lower()
        if nl == "relu":
            act: nn.Module = nn.ReLU()
        elif nl == "gelu":
            act = nn.GELU()
        elif nl in ("silu", "swish"):
            act = nn.SiLU()
        elif nl in ("none", "linear", "identity"):
            act = nn.Identity()
        else:
            raise ValueError(
                f"Unsupported nonlinearity {nonlinearity!r}; "
                "expected 'relu', 'gelu', 'silu', or 'none'."
            )

        if hidden_dim > 0 and layers >= 2:
            # Build a small MLP: input → hidden → ... → hidden → proj_dim.
            modules: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), act]
            # Additional hidden layers (if any)
            for _ in range(2, layers):
                modules.append(nn.Linear(hidden_dim, hidden_dim))
                modules.append(act)
            modules.append(nn.Linear(hidden_dim, self.proj_dim))
            self.net = nn.Sequential(*modules)
        else:
            # Single linear projection when no hidden_dim or layers == 1.
            self.net = nn.Linear(input_dim, self.proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalise embeddings.

        Args:
            x: Input tensor of shape ``(B, input_dim)``.

        Returns:
            L2-normalised tensor of shape ``(B, output_dim)``.
        """
        x = x.float()

        if isinstance(self.net, nn.Sequential) and self.residual:
            # Residual connection from the first hidden representation to the
            # final output when shapes are compatible.
            first = self.net[0](x)
            h = self.net[1](first)
            for layer in self.net[2:-1]:
                h = layer(h)
            out = self.net[-1](h)
            if out.shape == first.shape:
                out = out + first
        else:
            out = self.net(x)

        return F.normalize(out, dim=-1)

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
