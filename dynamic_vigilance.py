"""dynamic_vigilance.py — Margin-based dynamic vigilance scheduling (G29).

Computes a per-query effective vigilance threshold based on the margin between
the best-matching prototype and the strongest competitor from a *different*
class.  Larger margins yield lower effective vigilance (more lenient), while
small margins keep vigilance close to the base value.

Formula
-------
Given a similarity matrix ``S`` of shape ``(B, N)`` and prototype class labels
``L`` of shape ``(N,)``:

1. For each query ``b`` find the best-matching prototype:

   ``sim_best[b] = max_j S[b, j]`` and ``c_best[b] = L[argmax_j S[b, j]]``.

2. Among prototypes whose class differs from ``c_best[b]``, find the strongest
   competitor similarity ``sim_other[b]``.

3. Margin: ``margin[b] = sim_best[b] - sim_other[b]``.

4. Effective vigilance per query:

   ``v_eff[b] = clamp(v_base - alpha * margin[b], v_floor, v_ceiling)``.

The default hyperparameters follow the spec in issue #43.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DynamicVigilance:
    """Margin-based dynamic vigilance scheduler.

    Args:
        v_base:    Baseline vigilance level.
        alpha:     Slope for margin → vigilance mapping.
        v_floor:   Minimum allowed vigilance.
        v_ceiling: Maximum allowed vigilance.
    """

    v_base: float = 0.92
    alpha: float = 0.3
    v_floor: float = 0.30
    v_ceiling: float = 0.95

    def compute(self, similarities: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-query effective vigilance and margins.

        Parameters
        ----------
        similarities:
            Tensor of shape ``(B, N)`` with cosine similarities between queries
            and all occupied prototypes.
        labels:
            Tensor of shape ``(N,)`` with integer class indices for prototypes.

        Returns
        -------
        v_eff : torch.Tensor
            Tensor of shape ``(B,)`` with per-query vigilance values after
            clamping to ``[v_floor, v_ceiling]``.
        margins : torch.Tensor
            Tensor of shape ``(B,)`` with ``sim_best - sim_second_best_other_class``.
        """
        if similarities.ndim != 2:
            raise ValueError(f"similarities must be (B, N), got {similarities.shape}")
        if labels.ndim != 1:
            raise ValueError(f"labels must be (N,), got {labels.shape}")
        if similarities.size(1) != labels.size(0):
            raise ValueError(
                f"similarities and labels mismatch: {similarities.size(1)} prototypes vs {labels.size(0)} labels"
            )

        # Best match per query
        sim_best, best_idx = similarities.max(dim=1)  # (B,)
        best_classes = labels[best_idx]  # (B,)

        # Mask prototypes whose class differs from the best-matching prototype's class
        # Broadcast labels to (B, N) and compare against best_classes[:, None].
        other_mask = labels.unsqueeze(0) != best_classes.unsqueeze(1)  # (B, N)

        # For numerical stability, replace non-other entries with a very low similarity
        # so they never win the max. If no other-class prototype exists in a row,
        # the max will be this floor value, yielding a very large margin which will
        # be clamped to ``v_floor``.
        very_low = similarities.new_full((), -1e9)
        sims_other = torch.where(other_mask, similarities, very_low)
        sim_other, _ = sims_other.max(dim=1)  # (B,)

        margins = sim_best - sim_other

        v_eff = self.v_base - self.alpha * margins
        v_eff = torch.clamp(v_eff, self.v_floor, self.v_ceiling)
        return v_eff, margins
