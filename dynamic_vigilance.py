"""dynamic_vigilance.py — Dynamic vigilance scheduling policies.

Two policies are provided:

DynamicVigilance (G29)
    Margin-based.  Computes a per-query effective vigilance threshold from the
    gap between the best-matching prototype and the strongest off-class
    competitor.  Calibrated for synthetic manifolds with well-separated,
    high-cosine prototypes.

RelativeVigilance (issue #73)
    Percentile-anchored.  Anchors the write gate to the live off-class
    similarity distribution rather than to fixed absolute cosine values.
    Designed for real text / frozen-feature manifolds where the attainable
    cosine range is compressed far below v_ceiling=0.95 (as shown by PR #72:
    real all-MiniLM-L6-v2 embeddings peak at ρ≈0.48, never reaching 0.95).

Protocol
--------
Both classes expose a ``compute(similarities, labels)`` method:

    similarities : (B, N) cosine-similarity matrix
    labels       : (N,) integer prototype class labels
    returns      : (v_eff, margins) each of shape (B,)

``ContinuousCAM.learn_local()`` calls ``dv.compute()`` and works with either
policy.  Both policies also expose ``v_floor``, ``v_ceiling`` attributes read
by ``probe_cross_class_similarity()``.

RetrievalFloorPolicy (issue #74)
    A separate, retrieval-side policy (not a vigilance scheduler). It adapts the
    inference similarity floor to the live within-class Δ so off-class
    prototypes below ``Δ - k*temp`` are masked before the softmax vote. Composes
    with either write-gate policy above. See its own docstring for details.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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


@dataclass
class RelativeVigilance:
    """Percentile-anchored vigilance calibrated to the live embedding manifold.

    PR #72 showed that on real text embeddings (all-MiniLM-L6-v2) the
    attainable inter-class cosine peaks around ρ≈0.48, far below the fixed
    v_ceiling=0.95 used by DynamicVigilance.  This policy anchors the write
    gate to the observed off-class similarity distribution instead:

        v_eff = clamp(rho_p + margin_guard, v_floor, v_ceiling)

    where ``rho_p`` is the ``percentile``-th quantile of top-1 off-class
    cosine similarities across the current query batch.  The gate therefore
    sits just above the observed cross-class cloud regardless of the manifold's
    absolute cosine range.

    Args:
        v_floor:      Minimum allowed effective vigilance.
        v_ceiling:    Maximum allowed effective vigilance (safety rail; also
                      read by ``probe_cross_class_similarity()`` for the
                      chimera-onset check).
        margin_guard: Safety margin added above the percentile anchor.
        percentile:   Quantile of the batch off-class sim distribution to
                      anchor on (0.95 = 95th percentile).
        ema_beta:     EMA smoothing coefficient applied across consecutive
                      ``compute()`` calls.  0.0 = no smoothing (fresh
                      per-batch estimate each time).
    """

    v_floor: float = 0.30
    v_ceiling: float = 0.95
    margin_guard: float = 0.05
    percentile: float = 0.95
    ema_beta: float = 0.0

    _rho_ema: float = field(default=float("nan"), init=False, repr=False)

    def __post_init__(self) -> None:
        if not (0.0 <= self.percentile <= 1.0):
            raise ValueError(f"percentile must be in [0, 1], got {self.percentile}")
        if not (0.0 <= self.ema_beta < 1.0):
            raise ValueError(f"ema_beta must be in [0, 1), got {self.ema_beta}")

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
            Tensor of shape ``(B,)`` — a single batch-level value broadcast to
            every query, clamped to ``[v_floor, v_ceiling]``.
        margins : torch.Tensor
            Tensor of shape ``(B,)`` with ``sim_best - sim_other`` per query.
        """
        if similarities.ndim != 2:
            raise ValueError(f"similarities must be (B, N), got {similarities.shape}")
        if labels.ndim != 1:
            raise ValueError(f"labels must be (N,), got {labels.shape}")
        if similarities.size(1) != labels.size(0):
            raise ValueError(
                f"similarities and labels mismatch: {similarities.size(1)} "
                f"prototypes vs {labels.size(0)} labels"
            )

        sim_best, best_idx = similarities.max(dim=1)          # (B,)
        best_classes = labels[best_idx]                        # (B,)

        other_mask = labels.unsqueeze(0) != best_classes.unsqueeze(1)  # (B, N)
        very_low = similarities.new_full((), -1e9)
        sims_other = torch.where(other_mask, similarities, very_low)
        sim_other, _ = sims_other.max(dim=1)                  # (B,)

        margins = sim_best - sim_other                         # (B,)

        # Live ρ percentile across valid (non-sentinel) off-class sims.
        valid_other = sim_other > (very_low / 2)
        if valid_other.any():
            rho_p_val = float(
                torch.quantile(sim_other[valid_other].float(), self.percentile).item()
            )
        else:
            # No cross-class competitors in the batch; fall back just below mean
            # within-class sim so the gate remains tighter than a random threshold.
            rho_p_val = float(sim_best.mean().item()) - 0.10

        # Optional EMA smoothing across calls.
        if self.ema_beta > 0.0 and not math.isnan(self._rho_ema):
            rho_p_val = self.ema_beta * self._rho_ema + (1.0 - self.ema_beta) * rho_p_val
        self._rho_ema = rho_p_val

        v_scalar = max(self.v_floor, min(self.v_ceiling, rho_p_val + self.margin_guard))
        v_eff = similarities.new_full((similarities.size(0),), v_scalar)
        return v_eff, margins


@dataclass
class RetrievalFloorPolicy:
    """Live Δ-relative inference similarity floor (issue #74).

    Maintains an EMA of observed within-class self-similarity (Δ) from
    confirmed same-class hits during ``learn_local()``.  At inference time
    the floor is placed ``k_logit_steps`` softmax-temperature steps below the
    running Δ estimate:

        floor = clamp(delta_ema - k * temp, floor_min, 1.0)

    Any prototype whose cosine similarity to the query falls below this floor
    is masked to ``-inf`` before the softmax vote, preventing off-class
    prototypes from accumulating aggregate vote mass even when they are
    individually similar (the root cause of retrieval-blend left-censoring
    on real text manifolds with small Δ, as diagnosed in PR #72 / issue #74).

    This is the retrieval-side analogue of ``RelativeVigilance`` (issue #73):
    the write gate adapts to the off-class distribution; the floor adapts to
    the within-class distribution.  The two policies compose.

    Args:
        k_logit_steps: Number of ``inference_temp`` steps below Δ to place
                       the floor.  Higher = more aggressive masking.
        floor_min:     Hard lower bound on the floor (0.0 = never negative).
        ema_beta:      EMA smoothing coefficient (higher = slower adaptation).
    """

    k_logit_steps: float = 3.0
    floor_min: float = 0.0
    ema_beta: float = 0.9

    _delta_ema: float = field(default=float("nan"), init=False, repr=False)

    def __post_init__(self) -> None:
        if self.k_logit_steps < 0:
            raise ValueError(f"k_logit_steps must be >= 0, got {self.k_logit_steps}")
        if not (0.0 <= self.ema_beta < 1.0):
            raise ValueError(f"ema_beta must be in [0, 1), got {self.ema_beta}")

    def update(self, within_class_sims: torch.Tensor) -> None:
        """Update the Δ EMA from a batch of within-class generalization sims.

        ``ContinuousCAM.learn_local()`` feeds the leave-one-out (2nd-best)
        same-class cosine by true label (see ``_within_class_loo``), NOT the
        vigilance-gated ``best_sims[hits]`` — the latter only sees near-duplicate
        hits and self-matches, which would inflate Δ. Pass a 1-D tensor of
        per-query within-class similarities; empty tensors are a no-op.
        """
        if within_class_sims.numel() == 0:
            return
        obs = float(within_class_sims.float().mean().item())
        if math.isnan(self._delta_ema):
            self._delta_ema = obs
        else:
            self._delta_ema = self.ema_beta * self._delta_ema + (1.0 - self.ema_beta) * obs

    def floor(self, temp: float) -> float | None:
        """Return the current similarity floor, or None if uninitialised.

        Returns ``None`` until the first ``update()`` call so the caller can
        distinguish "policy has no opinion yet" (fall back to the static floor)
        from "policy actively wants a floor of 0.0". Once initialised, returns
        ``clamp(delta_ema - k_logit_steps * temp, floor_min, 1.0)``.
        """
        if math.isnan(self._delta_ema):
            return None
        raw = self._delta_ema - self.k_logit_steps * temp
        return min(1.0, max(self.floor_min, raw))

    @property
    def delta_ema(self) -> float:
        """Current within-class Δ estimate (NaN if not yet initialised)."""
        return self._delta_ema


@dataclass
class RetrievalTruncationPolicy:
    """Adaptive vote-support truncation (issue #82 intervention).

    A per-probe, *relative-margin* cut applied to the top-k candidate cosine
    similarities **before** the softmax vote. For each probe row, candidates
    more than ``window_steps`` softmax-temperature steps below that row's own
    top-1 are masked to ``-inf``:

        keep[j]  iff  sim[j] >= top1 - window_steps * temp

    Unlike :class:`RetrievalFloorPolicy` (issue #74), which masks against a
    single *absolute* Δ-anchored floor shared by every probe, this cut is
    *relative to each probe's own top-1* — so the number of surviving voters
    adapts to the local neighbourhood geometry. In a confident neighbourhood
    (a tight within-class core with the off-class competitors sitting a margin
    below) the off-class tail is dropped and the vote sharpens onto the core;
    in a collapsed neighbourhood (within/off co-located, no margin) every
    candidate is within the window, so the cut is a near no-op.

    The top-1 candidate always survives (``top1 - top1 = 0 <= window``), so the
    truncation can only *sharpen toward the local winner*, never invert it.

    **The window is self-gating — this is the primary "do not gamble on the
    forced tail" guard.** In the recoverable regime a tight within-class core
    sits a clear margin above the off-class tail, so the window slices between
    them: the off-class tail is dropped and the (correct) core wins more
    cleanly. In the *forced* regime (issue #82: Δ−ρ negative, within/off
    co-located) there is no separated tail — every candidate sits within the
    window of the top-1 — so the cut removes (almost) nothing and the vote is a
    near no-op. The intervention therefore acts where the blend is *premature*
    and leaves the genuinely collapsed neighbourhood alone, without needing to
    know the labels. (Verified on a *separable-tail* neighbourhood: off-class
    vote mass collapses ~0.18→~0.00 while a co-located forced one barely moves.
    Caveat: when within- and off-class neighbours are *interleaved* in one narrow
    band — the real all-MiniLM 20NG mid-window — there is no separable tail, so
    the cut deflates the diffuse-vote support but cannot excise off-class mass
    selectively, and accuracy is a wash. See
    results/issue82_confidence_weighted_retrieval/README.md.)

    Both knobs are expressed in ``inference_temp`` (logit) steps, matching
    ``RetrievalFloorPolicy.k_logit_steps``: a candidate ``s`` steps below top-1
    carries softmax weight ``exp(-s)`` relative to it, so ``window_steps`` is
    directly the log-weight budget of the tail that is allowed to vote. The
    blend it targets is a *numerous* tail — many off-class candidates each
    carrying a small ``exp(-s)`` weight that sums to real vote mass and inflates
    ``effective_support`` — which is why dropping it both cuts off-class mass and
    deflates the effective support back toward the within-class core.

    ``gate_steps`` is an **optional, off-by-default** extra-conservative knob.
    When > 0 the cut is applied to a row only if its top1−top2 gap clears
    ``gate_steps`` steps:

        truncate[row]  iff  (top1 - top2) >= gate_steps * temp

    This restricts truncation to rows with a single clear winner and
    deliberately *skips* rows whose top-1 and top-2 are near-tied. Note this
    also skips recoverable rows that have within-class *siblings* near the top
    (top1 ≈ top2, both correct) — so it trades away mid-window gains for safety
    against sharpening a lone borderline winner. Leave it at 0.0 to rely on the
    window's self-gating (the recommended default); raise it only when you
    explicitly want to act on isolated-winner rows alone.

    Composes with the write-gate and the retrieval floor: it runs on whatever
    candidates survived the floor mask. Off by default (no policy attached) →
    retrieval is bit-identical to the pre-intervention baseline.

    Args:
        window_steps: Tail width in ``inference_temp`` steps below the per-row
                      top-1; candidates below ``top1 - window_steps*temp`` are
                      dropped. Smaller = more aggressive truncation. Must be > 0
                      for the policy to have any effect.
        gate_steps:   Optional minimum top1−top2 gap (in ``inference_temp``
                      steps) required to truncate a row. 0.0 = no gate (rely on
                      the window's self-gating, recommended). Higher = restrict
                      the cut to isolated-winner rows only.
    """

    window_steps: float = 2.0
    gate_steps: float = 0.0

    def __post_init__(self) -> None:
        if self.window_steps < 0:
            raise ValueError(
                f"window_steps must be >= 0, got {self.window_steps}")
        if self.gate_steps < 0:
            raise ValueError(
                f"gate_steps must be >= 0, got {self.gate_steps}")

    @property
    def active(self) -> bool:
        """Whether the policy can mask anything (window_steps > 0)."""
        return self.window_steps > 0.0

    def keep_mask(self, topk_sims: torch.Tensor, temp: float) -> torch.Tensor:
        """Per-candidate keep mask for the adaptive support truncation.

        Args:
            topk_sims: ``(B, k)`` candidate cosine similarities, in descending
                       order per row (as produced by ``topk``). May already
                       contain ``-inf`` entries masked out by an upstream floor;
                       those rows/entries are handled safely (see below).
            temp:      The softmax ``inference_temp`` the vote will use.

        Returns:
            ``(B, k)`` boolean tensor — ``True`` for candidates that should vote.
            When the policy is inactive every entry is ``True`` (no-op).

        The top-1 (column 0) always survives. A row whose top1−top2 gap is below
        ``gate_steps*temp`` is left fully intact (gate closed). Rows that were
        already fully ``-inf`` (e.g. floor-masked) yield an all-``True`` mask
        here — ``top1`` is ``-inf`` and ``sim >= -inf - w`` holds for every
        entry — so the existing fully-masked-row guard still owns that case.
        """
        if not self.active:
            return torch.ones_like(topk_sims, dtype=torch.bool)

        # Per-row top-1 (descending order ⇒ column 0). Use the tensor itself so
        # any upstream -inf masking is respected.
        top1 = topk_sims[:, :1]                                  # (B, 1)
        cutoff = top1 - self.window_steps * temp                 # (B, 1)
        keep = topk_sims >= cutoff                               # (B, k)

        # Confidence gate: only truncate rows with a clear local winner.
        if self.gate_steps > 0.0 and topk_sims.size(1) >= 2:
            top2 = topk_sims[:, 1:2]                             # (B, 1)
            gap = top1 - top2                                    # (B, 1), NaN if both -inf
            gate_open = gap >= self.gate_steps * temp            # (B, 1); NaN ⇒ False
            keep = keep | ~gate_open                             # closed-gate rows kept whole

        # The top-1 always votes (guards against degenerate temp/window combos).
        keep[:, 0] = True
        return keep
