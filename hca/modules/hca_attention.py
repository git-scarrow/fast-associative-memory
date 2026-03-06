"""HCAAttention: drop-in replacement for nn.MultiheadAttention using Lorentz geometry."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops.lorentz_distance import lorentz_squared_distance
from ..ops.lorentz_exp_map import lorentz_exp_map
from .curvature import CurvatureModule


class HCAAttention(nn.Module):
    """Hyperbolic Context Architecture attention module.

    Drop-in replacement for nn.MultiheadAttention. Uses Lorentz hyperboloid
    geometry for attention scoring with radial decay bias.

    Architecture:
        q_proj, k_proj, v_proj: nn.Linear (Euclidean)
        out_proj: nn.Linear (Euclidean)
        exp_map: project Q, K to Lorentz hyperboloid
        lorentz_score: LorentzSquaredDistance + decay bias
        curvature: CurvatureModule per layer
        gamma: softplus-parameterized decay strength
        radial_tracker: external (shared across layers)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        c_init: float = 1.0,
        gamma_init: float = 0.6,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Standard projections (Euclidean)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Curvature (per-layer)
        self.curvature = CurvatureModule(c_init=c_init, M=float(d_model))

        # Decay strength: softplus(gamma_raw) => gamma >= 0
        # softplus^{-1}(0.6) = log(exp(0.6) - 1) ~ 0.136
        gamma_raw_init = math.log(math.exp(gamma_init) - 1.0)
        self.gamma_raw = nn.Parameter(torch.tensor(gamma_raw_init))

        # Temperature beta = 1/sqrt(d_head)
        self.beta = 1.0 / math.sqrt(self.d_head)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Cached values for QoS
        self._cached_alpha_s = None
        self._cached_x0 = None

    def forward(self, query, key, value, need_weights=False, attn_mask=None):
        """Forward pass matching nn.MultiheadAttention signature.

        Args:
            query: [N, B, d_model] or [B, N, d_model]
            key: [N, B, d_model] or [B, N, d_model]
            value: [N, B, d_model] or [B, N, d_model]
            need_weights: return attention weights
            attn_mask: [N, N] or [B*H, N, N]

        Returns:
            output: same shape as query
            attn_weights: [B, H, N, N] if need_weights else None
        """
        # Handle [N, B, d] format (nn.MultiheadAttention default)
        is_batched_first = query.dim() == 3 and query.shape[0] != key.shape[0] if key.shape[0] != query.shape[0] else False
        # Assume batch_first=True for simplicity; transpose if needed
        if query.dim() == 3:
            B, N_q, _ = query.shape
            N_k = key.shape[1]
        else:
            raise ValueError(f"Expected 3D input, got {query.dim()}D")

        # Cast to fp32 if needed (H6: all Lorentz ops in fp32)
        query = query.float()
        key = key.float()
        value = value.float()

        # Linear projections
        Q = self.q_proj(query)  # [B, N_q, d_model]
        K = self.k_proj(key)    # [B, N_k, d_model]
        V = self.v_proj(value)  # [B, N_k, d_model]

        # Reshape to multi-head: [B, H, N, d_head]
        Q = Q.view(B, N_q, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, N_k, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, N_k, self.n_heads, self.d_head).transpose(1, 2)

        # Get curvature
        c = self.curvature(N_k)

        # Ensure fp32 for Lorentz ops
        Q = Q.float()
        K = K.float()

        # Project Q, K onto hyperboloid via exp map at origin
        Q_sp, Q_x0 = lorentz_exp_map(Q, c)  # Q_sp: [B,H,N_q,d_head], Q_x0: [B,H,N_q]
        K_sp, K_x0 = lorentz_exp_map(K, c)  # K_sp: [B,H,N_k,d_head], K_x0: [B,H,N_k]

        # Cache x0 for QoS compression check
        self._cached_x0 = K_x0.detach()

        # Compute Lorentz squared distance
        dist_sq = lorentz_squared_distance(Q_sp, K_sp, c)  # [B,H,N_q,N_k]

        # Gamma (decay strength)
        gamma = F.softplus(self.gamma_raw)

        # Radial decay bias: gamma * ln(K_x0)
        decay_bias = gamma * torch.log(K_x0.clamp(min=1e-7))  # [B,H,N_k]

        # Attention scores: -beta * dist_sq - decay_bias
        scores = -self.beta * dist_sq - decay_bias.unsqueeze(-2)  # [B,H,N_q,N_k]

        # Apply mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                scores = scores + attn_mask.unsqueeze(1)
            else:
                scores = scores + attn_mask

        # Softmax with max-subtraction (H3 — standard in F.softmax)
        alpha = F.softmax(scores, dim=-1)
        alpha = self.dropout(alpha)

        # alpha_s: mean attention to system token (index 0)
        self._cached_alpha_s = alpha[:, :, :, 0].mean().detach()

        # Weighted sum of V (Euclidean values)
        V = V.float()
        out = torch.matmul(alpha, V)  # [B, H, N_q, d_head]

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, N_q, self.d_model)
        out = self.out_proj(out)

        if need_weights:
            return out, alpha
        return out, None

    def get_alpha_s(self) -> float:
        """Return current alpha_s for QoS monitoring."""
        if self._cached_alpha_s is not None:
            return self._cached_alpha_s.item()
        return 0.5  # default before first forward

    def get_gamma(self) -> float:
        """Return current gamma value."""
        return F.softplus(self.gamma_raw).item()

    def get_c_eff(self, N=None) -> float:
        """Return current effective curvature."""
        return self.curvature(N).item()
