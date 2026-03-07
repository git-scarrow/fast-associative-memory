"""HCAAttention: drop-in replacement for nn.MultiheadAttention using Lorentz geometry."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops.lorentz_distance import lorentz_squared_distance
from ..ops.lorentz_exp_map import lorentz_exp_map
from ..ops.fused_attention import fused_hca_attention, HAS_TRITON
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
        use_fused: bool = False,
        amp_enabled: bool = False,
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

        # Fused kernel + bf16 mixed precision
        self.use_fused = use_fused and HAS_TRITON
        self.amp_enabled = amp_enabled

        # Cached values for QoS
        self._cached_alpha_s = None
        self._cached_x0 = None

    def forward(self, query, key, value, need_weights=False, attn_mask=None,
                is_causal=False):
        """Forward pass matching nn.MultiheadAttention signature.

        Args:
            query: [B, N, d_model]
            key: [B, N, d_model]
            value: [B, N, d_model]
            need_weights: return attention weights
            attn_mask: [N, N] or [B*H, N, N]
            is_causal: use causal mask (enables fused kernel path)

        Returns:
            output: same shape as query
            attn_weights: [B, H, N, N] if need_weights else None
        """
        if query.dim() != 3:
            raise ValueError(f"Expected 3D input, got {query.dim()}D")

        B, N_q, _ = query.shape
        N_k = key.shape[1]

        on_cuda = query.device.type == 'cuda'
        use_amp = self.amp_enabled and on_cuda

        # Linear projections (bf16 autocast when enabled)
        if use_amp:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                Q = self.q_proj(query)
                K = self.k_proj(key)
                V = self.v_proj(value)
        else:
            Q = self.q_proj(query.float())
            K = self.k_proj(key.float())
            V = self.v_proj(value.float())

        # Reshape to multi-head: [B, H, N, d_head]
        Q = Q.view(B, N_q, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, N_k, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, N_k, self.n_heads, self.d_head).transpose(1, 2)

        # Get curvature
        c = self.curvature(N_k)

        # fp32 for Lorentz ops (H6)
        Q_f = Q.float()
        K_f = K.float()

        # Project Q, K onto hyperboloid via exp map at origin
        Q_sp, Q_x0 = lorentz_exp_map(Q_f, c)
        K_sp, K_x0 = lorentz_exp_map(K_f, c)

        # Cache x0 for QoS compression check
        self._cached_x0 = K_x0.detach()

        # Gamma (decay strength)
        gamma = F.softplus(self.gamma_raw)
        inv_c = 1.0 / c.item()

        # Fused path: no attn_mask, no need_weights, CUDA, Triton available
        can_fuse = (self.use_fused and on_cuda and not need_weights
                    and attn_mask is None)

        if can_fuse:
            BH = B * self.n_heads
            Q_sp_r = Q_sp.reshape(BH, N_q, self.d_head)
            K_sp_r = K_sp.reshape(BH, N_k, self.d_head)
            V_r = V.float().reshape(BH, N_k, self.d_head)

            out = fused_hca_attention(
                Q_sp_r, K_sp_r, V_r, inv_c, self.beta, gamma,
                is_causal=is_causal,
            )
            out = out.view(B, self.n_heads, N_q, self.d_head)
            alpha = None
        else:
            # Unfused path
            dist_sq = lorentz_squared_distance(Q_sp, K_sp, c)
            decay_bias = gamma * torch.log(K_x0.clamp(min=1e-7))
            scores = -self.beta * dist_sq - decay_bias.unsqueeze(-2)

            if is_causal and attn_mask is None:
                causal = torch.triu(
                    torch.full((N_q, N_k), float("-inf"), device=query.device),
                    diagonal=1,
                )
                scores = scores + causal.unsqueeze(0).unsqueeze(0)
            elif attn_mask is not None:
                if attn_mask.dim() == 2:
                    scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
                elif attn_mask.dim() == 3:
                    scores = scores + attn_mask.unsqueeze(1)
                else:
                    scores = scores + attn_mask

            alpha = F.softmax(scores, dim=-1)
            alpha = self.dropout(alpha)
            self._cached_alpha_s = alpha[:, :, :, 0].mean().detach()

            V_f = V.float()
            out = torch.matmul(alpha, V_f)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, N_q, self.d_model)

        if use_amp:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = self.out_proj(out)
            out = out.float()
        else:
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
