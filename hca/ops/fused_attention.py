"""Fused HCA attention: Lorentz distance + decay + softmax + V accumulation.

Forward uses a Triton kernel that avoids materializing the N*N attention matrix.
Backward recomputes attention weights using standard PyTorch ops.

Inputs are spatial coordinates on the hyperboloid (after exp_map).
Temporal coordinates are derived from spatial coords (H7 mitigation).
"""

import math
import torch
import torch.nn.functional as F
from torch.autograd import Function

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

_MIN_DOT = 16


def _next_pow2(n):
    n = max(n, _MIN_DOT)
    return 1 << (n - 1).bit_length()


def _pad_to(x, target_d):
    d = x.shape[-1]
    if d < target_d:
        return F.pad(x, (0, target_d - d))
    return x


if HAS_TRITON:
    @triton.jit
    def _hca_attn_fwd(
        Q, K, V, Out, LSE,
        inv_c, beta, gamma,
        N_Q, N_K,
        stride_qb, stride_qn, stride_qd,
        stride_kb, stride_kn, stride_kd,
        stride_vb, stride_vn, stride_vd,
        stride_ob, stride_on, stride_od,
        stride_lb, stride_ln,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        # Load Q tile [BLOCK_M, BLOCK_D]
        q_ptrs = Q + pid_bh * stride_qb + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_Q, other=0.0)

        # Derive Q temporal coords: x0 = sqrt(1/c + ||x_sp||^2)
        q_x0 = tl.sqrt(inv_c + tl.sum(q * q, axis=1))

        # Online softmax accumulators
        m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        for start_n in range(0, N_K, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)

            # Load K tile [BLOCK_N, BLOCK_D]
            k_ptrs = K + pid_bh * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=offs_n[:, None] < N_K, other=0.0)

            # Load V tile [BLOCK_N, BLOCK_D]
            v_ptrs = V + pid_bh * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=offs_n[:, None] < N_K, other=0.0).to(tl.float32)

            # Derive K temporal coords
            k_x0 = tl.sqrt(inv_c + tl.sum(k * k, axis=1))

            # Lorentz inner product: -q_x0 * k_x0 + q_sp @ k_sp^T
            temporal = -(q_x0[:, None] * k_x0[None, :])
            spatial = tl.dot(q, tl.trans(k))
            lip = temporal + spatial

            # Distance proxy: max(-IP - 1/c, 0)
            dsq = tl.maximum(-lip - inv_c, 0.0)

            # Scores: -beta * dist_sq - gamma * log(k_x0)
            log_kx0 = tl.log(tl.maximum(k_x0, 1e-7))
            s = -beta * dsq - gamma * log_kx0[None, :]

            # Out-of-bounds masking
            s = tl.where(offs_n[None, :] < N_K, s, float('-inf'))
            if IS_CAUSAL:
                s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float('-inf'))

            # Online softmax update
            m_ij = tl.max(s, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            correction = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_new = correction * l_i + tl.sum(p, axis=1)

            # Rescale accumulator and add new contribution
            acc = acc * correction[:, None]
            acc += tl.dot(p.to(tl.float32), v)

            m_i = m_new
            l_i = l_new

        # Final normalization
        acc = acc / tl.maximum(l_i, 1e-10)[:, None]

        # Store output
        out_ptrs = Out + pid_bh * stride_ob + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
        tl.store(out_ptrs, acc, mask=offs_m[:, None] < N_Q)

        # Store logsumexp for backward
        lse_ptrs = LSE + pid_bh * stride_lb + offs_m * stride_ln
        tl.store(lse_ptrs, m_i + tl.log(tl.maximum(l_i, 1e-10)), mask=offs_m < N_Q)


def _triton_fwd(Q_sp, K_sp, V, inv_c, beta, gamma_val, is_causal):
    """Launch Triton kernel for fused HCA attention forward."""
    BH, N_Q, D = Q_sp.shape
    _, N_K, _ = K_sp.shape

    BD = _next_pow2(D)
    BM, BN = 64, 64

    Q_p = _pad_to(Q_sp.contiguous(), BD)
    K_p = _pad_to(K_sp.contiguous(), BD)
    V_p = _pad_to(V.contiguous(), BD)

    Out = torch.empty(BH, N_Q, BD, device=Q_sp.device, dtype=torch.float32)
    LSE = torch.empty(BH, N_Q, device=Q_sp.device, dtype=torch.float32)

    grid = (math.ceil(N_Q / BM), BH)

    _hca_attn_fwd[grid](
        Q_p, K_p, V_p, Out, LSE,
        float(inv_c), float(beta), float(gamma_val),
        N_Q, N_K,
        Q_p.stride(0), Q_p.stride(1), Q_p.stride(2),
        K_p.stride(0), K_p.stride(1), K_p.stride(2),
        V_p.stride(0), V_p.stride(1), V_p.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        LSE.stride(0), LSE.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=BD,
        IS_CAUSAL=is_causal,
    )

    return Out[:, :, :D].contiguous(), LSE


class _FusedHCAAttn(Function):
    """Fused forward (Triton), recompute backward (PyTorch).

    Memory optimization: aggressive del of NxN intermediates keeps peak at
    3 NxN tensors (lip, grad_scores, grad_lip) vs 11+ in naive approach.
    """

    @staticmethod
    def forward(ctx, Q_sp, K_sp, V, inv_c, beta, gamma, is_causal):
        Q_f = Q_sp.contiguous().float()
        K_f = K_sp.contiguous().float()
        V_f = V.contiguous().float()

        Out, _LSE = _triton_fwd(Q_f, K_f, V_f, inv_c, beta, gamma.item(), is_causal)

        # Only save what backward needs (Out and LSE not needed)
        ctx.save_for_backward(Q_f, K_f, V_f, gamma)
        ctx.inv_c = inv_c
        ctx.beta = beta
        ctx.is_causal = is_causal
        return Out

    @staticmethod
    def backward(ctx, grad_out):
        Q_sp, K_sp, V, gamma = ctx.saved_tensors
        inv_c = ctx.inv_c
        beta = ctx.beta
        is_causal = ctx.is_causal
        gamma_val = gamma.item()

        # Derive temporal coords (H7)
        Q_x0 = torch.sqrt((inv_c + (Q_sp * Q_sp).sum(-1)).clamp(min=1e-14))
        K_x0 = torch.sqrt((inv_c + (K_sp * K_sp).sum(-1)).clamp(min=1e-14))
        log_kx0 = torch.log(K_x0.clamp(min=1e-7))

        # Recompute Lorentz IP (NxN #1)
        lip = -(Q_x0.unsqueeze(-1) * K_x0.unsqueeze(-2))
        lip = lip + torch.bmm(Q_sp, K_sp.transpose(-2, -1))

        # Scores -> alpha (NxN #2, then #1 freed via reuse)
        scores = (-beta * (-lip - inv_c).clamp(min=0.0)
                  - gamma_val * log_kx0.unsqueeze(-2))

        if is_causal:
            Nq, Nk = scores.shape[-2], scores.shape[-1]
            causal = torch.triu(
                torch.ones(Nq, Nk, device=scores.device, dtype=torch.bool),
                diagonal=1,
            )
            scores.masked_fill_(causal, float('-inf'))

        alpha = F.softmax(scores, dim=-1)
        del scores  # live NxN: lip, alpha

        # Attention backward
        grad_V = torch.bmm(alpha.transpose(-2, -1), grad_out)
        grad_alpha = torch.bmm(grad_out, V.transpose(-2, -1))
        grad_scores = alpha * (grad_alpha - (alpha * grad_alpha).sum(-1, keepdim=True))
        del alpha, grad_alpha  # live NxN: lip, grad_scores

        # gamma gradient
        grad_gamma = -(grad_scores * log_kx0.unsqueeze(-2)).sum()

        # Decay gradient through K_sp (needs grad_scores)
        Gk_decay = -gamma_val * grad_scores.sum(-2) / (K_x0 * K_x0).clamp(min=1e-14)

        # Through scores -> dist_sq -> lip
        active = ((-lip - inv_c) > 0).float()
        grad_lip = (beta * grad_scores * active)
        del grad_scores, active  # live NxN: lip, grad_lip

        # Spatial gradients
        grad_Q_sp = torch.bmm(grad_lip, K_sp)
        grad_K_sp = torch.bmm(grad_lip.transpose(-2, -1), Q_sp)

        # Temporal gradients through derived x0
        Gq = -(grad_lip * K_x0.unsqueeze(-2)).sum(-1)
        Gk_ip = -(Q_x0.unsqueeze(-1) * grad_lip).sum(-2)
        del grad_lip, lip  # live NxN: 0

        grad_Q_sp = grad_Q_sp + Q_sp * (Gq / Q_x0.clamp(min=1e-14)).unsqueeze(-1)
        grad_K_sp = (grad_K_sp
                     + K_sp * ((Gk_ip / K_x0.clamp(min=1e-14)) + Gk_decay).unsqueeze(-1))

        return grad_Q_sp, grad_K_sp, grad_V, None, None, grad_gamma, None


def fused_hca_attention(Q_sp, K_sp, V, inv_c, beta, gamma, is_causal=False):
    """Fused HCA attention: Lorentz distance + decay + softmax + V accumulation.

    Args:
        Q_sp: [BH, N_q, D] spatial coords on hyperboloid (fp32)
        K_sp: [BH, N_k, D] spatial coords on hyperboloid (fp32)
        V: [BH, N_k, D] value vectors (fp32 or bf16)
        inv_c: float, 1/curvature
        beta: float, temperature 1/sqrt(d_head)
        gamma: scalar tensor (needs gradient), decay strength
        is_causal: bool, apply causal mask

    Returns:
        Out: [BH, N_q, D] attention output (fp32)
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton required for fused HCA attention")
    return _FusedHCAAttn.apply(Q_sp, K_sp, V, inv_c, beta, gamma, is_causal)
