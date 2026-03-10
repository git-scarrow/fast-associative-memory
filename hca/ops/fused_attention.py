"""Fused HCA attention: Lorentz distance + decay + softmax + V accumulation.

Forward uses a Triton kernel that avoids materializing the N*N attention matrix.
Backward uses two Triton kernels (dQ and dK/dV) following the FlashAttention-2
strategy: tiled recomputation of attention weights using stored LSE.

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
            spatial = tl.dot(q, tl.trans(k), input_precision="tf32")
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
            acc += tl.dot(p.to(tl.float32), v, input_precision="tf32")

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

    # ---- Backward kernels (DS-13) ----

    @triton.jit
    def _hca_attn_bwd_dq(
        Q, K, V, LSE, Delta, dO,
        dQ, dGamma_partial,
        inv_c, beta, gamma,
        N_Q, N_K,
        stride_qb, stride_qn, stride_qd,
        stride_kb, stride_kn, stride_kd,
        stride_vb, stride_vn, stride_vd,
        stride_lb, stride_ln,
        stride_deltab, stride_deltan,
        stride_dob, stride_don, stride_dod,
        stride_dqb, stride_dqn, stride_dqd,
        stride_dgb, stride_dgn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        """Compute dQ and partial gamma gradient. Parallelized over Q tiles."""
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < N_Q

        # Load Q tile
        q_ptrs = Q + pid_bh * stride_qb + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        q_x0 = tl.sqrt(inv_c + tl.sum(q * q, axis=1))

        # Load dO, Delta, LSE for this Q tile
        do_ptrs = dO + pid_bh * stride_dob + offs_m[:, None] * stride_don + offs_d[None, :] * stride_dod
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        delta_ptrs = Delta + pid_bh * stride_deltab + offs_m * stride_deltan
        delta = tl.load(delta_ptrs, mask=mask_m, other=0.0)

        lse_ptrs = LSE + pid_bh * stride_lb + offs_m * stride_ln
        lse = tl.load(lse_ptrs, mask=mask_m, other=0.0)

        # Accumulators
        dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        gq_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
        dgamma_acc = tl.zeros([BLOCK_M], dtype=tl.float32)

        for start_n in range(0, N_K, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)

            # Load K, V tiles
            k_ptrs = K + pid_bh * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=(offs_n[:, None] < N_K), other=0.0)
            k_x0 = tl.sqrt(inv_c + tl.sum(k * k, axis=1))

            v_ptrs = V + pid_bh * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=(offs_n[:, None] < N_K), other=0.0).to(tl.float32)

            # Recompute scores (identical to forward)
            temporal = -(q_x0[:, None] * k_x0[None, :])
            spatial = tl.dot(q, tl.trans(k), input_precision="tf32")
            lip = temporal + spatial
            dsq = tl.maximum(-lip - inv_c, 0.0)
            log_kx0 = tl.log(tl.maximum(k_x0, 1e-7))
            s = -beta * dsq - gamma * log_kx0[None, :]

            s = tl.where(offs_n[None, :] < N_K, s, float('-inf'))
            if IS_CAUSAL:
                s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float('-inf'))

            # Recompute attention weights using stored LSE
            p = tl.exp(s - lse[:, None])

            # Softmax backward: dS = P * (dO @ V^T - Delta)
            dp = tl.dot(do, tl.trans(v), input_precision="tf32")
            ds = p * (dp - delta[:, None])

            # HCA score backward: d_lip = beta * ds * active
            active = ((-lip - inv_c) > 0.0).to(tl.float32)
            d_lip = beta * ds * active

            # Spatial gradient for Q: dQ += d_lip @ K
            dq_acc += tl.dot(d_lip.to(tl.float32), k, input_precision="tf32")

            # Temporal gradient accumulation (negated after loop)
            gq_acc += tl.sum(d_lip * k_x0[None, :], axis=1)

            # Gamma gradient accumulation (negated after loop)
            dgamma_acc += tl.sum(ds * log_kx0[None, :], axis=1)

        # Finalize temporal contribution: dQ += Q * (-Gq / q_x0)
        gq_acc = -gq_acc
        dq_acc += q * (gq_acc / tl.maximum(q_x0, 1e-14))[:, None]

        # Store dQ
        dq_ptrs = dQ + pid_bh * stride_dqb + offs_m[:, None] * stride_dqn + offs_d[None, :] * stride_dqd
        tl.store(dq_ptrs, dq_acc, mask=mask_m[:, None])

        # Store partial gamma gradient
        dgamma_acc = -dgamma_acc
        dg_ptrs = dGamma_partial + pid_bh * stride_dgb + offs_m * stride_dgn
        tl.store(dg_ptrs, dgamma_acc, mask=mask_m)

    @triton.jit
    def _hca_attn_bwd_dkv(
        Q, K, V, LSE, Delta, dO,
        dK, dV,
        inv_c, beta, gamma,
        N_Q, N_K,
        stride_qb, stride_qn, stride_qd,
        stride_kb, stride_kn, stride_kd,
        stride_vb, stride_vn, stride_vd,
        stride_lb, stride_ln,
        stride_deltab, stride_deltan,
        stride_dob, stride_don, stride_dod,
        stride_dkb, stride_dkn, stride_dkd,
        stride_dvb, stride_dvn, stride_dvd,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        """Compute dK and dV. Parallelized over K/V tiles."""
        pid_n = tl.program_id(0)
        pid_bh = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        mask_n = offs_n < N_K

        # Load K, V tiles (constant across Q-tile loop)
        k_ptrs = K + pid_bh * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        k_x0 = tl.sqrt(inv_c + tl.sum(k * k, axis=1))
        log_kx0 = tl.log(tl.maximum(k_x0, 1e-7))

        v_ptrs = V + pid_bh * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        # Accumulators
        dk_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        dv_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        gk_ip_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        gk_decay_raw = tl.zeros([BLOCK_N], dtype=tl.float32)

        for start_m in range(0, N_Q, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)

            # Load Q tile
            q_ptrs = Q + pid_bh * stride_qb + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
            q = tl.load(q_ptrs, mask=(offs_m[:, None] < N_Q), other=0.0)
            q_x0 = tl.sqrt(inv_c + tl.sum(q * q, axis=1))

            # Load dO, Delta, LSE
            do_ptrs = dO + pid_bh * stride_dob + offs_m[:, None] * stride_don + offs_d[None, :] * stride_dod
            do = tl.load(do_ptrs, mask=(offs_m[:, None] < N_Q), other=0.0).to(tl.float32)

            delta_ptrs = Delta + pid_bh * stride_deltab + offs_m * stride_deltan
            delta = tl.load(delta_ptrs, mask=(offs_m < N_Q), other=0.0)

            lse_ptrs = LSE + pid_bh * stride_lb + offs_m * stride_ln
            lse = tl.load(lse_ptrs, mask=(offs_m < N_Q), other=0.0)

            # Recompute scores (identical to forward)
            temporal = -(q_x0[:, None] * k_x0[None, :])
            spatial = tl.dot(q, tl.trans(k), input_precision="tf32")
            lip = temporal + spatial
            dsq = tl.maximum(-lip - inv_c, 0.0)
            s = -beta * dsq - gamma * log_kx0[None, :]

            s = tl.where(offs_n[None, :] < N_K, s, float('-inf'))
            if IS_CAUSAL:
                s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float('-inf'))

            # Recompute attention weights using stored LSE
            p = tl.exp(s - lse[:, None])

            # dV += P^T @ dO
            dv_acc += tl.dot(tl.trans(p.to(tl.float32)), do, input_precision="tf32")

            # Softmax backward: dS = P * (dO @ V^T - Delta)
            dp = tl.dot(do, tl.trans(v), input_precision="tf32")
            ds = p * (dp - delta[:, None])

            # HCA score backward
            active = ((-lip - inv_c) > 0.0).to(tl.float32)
            d_lip = beta * ds * active

            # Spatial gradient for K: dK += d_lip^T @ Q
            dk_acc += tl.dot(tl.trans(d_lip.to(tl.float32)), q, input_precision="tf32")

            # Temporal gradients (finalized after loop)
            gk_ip_acc += tl.sum(d_lip * q_x0[:, None], axis=0)
            gk_decay_raw += tl.sum(ds, axis=0)

        # Finalize temporal contribution to dK
        # Gk_ip = -sum_i(d_lip_ij * q_x0_i), chain through k_x0: / k_x0
        # Gk_decay = -gamma * sum_i(dS_ij) / k_x0^2
        gk_ip = -gk_ip_acc / tl.maximum(k_x0, 1e-14)
        gk_decay = -gamma * gk_decay_raw / tl.maximum(k_x0 * k_x0, 1e-14)
        dk_acc += k * (gk_ip + gk_decay)[:, None]

        # Store dK, dV
        dk_ptrs = dK + pid_bh * stride_dkb + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
        tl.store(dk_ptrs, dk_acc, mask=mask_n[:, None])

        dv_ptrs = dV + pid_bh * stride_dvb + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
        tl.store(dv_ptrs, dv_acc, mask=mask_n[:, None])


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


def _triton_bwd(Q_sp, K_sp, V, Out, LSE, grad_out, inv_c, beta, gamma_val, is_causal):
    """Launch Triton kernels for fused HCA attention backward (DS-13)."""
    BH, N_Q, D = Q_sp.shape
    _, N_K, _ = K_sp.shape

    BD = _next_pow2(D)
    BM, BN = 64, 64

    Q_p = _pad_to(Q_sp.contiguous(), BD)
    K_p = _pad_to(K_sp.contiguous(), BD)
    V_p = _pad_to(V.contiguous(), BD)
    dO_p = _pad_to(grad_out.contiguous(), BD)

    # Delta_i = sum_d(Out_id * dO_id) — needed for softmax backward
    Delta = (Out * grad_out).sum(-1)  # [BH, N_Q]

    # ---- dQ kernel ----
    dQ = torch.empty(BH, N_Q, BD, device=Q_sp.device, dtype=torch.float32)
    dGamma_partial = torch.empty(BH, N_Q, device=Q_sp.device, dtype=torch.float32)

    grid_dq = (math.ceil(N_Q / BM), BH)
    _hca_attn_bwd_dq[grid_dq](
        Q_p, K_p, V_p, LSE, Delta, dO_p,
        dQ, dGamma_partial,
        float(inv_c), float(beta), float(gamma_val),
        N_Q, N_K,
        Q_p.stride(0), Q_p.stride(1), Q_p.stride(2),
        K_p.stride(0), K_p.stride(1), K_p.stride(2),
        V_p.stride(0), V_p.stride(1), V_p.stride(2),
        LSE.stride(0), LSE.stride(1),
        Delta.stride(0), Delta.stride(1),
        dO_p.stride(0), dO_p.stride(1), dO_p.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2),
        dGamma_partial.stride(0), dGamma_partial.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=BD,
        IS_CAUSAL=is_causal,
        num_stages=1,
    )

    # ---- dK/dV kernel ----
    dK = torch.empty(BH, N_K, BD, device=Q_sp.device, dtype=torch.float32)
    dV = torch.empty(BH, N_K, BD, device=Q_sp.device, dtype=torch.float32)

    grid_dkv = (math.ceil(N_K / BN), BH)
    _hca_attn_bwd_dkv[grid_dkv](
        Q_p, K_p, V_p, LSE, Delta, dO_p,
        dK, dV,
        float(inv_c), float(beta), float(gamma_val),
        N_Q, N_K,
        Q_p.stride(0), Q_p.stride(1), Q_p.stride(2),
        K_p.stride(0), K_p.stride(1), K_p.stride(2),
        V_p.stride(0), V_p.stride(1), V_p.stride(2),
        LSE.stride(0), LSE.stride(1),
        Delta.stride(0), Delta.stride(1),
        dO_p.stride(0), dO_p.stride(1), dO_p.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2),
        dV.stride(0), dV.stride(1), dV.stride(2),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=BD,
        IS_CAUSAL=is_causal,
        num_stages=1,
    )

    grad_gamma = dGamma_partial.sum()

    return (
        dQ[:, :, :D].contiguous(),
        dK[:, :, :D].contiguous(),
        dV[:, :, :D].contiguous(),
        grad_gamma,
    )


class _FusedHCAAttn(Function):
    """Fused forward + backward (Triton). DS-13: backward uses two-kernel split."""

    @staticmethod
    def forward(ctx, Q_sp, K_sp, V, inv_c, beta, gamma, is_causal):
        Q_f = Q_sp.contiguous().float()
        K_f = K_sp.contiguous().float()
        V_f = V.contiguous().float()

        Out, LSE = _triton_fwd(Q_f, K_f, V_f, inv_c, beta, gamma.item(), is_causal)

        ctx.save_for_backward(Q_f, K_f, V_f, gamma, Out, LSE)
        ctx.inv_c = inv_c
        ctx.beta = beta
        ctx.is_causal = is_causal
        return Out

    @staticmethod
    def backward(ctx, grad_out):
        Q_sp, K_sp, V, gamma, Out, LSE = ctx.saved_tensors
        grad_out = grad_out.contiguous().float()

        dQ, dK, dV, grad_gamma = _triton_bwd(
            Q_sp, K_sp, V, Out, LSE, grad_out,
            ctx.inv_c, ctx.beta, gamma.item(), ctx.is_causal,
        )

        return dQ, dK, dV, None, None, grad_gamma, None


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
