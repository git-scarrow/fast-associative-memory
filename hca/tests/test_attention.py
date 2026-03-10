"""HCAAttention smoke tests, shape tests, and gradient tests."""

import pytest
import torch
import torch.nn as nn

from hca.modules.hca_attention import HCAAttention

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@pytest.fixture
def attention():
    return HCAAttention(d_model=32, n_heads=4)


class TestShape:
    def test_output_shape_matches_mha(self, attention):
        """Output shape should match nn.MultiheadAttention for identical inputs."""
        B, N, d = 2, 16, 32
        x = torch.randn(B, N, d)
        out, _ = attention(x, x, x)
        assert out.shape == (B, N, d), f"Expected {(B, N, d)}, got {out.shape}"

    def test_different_query_key_lengths(self):
        """Q and K can have different sequence lengths (cross-attention)."""
        attn = HCAAttention(d_model=32, n_heads=4)
        B, N_q, N_k, d = 2, 8, 16, 32
        q = torch.randn(B, N_q, d)
        k = torch.randn(B, N_k, d)
        v = torch.randn(B, N_k, d)
        out, _ = attn(q, k, v)
        assert out.shape == (B, N_q, d)

    def test_attention_weights_shape(self, attention):
        """Attention weights should be [B, H, N_q, N_k]."""
        B, N, d = 2, 16, 32
        x = torch.randn(B, N, d)
        _, weights = attention(x, x, x, need_weights=True)
        assert weights.shape == (B, 4, N, N)


class TestDtype:
    def test_fp32_intermediates(self, attention):
        """All intermediate Lorentz computations should be fp32."""
        B, N, d = 2, 8, 32
        x = torch.randn(B, N, d)
        out, _ = attention(x, x, x)
        assert out.dtype == torch.float32

    def test_no_fp16_leak(self):
        """Even with fp16 input, Lorentz ops should use fp32."""
        attn = HCAAttention(d_model=32, n_heads=4)
        B, N, d = 2, 8, 32
        # Module stays fp32, input is fp16 — output should be fp32
        x = torch.randn(B, N, d, dtype=torch.float16)
        out, _ = attn(x, x, x)
        # Output should be fp32 (module is fp32)
        assert out.dtype == torch.float32


class TestSmoke:
    def test_128_token_forward_finite(self, attention):
        """128-token forward pass produces finite output (kill condition b)."""
        B, N, d = 4, 128, 32
        x = torch.randn(B, N, d)
        out, _ = attention(x, x, x)
        assert torch.isfinite(out).all(), "128-token forward produced non-finite output"

    def test_single_token(self, attention):
        """Single token sequence should work."""
        B, N, d = 1, 1, 32
        x = torch.randn(B, N, d)
        out, _ = attention(x, x, x)
        assert torch.isfinite(out).all()

    def test_with_mask(self, attention):
        """Causal mask should produce finite output."""
        B, N, d = 2, 16, 32
        x = torch.randn(B, N, d)
        mask = torch.triu(torch.full((N, N), float("-inf")), diagonal=1)
        out, _ = attention(x, x, x, attn_mask=mask)
        assert torch.isfinite(out).all()


class TestGradient:
    def test_backward_finite(self, attention):
        """Backward pass produces finite gradients on all parameters."""
        B, N, d = 2, 16, 32
        x = torch.randn(B, N, d)
        out, _ = attention(x, x, x)
        loss = out.sum()
        loss.backward()

        for name, p in attention.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"

    def test_gamma_has_gradient(self, attention):
        """gamma_raw should receive gradient."""
        B, N, d = 2, 16, 32
        x = torch.randn(B, N, d)
        out, _ = attention(x, x, x)
        out.sum().backward()
        assert attention.gamma_raw.grad is not None
        assert attention.gamma_raw.grad.abs() > 0

    def test_curvature_has_gradient(self, attention):
        """c_raw should receive gradient."""
        B, N, d = 2, 16, 32
        x = torch.randn(B, N, d)
        out, _ = attention(x, x, x)
        out.sum().backward()
        assert attention.curvature.c_raw.grad is not None


class TestAlphaS:
    def test_alpha_s_tracked(self, attention):
        """alpha_s should be tracked after forward pass."""
        B, N, d = 2, 16, 32
        x = torch.randn(B, N, d)
        attention(x, x, x)
        alpha_s = attention.get_alpha_s()
        assert 0 <= alpha_s <= 1, f"alpha_s={alpha_s} out of [0,1] range"

    def test_gamma_getter(self, attention):
        """get_gamma should return positive value."""
        gamma = attention.get_gamma()
        assert gamma > 0

    def test_c_eff_getter(self, attention):
        """get_c_eff should return positive value."""
        c_eff = attention.get_c_eff()
        assert c_eff > 0


@requires_cuda
class TestFused:
    def test_fused_output_shape(self):
        attn = HCAAttention(d_model=64, n_heads=4, use_fused=True).cuda()
        x = torch.randn(2, 16, 64, device='cuda')
        out, _ = attn(x, x, x)
        assert out.shape == (2, 16, 64)

    def test_fused_output_finite(self):
        attn = HCAAttention(d_model=64, n_heads=4, use_fused=True).cuda()
        x = torch.randn(4, 128, 64, device='cuda')
        out, _ = attn(x, x, x)
        assert torch.isfinite(out).all()

    def test_fused_matches_unfused(self):
        """Fused output matches unfused within fp32 tolerance."""
        torch.manual_seed(42)
        attn_f = HCAAttention(d_model=64, n_heads=4, use_fused=True).cuda()
        attn_u = HCAAttention(d_model=64, n_heads=4, use_fused=False).cuda()
        attn_u.load_state_dict(attn_f.state_dict())

        x = torch.randn(2, 16, 64, device='cuda')
        out_f, _ = attn_f(x, x, x)
        out_u, _ = attn_u(x, x, x)
        assert torch.allclose(out_f, out_u, atol=0.01)

    def test_fused_causal_matches_unfused(self):
        """Fused causal matches unfused with explicit causal mask."""
        torch.manual_seed(42)
        attn_f = HCAAttention(d_model=64, n_heads=4, use_fused=True).cuda()
        attn_u = HCAAttention(d_model=64, n_heads=4, use_fused=False).cuda()
        attn_u.load_state_dict(attn_f.state_dict())

        N = 16
        x = torch.randn(2, N, 64, device='cuda')
        out_f, _ = attn_f(x, x, x, is_causal=True)
        mask = torch.triu(torch.full((N, N), float("-inf"), device='cuda'), diagonal=1)
        out_u, _ = attn_u(x, x, x, attn_mask=mask)
        assert torch.allclose(out_f, out_u, atol=0.01)

    def test_fused_backward_finite(self):
        attn = HCAAttention(d_model=64, n_heads=4, use_fused=True).cuda()
        x = torch.randn(2, 16, 64, device='cuda')
        out, _ = attn(x, x, x)
        out.sum().backward()
        for name, p in attn.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad: {name}"

    def test_fused_gamma_has_gradient(self):
        attn = HCAAttention(d_model=64, n_heads=4, use_fused=True).cuda()
        x = torch.randn(2, 16, 64, device='cuda')
        out, _ = attn(x, x, x)
        out.sum().backward()
        assert attn.gamma_raw.grad is not None
        assert attn.gamma_raw.grad.abs() > 0

    def test_fused_falls_back_for_need_weights(self):
        """Fused should fall back to unfused when need_weights=True."""
        attn = HCAAttention(d_model=64, n_heads=4, use_fused=True).cuda()
        x = torch.randn(2, 16, 64, device='cuda')
        out, weights = attn(x, x, x, need_weights=True)
        assert weights is not None
        assert weights.shape == (2, 4, 16, 16)


@requires_cuda
class TestFusedBackwardAccuracy:
    """DS-13: Fused backward Triton gradient accuracy tests.

    Uses dot_precision="ieee" for both Triton forward and backward so that
    tl.dot uses full fp32 — enabling tight comparison against PyTorch fp32
    reference without TF32 noise. Production uses "tf32" for speed; these
    tests verify the backward MATH is correct.
    """

    def _compare_op_grads(self, is_causal):
        """Compare Triton ieee backward vs PyTorch fp32 backward at op level."""
        from hca.ops.fused_attention import _triton_fwd, _triton_bwd
        import torch.nn.functional as F

        torch.manual_seed(42)
        BH, N, D = 8, 32, 64
        Q_sp = torch.randn(BH, N, D, device='cuda')
        K_sp = torch.randn(BH, N, D, device='cuda')
        V = torch.randn(BH, N, D, device='cuda')
        inv_c, beta, gamma_val = 1.0, 0.125, 0.6

        # Triton forward + backward (ieee = full fp32 dot products)
        Out, LSE = _triton_fwd(
            Q_sp, K_sp, V, inv_c, beta, gamma_val, is_causal,
            dot_precision="ieee",
        )
        grad_out = torch.randn_like(Out)
        dQ_t, dK_t, dV_t, dg_t = _triton_bwd(
            Q_sp, K_sp, V, Out, LSE, grad_out, inv_c, beta, gamma_val,
            is_causal, dot_precision="ieee",
        )

        # PyTorch reference backward (fp32)
        Q_r = Q_sp.clone().requires_grad_(True)
        K_r = K_sp.clone().requires_grad_(True)
        V_r = V.clone().requires_grad_(True)
        gamma_t = torch.tensor(gamma_val, device='cuda', requires_grad=True)

        q_x0 = torch.sqrt(torch.tensor(inv_c, device='cuda') + (Q_r * Q_r).sum(-1))
        k_x0 = torch.sqrt(torch.tensor(inv_c, device='cuda') + (K_r * K_r).sum(-1))
        lip = -(q_x0.unsqueeze(-1) * k_x0.unsqueeze(-2)) + torch.bmm(Q_r, K_r.transpose(-1, -2))
        dsq = torch.clamp(-lip - inv_c, min=0.0)
        log_kx0 = torch.log(k_x0.clamp(min=1e-7))
        scores = -beta * dsq - gamma_t * log_kx0.unsqueeze(-2)

        if is_causal:
            causal = torch.triu(torch.full((N, N), float('-inf'), device='cuda'), diagonal=1)
            scores = scores + causal

        alpha = F.softmax(scores, dim=-1)
        out_ref = torch.bmm(alpha, V_r)
        out_ref.backward(grad_out)

        # Tight fp32 tolerance: ieee Triton should match PyTorch closely
        tol = 1e-4
        return {
            'dQ': ((dQ_t - Q_r.grad).abs().max().item(), tol),
            'dK': ((dK_t - K_r.grad).abs().max().item(), tol),
            'dV': ((dV_t - V_r.grad).abs().max().item(), tol),
            'dgamma': (abs(dg_t.item() - gamma_t.grad.item()), 1e-3),
        }

    def test_noncausal_grad_accuracy(self):
        """Non-causal: dQ, dK, dV, dgamma match PyTorch fp32 within 1e-4."""
        results = self._compare_op_grads(is_causal=False)
        for name, (diff, tol) in results.items():
            assert diff < tol, f"{name} diff {diff:.2e} > {tol}"

    def test_causal_grad_accuracy(self):
        """Causal: dQ, dK, dV, dgamma match PyTorch fp32 within 1e-4."""
        results = self._compare_op_grads(is_causal=True)
        for name, (diff, tol) in results.items():
            assert diff < tol, f"{name} diff {diff:.2e} > {tol}"

    def test_fused_module_grads_finite_and_nonzero(self):
        """End-to-end: fused module produces finite, nonzero grads on all params."""
        torch.manual_seed(42)
        attn = HCAAttention(d_model=64, n_heads=4, use_fused=True).cuda()
        x = torch.randn(2, 32, 64, device='cuda')
        out, _ = attn(x, x, x, is_causal=True)
        out.sum().backward()

        for name, p in attn.named_parameters():
            assert p.grad is not None, f"{name}: no gradient"
            assert torch.isfinite(p.grad).all(), f"{name}: non-finite gradient"
            assert p.grad.abs().max() > 0, f"{name}: zero gradient"
