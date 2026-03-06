"""HCAAttention smoke tests, shape tests, and gradient tests."""

import pytest
import torch
import torch.nn as nn

from hca.modules.hca_attention import HCAAttention


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
