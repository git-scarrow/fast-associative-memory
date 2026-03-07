"""bf16 mixed precision tests for HCA attention (DS-12 D4)."""

import pytest
import torch

from hca.modules.hca_attention import HCAAttention


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@requires_cuda
class TestBf16Forward:
    def test_bf16_input_produces_finite_output(self):
        attn = HCAAttention(d_model=64, n_heads=4, amp_enabled=True).cuda()
        x = torch.randn(2, 16, 64, device='cuda')
        out, _ = attn(x, x, x)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

    def test_bf16_fused_produces_finite_output(self):
        attn = HCAAttention(d_model=64, n_heads=4, use_fused=True, amp_enabled=True).cuda()
        x = torch.randn(2, 16, 64, device='cuda')
        out, _ = attn(x, x, x, is_causal=True)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

    def test_bf16_matches_fp32_within_tolerance(self):
        """bf16 output should be within bf16 precision (~1e-2) of fp32."""
        torch.manual_seed(123)
        attn_bf16 = HCAAttention(d_model=64, n_heads=4, amp_enabled=True).cuda()
        attn_fp32 = HCAAttention(d_model=64, n_heads=4, amp_enabled=False).cuda()
        attn_fp32.load_state_dict(attn_bf16.state_dict())

        x = torch.randn(2, 16, 64, device='cuda')
        out_bf16, _ = attn_bf16(x, x, x)
        out_fp32, _ = attn_fp32(x, x, x)
        assert torch.allclose(out_bf16, out_fp32, atol=0.05, rtol=0.05)


@requires_cuda
class TestBf16Backward:
    def test_bf16_backward_finite(self):
        attn = HCAAttention(d_model=64, n_heads=4, amp_enabled=True).cuda()
        x = torch.randn(2, 16, 64, device='cuda')
        out, _ = attn(x, x, x)
        out.sum().backward()
        for name, p in attn.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"

    def test_bf16_fused_backward_finite(self):
        attn = HCAAttention(d_model=64, n_heads=4, use_fused=True, amp_enabled=True).cuda()
        x = torch.randn(2, 16, 64, device='cuda')
        out, _ = attn(x, x, x, is_causal=True)
        out.sum().backward()
        for name, p in attn.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"


@requires_cuda
class TestBf16NaN:
    def test_no_nan_in_100_random_inputs(self):
        """Kill condition: NaN in >1% of steps. Check 100 random inputs."""
        attn = HCAAttention(d_model=64, n_heads=4, use_fused=True, amp_enabled=True).cuda()
        nan_count = 0
        for _ in range(100):
            x = torch.randn(2, 16, 64, device='cuda')
            out, _ = attn(x, x, x)
            if not torch.isfinite(out).all():
                nan_count += 1
        assert nan_count <= 1, f"NaN in {nan_count}/100 steps (>1% threshold)"
