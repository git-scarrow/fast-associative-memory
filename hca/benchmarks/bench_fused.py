"""DS-13: Benchmark fused HCA attention (fwd+bwd) vs unfused vs standard MHA.

Measures forward time, forward+backward time, peak memory.
Kill condition: fused fwd+bwd > 2.0x of standard MHA at N=2048.
Stop condition: fused fwd+bwd within 2.0x of MHA at N=2048.

DS-12 baseline (fwd-only fused, PyTorch bwd): 3.93x at N=2048.
DS-13 target: fused backward Triton kernels close the gap.
"""

import json
import time
import torch
import torch.nn as nn
from hca.modules.hca_attention import HCAAttention


def _warmup_cuda():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def _bench_forward(module, x, n_iters=50, is_causal=False, attn_mask=None):
    """Time forward pass (ms)."""
    # Warmup
    for _ in range(5):
        module(x, x, x, is_causal=is_causal, attn_mask=attn_mask)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        module(x, x, x, is_causal=is_causal, attn_mask=attn_mask)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iters


def _bench_fwd_bwd(module, x, n_iters=50, is_causal=False, attn_mask=None):
    """Time forward + backward pass (ms)."""
    for _ in range(5):
        out, _ = module(x, x, x, is_causal=is_causal, attn_mask=attn_mask)
        out.sum().backward()
        module.zero_grad()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        out, _ = module(x, x, x, is_causal=is_causal, attn_mask=attn_mask)
        out.sum().backward()
        module.zero_grad()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iters


def _peak_memory_mb(module, x, is_causal=False, attn_mask=None):
    """Peak GPU memory (MB) during forward + backward."""
    module.zero_grad()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    out, _ = module(x, x, x, is_causal=is_causal, attn_mask=attn_mask)
    out.sum().backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 / 1024


class StandardMHAWrapper(nn.Module):
    """Wrap nn.MultiheadAttention to match HCAAttention's call signature."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, q, k, v, need_weights=False, attn_mask=None, is_causal=False):
        if is_causal and attn_mask is None:
            N = q.shape[1]
            attn_mask = torch.triu(
                torch.full((N, N), float("-inf"), device=q.device), diagonal=1
            )
        out, w = self.mha(q, k, v, need_weights=need_weights, attn_mask=attn_mask)
        return out, w

    def zero_grad(self):
        self.mha.zero_grad()


def run_benchmark(d_model=512, n_heads=8, batch_size=8, seq_lengths=None, device='cuda'):
    if seq_lengths is None:
        seq_lengths = [128, 512, 1024, 2048]

    results = {}

    for N in seq_lengths:
        print(f"\n--- N={N}, d_model={d_model}, n_heads={n_heads}, B={batch_size} ---")
        x = torch.randn(batch_size, N, d_model, device=device)

        configs = {
            'unfused_fp32': HCAAttention(d_model, n_heads, use_fused=False).to(device),
            'fused_fp32': HCAAttention(d_model, n_heads, use_fused=True).to(device),
            'fused_bf16': HCAAttention(d_model, n_heads, use_fused=True, amp_enabled=True).to(device),
            'standard_mha': StandardMHAWrapper(d_model, n_heads).to(device),
        }

        row = {}
        for name, module in configs.items():
            _warmup_cuda()
            is_causal = name != 'standard_mha'  # standard MHA uses attn_mask

            fwd_ms = _bench_forward(module, x, is_causal=is_causal)
            fwd_bwd_ms = _bench_fwd_bwd(module, x, is_causal=is_causal)
            mem_mb = _peak_memory_mb(module, x, is_causal=is_causal)

            row[name] = {
                'fwd_ms': round(fwd_ms, 3),
                'fwd_bwd_ms': round(fwd_bwd_ms, 3),
                'mem_mb': round(mem_mb, 1),
                'tokens_per_sec': round(batch_size * N / (fwd_bwd_ms / 1000)),
            }
            print(f"  {name:15s}: fwd={fwd_ms:.3f}ms  fwd+bwd={fwd_bwd_ms:.3f}ms  mem={mem_mb:.1f}MB")

        # Compute ratios vs standard MHA
        mha_fwd = row['standard_mha']['fwd_ms']
        mha_fwd_bwd = row['standard_mha']['fwd_bwd_ms']
        for name in ['unfused_fp32', 'fused_fp32', 'fused_bf16']:
            fwd_ratio = row[name]['fwd_ms'] / mha_fwd
            fwdbwd_ratio = row[name]['fwd_bwd_ms'] / mha_fwd_bwd
            row[name]['fwd_ratio_vs_mha'] = round(fwd_ratio, 2)
            row[name]['ratio_vs_mha'] = round(fwdbwd_ratio, 2)
            print(f"  {name:15s}: fwd={fwd_ratio:.2f}x  fwd+bwd={fwdbwd_ratio:.2f}x vs MHA")

        results[N] = row

    return results


if __name__ == '__main__':
    assert torch.cuda.is_available(), "CUDA required"

    print("=" * 70)
    print("DS-13 Benchmark: Fused HCA Attention (fwd+bwd Triton)")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Triton: available")
    print("=" * 70)

    results = run_benchmark()

    # Check kill/stop conditions at N=2048
    if 2048 in results:
        r = results[2048]
        fwd_ratio = r['fused_bf16']['fwd_ratio_vs_mha']
        fwdbwd_ratio = r['fused_bf16']['ratio_vs_mha']
        unfused_ratio = r['unfused_fp32']['ratio_vs_mha']
        fused_fp32_ratio = r['fused_fp32']['ratio_vs_mha']

        print(f"\n--- DS-13 RESULTS SUMMARY (N=2048) ---")
        print(f"Forward only:    fused bf16 = {fwd_ratio:.2f}x vs MHA")
        print(f"Fwd+Bwd:         fused bf16 = {fwdbwd_ratio:.2f}x vs MHA")
        print(f"Fwd+Bwd:         fused fp32 = {fused_fp32_ratio:.2f}x vs MHA")
        print(f"Fwd+Bwd:         unfused    = {unfused_ratio:.2f}x vs MHA (DS-12 baseline)")
        print()

        if fwdbwd_ratio > 2.0:
            print(f"KILL CONDITION: fwd+bwd {fwdbwd_ratio:.2f}x > 2.0x MHA")
        else:
            print(f"STOP CONDITION MET: fwd+bwd {fwdbwd_ratio:.2f}x <= 2.0x MHA")
            speedup = unfused_ratio / fwdbwd_ratio
            print(f"DS-13 speedup vs DS-12 unfused: {speedup:.1f}x")

    # Save results
    out_path = 'results/ds13_bench.json'
    import os
    os.makedirs('results', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
