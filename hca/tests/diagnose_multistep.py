"""Diagnostic: run 10 training steps fused vs unfused, track grad norm and NaN."""

import torch
import torch.nn.functional as F
import math
import logging

logging.basicConfig(level=logging.WARNING)

from hca.training.harness import HCATransformerLM
from hca.training.optimizer import build_optimizer

device = "cuda"
d_model, n_heads, n_layers = 512, 8, 4
B, N = 8, 1024
lr = 3e-4
STEPS = 10

def run_steps(label, use_fused, steps=STEPS):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = HCATransformerLM(
        vocab_size=10000, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        max_seq_len=N, use_fused=use_fused, amp_enabled=False,
    ).cuda()

    optimizer = build_optimizer(model, lr=lr)

    # Fixed random data (same for both paths)
    torch.manual_seed(123)
    data = torch.randint(0, 10000, (100, N + 1), device=device)

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    nan_steps = []
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(data), (B,))
        batch = data[idx]
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        _, loss = model(input_ids, targets)

        if not torch.isfinite(loss):
            print(f"  Step {step}: LOSS IS {loss.item()}")
            break

        loss.backward()

        # Check individual param grad stats
        worst_name = None
        worst_nan_pct = 0
        total_nan = 0
        total_params = 0
        for name, p in model.named_parameters():
            if p.grad is not None:
                n = (~torch.isfinite(p.grad)).sum().item()
                total_nan += n
                total_params += p.grad.numel()
                pct = n / p.grad.numel()
                if pct > worst_nan_pct:
                    worst_nan_pct = pct
                    worst_name = name

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        is_nan = not torch.isfinite(grad_norm)

        status = "NaN" if is_nan else "OK "
        nan_info = f"worst={worst_name}({100*worst_nan_pct:.0f}%)" if total_nan > 0 else ""
        print(f"  Step {step}: loss={loss.item():.4f} grad_norm={'NaN' if is_nan else f'{grad_norm.item():.1f}'}"
              f" nan_params={total_nan}/{total_params} {nan_info} [{status}]")

        if is_nan:
            nan_steps.append(step)
            optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()

    print(f"  NaN steps: {nan_steps}")
    return nan_steps

# Run both
nan_fused = run_steps("FUSED (is_causal=True)", use_fused=True)
nan_unfused = run_steps("UNFUSED (explicit causal mask)", use_fused=False)

print(f"\nSummary: Fused NaN={len(nan_fused)}/{STEPS}, Unfused NaN={len(nan_unfused)}/{STEPS}")
