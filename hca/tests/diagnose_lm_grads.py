"""Diagnostic: compare per-param gradient norms in full LM, single step."""

import torch
import torch.nn.functional as F
import logging
logging.basicConfig(level=logging.WARNING)

from hca.training.harness import HCATransformerLM
from hca.training.optimizer import build_optimizer

device = "cuda"
d_model, n_heads, n_layers = 512, 8, 4
B, N = 8, 1024

# Fixed data
torch.manual_seed(123)
data = torch.randint(0, 10000, (100, N + 1), device=device)

for use_fused in [False, True]:
    label = "FUSED" if use_fused else "UNFUSED"

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = HCATransformerLM(
        vocab_size=10000, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        max_seq_len=N, use_fused=use_fused, amp_enabled=False,
    ).cuda()

    # Same batch (reset RNG state to same point after data gen)
    torch.manual_seed(999)
    idx = torch.randint(0, len(data), (B,))
    batch = data[idx]
    input_ids = batch[:, :-1]
    targets = batch[:, 1:]

    _, loss = model(input_ids, targets)
    loss.backward()

    print(f"\n=== {label} ===")
    print(f"Loss: {loss.item():.6f}")

    total_norm_sq = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            pnorm = p.grad.norm().item()
            total_norm_sq += pnorm ** 2
            nan = torch.isnan(p.grad).any().item()
            inf = torch.isinf(p.grad).any().item()
            if pnorm > 1000 or nan or inf:
                print(f"  {name:50s}: norm={pnorm:>15.2f} nan={nan} inf={inf} "
                      f"range=[{p.grad.min():.2f}, {p.grad.max():.2f}]")

    total_norm = total_norm_sq ** 0.5
    print(f"  TOTAL grad norm: {total_norm:.2f}")
