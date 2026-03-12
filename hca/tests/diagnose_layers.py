"""Diagnostic: vary number of layers to find where gradient explosion begins."""

import torch
import logging
logging.basicConfig(level=logging.WARNING)

from hca.training.harness import HCATransformerLM

device = "cuda"
d_model, n_heads = 512, 8
B, N = 8, 1024

torch.manual_seed(123)
data = torch.randint(0, 10000, (100, N + 1), device=device)
torch.manual_seed(999)
idx = torch.randint(0, len(data), (B,))
batch = data[idx]
input_ids = batch[:, :-1]
targets = batch[:, 1:]

for n_layers in [1, 2, 4]:
    for use_fused in [False, True]:
        label = f"{'FUSED' if use_fused else 'UNFUSED':>8s} {n_layers}L"

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model = HCATransformerLM(
            vocab_size=10000, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            max_seq_len=N, use_fused=use_fused, amp_enabled=False,
        ).cuda()

        _, loss = model(input_ids, targets)
        loss.backward()

        total_norm = 0.0
        max_param_name = None
        max_param_norm = 0.0
        for name, p in model.named_parameters():
            if p.grad is not None:
                pn = p.grad.norm().item()
                total_norm += pn ** 2
                if pn > max_param_norm:
                    max_param_norm = pn
                    max_param_name = name

        total_norm = total_norm ** 0.5
        print(f"  {label}: loss={loss.item():.4f}  grad_norm={total_norm:>15.1f}  "
              f"worst={max_param_name}({max_param_norm:.1f})")
