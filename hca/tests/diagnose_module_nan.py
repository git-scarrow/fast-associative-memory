"""Diagnostic: trace NaN through full HCAAttention module at benchmark scale.

Tests the complete forward+backward including exp_map, to find where NaN
is introduced in the gradient chain.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from hca.modules.hca_attention import HCAAttention

torch.manual_seed(42)
device = "cuda"

# Benchmark config
d_model, n_heads = 512, 8
B, N = 8, 1024

print(f"Config: d_model={d_model}, n_heads={n_heads}, B={B}, N={N}")
print(f"d_head={d_model // n_heads}")

# ---- Test 1: Fused module backward at benchmark scale ----
print("\n=== Test 1: Full module backward (fused, causal) ===")
attn_f = HCAAttention(d_model=d_model, n_heads=n_heads, use_fused=True).cuda()
x = torch.randn(B, N, d_model, device=device, requires_grad=True)

out_f, _ = attn_f(x, x, x, is_causal=True)
print(f"  Forward out: finite={torch.isfinite(out_f).all()}, "
      f"range=[{out_f.min():.4f}, {out_f.max():.4f}]")

loss = out_f.sum()
loss.backward()

for name, p in attn_f.named_parameters():
    if p.grad is not None:
        nan_count = torch.isnan(p.grad).sum().item()
        inf_count = torch.isinf(p.grad).sum().item()
        total = p.grad.numel()
        finite_vals = p.grad[torch.isfinite(p.grad)]
        if len(finite_vals) > 0:
            print(f"  {name:30s}: NaN={nan_count}/{total}, Inf={inf_count}, "
                  f"range=[{finite_vals.min():.4f}, {finite_vals.max():.4f}]")
        else:
            print(f"  {name:30s}: ALL NaN/Inf ({nan_count + inf_count}/{total})")
    else:
        print(f"  {name:30s}: no grad")

x_grad_nan = torch.isnan(x.grad).sum().item() if x.grad is not None else "no grad"
print(f"  Input grad NaN: {x_grad_nan}")

# ---- Test 2: Same but unfused ----
print("\n=== Test 2: Full module backward (unfused, causal mask) ===")
attn_u = HCAAttention(d_model=d_model, n_heads=n_heads, use_fused=False).cuda()
attn_u.load_state_dict(attn_f.state_dict())

x2 = torch.randn(B, N, d_model, device=device, requires_grad=True)
causal_mask = torch.triu(torch.full((N, N), float("-inf"), device=device), diagonal=1)
out_u, _ = attn_u(x2, x2, x2, attn_mask=causal_mask)
print(f"  Forward out: finite={torch.isfinite(out_u).all()}, "
      f"range=[{out_u.min():.4f}, {out_u.max():.4f}]")

loss_u = out_u.sum()
loss_u.backward()

for name, p in attn_u.named_parameters():
    if p.grad is not None:
        nan_count = torch.isnan(p.grad).sum().item()
        inf_count = torch.isinf(p.grad).sum().item()
        total = p.grad.numel()
        finite_vals = p.grad[torch.isfinite(p.grad)]
        if len(finite_vals) > 0:
            print(f"  {name:30s}: NaN={nan_count}/{total}, Inf={inf_count}, "
                  f"range=[{finite_vals.min():.4f}, {finite_vals.max():.4f}]")
        else:
            print(f"  {name:30s}: ALL NaN/Inf ({nan_count + inf_count}/{total})")

x2_grad_nan = torch.isnan(x2.grad).sum().item() if x2.grad is not None else "no grad"
print(f"  Input grad NaN: {x2_grad_nan}")

# ---- Test 3: Isolate which gradient (dQ, dK, or dV) from fused attn is NaN ----
print("\n=== Test 3: Isolate dQ/dK/dV from fused attention ===")
from hca.ops.lorentz_exp_map import lorentz_exp_map
from hca.ops.fused_attention import fused_hca_attention, HAS_TRITON
from hca.modules.curvature import CurvatureModule

torch.manual_seed(42)
attn_test = HCAAttention(d_model=d_model, n_heads=n_heads, use_fused=True).cuda()
x3 = torch.randn(B, N, d_model, device=device)

d_head = d_model // n_heads

# Manually run the forward to capture intermediates
with torch.no_grad():
    Q = attn_test.q_proj(x3.float())
    K = attn_test.k_proj(x3.float())
    V = attn_test.v_proj(x3.float())

Q = Q.view(B, N, n_heads, d_head).transpose(1, 2).float()
K = K.view(B, N, n_heads, d_head).transpose(1, 2).float()
V = V.view(B, N, n_heads, d_head).transpose(1, 2).float()

c = attn_test.curvature(N)
gamma = F.softplus(attn_test.gamma_raw)
inv_c = 1.0 / c.item()
beta = attn_test.beta

print(f"  c={c.item():.6f}, gamma={gamma.item():.6f}, beta={beta:.6f}")

# Run exp_map with grad tracking
Q_f = Q.clone().requires_grad_(True)
K_f = K.clone().requires_grad_(True)
Q_sp, Q_x0 = lorentz_exp_map(Q_f, c)
K_sp, K_x0 = lorentz_exp_map(K_f, c)

print(f"  Q_sp: finite={torch.isfinite(Q_sp).all()}, range=[{Q_sp.min():.4f}, {Q_sp.max():.4f}]")
print(f"  K_sp: finite={torch.isfinite(K_sp).all()}, range=[{K_sp.min():.4f}, {K_sp.max():.4f}]")
print(f"  Q_x0: finite={torch.isfinite(Q_x0).all()}, range=[{Q_x0.min():.4f}, {Q_x0.max():.4f}]")
print(f"  K_x0: finite={torch.isfinite(K_x0).all()}, range=[{K_x0.min():.4f}, {K_x0.max():.4f}]")

# Reshape for fused attention
BH = B * n_heads
Q_sp_r = Q_sp.reshape(BH, N, d_head).detach().requires_grad_(True)
K_sp_r = K_sp.reshape(BH, N, d_head).detach().requires_grad_(True)
V_r = V.float().reshape(BH, N, d_head).detach().requires_grad_(True)

gamma_t = gamma.detach().requires_grad_(True)

out = fused_hca_attention(Q_sp_r, K_sp_r, V_r, inv_c, beta, gamma_t, is_causal=True)
print(f"  Fused out: finite={torch.isfinite(out).all()}, range=[{out.min():.4f}, {out.max():.4f}]")

out.sum().backward()
print(f"  dQ_sp: NaN={torch.isnan(Q_sp_r.grad).sum().item()}/{Q_sp_r.grad.numel()}")
print(f"  dK_sp: NaN={torch.isnan(K_sp_r.grad).sum().item()}/{K_sp_r.grad.numel()}")
print(f"  dV:    NaN={torch.isnan(V_r.grad).sum().item()}/{V_r.grad.numel()}")
print(f"  dgamma: {gamma_t.grad}")

# Check if NaN flows through exp_map backward
if torch.isnan(Q_sp_r.grad).any() or torch.isnan(K_sp_r.grad).any():
    print("\n  NaN in fused grads — checking if it originates in fused kernel vs exp_map")
else:
    print("\n  Fused kernel grads are CLEAN at module level")
    print("  NaN must originate elsewhere in the training pipeline")

# ---- Test 4: Full training step with gradient check ----
print("\n=== Test 4: Single training step (matching bench_e2e) ===")
from hca.training.harness import HCATransformerLM

torch.manual_seed(42)
model = HCATransformerLM(
    vocab_size=10000, d_model=d_model, n_heads=n_heads, n_layers=4,
    max_seq_len=N, use_fused=True, amp_enabled=False,
).cuda()

input_ids = torch.randint(0, 10000, (B, N), device=device)
targets = torch.randint(0, 10000, (B, N), device=device)

logits, loss = model(input_ids, targets)
print(f"  Loss: {loss.item():.4f}, finite={torch.isfinite(loss)}")
print(f"  Logits: finite={torch.isfinite(logits).all()}, range=[{logits.min():.4f}, {logits.max():.4f}]")

# Check radial reg loss
rad_loss = model.get_radial_reg_loss()
print(f"  Radial reg loss: {rad_loss.item():.6f}, requires_grad={rad_loss.requires_grad}")
total_loss = loss + rad_loss if rad_loss.requires_grad else loss

total_loss.backward()

# Check which parameters have NaN gradients
nan_params = []
for name, p in model.named_parameters():
    if p.grad is not None and not torch.isfinite(p.grad).all():
        nan_count = (~torch.isfinite(p.grad)).sum().item()
        nan_params.append((name, nan_count, p.grad.numel()))

if nan_params:
    print(f"\n  Parameters with NaN/Inf gradients ({len(nan_params)}):")
    for name, nan_count, total in sorted(nan_params, key=lambda x: -x[1]):
        print(f"    {name:50s}: {nan_count}/{total} ({100*nan_count/total:.1f}%)")
else:
    print("\n  ALL GRADIENTS FINITE!")

grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
print(f"  Grad norm: {grad_norm.item() if torch.isfinite(grad_norm) else 'NaN/Inf'}")
