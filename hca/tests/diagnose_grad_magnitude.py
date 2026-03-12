"""Diagnostic: compare fused vs unfused gradient magnitudes at Q_sp/K_sp level.

This isolates whether the magnitude difference is in the fused backward kernel
or in how gradients propagate through the rest of the autograd chain.
"""

import math
import torch
import torch.nn.functional as F

from hca.ops.lorentz_exp_map import lorentz_exp_map
from hca.ops.lorentz_distance import lorentz_squared_distance
from hca.ops.fused_attention import fused_hca_attention, _triton_fwd, _triton_bwd

torch.manual_seed(42)
device = "cuda"

# Benchmark dimensions
B, H, N, d_head = 8, 8, 1024, 64
BH = B * H
d_model = d_head * H
inv_c_val = 1.0 / 1.125
beta = 1.0 / math.sqrt(d_head)
gamma_val = 0.6

# Create realistic Q_sp, K_sp from exp_map
tangent_Q = torch.randn(B, H, N, d_head, device=device)
tangent_K = torch.randn(B, H, N, d_head, device=device)
V = torch.randn(B, H, N, d_head, device=device)

c = 1.125
Q_sp, Q_x0 = lorentz_exp_map(tangent_Q, c)
K_sp, K_x0 = lorentz_exp_map(tangent_K, c)

print(f"Q_sp range: [{Q_sp.min():.2f}, {Q_sp.max():.2f}], x0 range: [{Q_x0.min():.2f}, {Q_x0.max():.2f}]")
print(f"K_sp range: [{K_sp.min():.2f}, {K_sp.max():.2f}], x0 range: [{K_x0.min():.2f}, {K_x0.max():.2f}]")

# ---- Op-level comparison: fused vs unfused backward at same Q_sp, K_sp ----
print("\n=== Op-level gradient comparison (ieee precision) ===")

# Unfused reference (PyTorch autograd)
Q_u = Q_sp.detach().clone().reshape(BH, N, d_head).requires_grad_(True)
K_u = K_sp.detach().clone().reshape(BH, N, d_head).requires_grad_(True)
V_u = V.detach().clone().reshape(BH, N, d_head).float().requires_grad_(True)
gamma_u = torch.tensor(gamma_val, device=device, requires_grad=True)

q_x0_u = torch.sqrt(torch.tensor(inv_c_val, device=device) + (Q_u * Q_u).sum(-1))
k_x0_u = torch.sqrt(torch.tensor(inv_c_val, device=device) + (K_u * K_u).sum(-1))
lip_u = -(q_x0_u.unsqueeze(-1) * k_x0_u.unsqueeze(-2)) + torch.bmm(Q_u, K_u.transpose(-1, -2))
dsq_u = torch.clamp(-lip_u - inv_c_val, min=0.0)
log_kx0_u = torch.log(k_x0_u.clamp(min=1e-7))
scores_u = -beta * dsq_u - gamma_u * log_kx0_u.unsqueeze(-2)

causal = torch.triu(torch.full((N, N), float('-inf'), device=device), diagonal=1)
scores_u = scores_u + causal

alpha_u = F.softmax(scores_u, dim=-1)
out_u = torch.bmm(alpha_u, V_u)

# Same grad_out for both
grad_out = torch.randn_like(out_u)
out_u.backward(grad_out)

print(f"Unfused dQ norm: {Q_u.grad.norm():.4f}")
print(f"Unfused dK norm: {K_u.grad.norm():.4f}")
print(f"Unfused dV norm: {V_u.grad.norm():.4f}")
print(f"Unfused dQ range: [{Q_u.grad.min():.6f}, {Q_u.grad.max():.6f}]")
print(f"Unfused dK range: [{K_u.grad.min():.6f}, {K_u.grad.max():.6f}]")

# Fused (ieee)
Q_f = Q_sp.detach().clone().reshape(BH, N, d_head)
K_f = K_sp.detach().clone().reshape(BH, N, d_head)
V_f = V.detach().clone().reshape(BH, N, d_head).float()

Out_f, LSE_f = _triton_fwd(Q_f, K_f, V_f, inv_c_val, beta, gamma_val, True, dot_precision="ieee")
dQ_f, dK_f, dV_f, dg_f = _triton_bwd(
    Q_f, K_f, V_f, Out_f, LSE_f, grad_out, inv_c_val, beta, gamma_val, True, dot_precision="ieee"
)

print(f"\nFused dQ norm:   {dQ_f.norm():.4f}")
print(f"Fused dK norm:   {dK_f.norm():.4f}")
print(f"Fused dV norm:   {dV_f.norm():.4f}")
print(f"Fused dQ range:  [{dQ_f.min():.6f}, {dQ_f.max():.6f}]")
print(f"Fused dK range:  [{dK_f.min():.6f}, {dK_f.max():.6f}]")

# Ratios
dq_ratio = dQ_f.norm() / Q_u.grad.norm()
dk_ratio = dK_f.norm() / K_u.grad.norm()
dv_ratio = dV_f.norm() / V_u.grad.norm()
print(f"\nRatio fused/unfused: dQ={dq_ratio:.4f}x, dK={dk_ratio:.4f}x, dV={dv_ratio:.4f}x")

# Max absolute difference
dq_diff = (dQ_f - Q_u.grad).abs().max().item()
dk_diff = (dK_f - K_u.grad).abs().max().item()
dv_diff = (dV_f - V_u.grad).abs().max().item()
print(f"Max abs diff: dQ={dq_diff:.6f}, dK={dk_diff:.6f}, dV={dv_diff:.6f}")

# ---- Now compare through the full attention module ----
print("\n=== Full module gradient comparison ===")
from hca.modules.hca_attention import HCAAttention

torch.manual_seed(42)
attn_shared = HCAAttention(d_model=d_model, n_heads=H, use_fused=False).cuda()
state = attn_shared.state_dict()

# Unfused path
attn_u2 = HCAAttention(d_model=d_model, n_heads=H, use_fused=False).cuda()
attn_u2.load_state_dict(state)
x_u2 = torch.randn(B, N, d_model, device=device, requires_grad=True)
mask = torch.triu(torch.full((N, N), float("-inf"), device=device), diagonal=1)
out_u2, _ = attn_u2(x_u2, x_u2, x_u2, attn_mask=mask)
out_u2.sum().backward()

# Fused path
attn_f2 = HCAAttention(d_model=d_model, n_heads=H, use_fused=True).cuda()
attn_f2.load_state_dict(state)
x_f2 = x_u2.data.clone().requires_grad_(True)  # same input
out_f2, _ = attn_f2(x_f2, x_f2, x_f2, is_causal=True)
out_f2.sum().backward()

print(f"Output diff: {(out_f2 - out_u2).abs().max():.6f}")

for (name_u, p_u), (name_f, p_f) in zip(attn_u2.named_parameters(), attn_f2.named_parameters()):
    if p_u.grad is not None and p_f.grad is not None:
        ratio = p_f.grad.norm() / p_u.grad.norm() if p_u.grad.norm() > 0 else float('inf')
        diff = (p_f.grad - p_u.grad).abs().max()
        print(f"  {name_u:30s}: unfused_norm={p_u.grad.norm():.2f}, fused_norm={p_f.grad.norm():.2f}, "
              f"ratio={ratio:.4f}, max_diff={diff:.4f}")

x_grad_ratio = x_f2.grad.norm() / x_u2.grad.norm() if x_u2.grad.norm() > 0 else float('inf')
print(f"  {'input':30s}: unfused_norm={x_u2.grad.norm():.2f}, fused_norm={x_f2.grad.norm():.2f}, "
      f"ratio={x_grad_ratio:.4f}")
