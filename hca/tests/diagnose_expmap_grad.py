"""Diagnostic: compare gradient through exp_map between fused and unfused.

Hypothesis: the fused backward routes all gradient through K_sp, while the
unfused backward routes decay gradient through K_x0. These should be
mathematically equivalent, but numerically the exp_map backward amplifies
differently.
"""

import math
import torch
import torch.nn.functional as F

from hca.ops.lorentz_exp_map import lorentz_exp_map
from hca.ops.lorentz_distance import lorentz_squared_distance
from hca.ops.fused_attention import fused_hca_attention

torch.manual_seed(42)
device = "cuda"

B, H, N, d_head = 2, 2, 64, 64  # small for debugging
d_model = d_head * H
BH = B * H
c_val = 1.125
inv_c = 1.0 / c_val
beta = 1.0 / math.sqrt(d_head)

# Create tangent vectors (the actual input to exp_map)
K_tangent = torch.randn(B, H, N, d_head, device=device, requires_grad=True)
Q_tangent = torch.randn(B, H, N, d_head, device=device, requires_grad=True)
V = torch.randn(B, H, N, d_head, device=device)

gamma = torch.tensor(0.6, device=device, requires_grad=True)

# === UNFUSED PATH ===
Q_sp_u, Q_x0_u = lorentz_exp_map(Q_tangent, c_val)
K_sp_u, K_x0_u = lorentz_exp_map(K_tangent, c_val)

dist_sq = lorentz_squared_distance(Q_sp_u, K_sp_u, c_val)
decay_bias = gamma * torch.log(K_x0_u.clamp(min=1e-7))
scores = -beta * dist_sq - decay_bias.unsqueeze(-2)

causal = torch.triu(torch.full((N, N), float("-inf"), device=device), diagonal=1)
scores = scores + causal

alpha = F.softmax(scores, dim=-1)
out_u = torch.matmul(alpha, V.float())
grad_out = torch.randn_like(out_u)
out_u.backward(grad_out)

dK_unfused = K_tangent.grad.clone()
dQ_unfused = Q_tangent.grad.clone()

print("=== UNFUSED ===")
print(f"  Q_sp range: [{Q_sp_u.min():.1f}, {Q_sp_u.max():.1f}], x0: [{Q_x0_u.min():.1f}, {Q_x0_u.max():.1f}]")
print(f"  dK_tangent norm: {dK_unfused.norm():.6f}")
print(f"  dQ_tangent norm: {dQ_unfused.norm():.6f}")
print(f"  dK_tangent range: [{dK_unfused.min():.6f}, {dK_unfused.max():.6f}]")

# === FUSED PATH ===
K_tangent2 = K_tangent.data.clone().requires_grad_(True)
Q_tangent2 = Q_tangent.data.clone().requires_grad_(True)
gamma2 = torch.tensor(0.6, device=device, requires_grad=True)

Q_sp_f, Q_x0_f = lorentz_exp_map(Q_tangent2, c_val)
K_sp_f, K_x0_f = lorentz_exp_map(K_tangent2, c_val)

Q_sp_r = Q_sp_f.reshape(BH, N, d_head)
K_sp_r = K_sp_f.reshape(BH, N, d_head)
V_r = V.float().reshape(BH, N, d_head)

out_f = fused_hca_attention(Q_sp_r, K_sp_r, V_r, inv_c, beta, gamma2, is_causal=True)
out_f = out_f.view(B, H, N, d_head)

out_f.backward(grad_out)

dK_fused = K_tangent2.grad.clone()
dQ_fused = Q_tangent2.grad.clone()

print("\n=== FUSED ===")
print(f"  dK_tangent norm: {dK_fused.norm():.6f}")
print(f"  dQ_tangent norm: {dQ_fused.norm():.6f}")
print(f"  dK_tangent range: [{dK_fused.min():.6f}, {dK_fused.max():.6f}]")

print("\n=== COMPARISON ===")
print(f"  dK norm ratio (fused/unfused): {dK_fused.norm() / dK_unfused.norm():.6f}")
print(f"  dQ norm ratio (fused/unfused): {dQ_fused.norm() / dQ_unfused.norm():.6f}")
print(f"  dK max abs diff: {(dK_fused - dK_unfused).abs().max():.6f}")
print(f"  dQ max abs diff: {(dQ_fused - dQ_unfused).abs().max():.6f}")

# Check if K_x0 got gradient in unfused path
print(f"\n  K_x0 is leaf: {K_x0_u.is_leaf}")
print(f"  K_x0 grad_fn: {K_x0_u.grad_fn}")

# === Check: what if we manually zero the K_x0 gradient path? ===
# Detach K_x0 so decay gradient doesn't flow through it
print("\n=== UNFUSED with K_x0 DETACHED (no decay gradient through K_x0) ===")
K_tangent3 = K_tangent.data.clone().requires_grad_(True)
Q_tangent3 = Q_tangent.data.clone().requires_grad_(True)
gamma3 = torch.tensor(0.6, device=device, requires_grad=True)

Q_sp_3, Q_x0_3 = lorentz_exp_map(Q_tangent3, c_val)
K_sp_3, K_x0_3 = lorentz_exp_map(K_tangent3, c_val)

dist_sq_3 = lorentz_squared_distance(Q_sp_3, K_sp_3, c_val)
# DETACH K_x0 so decay gradient goes nowhere through exp_map
decay_bias_3 = gamma3 * torch.log(K_x0_3.detach().clamp(min=1e-7))
scores_3 = -beta * dist_sq_3 - decay_bias_3.unsqueeze(-2)
scores_3 = scores_3 + causal
alpha_3 = F.softmax(scores_3, dim=-1)
out_3 = torch.matmul(alpha_3, V.float())
out_3.backward(grad_out)

dK_detached = K_tangent3.grad.clone()
print(f"  dK_tangent norm: {dK_detached.norm():.6f}")
print(f"  dK norm ratio (detached/unfused): {dK_detached.norm() / dK_unfused.norm():.6f}")
print(f"  dK norm ratio (detached/fused): {dK_detached.norm() / dK_fused.norm():.6f}")
print(f"  dK max abs diff (detached vs fused): {(dK_detached - dK_fused).abs().max():.6f}")
print(f"  dK max abs diff (detached vs unfused): {(dK_detached - dK_unfused).abs().max():.6f}")
