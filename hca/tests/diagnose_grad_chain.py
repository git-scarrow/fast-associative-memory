"""Diagnostic: trace gradient through each node in the unfused computation graph.

Uses retain_grad() to check intermediate gradient values.
"""

import math
import torch
import torch.nn.functional as F

from hca.ops.lorentz_exp_map import lorentz_exp_map
from hca.ops.lorentz_distance import lorentz_squared_distance

torch.manual_seed(42)
device = "cuda"

B, H, N, d = 2, 2, 16, 16  # tiny for clear tracing
c_val = 1.125
inv_c = 1.0 / c_val
beta = 1.0 / math.sqrt(d)
gamma_val = 0.6

K_tangent = torch.randn(B, H, N, d, device=device, requires_grad=True)
Q_tangent = torch.randn(B, H, N, d, device=device, requires_grad=True)
V = torch.randn(B, H, N, d, device=device)
gamma = torch.tensor(gamma_val, device=device, requires_grad=True)

# Forward with retain_grad on intermediates
Q_sp, Q_x0 = lorentz_exp_map(Q_tangent, c_val)
K_sp, K_x0 = lorentz_exp_map(K_tangent, c_val)
Q_sp.retain_grad()
K_sp.retain_grad()
K_x0.retain_grad()

print(f"Q_sp requires_grad: {Q_sp.requires_grad}, grad_fn: {Q_sp.grad_fn}")
print(f"K_sp requires_grad: {K_sp.requires_grad}, grad_fn: {K_sp.grad_fn}")
print(f"K_x0 requires_grad: {K_x0.requires_grad}, grad_fn: {K_x0.grad_fn}")

dist_sq = lorentz_squared_distance(Q_sp, K_sp, c_val)
dist_sq.retain_grad()

decay_bias = gamma * torch.log(K_x0.clamp(min=1e-7))
decay_bias.retain_grad()

scores = -beta * dist_sq - decay_bias.unsqueeze(-2)
scores.retain_grad()

causal = torch.triu(torch.full((N, N), float("-inf"), device=device), diagonal=1)
scores_masked = scores + causal

alpha = F.softmax(scores_masked, dim=-1)
alpha.retain_grad()

out = torch.matmul(alpha, V.float())

grad_out = torch.randn_like(out)
out.backward(grad_out)

print(f"\nalpha grad norm: {alpha.grad.norm():.6f}" if alpha.grad is not None else "alpha: no grad")
print(f"scores grad norm: {scores.grad.norm():.6f}" if scores.grad is not None else "scores: no grad")
print(f"dist_sq grad norm: {dist_sq.grad.norm():.6f}" if dist_sq.grad is not None else "dist_sq: no grad")
print(f"decay_bias grad norm: {decay_bias.grad.norm():.6f}" if decay_bias.grad is not None else "decay_bias: no grad")
print(f"K_x0 grad norm: {K_x0.grad.norm():.6f}" if K_x0.grad is not None else "K_x0: no grad")
print(f"K_sp grad norm: {K_sp.grad.norm():.6f}" if K_sp.grad is not None else "K_sp: no grad")
print(f"Q_sp grad norm: {Q_sp.grad.norm():.6f}" if Q_sp.grad is not None else "Q_sp: no grad")
print(f"K_tangent grad norm: {K_tangent.grad.norm():.6f}" if K_tangent.grad is not None else "K_tangent: no grad")
print(f"Q_tangent grad norm: {Q_tangent.grad.norm():.6f}" if Q_tangent.grad is not None else "Q_tangent: no grad")

# Detailed K_sp grad analysis
if K_sp.grad is not None:
    print(f"\nK_sp grad range: [{K_sp.grad.min():.8f}, {K_sp.grad.max():.8f}]")
    print(f"K_sp grad abs mean: {K_sp.grad.abs().mean():.8f}")

# Check dist_sq stats
print(f"\ndist_sq range: [{dist_sq.min():.2f}, {dist_sq.max():.2f}]")
print(f"scores range: [{scores.min():.2f}, {scores.max():.2f}]")
print(f"alpha max: {alpha.max():.6f}, alpha min (nonzero): {alpha[alpha > 0].min():.6f}")

# Check if active mask is all 1
lip_check = -(Q_x0.unsqueeze(-1) * K_x0.unsqueeze(-2)) + torch.einsum("...id,...jd->...ij", Q_sp.detach(), K_sp.detach())
active = ((-lip_check - inv_c) > 0).float().mean()
print(f"Active fraction: {active:.4f}")
