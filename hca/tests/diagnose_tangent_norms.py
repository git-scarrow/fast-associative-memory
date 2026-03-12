"""Diagnostic: measure tangent vector norms and softmax saturation at init.

Hypothesis: Xavier init produces ||tangent|| ≈ sqrt(d_head), causing:
1. cosh(sqrt_c * sqrt(d_head)) ≈ 2500 for d_head=64
2. Squared distances ≈ 6M → scores ≈ -781K
3. Complete softmax saturation → zero Q/K gradients (unfused)
4. Leaked noise amplified by exp_map Jacobian → NaN (fused)
"""

import math
import torch
import torch.nn.functional as F
import logging
logging.basicConfig(level=logging.WARNING)

from hca.modules.hca_attention import HCAAttention
from hca.ops.lorentz_exp_map import lorentz_exp_map
from hca.training.harness import HCATransformerLM

device = "cuda"
d_model, n_heads = 512, 8
d_head = d_model // n_heads  # 64
B, N = 2, 64

torch.manual_seed(42)

# === 1. Measure tangent norms at init ===
model = HCATransformerLM(
    vocab_size=10000, d_model=d_model, n_heads=n_heads, n_layers=1,
    max_seq_len=N, use_fused=False, amp_enabled=False,
).cuda()

# Hook into the attention module to capture tangent vectors
attn = model.layers[0].self_attn
x = torch.randn(B, N, d_model, device=device)
h = model.layers[0].norm1(x)

Q = attn.q_proj(h.float())
K = attn.k_proj(h.float())

Q = Q.view(B, N, n_heads, d_head).transpose(1, 2)
K = K.view(B, N, n_heads, d_head).transpose(1, 2)

q_norms = Q.norm(dim=-1)  # [B, H, N]
k_norms = K.norm(dim=-1)

print("=== Tangent norms at init ===")
print(f"  Q norm: mean={q_norms.mean():.2f}, std={q_norms.std():.2f}, "
      f"range=[{q_norms.min():.2f}, {q_norms.max():.2f}]")
print(f"  K norm: mean={k_norms.mean():.2f}, std={k_norms.std():.2f}, "
      f"range=[{k_norms.min():.2f}, {k_norms.max():.2f}]")

c_val = attn.curvature(N).item()
sqrt_c = math.sqrt(c_val)
print(f"  curvature c={c_val:.4f}, sqrt_c={sqrt_c:.4f}")

# Exp_map quantities
sc_mean = sqrt_c * q_norms.mean().item()
sc_max = sqrt_c * q_norms.max().item()
print(f"  sc mean={sc_mean:.2f}, max={sc_max:.2f}")
print(f"  cosh(sc_mean)={math.cosh(sc_mean):.1f}, cosh(sc_max)={math.cosh(sc_max):.1f}")
print(f"  sinc_h(sc_mean)={math.sinh(sc_mean)/sc_mean:.1f}")

# === 2. Measure actual distances and scores ===
Q_sp, Q_x0 = lorentz_exp_map(Q.float(), c_val)
K_sp, K_x0 = lorentz_exp_map(K.float(), c_val)

print(f"\n=== Hyperboloid coordinates ===")
print(f"  Q_x0 range: [{Q_x0.min():.1f}, {Q_x0.max():.1f}]")
print(f"  K_x0 range: [{K_x0.min():.1f}, {K_x0.max():.1f}]")

from hca.ops.lorentz_distance import lorentz_squared_distance
dist_sq = lorentz_squared_distance(Q_sp, K_sp, c_val)
beta = 1.0 / math.sqrt(d_head)

scores = -beta * dist_sq
print(f"\n=== Distances and scores ===")
print(f"  dist_sq range: [{dist_sq.min():.1f}, {dist_sq.max():.1f}], mean={dist_sq.mean():.1f}")
print(f"  scores range: [{scores.min():.1f}, {scores.max():.1f}], mean={scores.mean():.1f}")

# Score range within each row (determines softmax saturation)
scores_row_range = scores.max(dim=-1).values - scores.min(dim=-1).values
print(f"  within-row score range: mean={scores_row_range.mean():.1f}, max={scores_row_range.max():.1f}")
print(f"  (softmax saturates when row range > ~88)")

# === 3. Now with scaled init ===
print("\n\n=== WITH SCALED Q/K INIT (1/sqrt(d_head)) ===")
torch.manual_seed(42)
model2 = HCATransformerLM(
    vocab_size=10000, d_model=d_model, n_heads=n_heads, n_layers=1,
    max_seq_len=N, use_fused=False, amp_enabled=False,
).cuda()

# Apply the fix: scale Q/K projections
with torch.no_grad():
    for layer in model2.layers:
        s = 1.0 / math.sqrt(layer.self_attn.d_head)
        layer.self_attn.q_proj.weight.mul_(s)
        layer.self_attn.k_proj.weight.mul_(s)

attn2 = model2.layers[0].self_attn
h2 = model2.layers[0].norm1(x)
Q2 = attn2.q_proj(h2.float()).view(B, N, n_heads, d_head).transpose(1, 2)
K2 = attn2.k_proj(h2.float()).view(B, N, n_heads, d_head).transpose(1, 2)

q_norms2 = Q2.norm(dim=-1)
k_norms2 = K2.norm(dim=-1)

print(f"  Q norm: mean={q_norms2.mean():.2f}, std={q_norms2.std():.2f}, "
      f"range=[{q_norms2.min():.2f}, {q_norms2.max():.2f}]")

c_val2 = attn2.curvature(N).item()
sqrt_c2 = math.sqrt(c_val2)
sc_mean2 = sqrt_c2 * q_norms2.mean().item()
sc_max2 = sqrt_c2 * q_norms2.max().item()
print(f"  sc mean={sc_mean2:.2f}, max={sc_max2:.2f}")
print(f"  cosh(sc_mean)={math.cosh(sc_mean2):.4f}, cosh(sc_max)={math.cosh(sc_max2):.4f}")

Q2_sp, Q2_x0 = lorentz_exp_map(Q2.float(), c_val2)
K2_sp, K2_x0 = lorentz_exp_map(K2.float(), c_val2)

print(f"  Q_x0 range: [{Q2_x0.min():.2f}, {Q2_x0.max():.2f}]")

dist_sq2 = lorentz_squared_distance(Q2_sp, K2_sp, c_val2)
scores2 = -beta * dist_sq2

print(f"  dist_sq range: [{dist_sq2.min():.2f}, {dist_sq2.max():.2f}], mean={dist_sq2.mean():.2f}")
print(f"  scores range: [{scores2.min():.2f}, {scores2.max():.2f}], mean={scores2.mean():.2f}")

scores_row_range2 = scores2.max(dim=-1).values - scores2.min(dim=-1).values
print(f"  within-row score range: mean={scores_row_range2.mean():.2f}, max={scores_row_range2.max():.2f}")
print(f"  (softmax saturates when row range > ~88)")

# === 4. Compare 1-step gradient norms ===
print("\n\n=== 1-STEP GRADIENT COMPARISON (4 layers) ===")
data = torch.randint(0, 10000, (B, N + 1), device=device)
input_ids = data[:, :-1]
targets = data[:, 1:]

for label, apply_fix in [("STANDARD INIT", False), ("SCALED QK INIT", True)]:
    for use_fused in [False, True]:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        m = HCATransformerLM(
            vocab_size=10000, d_model=d_model, n_heads=n_heads, n_layers=4,
            max_seq_len=N, use_fused=use_fused, amp_enabled=False,
        ).cuda()

        if apply_fix:
            with torch.no_grad():
                for layer in m.layers:
                    s = 1.0 / math.sqrt(layer.self_attn.d_head)
                    layer.self_attn.q_proj.weight.mul_(s)
                    layer.self_attn.k_proj.weight.mul_(s)

        _, loss = m(input_ids, targets)
        loss.backward()

        total_norm = 0.0
        has_nan = False
        for name, p in m.named_parameters():
            if p.grad is not None:
                pn = p.grad.norm().item()
                total_norm += pn ** 2
                if torch.isnan(p.grad).any():
                    has_nan = True

        total_norm = total_norm ** 0.5
        fused_label = "FUSED" if use_fused else "UNFUSED"
        nan_str = " NaN!" if has_nan else ""
        print(f"  {label} {fused_label:>8s}: loss={loss.item():.4f}  grad_norm={total_norm:>15.1f}{nan_str}")
