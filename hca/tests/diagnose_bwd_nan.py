"""Diagnostic: pinpoint NaN source in fused backward at benchmark scale.

Runs the fused forward+backward at the exact dimensions used by bench_e2e
(d_head=64, N=1024, BH=64) and checks each output tensor for NaN.
"""

import torch
import torch.nn.functional as F
from hca.ops.fused_attention import _triton_fwd, _triton_bwd

torch.manual_seed(42)
device = "cuda"

# Benchmark dimensions: d_model=512, n_heads=8, batch=8 → BH=64, d_head=64, N=1024
BH, N, D = 64, 1024, 64
inv_c, beta, gamma_val = 1.0, 0.125, 0.6

# Random inputs (mimicking post-exp_map spatial coords)
Q_sp = torch.randn(BH, N, D, device=device)
K_sp = torch.randn(BH, N, D, device=device)
V = torch.randn(BH, N, D, device=device)

print(f"Config: BH={BH}, N={N}, D={D}")
print(f"Q range: [{Q_sp.min():.3f}, {Q_sp.max():.3f}]")
print(f"K range: [{K_sp.min():.3f}, {K_sp.max():.3f}]")
print(f"V range: [{V.min():.3f}, {V.max():.3f}]")

# ---- Test 1: Forward ----
for is_causal in [False, True]:
    for prec in ["tf32", "ieee"]:
        Out, LSE = _triton_fwd(Q_sp, K_sp, V, inv_c, beta, gamma_val,
                                is_causal, dot_precision=prec)
        nan_out = torch.isnan(Out).sum().item()
        nan_lse = torch.isnan(LSE).sum().item()
        inf_out = torch.isinf(Out).sum().item()
        inf_lse = torch.isinf(LSE).sum().item()
        print(f"\nFwd causal={is_causal} prec={prec}:")
        print(f"  Out NaN={nan_out}, Inf={inf_out}, range=[{Out.min():.3f}, {Out.max():.3f}]")
        print(f"  LSE NaN={nan_lse}, Inf={inf_lse}, range=[{LSE.min():.3f}, {LSE.max():.3f}]")

# ---- Test 2: Backward (the suspect) ----
for is_causal in [False, True]:
    for prec in ["tf32", "ieee"]:
        Out, LSE = _triton_fwd(Q_sp, K_sp, V, inv_c, beta, gamma_val,
                                is_causal, dot_precision=prec)
        grad_out = torch.randn_like(Out)
        Delta = (Out * grad_out).sum(-1)

        print(f"\nBwd causal={is_causal} prec={prec}:")
        print(f"  Delta NaN={torch.isnan(Delta).sum().item()}, "
              f"range=[{Delta.min():.3f}, {Delta.max():.3f}]")

        dQ, dK, dV, dg = _triton_bwd(
            Q_sp, K_sp, V, Out, LSE, grad_out, inv_c, beta, gamma_val,
            is_causal, dot_precision=prec,
        )

        for name, t in [("dQ", dQ), ("dK", dK), ("dV", dV), ("dgamma", dg)]:
            if t.dim() == 0:
                val = t.item()
                print(f"  {name}: val={val:.6f}, nan={not torch.isfinite(t)}")
            else:
                nan = torch.isnan(t).sum().item()
                inf = torch.isinf(t).sum().item()
                total = t.numel()
                pct = 100 * nan / total if total > 0 else 0
                print(f"  {name}: NaN={nan}/{total} ({pct:.1f}%), "
                      f"Inf={inf}, range=[{t[torch.isfinite(t)].min():.3f}, "
                      f"{t[torch.isfinite(t)].max():.3f}]" if nan < total
                      else f"  {name}: ALL NaN/Inf ({nan}/{total})")

# ---- Test 3: Scale sweep (find threshold where NaN appears) ----
print("\n\n--- Scale sweep (causal, tf32) ---")
for n in [32, 64, 128, 256, 512, 1024]:
    Q_s = torch.randn(BH, n, D, device=device)
    K_s = torch.randn(BH, n, D, device=device)
    V_s = torch.randn(BH, n, D, device=device)
    Out_s, LSE_s = _triton_fwd(Q_s, K_s, V_s, inv_c, beta, gamma_val,
                                True, dot_precision="tf32")
    grad_s = torch.randn_like(Out_s)
    dQ_s, dK_s, dV_s, dg_s = _triton_bwd(
        Q_s, K_s, V_s, Out_s, LSE_s, grad_s, inv_c, beta, gamma_val,
        True, dot_precision="tf32",
    )
    nan_dq = torch.isnan(dQ_s).any().item()
    nan_dk = torch.isnan(dK_s).any().item()
    nan_dv = torch.isnan(dV_s).any().item()
    nan_dg = not torch.isfinite(dg_s).item()
    any_nan = nan_dq or nan_dk or nan_dv or nan_dg
    print(f"  N={n:5d}: {'NaN!' if any_nan else 'OK'}"
          f"  dQ={nan_dq} dK={nan_dk} dV={nan_dv} dg={nan_dg}")

# ---- Test 4: Non-causal at large scale ----
print("\n--- Non-causal at N=1024 ---")
Out_nc, LSE_nc = _triton_fwd(Q_sp, K_sp, V, inv_c, beta, gamma_val,
                               False, dot_precision="tf32")
grad_nc = torch.randn_like(Out_nc)
dQ_nc, dK_nc, dV_nc, dg_nc = _triton_bwd(
    Q_sp, K_sp, V, Out_nc, LSE_nc, grad_nc, inv_c, beta, gamma_val,
    False, dot_precision="tf32",
)
nan_any = (torch.isnan(dQ_nc).any() or torch.isnan(dK_nc).any() or
           torch.isnan(dV_nc).any() or not torch.isfinite(dg_nc))
print(f"  Non-causal N=1024: {'NaN!' if nan_any else 'OK'}")

# ---- Test 5: Compare with PyTorch reference ----
print("\n--- PyTorch reference comparison (causal, N=1024) ---")
Q_r = Q_sp.clone().requires_grad_(True)
K_r = K_sp.clone().requires_grad_(True)
V_r = V.clone().requires_grad_(True)
gamma_t = torch.tensor(gamma_val, device=device, requires_grad=True)

q_x0 = torch.sqrt(torch.tensor(inv_c, device=device) + (Q_r * Q_r).sum(-1))
k_x0 = torch.sqrt(torch.tensor(inv_c, device=device) + (K_r * K_r).sum(-1))
lip = -(q_x0.unsqueeze(-1) * k_x0.unsqueeze(-2)) + torch.bmm(Q_r, K_r.transpose(-1, -2))
dsq = torch.clamp(-lip - inv_c, min=0.0)
log_kx0 = torch.log(k_x0.clamp(min=1e-7))
scores = -beta * dsq - gamma_t * log_kx0.unsqueeze(-2)
causal = torch.triu(torch.full((N, N), float('-inf'), device=device), diagonal=1)
scores = scores + causal
alpha = F.softmax(scores, dim=-1)
out_ref = torch.bmm(alpha, V_r)

# Use the same grad_out as fused backward
Out_f, LSE_f = _triton_fwd(Q_sp, K_sp, V, inv_c, beta, gamma_val,
                             True, dot_precision="tf32")
grad_out_f = torch.randn_like(Out_f)
out_ref.backward(grad_out_f)

ref_nan_q = torch.isnan(Q_r.grad).any().item()
ref_nan_k = torch.isnan(K_r.grad).any().item()
ref_nan_v = torch.isnan(V_r.grad).any().item()
print(f"  PyTorch ref: dQ_nan={ref_nan_q}, dK_nan={ref_nan_k}, dV_nan={ref_nan_v}")
print(f"  Q_r.grad range: [{Q_r.grad.min():.3f}, {Q_r.grad.max():.3f}]")
print(f"  K_r.grad range: [{K_r.grad.min():.3f}, {K_r.grad.max():.3f}]")
print(f"  V_r.grad range: [{V_r.grad.min():.3f}, {V_r.grad.max():.3f}]")
