"""LorentzSquaredDistance: custom autograd.Function for squared Lorentz distance proxy."""

import torch
from torch.autograd import Function


class LorentzSquaredDistance(Function):
    """Compute squared Lorentz distance proxy: -<x,y>_L - 1/c.

    Inputs are spatial coordinates only; temporal x_0 is derived (H7 mitigation).
    Input shapes: x_sp, y_sp: [B, H, N, d] or [B, N, d]
    Output: [B, H, N_x, N_y] or [B, N_x, N_y]
    """

    @staticmethod
    def forward(ctx, x_sp, y_sp, c):
        # Derive temporal coordinates (H7: never store x_0)
        inv_c = 1.0 / c
        x0_sq = inv_c + (x_sp * x_sp).sum(dim=-1)  # [B, H, N] or [B, N]
        y0_sq = inv_c + (y_sp * y_sp).sum(dim=-1)
        x0 = torch.sqrt(x0_sq.clamp(min=1e-14))
        y0 = torch.sqrt(y0_sq.clamp(min=1e-14))

        # Lorentz inner product: -x0*y0 + x_sp . y_sp
        # temporal: [B,H,N_x,1] * [B,H,1,N_y] -> [B,H,N_x,N_y]
        temporal_dot = -(x0.unsqueeze(-1) * y0.unsqueeze(-2))
        spatial_dot = torch.einsum("...id,...jd->...ij", x_sp, y_sp)
        lorentz_ip = temporal_dot + spatial_dot

        # Squared distance proxy: -IP - 1/c, clamped >= 0 (H2/H3 mitigation)
        dist_sq = (-lorentz_ip - inv_c).clamp(min=0.0)

        ctx.save_for_backward(x_sp, y_sp, x0, y0, lorentz_ip)
        ctx.c = c
        return dist_sq

    @staticmethod
    def backward(ctx, grad_output):
        x_sp, y_sp, x0, y0, lorentz_ip = ctx.saved_tensors
        c = ctx.c

        # d(dist_sq)/d(lorentz_ip) = -1 where dist_sq > 0, else 0
        # dist_sq = max(-lorentz_ip - 1/c, 0)
        active = ((-lorentz_ip - 1.0 / c) > 0).float()
        grad_ip = -grad_output * active  # [B,H,Nx,Ny]

        # d(lorentz_ip)/d(x_sp):
        # lorentz_ip = -x0*y0 + x_sp . y_sp
        # d/d(x_sp[i,d]) of spatial_dot[i,j] = y_sp[j,d]
        # d/d(x_sp[i,d]) of -x0[i]*y0[j] = -y0[j] * d(x0[i])/d(x_sp[i,d])
        #   where x0 = sqrt(1/c + ||x_sp||^2), so dx0/dx_sp = x_sp / x0

        # Grad through spatial dot: sum_j grad_ip[i,j] * y_sp[j,d]
        grad_x_spatial = torch.einsum("...ij,...jd->...id", grad_ip, y_sp)
        grad_y_spatial = torch.einsum("...ij,...id->...jd", grad_ip, x_sp)

        # Grad through temporal dot: -x0[i]*y0[j]
        # d/d(x_sp[i,d]) (-x0[i]*y0[j]) = -y0[j] * x_sp[i,d]/x0[i]
        grad_temporal_x = -(grad_ip * y0.unsqueeze(-2)).sum(dim=-1)  # [B,H,Nx]
        grad_x_temporal = grad_temporal_x.unsqueeze(-1) * x_sp / x0.unsqueeze(-1).clamp(min=1e-14)

        grad_temporal_y = -(grad_ip * x0.unsqueeze(-1)).sum(dim=-2)  # [B,H,Ny]
        grad_y_temporal = grad_temporal_y.unsqueeze(-1) * y_sp / y0.unsqueeze(-1).clamp(min=1e-14)

        grad_x = grad_x_spatial + grad_x_temporal
        grad_y = grad_y_spatial + grad_y_temporal

        return grad_x, grad_y, None


def lorentz_squared_distance(x_sp, y_sp, c):
    """Functional interface for LorentzSquaredDistance."""
    return LorentzSquaredDistance.apply(x_sp, y_sp, c)
