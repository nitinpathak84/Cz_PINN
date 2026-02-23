# src/residuals.py
import torch


def axisym_laplace_residual(model, coords, r_eps: float):
    """
    Axisymmetric Laplacian residual (steady conduction):
        R = T_rr + (1/r) * T_r + T_zz
    coords: [N,2] with columns [r,z] (coming from x,y).
    """
    r = coords[:, 0:1].detach().clone().requires_grad_(True)
    z = coords[:, 1:2].detach().clone().requires_grad_(True)

    T = model(torch.cat([r, z], dim=1))  # [N,1]

    Tr = torch.autograd.grad(
        T, r, grad_outputs=torch.ones_like(T), create_graph=True, retain_graph=True
    )[0]
    Tz = torch.autograd.grad(
        T, z, grad_outputs=torch.ones_like(T), create_graph=True, retain_graph=True
    )[0]

    Trr = torch.autograd.grad(
        Tr, r, grad_outputs=torch.ones_like(Tr), create_graph=True, retain_graph=True
    )[0]
    Tzz = torch.autograd.grad(
        Tz, z, grad_outputs=torch.ones_like(Tz), create_graph=True, retain_graph=True
    )[0]

    r_safe = torch.clamp(r, min=r_eps)
    res = Trr + (1.0 / r_safe) * Tr + Tzz
    return res, Tr, T