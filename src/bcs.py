# src/bcs.py
import torch
from src.residuals import axisym_laplace_residual


def dirichlet_T_loss(model, coords, T_target: float):
    T = model(coords)
    return torch.mean((T - T_target) ** 2)


def axis_symmetry_loss(model, coords, r_eps: float):
    # enforce dT/dr = 0 along axis band
    _, Tr, _ = axisym_laplace_residual(model, coords, r_eps=r_eps)
    return torch.mean(Tr**2)