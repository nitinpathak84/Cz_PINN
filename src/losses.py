# src/losses.py
import torch


def axisym_laplace_residual(model, coords, r_eps: float):
    """
    Compute axisymmetric Laplace residual:
        R = T_rr + (1/r) T_r + T_zz
    coords: [N,2] with columns [r,z] (but coming from x,y)
    """
    r = coords[:, 0:1].clone().detach().requires_grad_(True)
    z = coords[:, 1:2].clone().detach().requires_grad_(True)

    T = model(torch.cat([r, z], dim=1))  # [N,1]

    # First derivatives
    Tr = torch.autograd.grad(
        T, r, grad_outputs=torch.ones_like(T), create_graph=True, retain_graph=True
    )[0]
    Tz = torch.autograd.grad(
        T, z, grad_outputs=torch.ones_like(T), create_graph=True, retain_graph=True
    )[0]

    # Second derivatives
    Trr = torch.autograd.grad(
        Tr, r, grad_outputs=torch.ones_like(Tr), create_graph=True, retain_graph=True
    )[0]
    Tzz = torch.autograd.grad(
        Tz, z, grad_outputs=torch.ones_like(Tz), create_graph=True, retain_graph=True
    )[0]

    r_safe = torch.clamp(r, min=r_eps)
    res = Trr + (1.0 / r_safe) * Tr + Tzz
    return res, Tr, T


def pde_loss(model, interior, r_eps: float):
    """
    interior: dict with keys x,y,sdf from GeometryDatapipe
      x->r, y->z
    """
    r = interior["x"].reshape(-1, 1)
    z = interior["y"].reshape(-1, 1)
    sdf = interior.get("sdf", None)
    coords = torch.cat([r, z], dim=1)

    res, _, _ = axisym_laplace_residual(model, coords, r_eps=r_eps)

    if sdf is not None:
        # same pattern as LDC: weight by sdf so boundary regions don't dominate
        res = res * sdf.reshape(-1, 1)

    return torch.mean(res**2)


def dirichlet_bc_loss(model, bc_coords, target_T: float):
    """Mean squared error enforcing T = target_T on boundary coords."""
    T = model(bc_coords)
    return torch.mean((T - target_T) ** 2)


def axis_symmetry_loss(Tr_on_axis):
    """Enforce dT/dr = 0 on axis points."""
    return torch.mean(Tr_on_axis**2)