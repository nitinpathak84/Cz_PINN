# src/losses.py
import torch
from src.residuals import axisym_laplace_residual
from src.bcs import dirichlet_T_loss, axis_symmetry_loss


def pde_loss(model, interior_batch, r_eps: float):
    r = interior_batch["x"].reshape(-1, 1)
    z = interior_batch["y"].reshape(-1, 1)
    sdf = interior_batch["sdf"].reshape(-1, 1)

    coords = torch.cat([r, z], dim=1)
    res, _, _ = axisym_laplace_residual(model, coords, r_eps=r_eps)

    # weight by sdf like in the LDC example
    res = res * sdf
    return torch.mean(res**2)


def total_loss(model, batches, cfg):
    """
    batches dict must contain:
      mi: melt interior batch (x,y,sdf)
      ci: crystal interior batch (x,y,sdf)
      hb: heater boundary batch (x,y)
      cb: cooling boundary batch (x,y)
      ab: axis boundary batch (x,y)
    """
    mi = batches["mi"]
    ci = batches["ci"]
    hb = batches["hb"]
    cb = batches["cb"]
    ab = batches["ab"]

    r_eps = cfg.physics.r_eps

    # PDE losses
    loss_pde_m = pde_loss(model, mi, r_eps=r_eps)
    loss_pde_c = pde_loss(model, ci, r_eps=r_eps)

    # Boundary losses (Dirichlet heater/cool)
    heat_coords = torch.cat([hb["x"].reshape(-1, 1), hb["y"].reshape(-1, 1)], dim=1)
    cool_coords = torch.cat([cb["x"].reshape(-1, 1), cb["y"].reshape(-1, 1)], dim=1)
    axis_coords = torch.cat([ab["x"].reshape(-1, 1), ab["y"].reshape(-1, 1)], dim=1)

    loss_heat = dirichlet_T_loss(model, heat_coords, cfg.physics.T_hot)
    loss_cool = dirichlet_T_loss(model, cool_coords, cfg.physics.T_cold)
    loss_axis = axis_symmetry_loss(model, axis_coords, r_eps=r_eps)

    # Weighted sum
    L = (
        cfg.training.w_pde_melt * loss_pde_m
        + cfg.training.w_pde_crystal * loss_pde_c
        + cfg.training.w_bc_heat * loss_heat
        + cfg.training.w_bc_cool * loss_cool
        + cfg.training.w_bc_axis * loss_axis
    )

    details = {
        "loss_total": L,
        "loss_pde_melt": loss_pde_m.detach(),
        "loss_pde_crystal": loss_pde_c.detach(),
        "loss_bc_heat": loss_heat.detach(),
        "loss_bc_cool": loss_cool.detach(),
        "loss_bc_axis": loss_axis.detach(),
    }
    return L, details
