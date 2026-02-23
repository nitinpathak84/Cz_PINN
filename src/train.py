# src/train.py
import os
import hydra
import numpy as np
import torch
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from torch.optim import Adam, lr_scheduler

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import PythonLogger
from physicsnemo.sym.geometry.geometry_dataloader import GeometryDatapipe

from geometry import CzGeom, build_geometries, boundary_masks
from model import build_model
from losses import pde_loss, dirichlet_bc_loss, axisym_laplace_residual, axis_symmetry_loss


@hydra.main(version_base="1.3", config_path="../conf", config_name="config.yaml")
def train(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    # logging
    log = PythonLogger(name="cz_pinn_v1")
    log.file_logging()

    # seeds
    torch.manual_seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)

    os.makedirs(cfg.run.out_dir, exist_ok=True)

    # build geometry
    g = CzGeom(**cfg.geometry)
    melt, crystal, box = build_geometries(g)

    # build model
    model = build_model(cfg.model.num_layers, cfg.model.layer_size, dist.device)

    optimizer = Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: cfg.training.lr_decay**step
    )

    # dataloaders: boundary points from outer box surface, interior points from melt/crystal volumes
    bc_dataloader = GeometryDatapipe(
        geom_objects=[box],
        batch_size=1,
        num_points=cfg.training.bc_points,
        sample_type="surface",
        device=dist.device,
        num_workers=1,
        requested_vars=["x", "y"],
    )

    melt_interior = GeometryDatapipe(
        geom_objects=[melt],
        batch_size=1,
        num_points=cfg.training.interior_points_melt,
        sample_type="volume",
        device=dist.device,
        num_workers=1,
        requested_vars=["x", "y", "sdf"],
    )

    crys_interior = GeometryDatapipe(
        geom_objects=[crystal],
        batch_size=1,
        num_points=cfg.training.interior_points_crystal,
        sample_type="volume",
        device=dist.device,
        num_workers=1,
        requested_vars=["x", "y", "sdf"],
    )

    # inference grid for plotting
    r_grid = np.linspace(0.0, g.R_w, cfg.inference.nx)
    z_grid = np.linspace(0.0, g.z_top, cfg.inference.nz)
    rr, zz = np.meshgrid(r_grid, z_grid, indexing="xy")
    rr_t = torch.from_numpy(rr).float().to(dist.device).reshape(-1, 1)
    zz_t = torch.from_numpy(zz).float().to(dist.device).reshape(-1, 1)
    grid_coords = torch.cat([rr_t, zz_t], dim=1)

    # training loop
    for step in range(cfg.run.steps):
        for bc_data, mi_data, ci_data in zip(bc_dataloader, melt_interior, crys_interior):
            optimizer.zero_grad()

            # unpack batches
            bc = {k: v.reshape(-1, 1) for k, v in bc_data[0].items()}
            mi = {k: v.reshape(-1, 1) for k, v in mi_data[0].items()}
            ci = {k: v.reshape(-1, 1) for k, v in ci_data[0].items()}

            # interior PDE losses
            loss_pde_m = pde_loss(model, mi, r_eps=cfg.physics.r_eps)
            loss_pde_s = pde_loss(model, ci, r_eps=cfg.physics.r_eps)

            # boundary masks (on box surface points)
            axis_mask, heat_mask, cool_mask = boundary_masks(bc["x"], bc["y"], g, tol=1e-6)

            # Build coords tensors for each BC segment
            bc_coords = torch.cat([bc["x"], bc["y"]], dim=1)
            heat_coords = bc_coords[heat_mask.squeeze(-1)]
            cool_coords = bc_coords[cool_mask.squeeze(-1)]
            axis_coords = bc_coords[axis_mask.squeeze(-1)]

            # Heater / cooling Dirichlet losses
            # If the sampler doesn't hit enough points exactly on the segment, enlarge tol or increase bc_points.
            loss_heat = torch.tensor(0.0, device=dist.device)
            loss_cool = torch.tensor(0.0, device=dist.device)
            if heat_coords.numel() > 0:
                loss_heat = dirichlet_bc_loss(model, heat_coords, target_T=cfg.physics.T_hot)
            if cool_coords.numel() > 0:
                loss_cool = dirichlet_bc_loss(model, cool_coords, target_T=cfg.physics.T_cold)

            # Axis symmetry: dT/dr = 0
            loss_axis = torch.tensor(0.0, device=dist.device)
            if axis_coords.numel() > 0:
                _, Tr_axis, _ = axisym_laplace_residual(model, axis_coords, r_eps=cfg.physics.r_eps)
                loss_axis = axis_symmetry_loss(Tr_axis)

            # total loss
            loss = (
                1.0 * loss_pde_m
                + 1.0 * loss_pde_s
                + 10.0 * loss_heat
                + 10.0 * loss_cool
                + 1.0 * loss_axis
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

        # logging + plots
        if step % cfg.run.plot_every == 0:
            with torch.no_grad():
                T_pred = model(grid_coords).detach().cpu().numpy().reshape(cfg.inference.nz, cfg.inference.nx)

            lr = optimizer.param_groups[0]["lr"]
            print(
                f"step={step} loss={loss.item():.4e} "
                f"pde_m={loss_pde_m.item():.2e} pde_s={loss_pde_s.item():.2e} "
                f"heat={loss_heat.item():.2e} cool={loss_cool.item():.2e} axis={loss_axis.item():.2e} lr={lr:.3e}"
            )

            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            im = ax.imshow(
                T_pred,
                origin="lower",
                extent=[0.0, g.R_w, 0.0, g.z_top],
                aspect="auto",
            )
            fig.colorbar(im, ax=ax)
            ax.set_xlabel("r (m)")
            ax.set_ylabel("z (m)")
            ax.set_title("Cz V1 PINN: T(r,z)")

            plt.tight_layout()
            plt.savefig(os.path.join(cfg.run.out_dir, f"T_v1_{step}.png"))
            plt.close(fig)


if __name__ == "__main__":
    train()