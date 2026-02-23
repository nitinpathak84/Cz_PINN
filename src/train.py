# src/train.py
import os
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.optim import Adam, lr_scheduler

from physicsnemo.distributed import DistributedManager

from src.geometry import CzGeomParams, build_geometries
from src.sampling import make_volume_sampler, make_surface_sampler
from src.model import build_model
from src.losses import total_loss
from src.plotting import make_grid, save_T_plot


@hydra.main(version_base="1.3", config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    # Decide device (DistributedManager already picks dist.device)
    device = dist.device

    # Seeds
    torch.manual_seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)

    os.makedirs(cfg.run.out_dir, exist_ok=True)

    # Build geometries
    p = CzGeomParams(**cfg.geometry)
    geoms = build_geometries(p)

    # Build model
    model = build_model(cfg.model.num_layers, cfg.model.layer_size, device=device)

    # Optimizer / scheduler
    optimizer = Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: cfg.training.lr_decay**step
    )

    # Samplers
    melt_int = make_volume_sampler(geoms.melt, cfg.training.n_int_melt, device=device)
    crys_int = make_volume_sampler(geoms.crystal, cfg.training.n_int_crystal, device=device)

    heat_bc = make_surface_sampler(geoms.heater_band, cfg.training.n_bc_heat, device=device)
    cool_bc = make_surface_sampler(geoms.cool_band, cfg.training.n_bc_cool, device=device)
    axis_bc = make_surface_sampler(geoms.axis_band, cfg.training.n_bc_axis, device=device)

    # Inference grid for plots
    grid_coords, (r_vec, z_vec) = make_grid(
        R_w=p.R_w, z_top=p.z_top, nr=cfg.inference.nr, nz=cfg.inference.nz, device=device
    )

    # Training loop
    for step in range(cfg.run.steps):
        # Grab one batch from each sampler
        mi_raw = next(iter(melt_int))[0]
        ci_raw = next(iter(crys_int))[0]
        hb_raw = next(iter(heat_bc))[0]
        cb_raw = next(iter(cool_bc))[0]
        ab_raw = next(iter(axis_bc))[0]

        # Reshape to column vectors
        mi = {k: v.reshape(-1, 1) for k, v in mi_raw.items()}
        ci = {k: v.reshape(-1, 1) for k, v in ci_raw.items()}
        hb = {k: v.reshape(-1, 1) for k, v in hb_raw.items()}
        cb = {k: v.reshape(-1, 1) for k, v in cb_raw.items()}
        ab = {k: v.reshape(-1, 1) for k, v in ab_raw.items()}

        batches = {"mi": mi, "ci": ci, "hb": hb, "cb": cb, "ab": ab}

        optimizer.zero_grad()
        loss, details = total_loss(model, batches, cfg)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % cfg.run.plot_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"step={step} "
                f"loss={details['loss_total'].item():.4e} "
                f"pde_m={details['loss_pde_melt'].item():.2e} "
                f"pde_c={details['loss_pde_crystal'].item():.2e} "
                f"heat={details['loss_bc_heat'].item():.2e} "
                f"cool={details['loss_bc_cool'].item():.2e} "
                f"axis={details['loss_bc_axis'].item():.2e} "
                f"lr={lr:.3e}"
            )

            with torch.no_grad():
                T_pred = model(grid_coords).detach().cpu().numpy().reshape(cfg.inference.nz, cfg.inference.nr)

            out_path = os.path.join(cfg.run.out_dir, f"T_v1_{step}.png")
            save_T_plot(
                T_grid=T_pred,
                r=r_vec,
                z=z_vec,
                title="Cz PINN V1 (steady axisym conduction)",
                path=out_path,
            )


if __name__ == "__main__":
    main()
