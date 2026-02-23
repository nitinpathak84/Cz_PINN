# src/geometry.py
from dataclasses import dataclass
from physicsnemo.sym.geometry.primitives_2d import Rectangle


@dataclass
class CzGeomParams:
    # melt + crystal
    R_cr: float
    h_m: float
    R_c: float
    H_s: float

    # overall extents
    R_w: float
    z_top: float

    # heater segment
    R_h: float
    z_h1: float
    z_h2: float

    # cooling segment (r=R_w)
    z_w1: float
    z_w2: float

    # thin boundary thickness
    bc_thickness: float


@dataclass
class CzGeometries:
    melt: Rectangle
    crystal: Rectangle
    heater_band: Rectangle
    cool_band: Rectangle
    axis_band: Rectangle


def build_geometries(p: CzGeomParams) -> CzGeometries:
    """
    Build computational domains + separate thin rectangles for BC sampling.
    Use x->r and y->z interpretation, but keep coordinate names x,y for datapipe.
    """
    # Interior domains
    melt = Rectangle((0.0, 0.0), (p.R_cr, p.h_m))
    crystal = Rectangle((0.0, p.h_m), (p.R_c, p.h_m + p.H_s))

    eps = p.bc_thickness

    # Heater boundary: thin rectangle around r=R_h, z in [z_h1,z_h2]
    heater_band = Rectangle((p.R_h - eps, p.z_h1), (p.R_h + eps, p.z_h2))

    # Cooling boundary: thin rectangle around r=R_w, z in [z_w1,z_w2]
    cool_band = Rectangle((p.R_w - eps, p.z_w1), (p.R_w + eps, p.z_w2))

    # Axis boundary: thin rectangle around r=0 over full z
    axis_band = Rectangle((0.0, 0.0), (eps, p.z_top))

    return CzGeometries(
        melt=melt,
        crystal=crystal,
        heater_band=heater_band,
        cool_band=cool_band,
        axis_band=axis_band,
    )
