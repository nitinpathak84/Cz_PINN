# src/geometry.py
from dataclasses import dataclass
from physicsnemo.sym.geometry.primitives_2d import Rectangle


@dataclass
class CzGeom:
    # melt + crystal
    R_cr: float
    h_m: float
    R_c: float
    H_s: float

    # outer box (for boundary sampling)
    R_w: float
    z_top: float

    # heater line
    R_h: float
    z_h1: float
    z_h2: float

    # cooling line (at r = R_w)
    z_w1: float
    z_w2: float


def build_geometries(g: CzGeom):
    """Return geometry objects used for sampling."""
    melt = Rectangle((0.0, 0.0), (g.R_cr, g.h_m))
    crystal = Rectangle((0.0, g.h_m), (g.R_c, g.h_m + g.H_s))
    # Outer box used only for sampling boundary points (axis, heater, cooling)
    box = Rectangle((0.0, 0.0), (g.R_w, g.z_top))
    return melt, crystal, box


def boundary_masks(x, y, g: CzGeom, tol: float = 1e-6):
    """
    Given surface-sampled points (x,y) from the outer box, return boolean masks
    for the boundary segments we care about.

    We keep variable names x,y because GeometryDatapipe returns x,y.
    Interpret x -> r and y -> z.
    """
    r = x
    z = y

    # Axis: r ~ 0
    axis = r <= tol

    # Cooling wall: r ~ R_w and z within [z_w1, z_w2]
    cool = (abs(r - g.R_w) <= tol) & (z >= g.z_w1 - tol) & (z <= g.z_w2 + tol)

    # Heater line: r ~ R_h and z within [z_h1, z_h2]
    heat = (abs(r - g.R_h) <= tol) & (z >= g.z_h1 - tol) & (z <= g.z_h2 + tol)

    return axis, heat, cool