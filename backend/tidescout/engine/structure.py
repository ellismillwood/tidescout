"""Derived flow structure. Pure -- callers hand in arrays, no I/O.

Every signal in this module comes from one velocity-gradient tensor, so the
four derivatives are computed once and everything else is algebra on them.

The flow library stores 1-D arrays masked to the model domain (587,325 of
4,808,881 cells). A gradient needs neighbours, so anything here that
differentiates works on 2-D grids; `to_grid` and `from_grid` are the bridge.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


def to_grid(
    values: np.ndarray,
    flat_index: np.ndarray,
    shape: tuple[int, int],
    fill: float = np.nan,
) -> np.ndarray:
    """Scatter a masked 1-D library array back onto the full raster.

    `fill` is NaN and not 0.0 on purpose. Zero is a real, common value here --
    it is slack water -- so filling the out-of-domain cells with it would make
    every land cell indistinguishable from a stagnant pocket, which is exactly
    what `ambush_contrast` hunts for. NaN propagates through the gradients and
    is masked off at the end instead.
    """
    grid = np.full(int(shape[0]) * int(shape[1]), fill, dtype="float64")
    grid[flat_index] = values
    return grid.reshape(shape)


def from_grid(grid: np.ndarray, flat_index: np.ndarray) -> np.ndarray:
    """Gather the in-domain cells back out of a full raster."""
    return grid.reshape(-1)[flat_index]


def xy_gradients(a: np.ndarray, cell_m: float) -> tuple[np.ndarray, np.ndarray]:
    """d/dx (true east) and d/dy (true north) of a grid-shaped field.

    `np.gradient` returns the ROW derivative first, and on a north-up raster
    rows run SOUTH while u/v are true east/north. So the row derivative is
    -d/dy. This is not cosmetic: it turns the stretching term du_dx - dv_dy
    into the divergence du_dx + dv_dy, a completely different field.
    """
    d_drow, d_dcol = np.gradient(a, cell_m)
    return d_dcol, -d_drow


@dataclass(frozen=True)
class GradientTensor:
    """The four spatial derivatives of the depth-averaged velocity field."""

    du_dx: np.ndarray
    du_dy: np.ndarray
    dv_dx: np.ndarray
    dv_dy: np.ndarray


def gradient_tensor(u: np.ndarray, v: np.ndarray, cell_m: float) -> GradientTensor:
    du_dx, du_dy = xy_gradients(u, cell_m)
    dv_dx, dv_dy = xy_gradients(v, cell_m)
    return GradientTensor(du_dx, du_dy, dv_dx, dv_dy)


def strain_rate(t: GradientTensor) -> np.ndarray:
    """Total deformation rate -- the spec's 'seams' signal.

    sqrt((du_dx - dv_dy)^2 + (du_dy + dv_dx)^2), the stretching/shearing pair.

    Deliberately NOT sqrt(du_dy^2 + dv_dx^2 + ...): that returns 1.41*omega for
    solid-body rotation, which contains no seam anywhere -- parcels in a rigidly
    turning eddy never slide past one another -- and would light up the interior
    of every eddy in the bay as holding water. It is also Galilean invariant, so
    a seam reads the same whether the whole body of water is drifting past it,
    and isotropic expansion correctly returns zero.
    """
    stretching = t.du_dx - t.dv_dy
    shearing = t.du_dy + t.dv_dx
    return np.sqrt(stretching**2 + shearing**2)


def vorticity(t: GradientTensor) -> np.ndarray:
    """Local rotation rate, dv_dx - du_dy. Positive is anticlockwise."""
    return t.dv_dx - t.du_dy


def okubo_weiss(t: GradientTensor) -> np.ndarray:
    """Strain-vs-rotation discriminant, W = S^2 - omega^2.

    Strain rate alone cannot find an eddy -- it is exactly zero for solid-body
    rotation, which is the whole point of the formula Task 3 chose. Vorticity
    alone cannot find a seam, since a shear line and a vortex both spin. The
    difference of their squares separates them, and it is the standard
    oceanographic test:

      W > 0  strain-dominated -- a seam: fast water sliding past slow
      W < 0  rotation-dominated -- an eddy core, the lee behind a point or bar

    Pure parallel shear sits exactly at W = 0, being half of each.
    """
    return strain_rate(t) ** 2 - vorticity(t) ** 2


def classify_structure(t: GradientTensor, quiet: float = 1e-5) -> np.ndarray:
    """+1 seam, -1 eddy, 0 quiet water. int8, grid-shaped.

    `quiet` is a floor on |W| in s^-2, not a tuned threshold: uniform flow has
    W = 0 up to floating-point noise, and without a dead band that noise would
    sign every cell of a featureless channel at random. 1e-5 s^-2 corresponds
    to velocity gradients around 3e-3 s^-1 -- roughly 0.06 m/s across a 20 m
    cell, which is below what the mesh resolves anyway.
    """
    w = okubo_weiss(t)
    out = np.zeros(w.shape, dtype="int8")
    out[w > quiet] = 1
    out[w < -quiet] = -1
    return out


def divergence(t: GradientTensor) -> np.ndarray:
    """du_dx + dv_dy. Positive where water spreads, negative where it closes.

    Differs from the stretching term du_dx - dv_dy by one sign, and the
    np.gradient row-orientation trap turns one into the other silently -- which
    is why `xy_gradients` negates the row derivative once, centrally, and
    nothing else in this module touches np.gradient.
    """
    return t.du_dx + t.dv_dy


def convergence(t: GradientTensor) -> np.ndarray:
    """-divergence: water closing on itself, which pins bait.

    Sign-flipped so that larger is more fish-relevant, the same convention every
    other field here follows. The scoring engine maps each structure field
    through a monotone response curve and would need a special case otherwise.
    """
    return -divergence(t)


def _disk(radius_m: float, cell_m: float) -> np.ndarray:
    """Circular footprint. A square window would reach 1.41x further on the
    diagonals, so a pocket would light up from fast water that is out of reach
    in one direction but not another -- an artefact of the grid, not the flow.
    """
    r = max(int(round(radius_m / cell_m)), 1)
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    return (xx**2 + yy**2) <= r**2


def ambush_contrast(
    speed: np.ndarray, cell_m: float, radius_m: float = 150.0
) -> np.ndarray:
    """How much faster the nearby water is than this cell, in m/s.

    The spec's "slow pockets adjacent to fast conveyors", and the mechanism
    Ellis describes at Georgetown Lighthouse: the spot works because it hides
    FROM the main channel current. Being slow is not enough and being fast is
    not enough -- what matters is a low-speed cell with a high-speed neighbour
    within a short dart, so a fish can hold out of the flow and feed in it.

    NaN marks out-of-domain. `maximum_filter` would propagate it across the
    whole footprint, so the max is taken over a copy with every NaN replaced by
    -inf, and the mask is reapplied afterwards; a cell beside land is credited
    with no neighbour rather than an infinitely fast one.
    """
    invalid = ~np.isfinite(speed)
    filled = np.where(invalid, -np.inf, speed)
    local_max = ndimage.maximum_filter(
        filled, footprint=_disk(radius_m, cell_m), mode="nearest"
    )
    out = local_max - np.where(invalid, np.nan, speed)
    out[invalid] = np.nan
    return np.maximum(out, 0.0, where=np.isfinite(out), out=out)
