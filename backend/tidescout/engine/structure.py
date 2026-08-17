"""Derived flow structure. Pure -- callers hand in arrays, no I/O.

Every signal in this module comes from one velocity-gradient tensor, so the
four derivatives are computed once and everything else is algebra on them.

The flow library stores 1-D arrays masked to the model domain (587,325 of
4,808,881 cells). A gradient needs neighbours, so anything here that
differentiates works on 2-D grids; `to_grid` and `from_grid` are the bridge.
"""

from dataclasses import dataclass

import numpy as np


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
