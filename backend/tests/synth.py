"""Idealized DEMs. Grid: 200x200 cells at 10 m; coords are raster CRS meters."""

import numpy as np
from rasterio.transform import from_origin

CELL = 10.0
TRANSFORM = from_origin(500000, 3700000, CELL, CELL)


def open_basin(depth=-5.0, size=200):
    return np.full((size, size), depth, dtype="float32")


def creek_mouth_dem(size=200):
    """Marsh plain (+1 m) north half; open water (-4 m) south half; a 30 m wide,
    -2 m creek carved north-south through the marsh, joining open water at row 100."""
    z = np.full((size, size), 1.0, dtype="float32")
    z[100:, :] = -4.0
    z[0:100, 98:101] = -2.0  # 3 cells = 30 m wide creek
    return z


def point_bar_dem(size=200):
    """-6 m basin with an elongated -1 m shoal ridge (140x20 cells) mid-grid."""
    z = np.full((size, size), -6.0, dtype="float32")
    z[90:110, 30:170] = -1.0
    return z


def dropoff_dem(size=200):
    """-1 m shelf west of column 100; sharp step to -10 m east of it.

    Deviation from the brief (-8 m): a single-column step run through the
    *actual* `terrain.slope_deg` (np.gradient central differencing) does not
    give the naive one-cell-run slope atan(delta/cell_m). np.gradient's
    interior stencil is `(z[c+1] - z[c-1]) / (2*cell_m)`, so a step confined
    to one column boundary is seen by *two* columns (the ones straddling it)
    each with a run of `2*cell_m`, not `cell_m` -- half the naive slope.
    Verified directly: an 8 m step (-1 to -9) yields 21.8 deg computed
    slope, and the brief's literal 7 m step (-1 to -8) yields only 19.29 deg
    -- short of the 20 deg `wall_slope_deg` default threshold, so the
    "sharp step must register as a wall" gate is geometrically unreachable
    at 7 m under this (correct, already-tested) slope math. A 9 m step
    (-1 to -10) yields 24.23 deg, ~17% clear of the 20 deg line -- enough
    margin to be robust, not just barely passing.
    """
    z = np.full((size, size), -1.0, dtype="float32")
    z[:, 100:] = -10.0
    return z


def hole_dem(size=200):
    """-3 m flat with a -10 m pocket (radius 8 cells) at center."""
    z = np.full((size, size), -3.0, dtype="float32")
    yy, xx = np.mgrid[0:size, 0:size]
    z[np.hypot(xx - 100, yy - 100) < 8] = -10.0
    return z
