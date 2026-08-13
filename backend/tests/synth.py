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


def border_nan(z, width=6):
    """Wrap the outer `width` cells of an array in NaN, simulating the real
    raster's mosaic/warp nodata edge -- for regression-testing detector
    behavior against an actual data boundary, not just the array's own edge."""
    z = z.copy()
    z[:width, :] = np.nan
    z[-width:, :] = np.nan
    z[:, :width] = np.nan
    z[:, -width:] = np.nan
    return z


def short_creek_mouth_dem(creek_len_m=60.0, size=200):
    """Same geometry as creek_mouth_dem, but the creek only runs `creek_len_m`
    north from open water instead of the full grid -- a short marsh feeder
    creek (Winyah's marsh has genuine creeks in the 60-100 m range), as
    opposed to the ~980 m creek in creek_mouth_dem."""
    z = np.full((size, size), 1.0, dtype="float32")
    z[100:, :] = -4.0
    rows = round(creek_len_m / CELL)
    z[100 - rows : 100, 98:101] = -2.0
    return z


def edge_touching_bar_dem(size=200):
    """Same ridge as point_bar_dem, shifted so its west side touches the
    array's own edge (col 0) instead of being interior on all sides."""
    z = np.full((size, size), -6.0, dtype="float32")
    z[90:110, 0:140] = -1.0
    return z


def donut_bar_dem(size=200):
    """-6 m deep water everywhere, with a 64x64 cell (640x640 m) -1 m
    shallow ring fully enclosing a 20x20 cell (200x200 m) -6 m deep 'island'
    at its center.

    The shallow ring is one 4-connected region whose rasterized polygon has
    an interior ring (a donut) -- the same shape `detect_bars` produces on
    real Winyah Bay bathymetry for a bar that scours out a deep pocket in
    its middle (e.g. bar-78). Used to regression-test that interior rings
    survive `_to4326`/`to_utm_geom` reprojection instead of being flattened
    into a filled blob that overclaims the hole's area.

    Net donut area is (64^2 - 20^2) * 10^2 = 369,600 m2, deliberately kept
    well under FeatureThresholds' default `bar_max_area_m2` (500,000 m2).
    This fixture's job is to prove an interior ring survives GeoJSON export,
    a property that holds at any size -- it must not be sized to *reach*
    the cap (that would let arbitrary test scaffolding dictate a production
    threshold; if the cap ever needs retuning, that should come from ground
    truth, not from this fixture)."""
    z = np.full((size, size), -6.0, dtype="float32")
    z[68:132, 68:132] = -1.0
    z[90:110, 90:110] = -6.0
    return z
