"""Pure-compute map rendering: hillshade, depth color ramp, contour lines.

No I/O here (no rasterio.open) -- callers in pipeline/ handle reading the
source raster and writing PNG/GeoTIFF/GeoJSON outputs. `rasterio.warp.transform`
and `rasterio.transform.Affine` are pure coordinate-math helpers, not I/O, so
they're fine to use directly.
"""

import numpy as np
from rasterio.transform import Affine
from rasterio.warp import transform as warp_transform
from skimage import measure


def hillshade(
    z: np.ndarray, cell_m: float, azimuth_deg: float = 315.0, altitude_deg: float = 45.0
) -> np.ndarray:
    """Standard Horn/ESRI hillshade. NaN cells (nodata) render as 0 (black).

    Aspect sign, worked from scratch (do not re-derive this casually -- it's
    easy to get backwards and have it look almost-plausible anyway, see
    test_hillshade_lights_the_azimuth_facing_slope's history):

    `np.gradient(z, cell_m)` returns `(gy, gx) = (dz/d(axis0), dz/d(axis1))`.
    axis1 (columns) increases eastward, so `gx == dz/dx_east` directly. But
    axis0 (rows) increases *southward* in this codebase's raster convention
    (confirmed by the source transform's negative e coefficient -- row+1
    moves to LOWER northing), so `gy` is actually `dz/d(south)`, i.e.
    `gy == -dz/dy_north`. Plugging `p = dz/dx_east = gx` and
    `q = dz/dy_north = -gy` into the standard Lambertian surface-normal
    aspect `atan2(-q, -p)` (downhill-direction bearing, in the same
    math-CCW-from-east convention as `az` below) gives
    `atan2(gy, -gx)` -- NOT `atan2(-gx, gy)` (that swaps which gradient
    component lands in which atan2 argument, which silently inverts which
    flank lights up rather than producing an obviously-wrong result).
    """
    az = np.radians(360.0 - azimuth_deg + 90.0)
    alt = np.radians(altitude_deg)
    gy, gx = np.gradient(np.nan_to_num(z, nan=0.0).astype("float64"), cell_m)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(gy, -gx)
    shaded = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    out = np.clip(shaded * 255.0, 0, 255).astype("uint8")
    out[np.isnan(z)] = 0
    return out


def depth_rgba(z: np.ndarray, deep_min_m: float, land_elev_m: float) -> np.ndarray:
    """Blue ramp for water (darker = deeper), tan for land, transparent for NaN."""
    h, w = z.shape
    rgba = np.zeros((h, w, 4), dtype="uint8")
    valid = ~np.isnan(z)
    water = valid & (z < land_elev_m)
    land = valid & ~water
    # deeper -> darker blue: map z in [2*deep_min, 0] to shade 0..1
    frac = np.clip(z / (2.0 * deep_min_m), 0.0, 1.0)  # 0 at surface, 1 at 2x deep_min
    rgba[..., 0][water] = (30 + 40 * (1 - frac[water])).astype("uint8")
    rgba[..., 1][water] = (90 + 110 * (1 - frac[water])).astype("uint8")
    rgba[..., 2][water] = (120 + 135 * (1 - frac[water])).astype("uint8")
    rgba[..., 0][land] = 205
    rgba[..., 1][land] = 190
    rgba[..., 2][land] = 160
    rgba[..., 3][valid] = 255
    return rgba


def _drop_nodata_artifacts(ring: np.ndarray, z: np.ndarray) -> list[np.ndarray]:
    """Split one find_contours ring into runs of vertices that are real depth
    crossings, dropping vertices that are artifacts of `contour_lines`'
    nan_to_num(1000.0) sentinel fill.

    `find_contours` runs on the sentinel-filled array so it never chokes on
    NaN, but that means it also happily traces the boundary between real
    data and the sentinel plateau itself whenever a configured depth lies
    between a real edge value and 1000.0 -- a contour that hugs the nodata
    mask's shape, not a real depth crossing. A vertex at fractional (row,
    col) is interpolated from the 2x2 cell block
    {floor,ceil}(row) x {floor,ceil}(col) in the ORIGINAL (unfilled) `z`; if
    any of those 4 source cells is NaN, this vertex only exists because of
    the sentinel and must be dropped. Splitting (rather than just dropping)
    preserves genuine sub-runs on either side of a dropped stretch as
    separate lines instead of silently stitching them together.
    """
    nrows, ncols = z.shape
    pieces: list[np.ndarray] = []
    current: list[tuple[float, float]] = []
    for row, col in ring:
        r0 = max(int(np.floor(row)), 0)
        r1 = min(int(np.ceil(row)), nrows - 1)
        c0 = max(int(np.floor(col)), 0)
        c1 = min(int(np.ceil(col)), ncols - 1)
        if np.isnan(z[r0 : r1 + 1, c0 : c1 + 1]).any():
            if current:
                pieces.append(np.array(current))
                current = []
        else:
            current.append((row, col))
    if current:
        pieces.append(np.array(current))
    return pieces


def contour_lines(
    z: np.ndarray, transform: Affine, crs_epsg: int, depths_m: list[float]
) -> list[dict]:
    """Depth contours as EPSG:4326 line coordinates, one dict per ring (or
    sub-run of a ring, after nodata-boundary artifacts are split out -- see
    `_drop_nodata_artifacts`).

    skimage `find_contours` operates in array-index space where an integer
    (row, col) sits exactly on a sample (pixel center); converting to the
    corner-based Affine convention therefore needs the + 0.5 offset below.
    Rings/pieces shorter than 5 points are dropped (too small to be
    meaningful).
    """
    out = []
    filled = np.nan_to_num(z, nan=1000.0)
    for depth in depths_m:
        for ring in measure.find_contours(filled, level=depth):
            for piece in _drop_nodata_artifacts(ring, z):
                if len(piece) < 5:
                    continue
                xs, ys = [], []
                for row, col in piece:
                    x, y = transform * (col + 0.5, row + 0.5)
                    xs.append(x)
                    ys.append(y)
                lons, lats = warp_transform(f"EPSG:{crs_epsg}", "EPSG:4326", xs, ys)
                out.append({"depth_m": float(depth), "coords": list(zip(lons, lats, strict=True))})
    return out
