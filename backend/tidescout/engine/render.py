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
    """Standard Horn/ESRI hillshade. NaN cells (nodata) render as 0 (black)."""
    az = np.radians(360.0 - azimuth_deg + 90.0)
    alt = np.radians(altitude_deg)
    gy, gx = np.gradient(np.nan_to_num(z, nan=0.0).astype("float64"), cell_m)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
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


def contour_lines(
    z: np.ndarray, transform: Affine, crs_epsg: int, depths_m: list[float]
) -> list[dict]:
    """Depth contours as EPSG:4326 line coordinates, one dict per ring.

    skimage `find_contours` operates in array-index space where an integer
    (row, col) sits exactly on a sample (pixel center); converting to the
    corner-based Affine convention therefore needs the + 0.5 offset below.
    Rings shorter than 5 points are dropped (too small to be meaningful).
    """
    out = []
    filled = np.nan_to_num(z, nan=1000.0)
    for depth in depths_m:
        for ring in measure.find_contours(filled, level=depth):
            if len(ring) < 5:
                continue
            xs, ys = [], []
            for row, col in ring:
                x, y = transform * (col + 0.5, row + 0.5)
                xs.append(x)
                ys.append(y)
            lons, lats = warp_transform(f"EPSG:{crs_epsg}", "EPSG:4326", xs, ys)
            out.append({"depth_m": float(depth), "coords": list(zip(lons, lats, strict=True))})
    return out
